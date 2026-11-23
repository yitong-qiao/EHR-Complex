"""Unified, question/gold-bound grading; no external grader or SQL is consulted.

evaluate() is the runtime entry for every finalized question. Structured field
contracts and the existing free-text format both use numeric_equal().

API: score(question, expected, answer, contract) -> 1, 0, or None (no matching
contract). Invalid matching contracts raise ContractError, not a learner failure.
contract_key is the existing build_contracts SHA-256 of JSON [question, expected].
Question and expected must be ordinary JSON data, not pre-parsed Decimal values.
build_contract(question, expected, plan) consumes required_fields, ordered and
decimal_places from an already audited preservation plan. It infers field types
from known key identities and gold values, never from final_answer. plan.fields
may supply explicit types, sources, aliases, casefold, lookup or numeric policies.
Unknown all-NULL/empty-result fields need explicit types. Other plan metadata,
including SQL, is ignored. This helper does not certify plan/source semantics.

Minimal contract (main supplies independently audited, per-question contracts):
  {"version": 1, "key": contract_key(question, expected),
   "required_fields": ["code"], "fields": {
     "code": {"source": "icd_code", "type": "code",
              "aliases": ["icd_code"]},
     "count": {"source": "frequency", "type": "integer"}},
   "ordered": false, "decimal_places": {}}
All required_fields are mandatory in EVERY answer row. Other declared fields
may be omitted, but must be correct if supplied. Unknown fields are rejected.
Field names/aliases are explicit and case-sensitive. Types: code (lexical,
preserving leading zeros; integral literals are not zero-padded), text, integer,
number, boolean. Text collapses only
whitespace; casefold is opt-in. nullable/null_tokens are opt-in. A lookup maps an
exact source value, stringified, to an audited label; it does not permit submitting
the ID instead of the required label. No names, units, counts or ranks are inferred.
number fields may explicitly set unit_suffixes, e.g. ["day", "days"]. Accept only
a complete finite number, whitespace, and one allowed unit (casefolded).
Unit handling uses the same numeric comparison and is forbidden for ID/count keys.

IDs, codes, counts and ranks are exact, including large integers. Continuous
measurements share the historical tolerance. Integer-form gold allows absolute
error strictly below 1; decimal-form gold below magnitude 10 uses 1%
relative or 0.01 absolute error; otherwise accept matching integer truncation,
matching integer rounding, or 2% relative error. Relative error uses the larger
absolute operand, as math.isclose did. Zero remains exact by default.
decimal_places overrides this default and is a map
from canonical field name to explicitly requested precision (0..12, ROUND_HALF_UP;
exact or exactly rounded gold accepted). Only number fields may set round_places.
ID/code/count/rank keys cannot use this policy, regardless of type. required
and per-field round_places are equivalent low-level spellings; conflicts fail.

ordered=false is the default; true enables order, using tie_groups if supplied.
The equivalent explicit order is unordered, ordered, or tied. tied requires tie_groups, e.g.
[[0,1],[2]], a contiguous partition of gold row indices. Only within-group order
is free. For a verified top-K boundary tie, boundary_ties={"start": 2,
"rows": [...]} supplies the COMPLETE eligible gold-format
pool for the final group. Exactly len(gold)-start distinct pool members are needed;
all earlier rows remain mandatory for tied order. For unordered output, start is
the mandatory-member count; original gold rows outside the pool are mandatory.
Do not infer a pool from tied counts alone. Ordered mode cannot use this option.

Composite gold needs an explicit decoder: composite={"source":"combined_result",
"separator":"|", "fields":["value","name"]}; field sources then use these decoded
names. Exactly that many components are required. NULL spellings need null_tokens.

Answers: plain scalar values; complete Python literal lists/tuples/dicts (no code,
comments, duplicate keys, sets or expressions); full JSON (optionally one full
json/python code fence); CSV rows (newline-separated, quote comma-containing
labels). A plain single-text-column list also accepts complete reference labels
with internal commas, separated by commas, semicolons or newlines.
Flat comma-separated values/lists use consecutive required-field chunks,
never independent column sets. Nested rows use required_fields order. An optional
positional_fields list enables a second, complete positional layout including
declared auxiliary fields; both layouts still require every required field.
Composite strings/objects are accepted only with the decoder. Complete plain-text
rows may include code/name pairs, labeled numbers, units and parenthetical counts;
these retain required fields, row binding, cardinality, ordering and numeric rules.
Arbitrary prose and Markdown tables are not semantically interpreted. Native None
means a submitted SQL NULL, not missing input. Empty results require [].

Runtime: evaluate(question, expected, answer) loads field_contracts.json and
the finalized questions_expected.jsonl. Output order and unit annotations are
resolved from question wording; the stored fields and boundary tie pools remain
unchanged. No audit files are required.
"""

import ast
from collections import Counter
import csv
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
from fractions import Fraction
import io
import json
from pathlib import Path
import re
import tokenize

import hashlib
from functools import lru_cache


def contract_key(question, expected):
    return hashlib.sha256(json.dumps([question, expected], sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()


NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", re.ASCII)


class ContractError(ValueError):
    """A matching audit contract is malformed or inconsistent with its gold."""


def _object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate JSON key")
    return result


def _constant(value):
    raise ValueError("Non-finite JSON number: " + value)


def _json(text, decimal=False):
    return json.loads(text, object_pairs_hook=_object, parse_constant=_constant,
                      parse_float=Decimal if decimal else float)


def _check(condition, message):
    if not condition:
        raise ContractError(message)


def _keys(value, allowed, label):
    _check(isinstance(value, dict) and not set(value) - set(allowed),
           "Invalid/unknown keys in " + label)


def _normalize(contract):
    contract = dict(contract)
    if "required_fields" in contract:
        fields = contract.pop("required_fields")
        _check("required" not in contract or contract["required"] == fields, "Conflicting required fields")
        contract["required"] = fields
    if "ordered" in contract:
        ordered = contract.pop("ordered")
        _check(type(ordered) is bool, "ordered must be boolean")
        inferred = "tied" if ordered and "tie_groups" in contract else "ordered" if ordered else "unordered"
        _check(contract.get("order", inferred) == inferred, "Conflicting order policies")
        contract["order"] = inferred
    if "decimal_places" in contract:
        places = contract.pop("decimal_places")
        _check(isinstance(places, dict), "decimal_places must map field names to precisions")
        _check(isinstance(contract.get("fields"), dict), "Missing fields")
        contract["fields"] = {name: dict(spec) for name, spec in contract["fields"].items()}
        for name, digits in places.items():
            _check(name in contract["fields"], "decimal_places uses unknown field")
            spec = contract["fields"][name]
            _check("round_places" not in spec or spec["round_places"] == digits, "Conflicting decimal places")
            spec["round_places"] = digits
    return contract


def _exact_numeric_key(name):
    name = name.casefold()
    if name in ("id", "itemid", "code", "icd_version") or name.endswith(("_id", "_code")):
        return True
    if name.startswith(("avg_", "average_", "mean_", "median_")):
        return False
    return (name in ("count", "counts", "frequency", "rank") or
            name.endswith(("_count", "_counts", "_frequency", "_rank")) or
            name in ("total_patients", "total_hospitalizations", "total_transfers"))


def _infer_type(name, values):
    key = name.casefold()
    if key == "code" or key.endswith("_code"):
        return "code"
    if _exact_numeric_key(key):
        return "integer"
    if key.startswith(("avg_", "average_", "mean_", "median_")):
        return "number"
    if not values:
        raise ContractError("Explicit plan.fields type needed for empty/all-NULL field: " + name)
    if all(type(v) is bool for v in values):
        return "boolean"
    if all(type(v) is int for v in values):
        return "integer"
    if all(type(v) in (int, float) for v in values):
        return "number"
    if all(isinstance(v, str) for v in values):
        return "text"
    raise ContractError("Mixed/unsupported gold types need an explicit field contract: " + name)


def build_contract(question, expected, plan):
    """Build, validate and return a contract without looking at any answer.

    expected is the release gold chosen by the caller. For verified closure
    validation, explicitly pass result['actual']; this never replaces release
    gold automatically. Required columns must exist under their declared source.
    No per-question tolerance, text casefold, NULL spellings or tie pools are inferred.
    The global continuous-number reporting-precision rule applies at comparison time.
    """
    _check(isinstance(plan, dict), "Plan must be an object")
    _check("question" not in plan or plan["question"] == question, "Plan/question mismatch")
    _check(isinstance(plan.get("required_fields"), list) and bool(plan["required_fields"]) and
           all(isinstance(k, str) and k for k in plan["required_fields"]), "Missing required_fields")
    _check(type(plan.get("ordered")) is bool, "Plan must explicitly specify ordered")
    _check(isinstance(plan.get("decimal_places", {}), dict), "decimal_places must be a field map")
    overrides = plan.get("fields", {})
    _check(isinstance(overrides, dict) and all(isinstance(k, str) and isinstance(v, dict)
                                            for k, v in overrides.items()), "Invalid plan.fields")
    try:
        rows = expected if isinstance(expected, list) else [expected]
        rows = [_unpack(row, plan.get("composite")) for row in rows]
        columns = list(dict.fromkeys(k for row in rows for k in row))
        sources = {spec.get("source", name) for name, spec in overrides.items()}
        names = list(dict.fromkeys(plan["required_fields"] + list(overrides) +
                                   [k for k in columns if k not in sources]))
        fields = {}
        for name in names:
            spec = dict(overrides.get(name, {}))
            source = spec.get("source", name)
            values = [row[source] for row in rows]
            if "lookup" in spec:
                values = [spec["lookup"][str(value)] for value in values]
            if "type" not in spec:
                spec["type"] = _infer_type(name if "lookup" in spec else source,
                                            [v for v in values if v is not None])
            if any(value is None for value in values):
                spec.setdefault("nullable", True)
            fields[name] = spec
        contract = {"version": 1, "key": contract_key(question, expected),
                    "required_fields": list(plan["required_fields"]), "ordered": plan["ordered"],
                    "decimal_places": dict(plan.get("decimal_places", {})), "fields": fields}
        for option in ("tie_groups", "boundary_ties", "composite", "positional_fields",
                       "evidence_files", "reason", "audit_id"):
            if option in plan:
                contract[option] = plan[option]
        _compile(expected, _normalize(contract))
        return contract
    except (ValueError, TypeError, KeyError, InvalidOperation) as exc:
        raise ContractError(str(exc)) from exc


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError("Expected a number")
    text = str(value).strip()
    if not NUMBER.fullmatch(text):
        raise ValueError("Expected a finite number")
    result = Decimal(text)
    if not result.is_finite():
        raise ValueError("Expected a finite number")
    return result


def _value(value, spec):
    if value is None or (isinstance(value, str) and value.strip().casefold() in
                         [s.casefold() for s in spec.get("null_tokens", [])]):
        if not spec.get("nullable", False):
            raise ValueError("NULL in a non-nullable field")
        return None
    kind = spec["type"]
    if kind in ("integer", "number"):
        if kind == "number" and isinstance(value, str) and spec.get("unit_suffixes"):
            units = "|".join(re.escape(unit.casefold()) for unit in spec["unit_suffixes"])
            match = re.fullmatch("(" + NUMBER.pattern + r")\s+(?:" + units + ")",
                                 value.strip().casefold(), re.ASCII)
            if match:
                value = match[1]
        value = _number(value)
        if kind == "integer" and value != value.to_integral_value():
            raise ValueError("Non-integer")
        return value
    if kind == "boolean":
        if not isinstance(value, bool):
            raise ValueError("Expected boolean")
        return value
    if kind == 'code' and type(value) in (int, Decimal) and _number(value) == int(value):
        value = str(int(value))
    if not isinstance(value, str):
        raise ValueError("Expected string")
    value = value.strip() if kind == "code" else " ".join(value.split())
    if kind == "text":
        value = re.sub(r'^(\d{4}-\d{2}-\d{2})[Tt](\d{2}:\d{2}:\d{2}(?:\.\d+)?)$', r'\1 \2', value)
    return value.casefold() if spec.get("casefold", False) else value


def numeric_equal(actual, expected, *, continuous=False, round_places=None):
    """Shared comparison for both structured answers and free-text numbers."""
    integer_gold = re.fullmatch(r'[+-]?\d+', str(expected).strip()) is not None
    try:
        actual, expected = _number(actual), _number(expected)
    except (ValueError, InvalidOperation):
        return False
    if actual == expected:
        return True
    if not continuous:
        return False
    if round_places is not None:
        with localcontext() as ctx:
            ctx.prec = max(28, len(expected.as_tuple().digits) +
                           abs(expected.as_tuple().exponent) + round_places + 2)
            rounded = expected.quantize(Decimal(1).scaleb(-round_places), rounding=ROUND_HALF_UP)
        return actual == rounded
    if actual == 0 or expected == 0:
        return False
    # Restore the old float policy for continuous fields in every answer format.
    # Fractions avoid binary float precision loss at tolerance boundaries.
    a, e = Fraction(actual), Fraction(expected)
    difference, scale = abs(a - e), max(abs(a), abs(e))
    if integer_gold:
        return difference < 1
    if abs(e) < 10:
        return difference <= max(scale / 100, Fraction(1, 100))
    return (int(actual) == int(expected) or round(actual) == round(expected) or
            difference <= scale / 50)


def _equal(actual, expected, spec, name=""):
    try:
        actual = _value(actual, spec)
    except (ValueError, InvalidOperation):
        return False
    if actual == expected:
        return True
    if actual is None or expected is None or spec["type"] != "number":
        return False
    exact_keys = (name, spec.get("source", name), *spec.get("aliases", []))
    if any(_exact_numeric_key(k) for k in exact_keys):
        return False
    if "round_places" in spec:
        return numeric_equal(actual, expected, continuous=True, round_places=spec["round_places"])
    return numeric_equal(actual, expected, continuous=True)


def _unpack(row, composite):
    if not isinstance(row, dict):
        raise ValueError("Expected a row object")
    if not composite or composite["source"] not in row:
        return row
    row = dict(row)
    value = row.pop(composite["source"])
    if not isinstance(value, str):
        raise ValueError("Composite must be a string")
    parts = value.split(composite["separator"])
    if len(parts) != len(composite["fields"]) or set(row) & set(composite["fields"]):
        raise ValueError("Incomplete or duplicate composite fields")
    row.update(zip(composite["fields"], (part.strip() for part in parts)))
    return row


def _gold_rows(rows, fields, composite):
    result = []
    for row in rows:
        row = _unpack(row, composite)
        projected = {}
        for name, spec in fields.items():
            source = row[spec.get("source", name)]
            if "lookup" in spec:
                source = spec["lookup"][str(source)]
            projected[name] = _value(source, spec)
        result.append(projected)
    return result


def _compile(expected, contract):
    _keys(contract, ("version", "key", "required", "fields", "order", "tie_groups",
                     "boundary_ties", "composite", "evidence_files", "reason", "audit_id",
                     "positional_fields"), "contract")
    _check(type(contract.get("version")) is int and contract["version"] == 1, "Expected version 1")
    fields, required = contract.get("fields"), contract.get("required")
    _check(isinstance(fields, dict) and bool(fields), "Fields must be nonempty")
    _check(isinstance(required, list) and bool(required) and all(isinstance(x, str) for x in required),
           "Required must be a nonempty field list")
    _check(len(set(required)) == len(required) and set(required) <= set(fields), "Invalid required fields")
    positional = contract.get("positional_fields", required)
    _check(isinstance(positional, list) and all(isinstance(k, str) for k in positional), "Invalid positional_fields")
    _check(len(set(positional)) == len(positional) and set(required) <= set(positional) <= set(fields),
           "positional_fields must include all required fields")
    _check(positional == required or len(positional) > len(required), "Ambiguous positional layouts")
    aliases = {}
    for name, spec in fields.items():
        _check(isinstance(name, str) and bool(name), "Invalid field name")
        _keys(spec, ("type", "source", "aliases", "nullable", "null_tokens", "casefold",
                     "lookup", "round_places", "unit_suffixes"), "field " + name)
        _check(spec.get("type") in ("integer", "number", "text", "code", "boolean"), "Invalid field type")
        _check(isinstance(spec.get("source", name), str), "Invalid source")
        for flag in ("nullable", "casefold"):
            _check(type(spec.get(flag, False)) is bool, "Invalid " + flag)
        _check(not spec.get("casefold") or spec["type"] == "text", "casefold is text-only")
        tokens = spec.get("null_tokens", [])
        _check(isinstance(tokens, list) and all(isinstance(t, str) and t.strip() == t and t for t in tokens),
               "Invalid null_tokens")
        _check(not tokens or spec.get("nullable"), "null_tokens requires nullable")
        _check(not {t.casefold() for t in tokens} & {"nan", "inf", "infinity", "+inf", "-inf"},
               "Non-finite numbers are not NULL aliases")
        _check("lookup" not in spec or isinstance(spec["lookup"], dict), "Invalid lookup")
        numeric_options = {"round_places"} & set(spec)
        _check(not numeric_options or spec["type"] == "number", "Numeric policies are number-only")
        exact_keys = (name, spec.get("source", name), *spec.get("aliases", []))
        _check(not numeric_options or not any(isinstance(k, str) and _exact_numeric_key(k)
                                              for k in exact_keys),
               "IDs, codes, counts and ranks must remain exact")
        if "unit_suffixes" in spec:
            units = spec["unit_suffixes"]
            _check(spec["type"] == "number" and isinstance(units, list) and
                   all(isinstance(unit, str) and re.fullmatch(r'[A-Za-z]+(?:/[A-Za-z]+)?', unit) for unit in units),
                   "unit_suffixes must be unit names on a number field")
            _check(not any(isinstance(k, str) and _exact_numeric_key(k) for k in exact_keys),
                   "Units are not allowed on IDs, codes, counts or ranks")
        if "round_places" in spec:
            _check(type(spec["round_places"]) is int and 0 <= spec["round_places"] <= 12,
                   "round_places must be 0..12")
        names = spec.get("aliases", [])
        _check(isinstance(names, list), "aliases must be a list")
        for alias in [name] + names:
            _check(isinstance(alias, str) and alias and alias not in aliases, "Duplicate/invalid field alias")
            aliases[alias] = name
    composite = contract.get("composite")
    if composite is not None:
        _keys(composite, ("source", "separator", "fields"), "composite")
        _check(isinstance(composite.get("source"), str) and composite["source"] not in aliases,
               "Invalid composite source")
        _check(isinstance(composite.get("separator"), str) and bool(composite["separator"]), "Invalid separator")
        parts = composite.get("fields")
        _check(isinstance(parts, list) and len(parts) >= 2 and all(isinstance(p, str) for p in parts),
               "Invalid composite fields")
        _check(len(set(parts)) == len(parts) and set(parts) <= set(fields), "Invalid composite field names")
    rows = expected if isinstance(expected, list) else [expected]
    gold = _gold_rows(rows, fields, composite)
    order = contract.get("order", "unordered")
    _check(order in ("unordered", "ordered", "tied"), "Invalid order")
    groups = contract.get("tie_groups")
    if order == "tied":
        _check(isinstance(groups, list) and all(isinstance(g, list) and g for g in groups), "Invalid tie_groups")
        flat = [i for group in groups for i in group]
        _check(all(type(i) is int for i in flat) and flat == list(range(len(gold))),
               "tie_groups must partition consecutive gold indices")
    else:
        _check(groups is None, "tie_groups requires tied order")
        groups = [list(range(len(gold)))] if order == "unordered" else [[i] for i in range(len(gold))]
    boundary = contract.get("boundary_ties")
    pool = None
    if boundary is not None:
        _keys(boundary, ("start", "rows"), "boundary_ties")
        start = boundary.get("start")
        _check(order != "ordered" and type(start) is int and 0 <= start < len(gold), "Invalid boundary start/order")
        _check(order != "tied" or groups[-1][0] == start, "Boundary must be the final tie group")
        _check(isinstance(boundary.get("rows"), list), "Boundary rows must be a list")
        pool = _gold_rows(boundary["rows"], fields, composite)
        identities = [tuple(row[k] for k in required) for row in pool]
        _check(len(set(identities)) == len(identities), "Boundary pool has duplicate required projections")
        if order == "unordered":
            pool_ids = set(identities)
            mandatory = [row for row in gold if tuple(row[k] for k in required) not in pool_ids]
            optional = [row for row in gold if tuple(row[k] for k in required) in pool_ids]
            _check(len(mandatory) == start, "Boundary mandatory-member count disagrees with start")
            # Only the internal matching order changes; expected and its key stay intact.
            gold = mandatory + optional
        _check(not set(identities) & {tuple(row[k] for k in required) for row in gold[:start]},
               "Boundary overlaps mandatory rows")
        full = lambda row: tuple(row[k] for k in fields)
        _check(not (Counter(map(full, gold[start:])) - Counter(map(full, pool))), "Boundary omits original gold rows")
    return fields, required, aliases, composite, gold, groups, pool


def _literal(node, text):
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal(child, text) for child in node.elts]
    if isinstance(node, ast.Dict):
        return _object([(_literal(k, text), _literal(v, text)) for k, v in zip(node.keys, node.values)])
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _number(ast.get_source_segment(text, node))
    if isinstance(node, ast.Constant):
        if type(node.value) in (int, float):
            return _number(ast.get_source_segment(text, node))
        if node.value is None or type(node.value) in (str, bool):
            return node.value
    raise ValueError("Expected literal values, not expressions")


def _decode(text, required, count, fields, composite):
    fence = re.fullmatch(r"```(?:json|python)?\s*\n?(.*?)\n?```", text, re.S)
    text = fence[1].strip() if fence else text
    if count == len(required) == 1 and fields[required[0]]["type"] in ("code", "text"):
        if not text.startswith(("[", "(", "{", "'", '"')) and not (
                fields[required[0]].get("nullable") and text in ("null", "None")):
            return text
    try:
        return _json(text, decimal=True)
    except json.JSONDecodeError:
        pass
    if text.startswith(("[", "(", "{", "'")) or text in ("None", "True", "False"):
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
        if any(t.type == tokenize.COMMENT for t in tokens):
            raise ValueError("Comments are not answer values")
        node = ast.parse(text, mode="eval").body
        # Python implicitly joins adjacent string literals; do not hide fragments.
        for left, right in zip(tokens, tokens[1:]):
            if left.type == right.type == tokenize.STRING:
                raise ValueError("Adjacent string fragments")
        return _literal(node, text)
    if composite and count == 1 and composite["separator"] in text:
        return text
    if ("," in text or "\n" in text) and (count != 1 or len(required) != 1):
        rows = list(csv.reader(io.StringIO(text), strict=True, skipinitialspace=True))
        return rows[0] if len(rows) == 1 else rows
    return text


def _answer_rows(answer, required, positional, aliases, composite, count, fields):
    if isinstance(answer, str):
        answer = _decode(answer.strip(), required, count, fields, composite)
    if isinstance(answer, str) and composite and count == 1:
        answer = {composite["source"]: answer}
    if isinstance(answer, dict):
        rows = [answer]
    elif isinstance(answer, (list, tuple)):
        if all(isinstance(row, dict) for row in answer):
            rows = list(answer)
        elif all(isinstance(row, (list, tuple)) for row in answer):
            rows = []
            for row in answer:
                layout = required if len(row) == len(required) else positional
                if len(row) != len(layout):
                    raise ValueError("Wrong positional row width")
                rows.append(dict(zip(layout, row)))
        elif count and len(answer) in (count * len(required), count * len(positional)):
            layout = required if len(answer) == count * len(required) else positional
            width = len(layout)
            rows = [dict(zip(layout, answer[i:i + width])) for i in range(0, len(answer), width)]
        else:
            raise ValueError("Use complete row objects or positional rows")
    elif count == 1 and len(required) == 1:
        rows = [{required[0]: answer}]
    else:
        raise ValueError("Expected complete rows")
    if len(rows) != count:
        raise ValueError("Wrong answer cardinality")
    result = []
    for row in rows:
        row = _unpack(row, composite)
        canonical = {}
        for key, value in row.items():
            name = aliases[key]
            if name in canonical:
                raise ValueError("Duplicate canonical field")
            canonical[name] = value
        if not set(required) <= set(canonical):
            raise ValueError("Missing required fields")
        result.append(canonical)
    return result


def _match(actual, candidates, mandatory, fields):
    """Bipartite matching: cover mandatory gold rows, then every submitted row.

    Augmentation preserves already covered mandatory candidates. This avoids a
    greedy false negative when explicit tolerances make multiple rows compatible.
    Each pool candidate can be consumed at most once.
    """
    edges = [[j for j, gold in enumerate(candidates)
              if all(_equal(value, gold[key], fields[key], key) for key, value in row.items())]
             for row in actual]
    by_actual, by_gold = {}, {}

    def cover_gold(j, seen):
        for i, options in enumerate(edges):
            if j in options and i not in seen:
                seen.add(i)
                if i not in by_actual or cover_gold(by_actual[i], seen):
                    by_actual[i] = j
                    by_gold[j] = i
                    return True
        return False

    def cover_actual(i, seen):
        for j in edges[i]:
            if j not in seen:
                seen.add(j)
                if j not in by_gold or cover_actual(by_gold[j], seen):
                    by_gold[j] = i
                    by_actual[i] = j
                    return True
        return False

    return (all(cover_gold(j, set()) for j in range(mandatory)) and
            all(i in by_actual or cover_actual(i, set()) for i in range(len(actual))))


def _plain_text_rows(answer, required, fields, gold, pool):
    """Decode complete labels without splitting commas inside their names."""
    if not isinstance(answer, str) or len(required) != 1 or not gold:
        return None
    name = required[0]
    if fields[name]['type'] != 'text' or answer.lstrip().startswith(('[', '{', '"', "'", '```')):
        return None
    text = re.sub(r'(?m)^\s*\d+[.)]\s+', '', answer.strip())
    text = re.sub(r'\s*\n\s*', '; ', text)
    text = ' '.join(text.split())
    if fields[name].get('casefold'):
        text = text.casefold()
    labels = sorted({row[name] for row in gold + (pool or [])
                     if isinstance(row[name], str) and row[name]}, key=len, reverse=True)

    @lru_cache(None)
    def parse(pos, remaining):
        if remaining == 0:
            return () if pos == len(text) else None
        for label in labels:
            match = re.match(re.escape(label).replace(',', ',?'), text[pos:])
            if not match:
                continue
            end = pos + match.end()
            separator = re.match(r'\s*[,;|]\s*', text[end:]) if end < len(text) else None
            if end < len(text) and not separator:
                continue
            tail = parse(end + (separator.end() if separator else 0), remaining - 1)
            if tail is not None:
                return (label,) + tail
        return None

    values = parse(0, len(gold))
    return [{name: value} for value in values] if values is not None else None


def _annotated_rows(answer, required, fields, gold, pool):
    """Decode complete answer rows with codes, counts, units or field labels."""
    if not isinstance(answer, str):
        return None
    if answer.lstrip().startswith('['):
        try:
            decoded = _decode(answer.strip(), required, len(gold), fields, None)
        except (ValueError, SyntaxError, tokenize.TokenError) as exc:
            # Bare labels/codes in brackets are presentation, never expressions.
            if not answer.strip().endswith(']') or '#' in answer or 'Adjacent string' in str(exc):
                return None
            answer = answer.strip()[1:-1] if answer.strip().endswith(']') else answer
            answer = re.sub(r'\]\s*,\s*\[', '; ', answer)
            if answer.startswith('[') and answer.endswith(']'):
                answer = answer[1:-1]
            if re.match(r'^\(\s*[A-Za-z]*\d', answer) and re.search(r'\)\s*,\s*\(', answer):
                answer = re.sub(r'\)\s*,\s*\(', '; ', answer[1:-1])
        else:
            if isinstance(decoded, list) and decoded and all(isinstance(row, list) for row in decoded):
                if not all(type(v) in (str, int, float, Decimal) for row in decoded for v in row):
                    return None
                answer = '; '.join(', '.join(map(str, row)) for row in decoded)
            elif isinstance(decoded, list) and all(isinstance(v, str) for v in decoded):
                answer = '; '.join(decoded)
            else:
                return None
    if answer.lstrip().startswith(('{', '"', "'", '```')):
        return None
    if re.search(r'@row\d+@', answer):
        return None
    text = re.sub(r'(?m)^\s*\d+[.)]\s+', '', answer.strip())
    synonyms = {'average': 'avg', 'mean': 'avg', 'maximum': 'max', 'peak': 'max',
                'minimum': 'min', 'patients': 'patient', 'values': 'value'}
    def words(value):
        return {synonyms.get(w, w) for w in re.findall(r'[a-z]+', value.lower())}
    def field_label(label, names):
        strengths = {k: len(words(label) & words(k)) for k in names}
        best = max(strengths.values(), default=0)
        matches = [k for k, strength in strengths.items() if strength == best]
        return matches[0] if best and len(matches) == 1 else None
    anchors = [k for k in required if fields[k]['type'] == 'text']
    if not anchors:
        anchors = [k for k in fields if fields[k]['type'] == 'text' and gold and
                   all(isinstance(row[k], str) and re.search(re.escape(row[k]), text, re.I)
                       for row in gold)]
    if not anchors:
        anchors = [k for k in required if fields[k]['type'] == 'code' or
                   fields[k]['type'] == 'integer' and (k in ('id', 'itemid') or k.endswith('_id'))]
    if not anchors:
        if ';' in text and len(gold) > 1:
            positional = _legacy_positional(fields, required)
            return _answer_rows(text.replace(';', '\n'), required, positional,
                                {k: k for k in fields}, None, len(gold), fields)
        if len(gold) != 1 or len(required) < 2 or not all(fields[k]['type'] in ('integer', 'number') for k in required):
            return None
        text = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', text)
        actual = {}
        for part in re.split(r'[,;\n]', text):
            numbers = list(NUMBER.finditer(part))
            if len(numbers) != 1:
                return None
            name = field_label(part[:numbers[0].start()] + part[numbers[0].end():], required)
            if name is None or name in actual:
                return None
            value = numbers[0][0]
            if fields[name].get('unit_suffixes'):
                suffix = part[numbers[0].end():].strip()
                if suffix:
                    value += ' ' + suffix
            actual[name] = value
        return [actual] if set(actual) == set(required) else None
    if len(anchors) != 1:
        return None
    anchor = anchors[0]
    labels = sorted({row[anchor] for row in gold + (pool or [])}, key=lambda v: len(str(v)), reverse=True)
    markers = {}
    for label in labels:
        marker = '@row%d@' % len(markers)
        boundary = r'\w.' if fields[anchor]['type'] in ('code', 'integer') else r'\w'
        pattern = '(?<![' + boundary + '])' + re.escape(str(label)).replace(',', ',?').replace(r'\ ', r'\s+') + '(?![' + boundary + '])'
        flags = re.I if fields[anchor].get('casefold') else 0
        text, count = re.subn(pattern, lambda _: marker, text, flags=flags)
        if count:
            markers[marker] = label
    if not markers:
        return None
    text = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', text)
    # A flat CSV table has a constant row width and one identity per row.
    # Protect full labels before recognizing columns, including internal commas.
    if len(gold) > 1 and not re.search(r'[;\n]', text):
        cells = next(csv.reader([text], skipinitialspace=True, strict=True))
        width, remainder = divmod(len(cells), len(gold))
        if not remainder and width > 1 and not (fields[anchor]['type'] == 'code' and width > len(fields)):
            chunks = [cells[i:i + width] for i in range(0, len(cells), width)]
            positions = [[j for j, cell in enumerate(row) if re.fullmatch(r'@row\d+@', cell.strip())] for row in chunks]
            if all(len(pos) == 1 and pos == positions[0] for pos in positions):
                text = '; '.join(', '.join(row) for row in chunks)
    # A list heading is presentation, not an additional answer row.
    if len(gold) > 1 and '@row' in text and ':' in text[:text.index('@row')]:
        heading, rest = text.split(':', 1)
        if len(heading.split()) > 3:
            text = rest.strip()
    explicit_rows = len([p for p in re.split(r'[;\n]', text) if p.strip()]) == len(gold)
    delimiters = ';\n' if len(gold) > 1 and explicit_rows else ',;\n'
    parts, start, depth = [], 0, 0
    for i, char in enumerate(text):
        depth += (char == '(') - (char == ')')
        if char in delimiters and depth == 0:
            if text[start:i].strip():
                parts.append(text[start:i].strip())
            start = i + 1
    if text[start:].strip():
        parts.append(text[start:].strip())
    null_note = r'(?P<value>NULL|None|' + NUMBER.pattern + r')\s*\((?:no|not|missing|unavailable)\b[^()]*\)\s*,?'
    if len(gold) == 1 and (len(required) > 1 or re.match(r'\d{4}-\d{2}-\d{2}', str(gold[0][anchor])) or
            any(spec.get('nullable') for spec in fields.values()) and re.match(null_note, text, re.I)):
        parts = [text]
    if len(parts) != len(gold):
        return None
    result = []
    for part in parts:
        hits = list(re.finditer(r'@row\d+@', part))
        if len(hits) != 1:
            return None
        hit = hits[0]
        actual = {anchor: markers[hit[0]]}
        before, after = part[:hit.start()].strip(), part[hit.end():].strip()
        identity = [k for k in fields if k != anchor and
                    (fields[k]['type'] == 'code' or _exact_numeric_key(k) and
                     (k in ('id', 'itemid') or k.endswith('_id')))]
        nullable = [k for k in fields if k != anchor and fields[k].get('nullable')]
        null_value = re.fullmatch(null_note, before, re.I)
        if len(nullable) == 1 and null_value:
            actual[nullable[0]] = None if null_value['value'].casefold() in ('null', 'none') else null_value['value']
            before = ''
        # A parenthetical code belongs to this label, not to its following count.
        suffix = re.match(r'^\(\s*(?:ICD(?:-(?P<version>9|10))?\s*[: ]\s*)?'
                          r'(?P<code>[A-Za-z]*\d[\w.]*)\s*'
                          r'(?:,\s*ICD-(?P<suffix_version>9|10))?'
                          r'(?:,\s*(?P<count>' + NUMBER.pattern + r'))?\s*\)', after, re.I)
        numeric = [k for k in fields if k != anchor and k not in identity and
                   fields[k]['type'] in ('integer', 'number') and fields[k].get('source', k) != 'icd_version']
        if suffix and not re.search(r'\d', before) and fields[anchor]['type'] == 'text' and (
                not numeric or re.search(r'[A-Za-z]', suffix['code']) and not NUMBER.fullmatch(suffix['code']) or
                re.match(r'^\(\s*ICD', after, re.I) or len(numeric) == 1 and (
                    suffix['count'] is not None or
                    re.match(r'\s*[-,:=\u2013\u2014]\s*' + NUMBER.pattern, after[suffix.end():]))):
            if len(identity) > 1:
                return None
            if identity:
                actual[identity[0]] = suffix['code']
            version = suffix['version'] or suffix['suffix_version']
            for name, spec in fields.items():
                if version is not None and spec.get('source', name) == 'icd_version':
                    actual[name] = version
            after = (suffix['count'] or '') + after[suffix.end():]
        code = re.fullmatch(r'(?:ICD(?:-(?P<version>\d+))?\s*[: ]\s*)?(?P<code>[A-Za-z]*\d[\w.]*)\s*(?:[,|]\s*(?P<column_version>9|10)\s*)?[-:,/|(\u2013\u2014]*', before, re.I)
        if code and (identity or fields[anchor]['type'] == 'text' and len(required) == 1):
            if len(identity) > 1:
                return None
            if identity:
                actual[identity[0]] = code['code']
            for name, spec in fields.items():
                version = code['version'] or code['column_version']
                if version is not None and spec.get('source', name) == 'icd_version':
                    actual[name] = version
            before = ''
        rest = (before + ' ' + after).strip()
        remaining = [k for k in fields if k not in actual and fields[k]['type'] in ('integer', 'number')]
        value_fields = []
        if fields[anchor]['type'] == 'code' and not identity:
            version = re.match(r'^\s*[,|]\s*(9|10)\s*[,|]', rest) if 'icd' in anchor else None
            if version:
                for name, spec in fields.items():
                    if spec.get('source', name) == 'icd_version':
                        actual[name] = version[1]
                rest = rest[version.end():]
                remaining = [k for k in remaining if k not in actual]
            if any(spec['type'] == 'text' for k, spec in fields.items() if k != anchor) and re.search(r'[A-Za-z]', rest):
                return None
            # Code descriptions have no reference field unless explicitly declared.
            if re.search(r'(?:\b(?:or|and|plus)\b|/)\s*(?:[A-Za-z]+\d[\w.]*|\d{3,})\b|'
                         r',\s*(?:[A-Za-z]+\d[\w.]*|\d{3,}(?=\s*[-\u2013\u2014:]\s*[A-Za-z]))\b', rest, re.I):
                raise ValueError('Additional answer code')
            values = re.findall(r'(?:^|[,:=\u2013\u2014]|\()\s*(' + NUMBER.pattern + r')(?=[\s),;]|$)', rest)
            rest = ''
        else:
            rest = re.sub(r'(?i)\b(?:and|itemid)\b', '', rest)
            values = []
            previous_end = 0
            def number(match):
                nonlocal previous_end
                label = rest[previous_end:match.start()]
                name = field_label(label, remaining)
                if name is None and len(remaining) > 1 and re.search(r'[A-Za-z][A-Za-z _-]*\s*[:=]', label):
                    name = '__unknown_field__'
                value_fields.append(name)
                values.append((match[1], match[2] or ''))
                previous_end = match.end()
                return ''
            units = r'(?:patients?|admissions?|hospitalizations?|occurrences?|tests?|days?|hours?|minutes?|seconds?|(?:[munp]?g|[km]?mol|mEq|U)(?:/[dm]?L)?|%)'
            declared_units = [re.escape(unit) for k in remaining for unit in fields[k].get('unit_suffixes', [])]
            if declared_units:
                units = '(?:' + '|'.join(declared_units) + '|' + units + ')'
            rest = re.sub(r'(' + NUMBER.pattern + r')\s*(' + units + r')?(?![A-Za-z])', number, rest, flags=re.I)
            rest = re.sub(r'[A-Za-z][A-Za-z _-]*\s*[:=]\s*', '', rest)
            if rest.strip(' \t\r\n(),;:/|.%=-\u2013\u2014'):
                return None
        if not value_fields:
            value_fields = [None] * len(values)
            values = [(value, '') for value in values]
        needed = [k for k in required if k not in actual]
        if any(k not in remaining for k in needed) or len(values) < len(needed):
            return None
        if values and remaining:
            # Bind numbers by explicit labels or field order, never by gold values.
            positional = needed if len(values) == len(needed) else remaining
            if not needed and len(values) == 1 and len(identity) == 1 and identity[0] in remaining:
                positional = identity
            for i, ((value, unit), name) in enumerate(zip(values, value_fields)):
                if name is None and unit:
                    unit_fields = [k for k in remaining if unit.casefold() in
                                   [u.casefold() for u in fields[k].get('unit_suffixes', [])]]
                    if len(unit_fields) == 1:
                        name = unit_fields[0]
                if name is not None and name not in remaining:
                    raise ValueError('Unknown answer field label')
                if name is None:
                    if len(values) != len(positional):
                        return None
                    name = positional[i]
                if name in actual:
                    return None
                actual[name] = value + ' ' + unit if unit and fields[name].get('unit_suffixes') else value
        if not set(required) <= set(actual) or len(values) > len(remaining) and remaining:
            return None
        if fields[anchor]['type'] == 'integer' and values and not remaining:
            return None
        result.append(actual)
    return result


def score(question, expected, answer, contract=None):
    """Only explicitly bound contracts grade; missing/stale contracts return None."""
    if contract is None:
        return None
    if not isinstance(contract, dict):
        raise ContractError("Contract must be an object")
    if contract.get("key") != contract_key(question, expected):
        return None
    try:
        contract = _normalize(contract)
        fields, required, aliases, composite, gold, groups, pool = _compile(expected, contract)
    except (ValueError, TypeError, KeyError, InvalidOperation) as exc:
        raise ContractError(str(exc)) from exc
    try:
        actual = _plain_text_rows(answer, required, fields, gold, pool)
        if actual is None:
            actual = _answer_rows(answer, required, contract.get("positional_fields", required),
                                  aliases, composite, len(gold), fields)
    except (ValueError, TypeError, KeyError, InvalidOperation, SyntaxError, tokenize.TokenError, csv.Error, RecursionError):
        actual = None
    try:
        annotated = _annotated_rows(answer, required, fields, gold, pool)
        if annotated is not None:
            actual = annotated
    except (ValueError, TypeError, KeyError, InvalidOperation, RecursionError):
        return 0
    if actual is None:
        return 0
    boundary = contract.get("boundary_ties")
    for group in groups:
        selected = [actual[i] for i in group]
        candidates = [gold[i] for i in group]
        mandatory = len(candidates)
        if pool is not None and group and group[-1] == len(gold) - 1:
            mandatory = sum(i < boundary["start"] for i in group)
            candidates = candidates[:mandatory] + pool
        if not _match(selected, candidates, mandatory, fields):
            return 0
    return 1


def _continuous_field(name, value):
    return not _exact_numeric_key(name) and (
        isinstance(value, float) or '.' in str(value) or
        name.startswith(('avg_', 'average_', 'mean_', 'median_')) or
        bool(set(name.split('_')) & {'day', 'days', 'hour', 'hours', 'minute', 'minutes', 'second', 'seconds'}))


def _requires_order(question):
    """An output-order instruction, not a top-K selection or a clinical rank."""
    q = ' '.join(question.lower().split())
    return bool(re.search(
        r'\branked\s+(?:by|from|most|least)\b|\brank\s+(?:the\s+)?(?:top|five|5|icu|lab)\b|'
        r'\b(?:ordered|sorted)\s+(?:by|from)\b|\border by prescribing frequency\b|'
        r'\bin (?:ascending|descending) order\b|\bfrom (?:highest to lowest|lowest to highest|most to least|least to most)\b', q))


def _order_groups(question, fields, gold):
    if not _requires_order(question) or len(gold) < 2:
        return [list(range(len(gold)))]
    metrics = [k for k, spec in fields.items() if spec['type'] in ('integer', 'number')
               and k not in ('id', 'itemid', 'icd_version') and not k.endswith(('_id', '_code'))]
    groups = []
    for i, row in enumerate(gold):
        if groups and len(metrics) == 1 and row[metrics[0]] == gold[i - 1][metrics[0]]:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def _text_order_matches(question, rows, answer, required):
    """Check order in legacy annotated prose after its identities/values validate."""
    if not _requires_order(question) or len(rows) < 2:
        return True
    fields = {k: {'type': 'number' if isinstance(v, (int, float, Decimal)) else 'text'}
              for k, v in rows[0].items()}
    names = [k for k in required if fields[k]['type'] == 'text' or k in ('id', 'itemid') or k.endswith('_id')]
    names.sort(key=lambda k: fields[k]['type'] != 'text')
    text = ' '.join(str(answer).split())
    for name in names:
        labels = [' '.join(str(row[name]).split()) for row in rows]
        if len(set(labels)) != len(rows):
            continue
        positions = []
        for label in labels:
            pattern = r'(?<![\w.])' + re.escape(label).replace(',', ',?') + r'(?![\w.])'
            matches = list(re.finditer(pattern, text, re.I))
            if len(matches) != 1:
                break
            positions.append(matches[0].start())
        if len(positions) == len(rows):
            groups = _order_groups(question, fields, rows)
            rank = {i: j for j, group in enumerate(groups) for i in group}
            actual = [rank[i] for i in sorted(range(len(rows)), key=lambda i: positions[i])]
            return actual == sorted(actual)
    return False


def _question_units(question, fields):
    fields = {k: dict(spec) for k, spec in fields.items()}
    continuous = [k for k, spec in fields.items() if spec['type'] == 'number' and not _exact_numeric_key(k)]
    for name in continuous:
        units = []
        for unit in ('day', 'hour', 'minute', 'second'):
            if re.search(r'(?:^|_)' + unit + r's?(?:_|$)', name):
                units.extend([unit, unit + 's'])
        if not units and len(continuous) == 1:
            named = [u for u in ('day', 'hour', 'minute', 'second') if re.search(
                r'\b(?:how many|number of|in|average|mean)\s+' + u + r's?\b', question, re.I)]
            units = ([named[0], named[0] + 's'] if len(named) == 1 else
                     re.findall(r'(?<!\w)(?:[munp]?g|[km]?mol|mEq|U)/[dm]?L(?!\w)', question, re.I))
        if units and not fields[name].get('unit_suffixes'):
            fields[name]['unit_suffixes'] = list(dict.fromkeys(units))
    return fields


def _legacy_positional(fields, required):
    if required == ['subject_id'] and 'hadm_id' in fields:
        return [k for k in fields if k != 'hadm_id']
    return list(fields)


def _legacy_required(question, rows, primary):
    """Require explicitly requested quantities as well as legacy identities."""
    required = list(dict.fromkeys(name for name, _ in primary))
    if not rows or not all(isinstance(row, dict) for row in rows):
        return required
    if not required:
        required = [k for k in rows[0] if k in ('id', 'itemid') or k.endswith('_id')] or list(rows[0])
    q = ' '.join(question.lower().split())
    # Select explicitly requested identities, retaining auxiliary fields for validation.
    codes_only = (re.search(r'\b(?:diagnosis|icd) codes?\b', q) and
                  not re.search(r'\b(?:descriptions?|titles?|names?)\b', q))
    if codes_only:
        required = [k for k in required if k == 'code' or k.endswith('_code')]
    if 'subject_id' in required and 'hadm_id' in required and re.search(r'\b(?:which|top)\s+(?:5|five)\s+patients?\b', q):
        required.remove('hadm_id')
    counts = bool(re.search(
        r'(?<!by )\bhow many\b|\bhow often\b|\b(?:total patient count|size of this cohort)\b|'
        r'\b(?:show|report|provide)\b[^?.;]{0,65}\bpatient volume\b|'
        r'\b(?:insurance|payer)(?: type)? (?:distribution|mix|breakdown)\b|\bdistribution of (?:patient counts|insurance)\b|'
        r'\b(?:what is|provide|report|indicate|reported as) (?:the |their )?(?:patient count|number of|counts? of)\b|'
        r'\b(?:and|with|include|including|corresponding|respective)\b'
        r'(?:(?!\b(?:rank|by|greatest|highest|largest|most|lab|laboratory)\b)[^?.;]){0,65}'
        r'\b(?:counts|patient count|count of|frequencies|number of)\b|'
        r'\b(?:and|with|include|including)\s+(?:(?:the|their|its|respective|corresponding)\s+)?count\b', q))
    quantities = (len(rows) == 1 and any(isinstance(value, str) for _, value in primary) or
                  bool(re.search(r'\band\b[^?.;]{0,65}\b(?:means|averages|values)\b', q)))
    for name, value in rows[0].items():
        identity = name in ('id', 'itemid', 'icd_version') or name.endswith(('_id', '_code'))
        if (counts and not identity and type(value) in (int, float) and not _continuous_field(name, value) or
                quantities and question and _continuous_field(name, value) and type(value) in (int, float) or
                name == 'icd_version' and re.search(r'\b(?:code/version|icd version)\b', q) or
                (name == 'itemid' or name.endswith('_item_id')) and re.search(r'\bitem ids?\b', q)):
            if name not in required:
                required.append(name)
    return required


def _legacy_rows_score(rows, answer, primary, required=None, question=''):
    """Use the shared row validator for explicit legacy answer formats.

    A failed structured parse is a failed answer, never a substring fallback.
    Keep the requested projection; supplied auxiliary fields still
    belong to the same row and use the same numeric comparator.
    """
    if not rows or not all(isinstance(row, dict) for row in rows):
        return None
    fields = {}
    for name, value in rows[0].items():
        if name == 'code' or name.endswith('_code'):
            kind = 'code'
        elif type(value) is bool:
            kind = 'boolean'
        else:
            try:
                _number(value)
                kind = 'number' if _continuous_field(name, value) else 'integer'
            except (ValueError, InvalidOperation):
                kind = 'text'
        fields[name] = {'type': kind, 'casefold': True,
                        'nullable': any(row[name] is None for row in rows)}
    required = required or list(dict.fromkeys(name for name, _ in primary)) or list(fields)
    fields = _question_units(question, fields)
    gold = _gold_rows(rows, fields, None)
    if isinstance(answer, str):
        answer = answer.strip()
        if re.match(r'^\([a-z]\)\s+', answer):
            answer = ', '.join(re.sub(r'^\([a-z]\)\s+', '', line.strip()) for line in answer.splitlines())
        embedded = re.search(r'```|(?<!\w)(?:\[\s*(?=[{\["\x27\d-])|\{\s*(?=["\x27]))', answer)
        if embedded:
            answer = answer[embedded.start():]
    text = answer.strip() if isinstance(answer, str) else ''
    explicit = not isinstance(answer, str) or text.startswith(('[', '{', '(', '"', "'", '```'))
    if not primary and len(rows) == len(fields) == 1:
        plain = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', text)
        if re.fullmatch(NUMBER.pattern + r'\s*(?:[,;]|or|and)\s*' + NUMBER.pattern, plain, re.I):
            return 0
    # Plain numeric rows are also positional, rather than independent token sets.
    numeric_rows = (all(fields[k]['type'] in ('integer', 'number') for k in required) and len(rows) * len(fields) > 1 and
                    re.fullmatch(r'[\d\s.,;+eE-]+', text) is not None)
    try:
        if explicit or numeric_rows:
            if text.startswith('```') and not re.fullmatch(r'```(?:json|python)?\s*\n?.*?\n?```', text, re.S):
                return 0
            try:
                source = answer.replace(';', '\n') if numeric_rows and isinstance(answer, str) else answer
                actual = _answer_rows(source, required, _legacy_positional(fields, required),
                                      {name: name for name in fields}, None, len(gold), fields)
            except (ValueError, TypeError, KeyError, SyntaxError, tokenize.TokenError):
                actual = _annotated_rows(answer, required, fields, gold, None)
                if actual is None:
                    return 0
            annotated = _annotated_rows(answer, required, fields, gold, None)
            if annotated is not None and not all(fields[k]['type'] in ('integer', 'number') for k in required):
                actual = annotated
        else:
            actual = _plain_text_rows(answer, required, fields, gold, None)
            if actual is None:
                actual = _annotated_rows(answer, required, fields, gold, None)
            if actual is None:
                return None
        return int(all(_match([actual[i] for i in group], [gold[i] for i in group], len(group), fields)
                       for group in _order_groups(question, fields, gold)))
    except (ValueError, TypeError, KeyError, InvalidOperation, SyntaxError, tokenize.TokenError, csv.Error, RecursionError):
        return 0


def _supplied_numbers_match(rows, answer, primary, required=()):
    """Validate explicit identity/value pairs before the primary-only fallback."""
    number = r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?'
    text = re.sub(r'(?m)^\s*\d+[.)]\s+', '', str(answer).strip().lower())
    identities = sorted({' '.join(str(value).strip().lower().split()) for _, value in primary},
                        key=len, reverse=True)
    if not identities:
        return True
    text = '\n\n'.join(part for part in re.split(r'\n\s*\n', text)
                       if any(label in ' '.join(part.split()) for label in identities) or
                       len(part.split()) <= 8 or not part.rstrip().endswith('.'))
    text = ' '.join(text.split())
    # Match complete, longest labels so numbers in names and overlapping names
    # cannot become values belonging to the preceding/shorter label.
    patterns = []
    for i, value in enumerate(identities):
        code_or_id = any((name == 'code' or name.endswith('_code') or isinstance(v, (int, float)))
                         and ' '.join(str(v).strip().lower().split()) == value for name, v in primary)
        boundary = r'\w.' if code_or_id else r'\w'
        patterns.append(r'(?<![%s])(?P<v%d>%s)(?![%s])' %
                        (boundary, i, re.escape(value).replace(',', ',?'), boundary))
    spans = list(re.finditer('|'.join(patterns), text))
    for row in rows:
        if not isinstance(row, dict):
            continue
        numeric = []
        labels = set()
        for name, value in row.items():
            if (name, value) in primary:
                labels.add(' '.join(str(value).strip().lower().split()))
                continue
            try:
                _number(value)
                numeric.append((name, value))
            except (ValueError, InvalidOperation):
                pass
        if not numeric:
            continue
        candidates = []
        positions = []
        for i, match in enumerate(spans):
            identity = identities[int(match.lastgroup[1:])]
            if identity not in labels:
                continue
            # Repeated descriptions can identify different code rows. Use an
            # immediately preceding identity (e.g. ICD code) to disambiguate.
            if i and re.fullmatch(r'[\s:(\u2013\u2014-]*', text[spans[i - 1].end():match.start()]):
                previous = identities[int(spans[i - 1].lastgroup[1:])]
                if previous not in labels:
                    continue
            positions.append(match.start())
            end = spans[i + 1].start() if i + 1 < len(spans) else len(text)
            suffix = text[match.end():end]
            # A code preceding the next label is not this row's count.
            if i + 1 < len(spans):
                suffix = re.sub(r'[,;]\s*\d+\s*[-\u2013\u2014]?\s*$', '', suffix)
            count_names = '|'.join(re.escape(name) for name, _ in numeric)
            suffix = re.sub(r'^([\s:=(,)\u2013\u2014-]*)(?:n|count|' + count_names + r')\s*[:=]\s*', r'\1', suffix)
            pair = re.match(r'(?:\s*\([^()]*[a-z][^()]*\)\s*[:=\u2013\u2014-]\s*|'
                            r'\s*\(\s*n\s*=\s*|\s*\([\w.]+;\s*|[\s:=(,)\u2013\u2014]+(?:-\s+)?)('
                            + number + r')(?![\d.])', suffix)
            if pair is None and len(numeric) == 1 and all(
                    name == 'code' or name.endswith('_code') for name, _ in primary):
                pair = re.search(r'[:=\u2013\u2014(]\s*(' + number + r')\s*'
                                 r'(?:unique\s+)?(?:patients?|admissions?|occurrences?)?\s*\)?\s*[,;]?\s*$', suffix)
            if pair:
                candidates.append(pair[1].replace(',', ''))
            if len(numeric) == 1:
                labeled = []
                for found in re.finditer(r'(?:\(|,)\s*([a-z][a-z_ ]*?)\s*(?:[:=]\s*|\s+)('
                                         + number + r')(?![\d.])', suffix):
                    label = found[1].strip().replace(' ', '_')
                    if label in ('id', 'itemid', 'code', 'version') or label.startswith('icd') or label.endswith(('_id', '_code')):
                        continue
                    labeled.append((label, found[2].replace(',', '')))
                aliases = {'peak': 'max', 'maximum': 'max', 'minimum': 'min', 'val': 'value'}
                words = lambda label: {aliases.get(w, w) for w in label.split('_')}
                bound = [value for label, value in labeled if words(label) & words(numeric[0][0])]
                candidates.extend(bound or ([labeled[0][1]] if len(labeled) == 1 else []))
        # A single-row positional answer can put its numeric field first. Only
        # its first value is an answer; later dates/IDs can be explanatory text.
        if len(rows) == 1 and positions and next(iter(row)) == numeric[0][0]:
            prefix = text[:min(positions)]
            candidate = re.search(r'(?<![\w.])' + number + r'(?![\w.])', prefix)
            if candidate:
                candidates.append(candidate[0].replace(',', ''))
        for candidate in candidates:
            if not any(numeric_equal(candidate, value, continuous=_continuous_field(name, value))
                       for name, value in numeric):
                return False
        if not candidates and any(name in required for name, _ in numeric):
            return False
    return True


def _identity_list_matches(rows, answer, primary):
    """Check whole identities and multiplicity in plain single-column lists."""
    names = {name for name, _ in primary}
    if len(names) > 1:
        return all(_identity_list_matches(rows, answer, [(key, value) for key, value in primary if key == name])
                   for name in names)
    if len(names) != 1 or not all(isinstance(row, dict) for row in rows):
        return True
    name = next(iter(names))
    if any(name not in row for row in rows):
        return True
    def normalize(value):
        if isinstance(rows[0][name], (int, float)):
            try:
                return _number(value)
            except (ValueError, InvalidOperation):
                pass
        value = re.sub(r'(\d{4}-\d{2}-\d{2})[Tt ](\d{2}:\d{2})', r'\1 \2', str(value).lower())
        return re.sub(r'\s+', ' ', re.sub(r"[',.\-/]", ' ', value)).strip()
    wanted = [normalize(row[name]) for row in rows]
    text = re.sub(r'(?m)^\s*\d+[.)]\s+', '', str(answer).strip())
    text = text.replace('**', '')
    if len(rows) == 1 and len(rows[0]) == 1 and normalize(text) == wanted[0]:
        return True
    text = re.sub(r'(?i)^yes\s*[,;.]\s*', '', text)
    if len(rows) > 1 and '\n' in text:
        heading, body = text.split('\n', 1)
        if heading.rstrip().endswith(':') and len(heading.split()) > 3:
            text = body.strip()
    if name == 'code' or name.endswith(('_code', '_id')) or name == 'id':
        parts = [part.strip() for part in text.split(',')]
        if all(re.fullmatch(r'[A-Za-z0-9.]+', part) for part in parts):
            return _legacy_rows_score(rows, parts, primary) == 1
    text = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', text)
    if name in ('code', 'id') or name.endswith(('_code', '_id')):
        code = r'(?:[A-Za-z]+\d[\w.]*|\d{3,}(?:\.\d+)?)'
        prefix = r'(?:' + re.escape(name) + r'\s*[:=]?\s*)?'
        entries = re.split(r'[;\n]|,(?=\s*' + prefix + code + r'(?:\s|$|[(:\u2013\u2014-]))', text)
        codes = [re.match(r'^\s*' + prefix + '(' + code + r')(?:\s|$|[(:\u2013\u2014-])', entry) for entry in entries]
        if codes and all(codes):
            return Counter(normalize(match[1]) for match in codes) == Counter(wanted)
    if (name == 'code' or name.endswith('_code')) and re.search(r'[()]', text):
        return True
    labels = sorted({str(row[name]).strip() for row in rows}, key=len, reverse=True)
    markers = {}
    pattern = '|'.join(re.escape(' '.join(label.split())).replace(',', ',?') for label in labels)
    text = re.sub(r'[^\S\n]+', ' ', text)

    def protect(match):
        marker = '@identity%d@' % len(markers)
        markers[marker] = normalize(match[0])
        return marker

    text = re.sub(r'(?<![\w.])(?:' + pattern + r')(?![\w.])', protect, text, flags=re.I)
    if re.search(r'@identity\d+@\s+(?:or|and)\s+\w', text, re.I):
        return False
    # Descriptive prose remains handled by the legacy matcher. This guard is
    # for complete value lists, not numbers in dates or explanatory context.
    paragraphs = re.split(r'\n\s*\n', text)
    text = '\n\n'.join(part for part in paragraphs if '@identity' in part or
                       len(part.split()) <= 8 or not part.rstrip().endswith('.'))
    pieces = [part.strip() for part in re.split(r'[,;\n](?![^()]*\))', text) if part.strip()]
    actual = []
    for part in pieces:
        if '@identity' not in part:
            if len(rows) == 1 and re.fullmatch(r'(?:[A-Za-z]*\d[\w.]*|\d+)\s*', part):
                # A scalar label may be accompanied by its unrequested ID/code.
                continue
            aliases = {'highest': 'max', 'maximum': 'max', 'lowest': 'min', 'minimum': 'min',
                       'peak': 'max', 'average': 'avg', 'mean': 'avg'}
            words = [aliases.get(word, word) for word in re.findall(r'[a-z]+', part.lower()) if word != 'the']
            numeric_words = {word for key, value in rows[0].items() if key != name
                             and isinstance(value, (int, float)) for word in key.split('_')}
            if len(rows) == 1 and re.search(NUMBER, part) and words and words[0] in numeric_words:
                continue
            if len(rows) == 1 and (re.fullmatch(r'\d+(?:\.\d+)?\s*(?:[munp]?g|mL|units?)', part, re.I)
                                  or re.fullmatch(r'(?:PO|NG|IV|IM|SC|SQ)(?:/(?:PO|NG|IV|IM|SC|SQ))*', part, re.I)):
                continue
            if '\n' in str(answer) and len(part.split()) > 8 and part.endswith('.'):
                # A paragraph outside the list can report cohort context.
                continue
        if len(rows[0]) > 1:
            count_labels = '|'.join(re.escape(key) for key in rows[0]
                                    if key != name and _exact_numeric_key(key))
            count_labels = '(?:n|count' + ('|' + count_labels if count_labels else '') + ')'
            count = (r'[-+]?\d+(?:\.\d+)?\s*'
                     r'(?:patients?|admissions?|hospitalizations?|occurrences?|tests?|cases?)?\s*')
            part = re.sub(r'\s*(?:[:=\u2013\u2014]|-\s+|\(\s*)\s*(?:' + count_labels + r'\s*[:=]\s*)?' + count + r'\)?$', '', part, flags=re.I)
            if '(' not in part and ')' not in part:
                part = re.sub(r'\s+(?:-\s+)?' + count + '$', '', part, flags=re.I)
        if len(rows[0]) > 1 and re.fullmatch(NUMBER, part):
            continue
        if isinstance(rows[0][name], str) and not (name == 'code' or name.endswith('_code')):
            part = re.sub(r'^(?:[A-Z]\d[\w.]*|\d{3,})\s*[-:\u2013\u2014]?\s+(?=\S)', '', part, flags=re.I)
            part = re.sub(r'\s*\(\s*ICD(?:-\d+)?\s*[: ]\s*[A-Za-z]*\d[\w.]*\s*\)', '', part, flags=re.I)
            if any(key == 'code' or key.endswith('_code') for key in rows[0]):
                part = re.sub(r'\s*\(\s*[A-Za-z]*\d[\w.]*\s*\)', '', part)
        if re.fullmatch(r'\(@identity\d+@\)', part):
            part = part[1:-1]
        if part in markers:
            actual.append(markers[part])
        elif re.search(r'[:=]', part) or (
                re.search(r'\d', re.sub(r'@identity\d+@', '', part)) and (
                    '@identity' in part or len(pieces) == 1 or re.match(r'^[+-]?[\d.]', part))):
            hits = re.findall(r'@identity\d+@', part)
            if len(hits) == 1 and hits[0] in markers:
                actual.append(markers[hits[0]])
            elif len(rows) == 1 and not hits and re.match(r'^[+-]?[\d.]', part):
                continue
            else:
                return False
        else:
            # Extra words after a complete scalar label can be an annotation.
            if len(rows) == 1 and len(re.findall(r'@identity\d+@', part)) == 1 and len(part.split()) > 5:
                actual.append(markers[re.search(r'@identity\d+@', part)[0]])
                continue
            if len(rows) == len(pieces) == 1 and re.match(r'^@identity\d+@\s+\w', part):
                return True
            actual.append(normalize(part))
    return Counter(actual) == Counter(wanted)


def _free_text_score(expected, answer, question=''):
    """Preserve the existing uncontracted text format, using shared numeric rules.

    Audited field/cardinality contracts remain data, not separate evaluator code.
    Without a contract, keep identities and explicitly requested quantities.
    Explicit structured formats always go through the shared row validator.
    """
    text = re.sub(r'(?m)^\s*\d+[.)]\s+', '', str(answer).strip().lower())
    text = re.sub(r"\s+", " ", text)
    rows = expected if isinstance(expected, list) else [expected]
    values, primary = [], []
    has_dict_rows = False
    for row in rows:
        has_dict_rows |= isinstance(row, dict)
        for name, value in (row.items() if isinstance(row, dict) else [("", row)]):
            item = (name, value)
            values.append(item)
            if isinstance(value, str) and (name == 'code' or name.endswith('_code')):
                primary.append(item)
                continue
            try:
                number = _number(value)
            except (ValueError, InvalidOperation):
                if (isinstance(value, str) and value.strip()) or not isinstance(row, dict):
                    primary.append(item)
            else:
                if isinstance(value, (int, float)) and number >= 1000000 and (name in ('id', 'itemid') or name.endswith('_id')):
                    primary.append(item)
    if not values:
        return int(text in {"", "none", "null", "[]", "{}"})
    required = _legacy_required(question, rows, primary)
    parsed = _legacy_rows_score(rows, answer, primary, required, question)
    if parsed is not None:
        return parsed

    def matches(name, value):
        target = str(value).strip().lower()
        if re.match(r"\d{4}-\d{2}-\d{2}", target):
            pattern = r"(\d{4}-\d{2}-\d{2})[Tt ](\d{2}:\d{2})"
            return re.sub(pattern, r"\1 \2", target) in re.sub(pattern, r"\1 \2", text)
        if target in {"null", "none", "", "nan", "na"}:
            return not text or any(x in text.split() for x in ["null", "none", "nan", "na"])
        # Preserve leading zeroes and version/code identity instead of casting codes.
        if isinstance(value, str) and (name == 'code' or name.endswith('_code')):
            return re.search(r'(?<![\w.])' + re.escape(target) + r'(?![\w.])', text) is not None
        try:
            number = _number(value)
        except (ValueError, InvalidOperation):
            if len(target) <= 2:
                return re.search(r'\b' + re.escape(target) + r'\b', text) is not None
            if target in text:
                return True
            normalize = lambda s: re.sub(r'\s+', ' ', re.sub(r"[',.\-/]", ' ', s)).strip()
            target_normalized = normalize(target)
            return len(target_normalized) > 2 and target_normalized in normalize(text)
        continuous = _continuous_field(name, value)
        # Legacy continuous-value text accepts units attached to the number (81mg).
        end = r'(?![\d.])' if continuous else r'(?![\w.])'
        numbers = re.findall(r'(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?' + end, text)
        return any(numeric_equal(candidate.replace(',', ''), number, continuous=continuous) for candidate in numbers)

    if not _supplied_numbers_match(rows, answer, primary, required):
        return 0
    if not _identity_list_matches(rows, answer, primary):
        return 0
    if not _text_order_matches(question, rows, answer, required):
        return 0
    return int(all(matches(*item) for item in values) or
               (has_dict_rows and bool(primary) and all(matches(*item) for item in primary) and
                all(matches(name, value) for name, value in values if name in required)))


@lru_cache(maxsize=1)
def _runtime_registry():
    root = Path(__file__).resolve().parent
    registry = _json((root / 'field_contracts.json').read_text())
    rows = [_json(s) for s in (root.parent / 'questions_expected.jsonl').read_text().splitlines() if s.strip()]
    allowed = frozenset(contract_key(r['question'], r['expected_result']) for r in rows)
    _check(len(allowed) == len(rows) and
           (len(rows), len(registry)) in ((3915, 522), (500, 74)),
           'Unexpected final release scope')
    _check(set(registry) <= allowed, 'Contracts do not match the final dataset')
    return registry, allowed


def evaluate(question, expected, answer):
    """The single runtime entry point for finalized questions and the released subset."""
    registry, allowed = _runtime_registry()
    key = contract_key(question, expected)
    _check(key in allowed, 'Question/answer pair is not in the finalized dataset')
    if key in registry:
        contract = _normalize(registry[key])
        contract['fields'] = _question_units(question, contract['fields'])
        if not _requires_order(question):
            contract['order'] = 'unordered'
            contract.pop('tie_groups', None)
        elif contract.get('order', 'unordered') == 'unordered':
            fields, _, _, _, gold, _, _ = _compile(expected, contract)
            groups = _order_groups(question, fields, gold)
            boundary = contract.get('boundary_ties')
            if boundary:
                groups = [g for g in groups if g[-1] < boundary['start']] + [list(range(boundary['start'], len(gold)))]
            contract.update(order='tied', tie_groups=groups)
        result = score(question, expected, answer, contract)
    else:
        result = _free_text_score(expected, answer, question)
    _check(type(result) is int and result in (0, 1), 'Invalid evaluator result')
    return result

