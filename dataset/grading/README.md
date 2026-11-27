# Scoring

The scorer implementation lives in this code repository. ModelScope supplies
`questions_expected.jsonl` and `grading/field_contracts.json`, which the installer
places under `dataset/`; the package does not contain another scorer.

After installing the question package, call
`evaluate(question, expected, answer)` from `ehr_complex_grader.py`. It returns
`1` for a correct answer and `0` otherwise, without an LLM or SQL execution.

Field contracts specify required fields, types, order, and accepted ties.
The fixed scorer also interprets explicit ordering requirements and units from
the question wording. Representative 500 and Full 3,915 use the same fixed
scoring logic, with the applicable field contracts included in each package.
Answers must include the required values with correct field associations;
missing values, extra entities, and contradictions fail.

IDs, codes, counts, and ranks are exact. For continuous values, integer-form
references allow absolute error below one. Nonzero decimal references below ten
in magnitude allow 1% relative or 0.01 absolute error; larger values allow equal
integer truncation, equal integer rounding, or 2% relative error. Relative error
uses the larger magnitude. Zero is exact; requested decimal precision takes priority.
