"""Install the authorized EHRComplex question package."""
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]


def install(source, check_only=False):
    payload = {
        Path('dataset') / name: (source / name).read_bytes()
        for name in ('questions_expected.jsonl', 'runtime_index_map.jsonl',
                     'grading/field_contracts.json')
    }
    grader_path = ROOT / 'dataset/grading/ehr_complex_grader.py'
    spec = importlib.util.spec_from_file_location('ehr_complex_grader', grader_path)
    grader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grader)
    questions = [json.loads(line) for line in payload[Path('dataset/questions_expected.jsonl')].splitlines()]
    mapping = [json.loads(line) for line in payload[Path('dataset/runtime_index_map.jsonl')].splitlines()]
    if len(questions) != 500 or len(mapping) != 500:
        raise ValueError('Expected exactly 500 questions and mapping entries.')
    if {row['source_index'] for row in mapping} != set(range(500)):
        raise ValueError('Question mapping is not one-to-one.')
    runtime = defaultdict(list)
    for task_id, entry in enumerate(mapping):
        relative = PurePosixPath(entry['source_file'])
        if entry['runtime_task_id'] != task_id or relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid runtime mapping.')
        row = questions[entry['source_index']]
        if grader.contract_key(row['question'], row['expected_result']) != entry['question_answer_sha256']:
            raise ValueError(f'Question mapping mismatch: {task_id}')
        # A single group, sorted by the stable runtime ID, preserves loader order.
        runtime[Path('data/ehr_complex') / str(relative)].append({
            'question': row['question'], 'expected_result': row['expected_result'],
            'paraphrase_index': task_id,
        })
    loaded_ids = [row['paraphrase_index'] for path in sorted(runtime) for row in runtime[path]]
    if loaded_ids != list(range(500)):
        raise ValueError('Runtime file order differs from the task mapping.')
    for path, rows in runtime.items():
        payload[path] = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode()
    existing_runtime = ROOT / 'data/ehr_complex'
    extras = {p.relative_to(ROOT) for p in existing_runtime.rglob('*.jsonl')} - set(runtime)
    if extras:
        raise ValueError('Unexpected runtime question files; use a clean checkout.')
    for relative, content in payload.items():
        destination = ROOT / relative
        if destination.exists() and destination.read_bytes() != content:
            raise ValueError(f'Refusing to overwrite different content: {relative}')
    if not check_only:
        for relative, content in payload.items():
            destination = ROOT / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                destination.write_bytes(content)
    print(f'{"Verified" if check_only else "Installed"}: 500 questions.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Authorized question package directory')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    install(args.source, args.check_only)
