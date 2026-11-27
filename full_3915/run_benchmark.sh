#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" - <<'PY'
from pathlib import Path

questions = Path('dataset/questions_expected.jsonl')
if not questions.is_file() or len(questions.read_text().splitlines()) != 3915:
    raise SystemExit('Install the Full 3,915 package in this checkout before evaluation.')
PY
exec bash "$PROJECT_DIR/run_benchmark.sh" \
  --end_idx 3915 --result_dir_tag full_3915 "$@"
