#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="$(command -v "${PYTHON_BIN:-python3}")"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONDONTWRITEBYTECODE=1
exec "$PYTHON_BIN" main.py --config_path configs/base.yaml --task ehr_complex \
  --data_path "$PROJECT_DIR/data/ehr_complex" --work_dir "$PROJECT_DIR/results" \
  --mode test --start_idx 0 --end_idx 500 "$@"
