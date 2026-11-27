# Full 3,915 evaluation

Evaluate all 3,915 finalized EHR-Complex questions with the same framework and
unified scorer used by Representative 500. Question text, reference answers,
and scoring rules are unchanged.

Use a clean checkout for this package. If Representative 500 is already
installed, use a separate checkout; the installer does not overwrite a different
question package. Both checkouts can share one database via `EHR_COMPLEX_DB`.

Follow the main README for [data access](../README.md#data-access),
[environment setup](../README.md#setup),
[MIMIC-IV CSV conversion](../README.md#1-convert-mimic-iv-csvs-to-duckdb),
and [model credentials](../README.md#3-configure-a-model-and-run-evaluation).
Run the commands below from the repository root.

## Download and install

After email approval, download from the same authorized
[ModelScope dataset](https://www.modelscope.cn/datasets/qiaoyitong/EHR-Complex):

```bash
python -m pip install modelscope
export MODELSCOPE_API_TOKEN=YOUR_MODELSCOPE_TOKEN
modelscope download --dataset qiaoyitong/EHR-Complex \
  --local_dir ./downloads/ehrcomplex --token "$MODELSCOPE_API_TOKEN"
python scripts/install_questions.py --source ./downloads/ehrcomplex/full_3915
```

For a website download, preserve this layout:

```text
downloads/ehrcomplex/full_3915/
  questions_expected.jsonl
  runtime_index_map.jsonl
  grading/field_contracts.json
```

The installer verifies all 3,915 question-to-answer mappings and preserves task
IDs 0–3914. It installs the matching field rules for the repository's existing
scorer. The ModelScope package contains no Python scorer or database.

## Run

```bash
bash full_3915/run_benchmark.sh \
  --credentials_path credentials.toml \
  --model_name YOUR_MODEL_NAME \
  --result_dir_tag full_3915_evaluation_01 \
  --async_run --parallel_backend joblib --n_jobs 32 \
  --temperature 1 --top_p 0.95 --top_k 20 --enable_thinking
```

This entry checks that the full package is installed and selects all 3,915 tasks.
The 500-question entry remains `run_benchmark.sh` in a checkout with the
Representative 500 package. Adjust concurrency and model-specific options as
described in the main README. Step limits, tool isolation, reasoning history,
and scoring behavior are shared by both entries.

Use a new result tag for each evaluation. [Scores and trajectories](../README.md#4-read-scores-and-trajectories)
use the existing format. Add `--resume` to continue an interrupted full-set run
or `--rescore` to rescore its saved final answers. Report these results as
**Full 3,915**, separately from **Representative 500**.
