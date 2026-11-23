# Data files

This release provides a representative subset of 500 questions selected by
clinical category from the finalized 3,915-question set for fast evaluation.
Question text and reference answers are unchanged. Follow the
[access and download instructions](../README.md#2-install-the-500-question-package),
then install the package from the repository root:

```bash
python scripts/install_questions.py --source ./downloads/ehrcomplex
```

| Installed file | Contents |
| --- | --- |
| `dataset/questions_expected.jsonl` | 500 questions and reference answers |
| `dataset/runtime_index_map.jsonl` | Subset IDs, original full-set IDs, and question/file mappings |
| `dataset/grading/field_contracts.json` | Matching answer-field requirements used by the unified scorer |
| `data/ehr_complex/` | Runtime question files |

Each question record contains `question` and `expected_result`. Runtime task IDs
are 0–499. The `full_runtime_task_id` field in the mapping identifies each original
question in the 3,915-question set. The code repository contains the single Python
scorer; this package supplies its reference answers and field rules. References remain outside the agent's
execution environment.

Obtain MIMIC-IV separately from PhysioNet and follow
[database preparation](../database_builder/README.md).
