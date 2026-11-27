# Data files

The default package provides a representative subset of 500 questions selected
by clinical category from the finalized 3,915-question set for fast evaluation.
The complete set is available through the [Full 3,915 entry](../full_3915/README.md).
Both packages preserve the finalized question text and reference answers. Follow the
[access and download instructions](../README.md#2-install-the-500-question-package),
then install the package from the repository root:

```bash
python scripts/install_questions.py --source ./downloads/ehrcomplex
```

| Installed file | Contents |
| --- | --- |
| `dataset/questions_expected.jsonl` | Questions and reference answers for the installed package |
| `dataset/runtime_index_map.jsonl` | Runtime IDs and question/file mappings |
| `dataset/grading/field_contracts.json` | Matching answer-field requirements used by the unified scorer |
| `data/ehr_complex/` | Runtime question files |

Each question record contains `question` and `expected_result`. Runtime task IDs
are 0–499 for Representative 500 and 0–3914 for Full 3,915. The subset mapping's
`full_runtime_task_id` identifies each original question in the full set; the
full package uses `runtime_task_id` directly. Install one package per clean checkout.
The code repository contains the single Python
scorer; this package supplies its reference answers and field rules. References remain outside the agent's
execution environment.

Obtain MIMIC-IV separately from PhysioNet and follow
[database preparation](../database_builder/README.md).
