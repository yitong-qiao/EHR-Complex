# [EMNLP'26] EHR-Complex

## Benchmarking Medical Agents for Complex Clinical Reasoning

**EMNLP 2026 Main Conference**

[Paper (arXiv)](https://arxiv.org/abs/2606.23301) | [PDF](https://arxiv.org/pdf/2606.23301)

Yitong Qiao, Lei Liu, Yue Shen, Jian Wang, Jinjie Gu, Zhixuan Chu, and Kui Ren.

### Overview

EHR-Complex is a benchmark for interactive clinical database reasoning built on the full MIMIC-IV database. Agents use SQL queries and Python code, inspect execution feedback, and integrate evidence across clinical events and admissions to answer patient-level and population-level questions.

### Highlights

- **Hospital-scale data:** 365K patients, 31 tables, and more than 500M records in the underlying MIMIC-IV database.
- **Systematic task coverage:** six clinical intents (Demographics, Vitals, Medications, Cost, Labs, and Diagnoses), each at patient and population levels, yielding 12 intent-scope categories.
- **Complex reasoning tasks:** 796 structural templates, 48,092 training tasks, and 3,915 test tasks.
- **Interactive evaluation:** multi-step SQL/Python execution with feedback, final-answer evaluation, and analysis of reasoning failures.

### Release status

This repository currently provides the project overview. Additional resources will be released in stages, subject to institutional approval and applicable data-use requirements.

- [x] Project overview
- [ ] Reviewed demonstration examples
- [ ] Benchmark tasks and accompanying documentation
- [ ] Environment and evaluation code

The full benchmark and demo data are **not yet publicly available**. Release updates will be posted here.

### Data access

This repository does not redistribute MIMIC-IV patient records. Access to the source database must be obtained through [PhysioNet](https://physionet.org/content/mimiciv/). Any release of derived tasks, answers, or execution trajectories will follow the applicable source-data agreement and institutional review requirements.

### Contact

For questions about EHR-Complex, please contact Yitong Qiao at [qiaoyt@zju.edu.cn](mailto:qiaoyt@zju.edu.cn).
