# Prepare the database

Download the **full MIMIC-IV 3.1** hospital and ICU files from
[PhysioNet](https://physionet.org/content/mimiciv/3.1/) using authorized access.
Keep the source layout:

```text
downloads/mimiciv/3.1/
  hosp/*.csv.gz
  icu/*.csv.gz
```

Uncompressed `.csv` files are also supported. From the repository root, using
the evaluation Python environment:

```bash
python database_builder/build_database.py \
  --source ./downloads/mimiciv/3.1 \
  --output ./data/mimic_iv_full.duckdb \
  --threads 8
```

The builder creates 31 tables using `schema.json`, preserving source rows,
column names, types, and leading zeros in text codes. The output path must be
new. Allow space for the source files, temporary build, and database
(approximately 18 GB).

Set `EHR_COMPLEX_DB` if the output is stored elsewhere, then follow
[the evaluation instructions](../README.md#2-install-the-500-question-package).
