"""Build the evaluation DuckDB from independently authorized MIMIC-IV 3.1 CSVs."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import tempfile

import duckdb


def build(source, output, specification, threads=8):
    if output.exists():
        raise FileExistsError('Output already exists; choose a new database path.')
    if threads < 1:
        raise ValueError('threads must be positive')
    inputs = {}
    for table, info in specification['tables'].items():
        path = source / info['source_file']
        if not path.is_file():
            path = path.with_suffix('')
        if not path.is_file():
            raise FileNotFoundError(f'Missing source table: {info["source_file"]}')
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt', encoding='utf-8-sig', newline='') as stream:
            if next(csv.reader(stream)) != list(info['columns']):
                raise ValueError(f'Column names/order differ from MIMIC-IV 3.1: {table}')
        inputs[table] = path
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='ehrcomplex-build-', dir=output.parent) as temporary:
        database = Path(temporary) / 'database.duckdb'
        conn = duckdb.connect(str(database), config={'threads': threads})
        try:
            for table, info in specification['tables'].items():
                columns = ', '.join(f"'{name}': '{kind}'" for name, kind in info['columns'].items())
                conn.execute(f'CREATE TABLE "{table}" AS SELECT * FROM read_csv('
                             f'?, columns={{{columns}}}, header=true, auto_detect=false, '
                             "nullstr='', strict_mode=true)", [str(inputs[table])])
                count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                if count != info['row_count']:
                    raise ValueError(f'Row count mismatch for {table}: {count} != {info["row_count"]}')
                actual = [(row[0], row[1]) for row in conn.execute(f'DESCRIBE "{table}"').fetchall()]
                if actual != list(info['columns'].items()):
                    raise ValueError(f'Database type mismatch: {table}')
                print(f'{table}: {count} rows')
            conn.execute('CHECKPOINT')
        finally:
            conn.close()
        if output.exists():
            raise FileExistsError('Output was created during the build; refusing to replace it.')
        database.rename(output)
    print('Database built and checked against the evaluation schema and row counts.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='Directory containing hosp/ and icu/')
    parser.add_argument('--output', type=Path, default=Path('data/mimic_iv_full.duckdb'))
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    spec = json.loads(Path(__file__).with_name('schema.json').read_text())
    build(args.source, args.output, spec, args.threads)
