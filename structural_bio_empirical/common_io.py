import gzip
import json
from pathlib import Path

def iter_work_records(directory):
    for path in sorted(Path(directory).rglob('*.gz')):
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)

def require_empty(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError(f'Output directory is not empty; use a new output directory or move the old results away first: {directory}')
