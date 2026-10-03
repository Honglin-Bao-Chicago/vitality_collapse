import gzip
import json
from pathlib import Path

def iter_work_records(directory):
    for path in sorted(Path(directory).rglob('*.gz')):
        with gzip.open(path, 'rt', encoding='utf-8') as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def load_topic_map(path):
    import pandas as pd
    frame = pd.read_csv(path, usecols=['topic_id', 'subfield_name_2_new'])
    frame = frame.dropna(subset=['topic_id', 'subfield_name_2_new'])
    return dict(zip(frame['topic_id'], frame['subfield_name_2_new']))


class YearWriter:
    def __init__(self, directory, years):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.handles = {}
        for year in years:
            path = self.directory / f'works_{year}.jsonl.gz'
            if path.exists():
                raise FileExistsError(f'Output file already exists; use an empty directory or move the file away: {path}')

    def write(self, year, record):
        if year not in self.handles:
            self.handles[year] = gzip.open(
                self.directory / f'works_{year}.jsonl.gz', 'xt', encoding='utf-8')
        self.handles[year].write(json.dumps(record, ensure_ascii=False) + '\n')

    def close(self):
        for handle in self.handles.values():
            handle.close()
