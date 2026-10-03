from __future__ import annotations
from typing import Any, Dict, Optional, Set
import os
import json
import gzip
import shutil
import multiprocessing as mp
from tempfile import TemporaryDirectory
from collections import OrderedDict
from pathlib import Path
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from tqdm import tqdm
from config import SNAPSHOT_WORKS_DIR, CUT_DIR, OUTPUT_ROOT, FILTER_WORKERS
from common_io import require_empty
MAX_OPEN = 40
ALLOWED_TYPES = {'article', 'journal-article'}
ALLOWED_LANGUAGES = {'en'}
STOPWORDS = set(ENGLISH_STOP_WORDS)

def get_work_value(work: Any, key: str, default=None):
    if isinstance(work, dict):
        return work.get(key, default)
    return getattr(work, key, default)

def normalize_token(token: str, stopwords: Set[str], min_token_len: int) -> Optional[str]:
    if token is None:
        return None
    t = str(token).strip().lower()
    if len(t) < min_token_len:
        return None
    if t in stopwords:
        return None
    if t.isdigit():
        return None
    return t

class GzipYearWriter:

    def __init__(self, out_dir: str, max_open_files: int=MAX_OPEN):
        self.out_dir = out_dir
        self.max_open_files = max_open_files
        self._handles: 'OrderedDict[int, Any]' = OrderedDict()
        os.makedirs(out_dir, exist_ok=True)

    def _open(self, year: int):
        path = os.path.join(self.out_dir, f'works_{year}.jsonl.gz')
        return gzip.open(path, mode='at', encoding='utf-8')

    def write(self, year: int, obj: Dict):
        if year in self._handles:
            f = self._handles.pop(year)
            self._handles[year] = f
        else:
            if len(self._handles) >= self.max_open_files:
                _, old = self._handles.popitem(last=False)
                old.close()
            f = self._open(year)
            self._handles[year] = f
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    def close(self):
        for _, f in self._handles.items():
            f.close()
        self._handles.clear()

def process_one_file(task):
    idx, in_path, tmp_dir = task
    shard_dir = os.path.join(tmp_dir, f'shard_{idx:05d}')
    if os.path.isdir(shard_dir):
        shutil.rmtree(shard_dir, ignore_errors=True)
    os.makedirs(shard_dir, exist_ok=True)
    writer = GzipYearWriter(shard_dir, max_open_files=MAX_OPEN)
    try:
        with gzip.open(in_path, 'rt', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                work = json.loads(line)
                work_id = get_work_value(work, 'id')
                wtype = work.get('type')
                language = work.get('language')
                year = get_work_value(work, 'publication_year')
                title = get_work_value(work, 'title', '')
                inv = get_work_value(work, 'abstract_inverted_index')
                topics = get_work_value(work, 'topics', [])
                primary_topic = get_work_value(work, 'primary_topic', {})
                counts_by_year = get_work_value(work, 'counts_by_year')
                if not topics or not primary_topic:
                    continue
                if primary_topic.get('field', dict()).get('id', '') not in ['https://openalex.org/fields/13']:
                    continue
                if not language or language not in ALLOWED_LANGUAGES:
                    continue
                if work_id is None or not inv:
                    continue
                if year is None or year < 1990 or year > 2025:
                    continue
                if ALLOWED_TYPES and wtype not in ALLOWED_TYPES:
                    continue
                has_token = False
                for tok, positions in inv.items():
                    t = normalize_token(tok, STOPWORDS, 2)
                    if t is None:
                        continue
                    try:
                        freq = len(positions)
                    except Exception:
                        continue
                    if freq > 0:
                        has_token = True
                if not has_token:
                    continue
                writer.write(int(year),
                    {'id': str(work_id),
                        'year': int(year),
                        'title': str(title),
                        'abstract_inverted_index': inv,
                        'counts_by_year': counts_by_year})
    finally:
        writer.close()
    return None

def merge_year_shards(tmp_dir: str, out_dir: str) -> Dict[str, int]:
    os.makedirs(out_dir, exist_ok=True)
    year_to_parts: Dict[str, list] = {}
    for shard_name in sorted(os.listdir(tmp_dir)):
        shard_path = os.path.join(tmp_dir, shard_name)
        if not os.path.isdir(shard_path):
            continue
        for fn in os.listdir(shard_path):
            if fn.startswith('works_') and fn.endswith('.jsonl.gz'):
                year = fn[len('works_'):-len('.jsonl.gz')]
                year_to_parts.setdefault(year, []).append(os.path.join(shard_path, fn))
    for year, parts in tqdm(sorted(year_to_parts.items()), desc='Merging by year'):
        parts.sort()
        out_path = os.path.join(out_dir, f'works_{year}.jsonl.gz')
        with open(out_path, 'wb') as out_f:
            for p in parts:
                with open(p, 'rb') as in_f:
                    shutil.copyfileobj(in_f, out_f, length=1024 * 1024)

def main():
    if SNAPSHOT_WORKS_DIR.resolve() in OUTPUT_ROOT.resolve().parents or SNAPSHOT_WORKS_DIR.resolve() == OUTPUT_ROOT.resolve():
        raise ValueError('The output directory must not be inside the input snapshot directory.')
    gz_files = sorted(Path(SNAPSHOT_WORKS_DIR).rglob('*.gz'))
    if not gz_files:
        raise FileNotFoundError(SNAPSHOT_WORKS_DIR)
    gz_files.sort(key=lambda p: p.stat().st_size, reverse=True)
    require_empty(CUT_DIR)
    with TemporaryDirectory(prefix='step1_', dir=OUTPUT_ROOT) as tmp:
        tasks = [(i, str(p), tmp) for i, p in enumerate(gz_files)]
        with mp.get_context('spawn').Pool(processes=FILTER_WORKERS) as pool:
            for _ in tqdm(pool.imap_unordered(process_one_file,
                    tasks,
                    chunksize=1),
                total=len(tasks),
                desc='Filtering'):
                pass
        merge_year_shards(tmp, CUT_DIR)
if __name__ == '__main__':
    main()
