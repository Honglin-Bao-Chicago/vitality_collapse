from __future__ import annotations
from typing import Any, Dict, List, Set
import os
import re
import json
import gzip
import shutil
import multiprocessing as mp
from pathlib import Path
from tempfile import TemporaryDirectory
from collections import Counter
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import spacy
from config import CUT_DIR, NOUN_DIR, OUTPUT_ROOT, NLP_WORKERS
from common_io import require_empty
NLP_ENGINE_NAME = 'en_core_web_lg'
KEEP_NOUN = {'NOUN', 'PROPN'}
LEMMATIZER = WordNetLemmatizer()
NLP = None
BATCH_SIZE = 200
CHUNK_LINES = 5000

def get_work_value(work: Any, key: str, default=None):
    if isinstance(work, dict):
        return work.get(key, default)
    return getattr(work, key, default)

def invert_index_to_text(inv: Dict[str, list]) -> str:
    max_pos = max((p for positions in inv.values() for p in positions))
    tokens = [None] * (max_pos + 1)
    for w, positions in inv.items():
        for p in positions:
            tokens[p] = w
    if any((t is None for t in tokens)):
        missing = [i for i, t in enumerate(tokens) if t is None]
        for i in missing:
            tokens[i] = ''
    result = ' '.join(tokens)
    if not result:
        return ''
    return result

def build_paragraph(inv: Dict[str, list], title: str) -> str:
    paragraph = invert_index_to_text(inv)
    paragraph = str(title) + ' ' + paragraph
    paragraph = re.sub('\\s+', ' ', paragraph).strip()
    return paragraph

def nouns_spacy_token_from_doc(doc) -> Dict[str, int]:
    result_noun = Counter()
    for tok in doc:
        if tok.is_space or tok.is_punct or tok.is_stop or tok.like_num:
            continue
        if tok.pos_ not in KEEP_NOUN:
            continue
        term = tok.lemma_
        if not term or term.isspace():
            continue
        term = term.lower()
        if tok.pos_ in KEEP_NOUN and len(term) > 1:
            result_noun[term] += 1
    return dict(result_noun)

def _normalize_np(chunk) -> str:
    if chunk.root.pos_ not in {'NOUN', 'PROPN'}:
        return ''
    tokens = [t for t in chunk if not (t.is_space or t.is_punct)]
    tokens = [t for t in tokens if t.pos_ != 'DET']
    text = ' '.join((t.lemma_.lower() if t.pos_ != 'PROPN' else t.text.lower() for t in tokens)).strip()
    return text if text else ''

def nouns_spacy_chunk_from_doc(doc) -> Dict[str, int]:
    phrases = Counter()
    for chunk in doc.noun_chunks:
        phrase = _normalize_np(chunk)
        if phrase:
            phrases[phrase] += 1
    return dict(phrases)

def extract_nouns_from_paragraph_nltk(text: str) -> Dict[str, int]:
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    noun_counter = Counter()
    for token, tag in tagged_tokens:
        if token.isalpha():
            if tag.startswith('NN'):
                lemmatized_token = LEMMATIZER.lemmatize(token, pos='n')
                lemmatized_token = lemmatized_token.lower()
                if len(lemmatized_token) > 1:
                    noun_counter[lemmatized_token] += 1
    return dict(noun_counter)

def year_of(name: str) -> str:
    stem = name
    for suf in ('.jsonl.gz', '.gz'):
        if stem.endswith(suf):
            stem = stem[:-len(suf)]
            break
    if stem.startswith('works_'):
        stem = stem[len('works_'):]
    return stem.split('_')[0]

def _init_worker():
    global NLP
    NLP = spacy.load(NLP_ENGINE_NAME, disable=['ner'])

def merge_year_chunks(processed_dir: str, out_dir: str, valid_years: Set[str]):
    os.makedirs(out_dir, exist_ok=True)
    year_to_parts: Dict[str, list] = {}
    for fn in sorted(os.listdir(processed_dir)):
        if fn.endswith('.jsonl.gz'):
            y = year_of(fn)
            if y in valid_years:
                year_to_parts.setdefault(y, []).append(os.path.join(processed_dir, fn))
    for year, parts in tqdm(sorted(year_to_parts.items()), desc='Merging by year'):
        parts.sort()
        out_path = os.path.join(out_dir, f'works_{year}.jsonl.gz')
        with open(out_path, 'wb') as out_f:
            for p in parts:
                with open(p, 'rb') as in_f:
                    shutil.copyfileobj(in_f, out_f, length=1024 * 1024)

def ensure_nltk():
    for pkg, path in [('punkt_tab',
            'tokenizers/punkt_tab'),
        ('averaged_perceptron_tagger_eng',
            'taggers/averaged_perceptron_tagger_eng'),
        ('wordnet',
            'corpora/wordnet')]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg)

def split_year_file(task):
    year_path, chunk_dir, chunk_lines = task
    year = year_of(os.path.basename(year_path))
    os.makedirs(chunk_dir, exist_ok=True)
    k = 0
    cnt = 0
    out = None
    try:
        with gzip.open(year_path, 'rt', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                if cnt % chunk_lines == 0:
                    if out is not None:
                        out.close()
                    out_path = os.path.join(chunk_dir, f'works_{year}_{k:05d}.jsonl.gz')
                    out = gzip.open(out_path, 'wt', encoding='utf-8')
                    k += 1
                out.write(line if line.endswith('\n') else line + '\n')
                cnt += 1
    finally:
        if out is not None:
            out.close()
    return None

def process_chunk(task):
    chunk_path, processed_dir = task
    name = os.path.basename(chunk_path)
    out_path = os.path.join(processed_dir, name)
    os.makedirs(processed_dir, exist_ok=True)
    records: List[Dict] = []
    with gzip.open(chunk_path, 'rt', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            work = json.loads(line)
            work_id = get_work_value(work, 'id')
            yr = get_work_value(work, 'year')
            title = get_work_value(work, 'title', '')
            inv = get_work_value(work, 'abstract_inverted_index', dict())
            counts_by_year = get_work_value(work, 'counts_by_year')
            if inv:
                paragraph = build_paragraph(inv, title)
                has_inv = True
            else:
                paragraph = ''
                has_inv = False
            records.append({'id': str(work_id),
                    'year': int(yr),
                    'counts_by_year': counts_by_year,
                    '_has_inv': has_inv,
                    '_paragraph': paragraph})
    nlp_idx = [i for i, r in enumerate(records) if r['_has_inv']]
    paragraphs = [records[i]['_paragraph'] for i in nlp_idx]
    docs = NLP.pipe(paragraphs, batch_size=BATCH_SIZE)
    spacy_results: Dict[int, Dict] = {}
    for i, doc in zip(nlp_idx, docs):
        spacy_results[i] = {'nouns_spacy': nouns_spacy_token_from_doc(doc),
            'nouns_chunk_spacy': nouns_spacy_chunk_from_doc(doc)}
    with gzip.open(out_path, 'wt', encoding='utf-8') as out_f:
        for i, r in enumerate(records):
            if r['_has_inv']:
                sr = spacy_results[i]
                nouns_spacy_token = sr['nouns_spacy']
                nouns_spacy_chunk = sr['nouns_chunk_spacy']
                nouns_nltk = extract_nouns_from_paragraph_nltk(r['_paragraph'])
            else:
                nouns_spacy_token = dict()
                nouns_spacy_chunk = dict()
                nouns_nltk = dict()
            obj = {'id': r['id'],
                'year': r['year'],
                'counts_by_year': r['counts_by_year'],
                'nouns_nltk': nouns_nltk,
                'nouns_spacy': nouns_spacy_token,
                'nouns_chunk_spacy': nouns_spacy_chunk}
            out_f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    return None

def main():
    ensure_nltk()
    valid = {str(y) for y in range(1990, 2026)}
    files = [str(p) for p in sorted(CUT_DIR.glob('works_*.jsonl.gz')) if year_of(p.name) in valid]
    if not files:
        raise FileNotFoundError(CUT_DIR)
    require_empty(NOUN_DIR)
    with TemporaryDirectory(prefix='step2_', dir=OUTPUT_ROOT) as tmp:
        chunks = Path(tmp) / 'chunks'
        processed = Path(tmp) / 'processed'
        chunks.mkdir()
        processed.mkdir()
        ctx = mp.get_context('spawn')
        tasks = [(p, str(chunks), CHUNK_LINES) for p in files]
        with ctx.Pool(processes=max(1, min(NLP_WORKERS, len(tasks)))) as pool:
            for _ in tqdm(pool.imap_unordered(split_year_file,
                    tasks,
                    chunksize=1),
                total=len(tasks),
                desc='Splitting'):
                pass
        tasks = [(str(p), str(processed)) for p in sorted(chunks.glob('*.jsonl.gz'))]
        with ctx.Pool(processes=NLP_WORKERS, initializer=_init_worker) as pool:
            for _ in tqdm(pool.imap_unordered(process_chunk, tasks, chunksize=1), total=len(tasks), desc='NLP'):
                pass
        merge_year_chunks(str(processed), str(NOUN_DIR), valid)
        
if __name__ == '__main__':
    main()
