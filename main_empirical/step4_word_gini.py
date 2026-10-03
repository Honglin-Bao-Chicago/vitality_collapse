"""Keep only f1000 and k=5/10/15; reuse the original TF-IDF, tie ordering, and Gini algorithms."""
import gzip
import json
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from config import WORKS_DIR, RESULT_DIR

def gini_coefficient(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[x > 0]
    if x.size == 0:
        return np.nan
    x = np.sort(x)
    n = x.size
    cum = np.sum(np.arange(1, n + 1) * x)
    return 2.0 * cum / (n * np.sum(x)) - (n + 1) / n

def top_k_from_sparse_row(indices: np.ndarray, data: np.ndarray, feature_names: np.ndarray, k: int):
    if data.size == 0:
        return []
    if data.size <= k:
        order = np.argsort(data)[::-1]
    else:
        idx = np.argpartition(data, -k)[-k:]
        order = idx[np.argsort(data[idx])[::-1]]
    return [(str(feature_names[indices[i]]), float(data[i])) for i in order[:k]]


def read_year_jsonl_gz(path):
    year_value = None
    tf_dicts = []
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            work = json.loads(line)
            year = work.get('year')
            tf = work.get('tf')
            # The original 'all' branch also assigned work_id as tf, so it does not actually filter on whether the id is missing.
            if year is None or not isinstance(tf, dict) or len(tf) == 0:
                continue
            if year_value is None:
                year_value = int(year)
            tf_dicts.append({str(key): int(value) for key, value in tf.items() if value})
    if year_value is None:
        raise ValueError(f'No valid records in file: {path}')
    return year_value, tf_dicts


def build_vocab(tf_dicts):
    df_counter, tf_total = Counter(), Counter()
    for tf in tf_dicts:
        tf_total.update(tf)
        df_counter.update(tf.keys())
    max_df = int(np.floor(0.8 * len(tf_dicts)))
    candidates = [term for term, df in df_counter.items() if df >= 2 and df <= max_df]
    # Keep the stable sort and original traversal order; do not add a sort key for words with equal frequency.
    candidates.sort(key=lambda term: tf_total[term], reverse=True)
    return set(candidates[:1000])


def process_one_year(path):
    year, tf_dicts = read_year_jsonl_gz(path)
    vocab = build_vocab(tf_dicts)
    filtered = [{term: count for term, count in tf.items() if term in vocab} for tf in tf_dicts]
    vectorizer = DictVectorizer(sparse=True)
    counts = vectorizer.fit_transform(filtered)
    feature_names = vectorizer.get_feature_names_out()
    tfidf = TfidfTransformer(norm='l2', smooth_idf=True, use_idf=True).fit_transform(counts)
    result = {}
    for top_k in (5, 10, 15):
        bucket = Counter()
        for i in range(tfidf.shape[0]):
            row = tfidf.getrow(i)
            pairs = top_k_from_sparse_row(row.indices, row.data, feature_names, top_k)
            for word, _ in pairs:
                bucket[word] += 1
        value = gini_coefficient(np.array(list(bucket.values()), dtype=np.float64))
        result[top_k] = float(value) if value == value else np.nan
    return year, result


def main():
    files = sorted(WORKS_DIR.glob('works_*.jsonl.gz'))
    if not files:
        raise FileNotFoundError(f'No yearly files: {WORKS_DIR}')
    rows = {k: [] for k in (5, 10, 15)}
    for path in files:
        year, result = process_one_year(path)
        for k, gini in result.items():
            rows[k].append({'year': int(year), 'gini': gini})
        print(f'{year} done')
    out_dir = RESULT_DIR / 'word_gini'
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, data in rows.items():
        pd.DataFrame(data).sort_values('year').to_csv(out_dir / f'word_gini_f1000_k{k}.csv', index=False)


if __name__ == '__main__':
    main()
