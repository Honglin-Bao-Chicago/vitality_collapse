from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from common_io import iter_work_records, YearWriter
from config import SNAPSHOT_DIR, CUT_DIR

def normalize_token(token, stopwords, min_token_len):
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

def invert_index_to_text(inv: dict[str, list[int]]) -> list[str]:
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


def select_work(work, stopwords):
    work_id = work.get('id')
    year = work.get('publication_year')
    inv = work.get('abstract_inverted_index')
    topics = work.get('topics', [])
    primary_topic = work.get('primary_topic', {})
    if not topics or not primary_topic or primary_topic.get('field', {}).get('id', '') != 'https://openalex.org/fields/17':
        return None
    if work.get('language') not in {'en'}:
        return None
    if work_id is None or not inv:
        return None
    if year is None or year < 1990 or year > 2025:
        return None
    if work.get('type') not in {'article', 'journal-article'}:
        return None

    valid_token = False
    for token, positions in inv.items():
        if normalize_token(token, stopwords, 2) is None:
            continue
        try:
            frequency = len(positions)
        except Exception:
            continue
        if frequency > 0:
            valid_token = True
            break
    if not valid_token:
        return None
    paragraph = invert_index_to_text(inv)
    if not paragraph or not isinstance(paragraph, str):
        return None

    authors = work.get('authorships')
    if authors is not None:
        authors = [{'author': {'id': (item.get('author') or {}).get('id')}} for item in authors]
    return {
        'year': int(year),
        'author': authors,
        'topics': [{'id': topic.get('id')} for topic in topics],
        'title': str(work.get('title', '')),
        'abstract_inverted_index': inv,
        'counts_by_year': work.get('counts_by_year'),
    }


def main():
    writer = YearWriter(CUT_DIR, range(1990, 2026))
    stopwords = set(ENGLISH_STOP_WORDS)
    n_written = 0
    try:
        for work in iter_work_records(SNAPSHOT_DIR):
            record = select_work(work, stopwords)
            if record is not None:
                writer.write(record['year'], record)
                n_written += 1
    finally:
        writer.close()
    print(f'step1 done: {n_written} papers')


if __name__ == '__main__':
    main()
