"""Keep author in-migration counts for the three fields"""
from collections import Counter, defaultdict
import pandas as pd
from common_io import iter_work_records, load_topic_map
from config import WORKS_DIR, TOPIC_CSV, RESULT_DIR

def paper_to_type(work, data_dict):
    topics = work.get('topics') or []
    if not topics:
        return 'Others'
    main_topic_id = topics[0].get('id')
    if not main_topic_id:
        return 'Others'
    if main_topic_id not in data_dict:
        return 'Others'
    work_subfield = data_dict[main_topic_id]
    if work_subfield not in ['Theory', 'System', 'AI']:
        return 'Others'
    return work_subfield


def build_author_year_counts(records, paper_type_data):
    result = defaultdict(lambda: defaultdict(Counter))
    for work in records:
        label = paper_to_type(work, paper_type_data)
        year = work.get('year')
        if year is None:
            continue
        try:
            year = int(year)
        except Exception:
            continue
        authors = work.get('author')
        if not authors or label not in {'AI', 'Theory', 'System'}:
            continue
        for item in authors:
            author = item.get('author')
            if not author or not author.get('id'):
                continue
            result[author['id']][year][label] += 1
    return result


def assign_author_year_label(author_year_counts):
    author_year_label = defaultdict(dict)
    active_authors_by_year = defaultdict(set)
    for aid, year_map in author_year_counts.items():
        for year, cnt in year_map.items():
            a_n = cnt.get('AI', 0)
            t_n = cnt.get('Theory', 0)
            s_n = cnt.get('System', 0)
            m = max(a_n, t_n, s_n)
            if m <= 0:
                continue
            winners = []
            if a_n == m:
                winners.append('AI')
            if t_n == m:
                winners.append('Theory')
            if s_n == m:
                winners.append('System')
            if len(winners) != 1:
                continue
            lab = winners[0]
            author_year_label[aid][year] = lab
            active_authors_by_year[year].add(aid)
    return (author_year_label, active_authors_by_year)

def compute_migration_by_year(author_year_label):
    migration_counts_by_year = defaultdict(Counter)
    for aid, year_map in author_year_label.items():
        years = sorted(year_map.keys())
        prev = None
        for year in years:
            cur = year_map[year]
            if prev is None:
                migration_counts_by_year[year][f'New_to_{cur}'] += 1
            elif cur != prev:
                migration_counts_by_year[year][f'Old_to_{cur}'] += 1
            prev = cur
    return migration_counts_by_year


def build_migration_report_df(migration_counts_by_year, active_authors_by_year):
    years = sorted(set(active_authors_by_year) | set(migration_counts_by_year))
    rows = []
    for year in years:
        row = {'year': year, 'active_authors': len(active_authors_by_year.get(year, set()))}
        row.update(migration_counts_by_year.get(year, Counter()))
        rows.append(row)
    frame = pd.DataFrame(rows).fillna(0)
    columns = ['year', 'active_authors'] + [
        f'{prefix}_to_{field}' for field in ['AI', 'System', 'Theory'] for prefix in ['New', 'Old']]
    return frame.reindex(columns=columns, fill_value=0)


def main():
    counts = build_author_year_counts(iter_work_records(WORKS_DIR), load_topic_map(TOPIC_CSV))
    labels, active = assign_author_year_label(counts)
    del counts
    frame = build_migration_report_df(compute_migration_by_year(labels), active)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULT_DIR / 'author_migration.csv', index=False, encoding='utf-8')


if __name__ == '__main__':
    main()
