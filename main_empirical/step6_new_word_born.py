from collections import Counter, defaultdict
import pandas as pd
from common_io import iter_work_records, load_topic_map
from config import WORKS_DIR, TOPIC_CSV, RESULT_DIR

NOUN_FIELDS = ('nouns_nltk', 'nouns_spacy', 'nouns_chunk_spacy')
GROUPS = ('Full', 'AI', 'NonAI')


def aggregate_document_frequencies(records, id_to_type):
    result = {field: {group: defaultdict(Counter) for group in GROUPS} for field in NOUN_FIELDS}
    for work in records:
        year = work.get('year')
        topics = work.get('topics') or []
        if year is None or not topics:
            continue
        topic_id = topics[0].get('id')
        if not topic_id or topic_id not in id_to_type:
            continue
        field_type = id_to_type[topic_id]
        if field_type not in {'AI', 'Theory', 'System'}:
            continue
        group = 'AI' if field_type == 'AI' else 'NonAI'
        for field in NOUN_FIELDS:
            words = set(work.get(field, {}))
            for target in ('Full', group):
                result[field][target][year].update(words)
    return result


def calculate_new_words(year2df, keep_population):
    years = sorted(year2df)
    seen = set()
    last_seen = {}
    dead = set()
    rows = []
    for year in years:
        if year == years[-1]:
            current = {word for word in year2df[year] if year2df[year][word] >= 10}
        else:
            next_year = year2df.get(year + 1, Counter())
            current = {word for word in year2df[year]
                       if year2df[year][word] >= 10 and next_year[word] >= 10}
        new_count = len(current - seen)
        for word in current:
            last_seen[word] = year
        deaths = 0
        for word in seen:
            if year - last_seen.get(word, -999) >= 3 and word not in current:
                if word not in dead:
                    deaths += 1
                    dead.add(word)
            elif word in dead and word in current:
                dead.remove(word)
        seen |= current
        row = {
            'year': year,
            'valid_new_count_noun': new_count,
            'valid_unique_count_noun': len(current),
            'valid_temp_dead_count_noun': deaths,
        }
        if keep_population:
            row['valid_accumulate_noun'] = len(seen)
            row['valid_temp_dead_accumulate_noun'] = len(dead)
        rows.append(row)
    columns = ['year', 'valid_new_count_noun', 'valid_unique_count_noun', 'valid_temp_dead_count_noun']
    if keep_population:
        columns += ['valid_accumulate_noun', 'valid_temp_dead_accumulate_noun']
    return pd.DataFrame(rows, columns=columns)


def main():
    data = aggregate_document_frequencies(iter_work_records(WORKS_DIR), load_topic_map(TOPIC_CSV))
    out_dir = RESULT_DIR / 'new_word_born'
    out_dir.mkdir(parents=True, exist_ok=True)
    for field in NOUN_FIELDS:
        for group in GROUPS:
            frame = calculate_new_words(data[field][group], keep_population=(group == 'Full'))
            if field == 'nouns_chunk_spacy':
                filename = f'new_words_spacy_chunk_{group}_spacy.csv'
            else:
                engine = 'nltk' if field == 'nouns_nltk' else 'spacy'
                filename = f'new_words_{group}_{engine}.csv'
            frame.to_csv(out_dir / filename, index=False)
            print(filename)


if __name__ == '__main__':
    main()
