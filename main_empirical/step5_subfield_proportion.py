from collections import defaultdict
import pickle
from common_io import iter_work_records, load_topic_map
from config import WORKS_DIR, TOPIC_CSV, RESULT_DIR


def compute_heatmap(records, id_to_type):
    counts = defaultdict(lambda: defaultdict(float))
    for work in records:
        topic_ids = [topic.get('id') for topic in (work.get('topics') or []) if topic.get('id')]
        year = work.get('year')
        if year is None or not topic_ids or topic_ids[0] not in id_to_type:
            continue
        counts[year][id_to_type[topic_ids[0]]] += 1.0
    labels = ['Theory', 'System', 'AI']
    data = {label: [] for label in labels}
    for year, by_type in counts.items():
        total = sum(by_type.values())
        if total == 0:
            continue
        for label in labels:
            data[label].append(by_type[label] / total if label in by_type else 0)
    return {'y_labels': labels, 'data': data}


def main():
    result = compute_heatmap(iter_work_records(WORKS_DIR), load_topic_map(TOPIC_CSV))
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULT_DIR / 'result_topics_3_topics_main_field.pkl', 'wb') as handle:
        pickle.dump(result, handle)


if __name__ == '__main__':
    main()
