from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Tuple
import numpy as np
import pandas as pd
from common_io import iter_work_records
from config import WORKS_DIR, RESULT_DIR

def iter_counts_by_year(counts_by_year_obj: Any) -> Iterable[Tuple[int, int]]:
    if counts_by_year_obj is None:
        return
    if isinstance(counts_by_year_obj, list):
        for item in counts_by_year_obj:
            if not isinstance(item, dict):
                continue
            y = item.get('year')
            c = item.get('cited_by_count')
            if y is None or c is None:
                continue
            yield (int(y), int(c))
        return
    if isinstance(counts_by_year_obj, dict):
        for y, c in counts_by_year_obj.items():
            try:
                yield (int(y), int(c))
            except Exception:
                continue
        return
    return

def gini_from_hist(hist: Dict[int, int]) -> float:
    if not hist:
        return np.nan
    items = sorted(((k, v) for k, v in hist.items() if v > 0))
    n = sum((v for _, v in items))
    if n == 0:
        return np.nan
    total = sum((k * v for k, v in items))
    if total <= 0:
        return 0.0
    cum_pop = 0.0
    cum_income = 0.0
    area = 0.0
    prev_L = 0.0
    prev_P = 0.0
    for k, v in items:
        cum_pop += v
        cum_income += k * v
        P = cum_pop / n
        L = cum_income / total
        area += (prev_L + L) * (P - prev_P) / 2.0
        prev_L = L
        prev_P = P
    return 1.0 - 2.0 * area

def compute_yearly_citation_inequality(records):
    histograms = defaultdict(Counter)
    for work in records:
        for year, count in iter_counts_by_year(work.get('counts_by_year')):
            if count is None:
                continue
            count = int(count)
            if count > 0:
                histograms[year][count] += 1
    if not histograms:
        raise ValueError('No valid counts_by_year entries found.')
    rows = []
    for year in range(min(histograms), max(histograms) + 1):
        value = gini_from_hist(Counter(histograms.get(year, {})))
        rows.append({'year': year, 'gini': float(value) if value == value else np.nan})
    return pd.DataFrame(rows).sort_values('year').reset_index(drop=True)


def main():
    frame = compute_yearly_citation_inequality(iter_work_records(WORKS_DIR))
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULT_DIR / 'citation3.csv', index=False)
    print('Saved citation3.csv; its correspondence to the citation.csv read by the plots is still to be verified.')

if __name__ == '__main__':
    main()
