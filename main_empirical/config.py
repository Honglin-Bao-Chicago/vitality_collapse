from pathlib import Path

PROJECT_ROOT = Path('/project/jevans/beichen/code')
SNAPSHOT_DIR = Path('/project/jevans/openalex-snapshot/data/works')
TOPIC_CSV = PROJECT_ROOT / '20260307_result_data/20260123_all_topics_hlb.csv'
OUTPUT_ROOT = PROJECT_ROOT / '20260307_simplified'
CUT_DIR = OUTPUT_ROOT / '20260307_works_cut'
WORKS_DIR = OUTPUT_ROOT / '20260307_works_final'
RESULT_DIR = OUTPUT_ROOT / '20260307_result_final'
YEARS = tuple(range(1990, 2026))
NLP_ENGINE_NAME = 'en_core_web_lg'
