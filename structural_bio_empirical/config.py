from pathlib import Path
import os
SNAPSHOT_WORKS_DIR = Path('F:\\openalex-snapshot\\data\\works')
OUTPUT_ROOT = Path('F:\\20260612_extra_data\\extra_AlphaFold_simplified')
CUT_DIR = OUTPUT_ROOT / 'cut'
NOUN_DIR = OUTPUT_ROOT / 'add_noun'
GINI_DIR = OUTPUT_ROOT / 'result_word_gini'
NEW_WORD_DIR = OUTPUT_ROOT / 'new_word_born'
CITATION_DIR = OUTPUT_ROOT / 'result_citation_gini'
FIGURE_DIR = OUTPUT_ROOT / 'figures'
FILTER_WORKERS = max(1, (os.cpu_count() or 3) - 2)
NLP_WORKERS = max(1, (os.cpu_count() or 2) - 1)
