"""Keep the original tf and the per-paper unique word lists for the three noun types"""
import gzip
import json
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from common_io import YearWriter
from config import CUT_DIR, WORKS_DIR, YEARS, NLP_ENGINE_NAME
from step1_data_cut import normalize_token, invert_index_to_text

def extract_nouns_from_paragraph_nltk(text):
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    noun_counter = {}
    for token, tag in tagged_tokens:
        if token.isalpha():
            if tag.startswith('NN'):
                lemmatized_token = lemmatizer.lemmatize(token, pos='n')
                lemmatized_token = lemmatized_token.lower()
                if len(lemmatized_token) > 1:
                    noun_counter[lemmatized_token] = None
    return list(noun_counter)

def extract_nouns_from_paragraph_spacy_token(paragraph: str, nlp):
    keep_noun = {'NOUN', 'PROPN'}
    result_noun = {}
    doc = nlp(paragraph)
    for tok in doc:
        if tok.is_space or tok.is_punct or tok.is_stop or tok.like_num:
            continue
        if tok.pos_ not in keep_noun:
            continue
        term = tok.lemma_
        if not term or term.isspace():
            continue
        term = term.lower()
        if tok.pos_ in keep_noun and len(term) > 1:
            result_noun[term] = None
    return list(result_noun)

def extract_nouns_from_paragraph_spacy_chunk(paragraph: str, nlp):

    def normalize_np(chunk):
        if chunk.root.pos_ not in {'NOUN', 'PROPN'}:
            return ''
        if chunk.root.pos_ == 'PRON':
            return ''
        tokens = [t for t in chunk if not (t.is_space or t.is_punct)]
        tokens = [t for t in tokens if t.pos_ != 'DET']
        text = ' '.join((t.lemma_.lower() if t.pos_ != 'PROPN' else t.text.lower() for t in tokens)).strip()
        if text:
            return text
        else:
            return ''
    phrases = {}
    doc = nlp(paragraph)
    for chunk in doc.noun_chunks:
        phrase = normalize_np(chunk)
        if phrase:
            phrases[phrase] = None
    return list(phrases)


def transform_work(work, nlp, stopwords):
    inv = work.get('abstract_inverted_index', {})
    title = work.get('title', '')
    if inv:
        tf = {}
        for token, positions in inv.items():
            term = normalize_token(token, stopwords, 2)
            if term is None:
                continue
            try:
                frequency = len(positions)
            except Exception:
                continue
            if frequency > 0:
                if term in tf:
                    tf[term] += frequency
                else:
                    tf[term] = frequency
        for token in title.split():
            term = normalize_token(token, stopwords, 2)
            if term is None:
                continue
            if term in tf:
                tf[term] += 1
            else:
                tf[term] = 1
        paragraph = str(title) + ' ' + invert_index_to_text(inv)
        paragraph = re.sub(r'\s+', ' ', paragraph).strip()
        nouns_nltk = extract_nouns_from_paragraph_nltk(paragraph)
        nouns_spacy = extract_nouns_from_paragraph_spacy_token(paragraph, nlp)
        nouns_chunk = extract_nouns_from_paragraph_spacy_chunk(paragraph, nlp)
    else:
        tf = {}
        nouns_nltk, nouns_spacy, nouns_chunk = [], [], []
    return {
        'year': int(work.get('year')),
        'author': work.get('author'),
        'topics': work.get('topics', []),
        'counts_by_year': work.get('counts_by_year'),
        'tf': tf,
        'nouns_nltk': nouns_nltk,
        'nouns_spacy': nouns_spacy,
        'nouns_chunk_spacy': nouns_chunk,
    }


def main():
    import spacy
    from nltk.tokenize import word_tokenize as tokenize
    from nltk import pos_tag as tag
    from nltk.stem import WordNetLemmatizer
    global word_tokenize, pos_tag, lemmatizer
    word_tokenize, pos_tag = tokenize, tag
    lemmatizer = WordNetLemmatizer()
    nlp = spacy.load(NLP_ENGINE_NAME)
    stopwords = set(ENGLISH_STOP_WORDS)
    writer = YearWriter(WORKS_DIR, YEARS)
    selected = {f'works_{year}.jsonl.gz' for year in YEARS}
    paths = [p for p in sorted(CUT_DIR.rglob('*.gz')) if p.name in selected]
    if not paths:
        raise FileNotFoundError(f'No yearly files to process: {CUT_DIR}')
    try:
        for path in paths:
            print(f'Processing {path.name}')
            with gzip.open(path, 'rt', encoding='utf-8') as handle:
                for line in handle:
                    if line.strip():
                        record = transform_work(json.loads(line), nlp, stopwords)
                        writer.write(record['year'], record)
    finally:
        writer.close()


if __name__ == '__main__':
    main()
