"""Classic bag-of-words text normalization (contractions, lowercase,
special characters, stopwords).

NOTE: this is NOT what the BERT model is trained on any more -- see
``textprep.clean_for_model``. Stopword and punctuation removal throw away
signal a transformer can use, so these helpers are kept only for
traditional (TF-IDF / bag-of-words) baselines and exploratory analysis.
"""

import re

import contractions
import nltk
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer

try:
    stopword_list = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stopword_list = set(stopwords.words('english'))

tokenizer = ToktokTokenizer()
stopword_list.remove('no')
stopword_list.remove('not')


def expand_contractions(text):
    expanded_text = contractions.fix(text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

URL_PATTERN = re.compile(r'(https?://\S+|www\.\S+)', flags=re.IGNORECASE)


def remove_urls(text):
    return URL_PATTERN.sub(' ', text)

def remove_special_characters(text, remove_digits= False):
    # A-Z, not A-z: the A-z range also matches [ \ ] ^ _ and backtick.
    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    text = re.sub(pattern, '', text)
    return text

def normalize_corpus(corpus, contraction_expansion=True,
                     text_lower_case=True, special_char_removal = True, remove_digits = False,
                     stopword_removal=True):
    
    normalized_corpus = []
    # normalize each document in the corpus
    for doc in corpus:
        # expand contractions    
        if contraction_expansion:
            doc = expand_contractions(doc)
        # lowercase the text    
        if text_lower_case:
            doc = doc.lower()
        # strip URLs first -- after special-character removal they turn
        # into junk tokens like "httpsexample com"
        doc = remove_urls(doc)
        # remove extra newlines
        doc = re.sub(r'[\r\n]+', ' ', doc)
        if special_char_removal:
            # insert spaces between special characters to isolate them    
            special_char_pattern = re.compile(r'([{.(-)!}])')
            doc = special_char_pattern.sub(" \\1 ", doc)
            doc = remove_special_characters(doc, remove_digits=remove_digits)  
        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)
        # remove stopwords
        if stopword_removal:
            doc = remove_stopwords(doc, is_lower_case=text_lower_case)
        normalized_corpus.append(doc.strip())

    return normalized_corpus
