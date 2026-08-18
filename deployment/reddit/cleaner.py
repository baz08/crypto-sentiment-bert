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

def remove_special_characters(text, remove_digits= False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
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
        # remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)
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
            normalized_corpus.append(doc)
        
    return normalized_corpus