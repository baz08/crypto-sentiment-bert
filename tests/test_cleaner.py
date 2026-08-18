from cleaner import expand_contractions, normalize_corpus, remove_special_characters, remove_stopwords


def test_expand_contractions():
    assert expand_contractions("I don't think it's going anywhere") == "I do not think it is going anywhere"


def test_remove_special_characters():
    assert remove_special_characters("to the moon!! rocket") == "to the moon rocket"


def test_remove_stopwords_keeps_negation():
    assert remove_stopwords("this is not good") == "not good"


def test_normalize_corpus_lowercases_and_strips_stopwords():
    result = normalize_corpus(["Check this out: https://example.com NOW!"])
    assert result == ["check httpsexample com"]
