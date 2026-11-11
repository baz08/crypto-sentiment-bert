from cleaner import expand_contractions, normalize_corpus, remove_special_characters, remove_stopwords


def test_expand_contractions():
    assert expand_contractions("I don't think it's going anywhere") == "I do not think it is going anywhere"


def test_remove_special_characters():
    assert remove_special_characters("to the moon!! rocket") == "to the moon rocket"


def test_remove_stopwords_keeps_negation():
    assert remove_stopwords("this is not good") == "not good"


def test_remove_special_characters_strips_brackets_and_underscores():
    # Regression test: the old [^a-zA-z...] range let [ \\ ] ^ _ ` through.
    assert remove_special_characters("a_b [c] ^d") == "ab c d"


def test_normalize_corpus_removes_urls_before_special_characters():
    result = normalize_corpus(["Check this out: https://example.com NOW!"])
    assert result == ["check"]


def test_normalize_corpus_keeps_docs_when_stopword_removal_disabled():
    # Regression test: documents were only appended inside the stopword branch.
    result = normalize_corpus(["Hello World"], stopword_removal=False)
    assert result == ["hello world"]
