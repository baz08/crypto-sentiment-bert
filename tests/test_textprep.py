from pathlib import Path

from textprep import clean_for_model, fix_mojibake

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_training_and_api_cleaning_are_identical():
    # The API container can only see deployment/api, so it carries its own
    # copy of textprep.py. If the copies drift, the deployed model would see
    # differently-cleaned text than it was trained on.
    pipeline_copy = (REPO_ROOT / "deployment" / "reddit" / "textprep.py").read_text()
    api_copy = (REPO_ROOT / "deployment" / "api" / "ML" / "textprep.py").read_text()
    assert pipeline_copy == api_copy


def test_removes_urls():
    assert clean_for_model("see https://example.com/x and www.foo.org now") == "see and now"


def test_unescapes_reddit_html_entities():
    assert clean_for_model("&gt; quoted &amp;#x200B; text") == "> quoted text"


def test_keeps_stopwords_punctuation_and_case():
    # BERT uses these, unlike the bag-of-words cleaner.
    assert clean_for_model("This is NOT good!!") == "This is NOT good!!"


def test_strips_markdown_headers_and_whitespace():
    assert clean_for_model("  \n\n#### Title\n\nbody  ") == "Title body"


def test_fixes_mojibake():
    assert fix_mojibake("isnâ€™t") == "isn’t"


def test_leaves_normal_unicode_alone():
    assert fix_mojibake("café — ok") == "café — ok"


def test_missing_text_becomes_empty_string():
    assert clean_for_model(None) == ""
    assert clean_for_model(float("nan")) == ""
