"""Light text cleaning applied to every comment before it reaches BERT.

This is the cleaning used for BOTH training (``merge_clean.py``) and
inference (``deployment/api/ML/textprep.py`` is an identical copy, kept in
sync by ``tests/test_textprep.py``). Using the same function in both places
avoids train/serve skew: the model sees the same kind of text in production
that it was trained on.

It is deliberately much lighter than ``cleaner.normalize_corpus``. BERT was
pretrained on natural text, so stopwords, punctuation, and casing all carry
signal ("not good" vs "good", "!!!"), and ``bert-base-uncased`` lowercases
on its own. We only remove things that are pure noise: URLs, HTML entities
left over from Reddit's markdown, encoding artifacts, and extra whitespace.

Standard library only, so it can be imported by the API container and the
data pipeline without extra dependencies.
"""

import html
import re

URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
MARKDOWN_HEADER_PATTERN = re.compile(r"^#{1,6}\s*", flags=re.MULTILINE)
WHITESPACE_PATTERN = re.compile(r"\s+")

# Characters that show up when UTF-8 text was decoded as cp1252/latin-1
# (e.g. "isnâ€™t" instead of "isn't").
MOJIBAKE_MARKERS = ("â€", "Ã", "Â")


def fix_mojibake(text):
    """Best-effort repair of UTF-8 text that was mis-decoded as cp1252."""
    if not any(marker in text for marker in MOJIBAKE_MARKERS):
        return text
    try:
        # cp1252 leaves five byte values undefined (0x81, 0x8D, 0x8F, 0x90,
        # 0x9D); those survive mis-decoding as raw control characters, so
        # fall back to latin-1 for them byte-by-byte.
        raw = b"".join(_to_byte(ch) for ch in text)
        return raw.decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text  # not actually mojibake (or mixed); leave it alone


def _to_byte(ch):
    try:
        return ch.encode("cp1252")
    except UnicodeEncodeError:
        return ch.encode("latin-1")


def clean_for_model(text):
    """Normalize a single comment for BERT. Returns '' for missing input."""
    if not isinstance(text, str):
        return ""
    text = fix_mojibake(text)
    # Reddit exports are sometimes double-escaped ("&amp;#x200B;"), so unescape twice.
    text = html.unescape(html.unescape(text))
    text = text.replace("​", " ")
    text = URL_PATTERN.sub(" ", text)
    text = MARKDOWN_HEADER_PATTERN.sub("", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()
