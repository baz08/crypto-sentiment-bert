"""Merge the four labeled comment sources into one training CSV for BERT.

Inputs (all in this folder):
    BTC_comments.csv      - scraped r/CryptoCurrency BTC comments, labeled (body, Sentiment)
    ETH_comments.csv      - scraped r/CryptoCurrency ETH comments, labeled (body, Sentiment)
    sentiment.csv         - labeled comments, positive/negative only (Comment Text, Sentiment)
    sentiment_labels.csv  - labeled posts (text, sentiment)

Output:
    Crypto_c.csv with columns
        DATA_COLUMN   - text after textprep.clean_for_model
        LABEL_COLUMN  - 0 = negative, 1 = neutral, 2 = positive
        source        - which input file the row came from

Usage:
    python merge_clean.py
    python merge_clean.py --output ./Crypto_c.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from textprep import clean_for_model

HERE = Path(__file__).resolve().parent
VALID_LABELS = {0, 1, 2}

# file name -> (text column, label column)
SOURCES = {
    "BTC_comments.csv": ("body", "Sentiment"),
    "ETH_comments.csv": ("body", "Sentiment"),
    "sentiment.csv": ("Comment Text", "Sentiment"),
    "sentiment_labels.csv": ("text", "sentiment"),
}


def load_source(path, text_column, label_column):
    df = pd.read_csv(path)
    return pd.DataFrame(
        {
            "DATA_COLUMN": df[text_column],
            "LABEL_COLUMN": df[label_column],
            "source": Path(path).stem,
        }
    )


def build_training_frame(frames):
    """Combine per-source frames, clean the text, and drop unusable rows."""
    df = pd.concat(frames, ignore_index=True)

    df = df.dropna(subset=["LABEL_COLUMN"])
    df["LABEL_COLUMN"] = df["LABEL_COLUMN"].astype(int)
    df = df[df["LABEL_COLUMN"].isin(VALID_LABELS)]

    df["DATA_COLUMN"] = df["DATA_COLUMN"].map(clean_for_model)
    df = df[df["DATA_COLUMN"].str.len() > 0]

    # The same comment can appear in more than one source; keep the first.
    df = df.drop_duplicates(subset="DATA_COLUMN", keep="first")
    return df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", default=HERE, type=Path)
    parser.add_argument("--output", default=HERE / "Crypto_c.csv", type=Path)
    args = parser.parse_args()

    frames = [load_source(args.input_dir / name, *cols) for name, cols in SOURCES.items()]
    df = build_training_frame(frames)
    df.to_csv(args.output, index=False)

    print(f"wrote {len(df)} rows to {args.output}")
    print(pd.crosstab(df["source"], df["LABEL_COLUMN"], margins=True))


if __name__ == "__main__":
    main()
