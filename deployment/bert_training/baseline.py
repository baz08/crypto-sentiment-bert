"""TF-IDF + logistic regression baseline for crypto Reddit sentiment.

Fine-tuning BERT is expensive and needs a Hugging Face Hub download; this
baseline needs neither, so it's what to reach for first to sanity-check a
change to the data pipeline, or to have a number to compare BERT against
("did fine-tuning actually buy us anything over a linear model?").

Uses the same held-out split as berttest.py (see load_dataset()) so the two
are comparable on the same test rows.

Usage:
    python baseline.py --data ../reddit/Crypto_com.csv
"""

import argparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from berttest import evaluate, load_dataset


def build_baseline():
    return make_pipeline(
        TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="../reddit/Crypto_com.csv", help="Labeled CSV from reddit/merge_clean.py")
    parser.add_argument("--text-column", default="DATA_COLUMN")
    parser.add_argument("--label-column", default="LABEL_COLUMN")
    parser.add_argument("--test-size", type=int, default=500, help="Number of rows held out for evaluation")
    return parser.parse_args()


def main():
    args = parse_args()

    train_df, test_df = load_dataset(args.data, args.text_column, args.label_column, args.test_size)
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[args.label_column] = train_df[args.label_column].astype(int)
    test_df[args.label_column] = test_df[args.label_column].astype(int)

    model = build_baseline()
    model.fit(train_df[args.text_column], train_df[args.label_column])
    predictions = model.predict(test_df[args.text_column])

    evaluate(test_df[args.label_column], predictions)


if __name__ == "__main__":
    main()
