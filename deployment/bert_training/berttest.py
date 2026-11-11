"""Fine-tune a BERT sequence classifier for crypto Reddit comment sentiment.

Trains a 3-class (negative / neutral / positive) classifier on top of
``bert-base-uncased`` using the labeled CSV produced by
``deployment/reddit/merge_clean.py``, evaluates it on a held-out split, and
optionally saves it locally and/or pushes it to the Hugging Face Hub (the
deployed API at ``deployment/api`` loads the model from there).

Usage:
    python berttest.py --data ../reddit/Crypto_c.csv
    python berttest.py --data ../reddit/Crypto_c.csv --output-dir ./model --push-to-hub baz08/crypto-Bert-test
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

LABELS = ["Negative", "Neutral", "Positive"]
BASE_MODEL = "bert-base-uncased"
# Keep in sync with MAX_LENGTH in deployment/api/ML/predmodel.py so the
# deployed model sees inputs truncated the same way it was trained.
MAX_LENGTH = 256
SEED = 42


def load_dataset(csv_path, text_column, label_column, test_size, seed=SEED):
    """Shuffled, label-stratified train/test split.

    ``test_size`` may be a fraction (0.2) or a row count (500). The input CSV
    is ordered by source file, so taking the last N rows (as the original
    notebook did) evaluated on a single source with a very different label
    mix than the training data.
    """
    df = pd.read_csv(csv_path).dropna(subset=[text_column, label_column]).reset_index(drop=True)
    df[label_column] = df[label_column].astype(int)
    if test_size >= 1:
        test_size = int(test_size)
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, shuffle=True, stratify=df[label_column]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_model():
    # Imported lazily so this module stays importable (and load_dataset()/
    # evaluate() stay unit-testable) without pulling in transformers/tensorflow.
    from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = TFAutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=len(LABELS))
    return model, tokenizer


def encode_dataset(tokenizer, texts, labels, max_length=MAX_LENGTH):
    import tensorflow as tf

    encodings = tokenizer(
        list(texts), max_length=max_length, padding=True, truncation=True, return_tensors="tf"
    )
    return tf.data.Dataset.from_tensor_slices((dict(encodings), list(labels)))


def train(model, tokenizer, train_df, test_df, text_column, label_column, epochs, batch_size):
    import tensorflow as tf

    train_data = encode_dataset(tokenizer, train_df[text_column], train_df[label_column])
    train_data = train_data.shuffle(500).batch(batch_size)

    val_data = encode_dataset(tokenizer, test_df[text_column], test_df[label_column]).batch(batch_size)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08, clipnorm=1.0),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy("accuracy")],
    )
    model.fit(train_data, epochs=epochs, validation_data=val_data)
    return model


def predict(model, tokenizer, texts, batch_size=100, max_length=MAX_LENGTH):
    import tensorflow as tf

    predictions = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        encodings = tokenizer(
            batch, max_length=max_length, padding=True, truncation=True, return_tensors="tf"
        )
        logits = model(encodings).logits
        predictions.extend(tf.argmax(logits, axis=-1).numpy())
    return predictions


def majority_baseline(y_train, y_true):
    """Predict the most common training label for every test row."""
    majority = pd.Series(list(y_train)).mode()[0]
    return [majority] * len(y_true)


def compute_metrics(y_true, y_pred):
    y_true, y_pred = list(y_true), list(y_pred)
    return {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "macro_f1": metrics.f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0),
        "per_class_f1": dict(
            zip(LABELS, metrics.f1_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0).tolist())
        ),
        "confusion_matrix": metrics.confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def evaluate(y_true, y_pred, y_train=None):
    """Print model metrics (and a majority-class baseline, if y_train is given) and return them."""
    results = {"model": compute_metrics(y_true, y_pred)}
    if y_train is not None:
        results["majority_baseline"] = compute_metrics(y_true, majority_baseline(y_train, y_true))

    for name, scores in results.items():
        print(f"{name}: accuracy {scores['accuracy']:.4f}  macro-F1 {scores['macro_f1']:.4f}")
    print(metrics.confusion_matrix(list(y_true), list(y_pred), labels=[0, 1, 2]))
    print(
        metrics.classification_report(
            list(y_true), list(y_pred), labels=[0, 1, 2], target_names=LABELS, zero_division=0
        )
    )
    return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="../reddit/Crypto_c.csv", help="Labeled CSV from reddit/merge_clean.py")
    parser.add_argument("--text-column", default="DATA_COLUMN")
    parser.add_argument("--label-column", default="LABEL_COLUMN")
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Held-out fraction (e.g. 0.2) or row count (e.g. 500)"
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", default=None, help="If set, save the fine-tuned model here")
    parser.add_argument("--push-to-hub", default=None, help="If set, push to this Hugging Face Hub repo id")
    parser.add_argument("--metrics-out", default="metrics.json", help="Where to write evaluation metrics as JSON")
    return parser.parse_args()


def main():
    args = parse_args()

    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.seed)

    train_df, test_df = load_dataset(args.data, args.text_column, args.label_column, args.test_size, args.seed)
    print(f"train: {len(train_df)} rows, test: {len(test_df)} rows")
    model, tokenizer = build_model()
    model = train(
        model, tokenizer, train_df, test_df, args.text_column, args.label_column, args.epochs, args.batch_size
    )

    test_predictions = predict(model, tokenizer, test_df[args.text_column])
    results = evaluate(test_df[args.label_column], test_predictions, y_train=train_df[args.label_column])
    results["config"] = {
        "base_model": BASE_MODEL,
        "max_length": MAX_LENGTH,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
    }
    Path(args.metrics_out).write_text(json.dumps(results, indent=2))
    print(f"wrote metrics to {args.metrics_out}")

    if args.output_dir:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        tokenizer.push_to_hub(args.push_to_hub)
        model.push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    main()
