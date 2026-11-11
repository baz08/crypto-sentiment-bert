"""Fine-tune a BERT sequence classifier for crypto Reddit comment sentiment.

Trains a 3-class (negative / neutral / positive) classifier on top of
``bert-base-uncased`` (PyTorch) using the labeled CSV produced by
``deployment/reddit/merge_clean.py``, evaluates it on a held-out split
against two baselines, and optionally saves it locally and/or pushes it to
the Hugging Face Hub (the deployed API at ``deployment/api`` loads the model
from there).

The same steps, with plots and error analysis, are in
``notebooks/02_train_evaluate.ipynb`` for running on Google Colab.

Usage:
    python berttest.py --data ../reddit/Crypto_c.csv
    python berttest.py --data ../reddit/Crypto_c.csv --output-dir ./model --push-to-hub baz08/crypto-sentiment-bert
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

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


# --- baselines ---------------------------------------------------------------


def majority_baseline(y_train, y_true):
    """Predict the most common training label for every test row."""
    majority = pd.Series(list(y_train)).mode()[0]
    return [majority] * len(y_true)


def tfidf_baseline(train_texts, train_labels, test_texts, seed=SEED):
    """TF-IDF + logistic regression: the 'is BERT worth it?' comparison."""
    clf = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
    )
    clf.fit(list(train_texts), list(train_labels))
    return clf.predict(list(test_texts)).tolist()


# --- BERT --------------------------------------------------------------------


def build_model():
    # Imported lazily so this module stays importable (and load_dataset()/
    # evaluate() stay unit-testable) without pulling in torch/transformers.
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
    )
    return model, tokenizer


def make_loader(tokenizer, texts, labels, batch_size, shuffle, seed=SEED):
    import torch
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    encodings = tokenizer(list(texts), truncation=True, max_length=MAX_LENGTH)
    items = [
        {**{key: values[i] for key, values in encodings.items()}, "labels": int(label)}
        for i, label in enumerate(labels)
    ]
    return DataLoader(
        items,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=DataCollatorWithPadding(tokenizer),  # pads each batch to its longest row
        generator=torch.Generator().manual_seed(seed),
    )


def train(model, tokenizer, train_df, test_df, text_column, label_column, epochs, batch_size, lr=2e-5, seed=SEED):
    import torch
    from transformers import get_linear_schedule_with_warmup

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_loader = make_loader(tokenizer, train_df[text_column], train_df[label_column], batch_size, True, seed)
    total_steps = len(train_loader) * epochs

    # Standard BERT fine-tuning recipe: small LR, linear warmup/decay, and
    # gradient clipping to avoid wrecking the pretrained weights.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running_loss += loss.item()

        val_pred = predict(model, tokenizer, test_df[text_column])
        val_acc = metrics.accuracy_score(test_df[label_column], val_pred)
        print(f"epoch {epoch}: train loss {running_loss / len(train_loader):.4f}  val accuracy {val_acc:.4f}")
    return model


def predict(model, tokenizer, texts, batch_size=64, max_length=MAX_LENGTH):
    import torch

    device = next(model.parameters()).device
    model.eval()
    texts = list(texts)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            encodings = tokenizer(
                texts[start : start + batch_size],
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            predictions.extend(model(**encodings).logits.argmax(dim=-1).cpu().tolist())
    return predictions


# --- evaluation --------------------------------------------------------------


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
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output-dir", default=None, help="If set, save the fine-tuned model here")
    parser.add_argument("--push-to-hub", default=None, help="If set, push to this Hugging Face Hub repo id")
    parser.add_argument("--metrics-out", default="metrics.json", help="Where to write evaluation metrics as JSON")
    return parser.parse_args()


def main():
    args = parse_args()

    from transformers import set_seed

    set_seed(args.seed)

    train_df, test_df = load_dataset(args.data, args.text_column, args.label_column, args.test_size, args.seed)
    print(f"train: {len(train_df)} rows, test: {len(test_df)} rows")

    tfidf_pred = tfidf_baseline(train_df[args.text_column], train_df[args.label_column], test_df[args.text_column])

    model, tokenizer = build_model()
    model = train(
        model,
        tokenizer,
        train_df,
        test_df,
        args.text_column,
        args.label_column,
        args.epochs,
        args.batch_size,
        args.lr,
        args.seed,
    )

    test_predictions = predict(model, tokenizer, test_df[args.text_column])
    results = evaluate(test_df[args.label_column], test_predictions, y_train=train_df[args.label_column])
    results["tfidf_logreg"] = compute_metrics(test_df[args.label_column], tfidf_pred)
    print(f"tfidf_logreg: accuracy {results['tfidf_logreg']['accuracy']:.4f}  macro-F1 {results['tfidf_logreg']['macro_f1']:.4f}")
    results["config"] = {
        "base_model": BASE_MODEL,
        "max_length": MAX_LENGTH,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
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
