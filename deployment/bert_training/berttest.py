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
    python berttest.py --data ../reddit/Crypto_c.csv --search          # tune first, then train
    python berttest.py --data ../reddit/Crypto_c.csv --output-dir ./model --push-to-hub baz08/crypto-sentiment-bert
"""

import argparse
import itertools
import json
import random
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

# Training hyperparameters used when no search is run.
DEFAULT_CONFIG = {"lr": 2e-5, "batch_size": 16, "epochs": 2, "weight_decay": 0.01, "warmup_ratio": 0.1}

# Default search space. Epochs are not in the grid: every trial trains for
# max_epochs and is scored after each epoch, so epoch count is searched for free.
SEARCH_GRID = {"lr": [1e-5, 2e-5, 3e-5, 5e-5], "batch_size": [16, 32]}


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


def fine_tune(train_texts, train_labels, eval_texts=None, eval_labels=None, config=None, seed=SEED, verbose=True):
    """Fine-tune a fresh BERT model with ``config`` (missing keys fall back to DEFAULT_CONFIG).

    If eval texts/labels are given, the model is scored on them after every
    epoch. Returns ``(model, tokenizer, history)``, where ``history`` has one
    dict per epoch (train_loss, plus accuracy/macro_f1 when evaluating).
    """
    import torch
    from transformers import get_linear_schedule_with_warmup, set_seed

    config = {**DEFAULT_CONFIG, **(config or {})}
    set_seed(seed)

    model, tokenizer = build_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_loader = make_loader(tokenizer, train_texts, train_labels, int(config["batch_size"]), True, seed)
    total_steps = len(train_loader) * int(config["epochs"])

    # Standard BERT fine-tuning recipe: small LR, linear warmup/decay, and
    # gradient clipping to avoid wrecking the pretrained weights.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]), eps=1e-8, weight_decay=float(config["weight_decay"])
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(float(config["warmup_ratio"]) * total_steps), total_steps
    )

    history = []
    for epoch in range(1, int(config["epochs"]) + 1):
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

        record = {"epoch": epoch, "train_loss": running_loss / len(train_loader)}
        if eval_texts is not None:
            scores = compute_metrics(eval_labels, predict(model, tokenizer, eval_texts))
            record.update(accuracy=scores["accuracy"], macro_f1=scores["macro_f1"])
        history.append(record)
        if verbose:
            print({key: round(value, 4) if isinstance(value, float) else value for key, value in record.items()})
    return model, tokenizer, history


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


# --- hyperparameter search ---------------------------------------------------


def expand_grid(grid, n_trials=None, seed=SEED):
    """All combinations of ``grid`` (dict of lists), or a random sample of ``n_trials`` of them."""
    keys = list(grid)
    combos = [dict(zip(keys, values)) for values in itertools.product(*(grid[key] for key in keys))]
    if n_trials is not None and n_trials < len(combos):
        combos = random.Random(seed).sample(combos, n_trials)
    return combos


def _bert_trial(config, fit_df, val_df, text_column, label_column, seed):
    """Train one config and return its per-epoch validation history."""
    model, _, history = fine_tune(
        fit_df[text_column],
        fit_df[label_column],
        val_df[text_column],
        val_df[label_column],
        config=config,
        seed=seed,
        verbose=False,
    )
    del model
    try:
        import torch

        torch.cuda.empty_cache()  # free GPU memory before the next trial
    except ImportError:
        pass
    return history


def search_hyperparameters(
    train_df,
    text_column,
    label_column,
    grid=None,
    max_epochs=4,
    val_size=0.15,
    n_trials=None,
    metric="macro_f1",
    seed=SEED,
    trial_fn=None,
):
    """Find the best fine-tuning config and return ``(best_config, results)``.

    - Splits ``train_df`` into a fit set and a stratified validation set
      (``val_size``). **The test set is never touched**, so the final test
      score stays an honest estimate.
    - Trains one model per combination in ``grid`` (or a random sample of
      ``n_trials``), each for ``max_epochs``, scoring ``metric`` on the
      validation set after every epoch.
    - Picks the highest-scoring (combination, epoch); ties go to fewer epochs.

    ``best_config`` is a full config (DEFAULT_CONFIG overridden by the winner,
    including ``epochs``) ready to pass to ``fine_tune``. ``results`` is a
    DataFrame with one row per (combination, epoch), best first.

    Note: the LR schedule decays over ``max_epochs``, so "epoch 2 of a
    4-epoch trial" is a close but not exact stand-in for a 2-epoch run.

    ``trial_fn(config, fit_df, val_df, text_column, label_column, seed)``
    can be swapped out (e.g. in tests); it must return per-epoch dicts with
    ``epoch`` and ``metric``.
    """
    grid = grid or SEARCH_GRID
    if "epochs" in grid:
        raise ValueError("Don't put 'epochs' in the grid; set max_epochs instead (every epoch is scored).")
    trial_fn = trial_fn or _bert_trial

    fit_df, val_df = train_test_split(
        train_df, test_size=val_size, random_state=seed, shuffle=True, stratify=train_df[label_column]
    )
    combos = expand_grid(grid, n_trials, seed)
    print(f"searching {len(combos)} configs x {max_epochs} epochs on {len(fit_df)} fit / {len(val_df)} validation rows")

    rows = []
    for i, combo in enumerate(combos, 1):
        config = {**DEFAULT_CONFIG, **combo, "epochs": max_epochs}
        history = trial_fn(config, fit_df, val_df, text_column, label_column, seed)
        for record in history:
            rows.append({**combo, "epochs": record["epoch"], "accuracy": record.get("accuracy"), metric: record[metric]})
        best_epoch = max(history, key=lambda record: record[metric])
        print(f"[{i}/{len(combos)}] {combo}: best {metric} {best_epoch[metric]:.4f} at epoch {best_epoch['epoch']}")

    results = pd.DataFrame(rows).sort_values([metric, "epochs"], ascending=[False, True]).reset_index(drop=True)
    # Read values column by column (a whole row would upcast ints like batch_size to floats).
    best_config = {**DEFAULT_CONFIG, "epochs": int(results["epochs"].iloc[0])}
    for key in grid:
        value = results[key].iloc[0]
        best_config[key] = value.item() if hasattr(value, "item") else value  # numpy -> plain Python (JSON-safe)
    print(f"best config: {best_config} (validation {metric} {results[metric].iloc[0]:.4f})")
    return best_config, results


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
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument(
        "--search", action="store_true", help="Run a hyperparameter search first and train with the best config"
    )
    parser.add_argument("--max-epochs", type=int, default=4, help="Epochs per search trial")
    parser.add_argument("--n-trials", type=int, default=None, help="Randomly sample this many grid configs")
    parser.add_argument("--output-dir", default=None, help="If set, save the fine-tuned model here")
    parser.add_argument("--push-to-hub", default=None, help="If set, push to this Hugging Face Hub repo id")
    parser.add_argument("--metrics-out", default="metrics.json", help="Where to write evaluation metrics as JSON")
    return parser.parse_args()


def main():
    args = parse_args()

    train_df, test_df = load_dataset(args.data, args.text_column, args.label_column, args.test_size, args.seed)
    print(f"train: {len(train_df)} rows, test: {len(test_df)} rows")

    search_results = None
    if args.search:
        config, search_results = search_hyperparameters(
            train_df, args.text_column, args.label_column, max_epochs=args.max_epochs, n_trials=args.n_trials, seed=args.seed
        )
        search_results.to_csv("search_results.csv", index=False)
    else:
        config = {**DEFAULT_CONFIG, "lr": args.lr, "batch_size": args.batch_size, "epochs": args.epochs}

    tfidf_pred = tfidf_baseline(train_df[args.text_column], train_df[args.label_column], test_df[args.text_column])

    # Final model: train on the full training set with the chosen config, then
    # evaluate on the test set exactly once.
    model, tokenizer, history = fine_tune(
        train_df[args.text_column], train_df[args.label_column], config=config, seed=args.seed
    )
    test_predictions = predict(model, tokenizer, test_df[args.text_column])
    results = evaluate(test_df[args.label_column], test_predictions, y_train=train_df[args.label_column])
    results["tfidf_logreg"] = compute_metrics(test_df[args.label_column], tfidf_pred)
    print(f"tfidf_logreg: accuracy {results['tfidf_logreg']['accuracy']:.4f}  macro-F1 {results['tfidf_logreg']['macro_f1']:.4f}")
    results["config"] = {
        "base_model": BASE_MODEL,
        "max_length": MAX_LENGTH,
        **config,
        "seed": args.seed,
        "searched": args.search,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
    }
    results["history"] = history
    if search_results is not None:
        results["search_top5"] = search_results.head(5).to_dict(orient="records")
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
