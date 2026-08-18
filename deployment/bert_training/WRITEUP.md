# Crypto Sentiment BERT — Training Write-up

## Overview

This project fine-tunes `bert-base-uncased` into a 3-class sentiment
classifier (**negative / neutral / positive**) for cryptocurrency-related
Reddit comments (BTC, ETH, and general crypto discussion). The fine-tuned
model is published to the Hugging Face Hub at
[`baz08/crypto-Bert-test`](https://huggingface.co/baz08/crypto-Bert-test)
and served behind a FastAPI endpoint (see `deployment/api`).

## Data pipeline

1. **Collection** — `deployment/reddit/redditpushshift.py` pulls BTC/ETH
   comments from r/CryptoCurrency via the Pushshift API.
2. **Cleaning** — `deployment/reddit/cleaner.py` expands contractions,
   lowercases, strips special characters/URLs, and removes stopwords
   (keeping "no"/"not", since they carry sentiment).
3. **Merging & labeling** — `deployment/reddit/merge_clean.py` combines the
   scraped comments with the labeled sets (`sentiment.csv`,
   `sentiment_labels.csv`), normalizes the text, and writes a single
   `Crypto_c.csv` with a `DATA_COLUMN` (text) and `LABEL_COLUMN`
   (0 = negative, 1 = neutral, 2 = positive).

Training used ~3,000 labeled examples, split into a training set and a
held-out evaluation set (500 rows by default).

## Model

- **Base**: `bert-base-uncased` via `TFAutoModelForSequenceClassification`
  (3-way classification head).
- **Tokenization**: max length 230 tokens, truncated/padded.
- **Optimizer**: Adam, lr `2e-5`, `epsilon=1e-08`, gradient clipping at
  `clipnorm=1.0` — standard low-LR BERT fine-tuning settings to avoid
  catastrophic forgetting of the pretrained weights.
- **Loss**: sparse categorical cross-entropy.
- **Epochs**: 2, batch size 16.
- **Evaluation**: accuracy, confusion matrix, and per-class
  precision/recall/F1 via `sklearn.metrics` on the held-out split.

Run it with:

```bash
pip install transformers tensorflow datasets scikit-learn pandas
python berttest.py --data ../reddit/Crypto_c.csv --output-dir ./model --push-to-hub baz08/crypto-Bert-test
```

## Cleanup notes (this pass)

The original `berttest.ipynb` was an exploratory notebook and has been
replaced by `berttest.py`, a single reusable, argument-driven script. What
changed:

- **Removed the legacy `InputExample`/`InputFeatures` pipeline.** It
  tokenized one example at a time via `tokenizer.encode_plus` in a Python
  loop — slow and effectively deprecated in `transformers`. Replaced with a
  single batched call to the tokenizer plus
  `tf.data.Dataset.from_tensor_slices`, which is simpler and does the same
  job.
- **Fixed a silent 4x-training bug.** The notebook called
  `train_data.shuffle(500).batch(16).repeat(2)` *and* passed `epochs=2` to
  `model.fit`. Since `model.fit` already iterates the dataset once per
  epoch, the extra `.repeat(2)` meant the model actually trained for 4
  effective passes over the data instead of 2. Removed the redundant
  `.repeat()`.
- **Removed dead/unreachable code**, e.g. a self-call to
  `convert_data_to_examples` placed after its own `return` statement, and
  several commented-out, never-run experiments.
- **Removed the hardcoded personal path**
  (`C:/Users/Barrett/mec-mini-projects-master/api/ML`) used to save/reload
  the model. Saving/loading is now controlled by `--output-dir`.
- **Consolidated five overlapping, partly-broken prediction/evaluation
  cells** (including one that referenced a `predicted` column that was
  never created, and another that evaluated a `train_pred` column on a
  DataFrame slice that never had it) into two functions: `predict()`
  (batched inference) and `evaluate()` (metrics on the held-out set).
  Renamed the leftover `gme_df` variable (copy-pasted from an unrelated
  GameStop sentiment project) to `train_df`/`test_df`.
- **Fixed a mislabeled sentiment string.** `deployment/api/ML/predmodel.py`
  returned `"Postive"` instead of `"Positive"` from the live API — one
  character off, but visible in every positive prediction. Fixed.

## Known follow-ups

- No metrics from an actual training run are checked into the repo; run
  `berttest.py` to reproduce accuracy/F1 numbers for the current model
  version. (Deliberately not run as part of this cleanup pass — a real
  fine-tuning run needs a GPU and a multi-GB TensorFlow/transformers
  install to be worth the time.)
- A fuller project report already exists at
  `deployment/docs/UCSD ML Capstone.pdf`, if a formal write-up is needed
  beyond this training summary.

## Repo-wide cleanup (later pass)

Beyond the training script itself, a follow-up pass fixed the rest of the
repo for portfolio readiness:

- `deployment/reddit/cleaner.py` imported a nonexistent `contractions`
  module attribute, so the data pipeline (`merge_clean.py`) could never
  actually run — switched to the real `contractions` pip package.
- `deployment/reddit/redditpushshift.py` had a copy-paste bug where
  `ETH_comments.csv` was silently populated with duplicated BTC data.
- Split the single API `requirements.txt` (which mixed API, training, and
  data-pipeline dependencies, several unused) into
  `deployment/api/requirements.txt`, `deployment/bert_training/requirements.txt`,
  and `deployment/reddit/requirements.txt`, and pinned `transformers`/
  `tensorflow` (previously unpinned).
- `predmodel.py`'s `Model` was instantiated eagerly at import time,
  meaning importing the module at all — including for tests — downloaded
  the full BERT model. It's now lazy (built on first request) and loads
  its tokenizer from the fine-tuned Hub repo instead of the base
  `bert-base-uncased` tokenizer.
- Added `tests/` (mocking the model, so the suite runs in seconds) and a
  GitHub Actions CI workflow.
