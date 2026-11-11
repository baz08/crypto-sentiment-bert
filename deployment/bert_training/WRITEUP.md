# Crypto Sentiment BERT — Training Write-up

## Overview

This project fine-tunes `bert-base-uncased` into a 3-class sentiment
classifier (**negative / neutral / positive**) for cryptocurrency-related
Reddit comments (BTC, ETH, and general crypto discussion). The fine-tuned
model is published to the Hugging Face Hub (`baz08/crypto-sentiment-bert`)
and served behind a FastAPI endpoint (see `deployment/api`).

## Data pipeline

1. **Collection** — `deployment/reddit/redditpushshift.py` pulls BTC/ETH
   comments from r/CryptoCurrency via the Pushshift API.
2. **Cleaning** — `deployment/reddit/cleaner.py` expands contractions,
   lowercases, strips special characters/URLs, and removes stopwords
   (keeping "no"/"not", since they carry sentiment).
3. **Merging & labeling** — `deployment/reddit/merge_clean.py` combines the
   labeled BTC/ETH comments with the other labeled sets (`sentiment.csv`,
   `sentiment_labels.csv`), cleans the text with
   `textprep.clean_for_model`, drops unlabeled/empty/duplicate rows, and
   writes `Crypto_c.csv` with `DATA_COLUMN` (text), `LABEL_COLUMN`
   (0 = negative, 1 = neutral, 2 = positive) and `source`.

| source               | negative | neutral | positive | total |
|----------------------|---------:|--------:|---------:|------:|
| BTC_comments         | 194      | 432     | 209      | 835   |
| ETH_comments         | 149      | 249     | 160      | 558   |
| sentiment            | 256      | 0       | 295      | 551   |
| sentiment_labels     | 78       | 707     | 213      | 998   |
| **total**            | 677      | 1,388   | 877      | 2,942 |

Note that `sentiment.csv` has no neutral class, so its labeling scheme
differs from the other sources.

The data is split 80/20 into train/test with a shuffled, label-stratified
split (seed 42).

## Model

- **Base**: `bert-base-uncased` via `AutoModelForSequenceClassification`
  (PyTorch, 3-way classification head).
- **Tokenization**: max length 256 tokens, truncated/padded (same limit in the API).
- **Optimizer**: AdamW, lr `2e-5`, `eps=1e-08`, linear warmup (10% of
  steps) then linear decay, gradient clipping at max norm 1.0 — standard low-LR BERT fine-tuning settings to avoid
  catastrophic forgetting of the pretrained weights.
- **Loss**: sparse categorical cross-entropy.
- **Epochs**: 2, batch size 16.
- **Evaluation**: accuracy, macro-F1, confusion matrix, and per-class
  precision/recall/F1 on the held-out split, compared against a
  majority-class baseline; written to `metrics.json`.

Run it with the Colab notebook `notebooks/02_train_evaluate.ipynb` (same
code, plus plots and error analysis), or locally:

```bash
pip install -r requirements.txt
python berttest.py --data ../reddit/Crypto_c.csv --output-dir ./model --push-to-hub baz08/crypto-sentiment-bert
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

## Data & evaluation fixes (third pass)

- **Evaluation split was not random.** `load_dataset` held out the last 500
  rows. Because the merged CSV is ordered by source file, that test set came
  from a single source and was ~75% neutral, while training data was far
  more balanced. A leftover `train_pred` column in the old `Crypto_com.csv`
  matched the labels on only ~67% of those rows, below the ~75% a model
  that always predicts "neutral" would score. The split is now shuffled and
  stratified, and every run reports a majority-class baseline next to the
  model's scores.
- **Training/serving skew.** The model was trained on heavily normalized
  text (lowercased, stopwords and punctuation removed, contractions
  expanded), but the API sent raw user text to the model. Both now use the
  same light `clean_for_model` function (`deployment/reddit/textprep.py`,
  copied into `deployment/api/ML/` because the Docker build only sees that
  folder; a test fails if the copies drift). Stopword removal was dropped
  for BERT input: words like "not" and punctuation carry sentiment.
- **The pipeline didn't produce the training file.** `merge_clean.py`
  wrote `body`/`Sentiment` columns plus scraper metadata, but `berttest.py`
  reads `DATA_COLUMN`/`LABEL_COLUMN`, and the checked-in file was a stale
  `Crypto_com.csv` with old prediction columns. `merge_clean.py` now writes
  `Crypto_c.csv` in the right shape; `Crypto_com.csv` was removed.
- **Mojibake** ("isnâ€™t") in ~175 rows of `sentiment_labels.csv` is now
  repaired during cleaning, and Reddit HTML entities (`&gt;`, `&amp;#x200B;`)
  are unescaped.
- **`cleaner.py` bugs:** `[^a-zA-z...]` let `[ \ ] ^ _` and backticks through
  (now `A-Z`); URLs were removed after punctuation stripping, so they
  survived as tokens like `httpsexample com`; documents were silently
  dropped when `stopword_removal=False`.
- **`redditpushshift.py`** wrote unlabeled output over the labeled
  `BTC_comments.csv`/`ETH_comments.csv`; it now writes `*_raw.csv`, and is
  documented as historical since Pushshift access ended in 2023.
- Max token length aligned at 256 for training and inference (was 230 vs
  512).

**The Hub model must be retrained** with the new pipeline before these
changes help in production: the currently published weights were trained on
the old heavily-normalized text and the old split.

## Move to PyTorch and Colab notebooks (fourth pass)

`transformers` v5 (July 2026) removed TensorFlow support, and Colab now runs
Python 3.12 / TensorFlow 2.19, so the old pins (TF 2.15, which doesn't
support Python 3.12) couldn't be installed there. Training and the API were
moved to PyTorch:

- `berttest.py` uses `AutoModelForSequenceClassification` with a plain
  PyTorch training loop (same hyperparameters, plus linear LR warmup/decay)
  and adds a TF-IDF + logistic regression baseline.
- The API loads the model with PyTorch on CPU; the Docker image installs
  the CPU-only torch wheel and is based on `python:3.11-slim`. The Hub repo
  is configurable with `MODEL_REPO` (default `baz08/crypto-sentiment-bert`).
- `notebooks/01_data_pipeline.ipynb` and `notebooks/02_train_evaluate.ipynb`
  run the whole pipeline on Colab with no local setup.

Before the BERT run, the baselines on the stratified test split (589 rows)
are: majority class 47.2% accuracy / 0.214 macro-F1; TF-IDF + logistic
regression 56.4% accuracy / 0.555 macro-F1.

## Known follow-ups

- No metrics from an actual training run are checked into the repo yet;
  run `berttest.py` (a free Colab GPU is enough) and commit the resulting
  `metrics.json` numbers to the README. (Deliberately not run as part of this cleanup pass — a real
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
