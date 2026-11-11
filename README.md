# Crypto Sentiment BERT

Sentiment analysis (negative / neutral / positive) for cryptocurrency-related
Reddit comments, powered by a `bert-base-uncased` model fine-tuned on ~3,000
labeled comments and served behind a FastAPI endpoint.

The fine-tuned model is loaded from the Hugging Face Hub
(`baz08/crypto-sentiment-bert` by default; set `MODEL_REPO` to override).

## Run it on Google Colab

| Notebook | What it does | Runtime |
|---|---|---|
| [`01_data_pipeline`](notebooks/01_data_pipeline.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/baz08/crypto-sentiment-bert/blob/main/notebooks/01_data_pipeline.ipynb) | Cleans and merges the labeled sources into `Crypto_c.csv`, with label/length charts | CPU, < 1 min |
| [`02_train_evaluate`](notebooks/02_train_evaluate.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/baz08/crypto-sentiment-bert/blob/main/notebooks/02_train_evaluate.ipynb) | Fine-tunes BERT, compares it with majority-class and TF-IDF baselines, error analysis, optional push to the Hub | T4 GPU, a few minutes |

Each notebook clones this repo for the data, so they run as-is with no setup.

## Quickstart: run the API

```bash
cd deployment/api
docker build -t crypto-sentiment-bert .
docker run -p 8000:8000 crypto-sentiment-bert
```

The first run downloads the fine-tuned model from the Hugging Face Hub, so
expect it to take a minute or two before the server is ready.

Once it's up:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love bitcoin"}'
# {"sentiment":"Positive"}
```

Interactive API docs (Swagger UI) are available at
http://localhost:8000/docs, and a `GET /health` endpoint is available for
readiness checks.

| URL              | Method | Description                                  |
|-------------------|--------|-----------------------------------------------|
| `/predict`         | POST   | Run sentiment analysis on the provided text   |
| `/health`          | GET    | Liveness/readiness check                      |

## Project structure

```
├── deployment
│   ├── api                  - FastAPI service
│   │   ├── main.py            - API routes
│   │   ├── ML
│   │   │   ├── predmodel.py   - loads the fine-tuned BERT model from the Hub
│   │   │   └── textprep.py    - input cleaning (copy of reddit/textprep.py)
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── bert_training         - model training
│   │   ├── berttest.py        - train/evaluate the classifier
│   │   ├── requirements.txt
│   │   └── WRITEUP.md         - training methodology and results
│   ├── reddit                - data collection & preprocessing
│   │   ├── redditpushshift.py - how comments were scraped (Pushshift; no longer works)
│   │   ├── *.csv              - frozen labeled data snapshot (2022)
│   │   ├── textprep.py        - light cleaning used for training AND inference
│   │   ├── cleaner.py         - bag-of-words normalization (for non-BERT baselines)
│   │   ├── merge_clean.py     - merge the labeled sources into Crypto_c.csv
│   │   └── requirements.txt
│   └── docs
│       └── UCSD ML Capstone.pdf
├── notebooks                 - Colab versions of the pipeline and training
└── tests                     - unit tests for the API and data pipeline
```

## Training

The API loads an already fine-tuned model from the Hub, so training is only
needed if you want to reproduce or improve it. See
[`deployment/bert_training/WRITEUP.md`](deployment/bert_training/WRITEUP.md)
for the full methodology, and run it with:

```bash
cd deployment/bert_training
pip install -r requirements.txt
python berttest.py --data ../reddit/Crypto_c.csv --output-dir ./model --push-to-hub baz08/crypto-sentiment-bert
```

Training uses PyTorch (`transformers` v5 dropped TensorFlow support). The
script holds out a shuffled, label-stratified 20% test split (fixed
seed), reports accuracy and macro-F1 alongside a majority-class baseline and a
TF-IDF + logistic regression baseline,
and writes everything to `metrics.json`.

### Data

`deployment/reddit/Crypto_c.csv` (2,942 labeled comments/posts) is checked
in and is built from four labeled sources by:

```bash
cd deployment/reddit
python merge_clean.py
```

which also prints the label counts per source. The comments were originally
collected from r/CryptoCurrency in 2022 with `redditpushshift.py`; Reddit
shut off public Pushshift access in 2023, so the CSVs in that folder are a
frozen snapshot rather than something that can be re-scraped.

Training and the API clean text with the same function
(`textprep.clean_for_model`), which only removes URLs, HTML entities, and
encoding artifacts: stopwords, punctuation and casing are left in, since
BERT uses them.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Tests mock the BERT model itself (it's large and network-dependent), so they
run in seconds without downloading any weights.

## License

[MIT](LICENSE)
