# Crypto Sentiment BERT

![CI](https://github.com/baz08/crypto-sentiment-bert/actions/workflows/ci.yml/badge.svg)

Sentiment analysis (negative / neutral / positive) for cryptocurrency-related
Reddit comments, powered by a `bert-base-uncased` model fine-tuned on ~3,000
labeled comments and served behind a FastAPI endpoint.

The fine-tuned model is published at
[`baz08/crypto-Bert-test`](https://huggingface.co/baz08/crypto-Bert-test) on
the Hugging Face Hub.

## Try it

**Demo (Gradio):**

```bash
pip install -r demo/requirements.txt
python demo/app.py
```

Also deployable as-is to a Hugging Face Space — point the Space's app file
at `demo/app.py`.

**API (Docker):**

```bash
cd deployment/api
docker build -t crypto-sentiment-bert .
docker run -p 8000:8000 crypto-sentiment-bert
```

The first run downloads the fine-tuned model from the Hugging Face Hub, so
expect it to take a minute or two before the server is ready.

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

## Results

The dataset (2,994 labeled Reddit comments) is imbalanced — nearly half
neutral:

![Class distribution](deployment/bert_training/assets/class_distribution.png)

As a reference point, a TF-IDF + logistic regression baseline
(`deployment/bert_training/baseline.py`) gets **54.2% accuracy** (macro F1
0.44) on a 500-row held-out split, and is weakest exactly where it matters —
telling negative from neutral:

![Confusion matrix](deployment/bert_training/assets/confusion_matrix.png)

Full metrics table, methodology, and how BERT compares:
[`deployment/bert_training/WRITEUP.md`](deployment/bert_training/WRITEUP.md).

## Project structure

```
├── demo
│   ├── app.py                - Gradio demo
│   └── requirements.txt
├── deployment
│   ├── api                  - FastAPI service
│   │   ├── main.py            - API routes
│   │   ├── ML
│   │   │   └── predmodel.py   - loads the fine-tuned BERT model from the Hub
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── bert_training         - model training
│   │   ├── berttest.py        - train/evaluate the BERT classifier
│   │   ├── baseline.py        - TF-IDF + logistic regression reference point
│   │   ├── visualize.py       - regenerates the PNGs used above
│   │   ├── assets/            - class distribution & confusion matrix PNGs
│   │   ├── requirements.txt
│   │   └── WRITEUP.md         - training methodology and results
│   ├── reddit                - data collection & preprocessing
│   │   ├── redditpushshift.py - scrape comments from r/CryptoCurrency
│   │   ├── cleaner.py         - text normalization utilities
│   │   ├── merge_clean.py     - merge + label the training CSV
│   │   └── requirements.txt
│   └── docs
│       └── UCSD ML Capstone.pdf
└── tests                     - unit tests for the API, demo, and data pipeline
```

## Training

The API loads an already fine-tuned model from the Hub, so training is only
needed if you want to reproduce or improve it. See
[`deployment/bert_training/WRITEUP.md`](deployment/bert_training/WRITEUP.md)
for the full methodology, and run it with:

```bash
cd deployment/bert_training
pip install -r requirements.txt
python berttest.py --data ../reddit/Crypto_com.csv --output-dir ./model --push-to-hub baz08/crypto-Bert-test
```

`Crypto_com.csv` is produced by the data pipeline in `deployment/reddit`
(`redditpushshift.py` → `cleaner.py` → `merge_clean.py`); see that folder's
`requirements.txt` to run it.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Tests mock the BERT model itself (it's large and network-dependent), so they
run in seconds without downloading any weights.

## License

[MIT](LICENSE)
