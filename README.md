# Crypto Sentiment BERT

Sentiment analysis (negative / neutral / positive) for cryptocurrency-related
Reddit comments, powered by a `bert-base-uncased` model fine-tuned on ~3,000
labeled comments and served behind a FastAPI endpoint.

The fine-tuned model is published at
[`baz08/crypto-Bert-test`](https://huggingface.co/baz08/crypto-Bert-test) on
the Hugging Face Hub.

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
│   │   │   └── predmodel.py   - loads the fine-tuned BERT model from the Hub
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── bert_training         - model training
│   │   ├── berttest.py        - train/evaluate the classifier
│   │   ├── requirements.txt
│   │   └── WRITEUP.md         - training methodology and results
│   ├── reddit                - data collection & preprocessing
│   │   ├── redditpushshift.py - scrape comments from r/CryptoCurrency
│   │   ├── cleaner.py         - text normalization utilities
│   │   ├── merge_clean.py     - merge + label the training CSV
│   │   └── requirements.txt
│   └── docs
│       └── UCSD ML Capstone.pdf
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
python berttest.py --data ../reddit/Crypto_c.csv --output-dir ./model --push-to-hub baz08/crypto-Bert-test
```

`Crypto_c.csv` is produced by the data pipeline in `deployment/reddit`
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
