import os

from ML.textprep import clean_for_model

# Hugging Face Hub repo holding the fine-tuned model. Override with the
# MODEL_REPO environment variable (e.g. `docker run -e MODEL_REPO=...`).
HUB_REPO = os.environ.get("MODEL_REPO", "baz08/crypto-sentiment-bert")
SENTIMENTS = ["Negative", "Neutral", "Positive"]
# Keep in sync with MAX_LENGTH in deployment/bert_training/berttest.py.
MAX_LENGTH = 256


class Model:
    def __init__(self):
        # Imported lazily so this module (and anything that imports it, like
        # main.py for tests) doesn't require torch/transformers or a
        # network call unless a Model is actually instantiated.
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(HUB_REPO)
        self.bert = AutoModelForSequenceClassification.from_pretrained(HUB_REPO)
        self.bert.eval()

    def predict(self, text):
        import torch

        # Same cleaning as the training data (see deployment/reddit/merge_clean.py),
        # so the model sees the same kind of text it was trained on.
        text = clean_for_model(text)
        encoded_text = self.tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        with torch.no_grad():
            logits = self.bert(**encoded_text).logits
        return SENTIMENTS[int(logits.argmax(dim=-1)[0])]


_model = None


def get_model():
    global _model
    if _model is None:
        _model = Model()
    return _model
