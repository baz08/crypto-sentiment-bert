from ML.textprep import clean_for_model

HUB_REPO = "baz08/crypto-Bert-test"
SENTIMENTS = ["Negative", "Neutral", "Positive"]
# Keep in sync with MAX_LENGTH in deployment/bert_training/berttest.py.
MAX_LENGTH = 256


class Model:
    def __init__(self):
        # Imported lazily so this module (and anything that imports it, like
        # main.py for tests) doesn't require transformers/tensorflow or a
        # network call unless a Model is actually instantiated.
        from transformers import AutoTokenizer, TFBertForSequenceClassification

        self.tokenizer = AutoTokenizer.from_pretrained(HUB_REPO)
        self.bert = TFBertForSequenceClassification.from_pretrained(HUB_REPO)

    def predict(self, text):
        import tensorflow as tf

        # Same cleaning as the training data (see deployment/reddit/merge_clean.py),
        # so the model sees the same kind of text it was trained on.
        text = clean_for_model(text)
        encoded_text = self.tokenizer(
            text, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="tf"
        )
        output = self.bert(encoded_text)
        label = tf.argmax(output.logits, axis=-1).numpy()[0]
        return SENTIMENTS[label]


_model = None


def get_model():
    global _model
    if _model is None:
        _model = Model()
    return _model
