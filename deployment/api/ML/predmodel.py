HUB_REPO = "baz08/crypto-Bert-test"
SENTIMENTS = ["Negative", "Neutral", "Positive"]


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

        encoded_text = self.tokenizer(
            text, padding=True, truncation=True, max_length=512, return_tensors="tf"
        )
        output = self.bert(encoded_text)
        label = tf.argmax(tf.nn.softmax(output[0], axis=-1), axis=1).numpy()[0]
        return SENTIMENTS[label]


_model = None


def get_model():
    global _model
    if _model is None:
        _model = Model()
    return _model
