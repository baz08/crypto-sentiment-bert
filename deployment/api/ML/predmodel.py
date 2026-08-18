from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf


path = "baz08/crypto-Bert-test"

sentiments = ['Negative', 'Neutral', 'Positive']


class Model:
    def __init__(self):

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.bert = TFBertForSequenceClassification.from_pretrained(path)

    def predict(self, text):
        encoded_text = self.tokenizer(text, padding=True,
                                      truncation=True, max_length=512, return_tensors='tf')
        output = self.bert(encoded_text)
        label = tf.argmax(tf.nn.softmax(output[0], axis=-1), axis=1).numpy()[0]
        return sentiments[label]


model = Model()


def get_model():
    return model
