"""Fine-tune a BERT sequence classifier for crypto Reddit comment sentiment.

Trains a 3-class (negative / neutral / positive) classifier on top of
``bert-base-uncased`` using the labeled CSV produced by
``deployment/reddit/merge_clean.py``, evaluates it on a held-out split, and
optionally saves it locally and/or pushes it to the Hugging Face Hub (the
deployed API at ``deployment/api`` loads the model from there).

Usage:
    python berttest.py --data ../reddit/Crypto_c.csv
    python berttest.py --data ../reddit/Crypto_c.csv --output-dir ./model --push-to-hub baz08/crypto-Bert-test
"""

import argparse

import pandas as pd
from sklearn import metrics

LABELS = ["Negative", "Neutral", "Positive"]
BASE_MODEL = "bert-base-uncased"
MAX_LENGTH = 230


def load_dataset(csv_path, text_column, label_column, test_size):
    df = pd.read_csv(csv_path).dropna(axis=0).reset_index(drop=True)
    train_df = df.iloc[:-test_size]
    test_df = df.iloc[-test_size:]
    return train_df, test_df


def build_model():
    # Imported lazily so this module stays importable (and load_dataset()/
    # evaluate() stay unit-testable) without pulling in transformers/tensorflow.
    from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = TFAutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=len(LABELS))
    return model, tokenizer


def encode_dataset(tokenizer, texts, labels, max_length=MAX_LENGTH):
    import tensorflow as tf

    encodings = tokenizer(
        list(texts), max_length=max_length, padding=True, truncation=True, return_tensors="tf"
    )
    return tf.data.Dataset.from_tensor_slices((dict(encodings), list(labels)))


def train(model, tokenizer, train_df, test_df, text_column, label_column, epochs, batch_size):
    import tensorflow as tf

    train_data = encode_dataset(tokenizer, train_df[text_column], train_df[label_column])
    train_data = train_data.shuffle(500).batch(batch_size)

    val_data = encode_dataset(tokenizer, test_df[text_column], test_df[label_column]).batch(batch_size)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08, clipnorm=1.0),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy("accuracy")],
    )
    model.fit(train_data, epochs=epochs, validation_data=val_data)
    return model


def predict(model, tokenizer, texts, batch_size=100, max_length=MAX_LENGTH):
    import tensorflow as tf

    predictions = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        encodings = tokenizer(
            batch, max_length=max_length, padding=True, truncation=True, return_tensors="tf"
        )
        logits = model(encodings).logits
        predictions.extend(tf.argmax(logits, axis=-1).numpy())
    return predictions


def evaluate(y_true, y_pred):
    print(f"accuracy: {metrics.accuracy_score(y_true, y_pred):.4f}")
    print(metrics.confusion_matrix(y_true, y_pred, labels=[0, 1, 2]))
    print(metrics.classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=LABELS))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="../reddit/Crypto_c.csv", help="Labeled CSV from reddit/merge_clean.py")
    parser.add_argument("--text-column", default="DATA_COLUMN")
    parser.add_argument("--label-column", default="LABEL_COLUMN")
    parser.add_argument("--test-size", type=int, default=500, help="Number of rows held out for evaluation")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", default=None, help="If set, save the fine-tuned model here")
    parser.add_argument("--push-to-hub", default=None, help="If set, push to this Hugging Face Hub repo id")
    return parser.parse_args()


def main():
    args = parse_args()

    train_df, test_df = load_dataset(args.data, args.text_column, args.label_column, args.test_size)
    model, tokenizer = build_model()
    model = train(
        model, tokenizer, train_df, test_df, args.text_column, args.label_column, args.epochs, args.batch_size
    )

    test_predictions = predict(model, tokenizer, test_df[args.text_column])
    evaluate(test_df[args.label_column], test_predictions)

    if args.output_dir:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        tokenizer.push_to_hub(args.push_to_hub)
        model.push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    main()
