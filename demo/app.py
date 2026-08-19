"""Gradio demo for the crypto sentiment model.

Run locally:
    pip install -r demo/requirements.txt
    python demo/app.py

Or deploy as a Hugging Face Space: point the Space's app file at demo/app.py
(Spaces have their own Hub access, so the model download that's blocked in
some sandboxed/offline dev environments works fine there).
"""

import sys
from pathlib import Path

import gradio as gr

API_DIR = Path(__file__).resolve().parents[1] / "deployment" / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from ML.predmodel import get_model  # noqa: E402

EXAMPLES = [
    "I love bitcoin, to the moon!",
    "This coin is a total scam, stay away.",
    "Price hasn't moved much today.",
]


def predict(text):
    if not text or not text.strip():
        return "Enter some text to analyze."
    return get_model().predict(text)


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Crypto-related text", placeholder="e.g. BTC is going to the moon!", lines=3),
    outputs=gr.Textbox(label="Sentiment"),
    examples=EXAMPLES,
    title="Crypto Sentiment BERT",
    description=(
        "Sentiment analysis for cryptocurrency-related text, powered by a BERT model "
        "fine-tuned on Reddit comments. Model: "
        "[baz08/crypto-Bert-test](https://huggingface.co/baz08/crypto-Bert-test)."
    ),
)

if __name__ == "__main__":
    demo.launch()
