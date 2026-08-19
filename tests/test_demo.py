from unittest.mock import patch

from app import predict


def test_predict_rejects_blank_input():
    assert predict("") == "Enter some text to analyze."
    assert predict("   ") == "Enter some text to analyze."


def test_predict_delegates_to_model():
    class FakeModel:
        def predict(self, text):
            return "Positive"

    with patch("app.get_model", return_value=FakeModel()):
        assert predict("I love bitcoin") == "Positive"
