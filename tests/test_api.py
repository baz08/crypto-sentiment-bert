from fastapi.testclient import TestClient

from main import app
from ML.predmodel import get_model


class FakeModel:
    def predict(self, text):
        return "Positive"


app.dependency_overrides[get_model] = lambda: FakeModel()
client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict():
    response = client.post("/predict", json={"text": "I love bitcoin"})
    assert response.status_code == 200
    assert response.json() == {"sentiment": "Positive"}


def test_predict_rejects_empty_text():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422
