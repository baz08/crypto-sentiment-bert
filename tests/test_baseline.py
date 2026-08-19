from baseline import build_baseline


def test_baseline_fits_and_predicts():
    texts = ["great news for bitcoin", "terrible crash today", "price unchanged"] * 5
    labels = [2, 0, 1] * 5

    model = build_baseline()
    model.fit(texts, labels)
    predictions = model.predict(texts[:3])

    assert len(predictions) == 3
    assert set(predictions).issubset({0, 1, 2})
