import pandas as pd

from berttest import evaluate, load_dataset, majority_baseline


def test_load_dataset_splits_by_test_size(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "DATA_COLUMN": [f"text {i}" for i in range(10)],
            "LABEL_COLUMN": [i % 3 for i in range(10)],
        }
    ).to_csv(csv_path, index=False)

    train_df, test_df = load_dataset(csv_path, "DATA_COLUMN", "LABEL_COLUMN", test_size=3)

    assert len(train_df) == 7
    assert len(test_df) == 3


def test_load_dataset_drops_missing_values(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "DATA_COLUMN": ["a", None, "c", "d", "e", "f", "g", "h"],
            "LABEL_COLUMN": [0, 1, 0, 1, 0, 1, None, 1],
        }
    ).to_csv(csv_path, index=False)

    train_df, test_df = load_dataset(csv_path, "DATA_COLUMN", "LABEL_COLUMN", test_size=2)

    assert len(train_df) + len(test_df) == 6


def test_load_dataset_is_shuffled_and_stratified(tmp_path):
    # Rows ordered by label, like Crypto_c.csv is ordered by source. Taking the
    # last N rows would give a test set of only one label.
    labels = [0] * 30 + [1] * 60 + [2] * 30
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"DATA_COLUMN": [f"t{i}" for i in range(120)], "LABEL_COLUMN": labels}).to_csv(
        csv_path, index=False
    )

    train_df, test_df = load_dataset(csv_path, "DATA_COLUMN", "LABEL_COLUMN", test_size=0.25)

    assert len(test_df) == 30
    counts = test_df["LABEL_COLUMN"].value_counts()
    assert counts[1] == 15  # half the test set, matching the full data
    assert counts[0] in (7, 8) and counts[2] in (7, 8)
    assert set(train_df["DATA_COLUMN"]).isdisjoint(test_df["DATA_COLUMN"])


def test_load_dataset_is_reproducible(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"DATA_COLUMN": [f"t{i}" for i in range(30)], "LABEL_COLUMN": [i % 3 for i in range(30)]}).to_csv(
        csv_path, index=False
    )

    _, first = load_dataset(csv_path, "DATA_COLUMN", "LABEL_COLUMN", test_size=6, seed=1)
    _, second = load_dataset(csv_path, "DATA_COLUMN", "LABEL_COLUMN", test_size=6, seed=1)

    assert first["DATA_COLUMN"].tolist() == second["DATA_COLUMN"].tolist()


def test_majority_baseline_predicts_most_common_training_label():
    assert majority_baseline([1, 1, 0, 2], [0, 2, 2]) == [1, 1, 1]


def test_evaluate_reports_model_and_baseline(capsys):
    results = evaluate([0, 1, 2, 0], [0, 1, 1, 0], y_train=[0, 0, 1])

    assert results["model"]["accuracy"] == 0.75
    assert results["majority_baseline"]["accuracy"] == 0.5
    assert "macro-F1" in capsys.readouterr().out


def test_tfidf_baseline_learns_obvious_words():
    from berttest import tfidf_baseline

    train_texts = ["great pump moon", "great moon gains", "terrible crash dump", "terrible dump loss", "price is flat today", "price flat sideways"] * 3
    train_labels = [2, 2, 0, 0, 1, 1] * 3

    predictions = tfidf_baseline(train_texts, train_labels, ["great moon", "terrible dump", "flat price"])

    assert predictions == [2, 0, 1]
