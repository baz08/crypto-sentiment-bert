import pandas as pd

from berttest import evaluate, load_dataset


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
            "DATA_COLUMN": ["a", None, "c", "d"],
            "LABEL_COLUMN": [0, 1, 2, 0],
        }
    ).to_csv(csv_path, index=False)

    train_df, test_df = load_dataset(csv_path, "DATA_COLUMN", "LABEL_COLUMN", test_size=1)

    assert len(train_df) + len(test_df) == 3


def test_evaluate_prints_accuracy(capsys):
    evaluate([0, 1, 2, 0], [0, 1, 1, 0])
    output = capsys.readouterr().out
    assert "accuracy: 0.7500" in output
