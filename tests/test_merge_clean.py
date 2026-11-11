import pandas as pd

from merge_clean import build_training_frame


def test_build_training_frame_outputs_training_columns():
    frames = [
        pd.DataFrame({"DATA_COLUMN": ["I love BTC", "meh"], "LABEL_COLUMN": [2.0, 1.0], "source": "btc"}),
        pd.DataFrame({"DATA_COLUMN": ["ETH is dead"], "LABEL_COLUMN": [0], "source": "eth"}),
    ]

    df = build_training_frame(frames)

    assert list(df.columns) == ["DATA_COLUMN", "LABEL_COLUMN", "source"]
    assert df["LABEL_COLUMN"].tolist() == [2, 1, 0]
    assert df["LABEL_COLUMN"].dtype.kind == "i"


def test_build_training_frame_drops_unusable_rows():
    frames = [
        pd.DataFrame(
            {
                "DATA_COLUMN": ["keep me", "no label", "https://only-a-link.com", None, "keep me", "bad label"],
                "LABEL_COLUMN": [1, None, 2, 0, 1, 7],
                "source": "x",
            }
        )
    ]

    df = build_training_frame(frames)

    assert df["DATA_COLUMN"].tolist() == ["keep me"]
