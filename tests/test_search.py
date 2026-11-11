import pandas as pd
import pytest

from berttest import DEFAULT_CONFIG, expand_grid, search_hyperparameters


def make_train_df(n=100):
    return pd.DataFrame({"DATA_COLUMN": [f"text {i}" for i in range(n)], "LABEL_COLUMN": [i % 3 for i in range(n)]})


def fake_trial(scores):
    """Return a trial_fn whose per-epoch macro-F1 comes from ``scores[(lr, batch_size)]``."""
    calls = []

    def trial(config, fit_df, val_df, text_column, label_column, seed):
        calls.append({"config": config, "fit": fit_df, "val": val_df})
        per_epoch = scores[(config["lr"], config["batch_size"])]
        return [{"epoch": e, "accuracy": f, "macro_f1": f} for e, f in enumerate(per_epoch, 1)]

    trial.calls = calls
    return trial


def test_expand_grid_returns_every_combination():
    combos = expand_grid({"lr": [1, 2, 3], "batch_size": [16, 32]})
    assert len(combos) == 6
    assert {"lr": 3, "batch_size": 32} in combos


def test_expand_grid_samples_n_trials_reproducibly():
    grid = {"lr": [1, 2, 3, 4], "batch_size": [16, 32]}
    first = expand_grid(grid, n_trials=3, seed=0)
    assert len(first) == 3
    assert first == expand_grid(grid, n_trials=3, seed=0)


def test_search_picks_best_config_and_epoch():
    scores = {
        (1e-5, 16): [0.50, 0.55, 0.56],
        (1e-5, 32): [0.48, 0.52, 0.54],
        (3e-5, 16): [0.55, 0.61, 0.59],  # best: epoch 2
        (3e-5, 32): [0.53, 0.58, 0.60],
    }
    trial = fake_trial(scores)

    best, results = search_hyperparameters(
        make_train_df(), "DATA_COLUMN", "LABEL_COLUMN",
        grid={"lr": [1e-5, 3e-5], "batch_size": [16, 32]}, max_epochs=3, trial_fn=trial,
    )

    assert best == {**DEFAULT_CONFIG, "lr": 3e-5, "batch_size": 16, "epochs": 2}
    assert len(results) == 4 * 3  # one row per (config, epoch)
    assert results.iloc[0]["macro_f1"] == 0.61
    assert all(call["config"]["epochs"] == 3 for call in trial.calls)


def test_search_breaks_ties_toward_fewer_epochs():
    trial = fake_trial({(2e-5, 16): [0.60, 0.60]})
    best, _ = search_hyperparameters(
        make_train_df(), "DATA_COLUMN", "LABEL_COLUMN", grid={"lr": [2e-5], "batch_size": [16]}, max_epochs=2, trial_fn=trial
    )
    assert best["epochs"] == 1


def test_search_validates_on_a_split_of_the_training_data():
    train_df = make_train_df(100)
    trial = fake_trial({(2e-5, 16): [0.5]})

    search_hyperparameters(
        train_df, "DATA_COLUMN", "LABEL_COLUMN", grid={"lr": [2e-5], "batch_size": [16]},
        max_epochs=1, val_size=0.2, trial_fn=trial,
    )

    fit, val = trial.calls[0]["fit"], trial.calls[0]["val"]
    assert len(val) == 20 and len(fit) == 80
    assert set(fit["DATA_COLUMN"]).isdisjoint(val["DATA_COLUMN"])
    assert set(fit["DATA_COLUMN"]) | set(val["DATA_COLUMN"]) == set(train_df["DATA_COLUMN"])


def test_search_best_config_is_json_serializable():
    import json

    trial = fake_trial({(2e-5, 16): [0.5], (2e-5, 32): [0.6]})
    best, _ = search_hyperparameters(
        make_train_df(), "DATA_COLUMN", "LABEL_COLUMN", grid={"lr": [2e-5], "batch_size": [16, 32]}, max_epochs=1, trial_fn=trial
    )
    assert json.loads(json.dumps(best))["batch_size"] == 32
    assert type(best["batch_size"]) is int  # not 32.0
    assert type(best["lr"]) is float


def test_epochs_in_grid_is_rejected():
    with pytest.raises(ValueError, match="max_epochs"):
        search_hyperparameters(make_train_df(), "DATA_COLUMN", "LABEL_COLUMN", grid={"epochs": [2, 3]}, trial_fn=fake_trial({}))
