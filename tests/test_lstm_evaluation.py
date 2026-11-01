"""Tests for LSTM walk-forward evaluation."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.feature_target_pipeline import FEATURE_COLUMNS
from src.lstm_evaluation import (
    DEFAULT_PARAMETERS,
    MODEL_NAME,
    evaluate_lstm_walk_forward,
)
from src.temporal_split import get_target_maturity_dates


def make_synthetic_data(periods=900):
    """Create deterministic chronological data for evaluation tests."""

    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=periods)

    t = np.arange(periods, dtype=float)
    close = 100 + 0.02 * t + np.sin(t / 8)
    open_price = close * (1 + 0.001 * np.sin(t / 5))

    history = pd.DataFrame(
        {
            "date": dates,
            "ticker": "TEST",
            "Open": open_price,
            "Close": close,
        }
    )

    maturity_dates = get_target_maturity_dates(history)

    features = pd.DataFrame(index=history.index)
    close_series = history["Close"]
    daily_returns = close_series.pct_change()

    features["close_return_1d"] = close_series.pct_change(1)
    features["close_return_5d"] = close_series.pct_change(5)
    features["close_return_10d"] = close_series.pct_change(10)
    features["close_return_20d"] = close_series.pct_change(20)
    features["close_volatility_5d"] = daily_returns.rolling(5).std()
    features["close_volatility_10d"] = daily_returns.rolling(10).std()
    features["close_volatility_20d"] = daily_returns.rolling(20).std()

    features = features.loc[:, FEATURE_COLUMNS]

    targets = pd.DataFrame(index=history.index)

    for horizon in (1, 5, 10):
        targets[f"forward_return_{horizon}d"] = (
            history["Close"].shift(-horizon)
            / history["Open"].shift(-1)
            - 1
        )

    return history, features, targets, maturity_dates


def run_evaluation(data, **overrides):
    """Run a small, deterministic synthetic evaluation."""

    history, features, targets, maturity_dates = data

    arguments = {
        "history": history,
        "features": features,
        "targets": targets,
        "maturity_dates": maturity_dates,
        "development_start": date(2020, 1, 1),
        "development_end": date(2021, 12, 31),
        "validation_start": date(2022, 1, 3),
        "validation_end": date(2022, 2, 28),
        "horizons": (1,),
        "min_training_samples": 30,
        "sequence_length": 20,
        "epochs": 1,
        "batch_size": 32,
        "learning_rate": 0.001,
        "random_state": 42,
    }

    arguments.update(overrides)

    return evaluate_lstm_walk_forward(**arguments)


def test_default_lstm_evaluation_parameters():
    assert DEFAULT_PARAMETERS["sequence_length"] == 20
    assert DEFAULT_PARAMETERS["epochs"] == 20
    assert DEFAULT_PARAMETERS["batch_size"] == 32
    assert DEFAULT_PARAMETERS["learning_rate"] == 0.001


def test_evaluation_returns_finite_predictions_and_metrics():
    predictions, metrics = run_evaluation(make_synthetic_data())

    assert not predictions.empty
    assert set(predictions["model"]) == {MODEL_NAME}
    assert set(predictions["horizon"]) == {1}
    assert np.isfinite(predictions["prediction"]).all()
    assert np.isfinite(predictions["actual_return"]).all()
    assert (predictions["training_sample_size"] >= 30).all()

    assert not metrics.empty
    assert set(metrics["horizon"]) == {1}
    assert {"mae", "rmse"}.issubset(metrics.columns)


def test_predictions_remain_inside_validation_period():
    predictions, _ = run_evaluation(make_synthetic_data())

    assert predictions["date"].min() >= date(2022, 1, 3)
    assert predictions["date"].max() <= date(2022, 2, 28)


def test_predictions_do_not_precede_their_cutoffs():
    predictions, _ = run_evaluation(make_synthetic_data())

    assert all(
        row.cutoff_date <= row.date
        for row in predictions.itertuples(index=False)
    )


def test_targets_maturing_after_validation_end_are_excluded():
    data = make_synthetic_data()
    predictions, _ = run_evaluation(data)

    history, _, _, maturity_dates = data

    for row in predictions.itertuples(index=False):
        position = history.index[
            history["date"].dt.date == row.date
        ][0]

        maturity = maturity_dates.loc[
            position,
            f"target_maturity_date_{row.horizon}d",
        ]

        assert maturity.date() <= date(2022, 2, 28)


def test_evaluator_rejects_invalid_learning_rate():
    with pytest.raises(ValueError, match="learning_rate"):
        run_evaluation(
            make_synthetic_data(),
            learning_rate=0,
        )


def test_evaluator_rejects_invalid_sequence_length():
    with pytest.raises(ValueError, match="sequence_length"):
        run_evaluation(
            make_synthetic_data(),
            sequence_length=0,
        )


def test_evaluator_rejects_duplicate_horizons():
    with pytest.raises(ValueError, match="duplicates"):
        run_evaluation(
            make_synthetic_data(),
            horizons=(1, 1),
        )


def test_evaluator_enforces_minimum_training_sequences():
    data = make_synthetic_data()

    with pytest.raises(
        ValueError,
        match="Insufficient eligible LSTM training sequences",
    ):
        run_evaluation(
            data,
            min_training_samples=10_000,
        )


def test_incomplete_validation_features_are_not_scored():
    data = list(make_synthetic_data())
    history, features, targets, maturity_dates = data

    validation_date = date(2022, 1, 3)
    matching = history.index[
        history["date"].dt.date == validation_date
    ]

    assert len(matching) == 1

    features.loc[
        matching[0],
        list(FEATURE_COLUMNS),
    ] = np.nan

    predictions, _ = run_evaluation(
        (history, features, targets, maturity_dates)
    )

    prediction_dates = set(predictions["date"])

    assert validation_date not in prediction_dates

    # A next-session window still contains the missing feature row.
    next_session = (
        history.loc[history["date"].dt.date > validation_date, "date"]
        .dt.date
        .iloc[0]
    )

    assert next_session not in prediction_dates

def test_future_target_perturbation_does_not_change_earlier_prediction():
    """Future targets must not influence an earlier forecast."""

    data = make_synthetic_data()

    original_history, features, targets, maturity_dates = data

    original_predictions, _ = run_evaluation(data)

    # Perturb validation-period targets without changing their
    # maturity dates or any historical feature.
    modified_targets = targets.copy()

    validation_start = date(2022, 1, 3)

    validation_mask = (
        original_history["date"].dt.date >= validation_start
    )

    for column in modified_targets.columns:
        modified_targets.loc[validation_mask, column] += 10.0

    modified_data = (
        original_history,
        features,
        modified_targets,
        maturity_dates,
    )

    modified_predictions, _ = run_evaluation(modified_data)

    # Compare the first validation prediction, before the modified
    # validation targets can have matured and become eligible.
    original_first = original_predictions.sort_values("date").iloc[0]
    modified_first = modified_predictions.sort_values("date").iloc[0]

    assert original_first["date"] == modified_first["date"]

    np.testing.assert_allclose(
        original_first["prediction"],
        modified_first["prediction"],
        rtol=0,
        atol=1e-7,
    )