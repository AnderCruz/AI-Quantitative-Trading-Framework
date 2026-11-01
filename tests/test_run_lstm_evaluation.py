"""Integration tests for the LSTM experiment runner."""

import json

import pandas as pd

from src import run_lstm_evaluation as runner
from src.feature_target_pipeline import FEATURE_COLUMNS
from src.target_generation import HORIZONS


def test_runner_excludes_final_test_and_saves_outputs(
    tmp_path,
    monkeypatch,
):
    """The runner must isolate Final Test data before feature generation."""

    raw_path = tmp_path / "LLOY.L.csv"
    output_dir = tmp_path / "reports"

    raw = pd.DataFrame(
        {
            "date": [
                "2023-10-20",
                "2023-10-23",
                "2025-10-22",
                "2025-10-23",
                "2026-10-23",
            ],
            "ticker": ["LLOY.L"] * 5,
            "Open": [40.0, 41.0, 45.0, 46.0, 47.0],
            "Close": [40.5, 41.5, 45.5, 46.5, 47.5],
        }
    )
    raw.to_csv(raw_path, index=False)

    monkeypatch.setattr(runner, "RAW_DATA_PATH", raw_path)
    monkeypatch.setattr(runner, "OUTPUT_DIR", output_dir)

    captured = {}

    def fake_build_feature_target_dataset(permitted_raw):
        """Record the input before returning a synthetic integrated dataset."""

        captured["raw_dates"] = permitted_raw["date"].copy()

        integrated = permitted_raw.copy()

        for column in FEATURE_COLUMNS:
            integrated[column] = 0.01

        for horizon in HORIZONS:
            integrated[f"forward_return_{horizon}d"] = 0.001
            integrated[f"target_maturity_date_{horizon}d"] = (
                pd.to_datetime(integrated["date"])
                + pd.offsets.BDay(horizon)
            )

        return integrated

    def fake_evaluate_lstm_walk_forward(**kwargs):
        """Record the evaluation contract without training a model."""

        captured["evaluation_arguments"] = kwargs

        predictions = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-02").date()],
                "cutoff_date": [pd.Timestamp("2023-12-29").date()],
                "horizon": [1],
                "model": ["lstm_regression"],
                "prediction": [0.001],
                "actual_return": [0.002],
                "training_sample_size": [500],
            }
        )

        metrics = pd.DataFrame(
            {
                "horizon": [1],
                "model": ["lstm_regression"],
                "mae": [0.001],
                "rmse": [0.001],
            }
        )

        return predictions, metrics

    monkeypatch.setattr(
        runner,
        "build_feature_target_dataset",
        fake_build_feature_target_dataset,
    )
    monkeypatch.setattr(
        runner,
        "evaluate_lstm_walk_forward",
        fake_evaluate_lstm_walk_forward,
    )

    runner.main()

    # Final Test dates must be removed before feature generation.
    assert captured["raw_dates"].max().date().isoformat() == "2025-10-22"

    assert all(
        date.date().isoformat() <= "2025-10-22"
        for date in captured["raw_dates"]
    )

    arguments = captured["evaluation_arguments"]

    assert arguments["development_start"].isoformat() == "2016-10-24"
    assert arguments["development_end"].isoformat() == "2023-10-20"
    assert arguments["validation_start"].isoformat() == "2023-10-23"
    assert arguments["validation_end"].isoformat() == "2025-10-22"
    assert arguments["horizons"] == HORIZONS
    assert arguments["min_training_samples"] == 500

    # The executor must persist all expected artefacts.
    predictions_path = output_dir / "lstm_validation_predictions.csv"
    metrics_path = output_dir / "lstm_validation_metrics.csv"
    metadata_path = output_dir / "lstm_validation_run.json"

    assert predictions_path.exists()
    assert metrics_path.exists()
    assert metadata_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["ticker"] == "LLOY.L"
    assert metadata["final_test_data_used"] is False
    assert metadata["backtesting_performed"] is False
    assert metadata["horizons"] == list(HORIZONS)
    assert metadata["prediction_counts_by_horizon"] == {"1": 1}