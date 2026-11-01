"""Run the LSTM validation experiment without accessing Final Test data."""

from datetime import date
import json
from pathlib import Path

import pandas as pd

from src.feature_target_pipeline import (
    FEATURE_COLUMNS,
    build_feature_target_dataset,
)
from src.lstm_evaluation import (
    DEFAULT_PARAMETERS,
    MODEL_NAME,
    evaluate_lstm_walk_forward,
)
from src.target_generation import HORIZONS


ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = (
    ROOT / "data" / "raw" / "yahoo" / "configured" / "LLOY.L.csv"
)

OUTPUT_DIR = (
    ROOT
    / "reports"
    / "model_evaluation"
    / "LLOY.L"
    / "lstm_regression"
)

DEVELOPMENT_START = date(2016, 10, 24)
DEVELOPMENT_END = date(2023, 10, 20)

VALIDATION_START = date(2023, 10, 23)
VALIDATION_END = date(2025, 10, 22)

MIN_TRAINING_SAMPLES = 500


def main() -> None:
    """Prepare the permitted data and run the LSTM validation experiment."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw price data not found: {RAW_DATA_PATH}"
        )

    raw = pd.read_csv(RAW_DATA_PATH)

    required_columns = {"date", "ticker", "Open", "Close"}
    missing = required_columns.difference(raw.columns)

    if missing:
        raise ValueError(
            f"Raw data is missing required columns: {sorted(missing)}"
        )

    raw["date"] = pd.to_datetime(raw["date"], errors="raise")

    if raw["date"].isna().any():
        raise ValueError("Raw data contains missing dates.")

    observed_tickers = raw["ticker"].dropna().astype(str).unique()

    if len(observed_tickers) != 1 or observed_tickers[0] != "LLOY.L":
        raise ValueError(
            "Expected raw data for LLOY.L only. "
            f"Found: {observed_tickers.tolist()}"
        )

    if raw["date"].duplicated().any():
        raise ValueError("Raw data contains duplicate dates.")

    # Exclude Final Test observations before constructing features and targets.
    raw = raw.loc[
        raw["date"].dt.date <= VALIDATION_END
    ].copy()

    raw = raw.sort_values("date").reset_index(drop=True)

    if raw.empty:
        raise ValueError(
            "No price observations remain within the permitted period."
        )

    latest_date = raw["date"].iloc[-1].date()

    if latest_date > VALIDATION_END:
        raise AssertionError(
            "Final Test observations entered the input data."
        )

    integrated = build_feature_target_dataset(raw)

    history = integrated.loc[:, ["date", "ticker"]].copy()

    features = integrated.loc[:, list(FEATURE_COLUMNS)].copy()

    target_columns = [
        f"forward_return_{horizon}d"
        for horizon in HORIZONS
    ]
    targets = integrated.loc[:, target_columns].copy()

    maturity_columns = [
        f"target_maturity_date_{horizon}d"
        for horizon in HORIZONS
    ]
    maturity_dates = integrated.loc[:, maturity_columns].copy()

    predictions, metrics = evaluate_lstm_walk_forward(
        history=history,
        features=features,
        targets=targets,
        maturity_dates=maturity_dates,
        development_start=DEVELOPMENT_START,
        development_end=DEVELOPMENT_END,
        validation_start=VALIDATION_START,
        validation_end=VALIDATION_END,
        horizons=HORIZONS,
        min_training_samples=MIN_TRAINING_SAMPLES,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions_path = (
        OUTPUT_DIR / "lstm_validation_predictions.csv"
    )
    metrics_path = (
        OUTPUT_DIR / "lstm_validation_metrics.csv"
    )
    run_path = OUTPUT_DIR / "lstm_validation_run.json"

    predictions.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    prediction_counts = {
        str(int(horizon)): int(count)
        for horizon, count in (
            predictions.groupby("horizon").size().items()
        )
    }

    run_metadata = {
        "ticker": "LLOY.L",
        "model": MODEL_NAME,
        "development_start": DEVELOPMENT_START.isoformat(),
        "development_end": DEVELOPMENT_END.isoformat(),
        "validation_start": VALIDATION_START.isoformat(),
        "validation_end": VALIDATION_END.isoformat(),
        "latest_input_session": latest_date.isoformat(),
        "final_test_data_used": False,
        "backtesting_performed": False,
        "horizons": list(HORIZONS),
        "minimum_training_samples": MIN_TRAINING_SAMPLES,
        "model_parameters": DEFAULT_PARAMETERS,
        "prediction_counts_by_horizon": prediction_counts,
    }

    run_path.write_text(
        json.dumps(run_metadata, indent=2),
        encoding="utf-8",
    )

    print("\nLSTM VALIDATION COMPLETED")
    print("=" * 80)
    print("Ticker: LLOY.L")
    print(f"Latest permitted input session: {latest_date}")
    print(f"Final Test data used: {run_metadata['final_test_data_used']}")
    print(f"Backtesting performed: {run_metadata['backtesting_performed']}")
    print(f"Predictions: {len(predictions)}")

    print("\nValidation metrics:")
    print(
        metrics.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nPrediction counts by horizon:")
    print(
        predictions.groupby("horizon")
        .size()
        .rename("sample_size")
        .to_string()
    )

    print("\nOutput files:")
    print(predictions_path)
    print(metrics_path)
    print(run_path)


if __name__ == "__main__":
    main()