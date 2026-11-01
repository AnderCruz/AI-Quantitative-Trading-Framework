"""Trace the project's canonical metric function against saved LSTM artefacts.

Run from the repository root. This is read-only: it does not train models,
modify files, fetch market data, use Final Test data, or perform backtesting.
"""
from pathlib import Path
import inspect
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "reports" / "model_evaluation" / "LLOY.L"
PREDICTIONS = BASE / "lstm_regression" / "lstm_validation_predictions.csv"
METRICS = BASE / "lstm_regression" / "lstm_validation_metrics.csv"


def main():
    if not PREDICTIONS.exists() or not METRICS.exists():
        raise FileNotFoundError("Expected saved LSTM validation artefacts were not found.")

    from src.baseline_evaluation import _calculate_metrics

    print("PROJECT METRIC IMPLEMENTATION AUDIT")
    print("=" * 78)
    print("Canonical function: src.baseline_evaluation._calculate_metrics")
    print("Source implementation:")
    print(inspect.getsource(_calculate_metrics))

    predictions = pd.read_csv(PREDICTIONS)
    metrics = pd.read_csv(METRICS)
    cols = {str(c).lower(): c for c in predictions.columns}
    def pick(*names):
        return next((cols[n] for n in names if n in cols), None)
    date_col = pick("date", "prediction_date", "session_date")
    horizon_col = pick("horizon", "forecast_horizon", "horizon_days")
    actual_col = pick("actual_return", "actual", "target", "y_true", "realised_return")
    pred_col = pick("prediction", "predicted_return", "y_pred", "forecast")
    if any(x is None for x in [date_col, horizon_col, actual_col, pred_col]):
        raise ValueError(f"Unrecognised prediction columns: {list(predictions.columns)}")
    predictions[horizon_col] = pd.to_numeric(predictions[horizon_col], errors="raise").astype(int)
    predictions[actual_col] = pd.to_numeric(predictions[actual_col], errors="raise")
    predictions[pred_col] = pd.to_numeric(predictions[pred_col], errors="raise")

    metric_cols = {str(c).lower(): c for c in metrics.columns}
    mh = next((metric_cols[n] for n in ("horizon", "forecast_horizon", "horizon_days") if n in metric_cols), None)
    md = next((metric_cols[n] for n in ("directional_accuracy", "da") if n in metric_cols), None)
    if mh is None or md is None:
        raise ValueError(f"Unrecognised metric columns: {list(metrics.columns)}")

    rows = []
    for horizon, group in predictions.groupby(horizon_col, sort=True):
        result = _calculate_metrics(
            actual=group[actual_col].reset_index(drop=True),
            predicted=group[pred_col].reset_index(drop=True),
        )
        reported = metrics.loc[pd.to_numeric(metrics[mh]) == int(horizon), md]
        canonical_da = result.get("directional_accuracy", np.nan) if isinstance(result, dict) else np.nan
        rows.append({
            "horizon": int(horizon),
            "n": len(group),
            "canonical_function_DA": canonical_da,
            "saved_original_DA": float(reported.iloc[0]) if len(reported) else np.nan,
            "absolute_difference": abs(float(canonical_da) - float(reported.iloc[0])) if len(reported) and pd.notna(canonical_da) else np.nan,
        })
    print("\nRecalculation using canonical function:")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.9f}"))
    print("\nNo files were modified.")

if __name__ == "__main__":
    main()
