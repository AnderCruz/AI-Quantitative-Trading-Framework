"""Audit LSTM directional-accuracy discrepancies across saved validation artefacts.

Run from the repository root. Reads existing CSV artefacts only; does not train,
fetch data, use Final Test observations, or perform backtesting.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "reports" / "model_evaluation" / "LLOY.L"
LSTM_PRED = BASE / "lstm_regression" / "lstm_validation_predictions.csv"
LSTM_METRICS = BASE / "lstm_regression" / "lstm_validation_metrics.csv"
COMPARISON_PRED = BASE / "model_comparison" / "all_models_including_lstm_aligned_predictions.csv"
COMPARISON_METRICS = BASE / "model_comparison" / "all_models_including_lstm_comparison_metrics.csv"

def normalise_predictions(path):
    df = pd.read_csv(path)
    aliases = {
        "date": ["date", "prediction_date", "session_date"],
        "horizon": ["horizon", "forecast_horizon", "horizon_days"],
        "prediction": ["prediction", "predicted_return", "y_pred", "forecast"],
        "actual_return": ["actual_return", "actual", "target", "y_true", "realised_return"],
    }
    lower = {str(c).lower(): c for c in df.columns}
    ren = {}
    for target, candidates in aliases.items():
        found = next((lower[c] for c in candidates if c in lower), None)
        if found is None:
            raise ValueError(f"{path.name}: cannot identify {target}; columns={list(df.columns)}")
        ren[found] = target
    df = df.rename(columns=ren)
    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize()
    df["horizon"] = pd.to_numeric(df["horizon"], errors="raise").astype(int)
    for c in ["prediction", "actual_return"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["date", "horizon", "prediction", "actual_return"]].dropna()

def metric_columns(path):
    df = pd.read_csv(path)
    lower = {str(c).lower(): c for c in df.columns}
    hcol = next((lower[c] for c in ["horizon", "forecast_horizon", "horizon_days"] if c in lower), None)
    dcol = next((lower[c] for c in ["directional_accuracy", "directional_accuracy_score", "da"] if c in lower), None)
    if hcol is None or dcol is None:
        print(f"\nMetrics columns in {path.name}: {list(df.columns)}")
        return None
    return df[[hcol, dcol]].rename(columns={hcol:"horizon", dcol:"reported_da"})

def main():
    for p in [LSTM_PRED, LSTM_METRICS, COMPARISON_PRED, COMPARISON_METRICS]:
        if not p.exists():
            raise FileNotFoundError(f"Required artefact missing: {p}")
    original = normalise_predictions(LSTM_PRED)
    comparison = normalise_predictions(COMPARISON_PRED)
    comparison = comparison.assign(model=pd.read_csv(COMPARISON_PRED)["model"].astype(str).values)
    comparison = comparison[comparison["model"].str.lower().eq("lstm_regression")].drop(columns="model")
    keys = ["date", "horizon"]
    merged = original.merge(comparison, on=keys, how="outer", suffixes=("_original", "_comparison"), indicator=True)
    print("LSTM DIRECTIONAL-ACCURACY AUDIT")
    print("=" * 78)
    print(f"Original prediction rows: {len(original)}")
    print(f"Comparison LSTM rows:     {len(comparison)}")
    print(f"Key matches:              {(merged['_merge'] == 'both').sum()}")
    print(f"Original-only rows:       {(merged['_merge'] == 'left_only').sum()}")
    print(f"Comparison-only rows:     {(merged['_merge'] == 'right_only').sum()}")
    matched = merged[merged["_merge"] == "both"].copy()
    for col in ["prediction", "actual_return"]:
        delta = (matched[f"{col}_original"] - matched[f"{col}_comparison"]).abs()
        print(f"Maximum absolute {col} difference: {delta.max():.12g}")
    print("\nRecomputed LSTM DA by horizon:")
    records=[]
    for horizon, g in matched.groupby("horizon"):
        actual = g["actual_return_original"].to_numpy(float)
        pred = g["prediction_original"].to_numpy(float)
        strict = float(np.mean(np.sign(actual) == np.sign(pred)))
        nonzero = np.sign(pred) != 0
        conditional = float(np.mean(np.sign(actual[nonzero]) == np.sign(pred[nonzero]))) if nonzero.any() else np.nan
        actual_nonzero = actual != 0
        both_nonzero = (actual != 0) & (pred != 0)
        records.append({"horizon":int(horizon),"n":len(g),"strict_sign_equality":strict,
                        "conditional_on_nonzero_prediction":conditional,
                        "conditional_on_nonzero_actual_and_prediction":
                            float(np.mean(np.sign(actual[both_nonzero]) == np.sign(pred[both_nonzero]))) if both_nonzero.any() else np.nan,
                        "zero_actual_count":int((actual == 0).sum()),
                        "zero_prediction_count":int((pred == 0).sum())})
    print(pd.DataFrame(records).to_string(index=False, float_format=lambda x:f"{x:.9f}"))
    print("\nReported metrics artefacts:")
    for path in [LSTM_METRICS, COMPARISON_METRICS]:
        print(f"\n{path.relative_to(ROOT)}")
        df = pd.read_csv(path)
        print(df.to_string(index=False) if "lstm_validation_metrics" in path.name else
              df[df.get("model", pd.Series(index=df.index, dtype=str)).astype(str).str.lower().eq("lstm_regression")].to_string(index=False))

if __name__ == "__main__":
    main()
