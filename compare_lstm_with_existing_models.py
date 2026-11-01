"""Align existing LSTM forecasts with prior validation forecasts and compare them.

This script does not train models, fetch market data, use Final Test data, or
perform backtesting. It compares saved validation predictions only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent
LSTM_DIR = ROOT / "reports" / "model_evaluation" / "LLOY.L" / "lstm_regression"
COMPARISON_DIR = ROOT / "reports" / "model_evaluation" / "LLOY.L" / "model_comparison"

LSTM_PREDICTIONS = LSTM_DIR / "lstm_validation_predictions.csv"
REFERENCE_CANDIDATES = [
    ROOT / "reports" / "model_evaluation" / "LLOY.L"
    / "all_models_including_mlp_aligned_predictions.csv",
    ROOT / "reports" / "model_evaluation" / "LLOY.L"
    / "lightgbm_regression" / "all_models_including_mlp_aligned_predictions.csv",
    ROOT / "reports" / "model_evaluation" / "LLOY.L"
    / "lightgbm_regression" / "all_models_aligned_predictions.csv",
]

# Use non-zero HAC lag even for the one-session horizon because forecast-loss
# differences can remain serially dependent even when targets do not overlap.
HAC_LAGS = {1: 5, 5: 4, 10: 9}
# Block lengths reflect at least one trading week, and the target horizon.
BOOTSTRAP_BLOCK_LENGTHS = {1: 5, 5: 5, 10: 10}
BOOTSTRAP_REPLICATES = 2000
RANDOM_SEED = 42
ALIGNMENT_TOLERANCE = 1e-10


def find_reference_file() -> Path:
    for candidate in REFERENCE_CANDIDATES:
        if candidate.exists():
            return candidate
    candidates = "\n".join(str(path) for path in REFERENCE_CANDIDATES)
    raise FileNotFoundError(
        "Could not find an existing aligned validation-predictions file. "
        "Checked:\n" + candidates
    )


def normalise_columns(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Normalise known column aliases to date/horizon/model/prediction/actual_return."""
    aliases = {
        "date": ["date", "prediction_date", "session_date"],
        "horizon": ["horizon", "forecast_horizon", "horizon_days"],
        "model": ["model", "model_name", "model_type"],
        "prediction": ["prediction", "predicted_return", "y_pred", "forecast"],
        "actual_return": ["actual_return", "actual", "target", "y_true", "realised_return"],
    }
    lower_to_original = {str(column).lower(): column for column in frame.columns}
    rename = {}
    missing = []
    for canonical, options in aliases.items():
        match = next((lower_to_original[name] for name in options if name in lower_to_original), None)
        if match is None:
            missing.append(canonical)
        else:
            rename[match] = canonical
    if missing:
        raise ValueError(
            f"{label} is missing required columns {missing}. "
            f"Available columns: {list(frame.columns)}"
        )

    result = frame.rename(columns=rename).copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    result["horizon"] = pd.to_numeric(result["horizon"], errors="raise").astype(int)
    result["model"] = result["model"].astype(str)
    result["prediction"] = pd.to_numeric(result["prediction"], errors="coerce")
    result["actual_return"] = pd.to_numeric(result["actual_return"], errors="coerce")
    result = result.loc[
        result["horizon"].isin([1, 5, 10])
        & np.isfinite(result["prediction"])
        & np.isfinite(result["actual_return"])
    ].copy()
    if result.empty:
        raise ValueError(f"{label} contains no usable validation predictions.")
    return result


def assert_unique_model_keys(frame: pd.DataFrame, label: str) -> None:
    duplicate_mask = frame.duplicated(["date", "horizon", "model"], keep=False)
    if duplicate_mask.any():
        examples = frame.loc[duplicate_mask, ["date", "horizon", "model"]].head(10)
        raise ValueError(
            f"{label} has duplicate date/horizon/model rows:\n{examples.to_string(index=False)}"
        )


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, preserving original order."""
    values = np.asarray(p_values, dtype=float)
    if not len(values):
        return []
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running_max = 0.0
    m = len(values)
    for rank, idx in enumerate(order):
        candidate = (m - rank) * values[idx]
        running_max = max(running_max, candidate)
        adjusted[idx] = min(1.0, running_max)
    return adjusted.tolist()


def hac_mean_test(values: np.ndarray, max_lag: int) -> tuple[float, float, float]:
    """Newey-West HAC test of a paired mean difference against zero."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        return np.nan, np.nan, np.nan

    mean_diff = float(values.mean())
    demeaned = values - mean_diff
    lag = min(max_lag, n - 1)

    # Long-run variance estimate using Bartlett weights.
    long_run_variance = float(np.dot(demeaned, demeaned) / n)
    for k in range(1, lag + 1):
        covariance = float(np.dot(demeaned[k:], demeaned[:-k]) / n)
        weight = 1.0 - k / (lag + 1.0)
        long_run_variance += 2.0 * weight * covariance

    long_run_variance = max(long_run_variance, 0.0)
    standard_error = float(np.sqrt(long_run_variance / n))
    if standard_error == 0:
        if mean_diff == 0:
            return mean_diff, 0.0, 1.0
        return mean_diff, np.sign(mean_diff) * np.inf, 0.0

    statistic = mean_diff / standard_error
    p_value = float(2.0 * norm.sf(abs(statistic)))
    return mean_diff, float(statistic), p_value


def circular_block_bootstrap_ci(
    values: np.ndarray,
    block_length: int,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    """Percentile CI for a mean difference using circular moving blocks."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        return np.nan, np.nan
    block_length = max(1, min(int(block_length), n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_length))
    offsets = np.arange(block_length)
    means = np.empty(replicates, dtype=float)

    for b in range(replicates):
        starts = rng.integers(0, n, size=n_blocks)
        indices = np.concatenate([(start + offsets) % n for start in starts])[:n]
        means[b] = values[indices].mean()

    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def directional_correct(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Use the same strict sign-equality convention for every model."""
    return np.sign(actual) == np.sign(predicted)


def calculate_metrics(frame: pd.DataFrame) -> dict:
    """Calculate regression metrics and directional accuracy with coverage."""
    actual = frame["actual_return"].to_numpy(dtype=float)
    predicted = frame["prediction"].to_numpy(dtype=float)
    predicted_direction = np.sign(predicted)
    actual_direction = np.sign(actual)

    # Directional accuracy is conditional on the model issuing a non-zero
    # directional forecast. Coverage reports the fraction of forecasts that do.
    directional_mask = predicted_direction != 0
    coverage = float(directional_mask.mean()) if len(directional_mask) else np.nan
    if directional_mask.any():
        directional_accuracy = float(
            np.mean(actual_direction[directional_mask] == predicted_direction[directional_mask])
        )
    else:
        directional_accuracy = np.nan

    if len(actual) and len(predicted):
        mae = float(mean_absolute_error(actual, predicted))
        rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
        r_squared = float(r2_score(actual, predicted)) if len(actual) >= 2 else np.nan
    else:
        mae = rmse = r_squared = np.nan

    if len(actual) >= 2 and np.unique(actual).size > 1 and np.unique(predicted).size > 1:
        correlation = float(spearmanr(actual, predicted).statistic)
    else:
        correlation = np.nan

    return {
        "sample_size": int(len(frame)),
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy": directional_accuracy,
        "directional_coverage": coverage,
        "spearman_correlation": correlation,
        "r_squared": r_squared,
    }


def main() -> None:
    if not LSTM_PREDICTIONS.exists():
        raise FileNotFoundError(f"LSTM predictions not found: {LSTM_PREDICTIONS}")
    reference_path = find_reference_file()

    reference = normalise_columns(pd.read_csv(reference_path), "Existing aligned predictions")
    lstm = normalise_columns(pd.read_csv(LSTM_PREDICTIONS), "LSTM predictions")
    assert_unique_model_keys(reference, "Existing aligned predictions")
    assert_unique_model_keys(lstm, "LSTM predictions")

    if (reference["model"].str.lower() == "lstm_regression").any():
        raise ValueError("Reference file already contains lstm_regression rows.")

    # Use only date/horizon pairs represented in both sources.
    reference_keys = reference[["date", "horizon"]].drop_duplicates()
    lstm_keys = lstm[["date", "horizon"]].drop_duplicates()
    common_keys = reference_keys.merge(
        lstm_keys, on=["date", "horizon"], how="inner", validate="one_to_one"
    )
    if common_keys.empty:
        raise ValueError("No common date/horizon observations between LSTM and reference predictions.")

    reference_common = reference.merge(
        common_keys, on=["date", "horizon"], how="inner", validate="many_to_one"
    )
    lstm_common = lstm.merge(
        common_keys, on=["date", "horizon"], how="inner", validate="one_to_one"
    )

    # Verify the realised return is consistent across models and sources.
    reference_actuals = reference_common.groupby(["date", "horizon"])["actual_return"].agg(["min", "max"])
    if ((reference_actuals["max"] - reference_actuals["min"]).abs() > ALIGNMENT_TOLERANCE).any():
        raise ValueError("Existing models disagree on actual returns for aligned date/horizon keys.")

    canonical_actuals = (
        reference_common.groupby(["date", "horizon"], as_index=False)["actual_return"].first()
    )
    lstm_checked = lstm_common.merge(
        canonical_actuals,
        on=["date", "horizon"],
        how="left",
        suffixes=("_lstm", "_reference"),
        validate="one_to_one",
    )
    mismatch = (
        lstm_checked["actual_return_lstm"] - lstm_checked["actual_return_reference"]
    ).abs() > ALIGNMENT_TOLERANCE
    if mismatch.any():
        example = lstm_checked.loc[mismatch, ["date", "horizon", "actual_return_lstm", "actual_return_reference"]].head()
        raise ValueError(
            "LSTM actual returns do not match the reference targets:\n"
            + example.to_string(index=False)
        )

    existing_models = sorted(reference_common["model"].unique().tolist())
    aligned = pd.concat(
        [
            reference_common[["date", "horizon", "model", "prediction", "actual_return"]],
            lstm_common[["date", "horizon", "model", "prediction", "actual_return"]],
        ],
        ignore_index=True,
    )
    assert_unique_model_keys(aligned, "Combined aligned predictions")

    metrics_records = []
    for (horizon, model), group in aligned.groupby(["horizon", "model"], sort=True):
        metrics_records.append(
            {"horizon": int(horizon), "model": model, **calculate_metrics(group)}
        )
    metrics = pd.DataFrame(metrics_records)

    # Paired loss-difference tests: negative differences favour LSTM.
    test_records = []
    bootstrap_records = []
    for horizon in sorted(aligned["horizon"].unique()):
        subset = aligned.loc[aligned["horizon"] == horizon]
        wide_prediction = subset.pivot(index="date", columns="model", values="prediction")
        wide_actual = subset.pivot(index="date", columns="model", values="actual_return")
        # Every model's actuals have already been checked; use one consistent series.
        actual = wide_actual.iloc[:, 0].to_numpy(dtype=float)
        dates = wide_actual.index

        if "lstm_regression" not in wide_prediction.columns:
            raise AssertionError("LSTM predictions missing after alignment.")
        lstm_pred = wide_prediction["lstm_regression"].to_numpy(dtype=float)
        competitors = [m for m in wide_prediction.columns if m != "lstm_regression"]

        for competitor in competitors:
            competitor_pred = wide_prediction[competitor].to_numpy(dtype=float)
            loss_pairs = {
                "absolute_error": (
                    np.abs(lstm_pred - actual) - np.abs(competitor_pred - actual)
                ),
                "squared_error": (
                    (lstm_pred - actual) ** 2 - (competitor_pred - actual) ** 2
                ),
            }

            # A model that always predicts zero has no directional opinion.
            # Keep it for error metrics, but do not run a directional superiority
            # test against a baseline with no positive/negative signal. For other
            # competitors, directional tests below use common non-zero forecasts.
            if competitor != "zero_return":
                # Compare directional errors only where both models issue a
                # non-zero forecast. Otherwise a model abstaining with a zero
                # forecast would be unfairly scored as directionally incorrect.
                common_direction_mask = (lstm_pred != 0) & (competitor_pred != 0)
                directional_differences = (
                    (~directional_correct(actual[common_direction_mask], lstm_pred[common_direction_mask])).astype(float)
                    - (~directional_correct(actual[common_direction_mask], competitor_pred[common_direction_mask])).astype(float)
                )
                if common_direction_mask.any():
                    loss_pairs["directional_error"] = directional_differences

            for loss_name, differences in loss_pairs.items():
                mean_diff, statistic, p_value = hac_mean_test(
                    differences, HAC_LAGS.get(int(horizon), int(horizon) - 1)
                )
                test_records.append(
                    {
                        "horizon": int(horizon),
                        "comparison": f"lstm_regression_vs_{competitor}",
                        "competitor": competitor,
                        "loss": loss_name,
                        "sample_size": int(len(differences)),
                        "mean_loss_difference_lstm_minus_competitor": mean_diff,
                        "hac_lag": HAC_LAGS.get(int(horizon), int(horizon) - 1),
                        "hac_statistic": statistic,
                        "p_value": p_value,
                    }
                )

                block_length = BOOTSTRAP_BLOCK_LENGTHS.get(int(horizon), int(horizon))
                ci_low, ci_high = circular_block_bootstrap_ci(
                    differences,
                    block_length=block_length,
                    seed=RANDOM_SEED + int(horizon) + len(bootstrap_records),
                )
                bootstrap_records.append(
                    {
                        "horizon": int(horizon),
                        "comparison": f"lstm_regression_vs_{competitor}",
                        "competitor": competitor,
                        "loss": loss_name,
                        "sample_size": int(len(differences)),
                        "mean_loss_difference_lstm_minus_competitor": float(np.mean(differences)),
                        "block_length": block_length,
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                        "ci_95_lower": ci_low,
                        "ci_95_upper": ci_high,
                    }
                )

    tests = pd.DataFrame(test_records)
    if not tests.empty:
        tests["p_value_holm"] = holm_adjust(tests["p_value"].fillna(1.0).tolist())
        tests["significant_holm_0_05"] = tests["p_value_holm"] < 0.05

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    aligned_path = COMPARISON_DIR / "all_models_including_lstm_aligned_predictions.csv"
    metrics_path = COMPARISON_DIR / "all_models_including_lstm_comparison_metrics.csv"
    tests_path = COMPARISON_DIR / "lstm_paired_hac_tests.csv"
    bootstrap_path = COMPARISON_DIR / "lstm_block_bootstrap_robustness.csv"
    metadata_path = COMPARISON_DIR / "lstm_comparison_run.json"

    aligned.to_csv(aligned_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    tests.to_csv(tests_path, index=False)
    pd.DataFrame(bootstrap_records).to_csv(bootstrap_path, index=False)

    metadata = {
        "ticker": "LLOY.L",
        "reference_predictions_file": str(reference_path.relative_to(ROOT)),
        "lstm_predictions_file": str(LSTM_PREDICTIONS.relative_to(ROOT)),
        "models_in_reference": existing_models,
        "models_in_combined_comparison": sorted(aligned["model"].unique().tolist()),
        "common_date_horizon_observations": int(len(common_keys)),
        "combined_prediction_rows": int(len(aligned)),
        "horizons": sorted(int(h) for h in aligned["horizon"].unique()),
        "hac_lags_by_horizon": HAC_LAGS,
        "directional_tests_exclude_zero_return_baseline": True,
        "paired_directional_tests_use_common_nonzero_forecasts": True,
        "directional_accuracy_definition": "Conditional on non-zero predicted direction; sign equality against realised return.",
        "directional_coverage_definition": "Fraction of forecasts with a non-zero predicted direction.",
        "bootstrap_block_lengths_by_horizon": BOOTSTRAP_BLOCK_LENGTHS,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "holm_correction_applied": True,
        "final_test_data_used": False,
        "backtesting_performed": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nLSTM ALIGNED MODEL COMPARISON COMPLETED")
    print("=" * 88)
    print(f"Reference file: {reference_path}")
    print(f"Common date/horizon observations: {len(common_keys)}")
    print(f"Models included: {', '.join(metadata['models_in_combined_comparison'])}")
    print(f"Combined prediction rows: {len(aligned)}")
    print("\nMetrics by horizon and model:")
    print(metrics.sort_values(["horizon", "mae"]).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nLSTM paired HAC tests (negative loss difference favours LSTM):")
    print(
        tests.sort_values(["horizon", "loss", "p_value_holm"])
        .to_string(index=False, float_format=lambda x: f"{x:.6g}")
    )
    print("\nOutput files:")
    for path in [aligned_path, metrics_path, tests_path, bootstrap_path, metadata_path]:
        print(path)


if __name__ == "__main__":
    main()
