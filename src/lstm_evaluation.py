"""Walk-forward evaluation for LSTM return forecasts."""

from __future__ import annotations

from datetime import date
from typing import Iterable, Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.baseline_evaluation import _calculate_metrics
from src.feature_target_pipeline import FEATURE_COLUMNS
from src.lstm_model import (
    DEFAULT_LSTM_PARAMETERS,
    create_lstm_regression_model,
)
from src.temporal_split import get_training_eligibility_mask
from src.walk_forward_evaluation import (
    SUPPORTED_HORIZONS,
    _as_date,
    _validate_inputs,
    _weekly_cutoffs,
)


MODEL_NAME = "lstm_regression"

DEFAULT_PARAMETERS = {
    **DEFAULT_LSTM_PARAMETERS,
    "learning_rate": 0.001,
    "epochs": 20,
    "batch_size": 32,
}

DEVICE = torch.device("cpu")


def _validate_training_parameters(
    *,
    sequence_length: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    """Validate LSTM training configuration."""

    for name, value in (
        ("sequence_length", sequence_length),
        ("epochs", epochs),
        ("batch_size", batch_size),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer.")

    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not np.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError(
            "learning_rate must be a positive finite number."
        )


def _prepare_sequences(
    scaled_features: np.ndarray,
    sequence_length: int,
) -> np.ndarray:
    """Build chronological sequences ending at each row."""

    if scaled_features.ndim != 2:
        raise ValueError("scaled_features must be two-dimensional.")

    if len(scaled_features) < sequence_length:
        return np.empty(
            (0, sequence_length, scaled_features.shape[1]),
            dtype=np.float32,
        )

    sequences = []
    for end_position in range(sequence_length - 1, len(scaled_features)):
        window = scaled_features[
            end_position - sequence_length + 1 : end_position + 1
        ]

        if np.isfinite(window).all():
            sequences.append(window)

    if not sequences:
        return np.empty(
            (0, sequence_length, scaled_features.shape[1]),
            dtype=np.float32,
        )

    return np.asarray(sequences, dtype=np.float32)


def _fit_lstm(
    sequences: np.ndarray,
    targets: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    random_state: int,
) -> torch.nn.Module:
    """Fit an LSTM using deterministic chronological mini-batches."""

    if len(sequences) != len(targets):
        raise ValueError(
            "Training sequences and targets must have equal lengths."
        )

    if len(sequences) == 0:
        raise ValueError("Cannot train an LSTM without sequences.")

    torch.manual_seed(random_state)

    model = create_lstm_regression_model(
        input_size=sequences.shape[2],
        random_state=random_state,
    ).to(DEVICE)

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )
    loss_function = nn.MSELoss()

    x_train = torch.as_tensor(
        sequences,
        dtype=torch.float32,
        device=DEVICE,
    )
    y_train = torch.as_tensor(
        targets,
        dtype=torch.float32,
        device=DEVICE,
    )

    # Use chronological batches without shuffling observations.
    model.train()

    for _ in range(epochs):
        for start in range(0, len(x_train), batch_size):
            end = min(start + batch_size, len(x_train))

            batch_x = x_train[start:end]
            batch_y = y_train[start:end]

            optimiser.zero_grad()
            predictions = model(batch_x)
            loss = loss_function(predictions, batch_y)
            loss.backward()
            optimiser.step()

    return model


def _predict_lstm(
    model: torch.nn.Module,
    sequence: np.ndarray,
) -> float:
    """Generate one finite prediction from a single sequence."""

    model.eval()

    inputs = torch.as_tensor(
        sequence[np.newaxis, :, :],
        dtype=torch.float32,
        device=DEVICE,
    )

    with torch.no_grad():
        prediction = float(model(inputs).cpu().item())

    if not np.isfinite(prediction):
        raise ValueError("LSTM produced a non-finite prediction.")

    return prediction


def evaluate_lstm_walk_forward(
    *,
    history: pd.DataFrame,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    maturity_dates: pd.DataFrame,
    development_start: date,
    development_end: date,
    validation_start: date,
    validation_end: date,
    horizons: Iterable[int] = SUPPORTED_HORIZONS,
    min_training_samples: int = 500,
    sequence_length: int = DEFAULT_PARAMETERS["sequence_length"],
    epochs: int = DEFAULT_PARAMETERS["epochs"],
    batch_size: int = DEFAULT_PARAMETERS["batch_size"],
    learning_rate: float = DEFAULT_PARAMETERS["learning_rate"],
    random_state: int = DEFAULT_PARAMETERS["random_state"],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate LSTM forecasts using weekly expanding-window retraining.

    Training labels must have matured by each retraining cutoff.
    Validation targets are scored only when their maturity dates do not
    exceed validation_end. Final Test observations must not be supplied.
    """

    _validate_inputs(
        history,
        features,
        targets,
        maturity_dates,
    )

    _validate_training_parameters(
        sequence_length=sequence_length,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    if (
        isinstance(random_state, bool)
        or not isinstance(random_state, int)
        or random_state < 0
    ):
        raise ValueError(
            "random_state must be a non-negative integer."
        )

    development_start = _as_date(development_start)
    development_end = _as_date(development_end)
    validation_start = _as_date(validation_start)
    validation_end = _as_date(validation_end)

    if development_start > development_end:
        raise ValueError(
            "development_start must not be after development_end."
        )

    if validation_start > validation_end:
        raise ValueError(
            "validation_start must not be after validation_end."
        )

    if development_end >= validation_start:
        raise ValueError(
            "Development and validation periods must not overlap."
        )

    if (
        isinstance(min_training_samples, bool)
        or not isinstance(min_training_samples, int)
        or min_training_samples < 1
    ):
        raise ValueError(
            "min_training_samples must be a positive integer."
        )

    horizons = tuple(horizons)

    if (
        not horizons
        or any(
            isinstance(h, bool)
            or not isinstance(h, (int, np.integer))
            or h not in SUPPORTED_HORIZONS
            for h in horizons
        )
    ):
        raise ValueError(
            f"horizons must be a non-empty subset of "
            f"{SUPPORTED_HORIZONS}."
        )

    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons must not contain duplicates.")

    date_values = history["date"].map(_as_date)

    if not date_values.is_monotonic_increasing:
        raise ValueError(
            "history must be sorted in ascending date order."
        )

    feature_values = (
        features.loc[:, FEATURE_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
    )

    feature_array = feature_values.to_numpy(dtype=float)
    finite_features = np.isfinite(feature_array).all(axis=1)

    in_scope = date_values.map(
        lambda d: development_start <= d <= validation_end
    )

    cutoffs = _weekly_cutoffs(
        date_values.loc[in_scope].tolist()
    )

    validation_dates = sorted(
        {
            d
            for d in date_values
            if validation_start <= d <= validation_end
        }
    )

    if not validation_dates:
        raise ValueError(
            "No observations are available in the validation period."
        )

    schedule: dict[date, date] = {}

    for session_date in validation_dates:
        prior_cutoffs = [
            cutoff for cutoff in cutoffs
            if cutoff <= session_date
        ]

        if not prior_cutoffs:
            raise ValueError(
                f"No retraining cutoff is available for {session_date}."
            )

        schedule[session_date] = prior_cutoffs[-1]

    # Cache one fitted model per horizon and weekly cutoff.
    fitted_cache: dict[
        tuple[int, date],
        tuple[Any, StandardScaler, int],
    ] = {}

    prediction_records: list[dict[str, Any]] = []

    for horizon in horizons:
        target_column = f"forward_return_{horizon}d"
        maturity_column = f"target_maturity_date_{horizon}d"

        if target_column not in targets.columns:
            raise ValueError(
                f"targets is missing {target_column!r}."
            )

        if maturity_column not in maturity_dates.columns:
            raise ValueError(
                f"maturity_dates is missing {maturity_column!r}."
            )

        y_all = pd.to_numeric(
            targets[target_column],
            errors="coerce",
        )

        target_array = y_all.to_numpy(dtype=float)
        finite_targets = np.isfinite(target_array)

        for session_date in validation_dates:
            cutoff_date = schedule[session_date]
            cache_key = (int(horizon), cutoff_date)

            if cache_key not in fitted_cache:
                eligibility = get_training_eligibility_mask(
                    history,
                    maturity_dates,
                    development_start,
                    cutoff_date,
                    cutoff_date,
                )[maturity_column].astype(bool)

                eligible = (
                    eligibility.to_numpy(dtype=bool)
                    & finite_features
                    & finite_targets
                )

                # Only observations whose labels are eligible at the
                # cutoff may contribute training endpoints.
                eligible_positions = np.flatnonzero(eligible)

                eligible_positions = eligible_positions[
                    date_values.iloc[eligible_positions]
                    .map(lambda d: d <= cutoff_date)
                    .to_numpy()
                ]

                # A sequence is usable only if all 20 feature rows
                # ending at its target observation are finite.
                training_positions = []

                for position in eligible_positions:
                    start_position = position - sequence_length + 1

                    if start_position < 0:
                        continue

                    window = feature_array[
                        start_position : position + 1
                    ]

                    if len(window) == sequence_length and np.isfinite(
                        window
                    ).all():
                        training_positions.append(position)

                if len(training_positions) < min_training_samples:
                    raise ValueError(
                        "Insufficient eligible LSTM training sequences "
                        f"for horizon {horizon}d at cutoff {cutoff_date}: "
                        f"{len(training_positions)}; minimum required is "
                        f"{min_training_samples}."
                    )

                # Fit the scaler only on feature rows that actually occur
                # inside eligible training sequences. This keeps the scaling
                # distribution tied to the current training fold while using
                # only historical feature rows available by this cutoff.
                training_window_positions = sorted(
                    {
                        window_position
                        for endpoint in training_positions
                        for window_position in range(
                            endpoint - sequence_length + 1,
                            endpoint + 1,
                        )
                    }
                )

                # Defensive invariant: no feature row after the retraining
                # cutoff may contribute to scaler fitting or model training.
                if any(
                    date_values.iloc[position] > cutoff_date
                    for position in training_window_positions
                ):
                    raise RuntimeError(
                        "A training sequence contains features after its cutoff."
                    )

                scaler = StandardScaler()
                scaler.fit(feature_array[training_window_positions])

                scaled_features = np.full_like(
                    feature_array,
                    np.nan,
                    dtype=np.float64,
                )

                scaled_features[finite_features] = scaler.transform(
                    feature_array[finite_features]
                )

                training_sequences = np.stack(
                    [
                        scaled_features[
                            position - sequence_length + 1 : position + 1
                        ]
                        for position in training_positions
                    ]
                ).astype(np.float32)

                training_targets = target_array[
                    training_positions
                ].astype(np.float32)

                model = _fit_lstm(
                    training_sequences,
                    training_targets,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    random_state=random_state,
                )

                fitted_cache[cache_key] = (
                    model,
                    scaler,
                    len(training_positions),
                )

            model, scaler, training_size = fitted_cache[cache_key]

            matching_positions = np.flatnonzero(
                date_values.to_numpy() == session_date
            )

            if len(matching_positions) != 1:
                raise ValueError(
                    f"Expected exactly one row for {session_date}."
                )

            position = int(matching_positions[0])

            if position < sequence_length - 1:
                continue

            start_position = position - sequence_length + 1
            window = feature_array[start_position : position + 1]

            if len(window) != sequence_length or not np.isfinite(
                window
            ).all():
                continue

            actual = y_all.iloc[position]
            maturity = maturity_dates.iloc[position][maturity_column]

            if pd.isna(actual) or not np.isfinite(float(actual)):
                continue

            if pd.isna(maturity):
                continue

            if _as_date(maturity) > validation_end:
                continue

            scaled_window = scaler.transform(window).astype(
                np.float32
            )

            prediction = _predict_lstm(
                model,
                scaled_window,
            )

            prediction_records.append(
                {
                    "date": session_date,
                    "cutoff_date": cutoff_date,
                    "horizon": int(horizon),
                    "model": MODEL_NAME,
                    "prediction": prediction,
                    "actual_return": float(actual),
                    "training_sample_size": training_size,
                }
            )

    predictions = pd.DataFrame(prediction_records)

    if predictions.empty:
        raise ValueError(
            "No matured validation targets are available for LSTM evaluation."
        )

    metric_records = []

    for horizon, group in predictions.groupby(
        "horizon",
        sort=True,
    ):
        metrics = _calculate_metrics(
            actual=group["actual_return"],
            predicted=group["prediction"],
        )

        metric_records.append(
            {
                "horizon": int(horizon),
                "model": MODEL_NAME,
                **metrics,
            }
        )

    return predictions, pd.DataFrame(metric_records)