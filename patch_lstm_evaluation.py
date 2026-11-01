"""Apply a focused temporal-scaling correction to src/lstm_evaluation.py.

Run from the Quantitative-Trading repository root. Creates a timestamp-free
backup named src/lstm_evaluation.py.pre_temporal_scaling.bak and refuses to
apply twice.
"""
from pathlib import Path
import shutil

path = Path("src/lstm_evaluation.py")
if not path.is_file():
    raise SystemExit(
        "Run this script from the Quantitative-Trading repository root; "
        "src/lstm_evaluation.py was not found."
    )

source = path.read_text(encoding="utf-8")
old = '''                # Fit the scaler only on eligible training endpoints.
                scaler = StandardScaler()
                scaler.fit(
                    feature_array[training_positions]
                )

                scaled_features = np.full_like(
'''
new = '''                # Fit the scaler only on feature rows that actually occur
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
'''
if new in source:
    raise SystemExit("The temporal-scaling correction is already present; no changes made.")
if old not in source:
    raise SystemExit(
        "Could not find the expected scaler block. No changes made to avoid "
        "overwriting a different implementation."
    )

backup = path.with_name(path.name + ".pre_temporal_scaling.bak")
if backup.exists():
    raise SystemExit(f"Backup already exists: {backup}. No changes made.")

shutil.copy2(path, backup)
path.write_text(source.replace(old, new, 1), encoding="utf-8")
print(f"Updated: {path}")
print(f"Backup:  {backup}")