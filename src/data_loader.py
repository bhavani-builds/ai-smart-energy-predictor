"""Load and validate hourly energy consumption data."""

from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = [
    "timestamp",
    "energy_kwh",
    "temperature_c",
    "day_type",
]

VALID_DAY_TYPES = {"weekday", "weekend"}


def load_energy_data(path=None):
    """Load the CSV and validate its schema and values."""
    if path is None:
        path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "energy_consumption.csv"
        )
    else:
        path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Energy dataset not found: {path}")

    df = pd.read_csv(path)

    missing = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    df = df[EXPECTED_COLUMNS].copy()

    if df.empty:
        raise ValueError("Energy dataset is empty.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df["energy_kwh"] = pd.to_numeric(df["energy_kwh"], errors="raise")
    df["temperature_c"] = pd.to_numeric(df["temperature_c"], errors="raise")

    if df.isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    if df["timestamp"].duplicated().any():
        raise ValueError("Dataset contains duplicate timestamps.")

    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamps must be sorted in ascending order.")

    if (df["energy_kwh"] < 0).any():
        raise ValueError("Energy consumption cannot be negative.")

    if not df["day_type"].isin(VALID_DAY_TYPES).all():
        raise ValueError(
            f"day_type must contain only {sorted(VALID_DAY_TYPES)}."
        )

    return df


def prepare_features(df):
    """Create time-based and weather features for forecasting."""
    required = set(EXPECTED_COLUMNS)
    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")

    result = df.copy()
    timestamps = pd.to_datetime(result["timestamp"], errors="raise")

    result["hour"] = timestamps.dt.hour
    result["day_of_week"] = timestamps.dt.dayofweek
    result["month"] = timestamps.dt.month
    result["is_weekend"] = (timestamps.dt.dayofweek >= 5).astype(int)

    return result
