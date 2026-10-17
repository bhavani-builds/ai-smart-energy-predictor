"""Prediction utilities for the smart energy forecasting project."""

from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "energy_model.joblib"

FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "temperature_c",
]


def load_model(model_path=None):
    """Load the trained energy forecasting pipeline."""
    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH

    if not path.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {path}. Run train_model.py first."
        )

    return joblib.load(path)


def predict_consumption(model, features):
    """Predict energy consumption from one or more feature records."""
    if isinstance(features, dict):
        features = pd.DataFrame([features])
    elif not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a dictionary or pandas DataFrame")

    missing = [column for column in FEATURE_COLUMNS if column not in features.columns]
    if missing:
        raise ValueError(f"Missing prediction features: {missing}")

    inputs = features[FEATURE_COLUMNS].copy()

    if inputs.isnull().any().any():
        raise ValueError("Prediction features cannot contain missing values.")

    predictions = model.predict(inputs)

    # Energy usage cannot be negative.
    predictions = [max(0.0, float(value)) for value in predictions]

    return predictions
