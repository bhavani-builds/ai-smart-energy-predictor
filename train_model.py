"""Train and evaluate the hourly energy consumption model."""

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_energy_data, prepare_features
from src.predictor import FEATURE_COLUMNS


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "energy_model.joblib"


def build_model():
    """Build a reproducible regression pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=6,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main():
    df = load_energy_data()
    featured = prepare_features(df)

    X = featured[FEATURE_COLUMNS]
    y = featured["energy_kwh"]

    # Keep later timestamps for testing to avoid random time-series leakage.
    split_index = int(len(featured) * 0.8)
    if split_index < 2 or len(featured) - split_index < 2:
        raise ValueError("Not enough records for a time-ordered train/test split.")

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5

    print(f"Training records: {len(X_train)}")
    print(f"Test records: {len(X_test)}")
    print(f"Mean absolute error (kWh): {mae:.4f}")
    print(f"Root mean squared error (kWh): {rmse:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
