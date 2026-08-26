"""
train.py
Day 3: builds the 3-day-ahead AQI prediction target, does a time-respecting
train/test split, and trains + evaluates baseline models (Ridge, Random Forest).

CRITICAL: this is a time series. The target for each row is the AQI value
72 hours *after* that row's timestamp - not the row's own AQI. And the
train/test split must be chronological (train on the past, test on the
most recent chunk), never a random shuffle, or the model gets to peek
at "future" patterns during training that it wouldn't have in production.
"""

import os

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from xgboost import XGBRegressor

import feature_store as fs

FORECAST_HORIZON_HOURS = 72  # predicting AQI 3 days ahead

# Columns that shouldn't be fed to the model as features
NON_FEATURE_COLUMNS = ["timestamp_utc", "station_id", "target_aqi_72h"]


FUTURE_WEATHER_COLS = ["temperature_c", "humidity_pct", "pressure_hpa", "wind_speed", "rain"]


def add_future_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the ACTUAL weather at T+72h as a feature - a stand-in for a real
    3-day weather forecast, which would genuinely be available at
    prediction time in production. This is a documented approximation:
    it assumes forecast accuracy equal to the true observed weather, so
    reported metrics are an upper bound on what a real deployed system
    (using an actual forecast API, not perfect hindsight) would achieve.
    State this assumption explicitly in the report.
    """
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    for col in FUTURE_WEATHER_COLS:
        df[f"{col}_forecast_72h"] = df[col].shift(-FORECAST_HORIZON_HOURS)
    return df


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add target_aqi_72h: the aqi value FORECAST_HORIZON_HOURS after each
    row. Rows near the end of the dataset have no future value to look up
    yet, so their target is NaN - these get dropped before training since
    you can't train on a label that doesn't exist."""
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    df["target_aqi_72h"] = df["aqi"].shift(-FORECAST_HORIZON_HOURS)
    return df


LAG_WINDOWS_HOURS = [24, 48, 72]


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling-window and lag features so the model sees recent trend,
    not just a single frozen snapshot. All windows/lags look BACKWARD only
    (rolling(window).mean(), shift(N) with positive N) - never forward,
    which would leak the future into a feature.
    """
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    for window in LAG_WINDOWS_HOURS:
        df[f"aqi_roll_mean_{window}h"] = df["aqi"].rolling(window, min_periods=1).mean()
        df[f"aqi_roll_max_{window}h"] = df["aqi"].rolling(window, min_periods=1).max()
        df[f"aqi_roll_min_{window}h"] = df["aqi"].rolling(window, min_periods=1).min()
        df[f"pm25_roll_mean_{window}h"] = df["pm25"].rolling(window, min_periods=1).mean()

    # Same hour, previous day / 2 days ago - captures daily cycle directly
    df["aqi_lag_24h"] = df["aqi"].shift(24)
    df["aqi_lag_48h"] = df["aqi"].shift(48)

    return df


def time_based_split(df: pd.DataFrame, test_size_days: int = 60):
    """Chronological split: the last `test_size_days` worth of rows become
    the test set, everything before is training. Never shuffle - shuffling
    a time series lets the model train on data adjacent to (sometimes
    literally surrounding) what it's tested on, which inflates every
    metric and hides how the model will actually perform on new data."""
    cutoff = df["timestamp_utc"].max() - pd.Timedelta(days=test_size_days)
    train_df = df[df["timestamp_utc"] < cutoff].copy()
    test_df = df[df["timestamp_utc"] >= cutoff].copy()
    return train_df, test_df


def get_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


class AqiMLP(nn.Module):
    """Small feedforward network for tabular AQI features. Unlike the tree
    models, a neural net needs SCALED inputs (StandardScaler) - raw AQI
    values in the hundreds alongside 0/1 flags like is_weekend would
    otherwise dominate the loss purely from magnitude, not signal."""

    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_pytorch_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
):
    """Trains AqiMLP on CPU. No GPU needed - this dataset (tens of
    thousands of rows, ~40 features) is small enough that a handful of
    epochs over mini-batches finishes in well under a minute on CPU."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)

    model = AqiMLP(n_features=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n_samples = X_train_t.shape[0]

    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(n_samples)
        epoch_loss = 0.0

        for i in range(0, n_samples, batch_size):
            idx = permutation[i:i + batch_size]
            batch_x, batch_y = X_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = loss_fn(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1}/{epochs}  train_mse={epoch_loss / n_samples:.3f}")

    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t).numpy()

    return test_preds


def evaluate(y_true, y_pred, label: str) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{label:20s}  RMSE={rmse:7.3f}  MAE={mae:7.3f}  R2={r2:6.4f}")
    return {"model": label, "rmse": rmse, "mae": mae, "r2": r2}


import joblib

MODEL_OUTPUT_PATH = "model_registry/xgboost_aqi_model.joblib"
SCALER_OUTPUT_PATH = "model_registry/feature_columns.joblib"


def save_model(model, feature_cols: list, path: str = MODEL_OUTPUT_PATH):
    """Save the trained model plus the exact feature column order it
    expects. Saving feature_cols alongside the model matters: if
    fetch_data.py's schema ever changes, whatever loads this model later
    (the dashboard, a CI/CD retrain job) needs to know the exact column
    order the model was trained on, not just the model weights."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    joblib.dump(feature_cols, SCALER_OUTPUT_PATH)
    print(f"Saved model to {path}")
    print(f"Saved feature column list to {SCALER_OUTPUT_PATH}")


def main():
    print("Loading features from the store...")
    df = fs.read()
    print(f"Loaded {len(df)} rows")

    df = add_lag_features(df)
    print(f"Added lag/rolling features (windows: {LAG_WINDOWS_HOURS}h, plus 24h/48h lags)")

    df = add_future_weather_features(df)
    print(f"Added future-weather features (stand-in for a 3-day forecast): {FUTURE_WEATHER_COLS}")

    df = build_target(df)
    before_drop = len(df)
    df = df.dropna(subset=["target_aqi_72h"]).reset_index(drop=True)
    print(f"Dropped {before_drop - len(df)} rows with no future target "
          f"(most recent {FORECAST_HORIZON_HOURS}h - can't have a 3-day-ahead "
          f"label for them yet)")

    # Also drop any remaining NaN feature rows (e.g. the single first-row
    # aqi_change_rate NaN from feature engineering) - Ridge/RF can't
    # handle NaN inputs.
    feature_cols = get_feature_columns(df)
    before_drop = len(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    print(f"Dropped {before_drop - len(df)} rows with missing feature values")

    train_df, test_df = time_based_split(df, test_size_days=60)
    print(f"\nTrain: {len(train_df)} rows ({train_df['timestamp_utc'].min()} to {train_df['timestamp_utc'].max()})")
    print(f"Test:  {len(test_df)} rows ({test_df['timestamp_utc'].min()} to {test_df['timestamp_utc'].max()})")

    X_train = train_df[feature_cols]
    y_train = train_df["target_aqi_72h"]
    X_test = test_df[feature_cols]
    y_test = test_df["target_aqi_72h"]

    print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")

    results = []

    # Naive baseline: "AQI in 3 days = AQI right now." If the trained
    # models can't clearly beat this trivial guess, the problem isn't
    # feature engineering - it's that the target is close to unpredictable
    # at this horizon with the signal available. If they DO beat it
    # substantially, that's the real evidence the models are learning
    # something, even if the raw R2 number looks modest in isolation.
    print("\nNaive persistence baseline (predict current AQI unchanged)...")
    persistence_preds = test_df["aqi"].values
    results.append(evaluate(y_test, persistence_preds, "Persistence"))

    print("\nTraining Ridge Regression...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_preds = ridge.predict(X_test)
    results.append(evaluate(y_test, ridge_preds, "Ridge"))

    print("\nTraining Random Forest...")
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    results.append(evaluate(y_test, rf_preds, "Random Forest"))

    print("\nTraining XGBoost...")
    xgb = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    results.append(evaluate(y_test, xgb_preds, "XGBoost"))

    # Save the winning model now, right after training - if a later step
    # in this script errors out, the model you actually want is already
    # safely on disk rather than lost with the Python session.
    save_model(xgb, feature_cols)

    print("\nTraining PyTorch MLP...")
    mlp_preds = train_pytorch_model(X_train, y_train, X_test, y_test)
    results.append(evaluate(y_test, mlp_preds, "PyTorch MLP"))

    results_df = pd.DataFrame(results)
    results_df.to_csv("model_results.csv", index=False)
    print("\nSaved results to model_results.csv")

    return train_df, test_df, feature_cols, ridge, rf, xgb


if __name__ == "__main__":
    main()