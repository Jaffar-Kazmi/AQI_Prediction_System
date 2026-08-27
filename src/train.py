"""
train.py
Builds AQI prediction targets for 24h, 48h, and 72h ahead (i.e. "tomorrow",
"the day after", "in 3 days"), does a time-respecting train/test split for
each horizon, and trains + evaluates baseline models (Ridge, Random Forest,
XGBoost, PyTorch MLP) per horizon.

CRITICAL: this is a time series. For a given horizon H, the target for each
row is the AQI value H hours *after* that row's timestamp - not the row's
own AQI. And the train/test split must be chronological (train on the past,
test on the most recent chunk), never a random shuffle, or the model gets to
peek at "future" patterns during training that it wouldn't have in
production.

Each horizon gets its own target column, its own future-weather stand-in
features (shifted by that horizon), its own train/test split, and its own
saved model - a 24h model and a 72h model are different models with
different feature values, not the same model evaluated at different points.
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

# One model trained per horizon: tomorrow, day-after-tomorrow, 3 days out.
FORECAST_HORIZONS_HOURS = [24, 48, 72]

# Columns that shouldn't be fed to the model as features. Populated per
# horizon inside main() since the target column name changes per horizon.
BASE_NON_FEATURE_COLUMNS = ["timestamp_utc", "station_id"]

FUTURE_WEATHER_COLS = ["temperature_c", "humidity_pct", "pressure_hpa", "wind_speed", "rain"]

LAG_WINDOWS_HOURS = [24, 48, 72]

MODEL_REGISTRY_DIR = "model_registry"


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling-window and lag features so the model sees recent trend,
    not just a single frozen snapshot. All windows/lags look BACKWARD only
    (rolling(window).mean(), shift(N) with positive N) - never forward,
    which would leak the future into a feature. These are horizon-independent,
    so they're computed once and reused for every horizon."""
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


def add_future_weather_features(df: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    """Add the ACTUAL weather at T+horizon as a feature - a stand-in for a
    real weather forecast at that horizon, which would genuinely be
    available at prediction time in production. This is a documented
    approximation: it assumes forecast accuracy equal to the true observed
    weather, so reported metrics are an upper bound on what a real deployed
    system (using an actual forecast API, not perfect hindsight) would
    achieve. State this assumption explicitly in the report.

    Column names are suffixed with the horizon (e.g. _forecast_24h vs
    _forecast_72h) so features for different horizons never collide if
    ever combined in the same dataframe.
    """
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    for col in FUTURE_WEATHER_COLS:
        df[f"{col}_forecast_{horizon_hours}h"] = df[col].shift(-horizon_hours)
    return df


def build_target(df: pd.DataFrame, horizon_hours: int) -> tuple[pd.DataFrame, str]:
    """Add target_aqi_{H}h: the aqi value `horizon_hours` after each row.
    Rows near the end of the dataset have no future value to look up yet,
    so their target is NaN - these get dropped before training since you
    can't train on a label that doesn't exist. Returns the modified df and
    the name of the target column that was added."""
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    target_col = f"target_aqi_{horizon_hours}h"
    df[target_col] = df["aqi"].shift(-horizon_hours)
    return df, target_col


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


def get_feature_columns(df: pd.DataFrame, non_feature_columns: list) -> list:
    return [c for c in df.columns if c not in non_feature_columns]


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
    verbose: bool = True,
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

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"    epoch {epoch + 1}/{epochs}  train_mse={epoch_loss / n_samples:.3f}")

    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t).numpy()

    return test_preds


def evaluate(y_true, y_pred, label: str, horizon_hours: int) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"  {label:20s}  RMSE={rmse:7.3f}  MAE={mae:7.3f}  R2={r2:6.4f}")
    return {"horizon_hours": horizon_hours, "model": label, "rmse": rmse, "mae": mae, "r2": r2}


import joblib


def save_model(model, feature_cols: list, horizon_hours: int):
    """Save the trained model plus the exact feature column order it
    expects, tagged with its horizon so a 24h model is never accidentally
    loaded where a 72h model is expected (or vice versa). Saving
    feature_cols alongside the model matters: if fetch_data.py's schema
    ever changes, whatever loads this model later (the dashboard, a
    CI/CD retrain job) needs to know the exact column order the model
    was trained on, not just the model weights."""
    os.makedirs(MODEL_REGISTRY_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_REGISTRY_DIR, f"xgboost_aqi_model_{horizon_hours}h.joblib")
    features_path = os.path.join(MODEL_REGISTRY_DIR, f"feature_columns_{horizon_hours}h.joblib")
    joblib.dump(model, model_path)
    joblib.dump(feature_cols, features_path)
    print(f"  Saved model to {model_path}")
    print(f"  Saved feature column list to {features_path}")


def run_for_horizon(df_with_lags: pd.DataFrame, horizon_hours: int) -> list:
    """Runs the full target-build -> split -> train -> evaluate -> save
    pipeline for a single forecast horizon. Returns a list of result dicts,
    one per model, so main() can concatenate results across horizons."""
    print(f"\n{'=' * 60}")
    print(f"HORIZON: {horizon_hours}h ahead ({horizon_hours / 24:.0f} day(s))")
    print(f"{'=' * 60}")

    df = add_future_weather_features(df_with_lags, horizon_hours)
    print(f"Added future-weather features (stand-in for a {horizon_hours}h forecast): "
          f"{FUTURE_WEATHER_COLS}")

    df, target_col = build_target(df, horizon_hours)
    before_drop = len(df)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    print(f"Dropped {before_drop - len(df)} rows with no future target "
          f"(most recent {horizon_hours}h - can't have a {horizon_hours}h-ahead "
          f"label for them yet)")

    non_feature_columns = BASE_NON_FEATURE_COLUMNS + [target_col]
    feature_cols = get_feature_columns(df, non_feature_columns)

    # Drop any remaining NaN feature rows (e.g. the single first-row
    # aqi_change_rate NaN from feature engineering) - Ridge/RF can't
    # handle NaN inputs.
    before_drop = len(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    print(f"Dropped {before_drop - len(df)} rows with missing feature values")

    train_df, test_df = time_based_split(df, test_size_days=60)
    print(f"Train: {len(train_df)} rows "
          f"({train_df['timestamp_utc'].min()} to {train_df['timestamp_utc'].max()})")
    print(f"Test:  {len(test_df)} rows "
          f"({test_df['timestamp_utc'].min()} to {test_df['timestamp_utc'].max()})")

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    results = []

    # Naive baseline: "AQI in H hours = AQI right now." If the trained
    # models can't clearly beat this trivial guess, the problem isn't
    # feature engineering - it's that the target is close to unpredictable
    # at this horizon with the signal available. If they DO beat it
    # substantially, that's the real evidence the models are learning
    # something, even if the raw R2 number looks modest in isolation.
    print("\nNaive persistence baseline (predict current AQI unchanged)...")
    persistence_preds = test_df["aqi"].values
    results.append(evaluate(y_test, persistence_preds, "Persistence", horizon_hours))

    print("Training Ridge Regression...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_preds = ridge.predict(X_test)
    results.append(evaluate(y_test, ridge_preds, "Ridge", horizon_hours))

    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    results.append(evaluate(y_test, rf_preds, "Random Forest", horizon_hours))

    print("Training XGBoost...")
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
    results.append(evaluate(y_test, xgb_preds, "XGBoost", horizon_hours))

    # Save the winning model now, right after training - if a later step
    # in this run errors out, the model you actually want is already
    # safely on disk rather than lost with the Python session.
    save_model(xgb, feature_cols, horizon_hours)

    print("Training PyTorch MLP...")
    mlp_preds = train_pytorch_model(X_train, y_train, X_test, y_test)
    results.append(evaluate(y_test, mlp_preds, "PyTorch MLP", horizon_hours))

    return results


def main():
    print("Loading features from the store...")
    df = fs.read()
    print(f"Loaded {len(df)} rows")

    df = add_lag_features(df)
    print(f"Added lag/rolling features (windows: {LAG_WINDOWS_HOURS}h, plus 24h/48h lags)")
    print(f"Training separate models for horizons: {FORECAST_HORIZONS_HOURS} hours "
          f"({[h // 24 for h in FORECAST_HORIZONS_HOURS]} day(s) respectively)")

    all_results = []
    for horizon_hours in FORECAST_HORIZONS_HOURS:
        # Pass a fresh copy per horizon: add_future_weather_features and
        # build_target both add horizon-specific columns, and we don't want
        # horizon N's columns leaking into horizon N+1's dataframe.
        horizon_results = run_for_horizon(df.copy(), horizon_hours)
        all_results.extend(horizon_results)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("model_results.csv", index=False)

    print(f"\n{'=' * 60}")
    print("SUMMARY (all horizons)")
    print(f"{'=' * 60}")
    print(results_df.pivot(index="model", columns="horizon_hours", values="r2")
          .round(4)
          .to_string())
    print("\nSaved full results to model_results.csv")

    return results_df


if __name__ == "__main__":
    main()