"""
predict.py
Loads the three saved horizon models (24h/48h/72h) and produces live
predictions from whatever's currently in the feature store.

IMPORTANT CAVEAT - read before trusting these numbers:
train.py's future-weather features were built from the TRUE weather at
T+horizon (a documented stand-in for a real forecast, since backfill data
has hindsight). At inference time we don't have hindsight - there's no
future to look up. This module falls back to the most recently observed
weather as a naive "persistence forecast" of temperature/humidity/etc.
That's a second, cruder approximation on top of the first, so live
prediction accuracy is likely somewhat below the R2 figures measured
during training - particularly at 72h, where weather has more room to
drift from today's values. This is surfaced in the API/dashboard, not
hidden.
"""

import os

import joblib
import pandas as pd
import shap

import feature_store as fs
from train import (
    FUTURE_WEATHER_COLS,
    add_lag_features,
)

MODEL_REGISTRY_DIR = "model_registry"
HORIZONS = [24, 48, 72]

AQI_CATEGORIES = [
    (50, "Good", "Air quality is satisfactory."),
    (100, "Moderate", "Acceptable air quality; unusually sensitive people should watch for symptoms."),
    (150, "Unhealthy for Sensitive Groups", "Sensitive groups should reduce prolonged outdoor exertion."),
    (200, "Unhealthy", "Everyone may begin to experience health effects."),
    (300, "Very Unhealthy", "Health alert: everyone may experience more serious effects."),
    (float("inf"), "Hazardous", "Health emergency: the entire population is likely affected."),
]

# Horizons at/above this AQI trigger an alert.
ALERT_THRESHOLD_AQI = 150


def categorize(aqi: float) -> tuple[str, str]:
    for ceiling, name, advice in AQI_CATEGORIES:
        if aqi <= ceiling:
            return name, advice
    return AQI_CATEGORIES[-1][1], AQI_CATEGORIES[-1][2]


def _load_model(horizon_hours: int):
    model_path = os.path.join(MODEL_REGISTRY_DIR, f"xgboost_aqi_model_{horizon_hours}h.joblib")
    features_path = os.path.join(MODEL_REGISTRY_DIR, f"feature_columns_{horizon_hours}h.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No saved model for {horizon_hours}h at {model_path}. Run train.py first."
        )
    model = joblib.load(model_path)
    feature_cols = joblib.load(features_path)
    return model, feature_cols


def _build_latest_feature_row(horizon_hours: int) -> tuple[pd.Series, pd.DataFrame]:
    """Rebuild the exact feature row the model expects for a prediction made
    right now: recent history for the lag/rolling features, and the most
    recent observed weather as a stand-in for the horizon's forecast.
    Returns (row_as_series_for_model, full_history_df) - the history df is
    returned too since SHAP wants the same row shape, and callers may want
    the raw context for debugging.
    """
    # Need at least `max(LAG_WINDOWS_HOURS)` hours of history for rolling
    # windows to be meaningful (fewer rows just means an early window uses
    # min_periods=1 and is noisier, not wrong).
    history = fs.read().sort_values("timestamp_utc").tail(200).reset_index(drop=True)
    if history.empty:
        raise ValueError("Feature store is empty - nothing to predict from.")

    history = add_lag_features(history)

    latest = history.iloc[-1].copy()

    # Forecast stand-in: persistence-of-weather (see module docstring).
    for col in FUTURE_WEATHER_COLS:
        latest[f"{col}_forecast_{horizon_hours}h"] = latest[col]

    return latest, history


def predict_horizon(horizon_hours: int) -> dict:
    model, feature_cols = _load_model(horizon_hours)
    latest_row, _ = _build_latest_feature_row(horizon_hours)

    missing = [c for c in feature_cols if c not in latest_row.index]
    if missing:
        raise ValueError(
            f"Latest feature row is missing columns the {horizon_hours}h model "
            f"expects: {missing}. Has the feature store schema drifted since "
            f"training? Consider re-running train.py."
        )

    X = pd.DataFrame([latest_row[feature_cols].values], columns=feature_cols)
    pred_aqi = float(model.predict(X)[0])
    pred_aqi = max(pred_aqi, 0.0)  # AQI can't be negative

    category, advice = categorize(pred_aqi)

    return {
        "horizon_hours": horizon_hours,
        "predicted_aqi": round(pred_aqi, 1),
        "category": category,
        "advice": advice,
        "alert": pred_aqi >= ALERT_THRESHOLD_AQI,
        "as_of": pd.Timestamp.utcnow().isoformat(),
        "based_on_timestamp": str(latest_row["timestamp_utc"]),
    }


def predict_all() -> dict:
    """Predictions for every horizon, plus the current reading, in one call -
    this is what the API's /predict endpoint returns."""
    history = fs.read().sort_values("timestamp_utc")
    current_row = history.iloc[-1]
    current_aqi = float(current_row["aqi"])
    current_category, current_advice = categorize(current_aqi)

    results = {
        "current": {
            "aqi": round(current_aqi, 1),
            "category": current_category,
            "advice": current_advice,
            "alert": current_aqi >= ALERT_THRESHOLD_AQI,
            "timestamp": str(current_row["timestamp_utc"]),
            "station_id": current_row.get("station_id"),
        },
        "forecast": {f"{h}h": predict_horizon(h) for h in HORIZONS},
        "caveat": (
            "Forecasts beyond 'now' assume weather stays at its most recently "
            "observed values, since no live weather forecast is wired in yet. "
            "Treat 48h/72h predictions as directional, not precise."
        ),
    }
    return results


def explain_horizon(horizon_hours: int, top_n: int = 8) -> dict:
    """SHAP explanation for a single horizon's prediction: which features
    pushed the forecast up or down, and by how much. Uses TreeExplainer,
    which is exact (not sampled) for tree ensembles like XGBoost."""
    model, feature_cols = _load_model(horizon_hours)
    latest_row, _ = _build_latest_feature_row(horizon_hours)

    X = pd.DataFrame([latest_row[feature_cols].values], columns=feature_cols)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)[0]
    base_value = float(explainer.expected_value)

    all_contributions = [
        {"feature": col, "value": float(X[col].iloc[0]), "shap_contribution": float(sv)}
        for col, sv in zip(feature_cols, shap_values)
    ]
    # Reconstructing the prediction from base_value + every contribution
    # (not just the top_n shown) is what makes this an exact explanation
    # rather than an approximate one - SHAP values sum exactly to
    # (prediction - base_value) for tree models.
    predicted_aqi = base_value + sum(c["shap_contribution"] for c in all_contributions)

    top_contributions = sorted(
        all_contributions, key=lambda d: abs(d["shap_contribution"]), reverse=True
    )[:top_n]

    return {
        "horizon_hours": horizon_hours,
        "base_value": round(base_value, 2),
        "predicted_aqi": round(predicted_aqi, 2),
        "top_contributions": top_contributions,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(predict_all(), indent=2, default=str))
    print("\nSHAP explanation for 24h:")
    print(json.dumps(explain_horizon(24), indent=2, default=str))