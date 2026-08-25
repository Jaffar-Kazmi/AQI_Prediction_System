"""
feature_engineering.py
Turns raw fetch_data.py rows into model-ready features:
- time-based features (hour, day, month, day-of-week)
- AQI change rate (requires at least one prior reading)
- basic cleanup (rain None -> 0, drop unreliable dew_point)
"""

from datetime import datetime

import pandas as pd

# Canonical column names used everywhere downstream (feature store, training,
# dashboard) - both the live fetch_data.py rows and the Open-Meteo backfill
# dataframe get renamed into this schema so nothing downstream needs to know
# which source a row originally came from.
BACKFILL_COLUMN_MAP = {
    "time": "timestamp_utc",
    "us_aqi": "aqi",
    "pm2_5": "pm25",
    "pm10": "pm10",
    "carbon_monoxide": "co",
    "nitrogen_dioxide": "no2",
    "sulphur_dioxide": "so2",
    "ozone": "o3",
    "temperature_2m": "temperature_c",
    "relative_humidity_2m": "humidity_pct",
    "surface_pressure": "pressure_hpa",
    "wind_speed_10m": "wind_speed",
    "precipitation": "rain",
}


def standardize_backfill_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename Open-Meteo's backfill columns to match the live pipeline's
    schema (fetch_data.py / feature_engineering.py row format)."""
    df = df.rename(columns=BACKFILL_COLUMN_MAP)
    df["station_id"] = "open-meteo-backfill"
    return df


def add_time_features(row: dict) -> dict:
    """Derive hour/day/month/day-of-week from the fetch timestamp.
    Uses timestamp_utc since it's always populated (unlike station_time_local,
    which depends on which station served the reading)."""
    ts = datetime.fromisoformat(row["timestamp_utc"])

    row["hour"] = ts.hour
    row["day"] = ts.day
    row["month"] = ts.month
    row["day_of_week"] = ts.weekday()  # 0 = Monday
    row["is_weekend"] = int(ts.weekday() >= 5)

    return row


def clean_fields(row: dict) -> dict:
    """rain=None means 'no rain reported', not missing data -> treat as 0.
    dew_point is dropped: neither AQICN nor OpenWeather's free tier
    provides it reliably, so it'd be mostly-null noise as a feature."""
    row["rain"] = row.get("rain") or 0.0
    row.pop("dew_point", None)
    return row


def add_change_rate(current_row: dict, previous_row: dict | None) -> dict:
    """AQI change rate = (current - previous) / hours_elapsed.
    Returns None for both fields when there's no prior reading yet
    (e.g. the very first row of the pipeline) - this is expected and
    should be handled downstream (e.g. dropped from training rows,
    kept as a null-able feature for inference).
    """
    if previous_row is None or previous_row.get("aqi") is None or current_row.get("aqi") is None:
        current_row["aqi_change_rate"] = None
        current_row["hours_since_previous"] = None
        return current_row

    try:
        current_ts = datetime.fromisoformat(current_row["timestamp_utc"])
        previous_ts = datetime.fromisoformat(previous_row["timestamp_utc"])
        hours_elapsed = (current_ts - previous_ts).total_seconds() / 3600

        if hours_elapsed <= 0:
            current_row["aqi_change_rate"] = None
            current_row["hours_since_previous"] = None
            return current_row

        aqi_delta = current_row["aqi"] - previous_row["aqi"]
        current_row["aqi_change_rate"] = round(aqi_delta / hours_elapsed, 4)
        current_row["hours_since_previous"] = round(hours_elapsed, 4)

    except (ValueError, KeyError, TypeError):
        current_row["aqi_change_rate"] = None
        current_row["hours_since_previous"] = None

    return current_row


def build_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized version of build_features() for an entire backfill
    dataframe, rather than looping build_features() 35,000 times.
    Assumes df is already in canonical schema (run standardize_backfill_columns
    first if it came from Open-Meteo) and sorted by timestamp_utc."""
    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    # rain: None/NaN -> 0, same rule as the single-row version
    if "rain" in df.columns:
        df["rain"] = df["rain"].fillna(0.0)
    df = df.drop(columns=["dew_point"], errors="ignore")

    # time features
    df["hour"] = df["timestamp_utc"].dt.hour
    df["day"] = df["timestamp_utc"].dt.day
    df["month"] = df["timestamp_utc"].dt.month
    df["day_of_week"] = df["timestamp_utc"].dt.weekday
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # aqi change rate: same formula as add_change_rate, vectorized via diff().
    # First row has no prior reading -> NaN, matching the single-row version's
    # None for the first-ever reading.
    hours_elapsed = df["timestamp_utc"].diff().dt.total_seconds() / 3600
    aqi_delta = df["aqi"].diff()
    df["hours_since_previous"] = hours_elapsed.round(4)
    df["aqi_change_rate"] = (aqi_delta / hours_elapsed).round(4)
    # Any non-positive gap (shouldn't happen in sorted backfill data, but
    # matches the single-row guard) is treated as missing rather than
    # producing a divide-by-zero or negative-time artifact.
    invalid_gap = hours_elapsed <= 0
    df.loc[invalid_gap, ["aqi_change_rate", "hours_since_previous"]] = None

    return df


def build_features(current_row: dict, previous_row: dict | None = None) -> dict:
    """Main entry point: apply all feature engineering steps to a single row.

    current_row: the latest row from fetch_data.fetch_current_reading()
    previous_row: the most recent PRIOR row from the feature store, if any.
                   Pass None for the very first row ever collected.
    """
    row = dict(current_row)  # don't mutate the caller's dict
    row = clean_fields(row)
    row = add_time_features(row)
    row = add_change_rate(row, previous_row)
    return row


if __name__ == "__main__":
    from fetch_data import fetch_current_reading

    # Demo: fetch two readings ~seconds apart just to prove the change-rate
    # logic runs end-to-end. In production, previous_row will come from the
    # feature store (the last row written), not a second live fetch.
    first = fetch_current_reading()
    first_features = build_features(first, previous_row=None)
    print("First reading (no prior row yet):")
    for k, v in first_features.items():
        print(f"  {k}: {v}")

    second = fetch_current_reading()
    second_features = build_features(second, previous_row=first)
    print("\nSecond reading (change rate computed):")
    for k, v in second_features.items():
        print(f"  {k}: {v}")