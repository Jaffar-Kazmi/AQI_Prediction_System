"""
feature_engineering.py
Turns raw fetch_data.py rows into model-ready features:
- time-based features (hour, day, month, day-of-week)
- AQI change rate (requires at least one prior reading)
- basic cleanup (rain None -> 0, drop unreliable dew_point)
"""

from datetime import datetime, timezone


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