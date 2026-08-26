"""
hourly_pipeline.py
The script GitHub Actions runs every hour: fetch the latest reading,
turn it into a feature row (using the last stored row for aqi_change_rate),
and upsert it into the feature store. This is the live-data counterpart
to backfill_data.py + prep.py, which only ran once for historical data.
"""

import pandas as pd

import feature_store as fs
from feature_engineering import build_features
from fetch_data import fetch_current_reading


def get_previous_row() -> dict | None:
    """Pull the most recent row already in the store, as a dict, so
    build_features() can compute aqi_change_rate against it. Returns
    None if the store is empty (shouldn't happen after Day 2's backfill,
    but handled so a fresh/empty store doesn't crash the pipeline)."""
    try:
        latest_df = fs.get_latest(1)
    except FileNotFoundError:
        return None

    if latest_df.empty:
        return None

    row = latest_df.iloc[0].to_dict()
    # timestamp_utc needs to be an ISO string for build_features()'s
    # datetime.fromisoformat() call, not a pandas Timestamp object.
    row["timestamp_utc"] = pd.Timestamp(row["timestamp_utc"]).isoformat()
    return row


def run():
    print("Fetching latest reading...")
    current = fetch_current_reading()
    print(f"  station={current.get('station_id')}  aqi={current.get('aqi')}  "
          f"time={current.get('timestamp_utc')}")

    previous = get_previous_row()
    if previous is None:
        print("No previous row found in store (empty store) - "
              "aqi_change_rate will be null for this row.")

    features = build_features(current, previous_row=previous)

    row_df = pd.DataFrame([features])
    total_rows = fs.insert(row_df)

    print(f"Inserted 1 row. Feature store now has {total_rows} total rows.")
    return features


if __name__ == "__main__":
    run()