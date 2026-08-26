"""
feature_store.py
A minimal local feature store: same insert/read contract a managed
feature store (Hopsworks, Vertex AI) would give you, backed by a
versioned Parquet file instead of a hosted service.

This exists specifically because Hopsworks' free-tier cluster
provisioning was unreliable during development - the abstraction here
means the rest of the pipeline (training, dashboard) never talks to
storage directly, so swapping in a real managed feature store later
is a one-file change, not a rewrite.
"""

import os

import pandas as pd

STORE_PATH = "feature_store/aqi_features.parquet"
PRIMARY_KEY = "timestamp_utc"


def _normalize_timestamps(series: pd.Series) -> pd.Series:
    """Backfilled rows (from Open-Meteo) have timezone-naive timestamps.
    Live fetch rows (from fetch_data.py's datetime.now(timezone.utc))
    are timezone-aware. Mixing the two in one column makes pandas raise
    on sort/compare, so every timestamp is normalized to UTC and then
    made naive here, once, on the way into or out of the store - nothing
    else downstream needs to know this distinction ever existed."""
    ts = pd.to_datetime(series, utc=True)
    return ts.dt.tz_localize(None)


def _ensure_store_dir():
    os.makedirs(os.path.dirname(STORE_PATH), exist_ok=True)


def insert(df: pd.DataFrame, upsert: bool = True) -> int:
    """Write rows into the feature store.

    upsert=True (default): merge with existing data, replacing any rows
    that share a timestamp_utc with incoming rows (so re-running backfill
    or an hourly fetch doesn't create duplicates).

    Returns the total row count in the store after the write.
    """
    _ensure_store_dir()
    df = df.copy()
    df[PRIMARY_KEY] = _normalize_timestamps(df[PRIMARY_KEY])

    if upsert and os.path.exists(STORE_PATH):
        existing = pd.read_parquet(STORE_PATH)
        existing[PRIMARY_KEY] = _normalize_timestamps(existing[PRIMARY_KEY])
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=PRIMARY_KEY, keep="last")
    else:
        combined = df

    combined = combined.sort_values(PRIMARY_KEY).reset_index(drop=True)
    combined.to_parquet(STORE_PATH, index=False)
    return len(combined)


def read(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Read features from the store, optionally filtered to a date range."""
    if not os.path.exists(STORE_PATH):
        raise FileNotFoundError(
            f"No feature store found at {STORE_PATH} - run insert() first."
        )

    df = pd.read_parquet(STORE_PATH)
    df[PRIMARY_KEY] = _normalize_timestamps(df[PRIMARY_KEY])

    if start:
        df = df[df[PRIMARY_KEY] >= pd.Timestamp(start)]
    if end:
        df = df[df[PRIMARY_KEY] <= pd.Timestamp(end)]

    return df.sort_values(PRIMARY_KEY).reset_index(drop=True)


def get_latest(n: int = 1) -> pd.DataFrame:
    """Return the n most recent rows - useful for computing aqi_change_rate
    against the last stored reading during live hourly fetches."""
    df = read()
    return df.tail(n).reset_index(drop=True)


if __name__ == "__main__":
    # Load the Day 2 backfill output into the store
    df = pd.read_csv("islamabad_aqi_features.csv")
    total_rows = insert(df)
    print(f"Inserted backfill data. Total rows in store: {total_rows}")

    latest = get_latest(3)
    print("\nMost recent 3 rows in the store:")
    print(latest[["timestamp_utc", "aqi", "station_id"]])