"""
schema.py
Single source of truth for the canonical row schema shared by every stage
of the pipeline (fetch_data.py -> feature_engineering.py -> feature_store.py
-> train.py). Import CANONICAL_COLUMNS wherever a row gets cleaned, stored,
or turned into model features.

WHY THIS EXISTS:
Two live-only fields (dominant_pollutant, station_time_local, and later the
ow_* fields) silently leaked into the feature store because clean_fields()
in feature_engineering.py used a blacklist ("drop these specific known-bad
fields"). Every time fetch_data.py added a new field, the blacklist needed
a matching update, and nothing enforced that - the bug was only caught by
manually inspecting df.columns after the fact.

A whitelist inverts this: only fields explicitly listed here are allowed
into the feature store / model. Any new field fetch_data.py adds in the
future is dropped by default and (in strict mode) raises loudly instead of
silently drifting through - so the failure mode changes from "quietly wrong
model six weeks from now" to "immediate error today".
"""

# Metadata columns: kept in the feature store for bookkeeping/joins, but
# explicitly excluded from the model's feature matrix (see NON_FEATURE_COLUMNS
# in train.py). Not pollutant/weather signal.
METADATA_COLUMNS = ["timestamp_utc", "station_id"]

# Actual model-input features. Every one of these must be numeric.
FEATURE_COLUMNS = [
    "pm10", "pm25", "co", "no2", "so2", "o3", "aqi",
    "temperature_c", "humidity_pct", "pressure_hpa", "wind_speed", "rain",
    "hour", "day", "month", "day_of_week", "is_weekend",
    "hours_since_previous", "aqi_change_rate",
]

CANONICAL_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS


def enforce_schema(row: dict, strict: bool = False) -> dict:
    """Keep only CANONICAL_COLUMNS from a row dict, in canonical order.

    strict=False (default): silently drop any extra fields not in the
    canonical schema. Safe default for production - a stray field from an
    API response shouldn't crash the hourly pipeline.

    strict=True: raise if the row contains fields outside the canonical
    schema. Use this in tests / CI, or when you deliberately want to be
    alerted the moment fetch_data.py's response shape changes, rather than
    silently dropping the new field and finding out weeks later.

    Also raises (regardless of strict) if a REQUIRED canonical column is
    missing - that's a different, more serious failure than an extra field.
    """
    missing = [c for c in CANONICAL_COLUMNS if c not in row]
    if missing:
        raise ValueError(f"Row is missing required canonical columns: {missing}")

    extra = [k for k in row if k not in CANONICAL_COLUMNS]
    if extra and strict:
        raise ValueError(
            f"Row contains fields outside the canonical schema: {extra}. "
            f"If these are intentional new features, add them to "
            f"FEATURE_COLUMNS in schema.py first."
        )

    return {col: row[col] for col in CANONICAL_COLUMNS}


def validate_dataframe_schema(df, strict: bool = False):
    """Same idea as enforce_schema(), but for a whole dataframe (e.g. right
    after standardize_backfill_columns(), or before writing to the feature
    store). Returns the dataframe restricted to canonical columns, in
    canonical order. Raises the same way enforce_schema() does."""
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required canonical columns: {missing}")

    extra = [c for c in df.columns if c not in CANONICAL_COLUMNS]
    if extra and strict:
        raise ValueError(
            f"DataFrame contains columns outside the canonical schema: {extra}. "
            f"If these are intentional new features, add them to "
            f"FEATURE_COLUMNS in schema.py first."
        )

    return df[CANONICAL_COLUMNS]