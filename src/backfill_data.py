"""
backfill_data.py
Pulls historical air quality + weather for Islamabad from Open-Meteo
(no API key required) and produces a feature-engineered training dataset
covering everything back to when their global CAMS coverage begins.
"""

from datetime import date

import pandas as pd
import requests

ISLAMABAD_LAT = 33.7294
ISLAMABAD_LON = 73.0931
TIMEZONE = "Asia/Karachi"

# Global CAMS coverage (non-Europe) starts August 2022. Requesting further
# back than a source's actual coverage just returns empty/null rows, so
# this is a genuine floor, not an arbitrary choice.
EARLIEST_AVAILABLE_DATE = "2022-08-01"

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

AIR_QUALITY_VARS = [
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone", "us_aqi",
]
WEATHER_VARS = [
    "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "wind_speed_10m", "precipitation",
]


def fetch_historical_air_quality(start_date: str, end_date: str) -> pd.DataFrame:
    """Pull hourly pollutant + US AQI history for the given date range."""
    params = {
        "latitude": ISLAMABAD_LAT,
        "longitude": ISLAMABAD_LON,
        "hourly": ",".join(AIR_QUALITY_VARS),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": TIMEZONE,
    }
    resp = requests.get(AIR_QUALITY_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    hourly = payload.get("hourly", {})
    df = pd.DataFrame(hourly)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"])
    return df


def fetch_historical_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """Pull hourly weather history for the same date range."""
    params = {
        "latitude": ISLAMABAD_LAT,
        "longitude": ISLAMABAD_LON,
        "hourly": ",".join(WEATHER_VARS),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": TIMEZONE,
    }
    resp = requests.get(WEATHER_ARCHIVE_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    hourly = payload.get("hourly", {})
    df = pd.DataFrame(hourly)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"])
    return df


def build_backfill_dataset(
    start_date: str = EARLIEST_AVAILABLE_DATE,
    end_date: str | None = None,
    chunk_days: int = 90,
) -> pd.DataFrame:
    """Fetch + merge air quality and weather in chunks (avoids single
    oversized requests) and return one combined, time-sorted dataframe.
    Does NOT apply feature engineering - that's a separate step so this
    raw merged dataset can also be inspected/cleaned on its own."""
    end_date = end_date or date.today().isoformat()

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    # Build chunk start boundaries, then add one final boundary just past
    # the end date so every real chunk (including the last, possibly
    # partial, one) has both a start and an end to pair against.
    boundaries = list(pd.date_range(start=start_ts, end=end_ts, freq=f"{chunk_days}D"))
    if not boundaries or boundaries[0] != start_ts:
        boundaries.insert(0, start_ts)
    boundaries.append(end_ts + pd.Timedelta(days=1))

    aq_chunks, weather_chunks = [], []

    for i in range(len(boundaries) - 1):
        chunk_start = boundaries[i].date().isoformat()
        chunk_end = (boundaries[i + 1] - pd.Timedelta(days=1)).date().isoformat()
        if chunk_end < chunk_start or chunk_start > end_date:
            continue

        print(f"Fetching {chunk_start} to {chunk_end}...")
        aq_chunks.append(fetch_historical_air_quality(chunk_start, chunk_end))
        weather_chunks.append(fetch_historical_weather(chunk_start, chunk_end))

    aq_df = pd.concat(aq_chunks, ignore_index=True).drop_duplicates(subset="time")
    weather_df = pd.concat(weather_chunks, ignore_index=True).drop_duplicates(subset="time")

    merged = pd.merge(aq_df, weather_df, on="time", how="inner")
    merged = merged.sort_values("time").reset_index(drop=True)
    return merged

def compute_us_aqi_from_pm25(pm25):
    """EPA breakpoint formula for PM2.5 -> US AQI."""
    if pd.isna(pm25):
        return None
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for conc_low, conc_high, aqi_low, aqi_high in breakpoints:
        if conc_low <= pm25 <= conc_high:
            return round(
                (aqi_high - aqi_low) / (conc_high - conc_low) * (pm25 - conc_low) + aqi_low
            )
    return 500


def fill_missing_aqi(df):
    """Fill any missing us_aqi values using pm2_5 via the EPA formula."""
    df = df.copy()
    missing_mask = df["us_aqi"].isna() & df["pm2_5"].notna()
    df.loc[missing_mask, "us_aqi"] = df.loc[missing_mask, "pm2_5"].apply(compute_us_aqi_from_pm25)
    return df

if __name__ == "__main__":
    # Full backfill: everything available (Aug 2022 -> today).
    # Switch back to a short date range above if you need to debug something.
    df = build_backfill_dataset()
    df.to_csv("islamabad_aqi_backfill.csv", index=False)

    print(df.head(10))
    print(f"\nTotal rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nDate range: {df['time'].min()} to {df['time'].max()}")
    print("Saved to islamabad_aqi_backfill.csv")