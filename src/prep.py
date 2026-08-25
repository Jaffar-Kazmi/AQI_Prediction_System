"""
prep.py
End-to-end prep of the Open-Meteo backfill data:
1. Load the raw backfill CSV
2. Trim leading NaN rows (model spin-up lag)
3. Fill any remaining scattered us_aqi gaps via the EPA PM2.5 formula
4. Standardize columns to the canonical schema (shared with fetch_data.py)
5. Apply time features + AQI change rate
6. Save the final training-ready CSV
"""

import pandas as pd

from backfill_data import fill_missing_aqi
from feature_engineering import build_features_df, standardize_backfill_columns

RAW_BACKFILL_PATH = "islamabad_aqi_backfill.csv"
FINAL_OUTPUT_PATH = "islamabad_aqi_features.csv"


def load_and_trim(path: str) -> pd.DataFrame:
    """Load the raw backfill CSV and trim any leading rows before the
    data source's coverage actually begins (model spin-up lag)."""
    df = pd.read_csv(path, parse_dates=["time"])

    print("NaN counts (raw):")
    print(df.isna().sum())

    first_valid = df[df["pm10"].notna()]["time"].min()
    print(f"\nFirst valid pm10 reading: {first_valid}")

    df_trimmed = df[df["time"] >= first_valid].reset_index(drop=True)

    print(f"\nNaN counts after trimming to {first_valid}:")
    print(df_trimmed.isna().sum())

    return df_trimmed


def main():
    df = load_and_trim(RAW_BACKFILL_PATH)

    df = fill_missing_aqi(df)
    remaining_aqi_nans = df["us_aqi"].isna().sum()
    print(f"\nRemaining us_aqi NaNs after EPA fallback: {remaining_aqi_nans}")

    if remaining_aqi_nans > 0:
        # Can't compute AQI without pm2_5 either - drop these rare rows
        # rather than leave a gap that would corrupt aqi_change_rate.
        before = len(df)
        df = df.dropna(subset=["us_aqi"]).reset_index(drop=True)
        print(f"Dropped {before - len(df)} rows with no recoverable us_aqi")

    df_std = standardize_backfill_columns(df)
    df_features = build_features_df(df_std)

    print(f"\nFinal columns: {list(df_features.columns)}")
    print(f"Final row count: {len(df_features)}")
    print(f"Date range: {df_features['timestamp_utc'].min()} to {df_features['timestamp_utc'].max()}")
    print(f"aqi_change_rate NaNs: {df_features['aqi_change_rate'].isna().sum()} (should be exactly 1 - the first row)")

    df_features.to_csv(FINAL_OUTPUT_PATH, index=False)
    print(f"\nSaved: {FINAL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()