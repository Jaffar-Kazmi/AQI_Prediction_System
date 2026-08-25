"""
fetch_data.py
Pulls current AQI + weather readings for the Islamabad US Embassy
AQICN station and returns a clean, flat dict ready for feature engineering.
"""

import os
import time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

load_dotenv()  # reads .env in the current working directory into os.environ

AQICN_TOKEN = os.environ.get("AQICN_TOKEN")  
OPENWEATHER_API_KEY = os.environ.get(
    "OPENWEATHER_API_KEY"
) 

# Islamabad coordinates, used for the OpenWeather call
ISLAMABAD_LAT = 33.7294
ISLAMABAD_LON = 73.0931

# Islamabad stations, in priority order. The US Embassy station (H11739) is
# the most authoritative source when it's live, but it went offline in
# Feb 2026, so we fall through to the citizen-network (GAIA) stations below.
STATION_IDS = [
    "H11739",  # Islamabad US Embassy (most authoritative, but currently offline)
    "A511660",  # Street 20, F-11/2
    "A345148",  # Street 3, F-8/3
    "A521242",  # Faisal Avenue
]

MAX_DATA_AGE_HOURS = 6  # readings older than this are treated as stale/offline


def _is_fresh(payload):
    """Check the station's own reported time, not just HTTP success -
    a station can respond 'ok' while still serving a months-old cached reading."""
    iso_time = payload.get("data", {}).get("time", {}).get("iso")
    if not iso_time:
        return False
    try:
        reading_time = datetime.fromisoformat(iso_time)
        age = datetime.now(reading_time.tzinfo) - reading_time
        return age.total_seconds() <= MAX_DATA_AGE_HOURS * 3600
    except ValueError:
        return False


def _fetch_station(station_id, token, retries = 2, backoff= 2.0):
    """Try one station. Returns the payload if it's reachable AND fresh, else None."""
    url = f"https://api.waqi.info/feed/@{station_id}/"
    params = {"token": token}

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            payload = resp.json()

            if payload.get("status") == "ok" and _is_fresh(payload):
                return payload

        except (requests.RequestException, ValueError):
            pass

        time.sleep(backoff * attempt)

    return None


def fetch_weather(
    api_key: str = OPENWEATHER_API_KEY,
    lat: float = ISLAMABAD_LAT,
    lon: float = ISLAMABAD_LON,
    retries: int = 2,
    backoff: float = 2.0,
) -> dict | None:
    """Fetch current weather from OpenWeather. Returns None on failure rather
    than raising - weather is a supplement, a missed call here shouldn't
    take down the whole pollutant reading."""
    if not api_key:
        return None

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError):
            time.sleep(backoff * attempt)

    return None


def parse_weather(payload: dict | None) -> dict:
    """Flatten the OpenWeather response into the same field names used
    elsewhere in the row, so downstream code doesn't care which source
    a given weather value came from."""
    if payload is None:
        return {
            "ow_temperature_c": None,
            "ow_humidity_pct": None,
            "ow_pressure_hpa": None,
            "ow_wind_speed": None,
            "ow_weather_condition": None,
        }

    main = payload.get("main", {})
    wind = payload.get("wind", {})
    weather_list = payload.get("weather", [])

    return {
        "ow_temperature_c": main.get("temp"),
        "ow_humidity_pct": main.get("humidity"),
        "ow_pressure_hpa": main.get("pressure"),
        "ow_wind_speed": wind.get("speed"),
        "ow_weather_condition": weather_list[0].get("main") if weather_list else None,
    }


def fetch_raw(token = AQICN_TOKEN):
    """Try each station in STATION_IDS in order, return the first fresh reading.
    Raises only if every station in the list is unreachable or stale."""
    if not token:
        raise ValueError("AQICN_TOKEN is not set. Add it to your .env or environment.")

    for station_id in STATION_IDS:
        payload = _fetch_station(station_id, token)
        if payload is not None:
            payload["_station_id"] = (
                station_id  # tag which station actually served this
            )
            return payload

    raise RuntimeError(
        f"All stations unreachable or stale (>{MAX_DATA_AGE_HOURS}h old): {STATION_IDS}"
    )


def parse_reading(payload: dict) -> dict:
    """Flatten the AQICN response into a single clean row.
    Missing pollutant/weather fields are filled with None rather than
    dropped, so the row shape stays consistent across every fetch."""
    data = payload["data"]
    iaqi = data.get("iaqi", {})

    def get_val(key):
        entry = iaqi.get(key)
        return entry.get("v") if entry else None

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "station_id": payload.get("_station_id"),
        "station_time_local": data.get("time", {}).get("iso"),
        "aqi": data.get("aqi"),
        "dominant_pollutant": data.get("dominentpol"),
        "pm25": get_val("pm25"),
        "pm10": get_val("pm10"),
        "no2": get_val("no2"),
        "so2": get_val("so2"),
        "co": get_val("co"),
        "o3": get_val("o3"),
        "temperature_c": get_val("t"),
        "humidity_pct": get_val("h"),
        "pressure_hpa": get_val("p"),
        "wind_speed": get_val("w"),
        "rain": get_val("r"),
        "dew_point": get_val("dew"),
    }
    return row


def fetch_current_reading():
    """Main entry point: pulls pollutant data from AQICN and weather from
    OpenWeather, merging into one row. AQICN's bundled weather fields (when
    a station provides them) are kept as-is; OpenWeather fields are added
    alongside as ow_* columns, and used to fill any AQICN weather gaps."""
    payload = fetch_raw()
    row = parse_reading(payload)

    weather_payload = fetch_weather()
    weather_row = parse_weather(weather_payload)
    row.update(weather_row)

    # Fill AQICN weather gaps (e.g. Street 20 reports none) from OpenWeather
    fallback_map = {
        "temperature_c": "ow_temperature_c",
        "humidity_pct": "ow_humidity_pct",
        "pressure_hpa": "ow_pressure_hpa",
        "wind_speed": "ow_wind_speed",
    }
    for aqicn_field, ow_field in fallback_map.items():
        if row.get(aqicn_field) is None:
            row[aqicn_field] = row.get(ow_field)

    return row


if __name__ == "__main__":
    reading = fetch_current_reading()
    for k, v in reading.items():
        print(f"{k}: {v}")
