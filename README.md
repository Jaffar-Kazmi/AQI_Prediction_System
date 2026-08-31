# Islamabad AQI Forecast

An end-to-end air quality forecasting system for Islamabad: hourly data ingestion, a daily-retrained machine learning pipeline, and an interactive dashboard predicting AQI 24, 48, and 72 hours ahead.

**Live dashboard:** [Islamabad AQI Prediction System](https://islamabad-aqi-prediction.streamlit.app/)

> This is an independent project, not an official air-quality authority. For health guidance during severe smog, refer to Pakistan's Ministry of Climate Change or your local health department.

---

## What it does

- Pulls live pollutant readings (AQICN) and weather data (OpenWeather) every hour, with automatic fallback across four Islamabad sensor stations.
- Maintains a growing feature store of historical + live readings, backfilled with four years of history from Open-Meteo's CAMS reanalysis archive.
- Trains three separate XGBoost models daily — one each for 24h, 48h, and 72h ahead — since forecasting further out is a genuinely harder, differently-shaped problem than forecasting tomorrow.
- Serves live predictions through a dashboard showing the current reading, a 3-day forecast, a SHAP-based explanation of *why* the model predicted what it did, and hazard alerts when AQI crosses an unhealthy threshold.

---

## Architecture

```
                    ┌──────────────────────┐
                    │ AQICN / OpenWeather  │
                    └──────────┬───────────┘
                               │ hourly (GitHub Actions)
                               ▼
                    ┌──────────────────────┐
   Open-Meteo   ──▶ │   feature_store/     │ ◀── one-time backfill
   (backfill)       │ aqi_features.parquet │
                    └──────────┬───────────┘
                               │ daily (GitHub Actions)
                               ▼
                    ┌──────────────────────┐
                    │   model_registry/    │
                    │xgboost_*_{h}h.joblib │
                    └──────────┬───────────┘
                               │ read directly, in-process
                               ▼
                    ┌──────────────────────┐
                    │    dashboard.py      │  ← Streamlit Community Cloud
                    │ (predict.py + SHAP)  │
                    └──────────────────────┘
```

Both the feature store and the trained models are committed to this repository (not an external database) so that a single `git push` — whether from a scheduled GitHub Action or a manual commit — is enough to update both the data and the live dashboard. Streamlit Cloud auto-redeploys on every push to `main`, which is what keeps the deployed app in sync with the latest hourly ingest and daily retrain.

---

## Project structure

```
src/
  fetch_data.py          # Pulls one live AQICN + OpenWeather reading
  feature_engineering.py # Shared feature logic (used by both training and serving)
  backfill_data.py       # One-time historical pull from Open-Meteo
  prep.py                # Cleans + merges backfill data into training-ready CSV
  schema.py              # Canonical column whitelist - the single source of truth
                          # for what counts as a valid feature, so a stray API
                          # field can never silently drift into the model
  feature_store.py       # Local versioned Parquet store (get/insert/read)
  hourly_pipeline.py     # The script GitHub Actions runs every hour
  train.py               # Trains + evaluates 24h/48h/72h models (Ridge, RF, XGBoost, MLP)
  predict.py             # Loads saved models, serves live predictions + SHAP
  api.py                 # Optional standalone FastAPI wrapper around predict.py
  dashboard.py           # The Streamlit app - imports predict.py directly

feature_store/           # Generated: the live feature data (tracked in git)
model_registry/          # Generated: trained models per horizon (tracked in git)
.github/workflows/
  hourly_ingest.yml      # Runs hourly_pipeline.py every hour, commits the result
  daily_retrain.yml      # Runs train.py daily, commits updated models
.streamlit/
  config.toml            # Forces light theme regardless of viewer's OS setting
requirements.txt
```

---

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate        # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Create a `.env` file in the project root with your API keys:
```
AQICN_TOKEN=your_token_here
OPENWEATHER_API_KEY=your_key_here
```

**One-time setup** (historical data + first trained models):
```bash
cd src
python backfill_data.py      # pulls ~4 years of history from Open-Meteo
python prep.py                # cleans and merges it
python feature_store.py       # loads it into the feature store
python train.py               # trains the 24h/48h/72h models
```

**Run the dashboard:**
```bash
streamlit run src/dashboard.py
```

**Keep the feature store current** (normally handled by GitHub Actions, but can be run manually):
```bash
python src/hourly_pipeline.py
```

**Optional standalone API** (not required for the dashboard, which calls `predict.py` directly — useful only if you want a separate service other tools can call):
```bash
uvicorn api:app --reload --port 8000 --app-dir src
```

---

## Automation (GitHub Actions)

Two scheduled workflows keep the deployed dashboard current without manual intervention:

| Workflow | Schedule | What it does |
|---|---|---|
| `hourly_ingest.yml` | Every hour | Fetches one live reading, appends it to the feature store, commits + pushes |
| `daily_retrain.yml` | Once daily | Retrains all three horizon models on the latest data, commits + pushes |

Both require these repository secrets to be set (Settings → Secrets and variables → Actions):
- `AQICN_TOKEN`
- `OPENWEATHER_API_KEY`

---

## Model performance

Each horizon is evaluated against a naive "assume nothing changes" baseline. R² is the share of actual variation the model explains — higher is better, 0 means "no better than guessing the current value holds."

| Model | 24h | 48h | 72h |
|---|---|---|---|
| Persistence (naive) | 0.55 | 0.26 | 0.01 |
| Ridge | 0.64 | 0.39 | 0.24 |
| Random Forest | 0.61 | 0.43 | 0.36 |
| **XGBoost (deployed)** | **0.64** | **0.45** | **0.38** |
| PyTorch MLP | 0.63 | 0.45 | 0.36 |

Accuracy declines with horizon, as expected — three days out is a meaningfully harder prediction problem than tomorrow, since more time exists for weather patterns to shift in ways the model can't yet see. XGBoost is the model saved to `model_registry/` and used in production; the others are trained and evaluated for comparison in `train.py`.

---

## Known limitations

- **Weather-forecast gap at inference time.** Training uses the *true* future weather as a documented stand-in for a real forecast (available in backfill data via hindsight). At prediction time there's no hindsight, so `predict.py` falls back to the most recently observed weather as a naive persistence forecast. This is a second, cruder approximation on top of the first — live prediction accuracy, especially at 72h, is likely somewhat below the R² figures above. The dashboard surfaces this caveat directly rather than hiding it.
- **Trend chart is currently limited to 4 points** (current + 3 forecasts). A full historical trend line would need a dedicated endpoint reading multiple days from the feature store, not yet built.
- **Alert threshold is a single constant** (AQI ≥ 150, "Unhealthy for Sensitive Groups"), configurable in `predict.py`.
- **Free-tier hosting sleeps when idle.** The first visitor after a period of inactivity may see a brief "waking up" delay.

---

## Data sources

- **AQICN** — live pollutant readings (PM2.5, PM10, NO₂, SO₂, CO, O₃) from Islamabad sensor stations, with automatic fallback across four stations if the primary is offline.
- **OpenWeather** — live temperature, humidity, pressure, and wind data.
- **Open-Meteo (CAMS reanalysis)** — four years of historical pollutant + weather data used for model training.

---
