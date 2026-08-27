"""
api.py
FastAPI backend for the AQI dashboard. Wraps predict.py's functions as
HTTP endpoints so the Streamlit dashboard (or anything else - a mobile
app, a cron job that sends alert emails) can consume predictions without
importing the ML code directly.

Run with:
    uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import predict

app = FastAPI(
    title="Islamabad AQI Forecast API",
    description="Live AQI predictions for 24h/48h/72h horizons, with SHAP explanations and hazard alerts.",
    version="1.0.0",
)

# Streamlit runs on a different port locally - CORS needs to allow it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict")
def get_predictions():
    """Current reading plus 24h/48h/72h forecasts."""
    try:
        return predict.predict_all()
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/predict/{horizon_hours}")
def get_single_prediction(horizon_hours: int):
    """A single horizon's forecast. horizon_hours must be 24, 48, or 72."""
    if horizon_hours not in predict.HORIZONS:
        raise HTTPException(
            status_code=400,
            detail=f"horizon_hours must be one of {predict.HORIZONS}, got {horizon_hours}",
        )
    try:
        return predict.predict_horizon(horizon_hours)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/explain/{horizon_hours}")
def get_explanation(horizon_hours: int, top_n: int = 8):
    """SHAP feature contributions behind one horizon's prediction."""
    if horizon_hours not in predict.HORIZONS:
        raise HTTPException(
            status_code=400,
            detail=f"horizon_hours must be one of {predict.HORIZONS}, got {horizon_hours}",
        )
    try:
        return predict.explain_horizon(horizon_hours, top_n=top_n)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/alerts")
def get_alerts():
    """Any current or forecast reading at/above the hazardous threshold.
    Returns an empty list when nothing is currently alert-worthy - the
    dashboard treats an empty list as 'all clear', not as an error."""
    try:
        data = predict.predict_all()
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=503, detail=str(e))

    alerts = []
    if data["current"]["alert"]:
        alerts.append({
            "when": "now",
            "aqi": data["current"]["aqi"],
            "category": data["current"]["category"],
            "advice": data["current"]["advice"],
        })
    for horizon_key, forecast in data["forecast"].items():
        if forecast["alert"]:
            alerts.append({
                "when": horizon_key,
                "aqi": forecast["predicted_aqi"],
                "category": forecast["category"],
                "advice": forecast["advice"],
            })

    return {"threshold_aqi": predict.ALERT_THRESHOLD_AQI, "active_alerts": alerts}