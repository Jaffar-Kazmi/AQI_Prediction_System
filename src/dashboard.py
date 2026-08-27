"""
dashboard.py
Streamlit front end for the AQI forecast API. Run the API first, then this:

    uvicorn api:app --reload --port 8000
    streamlit run dashboard.py
"""

import pandas as pd
import requests
import streamlit as st

API_BASE = "http://localhost:8000"

CATEGORY_COLORS = {
    "Good": "#2F6F4E",
    "Moderate": "#C98A2E",
    "Unhealthy for Sensitive Groups": "#B24C3A",
    "Unhealthy": "#8C2F3E",
    "Very Unhealthy": "#6B3F63",
    "Hazardous": "#472A42",
}

st.set_page_config(page_title="Islamabad AQI Forecast", page_icon="🌫️", layout="wide")


@st.cache_data(ttl=60)
def fetch(endpoint: str):
    resp = requests.get(f"{API_BASE}{endpoint}", timeout=10)
    resp.raise_for_status()
    return resp.json()


def color_for(category: str) -> str:
    return CATEGORY_COLORS.get(category, "#5C635C")


# ---------- load data ----------
try:
    predictions = fetch("/predict")
    alerts = fetch("/alerts")
except requests.exceptions.RequestException as e:
    st.error(
        "Can't reach the prediction API. Make sure it's running: "
        "`uvicorn api:app --reload --port 8000`\n\n"
        f"Details: {e}"
    )
    st.stop()

current = predictions["current"]
forecast = predictions["forecast"]

# ---------- alert banner ----------
if alerts["active_alerts"]:
    for a in alerts["active_alerts"]:
        st.error(
            f"**Hazard alert — {a['when']}**: AQI {a['aqi']} ({a['category']}). {a['advice']}"
        )
else:
    st.success("No hazardous AQI levels currently forecast in the next 3 days.")

# ---------- current reading ----------
st.title("🌫️ Islamabad AQI Forecast")
st.caption(f"Station: {current.get('station_id', 'unknown')} · as of {current['timestamp']}")

col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Current AQI", current["aqi"])
    st.markdown(
        f"<span style='color:{color_for(current['category'])}; font-weight:600'>"
        f"{current['category']}</span>",
        unsafe_allow_html=True,
    )
with col2:
    st.write(current["advice"])

st.divider()

# ---------- forecast cards ----------
st.subheader("Next 3 days")
st.caption(predictions["caveat"])

cols = st.columns(3)
for col, (horizon_key, f) in zip(cols, forecast.items()):
    with col:
        st.markdown(f"**{horizon_key} ahead**")
        st.markdown(
            f"<div style='font-size:2.2rem; font-weight:700; color:{color_for(f['category'])}'>"
            f"{f['predicted_aqi']}</div>",
            unsafe_allow_html=True,
        )
        st.caption(f["category"])
        if f["alert"]:
            st.warning("Hazardous")

st.divider()

# ---------- SHAP explanations ----------
st.subheader("Why the model predicted this")
explain_horizon = st.selectbox(
    "Explain which forecast?", options=["24h", "48h", "72h"], index=0
)
horizon_hours = int(explain_horizon.replace("h", ""))

try:
    explanation = fetch(f"/explain/{horizon_hours}")
    contrib_df = pd.DataFrame(explanation["top_contributions"])
    contrib_df = contrib_df.sort_values("shap_contribution")

    st.caption(
        f"Base rate: {explanation['base_value']} AQI. Each bar shows how much that "
        f"feature's current value pushed the {explain_horizon} prediction up or down "
        f"from the base rate, to arrive at {explanation['predicted_aqi']}."
    )
    st.bar_chart(contrib_df.set_index("feature")["shap_contribution"])

    with st.expander("Raw feature values behind this explanation"):
        st.dataframe(contrib_df[["feature", "value", "shap_contribution"]], hide_index=True)

except requests.exceptions.RequestException as e:
    st.warning(f"Couldn't load explanation: {e}")

st.divider()
st.caption(
    "Independent forecasting project, not an official air-quality authority. "
    "Data: AQICN, OpenWeather, Open-Meteo."
)