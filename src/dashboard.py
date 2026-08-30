"""
dashboard.py
Streamlit front end for the AQI forecast API.

Run the API first, then this:
    uvicorn api:app --reload --port 8000 --app-dir src
    streamlit run src/dashboard.py

Design notes (why it's built this way, not just what):
- Visibility of system status: a persistent status bar shows the station,
  last reading time, and page-load time, so the user always knows how
  fresh the data is rather than wondering if the page is stale
  (Nielsen heuristic #1).
- Recognition over recall: the current AQI and its category are always
  visible near the top, not tucked behind a tab - the user shouldn't have
  to remember it while browsing forecasts or explanations.
- Consistency: one shared EPA color scale and category vocabulary is used
  everywhere (gauge, cards, charts, alerts) - a color never means
  something different in one part of the page than another.
- Color is never the only signal: every color-coded element also carries
  a text label, for colorblind users and for anyone glancing quickly.
- Error prevention & recovery: API failures show the exact fix (the
  command to run) plus a retry button, not a generic "something went wrong".
- User control: the sidebar exposes real choices (auto-refresh, manual
  refresh) instead of hardcoding behavior.
- Progressive disclosure: forecast / trend / explanation / alerts are
  separated into tabs so the page isn't a wall of everything at once.
"""

import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE = "http://localhost:8000"

CATEGORY_COLORS = {
    "Good": "#6A9C81",
    "Moderate": "#D3A65E",
    "Unhealthy for Sensitive Groups": "#C98871",
    "Unhealthy": "#BC7079",
    "Very Unhealthy": "#9B7CA3",
    "Hazardous": "#8A7288",
}
# Slightly deeper versions of the same hues, used only where a bit more
# contrast is needed for legibility (small pill labels) - never for large
# fills like the gauge bar or big numbers, which is where the original
# palette felt too intense.
CATEGORY_COLORS_DEEP = {
    "Good": "#4C7E64",
    "Moderate": "#B4863E",
    "Unhealthy for Sensitive Groups": "#A6634C",
    "Unhealthy": "#9C4F58",
    "Very Unhealthy": "#775682",
    "Hazardous": "#6B5468",
}
INK = "#2A312D"
INK_SOFT = "#6A716B"
PAPER = "#FAFAF6"
MIST = "#EFF1EA"
LINE = "#D8DACF"

st.set_page_config(
    page_title="Islamabad AQI Forecast",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- global styling ----------
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {{
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 17px;
    }}
    .block-container {{ padding-top: 1.6rem; max-width: 1200px; }}
    h1, h2, h3 {{ font-family: 'Fraunces', serif; font-weight: 600; }}
    h1 {{ font-size: 2.6rem; }}
    h2 {{ font-size: 1.9rem; }}
    h3 {{ font-size: 1.5rem; }}
    p, li, label, div {{ font-size: 1.02rem; }}

    .status-bar {{
        display:flex; justify-content:space-between; align-items:center;
        padding:10px 18px; background:{MIST}; border-radius:8px;
        font-family:'IBM Plex Mono', monospace; font-size:0.92rem;
        color:{INK_SOFT}; margin-bottom:18px; flex-wrap:wrap; gap:8px;
    }}
    .metric-card {{
        background:{PAPER}; border:1px solid {LINE}; border-radius:12px;
        padding:20px 22px; font-size:1.05rem;
    }}
    .fc-label {{
        font-family:'IBM Plex Mono', monospace; font-size:0.85rem;
        letter-spacing:0.04em; color:{INK_SOFT}; margin-bottom:10px;
    }}
    .fc-number {{
        font-family:'Fraunces', serif; font-size:2.4rem; font-weight:600;
        line-height:1.1; margin-bottom:8px;
    }}
    .fc-advice {{
        font-size:0.92rem; color:{INK_SOFT}; margin-top:10px; line-height:1.4;
    }}
    .cat-pill {{
        display:inline-block; padding:5px 14px; border-radius:999px;
        font-size:0.95rem; font-weight:600;
    }}
    .caveat-box {{
        background:{MIST}; border-left:3px solid {INK_SOFT};
        padding:12px 16px; border-radius:6px; font-size:1rem;
        color:{INK_SOFT}; margin:10px 0 20px;
    }}
    div[data-testid="stMetricValue"] {{ font-family:'Fraunces', serif; font-size:1.9rem; }}
    div[data-testid="stMetricLabel"] {{ font-size:0.95rem; }}
    [data-testid="stCaptionContainer"] {{ font-size:0.95rem; }}

    /* Tabs restyled as a pill button group instead of underlined text tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap:8px; border-bottom:none; background:{MIST};
        padding:6px; border-radius:12px; display:inline-flex;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-size:1.02rem; font-weight:500; padding:10px 20px;
        border-radius:9px; background:transparent; color:{INK_SOFT};
        border:none; transition:background .15s ease, color .15s ease;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        background:{LINE}55; color:{INK};
    }}
    .stTabs [aria-selected="true"] {{
        background:{PAPER} !important; color:{INK} !important;
        font-weight:600; border:2px solid {INK} !important;
        padding:8px 18px !important;
    }}
    .stTabs [data-baseweb="tab-highlight"] {{ display:none; }}
    .stTabs [data-baseweb="tab-border"] {{ display:none; }}

    /* The default sidebar collapse/expand arrow is small and low-contrast
       by default - this makes it larger and more visible so users notice
       the sidebar is there and interactive. */
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="collapsedControl"] button {{
        opacity:1 !important; transform:scale(1.3);
    }}

    .legend-row {{
        display:flex; align-items:center; gap:8px; font-size:0.85rem;
        margin-bottom:6px; color:{INK};
    }}
    .legend-dot {{
        width:10px; height:10px; border-radius:50%; flex-shrink:0;
    }}
</style>
""", unsafe_allow_html=True)


def category_color(category: str) -> str:
    return CATEGORY_COLORS.get(category, INK_SOFT)


def category_color_deep(category: str) -> str:
    return CATEGORY_COLORS_DEEP.get(category, INK_SOFT)


def category_pill(category: str) -> str:
    """A soft tinted background with deep-colored text reads as far gentler
    than a solid saturated fill with white text, while staying just as
    legible and just as distinguishable between categories."""
    return (f"<span class='cat-pill' style='background:{category_color(category)}22; "
            f"color:{category_color_deep(category)}'>{category}</span>")


# A small custom icon (three fading arcs = the "haze layer" motif used
# throughout the site) so the sidebar has a clear, deliberate visual
# anchor - the default Streamlit sidebar-collapse arrow is small and
# low-contrast, and users frequently miss that the sidebar is there at all.
_SIDEBAR_ICON = (
    "data:image/svg+xml,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
    "<rect width='64' height='64' rx='14' fill='%23EFF1EA'/>"
    "<path d='M14 24c6-6 14-6 20 0s14 6 20 0' stroke='%232A312D' "
    "stroke-width='4' fill='none' stroke-linecap='round'/>"
    "<path d='M14 34c6-6 14-6 20 0s14 6 20 0' stroke='%232A312D' "
    "stroke-width='4' fill='none' stroke-linecap='round' opacity='0.6'/>"
    "<path d='M14 44c6-6 14-6 20 0s14 6 20 0' stroke='%232A312D' "
    "stroke-width='4' fill='none' stroke-linecap='round' opacity='0.3'/>"
    "</svg>"
)
st.logo(_SIDEBAR_ICON, icon_image=_SIDEBAR_ICON)


@st.cache_data(ttl=55)
def fetch(endpoint: str):
    resp = requests.get(f"{API_BASE}{endpoint}", timeout=10)
    resp.raise_for_status()
    return resp.json()


def api_error_screen(err: Exception):
    st.error(
        "**Can't reach the prediction API.**\n\n"
        "This dashboard reads live predictions from a separate service that "
        "isn't running yet. Start it in another terminal, from the project root:\n\n"
        "```\nuvicorn api:app --reload --port 8000 --app-dir src\n```\n\n"
        "Then click **Retry** below."
    )
    with st.expander("Technical details"):
        st.code(str(err))
    if st.button("Retry", type="primary"):
        st.cache_data.clear()
        st.rerun()
    st.stop()


def aqi_gauge(value: float, category: str) -> go.Figure:
    """A gauge is used here (rather than just a number) because it gives an
    at-a-glance sense of WHERE on the full severity range a value sits,
    matching the mental model people already have from car dashboards /
    speedometers (recognition over recall)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"family": "Fraunces, serif", "size": 46, "color": INK}},
        gauge={
            "axis": {"range": [0, 300], "tickwidth": 1, "tickcolor": INK_SOFT},
            "bar": {"color": category_color(category), "thickness": 0.28},
            "bgcolor": PAPER,
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "#6A9C8118"},
                {"range": [50, 100], "color": "#D3A65E18"},
                {"range": [100, 150], "color": "#C9887118"},
                {"range": [150, 200], "color": "#BC707918"},
                {"range": [200, 300], "color": "#9B7CA318"},
            ],
        },
    ))
    fig.update_layout(
        height=230, margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font_color=INK,
    )
    return fig


def contribution_chart(contrib_df: pd.DataFrame, horizon_label: str) -> go.Figure:
    contrib_df = contrib_df.sort_values("shap_contribution")
    colors = [category_color_deep("Good") if v < 0 else category_color_deep("Unhealthy")
              for v in contrib_df["shap_contribution"]]
    fig = go.Figure(go.Bar(
        x=contrib_df["shap_contribution"],
        y=contrib_df["feature"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.1f}" for v in contrib_df["shap_contribution"]],
        textposition="outside",
    ))
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=30, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_family="IBM Plex Sans", font_color=INK,
        xaxis=dict(title=f"Effect on {horizon_label} prediction (AQI points)",
                    gridcolor=LINE, zeroline=True, zerolinecolor=INK_SOFT),
        yaxis=dict(title=None),
    )
    return fig


# ================= SIDEBAR (user control) =================
with st.sidebar:
    st.markdown("### Settings")
    auto_refresh = st.toggle("Auto-refresh every 60s", value=False)
    st.caption("Leave off while exploring explanations — refreshing resets your selection.")

    st.markdown("---")
    st.markdown("### AQI scale")
    st.caption("Shown here so you never have to remember what a color means.")
    legend_rows = "".join(
        f"<div class='legend-row'><span class='legend-dot' style='background:{color}'></span>{name}</div>"
        for name, color in CATEGORY_COLORS.items()
    )
    st.markdown(legend_rows, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### About this project")
    st.caption(
        "Independent forecasting pipeline for Islamabad: hourly ingestion, "
        "a daily-retrained model per horizon, and this dashboard. Not an "
        "official air-quality authority."
    )
    st.markdown("**Data sources**")
    st.caption("AQICN sensor network · OpenWeather · Open-Meteo (training history)")

    st.markdown("---")
    if st.button("Refresh now"):
        st.cache_data.clear()
        st.rerun()


# ================= LOAD DATA =================
try:
    predictions = fetch("/predict")
    alerts = fetch("/alerts")
except requests.exceptions.RequestException as e:
    api_error_screen(e)

current = predictions["current"]
forecast = predictions["forecast"]

# The active tab's border reflects the worst (most severe) of the three
# forecasted days - a single color can't represent three different days,
# so the most severe one is the most useful thing to flag at a glance,
# consistent with how the alert system already prioritizes risk.
worst_forecast = max(forecast.values(), key=lambda f: f["predicted_aqi"])
tab_border_color = category_color_deep(worst_forecast["category"])
st.markdown(
    f"<style>.stTabs [aria-selected='true'] {{ border-color:{tab_border_color} !important; }}</style>",
    unsafe_allow_html=True,
)

# ================= STATUS BAR (visibility of system status) =================
st.markdown(
    f"<div class='status-bar'>"
    f"<span>STATION: {current.get('station_id', 'unknown')}</span>"
    f"<span>LAST READING: {current['timestamp']}</span>"
    f"<span>PAGE LOADED: {datetime.now().strftime('%H:%M:%S')}</span>"
    f"</div>",
    unsafe_allow_html=True,
)

# ================= HEADER =================
st.title("Islamabad AQI Forecast")

with st.expander("How to read this dashboard"):
    st.markdown(
        "- **The dial** shows the current AQI reading and where it falls on the 0–300 severity scale.\n"
        "- **3-Day Forecast** gives one prediction per day, each from a model trained "
        "specifically for that distance ahead — a 3-day forecast is a harder problem "
        "than a 1-day one, so treat later days as less certain.\n"
        "- **Trend** plots the current reading alongside the three forecast points.\n"
        "- **Why this prediction** shows which factors pushed a given forecast up or "
        "down, using SHAP — useful for sanity-checking the model, not just trusting it blindly.\n"
        "- **Alerts** lists any current or forecast reading at or above the hazard "
        "threshold, shown in the sidebar's AQI scale.\n\n"
        "Forecasts beyond 'now' assume today's weather holds steady, since no live "
        "weather forecast is connected yet — treat 48h/72h numbers as directional."
    )

if alerts["active_alerts"]:
    names = ", ".join(a["when"] for a in alerts["active_alerts"])
    st.error(f"**Hazard alert** — unhealthy or worse AQI expected: {names}. See the Alerts tab for details.")
else:
    st.success("No hazardous AQI currently forecast in the next 3 days.")

# ================= CURRENT CONDITIONS =================
col_gauge, col_detail = st.columns([1, 1.4], gap="large")

with col_gauge:
    st.plotly_chart(aqi_gauge(current["aqi"], current["category"]), use_container_width=True)
    st.markdown(
        f"<div style='text-align:center; margin-top:-14px'>{category_pill(current['category'])}</div>",
        unsafe_allow_html=True,
    )

with col_detail:
    st.markdown("#### Current conditions")
    st.write(current["advice"])
    m1, m2 = st.columns(2)
    m1.metric("Category", current["category"])
    m2.metric("Alert threshold", f"{alerts['threshold_aqi']} AQI")

st.divider()

# ================= TABS (progressive disclosure) =================
tab_forecast, tab_trend, tab_explain, tab_alerts = st.tabs(
    ["3-Day Forecast", "Trend", "Why this prediction", "Alerts"]
)

with tab_forecast:
    st.caption(predictions["caveat"])
    cols = st.columns(3)
    for col, (horizon_key, f) in zip(cols, forecast.items()):
        with col:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='fc-label'>{horizon_key} AHEAD</div>"
                f"<div class='fc-number' style='color:{category_color_deep(f['category'])}'>{f['predicted_aqi']}</div>"
                f"{category_pill(f['category'])}"
                f"<div class='fc-advice'>{f['advice']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

with tab_trend:
    st.caption(
        "Current reading plus the three forecast points. A full historical "
        "trend needs a dedicated /history endpoint reading more than the "
        "latest row from the feature store — not wired up yet."
    )
    points = [current["aqi"]] + [f["predicted_aqi"] for f in forecast.values()]
    labels = ["Now"] + list(forecast.keys())
    point_colors = [category_color(current["category"])] + [category_color(f["category"]) for f in forecast.values()]
    fig = go.Figure(go.Scatter(
        x=labels, y=points, mode="lines+markers",
        line=dict(color=INK, width=2),
        marker=dict(size=10, color=point_colors),
    ))
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="AQI", gridcolor=LINE), xaxis=dict(title=None),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_explain:
    st.caption(
        "SHAP values show how much each feature's current value pushed the "
        "prediction up or down from the model's baseline expectation."
    )
    horizon_label = st.radio("Explain which forecast?", ["24h", "48h", "72h"], horizontal=True)
    horizon_hours = int(horizon_label.replace("h", ""))

    try:
        with st.spinner("Computing feature contributions..."):
            explanation = fetch(f"/explain/{horizon_hours}")
        contrib_df = pd.DataFrame(explanation["top_contributions"])

        st.markdown(
            f"<div class='caveat-box'>Base rate: <b>{explanation['base_value']} AQI</b> → "
            f"adjusted to <b>{explanation['predicted_aqi']} AQI</b> after all feature effects. "
            f"Bars below show the largest individual contributors.</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(contribution_chart(contrib_df, horizon_label), use_container_width=True)

        with st.expander("Raw feature values"):
            st.dataframe(
                contrib_df[["feature", "value", "shap_contribution"]],
                hide_index=True, use_container_width=True,
            )
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e)) if e.response is not None else str(e)
        except (ValueError, AttributeError):
            # Response wasn't valid JSON at all - usually means the server
            # crashed outright rather than returning a clean error response.
            detail = (
                f"{e} (server sent no readable error body — check the "
                f"uvicorn terminal for a traceback)"
            )
        st.warning(f"Couldn't load explanation: {detail}")
    except requests.exceptions.RequestException as e:
        st.warning(f"Couldn't load explanation: {e}")

with tab_alerts:
    if not alerts["active_alerts"]:
        st.success("Nothing to show — all readings and forecasts are below the alert threshold.")
    else:
        for a in alerts["active_alerts"]:
            st.markdown(
                f"<div class='metric-card' style='border-left:4px solid {category_color(a['category'])}; margin-bottom:10px'>"
                f"<b>{a['when'].upper()}</b> — AQI {a['aqi']} {category_pill(a['category'])}"
                f"<div style='margin-top:6px; color:{INK_SOFT}'>{a['advice']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    st.caption(f"Alert threshold: AQI ≥ {alerts['threshold_aqi']} (Unhealthy for Sensitive Groups or worse).")

st.divider()
st.caption(
    "Independent forecasting project, not an official air-quality authority. "
    "For health guidance during severe smog, refer to Pakistan's Ministry of "
    "Climate Change or your local health department."
)

if auto_refresh:
    time.sleep(60)
    st.cache_data.clear()
    st.rerun()