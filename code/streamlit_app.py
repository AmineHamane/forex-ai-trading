# streamlit_app.py

import streamlit as st
import plotly.express as px
from datetime import datetime
from zoneinfo import ZoneInfo

from streamlit_autorefresh import st_autorefresh

from api.oanda_api import OandaApi
from ai_decision_engine import load_tft_model, get_live_ai_signal

CHECKPOINT_PATH = "tft_ret.ckpt"


# ---------------------------------------------------------------------
# Caching: Oanda API + Model
# ---------------------------------------------------------------------
@st.cache_resource
def get_api():
    return OandaApi()


@st.cache_resource
def get_model():
    return load_tft_model(CHECKPOINT_PATH)


# ---------------------------------------------------------------------
# Montreal time + session classification
# ---------------------------------------------------------------------
def get_montreal_time() -> datetime:
    # Montreal uses America/Toronto timezone
    return datetime.now(ZoneInfo("America/Toronto"))


def max_status(a: str, b: str) -> str:
    """
    Combine two statuses and return the "better" one.
    Order: AVOID < CAUTION < GOOD < BEST
    """
    order = {"AVOID": 0, "CAUTION": 1, "GOOD": 2, "BEST": 3}
    return a if order[a] >= order[b] else b


def classify_session_montreal(dt: datetime):
    """
    Returns:
      session_status: "BEST" | "GOOD" | "CAUTION" | "AVOID"
      session_label: human readable explanation
      news_warning: True if we are in a typical US news window
    """
    weekday = dt.weekday()  # 0=Mon ... 4=Fri
    hour = dt.hour + dt.minute / 60.0

    # default
    status = "CAUTION"
    label = "Mixed liquidity – trade with caution"
    news_warning = False

    # Friday late: avoid
    if weekday == 4 and hour >= 13:
        return (
            "AVOID",
            "Friday after 13:00 – banks closing, spreads widen",
            False,
        )

    # Asia session – low liquidity (roughly 19:00–02:00 Montreal)
    if hour >= 19 or hour < 2:
        return (
            "AVOID",
            "Asia session / illiquid hours – avoid trading",
            False,
        )

    # London session (early) 03:00–05:00
    if 3 <= hour < 5:
        status = "GOOD"
        label = "London session – good trend moves"

    # NY pre-market + overlap
    if 7 <= hour < 8:
        status = "GOOD"
        label = "NY pre-market – liquidity rising"

    if 8 <= hour < 11:
        status = "BEST"
        label = "London/NY overlap – BEST trading window for EUR/USD"

    # NY active 11:00–14:00
    if 11 <= hour < 14:
        status = max_status(status, "GOOD")
        label = "New York session – good volume and moves"

    # US afternoon 14:00–19:00
    if 14 <= hour < 19:
        status = max_status(status, "CAUTION")
        label = "US afternoon – can be slower / choppy"

    # Typical US news danger window around 08:30 & 14:00
    if 8 <= hour < 9.5 or 13 <= hour < 15:
        news_warning = True

    return status, label, news_warning


def status_color_hex(status: str) -> str:
    return {
        "BEST": "#00c853",      # green
        "GOOD": "#64dd17",      # light green
        "CAUTION": "#ffab00",   # amber
        "AVOID": "#d50000",     # red
    }.get(status, "#9e9e9e")


# ---------------------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------------------
st.set_page_config(page_title="EURUSD AI Trading Dashboard", layout="wide")

st.title("📈 EURUSD – AI + Indicators + Patterns (Montreal Time)")


api = get_api()
model = get_model()

# ---------------------------------------------------------------------
# Sidebar controls + news links
# ---------------------------------------------------------------------
st.sidebar.header("Settings")

granularity = st.sidebar.selectbox("Timeframe", ["M1", "M5", "H1"], index=2)
threshold_pips = st.sidebar.slider(
    "Prediction threshold (pips)", 1.0, 10.0, 2.0, 0.5
)
max_enc_len = st.sidebar.slider("Encoder length (bars)", 24, 200, 96, 4)
count = st.sidebar.slider("History candles", 200, 1000, 500, 50)

session_filter = st.sidebar.checkbox(
    "Apply session filter (adjust confidence by session quality)", value=True
)

auto_refresh = st.sidebar.checkbox("Auto-refresh every 30s", value=False)

if auto_refresh:
    st_autorefresh(interval=30_000, key="ai-refresh")

# ---- News filter / links ---------------------------------------------
with st.sidebar.expander("📰 News filter & calendars", expanded=False):
    st.markdown(
        """
High-impact **USD/EUR news** can completely change price behavior.
Before trading, always check:

- [Forex Factory Calendar](https://www.forexfactory.com/calendar)
- [Investing.com Economic Calendar](https://www.investing.com/economic-calendar/)
- [Myfxbook Forex Calendar](https://www.myfxbook.com/forex-economic-calendar)

Pay special attention to:
- **NFP, CPI, FOMC, GDP, Retail Sales (USD)**
- **ECB rate decisions, EU inflation (EUR)**

> ⚠ Best practice: avoid new positions **30–60 min before and after** red news.
        """,
        unsafe_allow_html=False,
    )

# ---------------------------------------------------------------------
# Session clock – Montreal time
# ---------------------------------------------------------------------
now_mt = get_montreal_time()
session_status, session_label, news_warning = classify_session_montreal(now_mt)
status_color = status_color_hex(session_status)

clock_col1, clock_col2 = st.columns([2, 3])

with clock_col1:
    st.markdown(
        f"""
<div style="padding: 0.8rem; border-radius: 0.5rem; border: 1px solid #444;">
  <div style="font-size: 0.9rem; color: #888;">Local time (Montreal)</div>
  <div style="font-size: 1.4rem; font-weight: 600;">{now_mt.strftime('%Y-%m-%d %H:%M')}</div>
</div>
        """,
        unsafe_allow_html=True,
    )

with clock_col2:
    st.markdown(
        f"""
<div style="
    padding: 0.8rem;
    border-radius: 0.5rem;
    background-color: {status_color};
    color: #ffffff;
    font-weight: 600;
">
  Session status: {session_status}  
  <div style="font-size: 0.9rem; font-weight: 400;">{session_label}</div>
</div>
        """,
        unsafe_allow_html=True,
    )

if news_warning:
    st.warning(
        "⚠ You are within a **typical US news window** (around 08:30 or 14:00). "
        "Check the economic calendar before trading."
    )

st.markdown("---")


# ---------------------------------------------------------------------
# Run prediction
# ---------------------------------------------------------------------
run_prediction = st.button("🔄 Get latest signal")

# If auto-refresh is on, always run prediction on every rerun
if auto_refresh:
    run_prediction = True

if run_prediction:
    try:
        df_feat, signal = get_live_ai_signal(
            api,
            model,
            pair="EUR_USD",
            granularity=granularity,
            count=count,
            max_encoder_length=max_enc_len,
            threshold_pips=threshold_pips,
        )

        # --- Top metrics -------------------------------------------------
        display_conf = signal["confidence"]
        if session_filter:
            if session_status == "BEST":
                pass
            elif session_status == "GOOD":
                display_conf *= 0.9
            elif session_status == "CAUTION":
                display_conf *= 0.6
            elif session_status == "AVOID":
                display_conf *= 0.3

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current price", f"{signal['current_price']:.5f}")
        with col2:
            st.metric(
                "Predicted next price",
                f"{signal['predicted_next_price']:.5f}",
                f"{signal['predicted_move_pips']:.2f} pips",
            )
        with col3:
            st.metric("AI Final Signal", signal["final_signal"])
        with col4:
            st.metric("AI Confidence (session-adjusted)", f"{display_conf:.0f}%")

        if session_filter and session_status in ("CAUTION", "AVOID"):
            st.info(
                f"Session filter: current time is **{session_status}** – "
                "treat signals as lower quality and consider waiting for a better session."
            )

        # --- Indicators block -------------------------------------------
        st.subheader("Indicators (with explanation + Buy/Sell/Hold confidence)")

        if "indicators_detail" in signal:
            for key, info in signal["indicators_detail"].items():
                st.markdown(
                    f"**{info['label']}**  \n"
                    f"Value: `{info['value_fmt']}`  \n"
                    f"➡ Buy: **{info['buy_conf']:.0f}%**, "
                    f"Sell: **{info['sell_conf']:.0f}%**, "
                    f"Hold: **{info['hold_conf']:.0f}%**"
                )
                st.markdown("---")
        else:
            st.write("No detailed indicator breakdown found in signal data.")

        # --- Candlestick Patterns block --------------------------------
        st.subheader("Candlestick Patterns (pure price action)")

        pat = signal.get("patterns_block", None)
        if pat is not None:
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Pattern decision", pat["signal"])
            with c2:
                st.metric("Pattern confidence", f"{pat['confidence']:.0f}%")

            st.markdown("**Active / detected patterns on the last candle(s):**")
            for txt in pat["patterns"]:
                st.markdown(f"- {txt}")
        else:
            st.write("No pattern block returned by the engine.")

        # --- Price chart -------------------------------------------------
        st.subheader("Price chart (last candles)")

        df_plot = df_feat.tail(200).copy()
        ymin = df_plot["mid_c"].min()
        ymax = df_plot["mid_c"].max()
        pad = max((ymax - ymin) * 0.2, 0.0005)

        fig = px.line(
            df_plot,
            x="time",
            y="mid_c",
            labels={"time": "Time", "mid_c": "Price"},
        )
        fig.update_yaxes(range=[ymin - pad, ymax + pad])
        fig.update_layout(margin=dict(l=40, r=20, t=10, b=40), height=400)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error while computing signal: {e}")

else:
    st.info("Click **Get latest signal** or enable **Auto-refresh** to start.")
