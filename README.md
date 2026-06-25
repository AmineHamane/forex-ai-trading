# EUR/USD AI Trading System — TFT Forecaster + Rule-Based Bot

An end-to-end algorithmic-trading **research project** for EUR/USD covering the full lifecycle:
**data collection → feature engineering → deep-learning forecasting (Temporal Fusion Transformer) → an explainable decision engine → a live, risk-managed execution bot → a real-time dashboard.**

> A research prototype running against an OANDA **practice** account, built to explore the complete ML-to-execution pipeline on real market data and a real broker API. Not financial advice.

---

## What it does

Two complementary tracks:

1. **Live rule-based bot** — polls OANDA every 10s on M1 candles, detects newly *completed* candles, computes Bollinger Bands, and fires mean-reversion entries (subject to spread/gain filters). Take-profit / stop-loss come from a configurable risk:reward, and position size is derived so that a stop-out risks a **fixed cash amount**.
2. **AI / ML track** — collects ~8 years of M5/H1/H4 candles, engineers indicator / EMA / pattern / time features, trains **Temporal Fusion Transformers** (PyTorch Lightning) to predict next-bar log-returns, then an **AI decision engine** fuses the model prediction with per-indicator Buy/Sell/Hold confidence and candlestick-pattern signals into a single, **explainable** signal — surfaced live in a **Streamlit** dashboard with trading-session and economic-news awareness.

## Architecture

```
code/
├─ api/oanda_api.py            # OANDA REST v3 wrapper: candles, pricing (home conversion), market orders w/ TP/SL
├─ infrastructure/             # bulk paged historical collection + instrument metadata + logging
├─ technicals/                 # vectorized pandas indicators (Bollinger, ATR, RSI, MACD, EMA) + candlestick patterns
├─ models/                     # thin dataclasses: TradeDecision, TradeSettings, Instrument, OpenTrade, ApiPrice...
├─ bot/                        # live loop: candle detection → signal → risk sizing → duplicate guard → order
├─ ai_decision_engine.py       # ML inference core: load TFT checkpoint → 1-step dataset → predict → fuse signals
├─ streamlit_app.py            # live dashboard (signal, session coloring, news windows, Plotly charts)
├─ simulation/ma_cross.py      # moving-average-crossover backtester
└─ constants/defs.py           # configuration + credential loading
```

## Tech stack

`Python` · `PyTorch` · `PyTorch Lightning` · `pytorch-forecasting (TemporalFusionTransformer)` · `pandas` · `NumPy` · `Streamlit` · `Plotly` · `OANDA REST API v3` · `requests` · `Jupyter`

## Getting started

### 1. Configuration

Credentials are loaded from the environment, or from a local git-ignored file:

```bash
# Option A — environment variables
export OANDA_API_KEY="your-practice-api-key"
export OANDA_ACCOUNT_ID="your-practice-account-id"

# Option B — local file
cp code/constants/defs.example.py code/constants/defs_local.py   # then fill in your values
```

### 2. Dependencies

```bash
cd code
python -m venv .venv && . .venv/Scripts/activate   # macOS/Linux: source .venv/bin/activate
pip install pandas numpy requests streamlit plotly torch pytorch-lightning pytorch-forecasting python-dateutil streamlit-autorefresh
```

### 3. Run

```bash
streamlit run streamlit_app.py   # live dashboard (TFT + indicator/pattern fusion)
python run_bot.py                # live rule-based bot
```

## Explainable by design

Instead of a black-box score, the decision engine produces a **transparent confidence table**: each indicator (RSI, MACD, EMA-stack trend, divergence, round-level S/R, …) contributes its own Buy / Sell / Hold vote; candlestick patterns are scored separately; and the model's predicted next-bar move (thresholded in pips) is fused on top into a capped global confidence — so every signal is interpretable.

## Project status & scope

- A research prototype against an OANDA practice account.
- The live bot (Bollinger mean-reversion) and the TFT model/dashboard are **two complementary tracks**: the model powers the advisory dashboard rather than placing live trades.
- Reported checkpoint values are **model validation losses**, not trading-performance figures; the project does not include a profitability backtest of the model signal.
- Multiple architectures were explored across the research (TFT at H1/H4/M5, Transformer, ConvLSTM, ConvBiLSTM, CNN).

## Disclaimer

For research and educational purposes only. Trading foreign exchange carries substantial risk. Nothing here is financial advice.
