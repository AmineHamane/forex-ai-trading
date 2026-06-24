# EUR/USD AI Trading System — TFT Forecaster + Rule-Based Bot

An end-to-end algorithmic-trading **research** project for EUR/USD that spans the full lifecycle:
**data collection → feature engineering → deep-learning forecasting (Temporal Fusion Transformer) → an explainable decision engine → a live, risk-managed execution bot → a real-time dashboard.**

> **Status: ambitious solo prototype.** It runs against an OANDA **practice** account. The live trading bot and the TFT model/dashboard are currently **two parallel tracks** (see [Honest status](#honest-status)); there is no automated backtest of the model's trading profitability. This is a learning/research codebase, not financial advice or a production trading system.

---

## What it does

Two complementary tracks:

1. **Live rule-based bot** — polls OANDA every 10s on M1 candles, detects newly *completed* candles, computes Bollinger Bands, and fires mean-reversion entries (subject to spread/gain filters). Take-profit / stop-loss come from a configurable risk:reward, and position size is derived so that a stop-out loses a **fixed cash amount**.
2. **AI / ML track** — collects ~8 years of M5/H1/H4 candles, engineers indicator/EMA/pattern/time features, trains **Temporal Fusion Transformers** (PyTorch Lightning) to predict next-bar log-returns, then an **AI decision engine** fuses the model prediction with per-indicator Buy/Sell/Hold confidence and candlestick-pattern votes into a single, **explainable** signal — surfaced live in a **Streamlit** dashboard with trading-session and economic-news awareness.

## Architecture

```
code/
├─ api/oanda_api.py            # OANDA REST v3 wrapper: candles, pricing (home conversion), market orders w/ TP/SL
├─ infrastructure/             # bulk paged historical collection + instrument metadata + logging
├─ technicals/                 # vectorized pandas indicators (Bollinger, ATR, RSI, MACD, EMA) + candlestick patterns
├─ models/                     # thin dataclasses: TradeDecision, TradeSettings, Instrument, OpenTrade, ApiPrice...
├─ bot/                        # live loop: candle detection → signal → risk sizing → duplicate guard → order
│  ├─ bot.py, candle_manager.py, technicals_manager.py, trade_manager.py, trade_risk_calculator.py
│  └─ settings.json            # traded pairs + per-trade cash risk + risk:reward
├─ ai_decision_engine.py       # ML inference core: load TFT checkpoint → 1-step dataset → predict → fuse signals
├─ streamlit_app.py            # live dashboard (signal, session coloring, news windows, Plotly charts)
├─ simulation/ma_cross.py      # standalone moving-average-crossover backtester
└─ constants/defs.py           # config + credential loading (see Setup)
```

## Tech stack

`Python` · `PyTorch` · `PyTorch Lightning` · `pytorch-forecasting (TemporalFusionTransformer)` · `pandas` · `NumPy` · `Streamlit` · `Plotly` · `OANDA REST API v3` · `requests` · `Jupyter`

## Setup

### 1. Credentials (required)

Credentials are **not** stored in source. Provide them one of two ways:

**Option A — environment variables (recommended):**
```bash
export OANDA_API_KEY="your-practice-api-key"
export OANDA_ACCOUNT_ID="your-practice-account-id"
```

**Option B — local file (git-ignored):**
```bash
cp code/constants/defs.example.py code/constants/defs_local.py
# then edit code/constants/defs_local.py with your values
```

`constants/defs_local.py` and `.env` are listed in `.gitignore` and will never be committed.

> 🔐 **Security note:** the API key was previously hard-coded in `defs.py`. It has been moved out of tracked source. Since it was exposed in plaintext, rotate it in your OANDA dashboard (Manage API Access → revoke & regenerate) and update your local value.

### 2. Dependencies

```bash
cd code
python -m venv .venv && . .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install pandas numpy requests streamlit plotly torch pytorch-lightning pytorch-forecasting python-dateutil streamlit-autorefresh
```

### 3. Run

```bash
# Live dashboard (TFT + indicator/pattern fusion)
streamlit run streamlit_app.py

# Live rule-based bot (OANDA practice account)
python run_bot.py
```

## How the decision engine stays explainable

Rather than emitting a black-box number, the engine produces a **transparent confidence table**: each indicator (RSI, MACD, EMA-stack trend, divergence, round-level S/R, …) contributes its own Buy / Sell / Hold vote; candlestick patterns are scored separately; and the model's predicted next-bar move (thresholded in pips) is fused on top. The blended global confidence is capped, so you can always read *why* a signal fired.

## Honest status

This is a research prototype with rough edges, documented here for transparency:

- **Two parallel systems.** The live bot trades a pure Bollinger mean-reversion rule and **does not use the TFT model**. The TFT is used only by the decision engine + Streamlit dashboard (advisory). No code path lets the model place a live trade yet.
- **Metrics are losses, not profits.** Checkpoint filenames encode validation losses (e.g. best TFT H1 `val_loss ≈ 0.000680`) — these are regression losses on tiny log-returns, **not** trading performance. There is no Sharpe / win-rate / P&L backtest of the TFT signal. The only backtester (`simulation/ma_cross.py`) covers a separate MA-crossover strategy.
- **Experimental breadth.** Multiple architectures were explored (TFT at H1/H4/M5, plain Transformer, ConvLSTM, ConvBiLSTM, CNN) plus a multi-horizon transfer experiment; several remain exploratory.
- **Resilience.** The bot loop exits on any unhandled exception, so it needs a supervisor (e.g. systemd / a process manager) for long-running use.

## Disclaimer

For research and educational purposes only. Trading foreign exchange carries substantial risk. Nothing here is financial advice.
