# ai_decision_engine.py

import numpy as np
import pandas as pd

from api.oanda_api import OandaApi
from technicals.indicators import BollingerBands, ATR, RSI, MACD
from technicals.patterns import apply_patterns

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer


# --- must match training setup ---
FEATURE_COLS_BASE = ["mid_o", "mid_h", "mid_l", "mid_c", "volume"]
FEATURE_COLS_IND = [
    "rsi",
    "macd",
    "macd_signal",
    "bb_lower",
    "bb_middle",
    "bb_upper",
    "atr14",
    "ema_5",
    "ema_20",
    "ema_50",
    "ema_200",
]
FEATURE_COLS_EXT = FEATURE_COLS_BASE + FEATURE_COLS_IND


# ---------------------------------------------------------------------
# 1. Fetch live candles from Oanda
# ---------------------------------------------------------------------
def get_live_candles_df(
    api: OandaApi,
    pair: str = "EUR_USD",
    granularity: str = "H1",
    count: int = 500,
) -> pd.DataFrame:
    """
    Uses your OandaApi.get_candles_df to fetch the last `count` candles.
    Expected columns: time, mid_o, mid_h, mid_l, mid_c, volume
    """
    df = api.get_candles_df(pair, granularity=granularity, count=count)
    if df is None or df.empty:
        raise RuntimeError("No candles returned from Oanda")

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    needed = ["mid_o", "mid_h", "mid_l", "mid_c", "volume"]
    for c in needed:
        if c not in df.columns:
            raise RuntimeError(f"Missing column in candles: {c}")

    return df


# ---------------------------------------------------------------------
# 2. Add indicators
# ---------------------------------------------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add indicators using your existing indicator functions
    (BollingerBands, ATR, RSI, MACD in technicals/indicators.py),
    then map them to the names used in TFT training and add extra features.
    """
    df = df.copy()

    # your existing indicator functions
    df = BollingerBands(df, n=20, s=2)  # BB_MA, BB_UP, BB_LW
    df = ATR(df, n=14)                  # ATR_14
    df = RSI(df, n=14)                  # RSI_14
    df = MACD(df, n_slow=26, n_fast=12, n_signal=9)  # MACD, SIGNAL, HIST

    # Map to TFT feature names
    df["rsi"] = df["RSI_14"]
    df["macd"] = df["MACD"]
    df["macd_signal"] = df["SIGNAL"]
    df["bb_lower"] = df["BB_LW"]
    df["bb_middle"] = df["BB_MA"]
    df["bb_upper"] = df["BB_UP"]
    df["atr14"] = df["ATR_14"]

    # EMAs for trend
    df["ema_5"] = df["mid_c"].ewm(span=5, min_periods=5).mean()
    df["ema_20"] = df["mid_c"].ewm(span=20, min_periods=20).mean()
    df["ema_50"] = df["mid_c"].ewm(span=50, min_periods=50).mean()
    df["ema_200"] = df["mid_c"].ewm(span=200, min_periods=200).mean()

    # Extra helper columns
    df["atr_pips"] = df["atr14"] / 0.0001
    df["atr_pct"] = df["atr14"] / df["mid_c"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

    # Drop warmup NaNs
    df = df.dropna().reset_index(drop=True)
    return df


# ---------------------------------------------------------------------
# 3. Add candlestick patterns
# ---------------------------------------------------------------------
def add_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use technicals.patterns.apply_patterns to compute candlestick pattern flags.

    Adds columns like:
    - HANGING_MAN, SHOOTING_STAR, SPINNING_TOP, MARUBOZU, ENGULFING,
      TWEEZER_TOP, TWEEZER_BOTTOM, MORNING_STAR, EVENING_STAR
    plus geometry fields (direction, body_perc, etc.).
    """
    df = apply_patterns(df)
    return df


# ---------------------------------------------------------------------
# 4. Add time features for TFT + simple session
# ---------------------------------------------------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["series_id"] = "eurusd"
    # minutes since first candle as time_idx
    df["time_idx"] = (df["time"] - df["time"].min()).dt.total_seconds() // 60
    df["time_idx"] = df["time_idx"].astype(int)
    df["hour"] = df["time"].dt.hour.astype(str)
    df["day_of_week"] = df["time"].dt.dayofweek.astype(str)

    hour = df["time"].dt.hour
    session = np.where(
        (hour >= 7) & (hour < 16),
        "London",
        np.where((hour >= 13) & (hour < 22), "NewYork", "Asia/Off"),
    )
    df["session"] = session
    return df


# ---------------------------------------------------------------------
# 5. Helpers: divergence / trend / S-R / round levels
# ---------------------------------------------------------------------
def detect_divergence(df: pd.DataFrame, window: int = 20):
    """
    Very simple divergence detector on last 2 windows.
    Returns "bullish", "bearish" or None.
    """
    if len(df) < 2 * window:
        return None

    close = df["mid_c"].values
    rsi = df["rsi"].values

    p1 = close[-window:]
    p0 = close[-2 * window : -window]

    r1 = rsi[-window:]
    r0 = rsi[-2 * window : -window]

    price_high1, price_high0 = p1.max(), p0.max()
    rsi_high1, rsi_high0 = r1.max(), r0.max()

    price_low1, price_low0 = p1.min(), p0.min()
    rsi_low1, rsi_low0 = r1.min(), r0.min()

    # bearish: price higher high, RSI lower high
    if price_high1 > price_high0 and rsi_high1 < rsi_high0:
        return "bearish"

    # bullish: price lower low, RSI higher low
    if price_low1 < price_low0 and rsi_low1 > rsi_low0:
        return "bullish"

    return None


def detect_trend(last_row: pd.Series):
    """
    Very simple trend classification based on EMAs and price.
    Returns "uptrend", "downtrend" or "range".
    """
    c = last_row["mid_c"]
    e20 = last_row["ema_20"]
    e50 = last_row["ema_50"]
    e200 = last_row["ema_200"]

    if e20 > e50 > e200 and c > e20:
        return "uptrend"
    if e20 < e50 < e200 and c < e20:
        return "downtrend"
    return "range"


def nearest_round_level(price: float, step_pips: float = 25.0):
    """
    Finds nearest round level (like 1.1500 / 1.1525 etc).
    step_pips: distance between levels in pips.
    """
    pip = 0.0001
    step = step_pips * pip
    level = round(price / step) * step
    distance_pips = abs(price - level) / pip
    return level, distance_pips


# ---------------------------------------------------------------------
# 6. Load TFT model
# ---------------------------------------------------------------------
def load_tft_model(checkpoint_path: str) -> TemporalFusionTransformer:
    model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model


# ---------------------------------------------------------------------
# 7. Prediction + indicators + separate pattern decision
# ---------------------------------------------------------------------
def predict_next_signal_live(
    df_feat: pd.DataFrame,
    model: TemporalFusionTransformer,
    max_encoder_length: int = 96,
    threshold_pips: float = 2.0,
):
    """
    Returns:
    - AI + indicators decision (final_signal + confidence)
    - Separate candlestick pattern decision (patterns_block)
    - Per-indicator confidence
    """
    from collections import OrderedDict
    from pytorch_forecasting.data import TimeSeriesDataSet

    df_feat = df_feat.sort_values("time_idx").reset_index(drop=True)

    # Ensure dummy target column exists (same name as during training)
    if "target_return" not in df_feat.columns:
        df_feat["target_return"] = 0.0

    n = len(df_feat)
    if n < 20:
        raise ValueError(
            f"Not enough data after indicators to make a prediction (got {n} rows)."
        )

    # adapt encoder length to available data
    effective_max_enc = min(max_encoder_length, n - 2)
    if effective_max_enc < 10:
        raise ValueError(
            f"Not enough history for encoder length. "
            f"Have {n} rows after indicators, need at least ~12. "
            f"Try increasing 'History candles' in the sidebar."
        )

    df_tail = df_feat.tail(effective_max_enc + 2).copy()

    pred_ds = TimeSeriesDataSet(
        df_tail,
        time_idx="time_idx",
        target="target_return",
        group_ids=["series_id"],
        max_encoder_length=effective_max_enc,
        max_prediction_length=1,
        min_encoder_length=1,
        min_prediction_length=1,
        time_varying_unknown_reals=FEATURE_COLS_EXT,
        time_varying_known_categoricals=["hour", "day_of_week"],
        target_normalizer=None,
        allow_missing_timesteps=True,
    )

    pred_loader = pred_ds.to_dataloader(train=False, batch_size=1, num_workers=0)
    pred_ret = model.predict(pred_loader).detach().cpu().numpy().flatten()[-1]

    last_row = df_tail.iloc[-1]
    current_price = float(last_row["mid_c"])
    predicted_next_price = float(current_price * np.exp(pred_ret))

    # core indicator values
    rsi = float(last_row["rsi"])
    macd = float(last_row["macd"])
    macd_signal = float(last_row["macd_signal"])
    bb_lower = float(last_row["bb_lower"])
    bb_middle = float(last_row["bb_middle"])
    bb_upper = float(last_row["bb_upper"])
    atr14 = float(last_row["atr14"])
    atr_pips = float(last_row["atr_pips"])
    bb_width = float(last_row["bb_width"])
    ema_20 = float(last_row["ema_20"])
    ema_50 = float(last_row["ema_50"])
    ema_200 = float(last_row["ema_200"])
    volume = float(last_row["volume"])
    session = str(last_row.get("session", "Unknown"))

    pip = 0.0001
    move_pips = (predicted_next_price - current_price) / pip
    up_thresh = current_price + threshold_pips * pip
    down_thresh = current_price - threshold_pips * pip

    # basic model-only signal
    if predicted_next_price > up_thresh:
        basic_signal = "BUY"
    elif predicted_next_price < down_thresh:
        basic_signal = "SELL"
    else:
        basic_signal = "HOLD"

    # trend + divergence + S/R info
    trend = detect_trend(last_row)
    divergence = detect_divergence(df_feat)
    sr_level, sr_dist_pips = nearest_round_level(current_price, step_pips=25.0)

    # AI + indicators final signal
    final_signal = "HOLD"
    if basic_signal == "BUY" and (rsi < 70) and (macd > macd_signal) and trend != "downtrend":
        final_signal = "BUY"
    elif basic_signal == "SELL" and (rsi > 30) and (macd < macd_signal) and trend != "uptrend":
        final_signal = "SELL"

    # -----------------------------------------------------------------
    # Per-indicator confidence
    # -----------------------------------------------------------------
    indicators_detail = OrderedDict()

    # 1) RSI
    if rsi < 25:
        rsi_buy, rsi_sell, rsi_hold = 80, 5, 15
    elif rsi < 35:
        rsi_buy, rsi_sell, rsi_hold = 60, 10, 30
    elif rsi > 75:
        rsi_buy, rsi_sell, rsi_hold = 5, 80, 15
    elif rsi > 65:
        rsi_buy, rsi_sell, rsi_hold = 10, 60, 30
    else:
        rsi_buy, rsi_sell, rsi_hold = 25, 25, 50

    indicators_detail["RSI"] = {
        "label": "RSI [Overbought / Oversold strength]",
        "value": rsi,
        "value_fmt": f"{rsi:.2f}",
        "buy_conf": rsi_buy,
        "sell_conf": rsi_sell,
        "hold_conf": rsi_hold,
    }

    # 2) MACD
    if macd > macd_signal and macd > 0:
        m_buy, m_sell, m_hold = 70, 10, 20
    elif macd < macd_signal and macd < 0:
        m_buy, m_sell, m_hold = 10, 70, 20
    elif macd > macd_signal:
        m_buy, m_sell, m_hold = 55, 15, 30
    elif macd < macd_signal:
        m_buy, m_sell, m_hold = 15, 55, 30
    else:
        m_buy, m_sell, m_hold = 30, 30, 40

    indicators_detail["MACD"] = {
        "label": "MACD [Momentum / trend direction]",
        "value": macd,
        "value_fmt": f"{macd:.6f} (signal {macd_signal:.6f})",
        "buy_conf": m_buy,
        "sell_conf": m_sell,
        "hold_conf": m_hold,
    }

    # 3) Trend (EMAs)
    if trend == "uptrend":
        t_buy, t_sell, t_hold = 75, 10, 15
    elif trend == "downtrend":
        t_buy, t_sell, t_hold = 10, 75, 15
    else:
        t_buy, t_sell, t_hold = 30, 30, 40

    indicators_detail["Trend_EMA"] = {
        "label": "Trend (EMA20 / EMA50 / EMA200) [Overall trend direction]",
        "value": trend,
        "value_fmt": f"{trend} (20={ema_20:.5f}, 50={ema_50:.5f}, 200={ema_200:.5f})",
        "buy_conf": t_buy,
        "sell_conf": t_sell,
        "hold_conf": t_hold,
    }

    # 4) Bollinger position
    rng = max(bb_upper - bb_lower, 1e-6)
    price_pos = (current_price - bb_lower) / rng  # 0=lower, 1=upper
    if price_pos < 0.2:
        b_buy, b_sell, b_hold = 75, 5, 20
    elif price_pos < 0.4:
        b_buy, b_sell, b_hold = 55, 15, 30
    elif price_pos > 0.8:
        b_buy, b_sell, b_hold = 5, 75, 20
    elif price_pos > 0.6:
        b_buy, b_sell, b_hold = 15, 55, 30
    else:
        b_buy, b_sell, b_hold = 25, 25, 50

    indicators_detail["Bollinger"] = {
        "label": "Bollinger Bands [Price location in volatility channel]",
        "value": current_price,
        "value_fmt": (
            f"Price {current_price:.5f}, BB L/M/U = "
            f"{bb_lower:.5f}/{bb_middle:.5f}/{bb_upper:.5f}"
        ),
        "buy_conf": b_buy,
        "sell_conf": b_sell,
        "hold_conf": b_hold,
    }

    # 5) Volatility (ATR)
    if atr_pips < 5:
        a_buy, a_sell, a_hold = 10, 10, 80  # too quiet
    elif atr_pips < 25:
        a_buy, a_sell, a_hold = 40, 40, 20  # good trading conditions
    else:
        a_buy, a_sell, a_hold = 25, 25, 50  # very volatile → caution

    indicators_detail["ATR"] = {
        "label": "ATR [Volatility / average candle size in pips]",
        "value": atr_pips,
        "value_fmt": f"{atr_pips:.2f} pips",
        "buy_conf": a_buy,
        "sell_conf": a_sell,
        "hold_conf": a_hold,
    }

    # 6) Session
    if session in ("London", "NewYork"):
        s_buy, s_sell, s_hold = 40, 40, 20
    else:
        s_buy, s_sell, s_hold = 20, 20, 60  # Asia/off: more hold

    indicators_detail["Session"] = {
        "label": "Session [Market session activity level]",
        "value": session,
        "value_fmt": session,
        "buy_conf": s_buy,
        "sell_conf": s_sell,
        "hold_conf": s_hold,
    }

    # 7) Volume
    vol_series = df_feat["volume"].tail(100)
    vol_mean = vol_series.mean()
    vol_std = vol_series.std(ddof=0) or 1.0
    vol_z = (volume - vol_mean) / vol_std

    if vol_z > 1.0:
        v_buy, v_sell, v_hold = 40, 40, 20  # strong participation
    elif vol_z < -1.0:
        v_buy, v_sell, v_hold = 15, 15, 70  # very low volume
    else:
        v_buy, v_sell, v_hold = 30, 30, 40

    indicators_detail["Volume"] = {
        "label": "Volume [Market participation strength]",
        "value": volume,
        "value_fmt": f"{volume:.0f} (z={vol_z:.2f})",
        "buy_conf": v_buy,
        "sell_conf": v_sell,
        "hold_conf": v_hold,
    }

    # 8) Divergence
    if divergence == "bullish":
        d_buy, d_sell, d_hold = 75, 5, 20
        div_text = "Bullish divergence (price down, RSI up)"
    elif divergence == "bearish":
        d_buy, d_sell, d_hold = 5, 75, 20
        div_text = "Bearish divergence (price up, RSI down)"
    else:
        d_buy, d_sell, d_hold = 25, 25, 50
        div_text = "No clear divergence"

    indicators_detail["Divergence"] = {
        "label": "Divergence [RSI vs price reversal signal]",
        "value": divergence or "none",
        "value_fmt": div_text,
        "buy_conf": d_buy,
        "sell_conf": d_sell,
        "hold_conf": d_hold,
    }

    # 9) Support / Resistance distance
    if sr_dist_pips < 10:
        sr_buy, sr_sell, sr_hold = 20, 20, 60  # too close, caution
    elif sr_dist_pips < 30:
        sr_buy, sr_sell, sr_hold = 35, 35, 30
    else:
        sr_buy, sr_sell, sr_hold = 40, 40, 20

    indicators_detail["SupportResistance"] = {
        "label": "Support / Resistance [Distance to nearest round level]",
        "value": sr_level,
        "value_fmt": f"Nearest level {sr_level:.5f}, distance {sr_dist_pips:.1f} pips",
        "buy_conf": sr_buy,
        "sell_conf": sr_sell,
        "hold_conf": sr_hold,
    }

    # -----------------------------------------------------------------
    #  🔥 Separate Candlestick Patterns Block
    # -----------------------------------------------------------------
    pattern_flags = {
        "HANGING_MAN": bool(last_row.get("HANGING_MAN", False)),
        "SHOOTING_STAR": bool(last_row.get("SHOOTING_STAR", False)),
        "SPINNING_TOP": bool(last_row.get("SPINNING_TOP", False)),
        "MARUBOZU": bool(last_row.get("MARUBOZU", False)),
        "ENGULFING": bool(last_row.get("ENGULFING", False)),
        "TWEEZER_TOP": bool(last_row.get("TWEEZER_TOP", False)),
        "TWEEZER_BOTTOM": bool(last_row.get("TWEEZER_BOTTOM", False)),
        "MORNING_STAR": bool(last_row.get("MORNING_STAR", False)),
        "EVENING_STAR": bool(last_row.get("EVENING_STAR", False)),
    }

    direction = int(last_row.get("direction", 0) or 0)  # 1 bull, -1 bear, 0 none

    patterns_text = []
    bull_score = 0.0
    bear_score = 0.0
    indecision_score = 0.0

    if pattern_flags["MORNING_STAR"]:
        patterns_text.append("MORNING STAR – bullish reversal")
        bull_score += 40
    if pattern_flags["EVENING_STAR"]:
        patterns_text.append("EVENING STAR – bearish reversal")
        bear_score += 40
    if pattern_flags["TWEEZER_BOTTOM"]:
        patterns_text.append("TWEEZER BOTTOM – bullish double bottom")
        bull_score += 30
    if pattern_flags["TWEEZER_TOP"]:
        patterns_text.append("TWEEZER TOP – bearish double top")
        bear_score += 30
    if pattern_flags["HANGING_MAN"]:
        patterns_text.append("HANGING MAN – bearish after up move")
        bear_score += 25
    if pattern_flags["SHOOTING_STAR"]:
        patterns_text.append("SHOOTING STAR – bearish rejection wick")
        bear_score += 25
    if pattern_flags["SPINNING_TOP"]:
        patterns_text.append("SPINNING TOP – indecision candle")
        indecision_score += 30
    if pattern_flags["MARUBOZU"]:
        if direction == 1:
            patterns_text.append("BULLISH MARUBOZU – strong trend up candle")
            bull_score += 20
        elif direction == -1:
            patterns_text.append("BEARISH MARUBOZU – strong trend down candle")
            bear_score += 20
        else:
            indecision_score += 10
    if pattern_flags["ENGULFING"]:
        if direction == 1:
            patterns_text.append("BULLISH ENGULFING – strong reversal up")
            bull_score += 35
        elif direction == -1:
            patterns_text.append("BEARISH ENGULFING – strong reversal down")
            bear_score += 35

    if not patterns_text:
        patterns_text.append("No strong candlestick pattern")

    # Decide pattern-only direction
    if bull_score > bear_score and bull_score > indecision_score:
        pattern_signal = "BUY"
        pattern_conf = min(100.0, bull_score)
    elif bear_score > bull_score and bear_score > indecision_score:
        pattern_signal = "SELL"
        pattern_conf = min(100.0, bear_score)
    else:
        pattern_signal = "HOLD"
        pattern_conf = min(100.0, max(indecision_score, 30.0))

    patterns_block = {
        "signal": pattern_signal,
        "confidence": float(pattern_conf),
        "patterns": patterns_text,
    }

    # -----------------------------------------------------------------
    # Global confidence (AI + indicators only, patterns separate)
    # -----------------------------------------------------------------
    move_score = min(abs(move_pips) / threshold_pips, 2.0)  # 0–2
    move_score = (move_score / 2.0) * 50.0                  # 0–50

    dir_inds = ["RSI", "MACD", "Trend_EMA", "Bollinger", "Divergence"]
    if final_signal == "BUY":
        dir_scores = [indicators_detail[k]["buy_conf"] for k in dir_inds]
    elif final_signal == "SELL":
        dir_scores = [indicators_detail[k]["sell_conf"] for k in dir_inds]
    else:
        dir_scores = [indicators_detail[k]["hold_conf"] for k in dir_inds]
    indicators_score = float(np.mean(dir_scores)) * 0.5 / 100.0 * 50.0  # 0–50

    confidence = min(move_score + indicators_score, 100.0)
    if final_signal == "HOLD":
        confidence = min(confidence, 30.0)

    return {
        "current_price": current_price,
        "predicted_return": float(pred_ret),
        "predicted_next_price": predicted_next_price,
        "predicted_move_pips": float(move_pips),
        "rsi": rsi,
        "macd": macd,
        "macd_signal": macd_signal,
        "bb_lower": bb_lower,
        "bb_middle": bb_middle,
        "bb_upper": bb_upper,
        "atr14": atr14,
        "basic_signal": basic_signal,
        "final_signal": final_signal,
        "confidence": float(confidence),
        "indicators_detail": indicators_detail,
        "patterns_block": patterns_block,  # 👈 separate pattern decision
    }


# ---------------------------------------------------------------------
# 8. High-level helper for Streamlit
# ---------------------------------------------------------------------
def get_live_ai_signal(
    api: OandaApi,
    model: TemporalFusionTransformer,
    pair: str = "EUR_USD",
    granularity: str = "H1",
    count: int = 500,
    max_encoder_length: int = 96,
    threshold_pips: float = 2.0,
):
    """
    MAIN FUNCTION used by Streamlit.
    """
    df = get_live_candles_df(api, pair=pair, granularity=granularity, count=count)
    df = add_indicators(df)
    df = add_patterns(df)
    df = add_time_features(df)
    signal = predict_next_signal_live(
        df, model, max_encoder_length=max_encoder_length, threshold_pips=threshold_pips
    )
    return df, signal
