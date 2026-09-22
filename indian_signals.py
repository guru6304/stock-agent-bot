"""Indian Market Signal Engines across 5 Horizons.

Implements deterministic strategies for:
1. Intraday Equity
2. Short-Term Equity
3. Swing Equity
4. Long-Term / Positional Equity
5. Intraday F&O (Index & Stock Options)

No hardcoded tickers or symbols. Strictly paper observation.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, asdict, replace
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from india_market_config import (
    CURRENCY_SYMBOL,
    DEFAULT_SCANNER_CONFIG,
    now_ist,
)
from instrument_discovery import InstrumentMetadata
from market_data_service import CandleData, QuoteData

logger = logging.getLogger("indian-signals")


class SignalHorizon(str, Enum):
    INTRADAY_EQUITY = "INTRADAY_EQUITY"
    SHORT_TERM_EQUITY = "SHORT_TERM_EQUITY"
    SWING_EQUITY = "SWING_EQUITY"
    LONG_TERM_EQUITY = "LONG_TERM_EQUITY"
    INTRADAY_FNO = "INTRADAY_FNO"


@dataclass(frozen=True)
class IndianTradeSignal:
    """Standardized institutional trade signal for the Indian Market."""
    signal_id: str
    horizon: SignalHorizon
    strategy_name: str
    strategy_version: str
    symbol: str
    exchange: str
    token: str
    direction: str                     # "BUY" or "BEARISH_SHORT_SETUP"
    signal_time_ist: str
    data_source: str
    data_timestamp: str
    entry_range_low: float
    entry_range_high: float
    stop_loss: float
    target_1: float
    target_2: float
    risk_reward_ratio: float
    validity_period: str
    thesis: str
    quality_score: float
    fno_details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["horizon"] = self.horizon.value
        return d


def _generate_signal_id(symbol: str, horizon: str, direction: str, date_str: str) -> str:
    """Generate a deterministic unique ID for the signal to prevent duplicates."""
    raw = f"{symbol}:{horizon}:{direction}:{date_str}"
    return f"SIG-{hashlib.sha256(raw.encode()).hexdigest()[:10].upper()}"


# ---------------------------------------------------------------------------
# Technical Calculation Helpers
# ---------------------------------------------------------------------------

def calculate_technicals(df: pd.DataFrame) -> Optional[dict]:
    """Calculate moving averages, ATR, RSI, and volume metrics."""
    if df is None or len(df) < 50:
        return None

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    price = float(close.iloc[-1])

    # EMAs & SMAs
    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
    sma200 = float(close.rolling(window=min(200, len(close))).mean().iloc[-1])

    # ATR (14)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    if atr <= 0:
        atr = price * 0.01

    # RSI (14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    last_gain = float(gain.iloc[-1]) if not pd.isna(gain.iloc[-1]) else 0.0
    last_loss = float(loss.iloc[-1]) if not pd.isna(loss.iloc[-1]) else 0.0
    if last_loss == 0.0:
        rsi = 100.0 if last_gain > 0 else 50.0
    elif last_gain == 0.0:
        rsi = 0.0 if last_loss > 0 else 50.0
    else:
        rs = last_gain / last_loss
        rsi = float(100.0 - (100.0 / (1.0 + rs)))

    # Volume ratio (last day vs 20-day average)
    avg_vol_20 = float(volume.iloc[-21:-1].mean()) if len(volume) > 21 else float(volume.mean())
    vol_ratio = float(volume.iloc[-1] / max(avg_vol_20, 1.0))

    # Breakout levels
    high_3d = float(high.iloc[-4:-1].max()) if len(high) >= 4 else price
    low_3d = float(low.iloc[-4:-1].min()) if len(low) >= 4 else price
    high_20d = float(high.iloc[-21:-1].max()) if len(high) >= 21 else price
    low_20d = float(low.iloc[-21:-1].min()) if len(low) >= 21 else price

    return {
        "price": price,
        "ema20": ema20,
        "ema50": ema50,
        "sma200": sma200,
        "atr": atr,
        "rsi": rsi,
        "vol_ratio": vol_ratio,
        "high_3d": high_3d,
        "low_3d": low_3d,
        "high_20d": high_20d,
        "low_20d": low_20d,
    }


# ---------------------------------------------------------------------------
# Smart Money Concepts (SMC) & Institutional Price Action Helpers
# ---------------------------------------------------------------------------

def detect_liquidity_sweep(df: pd.DataFrame) -> Optional[dict]:
    """Detect institutional liquidity sweep / stop hunt (Turtle Soup).
    Price probes below recent swing low to trigger retail stops, then aggressively
    absorbs liquidity and closes back inside the range with a substantial lower wick.
    """
    if df is None or len(df) < 15:
        return None

    c = df["Close"].astype(float)
    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)

    curr_low = float(l.iloc[-1])
    curr_close = float(c.iloc[-1])
    curr_open = float(o.iloc[-1])
    curr_high = float(h.iloc[-1])

    candle_range = curr_high - curr_low
    if candle_range <= 0:
        return None

    lower_wick = min(curr_open, curr_close) - curr_low
    lower_wick_ratio = lower_wick / candle_range

    # 5 to 15-day prior swing low (excluding current candle)
    prior_lows = l.iloc[-16:-1]
    swing_low = float(prior_lows.min())

    # Sweep condition: dipped below prior swing low, but close reclaimed back above it
    swept = curr_low < swing_low and curr_close > swing_low

    # Institutional absorption: lower rejection wick >= 40% of candle range
    if swept and lower_wick_ratio >= 0.40:
        return {
            "type": "BULLISH_SWEEP",
            "swept_level": swing_low,
            "rejection_low": curr_low,
            "wick_ratio": round(lower_wick_ratio * 100, 1),
            "close": curr_close,
        }
    return None


def detect_fair_value_gap(df: pd.DataFrame) -> Optional[dict]:
    """Detect Institutional Fair Value Gap (FVG) / Imbalance Mitigation.
    Identifies 3-candle imbalance where candle 1 high < candle 3 low,
    and current price is testing/mitigating this institutional demand zone.
    """
    if df is None or len(df) < 10:
        return None

    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)
    curr_price = float(c.iloc[-1])

    # Scan last 3 to 10 candles for a clean bullish FVG
    for i in range(len(df) - 3, max(len(df) - 10, 1), -1):
        c1_high = float(h.iloc[i - 1])
        c3_low = float(l.iloc[i + 1])
        # Bullish FVG: Gap exists between candle 1 high and candle 3 low
        if c3_low > c1_high:
            fvg_low = c1_high
            fvg_high = c3_low
            fvg_mid = (fvg_low + fvg_high) / 2.0

            # Check if current price is mitigating into this gap
            if fvg_low <= curr_price <= fvg_high * 1.01:
                return {
                    "fvg_low": round(fvg_low, 2),
                    "fvg_high": round(fvg_high, 2),
                    "fvg_mid": round(fvg_mid, 2),
                    "age_bars": len(df) - 1 - i,
                }
    return None


def detect_order_block_mss(df: pd.DataFrame) -> Optional[dict]:
    """Detect Market Structure Shift (MSS) + Retest of Institutional Order Block (OB).
    An impulsive displacement candle broke a prior 10-20 day swing high.
    The order block is the last down candle prior to the break.
    """
    if df is None or len(df) < 25:
        return None

    c = df["Close"].astype(float)
    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    curr_price = float(c.iloc[-1])

    # Find prior swing high in window [-25 to -5]
    prior_window = h.iloc[-25:-5]
    if len(prior_window) == 0:
        return None
    swing_high = float(prior_window.max())

    # Check if a recent candle (within last 5 days) created an MSS (closed above swing_high)
    recent_closes = c.iloc[-5:]
    mss_occurred = any(float(rc) > swing_high for rc in recent_closes)

    if mss_occurred:
        # Locate the origin down-candle (Order Block) before the push
        for k in range(len(df) - 6, max(len(df) - 20, 0), -1):
            if float(c.iloc[k]) < float(o.iloc[k]):  # Bearish down candle
                ob_low = float(l.iloc[k])
                ob_high = float(h.iloc[k])
                # Check if current price is retesting into the Order Block
                if ob_low * 0.995 <= curr_price <= ob_high * 1.02:
                    return {
                        "ob_low": round(ob_low, 2),
                        "ob_high": round(ob_high, 2),
                        "swing_high": round(swing_high, 2),
                    }
                break
    return None


def detect_volatility_contraction(df: pd.DataFrame) -> Optional[dict]:
    """Detect Minervini Volatility Contraction Pattern (VCP).
    Successive contraction waves (e.g. 15% -> 7% -> 3%) followed by tight pivot breakout.
    """
    if df is None or len(df) < 35:
        return None

    c = df["Close"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    curr_price = float(c.iloc[-1])

    # Wave 1: 20 to 35 days ago
    w1_low = float(l.iloc[-35:-15].min())
    w1_range = (float(h.iloc[-35:-15].max()) - w1_low) / max(w1_low, 1.0)

    # Wave 2: 10 to 20 days ago
    w2_low = float(l.iloc[-15:-5].min())
    w2_range = (float(h.iloc[-15:-5].max()) - w2_low) / max(w2_low, 1.0)

    # Wave 3: last 5 days tight pivot
    pivot_high = float(h.iloc[-6:-1].max())
    pivot_low = float(l.iloc[-6:-1].min())
    w3_range = (pivot_high - pivot_low) / max(pivot_low, 1.0)

    # VCP criteria: progressive contraction and tight pivot <= 5.5%
    if w1_range > w2_range and w2_range > w3_range and w3_range <= 0.055:
        if curr_price >= pivot_high * 0.998:  # Breaking out or right at pivot
            return {
                "w1_pct": round(w1_range * 100, 1),
                "w2_pct": round(w2_range * 100, 1),
                "pivot_pct": round(w3_range * 100, 1),
                "pivot_high": round(pivot_high, 2),
                "pivot_low": round(pivot_low, 2),
            }
    return None


def detect_wyckoff_volume_absorption(df: pd.DataFrame) -> Optional[dict]:
    """Detect Wyckoff Volume Climax & Institutional Absorption (Stopping Volume / Spring).
    Smart money absorbs high selling volume near major support without allowing price to fall,
    creating a classic volume climax followed by an immediate bullish response.
    """
    if df is None or len(df) < 20:
        return None

    c = df["Close"].astype(float)
    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    v = df["Volume"].astype(float)

    curr_close = float(c.iloc[-1])
    curr_open = float(o.iloc[-1])
    curr_low = float(l.iloc[-1])
    curr_high = float(h.iloc[-1])
    curr_vol = float(v.iloc[-1])

    avg_vol = float(v.iloc[-21:-1].mean()) if len(v) > 21 else float(v.mean())
    vol_multiple = curr_vol / max(avg_vol, 1.0)

    # Near 20-day low / major support
    low_20d = float(l.iloc[-21:-1].min()) if len(l) >= 21 else curr_low
    dist_to_support = abs(curr_low - low_20d) / max(low_20d, 1.0)

    # Stopping volume characteristics: lower rejection tail >= 35% or bullish close
    candle_range = curr_high - curr_low
    if candle_range <= 0:
        return None
    lower_tail = min(curr_open, curr_close) - curr_low
    tail_ratio = lower_tail / candle_range

    is_absorption_bar = (curr_close >= curr_open or tail_ratio >= 0.35)

    if vol_multiple >= 1.6 and dist_to_support <= 0.04 and is_absorption_bar:
        return {
            "vol_multiple": round(vol_multiple, 1),
            "support_level": round(low_20d, 2),
            "tail_pct": round(tail_ratio * 100, 1),
            "absorption_low": round(curr_low, 2),
        }
    return None


def detect_institutional_relative_strength(df: pd.DataFrame, tech: dict) -> Optional[dict]:
    """Detect Institutional Relative Strength Leader (Stage 2 Mark-Up).
    Stocks demonstrating exceptional alpha: trading above all major EMAs (20 > 50 > 200),
    within 3.5% of multi-month highs, showing persistent institutional accumulation.
    """
    if df is None or len(df) < 50:
        return None

    c = df["Close"].astype(float)
    p = tech["price"]
    ema20 = tech["ema20"]
    ema50 = tech["ema50"]
    sma200 = tech["sma200"]

    # Perfect trend template: Price > 20 EMA > 50 EMA > 200 SMA
    if not (p > ema20 > ema50 > sma200):
        return None

    # Within 3.5% of 20-day high (tight consolidation near highs)
    high_20d = tech["high_20d"]
    if p < high_20d * 0.965:
        return None

    # 20-day rate of change positive and strong
    ret_20d = (p - float(c.iloc[-21])) / max(float(c.iloc[-21]), 1.0)
    if ret_20d < 0.035:
        return None

    return {
        "ret_20d_pct": round(ret_20d * 100, 1),
        "high_20d": round(high_20d, 2),
        "alignment": "20 EMA > 50 EMA > 200 SMA",
    }


def get_micro_sentiment_catalyst(symbol: str) -> Optional[str]:
    """Fetch recent market-moving catalyst headlines for the stock.
    Non-blocking, gracefully returns None if unavailable or no positive catalyst found.
    """
    try:
        import yfinance as yf
        clean_sym = symbol.upper().replace(".NS", "").replace("-EQ", "")
        tk = yf.Ticker(f"{clean_sym}.NS")
        news_items = getattr(tk, "news", None) or []
        if not news_items:
            return None

        positive_keywords = {
            "order", "contract", "profit", "surge", "growth", "revenue", "beat",
            "dividend", "acquisition", "expansion", "approval", "record", "bonus",
            "partnership", "deal", "upgrade", "outperform", "target", "capex"
        }

        for item in news_items[:3]:
            content = item.get("content", item)
            title = (
                content.get("title", "")
                if isinstance(content, dict)
                else item.get("title", "")
            )
            title_lower = title.lower()
            matched = [kw for kw in positive_keywords if kw in title_lower]
            if matched:
                return title[:120].strip()
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Horizon Signal Engines
# ---------------------------------------------------------------------------

class IndianSignalEngine:
    """Evaluates candidate instrument candles across the 5 enabled horizons."""

    def __init__(self, config=None):
        self.config = config or DEFAULT_SCANNER_CONFIG

    def evaluate_equity_horizons(
        self,
        inst: InstrumentMetadata,
        candle_data: CandleData,
        live_quote: Optional[QuoteData] = None,
    ) -> List[IndianTradeSignal]:
        """Scan daily candles and evaluate Intraday, Short-Term, Swing, and Long-Term signals."""
        tech = calculate_technicals(candle_data.df)
        if not tech:
            return []

        p = live_quote.price if live_quote else tech["price"]
        atr = tech["atr"]
        ts_ist = now_ist().strftime("%Y-%m-%d %H:%M:%S IST")
        today_str = now_ist().strftime("%Y%m%d")

        # Provenance attribution
        sig_source = live_quote.data_source if live_quote else candle_data.data_source
        sig_ts = live_quote.quote_timestamp if live_quote else candle_data.data_timestamp

        signals: List[IndianTradeSignal] = []

        # -------------------------------------------------------------
        # 0. Intraday Institutional Momentum & Volume Surge Breakout
        # -------------------------------------------------------------
        if (
            p > tech["high_3d"]
            and p > tech["ema20"]
            and tech["vol_ratio"] >= 1.75
            and tech["rsi"] >= 52
        ):
            stop = round(max(p - 1.0 * atr, tech["ema20"]), 2)
            risk = p - stop
            if risk > 0:
                t1 = round(p + 2.0 * risk, 2)
                t2 = round(p + 3.5 * risk, 2)
                rr = round((t1 - p) / risk, 2)
                signals.append(IndianTradeSignal(
                    signal_id=_generate_signal_id(inst.symbol, "INTRADAY", "BUY", today_str),
                    horizon=SignalHorizon.INTRADAY_EQUITY,
                    strategy_name="Intraday Institutional Volume Breakout",
                    strategy_version="1.0.0",
                    symbol=inst.symbol,
                    exchange=inst.exch_seg,
                    token=inst.token,
                    direction="BUY",
                    signal_time_ist=ts_ist,
                    data_source=sig_source,
                    data_timestamp=sig_ts,
                    entry_range_low=round(p * 0.998, 2),
                    entry_range_high=round(p * 1.003, 2),
                    stop_loss=stop,
                    target_1=t1,
                    target_2=t2,
                    risk_reward_ratio=rr,
                    validity_period="Intraday (Square off by 15:15 IST)",
                    thesis=f"Intraday Institutional Velocity: Price breaking 3-day high with {tech['vol_ratio']:.1f}x relative volume surge above 20 EMA (RSI: {tech['rsi']:.1f}).",
                    quality_score=9.4,
                ))

        # -------------------------------------------------------------
        # 1. Short-Term Equity Engine (Multi-day breakout with volume)
        # -------------------------------------------------------------
        if p > tech["high_3d"] and p > tech["ema20"] and tech["vol_ratio"] >= self.config.volume_surge_ratio:
            stop = round(min(tech["ema20"], p - 1.2 * atr), 2)
            risk = p - stop
            if risk > 0:
                t1 = round(p + 1.5 * risk, 2)
                t2 = round(p + 3.0 * risk, 2)
                rr = round((t1 - p) / risk, 2)
                signals.append(IndianTradeSignal(
                    signal_id=_generate_signal_id(inst.symbol, "SHORT_TERM", "BUY", today_str),
                    horizon=SignalHorizon.SHORT_TERM_EQUITY,
                    strategy_name="Multi-Day High Momentum Breakout",
                    strategy_version="1.0.0",
                    symbol=inst.symbol,
                    exchange=inst.exch_seg,
                    token=inst.token,
                    direction="BUY",
                    signal_time_ist=ts_ist,
                    data_source=sig_source,
                    data_timestamp=sig_ts,
                    entry_range_low=round(p * 0.998, 2),
                    entry_range_high=round(p * 1.004, 2),
                    stop_loss=stop,
                    target_1=t1,
                    target_2=t2,
                    risk_reward_ratio=rr,
                    validity_period="1 to 5 trading sessions",
                    thesis=f"Price ({CURRENCY_SYMBOL}{p:.2f}) broke 3-day high ({CURRENCY_SYMBOL}{tech['high_3d']:.2f}) above 20 EMA with {tech['vol_ratio']:.1f}x volume surge.",
                    quality_score=round(min(tech["vol_ratio"] * 2.0, 10.0), 1),
                ))

        # -------------------------------------------------------------
        # 2. Swing Equity Engine (Consolidation breakout + trend alignment)
        # -------------------------------------------------------------
        if p > tech["high_20d"] and tech["ema20"] > tech["ema50"] and tech["vol_ratio"] >= 1.15:
            stop = round(min(tech["ema20"], p - 1.5 * atr), 2)
            risk = p - stop
            if risk > 0:
                t1 = round(p + 2.0 * risk, 2)
                t2 = round(p + 3.5 * risk, 2)
                rr = round((t1 - p) / risk, 2)
                signals.append(IndianTradeSignal(
                    signal_id=_generate_signal_id(inst.symbol, "SWING", "BUY", today_str),
                    horizon=SignalHorizon.SWING_EQUITY,
                    strategy_name="20-Day Range Expansion Trend Follower",
                    strategy_version="1.0.0",
                    symbol=inst.symbol,
                    exchange=inst.exch_seg,
                    token=inst.token,
                    direction="BUY",
                    signal_time_ist=ts_ist,
                    data_source=sig_source,
                    data_timestamp=sig_ts,
                    entry_range_low=round(p * 0.997, 2),
                    entry_range_high=round(p * 1.005, 2),
                    stop_loss=stop,
                    target_1=t1,
                    target_2=t2,
                    risk_reward_ratio=rr,
                    validity_period="2 to 4 weeks",
                    thesis=f"20-day high breakout with 20 EMA > 50 EMA bullish alignment (RSI: {tech['rsi']:.1f}).",
                    quality_score=8.5,
                ))

        # -------------------------------------------------------------
        # 3. Long-Term / Positional Equity Engine (200 SMA pullbacks)
        # -------------------------------------------------------------
        distance_to_sma200 = abs(p - tech["sma200"]) / max(tech["sma200"], 1.0)
        if distance_to_sma200 <= 0.035 and p > tech["sma200"] and tech["ema20"] > tech["ema50"]:
            stop = round(p - 2.5 * atr, 2)
            risk = p - stop
            if risk > 0:
                t1 = round(p * 1.15, 2)
                t2 = round(p * 1.25, 2)
                rr = round((t1 - p) / risk, 2)
                signals.append(IndianTradeSignal(
                    signal_id=_generate_signal_id(inst.symbol, "LONG_TERM", "BUY", today_str),
                    horizon=SignalHorizon.LONG_TERM_EQUITY,
                    strategy_name="200-SMA Institutional Support Anchor",
                    strategy_version="1.0.0",
                    symbol=inst.symbol,
                    exchange=inst.exch_seg,
                    token=inst.token,
                    direction="BUY",
                    signal_time_ist=ts_ist,
                    data_source=sig_source,
                    data_timestamp=sig_ts,
                    entry_range_low=round(p * 0.995, 2),
                    entry_range_high=round(p * 1.005, 2),
                    stop_loss=stop,
                    target_1=t1,
                    target_2=t2,
                    risk_reward_ratio=rr,
                    validity_period="1 to 6 months",
                    thesis=f"Price retesting rising 200 SMA support with intact primary trend and RSI at {tech['rsi']:.1f}.",
                    quality_score=9.0,
                ))

        # -------------------------------------------------------------
        # 4. Wyckoff Volume Climax & Institutional Absorption (Stopping Volume)
        # -------------------------------------------------------------
        wyckoff = detect_wyckoff_volume_absorption(candle_data.df)
        if wyckoff:
            stop = round(wyckoff["absorption_low"] - 0.5 * atr, 2)
            risk = p - stop
            if risk > 0:
                t1 = round(p + 2.0 * risk, 2)
                t2 = round(p + 3.5 * risk, 2)
                rr = round((t1 - p) / risk, 2)
                signals.append(IndianTradeSignal(
                    signal_id=_generate_signal_id(inst.symbol, "WYCKOFF_ABSORPTION", "BUY", today_str),
                    horizon=SignalHorizon.SWING_EQUITY,
                    strategy_name="Wyckoff Volume Climax Absorption",
                    strategy_version="1.0.0",
                    symbol=inst.symbol,
                    exchange=inst.exch_seg,
                    token=inst.token,
                    direction="BUY",
                    signal_time_ist=ts_ist,
                    data_source=sig_source,
                    data_timestamp=sig_ts,
                    entry_range_low=round(p * 0.996, 2),
                    entry_range_high=round(p * 1.004, 2),
                    stop_loss=stop,
                    target_1=t1,
                    target_2=t2,
                    risk_reward_ratio=rr,
                    validity_period="1 to 3 weeks",
                    thesis=f"Wyckoff Volume Climax: Smart money stopping volume ({wyckoff['vol_multiple']}x volume surge) absorbed selling at {CURRENCY_SYMBOL}{wyckoff['support_level']:.2f} support with a {wyckoff['tail_pct']}% rejection tail.",
                    quality_score=9.5,
                ))

        # -------------------------------------------------------------
        # 4b. Institutional Oversold Mean-Reversion (Confirmed Value Rebound)
        # -------------------------------------------------------------
        if tech["rsi"] <= 35 and p >= tech["low_20d"]:
            c_last = float(candle_data.df["Close"].iloc[-1])
            o_last = float(candle_data.df["Open"].iloc[-1])
            # Reversal confirmation: require bullish close or holding support
            if c_last >= o_last:
                stop = round(min(tech["low_20d"], p - 1.5 * atr), 2)
                risk = p - stop
                if risk > 0:
                    t1 = round(p + 1.5 * risk, 2)
                    t2 = round(p + 2.5 * risk, 2)
                    rr = round((t1 - p) / risk, 2)
                    signals.append(IndianTradeSignal(
                        signal_id=_generate_signal_id(inst.symbol, "OVERSOLD_BOUNCE", "BUY", today_str),
                        horizon=SignalHorizon.SWING_EQUITY,
                        strategy_name="Oversold Mean-Reversion Value Setup",
                        strategy_version="1.1.0",
                        symbol=inst.symbol,
                        exchange=inst.exch_seg,
                        token=inst.token,
                        direction="BUY",
                        signal_time_ist=ts_ist,
                        data_source=sig_source,
                        data_timestamp=sig_ts,
                        entry_range_low=round(p * 0.995, 2),
                        entry_range_high=round(p * 1.005, 2),
                        stop_loss=stop,
                        target_1=t1,
                        target_2=t2,
                        risk_reward_ratio=rr,
                        validity_period="1 to 3 weeks",
                        thesis=f"Confirmed Oversold Reversal: Bullish candle reaction (RSI {tech['rsi']:.1f}) presenting favorable mean-reversion risk/reward near support.",
                        quality_score=8.2,
                    ))

        # -------------------------------------------------------------
        # 5. Trend Pullback / EMA Support Rebound (Swing)
        # -------------------------------------------------------------
        pullback_dist = abs(p - tech["ema20"]) / max(tech["ema20"], 1.0)
        if pullback_dist <= 0.02 and p >= tech["ema50"] and tech["ema20"] > tech["ema50"] and 40 <= tech["rsi"] <= 62:
            stop = round(min(tech["ema50"], p - 1.2 * atr), 2)
            risk = p - stop
            if risk > 0:
                t1 = round(p + 1.8 * risk, 2)
                t2 = round(p + 3.0 * risk, 2)
                rr = round((t1 - p) / risk, 2)
                signals.append(IndianTradeSignal(
                    signal_id=_generate_signal_id(inst.symbol, "TREND_PULLBACK", "BUY", today_str),
                    horizon=SignalHorizon.SWING_EQUITY,
                    strategy_name="Bullish EMA20 Trend Pullback Anchor",
                    strategy_version="1.0.0",
                    symbol=inst.symbol,
                    exchange=inst.exch_seg,
                    token=inst.token,
                    direction="BUY",
                    signal_time_ist=ts_ist,
                    data_source=sig_source,
                    data_timestamp=sig_ts,
                    entry_range_low=round(p * 0.997, 2),
                    entry_range_high=round(p * 1.004, 2),
                    stop_loss=stop,
                    target_1=t1,
                    target_2=t2,
                    risk_reward_ratio=rr,
                    validity_period="1 to 3 weeks",
                    thesis=f"Bullish pullback retest of 20 EMA in an established uptrend (RSI {tech['rsi']:.1f}).",
                    quality_score=8.2,
                ))

        # -------------------------------------------------------------
        # 6. Bearish Short-Setup Signal (Explicitly labeled)
        # -------------------------------------------------------------
        if p < tech["low_20d"] and tech["ema20"] < tech["ema50"] and tech["rsi"] < 45:
            stop = round(max(tech["ema20"], p + 1.5 * atr), 2)
            risk = stop - p
            if risk > 0:
                t1 = round(p - 1.5 * risk, 2)
                t2 = round(p - 2.5 * risk, 2)
                rr = round((p - t1) / risk, 2)
                signals.append(IndianTradeSignal(
                    signal_id=_generate_signal_id(inst.symbol, "SWING", "BEARISH_SHORT_SETUP", today_str),
                    horizon=SignalHorizon.SWING_EQUITY,
                    strategy_name="Breakdown Below Consolidation Support",
                    strategy_version="1.0.0",
                    symbol=inst.symbol,
                    exchange=inst.exch_seg,
                    token=inst.token,
                    direction="BEARISH_SHORT_SETUP",
                    signal_time_ist=ts_ist,
                    data_source=sig_source,
                    data_timestamp=sig_ts,
                    entry_range_low=round(p * 0.995, 2),
                    entry_range_high=round(p * 1.003, 2),
                    stop_loss=stop,
                    target_1=t1,
                    target_2=t2,
                    risk_reward_ratio=rr,
                    validity_period="1 to 2 weeks",
                    thesis=f"20-day breakdown below {CURRENCY_SYMBOL}{tech['low_20d']:.2f} under falling 20/50 EMAs. (Note: Short-setup observation only; verify borrow/derivatives eligibility before any action).",
                    quality_score=7.8,
                ))

        # -------------------------------------------------------------
        # 7. Smart Money Liquidity Sweep Reversal (SMC Turtle Soup)
        # -------------------------------------------------------------
        sweep = detect_liquidity_sweep(candle_data.df)
        if sweep:
            stop = round(sweep["rejection_low"] - 0.5 * atr, 2)
            risk = p - stop
            if risk > 0:
                t1 = round(p + 2.0 * risk, 2)
                t2 = round(p + 3.5 * risk, 2)
                rr = round((t1 - p) / risk, 2)
                signals.append(IndianTradeSignal(
                    signal_id=_generate_signal_id(inst.symbol, "SMC_SWEEP", "BUY", today_str),
                    horizon=SignalHorizon.SHORT_TERM_EQUITY,
                    strategy_name="Smart Money Liquidity Sweep Reversal",
                    strategy_version="1.0.0",
                    symbol=inst.symbol,
                    exchange=inst.exch_seg,
                    token=inst.token,
                    direction="BUY",
                    signal_time_ist=ts_ist,
                    data_source=sig_source,
                    data_timestamp=sig_ts,
                    entry_range_low=round(p * 0.996, 2),
                    entry_range_high=round(p * 1.004, 2),
                    stop_loss=stop,
                    target_1=t1,
                    target_2=t2,
                    risk_reward_ratio=rr,
                    validity_period="1 to 5 trading sessions",
                    thesis=f"SMC Liquidity Sweep: Price raided stop-loss liquidity below {CURRENCY_SYMBOL}{sweep['swept_level']:.2f} before institutional absorption printed a {sweep['wick_ratio']}% rejection wick, establishing a high-probability demand reversal.",
                    quality_score=9.6,
                ))

        # -------------------------------------------------------------
        # 8. Institutional Fair Value Gap (FVG) Mitigation (SMC)
        # -------------------------------------------------------------
        fvg = detect_fair_value_gap(candle_data.df)
        if fvg and p >= tech["ema50"] and tech["rsi"] >= 40:
            stop = round(fvg["fvg_low"] - 0.5 * atr, 2)
            risk = p - stop
            if risk > 0:
                t1 = round(p + 2.0 * risk, 2)
                t2 = round(p + 3.0 * risk, 2)
                rr = round((t1 - p) / risk, 2)
                signals.append(IndianTradeSignal(
                    signal_id=_generate_signal_id(inst.symbol, "SMC_FVG", "BUY", today_str),
                    horizon=SignalHorizon.SWING_EQUITY,
                    strategy_name="Institutional Fair Value Gap Mitigation",
                    strategy_version="1.0.0",
                    symbol=inst.symbol,
                    exchange=inst.exch_seg,
                    token=inst.token,
                    direction="BUY",
                    signal_time_ist=ts_ist,
                    data_source=sig_source,
                    data_timestamp=sig_ts,
                    entry_range_low=round(p * 0.997, 2),
                    entry_range_high=round(p * 1.003, 2),
                    stop_loss=stop,
                    target_1=t1,
                    target_2=t2,
                    risk_reward_ratio=rr,
                    validity_period="1 to 3 weeks",
                    thesis=f"SMC Imbalance: Orderly mitigation into institutional Fair Value Gap ({CURRENCY_SYMBOL}{fvg['fvg_low']:.2f} – {CURRENCY_SYMBOL}{fvg['fvg_high']:.2f}) with primary trend alignment, offering an asymmetric entry.",
                    quality_score=9.4,
                ))

        # -------------------------------------------------------------
        # 9. Order Block & Market Structure Shift (MSS / CHoCH)
        # -------------------------------------------------------------
        ob = detect_order_block_mss(candle_data.df)
        if ob and tech["ema20"] > tech["ema50"]:
            stop = round(ob["ob_low"] - 0.5 * atr, 2)
            risk = p - stop
            if risk > 0:
                t1 = round(p + 2.5 * risk, 2)
                t2 = round(p + 4.0 * risk, 2)
                rr = round((t1 - p) / risk, 2)
                signals.append(IndianTradeSignal(
                    signal_id=_generate_signal_id(inst.symbol, "SMC_ORDER_BLOCK", "BUY", today_str),
                    horizon=SignalHorizon.SHORT_TERM_EQUITY,
                    strategy_name="Order Block & Market Structure Shift",
                    strategy_version="1.0.0",
                    symbol=inst.symbol,
                    exchange=inst.exch_seg,
                    token=inst.token,
                    direction="BUY",
                    signal_time_ist=ts_ist,
                    data_source=sig_source,
                    data_timestamp=sig_ts,
                    entry_range_low=round(p * 0.997, 2),
                    entry_range_high=round(p * 1.004, 2),
                    stop_loss=stop,
                    target_1=t1,
                    target_2=t2,
                    risk_reward_ratio=rr,
                    validity_period="1 to 5 trading sessions",
                    thesis=f"SMC Order Block: Retest of bullish institutional order block ({CURRENCY_SYMBOL}{ob['ob_low']:.2f} – {CURRENCY_SYMBOL}{ob['ob_high']:.2f}) following confirmed Market Structure Shift above {CURRENCY_SYMBOL}{ob['swing_high']:.2f}.",
                    quality_score=9.7,
                ))

        # -------------------------------------------------------------
        # 10. Minervini Volatility Contraction Pattern (VCP) Breakout
        # -------------------------------------------------------------
        vcp = detect_volatility_contraction(candle_data.df)
        if vcp and p > tech["sma200"] and tech["vol_ratio"] >= 1.15:
            stop = round(vcp["pivot_low"] - 0.5 * atr, 2)
            risk = p - stop
            if risk > 0:
                t1 = round(p + 2.0 * risk, 2)
                t2 = round(p + 3.5 * risk, 2)
                rr = round((t1 - p) / risk, 2)
                signals.append(IndianTradeSignal(
                    signal_id=_generate_signal_id(inst.symbol, "VCP_BREAKOUT", "BUY", today_str),
                    horizon=SignalHorizon.SWING_EQUITY,
                    strategy_name="Minervini Volatility Contraction Pattern Breakout",
                    strategy_version="1.0.0",
                    symbol=inst.symbol,
                    exchange=inst.exch_seg,
                    token=inst.token,
                    direction="BUY",
                    signal_time_ist=ts_ist,
                    data_source=sig_source,
                    data_timestamp=sig_ts,
                    entry_range_low=round(p * 0.998, 2),
                    entry_range_high=round(p * 1.005, 2),
                    stop_loss=stop,
                    target_1=t1,
                    target_2=t2,
                    risk_reward_ratio=rr,
                    validity_period="2 to 4 weeks",
                    thesis=f"Institutional VCP Breakout: Volatility progressively compressed ({vcp['w1_pct']}% → {vcp['w2_pct']}% → {vcp['pivot_pct']}%) into a micro-pivot; price breaking out of pivot resistance ({CURRENCY_SYMBOL}{vcp['pivot_high']:.2f}) on expanding volume.",
                    quality_score=9.5,
                ))

        # -------------------------------------------------------------
        # 11. Institutional Relative Strength Alpha Leader (Stage 2 Mark-Up)
        # -------------------------------------------------------------
        rs = detect_institutional_relative_strength(candle_data.df, tech)
        if rs and tech["vol_ratio"] >= 1.0:
            stop = round(min(tech["ema20"], p - 1.5 * atr), 2)
            risk = p - stop
            if risk > 0:
                t1 = round(p + 2.0 * risk, 2)
                t2 = round(p + 4.0 * risk, 2)
                rr = round((t1 - p) / risk, 2)
                signals.append(IndianTradeSignal(
                    signal_id=_generate_signal_id(inst.symbol, "ALPHA_RS_LEADER", "BUY", today_str),
                    horizon=SignalHorizon.SWING_EQUITY,
                    strategy_name="Institutional Relative Strength Leader",
                    strategy_version="1.0.0",
                    symbol=inst.symbol,
                    exchange=inst.exch_seg,
                    token=inst.token,
                    direction="BUY",
                    signal_time_ist=ts_ist,
                    data_source=sig_source,
                    data_timestamp=sig_ts,
                    entry_range_low=round(p * 0.997, 2),
                    entry_range_high=round(p * 1.004, 2),
                    stop_loss=stop,
                    target_1=t1,
                    target_2=t2,
                    risk_reward_ratio=rr,
                    validity_period="2 to 6 weeks",
                    thesis=f"Institutional Relative Strength: Stock demonstrates elite alpha ({rs['ret_20d_pct']}% 20-day gain) in structural Stage 2 mark-up ({rs['alignment']}), consolidating near 20-day high ({CURRENCY_SYMBOL}{rs['high_20d']:.2f}).",
                    quality_score=9.8,
                ))

        # Resolve contradictory directional signals if present
        has_buy = any(s.direction == "BUY" for s in signals)
        has_bear = any(s.direction == "BEARISH_SHORT_SETUP" for s in signals)
        if has_buy and has_bear:
            buy_max = max(s.quality_score for s in signals if s.direction == "BUY")
            bear_max = max(s.quality_score for s in signals if s.direction == "BEARISH_SHORT_SETUP")
            if buy_max >= bear_max:
                signals = [s for s in signals if s.direction == "BUY"]
            else:
                signals = [s for s in signals if s.direction == "BEARISH_SHORT_SETUP"]

        # Enrich signals with micro-sentiment news catalyst if available
        if signals:
            catalyst = get_micro_sentiment_catalyst(inst.symbol)
            if catalyst:
                signals = [
                    replace(
                        s,
                        thesis=f"{s.thesis} | ⚡ Micro-Sentiment Catalyst: {catalyst}",
                        quality_score=min(round(s.quality_score + 0.2, 1), 10.0),
                    )
                    for s in signals
                ]

        return signals

    def evaluate_intraday_fno(
        self,
        equity_signal: IndianTradeSignal,
        fno_candidates: List[InstrumentMetadata],
        underlying_spot: float,
    ) -> Optional[IndianTradeSignal]:
        """Select and validate dynamic F&O option contract corresponding to an equity conviction."""
        if not fno_candidates:
            return None

        # CE for BUY, PE for BEARISH_SHORT_SETUP
        target_option_type = "CE" if equity_signal.direction == "BUY" else "PE"

        # Filter active option contracts of desired type
        valid_options = [
            c for c in fno_candidates
            if c.is_fno and c.strike > 0 and c.raw_symbol.endswith(target_option_type)
        ]
        if not valid_options:
            return None

        # Group by nearest expiry date
        valid_options.sort(key=lambda c: (c.expiry or now_ist().date(), abs(c.strike - underlying_spot)))
        nearest_expiry = valid_options[0].expiry

        # Filter contracts in nearest expiry
        expiry_contracts = [c for c in valid_options if c.expiry == nearest_expiry]

        # Select ATM or 1-strike nearest strike
        best_contract = min(expiry_contracts, key=lambda c: abs(c.strike - underlying_spot))

        ts_ist = now_ist().strftime("%Y-%m-%d %H:%M:%S IST")
        today_str = now_ist().strftime("%Y%m%d")

        fno_info = {
            "underlying_symbol": equity_signal.symbol,
            "underlying_spot": underlying_spot,
            "contract_symbol": best_contract.raw_symbol,
            "contract_token": best_contract.token,
            "expiry": best_contract.expiry.isoformat() if best_contract.expiry else "N/A",
            "strike": best_contract.strike,
            "option_type": target_option_type,
            "lot_size": best_contract.lotsize,
        }

        thesis = (
            f"F&O Paper Observation: Dynamic {best_contract.raw_symbol} selected for {equity_signal.symbol} "
            f"({equity_signal.direction}) based on underlying spot ₹{underlying_spot:.2f} near strike ₹{best_contract.strike:.2f}."
        )

        return IndianTradeSignal(
            signal_id=_generate_signal_id(best_contract.symbol, "INTRADAY_FNO", equity_signal.direction, today_str),
            horizon=SignalHorizon.INTRADAY_FNO,
            strategy_name=f"Intraday ATM {target_option_type} Option Momentum",
            strategy_version="1.0.0",
            symbol=equity_signal.symbol,
            exchange="NFO",
            token=best_contract.token,
            direction=equity_signal.direction,
            signal_time_ist=ts_ist,
            data_source=equity_signal.data_source,
            data_timestamp=equity_signal.data_timestamp,
            entry_range_low=equity_signal.entry_range_low,
            entry_range_high=equity_signal.entry_range_high,
            stop_loss=equity_signal.stop_loss,
            target_1=equity_signal.target_1,
            target_2=equity_signal.target_2,
            risk_reward_ratio=equity_signal.risk_reward_ratio,
            validity_period="Valid until 15:15 IST",
            thesis=thesis,
            quality_score=equity_signal.quality_score,
            fno_details=fno_info,
        )
