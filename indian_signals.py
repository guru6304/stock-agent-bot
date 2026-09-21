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
from dataclasses import dataclass, asdict
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
    rs = gain / loss.replace(0, np.nan)
    rsi = float((100 - (100 / (1 + rs))).iloc[-1]) if not pd.isna(rs.iloc[-1]) else 50.0

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
        # 4. Oversold Mean-Reversion Value Bounce (Swing / Positional)
        # -------------------------------------------------------------
        if tech["rsi"] <= 35:
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
                    validity_period="1 to 3 weeks",
                    thesis=f"Deep oversold condition (RSI {tech['rsi']:.1f}) presenting favorable mean-reversion risk/reward near support.",
                    quality_score=7.8,
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
