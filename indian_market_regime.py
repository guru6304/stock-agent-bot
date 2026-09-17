"""Indian Market Regime Detector.

Classifies the prevailing market regime using verified Indian benchmarks:
- Nifty 50 Index (^NSEI) for price trend and market breadth.
- India VIX (^INDIAVIX) for institutional volatility and market fear.

Strictly zero US proxies (no SPY, no US VIX). If benchmarks cannot be verified,
safely degrades to NEUTRAL with explicit blocker notice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from india_market_config import (
    BENCHMARK_INDIA_VIX,
    BENCHMARK_NIFTY_50,
    now_ist,
)

logger = logging.getLogger("indian-market-regime")


@dataclass(frozen=True)
class IndianMarketRegimeResult:
    regime: str                       # BULL_STRONG, BULL_WEAK, NEUTRAL, BEAR_WEAK, BEAR_STRONG
    confidence: int                   # 0 - 100
    description: str
    threshold_adjustment: int         # -1 (more permissive) to +2 (more selective)
    benchmark_verified: bool
    benchmark_symbol: str
    nifty_price: Optional[float]
    nifty_sma200: Optional[float]
    india_vix: Optional[float]
    timestamp_ist: str
    blocker: str = ""

    def to_dict(self) -> Dict:
        return {
            "regime": self.regime,
            "confidence": self.confidence,
            "description": self.description,
            "threshold_adjustment": self.threshold_adjustment,
            "benchmark_verified": self.benchmark_verified,
            "benchmark_symbol": self.benchmark_symbol,
            "nifty_price": self.nifty_price,
            "nifty_sma200": self.nifty_sma200,
            "india_vix": self.india_vix,
            "timestamp_ist": self.timestamp_ist,
            "blocker": self.blocker,
        }


def detect_indian_market_regime(
    nifty_df: Optional[pd.DataFrame] = None,
    vix_df: Optional[pd.DataFrame] = None,
) -> IndianMarketRegimeResult:
    """Detect current Indian market regime using Nifty 50 and India VIX."""
    ts_ist = now_ist().strftime("%Y-%m-%d %H:%M:%S IST")

    # If data not supplied, attempt to fetch via yfinance
    if nifty_df is None:
        try:
            import yfinance as yf
            nifty_df = yf.Ticker(BENCHMARK_NIFTY_50).history(period="1y", interval="1d")
        except Exception as e:
            logger.warning("Could not fetch Nifty 50 benchmark data: %s", e)
            nifty_df = pd.DataFrame()

    if vix_df is None:
        try:
            import yfinance as yf
            vix_df = yf.Ticker(BENCHMARK_INDIA_VIX).history(period="6mo", interval="1d")
        except Exception as e:
            logger.debug("Could not fetch India VIX data: %s", e)
            vix_df = pd.DataFrame()

    if nifty_df.empty or len(nifty_df) < 50:
        return IndianMarketRegimeResult(
            regime="NEUTRAL",
            confidence=50,
            description="Neutral (Benchmark data unverified — strategy running at neutral default)",
            threshold_adjustment=0,
            benchmark_verified=False,
            benchmark_symbol=BENCHMARK_NIFTY_50,
            nifty_price=None,
            nifty_sma200=None,
            india_vix=None,
            timestamp_ist=ts_ist,
            blocker="Nifty 50 historical data unavailable for regime verification",
        )

    # Calculate trend metrics on Nifty 50
    close = nifty_df["Close"].astype(float)
    current_price = float(close.iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(min(200, len(close))).mean().iloc[-1])
    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])

    # India VIX level
    vix_val = float(vix_df["Close"].iloc[-1]) if not vix_df.empty and len(vix_df) > 0 else None

    # Trend scoring
    above_200 = current_price > sma200
    above_50 = current_price > sma50
    ema20_above_50 = ema20 > sma50

    vix_calm = (vix_val is not None) and (vix_val < 16.0)
    vix_fear = (vix_val is not None) and (vix_val > 22.0)

    if above_200 and above_50 and ema20_above_50:
        if vix_calm:
            regime = "BULL_STRONG"
            conf = 85
            desc = "Nifty in strong uptrend above 50/200 SMA with low India VIX. Favorable for momentum & long setups."
            adj = -1
        else:
            regime = "BULL_WEAK"
            conf = 70
            desc = "Nifty in uptrend above 200 SMA but showing volatility/resistance. Standard parameters."
            adj = 0
    elif not above_200 and not above_50:
        if vix_fear:
            regime = "BEAR_STRONG"
            conf = 85
            desc = "Nifty in confirmed downtrend below 50/200 SMA with elevated India VIX. Favor short setups or defensive cash."
            adj = 2
        else:
            regime = "BEAR_WEAK"
            conf = 65
            desc = "Nifty below 200 SMA with fading momentum. High selectivity required."
            adj = 1
    else:
        regime = "NEUTRAL"
        conf = 60
        desc = "Nifty in consolidation / mixed regime between 50 and 200 SMA. Standard selective criteria."
        adj = 0

    return IndianMarketRegimeResult(
        regime=regime,
        confidence=conf,
        description=desc,
        threshold_adjustment=adj,
        benchmark_verified=True,
        benchmark_symbol=BENCHMARK_NIFTY_50,
        nifty_price=round(current_price, 2),
        nifty_sma200=round(sma200, 2),
        india_vix=round(vix_val, 2) if vix_val else None,
        timestamp_ist=ts_ist,
    )
