#!/usr/bin/env python3
"""
Market Regime Detector — classifies the current market environment to adjust
trading strategy parameters dynamically.

Regimes:
    BULL_STRONG   — Strong uptrend, above rising 200 SMA, breadth expanding
    BULL_WEAK     — Uptrend fading, above 200 SMA but momentum slowing
    NEUTRAL       — Sideways/range-bound, mixed signals
    BEAR_WEAK     — Downtrend starting, below 200 SMA but not accelerating
    BEAR_STRONG   — Strong downtrend, below falling 200 SMA, breadth contracting

Each regime adjusts:
    - Signal threshold (higher in bear = more selective)
    - Position sizing multiplier (smaller in bear)
    - Preferred trade direction (long bias in bull, short bias in bear)
    - Max open positions

Uses SPY as the primary market proxy, with VIX for volatility context.

Usage:
    python3 market_regime.py              # Show current regime
    python3 market_regime.py --history    # Show regime history
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

REGIME_FILE = Path("logs/market_regime.json")

# Regime classifications
BULL_STRONG = "BULL_STRONG"
BULL_WEAK = "BULL_WEAK"
NEUTRAL = "NEUTRAL"
BEAR_WEAK = "BEAR_WEAK"
BEAR_STRONG = "BEAR_STRONG"

# Strategy adjustments per regime
REGIME_PARAMS = {
    BULL_STRONG: {
        "threshold_adjustment": -1,   # Lower threshold = more signals
        "position_size_mult": 1.2,    # Larger positions
        "max_positions_mult": 1.0,
        "long_bias": True,
        "short_bias": False,
        "description": "Strong uptrend — favor longs, larger size",
    },
    BULL_WEAK: {
        "threshold_adjustment": 0,
        "position_size_mult": 1.0,
        "max_positions_mult": 1.0,
        "long_bias": True,
        "short_bias": False,
        "description": "Weakening uptrend — normal parameters, watch for reversal",
    },
    NEUTRAL: {
        "threshold_adjustment": 1,     # Higher threshold = more selective
        "position_size_mult": 0.8,
        "max_positions_mult": 0.8,
        "long_bias": False,
        "short_bias": False,
        "description": "Sideways — be selective, smaller positions",
    },
    BEAR_WEAK: {
        "threshold_adjustment": 1,
        "position_size_mult": 0.7,
        "max_positions_mult": 0.7,
        "long_bias": False,
        "short_bias": True,
        "description": "Early downtrend — reduce exposure, favor shorts",
    },
    BEAR_STRONG: {
        "threshold_adjustment": 2,     # Very selective
        "position_size_mult": 0.5,     # Half size
        "max_positions_mult": 0.5,
        "long_bias": False,
        "short_bias": True,
        "description": "Strong downtrend — minimal longs, defensive",
    },
}


def compute_sma(series, period):
    # type: (pd.Series, int) -> pd.Series
    """Compute simple moving average."""
    return series.rolling(window=period, min_periods=period).mean()


def compute_ema(series, period):
    # type: (pd.Series, int) -> pd.Series
    """Compute exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(series, period=14):
    # type: (pd.Series, int) -> float
    """Compute RSI of the last value."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    vals = rsi.dropna()
    return float(vals.iloc[-1]) if len(vals) > 0 else 50.0


def analyze_trend(df):
    # type: (pd.DataFrame) -> Dict
    """Analyze price trend using multiple moving averages.

    Args:
        df: OHLCV DataFrame with at least 200 rows.

    Returns:
        Dict with trend metrics.
    """
    if df is None or df.empty or len(df) < 50:
        return {
            "above_50sma": False,
            "above_200sma": False,
            "sma50_slope": 0.0,
            "sma200_slope": 0.0,
            "sma50_above_200": False,
            "price_vs_200sma_pct": 0.0,
            "rsi": 50.0,
        }

    close = df["Close"]

    sma50 = compute_sma(close, 50)
    sma200 = compute_sma(close, min(200, len(close) - 1)) if len(close) > 200 else compute_sma(close, len(close) - 1)

    current_price = float(close.iloc[-1])
    current_sma50 = float(sma50.iloc[-1]) if not pd.isna(sma50.iloc[-1]) else current_price
    current_sma200 = float(sma200.iloc[-1]) if not pd.isna(sma200.iloc[-1]) else current_price

    # Slope of SMAs (5-day rate of change as %)
    sma50_5d_ago = float(sma50.iloc[-6]) if len(sma50) >= 6 and not pd.isna(sma50.iloc[-6]) else current_sma50
    sma200_5d_ago = float(sma200.iloc[-6]) if len(sma200) >= 6 and not pd.isna(sma200.iloc[-6]) else current_sma200

    sma50_slope = ((current_sma50 - sma50_5d_ago) / sma50_5d_ago * 100) if sma50_5d_ago > 0 else 0
    sma200_slope = ((current_sma200 - sma200_5d_ago) / sma200_5d_ago * 100) if sma200_5d_ago > 0 else 0

    price_vs_200 = ((current_price - current_sma200) / current_sma200 * 100) if current_sma200 > 0 else 0

    rsi = compute_rsi(close)

    return {
        "above_50sma": current_price > current_sma50,
        "above_200sma": current_price > current_sma200,
        "sma50_slope": round(sma50_slope, 3),
        "sma200_slope": round(sma200_slope, 3),
        "sma50_above_200": current_sma50 > current_sma200,
        "price_vs_200sma_pct": round(price_vs_200, 2),
        "rsi": round(rsi, 1),
        "current_price": round(current_price, 2),
        "sma50": round(current_sma50, 2),
        "sma200": round(current_sma200, 2),
    }


def analyze_volatility(vix_df):
    # type: (Optional[pd.DataFrame]) -> Dict
    """Analyze volatility using VIX data.

    Args:
        vix_df: OHLCV DataFrame for ^VIX.

    Returns:
        Dict with volatility metrics.
    """
    if vix_df is None or vix_df.empty or len(vix_df) < 10:
        return {
            "vix_current": 20.0,
            "vix_sma20": 20.0,
            "vix_elevated": False,
            "vix_extreme": False,
            "vix_trend": "stable",
        }

    close = vix_df["Close"]
    current = float(close.iloc[-1])
    sma20 = float(compute_sma(close, min(20, len(close))).iloc[-1])

    # VIX thresholds
    elevated = current > 20
    extreme = current > 30

    # Trend: compare current to 5-day ago
    prev = float(close.iloc[-6]) if len(close) >= 6 else current
    if current > prev * 1.1:
        trend = "rising"
    elif current < prev * 0.9:
        trend = "falling"
    else:
        trend = "stable"

    return {
        "vix_current": round(current, 2),
        "vix_sma20": round(sma20, 2),
        "vix_elevated": elevated,
        "vix_extreme": extreme,
        "vix_trend": trend,
    }


def analyze_breadth(df_spy):
    # type: (pd.DataFrame) -> Dict
    """Analyze market breadth using SPY price action.

    Uses percentage of recent days that closed higher as a breadth proxy.
    """
    if df_spy is None or df_spy.empty or len(df_spy) < 20:
        return {"up_day_pct_20d": 50.0, "up_day_pct_5d": 50.0}

    close = df_spy["Close"]

    # % of up days in last 20 and 5 days
    changes_20 = close.tail(21).pct_change().dropna()
    changes_5 = close.tail(6).pct_change().dropna()

    up_20 = (changes_20 > 0).sum() / len(changes_20) * 100 if len(changes_20) > 0 else 50
    up_5 = (changes_5 > 0).sum() / len(changes_5) * 100 if len(changes_5) > 0 else 50

    return {
        "up_day_pct_20d": round(float(up_20), 1),
        "up_day_pct_5d": round(float(up_5), 1),
    }


def classify_regime(trend, volatility, breadth):
    # type: (Dict, Dict, Dict) -> Tuple[str, int, Dict]
    """Classify market regime from trend, volatility, and breadth signals.

    Returns:
        (regime_name, confidence_score, details)
        Confidence is 0-100.
    """
    score = 0  # Positive = bullish, negative = bearish

    # Trend signals (strongest weight)
    if trend["above_200sma"]:
        score += 2
    else:
        score -= 2

    if trend["above_50sma"]:
        score += 1
    else:
        score -= 1

    if trend["sma50_above_200"]:
        score += 1  # Golden cross
    else:
        score -= 1  # Death cross

    if trend["sma200_slope"] > 0.05:
        score += 1
    elif trend["sma200_slope"] < -0.05:
        score -= 1

    if trend["sma50_slope"] > 0.1:
        score += 1
    elif trend["sma50_slope"] < -0.1:
        score -= 1

    # RSI context
    if trend["rsi"] > 60:
        score += 1
    elif trend["rsi"] < 40:
        score -= 1

    # Volatility signals
    if volatility["vix_extreme"]:
        score -= 2  # Extreme fear
    elif volatility["vix_elevated"]:
        score -= 1
    elif volatility["vix_current"] < 15:
        score += 1  # Complacency (can be bull)

    if volatility["vix_trend"] == "rising":
        score -= 1
    elif volatility["vix_trend"] == "falling":
        score += 1

    # Breadth signals
    if breadth["up_day_pct_20d"] > 60:
        score += 1
    elif breadth["up_day_pct_20d"] < 40:
        score -= 1

    # Classify
    if score >= 5:
        regime = BULL_STRONG
    elif score >= 2:
        regime = BULL_WEAK
    elif score <= -5:
        regime = BEAR_STRONG
    elif score <= -2:
        regime = BEAR_WEAK
    else:
        regime = NEUTRAL

    # Confidence based on signal agreement
    max_score = 12  # max possible absolute score
    confidence = min(100, int(abs(score) / max_score * 100))

    details = {
        "raw_score": score,
        "trend": trend,
        "volatility": volatility,
        "breadth": breadth,
    }

    return regime, confidence, details


def detect_regime(spy_data=None, vix_data=None):
    # type: (Optional[pd.DataFrame], Optional[pd.DataFrame]) -> Dict
    """Detect current market regime (supports Nifty 50 & India VIX for Indian markets).

    Args:
        spy_data: OHLCV DataFrame for SPY. If None, fetches automatically.
        vix_data: OHLCV DataFrame for ^VIX. If None, fetches automatically.

    Returns:
        Dict with regime classification and strategy adjustments.
    """
    if spy_data is None and vix_data is None and (os.getenv("MARKET", "").upper() == "INDIA" or os.getenv("TIMEZONE") == "Asia/Kolkata"):
        try:
            from indian_market_regime import detect_indian_market_regime
            ind = detect_indian_market_regime()
            base_params = REGIME_PARAMS.get(ind.regime, REGIME_PARAMS["NEUTRAL"])
            params = dict(base_params)
            params["threshold_adjustment"] = ind.threshold_adjustment
            params["description"] = ind.description
            res = {
                "regime": ind.regime,
                "confidence": ind.confidence,
                "description": ind.description,
                "params": params,
                "details": {
                    "benchmark": ind.benchmark_symbol,
                    "price": ind.nifty_price,
                    "vix": ind.india_vix,
                    "timestamp": ind.timestamp_ist,
                },
                "timestamp": ind.timestamp_ist,
            }
            _save_regime(res)
            return res
        except Exception as e:
            logger.warning("Indian market regime detection failed, falling back: %s", e)

    if spy_data is None:
        try:
            import yfinance as yf
            spy_data = yf.Ticker("SPY").history(period="1y", interval="1d")
        except Exception as e:
            logger.error("Failed to fetch SPY data: %s", e)
            spy_data = pd.DataFrame()

    if vix_data is None:
        try:
            import yfinance as yf
            vix_data = yf.Ticker("^VIX").history(period="6mo", interval="1d")
        except Exception as e:
            logger.error("Failed to fetch VIX data: %s", e)
            vix_data = pd.DataFrame()

    trend = analyze_trend(spy_data)
    volatility = analyze_volatility(vix_data)
    breadth = analyze_breadth(spy_data)

    regime, confidence, details = classify_regime(trend, volatility, breadth)
    params = REGIME_PARAMS[regime]

    result = {
        "regime": regime,
        "confidence": confidence,
        "description": params["description"],
        "params": params,
        "details": details,
        "timestamp": datetime.now().isoformat(),
    }

    # Save to history
    _save_regime(result)

    return result


def get_strategy_adjustments(regime_result=None):
    # type: (Optional[Dict]) -> Dict
    """Get strategy parameter adjustments for the current regime.

    Can be called with a cached regime result or will detect fresh.

    Returns:
        Dict with threshold_adjustment, position_size_mult, etc.
    """
    if regime_result is None:
        regime_result = detect_regime()

    return regime_result["params"]


def _save_regime(result):
    # type: (Dict) -> None
    """Append regime detection to history file."""
    history = _load_history()
    # Keep entry slim for history
    history.append({
        "regime": result["regime"],
        "confidence": result["confidence"],
        "timestamp": result["timestamp"],
        "raw_score": result["details"]["raw_score"],
    })
    # Keep last 365 entries
    history = history[-365:]

    os.makedirs(os.path.dirname(REGIME_FILE) or ".", exist_ok=True)
    with open(REGIME_FILE, "w") as f:
        json.dump(history, f, indent=2)


def _load_history():
    # type: () -> list
    """Load regime detection history."""
    if not REGIME_FILE.exists():
        return []
    try:
        with open(REGIME_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def get_history(limit=30):
    # type: (int) -> list
    """Get recent regime history entries."""
    history = _load_history()
    return list(reversed(history[-limit:]))


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_regime(result):
    # type: (Dict) -> None
    """Print formatted regime report."""
    regime = result["regime"]
    conf = result["confidence"]
    params = result["params"]
    details = result["details"]
    trend = details["trend"]
    vol = details["volatility"]
    breadth = details["breadth"]

    # Regime color indicator
    indicators = {
        BULL_STRONG: "+++",
        BULL_WEAK: "+ ",
        NEUTRAL: " = ",
        BEAR_WEAK: " -",
        BEAR_STRONG: "---",
    }

    print("\n" + "=" * 60)
    print("  MARKET REGIME ANALYSIS")
    print("=" * 60)

    print("\n  Regime:       %s  [%s]  (confidence: %d%%)" % (
        regime, indicators.get(regime, "?"), conf))
    print("  Description:  %s" % params["description"])
    print("  Score:        %+d" % details["raw_score"])

    print("\n  TREND (SPY)")
    print("  " + "-" * 50)
    print("  Price:        $%.2f" % trend.get("current_price", 0))
    print("  vs 200 SMA:   %+.2f%%" % trend["price_vs_200sma_pct"])
    print("  50 SMA slope:  %+.3f%%/week" % trend["sma50_slope"])
    print("  200 SMA slope: %+.3f%%/week" % trend["sma200_slope"])
    print("  RSI:          %.1f" % trend["rsi"])
    print("  Above 50 SMA: %s  |  Above 200 SMA: %s  |  Golden Cross: %s" % (
        trend["above_50sma"], trend["above_200sma"], trend["sma50_above_200"]))

    print("\n  VOLATILITY (VIX)")
    print("  " + "-" * 50)
    print("  VIX:          %.2f  (SMA20: %.2f)" % (vol["vix_current"], vol["vix_sma20"]))
    print("  Status:       %s  |  Trend: %s" % (
        "EXTREME" if vol["vix_extreme"] else "ELEVATED" if vol["vix_elevated"] else "Normal",
        vol["vix_trend"]))

    print("\n  BREADTH")
    print("  " + "-" * 50)
    print("  Up days (20d): %.0f%%  |  Up days (5d): %.0f%%" % (
        breadth["up_day_pct_20d"], breadth["up_day_pct_5d"]))

    print("\n  STRATEGY ADJUSTMENTS")
    print("  " + "-" * 50)
    print("  Threshold adj:  %+d" % params["threshold_adjustment"])
    print("  Position mult:  %.1fx" % params["position_size_mult"])
    print("  Long bias:      %s" % params["long_bias"])
    print("  Short bias:     %s" % params["short_bias"])

    print("\n" + "=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Market Regime Detector")
    parser.add_argument("--history", action="store_true", help="Show regime history")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    for lib in ("urllib3", "yfinance"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    if args.history:
        history = get_history(30)
        if not history:
            print("\n  No regime history found.\n")
            return
        print("\n  REGIME HISTORY (most recent first)")
        print("  " + "=" * 50)
        for h in history:
            print("  %s  %-12s  score=%+d  conf=%d%%" % (
                h["timestamp"][:16], h["regime"], h["raw_score"], h["confidence"]))
        print()
        return

    result = detect_regime()

    if args.json:
        # Remove non-serializable parts
        import copy
        out = copy.deepcopy(result)
        for key in list(out.get("details", {}).get("trend", {}).keys()):
            v = out["details"]["trend"][key]
            if isinstance(v, (np.bool_, np.integer, np.floating)):
                out["details"]["trend"][key] = v.item()
        print(json.dumps(out, indent=2, default=str))
    else:
        print_regime(result)


if __name__ == "__main__":
    main()
