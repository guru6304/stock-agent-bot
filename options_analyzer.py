"""
Options Trading Analyzer — suggests options strategies for any stock signal.

Fetches live options chains via yfinance, evaluates implied volatility context,
and recommends an appropriate strategy (spreads, long options, iron condors, etc.)
based on the directional signal and IV environment.
"""

import argparse
import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val, default: float = 0.0) -> float:
    """Convert a value to float safely, returning *default* on failure."""
    try:
        result = float(val)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def _mid_price(bid: float, ask: float) -> float:
    """Return the midpoint of bid/ask, guarding against missing quotes."""
    bid = _safe_float(bid)
    ask = _safe_float(ask)
    if bid <= 0 and ask <= 0:
        return 0.0
    if bid <= 0:
        return ask
    if ask <= 0:
        return bid
    return round((bid + ask) / 2, 2)


# ---------------------------------------------------------------------------
# Core: fetch_options_data
# ---------------------------------------------------------------------------

def fetch_options_data(ticker: str) -> dict:
    """Fetch options chain, IV estimate, IV rank, and put/call ratio.

    Returns a dict with keys:
        ticker, stock_price, expirations, selected_expiry, iv_estimate,
        iv_rank, iv_percentile, put_call_ratio, calls_df, puts_df,
        days_to_expiry
    """
    empty_result = {
        "ticker": ticker,
        "stock_price": 0.0,
        "expirations": [],
        "selected_expiry": None,
        "iv_estimate": 0.0,
        "iv_rank": 50.0,
        "iv_percentile": 50.0,
        "put_call_ratio": 1.0,
        "calls_df": pd.DataFrame(),
        "puts_df": pd.DataFrame(),
        "days_to_expiry": 0,
        "error": None,
    }

    try:
        from data_layer import resolve_ticker
        stock = yf.Ticker(resolve_ticker(ticker))
    except Exception as exc:
        logger.error("Failed to create Ticker object for %s: %s", ticker, exc)
        empty_result["error"] = str(exc)
        return empty_result

    # --- Current price ---
    try:
        hist = stock.history(period="1d")
        if hist.empty:
            logger.warning("No price history for %s", ticker)
            empty_result["error"] = "No price history"
            return empty_result
        stock_price = float(hist["Close"].iloc[-1])
    except Exception as exc:
        logger.error("Failed to fetch price for %s: %s", ticker, exc)
        empty_result["error"] = str(exc)
        return empty_result

    empty_result["stock_price"] = stock_price

    # --- Expiration dates ---
    try:
        expirations = stock.options  # tuple of date strings
    except Exception as exc:
        logger.error("No options expirations for %s: %s", ticker, exc)
        empty_result["error"] = "No options expirations available"
        return empty_result

    if not expirations:
        empty_result["error"] = "No options expirations available"
        return empty_result

    empty_result["expirations"] = list(expirations)

    # --- Select nearest monthly expiry (>= 14 days, <= 60 days out) ---
    today = datetime.now().date()
    selected_expiry = None
    days_to_expiry = 0

    for exp_str in expirations:
        try:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        dte = (exp_date - today).days
        if 14 <= dte <= 60:
            selected_expiry = exp_str
            days_to_expiry = dte
            break

    # Fallback: pick the first expiry that is at least 7 days out
    if selected_expiry is None:
        for exp_str in expirations:
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            dte = (exp_date - today).days
            if dte >= 7:
                selected_expiry = exp_str
                days_to_expiry = dte
                break

    if selected_expiry is None:
        empty_result["error"] = "No suitable expiration found"
        return empty_result

    empty_result["selected_expiry"] = selected_expiry
    empty_result["days_to_expiry"] = days_to_expiry

    # --- Fetch options chain ---
    try:
        chain = stock.option_chain(selected_expiry)
        calls_df = chain.calls.copy()
        puts_df = chain.puts.copy()
    except Exception as exc:
        logger.error("Failed to fetch chain for %s %s: %s", ticker, selected_expiry, exc)
        empty_result["error"] = str(exc)
        return empty_result

    if calls_df.empty and puts_df.empty:
        empty_result["error"] = "Options chain is empty"
        return empty_result

    empty_result["calls_df"] = calls_df
    empty_result["puts_df"] = puts_df

    # --- IV estimate from ATM options ---
    iv_estimate = _estimate_iv(calls_df, puts_df, stock_price)
    empty_result["iv_estimate"] = iv_estimate

    # --- IV rank / percentile (proxy via historical realised vol) ---
    iv_rank, iv_percentile = _compute_iv_rank(stock, iv_estimate)
    empty_result["iv_rank"] = iv_rank
    empty_result["iv_percentile"] = iv_percentile

    # --- Put/Call ratio from open interest ---
    put_call_ratio = _compute_put_call_ratio(calls_df, puts_df)
    empty_result["put_call_ratio"] = put_call_ratio

    empty_result["error"] = None
    return empty_result


def _estimate_iv(calls_df: pd.DataFrame, puts_df: pd.DataFrame, stock_price: float) -> float:
    """Estimate ATM implied volatility from the options chain."""
    best_iv = 0.0
    for df in [calls_df, puts_df]:
        if df.empty or "strike" not in df.columns:
            continue
        iv_col = "impliedVolatility" if "impliedVolatility" in df.columns else None
        if iv_col is None:
            continue
        idx = (df["strike"] - stock_price).abs().idxmin()
        iv_val = _safe_float(df.loc[idx, iv_col])
        if iv_val > 0:
            if best_iv == 0:
                best_iv = iv_val
            else:
                best_iv = (best_iv + iv_val) / 2.0
    return round(best_iv, 4)


def _compute_iv_rank(stock, current_iv: float) -> Tuple[float, float]:
    """Compute IV rank and percentile using 1-year historical close std-dev as proxy.

    IV Rank = (current_iv - 52wk_low_iv) / (52wk_high_iv - 52wk_low_iv) * 100
    IV Percentile = % of days in the past year where IV was below current level
    """
    try:
        hist = stock.history(period="1y")
        if hist.empty or len(hist) < 30:
            return 50.0, 50.0
    except Exception:
        return 50.0, 50.0

    # Compute rolling 21-day realised vol (annualised) as IV proxy per window
    closes = hist["Close"].dropna()
    log_returns = np.log(closes / closes.shift(1)).dropna()

    if len(log_returns) < 21:
        return 50.0, 50.0

    rolling_vol = log_returns.rolling(window=21).std() * np.sqrt(252)
    rolling_vol = rolling_vol.dropna()

    if rolling_vol.empty:
        return 50.0, 50.0

    vol_min = float(rolling_vol.min())
    vol_max = float(rolling_vol.max())

    # IV Rank
    if vol_max - vol_min > 0.001:
        iv_rank = ((current_iv - vol_min) / (vol_max - vol_min)) * 100.0
    else:
        iv_rank = 50.0
    iv_rank = max(0.0, min(100.0, iv_rank))

    # IV Percentile
    iv_percentile = float((rolling_vol < current_iv).sum()) / len(rolling_vol) * 100.0
    iv_percentile = max(0.0, min(100.0, iv_percentile))

    return round(iv_rank, 1), round(iv_percentile, 1)


def _compute_put_call_ratio(calls_df: pd.DataFrame, puts_df: pd.DataFrame) -> float:
    """Compute put/call ratio from total open interest."""
    call_oi = 0
    put_oi = 0
    if not calls_df.empty and "openInterest" in calls_df.columns:
        call_oi = int(calls_df["openInterest"].fillna(0).sum())
    if not puts_df.empty and "openInterest" in puts_df.columns:
        put_oi = int(puts_df["openInterest"].fillna(0).sum())
    if call_oi == 0:
        return 1.0
    return round(put_oi / call_oi, 2)


# ---------------------------------------------------------------------------
# Core: find_best_strike
# ---------------------------------------------------------------------------

def find_best_strike(chain_df: pd.DataFrame, target_delta: float,
                     option_type: str, stock_price: float = 0.0) -> dict:
    """Find the strike closest to *target_delta* in the chain DataFrame.

    If a ``delta`` column is not present (common with yfinance), we estimate
    delta from moneyness: ATM ~ 0.50, further OTM approaches 0.

    Parameters
    ----------
    chain_df : pd.DataFrame
        The calls or puts DataFrame from yfinance.
    target_delta : float
        Desired absolute delta (e.g. 0.30 for 30-delta).
    option_type : str
        ``"call"`` or ``"put"``.
    stock_price : float
        Current underlying price, used when delta is unavailable.

    Returns
    -------
    dict with keys: strike, bid, ask, mid, volume, openInterest,
    impliedVolatility, delta_est
    """
    empty = {
        "strike": 0.0, "bid": 0.0, "ask": 0.0, "mid": 0.0,
        "volume": 0, "openInterest": 0,
        "impliedVolatility": 0.0, "delta_est": 0.0,
    }
    if chain_df.empty or "strike" not in chain_df.columns:
        return empty

    df = chain_df.copy()

    # Attempt to use real delta column
    has_delta = "delta" in df.columns and df["delta"].notna().any()

    if has_delta:
        df["_abs_delta"] = df["delta"].abs()
        df["_delta_diff"] = (df["_abs_delta"] - abs(target_delta)).abs()
        idx = df["_delta_diff"].idxmin()
        delta_est = _safe_float(df.loc[idx, "delta"])
    else:
        # Estimate delta from moneyness
        if stock_price <= 0:
            return empty
        if option_type.lower() == "call":
            # Call delta decreases as strike goes above stock price
            df["_delta_est"] = df["strike"].apply(
                lambda s: max(0.01, min(0.99, 0.5 - (s - stock_price) / stock_price))
            )
        else:
            # Put delta (absolute) increases as strike goes below stock price
            df["_delta_est"] = df["strike"].apply(
                lambda s: max(0.01, min(0.99, 0.5 + (s - stock_price) / stock_price))
            )
        df["_delta_diff"] = (df["_delta_est"] - abs(target_delta)).abs()
        idx = df["_delta_diff"].idxmin()
        delta_est = _safe_float(df.loc[idx, "_delta_est"])

    row = df.loc[idx]
    bid = _safe_float(row.get("bid"))
    ask = _safe_float(row.get("ask"))

    return {
        "strike": _safe_float(row["strike"]),
        "bid": bid,
        "ask": ask,
        "mid": _mid_price(bid, ask),
        "volume": int(_safe_float(row.get("volume", 0))),
        "openInterest": int(_safe_float(row.get("openInterest", 0))),
        "impliedVolatility": _safe_float(row.get("impliedVolatility")),
        "delta_est": round(delta_est, 3),
    }


# ---------------------------------------------------------------------------
# Core: suggest_strategy
# ---------------------------------------------------------------------------

def suggest_strategy(ticker: str, direction: str, signal_score: int,
                     options_data: dict) -> dict:
    """Suggest an options strategy based on direction, signal strength, and IV.

    Parameters
    ----------
    ticker : str
    direction : str  ``"BUY"`` or ``"SELL"``
    signal_score : int  1-10 strength of the signal
    options_data : dict  Output of :func:`fetch_options_data`

    Returns
    -------
    dict with full strategy recommendation.
    """
    no_data = {
        "strategy_name": "No options data available",
        "direction": direction,
        "legs": [],
        "net_debit_credit": 0.0,
        "max_profit": 0.0,
        "max_loss": 0.0,
        "breakeven": 0.0,
        "risk_reward_ratio": 0.0,
        "probability_of_profit": 0,
        "iv_context": "",
        "rationale": "Options data is unavailable for this ticker.",
    }

    if options_data.get("error"):
        no_data["rationale"] = "Options data error: {}".format(options_data["error"])
        return no_data

    calls_df = options_data.get("calls_df", pd.DataFrame())
    puts_df = options_data.get("puts_df", pd.DataFrame())
    if calls_df.empty and puts_df.empty:
        return no_data

    stock_price = options_data["stock_price"]
    iv_rank = options_data["iv_rank"]
    expiry = options_data["selected_expiry"]
    days_to_expiry = options_data["days_to_expiry"]

    direction = direction.upper()

    # Classify IV environment
    low_iv = iv_rank < 30
    high_iv = iv_rank > 50
    neutral = direction not in ("BUY", "SELL")

    # ----- Strategy selection -----
    if neutral or (direction == "BUY" and signal_score <= 2) or (direction == "SELL" and signal_score <= 2):
        if high_iv:
            return _iron_condor(stock_price, calls_df, puts_df, expiry, days_to_expiry, iv_rank)
        else:
            # Low IV neutral — not ideal, default to a mild directional
            return _iron_condor(stock_price, calls_df, puts_df, expiry, days_to_expiry, iv_rank)

    if direction == "BUY":
        if low_iv:
            return _bull_call_spread(stock_price, calls_df, expiry, days_to_expiry, iv_rank, signal_score)
        else:
            return _bull_put_spread(stock_price, puts_df, expiry, days_to_expiry, iv_rank, signal_score)

    if direction == "SELL":
        if high_iv:
            return _bear_call_spread(stock_price, calls_df, expiry, days_to_expiry, iv_rank, signal_score)
        else:
            return _long_put(stock_price, puts_df, expiry, days_to_expiry, iv_rank, signal_score)

    return no_data


# ---------------------------------------------------------------------------
# Strategy builders
# ---------------------------------------------------------------------------

def _bull_call_spread(stock_price: float, calls_df: pd.DataFrame,
                      expiry: str, dte: int, iv_rank: float,
                      score: int) -> dict:
    """BUY ATM call, SELL OTM call."""
    long_leg = find_best_strike(calls_df, 0.50, "call", stock_price)
    short_leg = find_best_strike(calls_df, 0.30, "call", stock_price)

    # Ensure short strike is above long strike
    if short_leg["strike"] <= long_leg["strike"]:
        # Pick a strike ~5-10% above current price
        target = stock_price * 1.07
        subset = calls_df[calls_df["strike"] >= target]
        if not subset.empty:
            short_leg = find_best_strike(subset, 0.25, "call", stock_price)
        else:
            short_leg["strike"] = long_leg["strike"] + round(stock_price * 0.05, 0)
            short_leg["mid"] = 0.0

    long_premium = long_leg["mid"] if long_leg["mid"] > 0 else long_leg["ask"]
    short_premium = short_leg["mid"] if short_leg["mid"] > 0 else short_leg["bid"]

    net_debit = round(long_premium - short_premium, 2)
    spread_width = round(short_leg["strike"] - long_leg["strike"], 2)
    max_profit = round(spread_width - net_debit, 2) if net_debit > 0 else spread_width
    max_loss = max(net_debit, 0.01)
    breakeven = round(long_leg["strike"] + net_debit, 2)
    rr = round(max_profit / max_loss, 2) if max_loss > 0 else 0.0
    pop = max(10, min(80, int(50 - (net_debit / spread_width * 50)))) if spread_width > 0 else 40

    return {
        "strategy_name": "Bull Call Spread",
        "direction": "BUY",
        "legs": [
            {"action": "BUY", "type": "CALL", "strike": long_leg["strike"],
             "premium": long_premium, "expiry": expiry},
            {"action": "SELL", "type": "CALL", "strike": short_leg["strike"],
             "premium": short_premium, "expiry": expiry},
        ],
        "net_debit_credit": round(-net_debit, 2),
        "max_profit": max_profit,
        "max_loss": max_loss,
        "breakeven": breakeven,
        "risk_reward_ratio": rr,
        "probability_of_profit": pop,
        "iv_context": "Low IV (rank {:.0f}) — buying premium is favorable".format(iv_rank),
        "rationale": (
            "Bullish signal (score {}) with low implied volatility favors long premium "
            "strategies. A bull call spread limits cost while capturing upside to "
            "${:.0f}.".format(score, short_leg["strike"])
        ),
    }


def _bull_put_spread(stock_price: float, puts_df: pd.DataFrame,
                     expiry: str, dte: int, iv_rank: float,
                     score: int) -> dict:
    """SELL ATM/slightly-OTM put, BUY further-OTM put (credit spread)."""
    short_leg = find_best_strike(puts_df, 0.40, "put", stock_price)
    long_leg = find_best_strike(puts_df, 0.20, "put", stock_price)

    if long_leg["strike"] >= short_leg["strike"]:
        target = stock_price * 0.93
        subset = puts_df[puts_df["strike"] <= target]
        if not subset.empty:
            long_leg = find_best_strike(subset, 0.15, "put", stock_price)
        else:
            long_leg["strike"] = short_leg["strike"] - round(stock_price * 0.05, 0)
            long_leg["mid"] = 0.0

    short_premium = short_leg["mid"] if short_leg["mid"] > 0 else short_leg["bid"]
    long_premium = long_leg["mid"] if long_leg["mid"] > 0 else long_leg["ask"]

    net_credit = round(short_premium - long_premium, 2)
    spread_width = round(short_leg["strike"] - long_leg["strike"], 2)
    max_profit = max(net_credit, 0.01)
    max_loss = round(spread_width - net_credit, 2) if spread_width > net_credit else 0.01
    breakeven = round(short_leg["strike"] - net_credit, 2)
    rr = round(max_profit / max_loss, 2) if max_loss > 0 else 0.0
    pop = max(10, min(85, int(50 + (net_credit / spread_width * 50)))) if spread_width > 0 else 55

    return {
        "strategy_name": "Bull Put Spread",
        "direction": "BUY",
        "legs": [
            {"action": "SELL", "type": "PUT", "strike": short_leg["strike"],
             "premium": short_premium, "expiry": expiry},
            {"action": "BUY", "type": "PUT", "strike": long_leg["strike"],
             "premium": long_premium, "expiry": expiry},
        ],
        "net_debit_credit": round(net_credit, 2),
        "max_profit": max_profit,
        "max_loss": max_loss,
        "breakeven": breakeven,
        "risk_reward_ratio": rr,
        "probability_of_profit": pop,
        "iv_context": "High IV (rank {:.0f}) — selling premium is favorable".format(iv_rank),
        "rationale": (
            "Bullish signal (score {}) with elevated IV favors credit spreads. "
            "A bull put spread collects premium while defining risk below "
            "${:.0f}.".format(score, long_leg["strike"])
        ),
    }


def _bear_call_spread(stock_price: float, calls_df: pd.DataFrame,
                      expiry: str, dte: int, iv_rank: float,
                      score: int) -> dict:
    """SELL slightly-OTM call, BUY further-OTM call (credit spread)."""
    short_leg = find_best_strike(calls_df, 0.40, "call", stock_price)
    long_leg = find_best_strike(calls_df, 0.20, "call", stock_price)

    if long_leg["strike"] <= short_leg["strike"]:
        target = stock_price * 1.07
        subset = calls_df[calls_df["strike"] >= target]
        if not subset.empty:
            long_leg = find_best_strike(subset, 0.15, "call", stock_price)
        else:
            long_leg["strike"] = short_leg["strike"] + round(stock_price * 0.05, 0)
            long_leg["mid"] = 0.0

    short_premium = short_leg["mid"] if short_leg["mid"] > 0 else short_leg["bid"]
    long_premium = long_leg["mid"] if long_leg["mid"] > 0 else long_leg["ask"]

    net_credit = round(short_premium - long_premium, 2)
    spread_width = round(long_leg["strike"] - short_leg["strike"], 2)
    max_profit = max(net_credit, 0.01)
    max_loss = round(spread_width - net_credit, 2) if spread_width > net_credit else 0.01
    breakeven = round(short_leg["strike"] + net_credit, 2)
    rr = round(max_profit / max_loss, 2) if max_loss > 0 else 0.0
    pop = max(10, min(85, int(50 + (net_credit / spread_width * 50)))) if spread_width > 0 else 55

    return {
        "strategy_name": "Bear Call Spread",
        "direction": "SELL",
        "legs": [
            {"action": "SELL", "type": "CALL", "strike": short_leg["strike"],
             "premium": short_premium, "expiry": expiry},
            {"action": "BUY", "type": "CALL", "strike": long_leg["strike"],
             "premium": long_premium, "expiry": expiry},
        ],
        "net_debit_credit": round(net_credit, 2),
        "max_profit": max_profit,
        "max_loss": max_loss,
        "breakeven": breakeven,
        "risk_reward_ratio": rr,
        "probability_of_profit": pop,
        "iv_context": "High IV (rank {:.0f}) — selling premium is favorable".format(iv_rank),
        "rationale": (
            "Bearish signal (score {}) with elevated IV favors selling call spreads. "
            "A bear call spread profits if price stays below ${:.0f}.".format(
                score, short_leg["strike"]
            )
        ),
    }


def _long_put(stock_price: float, puts_df: pd.DataFrame,
              expiry: str, dte: int, iv_rank: float,
              score: int) -> dict:
    """BUY ATM or slightly-OTM put."""
    atm_put = find_best_strike(puts_df, 0.45, "put", stock_price)

    premium = atm_put["mid"] if atm_put["mid"] > 0 else atm_put["ask"]
    premium = max(premium, 0.01)
    breakeven = round(atm_put["strike"] - premium, 2)
    max_profit = round(breakeven, 2)  # theoretical max if stock goes to 0
    max_loss = premium

    rr = round(max_profit / max_loss, 2) if max_loss > 0 else 0.0
    pop = max(10, min(60, int(atm_put["delta_est"] * 100))) if atm_put["delta_est"] > 0 else 40

    return {
        "strategy_name": "Long Put",
        "direction": "SELL",
        "legs": [
            {"action": "BUY", "type": "PUT", "strike": atm_put["strike"],
             "premium": premium, "expiry": expiry},
        ],
        "net_debit_credit": round(-premium, 2),
        "max_profit": max_profit,
        "max_loss": max_loss,
        "breakeven": breakeven,
        "risk_reward_ratio": rr,
        "probability_of_profit": pop,
        "iv_context": "Low IV (rank {:.0f}) — buying premium is favorable".format(iv_rank),
        "rationale": (
            "Bearish signal (score {}) with low IV makes buying puts attractive. "
            "Long put profits from a move below ${:.2f}.".format(score, breakeven)
        ),
    }


def _iron_condor(stock_price: float, calls_df: pd.DataFrame,
                 puts_df: pd.DataFrame, expiry: str, dte: int,
                 iv_rank: float) -> dict:
    """Sell OTM put spread + OTM call spread (iron condor)."""
    # Put side: sell ~0.25 delta put, buy ~0.10 delta put
    short_put = find_best_strike(puts_df, 0.25, "put", stock_price)
    long_put = find_best_strike(puts_df, 0.10, "put", stock_price)
    if long_put["strike"] >= short_put["strike"]:
        long_put["strike"] = short_put["strike"] - round(stock_price * 0.05, 0)
        long_put["mid"] = 0.0

    # Call side: sell ~0.25 delta call, buy ~0.10 delta call
    short_call = find_best_strike(calls_df, 0.25, "call", stock_price)
    long_call = find_best_strike(calls_df, 0.10, "call", stock_price)
    if long_call["strike"] <= short_call["strike"]:
        long_call["strike"] = short_call["strike"] + round(stock_price * 0.05, 0)
        long_call["mid"] = 0.0

    sp_prem = short_put["mid"] if short_put["mid"] > 0 else short_put["bid"]
    lp_prem = long_put["mid"] if long_put["mid"] > 0 else long_put["ask"]
    sc_prem = short_call["mid"] if short_call["mid"] > 0 else short_call["bid"]
    lc_prem = long_call["mid"] if long_call["mid"] > 0 else long_call["ask"]

    put_credit = round(sp_prem - lp_prem, 2)
    call_credit = round(sc_prem - lc_prem, 2)
    net_credit = round(put_credit + call_credit, 2)

    put_width = round(short_put["strike"] - long_put["strike"], 2)
    call_width = round(long_call["strike"] - short_call["strike"], 2)
    wider_wing = max(put_width, call_width)

    max_profit = max(net_credit, 0.01)
    max_loss = round(wider_wing - net_credit, 2) if wider_wing > net_credit else 0.01
    be_low = round(short_put["strike"] - net_credit, 2)
    be_high = round(short_call["strike"] + net_credit, 2)
    rr = round(max_profit / max_loss, 2) if max_loss > 0 else 0.0
    pop = max(20, min(80, int(60 + (net_credit / wider_wing * 20)))) if wider_wing > 0 else 55

    return {
        "strategy_name": "Iron Condor",
        "direction": "NEUTRAL",
        "legs": [
            {"action": "BUY", "type": "PUT", "strike": long_put["strike"],
             "premium": lp_prem, "expiry": expiry},
            {"action": "SELL", "type": "PUT", "strike": short_put["strike"],
             "premium": sp_prem, "expiry": expiry},
            {"action": "SELL", "type": "CALL", "strike": short_call["strike"],
             "premium": sc_prem, "expiry": expiry},
            {"action": "BUY", "type": "CALL", "strike": long_call["strike"],
             "premium": lc_prem, "expiry": expiry},
        ],
        "net_debit_credit": round(net_credit, 2),
        "max_profit": max_profit,
        "max_loss": max_loss,
        "breakeven": be_low,  # primary breakeven (lower)
        "breakeven_upper": be_high,
        "risk_reward_ratio": rr,
        "probability_of_profit": pop,
        "iv_context": "High IV (rank {:.0f}) — selling premium via iron condor is favorable".format(iv_rank),
        "rationale": (
            "Neutral outlook with elevated IV favors selling premium on both sides. "
            "Iron condor profits if price stays between ${:.0f} and ${:.0f}.".format(
                be_low, be_high
            )
        ),
    }


# ---------------------------------------------------------------------------
# Core: format_options_report
# ---------------------------------------------------------------------------

def format_options_report(suggestion: dict) -> str:
    """Format the strategy suggestion as a human-readable text block."""
    if suggestion.get("strategy_name") == "No options data available":
        return (
            "OPTIONS STRATEGY: N/A\n"
            "Reason: {}\n".format(suggestion.get("rationale", "No data"))
        )

    lines = []  # type: List[str]
    name = suggestion["strategy_name"]
    expiry_str = ""
    dte_str = ""
    if suggestion["legs"]:
        raw_expiry = suggestion["legs"][0].get("expiry", "")
        if raw_expiry:
            try:
                dt = datetime.strptime(raw_expiry, "%Y-%m-%d")
                expiry_str = dt.strftime("%b %d, %Y")
                dte = (dt.date() - datetime.now().date()).days
                dte_str = " ({} days)".format(dte)
            except ValueError:
                expiry_str = raw_expiry

    lines.append("OPTIONS STRATEGY: {}".format(name))
    if expiry_str:
        lines.append("Expiry: {}{}".format(expiry_str, dte_str))
    lines.append("\u2500" * 40)

    for leg in suggestion["legs"]:
        action = leg["action"].ljust(4)
        otype = leg["type"].ljust(4)
        strike = "${:.0f}".format(leg["strike"]) if leg["strike"] >= 10 else "${:.2f}".format(leg["strike"])
        prem = "${:.2f}".format(leg["premium"])
        lines.append("{}{}  {} strike  @ {}".format(action, otype, strike, prem))

    lines.append("\u2500" * 40)

    ndc = suggestion["net_debit_credit"]
    if ndc < 0:
        lines.append("Net Cost:    ${:.2f} per contract (${:.0f} total)".format(
            abs(ndc), abs(ndc) * 100))
    else:
        lines.append("Net Credit:  ${:.2f} per contract (${:.0f} total)".format(
            ndc, ndc * 100))

    lines.append("Max Profit:  ${:.2f} (${:,.0f})".format(
        suggestion["max_profit"], suggestion["max_profit"] * 100))
    lines.append("Max Loss:    ${:.2f} (${:,.0f})".format(
        suggestion["max_loss"], suggestion["max_loss"] * 100))

    if suggestion.get("breakeven_upper"):
        lines.append("Breakeven:   ${:.2f} / ${:.2f}".format(
            suggestion["breakeven"], suggestion["breakeven_upper"]))
    else:
        lines.append("Breakeven:   ${:.2f}".format(suggestion["breakeven"]))

    lines.append("Risk/Reward: {:.1f}:1".format(suggestion["risk_reward_ratio"]))
    lines.append("Win Prob:    ~{}%".format(suggestion["probability_of_profit"]))
    lines.append("")
    lines.append("IV Context: {}".format(suggestion.get("iv_context", "")))

    if suggestion.get("rationale"):
        lines.append("")
        lines.append("Rationale: {}".format(suggestion["rationale"]))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core: analyze_ticker_options (main entry point)
# ---------------------------------------------------------------------------

def analyze_ticker_options(ticker: str, direction: str = "BUY",
                           signal_score: int = 5) -> dict:
    """Main entry point — fetch data, suggest strategy, return full analysis.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. ``"AAPL"``).
    direction : str
        ``"BUY"`` or ``"SELL"``.
    signal_score : int
        Signal strength from 1 (weak) to 10 (strong).

    Returns
    -------
    dict  Strategy suggestion from :func:`suggest_strategy`, enriched with
    ``options_data`` metadata.
    """
    logger.info("Analyzing options for %s direction=%s score=%d", ticker, direction, signal_score)

    options_data = fetch_options_data(ticker)

    suggestion = suggest_strategy(ticker, direction, signal_score, options_data)

    # Attach useful metadata to the result
    suggestion["ticker"] = ticker
    suggestion["stock_price"] = options_data.get("stock_price", 0.0)
    suggestion["iv_rank"] = options_data.get("iv_rank", 0.0)
    suggestion["iv_percentile"] = options_data.get("iv_percentile", 0.0)
    suggestion["put_call_ratio"] = options_data.get("put_call_ratio", 1.0)
    suggestion["selected_expiry"] = options_data.get("selected_expiry")
    suggestion["days_to_expiry"] = options_data.get("days_to_expiry", 0)

    return suggestion


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Options Trading Analyzer — suggest strategies for any stock signal."
    )
    parser.add_argument("ticker", help="Stock ticker symbol (e.g. AAPL)")
    parser.add_argument(
        "--direction", default="BUY", choices=["BUY", "SELL"],
        help="Signal direction (default: BUY)",
    )
    parser.add_argument(
        "--score", type=int, default=5,
        help="Signal strength 1-10 (default: 5)",
    )
    args = parser.parse_args()

    result = analyze_ticker_options(args.ticker, args.direction, args.score)
    print(format_options_report(result))
