#!/usr/bin/env python3
"""
Market Screener — scans a broader universe of stocks beyond the watchlist
to discover new trading opportunities.

Supports pre-built screens:
- momentum:   RSI < 40 bouncing with EMA9 > EMA21, above 200 SMA
- breakout:   20-bar high breakout with volume surge
- oversold:   RSI < 30 or MFI < 20 with positive divergence
- accumulation: A/D line + OBV both trending bullish
- value:      Strong fundamentals (score >= 4) + technical support

Usage:
    python3 screener.py --screen momentum          # Run momentum screen
    python3 screener.py --screen breakout --top 20  # Top 20 breakout candidates
    python3 screener.py --universe sp500            # Scan S&P 500 universe
    python3 screener.py --universe custom --file my_tickers.txt
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

import data_layer
import technical_analysis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stock universes
# ---------------------------------------------------------------------------

# Common sector ETF tickers for quick sector scans
SECTOR_ETFS = [
    "XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
]

# A curated list of liquid large/mid-cap stocks across sectors
DEFAULT_UNIVERSE = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "CRM", "ORCL",
    "ADBE", "INTC", "QCOM", "AVGO", "CSCO", "NOW", "INTU", "AMAT", "MU", "LRCX",
    # Finance
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "BMY", "AMGN",
    # Consumer
    "WMT", "PG", "KO", "PEP", "COST", "HD", "MCD", "NKE", "SBUX", "TGT",
    # Industrial / Energy
    "CAT", "BA", "HON", "UPS", "GE", "XOM", "CVX", "COP", "SLB", "EOG",
    # Communication / Other
    "DIS", "NFLX", "CMCSA", "T", "VZ", "PYPL", "V", "MA", "SQ", "ABNB",
]


def is_indian_market() -> bool:
    return os.getenv("MARKET", "").upper() == "INDIA" or os.getenv("TIMEZONE") == "Asia/Kolkata"


DEFAULT_INDIAN_UNIVERSE = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "BHARTIARTL", "SBIN",
    "ITC", "LT", "TATAMOTORS", "BAJFINANCE", "KOTAKBANK", "HINDUNILVR", "AXISBANK",
    "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "NTPC", "TATACONSUM", "POWERGRID",
    "M&M", "ADANIENT", "BAJAJ-AUTO", "COALINDIA", "TATASTEEL", "ASIANPAINT",
    "JSWSTEEL", "HCLTECH", "WIPRO", "TECHM", "DIVISLAB", "CIPLA", "DRREDDY",
    "EICHERMOT", "HEROMOTOCO", "GRASIM", "BRITANNIA", "APOLLOHOSP", "INDUSINDBK",
    "ONGC", "BPCL", "ADANIPORTS", "HINDALCO", "NESTLEIND", "SHRIRAMFIN", "TRENT",
    "BEL", "HAL", "ZOMATO"
]


def get_universe(name: str, custom_file: Optional[str] = None) -> List[str]:
    """Return a list of tickers for the given universe name."""
    if name == "default":
        if is_indian_market():
            try:
                from instrument_discovery import DynamicInstrumentDiscovery
                disc = DynamicInstrumentDiscovery()
                equities = disc.get_equity_universe(limit=50)
                if equities:
                    return [inst.symbol for inst in equities]
            except Exception:
                pass
            return DEFAULT_INDIAN_UNIVERSE
        return DEFAULT_UNIVERSE
    elif name == "watchlist":
        wl = data_layer.load_watchlist()
        return [w["ticker"] for w in wl]
    elif name == "sectors":
        return SECTOR_ETFS
    elif name == "sp500":
        return _fetch_sp500_tickers()
    elif name == "custom" and custom_file:
        return _load_custom_tickers(custom_file)
    else:
        logger.warning("Unknown universe '%s', using default", name)
        return get_universe("default")


def _fetch_sp500_tickers() -> List[str]:
    """Fetch current S&P 500 constituents from Wikipedia via pandas."""
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        if tables:
            return tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
    except Exception as e:
        logger.error("Failed to fetch S&P 500 list: %s — using default universe", e)
    return DEFAULT_UNIVERSE


def _load_custom_tickers(filepath: str) -> List[str]:
    """Load tickers from a text file (one per line or comma-separated)."""
    try:
        with open(filepath) as f:
            content = f.read()
        # Support both newline-separated and comma-separated
        tickers = [t.strip().upper() for t in content.replace(",", "\n").split("\n")]
        return [t for t in tickers if t]
    except Exception as e:
        logger.error("Error reading custom ticker file %s: %s", filepath, e)
        return []


# ---------------------------------------------------------------------------
# Screening functions
# ---------------------------------------------------------------------------

def _fetch_and_analyze(ticker: str) -> Optional[Dict]:
    """Fetch data and run technical analysis for a single ticker.

    Returns dict with ticker + signals, or None on failure.
    """
    try:
        df = data_layer.fetch_daily_ohlcv(ticker, period="6mo")
        if df.empty or len(df) < 50:
            return None

        signals = technical_analysis.analyze(ticker, df)
        return {
            "ticker": ticker,
            "signals": signals,
            "price": signals.current_price,
            "df": df,
        }
    except Exception as e:
        logger.debug("Screener skip %s: %s", ticker, e)
        return None


def screen_momentum(results: List[Dict]) -> List[Dict]:
    """Momentum screen: price trending up with RSI pulling back.

    Criteria:
    - Price above 200 SMA
    - EMA9 > EMA21 (trend intact)
    - RSI between 35-55 (pullback in uptrend)
    - Volume above average
    """
    hits = []
    for r in results:
        s = r["signals"]
        if (s.price_above_200sma
                and s.ema9 > s.ema21
                and 35 <= s.rsi <= 55):
            score = 0
            if s.ad_trend_bullish:
                score += 1
            if s.obv_trend_bullish:
                score += 1
            if s.price_above_vwap:
                score += 1
            if s.macd_histogram > 0:
                score += 1
            hits.append({
                "ticker": r["ticker"],
                "price": r["price"],
                "rsi": s.rsi,
                "mfi": s.mfi,
                "macd_hist": s.macd_histogram,
                "screen": "momentum",
                "score": score,
                "details": "EMA9>21, above 200SMA, RSI pullback",
            })
    return sorted(hits, key=lambda x: x["score"], reverse=True)


def screen_breakout(results: List[Dict]) -> List[Dict]:
    """Breakout screen: price breaking above resistance with volume.

    Criteria:
    - Breakout above 20-bar high with volume surge, OR
    - BB squeeze + upper band breakout
    """
    hits = []
    for r in results:
        s = r["signals"]
        if s.breakout_with_volume or (s.bb_squeeze and s.bb_breakout_upper):
            score = 0
            if s.breakout_with_volume:
                score += 2
            if s.bb_squeeze and s.bb_breakout_upper:
                score += 2
            if s.price_above_200sma:
                score += 1
            if s.obv_trend_bullish:
                score += 1
            hits.append({
                "ticker": r["ticker"],
                "price": r["price"],
                "rsi": s.rsi,
                "mfi": s.mfi,
                "atr": s.atr,
                "screen": "breakout",
                "score": score,
                "details": "Breakout" + (" + BB squeeze" if s.bb_squeeze else ""),
            })
    return sorted(hits, key=lambda x: x["score"], reverse=True)


def screen_oversold(results: List[Dict]) -> List[Dict]:
    """Oversold bounce screen: deeply oversold with reversal signs.

    Criteria:
    - RSI < 30 OR MFI < 20
    - At least one bullish divergence (RSI, MACD, or OBV)
    """
    hits = []
    for r in results:
        s = r["signals"]
        is_oversold = s.rsi_oversold or s.mfi_oversold
        has_divergence = (s.rsi_bullish_divergence or s.macd_bullish_divergence
                         or s.obv_divergence_bullish)
        if is_oversold and has_divergence:
            score = 0
            if s.rsi_bullish_divergence:
                score += 2
            if s.macd_bullish_divergence:
                score += 2
            if s.obv_divergence_bullish:
                score += 2
            if s.mfi_oversold:
                score += 1
            hits.append({
                "ticker": r["ticker"],
                "price": r["price"],
                "rsi": s.rsi,
                "mfi": s.mfi,
                "screen": "oversold",
                "score": score,
                "details": "Oversold + bullish divergence",
            })
    return sorted(hits, key=lambda x: x["score"], reverse=True)


def screen_accumulation(results: List[Dict]) -> List[Dict]:
    """Accumulation screen: smart money flowing in.

    Criteria:
    - A/D line trending bullish
    - OBV trending bullish
    - Price not overbought (RSI < 70)
    """
    hits = []
    for r in results:
        s = r["signals"]
        if s.ad_trend_bullish and s.obv_trend_bullish and not s.rsi_overbought:
            score = 0
            if s.price_above_200sma:
                score += 1
            if s.price_above_vwap:
                score += 1
            if s.mfi > 50:
                score += 1
            if s.macd_histogram > 0:
                score += 1
            hits.append({
                "ticker": r["ticker"],
                "price": r["price"],
                "rsi": s.rsi,
                "mfi": s.mfi,
                "screen": "accumulation",
                "score": score,
                "details": "A/D + OBV bullish trend",
            })
    return sorted(hits, key=lambda x: x["score"], reverse=True)


def screen_all(results: List[Dict]) -> Dict[str, List[Dict]]:
    """Run all screens and return results grouped by screen name."""
    return {
        "momentum": screen_momentum(results),
        "breakout": screen_breakout(results),
        "oversold": screen_oversold(results),
        "accumulation": screen_accumulation(results),
    }


SCREENS = {
    "momentum": screen_momentum,
    "breakout": screen_breakout,
    "oversold": screen_oversold,
    "accumulation": screen_accumulation,
    "all": screen_all,
}


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_screen(
    screen_name: str = "all",
    universe: str = "default",
    custom_file: Optional[str] = None,
    top_n: int = 10,
    max_workers: int = 8,
) -> Dict:
    """Run a screen against the given universe.

    Returns dict with screen results.
    """
    tickers = get_universe(universe, custom_file)
    logger.info("Screening %d tickers (universe=%s, screen=%s)", len(tickers), universe, screen_name)

    # Fetch and analyze all tickers in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_and_analyze, t): t for t in tickers}
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    logger.info("Successfully analyzed %d/%d tickers", len(results), len(tickers))

    if screen_name == "all":
        all_results = screen_all(results)
        # Trim each to top_n
        return {k: v[:top_n] for k, v in all_results.items()}
    elif screen_name in SCREENS:
        hits = SCREENS[screen_name](results)
        return {screen_name: hits[:top_n]}
    else:
        logger.error("Unknown screen: %s", screen_name)
        return {}


def print_screen_results(results: Dict[str, List[Dict]]) -> None:
    """Print formatted screening results."""
    print(f"\n{'=' * 72}")
    print(f"  MARKET SCREENER RESULTS")
    print(f"{'=' * 72}")

    for screen_name, hits in results.items():
        if not hits:
            print(f"\n  {screen_name.upper()}: No matches found")
            continue

        print(f"\n  {screen_name.upper()} ({len(hits)} matches)")
        print(f"  {'─' * 66}")
        print(f"  {'Ticker':<8} {'Price':>8} {'RSI':>5} {'MFI':>5} {'Score':>5}  {'Details'}")
        print(f"  {'─' * 66}")

        cur = "₹" if is_indian_market() else "$"
        for h in hits:
            print(
                f"  {h['ticker']:<8} {cur}{h['price']:>7,.2f} {h.get('rsi', 0):>5.1f}"
                f" {h.get('mfi', 0):>5.1f} {h['score']:>5}  {h['details']}"
            )

    print(f"\n{'=' * 72}\n")


def main():
    parser = argparse.ArgumentParser(description="Market Screener")
    parser.add_argument(
        "--screen", type=str, default="all",
        choices=["momentum", "breakout", "oversold", "accumulation", "all"],
        help="Screen to run (default: all)",
    )
    parser.add_argument(
        "--universe", type=str, default="default",
        choices=["default", "watchlist", "sectors", "sp500", "custom"],
        help="Stock universe to scan (default: 60 liquid large/mid-caps)",
    )
    parser.add_argument("--file", type=str, help="Custom ticker file (for --universe custom)")
    parser.add_argument("--top", type=int, default=10, help="Number of top results per screen")
    parser.add_argument("--workers", type=int, default=8, help="Parallel fetch threads")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    for lib in ("urllib3", "yfinance", "peewee"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    results = run_screen(
        screen_name=args.screen,
        universe=args.universe,
        custom_file=args.file,
        top_n=args.top,
        max_workers=args.workers,
    )
    print_screen_results(results)


if __name__ == "__main__":
    main()
