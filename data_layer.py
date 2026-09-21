"""
Data Layer — fetches price/OHLCV, fundamental, and news data for watchlist tickers.
"""

import os
import csv
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

def is_indian_market() -> bool:
    """Return True if system is configured for Indian markets."""
    return (
        os.getenv("MARKET", "").upper() == "INDIA"
        or os.getenv("TIMEZONE", "") == "Asia/Kolkata"
    )


def resolve_ticker(ticker: str) -> str:
    """Normalize ticker symbol for yfinance, adding .NS if Indian market."""
    clean = str(ticker).strip().upper()
    if clean.endswith(".NS") or clean.endswith(".BO") or clean.startswith("^"):
        return clean
    if is_indian_market():
        return f"{clean}.NS"
    return clean


# ---------------------------------------------------------------------------
# Watchlist & Portfolio helpers
# ---------------------------------------------------------------------------

def load_watchlist(path: Optional[str] = None) -> List[Dict]:
    """Return list of dicts with keys: ticker, sector, notes.
    
    Dynamically discovers liquid Indian equities from the market master by default.
    Only falls back to a file if an explicit custom non-default path is passed.
    """
    # If custom non-default path explicitly provided, use it
    if path and path != "watchlist.csv" and os.path.exists(path):
        try:
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
                if rows:
                    return rows
        except Exception as e:
            logger.warning("Failed to read custom watchlist %s: %s", path, e)

    # Dynamic discovery for Indian market (primary source)
    if is_indian_market():
        try:
            from instrument_discovery import DynamicInstrumentDiscovery
            disc = DynamicInstrumentDiscovery()
            equities = disc.get_equity_universe(limit=60)
            if equities:
                items = [{"ticker": inst.symbol, "sector": "Indian Equities", "notes": inst.name} for inst in equities]
                # Sync discovered items to watchlist.csv for backward compatibility
                try:
                    target_file = path or "watchlist.csv"
                    with open(target_file, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=["ticker", "sector", "notes"])
                        writer.writeheader()
                        writer.writerows(items)
                except Exception:
                    pass
                return items
        except Exception as e:
            logger.error("Dynamic discovery in load_watchlist failed: %s", e)

    if os.getenv("USE_SQLITE", "0") == "1":
        try:
            import database as db
            wl = db.get_watchlist()
            if wl:
                return wl
        except Exception:
            pass

    # Fallback to file if path exists
    target = path or "watchlist.csv"
    try:
        if os.path.exists(target):
            with open(target, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
                if rows:
                    return rows
    except Exception as e:
        logger.warning("Failed to read %s: %s", target, e)

    return []


def sync_portfolio_prices(portfolio: dict) -> dict:
    """Dynamically refresh current market prices and P&L for all open holdings."""
    holdings = portfolio.get("holdings", [])
    if not holdings:
        return portfolio

    total_holdings_val = 0.0
    for h in holdings:
        ticker = h.get("ticker")
        shares = float(h.get("shares", 0))
        avg_cost = float(h.get("avg_cost", 0))
        if ticker:
            try:
                resolved = resolve_ticker(ticker)
                tk = yf.Ticker(resolved)
                price = getattr(tk.fast_info, "last_price", None) or getattr(tk.fast_info, "previous_close", None)
                if price and price > 0:
                    h["current_price"] = round(float(price), 2)
                    h["current_value"] = round(shares * float(price), 2)
                    h["unrealized_pnl"] = round((float(price) - avg_cost) * shares, 2)
                    h["pnl_pct"] = round((float(price) - avg_cost) / avg_cost * 100, 2) if avg_cost else 0
            except Exception:
                pass
        total_holdings_val += float(h.get("current_value", 0) or 0)

    cash = float(portfolio.get("available_cash", 0) or 0)
    portfolio["total_portfolio_value"] = round(total_holdings_val + cash, 2)
    return portfolio


def load_portfolio(path: str = "portfolio.json", auto_sync: bool = False) -> dict:
    """Return dynamic portfolio dict.
    
    Dynamically maintains cash, open positions, and valuations without hardcoded mock stocks.
    """
    try:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "total_portfolio_value" in data:
                if is_indian_market() and data.get("currency") != "INR":
                    data["currency"] = "INR"
                    data["currency_symbol"] = "₹"
                if auto_sync and data.get("holdings"):
                    data = sync_portfolio_prices(data)
                return data
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # Fallback to SQLite if configured
    if os.getenv("USE_SQLITE", "0") == "1":
        try:
            import database as db
            config = db.get_portfolio_config()
            holdings = db.get_holdings()
            return {
                "total_portfolio_value": config.get("total_portfolio_value", 1000000.0),
                "available_cash": config.get("available_cash", 1000000.0),
                "currency": "INR" if is_indian_market() else "USD",
                "currency_symbol": "₹" if is_indian_market() else "$",
                "max_risk_per_trade_pct": config.get("max_risk_per_trade_pct", 1.0),
                "max_position_size_pct": config.get("max_position_size_pct", 10.0),
                "holdings": holdings,
            }
        except Exception:
            pass

    # Default clean dynamic portfolio in INR
    initial_cash = float(os.getenv("INITIAL_CAPITAL", "1000000.0"))
    portfolio = {
        "total_portfolio_value": initial_cash,
        "available_cash": initial_cash,
        "currency": "INR" if is_indian_market() else "USD",
        "currency_symbol": "₹" if is_indian_market() else "$",
        "max_risk_per_trade_pct": float(os.getenv("MAX_RISK_PER_TRADE_PCT", "1.0")),
        "max_position_size_pct": float(os.getenv("MAX_POSITION_SIZE_PCT", "10.0")),
        "holdings": [],
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(portfolio, f, indent=2)
    except Exception:
        pass
    return portfolio


# ---------------------------------------------------------------------------
# Price / OHLCV data  (primary: yfinance, fallback: Alpha Vantage)
# ---------------------------------------------------------------------------

def fetch_daily_ohlcv(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch daily OHLCV candles via yfinance with dynamic Indian market symbol resolution.

    Returns DataFrame with columns: Open, High, Low, Close, Volume
    indexed by Date.
    """
    resolved = resolve_ticker(ticker)
    try:
        tk = yf.Ticker(resolved)
        df = tk.history(period=period, interval="1d")
        if df.empty and resolved != ticker:
            # Try raw ticker as fallback
            tk = yf.Ticker(ticker)
            df = tk.history(period=period, interval="1d")
        elif df.empty and not resolved.endswith(".NS") and not resolved.startswith("^"):
            # Try with .NS as fallback
            tk = yf.Ticker(f"{ticker}.NS")
            df = tk.history(period=period, interval="1d")

        if df.empty and is_indian_market() and not ticker.startswith("^"):
            try:
                clean_sym = ticker.replace(".NS", "").replace(".BO", "")
                tk = yf.Ticker(f"{clean_sym}.BO")
                df = tk.history(period=period, interval="1d")
            except Exception:
                pass

        if df.empty:
            logger.warning("yfinance returned empty data for %s (%s), trying Alpha Vantage", ticker, resolved)
            return _fetch_daily_alpha_vantage(ticker)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        return df[cols].dropna()
    except Exception as e:
        logger.error("yfinance error for %s: %s", ticker, e)
        return _fetch_daily_alpha_vantage(ticker)


def _fetch_daily_alpha_vantage(ticker: str) -> pd.DataFrame:
    """Fallback: fetch daily OHLCV from Alpha Vantage."""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        logger.error("Alpha Vantage API key not configured")
        return pd.DataFrame()

    url = (
        "https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY&symbol={ticker}"
        f"&outputsize=full&apikey={api_key}"
    )
    resp = requests.get(url, timeout=30)
    data = resp.json()
    ts = data.get("Time Series (Daily)", {})
    if not ts:
        logger.error("Alpha Vantage returned no data for %s", ticker)
        return pd.DataFrame()

    rows = []
    for date_str, vals in ts.items():
        rows.append({
            "Date": pd.Timestamp(date_str),
            "Open": float(vals["1. open"]),
            "High": float(vals["2. high"]),
            "Low": float(vals["3. low"]),
            "Close": float(vals["4. close"]),
            "Volume": int(vals["5. volume"]),
        })
    df = pd.DataFrame(rows).set_index("Date").sort_index()
    return df


def fetch_intraday_ohlcv(ticker: str, period: str = "5d", interval: str = "1h") -> pd.DataFrame:
    """Fetch intraday (1h) candles for confirmation signals."""
    resolved = resolve_ticker(ticker)
    try:
        tk = yf.Ticker(resolved)
        df = tk.history(period=period, interval=interval)
        if df.empty and resolved != ticker:
            tk = yf.Ticker(ticker)
            df = tk.history(period=period, interval=interval)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
            return df[cols].dropna()
        return pd.DataFrame()
    except Exception as e:
        logger.error("Intraday fetch error for %s: %s", ticker, e)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Fundamental data  (Financial Modeling Prep)
# ---------------------------------------------------------------------------

def fetch_fundamentals_fmp(ticker: str) -> dict:
    """Fetch key fundamental metrics from Financial Modeling Prep.

    Returns dict with keys: pe_ratio, eps, eps_growth_yoy,
    revenue_growth_qoq, debt_to_equity, analyst_consensus.
    """
    api_key = os.getenv("FMP_API_KEY", "")
    base = "https://financialmodelingprep.com/api/v3"
    result = {
        "pe_ratio": None,
        "eps": None,
        "eps_growth_yoy": None,
        "revenue_growth_qoq": None,
        "debt_to_equity": None,
        "analyst_consensus": None,
    }

    if not api_key or api_key.startswith("your_"):
        logger.warning("FMP API key not configured — falling back to yfinance fundamentals")
        return _fetch_fundamentals_yfinance(ticker)

    try:
        # Key metrics
        url = f"{base}/key-metrics-ttm/{ticker}?apikey={api_key}"
        resp = requests.get(url, timeout=15)
        metrics = resp.json()
        if metrics and isinstance(metrics, list):
            m = metrics[0]
            result["pe_ratio"] = m.get("peRatioTTM")
            result["debt_to_equity"] = m.get("debtToEquityTTM")

        # Income statement for EPS / revenue growth
        url = f"{base}/income-statement/{ticker}?period=quarter&limit=5&apikey={api_key}"
        resp = requests.get(url, timeout=15)
        stmts = resp.json()
        if stmts and len(stmts) >= 2:
            result["eps"] = stmts[0].get("eps")
            eps_now = stmts[0].get("eps", 0) or 0
            eps_prev = stmts[4].get("eps", 0) if len(stmts) >= 5 else None
            if eps_prev and eps_prev != 0:
                result["eps_growth_yoy"] = (eps_now - eps_prev) / abs(eps_prev) * 100

            rev_now = stmts[0].get("revenue", 0) or 0
            rev_prev = stmts[1].get("revenue", 0) or 0
            if rev_prev and rev_prev != 0:
                result["revenue_growth_qoq"] = (rev_now - rev_prev) / abs(rev_prev) * 100

        # Analyst consensus
        url = f"{base}/analyst-estimates/{ticker}?limit=1&apikey={api_key}"
        resp = requests.get(url, timeout=15)
        est = resp.json()
        if est and isinstance(est, list) and est:
            result["analyst_consensus"] = est[0].get("estimatedEpsAvg")

    except Exception as e:
        logger.error("FMP fundamentals error for %s: %s", ticker, e)
        return _fetch_fundamentals_yfinance(ticker)

    return result


def _fetch_fundamentals_yfinance(ticker: str) -> dict:
    """Fallback: pull fundamentals from yfinance."""
    result = {
        "pe_ratio": None,
        "eps": None,
        "eps_growth_yoy": None,
        "revenue_growth_qoq": None,
        "debt_to_equity": None,
        "analyst_consensus": None,
    }
    try:
        resolved = resolve_ticker(ticker)
        tk = yf.Ticker(resolved)
        info = tk.info or {}
        if not info and resolved != ticker:
            tk = yf.Ticker(ticker)
            info = tk.info or {}
        if not info and is_indian_market() and not ticker.startswith("^"):
            try:
                clean_sym = ticker.replace(".NS", "").replace(".BO", "")
                tk = yf.Ticker(f"{clean_sym}.BO")
                info = tk.info or {}
            except Exception:
                pass
        result["pe_ratio"] = info.get("trailingPE") or info.get("forwardPE")
        result["eps"] = info.get("trailingEps")
        result["debt_to_equity"] = info.get("debtToEquity")
        if result["debt_to_equity"] is not None:
            # yfinance reports D/E as percentage (e.g. 102.63 = 1.0263)
            result["debt_to_equity"] = result["debt_to_equity"] / 100

        # EPS growth — try earnings_history, fall back to earnings_dates
        try:
            earnings = getattr(tk, "earnings_history", None)
            if earnings is not None and hasattr(earnings, "empty") and not earnings.empty and len(earnings) >= 5:
                eps_now = earnings.iloc[-1].get("epsActual", 0) or 0
                eps_prev = earnings.iloc[-5].get("epsActual", 0) or 0
                if eps_prev != 0:
                    result["eps_growth_yoy"] = (eps_now - eps_prev) / abs(eps_prev) * 100
        except Exception:
            pass

        # If EPS growth still not set, try computing from trailing vs forward EPS
        if result["eps_growth_yoy"] is None:
            trailing = info.get("trailingEps")
            forward = info.get("forwardEps")
            if trailing and forward and trailing != 0:
                result["eps_growth_yoy"] = (forward - trailing) / abs(trailing) * 100

        # Revenue growth from quarterly financials
        try:
            qf = tk.quarterly_income_stmt
            if qf is not None and not qf.empty:
                # yfinance may use "Total Revenue" or "TotalRevenue"
                rev_row = None
                for label in ("Total Revenue", "TotalRevenue", "Revenue"):
                    if label in qf.index:
                        rev_row = qf.loc[label]
                        break
                if rev_row is not None and len(rev_row) >= 2:
                    rev_now = float(rev_row.iloc[0]) if pd.notna(rev_row.iloc[0]) else 0
                    rev_prev = float(rev_row.iloc[1]) if pd.notna(rev_row.iloc[1]) else 0
                    if rev_prev and rev_prev != 0:
                        result["revenue_growth_qoq"] = (rev_now - rev_prev) / abs(rev_prev) * 100
        except Exception:
            pass

        # Analyst recommendation
        rec = info.get("recommendationKey", "") or ""
        result["analyst_consensus"] = rec  # e.g. "buy", "strong_buy", "hold"

    except Exception as e:
        logger.error("yfinance fundamentals error for %s: %s", ticker, e)

    return result


def fetch_fundamentals(ticker: str) -> dict:
    """Unified fundamental data fetcher — tries FMP first, falls back to yfinance."""
    return fetch_fundamentals_fmp(ticker)


# ---------------------------------------------------------------------------
# News sentiment  (NewsAPI primary, Finnhub fallback)
# ---------------------------------------------------------------------------

def fetch_news_newsapi(ticker: str, days: int = 7) -> List[Dict]:
    """Fetch recent news headlines from NewsAPI.

    Returns list of dicts with keys: title, description, publishedAt, source.
    """
    api_key = os.getenv("NEWSAPI_KEY", "")
    if not api_key or api_key.startswith("your_"):
        logger.warning("NewsAPI key not configured — trying Finnhub")
        return fetch_news_finnhub(ticker, days)

    from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    url = (
        "https://newsapi.org/v2/everything"
        f"?q={ticker}&from={from_date}&sortBy=publishedAt"
        f"&language=en&pageSize=20&apiKey={api_key}"
    )
    try:
        resp = requests.get(url, timeout=15)
        data = resp.json()
        articles = data.get("articles", [])
        return [
            {
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "publishedAt": a.get("publishedAt", ""),
                "source": a.get("source", {}).get("name", ""),
            }
            for a in articles
        ]
    except Exception as e:
        logger.error("NewsAPI error for %s: %s", ticker, e)
        return fetch_news_finnhub(ticker, days)


def fetch_news_finnhub(ticker: str, days: int = 7) -> List[Dict]:
    """Fallback: fetch news from Finnhub."""
    api_key = os.getenv("FINNHUB_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        logger.warning("Finnhub API key not configured")
        return []

    from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    to_date = datetime.utcnow().strftime("%Y-%m-%d")
    url = (
        f"https://finnhub.io/api/v1/company-news"
        f"?symbol={ticker}&from={from_date}&to={to_date}"
        f"&token={api_key}"
    )
    try:
        resp = requests.get(url, timeout=15)
        articles = resp.json()
        if not isinstance(articles, list):
            return []
        return [
            {
                "title": a.get("headline", ""),
                "description": a.get("summary", ""),
                "publishedAt": datetime.fromtimestamp(a.get("datetime", 0)).isoformat(),
                "source": a.get("source", ""),
            }
            for a in articles[:20]
        ]
    except Exception as e:
        logger.error("Finnhub error for %s: %s", ticker, e)
        return []


def fetch_news(ticker: str, days: int = 7) -> List[Dict]:
    """Unified news fetcher."""
    return fetch_news_newsapi(ticker, days)
