"""
Earnings Calendar Guard — checks upcoming earnings dates and blocks or
warns before entering positions too close to earnings announcements.

Earnings events create gap risk that can blow through stop-losses,
so we avoid new entries within a configurable window (default 3 days).

Usage:
    from earnings_guard import check_earnings_safe

    safe, info = check_earnings_safe("AAPL")
    if not safe:
        print(f"Blocked: {info['reason']}")
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import yfinance as yf
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Number of trading days before earnings to block new entries
EARNINGS_BLACKOUT_DAYS = int(os.getenv("EARNINGS_BLACKOUT_DAYS", "3"))


def get_next_earnings(ticker: str) -> Optional[datetime]:
    """Fetch the next earnings date for a ticker via yfinance.

    Returns datetime or None if unavailable.
    """
    try:
        from data_layer import resolve_ticker
        tk = yf.Ticker(resolve_ticker(ticker))

        # Method 1: calendar
        try:
            cal = tk.calendar
            if cal is not None:
                if isinstance(cal, dict):
                    earnings_date = cal.get("Earnings Date")
                    if earnings_date:
                        if isinstance(earnings_date, list) and earnings_date:
                            return _parse_date(earnings_date[0])
                        return _parse_date(earnings_date)
                elif hasattr(cal, "columns"):
                    # DataFrame format
                    if "Earnings Date" in cal.columns:
                        val = cal["Earnings Date"].iloc[0]
                        return _parse_date(val)
                    elif len(cal.columns) > 0:
                        val = cal.iloc[0, 0]
                        return _parse_date(val)
        except Exception:
            pass

        # Method 2: earnings_dates attribute
        try:
            ed = tk.earnings_dates
            if ed is not None and not ed.empty:
                future = ed[ed.index >= datetime.now()]
                if not future.empty:
                    return future.index[0].to_pydatetime().replace(tzinfo=None)
        except Exception:
            pass

        # Method 3: info dict
        try:
            info = tk.info or {}
            for key in ("earningsTimestamp", "earningsTimestampStart"):
                ts = info.get(key)
                if ts:
                    dt = datetime.fromtimestamp(ts)
                    if dt > datetime.now():
                        return dt
        except Exception:
            pass

    except Exception as e:
        logger.debug("Earnings lookup failed for %s: %s", ticker, e)

    return None


def _parse_date(val) -> Optional[datetime]:
    """Parse various date formats into datetime."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.replace(tzinfo=None)
    if hasattr(val, "to_pydatetime"):
        return val.to_pydatetime().replace(tzinfo=None)
    if isinstance(val, str):
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%b %d, %Y"):
            try:
                return datetime.strptime(val, fmt)
            except ValueError:
                continue
    return None


def check_earnings_safe(
    ticker: str,
    blackout_days: Optional[int] = None,
) -> Tuple[bool, Dict]:
    """Check if it's safe to enter a position (not too close to earnings).

    Returns (is_safe, info_dict).
    info_dict contains:
      - earnings_date: next earnings date (or None)
      - days_until: trading days until earnings (or None)
      - reason: explanation if blocked
      - blackout_days: configured blackout window
    """
    if blackout_days is None:
        blackout_days = EARNINGS_BLACKOUT_DAYS

    info = {
        "earnings_date": None,
        "days_until": None,
        "reason": "",
        "blackout_days": blackout_days,
    }

    earnings_date = get_next_earnings(ticker)
    if earnings_date is None:
        info["reason"] = "No earnings date found — allowing entry"
        return True, info

    info["earnings_date"] = earnings_date.isoformat()

    # Compute calendar days and approximate trading days
    delta = earnings_date - datetime.now()
    calendar_days = delta.days
    trading_days = int(calendar_days * 5 / 7)  # rough estimate
    info["days_until"] = trading_days

    if trading_days <= blackout_days:
        info["reason"] = (
            f"Earnings in ~{trading_days} trading days ({earnings_date.strftime('%Y-%m-%d')}) "
            f"— within {blackout_days}-day blackout window"
        )
        logger.warning(
            "%s: EARNINGS GUARD BLOCKED — %s",
            ticker, info["reason"],
        )
        return False, info

    info["reason"] = (
        f"Earnings in ~{trading_days} trading days ({earnings_date.strftime('%Y-%m-%d')}) "
        f"— outside blackout window"
    )
    return True, info


def check_watchlist_earnings() -> Dict[str, Dict]:
    """Check earnings proximity for all watchlist tickers.

    Returns dict of ticker -> earnings info.
    """
    try:
        import data_layer
        watchlist = data_layer.load_watchlist()
    except Exception:
        return {}

    results = {}
    for entry in watchlist:
        ticker = entry["ticker"]
        safe, info = check_earnings_safe(ticker)
        results[ticker] = {
            "safe": safe,
            **info,
        }

    return results


def print_earnings_report(results: Optional[Dict[str, Dict]] = None) -> None:
    """Print earnings calendar for watchlist."""
    if results is None:
        results = check_watchlist_earnings()

    print(f"\n{'=' * 65}")
    print(f"  EARNINGS CALENDAR GUARD")
    print(f"{'=' * 65}")

    blocked = []
    upcoming = []
    clear = []

    for ticker, info in sorted(results.items()):
        if not info["safe"]:
            blocked.append((ticker, info))
        elif info["days_until"] is not None and info["days_until"] <= 14:
            upcoming.append((ticker, info))
        else:
            clear.append((ticker, info))

    if blocked:
        print(f"\n  BLOCKED (within blackout window)")
        print(f"  {'─' * 55}")
        for ticker, info in blocked:
            date_str = info["earnings_date"][:10] if info["earnings_date"] else "?"
            print(f"  X {ticker:<6}  earnings: {date_str}  (~{info['days_until']}d)")

    if upcoming:
        print(f"\n  UPCOMING (watch closely)")
        print(f"  {'─' * 55}")
        for ticker, info in upcoming:
            date_str = info["earnings_date"][:10] if info["earnings_date"] else "?"
            print(f"  ! {ticker:<6}  earnings: {date_str}  (~{info['days_until']}d)")

    if clear:
        print(f"\n  CLEAR")
        print(f"  {'─' * 55}")
        for ticker, info in clear:
            if info["days_until"] is not None:
                date_str = info["earnings_date"][:10] if info["earnings_date"] else "?"
                print(f"  + {ticker:<6}  earnings: {date_str}  (~{info['days_until']}d)")
            else:
                print(f"  + {ticker:<6}  no earnings date found")

    print(f"\n  Blackout window: {EARNINGS_BLACKOUT_DAYS} trading days")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    for lib in ("urllib3", "yfinance"):
        logging.getLogger(lib).setLevel(logging.WARNING)
    print_earnings_report()
