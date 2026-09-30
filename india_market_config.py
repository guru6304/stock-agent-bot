"""Centralized Indian Market Configuration.

Defines all exchange parameters, trading sessions, timezone, currency,
and scanner parameters for the Indian Stock Market (NSE / BSE).
Strictly signal-only and paper-observation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional
import pytz
from dotenv import load_dotenv

load_dotenv()

# --- SAFETY BARRIER: PROHIBIT BROKER ORDERS ---
ORDER_EXECUTION_ENABLED: bool = False

def assert_broker_orders_prohibited() -> None:
    """Hard safety invariant: raises RuntimeError if order execution is ever attempted."""
    if ORDER_EXECUTION_ENABLED:
        raise RuntimeError("CRITICAL SAFETY VIOLATION: Order execution is strictly forbidden.")

# --- TIMEZONE & CURRENCY ---
TIMEZONE_NAME: str = "Asia/Kolkata"
IST: pytz.BaseTzInfo = pytz.timezone(TIMEZONE_NAME)
CURRENCY_SYMBOL: str = "₹"
CURRENCY_CODE: str = "INR"

# --- EXCHANGES & SEGMENTS ---
EXCHANGE_NSE: str = "NSE"
EXCHANGE_BSE: str = "BSE"
SEGMENT_EQUITY: str = "NSE_EQ"
SEGMENT_NFO: str = "NFO"

ALLOWED_EXCHANGES: set[str] = {EXCHANGE_NSE, EXCHANGE_BSE}
ALLOWED_SEGMENTS: set[str] = {"NSE", "NFO", "BSE"}

# --- BENCHMARK & REGIME IDENTIFIERS ---
BENCHMARK_NIFTY_50: str = "^NSEI"
BENCHMARK_INDIA_VIX: str = "^INDIAVIX"
BENCHMARK_BANK_NIFTY: str = "^NSEBANK"

SECTOR_INDICES: dict[str, str] = {
    "NIFTY_IT": "^CNXIT",
    "NIFTY_BANK": "^NSEBANK",
    "NIFTY_AUTO": "^CNXAUTO",
    "NIFTY_PHARMA": "^CNXPHARMA",
    "NIFTY_FMCG": "^CNXFMCG",
    "NIFTY_METAL": "^CNXMETAL",
    "NIFTY_REALTY": "^CNXREALTY",
    "NIFTY_ENERGY": "^CNXENERGY",
}

# --- MARKET SESSIONS (IST) ---
PRE_MARKET_OPEN: time = time(9, 0)
PRE_MARKET_CLOSE: time = time(9, 8)
REGULAR_MARKET_OPEN: time = time(9, 15)
REGULAR_MARKET_CLOSE: time = time(15, 30)
POST_MARKET_OPEN: time = time(15, 40)
POST_MARKET_CLOSE: time = time(16, 0)

# --- OFFICIAL NSE TRADING HOLIDAYS CALENDAR ---
NSE_TRADING_HOLIDAYS: dict[str, str] = {
    # 2024
    "2024-01-22": "Special Holiday (Ayodhya Ram Mandir)",
    "2024-01-26": "Republic Day",
    "2024-03-08": "Mahashivratri",
    "2024-03-25": "Holi",
    "2024-03-29": "Good Friday",
    "2024-04-11": "Id-Ul-Fitr (Ramzan Id)",
    "2024-04-17": "Shri Ram Navami",
    "2024-05-01": "Maharashtra Day",
    "2024-05-20": "General Parliamentary Elections",
    "2024-06-17": "Bakri Id",
    "2024-07-17": "Muharram",
    "2024-08-15": "Independence Day",
    "2024-10-02": "Mahatma Gandhi Jayanti",
    "2024-11-01": "Diwali Laxmi Pujan (Regular session closed)",
    "2024-11-15": "Gurunanak Jayanti",
    "2024-11-20": "Maharashtra Assembly Elections",
    "2024-12-25": "Christmas",
    # 2025
    "2025-02-26": "Mahashivratri",
    "2025-03-14": "Holi",
    "2025-03-31": "Id-Ul-Fitr (Ramzan Id)",
    "2025-04-10": "Mahavir Jayanti",
    "2025-04-14": "Dr. Baba Saheb Ambedkar Jayanti",
    "2025-04-18": "Good Friday",
    "2025-05-01": "Maharashtra Day",
    "2025-06-07": "Bakri Id",
    "2025-08-15": "Independence Day",
    "2025-08-27": "Ganesh Chaturthi",
    "2025-10-02": "Mahatma Gandhi Jayanti / Dussehra",
    "2025-10-21": "Diwali Laxmi Pujan (Regular session closed)",
    "2025-10-22": "Diwali Balipratipada",
    "2025-11-05": "Gurunanak Jayanti",
    "2025-12-25": "Christmas",
    # 2026
    "2026-01-26": "Republic Day",
    "2026-02-16": "Mahashivratri",
    "2026-03-04": "Holi",
    "2026-03-20": "Id-Ul-Fitr (Ramzan Id)",
    "2026-03-27": "Shri Ram Navami",
    "2026-04-03": "Good Friday",
    "2026-04-14": "Dr. Baba Saheb Ambedkar Jayanti",
    "2026-05-01": "Maharashtra Day",
    "2026-05-27": "Bakri Id",
    "2026-06-25": "Muharram",
    "2026-08-15": "Independence Day",
    "2026-09-14": "Milad-un-Nabi",
    "2026-10-02": "Mahatma Gandhi Jayanti",
    "2026-10-20": "Dussehra",
    "2026-11-09": "Diwali Balipratipada",
    "2026-11-24": "Gurunanak Jayanti",
    "2026-12-25": "Christmas",
    # 2027
    "2027-01-26": "Republic Day",
    "2027-03-08": "Mahashivratri",
    "2027-03-23": "Holi",
    "2027-03-26": "Good Friday",
    "2027-04-14": "Dr. Baba Saheb Ambedkar Jayanti",
    "2027-05-01": "Maharashtra Day",
    "2027-08-15": "Independence Day",
    "2027-10-02": "Mahatma Gandhi Jayanti",
    "2027-12-25": "Christmas",
}

def now_ist() -> datetime:
    """Return the current time in Asia/Kolkata."""
    return datetime.now(IST)

def get_holiday_name(dt: Optional[datetime] = None) -> Optional[str]:
    """Return holiday name if the date is an official NSE trading holiday, else None."""
    current = dt.astimezone(IST) if dt else now_ist()
    date_str = current.strftime("%Y-%m-%d")
    return NSE_TRADING_HOLIDAYS.get(date_str)

def is_trading_day(dt: Optional[datetime] = None) -> bool:
    """Return True if the date is an active NSE trading day (Mon-Fri and not an NSE holiday)."""
    current = dt.astimezone(IST) if dt else now_ist()
    if current.weekday() >= 5:  # 5=Sat, 6=Sun
        return False
    return get_holiday_name(current) is None

def is_market_hours(dt: Optional[datetime] = None, allow_extended: bool = False) -> bool:
    """Return True if current or provided datetime is within normal NSE trading hours (Mon-Fri 09:15-15:30 IST).
    
    If allow_extended=True or SCAN_PRE_POST_MARKET=1, allows pre/post market sessions (09:00 - 16:00 IST).
    Supports custom hours via MARKET_OPEN_HOUR/MINUTE and MARKET_CLOSE_HOUR/MINUTE env vars.
    """
    current = dt.astimezone(IST) if dt else now_ist()
    if not is_trading_day(current):
        return False
    
    current_time = current.time()
    
    scan_pre_post = allow_extended or os.getenv("SCAN_PRE_POST_MARKET", "0") in ("1", "true", "True")
    open_time = PRE_MARKET_OPEN if scan_pre_post else REGULAR_MARKET_OPEN
    close_time = POST_MARKET_CLOSE if scan_pre_post else REGULAR_MARKET_CLOSE
    
    # Env var overrides
    if os.getenv("MARKET_OPEN_HOUR") and os.getenv("MARKET_OPEN_MINUTE"):
        try:
            open_time = time(int(os.getenv("MARKET_OPEN_HOUR")), int(os.getenv("MARKET_OPEN_MINUTE")))
        except Exception:
            pass
    if os.getenv("MARKET_CLOSE_HOUR") and os.getenv("MARKET_CLOSE_MINUTE"):
        try:
            close_time = time(int(os.getenv("MARKET_CLOSE_HOUR")), int(os.getenv("MARKET_CLOSE_MINUTE")))
        except Exception:
            pass

    return open_time <= current_time <= close_time

def get_market_status(dt: Optional[datetime] = None) -> tuple[bool, str]:
    """Return (is_open: bool, description: str) with human-readable status."""
    current = dt.astimezone(IST) if dt else now_ist()
    day_name = current.strftime("%A")
    time_str = current.strftime("%H:%M:%S IST")
    
    if current.weekday() >= 5:
        return False, f"CLOSED (Weekend — {day_name})"
    
    holiday = get_holiday_name(current)
    if holiday:
        return False, f"CLOSED (NSE Holiday: {holiday})"
        
    current_time = current.time()
    if current_time < PRE_MARKET_OPEN:
        return False, f"CLOSED (Pre-market opens at 09:00 IST, Regular at 09:15 IST — Current: {time_str})"
    elif current_time < REGULAR_MARKET_OPEN:
        return False, f"PRE-MARKET (Order matching 09:00 - 09:08, Regular starts at 09:15 IST)"
    elif current_time <= REGULAR_MARKET_CLOSE:
        return True, f"OPEN (Regular trading session 09:15 - 15:30 IST — Current: {time_str})"
    elif current_time <= POST_MARKET_CLOSE:
        return False, f"POST-MARKET (Closing session 15:40 - 16:00 IST)"
    else:
        return False, f"CLOSED (Market closed for today at 15:30 IST — Current: {time_str})"

# --- TUNABLE SCANNER CONFIGURATION ---
@dataclass(frozen=True)
class ScannerConfig:
    """Centralized, validated scanner parameters."""
    scan_interval_minutes: int = int(os.getenv("RUN_INTERVAL_MINUTES", "15"))
    max_scan_universe: int = int(os.getenv("MAX_SCAN_UNIVERSE", "100"))
    batch_size: int = int(os.getenv("SCAN_BATCH_SIZE", "20"))
    request_delay_seconds: float = float(os.getenv("SCAN_REQUEST_DELAY", "0.2"))
    signal_dedup_hours: float = float(os.getenv("SIGNAL_DEDUP_HOURS", "4.0"))
    master_cache_ttl_hours: float = float(os.getenv("MASTER_CACHE_TTL_HOURS", "24.0"))
    max_signals_per_cycle: int = int(os.getenv("MAX_SIGNALS_PER_CYCLE", "10"))
    scrip_master_url: str = os.getenv(
        "SCRIP_MASTER_URL",
        "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json",
    )
    # Signal threshold (confluence score)
    signal_threshold: int = int(os.getenv("SIGNAL_THRESHOLD", "4"))
    # Minimum 20d volume ratio for breakout confirmation
    volume_surge_ratio: float = float(os.getenv("VOLUME_SURGE_RATIO", "1.4"))
    # Default risk-reward minimum ratio
    min_risk_reward: float = float(os.getenv("MIN_RISK_REWARD", "1.5"))

DEFAULT_SCANNER_CONFIG: ScannerConfig = ScannerConfig()
