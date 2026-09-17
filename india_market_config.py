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

def now_ist() -> datetime:
    """Return the current time in Asia/Kolkata."""
    return datetime.now(IST)

def is_market_hours(dt: Optional[datetime] = None) -> bool:
    """Return True if current or provided datetime is within normal NSE trading hours (Mon-Fri 09:15-15:30 IST)."""
    current = dt.astimezone(IST) if dt else now_ist()
    # 0 = Monday, 4 = Friday, 5 = Saturday, 6 = Sunday
    if current.weekday() >= 5:
        return False
    current_time = current.time()
    return REGULAR_MARKET_OPEN <= current_time <= REGULAR_MARKET_CLOSE

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
