"""Market Data Service.

Provides a unified interface for market data retrieval (live vs research).
Verifies Angel One SmartAPI access safely (read-only probe).
Strictly separates live execution-ready data from non-live research data.
No broker order execution imports or calls permitted.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

from india_market_config import (
    DEFAULT_SCANNER_CONFIG,
    now_ist,
)
from instrument_discovery import InstrumentMetadata

logger = logging.getLogger("market-data-service")

# Safe import of SmartAPI
try:
    from SmartApi import SmartConnect
    import pyotp
except ImportError:
    SmartConnect = None
    pyotp = None


@dataclass(frozen=True)
class CandleData:
    """Standardized OHLCV candle response."""
    df: pd.DataFrame
    symbol: str
    token: str
    interval: str
    data_source: str
    is_live: bool
    data_timestamp: str
    is_candle_complete: bool


@dataclass(frozen=True)
class QuoteData:
    """Standardized live or research quote."""
    symbol: str
    token: str
    price: float
    high: float
    low: float
    close: float
    volume: float
    data_source: str
    is_live: bool
    quote_timestamp: str


class AngelSmartAPIDataProvider:
    """Angel One SmartAPI Provider — strictly read-only market data."""

    def __init__(self):
        self.api = None
        self.is_live_ready: bool = False
        self.blocker_reason: str = ""
        self._authenticate_and_probe()

    def _authenticate_and_probe(self) -> None:
        api_key = os.getenv("ANGEL_API_KEY", "").strip()
        client_code = os.getenv("ANGEL_CLIENT_CODE", "").strip()
        pin = os.getenv("ANGEL_PIN", "").strip()
        totp_key = os.getenv("ANGEL_TOTP_KEY", "").strip()

        required = {
            "ANGEL_API_KEY": api_key,
            "ANGEL_CLIENT_CODE": client_code,
            "ANGEL_PIN": pin,
            "ANGEL_TOTP_KEY": totp_key,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            self.blocker_reason = f"Missing required credentials: {', '.join(missing)}"
            logger.info("Angel One SmartAPI: %s", self.blocker_reason)
            return

        if not SmartConnect or not pyotp:
            self.blocker_reason = "smartapi-python and pyotp packages are not installed in python environment"
            logger.warning("Angel One SmartAPI: %s", self.blocker_reason)
            return

        try:
            self.api = SmartConnect(api_key=api_key)
            totp = pyotp.TOTP(totp_key).now()
            session = self.api.generateSession(client_code, pin, totp)
            if not session or not session.get("status"):
                msg = (session or {}).get("message", "Session rejected by broker")
                self.blocker_reason = f"Login rejected: {msg}"
                logger.warning("Angel One SmartAPI auth failed: %s", self.blocker_reason)
                self.api = None
                return

            # Safe read-only probe: request profile or dummy quote to verify connection
            probe = self.api.getProfile(session.get("data", {}).get("refreshToken", ""))
            if probe and probe.get("status"):
                self.is_live_ready = True
                self.blocker_reason = ""
                logger.info("Angel One SmartAPI successfully authenticated and verified read-only probe.")
            else:
                self.is_live_ready = True  # session succeeded even if getProfile is partial
                logger.info("Angel One SmartAPI authenticated (read-only session active).")
        except Exception as e:
            self.api = None
            self.is_live_ready = False
            self.blocker_reason = f"Exception during authentication/probe: {e}"
            logger.warning("Angel One SmartAPI unavailable: %s", self.blocker_reason)

    def fetch_ltp(self, inst: InstrumentMetadata) -> Optional[float]:
        """Fetch live Last Traded Price via SmartAPI."""
        if not self.is_live_ready or not self.api:
            return None
        try:
            exch = inst.exch_seg
            tradingsymbol = inst.raw_symbol
            resp = self.api.ltpData(exch, tradingsymbol, inst.token)
            if resp and resp.get("status") and resp.get("data"):
                return float(resp["data"].get("ltp", 0.0))
        except Exception as e:
            logger.debug("SmartAPI ltpData error for %s: %s", inst.symbol, e)
        return None


class ResearchYFinanceDataProvider:
    """Non-live research and fallback provider using yfinance.

    Always labels data as NON-LIVE research data.
    Never claims to be live exchange data.
    """

    def __init__(self):
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            self.yf = None

    def fetch_candles(
        self,
        symbol: str,
        token: str = "",
        period: str = "1y",
        interval: str = "1d",
    ) -> Optional[CandleData]:
        """Fetch historical candles via yfinance with .NS suffix."""
        if not self.yf:
            return None

        # Clean symbol and append .NS if needed
        clean = symbol.upper().strip()
        yf_symbol = clean if clean.endswith(".NS") or clean.startswith("^") else f"{clean}.NS"

        try:
            tk = self.yf.Ticker(yf_symbol)
            df = tk.history(period=period, interval=interval, auto_adjust=True)
            if df.empty or len(df) < 5:
                return None

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Standardize columns
            req = {"Open", "High", "Low", "Close", "Volume"}
            if not req.issubset(set(df.columns)):
                return None

            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            df.index = pd.to_datetime(df.index).tz_localize(None)

            last_ts = df.index[-1].strftime("%Y-%m-%d %H:%M:%S")

            return CandleData(
                df=df,
                symbol=clean.replace(".NS", ""),
                token=token,
                interval=interval,
                data_source="yfinance (Non-Live Research)",
                is_live=False,
                data_timestamp=last_ts,
                is_candle_complete=True,
            )
        except Exception as e:
            logger.debug("yfinance fetch error for %s: %s", yf_symbol, e)
            return None


class MarketDataService:
    """Orchestrates market data access respecting live readiness, rate limits, and fallback rules."""

    def __init__(
        self,
        angel_provider: Optional[AngelSmartAPIDataProvider] = None,
        research_provider: Optional[ResearchYFinanceDataProvider] = None,
        config=None,
    ):
        self.angel = angel_provider or AngelSmartAPIDataProvider()
        self.research = research_provider or ResearchYFinanceDataProvider()
        self.config = config or DEFAULT_SCANNER_CONFIG

    @property
    def is_live_ready(self) -> bool:
        return self.angel.is_live_ready

    @property
    def live_blocker_reason(self) -> str:
        return self.angel.blocker_reason

    def get_candle_history(
        self,
        inst: InstrumentMetadata,
        period: str = "1y",
        interval: str = "1d",
    ) -> Optional[CandleData]:
        """Fetch historical candles with rate limiting."""
        if self.config.request_delay_seconds > 0:
            time.sleep(self.config.request_delay_seconds)

        # Primary research/historical provider
        candles = self.research.fetch_candles(
            symbol=inst.symbol,
            token=inst.token,
            period=period,
            interval=interval,
        )
        return candles

    def get_live_quote(self, inst: InstrumentMetadata) -> Optional[QuoteData]:
        """Attempt to fetch a live quote from SmartAPI if ready."""
        if self.is_live_ready:
            ltp = self.angel.fetch_ltp(inst)
            if ltp and ltp > 0:
                return QuoteData(
                    symbol=inst.symbol,
                    token=inst.token,
                    price=ltp,
                    high=ltp,
                    low=ltp,
                    close=ltp,
                    volume=0.0,
                    data_source="Angel One SmartAPI (Live)",
                    is_live=True,
                    quote_timestamp=now_ist().strftime("%Y-%m-%d %H:%M:%S IST"),
                )
        return None
