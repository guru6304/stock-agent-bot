"""Dynamic Instrument Discovery Engine.

Discovers eligible Indian equity and F&O instruments from authoritative
exchange / broker instrument master data (Angel One Scrip Master / NSE).
No hardcoded stock symbols, tickers, or fallback lists.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import requests

from india_market_config import (
    ALLOWED_EXCHANGES,
    DEFAULT_SCANNER_CONFIG,
    EXCHANGE_NSE,
    now_ist,
)

logger = logging.getLogger("instrument-discovery")

CACHE_FILE = Path("logs/instrument_master_cache.json")
CACHE_METADATA_FILE = Path("logs/instrument_master_meta.json")


class DiscoveryError(Exception):
    """Raised when instrument discovery fails and cannot safely proceed."""


@dataclass(frozen=True)
class InstrumentMetadata:
    """Validated metadata for an exchange-listed instrument."""
    token: str
    symbol: str              # Base clean trading symbol (e.g. clean equity symbol)
    raw_symbol: str          # Exchange scrip symbol (e.g. SYMBOL-EQ)
    name: str
    expiry: Optional[date]
    strike: float
    lotsize: int
    instrument_type: str     # "", "OPTIDX", "OPTSTK", "FUTIDX", "FUTSTK"
    exch_seg: str            # "NSE", "NFO", "BSE"
    tick_size: float
    is_fno: bool


@dataclass
class DiscoveryStats:
    """Observability metrics for the discovery pipeline."""
    retrieval_time_ist: str = ""
    source_url: str = ""
    records_received: int = 0
    records_valid: int = 0
    eligible_equity: int = 0
    eligible_fno: int = 0
    rejected_reasons: Dict[str, int] = field(default_factory=dict)

    def record_rejection(self, reason: str) -> None:
        self.rejected_reasons[reason] = self.rejected_reasons.get(reason, 0) + 1


def _parse_expiry_date(expiry_str: str) -> Optional[date]:
    """Parse expiry string from instrument master into a date object."""
    if not expiry_str or expiry_str.strip() == "":
        return None
    cleaned = expiry_str.strip().upper()
    # Common Angel Scrip Master formats: '28MAR2024', '2024-03-28', '28-Mar-2024'
    for fmt in ("%d%b%Y", "%Y-%m-%d", "%d-%b-%Y", "%d%B%Y"):
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


class DynamicInstrumentDiscovery:
    """Retrieves, validates, and filters instruments dynamically from verified metadata."""

    def __init__(
        self,
        master_url: Optional[str] = None,
        cache_ttl_hours: Optional[float] = None,
        session: Optional[requests.Session] = None,
    ):
        self.master_url = master_url or DEFAULT_SCANNER_CONFIG.scrip_master_url
        self.cache_ttl_hours = (
            cache_ttl_hours
            if cache_ttl_hours is not None
            else DEFAULT_SCANNER_CONFIG.master_cache_ttl_hours
        )
        self.http = session or requests.Session()
        self.http.headers.update({"User-Agent": "Mozilla/5.0 (Dynamic Indian Instrument Discovery)"})

        self._equity_universe: List[InstrumentMetadata] = []
        self._fno_universe: List[InstrumentMetadata] = []
        self._fno_by_underlying: Dict[str, List[InstrumentMetadata]] = {}
        self._latest_stats: Optional[DiscoveryStats] = None

    @property
    def latest_stats(self) -> Optional[DiscoveryStats]:
        return self._latest_stats

    def _is_cache_valid(self) -> bool:
        if not CACHE_FILE.exists() or not CACHE_METADATA_FILE.exists():
            return False
        try:
            meta = json.loads(CACHE_METADATA_FILE.read_text(encoding="utf-8"))
            cached_at = meta.get("timestamp", 0)
            age_hours = (time.time() - cached_at) / 3600.0
            return age_hours < self.cache_ttl_hours
        except Exception:
            return False

    def _load_from_cache(self) -> list:
        logger.info("Loading instrument master from local cache (%s)", CACHE_FILE)
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))

    def _save_to_cache(self, records: list) -> None:
        try:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            CACHE_FILE.write_text(json.dumps(records), encoding="utf-8")
            CACHE_METADATA_FILE.write_text(
                json.dumps({
                    "timestamp": time.time(),
                    "retrieved_ist": now_ist().isoformat(),
                    "records_count": len(records),
                    "source_url": self.master_url,
                }),
                encoding="utf-8",
            )
            logger.info("Saved %d instruments to cache", len(records))
        except Exception as e:
            logger.warning("Could not persist instrument cache: %s", e)

    def fetch_raw_master(self, force_refresh: bool = False) -> list:
        """Fetch raw scrip master list from remote source or valid local cache."""
        if not force_refresh and self._is_cache_valid():
            try:
                return self._load_from_cache()
            except Exception as e:
                logger.warning("Failed to read cache, falling back to download: %s", e)

        logger.info("Fetching authoritative instrument master from %s", self.master_url)
        try:
            resp = self.http.get(self.master_url, timeout=35)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list) or len(data) == 0:
                raise DiscoveryError(f"Instrument master returned empty or invalid schema from {self.master_url}")
            self._save_to_cache(data)
            return data
        except Exception as e:
            # Fallback to existing cache even if stale, if download failed
            if CACHE_FILE.exists():
                logger.warning("Remote fetch failed (%s); using stale cache as temporary fallback", e)
                return self._load_from_cache()
            raise DiscoveryError(
                f"Failed to fetch instrument master and no local cache available: {e}"
            ) from e

    def parse_record(self, raw: dict, stats: DiscoveryStats, today: date) -> Optional[InstrumentMetadata]:
        """Validate and parse a single raw record into InstrumentMetadata."""
        if not isinstance(raw, dict):
            stats.record_rejection("non_dict_record")
            return None

        # Check required fields
        token = str(raw.get("token", "")).strip()
        raw_symbol = str(raw.get("symbol", "")).strip()
        exch_seg = str(raw.get("exch_seg", "")).strip().upper()
        name = str(raw.get("name", "")).strip()

        if not token or not raw_symbol or not exch_seg:
            stats.record_rejection("missing_essential_fields")
            return None

        # Exchange/Segment validation
        if exch_seg not in ("NSE", "NFO", "BSE"):
            stats.record_rejection("unsupported_segment")
            return None

        # Lotsize & strike
        try:
            lotsize = int(float(raw.get("lotsize", 1) or 1))
        except (ValueError, TypeError):
            lotsize = 1

        try:
            strike = float(raw.get("strike", 0.0) or 0.0)
            # In Angel scrip master, strike is often multiplied by 100
            if strike > 100000.0:
                strike = strike / 100.0
        except (ValueError, TypeError):
            strike = 0.0

        try:
            tick_size = float(raw.get("tick_size", 0.05) or 0.05)
            if tick_size > 10.0:
                tick_size = tick_size / 100.0
        except (ValueError, TypeError):
            tick_size = 0.05

        expiry = _parse_expiry_date(str(raw.get("expiry", "")))
        instrument_type = str(raw.get("instrumenttype", "")).strip().upper()

        is_fno = exch_seg == "NFO"

        # If F&O, reject expired contracts
        if is_fno:
            if not expiry:
                stats.record_rejection("fno_missing_expiry")
                return None
            if expiry < today:
                stats.record_rejection("fno_expired")
                return None

        # Clean symbol: for NSE equity, remove '-EQ' suffix
        clean_symbol = raw_symbol
        if exch_seg == "NSE" and clean_symbol.endswith("-EQ"):
            clean_symbol = clean_symbol[:-3]

        return InstrumentMetadata(
            token=token,
            symbol=clean_symbol,
            raw_symbol=raw_symbol,
            name=name,
            expiry=expiry,
            strike=strike,
            lotsize=lotsize,
            instrument_type=instrument_type,
            exch_seg=exch_seg,
            tick_size=tick_size,
            is_fno=is_fno,
        )

    def discover(self, force_refresh: bool = False) -> DiscoveryStats:
        """Run complete discovery: fetch, validate, categorize."""
        stats = DiscoveryStats(
            retrieval_time_ist=now_ist().isoformat(),
            source_url=self.master_url,
        )

        raw_records = self.fetch_raw_master(force_refresh=force_refresh)
        stats.records_received = len(raw_records)

        equity_list: List[InstrumentMetadata] = []
        fno_list: List[InstrumentMetadata] = []
        fno_by_und: Dict[str, List[InstrumentMetadata]] = {}
        today = now_ist().date()

        for raw in raw_records:
            inst = self.parse_record(raw, stats, today)
            if inst is None:
                continue

            stats.records_valid += 1

            if inst.exch_seg == "NSE" and inst.raw_symbol.endswith("-EQ"):
                equity_list.append(inst)
                stats.eligible_equity += 1
            elif inst.exch_seg == "NFO":
                fno_list.append(inst)
                stats.eligible_fno += 1
                # Index by underlying name or base symbol
                und = inst.name.upper() if inst.name else inst.symbol
                fno_by_und.setdefault(und, []).append(inst)

        # Prioritize high-liquidity equities (those with active F&O derivatives listed on NSE)
        fno_symbols_clean = {s.upper().replace("-EQ", "") for s in fno_by_und.keys()}
        
        valid_equities = [
            inst for inst in equity_list 
            if "TEST" not in inst.symbol.upper() and not inst.symbol.startswith("111") and inst.symbol.replace("&", "").replace("-", "").replace("_", "").isalnum()
        ]
        
        fno_equities = [
            inst for inst in valid_equities 
            if inst.symbol in fno_symbols_clean or inst.name.upper() in fno_symbols_clean
        ]
        non_fno_equities = [
            inst for inst in valid_equities 
            if not (inst.symbol in fno_symbols_clean or inst.name.upper() in fno_symbols_clean)
        ]
        
        self._equity_universe = fno_equities + non_fno_equities
        self._fno_universe = fno_list
        self._fno_by_underlying = fno_by_und
        self._latest_stats = stats

        logger.info(
            "Discovery completed: %d total received, %d valid, %d equity eligible (%d liquid F&O), %d F&O eligible",
            stats.records_received,
            stats.records_valid,
            stats.eligible_equity,
            len(fno_equities),
            stats.eligible_fno,
        )
        return stats

    def get_equity_universe(self, limit: Optional[int] = None) -> List[InstrumentMetadata]:
        """Return discovered equity instruments."""
        if not self._equity_universe:
            self.discover()
        if limit and limit > 0:
            return self._equity_universe[:limit]
        return list(self._equity_universe)

    def get_fno_contracts_for_underlying(
        self,
        underlying: str,
        option_type: Optional[str] = None,  # "CE", "PE", "FUT"
        min_expiry: Optional[date] = None,
    ) -> List[InstrumentMetadata]:
        """Find active F&O contracts for a given underlying symbol dynamically."""
        if not self._fno_universe:
            self.discover()

        target = underlying.upper().strip()
        contracts = self._fno_by_underlying.get(target, [])
        if not contracts:
            # Fallback search if underlying differs from name
            contracts = [c for c in self._fno_universe if c.name.upper() == target or c.symbol.startswith(target)]

        filtered = []
        today = now_ist().date()
        target_min_expiry = min_expiry or today

        for c in contracts:
            if c.expiry and c.expiry < target_min_expiry:
                continue
            if option_type:
                if option_type == "FUT" and "FUT" not in c.instrument_type:
                    continue
                if option_type in ("CE", "PE"):
                    if not c.raw_symbol.endswith(option_type):
                        continue
            filtered.append(c)

        # Sort by expiry ascending, then by strike
        filtered.sort(key=lambda x: (x.expiry or date.max, x.strike))
        return filtered
