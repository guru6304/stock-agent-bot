"""Durable Signal Deduplication Store.

Persists emitted paper signals in a lightweight SQLite database to prevent
duplicate alerts across bot and scanner process restarts.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

from india_market_config import DEFAULT_SCANNER_CONFIG, now_ist
from indian_signals import IndianTradeSignal

logger = logging.getLogger("signal-dedup")

DB_PATH = Path("logs/indian_signal_dedup.db")


class DurableSignalDeduplicator:
    """Manages restart-safe deduplication of emitted trading signals."""

    def __init__(self, db_path: Optional[Path] = None, cooldown_hours: Optional[float] = None):
        self.db_path = db_path or DB_PATH
        self.cooldown_hours = (
            cooldown_hours
            if cooldown_hours is not None
            else DEFAULT_SCANNER_CONFIG.signal_dedup_hours
        )
        self._init_db()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS emitted_signals (
                    dedup_key TEXT PRIMARY KEY,
                    signal_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    horizon TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    emitted_epoch REAL NOT NULL,
                    emitted_time_ist TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_emitted_time ON emitted_signals(emitted_epoch)")
            conn.commit()
        finally:
            conn.close()

    def _make_key(self, signal: IndianTradeSignal) -> str:
        """Create a semantic deduplication key."""
        today_date = now_ist().strftime("%Y-%m-%d")
        return f"{signal.symbol}:{signal.horizon.value}:{signal.direction}:{today_date}"

    def is_duplicate(self, signal: IndianTradeSignal) -> bool:
        """Return True if the same signal was emitted within the cooldown window."""
        key = self._make_key(signal)
        now_epoch = time.time()
        cooldown_seconds = self.cooldown_hours * 3600.0

        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.execute(
                "SELECT emitted_epoch FROM emitted_signals WHERE dedup_key = ?",
                (key,),
            )
            row = cur.fetchone()
            if row:
                last_epoch = float(row[0])
                if (now_epoch - last_epoch) < cooldown_seconds:
                    return True
            return False
        finally:
            conn.close()

    def record_emission(self, signal: IndianTradeSignal) -> None:
        """Record an emitted signal into the persistent database."""
        key = self._make_key(signal)
        now_epoch = time.time()
        ts_ist = now_ist().strftime("%Y-%m-%d %H:%M:%S IST")

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO emitted_signals
                (dedup_key, signal_id, symbol, horizon, direction, emitted_epoch, emitted_time_ist, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    signal.signal_id,
                    signal.symbol,
                    signal.horizon.value,
                    signal.direction,
                    now_epoch,
                    ts_ist,
                    json.dumps(signal.to_dict()),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def cleanup_old_records(self, max_age_days: int = 7) -> int:
        """Prune records older than max_age_days."""
        cutoff_epoch = time.time() - (max_age_days * 86400.0)
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.execute(
                "DELETE FROM emitted_signals WHERE emitted_epoch < ?",
                (cutoff_epoch,),
            )
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()
