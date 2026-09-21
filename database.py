#!/usr/bin/env python3
"""
SQLite Database Layer — centralized persistence for the stock agent.

Replaces JSON/CSV file-based storage with a proper relational database.
Provides migration support, CRUD operations, and backward-compatible
interfaces for all modules.

Tables:
- alerts:          Trade alerts from the signal engine
- positions:       Open/closed positions (trailing stop + paper trading)
- paper_state:     Paper trading portfolio state (cash, config)
- paper_trades:    Paper trading closed trade history
- alert_history:   Alert deduplication cooldown tracking
- backtest_trades: Backtest simulation results
- watchlist:       Ticker watchlist
- portfolio:       Portfolio holdings and config

Usage:
    import database as db
    db.init()  # Create/migrate tables
    db.insert_alert(...)
    alerts = db.get_alerts(limit=50)
"""

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(os.getenv("DB_PATH", "logs/stock_agent.db"))

# Current schema version — increment when adding migrations
SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

def _get_db_path() -> str:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return str(DB_PATH)


@contextmanager
def get_connection():
    """Context manager for database connections with WAL mode."""
    conn = sqlite3.connect(_get_db_path(), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema creation & migration
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
-- Trade alerts logged by the signal engine
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,
    signal_score INTEGER NOT NULL DEFAULT 0,
    signals TEXT DEFAULT '',
    entry_price REAL DEFAULT 0,
    stop_loss REAL DEFAULT 0,
    shares INTEGER DEFAULT 0,
    position_value REAL DEFAULT 0,
    max_loss REAL DEFAULT 0,
    target_1 REAL DEFAULT 0,
    target_2 REAL DEFAULT 0,
    target_3 REAL DEFAULT 0,
    fundamental_score INTEGER DEFAULT 0,
    news_sentiment REAL DEFAULT 0
);

-- Positions tracked by trailing stop manager
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_price REAL NOT NULL,
    entry_date TEXT NOT NULL,
    initial_stop REAL DEFAULT 0,
    current_stop REAL DEFAULT 0,
    trailing_stop REAL DEFAULT 0,
    highest_price REAL DEFAULT 0,
    lowest_price REAL DEFAULT 0,
    shares INTEGER DEFAULT 0,
    target_1 REAL DEFAULT 0,
    target_2 REAL DEFAULT 0,
    target_3 REAL DEFAULT 0,
    t1_hit INTEGER DEFAULT 0,
    t2_hit INTEGER DEFAULT 0,
    t3_hit INTEGER DEFAULT 0,
    status TEXT DEFAULT 'open',
    exit_price REAL DEFAULT 0,
    exit_date TEXT DEFAULT '',
    exit_reason TEXT DEFAULT '',
    pnl REAL DEFAULT 0,
    pnl_pct REAL DEFAULT 0,
    updates_json TEXT DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_positions_ticker ON positions(ticker);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);

-- Paper trading portfolio state
CREATE TABLE IF NOT EXISTS paper_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    starting_capital REAL NOT NULL DEFAULT 100000,
    cash REAL NOT NULL DEFAULT 100000,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_updated TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Paper trading positions (open)
CREATE TABLE IF NOT EXISTS paper_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_price REAL NOT NULL,
    entry_date TEXT NOT NULL,
    shares INTEGER NOT NULL,
    cost_basis REAL DEFAULT 0,
    stop_loss REAL DEFAULT 0,
    current_stop REAL DEFAULT 0,
    highest_price REAL DEFAULT 0,
    lowest_price REAL DEFAULT 0,
    target_1 REAL DEFAULT 0,
    target_2 REAL DEFAULT 0,
    target_3 REAL DEFAULT 0,
    t1_hit INTEGER DEFAULT 0,
    t2_hit INTEGER DEFAULT 0,
    t3_hit INTEGER DEFAULT 0,
    signal_score INTEGER DEFAULT 0,
    triggered_signals TEXT DEFAULT '',
    risk_per_share REAL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_paper_positions_ticker ON paper_positions(ticker);

-- Paper trading closed trades
CREATE TABLE IF NOT EXISTS paper_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_price REAL NOT NULL,
    entry_date TEXT NOT NULL,
    exit_price REAL NOT NULL,
    exit_date TEXT NOT NULL,
    exit_reason TEXT DEFAULT '',
    shares INTEGER NOT NULL,
    pnl REAL DEFAULT 0,
    pnl_pct REAL DEFAULT 0,
    bars_held INTEGER DEFAULT 0,
    signal_score INTEGER DEFAULT 0,
    triggered_signals TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_paper_trades_ticker ON paper_trades(ticker);

-- Alert deduplication history
CREATE TABLE IF NOT EXISTS alert_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dedup_key TEXT NOT NULL UNIQUE,
    timestamp TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Backtest results
CREATE TABLE IF NOT EXISTS backtest_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_date TEXT DEFAULT '',
    entry_price REAL DEFAULT 0,
    stop_loss REAL DEFAULT 0,
    target_1 REAL DEFAULT 0,
    exit_date TEXT DEFAULT '',
    exit_price REAL DEFAULT 0,
    exit_reason TEXT DEFAULT '',
    pnl REAL DEFAULT 0,
    pnl_pct REAL DEFAULT 0,
    bars_held INTEGER DEFAULT 0,
    signal_score INTEGER DEFAULT 0,
    signals TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_backtest_run ON backtest_trades(run_id);

-- Watchlist
CREATE TABLE IF NOT EXISTS watchlist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL UNIQUE,
    sector TEXT DEFAULT 'Technology',
    notes TEXT DEFAULT ''
);

-- Portfolio holdings
CREATE TABLE IF NOT EXISTS portfolio (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL UNIQUE,
    shares INTEGER DEFAULT 0,
    avg_cost REAL DEFAULT 0,
    current_price REAL DEFAULT 0,
    current_value REAL DEFAULT 0,
    cost_basis REAL DEFAULT 0,
    unrealized_pnl REAL DEFAULT 0,
    pnl_pct REAL DEFAULT 0,
    sector TEXT DEFAULT ''
);

-- Portfolio config (single row)
CREATE TABLE IF NOT EXISTS portfolio_config (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    total_portfolio_value REAL DEFAULT 1000000,
    available_cash REAL DEFAULT 1000000,
    max_risk_per_trade_pct REAL DEFAULT 1.0,
    max_position_size_pct REAL DEFAULT 10.0,
    last_updated TEXT DEFAULT (datetime('now'))
);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);
"""


def init() -> None:
    """Initialize the database — create tables if they don't exist."""
    with get_connection() as conn:
        conn.executescript(SCHEMA_SQL)
        # Set initial schema version
        existing = conn.execute("SELECT version FROM schema_version").fetchone()
        if not existing:
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
    logger.info("Database initialized at %s", DB_PATH)


def get_schema_version() -> int:
    """Get current schema version."""
    with get_connection() as conn:
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        return row["version"] if row else 0


# ---------------------------------------------------------------------------
# Alerts CRUD
# ---------------------------------------------------------------------------

def insert_alert(
    ticker: str,
    direction: str,
    signal_score: int,
    signals: str,
    entry_price: float = 0,
    stop_loss: float = 0,
    shares: int = 0,
    position_value: float = 0,
    max_loss: float = 0,
    target_1: float = 0,
    target_2: float = 0,
    target_3: float = 0,
    fundamental_score: int = 0,
    news_sentiment: float = 0,
) -> int:
    """Insert a new trade alert. Returns the row id."""
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO alerts (ticker, direction, signal_score, signals,
               entry_price, stop_loss, shares, position_value, max_loss,
               target_1, target_2, target_3, fundamental_score, news_sentiment)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ticker, direction, signal_score, signals,
             entry_price, stop_loss, shares, position_value, max_loss,
             target_1, target_2, target_3, fundamental_score, news_sentiment),
        )
        return cur.lastrowid


def get_alerts(limit: int = 100, ticker: Optional[str] = None) -> List[Dict]:
    """Get recent alerts, optionally filtered by ticker."""
    with get_connection() as conn:
        if ticker:
            rows = conn.execute(
                "SELECT * FROM alerts WHERE ticker = ? ORDER BY id DESC LIMIT ?",
                (ticker, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Positions CRUD (trailing stop manager)
# ---------------------------------------------------------------------------

def insert_position(
    ticker: str, direction: str, entry_price: float, entry_date: str,
    initial_stop: float, shares: int,
    target_1: float = 0, target_2: float = 0, target_3: float = 0,
) -> int:
    """Insert a new tracked position."""
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO positions (ticker, direction, entry_price, entry_date,
               initial_stop, current_stop, trailing_stop, highest_price, lowest_price,
               shares, target_1, target_2, target_3)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ticker, direction, entry_price, entry_date,
             initial_stop, initial_stop, initial_stop,
             entry_price, entry_price, shares, target_1, target_2, target_3),
        )
        return cur.lastrowid


def get_open_positions() -> List[Dict]:
    """Get all open positions."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM positions WHERE status = 'open' ORDER BY entry_date"
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["t1_hit"] = bool(d["t1_hit"])
            d["t2_hit"] = bool(d["t2_hit"])
            d["t3_hit"] = bool(d["t3_hit"])
            try:
                d["updates"] = json.loads(d.pop("updates_json", "[]"))
            except (json.JSONDecodeError, TypeError):
                d["updates"] = []
            result.append(d)
        return result


def get_closed_positions(limit: int = 50) -> List[Dict]:
    """Get recently closed positions."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM positions WHERE status = 'closed' ORDER BY exit_date DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def update_position(position_id: int, **kwargs) -> None:
    """Update position fields by id."""
    allowed = {
        "current_stop", "trailing_stop", "highest_price", "lowest_price",
        "t1_hit", "t2_hit", "t3_hit", "status", "exit_price", "exit_date",
        "exit_reason", "pnl", "pnl_pct", "updates_json", "shares",
    }
    fields = {k: v for k, v in kwargs.items() if k in allowed}
    if not fields:
        return
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [position_id]
    with get_connection() as conn:
        conn.execute(f"UPDATE positions SET {set_clause} WHERE id = ?", values)


def has_open_position(ticker: str) -> bool:
    """Check if ticker has an open position."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM positions WHERE ticker = ? AND status = 'open'",
            (ticker,),
        ).fetchone()
        return row["cnt"] > 0


# ---------------------------------------------------------------------------
# Paper trading CRUD
# ---------------------------------------------------------------------------

def get_paper_state() -> Dict:
    """Get paper trading state, creating default if needed."""
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM paper_state WHERE id = 1").fetchone()
        if row:
            state = dict(row)
        else:
            conn.execute(
                "INSERT INTO paper_state (id, starting_capital, cash) VALUES (1, ?, ?)",
                (100000, 100000),
            )
            state = {"id": 1, "starting_capital": 100000, "cash": 100000,
                     "created_at": datetime.now().isoformat(),
                     "last_updated": datetime.now().isoformat()}
        # Attach positions and trades
        positions = conn.execute("SELECT * FROM paper_positions ORDER BY entry_date").fetchall()
        trades = conn.execute("SELECT * FROM paper_trades ORDER BY exit_date DESC").fetchall()
        state["open_positions"] = [dict(r) for r in positions]
        state["closed_trades"] = [dict(r) for r in trades]
        return state


def update_paper_cash(cash: float) -> None:
    """Update paper trading cash balance."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE paper_state SET cash = ?, last_updated = ? WHERE id = 1",
            (cash, datetime.now().isoformat()),
        )


def reset_paper_state(starting_capital: float = 1000000) -> None:
    """Reset paper trading to clean state."""
    with get_connection() as conn:
        conn.execute("DELETE FROM paper_positions")
        conn.execute("DELETE FROM paper_trades")
        conn.execute("DELETE FROM paper_state")
        conn.execute(
            "INSERT INTO paper_state (id, starting_capital, cash) VALUES (1, ?, ?)",
            (starting_capital, starting_capital),
        )


def insert_paper_position(
    ticker: str, direction: str, entry_price: float, entry_date: str,
    shares: int, cost_basis: float, stop_loss: float, current_stop: float,
    target_1: float = 0, target_2: float = 0, target_3: float = 0,
    signal_score: int = 0, triggered_signals: str = "", risk_per_share: float = 0,
) -> int:
    """Insert a paper trading position."""
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO paper_positions (ticker, direction, entry_price, entry_date,
               shares, cost_basis, stop_loss, current_stop, highest_price, lowest_price,
               target_1, target_2, target_3, signal_score, triggered_signals, risk_per_share)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ticker, direction, entry_price, entry_date, shares, cost_basis,
             stop_loss, current_stop, entry_price, entry_price,
             target_1, target_2, target_3, signal_score, triggered_signals, risk_per_share),
        )
        return cur.lastrowid


def update_paper_position(position_id: int, **kwargs) -> None:
    """Update a paper position by id."""
    allowed = {
        "shares", "current_stop", "highest_price", "lowest_price",
        "t1_hit", "t2_hit", "t3_hit",
    }
    fields = {k: v for k, v in kwargs.items() if k in allowed}
    if not fields:
        return
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [position_id]
    with get_connection() as conn:
        conn.execute(f"UPDATE paper_positions SET {set_clause} WHERE id = ?", values)


def delete_paper_position(position_id: int) -> None:
    """Remove a paper position (after full exit)."""
    with get_connection() as conn:
        conn.execute("DELETE FROM paper_positions WHERE id = ?", (position_id,))


def insert_paper_trade(
    ticker: str, direction: str, entry_price: float, entry_date: str,
    exit_price: float, exit_date: str, exit_reason: str, shares: int,
    pnl: float, pnl_pct: float, bars_held: int = 0,
    signal_score: int = 0, triggered_signals: str = "",
) -> int:
    """Record a closed paper trade."""
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO paper_trades (ticker, direction, entry_price, entry_date,
               exit_price, exit_date, exit_reason, shares, pnl, pnl_pct,
               bars_held, signal_score, triggered_signals)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ticker, direction, entry_price, entry_date, exit_price, exit_date,
             exit_reason, shares, pnl, pnl_pct, bars_held, signal_score,
             triggered_signals),
        )
        return cur.lastrowid


def get_paper_positions() -> List[Dict]:
    """Get all open paper positions."""
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM paper_positions ORDER BY entry_date").fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["t1_hit"] = bool(d["t1_hit"])
            d["t2_hit"] = bool(d["t2_hit"])
            d["t3_hit"] = bool(d["t3_hit"])
            result.append(d)
        return result


def get_paper_trades(limit: int = 100) -> List[Dict]:
    """Get closed paper trades."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM paper_trades ORDER BY exit_date DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Alert deduplication
# ---------------------------------------------------------------------------

def check_alert_duplicate(dedup_key: str, cooldown_hours: float = 24) -> bool:
    """Check if an alert key is within the cooldown period. Returns True if duplicate."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT timestamp FROM alert_history WHERE dedup_key = ?", (dedup_key,)
        ).fetchone()
        if not row:
            return False
        last_time = datetime.fromisoformat(row["timestamp"])
        elapsed = (datetime.now() - last_time).total_seconds() / 3600
        return elapsed < cooldown_hours


def record_alert_dedup(dedup_key: str) -> None:
    """Record an alert for deduplication tracking."""
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO alert_history (dedup_key, timestamp)
               VALUES (?, ?)
               ON CONFLICT(dedup_key) DO UPDATE SET timestamp = ?""",
            (dedup_key, datetime.now().isoformat(), datetime.now().isoformat()),
        )


def prune_alert_history(days: int = 7) -> int:
    """Remove alert history entries older than N days."""
    with get_connection() as conn:
        cur = conn.execute(
            "DELETE FROM alert_history WHERE timestamp < datetime('now', ?)",
            (f"-{days} days",),
        )
        return cur.rowcount


# ---------------------------------------------------------------------------
# Backtest trades
# ---------------------------------------------------------------------------

def insert_backtest_trade(run_id: str, trade: Dict) -> int:
    """Insert a single backtest trade result."""
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO backtest_trades (run_id, ticker, direction, entry_date,
               entry_price, stop_loss, target_1, exit_date, exit_price, exit_reason,
               pnl, pnl_pct, bars_held, signal_score, signals)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, trade.get("ticker", ""), trade.get("direction", ""),
             trade.get("entry_date", ""), trade.get("entry_price", 0),
             trade.get("stop_loss", 0), trade.get("target_1", 0),
             trade.get("exit_date", ""), trade.get("exit_price", 0),
             trade.get("exit_reason", ""), trade.get("pnl", 0),
             trade.get("pnl_pct", 0), trade.get("bars_held", 0),
             trade.get("signal_score", 0), trade.get("signals", "")),
        )
        return cur.lastrowid


def get_backtest_trades(run_id: Optional[str] = None) -> List[Dict]:
    """Get backtest trades, optionally filtered by run_id."""
    with get_connection() as conn:
        if run_id:
            rows = conn.execute(
                "SELECT * FROM backtest_trades WHERE run_id = ? ORDER BY id",
                (run_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM backtest_trades ORDER BY run_id, id"
            ).fetchall()
        return [dict(r) for r in rows]


def get_backtest_runs() -> List[str]:
    """Get list of distinct backtest run IDs."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT run_id FROM backtest_trades ORDER BY run_id DESC"
        ).fetchall()
        return [r["run_id"] for r in rows]


# ---------------------------------------------------------------------------
# Watchlist CRUD
# ---------------------------------------------------------------------------

def get_watchlist() -> List[Dict]:
    """Get all watchlist entries."""
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM watchlist ORDER BY ticker").fetchall()
        return [dict(r) for r in rows]


def add_to_watchlist(ticker: str, sector: str = "Technology", notes: str = "") -> bool:
    """Add ticker to watchlist. Returns False if already exists."""
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO watchlist (ticker, sector, notes) VALUES (?, ?, ?)",
                (ticker.upper(), sector, notes),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def remove_from_watchlist(ticker: str) -> bool:
    """Remove ticker from watchlist. Returns True if removed."""
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM watchlist WHERE ticker = ?", (ticker.upper(),))
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Portfolio CRUD
# ---------------------------------------------------------------------------

def get_portfolio_config() -> Dict:
    """Get portfolio configuration."""
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM portfolio_config WHERE id = 1").fetchone()
        if row:
            return dict(row)
        # Insert default
        conn.execute(
            """INSERT INTO portfolio_config (id, total_portfolio_value, available_cash)
               VALUES (1, 50000, 10000)"""
        )
        return {"total_portfolio_value": 50000, "available_cash": 10000,
                "max_risk_per_trade_pct": 1.0, "max_position_size_pct": 10.0}


def update_portfolio_config(**kwargs) -> None:
    """Update portfolio config values."""
    allowed = {"total_portfolio_value", "available_cash",
               "max_risk_per_trade_pct", "max_position_size_pct"}
    fields = {k: v for k, v in kwargs.items() if k in allowed}
    if not fields:
        return
    fields["last_updated"] = datetime.now().isoformat()
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values())
    with get_connection() as conn:
        conn.execute(f"UPDATE portfolio_config SET {set_clause} WHERE id = 1", values)


def get_holdings() -> List[Dict]:
    """Get all portfolio holdings."""
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM portfolio ORDER BY ticker").fetchall()
        return [dict(r) for r in rows]


def upsert_holding(
    ticker: str, shares: int, avg_cost: float, sector: str = "",
    current_price: float = 0,
) -> None:
    """Insert or update a portfolio holding."""
    current_value = current_price * shares
    cost_basis = avg_cost * shares
    unrealized_pnl = current_value - cost_basis
    pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0

    with get_connection() as conn:
        conn.execute(
            """INSERT INTO portfolio (ticker, shares, avg_cost, current_price,
               current_value, cost_basis, unrealized_pnl, pnl_pct, sector)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(ticker) DO UPDATE SET
               shares = ?, avg_cost = ?, current_price = ?,
               current_value = ?, cost_basis = ?, unrealized_pnl = ?,
               pnl_pct = ?, sector = ?""",
            (ticker, shares, avg_cost, current_price, current_value,
             cost_basis, unrealized_pnl, pnl_pct, sector,
             shares, avg_cost, current_price, current_value,
             cost_basis, unrealized_pnl, pnl_pct, sector),
        )


def remove_holding(ticker: str) -> bool:
    """Remove a holding. Returns True if removed."""
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM portfolio WHERE ticker = ?", (ticker.upper(),))
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Data migration from JSON/CSV files
# ---------------------------------------------------------------------------

def migrate_from_files(
    alert_history_path: str = "logs/alert_history.json",
    positions_path: str = "logs/open_positions.json",
    paper_path: str = "logs/paper_portfolio.json",
    dashboard_csv_path: str = "logs/dashboard.csv",
    watchlist_csv_path: str = "watchlist.csv",
    portfolio_json_path: str = "portfolio.json",
) -> Dict[str, int]:
    """Migrate existing file-based data into SQLite.

    Returns dict of table_name -> rows_migrated.
    """
    counts = {}
    init()

    # 1. Alert history
    if os.path.exists(alert_history_path):
        try:
            with open(alert_history_path) as f:
                history = json.load(f)
            with get_connection() as conn:
                for key, ts in history.items():
                    conn.execute(
                        """INSERT OR IGNORE INTO alert_history (dedup_key, timestamp)
                           VALUES (?, ?)""",
                        (key, ts),
                    )
            counts["alert_history"] = len(history)
        except Exception as e:
            logger.error("Failed to migrate alert history: %s", e)

    # 2. Positions (trailing stop)
    if os.path.exists(positions_path):
        try:
            with open(positions_path) as f:
                positions = json.load(f)
            with get_connection() as conn:
                for p in positions:
                    conn.execute(
                        """INSERT INTO positions (ticker, direction, entry_price, entry_date,
                           initial_stop, current_stop, trailing_stop, highest_price, lowest_price,
                           shares, target_1, target_2, target_3, t1_hit, t2_hit, t3_hit,
                           status, exit_price, exit_date, exit_reason, pnl, pnl_pct, updates_json)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (p.get("ticker", ""), p.get("direction", ""),
                         p.get("entry_price", 0), p.get("entry_date", ""),
                         p.get("initial_stop", 0), p.get("current_stop", 0),
                         p.get("trailing_stop", 0), p.get("highest_price", 0),
                         p.get("lowest_price", 0), p.get("shares", 0),
                         p.get("target_1", 0), p.get("target_2", 0), p.get("target_3", 0),
                         int(p.get("t1_hit", False)), int(p.get("t2_hit", False)),
                         int(p.get("t3_hit", False)),
                         p.get("status", "open"), p.get("exit_price", 0),
                         p.get("exit_date", ""), p.get("exit_reason", ""),
                         p.get("pnl", 0), p.get("pnl_pct", 0),
                         json.dumps(p.get("updates", []))),
                    )
            counts["positions"] = len(positions)
        except Exception as e:
            logger.error("Failed to migrate positions: %s", e)

    # 3. Paper trading
    if os.path.exists(paper_path):
        try:
            with open(paper_path) as f:
                paper = json.load(f)
            reset_paper_state(paper.get("starting_capital", 100000))
            update_paper_cash(paper.get("cash", 100000))
            with get_connection() as conn:
                for p in paper.get("open_positions", []):
                    sigs = ",".join(p.get("triggered_signals", []))
                    conn.execute(
                        """INSERT INTO paper_positions (ticker, direction, entry_price,
                           entry_date, shares, cost_basis, stop_loss, current_stop,
                           highest_price, lowest_price, target_1, target_2, target_3,
                           t1_hit, t2_hit, t3_hit, signal_score, triggered_signals, risk_per_share)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (p["ticker"], p["direction"], p["entry_price"], p["entry_date"],
                         p["shares"], p.get("cost_basis", 0), p.get("stop_loss", 0),
                         p.get("current_stop", 0), p.get("highest_price", 0),
                         p.get("lowest_price", 0), p.get("target_1", 0),
                         p.get("target_2", 0), p.get("target_3", 0),
                         int(p.get("t1_hit", False)), int(p.get("t2_hit", False)),
                         int(p.get("t3_hit", False)), p.get("signal_score", 0),
                         sigs, p.get("risk_per_share", 0)),
                    )
                for t in paper.get("closed_trades", []):
                    sigs = ",".join(t.get("triggered_signals", []))
                    conn.execute(
                        """INSERT INTO paper_trades (ticker, direction, entry_price,
                           entry_date, exit_price, exit_date, exit_reason, shares,
                           pnl, pnl_pct, bars_held, signal_score, triggered_signals)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (t["ticker"], t["direction"], t["entry_price"], t["entry_date"],
                         t["exit_price"], t["exit_date"], t.get("exit_reason", ""),
                         t["shares"], t["pnl"], t.get("pnl_pct", 0),
                         t.get("bars_held", 0), t.get("signal_score", 0), sigs),
                    )
            counts["paper_positions"] = len(paper.get("open_positions", []))
            counts["paper_trades"] = len(paper.get("closed_trades", []))
        except Exception as e:
            logger.error("Failed to migrate paper trading: %s", e)

    # 4. Dashboard CSV (alerts)
    if os.path.exists(dashboard_csv_path):
        import csv
        try:
            with open(dashboard_csv_path, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            with get_connection() as conn:
                for row in rows:
                    conn.execute(
                        """INSERT INTO alerts (timestamp, ticker, direction, signal_score,
                           signals, entry_price, stop_loss, shares, position_value, max_loss,
                           target_1, target_2, target_3, fundamental_score, news_sentiment)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (row.get("timestamp", ""), row.get("ticker", ""),
                         row.get("direction", ""), int(row.get("signal_score", 0)),
                         row.get("signals", ""), float(row.get("entry_price", 0)),
                         float(row.get("stop_loss", 0)), int(row.get("shares", 0)),
                         float(row.get("position_value", 0)), float(row.get("max_loss", 0)),
                         float(row.get("target_1", 0)), float(row.get("target_2", 0)),
                         float(row.get("target_3", 0)), int(row.get("fundamental_score", 0)),
                         float(row.get("news_sentiment", 0))),
                    )
            counts["alerts"] = len(rows)
        except Exception as e:
            logger.error("Failed to migrate dashboard CSV: %s", e)

    # 5. Watchlist
    if os.path.exists(watchlist_csv_path):
        import csv
        try:
            with open(watchlist_csv_path, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            for row in rows:
                add_to_watchlist(row["ticker"], row.get("sector", ""), row.get("notes", ""))
            counts["watchlist"] = len(rows)
        except Exception as e:
            logger.error("Failed to migrate watchlist: %s", e)

    # 6. Portfolio
    if os.path.exists(portfolio_json_path):
        try:
            with open(portfolio_json_path) as f:
                port = json.load(f)
            update_portfolio_config(
                total_portfolio_value=port.get("total_portfolio_value", 50000),
                available_cash=port.get("available_cash", 10000),
                max_risk_per_trade_pct=port.get("max_risk_per_trade_pct", 1.0),
                max_position_size_pct=port.get("max_position_size_pct", 10.0),
            )
            for h in port.get("holdings", []):
                upsert_holding(
                    ticker=h["ticker"], shares=h["shares"],
                    avg_cost=h["avg_cost"], sector=h.get("sector", ""),
                    current_price=h.get("current_price", 0),
                )
            counts["holdings"] = len(port.get("holdings", []))
        except Exception as e:
            logger.error("Failed to migrate portfolio: %s", e)

    logger.info("Migration complete: %s", counts)
    return counts


# ---------------------------------------------------------------------------
# CLI for migration and inspection
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stock Agent Database")
    parser.add_argument("--init", action="store_true", help="Initialize database")
    parser.add_argument("--migrate", action="store_true", help="Migrate JSON/CSV data to SQLite")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

    if args.init:
        init()
        print(f"Database initialized at {DB_PATH}")
    elif args.migrate:
        init()
        counts = migrate_from_files()
        print(f"\nMigration results:")
        for table, count in counts.items():
            print(f"  {table}: {count} rows")
        print(f"\nDatabase: {DB_PATH}")
    elif args.stats:
        init()
        with get_connection() as conn:
            tables = ["alerts", "positions", "paper_positions", "paper_trades",
                       "alert_history", "backtest_trades", "watchlist", "portfolio"]
            print(f"\n{'=' * 40}")
            print(f"  DATABASE: {DB_PATH}")
            print(f"{'=' * 40}")
            for t in tables:
                row = conn.execute(f"SELECT COUNT(*) as cnt FROM {t}").fetchone()
                print(f"  {t:<20} {row['cnt']:>6} rows")
            print(f"{'=' * 40}\n")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
