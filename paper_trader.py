#!/usr/bin/env python3
"""
Paper Trading Engine — simulates live trading with a virtual portfolio.

Automatically enters positions when the signal engine fires, manages
trailing stops, exits at targets or stops, and tracks full trade history
with performance metrics.

State is persisted in logs/paper_portfolio.json between runs.

Usage:
    python3 paper_trader.py --status           # Show current portfolio
    python3 paper_trader.py --history          # Show trade history
    python3 paper_trader.py --performance      # Show performance report
    python3 paper_trader.py --reset            # Reset to starting capital
    python3 paper_trader.py --reset --capital 100000  # Reset with custom capital
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from position_sizing import PositionPlan
from signal_engine import TradeAlert
import trade_journal
import data_layer

load_dotenv()
logger = logging.getLogger(__name__)


def is_indian_market() -> bool:
    return os.getenv("MARKET", "").upper() == "INDIA" or os.getenv("TIMEZONE") == "Asia/Kolkata"


def get_currency_symbol() -> str:
    return "₹" if is_indian_market() else "$"


PAPER_FILE = Path("logs/paper_portfolio.json")
DEFAULT_CAPITAL = float(os.getenv("INITIAL_CAPITAL", os.getenv("PAPER_STARTING_CAPITAL", "1000000")))
MAX_OPEN_POSITIONS = int(os.getenv("PAPER_MAX_POSITIONS", "10"))
MAX_RISK_PER_TRADE_PCT = float(os.getenv("MAX_RISK_PER_TRADE_PCT", "1.0"))
MAX_POSITION_SIZE_PCT = float(os.getenv("MAX_POSITION_SIZE_PCT", "10.0"))


# ---------------------------------------------------------------------------
# Portfolio state management
# ---------------------------------------------------------------------------

def _default_state() -> dict:
    """Create a fresh paper trading state."""
    return {
        "starting_capital": DEFAULT_CAPITAL,
        "cash": DEFAULT_CAPITAL,
        "open_positions": [],
        "closed_trades": [],
        "created_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
    }


def load_state() -> dict:
    """Load paper portfolio state from disk."""
    if not PAPER_FILE.exists():
        return _default_state()
    try:
        with open(PAPER_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        logger.warning("Corrupt paper portfolio file — starting fresh")
        return _default_state()


def save_state(state: dict) -> None:
    """Persist paper portfolio state to disk."""
    PAPER_FILE.parent.mkdir(parents=True, exist_ok=True)
    state["last_updated"] = datetime.now().isoformat()
    with open(PAPER_FILE, "w") as f:
        json.dump(state, f, indent=2)


def reset_state(starting_capital: float = DEFAULT_CAPITAL) -> dict:
    """Reset paper portfolio to clean state."""
    state = _default_state()
    state["starting_capital"] = starting_capital
    state["cash"] = starting_capital
    save_state(state)
    logger.info("Paper portfolio reset with $%,.2f starting capital", starting_capital)
    return state


# ---------------------------------------------------------------------------
# Trade execution
# ---------------------------------------------------------------------------

def _get_current_price(ticker: str) -> Optional[float]:
    """Fetch current price for a ticker."""
    try:
        resolved = data_layer.resolve_ticker(ticker)
        tk = yf.Ticker(resolved)
        hist = tk.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception as e:
        logger.error("Price fetch error for %s: %s", ticker, e)
    return None


def can_open_position(state: dict, plan: PositionPlan) -> tuple:
    """Check if we can open a new position. Returns (can_open, reason)."""
    # Max positions check
    if len(state["open_positions"]) >= MAX_OPEN_POSITIONS:
        return False, f"Max positions reached ({MAX_OPEN_POSITIONS})"

    # Duplicate check — no double entry on same ticker
    for pos in state["open_positions"]:
        if pos["ticker"] == plan.ticker:
            return False, f"Already have open position in {plan.ticker}"

    # Cash check
    cost = plan.shares * plan.entry_price
    if cost > state["cash"]:
        return False, f"Insufficient cash: need ${cost:,.2f}, have ${state['cash']:,.2f}"

    # Portfolio value risk check
    portfolio_value = compute_portfolio_value(state)
    max_risk = portfolio_value * (MAX_RISK_PER_TRADE_PCT / 100)
    trade_risk = plan.shares * plan.risk_per_share
    if trade_risk > max_risk * 1.5:  # Allow some slack
        return False, f"Trade risk ${trade_risk:,.2f} exceeds limit ${max_risk:,.2f}"

    return True, "OK"


def execute_entry(state: dict, alert: TradeAlert, plan: PositionPlan) -> Optional[dict]:
    """Execute a paper trade entry.

    Returns the position dict if successful, None otherwise.
    """
    can_open, reason = can_open_position(state, plan)
    if not can_open:
        logger.info("Paper trade blocked for %s: %s", plan.ticker, reason)
        return None

    cost = plan.shares * plan.entry_price

    position = {
        "ticker": plan.ticker,
        "direction": plan.direction,
        "entry_price": plan.entry_price,
        "entry_date": datetime.now().isoformat(),
        "shares": plan.shares,
        "cost_basis": round(cost, 2),
        "stop_loss": plan.stop_loss,
        "current_stop": plan.stop_loss,
        "highest_price": plan.entry_price,
        "lowest_price": plan.entry_price,
        "target_1": plan.target_1,
        "target_2": plan.target_2,
        "target_3": plan.target_3,
        "t1_hit": False,
        "t2_hit": False,
        "t3_hit": False,
        "signal_score": alert.signal_score,
        "triggered_signals": [s[0] for s in alert.triggered_signals],
        "risk_per_share": plan.risk_per_share,
    }

    state["cash"] -= cost
    state["open_positions"].append(position)
    save_state(state)

    logger.info(
        "PAPER ENTRY: %s %s %d shares @ $%.2f (stop: $%.2f, T1: $%.2f)",
        plan.direction, plan.ticker, plan.shares, plan.entry_price,
        plan.stop_loss, plan.target_1,
    )

    # Auto-create journal entry
    try:
        trade_journal.create_entry(
            ticker=plan.ticker,
            direction=plan.direction,
            entry_price=plan.entry_price,
            shares=plan.shares,
            stop_loss=plan.stop_loss,
            target_1=plan.target_1,
            signal_score=alert.signal_score,
            triggered_signals=[s[0] for s in alert.triggered_signals],
        )
    except Exception as e:
        logger.debug("Journal entry creation failed: %s", e)

    return position


def execute_exit(
    state: dict,
    ticker: str,
    exit_price: float,
    reason: str,
    partial_pct: float = 100.0,
) -> Optional[dict]:
    """Execute a paper trade exit (full or partial).

    partial_pct: percentage of position to close (100 = full exit).
    Returns the closed trade record.
    """
    for i, pos in enumerate(state["open_positions"]):
        if pos["ticker"] == ticker:
            if partial_pct < 100:
                exit_shares = max(1, int(pos["shares"] * partial_pct / 100))
            else:
                exit_shares = pos["shares"]

            # Compute P&L
            if pos["direction"] == "BUY":
                pnl = (exit_price - pos["entry_price"]) * exit_shares
            else:
                pnl = (pos["entry_price"] - exit_price) * exit_shares

            pnl_pct = (pnl / (pos["entry_price"] * exit_shares)) * 100
            proceeds = exit_shares * exit_price

            trade_record = {
                "ticker": ticker,
                "direction": pos["direction"],
                "entry_price": pos["entry_price"],
                "entry_date": pos["entry_date"],
                "exit_price": round(exit_price, 2),
                "exit_date": datetime.now().isoformat(),
                "exit_reason": reason,
                "shares": exit_shares,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "bars_held": _compute_bars_held(pos["entry_date"]),
                "signal_score": pos.get("signal_score", 0),
                "triggered_signals": pos.get("triggered_signals", []),
            }

            state["closed_trades"].append(trade_record)
            state["cash"] += proceeds

            # Update or remove position
            if exit_shares >= pos["shares"]:
                state["open_positions"].pop(i)
            else:
                pos["shares"] -= exit_shares

            save_state(state)

            logger.info(
                "PAPER EXIT: %s %s %d shares @ $%.2f — P&L: $%.2f (%.1f%%) [%s]",
                pos["direction"], ticker, exit_shares, exit_price,
                pnl, pnl_pct, reason,
            )

            # Auto-close journal entry (only on full exit)
            if exit_shares >= pos["shares"]:
                try:
                    trade_journal.close_entry(ticker, exit_price, reason, pnl, pnl_pct)
                except Exception as e:
                    logger.debug("Journal close failed: %s", e)

            return trade_record

    logger.warning("No open position found for %s", ticker)
    return None


def _compute_bars_held(entry_date_str: str) -> int:
    """Compute trading days held since entry."""
    try:
        entry = datetime.fromisoformat(entry_date_str)
        delta = datetime.now() - entry
        # Rough trading days estimate
        return max(1, int(delta.days * 5 / 7))
    except Exception:
        return 1


# ---------------------------------------------------------------------------
# Position management (trailing stops + target checks)
# ---------------------------------------------------------------------------

def update_positions(state: dict) -> List[dict]:
    """Update all open positions with current prices.

    Checks trailing stops, target hits, and executes exits.
    Returns list of actions taken.
    """
    actions = []

    for pos in list(state["open_positions"]):
        ticker = pos["ticker"]
        current_price = _get_current_price(ticker)
        if current_price is None:
            continue

        direction = pos["direction"]

        # Fetch ATR for trailing stop calculation
        atr = _fetch_atr(ticker)

        # Update high/low watermarks
        if direction == "BUY":
            if current_price > pos["highest_price"]:
                pos["highest_price"] = current_price
        else:
            if current_price < pos["lowest_price"]:
                pos["lowest_price"] = current_price

        # Compute trailing stop (hybrid: max of ATR and 3% trail)
        old_stop = pos["current_stop"]
        if direction == "BUY":
            atr_stop = pos["highest_price"] - 2.0 * atr if atr > 0 else 0
            pct_stop = pos["highest_price"] * 0.97
            new_stop = max(atr_stop, pct_stop)
            if new_stop > old_stop:
                pos["current_stop"] = round(new_stop, 2)
        else:
            atr_stop = pos["lowest_price"] + 2.0 * atr if atr > 0 else float("inf")
            pct_stop = pos["lowest_price"] * 1.03
            new_stop = min(atr_stop, pct_stop)
            if new_stop < old_stop:
                pos["current_stop"] = round(new_stop, 2)

        # Check stop hit
        stop_hit = False
        if direction == "BUY" and current_price <= pos["current_stop"]:
            stop_hit = True
        elif direction == "SELL" and current_price >= pos["current_stop"]:
            stop_hit = True

        if stop_hit:
            trade = execute_exit(state, ticker, pos["current_stop"], "trailing_stop")
            if trade:
                actions.append({"action": "STOP_EXIT", "trade": trade})
            continue

        # Check targets — partial exits
        if direction == "BUY":
            if not pos["t1_hit"] and pos.get("target_1") and current_price >= pos["target_1"]:
                pos["t1_hit"] = True
                trade = execute_exit(state, ticker, current_price, "target_1", partial_pct=33)
                if trade:
                    actions.append({"action": "T1_PARTIAL", "trade": trade})
            if not pos["t2_hit"] and pos.get("target_2") and current_price >= pos["target_2"]:
                pos["t2_hit"] = True
                trade = execute_exit(state, ticker, current_price, "target_2", partial_pct=50)
                if trade:
                    actions.append({"action": "T2_PARTIAL", "trade": trade})
            if not pos["t3_hit"] and pos.get("target_3") and current_price >= pos["target_3"]:
                pos["t3_hit"] = True
                trade = execute_exit(state, ticker, current_price, "target_3", partial_pct=100)
                if trade:
                    actions.append({"action": "T3_FULL_EXIT", "trade": trade})
        else:
            if not pos["t1_hit"] and pos.get("target_1") and current_price <= pos["target_1"]:
                pos["t1_hit"] = True
                trade = execute_exit(state, ticker, current_price, "target_1", partial_pct=33)
                if trade:
                    actions.append({"action": "T1_PARTIAL", "trade": trade})
            if not pos["t2_hit"] and pos.get("target_2") and current_price <= pos["target_2"]:
                pos["t2_hit"] = True
                trade = execute_exit(state, ticker, current_price, "target_2", partial_pct=50)
                if trade:
                    actions.append({"action": "T2_PARTIAL", "trade": trade})
            if not pos["t3_hit"] and pos.get("target_3") and current_price <= pos["target_3"]:
                pos["t3_hit"] = True
                trade = execute_exit(state, ticker, current_price, "target_3", partial_pct=100)
                if trade:
                    actions.append({"action": "T3_FULL_EXIT", "trade": trade})

        # Log stop movement
        if pos["current_stop"] != old_stop:
            actions.append({
                "action": "STOP_RATCHET",
                "ticker": ticker,
                "old_stop": old_stop,
                "new_stop": pos["current_stop"],
                "price": current_price,
            })

    save_state(state)
    return actions


def _fetch_atr(ticker: str, period: str = "1mo") -> float:
    """Fetch current ATR for a ticker."""
    try:
        resolved = data_layer.resolve_ticker(ticker)
        tk = yf.Ticker(resolved)
        hist = tk.history(period=period, interval="1d")
        if len(hist) < 5:
            return 0.0
        tr_vals = []
        for i in range(1, len(hist)):
            hl = hist["High"].iloc[i] - hist["Low"].iloc[i]
            hc = abs(hist["High"].iloc[i] - hist["Close"].iloc[i - 1])
            lc = abs(hist["Low"].iloc[i] - hist["Close"].iloc[i - 1])
            tr_vals.append(max(hl, hc, lc))
        return float(np.mean(tr_vals)) if tr_vals else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Portfolio valuation
# ---------------------------------------------------------------------------

def compute_portfolio_value(state: dict) -> float:
    """Compute total portfolio value (cash + open positions at current prices)."""
    total = state["cash"]
    for pos in state["open_positions"]:
        price = _get_current_price(pos["ticker"])
        if price is None:
            price = pos["entry_price"]
        total += price * pos["shares"]
    return total


def compute_portfolio_value_fast(state: dict) -> float:
    """Compute portfolio value using entry prices (no API calls)."""
    total = state["cash"]
    for pos in state["open_positions"]:
        total += pos["entry_price"] * pos["shares"]
    return total


# ---------------------------------------------------------------------------
# Performance analytics
# ---------------------------------------------------------------------------

def compute_performance(state: dict) -> dict:
    """Compute comprehensive performance metrics from trade history."""
    trades = state["closed_trades"]
    starting = state["starting_capital"]

    if not trades:
        return {
            "total_trades": 0,
            "message": "No closed trades yet",
        }

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    win_count = len(wins)
    loss_count = len(losses)
    total_trades = len(trades)

    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
    avg_win = (sum(wins) / win_count) if win_count > 0 else 0
    avg_loss = (sum(abs(l) for l in losses) / loss_count) if loss_count > 0 else 0

    profit_factor = (sum(wins) / sum(abs(l) for l in losses)) if losses else float("inf")

    # Expectancy
    win_rate_dec = win_count / total_trades if total_trades > 0 else 0
    loss_rate_dec = loss_count / total_trades if total_trades > 0 else 0
    expectancy = (win_rate_dec * avg_win) - (loss_rate_dec * avg_loss)

    # Max drawdown from equity curve
    equity = [starting]
    for t in trades:
        equity.append(equity[-1] + t["pnl"])
    peak = equity[0]
    max_dd = 0
    max_dd_pct = 0
    for e in equity:
        if e > peak:
            peak = e
        dd = peak - e
        dd_pct = (dd / peak * 100) if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd_pct

    # Average bars held
    bars = [t.get("bars_held", 0) for t in trades]
    avg_bars = sum(bars) / len(bars) if bars else 0

    # Return on capital
    current_value = compute_portfolio_value_fast(state)
    total_return_pct = ((current_value - starting) / starting * 100)

    # Win/loss streaks
    max_win_streak = 0
    max_loss_streak = 0
    current_streak = 0
    streak_type = None
    for p in pnls:
        if p > 0:
            if streak_type == "win":
                current_streak += 1
            else:
                current_streak = 1
                streak_type = "win"
            max_win_streak = max(max_win_streak, current_streak)
        else:
            if streak_type == "loss":
                current_streak += 1
            else:
                current_streak = 1
                streak_type = "loss"
            max_loss_streak = max(max_loss_streak, current_streak)

    # Best/worst by signal
    signal_pnl = {}
    for t in trades:
        for sig in t.get("triggered_signals", []):
            if sig not in signal_pnl:
                signal_pnl[sig] = []
            signal_pnl[sig].append(t["pnl"])

    best_signal = None
    worst_signal = None
    if signal_pnl:
        signal_avg = {s: sum(v) / len(v) for s, v in signal_pnl.items() if len(v) >= 2}
        if signal_avg:
            best_signal = max(signal_avg, key=signal_avg.get)
            worst_signal = min(signal_avg, key=signal_avg.get)

    return {
        "total_trades": total_trades,
        "wins": win_count,
        "losses": loss_count,
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
        "expectancy": round(expectancy, 2),
        "max_drawdown": round(max_dd, 2),
        "max_drawdown_pct": round(max_dd_pct, 1),
        "avg_bars_held": round(avg_bars, 1),
        "total_return_pct": round(total_return_pct, 2),
        "max_win": round(max(pnls), 2) if pnls else 0,
        "max_loss": round(min(pnls), 2) if pnls else 0,
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "best_signal": best_signal,
        "worst_signal": worst_signal,
        "equity_curve": [round(e, 2) for e in equity],
    }


# ---------------------------------------------------------------------------
# Display functions
# ---------------------------------------------------------------------------

def print_status(state: dict) -> None:
    """Print current paper portfolio status."""
    portfolio_val = compute_portfolio_value_fast(state)
    starting = state["starting_capital"]
    total_return = ((portfolio_val - starting) / starting * 100)
    open_pos = state["open_positions"]
    closed = state["closed_trades"]
    cur = get_currency_symbol()

    print(f"\n{'=' * 72}")
    print(f"  PAPER TRADING PORTFOLIO")
    print(f"{'=' * 72}")
    print(f"\n  Starting Capital:  {cur}{starting:>12,.2f}")
    print(f"  Cash:              {cur}{state['cash']:>12,.2f}")
    print(f"  Portfolio Value:   {cur}{portfolio_val:>12,.2f}  ({total_return:+.1f}%)")
    print(f"  Open Positions:    {len(open_pos)}")
    print(f"  Closed Trades:     {len(closed)}")

    if open_pos:
        print(f"\n  OPEN POSITIONS")
        print(f"  {'─' * 66}")
        print(f"  {'Ticker':<7} {'Dir':>4} {'Shares':>6} {'Entry':>8} {'Stop':>8}"
              f" {'T1':>3} {'T2':>3} {'Signals'}")
        print(f"  {'─' * 66}")
        for p in open_pos:
            t1 = "Y" if p.get("t1_hit") else "-"
            t2 = "Y" if p.get("t2_hit") else "-"
            sigs = ", ".join(p.get("triggered_signals", [])[:3])
            print(f"  {p['ticker']:<7} {p['direction']:>4} {p['shares']:>6}"
                  f" {cur}{p['entry_price']:>7,.2f} {cur}{p['current_stop']:>7,.2f}"
                  f" {t1:>3} {t2:>3}  {sigs}")

    print(f"\n  Last Updated: {state.get('last_updated', 'N/A')}")
    print(f"{'=' * 72}\n")


def print_history(state: dict, last_n: int = 20) -> None:
    """Print recent trade history."""
    trades = state["closed_trades"]

    print(f"\n{'=' * 72}")
    print(f"  PAPER TRADE HISTORY (last {min(last_n, len(trades))} of {len(trades)})")
    print(f"{'=' * 72}")

    if not trades:
        print(f"\n  No trades yet. Run the agent with --paper to start trading.")
        print(f"{'=' * 72}\n")
        return

    print(f"\n  {'Date':<12} {'Ticker':<7} {'Dir':>4} {'Entry':>8} {'Exit':>8}"
          f" {'P&L':>9} {'%':>6} {'Reason':<15}")
    print(f"  {'─' * 68}")

    for t in trades[-last_n:]:
        date_str = t["exit_date"][:10]
        pnl_str = f"${t['pnl']:>8,.2f}"
        print(f"  {date_str:<12} {t['ticker']:<7} {t['direction']:>4}"
              f" ${t['entry_price']:>7,.2f} ${t['exit_price']:>7,.2f}"
              f" {pnl_str} {t['pnl_pct']:>5.1f}% {t['exit_reason']:<15}")

    total_pnl = sum(t["pnl"] for t in trades)
    print(f"  {'─' * 68}")
    print(f"  {'Total P&L:':>46} ${total_pnl:>8,.2f}")
    print(f"{'=' * 72}\n")


def print_performance(state: dict) -> None:
    """Print comprehensive performance report."""
    perf = compute_performance(state)

    print(f"\n{'=' * 72}")
    print(f"  PAPER TRADING PERFORMANCE REPORT")
    print(f"{'=' * 72}")

    if perf["total_trades"] == 0:
        print(f"\n  {perf.get('message', 'No trades yet.')}")
        print(f"{'=' * 72}\n")
        return

    print(f"\n  SUMMARY")
    print(f"  {'─' * 40}")
    print(f"  Total Trades:        {perf['total_trades']}")
    print(f"  Win / Loss:          {perf['wins']} / {perf['losses']}")
    print(f"  Win Rate:            {perf['win_rate']:.1f}%")
    print(f"  Total P&L:           ${perf['total_pnl']:>10,.2f}")
    print(f"  Total Return:        {perf['total_return_pct']:>+.2f}%")

    print(f"\n  RISK METRICS")
    print(f"  {'─' * 40}")
    print(f"  Profit Factor:       {perf['profit_factor']}")
    print(f"  Expectancy:          ${perf['expectancy']:>10,.2f} / trade")
    print(f"  Avg Win:             ${perf['avg_win']:>10,.2f}")
    print(f"  Avg Loss:            ${perf['avg_loss']:>10,.2f}")
    print(f"  Max Win:             ${perf['max_win']:>10,.2f}")
    print(f"  Max Loss:            ${perf['max_loss']:>10,.2f}")
    print(f"  Max Drawdown:        ${perf['max_drawdown']:>10,.2f} ({perf['max_drawdown_pct']:.1f}%)")

    print(f"\n  TRADE STATS")
    print(f"  {'─' * 40}")
    print(f"  Avg Bars Held:       {perf['avg_bars_held']:.1f}")
    print(f"  Max Win Streak:      {perf['max_win_streak']}")
    print(f"  Max Loss Streak:     {perf['max_loss_streak']}")
    if perf["best_signal"]:
        print(f"  Best Signal:         {perf['best_signal']}")
    if perf["worst_signal"]:
        print(f"  Worst Signal:        {perf['worst_signal']}")

    # Equity curve (text sparkline)
    eq = perf.get("equity_curve", [])
    if len(eq) > 2:
        min_eq = min(eq)
        max_eq = max(eq)
        span = max_eq - min_eq if max_eq != min_eq else 1
        bars = "▁▂▃▄▅▆▇█"
        sparkline = ""
        step = max(1, len(eq) // 50)
        for i in range(0, len(eq), step):
            idx = int((eq[i] - min_eq) / span * (len(bars) - 1))
            sparkline += bars[idx]
        print(f"\n  EQUITY CURVE")
        print(f"  {'─' * 40}")
        print(f"  {sparkline}")
        print(f"  ${min_eq:,.0f} {'─' * 20} ${max_eq:,.0f}")

    print(f"\n{'=' * 72}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Paper Trading Engine")
    parser.add_argument("--status", action="store_true", help="Show portfolio status")
    parser.add_argument("--history", action="store_true", help="Show trade history")
    parser.add_argument("--performance", action="store_true", help="Show performance report")
    parser.add_argument("--reset", action="store_true", help="Reset paper portfolio")
    parser.add_argument("--capital", type=float, default=DEFAULT_CAPITAL,
                        help="Starting capital for reset")
    parser.add_argument("--update", action="store_true",
                        help="Update trailing stops and check exits")
    parser.add_argument("--last", type=int, default=20,
                        help="Number of recent trades to show in history")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    for lib in ("urllib3", "yfinance", "peewee"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    if args.reset:
        reset_state(args.capital)
        print(f"Paper portfolio reset with ${args.capital:,.2f}")
        return

    state = load_state()

    if args.update:
        actions = update_positions(state)
        if actions:
            for a in actions:
                if "trade" in a:
                    t = a["trade"]
                    print(f"  {a['action']}: {t['ticker']} P&L=${t['pnl']:,.2f}")
                else:
                    print(f"  {a['action']}: {a['ticker']} stop {a['old_stop']:.2f} → {a['new_stop']:.2f}")
        else:
            print("  No position updates needed.")
        return

    if args.performance:
        print_performance(state)
    elif args.history:
        print_history(state, args.last)
    else:
        print_status(state)


if __name__ == "__main__":
    main()
