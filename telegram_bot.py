#!/usr/bin/env python3
"""
Telegram Bot — two-way communication with the Stock Agent from your phone.

Commands:
    /buy        — BUY opportunities from full watchlist scan
    /sell       — Which positions should I sell?
    /check TSLA — Analyze any ticker (held or new)
    /scan       — Quick scan (holdings + top buy opportunities)
    /status     — Portfolio summary + P&L
    /positions  — List all open positions
    /risk       — Portfolio risk summary
    /regime     — Current market regime
    /briefing   — Morning briefing
    /volume     — Unusual volume spikes
    /upgrades   — Analyst upgrades/downgrades
    /alerts     — Active price alerts
    /help       — List commands

    Buy alerts also push automatically when the scan engine fires a signal.

Setup:
    1. Message @BotFather on Telegram, send /newbot
    2. Copy the token to .env as TELEGRAM_BOT_TOKEN
    3. Start a chat with your new bot
    4. Run: python3 telegram_bot.py --get-chat-id
    5. Copy the chat ID to .env as TELEGRAM_CHAT_ID
    6. Run: python3 telegram_bot.py

Usage:
    python3 telegram_bot.py              # Start the bot
    python3 telegram_bot.py --get-chat-id # Find your chat ID
"""

import argparse
import base64
import json
import logging
import os
import queue as _queue
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

PORTFOLIO_FILE = Path("portfolio.json")
OFFSET_FILE    = Path("logs/bot_offset.txt")


# ---------------------------------------------------------------------------
# Portfolio persistence helpers
# ---------------------------------------------------------------------------

def _load_portfolio() -> dict:
    try:
        return json.loads(PORTFOLIO_FILE.read_text())
    except Exception as e:
        logger.error("Failed to load portfolio.json: %s", e)
        return {}


def _save_portfolio(portfolio: dict) -> None:
    PORTFOLIO_FILE.write_text(json.dumps(portfolio, indent=2))


def _commit_portfolio_to_github(portfolio: dict) -> bool:
    """Commit updated portfolio.json back to the repo so changes survive restarts."""
    token = os.getenv("GITHUB_TOKEN", "")
    repo  = os.getenv("GITHUB_REPOSITORY", "")
    if not token or not repo:
        logger.debug("GITHUB_TOKEN/GITHUB_REPOSITORY not set — portfolio won't persist across restarts")
        return False
    try:
        content = json.dumps(portfolio, indent=2)
        encoded = base64.b64encode(content.encode()).decode()
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }
        url  = f"https://api.github.com/repos/{repo}/contents/portfolio.json"
        resp = requests.get(url, headers=headers, timeout=15)
        sha  = resp.json().get("sha", "")
        payload = {
            "message": "bot: record trade update",
            "content": encoded,
            "sha": sha,
            "branch": "main",
        }
        resp = requests.put(url, json=payload, headers=headers, timeout=15)
        if resp.status_code in (200, 201):
            logger.info("Portfolio committed to GitHub")
            return True
        logger.warning("GitHub portfolio commit failed (%d): %s", resp.status_code, resp.text[:200])
        return False
    except Exception as e:
        logger.error("GitHub commit error: %s", e)
        return False


# ---------------------------------------------------------------------------
# Trade feedback engine — principles from the world's best investors
# ---------------------------------------------------------------------------

def _get_trade_feedback(ticker: str, direction: str, shares: int, price: float, portfolio: dict) -> str:
    """Generate investor-quality feedback on a manually recorded trade."""
    lines = []
    try:
        import data_layer, technical_analysis, fundamental_analysis, signal_engine, position_sizing

        df   = data_layer.fetch_daily_ohlcv(ticker)
        if df.empty:
            return "Could not fetch data for %s to generate feedback." % ticker

        tech = technical_analysis.analyze(ticker, df)
        # Look up actual sector from portfolio instead of hardcoding "Technology"
        sector = "Technology"
        for h in portfolio.get("holdings", []):
            if h.get("ticker") == ticker:
                sector = h.get("sector", "Technology")
                break
        fund = fundamental_analysis.analyze(ticker, sector)

        current = tech.current_price
        diff_pct = (current - price) / price * 100 if price else 0
        diff_str = f"+{diff_pct:.1f}%" if diff_pct >= 0 else f"{diff_pct:.1f}%"

        # Position sizing check
        total_val  = float(portfolio.get("total_portfolio_value", 1) or 1)
        pos_val    = shares * price
        pos_pct    = pos_val / total_val * 100

        lines.append("Trade Feedback: %s %s" % (direction, ticker))
        lines.append("─" * 40)
        lines.append("Your price: $%.2f  |  Current: $%.2f  (%s)" % (price, current, diff_str))
        lines.append("Position: %d × $%.2f = $%s (%.1f%% of portfolio)" % (
            shares, price, f"{pos_val:,.0f}", pos_pct))
        lines.append("")

        # Technical timing assessment
        rsi_note = ""
        if tech.rsi > 75:
            rsi_note = "RSI %.0f — overbought territory. Lynch says: strong momentum, but check earnings growth justifies premium." % tech.rsi
        elif tech.rsi > 60:
            rsi_note = "RSI %.0f — healthy uptrend. Good timing." % tech.rsi
        elif tech.rsi < 35:
            rsi_note = "RSI %.0f — oversold. Buffett principle: be greedy when others are fearful." % tech.rsi
        else:
            rsi_note = "RSI %.0f — neutral zone. Watch for directional confirmation." % tech.rsi

        above_ema21 = current > tech.ema21
        above_200   = tech.price_above_200sma
        trend_note  = "Price is %s EMA21 and %s 200-SMA. " % (
            "above" if above_ema21 else "below",
            "above" if above_200 else "below")
        if direction == "BUY":
            if above_200 and above_ema21:
                trend_note += "Trend alignment is bullish — classic Lynch 'buy the trend' setup."
            elif not above_200:
                trend_note += "Caution: below 200-SMA. Buffett would want to see value, not just price action."
        else:
            if not above_ema21:
                trend_note += "Selling below EMA21 is technically sound — momentum has shifted."

        lines.append("Technical Timing:")
        lines.append("  " + rsi_note)
        lines.append("  " + trend_note)
        lines.append("")

        # Fundamental quality (Buffett / Greenblatt lens)
        fund_score = fund.fundamental_score
        lines.append("Fundamental Quality: %d/15" % fund_score)
        if fund_score >= 10:
            lines.append("  Strong quality business. Buffett principle: wonderful company at fair price.")
        elif fund_score >= 6:
            lines.append("  Decent fundamentals. Monitor for quality degradation.")
        else:
            lines.append("  Weak fundamentals. Greenblatt warns: only buy quality + cheap, not just cheap.")

        if fund.pe_ratio and fund.pe_ratio > 0:
            pe_note = "P/E: %.1f — " % fund.pe_ratio
            if fund.pe_ratio < 15:
                pe_note += "value zone (Buffett: margin of safety present)."
            elif fund.pe_ratio < 30:
                pe_note += "fairly valued for a growth stock (Lynch: PEG ratio matters more than raw P/E)."
            else:
                pe_note += "premium valuation. Ensure growth rate justifies the multiple (Lynch PEG rule)."
            lines.append("  " + pe_note)
        lines.append("")

        # Risk management check
        lines.append("Risk Management (Dalio principle):")
        if pos_pct > 15:
            lines.append("  ⚠ Position is %.1f%% of portfolio — concentrated. Dalio: diversify across uncorrelated assets." % pos_pct)
        elif pos_pct > 10:
            lines.append("  Position at %.1f%% — at max recommended size. No room to add." % pos_pct)
        else:
            lines.append("  Position size %.1f%% — within healthy limits." % pos_pct)

        # Suggested stop
        if direction == "BUY":
            atr = getattr(tech, "atr", None) or (current * 0.02)
            suggested_stop = round(current - 2 * atr, 2)
            lines.append("  Suggested stop: $%.2f (2×ATR below entry)" % suggested_stop)
        lines.append("")

        # Overall grade
        grade_score = 0
        if direction == "BUY":
            if above_200: grade_score += 2
            if above_ema21: grade_score += 1
            if 45 < tech.rsi < 70: grade_score += 2
            if fund_score >= 8: grade_score += 2
            if pos_pct <= 10: grade_score += 1
            if diff_pct >= 0: grade_score += 1
        else:  # SELL
            if not above_ema21: grade_score += 2
            if tech.rsi > 70: grade_score += 2
            if diff_pct > 0: grade_score += 3
            grade_score += 2  # Taking profits is always good discipline

        if grade_score >= 8:
            grade = "A — Excellent execution. Best investors would approve."
        elif grade_score >= 6:
            grade = "B — Good trade. Solid setup with manageable risk."
        elif grade_score >= 4:
            grade = "C — Average entry. Review timing and fundamentals."
        else:
            grade = "D — Below average setup. Consider waiting for better confirmation."

        lines.append("Trade Grade: %s" % grade)

    except Exception as e:
        lines.append("(Feedback analysis error: %s)" % e)

    return "\n".join(lines)

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
API_BASE = "https://api.telegram.org/bot%s" % TOKEN


def send_message(text, chat_id=None, parse_mode=None):
    """Send a message to the Telegram chat."""
    cid = chat_id or CHAT_ID
    if not TOKEN or not cid:
        return False

    # Telegram max message length is 4096
    chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
    for chunk in chunks:
        payload = {"chat_id": cid, "text": chunk}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        try:
            resp = requests.post(API_BASE + "/sendMessage", json=payload, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            logger.error("Telegram send failed: %s", e)
            return False
    return True


def get_updates(offset=None, timeout=30):
    """Long-poll for new messages."""
    params = {"timeout": timeout}
    if offset:
        params["offset"] = offset
    try:
        resp = requests.get(API_BASE + "/getUpdates", params=params, timeout=timeout + 5)
        resp.raise_for_status()
        return resp.json().get("result", [])
    except requests.exceptions.HTTPError:
        raise  # Let run_bot() handle HTTP errors (409 Conflict, etc.)
    except Exception as e:
        logger.error("getUpdates failed: %s", e)
        return []


def get_chat_id():
    """Get your chat ID by reading the first message sent to the bot."""
    print("Send any message to your bot on Telegram, then press Enter here...")
    input()
    updates = get_updates(timeout=5)
    if not updates:
        print("No messages found. Make sure you sent a message to your bot.")
        return
    for u in updates:
        msg = u.get("message", {})
        chat = msg.get("chat", {})
        print("\nFound chat:")
        print("  Chat ID: %s" % chat.get("id"))
        print("  Name: %s %s" % (chat.get("first_name", ""), chat.get("last_name", "")))
        print("\nAdd this to your .env file:")
        print("  TELEGRAM_CHAT_ID=%s" % chat.get("id"))


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_status(args=""):
    """Portfolio summary."""
    try:
        portfolio = _load_portfolio()
        total = portfolio.get("total_portfolio_value", 0)
        cash = portfolio.get("available_cash", 0)
        holdings = portfolio.get("holdings", [])
        unrealized = sum(float(h.get("unrealized_pnl", 0) or 0) for h in holdings)
        realized = float(portfolio.get("realized_pnl_ytd", 0) or 0)

        lines = ["Portfolio Summary"]
        lines.append("=" * 30)
        lines.append("Net Liq: $%s" % f"{total:,.0f}")
        lines.append("Cash: $%s" % f"{cash:,.0f}")
        lines.append("Positions: %d" % len(holdings))
        lines.append("Unrealized: $%s" % f"{unrealized:+,.0f}")
        lines.append("Realized YTD: $%s" % f"{realized:+,.0f}")
        lines.append("Total P&L: $%s" % f"{unrealized + realized:+,.0f}")

        # Top winners/losers
        sorted_h = sorted(holdings, key=lambda h: float(h.get("unrealized_pnl", 0) or 0))
        if sorted_h:
            worst = sorted_h[0]
            best = sorted_h[-1]
            lines.append("")
            lines.append("Best: %s $%s (%.1f%%)" % (
                best["ticker"], f"{float(best.get('unrealized_pnl', 0)):+,.0f}",
                float(best.get("pnl_pct", 0))))
            lines.append("Worst: %s $%s (%.1f%%)" % (
                worst["ticker"], f"{float(worst.get('unrealized_pnl', 0)):+,.0f}",
                float(worst.get("pnl_pct", 0))))

        return "\n".join(lines)
    except Exception as e:
        return "Error loading portfolio: %s" % e


def cmd_positions(args=""):
    """List all open positions."""
    try:
        portfolio = _load_portfolio()
        holdings = portfolio.get("holdings", [])
        if not holdings:
            return "No open positions."

        lines = ["Open Positions (%d)" % len(holdings)]
        lines.append("-" * 35)
        for h in sorted(holdings, key=lambda x: x.get("ticker", "")):
            pnl = float(h.get("unrealized_pnl", 0) or 0)
            pct = float(h.get("pnl_pct", 0) or 0)
            pnl_str = "${:+,.0f}".format(pnl)
            lines.append("%s: %d @ $%.2f | %s (%.1f%%)" % (
                h["ticker"], h.get("shares", 0),
                float(h.get("current_price", 0)), pnl_str, pct))

        return "\n".join(lines)
    except Exception as e:
        return "Error: %s" % e


def cmd_sell(args=""):
    """Check which positions should be sold."""
    try:
        import position_monitor
        results = position_monitor.check_all_positions()
        return position_monitor.format_report(results)
    except Exception as e:
        return "Error running position monitor: %s" % e


def cmd_check(args=""):
    """Check a single ticker."""
    ticker = args.strip().upper() if args else ""
    if not ticker:
        try:
            portfolio = _load_portfolio()
            tickers = sorted(h["ticker"] for h in portfolio.get("holdings", []))
            return "Usage: /check TSLA\n\nYour holdings:\n%s\n\nOr just type a ticker name (e.g. NVDA)" % " ".join(tickers)
        except Exception:
            return "Usage: /check TSLA\n\nOr just type a ticker name (e.g. NVDA)"

    try:
        import position_monitor
        results = position_monitor.check_all_positions(ticker_filter=ticker)
        if results:
            return position_monitor.format_report(results)

        # Not in portfolio — run a scan
        import data_layer
        import technical_analysis
        import fundamental_analysis
        import signal_engine

        df = data_layer.fetch_daily_ohlcv(ticker)
        if df.empty:
            return "No data for %s" % ticker

        tech = technical_analysis.analyze(ticker, df)
        fund = fundamental_analysis.analyze(ticker, "Technology")
        alert = signal_engine.evaluate(tech, fund)

        lines = ["%s Analysis" % ticker]
        lines.append("-" * 30)
        lines.append("Price: $%.2f" % tech.current_price)
        lines.append("RSI: %.1f" % tech.rsi)
        lines.append("MACD: %.2f" % tech.macd_value)
        lines.append("Above 200 SMA: %s" % tech.price_above_200sma)
        lines.append("Fundamental: %d/6" % fund.fundamental_score)

        if alert:
            lines.append("")
            lines.append("SIGNAL: %s (score %d)" % (alert.direction, alert.signal_score))
            signals = ", ".join(s[0] for s in alert.triggered_signals)
            lines.append("Signals: %s" % signals)
        else:
            lines.append("\nNo signal (below threshold)")

        if tech.pattern_details:
            lines.append("\nPatterns: %s" % ", ".join(tech.pattern_details[:3]))

        return "\n".join(lines)
    except Exception as e:
        return "Error checking %s: %s" % (ticker, e)


def cmd_scan(args=""):
    """Quick scan — holdings signals + top buy opportunities from full watchlist."""
    try:
        import data_layer
        import signal_engine
        import technical_analysis
        import fundamental_analysis

        watchlist = data_layer.load_watchlist()
        portfolio = _load_portfolio()
        held_tickers = set(h["ticker"] for h in portfolio.get("holdings", []))

        hold_signals = []
        buy_signals = []

        for entry in watchlist:
            ticker = entry["ticker"]
            try:
                df = data_layer.fetch_daily_ohlcv(ticker)
                if df.empty:
                    continue
                tech = technical_analysis.analyze(ticker, df)
                fund = fundamental_analysis.analyze(ticker, entry.get("sector", ""))
                alert = signal_engine.evaluate(tech, fund)
                if alert:
                    line = "%s %s (score %d)" % (alert.direction, ticker, alert.signal_score)
                    if ticker in held_tickers:
                        hold_signals.append(line)
                    elif alert.direction == "BUY":
                        buy_signals.append((alert.signal_score, line))
            except Exception:
                continue

        lines = []
        if hold_signals:
            lines.append("Holdings Signals:")
            lines.extend(hold_signals)
        if buy_signals:
            if lines:
                lines.append("")
            lines.append("New Buy Opportunities:")
            for _, l in sorted(buy_signals, reverse=True)[:5]:
                lines.append(l)
        if not lines:
            return "No signals right now. Use /buy for a full watchlist BUY scan."
        return "\n".join(lines)
    except Exception as e:
        return "Scan error: %s" % e


def cmd_buy(args=""):
    """Scan full watchlist for BUY opportunities (new positions to enter)."""
    try:
        import data_layer
        import signal_engine
        import technical_analysis
        import fundamental_analysis
        import position_sizing

        watchlist = data_layer.load_watchlist()
        portfolio = _load_portfolio()
        held_tickers = set(h["ticker"] for h in portfolio.get("holdings", []))

        # Optional: filter by sector or ticker prefix
        filter_text = args.strip().upper() if args.strip() else ""

        candidates = []
        for entry in watchlist:
            ticker = entry["ticker"]
            if filter_text and filter_text not in ticker and filter_text not in entry.get("sector", "").upper():
                continue
            try:
                df = data_layer.fetch_daily_ohlcv(ticker)
                if df.empty:
                    continue
                tech = technical_analysis.analyze(ticker, df)
                fund = fundamental_analysis.analyze(ticker, entry.get("sector", ""))
                alert = signal_engine.evaluate(tech, fund)
                if alert and alert.direction == "BUY":
                    plan = position_sizing.compute(alert)
                    in_portfolio = " [HELD]" if ticker in held_tickers else ""
                    if plan:
                        candidates.append((
                            alert.signal_score,
                            ticker,
                            tech.current_price,
                            plan.stop_loss,
                            plan.target_1,
                            plan.target_2,
                            alert.signal_score,
                            in_portfolio,
                        ))
                    else:
                        candidates.append((alert.signal_score, ticker, tech.current_price,
                                           0, 0, 0, alert.signal_score, in_portfolio))
            except Exception:
                continue

        if not candidates:
            return "No BUY signals on the watchlist right now."

        candidates.sort(reverse=True)
        lines = ["BUY Opportunities (%d found)" % len(candidates)]
        lines.append("-" * 38)
        for score, ticker, price, stop, t1, t2, _, held in candidates[:8]:
            lines.append("%s%s  $%.2f  score:%d" % (ticker, held, price, score))
            if stop:
                lines.append("  Stop: $%.2f | T1: $%.2f | T2: $%.2f" % (stop, t1, t2))
        lines.append("")
        lines.append("Use /check TICKER for full details.")
        return "\n".join(lines)
    except Exception as e:
        return "Buy scan error: %s" % e


def cmd_risk(args=""):
    """Portfolio risk summary."""
    try:
        import risk_monitor
        report = risk_monitor.analyze_risk()
        lines = ["Risk Summary"]
        lines.append("-" * 30)
        vol = report.get("portfolio_volatility_pct", 0) or report.get("portfolio_volatility", 0)
        lines.append("Volatility: %.1f%%" % float(vol or 0))
        cash_pct = report.get("cash_pct", 0)
        lines.append("Cash: %.1f%%" % float(cash_pct or 0))
        total = report.get("total_value", 0)
        unrealized = report.get("total_unrealized_pnl", 0)
        lines.append("Portfolio: $%s" % "{:,.0f}".format(float(total or 0)))
        lines.append("Unrealized: $%s" % "{:+,.0f}".format(float(unrealized or 0)))

        sector = report.get("sector_exposure", {})
        if sector:
            lines.append("\nSector Weights:")
            for s, pct in sorted(sector.items(), key=lambda x: x[1], reverse=True)[:5]:
                lines.append("  %s: %.1f%%" % (s, float(pct or 0)))

        high_corr = report.get("high_correlation_pairs", [])
        if high_corr:
            lines.append("\nHigh Correlations:")
            for pair in high_corr[:3]:
                lines.append("  %s / %s: %.2f" % (pair[0], pair[1], pair[2]))

        warnings = report.get("warnings", [])
        if warnings:
            lines.append("\nWarnings:")
            for w in warnings[:3]:
                lines.append("  ! %s" % w)

        return "\n".join(lines)
    except Exception as e:
        return "Risk error: %s" % e


def cmd_regime(args=""):
    """Current market regime."""
    try:
        regime_file = Path("logs/market_regime.json")
        if regime_file.exists():
            with open(regime_file) as _rf:
                data = json.load(_rf)
            if isinstance(data, list) and data:
                latest = data[-1]
                return "Market Regime: %s\nConfidence: %d%%\nScore: %d\nUpdated: %s" % (
                    latest.get("regime", "N/A"),
                    latest.get("confidence", 0),
                    latest.get("raw_score", 0),
                    str(latest.get("timestamp", ""))[:16],
                )
        # Run fresh
        import market_regime
        result = market_regime.detect_regime()
        return "Market Regime: %s\nConfidence: %d%%\nDescription: %s" % (
            result["regime"], result["confidence"],
            result.get("params", {}).get("description", ""))
    except Exception as e:
        return "Regime error: %s" % e


def cmd_briefing(args=""):
    """Generate morning briefing."""
    try:
        import daily_briefing
        briefing = daily_briefing.generate_briefing()
        return daily_briefing.format_briefing(briefing)
    except Exception as e:
        return "Briefing error: %s" % e


def cmd_alerts(args=""):
    """Show active price alerts."""
    try:
        import price_alerts
        alerts = price_alerts.get_active_alerts()
        if not alerts:
            return "No active price alerts set.\n\nTo add one:\n/alert TSLA above 400"

        lines = ["Active Price Alerts (%d)" % len(alerts)]
        lines.append("-" * 30)
        for a in alerts:
            note = a.get("note", "")
            note_str = " (%s)" % note if note else ""
            lines.append("%s %s $%.2f%s" % (
                a.get("ticker", ""), a.get("condition", ""),
                float(a.get("price", 0)), note_str))
        return "\n".join(lines)
    except Exception as e:
        return "Alerts error: %s" % e


def cmd_earnings(args=""):
    """Upcoming earnings for your holdings."""
    try:
        import earnings_calendar
        earnings = earnings_calendar.get_portfolio_earnings()
        return earnings_calendar.format_earnings_report(earnings)
    except Exception as e:
        return "Earnings error: %s" % e


def cmd_news(args=""):
    """Latest significant news for your holdings."""
    try:
        import news_monitor
        hours = 24
        try:
            hours = int(args.strip()) if args.strip() else 24
        except Exception:
            pass
        items = news_monitor.fetch_portfolio_news(max_age_hours=hours)
        significant = news_monitor.filter_significant(items)
        if not significant:
            all_items = items[:10]
            if not all_items:
                return "No news in the last %d hours." % hours
            return news_monitor.format_news_report(all_items)
        return news_monitor.format_news_report(significant)
    except Exception as e:
        return "News error: %s" % e


def cmd_dca(args=""):
    """DCA suggestions for long-term positions that are down."""
    try:
        import dca_advisor
        ticker = args.strip().upper() if args.strip() else None
        if ticker:
            portfolio = _load_portfolio()
            holdings = [h for h in portfolio.get("holdings", []) if h["ticker"] == ticker]
            if not holdings:
                return "%s not in your portfolio." % ticker
            import data_layer
            df = data_layer.fetch_daily_ohlcv(ticker)
            result = dca_advisor.analyze_dca(holdings[0], df)
            return dca_advisor.format_report([result])
        else:
            dca_list = dca_advisor.analyze_portfolio_dca()
            return dca_advisor.format_report(dca_list)
    except Exception as e:
        return "DCA error: %s" % e


def cmd_tax(args=""):
    """Tax loss harvesting opportunities."""
    try:
        import tax_harvesting
        portfolio = _load_portfolio()
        analysis = tax_harvesting.analyze_harvesting(portfolio)
        return tax_harvesting.format_report(analysis)
    except Exception as e:
        return "Tax error: %s" % e


def cmd_rotation(args=""):
    """Sector rotation — where is money flowing?"""
    try:
        import sector_rotation
        perf = sector_rotation.fetch_sector_performance()
        rotation = sector_rotation.detect_rotation(perf)
        portfolio = _load_portfolio()
        exposure = sector_rotation.get_portfolio_exposure(rotation, portfolio)
        return sector_rotation.format_report(perf, rotation, exposure)
    except Exception as e:
        return "Rotation error: %s" % e


def cmd_weekly(args=""):
    """Generate weekly portfolio report."""
    try:
        import weekly_report
        report = weekly_report.generate_weekly_report()
        return weekly_report.format_report(report)
    except Exception as e:
        return "Weekly report error: %s" % e


def cmd_ibsync(args=""):
    """Sync portfolio from Interactive Brokers."""
    try:
        import ib_sync
        if not ib_sync.is_available():
            return "IB sync not available. Make sure TWS or IB Gateway is running and ib_insync is installed."
        result = ib_sync.sync_portfolio(dry_run="--dry-run" in args)
        if result.get("synced"):
            return "IB Sync complete!\nPositions: %d\nNet Liq: $%s\nCash: $%s" % (
                result.get("positions", 0),
                "{:,.0f}".format(result.get("net_liq", 0)),
                "{:,.0f}".format(result.get("cash", 0)),
            )
        return "IB Sync failed: %s" % result.get("error", "Unknown error")
    except Exception as e:
        return "IB sync error: %s" % e


def cmd_volume(args=""):
    """Scan for unusual volume spikes."""
    try:
        import volume_scanner
        threshold = 2.0
        if args.strip():
            try:
                threshold = float(args.strip())
            except ValueError:
                pass
        spikes = volume_scanner.scan_volume_spikes(threshold=threshold)
        return volume_scanner.format_report(spikes)
    except Exception as e:
        return "Volume scan error: %s" % e


def cmd_upgrades(args=""):
    """Check analyst upgrades/downgrades."""
    try:
        import upgrade_tracker
        days = 7
        if args.strip():
            try:
                days = int(args.strip())
            except ValueError:
                pass
        changes = upgrade_tracker.scan_all_recommendations(days=days)
        return upgrade_tracker.format_report(changes)
    except Exception as e:
        return "Upgrade tracker error: %s" % e


def cmd_performance(args=""):
    """Portfolio performance history and stats."""
    try:
        import performance_tracker
        days = 30
        if args.strip():
            try:
                days = int(args.strip())
            except ValueError:
                pass
        history = performance_tracker.get_history(days=days)
        if not history:
            # Try taking a snapshot first
            performance_tracker.take_snapshot()
            history = performance_tracker.get_history(days=days)
        if not history:
            return "No performance history yet. Run /perf again after market close."
        stats = performance_tracker.compute_stats(history)
        report = performance_tracker.format_report(stats, history)
        chart = performance_tracker.format_chart(history)
        return report + "\n\n" + chart
    except Exception as e:
        return "Performance error: %s" % e


def cmd_options(args=""):
    """Options strategy analysis for a ticker."""
    ticker = args.strip().upper() if args.strip() else ""
    if not ticker:
        return "Usage: /options TSLA\n\nAnalyzes the options chain and suggests a strategy."
    try:
        import options_analyzer
        result = options_analyzer.analyze_ticker_options(ticker, "BUY")
        return options_analyzer.format_options_report(result)
    except Exception as e:
        return "Options error: %s" % e


def cmd_grade(args=""):
    """Grade a ticker — full analysis with grade, logic, risk, and options."""
    ticker = args.strip().upper() if args.strip() else ""
    if not ticker:
        return "Usage: /grade TSLA\n\nFull graded analysis with logic, risks, and options strategy."
    try:
        import data_layer
        import technical_analysis
        import fundamental_analysis
        import signal_engine
        import position_sizing
        import trade_grader
        import options_analyzer

        df = data_layer.fetch_daily_ohlcv(ticker)
        if df.empty:
            return "No data for %s" % ticker

        tech = technical_analysis.analyze(ticker, df)
        fund = fundamental_analysis.analyze(ticker, "Technology")
        alert = signal_engine.evaluate(tech, fund, threshold=1)  # Low threshold to always get a grade

        if not alert:
            # Still generate a basic analysis even without a signal
            lines = ["%s — Grade: N/A (no signal)" % ticker]
            lines.append("Price: $%.2f" % tech.current_price)
            lines.append("RSI: %.1f | MACD: %.2f" % (tech.rsi, tech.macd_value))
            lines.append("Above 200 SMA: %s" % tech.price_above_200sma)
            lines.append("Fundamental: %d/15" % fund.fundamental_score)
            lines.append("\nNo confluence signal — waiting for setup.")
            return "\n".join(lines)

        plan = position_sizing.compute(alert)
        if not plan:
            return "%s signal detected but could not size position." % ticker

        grade_info = trade_grader.grade_trade(alert, plan)
        options_suggestion = None
        try:
            options_suggestion = options_analyzer.analyze_ticker_options(
                ticker, alert.direction, alert.signal_score
            )
        except Exception:
            pass

        return trade_grader.format_graded_alert(alert, plan, grade_info, options_suggestion)
    except Exception as e:
        return "Grade error: %s" % e


def cmd_bought(args=""):
    """Record a buy trade. Usage: /bought TSLA 100 245.50"""
    parts = args.strip().upper().split()
    if len(parts) < 3:
        return (
            "Usage: /bought TICKER SHARES PRICE\n"
            "Example: /bought TSLA 100 245.50\n\n"
            "Records the trade, updates your portfolio, and gives investor-quality feedback."
        )
    ticker = parts[0]
    try:
        shares = float(parts[1])
        price  = float(parts[2])
    except ValueError:
        return "Invalid format. Example: /bought TSLA 100 245.50"

    portfolio = _load_portfolio()
    holdings  = portfolio.get("holdings", [])

    # Find existing position or create new one
    existing = next((h for h in holdings if h.get("ticker") == ticker), None)
    if existing:
        old_shares = float(existing.get("shares", 0))
        old_cost   = float(existing.get("avg_cost", 0))
        new_shares = old_shares + shares
        new_avg    = (old_shares * old_cost + shares * price) / new_shares if new_shares else price
        existing["shares"]        = round(new_shares, 4)
        existing["avg_cost"]      = round(new_avg, 4)
        existing["current_price"] = price
        existing["cost_basis"]    = round(new_shares * new_avg, 2)
        existing["current_value"] = round(new_shares * price, 2)
        existing["unrealized_pnl"] = round((price - new_avg) * new_shares, 2)
        existing["pnl_pct"]       = round((price - new_avg) / new_avg * 100, 2) if new_avg else 0
        msg = f"Added to existing {ticker} position. New: {new_shares:g} shares @ avg ${new_avg:.2f}"
    else:
        holdings.append({
            "ticker":        ticker,
            "shares":        shares,
            "avg_cost":      round(price, 4),
            "current_price": price,
            "cost_basis":    round(shares * price, 2),
            "current_value": round(shares * price, 2),
            "unrealized_pnl": 0,
            "pnl_pct":       0,
            "sector":        "Unknown",
            "strategy":      "Manual",
        })
        msg = f"New position opened: {ticker} {shares} shares @ ${price:.2f}"

    portfolio["holdings"] = holdings
    # Update total portfolio value
    total_val = sum(float(h.get("current_value", 0) or 0) for h in holdings)
    cash      = float(portfolio.get("available_cash", 0) or 0) - shares * price
    portfolio["available_cash"]       = round(max(cash, 0), 2)
    portfolio["total_portfolio_value"] = round(total_val + max(cash, 0), 2)
    portfolio["_last_updated"]         = datetime.now().isoformat()

    _save_portfolio(portfolio)
    _commit_portfolio_to_github(portfolio)

    # Get investor feedback
    feedback = _get_trade_feedback(ticker, "BUY", shares, price, portfolio)
    return msg + "\n\n" + feedback


def cmd_sold(args=""):
    """Record a sell trade. Usage: /sold TSLA 50 260.00"""
    parts = args.strip().upper().split()
    if len(parts) < 3:
        return (
            "Usage: /sold TICKER SHARES PRICE\n"
            "Example: /sold TSLA 50 260.00\n\n"
            "Records the exit, calculates P&L, and gives trade feedback."
        )
    ticker = parts[0]
    try:
        shares = float(parts[1])
        price  = float(parts[2])
    except ValueError:
        return "Invalid format. Example: /sold TSLA 50 260.00"

    portfolio = _load_portfolio()
    holdings  = portfolio.get("holdings", [])
    existing  = next((h for h in holdings if h.get("ticker") == ticker), None)

    if not existing:
        return f"{ticker} not found in your portfolio. Use /sold {ticker} {shares} {price:.2f} after adding it via /bought."

    avg_cost     = float(existing.get("avg_cost", price))
    held_shares  = float(existing.get("shares", 0))
    sold_shares  = min(float(shares), held_shares)
    realized_pnl = round((price - avg_cost) * sold_shares, 2)
    pnl_pct      = round((price - avg_cost) / avg_cost * 100, 2) if avg_cost else 0
    pnl_sign     = "+" if realized_pnl >= 0 else ""

    if sold_shares >= held_shares:
        holdings.remove(existing)
        position_msg = f"Position closed: {ticker} (all {held_shares:.4g} shares)"
    else:
        remaining = held_shares - sold_shares
        existing["shares"]        = round(remaining, 4)
        existing["current_value"] = round(remaining * price, 2)
        existing["unrealized_pnl"] = round((price - avg_cost) * remaining, 2)
        position_msg = f"Partial exit: {ticker} — {sold_shares:.4g} sold, {remaining:.4g} remaining"

    # Record realized trade
    realized_trades = portfolio.get("realized_trades_ytd", [])
    realized_trades.append({
        "ticker":       ticker,
        "shares":       sold_shares,
        "avg_cost":     avg_cost,
        "exit_price":   price,
        "realized_pnl": realized_pnl,
        "pnl_pct":      pnl_pct,
        "date":         datetime.now().strftime("%Y-%m-%d"),
    })
    portfolio["realized_trades_ytd"] = realized_trades

    # Update realized P&L YTD
    prev_realized = float(portfolio.get("realized_pnl_ytd", 0) or 0)
    portfolio["realized_pnl_ytd"] = round(prev_realized + realized_pnl, 2)
    portfolio["holdings"]         = holdings
    cash = float(portfolio.get("available_cash", 0) or 0) + sold_shares * price
    total_val = sum(float(h.get("current_value", 0) or 0) for h in holdings)
    portfolio["available_cash"]        = round(cash, 2)
    portfolio["total_portfolio_value"] = round(total_val + cash, 2)
    portfolio["_last_updated"]         = datetime.now().isoformat()

    _save_portfolio(portfolio)
    _commit_portfolio_to_github(portfolio)

    result_line = f"P&L: {sold_shares} × ${price - avg_cost:+.2f} = {pnl_sign}${realized_pnl:,.2f} ({pnl_sign}{pnl_pct:.1f}%)"
    feedback    = _get_trade_feedback(ticker, "SELL", sold_shares, price, portfolio)
    return f"{position_msg}\n{result_line}\n\n{feedback}"


def cmd_trades(args=""):
    """Show recently recorded trades."""
    try:
        portfolio = _load_portfolio()
        trades    = portfolio.get("realized_trades_ytd", [])
        if not trades:
            return "No realized trades recorded yet.\n\nUse /sold TSLA 50 260 to record a sale."

        limit  = 10
        recent = trades[-limit:][::-1]
        lines  = ["Recent Trades (%d total YTD)" % len(trades), "─" * 35]
        for t in recent:
            sign = "+" if float(t.get("realized_pnl", 0)) >= 0 else ""
            lines.append("%s  %s × %s  %s$%.0f (%s%.1f%%)" % (
                t.get("date", "?"),
                t.get("ticker", "?"),
                t.get("shares", "?"),
                sign,
                abs(float(t.get("realized_pnl", 0))),
                sign,
                abs(float(t.get("pnl_pct", 0))),
            ))

        ytd = float(portfolio.get("realized_pnl_ytd", 0) or 0)
        ytd_sign = "+" if ytd >= 0 else ""
        lines.append("")
        lines.append("Realized YTD: %s$%.2f" % (ytd_sign, ytd))
        return "\n".join(lines)
    except Exception as e:
        return "Trades error: %s" % e


def cmd_sync(args=""):
    """Force an immediate IB Flex trade sync on demand."""
    try:
        import ib_flex
        if not ib_flex.is_configured():
            return (
                "IB Flex not configured yet.\n\n"
                "To enable full auto-sync:\n"
                "1. Log in to IB Account Management\n"
                "2. Reports → Flex Queries → Create New\n"
                "   (Activity query, Trades section, XML format)\n"
                "   Note the Query ID shown in the list\n"
                "3. Settings → Account Settings → Flex Web Service\n"
                "   Enable it and copy your Token\n"
                "4. Add to GitHub Secrets:\n"
                "   IB_FLEX_TOKEN    = <your token>\n"
                "   IB_FLEX_QUERY_ID = <your query ID>\n\n"
                "Once set, trades auto-detect every 15 min.\n"
                "Use /sync to force an immediate check."
            )
        send_message("Checking IB for new trades... (may take ~15s)", chat_id=CHAT_ID)

        # Fetch raw trades first so we can give meaningful diagnostics
        all_trades = ib_flex.fetch_todays_trades()
        if not all_trades:
            return (
                "Flex returned 0 trades.\n\n"
                "Possible causes:\n"
                "• Your Flex query date range doesn't cover recent trades\n"
                "  (set it to 'Last 5 business days' or 'Month to Date')\n"
                "• No trades have been executed in that window\n"
                "• The Flex token or query ID is wrong\n"
                "• The query doesn't include the 'Trades' section\n\n"
                "Run /sync again after fixing the query in IB Account Management."
            )

        new_trades = ib_flex.filter_new_trades(all_trades)

        if not new_trades:
            return (
                f"Flex returned {len(all_trades)} trade(s) — all already processed.\n\n"
                "If you expected new ones to show up, your seen-set may be caching\n"
                "them. To force a fresh import, clear logs/flex_seen_trades.json on\n"
                "GitHub and run /sync again."
            )

        # Apply each new trade to portfolio
        portfolio = ib_flex._load_portfolio()
        applied = []
        failed  = []
        for trade in new_trades:
            try:
                portfolio = ib_flex.apply_trade_to_portfolio(trade, portfolio)
                ib_flex._save_portfolio(portfolio)
                ib_flex.mark_trade_seen(trade.get("_uid", ""))
                applied.append(trade)
            except Exception as e:
                failed.append((trade, str(e)))

        if applied:
            ib_flex._commit_portfolio(portfolio)

        lines = ["Synced %d of %d trade(s):" % (len(applied), len(new_trades))]
        for t in applied:
            pnl     = t.get("realized_pnl", 0)
            pnl_str = " | P&L: %s$%.2f" % ("+" if pnl >= 0 else "", pnl) if pnl else ""
            lines.append("  %s %s × %g @ $%.2f%s" % (
                t["action"], t["ticker"], t["quantity"], t["price"], pnl_str))
        if failed:
            lines.append("")
            lines.append("%d trade(s) failed and will retry next sync:" % len(failed))
            for t, err in failed:
                lines.append("  %s %s × %g — %s" % (
                    t.get("action"), t.get("ticker"), t.get("quantity"), err[:100]))
        if applied:
            lines.append("")
            lines.append("Portfolio updated and committed.")
        return "\n".join(lines)
    except Exception as e:
        return "Sync error: %s" % e


def cmd_help(args=""):
    """List available commands."""
    return (
        "Stock Agent Commands:\n"
        "─────────────────────\n"
        "TRADE RECORDING\n"
        "/sync      — Force IB Flex sync (auto every 15 min)\n"
        "/bought TSLA 100 245.50 — Manual buy entry + feedback\n"
        "/sold TSLA 50 260.00   — Manual sell entry + P&L\n"
        "/trades    — Recent trade history\n"
        "\n"
        "ANALYSIS\n"
        "/buy       — BUY opportunities (watchlist scan)\n"
        "/sell      — Which holdings to sell?\n"
        "/grade TSLA — Full graded analysis + options\n"
        "/options TSLA — Options strategy suggestion\n"
        "/check TSLA — Analyze any ticker\n"
        "/scan      — Quick scan (holdings + top buys)\n"
        "\n"
        "PORTFOLIO\n"
        "/status    — Portfolio summary\n"
        "/positions — All open positions\n"
        "/perf      — Performance history + stats\n"
        "/risk      — Risk summary\n"
        "/regime    — Market regime\n"
        "/briefing  — Morning briefing\n"
        "\n"
        "RESEARCH\n"
        "/earnings  — Upcoming earnings\n"
        "/news      — Portfolio news\n"
        "/volume    — Unusual volume spikes\n"
        "/upgrades  — Analyst upgrades/downgrades\n"
        "/dca       — DCA suggestions\n"
        "/tax       — Tax harvesting\n"
        "/rotation  — Sector rotation\n"
        "/weekly    — Weekly report\n"
        "/ibsync    — Sync from IB\n"
        "/alerts    — Price alerts\n"
        "\n"
        "INDIAN PAPER SIGNALS\n"
        "/indiascan — Live dynamic scan of Indian market (NSE)\n"
        "/indiaregime — Indian market regime (Nifty 50 & India VIX)\n"
        "\n"
        "Tip: Type a ticker (e.g. NVDA) to analyze it.\n"
        "All alerts include grade, logic, risk, and options play."
    )


def cmd_indiascan(args=""):
    """Trigger a live/paper dynamic scan of the Indian market."""
    try:
        from indian_scanner_runner import IndianMarketScannerRunner
        runner = IndianMarketScannerRunner()
        limit = 30
        if args.strip().isdigit():
            limit = int(args.strip())
        summary = runner.run_cycle(limit=limit, force=True, dry_run=False)
        return (
            f"🇮🇳 *Indian Market Scan Completed:*\n"
            f"• Scanned: {summary.get('instruments_scanned', 0)} dynamically discovered instruments\n"
            f"• Signals Generated: {summary.get('signals_generated', 0)}\n"
            f"• Dispatched to Telegram: {summary.get('signals_emitted', 0)}\n"
            f"• Regime: {summary.get('market_regime', {}).get('regime', 'N/A')}\n"
            f"• Data Source: {'Live SmartAPI' if summary.get('data_source_live_ready') else 'Research (yfinance)'}"
        )
    except Exception as e:
        return f"Indian scan failed: {e}"


def cmd_indiaregime(args=""):
    """Current Indian market regime using Nifty 50 & India VIX."""
    try:
        from indian_market_regime import detect_indian_market_regime
        res = detect_indian_market_regime()
        nifty = f"₹{res.nifty_price:,.2f}" if res.nifty_price else "N/A"
        vix = f"{res.india_vix:.2f}" if res.india_vix else "N/A"
        return (
            f"🇮🇳 *Indian Market Regime:*\n"
            f"• Regime: *{res.regime}* (Confidence: {res.confidence}%)\n"
            f"• Nifty 50: *{nifty}*\n"
            f"• India VIX: *{vix}*\n"
            f"• Details: {res.description}\n"
            f"• As of: {res.timestamp_ist}"
        )
    except Exception as e:
        return f"Error detecting Indian market regime: {e}"


COMMANDS = {
    "/indiascan": cmd_indiascan,
    "/indiaregime": cmd_indiaregime,
    "/status": cmd_status,
    "/positions": cmd_positions,
    "/pos": cmd_positions,
    "/sell": cmd_sell,
    "/buy": cmd_buy,
    "/bought": cmd_bought,
    "/sold": cmd_sold,
    "/trades": cmd_trades,
    "/check": cmd_check,
    "/scan": cmd_scan,
    "/risk": cmd_risk,
    "/regime": cmd_regime,
    "/briefing": cmd_briefing,
    "/alerts": cmd_alerts,
    "/earnings": cmd_earnings,
    "/news": cmd_news,
    "/dca": cmd_dca,
    "/tax": cmd_tax,
    "/rotation": cmd_rotation,
    "/weekly": cmd_weekly,
    "/ibsync": cmd_ibsync,
    "/sync": cmd_sync,
    "/volume": cmd_volume,
    "/upgrades": cmd_upgrades,
    "/performance": cmd_performance,
    "/perf": cmd_performance,
    "/options": cmd_options,
    "/grade": cmd_grade,
    "/help": cmd_help,
    "/start": cmd_help,
}


def handle_message(text, chat_id):
    """Process an incoming message and respond."""
    text = text.strip()
    if not text.startswith("/"):
        # Natural language — single ticker treated as check
        words = text.upper().split()
        if len(words) == 1 and words[0].isalpha() and 1 <= len(words[0]) <= 5:
            return cmd_check(words[0])
        # Route to AI trade advisor for natural language questions
        try:
            import trade_advisor
            answer = trade_advisor.answer_question(text)
            if answer:
                return answer
        except Exception as e:
            return "Sorry, I couldn't process that question.\nError: %s\n\nTry /help for commands." % str(e)
        return cmd_help()

    parts = text.split(maxsplit=1)
    command = parts[0].lower().split("@")[0]  # Remove @botname suffix
    args = parts[1] if len(parts) > 1 else ""

    handler = COMMANDS.get(command)
    if handler:
        return handler(args)
    return "Unknown command: %s\n\nType /help for available commands." % command


def _load_offset() -> int:
    """Load the last seen update offset from disk (persists within a run)."""
    try:
        if OFFSET_FILE.exists():
            val = OFFSET_FILE.read_text().strip()
            return int(val) if val else None
    except Exception:
        pass
    return None


def _save_offset(offset: int) -> None:
    try:
        OFFSET_FILE.parent.mkdir(exist_ok=True)
        OFFSET_FILE.write_text(str(offset))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Message processing queue — processes one message at a time in a background
# thread so the polling loop stays free to receive the next message immediately.
# ---------------------------------------------------------------------------

_msg_queue: _queue.Queue = _queue.Queue()


def _message_worker() -> None:
    """Background thread: dequeue and handle messages one at a time."""
    while True:
        text, chat_id = _msg_queue.get()
        try:
            response = handle_message(text, chat_id)
            send_message(response, chat_id=chat_id)
        except Exception as e:
            send_message("Error: %s" % str(e)[:200], chat_id=chat_id)
            logger.error("Handler error: %s", traceback.format_exc())
        finally:
            _msg_queue.task_done()


def run_bot():
    """Main bot loop — long polls for messages."""
    if not TOKEN:
        print("ERROR: TELEGRAM_BOT_TOKEN not set in .env")
        print("1. Message @BotFather on Telegram")
        print("2. Send /newbot and follow instructions")
        print("3. Copy the token to .env")
        sys.exit(1)

    if not CHAT_ID:
        print("WARNING: TELEGRAM_CHAT_ID not set — bot will respond to anyone")
        print("Run: python3 telegram_bot.py --get-chat-id")

    print("Stock Agent Telegram Bot starting...")
    print("Listening for commands...")

    # Start background message-processing thread
    worker = threading.Thread(target=_message_worker, daemon=True)
    worker.start()

    # Restore offset so we don't replay old messages after a restart
    offset = _load_offset()

    # Delete any pending webhook so long-polling works cleanly
    try:
        requests.post(API_BASE + "/deleteWebhook", timeout=10)
    except Exception:
        pass

    consecutive_errors = 0

    while True:
        try:
            updates = get_updates(offset=offset, timeout=30)
            consecutive_errors = 0  # Reset on success

            for update in updates:
                offset = update["update_id"] + 1
                _save_offset(offset)

                msg     = update.get("message", {})
                text    = msg.get("text", "")
                chat_id = msg.get("chat", {}).get("id")

                if not text or not chat_id:
                    continue

                # Security: only respond to authorized chat
                if CHAT_ID and str(chat_id) != str(CHAT_ID):
                    send_message("Unauthorized. Your chat ID: %s" % chat_id, chat_id=chat_id)
                    continue

                logger.info("Received: %s", text)

                # Send immediate acknowledgment so user knows bot is alive,
                # then dispatch to background worker to avoid blocking polling.
                send_message("⏳", chat_id=chat_id)
                _msg_queue.put((text, chat_id))

        except KeyboardInterrupt:
            print("\nBot stopped.")
            break
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 409:
                # 409 Conflict — another bot instance is already long-polling.
                # This happens briefly during restarts. Wait and retry.
                consecutive_errors += 1
                wait = min(10 * consecutive_errors, 60)
                logger.warning("409 Conflict (duplicate instance). Waiting %ds for other instance to die...", wait)
                time.sleep(wait)
            else:
                logger.error("HTTP error in bot loop: %s", e)
                time.sleep(5)
        except Exception as e:
            consecutive_errors += 1
            logger.error("Bot loop error: %s", e)
            if consecutive_errors >= 25:
                logger.critical("Bot hit %d consecutive errors — exiting for GitHub Actions restart", consecutive_errors)
                sys.exit(1)
            import random
            wait = min(5 * consecutive_errors + random.randint(0, 3), 45)
            logger.info("Retrying in %d seconds...", wait)
            time.sleep(wait)


def main():
    parser = argparse.ArgumentParser(description="Stock Agent Telegram Bot")
    parser.add_argument("--get-chat-id", action="store_true", help="Find your Telegram chat ID")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # Change to project directory
    os.chdir(Path(__file__).parent)

    if args.get_chat_id:
        get_chat_id()
    else:
        run_bot()


if __name__ == "__main__":
    main()
