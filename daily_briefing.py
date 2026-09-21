#!/usr/bin/env python3
"""
Daily Briefing Generator — produces a morning summary of portfolio status,
market regime, open positions, key levels, and today's watchlist.

Can output to console, file, or send via notification channels.

Usage:
    python3 daily_briefing.py                  # Print to console
    python3 daily_briefing.py --send           # Send via configured channels
    python3 daily_briefing.py --file brief.txt # Save to file
    python3 daily_briefing.py --json           # Output as JSON
"""

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def is_indian_market() -> bool:
    return os.getenv("MARKET", "").upper() == "INDIA" or os.getenv("TIMEZONE") == "Asia/Kolkata"


def get_currency_symbol() -> str:
    return "₹" if is_indian_market() else "$"


def get_portfolio_summary():
    # type: () -> Dict
    """Get real portfolio summary from portfolio.json (IB-synced)."""
    try:
        import data_layer
        portfolio = data_layer.load_portfolio()
        holdings = portfolio.get("holdings", [])
        cash = portfolio.get("available_cash", 0)
        total_value = portfolio.get("total_portfolio_value", 0)

        # Compute from holdings
        total_invested = sum(h.get("current_value", 0) for h in holdings)
        total_cost = sum(h.get("cost_basis", 0) for h in holdings)
        unrealized = sum(h.get("unrealized_pnl", 0) for h in holdings)

        # Realized P&L from trades this year
        realized_trades = portfolio.get("realized_trades_ytd", [])
        realized = portfolio.get("realized_pnl_ytd", sum(t.get("realized_pnl", 0) for t in realized_trades))

        total_pnl = unrealized + realized

        return {
            "total_equity": round(total_value, 2),
            "cash": round(cash, 2),
            "invested": round(total_invested, 2),
            "total_cost": round(total_cost, 2),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(total_pnl / total_cost * 100, 2) if total_cost > 0 else 0,
            "unrealized_pnl": round(unrealized, 2),
            "realized_pnl": round(realized, 2),
            "open_positions": len(holdings),
            "total_trades": len(realized_trades),
            "broker": portfolio.get("_broker", ""),
            "last_sync": portfolio.get("last_ib_sync", portfolio.get("_last_updated", "")),
        }
    except Exception as e:
        logger.debug("Portfolio summary failed: %s", e)
        return {}


def get_position_details():
    # type: () -> List[Dict]
    """Get details for each open position from real portfolio."""
    try:
        import data_layer
        portfolio = data_layer.load_portfolio()
        holdings = portfolio.get("holdings", [])

        details = []
        for h in holdings:
            ticker = h.get("ticker", "???")
            shares = h.get("shares", 0)
            avg_cost = h.get("avg_cost", 0)
            current = h.get("current_price", avg_cost)
            cost_basis = h.get("cost_basis", avg_cost * shares)
            unrealized = h.get("unrealized_pnl", 0)
            pnl_pct = h.get("pnl_pct", 0)
            sector = h.get("sector", "")
            strategy = h.get("strategy", "")
            current_value = h.get("current_value", current * shares)

            details.append({
                "ticker": ticker,
                "direction": "LONG",
                "shares": shares,
                "entry_price": round(avg_cost, 2),
                "current_price": round(current, 2),
                "current_value": round(current_value, 2),
                "cost_basis": round(cost_basis, 2),
                "unrealized_pnl": round(unrealized, 2),
                "unrealized_pct": round(pnl_pct, 2),
                "sector": sector,
                "strategy": strategy,
            })

        # Sort by unrealized P&L descending (best first)
        details.sort(key=lambda x: x["unrealized_pnl"], reverse=True)
        return details
    except Exception:
        return []


def get_regime_summary():
    # type: () -> Dict
    """Get current market regime info."""
    try:
        if is_indian_market():
            from indian_market_regime import detect_indian_market_regime
            res = detect_indian_market_regime()
            nifty = f"₹{res.nifty_price:,.2f}" if res.nifty_price else "N/A"
            vix = f"{res.india_vix:.2f}" if res.india_vix else "N/A"
            return {
                "regime": res.regime,
                "confidence": res.confidence,
                "description": f"Nifty 50: {nifty} | India VIX: {vix} | {res.description}",
                "threshold_adj": 0,
                "position_mult": 1.0,
            }
        import market_regime
        history = market_regime.get_history(1)
        if history:
            h = history[0]
            params = market_regime.REGIME_PARAMS.get(h["regime"], {})
            return {
                "regime": h["regime"],
                "confidence": h["confidence"],
                "description": params.get("description", ""),
                "threshold_adj": params.get("threshold_adjustment", 0),
                "position_mult": params.get("position_size_mult", 1.0),
            }
    except Exception:
        pass
    return {}


def get_risk_snapshot():
    # type: () -> Dict
    """Get key risk metrics."""
    try:
        import risk_metrics
        pnls, capital = risk_metrics.load_paper_pnls()
        if not pnls:
            pnls, capital = risk_metrics.load_backtest_pnls()
        if pnls:
            m = risk_metrics.compute_metrics(pnls, starting_capital=capital)
            return {
                "sharpe_ratio": m.get("sharpe_ratio", 0),
                "sortino_ratio": m.get("sortino_ratio", 0),
                "max_drawdown_pct": m.get("max_drawdown_pct", 0),
                "win_rate": m.get("win_rate", 0),
                "profit_factor": m.get("profit_factor", 0),
                "kelly_criterion": m.get("kelly_criterion", 0),
            }
    except Exception:
        pass
    return {}


def get_journal_pending():
    # type: () -> List[str]
    """Get tickers with closed trades that need journal reviews."""
    try:
        import trade_journal
        stats = trade_journal.get_stats()
        unreviewed = stats.get("unreviewed_count", 0)
        if unreviewed > 0:
            entries = trade_journal.get_entries(closed_only=True, limit=50)
            return [e["ticker"] for e in entries if not e.get("review_note")]
    except Exception:
        pass
    return []


def get_top_movers():
    # type: () -> List[Dict]
    """Get pre-market / recent price changes for holdings."""
    try:
        import data_layer
        portfolio = data_layer.load_portfolio()
        holdings = portfolio.get("holdings", [])
        tickers = [h["ticker"] for h in holdings]
        if not tickers:
            return []

        import yfinance as yf
        movers = []
        for ticker in tickers:
            try:
                resolved = data_layer.resolve_ticker(ticker)
                t = yf.Ticker(resolved)
                info = t.fast_info
                price = getattr(info, "last_price", None) or 0
                prev = getattr(info, "previous_close", None) or 0
                if price and prev:
                    change_pct = (price - prev) / prev * 100
                    movers.append({
                        "ticker": ticker,
                        "price": round(price, 2),
                        "change_pct": round(change_pct, 2),
                    })
            except Exception:
                continue

        # Sort by absolute change (biggest movers first)
        movers.sort(key=lambda x: abs(x["change_pct"]), reverse=True)
        return movers[:8]  # Top 8 movers
    except Exception:
        return []


def generate_briefing():
    # type: () -> Dict
    """Generate complete briefing dictionary."""
    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M"),
        "regime": get_regime_summary(),
        "portfolio": get_portfolio_summary(),
        "positions": get_position_details(),
        "top_movers": get_top_movers(),
        "risk": get_risk_snapshot(),
        "pending_reviews": get_journal_pending(),
    }


def format_briefing(briefing):
    # type: (Dict) -> str
    """Format briefing dict into readable text."""
    cur = get_currency_symbol()
    lines = []
    lines.append("☀️ DAILY BRIEFING — %s" % briefing["date"])
    lines.append("=" * 40)

    # Regime
    regime = briefing.get("regime", {})
    if regime:
        regime_emoji = {"BULL": "🐂", "BEAR": "🐻", "VOLATILE": "🌊", "SIDEWAYS": "↔️"}.get(
            regime.get("regime", ""), "📊")
        lines.append("\n%s Market: %s (%d%% confidence)" % (
            regime_emoji, regime.get("regime", "N/A"), regime.get("confidence", 0)))
        lines.append("  %s" % regime.get("description", ""))

    # Portfolio
    port = briefing.get("portfolio", {})
    if port:
        lines.append("\n📊 PORTFOLIO")
        lines.append("  " + "-" * 50)
        lines.append(f"  Total Value:  {cur}{port.get('total_equity', 0):,.2f}")
        lines.append(f"  Cash:         {cur}{port.get('cash', 0):,.2f}")
        lines.append(f"  Unrealized:   {cur}{port.get('unrealized_pnl', 0):+,.2f}")
        lines.append(f"  Realized YTD: {cur}{port.get('realized_pnl', 0):+,.2f}")
        lines.append(f"  Combined P&L: {cur}{port.get('total_pnl', 0):+,.2f} ({port.get('total_return_pct', 0):+.1f}%)")
        lines.append("  Positions:    %d holdings  |  %d closed trades YTD" % (
            port.get("open_positions", 0), port.get("total_trades", 0)))
        if port.get("broker"):
            sync = port.get("last_sync", "")[:16].replace("T", " ")
            lines.append("  Source:       %s (synced %s)" % (port["broker"], sync))

    # Positions
    positions = briefing.get("positions", [])
    if positions:
        # Separate winners and losers
        winners = [p for p in positions if p["unrealized_pnl"] >= 0]
        losers = [p for p in positions if p["unrealized_pnl"] < 0]

        lines.append("\n📈 YOUR HOLDINGS (%d positions)" % len(positions))
        lines.append("  " + "-" * 50)

        if winners:
            lines.append("  🟢 WINNERS:")
            for p in winners:
                lines.append(f"  {p['ticker']:<5}  {p['shares']} shares  avg {cur}{p['entry_price']:.2f} → {cur}{p['current_price']:.2f}  +{cur}{p['unrealized_pnl']:,.0f} ({p['unrealized_pct']:+.1f}%)")

        if losers:
            lines.append("  🔴 LOSERS:")
            for p in losers:
                lines.append(f"  {p['ticker']:<5}  {p['shares']} shares  avg {cur}{p['entry_price']:.2f} → {cur}{p['current_price']:.2f}  -{cur}{abs(p['unrealized_pnl']):,.0f} ({p['unrealized_pct']:.1f}%)")

    # Top movers (quick live price check)
    top_movers = briefing.get("top_movers", [])
    if top_movers:
        lines.append("\n🔥 PRE-MARKET MOVERS (your holdings)")
        lines.append("  " + "-" * 50)
        for m in top_movers:
            emoji = "🟢" if m["change_pct"] >= 0 else "🔴"
            lines.append(f"  {emoji} {m['ticker']:<5}  {cur}{m['price']:.2f}  {m['change_pct']:+.1f}%")

    # Risk
    risk = briefing.get("risk", {})
    if risk:
        lines.append("\n  RISK METRICS")
        lines.append("  " + "-" * 50)
        lines.append("  Sharpe: %.2f  |  Sortino: %.2f  |  Win rate: %.1f%%" % (
            risk.get("sharpe_ratio", 0), risk.get("sortino_ratio", 0),
            risk.get("win_rate", 0)))
        lines.append("  Max DD: %.1f%%  |  PF: %s  |  Kelly: %.1f%%" % (
            risk.get("max_drawdown_pct", 0),
            risk.get("profit_factor", 0),
            risk.get("kelly_criterion", 0)))

    # Realized trades summary
    port = briefing.get("portfolio", {})
    realized = port.get("realized_pnl", 0)
    if realized != 0:
        emoji = "✅" if realized >= 0 else "⚠️"
        lines.append(f"\n{emoji} REALIZED P&L YTD: {cur}{realized:+,.2f} ({port.get('total_trades', 0)} closed trades)")

    lines.append("\n" + "=" * 40)
    lines.append("💬 Chat with me anytime — just type a question!")
    return "\n".join(lines)


def send_briefing(text):
    # type: (str) -> None
    """Send briefing through configured notification channels."""
    import requests

    # Slack
    webhook = os.getenv("SLACK_WEBHOOK_URL", "")
    if webhook and not webhook.startswith("https://hooks.slack.com/services/YOUR"):
        try:
            requests.post(webhook, json={
                "text": "```%s```" % text,
                "username": "Stock Agent Briefing",
            }, timeout=10)
            logger.info("Briefing sent to Slack")
        except Exception as e:
            logger.error("Slack briefing failed: %s", e)

    # Discord
    discord_url = os.getenv("DISCORD_WEBHOOK_URL", "")
    if discord_url and "discord.com/api/webhooks" in discord_url:
        try:
            requests.post(discord_url, json={
                "content": "```%s```" % text[:1900],  # Discord limit
                "username": "Stock Agent",
            }, timeout=10)
            logger.info("Briefing sent to Discord")
        except Exception as e:
            logger.error("Discord briefing failed: %s", e)

    # Email
    smtp_host = os.getenv("SMTP_HOST", "")
    smtp_user = os.getenv("SMTP_USER", "")
    to_addr = os.getenv("ALERT_EMAIL_TO", "")
    if smtp_host and smtp_user and to_addr and not smtp_user.startswith("your_"):
        try:
            import smtplib
            from email.mime.text import MIMEText

            msg = MIMEText(text)
            msg["Subject"] = "Daily Briefing — %s" % datetime.now().strftime("%Y-%m-%d")
            msg["From"] = smtp_user
            msg["To"] = to_addr

            port = int(os.getenv("SMTP_PORT", "587"))
            password = os.getenv("SMTP_PASSWORD", "")
            with smtplib.SMTP(smtp_host, port) as server:
                server.starttls()
                server.login(smtp_user, password)
                server.sendmail(smtp_user, [to_addr], msg.as_string())
            logger.info("Briefing emailed to %s", to_addr)
        except Exception as e:
            logger.error("Email briefing failed: %s", e)

    # Telegram
    try:
        import telegram_bot
        telegram_bot.send_message(text)
        logger.info("Briefing sent to Telegram")
    except Exception as e:
        logger.debug("Telegram briefing failed: %s", e)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Daily Briefing Generator")
    parser.add_argument("--send", action="store_true", help="Send via notification channels")
    parser.add_argument("--file", type=str, help="Save to file")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

    briefing = generate_briefing()

    if args.json:
        print(json.dumps(briefing, indent=2, default=str))
        return

    text = format_briefing(briefing)
    print(text)

    if args.file:
        with open(args.file, "w") as f:
            f.write(text)
        print("  Saved to %s" % args.file)

    if args.send:
        send_briefing(text)


if __name__ == "__main__":
    main()
