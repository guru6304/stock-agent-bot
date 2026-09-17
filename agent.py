#!/usr/bin/env python3
"""
Stock Investment Agent — main runner.

Scans the watchlist on a schedule, runs technical + fundamental analysis,
applies the confluence engine, sizes positions, and sends notifications.

Usage:
    python agent.py              # Run once (single scan)
    python agent.py --schedule   # Run on schedule during market hours
    python agent.py --ticker AAPL  # Analyze a single ticker
    python agent.py --paper       # Paper trading mode (auto-execute virtual trades)
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict

import schedule
from dotenv import load_dotenv

import alert_tracker
import config_validator
import correlation_guard
import daily_briefing
import database
import data_layer
import earnings_guard
import eod_report
import health_monitor
import market_regime
import multi_timeframe
import paper_trader
import performance_tracker
import position_monitor
import price_alerts
import trailing_stop
import technical_analysis
import fundamental_analysis
import signal_engine
import position_sizing
import notifications
import strategy_optimizer
import watchlist_curator

load_dotenv()

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)-20s  %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/agent.log", mode="a"),
    ],
)
logger = logging.getLogger("stock-agent")

# Suppress noisy third-party loggers
for lib in ("urllib3", "yfinance", "peewee"):
    logging.getLogger(lib).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Market hours check
# ---------------------------------------------------------------------------

def is_market_hours() -> bool:
    """Return True if current time is within configured market hours."""
    try:
        import pytz
        tz_name = os.getenv("TIMEZONE", "America/New_York")
        tz = pytz.timezone(tz_name)
        now = datetime.now(tz)
    except ImportError:
        now = datetime.now()

    open_h = int(os.getenv("MARKET_OPEN_HOUR", "9"))
    open_m = int(os.getenv("MARKET_OPEN_MINUTE", "30"))
    close_h = int(os.getenv("MARKET_CLOSE_HOUR", "16"))
    close_m = int(os.getenv("MARKET_CLOSE_MINUTE", "0"))

    market_open = now.replace(hour=open_h, minute=open_m, second=0, microsecond=0)
    market_close = now.replace(hour=close_h, minute=close_m, second=0, microsecond=0)

    # Skip weekends
    if now.weekday() >= 5:
        return False

    return market_open <= now <= market_close


# ---------------------------------------------------------------------------
# Core scan logic
# ---------------------------------------------------------------------------

# Module-level data cache for correlation checks within a scan cycle
_data_cache = {}  # type: Dict[str, Any]

# Cache EOD report text generated at 16:15 so 16:25 digest reuses it (no double-generation)
_eod_report_cache = None
_eod_report_cache_date = None


def scan_ticker(ticker: str, sector: str = "Technology", paper_mode: bool = False) -> None:
    """Run the full analysis pipeline for a single ticker."""
    logger.info("─── Scanning %s (%s) ───", ticker, sector)

    # 1. Fetch daily OHLCV
    df_daily = data_layer.fetch_daily_ohlcv(ticker)
    if df_daily.empty:
        logger.warning("No price data for %s — skipping", ticker)
        return

    # Cache data for correlation guard
    _data_cache[ticker] = df_daily

    # 2. Technical analysis
    tech = technical_analysis.analyze(ticker, df_daily)
    logger.info(
        "%s technicals — price: %.2f | EMA9: %.2f | EMA21: %.2f | SMA200: %.2f | above 200: %s"
        " | RSI: %.1f | MACD: %.2f | BB: [%.2f–%.2f]",
        ticker, tech.current_price, tech.ema9, tech.ema21, tech.sma200,
        tech.price_above_200sma, tech.rsi, tech.macd_value,
        tech.bb_lower, tech.bb_upper,
    )
    if tech.pattern_details:
        logger.info("%s patterns: %s", ticker, ", ".join(tech.pattern_details))

    # 3. Fundamental analysis
    fund = fundamental_analysis.analyze(ticker, sector)
    logger.info(
        "%s fundamentals — score: %d/6 | P/E: %s | D/E: %s | sentiment: %.2f",
        ticker,
        fund.fundamental_score,
        f"{fund.pe_ratio:.1f}" if fund.pe_ratio else "N/A",
        f"{fund.debt_to_equity:.2f}" if fund.debt_to_equity else "N/A",
        fund.news_sentiment_score,
    )

    # 4. Confluence engine — should we alert?
    alert = signal_engine.evaluate(tech, fund)
    if alert is None:
        logger.info("%s — no alert (below threshold)", ticker)
        return

    # 4b. Deduplication check
    if alert_tracker.is_duplicate(alert.ticker, alert.direction, alert.triggered_signals):
        logger.info("%s — duplicate alert suppressed (cooldown active)", ticker)
        return

    # 4c. Earnings guard — block entries near earnings dates
    earnings_safe, earnings_info = earnings_guard.check_earnings_safe(ticker)
    if not earnings_safe:
        logger.warning("%s — blocked by earnings guard: %s", ticker, earnings_info["reason"])
        return

    # 4d. Correlation guard — prevent correlated position clustering
    if paper_mode:
        state = paper_trader.load_state()
        open_positions = state.get("open_positions", [])
        if open_positions:
            corr_safe, corr_info = correlation_guard.check_correlation_safe(
                ticker, sector, open_positions,
                data_cache=_data_cache, candidate_data=df_daily,
            )
            if not corr_safe:
                check = corr_info.get("check", "unknown")
                if check == "sector_concentration":
                    reason = corr_info["sector"]["reason"]
                else:
                    reason = corr_info.get("correlation", {}).get("reason", "high correlation")
                logger.warning("%s — blocked by correlation guard: %s", ticker, reason)
                return

    # 5. Position sizing
    plan = position_sizing.compute(alert)
    if plan is None:
        logger.warning("%s — alert triggered but could not size position", ticker)
        return

    # 6. Multi-timeframe confirmation
    df_intraday = data_layer.fetch_intraday_ohlcv(ticker)
    mtf = multi_timeframe.confirm_signal(tech, df_intraday, alert.direction)
    if mtf.details:
        logger.info("%s MTF: %s", ticker, " | ".join(mtf.details))

    if mtf.score_adjustment != 0:
        adjusted_score = alert.signal_score + mtf.score_adjustment
        logger.info(
            "%s score adjusted: %d → %d (MTF %+d)",
            ticker, alert.signal_score, adjusted_score, mtf.score_adjustment,
        )
        alert.signal_score = adjusted_score

        # Re-check threshold after adjustment
        threshold = int(os.getenv("SIGNAL_THRESHOLD", "5"))
        if adjusted_score < threshold:
            logger.info("%s — score dropped below threshold after MTF adjustment", ticker)
            return

    if not mtf.confirmed and mtf.score_adjustment <= -2:
        logger.info("%s — strong MTF contradiction, suppressing alert", ticker)
        return

    # 7. Execute trade or send notification
    if paper_mode:
        # Paper trading: auto-execute virtual trade
        state = paper_trader.load_state()
        position = paper_trader.execute_entry(state, alert, plan)
        if position:
            logger.info("%s — paper trade entered: %s %d shares @ $%.2f",
                        ticker, plan.direction, plan.shares, plan.entry_price)
        else:
            logger.info("%s — paper trade not executed (position limit or cash)", ticker)
    else:
        # Live mode: notify and track
        notifications.notify(alert, plan)

    alert_tracker.record_alert(alert.ticker, alert.direction, alert.triggered_signals)

    # 8. Register position for trailing stop tracking
    if not paper_mode:
        trailing_stop.add_position(
            ticker=alert.ticker,
            direction=alert.direction,
            entry_price=plan.entry_price,
            initial_stop=plan.stop_loss,
            shares=plan.shares,
            target_1=plan.target_1,
            target_2=plan.target_2,
            target_3=plan.target_3,
        )


def run_scan(paper_mode: bool = False) -> None:
    """Scan the entire watchlist."""
    if os.getenv("TIMEZONE") == "Asia/Kolkata" or os.getenv("MARKET", "").upper() == "INDIA":
        logger.info("Executing Dynamic Indian Market Scanner (NSE / BSE)...")
        try:
            from indian_scanner_runner import IndianMarketScannerRunner
            runner = IndianMarketScannerRunner()
            runner.run_cycle(force=True)
            return
        except Exception as e:
            logger.error("Indian market scan failed: %s", e, exc_info=True)
            return

    global _data_cache
    _data_cache = {}  # Clear cache for each scan cycle
    mode_label = "PAPER TRADING" if paper_mode else "STOCK AGENT"
    logger.info("=" * 60)
    logger.info("  %s SCAN — %s", mode_label, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 60)

    # Detect market regime and adjust strategy
    try:
        regime_result = market_regime.detect_regime()
        regime = regime_result["regime"]
        adjustments = regime_result["params"]
        logger.info("Market regime: %s (confidence: %d%%) — %s",
                     regime, regime_result["confidence"], adjustments["description"])

        # Apply threshold adjustment
        base_threshold = int(os.getenv("SIGNAL_THRESHOLD", "5"))
        adjusted_threshold = base_threshold + adjustments["threshold_adjustment"]
        os.environ["SIGNAL_THRESHOLD"] = str(adjusted_threshold)
        if adjustments["threshold_adjustment"] != 0:
            logger.info("Signal threshold adjusted: %d → %d (regime: %s)",
                         base_threshold, adjusted_threshold, regime)
    except Exception as e:
        logger.warning("Regime detection failed, using defaults: %s", e)
        regime_result = None

    watchlist = data_layer.load_watchlist()
    if not watchlist:
        logger.error("Watchlist is empty — nothing to scan")
        return

    logger.info("Scanning %d tickers: %s", len(watchlist), ", ".join(w["ticker"] for w in watchlist))

    for entry in watchlist:
        try:
            scan_ticker(entry["ticker"], entry.get("sector", "Technology"), paper_mode=paper_mode)
        except Exception as e:
            logger.error("Error scanning %s: %s", entry["ticker"], e, exc_info=True)

    # In paper mode, also update existing positions (trailing stops + exits)
    if paper_mode:
        state = paper_trader.load_state()
        if state["open_positions"]:
            logger.info("Updating %d paper positions...", len(state["open_positions"]))
            actions = paper_trader.update_positions(state)
            for a in actions:
                if "trade" in a:
                    t = a["trade"]
                    logger.info("Paper %s: %s P&L=$%.2f", a["action"], t["ticker"], t["pnl"])

    # Check price alerts
    try:
        triggered = price_alerts.check_all_alerts(include_position_alerts=paper_mode)
        if triggered:
            logger.info("Price alerts triggered: %d", len(triggered))
            price_alerts.send_triggered_alerts(triggered)
    except Exception as e:
        logger.debug("Price alert check failed: %s", e)

    # Monitor real positions for sell signals
    try:
        results = position_monitor.check_all_positions()
        actionable = position_monitor.get_actionable(results)
        if actionable:
            sell_count = sum(1 for r in actionable if r["action"] == "SELL")
            trim_count = sum(1 for r in actionable if r["action"] == "TRIM")
            watch_count = sum(1 for r in actionable if r["action"] == "WATCH")
            logger.info("Position monitor: %d SELL, %d TRIM, %d WATCH",
                         sell_count, trim_count, watch_count)
            for r in actionable:
                logger.info("  [%s] %s — %s", r["action"], r["ticker"],
                            "; ".join(r["reasons"][:2]))
            # Send notifications for SELL and TRIM
            urgent = [r for r in actionable if r["action"] in ("SELL", "TRIM")]
            if urgent:
                position_monitor.send_sell_alerts(results)
        else:
            logger.info("Position monitor: all positions healthy")
    except Exception as e:
        logger.debug("Position monitor failed: %s", e)

    # Flush all queued Telegram alerts as ONE consolidated message
    try:
        notifications.flush_telegram_batch()
    except Exception as e:
        logger.debug("Telegram batch flush failed: %s", e)

    logger.info("Scan complete.\n")


def _send_daily_briefing() -> None:
    """Generate morning briefing — Telegram only; email goes in the 4:25 PM digest."""
    try:
        briefing = daily_briefing.generate_briefing()
        text = daily_briefing.format_briefing(briefing)
        logger.info("Daily briefing generated")
        # Telegram: real-time delivery (send_briefing sends to Slack/Discord/Telegram only,
        # its internal email code is suppressed because send_email_text now queues)
        daily_briefing.send_briefing(text)
        # Queue for the daily digest email
        notifications.set_briefing_for_digest(text)
    except Exception as e:
        logger.error("Failed to send daily briefing: %s", e)


def _send_eod_report() -> None:
    """Generate and send end-of-day report (Telegram only; email is in the digest)."""
    global _eod_report_cache, _eod_report_cache_date
    try:
        report = eod_report.generate_report()
        eod_report.save_report(report)
        eod_report.send_report(report)
        # Cache formatted text so _send_daily_digest() reuses it instead of regenerating
        try:
            _eod_report_cache = eod_report.format_report(report)
            _eod_report_cache_date = datetime.now().date()
        except Exception:
            pass
        logger.info("EOD report generated and sent")
    except Exception as e:
        logger.error("Failed to generate EOD report: %s", e)


def _send_daily_digest() -> None:
    """Send the ONE daily email — all trade alerts + briefing + EOD report combined."""
    global _eod_report_cache, _eod_report_cache_date
    try:
        eod_text = None
        today = datetime.now().date()
        if _eod_report_cache and _eod_report_cache_date == today:
            # Reuse the report already generated at 16:15 — no double API call
            eod_text = _eod_report_cache
            logger.debug("Using cached EOD report for daily digest")
        else:
            # Cache miss (e.g. bot restarted between 16:15 and 16:25) — generate fresh
            try:
                report = eod_report.generate_report()
                eod_report.save_report(report)
                eod_text = eod_report.format_report(report)
            except Exception as e:
                logger.debug("EOD report for digest failed: %s", e)
        notifications.send_daily_digest(eod_text=eod_text)
        logger.info("Daily digest email sent")
    except Exception as e:
        logger.error("Failed to send daily digest: %s", e)


def _take_performance_snapshot() -> None:
    """Take a daily portfolio performance snapshot (P&L history)."""
    try:
        snapshot = performance_tracker.take_snapshot()
        if snapshot:
            logger.info(
                "Performance snapshot: net_liq=$%.0f total_pnl=$%.0f",
                snapshot["net_liq"], snapshot["total_pnl"],
            )
        else:
            logger.warning("Performance snapshot returned None")
    except Exception as e:
        logger.error("Failed to take performance snapshot: %s", e)


def _run_health_check() -> None:
    """Run system health checks and log results."""
    try:
        results = health_monitor.run_all_checks()
        overall = health_monitor.get_overall_status(results)
        health_monitor.save_health_report(results)
        if overall == "FAIL":
            logger.error("Health check FAILED — check logs/health_report.json")
            for r in results:
                if r["status"] == "FAIL":
                    logger.error("  %s: %s", r["name"], r["message"])
        elif overall == "WARN":
            logger.warning("Health check warnings detected")
            for r in results:
                if r["status"] == "WARN":
                    logger.warning("  %s: %s", r["name"], r["message"])
        else:
            logger.info("Health check: all systems OK")
    except Exception as e:
        logger.error("Health check crashed: %s", e)


def _commit_portfolio_to_github() -> None:
    """Commit current portfolio.json to GitHub so prices persist across restarts."""
    import base64
    import json as _json
    import requests as _requests
    token = os.getenv("GITHUB_TOKEN", "")
    repo = os.getenv("GITHUB_REPOSITORY", "")
    if not token or not repo:
        return
    try:
        with open("portfolio.json") as f:
            content = f.read()
        encoded = base64.b64encode(content.encode()).decode()
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
        url = f"https://api.github.com/repos/{repo}/contents/portfolio.json"
        resp = _requests.get(url, headers=headers, timeout=10)
        sha = resp.json().get("sha", "") if resp.status_code == 200 else ""
        payload = {"message": "bot: refresh prices", "content": encoded, "branch": "main"}
        if sha:
            payload["sha"] = sha
        _requests.put(url, json=payload, headers=headers, timeout=10)
        logger.info("Portfolio prices committed to GitHub")
    except Exception as e:
        logger.debug("Portfolio GitHub commit failed: %s", e)


def _refresh_portfolio_prices() -> None:
    """Refresh portfolio.json with live prices from yfinance and commit to GitHub."""
    if not is_market_hours():
        return
    try:
        import risk_monitor
        risk_monitor.analyze_risk(update_prices=True)
        _commit_portfolio_to_github()
        logger.info("Portfolio prices refreshed")
    except Exception as e:
        logger.debug("Portfolio price refresh failed: %s", e)


def _check_portfolio_news() -> None:
    """Check for significant news on portfolio holdings."""
    try:
        import news_monitor
        items = news_monitor.fetch_portfolio_news(max_age_hours=2)
        significant = news_monitor.filter_significant(items)
        if significant:
            logger.info("Portfolio news: %d significant items found", len(significant))
            news_monitor.send_news_alerts(significant)
            # Flush news alerts to Telegram immediately (not part of scan batch)
            notifications.flush_telegram_batch()
        else:
            logger.debug("Portfolio news: nothing significant")
    except Exception as e:
        logger.debug("News monitor failed: %s", e)


def _check_earnings_warnings() -> None:
    """Warn about upcoming earnings for holdings."""
    try:
        import earnings_calendar
        warnings = earnings_calendar.check_earnings_warnings(days_ahead=7)
        if warnings:
            logger.info("Earnings warnings: %d positions reporting within 7 days", len(warnings))
            earnings_calendar.send_earnings_alert(warnings)
        else:
            logger.debug("No earnings in next 7 days")
    except Exception as e:
        logger.debug("Earnings calendar failed: %s", e)


def _check_sector_rotation() -> None:
    """Check for sector rotation signals."""
    try:
        import sector_rotation
        perf = sector_rotation.fetch_sector_performance()
        rotation = sector_rotation.detect_rotation(perf)
        if rotation.get("rotating_into") or rotation.get("rotating_out_of"):
            import json
            if os.path.exists("portfolio.json"):
                with open("portfolio.json") as _pf:
                    portfolio = json.load(_pf)
            else:
                portfolio = {}
            exposure = sector_rotation.get_portfolio_exposure(rotation, portfolio)
            if exposure.get("misaligned"):
                logger.info("Sector rotation detected: into %s, out of %s",
                            rotation.get("rotating_into"), rotation.get("rotating_out_of"))
                sector_rotation.send_alert(rotation, exposure)
    except Exception as e:
        logger.debug("Sector rotation check failed: %s", e)


def _sync_ib_portfolio() -> None:
    """Sync portfolio positions from Interactive Brokers (read-only, local TWS)."""
    if not is_market_hours():
        return
    try:
        import ib_sync
        if ib_sync.is_available():
            result = ib_sync.sync_portfolio()
            if result.get("synced"):
                logger.info("IB sync: %d positions, net liq $%.0f",
                            result.get("positions", 0), result.get("net_liq", 0))
            else:
                logger.debug("IB sync skipped: %s", result.get("error", "unavailable"))
    except Exception as e:
        logger.debug("IB sync failed: %s", e)


def _flex_sync_trades() -> None:
    """Auto-detect new IB trades via Flex Web Service (works without local TWS)."""
    try:
        import ib_flex
        if not ib_flex.is_configured():
            logger.info("IB Flex: skipped (IB_FLEX_TOKEN / IB_FLEX_QUERY_ID not set)")
            return
        new_trades = ib_flex.sync_new_trades(notify=True)
        if new_trades:
            logger.info("IB Flex: %d new trade(s) auto-synced", len(new_trades))
        else:
            logger.info("IB Flex: check complete — no new trades")
    except Exception as e:
        # Upgrade from debug → warning so failures are actually visible in logs
        logger.warning("IB Flex sync failed: %s", e)


def _send_weekly_report() -> None:
    """Generate and send weekly portfolio report."""
    try:
        import weekly_report
        report = weekly_report.generate_weekly_report()
        weekly_report.send_weekly_report(report)
        logger.info("Weekly report sent")
    except Exception as e:
        logger.error("Weekly report failed: %s", e)


def _run_weekly_optimization() -> None:
    """Run strategy optimization and watchlist curation weekly."""
    try:
        # Strategy optimization
        trades = strategy_optimizer.load_all_trades()
        if len(trades) >= 10:
            perf = strategy_optimizer.compute_signal_performance(trades)
            suggestions = strategy_optimizer.suggest_weight_adjustments(perf)
            if suggestions:
                optimized = strategy_optimizer.compute_optimized_weights(suggestions)
                strategy_optimizer.save_weights(optimized)
                logger.info("Strategy weights optimized: %d signals adjusted",
                            len(suggestions))
    except Exception as e:
        logger.debug("Strategy optimization failed: %s", e)

    try:
        # Watchlist curation
        wl = watchlist_curator.load_watchlist()
        if wl:
            additions = watchlist_curator.suggest_additions(
                [w["ticker"] for w in wl], {}, {})
            removals = watchlist_curator.suggest_removals(wl, {}, {}, {})
            if additions or removals:
                logger.info("Watchlist suggestions: +%d additions, -%d removals",
                            len(additions), len(removals))
    except Exception as e:
        logger.debug("Watchlist curation failed: %s", e)


def run_scheduled(paper_mode: bool = False) -> None:
    """Only run scan if within market hours."""
    if is_market_hours():
        run_scan(paper_mode=paper_mode)
    else:
        logger.debug("Outside market hours — skipping scan")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stock Investment Agent")
    parser.add_argument(
        "--schedule", action="store_true",
        help="Run on a recurring schedule during market hours",
    )
    parser.add_argument(
        "--ticker", type=str, default=None,
        help="Analyze a single ticker instead of the full watchlist",
    )
    parser.add_argument(
        "--threshold", type=int, default=None,
        help="Override signal threshold (default from .env)",
    )
    parser.add_argument(
        "--skip-checks", action="store_true",
        help="Skip startup health checks",
    )
    parser.add_argument(
        "--paper", action="store_true",
        help="Paper trading mode: auto-execute virtual trades instead of notifying",
    )
    parser.add_argument(
        "--briefing", action="store_true",
        help="Send daily briefing and exit",
    )
    parser.add_argument(
        "--india", action="store_true",
        help="Run dynamic Indian market scanner (NSE / BSE)",
    )
    args = parser.parse_args()

    if args.india:
        logger.info("Executing Dynamic Indian Market Scanner...")
        from indian_scanner_runner import IndianMarketScannerRunner
        runner = IndianMarketScannerRunner()
        runner.run_cycle(force=True)
        return

    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    # Initialize database
    if os.getenv("USE_SQLITE", "0") == "1":
        database.init()

    # Startup health check
    if not args.skip_checks:
        healthy = config_validator.run_all()
        if not healthy:
            logger.error("Health check failed — fix errors above or use --skip-checks")
            sys.exit(1)
        _run_health_check()

    if args.threshold is not None:
        os.environ["SIGNAL_THRESHOLD"] = str(args.threshold)

    if args.paper:
        logger.info("Paper trading mode enabled")
        # Ensure paper portfolio exists
        state = paper_trader.load_state()
        logger.info("Paper portfolio: $%.2f cash, %d open positions, %d closed trades",
                     state["cash"], len(state["open_positions"]), len(state["closed_trades"]))

    if args.briefing:
        logger.info("Generating daily briefing...")
        briefing = daily_briefing.generate_briefing()
        text = daily_briefing.format_briefing(briefing)
        print(text)
        if os.getenv("BRIEFING_AUTO_SEND", "0") == "1":
            daily_briefing.send_briefing(text)
        return

    if args.ticker:
        logger.info("Single-ticker mode: %s", args.ticker)
        scan_ticker(args.ticker, paper_mode=args.paper)
        if args.paper:
            paper_trader.print_status(paper_trader.load_state())
        return

    if args.schedule:
        interval = int(os.getenv("RUN_INTERVAL_MINUTES", "15"))
        logger.info(
            "Scheduled mode: running every %d minutes during market hours", interval
        )
        schedule.every(interval).minutes.do(run_scheduled, paper_mode=args.paper)

        # Pre-market daily briefing at 9:15 AM
        briefing_time = os.getenv("BRIEFING_TIME", "09:15")
        schedule.every().day.at(briefing_time).do(_send_daily_briefing)
        logger.info("Daily briefing scheduled at %s", briefing_time)

        # End-of-day report at 16:15 (Telegram only — email is in the digest)
        eod_time = os.getenv("EOD_REPORT_TIME", "16:15")
        schedule.every().day.at(eod_time).do(_send_eod_report)
        logger.info("EOD report scheduled at %s", eod_time)

        # Daily performance snapshot at 16:20 (after EOD report)
        schedule.every().day.at("16:20").do(_take_performance_snapshot)
        logger.info("Performance snapshot scheduled at 16:20")

        # ONE daily digest email at 16:25 — all alerts + briefing + EOD report combined
        schedule.every().day.at("16:25").do(_send_daily_digest)
        logger.info("Daily digest email scheduled at 16:25")

        # IB Flex trade auto-detection every 15 min (works without local TWS)
        schedule.every(15).minutes.do(_flex_sync_trades)
        logger.info("IB Flex trade auto-sync scheduled every 15 minutes")

        # Health check every 4 hours
        schedule.every(4).hours.do(_run_health_check)

        # Auto-refresh portfolio prices every 15 min during market hours
        schedule.every(15).minutes.do(_refresh_portfolio_prices)
        logger.info("Portfolio price refresh scheduled every 15 minutes")

        # News monitor for holdings every hour
        schedule.every(1).hours.do(_check_portfolio_news)
        logger.info("Portfolio news monitor scheduled every hour")

        # Earnings warnings every morning at 8:30 AM
        schedule.every().day.at("08:30").do(_check_earnings_warnings)
        logger.info("Earnings calendar check scheduled at 08:30")

        # Sector rotation check once a day at 10:00 AM
        schedule.every().day.at("10:00").do(_check_sector_rotation)
        logger.info("Sector rotation check scheduled at 10:00")

        # IB sync every 15 min during market hours
        schedule.every(15).minutes.do(_sync_ib_portfolio)
        logger.info("IB portfolio sync scheduled every 15 minutes")

        # Weekly report on Sundays at 8:00 PM
        schedule.every().sunday.at("20:00").do(_send_weekly_report)
        logger.info("Weekly report scheduled for Sundays at 20:00")

        # Weekly optimization on Sundays
        schedule.every().sunday.at("18:00").do(_run_weekly_optimization)
        logger.info("Weekly optimization scheduled for Sundays at 18:00")

        # Run once immediately
        run_scheduled(paper_mode=args.paper)

        while True:
            schedule.run_pending()
            time.sleep(30)
    else:
        # Single run
        run_scan(paper_mode=args.paper)
        if args.paper:
            paper_trader.print_status(paper_trader.load_state())


if __name__ == "__main__":
    main()
