#!/usr/bin/env python3
"""
Minimal web server + background runner for cloud deployment (Render, Railway, Fly.io).

Runs the Telegram bot and agent scheduler as background threads,
while serving a health endpoint to keep the cloud service alive.

Usage:
    python3 web_server.py
"""

import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-20s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("web-server")

# Pre-import heavy dependencies sequentially on the main thread to prevent
# multithreaded import race conditions in CPython / pandas / requests
logger.info("Pre-loading core dependencies on main thread...")
import requests
import pandas as pd
import numpy as np
import schedule as sched_lib
import telegram_bot
import agent
logger.info("Core dependencies pre-loaded successfully.")

# Track service health
_status = {
    "started_at": datetime.now().isoformat(),
    "telegram_bot": "starting",
    "scheduler": "starting",
    "last_health_check": None,
}


class HealthHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for health checks."""

    def do_GET(self):
        _status["last_health_check"] = datetime.now().isoformat()

        if self.path == "/health" or self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "ok",
                "service": "stock-agent",
                "uptime_since": _status["started_at"],
                "telegram_bot": _status["telegram_bot"],
                "scheduler": _status["scheduler"],
                "checked_at": _status["last_health_check"],
            }).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress default request logging (too noisy with pings)."""
        pass


def run_telegram_bot():
    """Run the Telegram bot in a background thread."""
    while True:
        try:
            _status["telegram_bot"] = "running"
            logger.info("Starting Telegram bot thread...")
            telegram_bot.run_bot()
        except Exception as e:
            _status["telegram_bot"] = "error: %s" % str(e)
            logger.error("Telegram bot crashed: %s", e, exc_info=True)
            time.sleep(15)


def run_scheduler():
    """Run the agent scheduler in a background thread."""
    while True:
        try:
            _status["scheduler"] = "running"
            logger.info("Starting scheduler thread...")

            os.makedirs("logs", exist_ok=True)

            interval = int(os.getenv("RUN_INTERVAL_MINUTES", "15"))
            logger.info("Scheduler: scanning every %d minutes", interval)

            sched_lib.clear()
            sched_lib.every(interval).minutes.do(agent.run_scheduled, paper_mode=False)

            # Daily briefing
            briefing_time = os.getenv("BRIEFING_TIME", "09:15")
            sched_lib.every().day.at(briefing_time).do(agent._send_daily_briefing)

            # EOD report
            eod_time = os.getenv("EOD_REPORT_TIME", "16:15")
            sched_lib.every().day.at(eod_time).do(agent._send_eod_report)

            # Performance snapshot
            sched_lib.every().day.at("16:20").do(agent._take_performance_snapshot)

            # Health check every 4 hours
            sched_lib.every(4).hours.do(agent._run_health_check)

            # Portfolio price refresh every 15 min
            sched_lib.every(15).minutes.do(agent._refresh_portfolio_prices)

            # News monitor every hour
            sched_lib.every(1).hours.do(agent._check_portfolio_news)

            # Earnings warnings daily at 8:30
            sched_lib.every().day.at("08:30").do(agent._check_earnings_warnings)

            # Sector rotation daily at 10:00
            sched_lib.every().day.at("10:00").do(agent._check_sector_rotation)

            # Weekly report Sundays at 20:00
            sched_lib.every().sunday.at("20:00").do(agent._send_weekly_report)

            # Weekly optimization Sundays at 18:00
            sched_lib.every().sunday.at("18:00").do(agent._run_weekly_optimization)

            # Run once safely on boot if market is open
            try:
                agent.run_scheduled(paper_mode=False)
            except Exception as se:
                logger.error("Initial scheduled scan encountered error: %s", se)

            while True:
                sched_lib.run_pending()
                time.sleep(30)

        except Exception as e:
            _status["scheduler"] = "error: %s" % str(e)
            logger.error("Scheduler crashed: %s", e, exc_info=True)
            time.sleep(30)


def main():
    port = int(os.getenv("PORT", "10000"))

    logger.info("=" * 50)
    logger.info("  STOCK AGENT — Cloud Server Starting")
    logger.info("  Health endpoint: http://0.0.0.0:%d/health", port)
    logger.info("=" * 50)

    # Bind port immediately so Render health check succeeds on the very first ping
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    logger.info("Health server listening on port %d", port)

    # Start Telegram bot in background thread
    bot_thread = threading.Thread(target=run_telegram_bot, daemon=True)
    bot_thread.start()
    logger.info("Telegram bot thread started")

    # Start scheduler in background thread
    sched_thread = threading.Thread(target=run_scheduler, daemon=True)
    sched_thread.start()
    logger.info("Scheduler thread started")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
