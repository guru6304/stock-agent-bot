"""Master Indian Market Scanner and Paper Signal Runner.

Coordinates dynamic instrument discovery, market regime detection, multi-horizon
signal generation, durable deduplication, and Telegram paper dispatch.
Signal-only paper-observation system. Zero broker orders.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import schedule
from dotenv import load_dotenv

from india_market_config import (
    DEFAULT_SCANNER_CONFIG,
    assert_broker_orders_prohibited,
    is_market_hours,
    now_ist,
)
from indian_market_regime import detect_indian_market_regime
from indian_signals import IndianSignalEngine, IndianTradeSignal, SignalHorizon
from instrument_discovery import DynamicInstrumentDiscovery, InstrumentMetadata
from market_data_service import MarketDataService
from signal_deduplication import DurableSignalDeduplicator
from telegram_signal_dispatcher import TelegramSignalDispatcher, format_signal_card

load_dotenv()

# Ensure Windows console supports UTF-8 (including ₹ Rupee symbol)
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE = LOGS_DIR / "indian_scan_summary.json"

LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)-22s  %(message)s"
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/indian_scanner.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("indian-scanner-runner")


class IndianMarketScannerRunner:
    """End-to-end scanner discovering and evaluating Indian equity and F&O instruments."""

    def __init__(
        self,
        config=None,
        discovery: Optional[DynamicInstrumentDiscovery] = None,
        data_service: Optional[MarketDataService] = None,
        deduplicator: Optional[DurableSignalDeduplicator] = None,
        dispatcher: Optional[TelegramSignalDispatcher] = None,
    ):
        # Assert safety barrier on startup
        assert_broker_orders_prohibited()

        self.config = config or DEFAULT_SCANNER_CONFIG
        self.discovery = discovery or DynamicInstrumentDiscovery()
        self.data_service = data_service or MarketDataService(config=self.config)
        self.deduplicator = deduplicator or DurableSignalDeduplicator(
            cooldown_hours=self.config.signal_dedup_hours
        )
        self.dispatcher = dispatcher or TelegramSignalDispatcher()
        self.signal_engine = IndianSignalEngine(config=self.config)

    def run_cycle(
        self,
        limit: Optional[int] = None,
        force: bool = False,
        dry_run: bool = False,
    ) -> Dict:
        """Run a single complete discovery, scan, evaluation, and dispatch cycle."""
        assert_broker_orders_prohibited()

        ts_start = now_ist()
        logger.info("=" * 65)
        logger.info("  INDIAN MARKET SCAN CYCLE — %s", ts_start.strftime("%Y-%m-%d %H:%M:%S IST"))
        logger.info("=" * 65)

        # 1. Market Hours Check
        in_session = is_market_hours(ts_start)
        if not in_session and not force:
            logger.info("Outside normal NSE trading hours (09:15-15:30 IST). Skipping scan (use --force to override).")
            return {
                "status": "skipped_outside_market_hours",
                "timestamp_ist": ts_start.isoformat(),
            }

        # 2. Dynamic Discovery
        logger.info("Step 1: Running dynamic instrument discovery...")
        try:
            disc_stats = self.discovery.discover()
            logger.info(
                "Discovery stats: %d received, %d valid, %d eligible equity, %d eligible F&O",
                disc_stats.records_received,
                disc_stats.records_valid,
                disc_stats.eligible_equity,
                disc_stats.eligible_fno,
            )
        except Exception as e:
            logger.error("Dynamic discovery failed: %s", e)
            return {
                "status": "error_discovery_failed",
                "error": str(e),
                "timestamp_ist": ts_start.isoformat(),
            }

        # 3. Market Regime
        logger.info("Step 2: Detecting Indian market regime...")
        regime_result = detect_indian_market_regime()
        logger.info(
            "Regime: %s (Confidence: %d%%) | Nifty: %s | India VIX: %s",
            regime_result.regime,
            regime_result.confidence,
            f"₹{regime_result.nifty_price:,.2f}" if regime_result.nifty_price else "N/A",
            f"{regime_result.india_vix:.2f}" if regime_result.india_vix else "N/A",
        )

        # 4. Filter & Select Universe
        scan_limit = limit or self.config.max_scan_universe
        candidates = self.discovery.get_equity_universe(limit=scan_limit)
        if not candidates:
            logger.warning("No eligible equity candidates discovered from instrument master.")
            return {
                "status": "no_candidates_discovered",
                "timestamp_ist": ts_start.isoformat(),
            }

        logger.info("Step 3: Scanning %d dynamically discovered instruments...", len(candidates))

        # 5. Scan & Evaluate
        signals_found: List[IndianTradeSignal] = []
        scanned_count = 0
        error_count = 0

        for idx, inst in enumerate(candidates, 1):
            try:
                # Fetch daily candles
                candles = self.data_service.get_candle_history(inst, period="1y", interval="1d")
                if not candles or candles.df.empty:
                    continue

                scanned_count += 1
                # Optional live quote probe
                live_q = self.data_service.get_live_quote(inst)

                # Evaluate equity horizons
                eq_signals = self.signal_engine.evaluate_equity_horizons(inst, candles, live_q)

                for sig in eq_signals:
                    signals_found.append(sig)

                    # If equity conviction exists, also check for dynamic F&O option contract
                    fno_contracts = self.discovery.get_fno_contracts_for_underlying(inst.symbol)
                    if fno_contracts:
                        underlying_spot = live_q.price if live_q else float(candles.df["Close"].iloc[-1])
                        fno_sig = self.signal_engine.evaluate_intraday_fno(sig, fno_contracts, underlying_spot)
                        if fno_sig:
                            signals_found.append(fno_sig)

            except Exception as e:
                error_count += 1
                logger.debug("Error scanning %s: %s", inst.symbol, e)

        logger.info("Step 4: Scan completed (%d scanned, %d errors). %d total signals generated.", scanned_count, error_count, len(signals_found))

        # 6. Deduplication & Telegram Dispatch
        emitted_signals: List[IndianTradeSignal] = []
        suppressed_signals: List[str] = []

        for sig in signals_found:
            if self.deduplicator.is_duplicate(sig):
                suppressed_signals.append(f"{sig.symbol} ({sig.horizon.value} - duplicate cooldown)")
                logger.info("Suppressed duplicate signal for %s (%s)", sig.symbol, sig.horizon.value)
                continue

            # Record in deduplicator
            self.deduplicator.record_emission(sig)
            emitted_signals.append(sig)

            # Telegram Dispatch
            if not dry_run:
                sent, msg = self.dispatcher.send_signal(sig)
                if sent:
                    logger.info("Dispatched %s signal to Telegram: %s", sig.direction, sig.symbol)
                else:
                    logger.warning("Telegram dispatch skipped/failed for %s: %s", sig.symbol, msg)
            else:
                logger.info("[DRY RUN] Signal generated:\n%s", format_signal_card(sig))

        # 7. Persist Observability Summary
        cycle_duration = (now_ist() - ts_start).total_seconds()
        summary = {
            "cycle_start_ist": ts_start.isoformat(),
            "duration_seconds": round(cycle_duration, 2),
            "data_source_live_ready": self.data_service.is_live_ready,
            "live_blocker": self.data_service.live_blocker_reason,
            "discovery_records_received": disc_stats.records_received,
            "discovery_records_valid": disc_stats.records_valid,
            "eligible_equity_universe": disc_stats.eligible_equity,
            "eligible_fno_universe": disc_stats.eligible_fno,
            "instruments_scanned": scanned_count,
            "market_regime": regime_result.to_dict(),
            "signals_generated": len(signals_found),
            "signals_emitted": len(emitted_signals),
            "signals_suppressed_duplicates": len(suppressed_signals),
            "emitted_signal_ids": [s.signal_id for s in emitted_signals],
        }

        SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("Cycle finished in %.1fs. Summary written to %s\n", cycle_duration, SUMMARY_FILE)
        return summary


def main():
    parser = argparse.ArgumentParser(description="Indian Market Dynamic Scanner & Paper Signals")
    parser.add_argument("--scan-once", action="store_true", help="Run a single scan cycle and exit")
    parser.add_argument("--schedule", action="store_true", help="Run on schedule during market hours")
    parser.add_argument("--limit", type=int, default=None, help="Max dynamically discovered instruments to scan")
    parser.add_argument("--force", action="store_true", help="Force scan even if outside market hours")
    parser.add_argument("--dry-run", action="store_true", help="Print signals to console without Telegram delivery")
    args = parser.parse_args()

    runner = IndianMarketScannerRunner()

    if args.scan_once or not args.schedule:
        runner.run_cycle(limit=args.limit, force=args.force, dry_run=args.dry_run)
    elif args.schedule:
        interval = DEFAULT_SCANNER_CONFIG.scan_interval_minutes
        logger.info("Scheduling Indian scanner every %d minutes during market hours...", interval)
        schedule.every(interval).minutes.do(
            runner.run_cycle,
            limit=args.limit,
            force=args.force,
            dry_run=args.dry_run,
        )
        # Immediate first run
        runner.run_cycle(limit=args.limit, force=args.force, dry_run=args.dry_run)
        while True:
            schedule.run_pending()
            time.sleep(1)


if __name__ == "__main__":
    main()
