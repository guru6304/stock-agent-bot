"""Automated Test Suite for Dynamic Indian Market Scanner System.

Compatible with standard library `unittest` and `pytest`.
Strictly synthetic fixtures only. No real production stock symbols used as defaults.
"""

import os
import sqlite3
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytz

from india_market_config import (
    CURRENCY_SYMBOL,
    TIMEZONE_NAME,
    assert_broker_orders_prohibited,
    is_market_hours,
    now_ist,
)
from indian_market_regime import detect_indian_market_regime
from indian_signals import (
    IndianSignalEngine,
    IndianTradeSignal,
    SignalHorizon,
    calculate_technicals,
)
from instrument_discovery import (
    DiscoveryError,
    DiscoveryStats,
    DynamicInstrumentDiscovery,
    InstrumentMetadata,
)
from market_data_service import CandleData, MarketDataService
from signal_deduplication import DurableSignalDeduplicator
from telegram_signal_dispatcher import format_signal_card


def get_synthetic_equity_inst() -> InstrumentMetadata:
    return InstrumentMetadata(
        token="990001",
        symbol="SYNTH_ALPHA",
        raw_symbol="SYNTH_ALPHA-EQ",
        name="SYNTHETIC ALPHA LTD",
        expiry=None,
        strike=0.0,
        lotsize=1,
        instrument_type="",
        exch_seg="NSE",
        tick_size=0.05,
        is_fno=False,
    )


def get_synthetic_fno_contracts() -> list[InstrumentMetadata]:
    fut_date = date.today() + timedelta(days=14)
    return [
        InstrumentMetadata(
            token="880101",
            symbol="SYNTH_ALPHA",
            raw_symbol="SYNTH_ALPHA24MAR500CE",
            name="SYNTH_ALPHA",
            expiry=fut_date,
            strike=500.0,
            lotsize=100,
            instrument_type="OPTSTK",
            exch_seg="NFO",
            tick_size=0.05,
            is_fno=True,
        ),
        InstrumentMetadata(
            token="880102",
            symbol="SYNTH_ALPHA",
            raw_symbol="SYNTH_ALPHA24MAR520CE",
            name="SYNTH_ALPHA",
            expiry=fut_date,
            strike=520.0,
            lotsize=100,
            instrument_type="OPTSTK",
            exch_seg="NFO",
            tick_size=0.05,
            is_fno=True,
        ),
        InstrumentMetadata(
            token="880103",
            symbol="SYNTH_ALPHA",
            raw_symbol="SYNTH_ALPHA24MAR500PE",
            name="SYNTH_ALPHA",
            expiry=fut_date,
            strike=500.0,
            lotsize=100,
            instrument_type="OPTSTK",
            exch_seg="NFO",
            tick_size=0.05,
            is_fno=True,
        ),
    ]


def get_synthetic_bullish_candles() -> CandleData:
    dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
    base_price = 450.0
    prices = [base_price + i * 0.7 for i in range(100)]
    prices[-1] = prices[-2] + 15.0  # Breakout

    df = pd.DataFrame({
        "Open": [p - 1.0 for p in prices],
        "High": [p + 3.0 for p in prices],
        "Low": [p - 2.0 for p in prices],
        "Close": prices,
        "Volume": [100000.0] * 99 + [350000.0],
    }, index=dates)

    return CandleData(
        df=df,
        symbol="SYNTH_ALPHA",
        token="990001",
        interval="1d",
        data_source="Synthetic Test Source",
        is_live=True,
        data_timestamp="2026-09-17 15:30:00",
        is_candle_complete=True,
    )


class TestIndianMarketSuite(unittest.TestCase):
    """Full automated test suite for the Indian market scanner and paper signal pipeline."""

    def test_timezone_and_currency(self):
        self.assertEqual(TIMEZONE_NAME, "Asia/Kolkata")
        self.assertEqual(CURRENCY_SYMBOL, "₹")
        ist_time = now_ist()
        self.assertEqual(ist_time.tzinfo.zone, "Asia/Kolkata")

    def test_safety_order_execution_barrier(self):
        assert_broker_orders_prohibited()
        import india_market_config
        with patch.object(india_market_config, "ORDER_EXECUTION_ENABLED", True):
            with self.assertRaises(RuntimeError):
                assert_broker_orders_prohibited()

    def test_market_hours_validation(self):
        ist = pytz.timezone("Asia/Kolkata")
        wed_open = ist.localize(datetime(2026, 9, 16, 11, 0))
        self.assertTrue(is_market_hours(wed_open))

        wed_closed = ist.localize(datetime(2026, 9, 16, 20, 0))
        self.assertFalse(is_market_hours(wed_closed))

        sun_closed = ist.localize(datetime(2026, 9, 20, 11, 0))
        self.assertFalse(is_market_hours(sun_closed))

    def test_successful_parsing_and_categorization(self):
        future_date_str = (date.today() + timedelta(days=20)).strftime("%d%b%Y").upper()
        synthetic_records = [
            {
                "token": "1001",
                "symbol": "INST_ONE-EQ",
                "name": "INSTITUTION ONE",
                "expiry": "",
                "strike": "0",
                "lotsize": "1",
                "instrumenttype": "",
                "exch_seg": "NSE",
                "tick_size": "5.0",
            },
            {
                "token": "2001",
                "symbol": "INST_ONE" + future_date_str + "500CE",
                "name": "INST_ONE",
                "expiry": future_date_str,
                "strike": "50000.0",
                "lotsize": "250",
                "instrumenttype": "OPTSTK",
                "exch_seg": "NFO",
                "tick_size": "5.0",
            },
            {
                "token": "2002",
                "symbol": "INST_ONE20JAN2020500CE",
                "name": "INST_ONE",
                "expiry": "20JAN2020",
                "strike": "50000.0",
                "lotsize": "250",
                "instrumenttype": "OPTSTK",
                "exch_seg": "NFO",
                "tick_size": "5.0",
            },
            {
                "token": "3001",
                "symbol": "US_STOCK",
                "name": "US EQUITY",
                "expiry": "",
                "strike": "0",
                "lotsize": "1",
                "instrumenttype": "",
                "exch_seg": "NYSE",
                "tick_size": "1.0",
            },
            {
                "token": "",
                "symbol": "NO_TOKEN-EQ",
                "name": "NO TOKEN",
                "expiry": "",
                "strike": "0",
                "lotsize": "1",
                "instrumenttype": "",
                "exch_seg": "NSE",
                "tick_size": "5.0",
            },
        ]

        disc = DynamicInstrumentDiscovery()
        with patch.object(disc, "fetch_raw_master", return_value=synthetic_records):
            stats = disc.discover()
            self.assertEqual(stats.records_received, 5)
            self.assertEqual(stats.eligible_equity, 1)
            self.assertEqual(stats.eligible_fno, 1)
            self.assertEqual(stats.rejected_reasons.get("fno_expired", 0), 1)
            self.assertEqual(stats.rejected_reasons.get("unsupported_segment", 0), 1)

            eq_univ = disc.get_equity_universe()
            self.assertEqual(len(eq_univ), 1)
            self.assertEqual(eq_univ[0].symbol, "INST_ONE")
            self.assertEqual(eq_univ[0].raw_symbol, "INST_ONE-EQ")

    def test_no_hardcoded_stock_fallback_on_failure(self):
        disc = DynamicInstrumentDiscovery(master_url="http://invalid-url.local/master.json")
        with patch("instrument_discovery.CACHE_FILE", Path("/tmp/non_existent_cache_file_xyz.json")):
            with patch.object(disc.http, "get", side_effect=Exception("Connection refused")):
                with self.assertRaises(DiscoveryError):
                    disc.fetch_raw_master(force_refresh=True)

    def test_short_term_momentum_breakout(self):
        inst = get_synthetic_equity_inst()
        candles = get_synthetic_bullish_candles()
        engine = IndianSignalEngine()
        signals = engine.evaluate_equity_horizons(inst, candles)
        self.assertGreaterEqual(len(signals), 1)

        st_sig = next((s for s in signals if s.horizon == SignalHorizon.SHORT_TERM_EQUITY), None)
        self.assertIsNotNone(st_sig)
        self.assertEqual(st_sig.direction, "BUY")
        self.assertEqual(st_sig.symbol, "SYNTH_ALPHA")
        self.assertLess(st_sig.stop_loss, st_sig.entry_range_low)
        self.assertGreater(st_sig.target_1, st_sig.entry_range_high)
        self.assertGreaterEqual(st_sig.risk_reward_ratio, 1.5)

    def test_bearish_short_setup_signal_labeling(self):
        inst = get_synthetic_equity_inst()
        dates = pd.date_range(end=datetime.now(), periods=60, freq="D")
        prices = [500.0 - i * 1.5 for i in range(60)]
        prices[-1] = 390.0
        df = pd.DataFrame({
            "Open": prices,
            "High": [p + 1.0 for p in prices],
            "Low": [p - 2.0 for p in prices],
            "Close": prices,
            "Volume": [100000.0] * 60,
        }, index=dates)

        candles = CandleData(
            df=df,
            symbol="SYNTH_ALPHA",
            token="990001",
            interval="1d",
            data_source="Synthetic Test Source",
            is_live=True,
            data_timestamp="2026-09-17 15:30:00",
            is_candle_complete=True,
        )

        engine = IndianSignalEngine()
        signals = engine.evaluate_equity_horizons(inst, candles)
        bearish_sig = next((s for s in signals if s.direction == "BEARISH_SHORT_SETUP"), None)
        self.assertIsNotNone(bearish_sig)
        self.assertIn("BEARISH_SHORT_SETUP", bearish_sig.direction)
        self.assertGreater(bearish_sig.stop_loss, bearish_sig.entry_range_high)
        self.assertLess(bearish_sig.target_1, bearish_sig.entry_range_low)

    def test_dynamic_fno_option_matching(self):
        inst = get_synthetic_equity_inst()
        candles = get_synthetic_bullish_candles()
        fno_contracts = get_synthetic_fno_contracts()
        engine = IndianSignalEngine()
        eq_signals = engine.evaluate_equity_horizons(inst, candles)
        buy_signal = next(s for s in eq_signals if s.direction == "BUY")

        spot_price = 510.0
        fno_signal = engine.evaluate_intraday_fno(buy_signal, fno_contracts, spot_price)
        self.assertIsNotNone(fno_signal)
        self.assertEqual(fno_signal.horizon, SignalHorizon.INTRADAY_FNO)
        self.assertEqual(fno_signal.exchange, "NFO")
        self.assertIsNotNone(fno_signal.fno_details)
        self.assertEqual(fno_signal.fno_details["option_type"], "CE")
        self.assertIn(fno_signal.fno_details["strike"], (500.0, 520.0))
        self.assertEqual(fno_signal.fno_details["lot_size"], 100)

    def test_dedup_persistence_across_instances(self):
        inst = get_synthetic_equity_inst()
        candles = get_synthetic_bullish_candles()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_file = Path(tmpdir) / "test_dedup.db"
            dedup1 = DurableSignalDeduplicator(db_path=db_file, cooldown_hours=4.0)

            engine = IndianSignalEngine()
            signals = engine.evaluate_equity_horizons(inst, candles)
            sig = signals[0]

            self.assertFalse(dedup1.is_duplicate(sig))
            dedup1.record_emission(sig)
            self.assertTrue(dedup1.is_duplicate(sig))

            dedup2 = DurableSignalDeduplicator(db_path=db_file, cooldown_hours=4.0)
            self.assertTrue(dedup2.is_duplicate(sig))

    def test_signal_card_fields_and_disclaimer(self):
        inst = get_synthetic_equity_inst()
        candles = get_synthetic_bullish_candles()
        engine = IndianSignalEngine()
        signals = engine.evaluate_equity_horizons(inst, candles)
        card = format_signal_card(signals[0])

        self.assertIn("INDIA PAPER SIGNAL — NOT A TRADE", card)
        self.assertIn("Signal ID:", card)
        self.assertIn("SYNTH_ALPHA", card)
        self.assertIn(CURRENCY_SYMBOL, card)
        self.assertIn("Zero broker orders placed", card)

    def test_regime_detection(self):
        dates = pd.date_range(end=datetime.now(), periods=250, freq="D")
        prices = [18000.0 + i * 20.0 for i in range(250)]
        nifty_df = pd.DataFrame({"Close": prices}, index=dates)
        vix_df = pd.DataFrame({"Close": [13.5] * 250}, index=dates)

        res = detect_indian_market_regime(nifty_df, vix_df)
        self.assertIn(res.regime, ("BULL_STRONG", "BULL_WEAK"))
        self.assertTrue(res.benchmark_verified)
        self.assertEqual(res.india_vix, 13.5)

    def test_regime_fallback_on_missing_data(self):
        res = detect_indian_market_regime(nifty_df=pd.DataFrame(), vix_df=pd.DataFrame())
        self.assertEqual(res.regime, "NEUTRAL")
        self.assertFalse(res.benchmark_verified)
        self.assertIn("unavailable", res.blocker)

    def test_ticker_resolution_and_screener_universe(self):
        import data_layer
        import screener
        self.assertTrue(data_layer.is_indian_market())
        self.assertEqual(data_layer.resolve_ticker("RELIANCE"), "RELIANCE.NS")
        self.assertEqual(data_layer.resolve_ticker("TCS"), "TCS.NS")
        self.assertEqual(data_layer.resolve_ticker("^NSEI"), "^NSEI")

        universe = screener.get_universe("default")
        self.assertIn("TCS", universe)
        self.assertNotIn("AAPL", universe)
        self.assertNotIn("TSLA", universe)
        self.assertTrue(len(universe) >= 20)

    def test_trade_advisor_ticker_extraction(self):
        import trade_advisor
        self.assertIsNone(trade_advisor.extract_ticker("Which stock should I buy?"))
        self.assertIsNone(trade_advisor.extract_ticker("What are the best stocks today?"))
        self.assertEqual(trade_advisor.extract_ticker("Should I buy RELIANCE?"), "RELIANCE")
        self.assertEqual(trade_advisor.extract_ticker("Check HDFCBANK target"), "HDFCBANK")
        self.assertEqual(trade_advisor.extract_ticker("TATAMOTORS swing"), "TATAMOTORS")

    def test_telegram_bot_indian_routing(self):
        import telegram_bot
        self.assertTrue(telegram_bot.is_indian_market())
        self.assertEqual(telegram_bot.get_currency_symbol(), "₹")

        help_text = telegram_bot.cmd_help()
        self.assertIn("RELIANCE", help_text)
        self.assertNotIn("TSLA", help_text)

        status_text = telegram_bot.cmd_status()
        self.assertIn("₹", status_text)


if __name__ == "__main__":
    unittest.main()
