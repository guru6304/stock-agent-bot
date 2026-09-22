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
    detect_fair_value_gap,
    detect_liquidity_sweep,
    detect_order_block_mss,
    detect_volatility_contraction,
    detect_wyckoff_volume_absorption,
    detect_institutional_relative_strength,
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

    def test_nfo_strike_scaling_idfc_and_petronet(self):
        """Verify Angel One NFO strike scaling for stocks under ₹1000."""
        disc = DynamicInstrumentDiscovery()
        stats = DiscoveryStats()
        today = date.today()

        # IDFCFIRSTB ₹72 strike (stored in master as 7200.0)
        raw_idfc = {
            "token": "113042",
            "symbol": "IDFCFIRSTB29SEP2672CE",
            "name": "IDFCFIRSTB",
            "expiry": (today + timedelta(days=7)).strftime("%d%b%Y").upper(),
            "strike": "7200.000000",
            "lotsize": "9275",
            "instrumenttype": "OPTSTK",
            "exch_seg": "NFO",
            "tick_size": "5.0",
        }
        inst_idfc = disc.parse_record(raw_idfc, stats, today)
        self.assertIsNotNone(inst_idfc)
        self.assertEqual(inst_idfc.strike, 72.0)

        # PETRONET ₹240 strike (stored in master as 24000.0)
        raw_petro = {
            "token": "136111",
            "symbol": "PETRONET29SEP26240CE",
            "name": "PETRONET",
            "expiry": (today + timedelta(days=7)).strftime("%d%b%Y").upper(),
            "strike": "24000.000000",
            "lotsize": "1900",
            "instrumenttype": "OPTSTK",
            "exch_seg": "NFO",
            "tick_size": "5.0",
        }
        inst_petro = disc.parse_record(raw_petro, stats, today)
        self.assertIsNotNone(inst_petro)
        self.assertEqual(inst_petro.strike, 240.0)

        # MARUTI ₹12200 strike (stored in master as 1220000.0)
        raw_maruti = {
            "token": "126419",
            "symbol": "MARUTI29SEP2612200CE",
            "name": "MARUTI",
            "expiry": (today + timedelta(days=7)).strftime("%d%b%Y").upper(),
            "strike": "1220000.000000",
            "lotsize": "50",
            "instrumenttype": "OPTSTK",
            "exch_seg": "NFO",
            "tick_size": "5.0",
        }
        inst_maruti = disc.parse_record(raw_maruti, stats, today)
        self.assertIsNotNone(inst_maruti)
        self.assertEqual(inst_maruti.strike, 12200.0)

    def test_symbol_level_deduplication_across_horizons(self):
        """Verify that emitting a signal for a symbol suppresses different horizons for the same symbol within cooldown."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_file = Path(tmpdir) / "test_sym_dedup.db"
            dedup = DurableSignalDeduplicator(db_path=db_file, cooldown_hours=4.0)

            sig_swing = IndianTradeSignal(
                signal_id="SIG-TEST-001",
                horizon=SignalHorizon.SWING_EQUITY,
                strategy_name="Oversold Mean-Reversion Value Setup",
                strategy_version="1.0.0",
                symbol="OFSS",
                exchange="NSE",
                token="10738",
                direction="BUY",
                signal_time_ist="2026-09-22 09:36:23 IST",
                data_source="yfinance",
                data_timestamp="2026-09-22 00:00:00",
                entry_range_low=10800.0,
                entry_range_high=11000.0,
                stop_loss=10400.0,
                target_1=11600.0,
                target_2=12000.0,
                risk_reward_ratio=1.5,
                validity_period="1 to 3 weeks",
                thesis="Oversold bounce",
                quality_score=7.8,
            )

            # First emission succeeds
            self.assertFalse(dedup.is_duplicate(sig_swing))
            dedup.record_emission(sig_swing)
            self.assertTrue(dedup.is_duplicate(sig_swing))

            # A subsequent signal for the same symbol under a DIFFERENT horizon is suppressed
            sig_short_term = IndianTradeSignal(
                signal_id="SIG-TEST-002",
                horizon=SignalHorizon.SHORT_TERM_EQUITY,
                strategy_name="Multi-Day High Momentum Breakout",
                strategy_version="1.0.0",
                symbol="OFSS",
                exchange="NSE",
                token="10738",
                direction="BUY",
                signal_time_ist="2026-09-22 09:51:00 IST",
                data_source="yfinance",
                data_timestamp="2026-09-22 00:00:00",
                entry_range_low=10800.0,
                entry_range_high=11000.0,
                stop_loss=10400.0,
                target_1=11600.0,
                target_2=12000.0,
                risk_reward_ratio=1.5,
                validity_period="1 to 5 days",
                thesis="Breakout",
                quality_score=8.0,
            )
            self.assertTrue(dedup.is_duplicate(sig_short_term))

    def test_fno_details_consolidated_on_equity_card(self):
        """Verify that F&O details format cleanly directly on an equity trade card."""
        sig = IndianTradeSignal(
            signal_id="SIG-TEST-CONSOLIDATED",
            horizon=SignalHorizon.SWING_EQUITY,
            strategy_name="Oversold Mean-Reversion Value Setup",
            strategy_version="1.0.0",
            symbol="OFSS",
            exchange="NSE",
            token="10738",
            direction="BUY",
            signal_time_ist="2026-09-22 09:36:23 IST",
            data_source="yfinance",
            data_timestamp="2026-09-22 00:00:00",
            entry_range_low=10869.38,
            entry_range_high=10978.62,
            stop_loss=10414.43,
            target_1=11688.35,
            target_2=12197.92,
            risk_reward_ratio=1.5,
            validity_period="1 to 3 weeks",
            thesis="Deep oversold condition presenting favorable mean-reversion risk/reward near support.",
            quality_score=7.8,
            fno_details={
                "underlying_symbol": "OFSS",
                "underlying_spot": 10924.0,
                "contract_symbol": "OFSS29SEP2610900CE",
                "contract_token": "98626",
                "expiry": "2026-09-29",
                "strike": 10900.0,
                "option_type": "CE",
                "lot_size": 100,
            },
        )
        card = format_signal_card(sig)
        self.assertIn("OFSS", card)
        self.assertIn("SWING_EQUITY", card)
        self.assertIn("Validated Option Contract:", card)
        self.assertIn("OFSS29SEP2610900CE", card)
        self.assertIn("10,900.00", card)
        self.assertIn("CE", card)

    def test_contradictory_signal_suppression(self):
        """Verify that oversold mean reversion does not fire when price is in breakdown."""
        inst = get_synthetic_equity_inst()
        dates = pd.date_range(end=datetime.now(), periods=60, freq="D")
        # Steady drop creating deep oversold and new 20d low breakdown
        prices = [500.0 - i * 3.0 for i in range(60)]
        df = pd.DataFrame({
            "Open": prices,
            "High": [p + 0.5 for p in prices],
            "Low": [p - 1.0 for p in prices],
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
        # Should NOT contain BUY (Oversold) because price is breaking down
        directions = {s.direction for s in signals}
        self.assertNotIn("BUY", directions)

    def test_detect_liquidity_sweep_turtle_soup(self):
        """Verify institutional liquidity sweep (stop hunt) detection."""
        dates = pd.date_range(end=datetime.now(), periods=20, freq="D")
        # 19 days oscillating above 100
        opens = [105.0] * 19
        highs = [110.0] * 19
        lows = [100.0] * 19
        closes = [106.0] * 19

        # 20th day: sweeps below 100 to 95, then rallies and closes at 104
        # Range: 95 to 105 (10 pts). Lower wick: min(102, 104) - 95 = 7 pts (70%)
        opens.append(102.0)
        highs.append(105.0)
        lows.append(95.0)
        closes.append(104.0)

        df = pd.DataFrame({
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": [100000.0] * 20,
        }, index=dates)

        sweep = detect_liquidity_sweep(df)
        self.assertIsNotNone(sweep)
        self.assertEqual(sweep["type"], "BULLISH_SWEEP")
        self.assertEqual(sweep["swept_level"], 100.0)
        self.assertEqual(sweep["rejection_low"], 95.0)
        self.assertEqual(sweep["wick_ratio"], 70.0)

    def test_detect_fair_value_gap_mitigation(self):
        """Verify 3-candle institutional Fair Value Gap (FVG) and mitigation."""
        dates = pd.date_range(end=datetime.now(), periods=12, freq="D")
        # Base price around 100
        opens = [100.0] * 12
        highs = [101.0] * 12
        lows = [99.0] * 12
        closes = [100.0] * 12

        # Candle index -3 (3 days ago): High = 101.0
        # Candle index -2 (impulsive expansion): Low = 102.0, High = 112.0
        # Candle index -1: Low = 106.0 (Gap between 101.0 and 106.0)
        # Current candle: retests into gap at 103.0
        highs[-4] = 101.0
        opens[-3] = 102.0
        closes[-3] = 111.0
        highs[-3] = 112.0
        lows[-3] = 101.5

        opens[-2] = 111.0
        highs[-2] = 115.0
        lows[-2] = 106.0
        closes[-2] = 114.0

        opens[-1] = 112.0
        lows[-1] = 102.0
        closes[-1] = 103.5
        highs[-1] = 112.0

        df = pd.DataFrame({
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": [100000.0] * 12,
        }, index=dates)

        fvg = detect_fair_value_gap(df)
        self.assertIsNotNone(fvg)
        self.assertEqual(fvg["fvg_low"], 101.0)
        self.assertEqual(fvg["fvg_high"], 106.0)

    def test_detect_volatility_contraction_vcp(self):
        """Verify Minervini Volatility Contraction Pattern (VCP) detection."""
        dates = pd.date_range(end=datetime.now(), periods=40, freq="D")
        # Wave 1 (days -35 to -15): wide range 100 to 120 (20%)
        # Wave 2 (days -15 to -5): narrower range 110 to 120 (9.1%)
        # Wave 3 (days -5 to -1): tight pivot 118 to 121 (2.5%)
        # Current: breaking out at 121.5
        prices = [110.0] * 40
        highs = [112.0] * 40
        lows = [108.0] * 40

        # Set Wave 1
        for i in range(5, 25):
            highs[i] = 120.0
            lows[i] = 100.0

        # Set Wave 2
        for i in range(25, 34):
            highs[i] = 120.0
            lows[i] = 110.0

        # Set Wave 3 (tight pivot of 5 days)
        for i in range(34, 39):
            highs[i] = 121.0
            lows[i] = 118.0
            prices[i] = 120.0

        # Current breakout candle
        prices[39] = 121.5
        highs[39] = 122.0
        lows[39] = 120.5

        df = pd.DataFrame({
            "Open": [p - 0.5 for p in prices],
            "High": highs,
            "Low": lows,
            "Close": prices,
            "Volume": [100000.0] * 40,
        }, index=dates)

        vcp = detect_volatility_contraction(df)
        self.assertIsNotNone(vcp)
        self.assertEqual(vcp["pivot_high"], 121.0)
        self.assertLess(vcp["pivot_pct"], 5.0)

    def test_detect_order_block_mss(self):
        """Verify Market Structure Shift (MSS) and Order Block detection."""
        dates = pd.date_range(end=datetime.now(), periods=30, freq="D")
        opens = [100.0] * 30
        highs = [102.0] * 30
        lows = [98.0] * 30
        closes = [100.0] * 30

        # Swing high at index 10 (day -20): 110.0
        highs[10] = 110.0

        # Bearish Order block at index 23: Open 105, Close 101, Low 100, High 106
        opens[23] = 105.0
        closes[23] = 101.0
        lows[23] = 100.0
        highs[23] = 106.0

        # MSS displacement candle at index 26: Closes at 112 (above swing high 110)
        opens[26] = 103.0
        closes[26] = 112.0
        highs[26] = 113.0
        lows[26] = 102.0

        # Current candle (index 29): Retests Order Block at 102.0
        opens[29] = 104.0
        closes[29] = 102.0
        lows[29] = 101.0
        highs[29] = 105.0

        df = pd.DataFrame({
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": [100000.0] * 30,
        }, index=dates)

        ob = detect_order_block_mss(df)
        self.assertIsNotNone(ob)
        self.assertEqual(ob["swing_high"], 110.0)
        self.assertEqual(ob["ob_low"], 100.0)
        self.assertEqual(ob["ob_high"], 106.0)

    def test_detect_wyckoff_volume_absorption(self):
        """Verify Wyckoff volume climax and stopping absorption."""
        dates = pd.date_range(end=datetime.now(), periods=25, freq="D")
        opens = [100.0] * 25
        highs = [105.0] * 25
        lows = [95.0] * 25
        closes = [98.0] * 25
        volumes = [100000.0] * 25

        # 25th day: price drops to support 95, volume climaxes to 2.5x, lower rejection tail prints
        # Range: 94 to 102 (8 pts). Open 98, Close 101, Low 94 -> Lower tail = 98 - 94 = 4 pts (50%)
        opens[-1] = 98.0
        highs[-1] = 102.0
        lows[-1] = 94.0
        closes[-1] = 101.0
        volumes[-1] = 250000.0

        df = pd.DataFrame({
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        }, index=dates)

        wyckoff = detect_wyckoff_volume_absorption(df)
        self.assertIsNotNone(wyckoff)
        self.assertGreaterEqual(wyckoff["vol_multiple"], 1.6)
        self.assertEqual(wyckoff["absorption_low"], 94.0)

    def test_detect_institutional_relative_strength(self):
        """Verify Institutional Relative Strength leader detection."""
        dates = pd.date_range(end=datetime.now(), periods=60, freq="D")
        # Strong uptrend: prices rising from 100 to 200
        prices = [100.0 + i * 1.8 for i in range(60)]
        df = pd.DataFrame({
            "Open": [p - 1.0 for p in prices],
            "High": [p + 2.0 for p in prices],
            "Low": [p - 1.0 for p in prices],
            "Close": prices,
            "Volume": [150000.0] * 60,
        }, index=dates)

        tech = calculate_technicals(df)
        rs = detect_institutional_relative_strength(df, tech)
        self.assertIsNotNone(rs)
        self.assertIn("20 EMA > 50 EMA > 200 SMA", rs["alignment"])
        self.assertGreater(rs["ret_20d_pct"], 3.5)

    def test_intraday_momentum_breakout_strategy(self):
        """Verify Intraday equity signal fires under high relative volume surge."""
        inst = get_synthetic_equity_inst()
        dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
        base_price = 450.0
        prices = [base_price + i * 0.5 for i in range(100)]
        prices[-1] = prices[-2] + 12.0  # Big intraday breakout candle

        df = pd.DataFrame({
            "Open": [p - 1.0 for p in prices],
            "High": [p + 3.0 for p in prices],
            "Low": [p - 1.0 for p in prices],
            "Close": prices,
            "Volume": [100000.0] * 99 + [300000.0],  # 3x volume surge
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
        intraday_sigs = [s for s in signals if s.horizon == SignalHorizon.INTRADAY_EQUITY]
        self.assertTrue(len(intraday_sigs) >= 1)
        self.assertEqual(intraday_sigs[0].direction, "BUY")
        self.assertIn("Intraday Institutional Volume Breakout", intraday_sigs[0].strategy_name)


if __name__ == "__main__":
    unittest.main()

