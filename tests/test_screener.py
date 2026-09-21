"""Tests for the market screener module."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import screener
import technical_analysis


def _make_ohlcv(n=210, base=100, trend=0.1):
    """Generate synthetic OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    close = base + np.cumsum(np.random.randn(n) * 0.5 + trend)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.1
    volume = np.random.randint(500000, 2000000, size=n).astype(float)
    return pd.DataFrame({
        "Open": open_, "High": high, "Low": low,
        "Close": close, "Volume": volume,
    }, index=dates)


def _make_analyzed(ticker="TEST", n=210, **kwargs):
    """Create an analyzed result dict."""
    df = _make_ohlcv(n, **kwargs)
    signals = technical_analysis.analyze(ticker, df)
    return {"ticker": ticker, "signals": signals, "price": signals.current_price, "df": df}


class TestUniverses(unittest.TestCase):
    def test_default_universe(self):
        """Default universe should be dynamic based on market."""
        tickers = screener.get_universe("default")
        if screener.is_indian_market():
            self.assertTrue(len(tickers) >= 20)
            self.assertNotIn("AAPL", tickers)
        else:
            self.assertEqual(len(tickers), len(screener.DEFAULT_UNIVERSE))
            self.assertIn("AAPL", tickers)

    def test_sectors_universe(self):
        """Sectors universe should return ETFs."""
        tickers = screener.get_universe("sectors")
        self.assertIn("XLK", tickers)

    def test_unknown_universe_falls_back(self):
        """Unknown universe name should fall back to default."""
        tickers = screener.get_universe("nonexistent")
        default_tickers = screener.get_universe("default")
        self.assertEqual(len(tickers), len(default_tickers))


class TestScreenFunctions(unittest.TestCase):
    def setUp(self):
        """Create a set of analyzed results for screening."""
        self.results = [_make_analyzed("TEST")]

    def test_screen_momentum_returns_list(self):
        """Momentum screen should return a list."""
        hits = screener.screen_momentum(self.results)
        self.assertIsInstance(hits, list)

    def test_screen_breakout_returns_list(self):
        """Breakout screen should return a list."""
        hits = screener.screen_breakout(self.results)
        self.assertIsInstance(hits, list)

    def test_screen_oversold_returns_list(self):
        """Oversold screen should return a list."""
        hits = screener.screen_oversold(self.results)
        self.assertIsInstance(hits, list)

    def test_screen_accumulation_returns_list(self):
        """Accumulation screen should return a list."""
        hits = screener.screen_accumulation(self.results)
        self.assertIsInstance(hits, list)

    def test_screen_all_returns_dict(self):
        """All screens should return a dict of lists."""
        result = screener.screen_all(self.results)
        self.assertIsInstance(result, dict)
        self.assertIn("momentum", result)
        self.assertIn("breakout", result)
        self.assertIn("oversold", result)
        self.assertIn("accumulation", result)

    def test_hit_has_required_fields(self):
        """Screen hits should have ticker, price, score, details."""
        # Force a momentum hit by manually setting signals
        r = self.results[0]
        s = r["signals"]
        s.price_above_200sma = True
        s.ema9 = s.ema21 + 1  # EMA9 > EMA21
        s.rsi = 45  # In pullback range
        hits = screener.screen_momentum(self.results)
        if hits:
            h = hits[0]
            self.assertIn("ticker", h)
            self.assertIn("price", h)
            self.assertIn("score", h)
            self.assertIn("details", h)


class TestCustomTickers(unittest.TestCase):
    def test_load_custom_newline_separated(self):
        """Should load tickers from newline-separated file."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("AAPL\nMSFT\nGOOGL\n")
            f.flush()
            tickers = screener._load_custom_tickers(f.name)
        os.unlink(f.name)
        self.assertEqual(tickers, ["AAPL", "MSFT", "GOOGL"])

    def test_load_custom_comma_separated(self):
        """Should load tickers from comma-separated file."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("AAPL, MSFT, GOOGL")
            f.flush()
            tickers = screener._load_custom_tickers(f.name)
        os.unlink(f.name)
        self.assertEqual(tickers, ["AAPL", "MSFT", "GOOGL"])

    def test_load_nonexistent_file(self):
        """Should return empty list for nonexistent file."""
        tickers = screener._load_custom_tickers("/nonexistent/file.txt")
        self.assertEqual(tickers, [])


if __name__ == "__main__":
    unittest.main()
