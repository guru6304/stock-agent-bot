"""Tests for watchlist auto-curator module."""

import csv
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import watchlist_curator


class TestSuggestAdditions(unittest.TestCase):
    def test_multi_screen_ticker_suggested(self):
        candidates = {
            "NVDA": {"screens": ["momentum", "breakout"], "sector": "Technology"},
            "AAPL": {"screens": ["momentum"], "sector": "Technology"},
        }
        suggestions = watchlist_curator.suggest_additions(
            ["MSFT"], candidates, {}, min_screens=2
        )
        self.assertEqual(len(suggestions), 1)
        self.assertEqual(suggestions[0]["ticker"], "NVDA")

    def test_existing_ticker_excluded(self):
        candidates = {
            "MSFT": {"screens": ["momentum", "breakout"], "sector": "Technology"},
        }
        suggestions = watchlist_curator.suggest_additions(
            ["MSFT"], candidates, {}, min_screens=2
        )
        self.assertEqual(len(suggestions), 0)

    def test_single_screen_below_threshold(self):
        candidates = {
            "NVDA": {"screens": ["momentum"], "sector": "Technology"},
        }
        suggestions = watchlist_curator.suggest_additions(
            [], candidates, {}, min_screens=2
        )
        self.assertEqual(len(suggestions), 0)

    def test_sector_bonus(self):
        candidates = {
            "NVDA": {"screens": ["momentum", "breakout"], "sector": "Technology"},
            "XOM": {"screens": ["momentum", "accumulation"], "sector": "Energy"},
        }
        phases = {"Technology": "LEADING", "Energy": "LAGGING"}
        suggestions = watchlist_curator.suggest_additions(
            [], candidates, phases, min_screens=2
        )
        # NVDA should rank higher due to LEADING sector bonus
        self.assertEqual(suggestions[0]["ticker"], "NVDA")
        self.assertGreater(suggestions[0]["score"], suggestions[1]["score"])

    def test_empty_candidates(self):
        suggestions = watchlist_curator.suggest_additions([], {}, {})
        self.assertEqual(suggestions, [])


class TestSuggestRemovals(unittest.TestCase):
    def test_consecutive_losses_flagged(self):
        watchlist = [{"ticker": "BAD", "sector": "Technology"}]
        trade_history = {
            "BAD": [
                {"pnl": -50, "exit_date": "2026-03-20"},
                {"pnl": -30, "exit_date": "2026-03-18"},
                {"pnl": -40, "exit_date": "2026-03-15"},
            ]
        }
        suggestions = watchlist_curator.suggest_removals(
            watchlist, trade_history, {}, {}, max_losses=3
        )
        self.assertEqual(len(suggestions), 1)
        self.assertIn("consecutive", suggestions[0]["reason"])

    def test_no_removal_if_recent_win(self):
        watchlist = [{"ticker": "OK", "sector": "Technology"}]
        trade_history = {
            "OK": [
                {"pnl": 100, "exit_date": "2026-03-20"},  # Most recent is a win
                {"pnl": -30, "exit_date": "2026-03-18"},
                {"pnl": -40, "exit_date": "2026-03-15"},
            ]
        }
        suggestions = watchlist_curator.suggest_removals(
            watchlist, trade_history, {}, {}, max_losses=3
        )
        # Should not suggest removal since most recent trade is a win
        loss_reasons = [s for s in suggestions if "consecutive" in s.get("reason", "")]
        self.assertEqual(len(loss_reasons), 0)

    def test_lagging_sector_flagged(self):
        watchlist = [{"ticker": "XOM", "sector": "Energy"}]
        suggestions = watchlist_curator.suggest_removals(
            watchlist, {}, {}, {"Energy": "LAGGING"}, max_losses=3
        )
        self.assertEqual(len(suggestions), 1)
        self.assertIn("LAGGING", suggestions[0]["reason"])

    def test_no_removal_for_healthy_ticker(self):
        watchlist = [{"ticker": "AAPL", "sector": "Technology"}]
        recent_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        trade_history = {
            "AAPL": [
                {"pnl": 100, "exit_date": recent_date},
            ]
        }
        alert_dates = {"AAPL": (datetime.now() - timedelta(days=2)).isoformat()}
        suggestions = watchlist_curator.suggest_removals(
            watchlist, trade_history, alert_dates, {}, max_losses=3
        )
        self.assertEqual(len(suggestions), 0)


class TestApplySuggestions(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
        writer = csv.DictWriter(self._tmp, fieldnames=["ticker", "sector", "notes"])
        writer.writeheader()
        writer.writerow({"ticker": "AAPL", "sector": "Technology", "notes": ""})
        writer.writerow({"ticker": "BAD", "sector": "Energy", "notes": ""})
        self._tmp.close()
        self._orig = watchlist_curator.WATCHLIST_PATH
        watchlist_curator.WATCHLIST_PATH = Path(self._tmp.name)

    def tearDown(self):
        watchlist_curator.WATCHLIST_PATH = self._orig
        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass

    def test_apply_additions_and_removals(self):
        additions = [{"ticker": "NVDA", "sector": "Technology", "screens": ["momentum"]}]
        removals = [{"ticker": "BAD", "sector": "Energy"}]

        added, removed = watchlist_curator.apply_suggestions(additions, removals)
        self.assertEqual(added, 1)
        self.assertEqual(removed, 1)

        wl = watchlist_curator.load_watchlist()
        tickers = [e["ticker"] for e in wl]
        self.assertIn("AAPL", tickers)
        self.assertIn("NVDA", tickers)
        self.assertNotIn("BAD", tickers)

    def test_no_duplicate_additions(self):
        additions = [{"ticker": "AAPL", "sector": "Technology", "screens": []}]
        added, _ = watchlist_curator.apply_suggestions(additions, [])
        self.assertEqual(added, 0)


if __name__ == "__main__":
    unittest.main()
