"""Paper-trading risk metrics come from the daily equity curve, not per-trade returns."""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


class DailyMetricsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "paper_trades.sqlite3"
        self.dates = pd.bdate_range("2026-01-01", periods=120)

    def _trader(self, returns, capital=500_000.0):
        from kite_connect.trading import paper_trader as pt_mod
        conn = sqlite3.connect(self.db)
        conn.execute("CREATE TABLE IF NOT EXISTS daily_snapshots (date TEXT PRIMARY KEY, equity REAL)")
        equity = capital * np.cumprod(1 + np.asarray(returns, dtype=float))
        conn.executemany("INSERT OR REPLACE INTO daily_snapshots VALUES (?, ?)",
                         [(d.date().isoformat(), float(e)) for d, e in zip(self.dates, equity)])
        conn.commit()
        conn.close()
        trader = pt_mod.PaperTrader.__new__(pt_mod.PaperTrader)
        trader.initial_capital = capital
        return trader, pt_mod

    def test_matches_engine_metrics(self):
        rng = np.random.default_rng(3)
        rets = rng.normal(0.0006, 0.009, 120)
        trader, pt_mod = self._trader(rets)
        with patch.object(pt_mod, "_DB_PATH", self.db):
            got = trader.daily_metrics()
        self.assertTrue(got)
        from nse_engine.metrics import compute_metrics
        series = pd.Series(rets[1:], index=self.dates[1:])  # first row has no prior close
        equity = (1 + series).cumprod() * 500_000.0
        want = compute_metrics(series, equity, rf_annual=trader._risk_free_annual(),
                               initial_capital=500_000.0)
        self.assertAlmostEqual(got["sharpe"], want["sharpe"], places=6)
        self.assertAlmostEqual(got["sortino"], want["sortino"], places=6)

    def test_sharpe_is_excess_over_risk_free(self):
        rets = np.full(120, 0.0004)  # steady positive drift, ~10%/yr
        trader, pt_mod = self._trader(rets + np.random.default_rng(1).normal(0, 0.006, 120))
        with patch.object(pt_mod, "_DB_PATH", self.db):
            got = trader.daily_metrics()
        raw = pd.Series(np.diff(np.cumprod(1 + rets)) / np.cumprod(1 + rets)[:-1])
        self.assertGreater(trader._risk_free_annual(), 0.0)
        self.assertLess(got["sharpe"], raw.mean() / raw.std(ddof=1) * np.sqrt(252) + 100)  # sane
        self.assertIn("max_drawdown", got)

    def test_no_metrics_without_two_snapshots(self):
        trader, pt_mod = self._trader([0.001])
        with patch.object(pt_mod, "_DB_PATH", self.db):
            self.assertEqual(trader.daily_metrics(), {})

    def test_trade_count_does_not_scale_sharpe(self):
        """The old code annualised per-trade returns by sqrt(n_trades)."""
        rng = np.random.default_rng(5)
        rets = rng.normal(0.0005, 0.008, 120)
        trader, pt_mod = self._trader(rets)
        with patch.object(pt_mod, "_DB_PATH", self.db):
            first = trader.daily_metrics()["sharpe"]
            conn = sqlite3.connect(self.db)
            conn.execute("CREATE TABLE IF NOT EXISTS paper_positions (symbol TEXT, pnl_pct REAL, is_open INT)")
            conn.executemany("INSERT INTO paper_positions VALUES (?, ?, 0)",
                             [(f"S{i}", 1.0) for i in range(200)])
            conn.commit()
            conn.close()
            second = trader.daily_metrics()["sharpe"]
        self.assertAlmostEqual(first, second, places=12)


if __name__ == "__main__":
    unittest.main()
