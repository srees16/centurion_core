"""Tests for services.distribution_shift and its paper-trading wiring (stdlib unittest)."""

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from services.distribution_shift import (
    classify,
    compare_live_to_backtest,
    detect_distribution_shift,
    detect_distribution_shift_rolling,
    find_drift_onset,
    load_backtest_reference,
)

RNG = np.random.default_rng(7)
BACKTEST = RNG.normal(0.0006, 0.009, 2500)


class ThresholdTests(unittest.TestCase):
    def test_fixed_threshold_boundaries(self):
        self.assertEqual(classify(0.049, 0.29), "stable")
        self.assertEqual(classify(0.05, 0.10), "drifting")
        self.assertEqual(classify(0.01, 0.30), "drifting")
        self.assertEqual(classify(0.15, 1.0), "drifting")
        self.assertEqual(classify(0.151, 0.0), "regime_break")
        self.assertEqual(classify(0.0, 1.01), "regime_break")


class DetectTests(unittest.TestCase):
    def test_same_distribution_is_stable(self):
        live = np.random.default_rng(1).normal(0.0006, 0.009, 60)
        out = detect_distribution_shift(BACKTEST, live)
        self.assertEqual(out["verdict"], "stable")
        self.assertEqual(out["calibrated_verdict"], "stable")
        self.assertAlmostEqual(out["wasserstein"], wasserstein_distance(BACKTEST, live), places=6)
        self.assertGreaterEqual(out["kl_divergence"], 0.0)
        self.assertEqual((out["n_backtest"], out["n_live"]), (2500, 60))

    def test_volatility_shift_is_detected(self):
        live = np.random.default_rng(2).normal(0.0006, 0.027, 60)
        out = detect_distribution_shift(BACKTEST, live)
        self.assertEqual(out["calibrated_verdict"], "regime_break")
        self.assertNotEqual(out["verdict"], "stable")

    def test_mean_shift_is_detected_by_calibration(self):
        live = np.random.default_rng(3).normal(-0.006, 0.009, 90)
        out = detect_distribution_shift(BACKTEST, live)
        self.assertIn(out["calibrated_verdict"], ("drifting", "regime_break"))

    def test_insufficient_data(self):
        out = detect_distribution_shift(BACKTEST, BACKTEST[:29])
        self.assertEqual(out["verdict"], "insufficient_data")
        self.assertIsNone(out["wasserstein"])

    def test_sinkhorn_status_reported(self):
        out = detect_distribution_shift(BACKTEST, BACKTEST[:60], n_bootstrap=0)
        try:
            import geomloss  # noqa: F401
            self.assertEqual(out["sinkhorn_status"], "ok")
        except ImportError:
            self.assertIsNone(out["sinkhorn"])
            self.assertTrue(out["sinkhorn_status"].startswith("skipped"))
        self.assertIsNone(out["calibrated_verdict"])

    def test_nan_values_are_dropped(self):
        live = np.random.default_rng(4).normal(0.0006, 0.009, 40)
        live[[3, 7]] = np.nan
        self.assertEqual(detect_distribution_shift(BACKTEST, live, n_bootstrap=0)["n_live"], 38)


class RollingTests(unittest.TestCase):
    def test_rolling_dates_drift_onset(self):
        rng = np.random.default_rng(5)
        values = np.concatenate([rng.normal(0.0006, 0.009, 120), rng.normal(0.0006, 0.03, 60)])
        live = pd.Series(values, index=pd.bdate_range("2026-01-01", periods=180))
        rolling = detect_distribution_shift_rolling(BACKTEST, live, window=60, step=1)
        self.assertEqual(len(rolling), 121)
        self.assertEqual(rolling[0]["start_date"], "2026-01-01")
        self.assertEqual(rolling[0]["verdict"], "stable")
        onset = find_drift_onset(rolling)
        self.assertIsNotNone(onset)
        shift_start = live.index[120]
        self.assertGreater(pd.Timestamp(onset["end_date"]), shift_start)
        self.assertLessEqual(pd.Timestamp(onset["start_date"]), shift_start)

    def test_rolling_needs_a_full_window(self):
        self.assertEqual(detect_distribution_shift_rolling(BACKTEST, BACKTEST[:59], window=60), [])


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _csv(self, index, values):
        path = self.dir / "ref.csv"
        pd.Series(values, index=index, name="return").rename_axis("date").to_csv(path)
        return path

    def test_same_period_reference_and_tracking_error(self):
        idx = pd.bdate_range("2026-01-01", periods=80)
        ref = RNG.normal(0.0005, 0.008, 80)
        path = self._csv(idx, ref)
        live = pd.Series(ref + RNG.normal(0, 0.002, 80), index=idx)
        series, mode, _ = load_backtest_reference(idx, reference_csv=path, data_dir=self.dir)
        self.assertEqual(mode, "same_period")
        self.assertEqual(len(series), 80)
        report = compare_live_to_backtest(live, reference_csv=path, data_dir=self.dir, n_bootstrap=100)
        self.assertEqual(report["reference_mode"], "same_period")
        self.assertAlmostEqual(report["tracking_error_annual"], 0.002 * np.sqrt(252), delta=0.01)
        self.assertIn("drift_onset", report)

    def test_trailing_history_when_dates_do_not_overlap(self):
        path = self._csv(pd.bdate_range("2020-01-01", periods=700), RNG.normal(0.0005, 0.008, 700))
        series, mode, _ = load_backtest_reference(pd.bdate_range("2026-01-01", periods=40),
                                                  reference_csv=path, data_dir=self.dir, trailing_days=504)
        self.assertEqual(mode, "trailing_history")
        self.assertEqual(len(series), 504)

    def test_unavailable_without_any_reference(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CENTURION_SHIFT_REFERENCE_RUN", None)
            series, mode, _ = load_backtest_reference(pd.bdate_range("2026-01-01", periods=40),
                                                      reference_csv=self.dir / "missing.csv",
                                                      data_dir=self.dir)
        self.assertIsNone(series)
        self.assertEqual(mode, "unavailable")


class PaperTraderWiringTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.db = self.dir / "paper_trades.sqlite3"
        idx = pd.bdate_range("2026-01-01", periods=120)
        self.ref_path = self.dir / "shift_reference_returns.csv"
        pd.Series(RNG.normal(0.0005, 0.008, 120), index=idx, name="return").rename_axis("date").to_csv(self.ref_path)
        self.dates = idx

    def tearDown(self):
        self.tmp.cleanup()

    def _write_snapshots(self, n):
        conn = sqlite3.connect(self.db)
        conn.execute("CREATE TABLE IF NOT EXISTS daily_snapshots (date TEXT PRIMARY KEY, equity REAL)")
        equity = 100_000 * np.cumprod(1 + np.random.default_rng(9).normal(0.0005, 0.008, n))
        conn.executemany("INSERT OR REPLACE INTO daily_snapshots VALUES (?, ?)",
                         [(d.date().isoformat(), float(e)) for d, e in zip(self.dates[:n], equity)])
        conn.commit()
        conn.close()

    def _run(self):
        from kite_connect.trading import paper_trader as pt_mod
        trader = pt_mod.PaperTrader.__new__(pt_mod.PaperTrader)
        with patch.object(pt_mod, "_DB_PATH", self.db), \
             patch.dict(os.environ, {"CENTURION_SHIFT_REFERENCE_CSV": str(self.ref_path)}):
            return trader._run_distribution_shift()

    def test_not_run_before_30_live_returns(self):
        self._write_snapshots(30)  # 29 daily returns
        self.assertIsNone(self._run())

    def test_runs_and_persists_once_30_live_returns_exist(self):
        self._write_snapshots(31)  # 30 daily returns
        result = self._run()
        self.assertIsNotNone(result)
        self.assertEqual(result["reference_mode"], "same_period")
        self.assertEqual(result["n_live"], 30)
        state = json.loads((self.dir / "distribution_shift_state.json").read_text())
        self.assertIn(state["position_size_multiplier"], (1.0, 0.75, 0.5))
        self.assertTrue((self.dir / "distribution_shift_latest.json").exists())


if __name__ == "__main__":
    unittest.main()
