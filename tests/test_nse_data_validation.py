"""Tests for nse_engine.data.validation (stdlib unittest)."""

import unittest

import numpy as np
import pandas as pd

from nse_engine.data.validation import (
    adjust_for_factors,
    clean_ohlcv,
    factors_from_prev_close,
    snap_factor,
)


def _frame(closes, opens=None, volume=1_000_000, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(closes))
    closes = np.asarray(closes, dtype=float)
    opens = closes if opens is None else np.asarray(opens, dtype=float)
    return pd.DataFrame(
        {
            "Open": opens,
            "High": np.maximum(opens, closes) * 1.01,
            "Low": np.minimum(opens, closes) * 0.99,
            "Close": closes,
            "Volume": volume,
        },
        index=idx,
    )


class CleanOhlcvTests(unittest.TestCase):
    def test_split_is_back_adjusted(self):
        closes = [1000, 1010, 1005, 101, 102, 103, 104, 105]  # 1:10 split on day 3
        opens = [1000, 1005, 1008, 100, 101, 102, 103, 104]
        out, rep = clean_ohlcv(_frame(closes, opens), "X")
        self.assertEqual(len(rep.adjustments), 1)
        self.assertAlmostEqual(rep.adjustments[0][1], 0.1)
        self.assertLess(out["Close"].pct_change().abs().max(), 0.2)
        self.assertAlmostEqual(out["Close"].iloc[0], 100.0)

    def test_wrong_scale_segment_is_dropped_not_adjusted(self):
        closes = [33.4, 33.5, 33.6, 0.34, 0.34, 33.65, 33.75, 34.0]
        out, rep = clean_ohlcv(_frame(closes), "GOLDBEES")
        self.assertEqual(len(rep.bad_ticks), 2)
        self.assertEqual(rep.adjustments, [])
        self.assertLess(out["Close"].pct_change().abs().max(), 0.05)

    def test_real_crash_without_gap_is_reported_not_adjusted(self):
        closes = [36.8, 36.8, 16.2, 16.0, 16.5, 17.0, 16.8]  # YESBANK 2020-03-06 style
        opens = [36.5, 36.9, 25.0, 16.1, 16.2, 16.9, 16.9]
        out, rep = clean_ohlcv(_frame(closes, opens), "YESBANK")
        self.assertEqual(rep.adjustments, [])
        self.assertEqual(len(rep.large_moves), 1)
        self.assertAlmostEqual(out["Close"].iloc[2], 16.2)

    def test_demerger_opening_gap_is_adjusted(self):
        closes = [660, 665, 400, 402, 405, 401, 399]
        opens = [655, 661, 398, 401, 404, 403, 400]
        out, rep = clean_ohlcv(_frame(closes, opens), "TMPV")
        self.assertEqual(len(rep.adjustments), 1)
        self.assertLess(out["Close"].pct_change().abs().max(), 0.05)

    def test_duplicates_and_phantom_rows_removed(self):
        df = _frame([100, 101, 101, 102])
        df.iloc[2, df.columns.get_loc("Volume")] = 0
        for col in ("Open", "High", "Low"):
            df.iloc[2, df.columns.get_loc(col)] = 101
        df = pd.concat([df, df.iloc[[3]]])
        out, rep = clean_ohlcv(df, "X")
        self.assertEqual(rep.duplicates, 1)
        self.assertEqual(rep.phantom_rows, 1)
        self.assertEqual(len(out), 3)


class FactorTests(unittest.TestCase):
    def test_factors_from_prev_close(self):
        idx = pd.bdate_range("2021-01-01", periods=4)
        close = pd.Series([200.0, 202.0, 101.5, 102.0], index=idx)
        prev_close = pd.Series([np.nan, 200.0, 101.0, 101.5], index=idx)  # bonus 1:1 on day 3
        f = factors_from_prev_close(close, prev_close)
        self.assertEqual(list(f.index), [idx[2]])
        self.assertAlmostEqual(f.iloc[0], 0.5)

    def test_adjust_for_factors_scales_history_and_volume(self):
        df = _frame([200, 202, 101, 102], volume=1000)
        adj = adjust_for_factors(df, pd.Series({df.index[2]: 0.5}))
        self.assertAlmostEqual(adj["Close"].iloc[0], 100.0)
        self.assertAlmostEqual(adj["Volume"].iloc[0], 2000.0)
        self.assertAlmostEqual(adj["Close"].iloc[2], 101.0)

    def test_snap_factor(self):
        self.assertEqual(snap_factor(0.0101), (0.01, True))
        self.assertFalse(snap_factor(0.73)[1] and abs(snap_factor(0.73)[0] - 0.73) > 0.04)


if __name__ == "__main__":
    unittest.main()
