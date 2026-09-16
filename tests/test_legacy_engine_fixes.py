"""Tests for the Phase 0 fixes to the legacy daily backtester (stdlib unittest)."""

import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from services import full_pipeline_backtest as fpb


def _ohlcv(index, start_price=100.0, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.015, len(index))
    close = start_price * np.cumprod(1 + rets)
    return pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 1e6},
        index=index,
    )


class DateSlicingTests(unittest.TestCase):
    def setUp(self):
        nse_days = pd.bdate_range("2021-01-01", periods=300)
        self.full = _ohlcv(nse_days, seed=1)
        # symbol with missing days (every 7th session absent)
        self.gappy = _ohlcv(nse_days, seed=2).iloc[[i for i in range(300) if i % 7 != 3]]
        # 7-day calendar symbol (crypto-like), longer history
        self.crypto = _ohlcv(pd.date_range("2020-06-01", "2022-06-01", freq="D"), seed=3)
        self.frames = {"AAA.NS": self.full, "GAP.NS": self.gappy, "BTC-USD": self.crypto}

    def test_slice_never_returns_future_rows(self):
        master = fpb._build_master_calendar(self.frames, "IND")
        for ts in master:
            for sym, df in self.frames.items():
                sl = fpb._slice_until(df, ts)
                if len(sl):
                    self.assertLessEqual(sl.index[-1], ts, sym)
                # and nothing on/before ts was dropped
                self.assertEqual(len(sl), int((df.index <= ts).sum()), sym)

    def test_master_calendar_excludes_non_nse_for_ind(self):
        master = fpb._build_master_calendar(self.frames, "IND")
        self.assertTrue(master.equals(self.full.index.union(self.gappy.index)))
        self.assertFalse(any(d.dayofweek >= 5 for d in master))

    def test_fresh_bar_detection(self):
        missing = self.full.index[3]  # GAP.NS has no bar here
        sl = fpb._slice_until(self.gappy, missing)
        self.assertFalse(fpb._has_bar_on(sl, missing))
        self.assertLess(sl.index[-1], missing)
        present = self.full.index[4]
        self.assertTrue(fpb._has_bar_on(fpb._slice_until(self.gappy, present), present))


class EwmacUnitsTests(unittest.TestCase):
    def _series(self, base):
        idx = pd.bdate_range("2019-01-01", periods=400)
        rng = np.random.default_rng(7)
        rets = rng.normal(0.0008, 0.012, len(idx))
        return pd.Series(base * np.cumprod(1 + rets), index=idx)

    def test_forecast_is_scale_invariant_and_not_capped(self):
        from kite_connect.trading.carver_live_forecasts import _compute_ewmac

        ohlcv = {s: pd.DataFrame({"Close": self._series(b)}) for s, b in (("LOW", 100.0), ("HIGH", 10_000.0))}
        result = {"LOW": {}, "HIGH": {}}
        _compute_ewmac(ohlcv, result, {"ewmac_16_64", "ewmac_64_256"})
        self.assertTrue(result["LOW"])
        for key in result["LOW"]:
            self.assertAlmostEqual(result["LOW"][key], result["HIGH"][key], places=6)
            self.assertLess(abs(result["LOW"][key]), 20.0)
            self.assertGreater(abs(result["LOW"][key]), 0.0)

    def test_fpb_formula_matches_strategy_module(self):
        from services.forecast_scalar import ewmac_to_forecast
        from services.instrument_volatility import daily_price_volatility

        out = []
        for base in (100.0, 10_000.0):
            close = self._series(base)
            raw = float(close.ewm(span=16, adjust=False).mean().iloc[-1]
                        - close.ewm(span=64, adjust=False).mean().iloc[-1])
            out.append(ewmac_to_forecast(raw, float(close.iloc[-1]) * daily_price_volatility(close), 16, 64))
        self.assertAlmostEqual(out[0], out[1], places=6)
        self.assertLess(abs(out[0]), 20.0)


class RegimeLabelTests(unittest.TestCase):
    def test_normalize_regime(self):
        from services.forecast_combiner import normalize_regime

        for label in ("strong_bull", "bull", "trending_bull", "BULL "):
            self.assertEqual(normalize_regime(label), "bull")
        for label in ("severe_bear", "bear", "trending_bear", "high_volatility", "crisis"):
            self.assertEqual(normalize_regime(label), "bear")
        for label in ("neutral", "sideways", "range_bound", "", None, "weird"):
            self.assertEqual(normalize_regime(label), "sideways")

    def test_hold_days_accept_backtest_labels(self):
        from config import Config

        self.assertEqual(Config.get_regime_hold_days("strong_bull"), Config.get_regime_hold_days("trending_bull"))
        self.assertEqual(Config.get_regime_hold_days("bull"), 12)
        self.assertEqual(Config.get_regime_hold_days("bear"), 5)
        self.assertEqual(Config.get_regime_hold_days("severe_bear"), 5)
        self.assertEqual(Config.get_regime_hold_days("neutral"), 20)
        self.assertEqual(Config.get_regime_hold_days("unknown"), 15)

    def test_regime_blend_gated_off(self):
        from config import Config
        from services import forecast_combiner as fc

        self.assertFalse(Config.REGIME_SHARPE_BLEND_ENABLED)
        with mock.patch.object(fc, "apply_regime_sharpe_weights", side_effect=AssertionError("called")):
            fc.combine_forecasts("X", {"ewmac_16_64": 5.0, "carver_value": 3.0}, regime="bull")


class VixScalingTests(unittest.TestCase):
    def test_monotone_non_increasing(self):
        caps = [fpb._vix_leverage_cap(2.0, v, 20, 30, 40) for v in np.linspace(5, 90, 400)]
        self.assertTrue(all(b <= a + 1e-12 for a, b in zip(caps, caps[1:])))
        self.assertEqual(fpb._vix_leverage_cap(2.0, 25), 1.5)
        self.assertEqual(fpb._vix_leverage_cap(2.0, 35), 1.0)
        self.assertEqual(fpb._vix_leverage_cap(2.0, 45), 0.5)
        self.assertEqual(fpb._vix_leverage_cap(2.0, None), 2.0)

    def test_previous_day_value_only(self):
        s = pd.Series([10.0, 20.0, 30.0], index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))
        self.assertIsNone(fpb._value_before(s, "2024-01-01"))
        self.assertEqual(fpb._value_before(s, "2024-01-02"), 10.0)
        self.assertEqual(fpb._value_before(s, "2024-01-03"), 20.0)
        self.assertEqual(fpb._value_before(s, "2024-01-10"), 30.0)


class ForecastNormaliserTests(unittest.TestCase):
    def test_uses_only_past_observations(self):
        norm = fpb._ForecastNormaliser(target_abs=10.0, min_obs=60, cap=20.0)
        day = {"A": {"src": 2.0}, "B": {"src": -2.0}}
        for _ in range(59):
            self.assertEqual(norm.apply(day)["A"]["src"], 2.0)  # < min_obs -> scalar 1
            norm.update(day)
        self.assertEqual(norm.apply(day)["A"]["src"], 2.0)      # 59 past days
        norm.update(day)
        self.assertAlmostEqual(norm.scalar("src"), 5.0)          # 10 / mean|2|
        # A huge value today must not influence today's scalar
        spike = {"A": {"src": 1000.0}}
        out = norm.apply(spike)
        self.assertEqual(out["A"]["src"], 20.0)                 # capped, scalar still 5
        self.assertAlmostEqual(norm.scalar("src"), 5.0)
        norm.update(spike)
        self.assertLess(norm.scalar("src"), 5.0)

    def test_state_roundtrip(self):
        norm = fpb._ForecastNormaliser(min_obs=1)
        norm.update({"A": {"s": 4.0}})
        other = fpb._ForecastNormaliser(min_obs=1)
        other.load_state(norm.state())
        self.assertAlmostEqual(other.scalar("s"), norm.scalar("s"))


class PitLoaderTests(unittest.TestCase):
    def setUp(self):
        from kite_connect.nse import nse_universe
        self.mod = nse_universe
        self._saved = nse_universe._PIT_DATA
        nse_universe._PIT_DATA = None

    def tearDown(self):
        self.mod._PIT_DATA = self._saved

    def test_returns_none_when_file_absent(self):
        with mock.patch.object(self.mod, "_pit_path", return_value="/nonexistent/pit.json"):
            self.assertIsNone(self.mod.get_nse_universe_pit("2020-01-01"))
            self.assertIsNone(self.mod.get_nse_universe_pit_union())

    def test_no_backfill_and_meta_ignored(self):
        payload = {"2019-03": ["AAA", "BBB"], "2020-09": ["AAA", "CCC"],
                   "_meta": {"snapshot_dates": ["2019-02-01"]}}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pit.json")
            with open(path, "w") as f:
                json.dump(payload, f)
            with mock.patch.object(self.mod, "_pit_path", return_value=path):
                self.assertIsNone(self.mod.get_nse_universe_pit("2019-02-28"))
                self.assertEqual(self.mod.get_nse_universe_pit("2019-03-01"), ["AAA", "BBB"])
                self.assertEqual(self.mod.get_nse_universe_pit("2020-08-31"), ["AAA", "BBB"])
                self.assertEqual(self.mod.get_nse_universe_pit(pd.Timestamp("2021-01-05")), ["AAA", "CCC"])
                self.assertEqual(self.mod.get_nse_universe_pit_union(), ["AAA", "BBB", "CCC"])


class BuildPitJsonTests(unittest.TestCase):
    def test_uses_only_past_snapshots(self):
        import importlib.util
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "build_pit_json.py")
        spec = importlib.util.spec_from_file_location("build_pit_json", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out = mod.build({"2018-10-04": ["OLD"], "2020-07-25": ["NEW"]}, end_year=2021)
        self.assertNotIn("2018-09", out)          # before first snapshot: skipped
        self.assertEqual(out["2019-03"], ["OLD"])
        self.assertEqual(out["2020-03"], ["OLD"])  # later snapshot never used early
        self.assertEqual(out["2020-09"], ["NEW"])
        self.assertIn("_meta", out)


if __name__ == "__main__":
    unittest.main()
