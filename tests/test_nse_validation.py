"""
Tests for nse_engine.validation (Phase 2) and the legacy DSR unit fixes.

Run: python -m unittest tests.test_nse_validation -v

Engine-independent: backtests are stubbed via dependency injection and all
market data is synthetic.
"""

from __future__ import annotations

import json
import logging
import math
import tempfile
import unittest
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from nse_engine.config import EngineConfig
from nse_engine.types import BacktestResult, MarketData
from nse_engine.validation import (
    HoldoutLockedError,
    TrialRegistry,
    alpha_beta,
    aronson_detrended_sharpe,
    benchmark_gate,
    cscv_pbo,
    deflated_sharpe,
    deflated_sharpe_from_stats,
    effective_number_of_trials,
    excess_sharpe,
    expected_max_sharpe,
    full_report,
    generate_folds,
    lag_sensitivity,
    min_track_record_length,
    record_result,
    run_benchmarks,
    run_holdout,
    run_walk_forward,
)
from nse_engine.validation.benchmarks import decision_positions

logging.getLogger("nse_engine").setLevel(logging.ERROR)


# ----------------------------------------------------------------------------
# Synthetic fixtures
# ----------------------------------------------------------------------------

def bdays(n: int, start: str = "2010-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def make_market_data(n_days: int = 700, n_sym: int = 12, growth=None, start="2010-01-01",
                     seed: int = 0) -> MarketData:
    """Panel where symbol j grows by ``growth[j]`` per day (open == prev close)."""
    dates = bdays(n_days, start)
    rng = np.random.default_rng(seed)
    if growth is None:
        growth = np.linspace(-0.0005, 0.0015, n_sym)
    growth = np.asarray(growth, dtype=float)
    cols = [f"S{j:02d}" for j in range(n_sym)]
    close = 100.0 * np.cumprod(np.tile(1 + growth, (n_days, 1)), axis=0)
    opn = np.vstack([np.full(n_sym, 100.0), close[:-1]])
    close_df = pd.DataFrame(close, index=dates, columns=cols)
    open_df = pd.DataFrame(opn, index=dates, columns=cols)
    value = pd.DataFrame(1e9 * (1 + 0.01 * rng.random((n_days, n_sym))), index=dates, columns=cols)
    nifty = pd.DataFrame({"NIFTY50": 10000 * np.cumprod(1 + rng.normal(0.0004, 0.01, n_days))},
                         index=dates)
    return MarketData(dates=dates, open=open_df, high=close_df, low=close_df, close=close_df,
                      volume=value / close_df, value=value, index_close=nifty,
                      data_hash="synthetic")


def all_members_universe(data, config) -> pd.DataFrame:
    return pd.DataFrame(True, index=data.dates, columns=data.close.columns)


def zero_cost(trade, adv, date):
    return np.zeros(len(trade))


class StubBacktest:
    """Deterministic stand-in for nse_engine.engine.run_backtest.

    Returns for a config are a fixed noise stream (seeded by the grid value)
    plus a daily drift ``skill[target_positions]``, restricted to
    ``[config.start, config.end]``.  ``lag_days`` removes 0.0002/day per lag.
    """

    def __init__(self, dates: pd.DatetimeIndex, skill: Dict[int, float], vol: float = 0.01):
        self.dates = dates
        self.skill = skill
        self.vol = vol
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, data, config, *, record=True, tag="", lag_days=0):
        tp = config.portfolio.target_positions
        rng = np.random.default_rng(1000 + tp)
        full = pd.Series(rng.normal(self.skill.get(tp, 0.0), self.vol, len(self.dates)),
                         index=self.dates) - 0.0002 * lag_days
        lo, hi = pd.Timestamp(config.start), pd.Timestamp(config.end)
        r = full[(full.index >= lo) & (full.index <= hi)]
        self.calls.append({"tag": tag, "start": lo, "end": hi, "tp": tp, "lag": lag_days,
                           "record": record})
        w = pd.DataFrame({"X": 1.0}, index=r.index)
        return BacktestResult(equity=(1 + r).cumprod() * 1e6, returns=r, weights=w,
                              trades=pd.DataFrame(), metrics={"sharpe": excess_sharpe(r)},
                              config=config, data_hash="stubhash")


# ----------------------------------------------------------------------------
# DSR
# ----------------------------------------------------------------------------

class TestDeflatedSharpe(unittest.TestCase):

    def test_reference_values_estimator_variance(self):
        for n, expected in ((24, 0.97), (1550, 0.67), (20000, 0.42)):
            out = deflated_sharpe_from_stats(0.071, 3189, n, skew=-1.19, kurtosis=14.3)
            self.assertAlmostEqual(out["dsr"], expected, delta=0.005, msg=f"N={n}")
        self.assertTrue(deflated_sharpe_from_stats(0.071, 3189, 24, -1.19, 14.3)["passed"])
        self.assertFalse(deflated_sharpe_from_stats(0.071, 3189, 1550, -1.19, 14.3)["passed"])

    def test_legacy_function_converts_annual_sharpe(self):
        from services.deflated_sharpe import deflated_sharpe_ratio, min_backtest_length

        annual = deflated_sharpe_ratio(observed_sr=0.071 * math.sqrt(252), n_obs=3189, n_trials=24,
                                       skewness=-1.19, kurtosis=14.3)
        daily = deflated_sharpe_ratio(observed_sr=0.071, n_obs=3189, n_trials=1550,
                                      skewness=-1.19, kurtosis=14.3, annualized=False)
        self.assertAlmostEqual(annual, 0.967, delta=0.005)
        self.assertAlmostEqual(daily, 0.671, delta=0.005)
        # MinBTL is now in daily observations (annual SR 1.0 -> ~680 days)
        self.assertGreater(min_backtest_length(1.0), 500)

    def test_legacy_warns_on_annual_sr_passed_as_daily(self):
        from services.deflated_sharpe import deflated_sharpe_ratio

        with self.assertLogs("services.deflated_sharpe", level="WARNING"):
            deflated_sharpe_ratio(observed_sr=1.127, n_obs=3189, n_trials=24, annualized=False)

    def test_legacy_expected_max_requires_std(self):
        from services.deflated_sharpe import expected_max_sr

        with self.assertRaises(ValueError):
            expected_max_sr(10)

    def test_from_returns_matches_stats(self):
        rng = np.random.default_rng(3)
        r = pd.Series(rng.normal(0.0008, 0.01, 2500), index=bdays(2500))
        out = deflated_sharpe(r, n_trials=50)
        ref = deflated_sharpe_from_stats(out["sr_daily"], out["T"], 50, out["skew"], out["kurtosis"])
        self.assertAlmostEqual(out["dsr"], ref["dsr"], places=10)
        self.assertAlmostEqual(out["sr_annual"], out["sr_daily"] * math.sqrt(252), places=12)
        self.assertEqual(out["n_trials_source"], "explicit")
        self.assertLess(out["dsr"], deflated_sharpe(r, n_trials=2)["dsr"])

    def test_rf_reduces_sharpe(self):
        rng = np.random.default_rng(4)
        r = pd.Series(rng.normal(0.0008, 0.01, 1000), index=bdays(1000))
        self.assertLess(deflated_sharpe(r, n_trials=1, rf_annual=0.065)["sr_daily"],
                        deflated_sharpe(r, n_trials=1)["sr_daily"])

    def test_expected_max_and_mintrl(self):
        self.assertEqual(expected_max_sharpe(1, 0.01), 0.0)
        self.assertGreater(expected_max_sharpe(100, 1e-4), expected_max_sharpe(10, 1e-4))
        self.assertEqual(min_track_record_length(0.0), float("inf"))
        mtrl = min_track_record_length(0.05)
        self.assertAlmostEqual(mtrl, 1 + (1 + 0.5 * 0.05 ** 2) * (1.6448536 / 0.05) ** 2, places=3)

    def test_effective_n_clusters_correlated_trials(self):
        rng = np.random.default_rng(5)
        t = 1500
        cols = {}
        for g in range(3):
            factor = rng.normal(0, 0.01, t)
            for k in range(5):
                cols[f"g{g}_{k}"] = factor + rng.normal(0, 0.003, t)
        m = pd.DataFrame(cols, index=bdays(t))
        out = effective_number_of_trials(m)
        self.assertEqual(out["n_eff"], 3)
        self.assertEqual(out["labels"].groupby(out["labels"]).size().tolist(), [5, 5, 5])
        unequal = pd.concat([m.iloc[:, :5], m.iloc[:, 5:7]], axis=1)  # 5 + 2 of other group
        big = pd.DataFrame({f"b{k}": m.iloc[:, 0] + rng.normal(0, 0.003, t) for k in range(12)})
        small = pd.DataFrame({f"s{k}": m.iloc[:, 7] + rng.normal(0, 0.003, t) for k in range(3)})
        self.assertEqual(effective_number_of_trials(pd.concat([big, small], axis=1))["n_eff"], 2)
        self.assertEqual(effective_number_of_trials(unequal)["n_eff"], 2)
        indep = pd.DataFrame(rng.normal(0, 0.01, (t, 6)), index=bdays(t))
        self.assertEqual(effective_number_of_trials(indep)["n_eff"], 6)

    def test_effective_n_large_registry_path(self):
        rng = np.random.default_rng(6)
        t = 800
        cols = {}
        for g in range(4):
            factor = rng.normal(0, 0.01, t)
            for k in range(30):
                cols[f"g{g}_{k}"] = factor + rng.normal(0, 0.002, t)
        m = pd.DataFrame(cols, index=bdays(t))
        out = effective_number_of_trials(m, max_cluster_trials=40)
        self.assertEqual(out["method"], "average_linkage_subsample_assign")
        self.assertEqual(out["n_eff"], 4)

    def test_trials_matrix_variance_and_clusters(self):
        rng = np.random.default_rng(7)
        t = 1200
        m = pd.DataFrame(rng.normal(0, 0.01, (t, 20)), index=bdays(t))
        best = m.mean().idxmax()
        out = deflated_sharpe(m[best], trials_matrix=m)
        self.assertEqual(out["variance_source"], "trials_matrix")
        self.assertEqual(out["n_trials_eff"], 20)
        self.assertLess(out["dsr"], 0.95)  # best of 20 noise trials must not pass


# ----------------------------------------------------------------------------
# PBO
# ----------------------------------------------------------------------------

class TestCSCVPBO(unittest.TestCase):

    def test_iid_noise_trials_pbo_near_half(self):
        pbos = []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            m = pd.DataFrame(rng.normal(0, 0.01, (800, 20)))
            pbos.append(cscv_pbo(m, n_splits=8)["pbo"])
        self.assertAlmostEqual(float(np.mean(pbos)), 0.5, delta=0.12)

    def test_superior_trial_pbo_near_zero(self):
        rng = np.random.default_rng(1)
        m = pd.DataFrame(rng.normal(0, 0.01, (2000, 20)))
        m[5] += 0.002
        out = cscv_pbo(m)
        self.assertLess(out["pbo"], 0.05)
        self.assertEqual(out["n_combinations"], 12870)
        self.assertLess(out["prob_oos_loss"], 0.05)

    def test_overfit_construction_high_pbo(self):
        rng = np.random.default_rng(2)
        s, blk, n = 16, 60, 16
        x = rng.normal(0, 0.001, (s * blk, n))
        for i in range(n):  # trial i is lucky in block i, slightly unlucky elsewhere
            for b in range(s):
                x[b * blk:(b + 1) * blk, i] += 0.01 if b == i else -0.01 / 15
        out = cscv_pbo(pd.DataFrame(x), n_splits=16)
        self.assertGreater(out["pbo"], 0.9)
        self.assertLess(out["mean_oos_metric"], 0)
        self.assertLess(out["median_logit"], 0)

    def test_contract_and_validation(self):
        rng = np.random.default_rng(3)
        m = pd.DataFrame(rng.normal(0, 0.01, (1003, 4)))
        with self.assertLogs("nse_engine.validation.pbo", level="WARNING"):
            out = cscv_pbo(m, n_splits=16, max_combinations=500, metric="mean")
        self.assertEqual(out["n_combinations"], 500)
        self.assertEqual(out["n_dropped_rows"], 1003 % 16)
        self.assertEqual(out["n_obs"], 1003 - 1003 % 16)
        self.assertEqual(len(out["logits"]), 500)
        for key in ("pbo", "median_logit", "n_trials", "degradation_slope", "prob_oos_loss"):
            self.assertIn(key, out)
        with self.assertRaises(ValueError):
            cscv_pbo(m[[0]])
        with self.assertRaises(ValueError):
            cscv_pbo(m, n_splits=7)


# ----------------------------------------------------------------------------
# Trial registry
# ----------------------------------------------------------------------------

def _fake_result(cfg: EngineConfig, returns: pd.Series, data_hash="h1") -> BacktestResult:
    return BacktestResult(equity=(1 + returns).cumprod(), returns=returns, weights=pd.DataFrame(),
                          trades=pd.DataFrame(), metrics={"sharpe": 1.0, "cagr": 0.1},
                          config=cfg, data_hash=data_hash)


class TestTrialRegistry(unittest.TestCase):

    def test_list_dedupe_common_dates_and_hash_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = EngineConfig(runs_dir=tmp)
            d = bdays(300)
            rng = np.random.default_rng(0)
            a = base.replace(**{"portfolio.target_positions": 10})
            b = base.replace(**{"portfolio.target_positions": 20})
            record_result(_fake_result(a, pd.Series(rng.normal(0, .01, 300), index=d)), tag="t")
            record_result(_fake_result(b, pd.Series(rng.normal(0, .01, 250), index=d[50:])), tag="t")
            rid_latest = Path(record_result(_fake_result(
                a, pd.Series(rng.normal(0, .01, 300), index=d)), tag="t2")).name

            reg = TrialRegistry(tmp)
            trials = reg.list_trials()
            self.assertEqual(len(trials), 3)
            for col in ("run_id", "tag", "config_hash", "git_commit", "data_hash", "start",
                        "end", "created_at", "sharpe", "cagr"):
                self.assertIn(col, trials.columns)
            mat = reg.returns_matrix()
            self.assertEqual(mat.shape, (250, 2))
            self.assertIn(rid_latest, mat.columns)
            self.assertEqual(reg.returns_matrix(dedupe_config=False).shape[1], 3)
            self.assertEqual(len(reg.returns_matrix(start=d[100], end=d[199])), 100)

            record_result(_fake_result(base, pd.Series(rng.normal(0, .01, 300), index=d),
                                       data_hash="h2"), tag="t")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                reg.returns_matrix()
            self.assertTrue(any("data hashes" in str(w.message) for w in caught))
            self.assertEqual(reg.returns_matrix(data_hash="h2").shape[1], 1)


# ----------------------------------------------------------------------------
# Walk-forward
# ----------------------------------------------------------------------------

class TestWalkForward(unittest.TestCase):

    def test_folds_anchored_and_rolling(self):
        d = bdays(8 * 261, "2012-01-02")
        folds = generate_folds(d, train_years=4, test_months=12, anchored=True)
        self.assertGreaterEqual(len(folds), 3)
        for f, g in zip(folds, folds[1:]):
            self.assertEqual(f["train_start"], folds[0]["train_start"])
            self.assertLess(f["test_end"], g["test_start"])
        for f in folds:
            self.assertLess(f["train_end"], f["test_start"])
        rolling = generate_folds(d, 4, 12, anchored=False)
        self.assertGreater(rolling[1]["train_start"], rolling[0]["train_start"])

    def test_selection_on_train_only_and_trials_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = bdays(7 * 261, "2012-01-02")
            data = make_market_data(n_days=len(d), start="2012-01-02")
            base = EngineConfig(start=str(d[0].date()), end=str(d[-1].date()), runs_dir=tmp,
                                risk_free_annual=0.0)
            stub = StubBacktest(d, skill={10: 0.0, 20: 0.002, 30: -0.001})
            out = run_walk_forward(data, base, {"portfolio.target_positions": [10, 20, 30]},
                                   train_years=4, test_months=12, backtest_fn=stub)
            folds = out["folds"]
            self.assertGreaterEqual(len(folds), 2)
            for f in folds:
                self.assertEqual(f["params"], {"portfolio.target_positions": 20})
            for c in stub.calls:
                if c["tag"] == "wfo-train":
                    fold = next(f for f in folds if pd.Timestamp(f["train_end"]) == c["end"])
                    self.assertLess(c["end"], pd.Timestamp(fold["test_start"]))
            oos = out["oos_returns"]
            self.assertTrue(oos.index.is_monotonic_increasing)
            self.assertGreaterEqual(oos.index[0], pd.Timestamp(folds[0]["test_start"]))
            self.assertEqual(out["summary"]["n_backtests"], len(folds) * 4)
            self.assertGreater(out["summary"]["oos_sharpe"], 1.0)
            tags = TrialRegistry(tmp).list_trials()["tag"].value_counts()
            self.assertEqual(tags["wfo-train"], len(folds) * 3)
            self.assertEqual(tags["wfo-test"], len(folds))


# ----------------------------------------------------------------------------
# Holdout
# ----------------------------------------------------------------------------

class TestHoldout(unittest.TestCase):

    def test_lock_refuses_second_run_and_records_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = bdays(600, "2020-01-01")
            data = make_market_data(n_days=600, start="2020-01-01")
            cfg = EngineConfig(runs_dir=str(Path(tmp) / "runs"))
            stub = StubBacktest(d, {20: 0.001})
            lock = Path(tmp) / "holdout.lock"
            out = run_holdout(data, cfg, "2021-06-01", "2022-05-31", lock_path=lock, backtest_fn=stub)
            self.assertEqual(stub.calls[-1]["tag"], "holdout")
            rec = json.loads(lock.read_text())["evaluations"]
            self.assertEqual(len(rec), 1)
            self.assertEqual(rec[0]["status"], "completed")
            self.assertEqual(rec[0]["window"], {"start": "2021-06-01", "end": "2022-05-31"})
            for key in ("config_hash", "git_commit", "data_hash", "created_at", "metrics"):
                self.assertIn(key, rec[0])
            self.assertGreater(out["metrics"]["n_obs"], 200)
            with self.assertRaises(HoldoutLockedError):
                run_holdout(data, cfg, "2021-06-01", "2022-05-31", lock_path=lock, backtest_fn=stub)
            self.assertEqual(len(stub.calls), 1)
            with self.assertLogs("nse_engine.validation.holdout", level="WARNING"):
                run_holdout(data, cfg, "2021-06-01", "2022-05-31", lock_path=lock,
                            backtest_fn=stub, force=True)
            rec = json.loads(lock.read_text())["evaluations"]
            self.assertEqual(len(rec), 2)
            self.assertTrue(rec[1]["forced"])
            self.assertEqual(TrialRegistry(cfg.runs_dir).list_trials()["tag"].tolist(),
                             ["holdout", "holdout"])


# ----------------------------------------------------------------------------
# Benchmarks
# ----------------------------------------------------------------------------

class TestBenchmarks(unittest.TestCase):

    def setUp(self):
        self.growth = np.array([0.001] * 8)
        self.data = make_market_data(n_days=400, n_sym=8, growth=self.growth)
        self.cfg = EngineConfig(start=str(self.data.dates[10].date()),
                                end=str(self.data.dates[-1].date()), cash_yield_annual=0.0)

    def test_ew_hold_zero_cost_tracks_growth(self):
        out = run_benchmarks(self.data, self.cfg, universe_fn=all_members_universe, cost_fn=zero_cost)
        ew = out["ew_hold_universe"]
        self.assertEqual(ew.index[0], self.data.dates[10])
        self.assertEqual(ew.index[-1], self.data.dates[-1])
        # invested from the open after the first decision -> constant growth
        np.testing.assert_allclose(ew.iloc[2:].to_numpy(), 0.001, atol=1e-9)
        self.assertEqual(set(out), {"ew_hold_universe", "momentum_12_1_top15",
                                    "nifty50_price_index", "cash"})
        self.assertEqual(out["ew_hold_universe"].attrs["cost_model"], "custom")

    def test_costs_reduce_returns_and_gross_le_one(self):
        free = run_benchmarks(self.data, self.cfg, universe_fn=all_members_universe, cost_fn=zero_cost)
        costly = run_benchmarks(self.data, self.cfg, universe_fn=all_members_universe)
        self.assertIn(costly["ew_hold_universe"].attrs["cost_model"],
                      ("nse_engine.costs", "flat_15bp_fallback"))
        self.assertLess(costly["ew_hold_universe"].sum(), free["ew_hold_universe"].sum())
        # with growth 0.1%/day, gross <= 1 means return never exceeds 0.1%
        self.assertLessEqual(costly["ew_hold_universe"].max(), 0.001 + 1e-9)

    def test_momentum_picks_winners(self):
        growth = np.r_[np.full(15, 0.0), np.full(5, 0.002)]
        data = make_market_data(n_days=600, n_sym=20, growth=growth)
        cfg = EngineConfig(start=str(data.dates[300].date()), end=str(data.dates[-1].date()),
                           cash_yield_annual=0.0)
        out = run_benchmarks(data, cfg, universe_fn=all_members_universe, cost_fn=zero_cost,
                             momentum_top_n=5)
        np.testing.assert_allclose(out["momentum_12_1_top15"].iloc[2:].to_numpy(), 0.002, atol=1e-9)

    def test_decision_positions_monthly(self):
        d = bdays(90, "2020-01-01")
        pos = decision_positions(d, 3, 89)
        self.assertEqual(pos[0], 3)
        self.assertTrue(all(d[p].month != d[p + 1].month for p in pos[1:]))

    def test_gate(self):
        d = bdays(1000)
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 0.01, 1000)
        strat = pd.Series(noise + 0.0015, index=d)
        benches = {"ew_hold_universe": pd.Series(noise + 0.0002, index=d),
                   "momentum_12_1_top15": pd.Series(noise[100:] + 0.0003, index=d[100:]),
                   "nifty50_price_index": pd.Series(noise, index=d)}
        g = benchmark_gate(strat, benches, margin=0.3, rf_annual=0.0)
        self.assertTrue(g["passed"])
        self.assertEqual(g["benchmarks"]["momentum_12_1_top15"]["n_obs"], 900)
        close = {"ew_hold_universe": pd.Series(noise + 0.0014, index=d),
                 "momentum_12_1_top15": pd.Series(noise, index=d)}
        g2 = benchmark_gate(strat, close, margin=0.3, rf_annual=0.0)
        self.assertFalse(g2["passed"])
        self.assertFalse(g2["benchmarks"]["ew_hold_universe"]["passed"])
        self.assertTrue(g2["benchmarks"]["ew_hold_universe"]["beats"])
        self.assertFalse(benchmark_gate(strat, {"ew_hold_universe": strat}, rf_annual=0.0)["passed"])


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------

class TestDiagnostics(unittest.TestCase):

    def test_detrending_removes_beta_drift_not_skill(self):
        d = bdays(3000)
        rng = np.random.default_rng(1)
        bench = pd.Series(rng.normal(0.0008, 0.01, 3000), index=d)
        exposure = pd.Series(1.0, index=d)
        pure_beta = aronson_detrended_sharpe(bench, exposure, bench)
        self.assertGreater(pure_beta["raw_excess_sharpe"], 0.8)
        self.assertAlmostEqual(pure_beta["detrended_sharpe"], 0.0, places=6)
        skill = bench + 0.0008
        skilled = aronson_detrended_sharpe(skill, exposure, bench)
        self.assertGreater(skilled["detrended_sharpe"], 0.8)  # not ~0 by construction
        half = aronson_detrended_sharpe(0.5 * bench, exposure * 0.5, bench)
        self.assertAlmostEqual(half["detrended_sharpe"], 0.0, places=6)

    def test_alpha_beta_aligns_by_date(self):
        d = bdays(1500)
        rng = np.random.default_rng(2)
        bench = pd.Series(rng.normal(0.0005, 0.01, 1500), index=d)
        strat = (0.8 * bench + 0.0004 + rng.normal(0, 0.002, 1500)).iloc[200:]
        out = alpha_beta(strat.sample(frac=1.0, random_state=0), bench)
        self.assertAlmostEqual(out["beta"], 0.8, delta=0.03)
        self.assertAlmostEqual(out["alpha_daily"], 0.0004, delta=0.0002)
        self.assertEqual(out["n_obs"], 1300)
        self.assertGreater(out["alpha_t_hac"], 3)
        with self.assertRaises(TypeError):
            alpha_beta(strat.to_numpy(), bench)

    def test_lag_sensitivity_uses_lag_days(self):
        d = bdays(800)
        stub = StubBacktest(d, {20: 0.001})
        cfg = EngineConfig(start=str(d[0].date()), end=str(d[-1].date()), risk_free_annual=0.0)
        out = lag_sensitivity(None, cfg, lags=(0, 1, 3), backtest_fn=stub, record=False)
        self.assertEqual([c["lag"] for c in stub.calls], [0, 1, 3])
        self.assertEqual([c["tag"] for c in stub.calls], ["lag-0", "lag-1", "lag-3"])
        s = [out["lags"][k]["excess_sharpe"] for k in (0, 1, 3)]
        self.assertTrue(s[0] > s[1] > s[2])

    def test_full_report_is_json_serialisable(self):
        d = bdays(900)
        data = make_market_data(n_days=900)
        stub = StubBacktest(d, {20: 0.0008})
        cfg = EngineConfig(start=str(d[0].date()), end=str(d[-1].date()))
        res = stub(data, cfg)
        rng = np.random.default_rng(0)
        trials = pd.DataFrame(rng.normal(0, 0.01, (len(res.returns), 12)), index=res.returns.index)
        trials["best"] = res.returns
        bench = {"ew_hold_universe": pd.Series(rng.normal(0, .01, 900), index=d),
                 "nifty50_price_index": data.index_close["NIFTY50"].pct_change().fillna(0.0)}
        rep = full_report(res, data, bench, trials_matrix=trials)
        text = json.dumps(rep, allow_nan=False)
        self.assertIn("dsr", rep)
        self.assertIn("pbo", rep)
        self.assertIn("alpha_beta_nifty50", rep)
        self.assertIn("aronson_detrended", rep)
        self.assertNotIn('"error"', text)


# ----------------------------------------------------------------------------
# Legacy aronson_validator fixes (also covered by the pytest suite)
# ----------------------------------------------------------------------------

class TestLegacyAronsonFixes(unittest.TestCase):

    def test_detrend_without_benchmark_warns(self):
        from services.aronson_validator import demean_returns, detrend_returns

        r = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 300))
        with self.assertWarns(DeprecationWarning):
            out = detrend_returns(r)
        pd.testing.assert_series_equal(out, demean_returns(r))

    def test_detrend_with_benchmark_and_alpha_beta_alignment(self):
        from services.aronson_validator import compute_alpha_beta, detrend_returns

        d = bdays(500)
        rng = np.random.default_rng(1)
        bench = pd.Series(rng.normal(0.001, 0.01, 500), index=d)
        strat = bench.iloc[100:] * 1.0
        dt = detrend_returns(strat, benchmark_returns=bench, exposure=1.0)
        self.assertAlmostEqual(float((strat - dt).iloc[0]), float(bench.iloc[100:].mean()), places=12)
        ab = compute_alpha_beta(strat, bench)
        self.assertAlmostEqual(ab["beta"], 1.0, places=4)
        with self.assertRaises(ValueError):
            compute_alpha_beta(strat.to_numpy(), bench.to_numpy())


if __name__ == "__main__":
    unittest.main()
