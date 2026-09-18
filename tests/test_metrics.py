"""Metric formulas: annualisation, drawdown, and the deflated Sharpe's trial penalty."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nse_engine.metrics import compute_metrics, max_drawdown
from nse_engine.validation.dsr import deflated_sharpe, excess_sharpe, performance_summary


def _series(daily: float, n: int = 1260) -> pd.Series:
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(daily, index=idx)


def test_sharpe_of_a_constant_series_is_not_finite_nonsense():
    r = _series(0.0004)
    assert not np.isfinite(excess_sharpe(r, 0.0)) or excess_sharpe(r, 0.0) == 0.0


def test_excess_sharpe_matches_the_definition():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0005, 0.01, 2520), index=pd.bdate_range("2015-01-01", periods=2520))
    rf = 0.065
    rf_daily = rf / 252            # arithmetic, matching the engine's cash accrual
    expected = (r - rf_daily).mean() / (r - rf_daily).std(ddof=1) * np.sqrt(252)
    assert abs(excess_sharpe(r, rf) - expected) < 1e-9


def test_cagr_compounds_over_252_day_years():
    r = _series(0.001, n=252)
    p = performance_summary(r, 0.0)
    assert abs(p["cagr"] - ((1.001 ** 252) - 1)) < 1e-9


def test_max_drawdown_is_measured_from_the_peak():
    equity = pd.Series([100, 120, 90, 150], index=pd.bdate_range("2024-01-01", periods=4))
    assert abs(max_drawdown(equity) - (-0.25)) < 1e-12


def test_metrics_are_self_consistent():
    rng = np.random.default_rng(5)
    r = pd.Series(rng.normal(0.0006, 0.012, 756), index=pd.bdate_range("2021-01-01", periods=756))
    equity = 500_000 * (1 + r).cumprod()
    m = compute_metrics(r, equity, rf_annual=0.065, initial_capital=500_000.0)
    assert abs(m["total_return"] - (float(equity.iloc[-1]) / 500_000 - 1)) < 1e-9
    assert m["ann_vol"] > 0 and np.isfinite(m["sharpe"])
    assert -1 < m["max_drawdown"] <= 0
    assert abs(m["calmar"] - m["cagr"] / abs(m["max_drawdown"])) < 1e-9


def test_deflated_sharpe_penalises_more_trials():
    rng = np.random.default_rng(11)
    r = pd.Series(rng.normal(0.0007, 0.011, 2520), index=pd.bdate_range("2015-01-01", periods=2520))
    few = deflated_sharpe(r, n_trials=5, rf_annual=0.0, sr_variance=1e-4)
    many = deflated_sharpe(r, n_trials=5000, rf_annual=0.0, sr_variance=1e-4)
    assert many["sr0_annual"] > few["sr0_annual"], "the hurdle must rise with the number of trials"
    assert many["dsr"] <= few["dsr"]
