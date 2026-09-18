"""Execution realism: fills come from prices the decision could not see, and cost money."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nse_engine.config import CostConfig
from nse_engine.costs import impact_bps, statutory_cost


DAY = "2025-06-02"


def test_statutory_costs_are_charged_on_both_sides():
    buy = statutory_cost(100_000.0, "BUY", DAY)
    sell = statutory_cost(100_000.0, "SELL", DAY)
    assert buy > 0 and sell > 0
    assert sell > buy, "the sell side carries DP charges on top"


def test_costs_scale_with_traded_value():
    small = statutory_cost(10_000.0, "BUY", DAY)
    large = statutory_cost(1_000_000.0, "BUY", DAY)
    assert large > small * 50, "statutory charges are mostly proportional"


def test_impact_grows_with_participation():
    cfg = CostConfig()
    adv = 1e8
    light = impact_bps(1e5, adv, cfg)       # 0.1% of ADV
    heavy = impact_bps(5e6, adv, cfg)       # 5% of ADV
    assert cfg.spread_floor_bps <= light < heavy
    assert heavy > 2 * light, "square-root impact should grow with size"


def test_impact_never_below_the_spread_floor():
    cfg = CostConfig()
    assert impact_bps(1.0, 1e12, cfg) >= cfg.spread_floor_bps


@pytest.mark.slow
def test_backtest_fills_use_the_next_open_not_todays_close(store_path):
    """A one-day delay must change results; a same-day fill would make lag meaningless."""
    from nse_engine.deployment import load_deployment
    from nse_engine.engine import run_backtest
    from runners.run_nse_engine import _load_data

    cfg = load_deployment("config/nse_engine_deployed.json").engine.replace(
        start="2024-01-01", end="2024-12-31")
    data = _load_data(cfg, data_start="2022-01-03")
    base = run_backtest(data, cfg, record=False)
    lagged = run_backtest(data, cfg, record=False, lag_days=1)
    assert not np.allclose(base.returns.to_numpy(), lagged.returns.to_numpy(), atol=1e-12)


@pytest.mark.slow
def test_trading_costs_reduce_returns(store_path):
    """Doubling modelled impact must lower CAGR and raise the cost drag."""
    from nse_engine.deployment import load_deployment
    from nse_engine.engine import run_backtest
    from runners.run_nse_engine import _load_data

    cfg = load_deployment("config/nse_engine_deployed.json").engine.replace(
        start="2024-01-01", end="2024-12-31")
    data = _load_data(cfg, data_start="2022-01-03")
    cheap = run_backtest(data, cfg, record=False)
    dear = run_backtest(data, cfg.replace(**{
        "costs.impact_coefficient_bps": cfg.costs.impact_coefficient_bps * 3,
        "costs.spread_floor_bps": cfg.costs.spread_floor_bps * 3}), record=False)
    assert dear.metrics["cagr"] < cheap.metrics["cagr"]
    assert dear.metrics["cost_drag"] > cheap.metrics["cost_drag"]
