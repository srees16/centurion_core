"""Signals may only use the past: truncating the panel must not change history."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import truncate
from nse_engine.config import SignalConfig
from nse_engine.signals import (centred_rank, compute_signal_panels, daily_returns,
                                low_beta_forecast, near_high_forecast, residual_momentum_raw)
from nse_engine.universe import compute_universe_panel
from nse_engine.config import UniverseConfig

CUT = "2021-06-30"
UNI = UniverseConfig(top_n_liquid=20, min_history_days=60, liquidity_lookback_days=40,
                     refresh_every_n_days=21, min_median_value_inr=0.0, min_price_inr=0.0)


def _mask(data):
    return compute_universe_panel(data, UNI).mask


@pytest.mark.parametrize("group", ["fast_trend", "slow_trend", "low_vol", "delivery",
                                   "residual_momentum", "near_high", "low_beta"])
def test_group_forecast_is_causal(panel, group):
    """Row t of every group forecast depends only on rows <= t."""
    cfg = SignalConfig(group_weights=((group, 1.0),), momentum_lookback=126, momentum_skip=21,
                       low_vol_lookback=60, delivery_lookback=21, residual_momentum_months=12,
                       high_lookback=126, beta_lookback=126, normalizer_min_obs=20)
    full = compute_signal_panels(panel.close, _mask(panel), cfg, delivery_pct=panel.delivery_pct)
    short_panel = truncate(panel, CUT)
    short = compute_signal_panels(short_panel.close, _mask(short_panel), cfg,
                                  delivery_pct=short_panel.delivery_pct)
    a = full.groups[group].loc[:CUT].to_numpy()
    b = short.groups[group].to_numpy()
    assert a.shape == b.shape
    assert np.allclose(np.nan_to_num(a, nan=-7.7), np.nan_to_num(b, nan=-7.7), atol=1e-9), \
        f"{group} forecast changed when later data was removed"


def test_centred_rank_is_within_universe_and_centred(panel):
    mask = _mask(panel)
    ranks = centred_rank(panel.close, mask)
    inside = ranks.where(mask)
    assert float(np.nanmax(np.abs(inside.to_numpy()))) <= 0.5
    row_means = inside.mean(axis=1, skipna=True).dropna()
    assert np.allclose(row_means.to_numpy(), 0.0, atol=1e-12)
    assert ranks.where(~mask).notna().sum().sum() == 0, "ranked a name outside the universe"


def test_residual_momentum_updates_only_at_month_starts(panel):
    raw = residual_momentum_raw(panel.close, _mask(panel), months=12, window=6)
    changed = raw.diff().abs().sum(axis=1) > 0
    change_dates = raw.index[changed]
    months = {(d.year, d.month) for d in change_dates}
    assert len(change_dates) == len(months), "score changed more than once inside a month"


def test_near_high_is_bounded_by_construction(panel):
    cfg = SignalConfig(high_lookback=126, normalizer_min_obs=20)
    out, _ = near_high_forecast(panel.close, _mask(panel), cfg)
    assert np.nanmax(np.abs(out.to_numpy())) <= cfg.forecast_cap + 1e-9


def test_low_beta_prefers_low_beta_names(panel):
    """The forecast must rank a deliberately high-beta name below a low-beta one."""
    close = panel.close.copy()
    rng = np.random.default_rng(3)
    market = pd.Series(np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(close)))), index=close.index)
    close["SYM00"] = 100 * market ** 2.5                      # high beta
    close["SYM01"] = 100 * market ** 0.2                      # low beta
    everything = pd.DataFrame(True, index=close.index, columns=close.columns)
    cfg = SignalConfig(beta_lookback=126, normalizer_min_obs=20)
    out, _ = low_beta_forecast(close, everything, cfg, daily_returns(close))
    last = out.iloc[-1]
    assert last["SYM01"] > last["SYM00"], "low-beta name should score above the high-beta one"
