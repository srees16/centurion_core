"""Synthetic ``MarketData`` builders and small-scale configs for NSE engine tests."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from nse_engine.config import EngineConfig
from nse_engine.types import MarketData


def make_market_data(
    n_symbols: int = 40,
    n_days: int = 600,
    *,
    start: str = "2016-01-01",
    seed: int = 0,
    drift: float = 0.0,
    drift_dispersion: float = 0.0,
    daily_vol: float = 0.015,
    vol_dispersion: float = 0.5,
    price_level: float = 100.0,
    median_value_inr: float = 5e7,
    etf_symbols: Sequence[str] = ("NIFTYBEES",),
    metals: bool = True,
    metals_drift: float = 0.0003,
    index: bool = True,
    vix: bool = True,
    sectors: Optional[Mapping[str, str]] = None,
    delist: Optional[Mapping[str, int]] = None,
    listing: Optional[Mapping[str, int]] = None,
    float_dtype: str = "float64",
) -> MarketData:
    """Random-walk OHLCV panel on a business-day calendar.

    * each stock ``S000..`` has log drift ``drift + drift_dispersion * N(0,1)``
      (persistent, so trend/momentum rules can find winners) and daily vol
      ``daily_vol * exp(vol_dispersion * N(0,1) / 2)``;
    * ``etf_symbols`` are extra ETF columns (flagged in ``etfs``);
    * ``metals`` adds GOLDBEES and SILVERBEES (also ETFs);
    * ``NIFTY50`` is the equal-weight index of the stocks, ``INDIAVIX`` a
      scaled 20-day realised vol of it;
    * ``delist`` maps symbol -> last trading row, ``listing`` symbol -> first row.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    stocks = [f"S{i:03d}" for i in range(n_symbols)]
    etfs = list(etf_symbols)
    metal_syms = ["GOLDBEES", "SILVERBEES"] if metals else []
    symbols = stocks + etfs + metal_syms
    m = len(symbols)

    mu = np.full(m, drift) + drift_dispersion * rng.standard_normal(m)
    sig = daily_vol * np.exp(vol_dispersion * rng.standard_normal(m) / 2.0)
    for k, s in enumerate(symbols):
        if s in metal_syms:
            mu[k] = metals_drift
            sig[k] = 0.01
        elif s in etfs:
            mu[k] = drift
            sig[k] = daily_vol * 0.7
    log_ret = mu[None, :] + sig[None, :] * rng.standard_normal((n_days, m))
    level = price_level * np.exp(rng.uniform(-1.0, 1.0, m))
    close = level[None, :] * np.exp(np.cumsum(log_ret, axis=0))
    prev = np.vstack([close[:1], close[:-1]])
    open_ = prev * np.exp(0.2 * sig[None, :] * rng.standard_normal((n_days, m)))
    hi = np.maximum(open_, close) * (1 + np.abs(0.5 * sig[None, :] * rng.standard_normal((n_days, m))))
    lo = np.minimum(open_, close) * (1 - np.abs(0.5 * sig[None, :] * rng.standard_normal((n_days, m))))
    value = median_value_inr * np.exp(0.5 * rng.standard_normal((n_days, m))) * np.exp(rng.uniform(-1, 1, m))[None, :]
    volume = value / close

    frames = {}
    for name, arr in (("open", open_), ("high", hi), ("low", lo), ("close", close), ("volume", volume), ("value", value)):
        frames[name] = pd.DataFrame(arr.copy(), index=dates, columns=symbols)
    for s, last in (delist or {}).items():
        for f in frames.values():
            f.iloc[last + 1 :, symbols.index(s)] = np.nan
    for s, first in (listing or {}).items():
        for f in frames.values():
            f.iloc[:first, symbols.index(s)] = np.nan

    idx_close = pd.DataFrame(index=dates)
    if index:
        eq = np.log(frames["close"][stocks]).diff().mean(axis=1).fillna(0.0)
        nifty = 10_000 * np.exp(eq.cumsum())
        idx_close["NIFTY50"] = nifty
        if vix:
            rv = nifty.pct_change().rolling(20, min_periods=5).std() * np.sqrt(252) * 100
            idx_close["INDIAVIX"] = (rv * 1.1).bfill()
    for name in frames:
        frames[name] = frames[name].astype(float_dtype)

    data = MarketData(
        dates=dates,
        open=frames["open"], high=frames["high"], low=frames["low"], close=frames["close"],
        volume=frames["volume"], value=frames["value"], index_close=idx_close,
        etfs=frozenset(etfs + metal_syms), sectors=dict(sectors or {}), source="synthetic",
    )
    data.validate()
    data.data_hash = data.compute_hash()
    return data


def small_config(runs_dir: str = "data/nse_engine/runs_test", **overrides) -> EngineConfig:
    """Lookbacks scaled for panels of a few hundred days."""
    base = {
        "initial_capital": 5_000_000.0,
        "runs_dir": runs_dir,
        "universe.top_n_liquid": 30,
        "universe.min_history_days": 120,
        "universe.liquidity_lookback_days": 60,
        "universe.refresh_every_n_days": 21,
        "universe.min_median_value_inr": 1e6,
        "signals.slow_ewmac": ((32, 128),),
        "signals.momentum_lookback": 120,
        "signals.momentum_skip": 10,
        "signals.low_vol_lookback": 120,
        "signals.normalizer_min_obs": 20,
        "signals.fdm_lookback_days": 250,
        "portfolio.target_positions": 10,
        "portfolio.max_positions": 15,
        "portfolio.exit_rank": 20,
        "portfolio.max_weight": 0.15,
        "regime.trend_ma_days": 100,
        "regime.breadth_ma_days": 100,
        "sleeves.trend_ma_days": 100,
        "sleeves.min_history_days": 120,
    }
    base.update(overrides)
    start = base.pop("start", None)
    end = base.pop("end", None)
    cfg = EngineConfig().replace(**base)
    if start or end:
        cfg = cfg.replace(**{k: v for k, v in (("start", start), ("end", end)) if v})
    return cfg
