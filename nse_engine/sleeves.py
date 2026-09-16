"""
Gold and silver ETF trend sleeves.

A sleeve symbol is *in trend* on ``t`` when its close is above its
``trend_ma_days`` simple moving average and it has at least
``min_history_days`` closes.  In-trend sleeves share the sleeve capital by
inverse volatility; an out-of-trend sleeve's share goes to cash (the
allocator decides the sleeve capital).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from nse_engine.config import SleeveConfig

logger = logging.getLogger(__name__)


def sleeve_symbols(cfg: SleeveConfig, available: List[str] | pd.Index | None = None) -> List[str]:
    """Enabled sleeve symbols (restricted to ``available`` when given)."""
    syms = []
    if cfg.gold_enabled:
        syms.append(cfg.gold_symbol)
    if cfg.silver_enabled:
        syms.append(cfg.silver_symbol)
    if available is not None:
        avail = set(available)
        syms = [s for s in syms if s in avail]
    return syms


@dataclass
class SleevePanels:
    symbols: List[str]
    in_trend: pd.DataFrame  # date x sleeve symbol bool
    vol: pd.DataFrame  # annualised vol


def compute_sleeve_panels(close: pd.DataFrame, cfg: SleeveConfig, vol_lookback: int) -> SleevePanels:
    """Causal trend flags and vols for the enabled sleeve symbols present in ``close``."""
    syms = sleeve_symbols(cfg, close.columns)
    c = close.reindex(columns=syms).astype("float64")
    ma = c.rolling(cfg.trend_ma_days, min_periods=cfg.trend_ma_days).mean()
    hist = c.notna().cumsum()
    in_trend = (c > ma) & ma.notna() & (hist >= cfg.min_history_days)
    ret = c.pct_change(fill_method=None)
    vol = ret.rolling(vol_lookback, min_periods=max(vol_lookback // 2, 2)).std() * np.sqrt(252.0)
    return SleevePanels(symbols=syms, in_trend=in_trend, vol=vol)


def sleeve_weights(in_trend: pd.Series, vols: pd.Series) -> pd.Series:
    """Inverse-vol weights (sum 1) over in-trend sleeves; empty if none."""
    active = [s for s in in_trend.index if bool(in_trend[s])]
    if not active:
        return pd.Series(dtype="float64")
    v = vols.reindex(active).astype("float64")
    fallback = float(np.nanmedian(v.to_numpy())) if np.isfinite(v.to_numpy()).any() else 1.0
    v = v.where(np.isfinite(v) & (v > 0), fallback)
    inv = 1.0 / v
    return inv / inv.sum()
