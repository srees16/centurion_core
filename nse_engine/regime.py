"""
Market regime from index trend, universe breadth and India VIX.

The regime is computed from market data only (never from the strategy's own
equity curve) and is causal: the state on ``t`` uses rows ``<= t``.

Raw state per day
    risk_off  if (not trend_up and breadth < breadth_risk_off) or VIX >= vix_extreme
    risk_on   if trend_up and breadth >= breadth_risk_on and VIX < vix_elevated
    neutral   otherwise (also when trend or breadth cannot be computed yet)

A new raw state has to persist ``confirm_days`` consecutive days before the
confirmed state switches (hysteresis).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd

from nse_engine.config import RegimeConfig

logger = logging.getLogger(__name__)

RISK_ON, NEUTRAL, RISK_OFF = "risk_on", "neutral", "risk_off"
VIX_FALLBACK_WINDOW = 20
VIX_FALLBACK_MULTIPLIER = 1.25
INDEX_FALLBACKS = ("NIFTY500",)


@dataclass
class RegimePanel:
    state: pd.Series  # confirmed state (str)
    raw_state: pd.Series
    scale: pd.Series
    trend_up: pd.Series  # float: 1.0 / 0.0 / NaN (unknown)
    breadth: pd.Series
    vix: pd.Series
    vix_is_fallback: pd.Series

    def switched(self) -> pd.Series:
        """True on the dates where the confirmed state changed."""
        s = self.state
        return (s != s.shift(1)) & s.shift(1).notna()


def apply_hysteresis(raw: Sequence[str], confirm_days: int, initial: str = NEUTRAL) -> List[str]:
    """Switch to a new state only after it has been the raw state ``confirm_days`` days in a row."""
    confirm = max(int(confirm_days), 1)
    current, candidate, count = initial, None, 0
    out: List[str] = []
    for r in raw:
        if r == current:
            candidate, count = None, 0
        elif r == candidate:
            count += 1
        else:
            candidate, count = r, 1
        if candidate is not None and count >= confirm:
            current, candidate, count = candidate, None, 0
        out.append(current)
    return out


def raw_regime_state(trend_up: pd.Series, breadth: pd.Series, vix: pd.Series, cfg: RegimeConfig) -> pd.Series:
    t = trend_up.to_numpy(dtype="float64")
    b = breadth.to_numpy(dtype="float64")
    v = vix.to_numpy(dtype="float64")
    known = np.isfinite(t) & np.isfinite(b)
    with np.errstate(invalid="ignore"):
        extreme = np.isfinite(v) & (v >= cfg.vix_extreme)
        off = known & (t < 0.5) & (b < cfg.breadth_risk_off)
        on = known & (t > 0.5) & (b >= cfg.breadth_risk_on) & np.isfinite(v) & (v < cfg.vix_elevated)
    state = np.where(extreme | off, RISK_OFF, np.where(on, RISK_ON, NEUTRAL))
    return pd.Series(state, index=trend_up.index, dtype=object)


def compute_regime(
    index_close: pd.DataFrame, close: pd.DataFrame, universe_mask: pd.DataFrame, cfg: RegimeConfig
) -> RegimePanel:
    """Regime state and exposure scale per date."""
    idx = close.index
    col = cfg.index_symbol if cfg.index_symbol in index_close.columns else None
    if col is None:
        col = next((c for c in INDEX_FALLBACKS if c in index_close.columns), None)
        logger.warning("index %s missing from index_close; using %s", cfg.index_symbol, col)
    if col is not None:
        nifty = index_close[col].astype("float64")
        ma = nifty.rolling(cfg.trend_ma_days, min_periods=cfg.trend_ma_days).mean()
        trend = pd.Series(np.where(ma.notna() & nifty.notna(), (nifty > ma).astype(float), np.nan), index=idx)
        rv = nifty.pct_change(fill_method=None).rolling(VIX_FALLBACK_WINDOW, min_periods=VIX_FALLBACK_WINDOW).std()
        fallback = rv * 100.0 * np.sqrt(252.0) * VIX_FALLBACK_MULTIPLIER
    else:
        trend = pd.Series(np.nan, index=idx)
        fallback = pd.Series(np.nan, index=idx)

    c = close.astype("float64")
    bma = c.rolling(cfg.breadth_ma_days, min_periods=cfg.breadth_ma_days).mean()
    have = universe_mask.to_numpy() & bma.notna().to_numpy() & c.notna().to_numpy()
    above = have & (c.to_numpy() > bma.to_numpy())
    n_have = have.sum(axis=1)
    breadth = pd.Series(np.where(n_have > 0, above.sum(axis=1) / np.maximum(n_have, 1), np.nan), index=idx)

    if cfg.vix_symbol in index_close.columns:
        vix_raw = index_close[cfg.vix_symbol].astype("float64")
    else:
        vix_raw = pd.Series(np.nan, index=idx)
    is_fb = vix_raw.isna()
    vix = vix_raw.where(~is_fb, fallback)

    raw = raw_regime_state(trend, breadth, vix, cfg)
    state = pd.Series(apply_hysteresis(raw.tolist(), cfg.confirm_days), index=idx, dtype=object)
    scale_map = {RISK_ON: cfg.scale_risk_on, NEUTRAL: cfg.scale_neutral, RISK_OFF: cfg.scale_risk_off}
    scale = state.map(scale_map).astype("float64")
    return RegimePanel(
        state=state, raw_state=raw, scale=scale, trend_up=trend, breadth=breadth, vix=vix, vix_is_fallback=is_fb
    )
