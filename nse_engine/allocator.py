"""
Capital allocation between the core stock book and the metal sleeves.

Let sigma_c and sigma_m be the estimated annualised vols of the core book
and of the in-trend metals basket (current relative weights applied to the
past ``vol_lookback_days`` returns), s = clip(core_risk_share,
core_risk_share_min, core_risk_share_max) and G = max_gross.

* Full-scale split: choose capital c + m = G with
  c sigma_c / (c sigma_c + m sigma_m) = s  (correlation ignored; the tested
  metal/momentum correlation is slightly negative).
* The regime scale multiplies the core only: core = c x scale.
* ``risk_off_to_metals``: capital freed from the core may be moved to the
  in-trend metals, but only until the metals' risk reaches their own maximum
  risk budget, defined as (1 - core_risk_share_min) x R_full with
  R_full = c sigma_c + m sigma_m (the full-scale portfolio risk).  Gross never
  exceeds G; the rest is cash.
* No active metals: core = G x scale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from nse_engine.config import AllocatorConfig

logger = logging.getLogger(__name__)


@dataclass
class Allocation:
    core_capital: float
    sleeve_capital: float
    regime_scale: float
    core_vol: float = float("nan")
    sleeve_vol: float = float("nan")
    risk_share: float = float("nan")
    notes: List[str] = field(default_factory=list)

    @property
    def gross(self) -> float:
        return self.core_capital + self.sleeve_capital


def basket_vol(weights: pd.Series, returns_window: pd.DataFrame | np.ndarray) -> float:
    """Annualised vol of a fixed-weight basket over a window of daily returns.

    ``returns_window`` is rows x len(weights) (same column order); missing
    returns count as 0.
    """
    w = np.asarray(weights, dtype="float64")
    if w.size == 0 or not np.isfinite(w).all() or w.sum() <= 0:
        return float("nan")
    r = np.nan_to_num(np.asarray(returns_window, dtype="float64"), nan=0.0)
    if r.shape[0] < 5:
        return float("nan")
    port = r @ w
    return float(np.std(port, ddof=1) * np.sqrt(252.0))


def allocate(
    core_vol: float,
    sleeve_vol: float,
    has_core: bool,
    has_sleeves: bool,
    regime_scale: float,
    cfg: AllocatorConfig,
) -> Allocation:
    """Capital fractions (of equity) for the core book and the sleeves."""
    G = float(cfg.max_gross)
    scale = float(np.clip(regime_scale, 0.0, 1.0))
    notes: List[str] = []
    if not has_sleeves:
        core = G * scale if has_core else 0.0
        core, _ = _apply_vol_target(core, 0.0, core_vol, sleeve_vol, cfg, notes)
        return Allocation(core, 0.0, scale, core_vol, sleeve_vol, 1.0, notes)

    s = float(np.clip(cfg.core_risk_share, cfg.core_risk_share_min, cfg.core_risk_share_max))
    sc = core_vol if np.isfinite(core_vol) and core_vol > 0 else np.nan
    sm = sleeve_vol if np.isfinite(sleeve_vol) and sleeve_vol > 0 else np.nan
    if not has_core:
        # no core names: metals keep their own (full-scale) capital
        sc_eff = sc if np.isfinite(sc) else sm
    else:
        sc_eff = sc
    if np.isfinite(sc_eff) and np.isfinite(sm):
        ratio = sc_eff * (1.0 - s) / (s * sm)  # m / c
        c = G / (1.0 + ratio)
        m = G - c
    else:
        notes.append("allocator: vol estimate unavailable, capital split = risk share")
        c, m = G * s, G * (1.0 - s)
    r_full = (c * sc_eff + m * sm) if (np.isfinite(sc_eff) and np.isfinite(sm)) else np.nan

    core = c * scale if has_core else 0.0
    freed = c - core
    sleeve = m
    if cfg.risk_off_to_metals and freed > 0:
        if np.isfinite(r_full) and np.isfinite(sm):
            max_m = (1.0 - cfg.core_risk_share_min) * r_full / sm
        else:
            max_m = G * (1.0 - cfg.core_risk_share_min)
        extra = float(np.clip(max_m - m, 0.0, freed))
        if extra > 0:
            sleeve = m + extra
            notes.append(f"allocator: {extra:.3f} of freed core capital moved to metals")
    total = core + sleeve
    if total > G:
        k = G / total
        core, sleeve = core * k, sleeve * k
    core, sleeve = _apply_vol_target(core, sleeve, sc_eff, sm, cfg, notes)
    return Allocation(core, sleeve, scale, core_vol, sleeve_vol, s, notes)


def _apply_vol_target(core: float, sleeve: float, core_vol: float, sleeve_vol: float,
                      cfg: AllocatorConfig, notes: List[str]):
    """Scale the whole book down to ``target_vol_annual`` when it runs hotter.

    Portfolio vol is estimated as sqrt((core x sigma_c)^2 + (sleeve x sigma_m)^2)
    — zero correlation between the stock book and the metals, consistent with
    the risk-share split above. The scale is clipped to [vol_target_min_scale, 1]:
    down only (no leverage in a CNC account), and never to an empty book.
    """
    target = float(cfg.target_vol_annual)
    if target <= 0:
        return core, sleeve
    parts = []
    if core > 0 and np.isfinite(core_vol) and core_vol > 0:
        parts.append(core * core_vol)
    if sleeve > 0 and np.isfinite(sleeve_vol) and sleeve_vol > 0:
        parts.append(sleeve * sleeve_vol)
    if not parts:
        return core, sleeve
    port_vol = float(np.sqrt(sum(p * p for p in parts)))
    if port_vol <= target:
        return core, sleeve
    k = float(np.clip(target / port_vol, cfg.vol_target_min_scale, 1.0))
    notes.append(f"vol_target: est {port_vol:.3f} > {target:.3f}, book scaled x{k:.2f}")
    return core * k, sleeve * k
