"""
How much worse can costs run before a book stops being worth holding? (tracker OPS1)

Lahanis, Liu and Zhou's mean-reversion rules turn Rs 1 into Rs 1,000 with no costs and into less than Rs 1 at 25 bp a
side: a gross result says nothing until its break-even cost is known.  ``cost_margin`` asks that of any book with
recorded costs: by how many times its modelled costs can it be multiplied before its CAGR falls to a hurdle?

    net(k) = returns - (k - 1) x costs

``returns`` are net of the modelled costs, so ``k = 1`` is the book as recorded.  CAGR falls as ``k`` rises, so the root
is found by bisection, and a root that does not settle within ``max_iterations`` is an error, not an answer.

Live-like returns already carry D2's haircut.  If that gap between backtest and live is all costs, the same rupees are
counted twice, so read ``multiple`` as the floor and ``multiple_within_haircut`` as the ceiling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


DAYS_PER_YEAR = 365.25


@dataclass(frozen=True)
class CostMarginConfig:
    max_multiple: float = 20.0          # a book that survives this has no cost problem
    tolerance: float = 1e-6             # on the multiple
    max_iterations: int = 200
    min_sessions: int = 252


def _cagr(returns: np.ndarray, years: float) -> float:
    """Compound annual growth over calendar ``years``; -100% once a day loses everything."""
    if (returns <= -1.0).any():
        return -1.0
    return math.exp(float(np.log1p(returns).sum()) / years) - 1.0


def cost_margin(returns: pd.Series, costs: pd.Series, hurdle_cagr: float, haircut_annual: float = 0.0,
                cfg: CostMarginConfig = CostMarginConfig()) -> Dict[str, float]:
    """The cost multiple at which ``returns`` fall to ``hurdle_cagr``.  ``costs`` are daily fractions of equity, on
    the same dates; a date with no cost must say 0, not be missing."""
    both = pd.concat({"r": returns, "c": costs}, axis=1, join="outer")
    if len(both) < cfg.min_sessions:
        raise ValueError(f"only {len(both)} sessions, {cfg.min_sessions} needed")
    r, c = both["r"].to_numpy(dtype="float64"), both["c"].to_numpy(dtype="float64")
    if not np.isfinite(r).all() or not np.isfinite(c).all():
        raise ValueError("returns and costs must be finite on every session of either series")
    if (c < 0).any():
        raise ValueError("costs must not be negative")
    years = (both.index[-1] - both.index[0]).days / DAYS_PER_YEAR      # NSE has about 248 sessions a year, not 252
    annual_cost = float(c.sum()) / years
    if annual_cost <= 0:
        raise ValueError("no costs recorded: nothing to multiply")
    if haircut_annual < 0:
        raise ValueError(f"haircut must not be negative, got {haircut_annual}")
    at = lambda k: _cagr(r - (k - 1.0) * c, years)

    base = at(1.0)
    out = {"cagr": base, "annual_cost": annual_cost, "hurdle": hurdle_cagr,
           "multiple_within_haircut": 1.0 + haircut_annual / annual_cost,
           "cagr_at_1_5x": at(1.5), "cagr_at_2x": at(2.0)}
    if base <= hurdle_cagr:
        return {**out, "multiple": 1.0, "converged": True, "iterations": 0}
    if at(cfg.max_multiple) > hurdle_cagr:
        return {**out, "multiple": math.inf, "converged": True, "iterations": 0}
    low, high = 1.0, cfg.max_multiple
    for i in range(1, cfg.max_iterations + 1):
        mid = (low + high) / 2.0
        if at(mid) > hurdle_cagr:
            low = mid
        else:
            high = mid
        if high - low <= cfg.tolerance:
            return {**out, "multiple": (low + high) / 2.0, "converged": True, "iterations": i}
    raise ArithmeticError(f"cost multiple not settled in {cfg.max_iterations} iterations: [{low}, {high}]")
