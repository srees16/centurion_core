"""
Does a variable explain the cross-section of returns once the others are held fixed? (tracker FF1)

Fama and French (1992), after Fama and MacBeth (1973).  Each period, returns across stocks are regressed on what was
known before it; the premium on a variable is the time-series mean of its slope, and its t-statistic is that mean over
its standard error.  Unlike a sorted portfolio this holds the other variables fixed, so it asks what a signal adds.

Slopes are read month by month, so they carry no look-ahead only if ``variables`` were known before ``returns`` began:
the caller lags them.  Slopes are in the units of the variable: a stock at +1 on it earned the slope more than one at 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .diagnostics import _require_full_rank, newey_west_cov


@dataclass(frozen=True)
class FamaMacBethConfig:
    min_names: int = 30                 # a thinner month is skipped, and counted
    min_periods: int = 24
    max_skipped_share: float = 0.10     # more months skipped than this is an error, not a result
    hac_lags: int = 3                   # slopes of overlapping signals are autocorrelated


def fama_macbeth(periods: Mapping[pd.Timestamp, Tuple[pd.Series, pd.DataFrame]], variables: Sequence[str],
                 cfg: FamaMacBethConfig = FamaMacBethConfig()) -> Dict[str, object]:
    """``periods`` maps a date to (returns, variables), both indexed by stock.  One row per stock with a return and
    every variable; a column with no variation in a month is an error from ``_require_full_rank``."""
    names = ["intercept", *variables]
    slopes, counts, skipped = {}, [], 0
    for day, (returns, table) in sorted(periods.items()):
        frame = pd.concat([pd.Series(returns, dtype="float64").rename("__r"), table[list(variables)]], axis=1,
                          join="inner").replace([np.inf, -np.inf], np.nan).dropna()
        if len(frame) < max(cfg.min_names, len(names) + 1):
            skipped += 1
            continue
        x = np.column_stack([np.ones(len(frame)), frame[list(variables)].to_numpy(dtype="float64")])
        _require_full_rank(x)
        slopes[day] = np.linalg.lstsq(x, frame["__r"].to_numpy(dtype="float64"), rcond=None)[0]
        counts.append(len(frame))
    used = len(slopes)
    if used < cfg.min_periods:
        raise ValueError(f"only {used} usable periods, {cfg.min_periods} needed")
    if skipped / (used + skipped) > cfg.max_skipped_share:
        raise ValueError(f"{skipped} of {used + skipped} periods had under {cfg.min_names} names")
    table = pd.DataFrame.from_dict(slopes, orient="index", columns=names).sort_index()
    out: Dict[str, object] = {"periods": used, "skipped": skipped, "mean_names": float(np.mean(counts))}
    ones = np.ones((used, 1))
    for name in names:
        s = table[name].to_numpy()
        mean, sd = float(s.mean()), float(s.std(ddof=1))
        hac = float(newey_west_cov(ones, s - mean, cfg.hac_lags)[0, 0])
        out[name] = {"mean": mean, "std": sd, "t": mean / (sd / math.sqrt(used)) if sd > 0 else float("nan"),
                     "t_hac": mean / math.sqrt(hac) if hac > 0 else float("nan")}
    out["slopes"] = table
    return out
