"""
Is a market-timing rule picking the right days, or only being out of the market? (tracker MIT1)

From Aiuchi (MIT Sloan, 2008), Ch. 4, with two corrections.  A rule splits ``N`` sessions into ``N_in`` in the market
and the rest out.  With ``r`` the mean return, ``r_in`` the mean over the days in and ``p = N_in / N``:

* **Timing excess** ``p r_in - r`` is what the rule added to or took from buy-and-hold, cash at zero (Sect. 4.4).
  Being out of a rising market for half the time costs half of it, whatever the skill, so the fair question is
  whether ``r_in`` beats ``r``, the mean of days picked at random.
* **Upper boundary:** the mean return of perfect timing with the same ``p``: the ``N_in`` best days (Sect. 4.2).
  ``capture`` is the share of the gap between random and perfect that the rule achieved.
* **Lower boundary:** ``p r`` plus ``z`` standard errors.  Days are drawn without replacement, so the standard error
  carries ``(1 - p)``: ``z s sqrt(p (1 - p) / N)``.  Sect. 4.3 leaves that factor out, so its boundary sits above
  buy-and-hold at ``p = 1``, where a rule has no choice to make.
* **Rotation test:** random days are the wrong null for a rule that stays in for months, and the market's
  volatility clusters.  Every circular rotation of the real signal keeps its run lengths and share of days and
  breaks only its timing.  The p-value is the share of rotations that did at least as well.  Table 5.5 uses
  independent daily coins, which ignore both.

Nothing here reads costs or cash: it asks about days, not about whether a rule is worth trading.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


@dataclass(frozen=True)
class TimingConfig:
    alpha: float = 0.05                 # one-sided level of the lower boundary
    min_sessions: int = 252
    min_switches: int = 2               # fewer and there is no timing to test: one entry, one exit
    min_rotation_gap: int = 21          # rotations this close to the signal are nearly the signal itself
    periods_per_year: int = 252
    tolerance: float = 1e-12            # a rotation counts as a tie within this


def _aligned(returns: pd.Series, in_market: pd.Series, cfg: TimingConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Returns and a 0/1 signal on the dates both have.  The signal must already be the position held over each
    session's return (decided before it), or this test measures look-ahead."""
    if not isinstance(returns, pd.Series) or not isinstance(in_market, pd.Series):
        raise TypeError("returns and in_market must be pandas Series indexed by date")
    both = pd.concat({"r": returns, "s": in_market}, axis=1, join="inner")
    if len(both) < cfg.min_sessions:
        raise ValueError(f"only {len(both)} aligned sessions, {cfg.min_sessions} needed")
    r, s = both["r"].to_numpy(dtype="float64"), both["s"].to_numpy(dtype="float64")
    if not np.isfinite(r).all() or not np.isfinite(s).all():
        raise ValueError("returns or signal not finite on an aligned date")
    if not np.isin(s, (0.0, 1.0)).all():
        raise ValueError("in_market must be 0 or 1 on every session")
    if s.min() == s.max():
        raise ValueError("the signal never changes state: nothing to test")
    switches = int(np.abs(np.diff(s)).sum())
    if switches < cfg.min_switches:
        raise ValueError(f"only {switches} switches, {cfg.min_switches} needed")
    return r, s


def upper_boundary(returns: np.ndarray, p_in: float) -> float:
    """Mean return per session of perfect timing at ``p_in``: the best ``N_in`` days, all sessions in the divisor."""
    if not 0.0 <= p_in <= 1.0:
        raise ValueError(f"p_in must lie in [0, 1], got {p_in}")
    n_in = int(round(p_in * len(returns)))
    return float(np.sort(returns)[::-1][:n_in].sum() / len(returns))


def lower_boundary(mean: float, sd: float, p_in: float, n: int, alpha: float = 0.05) -> float:
    """Mean return per session a rule must reach at ``p_in`` to beat random days at level ``alpha``.  ``mean``, ``sd``
    and ``n`` must all be per session: annual volatility with a daily ``n`` shrinks the band by sqrt(periods a year)."""
    if not 0.0 <= p_in <= 1.0 or sd < 0 or n < 2 or not 0.0 < alpha < 1.0:
        raise ValueError(f"bad inputs: p_in {p_in}, sd {sd}, n {n}, alpha {alpha}")
    z = float(sp_stats.norm.ppf(1.0 - alpha))
    return z * sd * math.sqrt(p_in * (1.0 - p_in) / n) + p_in * mean


def rotation_pvalue(returns: np.ndarray, signal: np.ndarray, cfg: TimingConfig) -> Tuple[float, int]:
    """(share of rotations at least as good as the real signal, rotations used); the real one counts once."""
    n = len(returns)
    actual = float(returns @ signal)
    shifts = np.arange(cfg.min_rotation_gap, n - cfg.min_rotation_gap + 1)
    if not len(shifts):
        raise ValueError(f"{n} sessions leave no rotation beyond the gap of {cfg.min_rotation_gap}")
    # the circular correlation at every shift at once; the rounding it adds is far below the tolerance
    spectrum = np.fft.irfft(np.fft.rfft(returns) * np.conj(np.fft.rfft(signal)), n)
    rotated = spectrum[(-shifts) % n]
    return float((1 + np.sum(rotated >= actual - cfg.tolerance)) / (1 + len(shifts))), int(len(shifts))


def timing_skill(returns: pd.Series, in_market: pd.Series, cfg: TimingConfig = TimingConfig()) -> Dict[str, float]:
    """Everything above, annualised by simple sum.  ``returns`` are of the asset, whether or not the rule held it."""
    r, s = _aligned(returns, in_market, cfg)
    n, ppy = len(r), cfg.periods_per_year
    p_in = float(s.mean())
    mean, sd = float(r.mean()), float(r.std(ddof=1))
    if sd <= 0:
        raise ValueError("returns have no variance")
    held = float(r @ s) / n
    upper = upper_boundary(r, p_in)
    random_days = p_in * mean
    room = upper - random_days
    p_value, rotations = rotation_pvalue(r, s, cfg)
    return {"sessions": n, "p_in": p_in, "switches_year": float(np.abs(np.diff(s)).sum()) / n * ppy,
            "r_in": float(r[s == 1].mean()) * ppy, "r_out": float(r[s == 0].mean()) * ppy,
            "buy_hold": mean * ppy, "held": held * ppy, "timing_excess": (held - mean) * ppy,
            "random_days": random_days * ppy, "lower": lower_boundary(mean, sd, p_in, n, cfg.alpha) * ppy,
            "upper": upper * ppy, "capture": (held - random_days) / room if room > 0 else float("nan"),
            "rotation_p": p_value, "rotations": rotations}
