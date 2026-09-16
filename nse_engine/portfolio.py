"""
Core long-only stock book: selection, rank-drop exits, weights and stops.

Pure functions over one decision date; ``nse_engine.engine.generate_targets``
composes them.

* Candidates are universe symbols with combined forecast > 0, ranked
  descending (rank 1 = strongest).
* A held name is kept while rank <= ``exit_rank`` AND forecast > 0 AND its
  stop was not hit; otherwise it exits with reason ``stop``,
  ``forecast_exit`` or ``rank_exit`` (a name that left the universe has no
  rank and exits with ``rank_exit``).
* On rebalance days the book is filled up to ``target_positions`` (never more
  than ``max_positions``) from the top ranks, skipping names in stop cooldown.
* Weights are proportional to forecast / annualised vol, capped at
  ``max_weight`` and at ``sector_cap`` per known sector, normalised to sum 1
  within the core budget (less than 1 only when the caps make 1 infeasible).
* Trailing stop = highest close since entry - ``stop_atr_multiple`` x ATR,
  never lowered.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Mapping, Optional, Set

import numpy as np
import pandas as pd

from nse_engine.config import PortfolioConfig

logger = logging.getLogger(__name__)

EXIT_STOP = "stop"
EXIT_FORECAST = "forecast_exit"
EXIT_RANK = "rank_exit"
WEIGHT_CAP_MAX_ITER = 100


def atr_panel(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Simple moving average of true range (causal)."""
    h = high.astype("float64")
    lo = low.astype("float64")
    c = close.astype("float64")
    prev = c.shift(1)
    tr = np.maximum(h - lo, np.maximum((h - prev).abs(), (lo - prev).abs()))
    tr = tr.where(h.notna() & lo.notna())
    # first row of a symbol has no previous close: fall back to high - low
    tr = tr.fillna(h - lo)
    return tr.rolling(lookback, min_periods=max(lookback // 2, 1)).mean()


def rank_candidates(forecasts: pd.Series, universe: Iterable[str]) -> pd.Series:
    """Rank (1 = best) of universe symbols with forecast > 0."""
    f = forecasts.reindex(list(universe))
    f = f[np.isfinite(f.to_numpy(dtype="float64")) & (f > 0)]
    if f.empty:
        return pd.Series(dtype="int64")
    # stable tie-break by symbol name for determinism
    order = sorted(f.index, key=lambda s: (-float(f[s]), str(s)))
    return pd.Series(np.arange(1, len(order) + 1, dtype="int64"), index=order)


def trailing_stop(
    highest_close: float, atr: float, multiple: float, previous_stop: Optional[float] = None
) -> Optional[float]:
    """Highest close since entry minus ``multiple`` x ATR, never below ``previous_stop``."""
    new = highest_close - multiple * atr if np.isfinite(highest_close) and np.isfinite(atr) else np.nan
    prev = previous_stop if previous_stop is not None and np.isfinite(previous_stop) else np.nan
    if np.isfinite(new) and np.isfinite(prev):
        return float(max(new, prev))
    if np.isfinite(new):
        return float(new)
    if np.isfinite(prev):
        return float(prev)
    return None


def stop_fill_price(open_price: float, low: float, stop_price: Optional[float]) -> Optional[float]:
    """Fill price of a resting stop on one day, or None if not triggered.

    Triggered when ``low <= stop``; fills at ``min(open, stop)`` so a gap
    below the stop fills at the (worse) open.
    """
    if stop_price is None or not np.isfinite(stop_price):
        return None
    if np.isfinite(open_price) and open_price <= stop_price:
        return float(open_price)
    if np.isfinite(low) and low <= stop_price:
        return float(min(open_price, stop_price)) if np.isfinite(open_price) else float(stop_price)
    return None


def exit_reason(
    rank: Optional[int],
    forecast: float,
    low: float,
    stop_price: Optional[float],
    exit_rank: int,
) -> Optional[str]:
    """Reason a held core name must exit today, or None to keep it."""
    if stop_price is not None and np.isfinite(stop_price) and np.isfinite(low) and low <= stop_price:
        return EXIT_STOP
    if not (np.isfinite(forecast) and forecast > 0):
        return EXIT_FORECAST
    if rank is None or rank > exit_rank:
        return EXIT_RANK
    return None


def select_names(
    kept: List[str],
    ranks: pd.Series,
    blocked: Set[str],
    cfg: PortfolioConfig,
) -> tuple[List[str], List[str]]:
    """Kept names plus new entries from the top ranks.

    Returns ``(selected, dropped)`` where ``dropped`` are kept names beyond
    ``max_positions`` (worst ranks first to go).
    """
    kept_sorted = sorted(kept, key=lambda s: (int(ranks.get(s, 10**9)), s))
    max_pos = max(int(cfg.max_positions), 0)
    target = min(int(cfg.target_positions), max_pos)
    selected = kept_sorted[:max_pos]
    dropped = kept_sorted[max_pos:]
    held = set(kept)
    for sym in ranks.index:
        if len(selected) >= target:
            break
        if sym in held or sym in blocked:
            continue
        selected.append(sym)
    return selected, dropped


def capped_weights(
    raw: pd.Series,
    max_weight: float,
    sector_cap: Optional[float] = None,
    sectors: Optional[Mapping[str, str]] = None,
) -> pd.Series:
    """Normalise positive ``raw`` scores to sum 1 subject to name and sector caps.

    Water-filling: capped names/sectors are frozen and the excess is spread
    over the rest in proportion to their scores.  If the caps make a sum of 1
    infeasible the result sums to less than 1 (the remainder is cash).
    Symbols without a known sector are never capped as a group.
    """
    r = raw[np.isfinite(raw.to_numpy(dtype="float64")) & (raw > 0)].astype("float64")
    if r.empty:
        return pd.Series(dtype="float64")
    sectors = sectors or {}
    sec = pd.Series({s: sectors.get(s) for s in r.index}, dtype=object)
    use_sector = sector_cap is not None and sector_cap > 0 and sec.notna().any()
    w = r / r.sum()
    fixed = pd.Series(False, index=r.index)
    for _ in range(WEIGHT_CAP_MAX_ITER):
        changed = False
        over = (w > max_weight + 1e-12) & ~fixed
        if over.any():
            w[over] = max_weight
            fixed |= over
            changed = True
        if use_sector:
            sec_tot = w.groupby(sec).sum()
            for s_name, tot in sec_tot.items():
                if tot > sector_cap + 1e-12:
                    members = sec.index[sec == s_name]
                    w[members] *= sector_cap / tot
                    fixed[members] = True
                    changed = True
        free = ~fixed
        budget = 1.0 - w[fixed].sum()
        if free.any() and budget > 1e-12:
            new_free = r[free] / r[free].sum() * budget
            if not np.allclose(new_free.to_numpy(), w[free].to_numpy(), atol=1e-12):
                w[free] = new_free
                changed = True
        if not changed:
            break
        if not free.any():
            break
    # final guard (numerical)
    w = w.clip(upper=max_weight)
    if w.sum() > 1.0:
        w /= w.sum()
    return w


def core_weights(
    forecasts: pd.Series,
    vols: pd.Series,
    cfg: PortfolioConfig,
    sectors: Optional[Mapping[str, str]] = None,
) -> pd.Series:
    """Relative core weights (sum <= 1): forecast / annualised vol with caps."""
    f = forecasts.astype("float64")
    v = vols.reindex(f.index).astype("float64")
    fallback = float(np.nanmedian(v.to_numpy())) if np.isfinite(v.to_numpy()).any() else 0.3
    v = v.where(np.isfinite(v) & (v > 0), fallback)
    raw = f.clip(lower=0) / v
    return capped_weights(raw, cfg.max_weight, cfg.sector_cap, sectors)


def apply_no_trade_buffer(
    target: Mapping[str, float],
    current: Mapping[str, float],
    equity: float,
    buffer: float,
    min_trade_value_inr: float,
) -> Dict[str, float]:
    """Keep the current weight when the change is small.

    For names with a positive target: keep current weight if
    |target - current| < buffer x target, or if the traded value would be
    below ``min_trade_value_inr``.  A new entry below the minimum trade value
    is skipped.  Full exits (target 0) always trade.
    """
    out: Dict[str, float] = {}
    for sym in set(target) | set(current):
        t = float(target.get(sym, 0.0))
        c = float(current.get(sym, 0.0))
        if t <= 0:
            continue  # exit (or never held)
        diff = abs(t - c)
        if c > 0 and diff < buffer * t:
            out[sym] = c
        elif diff * equity < min_trade_value_inr:
            if c > 0:
                out[sym] = c
        else:
            out[sym] = t
    return out
