"""
Point-in-time liquidity universe.

On refresh positions (every ``refresh_every_n_days`` rows of the trading
calendar, counted from the first row of the panel) the universe is the
``top_n_liquid`` symbols by median traded value among those that have

* at least ``min_history_days`` closes,
* a last close >= ``min_price_inr``,
* median traded value (untraded days = 0) over ``liquidity_lookback_days``
  >= ``min_median_value_inr``,
* and are not ETFs (if configured) or explicitly excluded (sleeve ETFs).

Between refreshes the last universe is reused, minus symbols with no close on
the day (suspended or delisted).  Only rows <= the date are read, so the
result is survivorship-free when the panel contains delisted symbols.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from nse_engine.config import UniverseConfig
from nse_engine.types import MarketData

logger = logging.getLogger(__name__)


@dataclass
class UniversePanel:
    """Universe membership for every date of a panel."""

    mask: pd.DataFrame  # date x symbol bool (refresh universe AND trading that day)
    refresh: Dict[pd.Timestamp, List[str]]  # refresh date -> ranked members

    def members(self, as_of: pd.Timestamp) -> List[str]:
        row = self.mask.loc[pd.Timestamp(as_of)]
        return list(row.index[row.to_numpy()])


def is_refresh_position(pos: int, every: int) -> bool:
    return pos % max(int(every), 1) == 0


def last_refresh_position(pos: int, every: int) -> int:
    every = max(int(every), 1)
    return (pos // every) * every


def _eligible_mask(data: MarketData, cfg: UniverseConfig, exclude: Iterable[str]) -> np.ndarray:
    symbols = data.close.columns
    excluded = set(exclude)
    if cfg.exclude_etfs:
        excluded |= set(data.etfs)
    return ~np.asarray(symbols.isin(list(excluded)))


def _select_at(
    last_close: np.ndarray,
    value0: np.ndarray,
    hist_count_row: np.ndarray,
    pos: int,
    eligible: np.ndarray,
    cfg: UniverseConfig,
) -> np.ndarray:
    """Column indices (ranked by liquidity) of the universe at row ``pos``."""
    lo = max(0, pos - cfg.liquidity_lookback_days + 1)
    window = value0[lo : pos + 1]
    med = np.median(window, axis=0)
    ok = (
        eligible
        & (hist_count_row >= cfg.min_history_days)
        & np.isfinite(last_close)
        & (last_close >= cfg.min_price_inr)
        & (med >= cfg.min_median_value_inr)
    )
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return idx
    order = np.argsort(-med[idx], kind="stable")
    return idx[order][: cfg.top_n_liquid]


def _last_valid_row(arr: np.ndarray) -> np.ndarray:
    """Last finite value per column of a 2-D array (NaN if none)."""
    n = arr.shape[0]
    finite = np.isfinite(arr)
    # index of last finite row per column
    rev_idx = np.argmax(finite[::-1], axis=0)
    has = finite.any(axis=0)
    rows = n - 1 - rev_idx
    out = np.full(arr.shape[1], np.nan)
    cols = np.flatnonzero(has)
    out[cols] = arr[rows[cols], cols]
    return out


def history_counts(close: np.ndarray, window: int) -> np.ndarray:
    """Finite closes per symbol up to each row: since row 0 (``window`` 0) or within the trailing window.

    Counting since row 0 makes a name's eligibility depend on where the data
    was loaded from; the trailing window does not. Integer cumulative sums are
    exact, so both are the same numbers for any load start once ``window``
    rows precede the date.
    """
    cum = np.cumsum(np.isfinite(close), axis=0)
    if window <= 0:
        return cum
    out = cum.copy()
    out[window:] = cum[window:] - cum[:-window]
    return out


def refresh_positions(dates: pd.DatetimeIndex, cfg: UniverseConfig) -> np.ndarray:
    """Rows on which the universe is re-selected."""
    every = max(int(cfg.refresh_every_n_days), 1)
    if cfg.calendar_schedule:
        from nse_engine.calendar import period_start_mask
        return np.flatnonzero(period_start_mask(dates, every))
    return np.arange(0, len(dates), every)


def compute_universe_panel(data: MarketData, cfg: UniverseConfig, exclude: Iterable[str] = ()) -> UniversePanel:
    """Universe membership for every date (vectorised over refresh dates)."""
    close = data.close.to_numpy(dtype="float64")
    value0 = np.nan_to_num(data.value.to_numpy(dtype="float64"), nan=0.0)
    n, m = close.shape
    hist = history_counts(close, cfg.history_window_days)
    close_ff = data.close.astype("float64").ffill().to_numpy()
    eligible = _eligible_mask(data, cfg, exclude)
    mask = np.zeros((n, m), dtype=bool)
    refresh: Dict[pd.Timestamp, List[str]] = {}
    cols = data.close.columns
    trading = np.isfinite(close)
    starts = refresh_positions(data.dates, cfg)
    for i, pos in enumerate(starts):
        idx = _select_at(close_ff[pos], value0, hist[pos], pos, eligible, cfg)
        refresh[data.dates[pos]] = [cols[i2] for i2 in idx]
        base = np.zeros(m, dtype=bool)
        base[idx] = True
        end = int(starts[i + 1]) if i + 1 < len(starts) else n
        mask[pos:end] = base[None, :] & trading[pos:end]
    return UniversePanel(mask=pd.DataFrame(mask, index=data.dates, columns=cols), refresh=refresh)


def select_universe(
    data: MarketData, cfg: UniverseConfig, as_of: pd.Timestamp, exclude: Iterable[str] = ()
) -> List[str]:
    """Universe on ``as_of`` using only rows <= ``as_of`` (single-date version)."""
    as_of = pd.Timestamp(as_of)
    pos = int(data.dates.searchsorted(as_of, side="right")) - 1
    if pos < 0:
        return []
    if cfg.calendar_schedule:
        starts = refresh_positions(data.dates[: pos + 1], cfg)
        if starts.size == 0:
            return []
        rpos = int(starts[-1])
    else:
        rpos = last_refresh_position(pos, cfg.refresh_every_n_days)
    close = data.close.iloc[: pos + 1].to_numpy(dtype="float64")
    value0 = np.nan_to_num(data.value.iloc[: rpos + 1].to_numpy(dtype="float64"), nan=0.0)
    hist = history_counts(close[: rpos + 1], cfg.history_window_days)[rpos]
    idx = _select_at(_last_valid_row(close[: rpos + 1]), value0, hist, rpos, _eligible_mask(data, cfg, exclude), cfg)
    cols = data.close.columns
    return [cols[i] for i in idx if np.isfinite(close[pos, i])]
