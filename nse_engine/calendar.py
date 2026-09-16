"""Calendar-anchored schedules.

The engine used to rebalance, refresh the universe and re-estimate the FDM
every ``n`` rows counted from the first *loaded* row, so the same
configuration produced different trades depending on where the data was
loaded from (measured: +18.0% over 2026 loaded from 2011, +12.8% loaded from
2024). A schedule anchored to the calendar does not care where the data
starts: a session is a rebalance day because of its date, not its row number.

``every_n_days`` keeps its meaning as a period length in trading days and
maps to the calendar period it approximates:

    1        every session
    2 - 5    the first session of each ISO week
    6 - 21   the first session of each month
    22 - 63  the first session of each quarter
    > 63     the first session of each year
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def period_labels(dates: pd.DatetimeIndex, every_n_days: int) -> np.ndarray:
    """One integer label per session; sessions in the same period share it."""
    n = max(int(every_n_days), 1)
    if n == 1:
        return np.arange(len(dates))
    iso = dates.isocalendar()
    if n <= 5:
        return (iso.year.to_numpy() * 100 + iso.week.to_numpy()).astype(np.int64)
    if n <= 21:
        return (dates.year * 100 + dates.month).to_numpy().astype(np.int64)
    if n <= 63:
        return (dates.year * 10 + dates.quarter).to_numpy().astype(np.int64)
    return dates.year.to_numpy().astype(np.int64)


def period_start_mask(dates: pd.DatetimeIndex, every_n_days: int) -> np.ndarray:
    """True on the first session of each calendar period (see module docstring).

    The first loaded session counts as a period start only if it really is
    one, so a load that begins mid-week does not create a spurious schedule.
    """
    labels = period_labels(dates, every_n_days)
    if len(labels) == 0:
        return np.zeros(0, dtype=bool)
    mask = np.empty(len(labels), dtype=bool)
    mask[0] = _is_first_session_of_period(dates[0], every_n_days)
    mask[1:] = labels[1:] != labels[:-1]
    return mask


def _is_first_session_of_period(date: pd.Timestamp, every_n_days: int) -> bool:
    """Whether ``date`` is the earliest weekday of its period (calendar, not sessions).

    Used only for the very first loaded row, where the previous session is
    unknown. A holiday on the true first weekday makes this say False for the
    next session; the cost is one skipped rebalance in the warm-up region.
    """
    n = max(int(every_n_days), 1)
    if n == 1:
        return True
    date = pd.Timestamp(date).normalize()
    if n <= 5:
        first = date - pd.Timedelta(days=date.weekday())            # Monday
    elif n <= 21:
        first = date.replace(day=1)
    elif n <= 63:
        first = date.replace(month=3 * ((date.month - 1) // 3) + 1, day=1)
    else:
        first = date.replace(month=1, day=1)
    while first.weekday() >= 5:                                     # skip Sat/Sun
        first += pd.Timedelta(days=1)
    return first == date


def last_start_at_or_before(mask: np.ndarray, pos: int) -> int:
    """Index of the most recent period start at or before ``pos`` (-1 if none)."""
    starts = np.flatnonzero(mask[: pos + 1])
    return int(starts[-1]) if starts.size else -1
