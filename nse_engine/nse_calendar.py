"""One NSE cash-market calendar for Centurion (tracker LN-T7).

The store's own ``calendar.parquet`` (the dates that have a bhavcopy) is the
truth for the past.  This module holds what the store cannot know ahead of
time, for the code that runs before a session's data exists (the live
scheduler's freshness check, the market-hours ribbon, the archive sync):

* ``HOLIDAYS``: weekday closures from NSE's yearly holiday circulars;
* ``SPECIAL_SESSIONS``: sessions on a weekend or a listed holiday (Union
  Budget days, Muhurat trading, disaster-recovery drills), with their kind
  and source.  The archive probes every day anyway (a Sunday 404 costs one
  request, once); these dates are also re-checked on every sync, so a
  special session that the nightly window missed is fetched when it lands.

Muhurat trading (about an hour on Diwali evening) is a store session like
any other: backtests and the paper books process it.  The live book does
not trade it (``MUHURAT_TRADED_LIVE``): no live session runs on it, so the
orders decided before it go to the next regular open.

NSE publishes the next year's holidays in December (the 2027 list is not out
as of Oct 2026); until a year's list is here, only its weekends are known.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, FrozenSet, Tuple, Union

DateLike = Union[date, str]

#: Weekday closures, NSE trading-holiday circulars (2026: circular 212/2025 of 12 Dec 2025).
HOLIDAYS: FrozenSet[str] = frozenset({
    # 2025
    "2025-02-26", "2025-03-14", "2025-03-31", "2025-04-10", "2025-04-14",
    "2025-04-18", "2025-05-01", "2025-08-15", "2025-08-27", "2025-10-02",
    "2025-10-21", "2025-10-22", "2025-11-05", "2025-12-25",
    # 2026
    "2026-01-15", "2026-01-26", "2026-03-03", "2026-03-26", "2026-03-31",
    "2026-04-03", "2026-04-14", "2026-05-01", "2026-05-28", "2026-06-26",
    "2026-09-14", "2026-10-02", "2026-10-20", "2026-11-10", "2026-11-24",
    "2026-12-25",
})
#: Years whose NSE holiday list is in ``HOLIDAYS``.
HOLIDAY_YEARS: Tuple[int, ...] = (2025, 2026)

#: Sessions on a weekend or a listed holiday: date -> (kind, source).
SPECIAL_SESSIONS: Dict[str, Tuple[str, str]] = {
    "2013-11-03": ("muhurat", "store calendar"),
    "2016-10-30": ("muhurat", "store calendar"),
    "2019-10-27": ("muhurat", "store calendar"),
    "2020-02-01": ("budget", "store calendar"),
    "2020-11-14": ("muhurat", "store calendar"),
    "2023-11-12": ("muhurat", "store calendar"),
    "2024-01-20": ("special", "store calendar (Saturday live session)"),
    "2024-03-02": ("special", "store calendar (disaster-recovery drill)"),
    "2024-05-18": ("special", "store calendar (disaster-recovery drill)"),
    "2025-02-01": ("budget", "store calendar"),
    "2025-10-21": ("muhurat", "store calendar (on a listed holiday)"),
    "2026-02-01": ("budget", "NSE circular CM 11/2026 (NSE/CMTR/72349, 16 Jan 2026): full session 09:15-15:30"),
    "2026-11-08": ("muhurat", "NSE circular 212/2025 (12 Dec 2025); timing to be notified"),
}
MUHURAT_TRADED_LIVE = False


def _iso(d: DateLike) -> str:
    return d if isinstance(d, str) else d.isoformat()


def is_holiday(d: DateLike) -> bool:
    """A listed weekday closure (a special session on it still trades)."""
    return _iso(d) in HOLIDAYS


def is_special_session(d: DateLike) -> bool:
    return _iso(d) in SPECIAL_SESSIONS


def is_trading_day(d: date) -> bool:
    """A weekday that is not a listed holiday, or a special session (Budget day, Muhurat, drill)."""
    return (d.weekday() < 5 and not is_holiday(d)) or is_special_session(d)


def special_session_dates() -> Tuple[date, ...]:
    return tuple(sorted(date.fromisoformat(d) for d in SPECIAL_SESSIONS))
