"""NSE cash-equity instruments from Kite's daily dump: each symbol's tick and series (LN-T1, LN-T5).

Since 15 Apr 2025 an NSE share's tick depends on its price (NSE circular
33/2025, reviewed monthly on the last close): below Rs 250 Rs 0.01, to
1,000 Rs 0.05, to 5,000 Rs 0.10, to 10,000 Rs 0.50, to 20,000 Rs 1, above
that Rs 5.  Kite rejects a price off the symbol's tick, so every live order
and GTT price is rounded to the tick in Kite's instruments dump, read once a
day per process.  When the dump cannot be read or lacks the symbol, the price
is rounded to the tick one slab coarser than its own: the ticks divide each
other, so that price is on the grid whichever neighbouring slab NSE assigned
at its last review.  Callers report that fallback as an alert.

The engine names a stock by its canonical symbol; Kite by its tradingsymbol,
which carries the series for anything but EQ: a stock moved to BE (trade for
trade) is ``SYM-BE`` and has no bare instrument, so an order for ``SYM`` is
rejected (HFCL, MTARTECH and STLTECH trade only as BE in Oct 2026).
``to_broker`` sends the bare symbol when Kite lists it, else ``SYM-BE`` (the
only fallback, with a note: never other suffixes such as -RE rights or -D1),
else nothing.  ``to_engine`` maps Kite's names back by removing a series
suffix, so holdings, orders and GTTs of ``SYM-BE`` belong to ``SYM``.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

#: (upper price bound, tick) per slab, NSE circular 33/2025.
SLABS: Tuple[Tuple[float, float], ...] = ((250.0, 0.01), (1000.0, 0.05), (5000.0, 0.10),
                                         (10000.0, 0.50), (20000.0, 1.0), (math.inf, 5.0))
_IST = timezone(timedelta(hours=5, minutes=30))
_cache: Dict[str, Dict[str, float]] = {}            # IST date -> {tradingsymbol: tick}


def slab_tick(price: float) -> float:
    """The tick of the slab ``price`` falls in."""
    for bound, tick in SLABS:
        if float(price) < bound:
            return tick
    return SLABS[-1][1]


def safe_tick(price: float) -> float:
    """The tick one slab coarser than ``price``'s own: on the grid of either neighbouring slab."""
    ticks = [t for _, t in SLABS]
    return ticks[min(ticks.index(slab_tick(price)) + 1, len(ticks) - 1)]


def round_price(price: float, tick: float, mode: str = "nearest") -> float:
    """``price`` on the ``tick`` grid (``mode``: nearest | down | up), never below one tick."""
    n = float(price) / tick
    n = math.floor(n + 1e-9) if mode == "down" else math.ceil(n - 1e-9) if mode == "up" else math.floor(n + 0.5)
    return round(max(n, 1) * tick, 2)


def on_tick(price: float, tick: float) -> bool:
    """Whether ``price`` is a whole number of ticks."""
    n = float(price) / tick
    return abs(n - round(n)) < 1e-6


def ticks_for(kite) -> Optional[Dict[str, float]]:
    """{NSE tradingsymbol: tick} from Kite's instruments dump, once a day; None when it cannot be read."""
    day = datetime.now(_IST).date().isoformat()
    if day in _cache:
        return _cache[day]
    if kite is None:
        return None
    try:
        rows = kite.instruments("NSE") or []
        ticks = {str(r["tradingsymbol"]): float(r["tick_size"]) for r in rows
                 if str(r.get("segment") or "NSE") == "NSE" and str(r.get("instrument_type") or "EQ") == "EQ"
                 and r.get("tradingsymbol") and float(r.get("tick_size") or 0) > 0}
    except Exception as exc:                              # noqa: BLE001 - the caller falls back and alerts
        logger.warning("Kite instruments dump unavailable (%s): ticks fall back to the coarser slab", exc)
        return None
    if not ticks:
        return None
    _cache.clear()
    _cache[day] = ticks
    return ticks


def tick_for(symbol: str, price: float, ticks: Optional[Mapping[str, float]]) -> Tuple[float, bool]:
    """(tick, from the dump) for ``symbol`` at ``price``; the coarser slab's tick when the dump lacks it."""
    t = (ticks or {}).get(symbol)
    if t:
        return float(t), True
    return safe_tick(price), False


#: NSE series that Kite appends to a tradingsymbol (EQ has none).
SERIES_SUFFIXES = ("BE", "BZ", "BL", "SM", "ST", "IL", "T0")


def to_broker(symbol: str, ticks: Optional[Mapping[str, float]]) -> Tuple[Optional[str], str]:
    """(Kite tradingsymbol or None, note) for the engine's ``symbol``; as is when the dump is unknown."""
    if ticks is None or symbol in ticks:
        return symbol, ""
    if f"{symbol}-BE" in ticks:
        return f"{symbol}-BE", f"{symbol} trades as {symbol}-BE (series BE, trade for trade)"
    return None, f"{symbol}: not in Kite's NSE equity list (no EQ or BE instrument)"


def to_engine(tradingsymbol: str) -> str:
    """The engine's symbol for a Kite tradingsymbol: a series suffix removed (HFCL-BE -> HFCL)."""
    base, sep, suffix = str(tradingsymbol or "").rpartition("-")
    return base if sep and base and suffix in SERIES_SUFFIXES else str(tradingsymbol or "")
