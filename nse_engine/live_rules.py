"""The orders live actually sends, as daily-bar fill rules for a reference backtest (tracker LN-T21).

The model fills every order at the open.  Live sends after-market DAY LIMIT
orders instead - a buy at most ``ORDER_LIMIT_BAND_BPS`` above the decision
close, a sell at least ``EXIT_LIMIT_BAND_BPS`` below it - and protects a
position with a stop-limit GTT whose limit sits ``DEFAULT_LIMIT_BUFFER_PCT``
under the trigger, placed only the evening after the buy fills.  These pure
functions turn one daily bar into the fill those orders get;
``engine.run_backtest(fill_rule="live")`` applies them in a reference run
that is never recorded and stays outside the config hash.  On adjusted
prices, without tick rounding or a lower-circuit clamp (the backtest has no
circuit data), so the bands are approximations of live's.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

OPEN, AT_LIMIT, AT_TRIGGER, UNFILLED = "open", "at_limit", "at_trigger", "unfilled"


def limit_fill(side: str, open_: float, high: float, low: float, limit: float) -> Tuple[str, Optional[float]]:
    """(outcome, price) of a DAY limit order sent before the open.

    Lean's daily-bar LimitFill with strict penetration, at the open when the
    order is marketable in the call auction: a buy fills at the open when it
    is at or below the limit, else at the limit when the low goes below it; a
    sell mirrored on the high; otherwise it does not fill that day.
    """
    if not (np.isfinite(open_) and np.isfinite(limit)):
        return UNFILLED, None
    buy = str(side).upper() == "BUY"
    if (open_ <= limit) if buy else (open_ >= limit):
        return OPEN, float(open_)
    through = (np.isfinite(low) and low < limit) if buy else (np.isfinite(high) and high > limit)
    return (AT_LIMIT, float(limit)) if through else (UNFILLED, None)


def stop_gtt_fill(open_: float, high: float, low: float, trigger: float,
                  limit: float) -> Tuple[Optional[str], Optional[float]]:
    """(outcome, price) of a stop-limit GTT on one day; (None, None) when it does not trigger.

    An open at or below the trigger fills at the open while the open is at or
    above the limit, at the limit when the day trades above it, and otherwise
    not that day (the position carries to the next); a trigger touched during
    the day fills at the trigger.
    """
    if not np.isfinite(trigger):
        return None, None
    if np.isfinite(open_) and open_ <= trigger:
        if open_ >= limit:
            return OPEN, float(open_)
        return (AT_LIMIT, float(limit)) if np.isfinite(high) and high > limit else (UNFILLED, None)
    if np.isfinite(low) and low <= trigger:
        return AT_TRIGGER, float(min(open_, trigger)) if np.isfinite(open_) else float(trigger)
    return None, None
