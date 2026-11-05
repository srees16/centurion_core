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


#: Fill audit (golden-target review, "do now"): the proposed pass rule - buys whose open gapped most above the
#: decision close fill within this many points of the rest under live's DAY limit.  A fail re-opens the AMO entry
#: before any rung.
GAP_TERCILE_TOLERANCE_PTS = 10.0


def fill_audit(orders: "pd.DataFrame", bars: "pd.DataFrame", buy_band: float, sell_band: float,
               tick) -> "pd.DataFrame":
    """What live's after-market DAY limit would have done with each paper order (tracker LN-T21b).

    ``orders``: one row per order with ``symbol``, ``side``, ``decision_date`` and ``session_date``.  ``bars``:
    the store's as-printed daily bars (``date``, ``symbol``, ``open``, ``high``, ``low``, ``close``,
    ``prev_close``).  The limit is live's: the decision close through the band, rounded to the tick away from
    the close (``tick(price, "up" | "down")``).  An order whose session prints a different previous close than
    its decision close crossed an ex-date and is held back, as live does (LN-T4); one without both bars is
    excluded and says why.  Nothing is dropped silently: every order comes back with its ``outcome``.
    """
    import pandas as pd

    need = {"symbol", "side", "decision_date", "session_date"}
    missing = need - set(orders.columns)
    if missing:
        raise ValueError(f"fill_audit: orders lack {sorted(missing)}")
    if not 0 < buy_band < 1 or not 0 < sell_band < 1:
        raise ValueError(f"fill_audit: bands must be fractions in (0, 1), got {buy_band}, {sell_band}")
    b = bars.assign(date=pd.to_datetime(bars["date"]).dt.strftime("%Y-%m-%d"))
    b = b.drop_duplicates(["date", "symbol"]).set_index(["date", "symbol"])
    rows = []
    for o in orders.itertuples(index=False):
        side = str(o.side).upper()
        if side not in ("BUY", "SELL"):
            raise ValueError(f"fill_audit: side {o.side!r} for {o.symbol} on {o.session_date}")
        rec = {"symbol": o.symbol, "side": side, "decision_date": o.decision_date, "session_date": o.session_date}
        d, s = (o.decision_date, o.symbol), (o.session_date, o.symbol)
        if d not in b.index or s not in b.index:
            rows.append({**rec, "outcome": "no_bar"})
            continue
        dec, ses = b.loc[d], b.loc[s]
        close = float(dec["close"])
        if not (np.isfinite(close) and close > 0):
            raise ValueError(f"fill_audit: decision close {close} for {o.symbol} on {o.decision_date}")
        limit = tick(close * (1 + buy_band), "up") if side == "BUY" else tick(close * (1 - sell_band), "down")
        rec.update(decision_close=close, limit=limit, open=float(ses["open"]),
                   gap=float(ses["open"]) / close - 1.0)
        prev = float(ses["prev_close"])
        if np.isfinite(prev) and abs(prev / close - 1.0) > 1e-4:
            rows.append({**rec, "outcome": "ex_date"})
            continue
        kind, price = limit_fill(side, float(ses["open"]), float(ses["high"]), float(ses["low"]), limit)
        rows.append({**rec, "outcome": kind, "fill_price": price})
    return pd.DataFrame(rows)


def gap_terciles(audit: "pd.DataFrame", side: str = "BUY",
                 tolerance_pts: float = GAP_TERCILE_TOLERANCE_PTS) -> dict:
    """Fill rate by open-gap tercile for one side, and the pass rule: the tercile gapping most against the order
    (highest for a buy, lowest for a sell) fills within ``tolerance_pts`` of the other two together.  Only orders
    with an outcome on a bar count; ``ex_date`` and ``no_bar`` are reported, not rated."""
    import pandas as pd

    s = audit[audit["side"] == side]
    rated = s[s["outcome"].isin([OPEN, AT_LIMIT, UNFILLED])].copy()
    out = {"side": side, "orders": int(len(s)), "rated": int(len(rated)),
           "excluded": {k: int(v) for k, v in s.loc[~s.index.isin(rated.index), "outcome"].value_counts().items()}}
    if len(rated) < 3:
        return {**out, "verdict": "INSUFFICIENT", "terciles": []}
    rated["tercile"] = pd.qcut(rated["gap"].rank(method="first"), 3, labels=["low", "mid", "high"])
    filled = rated["outcome"] != UNFILLED
    terciles = []
    for t, g in rated.groupby("tercile", observed=True):
        f = filled[g.index]
        terciles.append({"tercile": str(t), "n": int(len(g)), "gap_min_pct": round(100 * g["gap"].min(), 2),
                         "gap_max_pct": round(100 * g["gap"].max(), 2), "fill_pct": round(100 * f.mean(), 1),
                         "at_open_pct": round(100 * (g["outcome"] == OPEN).mean(), 1)})
    worst = "high" if side == "BUY" else "low"
    in_worst = rated["tercile"] == worst
    worst_rate, rest_rate = 100 * filled[in_worst].mean(), 100 * filled[~in_worst].mean()
    gap_pts = rest_rate - worst_rate
    return {**out, "terciles": terciles, "worst_tercile": worst, "worst_fill_pct": round(worst_rate, 1),
            "rest_fill_pct": round(rest_rate, 1), "shortfall_pts": round(gap_pts, 1),
            "tolerance_pts": tolerance_pts, "verdict": "PASS" if gap_pts <= tolerance_pts else "FAIL"}
