"""The Options page's condor odds beside what NIFTY did (tracker WGF1b, 25 Nov 2026).

The page prices each strike at its own implied volatility and shows, for an iron condor, PoP (both short strikes
expire out of the money, from N(d2)) and Touch (either short strike reached at a daily close before expiry,
``theory.probability_touch``).  ``condor_calibration`` re-runs that pricing on the F&O store's history: on each
monthly expiry, the first session ``min_dte`` to ``target_dte`` days out, the out-of-the-money call and put nearest
``condor_short_delta`` among the strikes that traded that day, each at its own IV from the day's close; then what the
index did by expiry, and the same for the cycles with the IV rank the scan requires.  ``MEASURED_CONDOR`` is the
result the page states (a store guard comparing the two is queued for the 19 Dec guard review, tracker WGF2).

The odds are risk-neutral: prices grow at the risk-free rate, not at NIFTY's own drift, and the put skew carries a
crash premium.  On 2008-25 that made PoP right on average and wrong on each side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from kite_connect.options.iv_history import IV_VERSION, rank_and_percentile
from kite_connect.options.options_config import OptionsConfig
from kite_connect.options.theory import CALL, PUT, black_scholes, implied_volatility, probability_itm, probability_touch
from nse_engine.data import fo_store

#: Two years of monthly cycles before a rate is quoted.
MIN_CYCLES = 24


@dataclass(frozen=True)
class CondorOdds:
    """Rates 0-1 rounded to three places; the ``rank_`` fields are the cycles at the scan's IV rank (None if none)."""

    span: str
    cycles: int
    short_delta: float
    days_out: int
    pop_predicted: float
    pop_realised: float
    call_beyond_predicted: float
    call_beyond_realised: float
    put_beyond_predicted: float
    put_beyond_realised: float
    touch_predicted: float
    touch_realised: float
    min_rank: float
    rank_cycles: int
    rank_pop_predicted: Optional[float]
    rank_pop_realised: Optional[float]


#: ``condor_calibration()`` on the store, 25 Nov 2026 (tracker WGF1b).
MEASURED_CONDOR = CondorOdds(
    span="2008-25", cycles=207, short_delta=0.15, days_out=30,
    pop_predicted=0.694, pop_realised=0.671, call_beyond_predicted=0.138, call_beyond_realised=0.193,
    put_beyond_predicted=0.168, put_beyond_realised=0.135, touch_predicted=0.516, touch_realised=0.425,
    min_rank=50.0, rank_cycles=25, rank_pop_predicted=0.695, rank_pop_realised=0.6)


def _short_leg(day: pd.DataFrame, side: str, spot: float, dte: float, rate: float,
               delta: float) -> Optional[Tuple[float, float]]:
    """(strike, IV) of the out-of-the-money strike of ``side`` whose delta is nearest ``delta``, or None."""
    otm = day[(day["option_type"] == side) & ((day["strike"] > spot) if side == CALL else (day["strike"] < spot))]
    best: Optional[Tuple[float, float]] = None
    gap = math.inf
    for strike, close in zip(otm["strike"].to_numpy(dtype=float), otm["close"].to_numpy(dtype=float)):
        iv = implied_volatility(side, close, spot, strike, dte, rate)
        if iv is None:
            continue
        miss = abs(abs(black_scholes(side, spot, strike, dte, rate, iv).delta) - delta)
        if miss < gap:
            best, gap = (strike, iv), miss
    return best


def _pop(cycles: pd.DataFrame) -> Tuple[float, float]:
    """(predicted, realised) chance both short strikes expire out of the money."""
    realised = ((cycles["call_beyond"] == 0) & (cycles["put_beyond"] == 0)).mean()
    return round(float((1.0 - cycles["call_p"] - cycles["put_p"]).mean()), 3), round(float(realised), 3)


def condor_calibration(cfg: OptionsConfig = OptionsConfig(), start: str = "2008-01-01", end: str = "2025-12-31",
                       symbol: str = "NIFTY", min_cycles: int = MIN_CYCLES) -> CondorOdds:
    """The page's condor odds against the index, one cycle per monthly expiry from ``start`` to ``end``.

    Only strikes that traded that day (contracts and open interest above zero, a positive close) are priced: most
    listed strikes do not trade, and their close is stale.  The IV rank is the scan's: the day's 30-day IV against
    the ``lookback_sessions`` before it, from the cached IV history (read, never rebuilt here).  Raises when the store
    or the IV history is missing, or fewer than ``min_cycles`` cycles are found.
    """
    o, rate = cfg.overlay, cfg.market.risk_free_rate
    iv_path = Path(cfg.data.fo_store) / "iv" / f"{symbol}_v{IV_VERSION}.parquet"
    if not iv_path.exists():
        raise FileNotFoundError(f"no IV history at {iv_path} (iv_history.iv_history builds it)")
    hist = pd.read_parquet(iv_path, columns=["date", "iv30"]).dropna()
    iv30 = pd.Series(hist["iv30"].to_numpy(dtype=float), index=pd.DatetimeIndex(hist["date"])).sort_index()
    spot = fo_store.load_underlying(cfg.data.fo_store, symbol).dropna()
    opts = fo_store.load_options(cfg.data.fo_store, start, end, symbol)
    if opts.empty or spot.empty:
        raise ValueError(f"no {symbol} options or index closes in {cfg.data.fo_store} for {start} to {end}")
    opts = opts[(opts["contracts"] > 0) & (opts["oi"] > 0) & (opts["close"] > 0) & np.isfinite(opts["close"])]
    expiries = pd.Series(pd.to_datetime(opts["expiry"].unique())).sort_values()
    monthly = expiries.groupby(expiries.dt.to_period("M")).max()
    monthly = monthly[(monthly >= pd.Timestamp(start)) & (monthly <= pd.Timestamp(end)) & monthly.isin(spot.index)]
    pairs = []
    for e in monthly:
        gap = (e - spot.index).days
        days = spot.index[(gap >= o.min_dte) & (gap <= o.target_dte)]
        if len(days):
            pairs.append((days[0], e))
    chosen = opts[pd.MultiIndex.from_arrays([opts["date"], pd.to_datetime(opts["expiry"])]).isin(pairs)]
    rows: List[dict] = []
    for (d0, e), day in chosen.groupby(["date", pd.to_datetime(chosen["expiry"])]):
        s0, dte = float(spot.loc[d0]), float((e - d0).days)
        call, put = (_short_leg(day, side, s0, dte, rate, o.condor_short_delta) for side in (CALL, PUT))
        if call is None or put is None:
            continue
        path = spot[(spot.index > d0) & (spot.index <= e)]
        touch = sum(probability_touch(s0, k, dte, rate, iv, monitor_every_days=1.0) for k, iv in (call, put))
        rank = (rank_and_percentile(iv30[iv30.index < d0], float(iv30.loc[d0]), cfg.data.lookback_sessions)[0]
                if d0 in iv30.index else math.nan)
        rows.append({"year": e.year, "rank": rank,
                     "call_p": probability_itm(CALL, s0, call[0], dte, rate, call[1])["n_d2"],
                     "put_p": probability_itm(PUT, s0, put[0], dte, rate, put[1])["n_d2"],
                     "touch_p": min(1.0, touch),
                     "call_beyond": float(path.iloc[-1] >= call[0]), "put_beyond": float(path.iloc[-1] <= put[0]),
                     "touched": float(bool((path >= call[0]).any() or (path <= put[0]).any()))})
    if len(rows) < min_cycles:
        raise ValueError(f"{len(rows)} condor cycles found, fewer than the {min_cycles} needed to quote a rate")
    r = pd.DataFrame(rows)
    hit = r[r["rank"] >= o.min_rank_condor]
    pop, hit_pop = _pop(r), (_pop(hit) if len(hit) else (None, None))
    lo, hi = int(r["year"].min()), int(r["year"].max())
    return CondorOdds(
        span=str(lo) if lo == hi else f"{lo}-{hi % 100:02d}", cycles=len(r), short_delta=o.condor_short_delta,
        days_out=o.target_dte, pop_predicted=pop[0], pop_realised=pop[1],
        call_beyond_predicted=round(float(r["call_p"].mean()), 3),
        call_beyond_realised=round(float(r["call_beyond"].mean()), 3),
        put_beyond_predicted=round(float(r["put_p"].mean()), 3),
        put_beyond_realised=round(float(r["put_beyond"].mean()), 3),
        touch_predicted=round(float(r["touch_p"].mean()), 3), touch_realised=round(float(r["touched"].mean()), 3),
        min_rank=o.min_rank_condor, rank_cycles=len(hit), rank_pop_predicted=hit_pop[0],
        rank_pop_realised=hit_pop[1])


def measured_note(m: CondorOdds = MEASURED_CONDOR) -> str:
    """What the Options page says under the condors: the odds it shows beside what happened."""
    note = (f"Measured on NIFTY's monthly expiries {m.span} ({m.cycles} cycles, {m.short_delta:.2f}-delta shorts "
            f"about {m.days_out} days out): PoP {m.pop_predicted:.0%} predicted, {m.pop_realised:.0%} realised; "
            f"Touch {m.touch_predicted:.0%} predicted, {m.touch_realised:.0%} realised.  Each side is off, the "
            f"opposite ways: the short call finished beyond its strike {m.call_beyond_realised:.0%} of the time "
            f"({m.call_beyond_predicted:.0%} predicted), the short put {m.put_beyond_realised:.0%} "
            f"({m.put_beyond_predicted:.0%}).")
    if m.rank_cycles and m.rank_pop_predicted is not None and m.rank_pop_realised is not None:
        p = m.rank_pop_predicted
        z = (m.rank_pop_realised - p) / math.sqrt(p * (1.0 - p) / m.rank_cycles) if 0.0 < p < 1.0 else math.inf
        note += (f"  At IV rank {m.min_rank:.0f} or more, which the scan requires, {m.rank_cycles} cycles: PoP "
                 f"{m.rank_pop_realised:.0%} realised against {p:.0%} predicted"
                 + (", within chance at that count." if abs(z) < 2.0 else f", beyond chance (z {z:+.1f})."))
    return note + ("  The odds grow prices at the risk-free rate, not NIFTY's own drift, so the call side reads low "
                   "in rising markets.")
