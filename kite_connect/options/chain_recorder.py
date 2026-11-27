"""
NIFTY option-chain recorder (India tracker OX1; plan 5cg, pre-registered 27 Nov 2026, 13:50 IST, amended 13:52).

A forward record of what NIFTY's options cost to trade, read from Kite's five-level book in market hours by a GitHub
job with the day's stored login (a read-only quote call, no orders), as ``nse_engine.etf_costs`` records the ETFs.
Each snapshot holds NIFTY 50, India VIX and every listed out-of-the-money NIFTY option (calls at or above spot, puts
at or below) within ``BAND_SD`` standard deviations of spot, one SD being spot x VIX / 100 x sqrt(days / 365), on the
next two monthly expiries (each month's latest) and the overlay scan's target expiry.  The nearer monthly's calls go
to ``A1_CALL_BAND_SD``: A1's wing reached 6.5 such SDs on 2013-26 decision days.  Kept in Neon (schema
``option_chain``), one key per snapshot, so a snapshot never reads back the record.

``a1-spread`` is plan 5cg's measurement.  On each snapshot whose nearer monthly expiry is 8-20 days away it takes the
legs A1 would sell and buy (its own rule: the first quoted call strikes above spot x (1 + mu n + 2 or 3 sigma
sqrt n)) and each leg's half-spread less the slippage A1's backtest assumes (``fno_costs.slippage_per_unit``).  The
median of that excess, set against ``A1_BREAKEVEN_PER_LEG``, decides whether the measured spread blocks A1's path to
live; the decision is taken on 18 Jan 2027 with ``--until 2027-01-18``.

    python -m kite_connect.options.chain_recorder snapshot      # market hours; needs today's Kite login (daily_login)
    python -m kite_connect.options.chain_recorder status
    python -m kite_connect.options.chain_recorder a1-spread [--until YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import date, datetime, time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from kite_connect.options.broker import Broker
from kite_connect.options.fno_costs import slippage_per_unit
from kite_connect.options.instruments import Contract, InstrumentResolver, spot_key
from kite_connect.options.live_chain import IST, days_to_expiry
from kite_connect.options.options_config import OptionsConfig
from kite_connect.options.overlay_scan import pick_leg, target_expiry
from kite_connect.options.sleeves import CANDIDATES, SleeveConfig
from kite_connect.options.theory import CALL, PUT, black_scholes, implied_volatility

logger = logging.getLogger(__name__)

UNDERLYING = "NIFTY"
VIX_KEY = "NSE:INDIA VIX"
SCHEMA = "option_chain"
KEY_PREFIX = "option_chain_"
MARKET = (time(9, 20), time(15, 28))
BAND_SD = 4.0
A1_CALL_BAND_SD = 7.0
#: Calendar days from the snapshot to the nearer monthly expiry that make it A1-like (A1 enters at 15 or fewer).
A1_WINDOW_DAYS = (8, 20)
#: Points of extra cost per leg that leave A1 no profit: Rs 77,919 of P&L / (2 legs x 104,600 units) on its recorded
#: 2007-25 run (20261010T122956596793Z_ae503fc4, 138 cycles, settled at expiry but once).  Plan 5cg's threshold.
A1_BREAKEVEN_PER_LEG = 0.37
#: Sessions with an A1-like snapshot the rule needs; fewer by 18 Jan 2027 moves the decision to 16 Feb 2027.
MIN_SESSIONS = 10
DECISION_DATES = ("2027-01-18", "2027-02-16")
#: Plan 5cg's stop rule: the record stops once Neon holds this much of it.
CAP_BYTES = 50_000_000
#: NIFTY 50's closes may end at most this many days before a snapshot (a stale F&O store would move sigma).
CLOSES_MAX_LAG_DAYS = 7
FO_STORE = "data/nse_engine/fo_store"


def _book():
    from database.connection import get_db_manager
    from database.paper_cloud import PaperCloudSync

    book = PaperCloudSync(get_db_manager(), schema=SCHEMA)
    book.ensure_tables()
    return book


# ------------------------------------------------------------------ the snapshot

def monthly_expiries(expiries: Iterable[date]) -> List[date]:
    """Each month's latest listed expiry, ascending (the backtests' monthly expiry)."""
    latest: Dict[Tuple[int, int], date] = {}
    for e in expiries:
        latest[(e.year, e.month)] = max(e, latest.get((e.year, e.month), e))
    return sorted(latest.values())


def expiries_to_record(expiries: Sequence[date], now: datetime,
                       cfg: OptionsConfig) -> Tuple[List[date], Optional[date]]:
    """(the next two monthly expiries, the overlay scan's target expiry or None)."""
    live = [e for e in expiries if e >= now.astimezone(IST).date()]
    return monthly_expiries(live)[:2], target_expiry({e: days_to_expiry(e, now) for e in live}, cfg)


def band(spot: float, vix: float, days: float, sds: float) -> float:
    """``sds`` standard deviations of the index over ``days`` calendar days at India VIX, in points."""
    return sds * spot * vix / 100.0 * math.sqrt(max(days, 0.0) / 365.0)


def out_of_the_money(contracts: Iterable[Contract], spot: float, call_band: float,
                     put_band: float) -> List[Contract]:
    """Calls from spot up to ``call_band`` above it, puts from spot down to ``put_band`` below it."""
    return [c for c in contracts
            if (c.option_type == CALL and spot <= c.strike <= spot + call_band)
            or (c.option_type == PUT and spot - put_band <= c.strike <= spot)]


def compact(quote: Mapping) -> Dict:
    """Five bid and ask levels as [price, quantity] (empty levels dropped), last price, OI and volume."""
    depth = quote.get("depth") or {}

    def side(levels: Iterable[Mapping]) -> List[List[float]]:
        return [[float(x["price"]), float(x.get("quantity") or 0)] for x in levels if float(x.get("price") or 0) > 0]

    return {"b": side(depth.get("buy", [])), "a": side(depth.get("sell", [])),
            "l": float(quote.get("last_price") or 0.0), "oi": float(quote.get("oi") or 0.0),
            "v": float(quote.get("volume") or 0.0)}


def _strike_key(k: float) -> str:
    return str(int(k)) if float(k).is_integer() else repr(float(k))


def snapshot(now: Optional[datetime] = None, kite=None, book=None, cfg: Optional[OptionsConfig] = None,
             broker: Optional[Broker] = None) -> Dict:
    """One read of the recorded chain, stored under its own key in Neon.  Skips outside market hours (GitHub delivers
    crons late), on holidays and when nobody has logged in to Kite today; raises when Kite answers without prices."""
    from nse_engine.nse_calendar import HOLIDAYS

    now = (now or datetime.now(IST)).astimezone(IST)
    if now.weekday() >= 5 or now.date().isoformat() in HOLIDAYS or not (MARKET[0] <= now.time() <= MARKET[1]):
        return {"skipped": f"outside market hours ({now:%Y-%m-%d %H:%M} IST)"}
    if broker is None:
        if kite is None:
            from kite_connect.auth.daily_login import kite_from_stored_token

            kite = kite_from_stored_token(book=None)
        if kite is None:
            return {"skipped": "no Kite login today"}
        broker = Broker(kite)
    book = book or _book()
    resolver = InstrumentResolver(broker.instruments())
    head = broker.quotes([spot_key(UNDERLYING), VIX_KEY])
    spot = float((head.get(spot_key(UNDERLYING)) or {}).get("last_price") or math.nan)
    vix = float((head.get(VIX_KEY) or {}).get("last_price") or math.nan)
    if not (spot > 0 and vix > 0):
        raise RuntimeError(f"no NIFTY 50 or India VIX price from Kite (spot {spot}, VIX {vix})")
    monthly, target = expiries_to_record(resolver.expiries(UNDERLYING), now, cfg or OptionsConfig())
    if not monthly:
        raise RuntimeError("no monthly NIFTY expiry in Kite's instruments dump")
    chosen: Dict[date, List[Contract]] = {}
    for e in monthly + ([target] if target is not None and target not in monthly else []):
        days = days_to_expiry(e, now)
        calls_sd = A1_CALL_BAND_SD if e == monthly[0] else BAND_SD
        chosen[e] = out_of_the_money(resolver.chain(UNDERLYING, e), spot, band(spot, vix, days, calls_sd),
                                     band(spot, vix, days, BAND_SD))
    quotes = broker.quotes([c.quote_key for cs in chosen.values() for c in cs])
    snap: Dict = {"at": now.isoformat(timespec="seconds"), "spot": spot, "vix": vix,
                  "monthly": [e.isoformat() for e in monthly], "target": target.isoformat() if target else None,
                  "e": {}}
    count = 0
    for e, cs in chosen.items():
        rows: Dict = {"lot": cs[0].lot_size if cs else None, CALL: {}, PUT: {}}
        for c in cs:
            q = quotes.get(c.quote_key)
            if q:
                rows[c.option_type][_strike_key(c.strike)] = compact(q)
                count += 1
        snap["e"][e.isoformat()] = rows
    if count == 0:
        raise RuntimeError(f"Kite returned no quotes for {sum(len(cs) for cs in chosen.values())} NIFTY options")
    payload = json.dumps(snap, separators=(",", ":"))
    key = f"{KEY_PREFIX}{now:%Y-%m-%dT%H%M}"
    if not book.sync_state({key: payload}):
        raise RuntimeError("could not store the snapshot in Neon")
    return {"stored": key, "contracts": count, "bytes": len(payload), "expiries": list(snap["e"])}


# ------------------------------------------------------------------ the record

def snapshots(state: Mapping[str, str], until: Optional[str] = None) -> List[Dict]:
    """The stored snapshots in time order, those taken on or before ``until`` (a date) when given."""
    out = [json.loads(v) for k, v in sorted(state.items()) if k.startswith(KEY_PREFIX)]
    return [s for s in out if until is None or s["at"][:10] <= until]


def status(book=None, state: Optional[Mapping[str, str]] = None) -> Dict:
    """How much is recorded, and whether plan 5cg's 50 MB stop has been reached."""
    state = state if state is not None else (book or _book()).read_state()
    keys = sorted(k for k in state if k.startswith(KEY_PREFIX))
    size = sum(len(state[k]) for k in keys)
    out = {"snapshots": len(keys), "sessions": len({k[len(KEY_PREFIX):len(KEY_PREFIX) + 10] for k in keys}),
           "first": keys[0][len(KEY_PREFIX):] if keys else None, "last": keys[-1][len(KEY_PREFIX):] if keys else None,
           "megabytes": round(size / 1e6, 2), "cap_reached": size >= CAP_BYTES}
    if out["cap_reached"]:
        out["action"] = "plan 5cg's cap: stop the recorder (disable the option-chain workflow)"
    return out


# ------------------------------------------------------------------ the measurement

def best(q: Mapping) -> Tuple[float, float]:
    """Best bid and ask (NaN for an empty side)."""
    return (q["b"][0][0] if q.get("b") else math.nan), (q["a"][0][0] if q.get("a") else math.nan)


def two_sided(q: Mapping) -> bool:
    bid, ask = best(q)
    return bool(bid > 0 and ask >= bid)


def leg_cost(q: Mapping, lot: Optional[int], side: str) -> Dict:
    """Half-spread, the backtest's slippage at the mid, their difference (points), and the lots shown on the side
    the leg trades against (``side`` "a": a buy lifts the ask, "b": a sale hits the bid), at the best price and over
    the five levels."""
    bid, ask = best(q)
    mid = (bid + ask) / 2.0
    half = (ask - bid) / 2.0
    model = slippage_per_unit(mid, UNDERLYING)
    levels = q.get(side) or []
    per_lot = float(lot) if lot else math.nan
    return {"bid": bid, "ask": ask, "mid": mid, "half_spread": half, "model": model, "excess": half - model,
            "lots_best": levels[0][1] / per_lot if levels else 0.0,
            "lots_5": sum(x[1] for x in levels[:5]) / per_lot if levels else 0.0}


def sigma_mu(closes: pd.Series, day: date, lookback: int) -> Optional[Tuple[float, float]]:
    """A1's sigma and mu: the standard deviation and mean of daily log returns over the ``lookback`` sessions before
    ``day``; None with a shorter history.  Raises when the closes end over a week before ``day``."""
    c = closes[closes.index < pd.Timestamp(day)].dropna()
    if c.empty or (day - c.index[-1].date()).days > CLOSES_MAX_LAG_DAYS:
        last = c.index[-1].date() if len(c) else None
        raise ValueError(f"NIFTY 50 closes end on {last}, more than {CLOSES_MAX_LAG_DAYS} days before the snapshot of "
                         f"{day}: refresh the F&O store")
    r = np.log(c.astype("float64")).diff().dropna().tail(lookback)
    if len(r) < lookback:
        return None
    return float(r.std(ddof=1)), float(r.mean())


def a1_rows(snaps: Sequence[Mapping], closes: pd.Series, a1: SleeveConfig = CANDIDATES["A1"]) -> pd.DataFrame:
    """One row per A1 leg per A1-like snapshot, or one ``missing`` row when A1 would find no spread in the record."""
    rows: List[Dict] = []
    for s in snaps:
        at = datetime.fromisoformat(s["at"])
        day = at.date()
        nearer = date.fromisoformat(s["monthly"][0])
        n = (nearer - day).days
        if not A1_WINDOW_DAYS[0] <= n <= A1_WINDOW_DAYS[1]:
            continue
        base = {"at": s["at"], "day": day.isoformat(), "time": at.strftime("%H:%M"), "days": n}
        sm = sigma_mu(closes, day, a1.vol_lookback)
        if sm is None:
            rows.append({**base, "leg": "missing", "why": "fewer closes than A1's lookback"})
            continue
        sigma, mu = sm
        chain = s["e"][nearer.isoformat()]
        calls = {float(k): q for k, q in chain[CALL].items()}
        quoted = sorted(k for k, q in calls.items() if two_sided(q))
        levels = {leg: s["spot"] * (1 + mu * n + sd * sigma * math.sqrt(n))
                  for leg, sd in (("short", a1.short_sd), ("wing", a1.wing_sd))}
        short = next((k for k in quoted if k > levels["short"]), None)
        wing = next((k for k in quoted if k > levels["wing"]), None)
        if short is None or wing is None or wing <= short:
            rows.append({**base, "leg": "missing", "why": "no quoted call strike above A1's level in the record"})
            continue
        for leg, k in (("short", short), ("wing", wing)):
            unquoted = len([x for x, q in calls.items() if levels[leg] < x < k and not two_sided(q)])
            rows.append({**base, "leg": leg, "strike": k, "level": levels[leg], "unquoted_passed": unquoted,
                         **leg_cost(calls[k], chain.get("lot"), "b" if leg == "short" else "a")})
    return pd.DataFrame(rows)


def condor_rows(snaps: Sequence[Mapping], cfg: Optional[OptionsConfig] = None) -> pd.DataFrame:
    """The overlay scan's four condor legs on each snapshot's target expiry (``pick_leg`` on the recorded quotes, IV
    and delta from the mid as the scan computes them) with their half-spreads."""
    cfg = cfg or OptionsConfig()
    rate, o = cfg.market.risk_free_rate, cfg.overlay
    rows: List[Dict] = []
    for s in snaps:
        if not s.get("target"):
            continue
        at, expiry, spot = datetime.fromisoformat(s["at"]), date.fromisoformat(s["target"]), float(s["spot"])
        dte = days_to_expiry(expiry, at)
        chain_rows = []
        side = s["e"][s["target"]]
        for t in (CALL, PUT):
            for k, q in side[t].items():
                bid, ask = best(q)
                mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else q.get("l", math.nan)
                iv = implied_volatility(t, mid, spot, float(k), dte, rate) if mid and mid > 0 else None
                delta = black_scholes(t, spot, float(k), dte, rate, iv).delta if iv else math.nan
                chain_rows.append({"strike": float(k), "option_type": t, "bid": bid if bid > 0 else 0.0,
                                   "ask": ask if ask > 0 else 0.0, "oi": q.get("oi", 0.0), "delta": delta})
        chain = pd.DataFrame(chain_rows)
        if chain.empty:
            continue
        for t in (CALL, PUT):
            for leg, d in (("short", o.condor_short_delta), ("wing", o.condor_wing_delta)):
                pick = pick_leg(chain, t, spot, d, cfg)
                if pick is None:
                    rows.append({"at": s["at"], "leg": f"{t} {leg}", "strike": math.nan})
                    continue
                q = side[t][_strike_key(float(pick["strike"]))]
                rows.append({"at": s["at"], "leg": f"{t} {leg}", "strike": float(pick["strike"]),
                             **leg_cost(q, side.get("lot"), "b" if leg == "short" else "a")})
    return pd.DataFrame(rows)


def _dist(x: pd.Series) -> Dict:
    x = x.dropna()
    return {"n": len(x), "median": round(float(x.median()), 3) if len(x) else None,
            "p75": round(float(x.quantile(0.75)), 3) if len(x) else None,
            "p90": round(float(x.quantile(0.90)), 3) if len(x) else None}


def a1_spread(book=None, state: Optional[Mapping[str, str]] = None, closes: Optional[pd.Series] = None,
              store_dir: str = FO_STORE, until: Optional[str] = None, cfg: Optional[OptionsConfig] = None) -> Dict:
    """Plan 5cg's rule on the record (to ``until``), with what it reports beside the rule."""
    from nse_engine.data import fo_store

    state = state if state is not None else (book or _book()).read_state()
    snaps = snapshots(state, until)
    closes = closes if closes is not None else fo_store.load_underlying(store_dir, UNDERLYING)
    rows = a1_rows(snaps, closes)
    legs = rows[rows["leg"].isin(["short", "wing"])] if len(rows) else rows
    sessions = legs["day"].nunique() if len(legs) else 0
    out: Dict = {"snapshots": len(snaps), "a1_like_snapshots": int(rows["at"].nunique()) if len(rows) else 0,
                 "sessions": sessions, "until": until, "decision_dates": list(DECISION_DATES),
                 "threshold_points_per_leg": A1_BREAKEVEN_PER_LEG, "record": status(state=state)}
    if sessions < MIN_SESSIONS:
        out["rule"] = (f"not yet decidable: {sessions} session(s) with an A1-like snapshot, the rule needs "
                       f"{MIN_SESSIONS} (plan 5cg)")
    else:
        median = round(float(legs["excess"].median()), 6)          # prices in ticks: float noise never decides
        out["median_excess_points"] = round(median, 3)
        out["rule"] = ("above 0.37: A1 does not go live as it is (book a1 stays paper; its cost model and the A1C "
                       "case reopen, your call)" if median > A1_BREAKEVEN_PER_LEG else
                       "0.37 or below: the measured spread does not block A1's path to live")
    if len(legs):
        out["excess_by_leg"] = {leg: _dist(g["excess"]) for leg, g in legs.groupby("leg")}
        out["half_spread_by_leg"] = {leg: _dist(g["half_spread"]) for leg, g in legs.groupby("leg")}
        out["excess_by_time"] = {t: _dist(g["excess"]) for t, g in legs.groupby(legs["time"].str[:2] + ":00")}
        out["lots_on_the_traded_side"] = {leg: {"best_median": round(float(g["lots_best"].median()), 1),
                                                "five_levels_median": round(float(g["lots_5"].median()), 1)}
                                          for leg, g in legs.groupby("leg")}
        out["unquoted_strikes_passed"] = int(legs["unquoted_passed"].sum())
    missing = rows[rows["leg"] == "missing"] if len(rows) else rows
    out["missing"] = missing["why"].value_counts().to_dict() if len(missing) else {}
    condor = condor_rows(snaps, cfg)
    if len(condor) and "half_spread" in condor:
        out["condor_half_spread_by_leg"] = {leg: _dist(g["half_spread"]) for leg, g in condor.groupby("leg")}
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m kite_connect.options.chain_recorder",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["snapshot", "status", "a1-spread"])
    ap.add_argument("--until", help="a1-spread: only snapshots taken on or before this date (YYYY-MM-DD)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.cmd == "snapshot":
        out = snapshot()
    elif args.cmd == "status":
        out = status()
    else:
        out = a1_spread(until=args.until)
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
