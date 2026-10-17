"""
The book levered with real NSE stock futures, from its recorded positions (India tracker L7, validation
plan 5ae).

L6 levered the whole book synthetically at rf + 1%.  This builds the leverage from the contracts the F&O
store holds:
- the near month of each name that has one, rolled 3 sessions before expiry;
- whole lots, at the contract's own lot size;
- its own basis (carry), and daily mark-to-market in cash;
- NSE's adjustment on bonus and split days;
- margin per name with half of it in cash, collateral from pledged shares.

Two ways to place the exposure, at ``L`` times the book's:
- ``overlay`` (L7-A): the cash book as recorded, plus futures on its F&O names for the rest;
- ``replace`` (L7-B): F&O names only through futures; the others in cash, as far as collateral allows.

The positions are the walk-forward's recorded out-of-sample runs (``wf_stitched_r12a5``), fold by fold, each
starting from cash as the book did.  Nothing re-decides what to hold: this measures what the contracts do to
the result.  The drawdown rule (20/30/35) runs on the levered NAV, as in L6.

L7-C (plan 5af) is L7-A with a one-lot band: a held leg is resized only when its target moves 2 lots or more.

    python -m nse_engine.futures_book l7        # L7-A and L7-B at Rs 2 crore, recorded; diagnostics beside
    python -m nse_engine.futures_book l7c       # L7-C, recorded; L7-A re-run beside it for the resizes saved
    python -m nse_engine.futures_book holdout   # the family's 2026 holdout, opened once (lock in runs_leverage)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from nse_engine.config import CostConfig, DrawdownConfig
from nse_engine.costs import impact_bps, statutory_cost
from nse_engine.drawdown import DrawdownTracker

logger = logging.getLogger(__name__)

RUNS_DIR = "data/nse_engine/runs_leverage"
WALK_FORWARD = "data/nse_engine/wf_stitched_r12a5.json"
EQUITY_STORE = "data/nse_engine/store"
FO_STORE = "data/nse_engine/fo_store"
#: India's risk-free convention for the books' Sharpe.
RF = 0.065
#: L6's synthetic carry (rf + 1%) on the borrowed exposure, for the comparison.
SYNTHETIC_CARRY = 0.075
#: A raw/adjusted price ratio that moves more than this in a day is a corporate action NSE applies to the
#: contract too (bonus, split, demerger, a dividend above 2% of the price); smaller ones are dividends (A).
CA_THRESHOLD = 0.02
RULE = DrawdownConfig(enabled=True, halt_dd=0.20, half_dd=0.30, risk_off_dd=0.35, half_scale=0.5, rearm_sessions=60)
#: Plan 5ae's gate, per configuration at Rs 2 crore over 2017-25: within 2 points of L6's CAGR, 3 of its MaxDD.
GATE_CAGR, GATE_MAX_DD = 0.339, -0.421
#: Settings added after L7-A and L7-B were recorded; at these values they stay out of the hash.
HASH_NEUTRAL = {"band_lots": 0.0}
#: Plan 5af's holdout: the deployed book's recorded 2026 run (opened 15 Sep; L7a's input), opened once for the
#: family, with its own lock (the engine's ``holdout.lock`` is keyed by window and serves the core).
HOLDOUT_RUN = "20260915T161516314162Z_679cbd0c"
HOLDOUT_WINDOW = ("2026-01-01", "2026-09-11")
HOLDOUT_LOCK = Path(RUNS_DIR) / "holdout.lock"


@dataclass(frozen=True)
class FuturesBookConfig:
    """Plan 5ae's rules; ``mode`` "overlay" (L7-A) or "replace" (L7-B); plan 5af's ``band_lots`` (L7-C)."""

    name: str = "L7-A overlay"
    mode: str = "overlay"
    leverage: float = 1.75
    capital_inr: float = 2e7
    whole_lots: bool = True
    roll_sessions: int = 3
    margin_vol_multiple: float = 3.5
    margin_add: float = 0.05
    margin_min: float = 0.20
    margin_max: float = 0.50
    vol_lookback: int = 60
    pledge_value: float = 0.85          # pledged shares count at 85% (a 15% haircut)
    cash_share: float = 0.50            # SEBI: at least half the margin in cash
    buffer: float = 0.05                # of NAV, kept in cash for mark-to-market
    rebuy_slack: float = 0.10           # overlay: cash above need + this share of NAV goes back to the book
    adv_lookback: int = 20
    #: Futures impact: None keeps ``nse_engine.costs``'s equity model (3 bp floor + 25 bp at 1% participation),
    #: as pre-registered; the diagnostic sets a liquid contract's tighter spread.
    futures_spread_bps: Optional[float] = None
    futures_impact_bps: Optional[float] = None
    #: A held leg trades only when its target, in whole lots, moves more than this many lots; new legs, closes,
    #: rule changes, margin cuts and rolls always trade (plan 5af).
    band_lots: float = 0.0

    def config_hash(self) -> str:
        payload = {k: v for k, v in asdict(self).items()
                   if k != "name" and not (k in HASH_NEUTRAL and v == HASH_NEUTRAL[k])}
        payload.update(kind="futures_book", rule=asdict(RULE))
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


CONFIGS = {"L7-A": FuturesBookConfig(),
           "L7-B": FuturesBookConfig(name="L7-B replace", mode="replace"),
           "L7-C": FuturesBookConfig(name="L7-C band", band_lots=1.0)}


# ------------------------------------------------------------------ inputs

@dataclass
class Inputs:
    """Everything a run reads: the book's folds, spot prices, the contracts."""

    folds: List[Dict]
    open: pd.DataFrame
    close: pd.DataFrame
    raw_close: pd.DataFrame
    vol: pd.DataFrame
    adv: pd.DataFrame
    contracts: Dict[pd.Timestamp, Dict[str, List[Tuple[pd.Timestamp, float, float, float]]]]
    fut_adv: Dict[str, pd.Series]
    sessions: pd.DatetimeIndex
    hashes: Dict[str, str]


def load_inputs(walk_forward: str = WALK_FORWARD, runs_dir: str = "data/nse_engine/runs",
                store: str = EQUITY_STORE, fo_store: str = FO_STORE) -> Inputs:
    from nse_engine.data.panel import load_market_data

    doc = json.loads(Path(walk_forward).read_text())
    folds = []
    for f in doc["folds"]:
        run = Path(runs_dir) / f["test_run_id"]
        r = pd.read_csv(run / "returns.csv", index_col=0).squeeze("columns").astype("float64")
        r.index = pd.DatetimeIndex(r.index)
        lo, hi = pd.Timestamp(f["test_start"]), pd.Timestamp(f["test_end"])
        w = pd.read_parquet(run / "weights.parquet").fillna(0.0)
        t = pd.read_csv(run / "trades.csv", parse_dates=["date"])
        folds.append({"returns": r[(r.index >= lo) & (r.index <= hi)], "weights": w.loc[lo:hi],
                      "traded": set(zip(t["date"], t["symbol"])), "run_id": f["test_run_id"]})
    start = folds[0]["returns"].index[0] - pd.DateOffset(months=6)
    end = folds[-1]["returns"].index[-1]
    data = load_market_data(store, start.date().isoformat(), end.date().isoformat(), series=("EQ", "BE"),
                            min_median_value_inr=0.0, adjust_dividends=True, include_symbols=())
    close = data.close.astype("float64")
    raw = (data.close_unadj if data.close_unadj is not None else data.close).astype("float64")
    rets = close.pct_change(fill_method=None)
    vol = rets.rolling(60, min_periods=20).std()
    from nse_engine.costs import median_traded_value
    adv = median_traded_value(data.value.astype("float64"), 20)
    fut = pd.concat([pd.read_parquet(Path(fo_store) / "stock_futures" / f"{y}.parquet")
                     for y in range(start.year, end.year + 1)
                     if (Path(fo_store) / "stock_futures" / f"{y}.parquet").exists()])
    fut = fut[(fut["date"] >= start) & (fut["date"] <= end) & (fut["settle"] > 0)].copy()
    fut["date"], fut["expiry"] = pd.to_datetime(fut["date"]), pd.to_datetime(fut["expiry"])
    contracts: Dict[pd.Timestamp, Dict[str, List]] = {}
    for (day, sym), g in fut.sort_values("expiry").groupby(["date", "symbol"], sort=False):
        contracts.setdefault(day, {})[str(sym)] = list(zip(g["expiry"], g["settle"], g["lot_size"], g["value_inr"]))
    near = fut.sort_values("expiry").groupby(["date", "symbol"]).head(1)
    fut_adv = {str(s): g.set_index("date")["value_inr"].sort_index().rolling(20, min_periods=1).median()
               for s, g in near.groupby("symbol")}
    manifest = json.loads((Path(fo_store) / "manifest.json").read_text())
    return Inputs(folds=folds, open=data.open.astype("float64"), close=close, raw_close=raw, vol=vol, adv=adv,
                  contracts=contracts, fut_adv=fut_adv, sessions=pd.DatetimeIndex(data.dates),
                  hashes={"equity": data.data_hash, "fo": manifest.get("data_hash", ""), "walk_forward": walk_forward})


# ------------------------------------------------------------------ the simulation

def _contract(inp: Inputs, day, sym: str, expiry=None, roll_sessions: int = 3):
    """(expiry, settle, lot) of the contract held on ``day``: ``expiry``'s when given, else the nearest that is
    more than ``roll_sessions`` sessions from expiry."""
    rows = inp.contracts.get(day, {}).get(sym)
    if not rows:
        return None
    if expiry is not None:
        return next(((e, s, lot) for e, s, lot, _ in rows if e == expiry), None)
    pos = inp.sessions.searchsorted(day)
    for e, s, lot, _ in rows:
        if inp.sessions.searchsorted(e, side="right") - 1 - pos > roll_sessions:
            return e, s, lot
    return None


def _ratio(frame: pd.DataFrame, day, prev, sym: str) -> float:
    try:
        a, b = frame.at[day, sym], frame.at[prev, sym]
    except KeyError:
        return float("nan")
    return float(a / b) if (np.isfinite(a) and np.isfinite(b) and b > 0) else float("nan")


def _synthetic_open(inp: Inputs, day, prev, sym: str, expiry, base: Optional[float] = None) -> Optional[float]:
    """A contract's price at the session's open: its previous settle (``base`` when given, already adjusted)
    times the spot's raw open over its raw previous close (the F&O store has no opens)."""
    if base is None:
        c = _contract(inp, prev, sym, expiry) if prev is not None else None
        if c is None:
            c = _contract(inp, day, sym, expiry)
            return c[1] if c else None
        base = c[1]
    gap = float("nan")
    if prev is not None and sym in inp.open.columns:
        o, c_now, c_raw = inp.open.at[day, sym], inp.close.at[day, sym], inp.raw_close.at[day, sym]
        if np.isfinite(o) and np.isfinite(c_now) and c_now > 0:
            gap = (o * c_raw / c_now) / inp.raw_close.at[prev, sym]
    return base * (gap if np.isfinite(gap) and gap > 0 else 1.0)


def _margin_rate(inp: Inputs, cfg: FuturesBookConfig, day, sym: str) -> float:
    v = inp.vol.at[day, sym] if (day in inp.vol.index and sym in inp.vol.columns) else float("nan")
    v = float(v) if np.isfinite(v) else 0.03
    return float(np.clip(cfg.margin_vol_multiple * v + cfg.margin_add, cfg.margin_min, cfg.margin_max))


def _fut_cost(inp: Inputs, value: float, side: str, day, sym: str, cfg: Optional[FuturesBookConfig] = None,
              orders: int = 1) -> Tuple[float, float]:
    """(charges, impact) of a futures order of ``value`` rupees."""
    from kite_connect.options.fno_costs import futures_charges

    if value <= 0:
        return 0.0, 0.0
    costs = CostConfig()
    if cfg is not None and cfg.futures_spread_bps is not None:
        costs = replace(costs, spread_floor_bps=cfg.futures_spread_bps, impact_coefficient_bps=cfg.futures_impact_bps)
    adv_s = inp.fut_adv.get(sym)
    adv = float(adv_s.asof(day)) if adv_s is not None and len(adv_s) else float("nan")
    imp = (value * impact_bps(value, adv, costs) / 1e4 if np.isfinite(adv) and adv > 0
           else value * costs.spread_floor_bps / 1e4)
    return futures_charges(value, side, day, orders).total, imp


def _cash_cost(inp: Inputs, value: float, side: str, day, sym: Optional[str] = None) -> float:
    """Statutory charges and impact of a cash-share order (``nse_engine.costs``)."""
    if value <= 0:
        return 0.0
    adv = float(inp.adv.at[day, sym]) if (sym and day in inp.adv.index and sym in inp.adv.columns) else float("nan")
    imp = impact_bps(value, adv, CostConfig()) if np.isfinite(adv) and adv > 0 else 3.0
    return statutory_cost(value, side, day) + value * imp / 1e4


@dataclass
class Leg:
    expiry: pd.Timestamp
    qty: float          # shares
    ref: float          # last settle (adjusted scale)
    lot: float          # shares a lot


@dataclass
class Result:
    config: FuturesBookConfig
    returns: pd.Series
    equity: pd.Series
    daily: pd.DataFrame
    metrics: Dict = field(default_factory=dict)
    run_id: str = ""
    run_dir: str = ""
    data_hash: str = ""


def simulate(inp: Inputs, cfg: FuturesBookConfig) -> Result:
    """Run the book fold by fold with the futures legs of ``cfg``; NAV carries across folds, every leg is closed
    at a fold's end (the book restarts from cash)."""
    nav0 = cfg.capital_inr
    nav = nav0
    rows: List[Dict] = []
    tracker = DrawdownTracker(RULE)
    scale = 1.0
    for fold in inp.folds:
        r, w, traded = fold["returns"], fold["weights"], fold["traded"]
        legs: Dict[str, Leg] = {}
        cash_legs: Dict[str, float] = {}             # replace: rupees held in each non-F&O name
        book = nav if cfg.mode == "overlay" else 0.0  # overlay: the recorded book's sub-account
        cash = nav - book
        prev = None
        resize_all = True
        for day in r.index:
            costs = {"charges": 0.0, "impact": 0.0, "cash_costs": 0.0, "roll": 0.0}
            calls = resizes = skips = 0
            pnl = 0.0
            wt = w.loc[day] if day in w.index else pd.Series(dtype="float64")
            held = wt[wt > 1e-9]
            g = float(held.sum())
            # 1. NSE's contract adjustments on bonus / split days, then the day's futures P&L at the synthetic open
            opens: Dict[str, float] = {}
            for sym, leg in list(legs.items()):
                c = _contract(inp, day, sym, leg.expiry)
                if c is None:
                    continue
                if prev is not None:
                    k = _ratio(inp.raw_close, day, prev, sym) / _ratio(inp.close, day, prev, sym)
                    if np.isfinite(k) and abs(k - 1.0) > CA_THRESHOLD:
                        leg.qty /= k
                        leg.ref *= k
                opens[sym] = _synthetic_open(inp, day, prev, sym, leg.expiry, base=leg.ref)
            # 2. targets for names traded today, or for every name when the rule changed state
            nav_prev = nav
            fo = {s for s in held.index if _contract(inp, day, s, None, cfg.roll_sessions) is not None}
            g_fo = float(held[list(fo)].sum()) if fo else 0.0
            targets: Dict[str, float] = {}
            todo = set(legs) | set(cash_legs) | set(held.index)
            for sym in todo:
                if not (resize_all or (day, sym) in traded or sym not in held.index):
                    continue
                weight = float(held.get(sym, 0.0))
                if cfg.mode == "overlay":
                    extra = max(scale * cfg.leverage * g * nav_prev - g * book, 0.0)
                    targets[sym] = extra * weight / g_fo if (sym in fo and g_fo > 0) else 0.0
                else:
                    targets[sym] = scale * cfg.leverage * weight * nav_prev
            if cfg.mode == "replace":
                # cash legs bought today fit the collateral: cash keeps half the projected margin and the
                # buffer; cash plus 85% of the shares covers all of it (plan 5ae: cut pro rata)
                buying = {s: v for s, v in targets.items() if s not in fo and v > 0}
                if buying:
                    proj = {s: leg.qty * leg.ref for s, leg in legs.items()}
                    proj.update({s: v for s, v in targets.items() if s in fo})
                    proj_margin = sum(_margin_rate(inp, cfg, day, s) * v for s, v in proj.items())
                    have = sum(cash_legs.get(s, 0.0) for s in buying)
                    other = sum(v for s, v in cash_legs.items() if s not in buying)
                    room = cash + have
                    total = sum(buying.values())
                    phi = min(1.0, (room - cfg.cash_share * proj_margin - cfg.buffer * nav_prev) / total,
                              (room + cfg.pledge_value * other - proj_margin - cfg.buffer * nav_prev)
                              / ((1.0 - cfg.pledge_value) * total))
                    for s in buying:
                        targets[s] *= max(phi, 0.0)
            # 3. execute the futures side at the synthetic open; cash legs (replace) at the day's values
            for sym, value in targets.items():
                if cfg.mode == "replace" and (sym not in fo or _contract(inp, day, sym, None, cfg.roll_sessions) is None) \
                        and sym not in legs:
                    have = cash_legs.get(sym, 0.0)
                    trade = value - have
                    if abs(trade) > 1e-6:
                        side = "BUY" if trade > 0 else "SELL"
                        c = _cash_cost(inp, abs(trade), side, day, sym)
                        costs["cash_costs"] += c
                        cash -= trade + c
                        cash_legs[sym] = value
                        if value <= 1e-6:
                            cash_legs.pop(sym, None)
                    continue
                if cfg.mode == "replace" and sym not in fo:   # left the F&O list: close the contract
                    value = 0.0
                leg = legs.get(sym)
                c = _contract(inp, day, sym, leg.expiry if leg else None, cfg.roll_sessions)
                if c is None:
                    continue
                expiry, settle, lot = c
                px = opens.get(sym) if sym in opens else _synthetic_open(inp, day, prev, sym, expiry)
                px = px if px else settle
                lots = value / (lot * px) if px > 0 else 0.0
                qty = (round(lots) if cfg.whole_lots else lots) * lot
                old = leg.qty if leg else 0.0
                delta = qty - old
                if leg and qty > 1e-9 and not resize_all and 1e-9 < abs(delta) <= cfg.band_lots * lot + 1e-9:
                    skips += 1          # inside the band: the leg stays as it is, marked at the settle below
                    continue
                if leg:
                    pnl += old * (px - leg.ref)
                if abs(delta) > 1e-9:
                    ch, im = _fut_cost(inp, abs(delta) * px, "BUY" if delta > 0 else "SELL", day, sym, cfg)
                    costs["charges"] += ch
                    costs["impact"] += im
                    resizes += 1
                if qty > 1e-9:
                    legs[sym] = Leg(expiry, qty, px, lot)
                else:
                    legs.pop(sym, None)
            resize_all = False
            # 4. mark every leg to the settle; roll the ones near expiry
            for sym, leg in list(legs.items()):
                c = _contract(inp, day, sym, leg.expiry)
                if c is None:
                    continue
                _, settle, _ = c
                pnl += leg.qty * (settle - leg.ref)
                leg.ref = settle
                pos = inp.sessions.searchsorted(day)
                if inp.sessions.searchsorted(leg.expiry, side="right") - 1 - pos <= cfg.roll_sessions:
                    nxt = _contract(inp, day, sym, None, cfg.roll_sessions)
                    if nxt is None or nxt[0] == leg.expiry:
                        continue
                    e2, s2, lot2 = nxt
                    q2 = (round(leg.qty * settle / (s2 * lot2)) if cfg.whole_lots else leg.qty * settle / (s2 * lot2)) * lot2
                    for value, side in ((leg.qty * settle, "SELL"), (q2 * s2, "BUY")):
                        ch, im = _fut_cost(inp, value, side, day, sym, cfg)
                        costs["roll"] += ch + im
                    legs[sym] = Leg(e2, q2, s2, lot2)
            # 5. the cash book and the cash legs move with the day
            if cfg.mode == "overlay":
                book *= 1.0 + float(r.loc[day])
            else:
                for sym in list(cash_legs):
                    rr = _ratio(inp.close, day, prev, sym) if prev is not None else 1.0
                    cash_legs[sym] *= rr if np.isfinite(rr) else 1.0
            cash += pnl - costs["charges"] - costs["impact"] - costs["roll"]
            shares = book * g if cfg.mode == "overlay" else sum(cash_legs.values())
            nav = (book + cash) if cfg.mode == "overlay" else (cash + shares)
            # 6. margin: half in cash, the rest from pledged shares, plus the buffer
            notional = {s: leg.qty * leg.ref for s, leg in legs.items()}
            margin = sum(_margin_rate(inp, cfg, day, s) * v for s, v in notional.items())
            buffer = cfg.buffer * nav if margin > 0 else 0.0      # the buffer is for futures' mark-to-market
            need_cash = cfg.cash_share * margin + buffer
            need_all = margin + buffer
            if cfg.mode == "overlay":
                # the cash book's size: cash keeps half the margin and the buffer, cash plus 85% of the book's
                # shares covers all of it, and the rule caps the whole exposure (scale x L x the book's)
                by_cash = nav - need_cash
                by_collateral = (nav - need_all) / (1.0 - cfg.pledge_value * g) if g < 1.0 / cfg.pledge_value else nav
                target = max(min(by_cash, by_collateral, scale * cfg.leverage * nav, nav), 0.0)
                if book > target + 1e-6:             # sell a slice of the book
                    move = book - target
                    c = _cash_cost(inp, move * g, "SELL", day)
                    book -= move
                    cash += move - c
                    costs["cash_costs"] += c
                    calls += 1 if (target < scale * cfg.leverage * nav - 1e-6 and not targets) else 0
                elif book < target - cfg.rebuy_slack * nav:      # buy it back
                    move = target - cfg.rebuy_slack * nav / 2 - book
                    c = _cash_cost(inp, move * g, "BUY", day)
                    book += move - c
                    cash -= move
                    costs["cash_costs"] += c
            else:
                short_cash = need_cash - cash
                short_all = need_all - cash - cfg.pledge_value * shares
                if (short_cash > 0 or short_all > 0) and margin > 0:   # a margin call: cut the futures pro rata
                    cut = min(1.0, max(short_cash / (cfg.cash_share * margin), short_all / margin, 0.0))
                    for sym, leg in list(legs.items()):
                        keep = leg.qty * (1.0 - cut)
                        keep = math.floor(keep / leg.lot) * leg.lot if cfg.whole_lots else keep
                        sold = leg.qty - keep
                        ch, im = _fut_cost(inp, sold * leg.ref, "SELL", day, sym, cfg)
                        costs["charges"] += ch
                        costs["impact"] += im
                        cash -= ch + im
                        if keep > 1e-9:
                            leg.qty = keep
                        else:
                            legs.pop(sym)
                    calls += 1
            nav = (book + cash) if cfg.mode == "overlay" else (cash + sum(cash_legs.values()))
            exposure = (book * g if cfg.mode == "overlay" else sum(cash_legs.values())) + sum(
                leg.qty * leg.ref for leg in legs.values())
            biggest = max([leg.qty * leg.ref for leg in legs.values()] + [0.0])
            d = tracker.update(nav)
            if d.scale != scale:
                scale, resize_all = d.scale, True
            rows.append({"date": day, "nav": nav, "exposure": exposure / nav if nav > 0 else 0.0,
                         "in_futures": sum(leg.qty * leg.ref for leg in legs.values()) / nav if nav > 0 else 0.0,
                         "biggest_future": biggest / nav if nav > 0 else 0.0, "margin": margin / nav if nav > 0 else 0.0,
                         "cash": cash / nav if nav > 0 else 0.0, "margin_calls": calls, "state": d.state,
                         "resizes": resizes, "band_skips": skips, **{k: v / nav_prev for k, v in costs.items()}})
            prev = day
        # the fold ends: close every leg at the last settle (the book goes to cash)
        for sym, leg in legs.items():
            ch, im = _fut_cost(inp, leg.qty * leg.ref, "SELL", prev, sym, cfg)
            cash -= ch + im
        legs.clear()
        if cfg.mode == "replace":
            for sym, v in cash_legs.items():
                cash += v - _cash_cost(inp, v, "SELL", prev, sym)
            cash_legs.clear()
        nav = (book + cash) if cfg.mode == "overlay" else cash
        rows[-1]["nav"] = nav
    daily = pd.DataFrame(rows).set_index("date")
    equity = daily["nav"]
    returns = equity.pct_change().fillna(equity.iloc[0] / nav0 - 1.0)
    return Result(cfg, returns, equity, daily, metrics=_metrics(returns, daily), data_hash=inp.hashes["equity"])


def synthetic(inp: Inputs, leverage: float = 1.75) -> pd.Series:
    """L6's synthetic version on the same folds: exposure reset weekly, rf + 1% on the borrowed gross, the rule
    on the levered equity."""
    eq, out, tr, scale, week = 1.0, [], DrawdownTracker(RULE), 1.0, None
    x, gross_prev = leverage, 0.0
    for fold in inp.folds:
        r, w = fold["returns"], fold["weights"]
        for day, rt in r.items():
            wk = day.isocalendar()[:2]
            if wk != week:
                x, week = leverage * eq * scale, wk
            pnl = x * rt - max(x / eq - 1.0, 0.0) * gross_prev * eq * SYNTHETIC_CARRY / 252
            out.append((day, pnl / eq))
            x *= 1.0 + rt
            eq += pnl
            gross_prev = float(w.loc[day].clip(lower=0).sum()) if day in w.index else 0.0
            d = tr.update(eq)
            if d.scale != scale:
                scale = d.scale
                x = leverage * eq * scale
    return pd.Series(dict(out)).sort_index()


def _metrics(returns: pd.Series, daily: Optional[pd.DataFrame] = None) -> Dict:
    eq = (1.0 + returns).cumprod()
    years = (returns.index[-1] - returns.index[0]).days / 365.25
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0)
    dd = float((eq / eq.cummax() - 1.0).min())
    ex = returns - RF / 252
    out = {"cagr": cagr, "sharpe": float(ex.mean() / ex.std(ddof=1) * math.sqrt(252)), "max_drawdown": dd,
           "calmar": cagr / -dd if dd < 0 else float("nan"), "ann_vol": float(returns.std(ddof=1) * math.sqrt(252))}
    if daily is not None:
        out.update(exposure_mean=float(daily["exposure"].mean()), exposure_p5=float(daily["exposure"].quantile(0.05)),
                   exposure_p95=float(daily["exposure"].quantile(0.95)), in_futures_mean=float(daily["in_futures"].mean()),
                   biggest_future_max=float(daily["biggest_future"].max()), margin_mean=float(daily["margin"].mean()),
                   margin_calls=int(daily["margin_calls"].sum()),
                   futures_costs_a_year=float((daily["charges"] + daily["impact"]).sum() / years),
                   roll_costs_a_year=float(daily["roll"].sum() / years),
                   cash_costs_a_year=float(daily["cash_costs"].sum() / years),
                   resizes=int(daily["resizes"].sum()), band_skips=int(daily["band_skips"].sum()))
    return out


def record(res: Result, runs_dir: str = RUNS_DIR, extra: Optional[Dict] = None) -> str:
    """The run into the leverage family's registry (plan 5ae), with its daily book."""
    from kite_connect.options.backtest import _run_id
    from nse_engine.validation.trials import record_result

    cfg = res.config
    res.run_id = res.run_id or _run_id(cfg.config_hash())
    run_dir = record_result(res, tag=cfg.name, runs_dir=runs_dir,
                            window=(str(res.returns.index[0].date()), str(res.returns.index[-1].date())),
                            extra={"family": "leverage", "round": "L7", **(extra or {})})
    res.daily.to_csv(Path(run_dir) / "daily.csv")
    (Path(run_dir) / "config.json").write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True))
    return run_dir


def _trial(inp: Inputs, syn: pd.Series, cfg: FuturesBookConfig, record_runs: bool, round_: str) -> Dict:
    """One gated configuration at Rs 2 crore: its metrics, tracking error against L6, the pass rule, the record."""
    res = simulate(inp, cfg)
    te = float((res.returns - syn.reindex(res.returns.index).fillna(0.0)).std() * math.sqrt(252))
    m = {**res.metrics, "tracking_error_vs_L6": te}
    m["passed"] = bool(m["cagr"] >= GATE_CAGR and m["max_drawdown"] >= GATE_MAX_DD)
    if record_runs:
        m["run_dir"] = record(res, extra={"round": round_, "tracking_error_vs_L6": te, "passed": m["passed"]})
    return m


def l7(record_runs: bool = True) -> Dict:
    """Plan 5ae: L7-A and L7-B at Rs 2 crore (recorded), the pass rule against L6, and the diagnostics."""
    inp = load_inputs()
    syn = synthetic(inp)
    out: Dict = {"synthetic_L6": _metrics(syn), "configs": {}, "diagnostics": {}, "hashes": inp.hashes}
    for key in ("L7-A", "L7-B"):
        cfg = CONFIGS[key]
        out["configs"][key] = _trial(inp, syn, cfg, record_runs, "L7")
        for label, variant in (("Rs 1 crore", replace(cfg, capital_inr=1e7)), ("Rs 5 crore", replace(cfg, capital_inr=5e7)),
                               ("fractional lots", replace(cfg, whole_lots=False)),
                               ("futures spread 1 bp", replace(cfg, futures_spread_bps=1.0, futures_impact_bps=10.0))):
            out["diagnostics"][f"{key}, {label}"] = simulate(inp, variant).metrics
    return out


def l7c(record_runs: bool = True) -> Dict:
    """Plan 5af: L7-C at Rs 2 crore (recorded) against the same gate; L7-A re-run beside it, unrecorded, for the
    resizes the band saved; Rs 1 and 5 crore as diagnostics."""
    inp = load_inputs()
    syn = synthetic(inp)
    cfg = CONFIGS["L7-C"]
    return {"synthetic_L6": _metrics(syn), "L7-C": _trial(inp, syn, cfg, record_runs, "L7-C"),
            "L7-A, re-run": simulate(inp, CONFIGS["L7-A"]).metrics,
            "diagnostics": {f"L7-C, {label}": simulate(inp, replace(cfg, capital_inr=c)).metrics
                            for label, c in (("Rs 1 crore", 1e7), ("Rs 5 crore", 5e7))},
            "hashes": inp.hashes}


def holdout(lock_path: Path = HOLDOUT_LOCK, runs_dir: str = "data/nse_engine/runs") -> Dict:
    """Plan 5af: the family's 2026 holdout, opened once.  L7-A, L7-B and L7-C on the deployed book's recorded
    holdout run and the 2026 F&O file, beside L6's synthetic and the unlevered book.  The lock entry is written
    before anything runs, so a crash still counts as the look."""
    from datetime import datetime, timezone

    from nse_engine.validation.holdout import HoldoutLockedError, _read_lock, _write_lock
    from nse_engine.validation.trials import git_state

    window = {"start": HOLDOUT_WINDOW[0], "end": HOLDOUT_WINDOW[1]}
    lock = _read_lock(Path(lock_path))
    if any(e.get("window") == window for e in lock["evaluations"]):
        raise HoldoutLockedError(f"the leverage family's 2026 holdout was already opened (lock {lock_path})")
    entry = {"window": window, "family": "leverage", "run": HOLDOUT_RUN, "status": "started",
             "configs": {k: c.config_hash() for k, c in CONFIGS.items()}, **git_state(),
             "created_at": datetime.now(timezone.utc).isoformat(), "metrics": {}}
    lock["evaluations"].append(entry)
    _write_lock(Path(lock_path), lock)

    spec = Path(lock_path).with_name("holdout_2026_input.json")
    spec.write_text(json.dumps({"folds": [{"test_run_id": HOLDOUT_RUN, "test_start": window["start"],
                                           "test_end": window["end"]}]}))
    inp = load_inputs(walk_forward=str(spec), runs_dir=runs_dir)

    def summary(r: pd.Series, daily: Optional[pd.DataFrame] = None) -> Dict:
        return {"return": float((1.0 + r).prod() - 1.0), **_metrics(r, daily)}

    out = {"unlevered": summary(inp.folds[0]["returns"]), "synthetic_L6": summary(synthetic(inp))}
    for key, cfg in CONFIGS.items():
        res = simulate(inp, cfg)
        out[key] = summary(res.returns, res.daily)
    c = out["L7-C"]
    out["L7-C agrees"] = bool(c["return"] >= 0.23 and c["max_drawdown"] >= -0.252)   # binding only on a gate pass
    entry.update({"status": "completed", "completed_at": datetime.now(timezone.utc).isoformat(),
                  "data_hash": inp.hashes["equity"], "fo_hash": inp.hashes["fo"], "metrics": out})
    _write_lock(Path(lock_path), lock)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nse_engine.futures_book", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("l7", help="plan 5ae: L7-A and L7-B at Rs 2 crore, with diagnostics")
    p.add_argument("--no-record", action="store_true", help="do not record the runs (debugging)")
    p = sub.add_parser("l7c", help="plan 5af: L7-C (the one-lot band) at Rs 2 crore, with diagnostics")
    p.add_argument("--no-record", action="store_true", help="do not record the run (debugging)")
    sub.add_parser("holdout", help="plan 5af: the family's 2026 holdout, opened once")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.cmd == "holdout":
        out = holdout()
    else:
        out = (l7 if args.cmd == "l7" else l7c)(record_runs=not args.no_record)
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
