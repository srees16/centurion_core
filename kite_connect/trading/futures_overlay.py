"""
The futures leg of a paper book levered with NSE stock futures (India tracker L9, plan 5ae).

A book with ``futures_overlay`` in its deployment file is two ledgers:
- its shares: an ordinary engine paper book (``PaperTrader``), at ``paper_capital_inr`` less the reserve;
- its futures: this ledger.  The reserve is its cash, for margin and the daily mark-to-market.

Every session, after the share book has processed the day and planned the next (L7-A, the overlay):
1. pending futures orders fill at today's open, from NSE's F&O bhavcopy;
2. NSE's contract adjustment on a bonus or split day (the spot's raw / adjusted ratio moved);
3. every contract marked to its settle, the mark-to-market in cash;
4. contracts within ``roll_sessions`` of expiry rolled into the next month at the settle;
5. the drawdown rule on the combined NAV (shares + futures cash), as on the other books;
6. tomorrow's targets: futures on the share book's F&O names for ``leverage`` x its exposure, in proportion to
   its target weights, whole lots; a name is resized only when the book trades it, on a rule change or when
   it gains or loses a contract;
7. margin per name (clip(3.5 x 60-day volatility + 0.05, 20%, 50%) of notional; no SPAN history), half in
   cash plus a 5% buffer, the rest from the share book pledged at 85%; targets that do not fit are cut pro
   rata, and a shortfall on the day (mark-to-market) cuts the contracts at the settle (a margin call).

Costs are ``fno_costs.futures_charges`` plus impact from ``nse_engine.costs`` on the contract's traded value.
The ledger is JSON in the book's ``paper_state`` (key ``futures_ledger``), with one history row per session.
Paper only: nothing here can place an order.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STATE_KEY = "futures_ledger"
#: A raw / adjusted spot ratio that moves more than this in a day is an adjustment NSE applies to the
#: contract too (plan 5ae (A)).
CA_THRESHOLD = 0.02


@dataclass(frozen=True)
class OverlayConfig:
    """The ``futures_overlay`` block of a deployment file (plan 5ae's L7-A)."""

    leverage: float = 1.75
    reserve_share: float = 0.20        # of paper_capital_inr kept as the futures ledger's cash
    roll_sessions: int = 3
    margin_vol_multiple: float = 3.5
    margin_add: float = 0.05
    margin_min: float = 0.20
    margin_max: float = 0.50
    pledge_value: float = 0.85
    cash_share: float = 0.50
    buffer: float = 0.05

    @classmethod
    def from_dict(cls, raw: Mapping) -> "OverlayConfig":
        known = {f for f in cls.__dataclass_fields__}
        bad = sorted(set(raw) - known)
        if bad:
            raise ValueError(f"unknown futures_overlay key(s): {bad}; allowed: {sorted(known)}")
        cfg = cls(**dict(raw))
        if not (1.0 <= cfg.leverage <= 3.0) or not (0.0 < cfg.reserve_share < 1.0):
            raise ValueError("futures_overlay: leverage must be 1-3 and reserve_share between 0 and 1")
        return cfg


@dataclass
class Leg:
    expiry: str
    qty: float           # shares (lots x lot)
    lot: float
    ref: float           # last settle, in the contract's current scale


@dataclass
class Ledger:
    cash: float
    legs: Dict[str, Leg] = field(default_factory=dict)
    pending: Dict[str, Dict] = field(default_factory=dict)       # symbol -> {expiry, qty, lot}
    weights: Dict[str, float] = field(default_factory=dict)      # the share book's target weights last session
    last_session: str = ""
    history: List[Dict] = field(default_factory=list)
    fills: List[Dict] = field(default_factory=list)              # the last 500 futures fills

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, default=float)

    @classmethod
    def from_json(cls, text: str) -> "Ledger":
        d = json.loads(text)
        d["legs"] = {s: Leg(**v) for s, v in d.get("legs", {}).items()}
        return cls(**d)


# ------------------------------------------------------------------ data

def session_contracts(archive_root: str, day: pd.Timestamp) -> pd.DataFrame:
    """Every stock future traded on ``day``: symbol, expiry, open, settle, lot_size, value_inr, from the
    F&O bhavcopy (UDiFF, NSE's format since Jul 2024) the nightly job syncs."""
    import io

    from nse_engine.data.store import read_raw_text

    path = Path(archive_root) / "fo" / str(day.year) / f"udiff_fo{day:%Y%m%d}.csv.zip"
    if not path.exists():
        raise FileNotFoundError(f"no F&O bhavcopy for {day.date()} at {path} (the nightly sync's --kinds fo)")
    raw = pd.read_csv(io.StringIO(read_raw_text(path)), dtype=str)
    raw.columns = [c.strip() for c in raw.columns]
    raw = raw[raw["FinInstrmTp"].str.strip() == "STF"]
    expiry = "FininstrmActlXpryDt" if "FininstrmActlXpryDt" in raw else "XpryDt"
    num = lambda c: pd.to_numeric(raw[c], errors="coerce")
    out = pd.DataFrame({"symbol": raw["TckrSymb"].str.strip(), "expiry": pd.to_datetime(raw[expiry].str.strip()),
                        "open": num("OpnPric"), "settle": num("SttlmPric"), "lot_size": num("NewBrdLotQty"),
                        "value_inr": num("TtlTrfVal")})
    return out[(out["settle"] > 0) & (out["lot_size"] > 0)].reset_index(drop=True)


def _near(contracts: pd.DataFrame, sym: str, day: pd.Timestamp, sessions: pd.DatetimeIndex,
          roll_sessions: int) -> Optional[pd.Series]:
    """The contract a new position takes: the nearest more than ``roll_sessions`` sessions from expiry."""
    rows = contracts[contracts["symbol"] == sym].sort_values("expiry")
    pos = sessions.searchsorted(day)
    for _, r in rows.iterrows():
        if _sessions_to(r["expiry"], sessions) - pos > roll_sessions:
            return r
    return None


def _sessions_to(expiry, sessions: pd.DatetimeIndex) -> int:
    """Index of the last trading session on or before ``expiry``; past the calendar's end, counted on NSE's
    holiday calendar (``nse_engine.nse_calendar``)."""
    from nse_engine.nse_calendar import HOLIDAYS

    e = pd.Timestamp(expiry)
    if e <= sessions[-1]:
        return int(sessions.searchsorted(e, side="right") - 1)
    ahead = np.busday_count((sessions[-1] + pd.Timedelta(days=1)).date(), (e + pd.Timedelta(days=1)).date(),
                            holidays=sorted(HOLIDAYS))
    return int(len(sessions) - 1 + ahead)


def _margin_rate(cfg: OverlayConfig, vol: float) -> float:
    v = vol if np.isfinite(vol) else 0.03
    return float(np.clip(cfg.margin_vol_multiple * v + cfg.margin_add, cfg.margin_min, cfg.margin_max))


def _costs(value: float, side: str, day, adv: float) -> Tuple[float, float]:
    from kite_connect.options.fno_costs import futures_charges
    from nse_engine.config import CostConfig
    from nse_engine.costs import impact_bps

    if value <= 0:
        return 0.0, 0.0
    bps = impact_bps(value, adv, CostConfig()) if np.isfinite(adv) and adv > 0 else CostConfig().spread_floor_bps
    return futures_charges(value, side, day).total, value * bps / 1e4


# ------------------------------------------------------------------ the session

def load_ledger(cloud, start_cash: float) -> Ledger:
    state = cloud.read_state() if cloud is not None else {}
    text = state.get(STATE_KEY) if state else None
    return Ledger.from_json(text) if text else Ledger(cash=float(start_cash))


def save_ledger(cloud, ledger: Ledger) -> None:
    if cloud is None:
        raise RuntimeError("the futures ledger needs the book's cloud state (Neon)")
    if not cloud.sync_state({STATE_KEY: ledger.to_json()}):
        raise RuntimeError("could not save the futures ledger")


def run_session(ledger: Ledger, cfg: OverlayConfig, day: pd.Timestamp, contracts: pd.DataFrame,
                share_equity: float, share_invested: float, target_weights: Mapping[str, float],
                traded: set, spot: Mapping[str, Dict[str, float]], sessions: pd.DatetimeIndex,
                rule=None) -> Dict:
    """One session of the futures ledger; returns its summary (the email's).

    ``spot`` maps symbol -> {open, raw_open, raw_close, raw_prev, adj_close, adj_prev, vol} for the names held
    or targeted.  ``target_weights`` are the share book's for the next session (of its equity);
    ``traded`` the names it trades then.  Idempotent: a session already done returns its summary.
    """
    day = pd.Timestamp(day)
    key = day.date().isoformat()
    start_nav = ledger.cash + share_equity            # before today's futures (first session: the start)
    if ledger.last_session == key:
        return {"session": key, "repeat": True, **(ledger.history[-1] if ledger.history else {})}
    by = {s: g.sort_values("expiry") for s, g in contracts.groupby("symbol")}

    def contract(sym, expiry) -> Optional[pd.Series]:
        g = by.get(sym)
        if g is None:
            return None
        m = g[g["expiry"] == pd.Timestamp(expiry)]
        return m.iloc[0] if len(m) else None

    costs = {"charges": 0.0, "impact": 0.0, "roll": 0.0}
    pnl, fills, rolls, calls = 0.0, [], [], 0
    # 1-2. NSE's adjustments (contracts and the orders sized on them), then yesterday's orders at today's open
    for sym in set(ledger.legs) | set(ledger.pending):
        s = spot.get(sym, {})
        if not all(np.isfinite(s.get(k, np.nan)) for k in ("raw_close", "raw_prev", "adj_close", "adj_prev")):
            continue
        k = (s["raw_close"] / s["raw_prev"]) / (s["adj_close"] / s["adj_prev"])
        if abs(k - 1.0) <= CA_THRESHOLD:
            continue
        leg = ledger.legs.get(sym)
        if leg is not None:
            leg.qty /= k
            leg.ref *= k
            leg.lot /= k
        if sym in ledger.pending:
            ledger.pending[sym]["qty"] /= k
            ledger.pending[sym]["lot"] /= k
    for sym, order in list(ledger.pending.items()):
        c = contract(sym, order["expiry"])
        if c is None:
            continue
        px = float(c["open"]) if np.isfinite(c["open"]) and c["open"] > 0 else float(c["settle"])
        leg = ledger.legs.get(sym)
        old = leg.qty if leg else 0.0
        if leg:
            pnl += old * (px - leg.ref)
        delta = float(order["qty"]) - old
        if abs(delta) > 1e-9:
            ch, im = _costs(abs(delta) * px, "BUY" if delta > 0 else "SELL", day, float(c["value_inr"]))
            costs["charges"] += ch
            costs["impact"] += im
            fills.append({"date": key, "symbol": sym, "expiry": str(pd.Timestamp(order["expiry"]).date()),
                          "side": "BUY" if delta > 0 else "SELL", "lots": round(abs(delta) / order["lot"], 3),
                          "price": px, "value": abs(delta) * px, "costs": ch + im})
        if order["qty"] > 1e-9:
            ledger.legs[sym] = Leg(str(pd.Timestamp(order["expiry"]).date()), float(order["qty"]), float(order["lot"]), px)
        else:
            ledger.legs.pop(sym, None)
    ledger.pending = {}
    # 3-4. mark to the settle; roll the ones near expiry
    pos = sessions.searchsorted(day)
    for sym, leg in list(ledger.legs.items()):
        c = contract(sym, leg.expiry)
        if c is None:
            continue
        settle = float(c["settle"])
        pnl += leg.qty * (settle - leg.ref)
        leg.ref = settle
        if _sessions_to(leg.expiry, sessions) - pos <= cfg.roll_sessions:
            nxt = _near(contracts, sym, day, sessions, cfg.roll_sessions)
            if nxt is None or str(pd.Timestamp(nxt["expiry"]).date()) == leg.expiry:
                continue
            lot2 = float(nxt["lot_size"])
            q2 = round(leg.qty * settle / (float(nxt["settle"]) * lot2)) * lot2
            for value, side, v in ((leg.qty * settle, "SELL", float(c["value_inr"])),
                                   (q2 * float(nxt["settle"]), "BUY", float(nxt["value_inr"]))):
                ch, im = _costs(value, side, day, v)
                costs["roll"] += ch + im
            rolls.append({"symbol": sym, "from": leg.expiry, "to": str(pd.Timestamp(nxt["expiry"]).date()),
                          "lots": round(q2 / lot2, 3)})
            ledger.legs[sym] = Leg(str(pd.Timestamp(nxt["expiry"]).date()), q2, lot2, float(nxt["settle"]))
    ledger.cash += pnl - sum(costs.values())
    nav = share_equity + ledger.cash
    # 5. the drawdown rule on the combined NAV
    scale, state = 1.0, "normal"
    if rule is not None:
        from nse_engine.drawdown import replay
        navs = pd.Series([h["nav"] for h in ledger.history] + [nav],
                         index=pd.DatetimeIndex([h["date"] for h in ledger.history] + [key]))
        last = replay(navs, rule).iloc[-1]
        scale, state = float(last["scale"]), str(last["state"])
    previous_scale = float(ledger.history[-1].get("scale", 1.0)) if ledger.history else None
    # 6. tomorrow's targets
    weights = {s: float(w) for s, w in target_weights.items() if w > 1e-9}
    g = sum(weights.values())
    fo = {s: _near(contracts, s, day, sessions, cfg.roll_sessions) for s in weights}
    fo = {s: c for s, c in fo.items() if c is not None}
    g_fo = sum(weights[s] for s in fo)
    extra = max(scale * cfg.leverage * g * nav - g * share_equity, 0.0)
    resize_all = previous_scale is None or previous_scale != scale
    targets: Dict[str, Dict] = {}
    for sym in set(ledger.legs) | set(fo):
        held = ledger.legs.get(sym)
        if not (resize_all or sym in traded or sym not in weights or held is None
                or (sym in weights) != (sym in ledger.weights)):
            continue
        c = contract(sym, held.expiry) if held is not None else fo.get(sym)
        if c is None:
            continue
        value = extra * weights[sym] / g_fo if (sym in fo and g_fo > 0) else 0.0
        lot = float(c["lot_size"])
        qty = round(value / (lot * float(c["settle"]))) * lot if c["settle"] > 0 else 0.0
        targets[sym] = {"expiry": str(pd.Timestamp(c["expiry"]).date()), "qty": float(qty), "lot": lot,
                        "settle": float(c["settle"])}
    # 7. margin: fit tomorrow's book, then today's (a margin call if not)
    vol = {s: float(spot.get(s, {}).get("vol", np.nan)) for s in set(ledger.legs) | set(targets)}
    proj = {s: leg.qty * leg.ref for s, leg in ledger.legs.items()}
    proj.update({s: t["qty"] * t["settle"] for s, t in targets.items()})
    margin = sum(_margin_rate(cfg, vol[s]) * v for s, v in proj.items())
    buffer = cfg.buffer * nav if margin > 0 else 0.0
    room = min((ledger.cash - buffer) / (cfg.cash_share * margin) if margin > 0 else np.inf,
               (ledger.cash + cfg.pledge_value * share_invested - buffer) / margin if margin > 0 else np.inf)
    if room < 1.0:
        cut = max(room, 0.0)
        for sym, leg in ledger.legs.items():
            targets.setdefault(sym, {"expiry": leg.expiry, "qty": leg.qty, "lot": leg.lot, "settle": leg.ref})
        for t in targets.values():
            t["qty"] = math.floor(t["qty"] * cut / t["lot"]) * t["lot"]
        now_margin = sum(_margin_rate(cfg, vol[s]) * leg.qty * leg.ref for s, leg in ledger.legs.items())
        if now_margin > 0 and ledger.cash - buffer < cfg.cash_share * now_margin:
            calls += 1                                   # today's book no longer fits: a margin call
            for sym, leg in list(ledger.legs.items()):
                keep = math.floor(leg.qty * cut / leg.lot) * leg.lot
                sold = leg.qty - keep
                ch, im = _costs(sold * leg.ref, "SELL", day, float("nan"))
                ledger.cash -= ch + im
                costs["charges"] += ch
                costs["impact"] += im
                fills.append({"date": key, "symbol": sym, "expiry": leg.expiry, "side": "SELL (margin call)",
                              "lots": round(sold / leg.lot, 3), "price": leg.ref, "value": sold * leg.ref,
                              "costs": ch + im})
                if keep > 1e-9:
                    leg.qty = keep
                else:
                    ledger.legs.pop(sym)
            nav = share_equity + ledger.cash
    ledger.pending = {s: {"expiry": t["expiry"], "qty": t["qty"], "lot": t["lot"]} for s, t in targets.items()
                      if abs(t["qty"] - (ledger.legs[s].qty if s in ledger.legs else 0.0)) > 1e-9}
    ledger.weights = weights
    notional = sum(leg.qty * leg.ref for leg in ledger.legs.values())
    margin_now = sum(_margin_rate(cfg, vol.get(s, np.nan)) * leg.qty * leg.ref for s, leg in ledger.legs.items())
    biggest = max([leg.qty * leg.ref for leg in ledger.legs.values()] + [0.0])
    # the premium held over spot: what the contracts will give back as carry by expiry (rupees)
    premium = sum(leg.qty * (leg.ref - spot[s]["raw_close"]) for s, leg in ledger.legs.items()
                  if np.isfinite(spot.get(s, {}).get("raw_close", np.nan)))
    prev_nav = ledger.history[-1]["nav"] if ledger.history else start_nav
    row = {"date": key, "nav": nav, "share_equity": share_equity, "futures_cash": ledger.cash,
           "futures_pnl": pnl, "costs": sum(costs.values()), "roll_costs": costs["roll"],
           "notional": notional, "exposure": (share_invested + notional) / nav if nav > 0 else 0.0,
           "margin": margin_now, "margin_pct": margin_now / nav if nav > 0 else 0.0,
           "legs": len(ledger.legs), "biggest_pct": biggest / nav if nav > 0 else 0.0, "margin_calls": calls,
           "premium": premium, "premium_pct": premium / notional if notional > 0 else 0.0,
           "state": state, "scale": scale, "day_pnl": nav - prev_nav}
    ledger.history.append(row)
    ledger.fills = (ledger.fills + fills)[-500:]
    ledger.last_session = key
    return {"session": key, **row, "fills": fills, "rolls": rolls,
            "queued": [{"symbol": s, **o, "lots": round(o["qty"] / o["lot"], 3)} for s, o in ledger.pending.items()],
            "holdings": [{"symbol": s, "expiry": leg.expiry, "lots": round(leg.qty / leg.lot, 3),
                          "notional": leg.qty * leg.ref} for s, leg in sorted(ledger.legs.items())],
            "metrics": metrics(ledger)}


def metrics(ledger: Ledger, rf: float = 0.065) -> Dict:
    """The combined book since its first session: return, CAGR, Sharpe, MaxDD, costs, exposure."""
    if not ledger.history:
        return {}
    h = pd.DataFrame(ledger.history)
    nav = h["nav"].astype(float)
    start = nav.iloc[0] - h["day_pnl"].iloc[0]
    r = nav.pct_change().fillna(nav.iloc[0] / start - 1.0)
    days = len(r)
    eq = (1 + r).cumprod()
    out = {"sessions": days, "return": float(eq.iloc[-1] - 1.0), "max_drawdown": float((eq / eq.cummax() - 1).min()),
           "exposure_mean": float(h["exposure"].mean()), "margin_mean": float(h["margin_pct"].mean()),
           "costs_total": float(h["costs"].sum()), "roll_costs_total": float(h["roll_costs"].sum()),
           "margin_calls": int(h["margin_calls"].sum())}
    if days >= 20:
        ex = r - rf / 252
        out["sharpe"] = float(ex.mean() / ex.std(ddof=1) * math.sqrt(252)) if ex.std(ddof=1) > 0 else float("nan")
        out["cagr"] = float(eq.iloc[-1] ** (252 / days) - 1.0)
    return out


def spot_inputs(data, day: pd.Timestamp, symbols) -> Dict[str, Dict[str, float]]:
    """What ``run_session`` reads from the store for ``symbols`` on ``day``: opens, raw and adjusted closes
    today and the session before, 60-day volatility."""
    out: Dict[str, Dict[str, float]] = {}
    if day not in data.close.index:
        return out
    i = data.close.index.get_loc(day)
    raw = data.close_unadj if data.close_unadj is not None else data.close
    for s in symbols:
        if s not in data.close.columns or i == 0:
            continue
        adj = data.close[s]
        vol = float(adj.iloc[max(0, i - 60): i + 1].pct_change().std())
        out[s] = {"open": float(data.open[s].iloc[i]), "adj_close": float(adj.iloc[i]), "adj_prev": float(adj.iloc[i - 1]),
                  "raw_close": float(raw[s].iloc[i]), "raw_prev": float(raw[s].iloc[i - 1]), "vol": vol}
    return out
