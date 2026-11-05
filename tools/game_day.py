"""
Replay game day (golden-target review, "do now"; India tracker GL1): the live session's own code, night after
night over historic sessions, against a simulated Kite account built from the store's as-printed bars.

Nothing leaves this process.  The broker is ``SimKite`` (orders, fills, GTTs, holdings, margins, the instruments
dump); the live book is an in-memory ``MemoryBook``; emails, alerts, order records and the latched kill switch
are replaced for the run, and the database is never opened.  Each night runs ``live_session.run_live_session`` as
the 19:00 job does, with real orders (``dry_run=False``) to the simulator.

The simulator's market: AMOs placed tonight are released at the next open.  A buy is checked against the cash
free before that open (Zerodha could not confirm the same-open sale credit, ZR1), then filled by live's DAY-limit
rule on the day's bar (``live_rules.limit_fill``); a sell needs the shares.  A name locked at its circuit does not
fill against the lock.  Stop GTTs trigger on the bar after the AMOs (``live_rules.stop_gtt_fill``): a gap through
the limit leaves the triggered order unfilled, cancelled at the close.  Bought shares are T1 the next day and
settle the day after; a split, bonus or consolidation in the store's corporate-action table cancels the stock's
GTTs on its ex-date, as Kite does, and its shares arrive the next session.  Costs are the statutory model's.  Ticks are 0.05 throughout.

Faults, one per night: ``down`` (every broker call fails), ``rerun`` (the session runs twice), ``runaway`` (the
plan's quantities x50), ``kill`` (the kill switch on).  Pass rules (the review's): the regime exit and the
drawdown rule fire when due, no order is sent twice, no reject storm, the ledger ends where the broker is, and
each fault does what it should.

    python -m tools.game_day --scenario 2020            # 15 Jan - 30 Jun 2020
    python -m tools.game_day --scenario 2008            # the 2008 year (store_ext2006)
    python -m tools.game_day --scenario faults          # May - Jul 2024 with the four faults injected

Writes ``data/nse_engine/game_day/<scenario>.json`` (counts and verdicts, no book detail) and prints it.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OUT = Path("data/nse_engine/game_day")
TICK = 0.05
CIRCUIT_BAND = 0.20                       # the quote's band when the store holds none
CAPITAL = 300_000.0                       # the ladder's first rung (GT2I's canary)
RUNAWAY_FACTOR = 50
SCENARIOS = {
    "2008": {"start": "2008-01-01", "end": "2008-12-31", "faults": {}},
    "2020": {"start": "2020-01-15", "end": "2020-06-30", "faults": {}},
    "faults": {"start": "2024-05-02", "end": "2024-07-15",
               "faults": {"2024-05-21": "rerun", "2024-05-27": "down", "2024-06-03": "runaway",
                          "2024-06-04": "kill", "2024-06-10": "rerun"}},
}


# ---------------------------------------------------------------------------------------------- the broker

class BrokerDown(ConnectionError):
    pass


class SimKite:
    """A Kite account on historic bars: the calls the live session makes, with Kite's shapes."""

    def __init__(self, bars: pd.DataFrame, cash: float, dp_charge_inr: float,
                 actions: Optional[pd.DataFrame] = None):
        self.bars = bars                                   # (date, symbol) -> open high low close prev_close
        # share-count actions (bonus, split, consolidation): ex-date -> {symbol: shares per share held}
        self.actions: Dict[pd.Timestamp, Dict[str, float]] = {}
        for r in (actions.itertuples() if actions is not None else ()):
            self.actions.setdefault(pd.Timestamp(r.ex_date), {})[str(r.symbol)] = 1.0 / float(r.factor)
        self.credit: Dict[str, int] = {}                   # entitled on the ex-date, credited the next session
        self.cash = float(cash)
        self.dp = float(dp_charge_inr)
        self.settled: Dict[str, int] = {}
        self.t1: Dict[str, int] = {}
        self.today: Dict[str, int] = {}                    # CNC net traded today (positions)
        self.avg: Dict[str, float] = {}
        self.day: Optional[pd.Timestamp] = None
        self.book: List[dict] = []                         # today's order book
        self.amo: List[dict] = []                          # placed tonight, released at the next open
        self.gtts: Dict[int, dict] = {}
        self.down = False
        self.sent_tags: Counter = Counter()                # engine tags sent, all nights
        self.rejected: List[dict] = []
        self.events: Counter = Counter()
        self._oid = 0
        self._gid = 0
        self.symbols = sorted(set(bars.index.get_level_values("symbol")))

    # -- the market day
    def _bar(self, sym: str, day=None) -> Optional[pd.Series]:
        key = (pd.Timestamp(day if day is not None else self.day), sym)
        return self.bars.loc[key] if key in self.bars.index else None

    def held(self, sym: str) -> int:
        return self.settled.get(sym, 0) + self.t1.get(sym, 0) + self.today.get(sym, 0)

    def open(self, day: pd.Timestamp) -> None:
        """Roll to ``day``'s session: settle, corporate actions, release last night's AMOs, then the GTTs."""
        from kite_connect.trading.nse_instruments import to_engine
        from nse_engine import live_rules as lr
        from nse_engine.costs import statutory_cost

        prev_day, self.day = self.day, pd.Timestamp(day)
        for s, q in self.t1.items():                       # T1 settles, today's trades become T1
            self.settled[s] = self.settled.get(s, 0) + q
        self.t1 = {s: q for s, q in self.today.items() if q > 0}
        for s, q in self.today.items():
            if q < 0:
                self.settled[s] = self.settled.get(s, 0) + q
        self.today = {}
        self.settled = {s: q for s, q in self.settled.items() if q > 0}
        for s, extra in self.credit.items():               # last ex-date's shares arrive (a consolidation: go)
            self.settled[s] = self.settled.get(s, 0) + extra
        self.settled = {s: q for s, q in self.settled.items() if q > 0}
        self.credit = {}
        for s, per in self.actions.get(self.day, {}).items():
            q = self.settled.get(s, 0) + self.t1.get(s, 0)
            if q <= 0:
                continue
            self.credit[s] = int(np.floor(q * per + 1e-9)) - q    # entitled at the ex-date; fractions paid in cash
            self.avg[s] = self.avg.get(s, 0.0) / per
            for g in self.gtts.values():                   # Kite cancels the stock's GTTs on the action
                if g["condition"]["tradingsymbol"] in (s, f"{s}-BE") and g["status"] == "active":
                    g["status"], g["updated_at"] = "cancelled", f"{self.day.date()} 09:00:00"
            self.events["corporate_action"] += 1
        self.book = []
        start_cash = self.cash
        committed = 0.0
        released, self.amo = self.amo, []
        for o in sorted(released, key=lambda o: o["transaction_type"] != "SELL"):
            sym = to_engine(o["tradingsymbol"])
            b = self._bar(sym)
            o["status"], o["filled_quantity"], o["average_price"] = "CANCELLED AMO", 0, 0.0
            if o["transaction_type"] == "BUY":
                need = o["quantity"] * o["price"]
                if need > start_cash - committed + 1e-6:   # checked at release, before any sale credit
                    o["status"], o["status_message"] = "REJECTED", "Insufficient funds. Required margin is higher"
                    self.rejected.append({"day": str(self.day.date()), "reason": "funds", "tag": o["tag"]})
                    self.book.append(o)
                    continue
                committed += need
            elif o["quantity"] > self.held(sym):
                o["status"], o["status_message"] = "REJECTED", "Insufficient holdings"
                self.rejected.append({"day": str(self.day.date()), "reason": "holdings", "tag": o["tag"]})
                self.book.append(o)
                continue
            if b is not None and not self._locked(b, o["transaction_type"]):
                kind, px = lr.limit_fill(o["transaction_type"], float(b["open"]), float(b["high"]),
                                         float(b["low"]), float(o["price"]))
                if px is not None:
                    self._fill(o, sym, px, statutory_cost)
            elif b is not None:
                self.events["locked_unfilled"] += 1
            if o["status"] != "COMPLETE":
                o["status"] = "CANCELLED"                  # a DAY order lapses at the close
            self.book.append(o)
        for gid, g in sorted(self.gtts.items()):           # stops after the open's AMOs
            if g["status"] != "active":
                continue
            broker = g["condition"]["tradingsymbol"]
            sym = to_engine(broker)
            b = self._bar(sym)
            if b is None:
                continue
            leg = g["orders"][0]
            trigger, limit = float(g["condition"]["trigger_values"][0]), float(leg["price"])
            kind, px = lr.stop_gtt_fill(float(b["open"]), float(b["high"]), float(b["low"]), trigger, limit)
            if kind is None:
                continue
            self._oid += 1
            o = {"order_id": f"G{self._oid}", "tradingsymbol": broker, "exchange": "NSE", "transaction_type": "SELL",
                 "quantity": int(leg["quantity"]), "product": "CNC", "order_type": "LIMIT", "price": limit,
                 "trigger_price": trigger, "variety": "regular", "validity": "DAY", "tag": None,
                 "status": "OPEN", "filled_quantity": 0, "average_price": 0.0}
            g["status"], g["updated_at"] = "triggered", f"{self.day.date()} 09:15:00"
            leg["result"] = {"order_result": {"order_id": o["order_id"], "status": "success"}}
            if int(leg["quantity"]) > self.held(sym):
                o["status"], o["status_message"] = "REJECTED", "Insufficient holdings"
                self.events["gtt_rejected_not_held"] += 1
            elif px is None or kind == lr.UNFILLED:
                o["status"], o["status_message"] = "CANCELLED", "gapped through the stop's limit"
                self.events["gap_through_stop"] += 1
            else:
                self._fill(o, sym, px, statutory_cost)
                self.events["stop_filled"] += 1
            self.book.append(o)

    @staticmethod
    def _locked(b: pd.Series, side: str) -> bool:
        flat = abs(float(b["high"]) - float(b["low"])) <= 1e-9
        moved = float(b["close"]) - float(b["prev_close"])
        return flat and ((side == "SELL" and moved < 0) or (side == "BUY" and moved > 0))

    def _fill(self, o: dict, sym: str, px: float, statutory_cost) -> None:
        q, side = int(o["quantity"]), o["transaction_type"]
        value = q * px
        cost = statutory_cost(value, side, self.day, self.dp, symbol=sym)
        if side == "BUY":
            held = self.held(sym)
            self.avg[sym] = (self.avg.get(sym, 0.0) * held + value) / (held + q)
            self.cash -= value + cost
            self.today[sym] = self.today.get(sym, 0) + q
        else:
            self.cash += value - cost
            self.today[sym] = self.today.get(sym, 0) - q
        o.update(status="COMPLETE", filled_quantity=q, average_price=px)

    # -- Kite's API
    def _up(self) -> None:
        if self.down:
            raise BrokerDown("simulated: Kite unreachable")

    def profile(self):
        self._up()
        return {"user_id": "SIM01"}

    def instruments(self, exchange="NSE"):
        self._up()
        rows = []
        for s in self.symbols:
            rows.append({"tradingsymbol": s, "tick_size": TICK, "segment": "NSE", "instrument_type": "EQ",
                         "exchange": "NSE"})
        return rows

    def holdings(self):
        self._up()
        out = []
        for s in sorted(set(self.settled) | set(self.t1)):
            out.append({"tradingsymbol": s, "exchange": "NSE", "product": "CNC", "isin": f"SIM{s}",
                        "quantity": self.settled.get(s, 0), "t1_quantity": self.t1.get(s, 0),
                        "average_price": self.avg.get(s, 0.0)})
        return out

    def positions(self):
        self._up()
        net = [{"tradingsymbol": s, "exchange": "NSE", "product": "CNC", "quantity": q}
               for s, q in self.today.items() if q]
        return {"net": net, "day": net}

    def margins(self, segment=None):
        self._up()
        return {"available": {"live_balance": self.cash, "cash": self.cash}}

    def orders(self):
        self._up()
        return [dict(o) for o in self.book + self.amo]

    def order_history(self, order_id):
        self._up()
        return [dict(o) for o in self.book + self.amo if o["order_id"] == order_id]

    def ltp(self, keys):
        self._up()
        out = {}
        for k in keys:
            b = self._bar(k.split(":", 1)[-1].replace("-BE", ""))
            if b is not None:
                out[k] = {"last_price": float(b["close"])}
        return out

    def quote(self, keys):
        self._up()
        out = {}
        for k in keys:
            b = self._bar(k.split(":", 1)[-1].replace("-BE", ""))
            if b is not None:
                c = float(b["close"])
                out[k] = {"last_price": c, "ohlc": {"close": c}, "lower_circuit_limit": c * (1 - CIRCUIT_BAND),
                          "upper_circuit_limit": c * (1 + CIRCUIT_BAND)}
        return out

    def place_order(self, variety, exchange, tradingsymbol, transaction_type, quantity, product, order_type,
                    price=None, validity=None, tag=None, **_):
        self._up()
        if order_type != "LIMIT" or variety != "amo" or product != "CNC":
            raise ValueError(f"simulator takes CNC LIMIT AMOs only, got {product} {order_type} {variety}")
        self._oid += 1
        o = {"order_id": f"O{self._oid}", "tradingsymbol": tradingsymbol, "exchange": exchange,
             "transaction_type": transaction_type, "quantity": int(quantity), "product": product,
             "order_type": order_type, "price": float(price), "variety": variety, "validity": validity,
             "tag": tag, "status": "AMO REQ RECEIVED", "filled_quantity": 0, "average_price": 0.0}
        self.amo.append(o)
        if tag:
            self.sent_tags[tag] += 1
        return o["order_id"]

    def cancel_order(self, variety, order_id, **_):
        self._up()
        for o in self.amo:
            if o["order_id"] == order_id:
                o["status"] = "CANCELLED AMO"
        self.amo = [o for o in self.amo if o["status"] != "CANCELLED AMO"]
        return order_id

    def modify_order(self, variety, order_id, **kw):
        self._up()
        for o in self.amo:
            if o["order_id"] == order_id:
                o.update({k: v for k, v in kw.items() if v is not None})
        return order_id

    def get_gtts(self):
        self._up()
        return [json.loads(json.dumps(g)) for g in self.gtts.values()]

    def get_gtt(self, trigger_id):
        self._up()
        return json.loads(json.dumps(self.gtts[int(trigger_id)]))

    def place_gtt(self, trigger_type, tradingsymbol, exchange, trigger_values, last_price, orders):
        self._up()
        self._gid += 1
        self.gtts[self._gid] = self._gtt(self._gid, tradingsymbol, exchange, trigger_values, orders)
        return {"trigger_id": self._gid}

    def modify_gtt(self, trigger_id, trigger_type, tradingsymbol, exchange, trigger_values, last_price, orders):
        self._up()
        self.gtts[int(trigger_id)] = self._gtt(int(trigger_id), tradingsymbol, exchange, trigger_values, orders)
        return {"trigger_id": trigger_id}

    def delete_gtt(self, trigger_id):
        self._up()
        g = self.gtts.get(int(trigger_id))
        if g is not None:
            g["status"], g["updated_at"] = "deleted", f"{self.day.date()} 19:00:00"
        return {"trigger_id": trigger_id}

    def _gtt(self, gid, symbol, exchange, trigger_values, orders) -> dict:
        legs = [{**dict(o), "tradingsymbol": symbol, "exchange": exchange} for o in orders]
        return {"id": gid, "type": "single", "status": "active", "updated_at": f"{self.day.date()} 19:00:00",
                "condition": {"tradingsymbol": symbol, "exchange": exchange,
                              "trigger_values": [float(t) for t in trigger_values]}, "orders": legs}


# ---------------------------------------------------------------------------------------------- the live book

class MemoryBook:
    """The live book's store, in memory: the calls ``run_live_session`` makes."""

    schema = "game_day"

    def __init__(self):
        self.state: Dict[str, str] = {}
        self.snaps: Dict[str, dict] = {}
        self.fills: List[dict] = []
        self.sessions: List[dict] = []

    def read_state(self) -> Dict[str, str]:
        return dict(self.state)

    def sync_state(self, values) -> bool:
        self.state.update({k: (v if isinstance(v, str) else json.dumps(v)) for k, v in values.items()})
        return True

    def read_snapshots(self, since_epoch: bool = True, strict: bool = False) -> pd.DataFrame:
        return pd.DataFrame(sorted(self.snaps.values(), key=lambda r: r["date"]))

    def sync_snapshot(self, snap: dict) -> bool:
        self.snaps[snap["date"]] = dict(snap)
        return True

    def sync_fills(self, fills) -> bool:
        self.fills.extend(fills)
        return True

    def read_fills(self, since_epoch: bool = True, strict: bool = False) -> pd.DataFrame:
        return pd.DataFrame(self.fills)

    def sync_session(self, row: dict) -> bool:
        self.sessions = [s for s in self.sessions if s.get("date") != row.get("date")] + [dict(row)]
        return True

    def read_sessions(self, since_epoch: bool = True, strict: bool = False) -> pd.DataFrame:
        return pd.DataFrame(self.sessions)

    def read_positions(self, since_epoch: bool = True) -> pd.DataFrame:
        return pd.DataFrame()


# ---------------------------------------------------------------------------------------------- the run

def _isolate(alerts: List[tuple]) -> None:
    """Nothing leaves the process: no email, alert, order record, database or latched kill switch."""
    from kite_connect.trading import kill_switch, nse_instruments, order_service
    from services.notifications import alerts as al

    order_service._persist_to_db = lambda *a, **k: None
    order_service._send_order_email = lambda *a, **k: None
    order_service._is_nse_market_open = lambda *a, **k: False     # the 19:00 job: after-market orders
    order_service.ORDERS_PER_SECOND = 10_000                       # no pacing sleeps in a replay
    kill_switch.state = lambda *a, **k: {"on": os.environ.get("CENTURION_KILL_SWITCH", "").lower() == "true"}
    al.alert = lambda severity, key, title, lines=(), book=None, once=False: alerts.append((severity, title)) or True
    nse_instruments._cache.clear()

    import database.connection as dbc

    def no_db():
        raise RuntimeError("game day: the database is never opened")
    dbc.get_db_manager = no_db
    for name in ("CENTURION_DATABASE_URL", "DATABASE_URL", "CENTURION_EMAIL_USER", "CENTURION_EMAIL_PASS"):
        os.environ.pop(name, None)
    os.environ.update({"CENTURION_PAPER_TRADE": "false", "CENTURION_NSE_ENGINE_LIVE": "true",
                       "CENTURION_GO_LIVE_OVERRIDE": "true", "CENTURION_KILL_SWITCH": "false"})


def _bars(store: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    parts = [pd.read_parquet(Path(store) / "equity" / f"{y}.parquet",
                             columns=["date", "symbol", "series", "open", "high", "low", "close", "prev_close"])
             for y in range(start.year, end.year + 1)]
    b = pd.concat(parts, ignore_index=True)
    b = b[(b["date"] >= start - pd.Timedelta(days=10)) & (b["date"] <= end) & b["series"].isin(["EQ", "BE"])]
    b = b.assign(_r=(b["series"] != "EQ").astype(int)).sort_values("_r").drop_duplicates(["date", "symbol"])
    b["symbol"] = b["symbol"].astype(str)
    return b.drop(columns=["series", "_r"]).set_index(["date", "symbol"]).sort_index()


def run(scenario: str, start: str, end: str, faults: Dict[str, str]) -> dict:
    from kite_connect.trading import live_session as ls
    from kite_connect.trading import nse_engine_executor as nee
    from nse_engine.deployment import load_deployment

    alerts: List[tuple] = []
    _isolate(alerts)
    s0, s1 = pd.Timestamp(start), pd.Timestamp(end)
    dep = load_deployment()
    store = "data/nse_engine/store_ext2006" if s0.year < 2012 else dep.engine.data.store_dir
    engine = dep.engine.replace(**{"data.store_dir": store})
    anchor = min(dep.data_start(), date(s0.year - 2, 1, 1))
    dep = dataclasses.replace(dep, engine=engine, paper_start_date=s0.date(), data_anchor_date=anchor)
    bars = _bars(store, s0, s1)
    sessions = [d for d in sorted(set(bars.index.get_level_values("date"))) if s0 <= d <= s1]
    missing = sorted(set(faults) - {str(d.date()) for d in sessions})
    if missing:
        raise ValueError(f"fault dates that are not sessions in {start}..{end}: {missing}")
    acts = pd.read_parquet(Path(store) / "corporate_actions.parquet", columns=["symbol", "ex_date", "purpose", "factor"])
    acts = acts[acts["factor"].gt(0) & acts["factor"].ne(1.0) & (acts["ex_date"] >= s0) & (acts["ex_date"] <= s1)
                & ~acts["purpose"].astype(str).str.upper().str.match(r"^(DIV|INT|AGM)")]
    sim = SimKite(bars, CAPITAL, engine.costs.dp_charge_inr, acts)
    book, paper = MemoryBook(), MemoryBook()
    rows, plan_orig = [], nee.EngineExecutor.plan
    capital = CAPITAL
    for day in sessions:
        sim.open(day)
        fault = faults.get(str(day.date()))
        n_sent = sum(sim.sent_tags.values())
        sim.down = fault == "down"
        os.environ["CENTURION_KILL_SWITCH"] = "true" if fault == "kill" else "false"
        if fault == "runaway":
            def runaway(self, *a, **k):
                p = plan_orig(self, *a, **k)
                for o in p.orders:                       # a quantity bug: every order x50
                    o.quantity *= RUNAWAY_FACTOR
                    o.target_qty = o.current_qty + (o.quantity if o.side == "BUY" else -o.quantity)
                return p
            nee.EngineExecutor.plan = runaway
        row = {"day": str(day.date()), "fault": fault or ""}
        try:
            for attempt in range(2 if fault == "rerun" else 1):
                rep = ls.run_live_session(sim, as_of=day, book=book, deployment=dep, capital=capital, email=False,
                                          paper_book=paper, trial_gates={}, requested_capital=capital)
                if attempt == 0:
                    first_sent = sum(sim.sent_tags.values())
            row["error"] = ""
        except BrokerDown as exc:
            row["error"], rep = f"broker down: {exc}", None
        except Exception as exc:                          # noqa: BLE001 - recorded: a game day must see it
            row["error"], rep = f"{type(exc).__name__}: {exc}", None
            logger.exception("session %s failed", day.date())
        finally:
            nee.EngineExecutor.plan = plan_orig
            sim.down = False
        sent_now = sum(sim.sent_tags.values()) - n_sent
        row.update(sent=sent_now, rerun_extra=(sum(sim.sent_tags.values()) - first_sent) if fault == "rerun"
                   and rep is not None else 0)
        if rep is not None:
            plan = rep.get("plan")
            orders = [r for r in rep.get("results", []) if "tag" in r]
            row.update(alerts=len(rep["alerts"]), reconcile=sum("RECONCILE NEEDED" in a for a in rep["alerts"]),
                       cap=sum("SESSION CAP" in a for a in rep["alerts"]),
                       buys_sent=sum(1 for r in orders if r["side"] == "BUY" and r.get("success")),
                       sells_sent=sum(1 for r in orders if r["side"] == "SELL" and r.get("success")),
                       regime_scale=float(getattr(getattr(plan, "target", None), "regime_scale", np.nan)),
                       dd_state=getattr(plan, "drawdown_state", "") if plan is not None else "",
                       equity=float(rep["snapshot"]["equity"]) if rep.get("snapshot") else np.nan)
            new_cap = float((rep.get("ledger") or {}).get("capital") or capital)
            if abs(new_cap - capital) > 1.0:              # the owner funds (or withdraws) a ladder move
                sim.cash += new_cap - capital
                capital = new_cap
        rows.append(row)
        logger.warning("%s %s sent %d alerts %s %s", day.date(), fault or "-", sent_now, row.get("alerts"),
                       row["error"][:80])
    return _verdict(scenario, start, end, faults, pd.DataFrame(rows), sim, book, alerts)


def _verdict(scenario, start, end, faults, df: pd.DataFrame, sim: SimKite, book: MemoryBook, alerts) -> dict:
    eq = df["equity"].dropna() if "equity" in df else pd.Series(dtype=float)
    dd = float((eq / eq.cummax() - 1).min()) if len(eq) else 0.0
    from kite_connect.trading.live_session import LIVE_LEDGER_KEY

    ledger = json.loads(book.state.get(LIVE_LEDGER_KEY) or "{}")
    positions = {s: int(q) for s, q in (ledger.get("positions") or {}).items() if int(q) > 0}
    broker = {s: sim.held(s) for s in set(sim.settled) | set(sim.t1) | set(sim.today) if sim.held(s) > 0}
    after_fault = set()
    for d, f in faults.items():
        later = df.index[df["day"] > d]
        after_fault |= {d} | ({df.loc[later[0], "day"]} if len(later) else set())
    clean = df[~df["day"].isin(after_fault)]
    checks = {
        "no_order_sent_twice": all(v == 1 for v in sim.sent_tags.values()),
        "no_reject_storm": len(sim.rejected) == 0,
        "no_unexplained_reconcile": int(clean.get("reconcile", pd.Series(dtype=float)).fillna(0).sum()) == 0,
        "no_session_error_without_fault": not (clean["error"].astype(bool)).any(),
        "ledger_matches_broker": positions == broker,
        "drawdown_rule_fired_when_due": dd > -0.20 or bool((df.get("dd_state", pd.Series(dtype=str))
                                                             .fillna("normal") != "normal").any()),
    }
    off = df[df.get("regime_scale", pd.Series(np.nan, index=df.index)) <= 0] if "regime_scale" in df else df.iloc[:0]
    if scenario in ("2008", "2020"):                      # the regime turned the core off, and its sells went out
        checks["regime_exit_sent"] = bool(len(off)) and int(off["sells_sent"].fillna(0).sum()) > 0
    for d, f in faults.items():
        r = df[df["day"] == d]
        if r.empty:
            checks[f"fault_{f}_{d}"] = False
            continue
        r = r.iloc[0]
        if f == "down":
            checks[f"fault_down_{d}_fails_loudly_sends_nothing"] = bool(r["error"]) and int(r["sent"]) == 0
        elif f == "rerun":
            checks[f"fault_rerun_{d}_no_second_send"] = int(r["rerun_extra"]) == 0 and not r["error"]
        elif f == "runaway":
            checks[f"fault_runaway_{d}_capped_nothing_sent"] = (int(r.get("cap", 0) or 0) > 0 and int(r["sent"]) == 0)
        elif f == "kill":
            checks[f"fault_kill_{d}_no_buy_sent"] = int(r.get("buys_sent", 0) or 0) == 0 and not r["error"]
    out = {"scenario": scenario, "window": [start, end], "sessions": int(len(df)),
           "orders_sent": int(sum(sim.sent_tags.values())), "rejected": Counter(r["reason"] for r in sim.rejected),
           "events": dict(sim.events), "regime_off_sessions": int(len(off)),
           "sells_on_regime_off_nights": int(off["sells_sent"].fillna(0).sum()) if len(off) else 0,
           "drawdown_states": dict(Counter(df.get("dd_state", pd.Series(dtype=str)).fillna(""))),
           "max_drawdown_pct": round(100 * dd, 1), "alert_emails_suppressed": len(alerts),
           "session_errors": df.loc[df["error"].astype(bool), ["day", "fault", "error"]].to_dict("records"),
           "ledger_vs_broker": {s: [positions.get(s, 0), broker.get(s, 0)] for s in sorted(set(positions) | set(broker))
                                if positions.get(s, 0) != broker.get(s, 0)},           # local file only
           "checks": checks, "verdict": "PASS" if all(checks.values()) else "FAIL"}
    return {**out, "_sessions": df}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    ap.add_argument("--start", help="override the scenario's first session")
    ap.add_argument("--end", help="override its last")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sc = SCENARIOS[args.scenario]
    out = run(args.scenario, args.start or sc["start"], args.end or sc["end"], sc["faults"])
    OUT.mkdir(parents=True, exist_ok=True)
    sessions = out.pop("_sessions")
    (OUT / f"{args.scenario}.json").write_text(json.dumps(out, indent=1, default=str))
    sessions.to_csv(OUT / f"{args.scenario}_sessions.csv", index=False)        # counts per night
    print(json.dumps(out, indent=1, default=str))
    return 0 if out["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
