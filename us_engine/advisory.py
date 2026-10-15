"""US advisory book (tracker US-A1): the India deployed rules on today's S&P 500,
as suggestions you place by hand.  Never places an order.

UNVALIDATED: no US trial has passed a gate (US2-US7 do that).  The rules are
the India book's as deployed (``config/nse_engine_deployed.json``: EWMAC trend
forecasts, 20 names, weekly rebalance, 6-ATR trailing stops, the regime gate
and the drawdown rule), with US inputs (``US_OVERRIDES``): the S&P 500 and VIX
for the regime, thresholds in USD, schedules on the calendar, and no metals
sleeves (the India book holds them).

Each run after a US close (``python -m us_engine.advisory run``):

1. loads today's S&P 500 through the last final session (``us_engine.data``);
2. advances the model book over the sessions since its last decision: splits;
   the previous suggestions filled at the next open, sells first; resting
   stops (``portfolio.stop_fill_price``); the close mark;
3. plans the next session with ``nse_engine.engine.generate_targets`` and the
   deployment's drawdown rule on the book's own equity;
4. saves the book (Neon schema ``us_advisory``, or ``--state-file``), journals the
   run in that schema's tables as the India books do (fills, positions, daily
   snapshots, sessions, the suggested orders; ``journal_rows``) and emails the
   suggestions: sells, buys, raised stops, the book against the S&P 500.

The model book is notional (``NOTIONAL_USD``: whole shares that stand for
fractional ones) and assumes every suggestion filled at the next open; the
email scales it to your capital (``US_ADVISORY_CAPITAL_USD``) in dollars and
fractional shares.  Actions logs are public, so a run there logs counts only;
the names and amounts go by email.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, time, timezone
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from nse_engine.config import DrawdownConfig, EngineConfig
from nse_engine.deployment import load_deployment
from nse_engine.drawdown import DrawdownDecision, replay, scale_for
from nse_engine.engine import generate_targets
from nse_engine.portfolio import stop_fill_price
from nse_engine.types import Holding, TargetPortfolio
from us_engine.data import NEW_YORK, UsPanel, load_us_data

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT = REPO_ROOT / "config" / "nse_engine_deployed.json"
IST = ZoneInfo("Asia/Kolkata")
#: Large so whole shares stand for the fractional shares US brokers sell (rounding
#: under 0.01% of the book); the email scales every amount to your capital.
NOTIONAL_USD = 10_000_000.0
#: Capital the email sizes for when ``US_ADVISORY_CAPITAL_USD`` is not set.
REFERENCE_CAPITAL = 10_000.0
#: Per side, on every model fill: spread and slippage at a zero-commission broker (US3 replaces it).
FILL_COST_BPS = 10.0
SCHEMA = "us_advisory"
STATE_KEY = "us_advisory_book"
BOOK_VERSION = 1
ENV_CAPITAL = "US_ADVISORY_CAPITAL_USD"
US_OPEN = time(9, 30)
#: The deployed India rules with US inputs; every other field is the deployment's.
US_OVERRIDES = {
    "initial_capital": NOTIONAL_USD,
    "universe.top_n_liquid": 600,              # every member that passes the filters
    "universe.min_price_inr": 5.0,             # USD: the field names are the NSE engine's
    "universe.min_median_value_inr": 2.0e7,    # USD 20 million a day
    "universe.calendar_schedule": True,        # schedules on the calendar, not on row counts
    "signals.calendar_schedule": True,
    "portfolio.calendar_schedule": True,       # rebalance on each week's first session
    "portfolio.min_trade_value_inr": 0.01 * NOTIONAL_USD,  # 1%, as Rs 5,000 is of the India book's Rs 5 lakh
    "regime.index_symbol": "SPX",
    "regime.vix_symbol": "VIX",
    "sleeves.gold_enabled": False,             # the India book holds the metals
    "sleeves.silver_enabled": False,
}


def us_config(path: Path = DEPLOYMENT) -> Tuple[EngineConfig, Optional[DrawdownConfig]]:
    """(engine config, drawdown rule): the India deployment with ``US_OVERRIDES``."""
    dep = load_deployment(path)
    return dep.engine.replace(**US_OVERRIDES), dep.drawdown_rule


# ============================================================================
# the model book
# ============================================================================


@dataclass
class Book:
    """The model book, stored as JSON.  Dates are ISO strings; prices in USD."""

    start: str                       # session of the first decision
    last_session: str                # session of the latest decision
    cash: float
    #: symbol -> {qty, avg_price, entry_date, stop, last_price}
    holdings: Dict[str, Dict] = field(default_factory=dict)
    #: orders for the next open: {symbol, side, qty, reason, stop}
    pending: List[Dict] = field(default_factory=list)
    stopped_out: Dict[str, str] = field(default_factory=dict)   # symbol -> session of its last stop-out
    equity: List[List] = field(default_factory=list)            # [session, equity at the close]
    trades: List[Dict] = field(default_factory=list)
    emailed: str = ""                # session whose suggestions were emailed
    report_html: str = ""            # that email, for a resend
    version: int = BOOK_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "Book":
        return cls(**json.loads(text))

    def value(self) -> float:
        return float(self.cash + sum(h["qty"] * h["last_price"] for h in self.holdings.values()))


class FileStore:
    """The book in a local JSON file (dry runs and tests)."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> Optional[Book]:
        return Book.from_json(self.path.read_text()) if self.path.exists() else None

    def save(self, book: Book) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(book.to_json())

    def journal(self, rows: Dict[str, List[Dict]]) -> List[str]:
        """A local book keeps no journal tables (they live in Neon)."""
        return []


class NeonStore:
    """The book in Neon: one key of the key/value table in schema ``us_advisory``."""

    def __init__(self):
        if not (os.getenv("CENTURION_DATABASE_URL") or os.getenv("DATABASE_URL")):
            raise RuntimeError("no database URL: set CENTURION_DATABASE_URL, or pass --state-file")
        from database.connection import get_db_manager
        from database.paper_cloud import PaperCloudSync

        self.cloud = PaperCloudSync(get_db_manager(), schema=SCHEMA)
        self.cloud.ensure_tables()

    def load(self) -> Optional[Book]:
        raw = self.cloud.read_state().get(STATE_KEY)     # raises on a DB error: never mistaken for a new book
        return Book.from_json(raw) if raw else None

    def save(self, book: Book) -> None:
        if not self.cloud.sync_state({STATE_KEY: book.to_json()}):
            raise RuntimeError("the US advisory book was not saved to Neon")

    def journal(self, rows: Dict[str, List[Dict]]) -> List[str]:
        """Write this run's journal rows; the names of the tables that failed (upserts: a rerun is safe)."""
        c, failed = self.cloud, []
        if not c.sync_fills(rows["fills"]):
            failed.append("fills")
        if not all([c.sync_position(p) for p in rows["positions"]]):
            failed.append("positions")
        if not all([c.sync_snapshot(s) for s in rows["snapshots"]]):
            failed.append("snapshots")
        if not all([c.sync_session(s) for s in rows["session"]]):
            failed.append("session")
        if not c.sync_signals(rows["as_of"][0], rows["signals"]):
            failed.append("signals")
        return failed


def _px(frame: pd.DataFrame, day: pd.Timestamp, sym: str) -> float:
    return float(frame.at[day, sym]) if sym in frame.columns else float("nan")


def _fill_cost(value: float) -> float:
    return abs(value) * FILL_COST_BPS / 1e4


def _trade(book: Book, day: str, sym: str, side: str, qty: int, px: float, reason: str, meta: Dict,
           status: str = "FILLED", note: str = "", **extra) -> None:
    """One fill (or an order that did not fill) in ``book.trades``: the journal's ``paper_fills`` rows."""
    value = qty * px
    book.trades.append({"date": day, "symbol": sym, "side": side, "qty": qty, "price": round(px, 4),
                        "reason": reason, "status": status, "decided": meta.get("decided", ""),
                        "source": meta.get("source", "pending_open"), "requested": int(meta.get("requested", qty)),
                        "ref_price": round(float(meta.get("ref_price") or 0.0), 4),
                        "cost": round(_fill_cost(value), 2), "note": note or meta.get("note", ""), **extra})


def _sell(book: Book, sym: str, qty: int, px: float, day: str, reason: str, meta: Dict) -> None:
    h = book.holdings[sym]
    value = qty * px
    book.cash += value - _fill_cost(value)
    _trade(book, day, sym, "SELL", qty, px, reason, meta, pnl=round(qty * (px - h["avg_price"]) - _fill_cost(value), 2),
           entry_date=h["entry_date"], avg_price=round(h["avg_price"], 4), closed=h["qty"] - qty <= 0)
    h["qty"] -= qty
    if h["qty"] <= 0:
        del book.holdings[sym]


def _buy(book: Book, sym: str, qty: int, px: float, day: str, reason: str, stop: Optional[float], meta: Dict) -> None:
    value = qty * px
    book.cash -= value + _fill_cost(value)
    _trade(book, day, sym, "BUY", qty, px, reason, meta)
    h = book.holdings.get(sym)
    if h is None:
        book.holdings[sym] = {"qty": qty, "avg_price": px, "entry_date": day, "stop": stop, "last_price": px}
        return
    h["avg_price"] = (h["qty"] * h["avg_price"] + value) / (h["qty"] + qty)
    h["qty"] += qty


def _apply_splits(book: Book, splits: pd.DataFrame, day: pd.Timestamp, events: List[str]) -> None:
    def ratio(sym: str) -> float:
        r = _px(splits, day, sym)
        return r if np.isfinite(r) and r > 0 and r != 1 else 1.0

    for sym, h in book.holdings.items():
        r = ratio(sym)
        if r != 1.0:
            h["qty"] = int(round(h["qty"] * r))
            h["avg_price"] /= r
            h["last_price"] /= r
            if h.get("stop"):
                h["stop"] /= r
            events.append(f"{sym}: {r:g}-for-1 split on {day.date()}; shares and stop adjusted")
    for o in book.pending:
        r = ratio(o["symbol"])
        if r != 1.0:
            o["qty"] = int(round(o["qty"] * r))
            if o.get("stop"):
                o["stop"] /= r


def _fill_pending(book: Book, panel: UsPanel, day: pd.Timestamp, events: List[str]) -> None:
    """The previous suggestions at this session's open: sells first, buys within the cash."""
    data, d = panel.data, day.date().isoformat()

    def meta(o: Dict) -> Dict:
        return {"decided": book.last_session, "requested": o["qty"], "ref_price": o.get("price")}

    def lapse(o: Dict, why: str) -> None:
        events.append(f"{o['symbol']}: {why}")
        _trade(book, d, o["symbol"], o["side"], 0, 0.0, o["reason"], meta(o), status="CANCELLED", note=why)

    for o in [o for o in book.pending if o["side"] == "SELL"]:
        sym, h = o["symbol"], book.holdings.get(o["symbol"])
        if h is None:
            continue
        px = _px(data.open, day, sym)
        note = ""
        if sym not in data.close.columns:
            px = float(h["last_price"])
            note = f"no market data; sold in the model at its last close {px:.2f} (check your broker)"
            events.append(f"{sym}: {note}")
        elif not np.isfinite(px):
            lapse(o, f"no open on {d}; the sell did not fill")
            continue
        _sell(book, sym, min(int(o["qty"]), int(h["qty"])), px, d, o["reason"], {**meta(o), "note": note})
    for o in [o for o in book.pending if o["side"] == "BUY"]:
        sym = o["symbol"]
        px = _px(data.open, day, sym)
        if not np.isfinite(px) or px <= 0:
            lapse(o, f"no open on {d}; the buy did not fill")
            continue
        qty = min(int(o["qty"]), int(math.floor(max(book.cash, 0.0) / (px * (1 + FILL_COST_BPS / 1e4)))))
        if qty <= 0:
            lapse(o, "not enough cash at the open; the buy did not fill")
            continue
        if qty < int(o["qty"]):
            events.append(f"{sym}: bought {qty} of {o['qty']} shares (cash at the open)")
        _buy(book, sym, qty, px, d, o["reason"], o.get("stop"), meta(o))


def _apply_stops(book: Book, panel: UsPanel, day: pd.Timestamp, events: List[str]) -> None:
    data, d = panel.data, day.date().isoformat()
    for sym in list(book.holdings):
        h = book.holdings[sym]
        px = stop_fill_price(_px(data.open, day, sym), _px(data.low, day, sym), h.get("stop"))
        if px is None:
            continue
        _sell(book, sym, int(h["qty"]), px, d, "stop",
              {"decided": d, "source": "stop", "requested": h["qty"], "ref_price": h.get("stop")})
        book.stopped_out[sym] = d
        events.append(f"{sym}: stopped out on {d} at {px:.2f}")


def advance(book: Book, panel: UsPanel) -> List[str]:
    """Move the book through the sessions after its last decision; the events, oldest first."""
    data = panel.data
    sessions = data.dates[data.dates > pd.Timestamp(book.last_session)]
    events: List[str] = []
    for k, day in enumerate(sessions):
        _apply_splits(book, panel.splits, day, events)
        if k == 0:
            _fill_pending(book, panel, day, events)
        _apply_stops(book, panel, day, events)
        for sym, h in book.holdings.items():
            c = _px(data.close, day, sym)
            if np.isfinite(c):
                h["last_price"] = c
        book.equity.append([day.date().isoformat(), round(book.value(), 2)])
    if len(sessions):
        book.pending = []
        book.last_session = sessions[-1].date().isoformat()
    return events


# ============================================================================
# the plan
# ============================================================================


@dataclass
class Order:
    symbol: str
    side: str
    qty: int
    price: float                     # last close
    reason: str
    weight: float                    # target weight after the order
    stop: Optional[float] = None     # initial stop of a new name


@dataclass
class Plan:
    as_of: pd.Timestamp
    target: TargetPortfolio
    orders: List[Order]
    raised: List[Tuple[str, Optional[float], float]]   # (symbol, old stop, new stop)
    drawdown: Optional[DrawdownDecision]
    skipped: List[str] = field(default_factory=list)

    def side(self, side: str) -> List[Order]:
        return [o for o in self.orders if o.side == side]


def drawdown_decision(book: Book, rule: Optional[DrawdownConfig]) -> Optional[DrawdownDecision]:
    """The rule replayed over the book's closes, as the India executor does (nothing stored)."""
    if rule is None or not book.equity:
        return None
    series = pd.Series([e for _, e in book.equity], index=pd.DatetimeIndex([d for d, _ in book.equity]))
    frame = replay(series[~series.index.duplicated(keep="last")], rule)
    row = frame.iloc[-1]
    states = frame["state"].to_numpy()
    run = next((i for i, s in enumerate(states[::-1]) if s != row["state"]), len(states))
    return DrawdownDecision(state=str(row["state"]), scale=float(scale_for(str(row["state"]), rule)),
                            allow_entries=bool(row["allow_entries"]), drawdown=float(row["drawdown"]),
                            peak=float(row["peak"]), changed=bool(row["changed"]), sessions_in_state=run)


def plan_next(book: Book, panel: UsPanel, cfg: EngineConfig, rule: Optional[DrawdownConfig]) -> Plan:
    """Orders for the next open, sized as the India executor sizes them; raises the book's stops."""
    data = panel.data
    as_of = data.dates[-1]
    pcfg = cfg.portfolio
    equity = book.value()
    holdings = {s: Holding(s, int(h["qty"]), float(h["avg_price"]), pd.Timestamp(h["entry_date"]), h.get("stop"))
                for s, h in book.holdings.items()}
    decision = drawdown_decision(book, rule)
    target = generate_targets(data, cfg, as_of, holdings, equity=equity,
                              stopped_out={s: pd.Timestamp(d) for s, d in book.stopped_out.items()},
                              drawdown=decision)
    last_close = data.close.ffill().iloc[-1]
    orders: List[Order] = []
    skipped: List[str] = []
    for sym in dict.fromkeys(list(book.holdings) + list(target.weights) + list(target.exits)):
        cur = int(book.holdings[sym]["qty"]) if sym in book.holdings else 0
        px = float(last_close.get(sym, np.nan))
        if not np.isfinite(px) and sym in book.holdings:
            px = float(book.holdings[sym]["last_price"])
        if not np.isfinite(px) or px <= 0:
            skipped.append(f"{sym}: no price")
            continue
        exit_reason = target.exits.get(sym)
        w = 0.0 if exit_reason else float(target.weights.get(sym, 0.0))
        tgt = int(math.floor(w * equity / px + 1e-6)) if w > 0 else 0
        delta = tgt - cur
        if delta == 0:
            continue
        if not (tgt == 0 and cur > 0) and abs(delta) * px < pcfg.min_trade_value_inr:
            continue
        if delta < 0:
            orders.append(Order(sym, "SELL", -delta, px, exit_reason or ("exit" if tgt == 0 else "trim"), w))
        else:
            orders.append(Order(sym, "BUY", delta, px, "entry" if cur == 0 else "add", w,
                                target.stops.get(sym) if cur == 0 else None))

    budget = book.cash + sum(o.qty * o.price - _fill_cost(o.qty * o.price) for o in orders if o.side == "SELL")
    kept: List[Order] = []
    for o in sorted(orders, key=lambda o: (o.side != "SELL", -o.weight, o.symbol)):
        if o.side == "BUY":
            unit = o.price * (1 + FILL_COST_BPS / 1e4)
            qty = min(o.qty, int(math.floor(max(budget, 0.0) / unit)))
            if qty <= 0 or qty * o.price < pcfg.min_trade_value_inr:
                skipped.append(f"{o.symbol}: not enough cash")
                continue
            o.qty = qty
            budget -= qty * unit
        kept.append(o)

    raised: List[Tuple[str, Optional[float], float]] = []
    selling = {o.symbol for o in kept if o.side == "SELL" and o.weight == 0}
    for sym, h in book.holdings.items():
        new = target.stops.get(sym)
        if sym in selling or new is None:
            continue
        old = h.get("stop")
        if old is None or new > old + 1e-9:
            raised.append((sym, old, float(new)))
            h["stop"] = float(new)
    book.pending = [{"symbol": o.symbol, "side": o.side, "qty": o.qty, "reason": o.reason, "stop": o.stop,
                     "price": round(o.price, 4)} for o in kept]
    return Plan(as_of=as_of, target=target, orders=kept, raised=raised, drawdown=decision, skipped=skipped)


# ============================================================================
# the journal: the book's Neon tables, as the India books keep theirs
# ============================================================================


def _utc(day: str, at: time) -> str:
    return datetime.combine(pd.Timestamp(day).date(), at, tzinfo=NEW_YORK).astimezone(timezone.utc).isoformat()


def journal_rows(book: Book, plan: Plan, new_trades: Sequence[Dict], new_sessions: Sequence[str],
                 events: Sequence[str]) -> Dict[str, List[Dict]]:
    """This run's rows for the ``us_advisory`` tables, in the India books' shapes (``paper_fills``,
    ``paper_positions``, ``paper_daily_snapshots``, ``paper_sessions``, ``paper_signal_log``).

    Money is USD in every column, ``costs_inr`` included (the column names are the India book's).
    Quantities are the model's whole shares at ``NOTIONAL_USD``.
    """
    as_of = plan.as_of.date().isoformat()
    fills = [{
        "order_id": (f"stop-{t['date']}-{t['symbol']}" if t["source"] == "stop"
                     else f"us-{t['decided']}-{t['side']}-{t['symbol']}"),
        "session_date": t["date"], "decision_date": t["decided"], "source": t["source"], "symbol": t["symbol"],
        "side": t["side"], "status": t["status"], "requested_qty": int(t["requested"]), "quantity": int(t["qty"]),
        "ref_price": float(t["ref_price"]), "fill_price": float(t["price"]),
        "impact_bps": FILL_COST_BPS if t["qty"] else 0.0, "costs_inr": float(t["cost"]),
        "pnl": float(t.get("pnl", 0.0)), "note": str(t.get("note", ""))[:200],
        "occurred_at": _utc(t["date"], time(16, 0) if t["source"] == "stop" else US_OPEN),
    } for t in new_trades if "status" in t]
    positions = [{"symbol": s, "side": "LONG", "quantity": int(h["qty"]), "entry_price": float(h["avg_price"]),
                  "stop_loss": float(h.get("stop") or 0.0), "target_price": 0.0, "opened_at": h["entry_date"],
                  "is_open": True} for s, h in book.holdings.items()]
    positions += [{"symbol": t["symbol"], "side": "LONG", "quantity": int(t["qty"]), "entry_price": float(t["avg_price"]),
                   "stop_loss": 0.0, "target_price": 0.0, "opened_at": t["entry_date"], "closed_at": t["date"],
                   "exit_price": float(t["price"]), "exit_reason": t["reason"], "pnl": float(t["pnl"]),
                   "pnl_pct": round((t["price"] / t["avg_price"] - 1) * 100, 3) if t["avg_price"] else 0.0,
                   "is_open": False} for t in new_trades if t.get("closed") and t.get("status") == "FILLED"]
    curve = pd.Series({d: e for d, e in book.equity}, dtype="float64")
    curve.index = pd.DatetimeIndex(curve.index)
    snapshots = []
    for d in new_sessions:
        day = pd.Timestamp(d)
        path = curve[curve.index <= day]
        prev = float(path.iloc[-2]) if len(path) > 1 else NOTIONAL_USD
        done = [t for t in new_trades if t["date"] == d and t.get("status") == "FILLED"]
        snapshots.append({
            "date": d, "equity": round(float(path.iloc[-1]), 2), "cash": round(book.cash, 2),
            "open_positions": len(book.holdings), "closed_today": sum(1 for t in done if t.get("closed")),
            "day_pnl": round(float(path.iloc[-1]) - prev, 2),
            "cumulative_pnl": round(float(path.iloc[-1]) - NOTIONAL_USD, 2),
            "cumulative_pnl_pct": round((float(path.iloc[-1]) / NOTIONAL_USD - 1) * 100, 3),
            "max_drawdown_pct": round(float((1 - path / path.cummax()).max()) * 100, 2),
            "signals_generated": len(plan.orders) if d == as_of else 0, "signals_traded": len(done),
            "snapshot_json": json.dumps({"book": "us_advisory", "currency": "USD", "notional": NOTIONAL_USD,
                                         "cash_and_positions_as_of": as_of,
                                         "positions": {s: h["qty"] for s, h in book.holdings.items()}})})
    t = plan.target
    session = {
        "session_date": as_of, "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "equity": round(book.value(), 2), "cash": round(book.cash, 2), "open_positions": len(book.holdings),
        "rebalance_day": "rebalance_day" in (t.notes or []), "planned_buys": len(plan.side("BUY")),
        "planned_sells": len(plan.side("SELL")), "queued": len(plan.orders),
        "filled": sum(1 for x in fills if x["status"] == "FILLED"),
        "cancelled": sum(1 for x in fills if x["status"] == "CANCELLED"),
        "stops_triggered": sum(1 for x in fills if x["source"] == "stop"),
        "stops_armed": sum(1 for h in book.holdings.values() if h.get("stop")), "skipped": len(plan.skipped),
        "shift_multiplier": 1.0,
        "outcome": (f"{_count(len(plan.side('BUY')), 'buy')} and {_count(len(plan.side('SELL')), 'sell')} suggested; "
                    f"regime {t.regime}")[:200],
        "notes": "\n".join(list(events) + plan.skipped + [str(n) for n in (t.notes or [])]),
        "drawdown_state": plan.drawdown.state if plan.drawdown else "normal",
        "drawdown_pct": round(plan.drawdown.drawdown * 100, 2) if plan.drawdown else 0.0,
    }
    signals = [{"symbol": o.symbol, "forecast": float(t.forecasts.get(o.symbol, 0.0)),
                "combined_forecast": float(t.forecasts.get(o.symbol, 0.0)), "action": o.side,
                "entry_price": float(o.price), "stop_loss": float(o.stop or 0.0), "target_price": 0.0,
                "quantity": int(o.qty), "was_traded": False,
                "pipeline_sources": (f"us_advisory v{BOOK_VERSION}: {o.reason}; weight {o.weight:.4f}; "
                                     f"rank {t.ranks.get(o.symbol, '-')}")} for o in plan.orders]
    return {"fills": fills, "positions": positions, "snapshots": snapshots, "session": [session],
            "signals": signals, "as_of": [as_of]}


# ============================================================================
# the email
# ============================================================================


def next_open_ist(as_of: pd.Timestamp) -> datetime:
    """The next weekday's 09:30 New York open in IST (a US holiday shifts it a day)."""
    day = pd.Timestamp(as_of).normalize() + pd.offsets.BDay(1)
    return datetime.combine(day.date(), US_OPEN, tzinfo=NEW_YORK).astimezone(IST)


def _money(x: float) -> str:
    return f"${x:,.0f}"


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" + ("" if n == 1 else "s")


def _pct(x: float, signed: bool = False) -> str:
    return f"{x:+.1%}" if signed else f"{x:.1%}"


def report_tables(book: Book, plan: Plan, panel: UsPanel, capital: float) -> Dict[str, pd.DataFrame]:
    """The email's tables; "Your $" is the model's dollars x capital / notional."""
    k = capital / NOTIONAL_USD
    sectors = panel.data.sectors
    equity = book.value()
    sells = pd.DataFrame([{"Symbol": o.symbol, "Your shares": "all" if o.weight == 0 else f"{o.qty * k:,.4f}",
                           "Your $": _money(o.qty * o.price * k), "Reason": o.reason} for o in plan.side("SELL")])
    buys = pd.DataFrame([{"Symbol": o.symbol, "Sector": sectors.get(o.symbol, ""), "Weight": _pct(o.weight),
                          "Your $": _money(o.qty * o.price * k), "Your shares": f"{o.qty * k:,.4f}",
                          "Last close": f"{o.price:,.2f}", "Stop": f"{o.stop:,.2f}" if o.stop else "",
                          "Stop distance": _pct(o.stop / o.price - 1) if o.stop else ""}
                         for o in plan.side("BUY")])
    raised = pd.DataFrame([{"Symbol": s, "Old stop": f"{old:,.2f}" if old else "", "New stop": f"{new:,.2f}"}
                           for s, old, new in plan.raised])
    held = pd.DataFrame([{"Symbol": s, "Sector": sectors.get(s, ""), "Weight": _pct(h["qty"] * h["last_price"] / equity),
                          "Your $": _money(h["qty"] * h["last_price"] * k), "Your shares": f"{h['qty'] * k:,.4f}",
                          "Since": h["entry_date"], "Entry": f"{h['avg_price']:,.2f}",
                          "Last": f"{h['last_price']:,.2f}", "P&L": _pct(h["last_price"] / h["avg_price"] - 1, True),
                          "Stop": f"{h['stop']:,.2f}" if h.get("stop") else ""}
                         for s, h in sorted(book.holdings.items())])
    return {"sells": sells, "buys": buys, "raised": raised, "held": held}


def market_facts(book: Book, plan: Plan, panel: UsPanel) -> Dict:
    """The numbers both the email and the text report summarise."""
    spx = panel.data.index_close["SPX"].dropna()
    vix = panel.data.index_close["VIX"].dropna()
    start = pd.Timestamp(book.start)
    dd = plan.drawdown
    return {"regime": plan.target.regime, "scale": plan.target.regime_scale, "spx": float(spx.iloc[-1]),
            "spx_vs_ma200": float(spx.iloc[-1] / spx.rolling(200).mean().iloc[-1] - 1), "vix": float(vix.iloc[-1]),
            "start": start, "book_return": book.value() / NOTIONAL_USD - 1,
            "spx_return": float(spx.iloc[-1] / spx[spx.index <= start].iloc[-1] - 1),
            "names": len(book.holdings), "invested": 1 - book.cash / book.value(),
            "dd_state": dd.state if dd else "normal", "dd": dd.drawdown if dd else 0.0}


def summary_lines(book: Book, plan: Plan, panel: UsPanel) -> List[str]:
    f = market_facts(book, plan, panel)
    lines = [
        f"Market at the {plan.as_of:%d %b %Y} close: regime {f['regime']} (scale {f['scale']:.1f}); "
        f"S&P 500 {f['spx']:,.0f}, {f['spx_vs_ma200']:+.1%} vs its 200-day average; VIX {f['vix']:.1f}.",
        f"Model book since {f['start']:%d %b %Y}: {f['book_return']:+.2%} vs S&P 500 price {f['spx_return']:+.2%}; "
        f"{f['names']} names, cash {1 - f['invested']:.0%}"
        + (f"; drawdown rule {f['dd_state']} ({f['dd']:.1%} below its peak)" if plan.drawdown else "") + ".",
    ]
    if panel.missing:
        lines.append(f"Left out today (no data): {', '.join(panel.missing)}.")
    return lines


#: Order reasons as the email words them.
REASONS = {"stop": "stop hit", "forecast_exit": "trend turned down", "rank_exit": "fell below the exit rank",
           "exit": "no longer in the target", "trim": "trim to its target weight", "entry": "new position",
           "add": "add to its target weight"}
INK, MUTED, LINE, PANEL = "#111827", "#6b7280", "#e5e7eb", "#f9fafb"
GREEN, RED, AMBER, NAVY = "#15803d", "#dc2626", "#b45309", "#1a1a2e"


def render_html(book: Book, plan: Plan, panel: UsPanel, capital: float, capital_set: bool,
                events: Sequence[str]) -> str:
    """The suggestions email: inline styles and tables only, so Outlook and phone clients render it."""
    k = capital / NOTIONAL_USD
    f = market_facts(book, plan, panel)
    opens = next_open_ist(plan.as_of)
    sectors = panel.data.sectors
    sells, buys = plan.side("SELL"), plan.side("BUY")
    td = f"padding:7px 8px;border-bottom:1px solid {LINE};font-size:13px;color:{INK};"
    num = td + "text-align:right;white-space:nowrap;"

    def signed(x: float, digits: int = 1) -> str:
        color = GREEN if x >= 0 else RED
        return f"<span style=\"color:{color};font-weight:600;\">{x:+.{digits}%}</span>"

    def pill(text: str, color: str) -> str:
        return (f"<span style=\"background:{color};color:#fff;font-size:11px;font-weight:700;padding:2px 7px;"
                f"border-radius:10px;letter-spacing:.3px;\">{text}</span>")

    def table(headers: Sequence[str], rows: List[List[str]], right: Sequence[int] = ()) -> str:
        head = "".join(f"<th style=\"padding:7px 8px;background:{PANEL};border-bottom:1px solid {LINE};font-size:11px;"
                       f"color:{MUTED};text-transform:uppercase;letter-spacing:.4px;"
                       f"text-align:{'right' if i in right else 'left'};\">{h}</th>" for i, h in enumerate(headers))
        body = "".join("<tr>" + "".join(f"<td style=\"{num if i in right else td}\">{c}</td>" for i, c in enumerate(r))
                       + "</tr>" for r in rows)
        return (f"<table role=\"presentation\" width=\"100%\" cellpadding=\"0\" cellspacing=\"0\" "
                f"style=\"border-collapse:collapse;margin:6px 0 4px;\"><tr>{head}</tr>{body}</table>")

    def section(title: str, inner: str, note: str = "") -> str:
        return (f"<h3 style=\"margin:26px 0 4px;font-size:15px;color:{NAVY};\">{title}</h3>"
                + (f"<div style=\"font-size:12px;color:{MUTED};margin-bottom:4px;\">{note}</div>" if note else "")
                + inner)

    def tile(label: str, value: str, sub: str) -> str:
        return (f"<td width=\"50%\" style=\"padding:6px;\"><div style=\"background:{PANEL};border:1px solid {LINE};"
                f"border-radius:8px;padding:10px 12px;\"><div style=\"font-size:11px;color:{MUTED};text-transform:"
                f"uppercase;letter-spacing:.4px;\">{label}</div><div style=\"font-size:18px;font-weight:700;color:{INK};"
                f"margin:3px 0;\">{value}</div><div style=\"font-size:12px;color:{MUTED};\">{sub}</div></div></td>")

    regime_color = {"risk_on": GREEN, "neutral": AMBER, "risk_off": RED}.get(f["regime"], INK)
    regime = f["regime"].replace("_", "-").capitalize()
    buy_total = sum(o.qty * o.price * k for o in buys)
    tiles = (
        "<table role=\"presentation\" width=\"100%\" cellpadding=\"0\" cellspacing=\"0\"><tr>"
        + tile("Tonight", f"{_count(len(buys), 'buy')} &middot; {_count(len(sells), 'sell')}",
               _count(len(plan.raised), "stop") + " to raise" + (f" &middot; buys {_money(buy_total)}" if buys else ""))
        + tile("Market", f"<span style=\"color:{regime_color};\">{regime}</span>",
               f"S&amp;P 500 {f['spx_vs_ma200']:+.1%} vs 200-day &middot; VIX {f['vix']:.1f}")
        + "</tr><tr>"
        + tile("Model book", signed(f["book_return"], 2),
               f"S&amp;P 500 {f['spx_return']:+.2%} since {f['start']:%d %b}")
        + tile("Invested", f"{f['invested']:.0%}",
               f"{f['names']} holdings &middot; drawdown rule {f['dd_state']}")
        + "</tr></table>")

    orders = ""
    if sells:
        orders += table(["", "Stock", "Your shares", "Your $", "Why"], [
            [pill("SELL", RED), f"<b>{escape(o.symbol)}</b>", "all" if o.weight == 0 else f"{o.qty * k:,.4f}",
             _money(o.qty * o.price * k), escape(REASONS.get(o.reason, o.reason))] for o in sells], right=(2, 3))
    if buys:
        rows = [[pill("BUY", GREEN),
                 f"<b>{escape(o.symbol)}</b><br><span style=\"font-size:11px;color:{MUTED};\">"
                 f"{escape(sectors.get(o.symbol, ''))}</span>",
                 _pct(o.weight), _money(o.qty * o.price * k), f"{o.qty * k:,.4f}", f"{o.price:,.2f}",
                 (f"{o.stop:,.2f}<br><span style=\"font-size:11px;color:{MUTED};\">{o.stop / o.price - 1:+.1%}</span>"
                  if o.stop else "")] for o in buys]
        rows.append(["", "<b>Total</b>", f"<b>{sum(o.weight for o in buys):.1%}</b>", f"<b>{_money(buy_total)}</b>",
                     "", "", ""])
        orders += table(["", "Stock", "Weight", "Your $", "Your shares", "Last close", "Stop"], rows,
                        right=(2, 3, 4, 5, 6))
    if not orders:
        orders = (f"<p style=\"font-size:13px;color:{MUTED};margin:6px 0;\">No orders tonight: hold, and keep the "
                  "stops in place.</p>")

    raised = table(["Stock", "Old stop", "New stop"], [
        [f"<b>{escape(s)}</b>", f"{old:,.2f}" if old else "&mdash;", f"<b>{new:,.2f}</b>"]
        for s, old, new in plan.raised], right=(1, 2)) if plan.raised else ""

    equity = book.value()
    held = table(["Stock", "Weight", "Your $", "Since", "Entry &rarr; last", "P&amp;L", "Stop"], [
        [f"<b>{escape(s)}</b><br><span style=\"font-size:11px;color:{MUTED};\">{escape(sectors.get(s, ''))}</span>",
         _pct(h["qty"] * h["last_price"] / equity), _money(h["qty"] * h["last_price"] * k),
         pd.Timestamp(h["entry_date"]).strftime("%d %b"), f"{h['avg_price']:,.2f} &rarr; {h['last_price']:,.2f}",
         signed(h["last_price"] / h["avg_price"] - 1), f"{h['stop']:,.2f}" if h.get("stop") else "&mdash;"]
        for s, h in sorted(book.holdings.items(), key=lambda kv: -kv[1]["qty"] * kv[1]["last_price"])],
        right=(1, 2, 4, 5, 6)) if book.holdings else (
        f"<p style=\"font-size:13px;color:{MUTED};margin:6px 0;\">None yet: the first buys fill at this open.</p>")

    notes = list(events) + plan.skipped + ([f"Left out today (no data): {', '.join(panel.missing)}."]
                                           if panel.missing else [])
    notes_html = (f"<div style=\"background:{PANEL};border:1px solid {LINE};border-radius:8px;padding:8px 14px;"
                  f"font-size:13px;color:{INK};\">" + "<br>".join(escape(n) for n in notes) + "</div>") if notes else ""
    size = (f"Sized for your capital of {_money(capital)}." if capital_set else
            f"Sized per {_money(capital)} of capital (set {ENV_CAPITAL} to size for yours).")
    steps = [
        f"At the open, {opens:%H:%M} IST: <b>sells first</b>, then buys of the dollar amounts (fractional shares), "
        "as market orders or limits a little above the last close.",
        "Give every holding a <b>good-till-cancelled sell-stop</b> at its stop; raise stops when listed, never lower "
        "them.",
        f"{escape(size)} The model assumes each order filled at the open; tell Claude if one did not.",
        "Tax: gains on shares held under 24 months are taxed at your slab rate; US dividends lose 25% to "
        "withholding (credited in India).",
    ]
    how = (f"<ol style=\"margin:6px 0 0 18px;padding:0;font-size:13px;color:{INK};line-height:1.5;\">"
           + "".join(f"<li style=\"margin-bottom:4px;\">{s}</li>" for s in steps) + "</ol>")

    return "".join([
        f"<html><body style=\"margin:0;padding:20px 10px;background:#eef0f4;font-family:Segoe UI,Arial,sans-serif;\">",
        f"<div style=\"max-width:680px;margin:0 auto;background:#fff;border-radius:10px;overflow:hidden;"
        f"border:1px solid {LINE};\">",
        f"<div style=\"background:{NAVY};padding:18px 24px;\">"
        f"<div style=\"font-size:12px;color:#a5b4fc;letter-spacing:.5px;text-transform:uppercase;\">Centurion &middot; "
        f"US stocks</div><div style=\"font-size:20px;font-weight:700;color:#fff;margin-top:2px;\">Suggestions for the "
        f"{opens:%a %d %b} open</div><div style=\"font-size:13px;color:#c7d2fe;margin-top:4px;\">Place at "
        f"{opens:%H:%M} IST &middot; decided after the {plan.as_of:%d %b} US close</div></div>",
        f"<div style=\"background:#fff7ed;border-left:4px solid #f59e0b;padding:10px 20px;font-size:13px;"
        f"color:#7c2d12;\"><b>Unvalidated.</b> The India book's deployed rules on today's S&amp;P 500; no US backtest "
        f"or gate has passed yet. Advisory only: you place every order.</div>",
        f"<div style=\"padding:14px 18px 22px;\">", tiles,
        section("Tonight's orders", orders, "Sells first, then buys." if sells and buys else ""),
        section("Raise these stops", raised) if raised else "",
        section("Holdings", held, f"Model book at the {plan.as_of:%d %b} close, largest first."),
        section("Since the last email", notes_html) if notes_html else "",
        section("How to place", how),
        "</div>",
        f"<div style=\"background:{PANEL};border-top:1px solid {LINE};padding:10px 20px;font-size:11px;color:#9ca3af;"
        f"text-align:center;\">data {escape(panel.data.data_hash)} &middot; {panel.members} members &middot; universe "
        f"{plan.target.universe_size} &middot; {datetime.now(IST):%d %b %Y %H:%M} IST &middot; never an automated "
        f"order</div>",
        "</div></body></html>",
    ])


def render_text(book: Book, plan: Plan, panel: UsPanel, capital: float, events: Sequence[str]) -> str:
    tables = report_tables(book, plan, panel, capital)
    opens = next_open_ist(plan.as_of)
    out = [f"US suggestions for the {opens:%a %d %b} open ({opens:%H:%M} IST); UNVALIDATED, advisory only",
           *summary_lines(book, plan, panel)]
    for name, title in (("sells", "Sell"), ("buys", "Buy"), ("raised", "Raise stops"), ("held", "Model holdings")):
        df = tables[name]
        out += ["", f"{title} ({len(df)})", df.to_string(index=False) if len(df) else "None."]
    if list(events) + plan.skipped:
        out += ["", "Notes:", *[f"- {x}" for x in list(events) + plan.skipped]]
    return "\n".join(out)


def send(html: str, subject: str) -> bool:
    from services.notifications.manager import NotificationManager

    return bool(NotificationManager._send_html_email(subject, html))


# ============================================================================
# CLI
# ============================================================================


def _capital(arg: Optional[float]) -> Tuple[float, bool]:
    raw = arg if arg is not None else (os.getenv(ENV_CAPITAL) or "").strip()
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return REFERENCE_CAPITAL, False
    return (value, True) if value > 0 else (REFERENCE_CAPITAL, False)


def run(store, dry_run: bool = False, email: bool = True, capital: Optional[float] = None,
        quiet: bool = False) -> int:
    """One advisory session; 0 when planned (or already done), 1 when the email failed."""
    cfg, rule = us_config()
    book = store.load()
    panel = load_us_data(extra=list(book.holdings) if book else ())
    last = panel.data.dates[-1].date().isoformat()
    if book is not None and book.last_session >= last:
        if book.emailed == book.last_session or dry_run or not email:
            logger.info("US advisory: the %s session is already planned", book.last_session)
            return 0
        logger.info("US advisory: resending the %s suggestions", book.last_session)
        ok = send(book.report_html, f"[Centurion US] Suggestions after the {book.last_session} close (resent)")
        if ok:
            book.emailed = book.last_session
            store.save(book)
        return 0 if ok else 1
    if book is None:
        book = Book(start=last, last_session=last, cash=NOTIONAL_USD, equity=[[last, NOTIONAL_USD]])
        events: List[str] = [f"Model book started on the {last} close."]
        n_trades, n_marks = 0, 0
    else:
        n_trades, n_marks = len(book.trades), len(book.equity)
        events = advance(book, panel)
    plan = plan_next(book, panel, cfg, rule)
    cap, cap_set = _capital(capital)
    book.report_html = render_html(book, plan, panel, cap, cap_set, events)
    logger.info("US advisory %s: %d sells, %d buys, %d stops raised, %d holdings (names by email)",
                last, len(plan.side("SELL")), len(plan.side("BUY")), len(plan.raised), len(book.holdings))
    if not quiet:
        print(render_text(book, plan, panel, cap, events))
    if dry_run:
        return 0
    store.save(book)                       # first: a failed email is resent from the saved book
    failed = store.journal(journal_rows(book, plan, book.trades[n_trades:], [d for d, _ in book.equity[n_marks:]],
                                        events))
    if failed:                             # best effort: the book is saved; upserts make a rerun safe
        logger.warning("US advisory journal not written: %s", ", ".join(failed))
    if not email:
        return 0
    opens = next_open_ist(plan.as_of)
    subject = (f"[Centurion US] {opens:%a %d %b} open: {_count(len(plan.side('BUY')), 'buy')}, "
               f"{_count(len(plan.side('SELL')), 'sell')} (unvalidated)")
    if not send(book.report_html, subject):
        logger.error("US advisory email not sent; the next run resends it")
        return 1
    book.emailed = book.last_session
    store.save(book)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m us_engine.advisory", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="advance the model book and email the next session's suggestions")
    r.add_argument("--state-file", type=Path, help="keep the book in this JSON file instead of Neon")
    r.add_argument("--dry-run", action="store_true", help="print the suggestions; save and email nothing")
    r.add_argument("--no-email", action="store_true", help="save the book without emailing")
    r.add_argument("--capital", type=float, help=f"your US capital in USD (default: ${ENV_CAPITAL})")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    in_actions = os.getenv("GITHUB_ACTIONS") == "true"      # public logs: counts only
    store = FileStore(args.state_file) if args.state_file else NeonStore()
    try:
        return run(store, dry_run=args.dry_run, email=not args.no_email, capital=args.capital, quiet=in_actions)
    except Exception as exc:
        if not args.dry_run:
            from services.notifications.alerts import WARNING, alert

            alert(WARNING, "us_advisory_failed", "US advisory run failed", [f"{type(exc).__name__}: {exc}"])
        raise


if __name__ == "__main__":
    sys.exit(main())
