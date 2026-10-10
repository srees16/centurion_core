"""
Live session driver (tracker L5): one end-of-day session of the live book.

The live book is recorded in its own Neon schema (``live`` by default, env
``CENTURION_LIVE_SCHEMA``) with the same tables as the paper books, so the
drawdown rule, the paper gate (G4) and later the monitor read it exactly as
they read paper.

**The book is a ledger, not the whole account.**  A Zerodha account usually
holds other investments too.  The planner sells every holding it has no
target for, and GTT reconciliation touches every holding, so the engine must
never see the whole account.  The ledger (state ``live_ledger``) holds the
allocated capital (``--capital`` / ``CENTURION_LIVE_CAPITAL`` on the first
session; the D3 ladder changes it later), the book's cash, and the quantity
it bought of each symbol.  The planner sees only ledger symbols, at
``min(ledger, broker)`` quantity, and ``min(ledger cash, broker cash)``; stops
are reconciled only for ledger symbols.  Holding an engine symbol personally
as well is not supported: the broker cannot tell whose shares are whose.

A session, run after the NSE bhavcopy is in the store:

1. market data up to the latest store session ``S`` (same store and anchor
   as paper);
2. order outcomes: the engine's orders decided at the previous live session
   (tags ``NE<yymmdd>``) and filled at ``S``'s open, read from the broker's
   order book, become ``paper_fills`` rows: ``impact_bps`` against the session
   open (positive = cost) and ``costs_inr`` = that slippage plus statutory
   charges (``nse_engine.costs``), so G4's cost check reads live fills the
   way it reads paper ones.  Completed sells of ledger symbols without an
   engine tag (a triggered GTT stop, or a manual sale) are recorded as
   ``external`` and put the symbol on the stop cooldown.  The fills update
   the ledger, once per session;
3. snapshot: the ledger marked at ``S``'s close as a ``paper_daily_snapshots``
   row;
4. plan from ``S``'s close, with the deployment's drawdown rule replayed on
   the live snapshots;
5. execute: LIMIT CNC after-market orders with idempotent tags and scoped
   GTT stop reconciliation, or with ``dry_run`` the same orders built and
   none sent;
6. record the session (``paper_sessions``) and the orders just placed (state
   ``live_orders``, read by step 2 tomorrow), and send the daily email.

Real orders need ALL of: ``CENTURION_PAPER_TRADE=false``,
``CENTURION_NSE_ENGINE_LIVE=true``, an approved deployment (not a placeholder
or candidate) and a Kite session; otherwise the session refuses to start.
It never falls back to the paper book.  A dry run needs only a Kite session
(read-only calls); by default it records nothing (``--record`` to keep a
rehearsal book in Neon).

    python -m kite_connect.trading.live_session --dry-run --capital 600000
    python -m kite_connect.trading.live_session --capital 600000     # first real session
    python -m kite_connect.trading.live_session                       # later sessions

A connected account (trackers FA2, MU1, ``kite_connect.auth.accounts``) runs
the same session in its own book (schema ``live_<id>``) with its own Kite app,
token, capital and mode: ``--stored-token --account <id>``, and none while
its mode is locked (``accounts.trading_lock``: a dry run needs the holder's
terms, live orders also Centurion's registration).  While it is being
disconnected (FA3) its sessions only sell the ledger's positions
(:func:`unwind_orders`), then it turns off.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

ENV_LIVE_SCHEMA = "CENTURION_LIVE_SCHEMA"
ENV_LIVE_CAPITAL = "CENTURION_LIVE_CAPITAL"
DEFAULT_LIVE_SCHEMA = "live"
LIVE_LEDGER_KEY = "live_ledger"            # {"capital", "cash", "positions": {sym: qty}, "entries", "stops", "stop_basis", "symbols", "session"}
LIVE_ORDERS_KEY = "live_orders"            # orders placed at the last live session (JSON)
LIVE_ORDERS_PENDING_KEY = "live_orders_pending"   # placed orders the order book has not yet accounted for (LN-T6)
CA_CREDIT_SESSIONS = 5                     # sessions to wait for split/bonus shares at the broker before an alert
STOP_COOLDOWN_KEEP_DAYS = 30               # recent stop-outs kept in the ledger; the engine counts its 5 sessions
LIVE_LAST_SESSION_KEY = "live_last_session"
LIVE_DRY_LAST_SESSION_KEY = "live_dry_run_last_session"   # the scheduled dry run's once-per-session marker
LIVE_SKIP_NOTIFIED_KEY = "live_skip_notified"            # date of the last "no login today" email
SOURCE_ENGINE, SOURCE_EXTERNAL = "live_open", "external"


def live_schema() -> str:
    return (os.environ.get(ENV_LIVE_SCHEMA) or DEFAULT_LIVE_SCHEMA).strip()


def live_book(schema: Optional[str] = None):
    """The live book's Neon store (tables created on first use)."""
    from database.connection import get_db_manager
    from database.paper_cloud import PaperCloudSync

    if not (os.getenv("CENTURION_DATABASE_URL") or os.getenv("DATABASE_URL")):
        raise RuntimeError("CENTURION_DATABASE_URL is not set: the live book is recorded in Neon")
    book = PaperCloudSync(get_db_manager(), schema=schema or live_schema())
    book.ensure_tables()
    return book


# ── the ledger ──────────────────────────────────────────────────

def load_ledger(state: Dict[str, str], capital: Optional[float]) -> dict:
    """The live book's ledger; created from ``capital`` on the first session."""
    raw = state.get(LIVE_LEDGER_KEY)
    if raw:
        return json.loads(raw)
    if not capital or capital <= 0:
        raise RuntimeError(f"first live session: give the book's capital (--capital or {ENV_LIVE_CAPITAL}); "
                           "month 1 of the ladder is Rs 6,00,000")
    return {"capital": float(capital), "cash": float(capital), "positions": {}, "entries": {}, "stops": {},
            "stop_basis": {}, "symbols": [], "session": ""}


def apply_fills(ledger: dict, fills: List[dict], session, dp_charge_inr: float,
                planned_stops: Optional[Dict[str, Tuple[float, Optional[list]]]] = None, once: bool = True) -> dict:
    """The ledger after ``fills`` (engine and external), applied once per session.

    Keeps what the backtest keeps on each Holding, so live stops follow the
    same rule (tracker LS1): ``entries`` holds the session of a position's first
    BUY fill (kept on adds and partial sells, dropped at zero), and ``stops``
    the book's last stop, seeded for a new position from ``planned_stops``, the
    stop planned with its BUY order the night before: ``{symbol: (stop, basis)}``.
    ``stop_basis`` keeps each stop's ``[date, close]``, the close it was set
    against, so ``rescale_stops`` can follow the data's corporate-action
    adjustments.  ``once=False`` books a correction (``--reconcile``) without
    the once-per-session guard and leaves the ledger's session as it is.
    """
    from nse_engine.costs import statutory_cost

    day = pd.Timestamp(session).date().isoformat()
    if once and str(ledger.get("session") or "") >= day:
        return ledger                                   # this session's fills are already in
    led = {**ledger, "positions": dict(ledger.get("positions") or {}), "symbols": list(ledger.get("symbols") or []),
           "entries": dict(ledger.get("entries") or {}), "stops": dict(ledger.get("stops") or {}),
           "stop_basis": dict(ledger.get("stop_basis") or {}), "recent_stops": dict(ledger.get("recent_stops") or {})}
    for f in fills:
        if f.get("status") != "FILLED" or int(f.get("quantity") or 0) <= 0:
            continue
        sym, side, qty, px = str(f["symbol"]), str(f["side"]).upper(), int(f["quantity"]), float(f["fill_price"])
        have = int(led["positions"].get(sym, 0))
        if side == "SELL":
            qty = min(qty, have)
            if qty <= 0:
                continue
            # the backtest's stop cooldown (LN-T9): a GTT or manual sale on its session, the engine's stop
            # exit on the session the stop was hit (its decision date)
            if f.get("source") == SOURCE_EXTERNAL:
                led["recent_stops"][sym] = str(f.get("session_date") or day)[:10]
            elif str(f.get("reason") or "") == "exit:stop" and f.get("decision_date"):
                led["recent_stops"][sym] = str(f["decision_date"])[:10]
            led["positions"][sym] = have - qty
            led["cash"] += qty * px - statutory_cost(qty * px, "SELL", pd.Timestamp(session), dp_charge_inr, symbol=sym)
        else:
            if have <= 0:                               # a new position: its entry, and its planned stop
                led["entries"][sym] = day
                stop, basis = (planned_stops or {}).get(sym) or (None, None)
                led["stops"].pop(sym, None)
                led["stop_basis"].pop(sym, None)
                if stop:
                    led["stops"][sym] = float(stop)
                    if basis and basis[1]:
                        led["stop_basis"][sym] = [str(basis[0])[:10], float(basis[1])]
            led["positions"][sym] = have + qty
            led["cash"] -= qty * px + statutory_cost(qty * px, "BUY", pd.Timestamp(session), dp_charge_inr, symbol=sym)
            if sym not in led["symbols"]:
                led["symbols"].append(sym)
    led["positions"] = {s: q for s, q in led["positions"].items() if q > 0}
    led["entries"] = {s: d for s, d in led["entries"].items() if s in led["positions"]}
    led["stops"] = {s: v for s, v in led["stops"].items() if s in led["positions"]}
    led["stop_basis"] = {s: v for s, v in led["stop_basis"].items() if s in led["stops"]}
    horizon = (pd.Timestamp(day) - pd.Timedelta(days=STOP_COOLDOWN_KEEP_DAYS)).date().isoformat()
    led["recent_stops"] = {s: d for s, d in led["recent_stops"].items() if d >= horizon}
    if once:
        led["session"] = day
    return led


def apply_ledger_events(ledger: dict, data, session) -> Tuple[List[str], float, List[dict]]:
    """Splits, bonuses and dividends since the ledger's last session (tracker LN-T4): (notes, dividend
    income, event rows for ``paper_fills``).

    A share-count change rebases the ledger's quantity once per event (the
    broker credits bonus and split shares a few sessions later, so
    ``ca_pending`` keeps the position whole meanwhile: ``scoped_book``).
    Stops follow through ``rescale_stops``.  Dividends go to the bank
    account, not the trading account, so the ledger's cash does not move: the
    income is returned for the snapshot to book as an equal withdrawal, so
    the ex-date drop is not counted as a loss.
    """
    from kite_connect.trading.book_events import events_between, rebase_quantity

    last, positions = ledger.get("session"), ledger.get("positions") or {}
    if not last or not positions:
        return [], 0.0, []
    applied = list(ledger.get("ca_applied") or [])
    day = pd.Timestamp(session).date().isoformat()
    notes, income, rows = [], 0.0, []

    def row(sym, side, qty_before, qty, ref, px, note, key):
        return {"order_id": key[:60], "session_date": day, "decision_date": "", "source": "corporate_action",
                "symbol": sym, "side": side, "status": "FILLED", "requested_qty": int(qty_before),
                "quantity": int(qty), "ref_price": float(ref), "fill_price": float(px), "impact_bps": 0.0,
                "costs_inr": 0.0, "note": note[:200], "occurred_at": f"{day}T09:15:00+05:30"}

    for sym, e in sorted(events_between(data, positions, last, session).items()):
        q = int(positions.get(sym, 0))
        for when, per_share, key in e["dividends"]:
            if key in applied or q <= 0:
                continue
            income += q * per_share
            rows.append(row(sym, "", q, q, per_share, per_share, f"dividend Rs {per_share:g} x {q}, paid to the "
                            "bank account", key))
            applied.append(key)
        key = "|".join(e["share_keys"])
        if key and key not in applied:
            new, _ = rebase_quantity(q, e["share_factor"])
            positions[sym] = new
            ledger.setdefault("ca_pending", {})[sym] = {"ex_date": day, "quantity": new}
            rows.append(row(sym, "", q, new, e["share_factor"], 0.0, f"split/bonus factor {e['share_factor']:.4g}: "
                            f"{q} -> {new} shares", key))
            applied.append(key)
            notes.append(f"corporate action {sym}: factor {e['share_factor']:.4g}, {q} -> {new} shares"
                         + (" (INFERRED by the store: check)" if e["inferred"] else ""))
        if e["other"]:
            notes.append(f"corporate action {sym} with no share change (rights/demerger/price): " + "; ".join(e["other"]))
    ledger["positions"] = {s: v for s, v in positions.items() if v > 0}
    ledger["ca_applied"] = applied[-500:]
    return notes, income, rows


def rekey_ledger(ledger: dict, renames: Dict[str, str]) -> List[str]:
    """Move positions, entries, stops and their bases from an old symbol to its new one, once (LN-T5).

    The old name stays in ``symbols``, so the stop GTT on the old instrument
    stays in the reconcile's scope and is deleted once the new one is armed.
    """
    notes = []
    positions = ledger.get("positions") or {}
    for old, new in sorted(renames.items()):
        if old == new or old not in positions:
            continue
        if new in positions:
            notes.append(f"{old} is now {new}, but the ledger holds both: not merged, check Kite")
            continue
        for key in ("positions", "entries", "stops", "stop_basis", "ca_pending", "isins"):
            m = ledger.get(key) or {}
            if old in m:
                m[new] = m.pop(old)
                ledger[key] = m
        ledger.setdefault("symbols", [])
        if new not in ledger["symbols"]:
            ledger["symbols"].append(new)
        notes.append(f"{old} is now {new} (NSE rename or series change): the ledger moved to the new name")
    return notes


def rescale_stops(ledger: dict, close: pd.DataFrame) -> Dict[str, float]:
    """Put the ledger's stops on tonight's price scale; returns each moved symbol's factor.

    The store back-adjusts prices for dividends, splits, bonuses and demergers,
    as the backtest's data is adjusted, so a stop set against an older close
    moves with it: factor = the close of the stop's date in ``close`` (tonight's
    forward-filled closes) / the close it was set against.  The basis is moved
    to tonight's scale too, so a second run of the session changes nothing.
    """
    stops, basis = ledger.setdefault("stops", {}), ledger.setdefault("stop_basis", {})
    factors: Dict[str, float] = {}
    for sym, (day, then) in list(basis.items()):
        try:
            now = float(close.at[pd.Timestamp(day), sym])
        except (KeyError, ValueError, TypeError):
            continue
        f = now / float(then) if then and math.isfinite(now) and now > 0 else 1.0
        if sym in stops and abs(f - 1.0) > 1e-6:
            stops[sym] = round(float(stops[sym]) * f, 4)
            basis[sym] = [day, now]
            factors[sym] = f
    return factors


def scoped_book(ledger: dict, broker_holdings: Dict[str, dict], broker_cash: float,
                factors: Optional[Dict[str, float]] = None,
                bought_today: Optional[Dict[str, int]] = None) -> Tuple[Dict[str, dict], float, List[str]]:
    """(holdings, cash, alerts): the ledger's symbols as the broker holds them.

    Each holding carries its entry date and its stop, as on the backtest's
    Holding (LS1).  The stop is never lowered: the higher of the broker's GTT
    trigger and the ledger's last stop (a GTT the broker failed to raise keeps
    the book's level), the planned stop for a position filled today (its GTT
    does not exist yet), and the last stop when the GTTs could not be read or
    a GTT is gone (triggered without a sale, expired or deleted), so the engine
    still exits through it.  ``factors`` (from ``rescale_stops``) put the
    broker's triggers on tonight's price scale, as the ledger's stops already
    are.  A quantity that differs from the broker's raises an alert; if the
    data shows no corporate action for it, the broker's GTT is used, or the
    stop is recomputed when there is none.  ``t1_quantity`` is the part bought
    at today's open (``bought_today``): it settles during tomorrow's session,
    so Zerodha credits its sale only the day after, and the planner does not
    spend those proceeds on tomorrow's buys.
    """
    from kite_connect.trading.gtt_stops import TICK_SIZE

    alerts, holdings = [], {}
    entries, stops = ledger.get("entries") or {}, ledger.get("stops") or {}
    awaiting = ledger.get("ca_pending") or {}
    for sym, q in (ledger.get("positions") or {}).items():
        b = broker_holdings.get(sym) or {}
        bq = int(b.get("quantity") or 0)
        credit = sym in awaiting and bq < q                # LN-T4: split/bonus shares not yet credited
        if credit:
            alerts.append(f"{sym}: awaiting the broker's credit of split/bonus shares (ex {awaiting[sym]['ex_date']}): "
                          f"the broker shows {bq} of {q}")
            bq_eff = q
        else:
            bq_eff = bq
            if bq != q:
                alerts.append(f"{sym}: the ledger holds {q} but the broker {bq} - using {min(q, bq)}; check Kite")
        bq = bq_eff
        if min(q, bq) > 0:
            f = float((factors or {}).get(sym, 1.0))
            stop = None if b.get("stop_price") is None else float(b["stop_price"]) * f
            last = float(stops[sym]) if stops.get(sym) else None
            filled_today = entries.get(sym) == ledger.get("session")
            # the last stop is on tonight's scale unless the quantity moved with no adjustment in the data
            trusted = last is not None and (bq == q or f != 1.0)
            if b.get("stops_unknown"):
                stop = last if trusted else None
                alerts.append(f"{sym}: the broker's stop GTTs could not be read - "
                              + (f"keeping the book's last stop {last:,.2f}" if trusted
                                 else "its stop is recomputed from today's close"))
            elif stop is None and trusted:
                stop = last
                if not filled_today:                     # filled today: its GTT is placed tonight
                    alerts.append(f"{sym}: no stop GTT at the broker (triggered without a sale, expired or "
                                  f"deleted) - keeping the book's last stop {last:,.2f}; check Kite")
            elif stop is None and last is not None:
                alerts.append(f"{sym}: no stop GTT at the broker - its stop is recomputed from today's close")
            elif stop is not None and trusted and last > stop:
                if last - stop > TICK_SIZE / 2 + 1e-9:   # not just the trigger's rounding to the tick
                    alerts.append(f"{sym}: the broker's stop GTT {stop:,.2f} is below the book's last stop "
                                  f"{last:,.2f} - using the book's")
                stop = last                              # never lowered: a GTT the broker did not raise
            if not entries.get(sym):
                alerts.append(f"{sym}: no entry date in the ledger - its stop counts from today")
            holdings[sym] = {"quantity": min(q, bq), "avg_price": b.get("avg_price", 0.0), "stop_price": stop,
                             "entry_date": entries.get(sym),
                             "t1_quantity": min(int((bought_today or {}).get(sym, 0)), min(q, bq))}
    cash = float(ledger.get("cash") or 0.0)
    if broker_cash + 1.0 < cash:
        alerts.append(f"broker cash {broker_cash:,.0f} is below the ledger's {cash:,.0f} - planning with the broker's")
        cash = float(broker_cash)
    return holdings, cash, alerts


# ── step 2: what happened to yesterday's orders ─────────────────

def fills_from_outcomes(outcomes: List[dict], placed: Dict[str, dict], session, decision_date,
                        opens: Dict[str, float], dp_charge_inr: float) -> List[dict]:
    """``paper_fills`` rows for the engine's orders of ``decision_date``."""
    from nse_engine.costs import statutory_cost

    day = pd.Timestamp(session).date().isoformat()
    rows = []
    for o in outcomes:
        if o.get("status") == "unknown":                 # order book unavailable: nothing to record
            continue
        sym, side = str(o.get("symbol") or ""), str(o.get("side") or "").upper()
        filled, px = int(o.get("filled") or 0), float(o.get("average_price") or 0.0)
        spec = placed.get(str(o.get("tag")), {})
        open_px = float(opens.get(sym) or 0.0)
        impact = 0.0
        if filled > 0 and px > 0 and open_px > 0:
            impact = (1.0 if side == "BUY" else -1.0) * (px - open_px) / open_px * 1e4
        value = filled * px
        cost = (statutory_cost(value, side, pd.Timestamp(session), dp_charge_inr, symbol=sym)
                + value * impact / 1e4) if filled else 0.0
        outcome = str(o.get("outcome") or "")
        status = "FILLED" if filled > 0 else ("REJECTED" if outcome == "rejected" else "CANCELLED")
        note = outcome + (f" {filled}/{o.get('quantity')}" if outcome == "partial" else "")
        if o.get("error"):
            note += f": {o['error']}"
        rows.append({"order_id": str(o.get("order_id") or ""), "session_date": day,
                     "decision_date": pd.Timestamp(decision_date).date().isoformat(),
                     "source": SOURCE_ENGINE, "symbol": sym, "side": side, "status": status,
                     "requested_qty": int(o.get("quantity") or 0), "quantity": filled,
                     "ref_price": float(spec.get("ref_price") or 0.0), "fill_price": px,
                     "impact_bps": round(impact, 3), "costs_inr": round(cost, 2), "reason": spec.get("reason"),
                     "note": (note + (f" ({spec['reason']})" if spec.get("reason") else ""))[:200],
                     "occurred_at": f"{day}T09:15:00+05:30"})
    return rows


def external_sells(order_book: List[dict], symbols, session, dp_charge_inr: float) -> List[dict]:
    """Completed CNC sells today of ledger ``symbols`` the engine did not place (GTT stop, manual)."""
    from kite_connect.trading.nse_instruments import to_engine
    from nse_engine.costs import statutory_cost

    day = pd.Timestamp(session).date().isoformat()
    rows = []
    for o in order_book or []:
        sym = to_engine(o.get("tradingsymbol"))         # a GTT sale of SYM-BE belongs to SYM (LN-T5)
        if sym not in symbols or str(o.get("tag") or "").startswith("NE"):
            continue
        if str(o.get("transaction_type") or "").upper() != "SELL" or str(o.get("product") or "CNC").upper() != "CNC":
            continue
        filled = int(o.get("filled_quantity") or 0)
        if str(o.get("status") or "").upper() != "COMPLETE" or filled <= 0:
            continue
        px = float(o.get("average_price") or 0.0)
        rows.append({"order_id": str(o.get("order_id") or ""), "session_date": day, "decision_date": "",
                     "source": SOURCE_EXTERNAL, "symbol": sym, "side": "SELL", "status": "FILLED",
                     "requested_qty": int(o.get("quantity") or filled), "quantity": filled,
                     "ref_price": float(o.get("trigger_price") or 0.0), "fill_price": px, "impact_bps": 0.0,
                     "costs_inr": round(statutory_cost(filled * px, "SELL", pd.Timestamp(session), dp_charge_inr,
                                                       symbol=sym), 2),
                     "note": f"not placed by the engine (tag {o.get('tag') or '-'}): GTT stop or manual",
                     "occurred_at": f"{day}T09:15:00+05:30"})
    return rows


def _next_session(dates, day) -> Optional[pd.Timestamp]:
    later = pd.DatetimeIndex(dates)[pd.DatetimeIndex(dates) > pd.Timestamp(day)]
    return pd.Timestamp(later[0]).normalize() if len(later) else None


def reconcile_missed(ledger: dict, pending: List[dict], broker_qty: Dict[str, int], broker_avg: Dict[str, float],
                     triggered_gtts: List[dict], view, session, dp_charge_inr: float
                     ) -> Tuple[List[dict], Dict[str, str], Dict[str, pd.Timestamp], List[str]]:
    """Explain the broker's quantities after a missed or failed session (tracker LN-T6).

    Kite's order book lasts one day, so the fills of a session nobody ran are
    never read; the ledger would miss them, the next plan would re-send a
    filled BUY (a doubled position) and a sale's cash would never come back.
    For the ledger's symbols and the ``pending`` orders' symbols, broker
    quantity minus ledger quantity (after today's fills) is explained in
    order: a pending BUY adopts an excess (at the broker's average price for
    a new symbol, else the fill session's open capped at the limit), a
    pending SELL a shortfall (the open floored at the limit), a triggered
    stop GTT the rest of a shortfall (an external sale at its trigger, or the
    open on a gap below it, never under its limit).  Returns (fill rows dated
    their fill session, entry dates of adopted positions, stop cooldowns,
    what stays unexplained).
    """
    from nse_engine.costs import statutory_cost

    positions = ledger.get("positions") or {}
    rows, entries, cooldown, unexplained = [], {}, {}, []

    def open_at(day, sym) -> float:
        try:
            return float(view.open.at[day, sym])
        except (KeyError, ValueError, TypeError):
            return float("nan")

    def row(o, sym, side, qty, px, day, source, note):
        return {"order_id": str(o.get("order_id") or o.get("tag") or ""), "session_date": day.date().isoformat(),
                "decision_date": str(o.get("decision_date") or ""), "source": source, "symbol": sym, "side": side,
                "status": "FILLED", "requested_qty": int(o.get("quantity") or qty), "quantity": int(qty),
                "ref_price": float(o.get("ref_price") or o.get("trigger") or 0.0), "fill_price": float(px),
                "impact_bps": 0.0, "costs_inr": round(statutory_cost(qty * px, side, day, dp_charge_inr, symbol=sym), 2),
                "reason": o.get("reason"), "note": note, "occurred_at": f"{day.date().isoformat()}T09:15:00+05:30"}

    for sym in sorted(set(positions) | {str(o.get("symbol")) for o in pending}):
        delta = int(broker_qty.get(sym, 0)) - int(positions.get(sym, 0))
        for o in (o for o in pending if str(o.get("symbol")) == sym):
            day = _next_session(view.dates, o.get("decision_date"))
            side, qty, limit = str(o.get("side")).upper(), int(o.get("quantity") or 0), float(o.get("limit_price") or 0.0)
            if day is None or day > pd.Timestamp(session) or qty <= 0:
                continue
            op = open_at(day, sym)
            if side == "BUY" and delta > 0:
                q = min(delta, qty)
                px = (float(broker_avg[sym]) if sym not in positions and broker_avg.get(sym)
                      else min(op, limit) if limit > 0 and op == op else limit or op)
                rows.append(row(o, sym, "BUY", q, px, day, SOURCE_ENGINE, "adopted after a missed session"))
                entries[sym] = day.date().isoformat()
                delta -= q
            elif side == "SELL" and delta < 0:
                q = min(-delta, qty)
                px = max(op, limit) if op == op else limit
                rows.append(row(o, sym, "SELL", q, px, day, SOURCE_ENGINE, "adopted after a missed session"))
                delta += q
        if delta < 0:
            for g in (g for g in triggered_gtts if g.get("symbol") == sym):
                day = pd.Timestamp(str(g.get("updated_at") or session)[:10]).normalize()
                trig, limit = float(g.get("trigger") or 0.0), float(g.get("limit") or 0.0)
                op = open_at(day, sym)
                px = max(min(trig, op) if op == op else trig, limit)
                q = min(-delta, int(g.get("quantity") or 0))
                if q > 0:
                    rows.append(row(g, sym, "SELL", q, px, day, SOURCE_EXTERNAL,
                                    f"GTT stop {g.get('id')} triggered on a missed session"))
                    cooldown[sym] = day
                    delta += q
                if delta >= 0:
                    break
        if delta < 0:                                     # sold with no record: on cooldown from today (LN-T9)
            cooldown.setdefault(sym, pd.Timestamp(session).normalize())
        if delta != 0:
            unexplained.append(f"{sym}: the ledger holds {positions.get(sym, 0)}, the broker "
                               f"{broker_qty.get(sym, 0)} ({delta:+d} unexplained)")
    return rows, entries, cooldown, unexplained


def manual_reconcile(ledger: dict, items: Dict[str, Tuple[int, float]], session, dp_charge_inr: float
                     ) -> Tuple[dict, List[dict]]:
    """``--reconcile SYM=QTY@PX``: set the ledger's SYM to QTY, booking the difference at PX (LN-T6)."""
    day = pd.Timestamp(session).normalize()
    rows = []
    for sym, (qty, px) in items.items():
        delta = int(qty) - int((ledger.get("positions") or {}).get(sym, 0))
        if delta:
            side = "BUY" if delta > 0 else "SELL"
            rows.append({"order_id": f"manual-{sym}-{day.date()}", "session_date": day.date().isoformat(),
                         "decision_date": "", "source": SOURCE_EXTERNAL, "symbol": sym, "side": side,
                         "status": "FILLED", "requested_qty": abs(delta), "quantity": abs(delta),
                         "ref_price": float(px), "fill_price": float(px), "impact_bps": 0.0, "costs_inr": 0.0,
                         "note": "manual reconcile", "occurred_at": f"{day.date().isoformat()}T15:30:00+05:30"})
    return apply_fills(ledger, rows, day, dp_charge_inr, once=False), rows


# ── step 3: the book at the close ───────────────────────────────

def mark_book(holdings: Dict[str, dict], cash: float, closes: pd.Series) -> dict:
    """Equity at the session close (average price when a symbol has no close)."""
    value, marks = 0.0, {}
    for sym, h in holdings.items():
        qty = int(h.get("quantity") or 0)
        px = closes.get(sym) if sym in closes.index else None
        px = float(px) if px is not None and math.isfinite(float(px)) and float(px) > 0 else float(h.get("avg_price") or 0.0)
        marks[sym] = {"quantity": qty, "close": px, "value": round(qty * px, 2), "stop": h.get("stop_price")}
        value += qty * px
    return {"equity": float(cash) + value, "cash": float(cash), "invested": value, "positions": marks}


def unwind_orders(holdings: Dict[str, dict], closes: pd.Series, sessions_left: int) -> Tuple[list, List[str]]:
    """(sell orders, symbols without a price) for a session of a disconnect (FA3).

    Each held position sells ``ceil(quantity / sessions_left)``, everything
    on the last session, as the engine's exits are placed (LIMIT inside the
    order band below the close).
    """
    from kite_connect.trading.nse_engine_executor import ORDER_LIMIT_BAND_BPS, PlannedOrder, _tick

    orders, unpriced = [], []
    for sym, h in sorted(holdings.items()):
        qty, px = int(h["quantity"]), float(closes.get(sym, 0.0) or 0.0)
        if px <= 0 or math.isnan(px):
            unpriced.append(sym)
            continue
        sell = qty if sessions_left <= 1 else math.ceil(qty / sessions_left)
        orders.append(PlannedOrder(sym, "SELL", sell, px, _tick(px * (1 - ORDER_LIMIT_BAND_BPS / 1e4), "down"),
                                   "exit:disconnect", qty, qty - sell))
    return orders, unpriced


def snapshot_row(session, marked: dict, history: pd.Series, capital: float, closed_today: int) -> dict:
    """A ``paper_daily_snapshots`` row for the live book."""
    day = pd.Timestamp(session).normalize()
    hist = history[history.index < day] if len(history) else history
    prev = float(hist.iloc[-1]) if len(hist) else float(capital)
    eq = float(marked["equity"])
    path = pd.concat([hist, pd.Series([eq], index=pd.DatetimeIndex([day]))])
    max_dd = float((1.0 - path / path.cummax()).max()) if len(path) else 0.0
    return {"date": day.date().isoformat(), "equity": round(eq, 2), "cash": round(marked["cash"], 2),
            "open_positions": len(marked["positions"]), "closed_today": int(closed_today),
            "day_pnl": round(eq - prev, 2), "cumulative_pnl": round(eq - capital, 2),
            "cumulative_pnl_pct": round((eq / capital - 1.0) * 100.0, 3) if capital else 0.0,
            "max_drawdown_pct": round(max_dd * 100.0, 2), "signals_generated": 0, "signals_traded": 0,
            "snapshot_json": json.dumps({"mode": "live", "invested": round(marked["invested"], 2),
                                         "positions": marked["positions"]}, default=str)}


def _series(df: Optional[pd.DataFrame], col: str) -> pd.Series:
    if df is None or df.empty or col not in df.columns:
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([]))
    s = pd.Series(df[col].astype("float64").to_numpy(),
                  index=pd.DatetimeIndex(pd.to_datetime(df["date"].astype(str).str[:10])))
    return s[~s.index.duplicated(keep="last")].sort_index()


# ── the session ─────────────────────────────────────────────────

def run_live_session(kite, *, dry_run: bool = False, capital: Optional[float] = None, as_of=None,
                     book=None, deployment=None, record: Optional[bool] = None, email: bool = True,
                     executor_factory: Optional[Callable] = None, extra_notes: Optional[List[str]] = None,
                     once_per_session: bool = False, paper_book=None,
                     trial_gates: Optional[Dict[str, dict]] = None, requested_capital: Optional[float] = None,
                     label: str = "", unwind_sessions: int = 0) -> dict:
    """One live session end to end (see the module docstring).  Returns a report dict.

    ``requested_capital`` is the ladder request (default ``CENTURION_LIVE_CAPITAL``);
    ``label`` names a connected account in the email; ``unwind_sessions`` > 0 is a
    disconnect (FA3): tonight's orders only sell, over that many sessions.
    """
    from kite_connect.trading.nse_engine_executor import (EngineExecutor, kite_book, live_order_outcomes,
                                                          live_orders_allowed)
    from nse_engine.deployment import load_deployment

    if kite is None:
        raise RuntimeError("no Kite session: a live session needs the broker, even for a dry run")
    record = (not dry_run) if record is None else bool(record)
    dep = deployment or load_deployment()
    if not dry_run:
        allowed, reason = live_orders_allowed()
        if not allowed:
            raise RuntimeError(f"real orders refused: {reason} (use --dry-run to rehearse)")
        ok, why = dep.live_allowed()
        if not ok:
            raise RuntimeError(f"real orders refused: {why}")
    book = book if book is not None else live_book()
    state = book.read_state()
    capital = capital if capital is not None else (float(os.environ[ENV_LIVE_CAPITAL]) if os.environ.get(ENV_LIVE_CAPITAL) else None)
    first_real = not dry_run and not state.get(LIVE_LEDGER_KEY)
    ledger = load_ledger(state, capital)
    go_live_notes: List[str] = []
    if first_real:                                        # D3 go-live rule
        from nse_engine import capital_ladder as cl
        pstate = _paper_state(paper_book)
        try:
            own_gate = json.loads(pstate.get("paper_gate") or "null")
        except ValueError:
            own_gate = None
        pgate, source = cl.go_live_evidence(own_gate, dep.engine.config_hash(),
                                            trial_gates if trial_gates is not None else _trial_gates(dep))
        try:
            user_id = (kite.profile() or {}).get("user_id")
        except Exception:                                 # noqa: BLE001 - an unknown user fails the check
            user_id = None
        sell_path = cl.sell_path_check(json.loads(state.get(cl.SELL_PATH_KEY) or "null"), user_id)
        checks = cl.readiness(pgate, json.loads(state.get(cl.DRY_RUNS_KEY) or "[]"), source=source,
                              sell_path=sell_path)
        missing = [f"{n}: {d}" for n, ok, d in checks if not ok]
        if cl.rung_of(ledger["capital"]) is None:
            raise RuntimeError(f"live capital {ledger['capital']:,.0f} is not a ladder rung "
                               f"{[f'{c:,.0f}' for c in cl.RUNGS]}")
        if missing and os.environ.get("CENTURION_GO_LIVE_OVERRIDE", "").lower() != "true":
            raise RuntimeError("go-live refused: " + "; ".join(missing)
                               + " (CENTURION_GO_LIVE_OVERRIDE=true overrides)")
        go_live_notes = ["GO-LIVE " + ("OVERRIDDEN: " + "; ".join(missing) if missing else "checks passed: "
                                       + "; ".join(d for _, _, d in checks))]

    # the live book's own drift state (never the paper book's file)
    from database.paper_cloud import SHIFT_STATE_KEY
    shift_path = Path(tempfile.gettempdir()) / f"centurion_live_shift_{os.getpid()}.json"
    if state.get(SHIFT_STATE_KEY):
        shift_path.write_text(state[SHIFT_STATE_KEY])
    elif shift_path.exists():
        shift_path.unlink()

    strict = bool(record) and not dry_run                 # LN-T10: a real session fails closed on its book
    raw_equity, flows = _equity_and_flows(_book_read(book, "read_snapshots", strict))
    from nse_engine.capital_ladder import flow_adjusted
    history = flow_adjusted(raw_equity, flows)
    cooldown: Dict[str, pd.Timestamp] = {}
    scope: Dict[str, object] = {"holdings": {}, "cash": 0.0}
    factory = executor_factory or EngineExecutor
    ex = factory(kite=kite, paper=dry_run, deployment=dep, dry_run=dry_run,
                 holdings_fn=lambda: (scope["holdings"], scope["cash"]), stopped_out_fn=lambda: dict(cooldown),
                 equity_history_fn=lambda: history, shift_state_path=str(shift_path),
                 stop_scope_fn=lambda: ({s: h["quantity"] for s, h in scope["holdings"].items()},
                                        list(ledger.get("symbols") or [])))
    if not dry_run and ex.paper:
        raise RuntimeError(f"real orders refused by the executor: {ex.mode_reason}")
    cfg = ex._cfg()
    dp = cfg.costs.dp_charge_inr

    # 1. market data
    end = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(datetime.now(timezone.utc).date())
    data = ex._load(cfg, end)
    view = data.until(end) if hasattr(data, "until") else data
    if not len(view.dates):
        raise RuntimeError(f"no market data up to {end.date()} in {cfg.data.store_dir}")
    session = pd.Timestamp(view.dates[-1]).normalize()
    marker = LIVE_DRY_LAST_SESSION_KEY if dry_run else LIVE_LAST_SESSION_KEY
    if once_per_session and state.get(marker) == session.date().isoformat():
        logger.info("LIVE session %s (%s) already ran: nothing to do", session.date(), "dry run" if dry_run else "live")
        return {"session": session.date().isoformat(), "skipped": "already ran", "alerts": [], "notes": []}
    report: dict = {"session": session.date().isoformat(), "mode": "live dry run" if dry_run else "live",
                    "fills": [], "external": [], "alerts": [], "notes": list(extra_notes or []) + go_live_notes,
                    "results": [], "plan": None}

    # LN-T4: the ledger moves into tonight's units (splits, bonuses) before today's fills, which are in them
    from kite_connect.trading.book_events import events_between, unrecorded_gaps, upcoming_share_actions

    prev_session = ledger.get("session")
    broker_holdings, broker_cash = kite_book(kite)
    # LN-T5: a renamed stock moves to its new name first (the store's change table, then the broker's ISIN)
    from kite_connect.trading.book_events import renamed, upcoming_mergers

    renames = renamed(cfg.data.store_dir, ledger.get("positions") or {}, prev_session, session)
    by_isin = {h.get("isin"): s for s, h in broker_holdings.items() if h.get("isin")}
    for old, isin in (ledger.get("isins") or {}).items():
        new = by_isin.get(isin)
        if new and new != old and old in (ledger.get("positions") or {}) and old not in broker_holdings:
            renames.setdefault(old, new)
    report["alerts"].extend(rekey_ledger(ledger, renames))
    ca_notes, dividend_income, report["events"] = apply_ledger_events(ledger, data, session)
    report["notes"].extend(ca_notes)
    gaps = (unrecorded_gaps(data, ledger.get("positions") or {}, prev_session, session,
                            events_between(data, ledger.get("positions") or {}, prev_session, session))
            if prev_session else [])
    if gaps:
        report["alerts"].append("as-printed close down 10%+ with no corporate action in the store (an unrecorded "
                                "split?), buys of it held back: " + ", ".join(gaps))

    # 2. outcomes of the previous live session's orders, then the ledger
    placed_doc = json.loads(state.get(LIVE_ORDERS_KEY) or "{}")
    decision = placed_doc.get("decision_date")
    pending = list(json.loads(state.get(LIVE_ORDERS_PENDING_KEY) or "[]"))    # LN-T6: carried from earlier nights
    book_unknown = False
    if decision and pd.Timestamp(decision) < session:
        opens_row = view.open.iloc[-1]
        placed = {o["tag"]: o for o in placed_doc.get("orders", []) if o.get("tag")}
        opens = {s: float(opens_row[s]) for s in {o.get("symbol") for o in placed.values()} if s in opens_row.index}
        outcomes = live_order_outcomes(kite, decision)
        if outcomes and outcomes[0].get("status") == "unknown":
            report["alerts"].append(f"order book unavailable: {outcomes[0].get('error')}")
        report["fills"] = fills_from_outcomes(outcomes, placed, session, decision, opens, dp)
        seen = {o.get("tag") for o in outcomes}
        book_unknown = bool(outcomes and outcomes[0].get("status") == "unknown")
        # orders the book does not account for (a missed session's, or an UNKNOWN placement) wait for the
        # broker's quantities to explain them
        pending += [{**o, "decision_date": decision} for t, o in placed.items()
                    if (book_unknown or t not in seen) and o.get("status") in ("PLACED", "UNKNOWN", "INTENDED")]
        intended = [f"{o.get('side')} {o.get('symbol')}" for t, o in placed.items()
                    if o.get("status") == "INTENDED" and t not in seen and not book_unknown]
        if intended:                                      # LN-T10: the run stopped between write-ahead and send
            report["alerts"].append("intended last night but not in the order book: " + ", ".join(intended))
        bad = [f for f in report["fills"] if f["status"] != "FILLED" or f["note"].startswith("partial")]
        if bad:
            report["alerts"].append("not (fully) filled: " + "; ".join(f"{f['side']} {f['symbol']} {f['note']}" for f in bad))
    elif decision:
        report["notes"].append(f"orders decided {decision}: outcomes already read")
    try:
        order_book = kite.orders() or []
    except Exception as exc:                             # noqa: BLE001
        order_book = []
        report["alerts"].append(f"order book unavailable: {exc}")
    report["external"] = external_sells(order_book, set(ledger.get("positions") or {}), session, dp)
    for e in report["external"]:
        cooldown[str(e["symbol"])] = session
    if report["external"]:
        report["alerts"].append("sold outside the engine today (GTT stop or manual): "
                                + ", ".join(f"{e['symbol']} x{e['quantity']}" for e in report["external"]))
    planned_stops = ({o["symbol"]: (o["stop_price"], [decision, o.get("ref_price")]) for o in placed_doc.get("orders", [])
                      if str(o.get("side")).upper() == "BUY" and o.get("stop_price")}
                     if decision and pd.Timestamp(decision) < session else {})
    planned_stops.update({o["symbol"]: (o["stop_price"], [o["decision_date"], o.get("ref_price")]) for o in pending
                          if str(o.get("side")).upper() == "BUY" and o.get("stop_price")})

    # LN-T6: after a missed session (its fills left the order book) or with orders unaccounted for, the
    # broker's quantities decide; what they cannot explain blocks new buys until it is cleared
    fill_day = _next_session(view.dates, decision) if decision else None
    skipped = bool(fill_day is not None and fill_day < session and pd.Timestamp(decision) < session
                   and str(ledger.get("session") or "") < fill_day.date().isoformat())
    block_buys: List[str] = []
    adopted_entries: Dict[str, str] = {}
    if book_unknown:
        block_buys.append("the order book could not be read, so last night's fills are unknown")
    elif pending or skipped:
        provisional = apply_fills(ledger, report["fills"] + report["external"], session, dp)
        since = str(ledger.get("session") or "")
        from kite_connect.trading.gtt_stops import list_stop_gtts

        def sold(g) -> bool:                             # triggered since the last session, its order not refused
            r, when = g.get("order_result"), str((g.get("raw") or {}).get("updated_at") or "")[:10]
            ok = not isinstance(r, dict) or str(r.get("status") or "success").lower() == "success"
            return g.get("status") == "triggered" and ok and since < when <= session.date().isoformat()

        try:
            triggered = [{**g, "updated_at": (g.get("raw") or {}).get("updated_at")}
                         for g in list_stop_gtts(kite, active_only=False) if sold(g)]
        except Exception as exc:                          # noqa: BLE001 - then a shortfall stays unexplained
            triggered = []
            report["alerts"].append(f"GTT history unavailable for the reconcile: {exc}")
        awaiting = ledger.get("ca_pending") or {}
        seen_qty = {s: int(h["quantity"]) for s, h in broker_holdings.items()}
        seen_qty.update({s: max(seen_qty.get(s, 0), int((provisional.get("positions") or {}).get(s, 0)))
                         for s in awaiting})                  # split/bonus shares on their way (LN-T4)
        rows, adopted_entries, gtt_cooldown, unexplained = reconcile_missed(
            provisional, pending, seen_qty,
            {s: float(h.get("avg_price") or 0.0) for s, h in broker_holdings.items()}, triggered, view, session, dp)
        if rows:
            report["fills"] += [r for r in rows if r["source"] == SOURCE_ENGINE]
            report["external"] += [r for r in rows if r["source"] == SOURCE_EXTERNAL]
            cooldown.update(gtt_cooldown)
            report["alerts"].append("reconciled after a missed session: "
                                    + ", ".join(f"{r['side']} {r['symbol']} x{r['quantity']} ({r['session_date']})"
                                                for r in rows))
        if unexplained:
            block_buys.append("RECONCILE NEEDED: " + "; ".join(unexplained)
                              + " (fix: live_session --reconcile SYM=QTY@PX)")
        used = {r["order_id"] for r in rows}
        unfilled = [f"{o.get('side')} {o.get('symbol')}" for o in pending if str(o.get("tag")) not in used]
        if unfilled:
            report["alerts"].append("placed, but neither the order book nor the broker shows a fill: "
                                    + ", ".join(unfilled))
        pending = []
    ledger = apply_fills(ledger, report["fills"] + report["external"], session, dp, planned_stops=planned_stops)
    for sym, day in adopted_entries.items():
        if sym in (ledger.get("positions") or {}):
            ledger.setdefault("entries", {})[sym] = day          # the fill session, as the backtest dates it
    for sym, day in cooldown.items():                     # LN-T9: kept across sessions, as the backtest does
        ledger.setdefault("recent_stops", {})[sym] = max(str(ledger["recent_stops"].get(sym) or ""),
                                                         pd.Timestamp(day).date().isoformat())
    cooldown.update({s: pd.Timestamp(d) for s, d in (ledger.get("recent_stops") or {}).items()})
    report["alerts"].extend(block_buys)
    close_ff = view.close.ffill()
    factors = rescale_stops(ledger, close_ff)            # stops onto tonight's adjusted prices
    if factors:
        report["notes"].append("stops rescaled for corporate actions: "
                               + ", ".join(f"{s} x{f:.4f}" for s, f in sorted(factors.items())))

    # 3. the book at the close (broker_holdings read in step 2)
    bought_today: Dict[str, int] = {}                     # unsettled at tomorrow's open (T+1)
    for f in report["fills"]:
        if f.get("status") == "FILLED" and str(f.get("side")).upper() == "BUY":
            bought_today[str(f["symbol"])] = bought_today.get(str(f["symbol"]), 0) + int(f.get("quantity") or 0)
    holdings, cash, scope_alerts = scoped_book(ledger, broker_holdings, broker_cash, factors, bought_today)
    report["alerts"].extend(scope_alerts)
    ledger["isins"] = {s: v for s, v in {**(ledger.get("isins") or {}),        # LN-T5: to follow a rename by ISIN
                                         **{x: h["isin"] for x, h in broker_holdings.items() if h.get("isin")}}.items()
                       if s in (ledger.get("positions") or {})}
    for sym, w in list((ledger.get("ca_pending") or {}).items()):     # LN-T4: credited, or overdue
        bq = int((broker_holdings.get(sym) or {}).get("quantity") or 0)
        waited = int(((view.dates > pd.Timestamp(w["ex_date"])) & (view.dates <= session)).sum())
        if bq >= int((ledger.get("positions") or {}).get(sym, 0)) or sym not in (ledger.get("positions") or {}):
            ledger["ca_pending"].pop(sym)
        elif waited >= CA_CREDIT_SESSIONS:
            report["alerts"].append(f"{sym}: split/bonus shares still not credited {waited} sessions after the "
                                    f"ex-date {w['ex_date']}: check Kite and Console")
    scope["holdings"], scope["cash"] = holdings, cash
    outside = sorted(set(broker_holdings) - set(ledger.get("positions") or {}))
    if outside:
        report["notes"].append(f"{len(outside)} holding(s) outside the book, left alone: {', '.join(outside[:10])}")
    closes = close_ff.iloc[-1]
    marked = mark_book(holdings, float(ledger["cash"]), closes)
    closed_today = sum(1 for f in report["fills"] + report["external"] if f["side"] == "SELL" and f["status"] == "FILLED")
    prior, prior_flows = raw_equity[raw_equity.index < session], flows[flows.index < session]
    div_flow = -float(dividend_income)        # LN-T4: paid to the bank account, booked as income withdrawn
    if div_flow:
        prior_flows = pd.concat([prior_flows, pd.Series([div_flow], index=pd.DatetimeIndex([session]))])
    history = flow_adjusted(pd.concat([prior, pd.Series([marked["equity"]], index=pd.DatetimeIndex([session]))]),
                            prior_flows)                  # today's equity before any ladder flow
    today_flow = 0.0
    ahead: Dict[str, str] = {}                           # written with the ledger, in one transaction (LN-T10)
    if record and not dry_run and session.date() >= dep.paper_start_date:
        from nse_engine import capital_ladder as cl
        since, marker, replaced = cl.config_since(state.get(cl.LIVE_CONFIG_KEY), dep.engine.config_hash(),
                                                  history.index[0].date().isoformat(), session.date().isoformat())
        if marker:
            ahead[cl.LIVE_CONFIG_KEY] = marker
        if replaced:
            report["notes"].append(f"configuration {replaced[:8]} -> {dep.engine.config_hash()[:8]} from {since}: "
                                   "the live G4 window restarts here (V5)")
        decision, gate, lstate = _ladder_step(ex, data, view, dep, ledger, state, book, session, history, marked,
                                              since, requested_capital, label, strict=strict)
        ahead[cl.STATE_KEY] = lstate.dump()
        report["ladder"] = decision.line()
        report["alerts"].extend(decision.alerts)
        if gate is not None:
            from nse_engine import paper_gate as pg
            report["gate"] = pg.one_line(gate)
            ahead[pg.STATE_KEY] = pg.summary_json(gate, updated_at=datetime.now(timezone.utc).isoformat(),
                                                  config_hash=dep.engine.config_hash())
        if decision.flow:
            today_flow = float(decision.flow)
            ledger["capital"] = float(decision.capital)
            ledger["cash"] = float(ledger["cash"]) + today_flow
            holdings, cash, _ = scoped_book(ledger, broker_holdings, broker_cash, factors, bought_today)
            scope["holdings"], scope["cash"] = holdings, cash
            marked = mark_book(holdings, float(ledger["cash"]), closes)
    all_flows = pd.concat([prior_flows[prior_flows.index < session],
                           pd.Series([today_flow + div_flow], index=pd.DatetimeIndex([session]))])
    history = flow_adjusted(pd.concat([prior, pd.Series([marked["equity"]], index=pd.DatetimeIndex([session]))]),
                            all_flows)                    # in tonight's capital base, flows removed
    snap = snapshot_row(session, marked, history, float(ledger["capital"]), closed_today)
    if today_flow or div_flow:
        js = json.loads(snap["snapshot_json"]); js["flow"] = today_flow + div_flow
        if div_flow:
            js["dividends"] = -div_flow
        snap["snapshot_json"] = json.dumps(js, default=str)
    report["snapshot"], report["ledger"] = snap, ledger
    if record:
        if not state.get("epoch"):                       # the book begins at this session
            start = (session.tz_localize("Asia/Kolkata")).tz_convert("UTC").isoformat()
            book.sync_state({"epoch": start, "book_start": start, "book_owner": "live_engine",
                             "initial_capital": ledger["capital"]})
        _book_write(book.sync_fills(report["fills"] + report["external"] + report.get("events", [])), "fills", strict)
        _book_write(book.sync_snapshot(snap), "snapshot", strict)
        _book_write(book.sync_state({LIVE_LEDGER_KEY: json.dumps(ledger), **ahead}), "ledger and ladder state", strict)

    # 4-5. plan and execute
    plan = None
    if session.date() < dep.paper_start_date:
        report["notes"].append(f"session {session.date()} is before the deployment's start {dep.paper_start_date}: no plan")
    else:
        plan = ex.plan(as_of=session, data=data)
        if unwind_sessions:                              # FA3: sell out; the engine's stops stay on the rest
            plan.orders, unpriced = unwind_orders(holdings, closes, unwind_sessions)
            report["unwound"] = not holdings
            report["notes"].append(
                f"DISCONNECTING: selling {len(plan.orders)} position(s), "
                f"{'all at the next open' if unwind_sessions <= 1 else f'1/{unwind_sessions} of each'}"
                if holdings else "DISCONNECTED: the account holds none of the book's positions; Centurion "
                                 "stops trading it")
            if unpriced:
                report["alerts"].append("disconnect: no price, not sold tonight: " + ", ".join(unpriced))
        from kite_connect.trading.nse_engine_executor import ORDER_LIMIT_BAND_BPS, PlannedOrder, _tick

        carry: Dict[str, int] = {}                       # LN-T14: last night's sells that did not (fully) fill
        for f in report["fills"]:
            if str(f.get("side")).upper() == "SELL" and int(f.get("requested_qty") or 0) > int(f.get("quantity") or 0):
                carry[str(f["symbol"])] = carry.get(str(f["symbol"]), 0) + int(f["requested_qty"]) - int(f["quantity"])
        weights = getattr(getattr(plan, "target", None), "weights", None) or {}
        equity_now = float(getattr(plan, "equity", 0.0) or 0.0)
        for sym, residual in sorted(carry.items()):
            held, px = int((holdings.get(sym) or {}).get("quantity") or 0), float(closes.get(sym, 0.0) or 0.0)
            if sym in {o.symbol for o in plan.orders} or held <= 0 or px <= 0:
                continue                                  # tonight's plan decides it, or nothing is left
            keep = int(math.floor(float(weights.get(sym, 0.0)) * equity_now / px + 1e-6)) if equity_now > 0 else 0
            q = min(residual, held - keep)
            if q > 0:
                plan.orders.insert(0, PlannedOrder(sym, "SELL", q, px, _tick(px * (1 - ORDER_LIMIT_BAND_BPS / 1e4), "down"),
                                                   "exit:carry", held, held - q))
                report["notes"].append(f"{sym}: {q} share(s) of last night's sell did not fill, sent again")
        mergers = upcoming_mergers(cfg.data.store_dir, holdings, session)
        if mergers:                                       # LN-T5: exit before the stock stops trading

            report["alerts"].append("merger/delisting announced, selling the position: "
                                    + "; ".join(f"{s} ({w})" for s, w in sorted(mergers.items())))
            selling = {o.symbol for o in plan.orders if o.side == "SELL"}
            plan.orders = [o for o in plan.orders if not (o.side == "BUY" and o.symbol in mergers)]
            for sym in sorted(set(mergers) - selling):
                px, q = float(closes.get(sym, 0.0) or 0.0), int(holdings[sym]["quantity"])
                if px > 0 and q > 0:
                    plan.orders.insert(0, PlannedOrder(sym, "SELL", q, px, _tick(px * (1 - ORDER_LIMIT_BAND_BPS / 1e4),
                                                                                "down"), "exit:merger", q, 0))
        ex_next = upcoming_share_actions(cfg.data.store_dir, {o.symbol for o in plan.orders}, session)
        if ex_next:                                      # LN-T4: tonight's prices and quantities are stale tomorrow
            report["notes"].append("held back, ex-date at the next open: "
                                   + "; ".join(f"{s} ({w})" for s, w in sorted(ex_next.items())))
            plan.orders = [o for o in plan.orders if o.symbol not in ex_next]
        gap_syms = {g.split(" ")[0] for g in gaps}
        if gap_syms:
            plan.orders = [o for o in plan.orders if not (o.side == "BUY" and o.symbol in gap_syms)]
        for o in plan.orders:                            # LN-T4: sell only what the broker can deliver
            if o.side == "SELL" and o.symbol in (ledger.get("ca_pending") or {}):
                sellable = int((broker_holdings.get(o.symbol) or {}).get("quantity") or 0)
                if o.quantity > sellable:
                    report["notes"].append(f"{o.symbol}: sell cut {o.quantity} -> {sellable} (split/bonus shares "
                                           "not yet credited)")
                    o.quantity = sellable
        plan.orders = [o for o in plan.orders if o.quantity > 0]
        if block_buys and plan.buys:                    # LN-T6: exits only until the ledger balances
            report["notes"].append(f"buys held back: {', '.join(o.symbol for o in plan.buys)}")
            plan.orders = [o for o in plan.orders if o.side == "SELL"]
        report["plan"] = plan
        stale = any(s.get("reason") == "stale_data" for s in plan.skipped)
        if stale:
            report["alerts"].append("stale market data: no orders planned")
        if strict and not stale and plan.orders:         # LN-T10: what is about to be sent is saved first
            from kite_connect.trading.nse_engine_executor import EngineExecutor as _EE

            stops_planned = {x.symbol: x.trigger for x in plan.stop_instructions}
            intended = {"decision_date": session.date().isoformat(), "orders": [
                {"tag": _EE.order_tag(session, o.side, o.symbol), "symbol": o.symbol, "side": o.side,
                 "quantity": int(o.quantity), "reason": getattr(o, "reason", None), "ref_price": float(o.ref_price),
                 "limit_price": float(getattr(o, "limit_price", 0.0) or 0.0),
                 "stop_price": stops_planned.get(o.symbol) if o.side == "BUY" else None, "status": "INTENDED"}
                for o in plan.orders]}
            _book_write(book.sync_state({LIVE_ORDERS_KEY: json.dumps(intended),
                                         LIVE_ORDERS_PENDING_KEY: json.dumps(pending if book_unknown else [])}),
                        "intended orders (nothing sent)", strict)
        report["results"] = ex.dry_run_live(plan) if dry_run else ([] if stale else ex.execute(plan))
    preflight = [r["error"] for r in report["results"] if r.get("type") == "preflight"]
    if preflight:                                        # LN-T12: Kite would not accept everything
        report["alerts"].append("preflight: " + "; ".join(preflight))
    orders = [r for r in report["results"] if r.get("type") not in ("gtt_reconcile", "preflight")]
    series = sorted({r["series_note"] for r in orders if r.get("series_note")})
    if series:
        report["notes"].append("; ".join(series))
    clamps = [r["limit_note"] for r in orders if r.get("limit_note")]
    if clamps:                                           # LN-T14: limits set by the circuit or a fallback
        report["notes"].append("; ".join(clamps))
    refused = [r for r in orders if not r.get("success") and r.get("status") != "DUPLICATE"]
    if refused:
        report["alerts"].append("orders refused: " + "; ".join(
            f"{r.get('side')} {r.get('symbol')}: {r.get('error') or r.get('status')}" for r in refused))
    gtt = [r for r in report["results"] if r.get("type") == "gtt_reconcile"]
    if gtt and not gtt[0].get("success"):
        report["alerts"].append(f"GTT stop reconciliation errors: {(gtt[0].get('report') or {}).get('errors')}")
    off_grid = sorted({str(r["symbol"]) for r in orders if r.get("tick_fallback")}
                      | set((gtt[0].get("report") or {}).get("tick_fallback") or [] if gtt else []))
    if off_grid:                                         # LN-T1: priced on the coarser slab's tick
        report["alerts"].append("Kite's instrument list lacked a tick, priced on the coarser NSE slab: "
                                + ", ".join(off_grid))
    breached = (gtt[0].get("report") or {}).get("breached") if gtt else None
    if breached:                                         # no GTT placed; the book's stop exits it next session
        report["alerts"].append("stop already above the price, no GTT placed: "
                                + ", ".join(f"{b['symbol']} {b['trigger']:,.2f}" for b in breached))
    if plan is not None and plan.drawdown_changed:
        report["alerts"].append(f"DRAWDOWN RULE now {plan.drawdown_state.upper()} at {plan.drawdown_pct:.1f}% below the peak")

    # 6. record the session and today's orders for tomorrow
    if record:
        values = {LIVE_LAST_SESSION_KEY: session.date().isoformat()}
        if not dry_run:
            values[LIVE_ORDERS_PENDING_KEY] = json.dumps(pending if book_unknown else [])
        if plan is not None and not dry_run:
            refs = {o.symbol: o.ref_price for o in plan.orders}
            planned = {s.symbol: s.trigger for s in plan.stop_instructions}
            values[LIVE_ORDERS_KEY] = json.dumps({"decision_date": session.date().isoformat(), "orders": [
                {"tag": r.get("tag"), "symbol": r.get("symbol"), "side": r.get("side"), "quantity": r.get("quantity"),
                 "reason": r.get("reason"),
                 "limit_price": r.get("limit_price"), "ref_price": refs.get(r.get("symbol"), 0.0),
                 "stop_price": planned.get(r.get("symbol")) if str(r.get("side")).upper() == "BUY" else None,
                 "status": "PLACED" if r.get("status") == "DUPLICATE" else r.get("status")} for r in orders]})
            # tonight's stops become the book's last stops (LS1), kept if a GTT is later missing,
            # with tonight's close as their basis for later corporate-action adjustments
            held = {s: v for s, v in planned.items() if s in (ledger.get("positions") or {})}
            ledger["stops"] = {**(ledger.get("stops") or {}), **held}
            basis = dict(ledger.get("stop_basis") or {})
            for s in held:
                c = closes.get(s)
                if c is not None and pd.notna(c) and float(c) > 0:
                    basis[s] = [session.date().isoformat(), float(c)]
                else:
                    basis.pop(s, None)
            ledger["stop_basis"] = basis
            values[LIVE_LEDGER_KEY] = json.dumps(ledger)
        _book_write(book.sync_state(values), "the session's orders and ledger (orders were sent)", strict)
        _book_write(book.sync_session(_session_row(report, plan, marked, snap, orders)), "session record", strict)
    if once_per_session and dry_run:                     # the marker and the log only; a dry run records no book
        from nse_engine.capital_ladder import DRY_RUNS_KEY
        log = [r for r in json.loads(state.get(DRY_RUNS_KEY) or "[]") if r.get("session") != session.date().isoformat()]
        log.append({"session": session.date().isoformat(), "orders": len(orders), "alerts": len(report["alerts"]),
                    "clean": not report["alerts"]})
        book.sync_state({LIVE_DRY_LAST_SESSION_KEY: session.date().isoformat(), DRY_RUNS_KEY: json.dumps(log[-30:])})
    if shift_path.exists():
        shift_path.unlink()
    if email:
        _email(report, dep, snap, float(ledger["capital"]), getattr(book, "schema", None) or live_schema(), label)
    logger.info("LIVE session %s (%s): %d fills, %d external, %d orders %s, %d alert(s) (detail in the email)",
                report["session"], report["mode"], len(report["fills"]), len(report["external"]), len(orders),
                "built" if dry_run else "sent", len(report["alerts"]))
    return report


def record_sell_path(kite, book, ddpi_confirmed_on: str, amo_order_id: str, gtt_id) -> dict:
    """Verify the supervised sell test against Kite and keep it in the book (tracker LN-T2).

    The test, run once by hand: DDPI confirmed in Console, then two shares of
    a cheap liquid name outside the book sold by an AMO SELL (CNC) and by a
    single-leg SELL GTT with its trigger near the price.  Kite is read for
    both, so the record holds what the broker says, not what was typed.
    """
    from nse_engine.capital_ladder import SELL_PATH_KEY

    amo = (kite.order_history(amo_order_id) or [{}])[-1]
    g = kite.get_gtt(gtt_id) or {}
    result = (((g.get("orders") or [{}])[0].get("result") or {}).get("order_result") or {})
    gtt_order_id = result.get("order_id")
    gtt_order = (kite.order_history(gtt_order_id) or [{}])[-1] if gtt_order_id else {}
    rec = {"ddpi_confirmed_on": str(ddpi_confirmed_on)[:10], "user_id": (kite.profile() or {}).get("user_id"),
           "amo_order_id": str(amo_order_id), "amo_side": str(amo.get("transaction_type") or "").upper(),
           "amo_product": amo.get("product"), "amo_status": str(amo.get("status") or "").upper(),
           "amo_status_message": amo.get("status_message"), "gtt_id": str(gtt_id),
           "gtt_status": str(g.get("status") or "").lower(), "gtt_order_id": gtt_order_id,
           "gtt_order_status": str(gtt_order.get("status") or "").upper(),
           "gtt_rejection": result.get("rejection_reason"), "verified_at": datetime.now(timezone.utc).isoformat()}
    book.sync_state({SELL_PATH_KEY: json.dumps(rec)})
    return rec


def _book_read(book, name: str, strict: bool):
    """``book.<name>()``; with ``strict``, a store error raises instead of reading as empty (LN-T10)."""
    import inspect

    fn = getattr(book, name)
    return fn(strict=True) if strict and "strict" in inspect.signature(fn).parameters else fn()


def _book_write(ok, what: str, strict: bool) -> None:
    """A real session refuses to go on when its book could not be saved (LN-T10)."""
    if strict and ok is False:
        raise RuntimeError(f"live book write failed: {what}; nothing more is sent, the backup run retries")


def _equity_and_flows(snaps: Optional[pd.DataFrame]) -> Tuple[pd.Series, pd.Series]:
    """Snapshot equity and the ladder's flows (``snapshot_json.flow``) by date."""
    eq = _series(snaps, "equity")
    flows = pd.Series(0.0, index=eq.index)
    if snaps is not None and not snaps.empty and "snapshot_json" in snaps.columns:
        for d, raw in zip(pd.to_datetime(snaps["date"].astype(str).str[:10]), snaps["snapshot_json"]):
            try:
                f = float(json.loads(raw or "{}").get("flow") or 0.0)
            except (TypeError, ValueError):
                f = 0.0
            if f and d in flows.index:
                flows.loc[d] = f
    return eq, flows


def _trial_gates(dep) -> Dict[str, dict]:
    """Stored G4 reports of the trial books that paper-trade the deployed configuration (tracker V5)."""
    try:
        from database.connection import get_db_manager
        from database.paper_cloud import PaperCloudSync
        from nse_engine import forward_gate as fg
        from nse_engine.books import discover_books

        mgr, out = get_db_manager(), {}
        for b in discover_books():
            if b.is_deployed or b.config_hash != dep.engine.config_hash():
                continue
            gate = fg.stored_gate(PaperCloudSync(mgr, schema=b.schema).read_state())
            if gate:
                out[b.name] = gate
        return out
    except Exception as exc:                              # noqa: BLE001 - go-live then reads the deployed book only
        logger.warning("trial books unavailable: %s", exc)
        return {}


def _paper_state(paper_book=None) -> Dict[str, str]:
    """State of the deployed paper book (default schema): its G4 report."""
    if paper_book is None:
        from database.connection import get_db_manager
        from database.paper_cloud import PaperCloudSync
        paper_book = PaperCloudSync(get_db_manager(), schema=None)
    try:
        return paper_book.read_state()
    except Exception as exc:                              # noqa: BLE001
        logger.warning("paper book state unavailable: %s", exc)
        return {}


def _ladder_step(ex, data, view, dep, ledger: dict, state: Dict[str, str], book, session, history: pd.Series,
                 marked: dict, since: Optional[str] = None, requested: Optional[float] = None, label: str = "",
                 strict: bool = False):
    """G4 on the live book, then tonight's capital-ladder decision (``nse_engine.capital_ladder``).

    G4 compares the sessions since ``since`` (the first on the current
    configuration, ``capital_ladder.config_since``) with a backtest of that
    configuration; drawdown and the kill criteria still read the whole book.
    """
    from nse_engine import capital_ladder as cl
    from nse_engine import paper_gate as pg
    from nse_engine.engine import run_backtest

    gate = None
    window = history[history.index >= pd.Timestamp(since)] if since else history
    if len(window) >= 3:
        try:
            cfg = dep.reference_config().replace(start=window.index[0].date().isoformat(),
                                                 end=session.date().isoformat(),
                                                 initial_capital=float(ledger["capital"]))
            ref = run_backtest(data, cfg, record=False, tag="live-reference")
            gate = pg.evaluate(window, ref.returns, fills=_book_read(book, "read_fills", strict),
                               reference_trades=ref.trades, sessions=_book_read(book, "read_sessions", strict))
        except Exception as exc:                          # noqa: BLE001 - the ladder then holds
            logger.warning("live G4 unavailable: %s", exc)
    dd, _ = ex.drawdown_decision(session, marked["equity"])
    nifty = view.index_close["NIFTY50"] if "NIFTY50" in view.index_close.columns else pd.Series(dtype="float64")
    nifty = nifty[(nifty.index >= history.index[0]) & (nifty.index <= session)] if len(history) else nifty
    sessions = _book_read(book, "read_sessions", strict)
    mults = (pd.to_numeric(sessions.sort_values("session_date")["shift_multiplier"], errors="coerce").fillna(1.0).tolist()
             if sessions is not None and not sessions.empty and "shift_multiplier" in sessions.columns else [])
    lstate = cl.LadderState.load(state.get(cl.STATE_KEY), float(ledger["capital"]), session.date().isoformat())
    req = requested if requested is not None else os.environ.get(ENV_LIVE_CAPITAL)
    decision = cl.evaluate(lstate, session.date().isoformat(), gate=gate,
                           drawdown_state=getattr(dd, "state", "normal") if dd is not None else "normal",
                           book_dd=cl.current_drawdown(history), nifty_dd=cl.current_drawdown(nifty),
                           shift_multipliers=mults, requested_capital=float(req) if req else None,
                           backtest_maxdd=cl.backtest_maxdd_for(dep.engine.config_hash()),
                           capital_setting=f"{label}'s capital on Fly Kite" if label else ENV_LIVE_CAPITAL)
    return decision, gate, lstate                     # saved by the caller with the ledger (LN-T10)


def _session_row(report: dict, plan, marked: dict, snap: dict, orders: List[dict]) -> dict:
    filled = [f for f in report["fills"] if f["status"] == "FILLED"]
    missed = [f for f in report["fills"] if f["status"] != "FILLED"]
    verb = "would place" if report["mode"] == "live dry run" else "placed"
    outcome = (f"{len(filled)} filled at the open, {len(missed)} not filled; {verb} {len(orders)} order(s)"
               if plan is not None else "no plan")
    return {"session_date": report["session"], "ran_at": datetime.now(timezone.utc).isoformat(),
            "equity": snap["equity"], "cash": marked["cash"], "open_positions": snap["open_positions"],
            "rebalance_day": bool(plan is not None and "rebalance_day" in (plan.notes or [])),
            "planned_buys": len(plan.buys) if plan is not None else 0,
            "planned_sells": len(plan.sells) if plan is not None else 0,
            "queued": len(orders), "filled": len(filled), "cancelled": len(missed),
            "stops_triggered": len(report["external"]),
            "stops_armed": len(plan.stop_instructions) if plan is not None else 0,
            "skipped": len(plan.skipped) if plan is not None else 0,
            "shift_multiplier": float(plan.shift_multiplier) if plan is not None else 1.0,
            "outcome": f"[{report['mode']}] {outcome}"[:200],
            "notes": "; ".join(report["notes"] + report["alerts"])[:500],
            "drawdown_state": str(getattr(plan, "drawdown_state", "normal") or "normal") if plan is not None else "normal",
            "drawdown_pct": float(getattr(plan, "drawdown_pct", 0.0) or 0.0) if plan is not None else 0.0}


def _email(report: dict, dep, snap: dict, capital: float, schema: str, label: str = "") -> None:
    try:
        from services.notifications.manager import NotificationManager
        plan = report.get("plan")
        orders = [r for r in report["results"] if r.get("type") != "gtt_reconcile"]
        NotificationManager().email_engine_daily_report({
            "mode": report["mode"], "session": report["session"],
            "deployment": f"{dep.status} {dep.engine.config_hash()[:8]} · live book, schema {schema}, "
                          f"capital {capital:,.0f}",
            "equity": snap["equity"], "initial_capital": capital, "cash": snap["cash"],
            "pnl": snap["cumulative_pnl"], "pnl_pct": snap["cumulative_pnl_pct"],
            "max_drawdown_pct": snap["max_drawdown_pct"], "open_positions": snap["open_positions"],
            "filled": [{"symbol": f["symbol"], "side": f["side"], "quantity": f["quantity"],
                        "fill_price": f["fill_price"], "costs": f["costs_inr"]}
                       for f in report["fills"] + report["external"] if f["status"] == "FILLED"],
            "cancelled": [{"symbol": f["symbol"], "side": f["side"], "note": f["note"]}
                          for f in report["fills"] if f["status"] != "FILLED"],
            "stops": [], "queued": [{"symbol": r.get("symbol"), "side": r.get("side"), "quantity": r.get("quantity"),
                                     "reason": f"{r.get('status')} {r.get('reason') or ''}".strip()} for r in orders],
            "notes": list(report["notes"]), "alerts": list(report["alerts"]),
            "drawdown_rule": (f"{plan.drawdown_state} · {plan.drawdown_pct:.1f}% below the episode peak"
                              if plan is not None and dep.drawdown_rule is not None else None),
            "drawdown_state": getattr(plan, "drawdown_state", "normal") if plan is not None else "normal",
            "paper_gate": report.get("gate"),
            "ladder": report.get("ladder"),
            "book_label": label or None,
        })
    except Exception as exc:                             # noqa: BLE001 - reporting only
        logger.warning("Live daily email failed: %s", exc)


def _login_accepted(kite) -> bool:
    """Does Kite still accept the stored login?  One issued before the app's API secret was
    regenerated, or logged out since, is refused (TokenException): the session then skips as it
    does without a login, rather than failing."""
    from kiteconnect.exceptions import TokenException

    try:
        kite.profile()
    except TokenException:
        return False
    return True


def _email_skip(message: str, label: str = "") -> None:
    """CRITICAL (tracker AL2): no live session today, so no exits or rebalances (GTT stops still hold)."""
    from services.notifications.alerts import CRITICAL, alert

    alert(CRITICAL, f"live_no_login:{label or 'own'}",
          f"Centurion live{f' [{label}]' if label else ''}: no valid Kite login today, session skipped", [message],
          book=label or None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run one live session of the NSE engine book (tracker L5)")
    ap.add_argument("--dry-run", action="store_true", help="build every order, send none")
    ap.add_argument("--capital", type=float, default=None,
                    help=f"the book's capital, needed on the first session (or {ENV_LIVE_CAPITAL})")
    ap.add_argument("--as-of", default=None, help="session date (default: the latest store session)")
    ap.add_argument("--record", action="store_true", help="dry run: keep the rehearsal book in Neon")
    ap.add_argument("--no-email", action="store_true")
    ap.add_argument("--stored-token", action="store_true",
                    help="the scheduled run: today's token from the Kite login callback, calls through "
                         "CENTURION_KITE_PROXY (U23)")
    ap.add_argument("--account", default="",
                    help="a connected account's id (kite_connect.auth.accounts, FA2): its own book, Kite app, "
                         "capital and mode; needs --stored-token")
    ap.add_argument("--reconcile", nargs="+", metavar="SYM=QTY@PX",
                    help="set the ledger's SYM to QTY (what the broker holds), booking the difference at PX, "
                         "to clear a RECONCILE NEEDED alert (LN-T6); no Kite login needed")
    ap.add_argument("--record-sell-path", nargs=3, metavar=("DDPI_CONFIRMED_ON", "AMO_ORDER_ID", "GTT_ID"),
                    help="verify the supervised sell test against Kite and record it (go-live check, LN-T2)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    notes: List[str] = []
    acct, book, dry_run, capital = None, None, args.dry_run, args.capital
    if args.account:
        from kite_connect.auth import accounts
        acct = accounts.get(args.account)
        if acct.is_primary or not args.stored_token:
            ap.error("--account names a connected account (your own is the default) and needs --stored-token")
        if acct.mode == "off":
            print(f"{acct.id}: Centurion does not manage this account (mode off)")
            return 0
        dry_run = dry_run or acct.mode == "dry_run"      # never above the master switch (--dry-run)
        lock = accounts.trading_lock(acct, "dry_run" if dry_run else "live")
        if lock:
            print(f"{acct.id}: session skipped: {lock}")
            return 0
        capital = capital if capital is not None else acct.capital
    if args.reconcile:
        from nse_engine.deployment import load_deployment

        items = {}
        for item in args.reconcile:
            try:
                sym, rest = item.split("=", 1)
                qty, px = rest.split("@", 1)
                items[sym.strip().upper()] = (int(qty), float(px))
            except ValueError:
                ap.error(f"--reconcile {item!r}: expected SYM=QTY@PX, e.g. INFY=40@1512.5")
        rbook = live_book(acct.schema if acct else None)
        led = load_ledger(rbook.read_state(), None)
        led, rows = manual_reconcile(led, items, pd.Timestamp(datetime.now(timezone.utc).date()),
                                     load_deployment().engine.costs.dp_charge_inr)
        rbook.sync_fills(rows)
        rbook.sync_state({LIVE_LEDGER_KEY: json.dumps(led)})
        print(json.dumps({"reconciled": [f"{r['side']} {r['symbol']} x{r['quantity']} @ {r['fill_price']}" for r in rows],
                          "positions": {s: led["positions"].get(s, 0) for s in items}}))
        return 0
    if args.stored_token:
        from kite_connect.auth import daily_login as dl
        book = live_book(acct.schema if acct else None)
        kite = dl.kite_from_stored_token(book, key=acct.api_key if acct else None)
        rejected = kite is not None and not _login_accepted(kite)
        if kite is None or rejected:
            whose = f" for {acct.name}" if acct else ""
            msg = (f"Kite rejected today's login{whose} (made before the API secret was regenerated, or logged out "
                   "since), so the live session was skipped: no orders were placed. GTT stops stay active at "
                   "Zerodha. Log in again from the reminder email or Fly Kite." if rejected else
                   f"No Kite login today{whose}, so the live session was skipped: no orders were placed. "
                   "GTT stops stay active at Zerodha. Log in tomorrow from the reminder email.")
            logger.warning(msg)
            today = datetime.now(dl.IST).date().isoformat()
            if not args.no_email and book.read_state().get(LIVE_SKIP_NOTIFIED_KEY) != today:
                _email_skip(msg, acct.name if acct else "")   # once a day, not once per backup run
                book.sync_state({LIVE_SKIP_NOTIFIED_KEY: today})
            print(msg)
            return 0
        proxy = os.environ.get(dl.ENV_PROXY, "")
        if not dry_run and not proxy:
            raise SystemExit(f"real orders need {dl.ENV_PROXY}: Zerodha accepts API orders only from the "
                             "registered static IP")
        if proxy:
            ip, ok = dl.check_egress(proxy)
            notes.append(f"Kite calls through the static-IP proxy, egress IP {ip or 'unknown'}")
            if not ok:
                msg = (f"egress IP {ip or 'unknown'} is not the IP registered with Zerodha "
                       f"({os.environ.get(dl.ENV_STATIC_IP)}): orders would be rejected")
                if not dry_run:
                    raise SystemExit(f"real orders refused: {msg}")
                notes.append("WARNING: " + msg)
    else:
        from kite_connect.auth.kite_session import create_kite_session
        kite = create_kite_session()
    if args.record_sell_path:
        from nse_engine.capital_ladder import sell_path_check

        rec = record_sell_path(kite, book if book is not None else live_book(), *args.record_sell_path)
        ok, detail = sell_path_check(rec, rec.get("user_id"))
        print(f"sell path {'PASS' if ok else 'FAIL'}: {detail}")
        return 0 if ok else 1
    report = run_live_session(kite, dry_run=dry_run, capital=capital, as_of=args.as_of, book=book,
                              record=True if args.record else None, email=not args.no_email, extra_notes=notes,
                              once_per_session=args.stored_token, requested_capital=acct.capital if acct else None,
                              label=acct.name if acct else "", unwind_sessions=acct.unwind_sessions if acct else 0)
    if acct is not None and acct.unwind_sessions and not report.get("skipped"):
        accounts.unwind_step(acct.id, report, dry_run)
    # Counts only: Actions logs of the public repo are public; the detail is in the email
    print(json.dumps({"session": report.get("session"), "mode": report.get("mode"), "skipped": report.get("skipped"),
                      **{k: len(report.get(k) or []) for k in ("fills", "external", "results", "alerts")}}))
    return 0          # alerts are in the email; only an exception fails the run


if __name__ == "__main__":
    raise SystemExit(main())
