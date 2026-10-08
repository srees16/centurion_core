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
LIVE_LEDGER_KEY = "live_ledger"            # {"capital", "cash", "positions": {sym: qty}, "symbols", "session"}
LIVE_ORDERS_KEY = "live_orders"            # orders placed at the last live session (JSON)
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
    return {"capital": float(capital), "cash": float(capital), "positions": {}, "symbols": [], "session": ""}


def apply_fills(ledger: dict, fills: List[dict], session, dp_charge_inr: float) -> dict:
    """The ledger after ``fills`` (engine and external), applied once per session."""
    from nse_engine.costs import statutory_cost

    day = pd.Timestamp(session).date().isoformat()
    if str(ledger.get("session") or "") >= day:
        return ledger                                   # this session's fills are already in
    led = {**ledger, "positions": dict(ledger.get("positions") or {}), "symbols": list(ledger.get("symbols") or [])}
    for f in fills:
        if f.get("status") != "FILLED" or int(f.get("quantity") or 0) <= 0:
            continue
        sym, side, qty, px = str(f["symbol"]), str(f["side"]).upper(), int(f["quantity"]), float(f["fill_price"])
        have = int(led["positions"].get(sym, 0))
        if side == "SELL":
            qty = min(qty, have)
            if qty <= 0:
                continue
            led["positions"][sym] = have - qty
            led["cash"] += qty * px - statutory_cost(qty * px, "SELL", pd.Timestamp(session), dp_charge_inr, symbol=sym)
        else:
            led["positions"][sym] = have + qty
            led["cash"] -= qty * px + statutory_cost(qty * px, "BUY", pd.Timestamp(session), dp_charge_inr, symbol=sym)
            if sym not in led["symbols"]:
                led["symbols"].append(sym)
    led["positions"] = {s: q for s, q in led["positions"].items() if q > 0}
    led["session"] = day
    return led


def scoped_book(ledger: dict, broker_holdings: Dict[str, dict], broker_cash: float) -> Tuple[Dict[str, dict], float, List[str]]:
    """(holdings, cash, alerts): the ledger's symbols as the broker holds them."""
    alerts, holdings = [], {}
    for sym, q in (ledger.get("positions") or {}).items():
        b = broker_holdings.get(sym) or {}
        bq = int(b.get("quantity") or 0)
        if bq < q:
            alerts.append(f"{sym}: the ledger holds {q} but the broker {bq} - using {bq}; check Kite")
        if min(q, bq) > 0:
            holdings[sym] = {"quantity": min(q, bq), "avg_price": b.get("avg_price", 0.0), "stop_price": b.get("stop_price")}
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
                     "impact_bps": round(impact, 3), "costs_inr": round(cost, 2),
                     "note": note[:200], "occurred_at": f"{day}T09:15:00+05:30"})
    return rows


def external_sells(order_book: List[dict], symbols, session, dp_charge_inr: float) -> List[dict]:
    """Completed CNC sells today of ledger ``symbols`` the engine did not place (GTT stop, manual)."""
    from nse_engine.costs import statutory_cost

    day = pd.Timestamp(session).date().isoformat()
    rows = []
    for o in order_book or []:
        sym = o.get("tradingsymbol")
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
        checks = cl.readiness(pgate, json.loads(state.get(cl.DRY_RUNS_KEY) or "[]"), source=source)
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

    raw_equity, flows = _equity_and_flows(book.read_snapshots())
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

    # 2. outcomes of the previous live session's orders, then the ledger
    placed_doc = json.loads(state.get(LIVE_ORDERS_KEY) or "{}")
    decision = placed_doc.get("decision_date")
    if decision and pd.Timestamp(decision) < session:
        opens_row = view.open.iloc[-1]
        placed = {o["tag"]: o for o in placed_doc.get("orders", []) if o.get("tag")}
        opens = {s: float(opens_row[s]) for s in {o.get("symbol") for o in placed.values()} if s in opens_row.index}
        outcomes = live_order_outcomes(kite, decision)
        if outcomes and outcomes[0].get("status") == "unknown":
            report["alerts"].append(f"order book unavailable: {outcomes[0].get('error')}")
        report["fills"] = fills_from_outcomes(outcomes, placed, session, decision, opens, dp)
        seen = {o.get("tag") for o in outcomes}
        lost = [f"{s.get('side')} {s.get('symbol')}" for t, s in placed.items()
                if t not in seen and s.get("status") == "PLACED"]
        if lost and not (outcomes and outcomes[0].get("status") == "unknown"):
            report["alerts"].append("placed but missing from the order book: " + ", ".join(lost))
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
    ledger = apply_fills(ledger, report["fills"] + report["external"], session, dp)

    # 3. the book at the close
    broker_holdings, broker_cash = kite_book(kite)
    holdings, cash, scope_alerts = scoped_book(ledger, broker_holdings, broker_cash)
    report["alerts"].extend(scope_alerts)
    scope["holdings"], scope["cash"] = holdings, cash
    outside = sorted(set(broker_holdings) - set(ledger.get("positions") or {}))
    if outside:
        report["notes"].append(f"{len(outside)} holding(s) outside the book, left alone: {', '.join(outside[:10])}")
    closes = view.close.ffill().iloc[-1]
    marked = mark_book(holdings, float(ledger["cash"]), closes)
    closed_today = sum(1 for f in report["fills"] + report["external"] if f["side"] == "SELL" and f["status"] == "FILLED")
    prior, prior_flows = raw_equity[raw_equity.index < session], flows[flows.index < session]
    history = flow_adjusted(pd.concat([prior, pd.Series([marked["equity"]], index=pd.DatetimeIndex([session]))]),
                            prior_flows)                  # today's equity before any ladder flow
    today_flow = 0.0
    if record and not dry_run and session.date() >= dep.paper_start_date:
        from nse_engine import capital_ladder as cl
        since, marker, replaced = cl.config_since(state.get(cl.LIVE_CONFIG_KEY), dep.engine.config_hash(),
                                                  history.index[0].date().isoformat(), session.date().isoformat())
        if marker:
            book.sync_state({cl.LIVE_CONFIG_KEY: marker})
        if replaced:
            report["notes"].append(f"configuration {replaced[:8]} -> {dep.engine.config_hash()[:8]} from {since}: "
                                   "the live G4 window restarts here (V5)")
        decision, gate = _ladder_step(ex, data, view, dep, ledger, state, book, session, history, marked, since,
                                      requested_capital, label)
        report["ladder"] = decision.line()
        report["alerts"].extend(decision.alerts)
        if gate is not None:
            from nse_engine import paper_gate as pg
            report["gate"] = pg.one_line(gate)
            book.sync_state({pg.STATE_KEY: pg.summary_json(gate, updated_at=datetime.now(timezone.utc).isoformat(),
                                                           config_hash=dep.engine.config_hash())})
        if decision.flow:
            today_flow = float(decision.flow)
            ledger["capital"] = float(decision.capital)
            ledger["cash"] = float(ledger["cash"]) + today_flow
            holdings, cash, _ = scoped_book(ledger, broker_holdings, broker_cash)
            scope["holdings"], scope["cash"] = holdings, cash
            marked = mark_book(holdings, float(ledger["cash"]), closes)
    all_flows = pd.concat([prior_flows, pd.Series([today_flow], index=pd.DatetimeIndex([session]))])
    history = flow_adjusted(pd.concat([prior, pd.Series([marked["equity"]], index=pd.DatetimeIndex([session]))]),
                            all_flows)                    # in tonight's capital base, flows removed
    snap = snapshot_row(session, marked, history, float(ledger["capital"]), closed_today)
    if today_flow:
        js = json.loads(snap["snapshot_json"]); js["flow"] = today_flow; snap["snapshot_json"] = json.dumps(js, default=str)
    report["snapshot"], report["ledger"] = snap, ledger
    if record:
        if not state.get("epoch"):                       # the book begins at this session
            start = (session.tz_localize("Asia/Kolkata")).tz_convert("UTC").isoformat()
            book.sync_state({"epoch": start, "book_start": start, "book_owner": "live_engine",
                             "initial_capital": ledger["capital"]})
        book.sync_fills(report["fills"] + report["external"])
        book.sync_snapshot(snap)
        book.sync_state({LIVE_LEDGER_KEY: json.dumps(ledger)})

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
        report["plan"] = plan
        stale = any(s.get("reason") == "stale_data" for s in plan.skipped)
        if stale:
            report["alerts"].append("stale market data: no orders planned")
        report["results"] = ex.dry_run_live(plan) if dry_run else ([] if stale else ex.execute(plan))
    orders = [r for r in report["results"] if r.get("type") != "gtt_reconcile"]
    refused = [r for r in orders if not r.get("success") and r.get("status") != "DUPLICATE"]
    if refused:
        report["alerts"].append("orders refused: " + "; ".join(
            f"{r.get('side')} {r.get('symbol')}: {r.get('error') or r.get('status')}" for r in refused))
    gtt = [r for r in report["results"] if r.get("type") == "gtt_reconcile"]
    if gtt and not gtt[0].get("success"):
        report["alerts"].append(f"GTT stop reconciliation errors: {(gtt[0].get('report') or {}).get('errors')}")
    if plan is not None and plan.drawdown_changed:
        report["alerts"].append(f"DRAWDOWN RULE now {plan.drawdown_state.upper()} at {plan.drawdown_pct:.1f}% below the peak")

    # 6. record the session and today's orders for tomorrow
    if record:
        values = {LIVE_LAST_SESSION_KEY: session.date().isoformat()}
        if plan is not None and not dry_run:
            refs = {o.symbol: o.ref_price for o in plan.orders}
            values[LIVE_ORDERS_KEY] = json.dumps({"decision_date": session.date().isoformat(), "orders": [
                {"tag": r.get("tag"), "symbol": r.get("symbol"), "side": r.get("side"), "quantity": r.get("quantity"),
                 "limit_price": r.get("limit_price"), "ref_price": refs.get(r.get("symbol"), 0.0),
                 "status": "PLACED" if r.get("status") == "DUPLICATE" else r.get("status")} for r in orders]})
        book.sync_state(values)
        book.sync_session(_session_row(report, plan, marked, snap, orders))
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
                 marked: dict, since: Optional[str] = None, requested: Optional[float] = None, label: str = ""):
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
            gate = pg.evaluate(window, ref.returns, fills=book.read_fills(), reference_trades=ref.trades,
                               sessions=book.read_sessions())
        except Exception as exc:                          # noqa: BLE001 - the ladder then holds
            logger.warning("live G4 unavailable: %s", exc)
    dd, _ = ex.drawdown_decision(session, marked["equity"])
    nifty = view.index_close["NIFTY50"] if "NIFTY50" in view.index_close.columns else pd.Series(dtype="float64")
    nifty = nifty[(nifty.index >= history.index[0]) & (nifty.index <= session)] if len(history) else nifty
    sessions = book.read_sessions()
    mults = (pd.to_numeric(sessions.sort_values("session_date")["shift_multiplier"], errors="coerce").fillna(1.0).tolist()
             if sessions is not None and not sessions.empty and "shift_multiplier" in sessions.columns else [])
    lstate = cl.LadderState.load(state.get(cl.STATE_KEY), float(ledger["capital"]), session.date().isoformat())
    req = requested if requested is not None else os.environ.get(ENV_LIVE_CAPITAL)
    decision = cl.evaluate(lstate, session.date().isoformat(), gate=gate,
                           drawdown_state=getattr(dd, "state", "normal") if dd is not None else "normal",
                           book_dd=cl.current_drawdown(history), nifty_dd=cl.current_drawdown(nifty),
                           shift_multipliers=mults, requested_capital=float(req) if req else None,
                           capital_setting=f"{label}'s capital on Fly Kite" if label else ENV_LIVE_CAPITAL)
    book.sync_state({cl.STATE_KEY: lstate.dump()})
    return decision, gate


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
    import html

    try:
        from services.notifications.manager import NotificationManager
        NotificationManager._send_html_email(
            f"Centurion live{f' [{label}]' if label else ''}: no valid Kite login today, session skipped",
            f"<html><body style='font-family:Segoe UI,Arial,sans-serif;padding:20px;'><p>{html.escape(message)}</p></body></html>")
    except Exception as exc:                              # noqa: BLE001
        logger.warning("skip email failed: %s", exc)


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
