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
4. saves the book (Neon schema ``us_advisory``, or ``--state-file``) and emails
   the suggestions: sells, buys, raised stops, the book against the S&P 500.

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


def _px(frame: pd.DataFrame, day: pd.Timestamp, sym: str) -> float:
    return float(frame.at[day, sym]) if sym in frame.columns else float("nan")


def _fill_cost(value: float) -> float:
    return abs(value) * FILL_COST_BPS / 1e4


def _sell(book: Book, sym: str, qty: int, px: float, day: str, reason: str) -> None:
    h = book.holdings[sym]
    value = qty * px
    book.cash += value - _fill_cost(value)
    book.trades.append({"date": day, "symbol": sym, "side": "SELL", "qty": qty, "price": round(px, 4),
                        "reason": reason, "pnl": round(qty * (px - h["avg_price"]) - _fill_cost(value), 2)})
    h["qty"] -= qty
    if h["qty"] <= 0:
        del book.holdings[sym]


def _buy(book: Book, sym: str, qty: int, px: float, day: str, reason: str, stop: Optional[float]) -> None:
    value = qty * px
    book.cash -= value + _fill_cost(value)
    book.trades.append({"date": day, "symbol": sym, "side": "BUY", "qty": qty, "price": round(px, 4),
                        "reason": reason})
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
    for o in [o for o in book.pending if o["side"] == "SELL"]:
        sym, h = o["symbol"], book.holdings.get(o["symbol"])
        if h is None:
            continue
        px = _px(data.open, day, sym)
        if sym not in data.close.columns:
            px = float(h["last_price"])
            events.append(f"{sym}: no market data; sold in the model at its last close {px:.2f} (check your broker)")
        elif not np.isfinite(px):
            events.append(f"{sym}: no open on {d}; the sell did not fill")
            continue
        _sell(book, sym, min(int(o["qty"]), int(h["qty"])), px, d, o["reason"])
    for o in [o for o in book.pending if o["side"] == "BUY"]:
        sym = o["symbol"]
        px = _px(data.open, day, sym)
        if not np.isfinite(px) or px <= 0:
            events.append(f"{sym}: no open on {d}; the buy did not fill")
            continue
        qty = min(int(o["qty"]), int(math.floor(max(book.cash, 0.0) / (px * (1 + FILL_COST_BPS / 1e4)))))
        if qty <= 0:
            events.append(f"{sym}: not enough cash at the open; the buy did not fill")
            continue
        if qty < int(o["qty"]):
            events.append(f"{sym}: bought {qty} of {o['qty']} shares (cash at the open)")
        _buy(book, sym, qty, px, d, o["reason"], o.get("stop"))


def _apply_stops(book: Book, panel: UsPanel, day: pd.Timestamp, events: List[str]) -> None:
    data, d = panel.data, day.date().isoformat()
    for sym in list(book.holdings):
        h = book.holdings[sym]
        px = stop_fill_price(_px(data.open, day, sym), _px(data.low, day, sym), h.get("stop"))
        if px is None:
            continue
        _sell(book, sym, int(h["qty"]), px, d, "stop")
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
    book.pending = [{"symbol": o.symbol, "side": o.side, "qty": o.qty, "reason": o.reason, "stop": o.stop}
                    for o in kept]
    return Plan(as_of=as_of, target=target, orders=kept, raised=raised, drawdown=decision, skipped=skipped)


# ============================================================================
# the email
# ============================================================================


def next_open_ist(as_of: pd.Timestamp) -> datetime:
    """The next weekday's 09:30 New York open in IST (a US holiday shifts it a day)."""
    day = pd.Timestamp(as_of).normalize() + pd.offsets.BDay(1)
    return datetime.combine(day.date(), US_OPEN, tzinfo=NEW_YORK).astimezone(IST)


def _money(x: float) -> str:
    return f"${x:,.0f}"


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


def summary_lines(book: Book, plan: Plan, panel: UsPanel) -> List[str]:
    data = panel.data
    spx = data.index_close["SPX"].dropna()
    vix = data.index_close["VIX"].dropna()
    ma200 = spx.rolling(200).mean().iloc[-1]
    start = pd.Timestamp(book.start)
    spx_ret = spx.iloc[-1] / spx[spx.index <= start].iloc[-1] - 1
    t = plan.target
    dd = plan.drawdown
    lines = [
        f"Market at the {plan.as_of:%d %b %Y} close: regime {t.regime} (scale {t.regime_scale:.1f}); "
        f"S&P 500 {spx.iloc[-1]:,.0f}, {spx.iloc[-1] / ma200 - 1:+.1%} vs its 200-day average; VIX {vix.iloc[-1]:.1f}.",
        f"Model book since {start:%d %b %Y}: {book.value() / NOTIONAL_USD - 1:+.2%} vs S&P 500 price {spx_ret:+.2%}; "
        f"{len(book.holdings)} names, cash {book.cash / book.value():.0%}"
        + (f"; drawdown rule {dd.state} ({dd.drawdown:.1%} below its peak)" if dd else "") + ".",
    ]
    if panel.missing:
        lines.append(f"Left out today (no data): {', '.join(panel.missing)}.")
    return lines


def render_html(book: Book, plan: Plan, panel: UsPanel, capital: float, capital_set: bool,
                events: Sequence[str]) -> str:
    tables = report_tables(book, plan, panel, capital)
    opens = next_open_ist(plan.as_of)

    def table(df: pd.DataFrame) -> str:
        if df.empty:
            return "<p>None.</p>"
        head = "".join(f"<th style=\"text-align:left;padding:3px 8px;border-bottom:1px solid #999;\">{escape(c)}</th>"
                       for c in df.columns)
        body = "".join("<tr>" + "".join(f"<td style=\"padding:3px 8px;\">{escape(str(v))}</td>" for v in row) + "</tr>"
                       for row in df.itertuples(index=False))
        return f"<table style=\"border-collapse:collapse;font-size:13px;\"><tr>{head}</tr>{body}</table>"

    size_note = (f"Your $ and shares are for your capital of {_money(capital)}." if capital_set else
                 f"Your $ and shares are per {_money(capital)} of capital: set {ENV_CAPITAL} to size them for yours.")
    parts = [
        "<html><body style=\"font-family:Segoe UI,Arial,sans-serif;font-size:14px;\">",
        f"<h3>US suggestions for the {opens:%a %d %b} open ({opens:%H:%M} IST)</h3>",
        "<p style=\"background:#fff4e5;padding:6px;\"><b>Unvalidated.</b> The India book's deployed rules on "
        "today's S&amp;P 500; no US backtest or gate has passed yet (US2-US7). Advisory only: you place every "
        "order.</p>",
        "<ul>" + "".join(f"<li>{escape(x)}</li>" for x in summary_lines(book, plan, panel)) + "</ul>",
        f"<h4>Sell ({len(tables['sells'])})</h4>", table(tables["sells"]),
        f"<h4>Buy ({len(tables['buys'])})</h4>", table(tables["buys"]),
        f"<h4>Raise these stops ({len(tables['raised'])})</h4>", table(tables["raised"]),
        "<h4>How to place</h4><ul>"
        f"<li>At the open ({opens:%H:%M} IST): sells first (the shares listed, or the whole holding for \"all\"), "
        "then buys of the dollar amounts (fractional shares), as market orders or limits a little above the last "
        "close.</li>"
        "<li>Give every holding a good-till-cancelled sell-stop at its stop. Raise stops when listed; never lower "
        "them. Check that your broker takes stops on fractional quantities.</li>"
        f"<li>{escape(size_note)} The model assumes each order filled at the open; tell Claude if one did not.</li>"
        "<li>Indian tax: gains on shares held under 24 months are taxed at your slab rate; US dividends lose 25% "
        "to withholding (credited in India).</li></ul>",
        f"<h4>Model holdings after the {plan.as_of:%d %b} close ({len(tables['held'])})</h4>", table(tables["held"]),
    ]
    notes = list(events) + plan.skipped
    if notes:
        parts += ["<h4>Since the last email</h4><ul>", "".join(f"<li>{escape(x)}</li>" for x in notes), "</ul>"]
    parts.append(f"<p style=\"color:#6b7280;font-size:12px;\">data {escape(panel.data.data_hash)} &middot; "
                 f"{panel.members} members &middot; universe {plan.target.universe_size} &middot; "
                 f"{datetime.now(IST):%d %b %Y %H:%M} IST</p></body></html>")
    return "".join(parts)


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
    else:
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
    if not email:
        return 0
    opens = next_open_ist(plan.as_of)
    subject = (f"[Centurion US] {len(plan.side('BUY'))} buys, {len(plan.side('SELL'))} sells "
               f"for the {opens:%d %b} open (unvalidated)")
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
