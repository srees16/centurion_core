"""
The funding gate (owner's decision 5, 16 Nov 2026; tracker GTR0): no book is levered without the owner's recorded
answer, and never above 1.5x.

Leverage is sized only from the deflated Sharpe (decision 2): half-Kelly, ``0.5 x SR_deflated / volatility``, once
the deflated-Sharpe gate (DSR >= 0.95, every trial counted) has passed; 1.0x otherwise; capped at ``CAP``.  A
walk-forward or backtest Sharpe is never an input: :func:`implied_leverage` takes the deflated one by name.

Before any leverage, in paper and live:

1. the extra exposure is computed: ``(target - 1) x book equity``, in rupees (:func:`open_request`);
2. the owner is emailed the amount, the reason and the deadline, and the books' emails carry a line until it is
   answered (:func:`notice`, :func:`email_line`); the gate WAITS: nothing is levered on a pending request;
3. "top-up": the bigger book runs UNLEVERED on the owner's cash, once the cash is seen in the account (Kite's
   cash balance in live; the paper book's cash in paper);
4. "no funds": leverage only through index or stock futures (their cost is about the liquid-fund rate), with the
   margin covered by cash; never margin funding (MTF, 15-18%), which this module has no path to;
5. no reply by the deadline: 1x, and the owner is asked again at the next sizing change (a new target);
6. every request, reply and action is logged in the book's state (``STATE_KEY``); the reply command prints the
   tracker line to add.

:func:`decide` turns the state into tonight's leverage, capital and instrument.  It is pure: the caller passes the
date and the cash it read, and stores the returned state.  Paper simulates the futures leverage but sends the same
request, so the flow is tested end to end before any live use (live only after RA1 and the owner's dated approval).

    python -m nse_engine.funding_gate status --schema sharpe
    python -m nse_engine.funding_gate reply --schema sharpe --request <id> --answer topup --amount 300000
    python -m nse_engine.funding_gate reply --schema sharpe --request <id> --answer no-funds
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, Optional, Sequence

#: The hard cap on any book's leverage (decision 2).
CAP = 1.5
#: The deflated-Sharpe gate leverage waits for (decision 3, every trial counted).
DSR_BAR = 0.95
#: Days the owner has to answer a request (decision 5e: then 1x, and asked again at the next sizing change).
DEADLINE_DAYS = 7
#: Futures margin as a share of the extra exposure (SPAN plus exposure margin on index and stock futures runs about
#: 12-20%; the upper end, so cash covers a volatile day).  Covered by cash, never borrowed.
MARGIN_RATE = 0.20
#: Two targets closer than this are the same sizing (no new request).
SAME_TARGET = 0.05
STATE_KEY = "funding_gate"
TOPUP, NO_FUNDS, PENDING, EXPIRED = "topup", "no_funds", "pending", "expired"
#: What a decision may lever with.  There is no margin-funding (MTF) entry, on purpose.
INSTRUMENTS = ("none", "own cash", "futures")


def cap(leverage: float) -> float:
    """``leverage`` within [1, CAP]; anything unreadable is 1x."""
    return float(min(CAP, max(1.0, leverage))) if math.isfinite(leverage) else 1.0


def implied_leverage(sr_deflated: float, volatility: float, dsr: float) -> float:
    """Half-Kelly on the DEFLATED Sharpe (annual, excess), capped; 1.0 unless the deflated-Sharpe gate passed."""
    if not (math.isfinite(dsr) and dsr >= DSR_BAR) or not (math.isfinite(volatility) and volatility > 0) \
            or not math.isfinite(sr_deflated):
        return 1.0
    return cap(0.5 * sr_deflated / volatility)


def extra_exposure(target: float, equity: float) -> float:
    """The rupees of exposure the target adds to the book: ``(target - 1) x equity``, the target capped."""
    return max(0.0, (cap(target) - 1.0) * max(0.0, float(equity)))


def _log(doc: Dict[str, Any], day: date, event: str, detail: str) -> None:
    doc.setdefault("log", []).append({"date": day.isoformat(), "event": event, "detail": detail})


def _rupees(x: float) -> str:
    return f"Rs {x:,.0f}"


def open_request(doc: Optional[Dict[str, Any]], book: str, target: float, equity: float, reason: str,
                 today: date, deadline_days: int = DEADLINE_DAYS) -> Dict[str, Any]:
    """The state after a sizing decision of ``target``: a new request when the target needs more than 1x and is a
    new sizing (no request yet, or another target than the last one); the state as it was otherwise."""
    doc = copy.deepcopy(doc or {})
    target = cap(target)
    last = doc.get("request")
    if target <= 1.0:
        return doc
    if last and abs(float(last["target"]) - target) < SAME_TARGET:
        return doc                                        # the same sizing: no second ask (decision 5e)
    extra = extra_exposure(target, equity)
    rid = hashlib.sha256(f"{book}|{today}|{target:.2f}|{equity:.0f}".encode()).hexdigest()[:8]
    doc["request"] = {"id": rid, "book": book, "date": today.isoformat(), "target": target, "equity": float(equity),
                      "extra_inr": extra, "reason": reason,
                      "deadline": (today + timedelta(days=deadline_days)).isoformat(), "status": PENDING}
    _log(doc, today, "requested", f"{rid}: {_rupees(extra)} for {target:.2f}x by {doc['request']['deadline']}")
    return doc


def record_reply(doc: Dict[str, Any], request_id: str, answer: str, today: date,
                 amount: Optional[float] = None) -> Dict[str, Any]:
    """The owner's answer to a pending request: ``topup`` (with the rupees) or ``no_funds``.  Raises ValueError for
    another request, one already answered or past its deadline, or a top-up with no amount."""
    doc = copy.deepcopy(doc)
    req = doc.get("request")
    if not req or req["id"] != request_id:
        raise ValueError(f"no request {request_id}: the open one is {req['id'] if req else 'none'}")
    if req["status"] != PENDING:
        raise ValueError(f"request {request_id} is {req['status']}, not pending")
    if today > date.fromisoformat(req["deadline"]):
        raise ValueError(f"request {request_id} passed its deadline ({req['deadline']}): the book stays at 1x and "
                         "asks again at its next sizing change")
    if answer == TOPUP:
        if not amount or amount <= 0:
            raise ValueError("a top-up needs the amount in rupees")
        req.update(status=TOPUP, amount_inr=float(amount), answered=today.isoformat())
        _log(doc, today, "reply", f"{request_id}: top-up of {_rupees(amount)}")
    elif answer == NO_FUNDS:
        req.update(status=NO_FUNDS, answered=today.isoformat())
        _log(doc, today, "reply", f"{request_id}: no funds; futures only")
    else:
        raise ValueError(f"answer is {TOPUP} or {NO_FUNDS}, not {answer!r}")
    return doc


@dataclass(frozen=True)
class Decision:
    leverage: float                       # 1.0 to CAP
    capital_add_inr: float                # the owner's top-up, run unlevered
    instrument: str                       # one of INSTRUMENTS
    why: str


def decide(doc: Optional[Dict[str, Any]], today: date, cash_inr: float,
           margin_rate: float = MARGIN_RATE) -> tuple:
    """Tonight's (Decision, new state).  ``cash_inr``: the cash seen in the account (Kite's balance in live, the
    paper book's cash in paper).  Never above CAP; futures only on a recorded "no funds"; 1x on anything else."""
    doc = copy.deepcopy(doc or {})
    req = doc.get("request")
    one = lambda why: (Decision(1.0, 0.0, "none", why), doc)          # noqa: E731
    if not req:
        return one("no funding request: 1x")
    status, deadline = req["status"], date.fromisoformat(req["deadline"])
    if status == PENDING and today > deadline:
        req["status"] = EXPIRED
        _log(doc, today, "expired", f"{req['id']}: no reply by {req['deadline']}; 1x until the next sizing change")
        return one(f"request {req['id']} expired unanswered: 1x; asked again at the next sizing change")
    if status == PENDING:
        return one(f"waiting for your reply to request {req['id']} ({_rupees(req['extra_inr'])}) until "
                   f"{req['deadline']}: 1x")
    if status == EXPIRED:
        return one(f"request {req['id']} expired: 1x until the next sizing change")
    if status == TOPUP:
        amount = float(req["amount_inr"])
        if cash_inr < amount:
            return one(f"top-up of {_rupees(amount)} not in the account yet (cash {_rupees(cash_inr)}): 1x")
        if not req.get("applied"):
            req["applied"] = today.isoformat()
            _log(doc, today, "applied", f"{req['id']}: book enlarged by {_rupees(amount)}, unlevered")
        return Decision(1.0, amount, "own cash", f"top-up of {_rupees(amount)} run unlevered"), doc
    if status == NO_FUNDS:
        target = cap(float(req["target"]))
        need = margin_rate * extra_exposure(target, req["equity"])
        if cash_inr >= need:
            lev, why = target, f"futures to {target:.2f}x, margin {_rupees(need)} covered by cash"
        else:                                            # only what cash can margin
            lev = cap(1.0 + cash_inr / (margin_rate * req["equity"])) if req["equity"] > 0 else 1.0
            why = (f"futures to {lev:.2f}x only: cash {_rupees(cash_inr)} covers less than the "
                   f"{_rupees(need)} margin {target:.2f}x needs")
        if lev <= 1.0:
            return one("no funds and no cash for futures margin: 1x")
        if req.get("applied_leverage") != round(lev, 4):
            req["applied_leverage"] = round(lev, 4)
            _log(doc, today, "applied", f"{req['id']}: {why}")
        return Decision(lev, 0.0, "futures", why), doc
    return one(f"request {req['id']} is {status}: 1x")


def notice(doc: Dict[str, Any]) -> tuple:
    """(title, lines) of the email asking for funds; the book's emails carry :func:`email_line` until answered."""
    req = doc["request"]
    return (f"Funding needed for {req['book']}: {_rupees(req['extra_inr'])} by {req['deadline']}",
            [f"Why: {req['reason']}",
             f"Target {req['target']:.2f}x on equity {_rupees(req['equity'])}: {_rupees(req['extra_inr'])} more "
             "exposure.",
             "Your choices: a top-up (the bigger book runs unlevered on your cash), or no funds (futures only, "
             "margin from cash; never MTF).",
             f"Answer by {req['deadline']}: python -m nse_engine.funding_gate reply --schema {req['book']} "
             f"--request {req['id']} --answer topup --amount <rupees> | --answer no-funds",
             "No answer by then: the book stays at 1x and asks again at its next sizing change."])


def email_line(doc: Optional[Dict[str, Any]], today: date) -> Optional[str]:
    """One line for the books' emails while a request is open or acted on; None without one."""
    req = (doc or {}).get("request")
    if not req:
        return None
    if req["status"] == PENDING:
        return (f"Funding gate ({req['book']}): {_rupees(req['extra_inr'])} asked for {req['target']:.2f}x, "
                f"waiting for your reply by {req['deadline']} (request {req['id']}); 1x until then.")
    return f"Funding gate ({req['book']}): request {req['id']} {req['status'].replace('_', ' ')}."


def tracker_line(doc: Dict[str, Any], today: date) -> str:
    """The line to add to the tracker for the latest logged event."""
    e = doc.get("log", [])[-1]
    return f"{today:%d %b}  funding gate {doc['request']['book']}: {e['event']}, {e['detail']}"


# ── CLI (the owner's reply) ───────────────────────────────────────────────

def _cloud(schema: Optional[str]):
    import config  # noqa: F401 - loads .env (the database URL) on a laptop
    from database.connection import get_db_manager
    from database.paper_cloud import PaperCloudSync

    return PaperCloudSync(get_db_manager(), schema=None if schema == "deployed" else schema)   # deployed: default


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="The funding gate (tracker GTR0 decision 5)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status", help="the open request and the log")
    s.add_argument("--schema", required=True, help="the book's Neon schema, e.g. sharpe")
    r = sub.add_parser("reply", help="record your answer to a request")
    r.add_argument("--schema", required=True)
    r.add_argument("--request", required=True)
    r.add_argument("--answer", required=True, choices=("topup", "no-funds"))
    r.add_argument("--amount", type=float, help="rupees, for a top-up")
    args = p.parse_args(argv)
    cloud = _cloud(args.schema)
    doc = json.loads((cloud.read_state() or {}).get(STATE_KEY) or "{}")
    today = date.today()
    if args.cmd == "status":
        print(json.dumps(doc, indent=1) if doc else "no funding request")
        return 0
    try:
        doc = record_reply(doc, args.request, args.answer.replace("-", "_"), today, args.amount)
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    cloud.sync_state({STATE_KEY: json.dumps(doc)})
    print("recorded; add to the tracker:", tracker_line(doc, today))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
