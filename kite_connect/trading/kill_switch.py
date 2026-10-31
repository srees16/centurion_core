"""The latched kill switch (tracker DM2): one halt for every order path, on Actions and on the HF Space.

While it is on, ``order_service.is_kill_switch_active()`` is true everywhere:
new buys are refused, while reduce-only exits and protective stop GTTs still
go (G2).  It lives in Neon (the default schema's state, ``STATE_KEY``), so the
nightly job and the web app read the same switch; the repository variable
``CENTURION_KILL_SWITCH`` still turns it on as well.  It is latched: only a
person turns it off, with a reason (the "Kill switch" workflow, or this
command with the database URL), and nothing resumes on its own.  Each change
emails the owner with its reason; the output shows only on or off and counts,
since the Actions logs are public.

Turning it on also clears the buys already at the broker (tracker RB6): an
after-market BUY queued that evening would otherwise execute at the next open.
The latch comes first, so no failure at Kite can stop the halt, and ``drain``
repeats the clearing, for instance after the day's login, since no call reaches
Kite between 06:00 IST and that login.  ``check`` reads each book and cancels
nothing.  A cancelled buy is not placed again: the next session plans afresh.

    python -m kite_connect.trading.kill_switch status
    python -m kite_connect.trading.kill_switch on --reason "..." [--by NAME]
    python -m kite_connect.trading.kill_switch off --reason "..."
    python -m kite_connect.trading.kill_switch drain
    python -m kite_connect.trading.kill_switch check
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import List, Optional

STATE_KEY = "kill_switch"       # JSON {"on": bool, "reason": str, "at": IST timestamp, "by": str}


def _store():
    """The deployed book's Neon store (default schema), or None when no database is configured."""
    if not (os.environ.get("CENTURION_DATABASE_URL") or os.environ.get("DATABASE_URL")):
        return None
    from database.connection import get_db_manager
    from database.paper_cloud import PaperCloudSync

    return PaperCloudSync(get_db_manager(), schema=None)


def state() -> dict:
    """The latched switch, ``{"on": False}`` without a database; RAISES when the database cannot be read."""
    store = _store()
    if store is None:
        return {"on": False}
    return json.loads(store.read_state().get(STATE_KEY) or '{"on": false}')


def drain(dry_run: bool = False) -> List[dict]:
    """Clear Centurion's working buys from every account (tracker RB6): one result per account, and it never raises.

    A ``live`` account with no login today is reported, as yours always is, because its book cannot be read; any other
    account without one is passed over, since Centurion sends it no real orders.
    """
    from kite_connect.auth import accounts, daily_login
    from kite_connect.trading.order_service import drain_buys

    try:
        listed = accounts.all_accounts()
    except Exception as exc:                              # noqa: BLE001 - reported, never raised
        return [{"account": "every account", "status": "UNKNOWN", "error": f"the account list could not be read ({exc})"}]
    results = []
    for acct in listed:
        try:
            kite = daily_login.kite_from_stored_token(accounts.account_book(acct), key=acct.api_key)
            if kite is not None:
                results.append({"account": acct.name, **drain_buys(kite, dry_run=dry_run)})
            elif acct.is_primary or acct.mode == "live":
                results.append({"account": acct.name, "status": "UNKNOWN", "error": "no Kite login today"})
        except Exception as exc:                          # noqa: BLE001 - one account must not stop the others
            results.append({"account": acct.name, "status": "UNKNOWN", "error": str(exc)})
    return results


def _broker_lines(results: List[dict]) -> List[str]:
    """The email's lines on the broker: what happened, and what the owner must do when a buy may still execute."""
    lines = []
    for r in results:
        who = r["account"]
        if r["status"] == "UNKNOWN":
            lines.append(f"NOT CHECKED at the broker ({who}): {r['error']}. A buy queued there still executes: cancel it "
                         "in Kite, or log in and run the Kill switch workflow with action drain.")
        elif r["status"] == "LEFT":
            lines.append(f"STILL OPEN at the broker ({who}): {', '.join(r['left'])}. Cancel in Kite now"
                         + (f" (Kite said: {r['error']})." if r.get("error") else "."))
        elif r["cancelled"]:
            lines.append(f"Broker ({who}): {r['cancelled']} queued buy(s) cancelled and checked gone.")
        else:
            lines.append(f"Broker ({who}): no queued buys.")
        if r.get("filled"):
            lines.append(f"Executed before the cancel ({who}): {', '.join(r['filled'])}. Those shares are held, and the "
                         "next session's ledger takes them in.")
    return lines


def _report(results: List[dict]) -> int:
    """Print the broker's result as counts (the Actions log is public); 1 when a buy may still execute."""
    print(f"broker: {sum(r.get('found', 0) for r in results)} buys found, "
          f"{sum(r.get('cancelled', 0) for r in results)} cancelled, "
          f"{sum(len(r.get('left', [])) for r in results)} open, "
          f"{sum(r['status'] == 'UNKNOWN' for r in results)} account(s) not checked")
    return int(any(r["status"] not in ("CLEAR", "CHECKED") for r in results))


def _alert_drain(results: List[dict], title: str, key: str, lines: List[str]) -> None:
    """Email the result, with CHECK KITE NOW in the subject unless every account is clear."""
    from services.notifications.alerts import CRITICAL, alert

    clear = all(r["status"] == "CLEAR" for r in results)
    alert(CRITICAL, key, title + ("" if clear else ": CHECK KITE NOW"), lines + _broker_lines(results))


def set_switch(on: bool, reason: str, by: str = "") -> dict:
    """Turn the switch on or off with a reason, check it held, clear the broker's queued buys when on, and email the
    owner; returns the new state, with ``drain`` holding each account's result."""
    from kite_connect.auth.daily_login import IST

    if not (reason or "").strip():
        raise ValueError("a reason is required")
    store = _store()
    if store is None:
        raise RuntimeError("no database: set CENTURION_DATABASE_URL")
    doc = {"on": bool(on), "reason": reason.strip()[:500], "at": datetime.now(IST).isoformat(timespec="seconds"),
           "by": (by or "")[:100]}
    store.sync_state({STATE_KEY: json.dumps(doc)})
    if state().get("on") is not bool(on):
        raise RuntimeError("the kill switch did not persist in Neon: try again")
    drained = drain() if on else []
    _alert_drain(drained, f"Centurion kill switch {'ON' if on else 'OFF'}", f"kill_switch:{doc['at']}",
                 [f"Reason: {doc['reason']}", f"By {doc['by'] or 'unknown'} at {doc['at']}",
                  "ON: new buys are refused on every order path; exits and stop GTTs still go." if on else
                  "OFF: buys are allowed again (unless the repository variable CENTURION_KILL_SWITCH is true)."])
    return {**doc, "drain": drained}


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="The latched kill switch (tracker DM2)")
    p.add_argument("action", choices=("status", "on", "off", "drain", "check"))
    p.add_argument("--reason", default="")
    p.add_argument("--by", default=os.environ.get("GITHUB_ACTOR", ""))
    args = p.parse_args(argv)
    if args.action == "check":
        return _report(drain(dry_run=True))
    if args.action == "drain":
        from kite_connect.auth.daily_login import IST
        from kite_connect.trading.order_service import is_kill_switch_active

        if not is_kill_switch_active():
            print("kill switch OFF: nothing drained")
            return 1
        results, at = drain(), datetime.now(IST).isoformat(timespec="seconds")
        _alert_drain(results, "Centurion kill switch drain", f"kill_switch_drain:{at}", [f"By {args.by or 'unknown'} at {at}"])
        return _report(results)
    doc = state() if args.action == "status" else set_switch(args.action == "on", args.reason, args.by)
    print(f"kill switch {'ON' if doc.get('on') else 'OFF'}" + (f" since {doc['at']}" if doc.get("at") else ""))
    return _report(doc["drain"]) if doc.get("drain") else 0


if __name__ == "__main__":
    raise SystemExit(main())
