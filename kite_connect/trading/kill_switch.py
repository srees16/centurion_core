"""The latched kill switch (tracker DM2): one halt for every order path, on Actions and on the HF Space.

While it is on, ``order_service.is_kill_switch_active()`` is true everywhere:
new buys are refused, while reduce-only exits and protective stop GTTs still
go (G2).  It lives in Neon (the default schema's state, ``STATE_KEY``), so the
nightly job and the web app read the same switch; the repository variable
``CENTURION_KILL_SWITCH`` still turns it on as well.  It is latched: only a
person turns it off, with a reason (the "Kill switch" workflow, or this
command with the database URL), and nothing resumes on its own.  Each change
emails the owner with its reason; the output shows only on or off, since the
Actions logs are public.

    python -m kite_connect.trading.kill_switch status
    python -m kite_connect.trading.kill_switch on --reason "..." [--by NAME]
    python -m kite_connect.trading.kill_switch off --reason "..."
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Optional

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


def set_switch(on: bool, reason: str, by: str = "") -> dict:
    """Turn the switch on or off with a reason, check it held, and email the owner; returns the new state."""
    from kite_connect.auth.daily_login import IST
    from services.notifications.alerts import CRITICAL, alert

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
    alert(CRITICAL, f"kill_switch:{doc['at']}", f"Centurion kill switch {'ON' if on else 'OFF'}",
          [f"Reason: {doc['reason']}", f"By {doc['by'] or 'unknown'} at {doc['at']}",
           "ON: new buys are refused on every order path; exits and stop GTTs still go." if on else
           "OFF: buys are allowed again (unless the repository variable CENTURION_KILL_SWITCH is true)."])
    return doc


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="The latched kill switch (tracker DM2)")
    p.add_argument("action", choices=("status", "on", "off"))
    p.add_argument("--reason", default="")
    p.add_argument("--by", default=os.environ.get("GITHUB_ACTOR", ""))
    args = p.parse_args(argv)
    doc = state() if args.action == "status" else set_switch(args.action == "on", args.reason, args.by)
    print(f"kill switch {'ON' if doc.get('on') else 'OFF'}" + (f" since {doc['at']}" if doc.get("at") else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
