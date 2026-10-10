#!/usr/bin/env python3
"""Email when a paper session, or the live book's, fails, so a broken run is not silent.

25 Sep 2026: every run died on a dependency change before it could reach the
book. Actions showed red, but no mail went out and the nightly heartbeat
tolerates one missed weekday as a holiday, so the failure would have gone
unnoticed until Monday - with that day's queued orders expiring meanwhile.

    python -m tools.session_failed <run-url-or-id> ["(<label>)"]

The workflow passes the run id with a label: "(live book)", "(live book <id>)"
or "(<name> book)".  A live-book failure gets its own subject and leaves out
the paper books' stale-order warning: the paper books run in their own steps
and email their own results.
"""
from __future__ import annotations

import logging
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

IST = timezone(timedelta(hours=5, minutes=30))
logger = logging.getLogger("session_failed")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arg = " ".join(sys.argv[1:]) or os.environ.get("GITHUB_RUN_ID", "unknown run")
    match = re.fullmatch(r"\s*(\S+)\s*(?:\((.*)\))?\s*", arg)
    where, label = (match.group(1), (match.group(2) or "").strip()) if match else (arg, "")
    if where.isdigit():
        repo = os.environ.get("GITHUB_REPOSITORY", "srees16/centurion_core")
        where = f"https://github.com/{repo}/actions/runs/{where}"
    now = datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")
    link = f"Open the failed run to see why: {where}"
    # Tracker AL2: a live failure is CRITICAL, a paper one a warning; each once a day per book.
    from services.notifications.alerts import CRITICAL, WARNING, alert

    if label.startswith("live book"):
        sent = alert(CRITICAL, f"session_failed:{label}", f"Centurion {label} FAILED",
                     [f"The {label} session did not complete at {now}.", link,
                      "The paper books run in their own steps and email their own results."])
    else:
        # The workflow sets this when the failure came before the live step and the live book is on.
        live_skipped = os.environ.get("CENTURION_LIVE_SKIPPED", "").lower() == "true"
        sent = alert(CRITICAL if live_skipped else WARNING, f"session_failed:{label or 'paper'}",
                     f"Centurion paper session FAILED{f' ({label})' if label else ''}"
                     + (", live book not run" if live_skipped else ""),
                     [f"The paper trading run{f' ({label})' if label else ''} did not complete at {now}.", link,
                      *(["The live book did not run either: its step comes after the one that failed."]
                        if live_skipped else []),
                      "The next run catches up: each missed session's stops and the queued orders' fills are "
                      "applied in order, then one plan is made from the latest close (no plan is made for the "
                      "sessions in between). Re-running the workflow the same evening avoids even that."])
    logger.info("failure alert %s", "sent" if sent else "not sent (already today, or check the SMTP secrets)")
    return 0                                              # never mask the real failure


if __name__ == "__main__":
    raise SystemExit(main())
