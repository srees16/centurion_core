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

import html
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
    label = html.escape(label)
    if where.isdigit():
        repo = os.environ.get("GITHUB_REPOSITORY", "srees16/centurion_core")
        where = f"https://github.com/{repo}/actions/runs/{where}"
    now = datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")
    link = f"<p><a href=\"{where}\">Open the failed run</a> to see why.</p>"
    if label.startswith("live book"):
        subject = f"Centurion {label} FAILED - {now}"
        body = (f"<p>The {label} session did not complete at {now}.</p>{link}"
                "<p>The paper books run in their own steps and email their own results.</p>")
    else:
        subject = f"Centurion paper session FAILED{f' ({label})' if label else ''} - {now}"
        body = (f"<p>The paper trading run{f' ({label})' if label else ''} did not complete at {now}.</p>{link}"
                "<p>The next run catches up: each missed session's stops and the queued orders' fills "
                "are applied in order, then one plan is made from the latest close (no plan is made "
                "for the sessions in between). Re-running the workflow the same evening avoids even that.</p>")
    try:
        from services.notifications.manager import NotificationManager
        sent = NotificationManager()._send_html_email(subject, body)
        logger.info("failure alert %s", "sent" if sent else "NOT sent (check the SMTP secrets)")
        return 0
    except Exception as exc:                              # noqa: BLE001 - never mask the real failure
        logger.warning("failure alert could not be sent: %s", exc)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
