#!/usr/bin/env python3
"""Email when a paper session fails, so a broken run is not silent.

25 Sep 2026: every run died on a dependency change before it could reach the
book. Actions showed red, but no mail went out and the nightly heartbeat
tolerates one missed weekday as a holiday, so the failure would have gone
unnoticed until Monday - with that day's queued orders expiring meanwhile.

    python -m tools.session_failed <run-url-or-id>
"""
from __future__ import annotations

import logging
import os
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
    where = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GITHUB_RUN_ID", "unknown run")
    if where.isdigit():
        repo = os.environ.get("GITHUB_REPOSITORY", "srees16/centurion_core")
        where = f"https://github.com/{repo}/actions/runs/{where}"
    now = datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")
    try:
        from services.notifications.manager import NotificationManager
        sent = NotificationManager()._send_html_email(
            f"🔴 Centurion paper session FAILED - {now}",
            f"<p>The paper trading run did not complete at {now}.</p>"
            f"<p><a href=\"{where}\">Open the failed run</a> to see why.</p>"
            "<p>Orders queued at the previous close fill only at the next session's open, "
            "so a session missed on a fill day loses those orders: they are cancelled as "
            "stale on the following run. Re-running the workflow the same evening recovers them.</p>")
        logger.info("failure alert %s", "sent" if sent else "NOT sent (check the SMTP secrets)")
        return 0
    except Exception as exc:                              # noqa: BLE001 - never mask the real failure
        logger.warning("failure alert could not be sent: %s", exc)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
