#!/usr/bin/env python3
"""Has the paper book traded recently? Exit non-zero when it has not.

On 18 Sep 2026 no session ran at all: the external trigger never fired and
GitHub delivered its crons too late for the run window. Nothing noticed - the
missed-session warning only fires *inside* a run. This checks from outside and
fails the workflow, so GitHub's own failure notification reaches the owner.

A single NSE holiday must not raise the alarm, so the book is only stale when
it is more than one weekday behind.

``--ping`` is the nightly job's last step (tracker DM1): it pings the
healthchecks.io dead man's switch (``CENTURION_HEALTHCHECK_URL``) only when
the run left the books healthy (``nightly_health``), and otherwise sends its
``/fail`` ping with the reason, so a run that finished without doing its job
is reported from outside GitHub too.

    python -m tools.paper_heartbeat            # exits 1 when stale
    python -m tools.paper_heartbeat --ping     # the nightly job's dead man's switch ping
"""
from __future__ import annotations

import logging
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

IST = timezone(timedelta(hours=5, minutes=30))
logger = logging.getLogger("paper_heartbeat")


def weekdays_behind(latest: Optional[date], today: date) -> int:
    """Weekdays strictly between ``latest`` and ``today``, counting ``today`` itself.

    0 means the book is current (it traded today, or today is a weekend).
    1 is a single missed weekday, which an NSE holiday explains.
    2 or more means sessions were genuinely lost.
    """
    if latest is None:
        return 0
    n = 0
    day = today
    while day > latest:
        if day.weekday() < 5:
            n += 1
        day -= timedelta(days=1)
    return n


def check(max_behind: int = 1) -> dict:
    """Read the book's last session from Neon and judge it."""
    from database.paper_cloud import get_paper_cloud

    try:
        cloud = get_paper_cloud()
        state = (cloud.read_state() or {}) if cloud is not None else None
    except Exception as exc:                              # noqa: BLE001 - report, never traceback
        return {"ok": False, "stale": True, "reason": f"database unreachable: {exc}"[:200]}
    if cloud is None or state is None:
        return {"ok": False, "reason": "no database connection", "stale": True}
    if str(state.get("active", "")).lower() in ("false", "0") and not state.get("epoch"):
        return {"ok": True, "reason": "no book started", "stale": False}
    snaps = cloud.read_snapshots()
    latest = None
    if snaps is not None and len(snaps) and "date" in snaps.columns:
        latest = max(str(d) for d in snaps["date"])
    latest_date = date.fromisoformat(latest) if latest else None
    today = datetime.now(IST).date()
    behind = weekdays_behind(latest_date, today)
    return {
        "ok": behind <= max_behind,
        "stale": behind > max_behind,
        "latest_session": latest,
        "weekdays_behind": behind,
        "today_ist": today.isoformat(),
        "last_run_at": str(state.get("last_run_at") or ""),
        "book_writer": str(state.get("book_writer") or ""),
        "reason": ("current" if behind == 0 else
                   "one weekday behind (an NSE holiday explains this)" if behind == 1 else
                   f"{behind} weekdays behind - sessions were missed"),
    }


def nightly_health(now: Optional[datetime] = None) -> Tuple[bool, str]:
    """(healthy, why) when the nightly job ends: a heartbeat means healthy, not just alive (tracker DM1).

    Healthy: the deployed paper book reached the last completed NSE session
    and, while the live book is on (``CENTURION_LIVE_MODE`` dry_run or live),
    the live session ran for it or was skipped for no Kite login (that has its
    own CRITICAL alert, so it does not fail the heartbeat: the owner's call).
    Weekends and holidays pass by the same rule.  An unreadable book does not.
    """
    from database.connection import get_db_manager
    from database.paper_cloud import PaperCloudSync
    from kite_connect.trading.live_session import (LIVE_DRY_LAST_SESSION_KEY, LIVE_LAST_SESSION_KEY,
                                                   LIVE_SKIP_NOTIFIED_KEY, live_schema)
    from services.execution.carver_pipeline import last_completed_nse_session

    expected = last_completed_nse_session(now).isoformat()
    live_on = (os.environ.get("CENTURION_LIVE_MODE") or "").strip().lower() in ("dry_run", "live")
    try:
        mgr = get_db_manager()
        paper = PaperCloudSync(mgr, schema=None).read_state()
        live = PaperCloudSync(mgr, schema=live_schema()).read_state() if live_on else {}
    except Exception as exc:                              # noqa: BLE001 - unreadable is unhealthy
        return False, f"Neon unreadable: {exc}"[:200]
    if str(paper.get("active", "")).lower() in ("false", "0") and not paper.get("epoch"):
        return True, "no paper book started"
    done = str(paper.get("engine_last_session") or "")
    if done < expected:
        return False, f"the paper book's last session is {done or 'none'}, expected {expected}"
    if live_on:
        ran = max(str(live.get(k) or "") for k in (LIVE_LAST_SESSION_KEY, LIVE_DRY_LAST_SESSION_KEY,
                                                   LIVE_SKIP_NOTIFIED_KEY))
        if ran < expected:
            return False, f"the live session has not run for {expected} (last {ran or 'never'})"
    return True, f"session {expected} done"


def ping(url: str, healthy: bool, why: str) -> bool:
    """Ping healthchecks.io: success, or ``/fail``, with ``why`` as the body.  Never logs the URL."""
    import urllib.request

    target = url.rstrip("/") + ("" if healthy else "/fail")
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(urllib.request.Request(target, data=why.encode()[:10000], method="POST"),
                                        timeout=10):
                return True
        except Exception as exc:                          # noqa: BLE001 - retried, then reported
            logger.warning("dead man's switch ping failed (attempt %d): %s", attempt, type(exc).__name__)
            time.sleep(2 * attempt)
    return False


def send_test_email() -> int:
    """Prove the SMTP secrets work. Returns 0 when the mail was accepted.

    Gmail rejects a normal account password with "534 5.7.9 Application-specific
    password required"; an App Password is what works. Run this after changing
    the secret instead of waiting for the next session's report.
    """
    from services.notifications.manager import NotificationManager

    now = datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")
    ok = NotificationManager()._send_html_email(
        f"Centurion email check - {now}",
        f"<p>SMTP credentials work. Sent by the Paper Heartbeat workflow at {now}.</p>"
        "<p>Daily session reports will arrive from now on.</p>")
    logger.info("test email %s", "sent" if ok else "REJECTED (check CENTURION_EMAIL_PASS)")
    return 0 if ok else 1


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if "--ping" in sys.argv[1:]:                          # DM1: never fails the nightly job itself
        url = os.environ.get("CENTURION_HEALTHCHECK_URL", "")
        if not url:
            logger.info("no CENTURION_HEALTHCHECK_URL: no ping")
            return 0
        healthy, why = nightly_health()
        sent = ping(url, healthy, why)
        logger.info("dead man's switch: %s (%s)%s", "healthy" if healthy else "FAIL", why,
                    "" if sent else " - ping not delivered")
        return 0
    if os.environ.get("CENTURION_HEARTBEAT_TEST_EMAIL", "").lower() in ("true", "1", "yes"):
        return send_test_email()
    result = check(max_behind=int(os.environ.get("CENTURION_HEARTBEAT_MAX_BEHIND", "1")))
    logger.info("paper heartbeat: %s", result)
    if not result["stale"]:
        return 0
    # Tracker AL2: CRITICAL, since no session also means no live session; the exit code stays the backup alarm.
    from services.notifications.alerts import CRITICAL, alert

    alert(CRITICAL, "heartbeat_stale", f"Centurion paper book has not traded since {result.get('latest_session')}",
          [f"The paper book is {result.get('weekdays_behind')} weekdays behind (today {result.get('today_ist')} IST).",
           str(result.get("reason")), f"Last run recorded: {result.get('last_run_at') or 'never'}.",
           "Check the Paper Trading Cron workflow and the 19:00 IST dispatch."])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
