"""Severity-routed alerts to the owner (tracker AL1).

The nightly emails carry many lines; a few events need the owner's attention
now.  ``alert()`` sends each of those as its own email, the severity first in
the subject so one mail rule can surface them:

* ``CRITICAL`` - ``[Centurion CRITICAL] <title>``: act today (a failed live
  session, a kill criterion, a position without a stop).  Until the live book
  trades real money (``CENTURION_LIVE_MODE`` other than ``live``) the subject
  reads ``[Centurion would-be CRITICAL]``: the dry runs show what would have
  needed the owner, without the urgency.
* ``WARNING`` - ``[Centurion warning] <title>``: look when convenient.

A ``key`` alerts at most once per IST day: the keys sent lately live in the
book's Neon state, so the backup runs do not repeat them (without Neon, every
call sends; an unreadable history sends rather than risks a missed alert).
Delivery is best effort: a failure is logged, never raised.  Email only, to
the owner (``NotificationManager``'s default recipient); a push channel is
deferred (tracker AL6).

    from services.notifications.alerts import CRITICAL, alert
    alert(CRITICAL, "live_session_failed", "Live session failed", ["TokenException: ..."])
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from html import escape
from typing import Dict, Optional, Sequence, Tuple

from kite_connect.auth.daily_login import ENV_LIVE_MODE, IST

logger = logging.getLogger(__name__)

CRITICAL = "CRITICAL"
WARNING = "WARNING"
#: Neon state key of the alerts sent lately: {key: IST date of its last email}.
STATE_KEY = "alerts_sent"
KEEP_DAYS = 7


def subject(severity: str, title: str) -> str:
    """The email subject: the severity first, so one mail rule can surface it."""
    if severity == CRITICAL:
        live = (os.environ.get(ENV_LIVE_MODE) or "").strip().lower() == "live"
        return f"[Centurion {'CRITICAL' if live else 'would-be CRITICAL'}] {title}"
    return f"[Centurion warning] {title}"


def _run_link() -> str:
    """The GitHub Actions run that raised the alert, when there is one."""
    server, repo, run = (os.environ.get(k) for k in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID"))
    return f"{server}/{repo}/actions/runs/{run}" if server and repo and run else ""


def _history() -> Tuple[Optional[object], Dict[str, str]]:
    """(the book's Neon store or None, {key: IST date}) of the alerts sent lately."""
    try:
        from database.paper_cloud import get_paper_cloud

        cloud = get_paper_cloud()
        if cloud is None:
            return None, {}
        return cloud, json.loads((cloud.read_state() or {}).get(STATE_KEY) or "{}")
    except Exception as exc:                              # noqa: BLE001 - better a repeat than a missed alert
        logger.warning("alert history unreadable (%s): sending without the once-a-day check", exc)
        return None, {}


def alert(severity: str, key: str, title: str, lines: Sequence[str] = (), book: Optional[str] = None) -> bool:
    """Email one alert unless ``key`` already alerted today (IST); True when it was sent."""
    if severity not in (CRITICAL, WARNING):
        raise ValueError(f"severity must be {CRITICAL} or {WARNING}, not {severity!r}")
    now = datetime.now(IST)
    today = now.date().isoformat()
    cloud, sent = _history()
    if sent.get(key) == today:
        logger.info("alert %s already sent today", key)
        return False
    run = _run_link()
    items = "".join(f"<li>{escape(str(line))}</li>" for line in lines)
    html = (f"<html><body style=\"font-family:Segoe UI,Arial,sans-serif;\"><h3>{escape(title)}</h3>"
            + (f"<ul>{items}</ul>" if items else "")
            + f"<p style=\"color:#6b7280;font-size:12px;\">{now:%d %b %Y %H:%M} IST"
            + (f" &middot; book {escape(book)}" if book else "")
            + (f" &middot; <a href=\"{escape(run)}\">run</a>" if run else "") + "</p></body></html>")
    try:
        from services.notifications.manager import NotificationManager

        ok = NotificationManager._send_html_email(subject(severity, title), html)
    except Exception as exc:                              # noqa: BLE001 - an alert never stops its caller
        logger.warning("alert %s not sent: %s", key, exc)
        return False
    if not ok:
        logger.warning("alert %s not sent: email unavailable", key)
        return False
    if cloud is not None:
        cutoff = (now.date() - timedelta(days=KEEP_DAYS)).isoformat()
        kept = {k: d for k, d in sent.items() if d >= cutoff}
        kept[key] = today
        try:
            cloud.sync_state({STATE_KEY: json.dumps(kept, sort_keys=True)})
        except Exception as exc:                          # noqa: BLE001 - sent; at worst it repeats
            logger.warning("alert history not saved (%s): %s may repeat today", exc, key)
    return True
