"""
Background scheduler for the HF Space (``deployment/start.sh``).

Its live jobs dispatch GitHub Actions on time: the nightly paper/live
workflow (19:00 IST, retry 20:30) and the Kite login reminder (09:00 and
17:30 IST).  The legacy pipeline jobs are removed (tracker H1/H3,
docs/scheduler_audit.md); the strategy-maintenance and options/futures job
functions are kept, unregistered, for later use.  The cache, the job log, the
walk-forward audit and those job functions live in ``scheduling/`` (tracker H4).

Usage::

    # Activate virtualenv first, then:
    python scheduler.py

    # Or, detached (PowerShell):
    # Start-Process python -ArgumentList "scheduler.py" -WindowStyle Hidden

Requires: ``pip install apscheduler``

The SQLite cache and job log here are also read by the REST API, which runs
the walk-forward audit on demand.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from scheduling.cache import (_IST, _init_cache_db, _save_run, _tracked_job, get_cached_verdict, get_job_log,
                              get_latest_run)
from scheduling.walk_forward import run_walk_forward_audit

# Other code imports these from here: api/routers (health, pipeline, v1/verdict) and tests.
__all__ = ["start_scheduler", "get_cached_verdict", "get_job_log", "get_latest_run", "run_walk_forward_audit"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
)
logger = logging.getLogger("centurion.scheduler")

# â”€â”€ Ensure project root is on sys.path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_ROOT = str(Path(__file__).parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# GitHub Actions dispatch (the scheduler's live jobs)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _github_dispatch(workflow: str, inputs: dict, token: str) -> int:
    """Start a GitHub Actions workflow by ``workflow_dispatch``; returns the HTTP status (204 = started)."""
    import urllib.request

    repo = os.environ.get("CENTURION_GH_REPO", "srees16/centurion_core")
    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/dispatches"
    body = json.dumps({"ref": os.environ.get("CENTURION_GH_REF", "main"), "inputs": inputs}).encode()
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status


@_tracked_job("kite_login_reminder_dispatch", "Kite Login Reminder Dispatch")
def _dispatch_kite_login_reminder():
    """Start the Kite login reminder workflow at 09:00 / 17:30 IST, on time (U23).

    GitHub's cron delivered it 5-7 hours late (29-30 Sep 2026: 15:30 and 23:15
    IST), after the 19:00 session.  The workflow decides whether to email: live
    mode on, an NSE trading day, and no token for today yet.
    """
    import urllib.error

    token = os.environ.get("CENTURION_GH_DISPATCH_TOKEN", "")
    if not token:
        return
    try:
        http = _github_dispatch("kite-login-reminder.yml", {}, token)
        logger.info("Kite login reminder dispatch: HTTP %s", http)
        _save_run("kite_login_reminder_dispatch", {"status": "success" if http == 204 else "error", "http": http})
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:200].decode("utf-8", "replace")
        logger.error("Kite login reminder dispatch failed: HTTP %s %s", exc.code, detail)
        _save_run("kite_login_reminder_dispatch", {"status": "error", "http": exc.code, "detail": detail})
    except Exception as exc:                              # noqa: BLE001 - never kill the scheduler
        logger.error("Kite login reminder dispatch failed: %s", exc)
        _save_run("kite_login_reminder_dispatch", {"status": "error", "detail": str(exc)})


@_tracked_job("nse_engine_dispatch", "NSE Engine Dispatch")
def _dispatch_nse_paper_workflow():
    """Start the GitHub Actions paper session at 19:00 IST, on time.

    GitHub's own cron delivers 1-4 hours late and sometimes not at all (18 Sep
    2026: no scheduled run arrived, and the session had to be started by hand),
    so the punctual trigger lives here, where APScheduler fires to the minute.
    The Actions crons stay as backups.

    Needs ``CENTURION_GH_DISPATCH_TOKEN`` (a fine-grained token with Actions:
    read and write on the repository). Without it the job does nothing, so a
    Space without the secret is simply quiet.
    """
    import urllib.error

    def report(status: str, detail: str = "") -> None:
        """Leave a breadcrumb in Neon: the Space's own logs are not reachable
        from outside, so without this a silent day cannot be told apart from a
        day the Space never tried.  A failed dispatch also warns the owner
        (tracker AL2)."""
        try:
            from database.paper_cloud import get_paper_cloud
            cloud = get_paper_cloud()
            if cloud:
                cloud.sync_state({"nse_dispatch_at": datetime.now(timezone.utc).isoformat(),
                                  "nse_dispatch_status": status,
                                  "nse_dispatch_detail": str(detail)[:200]})
        except Exception as exc:                          # noqa: BLE001 - reporting only
            logger.debug("dispatch breadcrumb failed: %s", exc)
        if status not in ("dispatched", "no_token"):
            from services.notifications.alerts import WARNING, alert

            alert(WARNING, "nse_dispatch_failed", "Centurion: the HF Space could not start the nightly job",
                  [f"{status}: {str(detail)[:200]}", "GitHub's own crons (20:05, 21:05, 22:05 IST) are the backup; "
                   "the healthchecks.io switch emails you if no run finishes by 23:00 IST."])

    token = os.environ.get("CENTURION_GH_DISPATCH_TOKEN", "")
    if not token:
        logger.debug("NSE paper dispatch: no CENTURION_GH_DISPATCH_TOKEN, skipping")
        report("no_token")
        return
    # The retry at 20:30 IST does nothing when the 19:00 attempt already worked.
    try:
        from database.paper_cloud import get_paper_cloud
        cloud = get_paper_cloud()
        done = str((cloud.read_state() or {}).get("engine_last_session") or "") if cloud else ""
    except Exception:                                     # noqa: BLE001 - attempt anyway
        done = ""
    today_ist = datetime.now(_IST).date().isoformat()
    if done and done >= today_ist:
        logger.info("NSE paper dispatch: session %s already processed, skipping", done)
        return
    workflow = os.environ.get("CENTURION_GH_WORKFLOW", "nse-paper-trading.yml")
    try:
        http = _github_dispatch(workflow, {"reason": "hf scheduler 19:00 IST"}, token)
        ok = http == 204
        logger.info("NSE paper dispatch: %s (HTTP %s)", "started" if ok else "unexpected status", http)
        _save_run("nse_engine_dispatch", {"status": "success" if ok else "error", "http": http})
        report("dispatched" if ok else "unexpected_status", f"HTTP {http}")
    except urllib.error.HTTPError as exc:                 # 401 token, 403 scope, 404 path, 422 body
        detail = exc.read()[:200].decode("utf-8", "replace")
        logger.error("NSE paper dispatch failed: HTTP %s %s", exc.code, detail)
        _save_run("nse_engine_dispatch", {"status": "error", "http": exc.code, "detail": detail})
        report(f"http_{exc.code}", detail)
    except Exception as exc:                              # noqa: BLE001 - never kill the scheduler
        logger.error("NSE paper dispatch failed: %s", exc)
        _save_run("nse_engine_dispatch", {"status": "error", "detail": str(exc)})
        report("error", str(exc))


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Scheduler setup
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def start_scheduler():
    """Start the APScheduler background scheduler on the HF Space.

    Only GitHub Actions dispatches run here, all behind
    ``CENTURION_GH_DISPATCH_TOKEN``: the nightly paper/live workflow at 19:00
    IST (retry 20:30) and the Kite login reminder at 09:00 and 17:30 IST.  The
    trading runs in GitHub Actions; the legacy jobs, several of which logged
    in to Kite headlessly, are retired (tracker H1, docs/scheduler_audit.md).
    """
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.error("APScheduler not installed. Run: pip install apscheduler")
        return

    _init_cache_db()

    scheduler = BlockingScheduler(timezone="Asia/Kolkata")

    # ── NSE paper session: dispatch GitHub Actions at 19:00 IST, on time ──
    if os.environ.get("CENTURION_GH_DISPATCH_TOKEN"):
        scheduler.add_job(
            _dispatch_nse_paper_workflow,
            CronTrigger(hour=19, minute=0, day_of_week="mon-fri", timezone="Asia/Kolkata"),
            id="nse_engine_dispatch",
            name="NSE Engine Dispatch",
            misfire_grace_time=3600,          # a Space restart near 19:00 still fires
        )
        scheduler.add_job(
            _dispatch_nse_paper_workflow,
            CronTrigger(hour=20, minute=30, day_of_week="mon-fri", timezone="Asia/Kolkata"),
            id="nse_engine_dispatch_retry",
            name="NSE Engine Dispatch (retry)",
            misfire_grace_time=3600,
        )
        logger.info("  NSE paper start : 19:00 IST + retry 20:30 IST, Mon-Fri (GitHub Actions dispatch)")
        for _job_id, _hour, _minute in (("kite_login_reminder", 9, 0), ("kite_login_reminder_evening", 17, 30)):
            scheduler.add_job(
                _dispatch_kite_login_reminder,
                CronTrigger(hour=_hour, minute=_minute, day_of_week="mon-fri", timezone="Asia/Kolkata"),
                id=_job_id,
                name="Kite Login Reminder Dispatch",
                misfire_grace_time=1800,
            )
        logger.info("  Kite login mail : 09:00 + 17:30 IST, Mon-Fri (GitHub Actions dispatch)")
    else:
        logger.warning("CENTURION_GH_DISPATCH_TOKEN is not set: the scheduler has no jobs")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CLI entry point
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

if __name__ == "__main__":
    start_scheduler()
