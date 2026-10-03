"""The scheduler's SQLite run cache and job log, which the REST API reads.

Moved from scheduler.py (tracker H4).
"""

from __future__ import annotations

import functools
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# â”€â”€ Constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_IST = timezone(timedelta(hours=5, minutes=30))
_DB_PATH = Path(__file__).parent.parent / "data" / "scheduler_cache.sqlite3"


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Cache layer (SQLite â€” lightweight, no external DB dependency)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _init_cache_db():
    """Create the scheduler cache table if it doesn't exist."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_type    TEXT NOT NULL,          -- 'pre_market'
            timestamp   TEXT NOT NULL,
            universe_size  INTEGER DEFAULT 0,
            screened_count INTEGER DEFAULT 0,
            buy_signals    INTEGER DEFAULT 0,
            sell_signals   INTEGER DEFAULT 0,
            verdicts_json  TEXT,                -- JSON array of verdict summaries
            plans_json     TEXT,                -- JSON array of trade plan summaries
            status      TEXT DEFAULT 'success'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS job_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id      TEXT NOT NULL,
            job_name    TEXT NOT NULL,
            started_at  TEXT NOT NULL,
            finished_at TEXT,
            status      TEXT DEFAULT 'running',
            detail      TEXT
        )
    """)
    conn.commit()
    conn.close()


def _save_run(run_type: str, summary: dict):
    """Persist a pipeline run result to the cache."""
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute(
        """INSERT INTO pipeline_runs
           (run_type, timestamp, universe_size, screened_count,
            buy_signals, sell_signals, verdicts_json, plans_json, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_type,
            datetime.now(_IST).isoformat(),
            summary.get("universe_size", 0),
            summary.get("screened_count", 0),
            summary.get("buy_signals", 0),
            summary.get("sell_signals", 0),
            json.dumps(summary.get("verdicts", []), default=str),
            json.dumps(summary.get("plans", []), default=str),
            summary.get("status", "success"),
        ),
    )
    conn.commit()
    conn.close()


def get_latest_run(run_type: Optional[str] = None) -> Optional[dict]:
    """Read the most recent pipeline run from cache.

    This is called by the REST API to display
    the latest scheduled scan results without re-running.
    """
    if not _DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    if run_type:
        row = conn.execute(
            "SELECT * FROM pipeline_runs WHERE run_type=? ORDER BY id DESC LIMIT 1",
            (run_type,),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM pipeline_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
    conn.close()
    if row is None:
        return None
    return dict(row)


def _log_job_start(job_id: str, job_name: str) -> int:
    """Record that a scheduler job started. Returns the row id."""
    try:
        conn = sqlite3.connect(str(_DB_PATH))
        cur = conn.execute(
            "INSERT INTO job_log (job_id, job_name, started_at, status) VALUES (?, ?, ?, 'running')",
            (job_id, job_name, datetime.now(_IST).isoformat()),
        )
        row_id = cur.lastrowid
        conn.commit()
        conn.close()
        return row_id
    except Exception:
        return -1


def _log_job_end(row_id: int, status: str = "ok", detail: str = ""):
    """Mark a scheduler job as finished."""
    if row_id < 0:
        return
    try:
        conn = sqlite3.connect(str(_DB_PATH))
        conn.execute(
            "UPDATE job_log SET finished_at=?, status=?, detail=? WHERE id=?",
            (datetime.now(_IST).isoformat(), status, detail[:500] if detail else "", row_id),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def get_job_log(limit: int = 50, job_id: str | None = None) -> list:
    """Return the most recent job log entries, optionally filtered by job_id."""
    if not _DB_PATH.exists():
        return []
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    if job_id:
        rows = conn.execute(
            "SELECT * FROM job_log WHERE job_id = ? ORDER BY id DESC LIMIT ?",
            (job_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM job_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════════
# Verdict caching helpers
# ═══════════════════════════════════════════════════════════════

def get_cached_verdict(ticker: str):
    """Read a single cached verdict dict, or None if miss/expired."""
    try:
        from infrastructure.cache import cache
        return cache.get(f"verdict:{ticker}")
    except Exception:
        return None


def _tracked_job(job_id: str, job_name: str):
    """Decorator that logs job start/end to the job_log table."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            row_id = _log_job_start(job_id, job_name)
            try:
                result = fn(*args, **kwargs)
                _log_job_end(row_id, status="ok")
                return result
            except Exception as exc:
                _log_job_end(row_id, status="error", detail=str(exc))
                raise
        return wrapper
    return decorator
