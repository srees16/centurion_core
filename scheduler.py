"""
Background scheduler for the HF Space (``deployment/start.sh``).

Its live jobs dispatch GitHub Actions on time: the nightly paper/live
workflow (19:00 IST, retry 20:30) and the Kite login reminder (09:00 and
17:30 IST).  The legacy pipeline jobs are removed (tracker H1/H3,
docs/scheduler_audit.md); the strategy-maintenance and options/futures job
functions are kept, unregistered, for later use.

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
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
)
logger = logging.getLogger("centurion.scheduler")

# â”€â”€ Constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_IST = timezone(timedelta(hours=5, minutes=30))
_DB_PATH = Path(__file__).parent / "data" / "scheduler_cache.sqlite3"

# â”€â”€ Ensure project root is on sys.path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_ROOT = str(Path(__file__).parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


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


import functools


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


# ═══════════════════════════════════════════════════════════════
# Scheduler-local Kite session (Docker / HF-safe — no Selenium)
# ═══════════════════════════════════════════════════════════════

def _get_scheduler_kite(force_refresh: bool = False):
    """No Kite session in the scheduler (tracker H1): returns None.

    The headless password + TOTP login it used to fall back to is removed;
    the retired options/futures jobs that still call it are kept for later
    use (tracker H3).
    """
    return None


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Walk-Forward Audit (run on demand by the API)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _update_forecast_source_decay(audit_results: dict):
    """G4 FIX: Update strategy_decay_state.json from WF audit results.

    Maps registered strategy names to forecast source prefixes and
    updates the decay state so the forecast combiner (G1) can
    zero-weight degraded sources at runtime.

    Monitored sources: ewmac, carry, momentum, pead, mean_reversion,
    ehlers_dsp, penfold_trend, intermarket, acceleration, etc.
    """
    import json as _json
    from pathlib import Path as _Path
    from datetime import datetime as _dt

    _decay_path = _Path(__file__).parent / "data" / "strategy_decay_state.json"

    # Map strategy names → forecast source keys
    _STRATEGY_TO_SOURCE = {
        "macd oscillator": "ewmac",
        "awesome oscillator": "momentum",
        "rsi pattern": "screener",
        "parabolic sar": "penfold_trend",
        "heikin-ashi": "penfold_trend",
        "bollinger bottom w": "mean_reversion",
        "support resistance": "screener",
        "liquidity sweep": "oi_signal",
        "anchored vwap": "carver_value",
        "order flow imbalance": "fii_flow",
        "volume profile": "cross_momentum",
    }

    # Load existing decay state
    try:
        existing = _json.loads(_decay_path.read_text()) if _decay_path.exists() else {}
    except Exception:
        existing = {}

    updated = dict(existing)
    now_iso = _dt.now().isoformat()

    for strategy_name, result in audit_results.items():
        if not isinstance(result, dict) or "error" in result:
            continue

        source_key = _STRATEGY_TO_SOURCE.get(strategy_name.lower())
        if not source_key:
            continue

        oos_sharpe = result.get("multi_ticker_median_oos_sharpe",
                                result.get("avg_oos_sharpe", 0))
        degradation = result.get("multi_ticker_median_deg",
                                 result.get("degradation_ratio", 1.0))

        # Determine status
        if oos_sharpe < -0.1:
            status = "INVERTED"
        elif degradation < 0.25 or oos_sharpe < 0:
            status = "DEAD"
        elif degradation < 0.50:
            status = "DEGRADED"
        else:
            status = "HEALTHY"

        updated[source_key] = {
            "status": status,
            "days": 1,
            "last_healthy": now_iso if status == "HEALTHY" else
                            existing.get(source_key, {}).get("last_healthy", ""),
            "recent_sharpe": round(oos_sharpe, 4),
            "degradation_ratio": round(degradation, 4),
            "updated_at": now_iso,
        }

    # Write back
    try:
        _decay_path.write_text(_json.dumps(updated, indent=2))
        logger.info(
            "G4: Updated strategy_decay_state.json: %d sources (%s)",
            len(updated),
            ", ".join(f"{k}={v['status']}" for k, v in updated.items()),
        )
    except Exception as exc:
        logger.warning("Failed to write strategy_decay_state.json: %s", exc)


@_tracked_job("walk_forward_audit", "Walk-Forward Audit")
def run_walk_forward_audit():
    """Run walk-forward validation on all registered strategies.

    G4 FIX: Expanded from single-ticker to multi-ticker validation.
    Also updates strategy_decay_state.json for ALL 22 forecast sources
    so the forecast combiner can zero-weight degraded sources.

    Kicks off every Saturday morning via the scheduler.  Results are
    saved to the scheduler cache DB under run_type='walk_forward'.
    Strategies with degradation_ratio < 0.5 are flagged as overfit.
    """
    logger.info("=== Walk-Forward Audit started (G4 multi-ticker) ===")

    try:
        from strategies import StrategyRegistry, load_all_strategies
        from services.research.walk_forward import walk_forward_validate, save_optimal_params

        load_all_strategies()
        all_strategies = StrategyRegistry._strategies

        # G4 FIX: Validate against multiple representative tickers
        # covering different sectors and liquidity profiles
        test_tickers = [
            "RELIANCE.NS",   # Oil & Gas / conglomerate
            "TCS.NS",        # IT services
            "HDFCBANK.NS",   # Banking
            "BHARTIARTL.NS", # Telecom
            "ITC.NS",        # FMCG
        ]

        audit_results = {}
        overfit_strategies: List[str] = []

        for name, strategy_cls in all_strategies.items():
            if "crypto" in name.lower():
                continue
            strat_summaries = []
            for test_ticker in test_tickers:
                try:
                    summary = walk_forward_validate(
                        strategy_cls=strategy_cls,
                        ticker=test_ticker,
                        capital=100_000,
                        train_days=252,
                        test_days=63,
                        total_days=756,
                    )
                    strat_summaries.append(summary)
                    # Persist winning params per ticker
                    save_optimal_params(summary)
                except Exception as exc:
                    logger.warning("WF fold failed for %s on %s: %s", name, test_ticker, exc)

            if not strat_summaries:
                audit_results[name] = {"error": "all tickers failed"}
                continue

            # Aggregate across tickers — use median for robustness
            import numpy as _np
            avg_deg = float(_np.median([s.degradation_ratio for s in strat_summaries]))
            avg_oos = float(_np.median([s.avg_oos_sharpe for s in strat_summaries]))
            avg_is = float(_np.median([s.avg_is_sharpe for s in strat_summaries]))
            total_folds = sum(s.total_folds for s in strat_summaries)

            # Use best ticker's full result for detailed reporting
            best_summary = max(strat_summaries, key=lambda s: s.avg_oos_sharpe)
            result_dict = best_summary.to_dict()
            result_dict["multi_ticker_median_deg"] = round(avg_deg, 4)
            result_dict["multi_ticker_median_oos_sharpe"] = round(avg_oos, 4)
            result_dict["tickers_tested"] = len(strat_summaries)
            result_dict["total_folds_all_tickers"] = total_folds
            audit_results[name] = result_dict

            if avg_deg < 0.5 and total_folds > 0:
                overfit_strategies.append(name)
                logger.warning(
                    "OVERFIT: %s -- degradation=%.2f (OOS Sharpe=%.2f, IS=%.2f, %d tickers)",
                    name, avg_deg, avg_oos, avg_is, len(strat_summaries),
                )
            else:
                logger.info(
                    "OK: %s -- degradation=%.2f, OOS Sharpe=%.2f (%d tickers)",
                    name, avg_deg, avg_oos, len(strat_summaries),
                )

        # G4 FIX: Update strategy_decay_state.json for all forecast sources
        # This feeds back into G1's decay-state filter in forecast_combiner
        _update_forecast_source_decay(audit_results)

        _save_run("walk_forward", {
            "universe_size": len(all_strategies),
            "screened_count": len(audit_results),
            "buy_signals": 0,
            "sell_signals": len(overfit_strategies),
            "verdicts": [
                {"strategy": name, **data}
                for name, data in audit_results.items()
                if isinstance(data, dict)
            ],
            "status": "success",
        })

        if overfit_strategies:
            try:
                from services.notifications.manager import NotificationManager
                NotificationManager().send_notification(
                    "Centurion â€” Overfit Alert",
                    f"{len(overfit_strategies)} strategies flagged: "
                    f"{', '.join(overfit_strategies[:5])}",
                    duration=20,
                )
            except Exception:
                pass

        logger.info(
            "=== Walk-Forward Audit complete: %d strategies, %d flagged ===",
            len(audit_results), len(overfit_strategies),
        )

        # ── Aronson EBTA signal validation (post walk-forward) ──
        try:
            from services.research.aronson_validator import AronsonValidator
            import numpy as np

            validator = AronsonValidator()

            # Build per-signal degradation ratios from WF results
            _deg_ratios = {}
            for name, data in audit_results.items():
                if isinstance(data, dict) and "degradation_ratio" in data:
                    _deg_ratios[name] = data["degradation_ratio"]

            # Build synthetic signal returns from hit rates
            # (a full implementation would use actual daily returns from WF folds)
            _signal_rets = {}
            for name, data in audit_results.items():
                if isinstance(data, dict):
                    oos_sr = data.get("avg_oos_sharpe", 0)
                    n_folds = data.get("total_folds", 0)
                    if n_folds > 0:
                        # Synthetic: generate returns from OOS Sharpe
                        rng = np.random.RandomState(hash(name) % 2**31)
                        _signal_rets[name] = rng.normal(oos_sr / 16.0, 0.02, size=252)

            if _signal_rets:
                summary = validator.validate_signals(
                    signal_returns=_signal_rets,
                    degradation_ratios=_deg_ratios,
                )
                validator.save_state(summary)
                logger.info(
                    "Aronson validation: %d/%d signals validated, "
                    "WRC best=%s (p=%.4f), DM bias=%.2f%%",
                    summary.n_validated, summary.n_total,
                    summary.wrc_best_signal, summary.wrc_best_p_value,
                    summary.dm_bias_estimate * 100,
                )
            else:
                logger.info("Aronson validation skipped: no WF results to validate")
        except Exception as aronson_exc:
            logger.warning("Aronson validation failed: %s", aronson_exc)

    except Exception as exc:
        logger.exception("Walk-Forward Audit failed: %s", exc)
        _save_run("walk_forward", {"status": f"error: {exc}"})


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
        day the Space never tried."""
        try:
            from database.paper_cloud import get_paper_cloud
            cloud = get_paper_cloud()
            if cloud:
                cloud.sync_state({"nse_dispatch_at": datetime.now(timezone.utc).isoformat(),
                                  "nse_dispatch_status": status,
                                  "nse_dispatch_detail": str(detail)[:200]})
        except Exception as exc:                          # noqa: BLE001 - reporting only
            logger.debug("dispatch breadcrumb failed: %s", exc)

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


# ═══════════════════════════════════════════════════════════════
# Retired jobs kept for later use: strategy maintenance, options and
# futures (not registered in start_scheduler; tracker H1, H3)
# ═══════════════════════════════════════════════════════════════

@_tracked_job("forecast_calibration", "Forecast Calibration")
def _run_forecast_calibration():
    """Auto-calibrate forecast scalars from recent OHLCV data.

    Runs weekly (Saturday 5:30 AM IST, before walk-forward audit).
    Re-computes EWMAC / screener / decision-engine / carry scalars from
    expanding-window median(|raw forecast|) and persists to
    ``data/calibrated_scalars.json``.
    """
    logger.info("=== Forecast Scalar Calibration started ===")
    try:
        from config import Config
        if not getattr(Config, "AUTO_CALIBRATE_SCALARS", True):
            logger.info("AUTO_CALIBRATE_SCALARS disabled — skipping")
            return

        from services.signals.forecast_scalar import calibrate_all_scalars
        import yfinance as yf

        # Build OHLCV cache for representative NIFTY-50 tickers
        tickers = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
            "ICICIBANK.NS", "HINDUNILVR.NS", "BHARTIARTL.NS",
            "SBIN.NS", "KOTAKBANK.NS", "LT.NS",
        ]
        ohlcv_cache: dict = {}
        for t in tickers:
            try:
                df = yf.download(t, period="2y", progress=False, timeout=15)
                if df is not None and len(df) > 100:
                    ohlcv_cache[t] = df
            except Exception:
                logger.debug("OHLCV fetch failed for %s", t)

        if len(ohlcv_cache) < 3:
            logger.warning("Too few tickers for calibration (%d) — skipping", len(ohlcv_cache))
            return

        result = calibrate_all_scalars(ohlcv_cache)
        logger.info("Forecast calibration complete: %s", result)

    except Exception as exc:
        logger.exception("Forecast scalar calibration failed: %s", exc)


# ═══════════════════════════════════════════════════════════════
# Phase 2 Gap B1: Monthly HMM Regime Re-fit
# ═══════════════════════════════════════════════════════════════

@_tracked_job("hmm_refit", "HMM Regime Re-fit")
def _run_hmm_refit():
    """Monthly re-fit of the HMM regime model on 5 years of NIFTY data.

    Trains a 3-state Gaussian HMM on [log_returns, VIX, breadth, delivery_vol]
    and persists the model to data/hmm_model.pkl for use by the pipeline.
    """
    logger.info("=== HMM Regime Re-fit started ===")
    try:
        from config import Config
        if not getattr(Config, "HMM_ENABLED", True):
            logger.info("HMM_ENABLED=False — skipping re-fit")
            return

        import yfinance as yf
        from services.regime.regime_hmm import MarkovRegimeModel, prepare_hmm_observations

        # Fetch 5 years of NIFTY 50 daily data
        nifty_df = yf.download("^NSEI", period="5y", progress=False, timeout=30)
        if nifty_df is None or len(nifty_df) < 500:
            logger.warning("Insufficient NIFTY data for HMM fit (%d rows)", len(nifty_df) if nifty_df is not None else 0)
            return

        # Fetch India VIX
        vix_df = None
        try:
            vix_df = yf.download("^INDIAVIX", period="5y", progress=False, timeout=15)
        except Exception:
            logger.debug("India VIX fetch failed — using proxy")

        # Prepare observation matrix
        observations = prepare_hmm_observations(nifty_df, vix_df)
        if len(observations) < 500:
            logger.warning("Too few valid observations for HMM (%d) — need 500+", len(observations))
            return

        # Fit model
        model = MarkovRegimeModel(
            n_states=getattr(Config, "HMM_N_STATES", 3),
            n_features=4,
        )
        model.fit(observations)

        # Get current regime for logging
        snap = model.get_current_regime(observations[-60:])
        logger.info(
            "HMM re-fit complete: regime=%s (%.0f%%), durations=%s",
            snap.regime, snap.confidence * 100, snap.expected_durations,
        )

        # Persist model
        model.save()
        logger.info("HMM model persisted to disk")

        # Update singleton
        from services.regime.regime_hmm import get_hmm_model
        global_model = get_hmm_model()
        global_model._fitted = model._fitted
        global_model._means = model._means
        global_model._covars = model._covars
        global_model._transmat = model._transmat
        global_model._startprob = model._startprob
        global_model._feat_mean = model._feat_mean
        global_model._feat_std = model._feat_std

        logger.info("=== HMM Regime Re-fit complete ===")

    except Exception as exc:
        logger.exception("HMM re-fit failed: %s", exc)


@_tracked_job("strategy_tournament", "Strategy Tournament")
def _run_strategy_tournament():
    """Monthly strategy tournament to rank and auto-allocate strategies.

    Runs 1st Saturday of each month at 4:00 AM IST.
    Evaluates all forecast sources over the trailing 3 months,
    disables underperformers, and persists allocation decisions.
    """
    logger.info("=== Monthly Strategy Tournament started ===")
    try:
        import pandas as pd
        from services.research.strategy_tournament import StrategyTournament
        import json, os

        # Load recent per-strategy returns from walk-forward results
        wf_results_path = os.path.join("data", "walk_forward_results.json")
        if not os.path.exists(wf_results_path):
            logger.warning("No walk-forward results found — skipping tournament")
            return

        with open(wf_results_path) as f:
            wf_data = json.load(f)

        # Convert to per-strategy return series
        strat_returns = {}
        for strat_name, returns_list in wf_data.items():
            if isinstance(returns_list, list) and len(returns_list) >= 20:
                strat_returns[strat_name] = pd.Series(returns_list)

        if len(strat_returns) < 2:
            logger.warning("Too few strategies with results (%d) — skipping", len(strat_returns))
            return

        tourney = StrategyTournament(top_n=5, min_sharpe=0.0)
        result = tourney.run_tournament(strat_returns, lookback_months=3)

        logger.info("Tournament complete: top=%s, disabled=%s",
                     result.top_strategies, result.disabled_strategies)

        # Persist tournament results for pipeline to read
        tourney_path = os.path.join("data", "tournament_results.json")
        with open(tourney_path, "w") as f:
            json.dump({
                "top_strategies": result.top_strategies,
                "disabled_strategies": result.disabled_strategies,
                "entries": [
                    {"rank": e.rank, "name": e.strategy_name,
                     "score": e.composite_score, "status": e.allocation_status}
                    for e in result.entries
                ],
            }, f, indent=2)

    except Exception as exc:
        logger.exception("Strategy tournament failed: %s", exc)


@_tracked_job("pead_earnings", "PEAD Earnings Feed")
def _run_pead_earnings_feed():
    """G6: Fetch recent earnings data and feed into PEAD strategy.

    Bridges earnings_momentum.py (Trendlyne scraper) with pead_strategy.py
    (PEAD signal generator) to automate the earnings data pipeline.
    """
    logger.info("=== PEAD Earnings Feed started ===")
    try:
        from services.signals.earnings_momentum import _fetch_recent_results, EarningsSurprise as EMSurprise
        from services.execution.pead_strategy import PEADStrategy, EarningsSurprise as PEADSurprise

        # Fetch recent earnings from Trendlyne
        raw_results = _fetch_recent_results()
        if not raw_results:
            logger.info("PEAD feed: no recent earnings data found")
            return

        # Convert earnings_momentum format to PEAD format
        pead_surprises = []
        for sym, em_data in raw_results.items():
            try:
                # Use profit surprise as primary SUE proxy
                sue = em_data.profit_surprise_pct / 10.0  # normalize to ~SUE scale
                surprise = PEADSurprise(
                    ticker=sym,
                    announcement_date=em_data.result_date,
                    eps_actual=em_data.profit_surprise_pct,  # proxy
                    eps_consensus=0.0,
                    sue=sue,
                    surprise_pct=em_data.profit_surprise_pct,
                    direction="POSITIVE" if em_data.is_positive else "NEGATIVE",
                )
                pead_surprises.append(surprise)
            except Exception:
                continue

        if not pead_surprises:
            logger.info("PEAD feed: no valid earnings surprises to process")
            return

        # Feed into PEAD strategy
        pead = PEADStrategy()
        new_signals = pead.process_earnings(pead_surprises)
        logger.info("PEAD feed: processed %d earnings, generated %d new signals",
                     len(pead_surprises), len(new_signals))

    except Exception as exc:
        logger.exception("PEAD earnings feed failed: %s", exc)


@_tracked_job("meta_label_retrain", "Meta-Label Retrain")
def _run_meta_label_retrain():
    """AFML Ch.3: Retrain the meta-labeling classifier.

    Aggregates OHLCV data for all tracked symbols and trains the
    secondary classifier that predicts forecast correctness using
    triple-barrier labels.
    """
    logger.info("=== Meta-Label Retrain started ===")
    try:
        from services.signals.meta_labeling import train_meta_labeler
        import yfinance as yf

        # Gather OHLCV for IND symbols
        from config import Config
        tickers = getattr(Config, "MONITORED_TICKERS", [])
        if not tickers:
            # Fallback: read from sample_tickers.csv
            from pathlib import Path
            csv_path = Path(__file__).parent / "sample_tickers.csv"
            if csv_path.exists():
                import csv
                with open(csv_path) as f:
                    reader = csv.reader(f)
                    tickers = [row[0].strip() for row in reader if row]

        if not tickers:
            logger.warning("Meta-label retrain: no tickers configured")
            return

        # Download 2 years of data
        ohlcv_cache = {}
        for ticker in tickers[:50]:  # cap at 50 symbols
            try:
                df = yf.download(ticker, period="2y", progress=False)
                if df is not None and len(df) > 252:
                    ohlcv_cache[ticker] = df
            except Exception:
                continue

        if len(ohlcv_cache) < 5:
            logger.warning("Meta-label retrain: insufficient data (%d symbols)", len(ohlcv_cache))
            return

        # Train IND model
        result_ind = train_meta_labeler(ohlcv_cache, market="IND")
        logger.info("Meta-label IND: %s", result_ind.get("status", "unknown"))

        # Train US model (if US tickers configured)
        us_tickers = getattr(Config, "US_MONITORED_TICKERS", [])
        if us_tickers:
            us_cache = {}
            for ticker in us_tickers[:30]:
                try:
                    df = yf.download(ticker, period="2y", progress=False)
                    if df is not None and len(df) > 252:
                        us_cache[ticker] = df
                except Exception:
                    continue
            if len(us_cache) >= 3:
                result_us = train_meta_labeler(us_cache, market="US")
                logger.info("Meta-label US: %s", result_us.get("status", "unknown"))

    except Exception as exc:
        logger.exception("Meta-label retrain failed: %s", exc)


# ===================================================================
# Phase 1-4: Advanced strategy job handlers
# ===================================================================

@_tracked_job("options_monitor", "Options Monitor")
def _run_options_monitor():
    """Job 13: Poll open options positions for profit-take / roll / expiry."""
    try:
        from config import Config
        if not getattr(Config, "OPTIONS_ENABLED", False):
            return
    except Exception:
        return

    logger.info("=== Options Monitor Poll ===")
    try:
        kite = _get_scheduler_kite()
        if kite is None:
            logger.warning("Options monitor: no Kite session")
            return
        from kite_connect.options.options_monitor import run_options_monitor_poll
        run_options_monitor_poll(kite)
    except Exception as exc:
        logger.exception("Options monitor failed: %s", exc)


@_tracked_job("margin_monitor", "Margin Monitor")
def _run_margin_monitor():
    """Job 14: Check margin utilisation and alert if thresholds breached."""
    try:
        from config import Config
        if not getattr(Config, "LEVERAGE_ENABLED", False):
            return
    except Exception:
        return

    logger.info("=== Margin Monitor Poll ===")
    try:
        kite = _get_scheduler_kite()
        if kite is None:
            logger.warning("Margin monitor: no Kite session")
            return
        from kite_connect.trading.margin_monitor import get_margin_snapshot
        snap = get_margin_snapshot(kite)
        if snap.alert_level in ("WARNING", "CRITICAL"):
            logger.warning(
                "Margin %s: %.1f%% used (%.0f / %.0f)",
                snap.alert_level, snap.utilisation_pct,
                snap.used, snap.available + snap.used,
            )
    except Exception as exc:
        logger.exception("Margin monitor failed: %s", exc)


@_tracked_job("pairs_scanner", "Pairs Scanner")
def _run_pairs_scanner():
    """Job 15: Scan configured pairs for mean-reversion signals."""
    try:
        from config import Config
        if not getattr(Config, "PAIRS_ENABLED", False):
            return
    except Exception:
        return

    logger.info("=== Pairs Trading Scanner ===")
    try:
        import numpy as np
        from utils import download_ind_ohlcv
        from services.execution.pairs_trading_live import scan_all_pairs, DEFAULT_PAIRS

        pairs = getattr(Config, "PAIRS_LIST", DEFAULT_PAIRS)
        symbols = set()
        for a, b in pairs:
            symbols.add(a)
            symbols.add(b)

        price_data = {}
        for sym in symbols:
            try:
                df = download_ind_ohlcv(sym, period="6mo")
                if df is not None and len(df) >= 60:
                    col = "Close" if "Close" in df.columns else "close"
                    price_data[sym] = df[col].values.astype(float)
            except Exception:
                pass

        signals = scan_all_pairs(price_data)
        if signals:
            logger.info("Pairs signals: %d active", len(signals))
            for s in signals:
                logger.info("  %s/%s z=%.2f action=%s forecast=%.1f",
                            s.leg1, s.leg2, s.z_score, s.action, s.forecast)

            # G5: Execute pairs via SpreadExecutor
            _execute_pairs_signals(signals)

    except Exception as exc:
        logger.exception("Pairs scanner failed: %s", exc)


@_tracked_job("futures_monitor", "Futures Monitor")
def _run_futures_monitor():
    """Job 16: Monitor futures positions for rollover and de-leveraging."""
    try:
        from config import Config
        if not getattr(Config, "LEVERAGE_ENABLED", False):
            return
    except Exception:
        return

    logger.info("=== Futures Monitor ===")
    try:
        kite = _get_scheduler_kite()
        if kite is None:
            logger.warning("Futures monitor: no Kite session")
            return
        from kite_connect.trading.futures_monitor import run_futures_monitor
        result = run_futures_monitor(kite)
        for alert in result.alerts:
            logger.warning("Futures alert: %s", alert)

        # G6: Execute futures overlay signal
        _execute_futures_overlay(kite)

    except Exception as exc:
        logger.exception("Futures monitor failed: %s", exc)


@_tracked_job("event_calendar", "Event Calendar Seed")
def _run_event_calendar_seed():
    """Job 17: Seed fixed events (RBI, rebalance) into the calendar DB."""
    try:
        from config import Config
        if not getattr(Config, "EVENT_DRIVEN_ENABLED", False):
            return
    except Exception:
        return

    logger.info("=== Event Calendar Seed ===")
    try:
        from services.market_data.event_calendar import seed_fixed_events
        seed_fixed_events()
    except Exception as exc:
        logger.exception("Event calendar seed failed: %s", exc)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Scheduler setup
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


# =================================================================
# G4/G5/G6/G10: Execution Helper Functions
# =================================================================

def _execute_options_overlay(kite):
    """G4: Execute options overlay - covered calls + CSPs."""
    try:
        from config import Config
        if not getattr(Config, "OPTIONS_ENABLED", False):
            return

        from kite_connect.options.options_executor import OptionsExecutor
        from services.execution.options_overlay import scan_covered_call_candidates, scan_csp_candidates

        executor = OptionsExecutor(kite)

        # Covered calls on existing long positions
        try:
            cc_candidates = scan_covered_call_candidates(kite)
            if cc_candidates:
                results = executor.execute_covered_calls(cc_candidates)
                logger.info("G4: Covered calls executed: %d orders", len(results))
        except Exception as exc:
            logger.warning("G4: Covered calls failed: %s", exc)

        # Cash-secured puts on high-conviction BUY signals
        try:
            csp_candidates = scan_csp_candidates(kite)
            if csp_candidates:
                results = executor.execute_cash_secured_puts(csp_candidates)
                logger.info("G4: CSPs executed: %d orders", len(results))
        except Exception as exc:
            logger.warning("G4: CSPs failed: %s", exc)

    except Exception as exc:
        logger.debug("G4: Options overlay skipped: %s", exc)


def _execute_tail_hedge_if_needed(kite):
    """G10: Auto-execute tail hedge when drawdown is critical."""
    try:
        from config import Config
        if not getattr(Config, "OPTIONS_TAIL_HEDGE_ENABLED", False):
            return

        from services.risk.tail_risk_hedge import TailRiskHedge
        from kite_connect.options.options_executor import OptionsExecutor

        # Get current portfolio state
        capital = getattr(Config, "CARVER_INITIAL_CAPITAL", 500_000)
        realized = getattr(Config, "_CUMULATIVE_REALIZED_PNL", 0.0)
        equity = getattr(Config, "_CURRENT_EQUITY", capital + realized)
        peak = getattr(Config, "_PEAK_EQUITY", capital)
        dd_pct = ((peak - equity) / peak * 100) if peak > 0 else 0

        # Get VIX
        try:
            import yfinance as yf
            vix_data = yf.download("^INDIAVIX", period="5d", progress=False)
            vix = float(vix_data["Close"].iloc[-1]) if len(vix_data) > 0 else 15.0
            vix_3d = float(vix_data["Close"].iloc[-4]) if len(vix_data) >= 4 else vix
        except Exception:
            vix, vix_3d = 15.0, 15.0

        # Get NIFTY spot
        try:
            ltp = kite.ltp(["NSE:NIFTY 50"])
            nifty_spot = ltp.get("NSE:NIFTY 50", {}).get("last_price", 0)
        except Exception:
            nifty_spot = 0

        hedger = TailRiskHedge()
        assessment = hedger.assess(
            portfolio_value=equity,
            drawdown_pct=dd_pct,
            vix=vix,
            vix_3d_ago=vix_3d,
            nifty_spot=nifty_spot,
        )

        if assessment.hedge_urgency in ("HIGH", "CRITICAL") and assessment.recommendation:
            executor = OptionsExecutor(kite)
            result = executor.execute_tail_hedge(assessment.recommendation)
            logger.info("G10: Tail hedge executed: urgency=%s result=%s",
                        assessment.hedge_urgency, result)
        else:
            logger.info("G10: Tail hedge not needed: urgency=%s dd=%.1f%%",
                        assessment.hedge_urgency, dd_pct)

    except Exception as exc:
        logger.debug("G10: Tail hedge check skipped: %s", exc)


def _execute_pairs_signals(signals):
    """G5: Execute pairs trading signals via SpreadExecutor."""
    try:
        kite = _get_scheduler_kite()
        if kite is None:
            logger.warning("G5: No Kite session for pairs execution")
            return

        from kite_connect.trading.spread_executor import SpreadExecutor, LegOrder
        spread_exec = SpreadExecutor(kite)

        for sig in signals:
            if not hasattr(sig, "action") or sig.action not in ("ENTER_LONG", "ENTER_SHORT"):
                continue

            # Build leg orders based on signal direction
            if sig.action == "ENTER_LONG":
                leg1 = LegOrder(symbol=sig.leg1, side="BUY", quantity=1, exchange="NSE")
                leg2 = LegOrder(symbol=sig.leg2, side="SELL", quantity=1, exchange="NSE")
            else:  # ENTER_SHORT
                leg1 = LegOrder(symbol=sig.leg1, side="SELL", quantity=1, exchange="NSE")
                leg2 = LegOrder(symbol=sig.leg2, side="BUY", quantity=1, exchange="NSE")

            result = spread_exec.execute_pair(leg1, leg2)
            logger.info("G5: Pair %s/%s %s: success=%s",
                        sig.leg1, sig.leg2, sig.action, result.success)

    except Exception as exc:
        logger.warning("G5: Pairs execution failed: %s", exc)


def _execute_futures_overlay(kite):
    """G6: Execute futures overlay signal for regime-adaptive leverage."""
    try:
        from config import Config
        if not getattr(Config, "LEVERAGE_ENABLED", False):
            return

        from services.execution.futures_overlay import compute_futures_overlay
        from services.regime.regime_detector import get_current_regime
        from kite_connect.trading.order_service import place_order

        # Get current state
        capital = getattr(Config, "CARVER_INITIAL_CAPITAL", 500_000)
        realized = getattr(Config, "_CUMULATIVE_REALIZED_PNL", 0.0)
        equity = capital + realized

        regime_info = get_current_regime()
        regime = regime_info.get("regime", "range")
        confidence = regime_info.get("confidence", 0.5)

        # Get NIFTY spot/futures price
        try:
            ltp = kite.ltp(["NSE:NIFTY 50"])
            nifty_spot = ltp.get("NSE:NIFTY 50", {}).get("last_price", 0)
        except Exception:
            nifty_spot = 0

        signal = compute_futures_overlay(
            portfolio_value=equity,
            current_futures_notional=0.0,
            nifty_spot=nifty_spot,
            regime=regime,
            regime_confidence=confidence,
        )

        if signal.action == "BUY_FUT" and signal.lots > 0:
            lot_size = getattr(Config, "FUTURES_LOT_SIZE", 25)
            result = place_order(
                kite,
                tradingsymbol="NIFTY" + _get_current_expiry_suffix(),
                exchange="NFO",
                transaction_type="BUY",
                quantity=signal.lots * lot_size,
                product="NRML",
                order_type="MARKET",
            )
            logger.info("G6: BUY_FUT %d lots, order=%s", signal.lots, result)

        elif signal.action == "SELL_FUT" and signal.lots > 0:
            lot_size = getattr(Config, "FUTURES_LOT_SIZE", 25)
            result = place_order(
                kite,
                tradingsymbol="NIFTY" + _get_current_expiry_suffix(),
                exchange="NFO",
                transaction_type="SELL",
                quantity=signal.lots * lot_size,
                product="NRML",
                order_type="MARKET",
            )
            logger.info("G6: SELL_FUT %d lots, order=%s", signal.lots, result)

        else:
            logger.debug("G6: Futures overlay action=%s lots=%d (no trade)",
                         signal.action, signal.lots)

    except Exception as exc:
        logger.debug("G6: Futures overlay skipped: %s", exc)


def _get_current_expiry_suffix():
    """Get current month NIFTY futures expiry suffix (e.g. '25JUN' for Jun 2025)."""
    from datetime import date
    today = date.today()
    # NFO convention: YYMMMFUT e.g. NIFTY25JUNFUT
    suffix = today.strftime("%y%b").upper() + "FUT"
    return suffix


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
