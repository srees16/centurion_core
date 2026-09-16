"""
One-shot holdout evaluation guarded by a lock file.

The holdout window may be evaluated once per window.  The lock file is a
JSON document ``{"evaluations": [...]}``; each entry records the window,
config_hash, git commit, data_hash, timestamps, status (``started`` ->
``completed``) and metrics.  An entry is written BEFORE the backtest runs, so
a crash after results were seen still counts as an evaluation.

``force=True`` re-runs a window anyway: it logs a loud warning and the new
entry carries ``forced: true`` so the override is permanently visible.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import pandas as pd

from .dsr import performance_summary
from .trials import git_state, record_result, to_jsonable

logger = logging.getLogger(__name__)


class HoldoutLockedError(RuntimeError):
    """Raised when a holdout window has already been evaluated."""


def _read_lock(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"evaluations": []}
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "evaluations" in data:
        return data
    return {"evaluations": [data] if data else []}  # tolerate a single-record lock


def _write_lock(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True))
    os.replace(tmp, path)


def _norm_date(d: Any) -> str:
    return str(pd.Timestamp(d).date())


def holdout_evaluations(lock_path: Union[str, Path] = "data/nse_engine/holdout.lock") -> list:
    """Return the recorded holdout evaluations (empty list if no lock)."""
    return list(_read_lock(Path(lock_path))["evaluations"])


def run_holdout(data: Any, config: Any, start: Any, end: Any,
                lock_path: Union[str, Path] = "data/nse_engine/holdout.lock",
                force: bool = False, backtest_fn: Optional[Callable[..., Any]] = None,
                ) -> Dict[str, Any]:
    """Evaluate ``config`` once on the holdout window ``[start, end]``.

    Raises ``HoldoutLockedError`` if the lock already records this window
    (unless ``force``).  The run is tagged ``"holdout"``.

    Returns dict: ``returns``, ``metrics``, ``lock_record``, ``result``.
    """
    path = Path(lock_path)
    window = {"start": _norm_date(start), "end": _norm_date(end)}
    lock = _read_lock(path)
    previous = [e for e in lock["evaluations"] if e.get("window") == window]
    if previous and not force:
        raise HoldoutLockedError(
            f"holdout window {window['start']}..{window['end']} was already evaluated "
            f"{len(previous)} time(s) (first at {previous[0].get('created_at')}, "
            f"config {previous[0].get('config_hash')}); lock={path}. Re-running it turns "
            "the holdout into another selection sample. Use force=True only with a "
            "documented reason.")
    if previous and force:
        logger.warning("!" * 72)
        logger.warning("HOLDOUT OVERRIDE: window %s..%s already evaluated %d time(s); "
                       "force=True -- this result is NOT an untouched holdout and the "
                       "override is recorded in %s", window["start"], window["end"],
                       len(previous), path)
        logger.warning("!" * 72)
    overlapping = [e for e in lock["evaluations"] if e.get("window") != window and e.get("window")
                   and e["window"]["start"] <= window["end"] and window["start"] <= e["window"]["end"]]
    if overlapping:
        logger.warning("Holdout window %s..%s overlaps %d previously evaluated window(s)",
                       window["start"], window["end"], len(overlapping))

    cfg = config.replace(start=window["start"], end=window["end"])
    entry: Dict[str, Any] = {
        "window": window,
        "config_hash": cfg.config_hash(),
        **git_state(),
        "data_hash": getattr(data, "data_hash", ""),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "forced": bool(previous and force),
        "n_previous_evaluations": len(previous),
        "metrics": {},
    }
    lock["evaluations"].append(entry)
    _write_lock(path, lock)

    if backtest_fn is None:
        from nse_engine.engine import run_backtest as backtest_fn  # lazy
    result = backtest_fn(data, cfg, record=True, tag="holdout")
    if not getattr(result, "run_dir", None):
        record_result(result, tag="holdout", window=(window["start"], window["end"]))
    r = pd.Series(result.returns, dtype="float64").dropna()
    r.index = pd.DatetimeIndex(r.index)
    r = r[(r.index >= pd.Timestamp(window["start"])) & (r.index <= pd.Timestamp(window["end"]))]
    metrics = performance_summary(r, float(getattr(cfg, "risk_free_annual", 0.0)))
    metrics["engine_metrics"] = to_jsonable(getattr(result, "metrics", {}) or {})

    entry.update({"status": "completed", "completed_at": datetime.now(timezone.utc).isoformat(),
                  "run_id": getattr(result, "run_id", ""),
                  "data_hash": getattr(result, "data_hash", "") or entry["data_hash"],
                  "metrics": to_jsonable(metrics)})
    _write_lock(path, lock)
    logger.info("Holdout %s..%s evaluated: excess Sharpe %.3f", window["start"], window["end"],
                metrics["excess_sharpe"])
    return {"returns": r, "metrics": metrics, "lock_record": entry, "result": result}
