"""
Trial registry: every backtest ever run, read from run directories.

Run directory layout (``config.runs_dir/<run_id>/``, see docs/nse_engine.md):
``manifest.json`` (run_id, tag, config_hash, git_commit, git_dirty,
data_hash, start, end, created_at, metrics) and ``returns.csv``
(date, return).  PBO and DSR must see every configuration evaluated, so the
registry is the single source of the trial matrix.

``record_result`` writes a minimal run directory for a ``BacktestResult``
that was produced without the engine's own recorder (e.g. an injected
backtest function); the engine itself writes the full layout.
"""

from __future__ import annotations

import json
import logging
import subprocess
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MANIFEST_FIELDS = ("run_id", "tag", "config_hash", "git_commit", "git_dirty",
                   "data_hash", "start", "end", "created_at")


_GIT_CACHE: Dict[str, Any] = {}
_GIT_CACHE_TTL_S = 60.0


def git_state(repo_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Return ``{"git_commit": sha|"unknown", "git_dirty": bool|None}``.

    Cached for 60 s per directory (walk-forward records many runs quickly).
    """
    cwd = str(repo_dir) if repo_dir else None
    import time

    hit = _GIT_CACHE.get(cwd or "")
    if hit and time.monotonic() - hit[0] < _GIT_CACHE_TTL_S:
        return dict(hit[1])
    state = _git_state_uncached(cwd)
    _GIT_CACHE[cwd or ""] = (time.monotonic(), state)
    return dict(state)


def _git_state_uncached(cwd: Optional[str]) -> Dict[str, Any]:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
                             text=True, timeout=10, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True,
                               text=True, timeout=30, check=True).stdout.strip() != ""
        return {"git_commit": sha, "git_dirty": dirty}
    except Exception:  # pragma: no cover - git missing
        return {"git_commit": "unknown", "git_dirty": None}


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/pandas objects into JSON-serialisable values.

    Non-finite floats become ``None``.
    """
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, pd.Series):
        return {str(k.date() if isinstance(k, pd.Timestamp) else k): to_jsonable(v)
                for k, v in obj.items()}
    if isinstance(obj, pd.DataFrame):
        return to_jsonable(obj.to_dict(orient="list"))
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if np.isfinite(f) else None
    return obj


def record_result(result: Any, tag: str, runs_dir: Optional[Union[str, Path]] = None,
                  window: Optional[tuple] = None) -> str:
    """Write ``manifest.json`` + ``returns.csv`` for ``result``; return run_dir.

    ``result`` is a ``BacktestResult``; its ``config`` must provide
    ``config_hash()`` and ``runs_dir``.  Sets ``result.run_id`` /
    ``result.run_dir`` when they are empty.
    """
    config = result.config
    root = Path(runs_dir or getattr(config, "runs_dir", "data/nse_engine/runs"))
    created = datetime.now(timezone.utc)
    run_id = getattr(result, "run_id", "") or f"{created:%Y%m%dT%H%M%S}-{tag or 'run'}-{uuid.uuid4().hex[:8]}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    returns = pd.Series(result.returns, dtype="float64").dropna()
    start, end = window if window else (getattr(config, "start", None), getattr(config, "end", None))
    manifest = {
        "run_id": run_id,
        "tag": tag,
        "config_hash": config.config_hash() if hasattr(config, "config_hash") else "",
        **git_state(),
        "data_hash": getattr(result, "data_hash", ""),
        "start": str(start) if start is not None else None,
        "end": str(end) if end is not None else None,
        "created_at": created.isoformat(),
        "metrics": to_jsonable(getattr(result, "metrics", {}) or {}),
        "recorded_by": "nse_engine.validation.trials.record_result",
    }
    if hasattr(config, "to_json"):
        (run_dir / "config.json").write_text(config.to_json())
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    frame = pd.DataFrame({"date": pd.DatetimeIndex(returns.index).strftime("%Y-%m-%d"),
                          "return": returns.to_numpy()})
    frame.to_csv(run_dir / "returns.csv", index=False)
    try:
        if not getattr(result, "run_id", ""):
            result.run_id = run_id
        if not getattr(result, "run_dir", None):
            result.run_dir = str(run_dir)
    except Exception:  # frozen / foreign objects
        pass
    return str(run_dir)


class TrialRegistry:
    """Read-only view over ``runs_dir/<run_id>/{manifest.json, returns.csv}``."""

    def __init__(self, runs_dir: Union[str, Path]):
        self.runs_dir = Path(runs_dir)
        self._returns_cache: Dict[str, pd.Series] = {}

    # ------------------------------------------------------------------
    def _manifests(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not self.runs_dir.exists():
            return out
        for d in sorted(p for p in self.runs_dir.iterdir() if p.is_dir()):
            mf = d / "manifest.json"
            if not mf.exists() or not (d / "returns.csv").exists():
                continue
            try:
                man = json.loads(mf.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Skipping unreadable manifest %s: %s", mf, exc)
                continue
            man.setdefault("run_id", d.name)
            man["_dir"] = str(d)
            out.append(man)
        return out

    def list_trials(self) -> pd.DataFrame:
        """One row per recorded run: manifest fields plus scalar metrics."""
        rows = []
        for man in self._manifests():
            row = {k: man.get(k) for k in MANIFEST_FIELDS}
            for k, v in (man.get("metrics") or {}).items():
                if isinstance(v, (int, float, bool, str)) or v is None:
                    row[k if k not in row else f"metric_{k}"] = v
            row["run_dir"] = man["_dir"]
            rows.append(row)
        cols = list(MANIFEST_FIELDS)
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=cols + ["run_dir"])
        return df.sort_values(["created_at", "run_id"], na_position="first").reset_index(drop=True)

    def load_returns(self, run_id: str) -> pd.Series:
        """Daily returns of one run (cached)."""
        if run_id not in self._returns_cache:
            path = self.runs_dir / run_id / "returns.csv"
            df = pd.read_csv(path)
            date_col = "date" if "date" in df.columns else df.columns[0]
            ret_col = "return" if "return" in df.columns else df.columns[-1]
            s = pd.Series(df[ret_col].to_numpy(dtype="float64"),
                          index=pd.DatetimeIndex(pd.to_datetime(df[date_col])), name=run_id)
            s = s[~s.index.duplicated(keep="last")].sort_index()
            self._returns_cache[run_id] = s
        return self._returns_cache[run_id]

    def returns_matrix(self, start: Optional[str] = None, end: Optional[str] = None,
                       dedupe_config: bool = True, data_hash: Optional[str] = None,
                       tags: Optional[List[str]] = None,
                       window: Optional[tuple] = None) -> pd.DataFrame:
        """date x run_id daily returns restricted to dates common to all runs.

        ``dedupe_config`` keeps only the latest run (by created_at) per
        (config_hash, start, end) -- re-runs of one config on one window.
        ``data_hash`` / ``tags`` filter runs.  ``window=(start, end)`` keeps only
        runs recorded on exactly that trading window, so walk-forward fold runs
        do not shrink the common-date intersection.  Warns when the selected
        runs were computed on different data hashes.
        """
        trials = self.list_trials()
        if trials.empty:
            return pd.DataFrame()
        if data_hash is not None:
            trials = trials[trials["data_hash"] == data_hash]
        if tags is not None:
            trials = trials[trials["tag"].isin(tags)]
        if window is not None:
            lo, hi = (pd.Timestamp(w) for w in window)
            same = (pd.to_datetime(trials["start"], errors="coerce") == lo) & \
                   (pd.to_datetime(trials["end"], errors="coerce") == hi)
            trials = trials[same]
        if dedupe_config and not trials.empty:
            # A config re-run on the SAME window is a duplicate; the same config on
            # another window (e.g. a walk-forward fold) is a different trial.
            keyed = (trials["config_hash"].fillna("").astype(str) + "|"
                     + trials["start"].fillna("").astype(str) + "|"
                     + trials["end"].fillna("").astype(str))
            keyed = keyed.where(trials["config_hash"].fillna("").astype(str) != "", "")
            has_key = keyed != ""
            latest = trials[has_key].groupby(keyed[has_key], sort=False).tail(1)
            trials = pd.concat([latest, trials[~has_key]]).sort_values(["created_at", "run_id"])
        if trials.empty:
            return pd.DataFrame()
        hashes = set(trials["data_hash"].dropna().astype(str))
        if len(hashes) > 1:
            msg = (f"Trials span {len(hashes)} different data hashes {sorted(hashes)}; "
                   "returns are not comparable across data versions (pass data_hash=...)")
            logger.warning(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)
        series = [self.load_returns(r) for r in trials["run_id"]]
        mat = pd.concat(series, axis=1, join="inner")
        mat.columns = list(trials["run_id"])
        if start is not None:
            mat = mat.loc[mat.index >= pd.Timestamp(start)]
        if end is not None:
            mat = mat.loc[mat.index <= pd.Timestamp(end)]
        mat = mat.dropna(axis=0, how="any")
        mat.index.name = "date"
        return mat
