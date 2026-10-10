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
import time
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MANIFEST_FIELDS = ("run_id", "tag", "config_hash", "git_commit", "git_dirty",
                   "data_hash", "start", "end", "created_at", "refresh_of", "cost_model")
#: Runs recorded before the cost model was versioned (U25, 30 Sep 2026).
LEGACY_COST_MODEL = 1


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
                  window: Optional[tuple] = None, extra: Optional[Dict[str, Any]] = None) -> str:
    """Write ``manifest.json`` + ``returns.csv`` for ``result``; return run_dir.

    ``result`` is a ``BacktestResult``; its ``config`` must provide
    ``config_hash()`` and ``runs_dir``.  Sets ``result.run_id`` /
    ``result.run_dir`` when they are empty.  ``extra`` fields are added to
    the manifest without overriding the standard ones.
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
        "data_hash_version": _data_hash_version(),
        "start": str(start) if start is not None else None,
        "end": str(end) if end is not None else None,
        "created_at": created.isoformat(),
        "metrics": to_jsonable(getattr(result, "metrics", {}) or {}),
        "recorded_by": "nse_engine.validation.trials.record_result",
        "cost_model": _cost_model_version(),
    }
    for key, value in dict(extra or {}).items():
        manifest.setdefault(str(key), to_jsonable(value))
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


def _data_hash_version() -> int:
    from nse_engine.types import DATA_HASH_VERSION

    return DATA_HASH_VERSION


def _cost_model_version() -> int:
    from nse_engine.costs import COST_MODEL_VERSION
    return int(COST_MODEL_VERSION)


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
        df["cost_model"] = pd.to_numeric(df["cost_model"], errors="coerce").fillna(LEGACY_COST_MODEL).astype(int)
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
                       window: Optional[tuple] = None, cost_model: Optional[int] = None) -> pd.DataFrame:
        """date x run_id daily returns restricted to dates common to all runs.

        ``dedupe_config`` keeps only the latest run (by created_at) per
        (config_hash, start, end) -- re-runs of one config on one window.
        ``data_hash`` / ``tags`` / ``cost_model`` filter runs (results from two
        cost models are not comparable: U25).  ``window=(start, end)`` keeps only
        runs recorded on exactly that trading window, so walk-forward fold runs
        do not shrink the common-date intersection.  Warns when the selected
        runs were computed on different data hashes.
        """
        trials = self.list_trials()
        if trials.empty:
            return pd.DataFrame()
        if data_hash is not None:
            trials = trials[trials["data_hash"] == data_hash]
        if cost_model is not None:
            trials = trials[trials["cost_model"] == int(cost_model)]
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


# ----------------------------------------------------------------------------
# registry refresh after a data-fingerprint change
# ----------------------------------------------------------------------------

PLAN_COLUMNS = ("run_id", "config_hash", "tag", "lag_days", "created_at", "run_dir")


def refresh_plan(registry: "TrialRegistry", from_hash: str, window: tuple,
                 skip_hash: Optional[str] = None) -> pd.DataFrame:
    """Configurations recorded on ``from_hash`` over ``window``, one row each.

    The row is the configuration's latest run (the one ``returns_matrix``
    keeps).  Configurations already recorded on ``skip_hash`` over the same
    window are left out, so a refresh can be resumed.
    """
    trials = registry.list_trials()
    empty = pd.DataFrame(columns=list(PLAN_COLUMNS))
    if trials.empty:
        return empty
    lo, hi = (pd.Timestamp(w) for w in window)
    starts = pd.to_datetime(trials["start"], errors="coerce")
    ends = pd.to_datetime(trials["end"], errors="coerce")
    same_window = (starts == lo) & (ends == hi)
    hashes = trials["config_hash"].fillna("").astype(str)
    on_from = same_window & (trials["data_hash"].astype(str) == str(from_hash)) & (hashes != "")
    sub = trials[on_from].sort_values(["created_at", "run_id"], na_position="first")
    sub = sub.groupby(sub["config_hash"].astype(str), sort=False).tail(1)
    if skip_hash is not None:
        done = set(hashes[same_window & (trials["data_hash"].astype(str) == str(skip_hash))])
        sub = sub[~sub["config_hash"].astype(str).isin(done)]
    rows = []
    for rec in sub.itertuples(index=False):
        try:
            man = json.loads((Path(rec.run_dir) / "manifest.json").read_text())
        except (OSError, json.JSONDecodeError):
            man = {}
        rows.append({"run_id": rec.run_id, "config_hash": str(rec.config_hash),
                     "tag": str(man.get("tag") or ""), "lag_days": int(man.get("lag_days") or 0),
                     "created_at": rec.created_at, "run_dir": rec.run_dir})
    if not rows:
        return empty
    return pd.DataFrame(rows, columns=list(PLAN_COLUMNS)).sort_values(
        ["created_at", "run_id"], na_position="first").reset_index(drop=True)


def compare_returns(old: pd.Series, new: pd.Series, tolerance: float = 1e-9) -> Dict[str, Any]:
    """Do two daily return series agree day by day (within ``tolerance``)?"""
    o = pd.Series(old, dtype="float64").dropna()
    n = pd.Series(new, dtype="float64").dropna()
    o.index = pd.DatetimeIndex(o.index)
    n.index = pd.DatetimeIndex(n.index)
    joined = pd.concat([o.rename("old"), n.rename("new")], axis=1, join="inner")
    diff = (joined["new"] - joined["old"]).abs()
    same_dates = len(joined) == len(o) == len(n)
    return {
        "n_old": int(len(o)), "n_new": int(len(n)), "n_common": int(len(joined)),
        "max_abs_diff": float(diff.max()) if len(diff) else float("nan"),
        "n_differing": int((diff > tolerance).sum()) + (0 if same_dates else abs(len(o) - len(n))),
        "identical": bool(len(joined) > 0 and same_dates and bool((diff <= tolerance).all())),
    }


def refresh_registry(registry: "TrialRegistry", data: Any, from_hash: str, window: tuple, *,
                     backtest_fn: Optional[Any] = None, config_loader: Optional[Any] = None,
                     dry_run: bool = False, limit: Optional[int] = None, tolerance: float = 1e-9,
                     log: Optional[Any] = None) -> Dict[str, Any]:
    """Re-run every configuration recorded on ``from_hash`` over ``window`` on
    ``data``, whose fingerprint differs, and check that the returns reproduce.

    Why: ``returns_matrix`` compares only runs that share a data hash, so
    after a store rebuild (a symbol rename is enough) a new run would meet no
    prior configurations - no PBO, a deflated Sharpe at N = 1.  Re-running
    the same configurations keeps the same config hashes, so for the matrix
    they are duplicates (dedupe keeps the latest), not new trials.  Each new
    manifest carries ``refresh_of`` = the run it reproduces.

    ``backtest_fn(data, config, record=True, tag=, lag_days=, manifest_extra=)``
    defaults to the engine; ``config_loader(run_dir) -> config`` defaults to
    reading ``config.json``.  ``dry_run`` only reports the plan.
    """
    to_hash = getattr(data, "data_hash", "") or data.compute_hash()
    full_plan = refresh_plan(registry, from_hash, window)
    plan = refresh_plan(registry, from_hash, window, skip_hash=to_hash)
    if limit:
        plan = plan.head(int(limit))
    report: Dict[str, Any] = {
        "from_hash": str(from_hash), "to_hash": str(to_hash),
        "window": [str(pd.Timestamp(w).date()) for w in window],
        "n_configurations": int(len(full_plan)), "n_planned": int(len(plan)),
        "dry_run": bool(dry_run), "tolerance": float(tolerance), "rows": [],
    }
    if to_hash == str(from_hash):
        report["note"] = "data hash unchanged; nothing to refresh"
        return report
    if dry_run or plan.empty:
        report["rows"] = [dict(r) for r in plan.to_dict(orient="records")]
        report["complete"] = bool(len(full_plan)) and plan.empty
        return report
    if backtest_fn is None:
        from nse_engine.engine import run_backtest as backtest_fn  # lazy: heavy import
    if config_loader is None:
        from nse_engine.config import EngineConfig

        def config_loader(run_dir):  # noqa: E306
            return EngineConfig.from_dict(json.loads((Path(run_dir) / "config.json").read_text()))
    for i, rec in enumerate(plan.itertuples(index=False), 1):
        cfg = config_loader(rec.run_dir)
        t0 = time.perf_counter()
        res = backtest_fn(data, cfg, record=True, tag=rec.tag, lag_days=int(rec.lag_days),
                          manifest_extra={"refresh_of": rec.run_id})
        cmp = compare_returns(registry.load_returns(rec.run_id), pd.Series(res.returns), tolerance)
        row = {"run_id": rec.run_id, "new_run_id": str(getattr(res, "run_id", "") or ""),
               "config_hash": rec.config_hash, "tag": rec.tag, "lag_days": int(rec.lag_days),
               "seconds": round(time.perf_counter() - t0, 1), **cmp}
        report["rows"].append(row)
        if log is not None:
            verdict = ("identical" if cmp["identical"] else
                       f"DIFFERS: max {cmp['max_abs_diff']:.2e} on {cmp['n_differing']} days")
            log(f"[{i}/{len(plan)}] {rec.config_hash[:8]} {rec.tag[:44]:<44} {verdict} ({row['seconds']}s)")
    on_new = registry.returns_matrix(data_hash=to_hash, window=window)
    report["n_on_new_hash"] = int(on_new.shape[1])
    report["all_identical"] = bool(report["rows"]) and all(r["identical"] for r in report["rows"])
    report["complete"] = report["n_on_new_hash"] >= report["n_configurations"]
    return report


def registry_hash(registry: "TrialRegistry", window: tuple) -> Optional[str]:
    """Data hash of the most recently recorded run over ``window``, i.e. the
    fingerprint the registry was last extended on (None when no run exists)."""
    trials = registry.list_trials()
    if trials.empty:
        return None
    lo, hi = (pd.Timestamp(w) for w in window)
    same = ((pd.to_datetime(trials["start"], errors="coerce") == lo)
            & (pd.to_datetime(trials["end"], errors="coerce") == hi)
            & trials["data_hash"].fillna("").astype(str).ne(""))
    sub = trials[same].sort_values(["created_at", "run_id"], na_position="first")
    return str(sub["data_hash"].iloc[-1]) if len(sub) else None


def fingerprint_status(registry: "TrialRegistry", window: tuple, current_hash: str) -> Dict[str, Any]:
    """Is the store's fingerprint the one the registry was last extended on?

    Returns ``changed`` plus the counts needed to act: how many configurations
    sit on the registry's hash and how many already on the current one.
    """
    ref = registry_hash(registry, window)
    on_ref = len(refresh_plan(registry, ref, window)) if ref else 0
    on_cur = len(refresh_plan(registry, current_hash, window))
    return {"window": [str(pd.Timestamp(w).date()) for w in window], "current_hash": str(current_hash),
            "registry_hash": ref, "changed": bool(ref) and ref != str(current_hash),
            "n_configurations_on_registry_hash": int(on_ref), "n_configurations_on_current_hash": int(on_cur)}
