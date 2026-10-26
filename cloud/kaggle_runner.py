"""Run NSE-engine research jobs on Kaggle — fold by fold, several at a time.

Why this exists: the 2013–2025 walk-forward is 297 backtests, and it should
run off the research machine and off the daily Actions job. A Kaggle CPU
session gives 4 cores and 30 GB but stops at 12 hours, so this runner

  * evaluates each fold's grid across worker processes (measured: 4–5 minutes
    per 32-point fold at 4 workers), and
  * writes one file per finished fold, so a session that dies resumes where it
    stopped instead of starting again.

Selection is the same as ``nse_engine.validation.walk_forward.run_walk_forward``
— the whole grid on the train window, best annualised excess Sharpe, that one
configuration on the following test window — and every backtest is recorded, so
PBO and the deflated Sharpe still count every configuration evaluated.
``cloud.wf_stitch`` joins the fold files into the same result dictionary.

In a Kaggle session:

    !python -m cloud.kaggle_runner walk-forward --grid "$(cat grid.json)" \
        --start 2013-01-01 --end 2025-12-31 --data-start 2011-01-01 \
        --folds 0-3 --workers 4 --max-hours 11

Locally the defaults point at the repo's own data directory, so the same
command works for a dry run with a small grid.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger("kaggle_runner")

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")

# Worker globals: with a forked pool the children inherit the loaded panels, so
# 30 GB of RAM holds one copy, not one per worker.
_DATA: Any = None
_BASE: Any = None


# ── environment ──────────────────────────────────────────────────

def on_kaggle() -> bool:
    return KAGGLE_WORKING.is_dir()


def find_dataset_dir(marker: str, default: str, explicit: Optional[str] = None) -> str:
    """Locate a store by a file only it has: an explicit path, an attached dataset, or the repo.

    A dataset may hold the store as a directory or as the ``store.tar.gz`` that
    ``cloud.kaggle_local push-store`` uploads; archives are unpacked into
    /kaggle/working, since datasets themselves are read-only.  Kaggle unpacks
    an uploaded archive into a folder named after it, so the store can sit one
    or two levels deeper than the dataset root: the marker is searched for
    rather than a layout assumed.
    """
    if explicit:
        return explicit
    if not KAGGLE_INPUT.is_dir():
        return default
    found = _find_marker(KAGGLE_WORKING, marker) or _find_marker(KAGGLE_INPUT, marker)
    if found:
        return str(found)
    import shutil

    for archive in sorted(KAGGLE_INPUT.rglob("store.tar.gz")):
        logger.info("unpacking %s", archive)
        KAGGLE_WORKING.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(str(archive), str(KAGGLE_WORKING))
        found = _find_marker(KAGGLE_WORKING, marker)
        if found:
            return str(found)
    return default


def find_store_dir(explicit: Optional[str] = None) -> str:
    """The equity parquet store."""
    return find_dataset_dir("calendar.parquet", "data/nse_engine/store", explicit)


def find_fo_store_dir(explicit: Optional[str] = None) -> str:
    """The F&O store (tracker O2)."""
    return find_dataset_dir("underlying.parquet", "data/nse_engine/fo_store", explicit)


def find_fundamentals_dir(explicit: Optional[str] = None) -> str:
    """The point-in-time results store (tracker GT1)."""
    return find_dataset_dir("board_meetings.parquet", "data/nse_engine/fundamentals", explicit)


def find_mcx_store_dir(explicit: Optional[str] = None) -> str:
    """The MCX commodity-futures store (tracker GT2)."""
    return find_dataset_dir("continuous.parquet", "data/nse_engine/mcx_store", explicit)


def find_us_store_dir(explicit: Optional[str] = None) -> str:
    """The survivorship-free US store's panel (US tracker US2)."""
    return find_dataset_dir("member_sp500.parquet", "data/nse_engine/us_store/panel", explicit)


def _find_marker(root: Path, marker: str, max_depth: int = 8) -> Optional[Path]:
    """Directory holding ``marker`` somewhere under ``root`` (at most ``max_depth`` levels down)."""
    if not root.is_dir():
        return None
    base_depth = len(root.parts)
    for hit in sorted(root.rglob(marker)):
        if len(hit.parts) - base_depth <= max_depth:
            return hit.parent
    return None


def default_out_dir() -> Path:
    return (KAGGLE_WORKING / "wf") if on_kaggle() else (_ROOT / "data/nse_engine/wf")


def default_runs_dir() -> str:
    return str(KAGGLE_WORKING / "runs") if on_kaggle() else "data/nse_engine/runs"


# ── config ───────────────────────────────────────────────────────

def _parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def build_config(args):
    from nse_engine.config import EngineConfig

    cfg = EngineConfig()
    if getattr(args, "config", None):
        cfg = EngineConfig.from_dict(json.loads(Path(args.config).read_text()))
    overrides: Dict[str, Any] = {}
    for item in getattr(args, "set", None) or []:
        key, _, value = item.partition("=")
        overrides[key] = _parse_value(value)
    for key in ("start", "end"):
        if getattr(args, key, None):
            overrides[key] = getattr(args, key)
    overrides["data.store_dir"] = find_store_dir(getattr(args, "store_dir", None))
    overrides["runs_dir"] = getattr(args, "runs_dir", None) or default_runs_dir()
    return cfg.replace(**overrides)


def load_data(cfg, data_start: Optional[str], warmup_years: int = 2):
    """Load panels from the anchor date; rebalance days count from the first row."""
    from datetime import date

    from nse_engine.data.panel import load_market_data

    start = (date.fromisoformat(data_start) if data_start
             else date(date.fromisoformat(cfg.start).year - warmup_years, 1, 1))
    logger.info("Loading market data %s..%s from %s", start, cfg.end, cfg.data.store_dir)
    return load_market_data(
        cfg.data.store_dir, start.isoformat(), cfg.end,
        series=cfg.data.series,
        min_median_value_inr=cfg.data.load_min_median_value_inr,
        float_dtype=cfg.data.float_dtype,
        include_symbols=cfg.sleeves.symbols,
        adjust_dividends=cfg.data.adjust_dividends,
    )


def is_us(args) -> bool:
    return getattr(args, "market", "nse") == "us"


def build_us_config(args):
    """US round 1's configuration A (``us_engine.backtest.round1_configs``) with the run's overrides."""
    from us_engine.backtest import round1_configs

    cfg = round1_configs(args.universe)["A"]
    overrides: Dict[str, Any] = {}
    for item in getattr(args, "set", None) or []:
        key, _, value = item.partition("=")
        overrides[key] = _parse_value(value)
    for key in ("start", "end"):
        if getattr(args, key, None):
            overrides[key] = getattr(args, key)
    overrides["data.store_dir"] = find_us_store_dir(getattr(args, "store_dir", None))
    overrides["runs_dir"] = getattr(args, "runs_dir", None) or default_runs_dir()
    return cfg.replace(**overrides)


def load_us(cfg, universe: str):
    """The US store as engine data: every session from its 2006 start (warm-up), members of ``universe``."""
    from us_engine.backtest import load_us_market

    logger.info("Loading the US store (%s) from %s", universe, cfg.data.store_dir)
    return load_us_market(universe, panel_dir=Path(cfg.data.store_dir))


def parse_folds(spec: str, n_folds: int) -> List[int]:
    """``"all"``, ``"0-3"``, ``"0,2,5"`` or ``"2-"`` → fold indices."""
    if not spec or spec == "all":
        return list(range(n_folds))
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, _, hi = part.partition("-")
            out.extend(range(int(lo), (int(hi) if hi else n_folds - 1) + 1))
        elif part:
            out.append(int(part))
    return [i for i in out if 0 <= i < n_folds]


# ── workers ──────────────────────────────────────────────────────

def _evaluate(job: Dict[str, Any]) -> Dict[str, Any]:
    """One backtest in a worker process: train-window score for one grid point."""
    import pandas as pd

    from nse_engine.engine import run_backtest
    from nse_engine.validation.dsr import excess_sharpe
    # imported rather than copied so the window slicing cannot drift from
    # run_walk_forward's, which this runner has to match exactly
    from nse_engine.validation.walk_forward import _window_returns

    cfg = _BASE.replace(**job["params"], start=job["start"], end=job["end"])
    result = run_backtest(_DATA, cfg, record=True, tag=job["tag"])
    returns = _window_returns(result, pd.Timestamp(job["start"]), pd.Timestamp(job["end"]))
    return {
        "params": job["params"],
        "is_metric": float(excess_sharpe(returns, job["rf"])),
        "run_id": getattr(result, "run_id", ""),
        "n_days": int(returns.size),
    }


def _report(job: Dict[str, Any]) -> Dict[str, Any]:
    """One backtest that is reported, not recorded (a window outside the family's test window)."""
    import pandas as pd

    from nse_engine.engine import run_backtest
    from nse_engine.validation.walk_forward import _window_returns

    cfg = _BASE.replace(**job["params"], start=job["start"], end=job["end"])
    result = run_backtest(_DATA, cfg, record=False)
    returns = _window_returns(result, pd.Timestamp(job["start"]), pd.Timestamp(job["end"]))
    return {"params": job["params"], "window": [job["start"], job["end"]], "config_hash": cfg.config_hash(),
            "returns": {str(d.date()): float(r) for d, r in returns.items()}}


def _run_jobs(jobs: List[Dict[str, Any]], workers: int, fn=None) -> List[Dict[str, Any]]:
    """Evaluate jobs in order (``fn``, default a recorded backtest), in parallel when asked for more than one worker."""
    fn = fn or _evaluate
    if workers <= 1:
        return [fn(job) for job in jobs]
    # fork keeps the loaded panels shared; spawn would reload them per worker
    ctx = get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        return list(pool.map(fn, jobs))


# ── walk-forward ─────────────────────────────────────────────────

def provenance() -> Dict[str, str]:
    """Which interpreter and libraries produced a fold.

    Recorded because results are not identical across environments: the same
    toy fold scored 1.176 under Python 3.13 / pandas 3.0.2 and 1.116 on a
    Kaggle image running Python 3.12 with pandas 2.x.
    """
    import platform

    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "pandas", "pyarrow"):
        try:
            out[name] = __import__(name).__version__
        except Exception:                             # noqa: BLE001 - reporting only
            out[name] = "missing"
    return out


def job_signature(args, grid: List[Dict[str, Any]]) -> Dict[str, Any]:
    """What every session of one walk-forward must agree on.

    ``load_market_data`` applies its liquidity filter over the window it is
    asked for, so a session that loads a different ``end`` (or anchor) selects
    a different universe and its folds are not comparable with the others'.
    Measured on a 2-fold toy run: end 2018-12-31 vs 2019-12-31 moved the
    stitched OOS Sharpe by 0.003.
    """
    signature = {"start": args.start, "end": args.end, "data_start": args.data_start,
                 "train_years": args.train_years, "test_months": args.test_months,
                 "anchored": not args.rolling,
                 "grid": json.dumps(grid, sort_keys=True, default=str)}
    if is_us(args):                 # NSE signatures stay as recorded, so their resumes still match
        signature.update(market="us", universe=args.universe)
    return signature


def canonical_job(job: Dict[str, Any]) -> str:
    """A job signature that ignores the order grid points were expanded in.

    ``expand_grid`` walks keys in the order the grid JSON lists them, so the
    same set of points serialises differently if the keys are reordered (as a
    resume rebuilt from a fold file's signature does). Points are sorted
    here, so equal experiments compare equal.
    """
    j = dict(job or {})
    try:
        points = json.loads(j.get("grid") or "[]")
        j["grid"] = sorted(json.dumps(pt, sort_keys=True, default=str) for pt in points)
    except (TypeError, ValueError):
        pass
    return json.dumps(j, sort_keys=True, default=str)


def _check_signature(out_dir: Path, signature: Dict[str, Any], force: bool) -> None:
    from nse_engine.costs import COST_MODEL_VERSION

    for path in sorted(out_dir.glob("fold_*.json")):
        try:
            fold = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        previous = fold.get("job", {})
        if fold.get("cost_model") not in (None, COST_MODEL_VERSION):      # IC2: one cost model per walk-forward
            message = (f"{path.name} was produced under cost model {fold['cost_model']}, now {COST_MODEL_VERSION}: "
                       "use a fresh --out-dir, or --force to overwrite.")
            if not force:
                raise SystemExit(message)
            logger.warning("%s", message)
            return
        if not previous or canonical_job(previous) == canonical_job(signature):
            continue
        differing = {k: (previous.get(k), signature.get(k))
                     for k in set(previous) | set(signature) if previous.get(k) != signature.get(k)}
        message = (f"{path.name} was produced by a different job: "
                   + ", ".join(f"{k} was {was!r}, now {now!r}" for k, (was, now) in differing.items())
                   + "\nFolds from different windows cannot be stitched — use a fresh --out-dir, "
                     "or --force to overwrite.")
        if not force:
            raise SystemExit(message)
        logger.warning("%s", message)
        return


def run_walk_forward_folds(args) -> Dict[str, Any]:
    import numpy as np
    import pandas as pd

    from nse_engine.costs import COST_MODEL_VERSION
    from nse_engine.validation.walk_forward import expand_grid, generate_folds

    from cloud.heartbeat import Heartbeat

    global _DATA, _BASE

    cfg = build_us_config(args) if is_us(args) else build_config(args)
    grid = expand_grid(json.loads(args.grid))
    out_dir = Path(args.out_dir or default_out_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    signature = job_signature(args, grid)
    _check_signature(out_dir, signature, args.force)
    hb = Heartbeat(name=f"wf:{cfg.config_hash()}", state_path=out_dir / "heartbeat.json")

    _BASE = cfg
    _DATA = load_us(cfg, args.universe) if is_us(args) else load_data(cfg, args.data_start)
    rf = float(getattr(cfg, "risk_free_annual", 0.0))

    folds = generate_folds(_DATA.dates, args.train_years, args.test_months,
                           not args.rolling, start=cfg.start, end=cfg.end)
    if not folds:
        raise ValueError("no walk-forward folds fit inside the data/config window")
    wanted = parse_folds(args.folds, len(folds))
    hb.start(f"{len(wanted)} of {len(folds)} folds x {len(grid)} grid points, "
             f"{args.workers} workers", n_folds=len(folds), n_grid=len(grid))
    logger.info("Walk-forward: folds %s of %d, %d grid points, %d workers",
                args.folds or "all", len(folds), len(grid), args.workers)

    budget_s = args.max_hours * 3600
    started = time.time()
    done: List[int] = []
    skipped: List[int] = []
    slowest = 0.0

    for k in wanted:
        fold_path = out_dir / f"fold_{k:02d}.json"
        if fold_path.exists() and not args.force:
            logger.info("Fold %d already done (%s)", k, fold_path.name)
            skipped.append(k)
            continue
        elapsed = time.time() - started
        if done and elapsed + slowest * 1.1 > budget_s:
            hb.paused(f"stopping before fold {k}: {elapsed / 3600:.1f}h of "
                      f"{args.max_hours}h used", next_fold=k, done=done)
            logger.warning("Out of time budget before fold %d — resume with --folds %d-", k, k)
            break

        f = folds[k]
        tr_lo, tr_hi = f["train_start"], f["train_end"]
        te_lo, te_hi = f["test_start"], f["test_end"]
        t0 = time.time()
        hb.progress(f"fold {k}: {len(grid)} train backtests {tr_lo.date()}..{tr_hi.date()}",
                    fold=k, done=done)

        jobs = [{"params": p, "start": str(tr_lo.date()), "end": str(tr_hi.date()),
                 "tag": f"wfo-train:fold{k}", "rf": rf} for p in grid]
        scores = _run_jobs(jobs, args.workers)

        vals = np.array([s["is_metric"] for s in scores], dtype="float64")
        vals = np.where(np.isfinite(vals), vals, -np.inf)
        best = scores[int(np.argmax(vals))]

        test = _evaluate({"params": best["params"], "start": str(te_lo.date()),
                          "end": str(te_hi.date()), "tag": f"wfo-test:fold{k}", "rf": rf})
        returns_path = out_dir / f"fold_{k:02d}_oos.csv"
        _write_test_returns(cfg, test["run_id"], te_lo, te_hi, returns_path)

        row = {
            "fold": k,
            "train_start": str(tr_lo.date()), "train_end": str(tr_hi.date()),
            "test_start": str(te_lo.date()), "test_end": str(te_hi.date()),
            "params": best["params"], "is_metric": best["is_metric"],
            "oos_metric": test["is_metric"],
            "train_run_id": best["run_id"], "test_run_id": test["run_id"],
            "grid_scores": scores,
            "returns_file": returns_path.name,
            "seconds": round(time.time() - t0, 1),
            "config_hash": cfg.config_hash(),
            "cost_model": COST_MODEL_VERSION,
            "job": signature,
            "provenance": provenance(),
        }
        fold_path.write_text(json.dumps(row, indent=2, default=str))
        done.append(k)
        slowest = max(slowest, time.time() - t0)
        logger.info("Fold %d: IS=%.3f OOS=%.3f in %.0f min params=%s",
                    k, best["is_metric"], test["is_metric"], row["seconds"] / 60, best["params"])
        hb.progress(f"fold {k} done: IS={best['is_metric']:.2f} OOS={test['is_metric']:.2f}",
                    fold=k, done=done)

    state = {
        "task": "walk-forward",
        "config_hash": cfg.config_hash(),
        "n_folds": len(folds),
        "requested": wanted,
        "completed": done,
        "already_done": skipped,
        "remaining": [k for k in wanted if k not in done and k not in skipped],
        "elapsed_h": round((time.time() - started) / 3600, 2),
        "out_dir": str(out_dir),
        "provenance": provenance(),
    }
    (out_dir / "state.json").write_text(json.dumps(state, indent=2))
    if state["remaining"]:
        hb.paused(f"{len(done)} folds done, {len(state['remaining'])} left", **state)
    else:
        hb.done(f"all {len(wanted)} requested folds done", **state)
    return state


def _write_test_returns(cfg, run_id: str, lo, hi, dest: Path) -> None:
    """Copy the recorded OOS returns for one fold next to its fold file."""
    import pandas as pd

    src = Path(cfg.runs_dir) / run_id / "returns.csv"
    if not src.exists():
        logger.warning("no returns.csv for run %s", run_id)
        return
    s = pd.read_csv(src, index_col=0).squeeze("columns")
    s.index = pd.DatetimeIndex(s.index)
    s[(s.index >= lo) & (s.index <= hi)].to_csv(dest, header=["return"])


# ── grid sweep ───────────────────────────────────────────────────

def run_grid(args) -> Dict[str, Any]:
    """Full-window backtest of every grid point (Stage A), skipping recorded ones."""
    from nse_engine.validation.walk_forward import expand_grid

    from cloud.heartbeat import Heartbeat

    global _DATA, _BASE

    cfg = build_config(args)
    grid = expand_grid(json.loads(args.grid))
    out_dir = Path(args.out_dir or default_out_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    hb = Heartbeat(name=f"grid:{cfg.config_hash()}", state_path=out_dir / "heartbeat.json")

    _BASE = cfg
    _DATA = load_data(cfg, args.data_start)
    rf = float(getattr(cfg, "risk_free_annual", 0.0))

    known = _recorded_hashes(cfg.runs_dir, cfg.start, cfg.end)
    jobs, skipped = [], 0
    for params in grid:
        point_hash = cfg.replace(**params).config_hash()
        if point_hash in known and not args.force:
            skipped += 1
            continue
        jobs.append({"params": params, "start": cfg.start, "end": cfg.end,
                     "tag": args.tag or "grid", "rf": rf})

    hb.start(f"{len(jobs)} backtests ({skipped} already recorded)")
    logger.info("Grid: %d points, %d to run, %d already recorded", len(grid), len(jobs), skipped)
    results = _run_jobs(jobs, args.workers)
    results.sort(key=lambda r: r["is_metric"], reverse=True)

    state = {"task": "grid", "n_points": len(grid), "n_run": len(jobs),
             "n_skipped": skipped, "results": results, "out_dir": str(out_dir),
             "provenance": provenance()}
    (out_dir / "grid_results.json").write_text(json.dumps(state, indent=2, default=str))
    hb.done(f"{len(results)} backtests, best excess Sharpe "
            f"{results[0]['is_metric']:.3f}" if results else "nothing to run")
    return state


# ── configs: a registry refresh ──────────────────────────────────

_RUN_SPECIFIC = ("start", "end", "runs_dir")


def flatten_config(d: Dict[str, Any]) -> Dict[str, Any]:
    """An EngineConfig dict as dotted overrides for ``EngineConfig().replace``.

    Run-specific fields (window, runs_dir, store_dir) are dropped, so the
    result reproduces the configuration and its ``config_hash``.
    """
    out: Dict[str, Any] = {}
    for key, value in d.items():
        if key in _RUN_SPECIFIC:
            continue
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if (key, sub_key) != ("data", "store_dir"):
                    out[f"{key}.{sub_key}"] = sub_value
        else:
            out[key] = value
    return out


def run_configs(args) -> Dict[str, Any]:
    """Full-window backtest of each configuration in a JSON list.

    For a registry refresh: after a change that alters results without
    changing a config or the data (the cost model, tracker U25), every
    same-window configuration is re-run so PBO and the deflated Sharpe compare
    like with like.  The file holds ``[{"config": {...}, "tag": "..."}, ...]``;
    the configurations must share their data settings (one data load).
    """
    from nse_engine.config import EngineConfig

    from cloud.heartbeat import Heartbeat

    global _DATA, _BASE

    specs = json.loads(Path(args.configs).read_text())
    if not specs:
        raise SystemExit("no configurations in " + args.configs)
    first = flatten_config(specs[0]["config"])
    cfg = EngineConfig().replace(**first, start=args.start, end=args.end,
                                 **{"data.store_dir": find_store_dir(args.store_dir),
                                    "runs_dir": args.runs_dir or default_runs_dir()})
    out_dir = Path(args.out_dir or default_out_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    hb = Heartbeat(name=f"configs:{len(specs)}", state_path=out_dir / "heartbeat.json")
    _BASE = cfg
    _DATA = load_data(cfg, args.data_start)
    rf = float(getattr(cfg, "risk_free_annual", 0.0))
    jobs = [{"params": flatten_config(s["config"]), "start": cfg.start, "end": cfg.end,
             "tag": s.get("tag") or args.tag, "rf": rf} for s in specs]
    hb.start(f"{len(jobs)} backtests")
    logger.info("Configs: %d to run", len(jobs))
    results = _run_jobs(jobs, args.workers)
    state = {"task": "configs", "n_run": len(results), "results": results, "out_dir": str(out_dir),
             "provenance": provenance()}
    (out_dir / "configs_results.json").write_text(json.dumps(state, indent=2, default=str))
    hb.done(f"{len(results)} backtests")
    return state


def run_sepa_task(args) -> Dict[str, Any]:
    """The SEPA-lite breakout sleeve (tracker O9-A, plan 5x): one configuration, its own registry."""
    from dataclasses import replace

    from nse_engine.sepa import SepaConfig, record, run_sepa

    cfg = SepaConfig()
    res = run_sepa(replace(cfg, start=args.start or cfg.start, end=args.end or cfg.end),
                   store_dir=find_store_dir(args.store_dir))
    runs_dir = args.runs_dir or default_runs_dir()
    state = {"task": "sepa", "run_dir": record(res, runs_dir), "metrics": res.metrics, "provenance": provenance()}
    out_dir = Path(args.out_dir or default_out_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sepa_results.json").write_text(json.dumps(state, indent=2, default=str))
    return state


def run_value_task(args) -> Dict[str, Any]:
    """The value / quality sleeve (tracker GT4, plan 5z): GT4-A and GT4-B into the value family's registry."""
    from dataclasses import replace

    from nse_engine.value import CONFIGS, ValueConfig, load_inputs, record, run_value

    end = args.end or ValueConfig.end
    data, quarters, fhash = load_inputs(find_store_dir(args.store_dir), find_fundamentals_dir(args.fundamentals_dir), end)
    runs_dir = args.runs_dir or default_runs_dir()
    results = {}
    for key in args.configs.split(","):
        res = run_value(replace(CONFIGS[key], start=args.start or ValueConfig.start, end=end), data, quarters, fhash)
        results[key] = {"run_dir": record(res, runs_dir), "metrics": res.metrics}
    state = {"task": "value", "results": results, "fundamentals_hash": fhash, "provenance": provenance()}
    out_dir = Path(args.out_dir or default_out_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "value_results.json").write_text(json.dumps(state, indent=2, default=str))
    return state


def run_mcx_task(args) -> Dict[str, Any]:
    """The MCX commodity trend sleeve (tracker GT5, plan 5y): GT5-A and GT5-B into the commodity family's
    registry; ``--reported-start`` adds the wider window, reported and not recorded."""
    from dataclasses import replace

    from nse_engine.mcx_trend import CONFIGS, TrendConfig, prepare, record, run_trend

    prep = prepare(find_mcx_store_dir(args.mcx_store))
    runs_dir = args.runs_dir or default_runs_dir()
    results, reported = {}, {}
    for key in args.configs.split(","):
        cfg = replace(CONFIGS[key], start=args.start or TrendConfig.start, end=args.end or TrendConfig.end)
        res = run_trend(cfg, prep)
        results[key] = {"run_dir": record(res, runs_dir), "metrics": res.metrics}
        if args.reported_start:
            wide = run_trend(replace(cfg, start=args.reported_start), prep)
            reported[key] = {"window": [args.reported_start, cfg.end], "metrics": wide.metrics,
                             "returns": {str(d.date()): float(r) for d, r in wide.returns.items()}}
    state = {"task": "mcx", "results": results, "reported": reported, "data_hash": prep["data_hash"],
             "provenance": provenance()}
    out_dir = Path(args.out_dir or default_out_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mcx_results.json").write_text(json.dumps(state, indent=2, default=str))
    return state


def run_us_round(args) -> Dict[str, Any]:
    """US round 1 (US tracker US5 / US6, plan 5ad): A, B and C over the test window into the US registry,
    the wider window reported (not recorded), then the anchored walk-forward choosing among the three."""
    from us_engine.backtest import round1_points

    global _DATA, _BASE

    points = round1_points(args.universe)
    _BASE = build_us_config(argparse.Namespace(universe=args.universe, start=args.start, end=args.end,
                                               store_dir=args.store_dir, runs_dir=args.runs_dir))
    _DATA = load_us(_BASE, args.universe)
    rf = float(_BASE.risk_free_annual)
    recorded = _run_jobs([{"params": p, "start": args.start, "end": args.end, "tag": f"us5-{k}", "rf": rf}
                          for k, p in points.items()], args.workers)
    reported = _run_jobs([{"params": p, "start": args.reported_start, "end": args.end}
                          for p in points.values()], args.workers, fn=_report)
    wf_args = argparse.Namespace(
        market="us", universe=args.universe, set=None, start=args.start, end=args.end, data_start=None,
        store_dir=args.store_dir, runs_dir=args.runs_dir, out_dir=args.out_dir, workers=args.workers,
        force=False, folds="all", train_years=args.train_years, test_months=args.test_months, rolling=False,
        max_hours=args.max_hours, grid=json.dumps(list(points.values())))
    wf_state = run_walk_forward_folds(wf_args)
    state = {"task": "us-round", "universe": args.universe, "points": points,
             "results": dict(zip(points, recorded)), "reported": dict(zip(points, reported)),
             "walk_forward": wf_state, "data_hash": _DATA.data_hash, "provenance": provenance()}
    out_dir = Path(args.out_dir or default_out_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "us_round.json").write_text(json.dumps(state, indent=2, default=str))
    return state


def run_options(args) -> Dict[str, Any]:
    """Backtest the pre-registered options-family candidates (O2 sleeves, O3 signal futures, O4 F&O
    anomalies) and record them."""
    from dataclasses import replace

    from kite_connect.options import book_rounds, fo_anomalies, signal_futures
    from kite_connect.options.structures import record as record_structure
    from kite_connect.options.structures import run_structure
    from kite_connect.options.backtest import record, run_sleeve
    from kite_connect.options.sleeves import CANDIDATES

    store = find_fo_store_dir(args.fo_store)
    runs_dir = args.runs_dir or default_runs_dir()
    results = {}
    for key in args.candidates.split(","):
        if key in signal_futures.CANDIDATES:           # round 2 (plan 5r)
            cfg = signal_futures.CANDIDATES[key]
            res = signal_futures.run_strategy(replace(cfg, start=args.start or cfg.start, end=args.end or cfg.end), store)
            results[key] = {"run_dir": signal_futures.record(res, runs_dir), "metrics": res.metrics}
            logger.info("%s recorded in %s", key, results[key]["run_dir"])
            continue
        if key in book_rounds.CANDIDATES:              # rounds O5 onward: the owner's books (plan 5x)
            res = run_structure(book_rounds.CANDIDATES[key], store, args.start or "2013-01-01", args.end or "2025-12-31")
            results[key] = {"run_dir": record_structure(res, runs_dir, book_rounds.round_of(key)), "metrics": res.metrics}
            logger.info("%s recorded in %s", key, results[key]["run_dir"])
            continue
        if key in fo_anomalies.CANDIDATES:             # round 3 (plan 5s)
            cfg = fo_anomalies.CANDIDATES[key]
            res = fo_anomalies.run(replace(cfg, start=args.start or cfg.start, end=args.end or cfg.end), store)
            results[key] = {"run_dir": fo_anomalies.record(res, runs_dir), "metrics": res.metrics}
            logger.info("%s recorded in %s", key, results[key]["run_dir"])
            continue
        cfg = CANDIDATES[key]
        res = run_sleeve(replace(cfg, start=args.start or cfg.start, end=args.end or cfg.end), store)
        results[key] = {"run_dir": record(res, runs_dir), "metrics": res.metrics}
        logger.info("%s recorded in %s", key, results[key]["run_dir"])
    out_dir = Path(args.out_dir or default_out_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {"task": "options", "results": results, "fo_store": store, "provenance": provenance()}
    (out_dir / "options_results.json").write_text(json.dumps(state, indent=2, default=str))
    return state


def _recorded_hashes(runs_dir: str, start: str, end: str) -> set:
    """Config hashes already in the run registry over this window under today's cost model, so a resumed sweep
    skips them; one recorded under an older cost model or another window is run again (IC2)."""
    from nse_engine.costs import COST_MODEL_VERSION
    from nse_engine.validation.trials import run_cost_model

    out = set()
    for manifest in Path(runs_dir).glob("*/manifest.json"):
        try:
            m = json.loads(manifest.read_text())
        except (OSError, ValueError):
            continue
        if (str(m.get("start")), str(m.get("end"))) == (str(start), str(end)) and run_cost_model(m) == COST_MODEL_VERSION:
            out.add(m.get("config_hash", ""))
    return out


# ── CLI ──────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--config", help="JSON file with an EngineConfig")
        p.add_argument("--set", action="append", metavar="KEY=VALUE",
                       help="dotted EngineConfig override, repeatable")
        p.add_argument("--grid", required=True, help='JSON, e.g. {"portfolio.target_positions": [20, 30]}')
        p.add_argument("--start")
        p.add_argument("--end")
        p.add_argument("--data-start", help="anchor date, e.g. 2011-01-01")
        p.add_argument("--store-dir", help="parquet store (default: attached dataset, else repo)")
        p.add_argument("--runs-dir", help="where runs are recorded (default: /kaggle/working/runs)")
        p.add_argument("--out-dir", help="fold files and state (default: /kaggle/working/wf)")
        p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
        p.add_argument("--force", action="store_true", help="redo work that is already recorded")

    p = sub.add_parser("walk-forward", help="anchored walk-forward, one fold file per fold")
    common(p)
    p.add_argument("--folds", default="all", help='"all", "0-3", "0,2,5" or "2-"')
    p.add_argument("--train-years", type=int, default=4)
    p.add_argument("--test-months", type=int, default=12)
    p.add_argument("--rolling", action="store_true")
    p.add_argument("--max-hours", type=float, default=11.0,
                   help="stop between folds before the session is killed")
    p.set_defaults(func=run_walk_forward_folds)

    p = sub.add_parser("grid", help="full-window backtest of every grid point")
    common(p)
    p.add_argument("--tag", default="grid")
    p.set_defaults(func=run_grid)

    p = sub.add_parser("configs", help="full-window backtest of each config in a JSON list (registry refresh)")
    p.add_argument("--configs", required=True, help='JSON file: [{"config": {...}, "tag": "..."}, ...]')
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--data-start", help="anchor date, e.g. 2012-01-02")
    p.add_argument("--store-dir")
    p.add_argument("--runs-dir")
    p.add_argument("--out-dir")
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--tag", default="refresh")
    p.set_defaults(func=run_configs)

    p = sub.add_parser("sepa", help="the SEPA-lite breakout sleeve (O9-A, plan 5x)")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--store-dir", help="equity store (default: attached dataset, else repo)")
    p.add_argument("--runs-dir", help="registry (default: /kaggle/working/runs)")
    p.add_argument("--out-dir")
    p.set_defaults(func=run_sepa_task)

    p = sub.add_parser("value", help="the value / quality sleeve (GT4-A, GT4-B, plan 5z)")
    p.add_argument("--configs", default="GT4-A,GT4-B")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--store-dir", help="equity store (default: attached dataset, else repo)")
    p.add_argument("--fundamentals-dir", help="GT1's results store (default: attached dataset, else repo)")
    p.add_argument("--runs-dir", help="registry (default: /kaggle/working/runs)")
    p.add_argument("--out-dir")
    p.set_defaults(func=run_value_task)

    p = sub.add_parser("mcx", help="the MCX commodity trend sleeve (GT5-A, GT5-B, plan 5y)")
    p.add_argument("--configs", default="GT5-A,GT5-B")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--reported-start", help="also run from this date, reported only (plan 5y: 2007-01-01)")
    p.add_argument("--mcx-store", help="GT2's MCX store (default: attached dataset, else repo)")
    p.add_argument("--runs-dir", help="registry (default: /kaggle/working/runs)")
    p.add_argument("--out-dir")
    p.set_defaults(func=run_mcx_task)

    p = sub.add_parser("us-round", help="US round 1: A, B, C on the US store, then their walk-forward (plan 5ad)")
    p.add_argument("--universe", default="union", choices=["union", "sp500", "ndx"])
    p.add_argument("--start", default="2013-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--reported-start", default="2007-01-01", help="the wider window, reported and not recorded")
    p.add_argument("--train-years", type=int, default=4)
    p.add_argument("--test-months", type=int, default=12)
    p.add_argument("--max-hours", type=float, default=11.0)
    p.add_argument("--store-dir", help="US store panel (default: attached dataset, else repo)")
    p.add_argument("--runs-dir", help="US registry (default: /kaggle/working/runs)")
    p.add_argument("--out-dir")
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    p.set_defaults(func=run_us_round)

    p = sub.add_parser("options", help="the pre-registered options-family candidates (O2 A1/A2/B, O3 X1-X4, O4 Y1-Y3, "
                                       "O5-O8 from kite_connect.options.book_rounds)")
    p.add_argument("--candidates", default="A1,A2,B")
    p.add_argument("--start", help="window override (plan 5q addendum: 2007-01-02)")
    p.add_argument("--end")
    p.add_argument("--fo-store", help="F&O store (default: attached dataset, else repo)")
    p.add_argument("--runs-dir", help="options registry (default: /kaggle/working/runs)")
    p.add_argument("--out-dir")
    p.set_defaults(func=run_options)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    state = args.func(args)
    # results and reported windows (daily returns) go to the output files, not the log
    print(json.dumps({k: v for k, v in state.items() if k not in ("results", "reported")}, indent=2, default=str))


if __name__ == "__main__":
    main()
