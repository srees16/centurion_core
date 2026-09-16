"""Run NSE-engine research jobs on Kaggle — fold by fold, several at a time.

Why this exists: the 2013–2025 walk-forward is 775 backtests at roughly 108 s
each, about 23 hours on one core. A Kaggle CPU session gives 4 cores and 30 GB
but stops at 12 hours, so this runner

  * evaluates each fold's grid across worker processes (4 cores ≈ 6 hours), and
  * writes one file per finished fold, so the next session resumes where the
    last one stopped instead of starting again.

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


def find_store_dir(explicit: Optional[str] = None) -> str:
    """Locate the parquet store: an explicit path, an attached dataset, or the repo."""
    if explicit:
        return explicit
    for base in sorted(KAGGLE_INPUT.glob("*")) if KAGGLE_INPUT.is_dir() else []:
        for candidate in (base / "store", base / "data" / "nse_engine" / "store"):
            if candidate.is_dir():
                return str(candidate)
    return "data/nse_engine/store"


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
        include_symbols=(cfg.sleeves.gold_symbol, cfg.sleeves.silver_symbol),
        adjust_dividends=cfg.data.adjust_dividends,
    )


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


def _run_jobs(jobs: List[Dict[str, Any]], workers: int) -> List[Dict[str, Any]]:
    """Evaluate jobs in order, in parallel when asked for more than one worker."""
    if workers <= 1:
        return [_evaluate(job) for job in jobs]
    # fork keeps the loaded panels shared; spawn would reload them per worker
    ctx = get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        return list(pool.map(_evaluate, jobs))


# ── walk-forward ─────────────────────────────────────────────────

def job_signature(args, grid: List[Dict[str, Any]]) -> Dict[str, Any]:
    """What every session of one walk-forward must agree on.

    ``load_market_data`` applies its liquidity filter over the window it is
    asked for, so a session that loads a different ``end`` (or anchor) selects
    a different universe and its folds are not comparable with the others'.
    Measured on a 2-fold toy run: end 2018-12-31 vs 2019-12-31 moved the
    stitched OOS Sharpe by 0.003.
    """
    return {"start": args.start, "end": args.end, "data_start": args.data_start,
            "train_years": args.train_years, "test_months": args.test_months,
            "anchored": not args.rolling,
            "grid": json.dumps(grid, sort_keys=True, default=str)}


def _check_signature(out_dir: Path, signature: Dict[str, Any], force: bool) -> None:
    for path in sorted(out_dir.glob("fold_*.json")):
        try:
            previous = json.loads(path.read_text()).get("job", {})
        except (OSError, ValueError):
            continue
        if not previous or previous == signature:
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

    from nse_engine.validation.walk_forward import expand_grid, generate_folds

    from cloud.heartbeat import Heartbeat

    global _DATA, _BASE

    cfg = build_config(args)
    grid = expand_grid(json.loads(args.grid))
    out_dir = Path(args.out_dir or default_out_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    signature = job_signature(args, grid)
    _check_signature(out_dir, signature, args.force)
    hb = Heartbeat(name=f"wf:{cfg.config_hash()}", state_path=out_dir / "heartbeat.json")

    _BASE = cfg
    _DATA = load_data(cfg, args.data_start)
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
            "job": signature,
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

    known = _recorded_hashes(cfg.runs_dir)
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
             "n_skipped": skipped, "results": results, "out_dir": str(out_dir)}
    (out_dir / "grid_results.json").write_text(json.dumps(state, indent=2, default=str))
    hb.done(f"{len(results)} backtests, best excess Sharpe "
            f"{results[0]['is_metric']:.3f}" if results else "nothing to run")
    return state


def _recorded_hashes(runs_dir: str) -> set:
    """Config hashes already in the run registry, so a resumed sweep skips them."""
    out = set()
    for manifest in Path(runs_dir).glob("*/manifest.json"):
        try:
            out.add(json.loads(manifest.read_text()).get("config_hash", ""))
        except (OSError, ValueError):
            continue
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

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    state = args.func(args)
    print(json.dumps({k: v for k, v in state.items() if k != "results"}, indent=2, default=str))


if __name__ == "__main__":
    main()
