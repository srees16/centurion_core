"""Join Kaggle fold files into one walk-forward result, and import its runs.

``cloud.kaggle_runner`` writes ``fold_NN.json`` and ``fold_NN_oos.csv`` per
fold, possibly across several sessions. This stitches them into the dictionary
``nse_engine.validation.walk_forward.run_walk_forward`` would have returned —
same summary fields, same stitched OOS series — and copies the recorded runs
into the local registry so PBO and the deflated Sharpe count them.

    python -m cloud.wf_stitch stitch --dirs data/nse_engine/kaggle_out/*/wf
    python -m cloud.wf_stitch import-runs --src data/nse_engine/kaggle_out/session1/runs
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger("wf_stitch")


def collect_folds(dirs: List[str]) -> List[Dict[str, Any]]:
    """Every fold file under the given directories, in fold order."""
    rows: List[Dict[str, Any]] = []
    for d in dirs:
        base = Path(d)
        for path in sorted(base.glob("fold_*.json")):
            row = json.loads(path.read_text())
            row["_dir"] = str(base)
            row["_mtime"] = path.stat().st_mtime
            rows.append(row)
    return sorted(rows, key=lambda r: (int(r["fold"]), r["_mtime"]))


def latest_per_fold(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per fold index — the most recently written wins."""
    best: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        k = int(row["fold"])
        if k not in best or row["_mtime"] > best[k]["_mtime"]:
            best[k] = row
    return [best[k] for k in sorted(best)]


def stitch(dirs: List[str], rf_annual: Optional[float] = None,
           allow_mixed_env: bool = False) -> Dict[str, Any]:
    import numpy as np
    import pandas as pd

    from nse_engine.config import EngineConfig
    from nse_engine.validation.dsr import excess_sharpe, performance_summary

    every = collect_folds(dirs)
    if not every:
        raise SystemExit(f"no fold_*.json found under {dirs}")
    # Checked over every file, not just the ones that survive de-duplication:
    # two directories holding the same fold from different platforms must be
    # caught, not silently resolved by whichever was written last.
    _require_one_job(every)
    environments = _check_provenance(every, allow_mixed_env)
    folds = latest_per_fold(every)
    rf = rf_annual if rf_annual is not None else float(getattr(EngineConfig(), "risk_free_annual", 0.0))

    parts = []
    for row in folds:
        path = Path(row["_dir"]) / row.get("returns_file", f"fold_{int(row['fold']):02d}_oos.csv")
        if not path.exists():
            raise SystemExit(f"fold {row['fold']} has no returns file at {path}")
        s = pd.read_csv(path, index_col=0).squeeze("columns")
        s.index = pd.DatetimeIndex(s.index)
        parts.append(s.astype("float64"))

    stitched = pd.concat(parts).sort_index()
    if stitched.index.has_duplicates:
        raise SystemExit("overlapping OOS windows — two sessions ran the same fold differently")

    gaps = _fold_gaps(folds)
    is_vals = np.array([r["is_metric"] for r in folds], dtype="float64")
    oos_vals = np.array([r["oos_metric"] for r in folds], dtype="float64")
    oos_sharpe = excess_sharpe(stitched, rf)
    mean_is = float(np.nanmean(is_vals)) if is_vals.size else float("nan")
    summary = {
        "oos_sharpe": oos_sharpe,
        "mean_is_sharpe": mean_is,
        "mean_oos_sharpe": float(np.nanmean(oos_vals)) if oos_vals.size else float("nan"),
        "sharpe_degradation": mean_is - oos_sharpe,
        "oos_is_ratio": oos_sharpe / mean_is if mean_is and np.isfinite(mean_is) else float("nan"),
        "n_folds": len(folds),
        "n_grid_points": len(folds[0].get("grid_scores", [])),
        "n_backtests": sum(len(r.get("grid_scores", [])) + 1 for r in folds),
        "negative_oos_years": int((oos_vals <= 0).sum()),
        "missing_folds": gaps,
        "environments": environments,
        "oos_performance": performance_summary(stitched, rf),
    }
    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in folds]
    return {"oos_returns": stitched, "folds": rows, "summary": summary}


def _require_one_job(folds: List[Dict[str, Any]]) -> None:
    """Refuse to stitch folds produced by different windows or grids.

    The loaded window decides which symbols pass the liquidity filter, so folds
    run with a different ``end`` or anchor describe a different universe and
    their OOS series do not belong in one series.
    """
    from cloud.kaggle_runner import canonical_job

    jobs = {canonical_job(r.get("job", {})) for r in folds}
    if len(jobs) > 1:
        lines = []
        for spec in sorted(jobs):
            which = [r["fold"] for r in folds if canonical_job(r.get("job", {})) == spec]
            lines.append(f"  folds {which}: {spec}")
        raise SystemExit("fold files come from different jobs:\n" + "\n".join(lines))


def _check_provenance(folds: List[Dict[str, Any]], allow_mixed: bool) -> List[str]:
    """Refuse to stitch folds produced on different platforms.

    Measured: one fold, identical config hash and identical data hash, run on
    macOS/arm64 and on Kaggle's Linux/x86_64 with the same numpy, pandas and
    pyarrow versions. The two agreed for 84 sessions, then a marginal selection
    on 2016-05-09 went different ways and the paths compounded apart — 831
    trades against 850, train Sharpe 1.176 against 1.116. Each platform is
    internally deterministic, so a walk-forward is sound as long as it runs in
    one place; stitched across two it is not one experiment.
    """
    # A fold file without provenance counts as its own environment rather than
    # being skipped: "unknown" is exactly the case that must not pass silently.
    seen = sorted({json.dumps(r.get("provenance") or {"provenance": "not recorded"},
                              sort_keys=True) for r in folds})
    if len(seen) <= 1:
        return seen
    detail = "\n".join(f"  {spec}" for spec in seen)
    if not allow_mixed:
        raise SystemExit(f"folds come from {len(seen)} different environments:\n{detail}\n"
                         "Re-run the whole walk-forward in one place, or pass --allow-mixed-env "
                         "to stitch anyway (the summary records both).")
    logger.warning("stitching across %d environments:\n%s", len(seen), detail)
    return seen


def _fold_gaps(folds: List[Dict[str, Any]]) -> List[int]:
    """Fold indices missing from a run that otherwise looks contiguous."""
    seen = {int(r["fold"]) for r in folds}
    return [k for k in range(max(seen) + 1) if k not in seen]


def import_runs(src: str, dest: str = "data/nse_engine/runs") -> Dict[str, int]:
    """Copy recorded run directories into the local registry, skipping duplicates."""
    src_path, dest_path = Path(src), Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)
    copied = skipped = 0
    for run in sorted(p for p in src_path.iterdir() if p.is_dir()):
        target = dest_path / run.name
        if target.exists():
            skipped += 1
            continue
        shutil.copytree(run, target)
        copied += 1
    logger.info("imported %d runs into %s (%d already present)", copied, dest, skipped)
    return {"copied": copied, "skipped": skipped, "dest": str(dest_path)}


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("stitch", help="join fold files into one walk-forward result")
    p.add_argument("--dirs", nargs="+", required=True)
    p.add_argument("--rf", type=float, default=None, help="risk-free rate (default: EngineConfig)")
    p.add_argument("--out", default="data/nse_engine/wf_stitched.json")
    p.add_argument("--returns-out", default="data/nse_engine/wf_oos_returns.csv")
    p.add_argument("--allow-mixed-env", action="store_true",
                   help="stitch folds produced on different platforms (they do not "
                        "reproduce each other; see the module docstring)")

    p = sub.add_parser("import-runs", help="copy Kaggle run directories into the registry")
    p.add_argument("--src", required=True)
    p.add_argument("--dest", default="data/nse_engine/runs")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.command == "stitch":
        result = stitch(args.dirs, args.rf, args.allow_mixed_env)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"folds": result["folds"], "summary": result["summary"]}, indent=2, default=str))
        result["oos_returns"].to_csv(args.returns_out, header=["return"])
        print(json.dumps(result["summary"], indent=2, default=str))
        print(f"\nfolds -> {args.out}\nOOS returns -> {args.returns_out}")
    else:
        print(json.dumps(import_runs(args.src, args.dest), indent=2))


if __name__ == "__main__":
    main()
