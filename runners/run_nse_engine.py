"""
Command-line entry point for the NSE engine.

    python -m runners.run_nse_engine sync --start 2012-01-01
    python -m runners.run_nse_engine build-store
    python -m runners.run_nse_engine backtest --tag baseline
    python -m runners.run_nse_engine backtest --set portfolio.target_positions=25 --tag tp25
    python -m runners.run_nse_engine validate --run-id <run_id>
    python -m runners.run_nse_engine walk-forward --grid '{"portfolio.target_positions": [15, 20, 30]}'
    python -m runners.run_nse_engine holdout --start 2026-01-01 --end 2026-09-11

Every backtest is recorded under ``EngineConfig.runs_dir`` so that PBO and the
deflated Sharpe ratio cover every configuration ever evaluated.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from nse_engine.config import EngineConfig  # noqa: E402

logger = logging.getLogger("run_nse_engine")

ARCHIVE_DIR = "data/nse_engine/archive"


def _parse_value(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _build_config(args) -> EngineConfig:
    cfg = EngineConfig()
    if getattr(args, "config", None):
        cfg = EngineConfig.from_dict(json.loads(Path(args.config).read_text()))
    overrides = {}
    for item in getattr(args, "set", None) or []:
        key, _, value = item.partition("=")
        overrides[key] = _parse_value(value)
    for key in ("start", "end"):
        if getattr(args, key, None):
            overrides[key] = getattr(args, key)
    return cfg.replace(**overrides) if overrides else cfg


def _load_data(cfg: EngineConfig, warmup_years: int = 2, data_start: str = None):
    """Load market data from ``data_start`` (the anchor) or Jan 1, ``warmup_years`` before start.

    Rebalance days and expanding normalisers count from the first row, so runs that
    are compared (validation, holdout, shift reference, live) must share the anchor.
    """
    from nse_engine.data.panel import load_market_data

    start = (date.fromisoformat(data_start) if data_start else
             date(date.fromisoformat(cfg.start).year - warmup_years, 1, 1))
    return load_market_data(
        cfg.data.store_dir,
        start.isoformat(),
        cfg.end,
        series=cfg.data.series,
        min_median_value_inr=cfg.data.load_min_median_value_inr,
        float_dtype=cfg.data.float_dtype,
        include_symbols=(cfg.sleeves.gold_symbol, cfg.sleeves.silver_symbol),
        adjust_dividends=cfg.data.adjust_dividends,
    )


def _print_json(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_sync(args) -> None:
    from nse_engine.data.archive import BhavcopyArchive

    archive = BhavcopyArchive(ARCHIVE_DIR, requests_per_second=args.rps)
    _print_json(archive.sync_reference())
    end = date.fromisoformat(args.end) if args.end else date.today()
    _print_json(archive.sync(date.fromisoformat(args.start), end))


def cmd_build_store(args) -> None:
    from nse_engine.data.reference import build_sector_map
    from nse_engine.data.store import build_store

    cfg = EngineConfig()
    _print_json(build_store(ARCHIVE_DIR, cfg.data.store_dir))
    sectors = build_sector_map(ARCHIVE_DIR)
    print(f"sector map: {len(sectors)} symbols")


def cmd_backtest(args) -> None:
    from nse_engine.engine import run_backtest

    cfg = _build_config(args)
    data = _load_data(cfg, data_start=getattr(args, "data_start", None))
    result = run_backtest(data, cfg, record=True, tag=args.tag, lag_days=args.lag_days)
    _print_json({"run_id": result.run_id, "run_dir": result.run_dir, "metrics": result.metrics})


def cmd_validate(args) -> None:
    from nse_engine.validation.benchmarks import benchmark_gate, run_benchmarks
    from nse_engine.validation.dsr import deflated_sharpe
    from nse_engine.validation.pbo import cscv_pbo
    from nse_engine.validation.trials import TrialRegistry

    cfg = EngineConfig()
    registry = TrialRegistry(cfg.runs_dir)
    trials = registry.list_trials()
    if trials.empty:
        raise SystemExit("no recorded runs; run `backtest` first")
    run_id = args.run_id or trials.sort_values("created_at").iloc[-1]["run_id"]
    run_dir = Path(cfg.runs_dir) / run_id
    run_cfg = EngineConfig.from_dict(json.loads((run_dir / "config.json").read_text()))
    manifest = json.loads((run_dir / "manifest.json").read_text())

    window = (manifest.get("start"), manifest.get("end"))
    matrix = registry.returns_matrix(data_hash=manifest.get("data_hash"), window=window)
    returns = matrix[run_id] if run_id in matrix.columns else None
    if returns is None:
        raise SystemExit(f"run {run_id} not found in the returns matrix")

    n_configs = int(matrix.shape[1])
    report = {"run_id": run_id, "window": window, "n_configurations": n_configs,
              "metrics": manifest.get("metrics")}
    trials = matrix if n_configs > 1 else None
    # N = every recorded configuration (parameter variants are too correlated
    # for clustering to count them); the clustered figure is informational.
    report["dsr"] = deflated_sharpe(returns, trials_matrix=trials, rf_annual=run_cfg.risk_free_annual)
    report["dsr_clustered"] = deflated_sharpe(returns, trials_matrix=trials, rf_annual=run_cfg.risk_free_annual,
                                              trial_count="clustered")
    if n_configs >= 2:
        pbo = {k: v for k, v in cscv_pbo(matrix, n_splits=args.splits).items() if k != "logits"}
        pbo["verdict"] = ("likely real" if pbo["pbo"] < 0.30
                          else "caution" if pbo["pbo"] <= 0.50 else "reject")
        report["pbo"] = pbo
    else:
        report["pbo"] = "needs at least 2 recorded configurations"

    data = _load_data(run_cfg)
    benchmarks = run_benchmarks(data, run_cfg)
    report["benchmark_gate"] = benchmark_gate(returns, benchmarks, margin=args.margin,
                                              rf_annual=run_cfg.risk_free_annual)
    _print_json(report)
    out = run_dir / "validation.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"saved {out}")


def cmd_walk_forward(args) -> None:
    from nse_engine.validation.walk_forward import run_walk_forward

    cfg = _build_config(args)
    data = _load_data(cfg, data_start=getattr(args, "data_start", None))
    grid = json.loads(args.grid)
    result = run_walk_forward(data, cfg, grid, train_years=args.train_years,
                              test_months=args.test_months, anchored=not args.rolling)
    _print_json({k: v for k, v in result.items() if k != "oos_returns"})


def cmd_holdout(args) -> None:
    from nse_engine.validation.holdout import run_holdout

    cfg = _build_config(args)
    data = _load_data(cfg, data_start=getattr(args, "data_start", None))
    _print_json(run_holdout(data, cfg, args.start, args.end, force=args.force))


PROMOTION_GATES = {
    "pbo_max": 0.30,                  # CSCV PBO over every same-window configuration
    "dsr_min": 0.95,                  # deflated Sharpe, N = raw configuration count
    "benchmark_gate": True,           # beats EW hold and naive momentum by the margin
    "holdout_excess_sharpe_min": 0.0,
    "holdout_maxdd_ratio_max": 1.5,   # holdout MaxDD <= 1.5x full-period backtest MaxDD
}


def promotion_checks(run_cfg: EngineConfig, manifest: dict, validation: dict, holdout_lock: dict) -> list:
    """(name, passed, detail) for each promotion gate."""
    g = PROMOTION_GATES
    checks = []
    pbo = validation.get("pbo")
    pbo_val = pbo.get("pbo") if isinstance(pbo, dict) else None
    checks.append(("pbo", pbo_val is not None and pbo_val < g["pbo_max"],
                   f"PBO={pbo_val} (< {g['pbo_max']}, n={validation.get('n_configurations')})"))
    dsr = (validation.get("dsr") or validation.get("dsr_raw_count") or {}).get("dsr")
    checks.append(("dsr", dsr is not None and dsr >= g["dsr_min"], f"DSR={dsr} (>= {g['dsr_min']})"))
    gate = (validation.get("benchmark_gate") or {}).get("passed")
    checks.append(("benchmark_gate", bool(gate) == g["benchmark_gate"], f"passed={gate}"))

    evals = [e for e in holdout_lock.get("evaluations", [])
             if e.get("config_hash") == run_cfg.config_hash() and e.get("status") == "completed"]
    if not evals:
        checks.append(("holdout", False, "no completed holdout evaluation for this config"))
        return checks
    hm = evals[-1].get("metrics", {})
    hs = hm.get("excess_sharpe")
    checks.append(("holdout_sharpe", hs is not None and hs > g["holdout_excess_sharpe_min"],
                   f"holdout excess Sharpe={hs} ({evals[-1]['window']['start']}..{evals[-1]['window']['end']})"))
    bt_dd = abs((manifest.get("metrics") or {}).get("max_drawdown") or 0.0)
    h_dd = abs(hm.get("max_drawdown") or 0.0)
    checks.append(("holdout_maxdd", bt_dd > 0 and h_dd <= g["holdout_maxdd_ratio_max"] * bt_dd,
                   f"holdout MaxDD={h_dd:.3f} vs backtest {bt_dd:.3f} (<= {g['holdout_maxdd_ratio_max']}x)"))
    return checks


def cmd_promote(args) -> None:
    """Write config/nse_engine_deployed.json for a run that passed validation and the holdout."""
    from datetime import datetime, timedelta, timezone

    from nse_engine.deployment import load_deployment, resolve_path

    runs_dir = Path(EngineConfig().runs_dir)
    run_dir = runs_dir / args.run_id
    for name in ("config.json", "manifest.json", "validation.json"):
        if not (run_dir / name).exists():
            raise SystemExit(f"{run_dir / name} missing (run `validate --run-id {args.run_id}` first)")
    run_cfg = EngineConfig.from_dict(json.loads((run_dir / "config.json").read_text()))
    manifest = json.loads((run_dir / "manifest.json").read_text())
    validation = json.loads((run_dir / "validation.json").read_text())
    lock_path = Path(args.holdout_lock)
    holdout_lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}

    checks = promotion_checks(run_cfg, manifest, validation, holdout_lock)
    failed = [c for c in checks if not c[1]]
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if failed and not args.force:
        raise SystemExit(f"not promoted: {len(failed)} gate(s) failed (use --force to override and record it)")

    ist = timezone(timedelta(hours=5, minutes=30))
    notes = "; ".join(detail for _, _, detail in checks)
    if failed:
        notes = f"FORCED despite failed gates: {[c[0] for c in failed]}; " + notes
    deployment = {
        "status": "approved",
        "paper_start_date": args.paper_start or date.today().isoformat(),
        "source_run_id": args.run_id,
        "approved_at": datetime.now(ist).isoformat(timespec="seconds"),
        "notes": notes,
        "data_anchor_date": args.data_anchor,
        "engine": run_cfg.to_dict(),
    }
    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(deployment, indent=2) + "\n")
    dep = load_deployment(out)  # validates the file we just wrote
    print(f"promoted {args.run_id} -> {out} (config {dep.engine.config_hash()}, paper from {dep.paper_start_date})")


def cmd_shift_reference(args) -> None:
    """Backtest the deployed config over the live window for the shift detector."""
    from nse_engine.engine import run_backtest

    if not args.start:
        raise SystemExit("--start is required (first live/paper trading date)")
    if args.run_id:
        run_cfg = json.loads((Path(EngineConfig().runs_dir) / args.run_id / "config.json").read_text())
        cfg = EngineConfig.from_dict(run_cfg)
    else:
        cfg = _build_config(args)
    cfg = cfg.replace(start=args.start, end=args.end or date.today().isoformat())
    data = _load_data(cfg, data_start=getattr(args, "data_start", None))
    result = run_backtest(data, cfg, record=False, tag="shift-reference")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.returns.rename("return").rename_axis("date").to_csv(out)
    print(f"wrote {len(result.returns)} daily returns {cfg.start}..{cfg.end} -> {out}")


def cmd_lag(args) -> None:
    from nse_engine.validation.diagnostics import lag_sensitivity

    cfg = _build_config(args)
    data = _load_data(cfg, data_start=getattr(args, "data_start", None))
    _print_json(lag_sensitivity(data, cfg, lags=tuple(args.lags)))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_config_args(p):
        p.add_argument("--config", help="JSON file with an EngineConfig")
        p.add_argument("--set", action="append", metavar="KEY=VALUE",
                       help="dotted override, e.g. portfolio.target_positions=25")
        p.add_argument("--start")
        p.add_argument("--end")
        p.add_argument("--data-start", help="first data row to load (the anchor), e.g. 2011-01-01")

    p = sub.add_parser("sync", help="download NSE archives (resumable)")
    p.add_argument("--start", default="2012-01-01")
    p.add_argument("--end")
    p.add_argument("--rps", type=float, default=2.0)
    p.set_defaults(func=cmd_sync)

    p = sub.add_parser("build-store", help="parse archives into parquet and build the sector map")
    p.set_defaults(func=cmd_build_store)

    p = sub.add_parser("backtest", help="run and record a backtest")
    add_config_args(p)
    p.add_argument("--tag", default="")
    p.add_argument("--lag-days", type=int, default=0)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("validate", help="DSR, PBO and benchmark gate for a recorded run")
    p.add_argument("--run-id")
    p.add_argument("--splits", type=int, default=16)
    p.add_argument("--margin", type=float, default=0.3)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("walk-forward", help="anchored walk-forward re-fitting")
    add_config_args(p)
    p.add_argument("--grid", required=True, help='JSON, e.g. {"portfolio.target_positions": [15, 20, 30]}')
    p.add_argument("--train-years", type=int, default=4)
    p.add_argument("--test-months", type=int, default=12)
    p.add_argument("--rolling", action="store_true", help="rolling instead of anchored windows")
    p.set_defaults(func=cmd_walk_forward)

    p = sub.add_parser("holdout", help="one-shot evaluation on an untouched window")
    add_config_args(p)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_holdout)

    p = sub.add_parser("promote", help="approve a validated, holdout-tested run for paper/live trading")
    p.add_argument("--run-id", required=True)
    p.add_argument("--paper-start", help="first paper decision date (default today)")
    p.add_argument("--holdout-lock", default="data/nse_engine/holdout.lock")
    p.add_argument("--out", default=None, help="deployment file (default config/nse_engine_deployed.json)")
    p.add_argument("--data-anchor", default="2011-01-01",
                   help="data anchor the validation runs used (pinned for paper/live)")
    p.add_argument("--force", action="store_true", help="promote despite failed gates (recorded in notes)")
    p.set_defaults(func=cmd_promote)

    p = sub.add_parser("shift-reference",
                       help="backtest returns over the live window for the distribution shift detector")
    add_config_args(p)
    p.add_argument("--run-id", help="take the EngineConfig from this recorded run")
    p.add_argument("--out", default="data/shift_reference_returns.csv")
    p.set_defaults(func=cmd_shift_reference)

    p = sub.add_parser("lag", help="execution lag sensitivity")
    add_config_args(p)
    p.add_argument("--lags", type=int, nargs="+", default=[0, 1, 2, 3])
    p.set_defaults(func=cmd_lag)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
