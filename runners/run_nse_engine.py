"""
Command-line entry point for the NSE engine.

    python -m runners.run_nse_engine sync --start 2012-01-01
    python -m runners.run_nse_engine build-store
    python -m runners.run_nse_engine backtest --tag baseline
    python -m runners.run_nse_engine backtest --set portfolio.target_positions=25 --tag tp25
    python -m runners.run_nse_engine validate --run-id <run_id>
    python -m runners.run_nse_engine walk-forward --grid '{"portfolio.target_positions": [15, 20, 30]}'
    python -m runners.run_nse_engine holdout --start 2026-01-01 --end 2026-09-11
    python -m runners.run_nse_engine refresh-registry --dry-run   # after a store rebuild

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
from typing import List, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

import pandas as pd  # noqa: E402

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
        include_symbols=cfg.sleeves.symbols,
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
    summary = build_store(ARCHIVE_DIR, cfg.data.store_dir)
    _print_json(summary)
    sectors = build_sector_map(ARCHIVE_DIR)
    print(f"sector map: {len(sectors)} symbols")
    if summary.get("suspect_missing_sessions"):                 # tracker LN-T7: a failing check
        raise SystemExit("a session seems missing from the store before: "
                         + ", ".join(x["date"] for x in summary["suspect_missing_sessions"]))
    if getattr(args, "skip_registry_check", False):
        print("registry fingerprint check skipped (--skip-registry-check)")
        return
    registry_check_after_rebuild(cfg)


def _registry_anchor() -> str:
    """The data anchor the registry's same-window runs are recorded on: the deployment's
    validated anchor (D4 and RR1 refreshed every run from it).  The data hash covers the
    loaded rows, so a check from any other anchor reports a change that is not there."""
    from nse_engine.deployment import load_deployment

    return load_deployment().data_start().isoformat()


def registry_check_after_rebuild(cfg: EngineConfig) -> dict:
    """Did the rebuild change the data fingerprint the trial registry sits on?

    Loads the validation window, compares its hash with the one the registry
    was last extended on and, when they differ, prints the dry-run refresh
    plan and the command to run.  Called by ``build-store``; safe to call by
    hand.  Never raises: a rebuild must not fail because of the check.
    """
    from nse_engine.validation.trials import TrialRegistry, fingerprint_status, refresh_registry

    registry = TrialRegistry(cfg.runs_dir)
    window = (cfg.start, cfg.end)
    try:
        data = _load_data(cfg, data_start=_registry_anchor())
        status = fingerprint_status(registry, window, data.data_hash)
    except Exception as exc:  # noqa: BLE001 - report, do not fail the rebuild
        print(f"registry fingerprint check could not run: {exc}")
        return {"changed": None, "error": str(exc)}
    if status["registry_hash"] is None:
        print(f"registry fingerprint check: no recorded runs on {cfg.start}..{cfg.end}; nothing to compare")
    elif not status["changed"]:
        print(f"registry fingerprint check: unchanged ({status['current_hash']}); the registry is continuous")
    else:
        report = refresh_registry(registry, data, status["registry_hash"], window, dry_run=True)
        print("=" * 72)
        print(f"STORE FINGERPRINT CHANGED: {status['registry_hash']} -> {status['current_hash']}")
        print(f"  {status['n_configurations_on_registry_hash']} configurations on {cfg.start}..{cfg.end} sit on the "
              f"old hash, {status['n_configurations_on_current_hash']} on the new one; {report['n_planned']} to re-run.")
        print("  Any run recorded now would meet no prior configurations (no PBO, DSR at N=1).")
        print("  Before recording anything, run:")
        print("      python -m runners.run_nse_engine refresh-registry --dry-run")
        print("      python -m runners.run_nse_engine refresh-registry")
        print("=" * 72)
        status["dry_run"] = {k: v for k, v in report.items() if k != "rows"}
    return status


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
    from nse_engine.validation.trials import LEGACY_COST_MODEL
    cost_model = int(manifest.get("cost_model") or LEGACY_COST_MODEL)
    matrix = registry.returns_matrix(data_hash=manifest.get("data_hash"), window=window, cost_model=cost_model)
    returns = matrix[run_id] if run_id in matrix.columns else None
    if returns is None:
        raise SystemExit(f"run {run_id} not found in the returns matrix")

    n_configs = int(matrix.shape[1])
    report = {"run_id": run_id, "window": window, "n_configurations": n_configs, "cost_model": cost_model,
              "metrics": manifest.get("metrics")}
    trials = matrix if n_configs > 1 else None
    # N = every recorded configuration (parameter variants are too correlated
    # for clustering to count them); the clustered figure is informational.
    report["dsr"] = deflated_sharpe(returns, trials_matrix=trials, rf_annual=run_cfg.risk_free_annual)
    report["dsr_clustered"] = deflated_sharpe(returns, trials_matrix=trials, rf_annual=run_cfg.risk_free_annual,
                                              trial_count="clustered")
    if n_configs >= 2:
        pbo = {k: v for k, v in cscv_pbo(matrix, n_splits=args.splits,
                                         rf_annual=run_cfg.risk_free_annual).items() if k != "logits"}
        pbo["pbo_raw_basis"] = float(cscv_pbo(matrix, n_splits=args.splits)["pbo"])   # LN-T16: one cycle beside
        pbo["verdict"] = ("likely real" if pbo["pbo"] < 0.30
                          else "caution" if pbo["pbo"] <= 0.50 else "reject")
        report["pbo"] = pbo
    else:
        report["pbo"] = "needs at least 2 recorded configurations"

    if not Path(run_cfg.data.store_dir).exists():      # a run imported from Kaggle records /kaggle/... paths
        local_store = EngineConfig().data.store_dir
        logger.info("store %s not found here; using %s (store_dir is not part of the config hash)",
                    run_cfg.data.store_dir, local_store)
        run_cfg = run_cfg.replace(**{"data.store_dir": local_store})
    data = _load_data(run_cfg)
    benchmarks = run_benchmarks(data, run_cfg)
    report["benchmark_gate"] = benchmark_gate(returns, benchmarks, margin=args.margin,
                                              rf_annual=run_cfg.risk_free_annual)
    # D2: the haircut, from two unrecorded re-simulations on this machine (as recorded, and stressed)
    from nse_engine.engine import run_backtest
    from nse_engine.validation.diagnostics import haircut_report, stress_config, STRESS_LAG_DAYS
    base = run_backtest(data, run_cfg, record=False)
    stressed = run_backtest(data, stress_config(run_cfg), record=False, lag_days=STRESS_LAG_DAYS)
    report["haircut"] = haircut_report(manifest["metrics"], base.metrics, stressed.metrics,
                                       report["pbo"] if isinstance(report["pbo"], dict) else None)
    _print_json(report)
    out = run_dir / "validation.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"saved {out}")


def cmd_refresh_registry(args) -> None:
    """Re-run every same-window configuration on the current store after its
    data fingerprint changed, check the returns reproduce, and compare PBO and
    the deflated Sharpe before and after."""
    from nse_engine.validation.dsr import deflated_sharpe
    from nse_engine.validation.pbo import cscv_pbo
    from nse_engine.validation.trials import (TrialRegistry, refresh_plan, refresh_registry, registry_hash,
                                              to_jsonable)

    cfg = _build_config(args)
    registry = TrialRegistry(cfg.runs_dir)
    window = (cfg.start, cfg.end)
    from_hash = args.from_hash
    if not from_hash:
        from_hash = registry_hash(registry, window)
        if not from_hash:
            raise SystemExit(f"no recorded runs on {cfg.start}..{cfg.end}")
        print(f"from-hash not given: using {from_hash}, the hash of the latest recorded run on this window")
    data = _load_data(cfg, data_start=getattr(args, "data_start", None) or _registry_anchor())
    print(f"store fingerprint now {data.data_hash}; recorded runs carry {from_hash}")
    if data.data_hash == from_hash:
        print("the fingerprint has not changed; the registry is continuous, nothing to refresh")
        return
    print(f"configurations to refresh: {len(refresh_plan(registry, from_hash, window, skip_hash=data.data_hash))}"
          f" of {len(refresh_plan(registry, from_hash, window))}")
    report = refresh_registry(registry, data, from_hash, window, dry_run=args.dry_run,
                              limit=args.limit, tolerance=args.tolerance, log=print)
    if args.dry_run:
        _print_json({k: v for k, v in report.items() if k != "rows"})
        for row in report["rows"]:
            print(f"  {row['config_hash'][:8]}  lag {row['lag_days']}  {row['tag']}  ({row['run_id']})")
        return

    # PBO and DSR on both fingerprints: the refresh must not move them.
    checks = {}
    for label, h in (("before", from_hash), ("after", data.data_hash)):
        mat = registry.returns_matrix(data_hash=h, window=window)
        out = {"n_configurations": int(mat.shape[1])}
        if mat.shape[1] >= 2:
            out["pbo"] = float(cscv_pbo(mat, n_splits=args.splits, rf_annual=cfg.risk_free_annual)["pbo"])
            out["dsr"] = {col.split("_")[-1]: float(deflated_sharpe(mat[col], trials_matrix=mat,
                                                                     rf_annual=cfg.risk_free_annual)["dsr"])
                          for col in mat.columns}
        checks[label] = out
    before, after = checks["before"], checks["after"]
    summary = {"pbo_before": before.get("pbo"), "pbo_after": after.get("pbo")}
    common = set(before.get("dsr", {})) & set(after.get("dsr", {}))
    if common:
        summary["max_abs_dsr_change"] = max(abs(after["dsr"][k] - before["dsr"][k]) for k in common)
        summary["dsr_configs_compared"] = len(common)
    report["checks"] = checks
    report["summary"] = summary
    out = Path(cfg.runs_dir).parent / f"registry_refresh_{from_hash}_{data.data_hash}.json"
    out.write_text(json.dumps(to_jsonable(report), indent=2))
    _print_json({k: v for k, v in report.items() if k not in ("rows", "checks")})
    print(f"saved {out}")
    if not report.get("all_identical"):
        raise SystemExit("some configurations did not reproduce; see the report")


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


def cmd_promote(args) -> None:
    """Forward gate (V3, decision U19): replace the deployed config with the
    paper candidate once it has traded >= 60 sessions beside it, its G4 paper
    gate is PASS and its walk-forward OOS Sharpe is within 0.05 of the
    deployed config's.  PBO / deflated Sharpe / benchmark / holdout are
    printed with their counts, not gating (``nse_engine.forward_gate``)."""
    from datetime import datetime, timedelta, timezone

    from nse_engine import forward_gate as fg
    from nse_engine.deployment import load_deployment, resolve_path
    from nse_engine.engine import run_backtest

    cand = load_deployment(args.candidate)
    if cand.status != "candidate":
        raise SystemExit(f"{args.candidate} has status {cand.status!r}: only a paper candidate can be promoted")
    if args.run_id and args.run_id != cand.source_run_id:
        raise SystemExit(f"--run-id {args.run_id} is not the candidate's source run ({cand.source_run_id})")
    base = load_deployment(args.out)
    if cand.engine.config_hash() == base.engine.config_hash():
        raise SystemExit("the candidate's engine config is already the deployed one")
    if not (os.getenv("CENTURION_DATABASE_URL") or os.getenv("DATABASE_URL")):
        raise SystemExit("CENTURION_DATABASE_URL is not set: both paper books live in Neon")

    from database.connection import get_db_manager
    from database.paper_cloud import PaperCloudSync
    mgr = get_db_manager()
    cand_book = PaperCloudSync(mgr, schema=args.schema)       # reads only, never creates
    base_book = PaperCloudSync(mgr, schema=None)
    cand_days = fg.session_dates(cand_book.read_sessions())
    base_days = fg.session_dates(base_book.read_sessions())

    window = (args.oos_start, args.oos_end)
    sharpes = {}
    for name, dep in (("candidate", cand), ("deployed", base)):
        cfg = dep.engine.replace(start=fg.VALIDATION_WINDOW[0], end=fg.VALIDATION_WINDOW[1])
        data = _load_data(cfg, data_start=dep.data_start().isoformat())
        res = run_backtest(data, cfg, record=False, tag="forward-gate-oos")
        sharpes[name] = fg.oos_sharpe(res.returns, window, rf_annual=cfg.risk_free_annual)

    checks = [
        fg.sessions_check(cand_days, base_days),
        fg.gate_check(fg.stored_gate(cand_book.read_state()), cand_days[-1] if len(cand_days) else None),
        fg.wf_check(sharpes["candidate"], sharpes["deployed"], window),
    ]
    run_dir = Path(EngineConfig().runs_dir) / str(cand.source_run_id or "")
    validation = json.loads((run_dir / "validation.json").read_text()) if (run_dir / "validation.json").exists() else None
    info = fg.validation_report(validation)
    lock_path = Path(args.holdout_lock)
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    evals = [e for e in lock.get("evaluations", [])
             if e.get("config_hash") == cand.engine.config_hash() and e.get("status") == "completed"]
    info.append(f"holdout: {len(evals)} completed evaluation(s) of this config"
                + (f", last excess Sharpe {evals[-1].get('metrics', {}).get('excess_sharpe')}" if evals else ""))

    def equity(book):
        s = book.read_snapshots()
        if s is None or s.empty:
            return pd.Series(dtype="float64")
        return pd.Series(s["equity"].astype(float).to_numpy(), index=pd.to_datetime(s["date"].astype(str)))
    info.append(fg.paper_comparison(equity(cand_book), equity(base_book)))

    print(f"Forward gate: candidate {cand.engine.config_hash()[:8]} ({args.candidate}, schema {args.schema}) "
          f"vs deployed {base.engine.config_hash()[:8]}")
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print("  Reported, not gating:")
    for line in info:
        print(f"    - {line}")
    passed = fg.decide(checks)
    if args.check:
        print("check only: nothing written" + ("" if passed else " (the gate would refuse)"))
        return
    if not passed and not args.force:
        raise SystemExit("not promoted: the forward gate failed (use --force to override and record it)")

    ist = timezone(timedelta(hours=5, minutes=30))
    notes = ("Promoted by the forward gate (V3, U19) from " + str(args.candidate) + ": "
             + "; ".join(f"{n} {'PASS' if ok else 'FAIL'}: {d}" for n, ok, d in checks)
             + ". Reported: " + "; ".join(info))
    if not passed:
        notes = "FORCED despite the forward gate failing. " + notes
    raw_cand = json.loads(resolve_path(args.candidate).read_text())
    deployment = {
        "status": "approved",
        "paper_start_date": args.paper_start or date.today().isoformat(),
        "source_run_id": cand.source_run_id,
        "approved_at": datetime.now(ist).isoformat(timespec="seconds"),
        "notes": notes,
        "data_anchor_date": cand.data_anchor_date.isoformat() if cand.data_anchor_date else None,
        "engine": cand.engine.to_dict(),
    }
    overlay = raw_cand.get("risk_overlay")
    if not overlay:
        try:
            overlay = json.loads(resolve_path(args.out).read_text()).get("risk_overlay")
        except (OSError, json.JSONDecodeError):
            overlay = None
    if overlay:
        deployment["risk_overlay"] = overlay
    out = resolve_path(args.out)
    out.write_text(json.dumps({k: v for k, v in deployment.items() if v is not None}, indent=2) + "\n")
    dep = load_deployment(out)  # validates the file we just wrote
    print(f"promoted {cand.engine.config_hash()[:8]} -> {out} (paper from {dep.paper_start_date}). "
          f"The candidate book still trades it in schema {args.schema!r}: retire or replace "
          f"{args.candidate} before the next session.")


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
    from nse_engine.paper_gate import reference_trades_path
    trades_out = reference_trades_path(out)   # the paper gate's cost check (G4)
    result.trades.to_csv(trades_out, index=False)
    print(f"wrote {len(result.returns)} daily returns {cfg.start}..{cfg.end} -> {out} "
          f"(+ {len(result.trades)} trades -> {trades_out})")


def cmd_paper_gate(args) -> None:
    """Paper pass/fail gate (G4) for one book: paper record from Neon vs a
    same-period backtest (run here from the local store, or ``--reference``)."""
    from nse_engine import paper_gate
    from nse_engine.deployment import load_deployment

    dep = load_deployment(args.deployment)
    if args.reference:
        ref, trades = paper_gate.read_reference(args.reference)
    else:
        from nse_engine.engine import run_backtest
        cfg = dep.reference_config().replace(end=date.today().isoformat())
        data = _load_data(cfg, data_start=dep.data_start().isoformat())
        res = run_backtest(data, cfg, record=False, tag="paper-gate-reference")
        ref, trades = res.returns, res.trades
    if not (os.getenv("CENTURION_DATABASE_URL") or os.getenv("DATABASE_URL")):
        raise SystemExit("CENTURION_DATABASE_URL is not set: the paper record lives in Neon")
    from database.connection import get_db_manager
    from database.paper_cloud import PaperCloudSync
    cloud = PaperCloudSync(get_db_manager(), schema=args.schema)   # reads only, never creates
    snaps = cloud.read_snapshots()
    if snaps is None or snaps.empty:
        raise SystemExit(f"no paper snapshots in schema {args.schema or 'public'}")
    equity = pd.Series(snaps["equity"].astype(float).to_numpy(), index=pd.to_datetime(snaps["date"].astype(str)))
    report = paper_gate.evaluate(equity, ref, fills=cloud.read_fills(), reference_trades=trades,
                                 sessions=cloud.read_sessions())
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        label = f"{dep.status} {dep.engine.config_hash()[:8]}" + (f", schema {args.schema}" if args.schema else "")
        print(paper_gate.format_report(report, title=f"Paper gate (G4) - {label}"))


CANARY_BOOKS = {"deployed": "config/nse_engine_deployed.json", "candidate": "config/nse_engine_candidate.json",
                "e4": "config/nse_engine_e4.json"}
CANARY_EXPECTED = "config/nse_engine_canary.json"
CANARY_STATE_KEY = "canary_actions"
CANARY_WINDOW = ("2013-01-01", "2025-12-31")


def canary_stats(data, cfg) -> dict:
    """What a book's 2013-25 backtest produces, to the last trade (tracker LN-T13)."""
    import hashlib

    from nse_engine.costs import COST_MODEL_VERSION
    from nse_engine.engine import run_backtest

    res = run_backtest(data, cfg, record=False)
    t = res.trades.copy()
    t["date"] = pd.to_datetime(t["date"]).dt.strftime("%Y-%m-%d")
    t = t.sort_values(["date", "symbol", "side", "quantity"])
    blob = "\n".join(f"{r.date},{r.symbol},{r.side},{int(r.quantity)},{float(r.price):.4f}" for r in t.itertuples())
    m = res.metrics
    return {"config_hash": cfg.config_hash(), "data_hash": data.data_hash, "cost_model": COST_MODEL_VERSION,
            "n_trades": int(len(t)), "trades_sha256": hashlib.sha256(blob.encode()).hexdigest(),
            "sharpe": float(m["sharpe"]), "cagr": float(m["cagr"]), "max_drawdown": float(m["max_drawdown"]),
            "final_equity": float(res.equity.iloc[-1])}


def canary_compare(now: dict, expected: dict) -> Tuple[str, List[str]]:
    """("ok" | "data_differs" | "drift", what moved) for every book in ``expected``."""
    moved, data_moved = [], False
    for book, exp in expected.items():
        got = now.get(book)
        if got is None:
            moved.append(f"{book}: not run")
            continue
        if got["data_hash"] != exp["data_hash"]:
            data_moved = True
            moved.append(f"{book}: data {exp['data_hash']} -> {got['data_hash']}")
            continue
        for k in ("config_hash", "cost_model", "n_trades", "trades_sha256"):
            if got[k] != exp[k]:
                moved.append(f"{book}: {k} {exp[k]} -> {got[k]}")
        for k, tol in (("sharpe", 1e-9), ("cagr", 1e-9), ("max_drawdown", 1e-9), ("final_equity", 0.01)):
            if abs(float(got[k]) - float(exp[k])) > tol:
                moved.append(f"{book}: {k} {exp[k]} -> {got[k]}")
    return ("data_differs" if data_moved else "drift" if moved else "ok"), moved


def cmd_canary(args) -> None:
    """Nightly canary: the three books' 2013-25 backtests against their expected statistics (tracker LN-T13).

    Nothing else catches a code, dependency or runner change that moves a
    book while its hashes stay the same (cost model 4 did).  Tiers: the
    registry's figures in ``CANARY_EXPECTED`` (information), and the Actions
    runner's own, recorded in the paper book's state on its first run (the
    alarm: Actions and the Mac/Kaggle runtimes have never been compared).
    A data change is reported, never refreshed here; drift sets the step
    output ``drift=true`` for the workflow to act on.  Exits 0: alerts go by
    email and step output, never by failing the nightly job.
    """
    import platform

    import numpy as np

    from nse_engine.deployment import load_deployment

    books = {b: load_deployment(path).engine.replace(start=CANARY_WINDOW[0], end=CANARY_WINDOW[1])
             for b, path in CANARY_BOOKS.items()}
    data = _load_data(next(iter(books.values())), data_start="2012-01-02")
    now = {b: canary_stats(data, cfg) for b, cfg in books.items()}
    runtime = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
               "platform": platform.platform()}
    if args.record_registry:
        Path(CANARY_EXPECTED).write_text(json.dumps({"registry": now, "registry_runtime": runtime,
                                                      "recorded": date.today().isoformat()}, indent=1, sort_keys=True))
        print(f"recorded the registry tier in {CANARY_EXPECTED}")
        return
    try:
        registry = json.loads(Path(CANARY_EXPECTED).read_text())["registry"]
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"{CANARY_EXPECTED} unreadable ({exc}): record it with canary --record-registry")
    reg_status, reg_moved = canary_compare(now, registry)
    report = {"runtime": runtime, "registry": {"status": reg_status, "moved": reg_moved}}
    status, moved = reg_status, reg_moved
    if args.tier == "actions":
        from database.paper_cloud import get_paper_cloud

        cloud = get_paper_cloud()
        stored = json.loads((cloud.read_state() or {}).get(CANARY_STATE_KEY) or "null") if cloud else None
        if cloud and stored is None:
            cloud.sync_state({CANARY_STATE_KEY: json.dumps({"books": now, "runtime": runtime,
                                                             "recorded": date.today().isoformat()})})
            status, moved = "recorded", []
        elif stored is not None:
            status, moved = canary_compare(now, stored["books"])
        report["actions"] = {"status": status, "moved": moved}
    report["status"] = status
    _print_json(report)
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as fh:
            fh.write(f"canary={status}\ndrift={'true' if status == 'drift' else 'false'}\n")
    if status in ("drift", "data_differs") and not args.no_email:
        try:
            from services.notifications.manager import NotificationManager

            what = ("the store's data changed under the books (refresh the registry off Actions)"
                    if status == "data_differs" else "a book's backtest changed with its code and data hashes unchanged")
            NotificationManager()._send_html_email(
                f"Centurion canary: {status.replace('_', ' ')}",
                f"<p>The nightly canary found that {what}.</p><ul>"
                + "".join(f"<li>{m}</li>" for m in moved) + f"</ul><p>Runtime: {runtime}</p>")
        except Exception as exc:                          # noqa: BLE001 - the step output still carries it
            logger.warning("canary email not sent: %s", exc)


def cmd_scorecard(args) -> None:
    """SC1: the strategy scorecard of a book's latest recorded run (docs/scorecards/)."""
    from nse_engine.scorecard import main as scorecard_main

    argv = ["--book", args.book, "--md-dir", args.md_dir, "--json-dir", args.json_dir]
    if args.run:
        argv += ["--run", args.run]
    if args.paper_schema:
        argv += ["--paper-schema", args.paper_schema]
    scorecard_main(argv)


def cmd_anchor_check(args) -> None:
    """Run the same window from two load starts and report whether the results agree.

    An anchor-independent configuration must give identical daily returns
    once both loads contain ``required_warmup_days`` rows before ``start``.
    """
    from nse_engine.engine import run_backtest

    cfg = _build_config(args)
    need = cfg.required_warmup_days()
    results = {}
    for label, ds in (("a", args.data_start_a), ("b", args.data_start_b)):
        rows_before = None
        data = _load_data(cfg, data_start=ds)
        rows_before = int((data.dates < pd.Timestamp(cfg.start)).sum())
        res = run_backtest(data, cfg, record=False, tag="anchor-check")
        r = pd.Series(res.returns, dtype="float64")
        r.index = pd.DatetimeIndex(r.index)
        results[label] = {"returns": r, "rows_before_start": rows_before, "n_trades": int(res.metrics.get("n_trades", 0)),
                          "sharpe": res.metrics.get("sharpe"), "total_return": res.metrics.get("total_return")}
        logger.info("load %s from %s: %d rows before start (need %d), sharpe=%.3f total=%.4f, %d trades",
                    label, ds, rows_before, need, results[label]["sharpe"], results[label]["total_return"],
                    results[label]["n_trades"])
    ra, rb = results["a"]["returns"].align(results["b"]["returns"], join="inner")
    diff = (ra - rb).abs()
    first = diff[diff > args.tolerance]
    report = {
        "config_hash": cfg.config_hash(),
        "anchor_independent_config": cfg.anchor_independent(),
        "required_warmup_days": need,
        "rows_before_start": {k: v["rows_before_start"] for k, v in results.items()},
        "n_days_compared": int(len(diff)),
        "max_abs_return_diff": float(diff.max()) if len(diff) else 0.0,
        "n_days_differing": int((diff > args.tolerance).sum()),
        "first_differing_date": str(first.index[0].date()) if len(first) else None,
        "sharpe": {k: v["sharpe"] for k, v in results.items()},
        "total_return": {k: v["total_return"] for k, v in results.items()},
        "n_trades": {k: v["n_trades"] for k, v in results.items()},
        "identical": bool(len(diff) and (diff <= args.tolerance).all()),
    }
    _print_json(report)
    if not report["identical"]:
        sys.exit(1)


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

    p = sub.add_parser("build-store", help="parse archives into parquet and build the sector map; "
                                           "then check the trial registry's data fingerprint")
    p.add_argument("--skip-registry-check", action="store_true",
                   help="do not compare the rebuilt store's fingerprint with the trial registry")
    p.set_defaults(func=cmd_build_store)

    p = sub.add_parser("backtest", help="run and record a backtest")
    add_config_args(p)
    p.add_argument("--tag", default="")
    p.add_argument("--lag-days", type=int, default=0)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("scorecard", help="SC1 strategy scorecard of a book's latest recorded run: return/risk, "
                                         "factors, alpha decay, trading, capacity, robustness, correlation, paper G4")
    p.add_argument("--book", default="all", choices=["all", "deployed", "candidate", "e4"])
    p.add_argument("--run", help="a recorded run id (default: the book's latest on the current cost model)")
    p.add_argument("--paper-schema", help="the paper book's Neon schema (needs CENTURION_DATABASE_URL)")
    p.add_argument("--md-dir", default="docs/scorecards")
    p.add_argument("--json-dir", default="data/nse_engine/scorecard")
    p.set_defaults(func=cmd_scorecard)

    p = sub.add_parser("canary", help="the three books' 2013-25 backtests against their expected statistics "
                                      "(LN-T13)")
    p.add_argument("--tier", choices=("registry", "actions"), default="registry",
                   help="actions: also compare with (or record) the Actions runner's own figures")
    p.add_argument("--record-registry", action="store_true", help=f"write the registry tier to {CANARY_EXPECTED}")
    p.add_argument("--no-email", action="store_true")
    p.set_defaults(func=cmd_canary)

    p = sub.add_parser("validate", help="DSR, PBO and benchmark gate for a recorded run")
    p.add_argument("--run-id")
    p.add_argument("--splits", type=int, default=16)
    p.add_argument("--margin", type=float, default=0.3)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("refresh-registry",
                       help="after a store rebuild changed the data fingerprint: re-run every "
                            "same-window configuration, verify the returns reproduce, compare PBO/DSR")
    add_config_args(p)
    p.add_argument("--from-hash", help="data hash the runs were recorded on (default: the most common one)")
    p.add_argument("--dry-run", action="store_true", help="list what would run; write nothing")
    p.add_argument("--limit", type=int, help="refresh at most this many configurations (resumable)")
    p.add_argument("--tolerance", type=float, default=1e-9, help="max daily return difference to call identical")
    p.add_argument("--splits", type=int, default=16, help="CSCV blocks for the PBO comparison")
    p.set_defaults(func=cmd_refresh_registry)

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

    p = sub.add_parser("promote", help="forward gate (U19): replace the deployed config with the paper "
                                       "candidate after >= 60 sessions beside it, G4 PASS and WF OOS Sharpe "
                                       "within 0.05")
    p.add_argument("--candidate", default="config/nse_engine_candidate.json", help="the paper candidate's file")
    p.add_argument("--schema", default="candidate", help="Postgres schema of the candidate book")
    p.add_argument("--run-id", default=None, help="optional: must equal the candidate's source_run_id")
    p.add_argument("--paper-start", help="first session the deployed book trades the new config (default today)")
    p.add_argument("--oos-start", default="2017-01-01", help="walk-forward OOS years start (K5: 2017)")
    p.add_argument("--oos-end", default="2025-12-31")
    p.add_argument("--holdout-lock", default="data/nse_engine/holdout.lock")
    p.add_argument("--out", default=None, help="deployment file (default config/nse_engine_deployed.json)")
    p.add_argument("--check", action="store_true", help="print the gate, write nothing")
    p.add_argument("--force", action="store_true", help="promote despite a failed gate (recorded in notes)")
    p.set_defaults(func=cmd_promote)

    p = sub.add_parser("shift-reference",
                       help="backtest returns over the live window for the distribution shift detector")
    add_config_args(p)
    p.add_argument("--run-id", help="take the EngineConfig from this recorded run")
    p.add_argument("--out", default="data/shift_reference_returns.csv")
    p.set_defaults(func=cmd_shift_reference)

    p = sub.add_parser("paper-gate", help="paper pass/fail gate (G4): a paper book vs its same-period backtest")
    p.add_argument("--deployment", default=None, help="deployment file (default: the deployed book)")
    p.add_argument("--schema", default=None, help="Postgres schema of the book, e.g. candidate (default: public)")
    p.add_argument("--reference", default=None,
                   help="same-period reference returns CSV (with its _trades.csv) instead of a backtest here")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_paper_gate)

    p = sub.add_parser("anchor-check", help="same window from two load starts: identical results?")
    add_config_args(p)
    p.add_argument("--data-start-a", required=True, help="first load start, e.g. 2011-01-01")
    p.add_argument("--data-start-b", required=True, help="second load start, e.g. 2021-01-01")
    p.add_argument("--tolerance", type=float, default=1e-12)
    p.set_defaults(func=cmd_anchor_check)

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
