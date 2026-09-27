"""Rehearse the live order path without sending anything (tracker L3).

    python -m tools.live_dry_run                      # the paper book stands in for the live one
    python -m tools.live_dry_run --source kite        # the real Kite holdings and cash (needs a session)
    python -m tools.live_dry_run --as-of 2026-09-26 --json

Loads the deployment, plans exactly as a session would (drawdown rule
included) and prints every broker order ``EngineExecutor._execute_live``
would send - symbol, side, quantity, limit, variety (after-market after the
close), idempotency tag - plus the GTT stops it would reconcile, the gates
that currently stand between the engine and a real order, and the reasons
anything was skipped.  Nothing is placed.  Run it daily for at least a week
before the first live rupee; when its orders match the paper book's queued
orders, the live path is doing what the paper path does.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger("live_dry_run")


def gates() -> Dict[str, Any]:
    """Every condition a real order must pass right now."""
    from kite_connect.trading.nse_engine_executor import live_orders_allowed
    from kite_connect.trading.order_service import _is_nse_market_open, is_kill_switch_active, order_variety_now

    allowed, reason = live_orders_allowed()
    return {"env_allows_live": allowed, "env_reason": reason,
            "kill_switch": is_kill_switch_active(),
            "market_open": _is_nse_market_open(), "variety_now": order_variety_now(),
            "CENTURION_PAPER_TRADE": os.environ.get("CENTURION_PAPER_TRADE", "<unset: true>"),
            "CENTURION_NSE_ENGINE_LIVE": os.environ.get("CENTURION_NSE_ENGINE_LIVE", "<unset: false>")}


def build(source: str, as_of: Optional[str], deployment_path: Optional[str] = None):
    """Executor + plan + dry-run results for ``source`` ('paper' or 'kite')."""
    from kite_connect.trading.nse_engine_executor import EngineExecutor, kite_book
    from nse_engine.deployment import load_deployment

    dep = load_deployment(deployment_path) if deployment_path else load_deployment()
    if source == "kite":
        from kite_connect.zerodha_live import get_kite_session
        kite = get_kite_session()
        if kite is None:
            raise SystemExit("no Kite session: log in first, or use --source paper")
        ex = EngineExecutor(kite=kite, paper=True, deployment=dep, dry_run=True,
                            holdings_fn=lambda: kite_book(kite))
    else:
        ex = EngineExecutor(kite=None, paper=True, deployment=dep, dry_run=True)
    plan = ex.plan(as_of=as_of)
    return dep, ex, plan, ex.dry_run_live(plan)


def report(dep, plan, results: List[dict], gate: Dict[str, Any], source: str) -> Dict[str, Any]:
    dep_ok, dep_reason = dep.live_allowed()
    orders = [r for r in results if r.get("type") != "gtt_reconcile"]
    stops = next((r["stops"] for r in results if r.get("type") == "gtt_reconcile"), [])
    return {
        "as_of": str(plan.as_of.date()), "book_source": source,
        "deployment": {"status": dep.status, "config_hash": dep.engine.config_hash(),
                       "live_allowed": dep_ok, "reason": dep_reason,
                       "drawdown_rule": dep.summary().get("drawdown_rule")},
        "gates": gate,
        "book": {"equity": round(plan.equity, 2), "cash": round(plan.cash, 2)},
        "drawdown": {"state": plan.drawdown_state, "pct_below_peak": plan.drawdown_pct,
                     "scale": plan.drawdown_scale, "changed": plan.drawdown_changed},
        "orders": orders, "stops": stops, "skipped": list(plan.skipped), "notes": list(plan.notes),
        "would_send": sum(1 for o in orders if o.get("success")),
    }


def format_report(rep: Dict[str, Any]) -> str:
    g, d = rep["gates"], rep["deployment"]
    lines = [f"LIVE DRY RUN for session {rep['as_of']} (book: {rep['book_source']}) - nothing is sent",
             f"  deployment {d['status']} {d['config_hash']} - live {'allowed' if d['live_allowed'] else 'refused'}: {d['reason']}",
             f"  env: {'ALLOWS live orders' if g['env_allows_live'] else 'blocks live orders'} ({g['env_reason']})",
             f"  kill switch {'ON' if g['kill_switch'] else 'off'} | market {'open' if g['market_open'] else 'closed'} "
             f"-> orders would go as '{g['variety_now']}'",
             f"  book: equity {rep['book']['equity']:,.0f}, cash {rep['book']['cash']:,.0f} | drawdown rule "
             f"{rep['drawdown']['state']} ({rep['drawdown']['pct_below_peak']:.1f}% below peak)"
             + (" - rule not configured" if d.get("drawdown_rule") is None else ""),
             ""]
    orders = rep["orders"]
    if any(s.get("reason") == "stale_data" for s in rep["skipped"]):
        lines.append("  STALE STORE: the last bar is older than the last completed session, so no orders were")
        lines.append("  planned (a real session would refuse too).  Sync and rebuild the store first:")
        lines.append("      python -m nse_engine.data.archive --start <last bar> --root data/nse_engine/archive")
        lines.append("      python -m runners.run_nse_engine build-store")
        lines.append("  or pass --as-of with an older session to replay it.")
    elif orders:
        lines.append(f"  {len(orders)} order(s), {rep['would_send']} would be sent:")
        lines.append(f"  {'side':<5}{'symbol':<14}{'qty':>7}{'limit':>10}  {'variety':<8}{'tag':<20} {'reason':<16}status")
        for o in orders:
            lines.append(f"  {o['side']:<5}{o['symbol']:<14}{o['quantity']:>7}{o['limit_price']:>10.2f}  "
                         f"{o['variety']:<8}{o['tag']:<20} {o['reason']:<16}{o['status']}"
                         + (f"  ({o['error']})" if o.get("error") else ""))
    else:
        lines.append("  no orders: the book already matches the target")
    if rep["stops"]:
        lines.append(f"  {len(rep['stops'])} GTT stop(s) to reconcile: "
                     + ", ".join(f"{s['symbol']} {s['quantity']}@{s['trigger']:.2f}" for s in rep["stops"]))
    if rep["skipped"]:
        lines.append(f"  skipped ({len(rep['skipped'])}): "
                     + "; ".join(f"{s.get('symbol')}: {s.get('reason')}" for s in rep["skipped"][:12]))
    for n in rep["notes"]:
        lines.append(f"  note: {n}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--as-of", help="session date (default: the latest store session)")
    ap.add_argument("--source", choices=("paper", "kite"), default="paper",
                    help="whose holdings and cash to plan from (default paper)")
    ap.add_argument("--deployment", help="deployment file (default config/nse_engine_deployed.json)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    os.environ.setdefault("CENTURION_PAPER_TRADE", "true")     # belt and braces: never a real order from here
    dep, ex, plan, results = build(args.source, args.as_of, args.deployment)
    rep = report(dep, plan, results, gates(), args.source)
    print(json.dumps(rep, indent=2, default=str) if args.json else format_report(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
