#!/usr/bin/env python3
"""Which tests guard which functionality, and where they run (owner's ask, 15 Nov 2026).

The test files are organised by what they protect in production.  ``AREAS`` names each functionality, the code
that carries it and the test files that guard it; a test (``tests/test_process_loop.py``) fails when a test file
belongs to no area or to two, or when a listed path is gone, so a new test file must say what it protects.

Where a test runs, read from the test files themselves:
- **CI, core install**: every push; the paper path's own environment.  Tests that need the web packages skip here;
- **CI, full job**: every push, with the web packages; nothing may skip (``CENTURION_NO_SKIPS``);
- **this Mac, store guards**: the tests marked ``slow``, which need the data stores (``python -m tools.guards``).

    python -m tools.test_map                              # the table
    python -m tools.test_map --coverage coverage.json     # with the share of each area's code a run exercised
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

#: (functionality, the code that carries it, the test files that guard it).  A path ending in "/" is a package.
AREAS: List[Dict[str, Any]] = [
    {"area": "Live order path",
     "what": "every real order: the gate, the kill switch, stops at the broker, what Kite reports back",
     "code": ["kite_connect/trading/order_service.py", "kite_connect/trading/kill_switch.py",
              "kite_connect/trading/gtt_stops.py", "kite_connect/trading/order_status.py",
              "kite_connect/trading/postback_alerts.py", "kite_connect/trading/nse_instruments.py"],
     "tests": ["test_order_gate.py", "test_kill_switch.py", "test_stops_and_postbacks.py"]},
    {"area": "Live session and go-live",
     "what": "the nightly live session, its executor, the Kite login, the capital ladder and the go-live checks",
     "code": ["kite_connect/trading/live_session.py", "kite_connect/trading/nse_engine_executor.py",
              "kite_connect/auth/", "nse_engine/capital_ladder.py", "nse_engine/live_rules.py",
              "nse_engine/live_monitor.py", "nse_engine/mc_bands.py", "nse_engine/deployment.py",
              "nse_engine/funding_gate.py", "tools/game_day.py"],
     "tests": ["test_live_session.py", "test_live_executor.py", "test_kite_login.py", "test_go_live.py",
               "test_mc_rule_d.py"]},
    {"area": "Nightly paper books",
     "what": "the paper run and its books, sleeves and gate, the heartbeat and rescue, the emails, the backups",
     "code": ["cloud_paper_runner.py", "scheduler.py", "kite_connect/trading/paper_trader.py",
              "kite_connect/trading/futures_overlay.py", "kite_connect/trading/bond_sleeve.py",
              "kite_connect/trading/mcx_sleeve.py", "nse_engine/books.py", "nse_engine/paper_gate.py",
              "database/paper_cloud.py", "tools/paper_heartbeat.py", "tools/session_failed.py", "tools/ops_report.py",
              "tools/books_report.py", "tools/export_books.py", "cloud/backup.py", "services/notifications/"],
     "tests": ["test_paper_book.py", "test_paper_heartbeat.py"]},
    {"area": "Engine, costs and data",
     "what": "the backtest engine, the cost models, the universe and signals, the data stores, the configuration",
     "code": ["nse_engine/engine.py", "nse_engine/costs.py", "nse_engine/portfolio.py", "nse_engine/universe.py",
              "nse_engine/signals.py", "nse_engine/regime.py", "nse_engine/drawdown.py", "nse_engine/sleeves.py",
              "nse_engine/allocator.py", "nse_engine/nse_calendar.py", "nse_engine/config.py", "nse_engine/data/",
              "tools/data_catalogue.py"],
     "tests": ["test_costs_and_fills.py", "test_universe_lookahead.py", "test_signals_causal.py",
               "test_survivorship.py", "test_config_identity.py"]},
    {"area": "Research records and metrics",
     "what": "the metrics and gates a verdict rests on, the journal, the trial counts, the process loop's tools",
     "code": ["nse_engine/metrics.py", "nse_engine/journal.py", "nse_engine/scorecard.py", "nse_engine/validation/",
              "tools/lessons.py", "tools/preflight.py", "tools/prereg.py", "tools/guards.py", "tools/study.py",
              "tools/test_map.py", "shared/", "nse_engine/golden_gates.py"],
     "tests": ["test_metrics.py", "test_process_loop.py"]},
    {"area": "Options",
     "what": "option theory, F&O costs, the strategies and structures, the Kite option paths",
     "code": ["kite_connect/options/"],
     "tests": ["test_options_theory.py", "test_options_costs.py", "test_options_strategies.py",
               "test_options_kite.py"]},
    {"area": "Web app and API",
     "what": "the API the web app calls: sessions, roles, the manual order and postback routes",
     "code": ["api/", "auth/"],
     "tests": ["test_api.py"]},
    {"area": "Legacy strategy pages",
     "what": "the single-stock strategy backtest behind the legacy pages",
     "code": ["strategies/", "trading_strategies/", "tools/strategy_benchmark.py", "services/research/"],
     "tests": ["test_strategy_backtest.py"]},
]


def problems(root: Path = _ROOT) -> List[str]:
    """What is wrong with the map: a test file in no area or in two, or a listed path that does not exist."""
    out: List[str] = []
    seen: Dict[str, str] = {}
    for a in AREAS:
        for t in a["tests"]:
            if t in seen:
                out.append(f"tests/{t} is in two areas: {seen[t]} and {a['area']}")
            seen[t] = a["area"]
            if not (Path(root) / "tests" / t).is_file():
                out.append(f"{a['area']}: tests/{t} does not exist")
        for c in a["code"]:
            target = Path(root) / c
            if not (target.is_dir() if c.endswith("/") else target.is_file()):
                out.append(f"{a['area']}: {c} does not exist")
    for path in sorted((Path(root) / "tests").glob("test_*.py")):
        if path.name not in seen:
            out.append(f"tests/{path.name} belongs to no area: add it to tools/test_map.py AREAS")
    return out


def _counts(text: str) -> Dict[str, int]:
    return {"tests": len(re.findall(r"^\s*def test_", text, re.M)),
            "store": len(re.findall(r"^@pytest\.mark\.slow", text, re.M)),
            "web": len(re.findall(r"importorskip\(", text))}


def rows(root: Path = _ROOT, coverage: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """One row per area: test functions, how many need the stores or the web packages, and the coverage of its
    code when a coverage.py JSON report is given."""
    files = (coverage or {}).get("files", {})
    out = []
    for a in AREAS:
        n = {"tests": 0, "store": 0, "web": 0}
        for t in a["tests"]:
            path = Path(root) / "tests" / t
            for k, v in (_counts(path.read_text()) if path.is_file() else {}).items():
                n[k] += v
        covered = statements = 0
        for f, v in files.items():
            if any(f == c or (c.endswith("/") and f.startswith(c)) for c in a["code"]):
                covered += v["summary"]["covered_lines"]
                statements += v["summary"]["num_statements"]
        out.append({"area": a["area"], "what": a["what"], "files": a["tests"], **n,
                    "statements": statements, "coverage": round(100 * covered / statements, 1) if statements else None})
    return out


def render(table: Sequence[Dict[str, Any]]) -> str:
    lines = [f"{'Functionality':30} {'tests':>5} {'need stores':>11} {'need web':>8} {'code covered':>12}  test files"]
    for r in table:
        cov = "-" if r["coverage"] is None else f"{r['coverage']:.1f}%"
        lines.append(f"{r['area']:30} {r['tests']:5} {r['store']:11} {r['web']:8} {cov:>12}  {', '.join(r['files'])}")
    lines.append("tests = test functions (a parametrised one counts once); need stores = run only by python -m "
                 "tools.guards on this Mac; need web = skip in CI's core job and must run in its full job")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--coverage", help="a coverage.py JSON report (coverage json -o coverage.json)")
    args = p.parse_args(argv)
    found = problems()
    print(render(rows(coverage=json.loads(Path(args.coverage).read_text()) if args.coverage else None)))
    print("\n".join(f"! {f}" for f in found))
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
