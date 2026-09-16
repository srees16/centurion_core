"""Every live entry point must stay importable.

Guards refactors (file moves, package renames, deleted modules): the paper
trading workflow, the scheduler and the UI/API import these by path, so a
broken import here breaks automation that only runs once a day.

Each module is imported in a subprocess so import side effects of one module
cannot mask a failure in another.
"""

import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Modules reachable from automation, not from other imports.
ENTRY_POINTS = (
    # workflow: .github/workflows/nse-paper-trading.yml
    "cloud_paper_runner",
    "nse_engine.data.archive",
    "nse_engine.data.store",
    "nse_engine.deployment",
    "runners.run_nse_engine",
    # engine + validation
    "nse_engine.engine",
    "nse_engine.data.panel",
    "nse_engine.validation.trials",
    "nse_engine.validation.pbo",
    "nse_engine.validation.dsr",
    "nse_engine.validation.walk_forward",
    "nse_engine.validation.holdout",
    "nse_engine.validation.benchmarks",
    "nse_engine.validation.diagnostics",
    # broker / execution
    "kite_connect.trading.nse_engine_executor",
    "kite_connect.trading.paper_trader",
    "kite_connect.trading.order_service",
    "kite_connect.trading.gtt_stops",
    "kite_connect.trading.trade_monitor",
    "kite_connect.trading.auto_executor",
    "kite_connect.nse.nse_universe",
    # orchestration / analysis / legacy still wired to the scheduler
    "scheduler",
    "analyze_paper_results",
    "services.carver_pipeline",
    "services.distribution_shift",
    "database.paper_cloud",
    # interfaces
    "api.main",
    "app",
    # shared
    "config",
    "utils",
)


class EntryPointImportTests(unittest.TestCase):
    def test_entry_points_import(self):
        failures = []
        for module in ENTRY_POINTS:
            with self.subTest(module=module):
                proc = subprocess.run(
                    [sys.executable, "-c", f"import {module}"],
                    cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
                )
                if proc.returncode != 0:
                    tail = (proc.stderr.strip().splitlines() or ["<no stderr>"])[-1]
                    failures.append(f"{module}: {tail}")
                    self.fail(f"cannot import {module}: {tail}")
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
