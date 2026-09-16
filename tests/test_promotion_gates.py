"""Tests for the deployment promotion gates in runners.run_nse_engine (stdlib unittest)."""

import unittest

from nse_engine.config import EngineConfig
from runners.run_nse_engine import promotion_checks


def _inputs(pbo=0.2, dsr=0.97, gate=True, holdout_sharpe=0.5, holdout_dd=-0.20, bt_dd=-0.25, holdout=True):
    cfg = EngineConfig()
    manifest = {"metrics": {"max_drawdown": bt_dd}}
    validation = {"pbo": {"pbo": pbo}, "n_configurations": 36, "dsr": {"dsr": dsr},
                  "benchmark_gate": {"passed": gate}}
    lock = {"evaluations": []}
    if holdout:
        lock["evaluations"].append({
            "config_hash": cfg.replace(start="2026-01-01", end="2026-09-11").config_hash(),
            "status": "completed", "window": {"start": "2026-01-01", "end": "2026-09-11"},
            "metrics": {"excess_sharpe": holdout_sharpe, "max_drawdown": holdout_dd},
        })
    return cfg, manifest, validation, lock


def _failed(checks):
    return sorted(name for name, ok, _ in checks if not ok)


class PromotionGateTests(unittest.TestCase):
    def test_all_gates_pass(self):
        self.assertEqual(_failed(promotion_checks(*_inputs())), [])

    def test_each_gate_can_fail(self):
        self.assertEqual(_failed(promotion_checks(*_inputs(pbo=0.31))), ["pbo"])
        self.assertEqual(_failed(promotion_checks(*_inputs(dsr=0.90))), ["dsr"])
        self.assertEqual(_failed(promotion_checks(*_inputs(gate=False))), ["benchmark_gate"])
        self.assertEqual(_failed(promotion_checks(*_inputs(holdout_sharpe=-0.1))), ["holdout_sharpe"])
        self.assertEqual(_failed(promotion_checks(*_inputs(holdout_dd=-0.40))), ["holdout_maxdd"])

    def test_missing_holdout_fails(self):
        self.assertEqual(_failed(promotion_checks(*_inputs(holdout=False))), ["holdout"])

    def test_holdout_for_other_config_does_not_count(self):
        cfg, manifest, validation, lock = _inputs()
        other = cfg.replace(**{"portfolio.target_positions": 30})
        self.assertIn("holdout", _failed(promotion_checks(other, manifest, validation, lock)))



class DataAnchorTests(unittest.TestCase):
    def _deployment(self, anchor):
        from nse_engine.deployment import parse_deployment
        raw = {"status": "approved", "paper_start_date": "2026-09-16", "source_run_id": "r1",
               "approved_at": "2026-09-15T22:00:00+05:30", "engine": EngineConfig().to_dict()}
        if anchor:
            raw["data_anchor_date"] = anchor
        return parse_deployment(raw)

    def test_pinned_anchor_drives_data_and_bootstrap_start(self):
        dep = self._deployment("2011-01-01")
        self.assertEqual(dep.data_start().isoformat(), "2011-01-01")
        self.assertEqual(dep.bootstrap_start().isoformat(), "2011-01-01")
        unpinned = self._deployment(None)
        self.assertEqual(unpinned.data_start().isoformat(), "2024-01-01")

    def test_anchor_must_precede_paper_start(self):
        from nse_engine.deployment import DeploymentError
        with self.assertRaises(DeploymentError):
            self._deployment("2026-09-20")

    def test_executor_loads_from_pinned_anchor(self):
        from kite_connect.trading.nse_engine_executor import EngineExecutor
        calls = []

        def loader(store_dir, start, end, **kwargs):
            calls.append(start)
            return None

        dep = self._deployment("2011-01-01")
        ex = EngineExecutor(paper=True, config=dep.live_config(), data_loader=loader, deployment=dep)
        ex._load(dep.live_config(), "2026-09-16")
        self.assertEqual(calls, ["2011-01-01"])


if __name__ == "__main__":
    unittest.main()
