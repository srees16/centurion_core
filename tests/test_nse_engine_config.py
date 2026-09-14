"""Tests for nse_engine.config (stdlib unittest)."""

import json
import unittest

from nse_engine.config import EngineConfig


class EngineConfigTests(unittest.TestCase):
    def test_json_round_trip_preserves_values_and_hash(self):
        cfg = EngineConfig().replace(**{"portfolio.target_positions": 25, "signals.fast_ewmac": ((4, 16),)})
        restored = EngineConfig.from_dict(json.loads(cfg.to_json()))
        self.assertEqual(restored, cfg)
        self.assertEqual(restored.config_hash(), cfg.config_hash())
        self.assertEqual(restored.signals.weights()["fast_trend"], 1.0 / 3.0)

    def test_hash_ignores_dates_and_paths_but_not_parameters(self):
        base = EngineConfig()
        self.assertEqual(base.config_hash(), base.replace(start="2015-01-01", runs_dir="/tmp/x").config_hash())
        self.assertNotEqual(base.config_hash(), base.replace(**{"costs.max_participation": 0.02}).config_hash())

    def test_replace_does_not_mutate(self):
        base = EngineConfig()
        changed = base.replace(**{"regime.confirm_days": 5})
        self.assertEqual(base.regime.confirm_days, 3)
        self.assertEqual(changed.regime.confirm_days, 5)


if __name__ == "__main__":
    unittest.main()
