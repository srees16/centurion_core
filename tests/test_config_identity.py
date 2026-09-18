"""Configuration identity: the deployed hash must survive new, unused options.

PBO and the deflated Sharpe count configurations by hash, and the deployment
file pins one. A new field that changed the hash would silently detach the live
book from its recorded trials.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nse_engine.config import HASH_NEUTRAL_DEFAULTS, EngineConfig
from nse_engine.deployment import load_deployment

ROOT = Path(__file__).resolve().parent.parent
DEPLOYED_HASH = "679cbd0ceaf7c895"


def test_the_deployed_configuration_keeps_its_hash():
    dep = load_deployment(ROOT / "config/nse_engine_deployed.json")
    assert dep.engine.config_hash() == DEPLOYED_HASH


@pytest.mark.parametrize("path,legacy", list(HASH_NEUTRAL_DEFAULTS.items()))
def test_hash_neutral_fields_do_not_change_the_hash_at_their_legacy_value(path, legacy):
    section, field = path
    base = EngineConfig()
    same = base.replace(**{f"{section}.{field}": legacy})
    assert same.config_hash() == base.config_hash()


@pytest.mark.parametrize("path,value", [
    (("universe", "price_filter_unadjusted"), True),
    (("signals", "normalizer_window_days"), 504),
    (("allocator", "target_vol_annual"), 0.12),
])
def test_changing_a_behaviour_flag_changes_the_hash(path, value):
    section, field = path
    base = EngineConfig()
    assert base.replace(**{f"{section}.{field}": value}).config_hash() != base.config_hash()


def test_deployment_pins_the_validated_data_anchor():
    """The anchor decides rebalance days; a deployment without it drifts from validation."""
    raw = json.loads((ROOT / "config/nse_engine_deployed.json").read_text())
    assert raw["data_anchor_date"] == "2012-01-02"
    dep = load_deployment(ROOT / "config/nse_engine_deployed.json")
    assert dep.data_start().isoformat() == "2012-01-02"


def test_config_round_trips_through_json():
    cfg = EngineConfig().replace(**{"universe.price_filter_unadjusted": True,
                                    "portfolio.target_positions": 25})
    again = EngineConfig.from_dict(json.loads(cfg.to_json()))
    assert again.config_hash() == cfg.config_hash()
    assert again.universe.price_filter_unadjusted is True
