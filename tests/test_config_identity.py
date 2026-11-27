"""Configuration identity: the deployed hash must survive new, unused options.

PBO and the deflated Sharpe count configurations by hash, and the deployment
file pins one. A new field that changed the hash would silently detach the live
book from its recorded trials.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nse_engine.config import HASH_NEUTRAL_DEFAULTS, RANGES, EngineConfig, range_problems
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


# ── impossible values refuse the configuration (tracker QP1, after jsmidt/QuantPy's window checks) ──

def _edges(kind, low, high, low_ok, high_ok):
    """(inside, outside) pairs at each finite edge of a bound."""
    step = 1 if kind == "int" else 1e-9
    pairs = []
    if low is not None:
        pairs.append((low, low - step) if low_ok else (low + step, low))
    if high is not None:
        pairs.append((high, high + step) if high_ok else (high - step, high))
    return pairs


@pytest.mark.parametrize("path,bound", sorted(RANGES.items()))
def test_every_bound_takes_the_value_at_its_edge_and_refuses_the_one_past_it(path, bound):
    key = ".".join(path)
    for inside, outside in _edges(*bound):
        EngineConfig().replace(**{key: inside})
        with pytest.raises(ValueError, match=f"QP1.*{key}="):
            EngineConfig().replace(**{key: outside})


@pytest.mark.parametrize("path", sorted(p for p, b in RANGES.items() if b[0] == "num"))
def test_a_number_that_is_not_finite_is_refused(path):
    for bad in (float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="is not a finite number"):
            EngineConfig().replace(**{".".join(path): bad})


def test_the_values_that_mean_off_and_whole_number_forms_still_load():
    import numpy as np

    for key, off in (("portfolio.sector_cap", None), ("portfolio.sector_cap", 0.0), ("allocator.target_vol_annual", 0.0),
                     ("signals.normalizer_window_days", 0), ("signals.ewm_memory_spans", 0),
                     ("universe.history_window_days", 0), ("costs.commission_cap", 0.0), ("signals.momentum_skip", 0)):
        EngineConfig().replace(**{key: off})
    EngineConfig().replace(**{"portfolio.target_positions": 20.0, "portfolio.exit_rank": np.int64(40),
                              "portfolio.max_weight": np.float32(0.08)})
    with pytest.raises(ValueError, match="target_positions=20.5 is not a whole number"):
        EngineConfig().replace(**{"portfolio.target_positions": 20.5})


@pytest.mark.parametrize("name", ["nse_engine_deployed", "nse_engine_candidate", "nse_engine_e4", "nse_engine_a1",
                                  "nse_engine_bonds", "nse_engine_futures", "nse_engine_mcx"])
def test_every_shipped_book_passes_the_bounds(name):
    assert range_problems(load_deployment(ROOT / f"config/{name}.json").engine) == []


def test_the_gaps_found_are_refused_by_either_route():
    """QP1: these built without complaint and produced a book that stops every name at once, holds nothing or sells
    every holding the next day; each would also have counted as a trial."""
    for key, bad, why in (("portfolio.stop_atr_multiple", -6.0, "below its exclusive minimum 0"),
                          ("portfolio.target_positions", 0, "below its minimum 1"),
                          ("portfolio.exit_rank", 0, "below its minimum 1"),
                          ("signals.fast_ewmac", ((32, 8),), "1 <= fast < slow"),
                          ("initial_capital", 0.0, "positive finite amount")):
        with pytest.raises(ValueError, match=why):
            EngineConfig().replace(**{key: bad})
    raw = json.loads(EngineConfig().to_json())
    raw["portfolio"]["stop_atr_multiple"] = -6.0
    with pytest.raises(ValueError, match="stop_atr_multiple=-6.0"):
        EngineConfig.from_dict(raw)
