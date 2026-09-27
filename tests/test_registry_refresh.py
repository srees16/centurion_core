"""Registry continuity after a data-fingerprint change (tracker V2).

``returns_matrix`` compares only runs that share a data hash. When a store
rebuild changes the fingerprint (a symbol rename is enough), a new run would
meet no prior configurations: no PBO, deflated Sharpe at N = 1. The refresh
re-runs every same-window configuration on the new data, keeps the config
hashes (so they dedupe against the old runs rather than count as new trials)
and verifies the returns reproduce.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nse_engine.config import EngineConfig
from nse_engine.validation.trials import (TrialRegistry, compare_returns, record_result,
                                          refresh_plan, refresh_registry)

WINDOW = ("2020-01-01", "2021-12-31")
OLD, NEW = "oldhash000000001", "newhash000000002"
DATES = pd.bdate_range(WINDOW[0], WINDOW[1])


def _config(n_positions: int, runs_dir=None) -> EngineConfig:
    fields = {"start": WINDOW[0], "end": WINDOW[1], "portfolio.target_positions": n_positions}
    if runs_dir is not None:
        fields["runs_dir"] = str(runs_dir)          # frozen dataclass: set through replace
    return EngineConfig().replace(**fields)


def _returns(seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0005, 0.01, len(DATES)), index=DATES, name="return")


def _record(runs_dir: Path, cfg: EngineConfig, returns: pd.Series, data_hash: str, tag: str,
            extra=None) -> str:
    res = SimpleNamespace(returns=returns, data_hash=data_hash, config=cfg, metrics={"sharpe": 1.0},
                          run_id="", run_dir=None)
    record_result(res, tag=tag, runs_dir=runs_dir, window=WINDOW, extra=extra)
    return res.run_id


@pytest.fixture
def registry(tmp_path):
    """Three configurations on the old hash, one of them recorded twice."""
    runs = tmp_path / "runs"
    for n, seed in ((20, 1), (25, 2), (30, 3)):
        _record(runs, _config(n, runs), _returns(seed), OLD, tag=f"grid:tp{n}")
    _record(runs, _config(20, runs), _returns(1), OLD, tag="grid:tp20 rerun")   # duplicate config
    return TrialRegistry(runs)


def _fake_backtest(returns_by_hash, perturb: str = ""):
    """Engine stand-in: reproduces the recorded returns for the config hash,
    optionally perturbing one configuration, and records like the engine."""
    def run(data, config, *, record=True, tag="", lag_days=0, manifest_extra=None):
        r = returns_by_hash[config.config_hash()].copy()
        if perturb and config.config_hash() == perturb:
            r.iloc[10] += 1e-4
        res = SimpleNamespace(returns=r, data_hash=data.data_hash, config=config, metrics={},
                              run_id="", run_dir=None)
        if record:
            record_result(res, tag=tag, runs_dir=config.runs_dir, window=WINDOW, extra=manifest_extra)
        return res
    return run


def _returns_by_hash():
    return {_config(n).config_hash(): _returns(seed) for n, seed in ((20, 1), (25, 2), (30, 3))}


def test_plan_lists_each_configuration_once_with_its_tag_and_lag(registry):
    plan = refresh_plan(registry, OLD, WINDOW)
    assert len(plan) == 3
    assert set(plan["config_hash"]) == set(_returns_by_hash())
    assert (plan["lag_days"] == 0).all()
    tp20 = plan[plan["tag"].str.startswith("grid:tp20")]
    assert list(tp20["tag"]) == ["grid:tp20 rerun"], "the latest run of a duplicated config is the reference"


def test_refresh_reproduces_every_configuration_on_the_new_hash(registry):
    data = SimpleNamespace(data_hash=NEW)
    before = registry.returns_matrix(data_hash=OLD, window=WINDOW)
    report = refresh_registry(registry, data, OLD, WINDOW, backtest_fn=_fake_backtest(_returns_by_hash()),
                              config_loader=lambda d: EngineConfig.from_dict(json.loads((Path(d) / "config.json").read_text())))
    assert report["n_planned"] == 3 and report["all_identical"] and report["complete"]
    after = registry.returns_matrix(data_hash=NEW, window=WINDOW)
    assert after.shape[1] == 3 == before.shape[1]
    # same configurations, same numbers: PBO inputs are unchanged
    old_by_hash = {json.loads((Path(registry.runs_dir) / c / "manifest.json").read_text())["config_hash"]: before[c]
                   for c in before.columns}
    for c in after.columns:
        man = json.loads((Path(registry.runs_dir) / c / "manifest.json").read_text())
        assert man["refresh_of"] in before.columns
        assert man["data_hash"] == NEW
        pd.testing.assert_series_equal(after[c].rename(None), old_by_hash[man["config_hash"]].rename(None))
    # the old hash is untouched and the refresh is idempotent
    assert registry.returns_matrix(data_hash=OLD, window=WINDOW).shape[1] == 3
    assert refresh_plan(registry, OLD, WINDOW, skip_hash=NEW).empty
    assert registry.list_trials()["refresh_of"].notna().sum() == 3


def test_refresh_flags_a_configuration_that_does_not_reproduce(registry):
    bad = _config(25).config_hash()
    report = refresh_registry(registry, SimpleNamespace(data_hash=NEW), OLD, WINDOW,
                              backtest_fn=_fake_backtest(_returns_by_hash(), perturb=bad))
    rows = {r["config_hash"]: r for r in report["rows"]}
    assert not report["all_identical"]
    assert not rows[bad]["identical"] and rows[bad]["n_differing"] == 1 and rows[bad]["max_abs_diff"] > 1e-9
    assert all(rows[h]["identical"] for h in rows if h != bad)


def test_dry_run_and_unchanged_hash_write_nothing(registry):
    n_before = len(list(Path(registry.runs_dir).iterdir()))
    report = refresh_registry(registry, SimpleNamespace(data_hash=NEW), OLD, WINDOW, dry_run=True,
                              backtest_fn=lambda *a, **k: pytest.fail("dry run must not run backtests"))
    assert report["dry_run"] and report["n_planned"] == 3 and len(report["rows"]) == 3
    same = refresh_registry(registry, SimpleNamespace(data_hash=OLD), OLD, WINDOW,
                            backtest_fn=lambda *a, **k: pytest.fail("unchanged hash must not run backtests"))
    assert "unchanged" in same["note"]
    assert len(list(Path(registry.runs_dir).iterdir())) == n_before


def test_compare_returns_reports_length_and_value_differences():
    a = _returns(5)
    assert compare_returns(a, a)["identical"]
    b = a.copy(); b.iloc[3] += 2e-9
    out = compare_returns(a, b)
    assert not out["identical"] and out["n_differing"] == 1
    assert compare_returns(a, b, tolerance=1e-8)["identical"]
    short = compare_returns(a, a.iloc[:-5])
    assert not short["identical"] and short["n_common"] == len(a) - 5 and short["n_differing"] == 5


def test_fingerprint_status_follows_the_latest_recorded_run(registry):
    from nse_engine.validation.trials import fingerprint_status, registry_hash

    assert registry_hash(registry, WINDOW) == OLD
    unchanged = fingerprint_status(registry, WINDOW, OLD)
    assert not unchanged["changed"] and unchanged["n_configurations_on_registry_hash"] == 3
    changed = fingerprint_status(registry, WINDOW, NEW)
    assert changed["changed"] and changed["registry_hash"] == OLD
    assert changed["n_configurations_on_current_hash"] == 0
    # after a refresh the registry sits on the new hash: the check goes quiet again
    refresh_registry(registry, SimpleNamespace(data_hash=NEW), OLD, WINDOW,
                     backtest_fn=_fake_backtest(_returns_by_hash()))
    assert registry_hash(registry, WINDOW) == NEW
    after = fingerprint_status(registry, WINDOW, NEW)
    assert not after["changed"] and after["n_configurations_on_registry_hash"] == 3
    assert fingerprint_status(TrialRegistry(Path(registry.runs_dir) / "empty"), WINDOW, NEW)["registry_hash"] is None


def test_build_store_runs_the_fingerprint_check(monkeypatch, capsys, tmp_path):
    """The rebuild command must end with the registry check, and the check
    must print the refresh instruction when the hash changed."""
    import runners.run_nse_engine as cli

    calls = []
    real_check = cli.registry_check_after_rebuild
    monkeypatch.setattr("nse_engine.data.store.build_store", lambda *a, **k: {"years": []})
    monkeypatch.setattr("nse_engine.data.reference.build_sector_map", lambda *a: {})
    monkeypatch.setattr(cli, "registry_check_after_rebuild", lambda cfg: calls.append(cfg) or {})
    cli.cmd_build_store(SimpleNamespace(skip_registry_check=False))
    assert len(calls) == 1
    cli.cmd_build_store(SimpleNamespace(skip_registry_check=True))
    assert len(calls) == 1

    # the check itself, with the data load stubbed: changed hash -> plan and instruction
    runs = tmp_path / "runs"
    _record(runs, _config(20, runs), _returns(1), OLD, tag="grid:tp20")
    cfg = EngineConfig().replace(runs_dir=str(runs), start=WINDOW[0], end=WINDOW[1])
    monkeypatch.setattr(cli, "_load_data", lambda cfg, data_start=None: SimpleNamespace(data_hash=NEW))
    status = real_check(cfg)
    out = capsys.readouterr().out
    assert status["changed"] and "STORE FINGERPRINT CHANGED" in out and "refresh-registry --dry-run" in out
    assert status["dry_run"]["n_planned"] == 1
    monkeypatch.setattr(cli, "_load_data", lambda cfg, data_start=None: SimpleNamespace(data_hash=OLD))
    assert not real_check(cfg)["changed"]
    assert "unchanged" in capsys.readouterr().out
    # a failing load must not fail the rebuild
    monkeypatch.setattr(cli, "_load_data", lambda cfg, data_start=None: (_ for _ in ()).throw(RuntimeError("no store")))
    assert real_check(cfg)["changed"] is None
