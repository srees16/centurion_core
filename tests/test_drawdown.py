"""Drawdown rule (tracker E2 / G3): exposure control from the book's own equity.

Off by default and hash-neutral, so the deployed configuration keeps its
identity. On, it must (1) follow the declared state machine, (2) never open a
new core name or add to one outside ``normal``, (3) leave results untouched
when its thresholds cannot be reached, and (4) be a pure function of the
equity history, so live can replay it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nse_engine.config import DrawdownConfig, EngineConfig
from nse_engine.drawdown import (HALF, HALT, NORMAL, RISK_OFF, DrawdownTracker, replay,
                                 severity_for, summarise)
from nse_engine.engine import run_backtest
from tests.conftest import synthetic_panel

CFG = DrawdownConfig(enabled=True, halt_dd=0.15, half_dd=0.25, risk_off_dd=0.30, half_scale=0.5, rearm_sessions=5)


def _states(equities, cfg=CFG):
    t = DrawdownTracker(cfg)
    return [t.update(e).state for e in equities]


def test_severity_thresholds_are_strict():
    assert severity_for(0.15, CFG) == NORMAL and severity_for(0.1501, CFG) == HALT
    assert severity_for(0.25, CFG) == HALT and severity_for(0.2501, CFG) == HALF
    assert severity_for(0.30, CFG) == HALF and severity_for(0.31, CFG) == RISK_OFF


def test_state_only_worsens_within_an_episode_and_rearms_on_a_new_high():
    # peak 100, fall to 80 (halt), 71 (half), bounce to 78 (still half: no adds on a bounce),
    # then a new 5-session high re-arms and resets the peak
    eq = [100, 95, 80, 71, 78, 79, 80, 81, 82, 83]
    s = _states(eq)
    assert s[:4] == [NORMAL, NORMAL, HALT, HALF]
    assert s[4:7] == [HALF, HALF, HALF], "a bounce inside the episode never relaxes the state"
    assert s[7] == NORMAL, "81 beats the previous five sessions (80, 71, 78, 79, 80): re-armed"
    assert s[8:] == [NORMAL, NORMAL]
    t = DrawdownTracker(CFG)
    for e in eq:
        d = t.update(e)
    assert d.peak == 83 and d.state == NORMAL and d.drawdown == 0.0


def test_after_a_rearm_the_peak_is_the_new_episode_peak_not_the_old_high():
    eq = [100, 80, 81, 82, 83, 84, 85]          # halt at 80, re-arm at 85 (beats 80..84)
    t = DrawdownTracker(CFG)
    for e in eq:
        d = t.update(e)
    assert d.state == NORMAL and d.peak == 85
    d = t.update(75)                             # 12% below the new peak: no halt yet
    assert d.state == NORMAL and abs(d.drawdown - (1 - 75 / 85)) < 1e-12
    assert t.update(72).state == HALT            # 15.3% below 85


def test_risk_off_and_scales_and_flags():
    eq = [100, 69]                               # 31% in one step: straight to risk-off
    t = DrawdownTracker(CFG)
    t.update(100)
    d = t.update(69)
    assert d.state == RISK_OFF and d.scale == 0.0 and not d.allow_entries and d.changed
    d2 = t.update(69)
    assert d2.state == RISK_OFF and not d2.changed and d2.sessions_in_state == 2
    t2 = DrawdownTracker(CFG); t2.update(100)
    assert t2.update(74).scale == 0.5            # 26%: half


def test_replay_matches_the_incremental_tracker_and_summarise_counts_episodes():
    rng = np.random.default_rng(0)
    eq = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.03, 400))), index=pd.bdate_range("2020-01-01", periods=400))
    frame = replay(eq, CFG)
    assert list(frame["state"]) == _states(eq.to_numpy())
    assert set(frame.columns) >= {"equity", "peak", "drawdown", "state", "scale", "allow_entries", "changed"}
    summary = summarise(frame["state"])
    assert abs(sum(summary[f"share_{s}"] for s in (NORMAL, HALT, HALF, RISK_OFF)) - 1.0) < 1e-12
    assert summary["episodes"] == float(((frame["state"] != NORMAL) & (frame["state"].shift(1, fill_value=NORMAL) == NORMAL)).sum())


def test_bad_thresholds_are_rejected():
    with pytest.raises(ValueError):
        DrawdownTracker(DrawdownConfig(enabled=True, halt_dd=0.3, half_dd=0.2, risk_off_dd=0.4))


# ---------------------------------------------------------------- engine integration

def test_off_by_default_and_hash_neutral():
    base = EngineConfig()
    assert not base.drawdown.enabled
    # a config recorded before the section existed hashes the same as one with it at defaults
    legacy = {k: v for k, v in base.to_dict().items() if k != "drawdown"}
    assert EngineConfig.from_dict(legacy).config_hash() == base.config_hash()
    assert base.replace(**{"drawdown.enabled": False, "drawdown.halt_dd": 0.15}).config_hash() == base.config_hash()
    assert base.replace(**{"drawdown.enabled": True}).config_hash() != base.config_hash()
    assert base.replace(**{"drawdown.enabled": True, "drawdown.halt_dd": 0.2}).config_hash() != \
        base.replace(**{"drawdown.enabled": True}).config_hash()


@pytest.fixture(scope="module")
def panel():
    return synthetic_panel(n_days=900, n_symbols=40)


def _cfg(**dd):
    return EngineConfig().replace(**{"start": "2020-06-01", "end": "2022-06-30", "universe.top_n_liquid": 30,
                                     "universe.min_history_days": 120, "signals.normalizer_min_obs": 40,
                                     "portfolio.target_positions": 8, "portfolio.max_positions": 10,
                                     **{f"drawdown.{k}": v for k, v in dd.items()}})


def test_unreachable_thresholds_change_nothing(panel):
    off = run_backtest(panel, _cfg(), record=False)
    on = run_backtest(panel, _cfg(enabled=True, halt_dd=0.99, half_dd=0.99, risk_off_dd=0.99), record=False)
    pd.testing.assert_series_equal(off.returns, on.returns)
    assert off.daily_state is None and on.daily_state is not None
    assert (on.daily_state["state"] == NORMAL).all() and on.metrics["dd_rule_episodes"] == 0


def test_no_new_core_names_and_no_adds_outside_normal(panel):
    # the synthetic book runs near 20% gross and never loses more than ~1%, so the
    # thresholds sit inside that range to exercise every state
    cfg = _cfg(enabled=True, halt_dd=0.003, half_dd=0.006, risk_off_dd=0.009, rearm_sessions=20)
    res = run_backtest(panel, cfg, record=False)
    st = res.daily_state
    assert (st["state"] != NORMAL).any(), "the tight thresholds must trigger on this panel"
    sleeves = {cfg.sleeves.gold_symbol, cfg.sleeves.silver_symbol}
    dates = list(res.equity.index)
    prev = {d: dates[i - 1] for i, d in enumerate(dates) if i > 0}
    held = set()
    for row in res.trades.sort_values("date", kind="stable").itertuples(index=False):
        if row.symbol in sleeves:
            continue
        if row.side == "BUY":
            decision_state = st.loc[prev[row.date], "state"] if row.date in prev else NORMAL
            assert decision_state == NORMAL, f"{row.symbol} bought on {row.date.date()} while {decision_state}"
        held.add(row.symbol)
    assert res.metrics["dd_rule_share_normal"] < 1.0
    # the rule's own record is written with the run
    assert set(st.columns) >= {"state", "scale", "allow_entries", "drawdown", "peak", "changed"}
