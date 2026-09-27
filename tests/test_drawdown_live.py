"""The drawdown rule in the live book (tracker G3).

The deployment carries the rule as an overlay (the strategy keeps its config
hash); the executor replays it over the book's equity history every session,
blocks entries and scales exposure through ``generate_targets``, and the state
reaches the session record, the daily email and the monitor.
"""
from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nse_engine.config import DrawdownConfig, EngineConfig
from nse_engine.deployment import DeploymentError, load_deployment, parse_deployment
from nse_engine.types import TargetPortfolio
from tests.conftest import ROOT, synthetic_panel

RULE = {"enabled": True, "halt_dd": 0.20, "half_dd": 0.30, "risk_off_dd": 0.35, "half_scale": 0.5, "rearm_sessions": 60}


def _raw(overlay=None):
    raw = {"status": "placeholder", "paper_start_date": "2021-01-04", "engine": EngineConfig().to_dict()}
    if overlay is not None:
        raw["risk_overlay"] = overlay
    return raw


# ---------------------------------------------------------------- deployment

class TestDeploymentOverlay:
    def test_absent_overlay_means_no_rule_and_the_same_hash(self):
        dep = parse_deployment(_raw())
        assert dep.drawdown_rule is None
        with_rule = parse_deployment(_raw({"drawdown_rule": RULE}))
        assert with_rule.drawdown_rule == DrawdownConfig(**RULE)
        assert with_rule.engine.config_hash() == dep.engine.config_hash(), "an overlay never touches the strategy hash"
        assert with_rule.summary()["drawdown_rule"]["halt_dd"] == 0.20

    def test_disabled_rule_is_no_rule(self):
        assert parse_deployment(_raw({"drawdown_rule": {**RULE, "enabled": False}})).drawdown_rule is None

    @pytest.mark.parametrize("bad", [
        {"drawdown_rule": {**RULE, "halt_dd": 0.4}},          # thresholds out of order
        {"drawdown_rule": {**RULE, "bogus": 1}},              # unknown field
        {"something_else": {}},                                # unknown overlay key
        {"drawdown_rule": "halt at 20%"},                      # not an object
    ])
    def test_invalid_overlays_are_rejected(self, bad):
        with pytest.raises(DeploymentError):
            parse_deployment(_raw(bad))

    def test_the_deployed_file_carries_e2s_rule(self):
        dep = load_deployment(ROOT / "config/nse_engine_deployed.json")
        assert dep.drawdown_rule == DrawdownConfig(**RULE)
        assert dep.engine.config_hash() == "679cbd0ceaf7c895"


# ---------------------------------------------------------------- executor

@pytest.fixture(scope="module")
def panel():
    return synthetic_panel(n_days=900, n_symbols=40)


def _executor(panel, history, overlay=True, capture=None, holdings=None, cash=100_000.0):
    from kite_connect.trading.nse_engine_executor import EngineExecutor

    dep = parse_deployment(_raw({"drawdown_rule": RULE} if overlay else None))
    holdings = holdings if holdings is not None else {}      # a pure-cash book: equity == cash exactly
    captured = capture if capture is not None else {}

    def fake_targets(view, cfg, as_of, **kw):
        captured.update(kw)
        weights = {s: 0.05 for s in holdings}
        return TargetPortfolio(as_of=as_of, weights=weights, forecasts={}, ranks={}, exits={},
                               drawdown_state=kw["drawdown"].state if kw.get("drawdown") else "normal")

    return EngineExecutor(kite=None, paper=True, deployment=dep,
                          target_fn=fake_targets, data_loader=lambda *a, **k: panel,
                          holdings_fn=lambda: (holdings, cash), stopped_out_fn=lambda: {},
                          equity_history_fn=lambda: history,
                          shift_state_path=str(ROOT / "tests" / "_no_shift_state.json"))


def _history(path, start="2021-01-04"):
    idx = pd.bdate_range(start, periods=len(path))
    return pd.Series(path, index=idx, dtype="float64")


class TestExecutorDecision:
    AS_OF = pd.Timestamp("2021-09-01")

    def _plan(self, panel, path, **kw):
        captured = {}
        ex = _executor(panel, _history(path), capture=captured, **kw)
        plan = ex.plan(as_of=self.AS_OF)
        return plan, captured

    def test_flat_history_is_normal_and_the_decision_reaches_generate_targets(self, panel):
        plan, captured = self._plan(panel, [1e6] * 80, cash=1e6)
        assert plan.drawdown_state == "normal" and plan.drawdown_scale == 1.0
        assert captured["drawdown"].state == "normal" and captured["drawdown"].allow_entries
        assert any(n.startswith("drawdown rule: normal") for n in plan.notes)

    def test_halt_half_and_risk_off_from_the_history(self, panel):
        # peak 1.0m in the history; today's mark (cash only) decides the state
        for cash, state, scale in ((790_000, "halt", 1.0), (690_000, "half", 0.5), (640_000, "risk_off", 0.0)):
            plan, captured = self._plan(panel, [1_000_000] * 80, cash=cash)
            dd = 1 - plan.equity / 1_000_000
            assert plan.drawdown_state == state, (cash, plan.equity, dd)
            assert plan.drawdown_scale == scale
            assert captured["drawdown"].state == state and not captured["drawdown"].allow_entries
            assert plan.drawdown_changed, "first session past the threshold is a change"
            assert abs(plan.drawdown_pct - dd * 100) < 0.01

    def test_the_state_persists_and_rearms_only_on_a_new_high(self, panel):
        # the book fell 25% and has since oscillated below its 60-session high: still halted.
        # (a steady creep upward would re-arm by itself: every step is a new 60-session high)
        path = [1_000_000] * 5 + [750_000] + [760_000, 755_000] * 35
        plan, captured = self._plan(panel, path, cash=758_000)
        assert plan.drawdown_state == "halt" and not plan.drawdown_changed
        assert abs(plan.drawdown_pct - 24.2) < 0.01
        # equity above every one of the last 60 sessions: re-armed, peak reset, entries allowed again
        plan2, captured2 = self._plan(panel, path, cash=900_000)
        assert plan2.drawdown_state == "normal" and plan2.drawdown_changed
        assert captured2["drawdown"].allow_entries and plan2.drawdown_peak == plan2.equity

    def test_no_overlay_means_the_old_call_signature(self, panel):
        plan, captured = self._plan(panel, [1_000_000] * 80, overlay=False, cash=500_000)
        assert "drawdown" not in captured and plan.drawdown_state == "normal"
        assert not any("drawdown" in n for n in plan.notes)

    def test_a_broken_history_source_does_not_stop_the_session(self, panel):
        from kite_connect.trading.nse_engine_executor import EngineExecutor

        dep = parse_deployment(_raw({"drawdown_rule": RULE}))
        ex = EngineExecutor(kite=None, paper=True, deployment=dep,
                            target_fn=lambda view, cfg, as_of, **kw: TargetPortfolio(as_of=as_of, weights={}),
                            data_loader=lambda *a, **k: panel, holdings_fn=lambda: ({}, 1.0),
                            stopped_out_fn=lambda: {},
                            equity_history_fn=lambda: (_ for _ in ()).throw(RuntimeError("neon down")))
        plan = ex.plan(as_of=self.AS_OF)
        assert plan.drawdown_state == "normal"      # today only: nothing to compare with


# ---------------------------------------------------------------- paper trader history

def test_equity_history_comes_from_the_restored_snapshots(monkeypatch, tmp_path):
    import kite_connect.trading.paper_trader as ptmod

    monkeypatch.setattr(ptmod, "_DB_PATH", tmp_path / "paper.sqlite3")
    pt = ptmod.PaperTrader(kite=None, initial_capital=3_500_000)
    conn = sqlite3.connect(str(ptmod._DB_PATH))
    for d, e in (("2026-09-17", 3_523_526), ("2026-09-18", 3_577_634), ("2026-09-18", 3_577_700), ("2026-09-19", 3_560_000)):
        conn.execute("INSERT OR REPLACE INTO daily_snapshots (date, equity, cash, open_positions, closed_today, day_pnl, "
                     "cumulative_pnl, cumulative_pnl_pct, max_drawdown_pct, signals_generated, signals_traded, snapshot_json) "
                     "VALUES (?, ?, 0, 0, 0, 0, 0, 0, 0, 0, 0, '{}')", (d, e))
    conn.commit(); conn.close()
    hist = pt.equity_history()
    assert list(hist.index.strftime("%Y-%m-%d")) == ["2026-09-17", "2026-09-18", "2026-09-19"]
    assert hist.loc["2026-09-18"] == 3_577_700, "a re-snapshotted day keeps its last value"


# ---------------------------------------------------------------- record, email, columns

class TestSessionRecordAndEmail:
    class _Plan:
        def __init__(self, state="normal", pct=0.0, changed=False):
            self.notes, self.skipped, self.stop_instructions = ["rebalance_day"], [], []
            self.buys, self.sells, self.shift_multiplier = [], [], 1.0
            self.drawdown_state, self.drawdown_pct, self.drawdown_changed = state, pct, changed

    def _record(self, plan):
        import cloud_paper_runner as runner

        captured = {}
        pt = type("PT", (), {"_get_cloud": lambda self: type("C", (), {"sync_session": lambda s, row: captured.update(row) or True})(),
                             "cash": 1.0})()
        runner._record_session_activity(pt, {"session": "2026-10-01", "notes": [], "stops": [], "fills": {}},
                                        {"equity": 1.0, "open_positions": 20}, plan, 0, {})
        return captured

    def test_the_session_record_carries_the_state_and_the_outcome_says_why(self):
        row = self._record(self._Plan("halt", 21.3, True))
        assert row["drawdown_state"] == "halt" and row["drawdown_pct"] == 21.3
        assert row["outcome"].startswith("DRAWDOWN RULE halt (21.3% below peak): no new entries or adds; ")
        assert self._record(self._Plan())["outcome"].startswith("held:")
        assert self._record(self._Plan())["drawdown_state"] == "normal"

    def test_the_daily_email_gets_a_line_and_an_alert_on_change(self, monkeypatch):
        import cloud_paper_runner as runner

        sent = {}

        class FakeManager:
            def email_engine_daily_report(self, report):
                sent.update(report); return True

        monkeypatch.setattr("services.notifications.manager.NotificationManager", FakeManager)
        dash = SimpleNamespace(current_capital=1.0, initial_capital=1.0, total_pnl=0.0, total_pnl_pct=0.0,
                               max_drawdown_pct=0.0, open_positions=0)
        pt = SimpleNamespace(dashboard=lambda: dash, cash=1.0)
        dep = SimpleNamespace(status="approved", paper_start_date="2026-09-16", drawdown_rule=DrawdownConfig(**RULE))
        session = {"session": "2026-10-01", "fills": {}, "stops": [], "results": [], "notes": [],
                   "plan": self._Plan("half", 31.0, True)}
        runner._email_engine_session(pt, dep, session, {}, {})
        assert sent["drawdown_state"] == "half" and "31.0% below the episode peak" in sent["drawdown_rule"]
        assert any(a.startswith("DRAWDOWN RULE changed to HALF") for a in sent["alerts"])
        session["plan"] = self._Plan("normal", 2.0, True)          # a re-arm
        runner._email_engine_session(pt, dep, session, {}, {})
        assert any("re-armed" in a for a in sent["alerts"])

    def test_the_subject_and_summary_flag_the_state(self, monkeypatch):
        from services.notifications.manager import NotificationManager

        seen = {}
        nm = NotificationManager.__new__(NotificationManager)
        monkeypatch.setattr(NotificationManager, "_send_html_email", lambda self, subject, html: seen.update(subject=subject, html=html) or True)
        base = {"session": "2026-10-01", "equity": 3_000_000, "initial_capital": 3_500_000, "cash": 1, "pnl": -500_000,
                "pnl_pct": -14.3, "max_drawdown_pct": 21.0, "open_positions": 12, "filled": [], "cancelled": [],
                "stops": [], "queued": [], "notes": [], "alerts": [],
                "drawdown_rule": "halt · 21.3% below the episode peak", "drawdown_state": "halt"}
        assert nm.email_engine_daily_report(base)
        assert "[drawdown halt]" in seen["subject"] and "Drawdown rule" in seen["html"] and "21.3% below" in seen["html"]
        assert nm.email_engine_daily_report({**base, "drawdown_state": "normal", "drawdown_rule": "normal · 2.0% below the episode peak"})
        assert "[drawdown" not in seen["subject"]


def test_missing_session_columns_are_added_once():
    from sqlalchemy import create_engine, inspect, text

    from database.paper_cloud import SESSION_COLUMNS_ADDED, add_missing_columns

    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE paper_sessions (session_date VARCHAR(10) PRIMARY KEY, outcome VARCHAR(200))"))
    assert add_missing_columns(engine, "paper_sessions", SESSION_COLUMNS_ADDED) == ["drawdown_state", "drawdown_pct"]
    assert add_missing_columns(engine, "paper_sessions", SESSION_COLUMNS_ADDED) == []
    assert {"drawdown_state", "drawdown_pct"} <= {c["name"] for c in inspect(engine).get_columns("paper_sessions")}
    assert add_missing_columns(engine, "no_such_table", SESSION_COLUMNS_ADDED) == []
