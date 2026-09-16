"""
NSE engine paper path: deployment file, next-open pending fills, shift
multiplier, reality-gap alerts and the GitHub Actions workflow.

Run:  python -m unittest tests.test_paper_engine -v
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import unittest
from argparse import Namespace
from datetime import date
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from tests.test_live_execution import FakeCloud, FakeKite, PaperTestBase

ROOT = Path(__file__).resolve().parent.parent


def _engine_dict():
    from nse_engine.config import EngineConfig
    return EngineConfig().to_dict()


def _doc(**overrides):
    doc = {"status": "approved", "paper_start_date": "2026-01-05", "source_run_id": "run-abc",
           "approved_at": "2026-01-02T18:00:00+05:30", "notes": "", "engine": _engine_dict()}
    doc.update(overrides)
    return doc


class TempDirTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        # Isolate from the repo's real deployment file (its paper_start_date and
        # data anchor change whenever a configuration is promoted).
        env = mock.patch.dict(os.environ, {"CENTURION_NSE_DEPLOYMENT": str(self.tmp / "no_deployment.json")})
        env.start()
        self.addCleanup(env.stop)

    def write(self, name, obj):
        p = self.tmp / name
        p.write_text(obj if isinstance(obj, str) else json.dumps(obj))
        return p


# ─────────────────────────────────────────────────────────────
# 1. Deployment loader
# ─────────────────────────────────────────────────────────────

class TestDeployment(TempDirTest):
    def test_repo_deployment_loads(self):
        from nse_engine.deployment import REPO_ROOT, load_deployment
        dep = load_deployment(REPO_ROOT / "config" / "nse_engine_deployed.json")
        if dep.is_placeholder:
            self.assertFalse(dep.live_allowed()[0])
            self.assertIn("placeholder", dep.live_allowed()[1])
        else:
            self.assertTrue(dep.source_run_id and dep.approved_at)
            self.assertTrue(dep.live_allowed()[0])
            self.assertLess(dep.data_start(), dep.paper_start_date)

    def test_valid_file_and_live_config(self):
        from nse_engine.deployment import load_deployment
        eng = _engine_dict()
        eng["portfolio"]["target_positions"] = 25
        dep = load_deployment(self.write("d.json", _doc(engine=eng)))
        self.assertEqual(dep.status, "approved")
        self.assertTrue(dep.live_allowed()[0])
        self.assertEqual(dep.paper_start_date, date(2026, 1, 5))
        self.assertEqual(dep.engine.portfolio.target_positions, 25)
        self.assertEqual(dep.live_config().start, "2026-01-05")
        self.assertEqual(dep.bootstrap_start(), date(2023, 1, 5))

    def test_missing_file_is_a_clear_error(self):
        from nse_engine.deployment import DeploymentError, load_deployment
        with self.assertRaises(DeploymentError) as cm:
            load_deployment(self.tmp / "nope.json")
        self.assertIn("not found", str(cm.exception))
        self.assertIn("nope.json", str(cm.exception))

    def test_invalid_documents(self):
        from nse_engine.deployment import DeploymentError, load_deployment
        eng = _engine_dict()
        eng["portfolio"]["target_postions"] = 25  # typo must not be silently ignored
        cases = {
            "unknown top-level": _doc(extra=1),
            "unknown engine key": _doc(engine=eng),
            "unknown engine section": _doc(engine={**_engine_dict(), "risk": {}}),
            "missing status": {k: v for k, v in _doc().items() if k != "status"},
            "bad status": _doc(status="live"),
            "approved without run": _doc(source_run_id=None),
            "bad date": _doc(paper_start_date="15/09/2026"),
            "not an object": [1, 2],
        }
        for label, doc in cases.items():
            with self.subTest(label), self.assertRaises(DeploymentError):
                load_deployment(self.write("bad.json", doc))
        with self.assertRaises(DeploymentError):
            load_deployment(self.write("bad.json", "{not json"))

    def test_engine_json_is_accepted_by_run_nse_engine(self):
        from nse_engine.deployment import load_deployment, main
        path = self.write("d.json", _doc())
        out = self.tmp / "engine.json"
        with mock.patch("sys.stdout"):
            self.assertEqual(main(["--path", str(path), "write-engine-config", "--out", str(out)]), 0)
        cwd = os.getcwd()
        try:
            from runners import run_nse_engine
            cfg = run_nse_engine._build_config(Namespace(config=str(out), set=["initial_capital=100000"],
                                                         start="2026-01-05", end=None))
        finally:
            os.chdir(cwd)
        expected = load_deployment(path).live_config().replace(initial_capital=100000)
        self.assertEqual(cfg, expected)

    def test_runner_loads_deployment_from_env(self):
        import cloud_paper_runner as cpr
        path = self.write("d.json", _doc(status="placeholder", source_run_id=None, approved_at=None))
        with mock.patch.dict(os.environ, {"CENTURION_NSE_DEPLOYMENT": str(path)}), \
                self.assertLogs("cloud_paper_runner", level="WARNING") as logs:
            dep = cpr._load_engine_deployment()
        self.assertTrue(dep.is_placeholder)
        self.assertTrue(any("PLACEHOLDER" in m for m in logs.output))


class TestPlaceholderGuard(TempDirTest):
    LIVE_ENV = {"CENTURION_PAPER_TRADE": "false", "CENTURION_NSE_ENGINE_LIVE": "true"}

    def test_placeholder_refuses_live_but_allows_paper(self):
        from kite_connect.trading.nse_engine_executor import EngineExecutor
        placeholder = self.write("p.json", _doc(status="placeholder", source_run_id=None, approved_at=None))
        approved = self.write("a.json", _doc())
        with mock.patch.dict(os.environ, self.LIVE_ENV):
            ex = EngineExecutor(kite=FakeKite(), paper=False, deployment_path=str(placeholder))
            self.assertTrue(ex.paper)
            self.assertIn("placeholder", ex.mode_reason)
            ex = EngineExecutor(kite=FakeKite(), paper=False, deployment_path=str(self.tmp / "missing.json"))
            self.assertTrue(ex.paper)
            self.assertIn("deployment unavailable", ex.mode_reason)
            ex = EngineExecutor(kite=FakeKite(), paper=False, deployment_path=str(approved))
            self.assertFalse(ex.paper)
        # Paper: the placeholder config is used (start = paper_start_date)
        ex = EngineExecutor(kite=None, paper=True, deployment_path=str(placeholder))
        with self.assertLogs("kite_connect.trading.nse_engine_executor", level="WARNING"):
            cfg = ex._cfg()
        self.assertEqual(cfg.start, "2026-01-05")

    def test_missing_deployment_is_an_error_for_planning(self):
        from kite_connect.trading.nse_engine_executor import EngineExecutor
        from nse_engine.deployment import DeploymentError
        ex = EngineExecutor(kite=None, paper=True, deployment_path=str(self.tmp / "missing.json"))
        with self.assertRaises(DeploymentError):
            ex._cfg()


# ─────────────────────────────────────────────────────────────
# 2. Pending next-open orders
# ─────────────────────────────────────────────────────────────

SESSIONS = pd.bdate_range(end="2024-03-28", periods=10)
D_PREV, D_SESSION = SESSIONS[-2], SESSIONS[-1]


class TestPendingOrders(PaperTestBase):
    def _fill(self, pt, quotes, **kw):
        from nse_engine.config import CostConfig
        return pt.fill_pending_orders(D_SESSION, SESSIONS, quotes, CostConfig(), **kw)

    def test_lifecycle_next_open_fill_with_costs(self):
        from nse_engine.config import CostConfig
        from nse_engine.costs import simulate_fill
        cloud = FakeCloud()
        pt = self.trader(initial_capital=100_000, cloud=cloud)
        rows = pt.queue_pending_orders(D_PREV, [
            {"symbol": "AAA", "side": "BUY", "quantity": 300, "target_qty": 300, "ref_price": 100.0,
             "stop_price": 90.0, "reason": "entry"}])
        self.assertEqual(pt.pending_orders()[0]["status"], "PENDING")
        self.assertEqual(len(json.loads(cloud.state["engine_pending_orders"])), 1)
        # EOD plan does not touch the book
        self.assertEqual(pt.cash, 100_000)
        self.assertEqual(pt.holdings(), {})

        rep = self._fill(pt, {"AAA": {"open": 102.0, "adv": 2e8}})
        self.assertEqual(len(rep["filled"]), 1)
        exp = simulate_fill("BUY", 300, 102.0, 2e8, D_SESSION, CostConfig())
        self.assertGreater(exp.impact_bps, CostConfig().spread_floor_bps)
        self.assertAlmostEqual(pt.cash, 100_000 - exp.value_inr - exp.cost_inr, places=4)
        lot = pt._positions[0]
        self.assertAlmostEqual(lot.entry_price, 102.0 * (1 + exp.impact_bps / 1e4), places=4)
        self.assertEqual(lot.opened_at, f"{D_SESSION.date()}T09:15:00+05:30")
        self.assertEqual(lot.stop_loss, 90.0)
        done = pt.pending_orders(status=None)[0]
        self.assertEqual((done["status"], done["fill_qty"]), ("FILLED", 300))
        self.assertAlmostEqual(done["costs_inr"], exp.cost_inr, places=4)
        self.assertEqual(pt.pending_orders(), [])
        self.assertEqual(json.loads(cloud.state["engine_pending_orders"]), [])

        # Exit next session: SELL at that open, one order-level statutory charge
        pt.queue_pending_orders(D_SESSION, [{"symbol": "AAA", "side": "SELL", "quantity": 300, "target_qty": 0,
                                             "ref_price": 105.0, "reason": "exit:rank_exit"}])
        nxt = D_SESSION + pd.offsets.BDay(1)
        cash0 = pt.cash
        rep = pt.fill_pending_orders(nxt, SESSIONS.append(pd.DatetimeIndex([nxt])),
                                     {"AAA": {"open": 104.0, "adv": 2e8}}, CostConfig())
        sell = simulate_fill("SELL", 300, 104.0, 2e8, nxt, CostConfig())
        self.assertEqual(len(rep["filled"]), 1)
        self.assertNotIn("AAA", pt.holdings())
        self.assertAlmostEqual(pt.cash - cash0, sell.value_inr * (1 - sell.impact_bps / 1e4) - sell.statutory_inr,
                               delta=0.05)

    def test_stale_kept_stop_and_sells_before_cash_limited_buys(self):
        pt = self.trader(initial_capital=20_000)
        pt.buy("OLD", 100, price=100.0)  # 10k position, cash ~10k
        cash_before = pt.cash
        pt.queue_pending_orders(SESSIONS[-4], [{"symbol": "STALE", "side": "BUY", "quantity": 10,
                                                "target_qty": 10, "ref_price": 100.0, "reason": "entry"}])
        pt.queue_pending_orders(D_PREV, [
            {"symbol": "OLD", "side": "SELL", "quantity": 100, "target_qty": 0, "ref_price": 100.0,
             "reason": "exit:rank_exit"},
            {"symbol": "NEW1", "side": "BUY", "quantity": 150, "target_qty": 150, "ref_price": 100.0,
             "reason": "entry"},
            {"symbol": "NEW2", "side": "BUY", "quantity": 150, "target_qty": 150, "ref_price": 100.0,
             "reason": "entry"},
            {"symbol": "GAP", "side": "BUY", "quantity": 100, "target_qty": 100, "ref_price": 100.0,
             "stop_price": 95.0, "reason": "entry"},
        ])
        pt.queue_pending_orders(D_SESSION, [{"symbol": "LATER", "side": "BUY", "quantity": 1,
                                             "target_qty": 1, "ref_price": 100.0, "reason": "entry"}])
        quotes = {s: {"open": 100.0, "adv": 5e8} for s in ("OLD", "NEW1", "NEW2", "STALE", "LATER")}
        quotes["GAP"] = {"open": 94.0, "adv": 5e8}
        with self.assertLogs("kite_connect.trading.paper_trader", level="WARNING") as logs:
            rep = self._fill(pt, quotes, min_trade_value_inr=5_000.0)
        self.assertTrue(any("stale" in m for m in logs.output))
        notes = {c["symbol"]: c["note"] for c in rep["cancelled"]}
        self.assertIn("stale", notes["STALE"])
        self.assertIn("below stop", notes["GAP"])
        self.assertEqual([o["symbol"] for o in pt.pending_orders()], ["LATER"])
        held = pt.holdings()
        self.assertNotIn("OLD", held)
        # Sell proceeds funded the buys (30k wanted > ~20k available): scaled, never negative cash
        self.assertGreater(held["NEW1"]["quantity"], 50)
        self.assertLess(held["NEW1"]["quantity"] + held["NEW2"]["quantity"], 300)
        self.assertGreaterEqual(pt.cash, 0.0)
        self.assertLess(pt.cash, 200.0)
        self.assertGreater(cash_before, 9_000)

    def test_pending_survives_a_fresh_runner(self):
        cloud = FakeCloud()
        pt = self.trader(initial_capital=50_000, cloud=cloud)
        pt.buy("KEEP", 10, price=100.0)
        pt.queue_pending_orders(D_PREV, [{"symbol": "AAA", "side": "BUY", "quantity": 5, "target_qty": 5,
                                          "ref_price": 100.0, "stop_price": 80.0, "reason": "entry"}])
        pt.set_engine_last_session(D_PREV)
        # Fresh disk: restore from the cloud state
        cloud.positions = [dict(p.to_dict()) for p in pt._positions]
        os.remove(self.pt_mod._DB_PATH)
        pt2 = self.trader(initial_capital=1, cloud=cloud)
        self.assertEqual(pt2.restored_from, "cloud")
        pend = pt2.pending_orders()
        self.assertEqual([(o["symbol"], o["target_qty"], o["stop_price"]) for o in pend], [("AAA", 5, 80.0)])
        self.assertEqual(pt2.engine_last_session(), D_PREV.date())

    def test_lot_filled_at_open_is_exposed_to_that_sessions_low(self):
        pt = self.trader(initial_capital=100_000)
        pt.queue_pending_orders(D_PREV, [{"symbol": "AAA", "side": "BUY", "quantity": 100, "target_qty": 100,
                                          "ref_price": 100.0, "stop_price": 95.0, "reason": "entry"}])
        self._fill(pt, {"AAA": {"open": 100.0, "adv": 5e8}})
        ev = pt.simulate_gtt_stops({"AAA": {"date": D_SESSION, "open": 100.0, "low": 94.0, "close": 96.0}})
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0]["exit"], 95.0)  # min(open, stop)

    def test_engine_stop_uses_backtest_impact_not_paper_slippage(self):
        from nse_engine.config import CostConfig
        from nse_engine.costs import impact_bps
        pt = self.trader(initial_capital=100_000, slippage_bps=50.0)
        pt.buy("AAA", 100, price=100.0, stop_loss=95.0)
        pt._positions[0].opened_at = f"{D_PREV.date()}T09:15:00+05:30"
        ev = pt.simulate_gtt_stops({"AAA": {"date": D_SESSION, "open": 93.0, "low": 92.0, "close": 94.0,
                                            "adv": 5e7}}, cost_config=CostConfig())
        bps = impact_bps(9_300.0, 5e7, CostConfig())
        self.assertAlmostEqual(ev[0]["exit"], 93.0 * (1 - bps / 1e4), places=3)  # gap: fills at the open
        self.assertEqual(ev[0]["type"], "PAPER_GTT_SL_GAP")


# ─────────────────────────────────────────────────────────────
# 3. run_paper_session end to end (synthetic store)
# ─────────────────────────────────────────────────────────────

def _data_with_open(prices, open_last, end="2024-03-28", n=30):
    from nse_engine.types import MarketData
    dates = pd.bdate_range(end=end, periods=n)
    close = pd.DataFrame({s: [p] * n for s, p in prices.items()}, index=dates, dtype=float)
    opn = close.copy()
    for s, o in open_last.items():
        opn.loc[dates[-1], s] = o
    return MarketData(dates=dates, open=opn, high=pd.concat([close, opn]).groupby(level=0).max(),
                      low=pd.concat([close, opn]).groupby(level=0).min(), close=close,
                      volume=close * 0 + 1e6, value=close * 1e6, index_close=pd.DataFrame(index=dates))


class TestPaperSession(PaperTestBase):
    def test_eod_plan_then_next_open_fill(self):
        from nse_engine.config import EngineConfig
        from nse_engine.costs import simulate_fill
        from nse_engine.types import TargetPortfolio
        from kite_connect.trading.nse_engine_executor import EngineExecutor

        data = _data_with_open({"AAA": 100.0}, {"AAA": 103.0})
        d1, d2 = data.dates[-2], data.dates[-1]
        pt = self.trader(initial_capital=100_000)

        def target_fn(d, cfg, as_of, holdings=None, cache=None, *, equity=None, stopped_out=None):
            return TargetPortfolio(as_of=as_of, weights={"AAA": 0.5}, stops={"AAA": 60.0})

        ex = EngineExecutor(kite=None, paper=True, config=EngineConfig(), paper_trader=pt, target_fn=target_fn,
                            data_loader=lambda store, start, end, **kw: data.until(pd.Timestamp(end)))
        r1 = ex.run_paper_session(as_of=d1)
        self.assertEqual(r1["session"], str(d1.date()))
        self.assertEqual([(o["symbol"], o["quantity"]) for o in pt.pending_orders()], [("AAA", 500)])
        self.assertEqual(pt.holdings(), {})  # nothing filled at the close

        r2 = ex.run_paper_session(as_of=d2)
        self.assertEqual(len(r2["fills"]["filled"]), 1)
        lot = pt._positions[0]
        fill = simulate_fill("BUY", 500, 103.0, 1e8, d2, EngineConfig().costs)
        self.assertAlmostEqual(lot.entry_price, 103.0 * (1 + fill.impact_bps / 1e4), places=4)
        self.assertEqual(lot.stop_loss, 60.0)
        self.assertEqual(pt._price_overrides["AAA"], 100.0)  # marked at the close
        self.assertEqual(pt.engine_last_session(), d2.date())
        # Re-run on the same session: no second fill, only re-planning
        r3 = ex.run_paper_session(as_of=d2)
        self.assertIsNone(r3["fills"])
        self.assertEqual(sum(p.quantity for p in pt._positions if p.is_open), 500)


# ─────────────────────────────────────────────────────────────
# 4. Distribution-shift multiplier in the executor
# ─────────────────────────────────────────────────────────────

class TestShiftMultiplier(PaperTestBase):
    def _state(self, updated_at, mult=0.5, verdict="regime_break"):
        p = Path(self.tmp) / "shift_state.json"
        p.write_text(json.dumps({"verdict": verdict, "position_size_multiplier": mult,
                                 "updated_at": updated_at}))
        return p

    def _plan(self, state_path):
        from nse_engine.config import EngineConfig
        from nse_engine.types import TargetPortfolio
        from kite_connect.trading.nse_engine_executor import EngineExecutor
        from tests.test_live_execution import _market_data

        data = _market_data({"NEW": 100.0, "UP": 100.0, "HOLD": 100.0, "EXIT": 100.0, "DOWN": 100.0})
        holdings = {"UP": {"quantity": 100, "avg_price": 100.0}, "HOLD": {"quantity": 100, "avg_price": 100.0},
                    "EXIT": {"quantity": 100, "avg_price": 100.0}, "DOWN": {"quantity": 200, "avg_price": 100.0}}
        cash = 50_000.0  # equity 100k

        def target_fn(d, cfg, as_of, holdings=None, cache=None, *, equity=None, stopped_out=None):
            return TargetPortfolio(as_of=as_of, weights={"NEW": 0.20, "UP": 0.20, "HOLD": 0.10, "DOWN": 0.10},
                                   exits={"EXIT": "stop"}, stops={"NEW": 90.0})

        ex = EngineExecutor(kite=None, paper=True, config=EngineConfig(), target_fn=target_fn,
                            data_loader=lambda *a, **k: data, holdings_fn=lambda: (holdings, cash),
                            shift_state_path=str(state_path))
        return ex.plan(as_of="2024-03-28")

    def test_fresh_state_scales_new_risk_only(self):
        plan = self._plan(self._state("2024-03-26T19:40:00+05:30"))
        self.assertEqual(plan.shift_multiplier, 0.5)
        by = {o.symbol: o for o in plan.orders}
        self.assertEqual((by["NEW"].side, by["NEW"].quantity), ("BUY", 100))   # 0.20 -> 0.10
        self.assertNotIn("UP", by)                                            # max(0.10, 0.20*0.5) = 0.10
        self.assertNotIn("HOLD", by)                                           # unchanged weight
        self.assertEqual((by["EXIT"].side, by["EXIT"].quantity), ("SELL", 100))  # exits unaffected
        self.assertEqual((by["DOWN"].side, by["DOWN"].quantity), ("SELL", 100))  # reductions unaffected
        self.assertTrue(any("multiplier 0.50" in n for n in plan.notes))
        self.assertIn("NEW", {s.symbol for s in plan.stop_instructions})

    def test_stale_malformed_and_future_states_are_ignored(self):
        from kite_connect.trading.nse_engine_executor import load_shift_multiplier
        cal = pd.bdate_range(end="2024-03-28", periods=30)
        with self.assertLogs("kite_connect.trading.nse_engine_executor", level="WARNING"):
            plan = self._plan(self._state("2024-03-18T19:40:00+05:30"))  # 8 sessions old
        self.assertEqual(plan.shift_multiplier, 1.0)
        self.assertIn("stale", plan.shift_reason)
        by = {o.symbol: o for o in plan.orders}
        self.assertEqual(by["NEW"].quantity, 200)
        self.assertEqual(by["UP"].quantity, 100)
        # exactly 5 sessions old is still fresh
        self.assertEqual(load_shift_multiplier("2024-03-28", cal, self._state("2024-03-21T20:00:00+05:30"))[0], 0.5)
        bad = Path(self.tmp) / "bad.json"
        bad.write_text("{oops")
        with self.assertLogs("kite_connect.trading.nse_engine_executor", level="WARNING"):
            self.assertEqual(load_shift_multiplier("2024-03-28", cal, bad)[0], 1.0)
        with self.assertLogs("kite_connect.trading.nse_engine_executor", level="WARNING"):
            self.assertEqual(load_shift_multiplier("2024-03-28", cal, self._state("2024-03-27", mult=1.5))[0], 1.0)
        self.assertEqual(load_shift_multiplier("2024-03-28", cal, self._state("2024-04-02"))[0], 1.0)
        self.assertEqual(load_shift_multiplier("2024-03-28", cal, Path(self.tmp) / "absent.json")[0], 1.0)


# ─────────────────────────────────────────────────────────────
# 5. Reality-gap alerts
# ─────────────────────────────────────────────────────────────

class TestRealityGap(unittest.TestCase):
    def test_thresholds(self):
        from kite_connect.trading.paper_trader import reality_gap_alerts
        base = {"reference_mode": "same_period", "n_live": 30, "tracking_error_annual": 0.05,
                "mean_daily_gap": -0.0001}
        clean = {k: v for k, v in os.environ.items() if not k.startswith("CENTURION_SHIFT_MAX")}
        with mock.patch.dict(os.environ, clean, clear=True):
            self.assertEqual(reality_gap_alerts(base), [])
            self.assertEqual(len(reality_gap_alerts({**base, "tracking_error_annual": 0.081})), 1)
            self.assertEqual(reality_gap_alerts({**base, "tracking_error_annual": 0.079}), [])
            self.assertEqual(len(reality_gap_alerts({**base, "mean_daily_gap": -0.00031})), 1)
            self.assertEqual(reality_gap_alerts({**base, "mean_daily_gap": -0.00029}), [])
            both = {**base, "tracking_error_annual": 0.045, "mean_daily_gap": -0.00028}
            self.assertEqual(reality_gap_alerts(both), [])  # the simulated -2.8 bp/day case at default 3 bp
            self.assertEqual(reality_gap_alerts({**base, "n_live": 29, "tracking_error_annual": 0.5}), [])
            self.assertEqual(reality_gap_alerts({**base, "reference_mode": "trailing_history",
                                                 "tracking_error_annual": 0.5}), [])
        with mock.patch.dict(os.environ, {"CENTURION_SHIFT_MAX_TRACKING_ERROR": "0.04",
                                          "CENTURION_SHIFT_MAX_DAILY_GAP_BPS": "2"}):
            self.assertEqual(len(reality_gap_alerts(both)), 2)

    def test_gap_forces_drifting_multiplier_and_alert(self):
        from kite_connect.trading import paper_trader as pt_mod
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        db = tmp / "paper_trades.sqlite3"
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE daily_snapshots (date TEXT PRIMARY KEY, equity REAL)")
        idx = pd.bdate_range("2026-01-01", periods=41)
        conn.executemany("INSERT INTO daily_snapshots VALUES (?, ?)",
                         [(d.date().isoformat(), 100_000 + i) for i, d in enumerate(idx)])
        conn.commit()
        conn.close()
        report = {"reference_mode": "same_period", "reference_source": "csv", "verdict": "stable",
                  "calibrated_verdict": "stable", "effective_verdict": "stable", "n_live": 40,
                  "wasserstein": 0.001, "kl_divergence": 0.05, "tracking_error_annual": 0.045,
                  "mean_daily_gap": -0.0005}
        trader = pt_mod.PaperTrader.__new__(pt_mod.PaperTrader)
        alert = mock.MagicMock()
        with mock.patch.object(pt_mod, "_DB_PATH", db), \
                mock.patch("services.distribution_shift.compare_live_to_backtest", return_value=dict(report)), \
                mock.patch("services.notifications.manager.NotificationManager", return_value=alert):
            result = trader._run_distribution_shift()
        self.assertEqual(result["position_verdict"], "drifting")
        self.assertEqual(len(result["reality_gap_alerts"]), 1)
        state = json.loads((tmp / "distribution_shift_state.json").read_text())
        self.assertEqual(state["position_size_multiplier"], 0.75)
        self.assertEqual(state["reality_gap_alerts"], result["reality_gap_alerts"])
        alert.send_alert.assert_called_once()
        self.assertIn("REALITY GAP", alert.send_alert.call_args.kwargs["subject"])


# ─────────────────────────────────────────────────────────────
# 6. GitHub Actions workflow
# ─────────────────────────────────────────────────────────────

class TestWorkflow(unittest.TestCase):
    def setUp(self):
        import yaml
        self.text = (ROOT / ".github" / "workflows" / "paper-trade-cron.yml").read_text()
        self.doc = yaml.safe_load(self.text)

    def test_structure(self):
        doc = self.doc
        on = doc.get("on", doc.get(True))  # YAML 1.1 reads the bare key `on` as True
        crons = [c["cron"] for c in on["schedule"]]
        self.assertIn("0 14 * * 1-5", crons)
        self.assertIn("0 2 * * 6", crons)
        self.assertIn("full_bootstrap", on["workflow_dispatch"]["inputs"])
        self.assertEqual(doc["concurrency"]["group"], "paper-trade-book")

        legacy, engine = doc["jobs"]["paper-trade"], doc["jobs"]["nse-engine-paper"]
        self.assertIn("vars.CENTURION_NSE_ENGINE != 'true'", legacy["if"])
        self.assertIn("vars.CENTURION_NSE_ENGINE == 'true'", engine["if"])
        for cron in ("0 14 * * 1-5", "0 2 * * 6"):
            self.assertIn(cron, legacy["if"] + engine["if"])
        self.assertIn("python cloud_paper_runner.py", legacy["steps"][-1]["run"])
        self.assertLessEqual(engine["timeout-minutes"], 360)

        steps = engine["steps"]
        uses = [s.get("uses", "") for s in steps]
        restore = steps[uses.index("actions/cache/restore@v4")]
        save = steps[uses.index("actions/cache/save@v4")]
        self.assertLess(uses.index("actions/cache/restore@v4"), uses.index("actions/cache/save@v4"))
        self.assertIn("data/nse_engine/archive", restore["with"]["path"])
        self.assertEqual(restore["with"]["key"], save["with"]["key"])
        self.assertIn("github.run_id", save["with"]["key"])
        self.assertTrue(restore["with"]["restore-keys"].strip())
        self.assertEqual(save["if"], "always()")

        runs = "\n".join(s.get("run", "") for s in steps)
        self.assertIn("python -m nse_engine.data.archive --start", runs)
        self.assertIn("build_store(", runs)
        self.assertIn("years=", runs)
        self.assertIn("shift-reference", runs)
        self.assertIn("nse_engine.deployment write-engine-config", runs)
        self.assertIn("python cloud_paper_runner.py --engine", runs)
        order = [s["name"] for s in steps]
        self.assertLess(order.index("Sync NSE archive"), order.index("Build NSE store (changed years only)"))
        self.assertLess(order.index("Regenerate same-period shift reference"),
                        order.index("Run NSE engine paper session"))

    def test_archive_cli_flags_exist(self):
        from nse_engine.data import archive
        with mock.patch.object(archive.BhavcopyArchive, "sync", return_value={}) as sync, \
                mock.patch.object(archive.BhavcopyArchive, "sync_reference", return_value={}), \
                mock.patch("signal.signal"), mock.patch("logging.basicConfig"):
            tmp = tempfile.mkdtemp()
            self.addCleanup(shutil.rmtree, tmp, True)
            rc = archive.main(["--start", "2026-09-01", "--end", "2026-09-14", "--root", tmp, "--rps", "2"])
        self.assertEqual(rc, 0)
        self.assertEqual(sync.call_args.args[:2], ("2026-09-01", "2026-09-14"))


if __name__ == "__main__":
    unittest.main()
