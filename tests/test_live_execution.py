"""
Live / paper execution tests (stdlib unittest, mocked Kite — no network).

Run:  python -m unittest tests.test_live_execution -v
"""

from __future__ import annotations

import inspect
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

IST = timezone(timedelta(hours=5, minutes=30))


# ─────────────────────────────────────────────────────────────
# Fake Kite
# ─────────────────────────────────────────────────────────────

class FakeKite:
    """Minimal in-memory Kite: holdings, positions, LTP, orders and GTTs."""

    def __init__(self, holdings=None, ltp=None, positions=None):
        self._holdings = holdings or {}
        self._ltp = ltp or {}
        self._positions = positions or []
        self.gtts = {}
        self._next = 1000
        self.placed_orders = []
        self.calls = []

    # portfolio
    def holdings(self):
        return [{"tradingsymbol": s, "exchange": "NSE", "quantity": q, "t1_quantity": 0,
                 "last_price": self._ltp.get(s, 100.0), "average_price": 90.0}
                for s, q in self._holdings.items()]

    def positions(self):
        return {"net": list(self._positions), "day": []}

    def margins(self, segment=None):
        return {"net": 50_000.0, "available": {"cash": 50_000.0}}

    def ltp(self, keys):
        return {k: {"last_price": self._ltp.get(k.split(":", 1)[1], 100.0)} for k in keys}

    # orders
    def place_order(self, **params):
        self.calls.append(("place_order", params))
        self._next += 1
        self.placed_orders.append(params)
        return str(self._next)

    def orders(self):
        return []

    def order_history(self, order_id):
        return [{"status": "COMPLETE", "average_price": 100.0, "filled_quantity": 1}]

    # GTTs
    def _gtt(self, gid, tradingsymbol, exchange, trigger_values, orders, status="active"):
        return {"id": gid, "type": "single", "status": status,
                "condition": {"exchange": exchange, "tradingsymbol": tradingsymbol,
                              "trigger_values": trigger_values},
                "orders": [dict(o, tradingsymbol=tradingsymbol, exchange=exchange) for o in orders]}

    def get_gtts(self):
        return list(self.gtts.values())

    def get_gtt(self, trigger_id):
        return self.gtts[trigger_id]

    def place_gtt(self, trigger_type, tradingsymbol, exchange, trigger_values, last_price, orders):
        self.calls.append(("place_gtt", tradingsymbol, trigger_values, orders[0]["quantity"]))
        self._next += 1
        self.gtts[self._next] = self._gtt(self._next, tradingsymbol, exchange, trigger_values, orders)
        return {"trigger_id": self._next}

    def modify_gtt(self, trigger_id, trigger_type, tradingsymbol, exchange, trigger_values, last_price, orders):
        self.calls.append(("modify_gtt", trigger_id, trigger_values, orders[0]["quantity"]))
        self.gtts[trigger_id] = self._gtt(trigger_id, tradingsymbol, exchange, trigger_values, orders)
        return {"trigger_id": trigger_id}

    def delete_gtt(self, trigger_id):
        self.calls.append(("delete_gtt", trigger_id))
        self.gtts.pop(trigger_id)
        return {"trigger_id": trigger_id}


def _no_kill_switch_env():
    return mock.patch.dict(os.environ, {"CENTURION_KILL_SWITCH": "false"})


# ─────────────────────────────────────────────────────────────
# 1. GTT stops
# ─────────────────────────────────────────────────────────────

class TestGttStops(unittest.TestCase):
    def setUp(self):
        from kite_connect.trading import gtt_stops
        self.g = gtt_stops
        p1 = mock.patch.object(gtt_stops, "_notify", lambda *a, **k: None)
        p2 = mock.patch.object(gtt_stops, "_kill_switch_active", lambda: False)
        p1.start(); p2.start()
        self.addCleanup(p1.stop); self.addCleanup(p2.stop)

    def test_tick_rounding_and_limit(self):
        self.assertEqual(self.g.round_to_tick(101.23), 101.25)
        self.assertEqual(self.g.round_to_tick(101.23, mode="down"), 101.2)
        # 1% below 100.00 = 99.00; below 101.27 -> 100.2573 -> 100.25
        self.assertEqual(self.g.stop_limit_price(100.0, 1.0), 99.0)
        self.assertEqual(self.g.stop_limit_price(101.27, 1.0), 100.25)

    def test_place_is_idempotent_and_modifies(self):
        kite = FakeKite(holdings={"INFY": 10}, ltp={"INFY": 1500.0})
        r1 = self.g.place_or_update_stop_gtt(kite, "INFY", 10, 1400.02)
        self.assertTrue(r1["success"]); self.assertEqual(r1["action"], "placed")
        self.assertEqual(r1["trigger"], 1400.0)
        r2 = self.g.place_or_update_stop_gtt(kite, "INFY", 10, 1400.0)
        self.assertEqual(r2["action"], "unchanged")
        r3 = self.g.place_or_update_stop_gtt(kite, "INFY", 10, 1420.0)
        self.assertEqual(r3["action"], "modified")
        self.assertEqual(r3["trigger_id"], r1["trigger_id"])
        self.assertEqual(len(kite.gtts), 1)
        g = list(kite.gtts.values())[0]
        self.assertEqual(g["condition"]["trigger_values"], [1420.0])
        self.assertEqual(g["orders"][0]["price"], 1405.8)  # 1% below, tick down
        self.assertEqual(g["orders"][0]["product"], "CNC")
        self.assertEqual(g["orders"][0]["transaction_type"], "SELL")

    def test_duplicates_are_removed(self):
        kite = FakeKite(holdings={"TCS": 5}, ltp={"TCS": 4000.0})
        self.g.place_or_update_stop_gtt(kite, "TCS", 5, 3800.0)
        # A second GTT created outside the helper
        kite.place_gtt("single", "TCS", "NSE", [3700.0], 4000.0,
                       [{"transaction_type": "SELL", "quantity": 5, "order_type": "LIMIT",
                         "product": "CNC", "price": 3663.0}])
        self.assertEqual(len(kite.gtts), 2)
        res = self.g.place_or_update_stop_gtt(kite, "TCS", 5, 3800.0)
        self.assertTrue(res["success"])
        self.assertEqual(len(kite.gtts), 1)

    def test_breached_stop_not_placed(self):
        kite = FakeKite(holdings={"SBIN": 10}, ltp={"SBIN": 500.0})
        res = self.g.place_or_update_stop_gtt(kite, "SBIN", 10, 510.0)
        self.assertFalse(res["success"])
        self.assertEqual(res["error"], "stop_breached")
        self.assertEqual(kite.gtts, {})

    def test_reconciliation(self):
        kite = FakeKite(holdings={"INFY": 10, "TCS": 4, "HDFC": 7},
                        ltp={"INFY": 1500.0, "TCS": 4000.0, "HDFC": 1600.0, "WIPRO": 400.0})
        # TCS: existing stop at wrong quantity; WIPRO: orphan; HDFC: no stop, no level
        self.g.place_or_update_stop_gtt(kite, "TCS", 10, 3800.0)
        kite._holdings["WIPRO"] = 3
        self.g.place_or_update_stop_gtt(kite, "WIPRO", 3, 380.0)
        del kite._holdings["WIPRO"]
        report = self.g.reconcile_stop_gtts(kite, stops={"INFY": 1400.0})
        syms = lambda k: sorted(x["symbol"] if isinstance(x, dict) else x for x in report[k])  # noqa: E731
        self.assertEqual(syms("placed"), ["INFY"])
        self.assertEqual(syms("modified"), ["TCS"])
        self.assertEqual(syms("deleted"), ["WIPRO"])
        self.assertEqual(syms("missing_stop"), ["HDFC"])
        active = self.g.list_stop_gtts(kite)
        by = {g["symbol"]: g for g in active}
        self.assertEqual(set(by), {"INFY", "TCS"})
        self.assertEqual(by["TCS"]["quantity"], 4)
        self.assertEqual(by["TCS"]["trigger"], 3800.0)
        # Second pass: nothing to do
        report2 = self.g.reconcile_stop_gtts(kite, stops={"INFY": 1400.0})
        self.assertEqual(len(report2["unchanged"]), 2)
        self.assertFalse(report2["placed"] or report2["modified"] or report2["deleted"])

    def test_kill_switch_caps_gtt_quantity_to_holding(self):
        kite = FakeKite(holdings={"INFY": 6}, ltp={"INFY": 1500.0})
        with mock.patch.object(self.g, "_kill_switch_active", lambda: True):
            res = self.g.place_or_update_stop_gtt(kite, "INFY", 10, 1400.0)
            self.assertTrue(res["success"])
            self.assertEqual(res["quantity"], 6)
            res2 = self.g.place_or_update_stop_gtt(kite, "NOTHELD", 1, 10.0, last_price=20.0)
            self.assertFalse(res2["success"])

    def test_trade_monitor_uses_gtt_for_cnc(self):
        from kite_connect.trading import trade_monitor as tm
        kite = FakeKite(holdings={"INFY": 10}, ltp={"INFY": 1500.0})
        with mock.patch.object(tm.TradeMonitor, "_init_state_db", lambda self: None), \
             mock.patch.object(tm.TradeMonitor, "_restore_state", lambda self: None), \
             mock.patch.object(tm.TradeMonitor, "_persist_state", lambda self: None):
            mon = tm.TradeMonitor(kite=kite)
            trade = tm.MonitoredTrade(symbol="INFY", side="BUY", quantity=10, entry_price=1500.0,
                                      stop_loss=1400.0, target_price=1800.0, entry_order_id="1",
                                      entry_filled=True)
            mon._place_sl_for_trade(trade)
            self.assertIsNotNone(trade.sl_gtt_id)
            self.assertIsNone(trade.sl_order_id)
            self.assertFalse(any(c[0] == "place_order" for c in kite.calls))
            mon._trades["1"] = trade
            # Reconcile after a partial sale: quantity follows the holding
            kite._holdings["INFY"] = 7
            report = mon.reconcile_gtt_stops()
            self.assertEqual(len(report["modified"]), 1)
            self.assertEqual(self.g.list_stop_gtts(kite)[0]["quantity"], 7)


# ─────────────────────────────────────────────────────────────
# 2. Kill switch: reduce-only exits
# ─────────────────────────────────────────────────────────────

class TestKillSwitch(unittest.TestCase):
    def setUp(self):
        from kite_connect.trading import order_service
        self.os = order_service
        for name, val in (("_persist_to_db", lambda *a, **k: None),
                          ("_send_order_email", lambda *a, **k: None),
                          ("_is_nse_market_open", lambda: True)):
            p = mock.patch.object(order_service, name, val)
            p.start(); self.addCleanup(p.stop)
        p = mock.patch.object(order_service.time, "sleep", lambda s: None)
        p.start(); self.addCleanup(p.stop)
        if order_service._order_circuit:
            order_service._order_circuit.reset()

    def test_signature_has_is_exit(self):
        params = inspect.signature(self.os.place_order).parameters
        self.assertIn("is_exit", params)
        self.assertFalse(params["is_exit"].default)

    def test_blocks_buy_allows_reduce_only_sell(self):
        kite = FakeKite(holdings={"INFY": 10})
        with mock.patch.dict(os.environ, {"CENTURION_KILL_SWITCH": "true"}):
            buy = self.os.place_order(kite, "INFY", "NSE", "BUY", 1, order_type="LIMIT", price=100.0)
            self.assertFalse(buy["success"])
            buy_exit = self.os.place_order(kite, "INFY", "NSE", "BUY", 1, order_type="LIMIT",
                                           price=100.0, is_exit=True)
            self.assertFalse(buy_exit["success"])
            sell_plain = self.os.place_order(kite, "INFY", "NSE", "SELL", 1, order_type="LIMIT", price=100.0)
            self.assertFalse(sell_plain["success"])
            self.assertEqual(kite.placed_orders, [])
            sell_exit = self.os.place_order(kite, "INFY", "NSE", "SELL", 10, order_type="LIMIT",
                                            price=100.0, is_exit=True)
            self.assertTrue(sell_exit["success"], sell_exit)
            self.assertEqual(len(kite.placed_orders), 1)
            self.assertEqual(kite.placed_orders[0]["transaction_type"], "SELL")

    def test_non_cnc_exit_cannot_flip_short(self):
        kite = FakeKite(positions=[{"tradingsymbol": "NIFTYFUT", "exchange": "NFO",
                                    "product": "NRML", "quantity": 50}])
        with mock.patch.dict(os.environ, {"CENTURION_KILL_SWITCH": "true"}):
            too_much = self.os.place_order(kite, "NIFTYFUT", "NFO", "SELL", 100, product="NRML",
                                           order_type="LIMIT", price=1.0, is_exit=True)
            self.assertFalse(too_much["success"])
            ok = self.os.place_order(kite, "NIFTYFUT", "NFO", "SELL", 50, product="NRML",
                                     order_type="LIMIT", price=1.0, is_exit=True)
            self.assertTrue(ok["success"])

    def test_account_equity_and_fallback(self):
        kite = FakeKite(holdings={"INFY": 10}, ltp={"INFY": 1500.0})
        eq, src = self.os.get_account_equity(kite, fallback=500_000)
        self.assertEqual(src, "kite")
        self.assertAlmostEqual(eq, 50_000.0 + 15_000.0)
        broken = mock.Mock()
        broken.margins.side_effect = RuntimeError("down")
        eq2, src2 = self.os.get_account_equity(broken, fallback=500_000)
        self.assertEqual((eq2, src2), (500_000.0, "config_fallback"))


# ─────────────────────────────────────────────────────────────
# 3. Session-based freshness
# ─────────────────────────────────────────────────────────────

class TestSessionFreshness(unittest.TestCase):
    def setUp(self):
        from services import carver_pipeline as cp
        self.cp = cp
        p = mock.patch.dict(os.environ, {"CENTURION_NSE_HOLIDAYS": ""})
        p.start(); self.addCleanup(p.stop)

    def test_before_and_after_close(self):
        # Wednesday 2025-06-11 (not a holiday)
        self.assertEqual(self.cp.last_completed_nse_session(datetime(2025, 6, 11, 10, 0, tzinfo=IST)),
                         pd.Timestamp("2025-06-10").date())
        self.assertEqual(self.cp.last_completed_nse_session(datetime(2025, 6, 11, 15, 29, tzinfo=IST)),
                         pd.Timestamp("2025-06-10").date())
        self.assertEqual(self.cp.last_completed_nse_session(datetime(2025, 6, 11, 15, 30, tzinfo=IST)),
                         pd.Timestamp("2025-06-11").date())
        # tz-aware UTC input: 10:30 UTC = 16:00 IST
        self.assertEqual(self.cp.last_completed_nse_session(
            datetime(2025, 6, 11, 10, 30, tzinfo=timezone.utc)), pd.Timestamp("2025-06-11").date())

    def test_weekend_and_holiday_roll_back(self):
        # Saturday / Sunday / Monday morning -> Friday 2025-06-13
        for dt in (datetime(2025, 6, 14, 12, 0, tzinfo=IST), datetime(2025, 6, 15, 20, 0, tzinfo=IST),
                   datetime(2025, 6, 16, 9, 0, tzinfo=IST)):
            self.assertEqual(self.cp.last_completed_nse_session(dt), pd.Timestamp("2025-06-13").date())
        # Day after Independence Day holiday (Fri 2025-08-15), Monday morning -> Thu 08-14
        self.assertEqual(self.cp.last_completed_nse_session(datetime(2025, 8, 18, 9, 0, tzinfo=IST)),
                         pd.Timestamp("2025-08-14").date())

    def _df(self, last):
        idx = pd.bdate_range(end=last, periods=5)
        return pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0}, index=idx)

    def test_gate_keeps_midnight_daily_bars_and_drops_stale(self):
        now = datetime(2025, 6, 11, 10, 0, tzinfo=IST)  # expected session 06-10
        cache = {"FRESH": self._df("2025-06-10"), "TODAY": self._df("2025-06-11"),
                 "STALE": self._df("2025-06-05")}
        fresh, info = self.cp.apply_session_freshness_gate(cache, now=now)
        self.assertEqual(sorted(fresh), ["FRESH", "TODAY"])
        self.assertEqual([s for s, _ in info["dropped"]], ["STALE"])
        self.assertFalse(info["consensus_fallback"])
        # tz-aware index is converted to IST dates
        tz_df = self._df("2025-06-10")
        tz_df.index = tz_df.index.tz_localize("Asia/Kolkata")
        fresh2, _ = self.cp.apply_session_freshness_gate({"TZ": tz_df}, now=now)
        self.assertIn("TZ", fresh2)

    def test_consensus_fallback_for_unpublished_session(self):
        now = datetime(2025, 6, 11, 16, 0, tzinfo=IST)  # expects 06-11, vendor only has 06-10
        cache = {"A": self._df("2025-06-10"), "B": self._df("2025-06-10"), "C": self._df("2025-06-02")}
        fresh, info = self.cp.apply_session_freshness_gate(cache, now=now)
        self.assertTrue(info["consensus_fallback"])
        self.assertEqual(sorted(fresh), ["A", "B"])

    def test_rank_exits(self):
        forecasts = {f"S{i}": 20.0 - i * 0.4 for i in range(45)}
        forecasts["NEG"] = -1.0
        exits = self.cp.compute_rank_exits(forecasts, {"S0": 5, "S44": 5, "NEG": 5, "NODATA": 5},
                                           exit_rank=40)
        self.assertIn("S44", exits)
        self.assertIn("NEG", exits)
        self.assertNotIn("S0", exits)
        self.assertNotIn("NODATA", exits)


# ─────────────────────────────────────────────────────────────
# 4. Paper trader: cloud restore, gap-through stops, costs
# ─────────────────────────────────────────────────────────────

class FakeCloud:
    def __init__(self, state=None, positions=None, snapshots=None):
        self.state = dict(state or {})
        self.positions = list(positions or [])
        self.snapshots = list(snapshots or [])
        self.synced = []

    def read_state(self):
        return dict(self.state)

    def read_positions(self):
        return pd.DataFrame(self.positions)

    def read_snapshots(self):
        return pd.DataFrame(self.snapshots)

    def sync_state(self, values):
        self.state.update({k: str(v) for k, v in values.items()})
        return True

    def sync_position(self, pos):
        self.synced.append(pos)
        return True

    def sync_stop_loss(self, *a):
        return True

    def sync_snapshot(self, snap):
        return True


class PaperTestBase(unittest.TestCase):
    def setUp(self):
        from kite_connect.trading import paper_trader
        self.pt_mod = paper_trader
        self.tmp = tempfile.mkdtemp()
        p = mock.patch.object(paper_trader, "_DB_PATH",
                              paper_trader.Path(self.tmp) / "paper_trades.sqlite3")
        p.start(); self.addCleanup(p.stop)
        env = {k: v for k, v in os.environ.items() if k != "CENTURION_DATABASE_URL"}
        p2 = mock.patch.dict(os.environ, env, clear=True)
        p2.start(); self.addCleanup(p2.stop)
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def trader(self, **kw):
        kw.setdefault("slippage_bps", 0.0)
        kw.setdefault("cloud", FakeCloud())
        return self.pt_mod.PaperTrader(kite=None, **kw)


class TestPaperTrader(PaperTestBase):
    def test_restore_from_cloud(self):
        cloud = FakeCloud(
            state={"cash": "61234.5", "initial_capital": "100000", "epoch": "2025-06-01T00:00:00+00:00"},
            positions=[
                {"symbol": "INFY", "side": "BUY", "quantity": 10, "entry_price": 1500.0,
                 "stop_loss": 1420.0, "target_price": 1800.0, "opened_at": "2025-06-02T09:20:00+05:30",
                 "closed_at": "", "exit_price": 0, "exit_reason": "", "pnl": 0, "pnl_pct": 0, "is_open": True},
                {"symbol": "OLD", "side": "BUY", "quantity": 1, "entry_price": 10.0, "stop_loss": 9.0,
                 "target_price": 12.0, "opened_at": "2025-05-01T09:20:00+05:30", "closed_at": "",
                 "exit_price": 0, "exit_reason": "", "pnl": 0, "pnl_pct": 0, "is_open": True},
            ],
            snapshots=[{"date": "2025-06-02", "equity": 99000.0, "cash": 61234.5},
                       {"date": "2025-05-20", "equity": 1.0, "cash": 1.0}],
        )
        pt = self.trader(cloud=cloud, initial_capital=100_000)
        self.assertEqual(pt.restored_from, "cloud")
        self.assertAlmostEqual(pt.cash, 61234.5)
        self.assertEqual(pt.initial_capital, 100_000)
        open_syms = [p.symbol for p in pt._positions if p.is_open]
        self.assertEqual(open_syms, ["INFY"])  # legacy row before epoch ignored
        self.assertEqual(pt._positions[0].stop_loss, 1420.0)
        self.assertEqual(pt._positions[0].opened_at, "2025-06-02T09:20:00+05:30")
        # Next instance restores from local SQLite, not cloud
        pt2 = self.trader(cloud=FakeCloud(), initial_capital=5)
        self.assertEqual(pt2.restored_from, "local")
        self.assertAlmostEqual(pt2.cash, 61234.5)
        self.assertEqual(pt2.initial_capital, 100_000)

    def test_first_run_uses_initial_capital_and_persists_state(self):
        cloud = FakeCloud(positions=[{"symbol": "LEGACY", "side": "BUY", "quantity": 1, "entry_price": 1,
                                      "stop_loss": 1, "target_price": 1, "opened_at": "2025-01-01",
                                      "is_open": True}])
        pt = self.trader(cloud=cloud, initial_capital=100_000)
        self.assertEqual(pt.restored_from, "new")
        self.assertEqual(pt.cash, 100_000)
        self.assertEqual(pt._positions, [])
        self.assertEqual(float(cloud.state["cash"]), 100_000)

    def test_cloud_failure_disables_sync(self):
        class Broken(FakeCloud):
            def read_state(self):
                raise RuntimeError("neon down")
        cloud = Broken()
        pt = self.trader(cloud=cloud, initial_capital=100_000)
        self.assertEqual(pt.restored_from, "cloud_restore_failed")
        pt.buy("INFY", 1, price=100.0)
        self.assertEqual(cloud.state, {})  # nothing overwritten

    def test_gap_through_stop_fills_at_open_and_charges_costs(self):
        pt = self.trader(initial_capital=100_000)
        res = pt.buy("INFY", 10, price=1000.0, stop_loss=950.0)
        self.assertTrue(res["success"])
        buy_cost = self.pt_mod.statutory_cost_inr("BUY", 10_000.0)
        self.assertAlmostEqual(pt.cash, 100_000 - 10_000 - buy_cost, places=2)
        pos = pt._positions[0]
        pos.opened_at = "2025-06-02T10:00:00+05:30"
        # Same-day bar is ignored (entry was intraday)
        ev0 = pt.simulate_gtt_stops({"INFY": {"date": "2025-06-02", "open": 990, "low": 900, "close": 960}})
        self.assertEqual(ev0, [])
        # Next day gaps below the stop: fill at the open (920), not the stop (950)
        ev = pt.simulate_gtt_stops({"INFY": {"date": "2025-06-03", "open": 920, "low": 900, "close": 930}})
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0]["exit"], 920.0)
        self.assertEqual(ev[0]["type"], "PAPER_GTT_SL_GAP")
        sell_cost = self.pt_mod.statutory_cost_inr("SELL", 9_200.0)
        self.assertGreater(sell_cost, 15.0)  # includes DP charge
        self.assertAlmostEqual(ev[0]["pnl"], -800 - buy_cost - sell_cost, places=2)
        self.assertAlmostEqual(pt.cash, 100_000 - 10_000 - buy_cost + 9_200 - sell_cost, places=2)

    def test_intraday_stop_fills_at_stop_when_open_above(self):
        pt = self.trader(initial_capital=100_000)
        pt.buy("TCS", 2, price=4000.0, stop_loss=3900.0)
        pt._positions[0].opened_at = "2025-06-02T10:00:00+05:30"
        ev = pt.simulate_gtt_stops({"TCS": {"date": "2025-06-03", "open": 3950, "low": 3880, "close": 3890}})
        self.assertEqual(ev[0]["exit"], 3900.0)

    def test_poll_gap_uses_first_observed_price(self):
        pt = self.trader(initial_capital=100_000)
        pt.buy("SBIN", 10, price=500.0, stop_loss=480.0)
        pt._price_overrides["SBIN"] = 470.0
        with mock.patch.object(pt, "_trail_stop", lambda pos, ltp: None):
            ev = pt.poll()
        self.assertEqual(ev[0]["exit"], 470.0)

    def test_partial_close(self):
        pt = self.trader(initial_capital=100_000)
        pt.buy("INFY", 10, price=100.0)
        res = pt.close_position("INFY", quantity=4, price=110.0, reason="REBALANCE")
        self.assertTrue(res["success"])
        self.assertEqual(pt.holdings()["INFY"]["quantity"], 6)
        self.assertEqual(res["quantity"], 4)


# ─────────────────────────────────────────────────────────────
# 5. DailyRebalancer call signature
# ─────────────────────────────────────────────────────────────

class TestDailyRebalancer(unittest.TestCase):
    def test_place_order_calls_match_real_signature(self):
        from kite_connect.trading import daily_rebalancer as dr, order_service
        with mock.patch.dict(os.environ, {"CENTURION_PAPER_TRADE": "false"}):
            rb = dr.DailyRebalancer(kite=FakeKite(), paper_mode=False)
        self.assertFalse(rb.paper_mode)
        targets = [
            dr.TargetPosition("INFY.NS", 5.0, 10_000, 10, 0, 10, "BUY", 1000.0),
            dr.TargetPosition("TCS.NS", 5.0, 4_000, 1, 3, -2, "SELL", 4000.0),
        ]
        with mock.patch.object(order_service, "place_order", autospec=True,
                               return_value={"success": True, "order_id": "1"}) as po:
            placed, failed = rb._execute_orders(targets, ["HDFC"], {"HDFC": 7, "TCS": 3})
        self.assertEqual((placed, failed), (3, 0))
        calls = [c.kwargs for c in po.call_args_list]
        self.assertEqual([(c["symbol"], c["transaction_type"], c["quantity"]) for c in calls],
                         [("HDFC", "SELL", 7), ("TCS", "SELL", 2), ("INFY", "BUY", 10)])
        self.assertTrue(calls[0]["is_exit"] and calls[1]["is_exit"])
        self.assertFalse(calls[2]["is_exit"])
        self.assertTrue(all(c["product"] == "CNC" and c["exchange"] == "NSE" for c in calls))

    def test_env_forces_paper(self):
        from kite_connect.trading import daily_rebalancer as dr
        with mock.patch.dict(os.environ, {"CENTURION_PAPER_TRADE": "true"}):
            self.assertTrue(dr.DailyRebalancer(kite=FakeKite(), paper_mode=False).paper_mode)
            self.assertTrue(dr.DailyRebalancer(kite=FakeKite()).paper_mode)

    def test_sizing_uses_instrument_volatility(self):
        from kite_connect.trading import daily_rebalancer as dr
        rng = np.random.default_rng(0)
        idx = pd.bdate_range("2024-01-01", periods=120)
        low = pd.DataFrame({"Close": 100 * np.cumprod(1 + rng.normal(0, 0.005, 120))}, index=idx)
        high = pd.DataFrame({"Close": 100 * np.cumprod(1 + rng.normal(0, 0.03, 120))}, index=idx)
        rb = dr.DailyRebalancer(kite=None)
        with mock.patch.object(rb, "_get_config", return_value=SimpleNamespace(CARVER_ANNUAL_VOL_TARGET=0.01)):
            tgt = rb._rank_and_select({"LOW": 10.0, "HIGH": 10.0}, {}, 1_000_000, 1.0,
                                      {"LOW": low, "HIGH": high})
        self.assertGreater(tgt["LOW"], tgt["HIGH"] * 2)
        self.assertLessEqual(sum(tgt.values()), 1_000_000 + 1e-6)


# ─────────────────────────────────────────────────────────────
# 6. Engine executor
# ─────────────────────────────────────────────────────────────

def _market_data(prices: dict, end="2024-03-28", n=30):
    from nse_engine.types import MarketData
    dates = pd.bdate_range(end=end, periods=n)
    close = pd.DataFrame({s: [p] * n for s, p in prices.items()}, index=dates, dtype=float)
    empty = pd.DataFrame(index=dates)
    return MarketData(dates=dates, open=close.copy(), high=close.copy(), low=close.copy(),
                      close=close, volume=close * 0 + 1e6, value=close * 1e6, index_close=empty)


class TestEngineExecutor(PaperTestBase):
    def _executor(self, weights, holdings, cash, prices, exits=None, stops=None, **kw):
        from nse_engine.config import EngineConfig
        from nse_engine.types import TargetPortfolio
        from kite_connect.trading.nse_engine_executor import EngineExecutor

        data = _market_data(prices)
        seen = {"loader_calls": []}

        def loader(*a, **k):
            seen["loader_calls"].append((a, k))
            return data

        def target_fn(d, cfg, as_of, holdings=None, cache=None, *, equity=None, stopped_out=None):
            seen["as_of"] = as_of
            seen["holdings"] = holdings
            seen["equity"] = equity
            seen["stopped_out"] = stopped_out
            seen["last_date"] = d.dates[-1]
            return TargetPortfolio(as_of=as_of, weights=weights, exits=exits or {}, stops=stops or {})

        ex = EngineExecutor(kite=None, paper=True, config=EngineConfig(),
                            target_fn=target_fn, data_loader=loader,
                            holdings_fn=lambda: (holdings, cash), **kw)
        return ex, seen

    def test_sells_before_buys_exits_and_stops(self):
        ex, seen = self._executor(
            weights={"AAA": 0.10, "CCC": 0.20},
            holdings={"BBB": {"quantity": 100, "avg_price": 90.0},
                      "AAA": {"quantity": 50, "avg_price": 100.0}},
            cash=80_000.0,
            prices={"AAA": 100.0, "BBB": 200.0, "CCC": 50.0},
            exits={"BBB": "rank_exit"}, stops={"CCC": 45.0, "AAA": 90.0},
        )
        plan = ex.plan(as_of="2024-03-28")
        # equity = 80k + 100*200 + 50*100 = 105k, passed to the engine
        self.assertAlmostEqual(plan.equity, 105_000.0)
        self.assertAlmostEqual(seen["equity"], 105_000.0)
        self.assertIn("AAA", seen["holdings"])
        # Data is loaded from a fixed anchor (config.start - 2y), not a rolling window
        (args, kwargs), = seen["loader_calls"]
        self.assertEqual(args[1], "2011-01-01")
        self.assertIn("GOLDBEES", kwargs["include_symbols"])
        sides = [o.side for o in plan.orders]
        self.assertEqual(sides, sorted(sides, key=lambda s: 0 if s == "SELL" else 1))
        by = {o.symbol: o for o in plan.orders}
        self.assertEqual((by["BBB"].side, by["BBB"].quantity), ("SELL", 100))
        self.assertTrue(by["BBB"].reason.startswith("exit:"))
        self.assertEqual((by["CCC"].side, by["CCC"].quantity), ("BUY", 420))  # 21k / 50
        # AAA: target 10.5k = 105 sh vs 50 held (outside 25% buffer) -> buy 55
        self.assertEqual((by["AAA"].side, by["AAA"].quantity), ("BUY", 55))
        stops = {s.symbol: (s.quantity, s.trigger) for s in plan.stop_instructions}
        self.assertEqual(stops["CCC"], (420, 45.0))
        self.assertEqual(stops["AAA"], (105, 90.0))

    def test_no_trade_buffer_and_min_trade_value(self):
        ex, _ = self._executor(
            weights={"AAA": 0.10, "BBB": 0.001},
            holdings={"AAA": {"quantity": 90, "avg_price": 100.0}},
            cash=91_000.0, prices={"AAA": 100.0, "BBB": 100.0}, apply_buffer=True,
        )
        plan = ex.plan(as_of="2024-03-28")  # equity 100k; AAA target 100 sh vs 90 held (10% < 25%)
        reasons = {s["symbol"]: s["reason"] for s in plan.skipped}
        self.assertEqual(reasons.get("AAA"), "within_buffer")
        self.assertEqual(reasons.get("BBB"), "below_min_trade_value")  # 1 share = ₹100
        self.assertEqual(plan.orders, [])

    def test_drifted_weights_do_not_trade(self):
        # Non-rebalance day: engine returns current (drifted) weights -> no orders
        equity = 20_000.0 + 7 * 31_337.35
        ex, seen = self._executor(
            weights={"MRF": 7 * 31_337.35 / equity},
            holdings={"MRF": {"quantity": 7, "avg_price": 30_000.0, "stop_price": 29_000.0}},
            cash=20_000.0, prices={"MRF": 31_337.35},
        )
        plan = ex.plan(as_of="2024-03-28")
        self.assertEqual(plan.orders, [])
        self.assertEqual(seen["holdings"]["MRF"].stop_price, 29_000.0)

    def test_cash_limit(self):
        ex, _ = self._executor(
            weights={"AAA": 0.6, "BBB": 0.6}, holdings={}, cash=100_000.0,
            prices={"AAA": 100.0, "BBB": 100.0},
        )
        plan = ex.plan(as_of="2024-03-28")
        spend = sum(o.quantity * o.limit_price for o in plan.buys)
        self.assertLessEqual(spend, 100_000.0)
        self.assertEqual(plan.buys[0].quantity, 600)
        self.assertLess(plan.buys[1].quantity, 600)

    def test_paper_execute_updates_book(self):
        pt = self.pt_mod.PaperTrader(kite=None, initial_capital=100_000, slippage_bps=0.0, cloud=FakeCloud())
        pt.buy("BBB", 100, price=200.0)
        pt.buy("DDD", 1, price=100.0)
        pt.close_position("DDD", price=90.0, reason="GTT_SL")
        ex, _ = self._executor(
            weights={"CCC": 0.20}, holdings=None, cash=None,
            prices={"BBB": 200.0, "CCC": 50.0}, exits={"BBB": "stop"}, stops={"CCC": 45.0},
            paper_trader=pt,
        )
        ex._holdings_fn = None  # use the PaperTrader book
        plan = ex.plan(as_of="2024-03-28")
        self.assertIn("DDD", seen_stopped := (ex._stopped_out(pd.Timestamp.today())))
        self.assertIsInstance(seen_stopped["DDD"], pd.Timestamp)
        results = ex.execute(plan)
        self.assertTrue(all(r.get("success") for r in results), results)
        # Default paper execution queues next-open orders: the book is unchanged
        self.assertIn("BBB", pt.holdings())
        pending = {(o["symbol"], o["side"]): o for o in pt.pending_orders()}
        self.assertEqual(pending[("BBB", "SELL")]["target_qty"], 0)
        self.assertEqual(pending[("CCC", "BUY")]["stop_price"], 45.0)
        # Legacy immediate mode fills at the close at once
        pt.queue_pending_orders("2024-03-28", [])
        ex.paper_fill = "immediate"
        results = ex.execute(plan)
        self.assertTrue(all(r.get("success") for r in results), results)
        held = pt.holdings()
        self.assertNotIn("BBB", held)
        self.assertIn("CCC", held)
        self.assertEqual(held["CCC"]["stop_price"], 45.0)


def _deployment(status="approved"):
    from nse_engine.config import EngineConfig
    from nse_engine.deployment import Deployment
    return Deployment(engine=EngineConfig(), paper_start_date=pd.Timestamp("2026-01-01").date(), status=status,
                      source_run_id="run1" if status == "approved" else None,
                      approved_at="2026-01-01T00:00:00+05:30" if status == "approved" else None)


class TestLiveGuard(unittest.TestCase):
    def test_env_logic(self):
        from kite_connect.trading import nse_engine_executor as ne
        cases = [
            ({"CENTURION_PAPER_TRADE": "true", "CENTURION_NSE_ENGINE_LIVE": "true"}, False),
            ({"CENTURION_PAPER_TRADE": "false", "CENTURION_NSE_ENGINE_LIVE": "false"}, False),
            ({"CENTURION_PAPER_TRADE": "false"}, False),
            ({"CENTURION_NSE_ENGINE_LIVE": "true"}, False),  # paper defaults to true
            ({"CENTURION_PAPER_TRADE": "false", "CENTURION_NSE_ENGINE_LIVE": "true"}, True),
        ]
        for env, expected in cases:
            clean = {k: v for k, v in os.environ.items()
                     if k not in ("CENTURION_PAPER_TRADE", "CENTURION_NSE_ENGINE_LIVE")}
            clean.update(env)
            with mock.patch.dict(os.environ, clean, clear=True):
                self.assertEqual(ne.live_orders_allowed()[0], expected, env)
                ex = ne.EngineExecutor(kite=FakeKite(), paper=False, config=object(),
                                       deployment=_deployment())
                self.assertEqual(ex.paper, not expected, env)
                # A placeholder deployment never trades live
                ex = ne.EngineExecutor(kite=FakeKite(), paper=False, config=object(),
                                       deployment=_deployment("placeholder"))
                self.assertTrue(ex.paper, env)

    def test_live_without_kite_forces_paper(self):
        from kite_connect.trading import nse_engine_executor as ne
        with mock.patch.dict(os.environ, {"CENTURION_PAPER_TRADE": "false",
                                          "CENTURION_NSE_ENGINE_LIVE": "true"}):
            self.assertTrue(ne.EngineExecutor(kite=None, paper=False).paper)

    def test_runner_engine_flag_requires_env(self):
        import cloud_paper_runner as cpr
        with mock.patch.dict(os.environ, {"CENTURION_NSE_ENGINE": "false"}):
            self.assertFalse(cpr._engine_enabled(["--engine"]))
        with mock.patch.dict(os.environ, {"CENTURION_NSE_ENGINE": "true"}):
            self.assertTrue(cpr._engine_enabled(["--engine"]))
            self.assertFalse(cpr._engine_enabled([]))


if __name__ == "__main__":
    unittest.main()
