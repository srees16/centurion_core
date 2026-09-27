"""The live order path under test (tracker L3).

Before L3 ``EngineExecutor._execute_live`` had never run: no test, no dry run.
A fake Kite exercises every branch - order placement, sells before buys,
after-market variety once the market is closed, duplicate tags, broker
rejections, transient failures and retries, the kill switch, GTT stop
reconciliation, partial and rejected outcomes - and the dry run proves the
same order list can be rehearsed without a broker.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pandas as pd
import pytest
from kiteconnect import exceptions as kx

import kite_connect.trading.order_service as osvc
from kite_connect.trading.gtt_stops import stop_limit_price
from kite_connect.trading.nse_engine_executor import (EngineExecutor, ExecutionPlan, PlannedOrder,
                                                      StopInstruction, cloud_equity_history,
                                                      live_order_outcomes)
from nse_engine.config import EngineConfig
from nse_engine.deployment import parse_deployment

AS_OF = pd.Timestamp("2026-09-28")


# ---------------------------------------------------------------- fake broker

class FakeKite:
    """Enough of KiteConnect for the engine's live path, with switchable faults."""

    def __init__(self, holdings=None, cash=500_000.0, gtts=None, ltp=None, reject=None,
                 fail_times=0, order_book=None):
        self._holdings = dict(holdings or {})          # symbol -> (qty, avg_price)
        self._cash = cash
        self._gtts = list(gtts or [])
        self._ltp = dict(ltp or {})
        self.reject = dict(reject or {})
        self.fail_times = int(fail_times)
        self.placed: list = []
        self.gtt_calls: list = []
        self._book = list(order_book or [])
        self._n = 0

    # orders
    def place_order(self, **params):
        sym = params["tradingsymbol"]
        if sym in self.reject:
            raise self.reject[sym]
        if self.fail_times > 0:
            self.fail_times -= 1
            raise kx.NetworkException("timeout")
        self._n += 1
        oid = f"ORD{self._n}"
        self.placed.append(dict(params, order_id=oid))
        self._book.append({"order_id": oid, "tag": params["tag"], "status": "OPEN",
                           "tradingsymbol": sym, "transaction_type": params["transaction_type"],
                           "quantity": params["quantity"], "filled_quantity": 0, "average_price": 0.0,
                           "variety": params["variety"]})
        return oid

    def orders(self):
        return list(self._book)

    def order_history(self, order_id):
        return []

    # book
    def holdings(self):
        return [{"tradingsymbol": s, "exchange": "NSE", "quantity": q, "t1_quantity": 0, "average_price": p}
                for s, (q, p) in self._holdings.items()]

    def positions(self):
        return {"net": []}

    def margins(self, segment):
        return {"available": {"live_balance": self._cash}}

    def ltp(self, keys):
        return {k: {"last_price": self._ltp.get(k.split(":", 1)[1], 0.0)} for k in keys}

    # GTT
    def get_gtts(self):
        return list(self._gtts)

    def place_gtt(self, **kw):
        self.gtt_calls.append(("place", kw))
        gid = 900 + len(self.gtt_calls)
        self._gtts.append(_gtt(gid, kw["tradingsymbol"], kw["orders"][0]["quantity"], kw["trigger_values"][0],
                               kw["orders"][0]["price"]))
        return {"trigger_id": gid}

    def modify_gtt(self, **kw):
        self.gtt_calls.append(("modify", kw))

    def delete_gtt(self, trigger_id):
        self.gtt_calls.append(("delete", trigger_id))
        self._gtts = [g for g in self._gtts if g["id"] != trigger_id]


def _gtt(gid, symbol, qty, trigger, limit, status="active"):
    return {"id": gid, "status": status, "type": "single",
            "condition": {"exchange": "NSE", "tradingsymbol": symbol, "trigger_values": [trigger]},
            "orders": [{"exchange": "NSE", "tradingsymbol": symbol, "transaction_type": "SELL",
                        "quantity": qty, "order_type": "LIMIT", "product": "CNC", "price": limit}]}


def _deployment(status="approved"):
    raw = {"status": status, "paper_start_date": "2026-09-16", "engine": EngineConfig().to_dict()}
    if status == "approved":
        raw.update(source_run_id="run", approved_at="2026-09-15T21:48:04+05:30")
    return parse_deployment(raw)


def _plan(orders=(), stops=()):
    plan = ExecutionPlan(as_of=AS_OF, equity=1_000_000.0, cash=200_000.0)
    plan.orders = [PlannedOrder(sym, side, qty, px, limit, reason, cur, tgt, w)
                   for sym, side, qty, px, limit, reason, cur, tgt, w in orders]
    plan.stop_instructions = [StopInstruction(s, q, t) for s, q, t in stops]
    return plan


ORDERS = [("BEL", "BUY", 100, 300.0, 303.0, "entry", 0, 100, 0.03),
          ("RELIANCE", "SELL", 10, 2900.0, 2871.0, "exit:rank_exit", 10, 0, 0.0),
          ("HAL", "BUY", 5, 4500.0, 4545.0, "rebalance", 5, 10, 0.045)]


@pytest.fixture
def live_env(monkeypatch):
    """Live permitted, market closed, broker side effects and sleeps stubbed."""
    monkeypatch.setenv("CENTURION_PAPER_TRADE", "false")
    monkeypatch.setenv("CENTURION_NSE_ENGINE_LIVE", "true")
    monkeypatch.delenv("CENTURION_KILL_SWITCH", raising=False)
    monkeypatch.setattr(osvc, "_persist_to_db", lambda *a, **k: None)
    monkeypatch.setattr(osvc, "_send_order_email", lambda *a, **k: None)
    monkeypatch.setattr(osvc.time, "sleep", lambda s: None)
    monkeypatch.setattr(osvc, "_order_circuit", None)
    monkeypatch.setattr(osvc, "_is_nse_market_open", lambda: False)
    return monkeypatch


# ---------------------------------------------------------------- routing

class TestRouting:
    def test_without_the_env_flags_everything_is_paper(self, monkeypatch):
        monkeypatch.delenv("CENTURION_PAPER_TRADE", raising=False)
        monkeypatch.delenv("CENTURION_NSE_ENGINE_LIVE", raising=False)
        ex = EngineExecutor(kite=FakeKite(), paper=False, deployment=_deployment())
        assert ex.paper and "CENTURION_PAPER_TRADE" in ex.mode_reason

    def test_a_placeholder_deployment_forces_paper(self, live_env):
        ex = EngineExecutor(kite=FakeKite(), paper=False, deployment=_deployment("placeholder"))
        assert ex.paper and "placeholder" in ex.mode_reason

    def test_flags_plus_approved_deployment_plus_session_is_live(self, live_env):
        ex = EngineExecutor(kite=FakeKite(), paper=False, deployment=_deployment())
        assert not ex.paper

    def test_no_kite_session_falls_back_to_paper_at_execute(self, live_env):
        ex = EngineExecutor(kite=None, paper=False, deployment=_deployment())
        assert ex.paper, "constructor already knows there is no session"


# ---------------------------------------------------------------- the order list

class TestLiveOrders:
    def test_sells_first_limit_cnc_and_after_market_when_closed(self, live_env):
        ex = EngineExecutor(kite=FakeKite(), paper=False, deployment=_deployment())
        specs = ex.live_orders(_plan(ORDERS))
        assert [s["transaction_type"] for s in specs] == ["SELL", "BUY", "BUY"]
        assert all(s["order_type"] == "LIMIT" and s["product"] == "CNC" and s["variety"] == "amo" for s in specs)
        assert specs[0]["tag"] == "NE260928SRELIANCE" and specs[1]["tag"] == "NE260928BBEL"
        assert specs[0]["is_exit"] and not specs[1]["is_exit"]
        assert specs[0]["price"] == 2871.0

    def test_regular_variety_during_market_hours(self, live_env):
        live_env.setattr(osvc, "_is_nse_market_open", lambda: True)
        ex = EngineExecutor(kite=FakeKite(), paper=False, deployment=_deployment())
        assert {s["variety"] for s in ex.live_orders(_plan(ORDERS))} == {"regular"}

    def test_tags_never_exceed_kites_20_characters(self, live_env):
        ex = EngineExecutor(kite=FakeKite(), paper=False, deployment=_deployment())
        long = [("AVERYLONGSYMBOLNAME1", "BUY", 1, 10.0, 10.1, "entry", 0, 1, 0.01)]
        assert len(ex.live_orders(_plan(long))[0]["tag"]) == 20


# ---------------------------------------------------------------- execution

class TestExecuteLive:
    def test_orders_are_placed_and_stops_reconciled(self, live_env):
        kite = FakeKite(holdings={"RELIANCE": (10, 2500.0), "HAL": (5, 4000.0), "BEL": (100, 290.0)},
                        ltp={"RELIANCE": 2900.0, "HAL": 4500.0, "BEL": 300.0},
                        gtts=[_gtt(1, "HAL", 5, 4100.0, stop_limit_price(4100.0)),   # exactly what reconcile wants
                              _gtt(2, "GONE", 3, 50.0, 49.75)])
        ex = EngineExecutor(kite=kite, paper=False, deployment=_deployment())
        results = ex.execute(_plan(ORDERS, stops=[("BEL", 100, 280.0), ("HAL", 5, 4100.0)]))
        placed = [r for r in results if r.get("type") != "gtt_reconcile"]
        assert [r["status"] for r in placed] == ["PLACED", "PLACED", "PLACED"]
        assert [p["transaction_type"] for p in kite.placed] == ["SELL", "BUY", "BUY"], "sells go first"
        assert all(p["variety"] == "amo" and p["product"] == "CNC" and p["order_type"] == "LIMIT" for p in kite.placed)
        assert kite.placed[0]["tag"] == "NE260928SRELIANCE" and placed[0]["order_id"] == "ORD1"
        gtt = results[-1]
        assert gtt["type"] == "gtt_reconcile" and gtt["success"]
        rep = gtt["report"]
        assert [g["symbol"] for g in rep["placed"]] == ["BEL"]              # new stop
        assert [g["symbol"] for g in rep["unchanged"]] == ["HAL"]           # same trigger, same qty
        assert rep["deleted"] and rep["deleted"][0]["symbol"] == "GONE"     # orphan GTT removed
        assert "RELIANCE" in rep["missing_stop"]                            # held, no stop, no GTT

    def test_a_duplicate_tag_is_not_sent_again(self, live_env):
        kite = FakeKite(order_book=[{"order_id": "X", "tag": "NE260928BBEL", "status": "OPEN",
                                     "tradingsymbol": "BEL", "transaction_type": "BUY", "quantity": 100,
                                     "filled_quantity": 0, "average_price": 0.0}])
        ex = EngineExecutor(kite=kite, paper=False, deployment=_deployment())
        results = ex.execute(_plan(ORDERS[:1]))
        assert results[0]["status"] == "DUPLICATE" and not results[0]["success"] and kite.placed == []

    def test_a_broker_rejection_does_not_stop_the_other_orders(self, live_env):
        kite = FakeKite(reject={"BEL": kx.OrderException("RMS: insufficient margin")})
        ex = EngineExecutor(kite=kite, paper=False, deployment=_deployment())
        results = ex.execute(_plan(ORDERS))
        by = {r["symbol"]: r for r in results if r.get("type") != "gtt_reconcile"}
        assert by["BEL"]["status"] == "REJECTED" and "insufficient margin" in by["BEL"]["error"]
        assert by["RELIANCE"]["status"] == "PLACED" and by["HAL"]["status"] == "PLACED"
        assert [p["tradingsymbol"] for p in kite.placed] == ["RELIANCE", "HAL"]

    def test_transient_failures_are_retried_then_given_up(self, live_env):
        ex = EngineExecutor(kite=FakeKite(fail_times=2), paper=False, deployment=_deployment())
        r = ex.execute(_plan(ORDERS[:1]))[0]
        assert r["status"] == "PLACED", "two timeouts, third attempt succeeds"
        ex = EngineExecutor(kite=FakeKite(fail_times=3), paper=False, deployment=_deployment())
        r = ex.execute(_plan(ORDERS[:1]))[0]
        assert r["status"] == "REJECTED" and "timeout" in r["error"]

    def test_kill_switch_refuses_buys_and_lets_exits_through(self, live_env):
        live_env.setenv("CENTURION_KILL_SWITCH", "true")
        kite = FakeKite(holdings={"RELIANCE": (10, 2500.0)}, ltp={"RELIANCE": 2900.0})
        ex = EngineExecutor(kite=kite, paper=False, deployment=_deployment())
        results = ex.execute(_plan(ORDERS, stops=[("RELIANCE", 10, 2700.0)]))
        by = {r["symbol"]: r for r in results if r.get("type") != "gtt_reconcile"}
        assert by["RELIANCE"]["status"] == "PLACED"                          # reduce-only exit
        assert by["BEL"]["status"] == "REJECTED" and "KILL SWITCH" in by["BEL"]["error"]
        assert by["HAL"]["status"] == "REJECTED"
        assert [p["tradingsymbol"] for p in kite.placed] == ["RELIANCE"]
        assert results[-1]["report"]["placed"][0]["symbol"] == "RELIANCE"    # protection still armed

    def test_a_stop_above_the_market_is_reported_not_placed(self, live_env):
        kite = FakeKite(holdings={"BEL": (100, 290.0)}, ltp={"BEL": 300.0})
        ex = EngineExecutor(kite=kite, paper=False, deployment=_deployment())
        rep = ex.execute(_plan(stops=[("BEL", 100, 310.0)]))[-1]["report"]
        assert rep["breached"] and rep["breached"][0]["symbol"] == "BEL" and not rep["placed"]


class TestPlaceOrderGuards:
    def test_regular_orders_are_refused_after_hours_but_amo_goes_through(self, live_env):
        kite = FakeKite()
        r = osvc.place_order(kite, "BEL", "NSE", "BUY", 1, order_type="LIMIT", price=300.0, tag="t1")
        assert not r["success"] and "market closed" in r["error"] and kite.placed == []
        r = osvc.place_order(kite, "BEL", "NSE", "BUY", 1, order_type="LIMIT", price=300.0, tag="t2", variety="amo")
        assert r["success"] and r["variety"] == "amo" and kite.placed[0]["variety"] == "amo"

    def test_unknown_variety_is_refused(self, live_env):
        r = osvc.place_order(FakeKite(), "BEL", "NSE", "BUY", 1, order_type="LIMIT", price=300.0, variety="bogus")
        assert not r["success"] and "variety" in r["error"]


# ---------------------------------------------------------------- dry run

class TestDryRun:
    def test_dry_run_builds_the_same_orders_and_sends_none(self, live_env):
        kite = FakeKite()
        ex = EngineExecutor(kite=kite, paper=False, deployment=_deployment(), dry_run=True)
        plan = _plan(ORDERS, stops=[("BEL", 100, 280.0)])
        results = ex.execute(plan)
        orders = [r for r in results if r.get("type") != "gtt_reconcile"]
        assert kite.placed == [] and kite.gtt_calls == []
        assert [o["status"] for o in orders] == ["WOULD_PLACE"] * 3
        assert [(o["side"], o["symbol"], o["tag"], o["variety"]) for o in orders] == \
            [(s["transaction_type"], s["symbol"], s["tag"], s["variety"]) for s in ex.live_orders(plan)]
        assert results[-1]["stops"] == [{"symbol": "BEL", "quantity": 100, "trigger": 280.0}]

    def test_dry_run_works_from_paper_mode_and_shows_the_kill_switch(self, monkeypatch):
        monkeypatch.delenv("CENTURION_PAPER_TRADE", raising=False)
        monkeypatch.setenv("CENTURION_KILL_SWITCH", "true")
        monkeypatch.setattr(osvc, "_is_nse_market_open", lambda: False)
        ex = EngineExecutor(kite=None, paper=True, deployment=_deployment("placeholder"), dry_run=True)
        results = ex.dry_run_live(_plan(ORDERS))
        by = {r["symbol"]: r for r in results if r.get("type") != "gtt_reconcile"}
        assert by["RELIANCE"]["status"] == "WOULD_PLACE"
        assert by["BEL"]["status"] == "WOULD_REJECT" and "KILL SWITCH" in by["BEL"]["error"]

    def test_the_tool_report_reads_cleanly(self, monkeypatch):
        from tools.live_dry_run import format_report, report

        monkeypatch.delenv("CENTURION_PAPER_TRADE", raising=False)
        monkeypatch.delenv("CENTURION_KILL_SWITCH", raising=False)
        monkeypatch.setattr(osvc, "_is_nse_market_open", lambda: False)
        dep = _deployment()
        ex = EngineExecutor(kite=None, paper=True, deployment=dep, dry_run=True)
        plan = _plan(ORDERS, stops=[("BEL", 100, 280.0)])
        plan.skipped.append({"symbol": "OIL", "reason": "insufficient_cash"})
        from tools.live_dry_run import gates
        rep = report(dep, plan, ex.dry_run_live(plan), gates(), "paper")
        text = format_report(rep)
        assert rep["would_send"] == 3 and "nothing is sent" in text
        assert "NE260928SRELIANCE" in text and "amo" in text and "OIL: insufficient_cash" in text
        assert "blocks live orders" in text


# ---------------------------------------------------------------- outcomes and history

def test_live_order_outcomes_classify_every_case():
    book = [
        {"order_id": "1", "tag": "NE260928BBEL", "status": "COMPLETE", "tradingsymbol": "BEL", "transaction_type": "BUY",
         "quantity": 100, "filled_quantity": 100, "average_price": 301.2, "variety": "amo"},
        {"order_id": "2", "tag": "NE260928BHAL", "status": "CANCELLED", "tradingsymbol": "HAL", "transaction_type": "BUY",
         "quantity": 10, "filled_quantity": 4, "average_price": 4510.0, "variety": "amo"},
        {"order_id": "3", "tag": "NE260928SRELIANCE", "status": "REJECTED", "tradingsymbol": "RELIANCE",
         "transaction_type": "SELL", "quantity": 10, "filled_quantity": 0, "average_price": 0,
         "status_message": "RMS: holding not available", "variety": "amo"},
        {"order_id": "4", "tag": "NE260928BOIL", "status": "OPEN", "tradingsymbol": "OIL", "transaction_type": "BUY",
         "quantity": 50, "filled_quantity": 0, "average_price": 0, "variety": "amo"},
        {"order_id": "5", "tag": "NE260925BOLD", "status": "COMPLETE", "tradingsymbol": "OLD", "transaction_type": "BUY",
         "quantity": 1, "filled_quantity": 1, "average_price": 1.0},
        {"order_id": "6", "tag": "manual", "status": "COMPLETE", "tradingsymbol": "X", "transaction_type": "BUY",
         "quantity": 1, "filled_quantity": 1, "average_price": 1.0},
    ]
    out = {o["symbol"]: o for o in live_order_outcomes(SimpleNamespace(orders=lambda: book), AS_OF)}
    assert set(out) == {"BEL", "HAL", "RELIANCE", "OIL"}, "other days' and manual orders are ignored"
    assert out["BEL"]["outcome"] == "complete" and out["BEL"]["average_price"] == 301.2
    assert out["HAL"]["outcome"] == "partial" and out["HAL"]["filled"] == 4
    assert out["RELIANCE"]["outcome"] == "rejected" and "holding not available" in out["RELIANCE"]["error"]
    assert out["OIL"]["outcome"] == "open"
    broken = live_order_outcomes(SimpleNamespace(orders=lambda: (_ for _ in ()).throw(RuntimeError("down"))), AS_OF)
    assert broken[0]["status"] == "unknown"


def test_cloud_equity_history_reads_the_books_snapshots():
    df = pd.DataFrame({"date": ["2026-09-17", "2026-09-18", "2026-09-18"], "equity": [3.5e6, 3.57e6, 3.58e6]})
    s = cloud_equity_history(SimpleNamespace(read_snapshots=lambda: df))
    assert list(s.index.strftime("%Y-%m-%d")) == ["2026-09-17", "2026-09-18"] and s.iloc[-1] == 3.58e6
    assert cloud_equity_history(SimpleNamespace(read_snapshots=lambda: pd.DataFrame())).empty
    assert cloud_equity_history(None if os.environ.get("CENTURION_DATABASE_URL") else SimpleNamespace(read_snapshots=lambda: None)).empty


def test_live_executor_defaults_to_the_cloud_history(live_env, monkeypatch):
    import kite_connect.trading.nse_engine_executor as mod

    monkeypatch.setattr(mod, "cloud_equity_history", lambda cloud=None: pd.Series(
        [1.0, 2.0], index=pd.DatetimeIndex(["2026-09-17", "2026-09-18"])))
    ex = EngineExecutor(kite=FakeKite(), paper=False, deployment=_deployment())
    assert list(ex.equity_history()) == [1.0, 2.0]
