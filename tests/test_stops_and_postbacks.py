"""Stops at the broker and what Kite reports back: GTT stop-losses on live holdings (place, update, dedupe,
refuse, reconcile) and the alerts raised from Kite's order postbacks (AL5)."""
from __future__ import annotations

import json

import pytest


# ---- GTT stop-losses on live holdings (gtt_stops.py): place, update, dedupe, refuse, reconcile
class _GttKite:
    """A broker with just what gtt_stops calls; every write is recorded, any call can be made to fail."""

    def __init__(self, holdings=None, positions=None, ltp=None, gtts=None, fail=()):
        self.h, self.p, self.prices, self.fail = holdings or [], positions or [], ltp or {}, set(fail)
        self.gtts = {g["id"]: g for g in (gtts or [])}
        self.calls, self.next_id = [], 900

    def _maybe(self, name):
        if name in self.fail:
            raise RuntimeError(f"{name} refused")

    def holdings(self):
        self._maybe("holdings")
        return self.h

    def positions(self):
        return {"net": self.p}

    def ltp(self, keys):
        self._maybe("ltp")
        return {k: {"last_price": self.prices[k.split(":", 1)[1]]} for k in keys if k.split(":", 1)[1] in self.prices}

    def instruments(self, exchange="NSE"):
        raise RuntimeError("no dump")

    def get_gtts(self):
        self._maybe("get_gtts")
        return list(self.gtts.values())

    def get_gtt(self, trigger_id):
        return self.gtts[trigger_id]

    def place_gtt(self, trigger_type, tradingsymbol, exchange, trigger_values, last_price, orders):
        self._maybe("place_gtt")
        self.next_id += 1
        self.gtts[self.next_id] = _gtt(self.next_id, tradingsymbol, trigger_values[0], orders[0]["price"],
                                       orders[0]["quantity"])
        self.calls.append(("place", tradingsymbol, trigger_values[0], orders[0]))
        return {"trigger_id": self.next_id}

    def modify_gtt(self, trigger_id, trigger_type, tradingsymbol, exchange, trigger_values, last_price, orders):
        self.gtts[trigger_id] = _gtt(trigger_id, tradingsymbol, trigger_values[0], orders[0]["price"],
                                     orders[0]["quantity"])
        self.calls.append(("modify", trigger_id, trigger_values[0], orders[0]))

    def delete_gtt(self, trigger_id):
        self.gtts.pop(trigger_id)
        self.calls.append(("delete", trigger_id))


def _gtt(gid, symbol, trigger, limit, qty, status="active", side="SELL", product="CNC", legs=1, kind="single"):
    order = {"exchange": "NSE", "tradingsymbol": symbol, "transaction_type": side, "quantity": qty,
             "order_type": "LIMIT", "product": product, "price": limit}
    return {"id": gid, "type": kind, "status": status, "orders": [order] * legs,
            "condition": {"exchange": "NSE", "tradingsymbol": symbol, "trigger_values": [trigger]}}


@pytest.fixture
def gtt_env(monkeypatch):
    """No Neon, no email: the notifications are captured; the kill switch is off unless a test turns it on."""
    from kite_connect.trading import gtt_stops

    sent = []
    monkeypatch.setattr(gtt_stops, "_order_hooks", lambda: (None, lambda *a, **k: sent.append(a[6])))
    monkeypatch.setattr(gtt_stops, "_kill_switch_active", lambda: False)
    return gtt_stops, sent


def test_a_new_stop_is_placed_on_the_tick_with_its_limit_two_percent_below(gtt_env):
    gs, sent = gtt_env
    k = _GttKite(ltp={"INFY": 500.0})
    r = gs.place_or_update_stop_gtt(k, "INFY", 10, 457.01, tick=0.05, limit_buffer_pct=2.0)
    assert (r["success"], r["action"], r["trigger"], r["limit"]) == (True, "placed", 457.05, 447.90)   # up (not 457.00), then down
    _, sym, trig, order = k.calls[0]
    assert (sym, trig, order["transaction_type"], order["product"], order["order_type"], order["quantity"]) == \
        ("INFY", 457.05, "SELL", "CNC", "LIMIT", 10)
    assert sent == ["GTT PLACED"]


def test_the_limit_never_sits_below_the_lower_circuit(gtt_env):
    gs, _ = gtt_env
    k = _GttKite(ltp={"INFY": 500.0})
    assert gs.place_or_update_stop_gtt(k, "INFY", 10, 457.05, tick=0.05, limit_buffer_pct=2.0,
                                       limit_floor=450.0)["limit"] == 450.0          # raised to the circuit
    assert gs.stop_limit_price(457.05, 2.0, 0.05, floor=460.0) == 447.90            # a floor above the trigger: ignored
    assert gs.stop_limit_price(0.05, 50.0, 0.05) == 0.05                             # never under one tick


def test_an_identical_stop_is_left_alone_and_a_changed_one_modified_in_place(gtt_env):
    gs, sent = gtt_env
    k = _GttKite(ltp={"INFY": 500.0}, gtts=[_gtt(7, "INFY", 457.05, 447.90, 10)])
    assert gs.place_or_update_stop_gtt(k, "INFY", 10, 457.05, tick=0.05, limit_buffer_pct=2.0)["action"] == "unchanged"
    assert k.calls == [] and sent == []
    r = gs.place_or_update_stop_gtt(k, "INFY", 10, 470.0, tick=0.05, limit_buffer_pct=2.0)
    assert (r["action"], r["trigger_id"], r["previous_trigger"]) == ("modified", 7, 457.05)
    assert k.calls[0][:3] == ("modify", 7, 470.0) and sent == ["GTT MODIFIED"]


def test_duplicate_stops_are_cut_to_one(gtt_env):
    gs, _ = gtt_env
    k = _GttKite(ltp={"INFY": 500.0}, gtts=[_gtt(7, "INFY", 457.05, 447.90, 10), _gtt(8, "INFY", 440.0, 431.2, 10)])
    r = gs.place_or_update_stop_gtt(k, "INFY", 10, 457.05, tick=0.05, limit_buffer_pct=2.0)
    assert r["action"] == "unchanged" and r["trigger_id"] == 7
    assert ("delete", 8) in k.calls and list(k.gtts) == [7]


@pytest.mark.parametrize("qty,trigger,ltp,why", [(10, 500.0, 500.0, "stop_breached"), (10, 510.0, 500.0, "stop_breached"),
                                                 (0, 450.0, 500.0, "positive"), (10, 0.0, 500.0, "positive"),
                                                 (2.5, 450.0, 500.0, "whole shares")])
def test_a_stop_that_cannot_protect_is_refused_and_nothing_is_sent(gtt_env, qty, trigger, ltp, why):
    gs, _ = gtt_env
    k = _GttKite(ltp={"INFY": ltp})
    r = gs.place_or_update_stop_gtt(k, "INFY", qty, trigger, tick=0.05)
    assert not r["success"] and r["action"] == "rejected" and why in str(r["error"])
    assert k.calls == []


def test_under_the_kill_switch_a_stop_protects_only_what_is_held(gtt_env, monkeypatch):
    gs, _ = gtt_env
    monkeypatch.setattr(gs, "_kill_switch_active", lambda: True)
    held = _GttKite(holdings=[{"tradingsymbol": "INFY", "exchange": "NSE", "quantity": 6, "t1_quantity": 0}],
                    ltp={"INFY": 500.0})
    r = gs.place_or_update_stop_gtt(held, "INFY", 10, 450.0, tick=0.05)
    assert r["action"] == "placed" and r["quantity"] == 6 and held.calls[0][3]["quantity"] == 6   # capped
    none = _GttKite(ltp={"INFY": 500.0})
    assert "no holding" in gs.place_or_update_stop_gtt(none, "INFY", 10, 450.0, tick=0.05)["error"]
    blind = _GttKite(ltp={"INFY": 500.0}, fail={"holdings"})
    assert "holdings check failed" in gs.place_or_update_stop_gtt(blind, "INFY", 10, 450.0, tick=0.05)["error"]
    assert none.calls == [] and blind.calls == []


def test_a_broker_refusal_is_reported_and_alerted(gtt_env):
    gs, sent = gtt_env
    k = _GttKite(ltp={"INFY": 500.0}, fail={"place_gtt"})
    r = gs.place_or_update_stop_gtt(k, "INFY", 10, 450.0, tick=0.05)
    assert not r["success"] and "place_gtt refused" in r["error"] and sent == ["GTT FAILED"]
    k2 = _GttKite(ltp={"INFY": 500.0}, fail={"get_gtts"})
    assert "get_gtts failed" in gs.place_or_update_stop_gtt(k2, "INFY", 10, 450.0, tick=0.05)["error"]


def test_an_unknown_tick_falls_back_to_the_coarser_slab_and_says_so(gtt_env, monkeypatch):
    from kite_connect.trading import nse_instruments

    gs, _ = gtt_env
    monkeypatch.setattr(nse_instruments, "_cache", {})
    r = gs.place_or_update_stop_gtt(_GttKite(ltp={"INFY": 500.0}), "INFY", 10, 457.03, limit_buffer_pct=2.0)
    assert r["tick_fallback"] and r["tick"] == 0.10 and r["trigger"] == 457.10      # Rs 250-1,000 slab is 0.05


def test_only_single_leg_cnc_sell_gtts_count_as_stops(gtt_env):
    gs, _ = gtt_env
    k = _GttKite(gtts=[_gtt(1, "A", 90, 88, 5), _gtt(2, "B", 90, 88, 5, side="BUY"), _gtt(3, "C", 90, 88, 5, product="MIS"),
                       _gtt(4, "D", 90, 88, 5, legs=2), _gtt(5, "E", 90, 88, 5, kind="two-leg"),
                       _gtt(6, "F", 90, 88, 5, status="triggered")])
    assert [g["symbol"] for g in gs.list_stop_gtts(k)] == ["A"]
    assert [g["symbol"] for g in gs.list_stop_gtts(k, active_only=False)] == ["A", "F"]
    assert gs.get_stop_gtt_status(k, 6)["status"] == "triggered" and gs.get_stop_gtt_status(k, 2) is None


def test_held_quantity_counts_t1_and_todays_cnc_but_not_intraday():
    from kite_connect.trading.gtt_stops import get_held_quantities

    k = _GttKite(holdings=[{"tradingsymbol": "A", "exchange": "NSE", "quantity": 10, "t1_quantity": 5},
                           {"tradingsymbol": "B", "exchange": "BSE", "quantity": 7, "t1_quantity": 0}],
                 positions=[{"tradingsymbol": "A", "exchange": "NSE", "product": "CNC", "quantity": -3},
                            {"tradingsymbol": "C", "exchange": "NSE", "product": "CNC", "quantity": 4},
                            {"tradingsymbol": "D", "exchange": "NSE", "product": "MIS", "quantity": 9}])
    assert get_held_quantities(k) == {"A": 12, "C": 4}


def test_reconcile_gives_every_holding_one_stop_and_clears_orphans(gtt_env):
    gs, _ = gtt_env
    holdings = [{"tradingsymbol": s, "exchange": "NSE", "quantity": q, "t1_quantity": 0}
                for s, q in (("A", 10), ("B", 5), ("C", 3), ("E", 4))]
    k = _GttKite(holdings=holdings, ltp={"A": 100.0, "B": 50.0, "C": 30.0, "E": 40.0},
                 gtts=[_gtt(1, "B", 45.0, 44.0, 5), _gtt(2, "D", 20.0, 19.6, 2), _gtt(3, "X", 10.0, 9.8, 1)])
    rep = gs.reconcile_stop_gtts(k, stops={"A": 90.0, "E": 41.0}, ticks={"A": 0.01, "B": 0.01, "C": 0.01, "E": 0.01},
                                 limit_buffer_pct=2.0, scope={"A", "B", "C", "D", "E"})
    assert [r["symbol"] for r in rep["placed"]] == ["A"]                     # a new stop at the desired trigger
    assert [(r["symbol"], r["trigger"]) for r in rep["modified"]] == [("B", 45.0)]   # own trigger kept, stale limit re-priced
    assert rep["missing_stop"] == ["C"]                                       # no stop known, no default
    assert [r["symbol"] for r in rep["breached"]] == ["E"]                   # 41 >= LTP 40: not placed, reported
    assert rep["deleted"] == [{"symbol": "D", "trigger_id": 2}] and 3 in k.gtts   # X is outside the scope: untouched
    assert rep["tick_fallback"] == [] and rep["errors"] == []


def test_reconcile_caps_a_book_at_the_broker_holding_and_fails_loudly_when_blind(gtt_env):
    gs, _ = gtt_env
    k = _GttKite(holdings=[{"tradingsymbol": "A", "exchange": "NSE", "quantity": 6, "t1_quantity": 0}], ltp={"A": 100.0})
    rep = gs.reconcile_stop_gtts(k, stops={"A": 90.0, "Z": 50.0}, quantities={"A": 10, "Z": 3}, ticks={"A": 0.01})
    assert rep["placed"] == [{"symbol": "A", "quantity": 6, "trigger": 90.0, "trigger_id": 901}]   # 10 asked, 6 held
    blind = gs.reconcile_stop_gtts(_GttKite(fail={"get_gtts"}), stops={"A": 90.0}, ticks={})
    assert blind["errors"][0]["symbol"] == "*" and "broker fetch failed" in blind["errors"][0]["error"]
    fallback = gs.reconcile_stop_gtts(k, stops={"A": 90.0}, ticks={})
    assert fallback["tick_fallback"] == ["A"]


def test_deleting_every_stop_for_a_symbol_counts_what_went(gtt_env):
    gs, _ = gtt_env
    k = _GttKite(gtts=[_gtt(1, "A", 90, 88, 5), _gtt(2, "A", 80, 78, 5), _gtt(3, "B", 70, 68, 5)])
    assert gs.delete_stop_gtts_for_symbol(k, "A", reason="exit") == 2 and list(k.gtts) == [3]
    assert gs.delete_stop_gtts_for_symbol(_GttKite(fail={"get_gtts"}), "A") == 0


# ---------------------------------------------------------------- AL5: alerts from Kite's order postbacks

def _postback(**over):
    base = {"order_id": "2611160001", "status": "COMPLETE", "tradingsymbol": "INFY", "exchange": "NSE",
            "transaction_type": "SELL", "order_type": "LIMIT", "product": "CNC", "variety": "regular",
            "quantity": 5, "filled_quantity": 5, "average_price": 1140.5, "price": 1135.8, "tag": None,
            "status_message": None}
    return {**base, **over}


def _triggered(gid, symbol, trigger, order_id, status="triggered"):
    g = _gtt(gid, symbol, trigger, trigger * 0.996, 5, status=status)
    g["orders"] = [{**g["orders"][0], "result": {"order_result": {"order_id": order_id, "status": "success"}}}]
    return g


def test_the_engines_own_tag_is_recognised():
    from kite_connect.trading.nse_engine_executor import EngineExecutor
    from kite_connect.trading.postback_alerts import is_engine_order

    assert is_engine_order(EngineExecutor.order_tag("2026-12-15", "BUY", "M&M"))        # NE261215BMM
    assert is_engine_order(EngineExecutor.order_tag("2026-12-15", "SELL", "BAJAJ-AUTO"))
    assert not any(is_engine_order(t) for t in (None, "", "manual", "NEWS", "NE2612B", "OPT261215BNIFTY"))


@pytest.mark.parametrize("status,reason,needle", [
    ("REJECTED", "Insufficient funds. Required margin is 5000", "not enough cash at the 09:00 check"),
    ("CANCELLED AMO", "", "no reason given"),
    ("CANCELLED", "Order cancelled by exchange", "cancelled by exchange"),
])
@pytest.mark.lesson('L-AL5',
                    pattern='F2',
                    date='2026-11-14',
                    source='AL5',
                    summary='A rejected or cancelled order was seen only in the evening run')
def test_a_failed_engine_order_is_critical_with_kites_reason(status, reason, needle):
    from kite_connect.trading.postback_alerts import classify
    from services.notifications.alerts import CRITICAL

    a = classify(_postback(status=status, status_message=reason, transaction_type="BUY", tag="NE261215BINFY",
                           filled_quantity=0))
    assert a["severity"] == CRITICAL and a["key"] == "postback_failed:2611160001"
    assert a["title"].endswith("BUY 5 INFY") and needle in " ".join(a["lines"])


@pytest.mark.parametrize("status", ["COMPLETE", "OPEN", "TRIGGER PENDING", "AMO REQ RECEIVED", "UPDATE"])
def test_an_engine_order_that_did_not_fail_raises_nothing(status):
    from kite_connect.trading.postback_alerts import classify

    called = []
    assert classify(_postback(status=status, tag="NE261215SINFY"), lambda: called.append(1) or {}) is None
    assert not called                                     # its own sell is never looked up as a stop


def test_a_filled_stop_warns_and_a_sale_by_hand_does_not():
    from kite_connect.trading import postback_alerts as pa
    from services.notifications.alerts import WARNING

    k = _GttKite(gtts=[_triggered(11, "INFY", 1120.0, "2611160001"), _gtt(12, "TCS", 2400, 2390, 2),
                       _triggered(13, "SBIN", 700.0, "", status="triggered"),         # triggered, no order named
                       _triggered(14, "LT", 3000.0, "2611169999", status="active")])  # not triggered
    stops = pa.triggered_stop_orders(k)
    assert list(stops) == ["2611160001"]
    a = pa.classify(_postback(), lambda: stops)
    assert a["severity"] == WARNING and a["key"] == "postback_stop:2611160001"
    assert a["title"] == "Centurion stop filled: SELL 5 INFY" and "trigger was 1120.0 (GTT 11" in a["lines"][0]
    assert pa.classify(_postback(order_id="2611160002"), lambda: stops) is None       # sold by hand


@pytest.mark.parametrize("over", [{"transaction_type": "BUY"}, {"product": "MIS"}, {"status": "OPEN"},
                                  {"order_id": ""}, {"status": None}])
def test_only_a_completed_cnc_sell_is_checked_against_the_stops(over):
    from kite_connect.trading.postback_alerts import classify

    called = []
    assert classify(_postback(**over), lambda: called.append(1) or {}) is None and not called


def test_an_unreadable_stop_list_still_warns_as_unconfirmed():
    from kite_connect.trading.postback_alerts import classify
    from services.notifications.alerts import WARNING

    for lookup in (lambda: None, None):
        a = classify(_postback(), lookup)
        assert a["severity"] == WARNING and "outside the engine's orders" in a["title"]
        assert "could not be read" in a["lines"][1]


def test_handle_emails_each_order_once_and_never_raises(monkeypatch):
    from kite_connect.trading import postback_alerts as pa
    from services.notifications import alerts

    sent = []
    monkeypatch.setattr(alerts, "alert", lambda sev, key, title, lines=(), book=None, once=False:
                        sent.append((sev, key, book, once)) or True)
    rejected = _postback(status="REJECTED", status_message="RMS: margin exceeds", tag="NE261215BINFY",
                         transaction_type="BUY")
    assert pa.handle(rejected, None) == "postback_failed:2611160001"
    assert sent == [(alerts.CRITICAL, "postback_failed:2611160001", "live", True)]
    assert pa.handle(_postback(tag="NE261215SINFY"), None) is None and len(sent) == 1   # a normal fill: silence
    boom = lambda: (_ for _ in ()).throw(RuntimeError("kite down"))
    assert pa.handle(_postback(), boom) is None                                          # a lookup that raises
    monkeypatch.setattr(alerts, "alert", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("smtp down")))
    assert pa.handle(rejected, None) is None                                             # an email that raises


def test_the_postback_route_answers_first_and_alerts_in_the_background(monkeypatch):
    pytest.importorskip("fastapi")                        # the web stack is not in the core install (CI)
    pytest.importorskip("httpx")
    import hashlib

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.routers import streaming
    from kite_connect.trading import postback_alerts as pa

    seen = []
    monkeypatch.setattr(pa, "handle", lambda p: seen.append(p))
    monkeypatch.setattr(streaming, "_get_dispatcher", lambda: type("D", (), {"dispatch": lambda self, e: None})())
    monkeypatch.setenv("ZERODHA_API_SECRET", "s3cret")
    app = FastAPI()
    app.include_router(streaming.router)
    body = {"order_id": "2611160001", "order_timestamp": "2026-12-15 09:00:02", "status": "REJECTED",
            "status_message": "Insufficient funds", "tradingsymbol": "INFY", "transaction_type": "BUY",
            "tag": "NE261215BINFY", "quantity": 5}
    good = hashlib.sha256(f"{body['order_id']}{body['order_timestamp']}s3cret".encode()).hexdigest()
    with TestClient(app) as client:
        assert client.post("/stream/postback", json={**body, "checksum": "bad"}).status_code == 403
        assert not seen                                   # a forged postback alerts nobody
        r = client.post("/stream/postback", json={**body, "checksum": good})
    assert r.status_code == 200 and r.json()["success"] is True
    assert len(seen) == 1 and seen[0]["tag"] == "NE261215BINFY" and seen[0]["status"] == "REJECTED"
