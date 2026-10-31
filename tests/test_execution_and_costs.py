"""Execution realism: fills come from prices the decision could not see, and cost money; every order passes one
gate, so a repeated request or a lost answer never trades twice (RB5); a halt reaches what is already at the broker,
while a read that fails adds no new risk (RB6); getting out is always possible (RB7); and a backtest fills as a broker
would, on prices that mean something (BT1)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from kite_connect.trading.gtt_stops import round_to_tick
from kite_connect.trading.nse_engine_executor import TICK, _tick
from kite_connect.trading.nse_instruments import SLABS, on_tick, round_price
from nse_engine.config import CostConfig
from nse_engine.costs import impact_bps, statutory_cost


DAY = "2025-06-02"
TICKS = tuple(t for _, t in SLABS)
#: NSE's slab edges (circular 33/2025) with a paisa and a tick either side, and the smallest prices, where float
#: division is least forgiving.
EDGES = sorted({round(b + d, 2) for b, _ in SLABS[:-1] for d in (-0.05, -0.01, 0.0, 0.01, 0.05)}
               | {0.01, 0.05, 0.10, 0.15, 0.30, 0.35})


def _prices(n: int = 4000, seed: int = 5) -> list:
    """Seeded prices from Rs 0.05 to 1.5 lakh at 0-3 decimals (on and off every grid), plus ``EDGES``."""
    rng = np.random.default_rng(seed)
    raw = np.exp(rng.uniform(np.log(0.05), np.log(150_000), n))
    return [round(float(p), int(d)) for p, d in zip(raw, rng.integers(0, 4, n))] + EDGES


@pytest.mark.parametrize("rounder", [round_price, round_to_tick], ids=["round_price", "round_to_tick"])
def test_tick_rounding_lands_on_the_grid_in_the_asked_direction(rounder):
    """Kite rejects a price off the tick (LN-T1): every order and GTT price lands on it, ``down`` never rises,
    ``up`` never falls, and nothing moves a tick or more (half a tick for ``nearest``)."""
    for p in _prices():
        for tick in TICKS:
            if p < tick:
                continue
            for mode, most in (("nearest", tick / 2), ("down", tick), ("up", tick)):
                r = rounder(p, tick, mode)
                assert on_tick(r, tick), (p, tick, mode, r)
                assert abs(r - p) <= most + 1e-9, (p, tick, mode, r)
                assert (mode != "down" or r <= p + 1e-9) and (mode != "up" or r >= p - 1e-9), (p, tick, mode, r)


def test_the_planned_limit_rounding_keeps_the_band_side():
    """``_tick`` (the plan's 0.05 grid): a sell limit rounds down, a buy limit up, by less than a tick."""
    for p in _prices():
        if p < TICK:
            continue
        down, up = _tick(p, "down"), _tick(p, "up")
        assert on_tick(down, TICK) and on_tick(up, TICK), p
        assert p - TICK - 1e-9 < down <= p + 1e-9 and p - 1e-9 <= up < p + TICK + 1e-9, (p, down, up)


def test_each_slab_tick_divides_the_next():
    """The coarser-slab fallback (``safe_tick``) is valid whichever neighbouring slab NSE assigned only because
    each tick is a whole multiple of the one below it."""
    assert all(on_tick(coarse, fine) for fine, coarse in zip(TICKS, TICKS[1:]))


def test_statutory_costs_are_charged_on_both_sides():
    buy = statutory_cost(100_000.0, "BUY", DAY)
    sell = statutory_cost(100_000.0, "SELL", DAY)
    assert buy > 0 and sell > 0
    assert sell > buy, "the sell side carries DP charges on top"


def test_costs_scale_with_traded_value():
    small = statutory_cost(10_000.0, "BUY", DAY)
    large = statutory_cost(1_000_000.0, "BUY", DAY)
    assert large > small * 50, "statutory charges are mostly proportional"


def test_impact_grows_with_participation():
    cfg = CostConfig()
    adv = 1e8
    light = impact_bps(1e5, adv, cfg)       # 0.1% of ADV
    heavy = impact_bps(5e6, adv, cfg)       # 5% of ADV
    assert cfg.spread_floor_bps <= light < heavy
    assert heavy > 2 * light, "square-root impact should grow with size"


def test_impact_never_below_the_spread_floor():
    cfg = CostConfig()
    assert impact_bps(1.0, 1e12, cfg) >= cfg.spread_floor_bps


@pytest.mark.slow
def test_backtest_fills_use_the_next_open_not_todays_close(store_path):
    """A one-day delay must change results; a same-day fill would make lag meaningless."""
    from nse_engine.deployment import load_deployment
    from nse_engine.engine import run_backtest
    from runners.run_nse_engine import _load_data

    cfg = load_deployment("config/nse_engine_deployed.json").engine.replace(
        start="2024-01-01", end="2024-12-31")
    data = _load_data(cfg, data_start="2022-01-03")
    base = run_backtest(data, cfg, record=False)
    lagged = run_backtest(data, cfg, record=False, lag_days=1)
    assert not np.allclose(base.returns.to_numpy(), lagged.returns.to_numpy(), atol=1e-12)


@pytest.mark.slow
def test_trading_costs_reduce_returns(store_path):
    """Doubling modelled impact must lower CAGR and raise the cost drag."""
    from nse_engine.deployment import load_deployment
    from nse_engine.engine import run_backtest
    from runners.run_nse_engine import _load_data

    cfg = load_deployment("config/nse_engine_deployed.json").engine.replace(
        start="2024-01-01", end="2024-12-31")
    data = _load_data(cfg, data_start="2022-01-03")
    cheap = run_backtest(data, cfg, record=False)
    dear = run_backtest(data, cfg.replace(**{
        "costs.impact_coefficient_bps": cfg.costs.impact_coefficient_bps * 3,
        "costs.spread_floor_bps": cfg.costs.spread_floor_bps * 3}), record=False)
    assert dear.metrics["cagr"] < cheap.metrics["cagr"]
    assert dear.metrics["cost_drag"] > cheap.metrics["cost_drag"]


# ── RB5: the order gate ──────────────────────────────────────────────────

class _Broker:
    """A broker that accepts orders into a book.  ``lose`` answers are lost after the order is accepted, ``refuse``
    calls fail before it is, and ``book_down`` makes the order book unreadable."""

    def __init__(self, book=None, lose=0, refuse=0, book_down=False):
        self.book, self.lose, self.refuse, self.book_down, self.sent = list(book or []), lose, refuse, book_down, 0

    def positions(self):
        return {"day": []}

    def orders(self):
        if self.book_down:
            raise ConnectionError("order book down")
        return self.book

    def order_history(self, order_id):
        return []

    def place_order(self, **params):
        if self.refuse:
            self.refuse -= 1
            raise ConnectionError("no route to host")
        self.sent += 1
        order_id = f"O{self.sent}"
        self.book.append({"order_id": order_id, "tag": params["tag"], "status": "OPEN"})
        if self.lose:
            self.lose -= 1
            raise TimeoutError("read timed out")
        return order_id


@pytest.fixture
def gate(monkeypatch):
    from kite_connect.trading import order_service as os_

    monkeypatch.setattr(os_, "_is_nse_market_open", lambda: True)
    monkeypatch.setattr(os_, "is_kill_switch_active", lambda: False)
    monkeypatch.setattr(os_, "_persist_to_db", lambda *a, **k: None)
    monkeypatch.setattr(os_, "_send_order_email", lambda *a, **k: None)
    monkeypatch.setattr(os_, "_order_circuit", None)
    monkeypatch.setattr(os_.time, "sleep", lambda *_: None)
    return os_


LIMIT = dict(symbol="RELIANCE", exchange="NSE", transaction_type="BUY", quantity=5, order_type="LIMIT", price=2500.0)


@pytest.mark.parametrize("order_type, price, trigger, refused", [
    ("MARKET", None, None, False), ("LIMIT", 100.5, None, False), ("SL", 100.0, 101.0, False),
    ("SL-M", None, 101.0, False), ("MARKET", float("nan"), None, False),       # a market order's price is only a reference
    ("LIMIT", None, None, True), ("LIMIT", 0, None, True), ("LIMIT", -5, None, True),
    ("LIMIT", float("nan"), None, True), ("LIMIT", float("inf"), None, True), ("LIMIT", "abc", None, True),
    ("SL", 100.0, None, True), ("SL", None, 101.0, True), ("SL-M", None, 0, True), ("BRACKET", 100.0, None, True),
])
def test_order_refusal_known_answers(order_type, price, trigger, refused):
    from kite_connect.trading.order_service import order_refusal

    assert bool(order_refusal(order_type, price, trigger)) is refused


def test_an_impossible_price_never_reaches_the_broker(gate):
    broker = _Broker()
    assert gate.place_order(broker, **{**LIMIT, "price": 0})["success"] is False and broker.sent == 0


def test_a_repeated_request_returns_the_first_order(gate):
    """The browser repeated a 5xx: the same tag must not trade twice."""
    broker = _Broker()
    first = gate.place_order(broker, **LIMIT, tag="abc123", dedupe=True)
    again = gate.place_order(broker, **LIMIT, tag="abc123", dedupe=True)
    assert broker.sent == 1 and again["order_id"] == first["order_id"] and again["duplicate"] is True


def test_a_rejected_order_may_be_placed_again(gate):
    broker = _Broker(book=[{"order_id": "O0", "tag": "abc123", "status": "REJECTED"}])
    assert gate.place_order(broker, **LIMIT, tag="abc123", dedupe=True)["order_id"] == "O1" and broker.sent == 1


def test_an_unreadable_book_sends_nothing_when_deduping(gate):
    broker = _Broker(book_down=True)
    out = gate.place_order(broker, **LIMIT, tag="abc123", dedupe=True)
    assert out["status"] == "UNKNOWN" and out["success"] is False and broker.sent == 0


def test_without_dedupe_the_first_send_reads_no_book(gate):
    """The engine checks its own tags: its path makes no extra call."""
    broker = _Broker(book_down=True)
    assert gate.place_order(broker, **LIMIT, tag="abc123")["success"] is True and broker.sent == 1


def test_a_lost_answer_is_not_sent_again(gate):
    broker = _Broker(lose=1)
    out = gate.place_order(broker, **LIMIT, tag="abc123")
    assert out["success"] is True and out["order_id"] == "O1" and broker.sent == 1


def test_the_last_attempt_is_checked_against_the_book(gate):
    """Every attempt failed, the last after reaching the exchange: the order is live, so it is not reported failed."""
    broker = _Broker(refuse=2, lose=1)
    out = gate.place_order(broker, **LIMIT, tag="abc123")
    assert out["success"] is True and out["order_id"] == "O1" and broker.sent == 1


def test_a_lost_last_answer_with_an_unreadable_book_is_unknown_not_failed(gate):
    class Blind(_Broker):
        def orders(self):
            if self.sent or self.refuse < 2:
                raise ConnectionError("order book down")
            return []

    out = gate.place_order(Blind(refuse=3), **LIMIT, tag="abc123")
    assert out["status"] == "UNKNOWN" and out["success"] is False


def test_a_refused_order_with_nothing_in_the_book_still_fails(gate):
    broker = _Broker(refuse=3)
    out = gate.place_order(broker, **LIMIT, tag="abc123")
    assert out["success"] is False and "status" not in out and broker.sent == 0


@pytest.fixture
def route(gate, monkeypatch):
    pytest.importorskip("fastapi")
    from api.routers.v1 import kite as route_

    monkeypatch.setattr(route_, "get_kite_session", lambda: route_._test_broker, raising=False)
    return route_


def _post(route_, broker, **fields):
    import asyncio

    route_._test_broker = broker
    body = dict(tradingsymbol="RELIANCE", exchange="NSE", transaction_type="BUY", quantity=5, order_type="LIMIT",
                price=2500.0, tag="abc123")
    return asyncio.run(route_.kite_place_order(route_.KiteOrderRequest(**{**body, **fields})))


def test_the_manual_route_goes_through_the_gate(route):
    """The route used to call the broker directly, so a browser retry of a 5xx sent a real order again."""
    broker = _Broker()
    first, again = _post(route, broker), _post(route, broker)
    assert broker.sent == 1 and first["order_id"] == again["order_id"] == "O1"
    assert first["duplicate"] is False and again["duplicate"] is True


def test_the_manual_route_answers_failures_with_a_4xx(route):
    """Never a 5xx, which the browser would repeat."""
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as bad_price:
        _post(route, _Broker(), price=0)
    with pytest.raises(HTTPException) as unknown:
        _post(route, _Broker(book_down=True))
    with pytest.raises(HTTPException) as refused:
        _post(route, _Broker(refuse=3))
    assert (bad_price.value.status_code, unknown.value.status_code, refused.value.status_code) == (422, 409, 422)


def test_the_manual_route_refuses_fractions_and_unknown_fields(route):
    from pydantic import ValidationError

    for bad in (dict(quantity=2.5), dict(quantity=0), dict(quantity=-1), dict(iceberg_legs=5)):
        with pytest.raises(ValidationError):
            _post(route, _Broker(), **bad)


# ── RB6: a halt reaches what is already at the broker ────────────────────

ENGINE_BUY, WEB_BUY, DEFAULT_BUY = "NE261030BRELIANCE", "m" + "a1" * 9 + "b", "0123456789abcdef0123"
NOT_OURS = (None, "", "gtt", "smallcase1", "NE261030SRELIANCE", "M" + "A1" * 9 + "B", "m123", "0123456789ABCDEF0123",
            "TWAP", "VWAP1a", "xNE261030BRELIANCE")


def _order(order_id, tag=ENGINE_BUY, side="BUY", status="AMO REQ RECEIVED", exchange="NSE", variety="amo", filled=0):
    return {"order_id": order_id, "tag": tag, "transaction_type": side, "status": status, "exchange": exchange,
            "variety": variety, "tradingsymbol": f"SYM{order_id}", "quantity": 5, "filled_quantity": filled}


class _Queue:
    """A broker holding ``book``.  An accepted cancel takes effect ``lag`` reads later, an id in ``stuck`` refuses it,
    an id in ``fills`` has executed that many shares when the cancel arrives, and reads past ``down_after`` fail."""

    def __init__(self, book=(), lag=0, stuck=(), fills=None, down_after=None):
        self.book, self.lag, self.stuck, self.fills, self.down_after = [dict(o) for o in book], lag, set(stuck), fills or {}, down_after
        self.cancels, self.reads, self.waiting = [], 0, {}

    def orders(self):
        self.reads += 1
        if self.down_after is not None and self.reads > self.down_after:
            raise ConnectionError("order book down")
        for order_id in list(self.waiting):
            self.waiting[order_id] -= 1
            if self.waiting[order_id] < 0:
                del self.waiting[order_id]
                order = next(o for o in self.book if o["order_id"] == order_id)
                order["status"] = "CANCELLED AMO" if order["variety"] == "amo" else "CANCELLED"
        return [dict(o) for o in self.book]

    def cancel_order(self, variety, order_id):
        self.cancels.append((variety, order_id))
        order = next(o for o in self.book if o["order_id"] == order_id)
        if order_id in self.stuck:
            raise RuntimeError("Order cannot be cancelled as it is being processed")
        order["filled_quantity"] = self.fills.get(order_id, 0)
        if order["filled_quantity"] >= order["quantity"]:
            order["status"] = "COMPLETE"
            raise RuntimeError("Order cannot be cancelled: it is complete")
        self.waiting.setdefault(order_id, self.lag)


def test_a_drain_cancels_centurions_working_buys_and_nothing_else(gate):
    """Each with its own variety, or Kite refuses the cancel of an after-market order."""
    ours = [_order("E"), _order("W", WEB_BUY, status="OPEN", variety="regular"),
            _order("D", DEFAULT_BUY, status="TRIGGER PENDING", variety="regular"),
            _order("T", "TWAP3", status="OPEN", variety="regular")]
    others = [_order(f"X{i}", tag) for i, tag in enumerate(NOT_OURS)] + [
        _order("S", "NE261030SRELIANCE", side="SELL"), _order("S2", WEB_BUY, side="SELL"),
        _order("H", DEFAULT_BUY, exchange="NFO"), _order("C", status="COMPLETE", filled=5),
        _order("R", status="REJECTED"), _order("A", status="CANCELLED AMO")]
    broker = _Queue(ours + others)
    result = gate.drain_buys(broker)
    assert sorted(broker.cancels) == [("amo", "E"), ("regular", "D"), ("regular", "T"), ("regular", "W")]
    assert (result["status"], result["found"], result["cancelled"], result["left"]) == ("CLEAR", 4, 4, [])


def test_a_drain_believes_the_book_not_the_cancels_reply(gate):
    slow, never = _Queue([_order("E")], lag=1), _Queue([_order("E")], lag=99)
    assert gate.drain_buys(slow)["status"] == "CLEAR"
    assert slow.reads > 2                                 # the first read back still showed it working
    result = gate.drain_buys(never)
    assert (result["status"], result["left"], result["cancelled"]) == ("LEFT", ["SYME"], 0)


def test_a_cancel_refused_is_reported_with_kites_reason(gate):
    broker = _Queue([_order("E"), _order("F")], stuck=["E"])
    result = gate.drain_buys(broker)
    assert (result["status"], result["left"], result["cancelled"]) == ("LEFT", ["SYME"], 1)
    assert "being processed" in result["error"]
    assert broker.reads == 4 and broker.cancels.count(("amo", "E")) == 3 and broker.cancels.count(("amo", "F")) == 1


@pytest.mark.parametrize("down_after, found, cancels", [(0, 0, 0), (1, 1, 1)], ids=["before", "after the cancel"])
def test_an_unreadable_book_is_unknown_never_clear(gate, down_after, found, cancels):
    broker = _Queue([_order("E")], down_after=down_after)
    result = gate.drain_buys(broker)
    assert (result["status"], result["found"], len(broker.cancels)) == ("UNKNOWN", found, cancels)
    assert "order book down" in result["error"]


def test_an_empty_book_is_clear_and_a_second_drain_finds_nothing(gate):
    assert gate.drain_buys(_Queue())["status"] == "CLEAR"
    broker = _Queue([_order("E")])
    gate.drain_buys(broker)
    again = gate.drain_buys(broker)
    assert (again["status"], again["found"], again["cancelled"], len(broker.cancels)) == ("CLEAR", 0, 0, 1)


def test_a_check_reads_the_book_and_cancels_nothing(gate):
    broker = _Queue([_order("E")])
    result = gate.drain_buys(broker, dry_run=True)
    assert (result["status"], result["found"], result["left"], result["cancelled"]) == ("CHECKED", 1, ["SYME"], 0)
    assert broker.cancels == [] and broker.reads == 1


def test_shares_that_executed_first_are_named_never_called_cancelled(gate):
    broker = _Queue([_order("E"), _order("P")], fills={"E": 5, "P": 2})
    result = gate.drain_buys(broker)
    assert (result["status"], result["found"], result["cancelled"], result["filled"]) == ("CLEAR", 2, 1, ["SYME", "SYMP"])


def test_a_drain_keeps_its_promises_on_random_books(gate):
    """500 seeded books of every side, tag, status and segment: what is not ours is never cancelled, ours never stays
    working, and the counts add up."""
    rng = np.random.default_rng(6)
    tags = (ENGINE_BUY, WEB_BUY, DEFAULT_BUY, "VWAP2") + NOT_OURS
    statuses = ("AMO REQ RECEIVED", "OPEN", "TRIGGER PENDING", "VALIDATION PENDING", "COMPLETE", "REJECTED", "CANCELLED")
    for _ in range(500):
        book = [_order(f"O{i}", tags[rng.integers(len(tags))], ("BUY", "SELL")[rng.integers(2)],
                       statuses[rng.integers(len(statuses))], ("NSE", "BSE", "NFO")[rng.integers(3)],
                       ("amo", "regular")[rng.integers(2)]) for i in range(int(rng.integers(0, 12)))]
        ours = {o["order_id"] for o in book if gate._drainable(o)}
        assert ours == {o["order_id"] for o in book if o["transaction_type"] == "BUY" and o["exchange"] != "NFO"
                        and o["status"] not in ("COMPLETE", "REJECTED", "CANCELLED") and o["tag"] not in NOT_OURS}
        broker = _Queue(book, lag=int(rng.integers(0, 2)))
        result = gate.drain_buys(broker)
        assert {order_id for _, order_id in broker.cancels} == ours
        assert not any(gate._drainable(o) for o in broker.orders())
        assert (result["status"], result["found"], result["cancelled"], result["left"]) == ("CLEAR", len(ours), len(ours), [])


class _Neon:
    def __init__(self):
        self.values = {}

    def sync_state(self, values):
        self.values.update(values)
        return True

    def read_state(self):
        return dict(self.values)


@pytest.fixture
def switch(gate, monkeypatch):
    """The kill switch on a fake Neon, with ``sessions`` as each account's Kite session and ``mail`` the emails."""
    from types import SimpleNamespace

    from kite_connect.auth import accounts, daily_login
    from kite_connect.trading import kill_switch as ks
    from services.notifications import alerts

    sessions, mail, neon = {}, [], _Neon()
    listed = [accounts.Account(accounts.PRIMARY, "You", "", "key")]
    others = [accounts.Account("ann", "Ann", "AB1234", "key1", mode="live"),
              accounts.Account("bo", "Bo", "CD1234", "key2", mode="dry_run")]

    def session(book, key=None):
        found = sessions.get(book.id)
        if isinstance(found, Exception):
            raise found
        return found

    monkeypatch.setattr(ks, "_store", lambda: neon)
    monkeypatch.setattr(accounts, "all_accounts", lambda: listed)
    monkeypatch.setattr(accounts, "account_book", lambda acct: acct)
    monkeypatch.setattr(daily_login, "kite_from_stored_token", session)
    monkeypatch.setattr(alerts, "alert", lambda severity, key, title, lines, **_: mail.append((title, " ".join(lines))))
    return SimpleNamespace(ks=ks, sessions=sessions, mail=mail, order_service=gate, listed=listed, others=others)


def test_turning_it_on_clears_the_queue_and_says_so(switch, capsys):
    switch.sessions["primary"] = broker = _Queue([_order("E")])
    assert switch.ks.main(["on", "--reason", "drill"]) == 0
    assert switch.ks.state()["on"] is True and broker.cancels == [("amo", "E")]
    title, body = switch.mail[0]
    assert title == "Centurion kill switch ON" and "1 queued buy(s) cancelled and checked gone" in body
    assert "broker: 1 buys found, 1 cancelled, 0 open, 0 account(s) not checked" in capsys.readouterr().out


def test_the_latch_comes_first_and_holds_whatever_kite_does(switch):
    latched = []

    class Failing(_Queue):
        def orders(self):
            latched.append(switch.ks.state()["on"])
            raise ConnectionError("no route to Kite")

    switch.sessions["primary"] = Failing()
    assert switch.ks.main(["on", "--reason", "drill"]) == 1
    assert latched == [True] and switch.ks.state()["on"] is True
    title, body = switch.mail[0]
    assert title.endswith("CHECK KITE NOW") and "NOT CHECKED at the broker (You)" in body and "action drain" in body


def test_a_missing_client_or_token_key_is_loud_and_the_latch_holds(switch):
    switch.sessions["primary"] = ImportError("No module named 'kiteconnect'")
    assert switch.ks.main(["on", "--reason", "drill"]) == 1
    assert switch.ks.state()["on"] is True and "kiteconnect" in switch.mail[0][1]


def test_the_email_says_when_the_queue_was_empty_and_what_executed_first(switch):
    switch.sessions["primary"] = _Queue()
    assert switch.ks.main(["on", "--reason", "drill"]) == 0
    assert "Broker (You): no queued buys." in switch.mail[0][1]
    switch.sessions["primary"] = _Queue([_order("E")], fills={"E": 5})
    assert switch.ks.main(["on", "--reason", "drill"]) == 0
    assert "Executed before the cancel (You): SYME." in switch.mail[1][1]


def test_an_order_left_open_names_the_symbol_and_fails_the_run(switch):
    switch.sessions["primary"] = _Queue([_order("E")], stuck=["E"])
    assert switch.ks.main(["on", "--reason", "drill"]) == 1
    title, body = switch.mail[0]
    assert title.endswith("CHECK KITE NOW") and "STILL OPEN at the broker (You): SYME" in body and "being processed" in body


def test_no_login_is_loud_for_you_and_a_live_account_only(switch):
    switch.listed += switch.others
    results = switch.ks.drain()
    assert [(r["account"], r["status"], r["error"]) for r in results] == [
        ("You", "UNKNOWN", "no Kite login today"), ("Ann", "UNKNOWN", "no Kite login today")]


def test_every_account_with_a_login_is_drained_and_one_failure_stops_none(switch):
    switch.listed += switch.others
    switch.sessions.update(primary=RuntimeError("token key missing"), ann=(ann := _Queue([_order("E")])),
                           bo=(bo := _Queue([_order("F")])))
    results = {r["account"]: r["status"] for r in switch.ks.drain()}
    assert results == {"You": "UNKNOWN", "Ann": "CLEAR", "Bo": "CLEAR"} and ann.cancels and bo.cancels


def test_an_unreadable_account_list_is_reported_not_raised(switch, monkeypatch):
    from kite_connect.auth import accounts

    def down():
        raise ConnectionError("neon down")

    monkeypatch.setattr(accounts, "all_accounts", down)
    assert [r["status"] for r in switch.ks.drain()] == ["UNKNOWN"]


def test_off_and_status_never_touch_the_broker(switch):
    switch.sessions["primary"] = broker = _Queue([_order("E")])
    assert switch.ks.main(["status"]) == 0 and switch.ks.main(["off", "--reason", "done"]) == 0
    assert broker.reads == 0 and switch.mail[0][0] == "Centurion kill switch OFF"
    with pytest.raises(ValueError):
        switch.ks.set_switch(True, " ")
    assert broker.reads == 0


def test_a_drain_needs_the_switch_on_and_then_reports(switch, monkeypatch, capsys):
    switch.sessions["primary"] = broker = _Queue([_order("E")])
    assert switch.ks.main(["drain"]) == 1 and broker.reads == 0 and switch.mail == []
    monkeypatch.setattr(switch.order_service, "is_kill_switch_active", lambda: True)
    assert switch.ks.main(["drain"]) == 0 and broker.cancels == [("amo", "E")]
    assert switch.mail[0][0] == "Centurion kill switch drain" and "broker: 1 buys found" in capsys.readouterr().out


def test_a_check_cancels_nothing_and_sends_no_email(switch, capsys):
    switch.sessions["primary"] = broker = _Queue([_order("E")])
    assert switch.ks.main(["check"]) == 0
    assert broker.cancels == [] and switch.mail == [] and "1 buys found, 0 cancelled, 1 open" in capsys.readouterr().out
    switch.sessions["primary"] = None
    assert switch.ks.main(["check"]) == 1                 # the drill shows when Kite cannot be reached


def test_the_public_log_holds_counts_only(switch, capsys):
    """Actions logs are public: no symbol, account name or broker text in the output, wherever the run ends."""
    switch.listed += switch.others
    switch.sessions.update(primary=_Queue([_order("E")], stuck=["E"]), ann=RuntimeError("socks5h://user:pw@proxy down"))
    switch.ks.main(["on", "--reason", "private reason"])
    out = capsys.readouterr().out
    assert not any(word in out for word in ("SYME", "Ann", "You", "socks5h", "private reason", "being processed"))


@pytest.mark.parametrize("status, filled, outcome", [
    ("CANCELLED AMO", 0, "cancelled"), ("CANCELLED", 0, "cancelled"), ("CANCELLED", 2, "partial"),
    ("REJECTED", 0, "rejected"), ("AMO REQ RECEIVED", 0, "open"), ("COMPLETE", 5, "complete")])
def test_a_drained_after_market_order_reads_cancelled_the_next_evening(status, filled, outcome):
    from kite_connect.trading.live_session import fills_from_outcomes
    from kite_connect.trading.nse_engine_executor import live_order_outcomes

    outcomes = live_order_outcomes(_Queue([_order("E", status=status, filled=filled)]), "2026-10-30")
    assert outcomes[0]["outcome"] == outcome
    rows = fills_from_outcomes(outcomes, {}, "2026-10-31", "2026-10-30", {}, 15.34)
    assert rows[0]["status"] == ("FILLED" if filled else "REJECTED" if outcome == "rejected" else "CANCELLED")


class _NoPositions:
    def positions(self):
        raise ConnectionError("positions down")


def test_unreadable_positions_leave_no_capital_for_new_buys():
    from kite_connect.trading.risk_manager import RiskConfig, RiskManager

    class Holding:
        def positions(self):
            return {"net": [{"quantity": 10, "average_price": 500.0}, {"quantity": 0, "average_price": 900.0}]}

    cfg = RiskConfig(total_capital=100_000)
    assert RiskManager(cfg)._available_capital() == 100_000                        # research: no broker to ask
    assert RiskManager(cfg, kite=Holding())._available_capital() == 95_000
    assert RiskManager(cfg, kite=Holding(), max_deployment_cap=60_000)._available_capital() == 55_000
    blind = RiskManager(cfg, kite=_NoPositions())
    assert blind._available_capital() == 0.0
    blind._get_regime_scale = lambda: 1.0
    assert blind.plan_trades(pd.DataFrame({"symbol": ["RELIANCE"]})) == []


def _executor(monkeypatch, broker):
    """The real ``_place_orders`` over ``broker``, its order calls recorded in ``sent``; the checks that need market
    data or a clock pass."""
    from kite_connect.trading import order_service
    from kite_connect.trading import auto_executor as ae

    sent = []

    def place(**kwargs):
        sent.append((kwargs["transaction_type"], kwargs["symbol"]))
        return {"success": True, "order_id": f"O{len(sent)}"}

    monkeypatch.setattr(order_service, "place_order", place)
    monkeypatch.setattr(ae, "_ORDER_DELAY_S", 0)
    ex = ae.AutoExecutor.__new__(ae.AutoExecutor)
    ex.kite, ex._vol_target = broker, None
    ex._trade_monitor = type("Monitor", (), {"kite": None, "register_trade": lambda self, trade: None})()
    ex._is_nse_market_open = lambda: True
    ex._get_earnings_blackout_symbols = lambda symbols: set()
    ex._is_at_circuit_limit = lambda symbol: False
    ex._journal_entry = ex._notify_order = ex._notify_order_failure = lambda *a, **k: None
    return ex, sent


class _Desk:
    """Everything ``_place_orders`` reads from Kite, with the order book up or down."""

    def __init__(self, book=None):
        self.book = book

    def positions(self):
        return {"day": []}

    def profile(self):
        return {}

    def ltp(self, keys):
        return {}

    def orders(self):
        if self.book is None:
            raise ConnectionError("order book down")
        return self.book


def _plan(symbol, side):
    from kite_connect.trading.risk_manager import TradePlan

    return TradePlan(symbol, side, 100.0, 95.0, 110.0, 5, 25.0, 50.0, 2.0, 1.0)


def test_an_unreadable_book_holds_back_buys_and_lets_exits_go(monkeypatch):
    ex, sent = _executor(monkeypatch, _Desk(book=None))
    results = ex._place_orders([_plan("RELIANCE", "BUY"), _plan("TCS", "SELL")], lambda _: None)
    assert sent == [("SELL", "TCS")]
    held = next(r for r in results if r.symbol == "RELIANCE")
    assert held.success is False and "duplicate" in held.error


def test_a_readable_book_still_skips_a_symbol_with_an_open_order(monkeypatch):
    ex, sent = _executor(monkeypatch, _Desk(book=[{"status": "OPEN", "transaction_type": "BUY", "tradingsymbol": "RELIANCE"}]))
    results = ex._place_orders([_plan("RELIANCE", "BUY"), _plan("INFY", "BUY")], lambda _: None)
    assert sent == [("BUY", "INFY")] and results[0].error.startswith("Duplicate")


# ── RB7: getting out is always possible ──────────────────────────────────

EXIT = dict(exchange="NSE", transaction_type="SELL", quantity=5, order_type="LIMIT", price=100.0, variety="amo", is_exit=True)


@pytest.fixture
def breaker(gate, monkeypatch):
    """The real circuit breaker, as ``order_service`` sets it up."""
    from infrastructure.fault_isolation import CircuitBreaker

    monkeypatch.setattr(gate, "_order_circuit", CircuitBreaker("kite_orders", failure_threshold=3, reset_timeout=600.0))
    return gate._order_circuit


def test_one_blip_on_one_sell_does_not_stop_the_nights_other_sells(gate, breaker):
    """Three failed sends of one order open the breaker; the exits after it must still reach Kite."""
    broker = _Broker(refuse=3)
    results = [gate.place_order(broker, symbol=s, tag=f"NE261030S{s}", **EXIT) for s in ("AAA", "BBB", "CCC", "DDD")]
    assert [r["success"] for r in results] == [False, True, True, True] and broker.sent == 3
    assert breaker.state == "CLOSED"                      # the exit that got through proves Kite is back


def test_an_open_breaker_still_holds_back_new_buys(gate, breaker):
    for _ in range(3):
        breaker._on_failure()
    broker = _Broker()
    result = gate.place_order(broker, **LIMIT)
    assert result["success"] is False and "Circuit breaker OPEN" in result["error"] and broker.sent == 0
    assert gate.place_order(broker, symbol="TCS", tag="NE261030STCS", **EXIT)["success"] is True and broker.sent == 1


def test_a_manual_sell_still_goes_while_the_kill_switch_is_on(route, gate, monkeypatch):
    monkeypatch.setattr(gate, "is_kill_switch_active", lambda: True)
    broker = _Broker()
    assert _post(route, broker, transaction_type="SELL", tag="sell123")["order_id"] == "O1"
    with pytest.raises(route.HTTPException) as refused:
        _post(route, broker, tag="buy123")
    assert refused.value.status_code == 423 and broker.sent == 1


def test_a_manual_buy_waits_for_an_open_breaker(route, breaker):
    for _ in range(3):
        breaker._on_failure()
    with pytest.raises(route.HTTPException) as refused:
        _post(route, _Broker())
    assert refused.value.status_code == 422 and "Circuit breaker OPEN" in refused.value.detail


@pytest.mark.parametrize("stamp, is_open", [
    ("2026-11-02 10:00", True), ("2026-11-02 09:14", False), ("2026-11-02 15:31", False),
    ("2026-11-07 10:00", False),                          # a Saturday
    ("2026-11-10 10:00", False),                          # a weekday holiday
    ("2026-02-01 10:00", True),                           # Budget day: a Sunday with a full session
    ("2026-11-08 10:00", False),                          # Muhurat: its hour is in the evening
], ids=["open", "before", "after", "saturday", "holiday", "budget-sunday", "muhurat"])
def test_the_market_clock_reads_the_nse_calendar(stamp, is_open):
    from datetime import datetime, timedelta, timezone

    from kite_connect.trading.order_service import _is_nse_market_open

    ist = timezone(timedelta(hours=5, minutes=30))
    assert _is_nse_market_open(datetime.fromisoformat(stamp).replace(tzinfo=ist)) is is_open


def test_a_missing_holiday_list_is_named_before_it_matters(monkeypatch):
    from datetime import date

    from nse_engine import nse_calendar as cal

    monkeypatch.setattr(cal, "HOLIDAY_YEARS", (2025, 2026))
    days = (date(2026, 10, 30), date(2026, 12, 14), date(2026, 12, 15), date(2027, 1, 4))
    assert [cal.holiday_list_missing(d) for d in days] == [None, None, 2027, 2027]
    monkeypatch.setattr(cal, "HOLIDAY_YEARS", (2025, 2026, 2027))
    assert [cal.holiday_list_missing(d) for d in days] == [None, None, None, None]


def test_the_owner_hears_about_a_missing_holiday_list(monkeypatch):
    from datetime import date

    from kite_connect.trading import nse_engine_executor as ex
    from nse_engine import nse_calendar as cal
    from services.notifications import alerts

    mail = []
    monkeypatch.setattr(alerts, "alert", lambda severity, key, title, lines, once=False: mail.append((key, title, once)))
    monkeypatch.setattr(cal, "HOLIDAY_YEARS", (2025, 2026))
    assert ex.warn_if_no_holiday_list(date(2026, 10, 30)) is None and mail == []
    assert ex.warn_if_no_holiday_list(date(2027, 1, 4)) == 2027
    assert mail == [("holiday_list:2027", "Centurion: NSE's 2027 holiday list is missing", True)]


def test_each_live_plan_asks_for_the_holiday_list(monkeypatch):
    from types import SimpleNamespace

    from kite_connect.trading import nse_engine_executor as ex

    class Reached(Exception):
        pass

    def reached(*_):
        raise Reached

    asked = []
    monkeypatch.setattr(ex, "warn_if_no_holiday_list", lambda: asked.append(True))
    monkeypatch.setattr(ex, "presence_gaps", reached)     # the next step: the plan's own work needs no market here
    monkeypatch.setattr(ex.EngineExecutor, "_cfg", lambda self: SimpleNamespace(portfolio=None))
    monkeypatch.setattr(ex.EngineExecutor, "_current_book", lambda self: ({}, 0.0))
    today = pd.Timestamp.now(tz="Asia/Kolkata").normalize().tz_localize(None)
    with pytest.raises(Reached):
        ex.EngineExecutor.__new__(ex.EngineExecutor).plan(data=SimpleNamespace(dates=[today], until=lambda day: None))
    assert asked == [True]


# ── BT1: the single-stock backtest (strategies/, trading_strategies/, tools/strategy_benchmark.py) ──

def _bars(closes, opens=None, start="2024-01-01"):
    closes = np.asarray(closes, dtype="float64")
    opens = closes if opens is None else np.asarray(opens, dtype="float64")
    df = pd.DataFrame({"Open": opens, "High": np.maximum(opens, closes) * 1.01, "Low": np.minimum(opens, closes) * 0.99,
                       "Close": closes, "Volume": 1e6}, index=pd.bdate_range(start, periods=len(closes)))
    df.attrs["market"] = "IND"
    return df


def _plain():
    from strategies.base_strategy import BaseStrategy

    class Plain(BaseStrategy):
        def run(self, *a, **k):
            raise NotImplementedError

    return Plain()


def _risk(share=1.0):
    from strategies.base_strategy import RiskParams

    return RiskParams(max_position_size=share)


def _fees(strategy, value, side, day):
    return strategy._trade_cost(value, side, pd.Timestamp(day), "IND")


def test_a_decision_fills_at_the_next_open_never_its_own_close():
    """Decided on the 2nd close (100), bought at the 3rd open (110), though the close that decided was cheaper."""
    s = _plain()
    df = _bars([100, 100, 120, 120], opens=[100, 100, 110, 120])
    df["positions"] = [0, 1, 1, 1]
    p = s._fill_positions(df, df["positions"], 100_000, _risk())
    assert p["shares"].tolist() == [0, 0, 906, 906]               # 909 at 110 would leave Rs 10 for about Rs 340 of costs
    day = df.index[2]
    assert p["cash"].iloc[2] == pytest.approx(100_000 - 906 * 110 - _fees(s, 906 * 110, "BUY", day))
    assert p["cash"].iloc[2] >= 0
    assert p["total_value"].iloc[-1] == pytest.approx(p["cash"].iloc[-1] + 906 * 120)


def test_a_sale_fills_at_the_next_open_too_and_pays_both_ways():
    s = _plain()
    df = _bars([100, 100, 100, 130, 90], opens=[100, 100, 100, 100, 125])
    df["positions"] = [0, 1, 1, 0, 0]
    p = s._fill_positions(df, df["positions"], 100_000, _risk())
    buy, sell = df.index[2], df.index[4]
    qty = int(p["shares"].max())
    want = 100_000 - qty * 100 - _fees(s, qty * 100, "BUY", buy) + qty * 125 - _fees(s, qty * 125, "SELL", sell)
    assert p["total_value"].iloc[-1] == pytest.approx(want)       # sold at 125, not at the 130 close or the 90 after it
    assert p["cumulative_cost"].iloc[-1] > 0.002 * qty * 100


def test_a_flat_strategy_is_worth_its_capital_and_pays_nothing():
    df = _bars([100, 50, 200, 150])
    df["positions"] = 0
    p = _plain()._fill_positions(df, df["positions"], 100_000, _risk())
    assert p["total_value"].tolist() == [100_000.0] * 4 and p["cumulative_cost"].iloc[-1] == 0


def test_the_share_count_does_not_know_the_future():
    """The old sizing divided by the highest close of the whole sample."""
    calm = _bars([100.0] * 6)
    soar = _bars([100.0, 100, 100, 100, 100, 10_000])
    for df in (calm, soar):
        df["positions"] = [0, 1, 1, 1, 1, 1]
    s = _plain()
    a = s._fill_positions(calm, calm["positions"], 100_000, _risk(0.25))
    b = s._fill_positions(soar, soar["positions"], 100_000, _risk(0.25))
    assert a["shares"].iloc[-1] == b["shares"].iloc[-1] == 250


def test_position_size_is_a_share_of_the_equity_so_gains_compound():
    s = _plain()
    df = _bars([100, 100, 100, 200, 200, 200, 200], opens=[100, 100, 100, 200, 200, 200, 200])
    df["positions"] = [0, 1, 1, 0, 1, 1, 1]
    p = s._fill_positions(df, df["positions"], 100_000, _risk(0.5))
    first, second = int(p["shares"].iloc[2]), int(p["shares"].iloc[-1])
    assert first == 500 and second > first / 2                    # half the bigger equity, not of the first capital
    assert second == int(p["cash"].iloc[4] * 0.5 // 200)


def test_a_missing_open_holds_the_fill_to_the_next_session():
    df = _bars([100, 100, 100, 100, 100], opens=[100, 100, np.nan, np.nan, 104])
    df["positions"] = [0, 1, 1, 1, 1]
    p = _plain()._fill_positions(df, df["positions"], 100_000, _risk())
    assert p["shares"].tolist()[:4] == [0, 0, 0, 0] and p["shares"].iloc[4] == 958     # 104 and costs, in cash


def test_the_last_close_decision_has_nothing_to_fill_at():
    df = _bars([100, 100, 100])
    df["positions"] = [0, 0, 1]
    assert _plain()._fill_positions(df, df["positions"], 100_000, _risk())["shares"].iloc[-1] == 0


def test_a_share_above_the_cap_still_trades_and_never_overdraws():
    df = _bars([900.0, 900, 900])
    df["positions"] = [1, 1, 1]
    p = _plain()._fill_positions(df, df["positions"], 1_000, _risk(0.25))
    assert p["shares"].iloc[-1] == 1 and p["cash"].min() >= 0     # the old rule: one share while the cash covers it
    broke = _plain()._fill_positions(df, df["positions"], 500, _risk(0.25))
    assert broke["shares"].iloc[-1] == 0 and broke["cash"].iloc[-1] == 500


def test_a_long_only_book_never_goes_short():
    """A sell-side decision closes a long; delivery cannot open a short."""
    df = _bars([100.0, 100, 100, 100, 100])
    df["positions"] = [0, 1, -1, -1, -1]
    p = _plain()._calculate_portfolio_long_only(df, 100_000, _risk())
    assert p["shares"].min() == 0 and p["shares"].iloc[2] > 0 and p["shares"].iloc[-1] == 0
    assert (_plain()._calculate_portfolio(df, 100_000, _risk())["shares"].iloc[-1] < 0)     # the default still may


def test_a_portfolio_without_open_prices_is_refused():
    df = _bars([100, 100]).drop(columns=["Open"])
    with pytest.raises(ValueError, match="Open"):
        _plain()._fill_positions(df, pd.Series([0, 1], index=df.index), 100_000, _risk())


def test_us_bars_pay_us_costs():
    s = _plain()
    day = pd.Timestamp("2024-01-02")
    assert s._trade_cost(100_000, "BUY", day, "US") == pytest.approx(100_000 * 5 / 1e4)    # no STT, 5 bp of slippage
    assert s._trade_cost(100_000, "BUY", day, "IND") > 3 * s._trade_cost(100_000, "BUY", day, "US")


@pytest.mark.parametrize("seed", range(4))
def test_the_portfolio_never_reads_a_later_bar_and_keeps_its_books(seed):
    """Seeded random paths and decisions: values to a cut survive any change after it, shares stay whole, cash stays
    positive, and value always equals cash plus holdings."""
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300)))
    df = _bars(c, opens=c * (1 + rng.normal(0, 0.005, 300)))
    df["positions"] = (rng.random(300) > 0.7).astype(int)
    s = _plain()
    a = s._fill_positions(df, df["positions"], 100_000, _risk())
    cut = 200
    later = df.copy()
    k = rng.uniform(0.2, 5.0, 300 - cut - 1)
    for col in ("Open", "High", "Low", "Close"):
        later.iloc[cut + 1:, later.columns.get_loc(col)] = later[col].iloc[cut + 1:].to_numpy() * k
    later.iloc[cut + 1:, later.columns.get_loc("positions")] = rng.integers(0, 2, 300 - cut - 1)
    b = s._fill_positions(later, later["positions"], 100_000, _risk())
    pd.testing.assert_frame_equal(a.iloc[:cut + 1], b.iloc[:cut + 1])
    assert (a["shares"] == a["shares"].round()).all() and (a["cash"] >= 0).all()
    assert (a["total_value"] - a["holdings"] - a["cash"]).abs().max() < 1e-6


def test_metrics_measure_the_strategy_never_the_stock():
    """Swing Combo's equity column was missed, so its metrics were the stock's: a -15.8% run reported -99%."""
    s = _plain()
    df = _bars([1_000.0, 1_000, 1_000])
    book = pd.DataFrame({"close": [1000.0, 1000, 1000], "equity": [100_000.0, 101_000, 102_000]}, index=df.index)
    m = s.calculate_metrics(book, df, 100_000)
    assert m["total_return"] == pytest.approx(2.0) and m["final_value"] == 102_000
    assert "calculation_error" in s.calculate_metrics(book[["close"]], df, 100_000)      # no value column: no number


def test_sharpe_uses_the_markets_risk_free_rate():
    from config import Config

    s = _plain()
    df = _bars([100.0, 100, 100])
    assert s._risk_free_rate(df) == Config.RISK_FREE_RATE_IND
    df.attrs["market"] = "US"
    assert s._risk_free_rate(df) == Config.RISK_FREE_RATE_US
    df.attrs.pop("market")
    assert s._risk_free_rate(df) == 0.02


def test_a_raw_price_jump_is_a_corporate_action():
    from strategies.data_service import raw_price_jump

    assert raw_price_jump(_bars([100.0, 102, 104])) is None
    assert raw_price_jump(_bars([100.0, 100, 100], opens=[100, 120, 100])) is None      # the top of the widest band
    df = _bars([100.0, 100, 50, 51], opens=[100, 100, 50, 51])
    assert raw_price_jump(df) == df.index[2]


@pytest.fixture
def service(monkeypatch):
    from strategies.data_service import DataService

    d = DataService()
    d.clear_cache()
    monkeypatch.setattr("utils.download_ind_ohlcv", lambda t, start, end: _raw[t].copy())
    return d


_raw = {"BONUS.NS": _bars([100.0, 100, 50, 51], opens=[100, 100, 50, 51]), "CALM.NS": _bars([100.0, 101, 102, 103])}


def test_raw_prices_through_a_bonus_give_way_to_adjusted_ones(service, monkeypatch):
    adjusted = _bars([50.0, 50, 50, 51])
    monkeypatch.setattr(service, "_fetch_from_yfinance", lambda *a: adjusted.copy())
    df = service.get_ohlcv("BONUS.NS", "2024-01-01", "2024-01-05")
    assert df["Close"].tolist() == [50.0, 50, 50, 51] and df.attrs["market"] == "IND"


@pytest.mark.parametrize("yahoo", [lambda *a: pd.DataFrame(), lambda *a: _raw["BONUS.NS"].copy()], ids=["empty", "still raw"])
def test_a_bonus_with_no_adjusted_prices_is_refused_never_run(service, monkeypatch, yahoo):
    monkeypatch.setattr(service, "_fetch_from_yfinance", yahoo)
    with pytest.raises(ValueError, match="refused"):
        service.get_ohlcv("BONUS.NS", "2024-01-01", "2024-01-05")


def test_a_yahoo_failure_is_still_a_refusal(service, monkeypatch):
    def down(*a):
        raise ConnectionError("blocked")

    monkeypatch.setattr(service, "_fetch_from_yfinance", down)
    with pytest.raises(ValueError, match="refused"):
        service.get_ohlcv("BONUS.NS", "2024-01-01", "2024-01-05")


def test_bars_are_tagged_with_their_market(service, monkeypatch):
    """The tag picks the costs and the risk-free rate.  Neither frame here carries one of its own."""
    def bare(*a, **k):
        df = _bars([100.0, 101, 102])
        df.attrs.clear()
        return df

    monkeypatch.setattr("utils.download_ind_ohlcv", bare)
    monkeypatch.setattr(service, "_fetch_from_yfinance", bare)
    assert service.get_ohlcv("CALM.NS", "2024-01-01", "2024-01-03").attrs["market"] == "IND"
    assert service.get_ohlcv("AAPL", "2024-01-01", "2024-01-03").attrs["market"] == "US"
    assert service.get_ohlcv("CALM.NS", "2024-01-01", "2024-01-03").copy().attrs["market"] == "IND"    # the cached copy too


def test_quiet_raw_prices_are_used_as_they_are(service, monkeypatch):
    monkeypatch.setattr(service, "_fetch_from_yfinance", lambda *a: pytest.fail("no need for Yahoo"))
    assert service.get_ohlcv("CALM.NS", "2024-01-01", "2024-01-05")["Close"].iloc[-1] == 103


def test_the_avwap_restarts_at_each_anchor_and_starts_at_the_first():
    from strategies.utils import running_avwap

    df = pd.DataFrame({"High": [10.0, 20, 30, 40], "Low": [10.0, 20, 30, 40], "Close": [10.0, 20, 30, 40], "Volume": [1.0, 1, 3, 1]})
    got = running_avwap(df, pd.Series([False, True, False, True]))
    assert np.isnan(got[0]) and got[1:].tolist() == [20.0, (20 + 90) / 4, 40.0]


def test_a_volume_anchor_is_judged_against_the_past_only():
    from strategies.utils import volume_anchors

    v = pd.Series(np.arange(1.0, 301.0))
    a = volume_anchors(v, 95)
    v2 = v.copy()
    v2.iloc[250:] = 1e9                                           # a huge future volume must not lower a past anchor
    assert a.iloc[:250].equals(volume_anchors(v2, 95).iloc[:250]) and not a.iloc[:60].any() and a.iloc[100]


def _strategies():
    pytest.importorskip("matplotlib")
    from tools.strategy_benchmark import STRATEGIES, strategy_class

    return STRATEGIES, strategy_class


def _market(seed, n=600):
    rng = np.random.default_rng(seed)
    c = 500 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, n)))
    o = c * (1 + rng.normal(0, 0.005, n))
    df = pd.DataFrame({"Open": o, "High": np.maximum(o, c) * (1 + rng.uniform(0.001, 0.02, n)),
                       "Low": np.minimum(o, c) * (1 - rng.uniform(0.001, 0.02, n)), "Close": c,
                       "Volume": rng.uniform(1e5, 1e6, n)}, index=pd.bdate_range("2019-01-01", periods=n))
    df.attrs["market"] = "IND"
    return df


def _positions(key, df, tag):
    """A strategy's decisions and portfolio through its own ``run()``."""
    from strategies.data_service import DataService

    from tools.strategy_benchmark import _prime, strategy_class

    logging.disable(logging.CRITICAL)
    try:
        start, end = str(df.index[0].date()), str(df.index[-1].date())
        _prime(f"{tag}_{start}_{end}_1d", df)
        st = strategy_class(key)()
        st._create_charts = lambda *a, **k: []
        res = st.run([tag], start, end, 100_000.0, risk_params={"max_position_size": 1.0})
        assert res.success, res.error_message
        return res.signals["positions"].astype(float), res.portfolio, res.metrics[tag]
    finally:
        logging.disable(logging.NOTSET)
        DataService._cache.pop(f"{tag}_{df.index[0].date()}_{df.index[-1].date()}_1d", None)


SELLERS = ["volume_profile", "anchored_vwap", "order_flow", "liquidity_sweep", "bollinger_pattern"]
ALL_KEYS = ("macd", "awesome_oscillator", "heikin_ashi", "parabolic_sar", "rsi_pattern", "support_resistance",
            "bollinger_pattern", "volume_profile", "anchored_vwap", "order_flow", "liquidity_sweep", "swing_combo",
            "positional_combo")


@pytest.mark.parametrize("key", SELLERS)
def test_the_strategies_that_never_sold_now_sell(key):
    """Each wrote a sell as 0 and then erased it with replace(0, nan).ffill(), so it bought once and held for ever."""
    _strategies()
    exits = sum(int((_positions(key, _market(seed), f"S{seed}")[0].diff() < 0).sum()) for seed in range(6))
    assert exits > 0


@pytest.mark.parametrize("key", ALL_KEYS)
def test_no_strategy_reads_the_future(key):
    """Scramble every bar after a cut: no decision, value or cost up to it may move.  Found look-ahead in Support
    Resistance and Positional Combo; Anchored VWAP and Bollinger's were hidden behind their sell bugs."""
    _strategies()
    for seed in (1, 2):
        base = _market(seed)
        p0, b0, _ = _positions(key, base, f"A{seed}")
        for cut in (350, 500):
            rng = np.random.default_rng(seed * 1000 + cut)
            x = base.copy()
            k = rng.uniform(0.3, 2.5, len(x) - cut - 1)
            for col in ("Open", "High", "Low", "Close"):
                x.iloc[cut + 1:, x.columns.get_loc(col)] = x[col].iloc[cut + 1:].to_numpy() * k
            x.iloc[cut + 1:, x.columns.get_loc("Volume")] = x["Volume"].iloc[cut + 1:].to_numpy() * rng.uniform(0.2, 5, len(x) - cut - 1)
            p1, b1, _ = _positions(key, x, f"B{seed}{cut}")
            assert p0.iloc[:cut + 1].equals(p1.iloc[:cut + 1]), f"{key}: decisions moved with the future"
            assert (b0["total_value"].iloc[:cut + 1] - b1["total_value"].iloc[:cut + 1]).abs().max() < 1e-6


def test_the_combos_report_their_own_return():
    """Their metrics were the stock's: a -15.8% run reported -99%."""
    _strategies()
    for key in ("swing_combo", "positional_combo"):
        pos, book, m = _positions(key, _market(3, 900), "C1")
        assert pos.isin([0, 1]).all()
        assert m["final_value"] == pytest.approx(book["total_value"].iloc[-1])
        assert book["equity"].iloc[-1] == book["total_value"].iloc[-1] and m["costs_included"]


@pytest.mark.parametrize("key, feeds, args", [
    ("swing_combo", ("_sweep_signals", "_order_flow_signals", "_rsi_signals"), (10, 1.5, 20, 14, 14, 30, 70, 2)),
    ("positional_combo", ("_volume_profile_signals", "_anchored_vwap_signals", "_sweep_signals"), (60, 50, 0.7, 95, 0.005, 10, 1.5, 2)),
], ids=["swing", "positional"])
def test_a_combo_holds_from_its_buy_to_its_sell(key, feeds, args):
    """Their position was the day's signal, so a swing trade lasted one day and a sell shorted the next."""
    _strategies()
    from tools.strategy_benchmark import strategy_class

    st = strategy_class(key)()
    df = _market(1, 40)
    vote = pd.Series(0, index=df.index)
    vote.iloc[10], vote.iloc[20] = 1, -1
    for feed in feeds:
        setattr(st, feed, lambda *a, **k: vote)
    s = st._generate_signals(df, *args)
    assert s["positions"].tolist() == [0] * 10 + [1] * 10 + [0] * 20
    assert s["signal"].iloc[20] == -1                              # the charts still see the sell


@pytest.mark.parametrize("key", ALL_KEYS)
def test_a_strategy_keeps_its_markets_costs(key):
    """The market travels on the bars: a strategy that dropped it would pay India's costs on a US stock."""
    _strategies()
    df = _market(4, 500)
    us = df.copy()
    us.attrs["market"] = "US"
    _, india, _ = _positions(key, df, "M1")
    _, america, _ = _positions(key, us, "M2")
    if india["cumulative_cost"].iloc[-1] > 0:
        assert america["cumulative_cost"].iloc[-1] < india["cumulative_cost"].iloc[-1]


@pytest.mark.parametrize("method", ["_is_engulfing", "_is_star"])
def test_a_candle_is_judged_by_its_own_price_not_the_sample(method):
    """The minimum body came from the mean close of the whole history, later prices included."""
    _strategies()
    from tools.strategy_benchmark import strategy_class

    st = strategy_class("support_resistance")()
    df = pd.DataFrame({"Open": [100.0, 99.0, 100.0], "High": [100.5, 100.8, 100.2], "Low": [98.5, 98.9, 98.0], "Close": [99.0, 100.5, 100.1]})
    if method == "_is_star":
        df.loc[1, ["Open", "High", "Low", "Close"]] = [99.0, 104.0, 99.0, 100.0]      # a long upper wick
    now = getattr(st, method)(df, 1)
    soared = pd.concat([df, pd.DataFrame({c: [1e6] * 400 for c in df.columns})], ignore_index=True)
    assert now != 0 and getattr(st, method)(soared, 1) == now      # later prices change nothing


def test_support_resistance_opens_on_a_buy_after_repeated_sells():
    """cumsum().clip(0, 1) let two sells cancel the next buy."""
    _strategies()
    from tools.strategy_benchmark import strategy_class

    st = strategy_class("support_resistance")()
    df = _market(1, 80)
    picks = {"a": 60, "b": 62, "c": 65}
    st._is_engulfing = lambda d, i: {60: 1, 62: 1, 65: 2}.get(i, 0)
    st._is_star = lambda d, i: 0
    st._close_to_resistance = lambda d, i, levels, prox: i in (60, 62)
    st._close_to_support = lambda d, i, levels, prox: i == 65
    pos = st._generate_signals(df, 2, 2, 30, 0.01)["positions"]
    assert pos.iloc[64] == 0 and pos.iloc[65] == 1 and pos.iloc[-1] == 1 and picks


def test_a_pivot_is_used_only_once_its_later_bars_exist():
    _strategies()
    from tools.strategy_benchmark import strategy_class

    st = strategy_class("support_resistance")()
    seen = {}
    df = _market(2, 80)
    st._is_engulfing = lambda d, i: 2
    st._is_star = lambda d, i: 0
    st._close_to_resistance = lambda *a: False
    st._is_support = lambda d, i, n1, n2: i == 50
    st._is_resistance = lambda d, i, n1, n2: False

    def near(d, i, levels, prox):
        seen[i] = list(levels)
        return False

    st._close_to_support = near
    st._generate_signals(df, 2, 3, 30, 0.01)
    assert seen[52] == [] and seen[53] == [df["Low"].iloc[50]]     # known three bars on, not before


def test_the_benchmark_picks_its_names_without_hindsight():
    pytest.importorskip("matplotlib")
    from tools.strategy_benchmark import pick_names

    idx = pd.bdate_range("2012-01-02", periods=200)
    mask = pd.DataFrame(True, index=idx, columns=list("ABC"))
    value = pd.DataFrame({"A": 300.0, "B": 200.0, "C": 100.0}, index=idx)
    start = idx[50]
    assert pick_names(mask, value, start, 2) == ["A", "B"]
    later = value.copy()
    later.loc[start:, "C"] = 1e12                                  # a later winner fills the window, and is not picked
    assert pick_names(mask, later, start, 2) == ["A", "B"]
    with pytest.raises(ValueError):
        pick_names(mask, value, start, 4)


def test_the_benchmark_book_is_equal_sleeves_that_keep_their_own_money():
    pytest.importorskip("matplotlib")
    from tools.strategy_benchmark import book_of

    idx = pd.bdate_range("2013-01-01", periods=3)
    up = pd.DataFrame({"value": [100.0, 150.0, 150.0], "invested": 1.0}, index=idx)
    gone = pd.DataFrame({"value": [100.0, 50.0], "invested": 1.0}, index=idx[:2])    # delisted: its value stays
    ret, invested = book_of({"a": up, "b": gone}, idx[0], idx)
    assert ret.tolist() == pytest.approx([0.0, 0.0, 0.0])          # +50% and -50% on equal money, then nothing
    assert invested.iloc[0] == 1.0 and invested.iloc[-1] == pytest.approx(0.75)


def test_the_benchmark_tags_its_bars():
    pytest.importorskip("matplotlib")
    from types import SimpleNamespace

    from tools.strategy_benchmark import bars_for

    df = _bars([100.0, 101, 102])
    data = SimpleNamespace(**{k.lower(): pd.DataFrame({"X": df[k]}) for k in ("Open", "High", "Low", "Close", "Volume")})
    assert bars_for(data, "X", df.index[0], df.index[-1]).attrs["market"] == "IND"


def test_the_benchmark_will_not_open_the_holdout():
    pytest.importorskip("matplotlib")
    from tools.strategy_benchmark import main

    with pytest.raises(SystemExit, match="holdout"):
        main(["--end", "2026-01-31"])


@pytest.mark.parametrize("included, oos", [(True, 0.10), (False, 0.10 - 10 * 0.004)], ids=["simulated", "not simulated"])
def test_walk_forward_charges_costs_once(included, oos):
    """A portfolio that paid real costs is not charged the flat estimate again; one that did not still is."""
    from types import SimpleNamespace

    from services.research.walk_forward import walk_forward_validate

    class Fake:
        name = "fake"

        def run(self, **k):
            return SimpleNamespace(success=True, metrics={"T": {"sharpe_ratio": 1.0, "total_return": 10.0, "total_trades": 10,
                                                                "costs_included": included}})

    got = walk_forward_validate(Fake, "T", train_days=100, test_days=50, total_days=200, param_grid=[{}])
    assert got.folds and all(f.out_of_sample_return == pytest.approx(oos) for f in got.folds)


def test_a_run_cannot_start_before_its_regime_index(panel):
    """No index, no trend vote: the book would run another rule without a word.  The default store's NIFTY 50
    starts in 2012, so a 2008 run on it lost 9 points more that year than on store_ext2006."""
    from nse_engine.config import EngineConfig
    from nse_engine.engine import EngineCache
    from nse_engine.regime import require_index_from

    cfg = EngineConfig()
    dates = panel.index_close.index
    late = panel.index_close.copy()
    late.iloc[:100, late.columns.get_loc("NIFTY50")] = np.nan
    require_index_from(late, str(dates[100].date()), cfg.regime)        # from its first print: fine
    with pytest.raises(ValueError, match="regime index NIFTY50 starts"):
        require_index_from(late, str(dates[99].date()), cfg.regime)
    with pytest.raises(ValueError, match="regime index NIFTY50 starts"):
        require_index_from(late.assign(NIFTY50=np.nan), str(dates[0].date()), cfg.regime)
    require_index_from(late.drop(columns="NIFTY50"), str(dates[0].date()), cfg.regime)    # the fallback's case

    panel.index_close = late
    with pytest.raises(ValueError, match="regime index NIFTY50 starts"):
        EngineCache(panel, cfg.replace(start=str(dates[50].date()), end=str(dates[-1].date())))


def test_a_stock_moved_to_series_bz_can_be_sold_but_never_bought(monkeypatch):
    """A failing stock leaves EQ and BE: a holding must stay sellable (SB3), and nothing new goes there."""
    from types import SimpleNamespace

    from kite_connect.trading.nse_engine_executor import EngineExecutor
    from kite_connect.trading.nse_instruments import to_broker, to_engine

    ticks = {"DEAD-BZ": 0.01, "HFCL-BE": 0.01, "INFY": 0.05}
    assert to_broker("DEAD", ticks, exit_order=True)[0] == "DEAD-BZ"
    assert to_broker("DEAD", ticks) == (None, "DEAD: not in Kite's NSE equity list (no EQ or BE instrument)")
    assert to_broker("HFCL", ticks)[0] == "HFCL-BE" and to_broker("INFY", ticks, exit_order=True) == ("INFY", "")
    assert to_engine("DEAD-BZ") == "DEAD"

    order = lambda side: SimpleNamespace(symbol="DEAD", side=side, limit_price=4.0, ref_price=4.0, quantity=10,  # noqa: E731
                                         reason="delisted", current_qty=10, target_qty=0)
    plan = SimpleNamespace(sells=[order("SELL")], buys=[order("BUY")], as_of="2026-10-30")
    executor = EngineExecutor.__new__(EngineExecutor)
    executor.kite, executor.next_lower_circuits = object(), lambda symbols: {}
    monkeypatch.setattr("kite_connect.trading.nse_instruments.ticks_for", lambda kite: ticks)
    sell, buy = executor.live_orders(plan, variety="amo")
    assert sell["symbol"] == "DEAD-BZ" and sell["unresolved"] is None and "exit only" in sell["series_note"]
    assert buy["symbol"] == "DEAD" and buy["unresolved"], "a buy is not resolved to a restricted series"


def test_circuit_locks_known_answers():
    """No range and a moved close is a lock; a flat day, a ranging day or a missing bar is not (RB8)."""
    from nse_engine.costs import circuit_locks

    nan = float("nan")
    close = np.array([[100.0], [95.0], [99.75], [99.75], [nan], [94.76], [90.0], [90.0]])
    high = np.array([[101.0], [95.0], [99.75], [99.75], [nan], [94.76], [91.0], [90.0]])
    low = np.array([[99.0], [95.0], [99.75], [99.75], [nan], [94.76], [89.0], [nan]])
    down, up = circuit_locks(high, low, close)
    assert down[:, 0].tolist() == [False, True, False, False, False, True, False, False]
    assert up[:, 0].tolist() == [False, False, True, False, False, False, False, False]


def test_circuit_locks_properties():
    """Over random panels: a bar is never both, a ranging bar is never a lock, and scaling prices changes nothing."""
    from nse_engine.costs import circuit_locks

    rng = np.random.default_rng(8)
    for _ in range(50):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, (300, 6)), axis=0))
        spread = rng.uniform(0, 1, close.shape) * (rng.uniform(0, 1, close.shape) > 0.1) * close * 0.02
        high, low = close + spread, close - spread
        close[rng.uniform(0, 1, close.shape) < 0.02] = np.nan
        down, up = circuit_locks(high, low, close)
        assert not (down & up).any()
        assert not ((down | up) & (spread > 1e-3 * np.nan_to_num(close, nan=1e9))).any()
        d2, u2 = circuit_locks(high * 37, low * 37, close * 37)
        assert (d2 == down).all() and (u2 == up).all()


def _locked_bar(panel, symbol, row, move):
    """The panel with ``symbol`` frozen on ``row`` at ``move`` from the close before it."""
    import dataclasses

    frames = {n: getattr(panel, n).copy() for n in ("open", "high", "low", "close")}
    col = frames["close"].columns.get_loc(symbol)
    px = float(frames["close"].iloc[row - 1, col]) * (1 + move)
    for f in frames.values():
        f.iloc[row, col] = px
    return dataclasses.replace(panel, **frames)


def _lock_config(panel, on):
    from nse_engine.config import EngineConfig

    dates = panel.dates
    return EngineConfig().replace(start=str(dates[300].date()), end=str(dates[-1].date()),
                                  **{"portfolio.respect_circuit_locks": on, "universe.top_n_liquid": 40,
                                     "universe.min_history_days": 60, "universe.liquidity_lookback_days": 40,
                                     "universe.min_median_value_inr": 0.0, "universe.min_price_inr": 0.0})


@pytest.mark.parametrize("side, move, opposite", [("SELL", -0.02, 0.02), ("BUY", 0.02, -0.02)])
def test_a_locked_bar_stops_only_the_side_that_cannot_trade(panel, side, move, opposite):
    """Off, the order fills on the locked day as before.  On, it does not; the free side still fills (RB8)."""
    from nse_engine.engine import run_backtest

    trades = run_backtest(panel, _lock_config(panel, False), record=False).trades
    picked = trades[(trades["side"] == side) & (trades["reason"] == "rebalance") & (trades["date"] > panel.dates[320])]
    trade = picked.sort_values("value_inr", ascending=False).iloc[0]
    row, sym, day = panel.dates.get_loc(trade["date"]), trade["symbol"], trade["date"]

    def fills(data, on):
        out = run_backtest(data, _lock_config(data, on), record=False).trades
        return out[(out["symbol"] == sym) & (out["side"] == side) & (out["date"] == day)]

    locked = _locked_bar(panel, sym, row, move)
    assert len(fills(locked, False)) == 1, "legacy behaviour is unchanged"
    assert fills(locked, True).empty
    assert len(fills(_locked_bar(panel, sym, row, opposite), True)) == 1, "the free side still trades"


def test_a_stop_cannot_fill_on_a_locked_bar_and_goes_out_next_session(panel):
    """A stop under a lower lock waits, as a real GTT would, then sells at the next tradable open (RB8)."""
    from nse_engine.engine import run_backtest

    trades = run_backtest(panel, _lock_config(panel, False), record=False).trades
    stop = trades[(trades["side"] == "SELL") & (trades["reason"] == "stop") & (trades["date"] > panel.dates[320])].iloc[0]
    row, sym, day = panel.dates.get_loc(stop["date"]), stop["symbol"], stop["date"]
    data = _locked_bar(panel, sym, row, float(stop["price"]) / float(panel.close[sym].iloc[row - 1]) - 1.03)

    def sells(on):
        out = run_backtest(data, _lock_config(data, on), record=False).trades
        return out[(out["symbol"] == sym) & (out["side"] == "SELL") & (out["date"] >= day)]

    assert sells(False)["date"].iloc[0] == day, "legacy fills on the locked day"
    held = sells(True)
    assert held["date"].iloc[0] > day and held["reason"].iloc[0] == "stop"


class _ValueBroker:
    """Only what the manual limit reads: the last price and the long position."""

    def __init__(self, ltp=1000.0, long_qty=0, fail=False):
        self._ltp, self.long_qty, self.fail = ltp, long_qty, fail

    def ltp(self, keys):
        if self.fail:
            raise RuntimeError("quotes down")
        return {keys[0]: {"last_price": self._ltp}}


@pytest.mark.parametrize("side, qty, kind, price, trigger, product, refused", [
    ("BUY", 80, "LIMIT", 2500.0, None, "CNC", False),      # exactly Rs 2 lakh
    ("BUY", 81, "LIMIT", 2500.0, None, "CNC", True),
    ("BUY", 81, "SL", 2500.0, 2490.0, "CNC", True),
    ("BUY", 81, "SL-M", None, 2500.0, "CNC", True),
    ("BUY", 200, "MARKET", None, None, "CNC", False),
    ("BUY", 201, "MARKET", None, None, "CNC", True),
    ("SELL", 5000, "LIMIT", 2500.0, None, "CNC", False),   # an exit is never limited
    ("SELL", 5000, "MARKET", None, None, "CNC", False),
])
def test_the_manual_limit_known_answers(side, qty, kind, price, trigger, product, refused):
    from kite_connect.trading.order_service import manual_value_refusal

    got = manual_value_refusal(_ValueBroker(), "RELIANCE", "NSE", side, qty, kind, price, trigger, product)
    assert bool(got) is refused


def test_the_manual_limit_refuses_a_market_buy_it_cannot_value_but_never_an_exit():
    from kite_connect.trading.order_service import manual_value_refusal

    down = _ValueBroker(fail=True)
    assert "cannot value" in manual_value_refusal(down, "X", "NSE", "BUY", 1, "MARKET", None, None, "CNC")
    assert manual_value_refusal(down, "X", "NSE", "SELL", 1, "MARKET", None, None, "CNC") is None


def test_the_manual_limit_lets_a_verified_exit_go_and_caps_a_new_short(monkeypatch):
    from kite_connect.trading import order_service as gate

    monkeypatch.setattr(gate, "_long_quantity", lambda kite, s, e, p: 100)
    args = ("NIFTYFUT", "NFO", "SELL")
    assert gate.manual_value_refusal(_ValueBroker(), *args, 100, "LIMIT", 20000.0, None, "NRML") is None
    assert gate.manual_value_refusal(_ValueBroker(), *args, 101, "LIMIT", 20000.0, None, "NRML")

    def unreadable(*_):
        raise RuntimeError("positions down")

    monkeypatch.setattr(gate, "_long_quantity", unreadable)
    assert gate.manual_value_refusal(_ValueBroker(), *args, 100, "LIMIT", 20000.0, None, "NRML"), "unverified is new risk"


def test_the_manual_route_enforces_the_limit(route):
    from fastapi import HTTPException

    broker = _Broker()
    with pytest.raises(HTTPException) as refused:
        _post(route, broker, quantity=81)
    assert refused.value.status_code == 422 and "manual limit" in refused.value.detail and broker.sent == 0
    assert _post(route, broker, quantity=80)["order_id"] and broker.sent == 1


NIFTY_LOT, BANKNIFTY_LOT = 65 * 23098.9, 30 * 54823.4     # Rs 15.0 and 16.4 lakh on 30 Oct 2026


@pytest.mark.parametrize("capital, symbol, notional, refused", [
    (600_000, "NIFTY26NOVFUT", NIFTY_LOT, True),
    (1_200_000, "NIFTY26NOVFUT", NIFTY_LOT, True),
    (2_100_000, "NIFTY26NOVFUT", NIFTY_LOT, False),
    (3_000_000, "NIFTY26NOVFUT", 2 * NIFTY_LOT, True),
    (1_200_000, "BANKNIFTY26NOVFUT", BANKNIFTY_LOT, True),
    (2_100_000, "BANKNIFTY26NOVFUT", BANKNIFTY_LOT, False),
    (600_000, "RELIANCE26NOVFUT", 340_000, False),
    (600_000, "RELIANCE26NOVFUT", 350_000, True),
    (1_200_000, "RELIANCE26NOVFUT", 580_000, False),
    (2_100_000, "RELIANCE26NOVFUT", 1_210_000, False),
    (2_100_000, "RELIANCE26NOVFUT", 1_220_000, True),
])
def test_the_manual_fo_limit_follows_the_capital_rungs(monkeypatch, capital, symbol, notional, refused):
    """The stress loss stays within 15% of live capital: -17% on an index, -26% on a stock (MO2)."""
    from kite_connect.trading.order_service import manual_value_refusal

    monkeypatch.setenv("CENTURION_LIVE_CAPITAL", str(capital))
    got = manual_value_refusal(_ValueBroker(), symbol, "NFO", "BUY", 1, "LIMIT", notional, None, "NRML")
    assert bool(got) is refused


@pytest.mark.parametrize("raw", ["", "lots", "-5", "0", "nan", "inf"])
def test_a_missing_or_bad_capital_tightens_to_the_first_rung(monkeypatch, raw):
    from kite_connect.trading.order_service import manual_fo_capital

    monkeypatch.setenv("CENTURION_LIVE_CAPITAL", raw)
    assert manual_fo_capital() == 300_000


def test_manual_options_buy_up_to_the_budget_and_never_open_short(monkeypatch):
    from kite_connect.trading import order_service as gate

    monkeypatch.setenv("CENTURION_LIVE_CAPITAL", "600000")
    monkeypatch.setattr(gate, "_long_quantity", lambda kite, s, e, p: 0)
    buy = lambda q: gate.manual_value_refusal(_ValueBroker(), "NIFTY26NOV23000CE", "NFO", "BUY", q, "LIMIT", 100.0, None, "NRML")  # noqa: E731
    assert buy(900) is None and buy(901)
    short = gate.manual_value_refusal(_ValueBroker(), "NIFTY26NOV23000PE", "NFO", "SELL", 1, "LIMIT", 1.0, None, "NRML")
    assert "no defined loss" in short
    monkeypatch.setattr(gate, "_long_quantity", lambda kite, s, e, p: 65)
    assert gate.manual_value_refusal(_ValueBroker(), "NIFTY26NOV23000PE", "NFO", "SELL", 65, "LIMIT", 1.0, None, "NRML") is None


def test_an_unknown_contract_is_refused_and_a_cash_order_keeps_the_rupee_limit(monkeypatch):
    from kite_connect.trading.order_service import manual_value_refusal

    monkeypatch.setenv("CENTURION_LIVE_CAPITAL", "3000000")
    assert "not a known" in manual_value_refusal(_ValueBroker(), "ODDSYMBOL", "NFO", "BUY", 1, "LIMIT", 1.0, None, "NRML")
    assert manual_value_refusal(_ValueBroker(), "RELIANCE", "NSE", "BUY", 81, "LIMIT", 2500.0, None, "CNC")


class _FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        assert seconds > 0
        self.now += seconds


@pytest.fixture
def pace(monkeypatch):
    from kite_connect.trading import order_service as gate

    monkeypatch.setattr(gate, "_order_times", type(gate._order_times)(maxlen=gate.ORDERS_PER_SECOND))
    return gate


def test_no_second_ever_holds_more_than_ten_orders(pace):
    """Property: over bursts and random gaps, any one-second span holds at most 10 sends (GT2H)."""
    rng = np.random.default_rng(10)
    for _ in range(50):
        pace._order_times.clear()
        clock, sent = _FakeClock(), []
        for gap in rng.choice([0.0, 0.0, 0.0, 0.01, 0.2, 1.5], size=60):
            clock.now += float(gap)
            pace._pace_orders(clock, clock.sleep)
            sent.append(clock.now)
        times = np.array(sent)
        assert all((times[i + pace.ORDERS_PER_SECOND] - times[i]) >= 1.0 for i in range(len(times) - pace.ORDERS_PER_SECOND))


def test_ten_orders_go_at_once_and_the_eleventh_waits_its_second(pace):
    clock = _FakeClock()
    for _ in range(10):
        pace._pace_orders(clock, clock.sleep)
    assert clock.now == 1000.0
    pace._pace_orders(clock, clock.sleep)
    assert clock.now == pytest.approx(1001.0)


def test_an_idle_book_never_waits(pace):
    clock = _FakeClock()
    for _ in range(40):
        clock.now += 0.2
        before = clock.now
        pace._pace_orders(clock, clock.sleep)
        assert clock.now == before


def test_the_wait_holds_the_lock_so_no_other_thread_can_slip_past(pace):
    clock = _FakeClock()
    for _ in range(10):
        pace._pace_orders(clock, clock.sleep)
    held = []
    pace._pace_orders(clock, lambda seconds: held.append(pace._order_pace_lock.locked()))
    assert held == [True]


def test_threads_cannot_break_the_limit(pace):
    """Real clock: 15 sends from 15 threads, the last must wait for the window."""
    import threading

    stamps, lock = [], threading.Lock()

    def send():
        pace._pace_orders()
        with lock:
            stamps.append(pace.time.monotonic())

    threads = [threading.Thread(target=send) for _ in range(15)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stamps.sort()
    assert len(stamps) == 15 and stamps[10] - stamps[0] >= 0.99


@pytest.mark.parametrize("call", ["cancel", "modify"])
def test_cancels_and_modifies_are_paced_too(monkeypatch, pace, call):
    from unittest import mock

    paced = mock.Mock()
    monkeypatch.setattr(pace, "_pace_orders", paced)
    kite = mock.Mock()
    assert (pace.cancel_order(kite, "1") if call == "cancel" else pace.modify_order(kite, "1", price=10.0))["success"]
    paced.assert_called_once()


def test_every_order_send_is_paced_before_it_goes(monkeypatch, route):
    from kite_connect.trading import order_service as svc

    broker, seen = _Broker(), []
    monkeypatch.setattr(svc, "_pace_orders", lambda: seen.append(broker.sent))
    _post(route, broker)
    assert seen == [0] and broker.sent == 1, "the wait comes before the send"


# ---- the go-live ramp (GT2I)
def _ladder(rung=0, sessions=100):
    from nse_engine.capital_ladder import LadderState

    return LadderState(rung=rung, since="2027-01-01", sessions=sessions, last_session="2027-06-01")


def _ladder_gate(costs="PASS"):
    names = ("tracking error", "daily gap", "drawdown", "regime break")
    return {"checks": [{"name": n, "status": "PASS"} for n in names] + [{"name": "costs", "status": costs}]}


def _ladder_eval(state, gate=None, ramp=None, requested=None):
    from nse_engine.capital_ladder import evaluate

    return evaluate(state, "2027-06-02", gate=gate or _ladder_gate(), drawdown_state="normal", book_dd=0.02,
                    nifty_dd=0.02, shift_multipliers=[], requested_capital=requested, ramp=ramp)


def _ramp(alarm=False, low=-1.0, volatility=True, turnover=True):
    return {"cusum": {"low": low, "alarm": alarm, "sessions": 40},
            "volatility": {"value": 1.0 if volatility else 1.6, "ok": volatility, "display": "1.00x"},
            "turnover": {"value": 1.0 if turnover else 0.4, "ok": turnover, "display": "1.00x"}}


def test_the_canary_needs_three_months_before_go():
    held = _ladder(sessions=58)                    # evaluate counts tonight's session
    assert "59 of 60 sessions" in _ladder_eval(held, requested=600_000).reasons[0]
    assert held.rung == 0
    moved = _ladder(sessions=59)
    d = _ladder_eval(moved, ramp=_ramp(), requested=600_000)
    assert d.action == "GO" and d.capital == 600_000 and moved.rung == 1


@pytest.mark.parametrize("gate,ramp,why", [
    (_ladder_gate("n/a"), _ramp(), "costs"),
    (_ladder_gate("WATCH"), _ramp(), "costs"),
    (_ladder_gate(), _ramp(volatility=False), "volatility"),
    (_ladder_gate(), _ramp(turnover=False), "turnover"),
])
def test_go_needs_measured_fills_and_behaviour(gate, ramp, why):
    state = _ladder(rung=1)
    d = _ladder_eval(state, gate=gate, ramp=ramp, requested=1_200_000)
    assert d.action == "HOLD" and state.rung == 1
    assert why in d.alerts[0]


def test_the_cusum_alarm_steps_down_one_rung_and_alerts():
    state = _ladder(rung=2)
    d = _ladder_eval(state, ramp=_ramp(alarm=True, low=-12.5))
    assert d.action == "STEP DOWN" and d.capital == 600_000 and "CUSUM" in d.reasons[0]
    assert d.critical


def test_the_cusum_alarm_at_the_canary_alerts_and_keeps_the_rung():
    state = _ladder(rung=0)
    d = _ladder_eval(state, ramp=_ramp(alarm=True, low=-12.5))
    assert state.rung == 0 and d.flow == 0.0 and "kill switch" in d.critical[0]


def test_gap_cusum_catches_a_break_and_ignores_noise():
    from nse_engine.capital_ladder import gap_cusum

    rng = np.random.default_rng(3)
    noise = pd.Series(rng.normal(0.0, 0.0025, 120))
    assert not gap_cusum(noise)["alarm"]
    assert gap_cusum(noise - 0.0025)["alarm"]
    assert gap_cusum(pd.Series(-0.01, index=range(19)))["sessions"] == 19      # too few to read
    assert not gap_cusum(noise.head(19))["alarm"]
    assert not gap_cusum(pd.Series(0.0, index=range(40)))["alarm"]             # identical to the backtest


def test_ramp_checks_measure_fills_against_the_backtest():
    from nse_engine.capital_ladder import ramp_checks

    days = pd.bdate_range("2027-01-04", periods=30)
    ref = pd.Series(np.random.default_rng(1).normal(0.0005, 0.01, 30), index=days)
    equity = 300_000 * (1 + ref).cumprod()
    fills = pd.DataFrame({"session_date": [days[5].date().isoformat()] * 2, "status": ["FILLED", "CANCELLED"],
                          "quantity": [100, 100], "fill_price": [100.0, 100.0], "costs_inr": [5.0, 0.0],
                          "impact_bps": [2.0, 0.0]})
    trades = pd.DataFrame({"date": [days[5], days[6]], "value_inr": [15_000.0, 5_000.0]})
    out = ramp_checks(equity, ref, fills, trades)
    assert out["volatility"]["ok"] and not out["cusum"]["alarm"]
    assert out["turnover"]["value"] == pytest.approx(0.5) and not out["turnover"]["ok"]      # half the trading
    assert ramp_checks(equity, ref, None, trades)["turnover"]["display"] == "not measurable"


def test_go_live_refuses_any_start_but_the_canary():
    from nse_engine.capital_ladder import readiness

    assert dict((n, ok) for n, ok, _ in readiness({}, [], capital=300_000))["canary"]
    assert not dict((n, ok) for n, ok, _ in readiness({}, [], capital=600_000))["canary"]
    assert "canary" not in dict((n, ok) for n, ok, _ in readiness({}, []))
