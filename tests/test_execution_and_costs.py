"""Execution realism: fills come from prices the decision could not see, and cost money; and every order passes
one gate, so a repeated request or a lost answer never trades twice (RB5)."""
from __future__ import annotations

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
