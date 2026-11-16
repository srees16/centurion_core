"""The engine's executor on the live path: an unreadable book holds back buys and lets exits through, getting out
is always possible (RB7), a runaway session sends no buy, a missed session keeps the ledger true, and the drawdown
rule re-arms (DD1)."""
from __future__ import annotations

import pandas as pd
import pytest

from conftest import _Broker, _post, LIMIT


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


@pytest.mark.lesson("L-RB7", pattern="F1", date="2026-10-30", source="RB7",
                    summary="One failed sell opened the order breaker and the night's other exits were refused")
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


def test_live_buys_spend_only_the_cash_already_free(monkeypatch):
    """LN-T11 (Zerodha, 5 Nov 2026: same-open sale credit unconfirmed): with no sell credit, a night that sells OLD
    and buys NEW buys only what the free cash covers; with the model's full credit, the whole target.  The live
    session runs with none; paper keeps the model's."""
    from types import SimpleNamespace

    from kite_connect.trading import live_session
    from kite_connect.trading import nse_engine_executor as ex

    day = pd.Timestamp("2024-06-03")
    data = SimpleNamespace(dates=[day], close=pd.DataFrame({"OLD": [100.0], "NEW": [100.0]}, index=[day]))
    data.until = lambda as_of: data
    cfg = SimpleNamespace(portfolio=SimpleNamespace(no_trade_buffer=0.25, min_trade_value_inr=5000.0))
    target = SimpleNamespace(weights={"NEW": 0.5}, exits={"OLD": "rank_exit"}, stops={}, notes=[])

    def plan(credit):
        e = ex.EngineExecutor(paper=True, config=cfg, target_fn=lambda *a, **k: target, stopped_out_fn=dict,
                              holdings_fn=lambda: ({"OLD": {"quantity": 1000, "avg_price": 90.0}}, 20_000.0),
                              sell_credit=credit)
        e.drawdown_decision = lambda as_of, equity: (None, None)
        e._apply_shift_multiplier = lambda p, t, *a: dict(t.weights)
        return {o.symbol: o.quantity for o in e.plan(as_of=day, data=data).orders}, e

    full, _ = plan(1.0)
    free, _ = plan(ex.LIVE_SELL_PROCEEDS_CREDIT)
    assert full == {"OLD": 1000, "NEW": 600}              # half of Rs 1.2 lakh at Rs 100
    assert free == {"OLD": 1000, "NEW": 197}              # Rs 20,000 free / (Rs 101 limit + 25 bp allowance)
    assert ex.LIVE_SELL_PROCEEDS_CREDIT == 0.0 and ex.SELL_PROCEEDS_CREDIT == 1.0
    with pytest.raises(ValueError):
        plan(1.5)

    made = {}

    def factory(**kw):
        made.update(kw)
        raise RuntimeError("stop here")
    with pytest.raises(RuntimeError, match="stop here"):
        live_session.run_live_session(object(), dry_run=True, capital=300_000.0, deployment=SimpleNamespace(),
                                      executor_factory=factory,
                                      book=SimpleNamespace(read_state=dict, read_snapshots=lambda: None),
                                      email=False)
    assert made["sell_credit"] == ex.LIVE_SELL_PROCEEDS_CREDIT


def _capped_executor(monkeypatch):
    """An EngineExecutor over a stub Kite: deployed limits, an empty order book, every send recorded."""
    from types import SimpleNamespace

    from kite_connect.trading import gtt_stops, order_service
    from kite_connect.trading.nse_engine_executor import EngineExecutor

    sent = []
    monkeypatch.setattr("kite_connect.trading.nse_instruments.ticks_for", lambda kite: {"AAA": 0.05, "BBB": 0.05, "GOLDBEES": 0.01})
    monkeypatch.setattr(order_service, "is_kill_switch_active", lambda: False)
    monkeypatch.setattr(order_service, "place_order",
                        lambda kite, **kw: sent.append((kw["transaction_type"], kw["symbol"])) or {"success": True})
    monkeypatch.setattr(gtt_stops, "reconcile_stop_gtts", lambda kite, **kw: {"errors": []})
    ex = EngineExecutor.__new__(EngineExecutor)
    ex.kite, ex._stop_scope_fn = SimpleNamespace(orders=lambda: []), None
    ex.next_lower_circuits = lambda symbols: {}
    ex.config = SimpleNamespace(portfolio=SimpleNamespace(max_positions=30, max_weight=0.08),
                                allocator=SimpleNamespace(max_gross=1.0),
                                sleeves=SimpleNamespace(symbols=("GOLDBEES", "SILVERBEES")))
    return ex, sent


def test_a_runaway_session_sends_no_buy_and_still_exits(monkeypatch):
    """Golden-target review's session cap, on a stub broker: a stock BUY far above the book's largest position (a
    quantity bug) refuses every BUY that session, live and in the dry run; the exits still go.  A full book
    bought from cash in one session (a first deployment, a regime re-entry), or a sleeve near half the book, is
    not a runaway."""
    from kite_connect.trading.nse_engine_executor import ExecutionPlan, PlannedOrder

    def plan(buy_qty, cash, symbol="AAA"):
        return ExecutionPlan(as_of=pd.Timestamp("2026-11-05"), equity=300_000.0, cash=cash, orders=[
            PlannedOrder("BBB", "SELL", 10, 100.0, 95.0, "exit:stop", current_qty=10, target_qty=0),
            PlannedOrder(symbol, "BUY", buy_qty, 100.0, 101.0, "entry", current_qty=0, target_qty=buy_qty)])

    ex, sent = _capped_executor(monkeypatch)
    runaway = plan(2_000, 250_000.0)                      # Rs 2 lakh in one stock: the cap is 10% of Rs 3 lakh
    results = ex._execute_live(runaway)
    assert sent == [("SELL", "BBB")]
    assert any("SESSION CAP" in (r.get("error") or "") and r.get("type") == "preflight" for r in results)
    dry = {r["symbol"]: r for r in ex.dry_run_live(runaway) if "tag" in r}
    assert dry["AAA"]["status"] == "WOULD_REJECT" and "SESSION CAP" in dry["AAA"]["error"]
    assert dry["BBB"]["status"] == "WOULD_PLACE"

    sent.clear()
    results = ex._execute_live(plan(290, 300_000.0))      # a 9.7% stock position: inside the band
    assert sent == [("SELL", "BBB"), ("BUY", "AAA")]
    assert not any("SESSION CAP" in (r.get("error") or "") for r in results)
    sent.clear()
    ex._execute_live(plan(1_400, 300_000.0, "GOLDBEES"))  # a 47% gold sleeve bought from cash: allowed
    assert sent == [("SELL", "BBB"), ("BUY", "GOLDBEES")]
    sent.clear()
    ex._execute_live(plan(1_400, 100_000.0, "GOLDBEES"))  # gross after the session would be 113% of the book
    assert sent == [("SELL", "BBB")]


def test_the_game_day_broker_keeps_kites_rules():
    """tools.game_day's SimKite: a buy is checked against the cash free before the open, a DAY limit fills by
    live's rule or lapses, bought shares go T1 then settle, a gap through a stop's limit leaves its GTT order
    unfilled, a lower-circuit lock blocks the sell, and a split multiplies the holding and cancels its GTT."""
    from tools.game_day import SimKite

    days = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    rows = []
    for d, (o, h, l, c, p) in zip(days, [(100, 101, 99, 100, 100), (100, 102, 99.5, 101, 100), (101, 101, 101, 101, 101),
                                         (80, 81, 79, 80, 101), (40, 41, 39, 40, 80)]):
        rows.append((d, "AAA", o, h, l, c, p))
        rows.append((d, "BBB", 50, 51, 49, 50, 50))
    bars = pd.DataFrame(rows, columns=["date", "symbol", "open", "high", "low", "close", "prev_close"]) \
        .set_index(["date", "symbol"])
    k = SimKite(bars, 10_000.0, 15.0)
    k.open(days[0])
    k.place_order("amo", "NSE", "AAA", "BUY", 50, "CNC", "LIMIT", price=101.0, tag="NE240101BAAA")
    k.place_order("amo", "NSE", "BBB", "BUY", 200, "CNC", "LIMIT", price=50.5, tag="NE240101BBBB")
    k.open(days[1])
    book = {o["tradingsymbol"]: o for o in k.orders()}
    assert book["AAA"]["status"] == "COMPLETE" and book["AAA"]["average_price"] == 100.0
    assert book["BBB"]["status"] == "REJECTED"            # Rs 10,100 > the Rs 5,000 left before the open
    assert k.positions()["net"][0]["quantity"] == 50 and k.holdings() == []
    k.place_gtt("single", "AAA", "NSE", [95.0], 101.0,
                [{"transaction_type": "SELL", "quantity": 50, "price": 94.05, "order_type": "LIMIT", "product": "CNC"}])
    k.place_order("amo", "NSE", "AAA", "SELL", 50, "CNC", "LIMIT", price=96.0, tag="NE240102SAAA")
    k.open(days[2])                                       # a flat day up: no lock for a sell, it fills at the open
    assert k.holdings()[0]["t1_quantity"] == 50 and k.orders()[0]["status"] == "COMPLETE"
    assert k.held("AAA") == 0
    k.place_order("amo", "NSE", "BBB", "BUY", 10, "CNC", "LIMIT", price=50.5, tag="NE240103BBBB")
    k.open(days[3])
    k.place_gtt("single", "BBB", "NSE", [49.5], 50.0,
                [{"transaction_type": "SELL", "quantity": 10, "price": 49.95, "order_type": "LIMIT", "product": "CNC"}])
    k.open(days[4])                                       # BBB opens 50, low 49: triggers, fills at the trigger
    assert k.held("BBB") == 0 and k.events["stop_filled"] == 1
    assert k.sent_tags["NE240101BAAA"] == 1 and len(k.rejected) == 1

    bonus = pd.DataFrame({"symbol": ["AAA"], "ex_date": [days[3]], "factor": [0.5]})     # 1:1, from the store table
    for actions, credit, gtt in ((None, {}, "active"), (bonus, {"AAA": 10}, "cancelled")):
        s2 = SimKite(bars, 0.0, 15.0, actions)            # a crash alone keeps the shares; a bonus cancels the GTT on
        s2.open(days[2])                                  # the ex-date and its shares arrive the next session
        s2.settled["AAA"] = 10
        s2.place_gtt("single", "AAA", "NSE", [10.0], 80.0,
                     [{"transaction_type": "SELL", "quantity": 10, "price": 9.9, "order_type": "LIMIT", "product": "CNC"}])
        s2.open(days[3])
        assert s2.held("AAA") == 10 and s2.credit == credit and s2.gtts[1]["status"] == gtt
        s2.open(days[4])
        assert s2.held("AAA") == 10 + sum(credit.values())

    locked = pd.DataFrame([(days[0], "CCC", 100, 100, 100, 100, 100), (days[1], "CCC", 90, 90, 90, 90, 100)],
                          columns=["date", "symbol", "open", "high", "low", "close", "prev_close"]).set_index(["date", "symbol"])
    lk = SimKite(locked, 0.0, 15.0)
    lk.open(days[0])
    lk.settled["CCC"] = 5
    lk.place_order("amo", "NSE", "CCC", "SELL", 5, "CNC", "LIMIT", price=85.0, tag="L1")
    lk.open(days[1])
    assert lk.held("CCC") == 5 and lk.orders()[0]["status"] == "CANCELLED" and lk.events["locked_unfilled"] == 1


@pytest.mark.slow
def test_a_missed_session_during_a_bonus_credit_keeps_the_ledger_true(store_path):
    """Game day GD1: INOXWIND's 3:1 bonus (ex 24 May 2024), an exit of the deliverable 8 shares filled at the next
    open, and that night's session lost (broker down).  The next session must adopt the sale from the broker's
    quantity, not count the awaited bonus as held, so the ledger ends where the broker is."""
    from tools import game_day as gd

    out = gd.run("faults", "2024-05-02", "2024-05-31", {"2024-05-27": "down"})
    assert out["checks"]["ledger_matches_broker"], out["ledger_vs_broker"]
    assert out["checks"]["no_unexplained_reconcile"] and out["events"].get("corporate_action", 0) >= 1


@pytest.mark.lesson("L-DD1", pattern="F1", date="2026-10-11", source="DD1",
                    summary="An all-cash book never re-armed its drawdown rule: the re-arm needed equity strictly above its high")
def test_a_flat_book_rearms_after_its_sessions_and_a_falling_one_never_does():
    from nse_engine.config import DrawdownConfig
    from nse_engine.drawdown import HALT, NORMAL, DrawdownTracker

    cfg = DrawdownConfig(enabled=True, rearm_sessions=5)
    flat = DrawdownTracker(cfg)
    flat.update(100.0)
    assert flat.update(80.0).state == HALT                               # 20% down: no new entries
    assert [flat.update(80.0).state for _ in range(5)] == [HALT] * 4 + [NORMAL]   # five flat sessions: a fresh episode
    falling = DrawdownTracker(cfg)
    falling.update(100.0)
    assert all(falling.update(80.0 - 0.1 * i).state != NORMAL for i in range(40))


# ── what the executor reads before it plans ─────────────────────

def test_the_shift_multiplier_is_used_only_when_fresh_dated_and_sane(tmp_path):
    import json

    from kite_connect.trading.nse_engine_executor import SHIFT_STATE_MAX_AGE_SESSIONS, load_shift_multiplier

    path = tmp_path / "shift.json"
    assert load_shift_multiplier("2026-11-13", path=path) == (1.0, f"no distribution shift state at {path}")

    def state(**over):
        path.write_text(json.dumps({"position_size_multiplier": 0.5, "updated_at": "2026-11-12T19:05:00+05:30",
                                    "verdict": "SHIFT", **over}))

    state()
    assert load_shift_multiplier("2026-11-13", path=path) == (0.5, "verdict=SHIFT updated 2026-11-12 (1 sessions ago)")
    state(reality_gap_alerts=["costs 2.1x"])
    assert load_shift_multiplier("2026-11-13", path=path)[1].endswith("; reality gap: costs 2.1x")
    state()
    assert load_shift_multiplier("2026-11-11", path=path) == (
        1.0, "shift state updated 2026-11-12 after as_of 2026-11-11 (ignored)")        # a replay of an earlier night
    week = SHIFT_STATE_MAX_AGE_SESSIONS                                               # weekdays, with no calendar given
    assert load_shift_multiplier("2026-11-19", path=path)[0] == 0.5 and week == 5      # five sessions old: still used
    stale = load_shift_multiplier("2026-11-20", path=path)
    assert stale == (1.0, "stale shift state (updated 2026-11-12, 6 sessions old)")
    sessions = pd.to_datetime(["2026-11-12", "2026-11-13", "2026-11-16"])             # the data's calendar ends on the 16th
    assert load_shift_multiplier("2026-11-18", sessions=sessions, path=path)[1].endswith("(4 sessions ago)")
    for bad in (0.0, 1.5, -0.2, float("nan")):
        state(position_size_multiplier=bad)
        assert load_shift_multiplier("2026-11-13", path=path)[0] == 1.0, bad
    path.write_text("{not json")
    assert load_shift_multiplier("2026-11-13", path=path) == (1.0, "malformed shift state (JSONDecodeError)")
    path.write_text(json.dumps({"verdict": "OK"}))
    assert load_shift_multiplier("2026-11-13", path=path) == (1.0, "malformed shift state (KeyError)")


def test_the_brokers_book_is_read_in_engine_names_with_its_highest_stop(monkeypatch):
    from types import SimpleNamespace

    from kite_connect.trading import gtt_stops
    from kite_connect.trading.nse_engine_executor import kite_book

    kite = SimpleNamespace(
        holdings=lambda: [{"tradingsymbol": "INFY", "average_price": 1150.0, "isin": "INE009A01021"},
                          {"tradingsymbol": "HFCL-BE", "average_price": 80.0}],
        margins=lambda segment: {"available": {"live_balance": 4_321.5, "cash": 9_999.0}})
    monkeypatch.setattr(gtt_stops, "get_held_quantities", lambda k: {"INFY": 4, "HFCL-BE": 10, "HFCL": 5})
    monkeypatch.setattr(gtt_stops, "list_stop_gtts", lambda k: [{"symbol": "INFY", "trigger": 1100.0},
                                                               {"symbol": "INFY", "trigger": 1120.0},
                                                               {"symbol": "HFCL-BE", "trigger": 70.0}])
    book, cash = kite_book(kite)
    assert cash == 4_321.5                                                            # what is free now, not the opening cash
    assert book == {"INFY": {"quantity": 4, "avg_price": 1150.0, "stop_price": 1120.0, "stops_unknown": False,
                             "isin": "INE009A01021"},
                    "HFCL": {"quantity": 15, "avg_price": 80.0, "stop_price": 70.0, "stops_unknown": False, "isin": None}}

    def down(k):
        raise ConnectionError("gtt list down")

    monkeypatch.setattr(gtt_stops, "list_stop_gtts", down)
    blind, _ = kite_book(kite)
    assert all(h["stop_price"] is None and h["stops_unknown"] is True for h in blind.values())   # unknown, not "no stop"
    kite.margins = lambda segment: {"available": {"cash": 250.0}}
    assert kite_book(kite)[1] == 250.0
    kite.margins = lambda segment: None
    assert kite_book(kite)[1] == 0.0


def test_a_missing_index_or_sleeve_close_is_named_before_the_plan():
    from types import SimpleNamespace

    from kite_connect.trading.nse_engine_executor import presence_gaps
    from nse_engine.config import EngineConfig

    cfg = EngineConfig()
    days = pd.bdate_range("2025-01-01", periods=260)
    close = pd.DataFrame({"INFY": 100.0, "GOLDBEES": 60.0, "SILVERBEES": 70.0}, index=days)
    index = pd.DataFrame({"NIFTY50": 24_000.0}, index=days)
    assert presence_gaps(SimpleNamespace(close=close, index_close=index), cfg) == []
    close.loc[days[-1], "SILVERBEES"] = float("nan")
    index.loc[days[-1], "NIFTY50"] = float("nan")
    last = days[-1].date().isoformat()
    assert presence_gaps(SimpleNamespace(close=close, index_close=index), cfg) == [
        f"NIFTY50: no close on {last}", f"SILVERBEES: no close on {last}"]
    index.loc[days[-8:], "NIFTY50"] = float("nan")
    many = presence_gaps(SimpleNamespace(close=close, index_close=index), cfg)[0]
    assert many.endswith(f"{last} and 3 more of the last 200 sessions") and many.count("2026-") + many.count("2025-") == 5
    assert presence_gaps(SimpleNamespace(close=close, index_close=index.rename(columns={"NIFTY50": "X"})), cfg)[0] == (
        "NIFTY50: no index closes in the data")
    assert presence_gaps(SimpleNamespace(close=close.iloc[:0], index_close=index), cfg) == []


def test_the_live_books_equity_history_comes_from_its_snapshots():
    from types import SimpleNamespace

    from kite_connect.trading.nse_engine_executor import cloud_equity_history

    rows = pd.DataFrame({"date": ["2026-11-12", "2026-11-11", "2026-11-12T19:05:00"], "equity": [301.0, 300.0, 302.5]})
    history = cloud_equity_history(SimpleNamespace(read_snapshots=lambda: rows))
    assert list(history.index.strftime("%Y-%m-%d")) == ["2026-11-11", "2026-11-12"] and list(history) == [300.0, 302.5]
    for empty in (None, rows.iloc[:0], rows.drop(columns="equity")):
        assert cloud_equity_history(SimpleNamespace(read_snapshots=lambda e=empty: e)).empty


def test_the_drawdown_rule_is_replayed_over_the_books_own_history_each_night():
    from kite_connect.trading.nse_engine_executor import EngineExecutor
    from nse_engine.config import DrawdownConfig
    from nse_engine.drawdown import HALT, NORMAL

    history = {"2026-11-10": 100.0, "2026-11-11": 98.0, "2026-11-12": 84.0, "2026-11-13": 999.0, "2026-11-09": -5.0}
    rule = DrawdownConfig(enabled=True)
    ex = EngineExecutor(paper=True, equity_history_fn=lambda: history, drawdown_rule=rule)
    decision, frame = ex.drawdown_decision(pd.Timestamp("2026-11-13"), 83.0)        # today's mark replaces today's row
    assert (decision.state, decision.allow_entries, decision.sessions_in_state) == (HALT, False, 2)
    assert decision.peak == 100.0 and decision.drawdown == pytest.approx(0.17) and len(frame) == 4
    calm, _ = ex.drawdown_decision(pd.Timestamp("2026-11-11"), 99.0)
    assert (calm.state, calm.allow_entries, calm.scale) == (NORMAL, True, 1.0)

    def broken():
        raise RuntimeError("snapshots unreadable")

    alone, frame = EngineExecutor(paper=True, equity_history_fn=broken, drawdown_rule=rule).drawdown_decision(
        pd.Timestamp("2026-11-13"), 83.0)
    assert alone.state == NORMAL and len(frame) == 1                                 # the rule sees today only, and says so in the log
    off = EngineExecutor(paper=True, equity_history_fn=lambda: history, drawdown_rule=DrawdownConfig(enabled=False))
    assert off.drawdown_decision(pd.Timestamp("2026-11-13"), 83.0) == (None, None)
