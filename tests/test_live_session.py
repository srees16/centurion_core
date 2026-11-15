"""The live session (kite_connect/trading/live_session.py): the book the session plans from, its stops, what
happened to yesterday's orders, and the checks that run before any order."""
from __future__ import annotations

import pandas as pd
import pytest


DAY, EVE, DP = "2026-11-13", "2026-11-12", 15.93
BUY = {"symbol": "INFY", "side": "BUY", "status": "FILLED", "quantity": 10, "fill_price": 1000.0}


def _ledger(**over) -> dict:
    return {"capital": 300_000.0, "cash": 100_000.0, "positions": {}, "entries": {}, "stops": {}, "stop_basis": {},
            "symbols": [], "session": EVE, **over}


def _cost(value: float, side: str, day: str = DAY, symbol: str = "INFY") -> float:
    from nse_engine.costs import statutory_cost

    return statutory_cost(value, side, pd.Timestamp(day), DP, symbol=symbol)


# ── step 1: the ledger and its fills ────────────────────────────

def test_the_ledger_follows_its_fills_and_books_a_session_once():
    from kite_connect.trading.live_session import SOURCE_EXTERNAL, apply_fills

    led = apply_fills(_ledger(), [BUY], DAY, DP, planned_stops={"INFY": (900.0, [EVE, 1010.0])})
    assert led["positions"] == {"INFY": 10} and led["entries"] == {"INFY": DAY} and led["stops"] == {"INFY": 900.0}
    assert led["stop_basis"] == {"INFY": [EVE, 1010.0]} and led["symbols"] == ["INFY"] and led["session"] == DAY
    assert led["cash"] == pytest.approx(100_000 - 10_000 - _cost(10_000, "BUY"))
    assert apply_fills(led, [BUY], DAY, DP) is led                        # the same session: its fills are already in
    part = apply_fills(led, [{**BUY, "side": "SELL", "quantity": 4, "fill_price": 1100.0}], "2026-11-16", DP)
    assert part["positions"] == {"INFY": 6} and part["entries"] == {"INFY": DAY} and part["stops"] == {"INFY": 900.0}
    assert part["cash"] == pytest.approx(led["cash"] + 4_400 - _cost(4_400, "SELL", "2026-11-16"))
    by_hand = {**BUY, "side": "SELL", "quantity": 50, "fill_price": 1100.0, "source": SOURCE_EXTERNAL}
    out = apply_fills(part, [by_hand], "2026-11-17", DP)                  # a sale is capped at the six held
    assert out["positions"] == out["entries"] == out["stops"] == out["stop_basis"] == {}
    assert out["cash"] == pytest.approx(part["cash"] + 6_600 - _cost(6_600, "SELL", "2026-11-17"))
    assert out["recent_stops"] == {"INFY": "2026-11-17"} and out["symbols"] == ["INFY"]


def test_a_stop_out_starts_its_cooldown_and_old_ones_are_dropped():
    from kite_connect.trading.live_session import apply_fills

    held = _ledger(positions={"INFY": 6}, entries={"INFY": EVE}, stops={"INFY": 900.0})
    stop = {**BUY, "side": "SELL", "quantity": 6, "fill_price": 890.0, "reason": "exit:stop", "decision_date": DAY}
    led = apply_fills(held, [stop], "2026-11-16", DP)
    assert led["recent_stops"] == {"INFY": DAY}                           # from the session the stop was hit
    assert apply_fills(led, [], "2026-12-13", DP)["recent_stops"] == {"INFY": DAY}
    assert apply_fills(led, [], "2026-12-14", DP)["recent_stops"] == {}   # past the thirty days kept
    ignored = [{**BUY, "status": "REJECTED"}, {**BUY, "quantity": 0}, {**BUY, "side": "SELL", "symbol": "TCS"}]
    same = apply_fills(_ledger(), ignored, DAY, DP)
    assert same["positions"] == {} and same["cash"] == 100_000.0
    kept = apply_fills(_ledger(), [BUY], DAY, DP, once=False)             # a correction: the session marker stays
    assert kept["session"] == EVE and kept["positions"] == {"INFY": 10}


def test_a_split_or_bonus_rebases_the_ledger_once_and_a_dividend_is_income(monkeypatch):
    from kite_connect.trading import book_events
    from kite_connect.trading.live_session import apply_ledger_events

    events = {"INFY": {"share_factor": 0.5, "share_keys": ["INFY|bonus|2026-11-13"], "inferred": [], "other": [],
                       "dividends": [(DAY, 8.0, "INFY|div|2026-11-13")]},
              "TCS": {"share_factor": 1.0, "share_keys": [], "inferred": [], "other": ["rights 1:10"], "dividends": []}}
    monkeypatch.setattr(book_events, "events_between", lambda data, symbols, after, upto: events)
    led = _ledger(positions={"INFY": 10, "TCS": 3})
    notes, income, rows = apply_ledger_events(led, None, DAY)
    assert income == 80.0 and led["positions"] == {"INFY": 20, "TCS": 3} and led["cash"] == 100_000.0
    assert led["ca_pending"] == {"INFY": {"ex_date": DAY, "quantity": 20}}
    assert notes == ["corporate action INFY: factor 0.5, 10 -> 20 shares",
                     "corporate action TCS with no share change (rights/demerger/price): rights 1:10"]
    assert [(r["symbol"], r["requested_qty"], r["quantity"], r["source"]) for r in rows] == [
        ("INFY", 10, 10, "corporate_action"), ("INFY", 10, 20, "corporate_action")]
    again = apply_ledger_events(led, None, DAY)                           # each event once
    assert again[1] == 0.0 and again[2] == [] and led["positions"] == {"INFY": 20, "TCS": 3}
    assert apply_ledger_events(_ledger(), None, DAY) == ([], 0.0, [])     # nothing held
    assert apply_ledger_events(_ledger(positions={"INFY": 1}, session=""), None, DAY) == ([], 0.0, [])


def test_a_renamed_stock_moves_in_the_ledger_once_and_two_names_are_never_merged():
    from kite_connect.trading.live_session import rekey_ledger

    led = {"positions": {"OLD": 10, "A": 5, "B": 5}, "entries": {"OLD": "2026-11-02"}, "stops": {"OLD": 90.0},
           "stop_basis": {"OLD": ["2026-11-02", 100.0]}, "symbols": ["OLD", "A", "B"]}
    notes = rekey_ledger(led, {"OLD": "NEW", "A": "B", "GONE": "X", "B": "B"})
    assert led["positions"] == {"NEW": 10, "A": 5, "B": 5} and led["entries"] == {"NEW": "2026-11-02"}
    assert led["stops"] == {"NEW": 90.0} and led["stop_basis"] == {"NEW": ["2026-11-02", 100.0]}
    assert led["symbols"] == ["OLD", "A", "B", "NEW"]                     # the old name stays in scope for its GTT
    assert len(notes) == 2 and notes[0].startswith("A is now B, but the ledger holds both: not merged")
    assert notes[1].startswith("OLD is now NEW")
    assert rekey_ledger(led, {"OLD": "NEW"}) == []


def test_a_stop_follows_the_datas_adjustment_and_a_second_run_changes_nothing():
    from kite_connect.trading.live_session import rescale_stops

    close = pd.DataFrame({"INFY": [500.0, 505.0], "TCS": [3000.0, 3010.0]}, index=pd.to_datetime([EVE, DAY]))
    led = {"stops": {"INFY": 900.0, "TCS": 2700.0, "GONE": 10.0},
           "stop_basis": {"INFY": [EVE, 1000.0], "TCS": [EVE, 3000.0], "GONE": [EVE, 10.0], "TCS2": ["2020-01-01", 5.0]}}
    assert rescale_stops(led, close) == {"INFY": 0.5}                     # a 1:1 bonus since the stop was set
    assert led["stops"] == {"INFY": 450.0, "TCS": 2700.0, "GONE": 10.0} and led["stop_basis"]["INFY"] == [EVE, 500.0]
    assert rescale_stops(led, close) == {}


# ── step 2: what happened to yesterday's orders ─────────────────

def test_fill_rows_carry_the_cost_against_the_open_and_kites_reason():
    from kite_connect.trading.live_session import SOURCE_ENGINE, fills_from_outcomes

    outcomes = [
        {"symbol": "INFY", "side": "BUY", "filled": 10, "average_price": 1010.0, "tag": "T1", "order_id": "1",
         "quantity": 10, "outcome": "complete"},
        {"symbol": "TCS", "side": "SELL", "filled": 2, "average_price": 2970.0, "tag": "T2", "order_id": "2",
         "quantity": 5, "outcome": "partial"},
        {"symbol": "HDFCBANK", "side": "BUY", "filled": 0, "tag": "T3", "order_id": "3", "quantity": 4,
         "outcome": "rejected", "error": "Insufficient funds. Required margin is 5000"},
        {"symbol": "ITC", "side": "BUY", "filled": 0, "tag": "T4", "order_id": "4", "quantity": 1, "outcome": "cancelled"},
        {"status": "unknown", "symbol": "SBIN"}]
    placed = {"T1": {"ref_price": 1000.0, "reason": "entry"}, "T2": {"ref_price": 3000.0, "reason": "exit:stop"}}
    rows = fills_from_outcomes(outcomes, placed, DAY, EVE, {"INFY": 1000.0, "TCS": 3000.0}, DP)
    assert [(r["symbol"], r["status"], r["quantity"], r["requested_qty"]) for r in rows] == [
        ("INFY", "FILLED", 10, 10), ("TCS", "FILLED", 2, 5), ("HDFCBANK", "REJECTED", 0, 4), ("ITC", "CANCELLED", 0, 1)]
    assert all(r["source"] == SOURCE_ENGINE and r["session_date"] == DAY and r["decision_date"] == EVE for r in rows)
    assert rows[0]["impact_bps"] == 100.0 and rows[1]["impact_bps"] == 100.0       # paid above, sold below, the open
    assert rows[0]["costs_inr"] == round(_cost(10_100, "BUY") + 101.0, 2)
    assert rows[1]["costs_inr"] == round(_cost(5_940, "SELL", symbol="TCS") + 59.4, 2)
    assert rows[0]["note"] == "complete (entry)" and rows[1]["note"] == "partial 2/5 (exit:stop)"
    assert rows[1]["reason"] == "exit:stop" and rows[2]["costs_inr"] == 0.0
    assert rows[2]["note"].startswith("rejected: Insufficient funds. Required margin is 5000 (")


def test_a_sale_the_engine_did_not_place_is_booked_with_what_it_filled():
    from kite_connect.trading.live_session import SOURCE_EXTERNAL, external_sells

    def order(**over):
        return {"order_id": "1", "tradingsymbol": "INFY", "transaction_type": "SELL", "product": "CNC", "quantity": 10,
                "filled_quantity": 10, "average_price": 1100.0, "status": "COMPLETE", "tag": "", "trigger_price": 1105.0,
                **over}

    book = [order(),
            order(order_id="2", tradingsymbol="HFCL-BE", filled_quantity=4, status="CANCELLED", average_price=80.0),
            order(order_id="3", tag="NE261113SINFY"), order(order_id="4", transaction_type="BUY"),
            order(order_id="5", product="MIS"), order(order_id="6", status="OPEN"),
            order(order_id="7", filled_quantity=0, status="CANCELLED"), order(order_id="8", tradingsymbol="TCS")]
    rows = external_sells(book, {"INFY", "HFCL"}, DAY, DP)
    assert [(r["order_id"], r["symbol"], r["quantity"], r["fill_price"]) for r in rows] == [
        ("1", "INFY", 10, 1100.0), ("2", "HFCL", 4, 80.0)]                # a part fill sold what it filled (LS2)
    assert all(r["source"] == SOURCE_EXTERNAL and r["side"] == "SELL" and r["status"] == "FILLED" for r in rows)
    assert rows[0]["costs_inr"] == round(_cost(11_000, "SELL"), 2) and rows[0]["ref_price"] == 1105.0
    assert external_sells(None, {"INFY"}, DAY, DP) == []


def test_a_missed_session_is_explained_from_the_pending_orders_and_the_stops():
    """LN-T6: Kite's order book lasts a day, so after a session nobody ran the broker's quantities are explained
    by the orders left pending and the stop GTTs that triggered; what is left is named, never guessed."""
    from types import SimpleNamespace

    from kite_connect.trading.live_session import SOURCE_ENGINE, SOURCE_EXTERNAL, reconcile_missed

    dates = pd.to_datetime([EVE, DAY, "2026-11-16"])
    view = SimpleNamespace(dates=dates, open=pd.DataFrame(
        {"NEW": [100.0, 102.0, 104.0], "ADD": [200.0, 198.0, 210.0], "OUT": [50.0, 49.0, 48.0],
         "STOP": [300.0, 280.0, 290.0], "LOST": [10.0, 10.0, 10.0]}, index=dates))
    led = _ledger(positions={"ADD": 10, "OUT": 20, "STOP": 5, "LOST": 7})
    pending = [{"symbol": "NEW", "side": "BUY", "quantity": 8, "limit_price": 101.0, "decision_date": EVE, "tag": "T1"},
               {"symbol": "ADD", "side": "BUY", "quantity": 5, "limit_price": 199.0, "decision_date": EVE, "tag": "T2"},
               {"symbol": "OUT", "side": "SELL", "quantity": 20, "limit_price": 47.0, "decision_date": EVE, "tag": "T3"},
               {"symbol": "LATE", "side": "BUY", "quantity": 3, "limit_price": 9.0, "decision_date": "2026-11-16"}]
    gtts = [{"symbol": "STOP", "updated_at": f"{DAY} 10:00:00", "trigger": 285.0, "limit": 279.0, "quantity": 5, "id": 77}]
    broker = {"NEW": 8, "ADD": 15, "OUT": 0, "STOP": 0, "LOST": 4}
    rows, entries, cooldown, unexplained = reconcile_missed(led, pending, broker, {"NEW": 101.5}, gtts, view,
                                                            "2026-11-16", DP)
    assert [(r["symbol"], r["side"], r["quantity"], r["fill_price"], r["session_date"], r["source"]) for r in rows] == [
        ("ADD", "BUY", 5, 198.0, DAY, SOURCE_ENGINE),                     # the open, under its limit
        ("NEW", "BUY", 8, 101.5, DAY, SOURCE_ENGINE),                     # a new name: the broker's average price
        ("OUT", "SELL", 20, 49.0, DAY, SOURCE_ENGINE),                    # the open, above its limit
        ("STOP", "SELL", 5, 280.0, DAY, SOURCE_EXTERNAL)]                 # a gap under the trigger: the open
    assert entries == {"ADD": DAY, "NEW": DAY}
    assert cooldown == {"STOP": pd.Timestamp(DAY), "LOST": pd.Timestamp("2026-11-16")}
    assert unexplained == ["LOST: the ledger holds 7, the broker 4 (-3 unexplained)"]
    assert rows[3]["note"] == "GTT stop 77 triggered on a missed session"
    assert rows[2]["costs_inr"] == round(_cost(980.0, "SELL", symbol="OUT"), 2)
    quiet = reconcile_missed(_ledger(positions={"ADD": 10}), [], {"ADD": 10}, {}, [], view, "2026-11-16", DP)
    assert quiet == ([], {}, {}, [])


def test_a_manual_reconcile_sets_the_quantity_and_books_the_difference():
    from kite_connect.trading.live_session import manual_reconcile

    led = _ledger(positions={"INFY": 10}, entries={"INFY": "2026-11-02"}, cash=1_000.0)
    out, rows = manual_reconcile(led, {"INFY": (6, 1100.0), "TCS": (2, 3000.0), "SAME": (0, 5.0)}, DAY, DP)
    assert out["positions"] == {"INFY": 6, "TCS": 2} and out["session"] == EVE and led["positions"] == {"INFY": 10}
    assert [(r["symbol"], r["side"], r["quantity"], r["fill_price"]) for r in rows] == [
        ("INFY", "SELL", 4, 1100.0), ("TCS", "BUY", 2, 3000.0)]
    assert out["cash"] == pytest.approx(1_000 + 4_400 - _cost(4_400, "SELL") - 6_000 - _cost(6_000, "BUY", symbol="TCS"))
    assert out["entries"] == {"INFY": "2026-11-02", "TCS": DAY}


# ── step 3: the book at the close ───────────────────────────────

def test_the_book_is_marked_at_the_close_or_at_cost_when_there_is_none():
    from kite_connect.trading.live_session import mark_book

    holdings = {"INFY": {"quantity": 10, "avg_price": 1000.0, "stop_price": 900.0},
                "NOBAR": {"quantity": 2, "avg_price": 50.0}, "NAN": {"quantity": 1, "avg_price": 20.0}}
    marked = mark_book(holdings, 500.0, pd.Series({"INFY": 1100.0, "NAN": float("nan")}))
    assert marked["equity"] == 11_620.0 and marked["cash"] == 500.0 and marked["invested"] == 11_120.0
    assert marked["positions"]["INFY"] == {"quantity": 10, "close": 1100.0, "value": 11_000.0, "stop": 900.0}
    assert marked["positions"]["NOBAR"]["close"] == 50.0 and marked["positions"]["NAN"]["close"] == 20.0
    assert mark_book({}, 500.0, pd.Series(dtype=float)) == {"equity": 500.0, "cash": 500.0, "invested": 0.0, "positions": {}}


def test_the_snapshot_row_measures_the_day_against_the_last_session_only():
    from kite_connect.trading.live_session import snapshot_row

    marked = {"equity": 294_500.0, "cash": 500.0, "invested": 294_000.0, "positions": {"INFY": {"quantity": 1}}}
    history = pd.Series([300_000.0, 310_000.0], index=pd.to_datetime(["2026-11-11", EVE]))
    row = snapshot_row(DAY, marked, history, 300_000.0, 2)
    assert (row["date"], row["equity"], row["cash"], row["open_positions"], row["closed_today"]) == (
        DAY, 294_500.0, 500.0, 1, 2)
    assert (row["day_pnl"], row["cumulative_pnl"], row["cumulative_pnl_pct"], row["max_drawdown_pct"]) == (
        -15_500.0, -5_500.0, -1.833, 5.0)
    rerun = pd.concat([history, pd.Series([1.0], index=pd.to_datetime([DAY]))])      # the day's own earlier row
    assert snapshot_row(DAY, marked, rerun, 300_000.0, 2) == row
    first = snapshot_row(DAY, marked, pd.Series(dtype="float64", index=pd.DatetimeIndex([])), 300_000.0, 0)
    assert first["day_pnl"] == -5_500.0 and first["max_drawdown_pct"] == 0.0


@pytest.mark.parametrize("sessions_left,sold", [(3, [4, 1]), (2, [5, 2]), (1, [10, 3]), (0, [10, 3])])
def test_a_disconnect_sells_in_equal_parts_and_everything_on_the_last_session(sessions_left, sold):
    from kite_connect.trading.live_session import unwind_orders

    holdings = {"TCS": {"quantity": 3}, "INFY": {"quantity": 10}, "NOBAR": {"quantity": 5}, "NAN": {"quantity": 5}}
    orders, unpriced = unwind_orders(holdings, pd.Series({"INFY": 1000.0, "TCS": 3000.0, "NAN": float("nan")}),
                                     sessions_left)
    assert [(o.symbol, o.side, o.quantity, o.reason) for o in orders] == [
        ("INFY", "SELL", sold[0], "exit:disconnect"), ("TCS", "SELL", sold[1], "exit:disconnect")]
    assert unpriced == ["NAN", "NOBAR"] and all(o.limit_price < o.ref_price for o in orders)
    assert [(o.current_qty, o.target_qty) for o in orders] == [(10, 10 - sold[0]), (3, 3 - sold[1])]


# ── the checks before any order ─────────────────────────────────

@pytest.mark.lesson("L-LS2", pattern="F1", date="2026-10-10", source="LS2",
                    summary="A recorded dry run wrote the live book's ledger and its session marker")
def test_a_dry_run_is_never_recorded():
    from kite_connect.trading.live_session import run_live_session

    with pytest.raises(ValueError, match="never recorded"):
        run_live_session(object(), dry_run=True, record=True)


@pytest.mark.lesson("L-AL4", pattern="F1", date="2026-10-10", source="AL4",
                    summary="A stop sell that Kite rejected was reported as a missing GTT, without Kite's reason")
def test_a_stop_sell_that_failed_is_named_while_the_broker_still_holds_the_stock():
    from kite_connect.trading.live_session import failed_stop_sells

    day = "2026-11-13"
    gtt = {"symbol": "INFY", "status": "triggered", "quantity": 4, "trigger": 1100.0,
           "raw": {"updated_at": f"{day} 10:02:00"}, "order_result": {"order_id": "1", "status": "success"}}
    sell = {"order_id": "1", "tradingsymbol": "INFY", "transaction_type": "SELL", "product": "CNC", "quantity": 4,
            "status": "REJECTED", "status_message": "Insufficient holdings", "tag": ""}
    line = failed_stop_sells([gtt], [sell], {"INFY"}, {"INFY": 4}, day)["INFY"]
    assert line.startswith("STOP SELL NOT DONE: INFY x4 trigger 1,100.00 - Kite REJECTED: Insufficient holdings")
    assert failed_stop_sells([gtt], [sell], {"INFY"}, {"INFY": 0}, day) == {}      # the engine's exit sold it first
    assert failed_stop_sells([gtt], [sell], {"INFY"}, {"INFY": 4}, "2026-11-16") == {}   # another day's
    assert failed_stop_sells([gtt], [sell], {"TCS"}, {"INFY": 4}, day) == {}       # not the ledger's stock
    by_hand = {**sell, "order_id": "9"}
    assert "placed outside the engine" in failed_stop_sells([], [by_hand], {"INFY"}, {"INFY": 4}, day)["INFY"]
    assert failed_stop_sells([], [{**by_hand, "tag": "NE261113SINFY"}], {"INFY"}, {"INFY": 4}, day) == {}


@pytest.mark.lesson("L-LNT33", pattern="F2", date="2026-10-13", source="LN-T33",
                    summary="A late backup run tested an expired Kite login before asking whether the session had run")
def test_a_backup_run_asks_whether_the_session_ran_before_it_tests_the_login():
    import inspect

    from kite_connect.trading import live_session as ls

    class Book:
        def read_state(self):
            return {ls.LIVE_DRY_LAST_SESSION_KEY: "2026-10-12"}

    assert ls._already_ran(Book(), True, "2026-10-12")
    assert not ls._already_ran(Book(), False, "2026-10-12") and not ls._already_ran(Book(), True, "2026-10-13")
    assert not ls._already_ran(Book(), True, None)
    source = inspect.getsource(ls.main)
    assert 0 < source.index("_already_ran(") < source.index("kite_from_stored_token(")


@pytest.mark.lesson("L-LS1", pattern="F1", date="2026-10-10", source="LS1",
                    summary="A live position whose stop GTT was gone, unreadable or left low was never stopped out")
def test_a_live_stop_is_never_lowered_and_never_dropped():
    from kite_connect.trading.live_session import scoped_book

    names = ("LOW", "GONE", "UNREAD", "HIGHER")
    ledger = {"positions": dict.fromkeys(names, 10), "entries": dict.fromkeys(names, "2026-11-02"),
              "stops": dict.fromkeys(names, 100.0), "cash": 1_000.0, "session": "2026-11-13"}
    held = {"quantity": 10, "avg_price": 120.0}
    broker = {"LOW": {**held, "stop_price": 95.0},                        # a GTT the broker did not raise
              "GONE": {**held, "stop_price": None},                       # triggered without a sale, expired or deleted
              "UNREAD": {**held, "stop_price": None, "stops_unknown": True},
              "HIGHER": {**held, "stop_price": 104.0}}
    critical = []
    holdings, cash, alerts = scoped_book(ledger, broker, 5_000.0, critical=critical)
    assert {s: h["stop_price"] for s, h in holdings.items()} == {"LOW": 100.0, "GONE": 100.0, "UNREAD": 100.0, "HIGHER": 104.0}
    assert all(h["entry_date"] == "2026-11-02" and h["quantity"] == 10 for h in holdings.values()) and cash == 1_000.0
    assert any(a.startswith("LOW: the broker's stop GTT 95.00 is below the book's last stop 100.00") for a in alerts)
    assert [a.split(":")[0] for a in critical] == ["GONE", "UNREAD"]      # the two that need the owner today


# ── the scheduled run: live_session.main ────────────────────────

class _StateBook:
    """The live book's state and fills, in memory."""

    def __init__(self, **state):
        self.state, self.fills = dict(state), []

    def read_state(self):
        return dict(self.state)

    def sync_state(self, values):
        self.state.update(values)
        return True

    def sync_fills(self, rows):
        self.fills += list(rows)
        return True


@pytest.fixture
def night(monkeypatch):
    """The 19:00 job's surroundings: a book, the store's session, a Kite login, and what main() would start."""
    from types import SimpleNamespace

    from kite_connect.auth import daily_login as dl
    from kite_connect.trading import live_session as ls, public_log

    n = SimpleNamespace(ls=ls, book=_StateBook(), kite=object(), accepted=True, sessions=[], skips=[], logins=0,
                        egress=("140.238.1.1", True))

    def stored_login(book, key=None):
        n.logins += 1
        return n.kite

    def run(kite, **kw):
        n.sessions.append(kw)
        return {"session": DAY, "mode": "dry_run" if kw["dry_run"] else "live", "fills": [1, 2], "results": [1],
                "alerts": ["INFY: something private"]}

    monkeypatch.setattr(public_log, "install_public_log_filter", lambda: None)
    monkeypatch.setattr(ls, "live_book", lambda schema=None: n.book)
    monkeypatch.setattr(ls, "_store_session", lambda as_of=None: DAY)
    monkeypatch.setattr(ls, "_login_accepted", lambda kite: n.accepted)
    monkeypatch.setattr(ls, "_email_skip", lambda message, label="": n.skips.append(message))
    monkeypatch.setattr(ls, "run_live_session", run)
    monkeypatch.setattr(dl, "kite_from_stored_token", stored_login)
    monkeypatch.setattr(dl, "check_egress", lambda proxy: n.egress)
    monkeypatch.delenv(dl.ENV_PROXY, raising=False)
    return n


def test_a_session_that_already_ran_starts_nothing_and_asks_kite_nothing(night, capsys):
    night.book.state[night.ls.LIVE_DRY_LAST_SESSION_KEY] = DAY
    assert night.ls.main(["--stored-token", "--dry-run"]) == 0
    assert night.logins == 0 and night.sessions == [] and night.skips == []
    assert '"skipped": "already ran"' in capsys.readouterr().out


def test_the_dry_run_and_the_live_run_keep_separate_markers(night, monkeypatch):
    from kite_connect.auth import daily_login as dl

    night.book.state[night.ls.LIVE_DRY_LAST_SESSION_KEY] = DAY            # tonight's dry run is done
    monkeypatch.setenv(dl.ENV_PROXY, "http://10.0.0.1:3128")
    assert night.ls.main(["--stored-token"]) == 0                         # the live session has not run
    assert len(night.sessions) == 1 and night.sessions[0]["dry_run"] is False


def test_no_login_skips_the_session_and_emails_once_however_many_backups_run(night, capsys):
    night.kite = None
    for _ in range(3):                                                    # the 19:00 run and two backup crons
        assert night.ls.main(["--stored-token", "--dry-run"]) == 0
    assert night.sessions == [] and len(night.skips) == 1
    assert night.skips[0].startswith("No Kite login today, so the live session was skipped: no orders were placed.")
    assert night.book.state[night.ls.LIVE_SKIP_NOTIFIED_KEY] == DAY
    assert capsys.readouterr().out.count("GTT stops stay active at Zerodha") == 3
    quiet = _StateBook()
    night.book = quiet
    assert night.ls.main(["--stored-token", "--dry-run", "--no-email"]) == 0
    assert len(night.skips) == 1 and quiet.state == {}                    # --no-email: nothing sent, nothing marked


def test_a_login_kite_no_longer_accepts_is_a_skip_with_its_own_words(night):
    night.accepted = False
    assert night.ls.main(["--stored-token", "--dry-run"]) == 0
    assert night.sessions == [] and night.skips[0].startswith("Kite rejected today's login")


def test_real_orders_need_the_registered_static_ip(night, monkeypatch):
    from kite_connect.auth import daily_login as dl

    with pytest.raises(SystemExit, match="real orders need CENTURION_KITE_PROXY"):
        night.ls.main(["--stored-token"])                                 # live, no proxy
    monkeypatch.setenv(dl.ENV_PROXY, "http://10.0.0.1:3128")
    monkeypatch.setenv(dl.ENV_STATIC_IP, "140.238.1.1")
    night.egress = ("8.8.8.8", False)
    with pytest.raises(SystemExit, match="real orders refused: egress IP 8.8.8.8 is not the IP registered"):
        night.ls.main(["--stored-token"])
    assert night.sessions == []
    assert night.ls.main(["--stored-token", "--dry-run"]) == 0            # a dry run goes on and says so
    notes = night.sessions[0]["extra_notes"]
    assert notes[0] == "Kite calls through the static-IP proxy, egress IP 8.8.8.8" and notes[1].startswith("WARNING: egress IP")


def test_the_scheduled_run_starts_one_session_and_prints_counts_only(night, capsys, monkeypatch):
    from kite_connect.auth import daily_login as dl

    monkeypatch.setenv(dl.ENV_PROXY, "http://10.0.0.1:3128")
    assert night.ls.main(["--stored-token", "--capital", "300000"]) == 0
    (kw,) = night.sessions
    assert kw["dry_run"] is False and kw["once_per_session"] is True and kw["book"] is night.book
    assert kw["capital"] == 300_000.0 and kw["email"] is True and kw["label"] == "" and kw["unwind_sessions"] == 0
    out = capsys.readouterr().out
    assert '"fills": 2' in out and '"alerts": 1' in out and "INFY" not in out     # the Actions log is public


def test_a_manual_reconcile_from_the_command_line_books_the_difference(night, capsys):
    import json

    night.book.state[night.ls.LIVE_LEDGER_KEY] = json.dumps(_ledger(positions={"INFY": 10}, entries={"INFY": EVE}))
    assert night.ls.main(["--reconcile", "INFY=6@1100", "tcs=2@3000"]) == 0
    led = json.loads(night.book.state[night.ls.LIVE_LEDGER_KEY])
    assert led["positions"] == {"INFY": 6, "TCS": 2} and len(night.book.fills) == 2 and night.logins == 0
    assert '"positions": {"INFY": 6, "TCS": 2}' in capsys.readouterr().out
    for bad in ("INFY", "INFY=x@5", "INFY=-1@5", "INFY=5@0", "INFY=5@nan", "INFY=5@inf"):
        with pytest.raises(SystemExit):
            night.ls.main(["--reconcile", bad])
    assert json.loads(night.book.state[night.ls.LIVE_LEDGER_KEY])["positions"] == {"INFY": 6, "TCS": 2}


def test_clearing_the_step_down_needs_a_reason(night, capsys):
    from nse_engine.capital_ladder import MC_STEP_DOWN_KEY

    night.book.state[MC_STEP_DOWN_KEY] = "2026-11-12"
    with pytest.raises(SystemExit):
        night.ls.main(["--clear-mc-step-down", "  "])
    assert night.book.state[MC_STEP_DOWN_KEY] == "2026-11-12"
    assert night.ls.main(["--clear-mc-step-down", "reviewed with the owner"]) == 0
    assert night.book.state[MC_STEP_DOWN_KEY] == "" and '"cleared": true' in capsys.readouterr().out
