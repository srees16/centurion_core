"""The kill switch: a halt reaches what is already at the broker, cancelling Centurion's working buys and nothing
else, while a read that fails adds no new risk (RB6)."""
from __future__ import annotations

import numpy as np
import pytest

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


@pytest.mark.lesson("L-RB6", pattern="F1", date="2026-10-30", source="RB6",
                    summary="The kill switch left a queued after-market buy to execute at the next open")
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
