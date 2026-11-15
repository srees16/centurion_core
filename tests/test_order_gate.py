"""The order gate: every order passes one gate, so a repeated request or a lost answer never trades twice (RB5);
the manual route goes through it with its value limits; sends are paced under Kite's rate limit; and a game-day
replay leaves the gate as it found it (GD2)."""
from __future__ import annotations

import os

import numpy as np
import pytest

from conftest import _Broker, _post, LIMIT


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


@pytest.mark.parametrize("order_type, extra, protected", [
    ("MARKET", {}, True), ("SL-M", {"trigger_price": 101.0}, True),
    ("LIMIT", {"price": 100.0}, False), ("SL", {"price": 100.0, "trigger_price": 101.0}, False),
])
def test_market_and_sl_m_orders_carry_market_protection(gate, order_type, extra, protected):
    """Zerodha, 5 Nov 2026: an API MARKET or SL-M order with market_protection 0 is rejected by the exchanges."""
    class Recording(_Broker):
        def place_order(self, **params):
            self.params = params
            return super().place_order(**params)

    broker = Recording()
    out = gate.place_order(broker, symbol="RELIANCE", exchange="NSE", transaction_type="SELL", quantity=5,
                           order_type=order_type, is_exit=True, **extra)
    assert out["success"] is True
    assert broker.params.get("market_protection") == (gate.MARKET_PROTECTION if protected else None)
    assert gate.MARKET_PROTECTION == -1                   # kiteconnect's MARKET_PROTECTION_AUTO, never 0


def test_an_after_market_market_order_on_an_index_option_never_reaches_the_broker(gate):
    broker = _Broker()
    order = dict(exchange="NFO", transaction_type="SELL", quantity=75, order_type="MARKET", product="NRML",
                 is_exit=True, variety="amo")
    out = gate.place_order(broker, symbol="NIFTY26NOV24000CE", **order)
    assert out["success"] is False and "LIMIT" in out["error"] and broker.sent == 0
    assert gate.place_order(broker, symbol="NIFTY26NOV24000CE", **{**order, "order_type": "LIMIT", "price": 50.0})[
        "success"] is True


@pytest.mark.lesson("L-RB5", pattern="F1", date="2026-10-30", source="RB5",
                    summary="A manual order whose answer was lost was sent again: the browser repeated the request")
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


def _borrowed():
    """What a game-day replay patches: the module attributes it replaces."""
    import database.connection as dbc
    from kite_connect.trading import kill_switch, order_service
    from services.notifications import alerts as al

    return {"persist": order_service._persist_to_db, "email": order_service._send_order_email,
            "market_open": order_service._is_nse_market_open, "pace": order_service.ORDERS_PER_SECOND,
            "kill": kill_switch.state, "alert": al.alert, "db": dbc.get_db_manager}


@pytest.mark.lesson('L-GD2',
                    pattern='F1',
                    date='2026-11-09',
                    source='GD2',
                    summary='The game-day replay left its patches in place and broke later tests')
def test_a_replays_isolation_is_undone_and_pacing_works_again(monkeypatch):
    """GD2: _isolate patched order pacing (10 -> 10,000 a second), the database, alerts, the kill switch and the
    environment for good, so a test session that ran the slow game-day test saw the pacing tests fail later."""
    from kite_connect.trading import nse_instruments, order_service
    from tools import game_day as gd

    monkeypatch.setenv("CENTURION_PAPER_TRADE", "true")
    monkeypatch.delenv("CENTURION_GO_LIVE_OVERRIDE", raising=False)
    monkeypatch.setitem(nse_instruments._cache, "2026-11-09", {"INFY": 0.05})
    before, cache_before = _borrowed(), dict(nse_instruments._cache)
    alerts = []
    undo = gd._isolate(alerts)
    assert order_service.ORDERS_PER_SECOND == 10_000 and os.environ["CENTURION_GO_LIVE_OVERRIDE"] == "true"
    assert nse_instruments._cache == {}
    with pytest.raises(RuntimeError, match="never opened"):
        __import__("database.connection", fromlist=["get_db_manager"]).get_db_manager()
    undo()
    after = _borrowed()
    assert all(after[k] is before[k] or after[k] == before[k] for k in before), [k for k in before if after[k] != before[k]]
    assert os.environ["CENTURION_PAPER_TRADE"] == "true" and "CENTURION_GO_LIVE_OVERRIDE" not in os.environ
    assert dict(nse_instruments._cache) == cache_before
    order_service._order_times.clear()
    clock = _FakeClock()
    for _ in range(11):
        order_service._pace_orders(clock, clock.sleep)
    assert clock.now == pytest.approx(1001.0)                  # the eleventh order waits its second again


def test_a_replay_that_fails_still_puts_everything_back(monkeypatch):
    from tools import game_day as gd

    before = _borrowed()
    monkeypatch.setattr(gd, "_replay", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("replay broke")))
    with pytest.raises(RuntimeError, match="replay broke"):
        gd.run("faults", "2024-05-02", "2024-05-31", {})
    after = _borrowed()
    assert all(after[k] is before[k] or after[k] == before[k] for k in before)


@pytest.mark.parametrize("ssh_down, egress, backup_ip, expect", [
    ("", {"1080": "1.1.1.1", "1081": "2.2.2.2"}, "2.2.2.2", ("socks5h://127.0.0.1:1080", "1.1.1.1", 0)),
    ("primary", {"1081": "2.2.2.2"}, "2.2.2.2", ("socks5h://127.0.0.1:1081", "2.2.2.2", 0)),
    ("", {"1080": "9.9.9.9", "1081": "2.2.2.2"}, "2.2.2.2", ("socks5h://127.0.0.1:1081", "2.2.2.2", 0)),
    ("primary backup", {}, "2.2.2.2", ("", "1.1.1.1", 1)),
    ("primary", {"1081": "2.2.2.2"}, "", ("", "1.1.1.1", 1)),
], ids=["primary", "primary-down", "primary-wrong-ip", "both-down", "backup-unregistered"])
def test_the_kite_tunnel_falls_back_to_the_secondary_static_ip(tmp_path, ssh_down, egress, backup_ip, expect):
    """ZR2: deployment/oracle-proxy/static_ip_tunnel.sh takes the primary host only when a call through it leaves from
    its registered IP, else the secondary's; it exports the proxy and the IP in use, and fails when neither works."""
    import shutil
    import subprocess
    from pathlib import Path

    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    stubs = tmp_path / "bin"
    stubs.mkdir()
    (stubs / "ssh").write_text('#!/bin/sh\nfor a; do last="$a"; done\n'
                               'case " $SSH_DOWN " in *" ${last#kitetunnel@} "*) exit 255;; esac\nexit 0\n')
    (stubs / "curl").write_text('#!/bin/sh\nwhile [ $# -gt 0 ]; do [ "$1" = --socks5-hostname ] && p="${2##*:}"; '
                                'shift; done\neval "ip=\\${EGRESS_$p:-}"\n[ -n "$ip" ] && echo "$ip" || exit 7\n')
    (stubs / "timeout").write_text('#!/bin/sh\nshift\nexec "$@"\n')
    (stubs / "pgrep").write_text("#!/bin/sh\nexit 1\n")
    for f in stubs.iterdir():
        f.chmod(0o755)
    env = {"PATH": f"{stubs}:/usr/bin:/bin", "HOME": str(tmp_path), "SSH_DOWN": ssh_down,
           "CENTURION_PROXY_SSH_KEY": "key", "CENTURION_PROXY_HOST": "kitetunnel@primary",
           "CENTURION_PROXY_HOST_KEY": "primary ssh-ed25519 AAAA", "CENTURION_KITE_STATIC_IP": "1.1.1.1",
           "CENTURION_PROXY_HOST_2": "kitetunnel@backup", "CENTURION_PROXY_HOST_KEY_2": "backup ssh-ed25519 BBBB",
           "CENTURION_KITE_STATIC_IP_2": backup_ip, **{f"EGRESS_{p}": ip for p, ip in egress.items()}}
    script = Path(__file__).resolve().parents[1] / "deployment" / "oracle-proxy" / "static_ip_tunnel.sh"
    out = subprocess.run(["bash", "-c", f'. "{script}"; kite_tunnel; rc=$?; '
                                        'printf "%s|%s|%s" "${CENTURION_KITE_PROXY:-}" "$CENTURION_KITE_STATIC_IP" "$rc"'],
                         env=env, capture_output=True, text=True, timeout=60)
    proxy, ip, rc = out.stdout.strip().splitlines()[-1].split("|")
    assert (proxy, ip, int(rc)) == expect, out.stdout + out.stderr
    assert "1.1.1.1" not in out.stdout.replace("|1.1.1.1|", "") and "2.2.2.2" not in out.stdout.replace("|2.2.2.2|", "")


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
