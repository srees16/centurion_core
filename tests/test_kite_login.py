"""The daily Kite login (kite_connect/auth/daily_login.py): the day's access token is stored only encrypted, is
usable from the login until 06:00 IST the next morning and never after, belongs to one Zerodha user, and its
reminder goes out only on a trading day when nobody has logged in."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

pytest.importorskip("cryptography")                       # ships with kiteconnect, not with the core install

from kite_connect.auth import daily_login as dl  # noqa: E402

FRIDAY = datetime(2026, 11, 13, 19, 0, tzinfo=dl.IST)


class _Book:
    """The live book's state table, in memory."""

    def __init__(self, writes: bool = True):
        self.state, self.writes = {}, writes

    def read_state(self):
        return dict(self.state)

    def sync_state(self, values):
        if self.writes:
            self.state.update(values)
        return self.writes


class _Kite:
    made: list = []

    def __init__(self, api_key=None, proxies=None):
        self.api_key, self.proxies, self.token = api_key, proxies, None
        _Kite.made.append(self)

    def generate_session(self, request_token, api_secret=None):
        return {"user_id": "AB1234", "access_token": f"access-for-{request_token}"}

    def set_access_token(self, token):
        self.token = token


@pytest.fixture
def key(monkeypatch):
    from cryptography.fernet import Fernet

    value = Fernet.generate_key().decode()
    monkeypatch.setenv(dl.ENV_TOKEN_KEY, value)
    monkeypatch.delenv(dl.ENV_USER, raising=False)
    _Kite.made = []
    return value


@pytest.mark.parametrize("now,start", [
    (datetime(2026, 11, 13, 5, 59, tzinfo=dl.IST), datetime(2026, 11, 12, 6, 0, tzinfo=dl.IST)),
    (datetime(2026, 11, 13, 6, 0, tzinfo=dl.IST), datetime(2026, 11, 13, 6, 0, tzinfo=dl.IST)),
    (datetime(2026, 11, 13, 23, 59, tzinfo=dl.IST), datetime(2026, 11, 13, 6, 0, tzinfo=dl.IST)),
])
def test_a_token_day_runs_from_six_to_six(now, start):
    assert dl.token_window(now) == (start, start + timedelta(days=1))


def test_the_token_is_stored_encrypted_and_read_back_only_on_its_own_day(key):
    book = _Book()
    stored = dl.save_token("secret-access-token", "AB1234", login_at=FRIDAY.replace(hour=9), book=book)
    assert stored == {dl.KEY_LOGIN_AT: "2026-11-13T09:00:00+05:30", dl.KEY_USER: "AB1234"}     # never the token
    assert "secret-access-token" not in str(book.state) and book.state[dl.KEY_TOKEN]
    assert dl.load_token(book, FRIDAY) == "secret-access-token"
    assert dl.load_token(book, datetime(2026, 11, 14, 5, 59, tzinfo=dl.IST)) == "secret-access-token"
    assert dl.load_token(book, datetime(2026, 11, 14, 6, 0, tzinfo=dl.IST)) is None            # expired at 06:00
    status = dl.token_status(book, FRIDAY)
    assert status == {"valid": True, "login_at": "2026-11-13T09:00:00+05:30", "user_id": "AB1234",
                      "expires": "2026-11-14T06:00:00+05:30"}


def test_a_login_before_six_belongs_to_the_day_before(key):
    book = _Book()
    dl.save_token("t", "AB1234", login_at=FRIDAY.replace(hour=5, minute=30), book=book)
    assert dl.load_token(book, FRIDAY.replace(hour=5, minute=45)) == "t"
    assert dl.load_token(book, FRIDAY) is None and dl.token_status(book, FRIDAY)["valid"] is False


def test_a_missing_key_a_wrong_key_or_a_failed_write_is_loud(key, monkeypatch):
    from cryptography.fernet import Fernet, InvalidToken

    book = _Book()
    dl.save_token("t", "AB1234", login_at=FRIDAY, book=book)
    with pytest.raises(InvalidToken):
        dl.load_token(book, FRIDAY, key=Fernet.generate_key().decode())
    with pytest.raises(RuntimeError, match="could not store the Kite token"):
        dl.save_token("t", "AB1234", login_at=FRIDAY, book=_Book(writes=False))
    monkeypatch.delenv(dl.ENV_TOKEN_KEY)
    with pytest.raises(RuntimeError, match="CENTURION_KITE_TOKEN_KEY is not set"):
        dl.save_token("t", "AB1234", login_at=FRIDAY, book=_Book())


@pytest.mark.parametrize("state", [{}, {dl.KEY_TOKEN: "x"}, {dl.KEY_TOKEN: "x", dl.KEY_LOGIN_AT: "not a time"},
                                   {dl.KEY_LOGIN_AT: "2026-11-13T09:00:00+05:30"}])
def test_a_state_without_a_token_or_a_readable_login_time_is_no_login(state, key):
    book = _Book()
    book.state = state
    assert dl.token_status(book, FRIDAY)["valid"] is False and dl.load_token(book, FRIDAY) is None
    assert dl.kite_from_stored_token(book, FRIDAY, kite_factory=_Kite) is None and _Kite.made == []


def test_the_callback_stores_the_token_of_the_expected_user_only(key, monkeypatch):
    book = _Book()
    out = dl.exchange_and_store("req1", key="app-key", secret="app-secret", book=book, kite_factory=_Kite)
    assert out["user_id"] == "AB1234" and out["kite"].token == "access-for-req1" and dl.KEY_TOKEN not in out
    assert dl.load_token(book) == "access-for-req1"
    other = _Book()
    with pytest.raises(PermissionError, match="logged in as AB1234, but this app accepts only ZZ9999"):
        dl.exchange_and_store("req2", key="app-key", secret="app-secret", expected_user="ZZ9999", book=other,
                              kite_factory=_Kite)
    assert other.state == {}                                              # nothing stored for a stranger
    monkeypatch.setenv(dl.ENV_USER, "ZZ9999")
    with pytest.raises(PermissionError):
        dl.check_user("AB1234")
    dl.check_user("ZZ9999")
    dl.check_user("anyone", expected="")                                  # no expected user set: any login
    monkeypatch.delenv("ZERODHA_API_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="ZERODHA_API_KEY / ZERODHA_API_SECRET"):
        dl.exchange_and_store("req3", key="app-key", book=_Book(), kite_factory=_Kite)


def test_the_evening_session_gets_a_kite_from_todays_token_through_the_proxy(key, monkeypatch):
    book = _Book()
    dl.save_token("todays", "AB1234", login_at=FRIDAY.replace(hour=9), book=book)
    monkeypatch.setenv("ZERODHA_API_KEY", "own-key")
    kite = dl.kite_from_stored_token(book, FRIDAY, proxy_url="http://10.0.0.1:3128", kite_factory=_Kite)
    assert (kite.api_key, kite.token) == ("own-key", "todays")
    assert kite.proxies == {"http": "http://10.0.0.1:3128", "https": "http://10.0.0.1:3128"}
    family = dl.kite_from_stored_token(book, FRIDAY, proxy_url="", kite_factory=_Kite, key="their-key")
    assert family.api_key == "their-key" and family.proxies is None
    assert dl.kite_from_stored_token(book, FRIDAY + timedelta(days=1), kite_factory=_Kite) is None


def test_the_registered_ip_is_checked_against_what_zerodha_sees(monkeypatch):
    monkeypatch.setattr(dl, "egress_ip", lambda proxy_url=None: "140.238.1.1")
    assert dl.check_egress(expected="140.238.1.1") == ("140.238.1.1", True)
    assert dl.check_egress(expected="140.238.9.9") == ("140.238.1.1", False)
    assert dl.check_egress(expected="") == ("140.238.1.1", True)          # none registered: nothing to compare
    monkeypatch.setattr(dl, "egress_ip", lambda proxy_url=None: None)
    assert dl.check_egress(expected="140.238.1.1") == (None, False)


def test_the_reminder_goes_out_only_on_a_trading_day_with_no_login(key, monkeypatch):
    sent = []

    def send(subject, html):
        sent.append((subject, html))
        return True

    book, url = _Book(), "https://kite.zerodha.com/connect/login?v=3&api_key=k"
    morning = FRIDAY.replace(hour=9)
    monkeypatch.delenv(dl.ENV_LIVE_MODE, raising=False)
    assert dl.remind(book, morning, send=send, url=url) == "live mode is 'off': no reminder"
    saturday = dl.remind(book, morning + timedelta(days=1), send=send, url=url, mode="live")
    assert saturday == "2026-11-14 is not an NSE trading day: no reminder" and sent == []
    assert "not an NSE trading day" in dl.remind(book, datetime(2026, 11, 10, 9, 0, tzinfo=dl.IST), send=send, url=url,
                                                 mode="live")           # a weekday holiday
    out = dl.remind(book, morning, send=send, url=url, mode="dry_run")
    assert out == "reminder sent: Kite Login Request Fri 13 Nov (dry run)"
    assert len(sent) == 1 and f'href="{url}"' in sent[0][1] and "Friday 13 Nov" in sent[0][1]
    dl.save_token("t", "AB1234", login_at=morning, book=book)
    assert dl.remind(book, FRIDAY, send=send, url=url, mode="live").startswith("already logged in at 2026-11-13T09:00")
    assert len(sent) == 1
    forced = dl.remind(book, FRIDAY, send=send, url=url, mode="live", force=True, holder="A <b>")
    assert forced == "reminder sent: Kite Login Request Fri 13 Nov (live) - A <b>" and "A &lt;b&gt;'s account" in sent[1][1]
    failed = dl.remind(_Book(), morning, send=lambda s, h: False, url=url, mode="live")
    assert failed.startswith("reminder NOT sent (email failed)")
    monkeypatch.delenv("ZERODHA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ZERODHA_API_KEY is not set"):
        dl.remind(_Book(), morning, send=send, mode="live")
    assert dl.login_url("k") == url
