"""The web app's API (api/): every request needs a signed-in session, a signed-up user never reaches the
operator's broker routes, and only an admin can trade (trackers S1, S2, MU2).  The checks stop at the session
middleware: no handler runs, so nothing is traded, read from a broker or written to a database."""
from __future__ import annotations

import warnings

import pytest

#: Reached without a session, on purpose.  A route joins this list only by a decision recorded in the tracker.
PUBLIC = {
    ("POST", "/api/v1/auth/login"), ("POST", "/auth/login"), ("POST", "/api/v1/auth/logout"),
    ("POST", "/api/v1/auth/signup"), ("POST", "/api/v1/auth/activate"), ("POST", "/api/v1/auth/resend-activation"),
    ("POST", "/api/v1/auth/forgot-password"), ("POST", "/api/v1/auth/reset-password"),
    ("POST", "/api/v1/auth/newsletter/unsubscribe"),                     # NL1: a signed ticket from the email
    ("POST", "/stream/postback"), ("GET", "/ind-stocks/auth/callback"), ("POST", "/ind-stocks/auth/consent"),
    ("GET", "/"), ("HEAD", "/"), ("GET", "/health"), ("HEAD", "/health"), ("GET", "/favicon.ico"),
    ("GET", "/auth/login"), ("GET", "/auth/logout"), ("GET", "/docs"), ("GET", "/redoc"), ("GET", "/openapi.json"),
}


@pytest.fixture(scope="module")
def api():
    """(the API module, a client, every documented (method, path) with its path parameters filled)."""
    pytest.importorskip("fastapi")                        # the web stack is not in the core install
    pytest.importorskip("httpx")
    pytest.importorskip("bcrypt")
    from fastapi.testclient import TestClient

    import api.main as main

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        paths = main.app.openapi()["paths"]
    operations = sorted({(method.upper(), path.replace("{", "").replace("}", ""))
                         for path, methods in paths.items() for method in methods
                         if method.upper() in ("GET", "POST", "PUT", "PATCH", "DELETE")})
    return main, TestClient(main.app), operations         # no "with": the start-up work never runs


def _bearer(role: str) -> dict:
    from api.auth import create_session_token

    return {"Authorization": f"Bearer {create_session_token('tester', role)}"}


def test_the_public_routes_are_the_listed_ones_and_no_more(api):
    main, _, operations = api
    assert set(main._PUBLIC) == PUBLIC
    assert len(operations) > 100, "the app's routers did not load: this check would pass on an empty app"


def test_a_500_answers_a_generic_message_and_logs_the_cause(api, monkeypatch, caplog):
    """RF0 (AUDIT.md X9): 96 routes raise a 500 with an exception's text, which can name hosts, ports and paths.
    The app's handler logs that text and answers a generic message, so it never reaches the caller."""
    import sys
    from types import SimpleNamespace

    def failing_download(*args, **kwargs):
        raise RuntimeError("connect db.internal:5432 failed")

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=failing_download))
    with caplog.at_level("ERROR"):
        r = api[1].get("/api/v1/market/ticker-prices", params={"symbols": "RF0X9CHECK"}, headers=_bearer("user"))
    assert r.status_code == 500 and r.json() == {"detail": "Internal server error"}
    assert "db.internal" not in r.text and "db.internal:5432" in caplog.text


@pytest.mark.lesson("L-S2", pattern="F1", date="2026-10-07", source="S2",
                    summary="API routes answered without a session; nothing walked every route to confirm it asks")
def test_every_route_asks_for_a_session(api):
    _, client, operations = api
    open_ = [(m, p) for m, p in operations if (m, p) not in PUBLIC and client.request(m, p).status_code != 401]
    assert open_ == [], "these answered without a session"
    assert client.get("/api/v1/kite/holdings", headers={"Authorization": "Bearer forged"}).status_code == 401
    assert client.get("/api/v1/kite/holdings?token=forged").status_code == 401


def test_a_signed_up_user_never_reaches_the_operators_broker_routes(api):
    main, client, operations = api
    user = _bearer("user")
    denied = [(m, p) for m, p in operations if p.startswith(main._USER_DENIED_PREFIXES)
              and not p.startswith(main._USER_OWN_PREFIXES)]
    assert len(denied) > 10
    assert [(m, p) for m, p in denied if client.request(m, p, headers=user).status_code != 403] == []
    writes = [(m, p) for m, p in operations if m != "GET" and p.startswith(main._USER_DENIED_WRITES)]
    assert writes and all(client.request(m, p, headers=user).status_code == 403 for m, p in writes)


def test_only_an_admin_can_trade_or_change_a_broker_account(api):
    main, client, operations = api
    trades = [(m, p) for m, p in operations if m != "GET" and p.startswith(main._ADMIN_WRITE_PREFIXES)
              and not p.startswith(main._USER_OWN_PREFIXES)]
    assert ("POST", "/api/v1/kite/orders") in trades
    for role in ("user", "viewer", "analyst", ""):
        refused = [client.request(m, p, headers=_bearer(role)).status_code for m, p in trades]
        assert set(refused) == {403}, role


def test_heavy_work_is_limited_per_user_and_never_for_another(api, monkeypatch):
    main, _, _ = api
    monkeypatch.setattr(main, "_heavy_calls", {})
    assert all(main._heavy_allowed("a") for _ in range(main.HEAVY_LIMIT))
    assert not main._heavy_allowed("a") and main._heavy_allowed("b")


# ── the Kite login callback and the terms page: public routes that store the day's token ──

@pytest.fixture
def login(api, monkeypatch):
    """The callback's surroundings: accounts by id, what the token exchange returns, and the sessions it opened."""
    from types import SimpleNamespace

    import api.routers.ind_stocks as route
    from kite_connect.auth import accounts, terms

    ann = accounts.Account("ann_lee", "Ann <Lee>", "AB1234", "key")
    n = SimpleNamespace(client=api[1], opened=[], consents=[], exchange=None, record=None, terms=terms,
                        known={"": accounts.primary_account(), "ann_lee": ann})

    def get(account_id, book=None):
        if account_id not in n.known:
            raise accounts.AccountError(f"no account {account_id!r}")
        return n.known[account_id]

    def exchange(acct, request_token, book=None, kite_factory=None):
        if n.exchange is not None:
            raise n.exchange
        return {"kite": f"kite-for-{request_token}", "user_id": "OWN001" if acct.is_primary else "AB1234",
                "kite_login_at": "2026-11-13T09:05:00+05:30"}

    def record_consent(account_id, user_id, version, book=None, now=None):
        if n.record is not None:
            raise n.record
        n.consents.append((account_id, user_id, version))
        return accounts.Account(account_id, "Ann <Lee>", user_id, "key", consent_version=version, consent_by=user_id,
                                consent_at="2026-11-13T09:06:00+05:30")

    monkeypatch.setattr(accounts, "get", get)
    monkeypatch.setattr(accounts, "exchange", exchange)
    monkeypatch.setattr(accounts, "record_consent", record_consent)
    monkeypatch.setattr(route, "set_kite_session", n.opened.append)
    return n


def test_the_owners_login_stores_the_token_and_opens_the_apis_kite_session(login):
    page = login.client.get("/ind-stocks/auth/callback?request_token=req1&status=success")
    assert page.status_code == 200 and "Kite Login Successful" in page.text
    assert "OWN001, 09:05 IST, Fri 13 Nov" in page.text and "This tab closes in" in page.text
    assert login.opened == ["kite-for-req1"]


@pytest.mark.parametrize("query,status,title", [
    ("request_token=req1&status=cancelled", 400, "Kite login not completed"),
    ("status=success", 400, "Kite login not completed"),                      # no request token
    ("request_token=req1&status=success&account=nobody", 404, "Unknown account"),
])
def test_a_login_that_did_not_complete_or_names_no_account_stores_nothing(login, query, status, title):
    page = login.client.get(f"/ind-stocks/auth/callback?{query}")
    assert page.status_code == status and title in page.text and login.opened == []


def test_a_login_as_the_wrong_user_or_with_a_spent_token_is_refused_in_plain_words(login):
    login.exchange = PermissionError("logged in as ZZ9999, but this app accepts only OWN001")
    page = login.client.get("/ind-stocks/auth/callback?request_token=req1&status=success")
    assert page.status_code == 403 and "Login refused" in page.text and "accepts only OWN001" in page.text
    login.exchange = RuntimeError("Token is invalid or has <expired>")
    page = login.client.get("/ind-stocks/auth/callback?request_token=req1&status=success")
    assert page.status_code == 401 and "Request tokens work once" in page.text
    assert "&lt;expired&gt;" in page.text and "<expired>" not in page.text       # Kite's words, never as markup
    page = login.client.get("/ind-stocks/auth/callback?request_token=r&status=%3Cscript%3E")
    assert "<script>" not in page.text.replace(page.text[page.text.find("<script>(function"):], "")
    assert login.opened == []


def test_a_connected_account_sees_the_terms_before_centurion_may_trade_it(login):
    import re

    page = login.client.get("/ind-stocks/auth/callback?request_token=req1&status=success&account=ann_lee")
    assert page.status_code == 200 and "one more step" in page.text
    assert "Ann &lt;Lee&gt; (AB1234)" in page.text and login.opened == []         # not the API's own session
    ticket = re.search(r"name='ticket' value='([^']+)'", page.text).group(1)
    refused = login.client.post("/ind-stocks/auth/consent", data={"ticket": ticket})          # the box not ticked
    assert refused.status_code == 400 and "Terms not accepted" in refused.text and login.consents == []
    forged = login.client.post("/ind-stocks/auth/consent", data={"ticket": ticket + "x", "agree": "yes"})
    assert forged.status_code == 400 and login.consents == []
    agreed = login.client.post("/ind-stocks/auth/consent", data={"ticket": ticket, "agree": "yes"})
    assert agreed.status_code == 200 and login.consents == [("ann_lee", "AB1234", login.terms.TERMS_VERSION)]
    assert "09:06 IST, Fri 13 Nov" in agreed.text


def test_the_terms_page_fails_closed(login):
    import re

    from kite_connect.auth import accounts

    page = login.client.get("/ind-stocks/auth/callback?request_token=req1&status=success&account=ann_lee")
    ticket = re.search(r"name='ticket' value='([^']+)'", page.text).group(1)
    big = login.client.post("/ind-stocks/auth/consent", data={"ticket": ticket, "agree": "yes", "pad": "x" * 5000})
    assert big.status_code == 413
    login.record = accounts.AccountError("these terms have changed: log in again to see the current ones")
    changed = login.client.post("/ind-stocks/auth/consent", data={"ticket": ticket, "agree": "yes"})
    assert changed.status_code == 400 and "these terms have changed" in changed.text
    login.record = ConnectionError("neon down")
    down = login.client.post("/ind-stocks/auth/consent", data={"ticket": ticket, "agree": "yes"})
    assert down.status_code == 503 and "Terms not recorded" in down.text and "neon" not in down.text
    assert login.consents == []


# ── sign-up (api/users.py): what a stranger's form may contain, checked before anything is stored ──

@pytest.mark.parametrize("password,problem", [
    ("Tr4de!Wisely-2026", ""),
    ("Short1!", "Use 12 to 64 characters"), ("x" * 65 + "A1!", "Use 12 to 64 characters"),
    ("é" * 37 + "A1!a", "Use 12 to 64 characters"),                            # 41 characters, over bcrypt's 72 bytes
    ("alllowercase1!x", "Use upper and lower case letters"), ("NoDigitsHere!!x", "Include a digit"),
    ("NoSymbolHere123", "Include a symbol, such as ! or #"),
    ("Ravi-Kumar#2026x", "Leave your name and email out of the password"),
    ("rk.trader!A2026x", "Leave your name and email out of the password"),
    ("My-Passw0rd!2026", "Avoid common words such as 'password' or 'centurion'"),
])
def test_the_password_policy_names_what_is_wrong(password, problem):
    from api import users

    assert users.password_problem(password, "rk.trader@example.org", "Ravi Kumar") == problem


@pytest.mark.parametrize("email,ok", [
    ("ravi.k+trade@example.co.in", True), ("a@b.io", True),
    ("no-at-sign.example.org", False), ("two@@example.org", False), ("a@localhost", False), ("a@example.c0m", False),
    (".dot@example.org", False), ("a@-bad.example.org", False), ("a b@example.org", False),
    ("x" * 65 + "@example.org", False), ("a@" + "b" * 250 + ".org", False),
])
def test_an_email_must_look_deliverable(email, ok):
    from api import users

    assert (users.email_problem(email) == "") is ok


def test_names_and_places_take_letters_of_any_script_and_nothing_that_is_markup():
    from api import users

    for name in ("Anne-Marie", "D'Souza", "R. Kumar", "अनुष्का", "Müller"):
        assert users.text_problem(name, "First name", users.NAME_MAX) == "", name
    assert users.text_problem("1van", "First name", 40) == "First name: start with a letter"
    assert users.text_problem("", "First name", 40) == "First name: start with a letter"
    assert users.text_problem("Jo<script>", "First name", 40).startswith("First name: letters only")
    assert users.text_problem("x" * 41, "First name", 40) == "First name: up to 40 characters"
    assert users.text_problem("Bengaluru (Bangalore)", "City", 60, users._PLACE_JOINERS) == ""
    assert users.text_problem("Bengaluru (Bangalore)", "City", 60) != ""
    assert users._plain("  Ravi   Kumar ") == "Ravi Kumar"


@pytest.mark.parametrize("raw,country,result", [
    ("", "India", ""), ("098765 43210", "India", "+919876543210"), ("98765-43210", "india", "+919876543210"),
    ("+91 (98765) 43210", "Singapore", "+919876543210"), ("+44 20 7946 0958", "United Kingdom", "+442079460958"),
    ("9876543210", "Singapore", "start with + and your country code"),
    ("+91 12345 67890", "India", "10 digits starting with 6, 7, 8 or 9"),
    ("98765x3210", "India", "digits only"), ("+12345", "USA", "8 to 15 digits"),
])
def test_a_mobile_number_is_kept_in_one_form_or_refused(raw, country, result):
    from api import users

    if result.startswith("+") or result == "":
        assert users.clean_phone(raw, country) == result
    else:
        with pytest.raises(users.UserError) as refused:
            users.clean_phone(raw, country)
        assert result in str(refused.value)


def _signup(**over):
    return {"email": " Ravi.K@Example.org ", "first_name": " ravi ", "last_name": "kumar", "date_of_birth": "1990-05-17",
            "gender": "male", "city": "Navi  Mumbai", "state": "Maharashtra", "country": "India", "phone": "9876543210",
            "trading_experience": "1_to_5_years", "consent": True, "password": "Tr4de!Wisely-2026", **over}


def test_a_sign_up_is_normalised_once_and_kept_as_given_otherwise():
    from api import users

    clean = users.clean_signup(_signup())
    assert clean == {"email": "ravi.k@example.org", "first_name": "ravi", "last_name": "kumar", "full_name": "ravi kumar",
                     "date_of_birth": "1990-05-17", "gender": "male", "phone": "+919876543210", "city": "Navi Mumbai",
                     "state": "Maharashtra", "country": "India", "trading_experience": "1_to_5_years",
                     "newsletter": False}
    assert "password" not in clean and "consent" not in clean                  # neither is part of the profile
    older = users.clean_signup(_signup(first_name="", last_name="", full_name="  Ravi   Kumar "))
    assert (older["full_name"], older["first_name"]) == ("Ravi Kumar", "")     # a frontend from before first / last names
    assert users.clean_signup(_signup(last_name="", state="", phone=""))["full_name"] == "ravi"


@pytest.mark.parametrize("over,why", [
    ({"email": "not-an-email"}, "Enter a valid email address"), ({"first_name": ""}, "Enter your first name"),
    ({"first_name": "R2D2"}, "First name: letters only"), ({"date_of_birth": "17/05/1990"}, "use the date picker"),
    ({"date_of_birth": "2015-01-01"}, "at least 18"), ({"date_of_birth": "1850-01-01"}, "at least 18"),
    ({"gender": "robot"}, "Choose a gender option"), ({"city": ""}, "City and country are required"),
    ({"country": "I"}, "City and country are required"), ({"city": "Mumbai<b>"}, "City: letters only"),
    ({"trading_experience": "expert"}, "Choose a trading experience option"),
    ({"consent": "yes"}, "Agree to Centurion storing your details"), ({"consent": False}, "Agree to Centurion"),
    ({"password": "Ravi-Kumar#2026x"}, "Leave your name and email out"),
    ({"first_name": "", "last_name": "", "full_name": "R"}, "Full name: 2 to 80 characters"),
])
def test_a_sign_up_that_cannot_stand_is_refused_with_its_reason(over, why):
    from api import users

    with pytest.raises(users.UserError, match=why):
        users.clean_signup(_signup(**over))


def test_the_public_routes_rate_limit_counts_per_key_within_its_window(monkeypatch):
    from api import users

    now = [1_000.0]
    monkeypatch.setattr(users.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(users, "_hits", {})
    assert [users.allowed("ip:1.2.3.4", 3, 60) for _ in range(4)] == [True, True, True, False]
    assert users.allowed("ip:5.6.7.8", 3, 60)                                  # another client is not held back
    now[0] += 59
    assert not users.allowed("ip:1.2.3.4", 3, 60)
    now[0] += 2
    assert users.allowed("ip:1.2.3.4", 3, 60)                                  # the window has passed


def test_links_die_with_the_password_and_the_email_is_found_only_with_the_key(monkeypatch):
    pytest.importorskip("bcrypt")
    pytest.importorskip("itsdangerous")
    pytest.importorskip("cryptography")
    from cryptography.fernet import Fernet

    from api import users

    monkeypatch.setenv(users.ENV_FRONTEND, "https://app.example.org/")
    user = {"email": "ravi.k@example.org", "password_hash": users.hash_password("Tr4de!Wisely-2026")}
    url = users.link(users.ACTIVATE_PURPOSE, user)
    assert url.startswith("https://app.example.org/activate?token=")
    from urllib.parse import unquote

    token = unquote(url.split("token=", 1)[1])
    assert users.read_link(token, users.ACTIVATE_PURPOSE) == {"email": "ravi.k@example.org",
                                                              "fp": users.fingerprint(user["password_hash"])}
    for bad, purpose in ((token, users.RESET_PURPOSE), (token + "x", users.ACTIVATE_PURPOSE), ("", users.ACTIVATE_PURPOSE)):
        with pytest.raises(users.UserError, match="invalid or has expired"):
            users.read_link(bad, purpose)
    assert "/reset-password?token=" in users.link(users.RESET_PURPOSE, user)
    changed = users.hash_password("An0ther!Passphrase")
    assert users.fingerprint(changed) != users.fingerprint(user["password_hash"]) and len(users.fingerprint(changed)) == 16
    monkeypatch.setenv(users.ENV_DATA_KEY, Fernet.generate_key().decode())
    index = users.email_index(" Ravi.K@Example.org ")
    assert index == users.email_index("ravi.k@example.org") and "ravi" not in index and len(index) == 64
    monkeypatch.setenv(users.ENV_DATA_KEY, Fernet.generate_key().decode())
    assert users.email_index("ravi.k@example.org") != index                      # another key: another index
    monkeypatch.delenv(users.ENV_DATA_KEY)
    with pytest.raises(RuntimeError, match="CENTURION_USER_DATA_KEY is not set"):
        users.email_index("ravi.k@example.org")


# ── the NSE session badge on the Ind stocks pages (nse_engine.nse_calendar.session_status) ──

@pytest.mark.parametrize("ist,state,next_open,detail", [
    ("2026-11-17 10:00", "open", "2026-11-18", "closes 15:30 IST"),
    ("2026-11-17 09:05", "pre_open", "2026-11-18", "pre-open; opens 09:15 IST"),
    ("2026-11-17 08:00", "closed", "2026-11-18", "opens today, 09:15 IST"),
    ("2026-11-23 15:30", "closed", "2026-11-25", "closed for the day; opens Wed 25 Nov, 09:15 IST"),  # 24 Nov: holiday
    ("2026-11-24 11:00", "holiday", "2026-11-25", "trading holiday; opens Wed 25 Nov, 09:15 IST"),
    ("2026-11-21 11:00", "weekend", "2026-11-23", "opens Mon 23 Nov, 09:15 IST"),
    ("2026-11-08 18:00", "muhurat", "2026-11-09", "Muhurat trading this evening"),       # its hours are not held
    ("2026-02-01 10:00", "open", "2026-02-02", "closes 15:30 IST"),                     # Budget day, a Sunday
])
def test_the_nse_badge_reads_the_one_calendar_in_ist(ist, state, next_open, detail):
    from datetime import datetime

    from nse_engine import nse_calendar as cal

    s = cal.session_status(datetime.fromisoformat(ist).replace(tzinfo=cal.IST))
    assert (s["state"], s["next_open"], s["open"]) == (state, next_open, state == "open")
    assert s["detail"].startswith(detail) and s["note"] is None


def test_the_nse_badge_reads_a_utc_clock_in_ist_and_warns_without_next_years_holidays():
    from datetime import datetime, timezone

    from nse_engine import nse_calendar as cal

    assert cal.session_status(datetime(2026, 11, 17, 4, 0, tzinfo=timezone.utc))["state"] == "open"   # 09:30 IST
    assert cal.session_status(datetime(2026, 11, 17, 10, 0, tzinfo=timezone.utc))["state"] == "closed"  # 15:30 IST
    eve = cal.session_status(datetime(2026, 12, 31, 16, 0, tzinfo=cal.IST))
    assert eve["next_open"] == "2027-01-01" and "2027 holiday list is not loaded" in eve["note"]


def test_the_nse_session_route_answers_a_signed_up_user(api):
    client = api[1]
    body = client.get("/api/v1/market/nse-session", headers=_bearer("user")).json()
    assert body["state"] in ("open", "pre_open", "closed", "weekend", "holiday", "muhurat")
    assert set(body) == {"state", "open", "detail", "date", "time", "next_open", "note"}


# ── live gold, silver and bitcoin for the Ind stocks ribbon (api/live_prices.py, tracker GSB1) ──

def _mcx(name, expiry, symbol):
    from datetime import date

    return {"name": name, "instrument_type": "FUT", "expiry": date.fromisoformat(expiry), "tradingsymbol": symbol}


def test_the_ribbon_reads_the_near_month_contract_and_rolls_before_its_last_days():
    from datetime import date

    from api import live_prices as lp

    dump = [_mcx("GOLD", "2026-12-04", "GOLD26DECFUT"), _mcx("GOLD", "2027-02-05", "GOLD27FEBFUT"),
            _mcx("GOLDM", "2026-11-30", "GOLDM26NOVFUT"), {**_mcx("GOLD", "2026-11-25", "GOLDOPT"), "instrument_type": "CE"}]
    assert lp.near_month(dump, "GOLD", date(2026, 11, 16))["tradingsymbol"] == "GOLD26DECFUT"
    assert lp.near_month(dump, "GOLD", date(2026, 12, 1))["tradingsymbol"] == "GOLD27FEBFUT"     # 3 days left: rolled
    assert lp.near_month(dump, "SILVER", date(2026, 11, 16)) is None


def test_gold_is_shown_per_gram_and_silver_per_kg_with_their_change():
    from api import live_prices as lp

    quotes = {"MCX:GOLD26DECFUT": {"last_price": 135236.0, "ohlc": {"close": 136705.0}, "last_trade_time": "t1"},
              "MCX:SILVER26DECFUT": {"last_price": 168557.0, "ohlc": {"close": 176309.0}}}
    got = lp.metal_prices(quotes, {"gold": "MCX:GOLD26DECFUT", "silver": "MCX:SILVER26DECFUT"})
    assert got["gold"]["price"] == pytest.approx(13523.6) and got["gold"]["unit"] == "gram"
    assert got["gold"]["change_pct"] == pytest.approx((135236 / 136705 - 1) * 100) and got["gold"]["currency"] == "INR"
    assert got["silver"]["price"] == 168557.0 and got["silver"]["unit"] == "kg"
    none = lp.metal_prices({"MCX:GOLD26DECFUT": {"last_price": 0}}, {"gold": "MCX:GOLD26DECFUT", "silver": None})
    assert none == {"gold": None, "silver": None}                                # no price is shown as absent


def test_bitcoin_falls_back_from_binance_to_coingecko_and_is_absent_when_both_fail():
    from api import live_prices as lp

    class Reply:
        def __init__(self, body): self.body = body
        def json(self): return self.body

    def both(url, timeout):
        return Reply({"lastPrice": "64902.47", "priceChangePercent": "0.33"} if "binance" in url else {})

    def binance_down(url, timeout):
        if "binance" in url:
            raise ConnectionError("451")
        return Reply({"bitcoin": {"usd": 64709, "usd_24h_change": -1.2}})

    def all_down(url, timeout):
        raise ConnectionError("down")

    assert lp.bitcoin(both) == {"price": 64902.47, "change_pct": 0.33, "currency": "USD", "source": "Binance BTC/USDT"}
    assert lp.bitcoin(binance_down)["source"] == "CoinGecko" and lp.bitcoin(binance_down)["price"] == 64709.0
    assert lp.bitcoin(all_down) is None


def test_without_a_kite_login_the_metals_say_why_and_every_answer_is_cached(monkeypatch):
    from datetime import date

    from api import live_prices as lp

    monkeypatch.setattr(lp, "_cache", {"at": None, "value": None})
    monkeypatch.setattr(lp, "_contracts", {"day": None, "symbols": {}})
    calls, now = [], [100.0]
    get = lambda url, timeout: calls.append(url) or type("R", (), {"json": lambda self: {          # noqa: E731
        "lastPrice": "1", "priceChangePercent": "0"}})()
    out = lp.snapshot(kite_factory=lambda: None, clock=lambda: now[0], get=get)
    assert out["gold"] is None and out["silver"] is None and "Kite login" in out["note"] and out["bitcoin"]
    now[0] += 10
    assert lp.snapshot(kite_factory=lambda: None, clock=lambda: now[0], get=get) is out and len(calls) == 1

    class Kite:
        def instruments(self, exchange):
            return [_mcx("GOLD", "2026-12-04", "GOLD26DECFUT"), _mcx("SILVER", "2026-12-04", "SILVER26DECFUT")]

        def quote(self, symbols):
            return {s: {"last_price": 100.0, "ohlc": {"close": 100.0}} for s in symbols}

    now[0] += lp.CACHE_S
    fresh = lp.snapshot(kite_factory=Kite, today=date(2026, 11, 16), clock=lambda: now[0], get=get)
    assert fresh["gold"]["price"] == 10.0 and fresh["silver"]["price"] == 100.0 and fresh["note"] is None


def test_the_live_prices_route_answers_a_signed_up_user(api, monkeypatch):
    from api import live_prices as lp

    monkeypatch.setattr(lp, "snapshot", lambda: {"gold": None, "silver": None, "bitcoin": None, "note": "n"})
    body = api[1].get("/api/v1/market/live-prices", headers=_bearer("user")).json()
    assert set(body) == {"gold", "silver", "bitcoin", "note"}


# ── the RAG engine answers with the Claude model the user picks (rag_pipeline/llm/llm_service.py, tracker RAG2) ──

class _ClaudeStream:
    def __init__(self, texts, final):
        self.text_stream, self._final = iter(texts), final

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._final


class _ClaudeClient:
    """Stands in for anthropic.Anthropic(): records each request, answers or raises."""

    def __init__(self, texts=("An ", "answer."), stop="end_turn", error=None, model=None):
        from types import SimpleNamespace

        self.calls, self.error = [], error
        self._texts, self._stop, self._model = texts, stop, model
        self.beta = SimpleNamespace(messages=self)

    def _final(self, request):
        from types import SimpleNamespace

        return SimpleNamespace(model=self._model or request["model"], stop_reason=self._stop,
                               stop_details=SimpleNamespace(category="cyber") if self._stop == "refusal" else None,
                               content=[SimpleNamespace(type="text", text="".join(self._texts))],
                               usage=SimpleNamespace(input_tokens=10, output_tokens=5))

    def stream(self, **request):
        self.calls.append(request)
        if self.error:
            raise self.error
        return _ClaudeStream(self._texts, self._final(request))

    def create(self, **request):
        self.calls.append(request)
        if self.error:
            raise self.error
        return self._final(request)


def _claude(client):
    from rag_pipeline.llm.llm_service import ClaudeLLMBackend

    backend = ClaudeLLMBackend(api_key="sk-ant-api-test")
    backend._client = client
    return backend


def test_a_claude_request_carries_no_temperature_and_leaves_room_to_think():
    from rag_pipeline.llm import llm_service as ls

    client = _ClaudeClient()
    meta = {}
    assert "".join(_claude(client).generate_stream("What is RA1?", "context", meta=meta)) == "An answer."
    first = client.calls[0]
    assert first["model"] == "claude-opus-5-5" and first["max_tokens"] == 64000 and "temperature" not in first
    assert first["output_config"] == {"effort": "medium"} and "thinking" not in first
    assert first["fallbacks"] == "default" and first["betas"] == [ls.FALLBACK_BETA]
    assert meta == {"source": "llm", "provider": "Claude", "model": "claude-opus-5-5", "requested": "claude-opus-5-5",
                    "stop_reason": "end_turn"}
    list(_claude(client).generate_stream("q", "c", model="claude-haiku-5-5"))
    assert "fallbacks" not in client.calls[1] and "betas" not in client.calls[1]       # Haiku has no API fallback
    _claude(client).generate("q", "c", model="claude-fable-5-2")
    assert client.calls[2]["output_config"] == {"effort": "high"} and client.calls[2]["max_tokens"] == 16000
    with pytest.raises(ValueError, match="not offered"):
        list(_claude(client).generate_stream("q", "c", model="claude-sonnet-4-20250514"))
    assert [m["id"] for m in ls.CLAUDE_MODELS][0] == ls.DEFAULT_CLAUDE_MODEL == "claude-opus-5-5"


def test_the_claude_request_matches_the_sdk_the_space_runs():
    """The Space pins anthropic 2.x (constraints-hf.txt), which has no temperature: the old request raised a
    TypeError there before reaching Anthropic, so Claude never answered."""
    anthropic = pytest.importorskip("anthropic")
    import inspect

    from anthropic.resources.beta.messages import Messages

    client = _ClaudeClient()
    list(_claude(client).generate_stream("q", "c"))
    accepted = inspect.signature(Messages.stream).parameters
    assert set(client.calls[0]) <= set(accepted), set(client.calls[0]) - set(accepted)
    assert "temperature" not in accepted or anthropic.__version__.startswith("0.")


def test_a_rejected_key_says_pro_is_not_api_access_and_the_free_model_answers():
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    from rag_pipeline.llm.llm_service import _FallbackChainBackend

    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    rejected = anthropic.AuthenticationError("invalid x-api-key", response=httpx2.Response(401, request=request),
                                             body=None)

    class Free:
        def generate_stream(self, query, context):
            yield "a free answer"

    chain = _FallbackChainBackend(_claude(_ClaudeClient(error=rejected)), Free())
    meta = {}
    assert "".join(chain.generate_stream("q", "c", model="claude-sonnet-5-5", meta=meta)) == "a free answer"
    assert meta["fallback"] is True and meta["provider"] == "Free"
    assert "Claude API key rejected" in meta["reason"] and "Claude Pro" in meta["reason"]


def test_a_refusal_is_said_in_plain_words_and_names_the_model():
    meta = {}
    out = "".join(_claude(_ClaudeClient(texts=(), stop="refusal")).generate_stream("q", "c", meta=meta))
    assert "Claude declined to answer this question (cyber)" in out and not out.lstrip().startswith("\u26a0")
    assert meta["stop_reason"] == "refusal"                     # not an error: the free model is not asked instead


def test_the_rag_page_lists_the_models_and_explains_a_rejected_key(api, monkeypatch):
    from rag_pipeline.llm import llm_service as ls

    monkeypatch.setattr(ls, "claude_status", lambda key: {"available": False, "note": "rejected: not an API key"})
    body = api[1].get("/api/v1/rag/models", headers=_bearer("user")).json()
    assert [m["id"] for m in body["models"]] == [m["id"] for m in ls.CLAUDE_MODELS]
    assert body["default"] == "claude-opus-5-5" and body["available"] is False and "not an API key" in body["note"]
    assert body["models"][0] == {"id": "claude-opus-5-5", "label": "Claude Opus 5.5", "input_usd": 4.0,
                                 "output_usd": 20.0}


def test_the_query_route_takes_the_chosen_model_and_ends_with_what_answered(api, monkeypatch):
    import api.routers.v1.rag as rag_routes

    seen = {}

    class Engine:
        def query_stream(self, q, source_filter=None, llm_model=None, llm_meta=None, on_context=None):
            seen["model"] = llm_model
            llm_meta.update(source="llm", provider="Claude", model=llm_model)
            yield "An answer."

    monkeypatch.setattr(rag_routes, "get_rag_engine", lambda: Engine())
    client = api[1]
    sse = client.get("/api/v1/rag/query", params={"q": "What is RA1?", "model": "claude-haiku-5-5"},
                     headers=_bearer("user")).text
    assert seen["model"] == "claude-haiku-5-5" and 'event: meta\ndata: {"source": "llm", "provider": "Claude", ' \
        '"model": "claude-haiku-5-5"}' in sse and sse.rstrip().endswith("data: done")
    bad = client.get("/api/v1/rag/query", params={"q": "x", "model": "gpt-4o"}, headers=_bearer("user"))
    assert bad.status_code == 400 and "not offered" in bad.json()["detail"]


def test_the_query_route_searches_every_ticked_source_and_shows_the_passages_first(api, monkeypatch):
    """RAG3: two ticked sources both reach the engine (the route used to drop the filter unless exactly one was
    ticked), and the passages the answer is built from arrive as chunk events before its first token."""
    import json

    import api.routers.v1.rag as rag_routes

    seen = {}

    class Engine:
        def query_stream(self, q, source_filter=None, llm_model=None, llm_meta=None, on_context=None):
            seen["sources"] = source_filter
            on_context([{"source": "b.pdf", "page": 4, "chunk_index": 7, "score": 0.81, "content": "RA1 is..."},
                        {"source": "a.pdf", "page": None, "chunk_index": 2, "score": 0.64, "content": "Gate 2..."}])
            yield "An answer."

    monkeypatch.setattr(rag_routes, "get_rag_engine", lambda: Engine())
    sse = api[1].get("/api/v1/rag/query", params={"q": "What is RA1?", "sources": "b.pdf,a.pdf"},
                     headers=_bearer("user")).text
    assert seen["sources"] == ["b.pdf", "a.pdf"]
    events = [(b.split("\n")[0][len("event: "):], b.split("\n", 1)[1][len("data: "):])
              for b in sse.strip().split("\n\n")]
    assert [e for e, _ in events] == ["chunk", "chunk", "token", "meta", "done"]
    assert json.loads(events[0][1]) == {"source": "b.pdf", "page": 4, "chunk_index": 7, "score": 0.81,
                                        "content": "RA1 is..."}
    api[1].get("/api/v1/rag/query", params={"q": "x"}, headers=_bearer("user"))
    assert seen["sources"] is None                                            # nothing ticked: every source


def test_the_engine_filters_one_or_several_sources_and_names_each_passage():
    """RAG3: one source is an equality filter, several an $in (order and repeats do not matter, so the cache
    key and the filter agree); each passage reports its source, page, relevance and an excerpt."""
    from types import SimpleNamespace

    from rag_pipeline.core.query_engine import RAGQueryEngine, RetrievedChunk, _source_list, context_sources

    plain = SimpleNamespace(_config=SimpleNamespace(enforce_space_isolation=False, enable_metadata_filtering=False))
    where = RAGQueryEngine._build_where_filter
    assert where(plain, None, None) is None and where(plain, None, []) is None
    assert where(plain, None, "a.pdf") == where(plain, None, ["a.pdf"]) == {"source": "a.pdf"}
    assert where(plain, None, ["b.pdf", "a.pdf", "b.pdf"]) == {"source": {"$in": ["a.pdf", "b.pdf"]}}
    assert where(plain, {"type": "pdf"}, ["a.pdf", "b.pdf"]) == {
        "$and": [{"type": "pdf"}, {"source": {"$in": ["a.pdf", "b.pdf"]}}]}
    assert _source_list(["b.pdf", "a.pdf"]) == _source_list(["a.pdf", "b.pdf", ""]) == ["a.pdf", "b.pdf"]

    chunks = [RetrievedChunk("x" * 900, "a.pdf", 3, 0.25, {"page_number": 12}),
              RetrievedChunk("short", "b.pdf", 0, 1.4, {"page_number": "?"})]
    a, b = context_sources(chunks)
    assert a == {"source": "a.pdf", "page": 12, "chunk_index": 3, "score": 0.75, "content": "x" * 400}
    assert b["page"] is None and b["score"] == 0.0 and b["content"] == "short"


def test_the_broad_fallback_stays_inside_the_ticked_sources():
    """RAG3: when the ticked sources gave too few passages, the engine searched again with no filter at all, so
    the answer drew on sources nobody ticked (and on other spaces).  The fallback now keeps the scope and drops
    only the short-chunk filter; the passages reach the page before the model starts."""
    from rag_pipeline.config import RAGConfig
    from rag_pipeline.core.query_engine import RAGQueryEngine

    rows = [("a.pdf", 3, 0.20, "Golden gates require coverage from 2008 for every stream in the store."),
            ("b.pdf", 9, 0.30, "The RA1 review in December decides the deployed book's numbers."),
            ("c.pdf", 1, 0.10, "Bond sleeve duration and cost notes for the paper book.")]
    wheres, order = [], []

    def scope(where):                                     # the source clause, alone or inside an $and
        for part in (where or {}).get("$and", [where] if where else []):
            if "source" in part:
                v = part["source"]
                return set(v["$in"]) if isinstance(v, dict) else {v}
        return None

    class Store:                                          # answers like Chroma, honouring the source clause
        def query(self, query_embeddings, n_results, where=None, include=None):
            wheres.append(where)
            hit = [r for r in rows if scope(where) is None or r[0] in scope(where)][:n_results]
            return {"ids": [[f"{r[0]}-{i}" for i, r in enumerate(hit)]], "documents": [[r[3] for r in hit]],
                    "metadatas": [[{"source": r[0], "page_number": r[1], "chunk_index": i}
                                   for i, r in enumerate(hit)]], "distances": [[r[2] for r in hit]]}

    class Embedder:
        def embed_query(self, text):
            return [1.0, 0.0]

    class LLM:
        def generate_stream(self, query, context):
            order.append("llm")
            yield "ok"

    cfg = RAGConfig()
    for name in ("reranker_enabled", "query_rewrite_enabled", "hybrid_search_enabled", "cache_enabled",
                 "faq_enabled", "enable_metadata_filtering", "enforce_space_isolation"):
        setattr(cfg, name, False)
    engine = RAGQueryEngine(vector_store=Store(), config=cfg, embedding_service=Embedder(), llm_backend=LLM())
    seen = []
    out = list(engine.query_stream("Explain what the December review decides and how the gates treat coverage",
                                   source_filter=["b.pdf", "a.pdf"], skip_cache=True,
                                   on_context=lambda items: (order.append("context"), seen.extend(items))))
    assert len(wheres) == 2 and all(scope(w) == {"a.pdf", "b.pdf"} for w in wheres)   # 2 < 3: the fallback ran
    assert {c["source"] for c in seen} == {"a.pdf", "b.pdf"} and order == ["context", "llm"] and out == ["ok"]


# ── the RAG engine's two entry points, query and query_stream, pinned exit by exit (docs/AUDIT.md R1) ──

_NOTHING_FOUND = ("I could not find relevant information in the uploaded documents to answer this question.\n\n"
                  "**Possible reasons:**\n- The relevant document may not have been uploaded yet\n"
                  "- The query may need to be rephrased to match the document content\n"
                  "- The uploaded documents may not cover this topic\n\n"
                  "Please upload the relevant PDF or try rephrasing your query.")
_NOTHING_FOUND_STREAM = ("I could not find relevant information in the uploaded documents to answer this question.\n\n"
                         "Please upload the relevant PDF or try rephrasing your query.")


def _rag_rows(*specs):
    """(source, distance, text, extra metadata) per row; the row's position is its chunk index."""
    return [{"text": text, "distance": dist, "meta": {"source": src, "chunk_index": i, "page_number": i + 1, **extra}}
            for i, (src, dist, text, extra) in enumerate(specs)]


def _rag_match(where, meta):
    """A Chroma where clause on one row: equality, $in or $gte, alone or inside an $and."""
    for part in (where or {}).get("$and", [where] if where else []):
        for key, cond in part.items():
            value = meta.get(key)
            if isinstance(cond, dict):
                if "$in" in cond and value not in cond["$in"]:
                    return False
                if "$gte" in cond and (value is None or value < cond["$gte"]):
                    return False
            elif value != cond:
                return False
    return True


class _RagStore:
    """Answers like Chroma, nearest first, honouring the where clause; records (n_results, where) per search."""

    def __init__(self, rows, slow_first=0.0):
        self.rows, self.calls, self._slow = rows, [], slow_first

    def query(self, query_embeddings, n_results, where=None, include=None):
        import time

        self.calls.append((n_results, where))
        if self._slow:
            delay, self._slow = self._slow, 0.0
            time.sleep(delay)
        hit = sorted((r for r in self.rows if _rag_match(where, r["meta"])), key=lambda r: r["distance"])[:n_results]
        return {"ids": [[f"id{r['meta']['chunk_index']}" for r in hit]], "documents": [[r["text"] for r in hit]],
                "metadatas": [[dict(r["meta"]) for r in hit]], "distances": [[r["distance"] for r in hit]]}


class _RagEmbedder:
    def embed_query(self, text):
        if "boom" in text:
            raise RuntimeError("embedding failed")
        return [1.0, 0.0]


class _RagLLM:
    def __init__(self, tokens=("Gate ", "two.")):
        self.tokens, self.calls = tokens, []

    def generate(self, query, context):
        self.calls.append(("generate", query, context))
        return "".join(self.tokens)

    def generate_stream(self, query, context):
        self.calls.append(("stream", query, context))
        yield from self.tokens


class _RagCache:
    def __init__(self, hit=None):
        self.hit, self.lookups, self.stored = hit, [], []

    def lookup(self, query, space_id):
        from types import SimpleNamespace

        self.lookups.append(space_id)
        return SimpleNamespace(answer=self.hit) if self.hit else None

    def store(self, query, answer, chunks, space_id):
        self.stored.append((query, answer, chunks, space_id))


class _RagBudget:
    """The time budget: ``expired`` answers is_expired() call by call; ``llm_over`` answers llm_exceeded()."""

    elapsed = 1.0

    def __init__(self, expired=(False,), llm_over=False):
        self._expired, self._llm_over = list(expired), llm_over

    def start(self):
        return self

    def retrieval_exceeded(self, seconds):
        return False

    def is_expired(self):
        return self._expired.pop(0) if len(self._expired) > 1 else self._expired[0]

    def llm_exceeded(self, seconds):
        return self._llm_over

    def cutoff_message(self):
        return " [cut]"


class _RagLog:
    def __init__(self):
        self.calls = []

    def log(self, **record):
        self.calls.append(record)


def _rag(rows=(), *, store=None, llm=None, cache=None, faq=None, budget=None, reranker=None, rewriter=None,
         **settings):
    """The engine on stand-ins, every setting fixed here (RAGConfig reads the environment)."""
    from rag_pipeline.config import RAGConfig
    from rag_pipeline.core.query_engine import RAGQueryEngine

    cfg = RAGConfig()
    fixed = dict(reranker_enabled=False, query_rewrite_enabled=False, hybrid_search_enabled=False,
                 cache_enabled=False, faq_enabled=False, enable_metadata_filtering=False,
                 enforce_space_isolation=False, disable_reranking=False, debug_retrieval=False, top_k=6,
                 minimum_return_top_k=2, similarity_threshold=0.5, fallback_min_chunks=2, low_score_distance=0.9,
                 fallback_top_k=8, max_context_chunks=3, context_token_budget=4000, default_space_id="sp",
                 rerank_top_n=2, min_chunk_word_count=3)
    for name, value in {**fixed, **settings}.items():
        setattr(cfg, name, value)
    engine = RAGQueryEngine(vector_store=store or _RagStore(list(rows)), config=cfg, embedding_service=_RagEmbedder(),
                            llm_backend=llm or _RagLLM(), reranker=reranker, query_rewriter=rewriter)
    engine._cache, engine._tiered, engine._retrieval_logger = cache, faq, _RagLog()
    engine._time_budget_factory = lambda: budget or _RagBudget()
    return engine


def _rag_passages(engine, run, q, **kwargs):
    """The texts the answer is built from: query's chunks, or what query_stream hands on_context."""
    if run == "query":
        return [c.text for c in engine.query(q, **kwargs).chunks]
    texts = []
    list(engine.query_stream(q, on_context=lambda items: texts.extend(i["content"] for i in items), **kwargs))
    return texts


_FIVE = _rag_rows(("a.pdf", 0.10, "Alpha passage one", {"chunk_hash": "h1"}),
                  ("b.pdf", 0.20, "Beta passage two", {"chunk_hash": "h1"}),        # same hash: dedup drops it
                  ("a.pdf", 0.30, "Gamma passage three", {}),
                  ("b.pdf", 0.40, "Delta passage four", {}),
                  ("c.pdf", 0.45, "Epsilon passage five", {}),                     # past max_context_chunks
                  ("a.pdf", 0.60, "Zeta passage six", {}))                         # past the threshold


def test_rag_answers_from_the_cache_faq_or_fast_path_without_searching(monkeypatch):
    from types import SimpleNamespace

    import rag_pipeline.core.query_engine as qe

    q = "What does the gate decide?"
    cache, store = _RagCache(hit="cached answer"), _RagStore(_FIVE)
    engine = _rag(store=store, cache=cache, llm=(llm := _RagLLM()))
    r = engine.query(q, source_filter=["b.pdf", "a.pdf"])
    assert (r.answer, r.cached, r.faq_hit, r.chunks) == ("cached answer", True, False, [])
    meta = {"stale": 1}
    assert list(engine.query_stream(q, source_filter=["b.pdf", "a.pdf"], llm_model="claude-haiku-5-5",
                                    llm_meta=meta)) == ["cached answer"]
    assert meta == {"source": "cache"}
    assert cache.lookups == ["sp::src=a.pdf,b.pdf", "sp::src=a.pdf,b.pdf::llm=claude-haiku-5-5"]

    cache.hit = None
    faq = SimpleNamespace(check_faq=lambda text: SimpleNamespace(answer="faq answer"))
    engine = _rag(store=store, cache=cache, faq=faq, llm=llm)
    r = engine.query(q, skip_cache=True)
    assert (r.answer, r.cached, r.faq_hit, r.chunks) == ("faq answer", False, True, [])
    assert list(engine.query_stream(q, llm_meta=meta)) == ["faq answer"] and meta == {"source": "faq"}
    assert len(cache.lookups) == 3                                                   # skip_cache skips the lookup

    monkeypatch.setattr(qe, "try_fastpath", lambda text: "fast answer")
    engine = _rag(store=store, cache=cache, llm=llm)
    r = engine.query(q)
    assert (r.answer, r.cached, r.faq_hit, r.chunks) == ("fast answer", False, False, [])
    assert list(engine.query_stream(q, llm_model="m", llm_meta=meta)) == ["fast answer"]
    assert meta == {"source": "fastpath"}
    assert cache.stored == [(q, "fast answer", [], "sp"), (q, "fast answer", [], "sp::llm=m")]
    assert store.calls == [] and llm.calls == []


def test_rag_query_and_stream_send_the_same_context_and_record_the_answer():
    from rag_pipeline.core.query_engine import RAGQueryEngine, context_sources

    q = "What does the gate decide?"
    cache, llm, order = _RagCache(), _RagLLM(), []
    engine = _rag(_FIVE, cache=cache, llm=llm)
    r = engine.query(q)
    assert [c.text for c in r.chunks] == ["Alpha passage one", "Gamma passage three", "Delta passage four"]
    context = RAGQueryEngine._build_context(r.chunks, 4000)
    assert r.answer == "Gate two." and (r.cached, r.faq_hit) == (False, False)
    assert llm.calls == [("generate", q, context)] and engine._vs.calls == [(6, None)]
    summary = [{"source": "a.pdf", "chunk_index": 0}, {"source": "a.pdf", "chunk_index": 2},
               {"source": "b.pdf", "chunk_index": 3}]
    assert cache.stored == [(q, "Gate two.", summary, "sp")]
    (logged,) = engine._retrieval_logger.calls
    assert logged["retrieved_sources"] == ["a.pdf", "a.pdf", "b.pdf"] and logged["distances"] == [0.1, 0.3, 0.4]
    assert logged["retrieved_ids"] == ["h1", "a.pdf_2", "b.pdf_3"] and logged["extra"]["space_id"] == "sp"

    meta, seen = {}, []
    llm.calls.clear()
    out = list(engine.query_stream(q, llm_meta=meta, on_context=lambda items: (order.append("context"),
                                                                                seen.extend(items))))
    assert out == ["Gate ", "two."] and llm.calls == [("stream", q, context)] and order == ["context"]
    assert seen == context_sources(r.chunks) and meta == {"source": "llm", "provider": "_RagLLM"}
    assert cache.stored[-1] == (q, "Gate two.", summary, "sp") and len(engine._retrieval_logger.calls) == 1

    class Chosen(_RagLLM):                                # a backend that takes the user's model (RAG2)
        accepts_model = True

        def generate_stream(self, query, context, model=None, meta=None):
            meta.update(source="llm", model=model)
            yield from self.tokens

    meta = {}
    assert list(_rag(_FIVE, llm=Chosen()).query_stream(q, llm_model="claude-opus-5-5", llm_meta=meta)) == \
        ["Gate ", "two."] and meta == {"source": "llm", "model": "claude-opus-5-5"}

    class Blocking:                                       # no generate_stream: one piece
        def generate(self, query, context):
            return "whole answer"

    assert list(_rag(_FIVE, llm=Blocking()).query_stream(q)) == ["whole answer"]


def test_rag_says_it_found_nothing_rather_than_asking_the_model():
    q = "What does the gate decide?"
    cache, llm, seen = _RagCache(), _RagLLM(), []
    engine = _rag((), cache=cache, llm=llm)
    r = engine.query(q)
    assert (r.answer, r.chunks) == (_NOTHING_FOUND, [])
    assert list(engine.query_stream(q, on_context=seen.append)) == [_NOTHING_FOUND_STREAM]
    assert engine._vs.calls == [(6, None), (8, None)] * 2                            # the search, then the broad one
    assert llm.calls == [] and cache.stored == [] and seen == []


def test_rag_widens_a_thin_search_inside_the_ticked_sources():
    rows = _rag_rows(("a.pdf", 0.30, "Long enough passage on gates", {"word_count": 5}),
                     ("b.pdf", 0.10, "Short", {"word_count": 1}),
                     ("c.pdf", 0.05, "Unticked", {"word_count": 9}),
                     ("a.pdf", 0.20, "Tiny", {"word_count": 1}))
    q = "What does the gate decide?"
    for run in ("query", "stream"):
        engine = _rag(rows, enable_metadata_filtering=True)
        texts = _rag_passages(engine, run, q, source_filter=["b.pdf", "a.pdf"])
        scope = {"source": {"$in": ["a.pdf", "b.pdf"]}}
        assert engine._vs.calls == [(6, {"$and": [scope, {"word_count": {"$gte": 3}}]}), (8, scope)], run
        assert texts == ["Short", "Tiny", "Long enough passage on gates"], run


def test_rag_boosts_and_reranks_long_questions_and_skips_short_ones(monkeypatch):
    monkeypatch.delenv("RAG_FAST_MODE", raising=False)
    monkeypatch.setenv("RAG_SIMPLE_QUERY_THRESHOLD", "5")
    import re

    rows = _rag_rows(("a.pdf", 0.20, "First", {}), ("b.pdf", 0.25, "Second", {}),
                     ("c.pdf", 0.32, "Section passage", {"section_id": "4.2"}))       # boosted 0.10, to 0.22

    class Reranker:
        def __init__(self):
            self.calls = []

        def rerank(self, query, chunks, top_n):
            self.calls.append(([c.text for c in chunks], top_n))
            return list(reversed(chunks))[:top_n]

    long_q, short_q = "What does section 4.2 say about the gates", "Gates?"
    for run in ("query", "stream"):
        reranker, llm = Reranker(), _RagLLM()
        engine = _rag(rows, reranker=reranker, llm=llm)
        _rag_passages(engine, run, long_q)
        assert reranker.calls == [(["First", "Section passage", "Second"], 2)], run    # boosted ahead of Second
        assert re.findall(r"--- Content Start ---\n(.*?)\n--- Content End ---", llm.calls[0][2]) == \
            ["Second", "Section passage"], run                                         # the reranker's order
        _rag_passages(engine, run, short_q)                                            # simple: no boost, no rerank
        assert len(reranker.calls) == 1, run
        _rag(rows, reranker=reranker, disable_reranking=True).query(long_q)
        assert len(reranker.calls) == 1, run


def test_rag_answers_from_the_context_alone_when_time_runs_out():
    from rag_pipeline.core.query_engine import RAGQueryEngine

    q = "What does the gate decide?"
    cache, llm, seen = _RagCache(), _RagLLM(), []
    engine = _rag(_FIVE, cache=cache, llm=llm, budget=_RagBudget(expired=(True,)))
    r = engine.query(q)
    context = RAGQueryEngine._build_context(r.chunks, 4000)
    assert r.answer == ("_The system could not complete LLM generation within the time budget._\n\n"
                        "**Retrieved context (unprocessed):**\n\n" + context[:2000] + " [cut]")
    assert [c.text for c in r.chunks] == ["Alpha passage one", "Gamma passage three", "Delta passage four"]
    assert list(engine.query_stream(q, on_context=seen.append)) == [
        "_The system could not start LLM generation within the time budget._\n\n [cut]"]
    assert len(seen) == 1 and llm.calls == [] and cache.stored == []

    engine = _rag(_FIVE, cache=cache, llm=llm, budget=_RagBudget(expired=(False, True)))
    assert engine.query(q).answer == "Gate two. [cut]" and cache.stored[-1][1] == "Gate two. [cut]"
    engine = _rag(_FIVE, cache=cache, llm=llm, budget=_RagBudget(llm_over=True))
    assert list(engine.query_stream(q)) == ["Gate ", " [cut]"] and cache.stored[-1][1] == "Gate "


def test_rag_a_retrieval_timeout_falls_back_to_a_small_vector_search(monkeypatch):
    import rag_pipeline.core.query_engine as qe

    monkeypatch.setattr(qe, "_RETRIEVAL_HARD_TIMEOUT", 0.05)
    monkeypatch.delenv("RAG_FAST_MODE", raising=False)
    monkeypatch.setenv("RAG_SIMPLE_QUERY_THRESHOLD", "5")

    class Reranker:
        calls = 0

        def rerank(self, query, chunks, top_n):
            Reranker.calls += 1
            return chunks

    long_q = "What does section 4.2 say about the gates"
    for run in ("query", "stream"):
        store, llm = _RagStore(_FIVE, slow_first=0.3), _RagLLM()
        texts = _rag_passages(_rag(store=store, reranker=Reranker(), llm=llm), run, long_q)
        assert [n for n, _ in store.calls] == [6, 3], run                               # the slow search, then top 3
        assert texts == ["Alpha passage one", "Gamma passage three"], run              # 3 found, hash dedup drops one
        assert Reranker.calls == 0, run                                                 # forced fast: no rerank


def test_rag_a_failed_query_variant_does_not_stop_the_others():
    class Rewriter:
        def __init__(self):
            self.seen = []

        def rewrite(self, text):
            self.seen.append(text)
            return [text + " boom", text]

    q = "What does the gate decide?"
    for run in ("query", "stream"):
        rewriter, llm = Rewriter(), _RagLLM()
        engine = _rag(_FIVE, rewriter=rewriter, llm=llm)
        answer = engine.query(q).answer if run == "query" else "".join(engine.query_stream(q))
        assert answer == "Gate two." and rewriter.seen == [q] and len(engine._vs.calls) == 1, run


# ── the weekly newsletter (api/newsletter.py, tracker NL1): chosen, aggregates only, once a week, easy to stop ──

def test_the_newsletter_is_off_unless_the_box_is_ticked():
    from api import users

    assert users.clean_signup(_signup())["newsletter"] is False                 # an older frontend sends no field
    assert users.clean_signup(_signup(newsletter=True))["newsletter"] is True
    for unticked in (False, "yes", "true", 1, None):
        assert users.clean_signup(_signup(newsletter=unticked))["newsletter"] is False, unticked


def _equity(values, start="2026-09-17"):
    import pandas as pd

    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype="float64")


def test_the_newsletter_is_built_from_aggregates_alone_with_its_disclaimer():
    """Its fields are pinned: a stock, a position, a trade or the book's rupees cannot reach a reader without this
    test changing."""
    from datetime import date

    from api import newsletter

    e = _equity([100.0, 102.0, 99.0, 101.0, 103.0, 104.0, 104.5, 103.0])        # 17 Sep to 28 Sep 2026
    f = newsletter.figures(e, 100.0, _equity([200.0, 201, 202, 203, 204, 205, 206, 207]))
    assert tuple(f) == newsletter.PUBLIC_FIELDS
    assert all(v is None or isinstance(v, (int, float, date)) for v in f.values())
    assert (f["asof"], f["started"], f["sessions"]) == (date(2026, 9, 28), date(2026, 9, 17), 8)
    assert f["week"] == pytest.approx(103 / 99 - 1) and f["nifty_week"] == pytest.approx(207 / 202 - 1)
    assert f["since_start"] == pytest.approx(0.03) and f["nifty_since_start"] == pytest.approx(207 / 200 - 1)
    assert f["below_peak"] == pytest.approx(1 - 103 / 104.5) and f["deepest_fall"] == pytest.approx(1 - 99 / 102)
    mail = newsletter.render(f, "Ravi", "https://app.example.org/newsletter/unsubscribe?token=abc")
    assert newsletter.DISCLAIMER in mail["html"] and "SEBI" in newsletter.DISCLAIMER
    assert "newsletter/unsubscribe?token=abc" in mail["html"] and "₹" not in mail["html"]
    assert "rose 4.04%" in newsletter.summary(f) and "up 3.00%" in newsletter.summary(f)
    blind = newsletter.figures(e, 100.0, None)                                    # Yahoo unreachable
    assert blind["nifty_week"] is None and "NIFTY 50" not in newsletter.summary(blind)
    assert newsletter.figures(e[:1], 100.0) is None and newsletter.figures(e, 0.0) is None


def test_the_newsletter_reaches_each_active_reader_alone_and_once_a_week(monkeypatch):
    pytest.importorskip("cryptography")
    from datetime import datetime, timedelta

    from cryptography.fernet import Fernet

    from api import newsletter, users

    monkeypatch.setenv(users.ENV_DATA_KEY, Fernet.generate_key().decode())
    people = [{"id": 1, "email": "ravi@example.org", "first_name": "Ravi", "status": "active", "newsletter": True},
              {"id": 2, "email": "sita@example.org", "first_name": "Sita", "status": "active", "newsletter": True},
              {"id": 3, "email": "amir@example.org", "first_name": "Amir", "status": "pending", "newsletter": True},
              {"id": 4, "email": "lena@example.org", "first_name": "Lena", "status": "active", "newsletter": False}]
    marked, mails, loads, down = set(), [], [], {"sita@example.org"}
    monkeypatch.setattr(users, "list_users", lambda: people)
    monkeypatch.setattr(newsletter, "_sent", lambda week: {h for w, h in marked if w == week})
    monkeypatch.setattr(newsletter, "_mark", lambda week, index: marked.add((week, index)))
    monkeypatch.setattr(newsletter, "unsubscribe_link", lambda email: "https://app.example.org/newsletter/unsubscribe")
    f = newsletter.figures(_equity([100.0, 101, 102, 103, 104, 105, 106, 107]), 100.0)    # last session Mon 28 Sep
    monkeypatch.setattr(newsletter, "load", lambda: loads.append(1) or f)

    def send(subject, html, to):                          # one address per email: no reader sees another
        mails.append(to)
        return to not in down

    saturday = datetime(2026, 10, 3, 10, 0, tzinfo=newsletter.IST)
    assert newsletter.send_week(saturday, send) == {"week": "2026-W40", "readers": 2, "already": 0, "sent": 1,
                                                    "failed": 1}
    down.clear()                                          # the 13:00 retry sends only what did not go out
    assert newsletter.send_week(saturday.replace(hour=13), send)["sent"] == 1
    assert newsletter.send_week(saturday.replace(hour=13), send)["sent"] == 0
    assert mails == ["ravi@example.org", "sita@example.org", "sita@example.org"] and len(loads) == 2
    stale = newsletter.send_week(saturday + timedelta(days=7), send)              # the book stopped on 28 Sep
    assert stale["sent"] == 0 and "stale" in stale["skipped"] and len(mails) == 3
    for p in people:
        p["newsletter"] = False
    assert newsletter.send_week(saturday + timedelta(days=14), send)["readers"] == 0 and len(loads) == 3


def test_the_unsubscribe_link_asks_first_carries_no_address_and_stops_only_its_reader(api, monkeypatch):
    pytest.importorskip("cryptography")
    from urllib.parse import unquote

    from cryptography.fernet import Fernet

    from api import newsletter, users
    from api.auth import read_ticket

    monkeypatch.setenv(users.ENV_DATA_KEY, Fernet.generate_key().decode())
    monkeypatch.setenv(users.ENV_FRONTEND, "https://app.example.org/")
    monkeypatch.setattr(users, "_hits", {})
    url = newsletter.unsubscribe_link("Ravi@Example.org")
    assert url.startswith("https://app.example.org/newsletter/unsubscribe?token=")   # the website's page asks first
    token = unquote(url.split("token=", 1)[1])
    index = users.email_index("ravi@example.org")
    assert read_ticket(token, newsletter.UNSUBSCRIBE_PURPOSE, 60) == {"h": index}     # the keyed hash, no address
    changed = []
    monkeypatch.setattr(users, "set_newsletter", lambda i, on: changed.append((i, on)))
    client = api[1]
    assert client.post("/api/v1/auth/newsletter/unsubscribe", json={"token": token}).json() == {"ok": True}
    assert changed == [(index, False)]
    activation = unquote(users.link(users.ACTIVATE_PURPOSE, {"email": "ravi@example.org", "password_hash": "x"})
                         .split("token=", 1)[1])
    for bad in (token + "x", activation, ""):
        out = client.post("/api/v1/auth/newsletter/unsubscribe", json={"token": bad})
        assert out.status_code == 400 and "turn the newsletter off in Settings" in out.json()["detail"]
    assert len(changed) == 1


def test_a_signed_up_user_reads_and_sets_their_own_newsletter_choice_and_nobody_elses(api, monkeypatch):
    pytest.importorskip("cryptography")
    from cryptography.fernet import Fernet

    from api import users

    monkeypatch.setenv(users.ENV_DATA_KEY, Fernet.generate_key().decode())
    choice = {}
    monkeypatch.setattr(users, "set_newsletter", lambda index, on: choice.__setitem__(index, on))
    monkeypatch.setattr(users, "get", lambda email: {"newsletter": choice.get(users.email_index(email), False)})
    client, me = api[1], _as("ravi@example.org")
    assert client.get("/api/v1/auth/newsletter", headers=me).json() == {"subscribed": False}
    assert client.post("/api/v1/auth/newsletter", json={"subscribed": True}, headers=me).json()["subscribed"] is True
    assert client.get("/api/v1/auth/newsletter", headers=me).json() == {"subscribed": True}
    assert choice == {users.email_index("ravi@example.org"): True}
    for role in ("admin", "analyst"):                     # the operator gets the full Saturday email instead
        assert client.post("/api/v1/auth/newsletter", json={"subscribed": True},
                           headers=_as("me", role)).status_code == 400
    assert client.get("/api/v1/auth/newsletter").status_code == 401


# ── connected accounts through the API: a signed-up user sees and manages only their own ──

class _State:
    def __init__(self):
        self.state = {}

    def read_state(self):
        return dict(self.state)

    def sync_state(self, values):
        self.state.update(values)
        return True


@pytest.fixture
def registry_api(api, monkeypatch):
    """The account registry in memory, with an owner-signed-up user (ravi) and a stranger (sita)."""
    pytest.importorskip("cryptography")
    from collections import defaultdict

    from cryptography.fernet import Fernet

    from api import users
    from kite_connect.auth import accounts, daily_login

    monkeypatch.setenv(daily_login.ENV_TOKEN_KEY, Fernet.generate_key().decode())
    monkeypatch.setenv(users.ENV_DATA_KEY, Fernet.generate_key().decode())
    monkeypatch.setenv("ZERODHA_API_KEY", "ownkey12345")
    monkeypatch.delenv(daily_login.ENV_USER, raising=False)
    registry, books = _State(), defaultdict(_State)
    monkeypatch.setattr(accounts, "_registry_book", lambda book=None: book if book is not None else registry)
    monkeypatch.setattr(accounts, "account_book", lambda acct: books[acct.id])
    return api[1], accounts


def _as(name, role="user"):
    from api.auth import create_session_token

    return {"Authorization": f"Bearer {create_session_token(name, role)}"}


def _register(client, who, user_id):
    body = {"name": who.split("@")[0].title(), "zerodha_user_id": user_id, "api_key": f"{user_id}key12",
            "api_secret": f"{user_id}secret99"}
    return client.post("/api/v1/kite/accounts", json=body, headers=_as(who))


def test_a_signed_up_user_registers_one_account_and_never_sees_its_secret(registry_api):
    client, accounts = registry_api
    made = _register(client, "ravi@example.org", "AB1234")
    assert made.status_code == 200 and made.json()["id"] == "ravi" and "api_secret_enc" not in made.text
    assert "AB1234secret99" not in made.text and "redirect_params=account%3Dravi" in made.json()["login_url"]
    again = client.post("/api/v1/kite/accounts", json={"name": "Ravi 2", "zerodha_user_id": "CD5678",
                                                       "api_key": "otherkey12", "api_secret": "othersecret12"},
                        headers=_as("ravi@example.org"))
    assert again.status_code == 400 and "one account per user" in again.json()["detail"]
    listed = client.get("/api/v1/kite/accounts", headers=_as("ravi@example.org")).json()
    assert [a["id"] for a in listed["accounts"]] == ["ravi"]                     # not the owner's own account
    assert listed["setup"]["redirect_url"].endswith("/ind-stocks/auth/callback")


def test_a_user_cannot_touch_anyone_elses_account_and_is_told_it_does_not_exist(registry_api):
    client, accounts = registry_api
    _register(client, "ravi@example.org", "AB1234")
    _register(client, "sita@example.org", "CD5678")
    assert [a["id"] for a in client.get("/api/v1/kite/accounts", headers=_as("sita@example.org")).json()["accounts"]] == ["sita"]
    for path, body in (("update", {"email": "x@example.org"}), ("disconnect", {"how": "keep"}), ("remove", None)):
        out = client.post(f"/api/v1/kite/accounts/ravi/{path}", json=body, headers=_as("sita@example.org"))
        assert out.status_code == 400 and out.json()["detail"] == "no account 'ravi'", path
    assert client.get("/api/v1/kite/accounts/ravi/holdings", headers=_as("sita@example.org")).status_code == 400
    assert accounts.get("ravi").email == "" and "ravi" in accounts.load()          # untouched
    mine = client.post("/api/v1/kite/accounts/ravi/update", json={"email": "r@example.org"}, headers=_as("ravi@example.org"))
    assert mine.status_code == 200 and mine.json()["email"] == "r@example.org"
    assert client.get("/api/v1/kite/go-live", headers=_as("ravi@example.org")).status_code == 403


def test_the_owner_sees_every_account_and_an_unreadable_store_is_a_503_not_a_trace(registry_api, monkeypatch):
    client, accounts = registry_api
    _register(client, "ravi@example.org", "AB1234")
    owner = client.get("/api/v1/kite/accounts", headers=_as("me", "admin")).json()["accounts"]
    assert [a["id"] for a in owner] == ["primary", "ravi"]
    assert client.get("/api/v1/kite/accounts/ravi/holdings", headers=_as("me", "analyst")).status_code == 403
    gone = client.get("/api/v1/kite/accounts/ravi/holdings", headers=_as("me", "admin"))
    assert gone.status_code == 400 and "has not logged in to Kite today" in gone.json()["detail"]

    def down(*args, **kwargs):
        raise ConnectionError("password authentication failed for user neondb_owner")

    monkeypatch.setattr(accounts, "all_accounts", down)
    out = client.get("/api/v1/kite/accounts", headers=_as("me", "admin"))
    assert out.status_code == 503 and "neondb" not in out.text and "try again shortly" in out.json()["detail"]


# ── signing in and out (api/routers/v1/auth.py, api/auth.py) ─────

@pytest.fixture
def sign_in(api, monkeypatch):
    """One operator login (the credentials secret, in memory) and fresh failure counts."""
    import bcrypt

    from api import auth

    hashed = bcrypt.hashpw(b"Corr3ct!Horse-Battery", bcrypt.gensalt(rounds=4)).decode()
    monkeypatch.setattr(auth, "_get_credentials", lambda: {"users": {"owner": {
        "password": hashed, "name": "The Owner", "role": "admin"}}})
    monkeypatch.setattr(auth, "_login_failures", {})
    monkeypatch.setattr(auth, "_revoked", {})
    return api[1], auth


def test_a_correct_login_gets_a_token_that_works_and_logout_ends_it(sign_in):
    client, auth = sign_in
    out = client.post("/api/v1/auth/login", json={"username": "owner", "password": "Corr3ct!Horse-Battery"})
    assert out.status_code == 200 and out.json()["user"] == {"username": "owner", "name": "The Owner", "role": "admin"}
    token = out.json()["access_token"]
    header = {"Authorization": f"Bearer {token}"}
    assert client.get("/api/v1/auth/me", headers=header).json() == {"username": "owner", "name": "The Owner",
                                                                    "role": "admin"}
    other = client.post("/api/v1/auth/login", json={"username": "owner", "password": "Corr3ct!Horse-Battery"})
    assert other.json()["access_token"] != token                                 # each sign-in its own token
    assert client.post("/api/v1/auth/logout", headers=header).json() == {"ok": True}
    assert client.get("/api/v1/auth/me", headers=header).status_code == 401       # signed out for good
    assert client.get("/api/v1/kite/holdings", headers=header).status_code == 401
    still = {"Authorization": f"Bearer {other.json()['access_token']}"}
    assert client.get("/api/v1/auth/me", headers=still).status_code == 200       # the other session stands


def test_a_wrong_password_or_an_unknown_name_says_the_same_and_repeated_failures_lock_the_name(sign_in):
    client, auth = sign_in
    wrong = client.post("/api/v1/auth/login", json={"username": "owner", "password": "guess"})
    unknown = client.post("/api/v1/auth/login", json={"username": "nobody", "password": "guess"})
    assert wrong.status_code == unknown.status_code == 401 and wrong.json() == unknown.json()
    for _ in range(auth.MAX_LOGIN_FAILURES - 1):
        client.post("/api/v1/auth/login", json={"username": "owner", "password": "guess"})
    locked = client.post("/api/v1/auth/login", json={"username": "owner", "password": "Corr3ct!Horse-Battery"})
    assert locked.status_code == 429 and "try again later" in locked.json()["detail"]   # even the right password
    assert client.get("/api/v1/auth/me").status_code == 401
    assert client.get("/api/v1/auth/me", headers={"Authorization": "Bearer not-a-token"}).status_code == 401


# ── the owner's Kite routes (api/routers/v1/kite.py): session, holdings, P&L, orders ──

class _Kite:
    """The owner's Kite session: what each read returns, or the exception it raises; a tuple is served in turn,
    its last answer repeated."""

    def __init__(self, **reads):
        self.reads, self.calls, self.invalidated = {k: list(v) if isinstance(v, tuple) else v for k, v in reads.items()}, [], False
        self.turns = {k for k, v in reads.items() if isinstance(v, tuple)}

    def _read(self, name):
        self.calls.append(name)
        value = self.reads.get(name)
        if name in self.turns:
            value = value.pop(0) if len(value) > 1 else value[0]
        if isinstance(value, Exception):
            raise value
        return value

    def profile(self):
        return self._read("profile")

    def holdings(self):
        return self._read("holdings")

    def positions(self):
        return self._read("positions")

    def orders(self):
        return self._read("orders")

    def invalidate_access_token(self):
        self.invalidated = True
        return self._read("invalidate")


@pytest.fixture
def owner_kite(api, monkeypatch):
    """(client, owner's headers, a setter for tonight's Kite session, the route module)."""
    import api.routers.v1.kite as route

    session = {"kite": None}
    monkeypatch.setattr(route, "get_kite_session", lambda: session["kite"])
    monkeypatch.setattr(route, "_centurion_positions", lambda: {"INFY": 4, "TCS": 9})
    return api[1], _as("me", "admin"), lambda kite: session.__setitem__("kite", kite), route


def test_without_a_kite_session_every_broker_read_says_so_and_status_is_inactive(owner_kite):
    client, owner, use, _ = owner_kite
    for path in ("/api/v1/kite/holdings", "/api/v1/kite/positions", "/api/v1/kite/portfolio/pnl",
                 "/api/v1/kite/orders", "/api/v1/kite/quotes?symbols=INFY"):
        out = client.get(path, headers=owner)
        assert out.status_code == 409 and out.json()["detail"] == "Kite session not active", path
    assert client.get("/api/v1/kite/session/status", headers=owner).json()["active"] is False
    use(_Kite(profile={"user_id": "OWN001"}))
    assert client.get("/api/v1/kite/session/status", headers=owner).json() == {"active": True,
                                                                              "profile": {"user_id": "OWN001"}}
    use(_Kite(profile=RuntimeError("token gone")))
    assert client.get("/api/v1/kite/session/status", headers=owner).json()["active"] is False   # never a 500


def test_holdings_mark_what_centurion_bought_and_never_more_than_is_held(owner_kite):
    client, owner, use, _ = owner_kite
    use(_Kite(holdings=[{"tradingsymbol": "INFY", "quantity": 3, "t1_quantity": 1},       # bought 4: all four
                        {"tradingsymbol": "TCS", "quantity": 5, "t1_quantity": 0},        # 9 in the ledger, 5 held
                        {"tradingsymbol": "HDFCBANK", "quantity": 34}]))                  # your own: none
    marked = {h["tradingsymbol"]: h["centurion_qty"] for h in client.get("/api/v1/kite/holdings", headers=owner).json()}
    assert marked == {"INFY": 4, "TCS": 5, "HDFCBANK": 0}


def test_centurions_ledger_is_read_once_in_five_minutes_and_an_unreadable_one_marks_nothing(monkeypatch):
    pytest.importorskip("fastapi")
    import api.routers.v1.kite as route
    from infrastructure.cache import cache
    from kite_connect.auth import accounts

    reads = []
    cache.delete(route._CENTURION_POSITIONS_KEY)
    monkeypatch.setattr(accounts, "ledger_positions", lambda acct, book=None: reads.append(acct.id) or {"INFY": 4})
    assert route._centurion_positions() == {"INFY": 4} and route._centurion_positions() == {"INFY": 4}
    assert reads == ["primary"]                                                  # the second from the cache
    cache.delete(route._CENTURION_POSITIONS_KEY)

    def down(acct, book=None):
        raise ConnectionError("neon down")

    monkeypatch.setattr(accounts, "ledger_positions", down)
    assert route._centurion_positions() == {}
    assert cache.get(route._CENTURION_POSITIONS_KEY) is None                     # an outage is not cached
    cache.delete(route._CENTURION_POSITIONS_KEY)


def test_an_expired_token_asks_for_a_login_and_a_dropped_connection_is_retried_once(owner_kite):
    client, owner, use, _ = owner_kite
    TokenException = type("TokenException", (Exception,), {})
    use(_Kite(orders=TokenException("Incorrect `api_key` or `access_token`.")))
    out = client.get("/api/v1/kite/orders", headers=owner)
    assert out.status_code == 401 and out.json()["detail"] == "Kite session expired — log in again"
    flaky = _Kite(orders=(ConnectionError("reset by peer"), [{"order_id": "1"}]))
    use(flaky)
    assert client.get("/api/v1/kite/orders", headers=owner).json() == [{"order_id": "1"}]
    assert flaky.calls == ["orders", "orders"]
    dead = _Kite(orders=(ConnectionError("reset by peer"),))
    use(dead)
    out = client.get("/api/v1/kite/orders", headers=owner)
    assert out.status_code == 502 and "retried 1x" in out.json()["detail"] and dead.calls == ["orders", "orders"]


def test_the_pnl_view_adds_positions_and_holdings_once_each():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import api.main as main
    import api.routers.v1.kite as route

    kite = _Kite(positions={"net": [
        {"tradingsymbol": "NIFTY26NOVFUT", "quantity": -75, "average_price": 24_000.0, "last_price": 23_900.0,
         "pnl": 7_500.0, "day_m2m": 2_000.0},
        {"tradingsymbol": "CLOSED", "quantity": 0, "average_price": 10.0, "last_price": 11.0, "pnl": 50.0}]},
        holdings=[{"tradingsymbol": "INFY", "quantity": 4, "average_price": 1_150.0, "last_price": 1_160.0,
                   "pnl": 40.0, "day_change": 5.0},
                  {"tradingsymbol": "GONE", "quantity": 0, "average_price": 1.0, "last_price": 1.0, "pnl": 0.0}])
    original = route.get_kite_session
    route.get_kite_session = lambda: kite
    try:
        out = TestClient(main.app).get("/api/v1/kite/portfolio/pnl", headers=_as("me", "admin")).json()
    finally:
        route.get_kite_session = original
    assert out["total_invested"] == 75 * 24_000.0 + 4 * 1_150.0                   # a short counts at its size
    assert out["total_current"] == 75 * 23_900.0 + 4 * 1_160.0
    assert out["total_pnl"] == 7_540.0 and out["day_pnl"] == 2_000.0 + 5.0 * 4   # a holding's change is per share
    assert out["total_pnl_pct"] == round(7_540.0 / out["total_invested"] * 100, 2)
    assert [p["symbol"] for p in out["positions"]] == ["NIFTY26NOVFUT", "INFY"]   # nothing held: not listed
    assert out["positions"][1]["holding"] is True and out["positions"][1]["day_change"] == 20.0


def test_starting_a_session_uses_todays_stored_login_or_asks_for_one(api, monkeypatch):
    import api.dependencies as deps
    import api.routers.v1.kite as route
    from kite_connect.auth import daily_login

    client, opened = api[1], []
    monkeypatch.setattr(route, "get_kite_session", lambda: None)
    monkeypatch.setattr(deps, "set_kite_session", opened.append)
    monkeypatch.setattr(daily_login, "kite_from_stored_token", lambda *a, **k: _Kite(profile={"user_id": "OWN001"}))
    started = client.post("/api/v1/kite/session/start", headers=_as("me", "admin")).json()
    assert started == {"success": True, "profile": {"user_id": "OWN001"}} and len(opened) == 1
    monkeypatch.setattr(daily_login, "kite_from_stored_token", lambda *a, **k: None)
    asked = client.post("/api/v1/kite/session/start", headers=_as("me", "admin")).json()
    assert asked["success"] is False and asked["needs_login"] is True and "connect/login" in asked["login_url"]
    assert len(opened) == 1


def test_stopping_a_session_signs_out_at_kite_and_forgets_it_even_if_kite_is_down(api, monkeypatch):
    import api.dependencies as deps
    import api.routers.v1.kite as route

    client, cleared = api[1], []
    kite = _Kite(invalidate=ConnectionError("kite down"))
    monkeypatch.setattr(route, "get_kite_session", lambda: kite)
    monkeypatch.setattr(deps, "set_kite_session", cleared.append)
    assert client.post("/api/v1/kite/session/stop", headers=_as("me", "admin")).json() == {"success": True}
    assert kite.invalidated and cleared == [None]


def test_completing_a_login_by_hand_refuses_another_zerodha_user(api, monkeypatch):
    pytest.importorskip("kiteconnect")
    import kiteconnect

    import api.dependencies as deps
    from kite_connect.auth import daily_login

    client, opened = api[1], []

    class Kite:
        def __init__(self, api_key=None, pool=None):
            self.access_token = None

        def generate_session(self, request_token, api_secret):
            return {"user_id": "ZZ9999", "access_token": "t"}

        def set_access_token(self, token):
            self.access_token = token

    monkeypatch.setattr(kiteconnect, "KiteConnect", Kite)
    monkeypatch.setattr(deps, "set_kite_session", opened.append)
    monkeypatch.setenv(daily_login.ENV_USER, "OWN001")
    out = client.post("/api/v1/kite/session/complete", json={"request_token": "r"}, headers=_as("me", "admin"))
    assert out.status_code == 400 and "accepts only OWN001" in out.json()["detail"] and opened == []


# ── the screener (api/routers/v1/screener.py): orders only for symbols the scorer calls a buy ──

class _ScreenerKite:
    """What the manual checks read from Kite: today's trades and the last prices."""

    def __init__(self, sold_today=(), ltp=100.0):
        self.sold_today, self.last = set(sold_today), ltp

    def positions(self):
        return {"day": [{"tradingsymbol": s, "exchange": "NSE", "sell_quantity": 1} for s in self.sold_today]}

    def ltp(self, keys):
        if self.last is None:
            raise ConnectionError("quote service down")
        return {k: {"last_price": self.last} for k in keys}


@pytest.fixture
def screener(api, monkeypatch):
    """(client, owner headers, the orders sent, the scorer's verdicts by symbol, a session setter, the kill switch)."""
    import sys
    from types import SimpleNamespace

    import api.routers.v1.screener as route
    from kite_connect.trading import order_service

    n = SimpleNamespace(client=api[1], owner=_as("me", "admin"), sent=[], verdicts={}, kite=_ScreenerKite(),
                        scorer_error=None, killed=False, real_place_order=order_service.place_order)
    monkeypatch.setattr(order_service, "is_kill_switch_active", lambda: n.killed)

    class Scorer:
        def evaluate(self, tickers, market, date_range, skip_layers):
            if n.scorer_error:
                raise n.scorer_error
            assert market == "IND" and skip_layers == ["rag"] and all(t.endswith(".NS") for t in tickers)
            return [SimpleNamespace(ticker=t, classification=n.verdicts.get(t[:-3], "HOLD")) for t in tickers]

    def place(kite, **order):
        if order["symbol"] == "BOOM":
            raise ConnectionError("broker down")
        n.sent.append(order)
        return {"success": True, "symbol": order["symbol"]}

    monkeypatch.setitem(sys.modules, "services.signals.integrated_scorer", SimpleNamespace(IntegratedScorer=Scorer))
    monkeypatch.setattr(order_service, "place_order", place)
    monkeypatch.setattr(route, "get_kite_session", lambda: n.kite)
    return n


def test_the_screener_sends_orders_only_for_symbols_the_scorer_calls_a_buy(screener):
    screener.verdicts = {"INFY": "BUY", "TCS": "STRONG_BUY", "ITC": "HOLD", "SBIN": "SELL"}
    plans = [{"symbol": s, "quantity": 2, "price": 100.0, "order_type": "LIMIT"} for s in ("INFY", "TCS", "ITC", "SBIN")]
    out = screener.client.post("/api/v1/screener/execute", json={"plans": plans}, headers=screener.owner).json()
    assert [o["symbol"] for o in screener.sent] == ["INFY", "TCS"] and out["verdict_blocked"] == 2
    assert out["total_placed"] == 2 and all(o["product"] == "CNC" and o["exchange"] == "NSE" for o in screener.sent)
    assert screener.sent[0]["order_type"] == "LIMIT" and screener.sent[0]["price"] == 100.0


def test_the_screener_reports_a_failed_send_and_blocks_everything_it_cannot_score(screener):
    screener.verdicts = {"INFY": "BUY", "BOOM": "BUY"}
    out = screener.client.post("/api/v1/screener/execute", json={"plans": [{"symbol": "BOOM", "quantity": 1},
                                                                             {"symbol": "INFY", "quantity": 1}]},
                               headers=screener.owner).json()
    assert out["orders"][0] == {"success": False, "error": "broker down", "symbol": "BOOM"} and len(screener.sent) == 1
    screener.verdicts = {}
    none = screener.client.post("/api/v1/screener/execute", json={"plans": [{"symbol": "INFY", "quantity": 1}]},
                                headers=screener.owner).json()
    assert none == {"orders": [], "message": "No plans passed verdict filter (1 blocked)"} and len(screener.sent) == 1
    assert screener.client.post("/api/v1/screener/execute", json={}, headers=screener.owner).json() == {
        "orders": [], "message": "No plans provided"}
    screener.scorer_error = RuntimeError("scorer data missing")
    failed = screener.client.post("/api/v1/screener/execute", json={"plans": [{"symbol": "INFY", "quantity": 1}]},
                                  headers=screener.owner)
    assert failed.status_code == 500 and len(screener.sent) == 1                  # no score: no order
    screener.kite = None
    assert screener.client.post("/api/v1/screener/execute", json={"plans": [{"symbol": "INFY"}]},
                                headers=screener.owner).status_code == 409


def test_the_monitors_book_choice_is_a_known_paper_book_or_the_deployed_one(api, monkeypatch):
    from fastapi import HTTPException

    import api.routers.v1.screener as route

    books = route._paper_books()
    assert books[0]["book"] == "deployed" and books[0]["schema"] is None
    assert all(b["schema"] == b["book"] for b in books[1:]) and len(books) >= 2
    assert route._book_param(None) is None and route._book_param("deployed") is None
    with pytest.raises(HTTPException) as unknown:
        route._book_param("nope")
    assert unknown.value.status_code == 404
    monkeypatch.setattr(route, "_cloud_or_none", lambda book=None: None)
    with pytest.raises(HTTPException) as no_neon:
        route._book_param(books[1]["book"])
    assert no_neon.value.status_code == 503
    monkeypatch.setattr(route, "_cloud_or_none", lambda book=None: object())
    assert route._book_param(books[1]["book"]) == books[1]["book"]
    assert api[1].get("/api/v1/screener/monitor/sessions?book=nope", headers=_as("me", "admin")).status_code == 404


# ── which API routes may send an order to Kite ──────────────────

#: Every API function that sends a Kite order, with why it may.  A new one is added here only with the manual
#: route's checks or the owner's decision recorded in the tracker.
ORDER_SENDERS = {
    ("api/routers/v1/kite.py", "manual_send"): "every hand-placed order, after the manual checks (RB5, RB7, OR1)",
}
#: The routes that send through ``manual_send``.
MANUAL_SENDERS = {("api/routers/v1/kite.py", "kite_place_order"), ("api/routers/v1/screener.py", "screener_execute")}


def _order_senders(root):
    """{(file, function)} for every function under api/ that uses order_service.place_order."""
    import ast

    found = set()
    for path in sorted((root / "api").rglob("*.py")):
        tree = ast.parse(path.read_text())
        top = {a.asname or a.name for n in tree.body if isinstance(n, ast.ImportFrom)
               and n.module == "kite_connect.trading.order_service" for a in n.names if a.name == "place_order"}
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            local = {a.asname or a.name for n in ast.walk(fn) if isinstance(n, ast.ImportFrom)
                     and n.module == "kite_connect.trading.order_service" for a in n.names if a.name == "place_order"}
            names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
            if local or (top & names):
                found.add((path.relative_to(root).as_posix(), fn.name))
    return found


@pytest.mark.lesson("L-ORDER-ROUTE", pattern="F1", date="2026-11-16", source="TS2c",
                    summary="A second manual order route sent orders without the value limit, the duplicate check "
                            "or exits marked; retired")
def test_only_the_known_routes_send_orders_to_kite():
    from pathlib import Path

    import ast

    root = Path(__file__).resolve().parent.parent
    assert _order_senders(root) == set(ORDER_SENDERS)
    callers = set()
    for path in sorted((root / "api").rglob("*.py")):
        for fn in ast.walk(ast.parse(path.read_text())):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name != "manual_send" and any(
                    isinstance(n, ast.Name) and n.id == "manual_send" for n in ast.walk(fn)):
                callers.add((path.relative_to(root).as_posix(), fn.name))
    assert callers == MANUAL_SENDERS


@pytest.mark.lesson("L-ORDER-ROUTE")
def test_the_retired_ind_stocks_order_route_places_nothing(api):
    client = api[1]
    owner = _as("me", "admin")
    out = client.post("/ind-stocks/orders", headers=owner, json={"symbol": "INFY", "transaction_type": "BUY", "quantity": 1})
    assert out.status_code == 405                                                 # the path stays for the order book
    paths = api[0].app.openapi()["paths"]
    assert "post" not in paths["/ind-stocks/orders"] and "get" in paths["/ind-stocks/orders"]
    assert "post" in paths["/api/v1/kite/orders"]


# ── the screener's orders pass the manual route's checks (tracker OR1) ──

@pytest.mark.lesson("L-ORDER-ROUTE")
@pytest.mark.parametrize("plan,kite,killed,status,why", [
    ({"symbol": "INFY", "quantity": 300, "price": 1_150.0, "order_type": "LIMIT"}, {}, False, 422,
     "above the manual limit of Rs 200,000"),
    ({"symbol": "INFY", "quantity": 1}, {"ltp": None}, False, 422, "cannot value this MARKET order"),
    ({"symbol": "INFY", "quantity": 1, "price": 1_150.0, "order_type": "LIMIT"}, {"sold_today": ["INFY"]}, False, 422,
     "would make an intraday trade"),
    ({"symbol": "INFY", "quantity": 1, "price": 1_150.0, "order_type": "LIMIT"}, {}, True, 423, "KILL SWITCH"),
    ({"symbol": "INFY", "quantity": 1, "product": "MIS"}, {}, False, 422, None),
])
def test_a_screener_order_the_manual_route_would_refuse_is_refused_and_never_sent(screener, plan, kite, killed, status,
                                                                                why):
    screener.verdicts, screener.killed = {"INFY": "BUY"}, killed
    screener.kite = _ScreenerKite(**kite)
    out = screener.client.post("/api/v1/screener/execute", json={"plans": [plan]}, headers=screener.owner).json()
    ((refused,),) = (out["orders"],)
    assert refused["success"] is False and refused["status_code"] == status and refused["symbol"] == "INFY"
    assert why is None or why in refused["error"]
    assert screener.sent == []


def test_executing_the_same_plans_again_the_same_day_sends_each_order_once(screener, gate, monkeypatch):
    """The real order gate over a broker: the second Execute finds each plan's order by its tag and returns it."""
    from conftest import _Broker
    from kite_connect.trading import order_service

    broker = _Broker()                                                        # its positions: no trades today
    monkeypatch.setattr(order_service, "place_order", screener.real_place_order)
    screener.kite, screener.verdicts = broker, {"INFY": "BUY", "TCS": "BUY"}
    plans = [{"symbol": "INFY", "quantity": 2, "price": 1_150.0, "order_type": "LIMIT"},
             {"symbol": "TCS", "quantity": 1, "price": 3_000.0, "order_type": "LIMIT"}]
    first = screener.client.post("/api/v1/screener/execute", json={"plans": plans}, headers=screener.owner).json()
    again = screener.client.post("/api/v1/screener/execute", json={"plans": plans}, headers=screener.owner).json()
    assert broker.sent == 2 and [o["success"] for o in first["orders"]] == [True, True]
    assert [o.get("duplicate") for o in again["orders"]] == [True, True]
    assert [o["order_id"] for o in again["orders"]] == [o["order_id"] for o in first["orders"]]


def test_a_plans_tag_is_the_same_all_day_and_one_the_kill_switch_drain_cancels():
    pytest.importorskip("fastapi")
    from datetime import date

    from api.routers.v1.screener import plan_tag
    from kite_connect.trading.order_service import _OWN_BUY_TAG

    order = {"symbol": "INFY", "exchange": "NSE", "transaction_type": "BUY", "quantity": 2, "order_type": "LIMIT",
             "product": "CNC", "price": 1_150.0, "trigger_price": None}
    tag = plan_tag(order, date(2026, 11, 16))
    assert tag == plan_tag(dict(order), date(2026, 11, 16)) and len(tag) == 20 and _OWN_BUY_TAG.fullmatch(tag)
    assert plan_tag(order, date(2026, 11, 17)) != tag                        # tomorrow's plan is a new order
    assert plan_tag({**order, "quantity": 3}, date(2026, 11, 16)) != tag
