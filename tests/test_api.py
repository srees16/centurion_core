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
                     "state": "Maharashtra", "country": "India", "trading_experience": "1_to_5_years"}
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
