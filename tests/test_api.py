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
