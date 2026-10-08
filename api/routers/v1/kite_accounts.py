"""/api/v1/kite/accounts: the Zerodha accounts Centurion connects (trackers U33, MU1).

Your own account (the primary) and any user's, all on the same criteria:
each holder's acceptance of the current terms before a dry run, and
Centurion's registration too before live orders (``kite_connect.auth.accounts``
explains the rule).  Every
route needs a signed-in session, and another holder's holdings an admin; the
app secret is accepted, stored encrypted and never returned.  A signed-up
user (role ``user``, MU2) sees and manages only the one account they own;
the middleware lets them write here and nowhere else under /api/v1/kite.
"""

import asyncio
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from api.dependencies import require_session
from api.routers.v1.common import logger

router = APIRouter()


class AddAccount(BaseModel):
    name: str
    zerodha_user_id: str
    api_key: str
    api_secret: str
    email: str = ""


class UpdateAccount(BaseModel):
    email: Optional[str] = None
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    mode: Optional[str] = None
    capital: Optional[float] = None


class Disconnect(BaseModel):
    how: str                              # keep | next_open | sessions
    sessions: int = 0


def _callback_url(request: Request) -> str:
    """The redirect URL each account's Kite Connect app must register: this server's callback."""
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    return f"{proto}://{request.headers.get('host') or request.url.netloc}/ind-stocks/auth/callback"


def _view(acct) -> dict:
    from kite_connect.auth import accounts

    row = {**acct.public(), "login_url": accounts.login_url(acct), "consented": accounts.has_consent(acct),
           "locks": {m: accounts.trading_lock(acct, m) for m in ("dry_run", "live")}}
    try:
        book = accounts.account_book(acct)
        status = accounts.login_status(acct, book)
        row.update(logged_in_today=bool(status["valid"]), login_at=status["login_at"])
        if not acct.is_primary:
            row["positions"] = len(accounts.ledger_positions(acct, book))
    except Exception as exc:                              # noqa: BLE001 - one unreadable book must not hide the rest
        logger.warning("login status of %s unavailable: %s", acct.id, exc)
        row.update(logged_in_today=False, login_at=None)
    return row


def _owner(user: dict) -> str:
    """The signed-up user whose accounts the request is limited to, as their email's keyed hash
    (``api.users.email_index``); "" for your own logins."""
    from api.users import email_index

    return email_index(str(user.get("u", ""))) if user.get("r") == "user" else ""


def _check_owner(account_id: str, user: dict):
    """The account, refused (as unknown) when a signed-up user does not own it."""
    from kite_connect.auth import accounts

    acct = accounts.get(account_id)
    if _owner(user) and acct.owner != _owner(user):
        raise accounts.AccountError(f"no account {account_id!r}")
    return acct


def _refuse(exc: Exception):
    from kite_connect.auth.accounts import AccountError

    if isinstance(exc, AccountError):
        raise HTTPException(status_code=400, detail=str(exc))
    logger.exception("Kite accounts request failed")
    raise HTTPException(status_code=503, detail="account store unavailable: try again shortly")


@router.get("/kite/accounts")
async def list_accounts(request: Request, user: dict = Depends(require_session)):
    """Every account with today's login state and its login link, plus what a new account's app needs."""
    from kite_connect.auth import accounts, daily_login, terms
    from nse_engine.capital_ladder import RUNGS

    owner = _owner(user)
    try:
        rows = await asyncio.to_thread(lambda: [_view(a) for a in accounts.all_accounts()
                                                 if not owner or a.owner == owner])
    except Exception as exc:                              # noqa: BLE001
        _refuse(exc)
    return {"accounts": rows,
            "setup": {"redirect_url": _callback_url(request),
                      "static_ip": os.environ.get(daily_login.ENV_STATIC_IP) or None,
                      "rungs": list(RUNGS),
                      "terms": {"version": terms.TERMS_VERSION, "items": list(terms.TERMS)},
                      "registration_missing": accounts.registration_missing()}}


@router.post("/kite/accounts")
async def add_account(body: AddAccount, user: dict = Depends(require_session)):
    """Register a user's account: their Kite Connect app's key and secret, never a password.
    A signed-up user registers their own, once."""
    from kite_connect.auth import accounts

    try:
        acct = await asyncio.to_thread(accounts.add, body.name, body.zerodha_user_id,
                                       body.api_key, body.api_secret, body.email, owner=_owner(user))
    except Exception as exc:                              # noqa: BLE001
        _refuse(exc)
    logger.info("Kite account %s (%s) added by %s", acct.id, acct.zerodha_user_id, user.get("u"))
    return await asyncio.to_thread(_view, acct)


@router.post("/kite/accounts/{account_id}/update")
async def update_account(account_id: str, body: UpdateAccount, user: dict = Depends(require_session)):
    """Change the email for login links, rotate the app's key and secret together, or set the mode and capital."""
    from kite_connect.auth import accounts

    try:
        await asyncio.to_thread(_check_owner, account_id, user)
        acct = await asyncio.to_thread(accounts.update, account_id, email=body.email, api_key=body.api_key,
                                       api_secret=body.api_secret, mode=body.mode, capital=body.capital)
    except Exception as exc:                              # noqa: BLE001
        _refuse(exc)
    logger.info("Kite account %s updated by %s (mode %s, capital %s)", account_id, user.get("u"), acct.mode, acct.capital)
    return await asyncio.to_thread(_view, acct)


@router.post("/kite/accounts/{account_id}/disconnect")
async def disconnect_account(account_id: str, body: Disconnect, user: dict = Depends(require_session)):
    """Stop Centurion trading an account: keep its positions, or sell them at the next open or over N sessions."""
    from kite_connect.auth import accounts

    try:
        await asyncio.to_thread(_check_owner, account_id, user)
        acct = await asyncio.to_thread(accounts.disconnect, account_id, body.how, body.sessions)
    except Exception as exc:                              # noqa: BLE001
        _refuse(exc)
    logger.info("Kite account %s disconnect (%s, %s sessions) by %s", account_id, body.how, body.sessions, user.get("u"))
    return await asyncio.to_thread(_view, acct)


@router.post("/kite/accounts/{account_id}/remove")
async def remove_account(account_id: str, user: dict = Depends(require_session)):
    """Forget an account that has never traded."""
    from kite_connect.auth import accounts

    try:
        await asyncio.to_thread(_check_owner, account_id, user)
        await asyncio.to_thread(accounts.remove, account_id)
    except Exception as exc:                              # noqa: BLE001
        _refuse(exc)
    logger.info("Kite account %s removed by %s", account_id, user.get("u"))
    return {"removed": account_id}


@router.get("/kite/accounts/{account_id}/holdings")
async def account_holdings(account_id: str, user: dict = Depends(require_session)):
    """A connected account's holdings (each with ``centurion_qty``, as on Holdings) and available funds,
    read with its holder's login today: an admin, or the signed-up user who owns it."""
    from kiteconnect.exceptions import KiteException

    from kite_connect.auth import accounts, daily_login

    if user.get("r") not in ("admin", "user"):
        raise HTTPException(status_code=403, detail="only an admin can see another account's holdings")

    def read() -> dict:
        acct = _check_owner(account_id, user)
        if acct.is_primary:
            raise accounts.AccountError("your own holdings are on the Holdings tab")
        book = accounts.account_book(acct)
        kite = daily_login.kite_from_stored_token(book, key=acct.api_key)
        if kite is None:
            raise accounts.AccountError(f"{acct.name} has not logged in to Kite today")
        mine = accounts.ledger_positions(acct, book)
        try:
            holdings, margins = kite.holdings() or [], kite.margins("equity") or {}
        except KiteException as exc:                      # Kite's own message: a revoked token, a disabled app
            raise accounts.AccountError(f"Kite refused the read: {exc}")
        for h in holdings:
            held = int(h.get("quantity") or 0) + int(h.get("t1_quantity") or 0)
            h["centurion_qty"] = min(int(mine.get(h.get("tradingsymbol"), 0)), held)
        return {"holdings": holdings, "available_funds": margins.get("net")}

    try:
        return await asyncio.to_thread(read)
    except Exception as exc:                              # noqa: BLE001
        _refuse(exc)
