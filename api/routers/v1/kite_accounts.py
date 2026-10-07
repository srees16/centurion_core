"""/api/v1/kite/accounts: the Zerodha accounts Centurion may connect (decision U33).

Your own account (the primary) and your family's: a spouse, dependent
children and dependent parents, who may share your registered static IP
(``kite_connect.auth.accounts`` explains the rule).  Every route needs a
signed-in session; the app secret is accepted, stored encrypted and never
returned.
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
    relation: str
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

    row = {**acct.public(), "login_url": accounts.login_url(acct)}
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


def _refuse(exc: Exception):
    from kite_connect.auth.accounts import AccountError

    if isinstance(exc, AccountError):
        raise HTTPException(status_code=400, detail=str(exc))
    logger.exception("Kite accounts request failed")
    raise HTTPException(status_code=503, detail=f"account store unavailable: {exc}")


@router.get("/kite/accounts")
async def list_accounts(request: Request, user: dict = Depends(require_session)):
    """Every account with today's login state and its login link, plus what a new account's app needs."""
    from kite_connect.auth import accounts, daily_login
    from nse_engine.capital_ladder import RUNGS

    try:
        rows = await asyncio.to_thread(lambda: [_view(a) for a in accounts.all_accounts()])
    except Exception as exc:                              # noqa: BLE001
        _refuse(exc)
    return {"accounts": rows,
            "setup": {"redirect_url": _callback_url(request),
                      "static_ip": os.environ.get(daily_login.ENV_STATIC_IP) or None,
                      "relations": list(accounts.RELATIONS), "rungs": list(RUNGS)}}


@router.post("/kite/accounts")
async def add_account(body: AddAccount, user: dict = Depends(require_session)):
    """Register a family member's account: their Kite Connect app's key and secret, never a password."""
    from kite_connect.auth import accounts

    try:
        acct = await asyncio.to_thread(accounts.add, body.name, body.relation, body.zerodha_user_id,
                                       body.api_key, body.api_secret, body.email)
    except Exception as exc:                              # noqa: BLE001
        _refuse(exc)
    logger.info("Kite account %s (%s, %s) added by %s", acct.id, acct.relation, acct.zerodha_user_id, user.get("u"))
    return await asyncio.to_thread(_view, acct)


@router.post("/kite/accounts/{account_id}/update")
async def update_account(account_id: str, body: UpdateAccount, user: dict = Depends(require_session)):
    """Change the email for login links, rotate the app's key and secret together, or set the mode and capital."""
    from kite_connect.auth import accounts

    try:
        acct = await asyncio.to_thread(accounts.update, account_id, email=body.email, api_key=body.api_key,
                                       api_secret=body.api_secret, mode=body.mode, capital=body.capital)
    except Exception as exc:                              # noqa: BLE001
        _refuse(exc)
    logger.info("Kite account %s updated by %s (mode %s, capital %s)", account_id, user.get("u"), acct.mode, acct.capital)
    return await asyncio.to_thread(_view, acct)


@router.post("/kite/accounts/{account_id}/disconnect")
async def disconnect_account(account_id: str, body: Disconnect, user: dict = Depends(require_session)):
    """Stop Centurion trading a family account: keep its positions, or sell them at the next open or over N sessions."""
    from kite_connect.auth import accounts

    try:
        acct = await asyncio.to_thread(accounts.disconnect, account_id, body.how, body.sessions)
    except Exception as exc:                              # noqa: BLE001
        _refuse(exc)
    logger.info("Kite account %s disconnect (%s, %s sessions) by %s", account_id, body.how, body.sessions, user.get("u"))
    return await asyncio.to_thread(_view, acct)


@router.post("/kite/accounts/{account_id}/remove")
async def remove_account(account_id: str, user: dict = Depends(require_session)):
    """Forget a family account that has never traded."""
    from kite_connect.auth import accounts

    try:
        await asyncio.to_thread(accounts.remove, account_id)
    except Exception as exc:                              # noqa: BLE001
        _refuse(exc)
    logger.info("Kite account %s removed by %s", account_id, user.get("u"))
    return {"removed": account_id}
