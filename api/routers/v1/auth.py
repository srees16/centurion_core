"""/api/v1/auth/* routes: login, current user, logout, password change, and
self-service accounts (tracker MU2): sign-up, activation, resend, forgotten
password and reset (``api.users``).

Moved from v1_gateway.py (tracker H4), which includes this router under /api/v1.
The self-service routes are public (``api.main._PUBLIC``), answer the same
whether an email is registered or not, and are rate-limited per client and
per email.
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)

#: Public requests per client per hour, and emails per address per hour.
PER_CLIENT_HOUR, EMAILS_PER_ADDRESS_HOUR = 20, 3
CHECK_INBOX = "If the details are right, an email is on its way: check your inbox (and spam folder)."


class LoginRequest(BaseModel):
    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class SignupRequest(BaseModel):
    email: str
    password: str
    full_name: str
    date_of_birth: str
    gender: str
    city: str
    country: str
    state: str = ""
    phone: str = ""
    trading_experience: str = ""
    consent: bool = False


class EmailRequest(BaseModel):
    email: str


class TokenRequest(BaseModel):
    token: str


class ResetRequest(BaseModel):
    token: str
    new_password: str


class DeleteAccountRequest(BaseModel):
    password: str


def _client(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def _limit(request: Request, email: Optional[str] = None) -> bool:
    """False when this client, or this email address, is over its hourly allowance."""
    from api.users import allowed

    if not allowed(f"client:{_client(request)}", PER_CLIENT_HOUR, 3600):
        raise HTTPException(status_code=429, detail="Too many requests: try again in an hour")
    return email is None or allowed(f"email:{email.strip().lower()}", EMAILS_PER_ADDRESS_HOUR, 3600)


def _unavailable(exc: Exception):
    logger.exception("User accounts unavailable: %s", exc)
    raise HTTPException(status_code=503, detail="Accounts are unavailable right now: try again shortly")


# ─── Auth ────────────────────────────────────────────────────────────────

@router.post("/auth/login")
async def api_login(req: LoginRequest):
    """JWT login for the frontend."""
    from api.auth import (
        LoginThrottled,
        NotActivated,
        authenticate_user_async,
        create_session_token,
    )
    username = req.username.strip().lower() if "@" in req.username else req.username
    try:
        ok, display_name, role = await authenticate_user_async(username, req.password)
    except LoginThrottled:
        raise HTTPException(status_code=429, detail="Too many failed sign-ins: try again later")
    except NotActivated:
        raise HTTPException(status_code=403, detail="Your account is not activated yet: open the link we "
                                                    "emailed you, or send a new one")
    if not ok:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_session_token(username, role, display_name)
    return {
        "access_token": token,
        "refresh_token": token,
        "user": {"username": username, "name": display_name, "role": role},
    }


@router.get("/auth/me")
async def api_auth_me(request: Request):
    """Return the current user from the session token."""
    from api.auth import verify_session_token

    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = auth_header[7:]
    payload = verify_session_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return {"username": payload["u"], "name": payload.get("nm") or payload["u"], "role": payload["r"]}


@router.post("/auth/logout")
async def api_logout(request: Request):
    """Logout: the client clears its tokens and the server revokes the one it sent."""
    from api.auth import revoke_session_token

    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        revoke_session_token(auth_header[7:])
    return {"ok": True}


@router.post("/auth/change-password")
async def api_change_password(req: ChangePasswordRequest, request: Request):
    """Change the authenticated user's password (the policy of ``api.users.password_problem``).

    A signed-up user's other sessions end; the response carries a fresh token for this one.
    """
    from api.auth import (
        CREDENTIALS_YAML,
        _verify_password,
        create_session_token,
        end_sessions,
        verify_session_token,
    )
    from api.users import password_problem

    # Verify current session
    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    payload = verify_session_token(auth_header[7:])
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    username = payload["u"]

    if payload.get("r") == "user":                        # a signed-up account (MU2), kept in Neon
        from api import users
        try:
            user = await asyncio.to_thread(users.get, username)
        except Exception as exc:                          # noqa: BLE001
            _unavailable(exc)
        if not user or not _verify_password(req.current_password, user["password_hash"]):
            raise HTTPException(status_code=400, detail="Current password is incorrect")
        problem = password_problem(req.new_password, username, user["full_name"])
        if problem:
            raise HTTPException(status_code=400, detail=problem)
        await asyncio.to_thread(users.set_password, username, req.new_password)
        end_sessions(username)
        return {"ok": True, "access_token": create_session_token(username, "user", user["full_name"])}

    # Load credentials
    import yaml
    if not CREDENTIALS_YAML.exists():
        raise HTTPException(status_code=409, detail="Passwords on this server are set in the "
                                                    "CENTURION_CREDENTIALS_YAML secret: change them there")
    with open(CREDENTIALS_YAML, "r") as fh:
        creds = yaml.safe_load(fh) or {}
    users = creds.get("users", {})
    user = users.get(username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Verify current password
    if not _verify_password(req.current_password, user.get("password", "")):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    # Validate new password
    problem = password_problem(req.new_password, "", user.get("name", username))
    if problem:
        raise HTTPException(status_code=400, detail=problem)

    # Hash and save
    import bcrypt
    hashed = bcrypt.hashpw(req.new_password.encode(), bcrypt.gensalt()).decode()
    users[username]["password"] = hashed
    creds["users"] = users
    with open(CREDENTIALS_YAML, "w") as fh:
        yaml.dump(creds, fh, default_flow_style=False)

    # Invalidate the in-memory credential cache so login uses the new hash
    from api.auth import invalidate_credentials_cache
    invalidate_credentials_cache()

    return {"ok": True}


# ─── Self-service accounts (MU2) ─────────────────────────────────────────

@router.post("/auth/signup")
async def api_signup(req: SignupRequest, request: Request):
    """Create a pending account and email its activation link (24 hours)."""
    from api import users

    try:
        details = users.clean_signup(req.model_dump())
    except users.UserError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not _limit(request, details["email"]):
        return {"ok": True, "message": CHECK_INBOX}
    try:
        user = await asyncio.to_thread(users.sign_up, details, req.password)
        if user is None:                                  # already active: tell its owner, not the caller
            existing = await asyncio.to_thread(users.get, details["email"])
            await asyncio.to_thread(users.send_already_registered, existing)
            return {"ok": True, "message": CHECK_INBOX}
        sent = await asyncio.to_thread(users.send_activation, user)
    except Exception as exc:                              # noqa: BLE001
        _unavailable(exc)
    if not sent:
        raise HTTPException(status_code=503, detail="We could not send the activation email: try again in a "
                                                    "few minutes with 'Resend activation email'")
    return {"ok": True, "message": CHECK_INBOX}


@router.post("/auth/activate")
async def api_activate(req: TokenRequest, request: Request):
    """Activate an account from its emailed link."""
    from api import users

    _limit(request)
    try:
        data = users.read_link(req.token, users.ACTIVATE_PURPOSE)
        user = await asyncio.to_thread(users.activate, data["email"], data["fp"])
    except users.UserError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:                              # noqa: BLE001
        _unavailable(exc)
    logger.info("User account %s activated", user["id"])
    return {"ok": True, "email": user["email"]}


@router.post("/auth/resend-activation")
async def api_resend_activation(req: EmailRequest, request: Request):
    """Email a new activation link to a pending account (the same answer for any email)."""
    from api import users

    if _limit(request, req.email):
        try:
            user = await asyncio.to_thread(users.get, req.email)
            if user and user["status"] == "pending":
                await asyncio.to_thread(users.send_activation, user)
        except Exception as exc:                          # noqa: BLE001
            _unavailable(exc)
    return {"ok": True, "message": CHECK_INBOX}


@router.post("/auth/forgot-password")
async def api_forgot_password(req: EmailRequest, request: Request):
    """Email a password reset link (1 hour, once) to an active account (the same answer for any email)."""
    from api import users

    if _limit(request, req.email):
        try:
            user = await asyncio.to_thread(users.get, req.email)
            if user and user["status"] == "active":
                await asyncio.to_thread(users.send_reset, user)
        except Exception as exc:                          # noqa: BLE001
            _unavailable(exc)
    return {"ok": True, "message": CHECK_INBOX}


@router.post("/auth/reset-password")
async def api_reset_password(req: ResetRequest, request: Request):
    """Set a new password from a reset link; every existing session of the account ends."""
    from api import users
    from api.auth import end_sessions

    _limit(request)
    try:
        data = users.read_link(req.token, users.RESET_PURPOSE)
        user = await asyncio.to_thread(users.get, data["email"])
        if not user or user["status"] != "active" or users.fingerprint(user["password_hash"]) != data["fp"]:
            raise users.UserError("This reset link has been used or replaced: request a new one")
        problem = users.password_problem(req.new_password, user["email"], user["full_name"])
        if problem:
            raise users.UserError(problem)
        await asyncio.to_thread(users.set_password, user["email"], req.new_password)
    except users.UserError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:                              # noqa: BLE001
        _unavailable(exc)
    end_sessions(user["email"])
    logger.info("User account %s: password reset", user["id"])
    return {"ok": True}


@router.post("/auth/delete-account")
async def api_delete_account(req: DeleteAccountRequest, request: Request):
    """A signed-up user deletes their account, password first: their details, their connected Zerodha
    account's registry entry and Centurion's records of it (``api.users.delete``); every session ends."""
    from api import users
    from api.auth import _verify_password, end_sessions, session_from_request

    session = session_from_request(request) or {}
    if session.get("r") != "user":
        raise HTTPException(status_code=400, detail="Only a signed-up account can be deleted here")
    _limit(request)
    email = str(session.get("u", ""))
    try:
        user = await asyncio.to_thread(users.get, email)
        if not user or not _verify_password(req.password, user["password_hash"]):
            raise users.UserError("The password is incorrect")
        erased = await asyncio.to_thread(users.delete, email)
    except users.UserError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:                              # noqa: BLE001
        _unavailable(exc)
    end_sessions(email)
    logger.info("User account %s deleted (%d Zerodha account(s) erased)", user["id"], erased)
    try:
        await asyncio.to_thread(users.send_deleted, user)
    except Exception as exc:                              # noqa: BLE001 - the account is gone either way
        logger.warning("Deletion email failed: %s", exc)
    return {"ok": True}
