"""/api/v1/auth/* routes: login, current user, logout, password change.

Moved from v1_gateway.py (tracker H4), which includes this router under /api/v1.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class LoginRequest(BaseModel):
    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


# ─── Auth ────────────────────────────────────────────────────────────────

@router.post("/auth/login")
async def api_login(req: LoginRequest):
    """JWT login for the frontend."""
    from api.auth import LoginThrottled, authenticate_user_async, create_session_token
    try:
        ok, display_name, role = await authenticate_user_async(req.username, req.password)
    except LoginThrottled:
        raise HTTPException(status_code=429, detail="Too many failed sign-ins: try again later")
    if not ok:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_session_token(req.username, role)
    return {
        "access_token": token,
        "refresh_token": token,
        "user": {"username": req.username, "name": display_name, "role": role},
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
    return {"username": payload["u"], "name": payload["u"], "role": payload["r"]}


@router.post("/auth/logout")
async def api_logout():
    """Logout — client clears tokens; server acknowledges."""
    return {"ok": True}


@router.post("/auth/change-password")
async def api_change_password(req: ChangePasswordRequest, request: Request):
    """Change the authenticated user's password."""
    from api.auth import verify_session_token, _verify_password, CREDENTIALS_YAML

    # Verify current session
    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    payload = verify_session_token(auth_header[7:])
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    username = payload["u"]

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
    if len(req.new_password) < 6:
        raise HTTPException(status_code=400, detail="New password must be at least 6 characters")

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
