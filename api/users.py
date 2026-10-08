"""
Self-service user accounts (tracker MU2): sign-up with an email and a
password, activation by an emailed link, forgotten-password reset.

A user's account lives in Neon (``app.users``), next to the books; the
operator's own logins (admin, analyst) stay in the credentials secret
(``api.auth``).  Every personal detail (email, name, date of birth, gender,
mobile, city, state, country, experience) is one Fernet-encrypted field
(``profile_enc``, key ``CENTURION_USER_DATA_KEY``); the row is found by a
keyed hash of the email (``email_index``), which also names the user as the
owner of their Zerodha account (``kite_connect.auth.accounts``).  The
password is a bcrypt hash only.  Losing the key loses every profile: keep a
copy of it outside the Space.  A user deletes their account themselves
(``delete``, the ``/auth/delete-account`` route), which erases the row and
their connected Zerodha account's records.  A user signs in with their email; their role is ``user``,
which the API keeps away from the operator's broker account and books
(``api.main``).

Links: an activation link (24 hours) and a reset link (1 hour) are signed
tickets (``api.auth.sign_ticket``) bound to the email and to a fingerprint
of the current password hash, so a reset link works once (the password it
sets changes the fingerprint) and a newer sign-up for a pending email
voids the older links.  A link points at the frontend
(``CENTURION_FRONTEND_URL``), which posts the ticket back.

Email: ``NotificationManager._send_html_email`` with the server's SMTP
settings (``CENTURION_EMAIL_*``, needed on the Space as secrets).  The
public routes answer the same whether an email is registered or not, and
are rate-limited per client and per email (in this process).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import threading
import time
from datetime import date, datetime, timezone
from html import escape
from typing import Dict, List, Optional

ROLE = "user"
ENV_DATA_KEY = "CENTURION_USER_DATA_KEY"
ENV_FRONTEND = "CENTURION_FRONTEND_URL"
DEFAULT_FRONTEND = "https://centurion-core-fe.vercel.app"
ACTIVATE_PURPOSE, ACTIVATE_MAX_AGE_S = "activate", 24 * 3600
RESET_PURPOSE, RESET_MAX_AGE_S = "reset", 3600

GENDERS = ("female", "male", "other", "prefer_not_to_say")
EXPERIENCE = ("", "new", "under_1_year", "1_to_5_years", "over_5_years")
MIN_AGE = 18

#: Password policy (NIST SP 800-63B length, plus the classes most banks ask for): 12 to 64
#: characters within bcrypt's 72-byte limit, upper and lower case, a digit and a symbol,
#: and nothing guessable from the account or the site.
PASSWORD_MIN, PASSWORD_MAX, PASSWORD_MAX_BYTES = 12, 64, 72
_COMMON = ("password", "passw0rd", "centurion", "qwerty", "123456", "letmein", "welcome", "admin", "iloveyou")

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE = re.compile(r"^\+?[0-9][0-9 \-]{6,18}[0-9]$")

_TABLE = "app.users"
_ready = False
_ready_lock = threading.Lock()
_hits: Dict[str, List[float]] = {}
_hits_lock = threading.Lock()


class UserError(ValueError):
    """A sign-up, link or password the server refuses, with the reason to show."""


# ── Password policy ───────────────────────────────────────────────────────

def password_problem(password: str, email: str = "", name: str = "") -> str:
    """Why ``password`` breaks the policy ("" when it meets it)."""
    if not PASSWORD_MIN <= len(password) <= PASSWORD_MAX or len(password.encode()) > PASSWORD_MAX_BYTES:
        return f"Use {PASSWORD_MIN} to {PASSWORD_MAX} characters"
    if not (re.search(r"[a-z]", password) and re.search(r"[A-Z]", password)):
        return "Use upper and lower case letters"
    if not re.search(r"[0-9]", password):
        return "Include a digit"
    if not re.search(r"[^A-Za-z0-9]", password):
        return "Include a symbol, such as ! or #"
    lower = password.lower()
    personal = [email.split("@")[0].lower()] + [part.lower() for part in name.split()]
    if any(len(p) >= 3 and p in lower for p in personal):
        return "Leave your name and email out of the password"
    if any(word in lower for word in _COMMON):
        return "Avoid common words such as 'password' or 'centurion'"
    return ""


def hash_password(password: str) -> str:
    import bcrypt

    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def fingerprint(password_hash: str) -> str:
    """A short digest of the current password hash: links signed with it die when the password changes."""
    return hashlib.sha256(password_hash.encode()).hexdigest()[:16]


# ── Rate limits (public routes) ───────────────────────────────────────────

def allowed(key: str, limit: int, window_s: int) -> bool:
    """Record one request under ``key``; False when ``limit`` were already made in the window."""
    now = time.monotonic()
    with _hits_lock:
        recent = [t for t in _hits.get(key, []) if now - t < window_s]
        if len(recent) >= limit:
            _hits[key] = recent
            return False
        _hits[key] = recent + [now]
        return True


# ── Store (Neon) ──────────────────────────────────────────────────────────

#: The encrypted profile's fields, in the order the sign-up form gathers them.
PROFILE = ("email", "full_name", "date_of_birth", "gender", "phone", "city", "state", "country",
           "trading_experience")


def _key() -> bytes:
    key = os.environ.get(ENV_DATA_KEY, "").strip()
    if not key:
        raise RuntimeError(f"{ENV_DATA_KEY} is not set: user details are stored only encrypted")
    return key.encode()


def _fernet():
    from cryptography.fernet import Fernet

    return Fernet(_key())


def email_index(email: str) -> str:
    """A keyed hash of the email: finds the user's row and names them as an account owner, without
    the email itself being stored or comparable across keys."""
    secret = hashlib.sha256(b"centurion-email-index:" + _key()).digest()
    return hmac.new(secret, email.strip().lower().encode(), hashlib.sha256).hexdigest()


def _db():
    from database.connection import get_db_manager

    return get_db_manager()


def _ensure() -> None:
    global _ready
    if _ready:
        return
    from sqlalchemy import text

    with _ready_lock:
        if _ready:
            return
        with _db().get_session() as session:
            session.execute(text("CREATE SCHEMA IF NOT EXISTS app"))
            session.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {_TABLE} (
                    id                  BIGSERIAL PRIMARY KEY,
                    email_hash          CHAR(64)     NOT NULL UNIQUE,
                    profile_enc         TEXT         NOT NULL,
                    password_hash       VARCHAR(100) NOT NULL,
                    role                VARCHAR(20)  NOT NULL DEFAULT 'user',
                    status              VARCHAR(20)  NOT NULL DEFAULT 'pending',
                    consent_at          TIMESTAMPTZ  NOT NULL,
                    created_at          TIMESTAMPTZ  NOT NULL DEFAULT now(),
                    activated_at        TIMESTAMPTZ,
                    password_changed_at TIMESTAMPTZ  NOT NULL DEFAULT now(),
                    last_login_at       TIMESTAMPTZ
                )"""))
        _ready = True


def get(email: str) -> Optional[Dict]:
    """The user with this email (any status), their profile decrypted, or None."""
    from sqlalchemy import text

    _ensure()
    with _db().get_session() as session:
        row = session.execute(text(f"SELECT * FROM {_TABLE} WHERE email_hash = :h"),
                              {"h": email_index(email)}).mappings().fetchone()
    if row is None:
        return None
    user = {k: v for k, v in dict(row).items() if k != "profile_enc"}
    return {**user, **json.loads(_fernet().decrypt(row["profile_enc"].encode()))}


def _update(email: str, **values) -> None:
    from sqlalchemy import text

    _ensure()
    sets = ", ".join(f"{k} = :{k}" for k in values)
    with _db().get_session() as session:
        session.execute(text(f"UPDATE {_TABLE} SET {sets} WHERE email_hash = :h"), {**values, "h": email_index(email)})


def clean_signup(body: Dict) -> Dict:
    """The sign-up fields, checked and normalised (raises :class:`UserError`)."""
    email = str(body.get("email") or "").strip().lower()
    if len(email) > 254 or not _EMAIL.match(email):
        raise UserError("Enter a valid email address")
    name = " ".join(str(body.get("full_name") or "").split())
    if not 2 <= len(name) <= 80:
        raise UserError("Full name: 2 to 80 characters")
    try:
        dob = date.fromisoformat(str(body.get("date_of_birth") or ""))
    except ValueError:
        raise UserError("Date of birth: use the date picker") from None
    today = datetime.now(timezone.utc).date()
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    if not MIN_AGE <= age <= 120:
        raise UserError(f"You must be at least {MIN_AGE} to trade")
    gender = str(body.get("gender") or "")
    if gender not in GENDERS:
        raise UserError("Choose a gender option")
    phone = " ".join(str(body.get("phone") or "").split())
    if phone and not _PHONE.match(phone):
        raise UserError("Mobile number: digits, optionally starting with + and the country code")
    city, state, country = (" ".join(str(body.get(k) or "").split()) for k in ("city", "state", "country"))
    if not 1 <= len(city) <= 60 or not 2 <= len(country) <= 60 or len(state) > 60:
        raise UserError("City and country are required (up to 60 characters each)")
    experience = str(body.get("trading_experience") or "")
    if experience not in EXPERIENCE:
        raise UserError("Choose a trading experience option")
    if body.get("consent") is not True:
        raise UserError("Agree to Centurion storing your details to create the account")
    problem = password_problem(str(body.get("password") or ""), email, name)
    if problem:
        raise UserError(problem)
    return {"email": email, "full_name": name, "date_of_birth": dob.isoformat(), "gender": gender,
            "phone": phone, "city": city, "state": state, "country": country, "trading_experience": experience}


def sign_up(details: Dict, password: str) -> Optional[Dict]:
    """A pending user, their details encrypted (a pending one with this email is replaced: the newest
    sign-up's links win).  None when the email belongs to an active account."""
    from sqlalchemy import text

    existing = get(details["email"])
    if existing and existing["status"] != "pending":
        return None
    values = {"email_hash": email_index(details["email"]),
              "profile_enc": _fernet().encrypt(json.dumps({k: details[k] for k in PROFILE}).encode()).decode(),
              "password_hash": hash_password(password), "consent_at": datetime.now(timezone.utc)}
    with _db().get_session() as session:
        if existing:
            session.execute(text(f"DELETE FROM {_TABLE} WHERE email_hash = :h AND status = 'pending'"),
                            {"h": values["email_hash"]})
        session.execute(text(f"INSERT INTO {_TABLE} ({', '.join(values)}) "
                             f"VALUES ({', '.join(':' + c for c in values)})"), values)
    return get(details["email"])


def activate(email: str, fp: str) -> Dict:
    """Activate a pending account from its link (an active one is left as it is)."""
    user = get(email)
    if user is None or fingerprint(user["password_hash"]) != fp:
        raise UserError("This activation link is no longer valid: sign up again or request a new link")
    if user["status"] == "pending":
        _update(email, status="active", activated_at=datetime.now(timezone.utc))
    elif user["status"] != "active":
        raise UserError("This account is disabled")
    return get(email)


def set_password(email: str, password: str) -> None:
    _update(email, password_hash=hash_password(password), password_changed_at=datetime.now(timezone.utc))


def touch_login(email: str) -> None:
    _update(email, last_login_at=datetime.now(timezone.utc))


def delete(email: str) -> int:
    """Erase the user: their Zerodha account's records (``accounts.erase``), then their row.
    Returns how many connected accounts were erased."""
    from sqlalchemy import text

    from kite_connect.auth import accounts

    owner = email_index(email)
    owned = [a.id for a in accounts.load().values() if a.owner == owner]
    for account_id in owned:
        accounts.erase(account_id)
    _ensure()
    with _db().get_session() as session:
        session.execute(text(f"DELETE FROM {_TABLE} WHERE email_hash = :h"), {"h": owner})
    return len(owned)


# ── Links and emails ──────────────────────────────────────────────────────

def frontend_url() -> str:
    return (os.environ.get(ENV_FRONTEND) or DEFAULT_FRONTEND).rstrip("/")


def link(purpose: str, user: Dict) -> str:
    from urllib.parse import quote

    from api.auth import sign_ticket

    ticket = sign_ticket(purpose, {"e": user["email"], "f": fingerprint(user["password_hash"])})
    page = "activate" if purpose == ACTIVATE_PURPOSE else "reset-password"
    return f"{frontend_url()}/{page}?token={quote(ticket)}"


def read_link(token: str, purpose: str) -> Dict:
    """The email and fingerprint a link carries (raises :class:`UserError` when invalid or expired)."""
    from api.auth import read_ticket

    data = read_ticket(token, purpose, ACTIVATE_MAX_AGE_S if purpose == ACTIVATE_PURPOSE else RESET_MAX_AGE_S)
    if not isinstance(data, dict) or not data.get("e") or not data.get("f"):
        raise UserError("This link is invalid or has expired: request a new one")
    return {"email": str(data["e"]), "fp": str(data["f"])}


def _email(subject: str, heading: str, body: str, user: Dict, button: str = "", url: str = "") -> bool:
    from services.notifications.manager import NotificationManager

    action = (f"<p style='margin:24px 0;'><a href='{escape(url)}' style='background:#2563eb;color:#fff;"
              f"padding:10px 20px;border-radius:6px;text-decoration:none;'>{escape(button)}</a></p>"
              f"<p style='color:#666;font-size:13px;'>Or open this link: {escape(url)}</p>") if url else ""
    html = ("<html><body style='font-family:Segoe UI,Arial,sans-serif;background:#f9fafb;padding:20px;'>"
            "<div style='max-width:560px;margin:auto;background:#fff;border-radius:10px;padding:24px;'>"
            f"<h2 style='margin-top:0;'>{escape(heading)}</h2>"
            f"<p>Hello {escape(user['full_name'])},</p><p>{body}</p>{action}"
            "<p style='color:#666;font-size:13px;'>If this was not you, ignore this email: nothing changes.</p>"
            "</div></body></html>")
    return NotificationManager._send_html_email(subject, html, [user["email"]])


def send_activation(user: Dict) -> bool:
    return _email("Activate your Centurion account", "Activate your account",
                  "Thanks for signing up. Activate your account to sign in; the link works for 24 hours.",
                  user, "Activate account", link(ACTIVATE_PURPOSE, user))


def send_reset(user: Dict) -> bool:
    return _email("Reset your Centurion password", "Reset your password",
                  "Someone asked to reset the password for this account. The link works for 1 hour, once.",
                  user, "Choose a new password", link(RESET_PURPOSE, user))


def send_already_registered(user: Dict) -> bool:
    return _email("Your Centurion account", "You already have an account",
                  "Someone tried to sign up with this email, which already has an active account. Sign in, or "
                  "reset your password from the sign-in page if you have forgotten it.", user,
                  "Sign in", f"{frontend_url()}/login")


def send_deleted(user: Dict) -> bool:
    return _email("Your Centurion account was deleted", "Account deleted",
                  "Your Centurion account and the details you gave us have been deleted, with Centurion's records "
                  "of any Zerodha account you connected. Your holdings stay in your Zerodha account; any "
                  "stop-loss orders Centurion placed stay at Zerodha until you delete them in Kite.", user)
