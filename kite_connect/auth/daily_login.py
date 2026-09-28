"""
Daily Kite login for the live book (decision U23, tracker D3).

Zerodha's access tokens expire at 06:00 IST every day (a regulatory rule),
refresh tokens are for approved platforms only, and a scripted login
(stored password + TOTP secret) breaks Kite Connect's terms.  So a person
logs in once per trading day, and everything else is automatic:

1. reminder: on NSE trading days an email carries the Kite login link
   (09:00 IST, and again at 17:30 if nobody has logged in).  The link opens
   Zerodha's own login page: the password and authenticator code are typed
   there and nowhere else;
2. callback: Zerodha redirects to ``/ind-stocks/auth/callback`` on the HF
   Space, which exchanges the one-time request token for the day's access
   token and stores it in Neon, encrypted with Fernet (key
   ``CENTURION_KITE_TOKEN_KEY``, a secret on the Space and in GitHub Actions
   only), with the login time and the Zerodha user id;
3. the evening live session (``live_session --stored-token``) reads it: a
   token is usable when the login happened after 06:00 IST today.  Order
   calls go through the static-IP proxy ``CENTURION_KITE_PROXY`` whose IP is
   registered with Zerodha (required for API orders since April 2026).

A stolen token can read the account until 06:00 but cannot place orders
from any other IP.  ``CENTURION_KITE_USER_ID`` (optional) rejects a login by
any other Zerodha user.

    python -m kite_connect.auth.daily_login status
    python -m kite_connect.auth.daily_login remind [--force]
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, time, timedelta, timezone
from typing import Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))
TOKEN_EXPIRY = time(6, 0)                         # 06:00 IST, Kite Connect
KEY_TOKEN, KEY_LOGIN_AT, KEY_USER = "kite_access_token", "kite_login_at", "kite_user_id"
ENV_TOKEN_KEY, ENV_PROXY, ENV_USER = "CENTURION_KITE_TOKEN_KEY", "CENTURION_KITE_PROXY", "CENTURION_KITE_USER_ID"
ENV_STATIC_IP = "CENTURION_KITE_STATIC_IP"        # the IP registered with Zerodha (Oracle reserved IP)
ENV_LIVE_MODE = "CENTURION_LIVE_MODE"             # off | dry_run | live
LOGIN_URL = "https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"


def api_key() -> str:
    return (os.environ.get("ZERODHA_API_KEY") or "").strip()


def login_url(key: Optional[str] = None) -> str:
    key = key or api_key()
    if not key:
        raise RuntimeError("ZERODHA_API_KEY is not set")
    return LOGIN_URL.format(api_key=key)


def token_window(now: Optional[datetime] = None) -> Tuple[datetime, datetime]:
    """(start, end) in IST of the token day containing ``now``: 06:00 to 06:00."""
    now = (now or datetime.now(IST)).astimezone(IST)
    start = datetime.combine(now.date(), TOKEN_EXPIRY, tzinfo=IST)
    if now < start:
        start -= timedelta(days=1)
    return start, start + timedelta(days=1)


def _fernet(key: Optional[str] = None):
    from cryptography.fernet import Fernet     # ships with kiteconnect (pyOpenSSL)

    key = key or os.environ.get(ENV_TOKEN_KEY)
    if not key:
        raise RuntimeError(f"{ENV_TOKEN_KEY} is not set: the Kite token is stored only encrypted")
    return Fernet(key.encode() if isinstance(key, str) else key)


def _book(book=None):
    if book is not None:
        return book
    from kite_connect.trading.live_session import live_book
    return live_book()


def save_token(access_token: str, user_id: str, login_at: Optional[datetime] = None,
               book=None, key: Optional[str] = None) -> Dict[str, str]:
    """Store the day's access token (encrypted) with its login time and user."""
    login_at = (login_at or datetime.now(IST)).astimezone(IST)
    values = {KEY_TOKEN: _fernet(key).encrypt(access_token.encode()).decode(),
              KEY_LOGIN_AT: login_at.isoformat(timespec="seconds"), KEY_USER: str(user_id or "")}
    if not _book(book).sync_state(values):
        raise RuntimeError("could not store the Kite token in Neon")
    logger.info("Kite token stored for %s, logged in %s", user_id, values[KEY_LOGIN_AT])
    return {k: v for k, v in values.items() if k != KEY_TOKEN}


def token_status(book=None, now: Optional[datetime] = None) -> Dict[str, object]:
    """Whether today's token exists, without decrypting it."""
    state = _book(book).read_state()
    start, end = token_window(now)
    raw = state.get(KEY_LOGIN_AT) or ""
    try:
        login_at = datetime.fromisoformat(raw).astimezone(IST) if raw else None
    except ValueError:
        login_at = None
    valid = bool(state.get(KEY_TOKEN)) and login_at is not None and start <= login_at < end
    return {"valid": valid, "login_at": login_at.isoformat() if login_at else None,
            "user_id": state.get(KEY_USER) or None, "expires": end.isoformat()}


def load_token(book=None, now: Optional[datetime] = None, key: Optional[str] = None) -> Optional[str]:
    """Today's access token, or None when nobody has logged in since 06:00 IST."""
    b = _book(book)
    if not token_status(b, now)["valid"]:
        return None
    return _fernet(key).decrypt(b.read_state()[KEY_TOKEN].encode()).decode()


def exchange_and_store(request_token: str, *, key: Optional[str] = None, secret: Optional[str] = None,
                       expected_user: Optional[str] = None, book=None, kite_factory: Optional[Callable] = None) -> dict:
    """Request token -> access token (Kite), checked against the expected user, stored."""
    key = key or api_key()
    secret = secret or os.environ.get("ZERODHA_API_SECRET", "")
    if not key or not secret:
        raise RuntimeError("ZERODHA_API_KEY / ZERODHA_API_SECRET are not set on this server")
    if kite_factory is None:
        from kiteconnect import KiteConnect as kite_factory
    kite = kite_factory(api_key=key)
    data = kite.generate_session(request_token, api_secret=secret)
    user = str(data.get("user_id") or "")
    expected = expected_user if expected_user is not None else os.environ.get(ENV_USER, "")
    if expected and user != expected:
        raise PermissionError(f"logged in as {user}, but this app accepts only {expected}")
    kite.set_access_token(data["access_token"])
    stored = save_token(data["access_token"], user, book=book)
    return {"kite": kite, "user_id": user, **stored}


def proxies(url: Optional[str] = None) -> Optional[Dict[str, str]]:
    url = url if url is not None else os.environ.get(ENV_PROXY, "")
    return {"http": url, "https": url} if url else None


def egress_ip(proxy_url: Optional[str] = None) -> Optional[str]:
    """The public IP Zerodha sees for calls through the proxy (a diagnostic)."""
    try:
        import requests
        return requests.get("https://api.ipify.org", proxies=proxies(proxy_url), timeout=10).text.strip()
    except Exception as exc:                          # noqa: BLE001 - diagnostic only
        logger.warning("could not read the egress IP: %s", exc)
        return None


def check_egress(proxy_url: Optional[str] = None, expected: Optional[str] = None) -> Tuple[Optional[str], bool]:
    """(egress IP, matches) against the registered static IP; True when none is configured."""
    expected = (expected if expected is not None else os.environ.get(ENV_STATIC_IP, "")).strip()
    ip = egress_ip(proxy_url)
    return ip, (not expected) or ip == expected


def kite_from_stored_token(book=None, now: Optional[datetime] = None, proxy_url: Optional[str] = None,
                           kite_factory: Optional[Callable] = None):
    """A Kite session from today's stored token (through the proxy), or None."""
    token = load_token(book, now)
    if token is None:
        return None
    if kite_factory is None:
        from kiteconnect import KiteConnect as kite_factory
    kite = kite_factory(api_key=api_key(), proxies=proxies(proxy_url))
    kite.set_access_token(token)
    return kite


def remind(book=None, now: Optional[datetime] = None, force: bool = False,
           send: Optional[Callable[[str, str], bool]] = None) -> str:
    """Email the login link on a trading day when today's token is missing."""
    now = (now or datetime.now(IST)).astimezone(IST)
    mode = (os.environ.get(ENV_LIVE_MODE) or "off").strip().lower()
    if mode not in ("dry_run", "live") and not force:
        return f"live mode is {mode!r}: no reminder"
    from services.execution.carver_pipeline import is_nse_trading_day
    if not is_nse_trading_day(now.date()) and not force:
        return f"{now.date()} is not an NSE trading day: no reminder"
    status = token_status(book, now)
    if status["valid"] and not force:
        return f"already logged in at {status['login_at']}: no reminder"
    url = login_url()
    label = mode.replace("_", " ")
    html = (
        "<html><body style=\"font-family:Segoe UI,Arial,sans-serif;background:#f9fafb;padding:20px;\">"
        "<div style=\"max-width:560px;margin:0 auto;background:#fff;border-radius:10px;"
        "box-shadow:0 2px 8px rgba(0,0,0,0.08);overflow:hidden;\">"
        "<div style=\"background:#1a1a2e;padding:14px 24px;color:#fff;font-size:17px;\">"
        f"Centurion &mdash; Kite login for {now:%A %d %b}</div><div style=\"padding:20px 24px;font-size:14px;\">"
        f"<p>Tonight's live session ({label}) needs today's Kite login.</p>"
        f"<p style=\"margin:22px 0;\"><a href=\"{url}\" style=\"background:#387ed1;color:#fff;padding:10px 18px;"
        "border-radius:6px;text-decoration:none;font-weight:600;\">Log in to Kite</a></p>"
        "<p style=\"color:#555;\">The button opens Zerodha's own login page: the password and authenticator code "
        "are typed there. Zerodha then returns you to the Centurion server, which keeps the session until "
        "06:00 tomorrow.</p><p style=\"color:#555;\">No login, no orders tonight. Your GTT stops stay active "
        "at Zerodha either way.</p></div></div></body></html>")
    subject = f"Kite login needed for {now:%a %d %b} ({label})"
    if send is None:
        from services.notifications.manager import NotificationManager
        send = lambda s, h: NotificationManager._send_html_email(s, h)   # noqa: E731
    ok = send(subject, html)
    return f"reminder {'sent' if ok else 'NOT sent (email failed)'}: {subject}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Daily Kite login for the live book (U23)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="is there a valid token for today?")
    r = sub.add_parser("remind", help="email the login link when today's token is missing")
    r.add_argument("--force", action="store_true", help="send even when not needed (test the email)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cmd == "status":
        print(token_status())
    else:
        print(remind(force=args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
