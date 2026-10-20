"""
Email through Gmail's HTTPS API, for hosts that block SMTP (the HF Space: tracker MU2, Sentry CENTURION-CORE-4A,
20 Oct 2026, "[Errno 101] Network is unreachable" on port 587).

The Space sends the account emails (activation, resend, already registered).  Hugging Face blocks outbound SMTP,
so with these three settings ``NotificationManager._send_html_email`` sends through the Gmail API on port 443
instead, from the same Gmail account:
    CENTURION_GMAIL_CLIENT_ID, CENTURION_GMAIL_CLIENT_SECRET   an OAuth client (Desktop app) in Google Cloud
    CENTURION_GMAIL_REFRESH_TOKEN                              from ``authorize`` below, scope gmail.send only
Without them it sends over SMTP as before (GitHub Actions, this Mac).

    python -m services.notifications.gmail_api authorize        # once, on this Mac: prints the refresh token
    python -m services.notifications.gmail_api test --to you@example.com   # with the three settings exported

Publish the OAuth consent screen ("In production"): in Testing, Google expires refresh tokens after 7 days.
"""

from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import os
import secrets
import sys
import webbrowser
from email.message import Message
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, Optional, Sequence
from urllib.parse import parse_qs, urlencode, urlparse

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
SEND_URL = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"
SCOPE = "https://www.googleapis.com/auth/gmail.send"
ENV = ("CENTURION_GMAIL_CLIENT_ID", "CENTURION_GMAIL_CLIENT_SECRET", "CENTURION_GMAIL_REFRESH_TOKEN")


def _setting(name: str) -> str:
    return "".join(str(os.getenv(name) or "").split())


def configured() -> bool:
    """True when all three Gmail API settings are present."""
    return all(_setting(n) for n in ENV)


def _error(response) -> str:
    """A short reason from a Google error response, never echoing credentials."""
    try:
        body = response.json()
    except ValueError:
        return f"HTTP {response.status_code}"
    err = body.get("error")
    if isinstance(err, dict):
        return f"HTTP {response.status_code} {err.get('status') or ''} {err.get('message') or ''}".strip()
    return f"HTTP {response.status_code} {err or ''} {body.get('error_description') or ''}".strip()


def access_token() -> str:
    import requests

    r = requests.post(TOKEN_URL, data={"client_id": _setting(ENV[0]), "client_secret": _setting(ENV[1]),
                                       "refresh_token": _setting(ENV[2]), "grant_type": "refresh_token"}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Gmail token refresh failed: {_error(r)}")
    return r.json()["access_token"]


def send(message: Message) -> str:
    """Send a built MIME message; returns Gmail's message id.  Raises on failure."""
    import requests

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")
    r = requests.post(SEND_URL, headers={"Authorization": f"Bearer {access_token()}"}, json={"raw": raw}, timeout=30)
    if r.status_code >= 300:
        raise RuntimeError(f"Gmail send failed: {_error(r)}")
    return str(r.json().get("id", ""))


def authorize(client_id: str, client_secret: str, timeout: int = 300) -> str:
    """The one-time consent on this machine (loopback redirect with PKCE); returns the refresh token."""
    import requests

    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    state = secrets.token_urlsafe(16)
    got: Dict[str, str] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):                                         # noqa: N802 - http.server's name
            q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            if "code" in q or "error" in q:
                got.update(q)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"Centurion: done. You can close this tab and return to the terminal.")

        def log_message(self, *args):                              # keep the terminal clean
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    server.timeout = timeout
    redirect = f"http://127.0.0.1:{server.server_port}"
    url = AUTH_URL + "?" + urlencode({"client_id": client_id, "redirect_uri": redirect, "response_type": "code",
                                      "scope": SCOPE, "access_type": "offline", "prompt": "consent",
                                      "state": state, "code_challenge": challenge, "code_challenge_method": "S256"})
    print("Opening Google's consent page. Sign in with the Gmail account the emails come from.\n"
          f"If no browser opens, visit:\n{url}\n")
    webbrowser.open(url)
    while not got:
        server.handle_request()
        if not got:
            raise RuntimeError("no answer from Google's consent page within the time limit")
    server.server_close()
    if got.get("error"):
        raise RuntimeError(f"consent refused: {got['error']}")
    if got.get("state") != state:
        raise RuntimeError("the consent answer did not match this request (state); run authorize again")
    r = requests.post(TOKEN_URL, data={"code": got["code"], "client_id": client_id, "client_secret": client_secret,
                                       "redirect_uri": redirect, "grant_type": "authorization_code",
                                       "code_verifier": verifier}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"token exchange failed: {_error(r)}")
    token = r.json().get("refresh_token")
    if not token:
        raise RuntimeError("Google returned no refresh token: remove Centurion's access at "
                           "myaccount.google.com/permissions and run authorize again")
    return token


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m services.notifications.gmail_api", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("authorize", help="one-time consent on this Mac; prints the refresh token")
    t = sub.add_parser("test", help="send a test email through the Gmail API (the three settings exported)")
    t.add_argument("--to", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "authorize":
        client_id = input("OAuth client ID: ").strip()
        client_secret = getpass.getpass("OAuth client secret (hidden): ").strip()
        token = authorize(client_id, client_secret)
        print("\nRefresh token (store it as the secret CENTURION_GMAIL_REFRESH_TOKEN; do not share it):\n" + token)
        return 0
    from services.notifications.manager import NotificationManager

    if not configured():
        print("set CENTURION_GMAIL_CLIENT_ID, CENTURION_GMAIL_CLIENT_SECRET and CENTURION_GMAIL_REFRESH_TOKEN first",
              file=sys.stderr)
        return 2
    ok = NotificationManager._send_html_email("Centurion: Gmail API test",
                                              "<p>The Gmail API route works for the Space's account emails.</p>",
                                              [args.to])
    print("sent" if ok else "failed: see the log line above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
