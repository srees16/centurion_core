"""
The Zerodha accounts Centurion may connect: yours (the primary) and your
family's (decision U33, 7 Oct 2026).

Why only family: SEBI's retail algo framework (April 2026) ties a static IP
to one trader at a broker, except that a spouse, dependent children and
dependent parents may share it; and running a strategy for anyone else needs
exchange empanelment through the broker and, for a black-box strategy, a
SEBI Research Analyst licence.  So a family account may trade through the
same registered proxy as yours; nobody else's may.

How an account connects: the account holder creates their own Kite Connect
app on developers.kite.trade with this server's callback as the redirect URL
and the registered static IP in its IP whitelist, then enters the app's API
key and secret here.  Centurion stores the key and the secret (encrypted with
``CENTURION_KITE_TOKEN_KEY``, as the daily tokens are) and never a password
or a TOTP secret: the holder logs in on Zerodha's own page each trading day
(decision U23), through a login link that carries the account's id back to
the callback (Kite's ``redirect_params``), where the login is checked
against the account's Zerodha user id.

Storage: the registry is one JSON value in the primary live book's state
(``REGISTRY_KEY``).  Each family account has its own live-book schema,
``live_<id>``, which holds its daily token, ledger, capital ladder and dry
runs.  The primary account is not in the registry: it is the existing live
book with the server's own credentials, unchanged.

Management (tracker FA2): each family account has a mode and a capital.
``off`` leaves it alone; ``dry_run`` builds its orders each evening and sends
none; ``live`` places them.  Your own ``CENTURION_LIVE_MODE`` is the master
switch: no family session runs while it is off, and a ``live`` account
places real orders only while it is ``live``, after its own go-live checks
(the first real session needs clean dry runs on that account).  The capital
is a ladder rung (``nse_engine.capital_ladder.RUNGS``): the first session's
size, later a request the ladder grants one rung at a time, as
``CENTURION_LIVE_CAPITAL`` is for your account.  The evening workflow runs
``live_session --stored-token --account <id>`` for every managed account
(``python -m kite_connect.auth.accounts managed``) and the reminder workflow
emails each holder their login link (``... accounts remind``).

Disconnecting (tracker FA3, decision U33) asks each time what happens to
the positions Centurion opened (the account ledger's, nothing else): keep
them (trading stops now; their GTT stops stay at Zerodha until deleted in
Kite), sell them at the next open, or sell them over N sessions (about 1/N
of each per session, the engine's stops kept on the rest).  The account
stays managed while it sells and turns ``off`` at the first session whose
ledger holds nothing, which also deletes its leftover stops.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import quote

from kite_connect.auth import daily_login

PRIMARY = "primary"
RELATIONS = ("spouse", "child", "parent")
REGISTRY_KEY = "kite_accounts"
ACCOUNT_PARAM = "account"                 # the redirect_params key the callback reads
MODES = ("off", "dry_run", "live")        # least to most: an account never runs above the master switch
DISCONNECT = ("keep", "next_open", "sessions")
MAX_UNWIND_SESSIONS = 20
_USER_ID = re.compile(r"^[A-Z0-9]{4,12}$")
_KEY = re.compile(r"^[A-Za-z0-9]{8,64}$")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MAX_FAMILY_ACCOUNTS = 6


class AccountError(ValueError):
    """A request the registry refuses (bad field, unknown or duplicate account)."""


@dataclass
class Account:
    id: str
    name: str
    relation: str
    zerodha_user_id: str
    api_key: str
    email: str = ""
    api_secret_enc: str = ""
    created_at: str = ""
    mode: str = "off"                     # MODES; family accounts only
    capital: float = 0.0                  # a ladder rung; 0 until chosen
    unwind_sessions: int = 0              # FA3: sessions left to sell out over; 0 = not disconnecting

    @property
    def is_primary(self) -> bool:
        return self.id == PRIMARY

    @property
    def schema(self) -> str:
        """The account's live-book schema."""
        from kite_connect.trading.live_session import live_schema
        return live_schema() if self.is_primary else f"live_{self.id}"

    def public(self) -> Dict[str, str]:
        """Everything but the secret."""
        return {k: v for k, v in asdict(self).items() if k != "api_secret_enc"}


def primary_account() -> Account:
    """Your own account: the existing live book and the server's credentials."""
    return Account(PRIMARY, "You", "self", os.environ.get(daily_login.ENV_USER, ""), daily_login.api_key())


def _registry_book(book=None):
    if book is not None:
        return book
    from kite_connect.trading.live_session import live_book
    return live_book()


def load(book=None) -> Dict[str, Account]:
    """The family accounts, by id."""
    raw = _registry_book(book).read_state().get(REGISTRY_KEY) or "{}"
    try:
        data = json.loads(raw)
    except ValueError:
        data = {}
    return {k: Account(**v) for k, v in data.items()}


def _save(accounts: Dict[str, Account], book=None) -> None:
    payload = json.dumps({k: asdict(v) for k, v in accounts.items()}, sort_keys=True)
    if not _registry_book(book).sync_state({REGISTRY_KEY: payload}):
        raise RuntimeError("could not store the account registry in Neon")


def all_accounts(book=None) -> List[Account]:
    """The primary, then the family accounts by name."""
    return [primary_account()] + sorted(load(book).values(), key=lambda a: a.name.lower())


def managed(book=None) -> List[Account]:
    """The family accounts Centurion trades (mode not ``off``), by name."""
    return sorted((a for a in load(book).values() if a.mode != "off"), key=lambda a: a.name.lower())


def get(account_id: str, book=None) -> Account:
    if account_id in ("", PRIMARY):
        return primary_account()
    accounts = load(book)
    if account_id not in accounts:
        raise AccountError(f"no account {account_id!r}")
    return accounts[account_id]


def _make_id(name: str, taken: List[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:20] or "account"
    if base == PRIMARY:
        base = "family"
    candidate, n = base, 2
    while candidate in taken:
        candidate, n = f"{base}_{n}", n + 1
    return candidate


def _check(name: str, relation: str, zerodha_user_id: str, api_key: str, api_secret: str, email: str) -> None:
    if not name.strip() or len(name) > 40:
        raise AccountError("name: 1 to 40 characters")
    if relation not in RELATIONS:
        raise AccountError(f"relation must be one of {RELATIONS}: only family may share the static IP (SEBI)")
    if not _USER_ID.match(zerodha_user_id):
        raise AccountError("Zerodha user id: 4 to 12 letters and digits, e.g. AB1234")
    if not _KEY.match(api_key) or not _KEY.match(api_secret):
        raise AccountError("API key and secret: the values shown on developers.kite.trade (letters and digits)")
    if email and not _EMAIL.match(email):
        raise AccountError("email is not valid")


def add(name: str, relation: str, zerodha_user_id: str, api_key: str, api_secret: str, email: str = "",
        book=None, now: Optional[datetime] = None) -> Account:
    """Register a family account (its secret is stored encrypted)."""
    zerodha_user_id = zerodha_user_id.strip().upper()
    api_key, api_secret, email = api_key.strip(), api_secret.strip(), email.strip()
    _check(name, relation, zerodha_user_id, api_key, api_secret, email)
    accounts = load(book)
    if len(accounts) >= MAX_FAMILY_ACCOUNTS:
        raise AccountError(f"at most {MAX_FAMILY_ACCOUNTS} family accounts")
    if zerodha_user_id == primary_account().zerodha_user_id or any(
            a.zerodha_user_id == zerodha_user_id for a in accounts.values()):
        raise AccountError(f"{zerodha_user_id} is already connected")
    acct = Account(_make_id(name, list(accounts)), name.strip(), relation, zerodha_user_id, api_key, email,
                   daily_login._fernet().encrypt(api_secret.encode()).decode(),
                   (now or datetime.now(daily_login.IST)).isoformat(timespec="seconds"))
    accounts[acct.id] = acct
    _save(accounts, book)
    return acct


def update(account_id: str, *, email: Optional[str] = None, api_key: Optional[str] = None,
           api_secret: Optional[str] = None, mode: Optional[str] = None, capital: Optional[float] = None,
           book=None) -> Account:
    """Change a family account's email, rotate its app credentials, or set its mode and capital."""
    accounts = load(book)
    acct = accounts.get(account_id)
    if acct is None:
        raise AccountError(f"no account {account_id!r}")
    if email is not None:
        if email.strip() and not _EMAIL.match(email.strip()):
            raise AccountError("email is not valid")
        acct.email = email.strip()
    if (api_key is None) != (api_secret is None):
        raise AccountError("rotate the API key and secret together")
    if api_key is not None:
        if not _KEY.match(api_key.strip()) or not _KEY.match(api_secret.strip()):
            raise AccountError("API key and secret: letters and digits")
        acct.api_key = api_key.strip()
        acct.api_secret_enc = daily_login._fernet().encrypt(api_secret.strip().encode()).decode()
    if capital is not None:
        from nse_engine.capital_ladder import RUNGS, rung_of
        if rung_of(capital) is None:
            raise AccountError(f"capital must be a ladder rung: {', '.join(f'{c:,.0f}' for c in RUNGS)}")
        acct.capital = float(capital)
    if mode is not None:
        if mode not in MODES:
            raise AccountError(f"mode must be one of {MODES}")
        if mode != "off" and not acct.capital:
            raise AccountError("choose the capital first: the first session sizes the book from it")
        if mode == "off" and acct.mode != "off" and ledger_positions(acct):
            raise AccountError(f"Centurion holds positions in {acct.name}'s account: disconnect it and choose "
                               "to keep them, sell at the next open or sell over N sessions")
        acct.mode, acct.unwind_sessions = mode, 0          # setting the mode also cancels a disconnect
    _save(accounts, book)
    return acct


def ledger_positions(acct: Account, book=None) -> Dict[str, int]:
    """What Centurion holds in the account: its live book's ledger, by symbol."""
    from kite_connect.trading.live_session import LIVE_LEDGER_KEY

    raw = (book or account_book(acct)).read_state().get(LIVE_LEDGER_KEY)
    return dict(json.loads(raw).get("positions") or {}) if raw else {}


def disconnect(account_id: str, how: str, sessions: int = 0, book=None) -> Account:
    """Stop managing a family account (FA3): ``keep`` its positions, or sell them at the
    ``next_open`` or over ``sessions`` sessions.  Without positions it is simply off."""
    if how not in DISCONNECT:
        raise AccountError(f"choose one of {DISCONNECT}")
    accounts = load(book)
    acct = accounts.get(account_id)
    if acct is None:
        raise AccountError(f"no account {account_id!r}")
    if how == "keep" or not ledger_positions(acct):
        acct.mode, acct.unwind_sessions = "off", 0
    else:
        n = 1 if how == "next_open" else int(sessions)
        if how == "sessions" and not 2 <= n <= MAX_UNWIND_SESSIONS:
            raise AccountError(f"sell over 2 to {MAX_UNWIND_SESSIONS} sessions")
        if acct.mode != "live":
            raise AccountError("selling needs the account in Live mode: a dry run places no orders")
        acct.unwind_sessions = n
    _save(accounts, book)
    return acct


def unwind_step(account_id: str, report: dict, dry_run: bool, book=None) -> None:
    """After a disconnecting account's session: off once it holds none of the ledger's positions,
    else one session fewer when tonight's sells were placed (a session that placed none does not count)."""
    accounts = load(book)
    acct = accounts.get(account_id)
    if acct is None or not acct.unwind_sessions:
        return
    if report.get("unwound"):
        acct.mode, acct.unwind_sessions = "off", 0
    elif not dry_run and any(r.get("status") in ("PLACED", "DUPLICATE") for r in report.get("results") or []
                             if r.get("type") != "gtt_reconcile"):
        acct.unwind_sessions = max(1, acct.unwind_sessions - 1)
    else:
        return
    _save(accounts, book)


def remove(account_id: str, book=None, account_book=None) -> None:
    """Forget a family account that has never traded (one with a ledger is disconnected instead)."""
    from kite_connect.trading.live_session import LIVE_LEDGER_KEY, live_book

    accounts = load(book)
    acct = accounts.get(account_id)
    if acct is None:
        raise AccountError(f"no account {account_id!r}")
    state = (account_book or live_book(acct.schema)).read_state()
    if state.get(LIVE_LEDGER_KEY):
        raise AccountError(f"{acct.name} has traded: disconnect it instead, which settles its positions")
    del accounts[account_id]
    _save(accounts, book)


def api_secret(acct: Account) -> str:
    """The app secret, decrypted; the primary's comes from the server's environment."""
    if acct.is_primary:
        return os.environ.get("ZERODHA_API_SECRET", "")
    return daily_login._fernet().decrypt(acct.api_secret_enc.encode()).decode()


def login_url(acct: Account) -> str:
    """Zerodha's login page for the account's app; a family login carries its id back to the callback."""
    url = daily_login.login_url(acct.api_key)
    return url if acct.is_primary else f"{url}&redirect_params={quote(f'{ACCOUNT_PARAM}={acct.id}', safe='')}"


def account_book(acct: Account):
    from kite_connect.trading.live_session import live_book
    return live_book(acct.schema)


def login_status(acct: Account, book=None, now: Optional[datetime] = None) -> Dict[str, object]:
    """Whether the account has today's token (``daily_login.token_status`` on its own book)."""
    return daily_login.token_status(book or account_book(acct), now)


def remind(acct: Account, now: Optional[datetime] = None, send=None) -> str:
    """Email the holder the account's login link when today's login is missing (``daily_login.remind``).

    The mode is the account's, capped by the master switch; without an email
    on the account the link comes to you.
    """
    master = (os.environ.get(daily_login.ENV_LIVE_MODE) or "off").strip().lower()
    mode = min(acct.mode, master, key=MODES.index) if master in MODES else "off"
    if send is None and acct.email:
        from services.notifications.manager import NotificationManager
        send = lambda s, h: NotificationManager._send_html_email(s, h, [acct.email])   # noqa: E731
    return daily_login.remind(account_book(acct), now, send=send, url=login_url(acct), mode=mode, holder=acct.name)


def exchange(acct: Account, request_token: str, book=None, kite_factory=None) -> dict:
    """The callback's step for this account: request token -> today's token, checked against its user id."""
    return daily_login.exchange_and_store(
        request_token, key=acct.api_key, secret=api_secret(acct),
        expected_user=None if acct.is_primary else acct.zerodha_user_id,
        book=book if book is not None else (None if acct.is_primary else account_book(acct)),
        kite_factory=kite_factory)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Family Kite accounts (tracker FA2)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("managed", help="the ids of the family accounts Centurion trades, one per line")
    sub.add_parser("remind", help="email each managed account's holder the login link when it is missing")
    args = ap.parse_args(argv)
    if args.cmd == "managed":
        for acct in managed():
            print(acct.id)
        return 0
    failed = 0
    for acct in managed():
        try:
            print(f"{acct.id}: {remind(acct)}")
        except Exception as exc:                          # noqa: BLE001 - one account must not stop the rest
            failed += 1
            print(f"{acct.id}: reminder failed: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
