"""
The Zerodha accounts Centurion connects: yours (the primary, the server's
own credentials) and any user's (tracker MU1, 8 Oct 2026), all on the same
criteria: there is no relation or family tier.

Every connected account needs (``trading_lock``) its holder's acceptance of
the current terms (``kite_connect.auth.terms``), given on the page their own
Kite login returns to (``record_consent``), and Centurion's registration on
the server (``REGISTRATION_ENV``): SEBI's retail algo framework (April 2026)
makes running a strategy for another person's account an empanelled algo
provider's business through the broker (with a Research Analyst licence for
a black-box strategy), and discretionary management of other people's money
is portfolio management (PMS registration).  Until both settings hold, an
account connects, logs in, is read and runs dry runs (its orders built,
none sent: decision U35, 8 Oct 2026, so every feature can be tried), and
Centurion places no real orders in it.  The registration settings are set only on a
lawyer's advice: NSE runs a provider's strategies on the broker's servers
(NSE/INVG/69255), so holding the registrations may still not permit orders
from this server.  Your own account (the primary) needs neither the terms
nor any registration: its book, dry runs, go-live checks and every
research and deployment tool are untouched by this module's lock.

How an account connects: the holder's own Kite Connect app on
developers.kite.trade with this server's callback as the redirect URL, whose
API key and secret are entered here.  A signed-up user (``api.users``, MU2)
connects at most one account, their own (``owner``), and manages only it.  Centurion stores the key and the secret (encrypted with
``CENTURION_KITE_TOKEN_KEY``, as the daily tokens are) and never a password
or a TOTP secret: the holder logs in on Zerodha's own page each trading day
(decision U23), through a login link that carries the account's id back to
the callback (Kite's ``redirect_params``), where the login is checked
against the account's Zerodha user id.

Storage: the registry is one JSON value in the primary live book's state
(``REGISTRY_KEY``).  Each registered account has its own live-book schema,
``live_<id>``, which holds its daily token, ledger, capital ladder and dry
runs.  The primary account is not in the registry: it is the existing live
book with the server's own credentials, unchanged.

Management (tracker FA2): each registered account has a mode and a capital.
``off`` leaves it alone; ``dry_run`` builds its orders each evening and sends
none; ``live`` places them.  Your own ``CENTURION_LIVE_MODE`` is the master
switch: no other account's session runs while it is off, and a ``live`` account
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
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import quote

from kite_connect.auth import daily_login, terms

PRIMARY = "primary"
#: Set on the server once Centurion is registered: the exchange's algo / empanelment id (through the
#: broker) and the SEBI registration number (RA or PMS).  Both unlock live orders in every account.
REGISTRATION_ENV = ("CENTURION_ALGO_PROVIDER_ID", "CENTURION_SEBI_REGISTRATION")
REGISTRY_KEY = "kite_accounts"
ACCOUNT_PARAM = "account"                 # the redirect_params key the callback reads
MODES = ("off", "dry_run", "live")        # least to most: an account never runs above the master switch
DISCONNECT = ("keep", "next_open", "sessions")
MAX_UNWIND_SESSIONS = 20
_USER_ID = re.compile(r"^[A-Z0-9]{4,12}$")
_KEY = re.compile(r"^[A-Za-z0-9]{8,64}$")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MAX_ACCOUNTS = 20


class AccountError(ValueError):
    """A request the registry refuses (bad field, unknown or duplicate account)."""


@dataclass
class Account:
    id: str
    name: str
    zerodha_user_id: str
    api_key: str
    email: str = ""
    api_secret_enc: str = ""
    created_at: str = ""
    mode: str = "off"                     # MODES; registered accounts only
    capital: float = 0.0                  # a ladder rung; 0 until chosen
    unwind_sessions: int = 0              # FA3: sessions left to sell out over; 0 = not disconnecting
    consent_version: str = ""             # MU1: the terms the holder accepted, when and as whom
    consent_at: str = ""
    consent_by: str = ""
    owner: str = ""                       # MU2: its signed-up user (api.users.email_index); "" = yours

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
    return Account(PRIMARY, "You", os.environ.get(daily_login.ENV_USER, ""), daily_login.api_key())


def _registry_book(book=None):
    if book is not None:
        return book
    from kite_connect.trading.live_session import live_book
    return live_book()


def load(book=None) -> Dict[str, Account]:
    """The registered accounts, by id."""
    raw = _registry_book(book).read_state().get(REGISTRY_KEY) or "{}"
    try:
        data = json.loads(raw)
    except ValueError:
        data = {}
    known = {f.name for f in fields(Account)}           # a stored account may carry a retired field (relation)
    return {k: Account(**{f: x for f, x in v.items() if f in known}) for k, v in data.items()}


def _save(accounts: Dict[str, Account], book=None) -> None:
    payload = json.dumps({k: asdict(v) for k, v in accounts.items()}, sort_keys=True)
    if not _registry_book(book).sync_state({REGISTRY_KEY: payload}):
        raise RuntimeError("could not store the account registry in Neon")


def all_accounts(book=None) -> List[Account]:
    """The primary, then the registered accounts by name."""
    return [primary_account()] + sorted(load(book).values(), key=lambda a: a.name.lower())


def managed(book=None) -> List[Account]:
    """The registered accounts Centurion trades (mode not ``off``), by name."""
    return sorted((a for a in load(book).values() if a.mode != "off"), key=lambda a: a.name.lower())


def registration_missing() -> List[str]:
    """The registration settings the server lacks (empty once Centurion is registered)."""
    return [name for name in REGISTRATION_ENV if not os.environ.get(name, "").strip()]


def has_consent(acct: Account) -> bool:
    """Has the holder accepted the current terms?  Your own account (the primary) needs none here."""
    return acct.is_primary or acct.consent_version == terms.TERMS_VERSION


def trading_lock(acct: Account, mode: str = "live") -> str:
    """Why Centurion may not run the account in ``mode`` ("" when it may): the same for every connected
    account, never for your own.  A dry run needs the holder's terms; live orders also Centurion's
    registration (U35)."""
    if acct.is_primary or mode == "off":
        return ""
    if not has_consent(acct):
        return "the holder has not accepted the current terms: they do on the page their next Kite login returns to"
    if mode == "live" and registration_missing():
        return ("live orders need Centurion's exchange empanelment and SEBI registration ("
                + ", ".join(registration_missing()) + " not set): dry runs only until then")
    return ""


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
        base = "account"
    candidate, n = base, 2
    while candidate in taken:
        candidate, n = f"{base}_{n}", n + 1
    return candidate


def _check(name: str, zerodha_user_id: str, api_key: str, api_secret: str, email: str) -> None:
    if not name.strip() or len(name) > 40:
        raise AccountError("name: 1 to 40 characters")
    if not _USER_ID.match(zerodha_user_id):
        raise AccountError("Zerodha user id: 4 to 12 letters and digits, e.g. AB1234")
    if not _KEY.match(api_key) or not _KEY.match(api_secret):
        raise AccountError("API key and secret: the values shown on developers.kite.trade (letters and digits)")
    if email and not _EMAIL.match(email):
        raise AccountError("email is not valid")


def add(name: str, zerodha_user_id: str, api_key: str, api_secret: str, email: str = "",
        book=None, now: Optional[datetime] = None, owner: str = "") -> Account:
    """Register an account (its secret is stored encrypted); a signed-up ``owner`` has at most one."""
    zerodha_user_id = zerodha_user_id.strip().upper()
    api_key, api_secret, email = api_key.strip(), api_secret.strip(), email.strip()
    _check(name, zerodha_user_id, api_key, api_secret, email)
    accounts = load(book)
    if len(accounts) >= MAX_ACCOUNTS:
        raise AccountError(f"at most {MAX_ACCOUNTS} accounts")
    if zerodha_user_id == primary_account().zerodha_user_id or any(
            a.zerodha_user_id == zerodha_user_id for a in accounts.values()):
        raise AccountError(f"{zerodha_user_id} is already connected")
    if owner and any(a.owner == owner for a in accounts.values()):
        raise AccountError("your Zerodha account is already connected: one account per user")
    acct = Account(_make_id(name, list(accounts)), name.strip(), zerodha_user_id, api_key, email,
                   daily_login._fernet().encrypt(api_secret.encode()).decode(),
                   (now or datetime.now(daily_login.IST)).isoformat(timespec="seconds"), owner=owner)
    accounts[acct.id] = acct
    _save(accounts, book)
    return acct


def update(account_id: str, *, email: Optional[str] = None, api_key: Optional[str] = None,
           api_secret: Optional[str] = None, mode: Optional[str] = None, capital: Optional[float] = None,
           book=None) -> Account:
    """Change an account's email, rotate its app credentials, or set its mode and capital."""
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
        if trading_lock(acct, mode):
            raise AccountError(trading_lock(acct, mode))
        if mode == "off" and acct.mode != "off" and ledger_positions(acct):
            raise AccountError(f"Centurion holds positions in {acct.name}'s account: disconnect it and choose "
                               "to keep them, sell at the next open or sell over N sessions")
        acct.mode, acct.unwind_sessions = mode, 0          # setting the mode also cancels a disconnect
    _save(accounts, book)
    return acct


def record_consent(account_id: str, user_id: str, version: str, book=None,
                   now: Optional[datetime] = None) -> Account:
    """The holder's acceptance of the terms, given right after their own Kite login as ``user_id``."""
    accounts = load(book)
    acct = accounts.get(account_id)
    if acct is None:
        raise AccountError(f"no account {account_id!r}")
    if version != terms.TERMS_VERSION:
        raise AccountError("these terms have changed: log in again to see the current ones")
    if user_id != acct.zerodha_user_id:
        raise AccountError(f"the terms are accepted by {acct.zerodha_user_id} only")
    acct.consent_version, acct.consent_by = version, user_id
    acct.consent_at = (now or datetime.now(daily_login.IST)).isoformat(timespec="seconds")
    _save(accounts, book)
    return acct


def ledger_positions(acct: Account, book=None) -> Dict[str, int]:
    """What Centurion holds in the account: its live book's ledger, by symbol."""
    from kite_connect.trading.live_session import LIVE_LEDGER_KEY

    raw = (book or account_book(acct)).read_state().get(LIVE_LEDGER_KEY)
    return dict(json.loads(raw).get("positions") or {}) if raw else {}


def disconnect(account_id: str, how: str, sessions: int = 0, book=None) -> Account:
    """Stop managing an account (FA3): ``keep`` its positions, or sell them at the
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
    """Forget an account that has never traded (one with a ledger is disconnected instead)."""
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


def erase(account_id: str, book=None) -> None:
    """Forget an account and everything Centurion recorded for it: its registry entry, then its
    live-book schema (daily token, ledger, dry runs).  For a holder who deletes their Centurion
    account (MU2): trading stops at once; their holdings, and any GTT stops Centurion placed, stay
    at Zerodha."""
    from sqlalchemy import text

    from database.connection import get_db_manager

    accounts = load(book)
    acct = accounts.pop(account_id, None)
    if acct is None:
        return
    _save(accounts, book)
    if not re.fullmatch(r"live_[a-z0-9_]+", acct.schema):
        raise AccountError(f"refusing to drop schema {acct.schema!r}")
    with get_db_manager().get_session() as session:
        session.execute(text(f'DROP SCHEMA IF EXISTS "{acct.schema}" CASCADE'))


def api_secret(acct: Account) -> str:
    """The app secret, decrypted; the primary's comes from the server's environment."""
    if acct.is_primary:
        return os.environ.get("ZERODHA_API_SECRET", "")
    return daily_login._fernet().decrypt(acct.api_secret_enc.encode()).decode()


def login_url(acct: Account) -> str:
    """Zerodha's login page for the account's app; a registered account's carries its id back to the callback."""
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
    ap = argparse.ArgumentParser(description="Connected Kite accounts (trackers FA2, MU1)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("managed", help="the ids of the accounts Centurion manages, one per line")
    sub.add_parser("remind", help="email each managed account's holder the login link when it is missing")
    sub.add_parser("mask", help="GitHub Actions: hide every account's name, id, user id and email in the "
                                "job's public log (run it before any account's output)")
    args = ap.parse_args(argv)
    if args.cmd == "managed":
        for acct in managed():
            print(acct.id)
        return 0
    if args.cmd == "mask":
        for acct in load().values():
            for value in {acct.id, acct.name, acct.zerodha_user_id, acct.email}:
                if len(value) >= 4:                       # a shorter mask would blank ordinary words
                    print(f"::add-mask::{value}")
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
