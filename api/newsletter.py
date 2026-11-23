"""
The weekly newsletter (tracker NL1): the deployed paper book's week, in aggregates, for the signed-up users who
asked for it.

The owner's Saturday email (``tools.books_report``) carries every book, its trades and the operations.  This one
carries the deployed paper book's aggregates only, as percentages: the week's return, the return since the book
started against NIFTY 50's, how far it is below its highest value, its deepest fall and its sessions, with a
disclaimer.  ``figures`` is all the email is built from, and it holds numbers and dates only: never a stock, a
position, a trade, a configuration or the book's size in rupees (a test pins its fields).

Who gets it: an active signed-up user who ticked the box at sign-up or turned it on in Settings
(``app.users.newsletter``, off by default).  Each gets an email of their own, never a shared recipient list, with a
link that stops it in two steps: the website's page asks, then posts the signed ticket back
(``POST /api/v1/auth/newsletter/unsubscribe``).  The ticket carries the keyed hash of the address, not the address.

When: the Space's scheduler, Saturdays at 10:00 IST with a retry at 13:00 (``scheduler.py``).
``app.newsletter_sent`` records each reader per ISO week, so a retry sends only what has not gone out.  A paper book
that has not updated for more than ``STALE_DAYS`` days sends nothing: its "week" would be no week.

    python -m api.newsletter            # this week's figures and email, printed; nothing sent
    python -m api.newsletter --send     # send to the opted-in users not yet sent this week
"""

from __future__ import annotations

import argparse
import logging
import math
from datetime import date, datetime
from html import escape
from typing import Any, Callable, Dict, List, Optional, Sequence

from shared.clock import IST

logger = logging.getLogger(__name__)

UNSUBSCRIBE_PURPOSE = "newsletter-off"
UNSUBSCRIBE_MAX_AGE_S = 400 * 24 * 3600                   # a link in last year's email still works
STALE_DAYS = 5                                            # Saturday's send expects Friday's (or Thursday's) session
_SENT = "app.newsletter_sent"

#: The only fields the email is built from (a test pins them): numbers and dates, nothing that names a holding.
PUBLIC_FIELDS = ("asof", "started", "sessions", "week", "nifty_week", "since_start", "nifty_since_start",
                 "below_peak", "deepest_fall")

DISCLAIMER = (
    "These are paper-trading results: simulated with no real money and no orders at an exchange, after modelled "
    "trading charges and before income tax. They are not investment advice, a research report or a recommendation "
    "to buy or sell any security. Past or simulated performance does not indicate future results. Centurion is not "
    "registered with SEBI as an investment adviser or a research analyst. Investments in the securities market are "
    "subject to market risks.")


# ── The figures (aggregates only) ─────────────────────────────────────────

def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def figures(equity, initial: float, nifty=None) -> Optional[Dict[str, Any]]:
    """The newsletter's numbers from the book's value by session (``initial``: its capital before the first one),
    computed as the owner's Saturday email does (``nse_engine.books``); None before the book has two sessions.
    NIFTY 50's figures are None when its closes are missing (Yahoo unreachable)."""
    import pandas as pd

    from nse_engine import books as bk

    e = pd.Series(equity, dtype="float64").dropna().sort_index()
    if len(e) < 2 or not initial > 0:
        return None
    last = e.index[-1]
    n = (pd.Series(nifty, dtype="float64").dropna().sort_index() if nifty is not None    # no NIFTY: an empty series
         else pd.Series(dtype="float64", index=pd.DatetimeIndex([])))                  # that dates still compare with

    def close(day) -> float:
        upto = n[n.index <= day]
        return float(upto.iloc[-1]) if len(upto) else math.nan

    def change(a: float, b: float) -> Optional[float]:
        return b / a - 1.0 if _finite(a) and _finite(b) and a > 0 else None

    return {"asof": last.date(), "started": e.index[0].date(), "sessions": int(len(e)),
            "week": float(bk._week_return(e, initial, last.date())),
            "nifty_week": change(close(last - pd.Timedelta(days=7)), close(last)),
            "since_start": float(e.iloc[-1] / initial - 1.0),
            "nifty_since_start": change(close(e.index[0]), close(last)),
            "below_peak": float(1.0 - e.iloc[-1] / e.max()),
            "deepest_fall": float(-(e / e.cummax() - 1.0).min())}


def load() -> Optional[Dict[str, Any]]:
    """This week's figures from the deployed book's Neon record and NIFTY 50 (read as the owner's Saturday email
    reads them, ``tools.books_report``); None when the book has not started."""
    from nse_engine import books as bk
    from tools import books_report

    book = next((b for b in bk.discover_books() if b.is_deployed), None)
    if book is None:
        raise RuntimeError("no deployed book (config/nse_engine_deployed.json)")
    day = books_report.load_day_books([book])[book.name]
    return figures(day.paper.equity, day.initial, books_report.nifty50({book.name: day.paper}))


# ── The email ─────────────────────────────────────────────────────────────

def _pct(x: Optional[float]) -> str:
    return f"{x:+.2%}" if _finite(x) else "n/a"


def _moved(x: float) -> str:
    return "was unchanged" if abs(x) < 0.00005 else f"{'rose' if x > 0 else 'fell'} {abs(x):.2%}"


def _updown(x: float) -> str:
    return "unchanged" if abs(x) < 0.00005 else f"{'up' if x > 0 else 'down'} {abs(x):.2%}"


def _day(d: date) -> str:
    return f"{d:%a} {d.day} {d:%b %Y}"


def summary(f: Dict[str, Any]) -> str:
    """The week in two plain sentences."""
    line = f"In the week to {_day(f['asof'])}, Centurion's paper portfolio {_moved(f['week'])}"
    line += f"; NIFTY 50 {_moved(f['nifty_week'])}." if _finite(f["nifty_week"]) else "."
    line += (f" Since it started on {_day(f['started'])} ({f['sessions']} trading sessions) it is "
             f"{_updown(f['since_start'])}")
    return line + (f", against {_pct(f['nifty_since_start'])} for NIFTY 50 over the same sessions."
                   if _finite(f["nifty_since_start"]) else ".")


def render(f: Dict[str, Any], name: str, unsubscribe_url: str) -> Dict[str, str]:
    """The subject and HTML of one reader's email, from ``figures`` only."""
    cell = "padding:8px 10px;border-bottom:1px solid #e5e7eb;"
    rows = [("This week", _pct(f["week"]), _pct(f["nifty_week"])),
            (f"Since the start ({_day(f['started'])})", _pct(f["since_start"]), _pct(f["nifty_since_start"])),
            ("Below its highest value", f"{f['below_peak']:.2%}" if f["below_peak"] >= 0.00005 else "at its highest", ""),
            ("Deepest fall since the start", f"{f['deepest_fall']:.2%}", "")]
    table = ("<table style='border-collapse:collapse;width:100%;font-size:14px;margin:16px 0;'>"
             f"<tr><th style='{cell}text-align:left;'></th><th style='{cell}text-align:right;'>Paper portfolio</th>"
             f"<th style='{cell}text-align:right;'>NIFTY 50</th></tr>"
             + "".join(f"<tr><td style='{cell}'>{escape(a)}</td><td style='{cell}text-align:right;'>{escape(b)}</td>"
                       f"<td style='{cell}text-align:right;'>{escape(c)}</td></tr>" for a, b, c in rows) + "</table>")
    html = ("<html><body style='font-family:Segoe UI,Arial,sans-serif;background:#f9fafb;padding:20px;'>"
            "<div style='max-width:560px;margin:auto;background:#fff;border-radius:10px;padding:24px;'>"
            "<h2 style='margin-top:0;'>Centurion weekly</h2>"
            f"<p>Hello {escape(name)},</p><p>{escape(summary(f))}</p>{table}"
            "<p style='font-size:13px;'>The paper portfolio is Centurion's strategy for NSE stocks run every trading "
            "day with simulated money, at the market's closing prices.</p>"
            f"<p style='color:#666;font-size:12px;background:#f3f4f6;padding:10px;border-radius:6px;'>{escape(DISCLAIMER)}</p>"
            "<p style='color:#666;font-size:12px;'>You receive this because you chose Centurion's weekly newsletter. "
            f"<a href='{escape(unsubscribe_url)}'>Stop the newsletter</a>, or turn it off in Settings.</p>"
            "</div></body></html>")
    return {"subject": f"Centurion weekly: the paper portfolio in the week to {_day(f['asof'])}", "html": html}


# ── The unsubscribe link ──────────────────────────────────────────────────

def unsubscribe_link(email: str) -> str:
    """The website's page that asks before it stops the newsletter, with a signed ticket of the address's keyed
    hash (``users.email_index``): the address itself never travels in a URL."""
    from urllib.parse import quote

    from api import users
    from api.auth import sign_ticket

    ticket = sign_ticket(UNSUBSCRIBE_PURPOSE, {"h": users.email_index(email)})
    return f"{users.frontend_url()}/newsletter/unsubscribe?token={quote(ticket)}"


def read_unsubscribe(token: str) -> str:
    """The email index an unsubscribe ticket carries (raises ``users.UserError`` when invalid or expired)."""
    from api import users
    from api.auth import read_ticket

    data = read_ticket(token, UNSUBSCRIBE_PURPOSE, UNSUBSCRIBE_MAX_AGE_S)
    if not isinstance(data, dict) or len(str(data.get("h") or "")) != 64:
        raise users.UserError("This link is invalid or has expired: turn the newsletter off in Settings")
    return str(data["h"])


# ── Sending ───────────────────────────────────────────────────────────────

def week_key(day: date) -> str:
    iso = day.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def _ensure_sent() -> None:
    from sqlalchemy import text

    from api import users

    users._ensure()                                       # the app schema and the users' newsletter column
    with users._db().get_session() as session:
        session.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {_SENT} (
                week       VARCHAR(8)  NOT NULL,
                email_hash CHAR(64)    NOT NULL,
                sent_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (week, email_hash)
            )"""))


def _sent(week: str) -> set:
    """The email indexes this week's newsletter already reached."""
    from sqlalchemy import text

    from api import users

    _ensure_sent()
    with users._db().get_session() as session:
        rows = session.execute(text(f"SELECT email_hash FROM {_SENT} WHERE week = :w"), {"w": week}).fetchall()
    return {str(r[0]) for r in rows}


def _mark(week: str, index: str) -> None:
    from sqlalchemy import text

    from api import users

    with users._db().get_session() as session:
        session.execute(text(f"INSERT INTO {_SENT} (week, email_hash) VALUES (:w, :h) ON CONFLICT DO NOTHING"),
                        {"w": week, "h": index})


def _send(subject: str, html: str, email: str) -> bool:
    from services.notifications.manager import NotificationManager

    return bool(NotificationManager._send_html_email(subject, html, [email]))


def readers(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The users who get the newsletter: active, and opted in."""
    return [u for u in rows if u.get("status") == "active" and u.get("newsletter") is True]


def send_week(now: Optional[datetime] = None, send: Optional[Callable[[str, str, str], bool]] = None) -> Dict[str, Any]:
    """Send this ISO week's newsletter to each reader it has not reached yet; a failed send is retried by the next
    run.  Returns the counts, and ``skipped`` with the reason when nothing could be sent."""
    from api import users

    now = (now or datetime.now(IST)).astimezone(IST)
    week, send = week_key(now.date()), send or _send
    everyone = readers(users.list_users())
    done = _sent(week) if everyone else set()
    todo = [u for u in everyone if users.email_index(u["email"]) not in done]
    out: Dict[str, Any] = {"week": week, "readers": len(everyone), "already": len(everyone) - len(todo),
                           "sent": 0, "failed": 0}
    if not todo:
        return out
    f = load()
    if f is None:
        return {**out, "skipped": "the paper book has not started"}
    if (now.date() - f["asof"]).days > STALE_DAYS:
        return {**out, "skipped": f"the paper book's last session is {f['asof']}: stale, nothing sent"}
    for u in todo:
        email = render(f, u.get("first_name") or u.get("full_name") or "there", unsubscribe_link(u["email"]))
        try:
            ok = send(email["subject"], email["html"], u["email"])
        except Exception as exc:                          # noqa: BLE001 - one reader's failure never stops the rest
            logger.warning("Newsletter to user %s failed: %s", u.get("id"), exc)
            ok = False
        if ok:
            _mark(week, users.email_index(u["email"]))
        out["sent" if ok else "failed"] += 1
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--send", action="store_true", help="send to the opted-in users not yet sent this week")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import config  # noqa: F401 - loads .env (the database URL) on a laptop

    if args.send:
        out = send_week()
        print(out)
        return 1 if out["failed"] else 0
    f = load()
    if f is None:
        print("The paper book has not started: no newsletter")
        return 0
    print(summary(f))
    print({k: f[k] for k in PUBLIC_FIELDS})
    print(render(f, "Reader", "https://example.org/newsletter/unsubscribe?token=...")["subject"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
