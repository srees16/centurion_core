#!/usr/bin/env python3
"""Saturday: every paper book against the deployed one, the forward gate per
trial, the register attached (tracker V4).

Reads each book's Neon schema, scores it over its whole record and over the
sessions it shares with the deployed book, applies the forward gate's three
checks, and emails one report.  A trial that has cleared all three gets its
promotion review in the same mail, with READY FOR YOUR REVIEW in the subject.
Backtest columns come from the committed register
(``nse_engine.books register`` rebuilds them from the run registry, which
lives only on the research machine).  Nothing here promotes.

    python -m tools.books_report              # needs CENTURION_DATABASE_URL and the SMTP secrets
    python -m tools.books_report --dry-run    # print the subject, headline and register; send nothing

``--daily`` (tracker DS1) is the one nightly email instead: one table of every
book (initial capital, value, since start against NIFTY 50, drawdown against
its backtest, G4, the day) and the findings, then each book's own report (kept
in its Neon state by the workflow's sessions, which no longer mail them one by
one), once per session.  The backup runs skip a session already sent; a night
with a book not yet updated waits for the next run, until ``--wait-until``
(IST), then goes out marked incomplete.

    python -m tools.books_report --daily [--dry-run] [--force]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nse_engine import books as bk  # noqa: E402

logger = logging.getLogger("books_report")


def load_paper_books(books: Sequence[bk.Book]) -> Dict[str, bk.PaperBook]:
    """Each book's snapshots, sessions and stored G4 report from its Neon schema (read only)."""
    from database.connection import get_db_manager
    from database.paper_cloud import PaperCloudSync

    mgr = get_db_manager()
    out = {}
    for b in books:
        cloud = PaperCloudSync(mgr, schema=b.schema)
        out[b.name] = bk.paper_book_from_frames(b.name, cloud.read_snapshots(), cloud.read_sessions(),
                                                cloud.read_state())
        logger.info("%s: %d sessions, G4 %s", b.label, len(out[b.name].equity),
                    (out[b.name].gate or {}).get("verdict") or "no report")
    return out


def nifty50(paper: Dict[str, bk.PaperBook]) -> Optional[pd.Series]:
    """NIFTY 50 closes over the books' span, for the alpha column; None when Yahoo is unreachable."""
    from nse_engine.data.external import _yahoo_close

    starts = [p.equity.index[0] for p in paper.values() if p.started]
    if not starts:
        return None
    try:
        return _yahoo_close("^NSEI", (min(starts) - timedelta(days=7)).date().isoformat(),
                            (datetime.now(bk.IST) + timedelta(days=1)).date().isoformat())
    except Exception as exc:                              # noqa: BLE001 - alpha is reported, never required
        logger.warning("NIFTY 50 unavailable, alpha left blank: %s", exc)
        return None


def load_day_books(books: Sequence[bk.Book]) -> Dict[str, bk.DayBook]:
    """Each book's record for the daily summary from its Neon schema (read only); the session's rows are picked
    later, once the session is known (``DayBook.activity`` holds the latest one meanwhile)."""
    from database.connection import get_db_manager
    from database.paper_cloud import PaperCloudSync

    mgr = get_db_manager()
    out = {}
    for b in books:
        cloud = PaperCloudSync(mgr, schema=b.schema)
        state, sessions = cloud.read_state() or {}, cloud.read_sessions()
        paper = bk.paper_book_from_frames(b.name, cloud.read_snapshots(), sessions, state)
        initial = b.deployment.paper_capital_inr or float(state.get("initial_capital") or 0) or \
            (float(paper.equity.iloc[0]) if paper.started else 0.0)
        kind = next((k for k in bk.SLEEVE_LEDGERS if state.get(k)), None)
        history = (json.loads(state[kind]).get("history") or []) if kind else []
        latest = sessions.iloc[-1].to_dict() if sessions is not None and not sessions.empty else None
        out[b.name] = bk.DayBook(b, paper, float(initial), latest, cloud.read_fills(),
                                 {"kind": kind, **history[-1]} if history else None,
                                 json.loads(state.get(bk.DAILY_EMAIL_KEY) or "null"))
    return out


def store_nifty(books: Sequence[bk.Book]) -> Optional[pd.Series]:
    """NIFTY 50 closes from the store the nightly run built (through tonight); None when it has none."""
    from nse_engine.crash_recovery import store_index

    try:
        return store_index("NIFTY50", str(books[0].deployment.engine.data.store_dir))
    except Exception as exc:                              # noqa: BLE001 - the column is reported, never required
        logger.warning("NIFTY 50 not in the store: %s", exc)
        return None


def daily(args: argparse.Namespace, books: Sequence[bk.Book]) -> int:
    """The nightly executive summary, once per session (tracker DS1)."""
    from database.connection import get_db_manager
    from database.paper_cloud import PaperCloudSync

    days = load_day_books(books)
    dates = [str((d.activity or {}).get("session_date") or "")[:10] for d in days.values()]
    if not any(dates):
        logger.info("daily summary: no book has a session yet")
        return 0
    session = date.fromisoformat(max(dates))
    for d in days.values():
        if str((d.activity or {}).get("session_date") or "")[:10] != session.isoformat():
            d.activity = None
        f = d.fills
        d.fills = (f[(f["session_date"].astype(str).str[:10] == session.isoformat()) & (f["status"] == "FILLED")]
                   if f is not None and not f.empty else None)
    started = [d for d in days.values() if d.book.deployment.paper_start_date <= session]
    deployed = PaperCloudSync(get_db_manager(), schema=None)
    if (deployed.read_state() or {}).get(bk.DAILY_STATE_KEY) == session.isoformat() and not (args.force or args.dry_run):
        logger.info("daily summary for %s already sent", session)
        return 0
    behind = [d.book.name for d in started if d.activity is None]
    now = datetime.now(bk.IST).strftime("%H:%M")
    if behind and now < args.wait_until and not (args.force or args.dry_run):
        logger.info("daily summary for %s waits for %d book(s) not yet updated (sends anyway after %s IST)",
                    session, len(behind), args.wait_until)
        return 0
    nifty = store_nifty(books)
    if nifty is None or nifty.index[-1].date() < session:              # a local store lags: Yahoo's then
        fresh = nifty50({n: d.paper for n, d in days.items()})
        nifty = fresh if fresh is not None else nifty
    report = bk.build_daily(started, nifty, session)
    on_actions = os.getenv("GITHUB_ACTIONS") == "true"   # public logs: counts only there
    if args.dry_run or not on_actions:
        print(report["subject"])
        print(report["text"])
    if args.dry_run:
        return 0
    from services.notifications.manager import NotificationManager

    sent = NotificationManager._send_html_email(report["subject"], report["html"])
    if sent:
        deployed.sync_state({bk.DAILY_STATE_KEY: session.isoformat()})
    logger.info("daily summary for %s: %d books, %s", session, len(started),
                "sent" if sent else "NOT sent (check the email secrets)")
    return 0 if sent else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dry-run", action="store_true", help="print the subject, headline and register; send nothing")
    p.add_argument("--daily", action="store_true", help="the nightly executive summary instead (tracker DS1)")
    p.add_argument("--wait-until", default="21:30", help="--daily: IST time after which a night with a book not "
                                                         "yet updated is sent anyway, marked incomplete")
    p.add_argument("--force", action="store_true", help="--daily: send even if this session's summary went out")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    books = bk.discover_books()
    if not books:
        logger.error("no config/nse_engine_*.json files")
        return 1
    if args.daily:
        return daily(args, books)
    paper = load_paper_books(books)
    benchmark = nifty50(paper)
    today = datetime.now(bk.IST).date()
    register = bk.build_register(books, runs_dir=None, paper=paper, benchmark=benchmark,
                                 existing=bk.read_register(), today=today)
    report = bk.build_report(books, paper, register, benchmark, as_of=today)
    csv_bytes = register.to_csv(index=False, float_format="%.6g").encode()

    if args.dry_run or os.getenv("GITHUB_ACTIONS") != "true":    # Actions logs are public: counts only there
        print(report["subject"])
        print(report["headline"])
        print(report["gate_line"])
        for _, row in register.iterrows():
            print(f"  {row['book']:10s} {row['fingerprint']}  {row['summary']}")
        for name, text in report["reviews"].items():
            print(f"\n{text}")
    else:
        print(f"books report: {len(books)} books, {len(report['cleared'])} cleared")
    if args.dry_run:
        return 0

    from services.notifications.manager import NotificationManager
    sent = NotificationManager._send_html_email(report["subject"], report["html"],
                                                attachments=[("books_register.csv", csv_bytes)])
    logger.info("books report %s", "sent" if sent else "NOT sent (check the SMTP secrets)")
    return 0 if sent else 1


if __name__ == "__main__":
    raise SystemExit(main())
