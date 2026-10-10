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
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dry-run", action="store_true", help="print the subject, headline and register; send nothing")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    books = bk.discover_books()
    if not books:
        logger.error("no config/nse_engine_*.json files")
        return 1
    paper = load_paper_books(books)
    benchmark = nifty50(paper)
    today = datetime.now(bk.IST).date()
    register = bk.build_register(books, runs_dir=None, paper=paper, benchmark=benchmark,
                                 existing=bk.read_register(), today=today)
    report = bk.build_report(books, paper, register, benchmark, as_of=today)
    csv_bytes = register.to_csv(index=False, float_format="%.6g").encode()

    print(report["subject"])
    print(report["headline"])
    print(report["gate_line"])
    for _, row in register.iterrows():
        print(f"  {row['book']:10s} {row['fingerprint']}  {row['summary']}")
    for name, text in report["reviews"].items():
        print(f"\n{text}")
    if args.dry_run:
        return 0

    from services.notifications.manager import NotificationManager
    sent = NotificationManager._send_html_email(report["subject"], report["html"],
                                                attachments=[("books_register.csv", csv_bytes)])
    logger.info("books report %s", "sent" if sent else "NOT sent (check the SMTP secrets)")
    return 0 if sent else 1


if __name__ == "__main__":
    raise SystemExit(main())
