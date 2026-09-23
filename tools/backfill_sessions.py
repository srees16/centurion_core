#!/usr/bin/env python3
"""Write the session records for 16-22 Sep 2026, which ran before recording existed.

Session activity (``paper_sessions``) started on 23 Sep 2026, so the trade
monitor cannot colour or explain the earlier days. Every figure below is copied
from a primary source, not inferred:

  * "EngineExecutor plan <date>: ... sells= buys= stops= skipped= shift_multiplier="
    and "engine session=<date> ... filled= cancelled= stops= queued=" lines in the
    GitHub Actions run logs;
  * the daily snapshots already in Neon (equity, open positions);
  * for 17 Sep, whose run was deleted from Actions, the summary Neon kept in
    ``paper_trading_state.last_run_message``:
    "engine session=2026-09-17 deployment=approved filled=21 cancelled=0 stops=0
     queued=0 skipped=0 shift_mult=1.00 cash=50635".

Rebalance days come from the engine's own schedule on the deployed anchor
(2012-01-02, every 5 sessions): 2, 9 and 17 September; 14 September was an NSE
holiday. 16 September is flagged as a build, not a rebalance: the book was empty
that evening, so the executor planned the whole portfolio.

    python -m tools.backfill_sessions            # skips dates already recorded
    python -m tools.backfill_sessions --force    # overwrite them
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger("backfill_sessions")

#: One row per session, with ``ran_at`` left blank because these are reconstructed.
SESSIONS = [
    {"session_date": "2026-09-16", "equity": 3_500_000.0, "cash": 3_500_000.0,
     "open_positions": 0, "rebalance_day": False, "planned_buys": 21, "planned_sells": 0,
     "queued": 21, "filled": 0, "cancelled": 0, "stops_triggered": 0, "stops_armed": 20,
     "skipped": 0, "shift_multiplier": 1.0,
     "outcome": "new book: 21 order(s) queued for the next open",
     "notes": "backfilled from the Actions run log; the Rs 35L book was opened this evening"},
    {"session_date": "2026-09-17", "equity": 3_523_525.68, "cash": 50_634.56,
     "open_positions": 21, "rebalance_day": True, "planned_buys": 0, "planned_sells": 0,
     "queued": 0, "filled": 21, "cancelled": 0, "stops_triggered": 0, "stops_armed": 20,
     "skipped": 0, "shift_multiplier": 1.0,
     "outcome": "21 order(s) filled at the open",
     "notes": "backfilled from paper_trading_state.last_run_message; this run was deleted from Actions"},
    {"session_date": "2026-09-18", "equity": 3_577_634.0, "cash": 50_634.56,
     "open_positions": 21, "rebalance_day": False, "planned_buys": 0, "planned_sells": 0,
     "queued": 0, "filled": 0, "cancelled": 0, "stops_triggered": 0, "stops_armed": 20,
     "skipped": 0, "shift_multiplier": 1.0,
     "outcome": "held: not a rebalance day, so no orders were planned",
     "notes": "backfilled from the Actions run log"},
    {"session_date": "2026-09-21", "equity": 3_563_861.0, "cash": 50_634.56,
     "open_positions": 21, "rebalance_day": False, "planned_buys": 0, "planned_sells": 0,
     "queued": 0, "filled": 0, "cancelled": 0, "stops_triggered": 0, "stops_armed": 20,
     "skipped": 0, "shift_multiplier": 1.0,
     "outcome": "held: not a rebalance day, so no orders were planned",
     "notes": "backfilled from the Actions run log"},
    {"session_date": "2026-09-22", "equity": 3_593_655.0, "cash": 50_634.56,
     "open_positions": 21, "rebalance_day": False, "planned_buys": 0, "planned_sells": 0,
     "queued": 0, "filled": 0, "cancelled": 0, "stops_triggered": 0, "stops_armed": 20,
     "skipped": 0, "shift_multiplier": 1.0,
     "outcome": "held: not a rebalance day, so no orders were planned",
     "notes": "backfilled from the Actions run log"},
]


def backfill(force: bool = False, dry_run: bool = False) -> dict:
    from database.paper_cloud import get_paper_cloud

    cloud = get_paper_cloud()
    if cloud is None:
        raise SystemExit("no database connection (CENTURION_DATABASE_URL)")
    existing = set()
    try:
        df = cloud.read_sessions(since_epoch=False)
        if df is not None and not df.empty:
            existing = {str(d) for d in df["session_date"]}
    except Exception as exc:                              # noqa: BLE001 - first run has no table
        logger.info("no sessions table yet (%s)", exc)

    written, skipped = [], []
    for row in SESSIONS:
        date = row["session_date"]
        if date in existing and not force:
            skipped.append(date)
            continue
        if dry_run:
            written.append(date)
            continue
        if cloud.sync_session({**row, "ran_at": ""}):
            written.append(date)
        else:
            logger.warning("could not write %s", date)
    return {"written": written, "skipped_already_present": skipped, "dry_run": dry_run}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="overwrite dates already recorded")
    ap.add_argument("--dry-run", action="store_true", help="report what would be written")
    args = ap.parse_args()
    result = backfill(force=args.force, dry_run=args.dry_run)
    logger.info("backfill: %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
