#!/usr/bin/env python3
"""Computed facts for the start of a session (tracker PL0, PL-04).

Answers and documents went wrong by starting from a remembered fact: an audit built on a gap register a
month old, "14 Dec" written for a verdict that falls on 15 Dec, a manual run recommended on a Saturday when
the nightly runner only does its weekly checkpoint.  This prints what can be computed instead, so a session
starts from it: the clock in IST and UTC, the last and next NSE sessions, what a nightly run started now
would do, the deployed book's gate sessions and the date of its 60th, the trial counts per family, what the
journal and the lessons ledger are missing, when the store guards last ran (``tools.guards``), when the cost
rates were last read against the published schedule (``nse_engine.validation.rates``), the owner's open items
and both repositories' uncommitted files.
Read-only.  The database is optional: without it the lines that need it say so.

    python -m tools.preflight            # everything
    python -m tools.preflight --offline  # skip the database
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.clock import IST  # noqa: E402 - after the path bootstrap

#: RA1 reads the deployed book at this many gate sessions (plan 5ao).
RA1_SESSIONS = 60


def clock(now: Optional[datetime] = None) -> Dict[str, Any]:
    """The moment in IST and UTC, and the NSE sessions around it."""
    from services.execution.carver_pipeline import is_nse_trading_day, last_completed_nse_session

    now = (now or datetime.now(IST)).astimezone(IST)
    nxt = now.date() if is_nse_trading_day(now.date()) and last_completed_nse_session(now) != now.date() \
        else now.date() + timedelta(days=1)
    while not is_nse_trading_day(nxt):
        nxt += timedelta(days=1)
    return {"ist": now.strftime("%a %d %b %Y %H:%M IST"), "utc": now.astimezone(timezone.utc).strftime("%a %d %b %H:%M UTC"),
            "trading_day": bool(is_nse_trading_day(now.date())),
            "last_completed_session": last_completed_nse_session(now).isoformat(), "next_session": nxt.isoformat()}


def runner_mode(now: Optional[datetime] = None) -> str:
    """What ``cloud_paper_runner`` would do if a nightly run started now.  The runner reads the weekday from
    its own clock, which on GitHub is UTC: on a UTC Saturday it runs the weekly checkpoint and no session
    (FL3, 14 Nov 2026: a Saturday run started to test a fix tested nothing)."""
    now = (now or datetime.now(IST)).astimezone(timezone.utc)
    if now.weekday() == 5:
        return "weekly checkpoint only (a UTC Saturday): no engine session, no sleeve or futures leg"
    return "a nightly session for the last completed NSE session"


def session_date(done: int, after: date, target: int = RA1_SESSIONS) -> Optional[date]:
    """The date of the ``target``-th session, given ``done`` sessions up to and including ``after``."""
    from services.execution.carver_pipeline import is_nse_trading_day

    if done >= target:
        return None
    day, n = after, done
    while n < target:
        day += timedelta(days=1)
        if is_nse_trading_day(day):
            n += 1
    return day


def owner_items(tracker: str, limit: int = 6) -> List[str]:
    """The first open headings of the tracker's NEEDS YOU section (those not marked done or superseded)."""
    lines = tracker.split("\n")
    start = next((i for i, line in enumerate(lines) if line.startswith(" 4. NEEDS YOU")), None)
    if start is None:
        return []
    out = []
    for line in lines[start + 1:]:
        if re.match(r"^ \d+\. [A-Z]", line):
            break
        if line.startswith(" >>> ") and not re.search(r"\(was\)|DONE|ON HOLD|- YOUR ACTIONS", line):
            out.append(line[5:].strip()[:100])
    return out[:limit]


def git_state(repo: Path) -> str:
    try:
        branch = subprocess.run(["git", "-C", str(repo), "branch", "--show-current"], capture_output=True, text=True,
                                timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True,
                               timeout=10).stdout.splitlines()
        return f"{branch or '?'}, {len(dirty)} uncommitted file(s)"
    except Exception as exc:                              # noqa: BLE001 - a fact that cannot be read says so
        return f"unreadable ({type(exc).__name__})"


def deployed_sessions() -> Dict[str, Any]:
    """The deployed paper book's gate sessions from Neon: snapshots less one (a return needs two)."""
    from database.paper_cloud import get_paper_cloud

    cloud = get_paper_cloud()
    if cloud is None:
        raise RuntimeError("no database configured")
    snaps = cloud.read_snapshots()
    days = sorted(str(d) for d in snaps["date"])
    return {"gate_sessions": max(len(days) - 1, 0), "last": date.fromisoformat(days[-1][:10])}


# Each section below returns its lines.  A section appends as it goes, so a failure part-way keeps the lines
# already computed and adds one saying what could not be read.

def _clock_lines(now: Optional[datetime]) -> List[str]:
    c = clock(now)
    return [f"Now: {c['ist']} ({c['utc']}); NSE trading day today: {'yes' if c['trading_day'] else 'no'}",
            f"Sessions: last completed {c['last_completed_session']}, next {c['next_session']}",
            f"A nightly run started now: {runner_mode(now)}"]


def _deployed_line(offline: bool) -> str:
    if offline:
        return "Deployed book and RA1: not read (offline)"
    try:
        d = deployed_sessions()
        when = session_date(d["gate_sessions"], d["last"])
        return (f"Deployed book: {d['gate_sessions']} gate sessions to {d['last']}; RA1 at {RA1_SESSIONS}: "
                + (f"{when:%a %d %b %Y}, if no session is missed (the freeze holds until then)" if when
                   else "reached"))
    except Exception as exc:                              # noqa: BLE001
        return f"Deployed book and RA1: not read ({type(exc).__name__})"


def _trial_lines(root: Path) -> List[str]:
    from nse_engine.validation import families

    out: List[str] = []
    try:
        fam = families.load(root)
        out.append("Trials: " + ", ".join(f"{k} {v['n']}" for k, v in fam.items())
                   + f"; core PBO {fam['core']['pbo']:.1%} of {fam['core']['pbo_cap']:.0%}")
        out += [f"  ! {p}" for p in families.problems(root, registries=False)]
    except Exception as exc:                              # noqa: BLE001
        out.append(f"Trials: not read ({type(exc).__name__}: {exc})")
    return out


def _journal_lines(root: Path) -> List[str]:
    from nse_engine import journal

    out: List[str] = []
    try:
        tracker = (root / journal.TRACKER_PATH).read_text()
        found = journal.check(tracker)
        out.append(f"Journal: {len(found['unjournaled'])} verdict(s) not journaled"
                   + (f" ({', '.join(e['id'] for e in found['unjournaled'][:8])})" if found["unjournaled"] else "")
                   + f", {len(found['no_postmortem'])} row(s) without a post-mortem")
        items = owner_items(tracker)
        out.append(f"Owner's open items ({len(items)} shown): " + ("; ".join(items) or "none"))
    except Exception as exc:                              # noqa: BLE001
        out.append(f"Journal and tracker: not read ({type(exc).__name__}: {exc})")
    return out


def _lesson_lines(root: Path) -> List[str]:
    from tools import lessons

    out: List[str] = []
    try:
        rows = lessons.build(root)
        s = lessons.summary(rows)
        out.append(f"Lessons: {s['lessons']}, {s['guarded']} guarded, {s['open']} open ("
                   + ", ".join(f"{r['id']} by {r.get('review_by')}" for r in rows if r["status"] == "open") + ")")
        out += [f"  ! {p}" for p in lessons.check(root)]
    except Exception as exc:                              # noqa: BLE001
        out.append(f"Lessons: not read ({type(exc).__name__}: {exc})")
    return out


def _operations_lines(root: Path, now: Optional[datetime], offline: bool) -> List[str]:
    """Store guards, the data catalogue, the week's operations (PL-16, PL-17) and the cost rates."""
    out: List[str] = []
    try:
        from nse_engine.validation import rates
        from tools import guards

        today = (now or datetime.now(IST)).astimezone(IST).date()
        out.append(f"Store guards: {guards.status(root, today)}")
        from tools import data_catalogue

        out.append(f"Data: {data_catalogue.status(root, today)}")
        from tools import ops_report

        ops = ops_report.block(root, now, offline=offline)
        out.append(ops["head"])
        out += [f"  ! {line}" for line in ops["lines"]]
        out.append(f"Cost rates: {rates.status(today)}")
    except Exception as exc:                              # noqa: BLE001
        out.append(f"Store guards and cost rates: not read ({type(exc).__name__}: {exc})")
    return out


def facts(now: Optional[datetime] = None, offline: bool = False, root: Path = _ROOT) -> List[str]:
    """The lines of the preflight.  Nothing here raises: a fact that cannot be computed says why."""
    out = (_clock_lines(now) + [_deployed_line(offline)] + _trial_lines(root) + _journal_lines(root)
           + _lesson_lines(root) + _operations_lines(root, now, offline))
    return out + [f"Repositories: centurion_core {git_state(root)}; centurion_core-fe "
                  + git_state(root.parent / "centurion_core-fe")]


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--offline", action="store_true", help="skip the database")
    args = p.parse_args(argv)
    print("\n".join(facts(offline=args.offline)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
