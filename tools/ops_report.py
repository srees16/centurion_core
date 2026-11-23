#!/usr/bin/env python3
"""The week's operations in one block: incidents, each against the lesson that guards it, and the loop's health
(tracker PL-16, PL-17).

Incidents were noticed only when someone read the right email.  This lists the week's:

- failed production workflow runs (GitHub's run history; ``tests.yml`` and branches other than main are not
  production);
- alerts sent (each book's ``alerts_sent`` history in Neon, kept 7 days), by key;
- sessions caught up or replayed after a missed night, and futures legs deferred (each book's session notes).

Each incident type names the lessons that guard it (``INCIDENT_LESSONS``); a type with none is flagged, so the
owner decides whether it needs a guard.  Risk events (G4 FAIL, the drawdown rule, the live risk checks) are the
strategy's signals, not operational faults: they are counted, not flagged.

The loop's health (section 4 of docs/process_loop_plan.md), from the repository's own records: lessons guarded and
due for review, recurrences, verdicts journaled, post-mortem hours and trials, trial counts.  Cost-model error waits
for PL-14's contract notes.

Only what needs action is printed; a quiet week is one line.  The Saturday email carries the block
(``tools.books_report``); the preflight shows its first line.  Reading only: nothing here writes.

    python -m tools.ops_report             # GitHub and Neon too (needs the network and CENTURION_DATABASE_URL)
    python -m tools.ops_report --offline   # the loop's health only
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import date, datetime, timedelta
from html import escape
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.clock import IST  # noqa: E402 - after the path bootstrap
from shared.settings import REPOSITORY  # noqa: E402

WINDOW_DAYS = 7
#: Open lessons whose review date falls within this many days are named: past it, the test suite fails.
DUE_SOON_DAYS = 30
#: Fewer post-mortem rows than this and hours, trials and Phase 0 stops are not judged.
MIN_VERDICTS = 5
GUARDED_TARGET = 0.9
NOT_PRODUCTION = ("tests.yml",)
FAILED = ("failure", "timed_out", "startup_failure")

#: Incident type -> the lessons that guard it.  An empty tuple: no lesson yet, flagged in the report.
INCIDENT_LESSONS: Dict[str, tuple] = {
    "heartbeat_stale": ("L-HB2",), "nightly_rescued": ("L-RS1",), "nightly_rescue_failed": ("L-RS1",),
    "nse_dispatch_failed": ("L-RS1",), "live_no_login": ("L-LNT33",), "futures_overlay_failed": ("L-FL1", "L-FL2"),
    "futures_deferred": ("L-FL1",), "caught_up": ("L-NEO6",), "replayed": ("L-NEO6",), "postback_failed": ("L-AL5",),
    "session_failed": (), "live_session_failed": (), "bond_sleeve_failed": (), "mcx_sleeve_failed": (),
    "options_overlay_failed": (), "mcx_sleeve_data": (), "options_overlay_data": (), "data_gap": (),
    "holiday_list": (),
}
#: A failed run of these workflows is guarded by: the heartbeat ends red when it raises its alarm (HB2); a deploy
#: failed on a binary file (B6) or on package drift (LOCK).  Any other workflow's failure has no lesson.
RUN_LESSONS: Dict[str, tuple] = {"paper-heartbeat.yml": ("L-HB2",), "deploy.yml": ("L-B6", "L-LOCK")}
#: Alert keys that are the strategy's signals, not operational faults.
RISK_EVENTS = ("paper_g4_fail", "drawdown_rule", "live_risk", "live_buys", "postback_stop")
LABELS = {
    "heartbeat_stale": "heartbeat: a paper book did not trade or could not be read",
    "nightly_rescued": "the nightly run was started by the rescue trigger",
    "nightly_rescue_failed": "the rescue trigger could not start the nightly run",
    "nse_dispatch_failed": "the scheduler could not start the nightly run",
    "live_no_login": "no Kite login: the live session was skipped",
    "futures_overlay_failed": "the futures leg failed", "futures_deferred": "the futures leg was deferred",
    "caught_up": "sessions caught up after a missed night", "replayed": "missed sessions replayed",
    "postback_failed": "an engine order was rejected or cancelled at Kite",
    "session_failed": "a nightly session failed", "live_session_failed": "the live session failed",
    "bond_sleeve_failed": "the gilt sleeve failed", "mcx_sleeve_failed": "the MCX sleeve failed",
    "options_overlay_failed": "the options overlay failed", "mcx_sleeve_data": "MCX data stale or revised",
    "options_overlay_data": "F&O data stale or revised", "data_gap": "an index or sleeve close was missing",
    "holiday_list": "NSE's holiday list is missing",
}


# ── what happened ─────────────────────────────────────────────────

def github_runs(since: date, token: str = "", repo: str = "", opener: Optional[Callable] = None) -> List[Dict[str, Any]]:
    """Workflow runs created since ``since``: {workflow, branch, event, conclusion, created, url}.  Raises when
    GitHub cannot be read (the caller reports it)."""
    import urllib.request

    repo = repo or os.environ.get("GITHUB_REPOSITORY") or REPOSITORY
    if opener is None:                                    # certifi's roots: a python.org build on a Mac has none
        try:
            import ssl

            import certifi

            context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            context = None
        opener = lambda req, timeout: urllib.request.urlopen(req, timeout=timeout, context=context)  # noqa: E731
    out: List[Dict[str, Any]] = []
    for page in (1, 2, 3):
        req = urllib.request.Request(
            f"https://api.github.com/repos/{repo}/actions/runs?created=%3E%3D{since.isoformat()}&per_page=100&page={page}",
            headers={"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
                     **({"Authorization": f"Bearer {token}"} if token else {})})
        with opener(req, timeout=30) as resp:
            runs = json.loads(resp.read() or b"{}").get("workflow_runs") or []
        out += [{"workflow": str(r.get("path") or "").rsplit("/", 1)[-1], "branch": r.get("head_branch"),
                 "event": r.get("event"), "conclusion": r.get("conclusion"), "created": r.get("created_at"),
                 "url": r.get("html_url")} for r in runs]
        if len(runs) < 100:
            break
    return out


def neon_books(books: Optional[Sequence[Any]] = None) -> Dict[str, Dict[str, Any]]:
    """Each paper book's alert history and session notes from its Neon schema: {name: {alerts, sessions}}."""
    from database.connection import get_db_manager
    from database.paper_cloud import PaperCloudSync
    from nse_engine import books as bk
    from services.notifications.alerts import STATE_KEY

    out = {}
    mgr = get_db_manager()
    for b in books if books is not None else bk.discover_books():
        cloud = PaperCloudSync(mgr, schema=b.schema)
        sessions = cloud.read_sessions(strict=True)
        out[b.name] = {"alerts": json.loads((cloud.read_state() or {}).get(STATE_KEY) or "{}"),
                       "sessions": [] if sessions is None or sessions.empty else
                       sessions[["session_date", "notes"]].to_dict("records")}
    return out


def incidents(runs: Optional[Iterable[Dict[str, Any]]], books: Optional[Dict[str, Dict[str, Any]]], since: date
              ) -> Dict[str, Any]:
    """The week's incidents and risk events from GitHub's runs and the books' Neon records (None: unreadable)."""
    found: List[Dict[str, Any]] = []
    risk: List[Dict[str, Any]] = []
    for r in runs or []:
        if r["workflow"] in NOT_PRODUCTION or r["branch"] not in ("main", None) or r["conclusion"] not in FAILED:
            continue
        day = datetime.fromisoformat(str(r["created"]).replace("Z", "+00:00")).astimezone(IST).date().isoformat()
        found.append({"type": f"run_failed:{r['workflow']}", "day": day, "where": r["workflow"], "url": r["url"]})
    for name, rec in (books or {}).items():
        for key, day in (rec.get("alerts") or {}).items():
            if str(day) < since.isoformat():
                continue
            kind = str(key).split(":", 1)[0]
            (risk if kind in RISK_EVENTS else found).append({"type": kind, "day": str(day), "where": name})
        for s in rec.get("sessions") or []:
            day, notes = str(s.get("session_date") or ""), str(s.get("notes") or "")
            if day < since.isoformat():
                continue
            for marker, kind in (("CAUGHT UP", "caught_up"), ("REPLAYED", "replayed"), ("futures deferred", "futures_deferred")):
                if marker in notes:
                    found.append({"type": kind, "day": day, "where": name})
    return {"incidents": found, "risk_events": risk, "since": since.isoformat()}


# ── how the loop is doing ─────────────────────────────────────────

def loop_health(root: Path = _ROOT, today: Optional[date] = None) -> List[Dict[str, Any]]:
    """Section 4's metrics that the records can give: [{text, action}]; ``action`` marks what needs doing."""
    from nse_engine import journal
    from nse_engine.validation import families
    from tools import lessons

    today = today or datetime.now(IST).date()
    root = Path(root)
    out: List[Dict[str, Any]] = []
    rows = lessons.build(root, today)
    s = lessons.summary(rows)
    out.append({"text": f"lessons: {s['guarded']} of {s['lessons']} guarded ({s['share_guarded']:.0%}, target "
                        f"{GUARDED_TARGET:.0%})", "action": False})
    due = sorted((r["review_by"], r["id"]) for r in rows if r["status"] == "open" and r.get("review_by")
                 and r["review_by"] <= (today + timedelta(days=DUE_SOON_DAYS)).isoformat())
    if due:
        days = sorted({d for d, _ in due})
        out.append({"text": f"{len(due)} open lesson(s) reach their review by {date.fromisoformat(days[-1]):%d %b} "
                            f"({', '.join(i for _, i in due)}): guard, or waive on a dated decision, or the test "
                            "suite fails", "action": True})
    since = (today - timedelta(days=WINDOW_DAYS)).isoformat()
    again = [(lid, d) for lid, dates in lessons.recurrences(root, today).items() for d in dates if d >= since]
    if again:
        out.append({"text": f"recurrence of a guarded lesson this week: {', '.join(f'{lid} ({d})' for lid, d in again)}",
                    "action": True})
    tracker = (root / journal.TRACKER_PATH).read_text()
    gaps = journal.check(tracker, today=today)
    if gaps["unjournaled"] or gaps["no_postmortem"]:
        out.append({"text": f"journal: {len(gaps['unjournaled'])} verdict(s) not journaled, "
                            f"{len(gaps['no_postmortem'])} row(s) without a post-mortem", "action": True})
    pm = journal.read_journal(root / journal.JOURNAL_PATH)
    pm = pm[(pm["kind"] == journal.EXPERIMENT) & (pm["date"].astype(str) >= journal.POSTMORTEM_SINCE)]
    if len(pm) < MIN_VERDICTS:
        out.append({"text": f"post-mortems: {len(pm)} verdict(s) since {journal.POSTMORTEM_SINCE}, too few to judge "
                            "hours, trials and Phase 0 stops", "action": False})
    else:
        def med(col):
            vals = [float(v) for v in pm[col] if str(v).strip() not in ("", "nan")]
            return statistics.median(vals) if vals else float("nan")
        phase0 = sum(str(v).startswith("0") for v in pm["phase_stopped"])
        out.append({"text": f"post-mortems: {len(pm)} verdicts, median {med('hours'):g} h and {med('trials_used'):g} "
                            f"trial(s); {phase0} stopped at Phase 0", "action": False})
    for problem in families.problems(root, registries=False):
        out.append({"text": f"trial counts: {problem}", "action": True})
    fam = families.load(root)
    core = fam.get("core") or {}
    if core.get("pbo") is not None and core.get("pbo_cap") is not None and core["pbo"] > core["pbo_cap"]:
        out.append({"text": f"core PBO {core['pbo']:.1%} is over its cap of {core['pbo_cap']:.0%}", "action": True})
    out.append({"text": "cost-model error against contract notes: waits for PL-14 (after the sell test)",
                "action": False})
    return out


# ── the block ─────────────────────────────────────────────────────

def lessons_for(kind: str) -> tuple:
    """The lessons that guard an incident type (a failed run: by its workflow)."""
    if kind.startswith("run_failed:"):
        return RUN_LESSONS.get(kind.split(":", 1)[1], ())
    return INCIDENT_LESSONS.get(kind, ())


def label(kind: str) -> str:
    if kind.startswith("run_failed:"):
        return f"a {kind.split(':', 1)[1]} run failed"
    return LABELS.get(kind, kind)


def summarise(found: Dict[str, Any], health: List[Dict[str, Any]], sources: Dict[str, str]) -> Dict[str, Any]:
    """{lines, html, action}: what needs action, or one line for a quiet week.  ``sources`` names a source that
    could not be read ({source: why})."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for i in found.get("incidents") or []:
        groups.setdefault(i["type"], []).append(i)
    lines: List[str] = []
    guarded: List[str] = []
    for kind, items in sorted(groups.items()):
        lessons = lessons_for(kind)
        days = sorted({i["day"] for i in items})
        if lessons:                                       # known and guarded: counted on one line, no action
            guarded.append(f"{label(kind)} {len(items)}x ({', '.join(lessons)})")
            continue
        where = sorted({i["where"] for i in items})
        lines.append(f"{label(kind)}: {len(items)}x ({', '.join(days)}; {', '.join(where)}): NO LESSON: decide "
                     "whether it needs a guard")
    for source, why in sources.items():
        lines.append(f"{source} could not be read ({why}): its incidents are not in this list")
    lines += [h["text"] for h in health if h["action"]]
    risk = found.get("risk_events") or []
    n = len(found.get("incidents") or [])
    week = f"Operations, week from {found.get('since')}"
    risk_note = f"; {len(risk)} risk event(s), in the books' reports" if risk else ""
    if not lines:
        head = (f"{week}: {n} incident(s), all guarded; nothing needs action{risk_note}" if n
                else f"{week}: no incidents; nothing in the loop needs action{risk_note}")
    else:
        head = f"{week}: {len(lines)} item(s) need action; {n} incident(s){risk_note}"
    info = [f"guarded, no action: {'; '.join(guarded)}"] if guarded else []
    html = (f"<h3 style='color:#1a1a2e;margin:22px 0 8px;'>Operations</h3><p style='margin:0 0 6px;'>{escape(head)}</p>"
            + ("<ul style='font-size:13px;color:#1f2937;padding-left:20px;margin:0;'>"
               + "".join(f"<li style='margin-bottom:4px;'>{escape(item)}</li>" for item in lines) + "</ul>" if lines else "")
            + "".join(f"<p style='margin:6px 0 0;font-size:12px;color:#6b7280;'>{escape(item)}</p>" for item in info))
    return {"head": head, "lines": lines, "info": info, "html": html, "action": len(lines)}


def block(root: Path = _ROOT, now: Optional[datetime] = None, offline: bool = False, books: Optional[Sequence[Any]] = None,
          token: str = "") -> Dict[str, Any]:
    """The Operations block for the week to ``now``.  Never raises: a source that cannot be read is named."""
    now = (now or datetime.now(IST)).astimezone(IST)
    since = now.date() - timedelta(days=WINDOW_DAYS)
    sources: Dict[str, str] = {}
    runs = recs = None
    if not offline:
        try:
            runs = github_runs(since, token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or "")
        except Exception as exc:                          # noqa: BLE001 - named in the block
            sources["GitHub's run history"] = type(exc).__name__
        try:
            recs = neon_books(books)
        except Exception as exc:                          # noqa: BLE001 - named in the block
            sources["Neon"] = type(exc).__name__
    try:
        health = loop_health(root, now.date())
    except Exception as exc:                              # noqa: BLE001 - named in the block
        health = [{"text": f"loop health could not be computed ({type(exc).__name__}: {exc})"[:200], "action": True}]
    out = summarise(incidents(runs, recs, since), health, sources)
    if offline:
        out["head"] = out["head"].replace("Operations,", "Operations (records only; GitHub and Neon not read),")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--offline", action="store_true", help="the loop's health only: no GitHub or Neon")
    args = p.parse_args(argv)
    out = block(offline=args.offline)
    print(out["head"])
    print("\n".join([f"  - {line}" for line in out["lines"]] + [f"  {line}" for line in out["info"]]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
