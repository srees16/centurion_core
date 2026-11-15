#!/usr/bin/env python3
"""The lessons ledger, generated from the guards (tracker PL0, PL-02).

A lesson counts as captured only when something executable holds it: a test, a pre-flight check, a guard in
the code or a required template field.  So the ledger is not written by hand.  It is built from:

- **tests**: a guard test (or class, or a module's ``pytestmark``) carries
  ``@pytest.mark.lesson("L-FL1", pattern="F1", date="2026-11-14", source="FL1", summary="...")``.
  A second test of the same lesson needs only ``@pytest.mark.lesson("L-FL1")``;
- **other guards** (``OTHER`` below): a pre-flight check, a guard in the code, a template field or a prior,
  each with the place it lives, which must exist;
- **open lessons** (``OPEN`` below): known, not yet guarded.  Each has a ``review_by`` date; past it the
  check fails until the lesson is guarded or waived on the owner's dated decision.

``docs/lessons.jsonl`` holds one line per lesson: id, date, source (tracker id), pattern, summary,
guard_type, guard_location (a list), status (guarded | open | waived), recurrence_count, last_seen.  A new
tracker entry that says ``recurrence of L-FL1`` raises that lesson's count.

    python -m tools.lessons build     # rewrite docs/lessons.jsonl
    python -m tools.lessons check     # exit 1 if the file is out of date, a guard is gone or a review is overdue
    python -m tools.lessons show      # the ledger, open lessons first
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

LEDGER_PATH = "docs/lessons.jsonl"
IST = timezone(timedelta(hours=5, minutes=30))
FIELDS = ("id", "date", "source", "pattern", "summary", "guard_type", "guard_location", "status",
          "recurrence_count", "last_seen")
#: The recurring failure patterns of docs/process_loop_plan.md, section 1.
PATTERNS = {
    "F1": "a bug in our own code found later, not by a test",
    "F2": "a scheduled job late, failed or falsely alarmed",
    "F3": "a required record step skipped, or records out of step",
    "F4": "a wrong or stale fact stated, then corrected",
    "F5": "a bias caught after the fact",
    "F6": "a data fault in a store or source",
    "F7": "a study stopped by data after work had begun",
    "F8": "a cost, margin or funding assumption wrong or unmeasured",
    "F9": "an idea reviewed that was already tested or in place",
    "F10": "environment or version drift",
    "F11": "a human step missed or repeated",
    "F12": "a source's own numbers failing replication",
    "F13": "leftover files after a failed study",
}
GUARD_TYPES = ("test", "preflight", "config_guard", "template_field", "prior")

#: Guards that are not tests: where each lives ("path::words that must appear in that file").
OTHER: List[Dict[str, Any]] = [
    {"id": "L-STALE-FACTS", "date": "2026-11-14", "source": "PG1", "pattern": "F4", "guard_type": "preflight",
     "summary": "An audit and several answers used facts that were out of date (a 10 Oct gap register, "
                "'14 Dec' for a 15 Dec verdict)",
     "guard_location": ["tools/preflight.py::def facts"]},
    {"id": "L-FL3", "date": "2026-11-14", "source": "FL3", "pattern": "F4", "guard_type": "preflight",
     "summary": "A manual nightly run was recommended on a Saturday, when the runner only does the weekly "
                "checkpoint",
     "guard_location": ["tools/preflight.py::def runner_mode"]},
    {"id": "L-PHASE0", "date": "2026-11-09", "source": "NS1-8", "pattern": "F7", "guard_type": "template_field",
     "summary": "Studies began before their data was known to be reachable, point-in-time and complete "
                "(NS1-4, NS1-8, NS1-10, BLK1)",
     "guard_location": ["docs/research_protocol.md::Phase 0 verdict:", "tools/prereg.py::def check_phase0"]},
    {"id": "L-PRIOR-ART", "date": "2026-11-06", "source": "BT7", "pattern": "F9", "guard_type": "preflight",
     "summary": "An idea was studied again without finding the earlier test (H7 of BT7 repeated Y3)",
     "guard_location": ["tools/prereg.py::def prior_art", "docs/research_protocol.md::Prior art:"]},
    {"id": "L-POSTMORTEM", "date": "2026-11-14", "source": "PL0", "pattern": "F3", "guard_type": "config_guard",
     "summary": "What a verdict cost, and where it could have been stopped, was never recorded",
     "guard_location": ["nse_engine/journal.py::POSTMORTEM_REQUIRED"]},
    {"id": "L-STORE-GUARDS", "date": "2026-11-14", "source": "PL-13", "pattern": "F3", "guard_type": "preflight",
     "summary": "The guards that need the data stores skip in CI and ran only when someone remembered",
     "guard_location": ["tools/guards.py::def status", "tools/preflight.py::guards.status"]},
]

#: Known lessons with no guard yet.  ``waived`` needs the owner's dated decision id in ``decision``.
OPEN: List[Dict[str, Any]] = [
    {"id": "L-CK1P2", "date": "2026-11-09", "source": "CK1F", "pattern": "F6", "status": "open",
     "summary": "A relisting after a long suspension is read as a split and rewrites the stock's history "
                "(CCCL, ATLASCYCLE, VIVIMEDLAB); the fix moves the data hash, so it waits for OPS1A",
     "review_by": "2027-01-31"},
    {"id": "L-COST-NOTES", "date": "2026-11-11", "source": "A1M", "pattern": "F8", "status": "open",
     "summary": "No cost or margin figure has been compared with a broker contract note (PL-14, from the "
                "INFY sell test's fills)",
     "review_by": "2026-12-31"},
    {"id": "L-VERDICT-LATE", "date": "2026-11-14", "source": "PL0", "pattern": "F3", "status": "open",
     "summary": "The journal check reads only an entry's opening words, so a verdict stated later (CC1) is "
                "not caught",
     "review_by": "2026-12-31"},
    {"id": "L-RATES-2026", "date": "2026-11-14", "source": "PL-15", "pattern": "F8", "status": "open",
     "summary": "Four published rates are not in the cost models (NSE's charge with the investor-protection "
                "levy on shares, futures and options; the DP charge, Rs 15.34 not 15.93): found inside the RA1 "
                "freeze, to apply with the next cost-model version (nse_engine/validation/rates.py PENDING)",
     "review_by": "2026-12-31"},
    # PL-12, the audit of earlier bugs in our own code: these have no regression test yet.
    {"id": "L-H7", "date": "2026-10-03", "source": "H7", "pattern": "F1", "status": "open",
     "summary": "Four undefined names sat in the code unnoticed (a margin check silently skipped); no linter "
                "runs in CI or in the core install",
     "review_by": "2027-01-31"},
    {"id": "L-IS1", "date": "2026-10-10", "source": "IS1", "pattern": "F1", "status": "open",
     "summary": "A nightly dispatch from a branch would have run the live session on unmerged code; the four "
                "isolation fixes were checked by a script that was then deleted",
     "review_by": "2027-01-31"},
    {"id": "L-G16", "date": "2026-09-30", "source": "G16", "pattern": "F1", "status": "open",
     "summary": "The daily email showed closed trades as the day's P&L (a loss in red on a day equity rose); "
                "no test holds day P&L to the change in equity",
     "review_by": "2027-01-31"},
    {"id": "L-BT1", "date": "2026-10-30", "source": "BT1", "pattern": "F1", "status": "open",
     "summary": "The legacy strategy backtest had seven faults (never sold, unadjusted prices, look-ahead); it "
                "still has no known-answer test.  Legacy page, banner shown: decide test or retire",
     "review_by": "2027-01-31"},
]


def _marks(node: ast.AST) -> List[ast.Call]:
    """The ``pytest.mark.lesson(...)`` calls on a test, a class, or a module's ``pytestmark``."""
    found: List[ast.Call] = []
    exprs: List[ast.AST] = list(getattr(node, "decorator_list", []))
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets):
        exprs = list(node.value.elts) if isinstance(node.value, (ast.List, ast.Tuple)) else [node.value]
    for e in exprs:
        if isinstance(e, ast.Call) and ast.unparse(e.func).endswith("mark.lesson"):
            found.append(e)
    return found


def lessons_in_tests(root: Path = _ROOT) -> Dict[str, Dict[str, Any]]:
    """{lesson id: fields} from the markers in ``tests/``; every marker adds its test to ``guard_location``."""
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted((Path(root) / "tests").glob("test_*.py")):
        rel = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text())
        nodes: List[tuple] = [(n, rel) for n in tree.body if isinstance(n, ast.Assign)]
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                nodes.append((n, f"{rel}::{n.name}"))
                if isinstance(n, ast.ClassDef):
                    nodes += [(m, f"{rel}::{n.name}::{m.name}") for m in n.body
                              if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for node, where in nodes:
            for call in _marks(node):
                if not call.args or not isinstance(call.args[0], ast.Constant):
                    raise ValueError(f"{where}: lesson marker without an id")
                lid = str(call.args[0].value)
                fields = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
                entry = out.setdefault(lid, {"id": lid, "guard_type": "test", "guard_location": []})
                clash = {k for k, v in fields.items() if entry.get(k, v) != v}
                if clash:
                    raise ValueError(f"{where}: lesson {lid} is described twice, differently ({sorted(clash)})")
                entry.update(fields)
                entry["guard_location"].append(where)
    return out


def recurrences(root: Path = _ROOT, today: Optional[date] = None) -> Dict[str, List[str]]:
    """{lesson id: dates of the tracker entries that say "recurrence of <id>"}."""
    from nse_engine import journal

    out: Dict[str, List[str]] = {}
    path = Path(root) / journal.TRACKER_PATH
    if not path.exists():
        return out
    for e in journal.tracker_entries(path.read_text(), today):
        for lid in re.findall(r"recurrence of (L-[A-Z0-9-]+)", e["text"]):
            out.setdefault(lid, []).append(e["date"] or "")
    return out


def build(root: Path = _ROOT, today: Optional[date] = None) -> List[Dict[str, Any]]:
    """The ledger's rows, sorted by id.  Raises when a lesson is malformed, defined twice or its guard is gone."""
    rows: Dict[str, Dict[str, Any]] = {}
    for lid, entry in lessons_in_tests(root).items():
        rows[lid] = {**entry, "status": "guarded"}
    for entry in OTHER:
        if entry["id"] in rows:
            raise ValueError(f"{entry['id']} is defined both in a test and in OTHER")
        for loc in entry["guard_location"]:
            file, _, words = loc.partition("::")
            target = Path(root) / file
            if not target.exists() or (words and words not in target.read_text()):
                raise ValueError(f"{entry['id']}: its guard is gone ({loc})")
        rows[entry["id"]] = {**entry, "status": "guarded"}
    for entry in OPEN:
        if entry["id"] in rows:
            raise ValueError(f"{entry['id']} is guarded: take it out of OPEN")
        if entry["status"] not in ("open", "waived") or (entry["status"] == "waived" and not entry.get("decision")):
            raise ValueError(f"{entry['id']}: an OPEN lesson is 'open', or 'waived' with the owner's decision id")
        rows[entry["id"]] = {**entry, "guard_type": "none", "guard_location": []}
    seen = recurrences(root, today)
    out = []
    for lid in sorted(rows):
        r = rows[lid]
        missing = [f for f in ("date", "source", "pattern", "summary") if not r.get(f)]
        if missing:
            raise ValueError(f"{lid}: no {', '.join(missing)} (give them on one of its markers)")
        if r["pattern"] not in PATTERNS:
            raise ValueError(f"{lid}: unknown pattern {r['pattern']!r}")
        if r["guard_type"] not in GUARD_TYPES + ("none",):
            raise ValueError(f"{lid}: unknown guard type {r['guard_type']!r}")
        again = sorted(d for d in seen.get(lid, []) if d)
        row = {f: r.get(f) for f in FIELDS}
        row.update(recurrence_count=len(seen.get(lid, [])), last_seen=again[-1] if again else r["date"])
        for extra in ("review_by", "decision"):
            if r.get(extra):
                row[extra] = r[extra]
        out.append(row)
    return out


def render(rows: Sequence[Dict[str, Any]]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def check(root: Path = _ROOT, today: Optional[date] = None) -> List[str]:
    """What is wrong with the ledger: out of date, a guard gone, or an open lesson past its review date."""
    today = today or datetime.now(IST).date()
    try:
        rows = build(root, today)
    except ValueError as exc:
        return [str(exc)]
    out: List[str] = []
    path = Path(root) / LEDGER_PATH
    if not path.exists() or path.read_text() != render(rows):
        out.append(f"{LEDGER_PATH} is out of date: run `python -m tools.lessons build`")
    for r in rows:
        if r["status"] == "open" and r.get("review_by", "9999") < today.isoformat():
            out.append(f"{r['id']} is open past its review date ({r['review_by']}): guard it, or waive it on the "
                       f"owner's dated decision")
    return out


def summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    guarded = sum(r["status"] == "guarded" for r in rows)
    return {"lessons": len(rows), "guarded": guarded, "open": sum(r["status"] == "open" for r in rows),
            "waived": sum(r["status"] == "waived" for r in rows),
            "share_guarded": round(guarded / len(rows), 3) if rows else None,
            "recurrences": sum(r["recurrence_count"] for r in rows)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("cmd", choices=("build", "check", "show"))
    args = p.parse_args(argv)
    if args.cmd == "check":
        found = check()
        print("\n".join(found) or "the ledger is current")
        return 1 if found else 0
    rows = build()
    if args.cmd == "build":
        (_ROOT / LEDGER_PATH).write_text(render(rows))
    for r in sorted(rows, key=lambda r: (r["status"] != "open", r["id"])):
        where = "; ".join(r["guard_location"]) or f"no guard (review by {r.get('review_by', '-')})"
        print(f"{r['status']:8} {r['id']:16} {r['pattern']:4} x{r['recurrence_count']}  {where[:96]}")
    print(summary(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
