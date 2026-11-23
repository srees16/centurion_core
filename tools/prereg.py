#!/usr/bin/env python3
"""Before a study is pre-registered: earlier work on the idea, and the Phase 0 check (tracker PL0, PL-05).

Two things cost studies that should not have started as they did:
- the idea had been tested or was already in place (H7 of BT7 repeated Y3; several reviews ended "already
  in Centurion");
- the data turned out unreachable, not point-in-time or incomplete after work had begun (NS1-4, NS1-8,
  NS1-10, BLK1, GT11).

``prior-art`` searches the tracker's DONE section and the metrics journal for every entry that holds all the
given words, with its verdict.  ``check`` reads the validation plan: a section pre-registered on or after
``PHASE0_SINCE`` must carry its Phase 0 block (the protocol's template), filled, with a GO or STOP verdict,
and no section may be stamped later than now.

``new`` writes the section (tracker PL-08).  Pre-registration times were typed by hand and three were in the
future (BT7, BT7-S2, GF1), and the family's trial count was copied from memory.  Without ``--body`` it prints
the template with the family's count from ``config/research_families.json`` and the prior art filled in.  With
``--body FILE`` (the filled template, no heading) it puts the section's next number and the clock's time in
the heading, refuses an unfilled Phase 0, and with ``--write`` adds it to the plan.  The time in the plan is
then the moment the complete section entered it.

    python -m tools.prereg prior-art open offer       # earlier entries holding both words
    python -m tools.prereg new --id GF2 --title "the gap's fade, long only" --family intraday --words gap fade
    python -m tools.prereg new --id GF2 --title "..." --family intraday --body draft.txt --write
    python -m tools.prereg check                      # sections without their Phase 0, or stamped in the future
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.clock import IST  # noqa: E402 - after the path bootstrap

PLAN_PATH = "docs/nse_engine_validation_plan.md"
#: Plan sections pre-registered from this date must carry Phase 0; earlier ones had no such block.
PHASE0_SINCE = "2026-11-15"
#: What a filled Phase 0 block states.
PHASE0_FIELDS = ("Prior art:", "Data:", "point-in-time:", "delisted names:", "Cost model for this instrument:")
_SECTION = re.compile(r"^## (5[a-z]*)\. (.*)$")
_PREREG = re.compile(r"pre-registered (\d{1,2} [A-Z][a-z]{2} \d{4})(?:,? (\d{1,2}:\d{2}) IST)?")
#: The template's lines after the heading; ``{prior}`` and ``{family}`` are filled by ``skeleton``.
TEMPLATE = """Phase 0 (before anything else is written or downloaded):
  Prior art: {prior}
  Data: <source>; reachable by script and allowed by its terms: <yes / no>
  coverage 2008-25: <from - to, gaps>; point-in-time: <yes / no>; delisted names: <yes / no>
  Cost model for this instrument: <exists, version / to be built>
  Phase 0 verdict: <GO / STOP (reason)>
Source / hypothesis: <why the return exists>
Family and budget: {family}
Data and window: 2013-01-01..2025-12-31 (test); 2007-25 reported; 2026 sealed
Rules: <exact, with every parameter; translations marked (A)>
Costs and sizing: <cost model; broker-feasible sizing>
Pass rules: gate 1 <...>; gate 2 <...>; then <paper / walk-forward>
Stop rules and cap: <hours, downloads, what ends it early>
Files this study will create: <paths, recorded with `tools.study start`; deleted with a FAIL after approval>
"""


def prior_art(words: Sequence[str], root: Path = _ROOT, today: Optional[date] = None) -> List[Dict[str, Any]]:
    """Tracker entries and journal experiments that hold every word (any case), newest first."""
    from nse_engine import journal

    words = [w.lower() for w in words if w.strip()]
    if not words:
        raise ValueError("give at least one word")
    out: Dict[str, Dict[str, Any]] = {}
    for e in journal.tracker_entries((Path(root) / journal.TRACKER_PATH).read_text(), today):
        if all(w in e["text"].lower() for w in words):
            out.setdefault(e["id"], {"id": e["id"], "date": e["date"], "verdict": journal.stated_verdict(e["text"]),
                                     "where": "tracker", "text": e["text"][:140]})
    rows = journal.read_journal(Path(root) / journal.JOURNAL_PATH)
    for r in rows[rows["kind"] == journal.EXPERIMENT].to_dict(orient="records"):
        text = f"{r['event']} {r['note']}"
        if all(w in text.lower() for w in words):
            hit = out.setdefault(str(r["ref"]), {"id": str(r["ref"]), "date": r["date"], "where": "journal",
                                                 "text": str(r["event"])[:140]})
            hit["verdict"] = hit.get("verdict") or r["verdict"]
    return sorted(out.values(), key=lambda h: str(h["date"]), reverse=True)


def check_phase0(plan: str, since: str = PHASE0_SINCE) -> List[str]:
    """Plan sections pre-registered on or after ``since`` whose Phase 0 block is missing or unfilled."""
    lines = plan.split("\n")
    heads = [(i, m) for i, m in ((i, _SECTION.match(line)) for i, line in enumerate(lines)) if m]
    ends = [i for i, line in enumerate(lines) if line.startswith("## ")] + [len(lines)]
    out: List[str] = []
    for i, m in heads:
        stamp = _PREREG.search(m.group(2))
        if not stamp or datetime.strptime(stamp.group(1), "%d %b %Y").date().isoformat() < since:
            continue
        body = "\n".join(lines[i:next(e for e in ends if e > i)])
        name = f"section {m.group(1)}"
        verdict = re.search(r"Phase 0 verdict:\s*(GO|STOP)\b", body)
        if "Phase 0" not in body or not verdict:
            out.append(f"{name}: no Phase 0 block with a GO or STOP verdict")
            continue
        for field in PHASE0_FIELDS:
            filled = re.search(re.escape(field) + r"[ \t]*(\S[^\n]*)", body)
            if not filled or re.search(r"<[^<>]*[A-Za-z][^<>]*>", filled.group(1).split(";")[0]):   # a placeholder left
                out.append(f"{name}: Phase 0 leaves '{field}' unfilled")
    return out


def stamp(now: Optional[datetime] = None) -> str:
    """The clock in the plan's form: "07 Nov 2026, 00:56 IST"."""
    return (now or datetime.now(IST)).astimezone(IST).strftime("%d %b %Y, %H:%M IST")


def stamp_problems(plan: str, now: Optional[datetime] = None) -> List[str]:
    """Plan sections whose pre-registration is stamped later than ``now``: a time typed by hand, not read."""
    now = (now or datetime.now(IST)).astimezone(IST)
    out: List[str] = []
    for line in plan.split("\n"):
        m, at = _SECTION.match(line), _PREREG.search(line)
        if not (m and at):
            continue
        day = datetime.strptime(at.group(1), "%d %b %Y")
        hour, minute = (int(x) for x in at.group(2).split(":")) if at.group(2) else (0, 0)
        if day.replace(hour=hour, minute=minute, tzinfo=IST) > now:
            out.append(f"section {m.group(1)}: pre-registered {at.group(0)[15:]}, which is after now ({stamp(now)})")
    return out


def next_section(plan: str) -> str:
    """The number after the plan's last ``5..`` section: 5cb -> 5cc, 5az -> 5ba."""
    ids = [m.group(1)[1:] for m in (_SECTION.match(line) for line in plan.split("\n")) if m and m.group(1)[1:]]
    if not ids:
        return "5a"
    n = 0
    for ch in max(ids, key=lambda s: (len(s), s)):
        n = n * 26 + ord(ch) - 96
    out = ""
    n += 1
    while n:
        n, r = divmod(n - 1, 26)
        out = chr(97 + r) + out
    return "5" + out


def family_line(family: str, root: Path = _ROOT) -> str:
    """The template's family line, with the count the families file holds (``nse_engine.validation.families``)."""
    from nse_engine.validation import families

    known = families.load(root)
    if family not in known:
        raise ValueError(f"unknown family {family!r}: one of {', '.join(known)}, or add it to {families.FAMILIES_PATH}")
    f = known[family]
    cap = f.get("max_configurations")
    return (f"{family}, <k> of {cap if cap else 'no fixed cap'} configurations; N {f['n']} -> <{f['n']} + k> "
            f"(count as of {f.get('as_of', '?')}; {f.get('status', '')})").replace("; )", ")")


def skeleton(family: str, words: Sequence[str] = (), root: Path = _ROOT, today: Optional[date] = None) -> str:
    """The template to fill: the family's count and, when ``words`` are given, the prior art."""
    prior = "<ids and verdicts from `tools.prereg prior-art`, or \"none found\">"
    if words:
        hits = prior_art(words, root, today)
        prior = (", ".join(f"{h['id']} ({h['verdict'] or 'no verdict stated'})" for h in hits[:12])
                 + (f" and {len(hits) - 12} more" if len(hits) > 12 else "")) if hits else \
            f"none found for: {' + '.join(words)}"
    return TEMPLATE.format(prior=prior, family=family_line(family, root))


def add_section(plan: str, study: str, title: str, body: str, now: Optional[datetime] = None) -> Tuple[str, str]:
    """(the plan with the new section, its heading).  The heading carries the next section number and the
    clock's time; a body whose Phase 0 is missing or unfilled is refused."""
    heading = f"## {next_section(plan)}. {study}: {title} (India tracker {study}; pre-registered {stamp(now)})"
    section = f"{heading}\n\n{body.strip()}\n"
    wrong = check_phase0(section, since="0000-00-00")
    if wrong:
        raise ValueError("not added: " + "; ".join(wrong))
    lines = plan.split("\n")
    last = max(i for i, line in enumerate(lines) if _SECTION.match(line))
    end = next((i for i in range(last + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return "\n".join(lines[:end] + section.split("\n") + lines[end:]), heading


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prior-art", help="earlier entries holding every word")
    a.add_argument("words", nargs="+")
    n = sub.add_parser("new", help="the template to fill, or the filled section stamped and added")
    n.add_argument("--id", required=True, help="the tracker id")
    n.add_argument("--title", required=True)
    n.add_argument("--family", required=True, help="a family of config/research_families.json")
    n.add_argument("--words", nargs="*", default=[], help="words for the prior-art search")
    n.add_argument("--body", help="a file holding the filled template (no heading)")
    n.add_argument("--write", action="store_true", help="add the section to the plan (needs --body)")
    sub.add_parser("check", help="plan sections without their Phase 0, or stamped in the future")
    args = p.parse_args(argv)
    if args.cmd == "prior-art":
        hits = prior_art(args.words)
        for h in hits:
            print(f"{h['date']}  {h['id']:9} {str(h['verdict'] or '-'):11} {h['text']}")
        print(f"{len(hits)} earlier entr{'y' if len(hits) == 1 else 'ies'} hold: {' + '.join(args.words)}")
        return 0
    plan = (_ROOT / PLAN_PATH).read_text()
    if args.cmd == "new":
        if not args.body:
            if args.write:
                p.error("--write needs --body: the section is stamped when it is complete")
            print(f"## {next_section(plan)}. {args.id}: {args.title} (India tracker {args.id}; pre-registered "
                  f"<stamped when the filled section is added with --body>)\n\n" + skeleton(args.family, args.words))
            return 0
        family_line(args.family)
        try:
            new_plan, heading = add_section(plan, args.id, args.title, Path(args.body).read_text())
        except ValueError as exc:
            print(exc)
            return 1
        if args.write:
            (_ROOT / PLAN_PATH).write_text(new_plan)
        print(heading + ("\nadded to " + PLAN_PATH if args.write else "\nnot written: pass --write to add it"))
        return 0
    found = check_phase0(plan) + stamp_problems(plan)
    print("\n".join(found) or f"every section pre-registered since {PHASE0_SINCE} has its Phase 0, and none is "
                              f"stamped in the future")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
