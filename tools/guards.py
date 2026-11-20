#!/usr/bin/env python3
"""The guards that need this machine's data stores, run on purpose and on record (tracker PL-13).

CI installs no store, so the tests marked ``slow`` skip there: fills at the next open, no bar the build would
refuse, stocks that stopped trading still present, the stated coverage, the families file against the
registries.  They ran only when someone remembered.  ``run`` runs them and records the outcome in a
git-ignored file; the preflight shows when they last ran and says so when that is over a week ago, when a
guard failed, or when one skipped (a store is missing, so the guard checked nothing).  Each run also rebuilds the
data catalogue and its store doctor (``tools.data_catalogue``, PL-10 and PL-11) and records its findings.

    python -m tools.guards            # run the store guards and record the outcome
    python -m tools.guards status     # the last recorded run
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.clock import IST  # noqa: E402 - after the path bootstrap

#: Git-ignored (data/nse_engine/): the record is this machine's, not the repository's.
RECORD = "data/nse_engine/guards_last_run.json"
#: Before every handover, and at least this often.
DUE_DAYS = 7
_COUNT = re.compile(r"(\d+) (passed|failed|skipped|errors?|deselected)")


def parse_summary(text: str) -> Dict[str, int]:
    """pytest's closing line as counts: passed, failed (errors included), skipped."""
    lines = [line for line in text.strip().splitlines() if _COUNT.search(line)]
    found = {k.rstrip("s") if k.startswith("error") else k: int(n) for n, k in _COUNT.findall(lines[-1])} if lines else {}
    return {"passed": found.get("passed", 0), "failed": found.get("failed", 0) + found.get("error", 0),
            "skipped": found.get("skipped", 0)}


def run(root: Path = _ROOT, now: Optional[datetime] = None) -> Dict[str, Any]:
    """Run the store guards and record the outcome.  A run that collected nothing counts as failed."""
    began = time.monotonic()
    done = subprocess.run([sys.executable, "-m", "pytest", "tests", "-m", "slow", "-p", "no:cacheprovider", "-rs"],  # -q: pytest.ini
                          cwd=root, capture_output=True, text=True)
    counts = parse_summary(done.stdout)
    if not sum(counts.values()):
        counts["failed"] = 1
    record = {"when": (now or datetime.now(IST)).astimezone(IST).isoformat(timespec="minutes"), **counts,
              "exit": done.returncode, "seconds": round(time.monotonic() - began)}
    try:                                                  # PL-10, PL-11: the catalogue and the store doctor
        from tools import data_catalogue

        catalogue = data_catalogue.build(root, now)
        data_catalogue.write(catalogue, root)
        record["data_findings"] = len(catalogue["findings"])
    except Exception as exc:                              # noqa: BLE001 - recorded, never stops the guards' record
        record["data_findings"] = f"not built ({type(exc).__name__}: {exc})"[:200]
    target = Path(root) / RECORD
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=1) + "\n")
    record["output"] = done.stdout[-3000:]
    return record


def last_run(root: Path = _ROOT) -> Optional[Dict[str, Any]]:
    path = Path(root) / RECORD
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except ValueError:
        return None


def status(root: Path = _ROOT, today: Optional[date] = None) -> str:
    """One line: the last run and what is wrong with it, if anything ("!" marks what needs doing)."""
    today = today or datetime.now(IST).date()
    r = last_run(root)
    if not r:
        return "! never run on this machine: python -m tools.guards"
    when = datetime.fromisoformat(r["when"]).date()
    line = f"last run {when:%a %d %b %Y}: {r['passed']} passed, {r['failed']} failed, {r['skipped']} skipped"
    wrong = []
    if "data_findings" in r:                              # runs since PL-10 also rebuild the data catalogue
        line += f"; store doctor: {r['data_findings']} finding(s)" if isinstance(r["data_findings"], int) \
            else f"; store doctor {r['data_findings']}"
        if r["data_findings"]:
            wrong.append("see python -m tools.data_catalogue")
    if r["failed"]:
        wrong.append("a guard failed")
    if r["skipped"]:
        wrong.append("a guard skipped (a store is missing: it checked nothing)")
    if (today - when).days > DUE_DAYS:
        wrong.append(f"over {DUE_DAYS} days ago: python -m tools.guards")
    return ("! " + line + "; " + "; ".join(wrong)) if wrong else f"{line}; due again by {when + timedelta(days=DUE_DAYS):%d %b}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("cmd", nargs="?", default="run", choices=("run", "status"))
    args = p.parse_args(argv)
    if args.cmd == "run":
        r = run()
        print(r["output"].strip().splitlines()[-1] if r["output"].strip() else "pytest printed nothing")
        print(f"Store guards: {status()} ({r['seconds']} s)")
        return 1 if r["failed"] or r["skipped"] else 0
    print(f"Store guards: {status()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
