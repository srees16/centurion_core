"""
Refresh every research store with the sessions published since its last build.

Live and paper decisions already read current data: the GitHub paper workflow
re-syncs NSE's archive on every run.  This keeps the local research stores on
the same data.  Those stores feed backtests, scorecards, forward-gate checks
and diagnostics.  Steps:

1. **NSE archive:** equity, delivery, indices, corporate actions, F&O and
   participant OI.  The window runs from 14 days before the equity store's
   last session (to catch late files) to today.
2. **Equity store** (``run_nse_engine build-store``).  It also checks the
   trial registry's data fingerprint and prints the refresh plan if it moved;
   the registry itself is never refreshed here.
3. **F&O store.**
4. **MCX:** the bhavcopy mirror (from 14 days before its last session) and
   the store.
5. **Fundamentals:** filing lists, new result documents and the store.  This
   step is skipped while a GT1 run holds the archive.
6. **Paper books** (``tools.export_books``, tracker JR2): every book's Neon record
   as CSV, so the backup carries it.  It needs ``CENTURION_DATABASE_URL`` in
   ``.env``; without it the step fails and the rest still runs.
7. **Backup** (``cloud.backup push``): every piece that changed goes to the
   private Kaggle backup datasets, so nothing lives only on this machine.

Each step is its own process, the same command as by hand, so one failing does
not stop the rest.  The outcome goes to ``data/nse_engine/refresh_status.json``:
each step's exit code, plus each store's last session and data hash.  The exit
code is 1 when any step failed.

New data is never used to re-tune a configuration (research protocol 4).  The
2026 holdout stays sealed; new sessions reach decisions through the rules,
which read every close as it arrives.

    python -m runners.refresh_data                      # every step
    python -m runners.refresh_data --only mcx,fundamentals
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
DATA = _ROOT / "data" / "nse_engine"
STATUS = DATA / "refresh_status.json"
LATE_FILES = timedelta(days=14)
STEPS = ("nse", "equity_store", "fo_store", "mcx", "fundamentals", "books", "backup")


def _last_session(calendar: Path, fallback: date) -> date:
    try:
        return pd.read_parquet(calendar)["date"].max().date()
    except (OSError, ValueError, KeyError):
        return fallback


def _mcx_last_session() -> date:
    days = sorted((DATA / "archive" / "mcx").glob("[0-9]" * 4 + "/mcx*.json.gz"))
    return pd.Timestamp(days[-1].name[3:11]).date() if days else date(2005, 1, 1)


def _gt1_running() -> bool:
    """A GT1 run (its own detached worker) still holds the filings archive."""
    pid_file = DATA / "fundamentals" / "run.pid"
    if not pid_file.exists() or (DATA / "fundamentals" / "run.done").exists():
        return False
    out = subprocess.run(["ps", "-p", pid_file.read_text().strip(), "-o", "command="],
                         capture_output=True, text=True)
    return "fundamentals/run.sh" in out.stdout


def commands(today: date) -> Dict[str, List[List[str]]]:
    py = sys.executable
    nse_from = _last_session(DATA / "store" / "calendar.parquet", date(2012, 1, 1)) - LATE_FILES
    mcx_from = _mcx_last_session() - LATE_FILES
    return {
        "nse": [[py, "-m", "nse_engine.data.archive", "--root", str(DATA / "archive"),
                 "--start", nse_from.isoformat(), "--end", today.isoformat(),
                 "--kinds", "equity,delivery,indices,corpact,fo,participant"]],
        "equity_store": [[py, "-m", "runners.run_nse_engine", "build-store"]],
        "fo_store": [[py, "-m", "nse_engine.data.fo_store"]],
        "mcx": [[py, "-m", "nse_engine.data.mcx_store", "sync", "--start", mcx_from.isoformat(),
                 "--end", today.isoformat()],
                [py, "-m", "nse_engine.data.mcx_store", "build"]],
        "fundamentals": [[py, "-m", "nse_engine.data.fundamentals", "sync"],
                         [py, "-m", "nse_engine.data.fundamentals", "build"]],
        "books": [[py, "-m", "tools.export_books"]],
        "backup": [[py, "-m", "cloud.backup", "push"]],
    }


def store_status() -> Dict[str, Dict[str, object]]:
    """Last session and data hash of each store, as built now."""
    def manifest(name: str) -> dict:
        path = DATA / name / "manifest.json"
        return json.loads(path.read_text()) if path.exists() else {}

    out: Dict[str, Dict[str, object]] = {
        "equity_store": {"last_session": str(_last_session(DATA / "store" / "calendar.parquet", date.min))},
        "fo_store": {"data_hash": manifest("fo_store").get("data_hash")},
        "mcx_store": {"last_session": str(_mcx_last_session()), "data_hash": manifest("mcx_store").get("data_hash")},
    }
    results = DATA / "fundamentals" / "results.parquet"
    if results.exists():
        last = pd.read_parquet(results, columns=["disseminated"])["disseminated"].max()
        out["fundamentals"] = {"last_filing": str(last), "data_hash": manifest("fundamentals").get("data_hash")}
    return out


def refresh(only: Optional[Sequence[str]] = None, today: Optional[date] = None) -> Dict[str, object]:
    today = today or date.today()
    plan = commands(today)
    steps: Dict[str, object] = {}
    for name in only or STEPS:
        if name == "fundamentals" and _gt1_running():
            steps[name] = "skipped: a GT1 run holds the filings archive"
            print(f"== {name}: skipped (GT1 running)", flush=True)
            continue
        codes = []
        for cmd in plan[name]:
            print(f"== {name}: {' '.join(cmd[1:])}", flush=True)
            codes.append(subprocess.run(cmd, cwd=str(_ROOT)).returncode)
            if codes[-1] != 0:
                break  # a failed sync leaves the store as it was rather than building on a partial mirror
        steps[name] = max(codes)
    status = {"finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "ok": all(v == 0 or isinstance(v, str) for v in steps.values()),
              "steps": steps, "stores": store_status()}
    STATUS.write_text(json.dumps(status, indent=1, default=str))
    return status


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Refresh every research store with new sessions")
    p.add_argument("--only", help=f"comma list of steps: {', '.join(STEPS)}")
    args = p.parse_args(argv)
    only = [s.strip() for s in args.only.split(",")] if args.only else None
    unknown = set(only or ()) - set(STEPS)
    if unknown:
        p.error(f"unknown steps: {sorted(unknown)}")
    status = refresh(only)
    print(json.dumps(status, indent=1, default=str))
    return 0 if status["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
