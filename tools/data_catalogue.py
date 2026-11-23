#!/usr/bin/env python3
"""What data this machine holds, how fresh it is, and what looks wrong in the live store (tracker PL-10, PL-11).

The catalogue: each store's first and last date, rows, build fingerprint or data hash, and how many NSE sessions
it is behind the last completed one.  Stores the refresh keeps (``refresh_status.json``) are expected to be
current; the others are research snapshots and only report their age.

The store doctor reads the live equity store (``data/nse_engine/store``) for faults the build and the nightly
canary do not catch:

- **calendar**: in years with NSE's holiday list (``nse_calendar.HOLIDAY_YEARS``), a session on a day NSE did not
  trade, or a trading day missing; in the other years, a weekday missing that the download did not record as a
  holiday (``archive/manifest/missing.json``);
- **weekend sessions** not in ``nse_calendar.SPECIAL_SESSIONS``: Sun 11 Nov 2012 entered the store once and moved
  every later rebalance day (CAN1);
- **short sessions**: fewer than ``SHORT_SESSION_SHARE`` of the rows of the 20 sessions before, the mark of a
  truncated or partial file (the build only warns under 100 rows);
- the build's own **suspect missing sessions** (LN-T7) not marked known.

Weekend and short sessions are judged against a baseline kept in the catalogue: what existed when it began (the
2006-15 Saturday sessions; 18 May 2009, the election-day circuit halt) and whatever is acknowledged since with
``ack``.  A new one is reported every run until it is.  Nothing here changes a store.  The catalogue is written
to ``data/nse_engine/catalogue.json`` (git-ignored: it describes this machine's data); ``tools.guards`` rebuilds
it with the weekly store guards and the preflight reads it.

    python -m tools.data_catalogue                       # rebuild, write, print
    python -m tools.data_catalogue status                # one line from the saved catalogue
    python -m tools.data_catalogue ack 2026-11-21 --reason "Muhurat session, NSE circular ..."
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.clock import IST  # noqa: E402 - after the path bootstrap

DATA = "data/nse_engine"
CATALOGUE = f"{DATA}/catalogue.json"
LIVE_STORE = "store"
#: A session with fewer rows than this share of the median of the 20 before it is short.
SHORT_SESSION_SHARE = 0.7
SHORT_SESSION_WINDOW = 20
#: The catalogue counts as current for the preflight for this many days (the store guards run weekly).
STALE_DAYS = 8

#: (name, folder, how its dates are read, sessions it may be behind when refreshed, key in refresh_status).
STORES = (
    ("equity", "store", "calendar", 0, "equity_store"),
    ("equity 2006", "store_ext2006", "calendar", 0, None),
    ("F&O", "fo_store", "market.parquet:date", 0, "fo_store"),
    ("MCX", "mcx_store", "continuous.parquet:date", 1, "mcx_store"),
    ("fundamentals", "fundamentals", "results.parquet:disseminated", 5, "fundamentals"),
    ("bonds", "bond_store", "levels.parquet:index", 0, None),
)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _dates(folder: Path, how: str) -> pd.DatetimeIndex:
    if how == "calendar":
        return pd.DatetimeIndex(pd.read_parquet(folder / "calendar.parquet")["date"]).normalize()
    file, column = how.split(":")
    df = pd.read_parquet(folder / file, columns=None if column == "index" else [column])
    values = df.index if column == "index" else df[column]
    return pd.DatetimeIndex(pd.to_datetime(values).dropna()).normalize().unique().sort_values()


def _identity(folder: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    """The store's data hash, or for the equity stores a fingerprint of the year builds (not the canary's hash)."""
    if manifest.get("data_hash"):
        return {"data_hash": manifest["data_hash"]}
    years = manifest.get("years") or {}
    prints = [str(v.get("fingerprint")) for _, v in sorted(years.items()) if isinstance(v, dict)]
    if not prints:
        return {}
    return {"build_fingerprint": hashlib.sha256("|".join(prints).encode()).hexdigest()[:16]}


def _rows(manifest: Dict[str, Any]) -> Optional[int]:
    years = manifest.get("years") or {}
    if years and all(isinstance(v, dict) and "equity_rows" in v for v in years.values()):
        return int(sum(v["equity_rows"] for v in years.values()))
    for key in ("continuous_rows", "results_rows"):        # F&O is partitioned: its manifest counts dates only
        if key in manifest:
            return int(manifest[key])
    return None


def sessions_behind(last: date, done: date) -> int:
    """NSE trading days after ``last`` up to and including ``done``, the last completed session."""
    from services.execution.carver_pipeline import is_nse_trading_day

    return sum(1 for d in pd.date_range(last + timedelta(days=1), done) if is_nse_trading_day(d.date()))


def store_rows(root: Path, done: date, refresh: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One row per store found on this machine."""
    refreshed = set(refresh.get("stores") or [])
    out = []
    for name, folder, how, allowed, key in STORES:
        path = Path(root) / DATA / folder
        if not path.is_dir():
            continue
        manifest = _read_json(path / "manifest.json")
        row: Dict[str, Any] = {"name": name, "folder": folder, "refreshed": bool(key and key in refreshed)}
        try:
            dates = _dates(path, how)
        except Exception as exc:                          # noqa: BLE001 - a store that cannot be read is a finding
            out.append({**row, "error": f"{type(exc).__name__}: {exc}"[:200]})
            continue
        row.update(first=dates[0].date().isoformat() if len(dates) else None,
                   last=dates[-1].date().isoformat() if len(dates) else None, dates=int(len(dates)),
                   rows=_rows(manifest), **_identity(path, manifest))
        if len(dates):
            row["behind"] = sessions_behind(dates[-1].date(), done)
            row["stale"] = row["refreshed"] and row["behind"] > allowed
        out.append(row)
    return out


def _session_rows(store: Path) -> pd.Series:
    parts = [pd.read_parquet(f, columns=["date"]).groupby("date").size() for f in sorted((store / "equity").glob("*.parquet"))]
    return pd.concat(parts).sort_index() if parts else pd.Series(dtype="int64")


def doctor(store: Path, archive: Path, baseline: Dict[str, Any]) -> Dict[str, Any]:
    """The store doctor's checks on the live equity store; ``findings`` is what needs a look."""
    from nse_engine import nse_calendar as nc

    sessions = pd.DatetimeIndex(pd.read_parquet(store / "calendar.parquet")["date"]).normalize()
    have = set(sessions.date)
    first, last = sessions.min().date(), sessions.max().date()
    span = [d.date() for d in pd.date_range(first, last)]
    listed = [d for d in span if d.year in nc.HOLIDAY_YEARS]
    extra = sorted(d.isoformat() for d in have if d.year in nc.HOLIDAY_YEARS and not nc.is_trading_day(d))
    missing = sorted(d.isoformat() for d in listed if nc.is_trading_day(d) and d not in have)
    recorded = _read_json(archive / "manifest" / "missing.json")
    if recorded.get("equity") is not None:
        holidays = set(recorded["equity"])
        unexplained = sorted(d.isoformat() for d in span if d.weekday() < 5 and d not in have   # the listed
                             and d.year not in nc.HOLIDAY_YEARS and d.isoformat() not in holidays)  # years: above
    else:
        unexplained = None                                # the download's record is not on this machine
    weekend = sorted(d.isoformat() for d in have if d.weekday() >= 5 and not nc.is_special_session(d))
    rows = _session_rows(store)
    median = rows.rolling(SHORT_SESSION_WINDOW, min_periods=5).median().shift(1)
    short = sorted(d.date().isoformat() for d, r in (rows / median).items() if r < SHORT_SESSION_SHARE)
    suspects = [s["date"] for s in _read_json(store / "manifest.json").get("suspect_missing_sessions", [])
                if not s.get("known")]
    known_weekend = set((baseline.get("weekend_sessions") or {}))
    known_short = set((baseline.get("short_sessions") or {}))
    findings = ([f"{d}: a session on a day NSE did not trade" for d in extra]
                + [f"{d}: an NSE trading day missing from the store" for d in missing]
                + [f"{d}: a weekday missing that the download did not record as a holiday" for d in unexplained or []]
                + [f"{d}: a weekend session not listed in nse_calendar.SPECIAL_SESSIONS (CAN1)" for d in weekend
                   if d not in known_weekend]
                + [f"{d}: a short session, under {SHORT_SESSION_SHARE:.0%} of the rows of the "
                   f"{SHORT_SESSION_WINDOW} before" for d in short if d not in known_short]
                + [f"{d}: the build's suspect missing session, not marked known (LN-T7)" for d in suspects])
    return {"store": str(store.name), "first": first.isoformat(), "last": last.isoformat(),
            "calendar_years": sorted(y for y in nc.HOLIDAY_YEARS if first.year <= y <= last.year),
            "extra": extra, "missing": missing, "unexplained_gaps": unexplained, "weekend_unlisted": weekend,
            "short_sessions": short, "unknown_suspects": suspects, "findings": findings}


def build(root: Path = _ROOT, now: Optional[datetime] = None, previous: Optional[Dict[str, Any]] = None
          ) -> Dict[str, Any]:
    """The catalogue.  ``previous`` (default: the saved one) carries the doctor's baseline forward; a first
    catalogue takes what exists today as its baseline."""
    from services.execution.carver_pipeline import last_completed_nse_session

    now = (now or datetime.now(IST)).astimezone(IST)
    root = Path(root)
    previous = previous if previous is not None else _read_json(root / CATALOGUE)
    refresh = _read_json(root / DATA / "refresh_status.json")
    done = last_completed_nse_session(now)
    out: Dict[str, Any] = {"built_at": now.isoformat(timespec="minutes"), "last_completed_session": done.isoformat(),
                           "refresh": {k: refresh.get(k) for k in ("finished_at", "ok", "steps")},
                           "stores": store_rows(root, done, refresh)}
    baseline = previous.get("baseline")
    store = root / DATA / LIVE_STORE
    if not (store / "calendar.parquet").exists():
        out.update(baseline=baseline or {}, doctor=None, findings=["the live equity store is not on this machine"])
        return out
    if baseline is None:                                  # the first catalogue: today's state is the baseline
        first = doctor(store, root / DATA / "archive", {})
        why = f"present when the catalogue began ({now:%d %b %Y})"
        baseline = {"weekend_sessions": {d: why for d in first["weekend_unlisted"]},
                    "short_sessions": {d: why for d in first["short_sessions"]}}
    out["baseline"] = baseline
    out["doctor"] = doctor(store, root / DATA / "archive", baseline)
    out["findings"] = (out["doctor"]["findings"]
                       + [f"{r['name']} store: {r['behind']} session(s) behind {done:%d %b}" for r in out["stores"]
                          if r.get("stale")]
                       + [f"{r['name']} store unreadable: {r['error']}" for r in out["stores"] if r.get("error")])
    return out


def write(catalogue: Dict[str, Any], root: Path = _ROOT) -> Path:
    path = Path(root) / CATALOGUE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalogue, indent=1, default=str) + "\n")
    return path


def acknowledge(day: str, reason: str, root: Path = _ROOT) -> Dict[str, Any]:
    """Add a weekend or short session the doctor reports to its baseline, with the reason it is genuine."""
    if not reason.strip():
        raise ValueError("give the reason the session is genuine (an NSE circular, a known event)")
    day = date.fromisoformat(day).isoformat()
    cat = _read_json(Path(root) / CATALOGUE)
    doc = cat.get("doctor") or {}
    kinds = [k for k, found in (("weekend_sessions", doc.get("weekend_unlisted")), ("short_sessions", doc.get("short_sessions")))
             if day in (found or [])]
    if not kinds:
        raise ValueError(f"{day} is not a weekend or short session in the saved catalogue: rebuild it first")
    for k in kinds:
        cat.setdefault("baseline", {}).setdefault(k, {})[day] = reason.strip()
    write(cat, root)
    return cat


def status(root: Path = _ROOT, today: Optional[date] = None) -> str:
    """One line for the preflight ("!" when the catalogue is missing, old, or has findings)."""
    today = today or datetime.now(IST).date()
    cat = _read_json(Path(root) / CATALOGUE)
    if not cat:
        return "! no catalogue on this machine: python -m tools.guards (or python -m tools.data_catalogue)"
    built = datetime.fromisoformat(cat["built_at"]).date()
    by_day: Dict[str, List[str]] = {}
    for r in cat.get("stores", []):
        if r.get("refreshed") and r.get("last"):
            by_day.setdefault(r["last"], []).append(r["name"] + (f" ({r['behind']} behind)" if r.get("behind") else ""))
    stores = "; ".join(f"{', '.join(names)} to {date.fromisoformat(d):%d %b}" for d, names in sorted(by_day.items()))
    line = f"catalogue of {built:%d %b}: {len(cat.get('findings') or [])} finding(s); {stores}"
    wrong = []
    if cat.get("findings"):
        wrong.append("; ".join(cat["findings"][:3]) + (" ..." if len(cat["findings"]) > 3 else ""))
    if (today - built).days > STALE_DAYS:
        wrong.append(f"over {STALE_DAYS} days old: python -m tools.guards")
    return ("! " + line + " | " + " | ".join(wrong)) if wrong else line


def render(cat: Dict[str, Any]) -> str:
    lines = [f"Data on this machine, {cat['built_at']} (last completed NSE session {cat['last_completed_session']})"]
    for r in cat["stores"]:
        if r.get("error"):
            lines.append(f"  {r['name']:13} UNREADABLE {r['error']}")
            continue
        ident = r.get("data_hash") or r.get("build_fingerprint") or "-"
        kind = "refreshed" if r["refreshed"] else "snapshot"
        rows = f"{r['rows']:>10,}" if r.get("rows") is not None else f"{'-':>10}"
        lines.append(f"  {r['name']:13} {r['first']} .. {r['last']}  {r['dates']:>5} dates  "
                     f"{rows} rows  {ident}  {kind}, {r.get('behind', 0)} behind")
    doc = cat.get("doctor")
    if doc:
        gaps = "not checked (no download record)" if doc["unexplained_gaps"] is None else len(doc["unexplained_gaps"])
        lines.append(f"Store doctor ({doc['store']}, {doc['first']} .. {doc['last']}): calendar years "
                     f"{doc['calendar_years']}: {len(doc['extra'])} extra, {len(doc['missing'])} missing; gaps without "
                     f"a recorded holiday: {gaps}; weekend sessions unlisted {len(doc['weekend_unlisted'])} "
                     f"({len(cat['baseline'].get('weekend_sessions', {}))} in the baseline); short sessions "
                     f"{len(doc['short_sessions'])} ({len(cat['baseline'].get('short_sessions', {}))} in the baseline)")
    lines += [f"! {f}" for f in cat.get("findings") or []] or ["No findings."]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("cmd", nargs="?", default="build", choices=("build", "status", "ack"))
    p.add_argument("day", nargs="?")
    p.add_argument("--reason", default="")
    args = p.parse_args(argv)
    if args.cmd == "status":
        print(f"Data: {status()}")
        return 0
    if args.cmd == "ack":
        if not args.day:
            p.error("ack needs the session's date")
        acknowledge(args.day, args.reason)
        print(f"{args.day} added to the store doctor's baseline")
        return 0
    cat = build()
    write(cat)
    print(render(cat))
    return 1 if cat["findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
