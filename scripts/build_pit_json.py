"""Build data/nifty500_historical_constituents.json from Wayback snapshots.

Rules (point-in-time, no look-ahead):
  * Each semi-annual period key "YYYY-03" / "YYYY-09" takes effect on the
    first day of that month and uses the nearest snapshot captured ON OR
    BEFORE that day.  A later snapshot is never used for an earlier period.
  * Periods before the first snapshot are skipped (no backfilling).
  * A ``_meta`` entry records snapshot dates, source and build time; the
    loader (kite_connect.nse.nse_universe) ignores keys starting with "_".

Run from anywhere:
    python scripts/build_pit_json.py
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = REPO_ROOT / "data" / "nifty500_wayback_raw.json"
OUT_PATH = REPO_ROOT / "data" / "nifty500_historical_constituents.json"

# Legacy raw labels (pre-rewrite fetch script) -> capture dates.
_LEGACY_LABEL_DATES = {
    "2018-10": "2018-10-04", "2019-02": "2019-02-01", "2020-07": "2020-07-25",
    "2022-05": "2022-05-04", "2022-10": "2022-10-09", "2023-04": "2023-04-04",
    "2024-02": "2024-02-07", "2024-02b": "2024-02-26", "2025-06": "2025-06-16",
    "2025-08": "2025-08-21",
}


def _snapshot_date(label: str) -> _dt.date:
    label = _LEGACY_LABEL_DATES.get(label, label)
    return _dt.date.fromisoformat(label[:10])


def build(raw: Dict, end_year: int | None = None) -> Dict:
    snapshots = sorted(
        ((_snapshot_date(k), sorted(set(v))) for k, v in raw.items()
         if not str(k).startswith("_") and isinstance(v, list) and v),
        key=lambda x: x[0],
    )
    if not snapshots:
        raise SystemExit(f"No snapshots in {RAW_PATH}")
    first = snapshots[0][0]
    end_year = end_year or _dt.date.today().year

    pit: Dict[str, List[str]] = {}
    used: Dict[str, str] = {}
    for y in range(first.year, end_year + 1):
        for m in (3, 9):
            start = _dt.date(y, m, 1)
            if start > _dt.date.today():
                continue
            past = [s for s in snapshots if s[0] <= start]
            if not past:
                continue  # before first snapshot: skip, never backfill
            snap_date, syms = past[-1]
            key = f"{y}-{m:02d}"
            pit[key] = syms
            used[key] = snap_date.isoformat()

    out: Dict = dict(sorted(pit.items()))
    out["_meta"] = {
        "source": "Wayback Machine snapshots of ind_nifty500list.csv "
                  "(scripts/fetch_wayback_nifty500.py)",
        "snapshot_dates": [d.isoformat() for d, _ in snapshots],
        "period_snapshot": used,
        "rule": "period effective on 1st of month uses nearest snapshot captured on/before it; "
                "periods before first snapshot skipped",
        "built_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    return out


def main() -> None:
    raw = json.loads(RAW_PATH.read_text())
    out = build(raw)
    periods = [k for k in out if not k.startswith("_")]
    print(f"Snapshots: {out['_meta']['snapshot_dates']}")
    for p in periods:
        print(f"  {p}: {len(out[p])} symbols (snapshot {out['_meta']['period_snapshot'][p]})")
    for sym in ("DHFL", "YESBANK", "JETAIRWAYS", "ZOMATO", "PAYTM", "LICI", "RCOM"):
        found = [p for p in periods if sym in out[p]]
        print(f"  {sym}: {found[0] + '..' + found[-1] if found else 'not found'} ({len(found)} periods)")
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {len(periods)} periods to {OUT_PATH}")


if __name__ == "__main__":
    main()
