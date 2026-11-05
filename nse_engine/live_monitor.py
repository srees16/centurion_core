"""
Live monitoring fixed before the first live session (plan 5bu, MON1; India tracker GL1).

1. Cumulative gap bands.  The daily gap is G4's: the live book's daily return less its same-period reference.
   With sigma the daily gap's standard deviation on the deployed paper book at its 60th G4 session (RA1),
   floored at 10 bp and then fixed (``fix-sigma``, once), the cumulative gap G(n) is read at n = 60, 120, 250
   and every 250 after.  A breach is G(n) < -2 sigma sqrt(n).  At 60 it is a WATCH line; at 120 or later it
   holds the capital ladder at its rung (no step up) until the next checkpoint clears it or the owner records
   the cause and sets ``CENTURION_GAP_BAND_ACK`` to that checkpoint.  Rule D, the kill switch and the drawdown
   rule are untouched.

2. Monthly universe rank IC (supports, never proves).  At each month-end of live trading, the Spearman
   correlation across the deployed universe between the engine's combined forecast at that close and each
   name's return over the next 21 sessions, scored once those sessions exist.  Reported monthly with its
   running mean and t over months; read at 12 months only (t >= 2 supports; mean <= 0 is a review).

    python -m nse_engine.live_monitor fix-sigma            # on RA1 day: from the deployed paper book's G4
    python -m nse_engine.live_monitor fix-sigma --te 0.062 # or the RA1 email's tracking error, by hand
    python -m nse_engine.live_monitor status
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
SIGMA_PATH = _REPO_ROOT / "config" / "mon1_gap_sigma.json"
SIGMA_FLOOR = 0.0010                       # 10 bp a day
BAND_Z = 2.0
FIRST_CHECKPOINTS = (60, 120, 250)
CHECKPOINT_EVERY = 250
HOLD_FROM = 120                            # a breach at 120 or later holds the ladder
ACK_ENV = "CENTURION_GAP_BAND_ACK"
RA1_SESSIONS = 60
IC_HORIZON = 21
IC_READ_MONTHS = 12
IC_STATE_KEY = "mon1_rank_ic"              # {month_end: ic}, in the live book's state


# --------------------------------------------------------------------------------------------- 1. gap bands

def load_sigma(path: Path = SIGMA_PATH) -> Optional[float]:
    """The fixed daily gap sigma, or None before RA1 fixes it."""
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    s = float(doc["sigma_daily"])
    if not (math.isfinite(s) and s >= SIGMA_FLOOR):
        raise ValueError(f"{path}: sigma_daily {s} is not a finite value at or above the {SIGMA_FLOOR} floor")
    return s


def sigma_from_te(te_annual: float) -> float:
    """Daily gap sigma from G4's annualised tracking error, floored."""
    if not (math.isfinite(te_annual) and te_annual >= 0):
        raise ValueError(f"tracking error {te_annual} is not a finite non-negative number")
    return max(te_annual / math.sqrt(252), SIGMA_FLOOR)


def fix_sigma(te_annual: float, source: str, sessions: Optional[int], path: Path = SIGMA_PATH,
              force: bool = False) -> Dict[str, Any]:
    """Fix sigma once (plan 5bu): refuses to overwrite, and before the paper book's 60th G4 session."""
    if path.exists() and not force:
        raise RuntimeError(f"{path} already fixed sigma; it is fixed once (plan 5bu)")
    if sessions is not None and sessions < RA1_SESSIONS:
        raise RuntimeError(f"the paper book has {sessions} G4 sessions; sigma is fixed at {RA1_SESSIONS} (RA1)")
    doc = {"sigma_daily": sigma_from_te(te_annual), "tracking_error_annual": te_annual, "source": source,
           "paper_sessions": sessions, "fixed_on": date.today().isoformat(), "plan": "5bu MON1"}
    path.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def te_from_gate_summary(summary: Dict[str, Any]) -> float:
    """The tracking error G4 displayed (``"6.2%/yr"``), as a fraction; refuses any other form."""
    for c in summary.get("checks", []):
        if c.get("name") == "tracking error":
            m = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)%/yr\s*", str(c.get("display", "")))
            if not m:
                raise ValueError(f"G4 tracking error display {c.get('display')!r} is not 'x.x%/yr'")
            return float(m.group(1)) / 100.0
    raise ValueError("the G4 summary has no tracking error check")


def checkpoint_for(n: int) -> Optional[int]:
    """The latest checkpoint reached after ``n`` sessions, or None before 60."""
    reached = [c for c in FIRST_CHECKPOINTS if c <= n]
    if n >= FIRST_CHECKPOINTS[-1] + CHECKPOINT_EVERY:
        reached.append(FIRST_CHECKPOINTS[-1] + CHECKPOINT_EVERY * ((n - FIRST_CHECKPOINTS[-1]) // CHECKPOINT_EVERY))
    return max(reached) if reached else None


def gap_band(daily_gap: pd.Series, sigma: Optional[float], ack: Optional[str] = None) -> Dict[str, Any]:
    """The band read at the latest checkpoint reached: status, whether it holds the ladder, and an email line.

    ``daily_gap``: live less reference, one value per live session, oldest first.  ``ack``: the checkpoint the
    owner cleared after recording the cause (``CENTURION_GAP_BAND_ACK``).
    """
    g = pd.Series(daily_gap, dtype="float64").dropna()
    n = int(len(g))
    out: Dict[str, Any] = {"sessions": n, "checkpoint": None, "cum_gap": float(g.sum()), "band": None,
                           "status": "", "hold": False}
    if sigma is None:
        out.update(status="not armed", line="Gap band (MON1): not armed until sigma is fixed at RA1 (fix-sigma)")
        return out
    cp = checkpoint_for(n)
    if cp is None:
        out.update(status="waiting", line=f"Gap band (MON1): {n} of {FIRST_CHECKPOINTS[0]} sessions to the first read")
        return out
    at = float(g.iloc[:cp].sum())
    band = -BAND_Z * sigma * math.sqrt(cp)
    breach = at < band
    out.update(checkpoint=cp, cum_gap=at, band=band)
    acked = ack is not None and str(ack).strip() == str(cp)
    if not breach:
        out["status"] = "OK"
    elif cp < HOLD_FROM:
        out["status"] = "WATCH"
    else:
        out.update(status="BREACH", hold=not acked)
    out["line"] = (f"Gap band (MON1) at {cp} sessions: cumulative gap {at:+.2%} against the band {band:+.2%}: "
                   f"{out['status']}" + (" - the ladder holds at its rung until the cause is recorded "
                                         f"(then {ACK_ENV}={cp})" if out["hold"] else "")
                   + (" (cleared by you)" if breach and acked else ""))
    return out


# --------------------------------------------------------------------------------------------- 2. rank IC

def month_ends(dates: pd.DatetimeIndex, start: pd.Timestamp) -> List[pd.Timestamp]:
    """The last session of each month on or after ``start`` (a month still running has none)."""
    d = pd.DatetimeIndex(dates)
    d = d[d >= pd.Timestamp(start)]
    if not len(d):
        return []
    s = pd.Series(d, index=d)
    last = s.groupby(d.to_period("M")).max()
    complete = last[last.index < d[-1].to_period("M")] if len(last) else last
    return [pd.Timestamp(x) for x in complete.tolist()]


def rank_ic(combined: np.ndarray, universe: np.ndarray, close: np.ndarray, pos: int,
            horizon: int = IC_HORIZON) -> Optional[float]:
    """Spearman correlation at row ``pos`` across the universe between the forecast and the next ``horizon``
    sessions' return; None when the horizon is not complete or fewer than 20 names have both."""
    if pos + horizon >= close.shape[0]:
        return None
    fwd = close[pos + horizon] / close[pos] - 1.0
    ok = universe[pos] & np.isfinite(combined[pos]) & np.isfinite(fwd)
    if int(ok.sum()) < 20:
        return None
    a = pd.Series(combined[pos][ok]).rank().to_numpy()
    b = pd.Series(fwd[ok]).rank().to_numpy()
    return float(np.corrcoef(a, b)[0, 1])


def due_ics(dates: pd.DatetimeIndex, start: pd.Timestamp, done: Dict[str, float],
            horizon: int = IC_HORIZON) -> List[pd.Timestamp]:
    """Month-ends since ``start`` whose horizon is complete in ``dates`` and not yet scored."""
    idx = pd.DatetimeIndex(dates)
    out = []
    for m in month_ends(idx, start):
        p = int(idx.get_loc(m))
        if p + horizon < len(idx) and m.date().isoformat() not in done:
            out.append(m)
    return out


def ic_summary(ics: Dict[str, float]) -> Dict[str, Any]:
    """Running mean and t over months; the verdict only once 12 months are scored."""
    v = np.array([float(x) for _, x in sorted(ics.items())], dtype="float64")
    n = int(len(v))
    mean = float(v.mean()) if n else float("nan")
    t = float(mean / (v.std(ddof=1) / math.sqrt(n))) if n >= 2 and v.std(ddof=1) > 0 else float("nan")
    if n < IC_READ_MONTHS:
        verdict = f"read at {IC_READ_MONTHS} months ({n} so far)"
    elif t >= 2.0:
        verdict = "supports: the signal still ranks"
    elif mean <= 0:
        verdict = "REVIEW: the signal is not ranking"
    else:
        verdict = "inconclusive"
    line = (f"Rank IC (MON1): {n} month(s), mean {mean:+.3f}, t {t:+.1f} (benchmark +0.051): {verdict}"
            if n else "Rank IC (MON1): no month scored yet")
    return {"months": n, "mean": mean, "t": t, "verdict": verdict, "line": line}


def score_due_ics(data, config, start: pd.Timestamp, done: Dict[str, float]) -> Dict[str, float]:
    """Score every due month-end with the engine's own panels (one cache build); returns the new entries."""
    due = due_ics(data.dates, start, done)
    if not due:
        return {}
    from nse_engine.engine import EngineCache

    cache = EngineCache(data, config)
    new = {}
    for m in due:
        ic = rank_ic(cache.combined, cache.universe_mask, cache.close, int(cache.dates.get_loc(m)))
        if ic is not None:
            new[m.date().isoformat()] = ic
    return new


# --------------------------------------------------------------------------------------------- CLI

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="MON1 live monitors (plan 5bu)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fix-sigma", help="fix the gap band's sigma once, at RA1")
    f.add_argument("--te", type=float, help="G4's annualised tracking error (fraction); default: the deployed "
                                            "paper book's stored G4 report")
    f.add_argument("--force", action="store_true", help="replace a sigma already fixed (needs a recorded reason)")
    sub.add_parser("status", help="sigma and the band's thresholds")
    args = ap.parse_args(argv)
    if args.cmd == "fix-sigma":
        if args.te is not None:
            te, source, sessions = args.te, "given by hand (--te)", None
        else:
            from database.paper_cloud import get_paper_cloud
            from nse_engine import paper_gate as pg

            cloud = get_paper_cloud()
            if cloud is None:
                raise SystemExit("no database: give the RA1 email's tracking error with --te")
            summary = json.loads(cloud.read_state().get(pg.STATE_KEY) or "null")
            if not summary:
                raise SystemExit("the deployed paper book has no stored G4 report")
            te, sessions = te_from_gate_summary(summary), int(summary.get("sessions") or 0)
            source = f"deployed paper book G4 ({summary.get('window')})"
        print(json.dumps(fix_sigma(te, source, sessions, force=args.force), indent=1))
        return 0
    s = load_sigma()
    print(json.dumps({"sigma_daily": s, "checkpoints": list(FIRST_CHECKPOINTS) + [f"every {CHECKPOINT_EVERY}"],
                      "bands": {cp: (-BAND_Z * s * math.sqrt(cp) if s else None) for cp in FIRST_CHECKPOINTS},
                      "hold_from": HOLD_FROM}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
