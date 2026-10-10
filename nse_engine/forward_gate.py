"""
Forward promotion gate (tracker V3, decision U19 of 27 Sep 2026).

A configuration replaces the deployed one only after it has traded forward
beside it.  PBO over the same-window registry (45-49% at 46-47 configs) can
no longer be passed by anything, so it is reported with its trial count
instead of blocking.  ``promote`` passes a candidate only when all three hold:

1. paper sessions: the candidate book (``config/nse_engine_candidate.json``,
   its own Neon schema) has >= 60 sessions, and the deployed book ran >= 60
   of those same sessions ("beside" it);
2. paper gate: the candidate book's latest G4 report (``nse_engine.paper_gate``,
   kept in its Neon state by the daily session) is PASS and covers its latest
   session;
3. 2017-25 Sharpe: the candidate's excess Sharpe over the walk-forward test
   years (2017-2025, as in K5) in its own recorded backtest is at least the
   deployed config's minus 0.05.  One-sided: better than base always passes.
   A fixed configuration's backtest, so not a walk-forward (that re-fits each
   family's settings fold by fold: ``scorecard.WALK_FORWARD_OOS``).

Reported, not gating: PBO and deflated Sharpe with their configuration
counts, the benchmark gate, any holdout evaluation, and both books' paper
returns over the common sessions (a 60-session Sharpe has a standard error
of about 2, so it informs but cannot decide).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

FORWARD_MIN_SESSIONS = 60
WF_SHARPE_TOLERANCE = 0.05
WF_OOS_WINDOW = ("2017-01-01", "2025-12-31")      # the K5 walk-forward's test years
VALIDATION_WINDOW = ("2013-01-01", "2025-12-31")  # full backtest the OOS years are cut from

Check = Tuple[str, bool, str]


def session_dates(sessions: Optional[pd.DataFrame]) -> pd.DatetimeIndex:
    """Distinct session dates of a book's ``paper_sessions`` rows."""
    if sessions is None or sessions.empty or "session_date" not in sessions.columns:
        return pd.DatetimeIndex([])
    d = pd.to_datetime(sessions["session_date"].astype(str).str[:10], errors="coerce").dropna()
    return pd.DatetimeIndex(sorted(set(d.dt.normalize())))


def sessions_check(candidate: pd.DatetimeIndex, base: pd.DatetimeIndex,
                   min_sessions: int = FORWARD_MIN_SESSIONS) -> Check:
    beside = candidate.intersection(base)
    ok = len(candidate) >= min_sessions and len(beside) >= min_sessions
    span = f", {candidate[0].date()}..{candidate[-1].date()}" if len(candidate) else ""
    return ("paper sessions", ok,
            f"{len(candidate)} candidate sessions, {len(beside)} of them beside the deployed book "
            f"(each >= {min_sessions}){span}")


def gate_check(gate: Optional[Dict[str, Any]], last_session: Optional[pd.Timestamp]) -> Check:
    """The candidate book's stored G4 report must be PASS and current."""
    from nse_engine import paper_gate

    if not gate:
        return ("paper gate (G4)", False, "no G4 report stored for the candidate book yet")
    verdict = gate.get("verdict")
    as_of = gate.get("book_last")
    current = last_session is None or (as_of is not None and str(as_of) >= last_session.date().isoformat())
    checks = "; ".join(f"{c['name']} {c['display']} {c['status']}" for c in gate.get("checks", []))
    detail = f"{verdict} as of {as_of}, {gate.get('sessions')} sessions: {checks}"
    if not current:
        detail += f" (STALE: the book's last session is {last_session.date()})"
    return ("paper gate (G4)", verdict == paper_gate.PASS and current, detail)


def oos_sharpe(returns: pd.Series, window: Sequence[str] = WF_OOS_WINDOW, rf_annual: float = 0.065) -> float:
    """Annualised excess Sharpe of daily returns inside ``window``."""
    from nse_engine.validation.dsr import excess_sharpe

    r = pd.Series(returns, dtype="float64").dropna()
    r.index = pd.DatetimeIndex(pd.to_datetime(r.index))
    return float(excess_sharpe(r[window[0]:window[1]], rf_annual))


def wf_check(candidate_sharpe: float, base_sharpe: float, window: Sequence[str] = WF_OOS_WINDOW,
             tolerance: float = WF_SHARPE_TOLERANCE) -> Check:
    floor = base_sharpe - tolerance
    ok = candidate_sharpe == candidate_sharpe and candidate_sharpe >= floor   # NaN fails
    return ("2017-25 Sharpe", ok,
            f"candidate {candidate_sharpe:.3f} vs deployed {base_sharpe:.3f} over {window[0][:4]}-{window[1][:4]} "
            f"(>= {floor:.3f})")


def paper_comparison(candidate_equity: pd.Series, base_equity: pd.Series) -> str:
    """Both books' returns over their common sessions (information only)."""
    c = pd.Series(candidate_equity, dtype="float64").dropna()
    b = pd.Series(base_equity, dtype="float64").dropna()
    c.index = pd.DatetimeIndex(pd.to_datetime(c.index)).normalize()
    b.index = pd.DatetimeIndex(pd.to_datetime(b.index)).normalize()
    common = c.index.intersection(b.index)
    if len(common) < 2:
        return "paper returns: fewer than 2 common sessions"
    rc = c[common].iloc[-1] / c[common].iloc[0] - 1
    rb = b[common].iloc[-1] / b[common].iloc[0] - 1
    return (f"paper returns over {len(common)} common sessions ({common[0].date()}..{common[-1].date()}): "
            f"candidate {rc:+.2%}, deployed {rb:+.2%} (not gating: a 60-session Sharpe has SE ~2)")


def validation_report(validation: Optional[Dict[str, Any]]) -> List[str]:
    """PBO / deflated Sharpe / benchmark lines with their configuration counts."""
    if not validation:
        return ["validation: no validation.json for the candidate's source run"]
    pbo = validation.get("pbo")
    pbo_val = pbo.get("pbo") if isinstance(pbo, dict) else pbo
    n = validation.get("n_configurations")
    dsr = (validation.get("dsr") or {}).get("dsr")
    bench = (validation.get("benchmark_gate") or {}).get("passed")
    out = []
    out.append(f"PBO {pbo_val:.1%} over {n} same-window configurations" if pbo_val is not None
               else f"PBO n/a ({n} configurations)")
    out.append(f"deflated Sharpe {dsr:.3f} at N = {n}" if dsr is not None else "deflated Sharpe n/a")
    out.append(f"benchmark gate passed = {bench}")
    if (validation.get("haircut") or {}).get("line"):
        out.append(validation["haircut"]["line"])
    return out


def decide(checks: Sequence[Check]) -> bool:
    return all(ok for _, ok, _ in checks)


def stored_gate(state: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """The G4 report a book's daily session keeps in its Neon state."""
    from nse_engine import paper_gate

    raw = (state or {}).get(paper_gate.STATE_KEY)
    try:
        return json.loads(raw) if raw else None
    except (TypeError, ValueError):
        return None
