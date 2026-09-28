"""
Paper pass/fail gate (tracker G4): does a paper book behave like its backtest?

A paper book is compared with the same-period backtest of what it trades
(``run_nse_engine shift-reference``: the deployed engine config plus the
deployment's drawdown overlay, from the book's first session, same capital)
on five checks.  The limits were fixed on 28 Sep 2026, before either book had
the sample, and must not be tuned on paper results:

=============== ==================================================== ================== ===================
check           measure                                              PASS               FAIL
=============== ==================================================== ================== ===================
tracking error  annualised sd of (paper - backtest) daily returns    <= 8% a year       > 12% a year
daily gap       mean of (paper - backtest), bp a day, and its t      >= -3 bp           < -3 bp and t <= -2
costs           paper cost per rupee traded / the backtest's, same   <= 1.5x            > 2.0x
                days; needs >= 20 paper fills
drawdown        paper MaxDD vs the backtest's MaxDD, same days       <= max(1.5x, +2    > max(2x, +4 pts)
                                                                     pts)
regime break    sessions sized down by a drift "regime_break"        none               any
                (multiplier <= 0.5) among the last 20
=============== ==================================================== ================== ===================

Between the two limits a check is WATCH.  Overall: NOT ENOUGH DATA below 30
aligned sessions; otherwise FAIL if any check fails, PASS only if all five
pass, else WATCH (a check between its limits, or one not measurable yet).
The drift check behind "regime break" runs from session 31, so before that
it reads "none".

Paper proves behaviour, not the Sharpe: over 60 sessions a Sharpe estimate
has a standard error of about 2, so no return threshold is gated here.  In
paper the cost check confirms the fill simulator matches the backtest's cost
model; with live fills (L5) it measures real slippage against the model.

    python -m runners.run_nse_engine paper-gate                       # deployed book
    python -m runners.run_nse_engine paper-gate --deployment config/nse_engine_candidate.json --schema candidate
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

MIN_SESSIONS = 30
TE_PASS, TE_FAIL = 0.08, 0.12
GAP_BPS, GAP_T = 3.0, 2.0
COST_PASS, COST_FAIL, MIN_FILLS = 1.5, 2.0, 20
DD_PASS_X, DD_PASS_PTS = 1.5, 0.02
DD_FAIL_X, DD_FAIL_PTS = 2.0, 0.04
BREAK_LOOKBACK, BREAK_MULTIPLIER = 20, 0.5

PASS, WATCH, FAIL, NA = "PASS", "WATCH", "FAIL", "n/a"
NOT_ENOUGH = "NOT ENOUGH DATA"
REFERENCE_TRADES_SUFFIX = "_trades.csv"
STATE_KEY = "paper_gate"   # the latest report, kept in the book's Neon state for the weekly email


def reference_trades_path(returns_csv: Union[str, Path]) -> Path:
    """Trades file written next to a shift-reference returns CSV."""
    p = Path(returns_csv)
    return p.with_name(p.stem + REFERENCE_TRADES_SUFFIX)


def read_reference(returns_csv: Union[str, Path]) -> tuple:
    """(daily returns, trades or None) of a same-period reference backtest."""
    p = Path(returns_csv)
    df = pd.read_csv(p)
    col = "return" if "return" in df.columns else df.columns[-1]
    r = pd.Series(df[col].astype("float64").to_numpy(), index=pd.DatetimeIndex(pd.to_datetime(df.iloc[:, 0])))
    tp = reference_trades_path(p)
    trades = pd.read_csv(tp) if tp.exists() else None
    return r, trades


def _dated(s: pd.Series) -> pd.Series:
    s = pd.Series(s, dtype="float64").dropna()
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index)).normalize()
    return s[~s.index.duplicated(keep="last")].sort_index()


def _max_dd(returns: pd.Series) -> float:
    """Largest peak-to-trough loss (a positive fraction) of a return stream,
    starting from its opening value."""
    if returns.empty:
        return 0.0
    e = np.concatenate([[1.0], np.cumprod(1.0 + returns.to_numpy())])
    peak = np.maximum.accumulate(e)
    return float(np.max(1.0 - e / peak))


def _fill_costs(fills: pd.DataFrame) -> tuple:
    """(cost INR, traded value INR, number of fills) of filled paper orders.

    Open fills record impact + statutory costs in ``costs_inr``; stop exits
    carry the impact in the exit price and only statutory costs in
    ``costs_inr``, so their impact is added back from ``impact_bps``.
    """
    f = fills.copy()
    f = f[(f.get("status", "").astype(str).str.upper() == "FILLED") & (pd.to_numeric(f["quantity"]) > 0)]
    if f.empty:
        return 0.0, 0.0, 0
    qty = pd.to_numeric(f["quantity"]).astype(float)
    px = pd.to_numeric(f["fill_price"]).astype(float)
    value = qty * px
    cost = pd.to_numeric(f["costs_inr"]).fillna(0.0).astype(float)
    stops = f.get("source", pd.Series("", index=f.index)).astype(str) == "stop"
    bps = pd.to_numeric(f.get("impact_bps", 0.0)).fillna(0.0).astype(float)
    cost = cost + np.where(stops, value * bps / 1e4, 0.0)
    return float(cost.sum()), float(value.sum()), int(len(f))


def _check(name: str, value: Optional[float], status: str, display: str, limits: str, note: str = "") -> Dict[str, Any]:
    return {"name": name, "value": value, "status": status, "display": display, "limits": limits, "note": note}


def evaluate(live_equity: pd.Series, reference_returns: pd.Series, *,
             fills: Optional[pd.DataFrame] = None, reference_trades: Optional[pd.DataFrame] = None,
             sessions: Optional[pd.DataFrame] = None,
             min_sessions: int = MIN_SESSIONS) -> Dict[str, Any]:
    """Gate report for one paper book.

    ``live_equity``: end-of-session equity by date.  ``reference_returns``:
    the same-period backtest's daily returns.  ``fills``: the book's
    ``paper_fills`` rows; ``reference_trades``: the backtest's trades
    (``date``, ``value_inr``, ``cost_inr``); ``sessions``: the book's
    ``paper_sessions`` rows (``session_date``, ``shift_multiplier``).  Days are matched on dates both
    series have; a reference that ends early shortens the window.
    """
    eq = _dated(live_equity)
    live = eq.pct_change().dropna()
    ref = _dated(reference_returns)
    aligned = pd.concat({"live": live, "ref": ref}, axis=1, join="inner").dropna()
    n = int(len(aligned))
    report: Dict[str, Any] = {
        "sessions": n, "min_sessions": int(min_sessions), "book_sessions": int(len(eq)),
        "book_last": eq.index[-1].date().isoformat() if len(eq) else None,
        "reference_last": ref.index[-1].date().isoformat() if len(ref) else None,
        "window": [aligned.index[0].date().isoformat(), aligned.index[-1].date().isoformat()] if n else None,
        "checks": [],
    }
    if n < 2:
        report["verdict"] = NOT_ENOUGH
        return report
    diff = aligned["live"] - aligned["ref"]
    sd = float(diff.std(ddof=1))
    checks: List[Dict[str, Any]] = []

    te = sd * math.sqrt(252)
    checks.append(_check("tracking error", te, PASS if te <= TE_PASS else FAIL if te > TE_FAIL else WATCH,
                         f"{te:.1%}/yr", f"PASS <= {TE_PASS:.0%}, FAIL > {TE_FAIL:.0%}"))

    gap = float(diff.mean()) * 1e4
    if abs(gap) < 1e-6:            # float dust on identical series
        t = 0.0
    else:                          # a constant gap has no noise: its t is capped, not infinite
        t = float(np.clip(diff.mean() / (sd / math.sqrt(n)), -99.0, 99.0)) if sd > 1e-12 else math.copysign(99.0, gap)
    gs = PASS if gap >= -GAP_BPS else FAIL if t <= -GAP_T else WATCH
    checks.append(_check("daily gap", gap, gs, f"{gap:+.1f} bp/day (t {t:+.1f})",
                         f"PASS >= -{GAP_BPS:g} bp, FAIL < -{GAP_BPS:g} bp with t <= -{GAP_T:g}"))

    lo, hi = aligned.index[0], aligned.index[-1]
    cost_limits = f"PASS <= {COST_PASS:g}x, FAIL > {COST_FAIL:g}x, needs {MIN_FILLS} fills"
    ratio = None
    if fills is not None and not fills.empty and reference_trades is not None and not reference_trades.empty:
        f = fills.copy()
        d = pd.to_datetime(f["session_date"].astype(str).str[:10], errors="coerce")
        f = f[(d >= lo) & (d <= hi)]
        pc, pv, nf = _fill_costs(f)
        rt = reference_trades.copy()
        rd = pd.to_datetime(rt["date"], errors="coerce")
        rt = rt[(rd >= lo) & (rd <= hi)]
        rv = float(pd.to_numeric(rt["value_inr"]).sum())
        rc = float(pd.to_numeric(rt["cost_inr"]).sum())
        if nf < MIN_FILLS or pv <= 0 or rv <= 0 or rc <= 0:
            checks.append(_check("costs", None, NA, f"{nf} of {MIN_FILLS} fills", cost_limits))
        else:
            ratio = (pc / pv) / (rc / rv)
            checks.append(_check("costs", ratio, PASS if ratio <= COST_PASS else FAIL if ratio > COST_FAIL else WATCH,
                                 f"{ratio:.2f}x ({pc / pv * 1e4:.0f} vs {rc / rv * 1e4:.0f} bp, {nf} fills)",
                                 cost_limits))
    else:
        missing = "no paper fills" if fills is None or fills.empty else "no reference trades"
        checks.append(_check("costs", None, NA, missing, cost_limits))

    ldd, rdd = _max_dd(aligned["live"]), _max_dd(aligned["ref"])
    p_lim, f_lim = max(DD_PASS_X * rdd, rdd + DD_PASS_PTS), max(DD_FAIL_X * rdd, rdd + DD_FAIL_PTS)
    checks.append(_check("drawdown", ldd, PASS if ldd <= p_lim else FAIL if ldd > f_lim else WATCH,
                         f"{ldd:.1%} vs {rdd:.1%} backtest",
                         f"PASS <= {p_lim:.1%}, FAIL > {f_lim:.1%}"))

    b_limits = f"PASS none, FAIL any, last {BREAK_LOOKBACK} sessions"
    if sessions is None or sessions.empty or "shift_multiplier" not in sessions.columns:
        checks.append(_check("regime break", None, NA, "no session records", b_limits))
    else:
        s = sessions.sort_values("session_date").tail(BREAK_LOOKBACK)
        mult = pd.to_numeric(s["shift_multiplier"], errors="coerce").fillna(1.0)
        breaks = int((mult <= BREAK_MULTIPLIER + 1e-9).sum())
        checks.append(_check("regime break", float(breaks), PASS if breaks == 0 else FAIL,
                             "none" if breaks == 0 else f"{breaks} of the last {len(s)} sessions", b_limits))

    report["checks"] = checks
    statuses = [c["status"] for c in checks]
    if n < min_sessions:
        report["verdict"] = NOT_ENOUGH
    elif FAIL in statuses:
        report["verdict"] = FAIL
    elif all(s == PASS for s in statuses):
        report["verdict"] = PASS
    else:
        report["verdict"] = WATCH
    return report


def headline(report: Dict[str, Any]) -> str:
    """The verdict with its sample, e.g. 'NOT ENOUGH DATA (9 of 30 sessions)'."""
    v = report.get("verdict", NOT_ENOUGH)
    n, m = report.get("sessions", 0), report.get("min_sessions", MIN_SESSIONS)
    return f"{v} ({n} of {m} sessions)" if v == NOT_ENOUGH else f"{v} ({n} sessions)"


def one_line(report: Dict[str, Any]) -> str:
    """Verdict plus every check on one line, for the daily email."""
    parts = [headline(report)]
    for c in report.get("checks", []):
        tag = "" if c["status"] in (PASS, NA) or report.get("verdict") == NOT_ENOUGH else f" {c['status']}"
        parts.append(f"{c['name']} {c['display']}{tag}")
    return " · ".join(parts)


def format_report(report: Dict[str, Any], title: str = "Paper gate (G4)") -> str:
    """Multi-line text report for the command line."""
    lines = [f"{title}: {headline(report)}"]
    if report.get("window"):
        lines.append(f"  window {report['window'][0]} .. {report['window'][1]}; book last session "
                     f"{report.get('book_last')}, reference last {report.get('reference_last')}")
    if report.get("book_last") and report.get("reference_last") and report["reference_last"] < report["book_last"]:
        lines.append("  NOTE: the reference ends before the book: refresh the store to cover every session")
    for c in report.get("checks", []):
        lines.append(f"  {c['name']:<15} {c['status']:<6} {c['display']:<42} {c['limits']}")
    if report.get("verdict") == NOT_ENOUGH:
        lines.append(f"  Checks are shown for information; the gate decides from {report.get('min_sessions')} sessions.")
    return "\n".join(lines)


def summary_json(report: Dict[str, Any], **extra: Any) -> str:
    """Compact JSON kept in the book's state (Neon) for the weekly report."""
    keep = {k: report.get(k) for k in ("verdict", "sessions", "min_sessions", "window", "book_last", "reference_last")}
    keep["checks"] = [{k: c[k] for k in ("name", "status", "display")} for c in report.get("checks", [])]
    keep.update(extra)
    return json.dumps(keep, default=str)
