"""
Capital ladder for the live book (tracker D3, decisions U6 and U22).

Live capital grows in four rungs of the Rs 30 lakh decided in U6, each held
for at least a month, and falls back automatically when the book stops
behaving like its backtest:

    rung   capital       share
    1      Rs  6,00,000   20%
    2      Rs 12,00,000   40%
    3      Rs 21,00,000   70%
    4      Rs 30,00,000  100%

Every live session evaluates the ladder on the live book's own paper gate
(G4 against its same-period backtest), the drawdown rule, and NIFTY:

* STEP DOWN (automatic, one rung, applied before tonight's orders): after at
  least ``MIN_SESSIONS_PER_RUNG`` sessions at the rung, any G4 check at FAIL.
  G4 compares the book with a backtest of the same days, so a market-wide
  crash that the backtest also suffers is not a reason (U22); behaving unlike
  the backtest is.
* GO (one rung up, only when you ask): at least ``MIN_SESSIONS_PER_RUNG``
  sessions at the rung, tracking error, daily gap, drawdown and regime-break
  checks all PASS (or not measurable yet), the drawdown rule "normal", no
  kill criterion.  Adding capital means adding money, so the step happens
  only when ``CENTURION_LIVE_CAPITAL`` is set to the next rung (after the
  money is in the account); the email says when that is allowed.
* HOLD otherwise.  Setting ``CENTURION_LIVE_CAPITAL`` to a lower rung steps
  down at once (a withdrawal is always allowed).
* KILL criteria (a human decision, alerted, never automatic; the ladder is
  frozen while one holds): (1) the book's drawdown exceeds 1.5x the backtest
  MaxDD AND NIFTY's drawdown over the same days (in a market-wide crash the
  book is judged against the market, U22); (2) regime_break in two
  consecutive sessions; (3) G4's cost check FAIL (costs above 2x the model).
  The response is ``CENTURION_KILL_SWITCH=true``, which refuses new buys.

A step changes the live ledger's capital and cash by the difference and is
recorded as a flow on that day's snapshot.  ``flow_adjusted`` removes flows
from the equity history, so a withdrawal never reads as a drawdown and a
deposit never as a gain (drawdown rule, G4, max drawdown).

Go-live (``readiness``): the first real session is refused unless the
deployed paper book's G4 is PASS with >= 60 sessions and >= 5 scheduled dry
runs finished clean (no alert: token, tunnel, egress IP, broker reads and
order building all worked).  Dry-run orders are not compared with the paper
book's: a dry run records nothing, so every night it plans a fresh Rs 6 lakh
book, while the paper book only trades changes to what it holds.
``CENTURION_GO_LIVE_OVERRIDE=true`` overrides, and the email says so.  U2
(Kaggle token) and L4 (no leverage) stay manual checks.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

RUNGS: Tuple[float, ...] = (600_000.0, 1_200_000.0, 2_100_000.0, 3_000_000.0)
MIN_SESSIONS_PER_RUNG = 20            # about one month of sessions
KILL_DD_MULTIPLE = 1.5
# MaxDD of the deployed configuration's backtest, 2013-25 on the honest data
# (679cbd0c, B1-era data; tracker section 1).  Override when the deployment
# changes: CENTURION_BACKTEST_MAXDD=0.234 (candidate 2d64ba4c), for example.
BACKTEST_MAXDD_DEFAULT = 0.247
GO_LIVE_MIN_PAPER_SESSIONS = 60
GO_LIVE_MIN_DRY_RUNS = 5
STATE_KEY = "live_ladder"
DRY_RUNS_KEY = "live_dry_runs"
GO, HOLD, STEP_DOWN = "GO", "HOLD", "STEP DOWN"
GO_CHECKS = ("tracking error", "daily gap", "drawdown", "regime break")


def rung_of(capital: float) -> Optional[int]:
    """0-based rung index of ``capital``, or None when it is not a rung."""
    for i, c in enumerate(RUNGS):
        if abs(float(capital) - c) < 1.0:
            return i
    return None


@dataclass
class LadderState:
    rung: int = 0
    since: str = ""                     # session the rung started
    sessions: int = 0                   # sessions completed at this rung
    last_session: str = ""
    eligible_since: str = ""            # first session GO became allowed (alert once)
    history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def capital(self) -> float:
        return RUNGS[self.rung]

    @classmethod
    def load(cls, raw: Optional[str], capital: float, session: str) -> "LadderState":
        if raw:
            d = json.loads(raw)
            return cls(**{k: d[k] for k in ("rung", "since", "sessions", "last_session", "eligible_since", "history")
                          if k in d})
        r = rung_of(capital)
        if r is None:
            raise ValueError(f"live capital {capital:,.0f} is not a ladder rung {[f'{c:,.0f}' for c in RUNGS]}")
        return cls(rung=r, since=session, history=[{"session": session, "event": "start", "capital": RUNGS[r]}])

    def dump(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class LadderDecision:
    action: str                         # GO | HOLD | STEP DOWN
    rung: int                           # rung after the decision
    capital: float
    flow: float                         # capital change applied today (+ deposit, - withdrawal)
    reasons: List[str] = field(default_factory=list)
    kill: List[str] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)

    def line(self) -> str:
        head = f"{self.action} · rung {self.rung + 1} of {len(RUNGS)}, Rs {self.capital:,.0f}"
        return head + (f" · {'; '.join(self.reasons)}" if self.reasons else "")


def _status(gate: Optional[Dict[str, Any]], name: str) -> Optional[str]:
    for c in (gate or {}).get("checks", []):
        if c.get("name") == name:
            return c.get("status")
    return None


def evaluate(state: LadderState, session: str, *, gate: Optional[Dict[str, Any]], drawdown_state: str,
             book_dd: float, nifty_dd: float, shift_multipliers: Sequence[float],
             requested_capital: Optional[float], backtest_maxdd: Optional[float] = None) -> LadderDecision:
    """Tonight's ladder decision; ``state`` is updated in place (call once per session)."""
    backtest_maxdd = float(backtest_maxdd if backtest_maxdd is not None
                           else os.environ.get("CENTURION_BACKTEST_MAXDD", BACKTEST_MAXDD_DEFAULT))
    if state.last_session != session:
        state.sessions += 1
        state.last_session = session
    checks = (gate or {}).get("checks", [])
    fails = [f"{c['name']} {c.get('display', '')}".strip() for c in checks if c.get("status") == "FAIL"]

    kill: List[str] = []
    if book_dd > KILL_DD_MULTIPLE * backtest_maxdd and book_dd > nifty_dd:
        kill.append(f"drawdown {book_dd:.1%} > {KILL_DD_MULTIPLE}x the backtest's {backtest_maxdd:.1%} "
                    f"and worse than NIFTY's {nifty_dd:.1%}")
    last2 = list(shift_multipliers)[-2:]
    if len(last2) == 2 and all(m <= 0.5 + 1e-9 for m in last2):
        kill.append("regime_break in two consecutive sessions")
    if _status(gate, "costs") == "FAIL":
        kill.append("costs above 2x the model (G4)")

    d = LadderDecision(HOLD, state.rung, state.capital, 0.0, kill=kill)
    if kill:
        d.alerts.append("KILL criterion met (your decision): " + "; ".join(kill)
                        + ". Set CENTURION_KILL_SWITCH=true to refuse new buys. The ladder is frozen.")

    req = rung_of(requested_capital) if requested_capital else None
    if requested_capital and req is None:
        d.alerts.append(f"CENTURION_LIVE_CAPITAL={requested_capital:,.0f} is not a ladder rung: ignored")

    def move(to: int, action: str, why: str) -> None:
        d.flow = RUNGS[to] - state.capital
        state.history.append({"session": session, "event": action, "from": state.capital, "to": RUNGS[to],
                              "why": why})
        state.rung, state.since, state.sessions, state.eligible_since = to, session, 0, ""
        d.action, d.rung, d.capital = action, to, RUNGS[to]
        d.reasons.append(why)

    if req is not None and req < state.rung:                       # your withdrawal: always allowed
        move(req, STEP_DOWN, f"you lowered the capital to Rs {RUNGS[req]:,.0f}")
        return d
    if fails and state.sessions >= MIN_SESSIONS_PER_RUNG:
        if state.rung > 0:
            move(state.rung - 1, STEP_DOWN, "G4 FAIL: " + "; ".join(fails))
            d.alerts.append(f"LADDER STEP DOWN to Rs {d.capital:,.0f}: {'; '.join(fails)}. "
                            f"Set CENTURION_LIVE_CAPITAL={d.capital:.0f} to match.")
        else:
            d.reasons.append("G4 FAIL at the lowest rung: " + "; ".join(fails))
            d.alerts.append("G4 FAIL at the lowest rung: consider the kill switch. " + "; ".join(fails))
        return d

    blockers = []
    if state.sessions < MIN_SESSIONS_PER_RUNG:
        blockers.append(f"{state.sessions} of {MIN_SESSIONS_PER_RUNG} sessions at this rung")
    bad = [n for n in GO_CHECKS if _status(gate, n) not in (None, "PASS", "n/a")]
    if gate is None:
        blockers.append("no live G4 report yet")
    if bad:
        blockers.append("G4 not PASS: " + ", ".join(bad))
    if drawdown_state != "normal":
        blockers.append(f"drawdown rule {drawdown_state}")
    if kill:
        blockers.append("a kill criterion holds")
    top = state.rung >= len(RUNGS) - 1

    if req is not None and req > state.rung:
        if req != state.rung + 1:
            d.alerts.append(f"requested Rs {RUNGS[req]:,.0f} skips a rung: the next is Rs {RUNGS[state.rung + 1]:,.0f}")
        elif blockers:
            d.alerts.append(f"requested Rs {RUNGS[req]:,.0f} refused for now: " + "; ".join(blockers))
        else:
            move(req, GO, f"GO: {MIN_SESSIONS_PER_RUNG}+ sessions, G4 and the drawdown rule clear")
            return d
    if top:
        d.reasons.append("top rung")
    elif blockers:
        d.reasons.append("; ".join(blockers))
    else:
        nxt = RUNGS[state.rung + 1]
        d.reasons.append(f"GO allowed: set CENTURION_LIVE_CAPITAL={nxt:.0f} once the money is in the account")
        if not state.eligible_since:
            state.eligible_since = session
            d.alerts.append(f"Capital ladder: GO to Rs {nxt:,.0f} is allowed. Add the money to the account, "
                            f"then set CENTURION_LIVE_CAPITAL={nxt:.0f}.")
    return d


# ── flows: deposits and withdrawals are not returns ───────────────

def flow_adjusted(equity: pd.Series, flows: Optional[pd.Series] = None) -> pd.Series:
    """Equity history with the ladder's flows taken out, in today's capital base.

    ``A[T] = E[T]`` and ``A[t-1] = A[t] * E[t-1] / (E[t] - flow[t])``, so
    ``A[t] / A[t-1] - 1`` is the day's return net of that day's flow.
    """
    e = pd.Series(equity, dtype="float64").dropna().sort_index()
    if flows is None or e.empty:
        return e
    f = pd.Series(flows, dtype="float64").reindex(e.index).fillna(0.0)
    if not f.abs().gt(0).any():
        return e
    vals = e.to_numpy().copy()
    out = vals.copy()
    for i in range(len(vals) - 1, 0, -1):
        denom = vals[i] - f.iloc[i]
        out[i - 1] = out[i] * vals[i - 1] / denom if denom > 0 else out[i]
    return pd.Series(out, index=e.index)


def current_drawdown(series: pd.Series) -> float:
    """Drawdown of the last value from the series' peak (a positive fraction)."""
    s = pd.Series(series, dtype="float64").dropna()
    return float(1.0 - s.iloc[-1] / s.max()) if len(s) and s.max() > 0 else 0.0


# ── go-live ──────────────────────────────────────────────────────

def readiness(paper_gate: Optional[Dict[str, Any]], dry_runs: Sequence[Dict[str, Any]]) -> List[Tuple[str, bool, str]]:
    """Go-live checks: (name, ok, detail)."""
    out = []
    v, n = (paper_gate or {}).get("verdict"), int((paper_gate or {}).get("sessions") or 0)
    out.append(("paper book G4", v == "PASS" and n >= GO_LIVE_MIN_PAPER_SESSIONS,
                f"{v or 'no report'}, {n} sessions (PASS at >= {GO_LIVE_MIN_PAPER_SESSIONS} needed)"))
    clean = [r for r in dry_runs if r.get("clean")]
    out.append(("live dry runs", len(clean) >= GO_LIVE_MIN_DRY_RUNS,
                f"{len(clean)} clean of {len(dry_runs)} scheduled dry runs ({GO_LIVE_MIN_DRY_RUNS} needed)"))
    return out
