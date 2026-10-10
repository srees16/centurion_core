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
configuration about to trade has a paper G4 PASS with >= 60 sessions,
>= 5 scheduled dry runs finished clean (no alert: token, tunnel, egress IP,
broker reads and order building all worked), and the account has shown it
can sell unattended (``sell_path_check``: DDPI, a supervised AMO sell and a
triggered GTT sell on this Kite user, tracker LN-T2).  The paper record is the
deployed book's own, or, after a promotion, the trial book's that traded
the same configuration (``go_live_evidence``, tracker V5), so promoting a
trial that cleared the forward gate does not restart the go-live clock.
After a configuration change while live, the live G4 restarts its window at
the change (``config_since``), so the old configuration's weeks never read
as tracking error.  Dry-run orders are not compared with the paper
book's: a dry run records nothing, so every night it plans a fresh Rs 6 lakh
book, while the paper book only trades changes to what it holds.
``CENTURION_GO_LIVE_OVERRIDE=true`` overrides, and the email says so.  U2
(Kaggle token) and L4 (no leverage) stay manual checks.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

RUNGS: Tuple[float, ...] = (600_000.0, 1_200_000.0, 2_100_000.0, 3_000_000.0)
MIN_SESSIONS_PER_RUNG = 20            # about one month of sessions
KILL_DD_MULTIPLE = 1.5
# MaxDD of the deployed configuration's backtest, 2013-25 (679cbd0c, cost model 4;
# tracker IC1b, was 0.247 from cost model 1): the fallback of backtest_maxdd_for,
# which reads the books register so a promotion moves the kill threshold with it.
BACKTEST_MAXDD_DEFAULT = 0.239
GO_LIVE_MIN_PAPER_SESSIONS = 60
GO_LIVE_MIN_DRY_RUNS = 5
STATE_KEY = "live_ladder"
DRY_RUNS_KEY = "live_dry_runs"
#: The supervised sell test, as verified against Kite (tracker LN-T2): without DDPI every CNC sell
#: needs a same-day CDSL TPIN, which no unattended session can give.
SELL_PATH_KEY = "live_sell_path"
LIVE_CONFIG_KEY = "live_config"       # {"config_hash", "since"}: what the live book trades, since when (V5)
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


def backtest_maxdd_for(config_hash: str) -> float:
    """The kill threshold's backtest MaxDD (positive) for the configuration live trades.

    ``CENTURION_BACKTEST_MAXDD`` when set; else the books register's like-for-like
    2013-25 MaxDD of that configuration (``nse_engine.books register`` re-scores
    it after a promotion or a re-baseline); else ``BACKTEST_MAXDD_DEFAULT``.
    """
    if os.environ.get("CENTURION_BACKTEST_MAXDD"):
        return float(os.environ["CENTURION_BACKTEST_MAXDD"])
    try:
        from nse_engine.books import read_register

        reg = read_register()
        dd = pd.to_numeric(reg.loc[reg["config_hash"] == config_hash, "bt_max_dd"], errors="coerce").dropna()
        if len(dd) and dd.iloc[0] != 0:
            return abs(float(dd.iloc[0]))
        logger.warning("books register has no backtest MaxDD for %s: kill threshold from %.1f%%",
                       config_hash[:8], BACKTEST_MAXDD_DEFAULT * 100)
    except Exception as exc:                              # noqa: BLE001 - the constant still guards
        logger.warning("books register unreadable (%s): kill threshold from %.1f%%", exc, BACKTEST_MAXDD_DEFAULT * 100)
    return BACKTEST_MAXDD_DEFAULT


def evaluate(state: LadderState, session: str, *, gate: Optional[Dict[str, Any]], drawdown_state: str,
             book_dd: float, nifty_dd: float, shift_multipliers: Sequence[float],
             requested_capital: Optional[float], backtest_maxdd: Optional[float] = None,
             capital_setting: str = "CENTURION_LIVE_CAPITAL") -> LadderDecision:
    """Tonight's ladder decision; ``state`` is updated in place (call once per session).

    ``capital_setting`` names where the capital is set, in the advice (a
    family account's is on the Fly Kite page, tracker FA2).
    """
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
        d.alerts.append(f"{capital_setting}={requested_capital:,.0f} is not a ladder rung: ignored")

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
                            f"Set {capital_setting}={d.capital:.0f} to match.")
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
        d.reasons.append(f"GO allowed: set {capital_setting}={nxt:.0f} once the money is in the account")
        if not state.eligible_since:
            state.eligible_since = session
            d.alerts.append(f"Capital ladder: GO to Rs {nxt:,.0f} is allowed. Add the money to the account, "
                            f"then set {capital_setting}={nxt:.0f}.")
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

def _gate_sessions(gate: Optional[Dict[str, Any]]) -> int:
    return int((gate or {}).get("sessions") or 0)


def go_live_evidence(own_gate: Optional[Dict[str, Any]], config_hash: str,
                     trial_gates: Optional[Dict[str, Dict[str, Any]]] = None) -> Tuple[Optional[Dict[str, Any]], str]:
    """(G4 report, its source): the paper record of the configuration that will trade live (tracker V5).

    The deployed paper book's own report when it judged ``config_hash`` (a
    report stored before reports carried the hash counts as the deployed
    book's) and has the sessions.  Otherwise a trial book that paper-traded
    the same configuration: PASS, the sessions, and current (its latest
    session not older than the deployed book's).  A promotion rewrites the
    deployed book's paper start, so its own count restarts; the promoted
    configuration's trial sessions are the evidence go-live needs, and the
    clock does not restart with it.  An old configuration's report never
    counts for a new one.
    """
    own = own_gate if own_gate and own_gate.get("config_hash", config_hash) == config_hash else None
    if own and _gate_sessions(own) >= GO_LIVE_MIN_PAPER_SESSIONS:
        return own, "deployed paper book"
    latest = str((own_gate or {}).get("book_last") or "")
    for name, gate in sorted((trial_gates or {}).items()):
        if (gate and gate.get("config_hash") == config_hash and gate.get("verdict") == "PASS"
                and _gate_sessions(gate) >= GO_LIVE_MIN_PAPER_SESSIONS and str(gate.get("book_last") or "") >= latest):
            return gate, f"trial book '{name}' (same configuration)"
    return own, "deployed paper book"


def config_since(raw: Optional[str], config_hash: str, first_session: str,
                 session: str) -> Tuple[str, Optional[str], Optional[str]]:
    """(first live session on ``config_hash``, marker to store or None, the configuration it replaces or None).

    The live G4 compares only the sessions since the live book began trading
    its current configuration with a backtest of that configuration (tracker
    V5): after a promotion, the weeks traded on the old configuration would
    otherwise read as tracking error and step the ladder down.  Without a
    marker the configuration has traded since the book's first session.
    """
    try:
        stored = json.loads(raw) if raw else {}
    except ValueError:
        stored = {}
    if stored.get("config_hash") == config_hash and stored.get("since"):
        return str(stored["since"]), None, None
    since = session if stored.get("config_hash") else first_session
    return since, json.dumps({"config_hash": config_hash, "since": since}), stored.get("config_hash")


def sell_path_check(record: Optional[Dict[str, Any]], user_id: Optional[str]) -> Tuple[bool, str]:
    """Whether this Kite account has shown it can sell with no one present (tracker LN-T2).

    ``record`` is the supervised test as ``live_session --record-sell-path``
    verified it: DDPI confirmed, an AMO SELL (CNC) COMPLETE and a stop GTT
    triggered with its SELL COMPLETE, all on the Kite user now logged in.
    """
    r = record or {}
    if not r:
        return False, "no supervised sell test recorded (live_session --record-sell-path)"
    problems = []
    if not r.get("ddpi_confirmed_on"):
        problems.append("DDPI not confirmed")
    if not user_id or r.get("user_id") != user_id:
        problems.append(f"recorded for Kite user {r.get('user_id') or '?'}, logged in as {user_id or '?'}")
    if r.get("amo_status") != "COMPLETE" or r.get("amo_side") != "SELL":
        problems.append(f"AMO sell {r.get('amo_side') or '?'} {r.get('amo_status') or 'missing'}")
    if r.get("gtt_status") != "triggered" or r.get("gtt_order_status") != "COMPLETE":
        problems.append(f"GTT sell {r.get('gtt_status') or 'missing'}, its order {r.get('gtt_order_status') or '-'}")
    if problems:
        return False, "; ".join(problems)
    return True, (f"DDPI confirmed {r['ddpi_confirmed_on']}; AMO sell and GTT sell COMPLETE "
                  f"(verified {str(r.get('verified_at') or '')[:10]})")


def readiness(paper_gate: Optional[Dict[str, Any]], dry_runs: Sequence[Dict[str, Any]],
              source: str = "deployed paper book", sell_path: Optional[Tuple[bool, str]] = None
              ) -> List[Tuple[str, bool, str]]:
    """Go-live checks: (name, ok, detail).  ``paper_gate`` comes from :func:`go_live_evidence`;
    ``sell_path`` from :func:`sell_path_check` (missing counts as not shown)."""
    out = []
    v, n = (paper_gate or {}).get("verdict"), _gate_sessions(paper_gate)
    out.append(("paper book G4", v == "PASS" and n >= GO_LIVE_MIN_PAPER_SESSIONS,
                f"{source}: {v or 'no report'}, {n} sessions (PASS at >= {GO_LIVE_MIN_PAPER_SESSIONS} needed)"))
    clean = [r for r in dry_runs if r.get("clean")]
    out.append(("live dry runs", len(clean) >= GO_LIVE_MIN_DRY_RUNS,
                f"{len(clean)} clean of {len(dry_runs)} scheduled dry runs ({GO_LIVE_MIN_DRY_RUNS} needed)"))
    ok, detail = sell_path if sell_path is not None else (False, "not checked")
    out.append(("sell path", ok, detail))
    return out
