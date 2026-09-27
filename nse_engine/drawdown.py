"""
Drawdown rule: exposure control from the book's OWN equity curve.

The regime gate reads the market (NIFTY trend, breadth, VIX); this reads the
account.  Evaluated at every session close, applied to the next decision:

    normal     drawdown from the episode peak <= halt_dd
    halt       > halt_dd      no new entries, no adds; exits and stops still work
    half       > half_dd      core exposure scaled to ``half_scale``, no adds
    risk_off   > risk_off_dd  core scale 0 (cash, or metals if the allocator moves it)

Within an episode the state only worsens.  It re-arms - back to ``normal``
with the peak reset to today's equity, so a new episode starts - when equity
makes a new high over the previous ``rearm_sessions`` sessions.  The rule is a
pure function of the equity history: :func:`replay` over the history from the
start of the book gives today's state, so a live executor needs no stored
state and cannot drift from the backtest.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import numpy as np
import pandas as pd

from nse_engine.config import DrawdownConfig

NORMAL, HALT, HALF, RISK_OFF = "normal", "halt", "half", "risk_off"
STATES = (NORMAL, HALT, HALF, RISK_OFF)
_RANK = {s: i for i, s in enumerate(STATES)}


@dataclass(frozen=True)
class DrawdownDecision:
    """What the rule says for the next decision, given today's close."""

    state: str
    scale: float            # multiplier on core exposure (1, half_scale or 0)
    allow_entries: bool     # may the book open new core names or add to held ones?
    drawdown: float         # 1 - equity / episode peak
    peak: float
    changed: bool           # state differs from the previous session's
    sessions_in_state: int


def severity_for(drawdown: float, cfg: DrawdownConfig) -> str:
    """The worst state a drawdown of this size warrants on its own."""
    if drawdown > cfg.risk_off_dd:
        return RISK_OFF
    if drawdown > cfg.half_dd:
        return HALF
    if drawdown > cfg.halt_dd:
        return HALT
    return NORMAL


def scale_for(state: str, cfg: DrawdownConfig) -> float:
    return {NORMAL: 1.0, HALT: 1.0, HALF: float(cfg.half_scale), RISK_OFF: 0.0}[state]


class DrawdownTracker:
    """Incremental state machine; call :meth:`update` once per session close."""

    def __init__(self, cfg: DrawdownConfig):
        if not (0 < cfg.halt_dd <= cfg.half_dd <= cfg.risk_off_dd):
            raise ValueError("drawdown thresholds must satisfy 0 < halt_dd <= half_dd <= risk_off_dd")
        if not (0.0 <= cfg.half_scale <= 1.0):
            raise ValueError("half_scale must lie in [0, 1]")
        self.cfg = cfg
        self.peak = float("nan")
        self.state = NORMAL
        self._recent: Deque[float] = deque(maxlen=max(int(cfg.rearm_sessions), 1))
        self._in_state = 0

    def update(self, equity: float) -> DrawdownDecision:
        e = float(equity)
        n = max(int(self.cfg.rearm_sessions), 1)
        prev = self.state
        if self.state != NORMAL and len(self._recent) >= n and e > max(self._recent):
            self.state = NORMAL          # re-arm: a new n-session high starts a fresh episode
            self.peak = e
        if self.state == NORMAL:
            self.peak = e if not np.isfinite(self.peak) else max(self.peak, e)
        dd = float(1.0 - e / self.peak) if np.isfinite(self.peak) and self.peak > 0 else 0.0
        worst = severity_for(dd, self.cfg)
        if _RANK[worst] > _RANK[self.state]:
            self.state = worst           # an episode only ever worsens until it re-arms
        changed = self.state != prev
        self._in_state = 1 if changed else self._in_state + 1
        self._recent.append(e)
        return DrawdownDecision(state=self.state, scale=scale_for(self.state, self.cfg),
                                allow_entries=self.state == NORMAL, drawdown=dd, peak=float(self.peak),
                                changed=changed, sessions_in_state=self._in_state)


def replay(equity: pd.Series, cfg: DrawdownConfig) -> pd.DataFrame:
    """Run the rule over a dated equity history; one row per session.

    Columns: equity, peak, drawdown, state, scale, allow_entries, changed.
    The last row is the decision for the next session.
    """
    tracker = DrawdownTracker(cfg)
    rows: List[Dict[str, object]] = []
    for e in pd.Series(equity, dtype="float64").to_numpy():
        d = tracker.update(e)
        rows.append({"equity": float(e), "peak": d.peak, "drawdown": d.drawdown, "state": d.state,
                     "scale": d.scale, "allow_entries": d.allow_entries, "changed": d.changed})
    out = pd.DataFrame(rows, index=pd.Index(equity.index, name="date"))
    return out


def summarise(states: pd.Series) -> Dict[str, float]:
    """Share of sessions in each state and the number of episodes (entries into a non-normal state)."""
    s = pd.Series(states, dtype="object")
    n = max(len(s), 1)
    out: Dict[str, float] = {f"share_{st}": float((s == st).sum()) / n for st in STATES}
    prev_normal = s.shift(1, fill_value=NORMAL) == NORMAL
    out["episodes"] = float(((s != NORMAL) & prev_normal).sum())
    return out
