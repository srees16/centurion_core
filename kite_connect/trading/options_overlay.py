"""
The options overlay of a paper book (India tracker A1P, plan 5al): A1, a bear call spread on NIFTY's monthly
options (a 2-SD short call and a 3-SD wing, 15 days out, early exit at the money), beside the deployed rules at
3x the book's NAV.  RT1 (plan 5ak) passed it under GT13 and the options family's 2026 holdout agreed (25 Oct 2026).

A book with ``options_overlay`` in its deployment file is two ledgers:
- its shares: an ordinary engine paper book on the whole ``paper_capital_inr``.  An overlay is not funded from the
  book: its margin is pledged from the book's holdings;
- its options: this ledger.

Each night, after the share book's session, A1's own simulator (``kite_connect.options.backtest.run_sleeve`` with
the recorded O2-A1 configuration) replays from the overlay's paper start to the latest session in the F&O store
the workflow rebuilt that night, on A1's own capital.  So the paper record follows exactly the rules that passed.
The overlay earns ``weight x A1's return`` on the book's previous NAV, as RT1 and the holdout tested
(book + 3 x A1).  With ``max_loss_cap`` (A1C, plan 5am: the owner's sizing, 26 Oct 2026) each cycle's weight is
instead min(weight, cap x A1's equity before the fill / the spread's maximum loss), so one spread can lose at most
that share of the book.  The open spread is reported in lots scaled to the book (A1's lots x its weight x NAV /
A1's equity) with its maximum loss as a share of NAV: the margin the book must pledge.

The ledger appends one row per night and never rewrites one.  An alert is raised when:
- a replay moves a recorded date by more than 0.1% of the book (NSE revised its data);
- the F&O data falls more than four days behind (the nightly sync failing).

A1's paper fills are the simulator's (at the legs' closes, with ``fno_costs``): there is no broker paper for
options, so real fills come with the dry runs.  The ledger is JSON in the book's ``paper_state`` (key
``options_overlay_ledger``).  Paper only: nothing here places an order.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Dict, List, Mapping, Optional

import pandas as pd

logger = logging.getLogger(__name__)

STATE_KEY = "options_overlay_ledger"
DRIFT_ALERT = 0.001
STALE_DAYS = 4
MAX_WEIGHT = 3.0                                          # protocol 5.1's overlay cap
#: Each candidate's recorded run, whose config.json is the configuration paper replays (A1G: beside the
#: walk-forward book, plan 5aj).
RECORDED_RUNS = {"A1": "20261010T122956596793Z_ae503fc4", "A1G": "20261026T173041043859Z_358e73d3"}
CYCLE_KEYS = ("expiry", "decision_date", "action", "short", "wing", "lots", "credit", "max_loss_inr", "fill_date",
              "exit", "exit_date", "pnl_inr")


@dataclass(frozen=True)
class OverlayConfig:
    """The ``options_overlay`` block of a deployment file."""

    candidate: str = "A1"
    weight: float = 3.0
    #: A1C (plan 5am, the owner's sizing for live): each spread's maximum loss at most this share of the book, so a
    #: cycle's weight is min(weight, cap x A1's equity before the fill / its maximum loss); 0 = no cap.
    max_loss_cap: float = 0.0

    @classmethod
    def from_dict(cls, raw: Mapping) -> "OverlayConfig":
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown options_overlay keys: {sorted(unknown)}")
        cfg = cls(**dict(raw))
        if cfg.candidate not in RECORDED_RUNS:
            raise ValueError(f"options_overlay candidate must be one of {sorted(RECORDED_RUNS)}")
        if not (isinstance(cfg.weight, (int, float)) and 0.0 < cfg.weight <= MAX_WEIGHT):
            raise ValueError(f"options_overlay weight must be above 0 and at most {MAX_WEIGHT}")
        if not (isinstance(cfg.max_loss_cap, (int, float)) and 0.0 <= cfg.max_loss_cap < 1.0):
            raise ValueError("options_overlay max_loss_cap must be 0 (none) or a share of the book below 1")
        return cfg


def sleeve_config(cfg: OverlayConfig, start: str, end: str):
    """The candidate's configuration over [start, end]: the code's copy, the same as its recorded run's
    ``config.json`` but for the window (the runner has no research registry)."""
    from kite_connect.options.sleeves import CANDIDATES

    return replace(CANDIDATES[cfg.candidate], start=start, end=end)


@dataclass
class Ledger:
    start: str
    last_session: str = ""
    history: List[Dict] = field(default_factory=list)
    cycles: List[Dict] = field(default_factory=list)      # every cycle since the start, as the replay sees it

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)

    @classmethod
    def from_json(cls, text: str) -> "Ledger":
        return cls(**json.loads(text))


def load_ledger(cloud, start: str) -> Ledger:
    state = cloud.read_state() if cloud is not None else {}
    text = state.get(STATE_KEY) if state else None
    return Ledger.from_json(text) if text else Ledger(start=start)


def save_ledger(cloud, ledger: Ledger) -> None:
    if cloud is None:
        raise RuntimeError("the options overlay needs the book's cloud state (Neon)")
    if not cloud.sync_state({STATE_KEY: ledger.to_json()}):
        raise RuntimeError("could not save the options overlay's ledger")


def _iso(v) -> Optional[str]:
    return pd.Timestamp(v).date().isoformat() if v is not None and not (isinstance(v, float) and math.isnan(v)) else None


def _cycle(c: Dict) -> Dict:
    out = {k: c.get(k) for k in CYCLE_KEYS}
    for k in ("expiry", "decision_date", "fill_date", "exit_date"):
        out[k] = _iso(out[k])
    if out["action"] is None and out["short"] is not None:
        out["action"] = "decided: fills at the next session"
    return out


def cycle_weights(cfg: OverlayConfig, cycles: List[Dict], eq: pd.Series) -> pd.Series:
    """The overlay's weight per F&O session: ``weight`` throughout without a cap; with one, each traded cycle's
    min(weight, cap x A1's equity before its fill / its maximum loss) from its fill to its exit, 0 between cycles
    (A1 holds nothing then).  The same rule as ``nse_engine.diversifier_gate.capped_weights`` (plan 5am)."""
    if not cfg.max_loss_cap:
        return pd.Series(cfg.weight, index=eq.index)
    w = pd.Series(0.0, index=eq.index)
    for c in cycles:
        if c.get("action") != "traded":
            continue
        fill = pd.Timestamp(c["fill_date"])
        exit_ = pd.Timestamp(c["exit_date"]) if c.get("exit_date") is not None else eq.index[-1]
        before = eq[eq.index < fill]
        base = float(before.iloc[-1]) if len(before) else float(eq.iloc[0])
        w[(w.index >= fill) & (w.index <= exit_)] = min(cfg.weight, cfg.max_loss_cap * base / float(c["max_loss_inr"]))
    return w


def run_session(ledger: Ledger, cfg: OverlayConfig, day: pd.Timestamp, share_equity: float, initial: float,
                fo_store_dir: str, guarded: Optional[pd.DatetimeIndex] = None) -> Dict:
    """One night of the overlay; idempotent (a session already done returns its summary).  ``guarded``: for a
    candidate with a crash guard (A1G), the sessions on which it opens no spread (``sleeves.guarded_sessions``)."""
    from kite_connect.options.backtest import run_sleeve

    key = day.date().isoformat()
    if ledger.last_session == key and ledger.history:
        return summary(ledger, cfg, initial, alerts=[])
    scfg = sleeve_config(cfg, ledger.start, key)
    if scfg.entry_guard and guarded is None:
        raise ValueError(f"{cfg.candidate} has a crash guard: the guarded sessions are needed")
    res = run_sleeve(scfg, fo_store_dir, skip_entry_dates=guarded if scfg.entry_guard else None)
    eq = res.equity
    capital = float(scfg.initial_capital)
    asof = eq.index[-1] if len(eq) else pd.Timestamp(ledger.start)
    alerts: List[str] = []
    if len(eq) and (day - asof).days > STALE_DAYS:
        alerts.append(f"F&O data is stale: latest session {asof.date()} (the nightly F&O sync may be failing)")
    drift = max([abs(float(eq.get(pd.Timestamp(r["a1_asof"]), r["a1_equity"])) - r["a1_equity"]) / capital * cfg.weight
                 for r in ledger.history if r.get("a1_asof")] + [0.0])
    if drift > DRIFT_ALERT:
        alerts.append(f"the {cfg.candidate} replay moved {drift:.2%} of the book on a recorded date (NSE revised its data)")
    prev = ledger.history[-1] if ledger.history else None
    a1_now = float(eq.iloc[-1]) if len(eq) else capital
    a1_prev = float(eq.get(pd.Timestamp(prev["a1_asof"]), prev["a1_equity"])) if prev and prev.get("a1_asof") else capital
    a1_ret = a1_now / a1_prev - 1.0 if a1_prev > 0 else 0.0
    nav_prev = prev["nav"] if prev else float(initial)
    weights = cycle_weights(cfg, res.cycles, eq) if len(eq) else pd.Series(dtype="float64")
    daily = eq.pct_change().fillna(float(eq.iloc[0]) / capital - 1.0) if len(eq) else pd.Series(dtype="float64")
    if prev and prev.get("a1_asof"):
        daily = daily[daily.index > pd.Timestamp(prev["a1_asof"])]
    overlay_pnl = float((weights.reindex(daily.index).fillna(0.0) * daily).sum() * nav_prev)
    overlay = (prev["overlay"] if prev else 0.0) + overlay_pnl
    nav = float(share_equity) + overlay
    ledger.cycles = [_cycle(c) for c in res.cycles]
    open_ = next((c for c in reversed(ledger.cycles) if c["action"] == "traded" and not c["exit"]), None)
    open_weight = float(weights.iloc[-1]) if open_ and len(weights) else 0.0
    max_loss = (open_["max_loss_inr"] / a1_now * open_weight) if open_ and a1_now > 0 else 0.0
    ledger.history.append({"date": key, "nav": nav, "share_equity": float(share_equity), "overlay": overlay,
                           "a1_asof": str(asof.date()) if len(eq) else "", "a1_equity": a1_now, "a1_return": a1_ret,
                           "day_pnl": nav - nav_prev, "overlay_pnl": overlay_pnl, "open_weight": open_weight,
                           "max_loss_share": max_loss, "alerts": len(alerts)})
    ledger.last_session = key
    return summary(ledger, cfg, initial, alerts=alerts)


def metrics(ledger: Ledger, initial: float, rf: float = 0.065) -> Dict:
    """The combined book (shares + the overlay) since its first session, and the cycles closed since."""
    if not ledger.history:
        return {}
    h = pd.DataFrame(ledger.history)
    nav = h["nav"].astype(float)
    r = nav.pct_change().fillna(nav.iloc[0] / float(initial) - 1.0)
    eq = (1 + r).cumprod()
    closed = [c for c in ledger.cycles if c.get("pnl_inr") is not None]
    wins = [c["pnl_inr"] for c in closed if c["pnl_inr"] > 0]
    losses = [c["pnl_inr"] for c in closed if c["pnl_inr"] <= 0]
    out = {"sessions": len(r), "return": float(eq.iloc[-1] - 1.0), "max_drawdown": float((eq / eq.cummax() - 1).min()),
           "overlay_pnl": float(h["overlay"].iloc[-1]), "cycles": len(ledger.cycles),
           "traded": sum(1 for c in ledger.cycles if c["action"] == "traded"), "closed": len(closed),
           "hit_rate": len(wins) / len(closed) if closed else None,
           "profit_factor": (sum(wins) / -sum(losses)) if losses and sum(losses) < 0 else None}
    if len(r) >= 20:
        ex = r - rf / 252
        out["sharpe"] = float(ex.mean() / ex.std(ddof=1) * math.sqrt(252)) if ex.std(ddof=1) > 0 else float("nan")
    return out


def summary(ledger: Ledger, cfg: OverlayConfig, initial: float, alerts: List[str]) -> Dict:
    last = ledger.history[-1]
    open_ = next((c for c in reversed(ledger.cycles) if c["action"] == "traded" and not c["exit"]), None)
    decided = next((c for c in reversed(ledger.cycles) if c["action"] == "decided: fills at the next session"), None)
    w_open = last.get("open_weight", cfg.weight)
    scale = w_open * last["nav"] / last["a1_equity"] if last["a1_equity"] > 0 else 0.0
    position = None
    if open_:
        position = {**open_, "book_lots": round((open_["lots"] or 0) * scale, 1), "weight": w_open,
                    "max_loss_share": last["max_loss_share"]}
    return {"candidate": cfg.candidate, "weight": cfg.weight, "max_loss_cap": cfg.max_loss_cap,
            "sizing": (f"max loss ≤ {cfg.max_loss_cap:.0%} of the book" if cfg.max_loss_cap else f"{cfg.weight:g}x"),
            "nav": last["nav"], "overlay": last["overlay"],
            "overlay_pnl": last["overlay_pnl"], "day_pnl": last["day_pnl"], "a1_asof": last["a1_asof"],
            "initial": float(initial), "position": position, "decided": decided,
            "recent_cycles": ledger.cycles[-4:], "alerts": alerts, "metrics": metrics(ledger, initial)}
