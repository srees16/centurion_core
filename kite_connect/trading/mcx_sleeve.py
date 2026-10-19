"""
The MCX trend sleeve of a paper book (India tracker DL1, plan 5ai): GT5-B, your second drawdown lever (19 Oct
2026; provisional, so paper is its test).

A book with ``mcx_sleeve`` in its deployment file is two ledgers:
- its shares: an ordinary engine paper book (``PaperTrader``) on ``paper_capital_inr`` less the sleeve's weight;
- its MCX trend: this ledger, at the sleeve's capital.

Each night, after the share book's session, GT5-B's own simulator (``nse_engine.mcx_trend.run_trend``) runs from
the paper start to the latest MCX session in the nightly-rebuilt MCX store, at the sleeve's capital:
- monthly multi-horizon signals at the last MCX close of each month;
- a 40% volatility target per instrument over six, in whole mini lots;
- fills at the next close, with ``mcx_costs``;
- GT2's rolls, and margin at 25% of notional within 80% of the sleeve.

So the paper record follows exactly the rules DG2 passed.  MCX trades until about 23:30 IST, after the 19:00
run, so the leg is marked at the latest MCX close the store has (normally the previous session's), and each row
records that date.  The sleeve's cash and margin earn a liquid ETF's daily return (LIQUIDCASE, from the equity
store; live they would be a pledged liquid ETF).

The ledger appends one row per night and never rewrites one.  An alert is raised when:
- a replay of a date already recorded moves by more than 0.5% of the sleeve (MCX revised its data);
- the MCX data falls more than four days behind (the nightly sync failing).

MCX paper fills are the model's fills: there is no broker paper for MCX, so real fills come with the dry runs.
The ledger is JSON in the book's ``paper_state`` (key ``mcx_ledger``).  Paper only: nothing here places an order.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Dict, List, Mapping, Optional

import pandas as pd

logger = logging.getLogger(__name__)

STATE_KEY = "mcx_ledger"
LIQUID_SYMBOL = "LIQUIDCASE"
DRIFT_ALERT = 0.005
STALE_DAYS = 4


@dataclass(frozen=True)
class SleeveConfig:
    """The ``mcx_sleeve`` block of a deployment file."""

    weight: float = 0.25
    config: str = "GT5-B"

    @classmethod
    def from_dict(cls, raw: Mapping) -> "SleeveConfig":
        from nse_engine.mcx_trend import CONFIGS

        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown mcx_sleeve keys: {sorted(unknown)}")
        cfg = cls(**dict(raw))
        if not (isinstance(cfg.weight, (int, float)) and 0.0 < cfg.weight < 1.0):
            raise ValueError("mcx_sleeve weight must be between 0 and 1")
        if cfg.config not in CONFIGS:
            raise ValueError(f"mcx_sleeve config must be one of {sorted(CONFIGS)}")
        return cfg


@dataclass
class Ledger:
    start: str
    capital: float
    last_session: str = ""
    history: List[Dict] = field(default_factory=list)
    trades: List[Dict] = field(default_factory=list)       # the last 500 MCX fills
    decisions: List[Dict] = field(default_factory=list)    # the last 24 monthly decisions

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=float)

    @classmethod
    def from_json(cls, text: str) -> "Ledger":
        return cls(**json.loads(text))


def load_ledger(cloud, start: str, capital: float) -> Ledger:
    state = cloud.read_state() if cloud is not None else {}
    text = state.get(STATE_KEY) if state else None
    return Ledger.from_json(text) if text else Ledger(start=start, capital=float(capital))


def save_ledger(cloud, ledger: Ledger) -> None:
    if cloud is None:
        raise RuntimeError("the MCX sleeve needs the book's cloud state (Neon)")
    if not cloud.sync_state({STATE_KEY: ledger.to_json()}):
        raise RuntimeError("could not save the MCX sleeve's ledger")


def liquid_closes(store_dir: str, start: pd.Timestamp, day: pd.Timestamp) -> pd.Series:
    """LIQUIDCASE's daily closes from the equity store (its growth is the liquid rate after expenses)."""
    from nse_engine.data.panel import read_equity_rows

    rows = read_equity_rows(store_dir, start - pd.Timedelta(days=10), day, series=("EQ",))
    rows = rows[rows["symbol"].astype(str) == LIQUID_SYMBOL]
    s = rows.assign(date=pd.to_datetime(rows["date"])).set_index("date")["close"].astype("float64").sort_index()
    return s[~s.index.duplicated(keep="last")]


def positions(trades: List[Dict], prepared: Dict, asof: pd.Timestamp) -> List[Dict]:
    """Lots held per instrument after ``trades``, with each contract's notional at the ``asof`` close."""
    net: Dict[tuple, int] = {}                  # (instrument, contract) -> lots; a rolled-out contract nets to 0
    for t in trades:
        k = (t["instrument"], t["contract"])
        net[k] = net.get(k, 0) + (t["lots"] if t["side"] == "BUY" else -t["lots"])
    insts = {x.name: x for x in prepared["instruments"]}
    out = []
    for (name, contract), lots in sorted(net.items()):
        if not lots:
            continue
        sym, exp = contract.split(" ")
        c = (sym, pd.Timestamp(exp))
        x = insts[name]
        px = x.close(c, asof)
        px = px if math.isfinite(px) else next((x.close(c, d) for d in reversed(prepared["calendar"][
            prepared["calendar"] <= asof]) if math.isfinite(x.close(c, d))), float("nan"))
        out.append({"instrument": name, "contract": contract, "lots": lots,
                    "notional": abs(lots) * x.multiplier(c) * px if math.isfinite(px) else float("nan")})
    return out


def run_session(ledger: Ledger, cfg: SleeveConfig, day: pd.Timestamp, share_equity: float, initial: float,
                prepared: Dict, liquid: pd.Series) -> Dict:
    """One night of the sleeve; idempotent (a session already done returns its summary)."""
    from nse_engine.mcx_trend import CONFIGS, run_trend

    key = day.date().isoformat()
    if ledger.last_session == key and ledger.history:
        return summary(ledger, cfg, initial, prepared, today=[], alerts=[])
    cal: pd.DatetimeIndex = prepared["calendar"]
    asof = cal[cal <= day][-1]
    start = pd.Timestamp(ledger.start)
    alerts: List[str] = []
    if asof < start:                             # no MCX session since the start yet
        res, eq = None, pd.Series({start: ledger.capital})
    else:
        res = run_trend(replace(CONFIGS[cfg.config], initial_capital=ledger.capital, start=ledger.start,
                                end=asof.date().isoformat()), prepared)
        eq = res.equity
    if (day - asof).days > STALE_DAYS:
        alerts.append(f"MCX data is stale: latest session {asof.date()} (the nightly MCX sync may be failing)")
    drift = max([abs(float(eq.get(pd.Timestamp(r["mcx_asof"]), r["mcx_equity"])) - r["mcx_equity"]) / ledger.capital
                 for r in ledger.history] + [0.0])
    if drift > DRIFT_ALERT:
        alerts.append(f"MCX replay moved {drift:.2%} of the sleeve on a recorded date (MCX revised its data)")
    liq = liquid[(liquid.index >= start) & (liquid.index <= day)]
    liquid_factor = float(liq.iloc[-1] / liq.iloc[0]) if len(liq) > 1 else 1.0
    mcx_equity = float(eq.iloc[-1])
    sleeve = mcx_equity * liquid_factor
    nav = float(share_equity) + sleeve
    prev = ledger.history[-1]["nav"] if ledger.history else float(initial)
    prev_asof = ledger.history[-1]["mcx_asof"] if ledger.history else ""
    today = [t for t in (res.trades if res is not None else []) if t["date"] > prev_asof]
    if res is not None and len(res.decisions):
        known = {d["date"] for d in ledger.decisions}
        ledger.decisions = (ledger.decisions + [d for d in res.decisions.to_dict("records")
                                                if d["date"] not in known])[-24:]
    held = positions(res.trades, prepared, asof) if res is not None else []
    gross = sum(p["notional"] for p in held if math.isfinite(p["notional"]))
    ledger.history.append({"date": key, "nav": nav, "share_equity": float(share_equity), "sleeve": sleeve,
                           "mcx_asof": str(asof.date()), "mcx_equity": mcx_equity, "liquid_factor": liquid_factor,
                           "gross": gross / sleeve if sleeve > 0 else 0.0,
                           "weight": sleeve / nav if nav > 0 else 0.0,
                           "costs": sum(t["fee"] + t["slippage"] for t in today), "day_pnl": nav - prev,
                           "alerts": len(alerts)})
    ledger.trades = (ledger.trades + today)[-500:]
    ledger.last_session = key
    return summary(ledger, cfg, initial, prepared, today=today, alerts=alerts, held=held)


def metrics(ledger: Ledger, initial: float, rf: float = 0.065) -> Dict:
    """The combined book (shares + MCX trend) since its first session."""
    if not ledger.history:
        return {}
    h = pd.DataFrame(ledger.history)
    nav = h["nav"].astype(float)
    r = nav.pct_change().fillna(nav.iloc[0] / float(initial) - 1.0)
    eq = (1 + r).cumprod()
    out = {"sessions": len(r), "return": float(eq.iloc[-1] - 1.0), "max_drawdown": float((eq / eq.cummax() - 1).min()),
           "sleeve_return": float(h["sleeve"].iloc[-1] / ledger.capital - 1.0), "costs_total": float(h["costs"].sum()),
           "gross_mean": float(h["gross"].mean()), "decisions": len(ledger.decisions),
           "zero_lot_decisions": {k[:-5]: int(sum(1 for d in ledger.decisions if d.get(k) == 0))
                                  for k in (ledger.decisions[-1] if ledger.decisions else {}) if k.endswith("_lots")}}
    if len(r) >= 20:
        ex = r - rf / 252
        out["sharpe"] = float(ex.mean() / ex.std(ddof=1) * math.sqrt(252)) if ex.std(ddof=1) > 0 else float("nan")
    return out


def summary(ledger: Ledger, cfg: SleeveConfig, initial: float, prepared: Dict, today: List[Dict],
            alerts: List[str], held: Optional[List[Dict]] = None) -> Dict:
    last = ledger.history[-1]
    if held is None:
        held = positions(ledger.trades, prepared, pd.Timestamp(last["mcx_asof"]))
    decision = ledger.decisions[-1] if ledger.decisions else {}
    signals = [{"instrument": k[:-7], "signal": v, "lots": decision.get(f"{k[:-7]}_lots", 0),
                "vol": decision.get(f"{k[:-7]}_vol")} for k, v in decision.items() if k.endswith("_signal")]
    return {"config": cfg.config, "target_weight": cfg.weight, "nav": last["nav"], "sleeve": last["sleeve"],
            "mcx_asof": last["mcx_asof"], "mcx_equity": last["mcx_equity"], "liquid_factor": last["liquid_factor"],
            "gross": last["gross"], "weight": last["weight"], "day_pnl": last["day_pnl"], "initial": float(initial),
            "capital": ledger.capital, "fills": today, "holdings": held, "decision_date": decision.get("date"),
            "signals": signals, "alerts": alerts, "metrics": metrics(ledger, initial)}
