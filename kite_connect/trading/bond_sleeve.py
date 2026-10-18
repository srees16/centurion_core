"""
The gilt sleeve of a paper book (India tracker BD3, plan 5ag): the drawdown lever you accepted on 17 Oct 2026.

A book with ``bond_sleeve`` in its deployment file is two ledgers:
- its shares: an ordinary engine paper book (``PaperTrader``) on ``paper_capital_inr`` less the sleeve's weight;
- its gilts: this ledger, a 10-year gilt ETF bought for delivery in whole units.

Every session, after the share book has processed the day:
1. a queued order fills at the ETF's open from the day's bhavcopy (its close when the open strays more than
   ``nse_engine.costs.ETF_OPEN_BAND`` from it, as the engine fills ETFs), at the bond cost model: statutory
   charges with no STT, a 10 bp spread floor and the square-root impact on the ETF's 20-session median traded
   value (``nse_engine.bond_stream.trade_cost``);
2. the units are marked at the ETF's close (the last close when it did not trade);
3. on the first session the ledger's cash is queued for the next open.

The split is static in paper: the paper share book cannot pass cash to the sleeve or take it back, so BD1's
monthly rebalancing waits for the live ledger.  Over 60 paper sessions that drift is a few basis points.
The ledger is JSON in the book's ``paper_state`` (key ``bond_ledger``), one history row per session.
Paper only: nothing here can place an order.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Dict, List, Mapping, Optional

import pandas as pd

logger = logging.getLogger(__name__)

STATE_KEY = "bond_ledger"
#: Kept back from the first buy for its own costs.
COST_RESERVE = 0.005


@dataclass(frozen=True)
class SleeveConfig:
    """The ``bond_sleeve`` block of a deployment file."""

    weight: float = 0.25
    symbol: str = "LTGILTBEES"

    @classmethod
    def from_dict(cls, raw: Mapping) -> "SleeveConfig":
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown bond_sleeve keys: {sorted(unknown)}")
        cfg = cls(**dict(raw))
        if not (isinstance(cfg.weight, (int, float)) and 0.0 < cfg.weight < 1.0):
            raise ValueError("bond_sleeve weight must be between 0 and 1")
        if not (isinstance(cfg.symbol, str) and cfg.symbol):
            raise ValueError("bond_sleeve symbol must be an NSE symbol")
        return cfg


@dataclass
class Ledger:
    cash: float
    units: int = 0
    pending: Dict = field(default_factory=dict)      # {"side": "BUY", "value": rupees}
    last_session: str = ""
    history: List[Dict] = field(default_factory=list)
    fills: List[Dict] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=float)

    @classmethod
    def from_json(cls, text: str) -> "Ledger":
        return cls(**json.loads(text))


def load_ledger(cloud, start_cash: float) -> Ledger:
    state = cloud.read_state() if cloud is not None else {}
    text = state.get(STATE_KEY) if state else None
    return Ledger.from_json(text) if text else Ledger(cash=float(start_cash))


def save_ledger(cloud, ledger: Ledger) -> None:
    if cloud is None:
        raise RuntimeError("the gilt sleeve needs the book's cloud state (Neon)")
    if not cloud.sync_state({STATE_KEY: ledger.to_json()}):
        raise RuntimeError("could not save the gilt sleeve's ledger")


def etf_bars(store_dir: str, symbol: str, day: pd.Timestamp) -> pd.DataFrame:
    """The ETF's daily bars (open, close, value_inr) for about three months up to ``day``, from the store."""
    from nse_engine.data.panel import read_equity_rows

    rows = read_equity_rows(store_dir, day - pd.Timedelta(days=100), day, series=("EQ",))   # ETFs trade as EQ
    rows = rows[rows["symbol"].astype(str) == symbol]
    bars = rows.assign(date=pd.to_datetime(rows["date"])).set_index("date").sort_index()
    return bars[~bars.index.duplicated(keep="last")][["open", "close", "value_inr"]].astype("float64")


def _cost(value: float, side: str, day: pd.Timestamp, adv: float) -> float:
    """The bond cost model at the ETF's own traded value (debt ETFs share one statutory schedule: no STT)."""
    from nse_engine.bond_stream import BondConfig, trade_cost

    cfg = BondConfig() if not (adv > 0) else replace(BondConfig(), gilt_adv_inr=float(adv))
    return trade_cost(value, side, day, cfg, "gilt")


def run_session(ledger: Ledger, cfg: SleeveConfig, day: pd.Timestamp, bars: pd.DataFrame, share_equity: float,
                initial: float) -> Dict:
    """One session of the sleeve; idempotent (a session already done returns its summary)."""
    from nse_engine.costs import ETF_OPEN_BAND

    key = day.date().isoformat()
    if ledger.last_session == key and ledger.history:
        return summary(ledger, cfg, initial, today=[f for f in ledger.fills if f["date"] == key])
    bar = bars.loc[day] if day in bars.index else None
    past = bars[bars.index < day]["value_inr"].tail(20)
    adv = float(past.median()) if len(past) else float("nan")
    today: List[Dict] = []
    costs = 0.0
    # 1. the queued order at today's open
    if ledger.pending and bar is not None and bar["close"] > 0:
        o, c = float(bar["open"]), float(bar["close"])
        px = o if o > 0 and abs(o / c - 1.0) <= ETF_OPEN_BAND else c
        if ledger.pending["side"] == "BUY":
            units = int(math.floor(min(ledger.pending["value"], ledger.cash) * (1.0 - COST_RESERVE) / px))
            if units > 0:
                value = units * px
                costs = _cost(value, "BUY", day, adv)
                ledger.cash -= value + costs
                ledger.units += units
                today.append({"date": key, "side": "BUY", "units": units, "price": px, "value": value,
                              "costs": costs, "open_used": px == o})
        ledger.pending = {}
    # 2. mark at the close
    price = float(bar["close"]) if bar is not None and bar["close"] > 0 else (
        ledger.history[-1]["price"] if ledger.history else float("nan"))
    value = ledger.units * price if ledger.units else 0.0
    sleeve = ledger.cash + value
    nav = float(share_equity) + sleeve
    prev = ledger.history[-1]["nav"] if ledger.history else float(initial)
    # 3. the first session queues the whole cash for the next open
    if ledger.units == 0 and not ledger.pending:
        ledger.pending = {"side": "BUY", "value": ledger.cash}
    ledger.history.append({"date": key, "nav": nav, "share_equity": float(share_equity), "sleeve": sleeve,
                           "value": value, "cash": ledger.cash, "units": ledger.units, "price": price,
                           "weight": value / nav if nav > 0 else 0.0, "costs": costs, "day_pnl": nav - prev,
                           "traded": bar is not None})
    ledger.fills = (ledger.fills + today)[-500:]
    ledger.last_session = key
    return summary(ledger, cfg, initial, today)


def metrics(ledger: Ledger, initial: float, rf: float = 0.065) -> Dict:
    """The combined book (shares + gilts) since its first session."""
    if not ledger.history:
        return {}
    h = pd.DataFrame(ledger.history)
    nav = h["nav"].astype(float)
    r = nav.pct_change().fillna(nav.iloc[0] / float(initial) - 1.0)
    eq = (1 + r).cumprod()
    out = {"sessions": len(r), "return": float(eq.iloc[-1] - 1.0), "max_drawdown": float((eq / eq.cummax() - 1).min()),
           "weight_mean": float(h["weight"].mean()), "costs_total": float(h["costs"].sum()),
           "untraded_days": int((~h["traded"].astype(bool)).sum())}
    if len(r) >= 20:
        ex = r - rf / 252
        out["sharpe"] = float(ex.mean() / ex.std(ddof=1) * math.sqrt(252)) if ex.std(ddof=1) > 0 else float("nan")
    return out


def summary(ledger: Ledger, cfg: SleeveConfig, initial: float, today: List[Dict]) -> Dict:
    last = ledger.history[-1]
    return {"symbol": cfg.symbol, "target_weight": cfg.weight, "nav": last["nav"], "sleeve": last["sleeve"],
            "value": last["value"], "cash": last["cash"], "units": last["units"], "price": last["price"],
            "weight": last["weight"], "day_pnl": last["day_pnl"], "initial": float(initial), "fills": today,
            "queued": ([{"side": ledger.pending["side"], "value": ledger.pending["value"]}] if ledger.pending else []),
            "metrics": metrics(ledger, initial)}
