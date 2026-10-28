"""
Options research rounds O5 onward (plan 5v): one engine for the multi-leg
structures the owner's sources describe, and the round's gates.

A *strategy* decides; the engine trades, marks and settles.  Each session:

1. hedge futures are marked to the close (and rolled on their settlement
   session);
2. orders decided at the previous close fill at this close: each leg at its
   close when it traded, else its settlement price, with ``fno_costs``
   slippage and charges (``_Quotes.price``, the LN-T20 settle guard);
3. legs on their settlement session expire at intrinsic value against the
   exact index close, the buyer paying exercise STT;
4. open positions are marked; equity = cash + the marked legs;
5. the strategy reads the close (``Context``) and returns exits,
   adjustments, hedges and entries for the next session.

End-of-day only: nothing is decided and filled in one session, so no
same-day exit.  Contracts are keyed by the session they settle on
(``rekey_to_settlement``).  Greeks are Black-76 on the expiry's forward:
its futures close where one trades, else spot carried at the risk-free
rate.  An ITM strike takes its OTM mirror's IV (LN-T27).  Positions are
sized in lots so that the maximum loss of all open positions stays within
the strategy's share of equity.  Results go
to the options registry, and ``evaluate_round`` applies plan 5v's gates.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from kite_connect.options.fno_costs import (FNO_COST_MODEL_VERSION, exercise_stt, fill_price, futures_charges,
                                            option_charges)
from kite_connect.options.options_config import OptionsConfig
from kite_connect.options.signal_futures import rekey_to_settlement
from kite_connect.options.theory import BUY, CALL, PUT, SELL, black_scholes, implied_volatility, intrinsic_value
from nse_engine.data import fo_store

logger = logging.getLogger(__name__)

#: Rounds O5 onward are judged beside the live book (plan 5v): the deployed configuration's latest run.
LIVE_BOOK_CONFIG = "config/nse_engine_deployed.json"
BOOK_RF = 0.065
OVERLAY_WEIGHT = 0.25
TREND_MA_DAYS, TREND_RETURN_DAYS = 200, 63
IV_RANK_DAYS = 252
RV_DAYS = 20
#: Natenberg's forecast: 6/12/26/52-week realised volatility, weighted toward the shortest (OVP p.392-394).
FORECAST_WEIGHTS = ((30, 0.40), (60, 0.30), (130, 0.20), (260, 0.10))


# ---------------------------------------------------------------- data

class _Quotes:
    """One underlying's monthly options, futures, index and 30-day IV history, keyed by settlement session."""

    def __init__(self, store_dir: str, symbol: str, start: str, end: str, rate: float):
        lo = (pd.Timestamp(start) - pd.Timedelta(days=420)).strftime("%Y-%m-%d")
        opts = rekey_to_settlement(fo_store.load_options(store_dir, lo, end, symbol))
        exp = pd.DatetimeIndex(opts["expiry"].unique())
        self.monthly = pd.DatetimeIndex(sorted(pd.Series(exp, index=exp).groupby([exp.year, exp.month]).max()))
        opts = opts[opts["expiry"].isin(self.monthly)]
        self.quotes = {d: g.set_index(["expiry", "option_type", "strike"]).sort_index()
                       for d, g in opts.groupby("date")}
        futs = rekey_to_settlement(fo_store.load_futures(store_dir, lo, end, symbol))
        futs = futs.assign(price=np.where((futs["contracts"] > 0) & (futs["close"] > 0), futs["close"], futs["settle"]))
        self.futures = {(r.date, r.expiry): float(r.price) for r in futs.itertuples() if r.price > 0}
        self.final_settle = {r.expiry: float(r.settle) for r in futs.itertuples() if r.date == r.expiry and r.settle > 0}
        self.future_expiries = futs.groupby("date")["expiry"].apply(lambda s: sorted(set(s))).to_dict()
        self.index = fo_store.load_underlying(store_dir, symbol)
        self.exact = fo_store.load_underlying(store_dir, symbol, exact_only=True)
        self.sessions = pd.DatetimeIndex(sorted(self.quotes))
        iv = pd.read_parquet(Path(store_dir) / "iv" / f"{symbol}_v1.parquet")
        self.iv = iv.assign(date=pd.to_datetime(iv["date"])).set_index("date").sort_index()
        self.rate = rate
        self.symbol = symbol
        r2 = np.log(self.index.dropna()).diff() ** 2           # zero-mean realised variance (Ursone)
        self.hv = {n: np.sqrt(r2.rolling(n).mean() * 252) for n, _ in FORECAST_WEIGHTS}
        self.long_run = self.hv[30].expanding(min_periods=252).mean()

    def spot(self, d: pd.Timestamp) -> float:
        return float(self.index.get(d, math.nan))

    def forward(self, d: pd.Timestamp, expiry: pd.Timestamp) -> float:
        """The expiry's futures close, else spot carried at the risk-free rate."""
        f = self.futures.get((d, expiry))
        if f:
            return float(f)
        return self.spot(d) * math.exp(self.rate * max((expiry - d).days, 0) / 365.0)

    def chain(self, d: pd.Timestamp, expiry: pd.Timestamp, option_type: str) -> pd.DataFrame:
        q = self.quotes.get(d)
        try:
            return q.loc[(expiry, option_type)] if q is not None else pd.DataFrame()
        except KeyError:
            return pd.DataFrame()

    def price(self, d: pd.Timestamp, expiry: pd.Timestamp, option_type: str, strike: float) -> Tuple[float, bool]:
        """(price, traded): close when traded, intrinsic against the final settlement on its own settlement
        session (never the settle column, LN-T20), else the settlement price; NaN when none."""
        ch = self.chain(d, expiry, option_type)
        if strike not in ch.index:
            if d == expiry:
                s = self.settlement(expiry)
                return (intrinsic_value(option_type, s, strike) if np.isfinite(s) else math.nan), False
            return math.nan, False
        row = ch.loc[strike]
        if row["contracts"] > 0 and row["close"] > 0:
            return float(row["close"]), True
        if d == expiry:
            s = self.settlement(expiry)
            return (intrinsic_value(option_type, s, strike) if np.isfinite(s) else math.nan), False
        return (float(row["settle"]) if row["settle"] > 0 else math.nan), False

    def settlement(self, expiry: pd.Timestamp) -> float:
        """The final settlement: the expiring future's settlement, else the exact index close (equal 2007-Sep
        2026; from 7 Dec 2026 the close is the closing auction's, settlement the 30-minute VWAP: CAS1)."""
        s = float(self.final_settle.get(expiry, math.nan))
        return s if np.isfinite(s) else float(self.exact.get(expiry, math.nan))

    def lot_size(self, d: pd.Timestamp, expiry: pd.Timestamp) -> float:
        for t in (CALL, PUT):
            ch = self.chain(d, expiry, t)
            if len(ch):
                return float(ch["lot_size"].iloc[0])
        return math.nan

    def front_future(self, d: pd.Timestamp) -> Optional[pd.Timestamp]:
        return next((e for e in self.future_expiries.get(d, []) if e > d), None)


# ---------------------------------------------------------------- the strategy's view of a close

@dataclass
class Leg:
    expiry: pd.Timestamp
    option_type: str
    strike: float
    units: int                    # signed: + long, - short
    entry_price: float = math.nan
    mark: float = math.nan


@dataclass
class Position:
    id: int
    legs: List[Leg]
    opened: pd.Timestamp
    entry_value: float            # signed premium at entry per the fills: + paid (debit), - received (credit)
    max_loss: float               # INR, the sizing basis
    hedge_units: float = 0.0      # futures units (signed)
    hedge_contract: Optional[pd.Timestamp] = None
    hedge_mark: float = math.nan
    info: Dict[str, Any] = field(default_factory=dict)

    def value(self) -> float:
        """Signed mark-to-market of the legs: what closing would pay (+) or cost (-)."""
        return float(sum(l.units * l.mark for l in self.legs))


class Context:
    """Everything known at one close: prices, Greeks, the regime and volatility labels."""

    def __init__(self, q: _Quotes, d: pd.Timestamp, equity: float):
        self.q, self.d, self.equity = q, d, equity
        self.spot = q.spot(d)
        self._iv_cache: Dict[Tuple, Optional[float]] = {}

    # sessions and expiries
    def sessions_to(self, expiry: pd.Timestamp) -> int:
        """Sessions after today up to and including ``expiry``'s settlement session."""
        s = self.q.sessions
        return int(s.searchsorted(expiry, side="right") - s.searchsorted(self.d, side="right"))

    def first_after_expiry(self) -> bool:
        """Is this the first session after a monthly expiry (the books' monthly decision date)?"""
        i = self.q.sessions.searchsorted(self.d)
        return i > 0 and self.q.sessions[i - 1] in self.q.monthly

    def monthly(self, n: int, min_days: int = 1) -> Optional[pd.Timestamp]:
        """The ``n``-th monthly expiry settling at least ``min_days`` after today (1 = the nearest)."""
        future = [e for e in self.q.monthly if (e - self.d).days >= min_days]
        return future[n - 1] if len(future) >= n else None

    def dte(self, expiry: pd.Timestamp) -> int:
        return int((expiry - self.d).days)

    # prices and Greeks
    def price(self, expiry, option_type, strike) -> Tuple[float, bool]:
        return self.q.price(self.d, expiry, option_type, strike)

    def iv(self, expiry: pd.Timestamp, option_type: str, strike: float) -> Optional[float]:
        """Black-76 IV from a traded close; an ITM strike takes its OTM mirror's (LN-T27)."""
        key = (expiry, option_type, strike)
        if key not in self._iv_cache:
            f, dte = self.q.forward(self.d, expiry), max(self.dte(expiry), 1)
            otm = CALL if strike >= f else PUT
            out = None
            for t in (otm, PUT if otm == CALL else CALL):
                px, traded = self.price(expiry, t, strike)
                if traded and px > 0:
                    out = implied_volatility(t, px, f, strike, dte, self.q.rate, dividend=self.q.rate)
                    if out is not None:
                        break
            self._iv_cache[key] = out
        return self._iv_cache[key]

    def delta(self, expiry: pd.Timestamp, option_type: str, strike: float) -> Optional[float]:
        iv = self.iv(expiry, option_type, strike)
        if iv is None:
            return None
        f = self.q.forward(self.d, expiry)
        return black_scholes(option_type, f, strike, max(self.dte(expiry), 1), self.q.rate, iv,
                             dividend=self.q.rate).delta

    def traded_strikes(self, expiry: pd.Timestamp, option_type: str) -> List[float]:
        ch = self.q.chain(self.d, expiry, option_type)
        return sorted(ch.index[(ch["contracts"] > 0) & (ch["close"] > 0)]) if len(ch) else []

    def atm(self, expiry: pd.Timestamp) -> Optional[float]:
        """The traded strike nearest the expiry's forward (Ursone: ATM against the forward, not spot)."""
        strikes = sorted(set(self.traded_strikes(expiry, CALL)) & set(self.traded_strikes(expiry, PUT)))
        if not strikes:
            return None
        f = self.q.forward(self.d, expiry)
        return min(strikes, key=lambda k: abs(k - f))

    def strike_by_delta(self, expiry: pd.Timestamp, option_type: str, max_abs_delta: float) -> Optional[float]:
        """The traded OTM strike with the largest |delta| not above ``max_abs_delta`` (the richest premium
        inside the cap)."""
        best, best_d = None, -1.0
        for k in self.traded_strikes(expiry, option_type):
            dl = self.delta(expiry, option_type, k)
            if dl is not None and best_d < abs(dl) <= max_abs_delta:
                best, best_d = k, abs(dl)
        return best

    # labels, each from data up to this close
    def regime(self) -> str:
        """bull: NIFTY above its 200-day mean and up over 63 sessions; bear: below and down; else range."""
        s = self.q.index.loc[: self.d].dropna()
        if len(s) < TREND_MA_DAYS + 1:
            return "range"
        above = s.iloc[-1] > s.iloc[-TREND_MA_DAYS:].mean()
        up = s.iloc[-1] > s.iloc[-TREND_RETURN_DAYS - 1]
        return "bull" if above and up else "bear" if (not above and not up) else "range"

    def iv30(self) -> float:
        return float(self.q.iv["iv30"].asof(self.d)) if len(self.q.iv) else math.nan

    def iv_rank(self) -> float:
        """Percent rank of today's 30-day IV in the last 252 sessions (0-100)."""
        h = self.q.iv["iv30"].loc[: self.d].dropna().tail(IV_RANK_DAYS)
        if len(h) < 60:
            return math.nan
        return float((h.iloc[:-1] < h.iloc[-1]).mean() * 100)

    def term_spread(self) -> float:
        """Near-month minus far-month ATM IV (vol points / 100), from the IV history."""
        r = self.q.iv.loc[: self.d].tail(1)
        return float(r["near_iv"].iloc[0] - r["far_iv"].iloc[0]) if len(r) else math.nan

    def vol_forecast(self) -> float:
        """Natenberg's weighted blend of realised volatility (``FORECAST_WEIGHTS``) at this close."""
        return float(sum(w * self.q.hv[n].asof(self.d) for n, w in FORECAST_WEIGHTS))

    def long_run_vol(self) -> float:
        """The long end of the volatility cone: the mean 30-day realised volatility up to this close."""
        return float(self.q.long_run.asof(self.d))

    def realised_vol(self, days: int = RV_DAYS) -> float:
        """Zero-mean close-to-close volatility, annualised (Ursone: no mean in a volatility estimate)."""
        r = np.log(self.q.index.loc[: self.d].dropna()).diff().dropna().tail(days)
        return float(np.sqrt((r ** 2).mean() * 252)) if len(r) >= days else math.nan


@dataclass
class Decisions:
    close: List[Tuple[int, str]] = field(default_factory=list)          # (position id, reason)
    open: List[Dict[str, Any]] = field(default_factory=list)            # {"legs", "max_loss_per_lot", "scale"?, "info"?}
    adjust: List[Tuple[int, List[Leg], str]] = field(default_factory=list)   # (id, legs to add (signed units per lot), reason)
    hedge: List[Tuple[int, float]] = field(default_factory=list)        # (id, target futures units)


class Strategy:
    """A round's candidate: ``decide`` reads the close and the open positions and returns Decisions."""

    name: str = ""
    symbol: str = "NIFTY"
    capital: float = 750_000.0
    deploy_fraction: float = 0.5

    def decide(self, ctx: Context, positions: List[Position]) -> Decisions:     # pragma: no cover - interface
        raise NotImplementedError

    def params(self) -> Dict[str, Any]:
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}

    def config_hash(self) -> str:
        payload = {"class": type(self).__name__, **{k: v for k, v in self.params().items() if k != "name"},
                   "fno_cost_model": FNO_COST_MODEL_VERSION}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def to_json(self) -> str:
        return json.dumps({"class": type(self).__name__, **self.params()}, indent=2, sort_keys=True, default=str)


# ---------------------------------------------------------------- the engine

@dataclass
class StructureResult:
    config: Any
    returns: pd.Series
    equity: pd.Series
    trades: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    data_hash: str
    run_id: str = ""
    run_dir: Optional[str] = None


def run_structure(strategy: Strategy, store_dir: str, start: str = "2013-01-01", end: str = "2025-12-31",
                  opt_cfg: OptionsConfig = OptionsConfig()) -> StructureResult:
    """Replay ``strategy`` session by session over [start, end] (module docstring)."""
    q = _Quotes(store_dir, strategy.symbol, start, end, opt_cfg.market.risk_free_rate)
    sessions = q.sessions[(q.sessions >= pd.Timestamp(start)) & (q.sessions <= pd.Timestamp(end))]
    sym = strategy.symbol
    cash, positions, closed = float(strategy.capital), [], []
    pending, next_id = Decisions(), 0
    curve: Dict[pd.Timestamp, float] = {}
    costs_total = 0.0

    def leg_fill(d, leg: Leg, units: int) -> Tuple[float, float, bool]:
        """(cash flow, charges, priced) of trading ``units`` of ``leg`` at d's close."""
        px, _ = q.price(d, leg.expiry, leg.option_type, leg.strike)
        if not np.isfinite(px):
            return 0.0, 0.0, False
        side = BUY if units > 0 else SELL
        fill = fill_price(px, side, sym, opt_cfg.slippage)
        charges = option_charges(fill * abs(units), side, d, 1, opt_cfg.charges).total
        return -units * fill, charges, True

    def futures_trade(d, contract, units: float) -> float:
        px = q.futures.get((d, contract), math.nan)
        if not np.isfinite(px) or units == 0:
            return 0.0
        return futures_charges(abs(units) * px, BUY if units > 0 else SELL, d, 1, opt_cfg.charges).total

    for d in sessions:
        # 1. hedges: mark, roll on the contract's settlement session
        for p in positions:
            if p.hedge_units:
                px = q.futures.get((d, p.hedge_contract), p.hedge_mark)
                cash += p.hedge_units * (px - p.hedge_mark)
                p.hedge_mark = px
                if p.hedge_contract <= d:
                    nxt = q.front_future(d)
                    c = futures_trade(d, p.hedge_contract, -p.hedge_units)
                    if nxt is not None:
                        c += futures_trade(d, nxt, p.hedge_units)
                        p.hedge_contract, p.hedge_mark = nxt, q.futures.get((d, nxt), math.nan)
                    else:
                        p.hedge_units = 0.0
                    cash -= c
                    costs_total += c

        # 2. orders decided at the last close fill at this close
        by_id = {p.id: p for p in positions}
        for pid, reason in pending.close:
            p = by_id.get(pid)
            if p is None or p not in positions:
                continue
            for leg in p.legs:
                flow, ch, ok = leg_fill(d, leg, -leg.units)
                if not ok:                                    # no price: close at the last mark
                    flow, ch = leg.units * leg.mark, 0.0
                cash += flow - ch
                costs_total += ch
            if p.hedge_units:
                c = futures_trade(d, p.hedge_contract, -p.hedge_units)
                cash -= c
                costs_total += c
                p.hedge_units = 0.0
            p.info.update(closed=d, exit_reason=reason)
            positions.remove(p)
            closed.append(p)
        for pid, add, reason in pending.adjust:
            p = by_id.get(pid)
            if p is None or p not in positions:
                continue
            for leg in add:
                units = int(leg.units * p.info["lots"] * p.info["lot_size"])
                flow, ch, ok = leg_fill(d, leg, units)
                if not ok:
                    continue
                cash += flow - ch
                costs_total += ch
                p.legs.append(Leg(leg.expiry, leg.option_type, leg.strike, units, -flow / units))
            p.info.setdefault("adjustments", []).append({"date": str(d.date()), "reason": reason})
        for pid, target in pending.hedge:
            p = by_id.get(pid)
            if p is None or p not in positions:
                continue
            contract = p.hedge_contract or q.front_future(d)
            if contract is None:
                continue
            trade = target - p.hedge_units
            c = futures_trade(d, contract, trade)
            cash -= c
            costs_total += c
            if p.hedge_contract is None:
                p.hedge_contract, p.hedge_mark = contract, q.futures.get((d, contract), math.nan)
            p.hedge_units = target
        for order in pending.open:
            legs: List[Leg] = order["legs"]
            equity_now = cash + sum(p.value() for p in positions)
            per_lot = order["max_loss_per_lot"]
            lot = q.lot_size(d, legs[0].expiry)
            # one budget for every open position's maximum loss (Cordier & Gross: total margin <= 50%)
            budget = strategy.deploy_fraction * float(order.get("scale", 1.0)) * equity_now - sum(p.max_loss for p in positions)
            lots = int(budget // per_lot) if per_lot > 0 and budget > 0 else 0
            if not np.isfinite(lot) or lots <= 0:
                continue
            fills, ok = [], True
            for leg in legs:
                units = int(leg.units * lots * lot)
                flow, ch, priced = leg_fill(d, leg, units)
                ok &= priced
                fills.append((leg, units, flow, ch))
            if not ok:
                continue
            p = Position(next_id, [], d, 0.0, per_lot * lots,
                         info={"lots": lots, "lot_size": lot, **order.get("info", {})})
            next_id += 1
            for leg, units, flow, ch in fills:
                cash += flow - ch
                costs_total += ch
                p.legs.append(Leg(leg.expiry, leg.option_type, leg.strike, units, -flow / units))
                p.entry_value -= flow
            positions.append(p)
        pending = Decisions()

        # 3. settlement at intrinsic against the exact close
        for p in list(positions):
            for leg in [l for l in p.legs if l.expiry <= d]:
                s = q.settlement(leg.expiry)
                s = s if np.isfinite(s) else q.spot(d)
                value = intrinsic_value(leg.option_type, s, leg.strike)
                cash += leg.units * value
                if leg.units > 0:
                    stt = exercise_stt(leg.option_type, leg.strike, s, leg.units, d, opt_cfg.charges)
                    cash -= stt
                    costs_total += stt
                p.legs.remove(leg)
            if not p.legs:
                if p.hedge_units:
                    c = futures_trade(d, p.hedge_contract, -p.hedge_units)
                    cash -= c
                    costs_total += c
                    p.hedge_units = 0.0
                p.info.update(closed=d, exit_reason="expiry")
                positions.remove(p)
                closed.append(p)

        # 4. mark
        for p in positions:
            for leg in p.legs:
                px, _ = q.price(d, leg.expiry, leg.option_type, leg.strike)
                if np.isfinite(px):
                    leg.mark = px
                elif not np.isfinite(leg.mark):
                    leg.mark = leg.entry_price
        equity = cash + sum(p.value() for p in positions)
        curve[d] = equity

        # 5. decide on this close
        pending = strategy.decide(Context(q, d, equity), positions)

    eq = pd.Series(curve, name="equity").sort_index()
    rets = eq.pct_change().fillna(eq.iloc[0] / strategy.capital - 1.0)
    trades = [_trade_row(p, eq) for p in closed + positions]
    return StructureResult(strategy, rets, eq, trades, _metrics(rets, trades, costs_total),
                           fo_store.data_hash(store_dir))


def _trade_row(p: Position, eq: pd.Series) -> Dict[str, Any]:
    return {"id": p.id, "opened": str(p.opened.date()), "closed": str(p.info.get("closed", pd.NaT))[:10],
            "exit_reason": p.info.get("exit_reason", "open"), "lots": p.info.get("lots"),
            "entry_value": p.entry_value, "max_loss": p.max_loss,
            "adjustments": len(p.info.get("adjustments", [])),
            **{k: v for k, v in p.info.items() if k not in ("closed", "exit_reason", "lots", "adjustments")}}


def _metrics(rets: pd.Series, trades: List[Dict[str, Any]], costs: float) -> Dict[str, Any]:
    from nse_engine.validation.dsr import performance_summary

    m = performance_summary(rets, rf_annual=0.0)
    m.update(sharpe=m["excess_sharpe"], calmar=(m["cagr"] / abs(m["max_drawdown"]) if m["max_drawdown"] else None),
             positions=len(trades), charges_inr=costs,
             exit_reasons={r: sum(t["exit_reason"] == r for t in trades) for r in sorted({t["exit_reason"] for t in trades})})
    return m


def record(result: StructureResult, runs_dir: str, round_label: str) -> str:
    """Write the run to the options registry (U32), with its positions."""
    from kite_connect.options.backtest import _run_id
    from nse_engine.validation.trials import record_result

    cfg = result.config
    result.run_id = result.run_id or _run_id(cfg.config_hash())
    window = (str(result.equity.index[0].date()), str(result.equity.index[-1].date()))
    run_dir = record_result(result, tag=cfg.name, runs_dir=runs_dir, window=window,
                            extra={"family": "options", "round": round_label, "fno_cost_model": FNO_COST_MODEL_VERSION})
    pd.DataFrame(result.trades).to_csv(Path(run_dir) / "trades.csv", index=False)
    return run_dir


# ---------------------------------------------------------------- plan 5v's gates

def _live_book_returns(book_runs: str) -> Tuple[pd.Series, str]:
    from nse_engine.costs import COST_MODEL_VERSION
    from nse_engine.deployment import load_deployment
    from nse_engine.scorecard import latest_run
    from nse_engine.validation.trials import TrialRegistry

    reg = TrialRegistry(book_runs)
    trial = latest_run(reg, load_deployment(LIVE_BOOK_CONFIG).engine.config_hash(), cost_model=COST_MODEL_VERSION)
    return reg.load_returns(str(trial["run_id"])), str(trial["run_id"])


def _nifty_tri_returns(store_dir: str, index: pd.DatetimeIndex) -> pd.Series:
    import glob

    from nse_engine.data.panel import build_index_close

    files = sorted(glob.glob(str(Path(store_dir) / "indices" / "*.parquet")))
    idx = pd.concat([pd.read_parquet(f) for f in files])
    idx["date"] = pd.to_datetime(idx["date"])
    tri = build_index_close(idx, pd.DatetimeIndex(sorted(idx["date"].unique())), ("NIFTY50_TRI",))["NIFTY50_TRI"]
    return tri.ffill().pct_change().reindex(index).dropna()


def evaluate_round(label: str, candidates: Dict[str, Strategy], options_runs: str, book_runs: str,
                   equity_store: str = "data/nse_engine/store", start: str = "2013-01-01",
                   end: str = "2025-12-31") -> Dict[str, Any]:
    """Plan 5v: gate 1 on each candidate alone at the options family's N; gate 2 beside the live book
    (raises CAGR, excess Sharpe and alpha against NIFTY 50 TRI; MaxDD at most 2 points worse)."""
    from kite_connect.options.backtest import overlay
    from nse_engine.validation.diagnostics import alpha_beta
    from nse_engine.validation.dsr import deflated_sharpe, performance_summary
    from nse_engine.validation.pbo import cscv_pbo
    from nse_engine.validation.trials import TrialRegistry

    reg = TrialRegistry(options_runs)
    by_tag = reg.list_trials().set_index("run_id")["tag"]
    mat = reg.returns_matrix(window=(start, end))
    keys = {s.name: k for k, s in candidates.items()}
    book, book_run = _live_book_returns(book_runs)
    book = book[(book.index >= pd.Timestamp(start)) & (book.index <= pd.Timestamp(end))]
    tri = _nifty_tri_returns(equity_store, book.index)
    base = performance_summary(book, BOOK_RF)
    base_alpha = alpha_beta(book, tri, rf_annual=BOOK_RF)["alpha_annual"]
    report: Dict[str, Any] = {"round": label, "options_configurations": int(mat.shape[1]), "live_book_run": book_run,
                              "base": {**{k: base[k] for k in ("cagr", "excess_sharpe", "max_drawdown")},
                                       "alpha_annual": base_alpha},
                              "candidates": {}}
    if mat.shape[1] >= 2:
        report["options_pbo"] = {k: v for k, v in cscv_pbo(mat, n_splits=16, rf_annual=0.0).items() if k != "logits"}
    for run_id in mat.columns:
        key = keys.get(by_tag[run_id])
        if key is None:
            continue
        r = mat[run_id]
        summ = performance_summary(r, rf_annual=0.0)
        dsr = deflated_sharpe(r, trials_matrix=mat, rf_annual=0.0)["dsr"]
        gate1 = bool(summ["excess_sharpe"] > 0 and dsr >= 0.95)
        comb = overlay(book, reg.load_returns(run_id), OVERLAY_WEIGHT)
        cs = performance_summary(comb, BOOK_RF)
        ca = alpha_beta(comb, tri, rf_annual=BOOK_RF)["alpha_annual"]
        checks = {"cagr": cs["cagr"] > base["cagr"], "excess_sharpe": cs["excess_sharpe"] > base["excess_sharpe"],
                  "alpha": ca > base_alpha, "max_drawdown": cs["max_drawdown"] >= base["max_drawdown"] - 0.02}
        report["candidates"][key] = {
            "run_id": run_id, "sharpe": summ["excess_sharpe"], "cagr": summ["cagr"], "max_drawdown": summ["max_drawdown"],
            "dsr": dsr, "gate1": gate1,
            "combined": {"cagr": cs["cagr"], "excess_sharpe": cs["excess_sharpe"], "max_drawdown": cs["max_drawdown"],
                         "alpha_annual": ca, "checks": checks},
            "gate2": all(checks.values()), "passes": gate1 and all(checks.values())}
    passed = [k for k, v in report["candidates"].items() if v["passes"]]
    report["verdict"] = (f"PASS: paper-trade {', '.join(passed)} (plan 5v step 6)" if passed
                         else f"{label} closed: no candidate passed both gates")
    return report
