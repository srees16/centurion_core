"""
Options round 3 (tracker O4, plan § 5s): the three F&O strategies of
github.com/paperswithbacktest/awesome-systematic-trading that the 17 Sep 2026
screen could not test, as pre-registered on 9 Oct 2026, 20:47 IST (amended
20:50, before any code or result).

- Y1 ``option-expiration-week-effect.py``: long the near-month NIFTY future
  through the week of each monthly expiry (``signal_futures``' calendar rule).
- Y2 ``volatility-risk-premium-effect.py``: long NIFTY through the near-month
  future, plus each cycle a short ATM straddle and a long put near 0.85 x
  spot on the monthly expiry closest to 30 days, held to expiry.
- Y3 ``market-sentiment-and-an-overnight-anomaly.py``: long NIFTY from the
  close to the next open, a third of capital for each of NIFTY 50 above its
  20-session mean and India VIX below its own, on the previous session's
  closes; NSE's index open and close, at futures costs.

``evaluate`` applies § 5s's gates through ``signal_futures.evaluate``.  CLI::

    python -m kite_connect.options.fo_anomalies run --candidates Y1,Y2,Y3 \\
        --fo-store data/nse_engine/fo_store --runs-dir data/nse_engine/runs_options
    python -m kite_connect.options.fo_anomalies evaluate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from kite_connect.options.fno_costs import (
    FNO_COST_MODEL_VERSION,
    exercise_stt,
    fill_price,
    option_charges,
)
from kite_connect.options.options_config import OptionsConfig
from kite_connect.options.signal_futures import (
    SignalConfig,
    SignalResult,
    _Futures,
    _metrics,
    _trade_cost,
    run_strategy,
    settlement_sessions,
)
from kite_connect.options.signal_futures import evaluate as _evaluate
from kite_connect.options.signal_futures import record as _record
from kite_connect.options.theory import BUY, CALL, PUT, SELL, intrinsic_value
from nse_engine.data import fo_store

logger = logging.getLogger(__name__)

ROUND = "O4"
CAPITAL = 750_000.0                                    # the 0.25 overlay's share of the Rs 30 lakh book


def _hash(cfg: Any, kind: str) -> str:
    payload = {k: v for k, v in asdict(cfg).items() if k not in ("name", "start", "end")}
    payload.update(fno_cost_model=FNO_COST_MODEL_VERSION, strategy=kind)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class VrpConfig:
    """Y2: long index + short ATM straddle + long put near ``put_moneyness`` x spot, one expiry at a time."""

    name: str
    symbol: str = "NIFTY"
    target_dte: int = 30
    min_dte: int = 25
    max_dte: int = 35
    put_moneyness: float = 0.85
    slippage_bp: float = 1.0                           # the future's leg; options use OptionsConfig's slippage
    initial_capital: float = CAPITAL
    start: str = "2013-01-01"
    end: str = "2025-12-31"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def config_hash(self) -> str:
        return _hash(self, "volatility risk premium")


@dataclass(frozen=True)
class OvernightConfig:
    """Y3: long NIFTY close to open, ``step`` of capital per condition met on the previous session's closes."""

    name: str
    symbol: str = "NIFTY"
    mean_sessions: int = 20
    step: float = 1.0 / 3.0
    cash_rate: float = 0.065
    slippage_bp: float = 1.0
    initial_capital: float = CAPITAL
    start: str = "2013-01-01"
    end: str = "2025-12-31"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def config_hash(self) -> str:
        return _hash(self, "overnight with sentiment")


Config = Union[SignalConfig, VrpConfig, OvernightConfig]

CANDIDATES: Dict[str, Config] = {
    "Y1": SignalConfig("O4-Y1 option-expiry week (awesome-systematic-trading)", "expiry_week",
                       initial_capital=CAPITAL),
    "Y2": VrpConfig("O4-Y2 volatility risk premium (awesome-systematic-trading)"),
    "Y3": OvernightConfig("O4-Y3 overnight with sentiment (awesome-systematic-trading)"),
}


# ---------------------------------------------------------------- Y2

class _Chain:
    """One expiry's option prices: close if traded, else settlement; and which strikes traded each session."""

    def __init__(self, rows: pd.DataFrame):
        rows = rows.assign(price=np.where((rows["contracts"] > 0) & (rows["close"] > 0), rows["close"], rows["settle"]))
        self.price = rows.set_index(["date", "strike", "option_type"])["price"].to_dict()
        traded = rows[(rows["contracts"] > 0) & (rows["close"] > 0)]
        self.traded = traded.groupby(["date", "option_type"])["strike"].apply(lambda s: np.sort(s.unique())).to_dict()

    def at(self, date: pd.Timestamp, strike: float, option_type: str, default: float) -> float:
        px = self.price.get((date, strike, option_type), default)
        return float(px) if px and px > 0 else float(default)

    def nearest(self, date: pd.Timestamp, option_type: str, level: float) -> Optional[float]:
        strikes = self.traded.get((date, option_type))
        if strikes is None or not len(strikes):
            return None
        return float(strikes[int(np.argmin(np.abs(strikes - level)))])


def monthly_expiries(options: pd.DataFrame) -> Dict[pd.Timestamp, pd.Timestamp]:
    """The last option expiry of each calendar month (weeklies dropped, as in rounds 1 and 2), mapped to
    the session it settled on (``settlement_sessions``: holiday moves and relabelled contracts)."""
    settled = settlement_sessions(options)
    labels = pd.Series(sorted(settled))
    return {e: settled[e] for e in labels.groupby(labels.dt.to_period("M")).max()}


def _front(fut: _Futures, settled: Dict[pd.Timestamp, pd.Timestamp], d: pd.Timestamp) -> pd.Timestamp:
    """The nearest future still trading after ``d`` (its settlement session, not its label, is after d)."""
    return next(e for e in fut.expiries[d] if settled.get(e, e) > d)


def run_vrp(cfg: VrpConfig, store_dir: str, opt_cfg: OptionsConfig = OptionsConfig()) -> SignalResult:
    """Replay Y2 session by session (§ 5s)."""
    start, end = pd.Timestamp(cfg.start), pd.Timestamp(cfg.end)
    load_to = (end + pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    options = fo_store.load_options(store_dir, (start - pd.Timedelta(days=10)).strftime("%Y-%m-%d"), load_to, cfg.symbol)
    settle_on = monthly_expiries(options)
    by_expiry = {e: g for e, g in options[options["expiry"].isin(list(settle_on))].groupby("expiry")}
    futures = fo_store.load_futures(store_dir, (start - pd.Timedelta(days=10)).strftime("%Y-%m-%d"), load_to,
                                    cfg.symbol)
    fut, fut_settle = _Futures(futures), settlement_sessions(futures)
    dates = pd.DatetimeIndex(sorted(fut.expiries))
    window = dates[(dates >= start) & (dates <= end)]
    spot = fo_store.load_underlying(store_dir, cfg.symbol).reindex(dates).ffill()

    equity, f_units, contract, f_mark = cfg.initial_capital, 0.0, None, 0.0
    legs: List[Dict[str, Any]] = []
    chain: Optional[_Chain] = None
    cycle: Optional[Dict[str, Any]] = None
    pending: Optional[Dict[str, Any]] = None
    expired_today = False
    trades: List[Dict[str, Any]] = []
    curve, costs_total, rolls = {}, 0.0, 0
    for d in window:
        expired_today = False
        # 1. mark the future and the options at the close (settlement if untraded)
        if f_units:
            px = fut.at(d, contract, f_mark)
            equity += f_units * (px - f_mark)
            f_mark = px
        for leg in legs:
            px = chain.at(d, leg["strike"], leg["type"], leg["mark"])
            equity += leg["units"] * (px - leg["mark"])
            leg["mark"] = px
        # 2. yesterday's decision fills at today's closes
        if pending is not None:
            chain = _Chain(by_expiry[pending["expiry"]])
            refs = [chain.at(d, k, t, np.nan) for t, k, _ in pending["legs"]]
            if all(np.isfinite(refs)):
                units = equity / float(spot[d])
                cycle = {"side": "short vol", "entry_date": d, "expiry": pending["expiry"], "units": units,
                         "equity_before": equity, "strikes": [k for _, k, _ in pending["legs"]]}
                for (t, k, side), ref in zip(pending["legs"], refs):
                    fill = fill_price(ref, side, cfg.symbol, opt_cfg.slippage)
                    cost = abs(fill - ref) * units + option_charges(fill * units, side, d, cfg=opt_cfg.charges).total
                    equity -= cost
                    costs_total += cost
                    legs.append({"type": t, "strike": k, "units": units if side == BUY else -units, "mark": ref})
                if not f_units:
                    contract = _front(fut, fut_settle, d)
                    f_mark = fut.at(d, contract, np.nan)
                    f_units = equity / f_mark
                    cost = _trade_cost(f_units, f_mark, BUY, d, cfg, opt_cfg)
                    equity -= cost
                    costs_total += cost
            pending = None
        # 3. options settle on their settlement session against NIFTY 50's close, else the expiring future's
        if legs and cycle is not None and d == settle_on[cycle["expiry"]]:
            settle = float(spot[d]) if np.isfinite(spot[d]) else fut.at(d, cycle["expiry"], np.nan)
            for leg in legs:
                value = intrinsic_value(leg["type"], settle, leg["strike"])
                equity += leg["units"] * (value - leg["mark"])
                if leg["units"] > 0:
                    stt = exercise_stt(leg["type"], leg["strike"], settle, int(round(leg["units"])), d,
                                       cfg=opt_cfg.charges)
                    equity -= stt
                    costs_total += stt
            cycle.update(exit_date=d, exit_reason="expiry", settlement=settle)
            cycle["pnl_inr"] = equity - cycle.pop("equity_before")
            trades.append(cycle)
            legs, cycle, expired_today = [], None, True
        # 4. the future rolls at its settlement session's close, resized to equity
        if f_units and fut_settle.get(contract, contract) <= d:
            old_px = fut.at(d, contract, f_mark)
            contract = _front(fut, fut_settle, d)
            f_mark = fut.at(d, contract, np.nan)
            new_units = equity / f_mark
            cost = (_trade_cost(f_units, old_px, SELL, d, cfg, opt_cfg)
                    + _trade_cost(new_units, f_mark, BUY, d, cfg, opt_cfg))
            f_units = new_units
            equity -= cost
            costs_total += cost
            rolls += 1
        curve[d] = equity
        # 5. decide on today's close: options are chosen the session after the last ones expire
        if not legs and pending is None and not expired_today:
            pending = _choose(d, float(spot[d]), settle_on, by_expiry, cfg)

    eq = pd.Series(curve, name="equity")
    rets = eq.pct_change().fillna(eq.iloc[0] / cfg.initial_capital - 1.0)
    metrics = _metrics(rets, trades, costs_total, rolls, 1.0)
    return SignalResult(cfg, rets, eq, trades, metrics, fo_store.data_hash(store_dir), store_dir=store_dir)


def _choose(d: pd.Timestamp, spot: float, settle_on: Dict[pd.Timestamp, pd.Timestamp],
            by_expiry: Dict[pd.Timestamp, pd.DataFrame], cfg: VrpConfig) -> Optional[Dict[str, Any]]:
    """The expiry settling closest to ``target_dte`` days away, within [min_dte, max_dte], and the three
    strikes, or None."""
    eligible = [e for e, at in settle_on.items() if cfg.min_dte <= (at - d).days <= cfg.max_dte and e in by_expiry]
    if not eligible or not np.isfinite(spot):
        return None
    expiry = min(eligible, key=lambda e: abs((settle_on[e] - d).days - cfg.target_dte))
    chain = _Chain(by_expiry[expiry][by_expiry[expiry]["date"] == d])
    call, put = chain.nearest(d, CALL, spot), chain.nearest(d, PUT, spot)
    wing = chain.nearest(d, PUT, cfg.put_moneyness * spot)
    if None in (call, put, wing):
        return None
    return {"expiry": expiry, "legs": [(CALL, call, SELL), (PUT, put, SELL), (PUT, wing, BUY)]}


# ---------------------------------------------------------------- Y3

def run_overnight(cfg: OvernightConfig, store_dir: str, opt_cfg: OptionsConfig = OptionsConfig()) -> SignalResult:
    """Replay Y3 night by night (§ 5s and its amendment)."""
    start, end = pd.Timestamp(cfg.start), pd.Timestamp(cfg.end)
    m = fo_store.load_market(store_dir)
    m = m[m["nifty50_open"].notna() & m["nifty50_close"].notna() & (m.index <= end)]
    close, vix = m["nifty50_close"], m["india_vix"].ffill()     # a missing VIX day keeps the last one
    met = ((close > close.rolling(cfg.mean_sessions).mean()).astype(float)
           + (vix < vix.rolling(cfg.mean_sessions).mean()).astype(float))
    exposure = (cfg.step * met).shift(1).fillna(0.0)    # decided on the previous session's closes
    dates = m.index[m.index >= start]

    equity = cfg.initial_capital
    trades: List[Dict[str, Any]] = []
    curve, costs_total, held = {}, 0.0, 0
    prev = None
    for d in dates:
        if prev is not None and exposure[prev] > 0:    # the night from prev's close to d's open
            units = exposure[prev] * equity / close[prev]
            before = equity
            equity += units * (m.at[d, "nifty50_open"] - close[prev])
            equity -= exposure[prev] * before * cfg.cash_rate / 252
            cost = (_trade_cost(units, close[prev], BUY, prev, cfg, opt_cfg)
                    + _trade_cost(units, m.at[d, "nifty50_open"], SELL, d, cfg, opt_cfg))
            equity -= cost
            costs_total += cost
            held += 1
            trades.append({"side": "long", "entry_date": prev, "exit_date": d, "exposure": exposure[prev],
                           "exit_reason": "open", "pnl_inr": equity - before})
        curve[d] = equity
        prev = d
    eq = pd.Series(curve, name="equity")
    rets = eq.pct_change().fillna(eq.iloc[0] / cfg.initial_capital - 1.0)
    metrics = _metrics(rets, trades, costs_total, 0, held / max(len(dates) - 1, 1))
    return SignalResult(cfg, rets, eq, trades, metrics, fo_store.data_hash(store_dir), store_dir=store_dir)


# ---------------------------------------------------------------- dispatch, gates, CLI

def run(cfg: Config, store_dir: str) -> SignalResult:
    if isinstance(cfg, VrpConfig):
        return run_vrp(cfg, store_dir)
    if isinstance(cfg, OvernightConfig):
        return run_overnight(cfg, store_dir)
    return run_strategy(cfg, store_dir)


def record(result: SignalResult, runs_dir: str) -> str:
    return _record(result, runs_dir, round_label=ROUND)


def evaluate(**kwargs) -> Dict[str, Any]:
    """§ 5s's gates: round 2's, with these candidates, N = the whole options family."""
    return _evaluate(candidates=CANDIDATES, label=ROUND, **kwargs)


def main(argv: Optional[List[str]] = None) -> int:
    from kite_connect.options.backtest import OPTIONS_RUNS_DIR

    ap = argparse.ArgumentParser(description="Options round 3 (O4): awesome-systematic-trading's F&O strategies")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="backtest and record candidates in the options registry")
    p.add_argument("--candidates", default=",".join(CANDIDATES))
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--fo-store", default="data/nse_engine/fo_store")
    p.add_argument("--runs-dir", default=OPTIONS_RUNS_DIR)
    p = sub.add_parser("evaluate", help="§ 5s's gates over the recorded runs")
    p.add_argument("--no-record", action="store_true", help="do not record the combined run in the book registry")
    p.add_argument("--out", default="data/nse_engine/o4_results.json")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cmd == "run":
        for key in args.candidates.split(","):
            cfg = CANDIDATES[key]
            res = run(replace(cfg, start=args.start or cfg.start, end=args.end or cfg.end), args.fo_store)
            print(key, record(res, args.runs_dir), {k: res.metrics[k] for k in ("sharpe", "cagr", "max_drawdown")})
        return 0
    report = evaluate(record_combined=not args.no_record)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in ("gate1", "selected", "verdict")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
