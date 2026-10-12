"""
MCX commodity trend sleeve (tracker GT5, plan 5y): time-series momentum on six MCX minis, as the commodity family.

Moskowitz, Ooi & Pedersen (2012) and Hurst, Ooi & Pedersen (2017): a futures contract that rose over the past
year tends to keep rising.  On silver, crude oil, natural gas, aluminium, zinc and lead:

* at the close of each month's last MCX session, GT5-A takes the sign of the 252-session return and GT5-B the
  mean of the 21-, 63- and 252-session signs;
* each instrument targets s x equity x (40% / its EWMA volatility) / 6 of notional, in whole lots rounded to
  the nearest (a half towards zero), scaled down together when margin at 25% of notional would pass 80% of
  equity;
* orders fill at the next session's close at ``mcx_costs.fill_price`` with ``futures_charges``; lots are held
  to the next decision and rolled with GT2's continuous series.

Each instrument trades its mini.  Where MCX did not list the mini (before its first session; the crude and
base-metal minis from 2019-20 to 2023), the large contract stands in, sized in the mini's first lot (plan 5y
and its 12 Oct clarification).  Results go to their own registry (``data/nse_engine/runs_mcx``) and
``evaluate`` applies 5y's gates.  CLI::

    python -m nse_engine.mcx_trend run [--configs GT5-A,GT5-B] [--no-record]   # recorded runs: Kaggle, `mcx` task
    python -m nse_engine.mcx_trend evaluate

Tested 12 Oct 2026 on Kaggle (plan 5y): both FAIL.  GT5-A: CAGR 2.4%, Sharpe -0.06; GT5-B: 3.4%, -0.09 (over 6.5%).
Correlation with the book is about 0 and the combined MaxDD improves, but the sleeve earns less than the hurdle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MCX_RUNS_DIR = "data/nse_engine/runs_mcx"
MCX_STORE_DIR = "data/nse_engine/mcx_store"
BOOK_RF = 0.065
COMBINED_WEIGHT = 0.25
#: (traded mini, its large-contract stand-in): the review's six, gold and copper excluded (plan 5y).
INSTRUMENTS: Tuple[Tuple[str, str], ...] = (("SILVERMIC", "SILVER"), ("CRUDEOILM", "CRUDEOIL"),
                                            ("NATGASMINI", "NATURALGAS"), ("ALUMINI", "ALUMINIUM"),
                                            ("ZINCMINI", "ZINC"), ("LEADMINI", "LEAD"))
SPAN_GAP_DAYS = 7                # a mini's listing ends where its series skips more calendar days than this


@dataclass(frozen=True)
class TrendConfig:
    """Plan 5y: GT5-A reads the 252-session sign (MOP), GT5-B the mean of the 21/63/252 signs (HOP)."""

    name: str = "GT5-A 12-month time-series momentum (MOP)"
    horizons: Tuple[int, ...] = (252,)
    vol_target: float = 0.40
    vol_com: int = 60
    vol_min_returns: int = 63
    annualise: int = 261
    min_history: int = 252
    margin_rate: float = 0.25
    margin_cap: float = 0.80
    initial_capital: float = 500_000.0
    start: str = "2013-01-01"
    end: str = "2025-12-31"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def config_hash(self) -> str:
        from nse_engine.mcx_costs import MCX_COST_MODEL_VERSION

        payload = {k: v for k, v in asdict(self).items() if k not in ("name", "start", "end")}
        payload.update(kind="mcx_trend", mcx_cost_model=MCX_COST_MODEL_VERSION, instruments=INSTRUMENTS)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


CONFIGS: Dict[str, TrendConfig] = {
    "GT5-A": TrendConfig(),
    "GT5-B": TrendConfig(name="GT5-B multi-horizon trend (HOP)", horizons=(21, 63, 252)),
}


@dataclass
class TrendResult:
    config: TrendConfig
    returns: pd.Series
    equity: pd.Series
    trades: List[Dict[str, Any]]
    decisions: pd.DataFrame
    metrics: Dict[str, Any]
    data_hash: str
    run_id: str = ""
    run_dir: Optional[str] = None


# ---------------------------------------------------------------- the instruments

class Instrument:
    """A mini and its large stand-in on the MCX calendar: the contract held after each close, its marks and
    multiplier, and the one daily return series the signals read."""

    def __init__(self, mini: str, large: str, cont: pd.DataFrame, closes: Dict[Tuple[str, pd.Timestamp], Dict],
                 multipliers: pd.Series, calendar: pd.DatetimeIndex):
        self.name, self.large = mini, large
        self._closes, self._multipliers = closes, multipliers
        rows = {s: cont[cont["symbol"] == s].set_index("date").sort_index() for s in (mini, large)}
        self.lot = float(rows[mini]["multiplier"].iloc[0])           # the mini's first known lot
        spans, start = [], None
        dates = rows[mini].index
        for a, b in zip(dates, list(dates[1:]) + [None]):
            start = a if start is None else start
            if b is None or (b - a).days > SPAN_GAP_DAYS:
                spans.append((start, a if b is not None else pd.Timestamp.max))   # the last span stays open
                start = None
        after = {s: r["expiry"].shift(-1).where(r["rolled"], r["expiry"]).fillna(r["expiry"]) for s, r in rows.items()}
        held: List[Optional[Tuple[str, pd.Timestamp]]] = []
        ret, standin = np.zeros(len(calendar)), np.zeros(len(calendar), dtype=bool)
        observed = np.zeros(len(calendar), dtype=bool)
        prev: Optional[Tuple[str, pd.Timestamp]] = None
        for i, d in enumerate(calendar):
            span = next(((a, b) for a, b in spans if a <= d <= b), None)
            on_mini = span is not None and d < span[1]           # held into the next session as the mini
            src = mini if on_mini else large
            if src in rows and d in rows[src].index:
                prev = (src, after[src].loc[d])
            elif prev is not None and prev[0] != src and d in rows[large].index:
                prev = (large, after[large].loc[d])
            held.append(prev)
            standin[i] = not on_mini
            reader = mini if span is not None and d in rows[mini].index else large
            if d in rows[reader].index and np.isfinite(rows[reader].at[d, "ret"]):
                ret[i], observed[i] = float(rows[reader].at[d, "ret"]), True
        self.held = held
        self.standin = standin
        self.returns = pd.Series(ret, index=calendar)
        self.history = np.cumsum(observed)
        self.level = (1 + self.returns).cumprod()

    def multiplier(self, contract: Tuple[str, pd.Timestamp]) -> float:
        sym, expiry = contract
        return self.lot if sym == self.large else float(self._multipliers.get((sym, expiry), self.lot))

    def close(self, contract: Tuple[str, pd.Timestamp], day: pd.Timestamp) -> float:
        """The contract's raw close on ``day``, NaN when it did not print."""
        return self._closes.get(contract, {}).get(day, math.nan)


def prepare(store_dir: str = MCX_STORE_DIR) -> Dict[str, Any]:
    """The MCX calendar, the six instruments and the store's data hash."""
    from nse_engine.data.mcx_store import data_hash, implied_multipliers, load_continuous, load_futures

    symbols = [s for pair in INSTRUMENTS for s in pair]
    cont = load_continuous(store_dir)
    cont = cont[cont["symbol"].isin(symbols)]
    futures = load_futures(store_dir, symbols)
    calendar = pd.DatetimeIndex(sorted(futures["date"].unique()))
    closes: Dict[Tuple[str, pd.Timestamp], Dict] = {}
    for (sym, expiry), g in futures.groupby(["symbol", "expiry"]):
        closes[(sym, pd.Timestamp(expiry))] = dict(zip(g["date"], g["close"].astype("float64")))
    multipliers = implied_multipliers(futures)
    return {"calendar": calendar, "data_hash": data_hash(store_dir),
            "instruments": [Instrument(m, l, cont, closes, multipliers, calendar) for m, l in INSTRUMENTS]}


# ---------------------------------------------------------------- the simulation

def _round_half_to_zero(x: float) -> int:
    return int(math.copysign(math.ceil(abs(x) - 0.5), x)) if np.isfinite(x) else 0


def run_trend(cfg: TrendConfig, prepared: Optional[Dict[str, Any]] = None,
              store_dir: str = MCX_STORE_DIR) -> TrendResult:
    from nse_engine.mcx_costs import fill_price, futures_charges
    from nse_engine.metrics import compute_metrics

    prep = prepared or prepare(store_dir)
    cal: pd.DatetimeIndex = prep["calendar"]
    insts: List[Instrument] = prep["instruments"]
    n = len(insts)
    vols = {x.name: x.returns.ewm(com=cfg.vol_com, min_periods=cfg.vol_min_returns).std().to_numpy()
            * math.sqrt(cfg.annualise) for x in insts}
    s0 = int(cal.searchsorted(pd.Timestamp(cfg.start)))
    s1 = int(cal.searchsorted(pd.Timestamp(cfg.end), side="right")) - 1
    month_end = np.r_[cal.month[1:] != cal.month[:-1], False]

    equity = float(cfg.initial_capital)
    lots = {x.name: 0 for x in insts}
    contract: Dict[str, Optional[Tuple[str, pd.Timestamp]]] = {x.name: None for x in insts}
    mark = {x.name: math.nan for x in insts}
    pnl_by = {x.name: 0.0 for x in insts}
    costs = {"fees": 0.0, "slippage": 0.0, "roll_fees": 0.0, "roll_slippage": 0.0}
    trades: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    curve: Dict[pd.Timestamp, float] = {}
    plan: Optional[Dict[str, int]] = None

    def trade(x: Instrument, c: Tuple[str, pd.Timestamp], qty: int, t: int, reason: str) -> None:
        nonlocal equity
        d, ref = cal[t], x.close(c, cal[t])
        if not np.isfinite(ref):
            ref = mark[x.name]
        side = "BUY" if qty > 0 else "SELL"
        px = fill_price(ref, side)
        notional = abs(qty) * x.multiplier(c) * px
        fee = futures_charges(notional, side, d).total
        slip = abs(qty) * x.multiplier(c) * abs(px - ref)
        equity -= fee + slip
        roll = reason in ("roll", "switch")
        costs["roll_fees" if roll else "fees"] += fee
        costs["roll_slippage" if roll else "slippage"] += slip
        trades.append({"date": str(d.date()), "instrument": x.name, "contract": f"{c[0]} {c[1].date()}",
                       "side": side, "lots": abs(qty), "price": px, "notional": notional, "fee": fee,
                       "slippage": slip, "reason": reason})

    def settle(t: int, target: Optional[Dict[str, int]]) -> None:
        """At session ``t``'s close: rolls and switches into the contract held after it, then the plan."""
        for x in insts:
            new = x.held[t]
            want = target[x.name] if target is not None else lots[x.name]
            if new is None:
                continue
            old = contract[x.name]
            if old is not None and new != old and lots[x.name]:
                reason = "switch" if new[0] != old[0] else "roll"
                trade(x, old, -lots[x.name], t, reason)
                if want:
                    trade(x, new, want, t, reason if want == lots[x.name] else "rebalance")
            elif want != lots[x.name]:
                trade(x, new, want - lots[x.name], t, "rebalance")
            lots[x.name], contract[x.name] = want, new
            px = x.close(new, cal[t])
            mark[x.name] = px if np.isfinite(px) else mark[x.name]

    def decide(t: int) -> Dict[str, int]:
        """At the month's last close: the signal, the volatility target, whole lots and the margin cap."""
        raw, rows = {}, []
        for x in insts:
            c = x.held[t]
            px = x.close(c, cal[t]) if c is not None else math.nan
            px = px if np.isfinite(px) else mark[x.name]
            sigma = vols[x.name][t]
            if (x.history[t] < cfg.min_history or t < max(cfg.horizons) or not np.isfinite(sigma) or sigma <= 0
                    or not np.isfinite(px)):
                s = 0.0
            else:
                s = float(np.mean([np.sign(x.level.iloc[t] / x.level.iloc[t - h] - 1) for h in cfg.horizons]))
            unit = x.multiplier(c) * px if c is not None and np.isfinite(px) else math.nan
            raw[x.name] = s * equity * (cfg.vol_target / sigma) / n / unit if s else 0.0
            rows.append((x, s, sigma, unit))
        want = {k: _round_half_to_zero(v) for k, v in raw.items()}
        units = {x.name: u for x, _, _, u in rows}

        def margin(w: Dict[str, int]) -> float:
            return sum(abs(q) * units[k] for k, q in w.items() if q) * cfg.margin_rate

        scaled, factor = False, 1.0
        while margin(want) > cfg.margin_cap * equity and factor > 1e-3:
            scaled = True
            factor *= min(0.95, cfg.margin_cap * equity / margin(want))
            want = {k: _round_half_to_zero(v * factor) for k, v in raw.items()}
        decisions.append({"date": str(cal[t].date()), "equity": equity, "margin_scaled": scaled,
                          **{f"{x.name}_signal": s for x, s, _, _ in rows},
                          **{f"{x.name}_vol": round(float(v), 4) for x, _, v, _ in rows},
                          **{f"{x.name}_lots": want[x.name] for x in insts},
                          "gross_exposure": sum(abs(want[k]) * units[k] for k in want if want[k]) / equity})
        return want

    if s0 > 0:
        settle(s0 - 1, None)
        if month_end[s0 - 1]:
            plan = decide(s0 - 1)
    for t in range(s0, s1 + 1):
        for x in insts:                                    # the day's P&L on the lots held since the last close
            c = contract[x.name]
            if c is None or not lots[x.name]:
                continue
            px = x.close(c, cal[t])
            if np.isfinite(px) and np.isfinite(mark[x.name]):
                change = lots[x.name] * x.multiplier(c) * (px - mark[x.name])
                equity += change
                pnl_by[x.name] += change
                mark[x.name] = px
        settle(t, plan)
        plan = None
        curve[cal[t]] = equity
        if month_end[t] and t < s1:
            plan = decide(t)

    eq = pd.Series(curve, name="equity")
    rets = eq.pct_change().fillna(eq.iloc[0] / cfg.initial_capital - 1.0)
    dec = pd.DataFrame(decisions)
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    window = slice(s0, s1 + 1)
    m = compute_metrics(rets, eq, rf_annual=BOOK_RF, initial_capital=cfg.initial_capital)
    m.update(trades=len(trades), rolls=sum(x["reason"] in ("roll", "switch") for x in trades),
             cost_drag_annual=float(sum(costs.values()) / eq.mean() / years), costs_inr=costs,
             pnl_by_instrument={k: round(v, 2) for k, v in pnl_by.items()},
             zero_lot_months={x.name: round(float((dec[f"{x.name}_lots"] == 0).mean()), 3) for x in insts},
             lots_traded={x.name: int(sum(tr["lots"] for tr in trades if tr["instrument"] == x.name)) for x in insts},
             standin_share={x.name: round(float(x.standin[window].mean()), 3) for x in insts},
             average_gross_exposure=float(dec["gross_exposure"].mean()), margin_scaled_months=int(dec["margin_scaled"].sum()))
    return TrendResult(cfg, rets, eq, trades, dec, m, prep["data_hash"])


def record(result: TrendResult, runs_dir: str = MCX_RUNS_DIR) -> str:
    from kite_connect.options.backtest import _run_id
    from nse_engine.mcx_costs import MCX_COST_MODEL_VERSION
    from nse_engine.validation.trials import record_result

    cfg = result.config
    result.run_id = result.run_id or _run_id(cfg.config_hash())
    run_dir = record_result(result, tag=cfg.name, runs_dir=runs_dir, window=(cfg.start, cfg.end),
                            extra={"family": "commodity", "round": "GT5", "mcx_cost_model": MCX_COST_MODEL_VERSION})
    pd.DataFrame(result.trades).to_csv(Path(run_dir) / "trades.csv", index=False)
    result.decisions.to_csv(Path(run_dir) / "decisions.csv", index=False)
    return run_dir


# ---------------------------------------------------------------- the gates

def evaluate(runs_dir: str = MCX_RUNS_DIR, book_runs: str = "data/nse_engine/runs",
             equity_store: str = "data/nse_engine/store") -> Dict[str, Any]:
    """Plan 5y's pass rules (``nse_engine.validation.sleeve_gates``) for each configuration's latest run, at the
    commodity family's N, beside the live book; MCX returns are carried to NSE sessions."""
    from kite_connect.options.structures import _live_book_returns, _nifty_tri_returns
    from nse_engine.validation.sleeve_gates import carry_to, sleeve_gates
    from nse_engine.validation.trials import TrialRegistry

    reg = TrialRegistry(runs_dir)
    trials = reg.list_trials()
    book, book_run = _live_book_returns(book_runs)
    tri = _nifty_tri_returns(equity_store, book.index)
    mat = reg.returns_matrix(window=(TrendConfig.start, TrendConfig.end))
    mat = pd.DataFrame({c: carry_to(mat[c].dropna(), book.index) for c in mat.columns})
    out: Dict[str, Any] = {"round": "GT5", "live_book_run": book_run, "family_n": int(mat.shape[1]), "configs": {}}
    for key, cfg in CONFIGS.items():
        rows = trials[trials["tag"] == cfg.name]
        if rows.empty:
            continue
        run_id = str(rows["run_id"].iloc[-1])
        gates = sleeve_gates(carry_to(reg.load_returns(run_id), book.index), mat, book, tri, BOOK_RF, COMBINED_WEIGHT)
        out["book"] = gates.pop("book")
        manifest = json.loads((Path(runs_dir) / run_id / "manifest.json").read_text())
        mm = manifest.get("metrics", {})
        out["configs"][key] = {
            "run_id": run_id, **{k: gates[k] for k in ("dsr", "sleeve", "combined", "rules", "pass")},
            "reported": {"monthly_correlation_with_book": gates["monthly_correlation_with_book"], "years": gates["years"],
                         **{k: mm.get(k) for k in ("pnl_by_instrument", "zero_lot_months", "lots_traded", "standin_share",
                                                   "costs_inr", "cost_drag_annual", "rolls", "trades",
                                                   "average_gross_exposure", "margin_scaled_months")},
                         "data_hash": manifest.get("data_hash"), "runtime_origin": manifest.get("runtime", {}).get("origin")},
        }
    passed = [k for k, v in out["configs"].items() if v["pass"]]
    out["verdict"] = (f"PASS: paper-trade {', '.join(passed)} for at least 60 sessions" if passed
                      else "GT5 closed: no configuration passed its gates")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="MCX commodity trend sleeve (GT5, plan 5y)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--configs", default="GT5-A,GT5-B")
    p.add_argument("--no-record", action="store_true")
    p.add_argument("--start")
    p.add_argument("--end")
    sub.add_parser("evaluate")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cmd == "run":
        prep = prepare()
        for key in args.configs.split(","):
            cfg = replace(CONFIGS[key], start=args.start or TrendConfig.start, end=args.end or TrendConfig.end)
            res = run_trend(cfg, prep)
            where = "not recorded" if args.no_record else record(res)
            print(key, where, json.dumps({k: res.metrics.get(k) for k in (
                "cagr", "sharpe", "max_drawdown", "trades", "rolls", "average_gross_exposure")}, default=str))
        return 0
    print(json.dumps(evaluate(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
