"""
Backtest of the options sleeves (tracker O2) on the NSE F&O store, and their
evaluation as pre-registered in docs/nse_engine_validation_plan.md § 5q.

``run_sleeve`` replays one :class:`SleeveConfig` session by session: on each
cycle's decision day it picks the bear call spread from that day's close,
fills it a session later at the legs' closes (settlement price when a leg did
not trade) with ``fno_costs`` charges and slippage, marks it daily, exits early
when the rule says so, and settles it at expiry against the index close.

``evaluate`` applies gate 1 (each sleeve alone, options family) and gate 2
(the best passing sleeve as an overlay on E4, recorded in the book's
registry).  CLI::

    python -m kite_connect.options.backtest run --candidates A1,A2,B \\
        --fo-store data/nse_engine/fo_store --runs-dir data/nse_engine/runs_options
    python -m kite_connect.options.backtest evaluate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from kite_connect.options.fno_costs import (FNO_COST_MODEL_VERSION, exercise_stt, fill_price,
                                            option_charges)
from kite_connect.options.options_config import OptionsConfig
from kite_connect.options.sleeves import CANDIDATES, EVENT_DATES, EXTENDED_WINDOW, SLEEVE_WEIGHT, SleeveConfig
from kite_connect.options.strategies import max_pain
from kite_connect.options.theory import BUY, CALL, SELL, atm_strike
from nse_engine.data import fo_store

logger = logging.getLogger(__name__)

OPTIONS_RUNS_DIR = "data/nse_engine/runs_options"
BOOK_RUNS_DIR = "data/nse_engine/runs"
#: E4's recorded 2013-25 run on the current equity data, cost model 4, the base of gate 2 (§ 5q;
#: § 5q used the model-3 run 20261001T104724595094Z_93cf6c4d; tracker IC1).
E4_RUN_ID = "20261009T185901820673Z_93cf6c4d"
#: E4's 2007-25 run on store_ext2006, cost model 4, the base of the reported extended check
#: (§ 5q addendum used the model-3 run 20261001T070903385179Z_93cf6c4d; tracker IC1).
E4_EXT_RUN_ID = "20261009T191853176431Z_93cf6c4d"
BOOK_RF = 0.065


@dataclass
class SleeveResult:
    config: SleeveConfig
    returns: pd.Series
    equity: pd.Series
    cycles: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    data_hash: str = ""
    run_id: str = ""
    run_dir: Optional[str] = None


@dataclass
class _Position:
    expiry: pd.Timestamp
    short_k: float
    long_k: float
    qty: int
    cycle: Dict[str, Any]
    marks: Dict[float, float] = field(default_factory=dict)


class _Market:
    """Per-session option quotes of one symbol's monthly expiries, plus the index and futures."""

    def __init__(self, store_dir: str, cfg: SleeveConfig):
        lo = (pd.Timestamp(cfg.start) - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
        opts = fo_store.load_options(store_dir, lo, cfg.end, cfg.symbol)
        exp = pd.DatetimeIndex(opts["expiry"].unique())
        monthly = pd.Series(exp, index=exp).groupby([exp.year, exp.month]).max()
        self.expiries = sorted(pd.DatetimeIndex(monthly.to_numpy()))
        opts = opts[opts["expiry"].isin(self.expiries)]
        self.quotes = {d: g.set_index(["expiry", "option_type", "strike"]).sort_index()
                       for d, g in opts.groupby("date")}
        futs = fo_store.load_futures(store_dir, lo, cfg.end, cfg.symbol)
        self.future_settle = {(r.date, r.expiry): r.settle for r in futs.itertuples() if r.settle > 0}
        self.index = fo_store.load_underlying(store_dir, cfg.symbol)
        self.exact = fo_store.load_underlying(store_dir, cfg.symbol, exact_only=True)
        self.sessions = [d for d in sorted(self.quotes) if pd.Timestamp(cfg.start) <= d <= pd.Timestamp(cfg.end)]

    def chain(self, d: pd.Timestamp, expiry: pd.Timestamp, option_type: str = CALL) -> pd.DataFrame:
        q = self.quotes.get(d)
        if q is None:
            return pd.DataFrame()
        try:
            return q.loc[(expiry, option_type)]
        except KeyError:
            return pd.DataFrame()

    def price(self, d: pd.Timestamp, expiry: pd.Timestamp, strike: float) -> Tuple[float, float]:
        """(reference price, lot size): close if traded, else settlement; NaN when neither."""
        ch = self.chain(d, expiry)
        if strike not in ch.index:
            return math.nan, math.nan
        row = ch.loc[strike]
        if row["contracts"] > 0 and row["close"] > 0:
            return float(row["close"]), float(row["lot_size"])
        return (float(row["settle"]) if row["settle"] > 0 else math.nan), float(row["lot_size"])

    def spot(self, d: pd.Timestamp) -> float:
        return float(self.index.get(d, math.nan))

    def settlement(self, expiry: pd.Timestamp) -> float:
        """The index close (only an exact one, never the pre-2007 proxy), else the expiring future's settlement."""
        s = float(self.exact.get(expiry, math.nan))
        return s if np.isfinite(s) else float(self.future_settle.get((expiry, expiry), math.nan))


def _choose_strikes(cfg: SleeveConfig, mkt: _Market, d: pd.Timestamp, expiry: pd.Timestamp,
                    spot: float, sigma: float, mu: float) -> Tuple[Optional[float], Optional[float], str]:
    """(short, wing, reason) for the decision on ``d``; a None strike means no trade."""
    ch = mkt.chain(d, expiry)
    if ch.empty:
        return None, None, "no call chain for the expiry"
    eligible = sorted(ch.index[(ch["contracts"] > 0) & (ch["close"] > 0)])
    n = (expiry - d).days
    sd_n = sigma * math.sqrt(n)

    def first_above(level: float, inclusive: bool = False) -> Optional[float]:
        return next((k for k in eligible if (k >= level if inclusive else k > level)), None)

    if cfg.rule == "sd":
        short = first_above(spot * (1 + mu * n + cfg.short_sd * sd_n))
        wing = first_above(spot * (1 + mu * n + cfg.wing_sd * sd_n))
    else:
        puts = mkt.chain(d, expiry, "PE")
        strikes = sorted(set(ch.index) | set(puts.index))
        call_oi = ch["oi"].reindex(strikes).fillna(0.0)
        put_oi = puts["oi"].reindex(strikes).fillna(0.0) if not puts.empty else pd.Series(0.0, index=strikes)
        mp, _ = max_pain(strikes, call_oi.to_numpy(), put_oi.to_numpy())
        short = first_above(mp * (1 + cfg.max_pain_buffer), inclusive=True)
        if short is not None and short <= spot:
            return None, None, "max-pain band not above spot"
        wing = first_above(short + cfg.wing_distance_sd * sd_n * spot, inclusive=True) if short else None
    if short is None or wing is None or wing <= short:
        return None, None, "no eligible strike"
    return short, wing, ""


def run_sleeve(cfg: SleeveConfig, store_dir: str, opt_cfg: OptionsConfig = OptionsConfig()) -> SleeveResult:
    mkt = _Market(store_dir, cfg)
    log_ret = np.log(mkt.index).diff()
    events = pd.DatetimeIndex(EVENT_DATES)
    cash = float(cfg.initial_capital)
    pos: Optional[_Position] = None
    pending: Optional[Tuple[str, Dict[str, Any]]] = None   # ("entry" | "exit", details), filled next session
    decided: set = set()
    cycles: List[Dict[str, Any]] = []
    equity: Dict[pd.Timestamp, float] = {}

    def charges(value: float, side: str, d: pd.Timestamp) -> float:
        return option_charges(value, side, d, 1, opt_cfg.charges).total

    for d in mkt.sessions:
        # 1. fills decided on the previous session
        if pending is not None:
            kind, info = pending
            pending = None
            if kind == "entry":
                cyc = info["cycle"]
                ps, lot = mkt.price(d, cyc["expiry"], cyc["short"])
                pl, _ = mkt.price(d, cyc["expiry"], cyc["wing"])
                if not (np.isfinite(ps) and np.isfinite(pl) and np.isfinite(lot)):
                    cyc["action"] = "skipped: no price at fill"
                else:
                    fs = fill_price(ps, SELL, cfg.symbol, opt_cfg.slippage)
                    fl = fill_price(pl, BUY, cfg.symbol, opt_cfg.slippage)
                    credit = fs - fl
                    max_loss = (cyc["wing"] - cyc["short"]) - credit
                    spreads = int(cfg.deploy_fraction * cash // (max_loss * lot)) if credit > 0 else 0
                    if credit <= 0:
                        cyc["action"] = "skipped: no net credit"
                    elif spreads <= 0:
                        cyc["action"] = "skipped: sleeve too small"
                    else:
                        qty = int(spreads * lot)
                        cost = charges(fs * qty, SELL, d) + charges(fl * qty, BUY, d)
                        cash += credit * qty - cost
                        cyc.update(action="traded", fill_date=d, credit=credit, qty=qty, lots=spreads,
                                   lot_size=lot, max_loss_inr=max_loss * qty, charges=cost)
                        pos = _Position(cyc["expiry"], cyc["short"], cyc["wing"], qty, cyc,
                                        {cyc["short"]: ps, cyc["wing"]: pl})
            elif pos is not None:
                ps, _ = mkt.price(d, pos.expiry, pos.short_k)
                pl, _ = mkt.price(d, pos.expiry, pos.long_k)
                ps = ps if np.isfinite(ps) else pos.marks[pos.short_k]
                pl = pl if np.isfinite(pl) else pos.marks[pos.long_k]
                bs = fill_price(ps, BUY, cfg.symbol, opt_cfg.slippage)
                sl = fill_price(pl, SELL, cfg.symbol, opt_cfg.slippage)
                cost = charges(bs * pos.qty, BUY, d) + charges(sl * pos.qty, SELL, d)
                cash -= (bs - sl) * pos.qty + cost
                pos.cycle.update(exit="early (short strike ATM)", exit_date=d, debit=bs - sl,
                                 charges=pos.cycle["charges"] + cost)
                pos = None

        # 2. expiry: cash settlement against the index close
        if pos is not None and d >= pos.expiry:
            s = mkt.settlement(pos.expiry)
            s = s if np.isfinite(s) else mkt.spot(d)
            pay = max(0.0, s - pos.long_k) - max(0.0, s - pos.short_k)
            stt = exercise_stt(CALL, pos.long_k, s, pos.qty, pos.expiry, opt_cfg.charges)
            cash += pay * pos.qty - stt
            pos.cycle.update(exit="expiry", exit_date=pos.expiry, settlement=s, debit=-pay,
                             charges=pos.cycle["charges"] + stt)
            pos = None

        # 3. mark to market
        value = 0.0
        if pos is not None:
            for k, sign in ((pos.short_k, -1), (pos.long_k, 1)):
                p, _ = mkt.price(d, pos.expiry, k)
                if np.isfinite(p):
                    pos.marks[k] = p
                value += sign * pos.marks[k] * pos.qty
        equity[d] = cash + value

        # 4. decisions on today's close
        spot = mkt.spot(d)
        if pos is not None and pending is None and cfg.early_exit_atm and d < pos.expiry and np.isfinite(spot):
            strikes = mkt.chain(d, pos.expiry).index
            if len(strikes) and atm_strike(spot, strikes) >= pos.short_k:
                pending = ("exit", {})
        if pos is not None or pending is not None:
            continue
        nxt = next((e for e in mkt.expiries if e > d), None)
        if nxt is None or nxt in decided or not (1 <= (nxt - d).days <= cfg.entry_max_dte):
            continue
        hist = log_ret.loc[:d].dropna().tail(cfg.vol_lookback)
        if not np.isfinite(spot) or len(hist) < cfg.vol_lookback:
            continue                                       # not a valid decision day; try the next session
        decided.add(nxt)
        cyc: Dict[str, Any] = {"expiry": nxt, "decision_date": d, "spot": spot}
        cycles.append(cyc)
        if cfg.skip_events and ((events >= d) & (events <= nxt)).any():
            cyc["action"] = "skipped: event"
            continue
        sigma, mu = float(hist.std(ddof=1)), float(hist.mean())
        short, wing, why = _choose_strikes(cfg, mkt, d, nxt, spot, sigma, mu)
        cyc.update(sigma=sigma, mu=mu, short=short, wing=wing)
        if short is None:
            cyc["action"] = f"skipped: {why}"
            continue
        pending = ("entry", {"cycle": cyc})

    eq = pd.Series(equity).sort_index()
    rets = eq.pct_change().fillna(0.0)
    for cyc in cycles:
        if cyc.get("action") == "traded":
            cyc["pnl_inr"] = (cyc["credit"] - cyc["debit"]) * cyc["qty"] - cyc["charges"]
    return SleeveResult(cfg, rets, eq, cycles, _sleeve_metrics(rets, cycles), fo_store.data_hash(store_dir))


def _sleeve_metrics(rets: pd.Series, cycles: List[Dict[str, Any]]) -> Dict[str, Any]:
    from nse_engine.validation.dsr import performance_summary

    m = performance_summary(rets, rf_annual=0.0)
    traded = [c for c in cycles if c.get("action") == "traded"]
    pnl = [c["pnl_inr"] for c in traded]
    m.update(sharpe=m["excess_sharpe"], calmar=(m["cagr"] / abs(m["max_drawdown"]) if m["max_drawdown"] else None),
             cycles=len(cycles), traded=len(traded),
             skipped={k: sum(1 for c in cycles if c.get("action") == k) for k in
                      sorted({c.get("action") for c in cycles if c.get("action") != "traded"})},
             win_rate=(sum(p > 0 for p in pnl) / len(pnl)) if pnl else None,
             worst_cycle_inr=min(pnl) if pnl else None, best_cycle_inr=max(pnl) if pnl else None,
             charges_inr=sum(c["charges"] for c in traded),
             early_exits=sum(1 for c in traded if str(c.get("exit", "")).startswith("early")))
    return m


def _run_id(config_hash: str) -> str:
    """The engine's run id convention: UTC timestamp and the config hash prefix."""
    return f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_{config_hash[:8]}"


def record(result: SleeveResult, runs_dir: str) -> str:
    """Write the run to the options registry (U32), with its cycles."""
    from nse_engine.validation.trials import record_result

    cfg = result.config
    result.run_id = result.run_id or _run_id(cfg.config_hash())
    run_dir = record_result(result, tag=cfg.name, runs_dir=runs_dir, window=(cfg.start, cfg.end),
                            extra={"family": "options", "fno_cost_model": FNO_COST_MODEL_VERSION})
    pd.DataFrame(result.cycles).to_csv(Path(run_dir) / "cycles.csv", index=False)
    return run_dir


# ---------------------------------------------------------------- evaluation (§ 5q)

@dataclass(frozen=True)
class CombinedConfig:
    """The book with an options overlay: r = r(E4) + weight x r(sleeve)."""

    book_config_hash: str
    sleeve_config_hash: str
    weight: float
    start: str
    end: str

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, sort_keys=True)

    def config_hash(self) -> str:
        key = f"{self.book_config_hash}|{self.sleeve_config_hash}|{self.weight}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]


@dataclass
class _Recorded:
    config: Any
    returns: pd.Series
    metrics: Dict[str, Any]
    data_hash: str
    run_id: str = ""
    run_dir: Optional[str] = None


def overlay(book: pd.Series, sleeve_returns: pd.Series, weight: float) -> pd.Series:
    """Book returns plus ``weight`` x sleeve returns, the sleeve's equity carried onto the book's dates."""
    sleeve_eq = (1 + sleeve_returns).cumprod().reindex(book.index.union(sleeve_returns.index)).ffill()
    r_s = sleeve_eq.reindex(book.index).pct_change().fillna(0.0)
    return book + weight * r_s


def evaluate(options_runs: str = OPTIONS_RUNS_DIR, book_runs: str = BOOK_RUNS_DIR,
             start: str = "2013-01-01", end: str = "2025-12-31", record_combined: bool = True) -> Dict[str, Any]:
    from nse_engine.validation.dsr import deflated_sharpe, performance_summary
    from nse_engine.validation.pbo import cscv_pbo
    from nse_engine.validation.trials import TrialRegistry

    reg = TrialRegistry(options_runs)
    trials = reg.list_trials()
    mat = reg.returns_matrix(window=(start, end))
    n_opt = int(mat.shape[1])
    report: Dict[str, Any] = {"options_configurations": n_opt, "gate1": {}}
    if n_opt >= 2:
        report["options_pbo"] = {k: v for k, v in cscv_pbo(mat, n_splits=16, rf_annual=0.0).items()   # sleeve P&L
                                 if k != "logits"}
    passing = []
    order = {cfg.name: i for i, cfg in enumerate(CANDIDATES.values())}
    by_tag = trials.set_index("run_id")["tag"]
    for run_id in mat.columns:
        tag = by_tag[run_id]
        dsr = deflated_sharpe(mat[run_id], trials_matrix=mat if n_opt > 1 else None, rf_annual=0.0)
        summ = performance_summary(mat[run_id], rf_annual=0.0)
        ok = bool(summ["excess_sharpe"] > 0 and dsr["dsr"] >= 0.95)
        report["gate1"][tag] = {"run_id": run_id, "sharpe": summ["excess_sharpe"], "cagr": summ["cagr"],
                                "max_drawdown": summ["max_drawdown"], "dsr": dsr["dsr"], "pass": ok}
        if ok:
            passing.append((-summ["excess_sharpe"], order.get(tag, 99), tag, run_id))
    if not passing:
        report["selected"] = None
        report["verdict"] = "round 1 closed: no sleeve passed gate 1; the book's last configuration stays unused"
        report["extended"] = _extended(reg, by_tag, None, options_runs, book_runs, record_combined)
        return report

    _, _, tag, run_id = sorted(passing)[0]
    report["selected"] = tag
    book_reg = TrialRegistry(book_runs)
    e4_manifest = json.loads((Path(book_runs) / E4_RUN_ID / "manifest.json").read_text())
    sleeve_manifest = json.loads((Path(options_runs) / run_id / "manifest.json").read_text())
    e4, combined = _combine(book_reg, E4_RUN_ID, reg.load_returns(run_id), start, end)

    base, comb = performance_summary(e4, BOOK_RF), performance_summary(combined, BOOK_RF)

    def oos(r: pd.Series) -> float:                      # the walk-forward test years
        return performance_summary(r[r.index >= pd.Timestamp("2017-01-01")], BOOK_RF)["excess_sharpe"]

    checks = {
        "cagr": (comb["cagr"], comb["cagr"] >= 0.25),
        "excess_sharpe": (comb["excess_sharpe"], comb["excess_sharpe"] >= base["excess_sharpe"]),
        "max_drawdown": (comb["max_drawdown"], comb["max_drawdown"] >= base["max_drawdown"] - 0.02),
    }
    report["gate2"] = {"e4": base, "combined": comb, "checks": {k: {"value": v, "pass": p} for k, (v, p) in checks.items()},
                       "pass": all(p for _, p in checks.values()),
                       "reported": {"calmar_e4": base["cagr"] / abs(base["max_drawdown"]),
                                    "calmar_combined": comb["cagr"] / abs(comb["max_drawdown"]),
                                    "excess_sharpe_2017_25_e4": oos(e4), "excess_sharpe_2017_25_combined": oos(combined)}}
    if record_combined:
        report["gate2"]["run_dir"] = _record_combined(combined, comb, e4_manifest, sleeve_manifest, run_id, tag,
                                                      book_runs, (start, end))
        book_mat = book_reg.returns_matrix(data_hash=e4_manifest["data_hash"], window=(start, end),
                                           cost_model=int(e4_manifest.get("cost_model", 3)))
        combined_id = Path(report["gate2"]["run_dir"]).name
        n_total = int(book_mat.shape[1]) + n_opt
        report["gate2"]["reported"].update(
            book_configurations=int(book_mat.shape[1]), total_trials=n_total,
            dsr_total=deflated_sharpe(book_mat[combined_id], trials_matrix=book_mat, n_trials=n_total,
                                      rf_annual=BOOK_RF)["dsr"],
            book_pbo=float(cscv_pbo(book_mat, n_splits=16, rf_annual=BOOK_RF)["pbo"]))
    report["verdict"] = ("PASS: paper-trade the sleeve beside E4" if report["gate2"]["pass"]
                         else "FAIL: round 1 closed without re-tuning")
    report["extended"] = _extended(reg, by_tag, sleeve_manifest["config_hash"], options_runs, book_runs,
                                   record_combined)
    return report


def _combine(book_reg, book_run_id: str, sleeve: pd.Series, start: str, end: str) -> Tuple[pd.Series, pd.Series]:
    """(book returns on the window, book + sleeve overlay)."""
    book = book_reg.load_returns(book_run_id)
    book = book[(book.index >= pd.Timestamp(start)) & (book.index <= pd.Timestamp(end))]
    return book, overlay(book, sleeve, SLEEVE_WEIGHT)


def _record_combined(combined: pd.Series, metrics: Dict[str, Any], book_manifest: Dict[str, Any],
                     sleeve_manifest: Dict[str, Any], sleeve_run: str, tag: str, book_runs: str,
                     window: Tuple[str, str], label: str = "O2", weight: float = SLEEVE_WEIGHT) -> str:
    from nse_engine.costs import COST_MODEL_VERSION
    from nse_engine.validation.trials import LEGACY_COST_MODEL, record_result

    # The combined run is stamped with today's cost model, so its book half must be on it too (tracker IC1).
    book_model = int(book_manifest.get("cost_model") or LEGACY_COST_MODEL)
    if book_model != COST_MODEL_VERSION:
        raise ValueError(f"E4's base run is on cost model {book_model}, not {COST_MODEL_VERSION}: point "
                         "E4_RUN_ID / E4_EXT_RUN_ID at runs recorded under the current cost model")
    cfg = CombinedConfig(book_manifest["config_hash"], sleeve_manifest["config_hash"], weight, *window)
    rec = _Recorded(cfg, combined, metrics, book_manifest["data_hash"], run_id=_run_id(cfg.config_hash()))
    return record_result(rec, tag=f"{label} combined: E4 + {tag}", runs_dir=book_runs, window=window,
                         extra={"family": "book", "options_run": sleeve_run,
                                "fo_data_hash": sleeve_manifest.get("data_hash")})


def _extended(reg, by_tag: pd.Series, selected_hash: Optional[str], options_runs: str, book_runs: str,
              record_combined: bool) -> Dict[str, Any]:
    """§ 5q addendum: the same sleeves over 2007-25, and E4 2007-25 + the selected one.  Reported, not gating."""
    from nse_engine.validation.dsr import performance_summary
    from nse_engine.validation.trials import TrialRegistry

    mat = reg.returns_matrix(window=EXTENDED_WINDOW)
    out: Dict[str, Any] = {"window": EXTENDED_WINDOW, "sleeves": {}}
    for run_id in mat.columns:
        manifest = json.loads((Path(options_runs) / run_id / "manifest.json").read_text())
        out["sleeves"][by_tag[run_id]] = {"run_id": run_id, **performance_summary(mat[run_id], rf_annual=0.0),
                                          "cycles": {k: manifest["metrics"].get(k) for k in
                                                     ("traded", "skipped", "win_rate", "worst_cycle_inr")}}
        if manifest["config_hash"] != selected_hash:
            continue
        book_reg = TrialRegistry(book_runs)
        book_manifest = json.loads((Path(book_runs) / E4_EXT_RUN_ID / "manifest.json").read_text())
        e4, combined = _combine(book_reg, E4_EXT_RUN_ID, mat[run_id], *EXTENDED_WINDOW)
        comb = performance_summary(combined, BOOK_RF)
        out["combined"] = {"sleeve": by_tag[run_id], "e4": performance_summary(e4, BOOK_RF), "combined": comb}
        if record_combined:
            out["combined"]["run_dir"] = _record_combined(combined, comb, book_manifest, manifest, run_id,
                                                          by_tag[run_id], book_runs, EXTENDED_WINDOW)
    return out


# ---------------------------------------------------------------- CLI

def main(argv: Optional[List[str]] = None) -> int:
    from nse_engine.validation.trials import to_jsonable

    p = argparse.ArgumentParser(description="Options sleeve backtests (tracker O2)")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="backtest pre-registered sleeves and record them")
    r.add_argument("--candidates", default=",".join(CANDIDATES))
    r.add_argument("--fo-store", default="data/nse_engine/fo_store")
    r.add_argument("--runs-dir", default=OPTIONS_RUNS_DIR)
    r.add_argument("--start", help="window override, e.g. 2007-01-02 (plan 5q addendum)")
    r.add_argument("--end")
    r.add_argument("--out", default=None, help="JSON summary path")
    e = sub.add_parser("evaluate", help="gates 1 and 2 of § 5q")
    e.add_argument("--options-runs", default=OPTIONS_RUNS_DIR)
    e.add_argument("--book-runs", default=BOOK_RUNS_DIR)
    e.add_argument("--no-record", action="store_true", help="do not record the combined run")
    e.add_argument("--out", default="data/nse_engine/o2_results.json")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cmd == "run":
        out = {}
        for key in args.candidates.split(","):
            cfg = CANDIDATES[key]
            cfg = replace(cfg, start=args.start or cfg.start, end=args.end or cfg.end)
            res = run_sleeve(cfg, args.fo_store)
            out[key] = {"run_dir": record(res, args.runs_dir), "metrics": res.metrics}
            logger.info("%s: %s", key, json.dumps(to_jsonable(res.metrics)))
        text = json.dumps(to_jsonable(out), indent=2)
    else:
        text = json.dumps(to_jsonable(evaluate(args.options_runs, args.book_runs,
                                               record_combined=not args.no_record)), indent=2)
    if getattr(args, "out", None):
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
