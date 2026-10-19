"""
Government bonds as a return stream (India tracker BD1, validation plan 5ag).

Three forms beside the book, each on two bases (the deployed rules' 2008-25 backtest with the drawdown rule, and
R4's 2008-25 walk-forward), at Rs 35 lakh:
- B1: the book's idle cash in a 10-year gilt ETF;
- B2: the same while gilts trend up (index above its 252-session average), else in a liquid ETF;
- B3: a fixed 25% gilt sleeve, rebalanced monthly.
The baseline, reported beside them and not a trial, parks idle cash in a liquid ETF (LN-T29's plan).

The bond cost model: gilt and liquid ETFs bought for delivery on Kite pay ``nse_engine.costs``'s statutory charges
with no STT (debt ETFs), plus a spread floor and the square-root impact on the ETF's traded value, capped at 5% of
it a day; the ETF's expense ratio comes off its index's return.  The book's path is as recorded: bonds change none
of its decisions.

    python -m nse_engine.bond_stream bd1        # B1-B3 and the baseline, gates, recorded in runs_bonds
    python -m nse_engine.bond_stream holdout    # BD3: the family's 2026 holdout, opened once (lock in runs_bonds)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from nse_engine.config import CostConfig
from nse_engine.costs import impact_bps, statutory_cost

logger = logging.getLogger(__name__)

RUNS_DIR = "data/nse_engine/runs_bonds"
BACKTEST_RUN = "20261017T110545845367Z_2cb1c053"      # deployed rules + the drawdown rule, 2007-25, cost model 4
WALK_FORWARD = "data/nse_engine/wf_stitched_r4b.json"  # R4's 2008-25 walk-forward (LN-T28b)
BOOK_RUNS = "data/nse_engine/runs"
INDEX_STORE = "data/nse_engine/store_ext2006"
WINDOW = ("2008-01-01", "2025-12-31")
#: BD3: the deployed book's recorded 2026 holdout run (opened 15 Sep, cost model 1 as recorded), opened once for
#: the bond family with its own lock (the engine's ``holdout.lock`` is keyed by window and serves the core).
HOLDOUT_RUN = "20260915T161516314162Z_679cbd0c"
HOLDOUT_WINDOW = ("2026-01-01", "2026-09-11")
HOLDOUT_LOCK = Path(RUNS_DIR) / "holdout.lock"
RF = 0.065
#: Before the store has NIFTY 50 TRI (2012), its price index plus this dividend yield stands in (plan 5ag, A).
NIFTY_DIVIDEND_YIELD = 0.013
GILT_SYMBOL, LIQUID_SYMBOL = "LTGILTBEES", "LIQUIDCASE"


@dataclass(frozen=True)
class BondConfig:
    """Plan 5ag's rules; ``mode`` "idle" (B1), "trend" (B2), "sleeve" (B3) or "liquid" (the baseline)."""

    name: str = "B1 idle cash in gilts"
    mode: str = "idle"
    capital_inr: float = 35e5
    gilt_ter: float = 0.0015
    liquid_ter: float = 0.0010
    gilt_spread_bps: float = 10.0
    liquid_spread_bps: float = 3.0
    gilt_adv_inr: float = 3.1e7          # LTGILTBEES's 2022-25 median traded value (A: assumed before 2022)
    liquid_adv_inr: float = 7.46e8       # LIQUIDCASE's median
    max_participation: float = 0.05
    buffer: float = 0.02                 # of NAV, never parked
    band: float = 0.05                   # of NAV: buy only when the gap to the target is wider
    trend_sessions: int = 252
    sleeve_weight: float = 0.25

    def config_hash(self) -> str:
        payload = {k: v for k, v in asdict(self).items() if k != "name"}
        payload.update(kind="bond_stream")
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


CONFIGS = {"B1": BondConfig(),
           "B2": BondConfig(name="B2 idle cash in gilts while they trend up", mode="trend"),
           "B3": BondConfig(name="B3 a fixed 25% gilt sleeve", mode="sleeve")}
BASELINE = BondConfig(name="baseline: idle cash in a liquid ETF", mode="liquid")


# ------------------------------------------------------------------ inputs

@dataclass
class Base:
    """One book: its daily returns and invested weight on its sessions."""

    name: str
    returns: pd.Series
    invested: pd.Series


def backtest_base(run: str = BACKTEST_RUN, runs_dir: str = BOOK_RUNS) -> Base:
    r = pd.read_csv(Path(runs_dir) / run / "returns.csv", index_col=0).squeeze("columns").astype("float64")
    r.index = pd.DatetimeIndex(r.index)
    g = pd.read_parquet(Path(runs_dir) / run / "weights.parquet").fillna(0.0).clip(lower=0.0).sum(axis=1)
    r = r[WINDOW[0]:WINDOW[1]]
    return Base("backtest", r, g.reindex(r.index).ffill().fillna(0.0))


def walk_forward_base(path: str = WALK_FORWARD, runs_dir: str = BOOK_RUNS) -> Base:
    doc = json.loads(Path(path).read_text())
    rs, gs = [], []
    for f in doc["folds"]:
        run = Path(runs_dir) / f["test_run_id"]
        r = pd.read_csv(run / "returns.csv", index_col=0).squeeze("columns").astype("float64")
        r.index = pd.DatetimeIndex(r.index)
        g = pd.read_parquet(run / "weights.parquet").fillna(0.0).clip(lower=0.0).sum(axis=1)
        lo, hi = pd.Timestamp(f["test_start"]), pd.Timestamp(f["test_end"])
        rs.append(r[(r.index >= lo) & (r.index <= hi)])
        gs.append(g[(g.index >= lo) & (g.index <= hi)])
    r = pd.concat(rs).sort_index()[WINDOW[0]:WINDOW[1]]
    return Base("walk_forward", r, pd.concat(gs).sort_index().reindex(r.index).ffill().fillna(0.0))


def bond_returns(sessions: pd.DatetimeIndex, cfg: BondConfig) -> pd.DataFrame:
    """Per-session returns of the gilt and liquid ETFs (index less expense ratio) and the trend signal known at
    the previous close."""
    from nse_engine.data.bond_store import load

    lv = load()[["gsec10", "rate1d"]].ffill()
    lv = lv.reindex(lv.index.union(sessions)).ffill().reindex(sessions)
    out = pd.DataFrame({"gilt": lv["gsec10"].pct_change() - cfg.gilt_ter / 252,
                        "liquid": lv["rate1d"].pct_change() - cfg.liquid_ter / 252}, index=sessions).fillna(0.0)
    # the trend counts NSE sessions (the store's calendar from 2006), not the funds' NAV dates
    cal = pd.DatetimeIndex(pd.read_parquet(Path(INDEX_STORE) / "calendar.parquet")["date"]).union(sessions)
    full = load()["gsec10"].dropna()
    full = full.reindex(full.index.union(cal)).ffill().reindex(cal)
    trend = (full > full.rolling(cfg.trend_sessions, min_periods=cfg.trend_sessions).mean())
    out["trend_up"] = trend.shift(1).reindex(sessions).fillna(False).astype(bool)
    return out


def nifty_tri(sessions: pd.DatetimeIndex, store: str = INDEX_STORE) -> pd.Series:
    """NIFTY 50 TRI returns on ``sessions``; before the TRI exists, price returns plus the dividend yield (A)."""
    from nse_engine.data.external import read_index_history
    from nse_engine.data.panel import backfill_column, build_index_close, read_indices

    ic = build_index_close(read_indices(store, sessions[0] - pd.Timedelta(days=10), sessions[-1]), sessions,
                           ("NIFTY50", "NIFTY50_TRI"))
    # before NSE's daily index files (Feb 2012), NIFTY 50 comes from the store's cached external history
    ic, _ = backfill_column(ic, "NIFTY50", read_index_history(store, "NIFTY50"))
    tri = ic["NIFTY50_TRI"].pct_change() if "NIFTY50_TRI" in ic else pd.Series(np.nan, index=sessions)
    price = ic["NIFTY50"].pct_change() + NIFTY_DIVIDEND_YIELD / 252
    return tri.where(tri.notna(), price).fillna(0.0)


# ------------------------------------------------------------------ costs

def trade_cost(value: float, side: str, day, cfg: BondConfig, instrument: str) -> float:
    """The bond cost model: statutory charges (no STT on debt ETFs) plus spread and impact, in rupees."""
    if value <= 0:
        return 0.0
    gilt = instrument == "gilt"
    symbol = GILT_SYMBOL if gilt else LIQUID_SYMBOL
    costs = replace(CostConfig(), spread_floor_bps=cfg.gilt_spread_bps if gilt else cfg.liquid_spread_bps)
    adv = cfg.gilt_adv_inr if gilt else cfg.liquid_adv_inr
    return statutory_cost(value, side, day, symbol=symbol) + value * impact_bps(value, adv, costs) / 1e4


def equity_cost(value: float, side: str, day) -> float:
    """B3's rebalancing of the book: statutory charges of equity delivery plus a 3 bp floor (no symbol)."""
    return statutory_cost(value, side, day) + value * 3.0 / 1e4 if value > 0 else 0.0


# ------------------------------------------------------------------ the simulation

@dataclass
class Result:
    config: BondConfig
    base: str
    returns: pd.Series
    equity: pd.Series
    daily: pd.DataFrame
    alone: pd.Series
    metrics: Dict = field(default_factory=dict)
    run_id: str = ""
    run_dir: str = ""
    data_hash: str = ""


def simulate(base: Base, bonds: pd.DataFrame, cfg: BondConfig) -> Result:
    """The book plus the bond position, session by session, in rupees."""
    nav = cfg.capital_inr
    held: Dict[str, float] = {"gilt": 0.0, "liquid": 0.0}
    book = nav                                   # B3: the book's pot
    rows: List[Dict] = []
    alone: List[Tuple[pd.Timestamp, float]] = []
    month = None
    for day, r in base.returns.items():
        rb = bonds.loc[day]
        # 1. the day moves: the book (as recorded) and what is held in each ETF
        if cfg.mode == "sleeve":
            book *= 1.0 + r
            held["gilt"] *= 1.0 + rb["gilt"]
            nav = book + held["gilt"]
        else:
            pnl = nav * r + held["gilt"] * rb["gilt"] + held["liquid"] * rb["liquid"]
            held["gilt"] *= 1.0 + rb["gilt"]
            held["liquid"] *= 1.0 + rb["liquid"]
            nav += pnl
        # 2. targets at the close
        cost, traded, capped = 0.0, 0.0, 0
        want = held.copy()
        if cfg.mode == "sleeve":
            if day.to_period("M") != month:
                month = day.to_period("M")
                want["gilt"] = cfg.sleeve_weight * nav
        else:
            room = max((1.0 - float(base.invested.loc[day]) - cfg.buffer) * nav, 0.0)
            inst = {"idle": "gilt", "liquid": "liquid"}.get(cfg.mode) or ("gilt" if rb["trend_up"] else "liquid")
            other = "liquid" if inst == "gilt" else "gilt"
            want[other] = 0.0
            if held[inst] > room:
                want[inst] = room                          # the book needs the cash: sell now
            elif room - held[inst] > cfg.band * nav or held[other] > 0:
                want[inst] = room
        # 3. trade, at most 5% of the ETF's traded value a day (the rest waits)
        for inst in ("gilt", "liquid"):
            delta = want[inst] - held[inst]
            if abs(delta) < 1.0:
                continue
            cap = cfg.max_participation * (cfg.gilt_adv_inr if inst == "gilt" else cfg.liquid_adv_inr)
            if abs(delta) > cap:
                delta, capped = math.copysign(cap, delta), capped + 1
            c = trade_cost(abs(delta), "BUY" if delta > 0 else "SELL", day, cfg, inst)
            if cfg.mode == "sleeve":
                c += equity_cost(abs(delta), "SELL" if delta > 0 else "BUY", day)   # the book's side
                book -= delta
            held[inst] += delta
            cost += c
            traded += abs(delta)
        if cfg.mode == "sleeve":
            book -= cost
            nav = book + held["gilt"]
        else:
            nav -= cost
        # the stream alone: a rupee in the instrument the configuration holds, less its switching costs
        inst_now = "liquid" if cfg.mode == "liquid" or (cfg.mode == "trend" and not rb["trend_up"]) else "gilt"
        alone.append((day, float(rb[inst_now])))
        rows.append({"date": day, "nav": nav, "gilt": held["gilt"] / nav, "liquid": held["liquid"] / nav,
                     "idle": 1.0 - float(base.invested.loc[day]) if cfg.mode != "sleeve" else None,
                     "cost": cost / nav, "traded": traded / nav, "capped": capped, "book_return": r})
    daily = pd.DataFrame(rows).set_index("date")
    equity = daily["nav"]
    returns = equity.pct_change().fillna(equity.iloc[0] / cfg.capital_inr - 1.0)
    a = pd.Series(dict(alone)).sort_index()
    if cfg.mode == "trend":   # each switch pays a round trip on the rupee
        switches = (bonds["trend_up"].reindex(a.index).astype(int).diff().abs() > 0)
        a = a - switches * 2 * (cfg.gilt_spread_bps + cfg.liquid_spread_bps) / 2 / 1e4
    return Result(cfg, base.name, returns, equity, daily, a)


# ------------------------------------------------------------------ gates

def _summary(r: pd.Series, tri: pd.Series) -> Dict:
    from nse_engine.validation.diagnostics import alpha_beta
    from nse_engine.validation.dsr import performance_summary

    s = performance_summary(r, RF)
    a = alpha_beta(r, tri.reindex(r.index).fillna(0.0), rf_annual=RF)
    eq = (1 + r).cumprod()
    y08 = float((1 + r[r.index.year == 2008]).prod() - 1) if (r.index.year == 2008).any() else float("nan")
    return {"cagr": s["cagr"], "excess_sharpe": s["excess_sharpe"], "max_drawdown": s["max_drawdown"],
            "calmar": s["cagr"] / -s["max_drawdown"] if s["max_drawdown"] < 0 else float("nan"),
            "alpha_annual": a["alpha_annual"], "beta": a["beta"], "y2008": y08,
            "worst_drawdown_end": str((eq / eq.cummax() - 1).idxmin().date())}


def gates(results: Dict[str, Result], baseline: Result, book: pd.Series, tri: pd.Series) -> Dict:
    """Plan 5ag's gates for one base: gate 1 alone (DSR at the family's N), gate 2 beside the book, and B1/B2
    against the liquid baseline."""
    from nse_engine.validation.dsr import deflated_sharpe, performance_summary

    matrix = pd.DataFrame({k: v.alone for k, v in results.items()}).fillna(0.0)
    b = _summary(book, tri)
    base = _summary(baseline.returns, tri)
    out = {"book": b, "baseline_liquid": base, "configs": {}}
    for k, res in results.items():
        c = _summary(res.returns, tri)
        alone = performance_summary(res.alone, RF)
        dsr = float(deflated_sharpe(res.alone, trials_matrix=matrix, rf_annual=RF)["dsr"])
        g2 = {"cagr": c["cagr"] > b["cagr"], "excess_sharpe": c["excess_sharpe"] > b["excess_sharpe"],
              "alpha": c["alpha_annual"] > b["alpha_annual"],
              "max_drawdown": c["max_drawdown"] >= b["max_drawdown"] - 0.02}
        checks = {"gate1": bool(alone["excess_sharpe"] > 0 and dsr >= 0.95), "gate2": all(g2.values())}
        if res.config.mode in ("idle", "trend"):
            checks["beats_liquid"] = bool(c["excess_sharpe"] > base["excess_sharpe"])
        d = res.daily
        yrs = (d.index[-1] - d.index[0]).days / 365.25
        out["configs"][k] = {
            "combined": c, "alone": {"cagr": alone["cagr"], "excess_sharpe": alone["excess_sharpe"],
                                     "max_drawdown": alone["max_drawdown"], "dsr": dsr},
            "gate2_checks": g2, "checks": checks, "pass": all(checks.values()),
            "in_gilts_mean": float(d["gilt"].mean()), "turnover_a_year": float(d["traded"].sum() / yrs),
            "costs_a_year": float(d["cost"].sum() / yrs), "capped_days": int((d["capped"] > 0).sum()),
            "correlation_with_book": float(((1 + res.alone).resample("ME").prod() - 1).corr(
                (1 + book).resample("ME").prod() - 1)),
            "slices": {lab: _summary(res.returns[lo:], tri) for lab, lo in (("2013-25", "2013"), ("2017-25", "2017"))},
            "years": {int(y): round(float(v), 4) for y, v in ((1 + res.returns).groupby(res.returns.index.year).prod() - 1).items()},
        }
    out["book_years"] = {int(y): round(float(v), 4) for y, v in ((1 + book).groupby(book.index.year).prod() - 1).items()}
    return out


def record(res: Result, extra: Dict, runs_dir: str = RUNS_DIR) -> str:
    """The configuration's backtest-base run into the bond family's registry, with both bases' daily books."""
    from kite_connect.options.backtest import _run_id
    from nse_engine.validation.trials import record_result

    res.run_id = res.run_id or _run_id(res.config.config_hash())
    run_dir = record_result(res, tag=res.config.name, runs_dir=runs_dir,
                            window=(str(res.returns.index[0].date()), str(res.returns.index[-1].date())),
                            extra={"family": "bonds", "round": "BD1", **extra})
    res.daily.to_csv(Path(run_dir) / "daily.csv")
    (Path(run_dir) / "config.json").write_text(json.dumps(asdict(res.config), indent=2, sort_keys=True))
    return run_dir


def bd1(record_runs: bool = True) -> Dict:
    """Plan 5ag: B1-B3 and the liquid baseline on both bases, the gates, the capital diagnostics."""
    from nse_engine.data.bond_store import ROOT

    bases = [backtest_base(), walk_forward_base()]
    manifest = json.loads((ROOT / "manifest.json").read_text())
    out: Dict = {"bond_store": manifest["data_hash"], "bases": {}, "verdict": {}, "diagnostics": {}}
    sims: Dict[str, Dict[str, Result]] = {}
    for base in bases:
        bonds = bond_returns(base.returns.index, BondConfig())
        tri = nifty_tri(base.returns.index)
        res = {k: simulate(base, bonds, cfg) for k, cfg in CONFIGS.items()}
        baseline = simulate(base, bonds, BASELINE)
        out["bases"][base.name] = gates(res, baseline, base.returns, tri)
        sims[base.name] = res
        for label, cap in (("Rs 1 crore", 1e7), ("Rs 2 crore", 2e7)):
            for k in ("B1", "B2"):
                m = simulate(base, bonds, replace(CONFIGS[k], capital_inr=cap))
                out["diagnostics"][f"{base.name}, {k}, {label}"] = _summary(m.returns, tri)
    for k in CONFIGS:
        out["verdict"][k] = all(out["bases"][b]["configs"][k]["pass"] for b in out["bases"])
        if record_runs:
            res = sims["backtest"][k]
            res.data_hash = manifest["data_hash"]
            res.metrics = out["bases"]["backtest"]["configs"][k]["combined"]
            run_dir = record(res, {"passed": out["verdict"][k],
                                   "walk_forward": out["bases"]["walk_forward"]["configs"][k]["combined"]})
            sims["walk_forward"][k].daily.to_csv(Path(run_dir) / "daily_walk_forward.csv")
            out["verdict"][f"{k} run"] = run_dir
    return out


def blend(book: pd.Series, gilt: pd.Series, weight: float = 0.25) -> pd.Series:
    """The costless sleeve: (1 - weight) x the book + weight x gilts, rebalanced on each month's first session."""
    a, b, out, month = 1.0 - weight, weight, [], None
    for day in book.index:
        if day.to_period("M") != month:
            month = day.to_period("M")
            total = a + b
            a, b = (1.0 - weight) * total, weight * total
        before = a + b
        a *= 1.0 + book[day]
        b *= 1.0 + gilt[day]
        out.append((a + b) / before - 1.0)
    return pd.Series(out, index=book.index)


def holdout(lock_path: Path = HOLDOUT_LOCK, runs_dir: str = BOOK_RUNS) -> Dict:
    """BD3 (plan 5ag): the bond family's 2026 holdout, opened once.  B3 is judged (shallower MaxDD than the book;
    return within 1 point of the costless blend); B1, B2 and the liquid baseline are reported beside it.  The
    lock entry is written before anything runs, so a crash still counts as the look."""
    from datetime import datetime, timezone

    from nse_engine.validation.holdout import HoldoutLockedError, _read_lock, _write_lock
    from nse_engine.validation.trials import git_state

    window = {"start": HOLDOUT_WINDOW[0], "end": HOLDOUT_WINDOW[1]}
    lock = _read_lock(Path(lock_path))
    if any(e.get("window") == window for e in lock["evaluations"]):
        raise HoldoutLockedError(f"the bond family's 2026 holdout was already opened (lock {lock_path})")
    entry = {"window": window, "family": "bonds", "run": HOLDOUT_RUN, "status": "started",
             "configs": {k: c.config_hash() for k, c in CONFIGS.items()}, **git_state(),
             "created_at": datetime.now(timezone.utc).isoformat(), "metrics": {}}
    lock["evaluations"].append(entry)
    _write_lock(Path(lock_path), lock)

    r = pd.read_csv(Path(runs_dir) / HOLDOUT_RUN / "returns.csv", index_col=0).squeeze("columns").astype("float64")
    r.index = pd.DatetimeIndex(r.index)
    g = pd.read_parquet(Path(runs_dir) / HOLDOUT_RUN / "weights.parquet").fillna(0.0).clip(lower=0.0).sum(axis=1)
    r = r[window["start"]:window["end"]]
    base = Base("holdout_2026", r, g.reindex(r.index).ffill().fillna(0.0))
    bonds = bond_returns(r.index, BondConfig())
    tri = nifty_tri(r.index)

    def summary(x: pd.Series) -> Dict:
        s = _summary(x, tri)
        s.pop("y2008", None)
        return {"return": float((1 + x).prod() - 1), **s}

    out = {"book": summary(r), "gilts": summary(bonds["gilt"]),
           "baseline_liquid": summary(simulate(base, bonds, BASELINE).returns)}
    for k, cfg in CONFIGS.items():
        res = simulate(base, bonds, cfg)
        out[k] = {**summary(res.returns), "costs": float(res.daily["cost"].sum()),
                  "in_gilts_mean": float(res.daily["gilt"].mean())}
    mech = summary(blend(r, bonds["gilt"], CONFIGS["B3"].sleeve_weight))
    out["B3 costless blend"] = mech
    out["B3 agrees"] = {"shallower_maxdd": bool(out["B3"]["max_drawdown"] > out["book"]["max_drawdown"]),
                        "within_1pt_of_blend": bool(abs(out["B3"]["return"] - mech["return"]) <= 0.01)}
    out["B3 agrees"]["all"] = all(out["B3 agrees"].values())
    entry.update({"status": "completed", "completed_at": datetime.now(timezone.utc).isoformat(),
                  "metrics": out})
    _write_lock(Path(lock_path), lock)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nse_engine.bond_stream", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("bd1", help="plan 5ag: B1-B3 beside the book, gates, recorded")
    p.add_argument("--no-record", action="store_true", help="do not record the runs (debugging)")
    sub.add_parser("holdout", help="BD3: the bond family's 2026 holdout, opened once")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out = holdout() if args.cmd == "holdout" else bd1(record_runs=not args.no_record)
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
