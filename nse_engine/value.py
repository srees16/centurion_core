"""
Selective value / quality sleeve (tracker GT4, plan 5z): cheap stocks with real profits, as the value family.

Monthly, on the engine's universe at top 500, with GT1's quarterly results read point in time:

* at the close of each month's last session, a name with a trailing-twelve-month (TTM) result is eligible
  when its TTM profit, latest-quarter profit and TTM revenue are all above zero;
* GT4-A ranks the eligible names by earnings yield (TTM profit / market capitalisation); GT4-B by the mean
  of four percentile ranks: earnings yield, TTM margin (profit before tax / revenue), profit growth on the
  TTM a year earlier, and low 252-session volatility;
* the top 25 are held at equal weight, and a holding is sold only when it leaves the top 50;
* orders fill at the next session's open with the core's costs and fill limits; no stops, no regime gate.

Plan 5z's clarifications (11-12 Oct 2026) fix the translations: quarters of at most 100 days, four
consecutive quarters per TTM, the share count, the weights and the ranks of missing components.  Results go
to their own registry (``data/nse_engine/runs_value``) and ``evaluate`` applies 5z's gates.  CLI::

    python -m nse_engine.value run [--configs GT4-A,GT4-B] [--no-record]   # recorded runs: Kaggle, `value` task
    python -m nse_engine.value evaluate

Tested 12 Oct 2026 on Kaggle (plan 5z): both FAIL.  GT4-A: CAGR 14.7%, Sharpe 0.43, MaxDD -68%.  GT4-B: 16.8% /
0.69 / -41% with DSR 0.98, but a quarter of the book in it lowers CAGR, Sharpe and alpha (correlation 0.63).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VALUE_RUNS_DIR = "data/nse_engine/runs_value"
FUNDAMENTALS_DIR = "data/nse_engine/fundamentals"
COMBINED_WEIGHT = 0.25
BOOK_RF = 0.065
MAX_QUARTER_DAYS = 100           # 5z clarification: a filing counts as a quarter up to this period length
EP_AGREEMENT = 2.0               # 5z clarification: the two earnings yields must agree within this factor
_TTM_COLUMNS = ["basis", "ttm_profit", "ttm_revenue", "ttm_pbt", "latest_profit", "prev_ttm_profit", "ttm_eps",
                "shares", "shares_as_of"]


@dataclass(frozen=True)
class ValueConfig:
    """Plan 5z: GT4-A ranks by earnings yield (``rank="ep"``), GT4-B by the four-part composite."""

    name: str = "GT4-A cheap and profitable"
    rank: str = "ep"                 # "ep" (GT4-A) or "composite" (GT4-B)
    holdings: int = 25
    exit_rank: int = 50
    top_n: int = 500
    vol_window: int = 252
    initial_capital: float = 500_000.0
    start: str = "2013-01-01"
    end: str = "2025-12-31"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def config_hash(self) -> str:
        from nse_engine.costs import COST_MODEL_VERSION

        payload = {k: v for k, v in asdict(self).items() if k not in ("name", "start", "end")}
        payload.update(kind="value", cost_model=COST_MODEL_VERSION, max_quarter_days=MAX_QUARTER_DAYS)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


CONFIGS: Dict[str, ValueConfig] = {
    "GT4-A": ValueConfig(),
    "GT4-B": ValueConfig(name="GT4-B value plus quality (QMJ-lite)", rank="composite"),
}


@dataclass
class ValueResult:
    config: ValueConfig
    returns: pd.Series
    equity: pd.Series
    trades: List[Dict[str, Any]]
    decisions: pd.DataFrame
    holdings: pd.DataFrame
    metrics: Dict[str, Any]
    data_hash: str
    fundamentals_hash: str = ""
    run_id: str = ""
    run_dir: Optional[str] = None


# ---------------------------------------------------------------- fundamentals, point in time

def quarterly_results(results: pd.DataFrame) -> pd.DataFrame:
    """Parsed quarterly filings, with 5z's profit and a quarter number for the consecutive-quarter test.

    Profit is the owners' share where a consolidated filing gives it.  An owners' profit of exactly 0 beside
    a non-zero profit for the period is a blank field parsed as zero (5% of quarters), so it counts as not
    given (5z clarification of 12 Oct).
    """
    days = (results["period_end"] - results["period_start"]).dt.days
    q = results[results["status"].isin(["ok", "partial"]) & (days <= MAX_QUARTER_DAYS)].copy()
    blank = (q["pat_owners"] == 0) & (q["pat"].fillna(0) != 0)
    q["profit"] = q["pat_owners"].where(q["consolidated"] & q["pat_owners"].notna() & ~blank, q["pat"])
    q["qn"] = q["period_end"].dt.year * 4 + (q["period_end"].dt.month - 1) // 3
    return q


def ttm_table(quarters: pd.DataFrame, as_of: pd.Timestamp, names: Iterable[str],
              events: Optional["ShareEvents"] = None) -> pd.DataFrame:
    """One row per name with a TTM known at the close of ``as_of`` (5z and its 12 Oct clarification).

    The four latest quarters must be consecutive and all on one basis, consolidated first; the year-earlier
    TTM is the four quarters before them on the same basis.  ``ttm_eps`` sums their basic EPS, each restated
    per share at ``as_of`` through the split and bonus ``events``.  The share count is the latest quarter's
    that has one, with the period end it refers to.
    """
    from nse_engine.data.fundamentals import point_in_time

    pit = point_in_time(quarters[quarters["canonical"].isin(set(names))], as_of)
    if pit.empty:
        return pd.DataFrame(columns=_TTM_COLUMNS)
    pit = pit.drop_duplicates(["canonical", "consolidated", "qn"], keep="last")   # sorted by dissemination
    pit = pit.assign(lag=pit.groupby("canonical")["qn"].transform("max") - pit["qn"])
    restate = [events.factor(s, pe, as_of) if events else 1.0 for s, pe in zip(pit["canonical"], pit["period_end"])]
    pit = pit.assign(eps=pit["eps_basic"] * np.asarray(restate, dtype="float64"))
    window = pit[pit["lag"].between(0, 3)]
    consecutive = window.groupby("canonical")["lag"].nunique() == 4
    parts = []
    for basis in (True, False):
        b = pit[pit["consolidated"] == basis]
        cur = b[b["lag"].between(0, 3)]
        g = cur.groupby("canonical")
        part = pd.DataFrame({
            "n": cur[cur["profit"].notna()].groupby("canonical")["lag"].nunique(),
            "ttm_profit": g["profit"].sum(min_count=4),
            "ttm_revenue": g["revenue"].sum(min_count=4),
            "ttm_pbt": g["pbt"].sum(min_count=4),
            "ttm_eps": g["eps"].sum(min_count=4),
            "latest_profit": b[b["lag"] == 0].set_index("canonical")["profit"],
            "prev_ttm_profit": b[b["lag"].between(4, 7)].groupby("canonical")["profit"].sum(min_count=4),
        })
        part = part[part["n"] == 4].drop(columns="n")
        part["basis"] = "consolidated" if basis else "standalone"
        parts.append(part)
    table = pd.concat([parts[0], parts[1][~parts[1].index.isin(parts[0].index)]])
    table = table[table.index.isin(consecutive[consecutive].index)]
    shares = (pit[pit["shares"].notna()].sort_values(["qn", "disseminated"], kind="stable")
              .groupby("canonical").tail(1).set_index("canonical"))
    table["shares"] = shares["shares"]
    table["shares_as_of"] = shares["period_end"]
    return table[_TTM_COLUMNS]


class ShareEvents:
    """Split, bonus and consolidation ratios from the store (the books' share-count sources)."""

    def __init__(self, events: Optional[pd.DataFrame]):
        from kite_connect.trading.book_events import SHARE_SOURCES

        self._by: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        if events is None or not len(events):
            return
        e = events[events["source"].astype(int).isin(SHARE_SOURCES)].sort_values("date")
        for sym, g in e.groupby("symbol"):
            self._by[str(sym)] = (pd.DatetimeIndex(g["date"]).to_numpy(dtype="datetime64[ns]"),
                                  np.cumprod(g["factor"].to_numpy(dtype="float64")))

    def factor(self, symbol: str, after, upto) -> float:
        """Product of the factors dated after ``after`` up to ``upto``; shares then are shares / factor."""
        if symbol not in self._by or pd.isna(after):
            return 1.0
        d, cp = self._by[symbol]
        i = int(np.searchsorted(d, np.datetime64(pd.Timestamp(after), "ns"), side="right"))
        j = int(np.searchsorted(d, np.datetime64(pd.Timestamp(upto), "ns"), side="right"))
        return float((cp[j - 1] if j else 1.0) / (cp[i - 1] if i else 1.0))


# ---------------------------------------------------------------- ranking

def _pct_rank(x: pd.Series) -> pd.Series:
    """Percentile rank, higher is better; a missing value ranks lowest and ties take the average rank."""
    return x.replace([np.inf, -np.inf], np.nan).fillna(-np.inf).rank(method="average", pct=True)


def scores(table: pd.DataFrame, rank: str, volatility: pd.Series) -> pd.Series:
    """Eligible names and their score (higher is better): earnings yield, or GT4-B's composite.

    Eligible: TTM profit, latest-quarter profit and TTM revenue above zero, and the earnings yield from TTM
    profit / market capitalisation within ``EP_AGREEMENT`` of TTM EPS / printed close.  EPS is per share, so
    the check catches NSE pages with mis-scaled amounts or a wrong share count (5z clarification of 12 Oct).
    """
    ep_all = table["ttm_profit"] / table["mcap"]
    agree = (ep_all / (table["ttm_eps"] / table["price"])).between(1 / EP_AGREEMENT, EP_AGREEMENT)
    elig = table[(table["ttm_profit"] > 0) & (table["latest_profit"] > 0) & (table["ttm_revenue"] > 0)
                 & (table["mcap"] > 0) & agree]
    ep = elig["ttm_profit"] / elig["mcap"]
    if rank == "ep":
        return ep.astype("float64")
    margin = elig["ttm_pbt"] / elig["ttm_revenue"]
    growth = (elig["ttm_profit"] / elig["prev_ttm_profit"] - 1).where(elig["prev_ttm_profit"] > 0)
    safety = -volatility.reindex(elig.index)
    return sum(_pct_rank(x.astype("float64")) for x in (ep, margin, growth, safety)) / 4


def select(score: pd.Series, held: Iterable[str], holdings: int, exit_rank: int) -> List[str]:
    """Holdings still in the top ``exit_rank``, then the best-ranked others up to ``holdings``."""
    order = sorted(score.index, key=lambda s: (-float(score[s]), str(s)))
    rank = {s: i + 1 for i, s in enumerate(order)}
    keep = sorted((s for s in held if rank.get(s, math.inf) <= exit_rank), key=rank.get)[:holdings]
    return keep + [s for s in order if s not in keep][: holdings - len(keep)]


# ---------------------------------------------------------------- the simulation

def load_inputs(store_dir: Optional[str] = None, fundamentals_dir: str = FUNDAMENTALS_DIR,
                end: str = ValueConfig.end):
    """(market data from 2011, quarterly results, the fundamentals store's data hash)."""
    from nse_engine.config import EngineConfig
    from nse_engine.data.fundamentals import data_hash
    from runners.run_nse_engine import _load_data

    engine_cfg = EngineConfig().replace(end=end, **({"data.store_dir": store_dir} if store_dir else {}))
    data = _load_data(engine_cfg, data_start="2011-01-01")
    results = pd.read_parquet(Path(fundamentals_dir) / "results.parquet")
    return data, quarterly_results(results), data_hash(fundamentals_dir)


def run_value(cfg: ValueConfig, data=None, quarters: Optional[pd.DataFrame] = None, fundamentals_hash: str = "",
              store_dir: Optional[str] = None, fundamentals_dir: str = FUNDAMENTALS_DIR) -> ValueResult:
    from nse_engine.config import CostConfig, EngineConfig, PortfolioConfig, UniverseConfig
    from nse_engine.costs import cap_quantity, impact_bps, median_traded_value, statutory_cost
    from nse_engine.metrics import compute_metrics
    from nse_engine.universe import compute_universe_panel

    if data is None or quarters is None:
        data, quarters, fundamentals_hash = load_inputs(store_dir, fundamentals_dir, cfg.end)
    costs, min_trade = CostConfig(), PortfolioConfig().min_trade_value_inr
    ucfg = replace(UniverseConfig(), top_n_liquid=cfg.top_n, price_filter_unadjusted=True)
    mask = compute_universe_panel(data, ucfg, exclude=EngineConfig().sleeves.symbols).mask
    close = data.close.astype("float64")
    cols = list(close.columns)
    col = {s: j for j, s in enumerate(cols)}
    C = close.ffill().to_numpy()
    O = data.open.astype("float64").to_numpy()
    CU = data.close_unadj.reindex(columns=cols).astype("float64").ffill().to_numpy()
    ADV = median_traded_value(data.value.astype("float64"), costs.adv_lookback_days).to_numpy()
    vol = close.pct_change(fill_method=None).rolling(cfg.vol_window, min_periods=200).std() * math.sqrt(252)
    events = ShareEvents(data.corporate_events)
    sectors = dict(data.sectors or {})
    dates = close.index
    s0 = int(dates.searchsorted(pd.Timestamp(cfg.start)))
    s1 = int(dates.searchsorted(pd.Timestamp(cfg.end), side="right")) - 1
    month_end = np.r_[dates.month[1:] != dates.month[:-1], False]

    cash, positions = float(cfg.initial_capital), {}
    trades: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    held_log: List[Dict[str, Any]] = []
    plan: Optional[Tuple[List[str], float]] = None
    pending_exits: set = set()
    curve: Dict[pd.Timestamp, float] = {}
    invested: Dict[pd.Timestamp, float] = {}

    def charge(value: float, side: str, t: int, j: int) -> float:
        return statutory_cost(value, side, dates[t], costs.dp_charge_inr) + value * impact_bps(value, ADV[t - 1, j], costs) / 1e4

    def fill(sym: str, t: int, qty: int, reason: str) -> None:
        nonlocal cash
        j, px = col[sym], float(O[t, col[sym]])
        value = abs(qty) * px
        side = "BUY" if qty > 0 else "SELL"
        cost = charge(value, side, t, j)
        cash += -(value + cost) if qty > 0 else value - cost
        positions[sym] = positions.get(sym, 0) + qty
        if positions[sym] == 0:
            positions.pop(sym)
        trades.append({"date": str(dates[t].date()), "symbol": sym, "side": side, "shares": abs(qty),
                       "price": px, "value": value, "cost": cost, "reason": reason})

    def decide(t: int) -> None:
        """At the close of month-end session ``t``: rank, select and size the next open's targets."""
        nonlocal plan
        d = dates[t]
        members = [s for s in mask.columns[mask.iloc[t].to_numpy()] if s in col]
        table = ttm_table(quarters, d, members, events)
        divisor = np.array([events.factor(s, a, d) for s, a in zip(table.index, table["shares_as_of"])])
        table["price"] = np.array([CU[t, col[s]] for s in table.index], dtype="float64")
        table["mcap"] = table["shares"].to_numpy(dtype="float64") / divisor * table["price"].to_numpy()
        score = scores(table, cfg.rank, vol.iloc[t])
        targets = select(score, positions, cfg.holdings, cfg.exit_rank)
        equity = cash + sum(n * C[t, col[s]] for s, n in positions.items())
        plan = (targets, equity / cfg.holdings)
        decisions.append({"date": str(d.date()), "universe": len(members), "with_ttm": len(table),
                          "eligible": len(score), "holdings": len(targets),
                          "kept": sum(s in positions for s in targets), "equity": equity})
        held_log.extend({"date": str(d.date()), "symbol": s, "rank": i + 1, "score": float(score[s]),
                         "sector": sectors.get(s, "unknown")} for i, s in enumerate(targets))

    def execute(t: int) -> None:
        """Sells, then buys in rank order, at session ``t``'s open; the core's fill limits apply."""
        targets, target_value = plan
        wanted = set(targets)
        for sym in sorted(set(positions) | pending_exits):
            if sym not in positions:
                pending_exits.discard(sym)
                continue
            j, px = col[sym], float(O[t, col[sym]])
            exiting = sym not in wanted
            if not np.isfinite(px) or px <= 0:
                if exiting:
                    pending_exits.add(sym)
                continue
            want = 0 if exiting else int(target_value // px)
            q = positions[sym] - want
            if q <= 0:
                continue
            q_now = cap_quantity(q, px, ADV[t - 1, j], costs.max_participation)
            if not exiting and q_now * px < min_trade:
                continue                                  # a small re-weighting waits
            if q_now > 0:
                fill(sym, t, -q_now, "exit" if exiting else "trim")
            if exiting and sym in positions:
                pending_exits.add(sym)                    # the rest at the next open
            else:
                pending_exits.discard(sym)
        for sym in targets:
            j, px = col[sym], float(O[t, col[sym]])
            if not np.isfinite(px) or px <= 0:
                continue
            q = cap_quantity(int(target_value // px) - positions.get(sym, 0), px, ADV[t - 1, j], costs.max_participation)
            while q > 0 and q * px + charge(q * px, "BUY", t, j) > cash:
                q = int(min(q - 1, cash / (px * 1.01)))
            if q <= 0 or q * px < min_trade:
                continue
            fill(sym, t, q, "top-up" if sym in positions else "entry")

    if s0 > 0 and month_end[s0 - 1]:
        decide(s0 - 1)
    for t in range(s0, s1 + 1):
        if plan is not None:                              # decided at the last close
            execute(t)
            plan = None
        elif pending_exits:
            for sym in sorted(pending_exits):
                if sym in positions and np.isfinite(O[t, col[sym]]) and O[t, col[sym]] > 0:
                    fill(sym, t, -positions[sym], "exit")
            pending_exits.intersection_update(positions)
        held = sum(n * C[t, col[s]] for s, n in positions.items())
        curve[dates[t]] = cash + held
        invested[dates[t]] = held / curve[dates[t]] if curve[dates[t]] > 0 else 0.0
        if month_end[t] and t < s1:
            decide(t)

    eq = pd.Series(curve, name="equity")
    rets = eq.pct_change().fillna(eq.iloc[0] / cfg.initial_capital - 1.0)
    dec = pd.DataFrame(decisions)
    hold = pd.DataFrame(held_log)
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    traded = sum(x["value"] for x in trades)
    m = compute_metrics(rets, eq, rf_annual=BOOK_RF, initial_capital=cfg.initial_capital)
    m.update(trades=len(trades), exits=sum(x["reason"] == "exit" for x in trades),
             annual_turnover=float(traded / 2 / eq.mean() / years),
             cost_drag_annual=float(sum(x["cost"] for x in trades) / eq.mean() / years),
             average_invested=float(np.mean(list(invested.values()))),
             average_holdings=float(dec["holdings"].mean()) if len(dec) else 0.0,
             ttm_share=float((dec["with_ttm"] / dec["universe"]).mean()) if len(dec) else 0.0,
             eligible_average=float(dec["eligible"].mean()) if len(dec) else 0.0,
             sector_mix={k: round(float(v), 4) for k, v in
                         hold["sector"].value_counts(normalize=True).head(10).items()} if len(hold) else {})
    return ValueResult(cfg, rets, eq, trades, dec, hold, m, getattr(data, "data_hash", "") or data.compute_hash(),
                       fundamentals_hash)


def record(result: ValueResult, runs_dir: str = VALUE_RUNS_DIR) -> str:
    from kite_connect.options.backtest import _run_id
    from nse_engine.validation.trials import record_result

    cfg = result.config
    result.run_id = result.run_id or _run_id(cfg.config_hash())
    run_dir = record_result(result, tag=cfg.name, runs_dir=runs_dir, window=(cfg.start, cfg.end),
                            extra={"family": "value", "round": "GT4", "fundamentals_hash": result.fundamentals_hash})
    pd.DataFrame(result.trades).to_csv(Path(run_dir) / "trades.csv", index=False)
    result.decisions.to_csv(Path(run_dir) / "decisions.csv", index=False)
    result.holdings.to_csv(Path(run_dir) / "holdings.csv", index=False)
    return run_dir


# ---------------------------------------------------------------- the gates

def evaluate(runs_dir: str = VALUE_RUNS_DIR, book_runs: str = "data/nse_engine/runs",
             equity_store: str = "data/nse_engine/store") -> Dict[str, Any]:
    """Plan 5z's pass rules (``nse_engine.validation.sleeve_gates``) for each configuration's latest run, at
    the value family's N, beside the live book."""
    from kite_connect.options.structures import _live_book_returns, _nifty_tri_returns
    from nse_engine.validation.sleeve_gates import sleeve_gates
    from nse_engine.validation.trials import TrialRegistry

    reg = TrialRegistry(runs_dir)
    trials = reg.list_trials()
    mat = reg.returns_matrix(window=(ValueConfig.start, ValueConfig.end))
    book, book_run = _live_book_returns(book_runs)
    tri = _nifty_tri_returns(equity_store, book.index)
    out: Dict[str, Any] = {"round": "GT4", "live_book_run": book_run, "family_n": int(mat.shape[1]), "configs": {}}
    for key, cfg in CONFIGS.items():
        rows = trials[trials["tag"] == cfg.name]
        if rows.empty:
            continue
        run_id = str(rows["run_id"].iloc[-1])
        gates = sleeve_gates(reg.load_returns(run_id), mat, book, tri, BOOK_RF, COMBINED_WEIGHT)
        out["book"] = gates.pop("book")
        manifest = json.loads((Path(runs_dir) / run_id / "manifest.json").read_text())
        mm = manifest.get("metrics", {})
        out["configs"][key] = {
            "run_id": run_id, **{k: gates[k] for k in ("dsr", "sleeve", "combined", "rules", "pass")},
            "reported": {"monthly_correlation_with_book": gates["monthly_correlation_with_book"],
                         "ttm_share": mm.get("ttm_share"), "annual_turnover": mm.get("annual_turnover"),
                         "average_holdings": mm.get("average_holdings"), "sector_mix": mm.get("sector_mix"),
                         "years": gates["years"], "fundamentals_hash": manifest.get("fundamentals_hash"),
                         "data_hash": manifest.get("data_hash"),
                         "runtime_origin": manifest.get("runtime", {}).get("origin")},
        }
    passed = [k for k, v in out["configs"].items() if v["pass"]]
    out["verdict"] = (f"PASS: paper-trade {', '.join(passed)} for at least 60 sessions" if passed
                      else "GT4 closed: no configuration passed its gates")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Selective value / quality sleeve (GT4, plan 5z)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--configs", default="GT4-A,GT4-B")
    p.add_argument("--no-record", action="store_true")
    p.add_argument("--start")
    p.add_argument("--end")
    sub.add_parser("evaluate")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cmd == "run":
        data, quarters, fhash = load_inputs(end=args.end or ValueConfig.end)
        for key in args.configs.split(","):
            cfg = replace(CONFIGS[key], start=args.start or ValueConfig.start, end=args.end or ValueConfig.end)
            res = run_value(cfg, data, quarters, fhash)
            where = "not recorded" if args.no_record else record(res)
            print(key, where, json.dumps({k: res.metrics.get(k) for k in (
                "cagr", "sharpe", "max_drawdown", "trades", "annual_turnover", "average_holdings", "ttm_share")},
                default=str))
        return 0
    print(json.dumps(evaluate(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
