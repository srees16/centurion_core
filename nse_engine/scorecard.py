"""
Strategy scorecard (tracker SC1): one report per book answering whether the
Sharpe survives costs, out-of-sample testing and live trading at the size
Centurion means to run.  Centurion trades non-HFT, so robustness and
overfitting weigh most.

Sections and where each number comes from
-----------------------------------------
* Return / risk: ``nse_engine.metrics`` plus the information ratio against
  NIFTY 50 TRI (classic: active return over tracking error) and CVaR 95%.
* Attribution: daily long-short style factors built from the store, point in
  time within the engine's own universe (market = NIFTY 50 TRI, size by
  traded value as the store has no shares outstanding, 12-1 momentum, low
  volatility; value needs fundamentals the store lacks), a Newey-West
  regression, and the alpha left after them.  Alpha decay two ways: the
  combined forecast's rank IC against forward returns by horizon, and the
  annual alpha's trend over the sample.
* Trading: turnover, hit rate, win/loss ratio, profit factor and P&L per
  round trip from the recorded trades; modelled impact by participation
  (the cost model re-applied to each fill).
* Robustness: the anchored walk-forward's stitched OOS returns, DSR and
  PBO from the trial registry (as ``validate``), the registry's one-change
  neighbours of the configuration, and returns split by NIFTY trend / VIX
  regime.
* Capacity: the last two years' fills re-sized to each capital, the cost
  model's impact against the gross edge: the capital at which impact eats
  half of it.
* Correlation: daily returns against the other books, the options sleeves
  and the metal ETFs.
* Live: the paper book's G4 gate (tracking error, fill cost ratio, drawdown)
  when Neon is reachable (``CENTURION_DATABASE_URL``), else pending.

Pass rules are section 1's targets, fixed before the data is read; the rest
is reported.  ``python -m runners.run_nse_engine scorecard --book all``
writes ``docs/scorecards/<as-of>_<book>.md`` and the JSON under
``data/nse_engine/scorecard/``.
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from nse_engine.config import EngineConfig
from nse_engine.costs import cap_quantity, impact_bps, median_traded_value
from nse_engine.metrics import TRADING_DAYS, compute_metrics, drawdown_periods, round_trips
from nse_engine.validation import significance
from nse_engine.validation.diagnostics import alpha_beta, newey_west_cov
from nse_engine.validation.dsr import (daily_rf, deflated_sharpe, min_track_record_length, probabilistic_sharpe,
                                       sharpe_estimator_variance)

logger = logging.getLogger(__name__)

BOOKS = {"deployed": "config/nse_engine_deployed.json", "candidate": "config/nse_engine_candidate.json",
         "e4": "config/nse_engine_e4.json"}
BENCHMARK = "NIFTY50_TRI"
HORIZONS = (5, 21, 63, 126, 252)
LADDER_INR = (6e5, 1.2e6, 2.1e6, 3e6, 5e6, 1e7, 2e7, 5e7, 1e8)
CAPACITY_WINDOW_DAYS = 504
#: The anchored walk-forward of each book's configuration family (R12 re-run on cost model 4 and data
#: hash v2 with point-in-time name ties, trackers IC1 and LN-T15; test years 2017-25): arm A re-fits K5's
#: 32-point grid (the deployed rule and the candidate's neutral 0.6 are grid points), arm B the same grid
#: times exit rank {40, 60} (E4's rule).  The family's base is bd79bf28, not the book's own hash: a
#: walk-forward re-fits the parameters, so it judges the family, not one setting.  Earlier runs stay as
#: wf_oos_returns_r12a.csv / _r12b.csv (cost model 3) and _r12a4 / _r12b4 (cost model 4, data hash v1).
WALK_FORWARD_OOS = {"deployed": "data/nse_engine/wf_oos_returns_r12a5.csv",
                    "candidate": "data/nse_engine/wf_oos_returns_r12a5.csv",
                    "e4": "data/nse_engine/wf_oos_returns_r12b5.csv"}
OPTIONS_RUNS = "data/nse_engine/runs_options"
#: The Sharpe rules' target, also the bar the probabilistic Sharpe is reported against (tracker LN-T24).
SHARPE_TARGET = 1.2
#: Section 1's targets, fixed before any number is read.
PASS_RULES = (("net Sharpe", "sharpe", ">", SHARPE_TARGET), ("max drawdown", "max_drawdown", ">=", -0.30),
              ("Calmar", "calmar", ">=", 1.0), ("deflated Sharpe", "dsr", ">=", 0.95),
              ("walk-forward OOS Sharpe", "oos_sharpe", ">=", SHARPE_TARGET))
#: How many of the deepest drawdowns are listed, the book's own and NIFTY 50 TRI's (tracker LN-T25).
TOP_DRAWDOWNS = 5
#: A NIFTY 50 fall of at least this share from a closing high to its lowest close before the high is regained is
#: a crash episode for ``crash_windows`` (tracker TH1: the "extreme market" measure of Sun's PRUDEX-Compass,
#: thesis sec. 7.3.7, found from the index rather than from hand-picked dates).
CRASH_MIN_FALL = 0.15
#: A fall within this of the threshold counts (125 -> 112.5 is -0.09999999999999998 in floating point).
CRASH_FALL_TOL = 1e-12


def _f(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def _clean(obj: Any) -> Any:
    """JSON-safe: NaN -> None, numpy -> Python, Timestamp -> str."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return str(obj)[:10]
    return obj


# ── loading ────────────────────────────────────────────────────────────────

def latest_run(registry, config_hash: str, cost_model: Optional[int] = None) -> Dict[str, Any]:
    """The newest recorded full-window run of ``config_hash`` (on the current cost model when given).

    Full window is the forward gate's validation window, as for the books register: a run over another
    window (the 2007-25 extended checks) is a different experiment, not a newer one.
    """
    from nse_engine.forward_gate import VALIDATION_WINDOW

    trials = registry.list_trials()
    mine = trials[(trials["config_hash"] == config_hash) & (trials["start"] == VALIDATION_WINDOW[0])
                  & (trials["end"] == VALIDATION_WINDOW[1])]
    if cost_model is not None:
        mine = mine[mine["cost_model"] == int(cost_model)]
    if mine.empty:
        raise FileNotFoundError(f"no recorded run of {config_hash} (cost model {cost_model})")
    row = mine.sort_values("created_at").iloc[-1]
    return row.to_dict()


def read_run(run_dir: Path) -> Dict[str, Any]:
    man = json.loads((run_dir / "manifest.json").read_text())
    ret = pd.read_csv(run_dir / "returns.csv")
    returns = pd.Series(ret.iloc[:, -1].to_numpy(dtype="float64"), index=pd.DatetimeIndex(pd.to_datetime(ret.iloc[:, 0])))
    eq = pd.read_csv(run_dir / "equity.csv")
    equity = pd.Series(eq.iloc[:, -1].to_numpy(dtype="float64"), index=pd.DatetimeIndex(pd.to_datetime(eq.iloc[:, 0])))
    trades = pd.read_csv(run_dir / "trades.csv", parse_dates=["date"])
    weights = pd.read_parquet(run_dir / "weights.parquet")
    cfg = EngineConfig.from_dict(json.loads((run_dir / "config.json").read_text()))
    return {"manifest": man, "returns": returns.sort_index(), "equity": equity.sort_index(), "trades": trades,
            "weights": weights, "config": cfg}


def load_data_for(cfg: EngineConfig, data_start: str):
    """The panel, universe mask and combined forecast the configuration trades on."""
    from nse_engine.signals import compute_signal_panels
    from nse_engine.universe import compute_universe_panel
    from runners.run_nse_engine import _load_data

    store = cfg.data.store_dir
    if not Path(store).exists():                           # a run imported from Kaggle records /kaggle/... paths
        cfg = cfg.replace(**{"data.store_dir": EngineConfig().data.store_dir})
    data = _load_data(cfg, data_start=data_start)
    mask = compute_universe_panel(data, cfg.universe, exclude=cfg.sleeves.symbols).mask
    panels = compute_signal_panels(data.close, mask, cfg.signals, delivery_pct=data.delivery_pct)
    return data, mask, panels.combined


# ── return / risk ──────────────────────────────────────────────────────────

def benchmark_returns(data, window: Tuple[pd.Timestamp, pd.Timestamp]) -> pd.Series:
    ic = data.index_close
    name = BENCHMARK if BENCHMARK in ic.columns else "NIFTY50"
    s = ic[name].astype("float64").ffill().pct_change()
    return s.loc[(s.index >= window[0]) & (s.index <= window[1])].dropna().rename(name)


def universe_average(data, mask: pd.DataFrame) -> pd.Series:
    """The "market average" of Sun's PRUDEX-Compass (thesis sec. 7.2.4, tracker TH1): every member of the book's
    own point-in-time universe in equal weight, members as known at the previous close, rebalanced daily, before
    costs (generous to the benchmark).  Prices as the engine's (adjusted for splits, bonuses and dividends) and
    marked at the last close, as the engine marks equity.  Raises on an infinite day."""
    close = data.close.astype("float64").ffill()
    ret = close.pct_change(fill_method=None)
    members = mask.reindex(index=ret.index, columns=ret.columns, fill_value=False).astype(bool).shift(1, fill_value=False)
    avg = ret.where(members).mean(axis=1, skipna=True).dropna()
    if np.isinf(avg.to_numpy()).any():
        raise ValueError("the universe average holds an infinite return")
    return avg.rename("market_average")


def market_average(returns: pd.Series, average: pd.Series, bench: pd.Series, rf: float) -> Dict[str, Any]:
    """The book's walk-forward record beside the equal-weight universe and NIFTY 50 TRI over their common sessions:
    can the book beat holding everything it could have picked?"""
    al = pd.concat({"book": returns, "market_average": average, "nifty": bench}, axis=1, join="inner").dropna()
    if len(al) < 60:
        return {"error": f"{len(al)} common sessions (60 needed)"}
    out: Dict[str, Any] = {"window": [str(al.index[0].date()), str(al.index[-1].date())], "sessions": int(len(al))}
    for k in al.columns:
        m = compute_metrics(al[k], (1 + al[k]).cumprod(), rf_annual=rf, initial_capital=1.0)
        out[k] = {"cagr": m["cagr"], "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"]}
    by = (1 + al).groupby(al.index.year).prod() - 1
    out["by_year"] = {str(y): {k: float(v) for k, v in row.items()} for y, row in by.iterrows()}
    return out


def _month_end() -> str:
    major, minor = (int(x) for x in pd.__version__.split(".")[:2])
    return "ME" if (major, minor) >= (2, 2) else "M"


def cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    """Expected shortfall: the mean of the worst ``alpha`` share of daily returns (negative)."""
    r = np.sort(returns.dropna().to_numpy(dtype="float64"))
    k = max(1, int(math.ceil(alpha * len(r))))
    return float(r[:k].mean()) if len(r) else float("nan")


def drawdowns(equity: pd.Series, bench: pd.Series) -> Dict[str, Any]:
    """The book's drawdown periods (depth, duration, recovery) and its fall and recovery legs over NIFTY 50
    TRI's deepest drawdowns (tracker LN-T25).  Descriptive: no pass rule, gate or ladder reads them."""
    def period(p) -> Dict[str, Any]:
        return {"peak": str(p.peak.date()), "trough": str(p.trough.date()),
                "recovery": None if pd.isna(p.recovery) else str(p.recovery.date()), "depth": p.depth, "days": p.days}

    e = equity.dropna().astype("float64")
    periods = drawdown_periods(e)
    out: Dict[str, Any] = {"share_of_days_under_water": float((e < e.cummax()).mean()), "n_periods": int(len(periods))}
    if len(periods):
        out["longest"] = period(periods.loc[periods["days"].idxmax()])
        out["deepest"] = [period(p) for p in periods.nsmallest(TOP_DRAWDOWNS, "depth").itertuples()]
        if pd.isna(periods["recovery"].iloc[-1]):
            out["open"] = period(periods.iloc[-1])
    tri = (1 + bench.dropna()).cumprod()
    episodes = []
    for p in drawdown_periods(tri).nsmallest(TOP_DRAWDOWNS, "depth").itertuples():
        end = tri.index[-1] if pd.isna(p.recovery) else p.recovery
        episodes.append({**period(p), "benchmark_recovery_leg": float(tri.asof(end) / tri.asof(p.trough) - 1.0),
                         "book_fall_leg": float(e.asof(p.trough) / e.asof(p.peak) - 1.0),
                         "book_recovery_leg": float(e.asof(end) / e.asof(p.trough) - 1.0)})
    out["benchmark_episodes"] = episodes
    return out


def return_risk(returns: pd.Series, equity: pd.Series, trades: pd.DataFrame, weights: pd.DataFrame,
                bench: pd.Series, rf: float, initial_capital: float) -> Dict[str, Any]:
    out = compute_metrics(returns, equity, trades, weights, rf_annual=rf, initial_capital=initial_capital)
    out["cvar_95_daily"] = cvar(returns, 0.05)
    out["cvar_99_daily"] = cvar(returns, 0.01)
    monthly = (1 + returns).resample(_month_end()).prod() - 1
    out["worst_month"] = float(monthly.min()) if len(monthly) else float("nan")
    out["worst_day"] = float(returns.min())
    aligned = pd.concat({"r": returns, "b": bench}, axis=1, join="inner").dropna()
    active = aligned["r"] - aligned["b"]
    te = float(active.std(ddof=1)) * math.sqrt(TRADING_DAYS)
    out["tracking_error_vs_benchmark"] = te
    out["information_ratio"] = float(active.mean() * TRADING_DAYS / te) if te > 0 else float("nan")
    out["active_return_annual"] = float(active.mean() * TRADING_DAYS)
    ab = alpha_beta(returns, bench, rf_annual=rf)
    out.update({"beta_market": ab["beta"], "alpha_annual_market": ab["alpha_annual"], "alpha_t_market": ab["alpha_t_hac"],
                "r_squared_market": ab["r_squared"], "information_ratio_regression": ab["information_ratio"]})
    down = aligned[aligned["b"] < 0]
    if len(down) > 30:
        out["beta_down_market"] = float(np.polyfit(down["b"], down["r"], 1)[0])
    out["drawdowns"] = drawdowns(equity, bench)
    return out


# ── attribution: style factors from the store ──────────────────────────────

def style_factors(data, mask: pd.DataFrame, window: Tuple[pd.Timestamp, pd.Timestamp], rf: float,
                  refresh_every: int = 21, lookback_vol: int = 252) -> pd.DataFrame:
    """Daily long-short factor returns, rebuilt every ``refresh_every`` sessions from data up to that day,
    equal-weight top vs bottom terciles of the universe's members: size (small minus big by 126-day median
    traded value: a liquidity proxy), momentum (12-1 winners minus losers), low volatility (low minus high
    252-day realised vol).  Market is NIFTY 50 TRI over the risk-free rate."""
    close = data.close.astype("float64")
    rets = close.pct_change()
    size_char = median_traded_value(data.value.astype("float64"), 126)
    mom_char = close.shift(21) / close.shift(252) - 1.0
    vol_char = rets.rolling(lookback_vol, min_periods=126).std()
    dates = close.index
    lo = int(dates.searchsorted(window[0]))
    hi = int(dates.searchsorted(window[1], side="right"))
    legs = {"size": [], "momentum": [], "low_vol": []}
    frames = {k: pd.Series(np.nan, index=dates[lo:hi]) for k in legs}
    m = mask.reindex(index=dates, columns=close.columns).fillna(False).to_numpy()
    chars = {"size": (size_char, False), "momentum": (mom_char, True), "low_vol": (vol_char, False)}
    arrs = {k: v.to_numpy() for k, (v, _) in chars.items()}
    r_np = rets.to_numpy()
    for start in range(lo, hi, refresh_every):
        end = min(start + refresh_every, hi)
        members = m[start]
        for name, (_, high_is_long) in chars.items():
            c = arrs[name][start]
            ok = members & np.isfinite(c)
            if ok.sum() < 30:
                continue
            vals = c[ok]
            q1, q2 = np.quantile(vals, [1 / 3, 2 / 3])
            top = ok & (c >= q2)
            bottom = ok & (c <= q1)
            long_, short = (top, bottom) if high_is_long else (bottom, top)
            stop = min(end + 1, hi)
            seg = r_np[start + 1:stop]
            idx = dates[start + 1:stop]
            lr = np.nanmean(np.where(long_[None, :], seg, np.nan), axis=1)
            sr = np.nanmean(np.where(short[None, :], seg, np.nan), axis=1)
            frames[name].loc[idx] = lr - sr
    out = pd.DataFrame(frames)
    out["market"] = benchmark_returns(data, window).reindex(out.index) - daily_rf(rf)
    return out.dropna(how="all")


def factor_exposures(returns: pd.Series, factors: pd.DataFrame, rf: float) -> Dict[str, Any]:
    """Newey-West OLS of the book's excess returns on the factors; the alpha left after them."""
    df = pd.concat([returns.rename("r") - daily_rf(rf), factors], axis=1, join="inner").dropna()
    n = len(df)
    if n < 100:
        return {"error": f"only {n} aligned days"}
    names = [c for c in factors.columns]
    x = np.column_stack([np.ones(n)] + [df[c].to_numpy() for c in names])
    y = df["r"].to_numpy()
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    lags = int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    se = np.sqrt(np.clip(np.diag(newey_west_cov(x, resid, lags)), 0, None))
    t = np.where(se > 0, coef / np.where(se > 0, se, 1), np.nan)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = float(1 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan")
    mean_ex = float(y.mean() * TRADING_DAYS)
    alpha = float(coef[0] * TRADING_DAYS)
    explained = {c: float(coef[i + 1] * df[c].mean() * TRADING_DAYS) for i, c in enumerate(names)}
    return {"n_days": n, "alpha_annual": alpha, "alpha_t_hac": float(t[0]),
            "alpha_share_of_excess_return": alpha / mean_ex if mean_ex else float("nan"),
            "excess_return_annual": mean_ex, "r_squared": r2,
            "betas": {c: float(coef[i + 1]) for i, c in enumerate(names)},
            "t_stats": {c: float(t[i + 1]) for i, c in enumerate(names)},
            "return_explained_by_factor_annual": explained,
            "value_factor": "not available: the store holds no fundamentals (book value, earnings)"}


def alpha_decay(returns: pd.Series, bench: pd.Series, rf: float, forecast: pd.DataFrame, close: pd.DataFrame,
                mask: pd.DataFrame, window: Tuple[pd.Timestamp, pd.Timestamp], rt: pd.DataFrame) -> Dict[str, Any]:
    """Three views: (1) the combined forecast's rank IC against forward returns at each horizon (how fast
    the signal's information decays), (2) the annual alpha vs NIFTY 50 TRI and its trend over the sample
    (whether the edge is fading), (3) round-trip P&L by holding period."""
    out: Dict[str, Any] = {}
    dates = close.index
    lo, hi = int(dates.searchsorted(window[0])), int(dates.searchsorted(window[1], side="right"))
    f = forecast.reindex(index=dates, columns=close.columns).to_numpy()
    c = close.astype("float64").to_numpy()
    m = mask.reindex(index=dates, columns=close.columns).fillna(False).to_numpy()
    ic: Dict[int, List[float]] = {h: [] for h in HORIZONS}
    for pos in range(lo, hi, 21):
        for h in HORIZONS:
            if pos + h >= len(dates):
                continue
            fwd = c[pos + h] / c[pos] - 1.0
            ok = m[pos] & np.isfinite(f[pos]) & np.isfinite(fwd)
            if ok.sum() >= 30 and np.nanstd(f[pos][ok]) > 0:
                ic[h].append(float(sp_stats.spearmanr(f[pos][ok], fwd[ok])[0]))
    out["signal_rank_ic_by_horizon"] = {str(h): {"mean_ic": float(np.mean(v)), "t_stat": float(np.mean(v) / np.std(v, ddof=1) * math.sqrt(len(v))) if len(v) > 2 else float("nan"), "n_dates": len(v)}
                                        for h, v in ic.items() if v}
    yearly = []
    for year, r in returns.groupby(returns.index.year):
        b = bench.reindex(r.index).dropna()
        if len(b) < 100:
            continue
        ab = alpha_beta(r, b, rf_annual=rf)
        yearly.append({"year": int(year), "alpha_annual": ab["alpha_annual"], "alpha_t": ab["alpha_t_hac"], "beta": ab["beta"]})
    out["alpha_by_year"] = yearly
    if len(yearly) >= 4:
        ys = np.array([y["year"] for y in yearly], dtype=float)
        al = np.array([y["alpha_annual"] for y in yearly])
        fit = sp_stats.linregress(ys, al)
        slope, pvalue = fit.slope, fit.pvalue
        out["alpha_trend"] = {"slope_per_year": float(slope), "p_value": float(pvalue),
                              "first_half_mean": float(al[: len(al) // 2].mean()), "second_half_mean": float(al[len(al) // 2:].mean())}
    if len(rt):
        days = (pd.to_datetime(rt["close_date"]) - pd.to_datetime(rt["open_date"])).dt.days
        buckets = pd.cut(days, [-1, 21, 63, 126, 252, 10_000], labels=["<=21d", "22-63d", "64-126d", "127-252d", ">252d"])
        g = rt.groupby(buckets, observed=True)["pnl_inr"]
        out["round_trip_pnl_by_holding_period"] = {str(k): {"n": int(v.size), "mean_pnl_inr": float(v.mean()),
                                                            "hit_rate": float((v > 0).mean())} for k, v in g}
    return out


# ── trading ────────────────────────────────────────────────────────────────

def trading_stats(trades: pd.DataFrame, equity: pd.Series, adv: pd.DataFrame, cfg: EngineConfig) -> Dict[str, Any]:
    rt = round_trips(trades)
    out: Dict[str, Any] = {"n_round_trips": int(len(rt))}
    if len(rt):
        pnl = rt["pnl_inr"].to_numpy(dtype="float64")
        wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
        eq_at_open = equity.reindex(pd.to_datetime(rt["open_date"]), method="ffill").to_numpy()
        out.update({"hit_rate": float((pnl > 0).mean()),
                    "win_loss_ratio": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) and losses.mean() != 0 else float("nan"),
                    "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else float("inf"),
                    "avg_pnl_per_round_trip_inr": float(pnl.mean()),
                    "avg_pnl_per_round_trip_bp_of_equity": float(np.nanmean(pnl / eq_at_open) * 1e4),
                    "median_holding_days": float((pd.to_datetime(rt["close_date"]) - pd.to_datetime(rt["open_date"])).dt.days.median()),
                    "largest_win_inr": float(wins.max()) if len(wins) else 0.0,
                    "largest_loss_inr": float(losses.min()) if len(losses) else 0.0})
    t = trades.copy()
    t["adv"] = [_adv_at(adv, d, s) for d, s in zip(t["date"], t["symbol"])]
    t["participation"] = t["value_inr"] / t["adv"]
    t["impact_bps"] = [impact_bps(v, a, cfg.costs) for v, a in zip(t["value_inr"], t["adv"])]
    t["impact_inr"] = t["value_inr"] * t["impact_bps"] / 1e4
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    avg_eq = float(equity.mean())
    p = t["participation"].replace([np.inf, -np.inf], np.nan).dropna()
    # A fill short of its request was cut by the participation cap, or else (buys) scaled down for cash.
    reduced = (t["requested_quantity"] > t["quantity"]).to_numpy()
    capped = np.array([cap_quantity(q, px, a, cfg.costs.max_participation) < q
                       for q, px, a in zip(t["requested_quantity"], t["price"], t["adv"])], dtype=bool)
    out["market_impact"] = {
        "participation_median": float(p.median()), "participation_p95": float(p.quantile(0.95)),
        "n_fills": int(len(t)), "fills_cut_by_cap": int((reduced & capped).sum()),
        "fills_scaled_for_cash": int((reduced & ~capped).sum()),
        "impact_bps_value_weighted": float(t["impact_inr"].sum() / t["value_inr"].sum() * 1e4),
        "impact_drag_annual": float(t["impact_inr"].sum() / avg_eq / years),
        "statutory_drag_annual": float((t["cost_inr"].sum() - t["impact_inr"].sum()) / avg_eq / years),
        "fills_by_reason": {str(k): int(v) for k, v in t["reason"].value_counts().items()},
    }
    out["_fills"] = t
    return out


def _adv_at(adv: pd.DataFrame, d: pd.Timestamp, sym: str) -> float:
    if sym not in adv.columns:
        return float("nan")
    pos = int(adv.index.searchsorted(d)) - 1              # the decision's row: the session before the fill
    return float(adv[sym].iloc[pos]) if pos >= 0 else float("nan")


# ── capacity ───────────────────────────────────────────────────────────────

def capacity(fills: pd.DataFrame, equity: pd.Series, cfg: EngineConfig, gross_edge_annual: float,
             rungs: Sequence[float] = LADDER_INR, window_days: int = CAPACITY_WINDOW_DAYS) -> Dict[str, Any]:
    """The last ``window_days`` sessions' fills re-sized to each capital (the same weights of the book):
    modelled impact per year as a share of that capital, the share of fills (and of traded value) the
    participation cap would cut, and the capital at which impact drag reaches half the gross edge.

    Each fill's cap capital (max participation x ADV / weight) is the capital above which the cap cuts it;
    the assets with the lowest bind first (Lean's LowestCapacityAsset, tracker LN-T23)."""
    f = fills[fills["date"] >= equity.index[-min(window_days, len(equity))]].copy()
    f = f[np.isfinite(f["adv"]) & (f["adv"] > 0)]
    if f.empty:
        return {"error": "no fills with liquidity in the window"}
    f["weight"] = f["value_inr"] / equity.reindex(f["date"], method="ffill").to_numpy()
    f["cap_capital"] = cfg.costs.max_participation * f["adv"] / f["weight"]
    years = (f["date"].max() - f["date"].min()).days / 365.25 or 1.0
    w = f["weight"].to_numpy()

    def drag(capital: float) -> Tuple[float, float, float]:
        value = w * capital
        over = value / f["adv"].to_numpy() > cfg.costs.max_participation
        bps = np.array([impact_bps(v, a, cfg.costs) for v, a in zip(value, f["adv"].to_numpy())])
        return float((value * bps / 1e4).sum() / capital / years), float(over.mean()), float(w[over].sum() / w.sum())

    table = []
    for c in rungs:
        d, capped, capped_value = drag(c)
        table.append({"capital_inr": c, "impact_drag_annual": d, "share_of_fills_over_cap": capped,
                      "share_of_value_over_cap": capped_value, "edge_left_after_impact": gross_edge_annual - d})
    target = 0.5 * gross_edge_annual
    lo, hi = math.log10(1e5), math.log10(1e11)
    solved = None
    if drag(10 ** lo)[0] < target < drag(10 ** hi)[0]:
        for _ in range(60):
            mid = (lo + hi) / 2
            if drag(10 ** mid)[0] < target:
                lo = mid
            else:
                hi = mid
        solved = 10 ** ((lo + hi) / 2)
    sleeves = set(cfg.sleeves.symbols)
    binding = f.groupby("symbol")["cap_capital"].min().sort_values()
    core = [s for s in binding.index if s not in sleeves]

    def asset(sym: str) -> Dict[str, Any]:
        return {"symbol": sym, "sleeve": sym in sleeves, "capital_inr": float(binding[sym])}

    order = np.argsort(f["cap_capital"].to_numpy())
    cap_sorted, share = f["cap_capital"].to_numpy()[order], np.cumsum(w[order]) / w.sum()
    at_solved = drag(solved) if solved else (None, None, None)
    return {"window": [str(f["date"].min().date()), str(f["date"].max().date())], "n_fills": int(len(f)),
            "gross_edge_annual": gross_edge_annual, "by_capital": table,
            "capital_where_impact_eats_half_the_edge_inr": solved,
            "share_of_fills_over_cap_at_that_capital": at_solved[1],
            "share_of_value_over_cap_at_that_capital": at_solved[2],
            "first_binding": [asset(s) for s in binding.index[:5]],
            "first_core_binding": asset(core[0]) if core else None,
            "capital_where_value_over_cap": {f"{q:.0%}": float(cap_sorted[np.searchsorted(share, q)])
                                             for q in (0.05, 0.25, 0.5)},
            "note": "the engine would cap fills over the participation limit rather than pay the impact shown; "
                    "the capped share says how much of the book would then go untraded"}


# ── robustness ─────────────────────────────────────────────────────────────

def sharpe_precision(returns: pd.Series, rf: float) -> Dict[str, Any]:
    """How much of the annual Sharpe is noise (tracker LN-T24), reported beside the pass rules and never one:
    its standard error (with the sample's skew and kurtosis, widened for autocorrelation: AC1), the probabilistic
    Sharpe against the rules' target, and the minimum track record for 95% confidence the true Sharpe beats 0."""
    d = deflated_sharpe(returns, n_trials=1, rf_annual=rf)                # the moments, no deflation
    sr, n, skew, kurt, kappa = d["sr_daily"], d["T"], d["skew"], d["kurtosis"], d["autocorr_factor"]
    ann = math.sqrt(TRADING_DAYS)
    min_trl = min_track_record_length(sr, 0.0, skew, kurt, autocorr_factor=kappa)
    return {"sharpe_se": math.sqrt(sharpe_estimator_variance(sr, n, skew, kurt, kappa)) * ann,
            "psr_target": SHARPE_TARGET, "psr": probabilistic_sharpe(sr, SHARPE_TARGET / ann, n, skew, kurt, kappa),
            "min_trl_sessions": min_trl, "min_trl_years": min_trl / TRADING_DAYS, "autocorr_factor": kappa}


def _wf_returns(path: str) -> Optional[pd.Series]:
    """The stitched walk-forward OOS daily returns at ``path``; None when the file is missing."""
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return pd.Series(df.iloc[:, -1].to_numpy(dtype="float64"), index=pd.DatetimeIndex(pd.to_datetime(df.iloc[:, 0])))


def walk_forward_oos(path: str, rf: float) -> Dict[str, Any]:
    p = Path(path)
    r = _wf_returns(path)
    if r is None:
        return {"error": f"{path} missing"}
    eq = (1 + r).cumprod()
    m = compute_metrics(r, eq, rf_annual=rf, initial_capital=1.0)
    out = {k: m[k] for k in ("sharpe", "cagr", "max_drawdown", "calmar", "ann_vol", "sortino")}
    out.update({"start": str(r.index[0].date()), "end": str(r.index[-1].date()), "source": path,
                "sharpe_precision": sharpe_precision(r, rf), "significance": significance.report(r, rf)})
    stitched = p.with_name(p.name.replace("wf_oos_returns", "wf_stitched").replace(".csv", ".json"))
    if stitched.exists():
        doc = json.loads(stitched.read_text())
        s = doc.get("summary", {})
        out.update({k: s.get(k) for k in ("n_folds", "n_grid_points", "mean_is_sharpe", "mean_oos_sharpe",
                                           "sharpe_degradation", "oos_is_ratio", "negative_oos_years")})
        folds = doc.get("folds") or []
        out["base_config_hash"] = folds[0].get("config_hash") if folds else None
        by_year = (1 + r).groupby(r.index.year).prod() - 1
        out["oos_return_by_year"] = {str(k): float(v) for k, v in by_year.items()}
    return out


def overfitting(registry, run_id: str, manifest: Dict[str, Any], rf: float, splits: int = 16,
                cache: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """DSR of the run and PBO of its registry window (the same matrix ``validate`` uses; the PBO is a
    property of the whole set of trials, so it is computed once and shared across the books)."""
    from nse_engine.validation.pbo import cscv_pbo
    from nse_engine.validation.trials import run_cost_model

    window = (manifest.get("start"), manifest.get("end"))
    cost_model = run_cost_model(manifest)
    key = f"{manifest.get('data_hash')}|{window}|{cost_model}|{rf}"
    cache = cache if cache is not None else {}
    if key not in cache:
        matrix = registry.returns_matrix(data_hash=manifest.get("data_hash"), window=window, cost_model=cost_model)
        pbo = None
        if matrix.shape[1] >= 2:
            p = cscv_pbo(matrix, n_splits=splits, rf_annual=rf)               # excess basis, as the DSR
            pbo = {k: p[k] for k in ("pbo", "n_trials", "n_combinations", "degradation_slope", "prob_oos_loss")}
        cache[key] = (matrix, pbo)
    matrix, pbo = cache[key]
    if run_id not in matrix.columns:
        return {"error": f"{run_id} not in the registry's returns matrix"}
    out: Dict[str, Any] = {"n_configurations": int(matrix.shape[1]), "data_hash": manifest.get("data_hash"),
                           "cost_model": cost_model}
    trials = matrix if matrix.shape[1] > 1 else None
    d = deflated_sharpe(matrix[run_id], trials_matrix=trials, rf_annual=rf)
    out["dsr"] = {k: d.get(k) for k in ("dsr", "sr_annual", "sr0_annual", "n_trials_eff", "passed", "dsr_iid",
                                        "autocorr_factor")}
    dc = deflated_sharpe(matrix[run_id], trials_matrix=trials, rf_annual=rf, trial_count="clustered")
    out["dsr_clustered"] = {"dsr": dc.get("dsr"), "n_trials_eff": dc.get("n_trials_eff")}
    out["sharpe_precision"] = sharpe_precision(matrix[run_id], rf)
    out["significance"] = significance.report(matrix[run_id], rf, n_trials=int(matrix.shape[1]),
                                              cost_drag_annual=(manifest.get("metrics") or {}).get("cost_drag"))
    if pbo is not None:
        out["pbo"] = pbo
    return out


def parameter_sensitivity(registry, run: Dict[str, Any], manifest: Dict[str, Any]) -> Dict[str, Any]:
    """The registry's recorded configurations that differ from this one in exactly one setting."""
    from cloud.kaggle_runner import flatten_config
    from nse_engine.validation.trials import run_cost_model

    trials = registry.list_trials()
    same = trials[(trials["data_hash"] == manifest.get("data_hash")) & (trials["start"] == manifest.get("start"))
                  & (trials["end"] == manifest.get("end")) & (trials["cost_model"] == run_cost_model(manifest))]
    same = same.sort_values("created_at").groupby("config_hash").tail(1)
    base = flatten_config(run["config"].to_dict())
    base_sharpe, base_cagr = _f(manifest["metrics"].get("sharpe")), _f(manifest["metrics"].get("cagr"))
    rows = []
    for _, t in same.iterrows():
        if t["config_hash"] == run["config"].config_hash():
            continue
        try:
            other = flatten_config(EngineConfig.from_dict(json.loads((Path(t["run_dir"]) / "config.json").read_text())).to_dict())
        except (OSError, ValueError, TypeError):
            continue
        diff = [k for k in set(base) | set(other) if base.get(k) != other.get(k)]
        if len(diff) != 1:
            continue
        k = diff[0]
        rows.append({"setting": k, "from": base.get(k), "to": other.get(k), "sharpe": _f(t.get("sharpe")),
                     "delta_sharpe": _f(t.get("sharpe")) - base_sharpe, "delta_cagr": _f(t.get("cagr")) - base_cagr,
                     "max_drawdown": _f(t.get("max_drawdown"))})
    rows.sort(key=lambda r: r["delta_sharpe"])
    deltas = [r["delta_sharpe"] for r in rows if np.isfinite(r["delta_sharpe"])]
    return {"n_neighbours": len(rows), "neighbours": rows,
            "delta_sharpe_min": min(deltas) if deltas else None, "delta_sharpe_max": max(deltas) if deltas else None,
            "fragile_settings": sorted({r["setting"] for r in rows if abs(r["delta_sharpe"]) > 0.10}),
            "n_configurations_same_window": int(len(same))}


def regime_stability(returns: pd.Series, index_close: pd.DataFrame, cfg: EngineConfig, rf: float) -> Dict[str, Any]:
    """Returns split by NIFTY 50 trend (bull: above its 200-day mean and up over 63 days; bear: below and
    down; sideways: the rest) and by India VIX (elevated at the regime gate's threshold).

    Each day carries the regime known at the previous close (tracker LN-T18): a same-day label pulled the
    down days that flip the regime into "bear" and "elevated" by their own move, a look-ahead.
    """
    rc = cfg.regime
    nifty = index_close[rc.index_symbol].astype("float64").ffill()
    vix = index_close[rc.vix_symbol].astype("float64").ffill() if rc.vix_symbol in index_close else None
    ma = nifty.rolling(rc.trend_ma_days).mean()
    r63 = nifty / nifty.shift(63) - 1
    trend = pd.Series("sideways", index=nifty.index)
    trend[(nifty > ma) & (r63 > 0)] = "bull"
    trend[(nifty < ma) & (r63 < 0)] = "bear"
    labels = {"trend": trend.shift(1).reindex(returns.index).ffill()}     # one session on the index calendar
    if vix is not None:
        state = pd.Series(np.where(vix > rc.vix_elevated, "elevated", "calm"), index=vix.index)
        labels["vix"] = state.shift(1).reindex(returns.index).ffill()
    rfd = daily_rf(rf)
    out: Dict[str, Any] = {}
    for kind, lab in labels.items():
        out[kind] = {}
        for name, r in returns.groupby(lab):
            if len(r) < 20:
                continue
            ex = r - rfd
            out[kind][str(name)] = {"share_of_days": float(len(r) / len(returns)), "n_days": int(len(r)),
                                    "annual_return": float(r.mean() * TRADING_DAYS),
                                    "sharpe": float(ex.mean() / ex.std(ddof=1) * math.sqrt(TRADING_DAYS)) if ex.std(ddof=1) > 0 else float("nan"),
                                    "hit_rate_days": float((r > 0).mean()), "worst_day": float(r.min())}
    return out


def crash_episodes(index_close: pd.Series, min_fall: float = CRASH_MIN_FALL) -> List[Dict[str, Any]]:
    """Every fall of the index of at least ``min_fall`` from a closing high to the lowest close before that high
    is regained (``open`` when it has not been by the series' end).  Missing closes are left out, never filled;
    raises on a close that is not positive and finite."""
    if not 0.0 < min_fall < 1.0:
        raise ValueError(f"min_fall must be a share between 0 and 1, got {min_fall}")
    s = pd.Series(index_close, dtype="float64").dropna().sort_index()
    v = s.to_numpy()
    if not (np.isfinite(v) & (v > 0)).all():
        raise ValueError("index closes must be positive and finite")
    out: List[Dict[str, Any]] = []
    i, n = 0, len(v)
    while i < n:                                          # i is always the highest close so far
        j = i + 1
        while j < n and v[j] < v[i]:
            j += 1
        if j > i + 1:
            k = i + 1 + int(np.argmin(v[i + 1:j]))
            if v[k] / v[i] - 1.0 <= -min_fall + CRASH_FALL_TOL:
                out.append({"peak": s.index[i].date().isoformat(), "trough": s.index[k].date().isoformat(),
                            "fall": float(v[k] / v[i] - 1.0), "open": bool(j >= n)})
        i = j
    return out


def crash_windows(returns: pd.Series, index_close: pd.Series, min_fall: float = CRASH_MIN_FALL) -> Dict[str, Any]:
    """The book through each NIFTY crash episode its record spans: its return from the index's peak close to the
    trough close and its deepest drawdown on the way, beside the index's fall.  Episodes outside the record are
    counted, not judged.  Raises on an infinite return (RB4)."""
    r = pd.Series(returns, dtype="float64").dropna()
    if np.isinf(r.to_numpy()).any():
        raise ValueError("returns hold an infinite value: no crash table from them")
    r.index = pd.DatetimeIndex(pd.to_datetime(r.index)).normalize()
    r = r.sort_index()
    episodes = crash_episodes(index_close, min_fall)
    judged = []
    for ep in episodes:
        peak, trough = pd.Timestamp(ep["peak"]), pd.Timestamp(ep["trough"])
        if not len(r) or r.index[0] > peak or r.index[-1] < trough:
            continue
        seg = r[(r.index > peak) & (r.index <= trough)].to_numpy()
        eq = np.cumprod(1.0 + seg)
        judged.append({**ep, "sessions": int(len(seg)), "book_return": float(eq[-1] - 1.0) if len(seg) else 0.0,
                       "book_max_drawdown": float((eq / np.maximum.accumulate(np.maximum(eq, 1.0)) - 1.0).min())
                       if len(seg) else 0.0})
    return {"min_fall": float(min_fall), "episodes": judged, "outside_record": len(episodes) - len(judged)}


def correlations(returns: pd.Series, others: Dict[str, pd.Series]) -> Dict[str, Any]:
    out = {}
    for name, s in others.items():
        a = pd.concat({"r": returns, "o": s}, axis=1, join="inner").dropna()
        if len(a) < 60:
            continue
        monthly = (1 + a).resample(_month_end()).prod() - 1
        out[name] = {"daily": float(a["r"].corr(a["o"])), "monthly": float(monthly["r"].corr(monthly["o"])),
                     "n_days": int(len(a)), "window": [str(a.index[0].date()), str(a.index[-1].date())]}
    return out


# ── live / paper ───────────────────────────────────────────────────────────

def paper_section(dep, schema: Optional[str], rf: float) -> Dict[str, Any]:
    """The paper book's G4 gate (tracking error, fill-cost ratio = implementation shortfall vs the model,
    drawdown vs backtest, regime breaks) and its own metrics, when Neon is reachable.  The reference is
    the same-period backtest ``paper-gate`` uses: the book's configuration from the anchor to today."""
    if not (os.getenv("CENTURION_DATABASE_URL") or os.getenv("DATABASE_URL")):
        return {"status": "pending", "reason": "CENTURION_DATABASE_URL is not set here: the paper record lives in Neon; "
                                                "run with it set (or on GitHub Actions) once paper trading completes"}
    from database.connection import get_db_manager
    from database.paper_cloud import PaperCloudSync
    from nse_engine import paper_gate
    from nse_engine.engine import run_backtest
    from runners.run_nse_engine import _load_data

    cloud = PaperCloudSync(get_db_manager(), schema=schema)
    snaps = cloud.read_snapshots()
    if snaps is None or snaps.empty:
        return {"status": "pending", "reason": f"no paper snapshots in schema {schema or 'public'}"}
    equity = pd.Series(snaps["equity"].astype(float).to_numpy(), index=pd.to_datetime(snaps["date"].astype(str)))
    cfg = dep.reference_config().replace(end=datetime.now(timezone.utc).date().isoformat())
    res = run_backtest(_load_data(cfg, data_start=dep.data_start().isoformat()), cfg, record=False,
                       tag="scorecard-reference")
    gate = paper_gate.evaluate(equity, res.returns, fills=cloud.read_fills(), reference_trades=res.trades,
                               sessions=cloud.read_sessions())
    returns = equity.pct_change().dropna()
    out = {"status": "ok", "sessions": int(len(equity)), "g4": gate}
    if len(returns) > 5:
        out["metrics"] = compute_metrics(returns, equity, rf_annual=rf)
        out["sharpe_se"] = sharpe_precision(returns, rf)["sharpe_se"]
    return out


# ── assembly ───────────────────────────────────────────────────────────────

def pass_rules(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    values = {"sharpe": report["return_risk"].get("sharpe"), "max_drawdown": report["return_risk"].get("max_drawdown"),
              "calmar": report["return_risk"].get("calmar"),
              "dsr": (report.get("overfitting", {}).get("dsr") or {}).get("dsr"),
              "oos_sharpe": report.get("walk_forward_oos", {}).get("sharpe")}
    rows = []
    for label, key, op, target in PASS_RULES:
        v = _f(values.get(key))
        ok = None if not np.isfinite(v) else (v > target if op == ">" else v >= target)
        rows.append({"rule": label, "target": f"{op} {target}", "value": v, "pass": ok})
    return rows


def build(book: str, run_id: Optional[str] = None, paper_schema: Optional[str] = None,
          shared: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The scorecard of one book from its latest recorded run (or ``run_id``)."""
    from nse_engine.deployment import load_deployment
    from nse_engine.validation.trials import TrialRegistry

    dep = load_deployment(BOOKS[book])
    registry = TrialRegistry(EngineConfig().runs_dir)
    cfg_hash = dep.engine.config_hash()
    if run_id is None:
        trial = latest_run(registry, cfg_hash, cost_model=_current_cost_model())
        run_id = str(trial["run_id"])
    run = read_run(Path(EngineConfig().runs_dir) / run_id)
    man, returns, equity, trades, weights, cfg = (run[k] for k in ("manifest", "returns", "equity", "trades", "weights", "config"))
    rf = float(cfg.risk_free_annual)
    window = (returns.index[0], returns.index[-1])
    shared = shared if shared is not None else {}
    if "data" not in shared:
        shared["data"], shared["mask"], shared["forecast"] = load_data_for(cfg, dep.data_start().isoformat())
        shared["signals"] = cfg.signals
    data, mask = shared["data"], shared["mask"]
    if cfg.signals == shared["signals"]:
        forecast = shared["forecast"]
    else:                                                  # another signal set: its own forecast on the shared panel
        from nse_engine.signals import compute_signal_panels
        forecast = compute_signal_panels(data.close, mask, cfg.signals, delivery_pct=data.delivery_pct).combined
    bench = benchmark_returns(data, window)
    adv = median_traded_value(data.value.astype("float64"), cfg.costs.adv_lookback_days)

    report: Dict[str, Any] = {
        "book": book, "status": dep.status, "config_hash": cfg_hash, "run_id": run_id, "data_hash": man.get("data_hash"),
        "cost_model": man.get("cost_model"), "window": [str(window[0].date()), str(window[1].date())],
        "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"), "risk_free_annual": rf,
        "note": ("The recorded run is the registry's configuration; the paper and live books add the deployment's "
                 "drawdown overlay (E2)." if dep.reference_config().drawdown != dep.engine.drawdown else ""),
    }
    report["return_risk"] = return_risk(returns, equity, trades, weights, bench, rf, cfg.initial_capital)
    if "factors" not in shared:
        shared["factors"] = style_factors(data, mask, window, rf)
    report["factor_exposures"] = factor_exposures(returns, shared["factors"], rf)
    rt = round_trips(trades)
    report["alpha_decay"] = alpha_decay(returns, bench, rf, forecast, data.close, mask, window, rt)
    tr = trading_stats(trades, equity, adv, cfg)
    fills = tr.pop("_fills")
    report["trading"] = tr
    gross_edge = (report["return_risk"]["cagr"] - rf) + tr["market_impact"]["impact_drag_annual"]
    report["capacity"] = capacity(fills, equity, cfg, gross_edge)
    report["walk_forward_oos"] = walk_forward_oos(WALK_FORWARD_OOS[book], rf)
    report["overfitting"] = overfitting(registry, run_id, man, rf, cache=shared.setdefault("overfitting", {}))
    report["parameter_sensitivity"] = parameter_sensitivity(registry, run, man)
    report["regime_stability"] = regime_stability(returns, data.index_close, cfg, rf)
    wf = _wf_returns(WALK_FORWARD_OOS[book])            # live-like: the walk-forward record, not the in-sample run
    report["crash_windows"] = (crash_windows(wf, data.index_close[cfg.regime.index_symbol]) if wf is not None
                               else {"error": f"{WALK_FORWARD_OOS[book]} missing"})
    report["market_average"] = (market_average(wf, universe_average(data, mask), bench, rf) if wf is not None
                                else {"error": f"{WALK_FORWARD_OOS[book]} missing"})
    report["correlations"] = correlations(returns, _other_series(registry, book, man, data))
    report["paper"] = paper_section(dep, paper_schema, rf)
    report["pass_rules"] = pass_rules(report)
    return report


def _current_cost_model() -> int:
    from nse_engine.costs import COST_MODEL_VERSION
    return int(COST_MODEL_VERSION)


def _other_series(registry, book: str, manifest: Dict[str, Any], data) -> Dict[str, pd.Series]:
    """The other books' latest runs on the same data, the options sleeves, the metal ETFs and the benchmark."""
    from nse_engine.deployment import load_deployment

    others: Dict[str, pd.Series] = {}
    for name, path in BOOKS.items():
        if name == book:
            continue
        try:
            trial = latest_run(registry, load_deployment(path).engine.config_hash(), cost_model=manifest.get("cost_model"))
            others[f"book:{name}"] = registry.load_returns(str(trial["run_id"]))
        except (FileNotFoundError, OSError, ValueError) as exc:
            logger.warning("no run for book %s: %s", name, exc)
    opt = Path(OPTIONS_RUNS)
    if opt.exists():
        for d in sorted(p for p in opt.iterdir() if p.is_dir()):
            try:
                m = json.loads((d / "manifest.json").read_text())
                if m.get("start") != manifest.get("start"):
                    continue
                r = pd.read_csv(d / "returns.csv")
                others[f"options:{m.get('tag')}"] = pd.Series(r.iloc[:, -1].to_numpy(dtype="float64"),
                                                             index=pd.DatetimeIndex(pd.to_datetime(r.iloc[:, 0])))
            except (OSError, ValueError):
                continue
    for etf in ("GOLDBEES", "SILVERBEES"):
        if etf in data.close.columns:
            others[f"etf:{etf}"] = data.close[etf].astype("float64").pct_change().dropna()
    ic = data.index_close
    if BENCHMARK in ic.columns:
        others[f"index:{BENCHMARK}"] = ic[BENCHMARK].astype("float64").ffill().pct_change().dropna()
    return others


# ── rendering ──────────────────────────────────────────────────────────────

def _pct(x: Any, d: int = 1) -> str:
    v = _f(x)
    return "n/a" if not np.isfinite(v) else f"{v:.{d}%}"


def _num(x: Any, d: int = 2) -> str:
    v = _f(x)
    return "n/a" if not np.isfinite(v) else f"{v:.{d}f}"


def _inr(x: Any) -> str:
    v = _f(x)
    if not np.isfinite(v):
        return "n/a"
    if abs(v) >= 1e7:
        return f"Rs {v / 1e7:.2f} cr"
    return f"Rs {v / 1e5:.1f} L" if abs(v) >= 1e5 else f"Rs {v:,.0f}"


def _precision(sp: Dict[str, Any]) -> str:
    return (f"Sharpe standard error {_num(sp.get('sharpe_se'))}, PSR({_num(sp.get('psr_target'), 1)}) {_num(sp.get('psr'), 3)}, "
            f"MinTRL vs 0 {_num(sp.get('min_trl_sessions'), 0)} sessions ({_num(sp.get('min_trl_years'), 1)} years); "
            f"reported, not a pass rule")


def render_markdown(r: Dict[str, Any]) -> str:
    rr, tr, mi = r["return_risk"], r["trading"], r["trading"]["market_impact"]
    fx, ad, cap, wf, ov, ps, rg = (r.get(k, {}) for k in ("factor_exposures", "alpha_decay", "capacity", "walk_forward_oos",
                                                           "overfitting", "parameter_sensitivity", "regime_stability"))
    L: List[str] = []
    L.append(f"# Scorecard: {r['book']} ({r['config_hash'][:8]}), {r['window'][0]} to {r['window'][1]}")
    L.append("")
    L.append(f"Run {r['run_id']}, data {r['data_hash']}, cost model {r['cost_model']}, risk-free {_pct(r['risk_free_annual'])}, "
             f"as of {r['as_of'][:16]} UTC. {r.get('note', '')}".rstrip())
    L.append("")
    L.append("## Pass rules (fixed before the data was read)")
    L.append("")
    L.append("| Rule | Target | Value | Verdict |")
    L.append("|---|---|---|---|")
    for p in r["pass_rules"]:
        v = p["value"]
        shown = _pct(v) if p["rule"] == "max drawdown" else _num(v)
        L.append(f"| {p['rule']} | {p['target']} | {shown} | {'PASS' if p['pass'] else 'FAIL' if p['pass'] is False else 'n/a'} |")
    L.append("")
    L.append("## Return and risk")
    L.append("")
    L.append("| Metric | Value |")
    L.append("|---|---|")
    for label, key, fn in (("CAGR", "cagr", _pct), ("Sharpe (excess)", "sharpe", _num), ("Sortino", "sortino", _num),
                           ("Information ratio vs NIFTY 50 TRI", "information_ratio", _num),
                           ("Active return vs NIFTY 50 TRI", "active_return_annual", _pct),
                           ("Tracking error vs NIFTY 50 TRI", "tracking_error_vs_benchmark", _pct),
                           ("Calmar", "calmar", _num), ("Max drawdown", "max_drawdown", _pct), ("Volatility", "ann_vol", _pct),
                           ("CVaR 95% (daily)", "cvar_95_daily", lambda x: _pct(x, 2)), ("CVaR 99% (daily)", "cvar_99_daily", lambda x: _pct(x, 2)),
                           ("Worst day", "worst_day", _pct), ("Worst month", "worst_month", _pct),
                           ("Skew", "skew", _num), ("Kurtosis (excess)", "kurtosis", _num),
                           ("Beta to NIFTY 50 TRI", "beta_market", _num), ("Beta on NIFTY down days", "beta_down_market", _num),
                           ("Market alpha (annual, t)", None, None)):
        if key is None:
            L.append(f"| {label} | {_pct(rr.get('alpha_annual_market'))} (t {_num(rr.get('alpha_t_market'), 1)}) |")
        else:
            L.append(f"| {label} | {fn(rr.get(key))} |")
    dd = rr.get("drawdowns") or {}
    if dd.get("longest"):
        lg, op = dd["longest"], dd.get("open")
        L.append("")
        L.append(f"**Drawdowns** ({dd['n_periods']} periods): the longest under water ran {lg['peak']} to "
                 f"{lg['recovery'] or 'the end, not recovered'}, {lg['days']} days, {_pct(lg['depth'])} deep; "
                 f"{_pct(dd['share_of_days_under_water'])} of days under water; "
                 + (f"the open drawdown is {op['days']} days old (peak {op['peak']}, {_pct(op['depth'])} at its deepest)."
                    if op else "none open at the end."))
        L.append("")
        L.append("| Peak | Trough | Recovered | Depth | Days |")
        L.append("|---|---|---|---|---|")
        for d in dd["deepest"]:
            L.append(f"| {d['peak']} | {d['trough']} | {d['recovery'] or 'open'} | {_pct(d['depth'])} | {d['days']} |")
    if dd.get("benchmark_episodes"):
        L.append("")
        L.append("**Across NIFTY 50 TRI's deepest drawdowns** (descriptive: the fall is peak to trough, the recovery "
                 "trough to the TRI's recovery or the end)")
        L.append("")
        L.append("| TRI peak | Trough | Recovered | TRI fall | Book fall | TRI recovery | Book recovery |")
        L.append("|---|---|---|---|---|---|---|")
        for d in dd["benchmark_episodes"]:
            L.append(f"| {d['peak']} | {d['trough']} | {d['recovery'] or 'open'} | {_pct(d['depth'])} | {_pct(d['book_fall_leg'])} | "
                     f"{_pct(d['benchmark_recovery_leg'])} | {_pct(d['book_recovery_leg'])} |")
    L.append("")
    L.append("## Attribution: style factors and the alpha left")
    L.append("")
    if "error" in fx:
        L.append(f"Not computed: {fx['error']}")
    else:
        L.append("| Factor | Beta | t (HAC) | Return explained per year |")
        L.append("|---|---|---|---|")
        for c, b in fx["betas"].items():
            L.append(f"| {c} | {_num(b)} | {_num(fx['t_stats'][c], 1)} | {_pct(fx['return_explained_by_factor_annual'][c])} |")
        L.append(f"| **alpha** | {_pct(fx['alpha_annual'])} per year | {_num(fx['alpha_t_hac'], 1)} | "
                 f"{_pct(fx['alpha_share_of_excess_return'], 0)} of the excess return |")
        L.append("")
        L.append(f"R² {_num(fx['r_squared'])} over {fx['n_days']} days. Factors are built from the store, point in time: "
                 f"size is a liquidity proxy (traded value), value is {fx['value_factor']}.")
    L.append("")
    L.append("## Alpha decay")
    L.append("")
    ic = ad.get("signal_rank_ic_by_horizon", {})
    if ic:
        L.append("Rank IC of the combined forecast against forward returns (universe members, every 21 sessions):")
        L.append("")
        L.append("| Horizon (sessions) | " + " | ".join(ic) + " |")
        L.append("|---|" + "---|" * len(ic))
        L.append("| mean IC | " + " | ".join(_num(v["mean_ic"], 3) for v in ic.values()) + " |")
        L.append("| t-stat | " + " | ".join(_num(v["t_stat"], 1) for v in ic.values()) + " |")
        L.append("")
    if ad.get("alpha_by_year"):
        L.append("| Year | Alpha vs NIFTY 50 TRI | t | Beta |")
        L.append("|---|---|---|---|")
        for y in ad["alpha_by_year"]:
            L.append(f"| {y['year']} | {_pct(y['alpha_annual'])} | {_num(y['alpha_t'], 1)} | {_num(y['beta'])} |")
        t = ad.get("alpha_trend")
        if t:
            L.append("")
            L.append(f"Alpha trend {_pct(t['slope_per_year'])} per year (p {_num(t['p_value'])}); first half mean "
                     f"{_pct(t['first_half_mean'])}, second half {_pct(t['second_half_mean'])}.")
    hp = ad.get("round_trip_pnl_by_holding_period")
    if hp:
        L.append("")
        L.append("| Holding period | Round trips | Mean P&L | Hit rate |")
        L.append("|---|---|---|---|")
        for k, v in hp.items():
            L.append(f"| {k} | {v['n']} | {_inr(v['mean_pnl_inr'])} | {_pct(v['hit_rate'], 0)} |")
    L.append("")
    L.append("## Trading")
    L.append("")
    L.append("| Metric | Value |")
    L.append("|---|---|")
    L.append(f"| Turnover (one-way, per year) | {_num(rr.get('annual_turnover'), 1)}x |")
    L.append(f"| Cost drag (modelled, per year) | {_pct(rr.get('cost_drag'))} = impact {_pct(mi['impact_drag_annual'])} + statutory {_pct(mi['statutory_drag_annual'])} |")
    L.append(f"| Round trips | {tr.get('n_round_trips')} (median holding {_num(tr.get('median_holding_days'), 0)} days) |")
    L.append(f"| Hit rate | {_pct(tr.get('hit_rate'))} |")
    L.append(f"| Win/loss ratio (avg win / avg loss) | {_num(tr.get('win_loss_ratio'))} |")
    L.append(f"| Profit factor | {_num(tr.get('profit_factor'))} |")
    L.append(f"| Average P&L per round trip | {_inr(tr.get('avg_pnl_per_round_trip_inr'))} ({_num(tr.get('avg_pnl_per_round_trip_bp_of_equity'), 0)} bp of equity) |")
    L.append(f"| Largest win / loss | {_inr(tr.get('largest_win_inr'))} / {_inr(tr.get('largest_loss_inr'))} |")
    L.append(f"| Participation (order / median traded value) | median {_pct(mi['participation_median'], 2)}, p95 {_pct(mi['participation_p95'], 2)}; "
             f"{mi['fills_cut_by_cap']} of {mi['n_fills']:,} fills cut by the 5% cap, {mi['fills_scaled_for_cash']} buys scaled down for cash |")
    L.append(f"| Modelled impact | {_num(mi['impact_bps_value_weighted'], 1)} bp of traded value |")
    L.append("")
    L.append("## Capacity")
    L.append("")
    if "error" in cap:
        L.append(f"Not computed: {cap['error']}")
    else:
        L.append(f"Fills of {cap['window'][0]} to {cap['window'][1]} ({cap['n_fills']}) re-sized to each capital; "
                 f"gross edge {_pct(cap['gross_edge_annual'])} per year (net excess CAGR plus modelled impact).")
        L.append("")
        L.append("| Capital | Impact drag per year | Edge left | Fills over the 5% cap | Traded value over it |")
        L.append("|---|---|---|---|---|")
        for row in cap["by_capital"]:
            L.append(f"| {_inr(row['capital_inr'])} | {_pct(row['impact_drag_annual'], 2)} | {_pct(row['edge_left_after_impact'])} | "
                     f"{_pct(row['share_of_fills_over_cap'])} | {_pct(row['share_of_value_over_cap'])} |")
        L.append("")
        L.append(f"**Impact eats half the edge at {_inr(cap['capital_where_impact_eats_half_the_edge_inr'])}**, where "
                 f"{_pct(cap.get('share_of_fills_over_cap_at_that_capital'))} of fills ({_pct(cap.get('share_of_value_over_cap_at_that_capital'))} "
                 f"of traded value) would exceed the participation cap. {cap['note'].capitalize()}.")
        first, core, q = cap.get("first_binding") or [], cap.get("first_core_binding"), cap.get("capital_where_value_over_cap") or {}
        if first:
            L.append("")
            L.append("The cap binds first on " + ", ".join(f"{a['symbol']}{' (sleeve)' if a['sleeve'] else ''} at {_inr(a['capital_inr'])}"
                                                           for a in first)
                     + (f"; the first core stock is {core['symbol']} at {_inr(core['capital_inr'])}" if core else "")
                     + (f"; {', '.join(q)} of traded value is over it at {', '.join(_inr(v) for v in q.values())}" if q else "")
                     + ".")
    L.append("")
    L.append("## Robustness")
    L.append("")
    if "error" not in wf:
        L.append(f"**Walk-forward OOS** ({wf['start']} to {wf['end']}, {wf.get('n_folds')} folds over {wf.get('n_grid_points')} grid points "
                 f"of the configuration family, base {str(wf.get('base_config_hash') or '')[:8]}; {wf['source']}): Sharpe {_num(wf['sharpe'])}, CAGR {_pct(wf['cagr'])}, MaxDD {_pct(wf['max_drawdown'])}, "
                 f"Calmar {_num(wf['calmar'])}; mean IS {_num(wf.get('mean_is_sharpe'))} vs OOS {_num(wf.get('mean_oos_sharpe'))} per fold "
                 f"(OOS/IS {_num(wf.get('oos_is_ratio'))}), {wf.get('negative_oos_years')} negative OOS years; "
                 f"{_precision(wf.get('sharpe_precision') or {})}.")
        if wf.get("significance"):
            L.append("")
            L.append(f"**Hypothesis test, walk-forward OOS** (HT1): {significance.one_line(wf['significance'])}.")
    if "error" not in ov:
        d, pb = ov.get("dsr", {}), ov.get("pbo", {})
        L.append("")
        L.append(f"**Overfitting** over {ov['n_configurations']} recorded configurations (data {ov['data_hash']}, cost model {ov['cost_model']}): "
                 f"deflated Sharpe {_num(d.get('dsr'), 3)} (widened for autocorrelation, kappa {_num(d.get('autocorr_factor'))}; "
                 f"i.i.d. {_num(d.get('dsr_iid'), 3)}; annual Sharpe {_num(d.get('sr_annual'))} vs the expected best of "
                 f"{_num(d.get('sr0_annual'))} from {_num(d.get('n_trials_eff'), 0)} trials; clustered {_num(ov.get('dsr_clustered', {}).get('dsr'), 3)}), "
                 f"PBO {_pct(pb.get('pbo'))}, probability of an OOS loss {_pct(pb.get('prob_oos_loss'))}; "
                 f"{_precision(ov.get('sharpe_precision') or {})}.")
        if ov.get("significance"):
            L.append("")
            L.append(f"**Hypothesis test, backtest** (HT1): {significance.one_line(ov['significance'])}. "
                     "A p-value is not the chance the strategy works; it assumes the past repeats.")
    L.append("")
    L.append(f"**Parameter sensitivity**: {ps.get('n_neighbours')} recorded one-setting neighbours "
             f"(Sharpe change {_num(ps.get('delta_sharpe_min'))} to {_num(ps.get('delta_sharpe_max'))}); "
             f"fragile settings (|change| > 0.10): {', '.join(ps.get('fragile_settings') or []) or 'none'}.")
    if ps.get("neighbours"):
        L.append("")
        L.append("| Setting | From | To | Sharpe | Change | CAGR change |")
        L.append("|---|---|---|---|---|---|")
        for n in ps["neighbours"][:25]:
            L.append(f"| {n['setting']} | {n['from']} | {n['to']} | {_num(n['sharpe'])} | {_num(n['delta_sharpe'], 2)} | {_pct(n['delta_cagr'])} |")
    for kind, title in (("trend", "NIFTY 50 trend"), ("vix", "India VIX")):
        if rg.get(kind):
            L.append("")
            L.append(f"**Regime stability by {title}** (the regime known at the previous close)")
            L.append("")
            L.append("| Regime | Days | Annual return | Sharpe | Up days | Worst day |")
            L.append("|---|---|---|---|---|---|")
            for name, v in rg[kind].items():
                L.append(f"| {name} | {_pct(v['share_of_days'], 0)} | {_pct(v['annual_return'])} | {_num(v['sharpe'])} | {_pct(v['hit_rate_days'], 0)} | {_pct(v['worst_day'])} |")
    ma = r.get("market_average") or {}
    if ma.get("by_year"):
        L.append("")
        L.append(f"**Against the market average** (every member of the book's universe in equal weight, before costs; "
                 f"the walk-forward record, {ma['window'][0]} to {ma['window'][1]})")
        L.append("")
        L.append("| | CAGR | Sharpe | Max drawdown |")
        L.append("|---|---|---|---|")
        for k, label in (("book", "Book"), ("market_average", "Market average"), ("nifty", "NIFTY 50 TRI")):
            L.append(f"| {label} | {_pct(ma[k]['cagr'])} | {_num(ma[k]['sharpe'])} | {_pct(ma[k]['max_drawdown'])} |")
        L.append("")
        L.append("| Year | Book | Market average | NIFTY 50 TRI |")
        L.append("|---|---|---|---|")
        for y, v in ma["by_year"].items():
            L.append(f"| {y} | {_pct(v['book'])} | {_pct(v['market_average'])} | {_pct(v['nifty'])} |")
    cw = r.get("crash_windows") or {}
    if cw.get("episodes"):
        L.append("")
        L.append(f"**Through NIFTY 50's crashes** (falls of {_pct(cw['min_fall'], 0)} or more, peak close to trough "
                 "close; the walk-forward record)")
        L.append("")
        L.append("| Peak | Trough | NIFTY 50 | Book | Book's deepest drawdown |")
        L.append("|---|---|---|---|---|")
        for e in cw["episodes"]:
            L.append(f"| {e['peak']} | {e['trough']}{' (open)' if e['open'] else ''} | {_pct(e['fall'])} | "
                     f"{_pct(e['book_return'])} | {_pct(e['book_max_drawdown'])} |")
    L.append("")
    L.append("## Correlation of daily returns")
    L.append("")
    L.append("| Series | Daily | Monthly | Days |")
    L.append("|---|---|---|---|")
    for name, v in r.get("correlations", {}).items():
        L.append(f"| {name} | {_num(v['daily'])} | {_num(v['monthly'])} | {v['n_days']} |")
    L.append("")
    L.append("## Live: implementation shortfall, slippage, tracking error (paper book)")
    L.append("")
    p = r.get("paper", {})
    if p.get("status") != "ok":
        L.append(f"Pending: {p.get('reason')}. Real fills to date (tracker, 8 Oct 2026): 2, at -8 bp and +10 bp against the model.")
    else:
        g = p["g4"]
        L.append(f"{p['sessions']} paper sessions; G4 verdict **{g.get('verdict')}**.")
        L.append("")
        L.append("| Check | Value | Status | Limits |")
        L.append("|---|---|---|---|")
        for c in g.get("checks", []):
            L.append(f"| {c.get('name')} | {c.get('display')} | {c.get('status')} | {c.get('limits')} |")
        m = p.get("metrics", {})
        if m:
            L.append("")
            L.append(f"Paper book: Sharpe {_num(m.get('sharpe'))} (standard error {_num(p.get('sharpe_se'))}), CAGR {_pct(m.get('cagr'))}, MaxDD {_pct(m.get('max_drawdown'))}, "
                     f"turnover {_num(m.get('annual_turnover'), 1)}x, hit rate {_pct(m.get('hit_rate'))}.")
    L.append("")
    return "\n".join(L)


def write(report: Dict[str, Any], md_dir: str = "docs/scorecards", json_dir: str = "data/nse_engine/scorecard") -> Tuple[Path, Path]:
    stamp = report["as_of"][:10]
    md = Path(md_dir) / f"{stamp}_{report['book']}.md"
    js = Path(json_dir) / f"{stamp}_{report['book']}.json"
    md.parent.mkdir(parents=True, exist_ok=True)
    js.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(render_markdown(report), encoding="utf-8")
    js.write_text(json.dumps(_clean(report), indent=1), encoding="utf-8")
    return md, js


# ── README section (the deployed book's current performance) ─────────────────

README = "docs/README.md"
README_START, README_END = "<!-- scorecard:deployed -->", "<!-- /scorecard:deployed -->"
EN_DASH = "\u2013"
_FRIENDLY = {"portfolio.rebalance_every_n_days": "rebalance frequency", "portfolio.stop_atr_multiple": "stop distance",
             "signals.group_weights": "signal weights"}
#: Headline metrics compared with the previous scorecard: (label, path, higher is better, as a percentage)
_HEADLINE = (("CAGR", ("return_risk", "cagr"), True, True), ("Sharpe", ("return_risk", "sharpe"), True, False),
             ("Sortino", ("return_risk", "sortino"), True, False), ("Calmar", ("return_risk", "calmar"), True, False),
             ("max drawdown", ("return_risk", "max_drawdown"), True, True),
             ("out-of-sample Sharpe", ("walk_forward_oos", "sharpe"), True, False),
             ("deflated Sharpe", ("overfitting", "dsr", "dsr"), True, False))


def _get(r: Dict[str, Any], path: Sequence[str]) -> float:
    for k in path:
        r = (r or {}).get(k) if isinstance(r, dict) else None
    return _f(r)


def _signed(x: float, pct: bool = True, digits: int = 1) -> str:
    text = f"{x * 100:+.{digits}f}%" if pct else f"{x:+.{digits}f}"
    return text.replace("-", "\u2212")


def _changes(r: Dict[str, Any], previous: Optional[Dict[str, Any]]) -> Optional[str]:
    """What moved since the previous scorecard of the same book, better or worse."""
    if not previous:
        return None
    moved = []
    for label, path, higher, pct in _HEADLINE:
        a, b = _get(previous, path), _get(r, path)
        if not (np.isfinite(a) and np.isfinite(b)) or abs(b - a) < (5e-4 if pct else 5e-3):
            continue
        fmt = (lambda x: f"{x * 100:.1f}%".replace("-", "\u2212")) if pct else (lambda x: f"{x:.2f}")
        moved.append(f"{label} {fmt(a)} \u2192 {fmt(b)} ({'better' if (b > a) == higher else 'worse'})")
    since = str(previous.get("as_of", ""))[:10]
    basis = [f"{label} {previous.get(k)} \u2192 {r.get(k)}" for label, k in (("configuration", "config_hash"),
             ("cost model", "cost_model"), ("data", "data_hash")) if previous.get(k) != r.get(k)]
    why = (f" The basis changed ({'; '.join(basis)}), so this is mostly a re-measurement, not the strategy moving."
           if basis and not any("configuration" in x for x in basis) else
           f" A new configuration ({'; '.join(basis)})." if basis else "")
    return (f"**Since the {since} scorecard:** " + ("; ".join(moved) + "." if moved else "no headline metric moved.")
            + why)


def readme_section(r: Dict[str, Any], previous: Optional[Dict[str, Any]] = None) -> str:
    """The README's "Current performance" section, from the deployed book's scorecard (and the previous one)."""
    rr, fx, tr = r["return_risk"], r.get("factor_exposures", {}), r["trading"]
    mi, ad, cap, wf = tr.get("market_impact", {}), r.get("alpha_decay", {}), r.get("capacity", {}), r.get("walk_forward_oos", {})
    ov, ps, rg, co, pa = (r.get(k, {}) for k in ("overfitting", "parameter_sensitivity", "regime_stability", "correlations", "paper"))
    pct = lambda x, d=1: f"{_f(x) * 100:.{d}f}%".replace("-", "\u2212")
    num = lambda x, d=2: f"{_f(x):.{d}f}".replace("-", "\u2212")
    betas = fx.get("betas", {})
    market = sorted(b for b in (_f(betas.get("market")), _f(rr.get("beta_market"))) if np.isfinite(b))
    trend, vix = rg.get("trend", {}), rg.get("vix", {})
    ic = (ad.get("signal_rank_ic_by_horizon") or {})
    at = ad.get("alpha_trend") or {}
    decay = ("none" if _f(at.get("p_value")) > 0.1 else "declining" if _f(at.get("slope_per_year")) < 0 else "rising")
    if pa.get("status") == "ok":
        checks = "; ".join(f"{c.get('name')} {c.get('display')} ({c.get('status')})" for c in (pa.get("g4") or {}).get("checks", []))
        shortfall = f"paper vs model over {pa.get('sessions')} sessions: {checks}; real fills from go-live"
    else:
        shortfall = "paper fills against the model, verdicts from about session 30; real fills from go-live"
    corr = {k.split(":", 1)[1]: v["monthly"] for k, v in co.items()}
    books = [v for k, v in corr.items() if k in BOOKS]
    rows = [
        ("CAGR / Sharpe / Sortino / Calmar", f"{pct(rr.get('cagr'))} / {num(rr.get('sharpe'))} / {num(rr.get('sortino'))} / "
                                             f"{num(rr.get('calmar'))}", "every backtest, scorecard"),
        ("Max drawdown / volatility", f"{pct(rr.get('max_drawdown'))} / {pct(rr.get('ann_vol'))} a year", "every backtest, scorecard"),
        ("CVaR 95% (daily) / skew / kurtosis", f"{pct(rr.get('cvar_95_daily'))} / {num(rr.get('skew'))} / {num(rr.get('kurtosis'), 1)}",
         "scorecard (skew and kurtosis also every backtest)"),
        ("Information ratio (vs NIFTY 50 TRI)", num(rr.get("information_ratio_regression")), "scorecard"),
        ("Beta and factor exposures", f"market {EN_DASH.join(num(b) for b in market)}, momentum "
                                      f"{_signed(_f(betas.get('momentum')), False, 2)}, low volatility "
                                      f"{_signed(_f(betas.get('low_vol')), False, 2)}, size {_signed(_f(betas.get('size')), False, 2)} "
                                      f"(R\u00b2 {num(fx.get('r_squared'))})", "scorecard"),
        ("Alpha after factors", f"{pct(fx.get('alpha_annual'))} a year (t {num(fx.get('alpha_t_hac'), 1)}), "
                                f"{pct(fx.get('alpha_share_of_excess_return'), 0)} of the excess return", "scorecard"),
        ("Implementation shortfall / slippage", shortfall, "daily paper email (G4)"),
        ("Market impact", f"{num(mi.get('impact_bps_value_weighted'), 0)} bp per trade (value-weighted), "
                          f"{pct(mi.get('impact_drag_annual'))} a year drag", "backtest cost model, scorecard"),
        ("Turnover / cost drag", f"{num(rr.get('annual_turnover'), 1)}\u00d7 a year / {pct(rr.get('cost_drag'))} a year", "every backtest"),
        ("Hit rate / win-loss / profit factor", f"{pct(tr.get('hit_rate'), 0)} / {num(tr.get('win_loss_ratio'), 1)} / "
                                                f"{num(tr.get('profit_factor'))}", "scorecard"),
        ("Average P&L per round trip", f"{_signed(_f(tr.get('avg_pnl_per_round_trip_bp_of_equity')), False)} bp of equity; "
                                       f"median hold {num(tr.get('median_holding_days'), 0)} days", "scorecard"),
        (f"Walk-forward, out of sample ({str(wf.get('start', ''))[:4]}\u2013{str(wf.get('end', ''))[2:4]}, {wf.get('n_folds')} folds)",
         f"Sharpe {num(wf.get('sharpe'))}, CAGR {pct(wf.get('cagr'))}, max drawdown {pct(wf.get('max_drawdown'))}; "
         f"{wf.get('negative_oos_years')} negative years", "scorecard"),
        ("Deflated Sharpe / PBO", f"{num((ov.get('dsr') or {}).get('dsr'))} (bar 0.95) / {pct((ov.get('pbo') or {}).get('pbo'), 0)} "
                                  f"(bar 30%), {(ov.get('pbo') or {}).get('n_trials')} configurations", "scorecard, `validate`"),
        ("Parameter sensitivity", f"{ps.get('n_neighbours')} one-setting changes move Sharpe {num(ps.get('delta_sharpe_min'))} to "
                                  f"{num(ps.get('delta_sharpe_max'))}; fragile settings: "
                                  + ", ".join(_FRIENDLY.get(x, x) for x in ps.get("fragile_settings", [])), "scorecard"),
        ("Regime stability", "; ".join(f"{k} markets ({pct((trend.get(k) or {}).get('share_of_days'), 0)} of days) Sharpe "
                                       f"{num((trend.get(k) or {}).get('sharpe'))}" for k in ("bull", "bear", "sideways") if trend.get(k))
         + (f"; elevated VIX {num((vix.get('elevated') or {}).get('sharpe'))}" if vix.get("elevated") else ""), "scorecard"),
        ("Alpha decay", f"{decay}: {_signed(_f(at.get('slope_per_year')))} a year (p {num(at.get('p_value'))}); the signal's rank "
                        f"IC {num((ic.get('5') or {}).get('mean_ic'), 3)} at 5 days, {num((ic.get('252') or {}).get('mean_ic'), 3)} at 252",
         "scorecard"),
        ("Capacity", f"about \u20b9{_f(cap.get('capital_where_impact_eats_half_the_edge_inr')) / 1e7:,.0f} crore before impact eats "
                     f"half the edge", "scorecard"),
        ("Correlation (monthly)", ", ".join(f"{k.replace('NIFTY50_TRI', 'NIFTY 50 TRI')} {num(v)}" for k, v in corr.items()
                                           if k in BOOKS or k in ("NIFTY50_TRI", "GOLDBEES", "SILVERBEES")),
         "scorecard; sleeve tests"),
    ]
    head = (f"From the strategy scorecard of {str(r['as_of'])[:10]}:\n"
            f"- **Book:** configuration `{r['config_hash'][:8]}`, run `{r['run_id']}`.\n"
            f"- **Backtest:** {r['window'][0][:4]}\u2013{r['window'][1][2:4]}, cost model {r['cost_model']}.\n"
            f"- **Convention:** Sharpe is excess over the {pct(r['risk_free_annual'])} risk-free rate.\n\n"
            "Refresh it with `python -m runners.run_nse_engine scorecard --book all`\n"
            "(JSON in `data/nse_engine/scorecard/`).  The deployed book's run rewrites this section.")
    table = "\n".join(["| Metric | Latest | Tracked in |", "|---|---|---|", *(f"| {a} | {b} | {c} |" for a, b, c in rows)])
    change = _changes(r, previous)
    tail = ("**How often it updates:**\n"
            "- every recorded backtest stores the return, risk, turnover and hit-rate\n  figures;\n"
            "- the daily paper email carries the paper-vs-model checks;\n"
            "- the full scorecard runs at milestones (SC1b, a promotion, the January\n  roll-forward) and quarterly.\n\n"
            "**Gaps:**\n- real slippage can be measured only once live;\n"
            + (f"- the books correlate {num(min(books))}\u2013{num(max(books))}, so they are one return engine, and\n"
               "  only the gold sleeve diversifies.\n\n" if books else "\n")
            + "A mediocre but uncorrelated Sharpe can be worth more than a great\n"
              "correlated one. New sleeves are therefore judged on the combined book: a\n"
              "quarter of the book in the sleeve must raise its Sharpe.")
    parts = ["## Current performance: NSE engine, deployed book", head, table] + ([change] if change else []) + [tail]
    return "\n\n".join(parts)


def update_readme(report: Dict[str, Any], json_dir: str = "data/nse_engine/scorecard", readme: str = README) -> bool:
    """Rewrite the README section between the scorecard markers from the deployed book's report."""
    path = Path(readme)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if README_START not in text or README_END not in text:
        logger.warning("README markers not found in %s: section not updated", readme)
        return False
    older = sorted(p for p in Path(json_dir).glob(f"*_{report['book']}.json") if p.name[:10] < str(report["as_of"])[:10])
    previous = json.loads(older[-1].read_text()) if older else None
    head, rest = text.split(README_START, 1)
    tail = rest.split(README_END, 1)[1]
    path.write_text(f"{head}{README_START}\n{readme_section(report, previous)}\n{README_END}{tail}", encoding="utf-8")
    return True


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Strategy scorecard (SC1) for a book's latest recorded run")
    ap.add_argument("--book", default="all", choices=["all", *BOOKS])
    ap.add_argument("--run", help="a recorded run id (default: the book's latest on the current cost model)")
    ap.add_argument("--paper-schema", help="the paper book's Neon schema (default public)")
    ap.add_argument("--md-dir", default="docs/scorecards")
    ap.add_argument("--json-dir", default="data/nse_engine/scorecard")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    shared: Dict[str, Any] = {}
    for book in (list(BOOKS) if args.book == "all" else [args.book]):
        report = build(book, run_id=args.run if args.book != "all" else None, paper_schema=args.paper_schema, shared=shared)
        md, _ = write(report, args.md_dir, args.json_dir)
        if book == "deployed" and update_readme(report, args.json_dir):
            print(f"deployed: {README} section updated")
        verdicts = ", ".join(f"{p['rule']} {'PASS' if p['pass'] else 'FAIL' if p['pass'] is False else 'n/a'}" for p in report["pass_rules"])
        print(f"{book}: {md} ({verdicts})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
