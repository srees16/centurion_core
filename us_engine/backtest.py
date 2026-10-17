"""US backtests (tracker US4): the NSE engine on the survivorship-free US store, adapted to the US market.

The engine and its rules are India's; what changes for US stocks, and why:

- Universe: each day's point-in-time members of the S&P 500, the Nasdaq-100 or both
  (``universe.members_only``; ``us_store/panel/member*.parquet``), then the liquidity filter in
  dollars ($20 million a day).  No price floor: the store's prices are split-adjusted, so a floor
  would drop the names that later split most (NVDA under $5 before 2016), and index membership
  already screens penny stocks.
- Prices: the store's split-adjusted OHLC, back-adjusted for cash dividends as the NSE panel is
  (total return, ``panel.dividend_multipliers``).  Yahoo records most spin-offs and special
  distributions as dividends (MO 2008 $51.06, WY 2010 $26.42, TGNA 2017 $8.53); the few it does not
  are added here (``UNRECORDED_DISTRIBUTIONS``), so a distribution is never read as a loss.
- Regime: the index the universe tracks and its volatility index: the S&P 500 and VIX, or the
  Nasdaq-100 and VXN; breadth over the members.
- Sizing: your capital ($10,000, US-Y2) in fractional shares, as Vested and INDmoney sell them
  (``portfolio.fractional_shares``); India trades whole shares.
- Costs: the US schedule (``costs.market`` "US": SEC and FINRA fees on sells), the broker's commission
  (``us_engine.costs.BROKERS``, default Vested Premium; round 1 ran without one, as pre-registered) and
  the engine's impact model; FX, plan fees, LRS and tax are in ``us_engine.costs``.
- No sector cap: the store has no point-in-time sectors, and today's GICS map applied to history
  leaks the future (India's SB2).  No metal sleeves: the India book holds gold and silver.
- Results in USD, and in INR through ``in_inr`` (US-Y2), against SPY, RSP, QQQ and QQQE total
  returns (``us_store/panel/benchmarks.parquet``).

Results come only from pre-registered rounds, so ``smoke`` prints how the book traded (members only,
fractions, costs, regime), never its returns.

Round 1 (tracker US5 / US6, validation plan 5ad): the three India books' rules (``ROUND1``: A
deployed, B candidate, C E4) run on Kaggle (``cloud.kaggle_runner us-round``) into the US core
family's registry (``RUNS_DIR``); ``verdict`` then applies the pre-registered gates.

    python -m us_engine.backtest smoke --universe union      # or sp500, ndx
    python -m us_engine.backtest verdict --wf-returns data/nse_engine/wf_oos_returns_us1.csv \\
        --reported data/nse_engine/kaggle_out/latest/wf/us_round.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from nse_engine.config import EngineConfig
from nse_engine.data.panel import dividend_multipliers
from nse_engine.deployment import REPO_ROOT, load_deployment
from nse_engine.metrics import cagr_from_growth, years_elapsed
from nse_engine.types import MarketData
from us_engine.advisory import DEPLOYMENT
from us_engine.costs import BROKERS, DEFAULT_BROKER
from us_engine.store import PANEL_DIR

logger = logging.getLogger(__name__)

#: Your US capital (US-Y2, 16 Oct 2026); results are judged in USD and in INR.
CAPITAL_USD = 10_000.0
#: The 3-month T-bill's 2013-25 average (``market.parquet`` IRX): the constant the engine's Sharpe
#: subtracts; US5's scorecard uses the dated series.
RISK_FREE_ANNUAL = 0.017
#: universe -> (membership file, regime index, its volatility index)
UNIVERSES: Dict[str, Tuple[str, str, str]] = {
    "union": ("member.parquet", "SPX", "VIX"),
    "sp500": ("member_sp500.parquet", "SPX", "VIX"),
    "ndx": ("member_ndx.parquet", "NDX", "VXN"),
}
#: Spin-off days whose distribution the data leaves out (in full or in part), by panel instrument: the
#: missing part is added as a dividend, sized so the day's total return matches the S&P 500's
#: (checked 16 Oct: each drop is the distribution, not news).
UNRECORDED_DISTRIBUTIONS = {
    ("JCI", "2007-07-02"): "Tyco (Yahoo's JCI before 2016) spun off Covidien and TE Connectivity; Yahoo records "
                           "$10.69 of about $43",
    ("TWX", "2009-03-30"): "Time Warner spun off Time Warner Cable (with a 1-for-3 reverse split); Tiingo "
                           "records no distribution",
    ("EXPE", "2011-12-21"): "Expedia spun off TripAdvisor (with a 1-for-2 reverse split); Yahoo records $15.12 "
                            "of about $29",
    ("LMCA", "2014-07-24"): "Liberty Media paid two Series C shares per share; Tiingo records no distribution",
}
#: The US settings on top of the India deployment's rules (dotted EngineConfig fields).
US_MARKET = {
    "risk_free_annual": RISK_FREE_ANNUAL,
    "data.store_dir": str(PANEL_DIR),
    "universe.top_n_liquid": 600,                  # every member that passes the filters
    "universe.min_price_inr": 0.0,                 # USD; see the module docstring
    "universe.min_median_value_inr": 2.0e7,        # USD 20 million a day
    "universe.members_only": True,
    "universe.calendar_schedule": True,            # schedules on the calendar, not on row counts
    "signals.calendar_schedule": True,
    "portfolio.calendar_schedule": True,           # rebalance on each week's first session
    "portfolio.fractional_shares": True,
    "portfolio.sector_cap": 1.0,
    "sleeves.gold_enabled": False,
    "sleeves.silver_enabled": False,
    "costs.market": "US",
}


def _distribution_amounts(close: pd.DataFrame, dividends: pd.DataFrame, spx: pd.Series) -> pd.DataFrame:
    """``dividends`` plus the unrecorded part of the ``UNRECORDED_DISTRIBUTIONS``."""
    out = dividends.copy()
    prior = close.ffill().shift(1)
    market = spx / spx.shift(1) - 1.0
    for (name, day), _ in UNRECORDED_DISTRIBUTIONS.items():
        t = pd.Timestamp(day)
        if name not in close.columns or t not in close.index:
            continue
        missing = prior.at[t, name] * (1.0 + market.get(t, 0.0)) - close.at[t, name] - out.at[t, name]
        if np.isfinite(missing) and missing > 0:
            out.at[t, name] += float(missing)
    return out


def load_us_market(universe: str = "union", panel_dir: Path = PANEL_DIR, float_dtype: str = "float32") -> MarketData:
    """The US store as the engine's ``MarketData``: total-return prices, the universe's point-in-time
    membership, the regime's index closes; only instruments that were members of ``universe``."""
    if universe not in UNIVERSES:
        raise ValueError(f"universe must be one of {sorted(UNIVERSES)}, not {universe!r}")
    member_file, _, _ = UNIVERSES[universe]
    panel_dir = Path(panel_dir)
    member = pd.read_parquet(panel_dir / member_file)
    cols = member.columns[member.to_numpy().any(axis=0)]
    member = member[cols]
    f = {name: pd.read_parquet(panel_dir / f"{name}.parquet")[cols].astype("float64")
         for name in ("open", "high", "low", "close", "volume", "dividends")}
    market = pd.read_parquet(panel_dir / "market.parquet").reindex(member.index)
    amounts = _distribution_amounts(f["close"], f["dividends"].fillna(0.0), market["SPX"])
    mult, _ = dividend_multipliers(f["close"], amounts.where(amounts > 0).to_numpy(), open_=f["open"])
    value = f["close"] * f["volume"]                       # dollars traded, before any adjustment
    prices = {k: (f[k] * mult).astype(float_dtype) for k in ("open", "high", "low", "close")}
    manifest = json.loads((panel_dir / "manifest.json").read_text())
    data = MarketData(
        dates=pd.DatetimeIndex(member.index), volume=f["volume"].astype(float_dtype), value=value.astype(float_dtype),
        index_close=market[["SPX", "NDX", "VIX", "VXN"]].astype("float64"), members=member.astype(bool),
        source=f"us_store {manifest['data_hash']} ({universe})",
        data_hash=hashlib.sha256(f"{manifest['data_hash']}|{universe}|tr|{sorted(UNRECORDED_DISTRIBUTIONS)}"
                                 .encode()).hexdigest()[:16],
        **prices,
    )
    data.validate()
    return data


def us_engine_config(universe: str = "union", base: Optional[EngineConfig] = None,
                     capital: float = CAPITAL_USD, broker: Optional[str] = DEFAULT_BROKER) -> EngineConfig:
    """The India deployment's rules (``base``, default the deployed book) with the US settings: members only,
    fractional shares at ``capital`` dollars, US costs with ``broker``'s commission (None: none, as round 1
    was pre-registered), the universe's own index and volatility index."""
    _, index, vix = UNIVERSES[universe]
    base = base if base is not None else load_deployment(DEPLOYMENT).engine
    fees = ({"costs.commission_rate": BROKERS[broker].commission_rate,
             "costs.commission_cap": BROKERS[broker].commission_cap} if broker else {})
    return base.replace(**US_MARKET, **fees, initial_capital=float(capital),
                        **{"portfolio.min_trade_value_inr": 0.01 * capital,      # 1%, as Rs 5,000 of Rs 5 lakh
                           "regime.index_symbol": index, "regime.vix_symbol": vix})


def in_inr(series_usd: pd.Series, panel_dir: Path = PANEL_DIR) -> pd.Series:
    """A dollar series (equity, prices) in rupees at each day's USD/INR (``market.parquet``), for US-Y2."""
    rate = pd.read_parquet(Path(panel_dir) / "market.parquet")["USDINR"].ffill()
    return series_usd * rate.reindex(series_usd.index).ffill()


# ── round 1 (tracker US5 / US6, validation plan 5ad) ─────────────

#: Round 1's configurations: the India books' rules, each with its drawdown overlay.
ROUND1 = {"A": "nse_engine_deployed.json", "B": "nse_engine_candidate.json", "C": "nse_engine_e4.json"}
#: The US core family's registry (research protocol section 1); round 1 spends 3 of its 3 configurations.
RUNS_DIR = "data/nse_engine/runs_us"
#: Pre-registered gates (plan 5ad): deflated Sharpe at the family's N, and the Newey-West t of the
#: alpha over SPY total return.
DSR_MIN = 0.95
ALPHA_T_MIN = 2.0
#: The combined India + US test (US-Y7): fixed INR weights from your capital, India Rs 30 lakh (U6) and
#: $10,000 (US-Y2) at the 2025 close (Rs 89.8), 23% US, rounded to 25%; rebalanced monthly.
US_WEIGHT = 0.25
INDIA_RF = 0.065
#: The India leg: the deployed book's register backtest (2013-25, cost model 4, data dab49d3c).
INDIA_RUN = "20261010T032441064151Z_679cbd0c"


def book_rules(book_file: str) -> EngineConfig:
    """An India book's rules as one backtestable config: its engine with its drawdown overlay folded in
    (``Deployment.reference_config`` without the paper start date)."""
    dep = load_deployment(REPO_ROOT / "config" / book_file)
    rule = asdict(dep.drawdown_rule) if dep.drawdown_rule is not None else {}
    return dep.engine.replace(**{f"drawdown.{k}": v for k, v in rule.items()})


def round1_configs(universe: str = "union") -> Dict[str, EngineConfig]:
    """A, B and C through the US adapter, recorded in the US registry.  As pre-registered: no commission
    (US4 took Vested and INDmoney for commission-free; US3 found 0.15-0.3% a trade, plan 5ad)."""
    return {key: us_engine_config(universe, base=book_rules(name), broker=None).replace(runs_dir=RUNS_DIR)
            for key, name in ROUND1.items()}


def _flat(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in d.items():
        if isinstance(value, dict):
            out.update(_flat(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def round1_points(universe: str = "union") -> Dict[str, Dict[str, Any]]:
    """Each configuration as dotted overrides of A, the walk-forward's grid; checked by config hash."""
    cfgs = round1_configs(universe)
    base = _flat(cfgs["A"].to_dict())
    points = {}
    for key, cfg in cfgs.items():
        diff = {k: v for k, v in _flat(cfg.to_dict()).items() if base.get(k) != v}
        if cfgs["A"].replace(**diff).config_hash() != cfg.config_hash():
            raise RuntimeError(f"configuration {key} is not A with {diff}")
        points[key] = diff
    return points


def tbill(index: pd.DatetimeIndex, panel_dir: Path = PANEL_DIR) -> pd.Series:
    """The 3-month T-bill (``market.parquet`` IRX, % a year) as a daily rate on ``index``: rate / 252, as
    the engine's Sharpe subtracts its constant."""
    irx = pd.read_parquet(Path(panel_dir) / "market.parquet")["IRX"].ffill()
    return (irx / 100.0 / 252).reindex(pd.DatetimeIndex(index)).ffill().bfill()


def fund_returns(name: str, index: pd.DatetimeIndex, panel_dir: Path = PANEL_DIR) -> pd.Series:
    """A benchmark fund's daily total return (``benchmarks.parquet``: SPY, RSP, QQQ, QQQE) on ``index``."""
    level = pd.read_parquet(Path(panel_dir) / "benchmarks.parquet")[name].dropna()
    return level.pct_change().reindex(pd.DatetimeIndex(index)).fillna(0.0)


def stats(r: pd.Series, rf_daily: pd.Series, periods: int = 252) -> Dict[str, float]:
    """CAGR (calendar years), excess Sharpe over ``rf_daily``, max drawdown, Calmar and volatility."""
    ex = r - rf_daily.reindex(r.index).fillna(0.0)
    eq = (1.0 + r).cumprod()
    cagr = cagr_from_growth(float(eq.iloc[-1]), years_elapsed(r.index))
    dd = float((eq / eq.cummax() - 1.0).min())
    return {"cagr": cagr, "sharpe": float(ex.mean() / ex.std(ddof=1) * np.sqrt(periods)), "max_dd": dd,
            "calmar": cagr / -dd if dd < 0 else float("nan"), "vol": float(r.std(ddof=1) * np.sqrt(periods))}


def _monthly(r: pd.Series) -> pd.Series:
    return (1.0 + r).groupby(pd.DatetimeIndex(r.index).to_period("M")).prod() - 1.0


def monthly_inr(r_usd: pd.Series, panel_dir: Path = PANEL_DIR) -> pd.Series:
    """Monthly returns of a dollar return series held in rupees (``in_inr`` at month ends)."""
    equity = in_inr((1.0 + r_usd).cumprod(), panel_dir)
    month_end = equity.groupby(pd.DatetimeIndex(equity.index).to_period("M")).last()
    first = float(equity.iloc[0] / (1.0 + r_usd.iloc[0]))
    return month_end.pct_change().fillna(month_end.iloc[0] / first - 1.0)


def combined(india_m: pd.Series, us_m: pd.Series, weight: float = US_WEIGHT) -> Dict[str, float]:
    """India and US monthly INR returns at fixed weights, rebalanced monthly: Sharpe over India's 6.5%."""
    both = pd.concat([india_m, us_m], axis=1, join="inner").dropna()
    r = (1.0 - weight) * both.iloc[:, 0] + weight * both.iloc[:, 1]
    eq = (1.0 + r).cumprod()
    cagr = float(eq.iloc[-1] ** (12.0 / len(r)) - 1.0)
    dd = float((eq / eq.cummax() - 1.0).min())
    ex = r - INDIA_RF / 12.0
    return {"cagr": cagr, "sharpe": float(ex.mean() / ex.std(ddof=1) * np.sqrt(12)), "max_dd_monthly": dd,
            "calmar": cagr / -dd if dd < 0 else float("nan"), "months": int(len(r))}


def _latest_runs(runs_dir: Path, prefix: str) -> Dict[str, Path]:
    """The latest recorded run per tag ``<prefix><key>``."""
    best: Dict[str, Tuple[str, Path]] = {}
    for manifest in Path(runs_dir).glob("*/manifest.json"):
        m = json.loads(manifest.read_text())
        tag = str(m.get("tag", ""))
        if tag.startswith(prefix) and (tag not in best or m["created_at"] > best[tag][0]):
            best[tag] = (m["created_at"], manifest.parent)
    return {tag[len(prefix):]: path for tag, (_, path) in sorted(best.items())}


def _returns(run_dir: Path) -> pd.Series:
    s = pd.read_csv(Path(run_dir) / "returns.csv", index_col=0).squeeze("columns").astype("float64")
    s.index = pd.DatetimeIndex(s.index)
    return s


def round1_verdict(wf_returns: Path, reported: Optional[Path] = None, runs_dir: Path = REPO_ROOT / RUNS_DIR,
                   india_runs: Path = REPO_ROOT / "data/nse_engine/runs", panel_dir: Path = PANEL_DIR) -> Dict:
    """Plan 5ad's gates on the recorded round: per configuration G1-G4 and G6, the family's walk-forward
    G5, and the reported views (INR, other funds, the India book, 2007-25)."""
    from nse_engine.validation.diagnostics import alpha_beta
    from nse_engine.validation.dsr import deflated_sharpe
    from nse_engine.validation.pbo import cscv_pbo

    runs = _latest_runs(runs_dir, "us5-")
    if sorted(runs) != sorted(ROUND1):
        raise SystemExit(f"need the recorded us5-A/B/C runs in {runs_dir}, found {sorted(runs)}")
    rets = {k: _returns(p).iloc[1:] for k, p in runs.items()}        # day one is the flat start
    index = rets["A"].index
    rf = tbill(index, panel_dir)
    excess = pd.DataFrame({k: r - rf.reindex(r.index) for k, r in rets.items()}).dropna()
    spy = fund_returns("SPY", index, panel_dir)
    spy_stats = stats(spy, rf)
    india = _monthly(_returns(Path(india_runs) / INDIA_RUN).iloc[1:])
    with_spy = combined(india, monthly_inr(spy, panel_dir))
    pbo = cscv_pbo(excess, n_splits=16)
    out: Dict[str, Any] = {"window": [str(index[0].date()), str(index[-1].date())], "spy": spy_stats,
                           "india_plus_spy": with_spy, "pbo": pbo.get("pbo"), "configs": {}}
    for key, r in rets.items():
        s = stats(r, rf)
        ab = alpha_beta(excess[key], spy.reindex(excess.index) - rf.reindex(excess.index), rf_annual=0.0)
        dsr = deflated_sharpe(excess[key], trials_matrix=excess, rf_annual=0.0)
        mix = combined(india, monthly_inr(r, panel_dir))
        manifest = json.loads((runs[key] / "manifest.json").read_text())
        gates = {"G1_dsr": dsr["dsr"] >= DSR_MIN, "G2_sharpe_over_spy": s["sharpe"] > spy_stats["sharpe"],
                 "G3_alpha_t": ab["alpha_t_hac"] >= ALPHA_T_MIN, "G4_maxdd_vs_spy": s["max_dd"] >= spy_stats["max_dd"],
                 "G6_earns_its_place": mix["sharpe"] > with_spy["sharpe"]}
        inr = in_inr((1.0 + r).cumprod(), panel_dir)
        out["configs"][key] = {
            "run_id": manifest["run_id"], "config_hash": manifest["config_hash"], "usd": s,
            "inr": {"cagr": cagr_from_growth(float(inr.iloc[-1] / inr.iloc[0]), years_elapsed(inr.index)),
                    "max_dd": float((inr / inr.cummax() - 1.0).min())},
            "alpha_annual": ab["alpha_annual"], "alpha_t": ab["alpha_t_hac"], "beta": ab["beta"],
            "dsr": dsr["dsr"], "india_plus_us": mix,
            "corr_with_india_monthly": float(pd.concat([india, monthly_inr(r, panel_dir)], axis=1).dropna().corr().iloc[0, 1]),
            "turnover": manifest["metrics"].get("annual_turnover"), "cost_drag": manifest["metrics"].get("cost_drag"),
            "years": {str(y): float(v) for y, v in ((1.0 + r).groupby(r.index.year).prod() - 1.0).items()},
            "gates": gates, "passed": all(gates.values())}
    for fund in ("RSP", "QQQ", "QQQE"):
        out[fund.lower()] = stats(fund_returns(fund, index, panel_dir), rf)
    oos = pd.read_csv(wf_returns, index_col=0).squeeze("columns").astype("float64")
    oos.index = pd.DatetimeIndex(oos.index)
    oos_rf = tbill(oos.index, panel_dir)
    wf = {"oos": stats(oos, oos_rf), "spy": stats(fund_returns("SPY", oos.index, panel_dir), oos_rf)}
    wf["G5_oos_sharpe_over_spy"] = wf["oos"]["sharpe"] > wf["spy"]["sharpe"]
    out["walk_forward"] = wf
    if reported is not None:
        rep = json.loads(Path(reported).read_text()).get("reported", {})
        wide = {}
        for key, row in rep.items():
            r = pd.Series(row["returns"], dtype="float64")
            r.index = pd.DatetimeIndex(r.index)
            r = r.iloc[1:]
            wrf = tbill(r.index, panel_dir)
            spy_w = fund_returns("SPY", r.index, panel_dir)
            wide[key] = {"usd": stats(r, wrf), "spy": stats(spy_w, wrf),
                         "y2008": float((1.0 + r[r.index.year == 2008]).prod() - 1.0),
                         "spy_2008": float((1.0 + spy_w[spy_w.index.year == 2008]).prod() - 1.0)}
        out["window_2007_25"] = wide
    passers = [k for k, v in out["configs"].items() if v["passed"]] if wf["G5_oos_sharpe_over_spy"] else []
    out["passers"] = passers
    out["pick"] = max(passers, key=lambda k: out["configs"][k]["usd"]["sharpe"]) if passers else None
    for key, row in out["configs"].items():
        r = rets[key]
        late = r[r.index >= "2017-01-01"]
        row["usd"]["sharpe_2017_25"] = stats(late, tbill(late.index, panel_dir))["sharpe"]
    out["pbo_n"] = len(rets)
    return out


def record_journal(result: Dict, day: str, runs_dir: Path = REPO_ROOT / RUNS_DIR) -> int:
    """Round 1 in the metrics journal (US tracker US6): a backtest row per configuration (books us_a, us_b,
    us_c; USD, Sharpe over the T-bill) and the family's walk-forward (book us).  Idempotent on (book, kind,
    ref); returns the rows added."""
    from nse_engine import journal

    old = journal.read_journal()
    seen = set(zip(old["book"], old["kind"], old["ref"]))
    rows = []
    for key, c in result["configs"].items():
        m = json.loads((Path(runs_dir) / c["run_id"] / "manifest.json").read_text())
        failed = [g.split("_")[0] for g, ok in c["gates"].items() if not ok]
        rows.append(dict(
            date=day, book=f"us_{key.lower()}", kind="backtest",
            event=f"US round 1 (plan 5ad): {key} = India {ROUND1[key][11:-5]} rules, S&P 500 + Nasdaq-100",
            cost_model=m.get("cost_model"), data_hash=str(m.get("data_hash"))[:8], config_hash=c["config_hash"][:8],
            ref=c["run_id"], cagr=c["usd"]["cagr"], sharpe=c["usd"]["sharpe"], max_dd=c["usd"]["max_dd"],
            calmar=c["usd"]["calmar"], sharpe_2017_25=c["usd"]["sharpe_2017_25"], dsr=c["dsr"], pbo=result["pbo"],
            pbo_n=result["pbo_n"],
            note=(f"USD; Sharpe over the T-bill. SPY {result['spy']['cagr']:.1%} / {result['spy']['sharpe']:.2f} / "
                  f"{result['spy']['max_dd']:.1%}. " + (f"Failed {', '.join(failed)}." if failed else "Passed G1-G4, G6."))))
    wf = result["walk_forward"]
    rows.append(dict(
        date=day, book="us", kind="walk_forward", event="US round 1 walk-forward: 9 folds choosing A, B or C",
        ref="us_round1_wf", cagr=wf["oos"]["cagr"], sharpe=wf["oos"]["sharpe"], max_dd=wf["oos"]["max_dd"],
        calmar=wf["oos"]["calmar"],
        note=f"OOS 2017-25, USD. SPY {wf['spy']['cagr']:.1%} / {wf['spy']['sharpe']:.2f} / {wf['spy']['max_dd']:.1%}."))
    new = [r for r in rows if (r["book"], r["kind"], r["ref"]) not in seen]
    if new:
        journal.write(pd.concat([old, pd.DataFrame(new)], ignore_index=True))
    return len(new)


def smoke(universe: str = "union") -> Dict:
    """Run the deployed rules through the US adapter, 2013-25, unrecorded, and report how the book traded
    - never its returns (those come only from pre-registered rounds)."""
    from nse_engine.engine import EngineCache, run_backtest

    data = load_us_market(universe)
    cfg = us_engine_config(universe)
    quiet = logging.getLogger("nse_engine.engine")
    level = quiet.level
    quiet.setLevel(logging.WARNING)                        # its summary line prints the Sharpe and CAGR
    try:
        cache = EngineCache(data, cfg)
        res = run_backtest(data, cfg, record=False, cache=cache)
    finally:
        quiet.setLevel(level)
    t = res.trades
    held = (res.weights > 0).sum(axis=1)
    members = data.members
    decided = members.shift(1, fill_value=False)           # an order fills the session after its decision
    buys = t[t["side"] == "BUY"]
    off_list = int(sum(not decided.at[pd.Timestamp(r.date), r.symbol] for r in buys.itertuples(index=False)))
    whole = int(np.isclose(t["quantity"], np.round(t["quantity"])).sum())
    regime = cache.regime.state.loc[res.equity.index].value_counts(normalize=True)
    report = {
        "universe": universe, "data_hash": data.data_hash, "config_hash": cfg.config_hash()[:8],
        "window": [str(res.equity.index[0].date()), str(res.equity.index[-1].date())],
        "instruments": int(data.close.shape[1]),
        "members_a_day": [int(members.sum(axis=1).min()), int(members.sum(axis=1).max())],
        "universe_a_day_median": int(np.median(cache.universe_mask.sum(axis=1))),
        "names_held_median": int(held.median()), "days_invested": round(float((held > 0).mean()), 3),
        "trades": int(len(t)), "buys_decided_off_the_index": off_list,
        "trades_in_whole_shares": whole,
        "cost_bps_per_trade": round(float(t["cost_inr"].sum() / max(t["value_inr"].sum(), 1e-9) * 1e4), 2),
        "regime_share": {str(k): round(float(v), 3) for k, v in regime.items()},
        "returns": "withheld: smoke checks mechanics; returns come only from pre-registered rounds (research protocol)",
    }
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m us_engine.backtest", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke", help="run the deployed rules on the US store; report mechanics, not returns")
    s.add_argument("--universe", choices=sorted(UNIVERSES), default="union")
    v = sub.add_parser("verdict", help="round 1's pre-registered gates on the recorded runs (plan 5ad)")
    v.add_argument("--wf-returns", required=True, help="the stitched walk-forward OOS returns (cloud.wf_stitch)")
    v.add_argument("--reported", help="us_round.json from the Kaggle run (the 2007-25 window, reported)")
    v.add_argument("--runs-dir", default=str(REPO_ROOT / RUNS_DIR))
    v.add_argument("--out", default=str(REPO_ROOT / "data/nse_engine/us_round1_verdict.json"))
    v.add_argument("--journal", metavar="DATE", help="also add the rows to docs/metrics_journal.csv, dated DATE")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.cmd == "verdict":
        result = round1_verdict(Path(args.wf_returns), Path(args.reported) if args.reported else None,
                                Path(args.runs_dir))
        Path(args.out).write_text(json.dumps(result, indent=1, default=str))
        print(json.dumps(result, indent=1, default=str))
        if args.journal:
            print(f"journal: {record_journal(result, args.journal, Path(args.runs_dir))} rows added")
        return 0
    print(json.dumps(smoke(args.universe), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
