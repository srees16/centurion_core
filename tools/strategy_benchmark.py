#!/usr/bin/env python3
"""Benchmark the single-stock strategies on one honest footing (tracker BT1).

Each strategy runs on the same names and the same bars, through its own ``run()``, so what is measured is what the
app shows.  The research store supplies the prices: official splits, bonuses and dividends, delisted names included.
The names are the most traded universe members on the last session before the window, chosen from what was known
then, so later winners are not picked by hindsight.  A name's sleeve gets an equal share of capital, kept to itself;
the book is the sum of the sleeves, and idle cash earns the liquid rate, as the diversifier gate's books do.  Bars
before the window warm up the indicators.  Metrics are the engine's (``nse_engine.metrics``, its risk-free rate).

These are in-sample figures at each strategy's default parameters: a yardstick, not evidence of an edge.  Anything
that looks good here goes through docs/research_protocol.md like any other idea.  The 2026 holdout stays sealed.

    python -m tools.strategy_benchmark                               # 50 names, 2013-25, every strategy below
    python -m tools.strategy_benchmark --names 30 --only volume_profile,order_flow --workers 4
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import multiprocessing
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger("strategy_benchmark")

#: Key -> (module, class).  Pairs, basket and crypto strategies need several tickers at once, and Shooting Star only
#: goes short, which a delivery book cannot do: none of them fits one long sleeve per stock.
STRATEGIES: Dict[str, Tuple[str, str]] = {
    "macd": ("momentum_trading.macd_oscillator", "MACDOscillatorStrategy"),
    "awesome_oscillator": ("momentum_trading.awesome_oscillator", "AwesomeOscillatorStrategy"),
    "heikin_ashi": ("momentum_trading.heikin_ashi", "HeikinAshiStrategy"),
    "parabolic_sar": ("momentum_trading.parabolic_sar", "ParabolicSARStrategy"),
    "rsi_pattern": ("pattern_recognition.rsi_pattern", "RSIPatternStrategy"),
    "support_resistance": ("pattern_recognition.support_resistance", "SupportResistanceStrategy"),
    "bollinger_pattern": ("pattern_recognition.bollinger_pattern", "BollingerPatternStrategy"),
    "volume_profile": ("volume_microstructure.volume_profile", "VolumeProfileStrategy"),
    "anchored_vwap": ("volume_microstructure.anchored_vwap", "AnchoredVWAPStrategy"),
    "order_flow": ("volume_microstructure.order_flow", "OrderFlowStrategy"),
    "liquidity_sweep": ("volume_microstructure.liquidity_sweep", "LiquiditySweepStrategy"),
    "swing_combo": ("volume_microstructure.swing_combo", "SwingComboStrategy"),
    "positional_combo": ("volume_microstructure.positional_combo", "PositionalComboStrategy"),
}
CAPITAL = 100_000.0
WARMUP_YEARS = 2
_BARS: Dict[str, pd.DataFrame] = {}                       # read by the forked workers


def strategy_class(key: str):
    module, cls = STRATEGIES[key]
    return getattr(importlib.import_module(f"trading_strategies.{module}"), cls)


def pick_names(mask: pd.DataFrame, value: pd.DataFrame, start: pd.Timestamp, n: int) -> List[str]:
    """The ``n`` most traded universe members on the last session before ``start``: known then, no hindsight."""
    day = mask.index[mask.index < start][-1]
    members = mask.columns[mask.loc[day].to_numpy(dtype=bool)]
    if len(members) < n:
        raise ValueError(f"only {len(members)} universe members on {day.date()}, {n} asked for")
    liquid = value.loc[:day, members].tail(126).median().sort_values(ascending=False, kind="stable")
    return list(liquid.index[:n])


def bars_for(data, name: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """A name's adjusted bars; a session it did not trade has no row, as in a live feed."""
    df = pd.DataFrame({"Open": data.open[name], "High": data.high[name], "Low": data.low[name],
                       "Close": data.close[name], "Volume": data.volume[name]}, dtype="float64")
    df = df.loc[start:end].dropna(subset=["Open", "High", "Low", "Close"])
    df.attrs["market"] = "IND"
    return df


def _prime(key: str, df: pd.DataFrame) -> None:
    """Hand the bars to ``DataService`` through its own cache, so the strategy's ``run()`` is the code that is tested."""
    from strategies.data_service import DataService

    DataService._cache[key] = df
    DataService._cache_expiry[key] = datetime.now() + timedelta(hours=6)


def run_sleeve(job: Tuple[str, str, str, str]) -> Tuple[str, str, Optional[pd.DataFrame], Dict[str, object]]:
    """One strategy on one name: its daily value and exposure, and what the strategy itself reports."""
    key, name, start, end = job
    logging.disable(logging.CRITICAL)
    try:
        strategy = strategy_class(key)()
        strategy._create_charts = lambda *a, **k: []
        _prime(f"{name}_{start}_{end}_1d", _BARS[name])
        res = strategy.run([name], start, end, CAPITAL, risk_params={"max_position_size": 1.0})
        if not res.success or res.portfolio is None:
            return key, name, None, {"error": res.error_message or str(res.metrics.get(name))}
        p = res.portfolio
        value = (p["total_value"] if "total_value" in p.columns else p["equity"]).astype("float64")
        s = res.signals
        pos_col = next(c for c in ("positions", "signal") if c in s.columns)
        pos = s[pos_col].astype("float64").fillna(0.0)
        held = (p["holdings"].astype("float64") / value).clip(0, 1) if "holdings" in p.columns else (pos.shift(1).fillna(0.0) != 0).astype("float64")
        out = pd.DataFrame({"value": value, "invested": held})
        rep = res.metrics.get(name, {})
        return key, name, out, {"trades": int((pos.diff().fillna(0.0) != 0).sum()),
                                "reported_return": rep.get("total_return"), "reported_sharpe": rep.get("sharpe_ratio")}
    except Exception as exc:                              # noqa: BLE001 - one sleeve must not stop the table
        return key, name, None, {"error": f"{type(exc).__name__}: {exc}"}


def book_of(sleeves: Dict[str, pd.DataFrame], start: pd.Timestamp, sessions: pd.DatetimeIndex) -> Tuple[pd.Series, pd.Series]:
    """(daily returns, invested share) of equal sleeves from ``start``, each carrying its last value once it stops."""
    values, invested = [], []
    for df in sleeves.values():
        v = df["value"].reindex(sessions.union(df.index)).ffill()
        first = v.loc[:start].iloc[-1] if (v.index <= start).any() and pd.notna(v.loc[:start].iloc[-1]) else v.dropna().iloc[0]
        v = (v / first).reindex(sessions).fillna(1.0)
        values.append(v)
        invested.append(df["invested"].reindex(sessions).fillna(0.0) * v)
    total = sum(values)
    return total.pct_change().fillna(0.0), (sum(invested) / total).clip(0.0, 1.0)


def buy_and_hold(data, names: Sequence[str], start: pd.Timestamp, end: pd.Timestamp, sessions: pd.DatetimeIndex) -> pd.Series:
    """The same names bought at the first open and never touched: the yardstick for a sleeve book."""
    sleeves = {}
    for n in names:
        o, c = data.open[n].loc[start:end].dropna(), data.close[n].loc[start:end].ffill()
        if o.empty:
            continue
        sleeves[n] = pd.DataFrame({"value": c / float(o.iloc[0]), "invested": 1.0})
    return book_of(sleeves, start, sessions)[0]


def summarise(returns: pd.Series, rf: float) -> Dict[str, float]:
    from nse_engine.metrics import compute_metrics

    m = compute_metrics(returns, (1 + returns).cumprod(), rf_annual=rf, initial_capital=1.0)
    return {k: float(m[k]) for k in ("cagr", "sharpe", "max_drawdown", "calmar", "ann_vol")}


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--names", type=int, default=50, help="stocks in the book")
    p.add_argument("--start", default="2013-01-01")
    p.add_argument("--end", default="2025-12-31", help="2026 is the sealed holdout")
    p.add_argument("--only", default="", help="comma-separated strategy keys (default: all)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default="data/nse_engine/strategy_benchmark/results.json")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from nse_engine import bond_stream as bs
    from nse_engine import diversifier_gate as dg
    from nse_engine.deployment import load_deployment
    from nse_engine.universe import compute_universe_panel
    from runners.run_nse_engine import _load_data

    t0 = time.time()
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if end > pd.Timestamp("2025-12-31"):
        raise SystemExit("2026 is the sealed holdout: --end must stay in 2025 or earlier")
    keys = [k.strip() for k in args.only.split(",") if k.strip()] or list(STRATEGIES)
    unknown = sorted(set(keys) - set(STRATEGIES))
    if unknown:
        raise SystemExit(f"unknown strategies {unknown}; known: {sorted(STRATEGIES)}")
    cfg = load_deployment().engine
    warm = start - pd.DateOffset(years=WARMUP_YEARS)
    data = _load_data(cfg, data_start=(warm - pd.DateOffset(years=1)).date().isoformat())
    mask = compute_universe_panel(data, cfg.universe, exclude=cfg.sleeves.symbols).mask
    names = pick_names(mask, data.value, start, args.names)
    logger.info("%d names, the most traded before %s: %s ...", len(names), start.date(), ", ".join(names[:8]))
    for n in names:
        _BARS[n] = bars_for(data, n, warm, end)
    sessions = pd.DatetimeIndex(data.dates)
    sessions = sessions[(sessions >= start) & (sessions <= end)]
    jobs = [(k, n, warm.date().isoformat(), end.date().isoformat()) for k in keys for n in names]
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        done = pool.map(run_sleeve, jobs, chunksize=1)

    liquid = bs.bond_returns(sessions, bs.BondConfig())["liquid"]
    rf = float(cfg.risk_free_annual)
    results: Dict[str, Dict[str, object]] = {}
    for k in keys:
        rows = [d for d in done if d[0] == k]
        good = {d[1]: d[2] for d in rows if d[2] is not None}
        errors = {d[1]: d[3]["error"] for d in rows if d[2] is None}
        if not good:
            results[k] = {"error": next(iter(errors.values()), "no sleeves"), "failed": len(errors)}
            continue
        ret, invested = book_of(good, start, sessions)
        ret = ret + (1 - invested) * liquid
        stats = [d[3] for d in rows if d[2] is not None]
        reported = [s["reported_return"] for s in stats if s.get("reported_return") is not None]
        results[k] = {**summarise(ret, rf), "sleeves": len(good), "failed": len(errors), "invested": float(invested.mean()),
                      "trades_per_name_year": float(np.mean([s["trades"] for s in stats]) / ((end - warm).days / 365.25)),
                      "reported_return_mean": float(np.mean(reported)) if reported else None}
        if errors:
            results[k]["errors"] = dict(list(errors.items())[:3])

    hold = buy_and_hold(data, names, start, end, sessions)
    tri = data.index_close["NIFTY50_TRI"].astype("float64").reindex(sessions).ffill().pct_change().fillna(0.0)
    book = dg._live_book(bs.backtest_base(), liquid, sessions).dropna()
    refs = {"buy and hold, same names": summarise(hold, rf), "NIFTY 50 TRI": summarise(tri, rf), "deployed book (honest base)": summarise(book, rf)}
    out = {"window": [str(start.date()), str(end.date())], "names": names, "rf": rf, "strategies": results, "references": refs}
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=float))

    print(f"\n{args.names} names, {start.date()} to {end.date()}, risk-free {rf:.1%}, {time.time() - t0:.0f}s\n")
    print(f"{'strategy':28s}{'CAGR':>8s}{'Sharpe':>8s}{'MaxDD':>8s}{'invested':>10s}{'trades/yr':>11s}  sleeves")
    for k, r in sorted(results.items(), key=lambda kv: -(kv[1].get("sharpe", -9) if isinstance(kv[1].get("sharpe"), float) else -9)):
        if "error" in r:
            print(f"{k:28s}  ERROR {str(r['error'])[:80]}")
            continue
        print(f"{k:28s}{r['cagr']:8.1%}{r['sharpe']:8.2f}{r['max_drawdown']:8.1%}{r['invested']:10.0%}{r['trades_per_name_year']:11.1f}  "
              f"{r['sleeves']}" + (f" ({r['failed']} failed)" if r["failed"] else ""))
    print()
    for k, r in refs.items():
        print(f"{k:28s}{r['cagr']:8.1%}{r['sharpe']:8.2f}{r['max_drawdown']:8.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
