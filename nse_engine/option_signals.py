"""
Stock selection from option prices (India tracker IVX1, plan 5ap, pre-registered 27 Oct 2026, 22:00 IST).

Two published signals on NSE's F&O stocks, each taken at the last session of each month from that session's
closes, through Black-Scholes (``theory.implied_volatility``, ``OptionsConfig``'s rate, no dividend), on the
near-month expiry (the next one when fewer than ``MIN_DTE`` days remain):

* **S1, the call-put IV spread** (Cremers & Weinbaum, 2010): the at-the-money call's IV minus the put's, at the
  strike nearest spot with both traded that day.  Published: a higher spread earns more.
* **S2, the put skew** (Xing, Zhang & Zhao, 2010): the IV of the traded put nearest 0.95 x spot (within
  0.93-0.97) minus the at-the-money call's.  Published: a steeper skew earns less.

Quintiles each month, equal weight, bought at the next session's open and held to the open after the next
month-end (the last period, formed in November 2025, ends at the 31 Dec 2025 close: 2026 is sealed), on the
engine's dividend-adjusted prices.  Long-short in the published direction, 25 bp a side on each month's turnover
in both legs.  Gate: Newey-West t (3 lags) of the net monthly long-short >= 2.24 (t >= 2, Bonferroni over the two
signals) and a positive mean in both 2013-19 and 2020-25.

    python -m nse_engine.option_signals run        # 2013-25, both signals, the gate; results in data/nse_engine/ivx1/

The 2026 holdout is opened once, and only for a signal that passes (plan 5ap).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WINDOW = ("2013-01-01", "2025-12-31")
HOLDOUT = ("2026-01-01", "2026-09-30")
MIN_DTE = 7
PUT_WING, PUT_BAND = 0.95, (0.93, 0.97)
COST_BPS = 25.0
T_GATE = 2.24
HAC_LAGS = 3
HALVES = (("2013-01-01", "2019-12-31"), ("2020-01-01", "2025-12-31"))
EQUITY_STORE = "data/nse_engine/store"
FO_STORE = "data/nse_engine/fo_store"
RUNS_DIR = "data/nse_engine/runs_ivx1"
OUT_DIR = "data/nse_engine/ivx1"


@dataclass(frozen=True)
class SignalConfig:
    """One pre-registered signal (plan 5ap); the window is a run's, not the configuration's."""

    name: str
    signal: str                      # "iv_spread" | "skew"
    direction: int                   # +1: the highest quintile earns more; -1: the lowest does
    quantiles: int = 5
    cost_bps: float = COST_BPS
    min_dte: int = MIN_DTE
    put_wing: float = PUT_WING
    start: str = WINDOW[0]
    end: str = WINDOW[1]
    runs_dir: str = RUNS_DIR

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def config_hash(self) -> str:
        payload = {k: v for k, v in asdict(self).items() if k not in ("name", "start", "end", "runs_dir")}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


SIGNALS = {"S1": SignalConfig("IVX1-S1 call-put IV spread", "iv_spread", +1),
           "S2": SignalConfig("IVX1-S2 put skew", "skew", -1)}


# ── the signals on one month-end ─────────────────────────────────────────

def stock_signals(day: pd.DataFrame, spot: float, session: pd.Timestamp, rate: float,
                  min_dte: int = MIN_DTE, put_wing: float = PUT_WING) -> Dict[str, float]:
    """S1 and S2 for one stock on one session from its traded contracts (``contracts`` > 0, ``close`` > 0); NaN for
    a signal whose contracts did not trade."""
    from kite_connect.options.theory import CALL, PUT, implied_volatility

    out = {"iv_spread": np.nan, "skew": np.nan}
    t = day[(day["contracts"] > 0) & (day["close"] > 0)]
    if t.empty or not np.isfinite(spot) or spot <= 0:
        return out
    dtes = sorted({(pd.Timestamp(e) - session).days: pd.Timestamp(e) for e in day["expiry"].unique()}.items())
    near = [(d, e) for d, e in dtes if d > 0]
    if not near:
        return out
    dte, expiry = near[0] if near[0][0] >= min_dte else (near[1] if len(near) > 1 else (None, None))
    if expiry is None:
        return out
    e = t[pd.to_datetime(t["expiry"]) == expiry]
    calls = e[e["option_type"] == CALL].set_index("strike")["close"]
    puts = e[e["option_type"] == PUT].set_index("strike")["close"]

    def iv(kind: str, price: float, strike: float) -> Optional[float]:
        return implied_volatility(kind, float(price), spot, float(strike), dte, rate)

    both = sorted(set(calls.index) & set(puts.index), key=lambda k: abs(k - spot))
    if both:
        k = both[0]
        c, p = iv(CALL, calls[k], k), iv(PUT, puts[k], k)
        if c and p:
            out["iv_spread"] = c - p
    if len(calls):
        k_atm = min(calls.index, key=lambda k: abs(k - spot))
        c_atm = iv(CALL, calls[k_atm], k_atm)
        band = [k for k in puts.index if PUT_BAND[0] * spot <= k <= PUT_BAND[1] * spot]
        if c_atm and band:
            k_put = min(band, key=lambda k: abs(k - put_wing * spot))
            p_otm = iv(PUT, puts[k_put], k_put)
            if p_otm:
                out["skew"] = p_otm - c_atm
    return out


def month_ends(dates: pd.DatetimeIndex) -> List[pd.Timestamp]:
    s = pd.Series(dates, index=dates)
    return list(s.groupby([dates.year, dates.month]).max())


def build_panel(window: Sequence[str] = WINDOW, equity_store: str = EQUITY_STORE, fo_store_dir: str = FO_STORE) -> pd.DataFrame:
    """Every (month-end, stock): S1, S2 and the next holding period's return (open to the open after the next
    month-end; the window's last period ends at its last close).  Prices are read only inside ``window``."""
    from kite_connect.options.options_config import OptionsConfig
    from nse_engine.data.panel import load_change_table, load_market_data
    from nse_engine.data.reference import resolve_symbols

    rate = OptionsConfig().market.risk_free_rate
    lo, hi = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    years = range(lo.year, hi.year + 1)
    frames = [pd.read_parquet(Path(fo_store_dir) / "stock_options" / f"{y}.parquet") for y in years
              if (Path(fo_store_dir) / "stock_options" / f"{y}.parquet").exists()]
    opts = pd.concat(frames, ignore_index=True)
    opts["date"] = pd.to_datetime(opts["date"])
    opts = opts[(opts["date"] >= lo) & (opts["date"] <= hi)]
    # the equity store files a company under its latest symbol (TATAMOTORS -> TMPV): map each option row to it, so
    # a renamed stock is not silently dropped
    opts = opts.assign(symbol=resolve_symbols(opts["symbol"], opts["date"].astype("datetime64[ns]"),
                                                  load_change_table(equity_store)).to_numpy())
    symbols = sorted(opts["symbol"].unique())
    data = load_market_data(equity_store, lo - pd.Timedelta(days=40), hi, symbols=symbols, min_median_value_inr=0.0,
                            adjust_dividends=True)
    dates = data.dates[(data.dates >= lo) & (data.dates <= hi)]
    ends = month_ends(dates)
    printed = data.close_unadj if data.close_unadj is not None else data.close
    rows = []
    for i, m in enumerate(ends[:-1]):
        nxt = ends[i + 1]
        entry = dates[dates > m][0]
        after = dates[dates > nxt]
        day = opts[opts["date"] == m]
        for sym, g in day.groupby("symbol"):
            if sym not in data.open.columns:
                continue
            spot = float(printed.at[m, sym]) if m in printed.index else np.nan
            sig = stock_signals(g, spot, m, rate)
            if not (np.isfinite(sig["iv_spread"]) or np.isfinite(sig["skew"])):
                continue
            p0 = data.open.at[entry, sym]
            p1 = data.open.at[after[0], sym] if len(after) else data.close.at[dates[-1], sym]
            ret = float(p1 / p0 - 1.0) if np.isfinite(p0) and np.isfinite(p1) and p0 > 0 else np.nan
            rows.append({"month_end": m, "symbol": sym, **sig, "ret": ret})
        if (i + 1) % 24 == 0:
            logger.info("IVX1 panel: %s, %d month-ends done", m.date(), i + 1)
    nifty = data.index_close["NIFTY50"].reindex(ends).pct_change().shift(-1) if "NIFTY50" in data.index_close else None
    panel = pd.DataFrame(rows)
    panel.attrs["nifty_next_month"] = nifty
    panel.attrs["data_hash"] = getattr(data, "data_hash", "")
    return panel


# ── portfolios and the gate ──────────────────────────────────────────────

def long_short(panel: pd.DataFrame, cfg: SignalConfig) -> pd.DataFrame:
    """Per month-end: the quintile means, the gross long-short in the published direction, its turnover cost and
    the net return; the stocks with the signal."""
    rows, prev = [], {"long": {}, "short": {}}
    for m, g in panel.dropna(subset=[cfg.signal, "ret"]).groupby("month_end"):
        if len(g) < cfg.quantiles * 2:
            continue
        q = pd.qcut(g[cfg.signal].rank(method="first"), cfg.quantiles, labels=False) + 1
        means = g.groupby(q)["ret"].mean()
        top, bottom = (cfg.quantiles, 1) if cfg.direction > 0 else (1, cfg.quantiles)
        legs = {"long": set(g.loc[q == top, "symbol"]), "short": set(g.loc[q == bottom, "symbol"])}
        turnover = 0.0
        for leg, names in legs.items():
            w_new = {s: 1.0 / len(names) for s in names}
            keys = set(w_new) | set(prev[leg])
            turnover += sum(abs(w_new.get(k, 0.0) - prev[leg].get(k, 0.0)) for k in keys)
            prev[leg] = w_new
        gross = float(means[top] - means[bottom])
        cost = cfg.cost_bps / 1e4 * turnover
        rows.append({"month_end": m, "n": len(g), "gross": gross, "turnover": turnover, "cost": cost,
                     "net": gross - cost, **{f"q{k}": float(means.get(k, np.nan)) for k in range(1, cfg.quantiles + 1)}})
    return pd.DataFrame(rows).set_index("month_end")


def judge(ls: pd.DataFrame, cfg: SignalConfig, nifty_next: Optional[pd.Series] = None) -> Dict[str, Any]:
    """The pre-registered gate and the reported statistics for one signal."""
    from nse_engine.validation.diagnostics import alpha_beta
    from nse_engine.validation.significance import mean_test

    net = ls["net"]
    test = mean_test(net, rf_annual=0.0, hac_lags=HAC_LAGS, periods=12)
    halves = {f"{a[:4]}-{b[2:4]}": float(net[a:b].mean()) for a, b in HALVES}
    qmeans = [float(ls[f"q{k}"].mean()) for k in range(1, cfg.quantiles + 1)]
    diffs = np.diff(qmeans) * cfg.direction
    out = {"signal": cfg.name, "config_hash": cfg.config_hash(), "months": int(len(net)),
           "stocks_per_month_median": float(ls["n"].median()),
           "gross_mean_monthly": float(ls["gross"].mean()), "net_mean_monthly": float(net.mean()),
           "net_mean_annual": float(net.mean() * 12), "net_sharpe_annual": float(net.mean() / net.std(ddof=1) * np.sqrt(12)),
           "hit_rate": float((net > 0).mean()), "turnover_mean": float(ls["turnover"].mean()),
           "cost_mean_monthly": float(ls["cost"].mean()), "t": test["t"], "p_two_sided": test["p_two_sided"],
           "halves_net_mean_monthly": halves, "quintile_means_monthly": qmeans,
           "monotone": bool((diffs > 0).all())}
    if nifty_next is not None:
        try:
            ab = alpha_beta(net, nifty_next.reindex(net.index), rf_annual=0.0, hac_lags=HAC_LAGS, periods_per_year=12)
            out["alpha_vs_nifty"] = {k: ab[k] for k in ("alpha_annual", "alpha_t_hac", "beta", "n_obs")}
        except ValueError as exc:
            out["alpha_vs_nifty"] = {"error": str(exc)}
    out["pass"] = bool(np.isfinite(out["t"]) and out["t"] >= T_GATE and all(v > 0 for v in halves.values()))
    return out


class _Run:
    """What ``record_result`` needs: a configuration, the net monthly returns, metrics and the data hash."""

    def __init__(self, cfg: SignalConfig, returns: pd.Series, metrics: Dict[str, Any], data_hash: str):
        self.config, self.returns, self.metrics, self.data_hash = cfg, returns, metrics, data_hash
        self.run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_{cfg.config_hash()[:8]}"
        self.run_dir = None


def run(out_dir: str = OUT_DIR, record: bool = True) -> Dict[str, Any]:
    """Plan 5ap: both signals over 2013-25, the gate, the runs recorded in the family's registry."""
    from nse_engine.data import fo_store
    from nse_engine.validation.trials import record_result

    panel = build_panel()
    results: Dict[str, Any] = {"plan": "5ap", "window": list(WINDOW), "t_gate": T_GATE, "cost_bps": COST_BPS,
                               "panel_rows": int(len(panel)), "signals": {}}
    for key, cfg in SIGNALS.items():
        ls = long_short(panel, cfg)
        res = judge(ls, cfg, panel.attrs.get("nifty_next_month"))
        if record:
            data_hash = f"{panel.attrs.get('data_hash', '')}+{fo_store.data_hash(FO_STORE)}"
            run_dir = record_result(_Run(cfg, ls["net"], res, data_hash), tag=cfg.name, runs_dir=RUNS_DIR,
                                    window=WINDOW, extra={"family": "option_signals", "frequency": "monthly"})
            ls.to_csv(Path(run_dir) / "long_short.csv")
            res["run_dir"] = run_dir
        results["signals"][key] = res
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "ivx1_results.json"
    path.write_text(json.dumps(results, indent=1, default=str))
    Path(RUNS_DIR).mkdir(parents=True, exist_ok=True)
    shutil.copy(path, Path(RUNS_DIR) / path.name)          # the family's registry is in the backup's state piece
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nse_engine.option_signals", description=__doc__.strip().splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run", help="plan 5ap: 2013-25, both signals, the gate")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.cmd == "run":
        print(json.dumps(run(), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
