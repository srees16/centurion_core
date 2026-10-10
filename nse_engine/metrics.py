"""
Performance metrics for engine runs.

Conventions (the only ones used anywhere in the engine, its validation and
the paper book):

* Sharpe and Sortino: excess daily returns over ``rf_annual / 252``,
  annualised by sqrt(252); volatility by sqrt(252).
* CAGR: compounded over elapsed **calendar** time, ``days / 365.25`` between
  the first and last date.  Not ``sessions / 252``: NSE trades about 248
  sessions a year, so that convention understates elapsed time and
  overstates CAGR by roughly 0.4 points a year.  Turnover and cost drag are
  per calendar year too.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

TRADING_DAYS = 252.0
CAGR_CONVENTION = "calendar"   # reported by performance_summary so a reader knows which years were used


def years_elapsed(index: pd.Index) -> float:
    """Calendar years between the first and last date of ``index`` (nan if < 2 rows or no span)."""
    if len(index) < 2:
        return float("nan")
    days = (pd.Timestamp(index[-1]) - pd.Timestamp(index[0])).days
    return days / 365.25 if days > 0 else float("nan")


def cagr_from_growth(growth: float, years: float) -> float:
    """Compound annual growth rate of a total growth factor (end / start) over ``years``."""
    if not (np.isfinite(growth) and growth > 0 and np.isfinite(years) and years > 0):
        return float("nan")
    return float(growth ** (1.0 / years) - 1.0)


_years = years_elapsed   # backwards-compatible name


def max_drawdown(equity: pd.Series) -> float:
    """Most negative peak-to-trough decline (e.g. -0.25)."""
    e = equity.dropna().astype("float64")
    if e.empty:
        return float("nan")
    return float((e / e.cummax() - 1.0).min())


def drawdown_periods(equity: pd.Series) -> pd.DataFrame:
    """Every drawdown of a date-indexed equity curve, in date order (tracker LN-T25).

    Lean's DrawdownCollection with two fixes: an episode is keyed by a count
    of the days at the high-water mark, not by the peak's value, so drawdowns
    from equal peaks stay separate; and one still open at the end is kept,
    with no recovery date.  The first day back at the peak is the recovery.
    ``days`` are calendar days from the peak to the recovery (or the last
    date), ``sessions`` the days under water.
    """
    e = equity.dropna().astype("float64")
    groups = [g for _, g in e.groupby((e >= e.cummax()).cumsum())]
    rows = []
    for i, g in enumerate(groups):
        if len(g) < 2:
            continue
        recovery = groups[i + 1].index[0] if i + 1 < len(groups) else pd.NaT
        end = g.index[-1] if pd.isna(recovery) else recovery
        rows.append({"peak": g.index[0], "trough": g.idxmin(), "recovery": recovery,
                     "depth": float(g.min() / g.iloc[0] - 1.0), "days": int((end - g.index[0]).days),
                     "sessions": int(len(g) - 1)})
    return pd.DataFrame(rows, columns=["peak", "trough", "recovery", "depth", "days", "sessions"])


def round_trips(trades: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Round trips per symbol: from a flat position to flat again.

    PnL = sell proceeds - buy cost - all costs within the trip.
    """
    cols = ["symbol", "open_date", "close_date", "pnl_inr"]
    if trades is None or len(trades) == 0:
        return pd.DataFrame(columns=cols)
    t = trades.sort_values("date", kind="stable")
    qty: Dict[str, int] = {}
    pnl: Dict[str, float] = {}
    opened: Dict[str, pd.Timestamp] = {}
    rows = []
    for rec in t.itertuples(index=False):
        sym = rec.symbol
        q = int(rec.quantity)
        if q <= 0:
            continue
        sign = 1 if str(rec.side).upper() == "BUY" else -1
        if qty.get(sym, 0) == 0 and sign > 0:
            opened[sym] = rec.date
            pnl[sym] = 0.0
        pnl[sym] = pnl.get(sym, 0.0) - sign * float(rec.value_inr) - float(rec.cost_inr)
        qty[sym] = qty.get(sym, 0) + sign * q
        if qty[sym] <= 0 and sym in opened:
            rows.append((sym, opened.pop(sym), rec.date, pnl.pop(sym)))
            qty[sym] = 0
    return pd.DataFrame(rows, columns=cols)


def compute_metrics(
    returns: pd.Series,
    equity: pd.Series,
    trades: Optional[pd.DataFrame] = None,
    weights: Optional[pd.DataFrame] = None,
    rf_annual: float = 0.0,
    initial_capital: Optional[float] = None,
) -> Dict[str, float]:
    """Headline metrics for a daily return series and its INR equity curve."""
    r = returns.dropna().astype("float64")
    e = equity.dropna().astype("float64")
    n = int(len(r))
    out: Dict[str, float] = {"n_days": float(n)}
    years = _years(e.index if len(e) else r.index)
    start_val = float(initial_capital) if initial_capital else (float(e.iloc[0]) if len(e) else float("nan"))
    end_val = float(e.iloc[-1]) if len(e) else float("nan")
    out["cagr"] = cagr_from_growth(end_val / start_val, years) if len(e) and start_val > 0 else float("nan")
    out["total_return"] = float(end_val / start_val - 1.0) if start_val and np.isfinite(end_val) else float("nan")
    rf_d = rf_annual / TRADING_DAYS
    ex = r - rf_d
    vol_d = float(r.std(ddof=1)) if n > 1 else float("nan")
    out["ann_vol"] = vol_d * np.sqrt(TRADING_DAYS) if np.isfinite(vol_d) else float("nan")
    ex_sd = float(ex.std(ddof=1)) if n > 1 else float("nan")
    out["sharpe"] = float(ex.mean() / ex_sd * np.sqrt(TRADING_DAYS)) if n > 1 and ex_sd > 0 else float("nan")
    downside = np.sqrt(np.mean(np.minimum(ex.to_numpy(), 0.0) ** 2)) if n > 1 else float("nan")
    out["sortino"] = float(ex.mean() / downside * np.sqrt(TRADING_DAYS)) if n > 1 and downside > 0 else float("nan")
    out["max_drawdown"] = max_drawdown(e)
    mdd = out["max_drawdown"]
    out["calmar"] = float(out["cagr"] / abs(mdd)) if np.isfinite(mdd) and mdd < 0 and np.isfinite(out["cagr"]) else float("nan")
    out["skew"] = float(stats.skew(r.to_numpy())) if n > 2 else float("nan")
    # Excess (Fisher) kurtosis, 0 for a normal; validation.dsr's PSR and MinTRL take the raw value (3).
    out["kurtosis"] = float(stats.kurtosis(r.to_numpy())) if n > 3 else float("nan")

    if weights is not None and len(weights):
        w = weights.fillna(0.0)
        out["avg_gross"] = float(w.sum(axis=1).mean())
        out["avg_positions"] = float((w > 1e-9).sum(axis=1).mean())
    else:
        out["avg_gross"] = float("nan")
        out["avg_positions"] = float("nan")

    avg_eq = float(e.mean()) if len(e) else float("nan")
    if trades is not None and len(trades) and np.isfinite(years) and years > 0 and avg_eq > 0:
        traded = float(trades["value_inr"].abs().sum())
        costs = float(trades["cost_inr"].sum())
        out["annual_turnover"] = traded / 2.0 / avg_eq / years
        out["cost_drag"] = costs / avg_eq / years
        out["n_trades"] = float(len(trades))
    else:
        out["annual_turnover"] = 0.0
        out["cost_drag"] = 0.0
        out["n_trades"] = 0.0
    rt = round_trips(trades)
    out["n_round_trips"] = float(len(rt))
    out["hit_rate"] = float((rt["pnl_inr"] > 0).mean()) if len(rt) else float("nan")
    return out
