"""
Walk-forward re-fitting over a parameter grid.

For each fold the whole grid is evaluated on the TRAIN window only, the
best configuration by annualised excess Sharpe is chosen, and that single
configuration is run on the following TEST window.  OOS test returns are
stitched into one series.  Nothing chosen in fold ``k`` ever sees data from
its own test window.

Assumptions the engine (``nse_engine.engine.run_backtest``) must satisfy
-----------------------------------------------------------------------
* ``run_backtest(data, config, record=..., tag=..., lag_days=...)``.
* ``config.start`` / ``config.end`` (``"YYYY-MM-DD"``) bound the TRADING
  window: the first decision is made at the close of the first session
  ``>= start`` and ``result.returns`` covers sessions in ``[start, end]``
  only.  Data BEFORE ``start`` is used for indicator warm-up, which is why
  the full ``MarketData`` is passed rather than a sliced panel.  Data AFTER
  ``end`` must not be read (point-in-time rule).
* Each window starts flat (in cash).  The stitched OOS series therefore
  includes the cost of re-entering the book at each fold boundary.
* With ``record=True`` the engine writes a run directory; if a backtest
  function returns a result without ``run_dir`` this module records it via
  ``trials.record_result`` so every train evaluation reaches the registry.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .dsr import excess_sharpe, performance_summary
from .trials import record_result

logger = logging.getLogger(__name__)

BacktestFn = Callable[..., Any]
MetricArg = Union[str, Callable[[pd.Series, float], float]]


def _default_backtest_fn() -> BacktestFn:
    from nse_engine.engine import run_backtest  # lazy: built in parallel

    return run_backtest


def expand_grid(param_grid: Optional[Dict[str, List[Any]]]) -> List[Dict[str, Any]]:
    """Cartesian product of a ``{dotted.key: [values]}`` grid (``[{}]`` if empty)."""
    if not param_grid:
        return [{}]
    keys = list(param_grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(param_grid[k] for k in keys))]


def generate_folds(dates: pd.DatetimeIndex, train_years: float = 4, test_months: int = 12,
                   anchored: bool = True, start: Optional[str] = None,
                   end: Optional[str] = None, min_test_days: int = 21) -> List[Dict[str, pd.Timestamp]]:
    """Train/test folds on the trading calendar.

    The first train window begins at the first session ``>= start`` (default
    ``dates[0]``) and spans ``train_years``; test windows of ``test_months``
    follow back to back until ``end``.  ``anchored=True`` keeps the train
    start fixed (expanding); otherwise the train window rolls.  A final
    partial test window is kept if it has at least ``min_test_days``
    sessions.  All boundaries are actual sessions in ``dates``.
    """
    d = pd.DatetimeIndex(dates).sort_values()
    if start is not None:
        d = d[d >= pd.Timestamp(start)]
    if end is not None:
        d = d[d <= pd.Timestamp(end)]
    if len(d) == 0:
        return []
    origin = d[0]
    months = int(round(train_years * 12))
    test_start_bound = origin + pd.DateOffset(months=months)
    folds = []
    while True:
        test_idx = d[d >= test_start_bound]
        if len(test_idx) == 0:
            break
        test_end_bound = test_start_bound + pd.DateOffset(months=test_months)
        test_days = test_idx[test_idx < test_end_bound]
        if len(test_days) < min_test_days:
            break
        train_lo = origin if anchored else test_start_bound - pd.DateOffset(months=months)
        train_days = d[(d >= train_lo) & (d < test_days[0])]
        if len(train_days) == 0:
            break
        folds.append({"train_start": train_days[0], "train_end": train_days[-1],
                      "test_start": test_days[0], "test_end": test_days[-1]})
        test_start_bound = test_end_bound
    return folds


def _metric_value(metric: MetricArg, returns: pd.Series, rf_annual: float) -> float:
    if callable(metric):
        return float(metric(returns, rf_annual))
    if metric == "sharpe":
        return excess_sharpe(returns, rf_annual)
    raise ValueError(f"unsupported metric {metric!r}")


def _window_returns(result: Any, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.Series:
    r = pd.Series(result.returns, dtype="float64").dropna()
    r.index = pd.DatetimeIndex(r.index)
    return r[(r.index >= lo) & (r.index <= hi)]


def _run(backtest_fn: BacktestFn, data: Any, config: Any, tag: str, record: bool,
         lo: pd.Timestamp, hi: pd.Timestamp) -> Any:
    result = backtest_fn(data, config, record=record, tag=tag)
    if record and not getattr(result, "run_dir", None):
        record_result(result, tag=tag, window=(lo.date(), hi.date()))
    return result


def run_walk_forward(data: Any, base_config: Any, param_grid: Optional[Dict[str, List[Any]]],
                     train_years: float = 4, test_months: int = 12, anchored: bool = True,
                     backtest_fn: Optional[BacktestFn] = None, metric: MetricArg = "sharpe",
                     record_trials: bool = True, min_test_days: int = 21) -> Dict[str, Any]:
    """Anchored (or rolling) walk-forward parameter selection.

    Folds are generated on ``data.dates`` within ``[base_config.start,
    base_config.end]`` -- keep ``end`` before any holdout window.  Grid keys
    are dotted ``EngineConfig.replace`` paths.  Selection uses the
    annualised excess Sharpe over ``base_config.risk_free_annual`` (or a
    callable ``metric(returns, rf_annual)``).

    Returns
    -------
    dict with ``oos_returns`` (stitched Series), ``folds`` (per-fold chosen
    params, IS/OOS metrics, grid scores, run ids) and ``summary``
    (oos_sharpe, mean_is_sharpe, mean_oos_sharpe, sharpe_degradation,
    oos_is_ratio, n_folds, n_grid_points, n_backtests).
    """
    backtest_fn = backtest_fn or _default_backtest_fn()
    rf = float(getattr(base_config, "risk_free_annual", 0.0))
    grid = expand_grid(param_grid)
    folds = generate_folds(data.dates, train_years, test_months, anchored,
                           start=base_config.start, end=base_config.end,
                           min_test_days=min_test_days)
    if not folds:
        raise ValueError("no walk-forward folds fit inside the data/config window")
    logger.info("Walk-forward: %d folds x %d grid points (anchored=%s)", len(folds), len(grid), anchored)

    fold_rows: List[Dict[str, Any]] = []
    oos_parts: List[pd.Series] = []
    n_backtests = 0
    for k, f in enumerate(folds):
        tr_lo, tr_hi, te_lo, te_hi = f["train_start"], f["train_end"], f["test_start"], f["test_end"]
        scores = []
        for params in grid:
            cfg = base_config.replace(**params, start=str(tr_lo.date()), end=str(tr_hi.date()))
            res = _run(backtest_fn, data, cfg, "wfo-train", record_trials, tr_lo, tr_hi)
            n_backtests += 1
            val = _metric_value(metric, _window_returns(res, tr_lo, tr_hi), rf)
            scores.append({"params": params, "is_metric": val,
                           "run_id": getattr(res, "run_id", "")})
        vals = np.array([s["is_metric"] for s in scores], dtype="float64")
        vals = np.where(np.isfinite(vals), vals, -np.inf)
        best_i = int(np.argmax(vals))
        best = scores[best_i]
        cfg_test = base_config.replace(**best["params"], start=str(te_lo.date()), end=str(te_hi.date()))
        res_test = _run(backtest_fn, data, cfg_test, "wfo-test", record_trials, te_lo, te_hi)
        n_backtests += 1
        oos = _window_returns(res_test, te_lo, te_hi)
        oos_val = _metric_value(metric, oos, rf)
        oos_parts.append(oos)
        fold_rows.append({
            "fold": k,
            "train_start": str(tr_lo.date()), "train_end": str(tr_hi.date()),
            "test_start": str(te_lo.date()), "test_end": str(te_hi.date()),
            "params": best["params"], "is_metric": best["is_metric"], "oos_metric": oos_val,
            "train_run_id": best["run_id"], "test_run_id": getattr(res_test, "run_id", ""),
            "grid_scores": scores,
        })
        logger.info("Fold %d: train %s..%s IS=%.3f -> test %s..%s OOS=%.3f params=%s", k,
                    tr_lo.date(), tr_hi.date(), best["is_metric"], te_lo.date(), te_hi.date(),
                    oos_val, best["params"])

    stitched = pd.concat(oos_parts).sort_index() if oos_parts else pd.Series(dtype="float64")
    if stitched.index.has_duplicates:
        raise RuntimeError("overlapping OOS test windows in walk-forward stitching")
    is_vals = np.array([r["is_metric"] for r in fold_rows], dtype="float64")
    oos_vals = np.array([r["oos_metric"] for r in fold_rows], dtype="float64")
    oos_sharpe = excess_sharpe(stitched, rf)
    mean_is = float(np.nanmean(is_vals)) if is_vals.size else float("nan")
    summary = {
        "oos_sharpe": oos_sharpe,
        "mean_is_sharpe": mean_is,
        "mean_oos_sharpe": float(np.nanmean(oos_vals)) if oos_vals.size else float("nan"),
        "sharpe_degradation": mean_is - oos_sharpe,
        "oos_is_ratio": oos_sharpe / mean_is if mean_is and np.isfinite(mean_is) else float("nan"),
        "n_folds": len(fold_rows),
        "n_grid_points": len(grid),
        "n_backtests": n_backtests,
        "anchored": anchored,
        "oos_performance": performance_summary(stitched, rf),
    }
    return {"oos_returns": stitched, "folds": fold_rows, "summary": summary}
