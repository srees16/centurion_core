"""
Diagnostics: Aronson benchmark detrending, date-aligned alpha/beta with HAC
errors, execution-lag sensitivity and a JSON report.

Aronson detrending (Evidence-Based Technical Analysis, ch. 1) removes the
part of a long-biased rule's return that comes from simply being exposed to
a rising market: ``r_detrended[t] = r[t] - mean(benchmark) * exposure[t]``.
It does NOT subtract the strategy's own mean (that makes the Sharpe ~0 by
construction, the legacy bug).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .dsr import daily_rf, excess_sharpe, performance_summary
from .trials import to_jsonable

logger = logging.getLogger(__name__)


def _as_dated(s: Any, name: str) -> pd.Series:
    if not isinstance(s, pd.Series):
        raise TypeError(f"{name} must be a pandas Series indexed by date")
    out = s.astype("float64").copy()
    out.index = pd.DatetimeIndex(out.index)
    return out[~out.index.duplicated(keep="last")].sort_index()


def aronson_detrended_sharpe(strategy_returns: pd.Series, exposure: pd.Series,
                             benchmark_returns: pd.Series, rf_annual: float = 0.0,
                             ) -> Dict[str, float]:
    """Excess Sharpe after removing average benchmark drift x exposure.

    ``exposure[t]`` is the gross long exposure that earned ``r[t]`` (for a
    ``BacktestResult`` use ``weights.sum(axis=1).shift(1)``: end-of-day
    weights of ``t-1``).  All inputs are aligned by date (inner join); the
    benchmark mean is taken over the aligned sample.
    """
    df = pd.concat([_as_dated(strategy_returns, "strategy_returns").rename("r"),
                    _as_dated(exposure, "exposure").rename("x"),
                    _as_dated(benchmark_returns, "benchmark_returns").rename("b")],
                   axis=1, join="inner").dropna()
    if len(df) < 3:
        raise ValueError("fewer than 3 aligned observations for detrending")
    drift = float(df["b"].mean())
    detrended = df["r"] - drift * df["x"]
    return {
        "detrended_sharpe": excess_sharpe(detrended, rf_annual),
        "raw_excess_sharpe": excess_sharpe(df["r"], rf_annual),
        "benchmark_mean_daily": drift,
        "mean_exposure": float(df["x"].mean()),
        "drift_removed_annual": float(drift * df["x"].mean() * 252),
        "n_obs": int(len(df)),
    }


def newey_west_cov(x: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    """HAC (Newey-West, Bartlett kernel) covariance of OLS coefficients."""
    n = x.shape[0]
    xtx_inv = np.linalg.inv(x.T @ x)
    u = x * resid[:, None]
    s = u.T @ u
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        g = u[lag:].T @ u[:-lag]
        s += w * (g + g.T)
    return xtx_inv @ s @ xtx_inv * (n / max(n - x.shape[1], 1))


def alpha_beta(returns: pd.Series, benchmark_returns: pd.Series, rf_annual: float = 0.0,
               hac_lags: Optional[int] = None, periods_per_year: int = 252) -> Dict[str, float]:
    """OLS of strategy excess returns on benchmark excess returns, aligned BY DATE.

    ``hac_lags`` defaults to ``floor(4 (T/100)^(2/9))``.  The alpha t-stat and
    p-value use Newey-West standard errors.
    """
    df = pd.concat([_as_dated(returns, "returns").rename("r"),
                    _as_dated(benchmark_returns, "benchmark_returns").rename("b")],
                   axis=1, join="inner").dropna()
    n = len(df)
    if n < 10:
        raise ValueError(f"only {n} date-aligned observations for alpha/beta")
    rf = daily_rf(rf_annual, periods_per_year)
    y = df["r"].to_numpy() - rf
    xb = df["b"].to_numpy() - rf
    x = np.column_stack([np.ones(n), xb])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    lags = int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0))) if hac_lags is None else int(hac_lags)
    cov = newey_west_cov(x, resid, lags)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    t_alpha = float(coef[0] / se[0]) if se[0] > 0 else float("nan")
    t_beta = float(coef[1] / se[1]) if se[1] > 0 else float("nan")
    ss_tot = float(((y - y.mean()) ** 2).sum())
    resid_sd = float(resid.std(ddof=2))
    return {
        "alpha_daily": float(coef[0]),
        "alpha_annual": float(coef[0] * periods_per_year),
        "beta": float(coef[1]),
        "alpha_t_hac": t_alpha,
        "alpha_p_hac": float(2 * sp_stats.t.sf(abs(t_alpha), df=n - 2)) if np.isfinite(t_alpha) else float("nan"),
        "beta_t_hac": t_beta,
        "r_squared": float(1 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan"),
        "residual_vol_annual": resid_sd * math.sqrt(periods_per_year),
        "information_ratio": float(coef[0] / resid_sd * math.sqrt(periods_per_year)) if resid_sd > 0 else float("nan"),
        "hac_lags": lags,
        "n_obs": n,
        "start": str(df.index[0].date()),
        "end": str(df.index[-1].date()),
    }


def lag_sensitivity(data: Any, config: Any, lags: Iterable[int] = (0, 1, 2, 3),
                    backtest_fn: Optional[Callable[..., Any]] = None, record: bool = True,
                    ) -> Dict[str, Any]:
    """Re-run the backtest with execution delayed by ``lag_days`` extra sessions.

    Uses the engine's ``lag_days`` argument (a true delay of every fill), not
    a coarser decision grid.  Runs are tagged ``lag-<n>``.
    """
    if backtest_fn is None:
        from nse_engine.engine import run_backtest as backtest_fn  # lazy
    rf = float(getattr(config, "risk_free_annual", 0.0))
    rows: Dict[int, Dict[str, Any]] = {}
    for lag in lags:
        res = backtest_fn(data, config, record=record, tag=f"lag-{int(lag)}", lag_days=int(lag))
        perf = performance_summary(pd.Series(res.returns, dtype="float64"), rf)
        perf["run_id"] = getattr(res, "run_id", "")
        rows[int(lag)] = perf
    base = rows.get(0, {}).get("excess_sharpe", float("nan"))
    decay = {lag: (r["excess_sharpe"] / base if base and np.isfinite(base) else float("nan"))
             for lag, r in rows.items()}
    return {"lags": rows, "sharpe_ratio_to_lag0": decay}


def full_report(result: Any, data: Any, benchmarks: Optional[Dict[str, pd.Series]] = None,
                trials_matrix: Optional[pd.DataFrame] = None, n_trials: Optional[float] = None,
                rf_annual: Optional[float] = None, margin: float = 0.3,
                pbo_splits: int = 16) -> Dict[str, Any]:
    """JSON-serialisable validation summary of one ``BacktestResult``.

    Includes performance, DSR (with ``trials_matrix``/``n_trials`` when
    given), CSCV PBO (when ``trials_matrix`` has >= 2 columns), benchmark
    gate, alpha/beta and Aronson detrending against NIFTY50.
    """
    from .benchmarks import benchmark_gate
    from .dsr import deflated_sharpe
    from .pbo import cscv_pbo

    config = result.config
    rf = float(getattr(config, "risk_free_annual", 0.0)) if rf_annual is None else float(rf_annual)
    r = pd.Series(result.returns, dtype="float64").dropna()
    r.index = pd.DatetimeIndex(r.index)
    report: Dict[str, Any] = {
        "run_id": getattr(result, "run_id", ""),
        "config_hash": config.config_hash() if hasattr(config, "config_hash") else None,
        "data_hash": getattr(result, "data_hash", ""),
        "start": str(r.index[0].date()) if len(r) else None,
        "end": str(r.index[-1].date()) if len(r) else None,
        "rf_annual": rf,
        "performance": performance_summary(r, rf),
        "engine_metrics": getattr(result, "metrics", {}),
    }

    def attempt(key: str, fn: Callable[[], Any]) -> None:
        try:
            report[key] = fn()
        except Exception as exc:  # report must not die on one diagnostic
            logger.warning("full_report: %s failed: %s", key, exc)
            report[key] = {"error": str(exc)}

    attempt("dsr", lambda: deflated_sharpe(r, trials_matrix=trials_matrix, n_trials=n_trials,
                                           rf_annual=rf))
    if trials_matrix is not None and trials_matrix.shape[1] >= 2:
        def _pbo():
            out = cscv_pbo(trials_matrix, n_splits=pbo_splits)
            logits = np.asarray(out.pop("logits"))  # omit the bulk array
            out["logit_quantiles"] = dict(zip(("p05", "p25", "p50", "p75", "p95"),
                                              np.quantile(logits, [.05, .25, .5, .75, .95])))
            return out
        attempt("pbo", _pbo)
    if benchmarks:
        attempt("benchmark_gate", lambda: benchmark_gate(r, benchmarks, margin=margin, rf_annual=rf))
    bench = None
    bench_key = next((k for k in ("nifty50_tri", "nifty50_price_index") if benchmarks and k in benchmarks), None)
    if bench_key:
        bench = benchmarks[bench_key]
    elif data is not None and "NIFTY50" in getattr(data, "index_close", pd.DataFrame()).columns:
        bench = data.index_close["NIFTY50"].astype("float64").pct_change(fill_method=None)
    if bench is not None:
        attempt("alpha_beta_nifty50", lambda: alpha_beta(r, bench, rf))
        weights = getattr(result, "weights", None)
        if isinstance(weights, pd.DataFrame) and not weights.empty:
            exposure = weights.sum(axis=1).shift(1)
            attempt("aronson_detrended", lambda: aronson_detrended_sharpe(r, exposure, bench, rf))
    return to_jsonable(report)
