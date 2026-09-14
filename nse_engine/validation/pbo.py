"""
Probability of Backtest Overfitting via CSCV (Bailey, Borwein, López de Prado
& Zhu, 2017, "The Probability of Backtest Overfitting").

Input is a date x trial matrix of daily returns covering EVERY configuration
that was evaluated (``TrialRegistry.returns_matrix()``) -- not shares of one
portfolio's P&L.

Algorithm
---------
1. Drop rows with any NaN; drop ``T mod S`` rows at the START so the matrix
   splits into ``S`` contiguous equal blocks.
2. For every combination of ``S/2`` blocks as the in-sample (IS) set -- all
   ``C(S, S/2)`` of them, or a random subsample of ``max_combinations`` --
   the complement is out-of-sample (OOS).
3. Pick the trial with the best IS metric; compute its OOS rank among all
   ``N`` trials (1 = worst, ties averaged), ``omega = rank / (N + 1)`` and
   ``lambda = ln(omega / (1 - omega))``.
4. ``PBO = share of lambda <= 0``.

Also reported: performance degradation (OLS slope of the IS-best trial's OOS
metric on its IS metric across combinations) and probability of loss (share
of combinations where the IS-best trial's OOS metric is negative).

Metrics are computed from per-block sufficient statistics, so each
combination costs O(N); combinations are processed in chunks to bound memory.
"""

from __future__ import annotations

import logging
import math
from itertools import combinations
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_METRICS = ("sharpe", "mean", "sortino")


def _combination_masks(n_splits: int, max_combinations: Optional[int],
                       random_state: int) -> np.ndarray:
    """Boolean (C x S) array; True marks an in-sample block."""
    half = n_splits // 2
    total = math.comb(n_splits, half)
    if max_combinations is None or max_combinations >= total:
        combos = np.array(list(combinations(range(n_splits), half)), dtype=np.int64)
    else:
        rng = np.random.default_rng(random_state)
        seen = set()
        rows: List[tuple] = []
        while len(rows) < max_combinations:
            c = tuple(sorted(rng.choice(n_splits, size=half, replace=False).tolist()))
            if c not in seen:
                seen.add(c)
                rows.append(c)
        combos = np.array(rows, dtype=np.int64)
    masks = np.zeros((combos.shape[0], n_splits), dtype=bool)
    masks[np.arange(combos.shape[0])[:, None], combos] = True
    return masks


def _metric_from_stats(s1: np.ndarray, s2: np.ndarray, sd2: np.ndarray, n: np.ndarray,
                       metric: str) -> np.ndarray:
    """Metric per (combination, trial) from summed block statistics."""
    mean = s1 / n
    if metric == "mean":
        return mean
    if metric == "sharpe":
        var = (s2 - n * mean ** 2) / (n - 1.0)
        sd = np.sqrt(np.clip(var, 0.0, None))
    else:  # sortino: downside deviation around 0
        sd = np.sqrt(np.clip(sd2 / n, 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(sd > 1e-15, mean / np.where(sd > 1e-15, sd, 1.0), 0.0)
    return out


def cscv_pbo(returns_matrix: pd.DataFrame, n_splits: int = 16, metric: str = "sharpe",
             max_combinations: Optional[int] = None, random_state: int = 0,
             chunk_cells: int = 4_000_000) -> Dict[str, object]:
    """CSCV probability of backtest overfitting.

    Parameters
    ----------
    returns_matrix : date x trial daily returns (rows with NaN are dropped).
    n_splits : number of contiguous blocks S (even, >= 2).
    metric : ``"sharpe"`` (default), ``"mean"`` or ``"sortino"``.  Per-period,
        unannualised; ranking is scale-free so annualisation is irrelevant.
    max_combinations : optional random subsample size of the C(S, S/2) splits.
    random_state : seed for the subsample.

    Returns
    -------
    dict: pbo, logits (ndarray), median_logit, n_combinations, n_trials, n_obs,
    n_dropped_rows, degradation_slope, degradation_intercept, prob_oos_loss,
    mean_is_metric, mean_oos_metric, metric, n_splits.
    """
    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {_METRICS}, got {metric!r}")
    if n_splits < 2 or n_splits % 2:
        raise ValueError(f"n_splits must be an even integer >= 2, got {n_splits}")
    if not isinstance(returns_matrix, pd.DataFrame):
        returns_matrix = pd.DataFrame(returns_matrix)
    m = returns_matrix.dropna(axis=0, how="any")
    if len(m) < len(returns_matrix):
        logger.warning("cscv_pbo: dropped %d rows containing NaN", len(returns_matrix) - len(m))
    n_trials = int(m.shape[1])
    if n_trials < 2:
        raise ValueError(f"CSCV needs at least 2 trials, got {n_trials}")
    if n_trials < 10:
        logger.warning("cscv_pbo: only %d trials; PBO is coarse and unreliable with N < 10",
                       n_trials)
    t_all = int(m.shape[0])
    block = t_all // n_splits
    if block < 2:
        raise ValueError(f"need at least {2 * n_splits} rows for {n_splits} splits, got {t_all}")
    drop = t_all - block * n_splits
    x = m.to_numpy(dtype="float64")[drop:]
    xb = x.reshape(n_splits, block, n_trials)
    b1 = xb.sum(axis=1)                                  # S x N
    b2 = (xb ** 2).sum(axis=1)
    bd = (np.minimum(xb, 0.0) ** 2).sum(axis=1)

    masks = _combination_masks(n_splits, max_combinations, random_state)
    n_comb = masks.shape[0]
    half_n = float(block * (n_splits // 2))
    chunk = max(1, int(chunk_cells // max(n_trials, 1)))

    logits = np.empty(n_comb)
    is_best_metric = np.empty(n_comb)
    oos_best_metric = np.empty(n_comb)
    for lo in range(0, n_comb, chunk):
        mk = masks[lo:lo + chunk].astype("float64")       # c x S
        ok = 1.0 - mk
        is_m = _metric_from_stats(mk @ b1, mk @ b2, mk @ bd, half_n, metric)
        oos_m = _metric_from_stats(ok @ b1, ok @ b2, ok @ bd, half_n, metric)
        best = np.argmax(is_m, axis=1)
        rows = np.arange(best.size)
        oos_b = oos_m[rows, best]
        below = (oos_m < oos_b[:, None]).sum(axis=1)
        equal = (oos_m == oos_b[:, None]).sum(axis=1)     # includes itself
        rank = below + (equal + 1) / 2.0                  # average rank, 1..N
        omega = rank / (n_trials + 1.0)
        logits[lo:lo + chunk] = np.log(omega / (1.0 - omega))
        is_best_metric[lo:lo + chunk] = is_m[rows, best]
        oos_best_metric[lo:lo + chunk] = oos_b

    if np.ptp(is_best_metric) > 1e-15:
        slope, intercept = np.polyfit(is_best_metric, oos_best_metric, 1)
    else:
        slope, intercept = float("nan"), float(np.mean(oos_best_metric))
    pbo = float(np.mean(logits <= 0.0))
    logger.info("CSCV PBO=%.3f over %d combinations, N=%d trials, T=%d obs",
                pbo, n_comb, n_trials, block * n_splits)
    return {
        "pbo": pbo,
        "logits": logits,
        "median_logit": float(np.median(logits)),
        "n_combinations": int(n_comb),
        "n_trials": n_trials,
        "n_obs": int(block * n_splits),
        "n_dropped_rows": int(drop),
        "degradation_slope": float(slope),
        "degradation_intercept": float(intercept),
        "prob_oos_loss": float(np.mean(oos_best_metric < 0.0)),
        "mean_is_metric": float(np.mean(is_best_metric)),
        "mean_oos_metric": float(np.mean(oos_best_metric)),
        "metric": metric,
        "n_splits": int(n_splits),
    }
