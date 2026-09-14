"""
Deflated Sharpe Ratio (Bailey & López de Prado, 2014) in DAILY units.

Everything in this module works on per-period (daily) Sharpe ratios.  The
classic misuse -- plugging an annualised Sharpe into a formula whose
``sqrt(T - 1)`` counts daily observations -- inflates the z-score by
``sqrt(252)``.  Annualised values are reported for readability only.

Effective number of trials
--------------------------
When a ``trials_matrix`` (date x trial daily returns) is given, trials are
clustered on their return correlation:

* distance ``d_ij = sqrt(0.5 * (1 - rho_ij))`` (a proper metric),
* hierarchical clustering with average linkage,
* the dendrogram is cut at height ``sqrt(0.5 * (1 - min_intra_corr))``
  (0.5 for the default ``min_intra_corr = 0.5``).  With average linkage a
  merge happens only when the AVERAGE pairwise distance between two clusters
  is below the cut, i.e. (to first order) when their average
  cross-correlation is at least ``min_intra_corr``.  This is a per-merge
  criterion; a global "mean intra-cluster rho >= 0.5" rule was rejected
  because a large tight cluster can absorb a small uncorrelated one while
  keeping the pair-weighted mean above 0.5.  The resulting mean intra-cluster
  correlation is reported as ``mean_intra_corr``.

The cluster count is the effective ``N``.  For very large registries
(more than ``max_cluster_trials`` columns) a random subsample is clustered,
the remaining trials are assigned to the most correlated cluster centroid
when that correlation is ``>= min_intra_corr``, and the unassigned rest is
clustered recursively -- so ``N`` is never silently under-counted by
subsampling.

The cross-trial variance ``V`` of daily Sharpe ratios comes from the
``trials_matrix`` when given (variance over all trials, ``ddof=1``);
otherwise from the Sharpe estimator variance
``(1 - g3*SR + (g4 - 1)/4 * SR^2) / (T - 1)``.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

EULER_GAMMA = 0.5772156649015329
PERIODS_PER_YEAR = 252


# ----------------------------------------------------------------------------
# Shared return helpers (used across nse_engine.validation)
# ----------------------------------------------------------------------------

def daily_rf(rf_annual: float, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Per-period risk-free rate ``rf_annual / periods_per_year``.

    Arithmetic, matching ``nse_engine.metrics`` and the engine's cash accrual.
    """
    return float(rf_annual) / float(periods_per_year)


def excess_returns(returns: Union[pd.Series, pd.DataFrame], rf_annual: float = 0.0,
                   periods_per_year: int = PERIODS_PER_YEAR):
    """Daily simple returns minus the daily risk-free rate."""
    return returns - daily_rf(rf_annual, periods_per_year)


def sharpe_daily(returns: Union[pd.Series, np.ndarray], rf_annual: float = 0.0,
                 periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Per-period excess Sharpe ``mean(excess) / std(excess, ddof=1)``.

    NaNs are dropped.  Returns ``nan`` with fewer than 2 observations and
    ``0.0`` for a constant series.
    """
    arr = np.asarray(returns, dtype="float64")
    arr = arr[np.isfinite(arr)] - daily_rf(rf_annual, periods_per_year)
    if arr.size < 2:
        return float("nan")
    sd = float(arr.std(ddof=1))
    if sd < 1e-15:
        return 0.0
    return float(arr.mean() / sd)


def excess_sharpe(returns: Union[pd.Series, np.ndarray], rf_annual: float = 0.0,
                  periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Annualised excess Sharpe: daily excess Sharpe x sqrt(periods_per_year)."""
    return sharpe_daily(returns, rf_annual, periods_per_year) * math.sqrt(periods_per_year)


def performance_summary(returns: pd.Series, rf_annual: float = 0.0,
                        periods_per_year: int = PERIODS_PER_YEAR) -> Dict[str, float]:
    """Small, dependency-free metrics dict for a daily return series."""
    r = pd.Series(returns, dtype="float64").dropna()
    n = int(r.size)
    if n == 0:
        return {"n_obs": 0, "excess_sharpe": float("nan"), "cagr": float("nan"),
                "vol_annual": float("nan"), "max_drawdown": float("nan"),
                "total_return": float("nan")}
    equity = (1.0 + r).cumprod()
    total = float(equity.iloc[-1] - 1.0)
    years = n / periods_per_year
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and equity.iloc[-1] > 0 else float("nan")
    dd = float((equity / equity.cummax() - 1.0).min())
    return {
        "n_obs": n,
        "excess_sharpe": excess_sharpe(r, rf_annual, periods_per_year),
        "cagr": cagr,
        "vol_annual": float(r.std(ddof=1) * math.sqrt(periods_per_year)) if n > 1 else float("nan"),
        "max_drawdown": dd,
        "total_return": total,
    }


# ----------------------------------------------------------------------------
# DSR building blocks
# ----------------------------------------------------------------------------

def sharpe_estimator_variance(sr: float, n_obs: int, skew: float = 0.0,
                              kurtosis: float = 3.0) -> float:
    """Variance of the per-period Sharpe estimator (Mertens / Lo, non-normal)."""
    if n_obs <= 1:
        return float("nan")
    return float((1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr ** 2) / (n_obs - 1))


def expected_max_sharpe(n_trials: float, sr_variance: float) -> float:
    """E[max SR] of ``n_trials`` independent zero-skill trials.

    ``sqrt(V) * ((1 - gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N e)))``;
    0 for ``N <= 1``.  Units follow ``sr_variance`` (daily here).
    """
    n = float(n_trials)
    if not np.isfinite(n) or n <= 1.0 or not np.isfinite(sr_variance) or sr_variance <= 0:
        return 0.0
    z1 = sp_stats.norm.ppf(1.0 - 1.0 / n)
    z2 = sp_stats.norm.ppf(1.0 - 1.0 / (n * math.e))
    return float(math.sqrt(sr_variance) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def probabilistic_sharpe(sr: float, sr_benchmark: float, n_obs: int,
                         skew: float = 0.0, kurtosis: float = 3.0) -> float:
    """PSR = Phi((SR - SR*) sqrt(T-1) / sqrt(1 - g3 SR + (g4-1)/4 SR^2)), per-period SR."""
    if n_obs <= 2 or not np.isfinite(sr):
        return float("nan")
    denom_sq = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr ** 2
    if denom_sq <= 0:
        return float("nan")
    z = (sr - sr_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(denom_sq)
    return float(sp_stats.norm.cdf(z))


def min_track_record_length(sr_daily: float, sr_benchmark_daily: float = 0.0,
                            skew: float = 0.0, kurtosis: float = 3.0,
                            confidence: float = 0.95) -> float:
    """Minimum number of DAILY observations for PSR(SR*) >= ``confidence``.

    ``MinTRL = 1 + (1 - g3 SR + (g4-1)/4 SR^2) * (z_alpha / (SR - SR*))^2``.
    Returns ``inf`` when ``SR <= SR*``.
    """
    diff = float(sr_daily) - float(sr_benchmark_daily)
    if not np.isfinite(diff) or diff <= 0:
        return float("inf")
    z = sp_stats.norm.ppf(confidence)
    bracket = 1.0 - skew * sr_daily + (kurtosis - 1.0) / 4.0 * sr_daily ** 2
    return float(1.0 + bracket * (z / diff) ** 2)


# ----------------------------------------------------------------------------
# Effective number of trials
# ----------------------------------------------------------------------------

def _clean_matrix(trials_matrix: pd.DataFrame) -> pd.DataFrame:
    m = trials_matrix.dropna(axis=0, how="any")
    sd = m.std(ddof=1)
    const = sd[sd < 1e-15].index
    if len(const):
        logger.info("Dropping %d constant trial columns before clustering", len(const))
        m = m.drop(columns=const)
    return m


def _mean_intra_corr(corr: np.ndarray, labels: np.ndarray) -> float:
    """Average rho over all pairs i<j in the same cluster (1.0 if no pairs)."""
    from scipy import sparse

    uniq, inv = np.unique(labels, return_inverse=True)
    n = labels.size
    sizes = np.bincount(inv).astype("float64")
    n_pairs = float(np.sum(sizes * (sizes - 1) / 2.0))
    if n_pairs == 0:
        return 1.0
    onehot = sparse.csr_matrix((np.ones(n), (np.arange(n), inv)), shape=(n, uniq.size))
    per_row = np.asarray(onehot.T.dot(corr.T).T)  # N x k: sum of rho_ij over j in cluster k
    within_total = float(per_row[np.arange(n), inv].sum())
    within_pairs = (within_total - float(np.trace(corr))) / 2.0
    return within_pairs / n_pairs


def _cluster_labels(values: np.ndarray, min_intra_corr: float) -> np.ndarray:
    """Hierarchical average-linkage labels on columns of ``values`` (T x N)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    n = values.shape[1]
    if n == 1:
        return np.ones(1, dtype=int)
    corr = np.corrcoef(values, rowvar=False)
    corr = np.clip(np.nan_to_num(corr, nan=0.0), -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    z = linkage(squareform(dist, checks=False), method="average")

    cut = math.sqrt(0.5 * (1.0 - min_intra_corr))
    return fcluster(z, t=cut, criterion="distance")


def effective_number_of_trials(trials_matrix: pd.DataFrame, min_intra_corr: float = 0.5,
                               max_cluster_trials: int = 3000,
                               random_state: int = 0) -> Dict[str, object]:
    """Cluster trials by return correlation; the cluster count is effective N.

    See the module docstring for the method.  Returns ``n_eff``, ``labels``
    (Series trial -> cluster id), ``n_raw`` and ``method``.
    """
    m = _clean_matrix(trials_matrix)
    cols = list(m.columns)
    n_raw = len(cols)
    if n_raw == 0:
        return {"n_eff": 0, "labels": pd.Series(dtype=int), "n_raw": 0, "method": "empty"}
    values = m.to_numpy(dtype="float64")
    if n_raw <= max_cluster_trials:
        labels = _cluster_labels(values, min_intra_corr)
        method = "average_linkage"
    else:
        labels = _cluster_large(values, min_intra_corr, max_cluster_trials,
                                np.random.default_rng(random_state))
        method = "average_linkage_subsample_assign"
    labels_s = pd.Series(labels, index=cols, name="cluster")
    mean_rho = float("nan")
    if n_raw <= max_cluster_trials:
        corr = np.nan_to_num(np.corrcoef(values, rowvar=False), nan=0.0)
        mean_rho = _mean_intra_corr(np.atleast_2d(corr), labels)
    return {"n_eff": int(np.unique(labels).size), "labels": labels_s, "n_raw": n_raw,
            "method": method, "mean_intra_corr": mean_rho}


def _cluster_large(values: np.ndarray, min_intra_corr: float, max_n: int,
                   rng: np.random.Generator, offset: int = 0) -> np.ndarray:
    n = values.shape[1]
    if n <= max_n:
        return _cluster_labels(values, min_intra_corr) + offset
    sample = rng.choice(n, size=max_n, replace=False)
    sample_labels = _cluster_labels(values[:, sample], min_intra_corr)
    z = (values - values.mean(axis=0)) / values.std(axis=0, ddof=0)
    ids = np.unique(sample_labels)
    centroids = np.column_stack([z[:, sample[sample_labels == c]].mean(axis=1) for c in ids])
    centroids = (centroids - centroids.mean(axis=0)) / centroids.std(axis=0, ddof=0)
    corr_to_c = (z.T @ centroids) / values.shape[0]  # N x k
    best = corr_to_c.argmax(axis=1)
    assigned = corr_to_c[np.arange(n), best] >= min_intra_corr
    assigned[sample] = True
    labels = np.empty(n, dtype=int)
    labels[sample] = sample_labels + offset
    rest = np.where(assigned)[0]
    rest = rest[~np.isin(rest, sample)]
    labels[rest] = ids[best[rest]] + offset
    unassigned = np.where(~assigned)[0]
    if unassigned.size:
        next_offset = int(labels[assigned].max()) + 1
        labels[unassigned] = _cluster_large(values[:, unassigned], min_intra_corr,
                                            max_n, rng, offset=next_offset)
    return labels


# ----------------------------------------------------------------------------
# Deflated Sharpe
# ----------------------------------------------------------------------------

def deflated_sharpe(returns: pd.Series, trials_matrix: Optional[pd.DataFrame] = None,
                    n_trials: Optional[float] = None, rf_annual: float = 0.0,
                    periods_per_year: int = PERIODS_PER_YEAR, threshold: float = 0.95,
                    min_intra_corr: float = 0.5, sr_variance: Optional[float] = None,
                    ) -> Dict[str, object]:
    """Deflated Sharpe ratio of ``returns`` (daily simple returns).

    Parameters
    ----------
    returns : daily returns of the selected strategy.
    trials_matrix : optional date x trial daily returns of EVERY configuration
        evaluated (e.g. ``TrialRegistry.returns_matrix()``); supplies the
        cross-trial Sharpe variance and the clustered effective N.
    n_trials : explicit effective N (overrides the cluster count).
    rf_annual : annual risk-free rate subtracted (geometrically) per day.
    sr_variance : explicit daily-Sharpe variance V (overrides both sources).

    Returns
    -------
    dict with dsr, sr_daily, sr_annual, sr0_daily, sr0_annual, n_trials_eff,
    n_trials_raw, T, skew, kurtosis (non-excess), sr_variance,
    variance_source, n_trials_source, psr_zero, min_trl_days, passed.
    """
    r = pd.Series(returns, dtype="float64").dropna()
    ex = (r - daily_rf(rf_annual, periods_per_year)).to_numpy()
    t_obs = int(ex.size)
    if t_obs < 3:
        raise ValueError(f"need at least 3 return observations, got {t_obs}")
    sd = float(ex.std(ddof=1))
    sr = float(ex.mean() / sd) if sd > 1e-15 else 0.0
    skew = float(sp_stats.skew(ex, bias=False)) if sd > 1e-15 else 0.0
    kurt = float(sp_stats.kurtosis(ex, fisher=False, bias=False)) if sd > 1e-15 else 3.0

    n_raw: Optional[int] = None
    n_source = "none"
    n_eff: float = 1.0
    var_source = "estimator"
    if trials_matrix is not None and trials_matrix.shape[1] > 0:
        tm = trials_matrix.dropna(axis=0, how="any")
        n_raw = int(tm.shape[1])
        if sr_variance is None and n_raw >= 2:
            srs = np.array([sharpe_daily(tm[c].to_numpy(), rf_annual, periods_per_year)
                            for c in tm.columns])
            srs = srs[np.isfinite(srs)]
            if srs.size >= 2:
                sr_variance = float(srs.var(ddof=1))
                var_source = "trials_matrix"
        if n_trials is None:
            clus = effective_number_of_trials(tm, min_intra_corr=min_intra_corr)
            n_eff = float(clus["n_eff"])
            n_source = f"clusters:{clus['method']}"
    elif sr_variance is not None:
        var_source = "explicit"
    if sr_variance is not None and var_source == "estimator":
        var_source = "explicit"
    if n_trials is not None:
        n_eff = float(n_trials)
        n_source = "explicit"
    if n_source == "none":
        logger.warning("deflated_sharpe: no trials_matrix or n_trials given; N=1 "
                       "(no multiple-testing deflation, DSR == PSR vs 0)")
    if sr_variance is None:
        sr_variance = sharpe_estimator_variance(sr, t_obs, skew, kurt)

    sr0 = expected_max_sharpe(n_eff, sr_variance)
    dsr = probabilistic_sharpe(sr, sr0, t_obs, skew, kurt)
    ann = math.sqrt(periods_per_year)
    return {
        "dsr": dsr,
        "sr_daily": sr,
        "sr_annual": sr * ann,
        "sr0_daily": sr0,
        "sr0_annual": sr0 * ann,
        "n_trials_eff": n_eff,
        "n_trials_raw": n_raw,
        "n_trials_source": n_source,
        "T": t_obs,
        "skew": skew,
        "kurtosis": kurt,
        "sr_variance": float(sr_variance),
        "variance_source": var_source,
        "psr_zero": probabilistic_sharpe(sr, 0.0, t_obs, skew, kurt),
        "min_trl_days": min_track_record_length(sr, sr0, skew, kurt, threshold),
        "passed": bool(np.isfinite(dsr) and dsr >= threshold),
    }


def deflated_sharpe_from_stats(sr_daily: float, n_obs: int, n_trials: float,
                               skew: float = 0.0, kurtosis: float = 3.0,
                               sr_variance: Optional[float] = None,
                               threshold: float = 0.95) -> Dict[str, object]:
    """DSR from summary statistics (daily Sharpe, daily T, non-excess kurtosis)."""
    var = sr_variance if sr_variance is not None else sharpe_estimator_variance(
        sr_daily, n_obs, skew, kurtosis)
    sr0 = expected_max_sharpe(n_trials, var)
    dsr = probabilistic_sharpe(sr_daily, sr0, n_obs, skew, kurtosis)
    return {"dsr": dsr, "sr_daily": sr_daily, "sr_annual": sr_daily * math.sqrt(PERIODS_PER_YEAR),
            "sr0_daily": sr0, "sr0_annual": sr0 * math.sqrt(PERIODS_PER_YEAR),
            "n_trials_eff": float(n_trials), "T": n_obs, "skew": skew, "kurtosis": kurtosis,
            "sr_variance": var, "passed": bool(np.isfinite(dsr) and dsr >= threshold)}

