"""
The maximal Sharpe ratio of a set of test assets, estimated without building the efficient portfolio (tracker HSR1).

Chernov, Kelly, Malamud and Schwab (2026), "The Hidden Sharpe Ratio", NBER Working Paper 35988.  For P test assets
(managed portfolios) and T periods with P/T not small, the plug-in ``mean' inv(cov) mean`` (the GRS statistic,
eq. 9) is biased upward and does not exist when P >= T.  The paper's lower bound needs no condition beyond its
Assumption 1, and is what this module computes:

* the ridge GRS statistic ``Theta(z) = mean' (zI + cov)^-1 mean`` (eq. 10) on returns rescaled by P^-1/2 (Section 2),
* its random-matrix debiasing ``Theta_d = (Theta - xi) / (1 + xi)`` with ``xi = 1 / (1 - c + c z m(-z)) - 1`` and
  ``m(-z) = tr((zI + cov)^-1) / P`` (eqs. 12-14), which estimates the population ridge Sharpe ratio at the
  effective shrinkage ``Z* = z (1 + xi)`` (Lemma 1),
* the first-order debiased statistic ``Theta_1`` (eq. 28, Appendix C.1) and the J = 1 lower bound on the squared
  maximal Sharpe ratio ``H_L1 = Theta_0^2 / (Theta_0 - Theta_1)`` (eq. 32, A.15),
* the ceiling on any ridge-regularised Markowitz portfolio's out-of-sample squared Sharpe ratio, ``H_L1 / Z*'``
  with ``Z*' >= 1`` (Proposition 4, eqs. 34-35).

Degrees of freedom (our reading, flagged).  Eq. 8 writes the centred sample covariance with 1/T and c = P/T.  A
centred covariance from T periods has rank T - 1, and the random-matrix identities behind eqs. 12-16 hold for its
T - 1 degrees of freedom: with 1/T and P/T, D(z) stops falling at 1/T and the mean's component outside the sample span
is multiplied by 1/z, so for P > T the debiased statistic missed its target by 4x to 25x in simulation.  This module
uses n = T - 1 for both, which the paper's asymptotics cannot tell apart from T; the simulated medians then sit within
about 10% of the truth for P/T from 0.5 to 10 (tests/test_max_sharpe.py).

Not implemented, because their formulas are in the paper's online appendix, which the working paper does not
contain: the upper bound's data-driven threshold (Appendix K), the correction of confidence limits for the search over
the ridge grid (Appendix L) and the pivotal confidence bands (Appendix H).  In their place ``detectability`` gives a
sign-flip randomisation p-value for the best lower bound, the search over the grid included.  Orders J >= 2 are
refused: the paper uses J = 1 because the second-order term needs samples far longer than 240 months (Section 5).

Where it fails: the estimate is noisy unless the squared Sharpe ratio is large next to sqrt(P/T / T).  The paper's
U.S. maximum (7.4 a year, 4.6 a month squared) clears that easily; a few Indian characteristics over 14 years may not,
which is what ``detectability`` is for.

The estimates are gross of trading costs and assume long-short, frictionless trading of every test asset, i.i.d.
returns within the window and P/T bounded away from zero (Assumption 1).  They measure what a set of characteristics
spans, not what a long-only, cost-bearing book can earn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MaxSharpeConfig:
    #: Ridge penalties; the paper's grid (Section 5.2).  How they are read is ``z_scale``.
    z_grid: Tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1)
    #: "mean_eigenvalue": each penalty is a multiple of the mean eigenvalue tr(cov)/P of the rescaled returns, so the
    #: estimate does not depend on the returns' units (our reading, flagged: under Assumption 1's normalisation that
    #: mean is of order one and the paper's absolute grid is the same thing; on NSE characteristic portfolios in
    #: decimal monthly returns the absolute grid is 18x to 7,000,000x the mean eigenvalue, which shrinks every
    #: estimate to nothing).  "absolute": the penalties as given, in squared return units.
    z_scale: str = "mean_eigenvalue"
    #: Periods a year of the input returns, for annualising: 12 for the paper's monthly returns.
    periods_per_year: int = 12
    #: Fewest periods accepted.  The theory is asymptotic in P and T together; the paper's windows are 240 months.
    min_periods: int = 60
    #: Truncation order of the series (Corollary 3).  Only 1 is implemented.
    order: int = 1
    #: Relative tolerance for the identities the theory guarantees (Z*' >= 1, a non-negative residual mass).
    tolerance: float = 1e-9
    #: Largest Theta_1 / Theta_0 for which H_L1 = Theta_0 / (1 - ratio) is reported.  Our guard, not the paper's: the
    #: ratio of two noisy estimates near 1 makes the bound explode (on pure noise, ratios of 0.96 gave 25x the
    #: truth); 0.9 caps H_L1 at 10x Theta_0.
    max_decay_ratio: float = 0.9


def _validated_returns(returns: pd.DataFrame, cfg: MaxSharpeConfig) -> np.ndarray:
    if cfg.order != 1:
        raise ValueError(f"order {cfg.order}: only the J = 1 bound is implemented (the paper's empirical choice)")
    if cfg.periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    if not cfg.z_grid or any(not (math.isfinite(z) and z > 0) for z in cfg.z_grid):
        raise ValueError(f"every ridge penalty must be finite and positive: {cfg.z_grid}")
    if cfg.z_scale not in ("mean_eigenvalue", "absolute"):
        raise ValueError(f"z_scale must be 'mean_eigenvalue' or 'absolute', not {cfg.z_scale!r}")
    if not isinstance(returns.index, pd.DatetimeIndex) or not returns.index.is_monotonic_increasing \
            or not returns.index.is_unique:
        raise ValueError("returns need a unique, increasing DatetimeIndex (one row per period)")
    values = returns.to_numpy(dtype="float64")
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("returns must be a periods x assets table with at least one asset")
    if not np.isfinite(values).all():
        raise ValueError("returns hold NaN or infinite values: fill or drop them before estimating")
    if values.shape[0] < cfg.min_periods:
        raise ValueError(f"{values.shape[0]} periods, fewer than min_periods = {cfg.min_periods}")
    return values


def _spectrum(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """Eigenvalues of the centred sample covariance of the P^-1/2-rescaled returns (eq. 8, with the T - 1 degrees of
    freedom of the module docstring), the squared projections of the mean on their eigenvectors, the mean's squared
    mass outside their span, and n = T - 1.  Computed from the T x P data matrix's SVD, so P = 5,000 costs O(T^2 P);
    the remaining P - rank eigenvalues are exactly zero."""
    t, p = values.shape
    scaled = values / math.sqrt(p)
    mean = scaled.mean(axis=0)
    _, s, vt = np.linalg.svd(scaled - mean, full_matrices=False)
    n = t - 1
    eig = s ** 2 / n
    proj2 = (vt @ mean) ** 2
    if eig.size and eig.max() > 0:                  # numerically zero eigenvalues are counted as exact zeros
        keep = eig > _ZERO_EIGENVALUE * eig.max()
        eig, proj2 = eig[keep], proj2[keep]
    residual = float(mean @ mean - proj2.sum())
    if residual < -1e-9 * max(float(mean @ mean), 1e-300):
        raise ValueError(f"the mean's mass outside the sample span is negative ({residual:.3g}): numerical failure")
    return eig, proj2, max(residual, 0.0), n


#: Eigenvalues below this share of the largest are exact zeros (a centred sample's rank is T - 1).
_ZERO_EIGENVALUE = 1e-12


def _at_penalty(z: float, eig: np.ndarray, proj2: np.ndarray, residual: float, p: int, n: int,
                cfg: MaxSharpeConfig) -> Dict[str, Any]:
    """Steps 1-3 of Appendix C.1 at one ridge penalty; ``n`` is the covariance's degrees of freedom.

    ``eig`` holds the r non-zero eigenvalues; the other P - r are zero and the mean's mass on them is ``residual``.
    The sums of eq. C.2 are split accordingly, and D, D' and D - z D' are written so that no two large terms cancel:
    D = 1 - c r/P + (c z/P) sum 1/(z+l),  D' = (c/P) sum l/(z+l)^2,  D - z D' = 1 - c r/P + (c z^2/P) sum 1/(z+l)^2.
    """
    c, r = p / n, eig.size
    inv1, inv2 = 1.0 / (z + eig), 1.0 / (z + eig) ** 2
    t1 = float((proj2 * inv1).sum() + residual / z)                    # T_1(z), eq. C.2
    t2 = float((proj2 * inv2).sum() + residual / z ** 2)               # T_2(z)
    base = 1.0 - c * r / p
    d = base + c * z / p * float(inv1.sum())                           # eq. C.4
    if not (math.isfinite(d) and d > 0):
        raise ValueError(f"bias-correction denominator D(z) = {d:.3g} is not positive at z = {z:g}")
    xi = 1.0 / d - 1.0                                                 # eq. 13
    z_star = z / d                                                     # eq. 16
    theta = t1                                                         # eq. 10
    theta_d = d * (theta + 1.0) - 1.0                                  # eq. 12 / C.5
    d_prime = c / p * float((eig * inv2).sum())                        # eq. C.6
    z_star_prime = (base + c * z ** 2 / p * float(inv2.sum())) / d ** 2   # eq. C.7
    if z_star_prime < 1.0 - cfg.tolerance:                             # eq. 35: Z*' >= 1
        raise ValueError(f"Z*'({z:g}) = {z_star_prime:.6f} < 1 contradicts Proposition 4 (eq. 35): numerical failure")
    theta_d_prime = d_prime * (t1 + 1.0) - d * t2                      # eq. C.8
    theta_1 = z_star * (-theta_d_prime / z_star_prime)                 # eq. C.9
    row: Dict[str, Any] = {"z": z, "D": d, "xi": xi, "z_star": z_star, "z_star_prime": z_star_prime,
                              "theta": theta, "theta_0": theta_d, "theta_1": theta_1,
                              "lower_bound_sq": float("nan"), "ridge_ceiling_sq": float("nan")}
    if not theta_d > 0:
        row["status"] = "no detectable Sharpe ratio: the debiased statistic is not positive"
    elif not 0 < theta_1 < theta_d:
        row["status"] = "the estimates break Proposition 1 (0 < Theta_1 < Theta_0 must hold): no bound"
    elif theta_1 > cfg.max_decay_ratio * theta_d:
        row["status"] = (f"Theta_1 / Theta_0 = {theta_1 / theta_d:.3f} above max_decay_ratio "
                         f"{cfg.max_decay_ratio:g}: H_L1 would be noise-driven")
    else:
        bound = theta_d ** 2 / (theta_d - theta_1)                     # eq. 32 at J = 1, A.15
        row.update(status="ok", lower_bound_sq=bound, ridge_ceiling_sq=bound / z_star_prime)
    return row


def _grid(values: np.ndarray, cfg: MaxSharpeConfig) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    eig, proj2, residual, n = _spectrum(values)
    p = values.shape[1]
    unit = 1.0
    if cfg.z_scale == "mean_eigenvalue":
        unit = float(eig.sum()) / p                                    # tr(cov) / P, the zero eigenvalues included
        if not (math.isfinite(unit) and unit > 0):
            raise ValueError("the returns do not vary: the covariance has no positive eigenvalue")
    rows = []
    for z in sorted(cfg.z_grid):
        row = _at_penalty(z * unit, eig, proj2, residual, p, n, cfg)
        row["z_grid"] = z
        rows.append(row)
    return rows, eig, proj2


def _best_debiased(values: np.ndarray, cfg: MaxSharpeConfig) -> float:
    """The largest debiased statistic Theta_0 over the grid: the zeroth-order lower bound (B_0(Z*) <= R*^2), which
    may be negative on noise.  Stable where H_L1, a ratio of two estimates, is not."""
    return max(float(r["theta_0"]) for r in _grid(values, cfg)[0])


def detectability(returns: pd.DataFrame, cfg: MaxSharpeConfig = MaxSharpeConfig(), draws: int = 200,
                  seed: int = 0) -> Dict[str, float]:
    """Is there any Sharpe ratio to find?  A sign-flip randomisation test of "no expected return in any test asset"
    on the best debiased statistic Theta_0 over the grid.  Each draw multiplies every period's returns by an
    independent random sign, which keeps their covariance and removes any mean, and recomputes the same maximum, so
    the search over penalties is in the null too.  Exact when periods are independent and symmetric about zero under
    the null; serial dependence or skew make it approximate.  Not the paper's confidence band (online Appendix H): a
    guard against reading noise as Sharpe ratio.  ``p_value = (1 + draws at or above the observed) / (1 + draws)``;
    statistics are per period, squared."""
    if draws < 19:
        raise ValueError("at least 19 draws are needed for a p-value of 0.05")
    values = _validated_returns(returns, cfg)
    observed = _best_debiased(values, cfg)
    rng = np.random.default_rng(seed)
    null = np.array([_best_debiased(values * rng.choice([-1.0, 1.0], size=(values.shape[0], 1)), cfg)
                     for _ in range(draws)])
    return {"observed_sq": observed, "null_median_sq": float(np.median(null)),
            "null_95_sq": float(np.quantile(null, 0.95)),
            "p_value": float((1 + (null >= observed).sum()) / (1 + draws)), "draws": draws}


def max_sharpe(returns: pd.DataFrame, cfg: MaxSharpeConfig = MaxSharpeConfig()) -> Dict[str, Any]:
    """The J = 1 lower bound on the maximal Sharpe ratio of ``returns`` (periods x test assets, decimal excess
    returns), at each ridge penalty of the grid and at the best of them (the largest lower bound, Section 5.2).

    Annualised figures are ``sqrt(periods_per_year * squared per-period Sharpe)``.  ``plug_in`` is the classical GRS
    statistic, reported only when the sample covariance is invertible (P < T): it overstates the maximum by about P/T.
    """
    values = _validated_returns(returns, cfg)
    t, p = values.shape
    rows, eig, proj2 = _grid(values, cfg)
    valid = [r for r in rows if r["status"] == "ok"]
    best = max(valid, key=lambda r: r["lower_bound_sq"]) if valid else None
    annual = lambda sq: math.sqrt(cfg.periods_per_year * sq) if sq == sq and sq >= 0 else float("nan")  # noqa: E731
    plug_in = float("nan")
    if p < t - 1 and eig.size == p and eig.min() > cfg.tolerance * eig.max():
        plug_in = float((proj2 / eig).sum())                           # eq. 9, on the same rescaled returns
    return {
        "periods": t, "assets": p, "c": p / (t - 1), "start": returns.index[0], "end": returns.index[-1],
        "grid": rows,
        "status": "ok" if best else "no valid bound at any ridge penalty",
        "lower_bound_sq": best["lower_bound_sq"] if best else float("nan"),
        "z_best": best["z"] if best else float("nan"),
        "annual_lower_bound": annual(best["lower_bound_sq"]) if best else float("nan"),
        "annual_ridge_ceiling": annual(best["ridge_ceiling_sq"]) if best else float("nan"),
        "z_star_prime": best["z_star_prime"] if best else float("nan"),
        "zeroth_order_sq": max(float(r["theta_0"]) for r in rows),
        "annual_zeroth_order": annual(max(float(r["theta_0"]) for r in rows)),
        "plug_in_sq": plug_in, "annual_plug_in": annual(plug_in),
    }


def _best_debiased_table(returns: pd.DataFrame, cfg: MaxSharpeConfig) -> float:
    return _best_debiased(_validated_returns(returns, cfg), cfg)


def _shuffled(periods: Mapping[pd.Timestamp, Tuple[pd.Series, pd.DataFrame]], name: str,
              rng: np.random.Generator) -> Dict[pd.Timestamp, Tuple[pd.Series, pd.DataFrame]]:
    """The periods with ``name`` permuted across the stocks of each month: its distribution kept, its link to the
    returns (and to the other characteristics) cut."""
    out = {}
    for day, (returns, table) in periods.items():
        table = table.copy()
        table[name] = rng.permutation(table[name].to_numpy())
        out[day] = (returns, table)
    return out


def spanning_increment(periods: Mapping[pd.Timestamp, Tuple[pd.Series, pd.DataFrame]], base: Sequence[str],
                       candidate: str, cfg: MaxSharpeConfig = MaxSharpeConfig(), random_features: int = 2000,
                       seeds: Sequence[int] = (1, 2, 3), permutations: int = 99, permutation_seed: int = 0,
                       min_names: int = 100, alpha: float = 0.10) -> Dict[str, Any]:
    """Does ``candidate`` widen the frontier that the ``base`` characteristics span?  Phase 0's spanning screen
    (research protocol, tracker HSR2), after the paper's Section 5.4 test of later characteristics.

    The statistic is the rise in the best debiased squared Sharpe ratio (Theta_0, per period, the stable
    zeroth-order bound) when the candidate joins the base: averaged over ``seeds`` for ``random_features`` nonlinear
    portfolios, and for the linear ones.  Its null comes from ``permutations`` copies of the panel with the
    candidate shuffled across the stocks of each month, which keeps its distribution and cuts its link to returns:
    ``p = (1 + null rises at or above the observed) / (1 + permutations)``.  GO needs the nonlinear p at most
    ``alpha`` and a rise in every seed.  Gross of costs and long-short, like the bound itself: it says whether the
    signal carries information the base does not, not whether the book can trade it.
    """
    base = list(base)
    if candidate in base:
        raise ValueError(f"{candidate} is already in the base set")
    if not seeds or permutations < 19:
        raise ValueError("at least one seed and 19 permutations (for p = 0.05) are needed")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    names = base + [candidate]

    def bound(panel, cols, rf, seed):
        return _best_debiased_table(managed_portfolios(panel, cols, random_features=rf, seed=seed,
                                                       min_names=min_names)[0], cfg)

    base_rf = [bound(periods, base, random_features, s) for s in seeds]
    base_linear = bound(periods, base, 0, 0)

    def rises(panel):
        nonlinear = [bound(panel, names, random_features, s) - b for s, b in zip(seeds, base_rf, strict=True)]
        return nonlinear, bound(panel, names, 0, 0) - base_linear

    observed_rf, observed_linear = rises(periods)
    rng = np.random.default_rng(permutation_seed)
    null_rf, null_linear = [], []
    for _ in range(permutations):
        rf, linear = rises(_shuffled(periods, candidate, rng))
        null_rf.append(float(np.mean(rf)))
        null_linear.append(linear)
    mean_rf = float(np.mean(observed_rf))
    p_rf = float((1 + sum(x >= mean_rf for x in null_rf)) / (1 + permutations))
    p_linear = float((1 + sum(x >= observed_linear for x in null_linear)) / (1 + permutations))
    annual = lambda sq: math.sqrt(cfg.periods_per_year * sq) if sq > 0 else 0.0   # noqa: E731
    return {
        "candidate": candidate, "base": base, "seeds": list(seeds), "permutations": permutations,
        "rise_sq_by_seed": observed_rf, "rise_sq": mean_rf, "p_value": p_rf,
        "annual_base": float(np.mean([annual(b) for b in base_rf])),
        "annual_with_candidate": float(np.mean([annual(b + r) for b, r in zip(base_rf, observed_rf, strict=True)])),
        "linear_rise_sq": observed_linear, "linear_p_value": p_linear,
        "linear_annual_base": annual(base_linear), "linear_annual_with_candidate": annual(base_linear + observed_linear),
        "verdict": "GO" if p_rf <= alpha and min(observed_rf) > 0 else "STOP",
    }


def attach_characteristic(periods: Mapping[pd.Timestamp, Tuple[pd.Series, pd.DataFrame]], values: pd.DataFrame,
                          name: str) -> Dict[pd.Timestamp, Tuple[pd.Series, pd.DataFrame]]:
    """Add a characteristic to every month of ``periods`` from ``values`` (columns date, symbol, value: the value
    known at that month's close).  Every month of the panel must be covered; a stock it lacks counts as missing (0
    after rank-standardising).  Fails on non-finite values, a name already present or a month it does not cover."""
    if not {"date", "symbol", "value"} <= set(values.columns):
        raise ValueError("the candidate needs columns date, symbol and value")
    frame = values.assign(date=pd.to_datetime(values["date"]), symbol=values["symbol"].astype(str))
    numbers = pd.to_numeric(frame["value"], errors="coerce")
    if numbers.isna().any() or not np.isfinite(numbers.to_numpy(dtype="float64")).all():
        raise ValueError("the candidate holds missing, non-numeric or infinite values: drop those rows instead")
    if frame.duplicated(["date", "symbol"]).any():
        raise ValueError("the candidate has more than one value for a (date, symbol)")
    by_day = {day: grp.set_index("symbol")["value"].astype("float64") for day, grp in frame.assign(value=numbers)
              .groupby("date")}
    missing = [d for d in periods if pd.Timestamp(d) not in by_day]
    if missing:
        raise ValueError(f"the candidate does not cover {len(missing)} panel months, first {min(missing).date()}: use "
                         f"the panel's month-end dates (the last session of each month)")
    out = {}
    for day, (returns, table) in periods.items():
        if name in table.columns:
            raise ValueError(f"{name} is already a characteristic of the panel")
        out[day] = (returns, table.assign(**{name: by_day[pd.Timestamp(day)].reindex(table.index)}))
    return out


def rank_standardise(table: pd.DataFrame) -> pd.DataFrame:
    """Each column ranked across the rows into [-0.5, 0.5] (ties averaged), missing values set to 0, the
    cross-sectional median under this scaling (Section 5.1).  A column with one value in a month is all 0."""
    n = table.notna().sum()
    ranks = table.rank(method="average")
    out = (ranks - 1.0).div((n - 1.0).where(n > 1)) - 0.5
    return out.fillna(0.0)


def managed_portfolios(periods: Mapping[pd.Timestamp, Tuple[pd.Series, pd.DataFrame]], names: Sequence[str],
                       random_features: int = 0, seed: int = 0, min_names: int = 30) -> Tuple[pd.DataFrame, int]:
    """Characteristic-managed portfolio returns ``S_t' R_t+1 / N_t`` (eq. 48), one row per formation date.

    ``periods`` maps a formation date to (next-period decimal returns, characteristics), both indexed by stock and
    known at that date, as ``runners.run_nse_engine._slope_panel`` builds them (its returns are in percent: divide by
    100 first).  Stocks lacking the return are dropped; missing characteristics count as 0 after rank-standardising.

    ``random_features = 0`` gives the linear portfolios (eq. 49), one per characteristic.  ``random_features = P``
    gives P nonlinear ones, ``S = ReLU(0.5 sqrt(2/d) X W')`` with ``W`` (P x d) i.i.d. standard normal drawn once from
    ``seed`` (eq. 50).  The scale 0.5 sqrt(2/d) is read from the paper's typesetting; ReLU is positively homogeneous,
    so it scales every portfolio alike and only moves where the ridge grid sits.  Returns the table and the number of
    dates skipped for having fewer than ``min_names`` stocks.
    """
    names = list(names)
    if not names:
        raise ValueError("no characteristics given")
    if random_features < 0:
        raise ValueError("random_features must be 0 (linear) or a positive count")
    weights = np.random.default_rng(seed).standard_normal((random_features, len(names))) if random_features else None
    rows: Dict[pd.Timestamp, np.ndarray] = {}
    skipped = 0
    for day, (returns, table) in sorted(periods.items()):
        missing = [n for n in names if n not in table.columns]
        if missing:
            raise ValueError(f"{day}: characteristics missing from the table: {missing}")
        r = pd.Series(returns, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
        x = rank_standardise(table[names].reindex(r.index).astype("float64"))
        if len(r) < min_names:
            skipped += 1
            continue
        s = x.to_numpy() if weights is None else \
            np.maximum(0.5 * math.sqrt(2.0 / len(names)) * x.to_numpy() @ weights.T, 0.0)
        rows[pd.Timestamp(day)] = s.T @ r.to_numpy() / len(r)
    if not rows:
        raise ValueError("no date had enough stocks for a managed portfolio")
    columns: List[str] = names if weights is None else [f"rf{i}" for i in range(random_features)]
    return pd.DataFrame.from_dict(rows, orient="index", columns=columns).sort_index(), skipped
