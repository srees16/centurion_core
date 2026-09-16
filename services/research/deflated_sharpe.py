"""
Deflated Sharpe Ratio (DSR) — Bailey & López de Prado (2014), AFML Ch.14.

Adjusts the Sharpe ratio for multiple testing when selecting among
strategies or parameter sets.

UNITS (read this)
-----------------
The PSR/DSR z-score is ``(SR - SR*) * sqrt(n_obs - 1) / sqrt(...)``.  ``SR``,
``SR*`` and the Sharpe dispersion must be PER OBSERVATION: with daily
``n_obs`` they must be DAILY Sharpe ratios.  Mixing an annualised Sharpe
(e.g. 1.127) with a daily observation count (e.g. 3,189) inflates the
z-score by ``sqrt(252)`` -- the misuse this module now prevents:

* every public function takes ``periods_per_year`` (observations per year
  of ``n_obs``, default 252 = daily) and ``annualized`` (default ``True``:
  ``observed_sr`` / ``sr_std`` are annualised and are converted to
  per-observation units by dividing by ``sqrt(periods_per_year)``);
* pass ``annualized=False`` when the Sharpe is already per observation; a
  warning is logged when such a value looks annualised;
* ``sr_std`` (cross-trial std of Sharpe ratios under the null) defaults to
  the Sharpe *estimator* standard error
  ``sqrt((1 - g3*SR + (g4-1)/4*SR^2) / (n_obs - 1))`` instead of the
  meaningless legacy default 1.0.

Reference values (daily SR 0.071 = 1.127 annualised, T=3189, skew -1.19,
kurtosis 14.3, estimator variance): DSR 0.97 at N=24, 0.67 at N=1,550,
0.42 at N=20,000.

For new code prefer ``nse_engine.validation.dsr.deflated_sharpe`` which works
from the return series and clusters trials for the effective N.

Usage:
    from services.research.deflated_sharpe import deflated_sharpe_ratio, min_backtest_length
    dsr = deflated_sharpe_ratio(observed_sr=1.127, n_obs=3189, n_trials=24,
                                skewness=-1.19, kurtosis=14.3)  # annual SR, daily obs
"""

import logging
import math
from typing import Optional

from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

_EULER_GAMMA = 0.5772156649015329
# A per-observation Sharpe above this (annualised > ~6.3 for daily data) is
# almost certainly an annualised value passed with annualized=False.
_IMPLAUSIBLE_PERIOD_SR = 0.4


def to_period_sharpe(
    sr: float,
    periods_per_year: float = 252.0,
    annualized: bool = True,
) -> float:
    """Convert a Sharpe ratio to per-observation units.

    ``annualized=True``: ``sr / sqrt(periods_per_year)``; otherwise ``sr``
    unchanged (with a warning if it looks annualised).
    """
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be positive, got {periods_per_year}")
    if annualized:
        return float(sr) / math.sqrt(periods_per_year)
    if periods_per_year > 12 and abs(sr) > _IMPLAUSIBLE_PERIOD_SR:
        logger.warning(
            "Per-observation Sharpe %.3f (= %.2f annualised at %g periods/year) is implausible; "
            "did you pass an ANNUALISED Sharpe with annualized=False?",
            sr, sr * math.sqrt(periods_per_year), periods_per_year,
        )
    return float(sr)


def _sr_estimator_std(sr_p: float, n_obs: int, skewness: float, kurtosis: float) -> float:
    bracket = 1.0 - skewness * sr_p + ((kurtosis - 1.0) / 4.0) * sr_p ** 2
    if n_obs <= 1 or bracket <= 0:
        return float("nan")
    return math.sqrt(bracket / (n_obs - 1))


def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    periods_per_year: float = 252.0,
    annualized: bool = True,
) -> float:
    """Probabilistic Sharpe Ratio: P(true SR > benchmark_sr).

    de Prado (2018) Eq. 14.1:
        PSR = Φ[(SR - SR*) × √(n-1) / √(1 - γ₃·SR + (γ₄-1)/4 · SR²)]
    evaluated with per-observation SR (see module docstring).

    Parameters
    ----------
    observed_sr : Sharpe ratio (annualised when ``annualized=True``)
    benchmark_sr : threshold SR* in the SAME units as ``observed_sr``
    n_obs : number of observations at ``periods_per_year`` frequency
    skewness : sample skewness (γ₃)
    kurtosis : sample kurtosis (γ₄, NOT excess kurtosis; normal = 3.0)
    periods_per_year : observations per year of ``n_obs`` (252 = daily)
    annualized : whether the Sharpe inputs are annualised

    Returns
    -------
    probability in [0, 1]. Higher = more confident SR exceeds benchmark.
    """
    if n_obs <= 2:
        return 0.0
    sr_p = to_period_sharpe(observed_sr, periods_per_year, annualized)
    sr_star_p = float(benchmark_sr) / math.sqrt(periods_per_year) if annualized else float(benchmark_sr)

    num = (sr_p - sr_star_p) * math.sqrt(n_obs - 1)
    denom_sq = 1.0 - skewness * sr_p + ((kurtosis - 1.0) / 4.0) * sr_p ** 2
    if denom_sq <= 0:
        return 0.0
    return float(sp_stats.norm.cdf(num / math.sqrt(denom_sq)))


def expected_max_sr(
    n_trials: float,
    mean_sr: float = 0.0,
    std_sr: Optional[float] = None,
) -> float:
    """Expected maximum Sharpe ratio of ``n_trials`` zero-skill trials.

    de Prado (2018) Eq. 14.4:
        E[max(SR)] ≈ mean_sr + std_sr × [(1 - γ) Φ⁻¹(1 - 1/N) + γ Φ⁻¹(1 - 1/(N e))]

    Unit-agnostic: the result is in the units of ``mean_sr`` / ``std_sr``.
    ``std_sr`` is required (the legacy default of 1.0 had no meaning).
    """
    if std_sr is None:
        raise ValueError("expected_max_sr: std_sr is required, in the same units as the Sharpe "
                         "ratios (per-observation for DSR)")
    if n_trials <= 1:
        return mean_sr
    z1 = sp_stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = sp_stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return mean_sr + std_sr * ((1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2)


def deflated_sharpe_ratio(
    observed_sr: float,
    n_obs: int,
    n_trials: float,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    sr_std: Optional[float] = None,
    periods_per_year: float = 252.0,
    annualized: bool = True,
) -> float:
    """Deflated Sharpe Ratio: PSR with SR* = E[max(SR)] under the null.

    Parameters
    ----------
    observed_sr : Sharpe of the selected strategy (annualised by default)
    n_obs : number of return observations at ``periods_per_year`` frequency
    n_trials : (effective) number of strategies/parameter sets tested
    skewness : sample skewness of returns
    kurtosis : sample kurtosis of returns (normal = 3.0)
    sr_std : cross-trial std of Sharpe ratios, same units as ``observed_sr``;
        ``None`` (default) uses the Sharpe estimator standard error
    periods_per_year : observations per year (252 = daily ``n_obs``)
    annualized : whether ``observed_sr`` / ``sr_std`` are annualised

    Returns
    -------
    DSR in [0, 1]. Values >= 0.95 indicate the SR is likely genuine.
    """
    if n_obs <= 2:
        return 0.0
    sr_p = to_period_sharpe(observed_sr, periods_per_year, annualized)
    if sr_std is None:
        std_p = _sr_estimator_std(sr_p, n_obs, skewness, kurtosis)
        if not math.isfinite(std_p):
            return 0.0
    else:
        std_p = float(sr_std) / math.sqrt(periods_per_year) if annualized else float(sr_std)
        if periods_per_year > 12 and std_p > _IMPLAUSIBLE_PERIOD_SR:
            logger.warning("deflated_sharpe_ratio: per-observation sr_std %.3f is implausibly "
                           "large (legacy default 1.0?); DSR will be near 0", std_p)
    sr_star_p = expected_max_sr(n_trials, mean_sr=0.0, std_sr=std_p)
    return probabilistic_sharpe_ratio(sr_p, sr_star_p, n_obs, skewness, kurtosis,
                                      periods_per_year=periods_per_year, annualized=False)


def min_backtest_length(
    target_sr: float,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    confidence: float = 0.95,
    periods_per_year: float = 252.0,
    annualized: bool = True,
) -> int:
    """Minimum track record length (observations) to trust SR > 0 at ``confidence``.

    de Prado (2018) Eq. 14.2 with per-observation SR:
        MinTRL = 1 + [1 - γ₃·SR + (γ₄-1)/4 · SR²] × (z_α / SR)²

    Returns the number of observations at ``periods_per_year`` frequency
    (trading days by default).  Non-positive SR returns 99999.
    """
    sr_p = to_period_sharpe(target_sr, periods_per_year, annualized)
    if sr_p <= 1e-9:
        return 99999
    z_alpha = sp_stats.norm.ppf(confidence)
    bracket = 1.0 - skewness * sr_p + ((kurtosis - 1.0) / 4.0) * sr_p ** 2
    n = 1.0 + bracket * (z_alpha / sr_p) ** 2
    return max(10, int(math.ceil(n)))


def compute_dsr_for_strategies(
    sharpe_ratios: dict,
    n_obs: int,
    returns_stats: Optional[dict] = None,
    periods_per_year: float = 252.0,
    annualized: bool = True,
    n_trials: Optional[float] = None,
) -> dict:
    """Compute DSR for a set of strategies.

    Parameters
    ----------
    sharpe_ratios : {strategy_name: Sharpe} (annualised unless ``annualized=False``)
    n_obs : common observation count at ``periods_per_year`` frequency
    returns_stats : optional {strategy_name: {"skewness": float, "kurtosis": float}}
    n_trials : effective number of trials (default: ``len(sharpe_ratios)``;
        pass the total number of configurations ever evaluated if larger)

    Returns
    -------
    {strategy_name: {"sharpe": float, "dsr_pvalue": float, "dsr_significant": bool,
                     "min_btl": int (observations)}}
    """
    n = float(n_trials) if n_trials is not None else float(len(sharpe_ratios))
    if not sharpe_ratios:
        return {}

    results = {}
    for name, sr in sharpe_ratios.items():
        skew = 0.0
        kurt = 3.0
        if returns_stats and name in returns_stats:
            skew = returns_stats[name].get("skewness", 0.0)
            kurt = returns_stats[name].get("kurtosis", 3.0)

        dsr_p = deflated_sharpe_ratio(
            observed_sr=sr, n_obs=n_obs, n_trials=n, skewness=skew, kurtosis=kurt,
            periods_per_year=periods_per_year, annualized=annualized,
        )
        results[name] = {
            "sharpe": round(sr, 4),
            "dsr_pvalue": round(dsr_p, 4),
            "dsr_significant": dsr_p >= 0.95,
            "min_btl": min_backtest_length(sr, skew, kurt, periods_per_year=periods_per_year,
                                           annualized=annualized),
        }
    return results
