"""
Statistical validation for the NSE engine (Phase 2).

Every function works on daily simple returns and daily Sharpe units
internally; annualised figures use sqrt(252) and excess returns over the
risk-free rate.  See docs/nse_engine.md for the contracts.
"""

from .benchmarks import benchmark_gate, run_benchmarks, simulate_weights
from .diagnostics import aronson_detrended_sharpe, alpha_beta, full_report, lag_sensitivity
from .dsr import (
    deflated_sharpe,
    deflated_sharpe_from_stats,
    effective_number_of_trials,
    excess_sharpe,
    expected_max_sharpe,
    min_track_record_length,
    performance_summary,
    probabilistic_sharpe,
    sharpe_daily,
)
from .holdout import HoldoutLockedError, holdout_evaluations, run_holdout
from .pbo import cscv_pbo
from .trials import TrialRegistry, record_result
from .walk_forward import expand_grid, generate_folds, run_walk_forward

__all__ = [
    "TrialRegistry", "record_result",
    "cscv_pbo",
    "deflated_sharpe", "deflated_sharpe_from_stats", "effective_number_of_trials",
    "expected_max_sharpe", "min_track_record_length", "probabilistic_sharpe",
    "sharpe_daily", "excess_sharpe", "performance_summary",
    "run_walk_forward", "generate_folds", "expand_grid",
    "run_holdout", "holdout_evaluations", "HoldoutLockedError",
    "run_benchmarks", "benchmark_gate", "simulate_weights",
    "aronson_detrended_sharpe", "alpha_beta", "lag_sensitivity", "full_report",
]
