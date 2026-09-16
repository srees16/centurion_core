"""Services package for Centurion Capital LLC.

Modules are grouped by the job they do. Import them by their full path
(``from services.execution.carver_pipeline import CarverPipeline``); the
lazy re-exports below cover the most-used entry points.

Sub-packages
------------
  app/                – Streamlit app services: analysis runs, session, cache
  decision_engine/    – Rules engine for trade decisions
  execution/          – Order execution, overlays and strategy runners
  market_data/        – Bhavcopy, delivery, corporate actions, events
  metrics/            – Technical indicator calculators
  notifications/      – Email / webhook / SMS alerts
  portfolio/          – Allocation, weighting, portfolio analysis
  regime/             – Regime detection and regime-conditional weights
  research/           – Backtests, statistical validation, live-vs-backtest
  risk/               – Volatility, position risk, stops, risk metrics
  rl_bot/             – Reinforcement-learning trade bot
  sentiment/          – Multi-source sentiment aggregation
  signals/            – Forecast generation and combination (Carver ±20)
  storage/            – MinIO / local file storage
  technical_analysis/ – RSI, MACD, Bollinger, Supertrend aggregator

The NSE engine that paper- and live-trades is a separate package
(``nse_engine/``); these services support research and the Streamlit app.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Convenience re-exports (lazy – only resolved when accessed)
# ---------------------------------------------------------------------------


def __getattr__(name: str):
    """Lazy import for frequently-used symbols."""
    _LAZY = {
        "CarverPipeline": ".execution.carver_pipeline",
        "IntegratedScorer": ".signals.integrated_scorer",
        "AronsonValidator": ".research.aronson_validator",
        "combine_forecasts": ".signals.forecast_combiner",
        "compute_position_size": ".risk.position_sizer",
        "compare_to_benchmark": ".research.benchmark_tracker",
        "run_full_backtest": ".research.full_pipeline_backtest",
        "run_full_evaluation": ".signals.signal_quality_evaluator",
        "walk_forward_validate": ".research.walk_forward",
        "hrp_weights": ".portfolio.hrp_allocator",
        "RiskMetrics": ".risk.risk_metrics",
        "MarketRegime": ".regime.regime_detector",
    }
    if name in _LAZY:
        import importlib

        module = importlib.import_module(_LAZY[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CarverPipeline",
    "IntegratedScorer",
    "AronsonValidator",
    "combine_forecasts",
    "compute_position_size",
    "compare_to_benchmark",
    "run_full_backtest",
    "run_full_evaluation",
    "walk_forward_validate",
    "hrp_weights",
    "RiskMetrics",
    "MarketRegime",
]
