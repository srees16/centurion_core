"""
NSE long-only engine: momentum/trend/low-vol core book plus gold and silver
ETF trend sleeves, shared by research backtests and live target generation.

Exports are resolved lazily so that importing a sub-package (for example
``nse_engine.data``) does not pull in the whole engine.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "EngineConfig": "nse_engine.config",
    "MarketData": "nse_engine.types",
    "Holding": "nse_engine.types",
    "TargetPortfolio": "nse_engine.types",
    "Trade": "nse_engine.types",
    "BacktestResult": "nse_engine.types",
    "EngineCache": "nse_engine.engine",
    "generate_targets": "nse_engine.engine",
    "run_backtest": "nse_engine.engine",
    "record_run": "nse_engine.engine",
    "compute_metrics": "nse_engine.metrics",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'nse_engine' has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
