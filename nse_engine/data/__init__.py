"""NSE engine data layer: archive download, parquet store, reference data, panel loading.

Heavy modules are imported lazily so ``import nse_engine.data.validation``
stays cheap and free of network dependencies.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BhavcopyArchive",
    "build_store",
    "build_sector_map",
    "load_market_data",
    "load_etf_symbols",
    "load_symbol_changes",
    "resolve_symbols",
]

_EXPORTS = {
    "BhavcopyArchive": "nse_engine.data.archive",
    "build_store": "nse_engine.data.store",
    "build_sector_map": "nse_engine.data.reference",
    "load_etf_symbols": "nse_engine.data.reference",
    "load_symbol_changes": "nse_engine.data.reference",
    "resolve_symbols": "nse_engine.data.reference",
    "load_market_data": "nse_engine.data.panel",
}


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'nse_engine.data' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)
