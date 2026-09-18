"""Shared fixtures: a small synthetic market panel, so most tests need no store.

The real parquet store (250 MB, gitignored) is only needed by tests marked
``slow``; they skip when it is absent, so a clean checkout can still run the suite.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nse_engine.types import MarketData  # noqa: E402

STORE = ROOT / "data" / "nse_engine" / "store"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs the parquet store or a long run")


@pytest.fixture(scope="session")
def store_path() -> Path:
    if not (STORE / "equity").exists():
        pytest.skip("no parquet store in this checkout")
    return STORE


def synthetic_panel(n_days: int = 900, n_symbols: int = 40, seed: int = 7,
                    split_symbol: str = "SYM07", split_at: int = 700,
                    split_factor: float = 0.1) -> MarketData:
    """Deterministic panel: random walks, one late split, one late listing.

    ``split_symbol`` halves-and-then-some at ``split_at`` (a 10:1 split), which is
    what back-adjustment rescales backwards - the B1 look-ahead.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_days, name="date")
    cols = [f"SYM{i:02d}" for i in range(n_symbols)]
    steps = rng.normal(0.0004, 0.016, size=(n_days, n_symbols))
    close = pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)) * rng.uniform(0.3, 6, n_symbols),
                         index=dates, columns=cols)
    unadj = close.copy()
    # a 10:1 split late in the sample: the printed price drops, the adjusted one does not
    unadj.iloc[split_at:, cols.index(split_symbol)] *= split_factor
    close_unadj = unadj
    # a symbol that lists half-way through
    close.iloc[: n_days // 2, cols.index("SYM39")] = np.nan
    close_unadj.iloc[: n_days // 2, cols.index("SYM39")] = np.nan
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.004, size=close.shape))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, size=close.shape))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, size=close.shape))
    volume = pd.DataFrame(rng.integers(5_000, 500_000, size=close.shape), index=dates, columns=cols).astype(float)
    value = close * volume
    index_close = pd.DataFrame({
        "NIFTY50": 18000 * np.exp(np.cumsum(rng.normal(0.0003, 0.009, n_days))),
        "INDIAVIX": 15 + rng.normal(0, 2, n_days).cumsum() % 10,
    }, index=dates)
    deliv = pd.DataFrame(rng.uniform(20, 80, size=close.shape), index=dates, columns=cols)
    data = MarketData(dates=dates, open=open_, high=high, low=low, close=close, volume=volume,
                      value=value, index_close=index_close, delivery_pct=deliv,
                      close_unadj=close_unadj, sectors={c: "Test" for c in cols}, source="synthetic")
    data.data_hash = data.compute_hash()
    data.validate()
    return data


@pytest.fixture
def panel() -> MarketData:
    return synthetic_panel()


def truncate(data: MarketData, upto: str) -> MarketData:
    """The same panel as it would have looked on ``upto`` (no later rows)."""
    cut = data.dates <= pd.Timestamp(upto)
    kw = {name: getattr(data, name).loc[cut] for name in
          ("open", "high", "low", "close", "volume", "value", "index_close")}
    out = MarketData(dates=data.dates[cut], **kw,
                     delivery_pct=data.delivery_pct.loc[cut] if data.delivery_pct is not None else None,
                     close_unadj=data.close_unadj.loc[cut] if data.close_unadj is not None else None,
                     etfs=data.etfs, sectors=data.sectors, source=data.source)
    out.data_hash = out.compute_hash()
    return out
