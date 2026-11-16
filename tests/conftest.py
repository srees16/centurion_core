"""Shared fixtures: a small synthetic market panel, so most tests need no store.

The real parquet store (250 MB, gitignored) is only needed by tests marked
``slow``; they skip when it is absent, so a clean checkout can still run the suite.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# The suite never touches Neon, as in CI: config.py loads .env, whose URL is the live paper books, so a test that
# builds a PaperTrader would restore them, and one that forgot its fake store could write to them.
os.environ["CENTURION_DATABASE_URL"] = ""

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nse_engine.types import MarketData  # noqa: E402

STORE = ROOT / "data" / "nse_engine" / "store"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs the parquet store or a long run")


# CENTURION_NO_SKIPS=1 marks a run that must exercise every test it collects (the full-install job in CI): a test
# that skips there checked nothing, so the run fails and names it.
_SKIPPED: list = []


def pytest_runtest_logreport(report):
    if report.skipped and not hasattr(report, "wasxfail"):
        _SKIPPED.append(report.nodeid)


def pytest_sessionfinish(session, exitstatus):
    if os.environ.get("CENTURION_NO_SKIPS") == "1" and _SKIPPED and session.exitstatus == 0:
        print(f"\nCENTURION_NO_SKIPS: {len(_SKIPPED)} test(s) skipped, so they checked nothing:\n  " + "\n  ".join(_SKIPPED))
        session.exitstatus = 1


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


@pytest.fixture(autouse=True)
def _fresh_order_pace():
    """Each test starts with no recent orders, so the limiter's real waits do not pile up across tests."""
    from kite_connect.trading import order_service

    order_service._order_times.clear()


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


# ── the order gate's fakes, shared by the order, kill-switch, executor, go-live and postback tests ──

class _Broker:
    """A broker that accepts orders into a book.  ``lose`` answers are lost after the order is accepted, ``refuse``
    calls fail before it is, and ``book_down`` makes the order book unreadable."""

    def __init__(self, book=None, lose=0, refuse=0, book_down=False):
        self.book, self.lose, self.refuse, self.book_down, self.sent = list(book or []), lose, refuse, book_down, 0

    def positions(self):
        return {"day": []}

    def orders(self):
        if self.book_down:
            raise ConnectionError("order book down")
        return self.book

    def order_history(self, order_id):
        return []

    def place_order(self, **params):
        if self.refuse:
            self.refuse -= 1
            raise ConnectionError("no route to host")
        self.sent += 1
        order_id = f"O{self.sent}"
        self.book.append({"order_id": order_id, "tag": params["tag"], "status": "OPEN"})
        if self.lose:
            self.lose -= 1
            raise TimeoutError("read timed out")
        return order_id


@pytest.fixture
def gate(monkeypatch):
    from kite_connect.trading import order_service as os_

    monkeypatch.setattr(os_, "_is_nse_market_open", lambda: True)
    monkeypatch.setattr(os_, "is_kill_switch_active", lambda: False)
    monkeypatch.setattr(os_, "_persist_to_db", lambda *a, **k: None)
    monkeypatch.setattr(os_, "_send_order_email", lambda *a, **k: None)
    monkeypatch.setattr(os_, "_order_circuit", None)
    monkeypatch.setattr(os_.time, "sleep", lambda *_: None)
    return os_


LIMIT = dict(symbol="RELIANCE", exchange="NSE", transaction_type="BUY", quantity=5, order_type="LIMIT", price=2500.0)


@pytest.fixture
def route(gate, monkeypatch):
    pytest.importorskip("fastapi")
    from api.routers.v1 import kite as route_

    monkeypatch.setattr(route_, "get_kite_session", lambda: route_._test_broker, raising=False)
    return route_


def _post(route_, broker, **fields):
    import asyncio

    route_._test_broker = broker
    body = dict(tradingsymbol="RELIANCE", exchange="NSE", transaction_type="BUY", quantity=5, order_type="LIMIT",
                price=2500.0, tag="abc123")
    return asyncio.run(route_.kite_place_order(route_.KiteOrderRequest(**{**body, **fields})))
