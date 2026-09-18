"""
Shared data types for the NSE engine.

These are the contracts between the data layer (``nse_engine.data``), the
engine (``nse_engine.engine``), validation (``nse_engine.validation``) and
live execution (``kite_connect.trading.nse_engine_executor``).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Mapping, Optional

import numpy as np
import pandas as pd

PRICE_FIELDS = ("open", "high", "low", "close", "volume", "value")


@dataclass
class MarketData:
    """Date-aligned daily NSE panel.

    Every frame is indexed by ``dates`` (the NSE trading calendar, sorted and
    unique) with one column per canonical symbol (the symbol's latest name
    after renames).  Prices and volumes are adjusted for splits, bonuses and
    other corporate actions; ``value`` is traded value in INR and needs no
    adjustment.  A symbol that did not trade on a date has NaN there.

    ``index_close`` holds index levels (columns such as ``NIFTY50``,
    ``NIFTY500``, ``INDIAVIX``) on the same calendar.
    """

    dates: pd.DatetimeIndex
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    value: pd.DataFrame
    index_close: pd.DataFrame
    delivery_pct: Optional[pd.DataFrame] = None
    #: close as printed by the bhavcopy that day, with no corporate-action
    #: back-adjustment. Prices in ``close`` are adjusted from the END of the
    #: loaded window, so a 2013 level moves when a 2024 split happens; any
    #: rule about the price a trader would have seen (eligibility, lot value)
    #: must use this frame instead. Never use it for returns.
    close_unadj: Optional[pd.DataFrame] = None
    etfs: FrozenSet[str] = frozenset()
    sectors: Dict[str, str] = field(default_factory=dict)
    source: str = "unknown"
    data_hash: str = ""

    @property
    def symbols(self) -> List[str]:
        return list(self.close.columns)

    def validate(self) -> None:
        """Raise ValueError if frames are not aligned to ``dates``."""
        if not self.dates.is_monotonic_increasing or self.dates.has_duplicates:
            raise ValueError("dates must be sorted and unique")
        cols = self.close.columns
        for name in PRICE_FIELDS:
            frame = getattr(self, name)
            if not frame.index.equals(self.dates):
                raise ValueError(f"{name} index is not aligned to dates")
            if not frame.columns.equals(cols):
                raise ValueError(f"{name} columns differ from close columns")
        if not self.index_close.index.equals(self.dates):
            raise ValueError("index_close index is not aligned to dates")
        if self.delivery_pct is not None and not self.delivery_pct.index.equals(self.dates):
            raise ValueError("delivery_pct index is not aligned to dates")
        if self.close_unadj is not None and (not self.close_unadj.index.equals(self.dates)
                                             or not self.close_unadj.columns.equals(cols)):
            raise ValueError("close_unadj is not aligned to dates/close columns")

    def until(self, as_of: pd.Timestamp) -> "MarketData":
        """Return a view containing only rows dated <= ``as_of``.

        Used by tests (and live) to prove that decisions at ``as_of`` never
        read later data.
        """
        as_of = pd.Timestamp(as_of)
        n = int(self.dates.searchsorted(as_of, side="right"))
        cut = lambda f: None if f is None else f.iloc[:n]  # noqa: E731
        return MarketData(
            dates=self.dates[:n],
            open=cut(self.open), high=cut(self.high), low=cut(self.low),
            close=cut(self.close), volume=cut(self.volume), value=cut(self.value),
            index_close=cut(self.index_close), delivery_pct=cut(self.delivery_pct),
            etfs=self.etfs, sectors=self.sectors, source=self.source,
            data_hash=self.data_hash,
        )

    def compute_hash(self) -> str:
        """Deterministic content hash of the panel (dates, symbols, closes)."""
        h = hashlib.sha256()
        h.update(np.asarray(self.dates.asi8).tobytes())
        h.update("|".join(map(str, self.close.columns)).encode())
        h.update(np.nan_to_num(self.close.to_numpy(dtype="float64")).round(4).tobytes())
        h.update(np.nan_to_num(self.value.to_numpy(dtype="float64")).round(0).tobytes())
        return h.hexdigest()[:16]


@dataclass
class Holding:
    """A live or simulated position (quantity in shares, CNC)."""

    symbol: str
    quantity: int
    avg_price: float
    entry_date: pd.Timestamp
    stop_price: Optional[float] = None


@dataclass
class TargetPortfolio:
    """Output of ``nse_engine.engine.generate_targets`` for one decision date.

    ``weights`` are fractions of total equity (long-only, sum <= max_gross),
    covering both core stocks and sleeve ETFs.  ``stops`` are trailing stop
    triggers (in the same adjusted price scale as ``MarketData`` at
    ``as_of``) for every core holding that should carry a GTT stop.
    """

    as_of: pd.Timestamp
    weights: Dict[str, float]
    stops: Dict[str, float] = field(default_factory=dict)
    forecasts: Dict[str, float] = field(default_factory=dict)
    ranks: Dict[str, int] = field(default_factory=dict)
    core_weights: Dict[str, float] = field(default_factory=dict)
    sleeve_weights: Dict[str, float] = field(default_factory=dict)
    exits: Dict[str, str] = field(default_factory=dict)  # symbol -> reason
    regime: str = "neutral"  # risk_on | neutral | risk_off
    regime_scale: float = 1.0
    universe_size: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def gross(self) -> float:
        return float(sum(self.weights.values()))


@dataclass
class Trade:
    date: pd.Timestamp
    symbol: str
    side: str  # BUY | SELL
    quantity: int
    price: float
    value_inr: float
    cost_inr: float
    reason: str  # rebalance | stop | rank_exit | sleeve | regime
    requested_quantity: Optional[int] = None  # before participation cap


@dataclass
class BacktestResult:
    equity: pd.Series  # INR, indexed by date
    returns: pd.Series  # daily net simple returns
    weights: pd.DataFrame  # end-of-day weights (date x symbol)
    trades: pd.DataFrame  # one row per Trade
    metrics: Dict[str, float]
    config: "object"  # EngineConfig (typed loosely to avoid an import cycle)
    data_hash: str
    run_id: str = ""
    run_dir: Optional[str] = None
    notes: List[str] = field(default_factory=list)


def holdings_from_mapping(rows: Mapping[str, Mapping]) -> Dict[str, Holding]:
    """Build ``Holding`` objects from plain dicts (broker / paper payloads)."""
    out: Dict[str, Holding] = {}
    for sym, r in rows.items():
        qty = int(r.get("quantity", 0))
        if qty <= 0:
            continue
        out[sym] = Holding(
            symbol=sym,
            quantity=qty,
            avg_price=float(r.get("avg_price", r.get("average_price", 0.0)) or 0.0),
            entry_date=pd.Timestamp(r.get("entry_date") or pd.Timestamp.today().normalize()),
            stop_price=r.get("stop_price"),
        )
    return out
