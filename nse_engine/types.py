"""
Shared data types for the NSE engine.

These are the contracts between the data layer (``nse_engine.data``), the
engine (``nse_engine.engine``), validation (``nse_engine.validation``) and
live execution (``kite_connect.trading.nse_engine_executor``).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, fields
from typing import Dict, FrozenSet, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

PRICE_FIELDS = ("open", "high", "low", "close", "volume", "value")


#: Version of ``MarketData.compute_hash``, recorded in run manifests (2: rename-invariant and widened, LN-T15).
DATA_HASH_VERSION = 2
#: MarketData fields that are event tables (one row per dated event), not frames on ``dates``.
EVENT_FIELDS = ("corporate_events", "dividend_events")


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
    #: Dated snapshots of ``sectors``, oldest first (tracker SB2).  When set, a decision uses
    #: the latest one dated on or before it and none before the first, so history is never
    #: capped by today's index members; None: ``sectors`` holds for every date.
    sector_history: Optional[List[Tuple[pd.Timestamp, Dict[str, str]]]] = None
    source: str = "unknown"
    data_hash: str = ""
    #: The adjustments behind the back-adjusted prices, one row per (date, symbol): ``factor`` and
    #: its ``source`` (``panel.SOURCE_NAMES``), and the cash ``dividend`` per share on ex-dates.  The
    #: paper and live books apply them to their own quantities, stops and cash (tracker LN-T4); not
    #: part of ``data_hash``.
    corporate_events: Optional[pd.DataFrame] = None
    dividend_events: Optional[pd.DataFrame] = None
    #: {column: [(first session, name traded under), ...]} for columns renamed in the data: decisions break
    #: ties by the name in force that day (tracker LN-T15); empty when no column was renamed.
    trade_names: Dict[str, List[Tuple[pd.Timestamp, str]]] = field(default_factory=dict)

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

        Every date-indexed frame is cut, ``close_unadj`` included, so a book
        that filters on as-printed prices plans as its registered runs did
        (tracker LN-T8: the candidate and E4 books had planned on adjusted
        prices); the event tables keep the rows dated <= ``as_of``; anything
        else passes through.  A field added later is cut without being listed.
        """
        as_of = pd.Timestamp(as_of)
        n = int(self.dates.searchsorted(as_of, side="right"))
        kw = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name == "dates":
                v = v[:n]
            elif f.name in EVENT_FIELDS:
                v = None if v is None else v[v["date"] <= as_of]
            elif isinstance(v, pd.DataFrame):
                v = v.iloc[:n]
            kw[f.name] = v
        return MarketData(**kw)

    def names_at_end(self) -> List[str]:
        """Each column's name in force on the panel's last date (a column renamed later keeps its old one)."""
        last = pd.Timestamp(self.dates[-1]) if len(self.dates) else None
        out = []
        for c in self.close.columns:
            name = str(c)
            for d, n in (self.trade_names or {}).get(str(c), ()):
                if last is not None and pd.Timestamp(d) <= last:
                    name = n
            out.append(name)
        return out

    def compute_hash(self) -> str:
        """Deterministic content hash of the panel (version ``DATA_HASH_VERSION``).

        Version 2 (tracker LN-T15): columns are named and ordered by the name in
        force on the panel's last date, and the renames inside the window are
        hashed, so a rename after the window never moves the hash (version 1
        keyed on the latest names: a 2026 rename re-hashed 2013-25); and every
        price frame (close, open, high, low, the as-printed close), volume,
        traded value, the ETF set and the index closes the regime gate reads are
        hashed (index closes since 27 Sep 2026, tracker K5), so a change to any
        of them does move it.  Missing index values hash as -1.
        """
        names = self.names_at_end()
        order = np.argsort(np.array(names, dtype=object), kind="stable")
        last = pd.Timestamp(self.dates[-1]) if len(self.dates) else None
        h = hashlib.sha256()
        h.update(f"v{DATA_HASH_VERSION}".encode())
        h.update(np.asarray(self.dates.asi8).tobytes())
        h.update("|".join(names[i] for i in order).encode())
        for frame, decimals in ((self.close, 4), (self.open, 4), (self.high, 4), (self.low, 4),
                                (self.close_unadj, 4), (self.volume, 0), (self.value, 0)):
            if frame is not None:
                h.update(np.nan_to_num(frame.to_numpy(dtype="float64")[:, order]).round(decimals).tobytes())
        renames = sorted((names[i], [(pd.Timestamp(d).date().isoformat(), n) for d, n in segs if pd.Timestamp(d) <= last])
                         for i, c in enumerate(self.close.columns)
                         for segs in [(self.trade_names or {}).get(str(c), [])]
                         if len([d for d, _ in segs if last is not None and pd.Timestamp(d) <= last]) > 1)
        h.update(repr(renames).encode())
        at_end = dict(zip(map(str, self.close.columns), names))
        h.update("|".join(sorted(at_end.get(str(e), str(e)) for e in self.etfs)).encode())
        ic = self.index_close
        if ic is not None and len(ic.columns):
            h.update("|".join(map(str, ic.columns)).encode())
            h.update(np.nan_to_num(ic.to_numpy(dtype="float64"), nan=-1.0).round(4).tobytes())
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
    drawdown_state: str = "normal"   # nse_engine.drawdown state used for this decision
    drawdown_scale: float = 1.0

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
    daily_state: Optional[pd.DataFrame] = None  # drawdown-rule state per session, when the rule is on


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
