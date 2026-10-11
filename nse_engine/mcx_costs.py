"""
Execution costs for MCX commodity futures, date-aware for backtests (tracker GT2, for GT5).

The same two parts as the equity and F&O models:

* **Charges** per order side, on notional (price x multiplier x lots):
  - brokerage (Zerodha: the lower of Rs 20 and 0.03% per executed order);
  - commodity transaction tax (CTT) on sells of non-agricultural futures;
  - MCX transaction charge;
  - SEBI fee;
  - stamp duty on buys;
  - GST (service tax before 1 Jul 2017) on brokerage + exchange + SEBI.
* **Slippage** per side: a share of the price, a placeholder until paper
  fills measure it (the X1 method).

Rates as of 11 Oct 2026 (Zerodha's charges page and the exchange notices):
- **CTT:** 0.01% of sell notional from 1 Jul 2013 (Finance Act 2013); none
  before; agricultural commodities are exempt.  Budget 2026 raised STT on
  equity futures to 0.05% from 1 Apr 2026, but not CTT.
- **MCX transaction charge:** 0.0021% from 1 Oct 2024; 0.0026% before
  (MCX's notice of 24 Sep 2024).  The tiered rates before about 2017 are
  not modelled; 0.0026% is used throughout.
- **SEBI fee:** Rs 10 per crore (Rs 1 for agri).
- **Stamp duty:** 0.002% of buy notional; uniform from 1 Jul 2020, earlier
  state rates are not modelled.
- **Rolls:** a roll is two orders, so charge both sides.

The SEBI fee and GST / service tax are the equity model's
(``nse_engine.costs``).  ``futures_charges`` takes ``agri=True`` for a farm
contract; GT5's metals and energy contracts are not.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Tuple, Union

import pandas as pd

from nse_engine.costs import SEBI_FEE_RATE, indirect_tax_rate

DateLike = Union[str, pd.Timestamp]
#: (start date, rate) pairs, oldest first; a rate holds until the next start.
Schedule = Tuple[Tuple[str, float], ...]

#: Recorded with every commodity backtest; runs are compared only within one version.
#: 1 = the schedules and slippage below as first written (11 Oct 2026).
MCX_COST_MODEL_VERSION = 1

SEBI_FEE_RATE_AGRI = 0.0000001  # Rs 1 per crore


@dataclass(frozen=True)
class McxChargesConfig:
    """MCX futures charges by date, and the slippage placeholder."""

    brokerage_per_order_inr: float = 20.0
    brokerage_pct: float = 0.0003  # the lower of the flat fee and this share of the order's notional
    ctt_sell: Schedule = (("2003-01-01", 0.0), ("2013-07-01", 0.0001))
    exchange: Schedule = (("2003-01-01", 0.000026), ("2024-10-01", 0.000021))
    stamp_buy: Schedule = (("2003-01-01", 0.00002),)
    slippage_pct: float = 0.0005  # per side, of price: about 2-3 bid-ask spreads on the mini contracts


@dataclass(frozen=True)
class McxCharges:
    """Charges in INR."""

    brokerage: float = 0.0
    ctt: float = 0.0
    exchange: float = 0.0
    sebi: float = 0.0
    stamp_duty: float = 0.0
    gst: float = 0.0

    @property
    def total(self) -> float:
        return self.brokerage + self.ctt + self.exchange + self.sebi + self.stamp_duty + self.gst

    def __add__(self, other: "McxCharges") -> "McxCharges":
        return McxCharges(*(getattr(self, f.name) + getattr(other, f.name) for f in fields(self)))


def rate_on(schedule: Schedule, date: DateLike) -> float:
    """The schedule's rate in force on ``date`` (``kite_connect.options.fno_costs`` has the same lookup;
    nse_engine does not import kite_connect)."""
    d = pd.Timestamp(date)
    if d < pd.Timestamp(schedule[0][0]):
        raise ValueError(f"{d.date()} is before the schedule starts ({schedule[0][0]})")
    rate = schedule[0][1]
    for start, r in schedule:
        if d >= pd.Timestamp(start):
            rate = r
    return rate


def _side(side: str) -> str:
    s = side.upper()
    if s not in ("BUY", "SELL"):
        raise ValueError(f"side must be BUY or SELL, not {side!r}")
    return s


def futures_charges(notional_inr: float, side: str, date: DateLike, orders: int = 1, agri: bool = False,
                    cfg: McxChargesConfig = McxChargesConfig()) -> McxCharges:
    """Charges for one futures order side of price x multiplier x lots = ``notional_inr``.

    ``orders`` counts executed orders, each paying brokerage on its share.
    """
    side, value = _side(side), abs(float(notional_inr))
    if value <= 0:
        return McxCharges()
    brokerage = min(cfg.brokerage_per_order_inr, cfg.brokerage_pct * value / orders) * orders
    ctt = rate_on(cfg.ctt_sell, date) * value if side == "SELL" and not agri else 0.0
    exchange = rate_on(cfg.exchange, date) * value
    sebi = (SEBI_FEE_RATE_AGRI if agri else SEBI_FEE_RATE) * value
    stamp = rate_on(cfg.stamp_buy, date) * value if side == "BUY" else 0.0
    gst = indirect_tax_rate(date) * (brokerage + exchange + sebi)
    return McxCharges(brokerage, ctt, exchange, sebi, stamp, gst)


def fill_price(price: float, side: str, cfg: McxChargesConfig = McxChargesConfig()) -> float:
    """Reference price moved against the order by the slippage: buys higher, sells lower."""
    slip = cfg.slippage_pct * abs(price)
    return price + slip if _side(side) == "BUY" else price - slip
