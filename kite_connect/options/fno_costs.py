"""
Execution costs for NSE F&O orders, date-aware for backtests (tracker: options toolkit).

The same two parts as ``nse_engine.costs`` for equities:

* **Charges** per order side: brokerage, STT (options: sell-side premium,
  plus exercise of ITM longs at expiry; futures: sell-side notional),
  exchange transaction charge, SEBI fee, stamp duty (buys) and GST on
  brokerage + exchange + SEBI.  Rates come from ``ChargesConfig`` schedules;
  the SEBI fee and GST/service tax are the equity model's.
* **Slippage** per unit and side: max(min ticks, a share of premium).

Live, Kite's basket-margin response itemises the same charges;
:func:`charges_from_kite` reads it so the two can be compared before a trade.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, Iterable, List, Mapping, Optional, Union

import pandas as pd

from kite_connect.options.options_config import ChargesConfig, OptionsConfig, Schedule, SlippageConfig
from kite_connect.options.theory import BUY, CALL, PUT, SELL, intrinsic_value, side_sign
from nse_engine.costs import SEBI_FEE_RATE, indirect_tax_rate

DateLike = Union[str, pd.Timestamp]

#: Recorded with every options backtest; runs are compared only within one version.
#: 1 = the schedules and slippage of ``OptionsConfig`` as first written (3 Oct 2026).
FNO_COST_MODEL_VERSION = 1


@dataclass(frozen=True)
class FnoCharges:
    """Charges in INR."""

    brokerage: float = 0.0
    stt: float = 0.0
    exchange: float = 0.0
    sebi: float = 0.0
    stamp_duty: float = 0.0
    gst: float = 0.0

    @property
    def total(self) -> float:
        return self.brokerage + self.stt + self.exchange + self.sebi + self.stamp_duty + self.gst

    def __add__(self, other: "FnoCharges") -> "FnoCharges":
        return FnoCharges(*(getattr(self, f.name) + getattr(other, f.name) for f in fields(self)))


def rate_on(schedule: Schedule, date: DateLike) -> float:
    """The schedule's rate in force on ``date``."""
    d = pd.Timestamp(date)
    if d < pd.Timestamp(schedule[0][0]):
        raise ValueError(f"{d.date()} is before the schedule starts ({schedule[0][0]})")
    rate = schedule[0][1]
    for start, r in schedule:
        if d >= pd.Timestamp(start):
            rate = r
    return rate


def _side(side: str) -> str:
    side_sign(side)
    return side.upper()


def option_charges(premium_value_inr: float, side: str, date: DateLike, orders: int = 1,
                   cfg: ChargesConfig = ChargesConfig()) -> FnoCharges:
    """Charges for one option order side of premium x quantity = ``premium_value_inr``.

    ``orders`` counts executed orders (autoslice makes one per slice), each paying brokerage.
    """
    side, value = _side(side), abs(float(premium_value_inr))
    if value <= 0:
        return FnoCharges()
    brokerage = cfg.brokerage_per_order_inr * orders
    stt = rate_on(cfg.stt_option_sell, date) * value if side == SELL else 0.0
    exchange = rate_on(cfg.exchange_option, date) * value
    sebi = SEBI_FEE_RATE * value
    stamp = rate_on(cfg.stamp_option_buy, date) * value if side == BUY else 0.0
    gst = indirect_tax_rate(date) * (brokerage + exchange + sebi)
    return FnoCharges(brokerage, stt, exchange, sebi, stamp, gst)


def futures_charges(notional_inr: float, side: str, date: DateLike, orders: int = 1,
                    cfg: ChargesConfig = ChargesConfig()) -> FnoCharges:
    """Charges for one futures order side of price x quantity = ``notional_inr``."""
    side, value = _side(side), abs(float(notional_inr))
    if value <= 0:
        return FnoCharges()
    per_order = value / orders
    brokerage = min(cfg.brokerage_per_order_inr, cfg.brokerage_futures_pct * per_order) * orders
    stt = rate_on(cfg.stt_futures_sell, date) * value if side == SELL else 0.0
    exchange = rate_on(cfg.exchange_futures, date) * value
    sebi = SEBI_FEE_RATE * value
    stamp = rate_on(cfg.stamp_futures_buy, date) * value if side == BUY else 0.0
    gst = indirect_tax_rate(date) * (brokerage + exchange + sebi)
    return FnoCharges(brokerage, stt, exchange, sebi, stamp, gst)


def exercise_stt(option_type: str, strike: float, settlement_price: float, quantity: int, date: DateLike,
                 cfg: ChargesConfig = ChargesConfig()) -> float:
    """STT the holder of a long option pays when it expires ITM (and is exercised).

    From 1 Sep 2019 on the intrinsic value; before, on the full settlement value
    (Varsity's "STT trap", which made exiting before expiry cheaper).
    """
    intrinsic = intrinsic_value(option_type, settlement_price, strike)
    if intrinsic <= 0 or quantity <= 0:
        return 0.0
    full_value_base = pd.Timestamp(date) < pd.Timestamp(cfg.stt_exercise_intrinsic_from)
    base = settlement_price if full_value_base else intrinsic
    return rate_on(cfg.stt_option_exercise, date) * base * quantity


def slippage_per_unit(premium: float, underlying: str, cfg: SlippageConfig = SlippageConfig()) -> float:
    """Adverse move from the reference price per unit, one side."""
    pct = cfg.index_pct if underlying.upper() in cfg.index_underlyings else cfg.stock_pct
    return max(cfg.min_ticks * cfg.tick_size, pct * abs(premium))


def fill_price(premium: float, side: str, underlying: str, cfg: SlippageConfig = SlippageConfig()) -> float:
    """Reference price moved against the order: buys higher, sells lower (never below one tick)."""
    slip = slippage_per_unit(premium, underlying, cfg)
    return premium + slip if _side(side) == BUY else max(premium - slip, cfg.tick_size)


def legs_costs(legs: Iterable, quantity: int, date: DateLike, underlying: str,
               cfg: OptionsConfig = OptionsConfig(), orders_per_leg: Optional[Mapping[int, int]] = None,
               closing: bool = False) -> Dict[str, object]:
    """Charges and slippage to open (or, with ``closing``, to square off) a set of legs.

    ``quantity`` is units per ratio (lot size x lots); leg premiums are the
    reference prices.  ``orders_per_leg`` maps leg index -> executed orders
    (autoslice slices), default 1.
    """
    total, slippage, rows = FnoCharges(), 0.0, []
    for i, leg in enumerate(legs):
        qty = leg.ratio * quantity
        side = leg.side if not closing else (SELL if leg.side == BUY else BUY)
        n = (orders_per_leg or {}).get(i, 1)
        if leg.option_type in (CALL, PUT):
            c = option_charges(leg.premium * qty, side, date, n, cfg.charges)
            slip = slippage_per_unit(leg.premium, underlying, cfg.slippage) * qty
        else:
            c = futures_charges(leg.strike * qty, side, date, n, cfg.charges)
            slip = cfg.slippage.min_ticks * cfg.slippage.tick_size * qty
        total, slippage = total + c, slippage + slip
        rows.append({"leg": leg.label, "side": side, "quantity": qty, "orders": n,
                     "charges": c.total, "slippage": slip})
    return {"charges": total, "slippage_inr": slippage, "rows": rows}


def expiry_costs(legs: Iterable, quantity: int, settlement_price: float, date: DateLike,
                 cfg: ChargesConfig = ChargesConfig()) -> float:
    """Exercise STT on the long legs that expire ITM (index options are cash settled)."""
    return sum(exercise_stt(l.option_type, l.strike, settlement_price, l.ratio * quantity, date, cfg)
               for l in legs if l.option_type in (CALL, PUT) and l.side == BUY)


def charges_from_kite(orders: List[Mapping]) -> FnoCharges:
    """Sum the ``charges`` blocks of Kite's order/basket margin response (one per order)."""
    total = FnoCharges()
    for o in orders:
        c = o.get("charges") or {}
        gst = c.get("gst") or {}
        total = total + FnoCharges(
            brokerage=float(c.get("brokerage", 0.0)),
            stt=float(c.get("transaction_tax", 0.0)),
            exchange=float(c.get("exchange_turnover_charge", 0.0)),
            sebi=float(c.get("sebi_turnover_charge", 0.0)),
            stamp_duty=float(c.get("stamp_duty", 0.0)),
            gst=float(gst.get("total", 0.0)) if isinstance(gst, Mapping) else float(gst),
        )
    return total
