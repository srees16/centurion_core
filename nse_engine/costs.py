"""
Execution costs for NSE cash-equity delivery (CNC) trades.

Two parts, both per side:

* **Statutory charges** on a date-aware schedule (zero brokerage assumed):
  STT, stamp duty (buys), NSE exchange transaction charge, SEBI turnover fee,
  GST (service tax before 2017-07-01) on exchange charge + SEBI fee, and the
  depository (DP) charge per sell per symbol.
* **Market impact**: ``spread_floor_bps + impact_coefficient_bps *
  sqrt(participation / 1%)`` where participation is order value over the
  median traded value of the past ``adv_lookback_days`` sessions.  Orders are
  cut so participation never exceeds ``max_participation``.

All functions are pure; the simulator in ``nse_engine.engine`` composes them.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

from nse_engine.config import CostConfig

logger = logging.getLogger(__name__)

DateLike = Union[str, pd.Timestamp, "np.datetime64"]

STT_RATE = 0.001  # 0.1% on buy and sell (delivery)
STAMP_DUTY_RATE_NEW = 0.00015  # 0.015% on buys from 2020-07-01
STAMP_DUTY_RATE_OLD = 0.0001  # 0.01% before (approximation of state rates)
STAMP_DUTY_CHANGE = pd.Timestamp("2020-07-01")
EXCHANGE_RATE_OLD = 0.0000325  # 0.00325% until 2024-09-30
EXCHANGE_RATE_NEW = 0.0000297  # 0.00297% from 2024-10-01
EXCHANGE_CHANGE = pd.Timestamp("2024-10-01")
SEBI_FEE_RATE = 0.000001  # 0.0001% (Rs 10 per crore)
GST_RATE = 0.18
SERVICE_TAX_RATE = 0.15
GST_START = pd.Timestamp("2017-07-01")
IMPACT_REFERENCE_PARTICIPATION = 0.01


@dataclass(frozen=True)
class StatutoryCharges:
    """Breakdown of per-side statutory charges in INR."""

    stt: float
    stamp_duty: float
    exchange: float
    sebi: float
    gst: float
    dp: float

    @property
    def total(self) -> float:
        return self.stt + self.stamp_duty + self.exchange + self.sebi + self.gst + self.dp


@dataclass(frozen=True)
class Fill:
    """Result of simulating one order (quantities in shares, money in INR)."""

    side: str
    requested_quantity: int
    quantity: int
    price: float  # reference fill price (open or stop), before impact
    value_inr: float  # quantity * price
    participation: float
    impact_bps: float
    impact_inr: float
    statutory_inr: float

    @property
    def cost_inr(self) -> float:
        return self.impact_inr + self.statutory_inr

    @property
    def capped(self) -> bool:
        return self.quantity < self.requested_quantity


def _ts(date: DateLike) -> pd.Timestamp:
    return pd.Timestamp(date)


def stamp_duty_rate(date: DateLike) -> float:
    return STAMP_DUTY_RATE_NEW if _ts(date) >= STAMP_DUTY_CHANGE else STAMP_DUTY_RATE_OLD


def exchange_rate(date: DateLike) -> float:
    return EXCHANGE_RATE_NEW if _ts(date) >= EXCHANGE_CHANGE else EXCHANGE_RATE_OLD


def indirect_tax_rate(date: DateLike) -> float:
    """GST from 2017-07-01, service tax before."""
    return GST_RATE if _ts(date) >= GST_START else SERVICE_TAX_RATE


def statutory_charges(
    value_inr: float, side: str, date: DateLike, dp_charge_inr: float = CostConfig.dp_charge_inr
) -> StatutoryCharges:
    """Statutory charges for one side of a delivery trade of ``value_inr``."""
    side = side.upper()
    if side not in ("BUY", "SELL"):
        raise ValueError(f"side must be BUY or SELL, got {side!r}")
    value = abs(float(value_inr))
    if value <= 0:
        return StatutoryCharges(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    stt = STT_RATE * value
    stamp = stamp_duty_rate(date) * value if side == "BUY" else 0.0
    exch = exchange_rate(date) * value
    sebi = SEBI_FEE_RATE * value
    gst = indirect_tax_rate(date) * (exch + sebi)
    dp = float(dp_charge_inr) if side == "SELL" else 0.0
    return StatutoryCharges(stt, stamp, exch, sebi, gst, dp)


def statutory_cost(
    value_inr: float, side: str, date: DateLike, dp_charge_inr: float = CostConfig.dp_charge_inr
) -> float:
    """Total statutory charges in INR (see :func:`statutory_charges`)."""
    return statutory_charges(value_inr, side, date, dp_charge_inr).total


def participation_rate(order_value_inr: float, adv_value_inr: float) -> float:
    """Order value / median traded value; ``inf`` when liquidity is unknown."""
    if adv_value_inr is None or not np.isfinite(adv_value_inr) or adv_value_inr <= 0:
        return math.inf if order_value_inr > 0 else 0.0
    return abs(float(order_value_inr)) / float(adv_value_inr)


def impact_bps(order_value_inr: float, adv_value_inr: float, cfg: CostConfig) -> float:
    """Square-root impact in bps: floor + coefficient * sqrt(participation / 1%)."""
    if order_value_inr <= 0:
        return 0.0
    p = participation_rate(order_value_inr, adv_value_inr)
    if not np.isfinite(p):
        p = cfg.max_participation  # unknown liquidity: charge as if at the cap
    return cfg.spread_floor_bps + cfg.impact_coefficient_bps * math.sqrt(p / IMPACT_REFERENCE_PARTICIPATION)


def cap_quantity(requested_quantity: int, price: float, adv_value_inr: float, max_participation: float) -> int:
    """Largest quantity <= requested with participation <= ``max_participation``."""
    q = max(int(requested_quantity), 0)
    if q == 0 or not np.isfinite(price) or price <= 0:
        return 0
    if adv_value_inr is None or not np.isfinite(adv_value_inr) or adv_value_inr <= 0:
        return 0
    max_qty = int(math.floor(max_participation * adv_value_inr / price + 1e-9))
    return min(q, max(max_qty, 0))


def simulate_fill(
    side: str,
    requested_quantity: int,
    price: float,
    adv_value_inr: float,
    date: DateLike,
    cfg: CostConfig,
    *,
    apply_cap: bool = True,
) -> Fill:
    """Simulate a fill: participation cap, impact cost and statutory charges.

    ``price`` is the reference execution price (the open, or the stop level);
    impact is charged as an INR cost rather than moving the recorded price.
    """
    side = side.upper()
    req = max(int(requested_quantity), 0)
    qty = cap_quantity(req, price, adv_value_inr, cfg.max_participation) if apply_cap else req
    value = qty * float(price) if qty > 0 else 0.0
    part = participation_rate(value, adv_value_inr) if qty > 0 else 0.0
    bps = impact_bps(value, adv_value_inr, cfg) if qty > 0 else 0.0
    stat = statutory_cost(value, side, date, cfg.dp_charge_inr) if qty > 0 else 0.0
    return Fill(
        side=side,
        requested_quantity=req,
        quantity=qty,
        price=float(price),
        value_inr=value,
        participation=part,
        impact_bps=bps,
        impact_inr=value * bps / 1e4,
        statutory_inr=stat,
    )


def buy_cash_needed(value_inr: float, adv_value_inr: float, date: DateLike, cfg: CostConfig) -> float:
    """Cash needed for a buy of ``value_inr`` including impact and charges."""
    if value_inr <= 0:
        return 0.0
    return value_inr * (1 + impact_bps(value_inr, adv_value_inr, cfg) / 1e4) + statutory_cost(
        value_inr, "BUY", date, cfg.dp_charge_inr
    )


def median_traded_value(value: pd.DataFrame, lookback: int, min_periods: Optional[int] = None) -> pd.DataFrame:
    """Rolling median of traded value over the last ``lookback`` rows (inclusive).

    Untraded days count as zero value.  Row ``t`` uses rows ``<= t`` only; the
    simulator reads the row of the decision date for a fill on a later day.
    """
    mp = min_periods if min_periods is not None else max(1, lookback // 4)
    return value.astype("float64").fillna(0.0).rolling(lookback, min_periods=mp).median()
