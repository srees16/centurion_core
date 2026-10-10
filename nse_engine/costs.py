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

STT_RATE = 0.001  # 0.1% on buy and sell (delivery), from 2012-07-01 and before 2006-06-01
STT_RATE_2006_2012 = 0.00125  # 0.125%: Finance Act 2006 (from 2006-06-01) until Finance Act 2012 (2012-07-01)
STT_HIGH_START = pd.Timestamp("2006-06-01")
STT_HIGH_END = pd.Timestamp("2012-07-01")
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

# ETFs the engine trades (the metal sleeves), tracker U25, 30 Sep 2026.  Delivery
# STT on equity shares is 0.1% a side, but ETFs differ (Zerodha, "Is STT levied
# on ETFs?"): none on gold, liquid and gilt ETFs; 0.001% on the sell side only
# for other ETFs.  Silver ETFs are non-equity funds and reported exempt, but
# Zerodha's page does not name them, so they are charged the other-ETF rate
# (Rs 1 per lakh sold): within a rupee of exempt either way.
STT_EXEMPT_ETFS = frozenset({"GOLDBEES"})
STT_OTHER_ETFS = frozenset({"SILVERBEES", "MON100"})   # MON100 (tracker R14): an equity ETF, the other-ETF rate
STT_OTHER_ETF_SELL = 0.00001
#: Recorded in every run's manifest; runs are compared only within one version.
#: 1 = until 30 Sep 2026 (equity delivery STT on the metal ETFs too); 2 = the ETF rates above;
#: 3 = ETF opens held within ETF_OPEN_BAND of the close for fills (tracker D4, 1 Oct 2026);
#: 4 = idle cash earns IDLE_CASH_YIELD_ANNUAL, nothing (tracker IC1, 10 Oct 2026).
COST_MODEL_VERSION = 4

#: What idle cash earns in the account, a year.  A Kite account pays no interest, and
#: neither the paper nor the live book sweeps cash into a liquid fund, so: nothing.
#: Cost models 1-3 credited EngineConfig.cash_yield_annual (6%), which no live path earns
#: (tracker IC1).  A future cash sweep would set this from the fund's dated net yield.
IDLE_CASH_YIELD_ANNUAL = 0.0

# An ETF's open is often a stray first trade of a few units far from where it
# traded all day (GOLDBEES 24 Dec 2020: open 48.98, low 43.58, close 43.73).  In
# 2013-25 the metal ETFs opened more than 5% from their close on 1-2% of days, 88%
# of those at the day's high or low, while gold's close-to-close move passed 2.95%
# on 1% of days.  Shares open in the call auction and keep their open.
ETF_OPEN_BAND = 0.03


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


def stt_rate(date: DateLike) -> float:
    """Delivery STT per side: 0.125% from 1 Jun 2006 to 30 Jun 2012, 0.1% otherwise."""
    d = _ts(date)
    return STT_RATE_2006_2012 if STT_HIGH_START <= d < STT_HIGH_END else STT_RATE


def stamp_duty_rate(date: DateLike) -> float:
    return STAMP_DUTY_RATE_NEW if _ts(date) >= STAMP_DUTY_CHANGE else STAMP_DUTY_RATE_OLD


def exchange_rate(date: DateLike) -> float:
    return EXCHANGE_RATE_NEW if _ts(date) >= EXCHANGE_CHANGE else EXCHANGE_RATE_OLD


def indirect_tax_rate(date: DateLike) -> float:
    """GST from 2017-07-01, service tax before."""
    return GST_RATE if _ts(date) >= GST_START else SERVICE_TAX_RATE


def stt_for(symbol: Optional[str], side: str, date: DateLike) -> float:
    """Delivery STT rate for one side: the equity schedule, or the ETF rate for the sleeves."""
    if symbol is not None:
        if symbol in STT_EXEMPT_ETFS:
            return 0.0
        if symbol in STT_OTHER_ETFS:
            return STT_OTHER_ETF_SELL if side == "SELL" else 0.0
    return stt_rate(date)


def statutory_charges(
    value_inr: float, side: str, date: DateLike, dp_charge_inr: float = CostConfig.dp_charge_inr,
    symbol: Optional[str] = None,
) -> StatutoryCharges:
    """Statutory charges for one side of a delivery trade of ``value_inr``.

    ``symbol`` selects the ETF STT rates (``stt_for``); None = equity shares.
    """
    side = side.upper()
    if side not in ("BUY", "SELL"):
        raise ValueError(f"side must be BUY or SELL, got {side!r}")
    value = abs(float(value_inr))
    if value <= 0:
        return StatutoryCharges(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    stt = stt_for(symbol, side, date) * value
    stamp = stamp_duty_rate(date) * value if side == "BUY" else 0.0
    exch = exchange_rate(date) * value
    sebi = SEBI_FEE_RATE * value
    gst = indirect_tax_rate(date) * (exch + sebi)
    dp = float(dp_charge_inr) if side == "SELL" else 0.0
    return StatutoryCharges(stt, stamp, exch, sebi, gst, dp)


def statutory_cost(
    value_inr: float, side: str, date: DateLike, dp_charge_inr: float = CostConfig.dp_charge_inr,
    symbol: Optional[str] = None,
) -> float:
    """Total statutory charges in INR (see :func:`statutory_charges`)."""
    return statutory_charges(value_inr, side, date, dp_charge_inr, symbol=symbol).total


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
    symbol: Optional[str] = None,
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
    stat = statutory_cost(value, side, date, cfg.dp_charge_inr, symbol=symbol) if qty > 0 else 0.0
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


def buy_cash_needed(value_inr: float, adv_value_inr: float, date: DateLike, cfg: CostConfig,
                    symbol: Optional[str] = None) -> float:
    """Cash needed for a buy of ``value_inr`` including impact and charges."""
    if value_inr <= 0:
        return 0.0
    return value_inr * (1 + impact_bps(value_inr, adv_value_inr, cfg) / 1e4) + statutory_cost(
        value_inr, "BUY", date, cfg.dp_charge_inr, symbol=symbol
    )


def fillable_open(open_: pd.DataFrame, close: pd.DataFrame, etfs) -> pd.DataFrame:
    """Opens that orders fill at: an ETF's open held within ``ETF_OPEN_BAND`` of the day's close."""
    cols = [s for s in open_.columns if s in etfs]
    if not cols:
        return open_
    o = open_[cols]
    lo, hi = close[cols] * (1 - ETF_OPEN_BAND), close[cols] * (1 + ETF_OPEN_BAND)
    out = open_.copy()
    out[cols] = o.mask(o > hi, hi).mask(o < lo, lo)
    return out


def median_traded_value(value: pd.DataFrame, lookback: int, min_periods: Optional[int] = None) -> pd.DataFrame:
    """Rolling median of traded value over the last ``lookback`` rows (inclusive).

    Untraded days count as zero value.  Row ``t`` uses rows ``<= t`` only; the
    simulator reads the row of the decision date for a fill on a later day.
    """
    mp = min_periods if min_periods is not None else max(1, lookback // 4)
    return value.astype("float64").fillna(0.0).rolling(lookback, min_periods=mp).median()
