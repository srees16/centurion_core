"""
Pre-trade report: everything to read before a basket goes out.

From a leg spec ("BUY CE 25000, SELL CE 25200") and live quotes it builds the
orders (contract, quantity, reference and LIMIT price) and reports:

* legs, lots, quantity, reference and limit prices;
* net premium, max profit and max loss in rupees, breakevens (``strategies``);
* net Greeks of the position (``theory``, from the chain's IVs);
* the expected range at 1, 2 and 3 SD to expiry (M5 ch. 18, linear method,
  from the ATM IV) and where each breakeven sits in it;
* margin and charges from Kite's basket-margin API beside the toolkit's own
  F&O cost model (``fno_costs``);
* breaches of the hard limits (``LimitsConfig``); a basket with any breach
  is refused by the executor.

Reference price: the ask for a buy and the bid for a sell (the last price
when that side is empty).  LIMIT price: the reference moved by
``limit_slippage_cap`` against the order, on the contract's tick.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

from kite_connect.options.fno_costs import FnoCharges, charges_from_kite, legs_costs
from kite_connect.options.instruments import Contract, InstrumentResolver
from kite_connect.options.live_chain import quote_prices
from kite_connect.options.options_config import LimitsConfig, OptionsConfig
from kite_connect.options.strategies import Leg, Strategy
from kite_connect.options.theory import BUY, expected_range

LegSpec = Tuple[str, str, float, int]           # (side, option type, strike, ratio)
_SPEC = re.compile(r"^\s*(BUY|SELL)\s+(?:(\d+)\s*[xX]?\s+)?(CE|PE)\s+([\d.]+)\s*$", re.IGNORECASE)


def parse_legs(text: str) -> List[LegSpec]:
    """``"BUY CE 25000, SELL 2 CE 25200"`` -> [(side, type, strike, ratio), ...]."""
    out = []
    for part in text.split(","):
        m = _SPEC.match(part)
        if not m:
            raise ValueError(f"cannot read leg {part.strip()!r}: use 'BUY|SELL [ratio] CE|PE strike'")
        out.append((m.group(1).upper(), m.group(3).upper(), float(m.group(4)), int(m.group(2) or 1)))
    if not out:
        raise ValueError("no legs")
    return out


def limit_price(reference: float, side: str, cap: float, tick: float) -> float:
    """The reference moved ``cap`` against the order, on the tick: buys round up, sells down (never below one tick)."""
    if side == BUY:
        return round(math.ceil(reference * (1 + cap) / tick - 1e-9) * tick, 2)
    return round(max(math.floor(reference * (1 - cap) / tick + 1e-9) * tick, tick), 2)


@dataclass(frozen=True)
class LegOrder:
    contract: Contract
    side: str
    ratio: int
    quantity: int
    reference: float
    limit: float
    iv: float = math.nan

    @property
    def label(self) -> str:
        return f"{self.side} {self.quantity} {self.contract.tradingsymbol} @ {self.limit:.2f}"

    def kite_order(self) -> dict:
        """The order as Kite's margin APIs take it."""
        return {"exchange": self.contract.exchange, "tradingsymbol": self.contract.tradingsymbol,
                "transaction_type": self.side, "variety": "regular", "product": "NRML", "order_type": "LIMIT",
                "quantity": self.quantity, "price": self.limit}


def order_legs(spec: Sequence[LegSpec], resolver: InstrumentResolver, underlying: str, expiry: date,
               quotes: Dict[str, dict], lots: int, cap: float, ivs: Optional[Dict[str, float]] = None) -> List[LegOrder]:
    """Contracts, quantities and prices for a leg spec; raises when a leg has no price to trade at."""
    legs = []
    for side, option_type, strike, ratio in spec:
        c = resolver.resolve(underlying, expiry, strike, option_type)
        p = quote_prices(quotes.get(c.quote_key))
        ref = p["ask"] if side == BUY else p["bid"]
        if not (ref and ref > 0):
            ref = p["ltp"]
        if not (ref and ref > 0):
            raise ValueError(f"{c.tradingsymbol}: no bid, ask or last price to trade at")
        legs.append(LegOrder(c, side, ratio, ratio * lots * c.lot_size, ref, limit_price(ref, side, cap, c.tick_size),
                             (ivs or {}).get(c.tradingsymbol, math.nan)))
    return legs


def strategy_of(legs: Sequence[LegOrder]) -> Strategy:
    """The legs as a strategy priced at their reference prices (what the basket should cost)."""
    return Strategy([Leg(l.contract.option_type, l.side, l.contract.strike, l.reference, ratio=l.ratio,
                         iv=l.iv if math.isfinite(l.iv) else None) for l in legs])


@dataclass
class PreTradeReport:
    underlying: str
    expiry: date
    days_to_expiry: float
    spot: float
    lots: int
    legs: List[LegOrder]
    strategy: Strategy
    units: int                                   # units per ratio: lots x lot size
    atm_iv: float
    rate: float
    margins: Optional[dict] = None
    cost_model: Optional[Dict[str, object]] = None
    violations: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def rupees(self, per_unit: float) -> float:
        return per_unit * self.units

    @property
    def net_premium_inr(self) -> float:
        """Premium paid (+) or received (-) for the basket."""
        return self.rupees(self.strategy.net_debit)

    @property
    def max_profit_inr(self) -> float:
        return self.rupees(self.strategy.max_profit)

    @property
    def max_loss_inr(self) -> float:
        return self.rupees(self.strategy.max_loss)

    def greeks(self) -> Optional[Dict[str, float]]:
        """Position Greeks: delta in units of the underlying, theta in rupees a day, vega in rupees a vol point."""
        if any(not math.isfinite(l.iv) for l in self.legs):
            return None
        g = self.strategy.net_greeks(self.spot, self.days_to_expiry, self.rate)
        return {"delta": g.delta * self.units, "gamma": g.gamma * self.units, "theta": g.theta * self.units,
                "vega": g.vega * self.units}

    def ranges(self) -> Dict[float, Tuple[float, float]]:
        """Expected spot range at 1, 2 and 3 SD by expiry, from the ATM IV (M5 ch. 18, linear, no drift)."""
        if not math.isfinite(self.atm_iv):
            return {}
        return expected_range(self.spot, self.atm_iv / math.sqrt(365), self.days_to_expiry, 0.0, (1, 2, 3))

    def breakeven_sd(self) -> List[Tuple[float, float]]:
        """(breakeven, its distance from spot in SDs of the period)."""
        if not math.isfinite(self.atm_iv):
            return []
        sd = self.spot * self.atm_iv / math.sqrt(365) * math.sqrt(self.days_to_expiry)
        return [(be, (be - self.spot) / sd) for be in self.strategy.breakevens()]

    def kite_charges(self) -> Optional[FnoCharges]:
        orders = (self.margins or {}).get("orders") or []
        return charges_from_kite(orders) if orders else None

    def text(self) -> str:
        """The report as printed before any order (and in the live confirmation)."""
        def inr(x: float) -> str:
            return "unlimited" if math.isinf(x) else f"Rs {x:,.0f}"

        lines = [f"PRE-TRADE REPORT  {self.underlying} {self.expiry} ({self.days_to_expiry:.1f} days)  "
                 f"spot {self.spot:,.2f}  lots {self.lots}",
                 f"{'side':5s} {'qty':>6s} {'contract':24s} {'ref':>9s} {'limit':>9s} {'IV':>7s}"]
        for l in self.legs:
            iv = f"{l.iv:.1%}" if math.isfinite(l.iv) else "n/a"
            lines.append(f"{l.side:5s} {l.quantity:>6d} {l.contract.tradingsymbol:24s} {l.reference:>9.2f} "
                         f"{l.limit:>9.2f} {iv:>7s}")
        lines.append(f"Net premium {'paid' if self.net_premium_inr >= 0 else 'received'}: "
                     f"{inr(abs(self.net_premium_inr))}   max profit {inr(self.max_profit_inr)}   "
                     f"max loss {inr(self.max_loss_inr)}")
        bes = ", ".join(f"{be:,.2f} ({sd:+.2f} SD)" for be, sd in self.breakeven_sd()) or \
            ", ".join(f"{be:,.2f}" for be in self.strategy.breakevens()) or "none"
        lines.append(f"Breakevens: {bes}")
        g = self.greeks()
        if g:
            lines.append(f"Net Greeks: delta {g['delta']:+.1f} units, gamma {g['gamma']:+.3f}, "
                         f"theta {g['theta']:+,.0f} Rs/day, vega {g['vega']:+,.0f} Rs/vol pt")
        r = self.ranges()
        if r:
            lines.append(f"Expected range by expiry (ATM IV {self.atm_iv:.1%}): " +
                         "; ".join(f"{k} SD {lo:,.0f}-{hi:,.0f}" for k, (lo, hi) in r.items()))
        if self.margins:
            fin = (self.margins.get("final") or {}).get("total")
            ini = (self.margins.get("initial") or {}).get("total")
            lines.append(f"Kite basket margin: {inr(fin) if fin is not None else 'n/a'} "
                         f"(without hedge benefit {inr(ini) if ini is not None else 'n/a'})")
        if self.cost_model:
            c = self.cost_model["charges"]
            lines.append(f"Entry costs (model): charges {inr(c.total)} (STT {inr(c.stt)}, brokerage {inr(c.brokerage)}), "
                         f"slippage {inr(self.cost_model['slippage_inr'])}")
        kc = self.kite_charges()
        if kc:
            lines.append(f"Entry charges (Kite): {inr(kc.total)} (STT {inr(kc.stt)}, brokerage {inr(kc.brokerage)})")
        lines.append("Limits: " + ("all within" if self.ok else "BREACHED: " + "; ".join(self.violations)))
        return "\n".join(lines)


def check_limits(report: PreTradeReport, limits: LimitsConfig) -> List[str]:
    """The hard limits a basket must keep (config, never tuned per trade)."""
    out = []
    if report.underlying not in limits.allowed_underlyings:
        out.append(f"{report.underlying} is not an allowed underlying {list(limits.allowed_underlyings)}")
    for l in report.legs:
        if l.ratio * report.lots > limits.max_lots_per_leg:
            out.append(f"{l.contract.tradingsymbol}: {l.ratio * report.lots} lots > {limits.max_lots_per_leg}")
    if math.isinf(report.max_loss_inr):
        out.append("unlimited maximum loss (a naked short)")
    elif report.max_loss_inr > limits.max_loss_per_trade_inr:
        out.append(f"max loss Rs {report.max_loss_inr:,.0f} > Rs {limits.max_loss_per_trade_inr:,.0f}")
    return out


def build_report(legs: List[LegOrder], spot: float, days: float, atm_iv: float, lots: int, today: date,
                 cfg: OptionsConfig = OptionsConfig(), margins: Optional[dict] = None) -> PreTradeReport:
    first = legs[0].contract
    units = lots * first.lot_size
    strategy = strategy_of(legs)
    report = PreTradeReport(first.underlying, first.expiry, days, spot, lots, legs, strategy, units, atm_iv,
                            cfg.market.risk_free_rate, margins=margins,
                            cost_model=legs_costs(strategy.legs, units, today, first.underlying, cfg))
    report.violations = check_limits(report, cfg.limits)
    return report
