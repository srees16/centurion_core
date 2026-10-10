"""
Positions opened by the toolkit and their live state.

``PositionLedger`` keeps every basket the executor filled (paper or live) in
``data/options/positions.json``: legs with filled quantities and prices, the
spot and ATM IV at entry.  ``monitor`` marks each open position to live
quotes and reports:

* P&L in rupees and the position's Greeks (IVs solved from today's prices:
  a leg's own, else its mirror's - the other option at its strike, which an
  ITM leg needs when its price sits below the no-arbitrage bound - else the
  entry ATM IV, flagged approximate; tracker LN-T27);
* **breakeven**: spot has crossed a breakeven from the profit side it was
  on at entry (the expiry payoff at today's spot is now a loss), or, still
  on the profit side, is within ``NEAR_BREAKEVEN`` of one;
* **volatility stop-loss** (M5 ch. 18): spot beyond entry spot x (1 -/+
  daily sigma x sqrt(days held)), against the position's delta, with the
  daily sigma from the entry ATM IV;
* **max loss**: the loss has reached ``MAX_LOSS_ALERT`` of the basket's
  maximum loss.

It alerts; it never trades.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from kite_connect.options.basket_executor import ExecutionReport
from kite_connect.options.instruments import spot_key
from kite_connect.options.live_chain import days_to_expiry, quote_prices
from kite_connect.options.pretrade import PreTradeReport
from kite_connect.options.strategies import Leg, Strategy
from kite_connect.options.theory import (BUY, CALL, PUT, SELL, black_scholes, implied_volatility,
                                         volatility_stop_loss)

DEFAULT_LEDGER = Path("data/options/positions.json")
NEAR_BREAKEVEN = 0.005
MAX_LOSS_ALERT = 0.8


@dataclass
class PositionLeg:
    tradingsymbol: str
    option_type: str
    strike: float
    side: str
    quantity: int
    price: float

    @property
    def sign(self) -> int:
        return 1 if self.side == BUY else -1


@dataclass
class OptionsPosition:
    id: str
    mode: str
    opened_at: str
    underlying: str
    expiry: str
    legs: List[PositionLeg]
    entry_spot: float
    entry_atm_iv: float
    max_loss_inr: Optional[float]
    status: str = "open"

    def strategy(self) -> Strategy:
        """Per-unit legs priced at the fills, for breakevens and the expiry payoff (quantities relative to the smallest)."""
        base = min(l.quantity for l in self.legs)
        return Strategy([Leg(l.option_type, l.side, l.strike, l.price, ratio=max(1, round(l.quantity / base)))
                         for l in self.legs])

    @property
    def units(self) -> int:
        return min(l.quantity for l in self.legs)


class PositionLedger:
    def __init__(self, path: Path = DEFAULT_LEDGER):
        self.path = Path(path)
        raw = json.loads(self.path.read_text()) if self.path.exists() else []
        self.positions = [OptionsPosition(**{**p, "legs": [PositionLeg(**l) for l in p["legs"]]}) for p in raw]

    def open_positions(self) -> List[OptionsPosition]:
        return [p for p in self.positions if p.status == "open"]

    def record(self, report: PreTradeReport, execution: ExecutionReport, opened_at: datetime) -> Optional[OptionsPosition]:
        """Store the filled part of a basket (nothing for a dry run or when nothing filled)."""
        legs = [PositionLeg(r.leg.contract.tradingsymbol, r.leg.contract.option_type, r.leg.contract.strike,
                            r.leg.side, r.filled, r.average_price) for r in execution.results if r.filled]
        if not legs:
            return None
        pos = OptionsPosition(uuid.uuid4().hex[:8], execution.mode, opened_at.isoformat(), report.underlying,
                              report.expiry.isoformat(), legs, report.spot, report.atm_iv,
                              None if math.isinf(report.max_loss_inr) else report.max_loss_inr)
        self.positions.append(pos)
        self.save()
        return pos

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([asdict(p) for p in self.positions], indent=1, default=str))


def mirror_symbol(tradingsymbol: str) -> str:
    """The other option at the same strike and expiry (NFO symbols end in CE or PE)."""
    return tradingsymbol[:-2] + (PUT if tradingsymbol.endswith(CALL) else CALL)


@dataclass
class PositionStatus:
    position: OptionsPosition
    spot: float
    pnl_inr: float
    greeks: Optional[Dict[str, float]]
    alerts: List[str] = field(default_factory=list)
    approximate: bool = False                 # a leg's Greeks use the entry ATM IV

    def text(self) -> str:
        p = self.position
        g = self.greeks
        greeks = (f"delta {g['delta']:+.1f}, theta {g['theta']:+,.0f} Rs/day, vega {g['vega']:+,.0f} Rs/vol pt"
                  + (" (approximate: entry ATM IV)" if self.approximate else "") if g else "Greeks n/a")
        head = (f"[{p.mode}] {p.id} {p.underlying} {p.expiry} " + ", ".join(f"{l.side} {l.quantity} {l.tradingsymbol}"
                                                                           for l in p.legs))
        lines = [head, f"  spot {self.spot:,.2f}  P&L Rs {self.pnl_inr:+,.0f}  {greeks}"]
        lines += [f"  ALERT {a}" for a in self.alerts] or ["  no alerts"]
        return "\n".join(lines)


def _leg_iv(leg: PositionLeg, prices: Dict[str, float], spot: float, dte: float, rate: float,
            entry_atm_iv: float) -> Tuple[Optional[float], bool]:
    """(IV, approximate) of a priced leg: its own, else its mirror's, else the entry ATM IV (approximate)."""
    mirror = (PUT if leg.option_type == CALL else CALL, prices.get(mirror_symbol(leg.tradingsymbol), math.nan))
    for option_type, px in ((leg.option_type, prices[leg.tradingsymbol]), mirror):
        if math.isfinite(px) and px > 0:
            iv = implied_volatility(option_type, px, spot, leg.strike, dte, rate)
            if iv is not None:
                return iv, False
    return (entry_atm_iv, True) if math.isfinite(entry_atm_iv) and entry_atm_iv > 0 else (None, False)


def evaluate(pos: OptionsPosition, prices: Dict[str, float], spot: float, now: datetime, rate: float) -> PositionStatus:
    """One position against today's leg prices (with each leg's mirror, when quoted) and spot."""
    pnl = sum(l.sign * l.quantity * (prices[l.tradingsymbol] - l.price) for l in pos.legs
              if math.isfinite(prices.get(l.tradingsymbol, math.nan)))
    dte = days_to_expiry(date.fromisoformat(pos.expiry), now)
    greeks: Optional[Dict[str, float]] = {"delta": 0.0, "theta": 0.0, "vega": 0.0}
    approximate = False
    for l in pos.legs:
        px = prices.get(l.tradingsymbol, math.nan)
        iv, approx = (_leg_iv(l, prices, spot, dte, rate, pos.entry_atm_iv) if math.isfinite(px) and px > 0
                      else (None, False))
        if iv is None:
            greeks = None
            break
        approximate = approximate or approx
        g = black_scholes(l.option_type, spot, l.strike, dte, rate, iv)
        for k in greeks:
            greeks[k] += l.sign * l.quantity * getattr(g, k)

    alerts: List[str] = []
    strat = pos.strategy()
    expiry_pnl = strat.payoff(spot) * pos.units
    bes = strat.breakevens()
    if bes and expiry_pnl < 0 <= strat.payoff(pos.entry_spot):
        alerts.append(f"spot {spot:,.2f} has crossed a breakeven ({', '.join(f'{b:,.2f}' for b in bes)}): "
                      f"held to expiry here the basket loses Rs {-expiry_pnl:,.0f}")
    elif expiry_pnl >= 0:
        near = [b for b in bes if abs(spot - b) <= NEAR_BREAKEVEN * spot]
        if near:
            alerts.append(f"spot {spot:,.2f} is within {NEAR_BREAKEVEN:.1%} of breakeven {near[0]:,.2f}")
    if greeks and math.isfinite(pos.entry_atm_iv) and greeks["delta"] != 0:
        held = max((now - datetime.fromisoformat(pos.opened_at)).total_seconds() / 86400.0, 1.0)
        side = BUY if greeks["delta"] > 0 else SELL
        stop = volatility_stop_loss(pos.entry_spot, pos.entry_atm_iv / math.sqrt(365), held, side)
        if (side == BUY and spot < stop) or (side == SELL and spot > stop):
            alerts.append(f"volatility stop-loss: spot {spot:,.2f} beyond {stop:,.2f} "
                          f"(entry {pos.entry_spot:,.2f}, {held:.0f} day(s), M5 ch. 18)")
    if pos.max_loss_inr and pnl <= -MAX_LOSS_ALERT * pos.max_loss_inr:
        alerts.append(f"loss Rs {-pnl:,.0f} is {-pnl / pos.max_loss_inr:.0%} of the maximum Rs {pos.max_loss_inr:,.0f}")
    return PositionStatus(pos, spot, pnl, greeks, alerts, approximate=approximate and greeks is not None)


def monitor(broker, ledger: PositionLedger, now: datetime, rate: float) -> List[PositionStatus]:
    """Every open position marked to live quotes (mid, else last price), each leg's mirror quoted too."""
    out = []
    for pos in ledger.open_positions():
        symbols = [l.tradingsymbol for l in pos.legs] + [mirror_symbol(l.tradingsymbol) for l in pos.legs]
        quotes = broker.quotes([f"NFO:{s}" for s in symbols] + [spot_key(pos.underlying)])
        prices = {s: quote_prices(quotes.get(f"NFO:{s}"))["mid"] for s in symbols}
        spot = float((quotes.get(spot_key(pos.underlying)) or {}).get("last_price") or math.nan)
        out.append(evaluate(pos, prices, spot, now, rate))
    return out
