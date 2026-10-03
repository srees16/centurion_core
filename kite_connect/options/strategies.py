"""
Strategy engine: Zerodha Varsity Module 6, chapters 2 to 13 (see STRATEGIES.md).

A :class:`Strategy` is a list of :class:`Leg` objects; everything it reports
(net premium, max profit and loss, breakevens, payoff table and chart, net
Greeks) is computed from the legs.  Each named strategy also gives the PDF's
closed-form ``generalization()``, and ``check_generalization()`` compares the
two, so a formula and the summed leg payoffs can never silently disagree.

Values are per unit of underlying (+ = money in); multiply by lot size x lots
for rupees.  Spot is bounded below by 0, so a "downside unlimited" payoff is
reported at its finite value at spot 0.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from datetime import date
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from kite_connect.options.theory import (BUY, CALL, PUT, SELL, Greeks, black_scholes, intrinsic_value,
                                         position_greeks, side_sign)

FUT = "FUT"
_TOL = 1e-9


@dataclass(frozen=True)
class Leg:
    """One option (CE/PE) or futures (FUT) position.

    For FUT, ``strike`` is the traded futures price and ``premium`` is 0.
    ``greeks`` overrides Black-Scholes with given per-unit Greeks (the PDF's
    examples quote deltas without the inputs to price them).
    """

    option_type: str
    side: str
    strike: float
    premium: float = 0.0
    ratio: int = 1
    expiry: Optional[date] = None
    iv: Optional[float] = None
    greeks: Optional[Greeks] = None

    def __post_init__(self):
        object.__setattr__(self, "option_type", self.option_type.upper())
        object.__setattr__(self, "side", self.side.upper())
        if self.option_type not in (CALL, PUT, FUT):
            raise ValueError(f"option_type must be CE, PE or FUT, got {self.option_type!r}")
        side_sign(self.side)
        if self.ratio < 1:
            raise ValueError("ratio must be >= 1")

    @property
    def sign(self) -> int:
        return side_sign(self.side)

    @property
    def label(self) -> str:
        x = f"{self.ratio}x " if self.ratio > 1 else ""
        return f"{self.side} {x}{self.strike:g} {self.option_type}"

    def payoff(self, spot):
        """P&L per unit at expiry for this leg (times its ratio)."""
        s = np.asarray(spot, dtype=float)
        if self.option_type == FUT:
            v = s - self.strike
        else:
            v = intrinsic_value(self.option_type, s, self.strike) - self.premium
        return self.sign * self.ratio * v

    def unit_greeks(self, spot: float, days_to_expiry: float, rate: float,
                    dividend: float = 0.0, days_per_year: int = 365) -> Greeks:
        """Per-unit, long-side Greeks (the sign is applied by the strategy)."""
        if self.greeks is not None:
            return self.greeks
        if self.option_type == FUT:
            return Greeks(price=spot, delta=1.0)
        if self.iv is None:
            raise ValueError(f"{self.label}: no iv and no given greeks")
        return black_scholes(self.option_type, spot, self.strike, days_to_expiry, rate, self.iv,
                             dividend, days_per_year)


class Strategy:
    """A position of legs on one underlying and expiry."""

    name = "Custom"
    chapter: Optional[int] = None

    def __init__(self, legs: Sequence[Leg], name: Optional[str] = None):
        if not legs:
            raise ValueError("a strategy needs at least one leg")
        self.legs: Tuple[Leg, ...] = tuple(legs)
        if name:
            self.name = name

    # ---- premium
    @property
    def net_debit(self) -> float:
        """Premium paid - received; negative for a net credit."""
        return float(sum(l.sign * l.ratio * l.premium for l in self.legs if l.option_type != FUT))

    @property
    def net_credit(self) -> float:
        return -self.net_debit

    @property
    def is_credit(self) -> bool:
        return self.net_debit < 0

    # ---- payoff
    def payoff(self, spot):
        s = np.asarray(spot, dtype=float)
        total = sum(l.payoff(s) for l in self.legs)
        return float(total) if np.ndim(total) == 0 else total

    def kinks(self) -> List[float]:
        return sorted({float(l.strike) for l in self.legs if l.option_type != FUT})

    def _slope_right(self) -> float:
        return float(sum(l.sign * l.ratio for l in self.legs if l.option_type in (CALL, FUT)))

    def _points(self) -> Tuple[List[float], List[float]]:
        xs = [0.0] + [k for k in self.kinks() if k > 0]
        return xs, [self.payoff(x) for x in xs]

    @property
    def max_profit(self) -> float:
        if self._slope_right() > _TOL:
            return math.inf
        return float(max(self._points()[1]))

    @property
    def max_loss(self) -> float:
        """Largest loss as a positive number (negative = a profit at every spot)."""
        if self._slope_right() < -_TOL:
            return math.inf
        return float(-min(self._points()[1]))

    def _region(self, value: float) -> Optional[Tuple[float, float]]:
        xs, vs = self._points()
        hits = [x for x, v in zip(xs, vs) if abs(v - value) <= 1e-7]
        return (min(hits), max(hits)) if hits else None

    def max_loss_region(self) -> Optional[Tuple[float, float]]:
        """Spot interval where the (finite) max loss occurs; (k, k) for a single point."""
        return None if math.isinf(self.max_loss) else self._region(-self.max_loss)

    def max_profit_region(self) -> Optional[Tuple[float, float]]:
        return None if math.isinf(self.max_profit) else self._region(self.max_profit)

    def breakevens(self) -> List[float]:
        xs, vs = self._points()
        out: List[float] = []
        for (x0, v0), (x1, v1) in zip(zip(xs, vs), zip(xs[1:], vs[1:])):
            if abs(v0) <= _TOL and x0 > 0:
                out.append(x0)
            elif v0 * v1 < 0 and abs(v1) > _TOL:
                out.append(x0 + (x1 - x0) * (-v0) / (v1 - v0))
        x_last, v_last, slope = xs[-1], vs[-1], self._slope_right()
        if abs(v_last) <= _TOL and x_last > 0:
            out.append(x_last)
        elif abs(slope) > _TOL and -v_last / slope > 0:
            out.append(x_last - v_last / slope)
        return sorted({round(x, 9) for x in out})

    def payoff_table(self, low: float, high: float, step: float) -> pd.DataFrame:
        spots = np.arange(low, high + step / 2, step)
        df = pd.DataFrame({"spot": spots})
        for i, leg in enumerate(self.legs):
            df[f"{i + 1}: {leg.label}"] = leg.payoff(spots)
        df["net"] = self.payoff(spots)
        return df

    def payoff_chart(self, low: float, high: float, step: float = 1.0, path: Optional[str] = None):
        """Expiry payoff with breakevens marked; saved to ``path`` when given."""
        import matplotlib
        if path:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        spots = np.arange(low, high + step / 2, step)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(spots, self.payoff(spots), lw=2)
        ax.axhline(0, color="grey", lw=0.8)
        for be in self.breakevens():
            ax.axvline(be, color="tab:orange", ls="--", lw=0.8)
        ax.set_title(f"{self.name}: " + ", ".join(l.label for l in self.legs))
        ax.set_xlabel("spot at expiry")
        ax.set_ylabel("P&L per unit")
        if path:
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
        return fig

    # ---- Greeks
    def with_ivs(self, ivs: Sequence[Optional[float]]) -> "Strategy":
        """A copy with each leg's implied volatility set (one per leg, in order)."""
        if len(ivs) != len(self.legs):
            raise ValueError(f"{len(ivs)} ivs for {len(self.legs)} legs")
        out = copy.copy(self)
        out.legs = tuple(replace(l, iv=iv) for l, iv in zip(self.legs, ivs))
        return out

    def net_greeks(self, spot: float, days_to_expiry: float, rate: float,
                   dividend: float = 0.0, days_per_year: int = 365) -> Greeks:
        """Signed sum of the legs' Greeks per unit; ``price`` is the position's value."""
        return position_greeks((l.sign * l.ratio, l.unit_greeks(spot, days_to_expiry, rate, dividend, days_per_year))
                               for l in self.legs)

    # ---- the PDF's formulas
    def conditions(self) -> List[str]:
        """Violated preconditions of ``generalization()`` (empty = it applies)."""
        return []

    def generalization(self) -> Dict[str, object]:
        raise NotImplementedError(f"{self.name} has no closed-form generalization")

    def leg_values(self) -> Dict[str, object]:
        return {"net_debit": self.net_debit, "net_credit": self.net_credit, "max_profit": self.max_profit,
                "max_loss": self.max_loss, "breakevens": tuple(self.breakevens())}

    def check_generalization(self, tol: float = 1e-6) -> Dict[str, Tuple[object, object]]:
        """{key: (formula, legs)} for every PDF formula that disagrees with the legs."""
        bad = self.conditions()
        if bad:
            raise ValueError(f"{self.name}: formulas do not apply: {'; '.join(bad)}")
        legs = self.leg_values()
        out: Dict[str, Tuple[object, object]] = {}
        for key, formula in self.generalization().items():
            actual = legs[key]
            if key == "breakevens":
                f = tuple(sorted(formula))
                ok = len(f) == len(actual) and all(abs(a - b) <= tol for a, b in zip(f, actual))
            elif math.isinf(formula) or math.isinf(actual):
                ok = formula == actual
            else:
                ok = abs(formula - actual) <= tol
            if not ok:
                out[key] = (formula, actual)
        return out

    def summary(self) -> Dict[str, object]:
        return {"name": self.name, "legs": [l.label for l in self.legs], "net_debit": self.net_debit,
                "max_profit": self.max_profit, "max_profit_region": self.max_profit_region(),
                "max_loss": self.max_loss, "max_loss_region": self.max_loss_region(),
                "breakevens": self.breakevens()}

    def __repr__(self) -> str:
        return f"{self.name}({', '.join(l.label for l in self.legs)})"


def _two_strikes(k_low: float, k_high: float) -> None:
    if not k_low < k_high:
        raise ValueError(f"need k_low < k_high, got {k_low} and {k_high}")


class BullCallSpread(Strategy):
    """Ch. 2: buy the lower-strike call, sell the higher-strike call (net debit)."""

    name, chapter = "Bull Call Spread", 2

    def __init__(self, k_low: float, p_low: float, k_high: float, p_high: float, **leg_kw):
        _two_strikes(k_low, k_high)
        self.k_low, self.p_low, self.k_high, self.p_high = k_low, p_low, k_high, p_high
        super().__init__([Leg(CALL, BUY, k_low, p_low, **leg_kw), Leg(CALL, SELL, k_high, p_high, **leg_kw)])

    def generalization(self):
        debit = self.p_low - self.p_high
        return {"net_debit": debit, "max_loss": debit, "max_profit": (self.k_high - self.k_low) - debit,
                "breakevens": (self.k_low + debit,)}


class BullPutSpread(Strategy):
    """Ch. 3: buy the lower-strike put, sell the higher-strike put (net credit)."""

    name, chapter = "Bull Put Spread", 3

    def __init__(self, k_low: float, p_low: float, k_high: float, p_high: float, **leg_kw):
        _two_strikes(k_low, k_high)
        self.k_low, self.p_low, self.k_high, self.p_high = k_low, p_low, k_high, p_high
        super().__init__([Leg(PUT, BUY, k_low, p_low, **leg_kw), Leg(PUT, SELL, k_high, p_high, **leg_kw)])

    def generalization(self):
        credit = self.p_high - self.p_low
        return {"net_credit": credit, "max_profit": credit, "max_loss": (self.k_high - self.k_low) - credit,
                "breakevens": (self.k_high - credit,)}


class CallRatioBackSpread(Strategy):
    """Ch. 4: sell 1 lower-strike (ITM) call, buy 2 higher-strike (OTM) calls, for a net credit."""

    name, chapter = "Call Ratio Back Spread", 4

    def __init__(self, k_low: float, p_low: float, k_high: float, p_high: float, **leg_kw):
        _two_strikes(k_low, k_high)
        self.k_low, self.p_low, self.k_high, self.p_high = k_low, p_low, k_high, p_high
        super().__init__([Leg(CALL, SELL, k_low, p_low, **leg_kw), Leg(CALL, BUY, k_high, p_high, ratio=2, **leg_kw)])

    def conditions(self):
        return [] if self.net_credit > 0 else ["needs a net credit (ch. 4, ch. 9)"]

    def generalization(self):
        credit = self.p_low - 2 * self.p_high
        max_loss = (self.k_high - self.k_low) - credit
        return {"net_credit": credit, "max_loss": max_loss, "max_profit": math.inf,
                "breakevens": (self.k_low + credit, self.k_high + max_loss)}


class BearCallLadder(Strategy):
    """Ch. 5: sell 1 ITM call (K1), buy 1 ATM (K2) and 1 OTM call (K3), for a net credit."""

    name, chapter = "Bear Call Ladder", 5

    def __init__(self, k1: float, p1: float, k2: float, p2: float, k3: float, p3: float, **leg_kw):
        if not k1 < k2 < k3:
            raise ValueError("need k1 < k2 < k3")
        self.k1, self.p1, self.k2, self.p2, self.k3, self.p3 = k1, p1, k2, p2, k3, p3
        super().__init__([Leg(CALL, SELL, k1, p1, **leg_kw), Leg(CALL, BUY, k2, p2, **leg_kw),
                          Leg(CALL, BUY, k3, p3, **leg_kw)])

    def conditions(self):
        return [] if self.net_credit > 0 else ["needs a net credit (ch. 5)"]

    def generalization(self):
        credit = self.p1 - self.p2 - self.p3
        return {"net_credit": credit, "max_loss": (self.k2 - self.k1) - credit, "max_profit": math.inf,
                "breakevens": (self.k1 + credit, self.k2 + self.k3 - self.k1 - credit)}


class SyntheticLong(Strategy):
    """Ch. 6: buy the call and sell the put at one strike (mimics long futures)."""

    name, chapter = "Synthetic Long", 6

    def __init__(self, k: float, p_call: float, p_put: float, **leg_kw):
        self.k, self.p_call, self.p_put = k, p_call, p_put
        super().__init__([Leg(CALL, BUY, k, p_call, **leg_kw), Leg(PUT, SELL, k, p_put, **leg_kw)])

    def generalization(self):
        debit = self.p_call - self.p_put
        return {"net_debit": debit, "breakevens": (self.k + debit,), "max_profit": math.inf}


def synthetic_long_arbitrage(futures_price: float, strike: float, call_ask: float, put_bid: float,
                             days_to_expiry: float = 0.0, rate: float = 0.0, charges: float = 0.0,
                             buffer: float = 0.0, days_per_year: int = 365) -> Dict[str, object]:
    """Ch. 6: long call + short put + short futures, all at one strike and expiry.

    The PDF's P&L, (F - K) - (C - P), is the same at every expiry price.  The
    toolkit also takes off the cost of carrying the net premium to expiry and
    the charges (per unit), and flags an opportunity only above ``buffer``.
    Use executable prices: the call's ask and the put's bid.
    """
    legs = [Leg(CALL, BUY, strike, call_ask), Leg(PUT, SELL, strike, put_bid), Leg(FUT, SELL, futures_price)]
    position = Strategy(legs, "Synthetic Long Arbitrage")
    gross = (futures_price - strike) - (call_ask - put_bid)
    t = max(days_to_expiry, 0.0) / days_per_year
    after_carry = (futures_price - strike) - (call_ask - put_bid) * math.exp(rate * t)
    net = after_carry - charges
    return {"position": position, "gross": gross, "after_carry": after_carry, "net": net,
            "opportunity": net > buffer}


class BearPutSpread(Strategy):
    """Ch. 7: buy the higher-strike put, sell the lower-strike put (net debit)."""

    name, chapter = "Bear Put Spread", 7

    def __init__(self, k_low: float, p_low: float, k_high: float, p_high: float, **leg_kw):
        _two_strikes(k_low, k_high)
        self.k_low, self.p_low, self.k_high, self.p_high = k_low, p_low, k_high, p_high
        super().__init__([Leg(PUT, BUY, k_high, p_high, **leg_kw), Leg(PUT, SELL, k_low, p_low, **leg_kw)])

    def generalization(self):
        debit = self.p_high - self.p_low
        return {"net_debit": debit, "max_loss": debit, "max_profit": (self.k_high - self.k_low) - debit,
                "breakevens": (self.k_high - debit,)}


class BearCallSpread(Strategy):
    """Ch. 8: sell the lower-strike call, buy the higher-strike call (net credit)."""

    name, chapter = "Bear Call Spread", 8

    def __init__(self, k_low: float, p_low: float, k_high: float, p_high: float, **leg_kw):
        _two_strikes(k_low, k_high)
        self.k_low, self.p_low, self.k_high, self.p_high = k_low, p_low, k_high, p_high
        super().__init__([Leg(CALL, SELL, k_low, p_low, **leg_kw), Leg(CALL, BUY, k_high, p_high, **leg_kw)])

    def generalization(self):
        credit = self.p_low - self.p_high
        return {"net_credit": credit, "max_profit": credit, "max_loss": (self.k_high - self.k_low) - credit,
                "breakevens": (self.k_low + credit,)}


class PutRatioBackSpread(Strategy):
    """Ch. 9: sell 1 higher-strike (ITM) put, buy 2 lower-strike (OTM) puts, for a net credit."""

    name, chapter = "Put Ratio Back Spread", 9

    def __init__(self, k_low: float, p_low: float, k_high: float, p_high: float, **leg_kw):
        _two_strikes(k_low, k_high)
        self.k_low, self.p_low, self.k_high, self.p_high = k_low, p_low, k_high, p_high
        super().__init__([Leg(PUT, SELL, k_high, p_high, **leg_kw), Leg(PUT, BUY, k_low, p_low, ratio=2, **leg_kw)])

    def conditions(self):
        return [] if self.net_credit > 0 else ["needs a net credit (ch. 9)"]

    def generalization(self):
        credit = self.p_high - 2 * self.p_low
        max_loss = (self.k_high - self.k_low) - credit
        return {"net_credit": credit, "max_loss": max_loss,
                "breakevens": (self.k_low - max_loss, self.k_low + max_loss)}


class LongStraddle(Strategy):
    """Ch. 10: buy the ATM call and put."""

    name, chapter = "Long Straddle", 10

    def __init__(self, k: float, p_call: float, p_put: float, **leg_kw):
        self.k, self.p_call, self.p_put = k, p_call, p_put
        super().__init__([Leg(CALL, BUY, k, p_call, **leg_kw), Leg(PUT, BUY, k, p_put, **leg_kw)])

    def generalization(self):
        debit = self.p_call + self.p_put
        return {"net_debit": debit, "max_loss": debit, "max_profit": math.inf,
                "breakevens": (self.k - debit, self.k + debit)}


class ShortStraddle(Strategy):
    """Ch. 11: sell the ATM call and put."""

    name, chapter = "Short Straddle", 11

    def __init__(self, k: float, p_call: float, p_put: float, **leg_kw):
        self.k, self.p_call, self.p_put = k, p_call, p_put
        super().__init__([Leg(CALL, SELL, k, p_call, **leg_kw), Leg(PUT, SELL, k, p_put, **leg_kw)])

    def generalization(self):
        credit = self.p_call + self.p_put
        return {"net_credit": credit, "max_profit": credit, "max_loss": math.inf,
                "breakevens": (self.k - credit, self.k + credit)}


class LongStrangle(Strategy):
    """Ch. 12: buy an OTM put and an OTM call."""

    name, chapter = "Long Strangle", 12

    def __init__(self, k_put: float, p_put: float, k_call: float, p_call: float, **leg_kw):
        _two_strikes(k_put, k_call)
        self.k_put, self.p_put, self.k_call, self.p_call = k_put, p_put, k_call, p_call
        super().__init__([Leg(PUT, BUY, k_put, p_put, **leg_kw), Leg(CALL, BUY, k_call, p_call, **leg_kw)])

    def generalization(self):
        debit = self.p_put + self.p_call
        return {"net_debit": debit, "max_loss": debit, "max_profit": math.inf,
                "breakevens": (self.k_put - debit, self.k_call + debit)}


class ShortStrangle(Strategy):
    """Ch. 12: sell an OTM put and an OTM call."""

    name, chapter = "Short Strangle", 12

    def __init__(self, k_put: float, p_put: float, k_call: float, p_call: float, **leg_kw):
        _two_strikes(k_put, k_call)
        self.k_put, self.p_put, self.k_call, self.p_call = k_put, p_put, k_call, p_call
        super().__init__([Leg(PUT, SELL, k_put, p_put, **leg_kw), Leg(CALL, SELL, k_call, p_call, **leg_kw)])

    def generalization(self):
        credit = self.p_put + self.p_call
        return {"net_credit": credit, "max_profit": credit, "max_loss": math.inf,
                "breakevens": (self.k_put - credit, self.k_call + credit)}


STRATEGIES = {cls.name: cls for cls in (BullCallSpread, BullPutSpread, CallRatioBackSpread, BearCallLadder,
                                        SyntheticLong, BearPutSpread, BearCallSpread, PutRatioBackSpread,
                                        LongStraddle, ShortStraddle, LongStrangle, ShortStrangle)}


# ---------------------------------------------------------------- ch. 13

def max_pain(strikes: Sequence[float], call_oi: Sequence[float], put_oi: Sequence[float]) -> Tuple[float, pd.DataFrame]:
    """(max pain strike, table): option writers' total loss if expiry is at each strike.

    L(X) = sum over K < X of (X - K) call OI + sum over K > X of (K - X) put OI.
    """
    k = np.asarray(strikes, dtype=float)
    c = np.asarray(call_oi, dtype=float)
    p = np.asarray(put_oi, dtype=float)
    if not (len(k) == len(c) == len(p)):
        raise ValueError("strikes, call_oi and put_oi must have equal length")
    call_loss = np.array([np.sum(np.clip(x - k, 0, None) * c) for x in k])
    put_loss = np.array([np.sum(np.clip(k - x, 0, None) * p) for x in k])
    table = pd.DataFrame({"strike": k, "call_oi": c, "put_oi": p, "call_writers_loss": call_loss,
                          "put_writers_loss": put_loss, "total_loss": call_loss + put_loss})
    return float(k[int(np.argmin(call_loss + put_loss))]), table


def put_call_ratio(put_oi: Iterable[float], call_oi: Iterable[float]) -> float:
    return float(sum(put_oi)) / float(sum(call_oi))


def pcr_signal(pcr: float, high: float = 1.3, low: float = 0.5) -> str:
    """Contrarian reading (ch. 13); 1 to ``high`` is undefined in the PDF and treated as normal."""
    if pcr > high:
        return "bullish reversal expected (extreme bearishness)"
    if pcr < low:
        return "bearish reversal expected (extreme bullishness)"
    return "normal"


def modified_max_pain_band(max_pain_strike: float, buffer: float = 0.05,
                           strikes: Optional[Sequence[float]] = None, strike_step: Optional[float] = None) -> Dict[str, float]:
    """The author's band [max pain, max pain x (1 + buffer)] and the first call strike to write above it."""
    top = max_pain_strike * (1 + buffer)
    if strikes is not None:
        above = [k for k in sorted(strikes) if k >= top]
        first = above[0] if above else math.nan
    elif strike_step:
        first = math.ceil(top / strike_step - 1e-12) * strike_step
    else:
        first = math.nan
    return {"low": max_pain_strike, "high": top, "write_calls_from": first}
