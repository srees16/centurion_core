"""
Options theory engine: Zerodha Varsity Module 5 (see CONCEPTS.md).

Payoffs and moneyness (ch. 3 to 8), Black-Scholes price and Greeks with an
implied-volatility solver and put-call parity (ch. 9 to 14, 19 to 21),
volatility, expected ranges and the ch. 18 applications (15 to 18, 20).

Conventions (CONCEPTS.md § Conventions): P&L per unit of underlying, + = money
in; T = calendar days / 365; theta per calendar day, vega per volatility
point, rho per rate point; rates and volatilities as decimals.  All functions
are pure and take no broker objects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

CALL, PUT = "CE", "PE"
BUY, SELL = "BUY", "SELL"


def _check_type(option_type: str) -> str:
    t = option_type.upper()
    if t not in (CALL, PUT):
        raise ValueError(f"option_type must be CE or PE, got {option_type!r}")
    return t


def side_sign(side: str) -> int:
    """+1 for a long (BUY), -1 for a short (SELL)."""
    s = side.upper()
    if s not in (BUY, SELL):
        raise ValueError(f"side must be BUY or SELL, got {side!r}")
    return 1 if s == BUY else -1


# ---------------------------------------------------------------- ch. 3 to 8

def intrinsic_value(option_type: str, spot, strike):
    """max(0, S - K) for a call, max(0, K - S) for a put (never negative, ch. 8)."""
    s = np.asarray(spot, dtype=float)
    v = np.maximum(s - strike, 0.0) if _check_type(option_type) == CALL else np.maximum(strike - s, 0.0)
    return float(v) if v.ndim == 0 else v


def time_value(premium: float, option_type: str, spot: float, strike: float) -> float:
    """Premium = intrinsic value + time value (ch. 14)."""
    return premium - intrinsic_value(option_type, spot, strike)


def expiry_pnl(option_type: str, side: str, strike: float, premium: float, spot):
    """P&L per unit held to expiry: long = intrinsic - premium; short is its mirror."""
    return side_sign(side) * (intrinsic_value(option_type, spot, strike) - premium)


def breakeven(option_type: str, strike: float, premium: float) -> float:
    """K + premium for calls, K - premium for puts; the same level for buyer and seller."""
    return strike + premium if _check_type(option_type) == CALL else strike - premium


def atm_strike(spot: float, strikes: Iterable[float]) -> float:
    """The listed strike nearest to spot; the lower one on a tie."""
    ks = sorted(float(k) for k in strikes)
    if not ks:
        raise ValueError("no strikes")
    return min(ks, key=lambda k: (abs(k - spot), k))


def moneyness(option_type: str, spot: float, strike: float,
              strikes: Optional[Sequence[float]] = None, deep_strikes: int = 3) -> str:
    """ATM / ITM / OTM, and DEEP ITM / DEEP OTM ``deep_strikes`` listed strikes from ATM.

    ATM is the listed strike nearest to spot when ``strikes`` is given (ch. 8);
    without it only ITM / OTM / ATM (strike == spot) is returned.
    """
    t = _check_type(option_type)
    if strikes is not None:
        ks = sorted(float(k) for k in strikes)
        atm = atm_strike(spot, ks)
        if strike == atm:
            return "ATM"
        steps = abs(ks.index(float(strike)) - ks.index(atm)) if float(strike) in ks else 0
    else:
        if strike == spot:
            return "ATM"
        steps = 0
    itm = intrinsic_value(t, spot, strike) > 0
    label = "ITM" if itm else "OTM"
    return f"DEEP {label}" if steps >= deep_strikes else label


# ------------------------------------------------------------ ch. 9 to 14, 21

@dataclass(frozen=True)
class Greeks:
    """Per unit; theta per calendar day, vega per vol point, rho per rate point."""

    price: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0

    def scaled(self, k: float) -> "Greeks":
        return Greeks(*(k * getattr(self, f) for f in self._fields()))

    def __add__(self, other: "Greeks") -> "Greeks":
        return Greeks(*(getattr(self, f) + getattr(other, f) for f in self._fields()))

    @staticmethod
    def _fields() -> Tuple[str, ...]:
        return ("price", "delta", "gamma", "theta", "vega", "rho")


def _d1_d2(spot: float, strike: float, t: float, rate: float, iv: float, dividend: float) -> Tuple[float, float]:
    vs = iv * math.sqrt(t)
    d1 = (math.log(spot / strike) + (rate - dividend + 0.5 * iv * iv) * t) / vs
    return d1, d1 - vs


def black_scholes(option_type: str, spot: float, strike: float, days_to_expiry: float,
                  rate: float, iv: float, dividend: float = 0.0, days_per_year: int = 365) -> Greeks:
    """European Black-Scholes(-Merton) price and Greeks (ch. 21).

    At or after expiry (or zero volatility) the price is the intrinsic value
    and the Greeks are their limits: delta 1/0 (call) or -1/0 (put).
    """
    t_type = _check_type(option_type)
    t = days_to_expiry / days_per_year
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if t <= 0 or iv <= 0:
        iv_now = intrinsic_value(t_type, spot, strike)
        itm = iv_now > 0
        delta = (1.0 if itm else 0.0) if t_type == CALL else (-1.0 if itm else 0.0)
        return Greeks(price=iv_now, delta=delta)
    d1, d2 = _d1_d2(spot, strike, t, rate, iv, dividend)
    dq, dr = math.exp(-dividend * t), math.exp(-rate * t)
    pdf = norm.pdf(d1)
    gamma = dq * pdf / (spot * iv * math.sqrt(t))
    vega = spot * dq * pdf * math.sqrt(t) / 100.0  # per volatility point
    common = -spot * dq * pdf * iv / (2.0 * math.sqrt(t))
    if t_type == CALL:
        price = spot * dq * norm.cdf(d1) - strike * dr * norm.cdf(d2)
        delta = dq * norm.cdf(d1)
        theta = common - rate * strike * dr * norm.cdf(d2) + dividend * spot * dq * norm.cdf(d1)
        rho = strike * t * dr * norm.cdf(d2) / 100.0
    else:
        price = strike * dr * norm.cdf(-d2) - spot * dq * norm.cdf(-d1)
        delta = dq * (norm.cdf(d1) - 1.0)
        theta = common + rate * strike * dr * norm.cdf(-d2) - dividend * spot * dq * norm.cdf(-d1)
        rho = -strike * t * dr * norm.cdf(-d2) / 100.0
    return Greeks(float(price), float(delta), float(gamma), float(theta / days_per_year), float(vega), float(rho))


def probability_itm(option_type: str, spot: float, strike: float, days_to_expiry: float,
                    rate: float, iv: float, dividend: float = 0.0, days_per_year: int = 365) -> Dict[str, float]:
    """Risk-neutral P(expires ITM) = N(d2) / N(-d2), beside ch. 11's |delta| approximation."""
    t_type = _check_type(option_type)
    t = days_to_expiry / days_per_year
    g = black_scholes(t_type, spot, strike, days_to_expiry, rate, iv, dividend, days_per_year)
    if t <= 0 or iv <= 0:
        p = 1.0 if intrinsic_value(t_type, spot, strike) > 0 else 0.0
    else:
        _, d2 = _d1_d2(spot, strike, t, rate, iv, dividend)
        p = norm.cdf(d2) if t_type == CALL else norm.cdf(-d2)
    return {"n_d2": float(p), "delta_approx": abs(g.delta)}


def implied_volatility(option_type: str, price: float, spot: float, strike: float, days_to_expiry: float,
                       rate: float, dividend: float = 0.0, days_per_year: int = 365,
                       lo: float = 1e-4, hi: float = 5.0, tol: float = 1e-8) -> Optional[float]:
    """The sigma with Black-Scholes(sigma) = price (Brent's method), or None.

    None when there is no time left, or the price sits outside the arbitrage
    bounds the model can reach between ``lo`` and ``hi``.
    """
    t_type = _check_type(option_type)
    if days_to_expiry <= 0 or price <= 0:
        return None

    def f(sigma: float) -> float:
        return black_scholes(t_type, spot, strike, days_to_expiry, rate, sigma, dividend, days_per_year).price - price

    f_lo, f_hi = f(lo), f(hi)
    if f_lo > 0 or f_hi < 0:
        return None
    return float(brentq(f, lo, hi, xtol=tol))


def parity_residual(call: float, put: float, spot: float, strike: float, days_to_expiry: float,
                    rate: float, dividend: float = 0.0, days_per_year: int = 365) -> float:
    """(C - P) - (S e^-qT - K e^-rT); zero when put-call parity holds (ch. 21, any strike)."""
    t = max(days_to_expiry, 0.0) / days_per_year
    return (call - put) - (spot * math.exp(-dividend * t) - strike * math.exp(-rate * t))


def delta_gamma_step(premium: float, delta: float, gamma: float, spot_change: float) -> Tuple[float, float]:
    """Ch. 13's first-order update: premium + old delta x dS, then delta + gamma x dS."""
    return premium + delta * spot_change, delta + gamma * spot_change


def position_greeks(items: Iterable[Tuple[float, Greeks]]) -> Greeks:
    """Sum of signed quantity x per-unit Greeks (deltas are additive, ch. 11).

    ``items`` holds (signed quantity, Greeks): +lots for longs, -lots for shorts.
    A futures position is Greeks(price=F, delta=1).
    """
    total = Greeks()
    for qty, g in items:
        total = total + g.scaled(qty)
    return total


# ----------------------------------------------------------- ch. 15 to 18, 20

def mean_sd(values: Sequence[float], ddof: int = 0) -> Tuple[float, float]:
    """Mean and SD; ddof 0 is ch. 15's population SD, 1 is Excel STDEV (ch. 16, 20)."""
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=ddof))


def log_returns(prices: Sequence[float]) -> np.ndarray:
    """ln(P_t / P_t-1), ch. 16."""
    p = np.asarray(prices, dtype=float)
    return np.log(p[1:] / p[:-1])


def daily_volatility(prices: Sequence[float], ddof: int = 1) -> float:
    return float(np.std(log_returns(prices), ddof=ddof))


def annualize_vol(sigma_daily: float, days_per_year: int = 365) -> float:
    return sigma_daily * math.sqrt(days_per_year)


def deannualize_vol(sigma_annual: float, days_per_year: int = 365) -> float:
    return sigma_annual / math.sqrt(days_per_year)


def period_vol(sigma_daily: float, days: float) -> float:
    """sigma over ``days``: sigma_daily x sqrt(days)."""
    return sigma_daily * math.sqrt(days)


def historical_volatility(prices: Sequence[float], days_per_year: int = 365, ddof: int = 1) -> float:
    """Annualised SD of daily log returns."""
    return annualize_vol(daily_volatility(prices, ddof), days_per_year)


def sd_coverage(k: float) -> float:
    """P(|Z| <= k): 68.27 %, 95.45 %, 99.73 % for 1, 2, 3 SD."""
    return math.erf(k / math.sqrt(2.0))


def price_range(spot: float, sigma: float, mean: float = 0.0, k: float = 1.0,
                method: str = "linear") -> Tuple[float, float]:
    """Range for a period with mean ``mean`` and SD ``sigma`` (both for that period).

    simple    S(1 +/- k sigma)          ch. 15 (no drift)
    linear    S(1 + mean +/- k sigma)   ch. 18
    lognormal S exp(mean +/- k sigma)   ch. 17
    """
    if method == "simple":
        return spot * (1 - k * sigma), spot * (1 + k * sigma)
    if method == "linear":
        return spot * (1 + mean - k * sigma), spot * (1 + mean + k * sigma)
    if method == "lognormal":
        return spot * math.exp(mean - k * sigma), spot * math.exp(mean + k * sigma)
    raise ValueError(f"unknown method {method!r}")


def expected_range(spot: float, sigma_daily: float, days: float, mean_daily: float = 0.0,
                   ks: Sequence[float] = (1, 2, 3), method: str = "linear") -> Dict[float, Tuple[float, float]]:
    """{k: (low, high)} over ``days`` from daily mean and SD (mean x n, SD x sqrt n)."""
    sigma, mean = period_vol(sigma_daily, days), mean_daily * days
    return {k: price_range(spot, sigma, mean, k, method) for k in ks}


def writing_sd_multiple(days_to_expiry: int, max_dte: int = 15, one_sd_max_dte: int = 4) -> Optional[int]:
    """Ch. 18's rules: no writing beyond ``max_dte``; 1 SD near expiry, else 2 SD."""
    if days_to_expiry > max_dte:
        return None
    return 1 if days_to_expiry <= one_sd_max_dte else 2


def sd_writing_strikes(spot: float, sigma_daily: float, days_to_expiry: float, strikes: Iterable[float],
                       option_type: str = CALL, k: float = 1.0, mean_daily: float = 0.0) -> List[float]:
    """Strikes outside the k-SD linear range (ch. 18): calls above it, puts below it.

    Nearest to the range first.
    """
    lo, hi = expected_range(spot, sigma_daily, days_to_expiry, mean_daily, (k,), "linear")[k]
    ks = sorted(float(x) for x in strikes)
    if _check_type(option_type) == CALL:
        return [x for x in ks if x > hi]
    return [x for x in reversed(ks) if x < lo]


def volatility_stop_loss(entry: float, sigma_daily: float, holding_days: float, side: str = BUY) -> float:
    """Ch. 18: entry x (1 -/+ sigma_daily sqrt(days)); place the stop beyond this level."""
    move = period_vol(sigma_daily, holding_days)
    return entry * (1 - move) if side_sign(side) > 0 else entry * (1 + move)


def risk_reward(entry: float, target: float, stop: float) -> float:
    """Reward / risk, both measured from the entry."""
    return abs(target - entry) / abs(entry - stop)


def cone_stats(window_vols: Sequence[float], ddof: int = 1) -> Dict[str, float]:
    """Ch. 20 cone row for one window: max, mean +/- 1 and 2 SD, min."""
    mean, sd = mean_sd(window_vols, ddof)
    a = np.asarray(window_vols, dtype=float)
    return {"max": float(a.max()), "plus2": mean + 2 * sd, "plus1": mean + sd, "mean": mean,
            "minus1": mean - sd, "minus2": mean - 2 * sd, "min": float(a.min())}


def volatility_cone(prices: Sequence[float], windows: Sequence[int] = (10, 20, 30, 45, 60, 90),
                    days_per_year: int = 365, ddof: int = 1) -> Dict[int, Dict[str, float]]:
    """Ch. 20: annualised realised volatility over rolling windows, summarised per window.

    Each row also carries ``current``, the latest window's value, to compare
    with today's implied volatility.
    """
    r = log_returns(prices)
    out: Dict[int, Dict[str, float]] = {}
    for w in windows:
        if len(r) < w + 1:
            continue
        vols = [annualize_vol(float(np.std(r[i - w:i], ddof=ddof)), days_per_year) for i in range(w, len(r) + 1)]
        row = cone_stats(vols, ddof)
        row["current"] = vols[-1]
        out[w] = row
    return out
