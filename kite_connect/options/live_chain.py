"""
Option chain from live Kite quotes: price, bid / ask, OI and volume per
contract, with our own implied volatility and Greeks (``theory``), and the
chain's ATM strike, ATM IV, max pain and PCR (``strategies``).

Implied volatility is solved from the bid / ask mid when both sides quote,
else from the last price; a price below the no-arbitrage bound gives no IV
rather than an invented one.  Priced on spot with no dividend, an ITM
option's mid often sits below that bound, so each strike also carries
``iv_strike``: the OTM option's IV for both options there, else the other
one's when it does not solve (Lean's smoothing rule, tracker LN-T27), with
``iv_source`` saying whose it is (own, mirror or none).
"""

from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from kite_connect.options.instruments import Contract, InstrumentResolver, spot_key
from kite_connect.options.strategies import max_pain, put_call_ratio
from kite_connect.options.theory import CALL, PUT, atm_strike, black_scholes, implied_volatility

IST = timezone(timedelta(hours=5, minutes=30))
EXPIRY_CLOSE = time(15, 30)
CHAIN_COLUMNS = ["strike", "option_type", "tradingsymbol", "lot_size", "ltp", "bid", "ask", "mid", "oi", "volume",
                 "iv", "delta", "gamma", "theta", "vega", "iv_strike", "iv_source"]


def days_to_expiry(expiry: date, now: datetime) -> float:
    """Calendar days from ``now`` to the 15:30 IST close of expiry day (at least one minute)."""
    close = datetime.combine(expiry, EXPIRY_CLOSE, tzinfo=IST)
    return max((close - now.astimezone(IST)).total_seconds() / 86400.0, 1.0 / 1440.0)


def quote_prices(quote: Optional[dict]) -> Dict[str, float]:
    """Last price, best bid / ask (NaN when that side is empty), mid, OI and volume of one Kite quote."""
    q = quote or {}
    depth = q.get("depth") or {}
    bid = next((float(d["price"]) for d in depth.get("buy", []) if d.get("price")), math.nan)
    ask = next((float(d["price"]) for d in depth.get("sell", []) if d.get("price")), math.nan)
    ltp = float(q.get("last_price") or math.nan)
    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else ltp
    return {"ltp": ltp, "bid": bid, "ask": ask, "mid": mid, "oi": float(q.get("oi") or 0.0),
            "volume": float(q.get("volume") or 0.0)}


def build_chain(contracts: Sequence[Contract], quotes: Dict[str, dict], spot: float, now: datetime,
                rate: float) -> pd.DataFrame:
    """One row per contract with prices, IV and Greeks (per unit; theta per day, vega per vol point), and the
    strike's IV (``iv_strike``, ``iv_source``)."""
    rows: List[dict] = []
    for c in contracts:
        p = quote_prices(quotes.get(c.quote_key))
        dte = days_to_expiry(c.expiry, now)
        iv = (implied_volatility(c.option_type, p["mid"], spot, c.strike, dte, rate)
              if np.isfinite(p["mid"]) and p["mid"] > 0 else None)
        g = black_scholes(c.option_type, spot, c.strike, dte, rate, iv) if iv else None
        rows.append({"strike": c.strike, "option_type": c.option_type, "tradingsymbol": c.tradingsymbol,
                     "lot_size": c.lot_size, **p, "iv": iv if iv else math.nan,
                     "delta": g.delta if g else math.nan, "gamma": g.gamma if g else math.nan,
                     "theta": g.theta if g else math.nan, "vega": g.vega if g else math.nan})
    own = {(r["strike"], r["option_type"]): r["iv"] for r in rows}
    for r in rows:
        otm = CALL if r["strike"] >= spot else PUT
        for t in (otm, PUT if otm == CALL else CALL):          # the OTM side first, then its mirror
            iv = own.get((r["strike"], t), math.nan)
            if np.isfinite(iv):
                r.update(iv_strike=iv, iv_source="own" if t == r["option_type"] else "mirror")
                break
        else:
            r.update(iv_strike=math.nan, iv_source="none")
    return pd.DataFrame(rows, columns=CHAIN_COLUMNS)


def chain_summary(chain: pd.DataFrame, spot: float) -> Dict[str, float]:
    """ATM strike, ATM IV (mean of the call and put), max pain and the OI put-call ratio."""
    strikes = sorted(chain["strike"].unique())
    atm = atm_strike(spot, strikes)
    atm_iv = float(chain.loc[chain["strike"] == atm, "iv"].mean())
    calls = chain[chain["option_type"] == "CE"].set_index("strike")["oi"].reindex(strikes).fillna(0.0)
    puts = chain[chain["option_type"] == "PE"].set_index("strike")["oi"].reindex(strikes).fillna(0.0)
    has_oi = calls.sum() > 0 and puts.sum() > 0
    mp = max_pain(strikes, calls.to_numpy(), puts.to_numpy())[0] if has_oi else math.nan
    return {"spot": spot, "atm_strike": atm, "atm_iv": atm_iv, "max_pain": mp,
            "pcr": put_call_ratio(puts, calls) if has_oi else math.nan}


def fetch_chain(broker, resolver: InstrumentResolver, underlying: str, expiry: date, now: datetime,
                rate: float, strikes_each_side: Optional[int] = None) -> Tuple[pd.DataFrame, float]:
    """(chain, spot) for an expiry from live quotes; ``strikes_each_side`` keeps that many strikes around ATM."""
    key = spot_key(underlying)
    spot = float((broker.quotes([key]).get(key) or {}).get("last_price") or math.nan)
    if not np.isfinite(spot):
        raise RuntimeError(f"no spot quote for {key}")
    strikes = resolver.strikes(underlying, expiry)
    if strikes_each_side is not None:
        i = strikes.index(atm_strike(spot, strikes))
        strikes = strikes[max(i - strikes_each_side, 0): i + strikes_each_side + 1]
    contracts = resolver.chain(underlying, expiry, strikes)
    quotes = broker.quotes([c.quote_key for c in contracts])
    return build_chain(contracts, quotes, spot, now, rate), spot
