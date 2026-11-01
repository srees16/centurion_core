"""
Overlay Strategies scan for the app's Options page: read only, it places nothing.

Every number comes from the market or the account.  Covered calls are written
only against what ``holdings`` really holds, in whole lots.  Cash-secured puts
are looked at only on ``wanted``, the names the engine's book holds or is about
to buy.  Iron condors are looked at on ``limits.allowed_underlyings``.

Strikes and expiries are listed contracts (``InstrumentResolver``), never a
strike solved from a formula or a date 30 days out; lot sizes come from the
same dump.  Premiums are quotes: the bid on a sale, the ask on a purchase, so a
leg with no two-sided quote, too wide a spread or too little open interest is
not offered.  Together secured puts and condor margin stay inside
``overlay.max_capital_pct`` of the capital.  IV is
solved from those quotes (``live_chain``), and the 30-day IV is built as its
history is, then ranked against the deployed year of it (``iv_history``).  A
symbol with no quote or no history is left out and named in the log: nothing
falls back to a default.

Short strangles are not offered: unlimited loss, which ``pretrade.check_limits``
refuses.  Condors stand down at or above ``RegimeConfig.vix_extreme``, the
regime gate's own limit, on the live India VIX; if it cannot be read they stand
down too.  Premiums are before charges.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from kite_connect.options.instruments import Contract, InstrumentResolver, spot_key
from kite_connect.options.iv_history import constant_maturity, iv30_expiries, rank_and_percentile, skew_warning
from kite_connect.options.live_chain import IST, build_chain, chain_summary, days_to_expiry
from kite_connect.options.options_config import OptionsConfig
from kite_connect.options.theory import CALL, PUT, atm_strike, probability_itm
from nse_engine.config import RegimeConfig

logger = logging.getLogger(__name__)

VIX_KEY = "NSE:INDIA VIX"


@dataclass
class Market:
    """One underlying's live read: its target expiry's chain, and today's 30-day IV and its rank."""

    symbol: str
    spot: float
    expiry: date
    dte: float
    chain: pd.DataFrame
    lot_size: int
    iv30: float
    rank: float


def target_expiry(dtes: Mapping[date, float], cfg: OptionsConfig) -> Optional[date]:
    """The expiry nearest ``overlay.target_dte`` inside the window, or None."""
    o = cfg.overlay
    inside = [e for e, d in dtes.items() if o.min_dte <= d <= o.max_dte]
    return min(inside, key=lambda e: abs(dtes[e] - o.target_dte)) if inside else None


def iv_rank_of(history: pd.DataFrame, symbol: str, iv30: float, today: date, cfg: OptionsConfig) -> Tuple[float, str]:
    """(rank 0-100, "") or (NaN, why): a symbol with no history, or a stale one, has no rank."""
    h = history[history["symbol"] == symbol].sort_values("date")
    if h.empty:
        return math.nan, "no IV history"
    age = (today - pd.Timestamp(h["date"].iloc[-1]).date()).days
    if age > cfg.overlay.history_max_age_days:
        return math.nan, f"IV history {age} days old"
    return rank_and_percentile(h["iv30"], iv30, cfg.data.lookback_sessions)[0], ""


def pick_leg(chain: pd.DataFrame, option_type: str, spot: float, delta: float,
             cfg: OptionsConfig) -> Optional[pd.Series]:
    """The out-of-the-money listed option nearest ``delta`` (absolute) with a tight two-sided quote and enough
    open interest."""
    c = chain[(chain["option_type"] == option_type) & (chain["bid"] > 0) & (chain["ask"] > 0)
              & (chain["oi"] >= cfg.overlay.min_open_interest) & np.isfinite(chain["delta"])]
    c = c[c["strike"] > spot] if option_type == CALL else c[c["strike"] < spot]
    c = c[(c["ask"] - c["bid"]) <= cfg.overlay.max_spread_pct * (c["ask"] + c["bid"]) / 2.0]
    if c.empty:
        return None
    return c.loc[(c["delta"].abs() - delta).abs().idxmin()]


def _order(m: Market, leg: pd.Series, lots: int) -> dict:
    """A sold option as the page's table shows it."""
    premium = round(float(leg["bid"]), 2)
    return {"symbol": m.symbol, "tradingsymbol": leg["tradingsymbol"], "strike": float(leg["strike"]),
            "expiry": m.expiry.isoformat(), "premium": premium,
            "total_premium": round(premium * lots * m.lot_size, 2), "delta": round(float(leg["delta"]), 3),
            "iv": round(float(leg["iv"]), 4), "lots": lots, "lot_size": m.lot_size,
            "underlying_price": m.spot, "iv_rank": round(m.rank, 1)}


def covered_calls(markets: Mapping[str, Market], held: Mapping[str, int], cfg: OptionsConfig,
                  log: List[str]) -> List[dict]:
    o, out = cfg.overlay, []
    for symbol, quantity in held.items():
        m = markets.get(symbol)
        if m is None:
            continue
        lots = quantity // m.lot_size
        if lots < 1:
            log.append(f"{symbol}: {quantity} shares held, below one lot of {m.lot_size}")
        elif m.rank < o.min_rank_call:
            log.append(f"{symbol}: IV rank {m.rank:.0f} below {o.min_rank_call:.0f}, no covered call")
        else:
            leg = pick_leg(m.chain, CALL, m.spot, o.call_delta, cfg)
            if leg is None:
                log.append(f"{symbol}: no call with a tight two-sided quote and OI {o.min_open_interest}+")
            elif leg["bid"] / m.spot < o.min_premium_pct:
                log.append(f"{symbol}: call premium {leg['bid'] / m.spot:.2%} of spot, below {o.min_premium_pct:.1%}")
            else:
                out.append(_order(m, leg, lots))
    return out


def cash_secured_puts(markets: Mapping[str, Market], wanted: Sequence[str], capital: float,
                      sectors: Mapping[str, str], cfg: OptionsConfig, log: List[str]) -> List[dict]:
    """One lot each, fully secured: strike x lot out of ``max_capital_pct`` of the capital."""
    o, out = cfg.overlay, []
    room, per_sector = capital * o.max_capital_pct, Counter()
    for symbol in wanted:
        m = markets.get(symbol)
        if m is None:
            continue
        sector = sectors.get(symbol, "Unknown")
        leg = pick_leg(m.chain, PUT, m.spot, o.put_delta, cfg) if m.rank >= o.min_rank_put else None
        if m.rank < o.min_rank_put:
            log.append(f"{symbol}: IV rank {m.rank:.0f} below {o.min_rank_put:.0f}, no put")
        elif leg is None:
            log.append(f"{symbol}: no put with a tight two-sided quote and OI {o.min_open_interest}+")
        elif leg["bid"] / m.spot < o.min_premium_pct:
            log.append(f"{symbol}: put premium {leg['bid'] / m.spot:.2%} of spot, below {o.min_premium_pct:.1%}")
        elif per_sector[sector] >= o.max_puts_per_sector:
            log.append(f"{symbol}: {o.max_puts_per_sector} puts already in {sector}")
        else:
            secured = float(leg["strike"]) * m.lot_size
            if secured > room:
                log.append(f"{symbol}: securing one lot takes Rs {secured:,.0f}, Rs {room:,.0f} left")
            else:
                out.append(_order(m, leg, 1))
                room -= secured
                per_sector[sector] += 1
    return out


def condor_margin(broker, m: Market, legs: Sequence[Tuple[pd.Series, str]]) -> float:
    """Kite's basket margin for the four legs, with hedge benefit."""
    orders = [{"exchange": "NFO", "tradingsymbol": leg["tradingsymbol"], "transaction_type": side, "variety": "regular",
               "product": "NRML", "order_type": "LIMIT", "quantity": m.lot_size,
               "price": float(leg["bid"] if side == "SELL" else leg["ask"])} for leg, side in legs]
    return float(((broker.basket_margins(orders) or {}).get("final") or {}).get("total") or math.nan)


def iron_condor(broker, m: Market, room: float, rate: float, now: datetime, cfg: OptionsConfig,
                log: List[str]) -> Optional[dict]:
    """Short ``condor_short_delta`` both sides, ``condor_wing_delta`` wings, one lot, inside the hard limits and
    with its margin inside ``room``."""
    o = cfg.overlay
    if m.rank < o.min_rank_condor:
        log.append(f"{m.symbol}: IV rank {m.rank:.0f} below {o.min_rank_condor:.0f}, no condor")
        return None
    short_call, short_put = (pick_leg(m.chain, t, m.spot, o.condor_short_delta, cfg) for t in (CALL, PUT))
    long_call, long_put = (pick_leg(m.chain, t, m.spot, o.condor_wing_delta, cfg) for t in (CALL, PUT))
    if any(x is None for x in (short_call, short_put, long_call, long_put)):
        log.append(f"{m.symbol}: no four legs with tight two-sided quotes and OI {o.min_open_interest}+")
        return None
    if long_call["strike"] <= short_call["strike"] or long_put["strike"] >= short_put["strike"]:
        log.append(f"{m.symbol}: wings not beyond the short strikes inside the quoted band")
        return None
    credit = float(short_call["bid"] + short_put["bid"] - long_call["ask"] - long_put["ask"])
    width = max(float(long_call["strike"] - short_call["strike"]), float(short_put["strike"] - long_put["strike"]))
    net, max_loss = credit * m.lot_size, (width - credit) * m.lot_size
    if credit <= 0 or max_loss <= 0:
        log.append(f"{m.symbol}: condor has no credit at the quotes")
        return None
    if max_loss > cfg.limits.max_loss_per_trade_inr:
        log.append(f"{m.symbol}: condor max loss Rs {max_loss:,.0f} above the limit "
                   f"Rs {cfg.limits.max_loss_per_trade_inr:,.0f}")
        return None
    legs = ((short_call, "SELL"), (long_call, "BUY"), (short_put, "SELL"), (long_put, "BUY"))
    margin = condor_margin(broker, m, legs)
    if not math.isfinite(margin) or margin > room:
        log.append(f"{m.symbol}: condor margin {margin:,.0f} above the Rs {room:,.0f} left")
        return None
    outside = sum(probability_itm(x["option_type"], m.spot, float(x["strike"]), m.dte, rate, float(x["iv"]))["n_d2"]
                  for x in (short_call, short_put))
    return {"strategy_type": "IRON_CONDOR", "underlying": m.symbol, "expiry": m.expiry.isoformat(),
            "legs": [{"strike": float(x["strike"]), "type": x["option_type"], "side": side, "lots": 1,
                      "premium": round(float(x["bid"] if side == "SELL" else x["ask"]), 2),
                      "delta": round(float(x["delta"]), 3)} for x, side in legs],
            "net_credit": round(net, 2), "max_loss": round(max_loss, 2), "max_profit": round(net, 2),
            "pop": round((1.0 - outside) * 100.0, 1), "margin_required": round(margin, 2),
            "iv_rank": round(m.rank, 1)}


def read_markets(broker, resolver: InstrumentResolver, symbols: Sequence[str], history: pd.DataFrame,
                 now: datetime, cfg: OptionsConfig, log: List[str]) -> Dict[str, Market]:
    """Each symbol's chain and ranked 30-day IV, on one batched quote pass."""
    o, rate, today = cfg.overlay, cfg.market.risk_free_rate, now.astimezone(IST).date()
    spots = broker.quotes([spot_key(s) for s in symbols])
    plans: Dict[str, Tuple[float, date, Dict[date, List[Contract]], Dict[date, float]]] = {}
    keys: List[str] = []
    for s in symbols:
        spot = float((spots.get(spot_key(s)) or {}).get("last_price") or math.nan)
        dtes = {e: days_to_expiry(e, now) for e in resolver.expiries(s) if e >= today}
        expiry = target_expiry(dtes, cfg)
        if not math.isfinite(spot) or expiry is None:
            log.append(f"{s}: {'no spot quote' if expiry is not None else 'no expiry in the window'}")
            continue
        contracts: Dict[date, List[Contract]] = {}
        for e in {expiry, *iv30_expiries(dtes, cfg)}:
            listed = resolver.strikes(s, e)
            if e == expiry:
                near = [k for k in listed if abs(k / spot - 1.0) <= o.strike_band]
            else:
                i = listed.index(atm_strike(spot, listed))
                near = listed[max(i - 1, 0): i + 2]
            contracts[e] = resolver.chain(s, e, near)
            keys += [c.quote_key for c in contracts[e]]
        plans[s] = spot, expiry, contracts, dtes
    quotes = broker.quotes(keys) if keys else {}
    out: Dict[str, Market] = {}
    for s, (spot, expiry, contracts, dtes) in plans.items():
        chains = {e: build_chain(cs, quotes, spot, now, rate) for e, cs in contracts.items()}
        points = []
        for e in iv30_expiries(dtes, cfg):
            atm_iv = chain_summary(chains[e], spot)["atm_iv"]
            points.append((dtes[e], atm_iv if math.isfinite(atm_iv) else None))
        warning = skew_warning(s, points, cfg)
        if warning:
            log.append(warning)
        iv30 = constant_maturity(points, cfg.data.iv_tenor_days)
        if iv30 is None:
            log.append(f"{s}: no live 30-day IV")
            continue
        rank, why = iv_rank_of(history, s, iv30, today, cfg)
        if why:
            log.append(f"{s}: {why}")
            continue
        out[s] = Market(s, spot, expiry, dtes[expiry], chains[expiry], contracts[expiry][0].lot_size, iv30, rank)
    return out


def scan(broker, resolver: InstrumentResolver, held: Mapping[str, int], wanted: Sequence[str], capital: float,
         history: pd.DataFrame, now: datetime, cfg: OptionsConfig = OptionsConfig(),
         sectors: Optional[Mapping[str, str]] = None) -> dict:
    """The tab's result.  ``held``: shares by symbol; ``wanted``: the book's names; ``capital``: the rupees that
    may secure positions."""
    log = [f"Capital Rs {capital:,.0f}; {len(held)} holdings, {len(wanted)} names the book wants"]
    held = {s.upper(): int(q) for s, q in held.items() if resolver.lists(s)}
    wanted = [s.upper() for s in wanted if resolver.lists(s)]
    indices = [s for s in cfg.limits.allowed_underlyings if resolver.lists(s)]
    symbols = list(dict.fromkeys([*indices, *held, *wanted]))
    if len(symbols) > cfg.overlay.max_symbols:
        log.append(f"Scanned the first {cfg.overlay.max_symbols} of {len(symbols)}; left out: "
                   + ", ".join(symbols[cfg.overlay.max_symbols:]))
        symbols = symbols[:cfg.overlay.max_symbols]
    markets = read_markets(broker, resolver, symbols, history, now, cfg, log)
    calls = covered_calls(markets, held, cfg, log)
    puts = cash_secured_puts(markets, [s for s in wanted if s not in held], capital, sectors or {}, cfg, log)
    vix = float((broker.quotes([VIX_KEY]).get(VIX_KEY) or {}).get("last_price") or math.nan)
    condors: List[dict] = []
    o = cfg.overlay
    room = min(capital * o.max_condor_capital_pct,
               capital * o.max_capital_pct - sum(p["strike"] * p["lots"] * p["lot_size"] for p in puts))
    if not math.isfinite(vix) or vix >= RegimeConfig().vix_extreme:
        log.append(f"India VIX {vix:.1f}: condors stand down")
    else:
        for s in indices:
            if s in markets:
                c = iron_condor(broker, markets[s], room, cfg.market.risk_free_rate, now, cfg, log)
                if c:
                    condors.append(c)
                    room -= c["margin_required"]
    overlay = sum(x["total_premium"] for x in calls + puts)
    multileg = sum(c["net_credit"] for c in condors)
    total = overlay + multileg
    # the premium is earned over the position's life, so a monthly figure spreads it over the average days held
    days = np.mean([m.dte for m in markets.values()]) if markets else o.target_dte
    monthly = total / capital * 100.0 * 30.0 / days if capital > 0 else 0.0
    log.append(f"Premiums at bid and ask, before charges.  India VIX {vix:.1f}.")
    return {"covered_calls": calls, "cash_secured_puts": puts, "iron_condors": condors,
            "summary": {"total_premium": round(total, 2), "overlay_premium": round(overlay, 2),
                        "multileg_premium": round(multileg, 2), "capital": capital,
                        "monthly_yield_pct": round(monthly, 2), "annualized_yield_pct": round(monthly * 12.0, 2),
                        "covered_call_count": len(calls), "csp_count": len(puts),
                        "iron_condor_count": len(condors)},
            "log": log}
