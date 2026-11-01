"""
The app's Options page, on the toolkit's live chain (``live_chain``) and scan (``overlay_scan``).

``chain_rows`` turns the toolkit's one-row-per-contract chain into the page's one
row per strike.  Each strike's call and put show the strike's single IV
(``iv_strike``, the OTM option's, by put-call parity), and an option that did
not solve for its own IV takes its Greeks from it.  ``moneyness`` reads from the
call's side.  Missing values are None: the page shows a dash, never a zero.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from datetime import date, datetime
from typing import List, Optional, Tuple

import pandas as pd

from kite_connect.options.broker import Broker
from kite_connect.options.instruments import InstrumentResolver
from kite_connect.options.live_chain import IST, chain_summary, days_to_expiry, fetch_chain
from kite_connect.options.options_config import OptionsConfig, load_config
from kite_connect.options.theory import CALL, PUT, black_scholes

logger = logging.getLogger(__name__)

_cached: Optional[Tuple[int, date, Broker, InstrumentResolver]] = None
_lock = threading.Lock()


def session(kite) -> Tuple[Broker, InstrumentResolver]:
    """One broker a Kite session, so calls share its throttle, and one instruments dump a day."""
    global _cached
    today = datetime.now(IST).date()
    with _lock:
        if _cached is None or _cached[0] != id(kite) or _cached[1] != today:
            broker = Broker(kite)
            _cached = id(kite), today, broker, InstrumentResolver(broker.instruments())
        return _cached[2], _cached[3]


def _clean(x) -> Optional[float]:
    return float(x) if x is not None and math.isfinite(x) else None


def _side(row: Optional[pd.Series], option_type: str, spot: float, dte: float, rate: float) -> dict:
    p = f"{option_type.lower()}_"
    if row is None:
        return {f"{p}{k}": None for k in ("oi", "volume", "iv", "ltp", "delta", "gamma", "theta", "vega")}
    iv = row["iv_strike"]
    greeks = {k: _clean(row[k]) for k in ("delta", "gamma", "theta", "vega")}
    if greeks["delta"] is None and math.isfinite(iv):
        g = black_scholes(option_type, spot, row["strike"], dte, rate, iv)
        greeks = {k: getattr(g, k) for k in greeks}
    return {f"{p}oi": _clean(row["oi"]), f"{p}volume": _clean(row["volume"]), f"{p}ltp": _clean(row["ltp"]),
            f"{p}iv": round(iv * 100.0, 2) if math.isfinite(iv) else None, **{p + k: v for k, v in greeks.items()}}


def chain_rows(chain: pd.DataFrame, spot: float, dte: float, rate: float) -> List[dict]:
    """One row per strike for the page."""
    atm = chain_summary(chain, spot)["atm_strike"]
    rows = []
    for strike, at in chain.groupby("strike", sort=True):
        by_type = {r["option_type"]: r for _, r in at.iterrows()}
        rows.append({"strike": float(strike),
                     "moneyness": "ATM" if strike == atm else "ITM" if strike < spot else "OTM",
                     **_side(by_type.get(CALL), CALL, spot, dte, rate),
                     **_side(by_type.get(PUT), PUT, spot, dte, rate)})
    return rows


def get_expiry_dates(kite, symbol: str) -> List[str]:
    """The listed expiries still to come, ISO dates."""
    _, resolver = session(kite)
    today = datetime.now(IST).date()
    return [e.isoformat() for e in resolver.expiries(symbol) if e >= today]


def get_option_chain(kite, symbol: str, expiry: str, cfg: Optional[OptionsConfig] = None) -> List[dict]:
    """The chain of ``symbol`` at ``expiry`` (``YYYY-MM-DD``) around ATM."""
    cfg = cfg or load_config()
    broker, resolver = session(kite)
    when, now, rate = date.fromisoformat(expiry), datetime.now(IST), cfg.market.risk_free_rate
    if when not in resolver.expiries(symbol):
        raise KeyError(f"{symbol} has no {expiry} expiry")
    chain, spot = fetch_chain(broker, resolver, symbol.upper(), when, now, rate, cfg.overlay.chain_strikes_each_side)
    return chain_rows(chain, spot, days_to_expiry(when, now), rate)


def book_wants(log: List[str]) -> List[str]:
    """The deployed book's open positions and pending buys: the names the engine wants to own."""
    from database.connection import get_db_manager
    from database.paper_cloud import PaperCloudSync

    try:
        cloud = PaperCloudSync(get_db_manager(), schema=None)
        positions = cloud.read_open_positions()
        pending = json.loads(cloud.read_state().get("engine_pending_orders") or "[]")
    except Exception as exc:                              # noqa: BLE001 - puts stand down, the scan goes on
        logger.warning("overlay scan: the book could not be read: %s", exc)
        log.append("The book could not be read: no cash-secured puts")
        return []
    held = [] if positions is None or positions.empty else [str(s) for s in positions["symbol"]]
    return list(dict.fromkeys([*held, *(str(o["symbol"]) for o in pending if o.get("side") == "BUY")]))


def overlay_scan(kite, symbols: Optional[List[str]] = None, capital: Optional[float] = None,
                 cfg: Optional[OptionsConfig] = None) -> dict:
    """The Overlay Strategies tab, on the account's real holdings and cash.  Read only."""
    from config import Config
    from kite_connect.options.iv_history import load_snapshot
    from kite_connect.options.overlay_scan import scan

    cfg = cfg or load_config()
    broker, resolver = session(kite)
    held = {str(h["tradingsymbol"]): int(h.get("quantity") or 0) + int(h.get("t1_quantity") or 0)
            for h in broker.holdings()}
    notes: List[str] = []
    wanted = [s.upper() for s in symbols] if symbols else book_wants(notes)
    cash = broker.available_cash() if capital is None else float(capital)
    result = scan(broker, resolver, {s: q for s, q in held.items() if q > 0}, wanted, cash,
                  load_snapshot(cfg), datetime.now(IST), cfg, getattr(Config, "NSE_SECTOR_MAP", {}))
    result["log"] = notes + result["log"]
    return result
