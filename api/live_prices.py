"""
Live gold, silver and bitcoin prices for the Ind stocks pages' ribbon (owner's ask, 16 Nov 2026; tracker GSB1).

Gold and silver are India's market prices: MCX's near-month futures through the API's Kite session (in the Kite
Connect subscription, no other cost).  GOLD is quoted per 10 grams and shown per gram; SILVER is quoted and shown
per kilogram.  Live while MCX trades (09:00 to 23:30 IST on weekdays; Kite answers the last trade outside them).
With no Kite session (before the morning login) they are absent, and ``note`` says why: a converted international
price would leave out India's import duty and read several percent low.

Bitcoin is in dollars: Binance's public market-data mirror (no key; data-api.binance.vision serves every region),
with CoinGecko's public price as the fallback.

Every answer is cached ``CACHE_S`` seconds, whoever asks, so the ribbon's polling never reaches Kite's quote limit.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import date
from typing import Any, Callable, Dict, Iterable, Optional

logger = logging.getLogger(__name__)

CACHE_S = 30
#: A contract this many days or fewer from expiry gives way to the next (its last days trade thin).
ROLL_DAYS = 3
#: (MCX name, divisor from the quoted unit to the shown one, the shown unit)
METALS = {"gold": ("GOLD", 10.0, "gram"), "silver": ("SILVER", 1.0, "kg")}
BINANCE = "https://data-api.binance.vision/api/v3/ticker/24hr?symbol=BTCUSDT"
COINGECKO = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true"
TIMEOUT_S = 8

_lock = threading.Lock()
_cache: Dict[str, Any] = {"at": None, "value": None}
_contracts: Dict[str, Any] = {"day": None, "symbols": {}}


def near_month(instruments: Iterable[Dict[str, Any]], name: str, today: date,
               roll_days: int = ROLL_DAYS) -> Optional[Dict[str, Any]]:
    """The nearest ``name`` future more than ``roll_days`` from expiry (Kite's instruments dump), or None."""
    live = [i for i in instruments if i.get("name") == name and i.get("instrument_type") == "FUT"
            and i.get("expiry") and (i["expiry"] - today).days > roll_days]
    return min(live, key=lambda i: i["expiry"]) if live else None


def metal_prices(quotes: Dict[str, Dict[str, Any]], symbols: Dict[str, Optional[str]]) -> Dict[str, Any]:
    """Each metal's price in its shown unit, its change on the previous close (%), the contract and the time."""
    out: Dict[str, Any] = {}
    for key, (_, per, unit) in METALS.items():
        sym = symbols.get(key)
        q = (quotes.get(sym) or {}) if sym else {}
        last, close = q.get("last_price"), (q.get("ohlc") or {}).get("close")
        if not last or last <= 0:
            out[key] = None
            continue
        out[key] = {"price": float(last) / per, "unit": unit, "currency": "INR",
                    "change_pct": (float(last) / float(close) - 1.0) * 100.0 if close else None,
                    "contract": sym.split(":", 1)[1], "quoted": float(last),
                    "as_of": str(q.get("last_trade_time") or q.get("timestamp") or "")}
    return out


def _symbols(kite: Any, today: date) -> Dict[str, Optional[str]]:
    """Today's near-month GOLD and SILVER contracts (the instruments dump is read once a day)."""
    if _contracts["day"] != today:
        dump = kite.instruments("MCX")
        symbols: Dict[str, Optional[str]] = {}
        for key, (name, _, _) in METALS.items():
            contract = near_month(dump, name, today)
            symbols[key] = f"MCX:{contract['tradingsymbol']}" if contract else None
        _contracts.update(day=today, symbols=symbols)
    return _contracts["symbols"]


def bitcoin(get: Optional[Callable[..., Any]] = None) -> Optional[Dict[str, Any]]:
    """Bitcoin in dollars: Binance's BTC/USDT, else CoinGecko; None when both fail."""
    if get is None:
        import requests

        get = requests.get
    try:
        d = get(BINANCE, timeout=TIMEOUT_S).json()
        return {"price": float(d["lastPrice"]), "change_pct": float(d["priceChangePercent"]), "currency": "USD",
                "source": "Binance BTC/USDT"}
    except Exception as exc:                              # noqa: BLE001 - the fallback answers
        logger.info("bitcoin from Binance failed (%s): CoinGecko", type(exc).__name__)
    try:
        d = get(COINGECKO, timeout=TIMEOUT_S).json()["bitcoin"]
        return {"price": float(d["usd"]), "change_pct": float(d.get("usd_24h_change") or 0.0), "currency": "USD",
                "source": "CoinGecko"}
    except Exception as exc:                              # noqa: BLE001 - shown as absent, never an error
        logger.warning("bitcoin price unavailable: %s", type(exc).__name__)
        return None


def api_kite() -> Any:
    """The API's Kite session, else today's stored token (as the Kite routes do); None without one."""
    from api.dependencies import get_kite_session, set_kite_session

    kite = get_kite_session()
    if kite is not None:
        return kite
    try:
        from kite_connect.auth.daily_login import kite_from_stored_token

        kite = kite_from_stored_token()
        if kite is not None:
            set_kite_session(kite)
        return kite
    except Exception as exc:                              # noqa: BLE001 - no session is an answer, not a failure
        logger.info("no Kite session for MCX prices: %s", type(exc).__name__)
        return None


def snapshot(kite_factory: Callable[[], Any] = api_kite, today: Optional[date] = None,
             clock: Callable[[], float] = time.monotonic, get: Optional[Callable[..., Any]] = None) -> Dict[str, Any]:
    """Gold, silver and bitcoin now (cached ``CACHE_S`` seconds), with ``note`` saying why a price is absent."""
    with _lock:
        now = clock()
        if _cache["at"] is not None and now - _cache["at"] < CACHE_S:
            return _cache["value"]
        out: Dict[str, Any] = {"gold": None, "silver": None, "bitcoin": bitcoin(get), "note": None}
        kite = kite_factory()
        if kite is None:
            out["note"] = "Gold and silver need today's Kite login (MCX prices)"
        else:
            try:
                symbols = _symbols(kite, today or date.today())
                wanted = [s for s in symbols.values() if s]
                out.update(metal_prices(kite.quote(wanted) if wanted else {}, symbols))
            except Exception as exc:                      # noqa: BLE001 - shown as absent with the reason
                logger.warning("MCX prices unavailable: %s", exc)
                out["note"] = f"MCX prices unavailable ({type(exc).__name__})"
        _cache.update(at=now, value=out)
        return out
