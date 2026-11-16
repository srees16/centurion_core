"""/api/v1/market/* routes: ticker prices for the ribbon.

Moved from v1_gateway.py (tracker H4), which includes this router under /api/v1.
"""

import asyncio
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from api.routers.v1.common import logger

router = APIRouter()


# ─── Market Ticker Prices ────────────────────────────────────────────────

_ticker_price_cache: Dict[str, Any] = {}   # L1 in-memory: cache_key -> response dict
_ticker_cache_ts: Dict[str, float] = {}    # L1 in-memory: cache_key -> monotonic ts
_ticker_cache_lock = asyncio.Lock()        # guards concurrent dict access
_TICKER_CACHE_TTL_OPEN = 10    # seconds – during market hours
_TICKER_CACHE_TTL_CLOSED = 120 # seconds – after market close

def _is_market_open(market: str) -> bool:
    """Check if the stock market is currently open."""
    from datetime import datetime, timezone, timedelta
    now_utc = datetime.now(timezone.utc)
    if market == "IND":
        # NSE: 09:15-15:30 IST on a trading day; the calendar's own reading (holidays, special sessions)
        from nse_engine.nse_calendar import session_status

        return bool(session_status(now_utc)["open"])
    else:
        # NYSE/NASDAQ: 9:30 AM – 4:00 PM ET (approx UTC-4/-5)
        # Use UTC-4 (EDT) as a safe approximation
        et = now_utc - timedelta(hours=4)
        if et.weekday() >= 5:
            return False
        t = et.hour * 60 + et.minute
        return 9 * 60 + 30 <= t < 16 * 60


@router.get("/market/nse-session")
async def market_nse_session():
    """NSE's cash market now: open (09:15-15:30 IST on a trading day), pre-open, closed, weekend, holiday or a
    Muhurat evening, from the one NSE calendar (``nse_engine.nse_calendar``); the Ind stocks pages' badge."""
    from nse_engine.nse_calendar import session_status

    return session_status()


@router.get("/market/live-prices")
async def market_live_prices():
    """Gold (₹ a gram) and silver (₹ a kg) from MCX through the API's Kite session, and bitcoin in dollars
    (``api.live_prices``; tracker GSB1): the Ind stocks pages' ribbon, beside crude."""
    from api import live_prices

    return await asyncio.to_thread(live_prices.snapshot)


@router.get("/market/ticker-prices")
async def market_ticker_prices(symbols: str, market: str = "US"):
    """Get current/last-traded prices for comma-separated ticker symbols."""
    import time
    cache_key = ""
    try:
        import yfinance as yf
        syms = [s.strip() for s in symbols.split(",") if s.strip()]
        if not syms:
            return {"is_market_open": _is_market_open(market), "prices": []}

        # ── cache lookup (lock-protected) ──
        cache_key = f"{market}:{',' .join(sorted(syms))}"
        now = time.monotonic()
        is_open = _is_market_open(market)
        ttl = _TICKER_CACHE_TTL_OPEN if is_open else _TICKER_CACHE_TTL_CLOSED

        async with _ticker_cache_lock:
            if cache_key in _ticker_price_cache and (now - _ticker_cache_ts.get(cache_key, 0)) < ttl:
                return _ticker_price_cache[cache_key]

        # L2: Redis (cross-restart persistence)
        try:
            from infrastructure.cache import cache as _redis_cache
            redis_val = _redis_cache.get(f"price:{cache_key}")
            if redis_val is not None:
                async with _ticker_cache_lock:
                    _ticker_price_cache[cache_key] = redis_val
                    _ticker_cache_ts[cache_key] = now
                return redis_val
        except Exception:
            pass

        # For IND market, append .NS suffix for NSE (with override map)
        if market == "IND":
            from utils import yf_nse_symbol
            yf_syms = [yf_nse_symbol(s) for s in syms]
        else:
            yf_syms = list(syms)

        def _fetch():
            # yf.download is ~4x faster than yf.Tickers for batch fetches
            df = yf.download(
                " ".join(yf_syms),
                period="2d",
                interval="1d",
                group_by="ticker",
                progress=False,
                threads=True,
            )
            results = []
            multi = len(yf_syms) > 1
            for orig, yf_sym in zip(syms, yf_syms):
                try:
                    close = df[yf_sym]["Close"].dropna() if multi else df["Close"].dropna()
                    if len(close) >= 2:
                        price, prev = float(close.iloc[-1]), float(close.iloc[-2])
                    elif len(close) == 1:
                        price, prev = float(close.iloc[-1]), float(close.iloc[-1])
                    else:
                        price, prev = 0, 0
                    change_pct = ((price - prev) / prev * 100) if prev else 0
                    results.append({"symbol": orig, "price": round(price, 2), "change_pct": round(change_pct, 2)})
                except Exception:
                    results.append({"symbol": orig, "price": 0, "change_pct": 0})
            return results

        prices = await asyncio.to_thread(_fetch)
        result = {"is_market_open": is_open, "prices": prices}

        # ── populate cache (L1 + L2, lock-protected) ──
        async with _ticker_cache_lock:
            _ticker_price_cache[cache_key] = result
            _ticker_cache_ts[cache_key] = now
        try:
            from infrastructure.cache import cache as _redis_cache
            _redis_cache.set(f"price:{cache_key}", result, ttl=ttl)
        except Exception:
            pass

        return result
    except Exception as e:
        logger.error(f"ticker-prices error: {e}")
        # Return stale cache on error if available
        async with _ticker_cache_lock:
            stale = _ticker_price_cache.get(cache_key)
        if stale is not None:
            return stale
        raise HTTPException(status_code=500, detail=str(e))
