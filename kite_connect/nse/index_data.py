"""Live index quotes through Kite Connect.

``GET /api/v1/options/indices`` imported ``get_index_quotes`` from here, but
the module never existed (Sentry issue 147501345: ModuleNotFoundError on
every call). The frontend's ``IndexQuote`` row is ``{index, ltp, change,
change_pct}``.

Kite quotes indices under the NSE exchange with their display names, e.g.
``NSE:NIFTY 50`` — spaces included.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

#: display name -> Kite instrument key
INDEX_INSTRUMENTS: Dict[str, str] = {
    "NIFTY 50": "NSE:NIFTY 50",
    "BANK NIFTY": "NSE:NIFTY BANK",
    "FIN NIFTY": "NSE:NIFTY FIN SERVICE",
    "MIDCAP NIFTY": "NSE:NIFTY MID SELECT",
    "INDIA VIX": "NSE:INDIA VIX",
}


def _row(name: str, quote: Optional[dict]) -> dict:
    if not quote:
        return {"index": name, "ltp": 0.0, "change": 0.0, "change_pct": 0.0, "available": False}
    ltp = float(quote.get("last_price") or 0.0)
    close = float((quote.get("ohlc") or {}).get("close") or 0.0)
    change = ltp - close if close else 0.0
    return {
        "index": name,
        "ltp": ltp,
        "change": round(change, 2),
        "change_pct": round(change / close * 100.0, 2) if close else 0.0,
        "open": float((quote.get("ohlc") or {}).get("open") or 0.0),
        "high": float((quote.get("ohlc") or {}).get("high") or 0.0),
        "low": float((quote.get("ohlc") or {}).get("low") or 0.0),
        "available": True,
    }


def get_index_quotes(kite, names: Optional[List[str]] = None) -> List[dict]:
    """Quotes for the major NSE indices (all in one Kite request, per-index fallback)."""
    wanted = {n: INDEX_INSTRUMENTS[n] for n in (names or INDEX_INSTRUMENTS) if n in INDEX_INSTRUMENTS}
    quotes: Dict[str, dict] = {}
    try:
        quotes = kite.quote(list(wanted.values())) or {}
    except Exception as exc:                              # noqa: BLE001 - fall back per index
        logger.warning("index quote batch failed (%s); retrying one by one", exc)
        for key in wanted.values():
            try:
                quotes.update(kite.quote([key]) or {})
            except Exception as exc1:                     # noqa: BLE001
                logger.warning("index quote failed for %s: %s", key, exc1)
    return [_row(name, quotes.get(key)) for name, key in wanted.items()]
