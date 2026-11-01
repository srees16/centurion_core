"""/api/v1/options/* routes: indices, expiries, chains, overlay scan.

Moved from v1_gateway.py (tracker H4), which includes this router under /api/v1.
"""

import asyncio
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.dependencies import get_kite_session, require_session
from api.routers.v1.common import logger

router = APIRouter()


class OverlayScanRequest(BaseModel):
    symbols: Optional[List[str]] = None    # default: the names the deployed book holds or is buying
    capital: Optional[float] = None        # default: the account's available cash


# ─── Options ─────────────────────────────────────────────────────────────

@router.get("/options/indices")
async def options_indices():
    """Get index quotes."""
    kite = get_kite_session()
    if not kite:
        # Return static data
        return [
            {"index": "NIFTY 50", "ltp": 0, "change": 0, "change_pct": 0},
            {"index": "BANK NIFTY", "ltp": 0, "change": 0, "change_pct": 0},
        ]
    try:
        from kite_connect.nse.index_data import get_index_quotes
        return await asyncio.to_thread(get_index_quotes, kite)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _kite():
    kite = get_kite_session()
    if not kite:
        raise HTTPException(status_code=503, detail="Kite is not connected: log in first.")
    return kite


@router.get("/options/expiries")
async def options_expiries(symbol: str):
    """The listed expiries of an option symbol still to come."""
    kite = _kite()
    from kite_connect.options.chain import get_expiry_dates
    try:
        return await asyncio.to_thread(get_expiry_dates, kite, symbol)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Option expiries failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/options/chain")
async def options_chain(symbol: str, expiry: str):
    """One row per strike for a symbol and expiry (YYYY-MM-DD): live quotes, IV and Greeks."""
    kite = _kite()
    from kite_connect.options.chain import get_option_chain
    try:
        return await asyncio.to_thread(get_option_chain, kite, symbol, expiry)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Option chain failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/options/overlay/scan")
async def options_overlay_scan(req: OverlayScanRequest = OverlayScanRequest(), user: dict = Depends(require_session)):
    """Covered calls, cash-secured puts and iron condors on real holdings, listed contracts and quotes.

    Read only: it places nothing.  It reads the owner's Kite holdings and cash, so it is the owner's alone.
    """
    from api.routers.v1.kite_accounts import _owner

    if _owner(user):
        raise HTTPException(status_code=403, detail="The overlay scan reads the owner's holdings.")
    kite = _kite()
    from kite_connect.options.chain import overlay_scan
    try:
        return await asyncio.to_thread(overlay_scan, kite, req.symbols, req.capital)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"No IV history on this server: {e}")
    except Exception as e:
        logger.exception("Options overlay scan failed")
        raise HTTPException(status_code=500, detail=str(e))
