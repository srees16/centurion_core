"""/api/v1/history/* routes: stored signals and backtests.

Moved from v1_gateway.py (tracker H4), which includes this router under /api/v1.
"""

from fastapi import APIRouter, HTTPException

from api.dependencies import get_db_service

router = APIRouter()


# ─── History ─────────────────────────────────────────────────────────────

@router.get("/history/signals")
async def history_signals(market: str = "US", page: int = 1, limit: int = 50):
    """Get signal history from the database."""
    db = get_db_service()
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        data = db.get_signal_history(market=market, page=page, limit=limit)
        total = db.count_signals(market=market)
        return {"data": data, "total": total}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/backtests")
async def history_backtests(market: str = "US", page: int = 1, limit: int = 50):
    """Get backtest history from the database."""
    db = get_db_service()
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        data = db.get_backtest_history(market=market, page=page, limit=limit)
        total = db.count_backtests(market=market)
        return {"data": data, "total": total}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
