"""/api/v1/drivewealth/* routes: the DriveWealth (US) broker.

Moved from v1_gateway.py (tracker H4), which includes this router under /api/v1.
"""

import asyncio
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class DWLoginRequest(BaseModel):
    client_id: str
    client_secret: str
    app_key: str
    user_id: str
    account_id: str


class OrderRequest(BaseModel):
    symbol: str
    side: str
    order_type: str
    quantity: int
    limit_price: Optional[float] = None


# ─── DriveWealth ─────────────────────────────────────────────────────────

_dw_session: Dict[str, Any] = {}


@router.post("/drivewealth/login")
async def dw_login(req: DWLoginRequest):
    """Login to DriveWealth API."""
    try:
        from services.execution.drivewealth import DriveWealthClient
        client = DriveWealthClient(
            client_id=req.client_id,
            client_secret=req.client_secret,
            app_key=req.app_key,
        )
        token = await asyncio.to_thread(client.authenticate)
        account = await asyncio.to_thread(client.get_account, req.account_id)
        _dw_session["client"] = client
        _dw_session["account_id"] = req.account_id
        return {"token": token, "account": account}
    except ImportError:
        raise HTTPException(status_code=501, detail="DriveWealth module not installed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/drivewealth/account")
async def dw_account():
    """Get DriveWealth account info."""
    client = _dw_session.get("client")
    if not client:
        raise HTTPException(status_code=401, detail="Not connected to DriveWealth")
    try:
        account = await asyncio.to_thread(client.get_account, _dw_session["account_id"])
        return account
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/drivewealth/positions")
async def dw_positions():
    """Get DriveWealth positions."""
    client = _dw_session.get("client")
    if not client:
        raise HTTPException(status_code=401, detail="Not connected to DriveWealth")
    try:
        positions = await asyncio.to_thread(client.list_positions, _dw_session["account_id"])
        return positions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/drivewealth/orders")
async def dw_place_order(req: OrderRequest):
    """Place a DriveWealth order."""
    client = _dw_session.get("client")
    if not client:
        raise HTTPException(status_code=401, detail="Not connected to DriveWealth")
    try:
        payload = {
            "accountNo": _dw_session["account_id"],
            "symbol": req.symbol,
            "side": req.side,
            "type": req.order_type,
            "quantity": str(req.quantity),
        }
        if req.limit_price is not None:
            payload["price"] = str(req.limit_price)
        result = await asyncio.to_thread(client.create_order, payload)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/drivewealth/carver-orders")
async def dw_carver_orders(tickers: Optional[list] = None):
    """Generate Carver vol-targeted trade plans and optionally place via DriveWealth.

    Runs the full Carver pipeline (EWMAC → forecast → vol-target → sizing)
    on the requested US tickers, then submits BUY orders through DriveWealth.
    If DriveWealth is not connected, returns dry-run plans only.
    """
    try:
        from config import Config
        if not getattr(Config, "CARVER_US_ENABLED", False):
            raise HTTPException(status_code=400, detail="Carver US is not enabled")

        from services.execution.us_carver_pipeline import run_us_carver_pipeline, DEFAULT_US_CARVER_TICKERS

        syms = tickers or DEFAULT_US_CARVER_TICKERS
        result = await asyncio.to_thread(run_us_carver_pipeline, syms)

        placed_orders = []
        client = _dw_session.get("client")
        account_id = _dw_session.get("account_id")

        if client and account_id and client.is_authenticated:
            for plan in result.trade_plans:
                try:
                    order_payload = {
                        "accountNo": account_id,
                        "symbol": plan["symbol"],
                        "side": "BUY",
                        "type": "MARKET",
                        "quantity": str(plan["quantity"]),
                    }
                    order_result = await asyncio.to_thread(client.create_order, order_payload)
                    placed_orders.append({
                        "symbol": plan["symbol"],
                        "quantity": plan["quantity"],
                        "status": "placed",
                        "order_id": order_result.get("orderID", ""),
                    })
                except Exception as exc:
                    placed_orders.append({
                        "symbol": plan["symbol"],
                        "quantity": plan["quantity"],
                        "status": "failed",
                        "error": str(exc),
                    })

        return {
            "success": True,
            "trade_plans": result.trade_plans,
            "orders_placed": placed_orders,
            "dry_run": not bool(client and account_id),
            "pipeline_log": result.pipeline_log,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
