"""/api/v1/drivewealth/* routes: the DriveWealth (US) broker.

Moved from v1_gateway.py (tracker H4), which includes this router under /api/v1.

Real orders are off unless ``CENTURION_US_ORDERS_ENABLED=true`` (tracker US1): the
US book is advisory only (``us_engine.advisory`` emails, you place every order), and
/carver-orders would buy every plan at market with no sells, position or risk
checks.  Off, /orders is refused and /carver-orders returns its plans as a dry run.
Options are refused always: a non-US account cannot buy or sell (or write) US
options on DriveWealth (the owner's fact, 16 Oct 2026); Indian options trade
through Kite.
"""

import asyncio
import os
import re
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

ENV_US_ORDERS = "CENTURION_US_ORDERS_ENABLED"
#: An OCC option symbol: root, expiry YYMMDD, C or P, strike x 1000 (e.g. "AAPL  261120C00200000").
_OCC_OPTION = re.compile(r"^[A-Z][A-Z.]{0,5}\s*\d{6}[CP]\d{8}$")


def us_option_refusal(symbol: str) -> Optional[str]:
    """Why ``symbol`` may not be ordered as a US option, or None for a stock symbol."""
    if _OCC_OPTION.match(str(symbol or "").strip().upper()):
        return "US options cannot be bought or sold from a non-US DriveWealth account (Indian options go through Kite)"
    return None


def us_orders_enabled() -> bool:
    """Whether these routes may send real DriveWealth orders (off by default)."""
    return os.environ.get(ENV_US_ORDERS, "false").strip().lower() in ("true", "1", "yes")


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
    """Place a DriveWealth order (only with ``CENTURION_US_ORDERS_ENABLED=true``)."""
    if not us_orders_enabled():
        raise HTTPException(status_code=403, detail=f"US orders are off ({ENV_US_ORDERS}): the US book is "
                                                    "advisory only (tracker US1)")
    refusal = us_option_refusal(req.symbol)
    if refusal:
        raise HTTPException(status_code=403, detail=refusal)
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
    on the requested US tickers, then submits BUY orders through DriveWealth
    only when ``CENTURION_US_ORDERS_ENABLED=true`` and DriveWealth is connected;
    otherwise returns the plans as a dry run.
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
        live = bool(client and account_id and client.is_authenticated) and us_orders_enabled()

        if live:
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
            "dry_run": not live,
            "orders_enabled": us_orders_enabled(),
            "pipeline_log": result.pipeline_log,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
