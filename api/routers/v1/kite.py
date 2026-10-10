"""/api/v1/kite/* routes: Kite session, quotes, holdings, positions, P&L, orders.

Moved from v1_gateway.py (tracker H4), which includes this router under /api/v1.
"""

import asyncio
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.dependencies import kite_call, get_kite_session
from api.routers.v1.common import KITE_SESSION_INACTIVE, logger

router = APIRouter()

_CENTURION_POSITIONS_KEY = "kite:centurion_positions"


# ─── Kite Connect ───────────────────────────────────────────────────────

@router.get("/kite/session/status")
async def kite_session_status():
    """Check if a Kite session is currently active."""
    kite = get_kite_session()
    if not kite:
        return {"active": False, "profile": None}
    try:
        profile = await kite_call(kite.profile)
        return {"active": True, "profile": profile}
    except Exception:
        return {"active": False, "profile": None}


@router.post("/kite/session/start")
async def kite_session_start():
    """Start a Kite Connect session: the current one, today's stored token, or a manual login.

    Today's token is the one stored by the daily login (tracker U23).  There is
    no automated login: without a token the frontend gets ``needs_login`` and
    the user completes Zerodha's own login.
    """
    from api.dependencies import set_kite_session, is_kite_token_expiring_soon

    # If already active and not expiring soon, return immediately
    existing = get_kite_session()
    if existing and not is_kite_token_expiring_soon():
        try:
            profile = await asyncio.to_thread(existing.profile)
            return {"success": True, "profile": profile, "message": "Session already active"}
        except Exception:
            pass  # session expired, fall through

    # Today's token from the daily login (U23)
    try:
        from kite_connect.auth.daily_login import kite_from_stored_token
        kite = await asyncio.to_thread(kite_from_stored_token)
        if kite:
            set_kite_session(kite)
            profile = await kite_call(kite.profile)
            return {"success": True, "profile": profile}
    except Exception as e:
        logger.info("No usable stored Kite token: %s", e)

    from kite_connect.core.config import LOGIN_URL
    return {
        "success": False,
        "needs_login": True,
        "login_url": LOGIN_URL,
        "message": "No Kite session today. Please complete the Kite login.",
    }


class KiteTokenRequest(BaseModel):
    request_token: str


@router.post("/kite/session/complete")
async def kite_session_complete(body: KiteTokenRequest):
    """Complete a Kite session using a manually-provided request_token.

    Called after the user completes OAuth login and obtains a request_token
    from the Kite redirect URL.
    """
    import os
    from api.dependencies import set_kite_session
    from kiteconnect import KiteConnect

    try:
        from kite_connect.core.config import API_KEY, API_SECRET
        pool_cfg = {"pool_maxsize": int(os.getenv("KITE_POOL_MAXSIZE", "20"))}
        kite = KiteConnect(api_key=API_KEY, pool=pool_cfg)
        data = await asyncio.to_thread(
            kite.generate_session, body.request_token, API_SECRET,
        )
        from kite_connect.auth.daily_login import check_user
        check_user(str(data.get("user_id") or ""))          # the same user check as the login callback
        kite.set_access_token(data["access_token"])
        set_kite_session(kite)

        profile = await kite_call(kite.profile)
        return {"success": True, "profile": profile}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Kite session complete failed: %s", e, exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/kite/session/stop")
async def kite_session_stop():
    """Disconnect the active Kite session."""
    from api.dependencies import set_kite_session

    kite = get_kite_session()
    if kite:
        try:
            await kite_call(kite.invalidate_access_token)
        except Exception:
            pass
    set_kite_session(None)
    return {"success": True}


@router.get("/kite/session/status")
async def kite_session_status():
    """Return Kite session health: active, remaining time, expiring flag."""
    from api.dependencies import is_kite_token_expiring_soon, kite_token_remaining_seconds

    kite = get_kite_session()
    if not kite:
        return {"active": False, "remaining_seconds": 0, "expiring_soon": True}

    remaining = kite_token_remaining_seconds()
    expiring = is_kite_token_expiring_soon()
    return {
        "active": True,
        "remaining_seconds": remaining,
        "remaining_minutes": remaining // 60,
        "expiring_soon": expiring,
    }


@router.get("/kite/quotes")
async def kite_quotes(symbols: str):
    """Get live quotes for comma-separated symbols."""
    kite = get_kite_session()
    if not kite:
        raise HTTPException(status_code=409, detail=KITE_SESSION_INACTIVE)
    try:
        from kite_connect.core.quotes import get_batch_quotes
        syms = [s.strip() for s in symbols.split(",") if s.strip()]
        quotes = await asyncio.to_thread(get_batch_quotes, kite, syms)
        return quotes
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _centurion_positions() -> Dict[str, int]:
    """What Centurion bought in your account (the live book's ledger), by symbol; {} when unreadable.

    Cached for 5 minutes: the ledger changes once per evening session.
    """
    from infrastructure.cache import cache
    from kite_connect.auth import accounts

    cached = cache.get(_CENTURION_POSITIONS_KEY)
    if cached is not None:
        return cached
    try:
        positions = accounts.ledger_positions(accounts.primary_account())
    except Exception as exc:                              # noqa: BLE001 - the holdings still load, unmarked
        logger.warning("Centurion's ledger unavailable: %s", exc)
        return {}
    cache.set(_CENTURION_POSITIONS_KEY, positions, ttl=300)
    return positions


@router.get("/kite/holdings")
async def kite_holdings():
    """Get portfolio holdings, each with ``centurion_qty``: the shares Centurion bought (tracker FK1)."""
    kite = get_kite_session()
    if not kite:
        raise HTTPException(status_code=409, detail=KITE_SESSION_INACTIVE)
    try:
        holdings = await kite_call(kite.holdings)
        mine = await asyncio.to_thread(_centurion_positions)
        for h in holdings or []:
            held = int(h.get("quantity") or 0) + int(h.get("t1_quantity") or 0)
            h["centurion_qty"] = min(int(mine.get(h.get("tradingsymbol"), 0)), held)
        return holdings
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/kite/positions")
async def kite_positions():
    """Get current positions."""
    kite = get_kite_session()
    if not kite:
        raise HTTPException(status_code=409, detail=KITE_SESSION_INACTIVE)
    try:
        positions = await kite_call(kite.positions)
        return positions.get("net", [])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/kite/portfolio/pnl")
async def kite_portfolio_pnl():
    """Real-time portfolio P&L summary.

    Aggregates net positions and holdings into a single P&L view
    with total invested, current value, unrealised P&L, and
    day change.
    """
    kite = get_kite_session()
    if not kite:
        raise HTTPException(status_code=409, detail=KITE_SESSION_INACTIVE)
    try:
        positions_data = await kite_call(kite.positions)
        holdings_data = await kite_call(kite.holdings)

        net_positions = positions_data.get("net", [])
        total_pnl = 0.0
        total_invested = 0.0
        total_current = 0.0
        day_pnl = 0.0
        position_details = []

        for p in net_positions:
            qty = p.get("quantity", 0)
            if qty == 0:
                continue
            avg = p.get("average_price", 0)
            ltp = p.get("last_price", 0)
            pnl = p.get("pnl", 0)
            day_m2m = p.get("day_m2m", 0)
            invested = abs(qty) * avg
            current = abs(qty) * ltp

            total_pnl += pnl
            total_invested += invested
            total_current += current
            day_pnl += day_m2m

            position_details.append({
                "symbol": p.get("tradingsymbol", ""),
                "quantity": qty,
                "avg_price": round(avg, 2),
                "ltp": round(ltp, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round((pnl / invested * 100) if invested else 0, 2),
                "day_change": round(day_m2m, 2),
            })

        # Add holdings (CNC delivery positions)
        for h in (holdings_data or []):
            qty = h.get("quantity", 0)
            if qty == 0:
                continue
            avg = h.get("average_price", 0)
            ltp = h.get("last_price", 0)
            pnl = h.get("pnl", 0)
            day_change = h.get("day_change", 0)
            invested = qty * avg
            current = qty * ltp

            total_pnl += pnl
            total_invested += invested
            total_current += current
            day_pnl += day_change * qty

            position_details.append({
                "symbol": h.get("tradingsymbol", ""),
                "quantity": qty,
                "avg_price": round(avg, 2),
                "ltp": round(ltp, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round((pnl / invested * 100) if invested else 0, 2),
                "day_change": round(day_change * qty, 2),
                "holding": True,
            })

        return {
            "total_invested": round(total_invested, 2),
            "total_current": round(total_current, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(
                (total_pnl / total_invested * 100) if total_invested else 0, 2
            ),
            "day_pnl": round(day_pnl, 2),
            "positions": position_details,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/kite/orders")
async def kite_orders():
    """Get order book."""
    kite = get_kite_session()
    if not kite:
        raise HTTPException(status_code=409, detail=KITE_SESSION_INACTIVE)
    try:
        orders = await kite_call(kite.orders)
        return orders
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/kite/orders")
async def kite_place_order(order: Dict[str, Any]):
    """Place an order via Kite (swing and positional only: an intraday order is refused)."""
    from kite_connect.trading.order_service import intraday_refusal, kill_switch_refusal, same_day_refusal

    refusal = intraday_refusal(order.get("product"), order.get("variety", "regular"))
    if refusal:
        raise HTTPException(status_code=422, detail=refusal)
    kite = get_kite_session()
    if not kite:
        raise HTTPException(status_code=409, detail=KITE_SESSION_INACTIVE)
    refusal = await asyncio.to_thread(same_day_refusal, kite, order.get("tradingsymbol"), order.get("exchange"),
                                      order.get("transaction_type"), order.get("product"),
                                      order.get("variety", "regular"))
    if refusal:
        raise HTTPException(status_code=422, detail=refusal)
    refusal = await asyncio.to_thread(kill_switch_refusal, kite, order.get("tradingsymbol"), order.get("exchange"),
                                      order.get("transaction_type"), order.get("quantity"), order.get("product"))
    if refusal:                                           # tracker DM0: the kill switch holds here too
        raise HTTPException(status_code=423, detail=refusal)
    try:
        variety = order.pop("variety", "regular")
        order_id = await asyncio.to_thread(
            kite.place_order,
            variety=variety,
            **order,
        )
        return {"order_id": str(order_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
