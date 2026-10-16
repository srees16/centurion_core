"""/api/v1/us-advisory: the US advisory book for the web UI (tracker US-UI1), read-only.

The book is the one ``us_engine.advisory`` saves after each US close (Neon schema
``us_advisory``): the next open's suggestions, the holdings, the equity curve and
the latest trades.  Advisory only, so there is no order route; a signed-up user
(role "user") is refused, as for the paper books (``api.main``).
"""

import asyncio

from fastapi import APIRouter, HTTPException

from api.routers.v1.common import _sanitize_floats

router = APIRouter()


@router.get("/us-advisory")
async def us_advisory():
    """The model book as the US Advisory page shows it (``us_engine.advisory.ui_view``)."""
    from api.routers.v1.screener import _cloud_or_none
    from us_engine.advisory import SCHEMA, STATE_KEY, Book, _capital, ui_view

    cloud = await asyncio.to_thread(_cloud_or_none, SCHEMA)
    if cloud is None:
        raise HTTPException(status_code=503, detail="the US advisory book is only in Neon, which is not configured")
    try:
        raw = (await asyncio.to_thread(cloud.read_state)).get(STATE_KEY)
    except Exception as e:                                # noqa: BLE001 - a DB error is never an empty book
        raise HTTPException(status_code=500, detail=f"the US advisory book could not be read: {e}")
    capital, capital_set = _capital(None)
    return _sanitize_floats(ui_view(Book.from_json(raw) if raw else None, capital, capital_set))
