"""
API v1 Gateway Router — maps Next.js frontend paths to existing service logic.

This router provides the /api/v1/* endpoints expected by the Next.js frontend,
delegating to existing internal modules. Missing functionality (DriveWealth,
FML, TTS chapters) is stubbed with minimal implementations.

The routes live in ``api/routers/v1/``, one module per path prefix (tracker H4);
they are included here in their original order, so route matching is unchanged.
"""

from fastapi import APIRouter

from api.routers.v1 import (analysis, auth, backtest, drivewealth, history, kite, kite_accounts, labs, macro,
                            market, options, rag, rl_bot, screener, verdict)

router = APIRouter(prefix="/api/v1", tags=["API v1"])
for _module in (auth, analysis, macro, backtest, verdict, history, screener, kite, kite_accounts, options,
                drivewealth, labs, rag, market, rl_bot):
    router.include_router(_module.router)
