"""/api/v1/options/* routes: indices, expiries, chains, overlay scan.

Moved from v1_gateway.py (tracker H4), which includes this router under /api/v1.
"""

import asyncio
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.dependencies import get_kite_session
from api.routers.v1.common import logger

router = APIRouter()


class OverlayScanRequest(BaseModel):
    symbols: Optional[List[str]] = None
    capital: float = 500_000
    regime: str = "RANGE_BOUND"


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


@router.get("/options/expiries")
async def options_expiries(symbol: str):
    """Get available expiry dates for an option symbol."""
    try:
        from kite_connect.options.chain import get_expiry_dates
        return await asyncio.to_thread(get_expiry_dates, symbol)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/options/chain")
async def options_chain(symbol: str, expiry: str):
    """Get option chain for a symbol and expiry."""
    try:
        from kite_connect.options.chain import get_option_chain
        return await asyncio.to_thread(get_option_chain, symbol, expiry)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/options/overlay/scan")
async def options_overlay_scan(req: OverlayScanRequest = OverlayScanRequest()):
    """Run full options overlay scan — covered calls, CSPs, iron condors, strangles.

    Returns combined strategy recommendations with yield estimates.
    """
    symbols = req.symbols
    capital = req.capital
    regime = req.regime
    try:
        from services.execution.options_overlay import OptionsOverlay
        from services.execution.iron_condor_strangle import IronCondorStrangleOverlay
        from services.signals.oi_signal import FNO_LOT_SIZES

        # ── Gather IV data ──────────────────────────
        iv_data: Dict[str, Dict] = {}
        spot_prices: Dict[str, float] = {}
        try:
            from services.signals.iv_rank import compute_iv_ranks_batch
            from infrastructure.cache import ohlcv_cache_store
            ohlcv_cache = ohlcv_cache_store.get_all() if hasattr(ohlcv_cache_store, "get_all") else {}
            iv_ranks = await asyncio.to_thread(compute_iv_ranks_batch, ohlcv_cache)
            iv_data = {
                sym: {"iv": ivr.current_iv, "iv_rank": ivr.iv_rank}
                for sym, ivr in iv_ranks.items()
            }
        except Exception:
            pass

        # ── Gather spot prices via Kite / cache ─────
        kite = get_kite_session()
        fno_symbols = symbols or list(FNO_LOT_SIZES.keys())
        if kite:
            try:
                from kite_connect.nse.live_quotes import get_nse_quotes
                quotes = await asyncio.to_thread(get_nse_quotes, kite, fno_symbols)
                spot_prices = {sym: q.get("last_price", 0) for sym, q in quotes.items() if q.get("last_price")}
            except Exception:
                pass

        # Fill in any missing IV data with defaults
        for sym in fno_symbols:
            if sym not in iv_data:
                iv_data[sym] = {"iv": 0.25, "iv_rank": 55.0}
            if sym not in spot_prices:
                spot_prices[sym] = 0

        # ── Covered Calls + CSPs ────────────────────
        overlay = OptionsOverlay()
        # For covered calls: treat all held positions as potential CC targets
        holdings_dict = {
            sym: {"quantity": FNO_LOT_SIZES.get(sym, 25), "avg_price": p, "current_price": p}
            for sym, p in spot_prices.items() if p > 0
        }
        # For CSPs: candidates with positive implied forecast
        csp_candidates = {
            sym: {"current_price": p, "forecast": 10.0}
            for sym, p in spot_prices.items() if p > 0
        }

        overlay_result = await asyncio.to_thread(
            overlay.run_overlay,
            holdings=holdings_dict,
            candidates=csp_candidates,
            iv_data=iv_data,
            available_capital=capital,
        )

        # ── Iron Condors + Strangles ────────────────
        ic_overlay = IronCondorStrangleOverlay()
        iv_rank_map = {sym: iv_data[sym].get("iv_rank", 0) for sym in iv_data}
        lot_sizes = {sym: FNO_LOT_SIZES.get(sym, 25) for sym in fno_symbols}

        ic_result = await asyncio.to_thread(
            ic_overlay.scan_all,
            symbols=list(iv_rank_map.keys()),
            spot_prices=spot_prices,
            iv_data=iv_rank_map,
            available_capital=capital,
            regime=regime,
            lot_sizes=lot_sizes,
        )

        # ── Build response ──────────────────────────
        return {
            "covered_calls": [
                {
                    "symbol": o.symbol, "strike": o.strike, "expiry": o.expiry_date,
                    "premium": o.premium, "total_premium": o.total_premium,
                    "delta": o.delta, "iv": o.iv, "lots": o.lots, "lot_size": o.lot_size,
                    "underlying_price": o.underlying_price,
                }
                for o in overlay_result.covered_call_orders
            ],
            "cash_secured_puts": [
                {
                    "symbol": o.symbol, "strike": o.strike, "expiry": o.expiry_date,
                    "premium": o.premium, "total_premium": o.total_premium,
                    "delta": o.delta, "iv": o.iv, "lots": o.lots, "lot_size": o.lot_size,
                    "underlying_price": o.underlying_price,
                }
                for o in overlay_result.put_write_orders
            ],
            "iron_condors": [ic.to_dict() for ic in ic_result.iron_condors],
            "strangles": [sg.to_dict() for sg in ic_result.strangles],
            "summary": {
                "total_premium": round(
                    overlay_result.total_premium_expected + ic_result.total_premium, 2
                ),
                "overlay_premium": overlay_result.total_premium_expected,
                "multileg_premium": ic_result.total_premium,
                "monthly_yield_pct": overlay_result.monthly_yield_pct,
                "annualized_yield_pct": overlay_result.annualized_yield_pct,
                "covered_call_count": len(overlay_result.covered_call_orders),
                "csp_count": len(overlay_result.put_write_orders),
                "iron_condor_count": len(ic_result.iron_condors),
                "strangle_count": len(ic_result.strangles),
            },
            "log": overlay_result.log + ic_result.log,
        }
    except ImportError as e:
        raise HTTPException(status_code=501, detail=f"Module not available: {e}")
    except Exception as e:
        logger.exception("Options overlay scan failed")
        raise HTTPException(status_code=500, detail=str(e))
