"""/api/v1/macro/* routes: macro indicators, fear & greed, portfolio risk.

Moved from v1_gateway.py (tracker H4), which includes this router under /api/v1.
"""

import asyncio

from fastapi import APIRouter, HTTPException

router = APIRouter()


# ─── Macro Indicators ────────────────────────────────────────────────────

@router.get("/macro/snapshot")
async def macro_snapshot(market: str = "IND"):
    """Get macro-economic indicators (VIX, yields, commodities)."""
    try:
        from scrapers.macro.macro_indicators import MacroIndicators
        mi = MacroIndicators()
        snap = await asyncio.to_thread(mi.fetch, market=market)
        return {
            "vix": snap.vix if market == "US" else snap.india_vix,
            "vix_label": "CBOE VIX" if market == "US" else "India VIX",
            "index_name": "S&P 500" if market == "US" else "Nifty 50",
            "index_price": snap.sp500_price if market == "US" else snap.nifty50_price,
            "index_change_pct": snap.sp500_change_pct if market == "US" else snap.nifty50_change_pct,
            "us_10y_yield": snap.us_10y_yield,
            "gold_price": snap.gold_price,
            "crude_oil_price": snap.crude_oil_price,
            "macro_sentiment_label": snap.macro_sentiment_label,
            "macro_sentiment_score": snap.macro_sentiment_score,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/macro/fear-greed")
async def macro_fear_greed():
    """Get India Fear & Greed index."""
    try:
        from scrapers.macro.macro_indicators import MacroIndicators
        from scrapers.macro.india_fear_greed import IndiaFearGreedIndex
        from scrapers.ind_news.fii_dii_flows import FIIDIIFlows

        mi = MacroIndicators()
        snap = await asyncio.to_thread(mi.fetch, market="IND")
        flows = await FIIDIIFlows().fetch()
        fg = IndiaFearGreedIndex()
        result = await fg.compute(
            india_vix=snap.india_vix,
            fii_net_crore=flows.fii_net,
            nifty_change_pct=snap.nifty50_change_pct,
        )
        return {"score": result.score, "label": result.label}
    except Exception as e:
        return {"score": None, "label": "N/A"}


@router.get("/macro/portfolio-risk")
async def macro_portfolio_risk(market: str = "IND"):
    """Get portfolio risk snapshot (drawdown, vol, concentration)."""
    try:
        from services.risk.portfolio_vol_monitor import assess_portfolio_risk

        # Attempt to gather live position data from Kite (IND) or DriveWealth (US)
        position_values: dict = {}
        instrument_vols: dict = {}
        total_capital = 500_000.0
        peak_equity = None

        if market == "IND":
            try:
                from auth.shared_session import get_kite
                kite = get_kite()
                if kite:
                    positions = kite.positions().get("net", [])
                    for p in positions:
                        sym = p.get("tradingsymbol", "")
                        qty = p.get("quantity", 0)
                        ltp = p.get("last_price", 0)
                        if qty != 0 and ltp > 0:
                            position_values[sym] = abs(qty * ltp)
                            instrument_vols[sym] = 0.02  # ~32% annual vol default
            except Exception:
                pass

        snap = await asyncio.to_thread(
            assess_portfolio_risk,
            position_values=position_values,
            instrument_daily_vols=instrument_vols,
            total_capital=total_capital,
            peak_equity=peak_equity,
        )
        return {
            "timestamp": snap.timestamp,
            "portfolio_daily_vol": snap.portfolio_daily_vol,
            "portfolio_annual_vol_pct": snap.portfolio_annual_vol_pct,
            "target_annual_vol_pct": snap.target_annual_vol_pct,
            "vol_ratio": snap.vol_ratio,
            "hhi": snap.hhi,
            "largest_position_pct": snap.largest_position_pct,
            "peak_equity": snap.peak_equity,
            "current_equity": snap.current_equity,
            "drawdown_pct": snap.drawdown_pct,
            "risk_level": snap.risk_level.value if hasattr(snap.risk_level, 'value') else str(snap.risk_level),
            "scale_factor": snap.scale_factor,
            "emergency_liquidate": snap.emergency_liquidate,
            "alerts": snap.alerts,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
