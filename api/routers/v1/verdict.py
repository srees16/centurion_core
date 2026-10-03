"""/api/v1/verdict/* routes: integrated-scorer verdict runs.

Moved from v1_gateway.py (tracker H4), which includes this router under /api/v1.
"""

import asyncio
from typing import Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.routers.v1.common import logger

router = APIRouter()


class VerdictRunRequest(BaseModel):
    tickers: List[str]
    market: str = "US"
    date_range: List[str] = ["", ""]
    skip_layers: List[str] = []
    weights: Dict[str, float] = {"core": 0.3, "strategy": 0.3, "ml_features": 0.2, "robustness": 0.2}
    batch_size: int = 5


# ─── Verdict ─────────────────────────────────────────────────────────────

@router.post("/verdict/run")
async def verdict_run(req: VerdictRunRequest):
    """Run the multi-layer verdict engine via IntegratedScorer.

    Checks the scheduler verdict cache first; only tickers with a cache
    miss are evaluated live (saves 60-90s per cached ticker).
    """
    try:
        from scheduler import get_cached_verdict
        from services.signals.integrated_scorer import IntegratedScorer

        # --- Serve cached verdicts where available ---
        # Gap C fix: validate cache age — reject entries older than 30 min
        cached_results = []
        uncached_tickers = []
        _MAX_CACHE_AGE_MIN = 30
        for t in req.tickers:
            cached = get_cached_verdict(t)
            if cached:
                try:
                    from datetime import datetime as _dt
                    cached_at = _dt.fromisoformat(cached.get("cached_at", ""))
                    age_min = (_dt.now(cached_at.tzinfo) - cached_at).total_seconds() / 60
                    if age_min > _MAX_CACHE_AGE_MIN:
                        logger.info(
                            "Verdict cache stale for %s (%.0f min old) — re-evaluating",
                            t, age_min,
                        )
                        uncached_tickers.append(t)
                        continue
                except Exception:
                    pass  # if cached_at missing/invalid, still serve the cache
                cached_results.append(cached)
            else:
                uncached_tickers.append(t)

        # --- Evaluate only the cache-miss tickers ---
        live_results = []
        if uncached_tickers:
            scorer = IntegratedScorer(weights=req.weights)
            date_range = tuple(req.date_range) if req.date_range and req.date_range[0] else None

            try:
                verdicts = await asyncio.wait_for(
                    asyncio.to_thread(
                        scorer.evaluate,
                        tickers=uncached_tickers,
                        market=req.market,
                        date_range=date_range,
                        skip_layers=req.skip_layers,
                    ),
                    timeout=540,  # 9-minute hard cap
                )
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="Verdict timed out after 9 minutes")

            for v in verdicts:
                ls = v.layer_scores or {}
                live_results.append({
                    "ticker": v.ticker,
                    "core_score": ls.get("core", 0) or 0,
                    "strategy_score": ls.get("strategy", 0) or 0,
                    "ml_score": ls.get("ml_features", 0) or 0,
                    "rl_score": ls.get("rl_bot", 0) or 0,
                    "robustness_score": ls.get("robustness", 0) or 0,
                    "weighted_score": v.final_score,
                    "verdict": v.classification,
                    "layer_details": v.layer_details,
                    "strategy_breakdown": v.layer_details.get("strategy", {}),
                })

        return cached_results + live_results
    except Exception as e:
        logger.error("Verdict error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
