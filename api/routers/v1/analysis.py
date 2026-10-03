"""/api/v1/analysis/* routes: news + sentiment + metrics analysis runs.

Moved from v1_gateway.py (tracker H4), which includes this router under /api/v1.
"""

import asyncio
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.dependencies import get_db_service
from api.routers.v1.common import _sanitize_floats, logger

router = APIRouter()


def _metrics_to_dict(m) -> dict:
    """Convert a StockMetrics dataclass to a JSON-safe dict."""
    from dataclasses import asdict
    d = asdict(m)
    if d.get("timestamp"):
        d["timestamp"] = d["timestamp"].isoformat() if hasattr(d["timestamp"], "isoformat") else str(d["timestamp"])
    return _sanitize_floats(d)


class AnalysisRunRequest(BaseModel):
    tickers: List[str]
    market: str = "US"
    period: str = "1y"


# ─── Analysis ───────────────────────────────────────────────────────────

@router.post("/analysis/run")
async def analysis_run(req: AnalysisRunRequest):
    """Run the full analysis pipeline for given tickers."""
    try:
        if req.market == "US":
            from scrapers.us_aggregator import USNewsAggregator
        else:
            from scrapers.ind_aggregator import IndianNewsAggregator as USNewsAggregator

        from services.sentiment import SentimentAnalyzer
        from services.metrics import MetricsCalculator
        from services.decision_engine import DecisionEngine

        aggregator = USNewsAggregator()
        analyzer = SentimentAnalyzer()
        calculator = MetricsCalculator()
        engine = DecisionEngine()

        news_items = await aggregator.fetch_news_for_tickers(req.tickers)
        analyzed = await asyncio.to_thread(analyzer.analyze_news_items, news_items)

        # Indian tickers need .NS suffix for yfinance / metrics lookups
        if req.market == "IND":
            from utils import yf_nse_symbol
            yf_tickers = {t: yf_nse_symbol(t) for t in req.tickers}
        else:
            yf_tickers = {t: t for t in req.tickers}

        metrics_map = {}
        for ticker in req.tickers:
            try:
                m = await asyncio.to_thread(calculator.get_stock_metrics, yf_tickers[ticker])
                metrics_map[ticker] = m
            except Exception:
                metrics_map[ticker] = None

        signals = []
        for item in analyzed:
            m = metrics_map.get(item.ticker)
            sig = engine.generate_signal(item, m)
            ni = sig.news_item
            signals.append({
                "news_item": {
                    "title": ni.title,
                    "summary": ni.summary,
                    "url": ni.url,
                    "timestamp": ni.timestamp.isoformat() if hasattr(ni.timestamp, "isoformat") else str(ni.timestamp),
                    "source": ni.source,
                    "ticker": ni.ticker,
                    "category": ni.category.value if ni.category else "general",
                    "sentiment_score": ni.sentiment_score,
                    "sentiment_label": ni.sentiment_label.value if ni.sentiment_label else None,
                    "sentiment_confidence": ni.sentiment_confidence,
                },
                "metrics": _metrics_to_dict(sig.metrics) if sig.metrics else None,
                "decision": sig.decision.value,
                "decision_score": sig.decision_score,
                "reasoning": sig.reasoning,
                "timestamp": sig.timestamp.isoformat() if hasattr(sig.timestamp, "isoformat") else str(sig.timestamp),
            })

        summary = {"total": len(signals), "strong_buy": 0, "buy": 0, "hold": 0, "sell": 0, "strong_sell": 0}
        for s in signals:
            key = s.get("decision", "hold").lower()
            if key in summary:
                summary[key] += 1

        # Persist to DB if available
        db = get_db_service()
        run_id = None
        if db:
            try:
                run_id = db.start_analysis_run(
                    run_type="stock_analysis",
                    tickers=req.tickers,
                    market=req.market,
                )
                if run_id:
                    db.save_signals(signals, analysis_run_id=run_id, market=req.market)
                    run_id = str(run_id)
            except Exception as e:
                logger.warning("Failed to save analysis run: %s", e)

        return _sanitize_floats({"run_id": run_id, "signals": signals, "summary": summary})
    except Exception as e:
        logger.error("Analysis error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis/latest")
async def analysis_latest(market: str = "US"):
    """Get the most recent analysis run."""
    db = get_db_service()
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        result = db.get_latest_analysis(market)
        return result or {"run_id": None, "signals": [], "summary": {}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis/metrics")
async def analysis_metrics(tickers: str, market: str = "US"):
    """Get stock metrics for given tickers."""
    from services.metrics import MetricsCalculator
    calc = MetricsCalculator()
    results = []
    for ticker in tickers.split(","):
        ticker = ticker.strip()
        if not ticker:
            continue
        try:
            m = await asyncio.to_thread(calc.get_stock_metrics, ticker)
            if m:
                results.append(_metrics_to_dict(m))
        except Exception:
            pass
    return results
