"""/api/v1/backtest/* routes: strategy list, runs, stored results.

Moved from v1_gateway.py (tracker H4), which includes this router under /api/v1.
"""

import asyncio
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.dependencies import get_db_service
from api.routers.v1.common import logger

router = APIRouter()


class BacktestRunRequest(BaseModel):
    strategy_id: str
    tickers: List[str]
    params: Dict[str, Any] = {}
    initial_capital: float = 100000
    period: str = "1y"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    market: str = "US"


# ─── Backtest ────────────────────────────────────────────────────────────

@router.get("/backtest/strategies")
async def backtest_strategies(market: str = "US"):
    """List available trading strategies (lightweight, no heavy imports)."""
    try:
        from trading_strategies import list_strategies, get_strategy

        result = []
        for info in list_strategies():
            try:
                strategy_cls = get_strategy(info["id"])
                params_raw = strategy_cls.get_parameters() if strategy_cls else {}
                params = [
                    {"name": k, **v}
                    for k, v in params_raw.items()
                ]
            except Exception:
                params = []
            result.append({
                "id": info["id"],
                "name": info.get("name", info["id"]),
                "category": info.get("category", "general"),
                "description": info.get("description", ""),
                "parameters": params,
            })

        return result
    except Exception as e:
        logger.error("Strategy listing error: %s", e, exc_info=True)
        return []


@router.post("/backtest/run")
async def backtest_run(req: BacktestRunRequest):
    """Run a strategy backtest using the strategy registry directly."""
    import uuid as _uuid
    from datetime import datetime, timedelta
    from trading_strategies import get_strategy

    try:
        # ── Normalise Indian tickers to yfinance format (.NS) ──
        # Raw NSE symbols (e.g. "SBIN", "MARUTI") fail on Yahoo Finance
        # without the .NS suffix.  US tickers are left untouched.
        if req.market == "IND":
            from utils import yf_nse_symbol
            req.tickers = [
                yf_nse_symbol(t) if not t.upper().endswith((".NS", ".BO")) else t
                for t in req.tickers
            ]

        strategy_cls = get_strategy(req.strategy_id)
        if strategy_cls is None:
            raise HTTPException(status_code=404, detail=f"Strategy '{req.strategy_id}' not found")

        # Resolve date range: explicit dates take priority, else derive from period
        end_date = req.end_date
        start_date = req.start_date
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            period_map = {"1m": 30, "3m": 90, "6m": 180, "1y": 365, "2y": 730, "5y": 1825}
            days = period_map.get(req.period, 365)
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        # Build kwargs: strategy-specific params + required run() args
        run_kwargs = {
            "tickers": req.tickers,
            "start_date": start_date,
            "end_date": end_date,
            "capital": req.initial_capital,
            **req.params,
        }

        strategy = strategy_cls()
        result = await asyncio.to_thread(strategy.run, **run_kwargs)

        if not result.success:
            msg = result.error_message or "Strategy execution failed"
            if "No valid data" in msg:
                # not a server fault: the symbols/dates yielded no bars from NSE bhavcopy or Yahoo
                raise HTTPException(
                    status_code=422,
                    detail=f"{msg} — tickers={req.tickers} window={req.start_date}..{req.end_date}. "
                           "Indian symbols are read from NSE bhavcopy first, then Yahoo; check the symbols "
                           "(NSE trading symbol, e.g. RELIANCE) and that the window covers trading days.")
            raise HTTPException(status_code=500, detail=msg)

        # Extract flat metrics (result.metrics may be nested by ticker)
        metrics = result.metrics or {}

        # Detect per-ticker nesting: {"AAPL": {sharpe: ...}, "MSFT": {...}}
        ticker_metrics = {}
        flat_metrics = {}
        for t in req.tickers:
            if t in metrics and isinstance(metrics[t], dict):
                ticker_metrics[t] = metrics[t]

        if ticker_metrics:
            # Aggregate across all tickers for top-level summary
            agg_keys = ["total_return", "sharpe_ratio", "sortino_ratio",
                        "max_drawdown", "total_trades", "win_rate", "final_value"]
            n = len(ticker_metrics)
            agg: dict = {}
            for key in agg_keys:
                vals = [float(tm.get(key, 0)) for tm in ticker_metrics.values()]
                if key == "total_trades":
                    agg[key] = int(sum(vals))
                elif key == "max_drawdown":
                    agg[key] = min(vals)          # worst drawdown
                elif key == "final_value":
                    agg[key] = sum(vals)           # total portfolio value
                else:
                    agg[key] = sum(vals) / n       # average
            m = agg
        else:
            m = metrics
            ticker_metrics = {}

        # Build equity_curve from portfolio DataFrame
        equity_curve = []
        if result.portfolio is not None and not result.portfolio.empty:
            df = result.portfolio
            date_col = next((c for c in df.columns if c.lower() in ("date", "datetime", "timestamp")), None)
            value_col = next((c for c in df.columns if c.lower() in ("value", "portfolio_value", "equity", "total")), None)
            dd_col = next((c for c in df.columns if "drawdown" in c.lower()), None)
            if date_col and value_col:
                for _, row in df.iterrows():
                    equity_curve.append({
                        "date": str(row[date_col])[:10],
                        "value": float(row[value_col]),
                        "drawdown": float(row[dd_col]) if dd_col else 0.0,
                    })

        # Build signals list from signals DataFrame
        signals = []
        if result.signals is not None and not result.signals.empty:
            df = result.signals
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                signals.append({
                    "date": str(row_dict.get("date", row_dict.get("datetime", "")))[:10],
                    "ticker": str(row_dict.get("ticker", row_dict.get("symbol", ""))),
                    "signal": str(row_dict.get("signal", row_dict.get("action", ""))),
                    "price": float(row_dict.get("price", row_dict.get("close", 0))),
                    "quantity": int(row_dict.get("quantity", row_dict.get("qty", 0))),
                })

        # Build charts list
        charts = []
        for c in (result.charts or []):
            charts.append({
                "type": c.chart_type,
                "data": c.data,
                "title": c.title,
            })

        response = {
            "id": _uuid.uuid4().hex,
            "strategy_id": req.strategy_id,
            "strategy_name": getattr(strategy_cls, "name", req.strategy_id),
            "tickers": req.tickers,
            "start_date": start_date,
            "end_date": end_date,
            "total_return": float(m.get("total_return", 0)),
            "sharpe_ratio": float(m.get("sharpe_ratio", 0)),
            "sortino_ratio": float(m.get("sortino_ratio", 0)),
            "max_drawdown": float(m.get("max_drawdown", 0)),
            "total_trades": int(m.get("total_trades", 0)),
            "win_rate": float(m.get("win_rate", 0)),
            "final_value": float(m.get("final_value", req.initial_capital)),
            "initial_capital": req.initial_capital,
            "charts": charts,
            "signals": signals,
            "equity_curve": equity_curve,
            "metrics": metrics,
            "per_ticker": {
                t: {
                    "total_return": float(tm.get("total_return", 0)),
                    "sharpe_ratio": float(tm.get("sharpe_ratio", 0)),
                    "sortino_ratio": float(tm.get("sortino_ratio", 0)),
                    "max_drawdown": float(tm.get("max_drawdown", 0)),
                    "total_trades": int(tm.get("total_trades", 0)),
                    "win_rate": float(tm.get("win_rate", 0)),
                    "final_value": float(tm.get("final_value", 0)),
                }
                for t, tm in ticker_metrics.items()
            },
            "created_at": datetime.now().isoformat(),
        }

        # Persist to database
        db = get_db_service()
        if db:
            try:
                db.save_backtest_result(result=response, market=req.market)
            except Exception as e:
                logger.warning("Failed to save backtest: %s", e)

        # Upload charts to R2 / MinIO object storage (non-blocking)
        if result.charts:
            try:
                from services.storage.minio_service import get_minio_service
                minio_svc = get_minio_service()
                if minio_svc.is_available:
                    saved = await asyncio.to_thread(
                        minio_svc.save_backtest_charts,
                        run_id=response["id"],
                        charts=result.charts,
                        strategy_name=response.get("strategy_name", req.strategy_id),
                    )
                    logger.info("Saved %d chart(s) to R2 for backtest %s",
                                len(saved) if saved else 0, response["id"])
            except Exception as e:
                logger.warning("R2 chart upload failed (non-fatal): %s", e)

        return response
    except Exception as e:
        logger.error("Backtest error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backtest/{backtest_id}")
async def backtest_get(backtest_id: str):
    """Get a specific backtest result."""
    db = get_db_service()
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable")
    result = db.get_backtest_result(backtest_id)
    if not result:
        raise HTTPException(status_code=404, detail="Backtest not found")
    return result
