"""/api/v1/rl-bot/* routes: RL bot training, evaluation, signals, uploads.

Moved from v1_gateway.py (tracker H4), which includes this router under /api/v1.
"""

import asyncio
from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from api.routers.v1.common import _sanitize_floats, logger

router = APIRouter()


# ─── RL Bot ─────────────────────────────────────────────────────────────


class RLTrainRequest(BaseModel):
    tickers: List[str]
    algorithm: str = "PPO"
    reward_type: str = "hybrid"
    total_timesteps: int = Field(500000, ge=1_000, le=5_000_000)   # bounded: one request, not hours of CPU
    lookback: int = 60
    train_days: int = 504
    test_days: int = 63
    folds: int = 6
    initial_capital: float = 100000


class RLEvalRequest(BaseModel):
    ticker: str
    algorithm: str = "PPO"
    eval_days: int = 252


class RLSignalRequest(BaseModel):
    ticker: str
    algorithm: str = "PPO"


@router.post("/rl-bot/train")
async def rl_bot_train(req: RLTrainRequest):
    """Train RL agents for given tickers."""
    try:
        from services.rl_bot.train_rl_agent import train_multi_ticker, TrainConfig

        cfg = TrainConfig(
            algorithm=req.algorithm,
            total_timesteps=req.total_timesteps,
            reward_type=req.reward_type,
            lookback=req.lookback,
            train_days=req.train_days,
            test_days=req.test_days,
            total_folds=req.folds,
            initial_capital=req.initial_capital,
        )

        results = await asyncio.to_thread(train_multi_ticker, req.tickers, cfg)

        out = {}
        for ticker, r in results.items():
            out[ticker] = {
                "algorithm": r.algorithm,
                "model_path": r.model_path,
                "avg_test_return": round(r.avg_test_return, 2),
                "avg_test_sharpe": round(r.avg_test_sharpe, 2),
                "avg_test_drawdown": round(r.avg_test_drawdown, 2),
                "folds": [
                    {
                        "fold": f.fold,
                        "train_period": f"{f.train_start} → {f.train_end}",
                        "test_period": f"{f.test_start} → {f.test_end}",
                        "return_pct": round(f.test_return_pct, 2),
                        "sharpe": round(f.test_sharpe, 2),
                        "max_dd_pct": round(f.test_max_drawdown_pct, 2),
                        "trades": f.test_n_trades,
                        "win_rate": round(f.test_win_rate * 100, 1),
                    }
                    for f in r.folds
                ],
            }

        return {"results": out}
    except Exception as e:
        logger.error("RL train error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rl-bot/evaluate")
async def rl_bot_evaluate(req: RLEvalRequest):
    """Evaluate a trained RL agent on recent data."""
    try:
        from pathlib import Path

        safe_ticker = req.ticker.replace(".", "_").replace(":", "_")
        model_dir = Path("data") / "rl_models"
        model_path = str(model_dir / f"{safe_ticker}_{req.algorithm.lower()}")

        # Check model exists
        if not (model_dir / f"{safe_ticker}_{req.algorithm.lower()}.zip").exists():
            raise HTTPException(
                status_code=404,
                detail=f"No trained model found for {req.ticker} ({req.algorithm}). Train first.",
            )

        from services.rl_bot.evaluate_agent import evaluate_agent
        from dataclasses import asdict

        metrics, signals, trades = await asyncio.to_thread(
            evaluate_agent,
            req.ticker,
            model_path,
            req.algorithm,
            eval_days=req.eval_days,
        )

        return _sanitize_floats({
            "metrics": asdict(metrics),
            "signals": [
                {"date": s.date, "action": s.action, "confidence": round(s.confidence, 3)}
                for s in signals[-30:]
            ],
            "trades": trades[-50:],
        })
    except Exception as e:
        logger.error("RL evaluate error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rl-bot/signal")
async def rl_bot_signal(req: RLSignalRequest):
    """Get latest RL signal for a ticker."""
    try:
        from pathlib import Path
        from services.rl_bot.evaluate_agent import get_latest_signal

        safe_ticker = req.ticker.replace(".", "_").replace(":", "_")
        model_dir = Path("data") / "rl_models"
        model_path = str(model_dir / f"{safe_ticker}_{req.algorithm.lower()}")

        if not (model_dir / f"{safe_ticker}_{req.algorithm.lower()}.zip").exists():
            return {"action": "HOLD", "confidence": 0.0, "status": "no_model"}

        signal = await asyncio.to_thread(
            get_latest_signal, req.ticker, model_path, req.algorithm,
        )

        return {
            "ticker": signal.ticker,
            "date": signal.date,
            "action": signal.action,
            "confidence": round(signal.confidence, 3),
            "portfolio_value": signal.portfolio_value,
            "position": signal.position,
        }
    except Exception as e:
        logger.error("RL signal error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rl-bot/models")
async def rl_bot_models():
    """List all trained RL models."""
    from pathlib import Path

    model_dir = Path("data") / "rl_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    models = []
    for f in model_dir.glob("*.zip"):
        parts = f.stem.rsplit("_", 1)
        ticker = parts[0].replace("_", ".") if len(parts) == 2 else f.stem
        algo = parts[1].upper() if len(parts) == 2 else "UNKNOWN"
        models.append({
            "ticker": ticker,
            "algorithm": algo,
            "filename": f.name,
            "size_kb": round(f.stat().st_size / 1024, 1),
        })

    return {"models": models}


@router.post("/rl-bot/upload-data")
async def rl_bot_upload_data(file: UploadFile = File(...)):
    """Upload a CSV file with OHLCV data for RL training.

    Expected columns: Date, Open, High, Low, Close, Volume
    Optional: Ticker column for multi-stock files.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    import pandas as pd
    from io import StringIO
    from pathlib import Path

    upload_dir = Path("data") / "rl_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    try:
        content = await file.read()
        text = content.decode("utf-8")
        df = pd.read_csv(StringIO(text))

        # Normalize column names
        col_map = {}
        for col in df.columns:
            low = col.strip().lower()
            if low in ("date", "datetime", "timestamp"):
                col_map[col] = "Date"
            elif low in ("open", "o"):
                col_map[col] = "Open"
            elif low in ("high", "h"):
                col_map[col] = "High"
            elif low in ("low", "l"):
                col_map[col] = "Low"
            elif low in ("close", "c", "adj close", "adj_close"):
                col_map[col] = "Close"
            elif low in ("volume", "vol", "v"):
                col_map[col] = "Volume"
            elif low in ("ticker", "symbol"):
                col_map[col] = "Ticker"
        df.rename(columns=col_map, inplace=True)

        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required - set(df.columns)
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {', '.join(missing)}. "
                       f"Found: {', '.join(df.columns.tolist())}",
            )

        # Detect tickers in file
        tickers = []
        if "Ticker" in df.columns:
            tickers = df["Ticker"].dropna().unique().tolist()

        # Save the file
        safe_name = file.filename.replace(" ", "_").replace("/", "_").replace("\\", "_")
        save_path = upload_dir / safe_name
        df.to_csv(save_path, index=False)

        return {
            "filename": safe_name,
            "rows": len(df),
            "columns": df.columns.tolist(),
            "tickers": tickers[:50],
            "date_range": {
                "start": str(df["Date"].iloc[0]) if "Date" in df.columns else None,
                "end": str(df["Date"].iloc[-1]) if "Date" in df.columns else None,
            },
        }
    except Exception as e:
        logger.error("RL upload error: %s", e)
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")


@router.get("/rl-bot/uploads")
async def rl_bot_list_uploads():
    """List uploaded CSV data files."""
    from pathlib import Path

    upload_dir = Path("data") / "rl_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for f in sorted(upload_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True):
        files.append({
            "filename": f.name,
            "size_kb": round(f.stat().st_size / 1024, 1),
        })

    return {"files": files}


@router.get("/rl-bot/portfolio-analysis")
async def rl_bot_portfolio_analysis():
    """
    Analyze the Green Energy Theme portfolio and generate Nifty buy predictions.
    Studies constituent stock selection patterns, factor characteristics, and
    applies the learned model to the broader Nifty universe.
    """
    try:
        from services.rl_bot.theme_analyzer import run_portfolio_analysis

        result = run_portfolio_analysis()
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Portfolio analysis error: %s", e)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")
