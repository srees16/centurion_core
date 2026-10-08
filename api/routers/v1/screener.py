"""/api/v1/screener/* routes: screener runs and the trade / paper-book monitor.

Moved from v1_gateway.py (tracker H4), which includes this router under /api/v1.
"""

import asyncio
from typing import Any, Dict, List, Optional
from pathlib import Path

from fastapi import Depends, APIRouter, HTTPException
from pydantic import BaseModel

from api.dependencies import get_kite_session
from api.routers.v1.common import KITE_SESSION_INACTIVE, _sanitize_floats, logger

router = APIRouter()


class ScreenerRunRequest(BaseModel):
    screener: Dict[str, Any]
    risk: Dict[str, Any]
    tickers: List[str] = []


# ─── Screener ────────────────────────────────────────────────────────────

@router.post("/screener/run")
async def screener_run(req: ScreenerRunRequest):
    """Run the stock screener pipeline."""
    try:
        from kite_connect.nse.screener import NSEScreener, ScreenerConfig
        from kite_connect.trading.risk_manager import RiskManager, RiskConfig
        from kite_connect.core.config import INDEX_CONSTITUENTS

        # Map frontend field names → ScreenerConfig field names
        scfg = req.screener
        screener_cfg = ScreenerConfig(
            min_price=scfg.get("min_price", 100),
            min_avg_volume=int(scfg.get("min_avg_volume", 500_000)),
            min_beta=scfg.get("min_beta", 1.0),
            max_workers=int(scfg.get("workers", 8)),
            breakout_vol_mult=scfg.get("volume_multiplier", 1.5),
            history_days=int(scfg.get("lookback_days", 250)),
            index_mode=scfg.get("index_mode", False),
        )

        # Map frontend field names → RiskConfig field names
        rcfg = req.risk
        risk_cfg = RiskConfig(
            total_capital=rcfg.get("total_capital", 500_000),
            max_open_trades=int(rcfg.get("max_open_trades", 6)),
            risk_per_trade_pct=rcfg.get("risk_per_trade_pct", 2) / 100,  # UI sends 2 → config wants 0.02
            min_rr_ratio=rcfg.get("min_rr_ratio", 2.5),
            sl_method=rcfg.get("stop_loss_method", "tighter"),
        )

        screener = NSEScreener(config=screener_cfg)
        risk_mgr = RiskManager(config=risk_cfg)

        # Carver: inject VolatilityTarget when enabled
        try:
            from config import Config as _Cfg
            if getattr(_Cfg, "CARVER_ENABLED", False):
                from services.risk.volatility_target import VolatilityTarget, VolatilityTargetConfig
                vt = VolatilityTarget(VolatilityTargetConfig(
                    initial_capital=risk_cfg.total_capital,
                    annual_vol_target_pct=getattr(_Cfg, "CARVER_ANNUAL_VOL_TARGET", 0.20),
                ))
                risk_mgr = RiskManager(config=risk_cfg, volatility_target=vt)
        except Exception:
            pass

        # Default to NIFTY50 when no tickers provided
        tickers = req.tickers if req.tickers else list(INDEX_CONSTITUENTS.get("NIFTY50", []))

        df = await asyncio.to_thread(screener.screen, tickers)
        stocks_raw = df.to_dict("records") if not df.empty else []

        # Map backend field names → frontend expectations
        stocks = []
        for s in stocks_raw:
            stocks.append({
                **s,
                "ticker": s.get("symbol", ""),
                "price": s.get("close", 0),
                "passed": True,  # all returned stocks passed Stage 1
            })

        plans = risk_mgr.plan_trades(df) if stocks else []
        plan_dicts = []
        for p in plans:
            d = p.to_dict()
            plan_dicts.append({
                **d,
                "ticker": d.get("symbol", ""),
                "risk": d.get("risk_amount", 0),
                "reward": d.get("reward_amount", 0),
            })

        return {"stocks": stocks, "trade_plans": plan_dicts, "summary": {"screened": len(stocks), "passed": len(stocks)}}
    except Exception as e:
        logger.error("Screener error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/screener/execute")
async def screener_execute(req: Dict[str, Any]):
    """Execute trade plans via Kite.

    SAFETY: Requires IntegratedScorer verdicts before order placement.
    Only BUY/STRONG_BUY symbols are forwarded to the order manager.
    """
    kite = get_kite_session()
    if not kite:
        raise HTTPException(status_code=409, detail=KITE_SESSION_INACTIVE)
    try:
        plans = req.get("plans", [])
        if not plans:
            return {"orders": [], "message": "No plans provided"}

        # ── Verdict enforcement: score all symbols first ──
        from services.signals.integrated_scorer import IntegratedScorer
        from datetime import date, timedelta

        symbols = list({p.get("symbol", "") for p in plans if p.get("symbol")})
        ns_tickers = [f"{s}.NS" for s in symbols]

        scorer = IntegratedScorer()
        end_dt = date.today()
        start_dt = end_dt - timedelta(days=365)
        verdicts = scorer.evaluate(
            tickers=ns_tickers, market="IND",
            date_range=(str(start_dt), str(end_dt)),
            skip_layers=["rag"],
        )

        buy_tags = {"BUY", "STRONG_BUY"}
        approved = {
            v.ticker.replace(".NS", "").replace(".BO", "")
            for v in verdicts if v.classification in buy_tags
        }

        # Filter plans to only approved symbols
        filtered_plans = [p for p in plans if p.get("symbol") in approved]
        blocked = len(plans) - len(filtered_plans)
        if blocked > 0:
            logger.info("Verdict filter blocked %d/%d plans (non-BUY)", blocked, len(plans))

        if not filtered_plans:
            return {"orders": [], "message": f"No plans passed verdict filter ({blocked} blocked)"}

        from kite_connect.trading.order_service import place_order

        order_results = []
        for plan in filtered_plans:
            try:
                res = await asyncio.to_thread(
                    place_order,
                    kite,
                    symbol=plan.get("symbol", ""),
                    exchange=plan.get("exchange", "NSE"),
                    transaction_type=plan.get("transaction_type", "BUY"),
                    quantity=int(plan.get("quantity", 0)),
                    order_type=plan.get("order_type", "MARKET"),
                    product=plan.get("product", "CNC"),
                    price=plan.get("price"),
                    trigger_price=plan.get("trigger_price"),
                )
                order_results.append(res)
            except Exception as e:
                order_results.append({"success": False, "error": str(e), "symbol": plan.get("symbol")})
        return {
            "orders": order_results,
            "verdict_blocked": blocked,
            "total_placed": len(filtered_plans),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/screener/monitor")
async def screener_monitor():
    """Get trade monitor summary."""
    try:
        from kite_connect.trading.trade_monitor import TradeMonitor
        monitor = TradeMonitor()
        return monitor.summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _kite_ltp_or_none():
    """Live last prices through the API's Kite session (set by the daily login), or None."""
    try:
        from api.dependencies import get_kite_session
        kite = get_kite_session()
    except Exception:                                     # noqa: BLE001 - no session: the close is used
        return None
    if kite is None:
        return None

    def ltp(symbols):
        data = kite.ltp([f"NSE:{s}" for s in symbols]) or {}
        return {k.split(":", 1)[1]: (v or {}).get("last_price") for k, v in data.items()}
    return ltp


def _paper_books() -> list:
    """The paper books, from config/nse_engine_<book>.json (tracker D5): deployed first."""
    from nse_engine.deployment import load_deployment
    root = Path(__file__).resolve().parent.parent.parent.parent
    books = []
    for path in sorted(root.glob("config/nse_engine_*.json")):
        name = path.stem[len("nse_engine_"):]
        try:
            dep = load_deployment(path)
        except Exception as exc:                          # noqa: BLE001 - a broken file is not a book
            logger.warning("paper book %s skipped: %s", path.name, exc)
            continue
        books.append({"book": name, "schema": None if name == "deployed" else name,
                      "label": f"{name} {dep.engine.config_hash()[:8]}", "status": dep.status,
                      "paper_start_date": dep.paper_start_date.isoformat()})
    books.sort(key=lambda b: (b["book"] != "deployed", b["paper_start_date"]))
    return books


_book_clouds: Dict[str, Any] = {}


def _book_param(book: Optional[str] = None) -> Optional[str]:
    """``?book=`` on the monitor endpoints (G12): a known paper book, or None for the deployed one.

    A dependency, so an unknown book is 404 (and a book without Neon 503)
    before the handler's own error handling turns it into a 500.
    """
    if not book or book == "deployed":
        return None
    if not any(b["book"] == book for b in _paper_books()):
        raise HTTPException(status_code=404, detail=f"unknown paper book {book!r}")
    if _cloud_or_none() is None:
        raise HTTPException(status_code=503, detail=f"paper book {book!r} is only in Neon, which is not configured")
    return book


@router.get("/screener/monitor/trades")
async def screener_monitor_trades(book: Optional[str] = Depends(_book_param)):
    """Active and closed paper trades — from the cloud book the Actions job writes.

    Orders decided at the close and filling at the next open appear as Pending.
    Local SQLite is only a fallback for a machine without a Neon connection.
    """
    try:
        cloud = _cloud_or_none(book)
        if cloud:
            from kite_connect.trading.paper_book_view import trades_view
            return trades_view(cloud, live_prices=_kite_ltp_or_none())

        import sqlite3 as _sql
        from pathlib import Path as _Path

        # Paper trades DB
        db_path = _Path(__file__).resolve().parent.parent.parent.parent / "data" / "paper_trades.sqlite3"
        if not db_path.exists():
            return {"active_trades": [], "closed_trades": [], "total_active": 0, "total_closed": 0}

        conn = _sql.connect(str(db_path))
        conn.row_factory = _sql.Row

        active_rows = conn.execute(
            "SELECT * FROM paper_positions WHERE is_open=1 ORDER BY opened_at DESC"
        ).fetchall()
        closed_rows = conn.execute(
            "SELECT * FROM paper_positions WHERE is_open=0 ORDER BY closed_at DESC LIMIT 200"
        ).fetchall()
        conn.close()

        def _row_to_trade(r, is_active: bool) -> dict:
            entry = float(r["entry_price"])
            sl = float(r["stop_loss"])
            tp = float(r["target_price"])
            pnl_pct = float(r["pnl_pct"]) if r["pnl_pct"] else 0.0
            return {
                "symbol": r["symbol"],
                "side": r["side"],
                "quantity": r["quantity"],
                "entry_price": entry,
                "stop_loss": sl,
                "target_price": tp,
                "entry_order_id": f"PAPER-{r['id']}",
                "sl_order_id": None,
                "tp_order_id": None,
                "entry_filled": True,
                "sl_triggered": r["exit_reason"] in ("SL", "TRAILING_SL") if not is_active else False,
                "tp_triggered": r["exit_reason"] == "TP" if not is_active else False,
                "closed": not is_active,
                "scaled_2r": False,
                "scaled_3r": False,
                "sl_failed": False,
                "opened_at": r["opened_at"],
                "direction": "LONG" if r["side"] == "BUY" else "SHORT",
                "product": "PAPER",
                "is_active": is_active,
                "unrealised_pnl_pct": pnl_pct if not is_active else 0.0,
            }

        active = [_row_to_trade(r, True) for r in active_rows]
        closed = [_row_to_trade(r, False) for r in closed_rows]

        return {
            "active_trades": active,
            "closed_trades": closed,
            "total_active": len(active),
            "total_closed": len(closed),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/screener/monitor/paper-dashboard")
async def screener_paper_dashboard(book: Optional[str] = Depends(_book_param)):
    """Paper dashboard from the cloud book: equity from the latest daily snapshot.

    Not built through ``PaperTrader`` here on purpose: its local SQLite copy is
    filled once from Neon and then read in preference to it, so the page would
    stop updating after the first day.
    """
    try:
        cloud = _cloud_or_none(book)
        if cloud:
            from kite_connect.trading.paper_book_view import dashboard_view
            return dashboard_view(cloud)
        from kite_connect.trading.paper_trader import PaperTrader
        pt = PaperTrader()
        dash = pt.dashboard()
        return dash.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _cloud_or_none(book: Optional[str] = None):
    """PaperCloudSync for a paper book: the deployed book (None without Neon), or ``book``'s own schema."""
    try:
        from database.paper_cloud import get_paper_cloud
        cloud = get_paper_cloud()
    except Exception:
        cloud = None
    if book is None or cloud is None:
        return cloud
    if book not in _book_clouds:
        from database.connection import get_db_manager
        from database.paper_cloud import PaperCloudSync
        _book_clouds[book] = PaperCloudSync(get_db_manager(), schema=book)
    return _book_clouds[book]


@router.get("/screener/monitor/books")
async def screener_paper_books():
    """The paper books the monitor can show (G12)."""
    books = _paper_books()
    return {"books": books, "count": len(books)}


def _sqlite_rows(table: str, sql: str):
    """Fallback: read rows from local SQLite."""
    import sqlite3 as _sql
    from pathlib import Path as _Path
    db_path = _Path(__file__).resolve().parent.parent.parent.parent / "data" / "paper_trades.sqlite3"
    if not db_path.exists():
        return []
    conn = _sql.connect(str(db_path))
    conn.row_factory = _sql.Row
    rows = conn.execute(sql).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.get("/screener/monitor/daily-snapshots")
async def screener_daily_snapshots(book: Optional[str] = Depends(_book_param)):
    """Get daily equity snapshots for chart rendering."""
    try:
        cloud = _cloud_or_none(book)
        if cloud:
            from kite_connect.trading.paper_book_view import snapshots_view
            snapshots = snapshots_view(cloud)          # day / cumulative P&L from equity
            if snapshots:
                return {"snapshots": snapshots, "count": len(snapshots)}

        snapshots = _sqlite_rows("daily_snapshots", "SELECT * FROM daily_snapshots ORDER BY date")
        return {"snapshots": snapshots, "count": len(snapshots)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/screener/monitor/sessions")
async def screener_sessions(book: Optional[str] = Depends(_book_param)):
    """What each session decided, for the current book.

    The Trade Center uses this to mark the days the portfolio actually
    changed: a rebalance decides orders at the close, and they fill at the
    next session's open, so the two are different days.
    """
    try:
        cloud = _cloud_or_none(book)
        if cloud:
            df = cloud.read_sessions()
            if not df.empty:
                rows = _sanitize_floats(df.to_dict(orient="records"))
                return {"sessions": rows, "count": len(rows)}
        return {"sessions": [], "count": 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/screener/monitor/signal-log")
async def screener_signal_log(book: Optional[str] = Depends(_book_param)):
    """Get signal audit log for backtest-vs-live comparison."""
    try:
        cloud = _cloud_or_none(book)
        if cloud:
            df = cloud.read_signals()
            if not df.empty:
                signals = df.head(500).to_dict(orient="records")
                total = len(df)
                traded = int(df["was_traded"].sum()) if "was_traded" in df.columns else 0
                daily = df.groupby("date").agg(
                    total_signals=("symbol", "count"),
                    traded_signals=("was_traded", "sum"),
                ).reset_index().sort_values("date", ascending=False).head(30).to_dict(orient="records")
                return {
                    "signals": signals,
                    "count": len(signals),
                    "summary": {
                        "total_signals": total,
                        "traded_signals": traded,
                        "hit_rate": round(traded / total, 4) if total > 0 else 0,
                    },
                    "daily_stats": daily,
                }

        import sqlite3 as _sql
        from pathlib import Path as _Path
        db_path = _Path(__file__).resolve().parent.parent.parent.parent / "data" / "paper_trades.sqlite3"
        if not db_path.exists():
            return {"signals": [], "count": 0, "summary": {}}

        conn = _sql.connect(str(db_path))
        conn.row_factory = _sql.Row
        rows = conn.execute(
            "SELECT * FROM signal_log ORDER BY date DESC, symbol LIMIT 500"
        ).fetchall()

        total = conn.execute("SELECT COUNT(*) as cnt FROM signal_log").fetchone()["cnt"]
        traded = conn.execute("SELECT COUNT(*) as cnt FROM signal_log WHERE was_traded=1").fetchone()["cnt"]

        date_stats = conn.execute("""
            SELECT date,
                   COUNT(*) as total_signals,
                   SUM(was_traded) as traded_signals
            FROM signal_log GROUP BY date ORDER BY date DESC LIMIT 30
        """).fetchall()
        conn.close()

        return {
            "signals": [dict(r) for r in rows],
            "count": len(rows),
            "summary": {
                "total_signals": total,
                "traded_signals": traded,
                "hit_rate": round(traded / total, 4) if total > 0 else 0,
            },
            "daily_stats": [dict(r) for r in date_stats],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/screener/monitor/weekly-checkpoints")
async def screener_weekly_checkpoints(book: Optional[str] = Depends(_book_param)):
    """Get weekly performance checkpoints."""
    try:
        cloud = _cloud_or_none(book)
        if cloud:
            df = cloud.read_weekly()
            if not df.empty:
                checkpoints = df.to_dict(orient="records")
                return {"checkpoints": checkpoints, "count": len(checkpoints)}

        checkpoints = _sqlite_rows("weekly_checkpoints", "SELECT * FROM weekly_checkpoints ORDER BY week_number")
        return {"checkpoints": checkpoints, "count": len(checkpoints)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/screener/monitor/daily-detail/{date}")
async def screener_daily_detail(date: str, book: Optional[str] = Depends(_book_param)):
    """Get full drill-down for a single trading day."""
    import json as _json
    import re

    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):   # it reaches the SQLite fallback's SQL text
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD")
    try:
        # 1. Snapshot for this date
        cloud = _cloud_or_none(book)
        snapshot = None
        snapshot_detail = {}
        if cloud:
            from kite_connect.trading.paper_book_view import snapshots_view
            snapshot = next((s for s in snapshots_view(cloud) if str(s.get("date")) == date), None)
        if snapshot is None:
            rows = _sqlite_rows("daily_snapshots", f"SELECT * FROM daily_snapshots WHERE date='{date}'")
            snapshot = rows[0] if rows else None
        if snapshot:
            sj = snapshot.get("snapshot_json", "{}")
            try:
                snapshot_detail = _json.loads(sj) if isinstance(sj, str) else (sj or {})
            except Exception:
                snapshot_detail = {}

        # 2. Signals for this date
        if cloud:
            df = cloud.read_signals()
            if not df.empty:
                day_sig = df[df["date"] == date]
                signals = day_sig.to_dict(orient="records")
            else:
                signals = []
        else:
            signals = _sqlite_rows("signal_log", f"SELECT * FROM signal_log WHERE date='{date}' ORDER BY combined_forecast DESC")
        total_signals = len(signals)
        traded_signals = sum(1 for s in signals if s.get("was_traded"))

        # 3. Positions opened on this date
        if cloud:
            df = cloud.read_positions()
            if not df.empty:
                opened = df[df["opened_at"].astype(str).str.startswith(date)].to_dict(orient="records")
            else:
                opened = []
        else:
            opened = _sqlite_rows("paper_positions", f"SELECT * FROM paper_positions WHERE opened_at LIKE '{date}%'")

        # 4. Positions closed on this date (SL/TP events)
        if cloud:
            df = cloud.read_positions()
            if not df.empty:
                closed = df[(df["closed_at"].astype(str).str.startswith(date)) & (df["is_open"] == 0)].to_dict(orient="records")
            else:
                closed = []
        else:
            closed = _sqlite_rows("paper_positions", f"SELECT * FROM paper_positions WHERE closed_at LIKE '{date}%' AND is_open=0")

        # 5. What the engine decided that session, and every execution it made.
        #    A day with no trades must be visibly a decision, not a blank page.
        session_activity = None
        executions = []
        if cloud:
            try:
                df = cloud.read_sessions()
                if not df.empty:
                    row = df[df["session_date"].astype(str) == date]
                    if not row.empty:
                        session_activity = _sanitize_floats(row.iloc[0].to_dict())
            except Exception as exc:                      # noqa: BLE001 - older books have no table
                logger.debug("session activity unavailable: %s", exc)
            try:
                df = cloud.read_fills()
                if not df.empty:
                    same_day = df[df["session_date"].astype(str) == date]
                    executions = _sanitize_floats(same_day.to_dict(orient="records"))
            except Exception as exc:                      # noqa: BLE001
                logger.debug("fills unavailable: %s", exc)

        return {
            "date": date,
            "snapshot": snapshot,
            "snapshot_detail": snapshot_detail,
            "session": session_activity,
            "executions": executions,
            "executions_count": len(executions),
            "signals": signals,
            "total_signals": total_signals,
            "traded_signals": traded_signals,
            "skipped_signals": total_signals - traded_signals,
            "trades_opened": opened,
            "trades_opened_count": len(opened),
            "trades_closed": closed,
            "trades_closed_count": len(closed),
            "exit_reasons": {},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
