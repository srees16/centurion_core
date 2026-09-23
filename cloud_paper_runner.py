#!/usr/bin/env python3
"""Cloud paper trading runner — executed by GitHub Actions cron.

Checks Neon for active paper trading state, runs the full CarverPipeline,
executes trades via PaperTrader, and syncs results to Neon.

Usage (GitHub Actions):
    python centurion_core/cloud_paper_runner.py

Required env vars:
    CENTURION_DATABASE_URL  — Neon PostgreSQL connection string
    CENTURION_PAPER_TRADE   — must be "true"

Optional:
    CENTURION_EMAIL_USER / CENTURION_EMAIL_PASS  — for daily reports
"""

import os
import sys
import logging
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

# Ensure centurion_core is on the path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("CENTURION_PAPER_TRADE", "true")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cloud_paper_runner")


# ── Neon state helpers ────────────────────────────────────────────────────

def _get_neon_engine():
    """Create a SQLAlchemy engine for Neon."""
    import re
    from sqlalchemy import create_engine

    url = os.environ.get("CENTURION_DATABASE_URL", "")
    if not url:
        raise RuntimeError("CENTURION_DATABASE_URL not set")

    # Strip channel_binding (psycopg2 doesn't support it)
    url = re.sub(r"[&?]channel_binding=[^&]*", "", url)
    if "sslmode" not in url:
        sep = "&" if "?" in url else "?"
        url += f"{sep}sslmode=require"

    return create_engine(url, pool_pre_ping=True, pool_size=2)


def _check_active() -> bool:
    """Check if paper trading is active and not expired in Neon."""
    from sqlalchemy import text

    engine = _get_neon_engine()
    with engine.connect() as conn:
        # Auto-create table if it doesn't exist
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS paper_trading_state (
                id            INTEGER PRIMARY KEY DEFAULT 1,
                active        BOOLEAN NOT NULL DEFAULT FALSE,
                started_at    TIMESTAMPTZ,
                expires_at    TIMESTAMPTZ,
                stopped_at    TIMESTAMPTZ,
                last_run_at   TIMESTAMPTZ,
                total_runs    INTEGER DEFAULT 0,
                last_run_status VARCHAR(20) DEFAULT 'none',
                last_run_message TEXT DEFAULT '',
                updated_at    TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            INSERT INTO paper_trading_state (id, active)
            VALUES (1, FALSE)
            ON CONFLICT (id) DO NOTHING
        """))
        conn.commit()

        row = conn.execute(
            text("SELECT active, expires_at FROM paper_trading_state WHERE id = 1")
        ).fetchone()

        if not row or not row[0]:
            return False

        # Check expiry
        if row[1] and row[1] < datetime.now(row[1].tzinfo or None):
            conn.execute(text(
                "UPDATE paper_trading_state SET active = FALSE, stopped_at = NOW(), updated_at = NOW() WHERE id = 1"
            ))
            conn.commit()
            logger.info("Paper trading expired at %s — deactivated", row[1])
            return False

        return True


def _update_run_status(status: str, message: str):
    """Update the last run status in Neon."""
    from sqlalchemy import text

    try:
        engine = _get_neon_engine()
        with engine.connect() as conn:
            conn.execute(text("""
                UPDATE paper_trading_state
                SET last_run_at = NOW(),
                    total_runs = total_runs + 1,
                    last_run_status = :status,
                    last_run_message = :message,
                    updated_at = NOW()
                WHERE id = 1
            """), {"status": status, "message": message})
            conn.commit()
    except Exception as exc:
        logger.warning("Failed to update run status in Neon: %s", exc)


# ── Paper book (persistent across GitHub Actions runs) ────────────────────

# Initial capital is used ONLY on the very first run; later runs restore the
# book (cash, positions, stops, snapshots) from Neon.
_INITIAL_CAPITAL = float(os.environ.get("CENTURION_PAPER_INITIAL_CAPITAL", "100000"))


def _open_paper_trader():
    """PaperTrader restored from local SQLite or Neon. Raises if the cloud
    book exists but cannot be read (never start over an unreadable book)."""
    from kite_connect.trading.paper_trader import PaperTrader

    pt = PaperTrader(kite=None, initial_capital=_INITIAL_CAPITAL)
    if pt.restored_from == "cloud_restore_failed":
        raise RuntimeError("Paper book restore from Neon failed — aborting run to protect the track record")
    logger.info("Paper book: source=%s cash=%.0f open=%d initial=%.0f",
                pt.restored_from, pt.cash, sum(1 for p in pt._positions if p.is_open),
                pt.initial_capital)
    return pt


def _latest_daily_bars(symbols):
    """{symbol: {date, open, low, close}} from the latest daily bar."""
    from utils import download_ind_ohlcv

    bars = {}
    for sym in symbols:
        try:
            df = download_ind_ohlcv(sym, period="5d")
            if df is None or df.empty:
                continue
            row = df.iloc[-1]
            val = lambda c: float(row[c].item() if hasattr(row[c], "item") else row[c])  # noqa: E731
            bars[sym] = {"date": df.index[-1], "open": val("Open"), "low": val("Low"),
                         "close": val("Close")}
        except Exception as exc:
            logger.debug("Daily bar fetch failed for %s: %s", sym, exc)
    return bars


def _mark_and_simulate_stops(pt):
    """Mark open positions to the latest close and simulate GTT stop fills."""
    held = sorted({p.symbol for p in pt._positions if p.is_open})
    if not held:
        return []
    bars = _latest_daily_bars(held)
    events = pt.simulate_gtt_stops(bars)
    for ev in events:
        logger.info("Paper GTT stop: %s %s exit=%.2f pnl=%.2f",
                    ev["symbol"], ev["type"], ev["exit"], ev["pnl"])
    return events


# ── Pipeline execution ────────────────────────────────────────────────────

def _run_paper_pipeline():
    """Run the full screening + CarverPipeline + PaperTrader flow."""
    # 0. Restore the paper book, mark to market, simulate GTT stops
    pt = _open_paper_trader()
    stop_events = _mark_and_simulate_stops(pt)
    ctx = {"universe_size": 0, "screened_count": 0, "buy_signals": 0}
    try:
        status, msg = _signals_and_trades(pt, ctx)
    finally:
        # EOD snapshot is recorded on every run (upsert per date), even when
        # no new trades were generated, so the equity curve has no gaps.
        try:
            pt.snapshot_daily()
        except Exception as exc:
            logger.warning("Snapshot failed: %s", exc)

    dashboard = pt.dashboard()
    msg = (
        f"{msg} | stops={len(stop_events)} | "
        f"capital={dashboard.current_capital:.0f} | "
        f"P&L={dashboard.total_pnl:.0f} ({dashboard.total_pnl_pct:.1f}%)"
    )
    logger.info("Paper trade: %s", msg)

    # Daily email (best-effort)
    try:
        from services.notifications.manager import NotificationManager
        nm = NotificationManager()
        sent = nm.email_daily_pipeline_report({
            **ctx,
            "sell_signals": len(stop_events) + ctx.get("exits", 0),
            "status": status,
        })
        if not sent:
            logger.warning("Daily email returned False — check CENTURION_EMAIL_USER / CENTURION_EMAIL_PASS env vars")
    except Exception as exc:
        logger.warning("Daily email failed: %s", exc)

    return status, msg


def _signals_and_trades(pt, ctx):
    """Screen → verdicts → CarverPipeline (with holdings) → exits → paper buys."""
    from kite_connect.nse.nse_universe import get_nse_universe
    from kite_connect.nse.screener import NSEScreener, ScreenerConfig
    from services.signals.integrated_scorer import IntegratedScorer

    holdings = {s: h["quantity"] for s, h in pt.holdings().items()}

    # 1. Universe
    symbols = get_nse_universe()
    ctx["universe_size"] = len(symbols)
    logger.info("Universe: %d symbols", len(symbols))

    # 2. Screen
    cfg = ScreenerConfig(index_mode=True)
    screener = NSEScreener(config=cfg)
    screened_df = screener.screen(symbols)
    logger.info("Screened: %d passed", len(screened_df))

    ctx["screened_count"] = len(screened_df)
    signal_dict = {}
    buy_symbols = []
    if not screened_df.empty:
        # 3. IntegratedScorer verdicts
        ns_tickers = [f"{s}.NS" for s in screened_df["symbol"].tolist()]
        end_dt = date.today()
        start_dt = end_dt - timedelta(days=365)

        scorer = IntegratedScorer()
        verdicts = scorer.evaluate(
            tickers=ns_tickers,
            market="IND",
            date_range=(str(start_dt), str(end_dt)),
        )

        signal_dict = {
            v.ticker.replace(".NS", "").replace(".BO", ""): v.classification
            for v in verdicts
        }
        buy_symbols = [
            sym for sym, tag in signal_dict.items()
            if tag in ("BUY", "STRONG_BUY")
        ]
    ctx["buy_signals"] = len(buy_symbols)
    buy_df = screened_df[screened_df["symbol"].isin(buy_symbols)] if buy_symbols else screened_df.iloc[0:0]

    if not buy_symbols and not holdings:
        return "success", "No BUY signals and no open positions"

    # 4. CarverPipeline — buy candidates AND current holdings (for exits)
    plans = None
    pipe_result = None
    fallback_reason = ""
    try:
        from services.execution.carver_pipeline import CarverPipeline, PipelineConfig
        from utils import download_ind_ohlcv

        ohlcv_cache = {}
        for sym in dict.fromkeys(list(buy_symbols) + list(holdings)):
            try:
                df = download_ind_ohlcv(sym, period="2y")
                if df is not None and len(df) >= 64:
                    ohlcv_cache[sym] = df
            except Exception:
                pass

        if ohlcv_cache:
            screener_scores = {}
            if "score" in screened_df.columns:
                for _, row in screened_df.iterrows():
                    sym = row.get("symbol", "")
                    if sym in ohlcv_cache:
                        screener_scores[sym] = float(row["score"])

            pipeline = CarverPipeline(PipelineConfig())
            pipe_result = pipeline.run(
                ohlcv_cache=ohlcv_cache,
                screener_scores=screener_scores,
                current_holdings=holdings or None,
            )
            # Only BUY candidates open positions (holdings were included for exits)
            plans = [p for p in pipe_result.trade_plans if p.symbol in set(buy_symbols)]
            logger.info("CarverPipeline: %d plans, %d exits", len(plans), len(pipe_result.exits))
            _fresh = pipe_result.freshness or {}
            if _fresh.get("dropped") and pipe_result.symbols_processed == 0:
                fallback_reason = (f"freshness gate dropped all {len(_fresh['dropped'])} symbols "
                                   f"(expected session {_fresh.get('expected_session')})")
        else:
            fallback_reason = "no OHLCV data"
    except Exception as exc:
        fallback_reason = f"CarverPipeline error: {exc}"

    # 4a. Rank / forecast exits for open paper positions
    exit_events = []
    if pipe_result is not None and pipe_result.exits:
        for sym, reason in pipe_result.exits.items():
            res = pt.close_position(sym, reason=f"RANK_EXIT:{reason}"[:30])
            if res.get("success"):
                exit_events.append(res)
                logger.info("Paper rank exit: %s x %d (%s)", sym, res["quantity"], reason)
    ctx["exits"] = len(exit_events)

    # 4b. Fallback: RiskManager — LOUD, with the reason recorded
    if fallback_reason and buy_symbols:
        logger.warning("FALLBACK to legacy RiskManager: %s", fallback_reason)
        ctx["fallback_reason"] = fallback_reason
        try:
            from kite_connect.trading.risk_manager import RiskManager, RiskConfig
            rm = RiskManager(RiskConfig())
            plans = rm.plan_trades(buy_df)
            logger.warning("Fallback RiskManager: %d plans (reason: %s)", len(plans), fallback_reason)
        except Exception as exc:
            return "error", f"Both pipelines failed: {fallback_reason}; {exc}"

    if not plans:
        return "success", f"No new plans | exits={len(exit_events)}"

    # 5. Execute via PaperTrader (restored book; skip symbols already held)
    results = pt.execute_plans(plans, skip_held=True)
    filled = sum(1 for r in results if r.get("success"))

    # 6. SL/TP poll
    pt.poll()

    # 7. Signal audit log
    try:
        today_str = date.today().isoformat()
        signal_entries = []
        traded_symbols = {r["symbol"] for r in results if r.get("success")}
        _indiv = getattr(pipe_result, "individual_forecasts", {}) if pipe_result else {}
        for plan in plans:
            _sym_fc = _indiv.get(plan.symbol, {})
            _active = sorted(k for k, v in _sym_fc.items() if v and abs(v) > 0.01)
            signal_entries.append({
                "symbol": plan.symbol,
                "forecast": getattr(plan, "score", 0),
                "combined_forecast": getattr(plan, "score", 0),
                "action": plan.side,
                "entry_price": plan.entry_price,
                "stop_loss": plan.stop_loss,
                "target_price": plan.target_price,
                "quantity": plan.quantity,
                "pipeline_sources": (",".join(_active) if _active else
                                     ("RiskManager-fallback" if fallback_reason else "CarverPipeline")),
                "was_traded": plan.symbol in traded_symbols,
            })
        pt.log_signals(today_str, signal_entries)
    except Exception as exc:
        logger.debug("Signal logging failed (non-fatal): %s", exc)

    planner = "legacy_risk_manager" if fallback_reason else "carver"
    return "success", f"{filled}/{len(plans)} filled ({planner}) | exits={len(exit_events)}"


def _load_engine_deployment():
    """Deployed engine config (config/nse_engine_deployed.json or $CENTURION_NSE_DEPLOYMENT).

    A placeholder deployment is paper traded with a warning; live trading is
    refused by the executor.
    """
    from nse_engine.deployment import load_deployment

    dep = load_deployment(os.environ.get("CENTURION_NSE_DEPLOYMENT") or None)
    if dep.is_placeholder:
        logger.warning("NSE engine deployment %s is a PLACEHOLDER (status='placeholder'): "
                       "paper trading the default EngineConfig; live trading is refused until the "
                       "file is replaced with an approved configuration", dep.path)
    logger.info("NSE engine deployment: %s", dep.summary())
    return dep


def _run_engine_paper():
    """NSE engine path (EOD, after the bhavcopy is in the store).

    Processes the latest store session — gap stops, pending orders filled at
    that session's open, intraday stops — marks to the close, plans from the
    close (distribution-shift multiplier applied) and queues the new orders
    as PENDING for the next open.  Stops and prices come from the NSE store,
    not yfinance, so paper and backtest see the same bars.
    """
    from kite_connect.trading.nse_engine_executor import EngineExecutor

    dep = _load_engine_deployment()
    pt = _open_paper_trader()
    previous_session = pt.engine_last_session()
    executor = EngineExecutor(kite=None, paper=True, paper_trader=pt, deployment=dep)
    session = executor.run_paper_session()
    missed = _missed_sessions(previous_session, session.get("session"))
    if missed:
        session.setdefault("notes", []).append(
            f"MISSED {missed} session(s) since {previous_session}: orders decided then were "
            "cancelled as stale, so the book sat in cash for those days")
        logger.warning("Paper book missed %d session(s) after %s", missed, previous_session)
    plan = session.get("plan")
    entries = _engine_signal_entries(plan)
    n_traded = sum(1 for e in entries if e["was_traded"])
    if entries:
        try:
            pt.log_signals(session["session"], entries)
        except Exception as exc:
            logger.warning("Signal log failed: %s", exc)
    snapshot = {}
    try:
        snapshot = pt.snapshot_daily(signals_generated=len(entries), signals_traded=n_traded,
                                     session_date=session.get("session")) or {}
    except Exception as exc:
        logger.warning("Snapshot failed: %s", exc)
    try:
        cloud = pt._get_cloud()
        if cloud and hasattr(cloud, "sync_state"):
            cloud.sync_state({                               # tells other runners to leave this book alone
                "book_owner": "nse_engine",
                "book_writer": "github_actions",
                "book_writer_seen_at": datetime.now(timezone.utc).isoformat(),
            })
    except Exception as exc:
        logger.debug("book_owner sync skipped: %s", exc)
    fills = session.get("fills") or {}
    queued = sum(1 for r in session.get("results", []) if r.get("status") == "PENDING")
    shift = snapshot.get("distribution_shift") or {}
    msg = (f"engine session={session['session']} deployment={dep.status} "
           f"filled={len(fills.get('filled', []))} cancelled={len(fills.get('cancelled', []))} "
           f"stops={len(session.get('stops', []))} queued={queued} "
           f"skipped={len(plan.skipped) if plan else 0} "
           f"shift_mult={plan.shift_multiplier if plan else 1.0:.2f} cash={pt.cash:.0f}")
    if shift.get("reality_gap_alerts"):
        msg += f" | REALITY GAP: {'; '.join(shift['reality_gap_alerts'])}"
    for note in session.get("notes", []):
        msg += f" | {note}"
    logger.info("NSE engine paper run: %s", msg)
    _record_session_activity(pt, session, snapshot, plan, queued, fills)
    _email_engine_session(pt, dep, session, snapshot, shift)
    return "success", msg


def _session_outcome(plan, queued: int, fills: dict, stops: int, rebalance: bool) -> str:
    """One sentence for the trade monitor: what this session did, or why it did nothing."""
    filled, cancelled = len(fills.get("filled", [])), len(fills.get("cancelled", []))
    parts = []
    if filled:
        parts.append(f"{filled} order(s) filled at the open")
    if cancelled:
        parts.append(f"{cancelled} cancelled")
    if stops:
        parts.append(f"{stops} stop exit(s)")
    if queued:
        parts.append(f"{queued} order(s) queued for the next open")
    if parts:
        return "; ".join(parts)
    if plan is None:
        return "no plan: the session was already processed, so only stops and marks were refreshed"
    if not rebalance:
        return "held: not a rebalance day, so no orders were planned"
    return ("held: rebalance day, but every position was already within the no-trade buffer, "
            "so nothing needed trading")


def _record_session_activity(pt, session: dict, snapshot: dict, plan, queued: int, fills: dict) -> None:
    """Persist what the engine decided, so a quiet day is visible as a decision."""
    try:
        cloud = pt._get_cloud()
        if not cloud or not hasattr(cloud, "sync_session"):
            return
        notes = list(session.get("notes") or [])
        rebalance = bool(plan is not None and "rebalance_day" in (plan.notes or []))
        stops = len(session.get("stops") or [])
        cloud.sync_session({
            "session_date": session.get("session"),
            "ran_at": datetime.now(timezone.utc).isoformat(),
            "equity": float(snapshot.get("equity") or 0.0),
            "cash": float(pt.cash),
            "open_positions": int(snapshot.get("open_positions") or 0),
            "rebalance_day": rebalance,
            "planned_buys": len(plan.buys) if plan is not None else 0,
            "planned_sells": len(plan.sells) if plan is not None else 0,
            "queued": int(queued),
            "filled": len(fills.get("filled", [])),
            "cancelled": len(fills.get("cancelled", [])),
            "stops_triggered": stops,
            "stops_armed": len(plan.stop_instructions) if plan is not None else 0,
            "skipped": len(plan.skipped) if plan is not None else 0,
            "shift_multiplier": float(plan.shift_multiplier) if plan is not None else 1.0,
            "outcome": _session_outcome(plan, queued, fills, stops, rebalance)[:200],
            "notes": "; ".join(notes)[:500],
        })
    except Exception as exc:                              # noqa: BLE001 - reporting only
        logger.warning("Session activity not recorded: %s", exc)


def _missed_sessions(previous, current) -> int:
    """Trading sessions between the last processed one and this one (0 when consecutive).

    Counted on NSE weekdays, so a normal Friday-to-Monday gap is 0; holidays can
    show 1 and are harmless. Anything larger means the scheduler dropped a day.
    """
    import pandas as pd

    if not previous or not current:
        return 0
    try:
        a, b = pd.Timestamp(previous).date(), pd.Timestamp(current).date()
    except Exception:                                    # noqa: BLE001
        return 0
    if b <= a:
        return 0
    return max(len(pd.bdate_range(a, b)) - 2, 0)


def _email_engine_session(pt, dep, session: dict, snapshot: dict, shift: dict) -> None:
    """Daily email for a newly processed session (best-effort).

    A re-run of a session already processed (a backup cron or a manual
    re-dispatch) only re-plans, so it sends nothing: one email per session.
    """
    if session.get("fills") is None:
        logger.info("Daily email skipped: session %s was already processed", session.get("session"))
        return
    try:
        from services.notifications.manager import NotificationManager
        dash = pt.dashboard()
        fills = session.get("fills") or {}
        plan = session.get("plan")
        alerts = list(shift.get("reality_gap_alerts") or [])
        verdict = shift.get("position_verdict") or shift.get("effective_verdict") or shift.get("verdict")
        if verdict in ("drifting", "regime_break"):
            alerts.append(f"Distribution shift: {verdict} (size multiplier {shift.get('position_size_multiplier', shift.get('multiplier', '—'))})")
        sent = NotificationManager().email_engine_daily_report({
            "session": session.get("session"),
            "deployment": f"{dep.status} · paper since {dep.paper_start_date}",
            "equity": dash.current_capital,
            "initial_capital": dash.initial_capital,
            "cash": pt.cash,
            "pnl": dash.total_pnl,
            "pnl_pct": dash.total_pnl_pct,
            "max_drawdown_pct": snapshot.get("max_drawdown_pct", dash.max_drawdown_pct),
            "open_positions": dash.open_positions,
            "filled": fills.get("filled", []),
            "cancelled": fills.get("cancelled", []),
            "stops": session.get("stops", []),
            "queued": [r for r in session.get("results", []) if r.get("status") == "PENDING"],
            "notes": list(session.get("notes", [])) + (
                [f"shift multiplier {plan.shift_multiplier:.2f}"] if plan and plan.shift_multiplier != 1.0 else []),
            "alerts": alerts,
        })
        if not sent:
            logger.warning("Daily email returned False — check CENTURION_EMAIL_USER / CENTURION_EMAIL_PASS / "
                           "CENTURION_EMAIL_HOST / CENTURION_EMAIL_PORT secrets")
    except Exception as exc:
        logger.warning("Daily email failed: %s", exc)


# ── Weekly checkpoint (Saturday) ──────────────────────────────────────

def _run_weekly_checkpoint():
    """Run weekly checkpoint + send weekly performance email.

    Mirrors scheduler.py _run_paper_weekly_checkpoint + _send_paper_weekly_email.
    Called only on Saturdays via the Saturday GitHub Actions cron.
    """
    import sqlite3 as _sq3
    from services.notifications.manager import NotificationManager

    pt = _open_paper_trader()
    checkpoint = pt.checkpoint_weekly()

    if not checkpoint:
        logger.info("Weekly checkpoint: no new data this week — skipping email")
        return "success", "No weekly data"

    wk = checkpoint["week_number"]
    logger.info(
        "Weekly checkpoint W%d: return=%.1f%% sharpe=%.2f dd=%.1f%%",
        wk, checkpoint["week_return_pct"],
        checkpoint["sharpe_ratio"], checkpoint["max_dd_pct"],
    )

    # ── Build weekly email HTML ───────────────────────────────────
    dash = pt.dashboard()
    pnl_color = "#15803d" if dash.total_pnl >= 0 else "#dc2626"
    wk_color = "#15803d" if checkpoint["week_return_pct"] >= 0 else "#dc2626"

    # Historical weeks table
    weeks_rows = ""
    try:
        _db = _sq3.connect(str(_ROOT / "data" / "paper_trades.sqlite3"))
        _db.row_factory = _sq3.Row
        all_weeks = _db.execute(
            "SELECT * FROM weekly_checkpoints ORDER BY week_number"
        ).fetchall()
        _db.close()
        for w in all_weeks:
            w_color = "#15803d" if w["week_return_pct"] >= 0 else "#dc2626"
            weeks_rows += (
                f"<tr><td style='padding:4px 10px;border:1px solid #e5e7eb;font-weight:bold;'>W{w['week_number']}</td>"
                f"<td style='padding:4px 10px;border:1px solid #e5e7eb;'>{w['week_start']} → {w['week_end']}</td>"
                f"<td style='padding:4px 10px;border:1px solid #e5e7eb;text-align:right;font-weight:bold;color:{w_color};'>"
                f"{w['week_return_pct']:+.1f}%</td>"
                f"<td style='padding:4px 10px;border:1px solid #e5e7eb;text-align:right;'>{w['sharpe_ratio']:.2f}</td>"
                f"<td style='padding:4px 10px;border:1px solid #e5e7eb;text-align:right;color:#dc2626;'>{w['max_dd_pct']:.1f}%</td>"
                f"<td style='padding:4px 10px;border:1px solid #e5e7eb;text-align:right;'>"
                f"{w['trades_closed']}/{w['trades_opened']}</td>"
                f"<td style='padding:4px 10px;border:1px solid #e5e7eb;text-align:right;'>{w['win_rate'] * 100:.0f}%</td></tr>"
            )
    except Exception:
        pass

    # Verdict. A Sharpe over a handful of sessions is noise, and win rate is
    # undefined until something closes, so the gates only apply once the book
    # has a sample: below that the honest answer is "too early", not FAIL.
    sessions = pt.session_count()
    min_sessions = int(os.environ.get("CENTURION_PAPER_MIN_SESSIONS", "20"))
    if sessions < min_sessions:
        verdict = (f"TOO EARLY — {sessions} of {min_sessions} sessions; "
                   f"{dash.closed_trades} trades closed so far")
        verdict_color = "#6b7280"
        verdict_detail = (f"Ratios need closed trades and a few weeks of returns. "
                          f"Equity {dash.initial_capital:,.0f} → {dash.current_capital:,.0f} "
                          f"({dash.total_pnl_pct:+.1f}%).")
    elif dash.sharpe_ratio >= 0.5 and dash.max_drawdown_pct < 30:
        verdict = "PASS — Ready for live trading"
        verdict_color = "#15803d"
        verdict_detail = ""
    elif dash.sharpe_ratio >= 0.2:
        verdict = "MARGINAL — Consider extending paper period"
        verdict_color = "#d97706"
        verdict_detail = ""
    else:
        verdict = "FAIL — Do not go live, needs investigation"
        verdict_color = "#dc2626"
        verdict_detail = ""

    html = f"""\
<html><body style="font-family:Segoe UI,Arial,sans-serif;background:#f9fafb;padding:20px;">
<div style="max-width:700px;margin:0 auto;background:#fff;border-radius:10px;
            box-shadow:0 2px 8px rgba(0,0,0,0.08);overflow:hidden;">
  <div style="background:#1a1a2e;padding:16px 24px;">
    <h2 style="margin:0;color:#fff;font-size:18px;">
      Centurion &mdash; Weekly Paper Trade Report (Week {wk})
    </h2>
    <p style="margin:4px 0 0;color:#9ca3af;font-size:13px;">{checkpoint['week_start']} → {checkpoint['week_end']}</p>
  </div>
  <div style="padding:20px 24px;">

    <div style="background:#f0fdf4;border-left:4px solid {verdict_color};padding:12px 16px;margin-bottom:20px;border-radius:4px;">
      <strong style="color:{verdict_color};font-size:14px;">VERDICT: {verdict}</strong>
      <p style="margin:4px 0 0;color:#666;font-size:12px;">
        {verdict_detail or f"Sharpe {dash.sharpe_ratio:.3f} | Max DD {dash.max_drawdown_pct:.1f}% | Win Rate {dash.win_rate:.0%}"}
      </p>
    </div>

    <h3 style="color:#1a1a2e;margin-top:0;">This Week</h3>
    <table style="border-collapse:collapse;width:100%;font-size:14px;">
      <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">Week Return</td>
          <td style="padding:6px 12px;border:1px solid #e5e7eb;font-weight:bold;color:{wk_color};">{checkpoint['week_return_pct']:+.1f}%</td></tr>
      <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">Equity</td>
          <td style="padding:6px 12px;border:1px solid #e5e7eb;">₹{checkpoint['start_equity']:,.0f} → ₹{checkpoint['end_equity']:,.0f}</td></tr>
      <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">Trades</td>
          <td style="padding:6px 12px;border:1px solid #e5e7eb;">{checkpoint['trades_opened']} opened, {checkpoint['trades_closed']} closed</td></tr>
      <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">Sharpe (weekly)</td>
          <td style="padding:6px 12px;border:1px solid #e5e7eb;">{checkpoint['sharpe_ratio']:.2f}</td></tr>
      <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">Max Drawdown</td>
          <td style="padding:6px 12px;border:1px solid #e5e7eb;">{checkpoint['max_dd_pct']:.1f}%</td></tr>
      <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">Win Rate</td>
          <td style="padding:6px 12px;border:1px solid #e5e7eb;">{checkpoint['win_rate']:.0%}</td></tr>
      <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">Avg Holding Days</td>
          <td style="padding:6px 12px;border:1px solid #e5e7eb;">{checkpoint['avg_holding_days']:.1f}</td></tr>
    </table>

    <h3 style="color:#1a1a2e;margin-top:24px;">Cumulative Performance</h3>
    <table style="border-collapse:collapse;width:100%;font-size:14px;">
      <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">Capital</td>
          <td style="padding:6px 12px;border:1px solid #e5e7eb;">₹{dash.initial_capital:,.0f} → ₹{dash.current_capital:,.0f}</td></tr>
      <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">Total P&amp;L</td>
          <td style="padding:6px 12px;border:1px solid #e5e7eb;font-weight:bold;color:{pnl_color};">
            ₹{dash.current_capital - dash.initial_capital:,.0f} ({dash.total_pnl_pct:+.1f}%)</td></tr>
      <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">of which realised</td>
          <td style="padding:6px 12px;border:1px solid #e5e7eb;">
            ₹{dash.total_pnl:,.0f} from {dash.closed_trades} closed trades;
            ₹{dash.current_capital - dash.initial_capital - dash.total_pnl:,.0f} unrealised
            on {dash.open_positions} open</td></tr>
      <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">Sharpe / Sortino</td>
          <td style="padding:6px 12px;border:1px solid #e5e7eb;">{dash.sharpe_ratio:.3f} / {dash.sortino_ratio:.3f}</td></tr>
      <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">Profit Factor</td>
          <td style="padding:6px 12px;border:1px solid #e5e7eb;">{dash.profit_factor:.2f}</td></tr>
      <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">Max Drawdown</td>
          <td style="padding:6px 12px;border:1px solid #e5e7eb;">{dash.max_drawdown_pct:.1f}%</td></tr>
    </table>

    {"<h3 style='color:#1a1a2e;margin-top:24px;'>All Weeks</h3>" + chr(10) + "    <table style='border-collapse:collapse;width:100%;font-size:13px;'>" + chr(10) + "      <thead><tr style='background:#f3f4f6;'>" + chr(10) + "        <th style='padding:4px 10px;border:1px solid #e5e7eb;'>Wk</th>" + chr(10) + "        <th style='padding:4px 10px;border:1px solid #e5e7eb;'>Period</th>" + chr(10) + "        <th style='padding:4px 10px;border:1px solid #e5e7eb;text-align:right;'>Return</th>" + chr(10) + "        <th style='padding:4px 10px;border:1px solid #e5e7eb;text-align:right;'>Sharpe</th>" + chr(10) + "        <th style='padding:4px 10px;border:1px solid #e5e7eb;text-align:right;'>Max DD</th>" + chr(10) + "        <th style='padding:4px 10px;border:1px solid #e5e7eb;text-align:right;'>Trades</th>" + chr(10) + "        <th style='padding:4px 10px;border:1px solid #e5e7eb;text-align:right;'>Win Rate</th>" + chr(10) + "      </tr></thead><tbody>" + weeks_rows + "</tbody></table>" if weeks_rows else ""}

  </div>
  <div style="padding:12px 24px;background:#f3f4f6;font-size:11px;color:#999;text-align:center;">
    Centurion Paper Trading &bull; Week {wk} of 4 &bull; Auto-generated
  </div>
</div></body></html>"""

    subject = f"[Centurion Paper] Week {wk} Report | {checkpoint['week_return_pct']:+.1f}% | Sharpe {checkpoint['sharpe_ratio']:.2f}"
    sent = NotificationManager._send_html_email(
        subject=subject,
        html_body=html,
        recipients=["s.srees@live.com"],
    )
    if not sent:
        logger.warning("Weekly email returned False — check CENTURION_EMAIL_USER / CENTURION_EMAIL_PASS env vars")

    msg = f"W{wk}: return={checkpoint['week_return_pct']:+.1f}% sharpe={checkpoint['sharpe_ratio']:.2f}"
    return "success", msg


# ── Entrypoint ─────────────────────────────────────────────────────────

def _parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Cloud paper trading runner")
    parser.add_argument("--engine", action="store_true",
                        help="Use the NSE engine executor (requires CENTURION_NSE_ENGINE=true)")
    parser.add_argument("--new-book", action="store_true",
                        help="Engine only: start a fresh paper book at CENTURION_PAPER_INITIAL_CAPITAL "
                             "before this session (history is kept, scoped by epoch)")
    args, _ = parser.parse_known_args(argv)
    return args


def _engine_enabled(argv=None) -> bool:
    """``--engine`` runs the NSE engine only when CENTURION_NSE_ENGINE=true."""
    args = _parse_args(argv)
    env_on = os.environ.get("CENTURION_NSE_ENGINE", "false").lower() in ("true", "1", "yes")
    if args.engine and not env_on:
        logger.warning("--engine ignored: CENTURION_NSE_ENGINE is not 'true' — running legacy pipeline")
    return bool(args.engine and env_on)


def _start_new_book():
    """Open a fresh cloud book at the configured capital and drop any local copy.

    The engine's first real session must not inherit the legacy book (its
    capital, epoch and pending orders); and a stale local SQLite would be
    restored in preference to the cloud, so it goes too.
    """
    from database.paper_cloud import get_paper_cloud
    from kite_connect.trading.paper_trader import _DB_PATH

    cloud = get_paper_cloud()
    if cloud is None:
        raise RuntimeError("--new-book needs a Neon connection (CENTURION_DATABASE_URL)")
    values = cloud.start_new_book(_INITIAL_CAPITAL, owner="nse_engine")
    if Path(_DB_PATH).exists():
        Path(_DB_PATH).unlink()
        logger.info("Removed local paper book %s so the new cloud book is restored", _DB_PATH)
    logger.info("New paper book started: capital=%.0f epoch=%s", _INITIAL_CAPITAL, values["epoch"])
    return values


def _engine_signal_entries(plan) -> list:
    """The session's decisions as signal-log rows: queued orders and skipped names.

    ``was_traded`` means "queued as a PENDING order for the next open"; the fill
    itself is recorded as a position when it happens.
    """
    if plan is None:
        return []
    target = getattr(plan, "target", None)
    forecasts = dict(getattr(target, "forecasts", None) or {})
    stops = {s.symbol: float(s.trigger) for s in getattr(plan, "stop_instructions", [])}
    entries = []
    for o in getattr(plan, "orders", []):
        entries.append({
            "symbol": o.symbol, "forecast": float(forecasts.get(o.symbol, 0.0) or 0.0),
            "combined_forecast": float(forecasts.get(o.symbol, 0.0) or 0.0),
            "action": o.side, "entry_price": float(o.ref_price), "stop_loss": stops.get(o.symbol, 0.0),
            "target_price": 0.0, "quantity": int(o.quantity),
            "pipeline_sources": f"nse_engine:{o.reason}"[:120], "was_traded": True,
        })
    for sk in getattr(plan, "skipped", []):
        sym = sk.get("symbol")
        if not sym or sym == "*":
            continue
        entries.append({
            "symbol": sym, "forecast": float(forecasts.get(sym, 0.0) or 0.0),
            "combined_forecast": float(forecasts.get(sym, 0.0) or 0.0),
            "action": "SKIP", "entry_price": 0.0, "stop_loss": 0.0, "target_price": 0.0,
            "quantity": int(sk.get("quantity", 0) or 0),
            "pipeline_sources": f"nse_engine:skipped:{sk.get('reason', '')}"[:120], "was_traded": False,
        })
    return entries


def main(argv=None):
    logger.info("=== Cloud Paper Trading Runner ===")
    use_engine = _engine_enabled(argv)

    if not _check_active():
        logger.info("Paper trading is NOT active in Neon — skipping.")
        return

    logger.info("Paper trading is ACTIVE — running pipeline...")

    is_saturday = datetime.now().weekday() == 5  # 5 = Saturday

    if is_saturday:
        # Saturday: run weekly checkpoint + email only (no daily pipeline)
        logger.info("Saturday detected — running weekly checkpoint...")
        try:
            status, message = _run_weekly_checkpoint()
            _update_run_status(status, message[:500])
            logger.info("Weekly checkpoint complete: [%s] %s", status, message)
        except Exception as exc:
            _update_run_status("error", f"weekly: {str(exc)[:480]}")
            logger.exception("Weekly checkpoint failed: %s", exc)
            sys.exit(1)
    else:
        # Weekday: run full daily pipeline (or the NSE engine when enabled)
        try:
            if use_engine:
                if _parse_args(argv).new_book:
                    _start_new_book()
                status, message = _run_engine_paper()
            else:
                status, message = _run_paper_pipeline()
            _update_run_status(status, message[:500])
            logger.info("Run complete: [%s] %s", status, message)
        except Exception as exc:
            _update_run_status("error", str(exc)[:500])
            logger.exception("Pipeline failed: %s", exc)
            sys.exit(1)


if __name__ == "__main__":
    main()
