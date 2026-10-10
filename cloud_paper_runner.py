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
from typing import Optional

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

def _neon_url(raw: str) -> str:
    """Normalise a Neon URL for psycopg2.

    The driver must be named explicitly. SQLAlchemy used to resolve a bare
    ``postgresql://`` to psycopg2, but a release in Sep 2026 made it psycopg
    (v3), which this project does not install - every paper run then died with
    "No module named 'psycopg'" before it could reach the book.
    ``database/connection.py`` has always pinned the driver; this mirrors it.
    """
    import re

    url = raw
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    # channel_binding is a libpq option psycopg2 does not accept
    url = re.sub(r"[&?]channel_binding=[^&]*", "", url)
    if "sslmode" not in url:
        url += ("&" if "?" in url else "?") + "sslmode=require"
    return url


def _get_neon_engine():
    """Create a SQLAlchemy engine for Neon."""
    from sqlalchemy import create_engine

    raw = os.environ.get("CENTURION_DATABASE_URL", "")
    if not raw:
        raise RuntimeError("CENTURION_DATABASE_URL not set")
    return create_engine(_neon_url(raw), pool_pre_ping=True, pool_size=2)


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


def _book_label() -> str:
    """Name of a second paper book (``CENTURION_PAPER_BOOK_LABEL``, e.g. 'candidate'), '' for the deployed one."""
    return (os.environ.get("CENTURION_PAPER_BOOK_LABEL") or "").strip()


def _book_tag(dep) -> str:
    """Book name and engine fingerprint for email subjects: 'deployed 679cbd0c', 'candidate 2d64ba4c'."""
    return f"{_book_label() or 'deployed'} {dep.engine.config_hash()[:8]}"


def _book_objective(dep) -> str:
    """What this paper book is for, at the foot of its daily email."""
    from nse_engine import capital_ladder as cl, forward_gate as fg
    from nse_engine.deployment import STATUS_CANDIDATE

    fingerprint = dep.engine.config_hash()[:8]
    if dep.status == STATUS_CANDIDATE:
        return (f"Objective: trial configuration {fingerprint} beside the deployed book, on the same data and "
                f"with no real money. After {fg.FORWARD_MIN_SESSIONS} sessions the forward gate decides whether "
                f"it replaces the deployed configuration: its paper gate (G4) must pass and its walk-forward "
                f"Sharpe (2017-25) must be within {fg.WF_SHARPE_TOLERANCE} of the deployed one's.")
    return (f"Objective: rehearse the deployed configuration {fingerprint} with no real money before live "
            f"capital. Going live needs {cl.GO_LIVE_MIN_PAPER_SESSIONS}+ sessions with the paper gate (G4) "
            f"passing ({', '.join(cl.GO_CHECKS)}) and {cl.GO_LIVE_MIN_DRY_RUNS} clean "
            f"live dry runs; it then starts at ₹{cl.RUNGS[0] / 1e5:.0f} lakh on your go-ahead.")


def _update_run_status(status: str, message: str):
    """Update the last run status in Neon."""
    from sqlalchemy import text

    if os.environ.get("CENTURION_PAPER_SCHEMA"):
        # The switch row is shared: a second book (tracker D1) must not overwrite
        # the deployed book's run status on the Trade Center.
        logger.info("Run status not written for book '%s' (the switch row belongs to the deployed book): [%s] %s",
                    _book_label() or os.environ.get("CENTURION_PAPER_SCHEMA"), status, message[:200])
        return
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
    missed = max(len(session.get("caught_up") or []) - 1, 0)   # store sessions, weekends included (LN-T7)
    if missed:
        session.setdefault("notes", []).append(
            f"CAUGHT UP {missed} session(s) since {previous_session}: each one's stops and fills were applied "
            "in order; no plan was made for the sessions in between")
        logger.warning("Paper book caught up %d session(s) after %s", missed, previous_session)
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
    gate = _paper_gate(pt, dep.engine.config_hash())
    if gate:
        msg += f" | gate {gate.get('verdict')}"
    _email_engine_session(pt, dep, session, snapshot, shift, gate)
    return "success", msg


def _paper_gate(pt, config_hash: Optional[str] = None) -> Optional[dict]:
    """Paper pass/fail gate (G4) for this book, best-effort.

    Compares the book's equity with the same-period reference backtest the
    job wrote before the session (CENTURION_SHIFT_REFERENCE_CSV, else
    data/shift_reference_returns.csv, with its _trades.csv) and its fills in
    Neon.  The report is kept in the book's Neon state for the weekly email,
    with the configuration it judged (go-live reads it, tracker V5).
    """
    try:
        from nse_engine import paper_gate
        from services.research.distribution_shift import DEFAULT_REFERENCE_CSV
        ref_csv = Path(os.environ.get("CENTURION_SHIFT_REFERENCE_CSV") or DEFAULT_REFERENCE_CSV)
        if not ref_csv.exists():
            logger.info("Paper gate: no same-period reference at %s yet", ref_csv)
            return None
        ref, trades = paper_gate.read_reference(ref_csv)
        cloud = pt._get_cloud()
        fills = cloud.read_fills() if cloud is not None and hasattr(cloud, "read_fills") else None
        sessions = cloud.read_sessions() if cloud is not None and hasattr(cloud, "read_sessions") else None
        report = paper_gate.evaluate(pt.equity_history(), ref, fills=fills, reference_trades=trades,
                                     sessions=sessions)
        logger.info("Paper gate (G4): %s", paper_gate.one_line(report))
        if cloud is not None and hasattr(cloud, "sync_state"):
            cloud.sync_state({paper_gate.STATE_KEY: paper_gate.summary_json(
                report, updated_at=datetime.now(timezone.utc).isoformat(), config_hash=config_hash)})
        return report
    except Exception as exc:                              # noqa: BLE001 - never block a session
        logger.warning("Paper gate failed: %s", exc)
        return None


def _week_session_count(pt, checkpoint: dict) -> int:
    """Sessions inside this checkpoint's window - the sample its ratios rest on."""
    import sqlite3
    from kite_connect.trading.paper_trader import _DB_PATH

    try:
        conn = sqlite3.connect(str(_DB_PATH))
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM daily_snapshots WHERE date >= ? AND date <= ?",
                (checkpoint["week_start"], checkpoint["week_end"])).fetchone()
        finally:
            conn.close()
        return int(row[0]) if row else 0
    except Exception:                                     # noqa: BLE001 - reporting only
        return 0


def _drawdown_prefix(plan) -> str:
    """'DRAWDOWN RULE halt (21.3% below peak): no new entries; ' outside the normal state."""
    state = str(getattr(plan, "drawdown_state", "normal") or "normal")
    if plan is None or state == "normal":
        return ""
    effect = {"halt": "no new entries or adds", "half": "core exposure halved, no adds",
              "risk_off": "core book to cash / metals"}.get(state, state)
    return f"DRAWDOWN RULE {state} ({float(getattr(plan, 'drawdown_pct', 0.0)):.1f}% below peak): {effect}; "


def _session_outcome(plan, queued: int, fills: dict, stops: int, rebalance: bool) -> str:
    """One sentence for the Trade Center: what this session did, or why it did nothing."""
    return (_drawdown_prefix(plan) + _session_outcome_body(plan, queued, fills, stops, rebalance))[:200]


def _session_outcome_body(plan, queued: int, fills: dict, stops: int, rebalance: bool) -> str:
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
    """Persist what the engine decided, so a quiet day is visible as a decision.

    Skipped when the session was already processed: a re-plan (a backup cron or
    a manual re-dispatch) knows nothing about the fills and would overwrite them
    with zeros. On 25 Sep 2026 four late backups did exactly that, and the day
    the book actually traded ended up reading "held: not a rebalance day".
    """
    if session.get("fills") is None:
        logger.info("Session activity kept as recorded: %s was already processed",
                    session.get("session"))
        return
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
            "drawdown_state": str(getattr(plan, "drawdown_state", "normal") or "normal") if plan is not None else "normal",
            "drawdown_pct": float(getattr(plan, "drawdown_pct", 0.0) or 0.0) if plan is not None else 0.0,
        })
    except Exception as exc:                              # noqa: BLE001 - reporting only
        logger.warning("Session activity not recorded: %s", exc)


def _drift_check_line(pt, shift: dict, plan) -> str:
    """One line for the daily email: the drift check's verdict, or how long until it runs."""
    from kite_connect.trading.paper_trader import SHIFT_MIN_LIVE_DAYS

    applied = float(getattr(plan, "shift_multiplier", 1.0) or 1.0) if plan is not None else 1.0
    if shift:
        verdict = shift.get("position_verdict") or shift.get("effective_verdict") or shift.get("verdict") or "?"
        parts = [f"{verdict}"]
        te, gap = shift.get("tracking_error_annual"), shift.get("mean_daily_gap")
        if te is not None:
            parts.append(f"tracking error {float(te):.1%}/yr")
        if gap is not None:
            parts.append(f"gap {float(gap) * 1e4:+.1f} bp/day")
        parts.append(f"{shift.get('n_live', '?')} days vs {shift.get('reference_mode', '?')}")
        return " · ".join(parts) + f" · size today ×{applied:.2f}"
    try:
        n = max(int(pt.session_count()) - 1, 0)
    except Exception:                                    # noqa: BLE001
        n = 0
    return (f"waiting: {n} of {SHIFT_MIN_LIVE_DAYS} daily returns "
            f"(runs from session {SHIFT_MIN_LIVE_DAYS + 1}) · size today ×{applied:.2f}")


def _email_engine_session(pt, dep, session: dict, snapshot: dict, shift: dict, gate: Optional[dict] = None) -> None:
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
        drift_check = _drift_check_line(pt, shift, plan)
        gate_line = None
        if gate:
            from nse_engine import paper_gate
            gate_line = paper_gate.one_line(gate)
            if gate.get("verdict") == paper_gate.FAIL:
                alerts.append("PAPER GATE (G4) FAIL: " + "; ".join(
                    f"{c['name']} {c['display']}" for c in gate.get("checks", []) if c["status"] == paper_gate.FAIL))
        dd_state = str(getattr(plan, "drawdown_state", "normal") or "normal") if plan is not None else "normal"
        dd_pct = float(getattr(plan, "drawdown_pct", 0.0) or 0.0) if plan is not None else 0.0
        dd_line = None
        if plan is not None and dep is not None and getattr(dep, "drawdown_rule", None) is not None:
            dd_line = f"{dd_state} · {dd_pct:.1f}% below the episode peak"
            if getattr(plan, "drawdown_changed", False):
                alerts.append(f"DRAWDOWN RULE changed to {dd_state.upper()} at {dd_pct:.1f}% below the peak: "
                              + _drawdown_prefix(plan).split(": ", 1)[-1].rstrip("; ") if dd_state != "normal"
                              else f"DRAWDOWN RULE re-armed: back to normal (new 60-session equity high)")
        sent = NotificationManager().email_engine_daily_report({
            "session": session.get("session"),
            "deployment": f"{dep.status} {dep.engine.config_hash()[:8]} · paper since {dep.paper_start_date}",
            "book_label": _book_tag(dep),
            "objective": _book_objective(dep),
            "equity": dash.current_capital,
            "initial_capital": dash.initial_capital,
            "cash": pt.cash,
            "pnl": dash.total_pnl,
            "pnl_pct": dash.total_pnl_pct,
            "realised_pnl": dash.realised_pnl,
            "closed_trades": dash.closed_trades,
            "max_drawdown_pct": snapshot.get("max_drawdown_pct", dash.max_drawdown_pct),
            "open_positions": dash.open_positions,
            "filled": fills.get("filled", []),
            "cancelled": fills.get("cancelled", []),
            "stops": session.get("stops", []),
            "queued": [r for r in session.get("results", []) if r.get("status") == "PENDING"],
            "notes": list(session.get("notes", [])) + (
                [f"shift multiplier {plan.shift_multiplier:.2f}"] if plan and plan.shift_multiplier != 1.0 else []),
            "alerts": alerts,
            "drawdown_rule": dd_line,
            "drawdown_state": dd_state,
            "drift_check": drift_check,
            "paper_gate": gate_line,
        })
        if not sent:
            logger.warning("Daily email returned False — check CENTURION_EMAIL_USER / CENTURION_EMAIL_PASS / "
                           "CENTURION_EMAIL_HOST / CENTURION_EMAIL_PORT secrets")
    except Exception as exc:
        logger.warning("Daily email failed: %s", exc)


# ── Weekly checkpoint (Saturday) ──────────────────────────────────────

def _weekly_gate_verdict(pt) -> tuple:
    """(verdict, colour, detail) for the weekly email from the stored paper gate (G4)."""
    import json as _json
    from nse_engine import paper_gate
    gate = None
    try:
        cloud = pt._get_cloud()
        raw = (cloud.read_state() or {}).get(paper_gate.STATE_KEY) if cloud is not None else None
        gate = _json.loads(raw) if raw else None
    except Exception as exc:                              # noqa: BLE001 - reading only
        logger.warning("Paper gate state unavailable: %s", exc)
    if not gate:
        return ("NOT ENOUGH DATA — the paper gate (G4) has not run yet", "#6b7280",
                "It runs in each daily session once the same-period reference exists.")
    v = gate.get("verdict") or paper_gate.NOT_ENOUGH
    checks = " · ".join(f"{c['name']} {c['display']}" + ("" if v == paper_gate.NOT_ENOUGH else f" ({c['status']})")
                        for c in gate.get("checks", []))
    as_of = f"as of {gate.get('book_last')}" if gate.get("book_last") else ""
    if v == paper_gate.NOT_ENOUGH:
        return (f"NOT ENOUGH DATA — {gate.get('sessions', 0)} of {gate.get('min_sessions', 30)} sessions (G4)",
                "#6b7280", f"For information only, {as_of}: {checks}" if checks else as_of)
    text = {paper_gate.PASS: "PASS — behaves like its backtest (G4)",
            paper_gate.WATCH: "WATCH — a check is between its limits or not measurable yet (G4)",
            paper_gate.FAIL: "FAIL — does not behave like its backtest (G4): investigate before any promotion"}[v]
    colour = {paper_gate.PASS: "#15803d", paper_gate.WATCH: "#d97706", paper_gate.FAIL: "#dc2626"}[v]
    detail = f"{gate.get('sessions')} sessions {as_of}: {checks}."
    if v == paper_gate.PASS:
        detail += " Promotion still needs 60 sessions and the forward gate (V3)."
    return text, colour, detail


def _run_weekly_checkpoint(use_engine: bool = False):
    """Run weekly checkpoint + send weekly performance email.

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
    week_days = max(_week_session_count(pt, checkpoint), 1)
    pnl_color = "#15803d" if dash.total_pnl >= 0 else "#dc2626"
    wk_color = "#15803d" if checkpoint["week_return_pct"] >= 0 else "#dc2626"

    # Historical weeks table
    weeks_rows = ""
    try:
        from kite_connect.trading.paper_trader import _DB_PATH   # this book's file (the candidate has its own)
        _db = _sq3.connect(str(_DB_PATH))
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

    sessions = pt.session_count()
    min_sessions = int(os.environ.get("CENTURION_PAPER_MIN_SESSIONS", "20"))

    # Ratios need a sample. Below `min_sessions` they are printed as "n/a" with
    # the session count, so nobody reads a 7-day Sharpe of 4.5 as information.
    def ratio(value: float, fmt: str = "{:.3f}") -> str:
        return fmt.format(value) if sessions >= min_sessions else f"n/a ({sessions} sessions)"

    # Verdict: the paper gate (G4), computed by the latest daily session and kept
    # in Neon.  It judges behaviour against the same-period backtest; a Sharpe
    # over a few weeks is noise, so returns are shown but never gate.
    verdict, verdict_color, verdict_detail = _weekly_gate_verdict(pt)

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
          <td style="padding:6px 12px;border:1px solid #e5e7eb;">{checkpoint['sharpe_ratio']:.2f}
            <span style="color:#9ca3af;font-size:12px;">&mdash; from {week_days} session(s), not yet meaningful</span></td></tr>
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
            ₹{dash.total_pnl:,.0f} ({dash.total_pnl_pct:+.1f}%)</td></tr>
      <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">of which realised</td>
          <td style="padding:6px 12px;border:1px solid #e5e7eb;">
            ₹{dash.realised_pnl:,.0f} from {dash.closed_trades} closed trades;
            ₹{dash.total_pnl - dash.realised_pnl:,.0f} unrealised
            on {dash.open_positions} open</td></tr>
      <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">Sharpe / Sortino</td>
          <td style="padding:6px 12px;border:1px solid #e5e7eb;">{ratio(dash.sharpe_ratio)} / {ratio(dash.sortino_ratio)}</td></tr>
      <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">Profit Factor</td>
          <td style="padding:6px 12px;border:1px solid #e5e7eb;">{ratio(dash.profit_factor, "{:.2f}")}</td></tr>
      <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">Max Drawdown</td>
          <td style="padding:6px 12px;border:1px solid #e5e7eb;">{dash.max_drawdown_pct:.1f}%</td></tr>
    </table>

    {"<h3 style='color:#1a1a2e;margin-top:24px;'>All Weeks</h3>" + chr(10) + "    <table style='border-collapse:collapse;width:100%;font-size:13px;'>" + chr(10) + "      <thead><tr style='background:#f3f4f6;'>" + chr(10) + "        <th style='padding:4px 10px;border:1px solid #e5e7eb;'>Wk</th>" + chr(10) + "        <th style='padding:4px 10px;border:1px solid #e5e7eb;'>Period</th>" + chr(10) + "        <th style='padding:4px 10px;border:1px solid #e5e7eb;text-align:right;'>Return</th>" + chr(10) + "        <th style='padding:4px 10px;border:1px solid #e5e7eb;text-align:right;'>Sharpe</th>" + chr(10) + "        <th style='padding:4px 10px;border:1px solid #e5e7eb;text-align:right;'>Max DD</th>" + chr(10) + "        <th style='padding:4px 10px;border:1px solid #e5e7eb;text-align:right;'>Trades</th>" + chr(10) + "        <th style='padding:4px 10px;border:1px solid #e5e7eb;text-align:right;'>Win Rate</th>" + chr(10) + "      </tr></thead><tbody>" + weeks_rows + "</tbody></table>" if weeks_rows else ""}

  </div>
  <div style="padding:12px 24px;background:#f3f4f6;font-size:11px;color:#999;text-align:center;">
    Centurion Paper Trading &bull; Week {wk} of 4 &bull; Auto-generated
  </div>
</div></body></html>"""

    tag = _book_tag(_load_engine_deployment()) if use_engine else _book_label()
    label = f"[{tag}] " if tag else ""
    subject = (f"[Centurion Paper] {label}Week {wk} | {checkpoint['week_return_pct']:+.1f}% | "
               f"{checkpoint['trades_opened']} opened, {checkpoint['trades_closed']} closed | "
               f"gate {verdict.split(' — ')[0]}")
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
            status, message = _run_weekly_checkpoint(use_engine)
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
