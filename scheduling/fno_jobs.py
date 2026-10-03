"""Retired scheduler jobs kept for later use: options and futures, with their
execution helpers.

Not registered in ``start_scheduler`` (tracker H1, H3); moved from scheduler.py
(tracker H4).
"""

from __future__ import annotations

import logging

from scheduling.cache import _tracked_job

logger = logging.getLogger("centurion.scheduler")


# ═══════════════════════════════════════════════════════════════
# Scheduler-local Kite session (Docker / HF-safe — no Selenium)
# ═══════════════════════════════════════════════════════════════

def _get_scheduler_kite(force_refresh: bool = False):
    """No Kite session in the scheduler (tracker H1): returns None.

    The headless password + TOTP login it used to fall back to is removed;
    the retired options/futures jobs that still call it are kept for later
    use (tracker H3).
    """
    return None


# ===================================================================
# Phase 1-4: Advanced strategy job handlers
# ===================================================================

@_tracked_job("options_monitor", "Options Monitor")
def _run_options_monitor():
    """Job 13: Poll open options positions for profit-take / roll / expiry."""
    try:
        from config import Config
        if not getattr(Config, "OPTIONS_ENABLED", False):
            return
    except Exception:
        return

    logger.info("=== Options Monitor Poll ===")
    try:
        kite = _get_scheduler_kite()
        if kite is None:
            logger.warning("Options monitor: no Kite session")
            return
        from kite_connect.options.options_monitor import run_options_monitor_poll
        run_options_monitor_poll(kite)
    except Exception as exc:
        logger.exception("Options monitor failed: %s", exc)


@_tracked_job("margin_monitor", "Margin Monitor")
def _run_margin_monitor():
    """Job 14: Check margin utilisation and alert if thresholds breached."""
    try:
        from config import Config
        if not getattr(Config, "LEVERAGE_ENABLED", False):
            return
    except Exception:
        return

    logger.info("=== Margin Monitor Poll ===")
    try:
        kite = _get_scheduler_kite()
        if kite is None:
            logger.warning("Margin monitor: no Kite session")
            return
        from kite_connect.trading.margin_monitor import get_margin_snapshot
        snap = get_margin_snapshot(kite)
        if snap.alert_level in ("WARNING", "CRITICAL"):
            logger.warning(
                "Margin %s: %.1f%% used (%.0f / %.0f)",
                snap.alert_level, snap.utilisation_pct,
                snap.used, snap.available + snap.used,
            )
    except Exception as exc:
        logger.exception("Margin monitor failed: %s", exc)


@_tracked_job("pairs_scanner", "Pairs Scanner")
def _run_pairs_scanner():
    """Job 15: Scan configured pairs for mean-reversion signals."""
    try:
        from config import Config
        if not getattr(Config, "PAIRS_ENABLED", False):
            return
    except Exception:
        return

    logger.info("=== Pairs Trading Scanner ===")
    try:
        import numpy as np
        from utils import download_ind_ohlcv
        from services.execution.pairs_trading_live import scan_all_pairs, DEFAULT_PAIRS

        pairs = getattr(Config, "PAIRS_LIST", DEFAULT_PAIRS)
        symbols = set()
        for a, b in pairs:
            symbols.add(a)
            symbols.add(b)

        price_data = {}
        for sym in symbols:
            try:
                df = download_ind_ohlcv(sym, period="6mo")
                if df is not None and len(df) >= 60:
                    col = "Close" if "Close" in df.columns else "close"
                    price_data[sym] = df[col].values.astype(float)
            except Exception:
                pass

        signals = scan_all_pairs(price_data)
        if signals:
            logger.info("Pairs signals: %d active", len(signals))
            for s in signals:
                logger.info("  %s/%s z=%.2f action=%s forecast=%.1f",
                            s.leg1, s.leg2, s.z_score, s.action, s.forecast)

            # G5: Execute pairs via SpreadExecutor
            _execute_pairs_signals(signals)

    except Exception as exc:
        logger.exception("Pairs scanner failed: %s", exc)


@_tracked_job("futures_monitor", "Futures Monitor")
def _run_futures_monitor():
    """Job 16: Monitor futures positions for rollover and de-leveraging."""
    try:
        from config import Config
        if not getattr(Config, "LEVERAGE_ENABLED", False):
            return
    except Exception:
        return

    logger.info("=== Futures Monitor ===")
    try:
        kite = _get_scheduler_kite()
        if kite is None:
            logger.warning("Futures monitor: no Kite session")
            return
        from kite_connect.trading.futures_monitor import run_futures_monitor
        result = run_futures_monitor(kite)
        for alert in result.alerts:
            logger.warning("Futures alert: %s", alert)

        # G6: Execute futures overlay signal
        _execute_futures_overlay(kite)

    except Exception as exc:
        logger.exception("Futures monitor failed: %s", exc)


# =================================================================
# G4/G5/G6/G10: Execution Helper Functions
# =================================================================

def _execute_options_overlay(kite):
    """G4: Execute options overlay - covered calls + CSPs."""
    try:
        from config import Config
        if not getattr(Config, "OPTIONS_ENABLED", False):
            return

        from kite_connect.options.options_executor import OptionsExecutor
        from services.execution.options_overlay import scan_covered_call_candidates, scan_csp_candidates

        executor = OptionsExecutor(kite)

        # Covered calls on existing long positions
        try:
            cc_candidates = scan_covered_call_candidates(kite)
            if cc_candidates:
                results = executor.execute_covered_calls(cc_candidates)
                logger.info("G4: Covered calls executed: %d orders", len(results))
        except Exception as exc:
            logger.warning("G4: Covered calls failed: %s", exc)

        # Cash-secured puts on high-conviction BUY signals
        try:
            csp_candidates = scan_csp_candidates(kite)
            if csp_candidates:
                results = executor.execute_cash_secured_puts(csp_candidates)
                logger.info("G4: CSPs executed: %d orders", len(results))
        except Exception as exc:
            logger.warning("G4: CSPs failed: %s", exc)

    except Exception as exc:
        logger.debug("G4: Options overlay skipped: %s", exc)


def _execute_tail_hedge_if_needed(kite):
    """G10: Auto-execute tail hedge when drawdown is critical."""
    try:
        from config import Config
        if not getattr(Config, "OPTIONS_TAIL_HEDGE_ENABLED", False):
            return

        from services.risk.tail_risk_hedge import TailRiskHedge
        from kite_connect.options.options_executor import OptionsExecutor

        # Get current portfolio state
        capital = getattr(Config, "CARVER_INITIAL_CAPITAL", 500_000)
        realized = getattr(Config, "_CUMULATIVE_REALIZED_PNL", 0.0)
        equity = getattr(Config, "_CURRENT_EQUITY", capital + realized)
        peak = getattr(Config, "_PEAK_EQUITY", capital)
        dd_pct = ((peak - equity) / peak * 100) if peak > 0 else 0

        # Get VIX
        try:
            import yfinance as yf
            vix_data = yf.download("^INDIAVIX", period="5d", progress=False)
            vix = float(vix_data["Close"].iloc[-1]) if len(vix_data) > 0 else 15.0
            vix_3d = float(vix_data["Close"].iloc[-4]) if len(vix_data) >= 4 else vix
        except Exception:
            vix, vix_3d = 15.0, 15.0

        # Get NIFTY spot
        try:
            ltp = kite.ltp(["NSE:NIFTY 50"])
            nifty_spot = ltp.get("NSE:NIFTY 50", {}).get("last_price", 0)
        except Exception:
            nifty_spot = 0

        hedger = TailRiskHedge()
        assessment = hedger.assess(
            portfolio_value=equity,
            drawdown_pct=dd_pct,
            vix=vix,
            vix_3d_ago=vix_3d,
            nifty_spot=nifty_spot,
        )

        if assessment.hedge_urgency in ("HIGH", "CRITICAL") and assessment.recommendation:
            executor = OptionsExecutor(kite)
            result = executor.execute_tail_hedge(assessment.recommendation)
            logger.info("G10: Tail hedge executed: urgency=%s result=%s",
                        assessment.hedge_urgency, result)
        else:
            logger.info("G10: Tail hedge not needed: urgency=%s dd=%.1f%%",
                        assessment.hedge_urgency, dd_pct)

    except Exception as exc:
        logger.debug("G10: Tail hedge check skipped: %s", exc)


def _execute_pairs_signals(signals):
    """G5: Execute pairs trading signals via SpreadExecutor."""
    try:
        kite = _get_scheduler_kite()
        if kite is None:
            logger.warning("G5: No Kite session for pairs execution")
            return

        from kite_connect.trading.spread_executor import SpreadExecutor, LegOrder
        spread_exec = SpreadExecutor(kite)

        for sig in signals:
            if not hasattr(sig, "action") or sig.action not in ("ENTER_LONG", "ENTER_SHORT"):
                continue

            # Build leg orders based on signal direction
            if sig.action == "ENTER_LONG":
                leg1 = LegOrder(symbol=sig.leg1, side="BUY", quantity=1, exchange="NSE")
                leg2 = LegOrder(symbol=sig.leg2, side="SELL", quantity=1, exchange="NSE")
            else:  # ENTER_SHORT
                leg1 = LegOrder(symbol=sig.leg1, side="SELL", quantity=1, exchange="NSE")
                leg2 = LegOrder(symbol=sig.leg2, side="BUY", quantity=1, exchange="NSE")

            result = spread_exec.execute_pair(leg1, leg2)
            logger.info("G5: Pair %s/%s %s: success=%s",
                        sig.leg1, sig.leg2, sig.action, result.success)

    except Exception as exc:
        logger.warning("G5: Pairs execution failed: %s", exc)


def _execute_futures_overlay(kite):
    """G6: Execute futures overlay signal for regime-adaptive leverage."""
    try:
        from config import Config
        if not getattr(Config, "LEVERAGE_ENABLED", False):
            return

        from services.execution.futures_overlay import compute_futures_overlay
        from services.regime.regime_detector import get_current_regime
        from kite_connect.trading.order_service import place_order

        # Get current state
        capital = getattr(Config, "CARVER_INITIAL_CAPITAL", 500_000)
        realized = getattr(Config, "_CUMULATIVE_REALIZED_PNL", 0.0)
        equity = capital + realized

        regime_info = get_current_regime()
        regime = regime_info.get("regime", "range")
        confidence = regime_info.get("confidence", 0.5)

        # Get NIFTY spot/futures price
        try:
            ltp = kite.ltp(["NSE:NIFTY 50"])
            nifty_spot = ltp.get("NSE:NIFTY 50", {}).get("last_price", 0)
        except Exception:
            nifty_spot = 0

        signal = compute_futures_overlay(
            portfolio_value=equity,
            current_futures_notional=0.0,
            nifty_spot=nifty_spot,
            regime=regime,
            regime_confidence=confidence,
        )

        if signal.action == "BUY_FUT" and signal.lots > 0:
            lot_size = getattr(Config, "FUTURES_LOT_SIZE", 25)
            result = place_order(
                kite,
                tradingsymbol="NIFTY" + _get_current_expiry_suffix(),
                exchange="NFO",
                transaction_type="BUY",
                quantity=signal.lots * lot_size,
                product="NRML",
                order_type="MARKET",
            )
            logger.info("G6: BUY_FUT %d lots, order=%s", signal.lots, result)

        elif signal.action == "SELL_FUT" and signal.lots > 0:
            lot_size = getattr(Config, "FUTURES_LOT_SIZE", 25)
            result = place_order(
                kite,
                tradingsymbol="NIFTY" + _get_current_expiry_suffix(),
                exchange="NFO",
                transaction_type="SELL",
                quantity=signal.lots * lot_size,
                product="NRML",
                order_type="MARKET",
            )
            logger.info("G6: SELL_FUT %d lots, order=%s", signal.lots, result)

        else:
            logger.debug("G6: Futures overlay action=%s lots=%d (no trade)",
                         signal.action, signal.lots)

    except Exception as exc:
        logger.debug("G6: Futures overlay skipped: %s", exc)


def _get_current_expiry_suffix():
    """Get current month NIFTY futures expiry suffix (e.g. '25JUN' for Jun 2025)."""
    from datetime import date
    today = date.today()
    # NFO convention: YYMMMFUT e.g. NIFTY25JUNFUT
    suffix = today.strftime("%y%b").upper() + "FUT"
    return suffix
