"""Retired scheduler jobs kept for later use: strategy maintenance.

Not registered in ``start_scheduler`` (tracker H1, H3); moved from scheduler.py
(tracker H4).
"""

from __future__ import annotations

import logging

from scheduling.cache import _tracked_job

logger = logging.getLogger("centurion.scheduler")


@_tracked_job("forecast_calibration", "Forecast Calibration")
def _run_forecast_calibration():
    """Auto-calibrate forecast scalars from recent OHLCV data.

    Runs weekly (Saturday 5:30 AM IST, before walk-forward audit).
    Re-computes EWMAC / screener / decision-engine / carry scalars from
    expanding-window median(|raw forecast|) and persists to
    ``data/calibrated_scalars.json``.
    """
    logger.info("=== Forecast Scalar Calibration started ===")
    try:
        from config import Config
        if not getattr(Config, "AUTO_CALIBRATE_SCALARS", True):
            logger.info("AUTO_CALIBRATE_SCALARS disabled — skipping")
            return

        from services.signals.forecast_scalar import calibrate_all_scalars
        import yfinance as yf

        # Build OHLCV cache for representative NIFTY-50 tickers
        tickers = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
            "ICICIBANK.NS", "HINDUNILVR.NS", "BHARTIARTL.NS",
            "SBIN.NS", "KOTAKBANK.NS", "LT.NS",
        ]
        ohlcv_cache: dict = {}
        for t in tickers:
            try:
                df = yf.download(t, period="2y", progress=False, timeout=15)
                if df is not None and len(df) > 100:
                    ohlcv_cache[t] = df
            except Exception:
                logger.debug("OHLCV fetch failed for %s", t)

        if len(ohlcv_cache) < 3:
            logger.warning("Too few tickers for calibration (%d) — skipping", len(ohlcv_cache))
            return

        result = calibrate_all_scalars(ohlcv_cache)
        logger.info("Forecast calibration complete: %s", result)

    except Exception as exc:
        logger.exception("Forecast scalar calibration failed: %s", exc)


# ═══════════════════════════════════════════════════════════════
# Phase 2 Gap B1: Monthly HMM Regime Re-fit
# ═══════════════════════════════════════════════════════════════

@_tracked_job("hmm_refit", "HMM Regime Re-fit")
def _run_hmm_refit():
    """Monthly re-fit of the HMM regime model on 5 years of NIFTY data.

    Trains a 3-state Gaussian HMM on [log_returns, VIX, breadth, delivery_vol]
    and persists the model to data/hmm_model.pkl for use by the pipeline.
    """
    logger.info("=== HMM Regime Re-fit started ===")
    try:
        from config import Config
        if not getattr(Config, "HMM_ENABLED", True):
            logger.info("HMM_ENABLED=False — skipping re-fit")
            return

        import yfinance as yf
        from services.regime.regime_hmm import MarkovRegimeModel, prepare_hmm_observations

        # Fetch 5 years of NIFTY 50 daily data
        nifty_df = yf.download("^NSEI", period="5y", progress=False, timeout=30)
        if nifty_df is None or len(nifty_df) < 500:
            logger.warning("Insufficient NIFTY data for HMM fit (%d rows)", len(nifty_df) if nifty_df is not None else 0)
            return

        # Fetch India VIX
        vix_df = None
        try:
            vix_df = yf.download("^INDIAVIX", period="5y", progress=False, timeout=15)
        except Exception:
            logger.debug("India VIX fetch failed — using proxy")

        # Prepare observation matrix
        observations = prepare_hmm_observations(nifty_df, vix_df)
        if len(observations) < 500:
            logger.warning("Too few valid observations for HMM (%d) — need 500+", len(observations))
            return

        # Fit model
        model = MarkovRegimeModel(
            n_states=getattr(Config, "HMM_N_STATES", 3),
            n_features=4,
        )
        model.fit(observations)

        # Get current regime for logging
        snap = model.get_current_regime(observations[-60:])
        logger.info(
            "HMM re-fit complete: regime=%s (%.0f%%), durations=%s",
            snap.regime, snap.confidence * 100, snap.expected_durations,
        )

        # Persist model
        model.save()
        logger.info("HMM model persisted to disk")

        # Update singleton
        from services.regime.regime_hmm import get_hmm_model
        global_model = get_hmm_model()
        global_model._fitted = model._fitted
        global_model._means = model._means
        global_model._covars = model._covars
        global_model._transmat = model._transmat
        global_model._startprob = model._startprob
        global_model._feat_mean = model._feat_mean
        global_model._feat_std = model._feat_std

        logger.info("=== HMM Regime Re-fit complete ===")

    except Exception as exc:
        logger.exception("HMM re-fit failed: %s", exc)


@_tracked_job("strategy_tournament", "Strategy Tournament")
def _run_strategy_tournament():
    """Monthly strategy tournament to rank and auto-allocate strategies.

    Runs 1st Saturday of each month at 4:00 AM IST.
    Evaluates all forecast sources over the trailing 3 months,
    disables underperformers, and persists allocation decisions.
    """
    logger.info("=== Monthly Strategy Tournament started ===")
    try:
        import pandas as pd
        from services.research.strategy_tournament import StrategyTournament
        import json, os

        # Load recent per-strategy returns from walk-forward results
        wf_results_path = os.path.join("data", "walk_forward_results.json")
        if not os.path.exists(wf_results_path):
            logger.warning("No walk-forward results found — skipping tournament")
            return

        with open(wf_results_path) as f:
            wf_data = json.load(f)

        # Convert to per-strategy return series
        strat_returns = {}
        for strat_name, returns_list in wf_data.items():
            if isinstance(returns_list, list) and len(returns_list) >= 20:
                strat_returns[strat_name] = pd.Series(returns_list)

        if len(strat_returns) < 2:
            logger.warning("Too few strategies with results (%d) — skipping", len(strat_returns))
            return

        tourney = StrategyTournament(top_n=5, min_sharpe=0.0)
        result = tourney.run_tournament(strat_returns, lookback_months=3)

        logger.info("Tournament complete: top=%s, disabled=%s",
                     result.top_strategies, result.disabled_strategies)

        # Persist tournament results for pipeline to read
        tourney_path = os.path.join("data", "tournament_results.json")
        with open(tourney_path, "w") as f:
            json.dump({
                "top_strategies": result.top_strategies,
                "disabled_strategies": result.disabled_strategies,
                "entries": [
                    {"rank": e.rank, "name": e.strategy_name,
                     "score": e.composite_score, "status": e.allocation_status}
                    for e in result.entries
                ],
            }, f, indent=2)

    except Exception as exc:
        logger.exception("Strategy tournament failed: %s", exc)


@_tracked_job("pead_earnings", "PEAD Earnings Feed")
def _run_pead_earnings_feed():
    """G6: Fetch recent earnings data and feed into PEAD strategy.

    Bridges earnings_momentum.py (Trendlyne scraper) with pead_strategy.py
    (PEAD signal generator) to automate the earnings data pipeline.
    """
    logger.info("=== PEAD Earnings Feed started ===")
    try:
        from services.signals.earnings_momentum import _fetch_recent_results, EarningsSurprise as EMSurprise
        from services.execution.pead_strategy import PEADStrategy, EarningsSurprise as PEADSurprise

        # Fetch recent earnings from Trendlyne
        raw_results = _fetch_recent_results()
        if not raw_results:
            logger.info("PEAD feed: no recent earnings data found")
            return

        # Convert earnings_momentum format to PEAD format
        pead_surprises = []
        for sym, em_data in raw_results.items():
            try:
                # Use profit surprise as primary SUE proxy
                sue = em_data.profit_surprise_pct / 10.0  # normalize to ~SUE scale
                surprise = PEADSurprise(
                    ticker=sym,
                    announcement_date=em_data.result_date,
                    eps_actual=em_data.profit_surprise_pct,  # proxy
                    eps_consensus=0.0,
                    sue=sue,
                    surprise_pct=em_data.profit_surprise_pct,
                    direction="POSITIVE" if em_data.is_positive else "NEGATIVE",
                )
                pead_surprises.append(surprise)
            except Exception:
                continue

        if not pead_surprises:
            logger.info("PEAD feed: no valid earnings surprises to process")
            return

        # Feed into PEAD strategy
        pead = PEADStrategy()
        new_signals = pead.process_earnings(pead_surprises)
        logger.info("PEAD feed: processed %d earnings, generated %d new signals",
                     len(pead_surprises), len(new_signals))

    except Exception as exc:
        logger.exception("PEAD earnings feed failed: %s", exc)


@_tracked_job("meta_label_retrain", "Meta-Label Retrain")
def _run_meta_label_retrain():
    """AFML Ch.3: Retrain the meta-labeling classifier.

    Aggregates OHLCV data for all tracked symbols and trains the
    secondary classifier that predicts forecast correctness using
    triple-barrier labels.
    """
    logger.info("=== Meta-Label Retrain started ===")
    try:
        from services.signals.meta_labeling import train_meta_labeler
        import yfinance as yf

        # Gather OHLCV for IND symbols
        from config import Config
        tickers = getattr(Config, "MONITORED_TICKERS", [])
        if not tickers:
            # Fallback: read from sample_tickers.csv
            from pathlib import Path
            csv_path = Path(__file__).parent.parent / "sample_tickers.csv"
            if csv_path.exists():
                import csv
                with open(csv_path) as f:
                    reader = csv.reader(f)
                    tickers = [row[0].strip() for row in reader if row]

        if not tickers:
            logger.warning("Meta-label retrain: no tickers configured")
            return

        # Download 2 years of data
        ohlcv_cache = {}
        for ticker in tickers[:50]:  # cap at 50 symbols
            try:
                df = yf.download(ticker, period="2y", progress=False)
                if df is not None and len(df) > 252:
                    ohlcv_cache[ticker] = df
            except Exception:
                continue

        if len(ohlcv_cache) < 5:
            logger.warning("Meta-label retrain: insufficient data (%d symbols)", len(ohlcv_cache))
            return

        # Train IND model
        result_ind = train_meta_labeler(ohlcv_cache, market="IND")
        logger.info("Meta-label IND: %s", result_ind.get("status", "unknown"))

        # Train US model (if US tickers configured)
        us_tickers = getattr(Config, "US_MONITORED_TICKERS", [])
        if us_tickers:
            us_cache = {}
            for ticker in us_tickers[:30]:
                try:
                    df = yf.download(ticker, period="2y", progress=False)
                    if df is not None and len(df) > 252:
                        us_cache[ticker] = df
                except Exception:
                    continue
            if len(us_cache) >= 3:
                result_us = train_meta_labeler(us_cache, market="US")
                logger.info("Meta-label US: %s", result_us.get("status", "unknown"))

    except Exception as exc:
        logger.exception("Meta-label retrain failed: %s", exc)


@_tracked_job("event_calendar", "Event Calendar Seed")
def _run_event_calendar_seed():
    """Job 17: Seed fixed events (RBI, rebalance) into the calendar DB."""
    try:
        from config import Config
        if not getattr(Config, "EVENT_DRIVEN_ENABLED", False):
            return
    except Exception:
        return

    logger.info("=== Event Calendar Seed ===")
    try:
        from services.market_data.event_calendar import seed_fixed_events
        seed_fixed_events()
    except Exception as exc:
        logger.exception("Event calendar seed failed: %s", exc)
