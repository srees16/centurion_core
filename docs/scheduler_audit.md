# Scheduler audit (H2, 1 Oct 2026)

`scheduler.py` runs on the HF Space beside the API (`deployment/start.sh`).
`start_scheduler()` registers 32 jobs, nearly all unconditionally. The NSE
engine is paper- and live-traded by GitHub Actions, not by the scheduler.
Read-only audit; the claims marked *verified* were re-checked by hand.

## Keep (4)

| Job | Schedule (IST) | Why |
|---|---|---|
| `nse_engine_dispatch` | 19:00 Mon–Fri | the punctual trigger of the nightly paper/live workflow |
| `nse_engine_dispatch_retry` | 20:30 Mon–Fri | its retry; no-op once the session is processed |
| `kite_login_reminder` | 09:00 Mon–Fri | dispatches the daily Kite login email (U23) |
| `kite_login_reminder_evening` | 17:30 Mon–Fri | its follow-up when nobody has logged in |

None of the four touches Kite.

## Retire (28)

| Job(s) | Schedule | Kite contact | Consumers |
|---|---|---|---|
| `trade_monitor_poll` | every 3 min, market hours | **logs in on every run** (*verified*), then places / modifies / cancels orders and GTTs for active trades, **no paper gate** | trade-monitor summary card (legacy SQLite) |
| `options_monitor` | every 5 min | **logs in on every run** (*verified*); MARKET NRML closes for open journal rows, no paper gate | nobody |
| `kite_token_refresh` | :00/:30, 09:00–16:30 | automated password + TOTP login, nothing else | nobody |
| `daily_carver_rebalance` | 09:30 | always logs in; reads the real account; MARKET CNC orders and liquidation if `CENTURION_PAPER_TRADE=false` | nobody |
| `gtt_reconcile_open` / `_close` | 09:05 / 15:45 | if `CENTURION_PAPER_TRADE=false`: GTTs on **every** CNC holding, orphan deletes account-wide | nobody |
| `margin_monitor` | every 10 min | logs in; read-only | nobody |
| `futures_monitor` | 14:00 | logs in; NIFTY futures orders blocked only by two bugs | nobody |
| `pairs_scanner` | :00/:30 | logs in on signals; orders blocked only by two bugs | nobody |
| `pre_market_scan`, `intraday_rescan`, `eod_scan` | 09:20; 10:30/12:30/14:30; 15:20 | live leg (AutoExecutor) if `CENTURION_PAPER_TRADE=false` | `/ind-stocks/pipeline/latest` (no frontend caller) |
| `nse_engine_executor` | 09:25 | only with three env flags; would duplicate the GitHub session | — |
| `paper_eod_snapshot`, `paper_trade_poll`, `paper_weekly_checkpoint`, `paper_live_reconciliation` | various | price reads on the legacy book | would be second writers to the Neon paper book; skipped today by an owner guard that **fails open** |
| `walk_forward_audit`, `forecast_calibration`, `hmm_refit`, `pead_earnings_feed`, `meta_label_retrain`, `event_calendar`, `strategy_tournament`, `trade_returns_collector` | weekly / monthly / daily | none | legacy R21A Carver and scorer state; `strategy_tournament` reads a file nothing writes |
| `us_pre_market`, `nightly_backup`, `scheduler_heartbeat` | 19:00; 23:00; every 5 min | none | nobody (no restore code; no heartbeat reader) |

## The login path

`_get_scheduler_kite()` returns a cached session (5 h), else
`try_stored_token()` — a single-use request token committed in
`kite_connect/auth/kite_token_store.py`, so it always fails — else
`http_login_kite()`: a headless login with `ZERODHA_PASSWORD` and
`ZERODHA_TOTP_SECRET`, which `deployment/sync_hf_secrets.py` pushes to the
Space (*verified*). No lock, no backoff, no proxy (orders would leave from
the Space's IP, not the registered static IP). With the jobs above, about
two automated logins a weekday when it succeeds, ~284 attempts when it
fails, in bursts of ~5 at :00 and :30.
