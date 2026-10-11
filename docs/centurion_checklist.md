# Centurion health checklist (CK1)

**What this is.** A test-and-validate questionnaire covering:
- strategies;
- data quality and freshness;
- backtests and forward tests;
- pipelines and paper trading;
- infrastructure, dependencies and prerequisites.

The aim is that Centurion stays reliable as markets, data sources and rules
change.

**When to run it:**
- before go-live (D3);
- then quarterly, with GT12's re-score: January, April, July, October;
- whenever a data source, broker rule or dependency changes.

**How to use it.** Each line gives the question, how to check it, and what
counts as a pass. Record a run in the tracker's DONE section as
"CK1 <date>: n of N pass", giving every fail a follow-up ID. The commands
assume `centurion_core/` as the working directory and
`ProjectX/myenv/bin/python`.

## 1. Strategy and research governance

| Question | How to check | Pass |
|---|---|---|
| Is the live config the validated one? | `config/nse_engine_deployed.json` hash vs its approval notes | Hash matches; PBO ≤ 30%, deflated Sharpe ≥ 0.95 |
| Was every trial pre-registered and counted? | `docs/nse_engine_validation_plan.md` sections 5v onward vs the `data/nse_engine/runs*` registries | No run without a dated pre-registration; nothing deleted |
| Is any family over budget or overfit? | `docs/research_protocol.md` section 1; each family's PBO | ≤ 3 configurations per source; family PBO ≤ 30% |
| Is the holdout still sealed? | `data/nse_engine/holdout.lock` | Opened once per family, never re-used |
| Was anything re-tuned after a fail? | Tracker DONE entries vs the plan's results | Fails stay closed; only bug fixes, dated |

## 2. Backtest integrity

| Question | How to check | Pass |
|---|---|---|
| Is it survivorship-free and free of look-ahead? | `python -m pytest tests/test_survivorship.py tests/test_universe_lookahead.py tests/test_signals_causal.py` | All pass |
| Are costs current and stamped? | Run manifests' `cost_model`; the dated rates in `nse_engine/costs.py`, `kite_connect/options/fno_costs.py`, `nse_engine/mcx_costs.py` | Latest model everywhere; Budget 2026 rates in place |
| Is the price data clean? | "unexplained price gap" warnings when the store loads | Every gap reviewed; none new |
| Is there one environment per trial set? | `python -m runners.run_nse_engine runtime-index` | No mixed runtimes within a family |
| Are known model gaps tracked? | Tracker LN-T30 (locked lower circuits), IC2 (cost-model stamping) | Scheduled, with impact sized |

## 3. Walk-forward and forward test

| Question | How to check | Pass |
|---|---|---|
| Does out-of-sample hold up? | Walk-forward 2017–25 (`data/nse_engine/wf_stitched_*.json`) | Excess Sharpe ≥ the deployed config's − 0.05 |
| Is paper tracking the backtest? | Daily email: the G4 verdict | Tracking error ≤ 8%/yr; costs ≤ 1.5× model; MaxDD within the backtest's |
| Has the paper run long enough? | Paper session count | ≥ 60 sessions before D3 and V3 |
| Is promotion rule-based? | `python -m runners.run_nse_engine promote --check` | Only the forward gate promotes; you run the command |

## 4. Data quality

| Question | How to check | Pass |
|---|---|---|
| Any missing sessions? | build-store's suspect-missing-sessions check | None flagged; special sessions present |
| Is the registry continuous after a rebuild? | `python -m runners.run_nse_engine refresh-registry --dry-run` | Fingerprint unchanged, or a refresh planned before recording |
| Are symbols and corporate actions right? | The rename table; dividend and split warnings | Renames resolve; implausible events logged |
| Is fundamentals coverage adequate? | `data/nse_engine/fundamentals/manifest.json`, `coverage` | Most top-500 filings `ok`; missing ones explained |
| Are MCX and F&O sane? | Lot multipliers vs specs; roll days; largest daily moves | Specs match; no jumps at rolls |

## 5. Freshness and pipelines

| Question | How to check | Pass |
|---|---|---|
| Are the research stores current? | `data/nse_engine/refresh_status.json` | `ok: true`, ≤ 7 days old, last session = last trading day |
| Did today's paper run happen? | `gh run list --workflow nse-paper-trading.yml --limit 5` | A success every trading day |
| Are the sources still reachable? | `data/nse_engine/refresh/refresh.log`: NSE archive, NSE API, MCX (curl_cffi) | No persistent 403s or timeouts; endpoints unchanged |
| Are the background jobs alive? | `launchctl list \| grep centurion` | `com.centurion.data-refresh` loaded |

## 6. Paper trading and operations

| Question | How to check | Pass |
|---|---|---|
| Is Kite logged in every day? | The 09:00 email; the dry-run email flags a missing login | No missed sessions |
| Are the dry runs and sell path proven? | Tracker D3 and ST2 | 5 clean dry runs; AMO and GTT sells recorded |
| Is funding understood? | Tracker LN-T11 (Zerodha's AMO answer) | Buys capped at free cash |
| Are the options flows checked? | Tracker O1d log | Both demo runs logged without errors |

## 7. Live readiness and risk

| Question | How to check | Pass |
|---|---|---|
| Kill switch and dead-man's switch? | The kill-switch workflow; a healthchecks.io drill | Both tested end to end |
| Do alerts fire? | Tracker AL1–AL5; Kite postbacks | Rejected or cancelled orders → CRITICAL email within a minute |
| Are the risk limits on? | Drawdown overlay (20 / 30 / 35%), capital ladder | In the deployed config; automatic step-down tested |
| Are the calendars current? | `nse_engine/nse_calendar.py` (LN-T31) | Next year's NSE holidays loaded before January |

## 8. Infrastructure and security

| Question | How to check | Pass |
|---|---|---|
| Are the secrets safe? (the repo is public) | GitHub secret-scanning alerts; `git grep` for keys and tokens | 0 alerts; credentials only in GitHub / HF secrets |
| Are the schedulers punctual? | cron-job.org / Hugging Face dispatch vs GitHub cron | Runs start inside their window |
| Are the cloud services healthy? | HF Space, Neon DB, Actions cache, Kaggle (`python -m cloud.kaggle_local check`) | All reachable and within quota |
| Is the research backed up? | `data/nse_engine/runs*`, stores and archives: only on this Mac? | A copy exists off the Mac |
| Enough disk? | `df -h` | ≥ 50 GB free |

## 9. Dependencies and environments

| Question | How to check | Pass |
|---|---|---|
| Does CI pass with core dependencies only? | A fresh venv from `requirements-core.txt`, then `python -m pytest tests -q -m "not slow"` | Passes, matching CI |
| Do local results match? | myenv vs the core pins; Kaggle runs with `--pin` | Same pandas / numpy / pyarrow for compared runs |
| Is the runner image pinned? | Workflows' `runs-on` | `ubuntu-24.04` pinned (`ubuntu-latest` moves to 26 from 19 Oct 2026) |
| Are the fragile libraries working? | kiteconnect, curl_cffi, lxml, the kaggle CLI | Import and work; versions pinned |

## 10. External prerequisites and regulation

| Question | How to check | Pass |
|---|---|---|
| Is the broker side ready? | Kite Connect subscription; the daily TOTP login (manual by rule) | Active; no automated login |
| Are the rule changes encoded? | STT, CTT and stamp schedules; tax (STCG 20%, MCX profits as business income) | Rates current; the target is pre-tax |
| Are the segments enabled? | Zerodha: equity and F&O now, commodity before GT5 goes live | Enabled when needed |
| Are regulatory blocks respected? | The RBI currency-derivatives rule; F&O margin and lot changes | Currency stays closed; lot sizes current |

## 11. Golden-target progress (quarterly)

| Question | How to check | Pass |
|---|---|---|
| Is it on track? | Scorecard (SC1b); live vs backtest Sharpe | Live ≥ 0.85× backtest (core) |
| Are the new streams progressing? | GT4 / GT5 / GT7 / MV1 verdicts; GT6 combination | Each pre-registered stream has a recorded pass or fail |
| Is drawdown inside expectations? | Live MaxDD vs the simulated 95th percentile | Inside; a breach triggers a review |
