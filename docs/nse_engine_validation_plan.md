# NSE Engine — Validation and Paper Trading Plan

How a configuration of the NSE engine (`docs/nse_engine.md`) moves from
backtest to walk-forward, holdout, paper trading and live capital. Each stage
has fixed pass/fail gates, set before the results are seen.

## 1. Why this replaces the 13 Apr 2026 snapshot

`my_todos.txt` lines 1–338 describe the legacy R21A system. An audit found
those numbers could not be reproduced or trusted:

| Snapshot claim | What the audit found |
|---|---|
| GM2: Sharpe 1.571, CAGR 61%, MaxDD 31.6% | Not tied to any commit or config; train/test rows copied from R21A base |
| PBO = 0.0% "likely real" | Computed on shares of one P&L, so ≈ 0 for any profitable backtest |
| DSR p = 0.0000 "fat tails" | Annual Sharpe mixed with daily sample count; recomputed at N = 1,550: 0.67 (fails) |
| Test beats train → generalises | 2020–25 used for selection and inflated by survivorship (today's NIFTY 100) |
| 1-day lag costs 0.127 Sharpe | Lag test read a 5-day refresh grid |
| Leverage 2×, fills at the close | Cash (CNC) account cannot hold > 1×; same-close fills |

How much the survivor universe alone was worth (measured 16 Sep 2026 on the
engine's own adjusted prices, 2013–2025): equal-weight buy-and-hold of
*today's* NIFTY 50 + Next 50 names returns 23.1% CAGR at excess Sharpe 0.92,
against 14.4% and 0.52 for the NIFTY 50 TRI that was actually investable.
That 9 points a year is passive and unrepeatable, and R21A base (30.4% CAGR at
up to 2× leverage, Sharpe 1.127 with rf = 0 ≈ 0.6 excess) sits barely above
it. On a like-for-like basis the engine below already scores higher (full
period excess Sharpe 1.22 ≈ 1.7 at rf = 0) — with lower CAGR only because it
carries no leverage.

The engine now used for every number below is survivorship-free (NSE bhavcopy
archives: 4,232 symbols with history, 1,855 of them no longer trading),
dividend- and split-adjusted, fills at the next open with impact and statutory
costs, holds gross ≤ 1, and records every run for PBO and DSR.

## 2. Target metrics

Sharpe is excess over a 6.5% risk-free rate (add about 0.5 for the rf = 0
convention used in the snapshot).

| Metric | Minimum to proceed | Aim |
|---|---|---|
| Excess Sharpe (walk-forward OOS) | ≥ 0.8 | ≥ 1.1 |
| CAGR (OOS) | ≥ 15% | 20–25% |
| MaxDD (OOS) | ≤ 25% | ≤ 18% |
| Calmar (OOS) | ≥ 0.8 | ≥ 1.2 |
| PBO (all same-window configurations) | < 30% | < 20% |
| Deflated Sharpe (N = every configuration) | ≥ 0.95 | ≥ 0.97 |
| Benchmark gate | beat EW hold and naive momentum by ≥ 0.3 Sharpe | also beat NIFTY 50 TRI by ≥ 0.3 |

55% CAGR with Sharpe 1.8 is not reachable on a cash account: the honest
walk-forward result is about 20% CAGR, and doubling it needs ~2× margin
leverage (≈ 14.6%/yr interest) with roughly double the drawdown.

## 3. Stage A — Backtest (2013-01-01 → 2025-12-31)

Done on corrected data (data hash `172913b826f0ffa4`):

- 36 configurations: a 32-point grid (low-vol group on/off, stops 3×/6×ATR,
  rebalance 5/21 days, neutral regime scale 0.6/1.0, 20/30 positions) plus
  4 ablations.
- Best full-period rows: excess Sharpe ≈ 1.22, CAGR 21–24%, MaxDD 17–25%,
  Calmar ≈ 1.0–1.25. Every top-8 row has no low-vol group and 6×ATR stops.

Rules: every run is recorded; never read data after 2025-12-31 in this stage.

## 4. Stage B — Walk-forward

Protocol: anchored, 4-year minimum train, 12-month test folds, OOS 2017–2025,
same 32-point grid, selection by train-window excess Sharpe only.

**Result (15 Sep 2026, corrected data):** every fold chose no low-vol group,
6×ATR stops, rebalance every 5 days, neutral scale 1.0, 20 positions
(config `679cbd0c`). Stitched OOS 2017–2025: CAGR 23.6%, vol 12.5%,
excess Sharpe 1.27, MaxDD 23.6%, Calmar ≈ 1.0; OOS/IS Sharpe 1.17.
OOS Sharpe by year: 2017 +2.73, 2018 −1.92, 2019 −0.01, 2020 +1.41,
2021 +3.18, 2022 −0.78, 2023 +1.96, 2024 +1.09, 2025 +2.16 (3 of 9 ≤ 0).

**Re-run on Kaggle (16 Sep 2026, Linux/x86_64, same grid, same window,
anchor 2011):** all 9 folds again chose `679cbd0c`. Stitched OOS 2017–2025:
CAGR 23.7%, vol 12.6%, excess Sharpe 1.24, MaxDD 22.3%, OOS/IS 1.18; OOS
Sharpe by year 2017 +2.80, 2018 −2.16, 2019 +0.01, 2020 +1.45, 2021 +3.15,
2022 −0.92, 2023 +2.00, 2024 +1.08, 2025 +2.04 (2 of 9 ≤ 0). 297 backtests in
two sessions of 19 and 28 minutes. The two platforms differ fold by fold at
the second decimal (results do not reproduce across machines — see
`docs/kaggle_research.md`) and agree on every choice and every gate. Files:
`data/nse_engine/wf_stitched_kaggle.json`, `data/nse_engine/wf_oos_returns_kaggle.csv`.

Validation of its full-period run (`20260914T183003557572Z_679cbd0c`,
CAGR 24.1%, excess Sharpe 1.22, MaxDD 25.1%): PBO 23.9% over 36
configurations (likely real), deflated Sharpe 0.994 (N = 36), benchmark gate
passed — beats naive momentum by 0.53, EW hold by 0.96, NIFTY 50 TRI by 0.71.
Re-validated after importing the 297 Kaggle runs (registry 1,150 runs): the
same 23.9% / 0.994 / pass, because PBO and DSR are computed over the 36
configurations that share the full 2013–2025 window; train- and test-window
runs are counted as trials but cannot enter a same-window returns matrix.

```
python -m runners.run_nse_engine walk-forward --start 2013-01-01 --end 2025-12-31 \
    --grid '{...32-point grid...}'
python -m runners.run_nse_engine validate --run-id <full-period run of the last fold's choice>
```

Gates (Section 2) plus: at most 3 of 9 OOS years negative; OOS / IS Sharpe
ratio ≥ 0.5. The configuration chosen by the most recent fold is the
candidate.

## 5. Stage C — Holdout (2026-01-01 → last complete session)

One evaluation only (enforced by `data/nse_engine/holdout.lock`).

```
python -m runners.run_nse_engine holdout --config <candidate config.json> \
    --start 2026-01-01 --end <last session>
python -m runners.run_nse_engine promote --run-id <candidate run> --paper-start <date>
```

**Result (run once, 2026-01-01 → 2026-09-11, 172 sessions):** +18.0%
(27.4% annualised), excess Sharpe 1.04, vol 18.7%, MaxDD 13.0%, turnover
8.8×/yr. Same window: NIFTY 50 TRI −9.6%, EW universe hold +2.5%, naive
momentum +46.5% (excess Sharpe 2.01). Passed both holdout gates; promoted to
`config/nse_engine_deployed.json` (paper start 2026-09-16, data anchor 2011-01-01).
Note that naive momentum beat the strategy in 2026.

`promote` refuses unless PBO < 30%, DSR ≥ 0.95, the benchmark gate passed,
holdout excess Sharpe > 0 and holdout MaxDD ≤ 1.5× the backtest MaxDD.
Eight months cannot confirm a Sharpe (standard error ≈ 1.2); the holdout
exists to catch a broken strategy, not to tune one. If it fails, do not
re-tune on 2026 data: return to Stage B with a new hypothesis, and treat
paper trading as the next clean test.

## 6. Stage D — Paper trading (60–90 trading days)

**Data anchor rule.** Rebalance-day counting and the expanding forecast
normalisers start at the first loaded row. The same config over 2026 returned
+18.0% loaded from 2011 but +12.8% loaded from 2024 (5.9%/yr tracking error),
so the deployment pins `data_anchor_date` (2011-01-01) and the executor, the
shift reference and the Actions bootstrap all load from it.

**Correction (17 Sep 2026).** The local store used for every validation run
starts at 2012-01-02, so "loaded from 2011" there actually meant 2012-01-02:
the walk-forwards, the promoted full-period run and the holdout were all
anchored at 2012-01-02. The Actions store was bootstrapped from 2011-01-01 and
does hold 2011, so the paper book is anchored a year earlier than what was
validated. Measured with the 2006-2026 store on the same machine, 2026 YTD:
validated anchor +15.7% / excess Sharpe 0.88 / MaxDD 13.0% / 329 trades;
paper anchor +15.5% / 0.84 / 16.3% / 339. The target portfolio on 16 Sep is
99.9% identical, but the paths drift. Aligning `data_anchor_date` to
2012-01-02 restores the validated path (tracker U8). Making the engine
anchor-independent is the first research item (Section 8).

Setup:
1. `promote` writes `config/nse_engine_deployed.json` (status `approved`); commit it.
2. GitHub: repository variable `CENTURION_NSE_ENGINE=true`; secrets as for the
   legacy cron. First run with the `full_bootstrap` dispatch input: syncing from
   the 2011 anchor is ≈ 3,650 sessions × 4 files at 2 req/s (≈ 2 h), so it may
   need a second dispatch — the archive cache keeps progress.
3. Daily job (19:30 IST): sync NSE archives → rebuild current-year store →
   same-period shift reference → decide after close → fill pending orders at
   the next session's open → GTT-style stops at min(open, stop) → snapshot.
4. The job runs only while the paper switch in Neon (`paper_trading_state`,
   toggled from the trade-monitor page or `POST /api/paper-trading
   {"action":"start","weeks":20}`) is active and unexpired — the page's
   default is 4 weeks, which is why the first engine runs on 16 Sep 2026
   skipped with "Paper trading is NOT active". 90 sessions need ~20 weeks.
5. The book is scoped by an `epoch` in `paper_cloud_state`; dispatch the
   workflow with `new_book=true` to start a fresh book at
   `CENTURION_PAPER_INITIAL_CAPITAL` (older rows stay, filtered out). The
   engine marks `book_owner=nse_engine`, which switches off the legacy paper
   jobs in the Hugging Face scheduler that would otherwise overwrite the
   day's snapshot with a stale copy.

Where to watch: https://centurion-core-fe.vercel.app/ind-stocks/trade-monitor —
active positions and orders pending for the next open, closed trades with
exit reason, the metrics grid (Sharpe/Sortino/Calmar/MaxDD from the daily
equity curve), daily P&L bars, the equity curve, weekly checkpoints, and a
per-day drill-down with that session's signals and fills. All of it reads the
Neon book directly, so it updates as soon as the day's job finishes.

Daily monitoring (automatic):

| Check | Alert / action |
|---|---|
| Distribution shift (≥ 30 live days) | drifting → size 0.75×; regime_break → 0.5× and alert |
| Tracking error vs same-period backtest | > 8%/yr → drifting |
| Mean daily gap vs backtest | < −3 bp/day → drifting |
| Fill price vs model open + impact | investigate if median shortfall > 2× model |
| Drawdown | > 1.5× backtest MaxDD for that horizon → halt new entries, review |

Pass criteria after 60 trading days (extend to 90 if borderline):
- tracking error ≤ 8%/yr and mean daily gap ≥ −3 bp/day;
- no `regime_break` in the last 20 sessions;
- realised costs within 1.5× the model;
- paper drawdown within the backtest's worst drawdown of the same length.

A 60-day paper Sharpe says little about skill (standard error ≈ 2); the gates
test whether live behaves like the backtest, which is what can be measured.

## 7. Stage E — Live capital (after paper passes)

Real orders need `CENTURION_PAPER_TRADE=false` and `CENTURION_NSE_ENGINE_LIVE=true`
and an approved deployment.

| Month | Capital | Condition to continue |
|---|---|---|
| 1 | 20% | Stage D gates still hold live |
| 2 | 40% | tracking error ≤ 8%, no regime_break |
| 3 | 70% | cumulative drawdown within backtest expectations |
| 4+ | 100% | quarterly re-validation passes |

Kill criteria at any time: drawdown > 1.5× backtest MaxDD, two consecutive
regime_break verdicts, or realised costs > 2× model for a month.

## 8. Stage F — Research loop (runs in parallel with paper trading)

Hypotheses, each tested only through Stage B on data up to 2025-12-31, with
every configuration recorded (so PBO/DSR count them):
0. Anchor independence — **built 17 Sep 2026** (`docs/nse_engine.md`, design
   rules): calendar-anchored rebalance / universe / FDM schedules, rolling
   normalisers, finite-memory EWMs, trailing history counts, load filter off.
   Verified bit-identical across a 2011 and a 2014-07 load of the same 2026
   window (329 trades both). Opt-in fields; the deployed configuration is
   untouched and keeps its hash. On the spent 2026 holdout the anchor-
   independent variant of the deployed settings shows excess Sharpe 0.55 /
   +11.3% against 1.04 / +18.0% — one 8-month window, not a verdict; its
   Stage-B walk-forward (same 32-point grid, Kaggle) decides. Early folds
   cannot be fully warmed (the store starts in 2012), which anchors them at
   the store's first row exactly as the legacy runs are; it is the paper /
   live path, loading from any date, that this makes reproducible.
   **Walk-forward result (Kaggle v8, 17 Sep 2026, 297 backtests):** every
   fold chose the deployed settings again (no low-vol, 6×ATR, rebalance 5,
   neutral 1.0, 20 positions). Stitched OOS 2017–2025: excess Sharpe 1.15,
   CAGR 22.8%, MaxDD 21.1%, OOS/IS 1.15, 2 of 9 years ≤ 0 (2018 −1.87,
   2022 −0.81) — against 1.24 / 23.7% / 22.3% / 1.18 for the legacy
   behaviour on the same platform. So load-independence costs about 0.09
   Sharpe and 1 point of CAGR and returns 1 point of drawdown: within the
   walk-forward's own noise (standard error ≈ 0.35 on nine years), and the
   price of a live path whose behaviour does not depend on when its data
   starts. It is the base for the delivery and vol-target tests below; it
   does not replace the deployed configuration until Phase 1 is complete and
   a candidate has paper-traded beside it. Cost: 25 min per fold on Kaggle
   against 5 for the legacy path (the finite-memory EWMs).
1. Downtrend defence for the negative OOS years (2018, 2019, 2022): core
   volatility targeting, stronger breadth/trend risk-off, absolute-momentum filter.
   **Tested 16–17 Sep 2026 (Kaggle, 48-point grid over the existing regime
   gates: NIFTY MA 100/150/200, confirm 3/10 days, breadth risk-off 0.35/0.45,
   neutral scale 0.6/1.0, VIX elevated 20/25; base = deployed `679cbd0c`).**
   Stitched OOS 2017–2025: excess Sharpe 1.25, CAGR 23.5%, MaxDD 21.8%,
   OOS/IS 1.09 — against 1.24 / 23.7% / 22.3% / 1.18 for the deployed
   configuration on the same platform. 2018 unchanged (−2.16), 2022 slightly
   worse (−1.06 vs −0.92), and the folds did not agree on a setting (MA 200
   then 100, breadth 0.35↔0.45, neutral 1.0 then 0.6, VIX 20↔25). Tuning
   these gates is closed: the negative years are not a regime-parameter
   problem in this family. 441 backtests recorded. What remains under this
   item is a *different* mechanism — portfolio vol targeting or an
   absolute-momentum filter — not more grid over the same gates.
   **Portfolio vol targeting tested 17 Sep 2026 (Kaggle d, anchor-independent
   base, target off / 12% / 15%): closed.** All 9 folds chose it off, and it
   lowered the train-window Sharpe in every fold, by 0.02–0.11. The book
   already runs at ~13% vol, so the target binds mainly in the high-vol
   stretches that precede recoveries, cutting exposure at the wrong time.
2. NSE data signals: delivery % confirmation; earnings dates from NSE board
   meeting records (if obtainable); FII/DII flows (data availability first).
3. Turnover reduction beyond monthly rebalancing.
4. R21A's independently viable rules as new signal groups — breakout,
   acceleration, Ehlers DSP, Carver value — one group per trial with fixed
   hand-set weights, so FDM does the combining and the registry counts every
   attempt. R21A's own incremental tests found each hurt v27, but that was on
   the survivor universe with optimised weights; the question is open here.
   Expectation: a Sharpe change of ±0.1, not a new regime of returns.

Not on the list, on purpose: re-optimising signal weights (R21A's 247% data-
mining bias estimate came from exactly that), leverage (MTF at ≈ 14.6%/yr
roughly doubles drawdown for the CAGR it adds), and anything tuned on 2026.

All research folds run on Kaggle (`docs/kaggle_research.md`): results do not
reproduce across platforms, so a walk-forward is compared only with
walk-forwards from the same place.

A research winner is never deployed on its walk-forward alone: it paper
trades beside the deployed configuration for at least 60 sessions first,
because the 2026 holdout will already have been used.

## 9. Command reference

| Step | Command |
|---|---|
| Sync data | `python -m nse_engine.data.archive --start <date> --end <date> --root data/nse_engine/archive` |
| Build store | `python -m runners.run_nse_engine build-store` |
| Backtest | `python -m runners.run_nse_engine backtest --set key=value --tag <tag>` |
| Walk-forward | `python -m runners.run_nse_engine walk-forward --grid '<json>'` |
| Walk-forward on Kaggle (4 cores, resumable) | `python -m cloud.kaggle_local run --task walk-forward --args "..."` — see `docs/kaggle_research.md` |
| Validate | `python -m runners.run_nse_engine validate --run-id <id>` |
| Holdout | `python -m runners.run_nse_engine holdout --config <json> --data-start 2011-01-01 --start <date> --end <date>` |
| Promote | `python -m runners.run_nse_engine promote --run-id <id> --paper-start <date> --data-anchor 2011-01-01` |
| Shift reference | `python -m runners.run_nse_engine shift-reference --run-id <id> --start <paper start> --data-start 2011-01-01` |
| Inspect deployment | `python -m nse_engine.deployment show` |
