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

**CAGR convention (unified 27 Sep 2026, tracker B2).** CAGR compounds over
calendar years, `days / 365.25` between the first and last session — the
engine's run manifests and the paper book always did this; the validation
summaries (walk-forward, holdout, `validate`, lag tests) used
`sessions / 252` until 27 Sep 2026. NSE trades about 248 sessions a year, so
the old figures read roughly 0.4 points high. Figures quoted below from
before that date are on the old convention; on calendar years they are:

| Figure | Quoted (sessions / 252) | Calendar years |
|---|---|---|
| Deployed `679cbd0c`, 2013–2025 | 24.5% (24.1% in its manifest) | 24.1% |
| Honest baseline `bd79bf28` (B1), 2013–2025 | 23.2% | 22.8% |
| Walk-forward OOS 2017–2025 (Kaggle) | 23.7% | 23.3% |
| Holdout 2026-01-01 → 09-11 | 27.4% annualised | 26.9% annualised |

Sharpe, volatility, drawdown and every gate are unaffected. The stitched
JSON files under `data/nse_engine/` keep the numbers they were written with;
anything produced from now on is on calendar years and carries
`cagr_convention: "calendar"`.

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

**Superseded 27 Sep 2026 (§5d, tracker K5).** Every walk-forward above ran
without a NIFTY 50 trend reading for 2014–2016. On complete data the same
grid picks the deployed settings in 1 of 9 folds, and the last five folds
(2021–2025) pick a neutral scale of 0.6; stitched OOS 2017–2025 is Sharpe
1.19, CAGR 22.0%, MaxDD −22.5% (local, B1 base).

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
`config/nse_engine_deployed.json` (paper start 2026-09-16, data anchor 2012-01-02 —
see the correction below).
Note that naive momentum beat the strategy in 2026.

`promote` refuses unless PBO < 30%, DSR ≥ 0.95, the benchmark gate passed,
holdout excess Sharpe > 0 and holdout MaxDD ≤ 1.5× the backtest MaxDD.
Eight months cannot confirm a Sharpe (standard error ≈ 1.2); the holdout
exists to catch a broken strategy, not to tune one. If it fails, do not
re-tune on 2026 data: return to Stage B with a new hypothesis, and treat
paper trading as the next clean test.

## 5b. Data integrity — the back-adjustment look-ahead (fixed 18 Sep 2026)

Prices in the panel are back-adjusted from the **end** of the loaded window, so
a 2013 close moves when a 2024 split, bonus or dividend happens. Returns are
unaffected (a truncated load reproduces them to 2.4e-07), but the universe's
`min_price_inr` test read those adjusted levels, so **which names were eligible
in 2013 depended on what happened later**: 4,462 price-filter cells passed only
because of later actions, and ~12 of 255 universe names differed per day
between a load ending in 2019 and one ending in 2025.

`UniverseConfig.price_filter_unadjusted` (hash-neutral, default False =
legacy) makes the test read `MarketData.close_unadj`, the close as printed
that day. With it on, universe membership is identical whatever the load ends
(0.00 names differ per day); the residual 1.2 bp/day end-date effect is
integer share rounding at adjusted price levels, and it no longer favours the
backtest.

Measured on the deployed configuration, same data, 2013-01-01..2025-12-31:

| Window | Legacy Sharpe / CAGR | As-printed filter |
|---|---|---|
| 2013-2016 | 0.98 / 22.3% | **0.76 / 18.3%** |
| 2017-2020 | 0.89 / 17.9% | 0.87 / 17.4% |
| 2021-2025 | 1.69 / 32.0% | 1.69 / 32.2% |
| Full | 1.22 / 24.5% | **1.14 / 23.2%** |

The bias grows with distance into the past, as later corporate actions
accumulate. The walk-forward OOS window (2017-2025) is materially unaffected,
so the headline OOS Sharpe 1.24 stands; the full-sample backtest and anything
measured on 2013-2016 was flattered.

Live trading was never affected: on the day itself an adjusted close equals
the printed one, so the paper book's eligibility has always been correct.

**Walk-forward re-run (B1b, 18 Sep 2026, both arms on one machine, 32-point
grid, 9 folds, 594 backtests):** OOS excess Sharpe 1.259 with the fix against
1.267 without (difference -0.008, 90% CI -0.08 to +0.07), CAGR 23.8% against
24.1%, MaxDD -23.7% either way. Two of nine folds chose differently (2022 and
2024 took `regime.scale_neutral` 0.6 instead of 1.0). The telling change is
in-sample: mean IS Sharpe fell from 1.079 to 0.938 and the OOS/IS ratio rose
from 1.17 to 1.34. The look-ahead flattered the *training* windows, which is
where the selection happens, not the out-of-sample record.

**Rule from here:** every new validation run sets
`universe.price_filter_unadjusted=true`. It changes the configuration hash, so
the deployed `679cbd0c` keeps its identity and its recorded trials; adopting
the fix means promoting a new configuration through the usual gates.

## 5c. Data fingerprint changes (registry continuity, fixed 27 Sep 2026)

Every run's manifest carries the `data_hash` of the panel it was computed
on, and `validate` builds the PBO/DSR matrix only from runs that share it.
A store rebuild can change the hash without changing a single return: the
23 Sep 2026 rebuild renamed symbols to their current tickers (HEG → HEGAM),
so a fresh 2013–2025 load fingerprints `5485474397ef3a5f` while all
recorded runs carry `172913b826f0ffa4`. Left alone, the next recorded run
would meet no prior configurations — no PBO, deflated Sharpe at N = 1.

`refresh-registry` re-runs every same-window configuration on the current
store, records each run with `refresh_of` = the run it reproduces, checks
that the daily returns agree to 1e-9, and compares PBO and every
configuration's deflated Sharpe before and after. The config hashes are
unchanged, so for `returns_matrix` the re-runs are duplicates of the old
ones (dedupe keeps the latest), not new trials; the raw run count grows,
the configuration count does not. Run it with `--dry-run` first, and after
every store rebuild that changes the fingerprint. Walk-forward fold runs
stay on the old hash: they count as trials but never share a window with
the full-period matrix.

`build-store` now ends with this check: it loads the validation window,
compares the fingerprint with the one the registry was last extended on, and
when they differ prints the dry-run plan and the two commands above. Nothing
is recorded until `refresh-registry` has run. `python -m nse_engine.data.store`
prints the same reminder. `--skip-registry-check` turns the check off.

## 5d. NIFTY 50 gaps: the regime gate's trend leg (found 27 Sep 2026)

NSE's daily index files lack NIFTY 50 on 47 sessions in the main store's
range: 2 Jan–20 Feb 2012 and 12 scattered days from 2013-10-09 to
2016-06-20. The regime gate's trend test needs a 200-session mean with no
gaps, so each missing day left the trend undefined for the next 200
sessions: **for all of 2014, 2015 and 2016, 23% of 2013 and 27% of 2017,
every validated backtest ran the regime gate without its trend leg** (the
state then defaults to neutral). The live book is unaffected: its 200-day
window has had no gaps since mid-2017.

Measured on the honest baseline `bd79bf28` with the gaps filled in memory
(unrecorded run):

| | Sharpe | CAGR | MaxDD | 2014 / 2015 / 2016 regime states |
|---|---|---|---|---|
| As validated | 1.140 | 22.75% | −24.7% | 100% neutral each year |
| Gaps filled | 1.144 | 22.79% | −24.7% | mostly risk-on; 2016: 14% risk-off |

The deployed configuration barely moves because it holds full exposure in
both risk-on and neutral. The grid configurations with a neutral scale of
0.6 do move: they were held at 60% through the 2014 bull run, so the
walk-forward's preference for a neutral scale of 1.0 was partly an artefact
of the gap. The fix (tracker K5) fills the main store from the same cache,
puts `index_close` in the data fingerprint so blind and complete runs can
never share a hash, refreshes the registry and re-runs the walk-forward.
The extended store (`store_ext2006`, K4) already carries the complete series:
with it the gate turns risk-off on 23 Jan 2008, two weeks after NIFTY's
peak, and stays risk-off for 90% of 2008; without it, 43%, from March.

**Fixed 27 Sep 2026 (K5).** The main store has the same cache (45 sessions
from Yahoo `^NSEI`, one Saturday session carried forward; 2 Jan 2012 has
no source and only delays the first 200-day mean by a session), and
`MarketData.compute_hash()` now covers the index closes, so the fingerprint
moved from `5485474397ef3a5f` to `6c94f4f4cc54041b`. `refresh-registry`
re-ran all 46 same-window configurations; none reproduced, as expected.

| Configuration, 2013–2025 | Sharpe before → after | CAGR after | MaxDD after |
|---|---|---|---|
| Baseline `bd79bf28` (B1) | 1.140 → 1.144 | 22.79% | −24.7% |
| Deployed `679cbd0c` (legacy filter) | 1.219 → 1.187 | 23.49% | −25.0% |
| E1 refill `942760af` | 1.211 → 1.195 | 24.39% (still fails ≥ 25%) | −25.3% |
| E2 B `893e041f` (the live rule) | 1.156 → 1.155 | 22.86% (still passes) | −23.3% |
| E2 A `9824859d` | 1.147 → 1.086 | 21.39% (still fails) | −33.8% |

Neutral-0.6 configurations gained 0.019 Sharpe on average, full-exposure ones
lost 0.011. PBO over the 46 configurations: 41.0% → 45.4%.

The walk-forward, re-run on this machine (same 32-point grid, anchored,
B1 base, 297 backtests, 29 minutes):

| | 18 Sep (gaps) | 27 Sep (complete) |
|---|---|---|
| OOS Sharpe / mean IS | 1.26 / 0.94 | 1.19 / 0.98 |
| OOS CAGR (calendar) / MaxDD | ~23.4% / −23.7% | 22.0% / −22.5% |
| Folds choosing the deployed settings | 7 of 9 | 1 of 9 (2018) |
| Other choices | neutral 0.6 in 2022, 2024 | monthly rebalance 2017, 2019, 2020; neutral 0.6 in every fold 2021–2025 |

OOS years: 2017 +40.3%, 2018 −11.0%, 2019 +5.5%, 2020 +23.6%, 2021 +81.5%,
2022 −5.0%, 2023 +28.9%, 2024 +22.7%, 2025 +34.2% (3 of 9 at or below the
risk-free rate). Files: `data/nse_engine/wf_stitched_k5.json`,
`data/nse_engine/wf_oos_returns_k5.csv`.

By the protocol (§4) the most recent fold's choice is the candidate: the B1
baseline with `regime.scale_neutral = 0.6` (`2d64ba4c`). Full period
2013–2025: Sharpe 1.189, CAGR 22.27%, MaxDD −23.4%, Calmar 0.953, average
gross 0.81; validate: PBO 49.4% over 47 configurations, deflated Sharpe
0.989, benchmark gate passed. Whether it takes the second paper slot is
decision U21 in the tracker; the deployed configuration is unchanged.

## 6. Stage D — Paper trading (60–90 trading days)

**Data anchor rule.** Rebalance-day counting and the expanding forecast
normalisers start at the first loaded row. The same config over 2026 returned
+18.0% loaded from 2011 but +12.8% loaded from 2024 (5.9%/yr tracking error),
so the deployment pins `data_anchor_date` (2012-01-02) and the executor, the
shift reference and the Actions bootstrap all load from it.

**Correction (17 Sep 2026).** The local store used for every validation run
starts at 2012-01-02, so "loaded from 2011" there actually meant 2012-01-02:
the walk-forwards, the promoted full-period run and the holdout were all
anchored at 2012-01-02. The Actions store was bootstrapped from 2011-01-01 and
does hold 2011, so the paper book is anchored a year earlier than what was
validated. Measured with the 2006-2026 store on the same machine, 2026 YTD:
validated anchor +15.7% / excess Sharpe 0.88 / MaxDD 13.0% / 329 trades;
paper anchor +15.5% / 0.84 / 16.3% / 339. The target portfolio on 16 Sep is
99.9% identical, but the paths drift. `data_anchor_date` is now 2012-01-02,
which restores the validated path (tracker U8). 2011 had 247 sessions, so the
switch moves the 5-day rebalance by 2 sessions (Tuesday -> Wednesday; first
affected rebalance 23 Sep 2026 instead of 22 Sep) and the 21-day universe
refresh by 16. No book reset: the next rebalance trades toward the validated
target. Record the anchor as the first session actually loaded, not the
requested load start. Making the engine
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

**Second slot (D1, 28 Sep 2026).** The candidate from §5d (`2d64ba4c`, neutral
scale 0.6) paper-trades beside the deployed book from the first session after
the merge, from ₹35 lakh, in its own Neon schema, with the same drawdown rule,
reports and gates. Under the forward gate (U19) it replaces the deployed
configuration only after at least 60 sessions, a G4 PASS and a walk-forward
OOS Sharpe within 0.05 of base (V3 automates the check). The trial is
informative from the first day: the regime read neutral on 74% of 2026's
sessions and on all of the last 20, and in neutral the candidate holds 60%
of the core book where the deployed book holds 100%.

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
| (how, since 27 Sep 2026 — G2) | the verdict is saved in Neon (`paper_cloud_state` key `distribution_shift_state`) and restored before the next plan, so the size multiplier actually applies on GitHub Actions, whose disk is discarded after each run; reality-gap and regime-break alerts are emailed (`NotificationManager.send_alert`, missing before, so every alert had been dropped); the daily email has a "Drift check" line ("waiting: n of 30" until the check can run). |
| Tracking error vs same-period backtest | > 8%/yr → drifting |
| Mean daily gap vs backtest | < −3 bp/day → drifting |
| Fill price vs model open + impact | investigate if median shortfall > 2× model |
| Drawdown | the deployed drawdown rule (section 8, item 6): > 20% from the episode peak → no new entries or adds; > 30% → core exposure halved; > 35% → core to cash / metals; re-arm on a 60-session equity high. Automatic, replayed from the book's snapshots every session, state in the daily email and on the monitor. Anything beyond that is the kill criterion below. |

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

Before month 1: run the executor's dry-run mode (`EngineExecutor(kite=...,
dry_run=True)`) after every session for at least a week and compare its
orders with the paper book's queued orders for the same day. The live path
was exercised against a fake broker (L3, 27 Sep 2026), which also
found that the order service's market-hours guard would have refused every
end-of-day order; live orders now go as after-market orders (`amo`) when the
market is closed. What live still lacks is a session driver: something that
reads the previous session's order outcomes, snapshots the live book to Neon
(the drawdown rule and the monitor read from there) and then plans and
executes - the paper book has `run_paper_session`, the live book does not yet
(tracker L5).

| Month | Capital | Condition to continue |
|---|---|---|
| 1 | 20% | Stage D gates still hold live |
| 2 | 40% | tracking error ≤ 8%, no regime_break |
| 3 | 70% | cumulative drawdown within backtest expectations |
| 4+ | 100% | quarterly re-validation passes |

Kill criteria at any time (a human decision, above the automatic rule):
drawdown > 1.5× backtest MaxDD, two consecutive
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
5. **Price-based strategies from awesome-systematic-trading — tested 17 Sep
   2026, all rejected.** Of the repository's 61 strategy files, 49 need data
   the store does not have (fundamentals, earnings dates, futures, options,
   FX, crypto, short interest) or short selling; the rest were screened on
   the deployed universe, 2013–2025, with pass rules fixed before running
   (incremental rank IC over the deployed forecast, t ≥ 2.5, positive in
   2013–19 and 2020–25, positive among the names the book buys; t ≥ 3.0 for
   the survivor-biased sector-map rules). Residual momentum, 52-week-high
   proximity and low beta passed; consistent momentum, momentum × volatility,
   12-month seasonality (lag 12 and 1–5-year average), short-term reversal,
   industry momentum and industry 52-week high did not. Turn-of-the-month and
   payday effects on NIFTY 50 were not significant (t = 1.20, 0.42).
   The three survivors became signal groups (`residual_momentum`,
   `near_high`, `low_beta`; hash-neutral) and were backtested at fixed weights
   (0.2 and 1/3 beside fast/slow trend), six recorded runs. None beat the
   deployed config: Sharpe change −0.04/−0.09, −0.00/−0.12 and −0.11/−0.25,
   CAGR lower in all six. The rule required +0.10, so none went to
   walk-forward. The information is real (it predicts next-month returns) but
   it overlaps the trend forecast and dilutes it in a 20-name book.

6. **Drawdown rule — tested 27 Sep 2026 (E2), adopted at 20/30/35%.**
   Exposure control from the book's own equity (`nse_engine.drawdown`,
   section "Design rules" of `docs/nse_engine.md`), pre-registered on the
   honest baseline `bd79bf28`, 2013–2025, two recorded runs, pass rule
   "Calmar above the baseline's 0.92 and CAGR ≥ 22%":

   | Run | Sharpe | CAGR | MaxDD | Calmar | halt sessions | verdict |
   |---|---|---|---|---|---|---|
   | Baseline, no rule | 1.140 | 22.75% | −24.69% | 0.922 | – | – |
   | A: halt 15% / half 25% / risk-off 30%, re-arm 60 | 1.147 | 22.22% | −25.90% | 0.858 | 10.8% | FAIL |
   | B: halt 20% / half 30% / risk-off 35%, re-arm 60 | 1.156 | 22.85% | −23.21% | 0.985 | 5.8% | PASS |

   A halted four times and whipsawed through 2015 (halt in May, re-arm in
   July, halt again in September), which deepened the 2015–16 episode to
   −25.9%; B halted twice (January 2016, December 2018) and improved every
   sub-period: 2013–16 Calmar 0.78 → 0.80, 2017–25 1.01 → 1.08, 2021–25
   unchanged. Only the `halt` leg has evidence: no drawdown in 2013–2025
   reached 30%, so `half` and `risk_off` are untested capital protection
   until the 2008 walk-forward (R4). B's thresholds are the live rule (G3);
   the deployed configuration itself is unchanged.

7. **Fully invested — tested 27 Sep 2026 (E1), failed its rule.** The honest
   baseline averages 12.4% cash: 44% of it in risk-off periods (the regime
   gate, kept), the rest building between rebalances as ~2 names a week
   exit (2.7% after a rebalance fills, 12% by the next). The mechanism
   `portfolio.refill_exits` (hash-neutral, off by default) fills freed
   slots between rebalances with the next-ranked names at their rebalance
   weights. Pre-registered rule: CAGR ≥ 25%, MaxDD ≤ 30%, Sharpe ≥ 1.09 on
   2013–2025, walk-forward only if that passes.

   | Run | Sharpe | CAGR | MaxDD | Calmar | gross | turnover | verdict |
   |---|---|---|---|---|---|---|---|
   | Baseline `bd79bf28` | 1.140 | 22.75% | −24.69% | 0.922 | 0.876 | 7.0× | – |
   | E1 `942760af` | 1.211 | 24.68% | −25.46% | 0.969 | 0.907 | 7.4× | FAIL (CAGR) |

   The Sharpe gain (+0.07, one-sided p ≈ 0.08, positive in 2013–16,
   2017–25 and 2021–25) is post-hoc and not established; testing it needs
   its own pre-registered walk-forward and paper trading (tracker U20).

Not on the list, on purpose: re-optimising signal weights (R21A's 247% data-
mining bias estimate came from exactly that), leverage (MTF at 14.6%/yr adds
0.5–0.75 CAGR points per 0.1× for about 3 points of MaxDD — see
`docs/leverage_due_diligence.md`, 27 Sep 2026), and anything tuned on 2026.

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
| Refresh the registry after a store rebuild | `python -m runners.run_nse_engine refresh-registry [--dry-run] [--from-hash <old>]` — see section 5c |
| Holdout | `python -m runners.run_nse_engine holdout --config <json> --data-start 2012-01-02 --start <date> --end <date>` |
| Promote | `python -m runners.run_nse_engine promote --run-id <id> --paper-start <date> --data-anchor 2012-01-02` |
| Shift reference | `python -m runners.run_nse_engine shift-reference --run-id <id> --start <paper start> --data-start 2012-01-02` |
| Inspect deployment | `python -m nse_engine.deployment show` |
