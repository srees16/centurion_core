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
```

(The 15 Sep 2026 promotion used `promote --run-id`; since 28 Sep `promote` is
the forward gate of section 6 and takes a paper candidate, not a run.)

**Result (run once, 2026-01-01 → 2026-09-11, 172 sessions):** +18.0%
(27.4% annualised), excess Sharpe 1.04, vol 18.7%, MaxDD 13.0%, turnover
8.8×/yr. Same window: NIFTY 50 TRI −9.6%, EW universe hold +2.5%, naive
momentum +46.5% (excess Sharpe 2.01). Passed both holdout gates; promoted to
`config/nse_engine_deployed.json` (paper start 2026-09-16, data anchor 2012-01-02 —
see the correction below).
Note that naive momentum beat the strategy in 2026.

Until 28 Sep 2026 `promote` refused unless PBO < 30%, DSR ≥ 0.95, the
benchmark gate passed, holdout excess Sharpe > 0 and holdout MaxDD ≤ 1.5× the
backtest MaxDD. PBO over the same-window registry reached 45–49%, so nothing
could pass it; decision U19 replaced these gates with the forward gate
(section 6), which reports all of them without gating.
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

Live trading was not affected for the deployed book: on the day itself an
adjusted close equals the printed one. The books that filter on as-printed
prices (candidate, E4: `price_filter_unadjusted`) were, until 10 Oct 2026:
`MarketData.until()`, which paper and live plan through, dropped
`close_unadj`, so their universe history fell back to adjusted prices. To
1 Oct 2025, 2,394 of 3,410 sessions of that history differed from the
registered runs'; since LN-T8 none do.

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

## 5e. The 2008 crash (R4, 27–28 Sep 2026)

The validation window starts in 2013, so no recorded run had seen a crash
like 2008 (NIFTY −60%). K4 extended NIFTY 50 back to 2006, and R4 ran two
tests on `store_ext2006`, loaded from 2006-01-02 with delivery STT at 0.125%
per side until June 2012. Both rules were written down before any run: the
2008 loss must stay within 35%, the level at which the drawdown rule moves
the core book to cash.

**Test 1, fixed configurations chosen on 2013–25, run over 2007–25.** PASS.

| Configuration | 2007–09 peak to trough | 2008 | Sharpe 2007–25 | CAGR 2007–25 |
|---|---|---|---|---|
| Live-book proxy: B1 + drawdown rule 20/30/35 | −32.9% | −29.1% | 1.03 | 21.1% |
| B1 baseline, no rule | −38.0% | −34.6% | 0.94 | 20.0% |
| Candidate 2d64ba4c (neutral 0.6) | −26.9% | −23.5% | 1.03 | 20.1% |

All three troughs run from 4 Jan to 5 Dec 2008. The regime gate was risk-off
for 90% of 2008. The rule spent 26% of 2008 at half size and 17% halted, and
cut the loss by about 5 points.

**Test 2, walk-forward.** PASS. Anchored from 2007 with one year of minimum
training, 12-month tests 2008–25, the K5 grid of 32 points on the B1 base:
18 folds, 594 backtests, 93 minutes. The 2008 fold, trained on 2007 alone,
chose neutral scale 1.0 and lost 32.9% (the year −29.7%). From 2009 on,
every fold chose neutral 0.6, with a 21-day rebalance until 2023 and 5-day
in 2024–25.

| Stitched OOS | Sharpe | CAGR | MaxDD |
|---|---|---|---|
| 2008–25 | 0.87 | 16.4% | −32.9% |
| 2017–25 subset | 1.17 | 20.2% | −16.8% |
| K5, main store, 2017–25 | 1.19 | 22.0% | −22.5% |

What it means:
- The deployed style needs about 33% of drawdown budget for a 2008-type year
  even with the rule. Decision U22 (28 Sep 2026): no hard 30% in such a
  crisis, because capturing the rebound matters more.
- 2013–25 is a friendly window. Including 2008 lowers Sharpe by 0.1–0.3.
- The walk-forward's preference for neutral 0.6 holds on 17 more years of
  data, which supports the candidate now on paper (U21).

**The rebounds (28 Sep 2026, for U22).** Returns from the NIFTY trough, with
the average gross exposure in brackets:

| | 12 months after 27 Oct 2008 | 3 months after 23 Mar 2020 | 12 months after 23 Mar 2020 |
|---|---|---|---|
| NIFTY 50 | +92.0% | +34.2% | +89.9% |
| B1, no rule | +17.1% (0.46) | +9.6% (0.51) | +73.9% (0.81) |
| B1 + drawdown rule | +17.1% (0.46) | +9.6% (0.51) | +73.8% (0.81) |
| Candidate, neutral 0.6 | +15.8% (0.45) | +8.5% (0.47) | +69.9% (0.78) |
| Regime gate off (2020 only) | | +20.4% (0.89) | +57.6% (0.88) |

The drawdown rule was back to normal before both troughs and stayed normal
for the next 12 months, so it never held back a rebound; it cut 2008's loss
from −35.5% to −30.1%. What holds the book back is the regime gate: gross
exposure took 309 days after the 2008 low to reach 0.8 (79 days in 2020,
106 for the candidate). Switching the gate off wins the first months but
loses the year and the crash. A faster re-entry after a crash is research
item R11, to be pre-registered: with 3–4 episodes it is easy to overfit.

**Data caveat (10 Oct 2026, tracker LN-T28).** `store_ext2006` has no
corporate-action or dividend records before 2010-01-04. About 17 bonus and
demerger gaps in 2006–09 that are smaller than the loader's 35% inference
threshold stay unadjusted, so a holder books a false loss on the ex-date:
ONGC 2006-10-27 (−31.6%, a 1:2 bonus), JPASSOCIAT 2009-12-17 (−31.2%),
DABUR, CROMPGREAV, RELIANCE 2006-01-18 (−25.3%, a demerger). Dividends
before 2010 are missing too. The 2008 drawdowns above read about 1 point too
deep. Comparisons on the same data (the configurations against each other,
the walk-forward's grid) are unaffected, and no verdict of R4, R11 or the
other 2007–25 runs flips. The 2013–25 window and live trading do not use
this data. Importing the events needs a source that covers 2006–09,
delisted names included; the 35% threshold stays.

**DD1: the drawdown rule's re-arm (pre-registered 11 Oct 2026, 17:37 IST,
before any run of the change).**
- **The fault:** a halt, half or risk-off episode re-arms only when equity
  closes strictly above its previous 60-session high. A book that is all
  cash has flat equity and never re-arms. That happens when stops and exits
  empty it during a halt, or when risk-off moves the core to cash, while the
  metal sleeve is out of trend and idle cash earns nothing. A replay of a 40%
  fall followed by 500 flat sessions stayed in risk-off throughout.
- **The change:** re-arm when equity closes at or above its previous
  60-session high. An all-cash book re-arms after 60 flat sessions, and the
  regime gate still decides whether the core invests. One comparison
  changes; there is no new parameter. The backtest and the live executor
  share `DrawdownTracker`, so both change together.
- **Pass rules:**
  1. The deployed reference config (live config plus the rule), 2013–25 on
     the main store: an identical equity path before and after.
  2. The same over 2007–10 on `store_ext2006`: an identical equity path.
     Its 2008 states (half 66 days, halt 42, never risk-off) are unchanged.
  3. Synthetic histories: flat equity in halt and in risk-off re-arms after
     exactly 60 sessions; falling and rising equity behave as before.
- **Not a trial:** it is a no-op on every window tested, so nothing is
  recorded in the registry.
- **Result (11 Oct 2026): PASS.**
  - 2013–25: equity identical (hash 2c3d35a4, final ₹72.33 lakh).
  - 2007–10: identical (hash 44e272a8).
  - Synthetic: flat halt and flat risk-off re-arm at session 60; falling
    equity never re-arms.
  - Measured beside it, the rule off vs on for today's deployed book:
    - 2013–25: CAGR 23.14% vs 22.83%, Sharpe 1.164 vs 1.151, MaxDD −23.9%
      vs −25.6%;
    - 2008: −37.0% vs −31.9%; 2007–10 MaxDD −40.3% vs −35.4%.

    The rule is crash insurance, so it is kept.

## 5f. Crash re-entry (R11, 28 Sep 2026): FAIL

Decision U22 asks that a crisis not keep the book out of the rebound. The
regime gate is what does that (section 5e). R11 tested one fixed rule,
written down at 14:11 IST before any run, with no fitted value:

- a crash episode starts when NIFTY closes 25% or more below its 252-session
  high and ends at a new 252-session high;
- inside an episode, while NIFTY is above its 50-day mean (confirmed over 3
  days, as the gate), the regime is forced to risk_on, over the trend,
  breadth and VIX legs.

Primary comparison: the live-book proxy, B1 with the drawdown rule, with
and without the re-entry rule. It had to pass all four checks.

| Check | Without | With | Rule | Verdict |
|---|---|---|---|---|
| 12 months after the 27 Oct 2008 low | +17.1% | +32.6% | +10 pts or more | PASS |
| 12 months after the 23 Mar 2020 low | +76.6% | +84.2% | +10 pts or more | FAIL (+7.6) |
| Excess Sharpe 2017–25 | 1.352 | 1.393 | at least −0.05 | PASS |
| 2007–09 peak to trough | −32.9% | −39.4% | at most 5 pts deeper | FAIL (6.5) |

The rule was active for 372 sessions on the 2006 store. In 2008 it bought
the April–May and August–September bear-market rallies, which deepened the
crash; a 50-day signal cannot tell a bear rally from a bottom. In 2020,
NIFTY confirmed above its 50-day mean only on 28 May, two months into the
rebound. B1 without the rule and the candidate showed the same pattern.
Sharpe rose by 0.02–0.05 over 2017–25 in all three.

Not adopted, and not re-tuned: 2008, 2011–12 and 2020 are every crash the
data holds, so a second rule would be fitted to the same episodes. The flag
`regime.crash_reentry` stays in the code, off and hash-neutral, so the
recorded runs can be reproduced.

Found on the way: a load starting before 2008 never counted the India VIX
cache as complete, so it re-downloaded from Yahoo every time, and a failed
download silently left VIX out. One R11 run got a different data hash and
was re-run on the right data; R4's runs were unaffected. The loader now
falls back to its cache.

## 5g. Turnover (R7, 28 Sep 2026): FAIL

The book trades 7.0× its equity a year one way and pays 3.2% a year in
costs: rebalance trims 3.3× (1.2%), rank exits 1.85× (0.7%), the metal sleeve
1.5× (1.1%). The rebalance cadence is already in the walk-forward grid, so R7
tested the two brakes outside it, written down at 21:30 IST before any run:
T1 doubles the no-trade buffer (0.25 → 0.50), T2 lowers the exit rank from
40 to 60 names, T3 does both. The eight backtests ran on Kaggle, bases
included, on the same data fingerprint as the local store.

Primary, T3 against B1, 2013–25:

| Check | B1 | T3 | Rule | Verdict |
|---|---|---|---|---|
| One-way turnover | 6.87× | 5.33× | at least 25% lower | FAIL (−23%) |
| Net CAGR | 22.83% | 23.38% | +0.5 pt or more | PASS |
| Excess Sharpe 2017–25 | 1.323 | 1.405 | at least −0.05 | PASS |
| MaxDD | −23.8% | −24.8% | at most 2 pts deeper | PASS |

Costs fell from 3.08% to 2.48% a year. Reported only: the buffer alone cut
turnover 5% and left CAGR flat; the exit rank alone cut it 17% and added
0.87 pt of CAGR and 0.09 of Sharpe. On the candidate book T3 added only 0.11
pt of CAGR, the exit rank alone 0.94. Not adopted. The exit-rank result was
found in R7's own data, so it is logged as R12 for evidence R7 did not see
(a walk-forward with the exit rank in the grid, or paper), not promoted.

## 5h. Cost model 2: ETF STT (U25, 30 Sep 2026)

The cost model charged the metal-sleeve ETFs the equity delivery STT, 0.1% a
side. Zerodha lists gold ETFs as exempt and charges other ETFs 0.001% on the
sell side only; silver ETFs are reported exempt but not named by Zerodha, so
they pay the other-ETF rate (₹1 per lakh sold). `nse_engine.costs` now takes
the symbol; the engine, the paper books and the live session pass it. Every
run records `cost_model` (1 before this change, 2 after), and `validate`
compares runs of one version only.

The 56 same-window configurations (2013–25, the data fingerprint 6c94f4f4)
were re-run on Kaggle under model 2 (no new trials). Same-platform effect,
Kaggle model 1 → model 2:

| | CAGR | Excess Sharpe | MaxDD | Cost drag |
|---|---|---|---|---|
| B1 baseline | 22.83% → 23.07% | 1.147 → 1.163 | −23.8% → −23.5% | 3.08% → 2.79% |
| Candidate 2d64ba4c | 22.03% → 22.42% | 1.174 → 1.201 | −23.4% → −23.0% | 3.01% → 2.72% |

Under model 2 on Kaggle, the deployed 679cbd0c reads Sharpe 1.197, CAGR
23.6%, MaxDD −23.1%, Calmar 1.02; PBO 46.4% over the 56 configurations,
deflated Sharpe 0.987, benchmark gate passed (candidate: DSR 0.988, gate
passed). The walk-forwards (K5, R4) and the 2006-store studies were measured
under model 1 and are not re-run: every configuration carries the same sleeve,
so choices between them do not change, and their levels are about 0.3 points
of CAGR conservative.

## 5i. Metal sleeve trend confirmation (R13, 1 Oct 2026)

On 29 Sep the deployed paper book sold its gold sleeve (28% of the book) at
a loss of ₹29,662 after one close below the 200-day average, 8 sessions
after buying it; gold closed back on the average the next day. The recorded
2013–25 backtest of 679cbd0c shows the same pattern: 22 of its 34 gold round
trips lasted 20 sessions or less and together cost about 1% of the book a
year (excluding one trade filled at a stray opening print, tracker D4),
while the longer trends made +18.9% of the book.

**Pre-registered 1 Oct 2026, 00:06 IST, before any run.** One variant, no
grid: `sleeves.trend_confirm_days = 3`. A sleeve enters its trend only after
3 closes in a row above the average and leaves it only after 3 in a row
below, the regime gate's own `confirm_days` and hysteresis
(`apply_hysteresis`). The default of 1 is the old rule and keeps every
config hash.

Primary comparison: the deployed engine 679cbd0c with and without the rule,
both run on Kaggle in one job under cost model 2. It passes only if all four
checks hold:

| Check | Window | Rule |
|---|---|---|
| Excess Sharpe | 2013–25, anchor 2012-01-02 (fingerprint 6c94f4f4) | at least base + 0.02 |
| CAGR (calendar years) | 2013–25 | at least the base |
| MaxDD | 2013–25 | at most 1 point deeper |
| MaxDD | 2007–25, `store_ext2006`, anchor 2006-01-02 | at most 1 point deeper |

Reported, not gating: the candidate 2d64ba4c with and without the rule;
gold and silver round trips of 20 sessions or less and their cost; sleeve
turnover; excess Sharpe 2017–25; PBO and deflated Sharpe with the new
configurations counted. A pass makes the rule eligible, not deployed: it
changes the engine hash, so adopting it is your decision under the forward
gate (U19, V3). A fail is logged and the flag stays in the code, off.

**Result: FAIL.** Four backtests in one Kaggle job (Python 3.12.13, pandas
2.3.3, the U25 image); the two bases reproduced their 30 Sep runs to 15
significant digits. 679cbd0c → 87a481bb:

| Check | Base | With the rule | Rule | Verdict |
|---|---|---|---|---|
| Excess Sharpe 2013–25 | 1.197 | 1.156 | at least 1.217 | FAIL (−0.040) |
| CAGR 2013–25 | 23.63% | 23.32% | at least the base | FAIL (−0.31 pt) |
| MaxDD 2013–25 | −23.1% | −21.5% | at most 1 pt deeper | PASS (1.6 pts shallower) |
| MaxDD 2007–25 (cost model 3, both) | −37.7% | −38.2% | at most 1 pt deeper | PASS (0.5 pt deeper) |

The rule did what it was built for: gold round trips of 20 sessions or less
fell from 22 (−1.22% of the book a year) to 8 (−0.79%), the sleeves' own P&L
over 2013–25 rose from +4.1% to +6.0% of the book, sleeve turnover fell from
2.9× to 1.7× equity a year and total costs from 2.80% to 2.44% a year.
Calmar rose from 1.02 to 1.09. The book still earned less: most years moved
by ±2 points, and two moved by more, both on the stray opening prints of
tracker D4. On 24 Dec 2020 the variant bought 26% of the book in gold at an
open of 48.98 against a day's range of 43.6–43.7 (−2.8% of the book; 2020
−8.1 points); on 18 Oct 2021 the base bought at 44.00 against a close of
40.98 (−3.2%; 2021 +4.3 points for the variant). The two roughly cancel, so
the verdict does not rest on them, but one stray print moves a single
configuration by about 3% of the book.

The 2007–25 check ran on Kaggle against `store_ext2006` after D4 (5j), so
base and variant both use cost model 3; over 2007–25 the rule also lowered
Sharpe (0.949 → 0.913) and CAGR (19.95% → 19.33%). Under cost model 3 the
2013–25 comparison is 1.206 / 23.74% against 1.166 / 23.40%: the same verdict.

Candidate 2d64ba4c → a80af9f5 (reported): Sharpe 1.201 → 1.163, CAGR 22.42%
→ 22.25%, MaxDD −23.1% → −23.4%. With both variants counted, PBO over the
58 same-window configurations is 47.6% (46.4% over 56), deflated Sharpe
0.981 and 0.982, benchmark gate passed. Not adopted and not re-tuned:
another hysteresis length would be fitted to these same whipsaws. The flag
stays in the code, off and hash-neutral, so the runs reproduce.

## 5j. Stray ETF opens: cost model 3 (D4, 1 Oct 2026)

The backtest and the paper books fill orders at the next session's open. For
the metal ETFs that open is often a stray first trade of a few units: on 24
Dec 2020 GOLDBEES opened at 48.98 and traded at 43.58–43.73 for the rest of
the day, and a sleeve bought there loses 2.8% of the book at once (R13, 5i).
Measured on the 2013–25 panel, before any backtest:

| Open more than 5% from the day's close | Share of days | At the day's high or low |
|---|---|---|
| Top-300 shares | 4.6% | 24% |
| GOLDBEES, SILVERBEES | 2.0%, 1.0% | 88% |

Shares open in the call auction and move a lot within a day, so their opens
stay. The metal ETFs do not: gold's close-to-close move exceeds 2.95% on 1%
of days, yet its open sits more than 3% from its close on 4.75%.

**Fixed 1 Oct 2026, 00:37 IST, before any run.** An ETF's open is held within
3% of the same day's close for every fill modelled at the open (orders, and
the paper books' pending orders); shares are unchanged. The 3% is gold's
99th-percentile daily move, set from the price data above, not from backtest
results. Live orders are LIMIT orders around the previous close and are not
affected. This is cost model 3; it is adopted on the data evidence, whatever
it does to the backtests, and the 58 same-window configurations are re-run
on Kaggle under it (no new trials), as for U25.

**Result.** All 58 configurations re-ran in one Kaggle job (the U25 image,
fingerprint 6c94f4f4); 154 GOLDBEES and 38 SILVERBEES opens are capped over
2012–25, no share price changes. Model 2 → model 3, every configuration:

| | Mean | Range |
|---|---|---|
| Excess Sharpe | +0.015 | 0.000 to +0.037 |
| CAGR | +0.14 pt | −0.01 to +0.30 pt |
| MaxDD | 0 | 0 |

The ranking does not move (Spearman 0.998), so no earlier choice between
configurations changes. Deployed 679cbd0c: Sharpe 1.206, CAGR 23.74%, MaxDD
−23.1%, Calmar 1.03; candidate 2d64ba4c: 1.213 / 22.56% / −23.0%. PBO over
the 58 is 48.6%, deflated Sharpe 0.989 and 0.990, benchmark gate passed for
both. The paper books fill the metal ETFs the same way from their next
session; live orders are unchanged.

## 5k. Exit rank 60 out of sample (R12, 1 Oct 2026)

In R7 (5g) lowering the exit rank from 40 to 60 names, alone, added 0.87
point of CAGR and 0.09 of Sharpe and cut turnover 17%, but that was found in
R7's own data. R12 asks whether a walk-forward picks it out of sample.

**Pre-registered 1 Oct 2026, 10:30 IST, before any run.** Two walk-forwards
on Kaggle under cost model 3, in parallel, identical except for the grid:
B1 base (`bd79bf28`), anchored, 4-year minimum training, 12-month tests
2017–25 (9 folds), data from 2012-01-02 (fingerprint 6c94f4f4), selection on
training excess Sharpe as in K5.

- Arm A, control: the K5 grid (32 points: group weights, stop 3/6 ATR,
  rebalance 5/21, neutral 0.6/1.0, 20/30 positions).
- Arm B: the same grid × `portfolio.exit_rank` {40, 60} (64 points).

PASS needs both: arm B picks exit rank 60 in at least 5 of the 9 folds, and
arm B's stitched OOS excess Sharpe 2017–25 is at least arm A's − 0.05.
Reported, not gating: OOS CAGR, MaxDD and turnover of both arms, each fold's
choice. A pass sends exit rank 60 to E4 (one combined configuration with
refill exits); a fail closes R12 and E4 tests refill exits alone.

**Result: PASS.** Both arms ran on Kaggle (the U25 image, 297 and 585
backtests, 48 and 93 minutes), every run on fingerprint 6c94f4f4 under cost
model 3.

| Stitched OOS 2017–25 | Arm A (K5 grid) | Arm B (+ exit rank) | Rule | Verdict |
|---|---|---|---|---|
| Folds choosing exit rank 60 | | 9 of 9 | at least 5 | PASS |
| Excess Sharpe | 1.150 | 1.273 | B at least A − 0.05 | PASS (+0.12) |
| CAGR (calendar) | 20.87% | 21.93% | | |
| MaxDD | −20.9% | −17.2% | | |
| Test-year turnover | 5.92× | 4.90× | | |

In every fold's training grid, exit rank 60 beat 40 on 25–30 of the 32
otherwise identical settings, rising from 26 in 2017 to 30 in 2024–25. OOS
years, arm A / arm B: 2017 +39.8% / +29.5%, 2018 −11.6% / −4.3%, 2019 +5.4% /
+6.5%, 2020 +27.3% / +24.3%, 2021 +77.2% / +71.7%, 2022 −4.7% / −5.1%, 2023
+21.1% / +32.7%, 2024 +22.5% / +26.0%, 2025 +32.5% / +33.3%: exit rank 60
gives up some of the strongest years and loses less in the bad one. Arm B
chose the low-vol group with 30 names for 2017–20 and the deployed-style
fast/slow trend with 20 names from 2021; the most recent fold's choice is the
B1 base with neutral 0.6, 5-day rebalance and exit rank 60. Arm A reproduced
K5's choices (monthly rebalance early, neutral 0.6 from 2022), at 1.150
against K5's 1.187 measured locally under cost model 1. The 882 runs are in
the registry. Next: E4 (exit rank 60 with refill exits, one configuration).
Files: `data/nse_engine/wf_stitched_r12a.json`, `wf_stitched_r12b.json` and
their `wf_oos_returns_*.csv`.

## 5l. Exit rank 60 with refill exits (E4, 1 Oct 2026)

Your target from 1 Oct is CAGR above 25% with Sharpe above 1.2. Two changes
each moved the B1 base toward it under cost model 3 (2013–25, Kaggle): exit
rank 60 (R12 passed it out of sample) and refilling exits between rebalances
(E1):

| B1 base, 2013–25 | Sharpe | CAGR | MaxDD | Turnover |
|---|---|---|---|---|
| B1 `bd79bf28` | 1.172 | 23.19% | −23.5% | 6.88× |
| + exit rank 60 `8cc54eda` | 1.215 | 24.23% | −25.2% | 5.70× |
| + refill exits `942760af` | 1.210 | 24.65% | −23.7% | 7.32× |

**Pre-registered 1 Oct 2026, 12:34 IST, before any run.** One configuration,
no grid: B1 with `portfolio.exit_rank = 60` and `portfolio.refill_exits =
true` (`93cf6c4d`), run on Kaggle under cost model 3 beside B1 in the same
job, 2013–25 from the 2012-01-02 anchor. It passes only if all four hold
against that B1 run:

| Check | Rule |
|---|---|
| CAGR (calendar) | at least 25.0% |
| Excess Sharpe | at least B1's |
| MaxDD | at most 2 points deeper than B1's |
| Annual turnover | not higher than B1's |

Reported, not gating: the same two configurations over 2007–25 on
`store_ext2006`; excess Sharpe 2017–25; PBO and deflated Sharpe with E4
counted. A pass sends E4 to a third paper slot (D5) under the forward gate;
a fail ends half 1 of the plan, and the target then needs a new return
stream (R5).

**Result: FAIL, on CAGR alone.** One Kaggle job per window (the U25 image),
B1 and E4 side by side; B1 reproduced its D4 run.

| 2013–25 | B1 | E4 | Rule | Verdict |
|---|---|---|---|---|
| CAGR | 23.19% | 24.64% | at least 25.0% | FAIL (0.36 pt short) |
| Excess Sharpe | 1.172 | 1.205 | at least B1's | PASS |
| MaxDD | −23.5% | −24.6% | at most 2 pts deeper | PASS |
| Annual turnover | 6.88× | 5.79× | not higher | PASS |

The two changes do not add up: each alone reached 24.2–24.7%, together
24.6%. Reported: excess Sharpe 2017–25 1.370 → 1.493, costs 2.79% → 2.38% a
year, average gross 0.87 → 0.90; over 2007–25 on `store_ext2006`, Sharpe
0.945 → 0.973, CAGR 19.94% → 21.11%, MaxDD −37.7% → −39.1%. With E4 counted,
PBO over the 59 same-window configurations is 50.5%, deflated Sharpe 0.988,
benchmark gate passed. Not re-tuned: the margin is small, but moving the
threshold or adding a third change after seeing this result is the fitting
the rule exists to prevent. Half 1 of the plan ends here; 25% needs a new
return stream (R5). E4 remains a better-quality book than B1 (higher
Sharpe, lower turnover and costs); whether to paper-trade it on those
grounds is decision U26. U26 (1 Oct): yes; E4 paper-trades from Mon 5 Oct
2026 as the `e4` book (D5, `config/nse_engine_e4.json`). R5, the new return
stream named above, was closed without a trial (5m).

## 5m. NIFTY trend sleeve (R5, 1 Oct 2026): closed without a trial

A NIFTY sleeve can only use capital the book leaves idle, so its room was
measured first, on E4's recorded 2013–25 run (no new run): mean gross 0.903.

| Regime | Share of days | Mean idle capital |
|---|---|---|
| risk_on | 66% | 3.8% |
| neutral | 23% | 5.0% |
| risk_off | 11% | 56% |

Holding NIFTYBEES with the idle cash only while NIFTY is above its 200-day
mean (consistent with the regime gate) deploys 3.6% of the book on average:
at most +0.50 point of CAGR a year before costs, against the +0.36 E4 needs
for 25%. A 12-month momentum rule deploys 6.8% and bounds +1.0 point, but
almost all of it by buying NIFTY while the gate is risk_off, the pattern
that deepened 2008 in R11. Net of trading costs (and the 0.1% delivery STT
the cost model would charge NIFTYBEES until ETF rates cover it) the gate-
consistent version would land near 25%, a coin flip on the rule. Your
decision, 1 Oct: close R5 without spending a trial; PBO stays 50.5% over 59.

## 5n. A second uncorrelated sleeve: MON100 (R14, 1 Oct 2026)

The metal sleeve is what lifts the book's Sharpe: on E4's recorded run its
daily contribution has correlation −0.02 with the core's, and without it
(capital idle) the book reads Sharpe 1.12 and MaxDD −26.4% against 1.22 and
−24.6%; moving its capital into the core instead gives CAGR +0.4 point with
Sharpe 1.03 and MaxDD −33.7%. An uncorrelated sleeve is therefore the one
lever that raises CAGR, Sharpe and Calmar together. Of the ETFs in the
store with history from 2012 and real liquidity, one qualifies: MON100 (the
Nasdaq-100 ETF), 24.6% CAGR in rupees 2013–25, 17.2% inside its 200-day
trend, volatility 22%, correlation 0.18 with the core, ₹7.6 crore traded a
day in 2021–25 (5% participation is ₹38 lakh). Caveat written down now:
from early 2022 the ETF has traded above its net asset value because of
SEBI's cap on overseas investment; the store's traded prices include that
premium on both entry and exit.

**Pre-registered 01 Oct 2026, 16:13 IST, before any run.** One configuration, no grid: E4
with `sleeves.extra_symbols = ("MON100",)` (`042a5b69`), the third trend
sleeve on the same 200-day rule, sharing the sleeve capital by inverse
volatility (about a third of it, ~9% of the book). MON100 pays the
other-ETF STT; no recorded run ever traded it, so cost model 3 is
unchanged. Run on Kaggle beside E4 in one job, 2013–25 from the 2012-01-02
anchor. It passes only if all three hold against that E4 run:

| Check | Rule |
|---|---|
| CAGR (calendar) | at least 25.0% |
| Excess Sharpe | at least E4's (1.205) |
| MaxDD | at most 2 points deeper than E4's |

Reported, not gating: excess Sharpe 2017–25, Calmar, turnover, the sleeve's
own round trips, and 2022–25 alone (the premium episode). A pass sends the
configuration to a fourth paper book (a config file, D5); a fail closes
R14 without re-tuning. The trial budget before the January forward-gate
decisions is two configurations; this is the first.

**Result: FAIL.** One Kaggle job (the U25 image), E4 and the variant side by
side, both on fingerprint 6c94f4f4 under cost model 3; E4 reproduced its
recorded run.

| 2013–25 | E4 | + MON100 | Rule | Verdict |
|---|---|---|---|---|
| CAGR | 24.64% | 22.76% | at least 25.0% | FAIL (−1.9 pts) |
| Excess Sharpe | 1.205 | 1.203 | at least E4's | FAIL (−0.002) |
| MaxDD | −24.6% | −23.8% | at most 2 pts deeper | PASS |

The sleeve took 15.6% of the book on average, from the core (64% → 53%)
and from silver (10% → 2%, the inverse-vol split), and its own trend return
(14–17% a year, 19 round trips, +28.8% of the book net) does not replace the
core's ~34% per unit of capital. The diversification it added (correlation
0.18) showed up as 0.8 point less drawdown and nothing else: Sharpe 2017–25
fell from 1.49 to 1.36, Calmar from 1.00 to 0.96, and the premium episode
2022–25 was a wash (Sharpe 1.19 → 1.18). The mechanism assumed from the
metal sleeve does not transfer: gold's return is uncorrelated *and* cheap
in capital (it was taken from cash the regime gate had already freed),
while a second sleeve of equity-like volatility is funded from the core.
With the variant counted, PBO over the 60 same-window configurations is
56.1%, deflated Sharpe 0.988, benchmark gate passed. Not adopted, not
re-tuned (a smaller sleeve share would be fitted to this result). The
flag `sleeves.extra_symbols` stays in the code, empty and hash-neutral.
One configuration of the two-trial budget remains before January.

## 5o. The haircut beside every validation (D2, 1 Oct 2026)

A backtest is the best case. `validate` now re-simulates the run twice on
this machine, unrecorded: as recorded, and with every fill one session late
and the market-impact model doubled (R8's stress). The difference is the
execution haircut, platform-free because both legs run here; it is applied
to the recorded metrics. The CSCV selection haircut, the in-sample pick's
mean OOS Sharpe over its mean IS Sharpe, is applied on top. The result is
in `validation.json` under `haircut` and on one line of the forward gate's
summary.

| 1 Oct 2026 | Recorded | After 1-day lag + 2× impact | Selection ×0.85 → expected OOS Sharpe |
|---|---|---|---|
| Deployed 679cbd0c | 1.197 / 23.6% / −23.1% | 1.00 / 20.2% / −22.8% | 0.85 |
| E4 93cf6c4d | 1.205 / 24.6% / −24.6% | 1.08 / 22.2% / −26.5% | 0.91 |

Those are the numbers to hold the live book to, not the recorded ones. The
realised haircut replaces them once the live book has fills: G4's cost
ratio and tracking error are its first two components.

## 5p. A futures short sleeve, scoped (R15, 1 Oct 2026): not now

The only route left to a higher ceiling (U28) is a short sleeve in
single-stock futures. Scoped with no build and no trial: one F&O bhavcopy
per year from NSE's public archive, and one standard definition fixed
before looking (each month, short the bottom decile of 12-1 momentum among
that year's futures stocks, equal weight; a short future earns about the
risk-free rate minus the stock's total return; costs 2.5% a year).

| | Finding |
|---|---|
| Data | every year from 2013 (legacy format to mid-2024, UDiFF after) |
| Universe | 135–223 stocks with futures; median near-month turnover ₹40–220 crore a day |
| Lot size, 30 Sep 2026 | ₹3.0 lakh minimum, **₹6.3 lakh median**, ₹16.9 lakh maximum |
| Short sleeve, 2013–25 | **−11.0% a year net**, volatility 36%, Sharpe −0.30, MaxDD −88% |
| Losers vs the futures universe | +6.0% a year spread |
| E4 + 20% short sleeve | CAGR 25.2% → 22.9%, Sharpe 1.09 → 1.05, MaxDD −23% → −18% |
| Worst months | 2014-05 −40%, 2020-06 −31% (rebounds) |

Three reasons, each sufficient. Size: one median lot is 21% of a ₹30 lakh
book and all of the ₹6 lakh first rung; ten names need ~₹60 lakh of short
notional, a book of ₹2–3 crore. Edge: momentum losers do trail the
universe (6% a year), but a naked short pays the market's rise, so the
sleeve is a hedge that costs CAGR and Sharpe; the spread needs a
market-neutral pair, which needs more capital still. Risk: its worst
months are rebounds, against decision U22. Revisit as a market-neutral
spread when the book is past ₹2 crore. One look, not a trial: no
configuration was recorded.

U28 closed on this evidence (10 Oct 2026), with that trigger to reopen it.
The scorecard's ex-ante regime split (LN-T18) has since removed the case for
a bear hedge: the book is about flat in bear phases, not −22% a year.

## 5q. Options sleeve, first round (O2, 3 Oct 2026)

The book is long-only, so the options that complement it are the ones
Varsity writes: calls, sold above the range the index should stay in, which
earn in flat and falling months and lose in the strongest rallies, when the
core gains most. The toolkit's hard limits refuse an unlimited loss, so
each short call carries a long call further out: a bear call spread (M6
ch. 8) with a defined maximum loss. Options are a separate trial family
(U32): their runs go to `data/nse_engine/runs_options/`, with their own PBO
and deflated Sharpe; only the combined book enters the main registry.

**Pre-registered 3 Oct 2026, 20:55 IST, before the F&O data was in hand and
before any options backtest existed.** Three configurations, no grid, all on
NIFTY's monthly options (the last NIFTY option expiry of each calendar
month; weeklies exist only from 2019), 2013-01-01 to 2025-12-31. 2026 stays
unseen.

| | A1: SD writer, 15 days | A2: SD writer, 4 days | B: max-pain writer |
|---|---|---|---|
| Source | M5 ch. 18 | M5 ch. 18 | M6 ch. 13 |
| Decision day | first session ≤ 15 calendar days before expiry | first session ≤ 4 days before | first session ≤ 15 days before |
| Short call | first strike above the 2 SD upper bound | above the 1 SD bound | first strike ≥ max pain × 1.05; skip the month if that is not above spot |
| Long call (wing) | first strike above the 3 SD bound | above the 2 SD bound | first strike ≥ short + 1 SD of the period, in points |
| Early exit | when the short strike becomes the ATM strike | same | none: held to expiry, as the PDF says |

Common to all three:

- **Range.** The ch. 18 linear method, S × (1 + μn ± kσ√n). σ and μ are the
  sample SD and mean of NIFTY 50's daily log returns over the 252 sessions
  ending on the decision day. n is the calendar days to expiry, as in the
  PDF. That is wider than a session count, so the strikes sit further out.
- **Strikes.** Only strikes of that expiry that traded (contracts > 0) on the
  decision day are eligible. No eligible strike, or no net credit at the
  fill, means no trade that month.
- **Events.** No position whose life spans a Union Budget day or a general
  election result day (ch. 18, "skip event days"). The dates are in
  `kite_connect/options/sleeves.py`.
- **Fills.** Decisions use the decision day's close; fills come one session
  later at each leg's close, or its settlement price if it did not trade.
  An exit signalled on a close fills the next session. Expiry settles at
  NIFTY 50's close, or failing that the expiring future's settlement price.
  Lot sizes come from the data: UDiFF's lot column, or before July 2024 the
  near-month NIFTY future's turnover ÷ contracts ÷ price.
- **Costs.** `fno_costs` cost model 1, at each day's rates:
  - brokerage ₹20 per order
  - STT on the sell side, and on exercise of an ITM long (on the full value
    before Sep 2019)
  - exchange charges, SEBI fee, stamp duty and GST
  - slippage of max(1 tick, 0.5% of premium) per side
- **Size.** Ch. 18's split puts 25% of capital into option writing and
  commits 35% of that per trade. The sleeve starts at ₹7.5 lakh (25% of the
  ₹30 lakh book), and each month's maximum loss is at most 35% of the
  sleeve's equity, in whole lots.
- **Marking.** The sleeve is marked daily at close (settlement if
  untraded). Its return is option P&L over sleeve equity, with no interest
  on the collateral; that yield belongs to the book.

**Gate 1: each configuration alone.** Deflated Sharpe ≥ 0.95, on the
sleeve's daily returns with rf = 0, at N = the options configurations
recorded (3). Reported, not gating:
- CAGR on sleeve capital, MaxDD and Calmar
- months traded and skipped, win rate, worst month, costs
- CSCV PBO across the family

**Selection.** Of the configurations that pass gate 1, the one with the
highest Sharpe goes forward. Ties go to A1, then A2, then B. If none passes,
round 1 closes and the book's last configuration stays unused.

**Gate 2: the combined book.** E4's recorded run
(`20261001T104724595094Z_93cf6c4d`, 2013–25, cost model 3) plus the sleeve
as an overlay: r = r(E4) + 0.25 × r(sleeve) each day. The margin is posted
from the book's holdings and cash, so no capital leaves the core. This is
the R14 lesson: a sleeve funded from the core costs more than it adds. It
passes only if all three hold against E4:

| Check | Rule |
|---|---|
| CAGR (calendar) | at least 25.0% |
| Excess Sharpe (rf 6.5%) | at least E4's (1.205) |
| MaxDD | at most 2 points deeper than E4's (−24.6%) |

Reported, not gating:
- Calmar
- excess Sharpe 2017–25 (the walk-forward test years)
- deflated Sharpe at the total trial count (book plus options)
- PBO over the book's same-window configurations with the combined run
  counted

The combined run is recorded in the main registry as one configuration of
the book's budget (U32). A pass sends the sleeve to paper trading with the
toolkit beside the E4 book, and live only after its own forward evidence. A
fail closes round 1 without re-tuning. Later options strategies are new
configurations in the options family. Caveat written down now: SEBI
requires half of F&O margin in cash or cash equivalents, and E4 holds about
4% idle cash while risk-on, so in full-invested months the overlay may
need a liquid-ETF pledge.

**Addendum, 3 Oct 2026, 21:40 IST, before any result: a 2007–25 check,
reported, not gating (your request).** NSE's F&O archive runs back to
June 2000, but NIFTY options barely traded before 2007, and NSE's own index
closes start only in February 2012. The same three configurations are
therefore also run over 2007-01-02 to 2025-12-31, the window of E4's
2007–25 run (`20261001T070903385179Z_93cf6c4d`, `store_ext2006`). That
window takes in the 2008 crash and the election-day rally of 18 May 2009.

- **Before 2012:** NIFTY comes from the external history (`nse_engine.data.external`).
  - Yahoo ^NSEI from 17 Sep 2007 matches NSE to the paisa.
  - Before that, decisions read the labelled Sensex proxy. Expiries then
    settle only on an exact close, or else on the expiring future's
    settlement price.
- **Not new configurations:** a window is part of a run, not of a
  configuration. The hash excludes dates, as the engine's does, so these
  runs do not add to gate 1's N of 3.
- **What is reported:** each sleeve's 2007–25 metrics. If gate 2 selected a
  sleeve, E4 2007–25 plus that sleeve at 0.25 against E4 2007–25 alone,
  recorded on that run's window.
- **No influence on the gates:** neither gate, nor the selection, reads any
  of it.

**Code fixes, 4 Oct 2026, 00:40 IST, before any result was read.** The first
Kaggle runs of both windows stopped on two defects, fixed and relaunched:
(1) the max-pain sleeve read open interest before checking that the
expiry had a call chain that day, so a month with none crashed it; it now
skips the month, as the SD sleeves already did; (2) the charge schedules
began on 1 Jun 2008, so the 2007–25 window raised on its first fill; they
now begin with STT on derivatives (1 Oct 2004: 0.01%, 0.0133% from Jun
2005, 0.017% from Jun 2006; no exercise levy before Jun 2008). No rate in
force for 2013–25 changed; the cost model stays version 1. The two partial
outputs were discarded unread.

**Result, 4 Oct 2026: round 1 closed, no sleeve passed gate 1.** Both
windows ran on Kaggle (the pinned image), on the F&O store built from NSE's
archive 2000–2026 (data hash `c3f44a34`), six runs in
`data/nse_engine/runs_options/`. Gate 1, each sleeve alone over 2013–25 on
its ₹7.5 lakh, rf 0, N = 3:

| Sleeve | Months traded (of 156) | Win rate | Sharpe | CAGR | MaxDD | Deflated Sharpe | Verdict |
|---|---|---|---|---|---|---|---|
| A1: 2 / 3 SD at 15 days | 103 | 95% | 0.82 | +0.9% | −2.1% | 0.865 | FAIL |
| A2: 1 / 2 SD at 4 days | 147 | 89% | −0.27 | −7.2% | −71.4% | 0.002 | FAIL |
| B: max pain + 5% | 145 | 94% | 0.26 | +2.2% | −26.1% | 0.239 | FAIL |

No selection, so gate 2 was not run and nothing entered the book's
registry; the book's last configuration stays unused. Reported, 2007–25
(E4's extended window): A1 Sharpe 0.19, CAGR +0.5%, MaxDD −8.6%; A2 −0.17,
−4.9%, −72.4%; B −0.11, −2.3%, −59.1%. The options-family PBO over 3 trials
is 1.9%, too few trials to mean anything.

What the months say. The profile is the one a short call has: most months
earn a small credit and a few rallies take back more than the rest earned.
A2 lost ₹3.3 lakh (35% of the sleeve) in May 2016 alone, bought back at
the money after a 2% move within days, and ₹4.0 lakh over 2020; its early
exits fired 8 times. B's worst months were January 2015 (short 8750 at
max pain + 5%, settled at 8952) and, in the extended window, March 2009
(short 2750, settled at 3082). A1 is the cautious version and shows what
the rule is worth without the blow-ups: in 33 of 156 months no listed
strike lay beyond 2 SD, and where it traded the credit was ₹1,000 to
₹8,000 a month, so it returned less than a treasury bill on its capital.
Charges over 13 years were ₹5,000 to ₹13,000 per sleeve: costs are not the
reason. Writing calls above the expected range and holding to expiry, as
Varsity ch. 18 and ch. 13 prescribe, does not produce a return stream worth
a quarter of the book on NIFTY's 2013–25 path.

Not re-tuned: a narrower buffer, a later entry or a stop would be fitted
to these months. The data and the harness remain (one Kaggle run per
round), so a second round is a new pre-registered set of configurations in
the options family (U32), for your go/no-go: the ones the evidence points
at are a protective bear put spread judged on the combined book's MaxDD
and Calmar rather than on its own return (a hedge costs carry by design),
and weekly-expiry versions, which exist only from 2019 and so have less
history to pass on.

## 5r. Options round 2: PyPatel's signal strategies (O3, 9 Oct 2026)

Your request: compare the strategies in
`github.com/PyPatel/Options-Trading-Strategies-in-Python` (last commit
`c7b8a0d`, Aug 2019) with Centurion's, test them, and paper-trade the best.
That repository's four strategies trade no options: three use an options or
breadth indicator to time the S&P 500 future (put-call ratio bands, TRIN
bands, a VIX threshold) and one is a 55-day breakout on three US stocks.
None runs as published (indentation errors, Python 2, swapped datasets,
a retired Quandl feed) and none was costed or tested outside about a year
of 2017–18 data. Centurion has every input on Indian data, so round 2
tests what each was meant to do, on NIFTY futures, in the options family
(U32).

**Pre-registered 9 Oct 2026, 19:41 IST, before any code for these strategies
existed and before any of their results.** Seen beforehand: the yearly mean
NIFTY open-interest put-call ratio (0.94 to 1.46, 2007–26), and nothing
else of these signals. Four configurations, the repository's own parameters,
no grid, 2013-01-01 to 2025-12-31; 2026 stays unseen until the verdict.

| | X1: PCR bands | X2: TRIN bands | X3: VIX threshold | X4: 55-day breakout |
|---|---|---|---|---|
| Source file | `PCR_strategy.py` | `TRIN_strategy.py` | `VIX_Strategy.py` | `Turtle Trading.py` |
| Indicator | NIFTY options put-call ratio by volume: put contracts ÷ call contracts, all expiries, each session | log TRIN = log[(advancers ÷ decliners) ÷ (advancing volume ÷ declining volume)], every EQ-series stock, close vs previous close | India VIX close | NIFTY 50 close against the prior 55 sessions |
| Bands / levels | mean over 20 sessions ± 1.5 σ; stop bands 2 σ beyond those | the same over 22 sessions | VIX ≥ 22 | high, low and mean of the 55 closes before the day |
| Enter (flat) | long when the indicator crosses above the upper band; short when it crosses below the lower band | the same | long when VIX ≥ 22 | long above the 55-day high; short below the 55-day low |
| Exit a long | crosses back below the mean, or above the upper stop band, or NIFTY 1% below the entry | the same | NIFTY 5% above or 5% below the entry | close below the 55-day mean |
| Exit a short | crosses back above the mean, or below the lower stop band, or NIFTY 1% above the entry | the same | (long only) | close above the 55-day mean |

Carried over as written:
- **σ.** The repository's estimator: the square root of the mean, over the
  last (window − 1) sessions, of (indicator − that day's moving average)².
- **Crossings.** As written: today's value beyond today's band, and
  yesterday's value on the other side of today's band.
- **Rule order.** Entries are read only when flat, exits only when holding,
  in the repository's order. A position closed by any rule needs a fresh
  crossing to re-open (X1, X2). X3 may re-enter at the next decision while
  VIX is still at or above 22.
- **Entry level.** The absolute stop is 25 S&P points on a ~2,470 future in
  the repository, about 1%, so 1% of NIFTY here.

Fixed, because the repository's code contradicts its own comments:
- The long absolute stop never fires: the code tests −price − entry > 25.
- The short absolute stop leaves the state machine "in a trade" with no
  position.
- X4 sums the long and short columns before forward-filling them, so the
  55-day-mean exit never applies. Each column is filled forward first.

Common to all four:
- **Instrument.** The near-month NIFTY future: the nearest monthly expiry
  after the session. The position is held through its expiry day and
  rolled at that close, charged as a sale and a purchase.
- **Rules versus P&L.** The rules (stops, targets, breakouts) read NIFTY 50's
  close, so a roll's basis jump never triggers a rule. P&L is the future's
  close to close, or its settlement price when it did not trade.
- **Fills.** A decision uses the session's close; the fill is the next
  session's close, as in round 1. The repository acts on the same close
  that makes the signal, which no one can do with a put-call ratio or TRIN
  that is known only after the market shuts.
- **Size.** Unlevered: at each entry the position's notional equals the
  strategy's equity, held in that quantity (whole lots are ignored; one
  NIFTY lot is about ₹16 lakh of notional). Returns are P&L over equity,
  with no interest on the collateral, which belongs to the book. A
  future's return is already in excess of cash, so rf = 0 is its excess
  Sharpe.
- **Costs.** `fno_costs` cost model 1, at each day's rates: brokerage, STT
  on the sell side, exchange charge, SEBI fee, stamp duty on the buy side,
  and GST. Slippage per side is max(1 tick, 1 bp of the price).
- **Data.**
  - The put-call ratio comes from the F&O store's options table.
  - TRIN and India VIX come from the equity stores, copied into the F&O
    store as `market.parquet`, as NIFTY's closes already are, so a Kaggle
    kernel needs that store alone.
  - India VIX is NSE's from May 2014 and Yahoo's before, from March 2008.
- **Runs.** On Kaggle, in `data/nse_engine/runs_options/`.

**Gate 1: each alone.** Excess Sharpe > 0 and deflated Sharpe ≥ 0.95 on daily
returns, rf = 0, at N = every options-family configuration on the window:
3 from round 1 plus these 4, so 7. Reported, not gating:
- CAGR, MaxDD and Calmar
- trades, win rate, average trade and time in the market
- costs
- the family's CSCV PBO

**Selection.** Of those passing gate 1, the highest Sharpe goes forward; ties
go X1, X2, X3, X4. If none passes, round 2 closes.

**Gate 2: the combined book.** As in § 5q: r = r(E4) + 0.25 × r(strategy)
each day, against E4's current run on today's equity data
(`20261008T080306839468Z_93cf6c4d`: Sharpe 1.205, CAGR 24.64%, MaxDD
−24.64%). All three must hold:
- CAGR at least 25.0%
- excess Sharpe (rf 6.5%) at least E4's
- MaxDD no more than 2 points deeper than E4's

Reported, not gating: the same overlay at 0.5, which is one NIFTY lot on the
₹30 lakh book; Calmar; Sharpe over 2017–25; and the deflated Sharpe at the
total trial count. The margin for 0.25 is about 3% of the book, inside
E4's idle cash.

**Reported, not gating.**
- The same four over 2007-01-02 to 2025-12-31 (E4's extended run
  `20261001T070903385179Z_93cf6c4d`). India VIX starts in March 2008, so
  X3 has no signal before then.
- For a strategy that passes both gates, its 2026 record to the last
  session.

**After the verdict.**
- **A pass:** the strategy goes to a paper book beside E4. That book is
  built then, since no recurring F&O paper book exists. It goes live only
  after its own forward evidence and your decision.
- **A fail:** round 2 closes without re-tuning; the code and inputs stay.

**Run note, 9 Oct 2026, 20:05 IST.** The first 2013–25 kernel started while
Kaggle was still creating the store version that holds `market.parquet`, so
it mounted the previous version and stopped at X2. Its one recorded run (X1)
was discarded unread and the kernel relaunched on the new version. The
2007–25 kernel had the new version from the start. All eight runs carry F&O
data hash `1dfa6257` and inputs hash `f349610b`, the local file's.

**Result, 9 Oct 2026: round 2 closed, no strategy passed gate 1.** Each alone,
2013–25, unlevered on the near-month future, rf 0, N = 7:

| Strategy | Trades | Win rate | Time in market | Sharpe | CAGR | MaxDD | Deflated Sharpe | Verdict |
|---|---|---|---|---|---|---|---|---|
| X1: PCR bands | 289 | 43% | 31% | −0.12 | −1.6% | −35.9% | 0.008 | FAIL |
| X2: TRIN bands | 325 | 52% | 19% | 0.67 | +4.8% | −16.0% | 0.671 | FAIL |
| X3: VIX ≥ 22 | 37 | 68% | 21% | 0.50 | +4.9% | −31.2% | 0.419 | FAIL |
| X4: 55-day breakout | 57 | 32% | 71% | 0.14 | +0.9% | −25.8% | 0.065 | FAIL |

Gate 2 was not run, and nothing entered the book's registry. For context
(not a gate): holding the near-month future throughout made Sharpe 0.48,
CAGR 6.6% and MaxDD −40.4% over the same years. Reported, 2007–25:

| Strategy | Sharpe | CAGR | MaxDD |
|---|---|---|---|
| X1 | 0.01 | −0.5% | −35.9% |
| X2 | 0.21 | +1.5% | −41.5% |
| X3 | 0.31 | +3.7% | −46.4% |
| X4 | −0.02 | −1.7% | −63.1% |

The options family's PBO over its 7 configurations is 29.9%. The best of
the family is still round 1's A1, with a deflated Sharpe of 0.865.

What the trades say:
- **X1** loses on its shorts: ₹−4.9 lakh over 126 trades. A put-call ratio
  falling through its lower band is no sell signal on NIFTY. Its longs make
  ₹1.1 lakh. It lost money in 8 of 13 years.
- **X2** is the only one with an edge on both sides:
  - ₹9.6 lakh on the longs and ₹7.1 lakh on the shorts
  - 3 losing years in 13
  - but 2020 alone gave ₹5.5 lakh, charges took ₹3.6 lakh, and the edge
    shrinks to Sharpe 0.21 once 2007–12 is included
- **X3** is buying the dip after fear spikes:
  - 37 trades, 25 at the +5% target
  - 2020 gave ₹7.2 lakh of its ₹17.4 lakh
  - over 2007–25 it carries 2008 (MaxDD −46%)
- **X4** wins on its longs and gives it back on the shorts: ₹−6.0 lakh over 22
  trades.

None comes near the book's bar (Sharpe 1.2) or an options-family deflated
Sharpe of 0.95. Not re-tuned: other bands, a long-only X1 or a cost filter
would be fitted to these years. The module
(`kite_connect/options/signal_futures.py`), the inputs (`market.parquet`) and
the Kaggle task stay, so a later round costs one run. The other ideas from
round 1, a protective bear put spread and weekly expiries, remain open
for your go/no-go.

## 5s. Options round 3: the F&O strategies of awesome-systematic-trading (O4, 9 Oct 2026)

Your request: compare `github.com/paperswithbacktest/awesome-systematic-trading`
with Centurion, test what is needed, and run it on Kaggle. Centurion
screened the same repository on 17 Sep 2026 (Stage F item 5):
- **Tested then and rejected:** every price-based stock strategy a long-only
  Indian book can use. Three passed the information screen (residual
  momentum, 52-week-high proximity, low beta) and each lowered the book's
  Sharpe. The rest failed the screen.
- **Not significant on NIFTY:** turn-of-the-month and payday.
- **Not tested then:** the files that needed futures or options data.

Since then:
- **The repository's 61 strategy files have not changed** (last change
  1 Aug 2026).
- **Its README's new table** lists the 61 strongest of 1,687 paper
  replications, gross of costs, on each paper's own window. Its code is on a
  paid site, and the table is mostly bonds, currencies, crypto and non-Indian
  equity. Nothing in it can be run here.
- **Centurion now has NSE's F&O archive** (round 1), with India VIX and
  breadth (round 2).

So the book is not re-tested: its one remaining configuration stays unspent.
Three of the 61 files had become testable, and they belong in the options
family (U32): two trade derivatives, and one holds the index only
overnight, beside the book.

**Pre-registered 9 Oct 2026, 20:47 IST, before any code for these three and
before any of their results.** The files' own parameters, no grid,
2013-01-01 to 2025-12-31; 2026 stays unseen until the verdict.

| | Y1: option-expiry week | Y2: volatility risk premium | Y3: overnight with sentiment |
|---|---|---|---|
| File | `option-expiration-week-effect.py` | `volatility-risk-premium-effect.py` | `market-sentiment-and-an-overnight-anomaly.py` |
| Position | long the near-month NIFTY future through the week of each monthly expiry | long NIFTY (near-month future) throughout, plus each cycle: short the ATM call and ATM put, long the put nearest 0.85 × spot | long NIFTYBEES from the close to the next open, 1/3 of capital for each condition met |
| Timing | from the close of the last session before the expiry's week to the close of the session before expiry day (the file holds Monday's open to expiry day's open) | options chosen the session after the last ones expire: the monthly expiry closest to 30 days, within 25–35; held to expiry | conditions: NIFTY 50 above its 20-session mean; India VIX below its 20-session mean |
| Size | notional = equity | index notional = option notional = equity | the exposure share of equity |

Carried over as written:
- **Y1:** the monthly expiry dates are NIFTY's.
- **Y2:** strikes come from that expiry's strikes that traded on the
  decision day. The ATM call and ATM put are each nearest the NIFTY close;
  the put is nearest 0.85 × that close. The file's ±20-strike filter is
  dropped: on SPY it usually cannot reach 15% below spot, which is the
  paper's put.
- **Y3:** the file's third condition (a US sentiment index) has no Indian
  series, and the file itself ran without it before 2018. So, as there, two
  conditions give 0, 1/3 or 2/3 of capital.

Translations:
- **Y1** is a calendar, so it fills on the planned closes.
- **Y2** decides on a close and fills at the next session's closes, with
  each option at its close, or at its settlement price when it did not
  trade, as in rounds 1 and 2.
  - Expiry settles on NIFTY 50's close, or failing that the expiring
    future's settlement price.
  - A long put that ends in the money pays STT on exercise.
- **Y3:**
  - The conditions use the previous session's closes, as the file uses
    15:44 values to place a close order.
  - It buys at NSE's closing price (the last half-hour's average) and sells
    at the opening price (the pre-open auction).
  - Each held night is charged 6.5% / 252, so its returns are in excess of
    cash like the futures' returns.

Common to all three:
- **Futures and options** as in round 2: the near-month future, rolled on
  expiry day's close; `fno_costs` cost model 1; slippage per side of max(1
  tick, 1 bp) on futures and max(1 tick, 0.5% of premium) on options (§ 5q).
- **NIFTYBEES** pays the book's own cost model 3: ETF STT, stamp duty,
  exchange charge, SEBI fee, GST and the depository charge on every sale,
  plus square-root impact on its 20-session median traded value.
- **Capital** ₹7.5 lakh, the 0.25 overlay's share of the ₹30 lakh book;
  unlevered, with whole lots ignored.
- **Data** from the F&O store: NIFTYBEES's open, close and traded value are
  added to `market.parquet` from the equity stores.
- **Runs** on Kaggle, in `data/nse_engine/runs_options/`.

**Gates.** As in § 5r: gate 1 is excess Sharpe > 0 and deflated Sharpe ≥ 0.95
at N = every options-family configuration on the window, that is 7 plus
these 3, so 10. The highest Sharpe that passes goes to gate 2, E4 + 0.25 ×
r(strategy) against E4's run `20261008T080306839468Z_93cf6c4d`:
- CAGR at least 25%
- excess Sharpe at least E4's
- MaxDD no more than 2 points deeper

Reported, not gating:
- the 2007–25 window (India VIX from March 2008)
- for a strategy that passes both gates, its 2026 record

**A pass** goes to a paper book beside E4, built then. **A fail** closes
round 3 without re-tuning.

**Amendment, 9 Oct 2026, 20:50 IST, before any code for these three and
before any result.** NIFTYBEES's opening prices in the bhavcopy are not
prices anyone could have traded at:
- **Stray first trades:** over a hundred nights show a gap beyond ±8%,
  some of them ±20%. This is the stray-ETF-open problem that cost model 3
  already handles.
- **An unadjusted split:** the 1:10 split of 19 Dec 2019 shows as a −90%
  night, and NSE's previous close does not adjust it.

Y3 therefore uses the NIFTY 50 index's own open and close, from NSE's daily
index files (computed from every constituent's opening auction; no stale
or stray opens since February 2012). It is traded as the near-month
NIFTY future: futures charges and 1 bp slippage on each side, the held
night charged 6.5% / 252 as above. NSE's index files start on 21 Feb 2012,
so Y3's 2007–25 run starts there. Nothing else changes.

Settled while writing the code, before any result:
- A session without an India VIX close keeps the last one. Otherwise each
  gap would blank the 20-session mean, and Y3's VIX condition with it, for
  four weeks.
- Y2's future is resized to equity at each monthly roll, as the file buys the
  index afresh each cycle.

**Code fix after the first result, 9 Oct 2026, 22:13 IST.** The first
evaluation (Kaggle runs of 21:00) showed Y2 with 13 cycles in 13 years. The
rule implies about 150.

The cause was in the code, not the strategy:
- NSE moved the April 2014 monthly expiry off a market holiday, 24 April
  (Mumbai voted in the general election that day), to 23 April.
- The simulator settled options only on a session dated exactly the label.
  It waited for a 24 April session that never came, held the dead legs to
  2025 and opened no further cycle.

Nine of the 237 monthly expiry labels since 2007 did not settle on their
date:

| Label | What happened |
|---|---|
| 27 Nov 2008 | settled 28 Nov (the Mumbai attacks) |
| 25 Dec 2008 | settled the 24th |
| 30 Apr 2009 | settled the 29th |
| 27 Feb 2014 | relabelled 26 Feb in December 2013 |
| 24 Apr 2014 | settled the 23rd |
| 29 Mar 2018 | relabelled the 28th |
| 30 Mar 2023 | relabelled the 29th |
| 29 Jun 2023 | relabelled the 28th on the 28th (settled the 28th; corrected 10 Oct 2026, LN-T20) |
| 31 Mar 2026 | relabelled the 30th |

The fix (`settlement_sessions`):
- A contract settles on the session its contracts last traded.
- A label whose contracts stopped trading more than a week before its date
  was relabelled; it is dropped, and its contracts carry on under the new
  label.
- Y2's future rolls on that session too, and Y1's expiry week follows it.

Effects:
- **Y1** changes only in those months. **Y3** uses no expiries and is unchanged.
- **Configurations:** no parameter changed, so the configurations and N are
  unchanged. Y1 and Y2 run again on Kaggle. The first runs stay in the
  registry, which reads the latest run of each configuration.
- **Round 2:** its strategies reproduced exactly then; that was wrong for
  two of them (corrected 10 Oct 2026, below).
- **Round 1:** its harness settles on the first session on or after the
  label, so it never stalled. But it settled a moved expiry up to two
  sessions late, at that later close, and skipped a relabelled month
  (February 2014). That touches about 3 of its 156 months. Its verdict, a
  best deflated Sharpe of 0.865 against 0.95, is not re-run.

**Result, 9 Oct 2026: round 3 closed, no strategy passed gate 1.**
- **Runs.** Y1 and Y2 come from the corrected runs (Kaggle, 22:14 IST). Y3's
  run is unchanged.
- **Data.** F&O data hash `1dfa6257`, inputs `e41e358b`.
- **Basis.** Each strategy alone, 2013–25, ₹7.5 lakh, rf = 0 (Y3 in excess
  of cash), N = 10.

| Strategy | Trades | Win rate | Time in market | Sharpe | CAGR | MaxDD | Deflated Sharpe | Verdict |
|---|---|---|---|---|---|---|---|---|
| Y1: option-expiry week | 156 | 48% | 13% | −0.16 | −1.2% | −26.1% | 0.001 | FAIL |
| Y2: volatility risk premium | 147 cycles | 73% | 100% | 0.34 | +4.6% | −47.8% | 0.101 | FAIL |
| Y3: overnight with sentiment | 2,414 nights | 58% | 75% | 1.04 | +3.8% | −10.4% | 0.884 | FAIL |

Gate 2 was not run, and nothing entered the book's registry. The
options-family PBO over its 10 configurations is 22.2%. Reported, 2007–25:
Y1 −0.23, Y2 0.33 (MaxDD −47.8%), Y3 0.91 (from February 2012). For
context, holding the near-month future made Sharpe 0.48 over 2013–25.

What the runs say:
- **Y1:** the option-expiry week has no edge on NIFTY.
- **Y2:** writing the monthly at-the-money straddle on top of a long future
  does worse than the future alone: Sharpe 0.34 against 0.48, MaxDD −47.8%
  against −40.4%. The 15%-out-of-the-money put barely caps a month like
  March 2020, and the straddle gives up the rallies.
- **Y3** is the strongest result of all three options rounds: NIFTY's
  overnight drift, taken only while the index is above its 20-session mean
  and India VIX below its own. But it does not clear the bar, and its numbers
  are optimistic:
  - Fills: it fills at NSE's closing price (a half-hour average) and opening
    price (the auction), which a futures order matches only approximately.
  - Costs: 2,414 nightly round trips paid ₹6.0 lakh of charges on ₹7.5 lakh,
    about 6 points a year.

Not re-tuned. A test of Y3 on the future's own opening and closing prices
would be a new configuration in a later round. The bhavcopy carries futures
opens, but the F&O store does not keep them yet.

**Settlement re-baseline, 10 Oct 2026 (tracker LN-T20).** The fix above
followed a relabel only in round 3's code, and only when the old label's
rows stopped more than a week early; a relabel inside that week (29 Jun 2023
to 28 Jun) settled a day early, and rounds 1 and 2 never used the map. Now
one settlement source serves every options backtest:
- `settlement_sessions` follows a relabel when a label of the same month
  starts the very next session (recursively: 26 Mar 2026 to 31 Mar to
  30 Mar), and leaves out a label still trading at the data's end.
- Rounds 1, 2 and 3 key every contract by the session it settles on
  (`rekey_to_settlement`): round 1 settles and quotes a moved month on its
  real day, round 2's future rolls on it and no longer carries a stale mark
  across a relabel, and Y2 settles June 2023 on the 28th (18,972.10, not the
  27th's close).
- Round 1 values an untraded leg on its own settlement session at intrinsic
  against the exact index close, never the settle column, which has held the
  index level that day since 2020.

Re-run on Kaggle (2013–25 and 2007–25), same configurations, so N stays 10;
a data re-baseline, as on 9 Oct. Sharpe, 2013–25, each alone:

| | A1 | A2 | B | X3 | X4 | Y2 |
|---|---|---|---|---|---|---|
| Before | 0.816 | −0.273 | 0.256 | 0.502 | 0.139 | 0.317 |
| After | 0.828 | −0.182 | 0.259 | 0.494 | 0.144 | 0.317 |

A2's phantom −26.3% day (30 Jun 2023: the month settled a session late,
against that day's close) is gone: 0% that day, −3.5% on the 28th. X1, X2, Y1 and Y3 reproduce
exactly. Every gate-1 verdict stands: no strategy passes (A1's deflated
Sharpe 0.897 on round 1's three sleeves, 0.696 at N = 10).

## 5t. Idle cash earns nothing: cost model 4 (IC1, 10 Oct 2026)

Cost models 1–3 credited idle cash with 6% a year (`cash_yield_annual`) in
the backtest and in the benchmarks. A Kite account pays no interest on idle
cash, and neither the paper nor the live book earns it. On the deployed
book's 27 sessions spent wholly in cash, the model-3 run gains exactly
6%/252 a day.

**Fixed 10 Oct 2026, before any run.** Cost model 4 credits idle cash with
`IDLE_CASH_YIELD_ANNUAL` = 0 (`nse_engine/costs.py`), in the engine and in
the benchmarks' cash leg. `cash_yield_annual` stays in `EngineConfig` at
0.06, unused, so that no configuration hash changes. The same 27 sessions
now gain exactly 0. Registry comparisons stay within one cost model.

**Re-run.** All 61 same-window configurations re-ran in one Kaggle job on
fingerprint ba7c0982 (no new trials), and Kaggle reproduced the local runs.
The three books, model 3 → model 4:

| Book | Sharpe | CAGR | MaxDD | Calmar | DSR |
|---|---|---|---|---|---|
| Deployed 679cbd0c | 1.206 → 1.158 | 23.74% → 22.97% | −23.1% → −23.9% | 1.03 → 0.96 | 0.989 → 0.965 |
| Candidate 2d64ba4c | 1.213 → 1.132 | 22.56% → 21.38% | −23.0% → −24.3% | 0.98 → 0.88 | 0.989 → 0.958 |
| E4 93cf6c4d | 1.205 → 1.161 | 24.64% → 23.90% | −24.6% → −25.7% | 1.00 → 0.93 | 0.988 → 0.966 |

No configuration now clears Sharpe 1.2 (the best is 1.194, the median 0.993).
PBO over the 61 rises from 57.8% to 60.4%. The benchmark gate passes for all
three books. With the D2 haircut (one-day lag, twice the impact):

| Book | Haircut Sharpe / CAGR / MaxDD | Expected OOS Sharpe |
|---|---|---|
| Deployed | 0.973 / 19.7% / −23.9% | 0.82 |
| Candidate | 0.948 / 18.5% / −27.3% | 0.80 |
| E4 | 1.040 / 21.6% / −27.7% | 0.87 |

**Walk-forward (R12 re-run on Kaggle, r12a4 and r12b4).** Out-of-sample
2017–25, model 3 → model 4:

| Arm | OOS Sharpe | CAGR | MaxDD |
|---|---|---|---|
| A: K5's 32-point grid (deployed, candidate) | 1.150 → 1.227 | 20.9% → 22.9% | −20.9% → −22.1% |
| B: the same grid × exit rank {40, 60} (E4) | 1.273 → 0.979 | 21.9% → 18.4% | −17.2% → −22.1% |

The two arms move apart because of what the folds select, not because the
strategy changed. Without the cash credit, the in-sample fits prefer staying
invested: neutral scale 1.0 over 0.6, a 5-day rebalance over 21, and 20
positions over 30. Those picks helped arm A out of sample and hurt arm B.
E4's own configuration loses only 0.04 Sharpe over 2017–25 (1.478 → 1.440).
The store refresh changed none of E4's returns. The re-run also used
Kaggle's newer image (Python 3.13, pandas 3.0.6). The scorecards now read
these walk-forwards: deployed and candidate pass the walk-forward rule
(1.23 ≥ 1.2) and E4 fails it (0.98). No book passes net Sharpe or Calmar.

**Options gates.** Gate 2's base is now E4 under model 4: run
20261009T185901820673Z_93cf6c4d for 2013–25, and
20261009T191853176431Z_93cf6c4d for the reported 2007–25 view. The 2007–25
run was recorded locally, not on Kaggle: Sharpe 0.973 → 0.913, CAGR 21.11%
→ 20.03%, MaxDD −39.1% → −40.8%. O2, O3 and O4 keep their verdicts. All
three closed at gate 1, which does not read E4, and none recorded a
combined run. A combined run now refuses a base from another cost model.
A round 4 (O5) starts from E4's model-4 CAGR of 23.90%, so a sleeve must
now add 1.1 points, not 0.4, to reach gate 2's 25% bar.

## 5u. Data re-baseline: special sessions, point-in-time names, excess-basis PBO (LN-T7, T8, T15, T16; 10 Oct 2026)

Four corrections from the Lean review (`docs/lean_review.md`), adopted on
principle before any result was read. None is a configuration change, so no
trial is spent; all 61 same-window configurations were re-run in one Kaggle
job (RR2), as for cost models 2-4.

**The Budget session of Sunday 1 Feb 2026 (LN-T7).** The archive probed
weekdays, Saturdays and four known Sundays, so NSE's full Sunday session for
the Union Budget (circular CM 11/2026) was missing: on 2 Feb, 2,082 stocks'
prev_close disagreed with their last close. The store now probes every day
and flags such a date at build time. With the session backfilled, the
deployed book's 2026 holdout (to 11 Sep) moves from Sharpe 0.49 / CAGR 15.1% /
MaxDD −18.7% to 0.58 / 17.5% / −17.4%: 4 stop exits move from the 2 Feb open
to the 1 Feb bar and 4 more stops fire that day. The 2013-25 window has no
missing session (only NSE's 24 Jun 2006 Saturday, which no archive has).

**The as-printed price filter in paper and live (LN-T8).** `MarketData.until()`
dropped `close_unadj`, so the candidate and E4 books, which filter on
as-printed prices, planned their universe history on adjusted prices. To 1
Oct 2025, 2,394 of 3,410 sessions of that history differed from what their
registered runs used; since the fix, none. Backtests never cut the panel, so
no recorded run changes; the two paper books plan as validated from the
merge.

**Ties broken by the name in force (LN-T15).** Forecasts are capped, and on a
quarter of sessions more than 20 names share the cap; ties were broken
alphabetically by each stock's latest name. A rename after the window
(about five a month in 2026) therefore reordered decisions inside it, and
moved the data hash. Ties now break by the name the stock traded under on
the decision date (`MarketData.trade_names`, 352 renamed columns), and the
data hash (version 2) is keyed on the names in force at the panel's end and
widened to every price frame, the as-printed close, volume and the ETF set.
Relabel gate, run before the change shipped: with the store's rename table
cut at 2025-12-31 (renames after the window hidden), all 61 configurations
give the same 2013-25 returns (to 1e-12) and the same hash. Data hash
`ba7c098240b4c9ec` → `dab49d3cbd323c16`.

**PBO on excess returns (LN-T16).** CSCV ranked trials on raw returns while
the deflated Sharpe, the walk-forward and every pass rule use returns over
6.5%; it now subtracts the risk-free rate first.

**Result (RR2, Kaggle = local to the last digit).**

| Book | Sharpe | CAGR | MaxDD | DSR |
|---|---|---|---|---|
| Deployed 679cbd0c | 1.158 → 1.163 | 22.97% → 23.13% | −23.9% → −23.9% | 0.964 |
| Candidate 2d64ba4c | 1.132 → 1.157 | 21.38% → 21.88% | −24.3% → −23.6% | 0.963 |
| E4 93cf6c4d | 1.161 → 1.139 | 23.90% → 23.55% | −25.7% → −25.3% | 0.957 |

Across the 61 configurations Sharpe moves by +0.002 on average (−0.023 to
+0.027; Spearman 0.990). PBO over the 61: 37.2% → 29.2% on the excess basis
("likely real"; probability of an out-of-sample loss 0.7%), 60.4% → 51.2% on
the raw basis. One configuration now clears Sharpe 1.2: `0381af7f`, the R11
crash re-entry rule (1.194 → 1.220), which failed its own pre-registered
rule on 28 Sep (§5f) and stays closed. Benchmark gate passed for all three
books. Haircut (one-day lag, twice the impact, CSCV selection ×0.82):
expected OOS Sharpe deployed 0.80, candidate 0.79, E4 0.85.

Walk-forwards (R12 re-run on Kaggle, r12a5 and r12b5), out of sample 2017-25:

| Arm | OOS Sharpe | CAGR | MaxDD |
|---|---|---|---|
| A: K5's 32-point grid (deployed, candidate) | 1.227 → 1.238 | 22.85% → 23.07% | −22.1% → −22.1% |
| B: the same grid × exit rank {40, 60} (E4) | 0.979 → 1.107 | 18.39% → 20.40% | −22.1% → −21.9% |

Arm A picks the same settings in every fold. Arm B's picks change in 2019,
2020, 2022 and 2024: its in-sample fits sit close together, so a re-ordered
tie can tip a fold. The scorecards read these runs. Deployed and candidate
pass the walk-forward rule (1.24 ≥ 1.2) and E4 misses it (1.11). No book
passes net Sharpe (1.16, 1.16, 1.14) or Calmar (0.97, 0.93, 0.93).

## 5v. Options research pipeline (O5 onward, the owner's directive, 10 Oct 2026)

The general research rules (families, budgets, pre-registration, gates)
live in `docs/research_protocol.md` (GT10); this section is the options
family's part of them.

**Directive.** Options strategies are researched continuously until ones that
make money consistently are onboarded. The owner supplies sources: PDFs,
GitHub repositories, research papers. Each is studied, implemented, tested
and, if it passes, paper-traded. The objective is that options trading adds
to the book's CAGR, alpha and Sharpe. Written before any O5 result.

**Per source (one round each, labelled O5, O6, …):**
1. **Intake note:**
   - what the source claims and the evidence behind it;
   - its rules, instruments, holding period and capital per position;
   - the data it needs.
   A rule that needs intraday prices or a same-day exit is out of scope:
   Centurion is swing and positional only, and the F&O store holds
   end-of-day bhavcopies. Anything that needs data the store lacks is logged
   with the data that would unlock it.
2. **Pre-registration:** at most 3 configurations, with parameters taken from
   the source, not fitted. Pass rules and costs (`fno_costs`, the current F&O
   cost model) are written before any run.
3. **Backtest.**
   - 2013–25 on Kaggle (one environment), recorded in the options registry
     (U32), every configuration counted in the options family's N.
   - The 2007–25 window is reported, never gating.
   - Settlement and IVs come from LN-T20 and LN-T27.
4. **Gate 1, unchanged:** each configuration alone has excess Sharpe > 0 and
   a deflated Sharpe ≥ 0.95 at the options family's N, and the family's PBO
   stays ≤ 30%.
5. **Gate 2, revised from O5 to match the objective.**
   - The base is the live book's recorded run (the deployed book today; E4
     if V3 promotes it), overlaid with 0.25 × the strategy, as in 5q.
   - The combined book must **raise CAGR, excess Sharpe and annual alpha
     against NIFTY 50 TRI** over the base, with MaxDD no more than 2 points
     worse.
   - The combined book's deflated Sharpe at the total trial count (book plus
     options) is reported.
   - This replaces the fixed 25% CAGR bar, which E4's model-4 CAGR had moved
     to "+1.5 points". Rounds O2–O4 keep their own gates.
6. **Paper trading:**
   - A passing strategy runs in an options paper book, decided after the
     close and filled by the strategy's own rule, with G4-style tracking
     against its backtest replay.
   - It runs for at least 60 sessions and at least 6 cycles of the
     strategy.
   - The scheduled paper runner is built when the first strategy passes, not
     before.
7. **Onboarding:**
   - Live at small size, under the kill switch and the live risk limits.
   - The order route is the typed-confirmation toolkit, or an automated
     route if the owner decides so at that point.
   - Sized in lots: one NIFTY lot is about Rs 15 lakh of notional.

**What carries over from rounds O2–O4:**
- Option premium selling and index-timing rules have failed after costs.
- India's F&O STT reportedly rose from 1 Apr 2026 (to be confirmed in the
  cost model before O5's first run).
- Each configuration raises N, so a strong source should be tested with one
  or two configurations, not three.

## 5w. Pre-announcement drift on NSE (GT13; pre-registered 10 Oct 2026, 22:11 IST)

**Source.** Lucca & Moench, "The Pre-FOMC Announcement Drift" (JF 2015).
From 1994 to 2011 the S&P 500 earned about 80% of its excess return over
cash in the 24 hours before the eight scheduled FOMC statements a year, and
nothing on the news itself. The effect has largely faded since 2011, except
before meetings with a press conference. It is reported gross of costs.

**Question.** Does Indian equity drift up while it waits for a scheduled
policy announcement, enough to pay for a NIFTYBEES round trip? The windows
are chosen so that Centurion's after-market orders can trade them, with no
intraday order.

**Hypotheses**, fixed before any number is read, scheduled announcements
only (an unscheduled one cannot be waited for):
- **H1, RBI** (announcements land during the session, 10:00–12:00 IST): long
  from the **open of the session before the announcement day to the open of
  the announcement day**. A buy AMO and a sell AMO, both filled at the open,
  hold exactly the waiting stretch.
- **H2, FOMC** (the statement lands after the Indian close, 23:30 or 00:30
  IST): the Indian session on or before the US decision date is all waiting.
  - Tradable window: **that session's open to the next session's open**. It
    includes the reaction, which the paper finds flat on average.
  - Reported only: the pure waiting window, previous close to that close.

**Data.**
- NIFTY 50 index opens and closes (gross).
- NIFTYBEES opens and closes, net of a round trip at Rs 6 lakh: statutory
  charges from `nse_engine.costs` with ETF STT, plus the impact model.
- Excess over cash at 6.5% a year.
- 2013-01-01 to 2025-12-31.

**Pass rule, per hypothesis**, which makes it a candidate sleeve for idle
cash, to paper-trade under GT10:
- NIFTYBEES mean excess return per window, net of costs, is above 0 with
  t ≥ 2.0. The effect is published, so this is a replication, not a
  discovery.
- The net mean is positive in both 2013–19 and 2020–25.
- Two hypotheses, N = 2, both reported.

**Reported, never gating:**
- the share of NIFTY's excess return earned inside the windows (the "80%");
- the fraction of sessions they occupy;
- the hit rate;
- the move after the announcement (RBI: announcement-day open to close;
  FOMC: next session open to close);
- 2026 to date, as a holdout opened once.

**Amendment after the first run (10 Oct 2026, between 22:11 and 23:17 IST).**
- NIFTYBEES's bhavcopy opens are stray first prints (tracker D4). Its open
  gap differs from the index's by a median of 32 bp, and by more than 50 bp
  on 36% of days.
- Its traded value grew from about Rs 2.6 crore a day (2013) to Rs 150 crore
  (2025), so a cost averaged over the whole period overstates today's.
- So the tradable estimate became the **index window less a NIFTYBEES round
  trip at its latest year's liquidity**: 18.5 bp, statutory charges plus the
  conservative stock-calibrated impact model. Round 3 did the same for Y3.
- The verdict is the same on every basis: the index fails t ≥ 2 before any
  cost.

**Result, 10 Oct 2026: both hypotheses FAIL; no sleeve.**

| 2013–25 | Events | NIFTY 50 gross per window | Net of the round trip | Halves (net, 2013–19 / 2020–25) | Share of NIFTY's excess return | After the announcement |
|---|---|---|---|---|---|---|
| H1 RBI | 78 | +10.8 bp, t 1.16, 56% up | −7.7 bp, t −0.83 | −11.6 / −2.7 bp | 12.8% in 2.4% of sessions | −13.9 bp |
| H2 FOMC | 103 | +8.4 bp, t 0.91, 50% up | −10.1 bp, t −1.10 | −9.4 / −11.0 bp | 12.9% in 3.2% of sessions | −12.0 bp |

What the result says:
- **The direction matches the paper.** Windows that wait on the RBI or the
  Fed earn about 4–5× their share of time, and the market gives some back
  after the news.
- **It is not significant, even before costs, and nowhere near the 80% of
  the US 1994–2011 sample.**
- **FOMC's pure waiting window** (previous close to that session's close)
  is the most consistent: +11.4 bp, t 1.61. Trading it needs orders at the
  close, which Centurion's after-market orders do not place, and it is still
  under the cost.
- **2026 (holdout, opened once):** RBI +97 bp over 5 events; FOMC
  tradable −27 bp, pure window +70 bp over 6 events, all up. Too few events
  to read.
- **Not re-tuned:** any other window would be a new trial. The module
  (`nse_engine/event_drift.py`) stays as the tool for re-checking with new
  data.

## 5x. Options rounds O5–O8 from the owner's books (pre-registered 10 Oct 2026, 23:17 IST)

**Sources.** Five books, placed in `research_inbox/` on 10 Oct 2026 and read
in full by intake agents:
- Cordier & Gross, *The Complete Guide to Option Selling* (O5).
- Natenberg, *Option Volatility and Pricing* (O6).
- Jabbour & Budwick, *The Option Trader Handbook* (O7).
- Ursone, *How to Calculate Options Prices and Their Greeks* (O8).
- Minervini, *Trade Like a Stock Market Wizard* (O9). It is an equity
  method, not options; it is studied separately (below).

None of the books contains a backtest. Each round follows plan 5v.

**Common to every configuration:**
- **Instrument and data:** NIFTY monthly options, 2013-01-01 to 2025-12-31,
  every contract keyed by its settlement session (LN-T20).
- **Timing:** decided at the close, filled at the next close. A leg fills at
  its close when it traded, else its settlement price, with `fno_costs`
  slippage and charges at the dated rates (2026 STT included). Legs settle
  at intrinsic against the exact index close; long ITM legs pay exercise
  STT.
- **Greeks:** Black-76 on the expiry's futures close, else spot carried at
  the risk-free rate. An ITM strike takes its OTM mirror's IV.
- **Entries:** only on strikes that traded that day.
- **Capital and sizing:** Rs 7.5 lakh, sized in lots so the position's
  maximum loss (or, for a naked short, 10% of notional as an assumed margin)
  stays within 50% of equity.
- **Monthly decision date:** the first session after a monthly expiry (one
  entry per month; positions may overlap).
- **Regime label:** Centurion's own, from data up to that close. Bull is
  NIFTY above its 200-day mean and up over 63 sessions; bear is below and
  down; range is otherwise.
- **IV labels:** IV rank is the 30-day ATM IV's percentile over 252
  sessions.
- **Volatility forecast:** σF = 0.40·HV30 + 0.30·HV60 + 0.20·HV130 +
  0.10·HV260, zero-mean close-to-close (Natenberg's 6/12/26/52-week blend).
  σLR is the expanding mean of HV30.
- **The "2–6 months" translation:** every book's 2–6-month horizon becomes
  the **second monthly expiry (≈ 55–60 days)**. Third-month NIFTY strikes
  3–10% out of the money traded on only 12–25% of entry days from 2018, so
  the books' own liquidity rule rules them out.

### O5 (Cordier & Gross): 2 configurations

**O5-A, trend-side credit spread** (the book's S4 with the 200% and 10%
rules).
- **Direction:** bull regime → bull put spread; bear regime → bear call
  spread; range → no trade.
- **Expiry:** the second monthly.
- **Short leg:** the traded strike with the largest |Δ| ≤ 0.20.
- **Long leg:** the first further-OTM traded strike whose premium is ≤ 1/3
  of the short's.
- **Entry filter:** credit ≥ 10% of (width − credit).
- **Exits:** close when the spread's value is ≥ 2× the entry credit (the
  200% rule), or ≤ 10% of it (the 10% rule). Otherwise settle at expiry.

**O5-B, trend-side naked short** (S1). Same direction, expiry, strike rule
and exits, on the option alone. Margin is assumed at 10% of notional.

### O6 (Natenberg): 2 configurations

**O6-A, long at-the-forward calendar.**
- **Legs:** sell the front-month call at the strike nearest the front
  forward; buy the second-month call at the same strike.
- **Entry:** only if forward volatility from front to back is ≤ σLR − 2
  points, and the front ATM IV is ≥ σF.
- **Hedge:** front futures whenever |net Δ| ≥ 0.5 lot.
- **Exits:** at the session before the front expiry, or earlier if forward
  volatility rises to ≥ σLR.

**O6-B, conditional short iron butterfly** (the book's Spread 3; a
falsification test of the failed short-vol result).
- **Expiry and legs:** the second monthly. Sell the at-the-forward straddle;
  buy wings at ±0.6 SD (SD = F·IV·√T, the nearest traded strikes beyond).
- **Entry:** only if ATM IV is ≥ σF + 2 points and no Budget or election
  date (`sleeves.EVENT_DATES`) falls before expiry.
- **Size:** scaled by the margin, ×min(1, margin / 7 points).
- **Hedge:** 0.5-lot futures band.
- **Exits:** when ATM IV falls to ≤ σF, or at 7 days to expiry.

### O7 (Jabbour & Budwick): 2 configurations

**O7-A, regime debit sleeve, no adjustments.** One "strike step" w = 5% of
spot, rounded to 100 points; that is the book's own range of strike gaps.

| Regime | IV rank | Position |
|---|---|---|
| Bull | < 33 | Long ATM call, second monthly |
| Bull | ≥ 33 | Bull call spread K / K+w, second monthly |
| Bear | < 33 | Long ATM put, second monthly |
| Bear | ≥ 33 | Bear put spread K / K−w, second monthly |
| Range | > 67 | Long ATM call butterfly ±w, front month |
| Range | otherwise | Flat. The book's calendar needs a 3-month long leg, which is illiquid. |

**Exits:**
- single long options at 30 days to expiry;
- other structures at the session before expiry;
- any position at a decision date whose regime label has changed.

**O7-B, the same plus the book's adjustments.**
- **Long-option lock-in:** sell C(K+w) or P(K−w) once its price is ≥ the
  entry debit.
- **Butterfly to condor:** add the adjacent butterfly, once, when spot closes
  beyond a breakeven with ≥ 7 days left.

### O8 (Ursone): 1 configuration

**O8-A, post-spike term-structure calendar straddle.**
- **Entry:** near-month ATM IV minus next-month ATM IV ≥ 3 points, IV rank ≥
  70, and the front expiry 14–30 days away.
- **Legs:** sell the front at-the-forward straddle; buy the second-month
  straddle at the same strike, in equal lots, only for a net debit.
- **Exits:**
  - at the session before the front expiry;
  - when the term spread falls to ≤ 0;
  - when spot moves more than one front straddle (its entry price) from the
    strike.

**Clarification (10 Oct 2026, after a 2014-only debugging run, before any
recorded run).**
- **The problem:** the engine sized each new position at 50% of equity.
  Positions overlap month to month, so total risk could exceed the cap: the
  long-option sleeve reached a −69% drawdown in 2014.
- **The fix:** the 50% applies to **the maximum loss of all open positions
  together**. That is the book's own rule (Cordier & Gross: total margin
  ≤ 50% of capital), and the only reading under which overlapping positions
  respect the cap.
- **Not changed:** no other parameter.

### Gates and what follows

- **Gates:** plan 5v. Gate 1: deflated Sharpe ≥ 0.95 at the options family's
  N, which these 7 take from 10 to 17. Gate 2: beside the live book, the
  strategy raises CAGR, Sharpe and alpha, with MaxDD no more than 2 points
  worse.
- **2007–25** is reported, never gating.
- **Not re-tuned:** no parameter above changes after a result.
- **Bug fixes** are logged with their date.

### Results, 10 Oct 2026: O5–O8 all closed; nothing to paper-trade

**Runs.** Kaggle, 2013–25, recorded in the options registry, which now
holds N = 17; the options family's PBO is 14.1%.
- **Gate 1:** none clears it.
- **Gate 2:** every overlay lowers the book's Sharpe from 1.163.
- **Base for gate 2:** the deployed book, CAGR 23.1%, Sharpe 1.163, MaxDD
  −23.9%, alpha 12.6%.

| Configuration | Positions | Sharpe | CAGR | MaxDD | DSR | Book + 0.25×: Sharpe / MaxDD |
|---|---|---|---|---|---|---|
| O5-A trend credit spread | 59 | 0.25 | 2.4% | −31.4% | 0.04 | 1.150 / −26.7% |
| O5-B trend naked short | 87 | 0.26 | 2.3% | −31.2% | 0.04 | 1.129 / −25.8% |
| O6-A long calendar | 49 | 0.24 | 0.2% | −66.2% | 0.03 | 1.009 / −23.2% |
| O6-B conditional iron fly | 18 | 0.20 | 1.8% | −44.2% | 0.02 | 1.146 / −30.3% |
| O7-A regime debit sleeve | 19 | −0.54 | −30.0% | −99.0% | 0.00 | 0.492 / −45.8% |
| O7-B with adjustments | 19 | −0.44 | −32.5% | −99.4% | 0.00 | 0.477 / −50.1% |
| O8-A term-structure calendar straddle | 34 | 0.06 | −3.1% | −65.1% | 0.01 | 0.981 / −29.4% |

**What the runs say:**
- **Cordier & Gross:** the trend-side 200% and 10% rules make about
  2% a year on NIFTY. The naked version is no better, and their own caveat
  about index options holds.
- **Natenberg:** conditioning on volatility forecasts does not rescue
  short volatility (O6-B), and the cheap-forward calendar earns nothing
  (O6-A).
- **Jabbour & Budwick:** buying premium by outlook loses steadily. The
  adjustments change almost nothing: −32.5% against −30.0%.
- **Ursone:** the post-spike calendar is flat to negative.
- **The edge:** none of the books' structures adds an edge after costs
  on NIFTY end-of-day data.

**Diagnostics** (reported, not trials):
- **Sizing is not the cause.** At a 10% risk budget instead of 50% (local
  runs), Sharpe stays between −0.33 and 0.20, and CAGR is at most 0.6%.
  O7's −99% is its 50%-of-equity debit sizing compounding a negative
  edge; at 10% it is still −72%.
- **Stale back-month prints.** The second-month leg of a calendar trades
  thinly, and its bhavcopy close is often a stale print. On 14 Dec 2017
  O6-A's January call marked 251.0 while spot rose, then 337.75 the next
  day, a ±30% swing in sleeve equity at this size. That inflates the
  multi-expiry configurations' volatility and drawdown, but not their
  return: O6-A earns about 0% a year however it is marked.
- **Gate 2 can be fooled by mark noise.** Noise that swings −30% then
  +30% has a positive arithmetic mean, so a 0.25 overlay of it raises the
  book's measured CAGR and alpha with no real profit: O6-A shows 25.8% and
  15.4%. Every overlay still lowers Sharpe, so no verdict changes.
  - **Before the next multi-expiry round:** mark an untraded or thin leg
    at a model price from its own expiry's smile (OP1).

**2007–25, reported only** (Kaggle, recorded, never gating): no better.

| Configuration | Sharpe | CAGR | MaxDD |
|---|---|---|---|
| O5-A | 0.09 | 0.0% | −49.4% |
| O5-B | 0.28 | 2.8% | −39.3% |
| O6-A | 0.17 | −2.5% | −77.7% |
| O6-B | 0.10 | −2.3% | −68.4% |
| O7-A | 0.09 | −23.8% | −99.6% |
| O7-B | −0.37 | equity below zero after 2 positions | — |
| O8-A | −0.01 | −9.7% | −92.1% |

**O7-B's negative equity is an engine flaw.** Its adjustments buy
additional debit without checking available cash, which a broker would
refuse; folded into OP1. The verdict does not depend on it: O7-A, with no
adjustments, loses as well.


### O9 (Minervini): an equity sleeve, separately (O9-A pre-registered 10 Oct 2026, 23:25 IST)

**Status.** The SEPA breakout sleeve is its own family:
- registry `data/nse_engine/runs_sepa`, so it never enters the core's
  trial matrix;
- 1 configuration;
- book numbers, with my translations marked (A).

Its filter on the core and the risk-rule tests (a 10% max-loss cap, a
breakeven move, progressive exposure, no topping up losers) belong to the
core family, after V3. The earnings screen waits for GT1.

**O9-A, SEPA-lite breakout sleeve.** 2013-01-01 to 2025-12-31, adjusted
store prices, the core's universe (top 300 by liquidity, refreshed every 21
sessions).

**Trend Template** (all eight, at the close):
- close above both the 150- and 200-day means; the 150-day above the
  200-day;
- the 200-day mean higher than 21 sessions earlier;
- the 50-day mean above the 150- and 200-day means; close above the 50-day;
- close ≥ 1.30 × the 52-week low and ≥ 0.75 × the 52-week high;
- 12-month return percentile in the universe ≥ 70 (A: the in-house stand-in
  for IBD's rating).

**VCP base.**
- **Swing points:** confirmed 5 sessions after they print (A). A swing high
  is the highest high of its ±5 sessions; a swing low likewise.
- **Base:** starts at the latest swing high that no close has exceeded since,
  15–325 sessions ago (3–65 weeks).
- **Contractions:** each swing high to its following swing low, 2–6 of them.
  Each is no deeper than the one before (A: the book says about half). The
  first is ≤ 35% (≤ 50% when NIFTY 50 is more than 20% below its 52-week
  high, A). The last is ≤ 10% (A).
- **Depth against the market:** when NIFTY 50 fell ≥ 8% over the base, the
  base is no deeper than 2.5× that fall (A).
- **Volume dry-up:** average volume over the last contraction is below its
  50-day average, with at least one day within 1.1× the base's lowest
  volume.

**Entry.**
- **Pivot:** the last contraction's swing high.
- **Trigger:** a close above the pivot on volume above the 50-day average.
- **Fill:** buy at the next open, only if the open is ≤ pivot × 1.03 (A: "a
  few percent"); otherwise the trade is logged as missed.

**Market gate.** NIFTY 50 is above its 200-day mean, and new 52-week highs
outnumber new lows in the universe. With the gate off, no new entries;
positions keep their stops.

**Size and exposure.** 10 slots of 10% of equity. Only 5 are open to new
entries unless the last 5 closed trades are net positive (the book's
progressive exposure, A).

**Stops and exits.**
- **Initial stop:** 7.5% below entry (half a 15% average gain; hard cap
  10%). It is live from the next session, so there is no same-day exit, and
  fills at the stop or at a lower open.
- **Breakeven:** the stop moves to entry once a close is ≥ +22.5% (3× the
  risk).
- **Exits at the next open:**
  - a close below the 20-day mean within 10 sessions of the breakout;
  - the largest one-day fall since the breakout on above-average volume;
  - a close below the 50-day mean.

**Costs.** The equity cost model (cost model 4), delivery charges and impact.

**Clarifications (10 Oct 2026, written before any O9 run).**
- **"Largest fall":** it means the largest one-day fall since **the base
  began**, which is the book's stage-2 reference (p.105). "Since the
  breakout" would fire on the first heavy down day of every trade.
- **New highs and lows** are counted on the day's high and low.
- **The pivot** is the latest confirmed swing high.

**Gates.**
- **Gate 1:** deflated Sharpe ≥ 0.95. N = 1 in this family, so this is the
  probabilistic Sharpe against 0.
- **Gate 2:** the live book with 25% of its capital moved to the sleeve:
  0.75 × book + 0.25 × sleeve, funded from the core, as R14 did. The
  combined book must raise CAGR, excess Sharpe and alpha against NIFTY 50
  TRI, with MaxDD no more than 2 points worse.
- **Next step if it passes:** paper trading.

**Result, 11 Oct 2026: O9 closed; nothing to paper-trade.**
- **Run:** Kaggle, 2013–25, its own registry (`runs_sepa`, N = 1).
- **Trades:** 80, plus 47 missed because the breakout opened more than 3%
  above the pivot.
- **Per trade:** hit rate 30%, win/loss 2.39. That is a slightly positive
  expectancy, about +0.02 of the risk per trade.
- **The catch:** the Trend Template plus the VCP rules leave the sleeve only
  **4% invested on average**.
- **Sleeve:** CAGR 0.05%, volatility 2.9%, Sharpe −2.24 against the 6.5%
  risk-free rate, MaxDD −12.9%. Daily correlation with the book is 0.27.
- **Gate 1:** DSR 0.00.
- **Gate 2:** 0.75 × book + 0.25 × sleeve gives CAGR 17.1%, Sharpe 0.98,
  alpha 7.8% against 23.1%, 1.16 and 12.6%. MaxDD improves to −18.5%,
  because a quarter of the capital sits in near-cash.
- **Exits:** 28 below the 20-day mean, 27 below the 50-day mean, 16 on the
  largest fall since the base began, 9 on the stop.

**What carries forward** (core family, after V3; the backlog item for the
Minervini rules on the core): the book's ideas that do not need a sleeve.
- VCP entry timing as a filter on the core's buys;
- a 10% max-loss cap;
- the breakeven move;
- progressive exposure;
- no topping up losers.

The earnings screen waits for GT1.

## 5y. GT5: MCX commodity trend sleeve (pre-registered 11 Oct 2026, 13:10 IST)

**Status: written before any backtest.** GT2's checks read some prices: lot
multipliers, one roll, and outlier returns in Jan–Apr 2015 and Aug–Oct 2025.
No trend signal or sleeve return has been computed. The runs come in
Jan–Mar 2027, on Kaggle, after GT2's close-out.

**Hypothesis.** Time-series momentum: a futures contract that rose over the
past year tends to keep rising. Two sources:
- Moskowitz, Ooi & Pedersen, "Time Series Momentum", JFE 2012 (MOP).
- Hurst, Ooi & Pedersen, "A Century of Evidence on Trend-Following
  Investing", 2017 (HOP).

Both find it in commodities and every other asset class. They explain it by
slow-moving information, hedging pressure and herding. For Centurion the
case is diversification: commodity trend tends to earn in crises, when
equity momentum loses. The review (section 6, item 3) expects a Sharpe of 0.3–0.5
alone and a correlation with the core of −0.2 to +0.1.

**Family and budget.** A new family, commodity, with registry
`data/nse_engine/runs_mcx`. This round uses 2 of its 3 configurations, so N
goes from 0 to 2.

**Data and window.**
- **Store:** GT2's store (`nse_engine/data/mcx_store`), data hash
  8350d7128f3d6fe6 at GT2's close-out (11 Oct 2026), 2005-01-03 to
  2026-10-09.
- **Test window:** 2013-01-01 to 2025-12-31.
- **Reported only:** 2007–25, using data from 2005 for lookbacks.
- **2026:** sealed.

**Instruments (6).** Silver, crude oil, natural gas, aluminium, zinc and
lead: the review's list.
- **Traded contracts:** the minis, SILVERMIC, CRUDEOILM, NATGASMINI,
  ALUMINI, ZINCMINI and LEADMINI.
- **Price series:** each mini's own continuous series (GT2's roll rule).
  Before a mini's first session in the store, the large contract's series
  (SILVER, CRUDEOIL, NATURALGAS, ALUMINIUM, ZINC, LEAD), sized in the mini's
  first known lot. **(A)** This is a synthetic mini, and it changes only lot
  rounding.
- **Excluded:**
  - gold, because the live metal sleeve holds it;
  - copper, whose 2.5 MT lot (about Rs 22 lakh) is beyond the sleeve;
  - farm contracts, which are thin and taxed differently.
- **Silver overlap:** silver overlaps the metal sleeve's SILVERBEES; gate 2
  measures the effect.

**Signals.** Each is decided at the close of the last MCX session of each
calendar month **(A: MOP rebalance monthly)**.
- **GT5-A, 12-month time-series momentum (MOP):** s = the sign of the
  instrument's return over the past 252 sessions, giving +1 or −1. s is 0
  with fewer than 252 sessions of history.
- **GT5-B, multi-horizon trend (HOP):** s = the mean of the signs of the
  21-, 63- and 252-session returns, giving ±1 or ±1/3.

**Sizing (both configurations).**
1. **Sleeve equity:** E starts at **Rs 5 lakh**, the owner's choice of
   11 Oct: 25% of a Rs 20 lakh book. Equity compounds and nothing is
   withdrawn or added.
2. **Target notional:** at each rebalance, target_i = s_i × E × (40% / σ_i) / 6.
   - σ_i is the annualised EWMA volatility of the instrument's daily
     continuous returns, with a centre of mass of 60 sessions, × √261, and
     at least 63 sessions of returns. This is MOP's 40% per-instrument
     ex-ante volatility, averaged over the 6 instruments.
3. **Lots:** target_i / (multiplier × close), rounded to the nearest integer,
   with a half rounded towards 0. At Rs 5 lakh, one base-metal lot is often
   bigger than its target, so it often rounds to 0.
4. **Margin:** assumed at 25% of gross notional **(A: MCX's SPAN plus
   exposure margin is about 10–30% for these contracts)**. It must stay
   within 80% of E. Otherwise every target scales down by the same factor
   and the lots are rounded again.
5. **Between rebalances:** the lots are held, with no daily resizing.

**Rolls.** At the continuous series' roll closes, close the old lots and
open as many in the new contract at the same close, charging both sides.
GT2's rule holds the most-open-interest contract more than 10 days from
expiry. **(A)** The series reads that close's open interest, which is
published with the bhavcopy; a live roll would use the previous close's.
The one-session difference is not modelled.

**Timing and fills.** Each decision is made at a close and filled at the
next session's close, at `mcx_costs.fill_price`.
- **Daily P&L:** lots × multiplier × the change in the held contract's raw
  close. On a roll day the old contract runs to the roll close and the new
  one from it.
- **Swing and positional only:** monthly decisions, no same-day exits.

**Costs.** `nse_engine.mcx_costs` version 1 at its dated rates:
- CTT 0.01% of sells from Jul 2013;
- MCX transaction charge 0.0026%, then 0.0021% from Oct 2024;
- stamp duty, SEBI fee and GST;
- brokerage, the lower of Rs 20 and 0.03%.

Slippage is fixed here at 0.05% of price a side. Idle cash and margin earn
nothing (cost model 4).

**Pass rules.**
- **Gate 1:** the sleeve's deflated Sharpe ≥ 0.95 at the family's N, on
  excess returns over 6.5%. That is the book's convention, as for O9, since
  the capital comes from the core.
- **Gate 2:** 0.75 × the live book (deployed run
  20261010T032441064151Z_679cbd0c) + 0.25 × the sleeve, daily. The combined
  book must raise CAGR, excess Sharpe and alpha against NIFTY 50 TRI, with
  MaxDD at most 2 points worse. The same as R14 and O9.
- **Also required (review section 5, phase 1):**
  - net Sharpe ≥ 0.5 alone;
  - combined Sharpe ≥ the book's + 0.10.
- **Significance:** t ≥ 2, as the replication of a published effect.
- **Reported, not gates:**
  - monthly return correlation with the book (a gate until the
    clarification below);
  - the 2007–25 window;
  - each instrument's P&L;
  - lots traded and the share of months each instrument held 0 lots,
    which is what Rs 5 lakh costs in diversification;
  - roll and fee costs;
  - the trades.

**What a result leads to.**
- **Pass:** no walk-forward is needed, since no parameter is fitted. Next
  come at least 60 sessions of paper trading, judged on tracking (protocol
  6), then live once the book reaches about Rs 20 lakh.
- **Fail:** recorded, never re-tuned. The family's third configuration stays
  unused unless a new source justifies it.

**Clarification (11 Oct 2026, 14:49 IST, before any run; owner's decision).** Monthly correlation with the book ≤ 0.3 is no longer a pass rule; it is reported. Gate 2 and "combined Sharpe ≥ the book's + 0.10" already test what low correlation is for, a better combined book. A sleeve that passes both at a higher correlation is a real improvement the correlation bar alone would have rejected.

**Clarification (12 Oct 2026, 08:46 IST, before any run; owner's decision).**
1. **Timing:** GT5 runs now, on 2013–25 as written (2007–25 reported),
   instead of Jan–Mar 2027. RF1's move of GT5 to 2013–26 no longer applies
   to this round. The MCX store's hash is unchanged (8350d7128f3d6fe6).
   - After the verdict, 2026 (to the latest session) is opened once and
     reported. It does not change the verdict.
   - If V3 promotes another configuration in January, gate 2 is
     re-computed against it, reported only.
2. **Translations, fixed before the run (A):**
   - **Mini gaps:** MCX did not list the crude, aluminium, zinc and lead
     minis from Dec 2019 / Apr 2020 to Feb–Mar 2023. As before a mini's
     first session, the large contract's series stands in, sized in the
     mini's first known lot.
     - A switch between the two is traded as a roll.
     - The share of instrument-sessions on a stand-in is reported. Real
       trading then needed the large lots, which are beyond Rs 5 lakh.
   - **One return series per instrument:** signals and volatility read the
     mini's own daily returns where it trades, else the large contract's.
   - **P&L:** lots × multiplier × the change in the held contract's raw
     close. A contract with no print that day keeps its last close. The
     multiplier is the held contract's implied one; a stand-in uses the
     mini's first known lot.
   - **Rolls and switches:** both legs trade at the roll close, each at
     `fill_price` and with its charges.
   - **Volatility:** the EWMA standard deviation with a centre of mass of
     60 (MOP's δ), × √261.
   - **Sessions:** MCX's own calendar, so the month's last MCX session
     decides. For gate 2, the sleeve's equity is carried to NSE sessions;
     a day only MCX trades rolls into the next NSE session.
   - **Significance:** t ≥ 2 is the Newey-West t-statistic of the sleeve's
     alpha against NIFTY 50 TRI, on excess returns over 6.5%, as in 5z.
     Commodity trend's beta to NIFTY is near zero, so in effect this tests
     its excess return.

**Result, 12 Oct 2026: both FAIL. GT5 closes and nothing is paper-traded.**

- **Runs:** Kaggle, 2013–25: `20261012T032317017129Z_afb53389` (A) and
  `20261012T032319911184Z_f9977384` (B).
- **MCX store:** `8350d7128f3d6fe6`.
- **Commodity family:** N = 2.

| | CAGR | Excess Sharpe | MaxDD | DSR | Alpha vs NIFTY 50 TRI (t) | Beta | Monthly correlation with the book |
|---|---|---|---|---|---|---|---|
| Book (679cbd0c) | 23.1% | 1.163 | −23.9% | | +12.6% | | |
| GT5-A alone | 2.4% | −0.06 | −54.5% | 0.40 | −0.8% (−0.12) | −0.03 | 0.04 |
| GT5-B alone | 3.4% | −0.09 | −38.6% | 0.37 | −0.6% (−0.13) | −0.08 | −0.01 |
| 0.75 book + 0.25 A | 18.3% | 0.96 | −20.9% | | +9.2% | | |
| 0.75 book + 0.25 B | 18.4% | 1.01 | −18.0% | | +9.3% | | |

**Rules:** both fail gate 1, gate 2 (CAGR, Sharpe and alpha fall; only
MaxDD improves), Sharpe ≥ 0.5, the combined + 0.10 and t ≥ 2.

What the result says:
- **The diversification is real; the return is not.** Correlation with the
  book is about 0, and a quarter of the book in the sleeve cuts the
  combined MaxDD by 3–6 points. But the sleeve earns less than the 6.5%
  hurdle, so the book loses 4.7 points of CAGR.
- **Gross trend profits are positive in all six instruments,** about Rs 4–5
  lakh over 13 years on Rs 5 lakh. Costs take Rs 2.2–2.3 lakh, 3.0–4.0% a
  year, mostly slippage on both legs of the monthly rolls (Rs 1.0–1.3
  lakh).
- **Margin cash earning 6.5%** (one look, not a trial; cost model 4 pays
  idle cash nothing): A 9.0% / Sharpe 0.19, B 10.2% / 0.25. Still far
  below 0.5.
- **2007–25 (reported):** A 1.3% / −0.14 / −71%; B 3.1% / −0.15 / −54%. In
  2008, A lost 5.9% while B made 31%: only the faster signals caught the
  crash.
- **Reported:**
  - stand-in share of sessions: crude 41%, natural gas 79%, base metals
    21%, silver 0%;
  - zero-lot months: A up to 15% (zinc); B 24–44% in the base metals;
  - gross exposure 2.0× (A) and 1.3× (B) of equity;
  - the margin cap never bound.
- **2026, opened once (to 9 Oct):** A +21.3%, B +18.1%, against NIFTY 50 TRI
  −13.0%. This is the crisis behaviour the hypothesis expects, but one
  reported year does not change the verdict.
- **Not re-tuned.** The family's third configuration stays unused, as 5y
  says, unless a new source justifies it. `nse_engine/mcx_trend.py` stays:
  the `mcx` task on Kaggle, and `evaluate`.

## 5z. GT4: selective value / quality sleeve (pre-registered 11 Oct 2026, 14:17 IST)

**Status: written before any backtest, and before GT1's store is built.**
GT1's tests read only sample filings. Coverage-dependent rules are fixed
below. Dated clarifications are allowed until the first run, Jan–Mar 2027,
on Kaggle with `run --attach centurion-nse-fundamentals`.

**Hypothesis.** Cheap stocks with real profits earn a premium. Sources:
- Basu 1977 and Fama & French 1992 (E/P and value);
- Asness, Frazzini & Pedersen, "Quality Minus Junk", RAS 2019 (quality:
  profitability, growth, safety);
- Agarwalla, Jacob & Varma's IIMA Indian factor data (an Indian value
  premium).

For Centurion the case is diversification: value earns in the rebounds that
momentum misses. The review's bar (section 6, item 2) is that the sleeve
must reach Sharpe of about 0.8 alone to lift the book. Unhedged, it will
correlate about 0.4–0.5 with the core.

**Family and budget.** A new family, value, with registry
`data/nse_engine/runs_value`. This round uses 2 of its 3 configurations, so
N goes from 0 to 2. The third is kept for the beta-hedged variant. It is
not registered now: a NIFTY lot (about Rs 19 lakh of notional) is beyond a
Rs 5 lakh sleeve, so it fails the protocol's broker-feasible sizing.

**Data and window.**
- **Stores:** the equity store, for prices, corporate actions and the
  calendar. GT1's fundamentals store (`nse_engine/data/fundamentals`) for
  results, read only through `point_in_time` (a filing counts from its
  `available_from` session). Its data hash is recorded at GT1's close-out.
- **Test window:** 2013-01-01 to 2025-12-31. Filings start in 2012, which
  is enough for trailing twelve months (TTM) by early 2013.
- **Reported only:** none earlier; the filings start in 2012.
- **2026:** sealed.

**Universe.** The engine's universe rule at top 500, refreshed every 21
sessions:
- median traded value ≥ Rs 1 crore over 126 sessions;
- 252 sessions of history;
- printed close ≥ Rs 20;
- ETFs excluded.

This matches GT1's download set.

**Fundamentals, point in time.** At a decision date, take each company's
four latest quarters known by then.
- **Basis:** consolidated if all four are consolidated, else standalone if
  all four are; otherwise the company has no TTM that month.
- **Profit:** profit to owners where given (consolidated), else profit for
  the period.
- **Market capitalisation:** the latest filing's share count (paid-up ÷
  face value), adjusted for splits and bonuses after it with the store's
  corporate-action factors, × that day's printed close.
- **Missing data:** a company without a TTM (gaps, such as the Q1 FY19
  filings NSE never published) is not ranked that month. Nothing is
  imputed.

**Configurations.** Both are decided at the close of the last session of
each month, filled at the next session's open (CNC, the core's fills and
cost model 4), and held to the next decision.
- **GT4-A, cheap and profitable:**
  - **Eligible:** TTM profit > 0, latest quarter's profit > 0, TTM revenue
    > 0.
  - **Rank:** by E/P = TTM profit ÷ market capitalisation.
  - **Holdings:** the top 25, equal weight.
  - **Exits:** a holding is sold only when it leaves the top 50, an exit
    band like the core's 20/40 **(A)**.
- **GT4-B, value plus quality (QMJ-lite):**
  - **Eligible:** the same as GT4-A.
  - **Composite:** the mean of four percentile ranks within the eligible
    set:
    1. E/P (value);
    2. TTM profit before tax ÷ TTM revenue (profitability);
    3. TTM profit ÷ TTM profit a year earlier − 1, with a negative base
       ranked lowest (growth);
    4. minus the 252-session volatility of daily returns (safety).
  - **Holdings and exits:** the same as GT4-A, on the composite.
- **Common to both:**
  - **Capital:** Rs 5 lakh, 25% of a Rs 20 lakh book, matching GT5.
  - **Weights:** equal, re-weighted only at decisions.
  - **Banks and NBFCs:** included. Revenue is total income for banks, as
    the store defines it.
  - **Risk controls:** no stops and no regime gate. The sleeve is meant to
    hold through the momentum book's bad stretches.

**Pass rules.**
- **Gate 1:** deflated Sharpe ≥ 0.95 at the family's N, on excess returns
  over 6.5%.
- **Gate 2:** 0.75 × the live book (deployed run
  20261010T032441064151Z_679cbd0c) + 0.25 × the sleeve, daily. It must
  raise CAGR, excess Sharpe and alpha against NIFTY 50 TRI, with MaxDD at
  most 2 points worse.
- **Also required (review section 5, phase 1):**
  - Sharpe ≥ 0.5 alone;
  - combined Sharpe ≥ the book's + 0.10.
- **Significance:** t ≥ 2, as the replication of a published effect.
- **Reported, not gates:**
  - monthly correlation with the book (a gate until the clarification
    below; unhedged it will likely be 0.4–0.5);
  - the share of the universe with a TTM each month;
  - turnover;
  - the sector mix;
  - each year's return beside the book's.

**What a result leads to.**
- **Pass:** at least 60 sessions of paper trading, judged on tracking.
- **Fail:** recorded, never re-tuned.

**Clarification (11 Oct 2026, 14:49 IST, before any run; owner's decision).** Monthly correlation with the book ≤ 0.3 is no longer a pass rule; it is reported. Gate 2 and "combined Sharpe ≥ the book's + 0.10" already test what low correlation is for, a better combined book. A sleeve that passes both at a higher correlation is a real improvement the correlation bar alone would have rejected.

**Clarification (12 Oct 2026, 07:30 IST, before any run): GT1's store at close-out.**
- **Data hash:** `83fd93702acbf1d1` (fundamentals v2).
- **Coverage:** the share of the universe with a TTM at each universe
  refresh. It is the same with TTM revenue required as well.
  - 89–91% in 2013–18;
  - 93% in 2019;
  - 95–98% in 2020–25.
- **Rules unchanged:** a name without a TTM is not ranked.
- **Reported with the run:**
  - In 2018, about 5% of the universe ranks on a TTM whose latest quarter
    is more than 200 days old (the Q1 FY19 gap).
  - On the 2012-start store, the universe averages 292 names in 2013 and
    393 in 2014.

**Clarification (12 Oct 2026, 08:05 IST, before any run; owner's decision).**
1. **Timing:** GT4 runs now, on the 2013–25 window as written, instead of
   Jan–Mar 2027. RF1's move of GT4 to 2013–26 no longer applies to this
   round.
   - After the verdict, 2026 (to the latest session) is opened once for
     both configurations and reported beside the book. It does not change
     the verdict.
   - If V3 promotes another configuration in January, gate 2 is
     re-computed against it, reported only.
2. **Translations, fixed before the run (A):**
   - **Quarters:** a filing counts as a quarter when its period is at
     most 100 days. That excludes 360 longer "integrated" filings.
   - **Profit to owners "where given":** in 4,777 consolidated quarters
     (5%), the owners' profit reads exactly 0 while the profit for the
     period does not. That is a blank field parsed as zero, so it counts as
     not given and the profit for the period is used. 5aa reads profit "as
     in 5z", so GT7 and MV1 inherit this.
   - **TTM:** the four latest quarters must be consecutive, with period
     ends three months apart; otherwise there is no TTM. The year-earlier
     TTM (GT4-B growth) is the four consecutive quarters ending 12 months
     before the latest, on the same basis.
   - **Data guard:** a name is ranked only when two earnings yields agree
     within a factor of 2: TTM profit ÷ market capitalisation, and the
     four quarters' basic EPS (restated per share at the decision through
     the same ratios) ÷ the printed close. EPS is per share, so it is
     immune to two faults found in NSE's pages before any run:
     - amounts entered in the wrong unit (about 450 quarter-on-quarter
       revenue jumps of 10× and 140 of 100×);
     - wrong share counts (e.g. face value and paid-up capital swapped).

     Without the guard, 3–9 of the top 25 earnings yields at a sample of
     decision dates were such faults. A name whose EPS check cannot be made
     is not ranked.
   - **Share count:** from the latest quarter with one. Split, bonus and
     consolidation ratios dated after its period end, the books'
     share-count sources (`book_events.SHARE_SOURCES`), divide it out.
   - **Universe:** the deployed engine's universe settings at top 500,
     with the Rs 20 test on the printed close
     (`price_filter_unadjusted`). Ranking uses the universe in force at
     each decision.
   - **Weights:** at each decision every holding is set to 1/25 of the
     sleeve's equity. With fewer than 25 names, the rest stays in cash,
     earning nothing, as in the core.
   - **GT4-B:** a name with a TTM but no year-earlier TTM, or no TTM
     margin or volatility, ranks lowest on that component, like a
     negative base. Ties take the average rank.
   - **Fills:** sells, then buys, at the next session's open on adjusted
     prices, in whole shares. Costs are the core's statutory charges plus
     impact (cost model 4), as for O9. A holding with no open is sold at
     the next open that prints.
   - **The core's fill limits apply:** a trade under Rs 5,000 is skipped
     (`min_trade_value_inr`), so small re-weightings wait; and an order is
     capped at 5% of the 20-session median traded value.
   - **Significance:** t ≥ 2 is the Newey-West t-statistic of the
     sleeve's alpha against NIFTY 50 TRI, on excess returns over 6.5%.
     The premium is the hypothesis.

**Result, 12 Oct 2026: both FAIL. GT4 closes and nothing is paper-traded.**

- **Runs:** Kaggle, 2013–25: `20261012T030202658081Z_5aed9d74` (A) and
  `20261012T030237590222Z_a0da7f72` (B).
- **Data hashes:** equity store `2a6f7d48f0e59b70`, fundamentals
  `83fd93702acbf1d1`.
- **Value family:** N = 2.

| | CAGR | Excess Sharpe | MaxDD | DSR | Alpha vs NIFTY 50 TRI (t) | Beta | Monthly correlation with the book |
|---|---|---|---|---|---|---|---|
| Book (679cbd0c) | 23.1% | 1.163 | −23.9% | | +12.6% | | |
| GT4-A alone | 14.7% | 0.43 | −68.4% | 0.88 | +2.6% (0.47) | 1.01 | 0.56 |
| GT4-B alone | 16.8% | 0.69 | −41.3% | 0.98 | +4.8% (1.65) | 0.73 | 0.63 |
| 0.75 book + 0.25 A | 21.4% | 1.00 | −35.2% | | +10.1% | | |
| 0.75 book + 0.25 B | 21.7% | 1.135 | −26.5% | | +10.6% | | |

**Rules:**
- **GT4-A** fails all five.
- **GT4-B** passes gate 1 (DSR 0.98) and Sharpe ≥ 0.5 alone (0.69). It fails:
  - gate 2 on every check: CAGR, Sharpe and alpha all fall, and MaxDD is
    2.65 points worse;
  - the combined Sharpe + 0.10;
  - t ≥ 2.

What the result says:
- **The diversification case did not hold.** Correlation with the book is
  0.56–0.63, above the 0.4–0.5 expected.
  - Value did earn where the book lagged: 2020 (A +38%, B +44%, book
    +28%), 2022 and 2023.
  - It lost more in 2013, 2018 and 2019 (A −28%, −32%, −23%; B −7%, −14%,
    −6%; book +11%, −12%, +6%) and trailed in 2021 and 2025.
- **GT4-B works as a stand-alone sleeve but not beside the book.** Moving
  25% of the book into it costs 1.4 points of CAGR and 0.03 of Sharpe.
- **GT4-A's −68% drawdown** comes from cheap stocks in 2018–19. It also
  buys one-off gains, as a trailing E/P does: Strides Arcolab's Agila sale
  and MTNL's settlement, both in 2014.
- **A beta-hedged variant is not worth the third configuration now.**
  B's alpha against NIFTY has an information ratio of about 0.46 (t 1.65
  over 13 years), below the 0.5 bar. The configuration stays reserved.
- **Reported:**
  - turnover 1.18 / 1.14 a year;
  - 86% of the universe with a TTM on average;
  - 25 holdings, 99.5% invested;
  - sectors (repo map), A: Financial Services 17%, Oil & Gas 11%, half
    unmapped;
  - sectors, B: Financial Services 21%, Power 10%, Healthcare 9%.
- **2026, opened once (to 9 Oct):** A +5.9%, B −6.8%, NIFTY 50 TRI −13.0%.
  Reported only; the verdict stands.
- **Not re-tuned.** `nse_engine/value.py` stays: the `value` task on
  Kaggle, and `evaluate`.

## 5aa. Core tests after V3: GT7 earnings drift and MV1 Minervini rules (pre-registered 11 Oct 2026, 14:17 IST)

**Status: written before any backtest, and before GT1's store is built.**
Both are **core-family** trials, so they count in the core's N, which is at
the PBO bar. Under the research protocol (section 1), each queued core test
takes **one configuration**. MV1's rules are therefore one bundled
configuration, not one per rule: this supersedes the tracker's earlier
"one configuration each". They run after V3's promotion decision in January
2027, each against the configuration V3 leaves live. That is the
"reference" below, today the deployed engine. Dated clarifications are
allowed until each runs.

**Common to both.**
- **Engine:** the core engine, 2013-01-01 to 2025-12-31, cost model 4.
- **Universe and portfolio:** the reference's universe, portfolio, regime,
  sleeves and drawdown overlay. Only the changes below differ.
- **Fundamentals:** read point in time from GT1's store, as in 5z (basis,
  profit, missing data).
- **Pass rules:**
  - **Gate 1:** deflated Sharpe ≥ 0.95 at the core family's N.
  - **Beat the reference:** CAGR, excess Sharpe and alpha against NIFTY 50
    TRI all higher, MaxDD at most 2 points worse.
  - **Family PBO:** ≤ 30% after the trial is added.
  - **Significance:** t ≥ 2 on the daily return difference against the
    reference.
- **Promotion:** a pass goes to the forward gate on paper, as V3 does.
  Nothing is promoted on the backtest alone.

### GT7: post-earnings-announcement drift as a fourth forecast group

**Hypothesis.** Prices underreact to earnings surprises and drift toward
them for about 60 sessions (Bernard & Thomas 1989; Chan, Jegadeesh &
Lakonishok 1996, whose earnings momentum adds to price momentum). The
review expects +0.1 to +0.2 to the core.

**Rule.**
- **Surprise:** for each company's latest quarter known at the date,
  standardised unexpected earnings:
  SUE = (profit_q − profit_q−4) ÷ the standard deviation of
  (profit − profit four quarters earlier) over the previous 8 quarters,
  which needs at least 4.
  - **(A)** It uses profit, not EPS, so splits and bonuses do not distort
    it.
  - **Basis:** consolidated when both quarters are consolidated, else
    standalone.
- **Group:** a new forecast group, "earnings":
  - the centred rank of SUE within the universe, normalised like the other
    rank groups (`centred_rank` → `normalise_forecast`);
  - a name with no SUE known within 126 sessions scores 0, not NaN. With
    NaN the engine would renormalise that name onto trend alone.
- **Weights:** fast_trend 1/3, slow_trend 1/3, earnings 1/3 **(A)**: equal,
  hand-set, as the engine's default three groups. The forecast
  diversification multiplier (FDM) and caps are unchanged.

### MV1: Minervini's rules on the core, one bundled configuration

**Source.** Minervini, *Trade Like a Stock Market Wizard* (2013), in
`research_inbox/`. O9-A's definitions are reused.

**The rules, all at once:**
1. **Earnings screen on new buys** (ch. 7–8; pp. 127–158 of the book):
   - **Growth:** profit growth of at least 20% year on year in each of the
     latest two quarters. The book: "a minimum of 20 to 25 percent
     year-over-year increases in the most recent one, two, or three
     quarters" **(A: 20%, two quarters, profit not EPS)**.
   - **Turnarounds:** a quarter with a year-ago loss qualifies only if
     profit is positive now and the other quarter grew ≥ 100%. The book:
     turnarounds "+100 percent or better".
   - **Sales support:** the latest quarter's revenue is higher than a
     year earlier (A of "earnings supported by revenue").
   - **Missing data:** a name with no filing data passes, so data gaps do
     not change the core.
   - **Held names:** never sold for failing the screen.
2. **VCP timing on new buys:** a name the core would buy is bought only if:
   - in the last 5 sessions it closed above its latest confirmed VCP pivot
     on volume above its 50-day average (O9-A's swing, base, contraction
     and volume rules, `nse_engine.sepa`);
   - and the next open is ≤ pivot × 1.03.

   Otherwise its slot waits for the next rebalance. Names already held are
   unaffected.
3. **Max-loss cap:** the stop is the higher of the reference's 6-ATR stop
   and 10% below the entry price.
4. **Breakeven:** once a close is ≥ 20% above entry, the stop rises to
   entry and never falls back.
5. **Progressive exposure:** after the regime gate re-opens from risk-off,
   new entries may fill only half the target positions (10 of 20). The rest
   open once the last 5 closed trades are net positive (O9-A's
   translation).
6. **No topping up below entry:** a rebalance never adds to a position
   whose close is below its average entry price.

## 5ab. CR1: crash-recovery capital advisory (pre-registered 11 Oct 2026, 21:17 IST)

**Question (owner, 11 Oct).** After a deep market fall, does adding fresh
money from a reserve kept outside the book, in stages or on a recovery
signal, beat keeping it in cash? Does it lose less than buying at the first
−20% close?

- **Advisory only:** an email; the owner places the trade. The money goes
  to a NIFTY index ETF, never the momentum core.
- **Family:** event study, as GT13 was. Its N is its 3 configurations.
- **Not tested:** breadth-thrust signals. NSE breadth starts in 2012, so
  they cannot be tested globally.

**Episode.** An index closes 20% or more below its running peak. The
episode starts at that close and ends when the index regains the prior
peak, or at the end of the data.

**Configurations:**
- **CR1-A, staged:** a third at −30%, a third at −40%, the rest at
  confirmation.
- **CR1-B, confirmation:** everything at confirmation.
- **CR1-C, depth:** a half at −30%, a half at −40%.

**Shared rules:**
- **Confirmation:** the first close above the 200-session average after
  the index has closed below it during the episode.
- **Fallback:** anything not yet invested goes in when the index regains
  its prior peak, so no reserve waits in cash for ever.
- **Fills:** each tranche fills at the next session's close, because the
  owner acts on the email.
- **Each level fires at most once per episode.**

**Benchmarks:**
- **Cash:** the reserve stays in cash at 4% a year **(A)**, a round global
  rate. India is also reported at 6.5%.
- **L20:** a lump sum at the next close after the first −20% close.

**Outcome per episode:** the reserve's value 3 years after the episode
starts. The index part uses the price index, without dividends; the
uninvested part earns the cash rate.

**Data.**
- Yahoo Finance daily closes in local currency, each from its first year to
  2025-12-31 (2026 sealed): ^GSPC, ^N225, ^FTSE, ^GDAXI, ^HSI, ^GSPTSE,
  ^AXJO, ^KS11, ^TWII, ^STI, ^BSESN and ^MXX.
- India: NIFTY 50 for 2006–25 from the stores.

**Pass rules, per configuration, over all episodes with 3 years of data:**
1. It beats cash in at least 70% of episodes, and its median excess over
   cash is above 0.
2. Its worst episode is better than L20's worst. That is the reason to
   stage.

The advisory is built on the first configuration that passes, in the order
A, B, C. If none passes, CR1 closes as an idea that history does not support.

**Reported, never gating:**
- the episode table, with India separately;
- each configuration's median against L20, which is the upside given up;
- how many tranches fired, and the days to full deployment.

**Result, 11 Oct 2026: all three FAIL; CR1 closes and no advisory is built.**

89 episodes in 13 indices, starting 1929–2022. Values are the reserve per
1.0 after 3 years; cash at 4% is 1.125.

| Configuration | Beats cash | Median excess over cash | Worst (L20: 0.347) | Median against L20 | Verdict |
|---|---|---|---|---|---|
| CR1-A staged | 66.3% | +0.120 | 0.331 | −0.041 | FAIL, rules 1 and 2 |
| CR1-B confirmation | 66.3% | +0.110 | 0.283 | −0.078 | FAIL, rules 1 and 2 |
| CR1-C depth | 65.2% | +0.104 | 0.355 | −0.015 | FAIL, rule 1 |
| L20 (benchmark) | 61% | | 0.347 | | |

What the result says:
- **Buying after a deep fall usually beats cash, but not reliably enough.**
  About one episode in three still trails cash after 3 years. One in five
  loses money outright: A 18, B 19, C 15, L20 20.
- **Staging gives up upside and barely protects the worst case.** Every
  configuration's median trails the lump sum at −20%. Rule 2 rests on a
  single episode: the S&P 500's 1929 fall of 86% is the worst for every arm.
- **The 200-day reclaim fires early.** It came before −30% in 51 of 89
  episodes, so A invested everything at once and matched B. C used both
  tranches in 55 episodes.
- **Days to full deployment:** A and B median 130 (90th percentile 388); C
  median 308 (917), counting the fallback.
- **India (reported, cash at 6.5%), 13 episodes:** NIFTY 5 and Sensex 8,
  with the 5 since 2006 counted twice, so about 8 independent falls.
  - C: beats cash 77%, median +0.216, worst 0.830 against L20's 0.666;
    it would pass;
  - A and B: 69%.
  
  Too few falls to build on. Choosing India after seeing this would be a
  fit.
- **Design caveat:** the outcome uses price indices, so every equity arm
  misses its dividends against cash (about 2% a year).
  - One crude look, reported only: all three arms clear 70% at a 2% yield
    (73–78%).
  - A total-return re-run would be a new trial designed after that look.
    It would not be clean evidence.
- **Implementation fix before recording:** the first run priced NIFTY's
  cash at 6.5% in the gating test. Every market now uses 4%, as
  pre-registered. The beats-cash counts did not change.
- **2026 holdout:** no 3-year horizon can complete in it, so nothing was
  opened.
- **Not re-tuned.** The module (`nse_engine/crash_recovery.py`) stays.
  `status` shows where NIFTY stands against the levels.

## 5ac. CR1b: crash-recovery tranches on recovery signals (pre-registered 11 Oct 2026, 22:10 IST)

**Question (owner, 11 Oct, after CR1 failed).** Staging the reserve
30 / 40 / 30 on signals that can be seen in real time: does it beat cash,
and does it lose less than buying at the first −20% close?

This was designed after seeing CR1's result. The bar is not lowered, and a
split-half rule is added.

- **Family:** CR1's event study. N goes from 3 to 6.
- **Unchanged from 5ab:**
  - the episodes, the fallback and fills at the next session's close;
  - each signal fires at most once per episode;
  - the 3-year horizon;
  - cash at 4%, with India also reported at 6.5%;
  - L20;
  - the data, with 2026 sealed.

**Signals,** on closes within the episode, from its start:
- **Bounce:** a close 10% above the lowest close since the episode started.
- **60-day high:** the close is the highest of the last 60 sessions.
- **Cross:** the 50-session average closes above the 200-session average,
  after being at or below it during the episode. The cross invests
  whatever is still in cash.

**Configurations:**

| Configuration | 30% | 40% | 30% |
|---|---|---|---|
| CR1b-D, confidence | bounce | 60-day high | cross |
| CR1b-E, depth | −30% | −40% | cross |
| CR1b-F, wait | everything at the 60-day high | | |

**Outcome:** as in 5ab, except that the invested part also earns an assumed
dividend yield, accrued daily, on every arm including L20.
- India: 1.3%.
- Japan: 1.0%.
- DAX: 0. It is a performance index and already includes dividends.
- Others: 2%.

These are conservative against long-run yields.

**Pass rules, per configuration:**
1. It beats cash in at least 70% of episodes, and its median excess over
   cash is above 0.
2. Its worst episode is better than L20's worst.
3. Rule 1 also holds on its own in each half of history: the episodes that
   start before the median start date, and those that start on or after it.

The advisory is built on the first configuration that passes, in the order
D, E, F. If none passes, CR1 closes for good. A facts-only alert on the
levels stays the owner's call.

**Reported, never gating:**
- the same tables at a 5-year horizon;
- each configuration without the 2000–03 episodes (those starting
  2000-01-01 to 2003-03-31), and separately without the 2008 episodes
  (those starting 2007-07-01 to 2009-03-31);
- India separately, at 6.5% cash;
- CR1's A, B and C re-scored with dividends. Their verdict stays FAIL.

**Result, 11 Oct 2026: all three FAIL. CR1 closes for good and no
advisory is built.**

89 episodes in all. The halves split at 29 Jun 2000: 44 episodes start
before it and 45 on or after it. L20's worst is 0.368.

| Configuration | Beats cash | Median excess over cash | Worst | Median against L20 | Halves (beats cash) | Verdict |
|---|---|---|---|---|---|---|
| CR1b-D confidence | 71.9% | +0.190 | 0.537 | −0.070 | 61% / 82% | FAIL, rule 3 |
| CR1b-E depth | 74.2% | +0.182 | 0.555 | −0.064 | 61% / 87% | FAIL, rule 3 |
| CR1b-F wait | 69.7% | +0.146 | 0.325 | −0.079 | 64% / 76% | FAIL, rules 1, 2 and 3 |

What the result says:
- **With dividends, D and E nearly worked.** Ending on the 50/200 cross
  beat cash about 3 times in 4. It also lifted the worst episode from L20's
  0.37 to about 0.55. Both still trail L20's median.
- **The edge does not hold across history at 3 years.** Only 61% of the
  episodes before mid-2000 beat cash. The losses come from slow bears:
  - Japan 1990;
  - Asia 1997–99: Taiwan, Korea, Singapore and the Sensex;
  - the first dot-com starts in early 2000;
  - 1929.

  Five of the 17 are shallow falls. The split date falls inside the
  2000 cluster, so the halves are not fully independent. It was
  pre-registered, so it was not re-cut.
- **F, which waits for a 60-day high, fails outright.** In 1929 it bought
  into the 1930 rally: worst 0.325.
- **Reported only:**
  - At 5 years (84 episodes), D (76%; halves 71% / 81%) and E (79%; 71% /
    86%) would pass.
  - Without 2000–03, D and E reach 81% and would pass.
  - Without 2008, only E passes, at 70.9%.
  - India (13 episodes, at 6.5% cash): D and E would pass, at 85%.

  Choosing the horizon or the sample after seeing this would be fitting,
  so none of these changes the verdict.
- **CR1 with dividends (reported):** A 71.9%, B 70.8%, C 73.0%. C would
  clear CR1's two rules, with a worst of 0.377 against 0.368. The verdict
  stays FAIL.
- **Lost money outright after 3 years:** D 13, E 15, F 16 and L20 15 of 89.
- **What the family shows:** buying after a deep fall pays about 3 times in
  4 for an investor who can wait 5 years. No rule tested here reliably
  times it at 3 years. The family now has 6 trials; another variant would
  be fitting.

## 5ad. US round 1: the India books' rules on US stocks (US tracker US5; pre-registered 16 Oct 2026, 23:15 IST)

**Hypothesis.** The India engine harvests trend and momentum: under-reaction
to news and slow-moving flows (Jegadeesh & Titman 1993; Moskowitz, Ooi &
Pedersen 2012). Both are documented in US large caps. Do the India books'
rules, unchanged, earn more than simply holding SPY?

**The US book's job (US-Y7, your go-ahead 16 Oct):**
- alone, in USD, it must beat SPY;
- beside India, in INR, it must raise the book's Sharpe more than SPY would.

The golden target applies to the combined book. Leverage stays India-only
(GT8).

- **Family:** US core, registry `data/nse_engine/runs_us`. This round uses
  3 of its 3 configurations: N goes from 0 to 3.
- **Data:**
  - the US store `us_store/panel`, hash d5854e5a3a4c2f2c; engine data
    fe0af5f9468e7ee0, union with total return;
  - point-in-time members of the S&P 500 and the Nasdaq-100 (the union,
    US-Y3);
  - split- and dividend-adjusted prices, with spin-offs as distributions
    (US4).
- **Window:**
  - 2013-01-01 to 2025-12-31 for the gates, with warm-up from 2006-01-03;
  - 2007-01-01 to 2025-12-31 reported only. 2007–12 is 91.6% priced and
    runs +1.1%/yr against RSP, because the 2008 failures are missing, so it
    flatters the result;
  - 2026 sealed: the store ends 2025-12-31.
- **Configurations:** the India books' rules, each with its drawdown overlay
  (20/30/35, half 0.5, re-arm 60):
  - **A**, `config/nse_engine_deployed.json`: what the US-A1 emails run,
    without their sector cap on today's lists. Hash 2ca6b779.
  - **B**, candidate: regime `scale_neutral` 0.6. Hash 2beeac17.
  - **C**, E4: exit rank 60 and refill exits. Hash 936f7472.

  B and C also carry `universe.price_filter_unadjusted`, which does nothing
  here (no price floor). All three use the US settings
  (`us_engine.backtest.US_MARKET`, unchanged since US4):
  - members only, $20M median daily value, top 600, no price floor;
  - calendar schedules: rebalance on each week's first session;
  - fractional shares at $10,000 (US-Y2), minimum trade $100;
  - no sector cap (no point-in-time sectors) and no metal sleeves;
  - regime on the S&P 500 and VIX;
  - risk-free 1.7%, the engine's constant, used only for walk-forward
    selection.
- **Costs:**
  - SEC and FINRA fees on sells, no commission (Vested, INDmoney);
  - a 3 bp spread floor plus square-root impact, at most 5% participation;
  - fills at the next open.
  - Not in the gates: FX, plan fees, TCS and tax (US3, UG11).
- **Walk-forward:** anchored, 4 training years from 2013, yearly test folds
  2017–25 (9). Each fold picks A, B or C by training excess Sharpe.
- **Gates.** Sharpe is excess over the dated 3-month T-bill (IRX / 252 a
  day). SPY is SPY total return on the same dates.
  - **G1:** deflated Sharpe ≥ 0.95 at N = 3.
  - **G2:** Sharpe above SPY's, 2013–25.
  - **G3:** alpha over SPY with Newey–West t ≥ 2 (lags
    ⌊4 (T/100)^(2/9)⌋).
  - **G4:** MaxDD no worse than SPY's, 2013–25.
  - **G5 (family):** walk-forward out-of-sample Sharpe above SPY's over
    2017–25.
  - **G6 (earns its place):** India 75% plus US 25% in INR, rebalanced
    monthly, has a higher monthly Sharpe over 6.5% than India 75% plus SPY
    25%, 2013–25.
    - The weights come from your capital: Rs 30 lakh in India (U6) and
      $10,000 in the US, about 23% US.
    - India is the deployed book's register backtest,
      run 20261010T032441064151Z_679cbd0c.
  - **Pass:** G1–G4 and G6, with G5 passing. If several pass, the one with
    the highest 2013–25 Sharpe goes forward.
- **A pass leads to:**
  - the 2026 holdout, opened once for the pick, after the store is
    extended;
  - US7: 60 sessions of forward test, G4-style;
  - US8: the emails move to the pick, on the union (US-Y9).
- **A fail:**
  - the emails keep their "unvalidated" banner and the US-Y8 rule;
  - the next family follows the golden-target plan (UG0, then UG7 or UG4),
    pre-registered as its own round;
  - A, B and C are never re-tuned.
- **Reported, not gates:**
  - INR results (US-Y2);
  - RSP, QQQ and QQQE;
  - correlation with the India book;
  - years, turnover and costs;
  - 2007–25 with 2008;
  - the family's PBO, which is coarse at N = 3;
  - the combined book against the golden target.
- **Code:**
  - `cloud.kaggle_runner us-round` runs the round on Kaggle, in one
    environment;
  - `python -m us_engine.backtest verdict` applies the gates above.

**Result (17 Oct 2026): FAIL, all three.** Run on Kaggle
(srees16/centurion-nse-research-us, 7 minutes after an hour in Kaggle's
queue); registry `runs_us` (39 runs). Figures are 2013–25, USD, with
Sharpe over the T-bill.

| | CAGR | Sharpe | MaxDD | Alpha (t) | DSR | Gates failed |
|---|---|---|---|---|---|---|
| A (deployed) | 5.6% | 0.36 | −25.7% | −1.2% (−0.41) | 0.82 | G1, G2, G3, G6 |
| B (candidate) | 6.3% | 0.43 | −16.0% | −0.1% (−0.03) | 0.88 | G1, G2, G3, G6 |
| C (E4) | 9.4% | 0.61 | −21.9% | +2.0% (0.68) | 0.96 | G2, G3, G6 |
| SPY | 14.6% | 0.79 | −33.7% | | | |

- **Walk-forward 2017–25 (G5): FAIL.** It picked C in every fold:
  - out of sample, 9.5% / Sharpe 0.54 / −24.5%, against SPY's 15.0% /
    0.73 / −33.7%;
  - 4 of 9 years lost money.
- **Earns its place (G6): FAIL.** India 75% plus US 25%, in INR, monthly
  Sharpe: with A 1.01, with B 1.03, with C 1.07. With SPY it is 1.14.
- **What the runs say.** The India rules make 6–9% a year on US large caps,
  with a beta of about 0.4. The S&P 500 regime gate kept the book in cash
  through rallies that India's gate would have caught:
  - SPY: 2019 +31%, 2021 +29%, 2023 +26%;
  - C, the same years: +16%, +18%, +1%.
  Stock selection added no alpha. Only drawdowns were smaller, and that
  was cash, not edge.
- **Diagnostics (reported, not trials):**
  - 2007–25: A 5.1%, B 5.6%, C 7.2%, against SPY 10.7%. In 2008 the
    books lost 3–7% while SPY lost 37%.
  - In INR, C made 13.7% a year.
  - Correlation with the India book: 0.17–0.20.
  - RSP 12.0% / 0.64, QQQ 19.6% / 0.88, QQQE 14.6% / 0.71.
  - Turnover about 10x, costs 0.7–0.8% a year.
  - PBO is 0.2%, meaningless at N = 3.
- **What carries forward:**
  - the US core family is closed (3 of 3 used), and A, B and C are never
    re-tuned;
  - SPY itself is the US leg to beat; it already raises the India book's
    Sharpe (G6);
  - the US-A1 emails run A, so US8's rule applies: stop them, or replace
    them, pending your call;
  - the next family follows the golden-target plan (UG0, then UG7 or
    UG4), pre-registered as its own round.

**Correction (17 Oct 2026, US3), dated as section 4 requires.** The round
ran without a broker commission: US4 took Vested and INDmoney to be
commission-free. They charge 0.25% a trade (Vested Premium 0.15%) and
0.295% (INDmoney). Charged on the recorded orders, 2013–25 CAGR in USD
becomes:
- A: 0.0% / 1.9% / −1.1% (Vested Basic / Premium / INDmoney);
- B: 0.7% / 2.5% / −0.4%;
- C: 4.4% / 6.1% / 3.5%.

The verdict stands: FAIL, by a wider margin. The runs are not re-recorded.
Future US backtests charge the commission (`us_engine.costs.BROKERS`).

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
1. The deployed file `config/nse_engine_deployed.json` (status `approved`) changes
   only through the forward gate below; commit it after `promote` writes it.
2. GitHub: repository variable `CENTURION_NSE_ENGINE=true`; secrets as for the
   legacy cron. First run with the `full_bootstrap` dispatch input: syncing from
   the 2011 anchor is ≈ 3,650 sessions × 4 files at 2 req/s (≈ 2 h), so it may
   need a second dispatch — the archive cache keeps progress.
3. Daily job (19:30 IST): sync NSE archives → rebuild current-year store →
   same-period shift reference → decide after close → fill pending orders at
   the next session's open → GTT-style stops at min(open, stop) → snapshot.
4. The job runs only while the paper switch in Neon (`paper_trading_state`,
   toggled from the Trade Center page or `POST /api/paper-trading
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

Where to watch: https://centurion-core-fe.vercel.app/ind-stocks/trade-center —
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

Pass criteria after 60 trading days (extend to 90 if borderline): the paper
gate (G4, `nse_engine/paper_gate.py`), fixed on 28 Sep 2026 before either
book reached its sample. Each book is compared with the same-period backtest
of what it trades: the engine config plus the deployment's drawdown overlay,
so a halt in the book is a halt in its reference too.

| Check | Measure | PASS | FAIL |
|---|---|---|---|
| Tracking error | annualised sd of (paper − backtest) daily returns | ≤ 8%/yr | > 12%/yr |
| Daily gap | mean of (paper − backtest), with its t-statistic | ≥ −3 bp/day | < −3 bp/day and t ≤ −2 |
| Costs | paper cost per rupee traded ÷ the backtest's, same days, ≥ 20 fills | ≤ 1.5× | > 2.0× |
| Drawdown | paper MaxDD vs the backtest's MaxDD over the same days | ≤ max(1.5×, +2 pts) | > max(2×, +4 pts) |
| Regime break | sessions sized down by a drift `regime_break` among the last 20 | none | any |

Between the limits a check is WATCH. The gate reads NOT ENOUGH DATA below 30
aligned sessions; after that, FAIL if any check fails, PASS only if all five
pass, else WATCH. It runs in every daily session of both books (a line in the
daily email, a FAIL also as an alert), the latest result is kept in Neon
(`paper_cloud_state` key `paper_gate`) and is the weekly email's verdict, and
one command recomputes it (section 9). Two changes from the criteria first
written here: the drawdown is compared with the backtest over the same days
rather than "the backtest's worst drawdown of the same length", which tests
behaviour instead of plausibility; and WATCH bands were added so a borderline
book is extended to 90 sessions rather than failed. In paper the cost check
confirms the fill simulator matches the cost model; with live fills (L5) it
measures real slippage.

Found while building it: the drift detector's default reference path pointed
at `services/data/`, while the job writes `data/shift_reference_returns.csv`,
so the deployed book (which sets no `CENTURION_SHIFT_REFERENCE_CSV`) would
never have had a same-period comparison once its drift check started at
session 31. Fixed on 28 Sep 2026, before it mattered.

A 60-day paper Sharpe says little about skill (standard error ≈ 2); the gates
test whether live behaves like the backtest, which is what can be measured.

**Forward promotion gate (V3, decision U19, 28 Sep 2026).** A configuration
replaces the deployed one only by `promote`, and only from a paper slot
(`--candidate config/nse_engine_<book>.json --schema <book>`; default the
`candidate` book). It passes when all three hold
(`nse_engine/forward_gate.py`):

| Check | Rule |
|---|---|
| Paper sessions | ≥ 60 candidate sessions, and the deployed book ran ≥ 60 of those same sessions |
| Paper gate | the candidate book's latest G4 report (Neon state) is PASS and covers its latest session |
| Walk-forward OOS Sharpe | candidate excess Sharpe over the walk-forward test years 2017–2025 ≥ the deployed config's − 0.05 |

A fixed configuration has no parameters left to choose, so its walk-forward
out-of-sample returns are its returns in the walk-forward's test years; both
configurations are backtested fresh on the current store and compared on the
same years. That check is weak evidence, since both were validated on those
years: it only stops a candidate that is clearly worse in history. The paper
sessions are the out-of-sample test. Printed, not gating: PBO and deflated
Sharpe with their configuration counts, the benchmark gate, holdout
evaluations, and both books' paper returns over the common sessions. On 28
Sep the candidate `2d64ba4c` scored 1.389 against 1.348 for the deployed
`679cbd0c` over 2017–2025, so only the paper checks stand between it and
promotion (earliest ~24 Dec).

`promote` writes the candidate's engine, anchor and drawdown overlay into the
deployed file with `paper_start_date` = `--paper-start` (default today) and the
gate's results in `notes`. The candidate book keeps trading in its schema
until its file is retired or replaced, which is a separate decision.

## 7. Stage E — Live capital (after paper passes)

Real orders need `CENTURION_PAPER_TRADE=false` and `CENTURION_NSE_ENGINE_LIVE=true`
and an approved deployment.

The live path was exercised against a fake broker (L3, 27 Sep 2026), which
also found that the order service's market-hours guard would have refused
every end-of-day order; live orders now go as after-market orders (`amo`)
when the market is closed.

**Live session driver (L5, 28 Sep 2026).** `python -m
kite_connect.trading.live_session` runs one session of the live book after
the bhavcopy, as the paper job does: it reads what became of the previous
session's orders from the broker's order book (fills with slippage against
the open and statutory costs, partial fills, rejections, orders missing from
the book, sells made outside the engine), snapshots the book at the close
into its own Neon schema (`live`), plans with the drawdown rule replayed on
those snapshots, places the orders and reconciles the stops, records the
session and emails a report titled LIVE.

The live book is a ledger, not the account. Zerodha accounts hold other
investments, and the planner sells every holding it has no target for, so
the engine only ever sees the ledger: the capital given on the first
session (`--capital`, month 1 = ₹6 lakh), the cash its own fills leave, and
the quantities it bought, with each position's entry session and last stop
(LS1, 10 Oct 2026). Before LS1 every live holding reached the planner dated
"today", so its stop counted only from that night's close. Replaying the
deployed backtest's 2013–25 history through the live ledger now rebuilds
the engine's holdings (entry date, stop, quantity) on every one of 3,219
nights, with each stop GTT on Kite's 5-paise tick and the stop within half
a tick of the engine's; with the old book, 3,183 differ. A live stop is
never lowered: a GTT left below the ledger's last stop, a GTT book that
cannot be read, or a GTT that is gone (triggered without a sale, expired,
deleted) keeps the ledger's level with an alert, so the engine still exits
through it. Ledger stops follow the store's corporate-action adjustments
(each keeps the close it was set against), as the backtest's adjusted
prices do. Stops are reconciled only for
those symbols. Holding an engine symbol personally as well is not supported.

Before month 1: `live_session --dry-run --capital 600000` after every paper
session for at least a week, comparing its orders with the paper book's
queued orders for the same day. A dry run reads the broker and writes
nothing; real orders need `CENTURION_PAPER_TRADE=false`,
`CENTURION_NSE_ENGINE_LIVE=true`, an approved deployment and a Kite session,
or the session refuses to start.

**Daily login and static IP (U23, 28 Sep 2026).** Kite tokens expire at
06:00 IST and a scripted login breaks Kite Connect's terms, so a person logs
in once per trading day. On trading days an email brings the Kite login link
(09:03 IST, again at 17:33 if still missing). The link opens Zerodha's own
login page; Zerodha then redirects to `/ind-stocks/auth/callback` on the HF
Space, which stores the day's token in Neon, encrypted. The paper job's
"Live book - session" step then runs the session with that token, once per
session, sending Kite calls through an SSH tunnel to an Oracle Cloud Always
Free VM whose reserved IP is registered with Zerodha: since April 2026
Zerodha accepts API orders only from a registered static IP, one account per
IP (`deployment/oracle-proxy/README.md`). The session checks its egress IP
against the registered one before any order and refuses real orders on a
mismatch. The step is off until the repository variable
`CENTURION_LIVE_MODE` is `dry_run` or `live`. Without a login, nothing is
placed that evening and the GTT stops keep protecting the book. The one-time
setup is tracker item U24.

**Capital ladder (D3, 28 Sep 2026; `nse_engine/capital_ladder.py`).** Live
capital grows in four rungs of the ₹30 lakh decided in U6, evaluated in
every live session and written in the daily email:

| Rung | Capital | Share |
|---|---|---|
| 1 | ₹6,00,000 | 20% |
| 2 | ₹12,00,000 | 40% |
| 3 | ₹21,00,000 | 70% |
| 4 | ₹30,00,000 | 100% |

- **Step up** one rung only when you ask, by setting `CENTURION_LIVE_CAPITAL`
  to the next rung after the money is in the account, and only if the book
  has spent 20 sessions at its rung, its own G4 checks (tracking error, daily
  gap, drawdown, regime break) pass, the drawdown rule is normal and no kill
  criterion holds. The email says when it is allowed.
- **Step down** one rung automatically after 20 sessions at the rung if any
  G4 check fails. G4 compares the book with a backtest of the same days, so a
  market-wide crash that the backtest also suffers is no reason (U22).
  Lowering `CENTURION_LIVE_CAPITAL` steps down at once. The same evening the
  engine sells down to the new capital.
- **Kill criteria** are alerted, never automatic: drawdown above 1.5× the
  backtest MaxDD *and* worse than NIFTY's over the same days (U22: in a
  crash the book is judged against the market); regime_break in two
  consecutive sessions; costs above 2× the model. The response is
  `CENTURION_KILL_SWITCH=true`, which refuses new buys.
- **Flows are not returns.** A step records its deposit or withdrawal on that
  day's snapshot, and the equity history the drawdown rule and G4 read has
  them removed.
- **Go-live.** The first real session is refused unless the deployed paper
  book's G4 is PASS with at least 60 sessions, five scheduled dry runs
  finished clean, and the account has shown it can sell unattended: DDPI
  confirmed and a supervised AMO sell and triggered GTT sell COMPLETE on the
  same Kite user, recorded with `live_session --record-sell-path` (LN-T2);
  `CENTURION_GO_LIVE_OVERRIDE=true` overrides and the email says so. The Kaggle token (U2) and leverage (L4, NO-GO) stay manual checks.

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
0.5–0.75 CAGR points per 0.1× for about 3 points of MaxDD; tracker L4 and
L5), and anything tuned on 2026.

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
| Promote (forward gate) | `python -m runners.run_nse_engine promote --check` prints the gate; without `--check` it writes the deployed file when the gate passes. Needs `CENTURION_DATABASE_URL` and the local store |
| Shift reference | `python -m runners.run_nse_engine shift-reference --run-id <id> --start <paper start> --data-start 2012-01-02` (also writes `<out>_trades.csv` for the gate's cost check) |
| Paper gate (G4) | `python -m runners.run_nse_engine paper-gate` for the deployed book; add `--deployment config/nse_engine_<book>.json --schema <book>` for another paper book, e.g. `candidate` or `e4`. Needs `CENTURION_DATABASE_URL` and a store covering the paper sessions, or `--reference <returns csv>` |
| Haircut (D2) | part of `validate`: two unrecorded re-simulations (as recorded; 1-day lag + 2× impact), the difference applied to the recorded metrics, plus the CSCV selection ratio; `validation.json` → `haircut` |
| Inspect deployment | `python -m nse_engine.deployment show` |
