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
- **Not re-tuned:** any other window would be a new trial. The module was
  removed on 31 Oct 2026 under your failed-ideas rule; git history holds it.

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
  of the short's. *(Corrected 1 Nov 2026: this is our rule, not the book's. The
  book gives no such limit; its nearest is one worked example, ch. 12: $600
  sold, $200 paid. The rule as tested is unchanged.)*
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

**Clarification (1 Nov 2026, before any run; owner's decisions).** MV1 stays
one bundled configuration, and a pass or fail is for the bundle. These replace
rule 2 and add to rules 1 and 5. Page numbers are the book's printed pages.

**Why.** O9-A's next-open entry is a day later than the book's: "the point at
which you want to buy is when the stock moves above the pivot point on
expanding volume" (p. 203; pp. 223–229). Of its 47 missed signals, 33 had
closed more than 3% above the pivot on the signal day, and the median
overnight gap was 0.8%.

**Rule 2, replaced: a resting order at the pivot.**
- **Eligible:** at a rebalance, a name the core would buy with a VCP base
  (O9-A's swing, base, contraction and dry-up rules, `nse_engine.sepa`) whose
  close is ≤ pivot × 1.03.
- **Order:** a buy resting from the next session until the next rebalance,
  trigger at the pivot, limit at pivot × 1.03 (p. 224).
- **Fill:** on the first session whose high exceeds the pivot, at the higher
  of the open and the pivot. An open above the limit cancels the order.
- **Cost:** cost model 4 on that fill.
- **Stop:** from the fill session. If that session's low reaches the initial
  stop, the trade stops there **(A: the order within the bar is unknown, so
  the worse case)**.
- **Breakout volume is dropped (A).** The book's "expanding volume" needs a
  day's final volume, which is not known at the fill, so using it would be
  look-ahead. The base's dry-up test stays.
- **Unfilled:** the slot waits for the next rebalance, as in rule 2. Names
  already held are unaffected.
- **Live:** a delivery order resting as a BUY GTT, not an intraday trade.
  `gtt_stops` manages SELL stops only, so this is built only on a pass.

**Rule 1, added: the deceleration test (pp. 133–134, 138).** The book calls
"a material deceleration, with results trending sharply lower for several
quarters or longer" a red flag, and says one quarter "may not be a big deal".
- **Growth, smoothed as on p. 133:** G = profit of the latest two quarters ÷
  the same two quarters a year earlier − 1, read on the basis in 5z.
- **Fail:** G fell at each of the last two quarterly readings, and the latest
  G is at most half the G two readings back **(A: two falls for "several
  quarters", half from the book's cases of 50–60% to 20–30%, 80% to 28% and
  41% to 11%)**.
- **Scope:** new buys only; held names are never sold for it.
- **Missing data:** it passes when any of the 8 quarters is missing, or when a
  year-earlier sum or the G two readings back is not positive, as rule 1 does.

**Rule 5, added: the step-down on losing streaks (pp. 304, 307).** "If you
experience an abnormal losing streak, first scale down your exposure."
- **Streak:** fewer than 1/3 of the last 20 closed core trades made a profit
  net of costs **(A: 1/3 is the book's floor at 2:1, p. 301; 20 is the core's
  target positions)**. Closed trades are full exits of any reason; trims and
  sleeves do not count.
- **Step:** s starts at 0 and is reassessed at every fifth closed trade once
  20 have closed: in a streak s rises by 1, to at most 2; otherwise it falls
  by 1, to at least 0 **(A)**.
- **Size:** a new buy takes its target weight × ½^s, and keeps that scale
  until it exits, so a rebalance does not top it up **(A: the book cuts in
  stages, 5,000 to 2,000 to 1,000 shares)**. The money not used stays in cash.
- **Scope:** new buys only; held names are not trimmed. It applies alongside
  the 10-of-20 limit after the gate re-opens.

**Rule 2's base, added: a shorter swing at the right edge (17:46 IST, owner's
decision).** O9-A's ±5-session swings cannot confirm a final pause shorter
than about 6–10 sessions, yet the book's pivot is the tightest part of the
base (pp. 202, 227). If the stock breaks out first, the depth and dry-up tests
fall on the contraction before it. In O9-A, 71 of 80 trades showed only 2
contractions.
- **Base:** its start and contractions are found with ±5 swings, as in O9-A.
- **Right edge (A):** after the latest ±5 swing, ±3 swings
  (`confirmed_swings(high, low, 3)`, usable 3 sessions after they print)
  extend the same alternating sequence. The earliest final pause it can see
  is about 4 sessions.
- **Tests:** the contractions from that sequence count towards the 2–6 limit.
  The no-deeper, last ≤ 10% and dry-up tests run on the extended list, and the
  pivot is its latest swing high.
- **Otherwise:** with no ±3 swing after the latest ±5 swing, the base is
  O9-A's. O9-A's own configuration keeps ±5, and `SepaConfig` needs a new
  setting that defaults to off, so its hash does not move.

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

## 5ae. L7: 1.75x built from real NSE stock futures (India tracker L7; pre-registered 17 Oct 2026, 16:40 IST)

**Question.** L6 levered the whole book synthetically, at rf + 1%. Real
stock futures differ in four ways:
- they exist for only part of the book (about 31% of its weight in 2021);
- they trade in whole lots (median about Rs 6.5 lakh);
- they carry their own basis and roll monthly;
- they need margin, half of it in cash.

Does a book built from real contracts keep L6's 1.75x result?

- **Family:** leverage (new), registry `data/nse_engine/runs_leverage`. This
  round uses 2 configurations: N goes from 0 to 2.
- **Inputs:**
  - the recorded walk-forward out-of-sample positions 2017–25 (runs of
    `wf_stitched_r12a5`: the K5 deployed family, RR2 data, cost model 4;
    each fold starts from cash);
  - spot prices from the main store (dab49d3c);
  - stock futures from the F&O store (13bb08a45e61d147), 2013–25 only; 2026
    stays sealed;
  - corporate-action factors from the store; NSE adjusts futures on a
    bonus or split ex-date by the same factor.
- **Common rules:**
  - Target: 1.75x the book's exposure. The drawdown rule (20/30/35, half
    0.5, re-arm 60) runs on the levered NAV and scales every target.
  - Contracts: the near month, rolled at the close 3 sessions before
    expiry.
  - Trading days: a name is resized only when the book trades it, on a roll,
    or when the rule changes state.
  - Fills: whole lots, the nearest number. A trade fills at a synthetic
    open, the previous futures close x spot open / spot previous close (A:
    the F&O store has no opens). Positions are marked to the futures settle
    daily, with mark-to-market in cash.
  - Margin per name: clip(3.5 x 60-day daily volatility + 0.05, 0.20,
    0.50) of notional (A: no SPAN history).
  - Collateral: cash plus pledged shares at 85% (A: a 15% haircut).
  - Cash must cover at least half the margin (SEBI's cash rule), plus a
    buffer of 5% of NAV (A).
  - A shortfall at a close cuts the futures pro rata that day, counted as
    a margin call.
  - Costs, futures: `kite_connect.options.fno_costs.futures_charges` (Rs 20
    an order, STT on sells, exchange, SEBI, stamp, GST), plus impact from
    `nse_engine.costs` on the contract's 20-day median traded value.
  - Costs, cash shares: `nse_engine.costs`.
  - Idle cash earns nothing (cost model 4).
- **Configurations:**
  - **L7-A, overlay.** The cash book as recorded, scaled down only as far
    as the cash rule needs. Extra futures exposure, (L - 1) x the book's
    exposure, is spread over the book's F&O names in proportion to their
    weights.
  - **L7-B, replace.** F&O names are held only through futures, at L x
    their weight. Other names are held in cash at L x their weight, as far
    as collateral allows, cut pro rata when it does not.
- **Capital:** Rs 2 crore for the gate. Rs 1 crore, Rs 5 crore and
  fractional lots are reported as diagnostics, not trials.
- **Pass,** per configuration, at Rs 2 crore over 2017–25:
  - CAGR within 2 points of L6's 35.9% (>= 33.9%);
  - MaxDD within 3 points of L6's -39.1% (>= -42.1%).

  A pass leads to L9 (the futures paper book). If both fail, the
  real-contract route cannot deliver 1.75x at this capital; the result
  states what it does deliver.
- **Reported:**
  - leverage reached (mean and range) and the share of exposure in futures;
  - the largest single-name weight;
  - basis (carry) and roll costs a year;
  - margin calls and the cash used for margin;
  - tracking error against L6's synthetic series.
- **Environment (A):** this Mac. The inputs are recorded positions and price
  files; no engine decisions run, so the result is plain arithmetic. All
  of the family's trials run here.
- **Code:** `nse_engine/futures_book.py` (`python -m nse_engine.futures_book
  l7`).
- **Clarifications (17 Oct, 17:05 IST, before any recorded run):** debugging
  runs on 2017 only, unrecorded, settled three points.
  - The cash buffer applies only while futures are held. With that, the
    overlay at 1.0x reproduces the recorded book exactly (largest daily gap
    1.7e-16).
  - The equity impact model charges 6–8 bp a leg on liquid stock futures,
    so a monthly roll costs about 16 bp of notional. The gate keeps the
    model as pre-registered. A run with a 1 bp floor and a 10 bp
    coefficient is reported as a diagnostic, not a trial.
  - Exposure reached: about 1.5x in L7-A (1.75 x the book's gross of
    about 0.86) and about 1.3x in L7-B. The F&O share of the book limits
    L7-B.

**Result (17 Oct 2026): FAIL, both.** Runs are in `runs_leverage`
(recorded 17 Oct, this Mac). Figures are 2017–25 at Rs 2 crore, Sharpe over
6.5%.

| | CAGR | Sharpe | MaxDD | Calmar | Average exposure | In futures | Futures + roll costs a year |
|---|---|---|---|---|---|---|---|
| L6 synthetic 1.75x | 35.9% | 1.25 | −39.1% | 0.92 | 1.50x | | 0.75% carry |
| L7-A overlay | 29.6% | 0.98 | −40.0% | 0.74 | 1.38x | 0.66 | 2.0% + 2.4% |
| L7-B replace | 26.2% | 1.09 | −34.7% | 0.75 | 1.15x | 0.39 | 1.3% + 1.2% (cash legs 8.4%) |

- **What the runs say:**
  - Real contracts cost L6's 1.75x about 6 points of CAGR and 0.27 of
    Sharpe.
  - Most of the loss is cost: monthly rolls and resizing each name when
    the book trades it, about 4.4% of NAV a year, which grows as the book
    grows. The rest is the exposure the cash rule leaves short.
  - Lot rounding barely matters at Rs 2 crore: fractional lots give 29.7%.
- **Concentration:** about 31% of the book's weight has stock futures, so
  the overlay loads its extra exposure onto those names. One contract
  reached 35% of NAV. The overlay's cash rule sold slices of the book 71
  times.
- **L7-B** reaches only 1.15x. Collateral caps the cash legs, and at Rs 2
  crore the mid-cap legs pay realistic impact: 8.4% of NAV a year.
- **Diagnostics (not trials):**
  - L7-A at Rs 1 crore: 31.0% / 1.03 / −39.0%;
  - L7-A at Rs 5 crore: 27.6% / 0.92 / −39.8%;
  - L7-A with a 1 bp futures spread: 33.3% / 1.11 / −39.7%, still 0.6
    points short of the gate;
  - L7-B at Rs 1 crore: 27.5% / 1.20 / −32.4%.
- **Caveat:** the overlay's cash book and L6 both use the walk-forward's
  runs, which were sized at Rs 5 lakh. At Rs 2 crore the book's own
  impact would be higher, so both read somewhat high; L7-B shows the
  effect.
- **What carries forward:**
  - The real-contract route delivers about 29–31% CAGR at Rs 1–2 crore,
    with Sharpe about 1.0 and MaxDD about −40%, over 2017–25. That is not
    35%.
  - L9 (the futures paper book) was gated on a pass and does not start by
    rule. Running it anyway is your call.
  - The family has one configuration left. The visible lever is cost:
    resize only on weekly rebalance days, with a one-lot band, and roll
    into the second month. That would be pre-registered first.

## 5af. L7-C: the cost lever for 1.75x from real stock futures (India tracker L7-C; pre-registered 17 Oct 2026, 20:29 IST)

**Question.** L7-A's recorded costs over 2017–25 were 4.4% of NAV a year:
- resizing 2.0% (impact 1.9, charges 0.1), on 28% of sessions;
- rolls 2.4%.

How much of that can an execution rule recover without changing what the
book holds?

- **Levers weighed, from recorded data only (no L7-C run made):**
  - **Rolling every two months: dropped.** It needs the far month, which
    traded a median Rs 11 lakh a day on the last 2021 session, against Rs
    125 crore for the near month. A Rs 2 crore book's legs, about Rs 25
    lakh each, would be most of a day's volume.
  - **Staggered rolls: dropped.** The model prices both roll legs on the
    near month's traded value. Slicing a roll over earlier sessions would
    cut modelled impact while the next month is in fact thinner then, so
    the saving would be an artefact.
  - **Weekly-only resizing: dropped.** The book rebalances every 5
    sessions, and L7-A already resizes a name only when the book trades it.
    Waiting longer would delay new legs, losing exposure, for little
    saving.
  - **One-lot band: kept.** Each time the book trades a name, L7-A
    re-targets its leg to that day's NAV, and whole-lot rounding turns
    small drifts into 1-lot trades at the full spread and impact.
- **Configuration, L7-C:** L7-A's overlay with one change. A held leg is
  resized only when its new target, rounded to whole lots, differs from the
  held lots by 2 or more; it then trades to the target. These always trade:
  - a new leg;
  - a leg whose target is zero (the name left the book or the F&O list);
  - a change of drawdown-rule state;
  - a margin-call cut;
  - rolls, as in L7-A.

  Everything else is as 5ae: inputs, common rules, Rs 2 crore, costs, idle
  cash earning nothing.
- **Family:** leverage. N goes from 2 to 3, and the family then closes.
- **Pass:** 5ae's gate, at Rs 2 crore over 2017–25: CAGR >= 33.9% and
  MaxDD >= −42.1%.
- **Expectation, stated before the run:**
  - The band reaches only part of the 2.0% resizing cost; entries and
    exits still trade in full.
  - Expected gain: 0.5–1 point, so about 30–31%, a FAIL.
  - A pass would need 4.3 points, more than the whole resizing cost.
- **Reported:**
  - as 5ae;
  - the resizes the band skipped, and the cost split;
  - diagnostics, not trials: Rs 1 crore and Rs 5 crore.
- **Then the 2026 holdout, opened once for the family:**
  - **Inputs:**
    - the deployed book's recorded holdout run,
      `20260915T161516314162Z_679cbd0c` (1 Jan–11 Sep 2026, 172 sessions,
      opened 15 Sep, the one L7a used; recorded on cost model 1 and pre-B1
      data, used as recorded);
    - the F&O store's 2026 file, unseen until then.
  - **One look, all three configurations:** L7-A, L7-B and L7-C, beside
    L7a's synthetic (+28.0%, MaxDD −22.2%) and the unlevered book (+18.0%,
    −13.0%).
  - **Read:** the drag in 2026 (synthetic minus real contracts) against
    2017–25's, about 4 points pro rata over 8 months.
  - **If L7-C passes the gate,** the holdout must agree:
    - its 2026 return is at least +23.0% (the unlevered +18.0% plus half of
      L7a's 10-point uplift);
    - its MaxDD is no worse than −25.2% (L7a's −22.2% less 3 points).
  - **If the family fails,** the look is a diagnostic for L9 and L10, and it
    does not reopen the family.
- **Environment (A):** this Mac, as 5ae.
- **Code:** `nse_engine/futures_book.py`:
  - a `band_lots` setting (0 keeps L7-A's and L7-B's hashes);
  - `python -m nse_engine.futures_book l7c`;
  - then `holdout`.

**Result (17 Oct 2026): FAIL. The leverage family closes, 3 of 3 failed.**
The run is `runs_leverage/20261017T150734009177Z_a8061af2` (this Mac).
Figures are 2017–25 at Rs 2 crore, Sharpe over 6.5%.

| | CAGR | Sharpe | MaxDD | Resizes | Skipped by the band | Resizing + rolls a year |
|---|---|---|---|---|---|---|
| L6 synthetic 1.75x | 35.9% | 1.25 | −39.1% | | | |
| L7-A (re-run, identical to the record) | 29.6% | 0.98 | −40.0% | 1,374 | | 2.0% + 2.4% |
| L7-C band | 29.5% | 0.98 | −39.8% | 1,312 | 83 | 2.0% + 2.4% |

- **The expectation (+0.5–1 point) was wrong: −0.1 point.** Almost every
  resize moves 2 lots or more (entries, exits, the book's reweights), so
  the band skipped 6% of them and saved nothing.
- **Diagnostics (not trials):** Rs 1 crore 30.5% / 1.02 / −39.2%; Rs 5
  crore 27.6% / 0.92 / −39.6%.

**2026 holdout, opened once (17 Oct, lock `runs_leverage/holdout.lock`).**
The deployed book's recorded run, 1 Jan–11 Sep 2026 (172 sessions), and
the 2026 F&O file.

| 1 Jan–11 Sep 2026 | Return | Sharpe | MaxDD | Average exposure | Costs a year (resizing + rolls) |
|---|---|---|---|---|---|
| Unlevered book | +18.0% | 1.04 | −13.0% | | |
| L6 synthetic 1.75x | +28.1% | 1.08 | −22.2% | | |
| L7-A overlay | +19.4% | 0.84 | −19.8% | 1.30x | 1.5% + 1.4% |
| L7-B replace | +26.4% | 1.08 | −21.8% | 1.14x | 1.5% + 0.9% |
| L7-C band | +18.9% | 0.81 | −20.0% | 1.33x | 1.3% + 1.6% |

- **The overlay lost about 9 points to the model in 8 months,** twice the
  rate 2017–25 implies. Costs explain about 2 points. The rest is
  composition: the overlay puts all the extra exposure on the book's F&O
  names, and those are the book's weaker half.
  - 2026: the F&O names (43% of the weight) made +15.3%, the rest +45.2%.
  - 2017–25: the F&O names (36%) made 19.5% a year, the rest 38.0%.
  - The book's edge sits in names that have no futures.
- **L7-B came within 1.7 points of the model** at lower exposure: it
  levers every name at its own weight, so it carries no such bet.
- **L7-C's check** (+23.0%, −25.2%) fails, but binds only on a gate pass.
- **What carries forward:**
  - real contracts deliver 26–30% in 2017–25, not 35%;
  - stock futures can lever only the book's weaker half, which the
    synthetic model hides;
  - L9's paper book uses the overlay. The replace design tracked the model
    better in 2026 and had the better Sharpe and MaxDD in 2017–25.

## 5ag. BD1: government bonds as a return stream (India tracker BD1; pre-registered 17 Oct 2026, 21:35 IST)

**Question.** Do Indian government bonds, held through an exchange-traded
gilt ETF, add to the book over 2008–25? Three forms: on its idle cash,
timed by their own trend, or as a fixed sleeve.

**Why it should exist.** Gilts pay a term premium over cash, and they rally
when equities crash and the RBI cuts: the 10-year index made +27.5% in
2008. The risk is rate shocks (2009 −7%, the 2013 taper).

- **Family:** bonds (new), registry `data/nse_engine/runs_bonds`, at most 3
  configurations. N goes from 0 to 3.
- **Data:** the bond store (`data/nse_engine/bond_store`, hash
  520b9eba4594fd89; `nse_engine/data/bond_store.py`).
  - **10-year gilts:** NSE's 10-year benchmark G-sec index, total return,
    from 3 Jan 2011. Before it, the NAV of ABSL's 10-year constant-maturity
    gilt fund (AMFI 100061). Over 2011–13 it tracked the index with
    correlation 0.95 and 2.1% tracking error.
  - **Overnight:** NSE's 1D rate index from 3 Jan 2011. Before it, SBI
    Liquid's NAV (AMFI 105274, correlation 0.92). A liquid fund holds paper
    out to 91 days, so cash reads about a point generous before 2011, which
    makes the test harder for bonds.
  - The proxies were chosen on the 2011–13 overlap, before any stream
    result.
- **The book, two bases (your rule 4: backtest, then walk-forward):**
  1. **Backtest:** the deployed rules with the drawdown rule, 2008–25. Run
     `20261017T110545845367Z_2cb1c053` (store_ext2006 01927c09, cost model
     4).
  2. **Walk-forward:** R4's 2008–25 out-of-sample returns
     (`wf_stitched_r4b`), with each fold's weights.

  Idle cash is 1 − the book's invested weight, each session: 19.6% on
  average in the backtest, about 70% in 2008; 30% in the walk-forward.
- **Benchmark for alpha:** NIFTY 50 TRI. Before 2012 the store has no TRI,
  so NIFTY 50's price index plus 1.3% a year of dividends stands in (A).
- **Instrument and costs: the bond cost model.**
  - **Vehicle:** a 10-year gilt ETF bought for delivery on Kite (the
    LTGILTBEES / SETF10GILT class). Its return is the G-sec index less a
    0.15% expense ratio (A).
  - **Statutory charges** (`nse_engine.costs.statutory_cost` with the
    ETF's symbol; debt ETFs added on 17 Oct):
    - no STT (a debt ETF);
    - stamp duty 0.015% on buys (no debt-ETF rate is published, so the
      equity delivery rate);
    - NSE's exchange charge, the SEBI fee, GST, and the DP charge on sells.
  - **Spread and impact:**
    - a 10 bp floor: gilt ETFs quote wider than shares, and LTGILTBEES's
      median day range is 43 bp;
    - plus `nse_engine.costs`'s square-root impact, 25 bp at 1%
      participation, on Rs 3.1 crore a day of traded value: LTGILTBEES's
      2022–25 median (A). Before 2022 no liquid gilt ETF existed, so the
      backtest assumes today's liquidity.
    - Orders are capped at 5% of that a day; the rest waits for the next
      session.
  - **The overnight vehicle** (B2's off state and the baseline): a liquid
    ETF at the 1D rate less a 0.10% expense ratio, the same statutory
    charges, and a 3 bp floor (LIQUIDCASE's median day range is 2 bp).
  - **Capital:** Rs 35 lakh, the paper book's. Rs 1 crore and Rs 2 crore
    are diagnostics.
- **Common rules:**
  - The bond position never exceeds the book's idle cash less a buffer of
    2% of NAV. When idle cash shrinks below it, gilts are sold that session,
    because the book's buys need the cash.
  - Gilts are bought only when the gap to the target exceeds 5% of NAV (a
    band against churn).
  - Trades and marks are at the session's close (the index has no open).
  - Combined NAV is the book's recorded NAV plus the bond P&L less costs.
    The book's own path, its idle cash earning nothing (cost model 4) and
    its drawdown rule are as recorded. Bonds change none of its decisions.
- **Configurations:**
  - **B1, idle cash in gilts:** the target is idle cash less the buffer.
  - **B2, idle cash in gilts while their trend is up:** gilts while the
    G-sec index is above its 252-session average (12-month time-series
    momentum, the literature's horizon, not fitted); otherwise the idle
    cash goes to the liquid ETF.
  - **B3, a fixed 25% sleeve:** 0.75 × the book + 0.25 × gilts (index less
    expense ratio), rebalanced on each month's first session, with costs on
    the rebalancing trades. This is the protocol's sleeve form.
- **Baseline, reported, not a trial:** idle cash in the liquid ETF (LN-T29's
  plan).
- **Gates,** each on both bases over 2008–25:
  - **Gate 1, alone:** deflated Sharpe >= 0.95 at the family's N, excess
    over 6.5%, on the return per rupee of the position (gilts, or the liquid
    ETF while B2 is out), less costs.
  - **Gate 2, beside the book:** the combined book raises CAGR, excess
    Sharpe and alpha against NIFTY 50 TRI, with MaxDD no more than 2 points
    worse.
  - **Bond-specific:** B1 and B2 must also beat the liquid baseline's excess
    Sharpe. Otherwise any yield on idle cash would pass gate 2.
  - A configuration passes when every gate holds on both bases.
- **Expectation, stated before the run:**
  - Gilts alone earn an excess Sharpe of about 0.1–0.3 over 6.5%, so gate
    1 fails for all three. Protocol section 5 parks the question of judging
    a diversifier on gate 2 plus a significance test with you (OX6).
  - B1: +1–2 points of CAGR, and a much shallower 2008. 2009 and 2013 cost.
  - B3: about 1.5 points less CAGR and a MaxDD several points better. Gate 2
    probably fails on CAGR.
- **Then, for a configuration you accept:** the 2026 holdout, opened once
  for the family; 60 paper sessions; 5 dry runs (your rule 4).
- **Reported:**
  - each year, especially 2008, 2009, 2013 and 2022;
  - the worst drawdown with and without bonds;
  - correlation with the book;
  - turnover and costs a year, and how much of the time gilts are held;
  - the 2013–25 and 2017–25 slices.
- **Environment (A):** this Mac, as 5ae. The inputs are recorded returns,
  weights and index files; no engine decisions run, so the result is plain
  arithmetic.
- **Code:** `nse_engine/bond_stream.py` (`python -m nse_engine.bond_stream
  bd1`).
- **Clarification (17 Oct, before the recorded run):** mechanical checks
  on 2017 only, unrecorded. With zero bond returns and zero costs, the
  combined book reproduces the recorded book (largest daily gap 2e-16).
  NIFTY 50 before 2012 is back-filled from the store's external history,
  as the engine's loader does.

**Result (17 Oct 2026): FAIL, all three. The bond family closes at N = 3.**
Runs are in `runs_bonds` (this Mac). Figures are 2008–25 at Rs 35 lakh,
compounding, with Sharpe over 6.5%.

| | Backtest base: CAGR / Sharpe / MaxDD | 2008 | Walk-forward base: CAGR / Sharpe / MaxDD | 2008 |
|---|---|---|---|---|
| Book alone | 16.8% / 0.78 / −31.9% | −26.9% | 13.2% / 0.62 / −32.5% | −29.7% |
| Baseline: idle cash in a liquid ETF | 16.4% / 0.76 / −28.7% | −23.3% | 13.2% / 0.61 / −29.7% | −26.8% |
| B1 idle cash in gilts | 14.5% / 0.62 / −31.0% | −19.3% | 10.8% / 0.40 / −32.2% | −22.7% |
| B2 gilts while trending up | 14.1% / 0.59 / −31.0% | −19.3% | 10.0% / 0.34 / −32.2% | −22.7% |
| B3 25% gilt sleeve | 14.4% / 0.78 / **−22.1%** | **−15.0%** | 11.7% / 0.61 / **−22.9%** | **−18.0%** |

- **Gilts alone: CAGR 6.5%, Sharpe 0.01 over 6.5%, MaxDD −12.8%; DSR
  0.45.** Gate 1 fails, as expected.
  - Over 2008–25 the 10-year index made 6.7% a year against 6.3% for
    overnight money: a term premium of about a third of a point.
  - Correlation with the book is −0.09 (monthly).
  - 2008: +27.5% while NIFTY lost 51%; 2009 −7%.
- **B1 and B2 fail on costs, not on bonds.**
  - The book's idle cash swings between 0% and 100% (the drawdown rule,
    the regime scale, weekly rebalances), so following it trades 6–10
    times NAV a year.
  - At the pre-registered impact (the equity model, about 20–50 bp a side
    as the book grows past Rs 1 crore), that costs 3.5–4.7% of NAV a year
    for gilts and 1.4% even for the liquid baseline.
  - 2008 improved by 7–8 points all the same.
- **B3 is the robust finding.**
  - A 25% gilt sleeve trades 0.1 times NAV a year and costs 0.05%.
  - It cuts the worst drawdown by about 10 points on both bases, with
    Sharpe unchanged, for 2.4 (backtest) and 1.5 (walk-forward) points of
    CAGR.
  - 2013–25: 17.6% / 1.03 / −17.3% against the book's 21.2% / 1.04 /
    −25.1%. 2017–25: 18.5% / 1.16 / −16.4% against 22.9% / 1.20 / −25.1%.
  - It fails gate 2 on CAGR, as expected.
- **Diagnostics, not trials:**
  - **Spread floor only, no square-root impact:**

    | | Backtest: CAGR / Sharpe / MaxDD | 2008 | Walk-forward: CAGR / Sharpe | Costs a year |
    |---|---|---|---|---|
    | Liquid baseline | 17.7% / 0.85 / −28.4% | −23.0% | 14.8% / 0.74 | 0.3% |
    | B1 | 17.9% / 0.85 / −30.2% | −17.4% | 14.5% / 0.70 | 0.6–0.7% |
    | B2 | 17.7% / 0.83 | | 14.3% / 0.68 | 0.7–0.9% |

    So with cheap execution, parking idle cash adds about 1–1.6 points.
    Gilts do no better than liquid on Sharpe: they buy a better 2008 with
    a worse 2009, 2013 and 2022.
  - **Rs 1 crore and Rs 2 crore** read better than Rs 35 lakh (B1 backtest
    15.4% and 16.4%). The 5% daily participation cap then throttles the
    churn: 430 days were capped at Rs 35 lakh.
- **What carries forward:**
  - Bonds are a drawdown tool for this book, not a return stream. B3's
    trade (about −2 points of CAGR for about −10 points of MaxDD, Sharpe
    flat) is the one the golden target's MaxDD leg can use.
  - The cost of trading debt and liquid ETFs must be measured (quotes or
    paper fills), not taken from the equity impact model. That also
    decides LN-T29 (the liquid sweep): under this model it loses 0.4
    point; with spread-only costs it gains about 1.
  - **Correction, 25 Oct 2026 (LN-T29d, a diagnostic):** the baseline row
    above prices a liquid ETF with the equity impact model (12–20 bp a
    trade; the daily-bar bounds put liquid ETFs at 0.1–5 bp). Re-priced at
    statutory charges plus 1–5 bp, LIQUIDCASE's 0.27% expense ratio, this
    plan's hysteresis and cash freed a session ahead, the sweep adds 1.3–1.6
    points on the walk-forward book and 0.8–1.0 on the backtest book, with
    MaxDD about 3 points shallower. BD2's measured spreads replace the
    1–5 bp range.
  - A low-churn rule for idle cash is the visible fix: park only cash
    that has stayed idle for N sessions, or band both sides. It would be a
    new round, and the family's budget is spent: your call.

**BD3, the 2026 holdout for B3 (fixed 18 Oct 2026, 22:00 IST, before it is
opened).** You accepted B3 as the drawdown lever on 17 Oct.
- **Inputs:**
  - the deployed book's recorded holdout run,
    `20260915T161516314162Z_679cbd0c` (1 Jan–11 Sep 2026, 172 sessions,
    opened 15 Sep). L7a and the leverage family used it too. It was
    recorded on cost model 1, so its idle cash earns 6%, and on pre-B1
    data. It is used as recorded;
  - the bond store's 2026 rows, unseen until now.
- **One look:** B3, with B1, B2 and the liquid baseline beside it, at Rs 35
  lakh. The lock is `runs_bonds/holdout.lock`.
- **B3 agrees** if both hold over the window:
  1. its MaxDD is shallower than the book's;
  2. its return is within 1 point of the costless 75/25 monthly blend of
     the book and gilts, so execution costs no more than the model says.
- **If it agrees,** BD3 moves to 60 paper sessions. **If not,** BD3 stops
  and the result goes to you. The family stays closed either way.

**BD3 holdout result (18 Oct 2026): B3 agrees.** It was opened once; the lock
is `runs_bonds/holdout.lock`.

| 1 Jan–11 Sep 2026 | Return | Sharpe | MaxDD | Costs over the window |
|---|---|---|---|---|
| Book | +18.0% | 1.04 | −13.0% | |
| Gilts (index less expense ratio) | +1.4% | −1.24 | −2.8% | |
| Liquid baseline | +17.2% | 0.99 | −13.1% | |
| B1 idle cash in gilts | +13.0% | 0.70 | −14.3% | 3.7% of NAV |
| B2 gilts while trending up | +11.0% | 0.56 | −14.3% | 5.0% |
| **B3 25% gilt sleeve** | **+13.6%** | 0.92 | **−9.8%** | 0.19% |
| B3's costless blend | +13.8% | 0.94 | −9.9% | |

- **Both tests hold:** MaxDD −9.8% against −13.0%, and the return is 0.2
  point from the costless blend.
- **2026 was a poor year for gilts** (+1.4% in eight months), so the sleeve
  cost 4.4 points of return for 3.2 points less drawdown. That is the price
  of the hedge in a year without a crash.
- **B1 and B2 churned again** (3.7–5.0% of NAV in eight months), which
  confirms BD1 out of sample.
- **Next: 60 paper sessions.** The paper book is `config/nse_engine_bonds.json`
  (`kite_connect/trading/bond_sleeve.py`), Rs 35 lakh: shares on Rs 26.25
  lakh, LTGILTBEES on Rs 8.75 lakh.
  - The split is static in paper: the paper share book cannot pass cash to
    the sleeve, so the monthly rebalance waits for the live ledger.
  - Then 5 dry runs (your rule 4).

## 5ah. DG1: re-testing streams under the diversifier gate (India tracker DG1; pre-registered 19 Oct 2026, 17:45 IST)

**Why.** Your decisions of 19 Oct:
- the diversifier gate (protocol 5.1) replaces "must raise CAGR" and the
  0.95 deflated Sharpe for streams beside the book;
- re-test the three options near misses;
- MCX trend joins bonds as a drawdown lever.

**What is not clean (stated before the run).** My 19 Oct look combined all
17 options strategies with the honest book over 2013–25. It showed Y3, A1
and X2 adding the most, at weights near this gate's caps. Their 2013–25
numbers are therefore selected, not out of sample. The out-of-sample
evidence is:
- 2008–12 for A1 and X2 (their recorded 2007–25 runs; never combined
  before);
- the 2026 holdout for all three (the options family's, unopened).

MCX's 2007–25 sleeve figures were reported in 5y. Its combinations with
the book over 2008–25 are new.

- **Streams (recorded configurations; no new parameters, so no family's N
  changes):**
  - **Sharpe-lever candidates, as overlays** (margin pledged from the
    book):
    - O2-A1, the 15-day 2/3-SD bear call spread (LN-T20 re-run,
      2007–25);
    - O3-X2, TRIN bands on the near-month NIFTY future (2007–25);
    - O4-Y3, NIFTYBEES from the close to the next open on two sentiment
      conditions (from Feb 2012; its data starts then). It trades every
      night, bought at the close and sold at the next open: a delivery
      position held overnight, never a same-day exit. Live, each held
      night is an order pair at the close and the open (market-hours
      events). Its exposure above the book's idle cash would be carried by
      NIFTY futures (A).
  - **Drawdown-lever candidates, as funded sleeves:** GT5-A and GT5-B, MCX
    trend. The 2007–25 series is re-run on Kaggle (`mcx` task,
    `--reported-start 2007-01-01`, the same configurations). Margin and
    idle cash earn the liquid rate.
- **Books:**
  1. the deployed rules with the drawdown rule, 2008–25 (run
     `20261017T110545845367Z_2cb1c053`);
  2. the 2008–25 walk-forward (`wf_stitched_r4b`).

  The book's idle cash earns the liquid rate too, in both the base and the
  combination. Y3 is judged from its first session.
- **Weight (protocol 5.1):** equal risk on trailing volatility (up to 252
  sessions, at least 63), re-set on each month's first session. Caps: an
  overlay at 3×, a sleeve at 50%. The tested weight (overlay 0.25, sleeve
  25%) is reported beside it.
- **Routes (both books):**
  - **Sharpe lever:** combined excess Sharpe above the book's. One-sided p
    ≤ 0.10 in a paired stationary bootstrap (63-session mean blocks,
    2,000 resamples, seed 7). MaxDD no more than 1 point worse.
  - **Drawdown lever:** MaxDD at least 3 points shallower, Sharpe no more
    than 0.02 lower.
- **Extra condition for A1 and X2 (their 2013–25 was seen):** their
  combined Sharpe must not fall in 2008–12 on either book.
- **Reported:**
  - CAGR, alpha against NIFTY 50 TRI, correlation;
  - 2008 and each year;
  - the weights reached and how often a cap bound;
  - the deflated Sharpe at the family's N (options 17, commodity 2).
- **Then the 2026 holdout, opened once per family, for the passers:**
  - **Options:** the passers re-run on Kaggle over 1 Jan–11 Sep 2026,
    beside the deployed book's recorded holdout run
    (`20260915T161516314162Z_679cbd0c`, cost model 1, as recorded). A pass
    agrees if, at its last 2008–25 weight, the combination's 2026 Sharpe
    is at least the book's and its MaxDD is no more than 1 point worse
    (Sharpe lever), or its MaxDD is shallower (drawdown lever).
  - **MCX:** its family opened 2026 on 12 Oct (A +21.3%, B +18.1%, NIFTY
    50 TRI −13.0%). That look cannot be repeated. The combination over
    2026 is reported, not judged.
- **Then, for an agreeing pass:** 60 paper sessions and 5 dry runs (your
  rule 4).
- **Environment (A):** Kaggle for the MCX re-run (an engine-like
  simulation). The combinations are arithmetic on recorded series, on
  this Mac (protocol 3).
- **Code:** `nse_engine/diversifier_gate.py` (`python -m
  nse_engine.diversifier_gate dg1`).
- **Clarification (19 Oct 2026, 18:20 IST, your instruction, before any
  combination was computed): honest, live-like figures only.**
  - **The book's part carries D2's measured live haircut:** a day's delay
    and twice the impact, 3% a year at full exposure, scaled by the
    invested weight. It applies in the base and the combination alike, and
    the gates are judged on these figures.
  - **The walk-forward book is the headline.** The deployed-rules backtest
    is in-sample for 2013–25, so it is only an extra condition: a pass
    still needs both books.
  - **Streams run as recorded, with their modelled costs.** Their own live
    gap is unmeasured (stated).
  - **The selected years (2013–25 for the options streams) are reported
    apart** from the out-of-sample ones (2008–12 for A1 and X2; 2026).
  - **The MCX re-run** (Kaggle, `centurion-nse-research-dg1mcx`, store
    8350d7128f3d6fe6) reproduces GT5: 2013–25 2.4% and 3.4%; 2007–25 1.3%
    / MaxDD −71% and 3.1% / −54%.

**Result (19 Oct 2026): FAIL, all five.** Nothing passes either route on
either book, so no 2026 holdout is opened: the options family's stays
sealed. Results are in `runs_options/dg1_results.json` (copied to
`runs_mcx`).

Figures are live-like: the walk-forward book, idle cash at the liquid rate,
and D2's haircut. Shown as CAGR / Sharpe over 6.5% / MaxDD, 2008–25 (Y3
from Feb 2012).

| | Walk-forward book | + stream at equal risk (weight) | Change in Sharpe (p) | + stream at 0.25 | Why it fails |
|---|---|---|---|---|---|
| Book alone | 13.0% / 0.60 / −29.5% | | | | |
| O2-A1 call spread | | 15.5% / 0.76 / −34.6% (2.7×) | +0.16 (0.03) | 13.2% / 0.61 / −30.0% | MaxDD 5 points worse: it lost 7.8% in the 2008 crash |
| O3-X2 TRIN timing | | 18.6% / 0.78 / −31.9% (1.4×) | +0.19 (0.20) | 13.6% / 0.63 / −34.8% | Not significant; worse in 2008–12 |
| O4-Y3 overnight (book from 2012: 16.4% / 0.86 / −20.9%) | | 24.0% / 1.08 / −27.1% (2.6×) | +0.22 (0.15) | 17.3% / 0.92 / −20.8% | Not significant over 14 years; MaxDD 6 points worse (2015–16) |
| GT5-A MCX trend | | 11.6% / 0.48 / −29.7% (30%) | −0.12 | 12.4% / 0.56 / −25.2% | Sharpe falls; MaxDD no better at equal risk |
| GT5-B MCX trend | | 10.8% / 0.45 / −26.5% (37%) | −0.15 | 12.6% / 0.63 / −20.7% | Sharpe falls at equal risk |

The backtest book (15.5% / 0.70 / −28.6%) gives the same verdicts. A1's p
there is 0.044, X2's 0.23, Y3's 0.13.

**Correction, 25 Oct 2026 (LN-T29d).** "Idle cash at the liquid rate"
credits every idle rupee with the 1D rate less 0.10%, every day, at no
cost. A realistic LIQUIDCASE sweep earns less:

| Live-like 2008–25 | Walk-forward book | Backtest book |
|---|---|---|
| As recorded (cost-free credit) | 13.0% / 0.60 / −29.5% | 15.5% / 0.70 / −28.6% |
| Realistic sweep (3 bp; 1–5 bp gives ±0.1 point) | 12.4% / 0.54 / −29.9% | 14.9% / 0.66 / −29.0% |
| No sweep (paper and live today) | 10.9% / 0.43 / −33.0% | 14.0% / 0.60 / −32.4% |

The verdicts above compare the book with and without a stream on the same
base, so they are expected to stand. They are re-checked on cost model 5
(LN-T29), not re-run now.

- **A1 is a real Sharpe lever with a drawdown cost.** Its gain is
  significant and holds in 2008–12. It fails only the MaxDD condition:
  short call spreads lose in sharp moves.
- **X2 and Y3 have Sharpe-sized point estimates, not evidence.** Y3 has no
  2008, and its 2013–25 was selected.
- **Equal risk is the wrong weight for a low-Sharpe drawdown lever.** It
  puts 30–40% into MCX trend, whose Sharpe on its own is 0.43–0.59 against
  the liquid rate, and the book's Sharpe falls.
- **At its own 25% (fixed in 5y, not chosen now), GT5-B meets the drawdown
  route on both books:**
  - walk-forward: MaxDD −29.5% → −20.7%, Sharpe +0.03;
  - backtest: −28.6% → −21.7%, Sharpe +0.02.

  That is reported, not judged: judging it at 25% now would change the
  weight after the result. Your call.
- **Reported:**
  - deflated Sharpe at the family's N: A1 0.14, X2 0.18, Y3 0.93, GT5-A
    0.57, GT5-B 0.71;
  - monthly correlations with the book, −0.05 to 0.17.

## 5ai. DG2: drawdown levers at their own weight (India tracker DG2; your decision 19 Oct 2026, 18:45 IST)

**Why.** You adopted protocol 5.1's drawdown route at a stream's own fixed
sleeve weight, with three safeguards:
- a bootstrap check of the MaxDD gain;
- joint testing of levers that run together;
- provisional status when the holdout was already seen.

**Not clean, stated first.** My look of 19 Oct already showed these
combinations: book 13.0% / 0.60 / −29.5%; + gilts 25% 11.5% / 0.59 /
−21.5%; + MCX-B 25% 12.6% / 0.63 / −20.7%; both 11.1% / 0.58 / −14.5%. The
weights were fixed before any result (gilts in 5ag on 17 Oct, MCX in 5y on
11 Oct). The choice of rule came after equal risk failed. So any pass here
is provisional by construction, and paper trading is the test that counts.

- **Streams at their own weight,** each a funded sleeve, monthly constant
  mix:
  - gilts 25% (BD1's B3: the 10-year G-sec index less a 0.15% expense
    ratio);
  - MCX-B 25% and MCX-A 25% (the 2007–25 re-run of DG1, with margin and
    idle cash at the liquid rate);
  - the pair, gilts 25% + MCX-B 25% (the book at 50%).
- **Books and figures:** as DG1. The live-like walk-forward is the
  headline; the deployed-rules backtest is an extra condition. Both use
  idle cash at the liquid rate and D2's haircut, over 2008–25.
- **Drawdown route (protocol 5.1),** on both books:
  1. MaxDD 3+ points shallower;
  2. Sharpe no more than 0.02 lower;
  3. MaxDD 3+ points shallower in at least 80% of 2,000 paired
     stationary-bootstrap resamples (63-session blocks, seed 7).
- **Status of a pass:**
  - **Gilts:** already on paper alone (BD3, holdout agreed).
  - **MCX-B and the pair:** provisional. 2026 was seen for both: MCX's on
    12 Oct, the gilts' on 18 Oct.
- **Next for a passing pair:** a paper book with shares 50%, gilts 25% and
  MCX-B 25% (tracker DL1), then 5 dry runs.
- **Code:** `nse_engine/diversifier_gate.py dg2`.

**Result (19 Oct 2026): gilts PASS and MCX-B PASS (provisional), alone;
MCX-A and the pair FAIL.** Results are in `runs_mcx/dg2_results.json`
(copied to `runs_bonds`). Figures are live-like, 2008–25, CAGR / Sharpe /
MaxDD (2008), with the share of bootstrap resamples where MaxDD is 3+
points shallower.

| At its own weight | Walk-forward book: 13.0% / 0.60 / −29.5% (−26.4%) | Backtest book: 15.5% / 0.70 / −28.6% (−23.2%) | Verdict |
|---|---|---|---|
| Gilts 25% | 11.5% / 0.59 / −21.5% (−15.4%); 99% | 13.4% / 0.69 / −21.3% (−12.6%); 100% | Pass |
| MCX-B 25% | 12.6% / 0.63 / −20.7% (−5.8%); 84% | 14.5% / 0.72 / −21.7% (−2.8%); 90% | Pass, provisional |
| MCX-A 25% | 12.4% / 0.56 / −25.2%; 62% | 14.2% / 0.66 / −24.6%; 74% | Fail: Sharpe −0.04, bootstrap below 80% |
| Gilts 25% + MCX-B 25% | 11.1% / 0.58 / −14.5% (+8.1%); 99% | 12.3% / 0.67 / −15.2% (+10.5%); 100% | Fail: Sharpe −0.03 on the backtest book |

- **The pair fails by 0.01 of Sharpe, and only on the in-sample backtest
  book.** On the walk-forward it passes all three checks: MaxDD −29.5% →
  −14.5%, 2008 from −26.4% to +8.1%. By the rule written before the
  result, the pair does not go to paper as a pair. Waiving the extra
  condition is your call.
- **What goes to paper by the rule:**
  - gilts alone (BD3, built);
  - MCX-B alone (tracker DL1: shares 75%, MCX-B 25%), provisional;
  - not both in one book.
- **Your decision (19 Oct): no waiver.** The pair is judged after the paper
  period from the two books' records, on the drawdown route, against the
  books' own shadow:
  - the pair's daily return is 0.5 × the share book + 0.25 × the gilt
    sleeve + 0.25 × the MCX sleeve;
  - the period is at least 60 sessions and 6 monthly MCX decisions;
  - a pass goes to 5 dry runs as the pair.
- **DL1 built (19 Oct):** `config/nse_engine_mcx.json`,
  `kite_connect/trading/mcx_sleeve.py`. The sleeve replays GT5-B's
  simulator nightly from its start, so paper follows the rules that
  passed.
- **Correction, 25 Oct 2026 (LN-T29d):** both books above credit idle cash
  at no cost; with a realistic sweep they read 12.4% / 0.54 / −29.9% and
  14.9% / 0.66 / −29.0% (section 5ah's correction). The verdicts are
  re-checked on cost model 5 (LN-T29).

## 5aj. A1G: A1 with a crash guard (India tracker A1G; pre-registered 20 Oct 2026, 10:42 IST)

**Why.** Under the diversifier gate (DG1, 5ah), A1 was the only real
Sharpe lever: +0.16 at equal risk, p 0.03, holding in 2008–12. It failed
only because the worst drawdown grew by 5 points: it lost 7.8% in the 2008
crash, when bear call spreads were run over by violent rallies. You asked
for a crash guard (20 Oct).

**Not clean, stated first.** The guard is designed knowing that A1 lost in
2008. So 2008–12 is not out of sample for A1G. The only clean check is the
options family's 2026 holdout, still sealed.

- **Configuration, A1G:** O2-A1 (5q: a 2-SD short call and a 3-SD wing on
  NIFTY's monthly options, 15 days out, early exit at the money; cost model
  as 5q's LN-T20 re-run) with one change. On a decision day **no new spread
  is opened** if either holds at that day's close:
  1. **the book's drawdown rule is not normal** (halt, half or risk-off),
     which is the book's own crash state, known live each session;
  2. **India VIX is 22 or higher,** the fear level round 2's X3 took from
     its source (5r), not fitted here. India VIX starts in Mar 2008; before
     that only the first condition applies.

  A spread already open runs to its exit as in A1.
- **Family:** options. N goes from 17 to 18.
- **The guard per book:**
  - the deployed rules' backtest uses its recorded rule states (run
    `20261017T110545845367Z_2cb1c053`, `drawdown.csv`);
  - the walk-forward uses the rule replayed on its equity (20/30/35, half
    0.5, re-arm 60).

  So A1G has one series per book: what it would have done beside that book.
- **Runs:** Kaggle, the options task, 2007–25. The skipped decision dates
  are an input; the configuration's hash names the guard, not the dates.
  The code change is a `skip_entry_dates` input to
  `kite_connect.options.backtest.run_sleeve`, beside its event skips.
- **Gate:** protocol 5.1's Sharpe-lever route, as DG1:
  - overlay, equal risk, capped at 3×;
  - live-like books (idle cash at the liquid rate, D2's haircut) over
    2008–25;
  - the walk-forward book as the headline and the backtest book as an
    extra condition;
  - combined Sharpe higher with p ≤ 0.10, and MaxDD no more than 1 point
    worse, on both books.
- **Expectation, stated before the run:**
  - the guard should remove most of 2008's loss: the rule halted in 2008
    and VIX was above 22 for most of it;
  - it also skips months A1 earned in: 2020's recovery, part of 2022;
  - the Sharpe gain likely shrinks from +0.16. A pass is far from certain.
- **Then, if it passes:**
  - **the 2026 holdout, opened once for the options family:** A1G over 1
    Jan–11 Sep 2026 beside the deployed book's recorded holdout run
    (`20260915T161516314162Z_679cbd0c`), at its last 2008–25 weight;
  - **it agrees** if the combination's 2026 Sharpe is at least the book's
    and its MaxDD is no more than 1 point worse;
  - **then paper** (at least 60 sessions and 6 monthly cycles), then 5 dry
    runs.
- **If it fails:** recorded, not re-tuned. A1 and A1G close.

**Amended 26 Oct 2026, 22:57 IST, before the run** (your calls: GT13 on 25 Oct,
A1C's sizing on 26 Oct; your "do A1G" on 26 Oct):
- **Sizing: A1C's, not equal risk.** Each cycle's weight is min(3, 0.10 ×
  A1G's equity before the fill ÷ the spread's maximum loss), 0 between
  cycles (5am). 15% is reported beside it, not judged.
- **Gate: A1C's, so the two compare.** On both books, the walk-forward the
  headline:
  - combined excess Sharpe higher, one-sided bootstrap p ≤ 0.10 (seed 7);
  - the gain holds in 2008–12.
  - **No MaxDD condition (GT13).** MaxDD, Calmar (full and rolling 3-year),
    CAGR and alpha are reported.
  - 2008–12 is not clean for A1G (stated above).
- **Also reported:**
  - A1C on the same book (A1 unguarded at 10%);
  - the cycles the guard skipped and what A1 made or lost in them;
  - the worst cycle and the largest maximum loss, as shares of the book.
- **Runs:** locally, on the same F&O store as A1's recorded run, if A1
  reproduces on it first; otherwise Kaggle. A1G's hash is `358e73d3`
  (`entry_guard` in its configuration; A1 keeps `ae503fc4`). One run per
  book, so N rises by one.
- **The 2026 holdout was opened for A1 (5al).** A1G's 2026 is reported:
  - beside the recorded holdout run;
  - the book's rule replayed on that run's equity, plus India VIX;
  - it agrees if the combination's Sharpe is at least the book's.
- **If it passes and 2026 agrees:** book `a1` adds the guard from its next
  decision. It has opened no spread yet, so no paper record is lost. It
  is recorded as a pass.
- **If it fails:** recorded, not re-tuned. A1G closes. Book `a1` goes on
  as your 26 Oct decision (5am).

**Result (26 Oct 2026, 23:0x IST): A1G PASSES, and 2026 agrees, though
2026 adds no evidence.** `python -m nse_engine.diversifier_gate a1g`;
results in `dg1/a1g_results.json`. Runs `20261026T173036332821Z_358e73d3`
(beside the backtest book) and `…173041043859Z_358e73d3` (beside the
walk-forward book), in `runs_options`. Options N is now 18.
- **Checked first:** A1 reproduces exactly on today's local F&O store (all
  4,706 sessions; the same 228 cycles), so the runs are local.

| 2008–25, 10% cap | CAGR | Excess Sharpe | MaxDD | 3-year Calmar | ΔSharpe (p) | Holds 2008–12 |
|---|---|---|---|---|---|---|
| Walk-forward book | 13.04% | 0.600 | −29.5% | 0.94 | | |
| + A1G | 13.23% | 0.616 | −29.5% | 0.95 | +0.015 (0.000) | yes |
| + A1C (5am) | 13.24% | 0.614 | −30.1% | 0.96 | +0.014 (0.117) | no |
| Backtest book | 15.48% | 0.697 | −28.6% | 1.09 | | |
| + A1G | 15.66% | 0.710 | −28.6% | 1.10 | +0.013 (0.000) | yes |

- **Why it passes where A1C failed:** A1's 2007–25 had two material
  losses, Jul 2008 (−₹43.6k on its ₹7.5 lakh) and Mar 2009 (−₹22.3k); its
  other five losing cycles lost ₹3–16, charges. The guard skipped both, so
  A1G's overlay has no material loss in sample and the bootstrap finds the
  gain in nearly every resample.
- **What the guard cost:** it skipped 57 decisions (walk-forward book).
  A1 traded 34 of them: 32 won, 2 lost, net +₹2,093. So it gave up about
  as much premium as it saved.
- **Sizing:** 0.30× while a spread is open (24% of days); the largest
  maximum loss is 10.0% of the book. 15% is reported: p 0.000, also holds.
- **2026, reported:** +18.0% / 1.04 → +18.5% / 1.08, the same as A1C. The
  guard never triggered: India VIX stayed under 22 and the rule was
  normal. So 2026 says nothing about the guard.
- **What the pass does not show:**
  - the guard was designed after A1's 2008 loss, and 2008–12 is not
    clean (stated before the run);
  - the significance rests on two cycles the guard was built to avoid;
  - out of sample, a sharp rally from a calm market (VIX under 22, book
    normal) can still cost a spread up to 10% of the book.
- **Then, as pre-registered:** book `a1` runs A1G from its next decision
  (`options_overlay.candidate` A1G). The guard reads its own rule state
  each session (`paper_sessions`) and India VIX from the store. a1 has
  opened no spread, so no paper record changes.

## 5ak. RT1: streams rejected on drawdown, re-judged under GT13 (India tracker RT1; pre-registered 25 Oct 2026, 23:31 IST)

**Why.** The owner's decision of 25 Oct 2026 (GT13): MaxDD is no longer a
target or gate metric. It is measured and reported, never a reason to
reject. Protocol 5.1 is amended to match. The owner asked to re-test the
sleeves that were rejected on drawdown.

**Audit, written before any run.** These verdicts had a MaxDD condition:
- 5x, options O5–O9: failed on the deflated Sharpe (O9 0.00) and on
  Sharpe. Not on drawdown.
- 5y, GT5 MCX trend: failed on CAGR, Sharpe and alpha. Its MaxDD improved.
- 5aa, GT7 and MV1: not run yet.
- 5ab, CR1: failed on beating cash (rules 1 and 2).
- 5af, L7-C: failed on CAGR (≥ 33.9%).
- Stage F E1: failed on CAGR.
- 5ah, DG1:
  - **O2-A1 failed only on MaxDD:** +0.16 Sharpe, p 0.03 (backtest book
    0.044), and its gain holds in 2008–12;
  - O4-Y3 failed on p (0.15) and on MaxDD;
  - O3-X2 failed on p (0.20) and in 2008–12;
  - GT5-A and GT5-B failed on Sharpe at equal risk.

So the one rejection that rested on drawdown is A1. DG1's five streams are
re-judged together, so the set stays complete.

**What runs.** DG1 exactly, the same inputs and settings:
- the five streams: O2-A1, O3-X2 and O4-Y3 as overlays; GT5-A and GT5-B as
  funded sleeves (`mcx_results_2007_25.json`);
- both books: the 2008–25 walk-forward (the headline) and the backtest;
- equal risk, the same base, the same paired bootstrap (seed 7).

The base keeps DG1's idle-cash credit, so the comparison is like for like.
Section 5ah's LN-T29d correction applies to both sides equally.

**Rule: the amended 5.1.** A Sharpe lever only:
- combined excess Sharpe above the book's;
- one-sided p ≤ 0.10;
- on both books;
- A1 and X2, whose 2013–25 was seen, must also hold their gain in 2008–12.

There is no MaxDD condition.

**Reported, never gates:**
- MaxDD;
- Calmar, full window and the median of rolling 3-year windows, for the book
  and the combination;
- CAGR, alpha, the deflated Sharpe at the family's N, and the year-by-year
  split;
- the 19 Oct verdict beside the new one.

**Expected, from DG1's recorded numbers:**
- A1 passes;
- X2 and Y3 fail on significance;
- GT5-A and GT5-B fail on Sharpe.

The run adds the Calmar figures and the formal record.

**Status of a pass: provisional.** The rule changed after DG1's results were
seen, and A1's 2013–25 was seen. So paper is the test:
- a pass goes to an options-overlay paper book, if the owner decides to
  build one;
- that book runs 60 sessions, then 5 dry runs.

A1G (5aj, A1 with a crash guard, not run) is no longer needed for the gate.
It stays available as a possible Calmar improvement: the owner's call.

**If something fails:** recorded, not re-tuned.

**Result (25 Oct 2026): A1 PASSES, provisionally; X2, Y3, GT5-A and GT5-B
fail, as expected.** `python -m nse_engine.diversifier_gate rt1`, results
in `data/nse_engine/dg1/rt1_results.json`. DG1's 19 Oct verdicts reproduce
exactly. Figures are CAGR / Sharpe over 6.5% / MaxDD; Calmar is shown full
and as the rolling 3-year median.

| Walk-forward book | Book | + stream at equal risk | ΔSharpe (p) | Calmar: full, 3-year | Verdict |
|---|---|---|---|---|---|
| O2-A1 call spread | 13.0% / 0.60 / −29.5% | 15.5% / 0.76 / −34.6% | +0.16 (0.026) | 0.44 → 0.45; 0.94 → 1.22 | **PASS, provisional** |
| O3-X2 TRIN | same | 18.6% / 0.78 / −31.9% | +0.18 (0.195) | 0.58; 1.02 | FAIL: p, and not in 2008–12 |
| O4-Y3 overnight (from 2012) | 16.4% / 0.86 / −20.9% | 24.0% / 1.08 / −27.1% | +0.22 (0.153) | 0.78 → 0.89; 0.94 → 1.25 | FAIL: p |
| GT5-A MCX | 13.0% / 0.60 / −29.5% | 11.6% / 0.48 / −29.7% | −0.12 | 0.39; 0.90 | FAIL: Sharpe |
| GT5-B MCX | same | 10.8% / 0.45 / −26.5% | −0.15 | 0.41; 0.99 | FAIL: Sharpe |

- **A1 on the backtest book:** 15.5% / 0.70 / −28.6% → 17.9% / 0.83 /
  −34.2%; ΔSharpe +0.13 (p 0.044); 3-year Calmar 1.09 → 1.35. Its gain
  holds in 2008–12 on both books.
- **What A1 costs:** about 5 points of MaxDD, mostly the 2008 crash (−7.8%).
  The full-window Calmar is flat, and the rolling 3-year Calmar rises above
  1.
- **Read A1's pass with care:**
  - its own deflated Sharpe at the options family's N (17) is 0.14, so
    the stream alone is weak evidence; the gate's test is the book-level
    gain, which is significant;
  - the equal-risk weight averages about 2.7×, near the 3× overlay cap.
    Margin pledged from the book must carry it, which the paper design
    checks;
  - at the tested 0.25× the gain is small: 13.2% / 0.61 / −30.0%.
- **Next (the owner's call):** an options-overlay paper book for A1, which
  runs 60 sessions, then 5 dry runs. Paper is the test, because the rule
  changed after these results were seen.

## 5al. A1P: A1's out-of-sample check and its options-overlay paper book (India tracker A1P; pre-registered 25 Oct 2026, 23:43 IST)

**Why.** The owner's decision of 25 Oct: build an options-overlay paper
book for A1 and promote A1 to paper if the conditions are met; enable A1G
too if it does not hold A1 up.

**The conditions are the owner's five steps:**
1. **Backtest 2008–25:** done. RT1 (5ak) passed A1 as a Sharpe lever under
   GT13, provisionally.
2. **Walk-forward:** A1 fits no parameter beyond its pre-registered rule,
   and the book's walk-forward is the gate's headline.
3. **Out of sample:** the options family's 2026 holdout, still sealed.
   This section's check.
4. **60 paper sessions.**
5. **5 dry runs.**

**The check: one look at the options family's 2026 holdout.**
- **A1, unchanged:** O2-A1 as recorded in run
  `20261010T122956596793Z_ae503fc4`. It is simulated by `run_sleeve` on the
  F&O store from 1 Jan 2026, its volatility look-back from the index.
- **Beside it:** the deployed book's recorded holdout run
  (`20260915T161516314162Z_679cbd0c`, 1 Jan–11 Sep 2026). That window is
  the one 5aj fixed for A1G.
- **As an overlay at 3×** (book + 3 × A1). That is the equal-risk weight
  at its last reset in RT1, and the cap, where it sat in 81–86% of months.
- **Agrees if** the combination's excess Sharpe over 6.5% in the window is
  at least the book's.
- **Reported:** the return, CAGR, MaxDD and Calmar of both, and A1's 2026
  cycles (traded or skipped, strikes, P&L).
- **The lock:** written before anything runs, so the look counts even if
  the run fails.

**If it agrees: A1 is promoted to paper.** Book `a1`:
- the deployed rules at Rs 35 lakh, plus A1 as an overlay at 3× the book's
  NAV;
- the overlay's margin is pledged from the book; the email reports the
  margin and the spread's maximum loss as a share of NAV;
- each night A1's own simulator replays from the overlay's paper start to
  the latest F&O session, so paper follows the rules that passed (as the
  MCX sleeve does);
- the workflow keeps the F&O files the replay needs;
- the email shows the cycle (decision, strikes, lots at 3×, maximum loss),
  the overlay's P&L and the combination;
- paper runs at least 60 sessions, then 5 dry runs. The options order path
  for dry runs is built at that point.

**If it does not agree:** recorded; A1 is not promoted.

**A1G** (5aj) runs separately on its own pre-registration, adjusted by
GT13: no MaxDD condition. It does not hold A1 up. Its holdout check then
reads a holdout that has been opened, so a pass is provisional and goes to
its own paper book.

**Not clean, stated:**
- the 3× weight is RT1's last reset, not re-estimated in 2026;
- the deployed book's 2026 was opened on 15 Sep for its own family, but
  A1's 2026 option P&L has not been looked at.

**Result (25 Oct 2026, 23:5x IST): A1 AGREES; promoted to paper.**
`python -m kite_connect.options.backtest holdout`. The lock is
`runs_options/holdout.lock`, so the options family's 2026 holdout is now
opened. The results are in `runs_options/a1p_holdout.json`.
- **First, a check on a seen window:** on 1 Jan–11 Sep 2025, against the
  deployed backtest with a throwaway lock, all 8 of A1's cycles matched its
  recorded run, in action and strikes.

| 1 Jan–11 Sep 2026 | Return | Excess Sharpe | MaxDD | Calmar |
|---|---|---|---|---|
| Deployed book (recorded holdout run) | +18.0% | 1.04 | −13.0% | 2.11 |
| Book + 3 × A1 | +22.7% | 1.35 | −12.3% | 2.84 |
| A1 alone, on its own capital | +1.3% | (below rf on its own) | −0.1% | |

- **A1's 2026 cycles:** 6 traded, all profitable; February and July
  skipped (no eligible strike). The spreads were 700–850 points wide, 4–5
  lots on A1's own Rs 7.5 lakh.
- **Not the March crash risk:** 2026's fall was a decline, and a bear call
  spread loses in sharp rallies, as in 2008, not in falls. Paper tests that
  risk going forward.

**The paper book, built 25 Oct 2026.**
- `config/nse_engine_a1.json` (book `a1`): the deployed rules at Rs 35
  lakh plus A1 at 3×; paper start Mon 26 Oct.
- `kite_connect/trading/options_overlay.py`, with the runner, emails,
  weekly block and deployment checks.
- The workflow keeps F&O files from 45 days before the paper start and
  builds the F&O store each night.
- **Checked on the local store** (1 Sep–23 Oct 2026, night by night):
  - the cycles equal a one-shot replay;
  - the overlay earns 3× A1 (+1.63% of the book against A1's +0.55%);
  - a re-run of a night appends nothing; no alerts;
  - the email and the weekly block render;
  - the code's A1 equals the recorded configuration (hash `ae503fc4`).

**The tail at 3×, measured on A1's 2007–25 cycles.** This is the decision
before dry runs:
- 95% of 138 traded cycles won, averaging +0.22% of the book;
- the worst were −17.2% (Jul 2008) and −9.2% (Mar 2009);
- every cycle carried a maximum loss of about 100% of the book (median
  102%): a rally through the wing before expiry loses it.

So equal-risk sizing on A1's low realised volatility takes the overlay to
its cap and hides a ruin-sized tail. The margin a spread needs is about its
maximum loss, likely more than the holdings can pledge. Paper runs at 3×
because that is what passed. Live needs a sizing rule set before the dry
runs, for example a cap on maximum loss per cycle. A cap of 10–15% of NAV
is about 0.3–0.45×; at 0.25× the gain was small (Sharpe 0.60 → 0.61).

## 5am. A1C: A1 with each month's maximum loss capped at 10% of the book (India tracker A1C; pre-registered 26 Oct 2026, 00:07 IST)

**Why.** The owner's decision of 26 Oct: take A1 live, but cap each month's
maximum loss at 10–15% of the book, so a wrong month cannot wipe it out.
His words: earning consistent profits, even small ones, is the best way
forward, not betting big and getting wiped out. At 3× every A1 cycle
carried a maximum loss of about 100% of the book (5al).

**Configuration A1C.** A1's rules are unchanged: its strikes, timing,
exits, and its own sizing on its own capital. Only the overlay's size
changes:
- for each cycle, the weight is min(3, 0.10 × A1's equity at the fill ÷
  the spread's maximum loss on A1's capital);
- so the spread can lose at most 10% of the book's NAV, and the 3× cap
  still holds;
- between cycles the overlay holds nothing.

10% is the owner's range at its safer end. 15% is reported beside it, not
judged. Family: options; N rises by one.

**The test: the owner's five steps.**
1. **Backtest 2008–25:** protocol 5.1's Sharpe lever (GT13) on both books,
   at A1C's weight series instead of equal risk:
   - one-sided bootstrap p ≤ 0.10, the same bootstrap (seed 7);
   - the walk-forward book is the headline;
   - the gain must hold in 2008–12, because A1's 2013–25 was seen.

   Reported: MaxDD, Calmar (full and rolling 3-year), CAGR, alpha, the
   weight while a spread is open, and the worst cycle and largest maximum
   loss as shares of the book.
2. **Walk-forward:** the book's, as in RT1.
3. **Out of sample:** the options family's 2026 holdout was opened for A1
   (5al). A1C's 2026 is reported from it, so a pass is provisional.
4. **Paper:** book `a1` runs A1C's sizing from its first session (26 Oct),
   for 60 sessions.
5. **5 dry runs,** with the options order path built before them.

**Expected, written before the run.** The weight while a spread is open
should be about 0.3×, since A1's maximum loss is about 30–35% of its own
capital. At a fixed 0.25×, RT1 had Sharpe 0.60 → 0.61, so the gain is
likely small and may not be significant (p > 0.10).

**If it fails significance:** recorded as a fail. The owner then decides
whether a small, bounded, positive-expectancy overlay goes on to paper and
dry runs. That decision is recorded as his, not as a pass.

**Result (26 Oct 2026, 00:3x IST): FAIL on significance.**
`python -m nse_engine.diversifier_gate a1c --cap 0.10`; results in
`dg1/a1c_results.json`.

| 2008–25 | CAGR | Excess Sharpe | MaxDD | Calmar (full / 3-year median) | ΔSharpe (p) |
|---|---|---|---|---|---|
| Walk-forward book | 13.0% | 0.600 | −29.5% | 0.44 / 0.94 | |
| + A1C, 10% cap | 13.2% | 0.614 | −30.1% | 0.44 / 0.96 | +0.014 (0.117) |
| + A1, 15% cap (reported) | 13.3% | 0.620 | −30.4% | 0.44 / 0.97 | +0.020 (0.134) |
| Backtest book | 15.5% | 0.697 | −28.6% | 0.54 / 1.09 | |
| + A1C, 10% cap | 15.7% | 0.709 | −29.2% | 0.54 / 1.09 | +0.012 (0.128) |

- **Fails both conditions:** p > 0.10 on both books, and the gain does not
  hold in 2008–12 (2008 −26.4% → −27.0%, 2009 +18.3% → +17.2%).
- **The tail is gone:** the weight while a spread is open is 0.30× (open
  32% of days); the worst cycle cost 1.6% of the book (2.5% at 15%); the
  largest maximum loss is 10.0% of the book by construction.
- **So is most of the gain:** about +0.2 points of CAGR, against +2.5 at 3×.
- **2026, reported (opened in 5al):** +18.0% / 1.04 → +18.5% / 1.08; the
  weight while open is 0.32×.

**The owner's decision (26 Oct 2026):** take A1 on with the cap, for
small, consistent profits rather than a ruin-sized tail. It is recorded as
his decision, not as a pass. Book `a1` runs A1C's sizing from its first
session: `options_overlay.max_loss_cap` 0.10 in `config/nse_engine_a1.json`.
Paper then runs 60 sessions, then 5 dry runs, and only then live.
- **Checked locally** (1 Sep–23 Oct 2026, night by night): October's open
  spread is 7.8 book lots at 0.34× (max loss 10.0% of NAV), against 70.8
  lots (89%) uncapped.

## 5an. AC1: the deflated and probabilistic Sharpe widened for autocorrelation (India tracker AC1; your decision 27 Oct 2026, written 21:20 IST before any number was recomputed)

**Why.** HT1 (27 Oct) measured the 61 book configurations: lag-1
autocorrelation of daily returns +0.14 (median), so the Newey-West t is
0.86 of the i.i.d. one (range 0.83–0.93). The deflated Sharpe, the
probabilistic Sharpe and the minimum track record all count every day as
an independent observation (Bailey & López de Prado; Mertens' variance with
skew and kurtosis, no serial correlation). So each is too confident. Your
call: make the promotion gate stricter and more honest.

**The change.** The per-period Sharpe's estimator variance is multiplied by
κ, the long-run variance over the variance of the daily excess returns:
- Newey-West with Bartlett weights and floor(4 (T/100)^(2/9)) lags, as HT1
  and the alpha t use;
- κ is floored at 1, so the gate never loosens. Negative autocorrelation
  is left unrewarded.

Equivalently, T becomes T / κ in the PSR's sqrt(T − 1), and in the
minimum track record. The cross-trial variance of Sharpe ratios behind
SR₀ is measured across the trials, so it is unchanged. The annualised
Sharpe itself is not rescaled (Lo's 2002 annualisation is not adopted
here).

**Where it applies:**
- every deflated Sharpe: gate 1 in every family, `validate`, the
  scorecards, the books register;
- the scorecard's probabilistic Sharpe and minimum track record.

The i.i.d. figure stays reported beside it as `dsr_iid`.

**Books already promoted or on paper keep their status.** Their adjusted
figures are recomputed and reported. Gate 1 at 0.95 uses the adjusted
figure from now on, for promotions and new configurations. Whether a book
that falls below 0.95 stays deployed is your decision, not automatic.

**Result (27 Oct 2026, 21:4x IST): all three book configurations fall
just below 0.95.** Registry window 2013–25, cost model 4, N = 61:

| Configuration | κ | Lag-1 | T_eff of 3,215 | DSR i.i.d. → AC1 | MinTRL vs 0, years |
|---|---|---|---|---|---|
| deployed `679cbd0c` | 1.35 | +0.15 | 2,381 | 0.964 → **0.939** | 2.1 → 2.9 |
| candidate `2d64ba4c` | 1.30 | +0.14 | 2,486 | 0.963 → **0.941** | 2.2 → 2.8 |
| E4 `93cf6c4d` | 1.37 | +0.14 | 2,356 | 0.957 → **0.928** | 2.3 → 3.1 |

- **Clustered N:** the DSR stays 1.000 for all three. The bar binds only at
  the raw count of 61 near-identical parameter variants.
- **Walk-forward 2008–25:** the PSR against 0 is 0.994 → 0.986 (κ 1.31).
  The out-of-sample edge stays clear.
- **Checked by simulation:**
  - on AR(1) 0.3 returns with no edge, a PSR ≥ 0.95 came in 12.2% of the
    time i.i.d. and 5.5% widened (nominal 5%);
  - on independent returns, κ's median is 1.000 and its 95th percentile
    1.09, so the floor costs at most about 4% of the z;
  - with the adjustment switched off, the old figure comes back exactly.
- **What it changes:**
  - the candidate and E4 cannot be promoted on gate 1 as they stand;
  - every new configuration faces the widened bar;
  - the deployed book stays deployed until you decide;
  - paper, dry runs and the live gates do not read the DSR.
- **The books register keeps the approval-time (i.i.d.) DSR** until each
  run is validated again; scorecards show both.

**Your decision (27 Oct 2026, after the result):**
- **The deployed book stays as is,** approved under the old rule.
- **Its acceptance as a real-time strategy** rests on how close its paper
  and then live performance comes to the actual Indian market, not on
  gate 1 again. That means:
  - the paper gate G4 (paper against its same-period backtest on NSE's
    prices: tracking error, daily gap, costs, drawdown, regime breaks);
  - HT1's test against the backtest's edge;
  - NIFTY 50 reported beside it.
- **For the candidate and E4,** gate 1 stands at the widened figure.

## 5ao. RA1: accept or reject the deployed book as a real-time strategy (India tracker RA1; your decision 27 Oct 2026, pre-registered 21:34 IST, before any verdict was possible)

**Why.** You kept the deployed book as approved under the old gate (5an).
It is to be accepted or rejected as a real-time strategy on how close its
performance comes to the actual Indian market. The rule is fixed now,
while the paper book has 26 of the 60 sessions it needs.

**The evidence, both measured on NSE's own prices each night:**
- **G4, the paper gate** (`nse_engine.paper_gate`): the paper book against
  its same-period backtest. Its thresholds are frozen for this verdict as
  they stand on 27 Oct 2026:
  - tracking error: PASS ≤ 8% a year, FAIL > 12%;
  - daily gap: PASS ≥ −3 bp a day, FAIL below that with t ≤ −2;
  - costs: PASS ≤ 1.5× the model, FAIL > 2×, from 20 fills;
  - drawdown: PASS ≤ the larger of 1.5× the backtest's and its +2 points,
    FAIL > the larger of 2× and +4 points;
  - regime break: none in the last 20 sessions.
- **HT1, short of the edge** (`significance.vs_expected`): is the paper mean
  daily excess return below the deployed configuration's 2013–25 backtest
  Sharpe (1.16, the register's) beyond chance? One-sided p, Newey-West.
  - The walk-forward's 0.60 is reported beside it.
  - At 60 sessions this flags only a collapse (a realised Sharpe below
    about −2.5). G4 does the real work.
- **NIFTY 50:** reported beside, never a condition. Weeks of returns cannot
  tell an edge from luck.

**The verdict**, on the first night G4 has 60 sessions (about 14 Dec 2026):

| G4 | Short of the edge (p < 0.05) | Verdict |
|---|---|---|
| FAIL | either | **REJECT** |
| either | yes | **REJECT** |
| PASS | no | **ACCEPT** |
| WATCH | no | **WATCH**: decided once more at 90 sessions (about 25 Jan 2027): ACCEPT only if G4 then passes and it is not short; otherwise REJECT |

- **No G4 report, or no HT1 p** (a tooling fault): no verdict until it is
  fixed. A G4 FAIL still rejects on its own. The verdict is never taken
  from incomplete evidence. (Clarified at 21:4x, before any verdict was
  possible.)
- **ACCEPT:** live may start at the first rung (₹6 lakh) after its 5 clean
  dry runs, on your go-ahead (the existing go-live path).
- **REJECT:** no live capital. The book stays on paper; `679cbd0c` is not
  re-tuned; you decide what follows.
- **WATCH:** paper continues; live waits.
- **Once live:** the same rule, on the live book's own G4 at 60 live
  sessions, decides whether live continues (ACCEPT) or stops and returns to
  paper (REJECT).
- **The verdict is computed automatically** (`paper_gate.realtime_verdict`)
  and shown in the nightly and Saturday emails from now on, so it cannot
  be re-read later.

## 5ap. IVX1: stock selection from option prices (India tracker IVX1; your decision 27 Oct 2026, pre-registered 22:00 IST, before any data was read)

**Why.** IVS1 (27 Oct) found that 46% of the deployed book's trades (47% of
traded value) are in stocks with listed options. Two published effects
say option prices predict a stock's next-month return. If either holds on
NSE, it could filter about half the book's picks.

**The signals.** Each is taken at the last session of each month, from that
session's closes:
- **S1, the call–put IV spread** (Cremers & Weinbaum, 2010): the IV of the
  at-the-money call minus the put at the same strike and expiry. Published
  direction: a higher spread earns more.
- **S2, the put skew** (Xing, Zhang & Zhao, 2010): the IV of the put
  nearest 0.95 × spot minus the at-the-money call IV, same expiry.
  Published direction: a steeper skew earns less.

**How the IVs are taken:**
- `theory.implied_volatility` (Black-Scholes) with `OptionsConfig`'s rate
  (5.52%) and no dividend;
- the near-month expiry, the next one when fewer than 7 days remain;
- the strike nearest spot with both its call and put traded that day
  (contracts > 0) for S1;
- for S2, a put that traded, within 0.93–0.97 of spot.

A stock without these contracts that month is left out of that signal.

**Universe and window.** The F&O stocks in the store (`stock_options`,
about 199 a year), January 2013 to December 2025: about 155 month-ends.

**Portfolios.**
- Quintiles each month, equal weight.
- Bought at the next session's open, held to the open after the next
  month-end; returns on the engine's dividend-adjusted prices.
- Long–short: S1 is Q5 − Q1 (highest spread minus lowest); S2 is Q1 − Q5
  (flattest skew minus steepest).
- **Costs:** 25 bp a side on each month's turnover in both legs. That is
  above the 16–18 bp the paper books pay a fill (G4), to allow for impact.

**Gate (protocol 5: a published effect's replication), per signal:**
- the monthly long–short return net of costs, in the published direction,
  with a Newey-West t (3 lags) of at least **2.24**. That is t ≥ 2,
  corrected for testing two signals (Bonferroni, two-sided p ≤ 0.025);
- the same sign in both halves, 2013–19 and 2020–25.

**Reported, not judged:** the gross and net means, Sharpe, hit rate, the
five quintiles' means (monotone or not), turnover, alpha against NIFTY 50
(Newey-West), and how many stocks each month had a signal.

**Family:** new, "option-implied stock signals", N = 2. Its own registry
(`data/nse_engine/runs_ivx1/`) and results file (`ivx1_results.json`).

**2026 holdout, for a passing signal only:** opened once, January to
September 2026 (9 months). It agrees if the long–short mean is above zero.
The sample is tiny, so it is reported more than relied on.

**Then:**
- **A pass:** a stage 2, pre-registered separately before it runs: the
  signal as a filter on the deployed rules' buys (skip the worst quintile),
  on the walk-forward book, under the diversifier gate's Sharpe route
  (GT13).
- **A fail:** recorded and not re-tuned. No other quantiles, horizons,
  wings or windows; the family closes.

**Not clean, stated:**
- the papers used US data from 1996–2005, and the effects have been public
  since 2010, so they may have faded;
- Indian stock options trade thinly away from the money, so S2 will be
  noisy and cover fewer stocks;
- dividends bias S2 (different strikes) more than S1 (same strike);
- the universe is the large caps, about half the book's names.

**Runs:** local, about 199 stocks × 155 month-ends of IV solves.
Prompt: do IVX1.

**Result (27 Oct 2026, 22:4x IST): both FAIL; the family closes.**
Run with `nse_engine/option_signals.py`, removed on 28 Oct 2026 once the
family closed (in git history at f6a16c7, to reproduce it). Results in
`ivx1/ivx1_results.json`, runs in `runs_ivx1/`. 25,998 stock-months;
155 months.
- **Kept to the window:** equity prices were read only to 31 Dec 2025, so
  the last period (formed in November 2025) ends at that day's close, not
  at the first open of 2026.

| Signal | Stocks a month | Gross a month (t) | Cost a month | Net a month (t) | Halves, net | Quintiles Q1→Q5, gross |
|---|---|---|---|---|---|---|
| S1 IV spread (`3a608a55`) | 171 | +0.36% (+1.48) | 0.73% | −0.36% (−1.47) | −0.84% / +0.21% | 0.96, 1.23, 1.37, 1.32, 1.33% |
| S2 put skew (`dd3542f6`) | 159 | +0.30% (+1.41) | 0.74% | −0.44% (**−2.10**) | −0.72% / −0.12% | 1.29, 1.47, 1.43, 1.13, 0.99% |

- **Direction:** both point the published way before costs, but neither
  is significant even gross (t 1.48 and 1.41, against 2.24).
- **Costs:** turnover is about 2.9 a month (both legs nearly replaced
  every month), so 25 bp a side costs about 0.73% a month, which is more
  than either edge.
- **Monotonicity:** neither is monotone across quintiles. S1's lift sits
  in Q1 alone.
- **Alpha against NIFTY 50:** S1 −5.2% a year (t −1.68); S2 −6.1% a year
  (t −2.36).
- **Not re-tuned, as pre-registered:** no other quantiles, horizons or
  costs. The 2026 holdout stays sealed for this family.

## 5aq. DT1: the day-trader-skill modules as swing diagnostics (India tracker DT1; your ask, pre-registered 30 Oct 2026, 21:50 IST)

**Source.** github.com/Jakeschincariol/day-trader-skill (MIT): eleven US
day-trading skills. Read in full, ideas only, no code taken. Its scanner,
news, entry, exit, time machine and alerts modules are tested here in the
only form Centurion may use them: delivery, daily bars, no same-day exit
(section 2: rules needing intraday prices or a same-day exit are out of
scope, so its opening-range entry and flat-by-close cannot be adapted as
written).

**Status.** An event study: no change to any book, no registry entry, N = 8
(S1 to S5, N1, E1, X1, T1 is reported beside them and A1 is a build decision,
neither a return claim). A pass earns one pre-registered trial through the
normal gates, never adoption. Anything that fails leaves no code.

**Hypothesis.** S1 to S5: chart setups capture continuation (breakout, flag,
reclaim) or overreaction (oversold), and a scan adds to the book's selection.
N1: buying just before results takes on event risk that is not paid. E1: a
completed close that confirms the signal filters false starts. X1: taking
profit at a fixed multiple of risk beats the book's rank exit. T1: a stock's
own recent bars, resampled, give honest odds of stop against target.

**Data and window.** The deployed configuration's data (`load_market_data`
with its own settings, adjusted prices, `data_start` 2012-01-02 replaced by
2007-01-01 for the reported years), its point-in-time universe mask, and its
recorded trades (run `20261017T110547431678Z_679cbd0c`, 2007-25). Gates are
read on 2013-25; 2008-12 is reported. 2026 is not opened.

**Common rules.** A decision uses closes up to day t only; the fill is the
open of t+1. The horizon is 20 sessions, open to open (A: the source has no
swing horizon; 20 sessions is the book's median hold). Excess = the return
minus the equal-weight universe's over the same window. Cost = one round trip
at `statutory_cost` on Rs 25,000 a side plus 3 bp of spread each way (A).
Sleeve ETFs are left out. A fresh entry is a BUY from a flat position.
Daily-event series use Newey-West with lag 20; tests on the book's entries
average each decision date first and use lag 4 (A: rebalances are 5 sessions
apart).

**S1 to S5, trader-scanner.** Over the universe, on adjusted closes, exactly
`find_setups`: S1 breakout_20d, close above each of the previous 20 closes;
S2 bull_flag, close[t-5]/close[t-15]-1 > 10% and the last 5 closes inside 5%
of the close; S3 reclaim_ma20, close[t-1] under its 20-day mean and close[t]
above its own; S4 oversold, Wilder RSI(14) < 30; S5 oversold_bounce, RSI(14)
< 30 at t-1 and close[t] > close[t-1]. **Pass:** mean net excess, t >= 3.
Also reported, no gate: the book's fresh entries with the flag against those
without.

**N1, trader-news.** Book's fresh entries with a results board meeting
(purpose naming results) in the 5 sessions after the fill and already
intimated at the decision (`available_from` <= t), against all others. **Pass
(a hold-back earns a trial):** near minus away < 0, |t| >= 3. Also reported:
the universe's mean excess over [-1, +1] sessions around results.

**E1, trader-entry.** The swing form of "a completed bar confirms": buy at the
open of t+2 only if close[t+1] > close[t], else skip the entry. Paired on the
book's fresh entries, excess to the same end date, a skip scoring 0. **Pass:**
rule minus actual > 0, t >= 3. Also reported: the cost of one day's delay.

**X1, trader-exit.** The source's default target, +2R (A: R = 6 x ATR(20) at
the decision, the book's stop distance), on the book's round trips: sell at
the target on the first session whose high reaches it before the recorded
exit; else the recorded exit. Paired per round trip. **Pass:** target minus
actual > 0, t >= 3.

**T1, trader-timemachine.** For each fresh entry: pool = the stock's previous
60 sessions' (high, low, close) / previous close (A: a day replaces a 5-minute
bar, so its time-of-day slots fall away); 10,000 replays, seed 19, of 20
sessions; stop = entry - 6 x ATR(20), target = entry + 2R, stop checked first
in a bar, as the source. Compared with what the stock then did over the same
20 sessions. **Pass (a per-trade line earns a build):** Brier skill above the
base rate, mean predicted within 5 points of realised, and the forward excess
rising with the source's lean (target share minus break-even), t >= 3. The
source's second half, the same rule on the stock's own past, is not tested: 12
entries a stock over 19 years is too few (A).

**A1, trader-alerts.** No return claim. The book's stop exits: the trigger is
the highest close since entry minus 6 x ATR(20) as `trailing_stop`, and the
live GTT's limit is 2% below it. Reported: the share of stop days whose open
is under that limit (the live order rests unfilled while the backtest sells
at the open), and for those the return from that open to the next session's
open. **Build a market-hours stop check only if** the share is at least 10%
and waiting loses at least 1% on average.

**Result (30 Oct 2026): every test failed its rule. Nothing is adopted and no
code stays; the study's script was temporary.** Net of 34.6 bp a round trip.

| Test | 2013-25 | 2008-12 | Rule | Verdict |
|---|---|---|---|---|
| S1 breakout_20d | −0.47%, t −3.19 | −0.66%, t −1.27 | t ≥ 3 | FAIL |
| S2 bull_flag | −0.03%, t −0.16 | +0.04%, t +0.13 | t ≥ 3 | FAIL |
| S3 reclaim_ma20 | −0.49%, t −6.58 | −0.45%, t −1.78 | t ≥ 3 | FAIL |
| S4 oversold | −1.43%, t −2.77 | −2.40%, t −3.07 | t ≥ 3 | FAIL |
| S5 oversold_bounce | −0.73%, t −2.39 | −1.18%, t −1.99 | t ≥ 3 | FAIL |
| N1 hold back before results | near minus away −0.99%, t −2.00 | no data before 2012 | < 0, \|t\| ≥ 3 | FAIL |
| E1 confirming close | rule minus actual −0.66%, t −2.47 | | > 0, t ≥ 3 | FAIL |
| X1 +2R target | target minus actual −0.83%, t −1.35 | | > 0, t ≥ 3 | FAIL |

- **Scanner.** 2013-25 events: 118,989, 47,995, 54,344, 40,380 and 19,994. Of the
  book's 1,822 fresh entries, 522 were 20-day breakouts (−0.36% against the
  rest), 269 flags (+0.82%), 39 reclaims (−0.73%) and none oversold.
- **N1.** 155 entries near results, 1,667 away. With the fill day counted as
  near: −0.62%, t −0.86, so the rule is also fragile. The universe's own excess
  over [−1, +1] around 16,635 results is −0.26%, t −3.12: no earnings premium
  on NSE, and too small to trade around.
- **E1.** One day's delay costs the confirmed trades 1.65%, t −14.88. Confirmed
  entries made +3.17% and the skipped −0.16%, but most of that gap is the day
  the rule waits out.
- **X1.** 170 of 1,822 round trips reach +2R. The book's mean round trip is
  +3.79%, its median −1.23%, and the best tenth holds 64% of gross wins: a
  target cuts the trades that pay for the rest.
- **T1 FAIL.** Stop first: predicted 1.4%, real 5.2%, Brier skill −0.030. Target
  first: predicted 3.2%, real 2.0%, Brier skill −0.024. Both are within 5 points
  but worse than the base rate. The lean's slope is +0.023, t +2.94, under the
  bar and by quartile (−0.18% to +2.16%) only the momentum the book already
  buys. Resampling days one at a time loses volatility clustering, so it
  understates stop-outs almost fourfold. `nse_engine.mc_bands` already uses a
  block bootstrap for this reason.
- **A1: not built.** 20 of 239 stop days (8.4%) opened under the GTT's limit,
  under the 10% bar; 6 of those never traded back up to it that day. From that
  open to the next open: +2.91% over all 20, −1.73% over the 6, so waiting for
  the evening session costs nothing on average. The backtest's fill at the open
  and the live resting limit differ on about one stop in 40, and not in a way
  that flatters the backtest.

A first run measured past returns by a sign error in the script (the book's own
entries showed −11%). It was caught before any verdict, fixed with known-answer
checks, and no rule changed. The study's ATR was also replaced by the engine's
`atr_panel` before anything was read.

## 5ar. WV1: Wyckoff, volume profile and order flow on NSE (India tracker WV1; your ask, pre-registered 30 Oct 2026, 22:17 IST)

**Why now.** None of the three is in the deployed engine. Wyckoff is not in the
repo at all, though the legacy `liquidity_sweep.py` is close to a spring. Volume
profile and an order-flow proxy exist only as legacy single-stock strategies, and
the order-flow forecast sits at weight 0 in the old combiner. OF1 (28 Oct) tested
one claim: whether price trends outside the value area. That is not rerun. This
study asks a different question: whether any of these predicts a stock's return,
or would change what this book does.

**Evidence read before writing this.** Wyckoff: no rigorous return study was
found; practitioners call it hard to backtest and say phases are easier to label
in hindsight, so every rule below uses only data up to the decision. Volume
profile: no direct study; the closest rigorous relative is Grinblatt & Han (2005),
a turnover-weighted cost basis. Order flow: Chordia & Subrahmanyam (2004) find
lagged imbalances predict daily returns, with signed trades. Bulk volume
classification estimates that from bars, but is less accurate than the tick rule
(Chakrabarty, Pascual & Shkilko, 2015), so a failure of O1 fails that proxy, not
order flow itself. Gervais, Kaniel & Mingelgrin (2001): high volume earns a
premium over the next month. Bulk deals: Indian studies find large returns around
them, mostly before publication.

**Family and N.** Event studies; N = 17 here. No change to any book. A pass earns
a stage 2, pre-registered separately before it runs, never adoption. Anything
that fails leaves no code; the legacy strategies stay, as strategy code does.

**Data and window.** The deployed configuration's adjusted data loaded from
2006-01-02, its point-in-time universe mask, and run
`20261017T110547431678Z_679cbd0c` for the book's trades and weights. Gates are read
on 2013-25; 2008-12 is reported; 2026 stays sealed.

**Not testable.** Footprint, absorption and true delta need each trade's aggressor
side, and Centurion has no tick data. Bulk and block deals: only the day's file
can be read; NSE's history endpoint refuses automated clients and that is not
worked around, so they stay untested until the history is obtained properly.

**Shared terms (t = the decision day; nothing after t is read).** ATR is the
engine's `atr_panel`(20). `vr` = volume[t] / mean volume of the previous 20 sessions.
`range` = high[t] - low[t]. Windows accept up to 10% missing days (halts).

*Trading range (A), 60 sessions to t-1:* support S = lowest low, resistance R =
highest high, efficiency = |net close move| / sum of |daily close moves|. A base
has efficiency <= 0.30. `prior` = the 120-session return ending where the base
starts. Accumulation = base and prior < 0. Distribution = base and prior > 0.

*Events* count once per stock in any 20 sessions, for a universe member that
traded on t. Fewer than 200 events in 2013-25 is reported as too few to judge.

**Wyckoff (StockCharts' tutorial for the definitions; thresholds are (A)).**
- W1 spring: accumulation, low[t] < S, close[t] >= S, vr <= 1. Bullish.
- W2 sign of strength: accumulation, close[t] > R, vr >= 1.5, range >= 1.5 ATR.
  Bullish.
- W3 upthrust: distribution, high[t] > R, close[t] <= R. Bearish.
- W4 sign of weakness: distribution, close[t] < S, vr >= 1.5, range >= 1.5 ATR.
  Bearish.
- W5 absorption (effort without result): 20-session return < 0, vr >= 2, no new
  20-session low, range <= ATR. Bullish.
- W6 supply (effort without result): 20-session return > 0, vr >= 2, no new
  20-session high, range <= ATR. Bearish.
- W7 the buying tests, at least 6 of these 8 (test 1, the point-and-figure count,
  cannot be made mechanical): T2 a climax (vr-like volume >= 3 x the 60-session
  mean, a fall of 2 sd, close in the upper half of the bar) in the last 120
  sessions, then a retest within 3% of its low on <= 0.7 of its volume; T3 mean
  volume on up days above down days, 20 sessions; T4 close above its 50-session
  mean; T5 and T6 the last 20 sessions' low and high above the 20 before; T7 the
  60-session return above the universe's median; T8 a base; T9 distance to the
  250-session high >= 3 x distance to S. Bullish.

**Volume profile.** 60 sessions to t, 50 bins from the lowest low to the highest
high, each day's volume spread evenly across the bins its range touches; the point
of control (POC) is the fullest bin; the value area grows from it, one bin at a
time, toward the fuller neighbour until it holds 70% (the OF1 settings).
- V1: close crosses above the value area's top. Bullish.
- V2: close crosses below its bottom. Bearish.
- V3: the legacy `VolumeProfileStrategy` buy, as coded and at its defaults (the
  profile to t-1, close within 0.3% above the value-area low, or close reclaiming
  the POC). Bullish.
- V4 (factor): close / POC - 1, high minus low.
- V5 (factor, published): capital-gains overhang, (P - RP) / P with RP[t] =
  (1 - v[t]) RP[t-1] + v[t] P[t], v = shares traded / shares outstanding. Share
  counts are the last filed, known by `available_from`; before filings start in
  2012, the first one filed (A, burn-in only). High minus low.

**Order flow (daily proxies).**
- O1 (factor, published): bulk volume classification, sum over 20 sessions of
  volume x (2 Phi(r / sd) - 1) / volume, sd over 60 sessions. High minus low.
- O2 (factor): mean of close / VWAP - 1 over 20 sessions, VWAP = traded value /
  shares. High minus low.
- O3 (factor, NSE only): sum over 20 sessions of the sign of the day's move x
  delivered shares / delivered shares. High minus low.
- O4 (factor): the legacy `compute_order_flow_forecast` as coded (OBV, MFI, close-in-
  range delta, delivery). High minus low.
- O5 (factor, published): high-volume premium, weekly form: this week's volume
  above all 9 prior weeks (high) against below all 9 (low). High minus low.

**Tests.**
- *Events:* fill at the open of t+1, 20 sessions, open to open, excess over the
  equal-weight universe, Newey-West lag 20 on the daily mean. Bullish: net of 34.6
  bp a round trip, mean > 0. Bearish: gross, mean < 0 (a veto pays no cost).
- *Factors:* month-end quintiles, equal weight, bought at the next open and held
  to the open after the next month-end, 25 bp a side on both legs' turnover,
  Newey-West lag 3 on the monthly high-minus-low.

**Pass.** |t| >= 3 in the claimed direction for a new idea; t >= 2.39 for a
published effect (V5, O1, O5: Bonferroni for three, two-sided p <= 0.0167); the same
sign in 2013-19 and 2020-25.

**The book's route, also a pass at |t| >= 3 in the claimed direction.** A long-only
book cannot short a signal, so what counts is what it would change. Entries: fresh
entries with a bullish flag in the 5 sessions to the decision against the rest, or
the factor's top against bottom quintile that day (a veto of the bottom). Holdings:
every 5th session, names held with a bearish flag in those 5 sessions against other
names held, gross 20-session excess. Same dating, same 20 sessions.

**Stage 2, for any pass.** Written and timestamped before it runs: the signal as a
veto on new buys, or as an early exit for a held name, in a real engine run of the
deployed rules on the honest base, under the diversifier gate's Sharpe route.

**Addendum (30 Oct 2026, 22:26 IST, written after stage 1's numbers were seen).** Two
additions, both before they run.

*W1b to W4b and W7b, five new hypotheses (N 17 -> 22), not a re-tune.* Stage 1
showed the base rule above did not select bases: 90.3% of member days passed it,
because efficiency <= 0.30 over 60 sessions is close to always true. W1 to W4 and
W7 were therefore tests of false breaks in general, not of Wyckoff's trading
range, and stay recorded as such. The variants differ only in the base: efficiency
<= 0.30 **and** (R - S) / ATR in the tightest fifth of that day's universe. Same
events, tests and bars, and T8 reads the tight base in W7b. These five are not
varied further, whatever they show.

*V2's stage 2.* V2 passed stage 1. Rule V2X: from the day of a V2 event, for 20
sessions, the stock's forecast is -1 in the engine, which stops new buys and exits
a holding at the next open through the engine's own `exit_reason`. Run:
`dep.reference_config()`, 2008-25, against the same config without the rule, on this
machine. **Pass:** excess Sharpe above the base with `diversifier_gate.bootstrap_p`
<= 0.10, and above it in 2008-12 too. A fail on this first book ends it; a pass goes
to the walk-forward book before anything else. Reported: CAGR, MaxDD, turnover, the
number of forced exits.

**Result (30 Oct 2026): nothing adds value. All 22 are discarded and no code
stays.** One test passed stage 1 and failed in the engine. All on 2013-25; the
checks reproduced DT1's book exactly (1,822 entries, +1.04%).

| Event | Claim | Events | 20-session excess | t | Verdict |
|---|---|---|---|---|---|
| W1 spring | up | 1,812 | −1.04% | −3.49 | FAIL |
| W2 sign of strength | up | 2,447 | −0.77% | −2.98 | FAIL |
| W3 upthrust | down | 7,764 | +0.08% | +0.40 | FAIL |
| W4 sign of weakness | down | 1,847 | −0.12% | −0.43 | FAIL |
| W5 absorption | up | 3,784 | −0.15% | −0.76 | FAIL |
| W6 supply | down | 2,988 | −0.00% | −0.02 | FAIL |
| W7 buying tests | up | 19,965 | −0.23% | −1.80 | FAIL |
| W1b spring, tight base | up | 413 | −0.55% | −1.43 | FAIL |
| W2b sign of strength, tight base | up | 830 | −1.33% | −4.08 | FAIL |
| W3b upthrust, tight base | down | 2,829 | −0.25% | −0.91 | FAIL |
| W4b sign of weakness, tight base | down | 460 | −0.76% | −1.51 | FAIL |
| W7b buying tests, tight base | up | 7,561 | −0.31% | −1.86 | FAIL |
| V1 above value | up | 16,165 | −0.37% | −2.40 | FAIL |
| V2 below value | down | 13,986 | −0.48% | −3.20 | stage 1 pass |
| V3 legacy buy | up | 22,611 | −0.31% | −3.94 | FAIL |

Bullish rows are net of 34.6 bp, bearish rows gross.

| Factor | Gross a month | Net a month | t | Bar | Verdict |
|---|---|---|---|---|---|
| V4 distance from POC | +0.47% | −0.18% | −0.52 | 3 | FAIL |
| V5 capital-gains overhang | +1.27% | +1.08% | +2.14 | 2.39 | FAIL |
| O1 bulk volume | +0.05% | −0.74% | −2.67 | 2.39 | FAIL |
| O2 close against VWAP | −0.00% | −0.63% | −1.68 | 3 | FAIL |
| O3 signed delivery | +0.04% | −0.74% | −3.41 | 3 | FAIL |
| O4 legacy forecast | −0.16% | −0.95% | −4.03 | 3 | FAIL |
| O5 high volume | −0.06% | −0.95% | −3.55 | 2.39 | FAIL |

- **The book's route: no pass.** The nearest is against its claim: 377 entries just
  above value did 1.92% worse than the rest (t −2.54). It stays a note, not a rule.
- **V2X in the engine: FAIL.** Sharpe 0.704 to 0.713, p 0.410 against 0.10, and
  2008-12 fell from −0.009 to −0.060. 2013-25 0.934 to 0.960; CAGR 15.59% to 15.68%;
  MaxDD −28.3% to −27.3%. It forced 215 exits. The book is rarely exposed: 4 of
  1,822 entries were flagged, and flagged holdings beat other holdings by 1.70%
  (t +1.78), the wrong way. Corrected 31 Oct: the first run used the default store,
  which has no NIFTY before 2012, so the regime gate had no trend vote for 2008-11
  (Sharpe 0.605 to 0.617, p 0.373, 2008-12 −0.408 to −0.451). Same verdict on
  `store_ext2006`; a run that starts before its regime index now refuses.
- **V5 is the nearest miss,** and it would add nothing: the book already holds it,
  with 1,009 of 1,822 entries in the top quintile and 7 in the bottom. 2008-12 was
  −0.26%.
- **The gross spreads of O1 to O5 are all within 0.16% of zero.** The proxies' net
  losses are their turnover, not a reverse signal. O4 supports its weight of 0.
- **Not a verdict on order flow itself.** These are daily proxies, and bulk volume
  is known to classify poorly. Tick-level order flow and bulk deals stay untested.
- **Implementation.** 36 temporary tests passed before any data was read, including
  an exact match with the legacy profile. The no-look-ahead test first missed planted
  leaks; it was rebuilt until all 11 were caught.

## 5as. BLK1: what bulk deals leave for a buyer the morning after (India tracker BLK1; your ask, pre-registered 31 Oct 2026, 10:06 IST)

**Why now.** WV1 left bulk deals untested because NSE's history endpoint refuses
automated clients. You exported the history by hand, 2006-25, so it can be tested.
The question is narrow: a bulk deal is public that evening, so what is still there
at the next open?

**Hypothesis.** A client must disclose a trade above 0.5% of a company's shares, so
each row is a large flow. Large investors split their orders (Chan & Lakonishok,
1995), so a one-sided buyer may not have finished, and a fund that takes that much
in a day has done the work. A one-sided seller may hold the same information the
other way, or have more to sell. Against: the deal is public by the open, and WV1's
reading was that Indian studies find the returns mostly before publication. No study
of what is left after publication was found to replicate, so this is a new signal.

**Family and N.** Event studies; N = 3. No change to any book. A pass earns a stage
2, pre-registered separately, never adoption. Anything that fails leaves no code.
Temporary code, run on the Mac (A), plain arithmetic with no engine decisions.

**Data and window.** `data/nse_engine/bulk_deals/`, 20 files, 211,188 rows, 2006-01-02
to 2025-12-31, no 2026 row, gitignored; client names stay out of the docs. Prices:
the deployed configuration's adjusted data on `store_ext2006`, loaded from
2006-01-02, its point-in-time universe mask, and run
`20261017T110547431678Z_679cbd0c` for the book's trades and weights. Gates are read
on 2013-25; 2008-12 is reported; 2026 stays sealed.

**Read before writing this (structure only, no return or later price).** 55% of
client-stock-days have the same client on both sides, mostly high-frequency desks.
12% of rows fall in the book's universe. Events below, once per stock in 20 sessions,
2013-25: any buyer 970, fund buyer 109, repeat buyer 61, any seller 1,138, fund seller
58. So the minimum is 100 events (5ar's 200 would rule out the fund buyer), and repeat
buyers and fund sellers are reported only, because a pass needs a move too large to be
worth counting as a trial.

**Rules (t = the deal date; nothing after t decides an event).**
- Clean: drop exact duplicates; drop every row with a remark other than "-" or blank
  (reported late, added or modified, so their publication date is unknown), 160 rows
  in all; symbols through the engine's rename table; sum each client's shares per
  stock, day and side.
- One-sided buyer: a client with a BUY and no SELL in that stock that day. One-sided
  seller: the reverse. A client on both sides is churn, not an event.
- Fund (A): the client's name matches `MUTUAL FUND|MUTUAL FD|\bMF\b|LIFE INSURANCE|
  GENERAL INSURANCE|INSURANCE CO|PENSION|PROVIDENT FUND|TRUSTEE CO`. Trustee companies
  of mutual funds count; debenture trustees (`TRUSTEESHIP`) do not, being mostly pledge
  transfers. Foreign names are not funds here: a name cannot split investors from
  arbitrage desks. 4% of these client-days are both-sided, against 55% overall.
- Publication (A): SEBI's 2004 rule has the broker report on the day and the exchange
  publish after the close, so the first tradable moment is the open of t+1. The
  exports carry no publication time, so this cannot be checked from them.

**Hypotheses.** A stock with both a buyer and a seller on t counts in both.
- B1 any one-sided buyer. Bullish.
- B2 a fund as one-sided buyer. Bullish.
- B3 any one-sided seller. Bearish.

**Tests, as 5ar.** Events: a universe member that traded on t, once per stock in any
20 sessions. Fill at the open of t+1, 20 sessions, open to open, excess over the
equal-weight universe, Newey-West lag 20 on the daily mean. Bullish: net of 34.6 bp a
round trip, mean > 0. Bearish: gross, mean < 0 (a veto pays no cost).

**Pass.** |t| >= 3 in the claimed direction, the same sign in 2013-19 and 2020-25, and
at least 100 events.

**The book's route, also a pass at |t| >= 3 in the claimed direction.** Entries: fresh
entries with a bullish flag in the 5 sessions to the decision against the rest, net.
Holdings: every 5th session, names held with a bearish flag in those 5 sessions
against other names held, gross. Same dating, same 20 sessions.

**Reported, no pass can come from these.**
- Where the move went, for B1 to B3: the 20 sessions to t-1, day t, and the close of t
  to the open of t+1.
- Repeat buyer: the same client's second one-sided buy in a stock within 10 sessions.
- Fund seller.
- Placebo: stock-days with only both-sided clients, which are volume without a
  directional holder.
- 2008-12, the 5-session horizon, and a fill at the open of t+2.

**Stage 2, for any pass.** Written and timestamped before it runs: a real engine run
of the deployed rules on `store_ext2006`, 2008-25, under the diversifier gate's Sharpe
route.

**Result (31 Oct 2026): all three FAIL. Nothing is left for a buyer the morning after,
so there is no stage 2 and no code.**

| Hypothesis | Claim | Events | Net or gross 20-session excess | t | 2013-19 | 2020-25 | Verdict |
|---|---|---|---|---|---|---|---|
| B1 any buyer | up | 965 | −1.59% | −3.22 | −1.75% | −1.34% | FAIL, wrong way |
| B2 fund buyer | up | 108 | −0.60% | −0.65 | −1.95% | +0.38% | FAIL |
| B3 any seller | down | 1,129 | −0.20% | −0.36 | −0.30% | −0.05% | FAIL |

Bullish rows are net of 34.6 bp, the bearish row gross.

| Where the move went | 20 sessions before | Day t | Overnight | 20 sessions after, gross |
|---|---|---|---|---|
| B1 any buyer | −0.06% | +1.86% | +0.54% | −1.06% |
| B2 fund buyer | +0.90% | +2.12% | +0.57% | −0.13% |
| B3 any seller | −0.93% | −1.24% | +0.11% | −0.18% |

- **The move is over by the open.** A fund buyer's stock is 3.6% ahead of the universe
  by the next open and flat after it. This agrees with WV1's reading of the Indian
  studies.
- **B1 is significant the wrong way, and that is not an edge.** The pass rule asks for
  the claimed direction, so it fails. Turning it into a veto would be a new hypothesis,
  and the evidence is against one: days with only both-sided clients do as badly
  (−1.73%, t −2.74), so it is a volume-spike effect rather than the buyer. The book's
  own 16 flagged entries beat its others by 8.17% (t +1.98).
- **The book's route: no pass, and little exposure.** Flagged entries: 16 for B1, 1 for
  B2, 13 for B3 of 1,823. Flagged holdings against others: +0.15% (t +0.10), and +0.09%
  (t +0.08) for sellers.
- **Reported, no pass possible.** Repeat buyer −1.30% (61 events), fund seller +1.97%
  (56), with the two halves of each opposite in sign. 2008-12: B1 −1.29% (t −1.70), B2
  +0.35% (t +0.28), B3 −0.47% (t −0.74). A fill at t+2: B1 −1.96% (t −4.22), B2 +0.61%,
  B3 −0.57%. Five sessions: B1 −0.84% (t −3.21), B2 −0.31%, B3 +0.09%.
- **Implementation.** 20 temporary tests passed before any return was read. Day t's
  move has the right sign on both sides, so the dates line up. The sanity line read
  1,823 book entries at +1.04% against DT1's 1,822 at +1.04%. The CSVs stay where they
  are.

## 5at. GT2: the rebuild after the golden target review (India tracker GT2; your ask, pre-registered 31 Oct 2026, 14:53 IST)

**Why now.** The audit and outside research agree that Sharpe 1.4 and 35% are not a sustained
outcome for a positional Indian book. Live-like 2008-25 is 13.0% / 0.60. Gold alone earned 13.8%
and 60 NIFTY / 40 gold 12.4%, so the book's edge is about +0.12 to +0.17 Sharpe, and metals earned
4.7 of its 13.3 points. You kept the golden target as the aspiration (31 Oct). This plan works
towards it through a first milestone: Sharpe about 0.8, 17-18% CAGR, about 15% vol.

**Rules for everything below.**
- The hurdle is a passive NIFTY and gold mix as well as the book. Alpha is reported against both.
- Live-like metrics only; D2 is already inside them and is not charged twice.
- Every new instrument comes from a rule stated before any result, never from a list of survivors.
- 2026 stays out of every test here.
- Failed ideas leave no code. Guardrails, strategies and backtest analysis scripts stay.

**Phase A: diagnostics, one look each, nothing recorded, no parameter chosen.**
- A1 Alpha by rank bucket against NIFTY plus gold, by period, to settle why IC and alpha disagree.
- A2 The book with metals replaced by the liquid rate: return, Sharpe, crash drawdown.
- A3 Each exit by removal, and sold names' excess over 63 sessions as well as 21.
- A4 DSR at N 61, 1,000 and 7,000; PBO on the metric actually used to select.
- A5 (added 31 Oct, before any run) Range-market losses by cause, including the cost of false signals.
- If stock-only alpha has t < 1 over 2013-25, this comes to you before Phase B: the honest product
  may be the passive mix with a defensive overlay.
- **Phase A result (31 Oct 2026): stop test passed.** Stock-only alpha 2013-25 +6.0% a year, t 2.03
  (2008-25 +3.6%, t 1.38). Detail in tracker GT2A; nothing recorded, no parameter chosen.

**Phase B: the core family's one remaining configuration.** A slower core: equal-weight lookbacks
centred on 63 days or longer, with hold bands. Values are fixed in a dated addendum after Phase A
and before any run, on Kaggle. Pass: walk-forward 2008-25 live-like Sharpe above the base with
`bootstrap_p` <= 0.10, above the base in 2008-12. Changed 31 Oct, before any run: the turnover
condition is dropped, because Phase A showed turnover alone does not pay.

**Phase B addendum (31 Oct 2026, 19:37 IST, after Phase A, before any GT2B run).** One change set, one
walk-forward, the core's one configuration left before January:
- `portfolio.exit_rank` 40 to 60. Why: sold names keep beating the universe over 63 sessions, and
  ranks 21-40 earn as much as the top 20.
- `signals.fast_ewmac` from (8,32), (16,64) to (16,64), (32,128); `slow_ewmac` stays (64,256). Why:
  the forecast's IC rises with horizon, 0.051 at 21 days and 0.079 at 63. All three spans stay equally
  weighted inside their groups, so nothing is picked among them.
- Nothing else moves. The rebalance stays weekly, because a monthly one gained nothing; the ATR stop
  stays in the grid; the regime exit stays. Phase A's other in-sample leads are not adopted.
- Same as the base `wf_stitched_r4b`: its 32-point grid, 18 folds, anchored from 2007, one-year first
  training window, 12-month tests, `store_ext2006`, `excess_sharpe` selection, Kaggle.
  The base is `engine_679cbd0c` with `universe.price_filter_unadjusted` true: it reproduces all 18 of the
  base's recorded fold hashes.
- Pass, on the live-like book against the base over 2008-25: higher Sharpe, `diversifier_gate.bootstrap_p`
  <= 0.10, and higher Sharpe in 2008-12. Reported, not gates: 2013-25, CAGR, MaxDD, Calmar, trades, and
  the passive NIFTY and gold hurdle. A failure is not re-tuned and leaves the deployed book unchanged.

**Phase B result (31 Oct 2026): FAIL, not adopted.** Live-like 2008-25, base to GT2B: 13.04% / 0.600 /
-29.5% to 13.19% / 0.593 / -31.2%; `bootstrap_p` 0.55; 2008-12 Sharpe -0.24 to -0.32. All three conditions
fail. 2020-25 improved (1.13 to 1.40) and 2013-19 fell (0.64 to 0.45). Trades a year 340 to 264. Not re-tuned.

**Phase C: new streams, each its own family, at most 3 configurations, the §5.1 gate.**
- C1 Reshaped 31 Oct, before any run: not another rank group (R6 and GT4 already failed that). A
  look at dilution with 30-40 names, then a quality veto and a post-crash rebound rule, at most one
  core configuration each. Gap: filings start in 2012.
- C2 Idle capital deployed: exposure 60-100% with hysteresis, no borrowing. One rule.
- C3 withdrawn 31 Oct: REITs and InvITs stay closed under GT14 (every blend cut CAGR 2-4 points).
- C4 withdrawn 31 Oct: single-stock futures shorts stay closed under U28 until the book passes Rs 2 crore.

**GT2C addendum (31 Oct 2026, 22:55 IST, before any run).** Idle capital, one rule, N = 1, its own family. No
borrowing and no change to the core: an overlay on the recorded walk-forward, plain arithmetic on the Mac (A).
- Target exposure each close: risk_on 100%, neutral 80%, risk_off nothing added, so the regime exit is untouched.
- The gap between that target and what the book holds goes into NIFTYBEES instead of the liquid rate: NIFTY
  total return less 0.05% a year. No hindsight: an index, not a picked asset.
- Hysteresis: the overlay moves only when its target differs by 5% of capital or more, or when risk_off takes
  it to zero. It never takes the book above 100%.
- Clarified 22:57 IST, before any result: the overlay is also zero whenever the drawdown rule is not normal
  (`dg.guard_dates` replay, the rule A1G uses), so it never buys back a cut the rule made.
- Timing: decided on the close, filled at the next open, so it is credited from the session after that. The
  same two-session lag applies when it comes out. Only the 100% cap is immediate: a stock buy is funded by
  selling NIFTYBEES at the same open.
- Costs: NIFTYBEES at dated statutory charges plus 5 bp a side; D2's haircut on what it holds.
- The gate, clarified before any run: §5.1 is for diversifiers, and this is not one. Gate 2 applies: on the
  live-like book 2008-25, CAGR, Sharpe and alpha against NIFTY TRI all higher, and Sharpe in 2008-12 not lower.
  Reported: `bootstrap_p`, MaxDD, Calmar, 2013-25, NIFTY's 30 worst weeks. If only CAGR rises, it is not a
  pass; it comes to you as a risk choice.

**GT2C result (31 Oct 2026): FAIL, not adopted, no code stays.** Live-like 2008-25, base to GT2C: 13.04% /
0.600 / alpha 5.70% to 12.63% / 0.525 / alpha 5.16%. CAGR, Sharpe and alpha all fall, so gate 2 fails; 2008-12
Sharpe -0.242 to -0.241 holds. Reported: `bootstrap_p` 0.93; 2013-25 16.87% / 0.88 to 16.37% / 0.77; MaxDD -29.5%
to -28.2%, but -20.9% to -24.9% in 2013-25; Calmar 0.44 to 0.45; NIFTY's 30 worst weeks -0.99% to -1.38% on
average. The overlay held 10% on average, a third of the 30% idle: most idle money sits in risk_off, where the
rule rightly adds nothing. Cost is not the cause: 0.34 points a year on 5.6x turnover. NIFTY did not beat the
liquid rate in the states it was held, as GT2A's causal test warned. Regime state rebuilt from the engine on
store_ext2006 and checked: all 22 regime-exit days follow a risk_off close. The +2 to +3 point estimate was wrong.
Results in `data/nse_engine/gt2c/results.json`.

**GT2E addendum (31 Oct 2026, 23:05 IST, before any run).** Reshaped: only the metals cap can be tested.
- Gilts: Bonds is closed at N = 3 of 3. MCX: closed, its third slot kept for a new source. Both stay on paper; a
  rule choosing between them would reopen closed families.
- Metals by rule: gold and silver are the only commodity classes among NSE ETFs, so a rule returns the same two.
- The cap, N = 1: metals at most 35% of capital, sized before any result: 10% bounded tail / gold's -27.3%
  worst drawdown = 37%, rounded down; GT2A's 26-32% mean weights stay untouched, so it cuts peaks only.
- Overlay arithmetic on the recorded walk-forward, the Mac (A). Metals above 35% go to the liquid rate, pro rata
  gold and silver, trimmed when over by 2 points or more, decided on the close, same two-session lag as GT2C.
  Costs: dated statutory charges plus 5 bp a side on the trims; trades the engine would then skip are not
  credited. D2's haircut is released on what is trimmed.
- The gate, stated before any run: a cap is a risk limit and is not expected to raise CAGR, so gate 2 does not fit.
  Pass: Sharpe not lower on the live-like book 2008-25 and in 2008-12. Reported: CAGR, alpha, MaxDD, Calmar,
  2013-25, days capped. A pass with lower CAGR comes to you as a risk choice. Building it would be an engine
  configuration, so it waits for January's budget; a fail leaves no code.

**GT2E result (31 Oct 2026): FAIL, not adopted, no code stays.** Live-like 2008-25, base to the 35% cap: 13.04% /
0.600 / alpha 5.70% to 12.58% / 0.581 / alpha 5.24%; 2008-12 Sharpe -0.242 to -0.381. Both conditions fail.
Reported: `bootstrap_p` 0.75; 2013-25 Sharpe 0.879 to 0.882, CAGR 16.87% to 16.64%; MaxDD -29.5% to -27.4%; Calmar
0.44 to 0.46. Capped on 40% of days, mean metals 29% to 24%. The damage is in 2009 (18.3% to 12.8%) and 2011 (7.4%
to 4.6%), when the book had little else working. Metals check: 4.65 points a year against the audit's 4.7. The
cap does not remove the hindsight risk; the book keeps it knowingly, about a third of its return from metals.
Gilts and MCX stay on paper under their own books. Results in `data/nse_engine/gt2e/results.json`.

**GT2D look (31 Oct 2026, 23:19 IST, before any run).** C1's first step, a look: nothing recorded, no parameter
chosen, no trial spent. 18 fixed-settings books on store_ext2006, the deployed rules with the drawdown rule: 20,
30 and 40 names with `exit_rank` at twice the names, each trend-only and with one of low_vol, delivery,
residual_momentum, near_high or low_beta added at a third. Read live-like, 2008-25, 2008-12 and 2013-25.
How it is read: dilution has gone only if a group at 30 or 40 names beats trend-only at the same names on Sharpe
over 2008-25 and is not lower in 2008-12. Even then, the result is only a candidate for a January configuration,
walk-forward and gated: these are in-sample settings. If no group does that, no signal group goes to January.

**GT2D look result (31 Oct 2026): dilution has not gone; no signal group goes to January.** Live-like 2008-25,
Sharpe, trend-only then each group added, low_vol / delivery / residual_momentum / near_high / low_beta. 20 names:
0.697, then 0.426 / 0.745 / 0.685 / 0.717 / 0.548. 30 names: 0.750, then 0.461 / 0.775 / 0.658 / 0.657 / 0.569. 40
names: 0.768, then 0.499 / 0.750 / 0.658 / 0.685 / 0.576. The 20-name trend book reproduces the recorded backtest.
Of ten comparisons at 30 and 40 names one clears the rule: delivery at 30, Sharpe +0.025, 2008-12 +0.007, CAGR
16.01% to 15.03%. That is the delivery tilt closed on 17 Sep at +0.06, not a new source, so it takes no January
configuration; that is a judgment beyond the rule, recorded as such. Delivery data covers 88-98% of names each year
from 2006. Width on trend alone: 20 / 30 / 40 names 15.48% / 0.697, 16.01% / 0.750, 16.32% / 0.768, but exit rank
widened with it, and GT2A's 20 names at exit rank 60 read 17.4% / 0.82 against 16.01% / 0.750 at 30 names on the
same exit rank. So the gain is the exit rank, not the width, and exit rank 60 did not survive walk-forward in GT2B.
Results in `data/nse_engine/gt2d/results.json`.

**Phase D: live safety, code and checks.**
- A hard limiter at 10 orders a second.
- Zerodha to confirm `market_protection` on manual MARKET orders, the daily logout, and whether
  paper trading on live Kite data is allowed.
- A gap-fill model that includes band-locked days, and re-placed limits clamped inside the band.
- A days-to-exit cap beside the 8% name cap, and a -13% no-exit stress (23 Mar 2020).
- A documented manual exit through a second broker.

**Phase E: going live.** The 60 sessions and 5 dry runs stay, as operational readiness only: 60
sessions has about 8% power to tell Sharpe 1.0 from 0.5. Added: canary capital at 5-10% for 3-6
months, then 25%, 50% and full, advancing on slippage, volatility ratio, turnover and
reconciliation, never on profit. A CUSUM alarm scales capital down. RA1's rule stands as fixed.

**Phase E result (31 Oct 2026): built, GT2I.** Before building, the look: one-year books 2013-25 at Rs 1.5, 3, 6 and
30 lakh, reset each January because a small book compounds out of the problem. Rs 1.5 lakh holds 12 names, 74%
invested, and misses the Rs 30 lakh book by -10 to +8 points a year: a different book. Rs 3 lakh holds 17, 85%
invested, mean gap +0.5; Rs 6 lakh 18, 87%, +0.4; Rs 30 lakh 18, 87%. So the canary is 10%, not 5%.
- Rungs Rs 3 / 6 / 12 / 21 / 30 lakh: the canary added under U6's ladder, not 25% and 50%, which would drop the
  rungs already decided. First session refused at any other rung; `CENTURION_GO_LIVE_OVERRIDE` still overrides.
- The canary holds at least 60 sessions, later rungs 20. About six months to full, against three.
- GO now also needs G4's costs measured and PASS, and live over backtest volatility and rupees traded inside 0.7
  to 1.3. Profit is not read. Reconciliation runs and alerts every session and does not block GO.
- CUSUM on the daily gap, k 0.25, h 12 sd, sessions at this rung: fixed before the first live session by simulation
  on t(4) days, 0.6% false alarms in 60 sessions, 4.6% in 250, a -0.5 sd gap caught in a median of 38 sessions. An
  alarm steps down one rung after 20 sessions at it; at the canary it alerts. It sits outside G4's five checks, so
  RA1 stands as fixed.
- Changed behaviour: manual F&O with no capital set now reads Rs 3 lakh, not Rs 6 lakh.

**Not doing.** Structural leverage (until live Sharpe is near 0.9 and futures drag is under 2
points, then at most 1.25-1.5x on index futures); permanent puts; naked short volatility; a fourth
overlapping exit; intraday; earnings drift, bulk deals, index-inclusion longs and size alone.

**Evidence.** `docs/Centurion golden target feasibility.md`.

## 5au. VP2: the value-area rotation plan, adapted to swing (India tracker VP2; your ask, pre-registered 31 Oct 2026, 20:51 IST)

**Why now.** A volume profile video, for intraday index futures, teaches a plan OF1 and WV1 did not test.
WV1 tested crossings of the value area, POC distance and the legacy buy, with no trend filter, stop or
target. The new claim: in an uptrend, buy a dip under the value-area low that reclaims it, stop at that
low, target POC then the value-area high.

**Hypothesis.** Sellers who broke the low are trapped when it reclaims, and price rotates back through
value. Against: Centurion's value-area tests so far failed, and the idea comes from intraday auctions.

**Family and N.** Event studies; N = 4. No change to any book. A pass earns a separate stage 2. Anything
that fails leaves no code; the legacy strategies stay. Temporary code, on the Mac (A).

**Adaptation (A).** The video's profile is the previous New York session from minute data. Centurion has
daily bars and trades swing, so the profile is the 60 sessions to t-1, 50 bins, 70% value area (OF1's and
WV1's settings), and every level is known before t. Shorts are out: cash trades long only. Data, universe
mask, run and window as 5as: 2013-25 gates, 2008-12 reported, 2026 sealed.

**Rules (t = decision day; nothing after t decides an event).**
- Uptrend (the video's "higher highs, higher low"): the highest high and the lowest low of the 60 sessions
  to t-1 are both above those of the 60 sessions before.
- P1 VAL reclaim: uptrend, low[t] < VAL, close[t] > VAL.
- P2 resistance becomes support: uptrend, close crossed above VAH in the 10 sessions to t-1, low[t] <= VAH,
  close[t] > VAH.
- P3 point of control holds: uptrend, close[t-1] > POC, low[t] <= POC, close[t] > POC. POC stands in for the
  high volume node.
- P4 the full plan on P1 events: fill at the open of t+1, stop at low[t], half out at POC, the rest at VAH,
  at most 20 sessions, then out at the open. A day touching stop and target counts as the stop.

**Tests, as 5ar.** P1-P3: a universe member that traded on t, once per stock in any 20 sessions, fill at
the open of t+1, 20 sessions, excess over the equal-weight universe, Newey-West lag 20, net of 34.6 bp. P4:
each trade's net return less the universe over the same days, by entry date, same t.

**Pass.** t >= 3, the same sign in 2013-19 and 2020-25, at least 200 events. The book's route as 5ar, at t >= 3.

**Reported, no pass can come from these.** The same events without the trend filter; a fill one session
later after a close above high[t], the video's "wait for a reaction"; 5 sessions; 2008-12.

**Not testable.** The video's own form needs minute data, which the store lacks; low volume node speed and
order-flow "trapped traders" are not observable on daily bars.

**Result (31 Oct 2026): all four FAIL. Nothing is adopted and no code stays.**

| Test | Events | Net excess | t | 2013-19 | 2020-25 | Verdict |
|---|---|---|---|---|---|---|
| P1 VAL reclaim | 6,978 | -0.55% | -3.05 | -0.40% | -0.73% | FAIL, wrong way |
| P2 resistance becomes support | 6,664 | -0.36% | -1.84 | -0.21% | -0.52% | FAIL |
| P3 POC holds | 7,826 | -0.00% | -0.03 | +0.28% | -0.33% | FAIL |
| P4 the full plan | 6,720 | -0.78% | -9.16 | -0.81% | -0.75% | FAIL, wrong way |

- **The plan loses on its own terms.** P4 stopped out on 71% of trades and hit its final target on 21%, with a
  30% hit rate and win/loss of 1.76: net -0.48% a trade. 2008-12 was -1.22% (t -5.04).
- **The trend filter helps a little, not enough.** Without it P1 is -0.73% (t -5.85) and P2 -0.52% (t -3.09).
- **Waiting for a reaction does not help.** P1 -0.58% (t -2.05), P2 -0.43%, P3 -0.43%.
- **The book's route: no pass.** Flagged entries against the rest: P1 -1.36% (t -0.57, 18 entries), P2 -1.66%
  (t -2.01, 269), P3 +0.21% (t +0.14, 108).
- **Implementation.** 11 temporary tests passed before any return was read. The sanity line read 1,823 entries at
  +1.04%, as DT1. This does not judge the video's intraday form, which needs minute data.

## 5av. INT1: the intraday programme (India tracker INT1; your amendment U37, plan written 31 Oct 2026, 21:09 IST)

**Purpose.** Find out whether any intraday stream adds to CAGR and Sharpe beside the book, net of real
costs, and stop cheaply if none does. This section is the programme. Each idea still gets its own dated
pre-registration before any result is read.

**Starts when.** GT2B has its verdict and GT2H (live safety) is done. The rest of GT2 does not block it.
Your call first: the data vendor. Budget: Rs 50,000 all in, approved by you 1 Nov 2026.

**Rules for every step.**
- Its own family, `data/nse_engine/runs_intraday`, at most 3 configurations a source; failures stay
  recorded and leave no code. 2026 stays sealed, opened once for the family.
- The first scope is NIFTY and BANKNIFTY futures: they have no dead names, so no survivorship question.
  Single stocks join only if the data keeps delisted names.
- Flat by 15:15 IST every session, nothing held overnight by this stream, and no size above the cap.
- `order_service.intraday_refusal` stays on until INT1E, then opens by a per-stream allow-list only.
- Exception granted by you (31 Oct): minute data will not reach 2008, so the gap is stated and the crash
  test uses the worst days the data does hold, such as 23 Mar 2020 and 4 Jun 2024.

**INT1A Data, about 1-2 weeks.** One vendor shared with OX2 if it fits. A `minute_store` with every session
checked: minute bars must reproduce the bhavcopy's daily high, low, volume and close within a stated
tolerance, and a session that fails is refused, never filled. Contract rolls and expiry days explicit.
Exit: at least 5 years, all checks clean. If not, stop and report.

**INT1B Simulator, about 1-2 weeks.** Fills at the next bar, never the signal bar; spread and slippage on
real quote data; each instrument's own dated costs in `fno_costs`; the 15:15 square-off; circuit halts;
planted look-ahead traps that must be caught. Exit: known-answer tests pass and a no-skill random strategy
loses exactly its costs.

**INT1C Research round, about 2 weeks, one look at each.** At most 3 ideas, each pre-registered:
the video's session value-area rotation on NIFTY futures (from the transcript), and suggestions, mine, to
be confirmed then: the opening gap's fade or follow, and expiry-day behaviour after the 7 Dec 2026 VWAP
settlement change. Evidence must come first, and any idea needs a reason someone else loses to it.
Pass, on data to 2025: t >= 3 net; the same sign in each half; net Sharpe holds at 2x slippage; DSR >= 0.95
at the family's N; the §5.1 diversifier gate beside the book, `bootstrap_p` <= 0.10.
If nothing passes, the family closes, the code goes, and the store stays only if OX2 uses it.

**INT1D Runner and paper, about 60 sessions.** A session-long process with its own market-hours risk loop,
on the Oracle VM or a small paid host, kept apart from the nightly book. GitHub Actions cannot do this: 6.25
hours a session is more than the minutes budget allows. Paper fills on live quotes. It measures slippage,
latency, missed fills and a real flat-by-close every day. It cannot prove edge; it proves the machine.
Exit: 60 sessions, measured slippage within the backtest's assumption, zero overnight positions, no breach.

**INT1E Live, staged.** Starts at no more than 5% of book capital, on a separate budget, with a daily loss
stop of 1% of the book (suggestion, stricter than the 10-15% overlay cap, since this runs every day), a stream
kill switch and the 10 orders a second limiter from GT2H. It scales on measured fills, never on profit, as
GT2I. CUSUM alarm cuts it. SEBI's retail algo rules are confirmed with Zerodha first.

## 5aw. S151: "151 Trading Strategies" (Kakushadze and Serur, 2018; your ask, looks declared 01 Nov 2026, 10:38 IST, before any run)

**What the book is.** A catalogue. By its own preface it holds no backtests or empirical results, so there is no
published table to replicate: replication here means reproducing its formulas in tests. Appendix A is intraday,
dollar-neutral R code under a no-warranty disclaimer; nothing is copied from it.

**Looks, not trials.** Fixed settings, the deployed rules with the drawdown rule, 20 names, live-like, 2008-25.
Nothing recorded, no parameter chosen, no change to any book. Each group is added at a third.
- L1 Residual momentum as the book writes it: (3.13)-(3.17), market factor only, no intercept in the residuals,
  regression ending before the skip month. Against Centurion's form, which subtracts the intercept.
- L2a, L2b Sector momentum (4.1), a sector's equal-weight return over 126 and 252 sessions.
- L3 Within-sector momentum (3.38)-(3.44): 12-1 return less its sector's mean.
- L4 The sector cap on 2008-25, a measurement: SB2 took it out of history, so no recorded backtest has it and live
  does. Bias stated: L2, L3 and L4 use today's map, so names that later dropped out carry no sector.
- L5 IBS (4.10) on NIFTYBEES and GOLDBEES, buying below 0.2, next open to the following open, net of costs.

**How they are read.** L1-L3 reach January's queue only if Sharpe beats the trend book by 0.05 or more over 2008-25,
is not lower in 2008-12, and is higher in both 2013-19 and 2020-25, and also at twice the costs. L5 only at t >= 3
net. L4 is a measurement with no pass. Anything that clears is a candidate for a dated walk-forward, not a result.

**Result (1 Nov 2026).** Fixed settings, live-like 2008-25, in-sample by design; base 15.48% / 0.697 / -28.6%, 2008-12
0.037, 2013-19 0.516, 2020-25 1.354, and 0.461 at twice the costs.

| Look | CAGR | Sharpe | 2008-12 | 2013-19 | 2020-25 | At 2x costs | Read |
|---|---|---|---|---|---|---|---|
| L1 Residual momentum, book's form | 15.37% | 0.698 | -0.010 | 0.480 | 1.411 | 0.497 | Fails |
| L2a Sector momentum, 126 days | 16.43% | 0.746 | 0.030 | 0.556 | 1.439 | 0.526 | Fails: +0.049, lower in 2008-12 |
| L2b Sector momentum, 252 days | 16.65% | 0.764 | 0.065 | 0.589 | 1.434 | 0.550 | Clears the read rule |
| L3 Within-sector momentum | 17.44% | 0.809 | 0.013 | 0.573 | 1.596 | 0.597 | Fails: lower in 2008-12 |
| L4 Sector cap, a measurement | 15.78% | 0.717 | 0.049 | 0.531 | 1.390 | 0.478 | The cap costs nothing |

- L5 IBS below 0.2: NIFTYBEES -5.9 bp net a trade (t -0.66), GOLDBEES +3.7 bp (t 0.54, on an assumed 8 bp round trip). Fails.
- L2b, checked after it cleared: 63 / 126 / 189 / 252 / 378 days read 0.726 / 0.746 / 0.702 / 0.764 / 0.823, all at or above
  the base but not smooth. Unmapped names at a neutral score: 16.48% / 0.756, 2008-12 0.034, so that one condition slips by
  0.003. Five placebos with shuffled sector labels: 0.607-0.635, all below the base, so the gain is not map membership.
  `bootstrap_p` against the base 0.24; DSR 0.60 at N 61 against the base's 0.51, far under 0.95. Walk-forward not run: the
  core has no configuration before January. A candidate only, and only after the map covers the names that dropped out.
- L4: the cap binds little, a sector above 25% on 11% of days without it and 4% with it, so live's cap does not undermine
  the live-like figures. 24% of the book's stock weight has no sector on today's map, so this reads the mapped names only.
- L3's p is 0.051, the best here, and its failure is 2008-12 by 0.024. It stays failed; no re-reading.

**Built.** `portfolio.check_book_weights` refuses a NaN, negative or over-gross book each decision; a NaN passed the old
`>` test. `compute_signal_panels` refuses negative, NaN or all-zero group weights. `newey_west_cov` refuses collinear
designs (scaled condition number above 1e8), so a flat benchmark gives an error, not a t-stat. No output changed: the base
run before and after the checks is identical, canary ok. New money code: none, so nothing to move to Decimal.

**Book's maths reproduced, 11 of 11:** (3.30)-(3.33), (3.41), (3.44)-(3.46), (3.48)-(3.53), (3.85), (3.93), (4.13),
(10.16)-(10.18). Centurion's matches (3.2), (3.11), (3.55), (3.85) on a diagonal covariance, (6.5)'s ratio and the FDM.
`capped_weights` held every cap on 60,000 random books, largest shortfall 0.0006%.

**Notation flagged.** (3.48) lacks its "= 0". (4.7)-(4.9) sum over stocks i where the serial regression needs dates t. (3.66)
needs <= and >=: P rarely equals a past extreme exactly. (4.10) is undefined at H = L, a locked day. (19.1) returns 100%
commodities on a negative headline inflation with HI = -1, CI = +1, and is undefined at HI = 0. Time in (3.1) runs backwards.
Sect. 3.20's combination needs far more alphas than periods; at Centurion's 2-7 groups it reduces to inverse-volatility weights.

## 5ax. MIT1: Aiuchi, "An Empirical Analysis of Quantitative Trading Strategies" (MIT Sloan MBA thesis, 2008; your ask, declared 01 Nov 2026, 10:58 IST, before any run)

**What it is.** Long-only timing rules on the S&P 500, 1998-2007, closing prices, no costs, cash at zero. Its "best
strategies" are grid-search winners over the same ten years; it has no out-of-sample test. Its lasting part is the
evaluation: the signal split (Sect. 4.1), the upper and lower boundaries (4.2, 4.3), the random-signal test (Table 5.5).

**Declared before any run.** Nothing here is a trial on any Centurion family; no parameter is chosen.
- R1 Replication: buy-and-hold (Table 5.3), Table 5.5's random signals, and his best rules at his printed parameters on
  S&P data, with stated assumptions where the text is silent. Tolerance: return within 1.5 points, in-market share within 0.05.
- R2 Out of sample: the same rules, parameters unchanged, on the S&P 2008-25 and NIFTY 2008-25, with 2 bp and 18.5 bp round
  trips and cash at the T-bill or liquid rate. A rule survives only if it beats buy-and-hold on net Sharpe in both.
- R3 Boundary check: his printed (4.3) against his Figure 5-48, and a corrected boundary with the factor (1 - p_in).
- R4 Centurion's regime gate, the one timing rule it runs, on NIFTY 2008-25: timing excess, share of the perfect-timing
  boundary captured, and every circular rotation of the real signal as the null, which keeps its persistence.
  Read: p <= 0.05 is skill at picking days; above that, the gate is judged only as crash protection, which is a different claim.

- R5, added 11:02 IST on 1 Nov, after R4 and before this run: the metal sleeves with their trend filter off, fixed settings, live-like
  2008-25, a look. Read as for S151: Sharpe up 0.05 or more, not lower in 2008-12, higher in 2013-19, 2020-25 and at 2x costs.

**Result (1 Nov 2026).**
- R1 Replication on S&P closes (Yahoo). Table 5.3: all ten annual returns and the 18.02% volatility exact; 4.23% exact as a
  compound rate. Table 5.5, 20,000 draws a row: means within 0.12 points, sd within 0.07, and sd = sqrt(p(1-p) sum r^2) / T in closed
  form. Rules, return thesis to mine: MACD(25,280,0.02) 8.37 to 8.37, 0.74 in market both; Contrarian weekly 7.53 to 7.28; NN(14,50,0)
  8.86 to 7.98. Not reproduced, outside tolerance: TBR(8,100,0.5) 7.59 to 3.77; TBR with volume 11.75 to 3.79; CPT 10.71 to 4.98;
  Momentum on the VIX 8.21 to 5.36; on the T-bill 5.97 to 3.52; Contrarian on volume 7.72 to 6.09. The text does not fix their
  history windows or how a change is measured, and the assumptions were not tuned to close the gaps, so these stay unexplained.
- R2 Out of sample, parameters unchanged, net of costs; buy-and-hold Sharpe 0.456 on the S&P and 0.266 on NIFTY. S&P: Contrarian
  0.477, the rest 0.00-0.45. NIFTY: MACD 0.121, Contrarian 0.038, TBR -0.016, CPT -0.087, Momentum on VIX -0.098, NN -0.740. The two
  VIX rules are partial: India VIX starts in 2014 in the store. No rule survives: Contrarian beat buy-and-hold only on the S&P.
- R3 Printed (4.3) gives 13.5% at p_in = 1; Figure 5-48 shows about 4.7%, which is annual volatility with daily N, so the plotted band is too narrow by
  sqrt(252). "All the best strategies are above the 5% boundary" rests on that. With the correct units and (1 - p_in): 6.6-7.2% at his
  strategies' p_in, and Voting 3.37% and Best Performer 3.80% fall below it. Against the grids alone, at least 2,904 rules, Bonferroni z is 4.14; his best is 3.41.
- R4 The regime gate on NIFTY 2008-25, lag 2: in the market 79% of days, NIFTY 7.6% a year in, 27.0% out, capture -3.8%, rotation p 0.94. It
  does not pick days. Out-of-market days ran at 33.9% volatility against 15.0%; NIFTY's 30 worst weeks -7.12% to -2.29%. So it stands as risk
  control, as GT2A found on the book (no exit: 14.9% / 0.64 / -40.1%). It curtails rebounds, as U22 records; R11's faster re-entry failed. Gold's filter: p 0.27.
- R5 Metal sleeves unfiltered: 14.05% / 0.687 / -17.1%, Calmar 0.82, against 15.48% / 0.697 / -28.6%, 0.54. 2008-12 Sharpe 0.04 to 0.51, 2013-19 0.52 to 0.28,
  2020-25 1.35 to 1.19, 2x costs 0.46 to 0.53; mean metals 29% to 39%; `bootstrap_p` 0.51, DSR 0.59 at N 61. Fails its read rule. Not queued; your call 1 Nov: the filter stays on.

**Built.** `nse_engine/validation/timing.py` and `run_nse_engine timing-skill`; a line in the protocol's gates. No output changed: no engine code touched.

**Flagged.** (4.3) omits (1 - p_in). Sect. 4.4 says Sharpe is return over volatility, but Table 5.27's do not match: 7.53 / 13.76 is 0.55, printed 0.63; 8.86 / 12.73 is 0.70, printed 0.53.
Table 5.27 prints the hedge fund index at 9.46%, Table 5.4 at 9.25%. Sect. 3.2.3 defines the moving averages on returns; only prices reproduce 8.37%. Sect. 3.3.1's |rho| counts a mirror-image
pattern as a neighbour, which its arithmetic-average forecast then reads with the wrong sign.

**S151A look (1 Nov 2026, 11:20 IST, before this run).** L2b and the sector cap again, on today's map plus `config/nse_sector_backfill.json`. A look, nothing recorded.
Read: the January configuration stays queued only if L2b still clears 5aw's rule on the fuller map.

**S151A result (1 Nov 2026): FAIL, the January configuration is dropped.** 574 of the 1,016 universe names had no sector (54% of the universe in 2008, 4% in 2025). 262 were labelled from
company knowledge, uncertain names left out; 57 of 59 matched NSE on a blind sample of current names, which are easier than these. Names without a sector fell to 15% in 2008 and 2% in 2025. On the fuller map:
sector momentum 15.89% / 0.702 / -29.6% against the base 15.48% / 0.697 / -28.6%; 2008-12 -0.036 against 0.037; 2013-19 0.505 against 0.516; 2020-25 1.432 against 1.354; 2x costs 0.498 against 0.461;
`bootstrap_p` 0.47, DSR 0.54 at N 61. It fails: Sharpe up 0.005, not 0.05, and lower in two periods. The earlier 0.764 was mostly survivorship.
Correction to 5aw: the placebo shuffled labels among surviving names only, so it could not detect this, and "the gain is not map membership" was wrong. L3 used the same map and is not to be trusted either.
The sector cap holds up: 15.70% / 0.711 / -28.8% against 15.48% / 0.697 / -28.6%, a sector above 25% on 7% of days against 19%, so it still costs nothing. No code stays, and the labels were discarded (your rule, 1 Nov).

## 5ay. OPS1: Lahanis, Liu and Zhou, "Online Quantitative Trading Strategies" (NYU Stern Glucksman Fellowship; your ask, declared 01 Nov 2026, 11:28 IST, before any run)

**What it is.** Online portfolio selection: long-only simplex weights, rebalanced daily, on NASDAQ 100 stocks 1998-2010, frictionless by its own
Sect. 6.4. Headlines: CWMR 1,590x, Sharpe 1.75; PAMR 788x, 1.63; CRP 12.16x, 0.76; buy-and-hold 9.17x, 0.58. Its repository is public; it is not cloned or copied.

**Declared before any run.** Looks only: nothing recorded, no parameter chosen, no change to any book.
- G1 Replication: PAMR, OLMAR, EG, CRP and buy-and-hold at Exhibit A's parameters on Yahoo data for stocks that were NASDAQ 100 names and still trade: survivors only,
  so a higher level than the paper is expected. Read: same order and same scale of PAMR over CRP. Plus known-answer checks on synthetic data.
- G2 India: PAMR and OLMAR on the 100 most liquid point-in-time names 2008-25, daily and weekly, at a 15 bp cost a side, with next-day fills as Centurion's orders have. CWMR is
  left out: the paper's (13) omits its constraint, so there is nothing to implement from it. Read: reaches January's queue only if net CAGR and Sharpe beat both equal-weight on the same names and the base book (15.48% / 0.697).
- G3 EG across Centurion's streams, the book, gilts, MCX-B and liquid, monthly, at 5 bp a side. Read: the diversifier gate of 5.1, against the book.
- G4 The paper's cleaning, non-positive prices set to a small positive number, on PAMR, with one bad print. A measurement.
- G5 The book's own cost margin: how many times its modelled costs it can bear before it falls to passive NIFTY and gold. A measurement.

**Result (1 Nov 2026).**
- G1 Replication on 39 survivors, not its members; thesis to mine: PAMR 788x to 1,115x (Sharpe 1.63 to 1.44), CRP 12.16x to 14.03x (0.76 to 0.69), buy-and-hold 9.17x to 8.61x. Same order and scale. Not its data, so no closer
  tolerance applies. PAMR breaks even with CRP at 13.4 bp a side; at 25 bp, 0.28x. OLMAR at Exhibit A's parameters moved on 0 of 3,272 sessions, so its 12.1584x is CRP's, not a result. EG at its parameters differs from CRP by 1.8% here, so the paper's exact tie on wealth, Sharpe and drawdown is
  unexplained. CWMR not run: (13) omits its constraint. One day's lag: PAMR 204x, Sharpe 1.10.
- G2 India, 100 most liquid point-in-time names 2008-25. Equal-weight 7.48% gross, 6.39% net, Sharpe 0.11. PAMR(0.9, 10): -30.5% a year gross, -52.3% net, MaxDD -100%, at 50% turnover a day. PAMR(0.5) and OLMAR(5, 10) worse. Weekly, net: PAMR -8.5%, OLMAR -38.3%. All fail.
  Weekly rows at a one-day lag are dropped: the simulator's turnover there is wrong; zero lag already fails. I expected survivorship to explain the gap with the US; a control on 40 Indian survivors explains only part: PAMR 0.50x gross, zero lag, 18 years; 4.8x at one day's lag; net of 15 bp it is nil. The effect did not carry over.
- G3 EG across the book, gilts, MCX-B and liquid, monthly: 9.38% / 0.478 at eta 0.05, the same as a fixed quarter each (9.37% / 0.480); worse as eta rises; `bootstrap_p` 0.68 and up. Fails 5.1. Learning added nothing to a fixed mix.
- G4 One bad print set to a small positive number, as Sect. 3.1 does: PAMR 1,115x to 4.5 billion x. Centurion leaves it NaN and never clips.
- G5 Cost margin, `run_nse_engine cost-margin`: modelled costs 2.28 points a year on the walk-forward book. It reaches passive NIFTY and gold's 12.4% at 1.25x costs; 1.5x gives 11.76%, 2x 10.49%. These sit on top of D2's haircut; if that gap is all costs the ceiling is 1.92x. G4 passes costs up to 1.5x and RA1 reads G4, so
  this is not changed. It does mean a costs reading of 1.5x is not comfortable: the first real fills matter more than any research item.
- Found by the audit: 14 corporate actions with a partial factor, 4 inside the universe: ADANIENT 2015-06-03 (raw -83%, factor 0.9, adjusted -81%), HINDZINC 2011-03-07, ZEENEWS 2010-04-15, TRIVENI 2011-05-03. No warning fired, because an event is on record. Repairing the four: 15.48% / 0.697 / -28.6% to 15.55% / 0.699 / -29.3%, fixed settings.
  Left as is until the next re-baseline: a repair moves the data hash.

**Built.** `validation/cost_margin.py` and `run_nse_engine cost-margin`; `panel.partial_adjustments`, a warning at load, numbers unchanged; a protocol line. Results in `data/nse_engine/gl/`.

**Flagged in the paper.** Tables are the grid's in-sample winners: 5.1.2 describes walk-forward validation but reports no out-of-sample figure. (2) defines Best Stock as yesterday's winner, yet the text calls it a hindsight upper bound; its 0.0153x shows which was run. Table 2 prints two exponential growth rates for the same 36.88x. (22) is (6). (23) has no projection to the simplex.
Exhibit A gives Anticor alpha 2.5, outside Table 2's grid. (12) and (13) are incomplete as stated.

## 5az. RQ1: Ge et al., "Research on Quantitative Trading Model" (BCP Business & Management 26, 2022; your ask, declared 01 Nov 2026, 11:53 IST, before any run)

**What it is.** An attention BiLSTM forecasting gold and bitcoin price levels, 7:3 split, plus position sizing that adds on falls. Claims: gold R-squared 0.94, bitcoin 0.99, $1,000 to $216,133 in five years.
No code published. Sizing, VaR and the greedy step are not specified well enough to rebuild exactly.

**Declared before any run.** Looks only: nothing recorded, no parameter chosen.
- Q1 Rebuild the model on Yahoo gold and bitcoin, 2016-09 to 2021-09, trained two ways: clean, on the first 70% with scaling from it alone; and on all the data, as Sect. 2 says. Against tomorrow = today. Read: the model has skill only if it beats that naive forecast on RMSE, clean.
- Q2 Trading: long or cash from the sign of the forecast, the test window, 1% gold and 2% bitcoin fees, against buy-and-hold.
- Q3 Eqs 4-8: compute the adds, and test whether runs of rises and falls differ from coin flips (its Apriori step), on gold, bitcoin and NIFTY.
- Q4 India: the same model on NIFTY, refit each year 2012-25, on levels and on returns, 18.5 bp a round trip. Read: reaches January's queue only if it beats the naive forecast, beats buy-and-hold on net Sharpe and has rotation p <= 0.05.

**Result (1 Nov 2026).**
- Q1 On Yahoo closes the naive forecast, tomorrow = today, beats the paper's best bitcoin row on all four measures: RMSE 1,364 against 1,479, MAE 812 against 906, MAPE 2.83 against 2.88, R-squared 0.9942 against 0.9929. On gold the paper's 19.01 beats naive's 21.54, but its LSTM, GRU, BiLSTM, ARIMA, Holt-Winters, HMM and XGBoost rows (27.8-40.0) are all worse than naive, so its gold data or its fit differs from mine.
  My rebuild is much worse than naive on both (gold 79-524, bitcoin 3,754-8,605), clean or on all the data. The paper gives no layer sizes, so this is not a replication of its network; the naive comparison needs none. Sect. 2's "all the data" did not reproduce its table, so leakage is not shown.
- Q2 Test window, net of fees, three seeds: gold 0.80-1.05x against buy-and-hold 1.18x; bitcoin 0.60-2.39x against 5.67x. The $216,133 could not be rebuilt: sizing, VaR and the greedy step are not specified.
- Q3 Runs of rises and falls match coin flips on gold and bitcoin (p 0.24-0.55), so the Apriori step found nothing there. NIFTY differs (p about 0.003) with longer runs than chance, and after 4 falls the next day rises 41% of the time against 53% overall: falls persist, against adding on them. The adds from (5)-(6) are not a growing martingale: 1, 1.17, 0.01 on gold, then almost nothing; my statement before the run was wrong.
- Q4 NIFTY, refit yearly 2012-25, 18.5 bp: on levels R-squared 0.969, RMSE 1,071 against naive's 127, 7.56% / 0.136. On returns, R-squared 0.9996 on levels and -0.002 on returns, in the market 99% of days: it learned "up". 12.31% / 0.416. Buy-and-hold 14.76% / 0.547. Rotation p 0.62 and 1.00. Fails.

**Built.** A protocol line only: Centurion has no price forecaster and reports no level R-squared, averaging down is absent (weights come from targets, capped at 8%), and CVaR is in the scorecard. A forecast-skill module would have no caller, so none was added.

**Flagged.** (4) divides by the current price, not the previous one. (5)-(6) put Mu on all terms but the first, and (7) calls an exponential fit linear regression. (8) leaves PRO undefined and says x where it sums to u_n. Sect. 2 says all the data trains the model; Sect. 2's comparison says the test set does not. RMSprop is called an activation function; it is an optimiser. Viterbi is said to estimate the transition matrix; it decodes states. XGBoost does not use a random forest. "The average predicted increase of gold is 0.43%, and the average predicted value of the increase is 0.41%" says one thing twice. VaR and the greedy step have no formula. The end date prints as 9/10/202.

## 5ba. FF1: Fama and French, "The Cross-Section of Expected Stock Returns" (Journal of Finance, 1992; your ask, declared 01 Nov 2026, 12:23 IST, before any run)

**What it is.** NYSE, AMEX and NASDAQ, July 1963 to December 1990. Monthly Fama-MacBeth regressions: ln(ME) -0.15 (t -2.58) alone; beta 0.15 (t 0.46); ln(ME) -0.11 (t -1.99) with ln(BE/ME) 0.35 (t 4.43) together. Table V: average monthly return rises from 0.64% to 1.63% across book-to-market.

**Declared before any run.** Looks only: nothing recorded, no parameter chosen; the value family is closed at N = 2 and its third slot is reserved.
- F1 Replication: Table V's All row and column from Ken French's 100 size and book-to-market portfolios, equal-weighted, July 1963 to December 1990. Tolerance 0.15 points a month: his data is revised since 1992.
- F2 Out of sample: SMB and HML by period, 1963-90, 1991-2007, 2008-25.
- F3 India: monthly Fama-MacBeth on the point-in-time universe 2013-25, on beta, ln(ME), ln(BE/ME), and Centurion's combined forecast; book values lagged to `available_from`. Read: size or book-to-market is worth a January look only at t >= 3 over the period and the same sign in 2013-19 and 2020-25, and only then if the top decile, long only, beats equal weight net of 15 bp a side.
- F4 Whether Centurion's own forecast keeps its slope once size, value and beta are controlled.
- F5 Beta on the book: simple against Dimson's sum with a lag (the paper's p. 431). If they differ, alpha is misstated.

**Result (1 Nov 2026).** The paper is a scan; pages 427, 431, 438, 446 and 448 were read directly, with the slopes cited from those.
- F1 Table V on French's 100 portfolios, 330 months as in the paper, tolerance 0.15 points. Size column 10 of 10 within it, largest gap 0.07. Book-to-market row 9 of 10: 0.73 against 0.64 low, 1.36 against 1.63 high. French sorts the two independently, the paper one within the other, and my All row averages cells, not stocks, so the corner differs. Same shape.
- F2 Monthly factor means, percent (t): SMB 0.23 (1.46) in 1963-90, 0.18 (0.72) in 1991-2007, -0.02 (-0.13) in 2008-25. HML by period is in `data/nse_engine/ff/factors.txt`. The size premium in the paper's own span is weaker on this factor than its regressions suggest, and is gone since 2008.
- F3 India, 155 months 2013-25, 241 names a month, Dimson beta. Alone: ln(ME) 0.012 (t 0.10), beta -0.24 (t -0.55), E/P 0.68 (t 0.28). Together: ln(ME) -0.11 (t -1.14), beta -0.42 (t -1.03), E/P 1.52 (t 0.69). Nothing reaches t 3. Decile portfolios net of 15 bp a side, against equal weight 14.50% / 0.443: smallest 6.74% / 0.173, cheapest 15.19% / 0.409. Fails.
  **BE/ME could not be tested:** quarterly filings carry no balance sheet: `reserves` is nearly empty, `equity` starts in 2022 at about 28% of filings. Today's price-to-book would be hindsight, so it was not used. E/P stands in, and the paper itself finds BE/ME absorbs E/P, so this is not its test.
- F4 Centurion's forecast, slope per target unit, percent a month: 0.448 (t 2.76) alone; with beta, size and E/P held fixed, 0.475 (t 3.61, HAC 3.81), 0.428 (t 2.44) in 2013-19 and 0.527 (t 2.6) in 2020-25. It carries information they do not explain. Settings were chosen on overlapping data, so this is support, not proof.
- F5 Book beta: simple 0.173, Dimson 0.203, alpha 5.70% to 5.55%; 2013-25 0.312 to 0.372, alpha 7.59% to 7.13%. Non-synchronous trading flatters alpha by 0.15-0.46 points. Not changed: gate 2 compares like with like.

**Built.** `validation/fama_macbeth.py` and `run_nse_engine signal-slopes` (forecast slope 0.478, t 3.63: the small gap to F4 is the deployment config against the run's), a protocol line. No engine code touched.

**Limits.** No costs or shorting limits in the paper; US data 1963-90; three sorted corners carry much of it. Fama-MacBeth treats each month's stocks equally, so small names weigh as much as large.

## 5bb. YZ1: Yang and Zhang, "Drift-Independent Volatility Estimation" (Journal of Business, 2000; your ask, declared 01 Nov 2026, 12:43 IST, before any run)

**What it is.** A variance estimator from open, high, low and close, eq. 7: V = VO + k VC + (1 - k) VRS, k from eq. 10, unbiased under any drift and opening jump, about 7.3 times as efficient as close-to-close at n = 10 and f = 0.25. Centurion sizes on 60-session close-to-close volatility (`realised_vol`).

**Declared before any run.** Looks only: nothing recorded, no parameter chosen.
- Y1 Replication by simulation: k0, efficiency, unbiasedness, drift independence against Garman-Klass, and the discreteness bias of Sect. III.
- Y2 NSE: the opening-jump share f by eq. 8-9, against the paper's 0.18-0.30 for US stocks, and whether the OHLC data meets what the estimator assumes.
- Y3 Forecasting: 60-session close-to-close against Yang-Zhang, at predicting the next 21 sessions' realised variance, point-in-time universe 2008-25.
- Y4 Sizing: fixed settings, `vol_ann` from Yang-Zhang. Read: adopted for a January configuration only if Sharpe rises 0.05 or more over 2008-25, is not lower in 2008-12, and is higher in 2013-19, 2020-25 and at twice the costs. A better forecast that does not move the book is not adopted, and under your rule of 1 Nov its code goes.

**Result (1 Nov 2026): not adopted, no code.**
- Y1 Reproduced by simulation, n = 10, f = 0.25: k0 0.1327 against 0.13; efficiency formula 7.29 against 7.3, simulated 7.56; n = 10 maximum 8.54 against 8.5; n = 2 peak 14.08 against 14. Under drift of 3 sd a period, Yang-Zhang 0.95, close-to-close 0.99, Garman-Klass 2.11, Parkinson 2.58, true 1.00. Sect. III's downward bias shows: 0.84 at 50 steps, 0.92 at 200, 0.96 at 1,000.
- Y2 NSE universe 2008-25: f median 0.195, quartiles 0.16-0.24, inside the paper's 0.18-0.30; RELIANCE 0.22, INFY 0.35; NIFTYBEES 0.47, GOLDBEES 0.49, SILVERBEES 0.49, because their markets trade overnight. OHLC is clean: of 1.3 million name-days none has a high below the open or close or a low above them, 87 lack an open, 1,960 have high = low, and no Yang-Zhang value is negative.
- Y3 It forecasts worse than close-to-close. Next 21 sessions' variance, 60-session estimators: QLIKE 0.782 against 0.425, t -11.3; log-error MSE 0.829 against 0.609; rank correlation 0.557 against 0.582. Close-to-close wins 89% of months, least in 2020-25 (80%).
  Why: two assumptions fail. The proof needs overnight and intraday moves independent; on NSE their median correlation is -0.19 (-0.31 in 2008-12, -0.11 in 2020-25), so VO + VC overstates variance by 18%. It needs E[VC] = E[VRS]; VRS runs 22% above VC. Together Yang-Zhang sits 39% above close-to-close. That reads as noisy opens and range noise, not more information.
- Y4 Sizing on it, fixed settings: 15.59% / 0.694 / -29.5% against 15.48% / 0.697 / -28.6%; 2008-12 0.048 against 0.037; 2013-19 0.510 against 0.516; 2020-25 1.350 against 1.354; 2x costs 0.466 against 0.461. Fails: Sharpe does not rise. A scaled variant is identical, because sizing reads only relative volatility.

**For later.** From 7 Dec 2026 the close can sit outside the day's range (CAS1), which breaks VRS; any range-based estimator would need a guard. Gaps above 3%: the rest of the day reverses 206 bp on average, 44,679 cases. It may be stray first prints that cannot be filled, so it is evidence for INT1C's gap idea to test on minute data, not a result. NSE opens being noisy also means open fills deserve the same doubt on gap days.

**Flagged.** Eq. 7's optimality rests on independence the data does not have. p. 486 prints the drift parameter as mu sigma / sqrt(T) where p. 481 has mu sqrt(T) / sigma; the second is taken as meant. No code was published.

## 5bc. IMG1: Imagawa, "Negative Convexity in Pass-Through MBS" (arXiv 2610.38275, 1 Nov 2026; your ask): does not fit, no code

**What it is.** A pricing paper on US fixed-rate mortgage pass-throughs: price split exactly into three parts (eq. 26), closed form by cumulant expansion under one-factor Hull-White (eq. 32), convexity split by part, and phase diagrams of its sign. No backtest, no market data, no published code.

**Why nothing is taken.** Centurion holds no MBS, Kite offers none, and the mechanism needs a borrower's option to refinance a fixed-rate loan; Indian home loans are mostly floating, and no Indian market gives that exposure. Its Hull-White and prepayment machinery has no input here. Forcing it in would add code with no caller.

**Checked.** Rebuilt its Monte Carlo at the parameters of Table 1 (page read directly: T = 30). It does not reproduce Table 2: 1.337 against 1.171 at R = 0, 0.737 against 0.850 at R = 0.07, 0.992 against 0.998 at R = m. T = 10 fits far better (1.158, 0.864, 1.000), so either the rebuild differs somewhere or the table used other settings; not settled, and not worth settling. Its claim that a second cumulant beats a first is therefore not verified here.
One idea was worth a look, a book's own convexity: monthly excess returns on NIFTY's and its square, 2008-25, live-like. Deployed curvature -0.32, t -1.03, beta 0.20, up capture 0.42, down capture 0.18. With a1, -0.32, t -1.05. No reliable sign, and capture favours the downside. Not a test of a1's tail: a1's weight is small in this composite.

## 5bd. JAC1: Jacquier, "Strong rate for one-dimensional driftless continuous strict local martingales" (arXiv 2610.39896, 1 Nov 2026; your ask): does not fit, no code

**What it is.** A numerical-analysis proof: an Euler scheme killed at R_h = h^-alpha converges in L1 at rate min(theta alpha, 1/2) for dX = sigma(X) dW when X is a strict local martingale (Theorem 2.10), worked for CEV (Prop. 3.1). No code published, no market data.

**Why nothing is taken.** Centurion simulates no SDE: its Monte Carlo is a bootstrap of real days (`mc_bands`, `services/risk/monte_carlo_risk.py`) and option pricing is Black-Scholes. It has no CEV, SABR or local-volatility model, so the failure the paper repairs cannot occur.

**Audited, the one link.** The paper notes that under such models a call can price below intrinsic and implied volatility is undefined. `theory.implied_volatility` is right: None below intrinsic or above the bound, ValueError on NaN and on a zero spot; covered by tests. `option_chain.compute_iv` returns 0.0 on failure; `compute_atm_iv` filters zeros and `enrich_chain_with_greeks` skips Greeks, so nothing is averaged wrongly. The one blemish: the Options page can show an IV of 0 for an unsolvable quote. Left as it is.

**Replication, by scratch code, 20,000 paths, same grid.** E[X_T] for gamma = 1: 0.684 against the exact 0.683. L1 slopes, mine against the paper's observed: 1/4 0.51 and 0.55; 3/4 0.50 and 0.48; 1 0.485 and 0.50; 3/2 0.42 and 0.43; 1/2 0.44 and 0.46. Plain Euler at gamma = 1: 0.67 at h = 2^-2, 1e18 at 2^-3, overflow at 2^-5, as Sect. 3.2 reports.

**Flagged.** Sect. 3.2 sets R_h = h^-alpha, Table 3.1's caption 2 h^-alpha; the second was used. Prop. 3.1(iv) with Table 3.1 gives alpha = 1 at gamma = 1/4, where Sect. 3.2's list omits that case.

## 5be. KES1: Meister, "The Generalised Kessler Syndrome and the Persian Gulf" (arXiv 2610.40129, 1 Nov 2026; your ask)

**Rule, set before the code.** Applied to operations, not to the book. Prop. 1: a stock V with a persistent deficit e ends at T = V(0)/e, whatever today's flow shows. Centurion's own stocks are the Actions cache egress, Neon's compute and the Kite Connect subscription. Adopt a gate for the backup runs only if (a) October's logs show the backups do no work on a finished night, (b) every act they did do stays covered, (c) the gate fails open, so any doubt runs as before, (d) it sits behind a repository variable that is off by default.

**What it is.** A 10-page q-fin.PM note: output is a minimum across inputs (eq. 1), repair is itself a minimum (eq. 3), damage is reversible only if restoring takes less time than the buffers last (eq. 4), and with a persistent deficit the visible flow says nothing until the stock runs out (Prop. 1). Prop. 2 is water-filling on known needs; Prop. 3 a two-input binary Kelly problem. No published code.

**Replication, by scratch code.** Prop. 3's example (q 0.4, B 0.2, p 0.9 and 0.5): growth -1.513 at t = 0.05, the paper's -1.513 at 0.05; equal split -1.578; water-filling on means -1.553. Prop. 2's Z* = (S_K + B)/K checked against a numerical optimum.

**Assumptions.** Min payoff: fails for the book (returns add, so neither proposition sizes a position) and holds for operations (a live session needs every one of data, broker, database, secrets). Persistent deficit: holds for cache egress at the old rate, fails for Neon since NEO1's fix, unknown for the egress limit, which GitHub does not publish. Known reserves: fails for egress, holds for Neon (console) and Kite (expiry in its console, read by hand).

**Audit.**

| Idea | Present? | Location | Risk if missing |
|---|---|---|---|
| Weakest link at go-live (eq. 1) | yes | `capital_ladder.readiness`, `go_live_status` | high: a book called ready that cannot sell. The sell path is required, and `record-sell-path` has never run |
| Reversibility (eq. 4) | yes | save on `always()`, four more attempts, window to 05:59 IST | med: a refused cache took more than the 150 minute sync limit on 30 Oct, the next run recovered |
| Reserves, not flows (Prop. 1): Neon | partial | NEO1's 6-hour `/health`; no runway check | med: suspended 20-21 Oct, after 17 days at 6 CU-hours a day against 100 a month |
| Prop. 1: Kite Connect app | no | nothing tracks it | high once live: lapsed 17 Oct unnoticed, no warning before the login fails |
| Prop. 1: cache egress | no | every run restores | high: refused 30 Oct, and it can refuse the first run of a night |
| Kelly on the book (Prop. 3) | not applicable | | none: additive returns, no min payoff |

**Measured, October's logs, all 100 runs.** A run transfers 802 MB (652 store, 150 pip); the cache holds 11.35 GB in 17 entries. On finished nights backups rebuilt nothing, bar the 15 years the refusal itself forced. On 26-30 Oct all 19 backups that reached planning matched the first run on all 7 paper books (sells, buys, stops, skips, shift, drawdown, counts only: the contents are not in the logs). The one act that mattered: the 26 Oct email went from a backup. The 30 Oct failure was a backup on a finished night.

**Built, off by default.** `tools/paper_heartbeat.py` `session_finished` and `--session-finished`; the `engine-window` job calls it when the repository variable `CENTURION_SKIP_DONE_RUNS` is `true`, and a positive answer stops a scheduled run before its cache restore. Finished means all started books at the last completed session, the live session run (a no-login skip does not count), the email gone. Neon unreadable, a failed step, or a book behind: the run goes ahead. Dispatches are never gated, so the 19:00 IST run and any manual run are untouched. Variable unset: nothing changes but the window job's time limit, 2 to 8 minutes.

**Replay, 26-30 Oct (25 runs).** 19 stopped, including the 30 Oct failure; 26 Oct's emailing backup kept. 15 GB of 20 saved. A month of 22 sessions: about 88 GB down to 18 on nights without a login gap. Before 26 Oct the replay says nothing: the nightly email did not exist, and 20-23 Oct were Neon-down nights, which the gate lets through.

**Where it fails.** Not yet run against real Neon: this machine has no URL for it, so the first real test is the first backup after the variable is set. A key missing there only keeps the runs going. A fix merged after 19:00 no longer reaches tonight's plan by itself: dispatch by hand. If a first run does the work but fails afterwards, no run pings the dead man's switch and it alarms by 23:00 IST: loud, and false. A no-login night still costs five restores. GitHub's limit is unpublished, so this widens the margin and proves no number. Neon's 0.5 CU-hours a day is the NEO1 estimate, not measured here; the gate adds a few short reads to it and removes about 40 minutes of database use a night.

**Skipped.** Gulf analysis and satellite plume data (no use). Water-filling and Kelly on the book (additive returns). Labour dynamics. Deleting old cache entries (storage is not the limit: egress is). A code register of expiry dates: Kite's date lives in its console and renews monthly, a calendar reminder does the job with less to break.

## 5bf. KOH1: Kohli, "Decorrelating the Advice Layer" (arXiv 2610.37705, 1 Nov 2026; your ask)

**Result: nothing adopted, no code.** All three rules below failed, so no variant ran and the trial registry is unchanged. 2026 stayed sealed.

**What it is.** A 30-page agent-based simulation of AI wealth platforms, on 20 US large caps 1990-2022. Prop. 1: views behave like about 2.0 independent investors however many clients there are. Prop. 2: loop gain, finite iff kappa < 1. Prop. 3: correlation measurable from noisy sums. Prop. 4: spreading execution. Eq. 1: impact = Y sigma sqrt(F), with F decaying. Eq. 3: SLaR99. Eq. 5: fragility index. Table 6: dispersing overlays, staged execution and a participation breaker cut SLaR99 63%. The paper says its code is released, but the PDF has no repository link, so licence and code are unknown; nothing was taken from it.

**Replication, by scratch code.** Table 1, tolerance 0.1 on Neff and 0.01 on the rest:

| Quantity | Paper | Ours |
|---|---|---|
| Neff, H 0.66 / 0.28 / 0.20 (theory) | 2.01 / 2.65 / 2.85 | 2.01 / 2.65 / 2.85 |
| Neff, Monte Carlo | 1.98 / 2.68 / 2.94 | 2.06 / 2.61 / 2.90 |
| Cascade at kappa 0.20 / 0.50 / 0.80 / 0.95 | 1.25 / 2.00 / 5.00 / 19.89 | 1.25 / 2.00 / 5.00 / 19.81 |
| Prop. 3 rel. s.e., four cases | 0.050 / 0.051 / 0.054 / 0.064 | 0.050 / 0.055 / 0.052 / 0.061 |
| Prop. 4 ratio, H 2 / 5 / 10 | 0.50 / 0.20 / 0.10 | 0.50 / 0.20 / 0.10 |

The agent-based Tables 2-9 were not rebuilt: nothing here would use them.

**Results.**

- **KOH1a fails: 0.10 points, not 0.25.** At Rs 30 lakh, volatility-scaled impact costs 0.10 of CAGR up only (cap 3 the same) and adds 0.04 both ways; Rs 3 lakh 0.06. The shape of eq. 1 is right on NSE: in stress, against calm, Amihud illiquidity is 2.62 times, volatility 1.58, value traded 0.67, and sigma / value gives 2.36. The book is too small for it to matter: median participation 0.013% at Rs 30 lakh, mean impact 7.5 bp. It is first order, on fills as simulated.
- **KOH1b fails.** 281 stop sells on 184 days, 15 cluster days. Cluster minus isolated, 10 sessions: +0.40%, t 0.15; 2008-19 -0.79, 2020-25 +5.54 on 3 days.
- **KOH1c fails.** Out-of-sample 2018-25, 91 rows: FI -0.002 against volatility -0.032 at 252 sessions (gain 0.03), -0.035 against -0.021 at 63. Nothing forecasts: VIX alone 0.04, and that is all 2022-25. FII loop gain: median 0.04, 5th-95th percentile -0.12 to 0.13. FII positions rise with same-day returns (0.44) but do not follow yesterday's (-0.02).
- **Base.** Raw 16.87% / 0.791 / -31.6%. Live-like by my approximate invested weight 15.59% / 0.704 / -28.3%, against the recorded 15.48% / 0.697 / -28.6%.
- **Described, not a rule.** Regime exits are the book's real synchronized act: 19 days, median 48% of equity in one session, largest 99.6%. NIFTY after them: +0.1% in 5 sessions, +1.4% in 21, so no sign of selling lows, on 19 cases. Fixed settings, raw, through six falls: NIFTY -59.9 / -27.7 / -21.7 / -38.4 / -17.0 / -15.8, book -29.4 / -4.0 / -13.0 / -10.3 / -6.2 / -3.9. The fast one, 2020, did not hurt the book most. In-sample, not evidence of skill.

**Assumptions.**

| Assumption | For Centurion |
|---|---|
| Book's flow moves prices (eq. 1, Prop. 2) | fails: 0.013% of daily value at Rs 30 lakh, so kappa is nil |
| Square-root impact, scaled by volatility | holds on NSE, by Amihud |
| 20 equal-cap US large caps, 1990-2022 | fails: NSE universe, 3,000 names |
| 7% market-wide breaker | fails: NSE has 10 / 15 / 20% index breakers and stock bands |
| Many agents on one model; views from an LLM | fails: no LLM in the engine's path; legacy `auto_executor` only |
| Agentic flow q is observable | fails: FII futures position change is a stand-in, an assumption of this test |
| SLaR's d, the agent-induced move | fails: unobservable in real prices, so total NIFTY tails stand in |
| Stationary bootstrap, 20 days | unknown for NSE; ours already resamples this way |

**Audit.**

| Idea | Present? | Location | Risk if missing |
|---|---|---|---|
| Square-root impact | yes | `costs.py:238` | none |
| Volatility in impact (eq. 1) | no | `costs.py:245` | low: 0.10 points at Rs 30 lakh |
| Participation cap (M6) | yes | `costs.py:248`, 5% of ADV | none |
| Effective bets (Prop. 1) | yes | tracker: PCA ENB 1.4-1.8; `dsr.py:274` for trials | none |
| Paired resampling (Sect. 3.4) | yes | `diversifier_gate.py:113` | none |
| Stationary bootstrap | yes | `mc_bands.py:79` | none |
| Dispersed stops (M5) | no | `config.py:153`, one 3 ATR | low: clusters do not rebound |
| Staged drawdown cuts (Sect. 8.2) | not applicable | `drawdown.py`: halts entries only, sells nothing | none |
| Shared feed (Sect. 6.3) | partial | `store.py:110`, `:504`: bad bars, prev_close against stored close | low: the exchange's own file, one source for all 7 books |
| Monoculture of LLM views | not applicable | no LLM in the engine | none |
| Differential privacy | not applicable | one owner, no platform to pool | none |

**Where the method fails.** In the paper, FI is scored on the simulator that its formula came from: hold-out R2 0.56 is the model checked against itself, and on real NIFTY tails it is nil. Its loop gain adds nothing there either (0.56 with and without), and eq. 4's correlation term scores 0.01. Elasticity 0.47 is the input: with the impact exponent set to 0.5, 0.7 and 1.0, a rule-driven flow returned 0.5, 0.7 and 1.0. The mitigations act through a layer's own impact, which this book does not have, so they are the wrong tool at any rung on the ladder.

**Flagged.** Sect. 5 gives real index volatility 18.3%, Table 3 18.9%; stress correlation 0.49 and 0.48. Sect. 7 calls selling after declines b < 0, but with eq. 1's buy-positive flow that is b > 0; kappa is unchanged by the sign, buy-positive assumed. Fig. 12 cites Section 6 for the slopes, which are in Section 7.

**New finding, nothing changed.** `monte_carlo_risk.block_bootstrap` never samples the last trade: `randint` excludes its high, so a block ends at n - 2. Its docstring cites the stationary bootstrap and the code uses fixed blocks. Only the legacy `carver_pipeline` imports it, not the engine.

**Rules, set before any test.** All on 2008-25 at most, 2026 stays sealed. No book changes unless a rule passes and a variant then clears the usual gates.

- **KOH1a, stress costs (eq. 1, Table 9).** Impact scales with current volatility; `costs.impact_bps` does not. Rescale the book's fills by each name's volatility against its own past year, capped at 2, as Table 9 caps it, then at 3. A config flag is built only if this costs 0.25 points of CAGR or more at the capital the ladder reaches (Rs 30 lakh).
- **KOH1b, synchronized exits (Sect. 8.2).** Stops share one rule (3 ATR). Cluster days are the top decile of days by stop sells, with at least 3. Sold names' 10-session return over NIFTY, cluster minus isolated: a dispersal variant is tried only if the gap is positive, t >= 3 with dates clustered, and has the same sign in 2008-19 and 2020-25.
- **KOH1c, fragility index (eq. 5).** With FII index-futures position changes as q: does walk-forward FI beat volatility alone at forecasting the next 63 sessions' 5th percentile 5-day NIFTY return, 2013-25? Adopted as a context line only if out-of-sample R2 gains 0.05 or more and both halves agree. Never scored.

## 5bg. HAN1: Han and Qu, "Sparse Autoencoders as Implicit Diversification Regularisers" (arXiv 2610.39205, 1 Nov 2026; your ask)

**Result: nothing adopted, no code.** HAN1a and HAN1b failed; HAN1c could not be applied, because NSE showed no reduction to explain. Nothing passed, so no variant goes to January, and 2026 stayed sealed.

**What it is.** An 8-page ICAIF paper. A tied, ReLU autoencoder with an L1 penalty on the code (eq. 1-2), on cross-sectional rank scores (eq. 3), 433 crypto pairs 2019-24, with 200 S&P stocks as a secondary check. Ranks by reconstruction, long the bottom quintile, short the top. Claims: 20-26% lower long-short volatility than matched PCA, five of five folds, at K 128; within-basket correlation 0.008 against 0.039. Its own limits: no Sharpe gain over a naive contrarian, factors not reproducible across seeds, equity Sharpe does not transfer, and a Ledoit-Wolf overlay recovers about 73% of the volatility gap. Entry at the signal's own close, pre-cost; the authors call it not executable. Paper CC BY 4.0; no code link.

**Results.**

- **HAN1a fails.** Live-like, fixed settings:

| | Sharpe | Vol | 2008-12 | 2013-19 | 2020-25 | 2x costs | Turnover |
|---|---|---|---|---|---|---|---|
| Base | 0.704 | 12.9 | -0.009 | 0.541 | 1.381 | 0.487 | 13.2 |
| Minimum variance | 0.647 | 12.4 | -0.063 | 0.593 | 1.169 | 0.38 | 16.4 |
| Half and half | 0.679 | 12.6 | 0.051 | 0.483 | 1.320 | 0.446 | 14.2 |

  1,573 solves, no failures, covariance positive definite throughout, condition number at most 63. The book does not need it: held names' 60-session correlation is 0.18 against 0.23 for the universe, because forecast / vol with caps already spreads them.
- **HAN1b fails, and reverses.** 825 names, about 297 valid a day, folds October 2020 to December 2025. Autoencoder volatility against PCA, by setting:

| Training | K 64 | K 128 | Hoyer (paper 0.497) |
|---|---|---|---|
| Batch 64 | +12.8% | +12.8% | 0.18 |
| Batch 256 | +17.0% | +15.9% | 0.35 |
| Full batch, 50 epochs | +19.1% | +18.8% | 0.47 |
| Full batch, 100 epochs | +20.4% | +19.2% | 0.45 |

  Higher in every fold of all eight settings, Sharpe lower by 0.24 to 0.96. The paper's equity panel: 10-19% lower, on survivors. Ours is point-in-time. Six of the eight were added after the first run, because the paper does not state a batch size; none changes the reading.
- **HAN1c.** Not applied. But the control still teaches: PCA with rising noise falls from 13.4 to 5.9 volatility while its return moves toward zero, and random baskets sit at 5.6, under every model. So lower volatility alone says nothing; the paper needs this control and has none. Its SAE earns less than its PCA (13.9 against 18.1).
- **Naive contrarian on NSE.** Direction checked: next-day rank IC -0.007 close to close, t -2.3, and -0.023 open to open, t -7.3. Sharpe -0.83 close to close, +0.96 open to open, before costs. Turnover is 78% a day, so at 25 bp a side both end near -16. Ledoit-Wolf on its legs: 12.2 volatility against 13.1.

**Assumptions.**

| Assumption | For Centurion |
|---|---|
| Long and short legs | fails: no short delivery; the long leg alone is also negative |
| Daily rebalance, 70-80% turnover | fails: 25 bp a side makes net Sharpe about -15 |
| Entry at the signal's close | fails: we fill next open |
| Pre-cost, no capacity | fails: costs are charged, and decide this |
| Static or ex-post universe | fails in the paper; ours is point-in-time |
| 24/7, 433 pairs, mostly weak common factors | fails: NSE, about 300 names, strong market factor |
| Stationary factors across folds | unknown; here more stable than the paper's (matched cosine 0.30, random 0.09) |
| Ledoit-Wolf on 60 sessions, N 30 | holds: positive definite, condition at most 63 |

**Audit.**

| Idea | Present? | Location | Risk if missing |
|---|---|---|---|
| Missing returns masked, not zeroed | yes | `signals.py:58`, NaN stays NaN | none |
| Average ranks on the observed set | yes | `signals.py:161`, `:162` | none |
| Next-open entry | yes | `engine.py:106` | none |
| Point-in-time universe | yes | `universe_mask` | none |
| Paired stationary bootstrap | yes | `diversifier_gate.py:113` | none |
| Selection strictly before test | yes | pre-registration, WF, sealed 2026 | none |
| False discovery control | partial | `aronson_validator.py:233`, legacy; the engine uses DSR and PBO | low |
| Covariance in weights | no | `portfolio.py:206`, forecast / vol | none: tested, lowers Sharpe |
| Latent factor model | no | none | none: reverses on NSE |
| Solver and seed noise | not applicable | no PCA or random fit in the engine | none |

**Where the method fails.** Costs first: a daily quintile turns over 70-80% a day. Then shorts, then the same-close entry. Its volatility edge reversed on the one market we can test. Torch is not in the core install, for good reason.

**Flagged.** Eq. 3 divides by N_t and calls it van der Waerden, whose form is r / (N + 1); the clip makes that harmless, and winsorising before ranking only adds ties. Batch size is unstated: 64, 256 and full batch tried, and full batch comes closest on Hoyer. Reconstruction MSE 1.043-1.047 (Sect. 4.3) is above the 0.96 that predicting zero scores on eq. 3's scale, if the crypto scale matches NSE's, and the authors say loss was still falling; ours reached 0.83 with the Hoyer near theirs. Without their code I cannot say why. Table 2's means and the equity reductions recompute correctly.

**Rules, set before any test.** All on data to 2025, 2026 stays sealed.

- **HAN1a, covariance in the book (Sect. 4.4).** The paper's control: Ledoit-Wolf minimum variance on 60 sessions. Two variants on the names the book picks, fixed settings: pure, and half and half with today's weights. Taken to January's queue only if Sharpe rises 0.05 or more over 2008-25, is not lower in 2008-12, and is higher in 2013-19, 2020-25 and at twice the costs.
- **HAN1b, replication on NSE (Sect. 4.5).** Tied ReLU autoencoder, K 64 and K 128, on the point-in-time universe, five expanding folds over the last 40% of 2013-25, five seeds. Replicated if long-short volatility is below matched PCA in 5 of 5 folds and 10% or more lower on average, the paper's equity range.
- **HAN1c, what the volatility is (not in the paper).** Controls the paper lacks: random baskets, and PCA's signal degraded with noise. If either matches the autoencoder's volatility at the same or better return, the reduction is a weaker signal, not diversification.

## 5bh. ESG1: Dispoto, Restelli and Ventre, "Eliciting ESG Preferences for Reinforcement Learning-Based Portfolio Optimization" (arXiv 2609.02677, 1 Nov 2026; your ask): does not fit, no code

**What it is.** An 8-page q-fin.PM paper. PPO conditioned on a preference vector, reward = lambda-weighted differential Sharpe plus ESG scores over equal weight (eq. 1-4). A pairwise probit Gaussian process with qEUBO then picks the vector from 28 comparisons (Sect. 4.3, 5.2). The comparisons come from LLM personas. Ten survivor US stocks, 2009 to mid-2016, three synthetic ESG agencies, frictionless. Table 3: Europe persona gains 17 / 32 / 58% in agency scores for a Sharpe drop of 0.023. No code link.

**Why nothing is taken.** The core method is reinforcement learning, closed by your decision of 11 Oct, so PPO and Table 3 were not run. Its data is synthetic, and Centurion has no ESG mandate, no ratings and no ESG code. Nothing was tested on NSE, because the paper offers nothing to test. The gates are thresholds, never a weighted sum, so linear scalarization's blind spot does not arise, and the owner states objectives outright, so there is no one to elicit from.

**Assumptions.**

| Assumption | For Centurion |
|---|---|
| Markov decision process, daily rebalance | fails: rules, 5-session rebalance, no RL |
| Frictionless | fails: costs decide most of our results |
| Ten large caps picked in hindsight | fails: point-in-time universe |
| ESG ratings exist per name per day | fails: none held; the paper's are simulated |
| Linear scalarization spans the frontier | not applicable: no scalar objective |
| A portfolio manager to elicit from | fails: one owner, explicit targets |
| LLM stands in for a human | unknown, and the paper shows it sensitive to wording |

**Audit.**

| Idea | Present? | Location | Risk if missing |
|---|---|---|---|
| Disjoint chronological stages | yes | backtest, WF, OOS, paper, sealed 2026 | none |
| Selection on one metric, not a blend | yes | `walk_forward.py:168`; gates are thresholds | none |
| Live drift check | yes | `capital_ladder.py:296`, CUSUM | none: differential Sharpe adds nothing |
| Identifiability of weights | yes | tracker: PBO note on near-ties | none |
| LLM kept out of decisions | yes | RAG layer disabled, `integrated_scorer.py:1126`; only the legacy `auto_executor` calls it, no workflow does | low |
| ESG objective | no | none | none: no mandate |
| RL policy | no, by your decision | `services/rl_bot` untouched | none |

**Replication, scratch.** Eq. 2 against a numerical derivative of the moving Sharpe: median relative error 4e-11, maximum 3e-7. Tables 2 and 3 cannot be rebuilt: the data process is not given.

**Where the method fails.** Costs ignored on a daily rebalance. Survivors. Agency numbers that only exist in the simulator. Table 3's Sharpe drops sit inside their own spread (mean over s.d. 0.15 to 0.94), and the 20 episodes share one 18-month window, where a single Sharpe's standard error is about 0.8. So the persona ordering shows nothing about Sharpe.

**Flagged.** Table 2's disagreement is put down to pillar weights, which differ by at most 0.15. Independent pillars give correlations of 0.94-0.97. To reach -0.43, 0.57 and -0.14 with these weights alone, assuming linear scores of equal spread, pillars must be 5-10 times as dispersed as any agency score, with S and G correlated -0.91 and E and G -0.82: possible, but built in, not a product of weights. As I recall, Berg et al., which the paper cites, report positive correlations. Table 2 averages 3,780 days, yet 2009 to mid-2016 is about 1,890. Sect. 2.1 sets horizon H and sums to T. Eq. 1 adds a differential Sharpe to rating differences with no stated scaling, so lambda is not comparable across terms, and lambda-1 and lambda-3 near zero for every persona may be that, not preference.

**One thing worth keeping, no code.** Sect. 5.3's prompt sensitivity is a reason to keep any LLM out of the decision path, as now.

## 5bi. REG1: Supriadi and Siswoyo, "Market Regime-Based Multi-Asset Portfolio Optimization Using Nested Clustered Optimization" (arXiv 2610.27751, 1 Nov 2026; your ask)

**Result: nothing adopted, no code.** REG1a replicated, and showed the paper's NCO never optimised. REG1b: noise gives as many switches. REG1c and REG1d failed. One protocol line added. 2026 stayed sealed.

**What it is.** A 12-page q-fin.PM paper on 5 Indonesian assets, monthly, 56 months out of sample. A rolling 3-state Gaussian HMM (eq. 2-4) filters a 24-month window; covariance on the filtered rows, shrunk toward the full window (eq. 5-6); average-linkage clustering (eq. 7); minimum variance inside and across clusters (eq. 8-10); rebalance on a regime change or drift over 0.05. Claims: static NCO 29.97% / MaxDD 2.18% / Calmar 2.65, regime NCO 1.60% / 9.99%, Sharpe gap z 2.68, p 0.0073. Code: github.com/obo-cloud/nco-regime-taa, MIT, read as data, never run.

**Replication, on its own dataset, rebuilt natively.**

| | Paper | Ours, repo's solver | Ours, solved properly |
|---|---|---|---|
| Static NCO, CR / MaxDD / turnover | 29.97 / 2.18 / 5.41 | 29.97 / 2.18 / 5.41 | 22.69 / 2.53 / 4.09 |
| Minimum variance | 18.45 / 4.75 / 1.36 | 18.45 / 4.75 / 1.36 | 16.77 / 3.06 / 2.49 |
| Regime NCO, seed 42 | 1.60 / 9.99 / 24.58 | 1.61 / 9.99 / 24.62 | 1.41 / 7.81 / 20.89 |
| 1/N | 76.70 / 15.27 | 76.70 / 15.27 | same |
| Eq. 13, static, Ann / MaxDD | 7.01 / 4.68 | 7.01 / 4.69 | 6.27 / 4.69 |

**The paper's NCO never optimises.** SLSQP at default tolerance on raw monthly variances stops at once: the cluster step takes 1.0 iterations on average and moves 0.0 from equal, 45% above the true variance; minimum variance stops 82% above. So "NCO" is equal weight across clusters, and "minimum variance" has more volatility than the real minimum (4.70 against 3.62). Solved properly, static NCO returns 4.48% a year against a 5.75% risk-free rate, so its excess Sharpe is negative, not +0.006. Static still beats regime.

**Other results.**

- **REG1b, met: labels not shown to carry information.** Noise through the same pipeline, 200 draws: 33 switches, 5th-95th 26-40. The paper's 26 is at the 6th percentile, so slightly stickier than noise, not distinguishable at 5%.
- **One seed is one draw.** Repo solver, 20 seeds: regime NCO -1.8% to 28.2%, median 14.3%; the paper's 1.60 sits near the bottom. All 20 are below static, so the direction survives, but the gap is about half the reported one, and p 0.0073 ignores this.
- **REG1c fails, narrowly at one end.** Fixed settings, live-like:

| `confirm_days` | Sharpe | 2008-12 | 2013-19 | 2020-25 | 2x costs | Switches |
|---|---|---|---|---|---|---|
| 1 | 0.768 | -0.047 | 0.592 | 1.535 | 0.512 | 284 |
| 3 (today) | 0.704 | -0.009 | 0.541 | 1.381 | 0.487 | 109 |
| 5 | 0.649 | -0.151 | 0.471 | 1.397 | 0.439 | 77 |
| 10 | 0.613 | -0.234 | 0.634 | 1.168 | 0.409 | 55 |

  1 misses only on 2008-12, by 0.038, so it stays as it is. It is a lead, not a result: in-sample, three variants looked at, and it needs its own rule and walk-forward. It runs against the paper's advice: here, more persistence does worse. 49% of switches reverse within 21 sessions, but the gate's exits are 2.9% of traded value and 2.8% of costs, so whipsaw is cheap.
- **Turnover by trigger.** Rebalance 6.35x a year, 42% of costs; rank exits 3.5x, 24%; metals sleeve 2.7x, 28%; stops 0.5x, 3%; regime 0.4x, 3%. Costs 2.77% a year. No buffer test: R7 already found a wider buffer takes 5% off turnover with CAGR flat.
- **REG1d fails.** 2010-25, monthly, 15 bp: NCO 7.25% / Sharpe 0.459 against 1/N 9.72% / 0.679, gap -0.22, p 0.89; without liquid 0.575 against 0.678, p 0.86. 1,235 solves, no failures. No stream-allocator study.
- **New: passive mix against the book.** Equal NIFTY, gold, gilt, before costs: Sharpe 0.590 over 2008-25 against the walk-forward book's 0.569, monthly, with 10.9% against 13.0%. The book leads 2010-17, 0.835 against 0.346; the mix leads 2018-25, 0.992 against 0.763. Correlation 0.38. The book earns more, not better per unit of risk. Tax not included in either.

**Assumptions.**

| Assumption | For Centurion |
|---|---|
| Monthly, 5 assets | fails: daily, about 300 names; holds for sleeves only |
| HMM stationary in 24 rows | fails: 32 parameters on 24 rows, shown as noise |
| Gaussian emissions | fails: fat tails; ours is rule-based |
| Bond series is a return | fails in the paper: it is a yield change; ours is a total-return index |
| 10 bp flat, no impact | fails: costs 2.77% a year here |
| Long-only, fully invested | partly: long-only; the gate holds cash |
| Tk >= N + 5 for a covariance | not applicable: no regime covariance |

**Audit.**

| Idea | Present? | Location | Risk if missing |
|---|---|---|---|
| Persistence before a regime switch | yes | `regime.py`, `config.py:183`, 3 days | none: more makes it worse |
| Regime or drift rebalance split | yes | `engine.py:325`, `:376`, 5 sessions plus buffer | none |
| Bond series a true total return | yes | `bond_stream.py:129`, NSE TR indices | none |
| Causal regime, no refit | yes | `regime.py`, rules, no fit | none: no label switching |
| No numerical optimiser in the live path | yes | none in `nse_engine`, `kite_connect/trading`, `services/risk` | none |
| Rolling HMM | legacy only | `regime_hmm.py:169`, full covariance, seed 42; `carver_pipeline.py:911` | low: not the engine |
| Sharpe difference test | yes | `diversifier_gate.py:124`, paired bootstrap, safer than eq. 12 | none |
| Solver and seed checks in research | no, until now | `research_protocol.md` | med: this paper's error, unnoticed |

**Where the method fails.** The headline is a stopped solver on a yield mislabelled as a return. Monthly HMMs on 24 rows hold more parameters than data rows. Costs of 10 bp on Bitcoin and sovereign paper are low. Regime change is the paper's own finding.

**Flagged.** Eq. 12 prints 2(1 - rho^2); Memmel's term, and the repo's code, is 2(1 - rho); assumed the latter. The abstract's "turnover erodes" cannot be right: its own 29.5 bp a year explains little of a 544 bp gap, and the repo's table shows 5.2 points at zero cost; Sect. VI.C says whipsaw. Code differs from the paper: inverse variance inside clusters, not eq. 8; a 60% cap and M at most 3 that the paper omits; 3-month features; diagonal covariance. Eq. 11 charges full change, the code half. 1/N is shown with zero turnover because the code compares the target with itself; charged, it is 76.45% and 2.56%. "Drift" compares against a moved target, not market drift. Headline bond series is a yield change; on eq. 13 the regime gap shrinks. Sect. V.C invokes the DSR, no figure given. Table V shows z 0.612 for HRP though its Sharpe is lower than regime NCO's.

**Rules, set before any test.** All on data to 2025, 2026 stays sealed. Published code (MIT): data read, code never run, everything rebuilt natively.

- **REG1a, replication.** From the paper's own dataset: 1/N, minimum variance, static NCO and regime NCO. Within 1 point of cumulative return, 0.5 of MaxDD and 1 of monthly turnover counts as reproduced.
- **REG1b, are the switches information?** The same rolling 3-state HMM on pure noise of the same shape (24 rows, 4 features, 56 steps). If noise gives 20 or more switches against the paper's 26, its regime labels are not shown to carry information.
- **REG1c, our gate.** Turnover and cost split by trigger (paper's Table VI). Then `confirm_days` 1, 5 and 10 against today's 3, fixed settings. Taken to January's queue only if Sharpe rises 0.05 or more over 2008-25, is not lower in 2008-12, and is higher in 2013-19, 2020-25 and at twice the costs.
- **REG1d, NCO on Indian assets.** Monthly, W 24, NIFTY TRI, gold, gilt, liquid: static NCO against 1/N, inverse volatility, minimum variance and passive 60/40. Worth a stream-allocator study only if Sharpe beats 1/N by 0.10 or more, one-sided p at most 0.10 on a paired bootstrap, and both halves agree.

## 5bj. DYN1: Cordoni, Sancetta and Zhao, "Dynamic Strategy Selection" (arXiv 2307.04754v3, 1 Nov 2026; your ask)

**Result: nothing adopted, no code.** DYN1a replicated; DYN1b failed. The method is reinforcement learning, closed by your decision, and the replication backs that decision at our sample size. 2026 stayed sealed.

**What it is.** An 83-page paper with supplement. Monthly choice among 4 long-short US factors, 1963-2025, with state = trends, market cycle, macro, plus the strategy held. Reward is utility of return less a switching penalty (eq. 4), larger than real costs on purpose, as a regulariser. Fitted Q-iteration: ridge on quadratic features, GCV, Q capped, actions simulated (Sect. 3, Alg. S.1). Greedy is one iteration, gamma 0. Headline: Trend + Cycle, 120 bp, eta 3: Sharpe 0.715 against Greedy 0.511 and momentum 0.445. Theorem 1: the policy's value converges to the optimum. No code link.

**Replication, scratch, on the paper's own process; 60 replications, 4 action draws, 40 iterations.**

| | Paper | Ours |
|---|---|---|
| Oracle policy, eta bound | strategy 1 or 2, eta <= 164 | same; holds at 164, fails at 165 |
| Oracle / strategy 1, % a step | 0.80 / 0.46 | 0.80 / 0.45 |
| RL, signals, n 240 / 600 / 1,200 | 0.56 / 0.70 / 0.78 | 0.55 / 0.73 / 0.77 |
| Greedy, same | 0.46 / 0.47 / 0.47 | 0.49 / 0.53 / 0.51 |
| RL, 8 noise, n 240 / 1,200 | 0.43 / 0.74 | 0.31 / 0.78 |

Paper values are read off Figure S.5. Signals-only sits within 0.06; the noise rows run up to 0.12 lower at n 240: the paper's noise basis is unstated, linear assumed (quadratic was worse). Fixed: **n 216, Centurion's months: RL 0.465, Greedy 0.451, oracle 0.753**, standard error about 0.03. With a 4-state process and clean signals, 216 rows capture about a tenth of the oracle's edge and nothing over Greedy. That matches why you closed RL.

**Greedy loses by construction in the simulation.** The one-period gain from switching is at most 80 bp and the penalty is 120 bp, so a myopic rule cannot switch: 0.8% of months at n 1,200, and it tracks strategy 1. "RL beats Greedy" there restates the design.

**Other results.**

- **The headline gap leans on the ridge choice.** Table S.1, RL less Greedy, monthly reward: under GCV 0.148 (eta 0) and 0.082 (eta 3); at fixed degrees of freedom about 0.04. GCV gives Greedy its lowest of the four results in 7 of 9 eta 0 cells. RL still wins 41 of 54 fixed cells, so it survives, at a third to a half the size. At df 8, Trend + Cycle, 120 bp, eta 0, Greedy beats RL 0.701 to 0.636.
- **Multiple testing.** Table 1 has 18 RL-Greedy comparisons, 8 starred at 10% or better, none adjusted, and the headline is the best cell. In Table 1 itself Greedy wins eta 3, Trend, 80 bp, Sharpe 0.684 to 0.564.
- **DYN1b fails.** 2009-25, walk-forward book, live-like, monthly:

| | CAGR | Sharpe | MaxDD | 2009-16 | 2017-25 |
|---|---|---|---|---|---|
| Book held | 15.92 | 0.822 | -17.8 | 0.686 | 0.948 |
| Equal book, gold, gilt | 11.85 | 0.817 | -6.7 | 0.514 | 1.132 |
| Rotate, 20 bp | 14.62 | 0.637 | -20.3 | 0.312 | 0.980 |
| Rotate, 80 bp | 15.26 | 0.672 | -19.9 | 0.327 | 1.031 |
| Rotate, 120 bp | 13.86 | 0.589 | -20.9 | 0.393 | 0.789 |

  p against the book 0.74-0.83. `timing_skill`: timing excess -8.0% a year, 0.45 switches a year. So REG1's finding, that the book and the passive mix take turns, cannot be caught from trailing trend. Next month's book-less-gold return against the trend gap: correlation 0.17, too weak net of a slow signal. Equal mix of the three matches the book on Sharpe here too, with a third of the drawdown (monthly, gross for the sleeves): that is a diversification fact, not a rotation one.

**Assumptions and theorem conditions.**

| Assumption | For Centurion |
|---|---|
| Raw state exogenous to the choice | holds: 0.013% of value traded (KOH1) |
| Strictly stationary, exponentially beta-mixing Markov state | fails: a 12-month mean is not first-order Markov; regimes shift (2008-12 Sharpe near 0, 2020-25 1.35) |
| Finite second moments of reward | holds |
| Gram matrix positive definite | holds with ridge |
| Zero approximation error (Theorem 1) | unknown, untestable |
| r_n = sqrt(K ln K ln n / n) -> 0 | fails in practice: K 15, n 216 gives r_n about 1.0; 0.64 at the paper's 629 |
| Bound's constant | 2 sqrt(4) / (0.25 x 0.01) = 1,600 at gamma 0.9, so the theorem says nothing at these sizes |
| Long-short factor legs, monthly | fails: no short delivery; only long sleeves |
| A switch costs less than the penalty | unknown: 60 bp assumed for a whole-book switch, before tax, not measured |
| 629 monthly rows | fails: 216 |

**Audit.**

| Idea | Present? | Location | Risk if missing |
|---|---|---|---|
| Hysteresis on exits | yes | `exit_rank`, `config.py:147`, 40 | none |
| Band around targets | yes | `no_trade_buffer`, `config.py:150`; R7 tested wider | none |
| Re-entry delay | yes | `stop_cooldown_days`, `config.py:155` | none |
| Regime and sleeve confirmation | yes | `config.py:183`, `:210`; more made it worse (REG1) | none |
| Penalty above real costs | partial, in effect | those four rules | low |
| Real switching cost in tests | yes | `costs.py`, D2 haircut | none |
| Timing skill test | yes | `validation/timing.py:95` | none |
| Learned selection among strategies | no, by your decision | none | none |
| Fitted Q on any Centurion data | no, by your decision | none | none: not run |

**Where the method fails.** Short data first. Regime shifts break the stationary Markov condition the theorem needs. Costs and tax on whole-book switches. The paper's factors are gross of their own trading, so "conservative" holds only against the penalty, not against reality; no shorts here. Each penalty, eta and state set is a trial.

**Flagged.** Sect. 4.1 starts out of sample in July 1973 after 120 complete months, yet data begins July 1963 and trends need 12 months, which points to mid-1974; assumed the shorter burn-in was meant, and 629 decisions fits July 1973. Lemma S.9 states m of order n^-1 ln n, and uses it again, where the proof needs n / ln n. The simulation's noise basis is unstated. "Actions" drawn per replication, 10 in the paper, 4 here.

**Rules, set before any test.** No learned policy touches the book: reinforcement learning is closed by your decision of 11 Oct, so fitted Q runs only on the paper's own synthetic process, as a replication. All on data to 2025, 2026 stays sealed.

- **DYN1a, replication (Sect. S.6).** The 16-state oracle and the Monte Carlo at 240, 600 and 1,200 rows, 0 to 8 noise variables, with a row at 216, the months Centurion has. Oracle within 1% of the exact Bellman solution; simulation read against Figure S.5.
- **DYN1b, fixed rule, no fitting: is relative performance predictable from trend?** Each month hold the best of walk-forward book, gold and gilt by 12-month mean return, switching only when the gap beats a penalty of 20, 80 or 120 bp a month (the paper's states and penalties; primary 120). Charged real costs of 60 bp a switch. Against the book held throughout and the equal mix. Worth a study only if Sharpe beats the book by 0.10 or more, one-sided p at most 0.10 on a paired bootstrap, both halves agree, and it passes `timing_skill` as a timing rule.

## 5bk. TDA1: Mazumder and Mazumder, "Depth-Efficient Quantum Topological Data Analysis" (arXiv 2607.09906v2; 1 Nov 2026; your ask)

**Result: nothing adopted, no code.** TDA1a replicated exactly; TDA1b failed
all three gates. Rules were written 1 Nov 2026, 18:00 IST, before any Indian
data was touched. The replication on the paper's own S&P file ran first.

**Fit.** The quantum stage does not apply: the paper claims no advantage at
these scales, and classical eigensolvers take ≤ 228 ms a window. Only the
classical β1 could matter, as a crash warning for the regime gate.

**Published code.** github.com/arulrhikm/Quantum-Market-Crash-TDA has no
licence file, so all rights are reserved. Its code was read, not run or
copied; only its S&P CSV was used.

**TDA1a, replication, native rebuild in scratch, tolerance exact.**

| | Paper | Ours |
|---|---|---|
| Table I, five windows, β0 / β1 | 284/15, 296/9, 327/6, 335/3, 338/3 | same |
| β1 range, mean | 0–22, 4.72 | 0–22, 4.72 |
| n1 range | 31–429 | 31–429 |
| Positive windows, train / test | 27 / 10 | 27 / 10 |
| AUC β1, train / test | 0.943 / 0.818 | 0.943 / 0.818 |
| AUC volatility, train / test | 0.905 / 0.652 | 0.905 / 0.652 |

**Flagged in the paper.**
- **The "lead" is a dating effect.** Sec. VII-A says β1 builds "more than a
  year before" Bear Stearns, but Fig. 2 plots each window at its mid-date.
  Dated at its last day, β1 first reached the paper's threshold of 7 on
  8 Apr 2008, after Bear Stearns, and 3 on 11 Sep 2007. Its peak of 22 came
  in April 2010, 13 months after the low: the crash stays in the window.
- **The test set misses the crisis.** The split is by window start, so
  "2007–2010" means decisions from 29 Dec 2008 to 29 Dec 2010. Lehman's
  decisions sit in the training set.
- **Time order does no work.** β1 with each window's days shuffled scores test
  AUC 0.96, and window kurtosis 0.965, against 0.818 for real β1 (10 clustered
  positives, so read neither as skill). Shuffled and real β1 have Spearman
  0.84. Standardising inside the window at a fixed ε makes one extreme day
  pull every other point closer, so β1 largely tracks fat tails.
- **Censoring.** The last 8 of 64 test windows have under 90 days ahead and
  are labelled 0, not dropped; none is positive.
- **Takens.** The theorem needs a smooth deterministic system. Daily returns
  are neither, and delay 13 and ε are set on the same sample.

**TDA1b result, 550 readings, 83 positive (15.1%).**

| Gate | Needed | Result | |
|---|---|---|---|
| 1. AUC | ≥ 0.60, lower bound > 0.50 | 0.496 [0.411, 0.578] | Fail |
| 2. Every period above 0.50 | 6 of 6 | 2 of 6 | Fail |
| 3. AUC less volatility's | Lower bound > 0 | −0.174 [−0.301, −0.027] | Fail |

| Period | Positive | β1 | Volatility | VIX |
|---|---|---|---|---|
| 2008–09 | 28 | 0.646 | 0.622 | n/a, starts Mar 2008 |
| 2010–12 | 20 | 0.441 | 0.529 | 0.422 |
| 2013–16 | 14 | 0.500 | 0.421 | 0.460 |
| 2017–19 | 8 | 0.464 | 0.289 | 0.320 |
| 2020 | 7 | 0.031 | 0.257 | 0.217 |
| 2021–25 | 6 | 0.453 | 0.725 | 0.632 |

- **Pooled baselines:** volatility 0.670, VIX 0.629, kurtosis 0.513, shuffled
  β1 0.551.
- **β1 ≥ 7:** fires on 7.8% of readings with precision 2.3%, below the 15.1%
  base rate. Apart from two readings in 2010 it fires only from May 2020 to
  January 2022, after the crash. β1 was 0 on 6 Jan and 2 Mar 2020.
- **Yearly refit, 2011–25:** the threshold falls to 0 and it fires always.
- **Sensitivity:** AUC 0.44–0.58 across delays 5 and 21, dimension 3, ε 0.25
  and 0.40, and 250-session windows. None reaches 0.60.
- **Not a win for volatility.** Its pooled 0.670 comes from differences
  between periods; it is below 0.50 in three of six. The gate's VIX leg reads
  the present and is not touched by this.

**Where it fails in production.** Regime change first: it fails on the shock
the paper's own OOS test names. A 500-day window also holds a crash for two
years, so it stays on through the recovery.

**TDA1b, one configuration, the paper's numbers, no tuning.**
- **Feature:** NIFTY 50 daily log returns, last 500 sessions, delay 13,
  dimension 4, each coordinate standardised in the window, Vietoris–Rips at
  ε = 0.32, β1. One reading every 8 sessions, dated at the window's **last**
  session. 2008-01-01 to 2025-12-31; 2026 stays sealed.
- **Label:** a close ≥ 10% below the reading's close within the next 63
  sessions (A: the paper's 90 calendar days). Readings without 63 sessions
  ahead are dropped, not labelled 0.
- **Baselines:** 30-session realised volatility and India VIX.
- **Pass, all three:**
  1. AUC ≥ 0.60, with the block-bootstrap 95% lower bound above 0.50.
  2. AUC above 0.50 in every period holding at least 3 positive readings:
     2008–09, 2010–12, 2013–16, 2017–19, 2020, 2021–25.
  3. AUC minus the volatility baseline's above 0, lower bound above 0.
- **Bootstrap:** circular blocks of 126 sessions, 2,000 draws, seed 0.
- **Reported only, nothing chosen from them:** the paper's threshold β1 ≥ 7;
  a threshold refitted each year on past readings; delays 5 and 21, dimension
  3, ε 0.25 and 0.40, 250-session windows; window kurtosis; β1 with the
  window's days shuffled.
- **On a pass:** a proposal for a January slot, with its own overlay rule.
  The core family stays closed until then. **On a fail:** the record stays
  here, and no code does.

## 5bl. PIM1: Hafsi, Lillo and Szymanski, "Calibration of Passive Market Impact with a Queue-Sensitive Propagator Model" (arXiv 2610.38442v1; 1 Nov 2026; your ask)

**Result: nothing adopted, no code.** The theory checks out on synthetic data.
The empirical work needs order-book messages we do not have, and state-dependent
impact misses its bar at Rs 30 lakh: −0.04 points net, ceiling 0.23.

**What it is.** A 41-page microstructure paper on CME E-mini S&P futures,
2025, with other index and Treasury futures as checks. Theorem 2.1 splits the
price into the Hawkes propagator at average sensitivity plus a queue-state
correction. With affine sensitivity, Corollary 4.1 gives an observable
queue-weighted imbalance H and an anticipation term J that is left out.
Claims: a negative queue slope, out-of-sample R² 0.458 to 0.469 (0.603 with
the full book state), impact 2.3× larger thin against thick, passive impact 22×
smaller than aggressive. No code is published.

**It explains, it does not forecast.** Every regression sets a price change
against flow over the same interval. Its R² is out of sample in parameters, not
in time, so it is no trading signal. The affine gain itself is small: 0.002 in
ES 2023, 0.006 in 2024.

**Replication, scratch, synthetic: the paper's data is not public.**

| Check | Paper | Ours |
|---|---|---|
| ∫ψ, Lemma A.2, from the roots | η/(1−η) = 4 | 4.000000 |
| Weighted-mid mechanical move, Sec. 4.1 | formula | exact, to 10 places |
| Table 5, immediate response, ×10⁻³ | 9.3 / 5.9 / 4.0 / 6.7 | 9.32 / 6.02 / 4.05 / 6.74 |
| κ1 by eq. 18 with J omitted (simulated, truth −1.62×10⁻⁴) | small bias at long δ | −1.624×10⁻⁴, 0.2% off |
| J-control coefficient, κ1µ/(−cλ) | −6.48×10⁻⁴ | −6.48×10⁻⁴ |
| Prop. C.1: price drift against imbalance | nil with J | 15× smaller with J (0.0020 / −0.0304), not exactly nil |
| Remark 2.3, g(ρδ) ≤ 1/(ρδ) | holds | holds at δ = 1, 10, 60, 212 |

**Flagged in the paper.**
- **δ\*:** Sec. 4.1 and Fig. 4 give 116 s for ES 2025; Tables 1 and 6 and
  Remark 2.3 use 212 s. Table 6 shows it matters little.
- **Table 1 s.e.:** −8.1×10⁻⁶ with s.e. 1.7×10⁻⁶ is t = −4.8, but Table 6 gives
  t = −7.8 at the same δ. Reading: the pooled coefficient is printed with the
  day-by-day mean's s.e. (assumption; the half-year means are −1.8 and
  −0.95×10⁻⁵).
- **Table 1 against Table 2:** log R² 0.444 and 0.446.
- **Queue scale:** κ is fitted on the best queue (mean 20, 99th percentile 85
  AES), but the stationary shift evaluates it at 101 AES, the three-level
  queue. That extrapolates beyond the fit and mixes two queues.
- **"22× smaller" leans on the model:** by the authors' own numbers, the
  mean-reversion rate alone moves passive impact 50-fold (1.5×10⁻² to 3.0×10⁻⁴),
  and there is no fill or adverse-selection model.
- **Sec. 3.1:** "use the intraday window 10:00 to 16:00 ... while the price
  regressions ... use 10:00 to 16:00" contrasts nothing. Probably a stale edit.

**Assumptions for us.**

| Assumption | Status |
|---|---|
| Message-level order book, queues reconstructed | fails: daily bars; minute data is not bought and holds no depth |
| Large tick: spread one tick on 97% of trades | fails for most NSE names; high-priced stocks are many ticks wide |
| Continuous trading against a best queue | fails for engine orders: AMO LIMIT for the open; GTT stops only, 4.6% of value |
| Hawkes stable, η < 1, on 15-minute bins | unknown here; even ES breaks it in the last 15 minutes |
| Queue mean reversion, mixing condition (13) | unknown; the authors treat it as a local diagnostic |
| Stationary, regular sessions only | fails for us: stress days are where costs matter |
| Book large enough for impact to matter | fails today: median 0.011% of daily value at Rs 30 lakh |

**Measured on the deployed book, 2008–25, 7,331 trades, modelled impact.**

| | Rs 3 lakh | Rs 30 lakh |
|---|---|---|
| Median participation | 0.0011% | 0.0113% |
| Impact, points a year | 0.186 | 0.587 |
| Table 5 multiples, market liquidity | −0.012 | −0.037 |
| Table 5 multiples, each stock's liquidity | −0.012 | −0.039 |
| Ceiling, every trade thin (× 1.39) | 0.072 | 0.229 |

The thin third of sessions carries 32% of impact, so the book does not trade
into thin markets. If impact is linear below 1% ([16], [24]), Rs 30 lakh costs
0.29 points, not 0.59: the square-root law is the cautious choice at this size.
At Rs 3 crore impact is 1.86 points, so this reopens before that size, with
real fills from the canary.

**Where it fails in production.** Liquidity first: the method needs data we do
not hold and a queue our orders do not meet. Regime change next: the paper
excludes halts, FOMC and shortened sessions, and says its results do not extend
to stressed conditions.

**Rule.** The paper finds immediate impact 2.3 times larger in thin books
(Table 5). Apply that multiple to all modelled impact in the deployed book,
2008–25, leaving the 3 bp spread floor alone: the worst case, since only thin
sessions should carry it. State-dependent impact is worth building only if the
extra drag reaches 0.25 points of CAGR at Rs 30 lakh, the bar KOH1a used.

**Result of that rule:** 0.76 points, so it clears the bar as written.

**Correction, written after that result and before the next one.** The rule
was too crude. 2.3 compares thin with thick books; the cost model is fitted to
the average, and Table 5 gives thin / average / thick as 1.39 / 0.88 / 0.60.
The test that decides is therefore: apply those three multiples by each
session's liquidity tercile within its year, for the sessions the book really
traded, against the same 0.25-point bar. Liquidity is the universe's median
traded value over its own 20-session median, and separately each stock's own.
All-thin at 1.39 is the ceiling.

## 5bm. TRA1: Naja et al., "The Control Layer Behind Rule-Following Agents" (arXiv 2610.26324v2; 1 Nov 2026; your ask)

**Result: one read-only report built, no gate, no book output moved.** The core
passes every conduct test. The battery found one real gap: the go-live checks
say nothing about which controls the paper record has actually run. Rules were
written 1 Nov 2026, before the battery ran on our records.

**Fit.** Crypto LLM agents, so the performance claims do not transfer. The audit
battery reads only a trade log, so it can run on ours. No new trial: this
measures conduct and reads no 2026 outcome beyond the paper book's own log.

**What it is.** A 47-page vendor report on its own four agents: 2,627 closed
positions, 12 Feb to 27 Aug 2026, crypto perpetuals, 82% simulated at a 0.09%
fee with no funding. It names a published snapshot but gives no link, so
trade-level replication is not possible.

**Replication, from the printed tables, tolerance 0.01 points (rounded inputs).**

| Check | Paper | Ours |
|---|---|---|
| Expectancy, Agents 2 / 4 / 3 / 1 | 0.836 / 0.062 / 0.028 / −0.043 | 0.835 / 0.063 / 0.025 / −0.045 |
| Margin over break-even, points | 9.0 / 2.7 / 1.0 / −2.1 | 9.0 / 2.7 / 0.9 / −2.2 |
| Pooled hit rate, expectancy | 45.3%, 0.050 | 45.25%, 0.0498 |
| Table 4.3a counterfactuals | as printed | within 0.004 |
| Holm, nine tests | 5 survive | 5 survive |

**Flagged in the paper.**
- **Its main finding is an identity.** Expectancy = (hit rate − break-even hit
  rate) × (average win + average loss), so the margin and expectancy always
  share a sign. Ranking by one and finding it ranks the other is close to
  arithmetic.
- **Four agents give no power.** ρ = 1 has a 4.2% chance in one draw, and the
  best-worst gap itself is not significant (p = 0.08). Yet the conclusion answers
  "what separated the agents' results?".
- **No expectancy interval.** Sec. 1.5 promises day-block intervals; I found
  none printed for expectancy.
- **"Net positive after fees" leans on exclusions.** Funding is left out on
  perpetuals held for days, position size is missing on 89% of rows, and without
  the outage trades Agent-3 turns negative.
- **Wording:** "closed paper positions" (Secs. 1.5, 9) against 483 real orders;
  Test 3 says "identical ... for every agent", then that Agent-3 tightens.

**Assumptions for us.**

| Assumption | Status |
|---|---|
| Trade log with the opening stop on every row | holds in paper; backtest stop rebuilt from ATR at entry (A) |
| Declared limits to test against | holds, in config |
| Independent trades | fails: 20 names at once, so month blocks are used |
| Enough regimes in the record | holds for 2008–25; fails for the paper record |
| Fixed % stops | fails: the core's stop scales with ATR by design |
| Closed trades represent the book | fails in a young book: winners are still open |

**Battery results.**

| Test | Backtest 2008–25, 2,161 trades | |
|---|---|---|
| 1. Loss containment | 99.9% inside, p95 0.76 R, 4 of 1,188 beyond 1 R | Pass |
| 2. Stop stable | 6 ATR fixed | By construction |
| 3. Risk after losses | 6 ATR always; in %, 21.1 / 21.7 / 21.1 after a win / 1 loss / 2 losses; first lot 3.4 / 3.1 / 3.2% of equity | Pass in ATR and size; 0.6 points wider in % after one loss, from volatility |
| 4. Post-loss trade share | 56–58% against a 57.0% loss rate, depending on how same-day exits are ordered | Pass |
| 5. Limits | 20 names at most (cap 30); weight peaks 9.08% on drift, cap 8% at target | Pass |
| 6. Asset dependence | best name 2.3% of positive P&L; ex-best-10 still 70% of the total | Pass |
| 7. Economics | hit 43.0% [40.9, 45.1], break-even 34.6%, mean trade +1.64% [0.69, 2.62] | Margin +8.4 points |
| 8. Regime calendar | all six cells covered, 277–1,483 days | Thin nowhere |

Margin by period: 2008–12 +3.7, 2013–19 +7.5, 2020–25 +11.2 points. All of this
repeats or passes the scorecard, so under the rule it leaves no code.

**Control coverage, the gap (test 9).** Counts from the exports of 31 Oct.

| Book | Sessions | Rebalance | Rank exit | Stop | Sleeve exit | Regime exit | Drawdown rule |
|---|---|---|---|---|---|---|---|
| Deployed | 31 | 4 | 7 | 3 | 1 | 0 | 0 |
| Candidate | 23 | 6 | 4 | 1 | 0 | 0 | 0 |
| E4 | 19 | 3 | 1 | 2 | 0 | 0 | 0 |

G4 compares the book with a backtest of the same days, so it can pass on 60 quiet
sessions. Nothing said that the regime exit and the drawdown rule had never run.
**Built:** `capital_ladder.control_coverage`, returned as `coverage` by
`live_session.go_live_status` and shown on Trade Center's go-live panel. `ready`
cannot move: it is information, and gates are the owner's. An unread record shows
nothing rather than every control at 0. Exits only: the paper tables hold no
per-session regime label.

**Not defects, noted.**
- `open_positions` counts lots, not names: 29 rows were 19 names, with 19–20
  stops armed. No limit reads it.
- Closed-trade hit rate in a young book runs low, because stops and rank exits
  close first and winners stay open. Do not read it against the backtest's 43%
  until the book has turned over.

**Where it fails in production.** Conduct is not edge, as the paper says. And
coverage counts that a path ran, not that it ran well.

**Battery, on the deployed backtest 2008–25 and on the paper book, the paper's
thresholds unless marked (A).**
1. **Loss containment:** ≥ 95% of gross loss inside the opening stop, 95th
   percentile ≤ 1.25 R.
2. **Stop stable across volatility:** in ATR multiples, because the core's stop
   scales with volatility by design (A); the paper's % test would fail it by
   construction.
3. **No risk escalation:** median stop and size no wider after one or two losses
   than after a win.
4. **No frequency escalation:** post-loss entry share within 2 points of the loss
   rate.
5. **Limits:** core names ≤ `max_positions`, weight ≤ `max_weight`, every session.
6. **Asset dependence:** largest name's share of positive P&L; result without the
   best name, and without the best 10.
7. **Economics:** hit rate with a Wilson interval, break-even hit rate, margin,
   and a block-bootstrap interval on mean trade return, by month (A).
8. **Regime calendar:** days and trades in each direction × volatility cell,
   labels from NIFTY, expanding terciles, prior day.
9. **Paper book only:** stops armed on every holding, every session.

**Adoption rule.** A permanent check is built only for a test that finds a real
defect in our records, or a gap in go-live evidence that nothing else reports.
A test that merely repeats the scorecard, or passes by construction, leaves no
code.

## 5bn. VOL1 and CAD1: Lis, Ślepaczuk and Sakowski, "Information Flow and Market Risk" (arXiv 2610.39636v1) and Wu and Huang, "CADENCE" (arXiv 2610.38084v1) (1 Nov 2026; your ask)

**Result: nothing adopted, no code.** VOL1b and VOL1c fail. CAD1 passes, so GT7's
window stands. Rules were written 1 Nov 2026, before any test on our data. 2026
stays sealed. No book or engine change; a pass would only have earned a proposal
for a January slot.

**What they are.**
- **Lis et al., 73 pages, a thesis chapter (pp. 214+).** ModernFinBERT scores on
  NYT news against HAR, VIX and AAII for 11 surviving US large caps, 2000–22.
  Text improves on HAR in 2 of 33 cells, and neither survives the longer HAC.
  Naive has the lowest mean QLIKE at every horizon. No public code or data link.
- **Wu and Huang, 19 pages.** A 2.7M-parameter causal Transformer over business
  bank transactions predicts a next-payment distribution over 200 days plus a
  no-event class. All results are relative changes on private data, so nothing
  can be replicated. It does not touch Indian stocks.

**VOL1a, five NSE names (RELIANCE, ICICIBANK, INFY, SBIN, TMPV), 15 cells, test
2012–25, mean QLIKE.** INFOSYSTCH was dropped: it has no data after its rename.

| h | Corr, target with naive | Paper's target: naive / HAR | True next-h: naive / HAR |
|---|---|---|---|
| 1 | 0.999 | 0.00021 / 0.00021 | 1.672 / 1.868 |
| 5 | 0.997 | 0.00042 / 0.00047 | 0.528 / 0.556 |
| 22 | 0.987 | 0.00198 / 0.00266 | 0.275 / 0.282 |

The paper's direction reproduces: naive lower at h = 5 and 22, with losses of the
same order (its 0.000286, 0.000676, 0.003806), and naive best in 6 of 15 cells
against its 13 of 33. Its target is nearly known at the origin: loss is about 140
to 8,000 times smaller than on the true target, so its tests compare models on
very little left to predict. **That does not make "naive wins" an artefact:** on
the true target the 252-day figure still wins 9 of 15 cells.

**VOL1b fails.** 33,543 rows, 138 names, 696 dates: QLIKE 0.285 (HAR) against
0.318 (60-day), t = 1.09, needed 3. Gains in both halves, 0.008 and 0.058. HAR
also ranks stocks worse, Spearman 0.59 against 0.67 with the truth, and ranking is
what forecast / volatility weights use.

**VOL1c fails.** HAR against HAR-VIX on NIFTY: h = 1 worse, t = −3.57; h = 5
t = 1.38; h = 22 t = 0.36, with gains of opposite sign across the two halves.

**CAD1 passes.** 92,274 gaps, 2,840 companies. Median 62 sessions, on-schedule
99th percentile 94, against GT7's 126. All gaps, 99th percentile 127, and 1.0% run
past 126. A 63-session window would have clipped 42%. No window near a quarter
turned up in `nse_engine`. Three calendar months do span 89 to 92 days.

**Assumptions.**

| Assumption | Status |
|---|---|
| Stationary window, OHLC valid (YZ) | fails on NSE: YZ1, noisy opens |
| Surviving large caps (paper) | fails for us: VOL1b keeps vanished names |
| Classifier available at each date | fails: weights dated 2025 scored on 2000–22 |
| Reliable sentiment labels | fails: κ 0.214 against its own reference |
| NYT-grade news history for India | unknown; we hold none, and no point-in-time set |
| Long stream history, bank-transaction data (CADENCE) | fails: not our domain |

**Audit.** Sentiment: FinBERT sits in the legacy path only
(`services/sentiment/analyzer.py`, `sentiment_forecast.py`); the engine has none.
A failed classification there returns neutral at 0.5 confidence
(`analyzer.py:101`), so an error reads as a calm view, the paper's no-news point.
Legacy only, not changed.

**Flagged.**
- **Lis et al.:** the abstract's "lowest mean QLIKE" at h = 22 comes from three
  stocks, 36.99 of 33.23 points; the other eight net negative. Stacking is
  near-constant by its own ridge on raw volatility, and its 66 tests shift the BH
  thresholds of every other model. Sec. 2.3's model-card scores are called
  "external validation", while its own audit finds κ 0.214. Nothing here tests
  news on a sensible target.
- **Wu and Huang:** "All reported direct metrics use the validation partition",
  so the holdout is not what is reported. The split is by customer, not by date,
  so the same calendar periods sit in training and evaluation.

**Where they fail in production.** Regime change and data: a classifier newer than
its sample cannot be run historically, and VIX loses its value at the short
horizon. Nothing here earns its costs, since YZ1 already moved Sharpe only 0.003
by swapping the estimator.

**Noticed, not tested, no action:** a 252-day figure did well on single names. Only
60-day against HAR was pre-registered, so this is a lead, not a result.

**VOL1d, that lead, rule written 1 Nov 2026 before the test (your go).** 252-day
against the book's 60-day volatility, VOL1b's design and names, forecast only: no
backtest, no core trial. **Pass, all three:** QLIKE lower with Diebold–Mariano
t ≥ 3 (bandwidth 252 sessions); lower in 2012–18 and in 2019–25; cross-sectional
rank correlation with the truth no worse than 60-day's. A pass earns a proposal
for a core trial, nothing more. Reported only: Feb–Apr 2020, for the slow window's
reaction to a spike.

**VOL1d fails, so no core trial.** 32,679 rows, 138 names, 696 dates, test 2012–25.

| Test | Needed | Result | |
|---|---|---|---|
| QLIKE, 252-day / 60-day | Lower, t ≥ 3 | 0.293 / 0.319, t 1.66 | Fail |
| Both halves | Lower in each | 2012–18 by 0.020, 2019–25 by 0.033 | Pass |
| Rank correlation with the truth | No worse | 0.694 / 0.668, t 4.2 on the gap | Pass |

The gain is real in sign but not established: lower in 10 of 14 years, flat in
2022, worse in 2012, 2013 and 2025, with 2020 the largest year. Without 2020,
0.019, t 1.62. In Feb–Apr 2020 the median stock's volatility ran 1.72 times the
252-day forecast and 1.43 times the 60-day one; on 19 March the truth was 0.87,
60-day 0.48, 252-day 0.37. The slow window is late in a spike, as expected, though
its QLIKE there was still lower. The better ranking is noted, not a reason to
reopen: all three tests were required, and YZ1 bounds what sizing can gain.

**VOL1a, is the paper's result built into its target?** On NIFTY 50, Eq. 4.13's
target (a future average of a trailing 252-session Yang–Zhang estimate) against
the true next-h-session realised volatility, h = 1, 5, 22: naive and HAR by
QLIKE, expanding, refit yearly, test 2012–25, purge h.

**VOL1b, can a better forecast beat what the book sizes on?** Next-22-session
realised volatility, top 50 names by traded value at each prior year-end, so
names that later vanish stay in. Deployed 60-day volatility against HAR(1, 5, 22),
pooled, refit yearly, test 2012–25. **Pass:** QLIKE lower, Diebold–Mariano t ≥ 3
on date-mean differentials with bandwidth 252, and lower in both 2012–18 and
2019–25.

**VOL1c, does India VIX add on NIFTY?** HAR against HAR-VIX, same design, h = 1,
5, 22. Same pass rule.

**CAD1, do our windows clear the quarterly cadence?** Gaps between a company's
consecutive result filings to 2025. GT7's 126-session window passes if it clears
the 99th percentile of on-schedule gaps.

## 5bo. LIQ1: Lipton, Lucic and Sepp, "Unified Approach for Hedging Impermanent Loss of Liquidity Provision" (arXiv 2407.05146v2; 1 Nov 2026; your ask)

**Result: nothing adopted, no code.** Replicated to its printed figures, and it
gave our pricer an independent check, which passed. CADENCE came with this request;
it is already closed in 5bn.

**What it is.** A 38-page derivatives paper on Uniswap V2 and V3 liquidity
positions, whose loss behaves like a short option position. Prop. 4.1 hedges any
such payoff with out-of-the-money puts and calls on a discrete grid anchored at the
entry price, the weights being differences of secant slopes. Sec. 5 values the
claim under Black–Scholes and, for any model with a moment-generating function,
with the Lipton–Lewis formula. Code: ArturSepp/StochVolModels, MIT; not used.

**Replication, scratch, from its definitions and the proof in its appendix.**

| Check | Paper | Ours |
|---|---|---|
| V2 units at 2,000 / 1,500 / 2,500 | (250, 500,000) / (289, 433,013) / (224, 559,017) | same |
| V3 units at 2,000 | (220.36, 559,282.18) | (220.36, 559,282.19) |
| V3 units at 1,500 / 2,500 | (543, 0) / (0, 1,052,020) | same |
| Max residual, strike width 50 | 0.025% of notional | 0.0245% |
| Max residual, strike width 100 | under 0.1% | 0.0956% |
| Error at the strikes | zero | zero |

**Audit of our code.**
- **Pricer:** `theory.black_scholes` against the Lipton–Lewis formula on 400 random
  cases with rates and dividends: largest gap 7.7e-9. `probability_itm` matches
  N(d2) to 2.2e-16. A separate formula, so a real check. No change.
- **"Liquidity sits out of the money":** `iv_history.expiry_atm_iv` and
  `live_chain.chain_summary` average the call and put at the nearest strike, so one
  leg is slightly in the money, which is fine at the money. `expiry_atm_iv` can step
  up to three strikes out, and there the in-the-money leg is the weaker quote. Not
  measured; no evidence of harm. Overlay legs are picked by delta, out of the money.

**Assumptions.**

| Assumption | Status |
|---|---|
| A liquidity position to hedge | fails: Centurion holds none, and it is not an Indian stock instrument |
| Liquid strip on the same maturity | fails for most NSE stock options; holds near the money on NIFTY |
| Strike at the entry price and range ends | fails in general on NSE's fixed grids; needs a cash or forward term |
| Zero carry, forward = entry | fails: Indian rates are about 6.5% |
| Fixed maturity | fails for an open-ended position: the hedge matches only at T |

**Why nothing is built.** Centurion has no non-linear payoff to replicate. Its
overlay structures are already vanillas, and its option decisions read market quotes,
so a stochastic-volatility model would add parameters and no decision. A bought
strip for book protection is not tested here; the paper itself says "purchasing
options is expensive in the long-term", and a pricing paper gives no reason to open
that question.

**Flagged.**
- The abstract's "hedged perfectly" holds only at fixed maturity, on available
  strikes, and leaves out fees and costs, which it defers to future work.
- Sec. 4 gives ETH strikes at 50 USD wide, then 100 up to one month, while Fig. 4
  uses 50.
- The decomposition of Props 3.5 and 3.6 was not re-derived: nothing here uses it.

**Where it fails in production.** Liquidity and costs: ask prices across a full
strip, on a thin chain, at retail size.

## 5bp. NS1: untested uncorrelated streams, a plan of action (India tracker NS1; your ask, written 02 Nov 2026, before any data was read)

**Why.** The golden target review (`docs/Golden target rebuild evidence.md`) puts
every queued idea at 0 to +0.20 Sharpe together. To lift 0.60 to 0.80, one stream
needs its own Sharpe of 0.53 at correlation 0 or 0.69 at 0.3. This plan tests the
streams that are in the docs but were never run (desk verdicts only), plus six
new ideas, in an order that spends no core trial. **This is a plan, not a
pre-registration:** each stream gets its own dated, timestamped section (5bq
onward) with exact rules and parameters before any number is read.

**Rules that apply to every item** (research protocol; tracker sections 3 and 6):
- **Pre-register first** (protocol §2): hypothesis (who pays and why), source,
  exact rules with every parameter, translations marked (A), window, costs, pass
  rule, and what a pass leads to.
- **Windows.** 2013–25 test; 2007–25 reported; **2008–25 for anything headed for
  money**, or the stream's own span with the gap stated (your call to go live
  without 2008, as U38). 2026 stays sealed, opened once per family.
- **No core trials.** The core has none before January (GT2B). Every item here is
  an event study (N = its hypotheses) or a new sleeve family with its own
  registry `data/nse_engine/runs_<name>`, at most 3 configurations, 1–2
  preferred.
- **Where it runs.** Engine-backed trials on Kaggle (one environment, LN-T26);
  arithmetic on recorded series may run on the Mac, stated (A), all of a
  family's trials in one place. Debugging runs: short windows, never recorded.
- **Costs.** Every instrument at its own dated cost model (`nse_engine.costs`,
  `fno_costs`, MCX's); a missing model is built with the stream. Break-even
  cost reported for anything high-turnover (OPS1). Sized against
  broker-feasible capital.
- **Gates.** Event signal: t ≥ 2 for a published effect, t ≥ 3 for a new one,
  Newey-West. Beside the book: the diversifier gate (§5.1) on both honest-base
  books, equal-risk weight, paired stationary bootstrap one-sided p ≤ 0.10, the
  2008–12 gain held if 2013–25 was seen. MaxDD and Calmar reported, never a
  condition (GT13). An in/out-of-market rule reports `timing_skill` (MIT1).
- **After a run:** bug fixes only, dated, effect stated. A failure is never
  re-tuned; it closes, and its code is removed unless something uses it (your
  rule: nothing unused is merged).
- **Code and records.** Claude marks the tracker item [~], works, leaves changes
  **uncommitted** for your review (you commit and push), adds tests to the fast
  suite, and logs the verdict in tracker section 8 and the stream register
  (`docs/golden_target_review.md` §6). A pass goes to a paper book (≥ 60
  sessions) before any live capital, then the ladder.

### Phase 0: no trial, before go-live (Nov to 11 Dec)

| ID | Stream | What is done | Done when |
|---|---|---|---|
| NS1-0a | Liquid-fund sweep on idle cash (LN-T29, cost model 5) | Waits on Zerodha Q5. Already queued; listed here because every later comparison uses its base. | Cost model 5 in the registry; honest base re-stated |
| NS1-0b | **Cash-futures carry on idle cash (new)** | A cost-model comparison, not alpha: buy stock, sell its near future, hold to expiry, roll. On the F&O archive 2013–25 (2008 contracts built from the raw archive): net annual yield after STT (2026 rates), brokerage, roll slippage and margin, against cost model 5's sweep. One look, not a trial. | Adopt only if it beats the sweep by ≥ 0.5 pt a year net in 2013–25 **and** in 2020–25, with cash free by the next session's buys |

### Phase 1: event studies, cheap (Nov to Dec; no core trial)

Each is one pre-registered section; N = its hypotheses (at most 2). Signal from
the event date's next open; costs at 15 bp a side plus impact; abnormal return
against NIFTY 500 TRI; 2008–25 where data exists.

| ID | Stream | Status before | Hypothesis | Data | Pass |
|---|---|---|---|---|---|
| NS1-1 | **Index deletion rebound** (NIFTY 50 / Next 50 / 100 / 500) | desk (§6 row 5) | Index funds sell on a known date; the price recovers over days 60–240 | **Not in hand** (corrected 2 Nov): the event list is built from NSE Indices' press releases (5bq); prices from the store | t ≥ 2, both halves positive; then a sleeve (3 configs) |
| NS1-2 | **Long-vol tail hedge (new)** | none | NIFTY's crash risk is under-priced in calm regimes (low IV rank) | `fo_store`, IV history; bought far-OTM monthly puts, rolled | A drawdown lever judged on Sharpe under §5.1 (no MaxDD route, GT13); its 2008 loss avoided reported. Options family budget (U32) |
| NS1-3 | **Promoter buying (SAST/PIT)** | desk | Insiders buy on information; outsiders follow late | PIT filings from Oct 2020 (gap stated) | t ≥ 2; a filter, not a sleeve |
| NS1-4 | **Demergers** | desk | Index and mandate holders sell the spun-off unit | **Not in hand** (data check 2 Nov, no trial): `corpact_api` 2008–25 holds 98 explicit demergers, 11 before 2019 (older ones filed as "Scheme of Arrangement", mixed with mergers and capital cuts), and never names the spun-off company; about 69 new symbols list within 150 days of each event, so the pairing cannot be automated. Needs a hand list from NSE listing circulars ("listing pursuant to scheme of arrangement") before 5bs can be pre-registered | t ≥ 2 after the listing day |
| NS1-5 | **F&O-segment entry/exit and MSCI rebalances (new)** | none | Flows on dates announced in advance | NSE F&O circulars, MSCI announcements (sourced first) | t ≥ 3 (new effect) |
| NS1-6 | **Anchor lock-in expiry, IPO days 30/90 (new)** | partly (IPO day 29–33 filter) | Mechanical supply when anchor shares unlock | IPO dates, anchor allotments (sourced first) | t ≥ 3; merged with the IPO filter if both hold |

### Phase 2: manual and opportunistic (any time; no code)

| ID | Stream | What is done |
|---|---|---|
| NS1-7 | **Buyback tenders, retail quota** | A hand table of past tenders 2013–25: acceptance ratio, price gap, holding days, net of the April 2026 buyback tax. Expected gain per Rs lakh at the Rs 3–30 lakh rungs. Tendering stays manual in Console (API has none), under MO1's Rs 2 lakh cap. **Done 3 Nov: not adopted** (result below the table). |
| NS1-8 | **Open-offer and delisting-offer arbitrage (new)** | The same table for SAST open offers and delisting offers: spread, acceptance, failure rate, worst case. A failed delisting is the stated risk. |

**NS1-7 result (3 Nov 2026; hand table, no trial).** Source: SEBI Filings →
Buybacks → Tender Offers, 1,232 documents 2013–25, 252 post-offer
announcements; 46 with a machine-readable small-shareholder row (the rest
scans or other layouts), 2013–22.
- Small-shareholder acceptance (reserved ÷ tendered, capped at 1): median 72%,
  mean 65%, quartiles 29% / 100%; 16 of 46 fully accepted, 13 below 30%.
- It fell as the trade got known: medians 2016 100%, 2017 60%, 2018 86%,
  2019 30%, 2020 22%; the 2021 cluster (TCS, NIIT 30%) and FDC 2022 (14%).
- The big names are crowded: Mindtree 14%, Aarti Drugs 7%, HEG 17%, eClerx
  18%, FDC 14%. Full acceptance came mostly where the offer was out of the
  money (tendered far below reserved: NALCO 0.02×, Welspun 0.25×, KG Denim).
- From 1 Oct 2024 the proceeds are a deemed dividend taxed at slab rate on the
  whole amount, the cost becoming a capital loss (Finance Act 2024; the April
  2026 change keeps the shareholder taxed). At a 30% slab a Rs 100 tender at
  a 15% premium accepted at 30% gains about 4.5% gross on the money tied up,
  and the slab tax on the accepted proceeds more than wipes it out before the
  unaccepted shares' price fall.
- Reading: not worth doing at any rung; no tendering rule for the book.
  Scratch code and downloads removed.

**NS1-8 pre-registration (09 Nov 2026, 14:03 IST, your go "do NS1-8 now,
capped at half a day"; written before any offer document was read).** A hand
table, no trial, no repo code; scratch files deleted with the reading.
- Events: (A) SAST open offers (2011 Takeover Code), public announcement (PA)
  2013–25; (B) delisting offers, initial announcement 2013–25. NSE-listed
  targets with a store price on the entry day. 2026 stays sealed.
- Trade A: buy at the store close of the first session after the PA; tender
  everything; accepted fraction a = min(1, offer size ÷ shares tendered) from
  the post-offer advertisement, paid the final offer price; the rest sold at
  the close of the 10th session after the tendering period ends. A withdrawn
  offer: sold at the close of the session after the withdrawal notice.
- Trade B: buy as in A after the initial announcement; tender at the floor
  price. Success: the final exit price for every share, at the 10th session
  after the bid window. Failure or rejection: sold at the close of the 2nd
  session after the result.
- Costs: 0.5% a side for impact in these small names plus STT 0.1% on each
  buy and sale, stamp and exchange charges; tax 20% of any positive event P&L,
  losses get no credit (conservative).
- Tradable: median traded value over the 20 sessions before the PA of at
  least Rs 5 lakh a day.
- Per event: net return r and days held h; annualised return on capital
  deployed = sum(r) ÷ sum(h) × 365.
- Adopted as a paper candidate only if A or B meets all of the following:
  - at least 10% a year after tax on capital deployed (the liquid fund earns
    about 6.5%);
  - positive in both 2013–19 and 2020–25;
  - no event worse than −10% (your bounded-tail rule, sleeve = one event);
  - at least 5 tradable events a year on average.
  B also reports its failure rate and worst case.
- Stop rule: if SEBI's documents give fewer than 100 open offers with a
  readable acceptance (or B fewer than 30 outcomes) by 16:30 IST, stop and
  report the data gap. Hard stop 18:30 IST.
- Amendment (14:12 IST, before any price or acceptance was read; data access
  only). SEBI's filing listing refuses scripted paging ("Unauthorized Request
  Blocked"), and it is not worked around. The store holds no open offers and
  only 10 delisting rows. Events come instead from SEBI's monthly open-offer
  statistics (statistics/open-offers/<mon><yyyy>.html): announcement date,
  tendering window, offer size, price and the acquirer's stake after a full
  response. Acceptance is bounded, not read:
  - low: a_min = offer size ÷ (100% − the acquirer's pre-offer stake), as if
    every other holder tenders;
  - high: a = 1.
  A passes only if it passes at a_min. It fails if it fails even at a = 1.
  Between the two is "inconclusive"; that is not a pass.
  Targets are matched by name to NSE symbols (EQUITY_L and symbol changes).
  Targets delisted since are missed, and that is reported. B (delisting
  offers) has no comparable statistics page, so it is reported as a data gap.
- Note (14:16 IST, while building the table, before the verdict was read):
  - Some pages swap the price and value columns, and the headers cannot be
    trusted. The price is the column that, times the share count,
    reproduces the other within 2%; 72 ambiguous rows are dropped.
  - Six rows dated their announcement after the tendering window opened
    (SEBI typos); they are dropped.
  - The return is linear in acceptance, but not always rising: when the
    stock ends above the offer price, full acceptance is the worse case. So
    the bounds are each event's minimum and maximum of r(a_min) and r(1).

**NS1-8 result (09 Nov 2026, 14:17 IST; hand table, no trial): FAIL, not
adopted.**
- Data:
  - SEBI's statistics: 654 offers, by tendering date Apr 2013–Feb 2025,
    with gaps Sep 2015–Mar 2016 and Mar 2024 (about 11.2 years).
  - 118 matched an NSE name (most targets are BSE-only small companies).
  - 58 were tradable events with prices: 25 not trading at the
    announcement, 25 with no NSE prices in the window, 10 under Rs 5 lakh a
    day.
  - About 5.2 events a year.
  - Median holding 128 days. Median offer premium over the entry close
    +0.8%: the market prices the offer on the next session. Median a_min 46%.
- Results, per year on capital deployed, after costs and tax:

  | Bound | Per year | 2013–19 | 2020–25 | Worst event | Hit rate |
  |---|---|---|---|---|---|
  | Low | −10.6% | −5.0% | −20.0% | −25.2% (AUTOAXLES 2022) | 33% |
  | High | +12.9% | +7.9% | +21.3% | −11.7% (NDTV 2022) | 60% |

  The high bound picks, for each event, whichever acceptance helps, so it
  overstates what can actually be reached.
- Information only, not pre-registered, entering only when the offer is more
  than 2% above the entry: 27 events (2.4 a year). Low +2.2% a year, high
  +19.6%. Below the liquid fund at the low bound, and under 5 events a year.
- Reading:
  - Fails at the low bound on every criterion except the event count.
  - Fails even at the high bound on the tail rule (−11.7% against −10%).
  - The spread is usually priced away by the first close, and about half
    the shares come back to fall with the stock.
- B, delisting offers: not tested. There is no statistics source, and the
  store has 10 rows against the 30 required: a data gap, closed with NS1-8.
- Scratch code and pages deleted.

**NS1-10 check (3 Nov 2026; desk, no trial): fails, closed.**
- Data: none in hand (the store has MCX only). NCDEX's bhavcopy page is
  rendered in the browser; no file or API link was found to script 2008–25
  history, so a backtest would start with a hand download or a vendor.
- Suspensions (public record, not re-verified here): futures in the most
  traded agri contracts were halted repeatedly (chana and others 2008–09,
  guar 2012, chana again 2016) and, from Dec 2021, SEBI suspended seven
  (paddy non-basmati, wheat, chana, mustard, soybean and its products, crude
  palm oil, moong), extended year after year. A trend rule loses its best
  markets exactly when prices move most, and a halted position cannot exit.
- Liquidity in what remains (guar, cotton seed oilcake, jeera, turmeric and
  similar) is thin for a sleeve, and suspensions made the 2008–25 history
  discontinuous.
- Reading: the precondition fails; GT5's third configuration stays unused.

### Phase 3: data first (quote before any test)

| ID | Stream | What is done |
|---|---|---|
| NS1-9 | **Analyst revision breadth, as a veto** | Ask two vendors for point-in-time consensus history from 2008. Weigh the quote against +0.05–0.15 Sharpe. Only then pre-register, as a core veto in a January slot (decision 2 of the GT3 list). |
| NS1-10 | **NCDEX agri trend (new)** | Check contract liquidity and suspensions 2008–25 first. A pass of that check joins the commodity family's last configuration (GT5's third), not a new budget. **Check 3 Nov: fails, closed** (below). |

### Not reopened

Value/quality (GT4), REITs (GT14), options premium selling (O2–O8), MON100 (R14),
bonds as alpha (BD1), residual momentum, 52-week high, sector momentum (S151A),
volatility targeting (R3, VT1), HMM regimes, pairs (Sharpe ~0.1), USDINR (RBI
rule) and leverage beyond your U36 call.

### Order and dependencies

1. NS1-0b after LN-T29 (needs cost model 5 as the comparison).
2. NS1-1 and NS1-2 first among the studies: data in hand, and NS1-2 aims at the
   2008–12 gap. Then NS1-3, NS1-4; NS1-5 and NS1-6 once their data is sourced.
3. NS1-7 and NS1-8 in parallel, by hand.
4. NS1-9 and NS1-10 are quote and data checks only until you decide.
5. Nothing here moves the deployed book before RA1's verdict (14 Dec). A pass
   becomes a paper book; live only after its own 60 sessions and your go-ahead.

**What would count as success.** Two or more streams passing §5.1 on both books,
their combined effect measured (not summed) on the walk-forward 2008–25 book.
Expectation, stated now: most will fail; the plan is worth running for the one
or two that do not.

## 5bq. NS1-1: the index deletion rebound (India tracker NS1; your ask "do NS1-1", pre-registered 02 Nov 2026, before any event list was built or any return read)

**Hypothesis.** Passive funds tracking a NIFTY index must sell a deleted stock
on or just before its effective date, whatever its value. The price is pushed
below fair value by forced, information-free selling and recovers as the
pressure ends. Who pays: index funds and ETFs, which accept the tracking cost.
The literature's long-side figure is +4–7% abnormal over days 60–240 after the
effective date (golden_target_review §6 row 5); it is a replication, so t ≥ 2.

**Data check, done before writing this (corrects plan 5bp, which said "in hand").**
- `data/nifty500_historical_constituents.json` holds 16 half-year NIFTY 500
  lists from Wayback snapshots, 2019-03 onward, some periods reusing an older
  snapshot. No dates, no NIFTY 50/Next 50/100, nothing before 2019. **Not usable
  as an event list.**
- `archive/indices` holds daily index *levels* 2012–26, not constituents.
- The source used instead: NSE Indices' press releases ("Replacements in
  indices", "Exclusion of ... from Nifty Indices"), each with an announcement
  date and an effective date. The archive must be built: tools/index_events.py
  (to write), saving every PDF under `archive/index_events/` and a parsed table
  `index_events.parquet` (symbol, index, action, announced, effective, reason).

**Events.**
- **Kept:** periodic-review deletions (semi-annual, March and September) from
  NIFTY 50, NIFTY Next 50, NIFTY 100 and NIFTY 500 (A: the four broad indices
  with the most passive money; NIFTY 500 has the least, kept to test size).
- **Dropped, fixed now:** deletions for a merger, delisting, demerger,
  suspension or surveillance (not forced, information-bearing); a stock deleted
  from one index but added to the next tier down on the same date (e.g. NIFTY
  50 to Next 50) is kept and flagged, and reported separately (A).
- A stock with no store price at the effective date + 1 is dropped and counted.

**Rules.**
- Entry: the open of the first session after the effective date (day 1).
  Exit: the close of day 240, or earlier if it leaves the store (exit at its
  last close; counted).
- Abnormal return: stock total return less NIFTY 500 TRI over the same days
  (A: the store's price-return NIFTY 500 plus 1.3 pt a year if the TRI is
  missing, stated).
- Windows reported: [−20, 0] (the pressure, information only), [1, 60],
  [60, 240] (the hypothesis), [1, 240].
- Costs: 15 bp a side plus the engine's impact model on the day-1 traded value,
  at a Rs 2 lakh position (A).

**Window.** 2008–25 if releases exist from 2008; otherwise from the first year
they do, the gap stated (your call, as U38). 2026 sealed. Halves: 2008–16 and
2017–25.

**Clarifications, 02 Nov 2026, before any run:**
- The release archive was checked: NSE Indices' press-release page lists every
  release from 1998, with index changes in every year 2008–25 (2010 and 2012
  under "Index Changes w.e.f."), so the full 2008–25 window holds.
- **The holdout stays sealed:** an event whose day 240 falls after 31 Dec 2025
  is dropped (about the last year of events; counted). The store is read only
  to 31 Dec 2025.
- Releases before Aug 2013 give company names, not symbols. Names map to NSE
  symbols through the corporate-action and board-meeting records (company name
  and symbol, 2006 on) and EQUITY_L, normalised (case, "Ltd"/"Limited",
  punctuation); an unmatched name is dropped and listed, never guessed.
- The reason is read per release section: kept only where the release or its
  section says periodic review, semi-annual review or rebalancing; any section
  naming a merger, amalgamation, scheme of arrangement, demerger, delisting,
  suspension or surveillance is dropped.
- **One event per stock per release:** a stock leaving several of the four
  indices in one release (e.g. Next 50 and 100) is one event, its indices
  listed; "by index" counts it under each.
- **Days:** day 0 is the effective date's session (the next session if it is a
  holiday); day 1 is the session after. The benchmark enters at NIFTY 500's
  open on day 1, like the stock. The store holds no NIFTY 500 dividend series,
  so the stated fallback applies: NIFTY 500 price plus 1.3 pt a year,
  compounded daily.
- [−20, 0] runs close to close; [1, 60] and [1, 240] from day 1's open;
  [60, 240] from day 60's close. Costs are charged on the three tradable
  windows: 15 bp a side plus the square-root impact (`costs.impact_bps`,
  default CostConfig) of Rs 2 lakh against the 20-session median traded value
  at entry and at exit.
- The store is loaded with no liquidity pre-filter, so thin NIFTY 500
  deletions are not dropped by the loader.
- Name normalisation (02 Nov 2026, before any run): the matcher removes
  Ltd/Limited/Co/Company/India/The/Of, treats "&" as "and", collapses tripled
  letters, and stems pharmaceuticals→pharma and industrial/industry→industries.
  "Corporation" is kept. Matching is exact on the normalised name with no
  fuzzy matching, and a name that maps to two symbols is excluded.

**Hypotheses (N = 2):**
- H1: mean net abnormal return over [60, 240] > 0.
- H2: the same over [1, 240].

**Pass.** H1 or H2 with Newey-West t ≥ 2 on calendar-month averages (events
in the same month averaged first, so overlapping events do not inflate t), and
**positive in both halves**. Also reported: count per year, hit rate, the worst
event, the tier-down subset, results by index.

**What a pass leads to.** A sleeve family `runs_index_del` (at most 3
configurations) under the diversifier gate (§5.1), each pre-registered. A fail
closes NS1-1; the event table stays only if another study uses it.

**Runs on** the Mac (A: arithmetic on store prices, no engine decisions), all
of it in one place.

**Result (02 Nov 2026): FAIL, closed.** 253 releases (1 download failed, 30
names unmatched, 7 with no effective date); 532 periodic deletions in 25 event
months. H1 [60,240]: −12.1% per event-month net, NW t −4.34 (2008–16 −9.6%,
2017–25 −13.5%). H2 [1,240]: −14.6%, t −3.90 (−9.0%, −17.7%). Event means:
[−20,0] +1.3%, [1,60] −1.8%, [60,240] −7.3%; hit rate 35–38%. Tier-down (17):
mean +5.4%, median −8.7%. Deleted stocks keep lagging; there is no rebound to
harvest. Known limits: the per-index split double-counts NIFTY 50 inside
NIFTY 500 labels, and 2022–23 yielded only 9 events; neither can flip a t of
−4. No other study uses the event table, so the code and data were removed.

## 5br. NS1-2: long-vol tail hedge (India tracker NS1; your ask "do NS1-2", pre-registered 02 Nov 2026, before any code or return)

**Hypothesis.** NIFTY's crash risk is under-priced when markets are calm, so
far out-of-the-money puts bought in calm regimes raise the book's Sharpe as an
overlay (a drawdown lever judged on Sharpe, GT13), most of all in 2008–12,
where the book has no alpha.

**Family.** Options (U32). Two configurations, the options family's N
becoming 20 (from 18). Runs go to `data/nse_engine/runs_options/`.

**Instrument and rule (parameters from common practice, not fitted).**
- NIFTY monthly put, the expiry settling closest to 30 days away (25–35),
  strike the nearest traded put to 0.90 × spot (10% out of the money).
- Decided at the close of the session after the previous put settles, filled
  at the next session's close (Y2's timing), held to expiry, settled at
  intrinsic on NIFTY 50's close with exercise STT.
- Notional 1× the stream's equity (units = equity / spot); premium paid from
  that equity; no other position, idle cash earns nothing in the stream (the
  gate's book already earns the liquid rate).
- Costs: `fno_costs` (fills with OptionsConfig slippage, `option_charges`,
  `exercise_stt`), Y2's helpers unchanged.
- **H-A (always):** buy every month.
- **H-B (calm only):** buy only when NIFTY's IV30 (`fo_store/iv`) on the
  decision day is at or below the median of its trailing 252 sessions
  (at least 126); else hold no put that month.

**Window.** 2008-01-01 to 2025-12-31 (NIFTY options in the store throughout);
2026 sealed.

**Gate (protocol 5.1 as amended by GT13, `diversifier_gate.evaluate`,
unchanged).** An overlay at equal risk (cap 3×), on both books (2008–25
backtest and walk-forward), live-like (liquid rate, D2 haircut):
- Pass = combined excess Sharpe above the book's with paired-bootstrap
  p ≤ 0.10 on **both** books, **and** the combined Sharpe in 2008–12 at least
  the book's on both books.
- Reported, not gating: MaxDD, Calmar, the 0.25 tested weight, the stream's
  own Sharpe and deflated Sharpe at N = 20, correlation, 2008's book loss
  avoided, the yearly table.

**What a pass leads to.** A third configuration at most (a strike or tenor
variant, pre-registered), then paper trading under the options family's rules.
A fail closes NS1-2 and its code is removed.

**Runs on** the Mac (A: arithmetic on store prices and recorded book runs, no
engine decisions).

**Result (02 Nov 2026): FAIL, both; closed.** Runs `20261102T175824345660Z_863902c9`
(T-A) and `20261102T175827290331Z_afe1ffc5` (T-B) in `runs_options`; options
family N = 20.
- Alone, 2008–25: T-A CAGR −1.1%, Sharpe −0.16, MaxDD −32%, 206 puts, 2.4%
  paid off; T-B CAGR −1.1%, Sharpe −0.70, 123 puts (83 months skipped), 1.6%
  paid off. Deflated Sharpe 0.006 and 0.000.
- Overlay at equal risk (weight ~2.6–2.9×), excess Sharpe book → combined:
  T-A backtest 0.70 → 0.45 (p 0.96), walk-forward 0.60 → 0.34 (p 0.96);
  T-B 0.70 → 0.47 and 0.60 → 0.34 (p 1.00). 2008–12 worse on every book.
- Reported: MaxDD worse with the hedge (backtest −28.6% → −32.4% T-A, −33.9%
  T-B); correlation to the book −0.17 (T-A), ~0 (T-B). 2008: T-A halves the
  backtest book's loss (−23.2% → −11.8%), T-B misses it (calm filter off
  through the crash).
- Reading: 10% OTM NIFTY puts are not under-priced; the premium bleed exceeds
  the crash payoff, and buying only in calm regimes keeps the bleed and loses
  the payoff. The code is removed; the two recorded runs stay in the options
  registry (they count in N).

## 5bs. NS1-3: promoter open-market buying (India tracker NS1; your ask "do NS1-3", pre-registered 02 Nov 2026, before any filing was downloaded beyond a 9-month depth probe)

**Data (checked 2 Nov).** NSE's insider-trading API (`/api/corporates-pit`)
returns filings from Jan 2016 (SEBI PIT Regulations 2015; nothing earlier),
not Oct 2020 as 5bp said. The depth probe read only counts and one row per
month, no prices.

**Hypothesis (published, t ≥ 2).** Promoters buying their own stock in the
open market know something; the price drifts up after the filing.

**Events.**
- Filings with category containing "promoter" (Promoters, Promoter Group),
  mode "Market Purchase", security "Equity Shares", transaction Buy.
- Summed per (symbol, filing date); kept if the sum is ≥ Rs 10 lakh.
- One event per stock per 60 sessions (the first; later ones in the window
  dropped), so holdings never overlap.
- Symbol mapped to the store's canonical symbol (symbol changes); unmatched
  dropped and counted.
- Tradable at entry: 20-session median traded value ≥ Rs 25 lakh (the
  engine's own floor).

**Timing.** Day 0 = the filing's date (filings land mostly after the close);
entry at day 1's open, the next session. Windows: [1,20], [1,60], [1,120] to
the close; a stock that stops trading exits at its last close.

**Abnormal return and costs.** Against NIFTY 500 plus 1.3 pt a year (5bq's
fallback, entry at day 0's close); 15 bp a side plus `impact_bps` at Rs 2
lakh on both legs (5bq's model).

**Hypotheses (N = 2).** H1: mean net abnormal [1,60] > 0. H2: the same over
[1,120].

**Window and pass.** Events with day 0 from 2016-01-01 and day 120 by
2025-12-31; 2026 sealed. Pass: H1 or H2 with Newey-West t ≥ 2 on
calendar-month averages, and a positive mean in both 2016–20 and 2021–25.

**What a pass leads to.** A filter on the core book (a positive tilt or a
veto), pre-registered for a January slot; not a sleeve. A fail closes NS1-3
and removes its code; the downloaded filings stay only if another study uses
them.

**Runs on** the Mac (A).

**Result (02 Nov 2026): FAIL, closed.** 333,330 filings (120 months; monthly
requests checked against weekly ones, no truncation; filings fall from ~57k a
year in 2017 to ~14k in 2025). 28,767 promoter market buys, 13,073 events ≥
Rs 10 lakh; dropped 377 unmatched, 515 too late, 7,308 inside a stock's
60-session gap, 2,019 thin; 2,854 used (173–418 a year). Net abnormal vs
NIFTY 500 + 1.3 pt:
- H1 [1,60]: −0.2% per event-month, NW t −0.16 (2016–20 −1.5%, 2021–25 +1.2%).
- H2 [1,120]: −1.5%, t −0.63 (−3.7%, +0.9%).
- Events: [1,20] +0.05%, [1,60] −0.2%, [1,120] −0.7%; hit rate 46–47%.
  Buys ≥ Rs 1 crore (1,395): [1,60] −0.7%.

Promoter open-market buying carries no drift in India after costs; it is no
filter for the core book. Code, filings and results removed.

## 5bt. FND1: live funding (LN-T11) and the liquid sweep's policy B′ (LN-T29) (India tracker FND1; your ask "build LN-T29's policy B′ now", pre-registered 05 Nov 2026, 18:23 IST, before any run)

**Why.** Zerodha could not confirm (5 Nov 2026, tracker ZR1) that a buy AMO
may spend a sell AMO filled at the same open, so the live book now buys only
from the cash already free (LN-T11, `nse_engine_executor.
LIVE_SELL_PROCEEDS_CREDIT = 0`). Every recorded run funds buys with that
open's sales. Policy A of the sweep (SWP0) depended on the same credit and is
not usable; B′ is the remaining design.

**Not a signal trial.** An execution and cost measurement, like SWP0 and
LN-T21: no family budget, the deployed configuration unchanged, 2026 sealed.

**Runs.** `engine.run_backtest(funding=...)`, reference runs (never recorded,
outside the config hash), on each of the 18 folds of R4b's 2008–25
walk-forward (`data/nse_engine/wf_stitched_r4b.json`): each fold's recorded
test configuration over its test window, stitched.
- **M (model):** funding as recorded. Harness check: M must reproduce each
  fold's recorded returns (largest daily difference under 1e-9), else the
  measurement stops.
  *Amended 05 Nov 2026, 18:25 IST, after fold 0's M only, before any L or S
  run:* the check failed (largest daily difference 0.11%), as it must: the
  engine and data code changed after R4b was recorded on 17 Oct (for-cause
  exits, circuit locks, the cost margin). M is now run on today's code and
  data, its stitched CAGR and Sharpe reported beside the record's; the
  measurement proceeds if they are within 1.0 pt and 0.10 of it (else it
  stops: the walk-forward needs re-running first). L and S are judged against
  this M, same code and data.
- **L (live funding, LN-T11):** buys spend only the cash free before the
  open; a sale funds the next session.
- **S (B′):** L, plus idle cash in a liquid ETF. Each night, with N tomorrow's
  buys at the close plus 1.25% (the live limit band and cost allowance) and C
  the cash: if N > C, redeem min(N − C, the fund) at tomorrow's open, which
  funds the session after; else, if C − N − 2% of equity ≥ 5% of equity, park
  C − N − 2% at tomorrow's open after the buys. Fund return: the bond store's
  overnight-rate index less LIQUIDCASE's 0.27% a year. Costs: the bond
  stream's liquid trade cost (statutory, no STT; 3 bp floor; impact at a
  Rs 74.6 crore ADV, 5% cap) plus the DP charge on each redemption. Buffer
  and band are the bond stream baseline's (plan 5ag), not chosen here.

**Basis.** As-run (idle cash earns nothing in M and L), D2's haircut on the
invested share; metrics on the stitched 2008–25 series: CAGR, excess Sharpe,
MaxDD (recorded, not gating), and both halves (2008–16, 2017–25).

**Reported.** L − M is the cost of LN-T11 (no decision: it stands, the
funding answer forces it). **B′ is wired into the paper and live books** if
S − L is at least +0.3 pt CAGR and Sharpe is not lower, on the full window
and in both halves; else LN-T29 closes on the record and its engine code is
removed. Either way the sweep trades only after RA1 (14 Dec) and your
go-ahead.

**Result (05 Nov 2026): B′ FAILS; LN-T29 closed, its engine code removed.**
On today's code M is 11.15% / 0.428 against the record's 11.14% / 0.426,
inside the amended tolerance.

| Run | CAGR | Sharpe | MaxDD | 2008–16 | 2017–25 | Invested |
|---|---|---|---|---|---|---|
| M (model) | 11.15% | 0.428 | −32.8% | 6.06% / −0.005 | 16.46% / 0.836 | 0.695 |
| L (LN-T11) | 10.85% | 0.414 | −31.1% | 6.13% / −0.002 | 15.78% / 0.805 | 0.678 |
| S (B′) | 4.16% | −0.332 | −23.0% | 3.64% / −0.365 | 4.69% / −0.296 | 0.363 |

- **LN-T11 costs 0.30 pt and 0.014 Sharpe** (L − M), all in 2017–25. The
  engine enters only on rebalance nights, so a buy the free cash cannot cover
  waits for the next rebalance (5 or 21 sessions), not one session. The live
  planner shares that engine, so live behaves the same way. Free cash is taken
  before the open's stop sales too (a GTT fires after the AMOs are released);
  rerun with that, L is unchanged.
- **S − L is −6.7 pt.** The cause is the specification, not the fund. In
  fold 2017 all 12 redemptions were re-parked the next session: with no
  rebalance that night, N = 0, so the rule parked the redeemed cash again
  before any buy could use it. Invested fell to 0.36.
- A corrected design would redeem the night before each rebalance night and
  hold the redemption until the buys go through. That is a new specification
  and needs its own pre-registration; it was not run.

### 5bt-2. FND2: the sweep's policy B″ (your ask "build a corrected version of B″", pre-registered 05 Nov 2026, 18:52 IST, before any code or run)

**Rule (S″).** L's funding (LN-T11), plus a standing cash reserve and a
liquid ETF for the rest. Each night u, with C the cash (free at the next
open), N tonight's buys at the close plus 1.25%, E the equity and F the fund:
- **Reserve** Rv = E × the largest buy need, as a share of equity, of the last
  4 scheduled rebalances (their wanted buys before any cash cut); before 4
  have run, Rv = E (nothing parked).
- **Night before a scheduled rebalance** (`rebalance_day[u + 1]`): redeem
  min(max(Rv + N − C, 0), F) at the open u + 1; its proceeds are free for
  the rebalance's buys at the open u + 2. A rebalance forced off schedule (a
  regime switch, an empty book) cannot be foreseen: its buys wait, as in L.
- **Any other night:** park C − N − Rv − 2% of E at the open u + 1, after the
  buys, if that is at least 5% of E. Never park on the night before a
  scheduled rebalance.
- Fund, costs, buffer (2%) and band (5%) as in 5bt; window, basis, runs (18
  folds of R4b, M and L as in 5bt), halves as in 5bt.

**Decision (unchanged from 5bt).** B″ is wired into the paper and live books
if S″ − L is at least +0.3 pt CAGR with Sharpe not lower, on the full window
and in both halves; else the sweep closes for good and its code is removed.
Reported beside it: the mean share parked, redemptions a year, and how often
a scheduled rebalance's buys were cut for cash in S″ against L.

**Result (05 Nov 2026): B″ FAILS; the sweep is closed for good, its code
removed.**

| Run | CAGR | Sharpe | MaxDD | 2008–16 | 2017–25 | Parked | Cut for cash |
|---|---|---|---|---|---|---|---|
| L (LN-T11) | 10.85% | 0.414 | −31.1% | 6.13% / −0.002 | 15.78% / 0.805 | 0 | 87 of 290 rebalances |
| S″ (B″) | 11.00% | 0.432 | −27.4% | 6.58% / 0.038 | 15.59% / 0.796 | 10.2% | 100 of 292 |

S″ − L is +0.15 pt overall (needed +0.3), +0.45 pt in 2008–16 and −0.19 pt
in 2017–25. The reserve keeps most idle cash unparked (10% of the book
parked, 6.6 redemptions a year), and the buys of rebalances forced off
schedule (regime switches) still wait, now more often (100 cuts against 87).
The sweep earns its yield only where the book is least invested; no third
specification is run (the protocol: a failed configuration is never
re-tuned). Idle cash earns nothing in the live plan; rule C's as-run basis
and `mc_bands.DEPLOYED_IDLE_CASH_SWEEP = False` stand.

## 5bu. MON1: live monitoring fixed before the first live session (India tracker GL1; golden-target review "do now", pre-registered 05 Nov 2026, 19:11 IST, before any live session)

Three readings of the live book, fixed now so that none is chosen after its
result. None is a trial or spends a family budget; none trades. Each is
judged only at the points named.

**1. Cumulative gap bands (a slow leak G4 cannot see).** The daily gap is
G4's: the live book's daily return less its same-period reference (the
live-rule backtest, `paper_gate`). With σ the daily gap's standard deviation
on the deployed paper book at its 60th G4 session (RA1, 14 Dec 2026), floored
at 10 bp and then fixed, the cumulative gap G(n) after n live sessions is read
at n = 60, 120 and 250 (then every 250):
- **Breach:** G(n) < −2σ√n (about 2.3% false alarms at each point under no
  leak; at 250 sessions it catches a steady leak of 0.13σ a day).
- **At 60:** a breach is a WATCH line in the email and a review of fills and
  costs; nothing else.
- **At 120 or 250:** a breach holds the ladder at its rung (no step up) until
  the cause is found and written in the tracker; the owner decides any step
  down. Rule D, the kill switch and the drawdown rule are unchanged by it.

**2. Monthly universe rank IC (Tier B: supports, never proves).** At each
month-end of live trading, the Spearman correlation across the deployed
universe between the engine's combined forecast at that close and each
name's return over the next 21 sessions (point in time, as the engine saw
it). Benchmark: 0.051 at 21 sessions (2013–25, Stage F item 5).
- Reported monthly with its running mean and t over months.
- **Read at 12 months only:** t ≥ 2 supports the signal still ranking;
  mean IC ≤ 0 is a "signal not ranking" review (no automatic action:
  12 months is too few to act on, the review's t 1–2.3 range).

**3. The after-tax line.** Once a year, on the financial year (April–March),
the live book's return after tax against a passive hurdle:
- **Book:** realised gains taxed as short-term equity (20%) under 12 months,
  long-term (12.5%) beyond, above the Rs 1.25 lakh yearly exemption; gold and
  silver ETF gains short-term at the 30% slab, long-term at 12.5%; 4% cess on
  all; statutory trading costs as booked.
- **Hurdle:** 50% NIFTYBEES and 50% GOLDBEES bought on the book's first live
  day, never rebalanced, its gain taxed only if sold (shown both untaxed and
  as if sold at the year's end).
- Reported beside the pre-tax figures. It feeds your decision 8 (scale past
  the canary only if the book beats the hurdle after tax). These rates are the
  2026–27 rules as I read them; the chartered accountant's view replaces them,
  recorded here, before the first full year closes.

**Built before each is first read:** the gap bands by the 60th live session,
the rank IC by the first live month-end, the after-tax line by 31 March 2027.

## 5bv. T2T1: no new buys in trade-for-trade stocks (India tracker T2T1; your call 5 Nov 2026 after GT11C, "do T2T1"; pre-registered 05 Nov 2026, 23:14 IST, before any code or run)

**Why.** A risk rule for exitability, not a return claim: a stock in
trade-for-trade (series BE: GSM stage I and above, ASM long-term stage IV)
has 5% bands and locks at its circuits more often. GT11C found 0.71% of the
walk-forward's buys in such stocks at the decision close.

**Rule (`portfolio.no_t2t_buys`, off by default and hash-neutral).** A stock
whose series at the decision close is BE is never newly selected (it joins
the day's blocked names, so neither a rebalance nor a refill picks it), and a
held one is never added to (its target is capped at its current weight).
Exits, stops and the metal sleeves are unchanged. The engine reads the series
from the store (`MarketData.t2t`, not part of the data hash).

**Test (decision 7's "not lower" tier; no core slot, no family budget).** On
R4b's 18 walk-forward folds (`wf_stitched_r4b.json`), each fold's recorded test
configuration over its test window, today's code and data, stitched 2008–25,
live-like as run (idle cash earns nothing, D2's haircut on the invested
share): M (the rule off) against T (on). Reference runs, never recorded.
- **Pass:** T's excess Sharpe at least M's less 0.005 (rounding noise) on
  2008–25. Reported, not gating: CAGR, MaxDD, both halves, the buys the rule
  removed.
- **Pass leads to:** the rule in the deployed configuration only on your
  go-ahead, between RA1 (14 Dec) and the canary's first live session (a
  configuration change restarts the paper or live G4 window). **Fail:** the
  switch and its code are removed; the record stays here.

**Result (05 Nov 2026): FAIL; the switch and its code removed.**

| Run | CAGR | Excess Sharpe | MaxDD | 2008–16 | 2017–25 | Buys |
|---|---|---|---|---|---|---|
| M (rule off) | 11.15% | 0.428 | −32.8% | 6.06% / −0.005 | 16.46% / 0.837 | 3,139 |
| T (rule on) | 10.60% | 0.383 | −33.4% | 5.79% / −0.029 | 15.63% / 0.773 | 3,138 |

T − M is −0.045 Sharpe (the bar: no worse than −0.005), lower in both halves,
and MaxDD is no better. The buy count is unchanged: the blocked names were
replaced by the next-ranked ones, which did worse. Stocks reach
trade-for-trade after surging, so for this momentum book they were winners
as often as traps, and the exitability the rule bought did not show in the
drawdown. One walk-forward cannot separate cause from path, but the rule set
before the run decides: no exclusion. Exitability stays with the existing
controls (circuit-lock handling in fills, the session cap, stops).

## 5bw. J1: refill exits alone (India tracker J1, U20; your decision 2 of 5 Nov 2026; pre-registered 06 Nov 2026, 18:54 IST, before any run)

**Claim.** Ranks 21–40 earn as much as the top 20 (GT2A), yet a slot freed by
an exit waits in cash until the next rebalance. Refilling it at once raises
deployment in rising markets, where the book holds most names, without
leverage, a new signal or an exposure tilt. A return claim: decision 7's
strict gate.

**Change, one setting.** `portfolio.refill_exits` true; nothing else moves.
On a non-rebalance session a slot freed by an exit is refilled from the rank
list at its rebalance weight, within the allocator's gross, never in a
drawdown-rule halt or at regime scale 0 (the engine's existing rule, E1). E4
paper-trades refill bundled with exit rank 60, which failed in GT2B, so E4's
fate does not decide this (the review's change of record to U20).

**When and against what.** Week of 11 Jan 2027, after RA1 (14 Dec), V3 and
the January re-baseline (LN-T30 locked circuits, OPS1A, for-cause exits;
cost model 5 is void, the sweep closed in FND1/FND2), against the configuration
they leave live (the base). If V3 settles refill (E4 promoted), J1 is not run
and its slot stays unused.

**Runs.** The base's walk-forward machinery unchanged: its grid, 18 folds
anchored from 2007, one-year first training window, 12-month tests,
`store_ext2006`, `excess_sharpe` selection, on Kaggle, with refill true at
every grid point (selection runs as for the base). The base is re-run on
today's code first and must reproduce its recorded fold hashes. Then each
fold's selected test configuration is re-run locally with `funding="live"`
(LN-T11, FND1), for the base and for J1.

**Basis.** The milestone's (your decision 1, revised 5 Nov): 2008–25
live-like as run: idle cash earns nothing, D2's haircut on the invested
share, buys funded only by free cash (`funding="live"`).

**Pass, all of them (decision 7, strict):**
1. J1's excess Sharpe above the base's, with `diversifier_gate.bootstrap_p`
   ≤ 0.025 (one-sided, the bootstrap form of t ≥ 2).
2. J1's Sharpe above the base's in 2008–12 (the failing era).
3. J1's recorded 2013–25 run: DSR ≥ 0.95 at the core's N after the trial
   (62), widened for autocorrelation (AC1); the core family's PBO ≤ 30% with
   it added.

**Reported, not gating.** CAGR, MaxDD, Calmar, alpha against NIFTY TRI,
2013–25 and 2017–25, trades a year, mean invested, the same comparison on
the model's funding, the passive NIFTY and gold hurdle after tax (MON1's
line).

**Power, stated before the run.** With two correlated 18-year books, a true
+0.10 Sharpe passes a t ≥ 2 test about 16–26% of the time (the review's
calc): a fail is likely even if refill helps.

**Fail.** U20 is dropped; `portfolio.refill_exits` is removed once E4 no
longer needs it (after V3); not re-tuned. A PBO above 30% closes the core
family for the year. **Pass.** 60 paper sessions on the configuration with
refill, then the canary rules, at a configuration change that restarts the
live G4 window: not before about April 2027.

## 5bx. J4: the half-step in risk-off rebounds (India tracker J4; your decisions 2 and 5 of 5 Nov 2026; pre-registered 06 Nov 2026, 18:54 IST, before any run)

**Claim.** The book's weakness is rebound lag (37% of the 2008–10 rebound
captured): after a crash the regime exit keeps it in cash while the market
climbs. Holding a capped index position in risk_off while NIFTY is rising
captures part of the rebound without changing the core. A return claim:
strict gate. Prior is modest: R11 bought 2008's bear rallies, and GT2C found
NIFTY did not beat the liquid rate in the states it held.

**Family.** The overlay family (GT2C), configuration 2 of 3; arithmetic on
the recorded walk-forward, on the Mac (A). Decision 5 waives MIT1's "no
exposure may be added" for this rule only, capped at 25%.

**Rule, fixed now.**
- Weekly, on each week's last session close: if the regime state is
  risk_off, the drawdown rule is normal (`dg.guard_dates`, as A1G and GT2C),
  and NIFTY's 21-session price return is above 0, the target is NIFTYBEES at
  25% of the book's equity, from idle capital only (never above 100% gross);
  otherwise the target is 0.
- Out at once, not weekly, when the regime leaves risk_off (the core
  re-enters) or the drawdown rule leaves normal; the core's buys are funded by
  selling NIFTYBEES at the same open (GT2C).
- Timing: decided on the close, filled at the next open, credited from the
  session after (GT2C's two-session lag both ways).
- Return: NIFTY 50 TRI less 0.05% a year; costs at dated statutory charges
  plus 5 bp a side; D2's haircut on what it holds.
- Size check, stated now: 25% × a −45% leg is −11.25% of the book, within
  the 10–15% overlay cycle cap.

**Base and basis.** The book J1 and MV1 leave (run by 5 Feb 2027, after
them), its recorded walk-forward stitched 2008–25, on the milestone basis
(as run, D2's haircut, free-cash funding); the regime state rebuilt from the
engine on `store_ext2006` and checked (every regime-exit day follows a
risk_off close).

**Pass, all of them (decision 7, strict, with GT2C's gate 2 kept):**
1. On 2008–25: CAGR, excess Sharpe and alpha against NIFTY TRI all higher
   than the base, with `bootstrap_p` ≤ 0.025 on the Sharpe difference.
2. Sharpe and Calmar not lower in at least 2 of the 3 eras (2008–12,
   2013–19, 2020–25).
3. The overlay's worst loss over any one holding cycle at most 15% of the
   book's equity at its entry.

**Reported, not gating.** MaxDD, weeks held, mean weight, the 2008–10 and
2020 rebound capture against the base, NIFTY's 30 worst weeks, the overlay's
cost a year.

**Fail.** No code; no third try (the family's last slot stays for a new
source). **Pass.** Built as an overlay switch, then 60 paper sessions; it
counts as a promotion trial.

## 5by. CC1: counter-cyclical investing, Phases 0–1 (India tracker CC1; your ask "validate it, then decide"; read rule fixed 06 Nov 2026, 19:08 IST, before any return was read)

**Question.** Does buying when the market or a sector is cheap and holding
back when it is dear carry information the book could use? Phases 0–1 are a
look (the pipeline's free look with a read rule): nothing recorded, no family
budget spent, no configuration chosen. They decide whether Phase 2 (at most
two configurations in a new family, CC, each pre-registered separately) is
worth a run.

**Prior, stated now.** Low. R11 (crash re-entry), GT4 (value), NS1-1
(beaten-down deletions) and GT2C (idle capital into NIFTY) failed; GT2A found
sold winners keep winning. Valuation is known to predict 5–10 year returns
far better than 1 year.

**Data (Phase 0).** NSE's daily index files in the archive, 21 Feb 2012 to
31 Dec 2025 (2026 sealed): P/E, P/B and dividend yield of NIFTY 50 and of 12
sectors (Auto, Bank, Energy, Financial Services, FMCG, IT, Infrastructure,
Media, Metal, Pharma, PSU Bank, Realty), names mapped across NSE's renames
("S&P CNX Nifty" = Nifty 50, "CNX Finance" = Nifty Financial Services,
"CNX X" = Nifty X). Nothing before 2012 exists here, and NSE's history
endpoints refuse automated clients: the 2008–11 gap is stated, not filled
(decision 4).
- **Break check.** NSE moved to consolidated earnings around April 2021. A
  series is unusable if its median day-on-day change across 2021-03-25 to
  2021-04-09 differs from the price's by more than 5%; then the next measure
  in the order P/B, dividend yield (inverted), P/E is used.
  *Amended 06 Nov 2026, 19:10 IST, after Phase 0's series were built and before any return was read:* a
  median cannot see a one-day jump, which is what a method change makes, so the check is the largest
  single-day gap between the measure's log change and the price's (inverted for the dividend yield) over
  24 Mar to 9 Apr 2021, a break at 5% or more; a sector with no usable measure is left out. Result: NIFTY
  50 P/B (gap 0.8%); P/B for Auto, Energy, FMCG, Financial Services, Infrastructure, Metal, PSU Bank,
  Pharma and Realty; Bank on its dividend yield (P/B 8.5%, yield 1.0%); IT (P/B 7.7%, yield 24%, P/E
  14%) and Media (P/B 6.1%, yield 233%, no P/E) left out: read 2 ranks 10 sectors, 3 cheapest and 3
  dearest.
- **Measure.** V = the series' percentile against its own trailing 1,260
  sessions (at least 504), so valuation is judged against its own recent
  range; low V = cheap.

**Read 1, the market.** Each session from the first with V, NIFTY 50 TRI's
return over the next 252 sessions, grouped by V's quintile (fixed cut points
0.2, 0.4, 0.6, 0.8). Spread = cheapest-quintile mean less dearest-quintile
mean; Newey-West t with 252 lags (overlapping windows).

**Read 2, sectors.** At each month-end, the 3 sectors with the lowest V and
the 3 with the highest (ties by name); spread = the 3 cheapest's mean price
return over the next 252 sessions less the 3 dearest's, averaged across
month-ends; Newey-West t with 12 lags.

**Decision rule.** Phase 2 runs only for a read whose spread is positive with
t ≥ 1.5: read 1 gates CC-A (the book-level valuation overlay), read 2 gates
CC-B (the beaten-down sector sleeve). Below that, the idea is discarded
for this book and recorded here; nothing is coded. Power, stated now: about
11 independent years, so only a large effect can clear even t 1.5.

**Result (06 Nov 2026): both reads fail; counter-cyclical investing is
discarded for this book, Phase 2 not run, nothing coded.**

| Read | Result | Bar |
|---|---|---|
| 1. Market: NIFTY 50 TRI's next 252 sessions by P/B quintile (24 Feb 2014 – 26 Dec 2024, 2,668 sessions) | cheapest +47.6%, then 21.7%, 12.9%, 14.1%, dearest +7.6%: spread 40.0 points, NW t 1.12 | t ≥ 1.5 |
| 2. Sectors: 3 cheapest less 3 dearest, next 252 sessions (130 month-ends) | +0.22% a year, NW t 0.06, positive 55% of months | t ≥ 1.5 |

- Read 1's spread is one episode. The cheapest quintile holds 162 sessions,
  nearly all around the March 2020 crash, while the dearest holds 1,216: in
  a rising market NIFTY's P/B sits near the top of its own 5-year range most
  of the time. The order of the quintiles is suggestive, but a single rebound
  cannot be told from a rule, and 2008 is outside the data.
- Holding back when dear has no case here: the dearest quintile still
  returned 7.6% a year, about the liquid rate, so trimming would at best trade
  equity for cash at no gain.
- Read 2 finds no information: cheap sectors did not beat dear ones.
- The closest testable form, buying the market in a risk-off rebound once
  NIFTY turns up, is J4 (5bx), already pre-registered for January.
Temporary script and extracted series deleted.

## 5bz. OPT2: covered calls, cash-secured puts, verticals and collars, closed at Phase 0 (India tracker OPT2; your brief "find out whether options strategies are wrong for Centurion"; closure accepted 06 Nov 2026, 19:55 IST)

A desk check: no configuration pre-registered or run, no trial spent (options
N stays 20). Lot sizes from `fo_store` at 31 Dec 2025: one stock-option lot is
Rs 2.9–9.4 lakh (median 6.6, 208 underlyings); a NIFTY lot is 65 units, about
Rs 17 lakh. A 20-name book holds about Rs 0.15 lakh a name at Rs 3 lakh and
Rs 1.5 lakh at Rs 30 lakh. The store has closes and settlements only, no
bid/ask.
- **A, covered calls on holdings:** no name's lot fits one position at either
  size (a full book needs about Rs 1.3–1.9 crore). The index form, a short
  NIFTY call against the book, is A1, which passed (RT1, A1G) and paper-trades
  as book a1. Closed, not re-tested; revisit near Rs 1.5 crore (your call).
- **B, cash-secured puts as entry:** the same lot problem, and a prior against
  it: assignment comes when the stock falls, so it buys the momentum names
  that fade and misses the ones that run. Closed, not re-tested.
- **C, verticals for futures exposure:** the deployed book holds no futures
  (GT8, U36, L7 closed); the standalone directional form is O5–O8's regime
  debit spreads (Sharpe −0.54 / −0.44). Closed, not re-tested.
- **D, collar:** stock collars fail on lots; an index collar is A1 plus a long
  put, and NS1-2 showed the put leg costs more than it pays (book Sharpe 0.60
  to 0.34). No new angle. Closed, not re-tested.

## 5ca. BT7: seven backtest strategies (India tracker BT7; your ask "validate and ensure they do not add value before discarding"; pre-registered 06 Nov 2026, 22:11 IST, before any run)

**Status.** Event and factor studies; no change to any book (RA1F). N = 8 here. A
pass earns a stage 2, written and timestamped before it runs (an engine run under
the diversifier gate, or minute data for H4 and H7), never adoption. A fail leaves
no code; the result stays in the tracker.

**Already decided, not re-run.** Pairs with a short leg: not viable below Rs 2
crore (U28, RSK1; backlog COINT1), so only the long leg is tested (H2). SUE
earnings drift is GT7 (5aa), withdrawn 5 Nov; H3 is the price-reaction form, a different
signal. DT1 (5aq) failed a 20-close breakout and RSI(14) < 30; H1 adds the
compression condition and H5 uses a 20-session z and a 5-session hold.

**Data and window.** The deployed configuration (679cbd0c), adjusted data loaded
from 2006-01-02 to its end (2025-12-31; 2026 is not loaded), the point-in-time
universe mask (`compute_universe_panel`), sector snapshots (SB2), results
`available_from` (filings from 2012), and run `20261017T110547431678Z_679cbd0c`
for the book's entries. Gates are read on 2013-25 events; 2008-12 is reported.
Nothing after the decision day t is read for a signal.

**Swing events (H1, H2, H3, H5).** A universe member on t; once per stock per
horizon. Fill at the open of t+1, horizon H sessions open to open; a name that
stops trading exits at its last close. Excess over the equal-weight universe of
t over the same window, net of 34.6 bp a round trip (statutory on Rs 25,000 a
side plus 3 bp spread each way, as 5ar). Daily mean of the day's events,
Newey-West lag H.
- H1 volatility compression breakout (new): bandwidth sd20/mean20 of close at t-1
  at or below the 10th percentile of its previous 250 sessions (200 valid), and
  close[t] above the highest high of t-20..t-1. H = 20.
- H2 pairs divergence, long leg (published: Gatev, Goetzmann & Rouwenhorst 2006):
  at each month-end, a stock's most correlated same-sector peer (250-session daily
  returns, correlation >= 0.8, 230 valid). Daily in the next month, z of
  log(P_a / P_b) over 60 sessions to t; event z[t] <= -2 < z[t-1]: buy the
  laggard a. H = 20. The peer's gross excess is reported (the untradeable leg).
- H3 earnings-announcement-return drift (published: Brandt, Kishore, Santa-Clara &
  Venkatachalam 2008): t0 = first `available_from` of a quarter's result; EAR =
  close[t0+1] / close[t0-2] - 1 less the universe's; event when EAR is at or above
  the 80th percentile of the EARs completed in the previous 250 sessions (200 of
  them); fill at the open of t0+2. H = 60.
- H5 z-score mean reversion (published short-term reversal: Jegadeesh 1990,
  Lehmann 1990): z = (close - mean20) / sd20 to t; event z[t] <= -2 < z[t-1].
  H = 5.

**H6 RSI rotation momentum (new, factor).** Each month-end, members ranked on
Wilder RSI(14); the top quintile, equal weight, bought at the next open and held to
the open after the next month-end. Monthly excess over the equal-weight universe,
less 25 bp a side on the top quintile's turnover; Newey-West lag 3. Top minus
bottom is reported.

**H4 gap continuation (new, intraday).** g = open[t] / close[t-1] - 1, 2% <= |g| <=
15%, member at t-1, the bar not locked (high above low). Trade in the gap's
direction from open[t] to close[t]: the signed return, net of 16.6 bp (Zerodha
intraday: brokerage 0.03% a side, STT 0.025% on the sell, exchange, stamp, SEBI and
GST about 10.6 bp, plus 3 bp spread each way). Raw, not hedged: the trade a book
would make; the excess over the universe's same-day move is reported. Daily mean,
Newey-West lag 5. A fill at the printed open is generous; a pass needs minute data.

**Close to open (published: Cliff, Cooper & Gulen 2008; Lou, Polk & Skouras 2019).**
Bought at the close of t, sold at the open of t+1, every session.
- H7 index: NIFTY 50's open[t+1] / close[t] - 1, the index opens from the call
  auction; net of a NIFTYBEES round trip on Rs 3 lakh (statutory 2.86 bp) plus 2
  bp spread each way. ETF opens in the bhavcopy are stray first prints (D4), so the
  index stands in; a pass needs the ETF's minute prices. Its intraday leg and buy
  and hold are reported.
- H8 stocks: the equal-weight universe's close[t] to open[t+1], net of 34.6 bp.
Raw net mean per night, Newey-West lag 5.

**Pass.** New ideas (H1, H4, H6): t >= 3 in the claimed direction. Published (H2,
H3, H5, H7, H8): t >= 2.58 (Bonferroni for five, two-sided 0.01). Both: the same
sign in 2013-19 and 2020-25, and at least 200 events (100 months for H6).

**The book's route (H1, H2, H5, H6), also a pass at |t| >= 3.** The book's fresh
entries with the signal in the 5 sessions to the decision (H6: in the top quintile
at the last month-end) against its other fresh entries; 20-session excess from the
fill's open, gross; Welch's t.

**Addendum (06 Nov 2026, 22:16 IST, before any H2 number existed).** The first run
gave H2 no events: the point-in-time sector snapshots begin 14 Sep 2026 (SB2), so
no same-sector peer exists in 2006-25. H2 is replaced, not re-tuned (N stays 8):
the peer is the most correlated universe member of any sector, same correlation
bar (>= 0.8, 230 valid of 250); everything else as above. The other seven results
were read before this note and do not change.

**Result (06 Nov 2026).** 2013-25, net, Newey-West t; `data/nse_engine/bt7/`.

| | Events | Mean | t | 2013-19 / 2020-25 | Verdict |
|---|---|---|---|---|---|
| H1 compression breakout, 20 sessions | 5,162 | -0.88% | -4.16 | -0.85 / -0.91 | FAIL |
| H2 pairs laggard (addendum), 20 | 545 | +0.14% | +0.24 | -0.48 / +0.87 | FAIL |
| H3 earnings-return drift, 60 | 2,637 | +1.22% | +2.39 | +0.54 / +2.06 | FAIL (bar 2.58) |
| H4 gap continuation, intraday | 45,331 | -0.86% | -22.86 | -1.09 / -0.60 | FAIL |
| H5 z-score reversion, 5 | 19,961 | -0.63% | -7.65 | -0.64 / -0.62 | FAIL |
| H6 RSI rotation, a month | 154 months | -0.24% | -1.21 | -0.04 / -0.47 | FAIL |
| H7 NIFTY close to open, a night | 3,207 | +0.050% | +4.48 | +0.06 / +0.04 | PASS, stage 1 |
| H8 stocks close to open, a night | 3,219 | -0.14% | -11.04 | -0.16 / -0.13 | FAIL |

The book's route: H1 -0.26% (t -0.23, 74 entries), H2 no flagged entry, H5 +6.0%
(t +2.38, 17 entries), H6 +0.11% (t +0.19): none passes. H4's gaps reverse: gross
-75 bp in the gap's direction for up gaps, -92 bp for down gaps, -117 bp for gaps of
3% or more. H2's peer leg earns +0.02% gross: no convergence from either side. H7:
gross 11.6 bp a night against 6.6 bp of cost (break-even 11.6), intraday leg -6.5
bp; net CAGR 13.0% against NIFTY's price return of 12.3%, Sharpe 0.65 against 0.40
(6.5% rate), volatility 9.6% against 16.2%, 11 of 13 years positive (2022 -9.4%),
correlation 0.57 with the same day's close-to-close move. Index data starts
Feb 2012, so 2008-12 is one year.

**H7's stage 2 (not run; needs data).** A pass on the index is not a tradeable
result: the close is the last 30 minutes' weighted average and the open comes from
the call auction, while a NIFTYBEES order fills at the 15:2x and 9:15 prices. Stage
2, pre-registered before it runs: NIFTYBEES minute prices (INT1A's vendor ask),
bought in the last minutes and sold in the first, at the real spread, then the
engine's diversifier gate against the deployed book, after tax (a nightly round
trip is likely business income). Until then nothing changes.

**Stage 2, BT7-S2 (pre-registered 07 Nov 2026, 00:45 IST, before any return from
the minute data was computed).** *Prior, missed when H7 was written:* Y3 (5s, 9
Oct) held NIFTY overnight on a sentiment condition, on the same index open and
close at futures costs: DSR 0.884 (bar 0.95), and beside the book +0.22 Sharpe at
p 0.15 (DG1, RT1): FAIL. H7 is its unconditional sibling. Stage 2 asks what Y3 did
not: whether the night survives NIFTYBEES's own prices, and whether the
unconditional, funded form passes the gate. No new trial (BT7's N stays 8).

*Data.* Kite minute candles, NIFTYBEES, Feb 2015 to Dec 2025, traded prices
only, split-adjusted by Kite (the 10:1 split of 19 Dec 2019 is in the levels).
Sessions with fewer than 200 candles (Muhurat, the 10 Jul 2017 outage) are not
trade days; a night runs from one trade day's close to the next one's open.

*Trade.* Buy at the volume-weighted typical price ((H+L+C)/3) of the 15:25-15:29
candles; sell at that of 09:16-09:20 on the next trade day (the 09:15 candle is
skipped, D4). A window that traded nothing uses its last candle's close; a
session missing a window drops that night, counted. The ex-dividend night of 18
Feb 2015 is credited Rs 0.80 a unit (Rs 8 before the split).

*Cost.* A NIFTYBEES round trip on Rs 3 lakh at dated statutory rates (as H7)
plus 3 bp of spread each way (H7 used 2).

*Data stop.* A night beyond +/-8% while NIFTY's own night moved less than 4%
stops the run: a price problem, not a result.

*S2a, tradeable.* Mean net a night > 0 with Newey-West (lag 5) t >= 2.58, the
same sign in 2015-19 and 2020-25, at least 1,000 nights. Reported: the index's
night on the same nights (stage 1's measure), the gap, the break-even cost.

*S2b, only if S2a passes.* The diversifier gate under GT13
(`diversifier_gate.evaluate`, both books, Sharpe up at p <= 0.10 on each). The
stream is a funded sleeve: the night's net return dated to the morning it is
sold, and nothing on idle cash (a Kite account pays none), so the gate's
liquid-rate credit for a sleeve is cancelled. DSR reported at N = 8.

*Reported, not gated.* After tax as business income: 30% of each year's net
gain, losses set off within the year.

A fail at S2a or S2b closes H7: the minute file and `data/nse_engine/bt7/` are
deleted; the tracker keeps the record.

**Stage 2 result (07 Nov 2026): S2a FAIL, H7 closed.** 2,687 nights (Feb 2015 to
Dec 2025; one dropped for a missing window; 15 short or special sessions not
traded). Gross +1.2 bp a night against 8.7 bp of cost: net -7.5 bp (t -6.12),
2015-19 -7.8 bp (t -6.02), 2020-25 -7.3 bp (t -3.71), every year negative (-4.4%
to -24.8%). On the same nights the index's night is +11.9 bp. Both add up to the
same session (NIFTYBEES +1.2 night + 3.8 day = +4.9 bp; the index +11.9 - 7.1 =
+4.8 bp), so the data agree and only the split differs: the index's overnight
return is an artefact of its closing average and auction open, not something an
order can earn. S2b was not run. Y3's Sharpe (5s) and any study of index or
bhavcopy opens against closes carry the same artefact (YZ1's split, H4's gap
reversal): INT1C must measure on minute prices.

## 5cb. GF1: the opening gap's fade on minute prices (India tracker GF1; INT1C idea 1 of 3; your go 07 Nov 2026; pre-registered 07 Nov 2026, 00:56 IST, before any data was downloaded)

**Why.** BT7's H4 (5ca) found stock gaps reversing 75-92 bp intraday on bhavcopy
prices, but stage 2 showed that bhavcopy opens and closes can manufacture such
splits. Who loses: an opening gap is set by overnight orders in the call auction;
if it overshoots, those orders pay the fade. This tests it on prices an order gets.

**Scope, under 5av's rules.** The test: NIFTYBEES, intraday (MIS), long or short;
no dead names, so no survivorship question. Stocks: an information look only,
never a pass route, because Kite's minute history holds no delisted names (5av:
single stocks join only with them). Intraday family N = 2 (both looks count).
2026 is not downloaded.

**Data (INT1A's checks, on Kite's free minute history).** Minute candles Feb 2015
to Dec 2025: NIFTYBEES, and for the look the universe members of
`compute_universe_panel` (deployed config) still listed on Kite. A session counts
only if its minutes reproduce the bhavcopy's high and low within 0.5% and volume
within 5%; one that fails is refused, never filled, and counted. Sessions under
200 candles (Muhurat, outages) are not traded. Stop and report if under 90% of
NIFTYBEES's sessions pass or fewer than 5 years remain.

**Rule.** Prior close C = the volume-weighted typical price ((H+L+C)/3) of
15:25-15:29 on the previous trade day; open O = that of 09:16-09:20 (the stray
09:15 print skipped, D4). Gap g = O / C - 1. Event: |g| >= 0.5% for NIFTYBEES,
>= 2% for stocks (as H4). Trade against the gap, filled on the next bar: entry =
the volume-weighted typical price of 09:21-09:25, exit = that of 15:10-15:14
(flat by 15:15). A window that traded nothing uses its last close; a session
missing a window is dropped and counted.

**Cost.** Zerodha intraday on Rs 3 lakh a side (NIFTYBEES: brokerage Rs 20 or
0.03%, whichever is lower; STT 0.001% on the sell side for an equity ETF;
exchange, SEBI, stamp 0.003% on the buy, GST 18%; stocks: STT 0.025% on the sell
side) plus 3 bp of spread each way. Robustness at 6 bp each way.

**Pass (5av, INT1C), NIFTYBEES only.** Net mean per event > 0 with Newey-West
(lag 5, on the daily series) t >= 3; the same sign in 2015-19 and 2020-25; net
Sharpe still positive at 6 bp each way; DSR >= 0.95 at N = 2; then the 5.1
diversifier gate beside the book (both books, `bootstrap_p` <= 0.10), the stream
a funded sleeve with nothing earned on idle cash, as BT7-S2. At least 200 events.

**Reported, not gated.** The follow (the same trade with the gap); the stocks'
look (net mean, t, both halves); the share of the bhavcopy-measured H4 reversal
that remains on minute prices.

**After.** A fail closes the idea (INT1C keeps 2 slots), and the data and code
are deleted.

**Note (09 Nov 2026, 11:18 IST, before any return was computed).** The bhavcopy
check's first run scaled volume by the split factor as well as price; Kite's
minute history adjusts prices for splits and bonuses but not volumes (NIFTYBEES:
minute high / bhavcopy high 0.1 before the 2019 split, minute volume / bhavcopy
volume 0.99-1.00 in every year). The check now compares split-adjusted prices
and raw volumes. The tolerances and the stop rule are unchanged.

**Amendment (09 Nov 2026, 11:48 IST, your call, before any return was computed).**
With volumes fixed, 88.9% of NIFTYBEES's full sessions passed (volume 99.6%, high
and low 89.3%), under the 90% stop line. In all 287 failures the bhavcopy's range
was wider than the minutes' (its low below every minute in 70%, by a median 1.17%;
its high above every minute in 33%); the minutes never traded outside it: stray
ETF prints in the reference (D4), not broken minute data. The high/low check is
now one-sided: a session fails if a minute bar trades more than 0.5% outside the
bhavcopy's high or low, or volume differs by more than 5%. Everything else as
pre-registered. A pass moves to INT1B (a proper simulator with next-bar fills,
circuit halts and look-ahead traps), then INT1D's 60 paper sessions; it is never
adopted from this look.

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
