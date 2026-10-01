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
the quantities it bought. Stops are reconciled only for those symbols.
Holding an engine symbol personally as well is not supported.

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
  book's G4 is PASS with at least 60 sessions and five scheduled dry runs
  finished clean; `CENTURION_GO_LIVE_OVERRIDE=true` overrides and the email
  says so. The Kaggle token (U2) and leverage (L4, NO-GO) stay manual checks.

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
| Promote (forward gate) | `python -m runners.run_nse_engine promote --check` prints the gate; without `--check` it writes the deployed file when the gate passes. Needs `CENTURION_DATABASE_URL` and the local store |
| Shift reference | `python -m runners.run_nse_engine shift-reference --run-id <id> --start <paper start> --data-start 2012-01-02` (also writes `<out>_trades.csv` for the gate's cost check) |
| Paper gate (G4) | `python -m runners.run_nse_engine paper-gate` for the deployed book; add `--deployment config/nse_engine_<book>.json --schema <book>` for another paper book, e.g. `candidate` or `e4`. Needs `CENTURION_DATABASE_URL` and a store covering the paper sessions, or `--reference <returns csv>` |
| Haircut (D2) | part of `validate`: two unrecorded re-simulations (as recorded; 1-day lag + 2× impact), the difference applied to the recorded metrics, plus the CSCV selection ratio; `validation.json` → `haircut` |
| Inspect deployment | `python -m nse_engine.deployment show` |
