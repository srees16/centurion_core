# Research protocol (GT10, 11 Oct 2026)

**What this is.** The rules every new return stream, signal or engine change
follows before it can touch a book. Their purpose is that a result can be
trusted, and that the golden target's PBO ≤ 30% leg survives continuous
research (GT12).

**Where it comes from:**
- the rounds run so far (O2–O9, GT13);
- plan section 5v (options);
- LN-T26 (one environment per trial set);
- `docs/golden_target_review.md` section 7;
- Bailey & López de Prado (the deflated Sharpe, PBO);
- Harvey, Liu & Zhu (t > 3);
- Arnott, Harvey & Markowitz (a backtesting protocol).

The stage gates for a book (plan sections 1–6) still apply; this protocol
governs research.

## 1. Families and budgets

Every trial belongs to one family. Each family has its own registry, so its
trials never enter another family's PBO or deflated-Sharpe matrix.

| Family | Registry | Budget and status |
|---|---|---|
| Core (the live engine and its variants) | `data/nse_engine/runs` | **Closed** except V3's forward gate: one configuration left before January. After V3, queued core tests (MV1, GT7, GT9, GT11) take **one configuration each**. |
| Options (U32) | `data/nse_engine/runs_options` | **At most 3 configurations per source**, one round each (O5, O6, …). N = 17 after O8; PBO 14.1%. Rounds needing bid/ask, calendars or weeklies wait for the OX bridge. |
| Equity sleeves | `data/nse_engine/runs_<sleeve>` (O9: `runs_sepa`) | At most 3 configurations per sleeve. A new sleeve gets its own registry. |
| Commodity (GT5, MCX futures) | `data/nse_engine/runs_mcx` | At most 3 configurations. N = 2 after GT5 (12 Oct, plan 5y: both FAIL, closed); the third stays unused unless a new source justifies it. |
| Value (GT4) | `data/nse_engine/runs_value` | At most 3 configurations. N = 2 after GT4 (12 Oct, plan 5z: both FAIL, closed); the third is kept for a beta-hedged variant once capital covers a NIFTY lot. |
| US core (US5, the India rules on US stocks) | `data/nse_engine/runs_us` | At most 3 configurations. N = 3 after round 1 (17 Oct, plan 5ad: all FAIL against SPY, closed). The next US idea starts its own family. |
| Bonds (BD1, gilts) | `data/nse_engine/runs_bonds` | At most 3 configurations. N = 3 after BD1 (17 Oct, plan 5ag: all FAIL, closed). The cost of trading debt ETFs was modelled, not measured; a low-churn round would need your call. |
| Leverage (L7, real stock futures) | `data/nse_engine/runs_leverage` | At most 3 configurations. N = 3 after L7-C (17 Oct, plans 5ae and 5af: all FAIL against L6's synthetic, closed). Its 2026 holdout is used (own lock, `runs_leverage/holdout.lock`). |
| Event studies (e.g. GT13) | none; the verdict goes in the plan | The study's hypotheses count as its N (GT13: N = 2). A study that becomes a sleeve then starts a family. |

**Budget rules:**
- A family's PBO must stay ≤ 30%; aim for 10–20%.
- A source strong enough to test should get 1–2 configurations, not 3. Every
  extra configuration raises the bar for all of them.
- A failed configuration is never re-tuned. A variant is a new
  configuration within the budget, pre-registered like the first.

## 2. Before any number is read: pre-registration

Write a timestamped section in `docs/nse_engine_validation_plan.md`
containing:

1. **The hypothesis.** Why the return should exist: a risk premium, a
   behavioural bias, or a structural flow. A pattern alone is not a
   hypothesis.
2. **The source** and, for a book or paper, an intake note: its claims and
   the evidence for them, what can and cannot be made mechanical, and the
   data it needs.
3. **The exact rules and every parameter.** Use the source's own values.
   Any translation (an expiry horizon, a strike step, a missing threshold)
   is fixed here and marked (A).
4. **Data and window.** The 2013–25 test window; the 2007–25 window,
   reported only; 2026 is the sealed holdout, opened once per family.
   Anything headed for live money is also tested on 2008–25 (section 6,
   your rule). Where no 2008 data exists (stock futures before 2013, filings
   before 2012), the pre-registration states the gap, and going live
   without 2008 is your call. Each January the window rolls forward (RF1, from Jan 2027): the past year joins the test window and the new year becomes the sealed holdout. Also
   the store and its data hash.
5. **Costs and sizing.** Every instrument at its own current cost model,
   at its dated rates: `nse_engine.costs` for stocks and ETFs, `fno_costs`
   for F&O, and MCX's and bonds' own models (built with the stream if
   missing; your rule, 17 Oct). Size against broker-feasible capital, never
   at the maximum loss alone (OX3).
6. **Pass rules** (section 5) and what a pass leads to.

Rules that need intraday prices or a same-day exit are out of scope;
Centurion is swing and positional only.

## 3. Running

- **One environment:** Kaggle, for every recorded trial in a family that
  runs the engine. The manifest's runtime stamp and LN-T26's warnings check
  this. A family whose trials are plain arithmetic on recorded series (no
  engine decisions: leverage 5ae–5af, bonds 5ag) may run on the Mac, stated
  as (A) in its pre-registration, with all of its trials in one place.
- **Debugging runs:** short windows, never recorded. If one leads to a
  change, the change is logged (section 4). Precedent: the 5x sizing
  clarification after a 2014-only run.
- **Scoping looks:** arithmetic on returns already recorded, labelled "one
  look, not a trial", with nothing recorded. Examples: the leverage and
  factor-ETF scoping in the golden-target review.
- **Counting:** every recorded configuration counts in its family's N and is
  never deleted. A re-run of the same configuration on new data replaces it
  in the matrix (newest run wins). It is not a new trial.

### How new data is used

- **Daily:** live and paper decisions read every new close. The GitHub paper
  workflow re-syncs NSE before each run, so the regime gate, trends and
  rankings respond to new data through the rules themselves.
- **Weekly:** the research stores (equity, F&O, MCX, fundamentals) are
  refreshed by `runners/refresh_data.py` (DR1). Forward-gate checks,
  scorecards and diagnostics run on current data.
- **Quarterly:** every book is re-scored on the refreshed data (GT12).
- **Never:** new data does not re-tune a configuration. 2026 stays the
  sealed holdout until it is opened, once per family.

## 4. Changes after pre-registration

- **Before any run:** clarifications are allowed, dated in the plan, with
  the reason. Examples: O9's "largest fall since the base began", and 5x's
  sizing.
- **After a run:** only bug fixes. Each is dated, with its effect on the
  result stated. Examples: GT13's NIFTYBEES stray opens, and LN-T20's
  settlement re-baseline.
- **Never after a run:** change a parameter or a pass rule.

## 5. Gates

| Gate | Test | Applies to |
|---|---|---|
| **1, alone** | Deflated Sharpe ≥ 0.95 at the family's N, on excess returns over the family's risk-free convention (books 6.5%; options sleeves 0). | Every configuration |
| **New signal** | t ≥ 3 for a newly discovered signal. t ≥ 2 for the replication of a published effect (GT13). | Signals and event studies |
| **2, beside the live book** | Must raise CAGR, excess Sharpe **and** alpha against NIFTY 50 TRI, with MaxDD no more than 2 points worse. | Overlays: book + 0.25 × strategy. Sleeves funded from the core: 0.75 × book + 0.25 × sleeve. |
| **Out of sample** | Walk-forward 2017–25 for anything with fitted parameters. | Passers |

**Also reported:**
- the combined book's deflated Sharpe at the total trial count across
  families;
- the 2007–25 window;
- the trades and exit reasons.

**Before trusting gate 2's CAGR and alpha, check that marks are clean.**
Mark noise that swings −30% then +30% has a positive arithmetic mean, so an
overlay of it inflates both with no real profit (OP1). Sharpe is the
reliable check.

A low-Sharpe uncorrelated sleeve can fail gate 1 at a large N even when it
would help the book. Judging overlays on gate 2 plus a significance test is
your decision, parked at OX6.

### 5.1 The diversifier gate (your decision, 19 Oct 2026; settles OX6)

For a stream added beside the book to diversify it, this gate replaces
gate 2's "must raise CAGR" and gate 1's deflated Sharpe of 0.95. Your
words: minimising MaxDD matters as much as maximising Sharpe or CAGR.
- **Base:** the honest base, on two books where they exist, over 2008–25
  (or the stream's own span when its data starts later):
  1. the deployed rules' backtest with the drawdown rule;
  2. their walk-forward.

  Both use the current cost model. It is never the 2013–25 in-sample
  book.
- **Weight,** fixed before any result:
  - **Sharpe levers: equal risk.**
    - Each month, the stream is sized so that its trailing 252-session
      volatility matches the book's. Funded sleeves are (1 − w) × book +
      w × stream. Overlays, whose margin is pledged from the book, are
      book + w × stream.
    - Caps: a sleeve at 50% of capital; an overlay at 3×.
    - The stream's tested weight (e.g. 25%) is reported beside it.
  - **Drawdown levers: their own fixed sleeve weight** (your decision, 19
    Oct 2026). The weight must be written in the stream's pre-registration
    before any result: gilts 25% (5ag), MCX-B 25% (5y). Equal risk
    over-allocates a low-Sharpe stream and lowers the book's Sharpe.
- **Idle cash and collateral** earn the liquid rate (the 1D rate less a
  0.10% expense ratio), never nothing.
  - **Correction, 25 Oct 2026 (LN-T29d):** this credit is an upper bound. It
    is cost-free, same-day and at a 0.10% expense ratio. A realistic sweep
    (LIQUIDCASE at 0.27%, its trades at statutory charges plus 1–5 bp,
    BD1's hysteresis, cash freed a session before the buys) earns about 0.6
    point and 0.06 Sharpe less on the walk-forward book. Cost model 5
    (LN-T29) replaces this rule; until then, read honest-base figures as
    about that much high.
- **A stream passes by either route, on both books:**
  - **Sharpe lever:** combined excess Sharpe above the book's, with
    one-sided p ≤ 0.10 in a paired stationary bootstrap (63-session mean
    blocks, 2,000 resamples). MaxDD no more than 1 point worse.
  - **Drawdown lever:** at its own fixed weight, all three hold:
    1. MaxDD at least 3 points shallower;
    2. combined Sharpe no more than 0.02 below the book's;
    3. in the same paired bootstrap, MaxDD 3+ points shallower in at least
       80% of resamples. MaxDD is one path, so this checks it is not one
       lucky crash.
- **Two or more drawdown levers** are tested together, at their own
  weights, before they run together, and go to paper only as that tested
  set.
- **A pass whose 2026 holdout was already seen is provisional.** Its 60
  paper sessions and 5 dry runs are the fresh evidence, and paper must agree
  with its shadow backtest before anything else follows.
- **CAGR, alpha and the deflated Sharpe at the family's N** are reported,
  not gates.
- **Overfitting control is your five steps** (section 6): 2008–25, a
  walk-forward for anything fitted, and the 2026 holdout opened once with a
  check written down before it is opened. Then paper and dry runs.

## 6. Paper, then live

**Before real money (your rule, 17 Oct 2026).** A configuration goes live
only after these steps, in order, each passed before the next:
1. a backtest over 2008–25 on the current cost model;
2. a walk-forward over 2008–25;
3. the out-of-sample holdout, opened once;
4. paper for at least 60 sessions;
5. 5 dry-run episodes: the live path builds the real orders without
   placing them.

Every data store a stream uses reaches back through 2008, so the MaxDD of a
repeat crisis is measured, not guessed.

1. **Paper:** a passer is paper-traded for at least 60 sessions (options:
   and at least 6 cycles), tracked against a shadow backtest G4-style:
   tracking error, cost ratio, drawdown.
   - **Judge paper on tracking, not Sharpe.** It takes about 2 years of live
     data to show a Sharpe is above 0, and decades to tell 1.2 from 0.8.
2. **Planning live:** expect 0.6–0.7× the backtest Sharpe for a new sleeve
   (published decay: McLean & Pontiff; live CTAs; NSE factor indices after
   launch), and 0.85× for the core (D2).
3. **Live:** at small size, under the kill switch and the live risk limits.
   A drawdown beyond the simulated 95th percentile triggers a review.
4. **Leverage:**
   - futures only.  Your call, 17 Oct 2026: MTF was reconsidered and
     dropped, because its 14.6% interest lowers CAGR and Sharpe, while
     futures barely do (L6, L11).
   - up to 1.75x, accepted with a MaxDD of about −40% (17 Oct);
   - at most half-Kelly on a haircut Sharpe of 0.7;
   - only after 12+ months of live tracking (GT8).

## 7. Never

- book-level volatility targeting (scoped: it hurts this book);
- naked option selling outside a defined-risk structure;
- more price signals inside the core (plan 6, item 5);
- re-tuning a failed configuration;
- committing third-party sources (books and papers go in the gitignored
  `research_inbox/`).

## 8. Recording a result

- **Headline numbers (your rule, 17 Oct 2026):** lead with the honest
  base: 2008–25, walk-forward, current cost model, real instruments, after
  the live haircut. Anything rosier (in-sample, a shorter window, a
  synthetic model) is labelled as such.
- **Discarding a stream:** only after a recorded test. A desk study or a
  regulatory bar is labelled as such, never as "tested", with what would
  reopen it.
- **The plan section's result:**
  - a table;
  - what the runs say, in plain words;
  - diagnostics, labelled as not trials;
  - what carries forward.
- **The tracker:** a DONE entry with the numbers; new IDs for follow-ups.
- **Scorecards:** only when a book's configuration or data changes.
- **The metrics journal:** those changes as backtest and walk-forward rows,
  and every concluded experiment that changed no book (FAIL, CLOSED or
  NO-GO) as an experiment row, with the whole book's 2013–25 numbers when
  the test gave them:
  `python -m nse_engine.journal experiment --ref <ID> --date <verdict date>
  --verdict <FAIL|CLOSED|NO-GO> --event "<what was tried>" --note "<the
  result in plain words>"`.

## Pre-registration template

```
## 5?. <title> (<tracker ID>; pre-registered <date, time IST>)
Source / hypothesis: <why the return exists>
Family and budget: <family>, <k> of <max> configurations; N before -> after
Data and window: 2013-01-01..2025-12-31 (test); 2007-25 reported; 2026 sealed
Rules: <exact, with every parameter; translations marked (A)>
Costs and sizing: <cost model; broker-feasible sizing>
Pass rules: gate 1 <...>; gate 2 <...>; then <paper / walk-forward>
```
