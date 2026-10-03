# Paper books, promotion and go-live

As of 4 Oct 2026. The operating reference for the books: what runs, how a
trial earns promotion, how you choose between trials, and what go-live
needs. Sources: `ACTION_TRACKER.txt`, `nse_engine_validation_plan.md` (both in `docs/`),
`nse_engine/forward_gate.py`, `nse_engine/paper_gate.py`,
`nse_engine/capital_ladder.py`, `docs/books_register.csv`.

## 1. The books today

Three paper books, ₹35 lakh each, no real money. One configuration of the
same engine per book, each in its own Neon schema, traded every weekday at
19:00 IST by GitHub Actions on the same data.

| Book | Config | Status | Paper since | Role | Decision |
|---|---|---|---|---|---|
| deployed | `679cbd0c` | approved | 16 Sep 2026 | the configuration that goes live; rehearses for go-live | go-live ~mid-Dec |
| candidate | `2d64ba4c` | candidate | 28 Sep 2026 | trial under the forward gate | ~24 Dec (60 sessions) |
| e4 | `93cf6c4d` | candidate | 5 Oct 2026 | trial under the forward gate | ~early Jan (60 sessions) |

Only the deployed file can ever be traded live. A candidate file is refused
by the live path, whatever its results.

## 2. What each book is, and its numbers

All backtests 2013–25 on the same data and cost model (cost model 3, the
registry's like-for-like runs). Sharpe is excess over 6.5%; CAGR on
calendar years. Walk-forward OOS is the Sharpe over the test years 2017–25
of the same run.

| | deployed `679cbd0c` | candidate `2d64ba4c` | e4 `93cf6c4d` |
|---|---|---|---|
| What it is | Grid pick: no low-vol signal, 6-ATR stop, 5-day rebalance, neutral scale 1.0, 20 positions | B1 baseline with `regime.scale_neutral` 0.6 (the K5 walk-forward's last-fold choice) | B1 with exit rank 60 and refill exits |
| Why it exists | Approved 15 Sep (PBO 23.9% at n = 36 then) | In the neutral regime it holds less: lower volatility for a little CAGR | Exit rank 60 passed out of sample (R12: MaxDD 20.9% → 17.2%); refill keeps the book invested |
| CAGR | 23.7% | 22.6% | **24.6%** |
| Excess Sharpe | 1.206 | **1.213** | 1.205 |
| MaxDD | −23.1% | **−23.0%** | −24.6% |
| Calmar | **1.03** | 0.98 | 1.00 |
| Turnover / yr | 6.89× | 6.72× | **5.79×** |
| Cost drag / yr | 2.8% | 2.7% | **2.4%** |
| Walk-forward OOS Sharpe 2017–25 | 1.38 | 1.43 | **1.48** |
| PBO (n configurations) | 48.6% (58) | 48.6% (58) | 56.1% (60) |
| Deflated Sharpe | 0.989 | 0.989 | 0.988 |
| After the execution haircut (D2) | 1.00 / 20.2% / −22.8% | not measured | 1.08 / 22.2% / −26.5% |
| Other evidence | Holdout Jan–Sep 2026: Sharpe 1.04, MaxDD −13.0% | 2008 crash (R4): −26.9% | 2007–25: 21.1% / 0.97 / −39.1%; missed its 25% CAGR rule (U26) |

Reading the table: the three are within 0.01 Sharpe of each other in the
backtest. The differences that matter are out of sample (e4 1.48 vs 1.38)
and in costs (e4 trades a third less). PBO near 50% is a property of the
trial set (near-duplicate configurations), not of any one book; it is
reported, never gating.

## 3. How a book is run and judged

- **Daily session** (19:00 IST, weekdays): fills yesterday's orders at the
  open, marks to the close, plans from the close, queues orders. Email per
  book: `[Centurion Paper] [<book> <config>] ...`.
- **G4 paper gate**, computed every session against a backtest of the same
  days with the same configuration and capital: does the book behave like
  its backtest? Five checks, limits fixed on 28 Sep 2026 and never tuned
  on results:

| Check | PASS | FAIL |
|---|---|---|
| tracking error (paper − backtest, annualised) | ≤ 8% / yr | > 12% / yr |
| daily gap (mean, bp / day) | ≥ −3 bp | < −3 bp with t ≤ −2 |
| costs (paper cost per rupee traded ÷ backtest's; needs ≥ 20 fills) | ≤ 1.5× | > 2.0× |
| drawdown (paper MaxDD ÷ backtest's, same days) | ≤ max(1.5×, +2 pts) | > max(2×, +4 pts) |
| regime break (sessions sized down ≤ 0.5 in the last 20) | none | any |

  Between the limits a check is WATCH. Below 30 aligned sessions the verdict
  is NOT ENOUGH DATA; otherwise FAIL if any check fails, PASS only if all
  five pass, else WATCH.
- **Saturday 07:30 IST**: the weekly report per book, then one *books*
  email: every book over its record, each trial against the deployed book
  over their common sessions, the three forward-gate checks per trial, and
  the promotion review of any trial that has cleared them. The register
  `docs/books_register.csv` is attached.

Paper proves behaviour, not the edge. Over 60 sessions a Sharpe estimate
has a standard error of about 2, so no return threshold gates anything.

## 4. The three checks a trial must hold (forward gate, V3 / U19)

| # | Check | Rule | How it shows |
|---|---|---|---|
| 1 | Paper sessions beside the deployed book | ≥ 60 sessions, and the deployed book ran ≥ 60 of the same sessions | PENDING until 60 |
| 2 | Its own G4 | PASS on its latest session (not stale) | PENDING while NOT ENOUGH DATA or WATCH; FAIL on FAIL |
| 3 | Walk-forward OOS Sharpe 2017–25 | ≥ the deployed configuration's − 0.05, from the recorded runs | PASS / FAIL, known today |

All three must be PASS. Reported beside them, never gating: PBO, deflated
Sharpe, the benchmark gate, the holdout, and the paper returns of both
books over their common sessions.

Today: candidate 1.43 vs 1.38 and e4 1.48 vs 1.38, so check 3 passes for
both; checks 1 and 2 are a matter of time (candidate ~24 Dec, e4 ~early Jan).

## 5. Choosing the best candidate

Nothing is promoted automatically. The gate says who is *eligible*; the
choice is yours, from the promotion review in the Saturday email.

1. **Eligible** = all three checks PASS. A trial that fails check 2 or 3
   is out for this round; check 1 only needs time.
2. **If one trial is eligible**, decide on its review: the backtest table,
   the walk-forward OOS Sharpe, and the paper comparison.
3. **If several are eligible**, rank in this order, each a tie-break for the
   one before:
   1. walk-forward OOS Sharpe 2017–25 (the evidence that outlives the paper
      sample);
   2. backtest Calmar at equal or better Sharpe (CAGR per unit of
      drawdown);
   3. turnover and cost drag (lower is more robust to real fills);
   4. the paper comparison, only when its t statistic is beyond ±2.
4. **What the paper comparison can and cannot say.** The books email gives
   each trial's return minus the deployed book's over their common sessions
   and the t of the daily differences. |t| ≥ 2 means the gap is unlikely to
   be noise; anything smaller is noise and must not decide. Alpha against
   NIFTY 50 is shown for context, not for ranking.
5. **Do not re-tune.** A trial that misses is closed, not adjusted; a
   changed configuration is a new trial under the budget (U27: at most 2
   new configurations before the January decisions; 1 left).

## 6. Promotion: what you do and what it changes

- **See the gate** from the research machine:
  `python -m runners.run_nse_engine promote --check --candidate config/nse_engine_<book>.json --schema <book>`.
  Nothing is written.
- **Promote**: the same command without `--check`. It rewrites
  `config/nse_engine_deployed.json` with the trial's configuration (status
  approved, paper start = today) for your review and commit. `--force`
  overrides a failed gate and records that in the file's notes.
- **What changes**: the deployed book trades the new configuration from its
  next session and keeps its Neon record; the live book (once live) trades
  it too, because the live path loads only the deployed file. The deployed
  book's same-period reference restarts at the promotion date, so its G4
  session count restarts.
- **Open decision (3 Oct)**: because of that restart, a promotion around
  24 Dec would push go-live from mid-December to about March. The trial's
  60 sessions are the evidence go-live wants, so the proposal is (a) go live
  mid-December on `679cbd0c` as planned, (b) treat a promotion as a
  separate event timed just before a ladder step-up, and (c) let the
  go-live and ladder checks read the promoted configuration's trial-book
  record while the deployed book's own record is younger than 60 sessions,
  after verifying how the ladder behaves in the first 30 sessions after a
  configuration change. Pending your answer.

## 7. Go-live checklist (D3)

Before the first real session, all of these:

- [ ] Deployed paper book: G4 **PASS** with **≥ 60 sessions** (from 16 Sep → ~mid-Dec).
- [ ] **5 clean live dry runs** (`CENTURION_LIVE_MODE=dry_run`): token, tunnel, egress IP, broker reads and order building all worked, no alert. The first was scheduled for 1 Oct.
- [ ] **Daily Kite login** on every trading day (U23): tap the link in the 09:00 / 17:30 IST email; the token lasts until 06:00 next day. A missed login is a missed session.
- [ ] **Static IP**: orders go through the registered proxy (`CENTURION_KITE_PROXY`, checked against `CENTURION_KITE_STATIC_IP`); mandatory for API orders since April 2026.
- [ ] **Funding**: ₹6 lakh in the account about a week before.
- [ ] **Switch**: set `CENTURION_LIVE_CAPITAL=600000` and `CENTURION_LIVE_MODE=live` (repository variables). `CENTURION_GO_LIVE_OVERRIDE=true` bypasses readiness and says so in the email; not for normal use.
- [ ] Manual checks that stay manual: Kaggle token (U2), no leverage (L4).

**The capital ladder** (U6, U22), judged every live session on the live
book's own G4, the drawdown rule and NIFTY:

| Rung | Capital | Share |
|---|---|---|
| 1 | ₹6,00,000 | 20% |
| 2 | ₹12,00,000 | 40% |
| 3 | ₹21,00,000 | 70% |
| 4 | ₹30,00,000 | 100% |

- **GO** (one rung up, only when you ask): ≥ 20 sessions at the rung;
  tracking error, daily gap, drawdown and regime-break checks PASS (or not
  measurable yet); drawdown rule "normal"; no kill criterion. The step
  happens only when you set `CENTURION_LIVE_CAPITAL` to the next rung after
  the money is in; the email says when that is allowed.
- **STEP DOWN** (automatic, one rung): after ≥ 20 sessions at the rung, any
  G4 check at FAIL. A market-wide crash the backtest also suffers is not a
  reason; behaving unlike the backtest is.
- **KILL** (your decision, alerted, never automatic): drawdown > 1.5× the
  backtest MaxDD *and* NIFTY's drawdown over the same days; regime break in
  two consecutive sessions; G4 cost check FAIL (costs > 2× the model).
  Response: `CENTURION_KILL_SWITCH=true`, which refuses new buys.
- **Drawdown rule** on the book's own equity (deployed overlay): halt new
  entries beyond 20% from the episode peak, halve exposure beyond 30%, cash
  beyond 35%; re-arms on a 60-session high.

**Hold the live book to**: tracking error ≤ 8% / yr, Sharpe ≥ 1.0,
CAGR ≥ 20%, MaxDD ≤ 30% (soft, U5; not in a 2008-type crisis, U22),
Calmar ≥ 0.7, realised costs ≤ 1.5× the model. Realistic expectation after
the haircut: Sharpe ~1.0, CAGR ~20%, MaxDD ~23%.

## 8. Where to look

| Need | Where |
|---|---|
| A book's day | its daily email; `/ind-stocks/trade-monitor?book=<book>` |
| All books, gate status, promotion review | Saturday "Paper books" email |
| Every book's configuration and scores in one place | `docs/books_register.csv` (rebuild backtest columns: `python -m nse_engine.books register`) |
| A review without waiting for Saturday | `python -m nse_engine.books review --book <book>` |
| The gate from the research machine | `python -m runners.run_nse_engine promote --check ...` |
| Why a rule is what it is | `docs/nse_engine_validation_plan.md` (pre-registrations and results) |

## 9. Adding a book

1. Record its 2013–25 backtest in the registry on the current cost model
   (Kaggle, `cloud.kaggle_local`) and run `validate` on it.
2. Write `config/nse_engine_<book>.json`: `status: candidate`, the
   `source_run_id`, a `paper_start_date`, a one-line `description`, `notes`.
3. `python -m nse_engine.books register`; commit both files.

The daily job starts the book on its date, the Saturday email includes it,
and after 60 sessions `promote --check` shows its gate. Each new book is a
pre-registered configuration under the trial budget; its pass rule is
written down before its first run.

## 10. Rules that do not bend

- Pass rules are fixed before a run; a miss closes the trial without re-tuning.
- Every backtest is recorded and counted (PBO and the deflated Sharpe see all of them).
- The paper sample proves behaviour (G4), never the edge.
- Nothing promotes or goes live on its own; both are your hand-run steps.
- The live path trades only the approved deployed file, from the registered IP, after that day's login.
