# Process improvement loop: plan (PL0, 14 Nov 2026)

A plan; items are built only on the owner's approval of their ids. **Built 14 Nov 2026: PL-01 to
PL-08, PL-12, PL-13 and PL-15** (sections 9 and 10). Scheduled: PL-14 after the sell test; PL-09 and PL-20
after RA1 (15 Dec); PL-10, PL-11, PL-16 and PL-17 to be planned on 16 or 17 Nov. PL-18 and PL-19 wait. Scope: the research
**process** (data, tests, costs, monitoring, hygiene, memory). Strategy Sharpe and CAGR are
not loop metrics, and the loop never tunes a parameter, signal or filter.

Owner's choices (14 Nov): design it first around **skipped steps and wrong records**; the
lessons ledger is **generated from the guards**, not kept by hand; counts are keyword
matches checked on a read sample.

## 0. Where things stand (read 14 Nov)

- **Most of the six areas exist.** `docs/research_protocol.md` has families, budgets and a
  pre-registration template. Trials are recorded per family (`validation/trials.py`).
  Guard tests exist for causal signals (6), survivorship (23), universe look-ahead (5) and
  rule D (5). CI runs the fast suite under a coverage ratchet. The checklist (CK1), the MC
  bands, the paper gate (G4) and the Saturday email exist.
- **Trials and PBO by family:**

  | Family | Configurations | PBO | State |
  |---|---|---|---|
  | Core | 61 (62 after J1) | 29.2% (cap 30%) | Closed until January |
  | Options | 20 | 14.1% at N = 17; not recomputed at 20 | 3 per source |
  | Commodity, value | 2 each | - | Closed |
  | US core, bonds, leverage | 3 each | - | Closed |
  | Intraday | 2 trials; 2 of 3 idea slots left | - | Open |
  | Indian event studies (NS1) | 4 studies, 0 passed | - | No registry |

- **Verdicts:** the metrics journal holds 65 experiments, 5 of them pass-type. That is the
  gates working, not a process fault.
- **The real defect is steps done by hand that get skipped:**
  - the protocol requires every verdict in the metrics journal; its last experiment row is
    dated 1 Nov, and none of 22 studies concluded 2-13 Nov is in it;
  - the protocol's own family table is stale (options "N = 17", intraday "N = 0");
  - three copies of the owner's rules had drifted apart (fixed 14 Nov, with a test).

## 1. Top 10 recurring failure patterns

Counted over the 341 completed tracker entries (16 Sep - 14 Nov). "Matches" are keyword
hits; "about" is after reading a sample of 8-14 matches per pattern. Treat every count as
approximate.

| # | Pattern | Matches | About | Examples |
|---|---|---|---|---|
| F1 | A bug in our own code found later, not by a test | 37 | 26 | NEO6 (replay stamped the wrong epoch), DQ1 (unparsable file skipped silently), FL1 (backup gate ignored the futures leg), HB2 (heartbeat counted weekdays) |
| F2 | A scheduled job late, failed or falsely alarmed | 42 | 21 | NEO1 (quota burn), crons 6-10 hours late, LN-T33 (late cron met an expired login), HB2 false alarm |
| F3 | A required record step skipped, or records out of step | 8 + 22 unjournaled verdicts | 30 | Journal empty since 1 Nov; protocol table stale; "GT7 queued" after it was withdrawn |
| F4 | A wrong or stale fact stated, then corrected | 25 | 15 | PG1 (audit on a 10 Oct register), "14 Dec" for a 15 Dec verdict, pre-registration times stamped in the future, FL3 (wrong precedent) |
| F5 | A bias caught after the fact | 32 | 11 | B1b (back-adjustment look-ahead), GT2B (in-sample gain gone in walk-forward), BT7-S2 (open-to-close artefact) |
| F6 | A data fault in a store or source | 15 | 10 | Dhanteras session shifting every rebalance (CAN1), relisting read as a split (CK1-P2), stray ETF prints (D4), mis-scaled NSE result pages |
| F7 | A study stopped by data after work had begun | 11 | 9 | NS1-4 (spin-offs unnamed), NS1-8 (SEBI refused scripted access), NS1-10, BLK1, GT11 (no list history) |
| F8 | A cost, margin or funding assumption wrong or unmeasured | 26 | 9 | A1M (margin 22% above the paper assumption), sale credit not usable at the open (ST2), bond costs modelled not measured |
| F9 | An idea reviewed that was already tested or in place | 15 | 8 | H7 of BT7 repeated Y3; MPT2, FBR1, RB1 ("nearly all already in Centurion") |
| F10 | Environment or version drift | 53 | 8 | `kiteconnect` 5.2.2 on the app against the tested 5.1.0; 99 of 226 packages differing; results not reproducing across platforms |

Smaller: a human step missed or repeated (about 5: the sell test re-dated four times),
a source's own numbers failing replication (about 5), leftover files returning after a
failed study (4: the empty files that came back twice).

## 2. Action items

Every item uses 0 trials. Effort is hours of work; compute is on the Mac unless stated.
Risk is to existing behaviour. Priority follows the owner's choice: records first.

| Id | Area | Problem (pattern) | Guard | Files touched | Effort | Compute | Risk | P |
|---|---|---|---|---|---|---|---|---|
| PL-01 | 5 | Verdicts never reach the journal (F3) | `journal check`: every tracker entry with a verdict since a cut-off has a journal row; a test fails otherwise. Backfill the 22 missing rows | `nse_engine/journal.py`, `tests/test_metrics.py`, `docs/metrics_journal.csv` | 3 | none | Low | P1 |
| PL-02 | 6 | Lessons live in prose and memory (all) | Guards carry a lesson id (`@pytest.mark.lesson`); `tools/lessons.py` builds `docs/lessons.jsonl` from them and fails when a lesson has no guard | `tools/lessons.py` (new), `pytest.ini`, guard tests | 5 | none | Low | P1 |
| PL-03 | 5 | Family counts kept by hand and stale (F3) | One file of budgets; counts read from the registries; a test fails when the protocol's table disagrees | `config/research_families.json` (new), `validation/trials.py`, `tests/test_metrics.py`, protocol table | 3 | none | Low | P1 |
| PL-04 | 6 | Answers and documents built on stale facts (F4) | `tools/preflight.py`: at the start of a session prints computed facts (dates in IST and UTC, last sessions, RA1 count and date, family counts, open owner items, journal gaps) | `tools/preflight.py` (new) | 4 | seconds | Low | P1 |
| PL-05 | 1, 5 | Studies stopped by data, or repeating old ones (F7, F9) | A mandatory Phase 0 block in the pre-registration; `tools/prior_art.py` searches tracker, plan and journal for earlier verdicts; a check refuses a section without Phase 0 filled | protocol template, `tools/prior_art.py` (new), test | 4 | none | Low | P1 |
| PL-06 | 5 | No record of cost per verdict (metrics) | Post-mortem fields on each journal row: phase stopped, hours, trials, compute, earliest possible stop, lesson id | `nse_engine/journal.py`, journal columns | 3 | none | Low | P1 |
| PL-07 | 5 | Failed-study files return (leftovers) | A test: no empty or untracked source file under the code folders; a study manifest listing what to delete, shown for approval | `tests/test_config_identity.py`, `tools/study.py` (new) | 2 | none | Low | P1 |
| PL-08 | 5 | Pre-registration times typed by hand, some in the future (F4) | `tools/prereg.py new` stamps the clock and fills the family count from PL-03 | `tools/prereg.py` (new) | 2 | none | Low | P2 |
| PL-09 | 5 | Trial counting is a convention | `record_result` refuses a run past its family's budget unless a dated decision id is given | `validation/trials.py`, tests | 3 | none | Medium: the research recording path | P2 |
| PL-10 | 1 | No single view of what data exists | A generated catalogue: each store's first and last date, rows, hash, known gaps, freshness | `nse_engine/data/catalogue.py` (new), `docs/data_catalogue.json` (generated) | 5 | 1-2 min | Low | P2 |
| PL-11 | 1 | Store faults found by accident (F6) | `store doctor`: calendar against NSE's, special sessions, zero-volume rows, unexplained-gap count against last week | `nse_engine/data/validation.py`, nightly step | 4 | 1 min a night | Low; reads only | P2 |
| PL-12 | 2 | Bugs fixed without a regression test (F1) | Audit the about 26 own-bug entries; each gets a test or a dated "no guard possible" note; the ledger then tracks the share | tests, ledger | 3 + tests | none | Low | P2 |
| PL-13 | 2 | Store-dependent guards do not run in CI | A `guard` marker and one command run before every handover and weekly on the Mac | `pytest.ini`, guard tests | 2 | minutes | Low | P2 |
| PL-14 | 3 | Cost model never compared with a contract note (F8) | Parse Zerodha contract notes; compare each fill's charges with `costs.statutory_cost`; drift over a threshold opens a ledger item. First fixtures: the INFY buy of 11 Nov and the sell test | `nse_engine/validation/cost_margin.py`, tests | 4 now; rest with X1 | none | Low | P2 now, P1 at go-live |
| PL-15 | 3 | Rate and tax changes found late | A dated regulatory table with the source of each change; a test that today's rates are the latest dated ones | `nse_engine/costs.py`, tests | 2 | none | Low | P2 |
| PL-16 | 4 | Incidents reach the tracker only when someone notices (F2) | A weekly incident list built from Neon and the Actions API: missed logins, deferred legs, late and failed runs, alerts sent; an incident type without a lesson id is flagged | `tools/incidents.py` (new), Saturday email | 6 | 1 min a week | Low; reads only | P2 |
| PL-17 | 4 | Loop health not measured | `tools/loop_health.py` computes section 4's metrics from the journal, ledger and incidents; one block in the Saturday email | new tool, `nse_engine/books.py` | 4 | seconds | Low | P2 |
| PL-18 | 6 | A new task repeats a known pattern | `preflight --task "<text>"` lists matching ledger patterns and earlier verdicts. Advisory only | PL-04, PL-05 | 2 | none | Low; false matches | P2 |
| PL-19 | 5 | The queue is ranked by feel | Priors generated from the journal by idea type and market, shown with their sample size; they rank and never veto | `config/research_priors.json` (generated), PL-06 | 3 | none | Low | P3 |
| PL-20 | 2 | Environment drift (F10) | The lock in the nightly workflow's legacy install (after RA1) | workflow | 1 | none | Medium: the nightly workflow | P3 |

Total: about 65 hours. P1 alone: about 24 hours, none of it touching the engine, the
deployed book, `live_session.py` or any configuration.

## 3. Schemas and templates

### 3.1 Lessons ledger (`docs/lessons.jsonl`, generated; one line per lesson)

```
{"id": "L-FL1", "date": "2026-11-14", "source": "FL1", "pattern": "F1",
 "summary": "The backup-run gate ignored the futures leg",
 "guard_type": "test", "guard_location": "tests/test_paper_heartbeat.py::test_a_futures_leg_not_marked_tonight_keeps_the_backup_runs_on",
 "status": "guarded", "recurrence_count": 0, "last_seen": "2026-11-14"}
```

- `guard_type`: `test`, `preflight`, `config_guard`, `template_field`, `prior`, or `none`.
- `status`: `guarded`; `open` (no guard yet); `waived` (with the owner's dated decision id).
- `recurrence_count` rises when a new tracker entry names the lesson id as recurring.
- The file is never edited by hand. A test regenerates it and fails on any difference, and
  on any `guard_location` that no longer exists.

### 3.2 Pre-registration template (extends the protocol's)

```
## 5?. <title> (<tracker ID>; pre-registered <stamped by the tool, IST>)
Phase 0 (before anything else is written or downloaded):
  Prior art: <ids of earlier tests and verdicts from prior_art, or "none found">
  Data: <source>; reachable by script and allowed by its terms: yes / no
        coverage 2008-25: <from - to, gaps>; point-in-time: yes / no; delisted names: yes / no
  Cost model for this instrument: <exists, version> / <to be built>
  Phase 0 verdict: GO / STOP (<reason>)
Source / hypothesis: <why the return exists>
Family and budget: <family>, <k> of <max>; N before -> after (filled from the registry)
Data and window: 2013-25 test; 2008-25 for live money; 2026 sealed
Rules: <exact, every parameter; translations marked (A)>
Costs and sizing: <cost model; broker-feasible sizing>
Pass rules: gate 1 <...>; gate 2 <...>; then <paper / walk-forward>
Stop rules and cap: <hours, downloads, what ends it early>
Files this study will create: <paths; deleted with a FAIL after approval>
```

### 3.3 Post-mortem (fields on the verdict's journal row)

```
ref, date, verdict, idea_type, market,
phase_stopped  (0 feasibility | 1 pre-registration | 2 run | 3 gate | 4 paper),
hours, trials_used, compute (Mac minutes / Kaggle hours), data_downloaded_mb,
what_decided_it  (one line),
earliest_possible_stop  (the phase at which it could have been stopped, and by what check),
lesson_ids  (ledger ids created or recurring; "none: <reason>"),
cleanup  (done | approved list pending)
```

## 4. Loop health metrics

| Metric | Baseline today | Target |
|---|---|---|
| Repeat rate of guarded patterns | Not measured; 12 of 13 patterns recurred 5+ times in 60 days | 0 recurrences of a guarded lesson; every recurrence raises the count |
| Share of lessons captured as guards | Not measured; the ledger does not exist | 90% within 7 days of the entry; 100% on the live path |
| Verdicts recorded in the journal | 0 of 22 since 1 Nov | 100%, enforced by a test |
| Median hours and trials per verdict | Not recorded | Recorded from PL-06; then 25% fewer hours on desk studies in a quarter |
| Studies stopped at Phase 0 | About 9 data-blocked studies began work first | 90% of data-blocked or duplicate studies stop with nothing downloaded |
| Cost-model error | Unmeasured (a handful of real fills) | Charges within 2% of the contract note; slippage inside the haircut, from 30 fills |
| Paper and live tracking | G4 PASS at 39 sessions | Inside G4's limits; live against paper once live |
| Trial burn and PBO by family | Core 61, PBO 29.2%; options 20 | No unrecorded trial; core PBO at or under 30% through J1 and MV1 |

## 5. Cadence

| When | Step | Automated? |
|---|---|---|
| Each session | Preflight facts; prior-art search on any new idea; guard suite before handover; ledger rebuilt | Tools, run by the assistant |
| Each verdict | Journal row with post-mortem fields; lesson ids; cleanup list for approval | Enforced by tests |
| Nightly | Store doctor; freshness in the catalogue | Yes: a step in the nightly workflow |
| Weekly (Saturday email) | Incident list; loop-health block; MC bands and G4 as today | Yes: built in `tools/books_report.py` |
| Monthly (GT12 cycle) | Re-rank the queue with priors; review open lessons; one research cycle | Report generated; ranking is the owner's |
| Quarterly (CK1) | The checklist; audit a sample of guards for false alarms and dead guards; trial burn and PBO | Report generated; the audit is by hand |

Automation belongs in `tools/`, the Actions workflows and `scheduler.py`. The `scheduling/`
folder holds retired legacy jobs and is not the place.

## 6. Risks of the loop itself

| Risk | Mitigation |
|---|---|
| Guard sprawl: every incident adds a test | A lesson needs a guard only if it recurred or touches the live path; others may be `waived` with a dated decision. Quarterly audit removes dead guards, by the owner's decision only |
| False alarms teach everyone to ignore alerts | Advisory checks (pattern match, priors) never fail a build or email; only exact checks do. Each false alarm is itself logged against its guard |
| Slow tests | The fast suite stays under 3 minutes (it runs in about 75 seconds today); store-dependent guards carry the `guard` marker and run before handover and weekly |
| A fifth record that drifts | The ledger, catalogue, priors and family counts are generated; a test fails when a generated file differs from its source |
| The loop becomes the work | A cap: at most 20% of a week's hours on process items after go-live; P1 first, then only what a real incident asks for |
| Priors on tiny samples kill good ideas | Priors show their sample size, rank only, and move an idea at most one place below n = 8 |
| Checks touching the live path before RA1 | No P1 item touches the engine, the deployed book, `live_session.py` or a configuration. PL-09 and PL-20 wait for approval after the verdict |

## 7. What would deliberately stay manual

- **Verdicts and promotions.** A gate computes; the owner decides. No automatic promotion,
  demotion or strategy change.
- **Deleting files.** Cleanup produces a list; deletion waits for approval, every time.
- **Weakening or removing a guard.** Only on the owner's dated decision.
- **Ranking the research queue.** Priors inform; the owner ranks.
- **Hypotheses and post-mortem judgement.** "Why did this fail" and "what would we check
  earlier" need thought; only their recording is enforced.
- **Vendor and broker contact, purchases, account changes, Kite logins and orders.**
- **Anything under the 2026 holdout.** No tool reads it.
- **Classifying a new incident into a pattern.** A keyword match proposes; a person confirms.

## 9. Built on 14 Nov (PL-01 to PL-07)

| Id | What exists now | Where |
|---|---|---|
| PL-01 | `python -m nse_engine.journal check`, run by a test, names every tracker verdict since 2 Nov with no journal row. 19 rows were backfilled from the tracker's own words | `nse_engine/journal.py`, `docs/metrics_journal.csv` |
| PL-02 | `docs/lessons.jsonl`, generated: 22 lessons, 18 guarded, 4 open with review dates. Guards carry `@pytest.mark.lesson`; a test fails when the file is out of date, a guard is gone or a review is overdue | `tools/lessons.py` |
| PL-03 | `config/research_families.json` holds each family's N, read from the registries; a test fails when the protocol's table states another. The table's three stale rows are corrected | `nse_engine/validation/families.py` |
| PL-04 | `python -m tools.preflight` prints computed facts: the clock in IST and UTC, sessions, what a run started now would do, RA1's date, trial counts, record gaps, open items | `tools/preflight.py` |
| PL-05 | `python -m tools.prereg prior-art <words>` and a Phase 0 block in the template; a test refuses a section pre-registered from 15 Nov without it | `tools/prereg.py`, the protocol |
| PL-06 | Nine post-mortem columns on the journal; five are required on rows from 15 Nov | `nse_engine/journal.py` |
| PL-07 | A test against empty source files; `python -m tools.study start / close` lists a study's files for approval and deletes nothing | `tools/study.py` |

Differences from the plan above:
- **`guard_location` is a list**, because one lesson may have several guards.
- **The verdict is read from one clause.** The first detector also read "FAILED" in an ordinary fix entry as
  a verdict. It now reads only the clause after the title ("...): FAIL, closed"). A verdict stated later in
  an entry is not caught; that limit is an open lesson (L-VERDICT-LATE).
- **Counts for the core** are the columns of the deployed book's same-window returns matrix (61), the N its
  PBO uses; for other families, the distinct configurations recorded.
- **Three patterns were added** to the ledger's list: F11 a human step missed, F12 a source's numbers not
  replicating, F13 leftover files.

## 10. Built on 14 Nov, second part (PL-08, PL-12, PL-13, PL-15)

| Id | What exists now | Where |
|---|---|---|
| PL-08 | `python -m tools.prereg new` prints the template with the family's count and the prior art; with `--body <file> --write` it numbers the section, stamps the clock's time and adds it to the plan. `check` also names any section stamped later than now | `tools/prereg.py` |
| PL-12 | The audit of earlier bugs in our own code: 29 entries. 24 are held by a marked test (6 marked under PL-02, 6 existing tests marked now, 12 tests written now); 5 are open lessons with a review date of 31 Jan 2027: undefined names (H7), sessions on every API route (S2), the paper / live isolation fixes (IS1), the day's P&L in the email (G16) and the legacy strategy backtest (BT1) | tests, `tools/lessons.py` |
| PL-13 | `python -m tools.guards` runs the nine tests that need the data stores and records the outcome on this machine; the preflight shows the last run and flags one that is over a week old, failed or skipped | `tools/guards.py` |
| PL-15 | One table of every dated rate of the share model with its source, which a test holds the model to; a source for every dated F&O and MCX rate; the published rates not applied yet; the date of the last reading and of the next | `nse_engine/validation/rates.py` |

Differences from the plan above:
- **PL-13 adds no `guard` marker.** All nine tests marked `slow` are store guards, so the command runs those.
- **PL-15 touches no cost file.** The table sits beside the models and a test binds them, so the frozen
  `nse_engine/costs.py` has no change.
- **PL-15 found four published rates the models lack** on its first reading (14 Nov): NSE's transaction
  charge now includes the investor-protection levy (shares 0.00307% against 0.00297%, futures 0.00183%
  against 0.00173%, options 0.03553% against 0.03503%), and Zerodha's depository charge is Rs 15.34 a stock
  sold against 15.93. They are small and partly offsetting. Cost figures are frozen until RA1, so they are
  listed as pending under open lesson L-RATES-2026 and go in with the next cost-model version, on the
  owner's decision.
- **PL-12 found more untested fixes than expected.** Fixes had been checked by temporary scripts that were
  then deleted (the "no new test files" rule read as "no tests"). The drawdown re-arm, the live stop logic,
  the go-live evidence rule and the dated face values had no test at all before today.
- **PL-05's check was tightened**: a Phase 0 field that still holds any `<placeholder>` is unfilled.

## 8. Corrections to the brief

- The RA1 verdict falls on **15 Dec**, the 60th gate session, not 14 Dec (FL1).
- `scheduling/` is legacy; see section 5.
- Areas 3 and 4 depend on real fills, which start at go-live. Only PL-14's first fixtures
  and PL-15 can be done before.
