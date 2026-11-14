# Platform gap review: Centurion against a quant firm's platform (PG1, 14 Nov 2026)

Scope: the Ind stocks module. Written at the owner's ask after the 11 Nov audit. It
covers the gaps, the best fix for each (free first), what was built now, a survey of
fundamentals data sources, and the plan. The configuration is frozen until the RA1
verdict on 14 Dec 2026 (tracker RA1F), so only plumbing was changed.

Quant firms' internal platforms are proprietary. The comparison uses the publicly
described pattern (one code path from research to production, independent risk,
measured costs, redundant infrastructure), not any firm's actual system.

## 0. Verdict

- **Research discipline is at par.** Pre-registration, a trial registry, deflated
  Sharpe and PBO, a sealed 2026 holdout, config hashes, canaries, paper before live.
- **Infrastructure is better protected than the audit said.** The nightly run has an
  on-time trigger, late backups and a dead man's switch. The real residual risks are a
  database that suspends at its free quota, and live-critical endpoints that share
  one free-tier app with the research pages.
- **The methodology gap is breadth, and it is structural.** One price-momentum family
  on daily data gives about 130 independent bets a year. The fundamentals store
  exists and its first two uses did not pass. More price signals in the core would
  spend a trial budget that is already at its limit.
- **Three plumbing items are done** (section 5): alerts from order postbacks, a locked
  package set for the app, and a notice on the pages that do not show the engine.
- **Cost of the recommended fixes: about Rs 200 a month**, against the Rs 2,500
  budget.

## 1. Corrections to the 11 Nov audit

The audit took three items from the 10 Oct gap register, which was out of date.

| The audit said | The record shows |
|---|---|
| No fundamentals or earnings dates | Since 12 Oct a point-in-time store from NSE filings: 166,910 quarterly results and 163,688 board-meeting dates, refreshed weekly (GT1). The value / quality sleeve failed (GT4); the earnings-surprise test was withdrawn (GT7) |
| The nightly run depends on best-effort cron | The app's own scheduler starts it on time at 19:00 IST (a 20:30 retry if it did not run), four GitHub crons are backups, and a healthchecks.io dead man's switch watches it. Corrected 14 Nov: the cron-job.org trigger the workflow describes (19:30) has not fired in 2-14 Nov, and the GitHub crons arrive 6-10 hours late, so there is one on-time trigger, not three |
| The database outage shows a weak free tier | The 21-23 Oct outage was a self-inflicted quota burn (a health check that never let the database sleep), fixed on 21 Oct. The owner then tried the paid plan and returned to Free on 2 Nov |

## 2. Gap register, by priority

Priority weighs the chance of a failure in live trading and the value to the golden
target against cost and effort. "Done" means built and tested on 14 Nov, uncommitted.

### 2.1 System design and architecture

| # | Gap | Priority | Fix | Cost | Status |
|---|---|---|---|---|---|
| A1 | A rejected or cancelled order is seen only in the evening run | High | Alerts from Kite's order postbacks within a minute (AL5) | Free | **Done** |
| A2 | The app's packages floated to the newest release at every rebuild; this Mac tests different versions (99 of 226 differ) | High | A lock file used by the app's build, guarded by a test | Free | **Done** |
| A3 | The database suspends when the free quota is used up; the day's Kite login and every book live in it | High | Neon's pay-as-you-go plan before go-live: no suspension, a day of restore history instead of 6 hours | About Rs 200 a month | **Done 14 Nov** (owner) |
| A4 | Login callback, order postbacks and the scheduler share one free app with the research pages (ML, RAG): a heavy page or a rebuild can take the live path down | Medium | A small live-only service on the Lightsail host already paid for (static IP, Mumbai); the free app keeps the research pages | Rs 0 extra; 2-3 days | After go-live is stable |
| A5 | A merge to `main` deploys straight to production | Medium | A second free Space fed from `develop`, without trading secrets; a health check after each deploy that alerts | Free; half a day | After 14 Dec |
| A6 | The Verdict and Screener pages show a model the engine does not trade | Medium | A notice on both pages now; later an "engine view" page | Free | **Notice done** |
| A7 | The heartbeat runs on GitHub's cron alone (it arrived 5 hours late on 11 Nov) | Low | Add it to the app's scheduler, which already starts two workflows on time | Free; 1 hour | Open |
| A8 | One broker, orders only at the open, no execution algorithms | Low at Rs 3 lakh | Nothing now. Revisit when an order exceeds about 1% of a day's volume | - | Not needed |
| A9 | Risk checks run inside the strategy's process, at end of day | Low for swing | The kill switch is already a separate workflow; A1 adds the intraday signal | - | Covered |
| A10 | One operator, one login a day | Fixed by regulation | Reminders at 09:00 and 17:30 exist | - | Accepted |

### 2.2 Stock analysis methodology

| # | Gap | Priority | Fix | Status |
|---|---|---|---|---|
| M1 | Breadth: about 130 independent bets a year at an IC of 0.037. A Sharpe of 1.4 needs about ten times the bets or three times the IC | High, structural | More uncorrelated streams, not more price signals in the core. Four run on paper (futures, bonds, MCX, the options overlay); each has its own gate | In progress |
| M2 | Costs are modelled, not measured on real fills | High after go-live | X1 after 30 real fills; HC1 at 60 live sessions | Dated |
| M3 | The book lags rebounds (37% of the 2008-10 rebound captured; 6 of 7 paper books trail the Oct-Nov rally) | Medium | J4, the half-step in risk-off rebounds, pre-registered for February | Dated |
| M4 | Data: no balance sheet or cash flow detail, shareholding, pledges or analyst estimates | Medium | Section 4: extend the free NSE store only for a pre-registered hypothesis; one paid candidate is estimate revisions (NS1-9) | Survey done |
| M5 | The core's trial budget is spent (61 configurations, PBO 29% against a bar of 30%) | High, governance | No new core trials except those already registered (J1, MV1); new ideas go to their own families | In force |
| M6 | Long-only, no leverage | Capital-gated | Shorts reopen past Rs 2 crore (U28); the levered futures book is on paper | Parked |
| M7 | A user cannot see the engine's own view of a stock | Medium, product | An "engine view" page: the deployed engine's rank, forecast, stop and regime for a symbol, read from its nightly output. Display only | After 14 Dec |

### 2.3 Low-level technical

| # | Gap | Priority | Fix | Status |
|---|---|---|---|---|
| L1 | Test coverage 19% overall; the live path 24-95% | Medium | The ratchet (COV1) already stops it falling. Raise the weakest live-path files as they are touched | In force |
| L2 | The nightly workflow's legacy branch installs unpinned packages | Low | Use the same lock there, after 14 Dec (the nightly run is not touched during the freeze) | Open |
| L3 | Ledgers are JSON documents in the database | Low | Nothing now: weekly exports to Kaggle exist (JR2), and A3 lengthens restore history | Covered |
| L4 | About 178,000 lines, mostly legacy, beside a small live path | Low | Removed on 9 Nov: the five book labs and 126 files. Strategy code stays, by the owner's rule | Ongoing |
| L5 | One encryption key copied to three places | Low | A local Keychain copy (in progress); rotate all three together | Owner's action |

## 3. Infrastructure: analysis and the best options

Prices are in US dollars as published; rupees assume about Rs 90 to the dollar.

### 3.1 Database (A3)

| Option | Monthly cost | What it gives | What it risks |
|---|---|---|---|
| Neon Free (today) | Rs 0 | 100 compute-hours a month per project; storage 0.5-1 GB (sources differ) | **Suspends** when the quota is used; restore history 6 hours |
| **Neon Launch, pay as you go** | About Rs 150-250 at today's use | No suspension; $0.106 a compute-hour, $0.35 a GB-month; a day of restore history | A runaway query costs money, not uptime: keep the $5 spending alert |
| Self-hosted Postgres on the Lightsail host | Rs 0 extra | Mumbai, so faster from India | Backups, upgrades and security become the owner's job; one host holds everything |

Measured on 14 Nov: the database holds 38 MB. Compute use was estimated at about half a
compute-hour a day after the 21 Oct fix, about 15 of the 100 free hours a month.

**Recommendation: Neon Launch before the first live session.** The free plan fits today's
use, but its failure mode is a full stop: no login token, no books, no nightly run. That
cost Rs 0 on paper and would cost real fills when live.

### 3.2 App host (A4)

| Option | Monthly cost | Notes |
|---|---|---|
| Hugging Face Space, free (today) | Rs 0 | Sleeps after 48 idle hours (the health pings prevent it); no uptime commitment; forum reports of tighter free limits in 2026, unconfirmed |
| Hugging Face CPU upgrade | About Rs 1,950 ($0.03 an hour) | Never sleeps. No uptime commitment either, and a rebuild still restarts it |
| **A small live-only service on the existing Lightsail host** | Rs 0 extra ($5 a month already paid) | Login callback, postbacks, kill switch and scheduler only, on the pinned core packages; no ML libraries; a fixed IP Zerodha already knows |

**Recommendation: do not buy the upgrade.** It removes sleeping, which is already handled,
and leaves the real exposure: the live path shares a process and a rebuild with pages that
load large ML libraries. Split the live path off instead, after go-live has settled (it
changes the Kite redirect and postback addresses, so not before the first live weeks).

### 3.3 Scheduling and monitoring (A7)

Already in place: the app's scheduler starts the nightly run at 19:00 (retry 20:30), GitHub's
crons back it up 6-10 hours late, a dead man's switch emails if no run finishes by 23:00,
Sentry, a heartbeat, and now order postback alerts. Two weak links, both free to fix:
- the heartbeat's own trigger: start it from the app's scheduler, like the nightly run;
- a second on-time trigger: the cron-job.org job the workflow describes has not fired since
  at least 2 Nov. Built 14 Nov as a conditional rescue (tracker RS1): cron-job.org starts the
  heartbeat workflow at 19:45 IST, which starts the nightly run only when it is still needed,
  so an ordinary night costs a minute. The owner sets up the token and the job.

### 3.4 Deployment (A5)

Free fix: a second Space fed from `develop` and holding no trading secrets, plus a health
check at the end of the deploy workflow that raises an alert. The lock (A2) already removes
the commonest cause of a surprise after a rebuild.

### 3.5 What the recommended set costs

| Item | Monthly |
|---|---|
| Neon Launch | About Rs 200 |
| Lightsail host (already paid) | Rs 0 extra |
| Everything else | Rs 0 |
| **Total new spend** | **About Rs 200** |

## 4. Fundamentals data sources

### 4.1 What Centurion already has

A point-in-time store built from NSE's own filings (tracker GT1), free:

- quarterly results from late 2011: revenue, profit before and after tax, owners' profit, EPS,
  paid-up capital, shares, equity; consolidated and standalone; with the time each result
  became public;
- board-meeting dates, which are the earnings dates;
- coverage of the top-500 universe with four known quarters: 89-91% in 2013-18, 95-98% in
  2020-25;
- delisted companies included, because it is built from the filings of the day.

Its known faults: mis-scaled amounts and swapped share counts on older NSE pages (a guard
rejects them), and 7,143 filings with no usable document, half of them in 2018-19.

What has been tested on it: a value / quality sleeve failed (GT4: correlation with the book
0.56-0.63, and it lowered the book's Sharpe); the earnings-surprise test was withdrawn
(GT7).

### 4.2 What is missing

| Data | Would allow | Free route |
|---|---|---|
| Balance sheet and cash flow (half-yearly and annual) | Accruals, leverage, return on capital | NSE's integrated filings and annual-report XBRL: the same source and parser family as GT1 |
| Shareholding pattern, promoter pledges | Ownership changes, pledge exclusion | NSE quarterly shareholding XBRL. Pledge and ASM/GSM list history could not be fetched (GT11) |
| Analyst estimates and revisions | Revision breadth (NS1-9) | None found that is point-in-time |
| Segment data, concall text | - | Not considered |

### 4.3 Sources surveyed

"PIT" means point-in-time: the value as first published, with its date. Without it a
backtest reads restated numbers it could not have known. Items marked "ask" could not be
confirmed from public pages.

| Source | What it has | PIT and delisted | Access for an individual | Price | Verdict |
|---|---|---|---|---|---|
| **NSE filings (in use)** | Results, board meetings; balance sheets and shareholding available as XBRL | Yes and yes, by construction | Public files | Free | **Keep. The base.** |
| BSE XBRL filings | Results and shareholding from 2015, incl. BSE-only names | Yes, by construction | Public pages; terms not found | Free | A cross-check for NSE's faulty old pages |
| NSE Data & Analytics, Corporate Data | Fundamentals, announcements, shareholding as a daily feed | Official; history depth: ask | Contract | Rs 10.6 lakh a year | Far over budget |
| CMIE Prowess | The standard academic source: listed and unlisted, from 1989 | "Vintages" (snapshots) exist; usability for PIT: ask | Institutional; individual terms: ask | One 2021 invoice: Rs 2.55 lakh a year | Best paid history; only if a test needs it |
| Capitaline | 55,000-75,000 companies | PIT: ask | A CFA Society India member offer once listed Rs 15,000 a year (list Rs 1.5 lakh) | See left | Terms forbid lifting data; check before any use |
| ACE Equity (Accord Fintech) | Listed-company fundamentals; feeds many Indian sites | PIT: ask | Contract; an API exists | Not published | Ask for a quote and a sample |
| LSEG Worldscope / Reuters Fundamentals PIT | Global, non-US PIT from 1997 | Yes; India depth: ask | Institutional | Enterprise | Over budget |
| S&P Compustat Global | 42,000 non-US companies | PIT snapshots are North America only | Institutional | Enterprise | Not PIT for India |
| LSEG I/B/E/S | Analyst estimates, non-US from 1987 | History kept; India count: ask | Institutional | Enterprise | The only solid estimates history; over budget |
| Trendlyne | Indian fundamentals, consensus estimates and revisions | Depth and PIT: ask | A developer plan from Rs 299 a month; data-feed terms: ask | See left | The one affordable estimates lead: ask for PIT history |
| Tijori Finance | Statements, shareholding, alternative data | Not PIT | No public API; an unofficial scraper exists | Rs 3,500 a year | Reading only. The scraper is against its terms: not used |
| Screener.in | 10+ years of statements | Restated, not PIT | No API; scraping against its terms | Low | Reading only |
| EODHD | Statements for NSE names | Not PIT; India depth and delisted: ask | API | $59.99 a month | Not PIT: unfit for backtests |
| Financial Modeling Prep | Global statements | Not PIT; India coverage unclear | API | $99+ a month (third-party listing) | Unfit |
| TrueData, IndianAPI and similar | Fundamentals APIs | Not PIT | API | Varies | Unfit |
| Yahoo (yfinance) | 4-5 years, restated | No; drops dead names | Unofficial | Free | Unfit: the legacy pipeline's flaw |

### 4.4 Recommendation

1. **Buy nothing now.** The two tests the store allowed did not pass, and no registered
   test is waiting for data.
2. **Extend the free NSE store only behind a hypothesis.** Balance sheets and shareholding
   come from the same filings. The build is a day or more; it earns that only when a
   pre-registered test names the fields it needs.
3. **The one paid lead is estimate revisions (NS1-9).** Ask Trendlyne, then ACE Equity,
   the questions below before any payment.
4. **Never use restated data in a backtest**, whatever it costs. That rules out every
   cheap API in the table.

Questions for any vendor, in writing:
- Is each value stored as first reported, with the date and time it became available?
- Are delisted and merged companies kept, from which year?
- How many NSE companies have history in 2013, 2018 and 2023?
- For estimates: is each analyst's revision kept with its date, or only today's consensus?
- May the data be used in an automated trading system by one individual? May derived
  results be published?
- A sample file for 20 named symbols, including 5 delisted ones, before payment.

## 5. Built on 14 Nov (plumbing; uncommitted)

All three are outside the freeze: no configuration, engine, sizing, cost or paper-book
number changes.

### 5.1 Alerts from order postbacks (A1, tracker AL5)

- Zerodha already posts every order update to the app, which checks its signature. Now
  each update is also classified (`kite_connect/trading/postback_alerts.py`):
  - an order the engine sent that is **rejected or cancelled**: a CRITICAL email with
    Kite's reason and what to do;
  - a **stop that filled**: a WARNING. The sell is matched to a triggered stop order by
    its order number, read with the day's Kite login;
  - everything else: nothing. The nightly email carries it.
- Each order alerts once. The alert runs after Kite has its reply and can never fail it.
- Open question, answered by the first supervised sell (ST2): whether Kite posts back
  stop orders and hand-placed orders at all. If it does not, a filled stop is still found
  by the evening run, as before.
- The app has no `CENTURION_LIVE_MODE` of its own (checked 14 Nov). Set it to `dry_run` now, matching
  GitHub's variable, and change both to `live` on go-live day. On the app it only labels alert
  emails (CRITICAL or "would-be CRITICAL") and gates login reminders; it places no orders there.

### 5.2 A locked package set for the app (A2)

- `constraints-hf.txt` pins all 226 packages to what the app has run since its last package
  build on 9 Nov. `deployment/Dockerfile.hf` installs against it.
- Checked: with the lock, today's resolution reproduces it exactly. Without it, a rebuild
  today would already change 18 packages, including `fastapi`, `anthropic` and `arch`
  (a major version).
- Two tests fail the build before a merge if a pin in a requirements file and the lock
  disagree. That mismatch would otherwise fail the app's build.
- After the next deploy: check that the login page, Fly Kite and `/health` work.

### 5.3 A notice on the Verdict and Screener pages (A6)

Both pages now say that their scores come from an older model the engine does not trade
and that was never validated, and link to the Trade Center. The Screener's notice adds
that orders sent from it are the user's own.

## 6. Plan of action

| When | Item | Who |
|---|---|---|
| Now | Review and merge section 5 (two repositories) | Owner |
| Done 14 Nov | A3: the database is on Neon Launch, with the $5 alert | Owner |
| Now, then go-live day | `CENTURION_LIVE_MODE` on the app: `dry_run` now, `live` with GitHub's variable (5.1) | Owner |
| 16-17 Nov | Sell test: note whether a postback email arrives for the stop sell | Owner, Claude |
| 14 Dec | RA1 verdict. Nothing above changes it | - |
| After 14 Dec | A7 heartbeat trigger; A5 staging Space and deploy check; L2 lock in the nightly workflow | Claude |
| After 14 Dec | M7: the engine-view page (display only) | Claude |
| January | J1, MV1 (the last registered core trials); NS1-9 vendor questions | Claude, Owner |
| February | J4 (rebound half-step) | Claude |
| After the first live weeks | A4: the live-only service on the Lightsail host | Claude |
| At 30 real fills / 60 live sessions | X1, HC1: measured costs replace the model | Claude |

## 7. Unverified

- Neon's free storage limit (0.5 GB or 1 GB) and whether Launch has a minimum charge:
  sources differ. Check the pricing page when switching.
- Hugging Face's free-tier limits in 2026: forum reports only.
- Every vendor price in section 4.3 except NSE's tariff and EODHD's page is second-hand
  or dated. Every "ask" is open.
- The lock reproduces the app's package set by date, not by reading the running app.
  It was checked for consistency, not by installing it on Linux.
- Whether Kite posts back stop orders: see 5.1.

## Sources

- [Neon: free plan limits and quotas](https://neon.com/faqs/free-plan-limits-and-quotas)
- [Neon pricing 2026, third-party summary](https://www.jetadmin.io/blog/neon-pricing/)
- [Hugging Face: Spaces overview](https://huggingface.co/docs/hub/en/spaces-overview)
- [Hugging Face forum: CPU Basic quota](https://discuss.huggingface.co/t/cpu-basic-quota-limit-with-only-one-space/180646)
- [NSE: XBRL information](https://www.nseindia.com/static/companies-listing/xbrl-information)
- [NSE: Corporate Data subscription](https://www.nseindia.com/market-data/corporate-data-subscription)
- [NSE pricing file, domestic clients, 2026](https://nsearchives.nseindia.com//web/mediaattachment/2026-03/NSE_Pricing_file_-_Domestic_clients_20260309171343.pdf)
- [BSE: about XBRL](https://bseindia.com/static/about/xbrl_info.aspx)
- [CMIE Prowess dx user guide (IIM Indore)](https://iimidr.ac.in/wp-content/uploads/2025/10/User-Guide-CMIE-Prowessdx.pdf)
- [Capitaline member offer (CFA Society India)](https://cfasocietyindia.org/wp-content/uploads/Media-Uploads-Membership/Member-offers/Member-Offer-Document_for-Society_Capitalin2.pdf)
- [LSEG: Worldscope fundamentals](https://www.lseg.com/en/data-analytics/financial-data/company-data/fundamentals-data/worldscope-fundamentals)
- [LSEG: point-in-time fundamentals](https://www.lseg.com/en/data-analytics/search/point-in-time-fundamentals)
- [LSEG: I/B/E/S broker estimates](https://www.lseg.com/en/data-catalogue/company-data/ibes-estimates/broker-estimates)
- [Compustat (Wikipedia)](https://en.wikipedia.org/wiki/Compustat)
- [Trendlyne: consensus estimates](https://trendlyne.com/equity/consensus-estimates/what-is/modal)
- [Trendlyne: developer plans](https://trendlyne.com/subscription/mcp/plans/)
- [Tijori Finance: FAQ](https://www.tijorifinance.com/in/faq/)
- [EODHD: pricing](https://eodhd.com/pricing)
- [Tapetide: Indian market data APIs compared](https://tapetide.com/compare/best-api-indian-stock-market-data)
- Internal: `docs/golden_target_review.md`, `docs/MEMORY_TRACKER.txt` (GT1, GT4, GT7,
  NEO1, AL5, COV1, CK1), `docs/research_protocol.md`.
