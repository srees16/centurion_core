# Centurion codebase audit

Refactor audit, one top-level module at a time. The rule for every step: functionality, workflow, commands, outputs and schemas stay exactly as they are. Only structure changes.

| Section | Module | Files | Lines | Status |
|---|---|---|---|---|
| 0 | The whole codebase | 481 | 162,800 | Audited 19 Nov 2026 |
| 1 | `tools/` | 15 | 3,448 | Approved 18 Nov; the steps allowed before RA1 done (1.9) |
| 2 | `nse_engine/` | 62 | 21,348 | Audited 19 Nov, waiting for approval |
| 3 | `kite_connect/` | 80 | 28,574 | Audited 19 Nov, waiting for approval |
| 4 | `services/` | 98 | 36,107 | Audited 19 Nov, waiting for approval |
| 5 | `api/`, `auth/` | 46 | 10,809 | Audited 19 Nov, waiting for approval |
| 6 | `rag_pipeline/` | 29 | 17,657 | Audited 19 Nov, waiting for approval |
| 7 | `database/` | 13 | 4,589 | Audited 19 Nov, waiting for approval |
| 8 | Root modules | 10 | 3,914 | Audited 19 Nov, waiting for approval |
| 9 | `runners/`, `optimizer/`, `scripts/` | 16 | 4,517 | Audited 19 Nov, waiting for approval |
| 10 | `strategies/`, `trading_strategies/` | 53 | 17,973 | Audited 19 Nov, waiting for approval |
| 11 | `scrapers/` | 28 | 5,544 | Audited 19 Nov, waiting for approval |
| 12 | `infrastructure/`, `scheduling/`, `deployment/` | 16 | 2,502 | Audited 19 Nov, waiting for approval |
| 13 | `cloud/` | 8 | 2,470 | Audited 19 Nov, waiting for approval |
| 14 | `us_engine/` | 8 | 3,382 | Audited only: the US module is on hold |
| 15 | `tests/` | 24 | 12,243 | Audited 19 Nov |

Two freezes are in force, both recorded in the tracker:

- **RA1 freeze, until the verdict on Tue 15 Dec 2026:** the engine, costs, `live_session.py`, the capital ladder and config, and (decision 1.7-2) the nightly scripts in `tools/`. Every step that touches the nightly or live path waits for it.
- **Process freeze PF1, until the guard review on Sat 19 Dec 2026:** no new checks, report blocks or emails. Steps that add a check (a CI gate, an import-rule test) wait for that review.

---

## 0. The whole codebase

### 0.1 How this was audited

Read-only, on `feature/fine-tuning` on 19 Nov 2026; four files modified by someone else (`cloud_paper_runner.py`, `nse_engine/books.py`, `tests/test_metrics.py`, `docs/paper_trade_strategy.md`) were read but not touched.

- **Static pass** over all 505 tracked Python files: sizes, function lengths, exception handlers, `print` calls, environment reads, `__main__` blocks, `sys.path` changes.
- **The internal import graph:** module-level and in-function imports, cycles, and which files each production entry point can reach.
- **Dead code and duplication:** top-level definitions and files that nothing references; identical function bodies; look-alike files.
- **Secrets scan** of every tracked file.
- **Statement coverage** of the fast suite (`-m "not slow"`). The store tests, which run only on this Mac, are not counted.
- **Lint and types:** `ruff` (correctness rules E4, E7, E9, F, B) and `mypy`, from `ProjectX/devenv`.
- **Targeted reading** of each package.

Not done in this pass: a line-by-line review of every module, and a security review of the API's authentication and input handling (that is a separate `/security-review`).

### 0.2 Architecture

**Entry points**

| Where | What runs | Code |
|---|---|---|
| GitHub Actions, nightly (Mon-Fri 19:00 IST) | Data sync, the engine's plans, the paper books, the live session, the report, the heartbeat | `nse-paper-trading.yml`: `nse_engine.data.{archive,fo_store,mcx_store}`, `runners.run_nse_engine`, `cloud_paper_runner.py --engine`, `kite_connect.trading.live_session`, `tools.{books_report,paper_heartbeat,session_failed}`, `nse_engine.{deployment,crash_recovery}` |
| GitHub Actions, other | Heartbeat 00:15 IST; login reminder 09:03; ETF depth 09:45; US advisory (disabled); kill switch, sell-path record and options demo by hand; tests; deploy | `tools.paper_heartbeat`, `kite_connect.auth.{accounts,daily_login}`, `nse_engine.etf_costs`, `us_engine.advisory`, `kite_connect.trading.kill_switch`, `kite_connect.options.cli` |
| Hugging Face Space (`deployment/start.sh`) | `scheduler.py` (dispatches the GitHub runs, the Saturday newsletter) and `run_api.py`, which serves `api.main` | |
| This Mac | Weekly data refresh and backup, research, the process loop | `runners.refresh_data`, `cloud.{backup,devsetup,kaggle_local}`, `tools.*`, 26 `nse_engine` command lines |
| Kaggle | Walk-forward folds | `cloud.kaggle_entry`, `cloud.kaggle_runner` |
| Frontend (`centurion_core-fe`) | Calls `/api/v1/*`, and `/api/paper-trading` | |

**Packages by role**

| Role | Code |
|---|---|
| Domain: the deployed engine and its research | `nse_engine/`; option theory and strategies in `kite_connect/options/` |
| Execution: the paper and live books | `cloud_paper_runner.py`; `kite_connect/trading/` (live session, executor, paper trader, sleeves) |
| Broker adapter | `kite_connect/{auth,core}`, and `order_service`, `gtt_stops`, `order_status`, `kill_switch`, `nse_instruments` in `kite_connect/trading/` |
| Persistence | `database/` (Neon through SQLAlchemy; `paper_cloud` holds every book) |
| Interfaces | `api/`, `scheduler.py` and `scheduling/`, `runners/`, `tools/`, module command lines |
| Reports and alerts | `services/notifications/`, `tools/books_report.py`, `api/newsletter.py` |
| RAG | `rag_pipeline/` |
| The legacy Carver path (before the engine) | most of `services/`, `strategies/`, `trading_strategies/`, `scrapers/`, `main.py`; `auto_executor`, `daily_rebalancer`, `trade_monitor` and others in `kite_connect/trading/` |
| Research one-offs | `optimizer/`, `scripts/`, `runners/run_r21a*.py`, `runners/run_r22_*.py` |
| Shared plumbing | `infrastructure/`; root `config.py`, `models.py`, `utils.py` |
| Off-GitHub operations | `cloud/` |

**Data flow of the deployed path:**
1. NSE archives go into `nse_engine.data`'s parquet stores.
2. `runners.run_nse_engine` builds each book's plan with `nse_engine.engine`.
3. `cloud_paper_runner.py --engine` fills the paper books (`paper_trader`, the sleeves) and writes Neon (`database.paper_cloud`).
4. `live_session` sends the live book's orders: `nse_engine_executor`, then `order_service`, then Kite.
5. `tools.books_report` emails the results; the API and the frontend read Neon.

**External integrations:**
- Broker and markets: Kite Connect (orders, quotes, MCX), nseindia.com and NSE's archives, Yahoo Finance.
- Data and compute: Neon Postgres, Kaggle, ChromaDB, Redis and MinIO (both optional), TimescaleDB (tick webhooks).
- Operations: GitHub REST, healthchecks.io, SMTP and the Gmail API, the Hugging Face Space.
- Others: Anthropic (RAG), Binance and CoinGecko, Tiingo and DriveWealth (US), news sites (scrapers).

### 0.3 Findings across the codebase

**High**

| # | Finding | Evidence | Risk |
|---|---|---|---|
| X1 | Two trading systems share the code, and the old one sits behind one switch | The paper runner runs the legacy CarverPipeline whenever the repository variable `CENTURION_NSE_ENGINE` is not `true`. The workflow and `cloud_paper_runner._engine_enabled` both default it to `false`; it has been `true` since 16 Sep 2026. Your decision TS2 (test, leave or retire the legacy path) is open. | A deleted or mistyped variable would silently run the paper books on the old pipeline. Refactoring legacy code may be wasted on code that is then retired. |
| X2 | Dependencies run both ways between layers | `nse_engine` imports from interfaces and adapters: `runners.run_nse_engine._load_data` (scorecard, sepa, value), `tools.books_report` (significance), `cloud.kaggle_runner` (scorecard, trials), `kite_connect` 28 times (order bands in `engine.py`, the sleeves in `deployment.py`, the private helpers `_run_id`, `_live_book_returns` and `_nifty_tri_returns` in five research streams). `kite_connect` imports `nse_engine` 123 times. `services` and `strategies` import each other, as do `services` and `kite_connect`. The root `config.py` is imported by 76 modules. | Nothing moves, tests alone or swaps out without touching the other side. Private names break silently. |
| X3 | Tests are thinnest where code runs every night | Fast-suite coverage is 27% of 78,779 statements. On the deployed path: `nse_engine/data/mcx_store.py` 0%, `nse_engine/crash_recovery.py` 0%, `runners/run_nse_engine.py` 7%, `services/execution/carver_pipeline.py` 6% (it holds the calendar the tools use), `cloud/backup.py` 0%. | Behaviour cannot be shown unchanged without characterisation tests first. |
| X4 | No lint or type gate | 928 `ruff` correctness findings (301 unused imports, 118 `raise` without `from`, 73 `zip` without `strict`) and 1,090 `mypy` errors. None checked is a live bug: the four undefined `pd` sit in quoted annotations, and `cloud_paper_runner`'s loop lambda is called in its own iteration. Lesson L-H7 is open. | Real defects hide among them, and no step can be checked automatically. |
| X5 | Very long functions | 425 functions over 60 lines. The largest: `run_full_backtest` 2,511 lines, `CarverPipeline.run` 1,634, `run_us_carver_pipeline` 607, `run_live_session` 543, `RAGQueryEngine.query` 457 and `query_stream` 383 (one flow written twice), `auto_executor._place_orders` 314. | Hard to test, review or change safely. |

**Medium**

| # | Finding | Evidence |
|---|---|---|
| X6 | Configuration is spread out | 211 environment variables read in 63 files. Three settings modules: `config.py`, `kite_connect/core/config.py`, `rag_pipeline/config.py`. (`nse_engine/config.py` is the engine's JSON schema and stays.) The database URL is read directly in 12 files, and a second database family `KITE_DB_*` serves Timescale. |
| X7 | Time zone and calendar duplicated | `IST` is defined in 29 non-test files. There is one holiday list (`nse_engine.nse_calendar.HOLIDAYS`) but two rules: `carver_pipeline.is_nse_trading_day` also honours `CENTURION_NSE_HOLIDAYS` (set nowhere today), and `nse_calendar.is_trading_day` does not. The heartbeat, preflight, Kite login and executor use the first; the engine and the API use the second. |
| X8 | Name collisions | Four `config.py`, four `cache.py`, three `walk_forward.py`, two each of `engine.py`, `models.py`, `store.py`, `costs.py`, `futures_overlay.py` and `options_overlay.py`. These are different code (19% line overlap at most), not copies. |
| X9 | Error handling | About 1,280 `except Exception` (services 376, kite_connect 292, api 196), most of them deliberate "never raises". There are no shared error types. Correction, 20 Nov: 96 API routes build a 500 from `str(exc)`, but `api/main.py`'s handler already logs that text and answers "Internal server error" (now pinned by a test). What still reaches the caller is the RAG stream's `Error: <exception>` token. |
| X10 | Library modules double as scripts | 104 `__main__` blocks, 65 `sys.path` insertions (27 in `trading_strategies`), about 940 `print` calls in library code (optimizer 252, runners 171, rag_pipeline 113). |
| X11 | Code no production entry point reaches | 58 files, most of them deliberate command lines or kept strategies, and 63 top-level definitions referenced nowhere. Whole folders are unreached: `optimizer/`, `scripts/`, the R21a and R22 runners, `scheduling/{fno,strategy}_jobs.py` (retired and kept on purpose). Your rule: each needs its own study before removal, and option, futures and strategy code always stays. |
| X12 | Self-tests inside production modules | `rag_pipeline` has `_run_tests` functions of 1,441 lines (in `retriever.py`), 638, 304 and 302, run by `__main__`. |

**Low**

| # | Finding |
|---|---|
| X13 | Two generations of API routes: nine unversioned routers stay mounted beside `/api/v1`, while the frontend calls `/api/v1` almost everywhere. This is a public API, so nothing changes without your yes. |
| X14 | Root-level legacy modules: `main.py` (the news-sentiment pipeline, imported by two API modules), `models.py` and `utils.py` (imported by 25 and 18 modules), and the one-off scripts `bootstrap_models.py`, `setup_database.py`, `run_backtest_production.py`. |
| X15 | 643 non-test functions have no return type, 207 of them in `api/`. |

**Checked and found clean**

- **Secrets:** none tracked. `.env` is not in git (only `.env.example`), and `config.py` builds its database URL from environment variables with empty defaults. The one key-like string is a dummy in a test.
- **Import cycles:** none at module level. In-function imports are used on purpose: the paper runner imports 6 modules at start against 252 reachable, which keeps the GitHub jobs' installs small but hides the coupling in X2.
- **Copy-paste:** only two identical functions (a date parser in two news scrapers, and one test helper in five RAG modules).

### 0.4 Target structure

Evolution, not a rewrite. The deployed path keeps its commands. Boundaries are fixed first; folders move last, and only with alias modules at the old paths.

```
centurion_core/
  shared/          new, standard library only: clock (IST, now), settings (database URL, live mode,
                   repository, paths), errors (base types), log set-up; started 20 Nov (clock, settings)
  nse_engine/      domain: engine, signals, portfolio, costs, calendar, gates; imports nothing from
                   runners, tools, cloud or kite_connect.trading
    data/          stores and loaders (run_nse_engine._load_data moves here)
    validation/
    research/      the research helpers now private in kite_connect.options
  kite_connect/
    broker/        auth, order_service, gtt_stops, order_status, kill_switch, instruments, quotes
    books/         live_session, nse_engine_executor, paper_trader, sleeves, book_events
    options/       unchanged
    legacy/        auto_executor, daily_rebalancer, trade_monitor, rsi_strategy, monitors (or retired: TS2)
  database/        one engine factory, settings from shared/
  api/             routers (HTTP only) -> api/services (one per feature) -> domain and adapters
  rag_pipeline/    one pipeline shared by query and query_stream; self-tests moved to tests/
  notifications/   services/notifications promoted: the deployed path's emails
  legacy/          the Carver path if TS2 keeps it: services/ (less notifications), strategies/,
                   trading_strategies/, scrapers/, main.py, behind one boundary
  tools/, runners/, cloud/, scheduling/   interfaces and operations
```

**Dependency rule:** interfaces depend on execution, execution on the domain, the domain on adapters. `shared/` may be imported by all and imports nothing of ours. A test of the import graph would enforce the rule; it is a new check, so it comes after 19 Dec.

### 0.5 Sequence

| Phase | When | What |
|---|---|---|
| 1 | Now to 14 Dec | Behaviour-neutral steps off the nightly and live paths: `shared/` and its use outside those paths; the API and RAG clean-ups; characterisation tests where a step needs them; per-item dead-code studies (read-only) |
| 2 | After RA1 (15 Dec) | The deployed path: X1's default, one calendar rule, the inverted imports, the long functions of the paper runner and the live session, the `kite_connect` split, `tools` T4 to T8 |
| 3 | After 19 Dec | `ruff` and `mypy` in CI as a ratchet (no new findings; old ones fixed module by module), the import-rule test, the test-map check |
| 4 | After TS2 | The legacy path: behind `legacy/`, or retired item by item after study |

**Checks for every step**, on top of the brief's:
- outputs recorded before and compared after;
- the full suite in `myenv` and on a core-only install;
- correctness lint and `mypy` no worse on the files touched;
- for a step on the nightly path, also a `--dry-run` session and an unchanged config hash (the RA1 rule).

### 0.6 Changes that would alter behaviour or a public API (flagged)

None of these is made without a yes.

- **X1:** an unset or unknown `CENTURION_NSE_ENGINE` fails loudly, or defaults to the engine. Today it falls back to the legacy pipeline.
- **X7:** one calendar rule: `CENTURION_NSE_HOLIDAYS` honoured everywhere, or nowhere.
- **X9:** API 500 responses carry a generic message, and the detail goes to the log. Already so (found 20 Nov); the RAG stream's error token is the one left, not yet approved.
- **X13:** unversioned routes retired, once the Space's access log shows them unused.
- **`zip(..., strict=True)`, 73 places:** it raises where lengths differ. Left out unless a case shows a silent truncation.
- **Legacy one-off scripts moved to `scripts/`:** `python setup_database.py` and similar commands change path.

`raise ... from exc` (118 places) changes only how a traceback is chained, so it counts as neutral.

### 0.7 Decisions (taken 20 Nov 2026: "go with your recommendations")

1. **TS2, the legacy Carver path:** taken off the paper runner after RA1, then quarantined behind `legacy/`, then retired item by item after study. Option, futures, strategy and `services/rl_bot` code stays, as before.
2. **The flagged changes in 0.6:**
   - X1 yes: the runner defaults to the engine and logs loudly; legacy only on an explicit `false`. After RA1.
   - X7 yes: `CENTURION_NSE_HOLIDAYS` honoured by the one calendar, everywhere. After RA1.
   - X9 yes: generic 500 messages, the detail in the log. Found already in place on 20 Nov; a test now pins it.
   - X13 yes, on evidence only: an unversioned route retires when its use is shown to be nil.
   - `zip(strict=True)` no.
   - One-off root scripts to `scripts/` yes. Now.
3. **Folder moves:** boundaries first. A folder moves only where a module shows the gain, with alias modules at the old paths, after RA1.
4. **Characterisation tests before 19 Dec:** yes, in the modules phase 1 refactors.

### 0.8 Progress (phase 1)

**Done 20 Nov 2026 (uncommitted).**

| Step | Change | Checked by |
|---|---|---|
| 1 | `shared/` (standard library only): `clock.IST`, `settings.REPOSITORY`, `settings.database_url()`. Used by the seven research and operations tools, `api/newsletter.py`, `api/routers/v1/screener.py`, `scheduling/cache.py`, `api/main.py`, `tools/export_books.py`. `tools/common.py` (18 Nov) folded in. | A test pins `database_url()`'s fallback; outputs unchanged |
| 2 | X9 checked: already in place (see the correction in 0.3). A test now pins it. | The test fails with the handler's 500 branch disabled |
| 3 | `bootstrap_models.py`, `setup_database.py` and `run_backtest_production.py` moved to `scripts/`. Each sets the repository root on its path; `bootstrap_models` keeps its models in `data/rl_models` at the root. The README's 4 commands and tree, and the estimate document, updated. | Each resolves its imports when started from `/tmp` |
| 4 | CL1: two characterisation tests of `cloud/backup.py`. They pin the dataset each kind of file goes to; that hidden files and a live run's task files are skipped; that empty pieces wait; that only changed pieces upload. | Pass on unchanged code |
| 5 | I1: a characterisation test of the Space scheduler's job table: six jobs at their IST times, none without the secrets. | Pass on unchanged code |
| 6 | Mojibake repaired in 13 comment lines of `scheduler.py` and `scheduling/` | Compile; suite |

**Checks:**
- Suite: 1,092 passed in `myenv` and in the no-skips job; 958 passed and 106 skipped on a core-only install (web and scheduler tests, by design).
- Lint: the correctness rules find nothing new, except one E402 in `api/main.py`, placed after its path set-up beside the two imports already there.
- Imports: 29 workflow modules import on the workflows' install; 7 Space modules import.

**Flagged, not changed:**
- The Overfit Alert email's subject reads `Centurion â€” Overfit Alert` (`scheduling/walk_forward.py:212`). Fixing it changes an output.
- `kite_connect/trading/auto_executor.py` has 20 mojibake lines in code or strings (legacy: TS2).
- The RAG stream's error token (X9).

**Next in phase 1:**
- R1 to R3: RAG tests, one pipeline, the self-tests.
- A1 to A3: API tests, then logic out of the largest routes.
- N1: tests for `mcx_store` and `crash_recovery`.
- The studies N7, R4, D3 and Q3.

S1, a test of the daily email's HTML, is blocked: the privacy plug-in stops me reading `services/notifications/manager.py`.

---

## 1. `tools/`

14 Python files, 3,391 lines, no `__init__.py`. Audited read-only on `feature/fine-tuning` with a clean working tree.

### 1.1 What is in it

`tools/` holds four unrelated kinds of script in one flat folder.

| Group | Files | Run by |
|---|---|---|
| Nightly production jobs | `books_report.py`, `paper_heartbeat.py`, `session_failed.py` | `nse-paper-trading.yml`, `paper-heartbeat.yml`, cron-job.org's dispatch |
| Operations and backup | `ops_report.py`, `export_books.py` | The Saturday email; `runners/refresh_data.py` on the Mac |
| Research process | `preflight.py`, `prereg.py`, `lessons.py`, `guards.py`, `study.py`, `test_map.py`, `data_catalogue.py` | By hand; the test suite |
| Simulation and benchmark | `game_day.py`, `strategy_benchmark.py` | By hand; `game_day.SimKite` is also a test fixture |

**External integrations:** Neon (Postgres) through `database.paper_cloud`; the GitHub Actions REST API; healthchecks.io; SMTP through `NotificationManager`; Yahoo for NIFTY closes; Kite login status; the local parquet stores; `pytest` and `git` as subprocesses.

**Data flow of the nightly path:** the workflow runs the session, then `books_report --daily` reads each book's Neon schema, builds one email and marks the session sent. `paper_heartbeat` reads the same state to decide whether a backup run is needed, whether to start a rescue run, and what to tell the dead man's switch. `session_failed` emails when a step fails.

### 1.2 The paths that must keep working

Any move or rename has to preserve all of these.

- **Workflow commands:** `python -m tools.books_report [--daily]`, `python -m tools.paper_heartbeat [--ping | --session-finished | --rescue]`, `python -m tools.session_failed "<run> (<label>)"`.
- **Runner commands:** `python -m tools.export_books` (`runners/refresh_data.py`).
- **Imports from core code:**
  - `api/newsletter.py:94` uses `tools.books_report.load_day_books` and `nifty50`.
  - `nse_engine/validation/significance.py:218` uses `tools.books_report.load_day_books`.
  - `scheduler.py:157` uses `tools.paper_heartbeat.futures_behind_tonight`.
- **Imports from tests:** 8 test files import 12 of the 14 modules, and several patch attributes on the module object (`import tools.paper_heartbeat as hb`).
- **Path strings held as data:**
  - `lessons.OTHER` names guards as `"tools/preflight.py::def facts"` and five similar strings. The ledger check fails if a file or function name moves.
  - `test_map.AREAS` lists `tools/*.py` paths. A test fails if a listed path is gone.
- **Documentation that quotes commands:** `CLAUDE.md`, `.github/copilot-instructions.md`, `docs/README.md`, `docs/research_protocol.md`, `docs/process_loop_plan.md`, `docs/MEMORY_TRACKER.txt`.

### 1.3 Findings, by severity

**High**

| # | Finding | Evidence | Risk |
|---|---|---|---|
| H1 | Layering is inverted: core code imports from `tools/` | The three imports in 1.2 | The web API, a validation module and the scheduler depend on a scripts folder. Nothing in `tools/` can move without touching production code. |
| H2 | No linter or type checker exists | None installed in `myenv`, no config file, `tests.yml` runs pytest only | "Run the linter and type checker after each step" cannot be met today. Open lesson L-H7 records four undefined names that sat unnoticed for this reason. |
| H3 | `export_books.py` has no test | No test file names it | It is the code that keeps tokens, passwords and users' Zerodha accounts out of the backup (`SECRET_KEY` regex). A regression would leak them silently. |
| H4 | Two production files belong to no test area | `books_report.py` and `export_books.py` are missing from `test_map.AREAS`, and nothing checks that every `tools/` file is listed | Coverage of the nightly email is not reported anywhere. |

**Medium**

| # | Finding | Evidence |
|---|---|---|
| M1 | Private names of other modules used across module boundaries | `NotificationManager._send_html_email` (4 calls, on the class in 3 and on an instance in 1), `_yahoo_close`, `runners.run_nse_engine._load_data`, `diversifier_gate._live_book`, `DataService._cache`; `game_day` patches 4 private names of `order_service` |
| M2 | Duplicated code | `IST` defined 9 times (99 across the repo); the `_ROOT` / `sys.path` bootstrap in 12 files; `PaperCloudSync(get_db_manager(), schema=...)` built 10 times; `CENTURION_LIVE_MODE` parsed twice; the repo name `srees16/centurion_core` hardcoded 3 times; two different functions both called `sessions_behind` |
| M3 | Two GitHub clients that behave differently | `ops_report.github_runs` uses certifi's certificates; `paper_heartbeat._github` uses the system's. One works on a python.org Mac build, the other may not. |
| M4 | Mixed responsibilities | `books_report.daily` and `weekly` each read Neon, decide whether to send, render, send and write state. `ops_report.summarise` builds HTML inline. `game_day.py` (608 lines) holds a simulated broker, an in-memory book, process isolation, the replay and the verdict. |
| M5 | Two sources for the NSE calendar | Most tools use `services.execution.carver_pipeline.is_nse_trading_day`; `data_catalogue.doctor` uses `nse_engine.nse_calendar.is_trading_day`. An operations tool depends on a strategy pipeline module for the calendar. |
| M6 | Paths that depend on the working directory | `export_books.OUT`, `game_day.OUT`, `game_day`'s `store_ext2006` path and `strategy_benchmark --out` are relative. The other tools resolve from the repository root. |
| M7 | Magic values repeated from elsewhere | `preflight.RA1_SESSIONS = 60` repeats `capital_ladder.GO_LIVE_MIN_PAPER_SESSIONS`; workflow file names and the `"main"` branch are literals. |
| M8 | Long functions | `SimKite.open` 90 lines, `strategy_benchmark.main` 80, `game_day._replay` 74, `books_report.weekly` 71, `preflight.facts` 65 |
| M9 | Weak typing | 19 of `game_day`'s 44 functions have no return type. Results travel as untyped dicts (`check()`, `incidents()`, `block()`), so a misspelt key fails only at run time. |

**Low**

| # | Finding |
|---|---|
| L1 | Inconsistent command-line handling: `paper_heartbeat` and `session_failed` read `sys.argv` by hand, the rest use `argparse`; `sys.exit(main())` in two files, `raise SystemExit(main())` in twelve. |
| L2 | `tools/test_map.py` is named like a test file. Pytest does not collect it (`testpaths = tests`), but the name misleads. |
| L3 | Short aliases hide what is being called: `bk`, `dg`, `bs`, `ls`, `nee`, `al`, `dbc`. |
| L4 | `books_report.request_funding` imports `json` a second time as `_json`. |
| L5 | 29 broad `except Exception` blocks. Each one is deliberate and commented ("never raises"), so they stay, but there are no shared error types. |

**Checked and found clean**

- **Dead code:** none. No unused imports, and every file is referenced by a workflow, a runner, a test or a document. `strategy_benchmark.py` is legacy, and lesson L-BT1 already asks whether to test or retire it.
- **Secrets:** none in `tools/`. Tokens come from the environment only, and logs avoid URLs and book detail.
- **Circular imports:** none at import time. Almost every project import sits inside a function, which avoids cycles but hides the coupling in H1. Some of this is required: the heartbeat workflow installs a minimal package set.

### 1.4 Target structure

Two stages. Stage A gives most of the value and moves no file.

**Stage A: same files, thinner and deduplicated.** Each script keeps its name and its command line. It parses arguments, calls library code, prints and returns an exit code.

- One helper for a book's Neon store, one for the live mode, one constant for the repository name.
- One GitHub Actions client used by both `ops_report` and `paper_heartbeat`.
- The functions core code imports (`load_day_books`, `nifty50`, `futures_behind_tonight`, `night_status`) move down into the packages that own the data, with the old names re-exported from `tools/`.
- Long functions split into read, decide, render and send.

**Stage B (optional): group by job, old paths kept as aliases.**

```
tools/
  nightly/     books_report, paper_heartbeat, session_failed
  ops/         ops_report, export_books
  research/    preflight, prereg, lessons, guards, study, test_map, data_catalogue
  sim/         game_day (broker, memory book, replay, verdict as separate files)
  legacy/      strategy_benchmark
  <14 alias files at the old paths>
```

Stage B costs more than it looks. Tests patch attributes on the old module objects, so a plain re-export would make those patches miss the real code. Each alias has to register the real module under the old name (`sys.modules`), and the six path strings in `lessons.OTHER`, the paths in `test_map.AREAS` and six documents need updating. My recommendation is to finish Stage A, then decide.

### 1.5 Plan

Each step is one commit, verified by the full test suite plus the checks listed. Steps T3 to T6 touch only files no workflow runs. T1, T2 and T9 add tests or checks, so PF1 holds them until 19 Dec unless exempted. Steps T7, T8 and T10 wait for RA1.

| Step | Change | Verified by | When |
|---|---|---|---|
| T0 | Record the baseline: full test suite, and the `--help` and offline output of all 14 commands, kept locally | n/a | Now |
| T1 | Tests first: `export_books` (secret keys dropped, missing schema reported, manifest counts) on a fake engine; fill gaps in `books_report`'s send, wait and fallback paths | New tests pass on unchanged code | PF1: after 19 Dec, or now if exempted |
| T2 | Add the two missing files to `test_map.AREAS`, and a check that every `tools/` file is listed | `test_process_loop.py` | The two entries now; the new check after 19 Dec |
| T3 | Deduplicate inside `tools/`: book-store helper, live-mode helper, repository constant, the second `json` import, `RA1_SESSIONS` read from the capital ladder | Tests; command outputs identical to T0 | Now, research tools only |
| T4 | One GitHub client | Existing rescue and ops tests, which inject the transport | Now for `ops_report`; `paper_heartbeat` after RA1 |
| T5 | Split `preflight.facts` and `strategy_benchmark.main` | Tests; `preflight --offline` output identical | Now |
| T6 | Anchor the working-directory paths to the repository root | Tests; same files written when run from the root | `export_books` after T1; `game_day` after RA1 |
| T7 | Split `books_report.daily` and `weekly`, `paper_heartbeat.main`, and `game_day` into broker, book, replay and verdict | Tests; a `--dry-run` email identical byte for byte; `game_day --scenario faults` verdict identical | After RA1 |
| T8 | Move the functions core code imports out of `tools/` | Tests; needs the `nse_engine/` and `services/` audits first | After RA1 |
| T9 | Return types and typed result records; linter and type checker in CI | The new tools | After 19 Dec and decision 1 |
| T10 | Stage B folders with aliases | Tests; every command and import in 1.2 | After RA1, if wanted |

### 1.6 Changes that would alter an output, flagged for approval

None of these is made without a yes.

- **T2:** `python -m tools.test_map` prints different counts and coverage for two areas, because two files join them.
- **T4:** `paper_heartbeat` would use certifi's certificates when certifi is installed. On GitHub's runners nothing changes; on the Mac a call that fails today could succeed.
- **T6:** `export_books` and `game_day` run from another directory would write under the repository instead of under that directory.
- **L1 (not planned):** moving `paper_heartbeat` to `argparse` would make an unknown flag exit with an error, where today it is ignored. Left as it is unless asked.

### 1.7 Decisions (taken 18 Nov 2026, delegated by the owner: "without altering the current workflow and functionality of the Ind stocks module")

1. **Linter and type checker:** `ruff` (0.17.5) and `mypy` (2.4.1) in their own venv, `ProjectX/devenv`, outside the repository and outside `myenv`, so no project pin moves. Not in any requirements file. Running them in CI is a new check: after 19 Dec (T9).
2. **RA1 freeze:** yes. `books_report.py`, `paper_heartbeat.py`, `session_failed.py` and `game_day.py` are not edited before RA1.
3. **`strategy_benchmark.py`:** kept and left untouched (its T5 split dropped); lesson L-BT1 decides test or retire by 31 Jan 2027.
4. **Stage B folders:** not now. They would either change commands that the owner, the docs and the workflows type, or add 14 alias files; half the files are frozen. Decided after RA1, once Stage A is finished.
5. **Flagged output changes:** T2 yes (done); T6 yes for `export_books` (done), `game_day` after RA1; T4 moved to after RA1 whole, so both GitHub clients change once, together.
6. **PF1:** T1's characterisation tests went in now: they pin existing behaviour and add no report or email. The check in T2 that every `tools/` file is mapped waits for 19 Dec.

Two corrections to the findings, from the code:

- **M2, the path bootstrap:** it cannot be shared. A tool run as `python tools/<name>.py` can import `shared` only after its own bootstrap has put the root on `sys.path`, so the three lines stay in each file.
- **M7, `RA1_SESSIONS`:** dropped. RA1's 60 is a fixed plan (5ao); the capital ladder's minimum is a rule that may be tuned. Reading one from the other would move RA1 if the ladder changed.

### 1.8 Carried to later module audits

- A single home for `IST` and the repository root: now X7 and the `shared/` package in 0.4.
- Public names for `NotificationManager._send_html_email`, `_yahoo_close`, `_load_data` and `_live_book`.
- One NSE calendar (M5).
- Shared error types (L5).

### 1.9 Progress

**Done 18 Nov 2026 (uncommitted).** Outputs compared with the T0 baseline (in-process, fixed clock): `preflight` facts, clock and runner mode, `ops_report`'s block, `data_catalogue`, `guards` and `lessons` status, the `--help` of nine commands: identical. `test_map` differs only by the three new tests.

| Step | Change |
|---|---|
| T0 | Baseline recorded locally |
| T1 | `tests/test_paper_heartbeat.py`: three tests of `export_books` on a real SQLAlchemy engine (SQLite, a schema per attached file): every paper table copied, tokens, secrets, passwords, API keys and `kite_accounts` dropped, a missing schema listed, other tables left out, the manifest written, the database required. The secrets test fails with the filter removed. |
| T2 | `books_report.py` and `export_books.py` in the Nightly paper books area, `tools/common.py` in Research records (since 20 Nov: `shared/`) |
| T3 | `tools/common.py` (standard library only): `IST` and `REPOSITORY`; on 20 Nov folded into `shared/` (0.4), which the seven tools now import. Seven tools import them (`data_catalogue`, `guards`, `lessons`, `ops_report`, `preflight`, `prereg`, `study`); unused `timezone` / `timedelta` imports dropped |
| T5 | `preflight.facts` split into one function per section; each still appends as it goes, so a failure part-way prints what it had |
| T6 | `export_books.OUT` anchored to the repository root (the weekly refresh runs from the root: same folder) |
| Lint | Ambiguous `l` names renamed in `guards`, `ops_report`, `preflight`, `prereg` |

Checks: fast suite 1,086 passed with no skips (`myenv`); 955 passed and 103 web tests skipped on a core-only install; `ruff --select E4,E7,E9,F,B,ISC` clean on every touched file; `mypy` reports the same three errors as before.

**Lint and type baseline** (`ProjectX/devenv`; commands below). ruff's default set in 0.17 mostly asks for newer annotation syntax (`List` to `list`, `Optional[X]` to `X | None`: about 180 findings in the touched files). The repository uses the older form throughout, so new code follows it; a switch is one mechanical step for the whole repository, not per module. mypy: three `str | None` errors in `lessons.py` (an `ast.keyword` without a name, which only `**kwargs` produces; no marker uses it), left for T9.

```
../devenv/bin/ruff check --line-length 120 --select E4,E7,E9,F,B,ISC tools/
../devenv/bin/mypy --python-executable ../myenv/bin/python --ignore-missing-imports \
    --follow-imports=silent --explicit-package-bases tools/
```

**Remaining**

| Step | When |
|---|---|
| T1, the rest: `books_report`'s send, wait and fallback paths | With T7, after RA1 |
| T2's check that every `tools/` file is in an area | After 19 Dec (PF1) |
| T3 in the four frozen files (`IST`, the repository name, the book-store and live-mode helpers, the second `json` import) | After RA1 |
| T4 one GitHub client; T6 for `game_day`; T7; T8 | After RA1 |
| T9 typed records, ruff and mypy in CI | After 19 Dec |
| T10 Stage B | Decided after RA1 |

---

## 2. `nse_engine/`

62 files, 21,348 lines; 43% covered by the fast suite (the store tests add more). The best-kept package: 16 broad `except`, 20 functions without a return type, 31 over 60 lines. 26 modules are also command lines; the workflows run 6 of them (5 each night).

**Must keep working:**
- The workflow commands: `nse_engine.deployment get|--path`, `data.archive`, `data.fo_store`, `data.mcx_store build|sync`, `crash_recovery alert`, `etf_costs snapshot`.
- The config hash of `config/nse_engine_*.json` and the deployment record.
- The journal, the registries and the trial counts on disk.
- The path strings that `tools/lessons.py` and `tools/test_map.py` hold.

**Findings**

| # | Severity | Finding |
|---|---|---|
| N1 | High | The domain imports interfaces and adapters (X2): `scorecard`, `sepa` and `value` call `runners.run_nse_engine._load_data`, a private function of a script; `scorecard` and `validation.trials` import `cloud.kaggle_runner`; `validation.significance` imports `tools.books_report`; `engine.py` reads its order bands from `kite_connect.trading.{gtt_stops,nse_engine_executor}`; `sepa`, `value`, `mcx_trend`, `bond_stream` and `futures_book` use private helpers of `kite_connect.options.{backtest,structures}`. |
| N2 | High | Nightly code with no fast test: `data/mcx_store.py` 0% and `crash_recovery.py` 0%; also `scorecard.py` 15%, `data/panel.py` 16%, `diversifier_gate.py` 18%. |
| N3 | Medium | A flat package: 36 top-level modules mix the engine (engine, signals, portfolio, costs, universe, regime, drawdown), research streams (sepa, value, mcx_trend, bond_stream, futures_book, sleeves, allocator), gates and governance (paper_gate, forward_gate, golden_gates, funding_gate, capital_ladder, diversifier_gate, books, scorecard, journal, mc_bands, live_monitor, live_rules, deployment), two calendars and three cost models. |
| N4 | Medium | 180 `mypy` errors; 40 `zip` without `strict` in research loops over paired series. |
| N5 | Medium | Long files: `books.py` 1,439 lines, `scorecard.py` 1,317 (`render_markdown` 245 lines). |
| N6 | Low | `IST` defined 4 times (`nse_calendar` is the natural home). `calendar.py` (schedules) and `nse_calendar.py` (the market calendar) have names alike and jobs different. |
| N7 | Low | Referenced nowhere, to study: `calendar.last_start_at_or_before`, `costs.buy_cash_needed`, `data.fo_store.load_stock_futures`, `data.fundamentals.load_results` and `load_board_meetings`, `data.validation.assert_no_unadjusted_gaps`, `portfolio.rank_candidates`, `universe.is_refresh_position`. `futures_book.py` is reached only by its own command line; it is futures code and stays. |

**Target:** first undo N1:
- `_load_data` moves to `nse_engine.data` (the runner re-exports it).
- The order bands move to `live_rules` (the executor imports them from there).
- The research helpers move to `nse_engine/research/` (`kite_connect.options` re-exports them).
- Kaggle provenance becomes an injected function, and `significance` takes the day books as an argument.

Subpackages (`core/`, `streams/`, `gates/`) come only later and with aliases, because 26 commands and many test patches name the flat paths.

| Step | Change | Verified by | When |
|---|---|---|---|
| N1 | Characterisation tests: `mcx_store` build and sync on a small fixture; `crash_recovery alert` | New tests pass on unchanged code | Now (decision 0.7-4) |
| N2 | `_load_data` into `nse_engine.data`, re-exported | Suite; the slow store tests; plans identical for a fixture | After RA1 |
| N3 | Order bands into `live_rules`; research helpers into `nse_engine/research/` with re-exports | Suite; config hash unchanged | After RA1 |
| N4 | Inject Kaggle provenance and the day books | Suite | After RA1 |
| N5 | Unused imports, `TYPE_CHECKING` imports for quoted `pd`, `mypy` per file | Lint and `mypy` lower | After RA1 |
| N6 | Study N7's definitions | Read-only | Now |
| N7 | Subpackages with aliases | Every command and import | Decision 0.7-3 |

---

## 3. `kite_connect/`

80 files, 28,574 lines; 42% covered; 292 broad `except`, 144 functions without a return type, `IST` defined 13 times, 50 environment reads. 12 command lines (login, accounts, kill switch, live session, options).

**Must keep working:**
- The workflow commands: `auth.accounts mask|managed|remind`, `auth.daily_login remind [--force]`, `trading.live_session --stored-token [--dry-run] [--no-email] [--record-sell-path]`, `trading.kill_switch`, `options.cli demo`.
- CI's coverage floors on 14 of its files.
- Order tags and Neon state keys.
- The whole-share and intraday refusals.

**Findings**

| # | Severity | Finding |
|---|---|---|
| K1 | High | `trading/` (27 files) holds three systems. **Broker adapter:** `order_service`, `gtt_stops`, `order_status`, `kill_switch`, `nse_instruments`, `postback_alerts`, `public_log`. **The engine's books:** `live_session`, `nse_engine_executor`, `paper_trader`, the four sleeves, `book_events`, `paper_book_view`. **The legacy auto-trader:** `auto_executor` (1,903 lines), `daily_rebalancer`, `trade_monitor`, `carver_live_forecasts`, `equity_tracker`, `rsi_strategy`, `risk_manager`, `futures_monitor`, `margin_monitor`, `spread_executor`. |
| K2 | High | Live-path functions too long to review: `run_live_session` 543 lines, `order_service.place_order` 265; on the legacy side `auto_executor._place_orders` 314, `run` 301 and `trade_monitor.poll` 240. `paper_trader.py` (2,253 lines) mixes the engine's book with the legacy yfinance trailing stop (TS2). |
| K3 | High | Two-way dependency with `nse_engine` (X2). |
| K4 | Medium | Configuration: `core/config.py`, plus direct database-URL reads in `kill_switch`, `live_session` and `paper_trader`; `webhooks/timescale_handler.py` reads its own `KITE_DB_*`. |
| K5 | Medium | `webhooks/` (ticker, alert engine, Timescale persistence, 8 files) has no test. Whether the Space runs it needs confirming. |
| K6 | Medium | 135 correctness findings (37 ambiguous names, 33 unused imports) and 153 `mypy` errors. |
| K7 | Low | Options research modules without tests (TS2 item 1). `put_selling_strategy`, `covered_call_strategy` and `options_monitor` are reached by nothing; they stay (options rule). |
| K8 | Low | Referenced nowhere: `nse_universe.fetch_nse_symbols_from_kite`, `options.theory.historical_volatility`, `order_service.get_circuit_breaker_status`. Nine option helpers are used only by tests. |

**Target:** `broker/`, `books/`, `options/`, `legacy/` (or retired), `auth/`, with settings from `shared/`. Alias modules stay at the old paths, because the workflow commands, the coverage-floor paths and the test patches name them.

| Step | Change | Verified by | When |
|---|---|---|---|
| K1 | Characterisation tests for the paper trader's legacy branch and `trade_monitor` | New tests on unchanged code | Now (decision 0.7-4) |
| K2 | One `IST`, one database URL, from `shared/` | Suite | After RA1 |
| K3 | Split `run_live_session` into read, plan, send and record; `place_order` into validate, build, send and log | Suite; coverage floors; a `--dry-run` session identical | After RA1 |
| K4 | Split `trading/` into `broker/`, `books/`, `legacy/` with aliases | Every workflow command and import | After RA1, decision 0.7-3 |
| K5 | Unused imports, names, `mypy` | Lint and `mypy` lower | Options research files now; live files after RA1 |

---

## 4. `services/`

98 files, 36,107 lines in 16 subpackages; 3% covered; 376 broad `except`, 122 functions over 60 lines, 361 `mypy` errors.

**What it is:** mostly the legacy Carver path: `signals` (7,776 lines), `research` (7,624), `execution` (5,496), `risk` (3,643), `rl_bot` (2,553; untouched by your rule), `regime`, `portfolio`, `market_data`, `technical_analysis`, `decision_engine`. The deployed path needs two parts of it: `notifications/` (the nightly and weekly emails) and the calendar in `execution/carver_pipeline.py`.

**Must keep working:**
- `NotificationManager` and the alerts, as the nightly emails use them.
- The CarverPipeline, as the paper runner's fallback until X1 is settled.
- The legacy pages' API routes.
- `services/rl_bot`, left untouched.

**Findings**

| # | Severity | Finding |
|---|---|---|
| S1 | High | The deployed path depends on a legacy module: the tools, the Kite login and the executor take the calendar from `carver_pipeline.py` (1,983 lines, which imports options, HRP and HMM code at start). See also X7. |
| S2 | High | `run_full_backtest` is one function of 2,511 lines; `CarverPipeline.run` has 1,634; `run_us_carver_pipeline` 607 (US, on hold); `_run_layer_strategy` 505. |
| S3 | High | 3% coverage. Refactoring here needs tests first, or TS2's retire. |
| S4 | Medium | Notifications: the private `_send_html_email` is called from other modules (4 callers), and `render_engine_daily_report` is 281 lines. |
| S5 | Medium | Concepts duplicated with the engine: `research/walk_forward.py` and `nse_engine/validation/walk_forward.py`, `research/deflated_sharpe.py` and `nse_engine/validation/`, `execution/{futures,options}_overlay.py` and the engine's sleeves. |
| S6 | Medium | 23 module-level imports of the root `config`, `models` and `utils`; 41 statements joined with `;`. |
| S7 | Low | 25 top-level functions referenced nowhere (three in `vince_leverage_space`, two in `forecast_scalar`, `integrated_scorer._run_layer_rag` and others). 7 files reached by no entry point: `signal_quality_evaluator` (1,726 lines), `vince_leverage_space`, `strategy_tournament`, `deflated_sharpe`, `trade_returns_collector`, `execution/futures_overlay`, `tail_risk_hedge`. |

**Target:**
- `notifications/` moves out to the deployed side, with public send and render functions.
- The calendar's callers import `nse_engine.nse_calendar`.
- The rest goes behind `legacy/`, or is retired (TS2).

| Step | Change | Verified by | When |
|---|---|---|---|
| S1 | Characterisation test: `render_engine_daily_report` HTML identical for a fixture day | New test on unchanged code | Now (decision 0.7-4) |
| S2 | The calendar's callers import `nse_engine.nse_calendar` (with 0.6's rule) | Suite; heartbeat and preflight outputs identical | After RA1 |
| S3 | A public `send_html_email`, the private name kept as an alias | Suite; S1's test | After RA1 |
| S4 | The legacy subpackages: quarantine or retire | Per item | After TS2 |
| S5 | Lint and long functions in the legacy code | Tests first | Only if TS2 keeps it |

---

## 5. `api/` and `auth/`

46 files, 10,809 lines; 47% covered; 196 broad `except`, 207 of 314 functions without a return type, 112 `raise` inside `except` without `from`.

**Must keep working:**
- Every `/api/v1/*` path, request and response shape the frontend uses.
- Authentication (sessions, roles, token checks).
- The SSE event names (`token`, `chunk`, `meta`, `done`).
- `api.main`'s app and `run_api.py`'s flags.

**Findings**

| # | Severity | Finding |
|---|---|---|
| A1 | High | Routes hold business logic and I/O: route functions call scrapers, yfinance, the paper books, the engine and Kite directly (95 in-function imports of `kite_connect`, 38 of `services`, 14 of `rag_pipeline`). `routers/v1/screener.py` is 623 lines, `routers/rag.py` 404. |
| A2 | Medium | Errors: 196 broad `except` and no mapping from error types to status codes. (A 500's text is already replaced by the app's handler: X9's correction.) |
| A3 | Medium | Two route generations (X13): nine unversioned routers (`/us-stocks`, `/ind-stocks`, `/rag`, `/crypto`, `/stream`, `/ind-stocks/pipeline`, `/portfolio`, `/r22`, health) beside `v1_gateway`. `routers/rag.py` and `routers/v1/rag.py` both serve RAG. |
| A4 | Medium | `dependencies.py` and `routers/us_stocks.py` import the legacy `main.py`; the singletons in `dependencies.py` live in module globals. |
| A5 | Low | Schemas never used: `PaginationParams`, `PaginatedResponse`, `OHLCRequest`, `BacktestMetrics`. 47 unused imports. B008's `Depends(...)` defaults are FastAPI's idiom and stay. |

**Target:** routers stay HTTP-only and call `api/services/<feature>.py` (market, rag, books, kite, screener), which call the domain and the adapters. One exception handler maps error types to status codes, and one settings object holds configuration.

| Step | Change | Verified by | When |
|---|---|---|---|
| A1 | Characterisation tests per router the frontend calls (status and keys) | New tests on unchanged code | Now (decision 0.7-4) |
| A2 | Return types, `raise ... from`, unused imports | Suite; lint and `mypy` lower | Now |
| A3 | Logic out of the largest routes (screener, market, rag) into service modules, router by router | A1's tests; responses identical | Now |
| A4 | Error types mapped to status codes in the existing handler; the RAG stream's error token generic (not yet approved) | Suite | On your yes |
| A5 | Retire the unused unversioned routers | The Space's access log | On evidence and your yes |

---

## 6. `rag_pipeline/`

29 files, 17,657 lines; 18% covered. Its 129 environment reads are mostly in `config.py` (81): centralised, as wanted. 113 `print` calls.

**Must keep working:**
- The signatures of `RAGQueryEngine.query` and `query_stream`, and their SSE behaviour (RAG2, RAG3).
- The configuration's environment names and the Chroma collection names.

**Findings**

| # | Severity | Finding |
|---|---|---|
| R1 | High | About 2,700 lines of self-test code sit inside production modules (X12): `retriever.py` is 2,860 lines, 1,441 of them `_run_tests`. |
| R2 | High | `query` (457 lines) and `query_stream` (383) repeat the pipeline step by step. RAG3's fallback fix had to be made twice. |
| R3 | Medium | Four modules are exposed only through `rag_pipeline/__init__.py`'s lazy names and used by nothing: `core/context_builder` (1,168 lines), `ingestion/chunking` (1,087), `storage/triplet_export`, `utils/retrieval_evaluator`. |
| R4 | Medium | `ingestion/pdf_ingestion.py`, 2,011 lines, has no test. |
| R5 | Low | 113 `print` calls in library code; `DefaultLLMBackend`, `extract_text_from_pdf` and two evaluation functions referenced nowhere. |

**Target:** one pipeline (cache, retrieve, rerank, fallback, context, generate) that both entry points call; self-tests moved to `tests/` or deleted; logging instead of `print`.

| Step | Change | Verified by | When |
|---|---|---|---|
| R1 | Characterisation tests of `query` and `query_stream` on stand-ins (as RAG3's) | New tests on unchanged code | Now (decision 0.7-4) |
| R2 | Extract the shared stages; both entry points call them | R1's tests; RAG2 and RAG3 tests | Now |
| R3 | The `_run_tests` harnesses: useful cases into `tests/`, the rest deleted | Suite | Now (new tests: decision 0.7-4) |
| R4 | Study R3's four modules | Read-only | Now |
| R5 | `print` to `logging` | Suite | Now |

---

## 7. `database/`

13 files, 4,589 lines; 54% covered; 148 `mypy` errors (SQLAlchemy columns typed as plain attributes), 47 broad `except`.

**Must keep working:** Neon's schemas, tables and columns; `paper_cloud`'s state keys; `get_db_manager()`.

**Findings**

| # | Severity | Finding |
|---|---|---|
| D1 | High | `paper_cloud.py` is every book's persistence and runs every night, which ties its refactor to RA1. |
| D2 | Medium | Database settings are read in `connection.py` (24 reads) and again in `config.py` and ten other files (X6). |
| D3 | Medium | Two persistence styles: the repositories (`repositories/*`; `service.py` 12% covered) serve the legacy pages, while the deployed path writes through `paper_cloud`. |
| D4 | Low | ORM models never referenced: `UserWatchlist`, `AlertConfiguration`, `PaperTradingState`. Their tables may still exist, so removal is a schema change (flagged). |
| D5 | Low | Seven `== True` in queries (SQLAlchemy needs them, or `.is_(True)`). |

**Target:** one engine factory reading `shared.settings`. `paper_cloud` keeps its behaviour, and the repositories stay with the legacy pages (TS2).

| Step | Change | Verified by | When |
|---|---|---|---|
| D1 | Database settings through `shared/` | Suite; a dry-run session | After RA1 |
| D2 | SQLAlchemy 2.0 `Mapped[...]` typing | `mypy` lower; schema unchanged | After RA1 |
| D3 | Study D4's models | Read-only; no schema change without your yes | Now |

---

## 8. Root modules

10 files, 3,914 lines; 50% covered.

| File | Lines | Role |
|---|---|---|
| `cloud_paper_runner.py` | 1,550 | The paper books' nightly runner, engine flow and legacy flow |
| `config.py` | 585 | About 140 settings; imported by 76 modules |
| `utils.py` | 580 | CSV and ticker helpers, the yfinance symbol map |
| `scheduler.py` | 275 | The Space's jobs |
| `main.py` | 250 | The legacy news-sentiment pipeline |
| `models.py` | 198 | Legacy data classes (news, metrics, signals) |
| `bootstrap_models.py`, `setup_database.py`, `run_backtest_production.py`, `run_api.py` | | One-off scripts, and the API launcher |

**Must keep working:** `python cloud_paper_runner.py [--engine]`, `python scheduler.py`, `python run_api.py --host --port`, and the `config.Config` attributes.

**Findings**

| # | Severity | Finding |
|---|---|---|
| P1 | High | `cloud_paper_runner.py` holds the legacy flow and the engine flow in one script, behind X1's fail-unsafe default. It is being edited elsewhere now. |
| P2 | Medium | `config.py` builds the database URL a second time and is imported by legacy and deployed code alike. |
| P3 | Medium | `models.py` and `utils.py` serve the legacy packages; three functions in `utils.py` are referenced nowhere. |
| P4 | Low | `bootstrap_models.py`, `setup_database.py` and `run_backtest_production.py` are reached by no entry point. |

**Target:** the root holds only entry scripts. `config` moves into `shared.settings`, with `config.Config` kept as an alias; `models` and `utils` go with the legacy code; the one-off scripts go to `scripts/`.

| Step | Change | Verified by | When |
|---|---|---|---|
| P1 | Split `cloud_paper_runner` into the engine flow, the legacy flow and the shared parts (with X1) | Suite; dry run; Neon state identical | After RA1 |
| P2 | `config` into `shared.settings`, alias kept | Suite | After RA1 |
| P3 | One-off scripts to `scripts/` (commands change: flagged) | Docs updated | On your yes |

---

## 9. `runners/`, `optimizer/`, `scripts/`

16 files, 4,517 lines; `runners/` 4% covered, `optimizer/` and `scripts/` 0%.

**Must keep working:** `runners.run_nse_engine` (called three times each night, with many subcommands) and `runners.refresh_data` (the Mac's weekly refresh).

**Findings**

| # | Severity | Finding |
|---|---|---|
| Q1 | High | `run_nse_engine.py` (1,106 lines, 7% covered by the fast suite) is both the nightly planner and a research command line, and its private `_load_data` is the engine's data loader (N1). |
| Q2 | Medium | Six R21a and R22-era runners, and all of `optimizer/` (2,276 lines, 252 `print`), are reached by nothing. They reproduce April 2026 results (`docs/R21A_ARCHITECTURE_AUDIT.md`). |
| Q3 | Low | `scripts/` builds the point-in-time Nifty 500 list once (the data is committed). |

**Target:** `run_nse_engine` becomes argument parsing plus calls into `nse_engine`. The research one-offs are retired after study, or kept in an archive folder.

| Step | Change | Verified by | When |
|---|---|---|---|
| Q1 | Characterisation tests: the plan for a fixture store identical | The slow suite | After RA1 |
| Q2 | Thin `run_nse_engine` (after N2) | Q1's tests; the nightly dry run | After RA1 |
| Q3 | Study `optimizer/` and the old runners | Read-only | Now |

---

## 10. `strategies/` and `trading_strategies/`

53 files, 17,973 lines; 13% and 39% covered. `trading_strategies/` has 27 `sys.path` insertions, 99 `print` calls and 13 command lines.

**What they are:** the legacy strategy pages (BT1 made their backtest honest), and the rule functions the Carver path combines (`services` imports `strategies` 58 times).

**Must keep working:** all strategy code (your rule) and the legacy pages' outputs.

**Findings**

| # | Severity | Finding |
|---|---|---|
| L1 | Medium | Two packages for one job: `trading_strategies/` (the page strategies) imports `strategies/` (the rule functions) 67 times. |
| L2 | Medium | `sys.path` insertions and `print` calls in library code (X10). |
| L3 | Low | 19 of the 28 `trading_strategies` files are reached by no entry point (volume microstructure, patterns, `vix_calculator`). They stay. |
| L4 | Low | Referenced nowhere: nine batch functions in `strategies/` (four in `penfold_trend`, two in `masters_prediction`, and others). |

**Target:** one `strategies/` package (rules) with a `pages/` subpackage, behind `legacy/` if TS2 keeps them. Kept code moves; it is never deleted.

| Step | Change | Verified by | When |
|---|---|---|---|
| L1 | Remove the `sys.path` insertions; `print` to `logging` | `test_strategy_backtest.py` (BT1) | Now |
| L2 | Merge the two packages | Every page's output identical | After TS2 |

---

## 11. `scrapers/`

28 files, 5,544 lines; 0% covered; 62 broad `except`; 23 module-level imports of root modules.

**What it is:** news from Indian and US sites, macro (FII and DII flows), fundamentals adapters. Used by the legacy pipeline, the API's news and macro routes, and `integrated_scorer`.

**Findings**

| # | Severity | Finding |
|---|---|---|
| C1 | Medium | No tests on code that parses third parties' HTML, which breaks silently when a site changes. |
| C2 | Low | Two identical `_parse_datetime`; its own `cache.py` beside `infrastructure.cache`. |

**Target:** one adapter per source behind one interface (fetch, then items), shared date parsing, and the cache from `infrastructure`.

| Step | Change | Verified by | When |
|---|---|---|---|
| C1 | Fixture tests from saved pages | New tests | Now (decision 0.7-4) |
| C2 | Shared date parsing | C1's tests | Now |

---

## 12. `infrastructure/`, `scheduling/`, `deployment/`

16 files, 2,502 lines.

- **`infrastructure/`:** a Redis cache with an in-memory fallback, the shared nseindia.com session, an event bus, a replay engine, fault isolation, a latency tracker, monitoring, and a shutdown backup to S3-style storage.
- **`scheduling/`:** the scheduler's run cache and job log, an on-demand walk-forward audit, and retired jobs kept on purpose.
- **`deployment/`:** the Space's start script and the secret sync.

**Findings**

| # | Severity | Finding |
|---|---|---|
| I1 | Medium | `scheduler.py` and `scheduling/` hold the Space's jobs (the GitHub dispatches, the newsletter) and are 8% covered. |
| I2 | Low | `event_bus`, `replay_engine`, `execution_context`, `fault_isolation`, `latency_tracker` and `monitoring` serve the legacy auto-trader (1 to 5 importers each). |
| I3 | Low | `backup_service` backs SQLite up to MinIO or R2 at shutdown; whether a bucket is configured needs checking. |
| I4 | Low | `fno_jobs.py` and `strategy_jobs.py` are retired and kept on purpose (F&O rule). |

| Step | Change | Verified by | When |
|---|---|---|---|
| I1 | Tests of the scheduler's job table (times, days, what each dispatches) | New tests | Now (decision 0.7-4) |
| I2 | The legacy plumbing moves with `legacy/` | Per item | After TS2 |

---

## 13. `cloud/`

8 files, 2,470 lines; 0% covered; 29 `print` calls.

**What it is:** Kaggle research (the runner, the local driver, the kernel entry, the fold stitcher), this Mac's backup and restore, the set-up of a fresh clone, and heartbeats.

**Findings**

| # | Severity | Finding |
|---|---|---|
| CL1 | High | `backup.py` holds the only off-machine copy of the stores, registries and book exports, and has no test. |
| CL2 | Medium | `nse_engine` imports it for provenance (N1). |

| Step | Change | Verified by | When |
|---|---|---|---|
| CL1 | Characterisation tests: the backup's file list and what it leaves out | New tests | Now (decision 0.7-4) |
| CL2 | With N4 | | After RA1 |

---

## 14. `us_engine/` (on hold)

8 files, 3,382 lines; 0% in the fast suite. It imports `nse_engine` 12 times at start, by design: one engine, with switches that keep India's results unchanged.

Audited only, with no steps until the US module restarts. Noted for then: `membership.members_on` is referenced nowhere, and the 607-line `run_us_carver_pipeline` sits in `services/`.

---

## 15. `tests/`

24 files, 12,243 lines, organised by functionality (`tools/test_map.py`); CI runs the suite twice.

- **E1, Medium:** the store tests run only on this Mac (`tools.guards`), so CI's coverage understates the engine's.
- **E2, Low:** the largest files are `test_paper_book.py` (1,847 lines) and `test_api.py` (1,398, holding web, RAG, newsletter and market tests). A split would change `test_map`; worth it only when a file is hard to work in.
