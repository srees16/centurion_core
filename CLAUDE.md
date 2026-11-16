# Claude Code briefing for centurion_core

The full guide is in `docs/README.md` under Quick Start, "Engine, research
data and coding assistants". Read it before changing code. The essentials:

- **First time on a laptop**, after you install git, Python 3.13 and the
  GitHub CLI, sign in to GitHub (`gh auth login`) and add a Kaggle API token
  to `~/.kaggle/`:
  ```
  python3.13 -m venv ../myenv && source ../myenv/bin/activate
  pip install -r requirements.txt
  python -m cloud.devsetup        # clones the books, makes .env, restores the data, prints a checklist
  ```
  Fill `.env` by hand; secrets cannot be fetched. Start each session with
  `python -m cloud.devsetup check`.
- **Data** (`data/nse_engine/`: stores, trial registries, raw archive) comes
  from private Kaggle datasets: `python -m cloud.backup restore`, adding
  `--archive <years>` for the raw downloads. After fetching new data or
  rebuilding a store, run `python -m cloud.backup push`. Only the writer
  machine pushes; `take-over` moves the lock.
- **Public repository:** never commit secrets, `.env`, credentials or the
  PDFs in `research_inbox/`.
- **Scope:**
  - change only what the task needs, and keep unrelated behaviour intact;
  - reuse existing utilities;
  - delete temporary scripts;
  - do not commit or push unless asked;
  - ask before creating any new `.md` file.
- **Trading:** swing and positional by default. Intraday is allowed only where the
  evidence shows it adds to CAGR and Sharpe (owner's amendment, 31 Oct 2026), and
  `order_service.intraday_refusal` stays on until a stream has passed its gates.
  Research follows
  `docs/research_protocol.md`: pre-register first; every trial counts;
  2026 is the sealed holdout.
- **Ind vs US stocks:**
  - India trades whole shares only, and its paper books and automated orders
    run through Kite. `order_service.whole_share_refusal` guards the order
    and GTT paths.
  - US stocks may be held in fractional shares, but there are no automated or
    broker paper orders: the US module only advises, and the owner places
    every order.
  - Options (buying, selling or writing) are possible for Ind stocks through
    Kite, and impossible for US stocks from a non-US DriveWealth account.
    The DriveWealth route refuses option symbols.
  - One engine for both: US backtests go through `us_engine.backtest`, which
    switches on `universe.members_only`, `portfolio.fractional_shares` and
    `costs.market="US"`. All three default off, so India's config hashes and
    results never move; keep any new US behaviour behind such a switch.
- **Tests:** `python -m pytest tests -q -m "not slow"`. The test files are
  organised by the functionality they guard (owner's ask, 15 Nov 2026):
  `python -m tools.test_map` lists each area, its code and its tests, and a
  new test file must be added to its `AREAS`.
  - A bug fix keeps one regression test, in the existing file for its
    functionality; throwaway verification scripts are deleted.
  - CI runs the suite twice: on the core install, and in a full job with the
    web and Kite packages where no test may skip. The deploy to the Space
    waits for it; the nightly run never does.
- **Process loop** (tracker PL0, `docs/process_loop_plan.md`): steps that used
  to be remembered are now checked.
  - Start a session with `python -m tools.preflight`: computed dates,
    sessions, RA1's date, trial counts and what the records are missing.
  - Before pre-registering: `python -m tools.prereg prior-art <words>`, then
    the template's Phase 0 block, filled, with a GO or STOP verdict.
    `python -m tools.prereg new` prints the template and, with
    `--body <file> --write`, numbers the section, stamps the clock's time and
    adds it to the plan: never type a pre-registration time.
  - On every verdict: state it right after the tracker entry's title
    ("...): FAIL, ..."), add the journal row with its post-mortem
    (`python -m nse_engine.journal experiment ...`), and list the study's
    files with `python -m tools.study close --id <ID>` for approval.
  - A lesson counts only with a guard: mark its test
    `@pytest.mark.lesson("L-<id>", ...)` or add it to `tools/lessons.py`, then
    `python -m tools.lessons build`.
  - After recording a trial: `python -m nse_engine.validation.families --write`.
  - Before a handover that touched the engine, the stores or the costs, and
    at least weekly: `python -m tools.guards` (the store guards CI cannot
    run, and the data catalogue with its store doctor; about seven minutes).
    A new weekend or short session the doctor reports is checked, then
    acknowledged: `python -m tools.data_catalogue ack <date> --reason ...`.
  - The preflight's "Operations" line is the week's incidents and the loop's
    health (`python -m tools.ops_report`); an incident type with no lesson
    gets one, or a reason it needs none.
  - Cost rates: `python -m nse_engine.validation.rates` lists each dated rate
    with its source and the published rates not applied yet. A new rate is
    never typed into a model without a row there.
- **Communication:** be candid when something misses the goal or is
  premature, and keep summaries crisp.
- **Challenge every request first** (owner's rule, 7 Nov 2026): before any
  work, gauge whether it adds value to the Ind stocks module (or US, when
  applicable): CAGR, Sharpe, robustness, workflow. Search the trackers and
  `docs/nse_engine_validation_plan.md` for prior tests and decisions. Weigh
  its cost (trials, freezes, guards, time against go-live work). If it is
  counter-productive, a duplicate or low-value, say so crisply with the
  reason and a better alternative, and ask before proceeding. Do not agree by
  default (14 Nov 2026): for an idea or strategy, name its biggest weakness,
  the assumption that may be false, what is being overlooked and what could
  make it fail. Keep `docs/MEMORY_TRACKER.txt`
  (India) and `docs/US_MEMORY_TRACKER.txt` (US) current.
- **Think it through before implementing** (owner's rule, 14 Nov 2026): after the
  challenge and before any change, reason step by step and state a short plan:
  the real goal; the root cause confirmed from evidence (logs, data, the code
  path), not inferred; at least one alternative and why the chosen fix is
  better; what else the change touches and how it can fail; how it will be
  verified, on real data where possible, and what cannot be. If the evidence
  contradicts the plan midway, stop and re-plan. For an objective, work
  backward from the outcome wanted: the steps, skills, tools, decisions and
  milestones, and the most efficient path. Keep the write-up to a few
  lines; a one-line change needs one line.
- **Ask before answering ideas and decisions** (owner's rule, 14 Nov 2026): for
  an idea, strategy, objective or design choice, and anything ambiguous,
  costly or hard to undo, first say what the owner is really trying to
  solve, ask the questions whose answers would change the answer (at most
  four), then stop: read-only investigation only until they answer.
  Operational, status and time-critical prompts get a direct answer with the
  assumption stated. Replies stay within one screen.
