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
- **Tests:** `python -m pytest tests -q -m "not slow"`.
- **Communication:** be candid when something misses the goal or is
  premature, and keep summaries crisp. Keep `docs/MEMORY_TRACKER.txt`
  (India) and `docs/US_MEMORY_TRACKER.txt` (US) current.
