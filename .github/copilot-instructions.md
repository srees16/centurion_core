# Copilot instructions for centurion_core

The full guide is `docs/README.md`, Quick Start, under "Engine, research
data and coding assistants". Read it before changing code. The essentials:

- **First time on a laptop**, after you install git, Python 3.13 and the
  GitHub CLI, sign in to GitHub (`gh auth login`) and add a Kaggle API token
  to `~/.kaggle/`:
  ```
  python3.13 -m venv ../myenv && source ../myenv/bin/activate
  pip install -r requirements.txt
  python -m cloud.devsetup        # clones the books, makes .env, restores the data, prints a checklist
  ```
  Fill `.env` by hand; secrets cannot be fetched. Check the state any time
  with `python -m cloud.devsetup check`.
- **Data** (`data/nse_engine/`: stores, trial registries, raw archive) comes
  from private Kaggle datasets: `python -m cloud.backup restore`, adding
  `--archive <years>` for the raw downloads. After fetching new data or
  rebuilding a store, run `python -m cloud.backup push`.
- **Public repository:** never commit secrets, `.env`, credentials or the
  PDFs in `research_inbox/`.
- **Scope:** change only what the task needs; keep unrelated behaviour
  intact; reuse existing utilities; delete temporary scripts; do not commit
  or push unless asked.
- **Trading:** swing and positional only, never intraday. Research follows
  `docs/research_protocol.md` (pre-register first; every trial counts).
- **Tests:** `python -m pytest tests -q -m "not slow"`.
