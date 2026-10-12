"""Set up a fresh clone of centurion_core: check prerequisites, then fetch what lives off GitHub.

Run once on a new laptop, from the repo root, inside the project venv::

    python -m cloud.devsetup                      # check, clone the books, make .env, restore the dev data
    python -m cloud.devsetup check                # report only (what an assistant should run first)
    python -m cloud.devsetup --install-scheduler  # macOS: also install the weekly refresh and backup job

Each check prints OK, or the exact step that fixes it.  Things the script
cannot do for you (installing tools, signing in, typing secrets) are listed,
never guessed:

- **GitHub:** ``gh`` signed in, for the private research books
  (``srees16/centurion-research``, cloned into ``research_inbox/``).
- **Kaggle:** the CLI with an API token, for the data backups
  (``cloud.backup``) and research runs.
- **Secrets:** ``.env``, copied from ``.env.example`` and filled by you.
  GitHub and Hugging Face secrets are write-only, so no tool can fetch them.
- **Data:** ``cloud.backup restore``, the dev set by default; the raw archive
  on demand.
- **Backups:** one machine writes them; ``python -m cloud.backup take-over``
  once this laptop is the active one.
- **Scheduler:** the weekly refresh and backup must run on a laptop, because
  NSE and MCX block cloud servers.  macOS uses a launch agent; other systems
  get the line to schedule.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

BOOKS_REPO = "srees16/centurion-research"
CORE_MODULES = ("pandas", "numpy", "pyarrow", "requests", "lxml", "curl_cffi", "dotenv", "sqlalchemy")
LABEL = "com.centurion.data-refresh"

Check = Tuple[str, bool, str]          # (name, ok, detail or the fix)


# ------------------------------------------------------------------ checks
def _run(*cmd: str) -> Tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return p.returncode, (p.stdout + p.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)


def check_python() -> Check:
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    ok = sys.version_info >= (3, 11) and in_venv
    return ("python", ok, f"{platform.python_version()} in {sys.prefix}" if ok else
            "use Python 3.11+ in a venv: python3.13 -m venv ../myenv && source ../myenv/bin/activate")


def check_packages() -> Check:
    missing = [m for m in CORE_MODULES if importlib.util.find_spec(m) is None]
    return ("packages", not missing, "core packages present" if not missing else
            f"missing {', '.join(missing)}: pip install -r requirements.txt (or requirements-core.txt for the engine only)")


def gh_path() -> Optional[str]:
    """The GitHub CLI on PATH, or in the usual user / Homebrew locations."""
    found = shutil.which("gh")
    if found:
        return found
    for cand in (Path.home() / ".local" / "bin" / "gh", Path("/opt/homebrew/bin/gh"), Path("/usr/local/bin/gh")):
        if cand.exists():
            return str(cand)
    return None


def check_gh() -> Check:
    gh = gh_path()
    if not gh:
        return ("github", False, "install the GitHub CLI (https://cli.github.com), then: gh auth login")
    code, out = _run(gh, "auth", "status")
    return ("github", code == 0, "gh signed in" if code == 0 else "gh auth login  (GitHub.com, HTTPS, browser)")


def check_kaggle() -> Check:
    from cloud.kaggle_local import check as kaggle_check
    state = kaggle_check()
    if state.get("kaggle_cli") == "missing":
        return ("kaggle", False, "pip install -r requirements.txt  (the kaggle CLI)")
    if "authenticated_as" in state:
        return ("kaggle", True, f"signed in as {state['authenticated_as']}")
    return ("kaggle", False, "kaggle.com > Settings > API > Create New Token; save it in ~/.kaggle/ "
            "(kaggle.json or access_token), then chmod 600 ~/.kaggle/*")


def check_books() -> Check:
    inbox = _ROOT / "research_inbox"
    ok = (inbox / ".git").is_dir()
    return ("research books", ok, str(inbox) if ok else
            f"gh repo clone {BOOKS_REPO} research_inbox  (setup does this when gh is signed in)")


def check_env() -> Check:
    env = _ROOT / ".env"
    if not env.exists():
        return ("secrets (.env)", False, "cp .env.example .env, then fill the values you need")
    names = [line.split("=", 1)[0].strip() for line in (_ROOT / ".env.example").read_text().splitlines()
             if "=" in line and not line.lstrip().startswith("#")]
    values = {}
    for line in env.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip()
    empty = [n for n in names if not values.get(n)]
    return ("secrets (.env)", True, ".env present" + (f"; empty: {', '.join(empty)}" if empty else ""))


def check_data() -> Check:
    data = _ROOT / "data" / "nse_engine"
    have = [n for n in ("store", "fo_store", "mcx_store", "runs") if (data / n).is_dir()]
    ok = len(have) == 4
    return ("data", ok, f"present: {', '.join(have)}" if ok else
            "python -m cloud.backup restore  (dev set; add --archive all for the raw downloads)")


def check_writer() -> Check:
    from cloud.backup import machine_id, remote_lock
    try:
        lock = remote_lock()
    except SystemExit as exc:
        return ("backup writer", False, f"could not read the lock: {str(exc)[:120]}")
    me = machine_id()
    if lock and lock.get("writer") == me:
        return ("backup writer", True, f"this machine ({me})")
    owner = lock.get("writer") if lock else "none yet"
    return ("backup writer", False, f"writer is {owner}; when this laptop is the active one: "
            "python -m cloud.backup take-over")


def check_neon() -> Check:
    url = os.getenv("CENTURION_DATABASE_URL")
    if not url:
        return ("paper state (Neon)", False, "set CENTURION_DATABASE_URL in .env (a read-only role is enough to develop)")
    try:
        from sqlalchemy import create_engine, text
        with create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 10}).connect() as conn:
            conn.execute(text("select 1"))
        return ("paper state (Neon)", True, "reachable")
    except Exception as exc:                          # noqa: BLE001 - reported, not raised
        return ("paper state (Neon)", False, f"not reachable: {type(exc).__name__}")


CHECKS: Sequence[Callable[[], Check]] = (check_python, check_packages, check_gh, check_kaggle, check_books,
                                         check_env, check_data, check_writer, check_neon)


def report(checks: Sequence[Callable[[], Check]] = CHECKS) -> List[Check]:
    try:
        from dotenv import load_dotenv
        load_dotenv(_ROOT / ".env")
    except ImportError:
        pass
    rows = []
    for fn in checks:
        try:
            rows.append(fn())
        except Exception as exc:                      # noqa: BLE001 - one broken check must not hide the rest
            rows.append((fn.__name__.replace("check_", ""), False, f"check failed: {exc}"))
    for name, ok, detail in rows:
        print(f"[{'OK' if ok else '!!'}] {name:20s} {detail}")
    return rows


# ------------------------------------------------------------------ actions
def clone_books() -> None:
    gh = gh_path()
    if not (_ROOT / "research_inbox" / ".git").is_dir() and gh:
        code, out = _run(gh, "repo", "clone", BOOKS_REPO, str(_ROOT / "research_inbox"))
        print(f"research books: {'cloned' if code == 0 else 'not cloned: ' + out[:200]}")


def make_env() -> None:
    env, example = _ROOT / ".env", _ROOT / ".env.example"
    if not env.exists() and example.exists():
        shutil.copy2(example, env)
        os.chmod(env, 0o600)
        print(".env created from .env.example: fill in the values you need (it is gitignored)")


def install_scheduler() -> None:
    """macOS: the weekly refresh + backup as a launch agent; elsewhere, print what to schedule."""
    py = sys.executable
    if platform.system() != "Darwin":
        print("schedule weekly (cron / Task Scheduler), from the repo root:\n"
              f"  {py} -m runners.refresh_data\n"
              "It refreshes the stores and ends by pushing the backup.")
        return
    home = Path.home()
    run_sh = _ROOT / "data" / "nse_engine" / "refresh" / "run.sh"
    due_sh = home / "Library" / "Application Support" / "Centurion" / "refresh-due.sh"
    plist = home / "Library" / "LaunchAgents" / f"{LABEL}.plist"
    for p in (run_sh, due_sh, plist):
        p.parent.mkdir(parents=True, exist_ok=True)
    run_sh.write_text(RUN_SH.format(repo=_ROOT, py=py, bindir=Path(py).parent))
    due_sh.write_text(DUE_SH.format(run_sh=run_sh))
    plist.write_text(PLIST.format(label=LABEL, due_sh=due_sh, log=home / "Library" / "Logs" / "centurion-data-refresh.log"))
    for p in (run_sh, due_sh):
        p.chmod(0o755)
    uid = os.getuid()
    _run("launchctl", "bootout", f"gui/{uid}/{LABEL}")
    code, out = _run("launchctl", "bootstrap", f"gui/{uid}", str(plist))
    print(f"weekly refresh + backup: {'installed' if code == 0 else 'not loaded: ' + out[:200]}")
    print("If macOS asks whether Terminal may access the folder holding the repo, allow it.")


RUN_SH = """#!/bin/zsh
# Weekly research-data refresh and backup (written by python -m cloud.devsetup --install-scheduler).
# Opened in Terminal by refresh-due.sh only when the last clean run is 6+ days old; hands over to a
# detached worker so the Terminal window holds no process.  Up to 3 attempts an hour apart.
DIR={repo}/data/nse_engine/refresh
STAMP="$HOME/Library/Application Support/Centurion/refresh_last_ok"
if [[ "$1" != --worker ]]; then
  nohup /bin/zsh $DIR/run.sh --worker </dev/null >/dev/null 2>&1 &
  exit 0
fi
exec >> $DIR/refresh.log 2>&1
cd {repo} || exit 1
if [[ -f $DIR/run.pid ]] && ps -p "$(cat $DIR/run.pid)" -o command= 2>/dev/null | grep -q "refresh/run.sh"; then
  echo "$(date '+%F %T') another refresh is active; this one exits"
  exit 0
fi
echo $$ > $DIR/run.pid
caffeinate -s -w $$ &
export PATH={bindir}:$PATH
for attempt in 1 2 3; do
  echo "$(date '+%F %T') refresh attempt $attempt"
  if {py} -m runners.refresh_data; then
    mkdir -p "${{STAMP:h}}" && date +%s > "$STAMP"
    echo "$(date '+%F %T') refresh complete"
    exit 0
  fi
  echo "$(date '+%F %T') refresh incomplete; retrying in 1 h"
  sleep 3600
done
exit 1
"""

DUE_SH = """#!/bin/zsh
# Opens the weekly refresh in Terminal (which may read protected folders) only when it is due.
STAMP="$HOME/Library/Application Support/Centurion/refresh_last_ok"
last=$(cat "$STAMP" 2>/dev/null || echo 0)
(( $(date +%s) - last >= 6 * 86400 )) || exit 0
exec /usr/bin/open -g -a Terminal {run_sh}
"""

PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{label}</string>
  <key>ProgramArguments</key><array><string>/bin/zsh</string><string>{due_sh}</string></array>
  <key>RunAtLoad</key><true/>
  <key>StartCalendarInterval</key><dict><key>Hour</key><integer>20</integer><key>Minute</key><integer>30</integer></dict>
  <key>StandardOutPath</key><string>{log}</string>
  <key>StandardErrorPath</key><string>{log}</string>
</dict>
</plist>
"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", nargs="?", default="setup", choices=["setup", "check"])
    p.add_argument("--no-data", action="store_true", help="skip the data restore")
    p.add_argument("--install-scheduler", action="store_true", help="install the weekly refresh + backup job")
    args = p.parse_args(argv)
    os.chdir(_ROOT)
    if args.command == "setup":
        clone_books()
        make_env()
        ok = dict((n, o) for n, o, _ in report((check_kaggle,)))
        if not args.no_data and ok.get("kaggle"):
            from cloud.backup import restore
            restore()
        if args.install_scheduler:
            install_scheduler()
        print()
    rows = report()
    return 0 if all(ok for _, ok, _ in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
