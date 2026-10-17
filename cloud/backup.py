"""Back up everything Centurion needs that lives only on this machine, and restore it on another.

Code is in git: the public repo, plus ``research_inbox/``, which has its own
private repo.  Paper trading state is in Neon.  Secrets are in GitHub and
Hugging Face.  Everything else under ``data/nse_engine`` exists only on disk,
so it goes to private Kaggle datasets named ``centurion-backup-*``.  Those are
kept apart from the research datasets that kernels read, so a backup never
changes a run's inputs.  Each dataset keeps only its latest version (old
versions are deleted), so the quota holds one copy, about 12 GB::

    centurion-backup-runs-2026w41     the core trial registry, one ISO week of run folders
    centurion-backup-archive-2025     raw NSE / MCX / filings downloads of one year
    centurion-backup-store            the equity store (likewise store-ext2006, fo-store,
                                      mcx-store, fundamentals, us-store)
    centurion-backup-state            the small things: the other registries (runs_*),
                                      walk-forward results, scorecards, the bond store,
                                      holdout.lock,
                                      archive manifests, the scheduler scripts and the
                                      assistant's project memory

``push`` uploads only the pieces whose files changed since the last push (a
fingerprint of names and sizes, kept in ``data/nse_engine/backup_state.json``).
Finished weeks and past years therefore go up once.

One machine writes at a time.  The ``centurion-backup-lock`` dataset names it,
and a push from any other machine refuses.  On a new laptop, ``take-over``
makes it the writer, so two machines never overwrite each other's registry
weeks.

``restore`` downloads into ``data/nse_engine``, and the assistant's memory into
this user's home.  The default ``dev`` profile skips the raw archive;
``--archive 2026`` or ``--archive all`` fetches years on demand.  It is
incremental: a piece already on disk with the backed-up fingerprint is
skipped.
``data/nse_engine/kaggle_out`` is not backed up: kernel outputs stay on
Kaggle, and the runs they produced are in the registry.

    python -m cloud.backup push            # changed pieces only; --all re-uploads everything
    python -m cloud.backup status          # what would be pushed, and who holds the lock
    python -m cloud.backup restore         # dev set; --full or --archive 2025,2026 for raw data
    python -m cloud.backup take-over       # make this machine the backup writer
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import socket
import sys
import tarfile
import tempfile
import time
import uuid
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cloud.kaggle_local import _kaggle, _push_dataset, kaggle_username  # noqa: E402

logger = logging.getLogger("backup")

DATA = _ROOT / "data" / "nse_engine"
STATE_FILE = DATA / "backup_state.json"
PREFIX = "centurion-backup"
LOCK_SLUG = f"{PREFIX}-lock"
MACHINE_ID_FILE = Path.home() / ".centurion" / "machine_id"
STAGE = _ROOT / "data" / "kaggle" / "backup_stage"
#: Each piece is one gzip tar under this name.  Kaggle unpacks an uploaded ``.tar.gz``, and then for
#: some datasets cannot serve the whole one (a 404 on download, e.g. archive-2008 and runs-2026w38 in
#: Oct 2026); a name it does not recognise stays one file, which downloads on its own.
BLOB_SUFFIX = ".tgz.blob"
STORES = ("store", "store_ext2006", "fo_store", "mcx_store", "fundamentals", "us_store")
#: Task files that live inside a store folder while a long run is active.
_TASK_FILES = ("run.sh", "run.pid", "run.done", "sync.log")
_ARCHIVE_KINDS = ("equity", "delivery", "indices", "corpact", "fo", "participant", "mcx")
_RUN_DIR = re.compile(r"^(\d{4})(\d{2})(\d{2})T")
#: Small files at the top of data/nse_engine that belong to the research record.
_STATE_GLOBS = ("wf_*.json", "wf_*.csv", "*_results.json", "*_results.csv", "*_config.json",
                "registry_refresh_*.json", "refresh_status.json", "holdout.lock")


def memory_dirs(root: Path = _ROOT) -> List[Path]:
    """Where Claude Code keeps project memory when opened on the repo's parent folder, or on the repo itself.

    Claude Code names the folder after the opened path, with every character
    other than a letter or digit turned into ``-``.
    """
    base = Path.home() / ".claude" / "projects"
    return [base / re.sub(r"[^A-Za-z0-9]", "-", str(folder)) / "memory" for folder in (root.parent, root)]


def memory_dir(root: Path = _ROOT) -> Path:
    """The memory to back up: the first of :func:`memory_dirs` that exists."""
    return next((d for d in memory_dirs(root) if d.is_dir()), memory_dirs(root)[0])


def machine_files() -> Dict[str, Path]:
    """Scheduler scripts and launch agents outside the repo: name in the backup -> path on this machine."""
    home = Path.home()
    out = {f"launch_agents/{p.name}": p for p in sorted((home / "Library" / "LaunchAgents").glob("com.centurion.*.plist"))}
    due = home / "Library" / "Application Support" / "Centurion" / "refresh-due.sh"
    if due.exists():
        out["app_support/refresh-due.sh"] = due
    return out


# ------------------------------------------------------------------ pieces
def pieces(data: Path = DATA) -> Dict[str, List[Tuple[Path, str]]]:
    """Backup slug -> [(source path, path inside the archive)], for every piece present on disk."""
    out: Dict[str, List[Tuple[Path, str]]] = {}
    runs = data / "runs"
    if runs.is_dir():
        for d in sorted(p for p in runs.iterdir() if p.is_dir()):
            m = _RUN_DIR.match(d.name)
            if m:
                y, w, _ = date(*map(int, m.groups())).isocalendar()
                out.setdefault(f"{PREFIX}-runs-{y}w{w:02d}", []).append((d, f"runs/{d.name}"))
    archive = data / "archive"
    years = set()
    for kind in _ARCHIVE_KINDS:
        years |= {p.name for p in (archive / kind).glob("[0-9]" * 4) if p.is_dir()}
    years |= {p.name for p in (archive / "filings" / "docs").glob("[0-9]" * 4)}
    for lists in (archive / "filings" / "lists").glob("*"):
        years |= {p.name for p in lists.glob("[0-9]" * 4)}
    for year in sorted(years):
        items = [(archive / k / year, f"archive/{k}/{year}") for k in _ARCHIVE_KINDS if (archive / k / year).is_dir()]
        items += [(archive / "filings" / "docs" / year, f"archive/filings/docs/{year}")] \
            if (archive / "filings" / "docs" / year).is_dir() else []
        items += [(p / year, f"archive/filings/lists/{p.name}/{year}")
                  for p in sorted((archive / "filings" / "lists").glob("*")) if (p / year).is_dir()]
        if items:
            out[f"{PREFIX}-archive-{year}"] = items
    for name in STORES:
        if (data / name).is_dir():
            out[f"{PREFIX}-{name.replace('_', '-')}"] = [(data / name, name)]
    state: List[Tuple[Path, str]] = [(p, p.name) for p in sorted(data.glob("runs_*")) if p.is_dir()]
    state += [(p, p.name) for g in _STATE_GLOBS for p in sorted(data.glob(g)) if p.is_file()]
    state += [(p, f"{p.parent.name}/{p.name}") for p in (data / "runs" / "runtime_index.json",) if p.exists()]
    state += [(p, p.name) for p in (data / "scorecard", data / "bond_store") if p.is_dir()]   # bond_store: BD1
    state += [(p, f"archive/{p.relative_to(archive)}") for p in
              (archive / "reference", archive / "manifest", archive / "filings" / "manifest",
               archive / "mcx" / "manifest") if p.exists()]
    state += [(p, f"tasks/{p.parent.name}/{p.name}") for p in sorted(data.glob("*/run.sh"))]
    state += [(p, f"machine/{name}") for name, p in machine_files().items()]
    if memory_dir().is_dir():
        state.append((memory_dir(), "machine/claude_memory"))
    if STATE_FILE.exists():
        state.append((STATE_FILE, STATE_FILE.name))
    out[f"{PREFIX}-state"] = state
    return out


def _files(items: Iterable[Tuple[Path, str]]) -> List[Tuple[Path, str]]:
    """Every file under the items, with its path inside the archive (task files of a live run skipped)."""
    out = []
    for src, arc in items:
        if src.is_file():
            out.append((src, arc))
            continue
        for f in sorted(src.rglob("*")):
            if f.is_file() and not f.name.startswith(".") and not (f.parent == src and f.name in _TASK_FILES):
                out.append((f, f"{arc}/{f.relative_to(src)}"))
    return out


def fingerprint(items: Iterable[Tuple[Path, str]]) -> str:
    h = hashlib.sha256(b"backup-v1")
    for f, arc in _files(items):
        h.update(f"{arc}:{f.stat().st_size}".encode())
    return h.hexdigest()[:20]


def _load_state() -> Dict[str, Dict[str, str]]:
    try:
        return json.loads(STATE_FILE.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def plan(push_all: bool = False) -> Dict[str, str]:
    """slug -> fingerprint for every piece to upload now."""
    done = _load_state()
    todo = {}
    for slug, items in pieces().items():
        if not _files(items):
            continue                      # nothing on disk yet (e.g. a store still being built)
        fp = fingerprint(items)
        if push_all or slug == f"{PREFIX}-state" or done.get(slug, {}).get("fingerprint") != fp:
            todo[slug] = fp
    return todo


# ------------------------------------------------------------------ writer lock
def _kaggle_retry(*args: str, capture: bool = False, attempts: int = 3) -> str:
    """A Kaggle CLI read or download, retried with backoff: dropped connections are common mid-upload."""
    for attempt in range(1, attempts + 1):
        try:
            return _kaggle(*args, capture=capture)
        except SystemExit:
            if attempt == attempts:
                raise
            logger.warning("kaggle %s failed (attempt %d); retrying", args[1] if len(args) > 1 else args[0], attempt)
            time.sleep(15 * attempt)
    return ""


def machine_id() -> str:
    """A stable name for this machine: hostname plus a random suffix, made once."""
    try:
        return MACHINE_ID_FILE.read_text().strip()
    except FileNotFoundError:
        mid = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
        MACHINE_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
        MACHINE_ID_FILE.write_text(mid)
        return mid


def remote_lock() -> Optional[Dict[str, str]]:
    """The backup writer on Kaggle, or None before any machine has claimed it.

    A failed listing or download raises (SystemExit), so a network fault can
    never look like a free lock.
    """
    ref = f"{kaggle_username()}/{LOCK_SLUG}"
    if ref not in _kaggle_retry("datasets", "list", "-m", "-s", LOCK_SLUG, capture=True):
        return None
    with tempfile.TemporaryDirectory(prefix="lock_") as tmp:
        _kaggle_retry("datasets", "download", "-d", ref, "-p", tmp, "--unzip", "--force", capture=True)
        found = next(Path(tmp).rglob("lock.json"), None)
        return json.loads(found.read_text()) if found else None


def take_over() -> Dict[str, str]:
    """Make this machine the only one that pushes backups."""
    lock = {"writer": machine_id(), "since": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    stage = STAGE / LOCK_SLUG
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    (stage / "lock.json").write_text(json.dumps(lock, indent=1))
    _push_dataset(stage, LOCK_SLUG, "Centurion backup writer lock", "take-over", wait=True)
    shutil.rmtree(stage)
    return lock


def _check_writer() -> None:
    lock = remote_lock()
    if lock is None:
        logger.info("no backup writer yet: this machine (%s) takes the lock", take_over()["writer"])
    elif lock.get("writer") != machine_id():
        raise SystemExit(f"backups are written by {lock.get('writer')} (since {lock.get('since')}). If this "
                         "machine is now the active one, run: python -m cloud.backup take-over")


# ------------------------------------------------------------------ push
def _stage(slug: str, items: Sequence[Tuple[Path, str]]) -> Path:
    """A folder holding the piece as ``<slug>`` + ``BLOB_SUFFIX``, written straight from the source files."""
    stage = STAGE / slug
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    with tarfile.open(stage / f"{slug}{BLOB_SUFFIX}", "w:gz") as tar:
        for f, arc in _files(items):
            tar.add(f, arcname=arc)
    return stage


def push(push_all: bool = False) -> Dict[str, str]:
    """Upload every changed piece; the state piece goes last, so it records the others."""
    _check_writer()
    todo = plan(push_all)
    state = _load_state()
    results: Dict[str, str] = {}
    ordered = sorted(s for s in todo if s != f"{PREFIX}-state") + [s for s in todo if s == f"{PREFIX}-state"]
    for slug in ordered:
        if slug == f"{PREFIX}-state":
            STATE_FILE.write_text(json.dumps(state, indent=1, sort_keys=True))
        items = pieces()[slug]
        try:
            stage = _stage(slug, items)
            size = sum(p.stat().st_size for p in stage.glob(f"*{BLOB_SUFFIX}"))
            logger.info("%s: uploading %.0f MB", slug, size / 1e6)
            _push_dataset(stage, slug, f"Centurion backup {slug[len(PREFIX) + 1:]}",
                          time.strftime("backup %Y-%m-%d"), wait=False)
            shutil.rmtree(stage)
        except SystemExit as exc:            # _kaggle raises SystemExit on a failed CLI call
            logger.warning("%s: upload failed (%s); retried on the next push", slug, str(exc)[:200])
            results[slug] = "failed"
            continue
        state[slug] = {"fingerprint": todo[slug], "pushed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        STATE_FILE.write_text(json.dumps(state, indent=1, sort_keys=True))
        results[slug] = "pushed"
    return results


# ------------------------------------------------------------------ restore
def backup_refs() -> List[str]:
    """Every centurion-backup dataset this account owns."""
    refs, page = [], 1
    while True:
        out = _kaggle_retry("datasets", "list", "-m", "-s", PREFIX, "--csv", "--page", str(page), capture=True)
        rows = [line.split(",")[0] for line in out.strip().splitlines()[1:] if line.strip()]
        new = [r for r in rows if r.split("/")[-1].startswith(PREFIX) and r not in refs]
        if not new:
            return sorted(refs)
        refs += new
        page += 1


def _download(ref: str, into: Path) -> None:
    """The piece's one archive file; a piece pushed before ``BLOB_SUFFIX`` (Kaggle unpacked it) comes down whole."""
    blob = f"{ref.split('/')[-1]}{BLOB_SUFFIX}"
    if blob in _kaggle_retry("datasets", "files", ref, "--csv", capture=True):
        _kaggle_retry("datasets", "download", "-d", ref, "-f", blob, "-p", str(into), "--force")
    else:
        _kaggle_retry("datasets", "download", "-d", ref, "-p", str(into), "--unzip", "--force")


def _unpack_into(download: Path, dest: Path) -> None:
    """Unpack the piece's archive, or copy a piece Kaggle served already unpacked.

    Unpacked, the files keep their paths inside the archive (``mcx_store/...``), so they are copied as
    they are; only a single wrapper folder named after the backup (``centurion-backup-...``) is dropped.
    """
    archives = sorted(download.rglob(f"*{BLOB_SUFFIX}")) + sorted(download.rglob("*.tar.gz"))
    if archives:
        for a in archives:
            shutil.unpack_archive(str(a), str(dest), format="gztar")
        return
    roots = list(download.iterdir())
    inner = roots[0] if len(roots) == 1 and roots[0].is_dir() and roots[0].name.startswith(PREFIX) else download
    shutil.copytree(inner, dest, dirs_exist_ok=True)


def restore(dest: Path = DATA, refs: Optional[Sequence[str]] = None, full: bool = False,
            archive_years: Sequence[str] = ()) -> Dict[str, str]:
    """Download backups into ``dest``: the state piece first, then the rest, skipping pieces already current.

    Without ``refs``: every piece except the raw archive (the dev set), plus the
    archive years asked for (``"all"`` for every year), or everything with ``full``.
    """
    dest.mkdir(parents=True, exist_ok=True)
    available = backup_refs()
    if refs:
        chosen = list(refs)
    else:
        def wanted(ref: str) -> bool:
            slug = ref.split("/")[-1]
            if slug == LOCK_SLUG:
                return False
            if slug.startswith(f"{PREFIX}-archive-"):
                year = slug.rsplit("-", 1)[-1]
                return full or "all" in archive_years or year in archive_years
            return True
        chosen = [r for r in available if wanted(r)]
    chosen.sort(key=lambda r: (not r.endswith(f"/{PREFIX}-state"), r))
    results: Dict[str, str] = {}
    remote: Dict[str, Dict[str, str]] = {}
    for ref in chosen:
        slug = ref.split("/")[-1]
        local = pieces(dest).get(slug)
        if slug != f"{PREFIX}-state" and local and _files(local) and \
                remote.get(slug, {}).get("fingerprint") == fingerprint(local):
            results[ref] = "up to date"
            continue
        with tempfile.TemporaryDirectory(prefix="restore_") as tmp:
            _download(ref, Path(tmp))
            _unpack_into(Path(tmp), dest)
        if slug == f"{PREFIX}-state" and (dest / STATE_FILE.name).exists():
            remote = json.loads((dest / STATE_FILE.name).read_text())
        results[ref] = "restored"
        logger.info("%s restored", ref)
    machine = dest / "machine"
    if (machine / "claude_memory").is_dir():
        for target in memory_dirs():          # either folder may be the one opened on this laptop
            shutil.copytree(machine / "claude_memory", target, dirs_exist_ok=True)
    for task in sorted((dest / "tasks").glob("*/run.sh")) if (dest / "tasks").is_dir() else []:
        target = dest / task.parent.name / "run.sh"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(task, target)
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("push", help="upload the pieces that changed since the last push")
    q.add_argument("--all", action="store_true", help="re-upload every piece")
    sub.add_parser("status", help="list the pieces a push would upload, and the backup writer")
    sub.add_parser("take-over", help="make this machine the only one that pushes backups")
    r = sub.add_parser("restore", help="download backups into data/nse_engine (dev set by default)")
    r.add_argument("--dest", default=str(DATA))
    r.add_argument("--full", action="store_true", help="also every year of the raw archive")
    r.add_argument("--archive", default="", help="raw archive years to fetch, comma list or 'all'")
    r.add_argument("--only", help="comma list of dataset refs (overrides the profile)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cmd == "status":
        todo = plan()
        sizes = {s: sum(f.stat().st_size for f, _ in _files(pieces()[s])) for s in todo}
        for s in sorted(todo):
            print(f"{s:45s} {sizes[s] / 1e6:9.1f} MB")
        print(f"{len(todo)} pieces, {sum(sizes.values()) / 1e9:.2f} GB")
        lock = remote_lock()
        print(f"backup writer: {lock.get('writer') if lock else 'none yet'}; this machine: {machine_id()}")
        return 0
    if args.cmd == "take-over":
        print(json.dumps(take_over(), indent=1))
        return 0
    if args.cmd == "push":
        res = push(args.all)
        print(json.dumps(res, indent=1))
        return 1 if "failed" in res.values() else 0
    only = [s.strip() for s in args.only.split(",")] if args.only else None
    years = [y.strip() for y in args.archive.split(",") if y.strip()]
    print(json.dumps(restore(Path(args.dest), only, args.full, years), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
