"""Drive a Kaggle research job from here: upload, push, watch, pull.

Two datasets, so a code change does not re-upload 200 MB of prices:

    <user>/centurion-nse-code    nse_engine/, runners/, cloud/, config/, job.json
    <user>/centurion-nse-store   data/nse_engine/store (the parquet store)

Typical run of the 2013–2025 walk-forward, four folds at a time:

    python -m cloud.kaggle_local push-store                 # once, then on new data
    python -m cloud.kaggle_local run --task walk-forward \\
        --args '--grid {...} --start 2013-01-01 --end 2025-12-31 \\
                --data-start 2011-01-01 --folds 0-3 --workers 4'
    python -m cloud.kaggle_local watch
    python -m cloud.kaggle_local pull
    python -m cloud.wf_stitch import-runs --src data/nse_engine/kaggle_out/latest/runs
    python -m cloud.wf_stitch stitch --dirs data/nse_engine/kaggle_out/*/wf

Needs the Kaggle CLI and an API token (``~/.kaggle/kaggle.json``, from Kaggle →
Account → Create New API Token); ``check`` reports what is missing.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger("kaggle_local")

CODE_SLUG = "centurion-nse-code"
STORE_SLUG = "centurion-nse-store"
KERNEL_SLUG = "centurion-nse-research"   # default; --kernel-suffix runs a second one alongside
STAGE_DIR = _ROOT / "data" / "kaggle" / "stage"
OUT_DIR = _ROOT / "data" / "nse_engine" / "kaggle_out"

# Only what a research run needs; the store travels as its own dataset.
CODE_PATHS = ["nse_engine", "runners", "cloud", "config", "requirements.txt"]


# ── kaggle CLI ───────────────────────────────────────────────────

def kaggle_username() -> str:
    """Resolve the account name, whichever way the CLI is authenticated.

    Three ways exist: ``KAGGLE_USERNAME``/``KAGGLE_KEY`` in the environment, the
    older ``~/.kaggle/kaggle.json``, and an access token in
    ``~/.kaggle/access_token`` (CLI 2.x), which carries the name itself — so it
    is asked for rather than read out of a file.
    """
    user = os.getenv("KAGGLE_USERNAME")
    if user:
        return user
    token = Path.home() / ".kaggle" / "kaggle.json"
    if token.exists():
        try:
            return json.loads(token.read_text())["username"]
        except (ValueError, KeyError):
            pass
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        name = api.get_config_value("username")
        if name:
            return name
    except Exception as exc:                          # noqa: BLE001 - fall through to advice
        logger.debug("username lookup via the Kaggle API failed: %s", exc)
    raise SystemExit("no Kaggle username: set KAGGLE_USERNAME, or authenticate the CLI "
                     "(~/.kaggle/kaggle.json or ~/.kaggle/access_token)")


def check() -> Dict[str, object]:
    """Report whether the CLI and credentials are usable, without failing."""
    cli = shutil.which("kaggle")
    json_token = (Path.home() / ".kaggle" / "kaggle.json").exists()
    access_token = (Path.home() / ".kaggle" / "access_token").exists()
    env = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    state: Dict[str, object] = {
        "kaggle_cli": cli or "missing",
        "kaggle_json": json_token,
        "access_token": access_token,
        "env_credentials": env,
        "store_present": (_ROOT / "data/nse_engine/store").is_dir(),
    }
    if not cli:
        state["install"] = "pip install kaggle"
    if not (json_token or access_token or env):
        state["credentials"] = "Kaggle → Settings → API → Create New Token"
        return state
    try:
        state["authenticated_as"] = kaggle_username()
    except SystemExit as exc:
        state["credentials"] = str(exc)
    return state


def _kaggle(*args: str, capture: bool = False) -> str:
    if not shutil.which("kaggle"):
        raise SystemExit("kaggle CLI not found — pip install kaggle (see `check`)")
    cmd = ["kaggle", *args]
    logger.info("$ %s", " ".join(cmd))
    proc = subprocess.run(cmd, check=False, text=True,
                          capture_output=capture, cwd=str(_ROOT))
    if proc.returncode != 0:
        out = (proc.stdout or "") + (proc.stderr or "")
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{out.strip()}")
    return (proc.stdout or "") if capture else ""


# ── datasets ─────────────────────────────────────────────────────

def _dataset_metadata(path: Path, slug: str, title: str) -> None:
    (path / "dataset-metadata.json").write_text(json.dumps({
        "title": title,
        "id": f"{kaggle_username()}/{slug}",
        "licenses": [{"name": "other"}],
    }, indent=2))


def _push_dataset(path: Path, slug: str, title: str, message: str) -> None:
    _dataset_metadata(path, slug, title)
    existing = _kaggle("datasets", "list", "-m", "-s", slug, capture=True)
    # -t keeps files as they are (Kaggle converts tabular files to CSV otherwise);
    # -r tar uploads directories whole, so the store arrives as directories, not
    # one file per parquet part. -d is only valid on `version`.
    if f"{kaggle_username()}/{slug}" in existing:
        _kaggle("datasets", "version", "-p", str(path), "-m", message, "-d", "-t", "-r", "tar")
    else:
        _kaggle("datasets", "create", "-p", str(path), "-t", "-r", "tar")
    wait_dataset_ready(slug)


def wait_dataset_ready(slug: str, timeout: int = 900, interval: int = 20) -> bool:
    """Block until Kaggle finishes ingesting the upload.

    A kernel started while its dataset is still processing mounts nothing, which
    is a confusing way to fail. ``datasets status`` is not readable with an
    access-token login, so readiness is judged by the file listing appearing.
    """
    ref = f"{kaggle_username()}/{slug}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            listing = _kaggle("datasets", "files", ref, capture=True)
        except SystemExit:
            listing = ""
        if listing and "name" in listing and len(listing.strip().splitlines()) > 2:
            logger.info("%s ready", ref)
            return True
        logger.info("waiting for %s to finish processing", ref)
        time.sleep(interval)
    logger.warning("%s still not listing files after %ds — pushing on anyway", ref, timeout)
    return False


def local_pins() -> List[str]:
    """Pin the libraries that decide backtest results to this machine's versions.

    The same fold scored 1.176 here (Python 3.13, pandas 3.0.2) and 1.116 on the
    stock Kaggle image (Python 3.12, pandas 2.x), so a walk-forward split across
    the two would stitch together numbers that do not reproduce each other.
    """
    pins = []
    for name in ("numpy", "pandas", "pyarrow"):
        try:
            pins.append(f"{name}=={__import__(name).__version__}")
        except Exception:                             # noqa: BLE001 - reporting only
            logger.warning("%s not importable here; not pinning it", name)
    return pins


def stage_code(task: str, args: List[str], heartbeat_url: Optional[str] = None,
               pins: Optional[List[str]] = None) -> Path:
    """Stage the code a research run needs, as one archive plus its job spec.

    Datasets travel as a single ``.tar.gz`` rather than a directory tree:
    Kaggle's own directory handling differs between upload modes, and one
    archive the Kaggle side unpacks itself is predictable either way.
    """
    stage = STAGE_DIR / "code"
    tree = STAGE_DIR / "_code_tree"
    for path in (stage, tree):
        if path.exists():
            shutil.rmtree(path)
    stage.mkdir(parents=True)
    tree.mkdir(parents=True)

    for rel in CODE_PATHS:
        src = _ROOT / rel
        if not src.exists():
            logger.warning("skipping missing %s", rel)
            continue
        if src.is_dir():
            shutil.copytree(src, tree / rel,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "runs", "store"))
        else:
            shutil.copy2(src, tree / rel)
    job = {"task": task, "args": args,
           "heartbeat_url": heartbeat_url or os.getenv("CENTURION_HEARTBEAT_URL", ""),
           "pip": pins or []}
    (tree / "job.json").write_text(json.dumps(job, indent=2))

    archive = shutil.make_archive(str(stage / "code"), "gztar", root_dir=tree)
    shutil.rmtree(tree)
    (stage / "job.json").write_text(json.dumps(job, indent=2))   # readable without unpacking
    logger.info("staged %s (%.1f MB, task %s)", archive, Path(archive).stat().st_size / 1e6, task)
    return stage


def push_code(task: str, args: List[str], heartbeat_url: Optional[str] = None,
              pins: Optional[List[str]] = None, suffix: str = "") -> str:
    """Upload code + job spec; returns the dataset slug (per suffix, so jobs don't collide)."""
    stage = stage_code(task, args, heartbeat_url, pins)
    slug = f"{CODE_SLUG}-{suffix}" if suffix else CODE_SLUG
    _push_dataset(stage, slug, f"Centurion NSE engine (research code{' ' + suffix if suffix else ''})", f"job: {task}")
    return slug


def push_store() -> None:
    """Upload the parquet store as one archive; run again when it is rebuilt."""
    store = _ROOT / "data" / "nse_engine" / "store"
    if not store.is_dir():
        raise SystemExit("data/nse_engine/store not found — run build-store first")
    stage = STAGE_DIR / "store"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    archive = shutil.make_archive(str(stage / "store"), "gztar",
                                  root_dir=store.parent, base_dir=store.name)
    logger.info("uploading %.0f MB (%s)", Path(archive).stat().st_size / 1e6, Path(archive).name)
    _push_dataset(stage, STORE_SLUG, "Centurion NSE parquet store",
                  time.strftime("store %Y-%m-%d"))


# ── kernel ───────────────────────────────────────────────────────

def kernel_ref(suffix: str = "") -> str:
    slug = f"{KERNEL_SLUG}-{suffix}" if suffix else KERNEL_SLUG
    return f"{kaggle_username()}/{slug}"


def push_kernel(enable_internet: bool = False, suffix: str = "", code_slug: str = CODE_SLUG) -> str:
    """Push the kernel that runs cloud/kaggle_entry.py against both datasets.

    ``suffix`` names a separate kernel (its own queue and output), so a second
    job can run while the first is still going.
    """
    user = kaggle_username()
    slug = f"{KERNEL_SLUG}-{suffix}" if suffix else KERNEL_SLUG
    stage = STAGE_DIR / f"kernel{'-' + suffix if suffix else ''}"
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_ROOT / "cloud" / "kaggle_entry.py", stage / f"{slug}.py")
    (stage / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{slug}",
        "title": f"Centurion NSE research{' ' + suffix if suffix else ''}",
        "code_file": f"{slug}.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": enable_internet,
        "dataset_sources": [f"{user}/{code_slug}", f"{user}/{STORE_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
    }, indent=2))
    _kaggle("kernels", "push", "-p", str(stage))
    return f"{user}/{slug}"


def status(kernel: Optional[str] = None) -> str:
    return _kaggle("kernels", "status", kernel or f"{kaggle_username()}/{KERNEL_SLUG}",
                   capture=True).strip()


def watch(kernel: Optional[str] = None, interval: int = 300, max_hours: float = 13.0) -> str:
    """Poll until the kernel stops, pinging the heartbeat URL each time."""
    from cloud.heartbeat import Heartbeat

    kernel = kernel or f"{kaggle_username()}/{KERNEL_SLUG}"
    hb = Heartbeat(name=f"kaggle:{kernel}", state_path=OUT_DIR / "watch_heartbeat.json")
    hb.start(f"watching {kernel}")
    deadline = time.time() + max_hours * 3600
    while time.time() < deadline:
        text = status(kernel)
        logger.info("%s", text)
        if any(word in text.lower() for word in ("complete", "error", "cancel")):
            (hb.done if "complete" in text.lower() else hb.fail)(text)
            return text
        hb.progress(text)
        time.sleep(interval)
    hb.fail(f"still running after {max_hours}h")
    return "timeout"


def pull(kernel: Optional[str] = None, dest: Optional[str] = None) -> Path:
    """Download kernel output into data/nse_engine/kaggle_out/<timestamp>."""
    kernel = kernel or f"{kaggle_username()}/{KERNEL_SLUG}"
    target = Path(dest) if dest else OUT_DIR / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    target.mkdir(parents=True, exist_ok=True)
    _kaggle("kernels", "output", kernel, "-p", str(target), "--page-size", "200")
    packed = target / "runs.tar.gz"
    if packed.exists() and not (target / "runs").is_dir():
        shutil.unpack_archive(str(packed), str(target))
        logger.info("unpacked %s", packed.name)
    latest = OUT_DIR / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(target.name)
    logger.info("output in %s (also linked as %s)", target, latest)
    return target


# ── CLI ──────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="report CLI, credentials and store state")
    sub.add_parser("push-store", help="upload data/nse_engine/store as a dataset")

    for name, help_text in (("stage", "build the staging dir only"),
                            ("push-code", "upload code + job spec"),
                            ("run", "push code, then the kernel")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--task", default="walk-forward", choices=["walk-forward", "grid"])
        p.add_argument("--args", default="", help="arguments for cloud.kaggle_runner, one string")
        p.add_argument("--heartbeat-url", default=None)
        p.add_argument("--pin", action="store_true",
                       help="install this machine's numpy/pandas/pyarrow in the kernel "
                            "(implies --internet on `run`)")
        if name == "run":
            p.add_argument("--internet", action="store_true",
                           help="kernel may reach the network (not needed with a store dataset)")
            p.add_argument("--kernel-suffix", default="",
                           help="run as a separate kernel (e.g. 'b') so it can run alongside the default one")

    p = sub.add_parser("watch", help="poll kernel status until it stops")
    p.add_argument("--kernel")
    p.add_argument("--interval", type=int, default=300)
    p.add_argument("--max-hours", type=float, default=13.0)

    p = sub.add_parser("status", help="print kernel status once")
    p.add_argument("--kernel")

    p = sub.add_parser("pull", help="download kernel output")
    p.add_argument("--kernel")
    p.add_argument("--dest")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")
    job_args = args.args.split() if getattr(args, "args", "") else []

    pins = local_pins() if getattr(args, "pin", False) else None

    if args.command == "check":
        print(json.dumps(check(), indent=2))
    elif args.command == "push-store":
        push_store()
    elif args.command == "stage":
        print(stage_code(args.task, job_args, args.heartbeat_url, pins))
    elif args.command == "push-code":
        push_code(args.task, job_args, args.heartbeat_url, pins)
    elif args.command == "run":
        code_slug = push_code(args.task, job_args, args.heartbeat_url, pins, args.kernel_suffix)
        print(f"kernel pushed: {push_kernel(args.internet or bool(pins), args.kernel_suffix, code_slug)}")
    elif args.command == "watch":
        print(watch(args.kernel, args.interval, args.max_hours))
    elif args.command == "status":
        print(status(args.kernel))
    elif args.command == "pull":
        print(pull(args.kernel, args.dest))


if __name__ == "__main__":
    main()
