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
KERNEL_SLUG = "centurion-nse-research"
STAGE_DIR = _ROOT / "data" / "kaggle" / "stage"
OUT_DIR = _ROOT / "data" / "nse_engine" / "kaggle_out"

# Only what a research run needs; the store travels as its own dataset.
CODE_PATHS = ["nse_engine", "runners", "cloud", "config", "requirements.txt"]


# ── kaggle CLI ───────────────────────────────────────────────────

def kaggle_username() -> str:
    user = os.getenv("KAGGLE_USERNAME")
    if user:
        return user
    token = Path.home() / ".kaggle" / "kaggle.json"
    if token.exists():
        try:
            return json.loads(token.read_text())["username"]
        except (ValueError, KeyError):
            pass
    raise SystemExit("no Kaggle username: set KAGGLE_USERNAME or install ~/.kaggle/kaggle.json")


def check() -> Dict[str, object]:
    """Report whether the CLI and credentials are usable, without failing."""
    cli = shutil.which("kaggle")
    token = (Path.home() / ".kaggle" / "kaggle.json").exists()
    env = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    state = {"kaggle_cli": cli or "missing", "token_file": token, "env_credentials": env,
             "store_present": (_ROOT / "data/nse_engine/store").is_dir()}
    if not cli:
        state["install"] = "pip install kaggle"
    if not (token or env):
        state["credentials"] = "Kaggle → Account → Create New API Token → ~/.kaggle/kaggle.json"
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
    if f"{kaggle_username()}/{slug}" in existing:
        _kaggle("datasets", "version", "-p", str(path), "-m", message, "-d", "-r", "zip")
    else:
        _kaggle("datasets", "create", "-p", str(path), "-d", "-r", "zip")


def stage_code(task: str, args: List[str], heartbeat_url: Optional[str] = None) -> Path:
    """Copy the code a research run needs, plus its job spec, into a staging dir."""
    stage = STAGE_DIR / "code"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for rel in CODE_PATHS:
        src = _ROOT / rel
        if not src.exists():
            logger.warning("skipping missing %s", rel)
            continue
        if src.is_dir():
            shutil.copytree(src, stage / rel,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "runs", "store"))
        else:
            shutil.copy2(src, stage / rel)
    job = {"task": task, "args": args, "heartbeat_url": heartbeat_url or os.getenv("CENTURION_HEARTBEAT_URL", "")}
    (stage / "job.json").write_text(json.dumps(job, indent=2))
    logger.info("staged %s (%s)", stage, task)
    return stage


def push_code(task: str, args: List[str], heartbeat_url: Optional[str] = None) -> None:
    stage = stage_code(task, args, heartbeat_url)
    _push_dataset(stage, CODE_SLUG, "Centurion NSE engine (research code)", f"job: {task}")


def push_store() -> None:
    """Upload the parquet store; run again whenever the store is rebuilt."""
    store = _ROOT / "data" / "nse_engine" / "store"
    if not store.is_dir():
        raise SystemExit("data/nse_engine/store not found — run build-store first")
    stage = STAGE_DIR / "store"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    shutil.copytree(store, stage / "store")
    size_mb = sum(f.stat().st_size for f in stage.rglob("*") if f.is_file()) / 1e6
    logger.info("uploading %.0f MB of parquet", size_mb)
    _push_dataset(stage, STORE_SLUG, "Centurion NSE parquet store",
                  time.strftime("store %Y-%m-%d"))


# ── kernel ───────────────────────────────────────────────────────

def push_kernel(enable_internet: bool = False) -> str:
    """Push the kernel that runs cloud/kaggle_entry.py against both datasets."""
    user = kaggle_username()
    stage = STAGE_DIR / "kernel"
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_ROOT / "cloud" / "kaggle_entry.py", stage / "centurion-nse-research.py")
    (stage / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{KERNEL_SLUG}",
        "title": "Centurion NSE research",
        "code_file": "centurion-nse-research.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": enable_internet,
        "dataset_sources": [f"{user}/{CODE_SLUG}", f"{user}/{STORE_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
    }, indent=2))
    _kaggle("kernels", "push", "-p", str(stage))
    return f"{user}/{KERNEL_SLUG}"


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
    _kaggle("kernels", "output", kernel, "-p", str(target))
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
        if name == "run":
            p.add_argument("--internet", action="store_true",
                           help="kernel may reach the network (not needed with a store dataset)")

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

    if args.command == "check":
        print(json.dumps(check(), indent=2))
    elif args.command == "push-store":
        push_store()
    elif args.command == "stage":
        print(stage_code(args.task, job_args, args.heartbeat_url))
    elif args.command == "push-code":
        push_code(args.task, job_args, args.heartbeat_url)
    elif args.command == "run":
        push_code(args.task, job_args, args.heartbeat_url)
        print(f"kernel pushed: {push_kernel(args.internet)}")
    elif args.command == "watch":
        print(watch(args.kernel, args.interval, args.max_hours))
    elif args.command == "status":
        print(status(args.kernel))
    elif args.command == "pull":
        print(pull(args.kernel, args.dest))


if __name__ == "__main__":
    main()
