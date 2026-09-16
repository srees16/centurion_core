"""The script a Kaggle kernel runs. It reads its job from the code dataset.

Kaggle kernels take no command-line arguments, so ``cloud.kaggle_local`` writes
``job.json`` beside the code it uploads:

    {"task": "walk-forward", "args": ["--grid", "{...}", "--folds", "0-3"]}

Datasets are mounted read-only, so the code is copied to /kaggle/working first
(the engine records runs next to it).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")


def prepare_code() -> Path:
    """Put the repo in a writable /kaggle/working/code, from an archive or a tree.

    ``cloud.kaggle_local`` ships one ``code.tar.gz``; a dataset that was built
    by hand may instead hold the tree directly, so both are accepted.
    """
    dest = WORKING / "code"
    if (dest / "cloud" / "kaggle_runner.py").exists():
        return dest

    # Two things vary: Kaggle unpacks an uploaded archive into a folder named
    # after it, and images differ on where datasets mount (/kaggle/input/<slug>
    # on some, /kaggle/input/datasets/<owner>/<slug> on others). So search for
    # the file instead of assuming a path. (This runs before the repo exists on
    # disk, so the search is written out here rather than imported.)
    for hit in sorted(INPUT.rglob("kaggle_runner.py")):
        if hit.parent.name == "cloud":
            src = hit.parent.parent
            print(f"copying code from {src}", flush=True)
            shutil.copytree(src, dest)
            return dest

    for archive in sorted(INPUT.rglob("code.tar.gz")):
        print(f"unpacking {archive}", flush=True)
        dest.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(str(archive), str(dest))
        return dest

    listing = []
    for base in sorted(INPUT.glob("*")) if INPUT.is_dir() else []:
        entries = sorted(p.name for p in base.glob("*"))[:10]
        listing.append(f"  {base}: {entries or 'empty'}")
    raise SystemExit(f"no code dataset under {INPUT} (need code.tar.gz or cloud/kaggle_runner.py)\n"
                     + ("\n".join(listing) or f"  {INPUT} has no datasets attached"))


def _print_versions() -> None:
    import platform

    versions = [f"python {platform.python_version()}"]
    for name in ("numpy", "pandas", "pyarrow"):
        try:
            versions.append(f"{name} {__import__(name).__version__}")
        except Exception:                             # noqa: BLE001 - reporting only
            versions.append(f"{name} missing")
    print("versions: " + " | ".join(versions), flush=True)


def main() -> None:
    dest = prepare_code()
    os.chdir(dest)
    sys.path.insert(0, str(dest))

    job_path = dest / "job.json"
    if not job_path.exists():
        raise SystemExit("job.json missing from the code dataset")
    job = json.loads(job_path.read_text())
    print(f"job: {job.get('task')} {' '.join(job.get('args', []))}", flush=True)

    # Results depend on the library versions: a fold run under pandas 2 and the
    # same fold under pandas 3 differ by more than float noise. `pip` in the job
    # pins them to whatever produced the rest of the walk-forward (needs the
    # kernel's internet switch on).
    if job.get("pip"):
        print(f"installing {job['pip']}", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *job["pip"]], check=True)
    _print_versions()

    if job.get("heartbeat_url"):
        os.environ.setdefault("CENTURION_HEARTBEAT_URL", job["heartbeat_url"])

    from cloud.kaggle_runner import main as run

    run([job["task"], *job.get("args", [])])

    # Kaggle keeps /kaggle/working as the kernel's output; make sure the run
    # registry and fold files are there rather than only inside the code copy.
    for name in ("runs", "wf"):
        produced = dest / "data" / "nse_engine" / name
        if produced.exists() and not (WORKING / name).exists():
            shutil.move(str(produced), str(WORKING / name))


if __name__ == "__main__":
    main()
