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
import sys
from pathlib import Path

INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")


def find_code_dir() -> Path:
    """The attached dataset holding this repo: the one with cloud/kaggle_runner.py."""
    for base in sorted(INPUT.glob("*")):
        for candidate in (base, base / "centurion_core"):
            if (candidate / "cloud" / "kaggle_runner.py").exists():
                return candidate
    raise SystemExit(f"no code dataset under {INPUT} (need cloud/kaggle_runner.py)")


def main() -> None:
    src = find_code_dir()
    dest = WORKING / "code"
    if not dest.exists():
        shutil.copytree(src, dest)
    os.chdir(dest)
    sys.path.insert(0, str(dest))

    job_path = dest / "job.json"
    if not job_path.exists():
        raise SystemExit("job.json missing from the code dataset")
    job = json.loads(job_path.read_text())
    print(f"job: {job.get('task')} {' '.join(job.get('args', []))}", flush=True)

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
