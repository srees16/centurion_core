#!/usr/bin/env python3
"""A study's files, declared at its start and listed for deletion at its end (tracker PL0, PL-07).

A failed idea leaves no code or data (the owner's rule), yet files have come back or been left: four empty
modules of failed NS1 studies returned twice, and a study's downloads live under ``data/nse_engine/<task>/``
until someone remembers them.  ``start`` records where a study will write; ``close`` lists every file found
there, with sizes, for the owner to approve.  Nothing here deletes anything.

    python -m tools.study start --id GF1 data/nse_engine/gf1
    python -m tools.study close --id GF1        # the list to approve
    python -m tools.study list
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

#: Git-ignored (data/nse_engine/), so a manifest never reaches the public repository.
MANIFESTS = "data/nse_engine/studies"
IST = timezone(timedelta(hours=5, minutes=30))


def start(study: str, paths: Sequence[str], root: Path = _ROOT) -> Path:
    """Record where ``study`` will write.  Paths are inside the repository; adding to a study keeps the old ones."""
    clean = []
    for p in paths:
        full = (Path(root) / p).resolve()
        if Path(root).resolve() not in full.parents:
            raise ValueError(f"{p} is outside the repository")
        clean.append(full.relative_to(Path(root).resolve()).as_posix())
    target = Path(root) / MANIFESTS / f"{study}.json"
    doc = json.loads(target.read_text()) if target.exists() else {"study": study, "started": datetime.now(IST).isoformat(), "paths": []}
    doc["paths"] = sorted(set(doc["paths"]) | set(clean))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(doc, indent=2) + "\n")
    return target


def files(study: str, root: Path = _ROOT) -> List[Dict[str, Any]]:
    """Every file now under the study's declared paths: {path, bytes}."""
    doc = json.loads((Path(root) / MANIFESTS / f"{study}.json").read_text())
    out = []
    for p in doc["paths"]:
        full = Path(root) / p
        found = [full] if full.is_file() else sorted(f for f in full.rglob("*") if f.is_file()) if full.is_dir() else []
        out += [{"path": f.relative_to(root).as_posix(), "bytes": f.stat().st_size} for f in found]
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start", help="record where a study will write")
    s.add_argument("--id", required=True)
    s.add_argument("paths", nargs="+")
    c = sub.add_parser("close", help="list a study's files for the owner to approve deleting")
    c.add_argument("--id", required=True)
    sub.add_parser("list", help="the studies with a manifest")
    args = p.parse_args(argv)
    if args.cmd == "start":
        print(f"recorded in {start(args.id, args.paths).relative_to(_ROOT)}")
        return 0
    if args.cmd == "list":
        for m in sorted((_ROOT / MANIFESTS).glob("*.json")):
            print(m.stem, len(files(m.stem)), "file(s)")
        return 0
    found = files(args.id)
    for f in found:
        print(f"{f['bytes']:>12,}  {f['path']}")
    print(f"{len(found)} file(s), {sum(f['bytes'] for f in found) / 1e6:.1f} MB, to delete once the owner approves "
          f"(this tool deletes nothing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
