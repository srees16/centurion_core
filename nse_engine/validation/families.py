"""Research families: their budgets in one file, their trial counts from the registries (tracker PL0, PL-03).

``docs/research_protocol.md`` gives every trial a family with its own registry and budget.  The counts in
that table were typed by hand and went stale (14 Nov 2026: options "N = 17" against 20 recorded, intraday
"N = 0" against 2).  Now:

- ``config/research_families.json`` holds each family's registry, budget and last recorded count ``n``;
- ``counts`` reads ``n`` from the registries, which live only on the research machine: the distinct
  configurations recorded (a re-run of one configuration is one trial), and for the core the columns of the
  deployed book's same-window returns matrix, the N its PBO and deflated Sharpe use;
- ``python -m nse_engine.validation.families --write`` rewrites the file's counts after a trial is recorded;
- a test fails when the protocol's table states another N than the file, and (where the registries exist)
  when the file states another N than the registries.

    python -m nse_engine.validation.families            # the table, and what disagrees
    python -m nse_engine.validation.families --write    # record the registries' counts in the file
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from nse_engine.deployment import REPO_ROOT

FAMILIES_PATH = "config/research_families.json"
PROTOCOL_PATH = "docs/research_protocol.md"
#: How a family's N is read from its registry.
DISTINCT_CONFIGS, VALIDATION_MATRIX = "distinct_configs", "validation_matrix"
IST = timezone(timedelta(hours=5, minutes=30))


def load(root: Path = REPO_ROOT) -> Dict[str, Dict[str, Any]]:
    return json.loads((Path(root) / FAMILIES_PATH).read_text())["families"]


def registry_count(family: Dict[str, Any], root: Path = REPO_ROOT) -> Optional[int]:
    """The family's N from its registry, or None when it has none on this machine."""
    if not family.get("registry") or not (Path(root) / family["registry"]).is_dir():
        return None
    from nse_engine.validation.trials import LEGACY_COST_MODEL, TrialRegistry

    registry = TrialRegistry(Path(root) / family["registry"])
    if family.get("count") == VALIDATION_MATRIX:
        from nse_engine import books
        from nse_engine.deployment import load_deployment

        run = books.latest_run(Path(root) / family["registry"], load_deployment().engine.config_hash())
        if run is None:
            return None
        matrix = registry.returns_matrix(data_hash=run.get("data_hash"), window=(run["start"], run["end"]),
                                         cost_model=int(run.get("cost_model") or LEGACY_COST_MODEL))
        return int(matrix.shape[1])
    trials = registry.list_trials()
    return int(trials["config_hash"].nunique()) if len(trials) else 0


def counts(root: Path = REPO_ROOT) -> Dict[str, Optional[int]]:
    return {name: registry_count(family, root) for name, family in load(root).items()}


def protocol_counts(root: Path = REPO_ROOT) -> Dict[str, Optional[int]]:
    """The N the protocol's table states for each family (its row is found by ``protocol_row``)."""
    rows = [l for l in (Path(root) / PROTOCOL_PATH).read_text().splitlines() if l.startswith("| ")]
    out: Dict[str, Optional[int]] = {}
    for name, family in load(root).items():
        if not family.get("protocol_row"):
            continue
        row = next((r for r in rows if r.startswith(f"| {family['protocol_row']}")), None)
        m = re.search(r"\bN = (\d+)", row or "")
        out[name] = int(m.group(1)) if m else None
    return out


def problems(root: Path = REPO_ROOT, registries: bool = True) -> List[str]:
    """What disagrees: the protocol's table against the file, the file against the registries (where they
    exist and ``registries``), and a family past its budget."""
    out: List[str] = []
    families = load(root)
    for name, stated in protocol_counts(root).items():
        if stated != families[name]["n"]:
            out.append(f"{name}: the protocol says N = {stated}, {FAMILIES_PATH} says {families[name]['n']}")
    for name, family in families.items():
        cap = family.get("max_configurations")
        if cap is not None and family["n"] > cap:
            out.append(f"{name}: N = {family['n']} is past its budget of {cap}")
        if registries:
            found = registry_count(family, root)
            if found is not None and found != family["n"]:
                out.append(f"{name}: its registry holds {found} configuration(s), {FAMILIES_PATH} says "
                           f"{family['n']} (run `python -m nse_engine.validation.families --write`)")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--write", action="store_true", help="record the registries' counts in the file")
    args = p.parse_args(argv)
    path = REPO_ROOT / FAMILIES_PATH
    doc = json.loads(path.read_text())
    if args.write:
        changed = []
        for name, found in counts().items():
            if found is not None and found != doc["families"][name]["n"]:
                changed.append(f"{name}: {doc['families'][name]['n']} -> {found}")
                doc["families"][name]["n"] = found
                doc["families"][name]["as_of"] = datetime.now(IST).date().isoformat()
        path.write_text(json.dumps(doc, indent=2) + "\n")
        print("\n".join(changed) or "no count changed")
    for name, family in doc["families"].items():
        cap = family.get("max_configurations")
        print(f"{name:14} N = {family['n']:<4} budget {cap if cap is not None else 'PBO cap':<8} {family.get('status', '')}")
    found = problems()
    print("\n".join(found) or "the protocol, the file and the registries agree")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
