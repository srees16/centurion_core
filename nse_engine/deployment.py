"""
Deployed NSE engine configuration (the one configuration paper/live trade).

``config/nse_engine_deployed.json`` pins the EngineConfig chosen after
validation and the holdout, together with its provenance::

    {
      "status": "approved",                 # or "placeholder"
      "paper_start_date": "2026-09-15",     # first paper decision date
      "source_run_id": "20260912-...",      # recorded run the config came from
      "approved_at": "2026-09-14T18:00:00+05:30",
      "notes": "...",
      "engine": { ...EngineConfig.to_dict()... }
    }

Rules enforced by :func:`load_deployment`:

* the file must exist and be a JSON object with exactly the keys above
  (``engine``, ``paper_start_date`` and ``status`` required);
* ``engine`` must only contain EngineConfig fields (checked recursively, so a
  typo such as ``portfolio.target_postions`` is an error rather than a
  silently ignored default);
* ``status`` is ``placeholder`` or ``approved``; an approved deployment needs
  ``source_run_id`` and ``approved_at``.

A ``placeholder`` deployment may be paper traded but never live traded
(:meth:`Deployment.live_allowed`).

The trading config is :meth:`Deployment.live_config`: the engine config with
``start`` set to ``paper_start_date``.  The executor loads market data from
``start.year - 2`` (as ``runners/run_nse_engine.py``), so live decisions and
the same-period backtest reference (``run_nse_engine shift-reference --start
<paper_start_date>``) share the same data anchor and rebalance-day count.

CLI (used by the GitHub Actions workflow)::

    python -m nse_engine.deployment show
    python -m nse_engine.deployment get paper_start_date
    python -m nse_engine.deployment get bootstrap_start      # paper_start - 3 years
    python -m nse_engine.deployment write-engine-config --out /tmp/engine.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import MISSING, dataclass, fields, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from nse_engine.config import EngineConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = "config/nse_engine_deployed.json"
ENV_PATH = "CENTURION_NSE_DEPLOYMENT"

STATUS_PLACEHOLDER = "placeholder"
STATUS_APPROVED = "approved"
STATUSES = (STATUS_PLACEHOLDER, STATUS_APPROVED)

REQUIRED_KEYS = ("engine", "paper_start_date", "status")
OPTIONAL_KEYS = ("source_run_id", "approved_at", "notes", "data_anchor_date")
ALLOWED_KEYS = frozenset(REQUIRED_KEYS + OPTIONAL_KEYS)

#: Years of archive history synced before paper_start_date on a fresh runner
#: (the executor loads from Jan 1 of paper_start.year - 2).
BOOTSTRAP_YEARS = 3


class DeploymentError(ValueError):
    """The deployment file is missing or invalid."""


@dataclass(frozen=True)
class Deployment:
    engine: EngineConfig
    paper_start_date: date
    status: str
    source_run_id: Optional[str] = None
    approved_at: Optional[str] = None
    notes: str = ""
    path: str = ""
    #: First row of market data loaded for signals.  Rebalance-day counting and the
    #: expanding forecast normalisers start at this row, so live trading must load
    #: from the same anchor the validation runs used (a 2024 vs 2011 anchor gave
    #: 5.9%/yr tracking error for the same config over 2026).
    data_anchor_date: Optional[date] = None

    @property
    def is_placeholder(self) -> bool:
        return self.status == STATUS_PLACEHOLDER

    def live_allowed(self) -> Tuple[bool, str]:
        """(allowed, reason) — placeholder deployments are paper only."""
        if self.is_placeholder:
            return False, (f"deployment {self.path or DEFAULT_PATH} is a placeholder "
                           "(status='placeholder'): paper trading only")
        return True, f"deployment approved at {self.approved_at} from run {self.source_run_id}"

    def live_config(self) -> EngineConfig:
        """Engine config used for trading: ``start`` = ``paper_start_date``."""
        return self.engine.replace(start=self.paper_start_date.isoformat())

    def data_start(self) -> date:
        """First market-data date to load (the validated anchor when pinned)."""
        if self.data_anchor_date is not None:
            return self.data_anchor_date
        return date(self.paper_start_date.year - 2, 1, 1)

    def bootstrap_start(self) -> date:
        """First archive date a fresh runner must sync."""
        if self.data_anchor_date is not None:
            return self.data_anchor_date
        d = self.paper_start_date
        try:
            return d.replace(year=d.year - BOOTSTRAP_YEARS)
        except ValueError:  # 29 Feb
            return d.replace(year=d.year - BOOTSTRAP_YEARS, day=28)

    def summary(self) -> Dict[str, Any]:
        return {"path": self.path, "status": self.status,
                "paper_start_date": self.paper_start_date.isoformat(),
                "source_run_id": self.source_run_id, "approved_at": self.approved_at,
                "config_hash": self.engine.config_hash(), "notes": self.notes,
                "data_start": self.data_start().isoformat()}


def resolve_path(path: Union[str, Path, None] = None) -> Path:
    """``path`` (or env CENTURION_NSE_DEPLOYMENT, or the default), relative to
    the working directory when it exists there, else to the repo root."""
    raw = Path(path or os.environ.get(ENV_PATH) or DEFAULT_PATH)
    if raw.is_absolute() or raw.exists():
        return raw
    return REPO_ROOT / raw


def _unknown_keys(cls, d: Dict[str, Any], prefix: str = "") -> List[str]:
    bad: List[str] = []
    known = {f.name: f for f in fields(cls)}
    for key, value in d.items():
        if key not in known:
            bad.append(prefix + key)
            continue
        f = known[key]
        default = f.default if f.default_factory is MISSING else f.default_factory()  # type: ignore[misc]
        if is_dataclass(default):
            if not isinstance(value, dict):
                bad.append(f"{prefix}{key} (expected an object)")
            else:
                bad.extend(_unknown_keys(type(default), value, f"{prefix}{key}."))
    return bad


def _parse_date(value: Any, key: str) -> date:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        raise DeploymentError(f"{key} must be an ISO date (YYYY-MM-DD), got {value!r}") from None


def parse_deployment(raw: Any, path: str = "") -> Deployment:
    """Validate a decoded deployment document."""
    where = f" in {path}" if path else ""
    if not isinstance(raw, dict):
        raise DeploymentError(f"deployment{where} must be a JSON object")
    unknown = sorted(set(raw) - ALLOWED_KEYS)
    if unknown:
        raise DeploymentError(f"unknown deployment key(s){where}: {unknown}; allowed: {sorted(ALLOWED_KEYS)}")
    missing = [k for k in REQUIRED_KEYS if k not in raw]
    if missing:
        raise DeploymentError(f"missing deployment key(s){where}: {missing}")
    status = raw["status"]
    if status not in STATUSES:
        raise DeploymentError(f"status{where} must be one of {list(STATUSES)}, got {status!r}")
    engine_raw = raw["engine"]
    if not isinstance(engine_raw, dict):
        raise DeploymentError(f"engine{where} must be an EngineConfig object")
    bad = _unknown_keys(EngineConfig, engine_raw)
    if bad:
        raise DeploymentError(f"unknown EngineConfig key(s){where}: {bad}")
    try:
        engine = EngineConfig.from_dict(engine_raw)
    except (TypeError, ValueError) as exc:
        raise DeploymentError(f"invalid engine config{where}: {exc}") from exc
    paper_start = _parse_date(raw["paper_start_date"], "paper_start_date")
    run_id = raw.get("source_run_id") or None
    approved_at = raw.get("approved_at") or None
    if approved_at is not None:
        try:
            datetime.fromisoformat(str(approved_at))
        except ValueError:
            raise DeploymentError(f"approved_at{where} must be an ISO timestamp, got {approved_at!r}") from None
    if status == STATUS_APPROVED and not (run_id and approved_at):
        raise DeploymentError(f"an approved deployment{where} needs source_run_id and approved_at")
    anchor = _parse_date(raw["data_anchor_date"], "data_anchor_date") if raw.get("data_anchor_date") else None
    if anchor is not None and anchor >= paper_start:
        raise DeploymentError(f"data_anchor_date{where} must be before paper_start_date")
    notes = raw.get("notes") or ""
    if not isinstance(notes, str):
        raise DeploymentError(f"notes{where} must be a string")
    return Deployment(engine=engine, paper_start_date=paper_start, status=status,
                      source_run_id=run_id, approved_at=approved_at, notes=notes, path=path,
                      data_anchor_date=anchor)


def load_deployment(path: Union[str, Path, None] = DEFAULT_PATH) -> Deployment:
    """Load and validate the deployed engine configuration.

    Raises :class:`DeploymentError` when the file is missing, is not valid
    JSON, or fails validation.
    """
    p = resolve_path(path)
    if not p.exists():
        raise DeploymentError(
            f"NSE engine deployment file not found: {p}. Create it from a validated run "
            f"(see nse_engine/deployment.py) or set {ENV_PATH}.")
    try:
        raw = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise DeploymentError(f"deployment file {p} is not valid JSON: {exc}") from exc
    return parse_deployment(raw, path=str(p))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect the deployed NSE engine configuration")
    parser.add_argument("--path", default=None, help=f"deployment file (default: ${ENV_PATH} or {DEFAULT_PATH})")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("show", help="print a validated summary")
    g = sub.add_parser("get", help="print one field")
    g.add_argument("field", choices=["paper_start_date", "bootstrap_start", "data_start", "status", "source_run_id",
                                     "config_hash", "store_dir"])
    w = sub.add_parser("write-engine-config", help="write the trading EngineConfig as plain JSON "
                                                   "(accepted by runners/run_nse_engine.py --config)")
    w.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        dep = load_deployment(args.path)
    except DeploymentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.command == "show":
        print(json.dumps(dep.summary(), indent=2))
    elif args.command == "get":
        value = {"paper_start_date": dep.paper_start_date.isoformat(),
                 "bootstrap_start": dep.bootstrap_start().isoformat(),
                 "data_start": dep.data_start().isoformat(),
                 "status": dep.status, "source_run_id": dep.source_run_id or "",
                 "config_hash": dep.engine.config_hash(),
                 "store_dir": dep.engine.data.store_dir}[args.field]
        print(value)
    else:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(dep.live_config().to_json())
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
