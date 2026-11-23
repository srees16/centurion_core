"""Phase 0's spanning screen (research protocol, section 2; tracker HSR2): does a new stock signal widen the frontier
that beta, size, value and the engine's combined forecast already span?

    python -m runners.run_spanning --candidate data/nse_engine/<study>/candidate.parquet --name <signal>

The candidate file (parquet or csv) holds one row per (date, symbol): the signal's value known at that month's close,
on the panel's month-end dates (the last session of each month), every month from 2012 to 2025 covered.  The panel is
FF1's (``runners.run_nse_engine._slope_panel``): the engine's liquid universe with fundamentals, next month's return.
2026 is the sealed holdout and is refused.  Prints the result of ``nse_engine.validation.max_sharpe.spanning_increment``
as JSON; GO or STOP goes in the study's Phase 0 block.  About eight minutes with the defaults on this Mac.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Sequence

import pandas as pd

BASE = ("beta", "ln_me", "ep_pos", "ep_neg", "forecast")
#: The first day of the sealed holdout (research protocol): the screen never reads it.
HOLDOUT_START = pd.Timestamp("2026-01-01")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m runners.run_spanning", description=__doc__.split("\n\n")[0])
    p.add_argument("--candidate", required=True, help="parquet or csv: date, symbol, value")
    p.add_argument("--name", required=True, help="the candidate characteristic's name")
    p.add_argument("--start", default="2008-01-01", help="first formation month (fundamentals begin in 2012)")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--data-start", default="2006-01-02")
    p.add_argument("--store", default="data/nse_engine/store_ext2006")
    p.add_argument("--random-features", type=int, default=2000)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--permutations", type=int, default=99)
    p.add_argument("--min-names", type=int, default=100)
    args = p.parse_args(argv)
    if pd.Timestamp(args.end) >= HOLDOUT_START:
        print(f"--end {args.end} reaches the sealed 2026 holdout: Phase 0 reads 2025 at the latest", file=sys.stderr)
        return 2
    path = Path(args.candidate)
    if not path.is_file():
        print(f"no candidate file at {path}", file=sys.stderr)
        return 2
    values = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    from nse_engine.validation.max_sharpe import attach_characteristic, spanning_increment
    from runners.run_nse_engine import _slope_panel

    panel = _slope_panel(SimpleNamespace(start=args.start, end=args.end, data_start=args.data_start,
                                         store=args.store))
    periods = {day: (returns / 100.0, table) for day, (returns, table) in panel.items()}   # FF1's returns are percent
    result = spanning_increment(attach_characteristic(periods, values, args.name), BASE, args.name,
                                random_features=args.random_features, seeds=args.seeds,
                                permutations=args.permutations, min_names=args.min_names)
    result["months"] = len(periods)
    result["window"] = [str(min(periods).date()), str(max(periods).date())]
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
