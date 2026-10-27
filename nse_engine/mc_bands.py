"""
Monte Carlo bands for the paper books (tracker MC1): where a book's return and drawdown since its start sit among
the paths its own honest history allows, over the same number of sessions.

Each book's reference is its live-like walk-forward record (the honest base, never the in-sample backtest):

* deployed, candidate: the deployed family's walk-forward book (R4b, 2008-25), idle cash at the liquid rate and
  D2's haircut, as the diversifier gate's books;
* e4: E4's family walk-forward (R12b4), the same way;
* futures: the L7-A run (the deployed rules levered 1.75x with stock futures, walk-forward folds, 2017-25);
* bonds, mcx: 0.75 x the walk-forward book + 0.25 x its sleeve (gilts; MCX-B), as DG2 measured them;
* a1: the walk-forward book plus A1G at A1C's 10% cap (plan 5aj), as its gate measured it.

``SIMS`` paths per horizon from a stationary block bootstrap of the reference's real days (mean block
``BLOCK`` sessions, so volatility clusters and runs survive; ``SEED`` fixed).  Only the percentiles are kept, in
``config/mc_bands.json``: the Saturday job has no research data and reads that file.  Rebuild it on the research
machine after a re-baseline: ``python -m nse_engine.mc_bands build``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SIMS, BLOCK, SEED = 5000, 20, 7
HORIZONS = (5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 125, 150, 200, 252, 378, 504, 756)
PCTS = (1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 99)
BANDS_PATH = "config/mc_bands.json"
E4_WALK_FORWARD = "data/nse_engine/wf_stitched_r12b4.json"
L7A_RUN = "data/nse_engine/runs_leverage/20261017T111547459708Z_33297924/returns.csv"
MCX_JSON = "data/nse_engine/dg1/mcx_results_2007_25.json"
A1G_RUN = "data/nse_engine/runs_options/20261026T173041043859Z_358e73d3"


# ── the references (research machine) ────────────────────────────────────

def reference_returns() -> Dict[str, pd.Series]:
    """Each book's live-like walk-forward daily returns (see the module docstring)."""
    from nse_engine import bond_stream as bs
    from nse_engine import diversifier_gate as dg

    def live(base: "bs.Base") -> pd.Series:
        liquid = bs.bond_returns(base.returns.index, bs.BondConfig())["liquid"]
        return dg._live_book(base, liquid, base.returns.index)

    wf = bs.walk_forward_base()
    book = live(wf)
    idx = book.index
    br = bs.bond_returns(idx, bs.BondConfig())
    mcx = json.loads(Path(MCX_JSON).read_text())["reported"]["GT5-B"]["returns"]
    mcx = pd.Series(mcx, dtype="float64")
    mcx.index = pd.DatetimeIndex(mcx.index)
    mcx = dg.carry_to(mcx.sort_index(), idx).reindex(idx).fillna(0.0) + br["liquid"]
    a1 = dg._read(Path(A1G_RUN) / "returns.csv")
    sessions = idx[idx >= a1.index[0]]
    a1_eq = (1 + a1).cumprod() * dg.A1_CAPITAL
    w, _ = dg.capped_weights(sessions, 0.10, pd.read_csv(Path(A1G_RUN) / "cycles.csv"), a1_eq)
    a1_book = dg.combine(book[sessions], dg.carry_to(a1, sessions).reindex(sessions).fillna(0.0), w, "overlay")
    l7a = pd.read_csv(L7A_RUN, index_col=0).squeeze("columns").astype("float64")
    l7a.index = pd.DatetimeIndex(l7a.index)
    return {"deployed": book, "candidate": book, "e4": live(bs.walk_forward_base(E4_WALK_FORWARD)),
            "futures": l7a.sort_index(), "bonds": dg.combine(book, br["gilt"], 0.25, "sleeve"),
            "mcx": dg.combine(book, mcx, 0.25, "sleeve"), "a1": a1_book}


def bands(returns: pd.Series, horizons: Sequence[int] = HORIZONS, sims: int = SIMS, block: int = BLOCK,
          seed: int = SEED) -> Dict[str, Dict[str, list]]:
    """Percentiles (``PCTS``) of the cumulative return and of the maximum drawdown over each horizon, from
    ``sims`` stationary block-bootstrap paths of ``returns``' real days."""
    x = pd.Series(returns, dtype="float64").dropna().to_numpy()
    n, rng = len(x), np.random.default_rng(seed)
    out: Dict[str, Dict[str, list]] = {"return": {}, "drawdown": {}}
    for h in horizons:
        paths = np.empty((sims, h))
        for s in range(sims):
            k = 0
            while k < h:
                start, length = rng.integers(n), min(int(rng.geometric(1.0 / block)), h - k)
                paths[s, k:k + length] = x[(start + np.arange(length)) % n]
                k += length
        eq = np.cumprod(1.0 + paths, axis=1)
        dd = (eq / np.maximum.accumulate(np.maximum(eq, 1.0), axis=1) - 1.0).min(axis=1)    # the start counts as a peak
        out["return"][str(h)] = [round(float(v), 5) for v in np.percentile(eq[:, -1] - 1.0, PCTS)]
        out["drawdown"][str(h)] = [round(float(v), 5) for v in np.percentile(dd, PCTS)]
    return out


def build(path: str = BANDS_PATH) -> Dict[str, Any]:
    """Every book's bands, written to ``path`` with how they were made."""
    refs = reference_returns()
    doc = {"created_at": datetime.now(timezone.utc).isoformat(), "sims": SIMS, "block": BLOCK, "seed": SEED,
           "percentiles": list(PCTS), "horizons": list(HORIZONS), "books": {}}
    for name, r in refs.items():
        r = r.dropna()
        doc["books"][name] = {"reference": f"{r.index[0].date()} to {r.index[-1].date()}, {len(r)} sessions",
                              **bands(r)}
        logger.info("MC bands: %s from %s", name, doc["books"][name]["reference"])
    Path(path).write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    return doc


# ── reading them (the Saturday email) ────────────────────────────────────

def load(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    from nse_engine.deployment import REPO_ROOT

    p = Path(path) if path else Path(REPO_ROOT) / BANDS_PATH
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def percentile(doc: Dict[str, Any], book: str, kind: str, sessions: int, value: float) -> Optional[float]:
    """Where ``value`` (a return, or a drawdown as a negative number) over ``sessions`` sits among the book's
    paths, as a percentile (0-100; 50 is typical; low is a worse return or a deeper drawdown).  The grid is
    interpolated between the horizons around ``sessions``; None outside them or without bands for the book."""
    b = (doc.get("books") or {}).get(book)
    if not b or not np.isfinite(value):
        return None
    hs = sorted(int(h) for h in b[kind])
    if sessions < hs[0] or sessions > hs[-1]:
        return None
    lo = max(h for h in hs if h <= sessions)
    hi = min(h for h in hs if h >= sessions)
    a, c = np.asarray(b[kind][str(lo)]), np.asarray(b[kind][str(hi)])
    grid = a if hi == lo else a + (c - a) * (sessions - lo) / (hi - lo)
    pcts = np.asarray(doc["percentiles"], dtype=float)
    return float(np.interp(value, grid, pcts, left=0.0, right=100.0))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nse_engine.mc_bands", description=__doc__.strip().splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help=f"rebuild {BANDS_PATH} from the research data (research machine)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.cmd == "build":
        doc = build()
        print(json.dumps({k: v["reference"] for k, v in doc["books"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
