"""
Does the book's stock selection beat random picks with the same timing?  (tracker ALG1)

The third significance test of Chan, *Algorithmic Trading* (2013), ch. 1, pp. 17-21 and Example 1.1, after Lo,
Mamaysky and Wang (2000): keep the number of trades and their holding periods, randomise the rest, and ask how
often chance does as well.  For a time-series rule Chan permutes the entry dates; ``validation.timing`` does the
same for the regime gate (every rotation of itself).  For a cross-sectional book the analogue keeps every holding's
dates and weights and draws its stock at random:

* **A holding** is a run of sessions on which a core stock's recorded weight is above ``WEIGHT_TOL``.  Sleeve
  symbols (gold, silver and any extra sleeve) are never redrawn: they are allocation, not selection.
* **A draw** replaces each holding's stock with one picked uniformly from the universe on the session before the
  holding starts (the decision close), among names no other holding of the same draw occupies on any of its
  sessions.  The holding keeps the recorded weight path, so timing, sizing and exposure are the book's own.
* **The statistic** is the core's mean daily gross return, sum of w(t-1) r(t) over core names, with ``r`` the
  dividend- and split-adjusted close-to-close return (a missing close counts 0, the book and every draw alike).
  Costs are left out on both sides: random names trade the same sessions and sizes.
* **p** = (1 + draws at or above the book) / (1 + draws).  Small p: the names the book picked earned more than
  names it could have picked with the same timing.  A large p says the edge, if any, is in timing and exposure.

What it does not test: exits.  A holding the book's stop ended keeps that end date for its random stock, which
gets no stop of its own.  Chan's second test (prices simulated with the
same four moments) is not built: the cross-sectional momentum it would question is already tested by Fama-MacBeth
(``validation.fama_macbeth``).

    python -m nse_engine.validation.selection_skill --run <run_id> [--draws 1000] [--seed 7] [--start ..] [--end ..]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WEIGHT_TOL = 1e-9
MIN_HOLDINGS = 30          # fewer holdings than this give no usable distribution
MIN_DRAWS = 100            # p is resolved to 1 / (draws + 1)
DEFAULT_DRAWS = 1000
DEFAULT_SEED = 7
QUICK_TRIES = 20           # random picks tried before the exhaustive search for a free name
TIE_TOL = 1e-12            # a draw this close to the book's mean daily return ties it (summation order)


@dataclass(frozen=True)
class Holding:
    column: int            # the book's stock (column of the weights panel)
    start: int             # first session with weight (row)
    end: int               # last session with weight (row, inclusive)


def holdings(weights: np.ndarray, columns: Iterable[int], tol: float = WEIGHT_TOL) -> List[Holding]:
    """Every run of sessions on which a column's weight is above ``tol``, ordered by start."""
    out: List[Holding] = []
    for c in columns:
        held = np.concatenate(([0], (weights[:, c] > tol).astype(np.int8), [0]))
        edges = np.diff(held)
        out.extend(Holding(int(c), int(s), int(e) - 1)
                   for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    return sorted(out, key=lambda h: (h.start, h.column))


def core_returns(weights: np.ndarray, returns: np.ndarray, columns: Sequence[int]) -> np.ndarray:
    """Daily gross return of ``columns``: sum of w(t-1) r(t), length T - 1."""
    cols = np.asarray(columns, dtype=int)
    return (weights[:-1, cols] * returns[1:, cols]).sum(axis=1)


def _draw(rng: np.random.Generator, weights: np.ndarray, returns: np.ndarray, eligible: np.ndarray,
          plan: Sequence[Holding]) -> Tuple[np.ndarray, List[int]]:
    """One draw: the daily core return with each holding's weight path on a random free eligible stock, and the
    stock picked for each holding (in ``plan`` order)."""
    n_rows, n_cols = weights.shape
    occupied = np.zeros((n_rows, n_cols), dtype=bool)
    daily = np.zeros(n_rows - 1)
    picks: List[int] = []
    for h in plan:
        decision = max(h.start - 1, 0)
        candidates = np.flatnonzero(eligible[decision])
        pick = -1
        for _ in range(QUICK_TRIES):
            j = int(candidates[rng.integers(candidates.size)])
            if not occupied[h.start:h.end + 1, j].any():
                pick = j
                break
        if pick < 0:
            free = candidates[~occupied[h.start:h.end + 1][:, candidates].any(axis=0)]
            if free.size == 0:
                raise RuntimeError(f"no free eligible stock for the holding starting at row {h.start}: the universe "
                                   f"is smaller than the book")
            pick = int(free[rng.integers(free.size)])
        occupied[h.start:h.end + 1, pick] = True
        picks.append(pick)
        last = min(h.end, n_rows - 2)                  # weight at row t earns the return of row t + 1
        if last >= h.start:
            daily[h.start:last + 1] += weights[h.start:last + 1, h.column] * returns[h.start + 1:last + 2, pick]
    return daily, picks


def selection_test(weights: pd.DataFrame, returns: pd.DataFrame, eligible: pd.DataFrame,
                   fixed: Iterable[str] = (), draws: int = DEFAULT_DRAWS, seed: int = DEFAULT_SEED,
                   min_holdings: int = MIN_HOLDINGS) -> Dict:
    """The randomised-selection test on aligned panels (rows: sessions; columns: symbols).

    ``weights`` are the book's end-of-session weights, ``returns`` the adjusted close-to-close returns of the same
    sessions, ``eligible`` the universe on each session (the names the book could have bought after that close).
    Raises on panels that do not line up, on negative or missing weights, on too few holdings or draws, and when a
    holding has no free eligible stock.
    """
    if draws < MIN_DRAWS:
        raise ValueError(f"{draws} draws resolve p only to 1/{draws + 1}; use at least {MIN_DRAWS}")
    w = weights.astype("float64")
    if not w.index.equals(returns.index) or not w.index.equals(eligible.index):
        raise ValueError("weights, returns and eligible must share the same sessions")
    if not (w.columns.equals(returns.columns) and w.columns.equals(eligible.columns)):
        raise ValueError("weights, returns and eligible must share the same columns, in order")
    if not np.isfinite(w.to_numpy()).all() or (w.to_numpy() < -WEIGHT_TOL).any():
        raise ValueError("weights must be finite and not negative (a long-only book)")
    fixed = set(fixed)
    core = [i for i, s in enumerate(w.columns) if s not in fixed]
    W = w.to_numpy()
    R = np.nan_to_num(returns.to_numpy(dtype="float64"), nan=0.0, posinf=0.0, neginf=0.0)
    E = eligible.to_numpy(dtype=bool).copy()
    E[:, [i for i, s in enumerate(w.columns) if s in fixed]] = False
    plan = holdings(W, core)
    if len(plan) < min_holdings:
        raise ValueError(f"{len(plan)} holdings: too few for a randomised test (at least {min_holdings})")
    off = [h for h in plan if not E[max(h.start - 1, 0), h.column]]
    book = core_returns(W, R, core)
    rng = np.random.default_rng(seed)
    null = np.array([_draw(rng, W, R, E, plan)[0].mean() for _ in range(draws)])
    observed = float(book.mean())
    above = int((null >= observed - TIE_TOL).sum())
    pct = np.percentile(null, [5, 50, 95]) * 252
    return {"holdings": len(plan), "draws": draws, "seed": seed,
            "sessions": [str(w.index[0].date()), str(w.index[-1].date())],
            "book_core_gross_annual": round(observed * 252, 4),
            "random_core_gross_annual": {"p5": round(float(pct[0]), 4), "median": round(float(pct[1]), 4),
                                         "p95": round(float(pct[2]), 4)},
            "draws_at_or_above_book": above, "p": (1 + above) / (1 + draws),
            "z": round((observed - float(null.mean())) / float(null.std(ddof=1)), 2) if null.std() > 0 else None,
            "holdings_not_in_universe_at_decision": len(off)}


def renamed(symbols: Iterable[str], known: set, changes: pd.DataFrame) -> Dict[str, str]:
    """A recorded run's symbols that the store now knows under a later name (the rename table's ``old`` -> ``new``,
    followed through chains).  Raises when two symbols would land on one name."""
    step = dict(zip(changes["old"].astype(str), changes["new"].astype(str))) if len(changes) else {}
    out: Dict[str, str] = {}
    for s in symbols:
        name, seen = str(s), {str(s)}
        while name not in known and name in step and step[name] not in seen:
            name = step[name]
            seen.add(name)
        if name != s and name in known:
            out[str(s)] = name
    targets = list(out.values()) + [s for s in symbols if s not in out]
    if len(targets) != len(set(targets)):
        raise ValueError("two of the run's symbols map to one name in the store")
    return out


def _panels(run_dir: Path, store: str, data_start: Optional[str], start: Optional[str],
            end: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Tuple[str, ...]]:
    """A recorded run's weights with the store's returns and the engine's universe on the same sessions."""
    from nse_engine.config import EngineConfig
    from nse_engine.data.panel import load_change_table, load_market_data
    from nse_engine.universe import compute_universe_panel

    cfg = EngineConfig.from_dict(json.loads((run_dir / "config.json").read_text()))
    weights = pd.read_parquet(run_dir / "weights.parquet")
    weights.index = pd.DatetimeIndex(weights.index)
    first = data_start or f"{weights.index[0].year - 2}-01-01"
    last = str(weights.index[-1].date())             # load as the run did: the pre-filter and universe use the whole
    weights = weights.loc[start or weights.index[0]:end or weights.index[-1]]
    weights = weights.loc[:, (weights.abs() > WEIGHT_TOL).any()]          # names held in the window
    data = load_market_data(store, first, last, series=cfg.data.series,
                            min_median_value_inr=cfg.data.load_min_median_value_inr,
                            float_dtype=cfg.data.float_dtype, include_symbols=cfg.sleeves.symbols,
                            adjust_dividends=cfg.data.adjust_dividends)
    sleeves = tuple(cfg.sleeves.symbols)
    mask = compute_universe_panel(data, cfg.universe, exclude=sleeves).mask
    close = data.close.astype("float64")
    weights = weights.rename(columns=renamed(weights.columns, set(close.columns), load_change_table(store)))
    missing = sorted(set(weights.columns) - set(close.columns))
    if missing:
        raise ValueError(f"{len(missing)} weighted symbols are not in the store, first {missing[0]}")
    cols = list(close.columns)
    weights = weights.reindex(columns=cols, fill_value=0.0)
    rets = close.pct_change(fill_method=None).reindex(index=weights.index, columns=cols)
    eligible = mask.reindex(index=weights.index, columns=cols, fill_value=False).fillna(False).astype(bool)
    if rets.index.difference(close.index).size:
        raise ValueError("the run has sessions the store does not")
    return weights, rets, eligible, sleeves


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nse_engine.validation.selection_skill",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, help="a recorded run's id under the runs directory")
    ap.add_argument("--runs-dir", default="data/nse_engine/runs")
    ap.add_argument("--store", default="data/nse_engine/store")
    ap.add_argument("--data-start", default="2012-01-02", help="the data anchor the universe is computed from")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    weights, rets, eligible, sleeves = _panels(Path(args.runs_dir) / args.run, args.store, args.data_start,
                                               args.start, args.end)
    out = {"run": args.run, **selection_test(weights, rets, eligible, sleeves, args.draws, args.seed)}
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
