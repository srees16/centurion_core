"""
The diversifier gate (research protocol 5.1, your decision 19 Oct 2026) and its first use, DG1 (validation plan
5ah): a stream beside the book passes as a Sharpe lever or as a drawdown lever, judged on the honest base.

- Books: the 2008-25 walk-forward (the headline) and the deployed rules' 2008-25 backtest with the drawdown
  rule (in-sample for 2013-25: an extra condition only); idle cash earns the liquid rate (the 1D rate less
  0.10%) and the book carries D2's live haircut, in the base and the combination alike.
- Weight: equal risk on trailing volatility (up to 252 sessions, at least 63), re-set each month; an overlay
  (book + w x stream, margin pledged from the book) is capped at 3x, a funded sleeve ((1 - w) x book + w x stream)
  at 50%.  The stream's tested weight (0.25) is reported beside it.
- Sharpe lever: combined excess Sharpe above the book's, one-sided p <= 0.10 in a paired stationary bootstrap
  (63-session mean blocks, 2,000 resamples), MaxDD no more than 1 point worse.
- Drawdown lever: MaxDD at least 3 points shallower, Sharpe no more than 0.02 lower.
CAGR, alpha and the deflated Sharpe are reported, not gates.

    python -m nse_engine.diversifier_gate dg1 --mcx <mcx_results.json>    # plan 5ah
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from nse_engine import bond_stream as bs
from nse_engine.validation.sleeve_gates import carry_to

logger = logging.getLogger(__name__)

RF = 0.065
#: D2's measured live haircut on the book (a day's delay and twice the impact), a year at full exposure; plan
#: 5ah's clarification of 19 Oct: every figure live-like, the base and the combination alike.
LIVE_HAIRCUT = 0.03
CAPS = {"overlay": 3.0, "sleeve": 0.5}
TESTED_WEIGHT = 0.25
VOL_WINDOW, VOL_MIN = 252, 63
SHARPE_P, SHARPE_DD_SLACK = 0.10, 0.01
DD_GAIN, DD_SHARPE_SLACK = 0.03, 0.02
BOOT_N, BOOT_BLOCK, BOOT_SEED = 2000, 63, 7

#: DG1's streams: (kind, registry, run) for recorded options overlays; MCX comes from the Kaggle task's JSON.
OPTIONS = {"O2-A1": "20261010T122956596793Z_ae503fc4",       # 15-day 2/3-SD bear call spread (LN-T20), 2007-25
           "O3-X2": "20261009T143542327966Z_ac5d03e7",       # TRIN bands, near-month NIFTY future, 2007-25
           "O4-Y3": "20261009T163304480870Z_9a12cbab"}       # NIFTYBEES overnight on sentiment, from 2012
OUT_OF_SAMPLE_2008_12 = ("O2-A1", "O3-X2")                   # their 2013-25 was seen (19 Oct look)
FAMILY_N = {"options": 17, "commodity": 2}


def _read(path: Path) -> pd.Series:
    r = pd.read_csv(path, index_col=0).squeeze("columns").astype("float64")
    r.index = pd.DatetimeIndex(r.index)
    return r


def excess_sharpe(x: pd.Series) -> float:
    ex = x - RF / 252
    return float(ex.mean() / ex.std(ddof=1) * np.sqrt(252)) if ex.std(ddof=1) > 0 else float("nan")


def max_drawdown(x: pd.Series) -> float:
    eq = (1 + x).cumprod()
    return float((eq / eq.cummax() - 1).min())


def equal_risk(book: pd.Series, stream: pd.Series, kind: str) -> pd.Series:
    """Daily weights, re-set on each month's first session from trailing volatility (no look-ahead)."""
    w = pd.Series(np.nan, index=book.index)
    months = book.index.to_period("M")
    firsts = book.index[np.r_[True, months[1:] != months[:-1]]]
    for day in firsts:
        sb, ss = book[book.index < day].tail(VOL_WINDOW), stream[stream.index < day].tail(VOL_WINDOW)
        if len(ss) < VOL_MIN or ss.std() <= 0:
            w[day] = 0.0
            continue
        vb, vs = sb.std(), ss.std()
        w[day] = min(vb / vs, CAPS[kind]) if kind == "overlay" else min(vb / (vb + vs), CAPS[kind])
    return w.ffill().fillna(0.0)


def combine(book: pd.Series, stream: pd.Series, w, kind: str) -> pd.Series:
    return book + w * stream if kind == "overlay" else (1 - w) * book + w * stream


def _boot_indices(T: int) -> np.ndarray:
    """Paired stationary-bootstrap resamples of T sessions (geometric blocks, mean BOOT_BLOCK, fixed seed)."""
    rng = np.random.default_rng(BOOT_SEED)
    idx = np.empty((BOOT_N, T), dtype=np.int64)
    idx[:, 0] = rng.integers(0, T, BOOT_N)
    jump = rng.random((BOOT_N, T)) < 1.0 / BOOT_BLOCK
    fresh = rng.integers(0, T, (BOOT_N, T))
    for t in range(1, T):
        idx[:, t] = np.where(jump[:, t], fresh[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def bootstrap_p(base: pd.Series, comb: pd.Series) -> float:
    """One-sided p of 'the combination's excess Sharpe is no better than the book's': the share of paired
    stationary-bootstrap resamples where it is not."""
    a, b = (base - RF / 252).to_numpy(), (comb - RF / 252).to_numpy()
    idx = _boot_indices(len(a))
    sa, sb = a[idx], b[idx]
    d = sb.mean(1) / sb.std(1, ddof=1) - sa.mean(1) / sa.std(1, ddof=1)
    return float((d <= 0).mean())


def bootstrap_dd_share(base: pd.Series, comb: pd.Series, gain: float = DD_GAIN) -> float:
    """Share of the same paired resamples in which the combination's MaxDD is at least ``gain`` shallower."""
    idx = _boot_indices(len(base))

    def mdd(x: np.ndarray) -> np.ndarray:
        eq = np.cumprod(1.0 + x[idx], axis=1)
        return (eq / np.maximum.accumulate(eq, axis=1) - 1.0).min(axis=1)

    return float((mdd(comb.to_numpy()) >= mdd(base.to_numpy()) + gain).mean())


def _live_book(base: bs.Base, liquid: pd.Series, sessions: pd.DatetimeIndex) -> pd.Series:
    """The book as it would trade: idle cash at the liquid rate, less D2's live haircut on what is invested."""
    invested = base.invested.reindex(sessions).clip(lower=0.0)
    idle = (1.0 - invested).clip(0.0, 1.0)
    return (base.returns.reindex(sessions) + idle * liquid.reindex(sessions).fillna(0.0)
            - LIVE_HAIRCUT * invested / 252)


def _card(x: pd.Series, tri: pd.Series) -> Dict:
    s = bs._summary(x, tri)
    return {"cagr": s["cagr"], "excess_sharpe": excess_sharpe(x), "max_drawdown": max_drawdown(x),
            "alpha_annual": s["alpha_annual"], "y2008": s["y2008"]}


def evaluate(name: str, kind: str, raw: pd.Series, base: bs.Base, tri: pd.Series, liquid: pd.Series,
             family: str) -> Dict:
    """One stream on one book: both routes at equal risk, the tested weight beside it."""
    from nse_engine.validation.dsr import deflated_sharpe

    first = raw.index[0]
    sessions = base.returns.index[base.returns.index >= max(first, base.returns.index[0])]
    book = _live_book(base, liquid, sessions)
    stream = carry_to(raw, sessions).reindex(sessions).fillna(0.0)
    if kind == "sleeve":                      # a funded sleeve's margin and idle cash earn the liquid rate
        stream = stream + liquid.reindex(sessions).fillna(0.0)
    w = equal_risk(book, stream, kind)
    comb = combine(book, stream, w, kind)
    tested = combine(book, stream, TESTED_WEIGHT, kind)
    b, c, t = _card(book, tri), _card(comb, tri), _card(tested, tri)
    p = bootstrap_p(book, comb)
    sharpe_route = bool(c["excess_sharpe"] > b["excess_sharpe"] and p <= SHARPE_P
                        and c["max_drawdown"] >= b["max_drawdown"] - SHARPE_DD_SLACK)
    dd_route = bool(c["max_drawdown"] >= b["max_drawdown"] + DD_GAIN
                    and c["excess_sharpe"] >= b["excess_sharpe"] - DD_SHARPE_SLACK)
    out = {"window": [str(sessions[0].date()), str(sessions[-1].date())], "book": b, "equal_risk": c,
           "tested_weight": t, "delta_sharpe": c["excess_sharpe"] - b["excess_sharpe"], "p": p,
           "sharpe_route": sharpe_route, "drawdown_route": dd_route,
           "weight_mean": float(w.mean()), "weight_last": float(w.iloc[-1]),
           "cap_bound_share": float((w >= CAPS[kind] - 1e-9).mean()),
           "correlation": float(((1 + stream).resample("ME").prod() - 1).corr((1 + book).resample("ME").prod() - 1)),
           "stream_alone": {"excess_sharpe_rf0": float(stream.mean() / stream.std() * np.sqrt(252)),
                            "max_drawdown": max_drawdown(stream),
                            "dsr": float(deflated_sharpe(stream, n_trials=FAMILY_N[family],
                                                         rf_annual=RF if kind == "sleeve" else 0.0)["dsr"])},
           "years": {int(y): [round(float((1 + book[book.index.year == y]).prod() - 1), 4),
                              round(float((1 + comb[comb.index.year == y]).prod() - 1), 4)]
                     for y in sorted(set(sessions.year))}}
    if name in OUT_OF_SAMPLE_2008_12:
        early = sessions[(sessions.year >= 2008) & (sessions.year <= 2012)]
        out["oos_2008_12"] = {"book": excess_sharpe(book[early]), "combined": excess_sharpe(comb[early]),
                              "holds": bool(excess_sharpe(comb[early]) >= excess_sharpe(book[early]))}
    return out


def dg1(mcx_json: Optional[str] = None, runs_dir: str = "data/nse_engine") -> Dict:
    """Plan 5ah: A1, X2, Y3 (overlays) and GT5-A, GT5-B (funded sleeves) on both books."""
    streams = {k: ("overlay", "options", _read(Path(runs_dir) / "runs_options" / run / "returns.csv"))
               for k, run in OPTIONS.items()}
    if mcx_json:
        doc = json.loads(Path(mcx_json).read_text())
        for key, rep in doc["reported"].items():
            r = pd.Series(rep["returns"], dtype="float64")
            r.index = pd.DatetimeIndex(r.index)
            streams[key] = ("sleeve", "commodity", r.sort_index())
    out: Dict = {"streams": {}, "verdict": {}}
    for base in (bs.backtest_base(), bs.walk_forward_base()):
        tri = bs.nifty_tri(base.returns.index)
        liquid = bs.bond_returns(base.returns.index, bs.BondConfig())["liquid"]
        for name, (kind, family, raw) in streams.items():
            out["streams"].setdefault(name, {})[base.name] = evaluate(name, kind, raw, base, tri, liquid, family)
    for name, by_base in out["streams"].items():
        sharpe = all(v["sharpe_route"] for v in by_base.values())
        if name in OUT_OF_SAMPLE_2008_12:
            sharpe = sharpe and all(v["oos_2008_12"]["holds"] for v in by_base.values())
        dd = all(v["drawdown_route"] for v in by_base.values())
        out["verdict"][name] = {"sharpe_lever": sharpe, "drawdown_lever": dd, "pass": sharpe or dd}
    return out


#: DG2 (plan 5ai): drawdown levers at the sleeve weight fixed in their own pre-registrations (gilts 5ag, MCX 5y),
#: alone and as the pair that would run together.
LEVERS = {"gilts 25%": {"gilt": 0.25}, "MCX-B 25%": {"GT5-B": 0.25}, "MCX-A 25%": {"GT5-A": 0.25},
          "gilts 25% + MCX-B 25%": {"gilt": 0.25, "GT5-B": 0.25}}
DD_BOOT_SHARE = 0.80


def drawdown_lever(sleeves: Dict[str, pd.Series], weights: Dict[str, float], base: bs.Base, tri: pd.Series,
                   liquid: pd.Series) -> Dict:
    """Protocol 5.1's drawdown route at the levers' own weights: (1 - sum w) x book + sum w_i x sleeve_i."""
    sessions = base.returns.index
    book = _live_book(base, liquid, sessions)
    comb = (1.0 - sum(weights.values())) * book
    for name, w in weights.items():
        comb = comb + w * sleeves[name].reindex(sessions).fillna(0.0)
    b, c = _card(book, tri), _card(comb, tri)
    share = bootstrap_dd_share(book, comb)
    checks = {"maxdd_3pts": bool(c["max_drawdown"] >= b["max_drawdown"] + DD_GAIN),
              "sharpe_within_0.02": bool(c["excess_sharpe"] >= b["excess_sharpe"] - DD_SHARPE_SLACK),
              "bootstrap_80pct": bool(share >= DD_BOOT_SHARE)}
    return {"book": b, "combined": c, "bootstrap_share_3pts": share, "checks": checks, "pass": all(checks.values()),
            "years": {int(y): [round(float((1 + book[book.index.year == y]).prod() - 1), 4),
                               round(float((1 + comb[comb.index.year == y]).prod() - 1), 4)]
                      for y in sorted(set(sessions.year))}}


def dg2(mcx_json: str) -> Dict:
    """Plan 5ai: gilts, MCX-B, MCX-A and the gilts + MCX-B pair at their own 25%, on both books."""
    doc = json.loads(Path(mcx_json).read_text())
    out: Dict = {"levers": {}, "verdict": {}}
    for base in (bs.walk_forward_base(), bs.backtest_base()):
        idx = base.returns.index
        tri = bs.nifty_tri(idx)
        br = bs.bond_returns(idx, bs.BondConfig())
        sleeves = {"gilt": br["gilt"]}
        for key in ("GT5-A", "GT5-B"):
            r = pd.Series(doc["reported"][key]["returns"], dtype="float64")
            r.index = pd.DatetimeIndex(r.index)
            sleeves[key] = carry_to(r.sort_index(), idx).reindex(idx).fillna(0.0) + br["liquid"]
        for name, weights in LEVERS.items():
            out["levers"].setdefault(name, {})[base.name] = drawdown_lever(sleeves, weights, base, tri, br["liquid"])
    for name, by_base in out["levers"].items():
        out["verdict"][name] = all(v["pass"] for v in by_base.values())
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nse_engine.diversifier_gate", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dg1", help="plan 5ah: the options near misses and MCX trend under the diversifier gate")
    p.add_argument("--mcx", help="mcx_results.json from the Kaggle `mcx` task with --reported-start")
    p.add_argument("--out", help="also write the result here")
    p = sub.add_parser("dg2", help="plan 5ai: drawdown levers (gilts, MCX) at their own 25%, alone and paired")
    p.add_argument("--mcx", required=True, help="mcx_results.json with the 2007-25 reported returns")
    p.add_argument("--out", help="also write the result here")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out = dg1(args.mcx) if args.cmd == "dg1" else dg2(args.mcx)
    text = json.dumps(out, indent=1, default=str)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
