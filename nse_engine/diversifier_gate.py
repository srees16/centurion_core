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

Amended 25 Oct 2026 (GT13, the owner's decision; protocol 5.1): MaxDD is no condition anywhere.  A stream
passes as a Sharpe lever only, Sharpe up with p <= 0.10 on both books (``sharpe_gain``); the drawdown route is
retired.  MaxDD and Calmar (full window and the median of rolling 3-year windows) are reported.  ``dg1`` and
``dg2`` still reproduce their recorded 19 Oct verdicts; ``rt1`` re-judges DG1's streams under GT13 (plan 5ak).

    python -m nse_engine.diversifier_gate dg1 --mcx <mcx_results.json>    # plan 5ah
    python -m nse_engine.diversifier_gate rt1 --mcx <mcx_results.json>    # plan 5ak
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

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
FAMILY_N = {"options": 18, "commodity": 2}                   # options: A1G is the 18th (26 Oct 2026)


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


CALMAR_WINDOW, CALMAR_STEP = 756, 21                          # 3 years of sessions, a month apart


def _calmar(x: pd.Series) -> float:
    """CAGR (252-session years) over |MaxDD|; nan without a drawdown."""
    growth, dd = float((1 + x).prod()), max_drawdown(x)
    return float((growth ** (252 / len(x)) - 1) / -dd) if len(x) and growth > 0 and dd < 0 else float("nan")


def rolling_calmar_median(x: pd.Series) -> float:
    """GT13's Calmar: the median of rolling 3-year windows, a month apart (Calmar's own 36-month horizon)."""
    vals = [_calmar(x.iloc[i:i + CALMAR_WINDOW]) for i in range(0, len(x) - CALMAR_WINDOW + 1, CALMAR_STEP)]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.median(vals)) if vals else float("nan")


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
    dd = max_drawdown(x)
    return {"cagr": s["cagr"], "excess_sharpe": excess_sharpe(x), "max_drawdown": dd,
            "alpha_annual": s["alpha_annual"], "y2008": s["y2008"],
            "calmar": float(s["cagr"] / -dd) if dd < 0 else float("nan"),
            "calmar_3y_median": rolling_calmar_median(x)}


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
    sharpe_gain = bool(c["excess_sharpe"] > b["excess_sharpe"] and p <= SHARPE_P)      # GT13's Sharpe lever
    sharpe_route = bool(sharpe_gain and c["max_drawdown"] >= b["max_drawdown"] - SHARPE_DD_SLACK)   # 19 Oct
    dd_route = bool(c["max_drawdown"] >= b["max_drawdown"] + DD_GAIN
                    and c["excess_sharpe"] >= b["excess_sharpe"] - DD_SHARPE_SLACK)
    out = {"window": [str(sessions[0].date()), str(sessions[-1].date())], "book": b, "equal_risk": c,
           "tested_weight": t, "delta_sharpe": c["excess_sharpe"] - b["excess_sharpe"], "p": p,
           "sharpe_gain": sharpe_gain, "sharpe_route": sharpe_route, "drawdown_route": dd_route,
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


def rt1(mcx_json: Optional[str] = None, runs_dir: str = "data/nse_engine") -> Dict:
    """Plan 5ak: DG1's streams re-judged under GT13.  A Sharpe lever on both books, and a stream whose 2013-25 was
    seen must hold its gain in 2008-12; no MaxDD condition.  Every pass is provisional (the rule changed after
    DG1's results were seen); the 19 Oct verdict stays beside it."""
    out = dg1(mcx_json, runs_dir)
    out["verdict_19oct"] = out.pop("verdict")
    out["verdict"] = {}
    for name, by_base in out["streams"].items():
        ok = all(v["sharpe_gain"] for v in by_base.values())
        if name in OUT_OF_SAMPLE_2008_12:
            ok = ok and all(v["oos_2008_12"]["holds"] for v in by_base.values())
        out["verdict"][name] = {"sharpe_lever": ok, "pass": ok, "provisional": ok}
    return out


#: A1C (plan 5am): A1 with each cycle's maximum loss capped at a share of the book, the owner's sizing for live.
A1_CAPITAL = 750_000.0


def capped_weights(sessions: pd.DatetimeIndex, cap: float, cycles: pd.DataFrame, a1_equity: pd.Series
                   ) -> Tuple[pd.Series, pd.DataFrame]:
    """While a cycle is open (fill to exit), the overlay weight min(3, cap x A1's equity before the fill / the spread's
    maximum loss), so it can lose at most ``cap`` of the book; 0 between cycles.  With each traded cycle's weight,
    P&L and maximum loss as shares of the book."""
    w = pd.Series(0.0, index=sessions)
    rows = []
    for c in cycles[cycles["action"] == "traded"].itertuples():
        fill, exit_ = pd.Timestamp(c.fill_date), pd.Timestamp(c.exit_date) if pd.notna(c.exit_date) else sessions[-1]
        before = a1_equity[a1_equity.index < fill]
        base = float(before.iloc[-1]) if len(before) else A1_CAPITAL
        wc = min(CAPS["overlay"], cap * base / float(c.max_loss_inr))
        w[(w.index >= fill) & (w.index <= exit_)] = wc
        pnl = float(c.pnl_inr) if pd.notna(getattr(c, "pnl_inr", float("nan"))) else float("nan")
        rows.append({"fill": str(fill.date()), "weight": wc, "pnl_share": wc * pnl / base,
                     "max_loss_share": wc * float(c.max_loss_inr) / base})
    return w, pd.DataFrame(rows)


def a1c(cap: float = 0.10, reported_caps: Sequence[float] = (0.15,), runs_dir: str = "data/nse_engine",
        fo_store_dir: str = "data/nse_engine/fo_store") -> Dict:
    """Plan 5am: A1 at A1C's capped weight on both books (the GT13 Sharpe lever, which must hold in 2008-12), and
    its 2026 from the options family's opened holdout, reported."""
    run = Path(runs_dir) / "runs_options" / OPTIONS["O2-A1"]
    raw = _read(run / "returns.csv")
    a1_eq = (1 + raw).cumprod() * A1_CAPITAL
    cycles = pd.read_csv(run / "cycles.csv")
    out: Dict = {"cap": cap, "books": {}, "reported": {}}
    for base in (bs.backtest_base(), bs.walk_forward_base()):
        for c in (cap, *reported_caps):
            res = _capped_eval(base, raw, cycles, a1_eq, c)
            (out["books"] if c == cap else out["reported"].setdefault(str(c), {}))[base.name] = res
    out["pass"] = all(v["sharpe_gain"] and v["holds_2008_12"] for v in out["books"].values())
    out["holdout_2026_reported"] = _a1c_2026(cap, runs_dir, fo_store_dir)
    return out


def _capped_eval(base: bs.Base, raw: pd.Series, cycles: pd.DataFrame, equity: pd.Series, cap: float) -> Dict:
    """One book with an options stream at A1C's capped weight (plan 5am): both cards, the bootstrap p, the GT13
    Sharpe lever (which must hold in 2008-12), the sizing, and the years (book, combined)."""
    tri = bs.nifty_tri(base.returns.index)
    liquid = bs.bond_returns(base.returns.index, bs.BondConfig())["liquid"]
    sessions = base.returns.index[base.returns.index >= raw.index[0]]
    book = _live_book(base, liquid, sessions)
    stream = carry_to(raw, sessions).reindex(sessions).fillna(0.0)
    early = sessions[(sessions.year >= 2008) & (sessions.year <= 2012)]
    w, cyc = capped_weights(sessions, cap, cycles, equity)
    comb = combine(book, stream, w, "overlay")
    b, k = _card(book, tri), _card(comb, tri)
    p = bootstrap_p(book, comb)
    return {"book": b, "capped": k, "delta_sharpe": k["excess_sharpe"] - b["excess_sharpe"], "p": p,
            "sharpe_gain": bool(k["excess_sharpe"] > b["excess_sharpe"] and p <= SHARPE_P),
            "holds_2008_12": bool(excess_sharpe(comb[early]) >= excess_sharpe(book[early])),
            "weight_open_mean": float(w[w > 0].mean()), "open_share": float((w > 0).mean()),
            "worst_cycle_share": float(cyc["pnl_share"].min()), "max_loss_share_max": float(cyc["max_loss_share"].max()),
            "years": {int(y): [round(float((1 + book[book.index.year == y]).prod() - 1), 4),
                               round(float((1 + comb[comb.index.year == y]).prod() - 1), 4)]
                      for y in sorted(set(sessions.year))}}


def _a1c_2026(cap: float, runs_dir: str, fo_store_dir: str, vix: Optional[pd.Series] = None) -> Dict:
    """A1C on 1 Jan-11 Sep 2026 beside the deployed book's recorded holdout run (with ``vix``: A1G, guarded by the
    book's rule replayed on that run's equity and India VIX).  The options family's holdout was opened for A1
    (plan 5al), so this is reported, not a fresh look."""
    from dataclasses import replace

    from kite_connect.options.backtest import HOLDOUT_BOOK_RUN, HOLDOUT_WINDOW, run_sleeve
    from kite_connect.options.sleeves import CANDIDATES

    book = _read(Path(runs_dir) / "runs" / HOLDOUT_BOOK_RUN / "returns.csv")[HOLDOUT_WINDOW[0]:HOLDOUT_WINDOW[1]]
    name, skip = ("A1", None) if vix is None else ("A1G", guard_dates(book, vix))
    res = run_sleeve(replace(CANDIDATES[name], start=HOLDOUT_WINDOW[0], end=HOLDOUT_WINDOW[1]), fo_store_dir,
                     skip_entry_dates=skip)
    a1 = res.returns.reindex(book.index).fillna(0.0)
    w, cyc = capped_weights(book.index, cap, pd.DataFrame(res.cycles), res.equity)
    comb = book + w * a1
    card = lambda x: {"return": float((1 + x).prod() - 1), "excess_sharpe": excess_sharpe(x), "max_drawdown": max_drawdown(x)}
    out = {"book": card(book), "capped": card(comb), "weight_open_mean": float(w[w > 0].mean()) if (w > 0).any() else 0.0,
           "cycles": cyc.to_dict("records")}
    if vix is not None:
        out["agrees"] = bool(out["capped"]["excess_sharpe"] >= out["book"]["excess_sharpe"])
        out["guard_skipped"] = [str(pd.Timestamp(c["decision_date"]).date()) for c in res.cycles
                                if c.get("action") == "skipped: guard"]
    return out


# ── A1G (plan 5aj, amended 26 Oct 2026): A1 with a crash guard ──────────────────
#: The walk-forward book's guard replays the deployed book's drawdown rule (20/30/35, half 0.5, re-arm 60).
GUARD_RULE = dict(enabled=True, halt_dd=0.20, half_dd=0.30, risk_off_dd=0.35, half_scale=0.5, rearm_sessions=60)


def guard_vix(store_dir: str = "data/nse_engine/store",
              cache: str = "data/nse_engine/store_ext2006/external/yf_indiavix.parquet") -> pd.Series:
    """India VIX closes: the store's (NSE, from May 2014), Yahoo's cached series before it (from Mar 2008), as the
    engine's panel back-fills it."""
    from nse_engine.crash_recovery import store_index

    nse = store_index("INDIAVIX", store_dir)
    yf = pd.read_parquet(cache)["close"].astype("float64")
    yf.index = pd.DatetimeIndex(yf.index)
    return nse.combine_first(yf).sort_index()


def guard_dates(book: pd.Series, vix: pd.Series, states: Optional[pd.Series] = None) -> pd.DatetimeIndex:
    """A1G's guarded sessions beside a book (daily returns): its drawdown rule not normal at the close (``states``
    as recorded, else the rule replayed on the book's equity), or India VIX at or above 22."""
    from kite_connect.options.sleeves import guarded_sessions
    from nse_engine.config import DrawdownConfig
    from nse_engine.drawdown import replay

    if states is None:
        states = replay((1 + book).cumprod(), DrawdownConfig(**GUARD_RULE))["state"]
    return guarded_sessions(states, vix)


def a1g(cap: float = 0.10, reported_caps: Sequence[float] = (0.15,), runs_dir: str = "data/nse_engine",
        fo_store_dir: str = "data/nse_engine/fo_store", store_dir: str = "data/nse_engine/store",
        record_runs: bool = True) -> Dict:
    """Plan 5aj as amended: A1G, one run per book (its own guard), at A1C's capped weight and judged by A1C's gate
    (the GT13 Sharpe lever, holding in 2008-12), beside A1C on the same book; its 2026 reported."""
    from dataclasses import replace

    from kite_connect.options.backtest import record, run_sleeve
    from kite_connect.options.sleeves import CANDIDATES

    a1_run = Path(runs_dir) / "runs_options" / OPTIONS["O2-A1"]
    a1_raw = _read(a1_run / "returns.csv")
    a1_cycles = pd.read_csv(a1_run / "cycles.csv")
    a1_eq = (1 + a1_raw).cumprod() * A1_CAPITAL
    window = (str(a1_raw.index[0].date()), str(a1_raw.index[-1].date()))
    vix = guard_vix(store_dir)
    recorded_states = pd.read_csv(Path(bs.BOOK_RUNS) / bs.BACKTEST_RUN / "drawdown.csv", index_col=0)["state"]
    recorded_states.index = pd.DatetimeIndex(recorded_states.index)
    out: Dict = {"cap": cap, "config_hash": CANDIDATES["A1G"].config_hash(), "books": {}, "reported": {}, "a1c": {},
                 "runs": {}, "skipped": {}}
    for base in (bs.backtest_base(), bs.walk_forward_base()):
        states = recorded_states if base.name == "backtest" else None
        skip = guard_dates(base.returns, vix, states)
        cfg = replace(CANDIDATES["A1G"], start=window[0], end=window[1],
                      name=f"{CANDIDATES['A1G'].name}, beside the {base.name} book")
        res = run_sleeve(cfg, fo_store_dir, skip_entry_dates=skip)
        if record_runs:
            out["runs"][base.name] = Path(record(res, str(Path(runs_dir) / "runs_options"))).name
        cycles = pd.DataFrame(res.cycles)
        for c in (cap, *reported_caps):
            r = _capped_eval(base, res.returns, cycles, res.equity, c)
            (out["books"] if c == cap else out["reported"].setdefault(str(c), {}))[base.name] = r
        a1c_same = _capped_eval(base, a1_raw, a1_cycles, a1_eq, cap)
        out["a1c"][base.name] = {k: a1c_same[k] for k in ("capped", "delta_sharpe", "p", "sharpe_gain", "holds_2008_12",
                                                          "worst_cycle_share")}
        a1_by_expiry = a1_cycles.assign(expiry=pd.to_datetime(a1_cycles["expiry"]).dt.date.astype(str)).set_index("expiry")
        out["skipped"][base.name] = [
            {"decision": str(pd.Timestamp(c["decision_date"]).date()), "expiry": str(pd.Timestamp(c["expiry"]).date()),
             "a1_action": a1_by_expiry["action"].get(str(pd.Timestamp(c["expiry"]).date())),
             "a1_pnl_inr": (lambda v: None if pd.isna(v) else float(v))(
                 a1_by_expiry["pnl_inr"].get(str(pd.Timestamp(c["expiry"]).date()), np.nan))}
            for c in res.cycles if c.get("action") == "skipped: guard"]
    out["pass"] = all(v["sharpe_gain"] and v["holds_2008_12"] for v in out["books"].values())
    out["holdout_2026_reported"] = _a1c_2026(cap, runs_dir, fo_store_dir, vix=vix)
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
    p = sub.add_parser("rt1", help="plan 5ak: DG1's streams re-judged under GT13 (no MaxDD condition)")
    p.add_argument("--mcx", help="mcx_results.json from the Kaggle `mcx` task with --reported-start")
    p.add_argument("--out", help="also write the result here")
    p = sub.add_parser("a1c", help="plan 5am: A1 with each cycle's maximum loss capped at a share of the book")
    p.add_argument("--cap", type=float, default=0.10)
    p.add_argument("--out", help="also write the result here")
    p = sub.add_parser("a1g", help="plan 5aj (amended 26 Oct): A1 with the crash guard, at A1C's capped weight")
    p.add_argument("--cap", type=float, default=0.10)
    p.add_argument("--out", help="also write the result here")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.cmd in ("a1c", "a1g"):
        out = (a1c if args.cmd == "a1c" else a1g)(args.cap)
    else:
        out = {"dg1": dg1, "rt1": rt1}.get(args.cmd, dg2)(args.mcx)
    text = json.dumps(out, indent=1, default=str)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
