"""
The golden-target gates for every paper book (tracker GTR0's decisions, GTR6): the yardstick the Saturday email
carries, and the leverage each book has earned.

Gates (all required): 1 deflated Sharpe on 2008-25, every trial of every family counted, >= 0.95 (in-sample, fixed
settings); 2 net Sharpe 2008-25 on the as-run basis, unlevered, >= 0.8; 3 walk-forward Sharpe on the as-run basis
>= 1.0; 4 the worst era (2008-12) Sharpe >= 0.2; 5 paper inside its Monte Carlo band (5th-95th percentile of the
return since its start) at 60 sessions.  Tail vetoes, never optimised, on the as-run book at its leverage, with 90%
block-bootstrap ranges: MaxDD worse than -30%; Calmar under 0.5; daily CVaR(5%) worse than 1.5x NIFTY TRI's;
monthly skew under -1; Sharpe lost to market impact at Rs 1 crore, the intended fund size (owner, 16 Nov 2026,
GTR10), over 0.1.

Every Sharpe is excess over 6.5% (protocol gate 1).  "As-run" is the book as it trades today: D2's 3% a year on
invested capital, idle cash earning nothing (the owner's basis since 5 Nov 2026).  Leverage is the funding gate's
(``funding_gate.implied_leverage``): half-Kelly on the as-run deflated Sharpe, 1.0x unless gate 1 passes, at most
1.5x; any leverage goes through the funding gate's request first.

Gates 1-4 and the vetoes need the research data (the run registry, the walk-forwards, the extended store), so
:func:`build` runs on the research machine and writes ``config/golden_gates.json``; the Saturday job reads it
(:func:`load`) and adds gate 5 from the paper record (:func:`block`).  Rebuild after a re-baseline, a new recorded
run of a book, or a change in a family's trial count:

    python -m nse_engine.golden_gates build      # research machine; writes config/golden_gates.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RF = 0.065
DOC_PATH = "config/golden_gates.json"
FAMILIES_PATH = "config/research_families.json"
GATES = {"dsr": 0.95, "net_sharpe": 0.8, "walk_forward": 1.0, "era_2008_12": 0.2}
VETOES = {"maxdd": -0.30, "calmar": 0.5, "cvar_vs_nifty": 1.5, "monthly_skew": -1.0, "capacity_drop": 0.1}
MC_SESSIONS, MC_LOW, MC_HIGH = 60, 5.0, 95.0
#: A series must start by then to stand for 2008-25.
COVER_FROM = "2008-01-31"
D2_HAIRCUT = 0.03

#: Each book's recorded series (run ids in data/nse_engine/runs; walk-forwards in data/nse_engine).  The candidate
#: has no walk-forward of its own: its family's R4b stands in, as for its MC band.
EQUITY_RUNS = {"deployed": "20261017T110545845367Z_2cb1c053",       # deployed + drawdown rule, cost model 4
               "candidate": "20261116T095349906391Z_baeaaac7",      # candidate + drawdown rule (GTR4)
               "e4": "20261116T095358132733Z_b837e415"}             # e4 + drawdown rule (GTR4)
WALK_FORWARDS = {"deployed": "data/nse_engine/wf_stitched_r4b.json", "candidate": "data/nse_engine/wf_stitched_r4b.json",
                 "e4": "data/nse_engine/wf_stitched_e4_2008_25.json"}       # GTR11: 2008-16 (gtr1w) + r12b4
#: The sleeve books trade the deployed rules plus a sleeve (their deployments); the levered futures book is read
#: from its real-contract run (2017-25 only).
SLEEVE_BOOKS = {"bonds": ("gilt", 0.25), "mcx": ("mcx", 0.25), "a1": ("a1", None)}
SCORECARD = "data/nse_engine/scorecard/2026-10-10_{book}.json"
#: The fund size Sharpe must hold at (owner, 16 Nov 2026, GTR10; it was Rs 30 lakh).
CAPACITY_INR = 10_000_000
#: Books the owner lets run above the 1.5x cap, with the decision (tracker GTR8).  Paper only: live leverage still
#: goes through the funding gate, after RA1 and the owner's dated approval.
CAP_EXCEPTIONS = {"futures": "your exception of 16 Nov 2026 (GTR8), until the book retires after RA1 (GTR10)"}


# ── the arithmetic (pure) ─────────────────────────────────────────────────

def excess_sharpe(r: pd.Series) -> float:
    ex = pd.Series(r, dtype="float64").dropna() - RF / 252
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * math.sqrt(252)) if sd > 0 else float("nan")


def card(r: pd.Series) -> Dict[str, float]:
    """CAGR, excess Sharpe, MaxDD (the start counts as a peak), Calmar, the 2008-12 Sharpe and the excess volatility."""
    r = pd.Series(r, dtype="float64").dropna()
    eq = (1 + r).cumprod()
    cagr = float(eq.iloc[-1] ** (252 / len(r)) - 1)
    mdd = float((eq / np.maximum(eq.cummax(), 1.0) - 1).min())
    era = r["2008-01-01":"2012-12-31"]
    return {"from": str(r.index[0].date()), "to": str(r.index[-1].date()), "cagr": cagr, "sharpe": excess_sharpe(r),
            "maxdd": mdd, "calmar": cagr / -mdd if mdd < 0 else float("nan"),
            "era_2008_12": excess_sharpe(era) if len(era) > 250 else None,
            "vol": float((r - RF / 252).std(ddof=1) * math.sqrt(252))}


def deflate(r: pd.Series, n_trials: float, sr_variance: Optional[float]) -> Dict[str, float]:
    from nse_engine.validation import dsr as D

    d = D.deflated_sharpe(pd.Series(r, dtype="float64").dropna(), n_trials=n_trials, rf_annual=RF,
                          sr_variance=sr_variance)
    return {"dsr": float(d["dsr"]), "sharpe": float(d["sr_annual"]), "sr0": float(d["sr0_annual"]),
            "deflated": float(d["sr_annual"] - d["sr0_annual"])}


def levered(r: pd.Series, leverage: float, carry: float = RF + 0.005) -> pd.Series:
    """The book at ``leverage`` through futures: the extra exposure costs about the liquid rate plus 0.5% a year of
    roll costs."""
    return r if leverage <= 1.0 else leverage * r - (leverage - 1.0) * carry / 252


def vetoes(r: pd.Series, nifty: pd.Series, capacity_drop: Optional[float],
           indices: Optional[np.ndarray] = None) -> Dict[str, Dict[str, Any]]:
    """Each tail veto: its value, 90% block-bootstrap range (the diversifier gate's paired resamples), its bar and
    whether it vetoes."""
    r = pd.Series(r, dtype="float64").dropna()
    a = r.to_numpy()
    b = pd.Series(nifty, dtype="float64").reindex(r.index).fillna(0.0).to_numpy()
    if indices is None:
        from nse_engine import diversifier_gate as dg

        indices = dg._boot_indices(len(a))
    A, B = a[indices], b[indices]
    eq = np.cumprod(1 + A, axis=1)
    md = (eq / np.maximum.accumulate(np.maximum(eq, 1.0), axis=1) - 1).min(axis=1)
    cal = (eq[:, -1] ** (252 / A.shape[1]) - 1) / -md
    k = max(1, int(round(0.05 * len(a))))
    cv = np.sort(A, axis=1)[:, :k].mean(1) / np.sort(B, axis=1)[:, :k].mean(1)
    c = card(r)
    point_cv = float(np.sort(a)[:k].mean() / np.sort(b)[:k].mean())
    mo = ((1 + r).resample("ME").prod() - 1).to_numpy()
    skew = float(((mo - mo.mean()) ** 3).mean() / mo.std() ** 3) if mo.std() > 0 else 0.0
    rng = lambda v: [float(np.percentile(v, 5)), float(np.percentile(v, 95))]  # noqa: E731
    return {"maxdd": {"value": c["maxdd"], "range": rng(md), "bar": VETOES["maxdd"], "veto": c["maxdd"] < VETOES["maxdd"]},
            "calmar": {"value": c["calmar"], "range": rng(cal), "bar": VETOES["calmar"],
                       "veto": c["calmar"] < VETOES["calmar"]},
            "cvar_vs_nifty": {"value": point_cv, "range": rng(cv), "bar": VETOES["cvar_vs_nifty"],
                              "veto": point_cv > VETOES["cvar_vs_nifty"]},
            "monthly_skew": {"value": skew, "range": None, "bar": VETOES["monthly_skew"],
                             "veto": skew < VETOES["monthly_skew"]},
            "capacity_drop": {"value": capacity_drop, "range": None, "bar": VETOES["capacity_drop"],
                              "veto": bool(capacity_drop is not None and capacity_drop > VETOES["capacity_drop"])}}


def assess(fixed_raw: pd.Series, fixed_as_run: pd.Series, wf_as_run: Optional[pd.Series], nifty: pd.Series,
           n_trials: float, sr_variance: Optional[float], impact_drag: Optional[float],
           book_leverage: float = 1.0) -> Dict[str, Any]:
    """Gates 1-4, the vetoes and the leverage earned, for one book.  ``book_leverage``: what the paper book already
    runs at (a book levered before GTR0 is reported against the cap, never resized here)."""
    from nse_engine import funding_gate as fg

    fixed = card(fixed_as_run)
    wf = card(wf_as_run) if wf_as_run is not None and len(wf_as_run.dropna()) else None
    d_in, d_live = deflate(fixed_raw, n_trials, sr_variance), deflate(fixed_as_run, n_trials, sr_variance)
    # the gates are on 2008-25: a series that starts later (skipping 2008 flatters Sharpe by about 0.3) cannot pass
    whole = fixed["from"] <= COVER_FROM
    wf_whole = bool(wf and wf["from"] <= COVER_FROM)
    lev = fg.implied_leverage(d_live["deflated"], fixed["vol"], d_in["dsr"] if whole else float("nan"))
    gates = {"dsr": {"value": d_in["dsr"], "live": d_live["dsr"], "from": fixed["from"],
                     "pass": whole and d_in["dsr"] >= GATES["dsr"]},
             "net_sharpe": {"value": fixed["sharpe"], "from": fixed["from"],
                            "pass": whole and fixed["sharpe"] >= GATES["net_sharpe"]},
             "walk_forward": {"value": wf["sharpe"] if wf else None, "from": wf["from"] if wf else None,
                              "pass": wf_whole and wf["sharpe"] >= GATES["walk_forward"]},
             "era_2008_12": {"value": fixed["era_2008_12"], "walk_forward": wf["era_2008_12"] if wf else None,
                             "pass": bool(fixed["era_2008_12"] is not None and fixed["era_2008_12"] >= GATES["era_2008_12"])}}
    drop = impact_drag / fixed["vol"] if impact_drag is not None and fixed["vol"] > 0 else None
    # the vetoes judge the book at the leverage it would run at: a series already levered (futures) as it is
    at_risk = fixed_as_run if book_leverage > 1.0 else levered(fixed_as_run, lev)
    return {"gates": gates, "vetoes": vetoes(at_risk, nifty, drop),
            "leverage": {"earned": lev, "half_kelly": 0.5 * d_live["deflated"] / fixed["vol"] if fixed["vol"] > 0 else None,
                         "deflated_sharpe": d_live["deflated"], "book_runs_at": book_leverage,
                         "above_cap": book_leverage > fg.CAP},
            "fixed": fixed, "walk_forward": wf}


def gate5(position: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Gate 5 from the paper record (``books.mc_positions``): not due before 60 sessions; then the return since the
    start inside the 5th-95th percentile of the book's band."""
    if not position:
        return {"due": False, "pass": False, "text": "no band"}
    n, p = int(position["sessions"]), float(position["return_pct"])
    if n < MC_SESSIONS:
        return {"due": False, "pass": False, "text": f"not due ({n} of {MC_SESSIONS}; now {p:.0f}th pct)"}
    ok = MC_LOW <= p <= MC_HIGH
    return {"due": True, "pass": ok, "text": f"{p:.0f}th pct at {n} sessions"}


# ── the research machine: build the file ──────────────────────────────────

def n_all_families(path: str = FAMILIES_PATH) -> int:
    fam = json.loads(Path(path).read_text())["families"]
    return int(sum(int(f.get("n") or 0) for f in fam.values()))


def _series(book: str) -> Dict[str, Any]:
    """(in-sample fixed, as-run fixed, as-run walk-forward, its leverage) for one paper book, 2008-25."""
    from nse_engine import bond_stream as bs
    from nse_engine import diversifier_gate as dg
    from nse_engine import mc_bands as mb

    def as_run(base: Any) -> pd.Series:
        s = base.returns.index
        return dg._live_book(base, pd.Series(0.0, index=s), s).dropna()

    if book in EQUITY_RUNS:
        base = bs.backtest_base(EQUITY_RUNS[book])
        wf = bs.walk_forward_base(WALK_FORWARDS[book])
        return {"raw": base.returns.dropna(), "as_run": as_run(base), "wf": as_run(wf), "leverage": 1.0}
    if book == "futures":
        r = pd.read_csv(mb.L7A_RUN, index_col=0).squeeze("columns").astype("float64")
        r.index = pd.DatetimeIndex(r.index)
        r = r.sort_index()["2008-01-01":"2025-12-31"]
        live = r - D2_HAIRCUT / 252                       # real contracts on walk-forward positions; D2 on the book
        return {"raw": r, "as_run": live, "wf": live, "leverage": 1.75}
    leg, w = SLEEVE_BOOKS[book]
    base = bs.backtest_base(EQUITY_RUNS["deployed"])
    wf = bs.walk_forward_base(WALK_FORWARDS["deployed"])
    out = {"leverage": 1.0}
    for key, equity in (("raw", base.returns.dropna()), ("as_run", as_run(base)), ("wf", as_run(wf))):
        idx = equity.index
        br = bs.bond_returns(idx, bs.BondConfig())
        if leg == "gilt":
            out[key] = dg.combine(equity, br["gilt"], w, "sleeve")
        elif leg == "mcx":
            m = pd.Series(json.loads(Path(mb.MCX_JSON).read_text())["reported"]["GT5-B"]["returns"], dtype="float64")
            m.index = pd.DatetimeIndex(m.index)
            mcx = dg.carry_to(m.sort_index(), idx).reindex(idx).fillna(0.0) + br["liquid"]
            out[key] = dg.combine(equity, mcx, w, "sleeve")
        else:                                             # a1: the A1G overlay at its 10% loss cap, as mc_bands
            a1 = dg._read(Path(mb.A1G_RUN) / "returns.csv")
            sessions = idx[idx >= a1.index[0]]
            a1_eq = (1 + a1).cumprod() * dg.A1_CAPITAL
            wt, _ = dg.capped_weights(sessions, 0.10, pd.read_csv(Path(mb.A1G_RUN) / "cycles.csv"), a1_eq)
            out[key] = dg.combine(equity[sessions], dg.carry_to(a1, sessions).reindex(sessions).fillna(0.0), wt,
                                  "overlay")
    return out


def _impact_drag(book: str) -> Optional[float]:
    """The impact drag at ``CAPACITY_INR`` from the book's scorecard (the sleeve books trade the deployed rules; the
    whole capital is counted as equity, which overstates their drag)."""
    name = book if book in EQUITY_RUNS else "deployed"
    try:
        doc = json.loads(Path(SCORECARD.format(book=name)).read_text())
        row = min(doc["capacity"]["by_capital"], key=lambda x: abs(x["capital_inr"] - CAPACITY_INR))
        if row["capital_inr"] != CAPACITY_INR:
            raise KeyError(f"no capacity row at Rs {CAPACITY_INR:,}")
        return float(row["impact_drag_annual"])
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("%s: no capacity figure (%s)", book, exc)
        return None


def build(path: str = DOC_PATH, books: Sequence[str] = ("deployed", "candidate", "e4", "futures", "bonds", "mcx", "a1")
          ) -> Dict[str, Any]:
    """Every book's gates 1-4, vetoes and leverage from the recorded series, written to ``path``."""
    from nse_engine import bond_stream as bs
    from nse_engine import books as bk
    from nse_engine.deployment import load_deployment
    from nse_engine.validation import dsr as D
    from nse_engine.validation.trials import TrialRegistry

    reg = TrialRegistry("data/nse_engine/runs")
    run = bk.latest_run(reg.runs_dir, load_deployment().engine.config_hash())
    matrix = reg.returns_matrix(data_hash=run.get("data_hash"), window=(run["start"], run["end"]),
                                cost_model=int(run.get("cost_model") or 4))
    ex = matrix - D.daily_rf(RF)
    variance = float((ex.mean() / ex.std(ddof=1)).var(ddof=1))
    n = n_all_families()
    doc: Dict[str, Any] = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "n_trials": n,
                           "sr_variance_daily": variance, "core_matrix": int(matrix.shape[1]), "rf": RF,
                           "basis": "excess Sharpe over 6.5%; as-run = D2's 3% on invested, idle cash at 0; "
                                    "2008-25 unless stated", "books": {}}
    nifty = None
    for book in books:
        s = _series(book)
        if nifty is None or not s["as_run"].index.isin(nifty.index).all():
            nifty = bs.nifty_tri(s["as_run"].index)
        doc["books"][book] = assess(s["raw"], s["as_run"], s["wf"], nifty, n, variance, _impact_drag(book),
                                    s["leverage"])
        logger.info("golden gates: %s done", book)
    Path(path).write_text(json.dumps(doc, indent=1, default=float) + "\n")
    return doc


def load(path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    from nse_engine.deployment import REPO_ROOT

    p = Path(path) if path else Path(REPO_ROOT) / DOC_PATH
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


# ── the Saturday email ────────────────────────────────────────────────────

def _mark(ok: bool) -> str:
    return "pass" if ok else "fail"


def _num(x: Any, fmt: str = "{:.2f}") -> str:
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else fmt.format(x)


def block(doc: Optional[Dict[str, Any]], positions: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The golden-gates section: one row per book in the file, gate 5 from ``positions`` (``books.mc_positions``);
    None without a file.  ``leverage`` maps each book to the leverage it has earned (above 1x only once every gate
    the funding gate waits for has passed)."""
    if not doc or not doc.get("books"):
        return None
    head = ["Book", "1 DSR (>= 0.95)", "2 Net Sharpe (>= 0.8)", "3 Walk-forward (>= 1.0)", "4 2008-12 (>= 0.2)",
            "5 MC band at 60", "Vetoes", "Leverage earned"]
    rows, lines, earned, flags = [], [], {}, []
    for name, b in doc["books"].items():
        g, v, lev = b["gates"], b["vetoes"], b["leverage"]
        g5 = gate5(positions.get(name))
        hit = [k.replace("_", " ") for k, x in v.items() if x["veto"]]
        wf_from = g["walk_forward"].get("from")
        since = lambda x: f" (from {x[:4]})" if x and x[:4] != "2008" else ""          # noqa: E731
        cells = [name,
                 f"{_num(g['dsr']['value'])} {_mark(g['dsr']['pass'])}{since(g['dsr'].get('from'))}",
                 f"{_num(g['net_sharpe']['value'])} {_mark(g['net_sharpe']['pass'])}{since(g['net_sharpe'].get('from'))}",
                 f"{_num(g['walk_forward']['value'])} {_mark(g['walk_forward']['pass'])}{since(wf_from)}",
                 f"{_num(g['era_2008_12']['value'], '{:+.2f}')} {_mark(g['era_2008_12']['pass'])}",
                 g5["text"] + (f" {_mark(g5['pass'])}" if g5["due"] else ""),
                 "VETO: " + ", ".join(hit) if hit else "clear",
                 f"{lev['earned']:.2f}x"]
        rows.append(cells)
        earned[name] = float(lev["earned"])
        all_pass = all(x["pass"] for x in g.values()) and g5["pass"] and not hit
        lines.append(f"{name}: " + ("every gate passes" if all_pass else
                                    "fails " + ", ".join(k for k, x in (("1", g["dsr"]), ("2", g["net_sharpe"]),
                                                                        ("3", g["walk_forward"]),
                                                                        ("4", g["era_2008_12"])) if not x["pass"])
                                    + ("" if g5["pass"] else (", 5" if g5["due"] else ", 5 not due")))
                     + (f"; vetoed on {', '.join(hit)}" if hit else "") + f"; leverage {lev['earned']:.2f}x")
        if lev.get("above_cap") and name in CAP_EXCEPTIONS:
            flags.append(f"{name} runs at {lev['book_runs_at']:.2f}x in paper, above the 1.5x cap by "
                         f"{CAP_EXCEPTIONS[name]}; live leverage still needs the funding gate.")
        elif lev.get("above_cap"):
            flags.append(f"{name} runs at {lev['book_runs_at']:.2f}x in paper, above GTR0's 1.5x cap and without the "
                         "funding gate (it predates GTR0): your call.")
    built = str(doc.get("created_at", ""))[:10]
    note = (f"Built {built} from recorded series (research machine), {doc.get('n_trials')} trials counted across "
            "families; Sharpe in excess of 6.5%, as-run (D2's haircut, idle cash at 0), 2008-25 unless stated. "
            "Leverage needs every gate; any leverage goes through the funding gate's request first.")
    table = ("<table style='border-collapse:collapse;font-size:12px;'><tr>"
             + "".join(f"<th style='border:1px solid #ddd;padding:4px 6px;background:#f3f4f6;'>{escape(h)}</th>"
                       for h in head) + "</tr>"
             + "".join("<tr>" + "".join(f"<td style='border:1px solid #ddd;padding:4px 6px;'>{escape(c)}</td>"
                                        for c in r) + "</tr>" for r in rows) + "</table>")
    html = ("<h3>Golden-target gates (GTR6)</h3>" + table
            + "".join(f"<p style='color:#b45309;'>{escape(f)}</p>" for f in flags)
            + f"<p style='color:#666;font-size:12px;'>{escape(note)}</p>")
    return {"html": html, "head": "Golden-target gates (GTR6)", "lines": lines + flags, "leverage": earned}


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("cmd", choices=("build", "show"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.cmd == "build":
        doc = build()
    else:
        doc = load()
        if not doc:
            print("no config/golden_gates.json: python -m nse_engine.golden_gates build")
            return 1
    out = block(doc, {})
    print("\n".join(out["lines"]) if out else "nothing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
