"""
Paper books: comparison with the deployed book, forward-gate status, the
register and the promotion review (tracker V4).

Every ``config/nse_engine_<book>.json`` is a book (D5): ``deployed`` trades
the approved configuration, the others are trials under the forward gate
(V3, decision U19).  This module answers, for all of them at once and from
the numbers the gate itself uses:

* which trial has done better than the deployed book on paper over their
  common sessions, and whether the gap is more than noise (the t statistic
  of the daily return differences);
* which trial has cleared the three forward-gate checks (60 sessions beside
  the deployed book, its own G4 PASS, walk-forward OOS Sharpe within 0.05);
* the register ``docs/books_register.csv``: one row per book with its
  configuration, backtest and walk-forward scores, PBO / DSR, paper scores
  and a one-line summary;
* the promotion review a cleared trial gets, for the owner to read before
  ``run_nse_engine promote``.  Nothing here promotes.

Paper data comes from each book's Neon schema (``tools.books_report``, the
Saturday job); backtest scores from the run registry, which lives only on
the research machine, so ``register`` runs there and the CSV is committed.
The Saturday job then reads the backtest columns from that CSV.

    python -m nse_engine.books register            # rebuild the backtest columns from the registry
    python -m nse_engine.books review --book e4    # the promotion review, from the register
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from nse_engine import forward_gate as fg
from nse_engine.config import EngineConfig
from nse_engine.deployment import REPO_ROOT, Deployment, load_deployment

logger = logging.getLogger(__name__)

REGISTER_PATH = "docs/books_register.csv"
DEPLOYED = "deployed"
RF = EngineConfig().risk_free_annual
PERIODS = 252
IST = timezone(timedelta(hours=5, minutes=30))
PASS, FAIL, PENDING, NA = "PASS", "FAIL", "PENDING", "n/a"
#: |t| of the mean daily return difference beyond which a paper gap is unlikely to be noise.
T_SIGNIFICANT = 2.0

STATIC_COLUMNS = ["scored_run_id", "data_hash", "cost_model", "bt_cagr", "bt_sharpe", "bt_max_dd", "bt_calmar",
                  "bt_turnover", "bt_cost_drag", "wf_oos_sharpe_2017_25", "pbo", "pbo_n", "dsr"]
PAPER_COLUMNS = ["paper_sessions", "paper_return", "paper_alpha_nifty50", "paper_sharpe", "paper_max_dd", "paper_g4",
                 "paper_as_of", "vs_deployed_pts", "vs_deployed_t", "gate_sessions", "gate_g4", "gate_wf", "gate_cleared"]
REGISTER_COLUMNS = (["book", "status", "fingerprint", "config_hash", "description", "paper_start", "source_run_id"]
                    + STATIC_COLUMNS + PAPER_COLUMNS + ["summary", "updated_at"])


# ── books ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Book:
    name: str
    path: Path
    deployment: Deployment

    @property
    def is_deployed(self) -> bool:
        return self.name == DEPLOYED

    @property
    def schema(self) -> Optional[str]:
        """The book's Neon schema: the deployed book lives in the default schema."""
        return None if self.is_deployed else self.name

    @property
    def config_hash(self) -> str:
        return self.deployment.engine.config_hash()

    @property
    def fingerprint(self) -> str:
        return self.config_hash[:8]

    @property
    def label(self) -> str:
        return f"{self.name} {self.fingerprint}"


def discover_books(root: Path = REPO_ROOT) -> List[Book]:
    """The books from ``config/nse_engine_<book>.json``: deployed first, then by paper start."""
    books = []
    for path in sorted(Path(root).glob("config/nse_engine_*.json")):
        name = path.stem[len("nse_engine_"):]
        try:
            dep = load_deployment(path)
        except Exception as exc:                          # noqa: BLE001 - a broken file is not a book
            logger.warning("book %s skipped: %s", path.name, exc)
            continue
        books.append(Book(name, path, dep))
    books.sort(key=lambda b: (not b.is_deployed, b.deployment.paper_start_date, b.name))
    return books


def deployed_book(books: Sequence[Book]) -> Optional[Book]:
    return next((b for b in books if b.is_deployed), None)


# ── backtest scores from the registry ────────────────────────────

def latest_run(runs_dir: Path, config_hash: str, window: Sequence[str] = fg.VALIDATION_WINDOW) -> Optional[Dict[str, Any]]:
    """The registry's like-for-like run of a configuration: same window, newest cost model, then newest."""
    from nse_engine.validation.trials import LEGACY_COST_MODEL

    best: Optional[Tuple[Tuple[int, str], Dict[str, Any]]] = None
    for mf in Path(runs_dir).glob("*/manifest.json"):
        try:
            m = json.loads(mf.read_text())
        except (OSError, ValueError):
            continue
        if m.get("config_hash") != config_hash or (m.get("start"), m.get("end")) != tuple(window):
            continue
        if not (mf.parent / "returns.csv").exists():
            continue
        m["_dir"] = str(mf.parent)
        key = (int(m.get("cost_model") or LEGACY_COST_MODEL), str(m.get("created_at") or ""))
        if best is None or key > best[0]:
            best = (key, m)
    return best[1] if best else None


def _validation(run_dir: Path) -> Optional[Dict[str, Any]]:
    path = run_dir / "validation.json"
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except (OSError, ValueError):
        return None


def static_row(book: Book, runs_dir: Path) -> Dict[str, Any]:
    """Backtest, walk-forward OOS, PBO and DSR of the book's configuration, from the registry."""
    from nse_engine.validation.trials import TrialRegistry

    row: Dict[str, Any] = {c: np.nan for c in STATIC_COLUMNS}
    run = latest_run(runs_dir, book.config_hash)
    if run is None:
        return row
    m = run.get("metrics") or {}
    row.update(scored_run_id=run["run_id"], data_hash=run.get("data_hash"), cost_model=run.get("cost_model"),
               bt_cagr=m.get("cagr"), bt_sharpe=m.get("sharpe"), bt_max_dd=m.get("max_drawdown"),
               bt_calmar=m.get("calmar"), bt_turnover=m.get("annual_turnover"), bt_cost_drag=m.get("cost_drag"))
    returns = TrialRegistry(runs_dir).load_returns(run["run_id"])
    row["wf_oos_sharpe_2017_25"] = fg.oos_sharpe(returns, fg.WF_OOS_WINDOW, RF)
    validation = _validation(Path(run["_dir"]))
    if validation is None and book.deployment.source_run_id:
        validation = _validation(Path(runs_dir) / book.deployment.source_run_id)
    if validation:
        pbo = validation.get("pbo")
        row.update(pbo=pbo.get("pbo") if isinstance(pbo, dict) else pbo, pbo_n=validation.get("n_configurations"),
                   dsr=(validation.get("dsr") or {}).get("dsr"))
    return row


# ── paper data ───────────────────────────────────────────────────

@dataclass
class PaperBook:
    """One book's paper record: equity by session, session dates, its stored G4 report."""

    name: str
    equity: pd.Series
    sessions: pd.DatetimeIndex
    gate: Optional[Dict[str, Any]] = None

    @property
    def started(self) -> bool:
        return len(self.equity) > 0

    @property
    def last(self) -> Optional[pd.Timestamp]:
        return self.equity.index[-1] if self.started else None


def paper_book_from_frames(name: str, snapshots: Optional[pd.DataFrame], sessions: Optional[pd.DataFrame],
                           state: Optional[Dict[str, str]]) -> PaperBook:
    """From the book's Neon tables (``PaperCloudSync.read_snapshots / read_sessions / read_state``)."""
    equity = pd.Series(dtype="float64")
    if snapshots is not None and not snapshots.empty and {"date", "equity"} <= set(snapshots.columns):
        idx = pd.DatetimeIndex(pd.to_datetime(snapshots["date"].astype(str))).normalize()
        s = pd.Series(snapshots["equity"].astype(float).to_numpy(), index=idx)
        equity = s[~s.index.duplicated(keep="last")].sort_index().dropna()
    return PaperBook(name, equity, fg.session_dates(sessions), fg.stored_gate(state or {}))


def _daily(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def window_metrics(equity: pd.Series, benchmark: Optional[pd.Series] = None) -> Dict[str, Any]:
    """Return, volatility, excess Sharpe, MaxDD and alpha against ``benchmark`` over the series' span."""
    from nse_engine.validation.dsr import excess_sharpe

    e = pd.Series(equity, dtype="float64").dropna()
    out: Dict[str, Any] = {"sessions": int(len(e)), "start": None, "end": None, "return": np.nan, "vol": np.nan,
                           "sharpe": np.nan, "max_dd": np.nan, "benchmark_return": np.nan, "alpha": np.nan}
    if len(e) == 0:
        return out
    out["start"], out["end"] = e.index[0].date(), e.index[-1].date()
    out["return"] = float(e.iloc[-1] / e.iloc[0] - 1.0)
    out["max_dd"] = float((e / e.cummax() - 1.0).min())
    r = _daily(e)
    if len(r) >= 2 and r.std(ddof=1) > 0:
        out["vol"] = float(r.std(ddof=1) * math.sqrt(PERIODS))
        out["sharpe"] = float(excess_sharpe(r, RF, PERIODS))
    if benchmark is not None and len(benchmark):
        b0, b1 = benchmark.get(e.index[0]), benchmark.get(e.index[-1])
        if b0 is not None and b1 is not None and np.isfinite(b0) and np.isfinite(b1) and b0 > 0:
            out["benchmark_return"] = float(b1 / b0 - 1.0)
            out["alpha"] = out["return"] - out["benchmark_return"]
    return out


def pair_comparison(trial: pd.Series, base: pd.Series, benchmark: Optional[pd.Series] = None) -> Dict[str, Any]:
    """A trial against the deployed book over their common sessions."""
    common = trial.dropna().index.intersection(base.dropna().index)
    t, b = trial[common], base[common]
    out: Dict[str, Any] = {"sessions": int(len(common)), "start": common[0].date() if len(common) else None,
                           "end": common[-1].date() if len(common) else None,
                           "trial": window_metrics(t, benchmark), "base": window_metrics(b, benchmark),
                           "diff_pts": np.nan, "t_stat": np.nan, "tracking_error": np.nan, "correlation": np.nan}
    if len(common) < 2:
        out["verdict"] = "fewer than 2 common sessions"
        return out
    out["diff_pts"] = out["trial"]["return"] - out["base"]["return"]
    d = _daily(t) - _daily(b)
    if len(d) >= 2 and d.std(ddof=1) > 0:
        out["t_stat"] = float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d))))
        out["tracking_error"] = float(d.std(ddof=1) * math.sqrt(PERIODS))
        out["correlation"] = float(_daily(t).corr(_daily(b)))
    out["verdict"] = verdict_text(out)
    return out


def verdict_text(c: Dict[str, Any]) -> str:
    diff, t, n = c["diff_pts"], c["t_stat"], c["sessions"]
    lead = "ahead of" if diff > 0 else "behind" if diff < 0 else "level with"
    if t != t:
        strength = "too few sessions to judge"
    else:
        strength = f"t {t:+.1f}, " + ("unlikely to be noise" if abs(t) >= T_SIGNIFICANT else "within noise")
    return f"{lead} the deployed book by {abs(diff):.2%} over {n} common sessions ({strength})"


# ── forward gate ─────────────────────────────────────────────────

def gate_status(trial: PaperBook, base: PaperBook, trial_wf: float, base_wf: float) -> Dict[str, Any]:
    """The three forward-gate checks as PASS / FAIL / PENDING, from ``nse_engine.forward_gate``.

    A check that time will settle (sessions, a G4 still gathering data) is
    PENDING; the walk-forward figures come from the register's recorded runs.
    """
    from nse_engine import paper_gate

    checks = []
    name, ok, detail = fg.sessions_check(trial.sessions, base.sessions)
    checks.append({"name": name, "status": PASS if ok else PENDING, "detail": detail})
    name, ok, detail = fg.gate_check(trial.gate, trial.last)
    verdict = (trial.gate or {}).get("verdict")
    checks.append({"name": name, "status": PASS if ok else FAIL if verdict == paper_gate.FAIL else PENDING,
                   "detail": detail})
    if _finite(trial_wf) and _finite(base_wf):
        name, ok, detail = fg.wf_check(float(trial_wf), float(base_wf))
        checks.append({"name": name, "status": PASS if ok else FAIL, "detail": detail + ", from the recorded runs"})
    else:
        checks.append({"name": "walk-forward OOS Sharpe", "status": PENDING,
                       "detail": "no recorded same-window run in the register for one of the two books"})
    return {"checks": checks, "cleared": all(c["status"] == PASS for c in checks)}


def _finite(x: Any) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


# ── register ─────────────────────────────────────────────────────

def read_register(path: Path = REPO_ROOT / REGISTER_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=REGISTER_COLUMNS)
    df = pd.read_csv(path, dtype={"config_hash": str, "fingerprint": str})
    for c in REGISTER_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df[REGISTER_COLUMNS]


def write_register(df: pd.DataFrame, path: Path = REPO_ROOT / REGISTER_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6g")
    return path


def _description(book: Book) -> str:
    if book.deployment.description:
        return book.deployment.description
    first = (book.deployment.notes or "").split(". ")[0].strip()
    return first[:160] if first else book.label


def summary_line(row: Dict[str, Any], today: Optional[date] = None) -> str:
    today = today or datetime.now(IST).date()
    parts = []
    if _finite(row.get("bt_sharpe")):
        s = (f"Backtest 2013-25: CAGR {row['bt_cagr']:.1%}, Sharpe {row['bt_sharpe']:.2f}, "
             f"MaxDD {row['bt_max_dd']:.1%}")
        if _finite(row.get("wf_oos_sharpe_2017_25")):
            s += f", walk-forward OOS 2017-25 Sharpe {row['wf_oos_sharpe_2017_25']:.2f}"
        parts.append(s)
    else:
        parts.append("Backtest: no recorded same-window run")
    n = row.get("paper_sessions")
    if _finite(n) and int(n) > 0:
        s = f"paper {int(n)} sessions since {row.get('paper_start')}: {row['paper_return']:+.1%}"
        if _finite(row.get("paper_alpha_nifty50")):
            s += f" ({row['paper_alpha_nifty50']:+.1%} vs NIFTY 50)"
        s += f", G4 {row.get('paper_g4') or 'no report'}"
        parts.append(s)
    else:
        start = str(row.get("paper_start") or "")
        parts.append(f"paper starts {start}" if start > today.isoformat() else "paper: no sessions yet")
    if row.get("book") != DEPLOYED:
        if row.get("gate_cleared") is True or str(row.get("gate_cleared")) == "True":
            parts.append("forward gate: all three checks cleared, awaiting your review")
        elif isinstance(row.get("gate_sessions"), str) and row["gate_sessions"] != NA:
            parts.append(f"forward gate: sessions {row['gate_sessions']}, G4 {row.get('gate_g4')}, "
                         f"walk-forward {row.get('gate_wf')}")
    return "; ".join(parts)


def build_register(books: Sequence[Book], runs_dir: Optional[Path] = None,
                   paper: Optional[Dict[str, PaperBook]] = None, benchmark: Optional[pd.Series] = None,
                   existing: Optional[pd.DataFrame] = None, today: Optional[date] = None) -> pd.DataFrame:
    """One row per book.  Backtest columns from ``runs_dir`` when it exists, else kept from ``existing``
    (the committed CSV); paper columns from ``paper`` when given, else kept from ``existing``."""
    today = today or datetime.now(IST).date()
    old = existing.set_index("book") if existing is not None and len(existing) else pd.DataFrame()
    base = paper.get(DEPLOYED) if paper else None
    rows: Dict[str, Dict[str, Any]] = {}
    for b in books:
        row: Dict[str, Any] = {"book": b.name, "status": b.deployment.status, "fingerprint": b.fingerprint,
                               "config_hash": b.config_hash, "description": _description(b),
                               "paper_start": b.deployment.paper_start_date.isoformat(),
                               "source_run_id": b.deployment.source_run_id}
        if runs_dir is not None and Path(runs_dir).exists():
            row.update(static_row(b, Path(runs_dir)))
        elif b.name in old.index and str(old.loc[b.name, "config_hash"]) == b.config_hash:
            row.update({c: old.loc[b.name, c] for c in STATIC_COLUMNS})
        else:
            row.update({c: np.nan for c in STATIC_COLUMNS})
        rows[b.name] = row
    base_wf = rows[DEPLOYED].get("wf_oos_sharpe_2017_25") if DEPLOYED in rows else np.nan
    for b in books:
        row = rows[b.name]
        pb = paper.get(b.name) if paper else None
        if pb is not None and pb.started:
            m = window_metrics(pb.equity, benchmark)
            row.update(paper_sessions=m["sessions"], paper_return=m["return"], paper_alpha_nifty50=m["alpha"],
                       paper_sharpe=m["sharpe"], paper_max_dd=m["max_dd"],
                       paper_g4=(pb.gate or {}).get("verdict") or "no report", paper_as_of=m["end"])
            if b.is_deployed:
                row.update(vs_deployed_pts=np.nan, vs_deployed_t=np.nan, gate_sessions=NA, gate_g4=NA, gate_wf=NA,
                           gate_cleared=NA)
            elif base is not None and base.started:
                c = pair_comparison(pb.equity, base.equity, benchmark)
                g = gate_status(pb, base, row.get("wf_oos_sharpe_2017_25"), base_wf)
                row.update(vs_deployed_pts=c["diff_pts"], vs_deployed_t=c["t_stat"],
                           gate_sessions=g["checks"][0]["status"], gate_g4=g["checks"][1]["status"],
                           gate_wf=g["checks"][2]["status"], gate_cleared=g["cleared"])
        elif paper is None and b.name in old.index:
            row.update({c: old.loc[b.name, c] for c in PAPER_COLUMNS})
        else:
            row.update({c: np.nan for c in PAPER_COLUMNS})
        row["summary"] = summary_line(row, today)
        row["updated_at"] = datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")
    return pd.DataFrame(list(rows.values()), columns=REGISTER_COLUMNS)


# ── report ───────────────────────────────────────────────────────

def _pct(x: Any, digits: int = 2, signed: bool = True) -> str:
    return (f"{x:+.{digits}%}" if signed else f"{x:.{digits}%}") if _finite(x) else NA


def _num(x: Any, digits: int = 2, signed: bool = False) -> str:
    return (f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}") if _finite(x) else NA


_TD = "padding:5px 10px;border:1px solid #e5e7eb;"
_COLOURS = {PASS: "#15803d", FAIL: "#dc2626", PENDING: "#b45309", NA: "#6b7280"}


def _html_table(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    head = "".join(f"<th style='{_TD}background:#f3f4f6;text-align:left;'>{c}</th>" for c in columns)
    body = ""
    for r in rows:
        cells = ""
        for v in r:
            colour = _COLOURS.get(str(v), "")
            style = f"{_TD}" + (f"color:{colour};font-weight:bold;" if colour else "")
            cells += f"<td style='{style}'>{v}</td>"
        body += f"<tr>{cells}</tr>"
    return f"<table style='border-collapse:collapse;width:100%;font-size:13px;margin:6px 0 14px;'>" \
           f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _md_table(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def review_sections(trial: str, register: pd.DataFrame, comparison: Optional[Dict[str, Any]],
                    gate: Optional[Dict[str, Any]], as_of: Optional[date] = None) -> List[Dict[str, Any]]:
    """The promotion review as sections (title, table or bullets), rendered by ``render_review``."""
    as_of = as_of or datetime.now(IST).date()
    reg = register.set_index("book")
    if trial not in reg.index or DEPLOYED not in reg.index:
        raise ValueError(f"register has no row for {trial!r} and {DEPLOYED!r}")
    t, d = reg.loc[trial], reg.loc[DEPLOYED]
    sections: List[Dict[str, Any]] = [{"title": f"Promotion review: {trial} {t['fingerprint']} vs deployed "
                                                f"{d['fingerprint']}, as of {as_of}", "intro": t.get("description", "")}]
    if gate:
        sections.append({"title": "Forward gate (V3, U19): all three must pass",
                         "columns": ["Check", "Status", "Detail"],
                         "rows": [[c["name"], c["status"], c["detail"]] for c in gate["checks"]]})
    metrics = [("CAGR 2013-25", "bt_cagr", "pct"), ("Excess Sharpe", "bt_sharpe", "num"),
               ("MaxDD", "bt_max_dd", "pct"), ("Calmar", "bt_calmar", "num"), ("Turnover (x/yr)", "bt_turnover", "num"),
               ("Cost drag (/yr)", "bt_cost_drag", "pct"), ("Walk-forward OOS Sharpe 2017-25", "wf_oos_sharpe_2017_25", "num"),
               ("PBO (n configurations)", "pbo", "pbo"), ("Deflated Sharpe", "dsr", "num3")]

    def fmt(kind: str, row: pd.Series, key: str) -> str:
        v = row.get(key)
        if kind == "pct":
            return _pct(v, 1, signed=False)
        if kind == "num3":
            return _num(v, 3)
        if kind == "pbo":
            return f"{v:.1%} ({int(row['pbo_n'])})" if _finite(v) and _finite(row.get("pbo_n")) else NA
        return _num(v)

    def delta(row_t: pd.Series, row_d: pd.Series, key: str, kind: str) -> str:
        a, b = row_t.get(key), row_d.get(key)
        if not (_finite(a) and _finite(b)):
            return NA
        return f"{(a - b) * 100:+.1f} pts" if kind in ("pct", "pbo") else f"{a - b:+.3f}"

    sections.append({"title": f"Backtest, same window and cost model (runs {d.get('scored_run_id')} / {t.get('scored_run_id')})",
                     "columns": ["Metric", f"deployed {d['fingerprint']}", f"{trial} {t['fingerprint']}", "Difference"],
                     "rows": [[name, fmt(kind, d, key), fmt(kind, t, key), delta(t, d, key, kind)]
                              for name, key, kind in metrics]})
    if comparison and comparison["sessions"] >= 2:
        ct, cb = comparison["trial"], comparison["base"]
        sections.append({"title": f"Paper, {comparison['sessions']} common sessions {comparison['start']} to {comparison['end']}",
                         "columns": ["Metric", "deployed", trial, "Difference"],
                         "rows": [["Return", _pct(cb["return"]), _pct(ct["return"]), f"{comparison['diff_pts'] * 100:+.2f} pts"],
                                  ["Alpha vs NIFTY 50", _pct(cb["alpha"]), _pct(ct["alpha"]), NA],
                                  ["Excess Sharpe (annualised)", _num(cb["sharpe"]), _num(ct["sharpe"]), NA],
                                  ["MaxDD", _pct(cb["max_dd"], signed=False), _pct(ct["max_dd"], signed=False), NA],
                                  ["G4 verdict", d.get("paper_g4") or NA, t.get("paper_g4") or NA, NA],
                                  ["t of daily differences", "", _num(comparison["t_stat"], 1, signed=True), ""],
                                  ["Tracking error vs deployed", "", _pct(comparison["tracking_error"], 1, signed=False), ""],
                                  ["Correlation of daily returns", "", _num(comparison["correlation"]), ""]]})
    sections.append({"title": "Rationale", "bullets": rationale(trial, t, d, comparison, gate)})
    sections.append({"title": "Your decision", "bullets": [
        "Nothing is promoted automatically. The checks above are the forward gate's; the comparison tables are "
        "there for the judgement the gate cannot make (a 60-session Sharpe has a standard error of about 2).",
        f"To see the gate from the research machine: `python -m runners.run_nse_engine promote --check "
        f"--candidate config/nse_engine_{trial}.json --schema {trial}`. Without `--check` it rewrites "
        f"config/nse_engine_deployed.json with this configuration for your review and commit.",
        "What a promotion changes: the deployed book (and the live book, once live) trades the new configuration "
        "from its next session. It does not restart the go-live clock: go-live reads this trial book's own "
        "record of the same configuration while the deployed book's is under 60 sessions, so keep this trial "
        "book running until then. If already live, the live G4 restarts its window at the change, so the old "
        "configuration's weeks never read as tracking error (tracker V5).",
    ]})
    return sections


def rationale(trial: str, t: pd.Series, d: pd.Series, comparison: Optional[Dict[str, Any]],
              gate: Optional[Dict[str, Any]]) -> List[str]:
    out = []
    if gate:
        failing = [c["name"] for c in gate["checks"] if c["status"] != PASS]
        out.append("Forward gate: all three checks pass." if gate["cleared"]
                   else "Forward gate: not cleared yet: " + ", ".join(f"{n} {s}" for n, s in
                                                                     ((c["name"], c["status"]) for c in gate["checks"]) if s != PASS))
    better = []
    for name, key, higher_is_better in (("Sharpe", "bt_sharpe", True), ("CAGR", "bt_cagr", True),
                                        ("MaxDD", "bt_max_dd", True), ("Calmar", "bt_calmar", True)):
        a, b = t.get(key), d.get(key)
        if _finite(a) and _finite(b):
            better.append((name, (a > b) if higher_is_better else (a < b), a - b))
    if better:
        wins = sum(1 for _, w, _ in better if w)
        detail = ", ".join(f"{n} {diff * 100:+.1f} pts" if n in ("CAGR", "MaxDD") else f"{n} {diff:+.3f}"
                           for n, _, diff in better)
        out.append(f"Backtest: {trial} is better on {wins} of {len(better)} ({detail}); MaxDD is a negative number, "
                   f"so a positive difference is a shallower drawdown.")
    a, b = t.get("wf_oos_sharpe_2017_25"), d.get("wf_oos_sharpe_2017_25")
    if _finite(a) and _finite(b):
        out.append(f"Out of sample: walk-forward years 2017-25 Sharpe {a:.3f} vs {b:.3f} ({a - b:+.3f}; the gate allows "
                   f"-{fg.WF_SHARPE_TOLERANCE:.2f}). This is the evidence that outlives the paper sample.")
    if _finite(t.get("pbo")) and _finite(t.get("pbo_n")):
        out.append(f"Overfitting context: PBO {t['pbo']:.1%} over {int(t['pbo_n'])} same-window configurations is a "
                   f"property of the trial set (near-duplicates), reported not gating; deflated Sharpe "
                   f"{_num(t.get('dsr'), 3)}.")
    if comparison and comparison["sessions"] >= 2:
        out.append(f"Paper: {verdict_text(comparison)}. G4 {t.get('paper_g4') or 'no report'}: paper proves the "
                   f"book behaves like its backtest (tracking error, costs, drawdown), not that it earns more.")
        if _finite(comparison["t_stat"]) and abs(comparison["t_stat"]) < T_SIGNIFICANT:
            out.append("The paper gap is within noise: do not read it as evidence either way.")
    return out


def render_review(sections: Sequence[Dict[str, Any]], html: bool = False) -> str:
    parts = []
    for s in sections:
        if html:
            parts.append(f"<h3 style='color:#1a1a2e;margin:18px 0 6px;'>{s['title']}</h3>")
            if s.get("intro"):
                parts.append(f"<p style='color:#444;font-size:13px;margin:0 0 8px;'>{s['intro']}</p>")
            if s.get("columns"):
                parts.append(_html_table(s["columns"], s["rows"]))
            if s.get("bullets"):
                parts.append("<ul style='font-size:13px;color:#333;'>" + "".join(f"<li>{b}</li>" for b in s["bullets"]) + "</ul>")
        else:
            parts.append(f"## {s['title']}\n")
            if s.get("intro"):
                parts.append(s["intro"] + "\n")
            if s.get("columns"):
                parts.append(_md_table(s["columns"], s["rows"]) + "\n")
            if s.get("bullets"):
                parts.append("\n".join(f"- {b}" for b in s["bullets"]) + "\n")
    return "\n".join(parts)


def build_report(books: Sequence[Book], paper: Dict[str, PaperBook], register: pd.DataFrame,
                 benchmark: Optional[pd.Series] = None, as_of: Optional[date] = None) -> Dict[str, Any]:
    """The weekly email: every book against the deployed one, the forward gate per trial, reviews for cleared ones."""
    as_of = as_of or datetime.now(IST).date()
    reg = register.set_index("book")
    base = paper.get(DEPLOYED)
    base_wf = reg.loc[DEPLOYED, "wf_oos_sharpe_2017_25"] if DEPLOYED in reg.index else np.nan
    comparisons: Dict[str, Dict[str, Any]] = {}
    gates: Dict[str, Dict[str, Any]] = {}
    for b in books:
        pb = paper.get(b.name)
        if b.is_deployed or pb is None or not pb.started or base is None or not base.started:
            continue
        comparisons[b.name] = pair_comparison(pb.equity, base.equity, benchmark)
        gates[b.name] = gate_status(pb, base, reg.loc[b.name, "wf_oos_sharpe_2017_25"] if b.name in reg.index else np.nan,
                                    base_wf)
    cleared = [n for n, g in gates.items() if g["cleared"]]
    judged = {n: c for n, c in comparisons.items() if c["sessions"] >= 2}
    if judged:
        ahead = sorted((c["diff_pts"], n) for n, c in judged.items() if c["diff_pts"] > 0)
        headline = ("No trial is ahead of the deployed book yet: " + "; ".join(
            f"{n} {verdict_text(c)}" for n, c in judged.items())) if not ahead else (
            "Ahead of the deployed book: " + "; ".join(f"{n} {verdict_text(judged[n])}" for _, n in reversed(ahead)))
    else:
        headline = "No trial has 2 common sessions with the deployed book yet."
    gate_line = ("Cleared all three forward-gate checks: " + ", ".join(cleared)) if cleared else \
        "No trial has cleared all three forward-gate checks yet."

    def g4(pb: Optional[PaperBook]) -> str:
        return ((pb.gate or {}).get("verdict") or "no report") if pb and pb.started else "not started"

    book_rows = []
    for b in books:
        pb = paper.get(b.name)
        m = window_metrics(pb.equity, benchmark) if pb and pb.started else window_metrics(pd.Series(dtype="float64"))
        sessions = m["sessions"] if m["sessions"] else f"0 (starts {b.deployment.paper_start_date})"
        book_rows.append([b.name, b.fingerprint, b.deployment.status, sessions, _pct(m["return"]), _pct(m["alpha"]),
                          _num(m["sharpe"]), _pct(m["max_dd"], signed=False), g4(pb)])
    pair_rows = [[n, c["sessions"], f"{c['start']} to {c['end']}" if c["start"] else NA, _pct(c["trial"]["return"]),
                  _pct(c["base"]["return"]), f"{c['diff_pts'] * 100:+.2f}" if _finite(c["diff_pts"]) else NA,
                  _num(c["t_stat"], 1, signed=True), _pct(c["tracking_error"], 1, signed=False), c.get("verdict", "")]
                 for n, c in comparisons.items()]
    gate_rows = [[n, c["name"], c["status"], c["detail"]] for n, g in gates.items() for c in g["checks"]]

    html = [f"<html><body style='font-family:Segoe UI,Arial,sans-serif;background:#f9fafb;padding:20px;'>"
            f"<div style='max-width:860px;margin:0 auto;background:#fff;border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.08);overflow:hidden;'>"
            f"<div style='background:#1a1a2e;padding:16px 24px;'><h2 style='margin:0;color:#fff;font-size:18px;'>"
            f"Centurion &mdash; Paper books, week ending {as_of}</h2>"
            f"<p style='margin:4px 0 0;color:#9ca3af;font-size:13px;'>{len(books)} books; the deployed one is the yardstick</p></div>"
            f"<div style='padding:20px 24px;'>",
            f"<div style='border-left:4px solid {'#15803d' if cleared else '#b45309'};background:#f0fdf4;padding:12px 16px;border-radius:4px;'>"
            f"<p style='margin:0;font-size:14px;'><strong>{headline}</strong></p>"
            f"<p style='margin:6px 0 0;font-size:13px;color:#444;'>{gate_line}</p></div>",
            "<h3 style='color:#1a1a2e;margin:18px 0 6px;'>1. Every book, its whole paper record</h3>",
            _html_table(["Book", "Config", "Status", "Sessions", "Return", "Alpha vs NIFTY 50", "Sharpe", "MaxDD", "G4"], book_rows),
            "<h3 style='color:#1a1a2e;margin:18px 0 6px;'>2. Each trial against the deployed book, common sessions only</h3>",
            _html_table(["Trial", "Sessions", "Window", "Trial", "Deployed", "Diff (pts)", "t", "Tracking error", "Reading"], pair_rows)
            if pair_rows else "<p style='font-size:13px;color:#666;'>No trial shares a session with the deployed book yet.</p>",
            "<h3 style='color:#1a1a2e;margin:18px 0 6px;'>3. Forward gate per trial (promotion needs all three)</h3>",
            _html_table(["Trial", "Check", "Status", "Detail"], gate_rows)
            if gate_rows else "<p style='font-size:13px;color:#666;'>No trial has started.</p>"]
    reviews = {}
    for n in cleared:
        sections = review_sections(n, register, comparisons.get(n), gates.get(n), as_of)
        reviews[n] = render_review(sections)
        html.append("<div style='border-top:2px solid #1a1a2e;margin-top:18px;padding-top:6px;'>"
                    + render_review(sections, html=True) + "</div>")
    html.append("<p style='font-size:12px;color:#666;margin-top:18px;'>Alpha is the book's return minus NIFTY 50's over "
                "the same sessions; Sharpe is annualised excess over 6.5%; t is the mean daily return difference over its "
                "standard error (|t| &ge; 2 is unlikely to be noise). Paper proves behaviour (G4), not the edge: a 60-session "
                "Sharpe has a standard error of about 2. The register is attached as books_register.csv.</p>"
                "</div><div style='padding:12px 24px;background:#f3f4f6;font-size:11px;color:#999;text-align:center;'>"
                "Centurion paper books &bull; tracker V4 &bull; auto-generated</div></div></body></html>")
    subject = (("READY FOR YOUR REVIEW: " + ", ".join(cleared) + " | ") if cleared else "") + \
        f"[Centurion Paper] Books week ending {as_of} | " + \
        (f"{len(judged)} trial(s) compared" if judged else "no comparison yet")
    return {"subject": subject, "html": "\n".join(html), "headline": headline, "gate_line": gate_line,
            "cleared": cleared, "comparisons": comparisons, "gates": gates, "reviews": reviews}


# ── CLI ──────────────────────────────────────────────────────────

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Paper books register and promotion review (tracker V4)")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register", help="rebuild the register's backtest columns from the run registry")
    r.add_argument("--runs-dir", default=EngineConfig().runs_dir)
    r.add_argument("--out", default=str(REPO_ROOT / REGISTER_PATH))
    v = sub.add_parser("review", help="print a trial's promotion review from the register")
    v.add_argument("--book", required=True)
    v.add_argument("--register", default=str(REPO_ROOT / REGISTER_PATH))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.cmd == "register":
        books = discover_books()
        df = build_register(books, runs_dir=Path(args.runs_dir), existing=read_register(Path(args.out)))
        path = write_register(df, Path(args.out))
        for _, row in df.iterrows():
            print(f"{row['book']:10s} {row['fingerprint']}  {row['summary']}")
        print(f"written: {path}")
        return 0
    register = read_register(Path(args.register))
    print(render_review(review_sections(args.book, register, None, None)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
