"""Metrics journal (tracker JR1): how each book's headline numbers have moved over time.

``docs/metrics_journal.csv`` is append-only: one row per book per new piece
of research evidence, never edited afterwards.

* ``backtest`` - the book's configuration over 2013-25 in the run registry, a
  row each time its cost model, data hash or results change (a re-baseline),
  with its 2017-25 Sharpe and, when the run was validated, DSR, PBO and the
  execution haircut's expected out-of-sample Sharpe and CAGR;
* ``walk_forward`` - the anchored walk-forward of the book's configuration
  family (``scorecard.WALK_FORWARD_OOS``), a row each time it is re-run.

The run registry lives only on the research machine, so rows are written
there: ``python -m nse_engine.books register`` calls :func:`record` after
every re-baseline, and the rows are committed with the change that moved
them (``git log -p docs/metrics_journal.csv`` says why).  Paper results are
journaled every night in Neon (the daily snapshots); the web page (Trade
Center > Journal) shows both against :func:`targets`.

    python -m nse_engine.journal record          # append what is new (idempotent)
    python -m nse_engine.journal show --book e4
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from nse_engine.config import EngineConfig
from nse_engine.deployment import REPO_ROOT

JOURNAL_PATH = "docs/metrics_journal.csv"
COLUMNS = ["date", "book", "kind", "event", "cost_model", "data_hash", "config_hash", "ref",
           "cagr", "sharpe", "max_dd", "calmar", "sharpe_2017_25", "dsr", "pbo", "pbo_n",
           "exp_sharpe", "exp_cagr", "note"]
_TEXT = {c: str for c in ("event", "data_hash", "config_hash", "ref", "note")}
#: Tracker section 1: CAGR above 25% (a goal; no scorecard rule holds it).
CAGR_TARGET = 0.25
IST = timezone(timedelta(hours=5, minutes=30))
EVENT_CHARS = 90


def _path(path: Optional[Path]) -> Path:
    return Path(path) if path is not None else REPO_ROOT / JOURNAL_PATH


def _ist_date(stamp: Any) -> str:
    """ISO date in IST of a manifest's UTC ``created_at``."""
    t = pd.Timestamp(stamp)
    return (t.tz_localize("UTC") if t.tzinfo is None else t).tz_convert(IST).date().isoformat()


def read_journal(path: Optional[Path] = None) -> pd.DataFrame:
    p = _path(path)
    if not p.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(p, dtype=_TEXT).reindex(columns=COLUMNS)


def read_rows(book: Optional[str] = None, path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Journal rows, oldest first, as JSON-ready dicts (one book's when ``book`` is given)."""
    df = read_journal(path)
    if book:
        df = df[df["book"] == book]
    rows = []
    for r in df.to_dict(orient="records"):
        r = {k: (None if pd.isna(v) else v) for k, v in r.items()}
        for k in ("cost_model", "pbo_n"):
            r[k] = int(r[k]) if r[k] is not None else None
        rows.append(r)
    return rows


def write(rows: pd.DataFrame, path: Optional[Path] = None) -> Path:
    """The whole journal, oldest first (rows are only ever added)."""
    p = _path(path)
    df = rows.reindex(columns=COLUMNS).sort_values(["date", "book", "kind"], kind="stable")
    df["cost_model"] = df["cost_model"].astype("Int64")
    df["pbo_n"] = df["pbo_n"].astype("Int64")
    df.to_csv(p, index=False, float_format="%.6g")
    return p


def targets() -> List[Dict[str, Any]]:
    """What each journal metric is judged against: the scorecard's pass rules plus the CAGR goal.
    ``kind`` None applies to every kind of row."""
    from nse_engine.scorecard import PASS_RULES

    column = {"sharpe": ("sharpe", "backtest"), "max_drawdown": ("max_dd", None), "calmar": ("calmar", None),
              "dsr": ("dsr", "backtest"), "oos_sharpe": ("sharpe", "walk_forward")}
    out = [{"label": "CAGR", "metric": "cagr", "kind": None, "op": ">", "value": CAGR_TARGET}]
    for label, key, op, value in PASS_RULES:
        metric, kind = column[key]
        out.append({"label": label, "metric": metric, "kind": kind, "op": op, "value": value})
    return out


# ── evidence ────────────────────────────────────────────────────

def backtest_rows(book, runs_dir: Path) -> List[Dict[str, Any]]:
    """The book's configuration over the validation window: a row per registry run whose cost model, data
    hash or results differ from the last row's (oldest first)."""
    from nse_engine import books as bk
    from nse_engine.validation.trials import LEGACY_COST_MODEL

    rows: List[Dict[str, Any]] = []
    last = None
    for run in bk.same_window_runs(runs_dir, book.config_hash):
        m = run.get("metrics") or {}
        cost_model = int(run.get("cost_model") or LEGACY_COST_MODEL)
        sig = (cost_model, run.get("data_hash"), *(round(float(m.get(k) or 0.0), 4)
                                                   for k in ("cagr", "sharpe", "max_drawdown")))
        if sig == last:
            continue
        last = sig
        s = bk.run_scores(run, runs_dir)
        v = bk._validation(Path(run["_dir"])) or {}
        haircut, pbo = v.get("haircut") or {}, v.get("pbo")
        rows.append(dict(
            date=_ist_date(run.get("created_at")), book=book.name, kind="backtest",
            event=str(run.get("tag") or "")[:EVENT_CHARS], cost_model=cost_model,
            data_hash=str(run.get("data_hash") or "")[:8], config_hash=book.fingerprint, ref=run["run_id"],
            cagr=s["bt_cagr"], sharpe=s["bt_sharpe"], max_dd=s["bt_max_dd"], calmar=s["bt_calmar"],
            sharpe_2017_25=s["bt_sharpe_2017_25"], dsr=s.get("dsr"), pbo=s.get("pbo"), pbo_n=s.get("pbo_n"),
            exp_sharpe=(haircut.get("selection") or {}).get("expected_sharpe_after_both"),
            exp_cagr=(haircut.get("expected_after_execution") or {}).get("cagr"),
            note="PBO on excess returns" if isinstance(pbo, dict) and pbo.get("return_basis") == "excess" else ""))
    return rows


def walk_forward_row(book: str, returns_csv: str, runs_dir: Path,
                     event: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """A walk-forward's out-of-sample result (the scorecard's figures) as a journal row, dated and costed
    by its last fold's test run; None when the returns file is missing."""
    from nse_engine import books as bk
    from nse_engine.scorecard import walk_forward_oos
    from nse_engine.validation.trials import LEGACY_COST_MODEL

    p = Path(returns_csv)
    p = p if p.is_absolute() else REPO_ROOT / p
    wf = walk_forward_oos(str(p), bk.RF)
    if "error" in wf:
        return None
    stitched = p.with_name(p.name.replace("wf_oos_returns", "wf_stitched").replace(".csv", ".json"))
    doc = json.loads(stitched.read_text()) if stitched.exists() else {}
    folds = doc.get("folds") or []
    test: Dict[str, Any] = {}
    if folds and folds[-1].get("test_run_id"):
        mf = Path(runs_dir) / folds[-1]["test_run_id"] / "manifest.json"
        test = json.loads(mf.read_text()) if mf.exists() else {}
    when = _ist_date(test["created_at"]) if test.get("created_at") else date.fromtimestamp(p.stat().st_mtime).isoformat()
    name = p.stem.replace("wf_oos_returns_", "")
    return dict(
        date=when, book=book, kind="walk_forward",
        event=(event or f"walk-forward {name}: {wf.get('n_folds')} folds, {wf.get('n_grid_points')} settings")[:EVENT_CHARS],
        cost_model=int(test.get("cost_model") or LEGACY_COST_MODEL) if test else None,
        data_hash=str(test.get("data_hash") or "")[:8], config_hash=str(wf.get("base_config_hash") or "")[:8],
        ref=p.name, cagr=wf["cagr"], sharpe=wf["sharpe"], max_dd=wf["max_drawdown"], calmar=wf["calmar"],
        note=str((doc.get("summary") or {}).get("note") or ""))


def record(books: Sequence, runs_dir: Path, path: Optional[Path] = None,
           event: Optional[str] = None) -> List[Dict[str, Any]]:
    """Append the evidence not yet journaled: each book's backtest re-baselines and its family's current
    walk-forward.  Idempotent (rows are keyed by book, kind and ref); returns the rows added."""
    from nse_engine.scorecard import WALK_FORWARD_OOS

    old = read_journal(path)
    seen = set(zip(old["book"], old["kind"], old["ref"]))
    new: List[Dict[str, Any]] = []
    for b in books:
        found = backtest_rows(b, runs_dir)
        if b.name in WALK_FORWARD_OOS:
            wf = walk_forward_row(b.name, WALK_FORWARD_OOS[b.name], runs_dir)
            found += [wf] if wf else []
        new += [r for r in found if (r["book"], r["kind"], r["ref"]) not in seen]
    if event:
        for r in new:
            r["event"] = event[:EVENT_CHARS]
    if new:
        write(pd.concat([old, pd.DataFrame(new)], ignore_index=True) if len(old) else pd.DataFrame(new), path)
    return new


# ── CLI ─────────────────────────────────────────────────────────

def format_table(df: pd.DataFrame, book: Optional[str] = None) -> str:
    if book:
        df = df[df["book"] == book]

    def num(v: Any, digits: int, scale: float = 1.0) -> str:
        return f"{float(v) * scale:.{digits}f}" if v is not None and not pd.isna(v) else "-"

    cols = [("date", 10), ("book", 9), ("kind", 12), ("cm", 2), ("data", 8), ("CAGR%", 6), ("Sharpe", 6),
            ("MaxDD%", 6), ("Calmar", 6), ("DSR", 5), ("PBO%", 5)]
    lines = [" ".join(f"{c:{w}s}" for c, w in cols) + "  event"]
    for r in df.to_dict(orient="records"):
        cells = [r["date"], r["book"], r["kind"], num(r["cost_model"], 0), r["data_hash"] if isinstance(r["data_hash"], str) else "-",
                 num(r["cagr"], 2, 100), num(r["sharpe"], 3), num(r["max_dd"], 1, 100), num(r["calmar"], 2),
                 num(r["dsr"], 3), num(r["pbo"], 1, 100)]
        lines.append(" ".join(f"{str(v):{w}s}" for v, (_, w) in zip(cells, cols)) + f"  {r['event']}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Metrics journal (tracker JR1)")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="append the evidence not yet journaled (idempotent)")
    r.add_argument("--runs-dir", default=EngineConfig().runs_dir)
    r.add_argument("--event", help="label for the rows this call adds (default: each run's tag)")
    s = sub.add_parser("show", help="print the journal")
    s.add_argument("--book")
    args = p.parse_args(argv)
    if args.cmd == "record":
        from nse_engine.books import discover_books

        added = record(discover_books(), Path(args.runs_dir), event=args.event)
        print(f"{len(added)} new row(s) in {JOURNAL_PATH}")
        return 0
    print(format_table(read_journal(), args.book))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
