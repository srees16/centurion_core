"""Metrics journal (tracker JR1): how each book's headline numbers have moved over time.

``docs/metrics_journal.csv`` is append-only: one row per book per new piece
of research evidence, never edited afterwards.

* ``backtest`` - the book's configuration over 2013-25 in the run registry, a
  row each time its cost model, data hash or results change (a re-baseline),
  with its 2017-25 Sharpe and, when the run was validated, DSR, PBO and the
  execution haircut's expected out-of-sample Sharpe and CAGR;
* ``walk_forward`` - the anchored walk-forward of the book's configuration
  family (``scorecard.WALK_FORWARD_OOS``), a row each time it is re-run;
* ``experiment`` - a concluded experiment that changed no book (a
  pre-registered trial or one measured look) with its ``verdict`` (FAIL,
  CLOSED, NO-GO), for every book (``book`` = ``all``), so the journal also
  shows what was tried when the numbers did not move.  Its metric columns
  hold the whole book's 2013-25 backtest when the test produced one; other
  figures are in ``note``.  Added by :func:`add_experiment` when the
  tracker's DONE entry is written.

The run registry lives only on the research machine, so rows are written
there: ``python -m nse_engine.books register`` calls :func:`record` after
every re-baseline, and the rows are committed with the change that moved
them (``git log -p docs/metrics_journal.csv`` says why).  Paper results are
journaled every night in Neon (the daily snapshots); the web page (Trade
Center > Journal) shows both against :func:`targets`.

    python -m nse_engine.journal record          # append what is new (idempotent)
    python -m nse_engine.journal experiment --ref GT4 --date 2026-10-12 --verdict FAIL --event "..." --note "..."
    python -m nse_engine.journal show --book e4
    python -m nse_engine.journal check           # verdicts in the tracker without a journal row (PL-01)

The check (tracker PL0, 14 Nov 2026): the experiment rows were added by hand and stopped on 1 Nov; 22
verdicts of 2-13 Nov never reached the journal.  ``check`` reads the tracker's DONE section and names every
entry from ``CHECK_SINCE`` whose opening states a verdict and has no row here; a test runs it.  From
``POSTMORTEM_SINCE`` an experiment row must also carry its post-mortem (PL-06): where it stopped, what it
cost, the earliest point it could have been stopped, and the lesson it left.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from nse_engine.config import EngineConfig
from nse_engine.deployment import REPO_ROOT

JOURNAL_PATH = "docs/metrics_journal.csv"
#: An experiment's post-mortem (PL-06).  ``phase_stopped``: one of ``PHASES``; ``hours`` of work and
#: ``trials_used`` (configurations added to the family's N); ``compute`` in words ("Mac 10 min", "Kaggle 3 h");
#: ``earliest_stop``: the phase at which a check could have stopped it, and which check; ``lesson_ids``: ledger
#: ids it created or repeated, or "none: <why>"; ``cleanup``: "done" or what awaits approval.
POSTMORTEM = ["idea_type", "market", "phase_stopped", "hours", "trials_used", "compute", "earliest_stop",
              "lesson_ids", "cleanup"]
PHASES = ("0 feasibility", "1 pre-registration", "2 run", "3 gate", "4 paper")
#: What a row from ``POSTMORTEM_SINCE`` may not leave empty.
POSTMORTEM_REQUIRED = ("phase_stopped", "hours", "trials_used", "earliest_stop", "lesson_ids")
COLUMNS = ["date", "book", "kind", "event", "cost_model", "data_hash", "config_hash", "ref",
           "cagr", "sharpe", "max_dd", "calmar", "sharpe_2017_25", "dsr", "pbo", "pbo_n",
           "exp_sharpe", "exp_cagr", "note", "verdict"] + POSTMORTEM
_TEXT = {c: str for c in ("event", "data_hash", "config_hash", "ref", "note", "verdict", "idea_type", "market",
                          "phase_stopped", "compute", "earliest_stop", "lesson_ids", "cleanup")}
EXPERIMENT = "experiment"
#: ``book`` of a row every book shows (experiments).
ALL_BOOKS = "all"
#: What ``add_experiment`` may fill besides its own fields.
EXPERIMENT_VALUES = ("cost_model", "config_hash", "cagr", "sharpe", "max_dd", "calmar", "sharpe_2017_25",
                     "dsr", "pbo", "pbo_n") + tuple(POSTMORTEM)
TRACKER_PATH = "docs/MEMORY_TRACKER.txt"
#: Tracker verdicts from this date must be journaled (PL-01); earlier ones were never checked.
CHECK_SINCE = "2026-11-02"
#: Experiment rows from this date must carry their post-mortem (PL-06); earlier costs were not recorded.
POSTMORTEM_SINCE = "2026-11-15"
#: Tracker entries that state a verdict but are not an experiment, with the reason (none today).
JOURNAL_EXEMPT: Dict[str, str] = {
    "ST2S": "an operational go-live check (the supervised sell test), not a research verdict; its record is the "
            "live book's sell_path state, read by capital_ladder.sell_path_check",
}
#: How a tracker entry states a verdict, and the journal's word for it.  Read only in the clause that follows
#: the title, "<title> (<plan, notes>): FAIL, closed, ...": the same words elsewhere ("the run FAILED once")
#: are not a verdict.
_VERDICT = re.compile(r"\b(FAIL(?:ED|S)?|CLOSED|NO-GO|DISCARDED|BLOCKED|WITHDRAWN|PASS(?:ED)?)\b"
                      r"|\b[Nn]ot adopted|\bdoes not fit|\bfails, closed|\bnot worth doing")
_HEAD_CHARS = 320
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
    """Journal rows, oldest first, as JSON-ready dicts (one book's and the experiments when ``book`` is given)."""
    df = read_journal(path)
    if book:
        df = df[df["book"].isin([book, ALL_BOOKS])]
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


def add_experiment(ref: str, day: str, event: str, verdict: str, note: str = "", book: str = ALL_BOOKS,
                   path: Optional[Path] = None, **values: Any) -> bool:
    """Append a concluded experiment that changed no book: ``ref`` its tracker ID, ``day`` the verdict's
    date, ``note`` the result in plain words.  ``values`` (``EXPERIMENT_VALUES``) fill the metric columns
    with the whole book's 2013-25 backtest when the test produced one.  Idempotent on ``ref``: False
    when it is already journaled."""
    unknown = set(values) - set(EXPERIMENT_VALUES)
    if unknown:
        raise ValueError(f"not experiment values: {sorted(unknown)}")
    if len(event) > EVENT_CHARS:
        raise ValueError(f"event longer than {EVENT_CHARS} characters: {event!r}")
    old = read_journal(path)
    if ((old["kind"] == EXPERIMENT) & (old["ref"] == ref)).any():
        return False
    phase = values.get("phase_stopped")
    if phase is not None and phase not in PHASES:
        raise ValueError(f"phase_stopped must be one of {PHASES}, not {phase!r}")
    row = {**values, "date": pd.Timestamp(day).date().isoformat(), "book": book, "kind": EXPERIMENT,
           "event": event, "ref": ref, "verdict": verdict.strip().upper(), "note": note}
    write(pd.concat([old, pd.DataFrame([row])], ignore_index=True) if len(old) else pd.DataFrame([row]), path)
    return True


# ── The tracker against the journal (PL-01, PL-06) ──────────────

def tracker_entries(text: str, today: Optional[date] = None) -> List[Dict[str, Any]]:
    """The DONE section's entries of ``docs/MEMORY_TRACKER.txt``: {date, id, text}, in file order.  The
    tracker writes "9 Nov" without a year: the latest year that does not put the entry after ``today``."""
    today = today or datetime.now(IST).date()
    lines = text.split("\n")
    start = next((i for i, l in enumerate(lines) if l.startswith(" 8. DONE")), None)
    if start is None:
        raise ValueError("the tracker has no ' 8. DONE' section")
    out: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    for ln in lines[start + 1:]:
        m = re.match(r"^(\d{1,2}) ([A-Z][a-z]{2})\s+(\S+)\s+(\S.*)$", ln)
        if m:
            month = datetime.strptime(m.group(2), "%b").month
            day = next((d for d in (date(y, month, int(m.group(1))) for y in (today.year, today.year - 1))
                        if d <= today + timedelta(days=2)), None)
            cur = {"date": day.isoformat() if day else None, "id": m.group(3), "text": m.group(4)}
            out.append(cur)
        elif cur is not None and ln.startswith(" "):
            cur["text"] += " " + ln.strip()
    return out


def stated_verdict(text: str) -> Optional[str]:
    """The journal's word for the verdict an entry states, or None when it states none.  The verdict is in the
    clause after the title's colon ("...): FAIL on its pass route."), within the entry's opening."""
    head = text[:_HEAD_CHARS]
    at = head.find("): ")
    at = at + 3 if at >= 0 else (head.find(": ") + 2 if ": " in head else -1)
    if at < 0:
        return None
    clause = re.split(r"\.(?:\s|$)", head[at:], maxsplit=1)[0]
    m = _VERDICT.search(clause)
    if not m:
        return None
    word = m.group(0).lower()
    return ("FAIL" if m.group(0).startswith("FAIL") else "PASS" if m.group(0).startswith("PASS")
            else "CLOSED" if "closed" in word else "NOT ADOPTED" if word in ("not adopted", "does not fit",
                                                                              "not worth doing")
            else m.group(0))


def check(tracker_text: Optional[str] = None, path: Optional[Path] = None, today: Optional[date] = None,
          since: str = CHECK_SINCE, postmortem_since: str = POSTMORTEM_SINCE) -> Dict[str, List[Dict[str, Any]]]:
    """What the journal is missing: ``unjournaled`` tracker verdicts dated ``since`` or later with no
    experiment row (and not in ``JOURNAL_EXEMPT``), and ``no_postmortem`` experiment rows dated
    ``postmortem_since`` or later with an empty required post-mortem field."""
    if tracker_text is None:
        tracker_text = (REPO_ROOT / TRACKER_PATH).read_text()
    rows = read_journal(path)
    exp = rows[rows["kind"] == EXPERIMENT]
    done = set(exp["ref"].astype(str))
    unjournaled = []
    for e in tracker_entries(tracker_text, today):
        verdict = stated_verdict(e["text"])
        if verdict and e["date"] and e["date"] >= since and e["id"] not in done and e["id"] not in JOURNAL_EXEMPT:
            unjournaled.append({"id": e["id"], "date": e["date"], "verdict": verdict})
            done.add(e["id"])
    blank = lambda v: v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == ""
    no_postmortem = [{"ref": r["ref"], "date": r["date"], "missing": [c for c in POSTMORTEM_REQUIRED if blank(r[c])]}
                     for r in exp[exp["date"].astype(str) >= postmortem_since].to_dict(orient="records")
                     if any(blank(r[c]) for c in POSTMORTEM_REQUIRED)]
    return {"unjournaled": unjournaled, "no_postmortem": no_postmortem}


# ── CLI ─────────────────────────────────────────────────────────

def format_table(df: pd.DataFrame, book: Optional[str] = None) -> str:
    if book:
        df = df[df["book"].isin([book, ALL_BOOKS])]

    def num(v: Any, digits: int, scale: float = 1.0) -> str:
        return f"{float(v) * scale:.{digits}f}" if v is not None and not pd.isna(v) else "-"

    cols = [("date", 10), ("book", 9), ("kind", 12), ("cm", 2), ("data", 8), ("CAGR%", 6), ("Sharpe", 6),
            ("MaxDD%", 6), ("Calmar", 6), ("DSR", 5), ("PBO%", 5)]
    lines = [" ".join(f"{c:{w}s}" for c, w in cols) + "  event"]
    for r in df.to_dict(orient="records"):
        cells = [r["date"], r["book"], r["kind"], num(r["cost_model"], 0), r["data_hash"] if isinstance(r["data_hash"], str) else "-",
                 num(r["cagr"], 2, 100), num(r["sharpe"], 3), num(r["max_dd"], 1, 100), num(r["calmar"], 2),
                 num(r["dsr"], 3), num(r["pbo"], 1, 100)]
        verdict = f"[{r['verdict']}] " if isinstance(r.get("verdict"), str) else ""
        lines.append(" ".join(f"{str(v):{w}s}" for v, (_, w) in zip(cells, cols)) + f"  {verdict}{r['event']}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Metrics journal (tracker JR1)")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="append the evidence not yet journaled (idempotent)")
    r.add_argument("--runs-dir", default=EngineConfig().runs_dir)
    r.add_argument("--event", help="label for the rows this call adds (default: each run's tag)")
    x = sub.add_parser("experiment", help="append a concluded experiment that changed no book (idempotent)")
    x.add_argument("--ref", required=True, help="its tracker ID, e.g. GT4")
    x.add_argument("--date", required=True, help="the verdict's date, YYYY-MM-DD")
    x.add_argument("--verdict", required=True, help="FAIL, CLOSED or NO-GO")
    x.add_argument("--event", required=True, help=f"what was tried, at most {EVENT_CHARS} characters")
    x.add_argument("--note", default="", help="the result in plain words")
    x.add_argument("--book", default=ALL_BOOKS)
    for col in EXPERIMENT_VALUES:
        typ = (int if col in ("cost_model", "pbo_n") else float if col in ("hours", "trials_used")
               else str if col == "config_hash" or col in POSTMORTEM else float)
        x.add_argument(f"--{col.replace('_', '-')}", dest=col, type=typ,
                       help="the whole book's 2013-25 backtest" if typ is float else None)
    s = sub.add_parser("show", help="print the journal")
    s.add_argument("--book")
    sub.add_parser("check", help="tracker verdicts with no journal row, and rows without a post-mortem")
    args = p.parse_args(argv)
    if args.cmd == "check":
        found = check()
        for e in found["unjournaled"]:
            print(f"not journaled: {e['id']} ({e['date']}, {e['verdict']})")
        for r in found["no_postmortem"]:
            print(f"no post-mortem: {r['ref']} ({r['date']}): {', '.join(r['missing'])}")
        print(f"{len(found['unjournaled'])} verdict(s) not journaled, {len(found['no_postmortem'])} row(s) "
              f"without a post-mortem")
        return 1 if found["unjournaled"] or found["no_postmortem"] else 0
    if args.cmd == "record":
        from nse_engine.books import discover_books

        added = record(discover_books(), Path(args.runs_dir), event=args.event)
        print(f"{len(added)} new row(s) in {JOURNAL_PATH}")
        return 0
    if args.cmd == "experiment":
        values = {c: getattr(args, c) for c in EXPERIMENT_VALUES if getattr(args, c) is not None}
        added = add_experiment(args.ref, args.date, args.event, args.verdict, args.note, args.book, **values)
        print(f"{args.ref}: {'added to' if added else 'already in'} {JOURNAL_PATH}")
        return 0
    print(format_table(read_journal(), args.book))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
