"""Point-in-time S&P 500 membership (tracker US2): which stocks were in the index on each date.

A backtest that picks from today's members drops every company that failed or
left, so it measures survivors (survivorship bias).  This module keeps, for each
ticker, the dated intervals it was a member, so a decision on a past date can
only pick that date's members.

Source: fja05680/sp500 (MIT), pinned to one commit (``FJA_COMMIT``): a member
list for every date the index changed since 1996.  Its author trusts the lists
from 2001 on.  Its tickers name companies, mostly by the ticker they last traded
under (bankrupt names keep their "Q" suffix, e.g. LEHMQ), so a ticker is an
identity for membership, not always a price symbol (see ``coverage``).  The lists
are checked against Wikipedia's dated changes ("Historical components of the
S&P 500", moved there from the main list in August 2026).

    python -m us_engine.membership build      # fetch the pinned lists, check, write data/sp500_membership.json
    python -m us_engine.membership coverage [--from 2013-01-01]   # yfinance coverage of the window's members
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence

import numpy as np
import pandas as pd

from us_engine.data import INDEX_TICKERS, USER_AGENT, _fetch

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
FJA_REPO = "fja05680/sp500"
#: Commit of 7 Sep 2026 ("2026-09-07 update"); lists through 18 Aug 2026.
FJA_COMMIT = "a2430f2af0c79ddf0748e91de11bdeb1616ab5a7"
FJA_FILE = "S&P 500 Historical Components & Changes (Updated).csv"
WIKI_HISTORY_URL = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"
US_STORE = REPO_ROOT / "data" / "nse_engine" / "us_store"
RAW_DIR = US_STORE / "raw"
#: Versioned: results depend on them (as data/nifty500_historical_constituents.json).
MEMBERSHIP_PATH = REPO_ROOT / "data" / "sp500_membership.json"
#: Curated price symbols for renamed members (evidence per entry).
SYMBOL_MAP_PATH = REPO_ROOT / "data" / "sp500_symbol_map.json"
#: Plausibility of a priced series (a reused ticker can price another security).
PRICE_RANGE = (1.0, 10_000.0)
MAX_DAY_MOVE = 0.6
#: The development window (US5) and the warm-up before it; 2026 is the sealed holdout.
WINDOW = ("2003-01-01", "2025-12-31")
#: Research data: prices from RESEARCH_ANCHOR (a year of warm-up), results from RESEARCH_START, so the
#: 2007-09 crash is in sample (your ask 16 Oct); the primary window stays 2013-25 (US-Y5), the earlier
#: years a stress window as India's R4.  One anchor for every US trial: the expanding forecast
#: normalisers make results depend on it.
RESEARCH_ANCHOR = "2006-01-03"
RESEARCH_START = "2007-01-01"
#: Days a Wikipedia change may sit from the lists' date for the two to agree (announced vs effective).
CHECK_TOLERANCE_DAYS = 7
COVERED_SHARE = 0.98
CHUNK = 100
RETRY_CHUNK = 25


def fetch_components(commit: str = FJA_COMMIT, raw_dir: Path = RAW_DIR) -> Path:
    """The pinned member lists, downloaded once into the (git-ignored) US store."""
    import requests
    from urllib.parse import quote

    path = Path(raw_dir) / f"sp500_components_{commit[:8]}.csv"
    if not path.exists():
        url = f"https://raw.githubusercontent.com/{FJA_REPO}/{commit}/{quote(FJA_FILE)}"
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
        resp.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)
    return path


def daily_lists(path: Path) -> pd.Series:
    """Date -> frozenset of member tickers, one row per date the index changed."""
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    lists = df.set_index("date")["tickers"].map(lambda s: frozenset(t.strip() for t in str(s).split(",") if t.strip()))
    if lists.index.has_duplicates:
        raise ValueError("member lists repeat a date")
    return lists


def intervals(lists: pd.Series) -> pd.DataFrame:
    """One row per membership spell: ``ticker``, ``start`` and ``until`` (first date out; NaT = still a member)."""
    open_: Dict[str, pd.Timestamp] = {}
    rows: List[Dict] = []
    prev: FrozenSet[str] = frozenset()
    for date, members in lists.items():
        for t in members - prev:
            open_[t] = date
        for t in prev - members:
            rows.append({"ticker": t, "start": open_.pop(t), "until": date})
        prev = members
    rows += [{"ticker": t, "start": s, "until": pd.NaT} for t, s in open_.items()]
    return pd.DataFrame(rows).sort_values(["ticker", "start"]).reset_index(drop=True)


def members_on(spells: pd.DataFrame, date) -> FrozenSet[str]:
    """Tickers that were members on ``date``."""
    d = pd.Timestamp(date)
    live = (spells["start"] <= d) & (spells["until"].isna() | (spells["until"] > d))
    return frozenset(spells.loc[live, "ticker"])


def in_window(spells: pd.DataFrame, window: Sequence[str] = WINDOW) -> pd.DataFrame:
    w0, w1 = (pd.Timestamp(x) for x in window)
    return spells[(spells["start"] <= w1) & (spells["until"].isna() | (spells["until"] > w0))]


def wikipedia_changes() -> pd.DataFrame:
    """Wikipedia's dated changes: ``date``, ``added``, ``removed`` (tickers; NaN when one side is empty)."""
    import requests

    resp = requests.get(WIKI_HISTORY_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    table = pd.read_html(io.StringIO(resp.text), match="Effective Date")[0]
    table.columns = ["_".join(str(x) for x in c) if isinstance(c, tuple) else str(c) for c in table.columns]
    out = pd.DataFrame({
        "date": pd.to_datetime(table["Effective Date_Effective Date"], format="mixed", errors="coerce"),
        "added": table["Added_Ticker"].astype("string").str.replace(r"\s*\|.*$", "", regex=True).str.strip(),
        "removed": table["Removed_Ticker"].astype("string").str.replace(r"\s*\|.*$", "", regex=True).str.strip(),
    }).dropna(subset=["date"])
    if len(out) < 300:
        raise ValueError(f"Wikipedia changes table has {len(out)} rows: the page changed")
    return out.sort_values("date").reset_index(drop=True)


def cross_check(spells: pd.DataFrame, changes: pd.DataFrame, window: Sequence[str] = WINDOW,
                tolerance_days: int = CHECK_TOLERANCE_DAYS) -> pd.DataFrame:
    """Each Wikipedia change in the window against the lists: a spell of the added ticker starting,
    and one of the removed ticker ending, within ``tolerance_days`` of its date."""
    w0, w1 = (pd.Timestamp(x) for x in window)
    tol = pd.Timedelta(days=tolerance_days)
    rows = []
    for c in changes[(changes["date"] >= w0) & (changes["date"] <= w1)].itertuples(index=False):
        for side, ticker in (("added", c.added), ("removed", c.removed)):
            if ticker is None or pd.isna(ticker):
                continue
            mine = spells[spells["ticker"] == ticker]
            edge = mine["start"] if side == "added" else mine["until"]
            gap = (edge - c.date).abs().min() if len(mine) and edge.notna().any() else pd.NaT
            rows.append({"date": c.date, "side": side, "ticker": ticker,
                         "known": bool(len(mine)), "days_off": gap.days if pd.notna(gap) else None,
                         "agrees": bool(pd.notna(gap) and gap <= tol)})
    return pd.DataFrame(rows)


def save(spells: pd.DataFrame, commit: str, source_sha256: str, check: Dict, path: Path = MEMBERSHIP_PATH) -> Path:
    payload = {
        "source": f"https://github.com/{FJA_REPO} ({FJA_FILE})",
        "license": "MIT, (c) fja05680",
        "commit": commit,
        "source_sha256": source_sha256,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wikipedia_check": check,
        "columns": ["ticker", "start", "until"],
    }
    rows = ",\n  ".join(json.dumps([r.ticker, r.start.date().isoformat(),
                                    r.until.date().isoformat() if pd.notna(r.until) else None])
                         for r in spells.itertuples(index=False))
    head = json.dumps(payload, indent=1)
    path.write_text(head[:-2] + ',\n "spells": [\n  ' + rows + "\n ]\n}\n")     # one spell per line: readable diffs
    return path


def load_membership(path: Path = MEMBERSHIP_PATH) -> pd.DataFrame:
    raw = json.loads(Path(path).read_text())
    df = pd.DataFrame(raw["spells"], columns=raw["columns"])
    df["start"] = pd.to_datetime(df["start"])
    df["until"] = pd.to_datetime(df["until"])
    return df


def build(commit: str = FJA_COMMIT) -> Dict:
    """Fetch the pinned lists, derive the spells, check them against Wikipedia, write the versioned file."""
    path = fetch_components(commit)
    lists = daily_lists(path)
    spells = intervals(lists)
    check = cross_check(spells, wikipedia_changes())
    summary = {
        "window": list(WINDOW),
        "changes_checked": int(len(check)),
        "agree": int(check["agrees"].sum()),
        "ticker_unknown": int((~check["known"]).sum()),
        "date_off": int((check["known"] & ~check["agrees"]).sum()),
        "tolerance_days": CHECK_TOLERANCE_DAYS,
    }
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    save(spells, commit, sha, summary)
    check.to_csv(US_STORE / "membership_check.csv", index=False)
    w = in_window(spells)
    return {**summary, "lists": f"{lists.index[0].date()}..{lists.index[-1].date()}",
            "spells": int(len(spells)), "spells_in_window": int(len(w)), "tickers_in_window": int(w["ticker"].nunique()),
            "left_in_window": int((w["until"].notna() & (w["until"] <= pd.Timestamp(WINDOW[1]))).sum())}


# ============================================================================
# price coverage
# ============================================================================


def yahoo_symbol(ticker: str) -> str:
    return ticker.replace(".", "-")


def load_symbol_map(path: Path = SYMBOL_MAP_PATH) -> pd.DataFrame:
    """The curated aliases: ``ticker``, ``symbol``, ``from``, ``until`` (NaT = open), ``evidence``."""
    df = pd.DataFrame(json.loads(Path(path).read_text())["aliases"])
    df["from"] = pd.to_datetime(df["from"])
    df["until"] = pd.to_datetime(df["until"])
    return df


def segments(ticker: str, lo: pd.Timestamp, hi: pd.Timestamp, aliases: pd.DataFrame) -> List[tuple]:
    """``[(lo, hi, yahoo symbol)]`` covering [lo, hi]: the aliases where they apply, the ticker elsewhere."""
    mine = aliases[aliases["ticker"] == ticker].to_dict("records")
    cuts = sorted({lo, hi + pd.Timedelta(days=1)} | {d for a in mine for d in (a["from"], a["until"])
                                                       if pd.notna(d) and lo < d <= hi})
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        hit = aliases[(aliases["ticker"] == ticker) & (aliases["from"].isna() | (aliases["from"] <= a))
                      & (aliases["until"].isna() | (aliases["until"] > a))]
        out.append((a, b - pd.Timedelta(days=1), hit["symbol"].iloc[0] if len(hit) else yahoo_symbol(ticker)))
    return out


def coverage(spells: Optional[pd.DataFrame] = None, window: Sequence[str] = WINDOW,
             out: Path = US_STORE / "coverage_yfinance.csv") -> pd.DataFrame:
    """For every spell in the window, the share of its sessions yfinance prices (aliases applied).

    Columns: ticker, start, until, symbols, sessions, priced, share, status
    (``covered`` >= COVERED_SHARE, ``partial``, ``none``) and ``flags``: a priced
    series is not proof of identity (Yahoo's PARA is another security), so a
    median close outside PRICE_RANGE or a one-day move beyond MAX_DAY_MOVE is
    flagged for review.
    """
    spells = in_window(load_membership() if spells is None else spells, window)
    aliases = load_symbol_map()
    w0, w1 = (pd.Timestamp(x) for x in window)
    tickers = sorted(spells["ticker"].unique())
    start = (w0 - pd.Timedelta(days=400)).date().isoformat()
    closes: List[pd.DataFrame] = []
    symbols = sorted({yahoo_symbol(t) for t in tickers} | set(aliases["symbol"]))
    for i in range(0, len(symbols), CHUNK):
        part = symbols[i:i + CHUNK] + ([INDEX_TICKERS["SPX"]] if i == 0 else [])
        got = _fetch(part, start, threads=True).get("Close")
        if got is not None:
            closes.append(got)
        logger.info("coverage: %d/%d tickers fetched", min(i + CHUNK, len(symbols)), len(symbols))
    close = pd.concat(closes, axis=1)
    close = close.loc[:, ~close.columns.duplicated()]
    failed = [s for s in symbols if s not in close.columns or close[s].isna().all()]
    for i in range(0, len(failed), RETRY_CHUNK):                 # Yahoo drops some of a big batch
        got = _fetch(failed[i:i + RETRY_CHUNK], start, threads=False).get("Close")
        if got is not None:
            got = got.dropna(axis=1, how="all")
            close = pd.concat([close.drop(columns=[c for c in got.columns if c in close.columns]), got], axis=1)
    logger.info("coverage: %d tickers retried, %d still without prices", len(failed),
                sum(1 for s in symbols if s not in close.columns or close[s].isna().all()))
    sessions = close[INDEX_TICKERS["SPX"]].dropna().index
    sessions = sessions[(sessions >= w0) & (sessions <= w1)]
    rows = []
    for s in spells.itertuples(index=False):
        lo = max(s.start, w0)
        hi = min(s.until - pd.Timedelta(days=1), w1) if pd.notna(s.until) else w1
        parts = []
        for a, b, sym in segments(s.ticker, lo, hi, aliases):
            days = sessions[(sessions >= a) & (sessions <= b)]
            parts.append(close[sym].reindex(days) if sym in close.columns else pd.Series(np.nan, index=days))
        series = pd.concat(parts) if parts else pd.Series(dtype="float64")
        priced = int(series.notna().sum())
        share = priced / len(series) if len(series) else 1.0
        flags = []
        if priced:
            med = float(series.median())
            if not PRICE_RANGE[0] <= med <= PRICE_RANGE[1]:
                flags.append(f"median close {med:,.2f}")
            # within each symbol's segment: a switch of symbol is a change of instrument, not a return
            moves = [p.dropna().pct_change().abs().max() for p in parts]
            move = max((float(m) for m in moves if pd.notna(m)), default=0.0)
            if move > MAX_DAY_MOVE:
                flags.append(f"one-day move {move:.0%}")
        rows.append({"ticker": s.ticker, "start": s.start.date(), "until": s.until.date() if pd.notna(s.until) else None,
                     "symbols": " ".join(dict.fromkeys(sym for _, _, sym in segments(s.ticker, lo, hi, aliases))),
                     "sessions": int(len(series)), "priced": priced, "share": round(share, 4),
                     "status": "covered" if share >= COVERED_SHARE else ("none" if priced == 0 else "partial"),
                     "flags": "; ".join(flags)})
    report = pd.DataFrame(rows)
    report.to_csv(out, index=False)
    return report


def coverage_summary(report: pd.DataFrame) -> Dict:
    """Member-days priced and the spells by status, split by whether the spell ended in the window."""
    left = report["until"].notna()
    by = lambda m: report.loc[m, "status"].value_counts().to_dict()
    return {"member_days": int(report["sessions"].sum()), "priced_days": int(report["priced"].sum()),
            "priced_share": round(float(report["priced"].sum() / max(report["sessions"].sum(), 1)), 4),
            "spells_still_members": by(~left), "spells_that_left": by(left)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m us_engine.membership", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="fetch the pinned lists, check them against Wikipedia, write the membership file")
    cov = sub.add_parser("coverage", help="yfinance price coverage of every member in the window")
    cov.add_argument("--from", dest="start", default=WINDOW[0], help=f"window start (default {WINDOW[0]})")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.cmd == "build":
        print(json.dumps(build(), indent=2))
    else:
        window = (args.start, WINDOW[1])
        out = US_STORE / f"coverage_yfinance_{window[0][:4]}.csv"
        print(json.dumps(coverage_summary(coverage(window=window, out=out)), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
