"""
NSE reference data: ETF list, date-aware symbol changes, sector map.

Symbol linking
--------------
NSE renames symbols (ZOMATO -> ETERNAL, TATAMOTORS -> TMPV) and changes
ISINs on face-value splits, and it re-uses retired symbols for unrelated
companies.  A bhavcopy row ``(symbol, date)`` is mapped to its canonical
(latest) name by repeatedly applying the first rename of the current label
that takes effect *after* the row's date.  Renames come from

* ``symbolchange.csv`` (old, new, effective date), and
* ISIN continuity: a symbol that stops trading and a brand-new symbol that
  starts within a few sessions with the same ISIN.

Each rename only applies to the old symbol's *contiguous* history before the
rename (``valid_from``): a gap in trading combined with a different ISIN
marks an earlier, unrelated company that used the same symbol.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

ETF_LIST = "eq_etfseclist.csv"
SYMBOL_CHANGES = "symbolchange.csv"
EQUITY_LIST = "EQUITY_L.csv"
NIFTY500_LIST = "ind_nifty500list.csv"

CHANGE_COLUMNS = ["old", "new", "date", "valid_from", "source"]


def reference_path(root: PathLike, name: str) -> Optional[Path]:
    """Locate a reference file under ``root/reference`` (archive or store)."""
    for cand in (Path(root) / "reference" / name, Path(root) / name):
        if cand.exists():
            return cand
    return None


def _read_text(root: PathLike, name: str) -> Optional[str]:
    path = reference_path(root, name)
    if path is None:
        logger.warning("reference file %s not found under %s", name, root)
        return None
    return path.read_bytes().decode("utf-8", errors="replace")


# ---------------------------------------------------------------- ETF list
def parse_etf_list(text: str) -> FrozenSet[str]:
    """Symbols from ``eq_etfseclist.csv`` (header ``Symbol,Underlying Asset,...``)."""
    df = pd.read_csv(io.StringIO(text), dtype=str)
    df.columns = [c.strip() for c in df.columns]
    col = next(c for c in df.columns if c.lower() == "symbol")
    return frozenset(s.strip().upper() for s in df[col].dropna() if s.strip())


def load_etf_symbols(root: PathLike) -> FrozenSet[str]:
    text = _read_text(root, ETF_LIST)
    return parse_etf_list(text) if text else frozenset()


# ---------------------------------------------------------- symbol changes
def parse_symbol_changes(text: str) -> pd.DataFrame:
    """Parse ``symbolchange.csv`` -> DataFrame[old, new, date].

    The file has no header; company names may contain commas, so the last
    three fields are taken as old symbol, new symbol and change date.
    """
    rows = []
    for rec in csv.reader(io.StringIO(text)):
        if len(rec) < 4:
            continue
        old, new, when = (x.strip().upper() for x in rec[-3:])
        dt = pd.to_datetime(when, format="%d-%b-%Y", errors="coerce")
        if pd.isna(dt) or not old or not new or old == new:
            continue  # header or malformed line
        rows.append((old, new, dt))
    df = pd.DataFrame(rows, columns=["old", "new", "date"])
    return df.drop_duplicates().sort_values(["date", "old"]).reset_index(drop=True)


def load_symbol_changes(root: PathLike) -> pd.DataFrame:
    text = _read_text(root, SYMBOL_CHANGES)
    if not text:
        return pd.DataFrame(columns=["old", "new", "date"])
    return parse_symbol_changes(text)


# ------------------------------------------------------------------ spans
def compute_spans(rows: pd.DataFrame) -> pd.DataFrame:
    """(symbol, isin) -> first/last trading date and row count."""
    df = rows.loc[rows["isin"].notna(), ["symbol", "isin", "date"]]
    g = df.groupby(["symbol", "isin"], observed=True)["date"]
    out = g.agg(first="min", last="max", n="size").reset_index()
    out["symbol"] = out["symbol"].astype(str)
    out["isin"] = out["isin"].astype(str)
    return out


def isin_continuity_renames(spans: pd.DataFrame, calendar: pd.DatetimeIndex,
                            max_gap_sessions: int = 3) -> pd.DataFrame:
    """Renames implied by a symbol ending and a new one starting on the same ISIN."""
    if spans.empty:
        return pd.DataFrame(columns=["old", "new", "date"])
    cal = pd.DatetimeIndex(calendar).sort_values()
    sym_first = spans.groupby("symbol")["first"].min()
    ends = spans[spans["last"] < cal[-1]]
    starts = spans.assign(sym_first=spans["symbol"].map(sym_first))
    starts = starts[starts["first"] == starts["sym_first"]]  # brand-new symbols only
    m = ends.merge(starts, on="isin", suffixes=("_old", "_new"))
    m = m[m["symbol_old"] != m["symbol_new"]]
    if m.empty:
        return pd.DataFrame(columns=["old", "new", "date"])
    gap = cal.searchsorted(m["first_new"].to_numpy()) - cal.searchsorted(m["last_old"].to_numpy())
    m = m[(gap >= 1) & (gap <= max_gap_sessions)]
    # the old symbol must not still be trading when the new one starts
    by_sym = {s: g[["first", "last"]].to_numpy() for s, g in spans.groupby("symbol")}
    still = [bool(((by_sym[o][:, 0] <= f) & (by_sym[o][:, 1] >= f)).any())
             for o, f in zip(m["symbol_old"], m["first_new"])]
    m = m[~np.asarray(still, dtype=bool)]
    out = pd.DataFrame({"old": m["symbol_old"], "new": m["symbol_new"], "date": m["first_new"]})
    return out.drop_duplicates().reset_index(drop=True)


def attach_valid_from(changes: pd.DataFrame, spans: pd.DataFrame, calendar: pd.DatetimeIndex,
                      max_gap_sessions: int = 5) -> pd.DataFrame:
    """Add ``valid_from``: start of the old symbol's contiguous history before the rename.

    Consecutive (symbol, isin) spans are the same company when the next span
    starts within ``max_gap_sessions`` of the previous one ending (ISIN
    changes on splits happen without a gap).  Unknown history -> no limit.
    """
    out = changes.copy()
    cal = pd.DatetimeIndex(calendar).sort_values()
    by_symbol = {s: g.sort_values("first") for s, g in spans.groupby("symbol")} if len(spans) else {}
    valid = []
    for old, when in zip(out["old"], out["date"]):
        g = by_symbol.get(old)
        g = None if g is None else g[g["first"] < when]
        if g is None or g.empty:
            valid.append(pd.Timestamp("1900-01-01"))
            continue
        firsts, lasts = g["first"].to_list(), g["last"].to_list()
        start, i = firsts[-1], len(firsts) - 1
        while i > 0:
            prev_last = max(lasts[:i])
            gap = int(cal.searchsorted(firsts[i]) - cal.searchsorted(prev_last))
            if gap > max_gap_sessions:
                break
            i -= 1
            start = min(start, firsts[i])
        valid.append(start)
    out["valid_from"] = pd.to_datetime(valid)
    return out


def build_change_table(root_or_changes: Union[PathLike, pd.DataFrame], spans: Optional[pd.DataFrame] = None,
                       calendar: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
    """Combined rename table (symbolchange.csv + ISIN continuity) with ``valid_from``."""
    if isinstance(root_or_changes, pd.DataFrame):
        file_changes = root_or_changes[["old", "new", "date"]].copy()
    else:
        file_changes = load_symbol_changes(root_or_changes)
    file_changes["source"] = "symbolchange"
    parts = [file_changes]
    if spans is not None and calendar is not None and len(calendar):
        isin = isin_continuity_renames(spans, calendar)
        isin["source"] = "isin"
        parts.append(isin)
    changes = pd.concat(parts, ignore_index=True)
    changes["date"] = pd.to_datetime(changes["date"])
    changes = changes.drop_duplicates(["old", "new", "date"]).sort_values(["date", "old"])
    if spans is not None and calendar is not None and len(calendar):
        changes = attach_valid_from(changes, spans, calendar)
    else:
        changes["valid_from"] = pd.Timestamp("1900-01-01")
    return changes[CHANGE_COLUMNS].reset_index(drop=True)


def resolve_symbols(symbols: pd.Series, dates: pd.Series, changes: pd.DataFrame,
                    max_iterations: int = 20) -> pd.Series:
    """Canonical symbol for each (symbol, date) row.

    For each row, the first rename of its current label effective strictly
    after the row date (and whose ``valid_from`` <= row date) is applied;
    this repeats to follow chains (TELCO -> TATAMOTORS -> TMPV).
    """
    labels = pd.Series(symbols.astype(str).to_numpy(), index=symbols.index, dtype=object)
    if changes is None or changes.empty:
        return labels
    ch = changes.copy()
    if "valid_from" not in ch:
        ch["valid_from"] = pd.Timestamp("1900-01-01")
    ch = ch.sort_values("date")
    olds = set(ch["old"])
    dates = pd.to_datetime(pd.Series(dates.to_numpy(), index=symbols.index))
    # valid_from only guards the first hop; later hops follow an established chain.
    hopped = np.zeros(len(labels), dtype=bool)
    right = ch.rename(columns={"old": "label", "date": "eff"})[["label", "eff", "new", "valid_from"]]
    right = right.assign(label=right["label"].astype("str"), eff=pd.to_datetime(right["eff"]),
                         valid_from=pd.to_datetime(right["valid_from"]))
    for _ in range(max_iterations):
        mask = labels.isin(olds).to_numpy()
        if not mask.any():
            break
        pos_all = np.flatnonzero(mask)
        left = pd.DataFrame({"label": labels.to_numpy()[mask], "date": dates.to_numpy()[mask],
                             "pos": pos_all, "hopped": hopped[mask]})
        left["label"] = left["label"].astype("str")
        left = left.sort_values("date", kind="stable")
        merged = pd.merge_asof(left, right, left_on="date", right_on="eff", by="label",
                               direction="forward", allow_exact_matches=False)
        hit = (merged["new"].notna() & (merged["hopped"] | (merged["date"] >= merged["valid_from"]))).to_numpy()
        if not hit.any():
            break
        pos = merged["pos"].to_numpy()[hit]
        labels.iloc[pos] = merged["new"].to_numpy()[hit]
        hopped[pos] = True
    else:
        logger.warning("symbol resolution did not converge (rename cycle?)")
    return labels


# ------------------------------------------------------------- sector map
def parse_nifty500_list(text: str) -> Dict[str, str]:
    """``ind_nifty500list.csv`` -> {symbol: industry}."""
    df = pd.read_csv(io.StringIO(text), dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=["Symbol", "Industry"])
    return {s.strip().upper(): i.strip() for s, i in zip(df["Symbol"], df["Industry"])}


def extend_with_old_symbols(sectors: Dict[str, str], changes: pd.DataFrame) -> Dict[str, str]:
    """Add old symbol names that were renamed into a mapped symbol."""
    out = dict(sectors)
    for _ in range(20):  # follow chains backwards
        added = 0
        for old, new in zip(changes["old"], changes["new"]):
            if new in out and old not in out:
                out[old] = out[new]
                added += 1
        if not added:
            break
    return out


def build_sector_map(archive_root: PathLike, out_path: PathLike = "data/nse_sector_map.json") -> Dict[str, str]:
    """Build {symbol: industry} from the NIFTY 500 list and write JSON atomically."""
    text = _read_text(archive_root, NIFTY500_LIST)
    if not text:
        raise FileNotFoundError(f"{NIFTY500_LIST} missing under {archive_root}; run sync_reference()")
    sectors = extend_with_old_symbols(parse_nifty500_list(text), load_symbol_changes(archive_root))
    sectors = dict(sorted(sectors.items()))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(sectors, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    logger.info("sector map: %d symbols -> %s", len(sectors), out)
    return sectors


def load_sector_map(paths: Iterable[PathLike]) -> Dict[str, str]:
    """Load the first existing sector-map JSON from ``paths`` ({} if none)."""
    for p in paths:
        p = Path(p)
        if p.exists():
            try:
                return {str(k).upper(): str(v) for k, v in json.loads(p.read_text()).items()}
            except ValueError:
                logger.warning("invalid sector map JSON at %s", p)
    return {}


# ------------------------------------------------------- corporate actions
#: Purposes that change the price scale by a ratio stated in the text.
_BONUS_RE = re.compile(r"\bBON(?:US)?\b[^0-9/]{0,12}?(\d+)\s*:\s*(\d+)")
_SPLIT_RE = re.compile(r"SPL?I?T\D*?(\d+(?:\.\d+)?)\s*(?:/-)?\s*(?:PER\s+SH\w*)?\s*TO\s*(?:RS|RE)?\.?\s*(\d+(?:\.\d+)?)")
_CONSOL_RE = re.compile(r"CONSOL\w*\D*?(\d+(?:\.\d+)?)\s*(?:/-)?\s*TO\s*(?:RS|RE)?\.?\s*(\d+(?:\.\d+)?)")
_RIGHTS_RE = re.compile(
    r"\bR(?:IG)?HTS?\b\s*(\d+)\s*:\s*(\d+)\s*(?:@\s*(?:PREM\w*|PRM)?\s*(?:RS|RE)?\.?\s*(\d+(?:\.\d+)?))?")
#: Purposes that change the price scale by an amount only prices reveal.
_PRICE_BASED_RE = re.compile(r"DEMERGER|SCHEME OF ARRANGEMENT|CAPITAL REDUCTION|REDUCTION OF CAPITAL|RETURN OF CAPITAL")

CA_COLUMNS = ["symbol", "series", "ex_date", "purpose", "file_date"]


def parse_corporate_actions(text: str, file_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Parse a PR-zip ``Bc`` file -> DataFrame[symbol, series, ex_date, purpose, file_date].

    Handles both ``dd/mm/yyyy`` (legacy) and ISO (recent) date formats and
    purposes containing commas.
    """
    rows = []
    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    if not header:
        return pd.DataFrame(columns=CA_COLUMNS)
    cols = [h.strip().upper() for h in header]
    try:
        i_series, i_symbol, i_ex, i_purpose = (cols.index(c) for c in ("SERIES", "SYMBOL", "EX_DT", "PURPOSE"))
    except ValueError:
        logger.warning("unexpected corporate action header: %s", header)
        return pd.DataFrame(columns=CA_COLUMNS)
    for rec in reader:
        if len(rec) <= i_purpose:
            continue
        ex = rec[i_ex].strip()
        if not ex:
            continue
        purpose = ",".join(rec[i_purpose:]).strip().upper()
        rows.append((rec[i_symbol].strip().upper(), rec[i_series].strip().upper(), ex, purpose))
    df = pd.DataFrame(rows, columns=["symbol", "series", "ex_raw", "purpose"])
    iso = df["ex_raw"].str.match(r"^\d{4}-\d{2}-\d{2}$")
    df["ex_date"] = pd.NaT
    df.loc[iso, "ex_date"] = pd.to_datetime(df.loc[iso, "ex_raw"], format="%Y-%m-%d", errors="coerce")
    df.loc[~iso, "ex_date"] = pd.to_datetime(df.loc[~iso, "ex_raw"], format="%d/%m/%Y", errors="coerce")
    df["ex_date"] = pd.to_datetime(df["ex_date"])
    df["file_date"] = pd.Timestamp(file_date) if file_date is not None else pd.NaT
    return df.dropna(subset=["ex_date"])[CA_COLUMNS].reset_index(drop=True)


def classify_purpose(purpose: str) -> Dict[str, float]:
    """Interpret a corporate-action purpose.

    Returns ``{"factor": f}`` for bonus / split / consolidation (product of
    all components, f = new price scale / old), ``{"rights_new": a,
    "rights_held": b, "rights_premium": p}`` for rights, ``{"price_based": 1}``
    for demergers / capital reductions, or ``{}`` for everything else
    (dividends, meetings, buybacks, mergers).
    """
    text = purpose.upper()
    out: Dict[str, float] = {}
    factor = 1.0
    for a, b in _BONUS_RE.findall(text):
        a, b = float(a), float(b)
        if a > 0 and b > 0:
            factor *= b / (a + b)
    for old, new in _SPLIT_RE.findall(text):
        old, new = float(old), float(new)
        if old > 0 and 0 < new < old:
            factor *= new / old
    for old, new in _CONSOL_RE.findall(text):
        old, new = float(old), float(new)
        if 0 < old < new:
            factor *= new / old
    if factor != 1.0:
        out["factor"] = factor
    m = _RIGHTS_RE.search(text)
    if m:
        out.update(rights_new=float(m.group(1)), rights_held=float(m.group(2)),
                   rights_premium=float(m.group(3)) if m.group(3) else float("nan"))
    if not out and _PRICE_BASED_RE.search(text):
        out["price_based"] = 1.0
    return out


def corporate_action_events(actions: pd.DataFrame, series: Iterable[str] = ("EQ", "BE")) -> pd.DataFrame:
    """Price-relevant events, one per (symbol, ex_date, purpose), latest listing wins.

    NSE sometimes revises an ex-date (the same purpose then appears with two
    dates within a few days); only the date from the most recent file is kept.
    Columns: symbol, ex_date, purpose, factor, rights_new, rights_held,
    rights_premium, price_based.
    """
    cols = ["symbol", "ex_date", "purpose", "factor", "rights_new", "rights_held", "rights_premium", "price_based"]
    if actions is None or actions.empty:
        return pd.DataFrame(columns=cols)
    df = actions[actions["series"].isin(list(series))].copy()
    df["purpose"] = df["purpose"].str.replace(r"\s+", " ", regex=True).str.strip()
    parsed = {p: classify_purpose(p) for p in df["purpose"].unique()}
    df = df[df["purpose"].map(lambda p: bool(parsed[p]))]
    if df.empty:
        return pd.DataFrame(columns=cols)
    df["category"] = df["purpose"].map(lambda p: _category(parsed[p]))
    # a revised purpose for the same ex-date (GRUH 2012: split to RS5, then to RS2):
    # keep only rows from the latest file that lists this (symbol, ex_date, category)
    latest = df.groupby(["symbol", "ex_date", "category"])["file_date"].transform("max")
    df = df[(df["file_date"] == latest) | latest.isna()]
    df = (df.sort_values("file_date").drop_duplicates(["symbol", "ex_date", "purpose"], keep="last")
          .sort_values(["symbol", "purpose", "ex_date"]))
    # collapse revised ex-dates: same symbol+purpose within 30 days -> latest file's date
    gap = df.groupby(["symbol", "purpose"])["ex_date"].diff().dt.days
    df["cluster"] = (gap.isna() | (gap > 30)).cumsum()
    df = df.sort_values("file_date").drop_duplicates("cluster", keep="last").drop(columns=["cluster", "category"])
    for key in cols[3:]:
        df[key] = df["purpose"].map(lambda p, k=key: parsed[p].get(k, np.nan))
    return df[cols].sort_values(["ex_date", "symbol"]).reset_index(drop=True)


def _category(parsed: Dict[str, float]) -> str:
    if "factor" in parsed:
        return "ratio"
    return "rights" if "rights_new" in parsed else "price"


def load_face_values(root: PathLike) -> Dict[str, float]:
    """Current face values from ``EQUITY_L.csv`` (used for rights issue prices)."""
    text = _read_text(root, EQUITY_LIST)
    if not text:
        return {}
    df = pd.read_csv(io.StringIO(text), dtype=str)
    df.columns = [c.strip().upper() for c in df.columns]
    fv = pd.to_numeric(df.get("FACE VALUE"), errors="coerce")
    return {s.strip().upper(): float(v) for s, v in zip(df["SYMBOL"], fv) if pd.notna(v)}
