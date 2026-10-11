"""
Point-in-time quarterly results and board meetings from NSE corporate filings (tracker GT1).

Three NSE APIs list the filings, each row stamped with the time the exchange
published it:

* ``corporates-financial-results``: quarterly results from 2012.  The
  detail document is NSE's own HTML table up to mid-2018 ("Old" format) or
  an XBRL instance from Q1 FY19 ("New" format; about a quarter of Q1 FY19
  filings have neither).  It is near-empty from May 2025.
* ``integrated-filing-results``: SEBI's integrated filing, from Feb 2025
  (it overlaps the endpoint above until about April 2025).  XBRL with the
  same Ind AS tags.
* ``corporate-board-meetings``: board-meeting intimations by meeting date,
  sparse before 2016, including meetings still ahead.

Raw files mirror under ``archive/filings``, resumable and written
atomically::

    filings/lists/results/2019/2019-10-14.json          one week (Mon-Sun) per file
    filings/lists/integrated/2025/2025-08-04.json
    filings/lists/board_meetings/2016/2016-01-04.json
    filings/docs/2019/<document name>.gz                detail page or XBRL, gzipped
    filings/manifest/missing.json                       documents that answered 404

The store is kept apart from the equity store, so the engine's data
fingerprint never moves::

    fundamentals/results.parquet         one row per results filing (RESULT_COLUMNS)
    fundamentals/board_meetings.parquet  one row per intimation (BOARD_COLUMNS)
    fundamentals/parsed/2019.parquet     document parse cache per publication year
    fundamentals/manifest.json           version, per-year fingerprints, coverage, data hash

**Point in time.**  ``disseminated`` is when the exchange published the
filing (IST).  ``available_from`` is the first session whose close could
reflect it: the same day for a filing published on a session day before
15:30, else the next session.  It is NaT past the equity store's calendar.
Revisions are separate rows; ``point_in_time`` keeps the latest filing per
(canonical, period_end, consolidated) known on a date.  ``canonical`` is the
engine's name for the company: the filing's symbol resolved through the
equity store's renames, or else the name its ISIN last traded under.

**Values** are for the quarter, in INR (XBRL facts are in rupees; HTML pages
state their unit, usually lakhs):

* ``revenue``: revenue from operations (banks: total income, interest
  earned plus other income).
* ``pbt`` and ``pat``: profit before tax and profit for the period.
* ``pat_owners``: profit after minority interest; consolidated XBRL
  filings and old-format pages only.
* EPS as reported (not annualised).
* ``shares``: paid-up capital / face value.
* ``equity``: book equity at the period end, from the balance sheet that
  half-year and year-end XBRL filings tag (Q2, Q4).
* ``reserves``: reserves excluding revaluation, as of the last audited
  balance sheet, where an old-format page states it.

``status`` is one of:

* ``ok``: profit and EPS were found;
* ``partial``: some values were found;
* ``unparsed``: the document gave none;
* ``not_fetched``: the document was not downloaded;
* ``missing``: the filing has no document, or the document answered 404.

**Coverage.**  Documents are fetched only for names in the top
``DOC_TOP_N`` by the engine's universe rule (126-session median traded
value, every 21 sessions since 2012, no price floor).  That is a superset
of any point-in-time universe of that size.  Every other filing keeps its
dates, with no values.

CLI::

    python -m nse_engine.data.fundamentals sync --start 2012-01-01
    python -m nse_engine.data.fundamentals build
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import logging
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
from lxml import etree

from nse_engine.config import UniverseConfig
from nse_engine.data import archive, panel, reference
from nse_engine.universe import compute_universe_panel

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

FUNDAMENTALS_VERSION = 1  # bump when parsing changes so every year is re-parsed
API_URL = archive.HOME_URL + "/api/"
#: The API answers only with the cookies this page sets.
FILINGS_PAGE = archive.HOME_URL + "/companies-listing/corporate-filings-financial-results"
LIST_KINDS = ("results", "integrated", "board_meetings")
RESULTS_START = date(2012, 1, 1)
INTEGRATED_START = date(2025, 1, 1)
INTEGRATED_TYPE = "Integrated Filing- Financials"
INTEGRATED_PAGE_SIZE = 500
#: Board meetings are listed by meeting date; fetch this far ahead of ``end``.
BOARD_LOOKAHEAD = timedelta(days=60)
DOC_TOP_N = 500
MISSING_DOCS = "docs"
#: A filing published at or after this time of day reaches the next session's close.
MARKET_CLOSE = pd.Timedelta(hours=15, minutes=30)

VALUE_COLUMNS = ["revenue", "pbt", "pat", "pat_owners", "eps_basic", "eps_diluted", "face_value",
                 "paid_up", "shares", "equity", "reserves"]
RESULT_COLUMNS = ["canonical", "symbol", "isin", "company", "period_start", "period_end", "consolidated",
                  "audited", "disseminated", "available_from", "source", "doc_url", "status"] + VALUE_COLUMNS
BOARD_COLUMNS = ["canonical", "symbol", "isin", "company", "meeting_date", "purpose", "description",
                 "intimated", "available_from"]
#: Amounts scaled by a document's unit; EPS and face value are per share.
_AMOUNTS = ("revenue", "pbt", "pat", "pat_owners", "paid_up", "equity", "reserves")

#: XBRL local names per field, first found wins.  ``OneD`` is the quarter, ``OneI`` its end.
_XBRL_DURATION: Dict[str, Tuple[str, ...]] = {
    "revenue": ("RevenueFromOperations", "Income"),
    "pbt": ("ProfitBeforeTax", "ProfitLossFromOrdinaryActivitiesBeforeTax"),
    "pat": ("ProfitLossForPeriod", "ProfitLossForThePeriod"),
    "pat_owners": ("ProfitOrLossAttributableToOwnersOfParent",
                   "ProfitLossAfterTaxesMinorityInterestAndShareOfProfitLossOfAssociates"),
    "eps_basic": ("BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
                  "BasicEarningsPerShareAfterExtraordinaryItems",
                  "BasicEarningsLossPerShareFromContinuingOperations",
                  "BasicEarningsPerShareBeforeExtraordinaryItems"),
    "eps_diluted": ("DilutedEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
                    "DilutedEarningsPerShareAfterExtraordinaryItems",
                    "DilutedEarningsLossPerShareFromContinuingOperations",
                    "DilutedEarningsPerShareBeforeExtraordinaryItems"),
    "face_value": ("FaceValueOfEquityShareCapital",),
    "paid_up": ("PaidUpValueOfEquityShareCapital",),
}
_XBRL_INSTANT: Dict[str, Tuple[str, ...]] = {
    "equity": ("EquityAttributableToOwnersOfParent", "Equity"),
}
_XBRL_PERIOD_START = "DateOfStartOfReportingPeriod"

#: HTML row labels (lower case) per field, in order of preference.
_HTML_FIELDS: Dict[str, Tuple[str, ...]] = {
    "revenue": (r"^revenue from operations", r"^total income from operations",
                r"net sales\s*/\s*income from operations", r"^net income from sales", r"^total income$"),
    "pbt": (r"^profit.*\bbefore tax\b",),
    "pat": (r"^net profit.*\bfor the period\b",),
    "pat_owners": (r"after taxes,? minority interest",),
    "face_value": (r"^face value",),
    "paid_up": (r"^paid-?up equity share capital",),
    "reserves": (r"^reserves? excluding revaluation",),
}
#: EPS rows come before and after extraordinary items: the last one reported wins.
_HTML_LAST: Dict[str, str] = {"eps_basic": r"\bbasic\b", "eps_diluted": r"\bdiluted\b"}
_HTML_UNITS = ((r"lakh|lac", 1e5), (r"crore", 1e7), (r"million", 1e6), (r"thousand", 1e3))

_CONSOLIDATED = {"consolidated": True, "non-consolidated": False, "standalone": False}
_AUDITED = {"audited": True, "un-audited": False, "unaudited": False}


# ------------------------------------------------------------------ fetch
def _api_session() -> requests.Session:
    """The archive's session, warmed on the filings page as well."""
    s = archive.BhavcopyArchive._new_session()
    try:
        s.get(FILINGS_PAGE, timeout=15)
    except requests.RequestException as exc:
        logger.debug("filings page warm-up failed: %s", exc)
    return s


def weeks(start: date, end: date) -> List[date]:
    """Mondays of the weeks that overlap [start, end]."""
    first = start - timedelta(days=start.weekday())
    return [first + timedelta(weeks=i) for i in range((end - first).days // 7 + 1)]


def doc_key(url: str, disseminated: pd.Timestamp) -> str:
    """Path of a filing's document under ``filings/docs``."""
    return f"{disseminated.year}/{url.rsplit('/', 1)[-1]}.gz"


def _looks_like_document(name: str, content: bytes) -> bool:
    head = content[:4096].lower()
    if name.endswith(".xml"):
        return b"xbrl" in head
    return b"<table" in content.lower()


class FilingsArchive:
    """Resumable mirror of NSE's filing lists and detail documents under ``root``."""

    def __init__(self, root: PathLike, api_per_second: float = 0.67, docs_per_second: float = 2.0) -> None:
        self.root = Path(root)
        self.api = archive.BhavcopyArchive(self.root, api_per_second, session_factory=_api_session)
        self.docs = archive.BhavcopyArchive(self.root, docs_per_second)

    def list_path(self, kind: str, week: date) -> Path:
        return self.root / "lists" / kind / str(week.year) / f"{week.isoformat()}.json"

    def _get_json(self, endpoint: str, params: Dict[str, object]) -> Optional[object]:
        status, content = self.api.fetch(API_URL + endpoint + "?" + urlencode(params))
        if status != 200:
            return None
        try:
            return json.loads(content)
        except ValueError:
            logger.warning("%s: not JSON (%d bytes); new session next call", endpoint, len(content))
            self.api.reset_session()
            return None

    def fetch_week(self, kind: str, week: date) -> Optional[list]:
        """One week's list; None when NSE did not answer with one."""
        span = {"index": "equities", "from_date": week.strftime("%d-%m-%Y"),
                "to_date": (week + timedelta(days=6)).strftime("%d-%m-%Y")}
        if kind == "results":
            got = self._get_json("corporates-financial-results", dict(span, period="Quarterly"))
            return got if isinstance(got, list) else None
        if kind == "board_meetings":
            got = self._get_json("corporate-board-meetings", span)
            return got if isinstance(got, list) else None
        rows: list = []
        for page in range(1, 1000):
            got = self._get_json("integrated-filing-results", dict(span, size=INTEGRATED_PAGE_SIZE, page=page))
            if not isinstance(got, dict):
                return None
            data = got.get("data") or []
            rows.extend(data)
            if not data or len(rows) >= int(got.get("totalCount") or 0):
                break
        return rows

    def sync_lists(self, start: date, end: date, today: Optional[date] = None) -> Dict[str, int]:
        """Fetch every week not yet saved; weeks too recent to be complete are re-fetched."""
        today = today or date.today()
        counts: Dict[str, int] = {}
        for kind in LIST_KINDS:
            lo = max(start, INTEGRATED_START) if kind == "integrated" else start
            hi = end + BOARD_LOOKAHEAD if kind == "board_meetings" else end
            fetched = failed = 0
            for week in weeks(lo, hi):
                path = self.list_path(kind, week)
                if path.exists() and week + timedelta(days=6) < today - archive.RECENT_GRACE:
                    continue
                rows = self.fetch_week(kind, week)
                if rows is None:
                    failed += 1
                    logger.warning("%s %s: no list; the next sync retries it", kind, week)
                    continue
                archive.atomic_write(path, json.dumps(rows).encode())
                fetched += 1
            counts[kind] = fetched
            logger.info("lists %s: %d weeks fetched, %d failed", kind, fetched, failed)
        return counts

    def sync_docs(self, filings: pd.DataFrame, log_every: int = 1000) -> Dict[str, int]:
        """Download the documents of ``filings`` (``doc_url``, ``disseminated``) not yet saved."""
        missing = self.docs.missing.setdefault(MISSING_DOCS, set())
        counts: Counter = Counter()
        todo = filings.dropna(subset=["doc_url"]).sort_values("disseminated")
        for i, (url, stamp) in enumerate(zip(todo["doc_url"], todo["disseminated"]), 1):
            key = doc_key(url, stamp)
            path = self.root / "docs" / key
            if path.exists():
                counts["have"] += 1
            elif key in missing:
                counts["missing"] += 1
            else:
                status, content = self.docs.fetch(url)
                if status == 200 and _looks_like_document(url, content):
                    archive.atomic_write(path, gzip.compress(content))
                    counts["fetched"] += 1
                elif status == 404:
                    missing.add(key)
                    counts["missing"] += 1
                else:
                    counts["failed"] += 1
            if i % log_every == 0:
                self.docs.save_missing()
                logger.info("docs %d/%d: %s", i, len(todo), dict(counts))
        self.docs.save_missing()
        return dict(counts)


# ------------------------------------------------------------------ lists
def _stamp(values: pd.Series, fmt: str = "%d-%b-%Y %H:%M:%S") -> pd.Series:
    return pd.to_datetime(values, format=fmt, errors="coerce").astype("datetime64[ns]")


def _flag(values: pd.Series, mapping: Dict[str, bool]) -> pd.Series:
    return values.astype("string").str.strip().str.lower().map(mapping).astype("boolean")


def _column(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df else pd.Series(pd.NA, index=df.index, dtype="object")


def _list_rows(root: Path, kind: str) -> List[dict]:
    rows: List[dict] = []
    for path in sorted((root / "lists" / kind).glob("*/*.json")):
        rows.extend(json.loads(path.read_text()))
    return rows


def read_filings(root: PathLike) -> pd.DataFrame:
    """Results filings from both list endpoints; integrated duplicates of a listed filing dropped."""
    root = Path(root)
    frames = []
    old = pd.DataFrame(_list_rows(root, "results"))
    if len(old):
        xbrl = _column(old, "xbrl").where(~_column(old, "xbrl").astype("string").str.endswith("/-", na=True))
        frames.append(pd.DataFrame({
            "source": "results", "symbol": old["symbol"], "isin": _column(old, "isin"),
            "company": _column(old, "companyName"),
            "period_start": _stamp(old["fromDate"], "%d-%b-%Y"), "period_end": _stamp(old["toDate"], "%d-%b-%Y"),
            "consolidated": _flag(old["consolidated"], _CONSOLIDATED), "audited": _flag(old["audited"], _AUDITED),
            "disseminated": pd.concat([_stamp(old["broadCastDate"]), _stamp(_column(old, "exchdisstime"))],
                                      axis=1).max(axis=1),
            "doc_url": xbrl.fillna(_column(old, "resultDetailedDataLink")),
        }))
    new = pd.DataFrame([r for r in _list_rows(root, "integrated") if r.get("type") == INTEGRATED_TYPE])
    if len(new):
        frames.append(pd.DataFrame({
            "source": "integrated", "symbol": new["symbol"], "isin": pd.NA, "company": _column(new, "smName"),
            "period_start": pd.NaT, "period_end": _stamp(new["qe_Date"], "%d-%b-%Y"),
            "consolidated": _flag(new["consolidated"], _CONSOLIDATED), "audited": _flag(new["audited"], _AUDITED),
            "disseminated": pd.concat([_stamp(new["broadcast_Date"]), _stamp(_column(new, "creation_Date"))],
                                      axis=1).max(axis=1),
            "doc_url": _column(new, "xbrl"),
        }))
    if not frames:
        return pd.DataFrame(columns=["source", "symbol", "isin", "company", "period_start", "period_end",
                                     "consolidated", "audited", "disseminated", "doc_url"])
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["symbol", "period_end", "disseminated"])
    out["doc_url"] = out["doc_url"].where(out["doc_url"].astype("string").str.startswith("http", na=False))
    out = out.drop_duplicates(subset=["source", "symbol", "period_end", "consolidated", "disseminated", "doc_url"])
    key = (out["symbol"].astype(str) + "|" + out["period_end"].dt.strftime("%Y%m%d") + "|"
           + out["consolidated"].astype("string").fillna("NA") + "|" + out["disseminated"].dt.strftime("%Y%m%d"))
    listed = set(key[out["source"] == "results"])
    out = out[~((out["source"] == "integrated") & key.isin(listed))]
    return out.reset_index(drop=True)


def read_board_meetings(root: PathLike) -> pd.DataFrame:
    df = pd.DataFrame(_list_rows(Path(root), "board_meetings"))
    if not len(df):
        return pd.DataFrame(columns=[c for c in BOARD_COLUMNS if c not in ("canonical", "available_from")])
    out = pd.DataFrame({
        "symbol": df["bm_symbol"], "isin": _column(df, "sm_isin"), "company": _column(df, "sm_name"),
        "meeting_date": _stamp(df["bm_date"], "%d-%b-%Y"), "purpose": _column(df, "bm_purpose"),
        "description": _column(df, "bm_desc"),
        "intimated": pd.concat([_stamp(df["bm_timestamp"]), _stamp(_column(df, "sysTime"))], axis=1).max(axis=1),
    })
    out = out.dropna(subset=["symbol", "meeting_date"])
    return out.drop_duplicates(subset=["symbol", "meeting_date", "purpose", "intimated"]).reset_index(drop=True)


def canonical_symbols(symbols: pd.Series, isins: pd.Series, dates: pd.Series, store_dir: PathLike) -> pd.Series:
    """Engine names: the symbol through the store's renames, else the name its ISIN last traded under."""
    changes = panel.load_change_table(store_dir)
    spans = pd.read_parquet(Path(store_dir) / "spans.parquet").sort_values("last").reset_index(drop=True)
    span_names = reference.resolve_symbols(spans["symbol"].astype(str), spans["last"], changes)
    by_isin = dict(zip(spans["isin"].astype(str), span_names))
    names = reference.resolve_symbols(symbols.astype(str).reset_index(drop=True),
                                      pd.Series(dates.to_numpy()), changes)
    names.index = symbols.index
    fallback = isins.astype("string").map(by_isin)
    use = ~names.isin(set(span_names)) & fallback.notna()
    return names.where(~use, fallback)


def available_from(stamps: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """First session whose close could reflect a publication at ``stamps`` (IST)."""
    cal = np.asarray(calendar.values, dtype="datetime64[ns]")
    day = stamps.dt.normalize().to_numpy(dtype="datetime64[ns]")
    left = np.searchsorted(cal, day, side="left")
    right = np.searchsorted(cal, day, side="right")
    session_day = (left < len(cal)) & (cal[np.minimum(left, len(cal) - 1)] == day)
    before_close = ((stamps - stamps.dt.normalize()) < MARKET_CLOSE).to_numpy()
    pos = np.where(session_day & before_close, left, right)
    out = np.where(pos < len(cal), cal[np.minimum(pos, len(cal) - 1)], np.datetime64("NaT"))
    out[stamps.isna().to_numpy()] = np.datetime64("NaT")
    return pd.Series(out, index=stamps.index, dtype="datetime64[ns]")


def doc_universe(store_dir: PathLike, top_n: int = DOC_TOP_N, start: str = "2012-01-01") -> Set[str]:
    """Canonical names ever in the top ``top_n`` at a universe refresh since ``start``, with no price floor."""
    data = panel.load_market_data(store_dir, start, date.today())
    cfg = UniverseConfig(top_n_liquid=top_n, min_price_inr=0.0)
    return {s for members in compute_universe_panel(data, cfg).refresh.values() for s in members}


# ------------------------------------------------------------------ parse
def _number(text: Optional[str]) -> float:
    try:
        return float(str(text).replace(",", "").strip())
    except (TypeError, ValueError):
        return float("nan")


def parse_xbrl(content: bytes) -> Dict[str, object]:
    """Quarter values from an NSE results XBRL instance (facts in rupees)."""
    parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=True)
    facts: Dict[Tuple[str, str], str] = {}
    for el in etree.fromstring(content, parser=parser).iter():
        ctx = el.get("contextRef") if isinstance(el.tag, str) else None
        if ctx in ("OneD", "OneI"):
            facts.setdefault((etree.QName(el).localname, ctx), el.text or "")
    out: Dict[str, object] = {}
    for fields, ctx in ((_XBRL_DURATION, "OneD"), (_XBRL_INSTANT, "OneI")):
        for field, names in fields.items():
            out[field] = next((v for v in (_number(facts.get((n, ctx))) for n in names) if np.isfinite(v)),
                              float("nan"))
    out["period_start"] = pd.to_datetime(facts.get((_XBRL_PERIOD_START, "OneD")), errors="coerce")
    return out


def _cells(table: pd.DataFrame) -> List[str]:
    """Header and body cells of a parsed table, as text."""
    return [str(c) for c in table.columns] + [str(v) for v in table.to_numpy().ravel()]


def parse_html(content: bytes) -> Dict[str, object]:
    """Quarter values from an NSE results page ("Old" format): Description | Amount (unit)."""
    tables = pd.read_html(io.StringIO(content.decode("utf-8", "replace")))
    pairs = [t for t in tables if t.shape[1] == 2 and any("amount" in c.lower() for c in _cells(t))]
    out: Dict[str, object] = {f: float("nan") for f in list(_HTML_FIELDS) + list(_HTML_LAST)}
    out["period_start"] = pd.NaT
    if not pairs:
        return out
    table = max(pairs, key=len)
    header = " ".join(c for c in _cells(table) if "amount" in c.lower())
    unit = next((scale for pattern, scale in _HTML_UNITS if re.search(pattern, header, re.I)), float("nan"))
    labels = table.iloc[:, 0].astype(str).str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
    values = table.iloc[:, 1].map(_number)
    for field, patterns in _HTML_FIELDS.items():
        for pattern in patterns:
            hit = values[labels.str.contains(pattern, regex=True) & values.notna()]
            if len(hit):
                out[field] = float(hit.iloc[0])
                break
    for field, pattern in _HTML_LAST.items():
        hit = values[labels.str.contains(pattern, regex=True) & values.notna()]
        if len(hit):
            out[field] = float(hit.iloc[-1])
    for field in _AMOUNTS:
        if field in out:
            out[field] = float(out[field]) * unit
    return out


def parse_document(name: str, content: bytes) -> Dict[str, object]:
    """Values of one document (XBRL by its ``.xml`` name, else an HTML page); {} when unreadable."""
    try:
        return parse_xbrl(content) if name.endswith(".xml") else parse_html(content)
    except (etree.XMLSyntaxError, ValueError, IndexError) as exc:
        logger.debug("%s: unreadable (%s)", name, exc)
        return {}


def _fingerprint(files: Sequence[Path]) -> str:
    h = hashlib.sha256(f"fundamentals-v{FUNDAMENTALS_VERSION}".encode())
    for p in files:
        h.update(f"{p.name}:{p.stat().st_size}".encode())
    return h.hexdigest()[:20]


def _year_docs(root: Path, year: int) -> List[Path]:
    return sorted(p for p in (root / "docs" / str(year)).glob("*.gz") if not p.name.startswith("."))


def _parse_year(root: str, out_dir: str, year: int) -> Dict[str, object]:
    rows = []
    for path in _year_docs(Path(root), year):
        values = parse_document(path.name[:-3], gzip.decompress(path.read_bytes()))
        rows.append(dict(values, doc=f"{year}/{path.name}"))
    df = pd.DataFrame(rows, columns=["doc", "period_start"] + [f for f in VALUE_COLUMNS if f != "shares"])
    target = Path(out_dir) / "parsed" / f"{year}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target, index=False)
    found = df[["pat", "eps_basic"]].notna().all(axis=1).sum() if len(df) else 0
    return {"year": year, "docs": len(df), "ok": int(found)}


# ------------------------------------------------------------------ build
def _status(results: pd.DataFrame, have_doc: pd.Series, missing: Set[str]) -> pd.Series:
    core = results[["pat", "eps_basic"]].notna().all(axis=1)
    some = results[[c for c in VALUE_COLUMNS if c != "shares"]].notna().any(axis=1)
    status = np.select([results["doc_url"].isna() | results["doc"].isin(missing), ~have_doc, core, some],
                       ["missing", "not_fetched", "ok", "partial"], default="unparsed")
    return pd.Series(status, index=results.index)


def _frame_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()).hexdigest()


def build_fundamentals(archive_root: PathLike, store_dir: PathLike, out_dir: PathLike,
                       workers: int = 4) -> Dict[str, object]:
    """Build (or update) the fundamentals store; a year's documents are re-parsed only when they changed."""
    root, out = Path(archive_root) / "filings", Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest.get("version") != FUNDAMENTALS_VERSION:
        manifest = {"version": FUNDAMENTALS_VERSION, "years": {}}
    years = sorted(int(p.name) for p in (root / "docs").glob("[0-9]" * 4))
    todo = [y for y in years if manifest["years"].get(str(y), {}).get("fingerprint")
            != _fingerprint(_year_docs(root, y))]
    logger.info("fundamentals: %d document years, %d to parse", len(years), len(todo))
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for info in pool.map(_parse_year, [str(root)] * len(todo), [str(out)] * len(todo), todo):
            info["fingerprint"] = _fingerprint(_year_docs(root, info["year"]))
            manifest["years"][str(info["year"])] = info
            logger.info("year %s: %s", info["year"], info)

    filings = read_filings(root)
    filings["canonical"] = canonical_symbols(filings["symbol"], filings["isin"], filings["disseminated"], store_dir)
    filings["doc"] = [doc_key(u, t) if isinstance(u, str) else None
                      for u, t in zip(filings["doc_url"], filings["disseminated"])]
    parsed_files = [out / "parsed" / f"{y}.parquet" for y in years if (out / "parsed" / f"{y}.parquet").exists()]
    parsed = (pd.concat([pd.read_parquet(p) for p in parsed_files], ignore_index=True) if parsed_files
              else pd.DataFrame(columns=["doc", "period_start"] + [f for f in VALUE_COLUMNS if f != "shares"]))
    parsed = parsed.rename(columns={"period_start": "doc_period_start"})
    results = filings.merge(parsed, on="doc", how="left")
    missing = archive.BhavcopyArchive(root).missing.get(MISSING_DOCS, set())
    results["status"] = _status(results, results["doc"].isin(set(parsed["doc"])), missing)
    quarter_start = results["period_end"] - pd.offsets.MonthBegin(3)
    results["period_start"] = (results["period_start"].fillna(results["doc_period_start"]).fillna(quarter_start))
    for col in ("face_value", "paid_up"):
        results[col] = results[col].where(results[col] > 0)
    results["reserves"] = results["reserves"].where(results["reserves"] != 0)  # 0.00 = not stated
    results["shares"] = results["paid_up"] / results["face_value"]
    calendar = pd.DatetimeIndex(pd.read_parquet(Path(store_dir) / "calendar.parquet")["date"])
    results["available_from"] = available_from(results["disseminated"], calendar)
    results = (results[RESULT_COLUMNS]
               .sort_values(["disseminated", "symbol", "period_end", "consolidated", "doc_url"], kind="stable")
               .reset_index(drop=True))
    for col in ("canonical", "symbol", "isin", "company", "source", "doc_url", "status"):
        results[col] = results[col].astype("string")
    results.to_parquet(out / "results.parquet", index=False)

    board = read_board_meetings(root)
    board["canonical"] = canonical_symbols(board["symbol"], board["isin"], board["intimated"].fillna(
        board["meeting_date"]), store_dir)
    board["available_from"] = available_from(board["intimated"], calendar)
    board = (board[BOARD_COLUMNS].sort_values(["meeting_date", "symbol", "intimated"], kind="stable")
             .reset_index(drop=True))
    for col in ("canonical", "symbol", "isin", "company", "purpose", "description"):
        board[col] = board[col].astype("string")
    board.to_parquet(out / "board_meetings.parquet", index=False)

    manifest["results_rows"] = len(results)
    manifest["board_rows"] = len(board)
    manifest["coverage"] = coverage(results)
    manifest["data_hash"] = hashlib.sha256(json.dumps(
        {"version": FUNDAMENTALS_VERSION, "results": _frame_hash(results), "board_meetings": _frame_hash(board)},
        sort_keys=True).encode()).hexdigest()[:16]
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest


def coverage(results: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """Per publication year: filings, companies, and filings by ``status``."""
    out: Dict[str, Dict[str, int]] = {}
    for year, g in results.groupby(results["disseminated"].dt.year):
        row = {"filings": len(g), "companies": int(g["canonical"].nunique())}
        row.update({str(k): int(v) for k, v in g["status"].value_counts().items()})
        out[str(int(year))] = row
    return out


# ------------------------------------------------------------- read
def load_results(store_dir: PathLike) -> pd.DataFrame:
    return pd.read_parquet(Path(store_dir) / "results.parquet")


def load_board_meetings(store_dir: PathLike) -> pd.DataFrame:
    return pd.read_parquet(Path(store_dir) / "board_meetings.parquet")


def point_in_time(results: pd.DataFrame, as_of: Union[str, pd.Timestamp]) -> pd.DataFrame:
    """Filings known at the close of ``as_of``: the latest per (canonical, period_end, consolidated)."""
    known = results[results["available_from"] <= pd.Timestamp(as_of)]
    return (known.sort_values("disseminated", kind="stable")
            .drop_duplicates(["canonical", "period_end", "consolidated"], keep="last").reset_index(drop=True))


def data_hash(store_dir: PathLike) -> str:
    return json.loads((Path(store_dir) / "manifest.json").read_text()).get("data_hash", "")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="NSE results filings and board meetings (point in time)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sync", help="mirror the filing lists, then the documents of the liquid names")
    s.add_argument("--start", default=RESULTS_START.isoformat())
    s.add_argument("--end", default=date.today().isoformat())
    s.add_argument("--top", type=int, default=DOC_TOP_N, help="fetch documents for the top-N universe")
    s.add_argument("--lists-only", action="store_true")
    b = sub.add_parser("build", help="parse the documents and write the store")
    b.add_argument("--out", default="data/nse_engine/fundamentals")
    b.add_argument("--workers", type=int, default=4)
    for q in (s, b):
        q.add_argument("--archive", default="data/nse_engine/archive")
        q.add_argument("--store", default="data/nse_engine/store", help="equity store (renames, calendar)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cmd == "build":
        manifest = build_fundamentals(args.archive, args.store, args.out, args.workers)
        logger.info("done: data hash %s, %d results, %d board meetings", manifest["data_hash"],
                    manifest["results_rows"], manifest["board_rows"])
        return 0
    filings_root = Path(args.archive) / "filings"
    mirror = FilingsArchive(filings_root)
    mirror.sync_lists(archive.to_date(args.start), archive.to_date(args.end))
    if args.lists_only:
        return 0
    filings = read_filings(filings_root)
    filings["canonical"] = canonical_symbols(filings["symbol"], filings["isin"], filings["disseminated"], args.store)
    names = doc_universe(args.store, args.top)
    wanted = filings[filings["canonical"].isin(names)]
    logger.info("documents: %d filings of %d names (top %d)", len(wanted), wanted["canonical"].nunique(), args.top)
    logger.info("docs done: %s", mirror.sync_docs(wanted))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
