"""
Resumable downloader for NSE daily archives (survivorship-free raw data).

``BhavcopyArchive`` mirrors four daily file kinds plus a few reference lists
into a local directory with deterministic names (``fo``, the F&O bhavcopy, is
fetched only when asked for with ``--kinds fo``, as is ``participant``)::

    root/equity/2013/cm20130115.csv.zip          legacy bhavcopy (<= 2024-07-05)
    root/equity/2024/udiff20240708.csv.zip       UDiFF bhavcopy (>= 2024-07-08)
    root/delivery/2013/mto20130115.dat           security-wise delivery (MTO)
    root/indices/2013/ind20130115.csv            ind_close_all index closes
    root/corpact/2013/bc20130115.csv             corporate actions (Bc file of the PR zip)
    root/corpact/2008/bc20081031.csv             2006-09: a month of NSE's corporate-actions API in Bc form
    root/reference/corpact_api/2011-02.json      the API's months as fetched (face-value history)
    root/fo/2013/fo20130115.csv.zip              F&O bhavcopy, legacy (<= 2024-07-05)
    root/fo/2024/udiff_fo20240708.csv.zip        F&O bhavcopy, UDiFF (>= 2024-07-08)
    root/participant/2026/poi20261001.csv        participant-wise F&O open interest (from 2012)
    root/reference/{eq_etfseclist,symbolchange,EQUITY_L,ind_nifty500list}.csv
    root/manifest/missing.json                   weekday 404s (holidays) per kind

A 404 on a candidate session (weekday or Saturday) is recorded as a holiday for that kind so reruns do not
refetch it.  Writes are atomic (temp file + rename), requests are rate
limited, and 403/429/5xx responses are retried with a fresh session.

CLI::

    python -m nse_engine.data.archive --start 2012-01-01 --end 2026-09-14 \
        --root data/nse_engine/archive
    python -m nse_engine.data.archive --start 2006-01-01 --end 2009-12-31 --backfill-corpact

PR zips carry no Bc file before 2010, so ``--backfill-corpact`` takes 2006-09 from NSE's
corporate-actions API, a month per file, in the Bc file's columns (tracker LN-T28b).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import re
import signal
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import requests

logger = logging.getLogger(__name__)

ARCHIVE_HOST = "https://nsearchives.nseindia.com"
HOME_URL = "https://www.nseindia.com"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

#: First session published only in UDiFF format (legacy zips stop 2024-07-05).
UDIFF_START = date(2024, 7, 8)
#: Around the boundary a 404 in one format is retried in the other.
FORMAT_FALLBACK_WINDOW = timedelta(days=180)
#: 404s this close to today may just be "not published yet": not holidays.
RECENT_GRACE = timedelta(days=3)

KINDS = ("equity", "delivery", "indices", "corpact")
#: Not in the default sync: the options research (tracker O2) asks for it.
FO_KIND = "fo"
#: Participant-wise open interest (client / DII / FII / pro), tracker OD2; also opt-in.
PARTICIPANT_KIND = "participant"

REFERENCE_FILES: Dict[str, str] = {
    "eq_etfseclist.csv": "/content/equities/eq_etfseclist.csv",
    "symbolchange.csv": "/content/equities/symbolchange.csv",
    "EQUITY_L.csv": "/content/equities/EQUITY_L.csv",
    "ind_nifty500list.csv": "/content/indices/ind_nifty500list.csv",
}

#: Sunday sessions (muhurat trading).  Saturdays are always probed because NSE
#: held many Saturday sessions (e.g. 2012-01-07, 2012-03-03, budget days, DR drills).
SPECIAL_SUNDAY_SESSIONS: Tuple[date, ...] = (
    date(2013, 11, 3), date(2016, 10, 30), date(2019, 10, 27), date(2023, 11, 12),
)

DateLike = Union[date, datetime, str]


@dataclass(frozen=True)
class RemoteFile:
    """One candidate remote file for a (kind, date)."""

    kind: str
    fmt: str  # legacy | udiff | mto | full | indices | pr
    url: str
    path: Path  # local destination


def to_date(value: DateLike) -> date:
    """Coerce a date, datetime or ISO string to ``date``."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def equity_format_for(d: date) -> str:
    """Primary bhavcopy format for a session date: ``legacy`` or ``udiff``."""
    return "udiff" if d >= UDIFF_START else "legacy"


def candidate_files(root: Path, kind: str, d: date) -> List[RemoteFile]:
    """Remote candidates for ``kind`` on ``d`` in the order they are tried.

    Equity uses the format implied by the date, then (near the 2024-07
    boundary) the other format.  Delivery uses MTO files, falling back to the
    ``sec_bhavdata_full`` file.
    """
    y, ymd = d.strftime("%Y"), d.strftime("%Y%m%d")
    if kind == "equity":
        mon = d.strftime("%b").upper()
        legacy = RemoteFile(
            kind, "legacy",
            f"{ARCHIVE_HOST}/content/historical/EQUITIES/{y}/{mon}/cm{d:%d}{mon}{y}bhav.csv.zip",
            root / "equity" / y / f"cm{ymd}.csv.zip",
        )
        udiff = RemoteFile(
            kind, "udiff",
            f"{ARCHIVE_HOST}/content/cm/BhavCopy_NSE_CM_0_0_0_{ymd}_F_0000.csv.zip",
            root / "equity" / y / f"udiff{ymd}.csv.zip",
        )
        first, second = (udiff, legacy) if equity_format_for(d) == "udiff" else (legacy, udiff)
        near = abs(d - UDIFF_START) <= FORMAT_FALLBACK_WINDOW
        return [first, second] if near else [first]
    if kind == FO_KIND:
        mon = d.strftime("%b").upper()
        legacy = RemoteFile(
            kind, "legacy",
            f"{ARCHIVE_HOST}/content/historical/DERIVATIVES/{y}/{mon}/fo{d:%d}{mon}{y}bhav.csv.zip",
            root / "fo" / y / f"fo{ymd}.csv.zip",
        )
        udiff = RemoteFile(
            kind, "udiff",
            f"{ARCHIVE_HOST}/content/fo/BhavCopy_NSE_FO_0_0_0_{ymd}_F_0000.csv.zip",
            root / "fo" / y / f"udiff_fo{ymd}.csv.zip",
        )
        first, second = (udiff, legacy) if equity_format_for(d) == "udiff" else (legacy, udiff)
        return [first, second] if abs(d - UDIFF_START) <= FORMAT_FALLBACK_WINDOW else [first]
    if kind == PARTICIPANT_KIND:
        return [RemoteFile(kind, "participant",
                           f"{ARCHIVE_HOST}/content/nsccl/fao_participant_oi_{d:%d%m%Y}.csv",
                           root / "participant" / y / f"poi{ymd}.csv")]
    if kind == "delivery":
        dmy = d.strftime("%d%m%Y")
        return [
            RemoteFile(kind, "mto", f"{ARCHIVE_HOST}/archives/equities/mto/MTO_{dmy}.DAT",
                       root / "delivery" / y / f"mto{ymd}.dat"),
            RemoteFile(kind, "full", f"{ARCHIVE_HOST}/products/content/sec_bhavdata_full_{dmy}.csv",
                       root / "delivery" / y / f"full{ymd}.csv"),
        ]
    if kind == "indices":
        return [RemoteFile(kind, "indices",
                           f"{ARCHIVE_HOST}/content/indices/ind_close_all_{d:%d%m%Y}.csv",
                           root / "indices" / y / f"ind{ymd}.csv")]
    if kind == "corpact":
        return [RemoteFile(kind, "pr",
                           f"{ARCHIVE_HOST}/archives/equities/bhavcopy/pr/PR{d:%d%m%y}.zip",
                           root / "corpact" / y / f"bc{ymd}.csv")]
    raise ValueError(f"unknown kind {kind!r}")


def extract_bc_member(content: bytes) -> Optional[bytes]:
    """Return the corporate-action ``Bc*.csv`` member of a PR zip (b"" if absent, None if bad zip)."""
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for name in zf.namelist():
                base = name.rsplit("/", 1)[-1].lower()
                if base.startswith("bc") and base.endswith(".csv"):
                    return zf.read(name)
    except zipfile.BadZipFile:
        return None
    return b""


def transform_content(cand: RemoteFile, content: bytes) -> Optional[bytes]:
    """Bytes to store for a downloaded candidate (None if unusable)."""
    if cand.fmt == "pr":
        if not content.startswith(b"PK"):
            return None
        member = extract_bc_member(content)
        if member is None:
            return None
        return member or b"SERIES,SYMBOL,SECURITY,RECORD_DT,BC_STRT_DT,BC_END_DT,EX_DT,ND_STRT_DT,ND_END_DT,PURPOSE\n"
    return content if looks_valid(content, cand.path) else None


def session_dates(start: date, end: date, extra: Iterable[date] = SPECIAL_SUNDAY_SESSIONS,
                  include_saturdays: bool = True) -> List[date]:
    """Candidate sessions in [start, end]: every day, weekends included (``include_saturdays``).

    NSE has held sessions on Saturdays (Budget days, drills) and on Sundays
    (Muhurat; the Union Budget of Sun 1 Feb 2026, which a Saturday-only probe
    missed: tracker LN-T7).  A weekend day without a bhavcopy is recorded as
    missing like any holiday, so it costs one request once.
    """
    out: Set[date] = {start + timedelta(days=i) for i in range((end - start).days + 1)}
    last_day = 6 if include_saturdays else 4
    out = {d for d in out if d.weekday() <= last_day}
    out |= {d for d in extra if start <= d <= end}
    return sorted(out)


def looks_valid(content: bytes, local_path: Path) -> bool:
    """Reject HTML error pages served with status 200 and corrupt zips."""
    if not content:
        return False
    if local_path.name.endswith(".zip"):
        if not content.startswith(b"PK"):
            return False
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                return len(zf.namelist()) > 0
        except zipfile.BadZipFile:
            return False
    head = content[:200].lstrip().lower()
    return not (head.startswith(b"<!doctype") or head.startswith(b"<html"))


#: NSE's corporate-actions API (the website's "Corporate Actions" page).  It reaches back past 2010, where
#: the PR zip has no Bc file.  Its records carry today's symbol and today's face value.
CORPACT_API = HOME_URL + "/api/corporates-corporateActions?index=equities&from_date={start}&to_date={end}"
#: Raw API months, kept as fetched: later months give each company's splits, hence its earlier face value.
CORPACT_API_CACHE = Path("reference") / "corpact_api"
#: Records whose purpose names a split without its amounts, checked against raw prices (17 Oct 2026): written
#: without ratios, so the store measures the factor from prices (reference.classify_purpose's hint).
CORPACT_FIXES = {
    ("SUNTV", "2007-07-23"): "AGM/BONUS/FV SPLIT/DIV-20%",     # "Agm/Spl/Bon-1:1/Div-20%": close 1758.75 -> 436.60
    ("NIRMA", "2007-03-14"): "FV SPLIT/INT DIV-80%",           # "Sch-Fv Spl-Int Div-80%": close 331.20 -> 173.35
}
BC_HEADER = ["SERIES", "SYMBOL", "SECURITY", "RECORD_DT", "BC_STRT_DT", "BC_END_DT", "EX_DT", "ND_STRT_DT",
             "ND_END_DT", "PURPOSE"]
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
#: "Spl-Rs10 To Rs2", "Spl-Rs 5/ To 2/": a face-value split written as "Spl", which the Bc parser reads as
#: "special" (its Bc files spell splits out), so it is rewritten as "SPLIT RS 10 TO RS 2".
_SPL_SPLIT_RE = re.compile(r"(?<![A-Z])SPL\W*(?:FROM\W*)?R[SE]\.?\s*(\d+(?:\.\d+)?)\W*TO\W*(?:R[SE]\.?\s*)?"
                           r"(\d+(?:\.\d+)?)", re.IGNORECASE)


def _api_date(value: Optional[str]) -> Optional[date]:
    try:
        return datetime.strptime(str(value).strip(), "%d-%b-%Y").date()
    except (TypeError, ValueError):
        return None


def _bc_date(value: Optional[str]) -> str:
    """The API's '27-Oct-2006' as a Bc file's '27/10/2006' ('' when absent)."""
    d = _api_date(value)
    return d.strftime("%d/%m/%Y") if d else ""


def bc_purpose(subject: str, face_value: Optional[float]) -> str:
    """The API's purpose with dividends stated as a share of face value ("Final Dividend-200%") written in
    rupees a share ("Final Dividend-RS 10.00 PER SHARE"), as Bc files state them; anything else unchanged."""
    text = re.sub(r"\s+", " ", str(subject or "")).strip()
    text = _SPL_SPLIT_RE.sub(lambda m: f"SPLIT RS {m.group(1)} TO RS {m.group(2)}", text)
    try:
        face = float(face_value)
    except (TypeError, ValueError):
        return text
    if "DIV" not in text.upper() or face <= 0:
        return text
    return _PERCENT_RE.sub(lambda m: f"RS {float(m.group(1)) / 100.0 * face:.2f} PER SHARE", text)


def face_value_history(records: Iterable[dict]) -> Dict[str, List[Tuple[date, float]]]:
    """{today's symbol: [(ex-date, face value before it), ...]} from the splits and consolidations in API
    records ("Face Value Split From Rs.10/- To Rs.5/-": Rs 10 until that ex-date)."""
    from nse_engine.data.reference import _CONSOL_RE, _SPLIT_RE

    out: Dict[str, List[Tuple[date, float]]] = {}
    for r in records:
        day = _api_date(r.get("exDate"))
        text = str(r.get("subject", "")).upper()
        pairs = _SPLIT_RE.findall(text) + _CONSOL_RE.findall(text)
        if day and pairs and r.get("symbol"):
            out.setdefault(r["symbol"].upper(), []).append((day, float(pairs[0][0])))
    return {k: sorted(set(v)) for k, v in out.items()}


def face_value_on(symbol: str, day: date, current: Optional[float],
                  history: Dict[str, List[Tuple[date, float]]]) -> Optional[float]:
    """Face value on ``day``: what it was before the first later split or consolidation, else today's."""
    for when, before in history.get(symbol.upper(), []):
        if when > day:
            return before
    return current


def symbol_on(symbol: str, day: date, changes) -> str:
    """The symbol a company traded under on ``day``, from today's, back through ``symbolchange.csv``."""
    sym = symbol.upper()
    for _ in range(10):
        later = changes[(changes["new"] == sym) & (changes["date"].dt.date > day)]
        if later.empty:
            return sym
        sym = later.sort_values("date").iloc[-1]["old"]
    return sym


def corporate_action_bc(records: Iterable[dict], history: Dict[str, List[Tuple[date, float]]],
                        changes) -> bytes:
    """API records as the bytes of a Bc file: each under the symbol of its ex-date, its percentage
    dividends at the face value of that date."""
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(BC_HEADER)
    for r in records:
        ex = _api_date(r.get("exDate"))
        if not ex or not r.get("symbol"):
            continue
        try:
            current = float(r.get("faceVal"))
        except (TypeError, ValueError):
            current = None
        face = face_value_on(r["symbol"], ex, current, history)
        subject = CORPACT_FIXES.get((r["symbol"].upper(), ex.isoformat()), r.get("subject", ""))
        w.writerow([r.get("series") or "EQ", symbol_on(r["symbol"], ex, changes), r.get("comp") or "",
                    _bc_date(r.get("recDate")), _bc_date(r.get("bcStartDate")), _bc_date(r.get("bcEndDate")),
                    ex.strftime("%d/%m/%Y"), _bc_date(r.get("ndStartDate")), _bc_date(r.get("ndEndDate")),
                    bc_purpose(subject, face)])
    return out.getvalue().encode()


def atomic_write(path: Path, content: bytes) -> None:
    """Write bytes to ``path`` via a temp file in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as fh:
        fh.write(content)
    os.replace(tmp, path)


class RateLimiter:
    """Enforce a minimum interval between successive calls."""

    def __init__(self, per_second: float, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.interval = 1.0 / per_second if per_second > 0 else 0.0
        self._clock, self._sleep = clock, sleep
        self._last = -1e18

    def wait(self) -> None:
        delay = self._last + self.interval - self._clock()
        if delay > 0:
            self._sleep(delay)
        self._last = self._clock()


class BhavcopyArchive:
    """Resumable, polite mirror of NSE daily archives under ``root``."""

    def __init__(self, root: Union[str, Path], requests_per_second: float = 2.0,
                 max_retries: int = 5, timeout: float = 30.0,
                 session_factory: Optional[Callable[[], requests.Session]] = None) -> None:
        self.root = Path(root)
        self.limiter = RateLimiter(requests_per_second)
        self.max_retries = max_retries
        self.timeout = timeout
        self._session_factory = session_factory or self._new_session
        self._session: Optional[requests.Session] = None
        self._missing_path = self.root / "manifest" / "missing.json"
        self.missing: Dict[str, Set[str]] = self._load_missing()
        self._dirty = 0

    # -- session / http -------------------------------------------------
    @staticmethod
    def _new_session() -> requests.Session:
        s = requests.Session()
        s.headers.update({
            "User-Agent": USER_AGENT,
            "Referer": HOME_URL + "/",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
        })
        try:  # primes cookies; the home page itself may answer 403
            s.get(HOME_URL, timeout=15)
        except requests.RequestException as exc:
            logger.debug("home page warm-up failed: %s", exc)
        return s

    @property
    def session(self) -> requests.Session:
        if self._session is None:
            self._session = self._session_factory()
        return self._session

    def reset_session(self) -> None:
        if self._session is not None:
            self._session.close()
        self._session = None

    def fetch(self, url: str) -> Tuple[Optional[int], bytes]:
        """GET ``url`` with retries. Returns (status, content); status None on failure."""
        for attempt in range(1, self.max_retries + 1):
            self.limiter.wait()
            try:
                resp = self.session.get(url, timeout=self.timeout)
            except requests.RequestException as exc:
                logger.warning("GET %s failed (%s), attempt %d", url, exc, attempt)
                self.reset_session()
                time.sleep(min(60.0, 2.0 ** attempt))
                continue
            if resp.status_code in (200, 404):
                return resp.status_code, resp.content
            logger.warning("GET %s -> %d, attempt %d", url, resp.status_code, attempt)
            if resp.status_code in (401, 403, 429) or resp.status_code >= 500:
                self.reset_session()
                time.sleep(min(120.0, 3.0 * 2 ** attempt))
                continue
            return resp.status_code, resp.content
        return None, b""

    # -- holidays manifest ------------------------------------------------
    def _load_missing(self) -> Dict[str, Set[str]]:
        try:
            raw = json.loads(self._missing_path.read_text())
            return {k: set(v) for k, v in raw.items()}
        except (FileNotFoundError, ValueError):
            return {}

    def save_missing(self) -> None:
        payload = {k: sorted(v) for k, v in sorted(self.missing.items())}
        atomic_write(self._missing_path, json.dumps(payload, indent=1).encode())
        self._dirty = 0

    def is_missing(self, kind: str, d: date) -> bool:
        return d.isoformat() in self.missing.get(kind, set())

    def mark_missing(self, kind: str, d: date) -> None:
        self.missing.setdefault(kind, set()).add(d.isoformat())
        self._dirty += 1
        if self._dirty >= 25:
            self.save_missing()

    # -- sync -------------------------------------------------------------
    @staticmethod
    def existing(candidates: Sequence[RemoteFile]) -> Optional[RemoteFile]:
        return next((c for c in candidates if c.path.exists()), None)

    def sync_one(self, kind: str, d: date, today: Optional[date] = None) -> str:
        """Fetch one (kind, date). Returns skipped|downloaded|holiday|failed."""
        cands = candidate_files(self.root, kind, d)
        if self.existing(cands) is not None or self.is_missing(kind, d):
            return "skipped"
        all_404 = True
        for cand in cands:
            status, content = self.fetch(cand.url)
            payload = transform_content(cand, content) if status == 200 else None
            if payload is not None:
                atomic_write(cand.path, payload)
                return "downloaded"
            if status != 404:
                all_404 = False
                logger.warning("unusable response for %s %s (status %s)", kind, d, status)
        today = today or date.today()
        if all_404 and d < today - RECENT_GRACE:
            if kind == "indices" and self.existing(candidate_files(self.root, "equity", d)) is not None:
                # LN-T19: NSE traded that day, so its index file is late, not a holiday; recording it as
                # one left 12 NIFTY gaps in 2013-16.  The sync retries it while it is in the lookback.
                logger.warning("indices file for %s not found though the equity session exists: retried", d)
                return "failed"
            self.mark_missing(kind, d)
            return "holiday"
        return "failed" if not all_404 else "not_published"

    def sync(self, start: DateLike, end: DateLike,
             kinds: Sequence[str] = KINDS, log_every: int = 20) -> Dict[str, Dict[str, int]]:
        """Mirror every session in [start, end] for ``kinds``. Resumable."""
        start_d, end_d = to_date(start), to_date(end)
        from nse_engine.nse_calendar import special_session_dates

        # a special session outside the window (one the nightly window passed before it was probed)
        # is re-checked every sync: a stat when it is already here or known missing
        dates = sorted(set(session_dates(start_d, end_d)) | {d for d in special_session_dates() if d <= end_d})
        counts: Dict[str, Dict[str, int]] = {k: {} for k in kinds}
        t0 = time.monotonic()
        logger.info("sync %s..%s: %d candidate sessions, kinds=%s", start_d, end_d, len(dates), list(kinds))
        try:
            for i, d in enumerate(dates, 1):
                for kind in kinds:
                    if kind != "equity" and "equity" in kinds and self.is_missing("equity", d):
                        # No equity session => no delivery/index file either.
                        result = "skipped" if self.is_missing(kind, d) else "holiday"
                        if result == "holiday":
                            self.mark_missing(kind, d)
                    else:
                        result = self.sync_one(kind, d)
                    counts[kind][result] = counts[kind].get(result, 0) + 1
                if i % log_every == 0 or i == len(dates):
                    elapsed = time.monotonic() - t0
                    logger.info("progress %d/%d sessions (%s) %.0fs elapsed, %.2f sessions/s | %s",
                                i, len(dates), d, elapsed, i / max(elapsed, 1e-9), _fmt_counts(counts))
        finally:
            self.save_missing()
        return counts

    def _corporate_actions_month(self, month: date) -> Optional[list]:
        """One month of NSE's corporate-actions API, from the cache or fetched (None on failure)."""
        cache = self.root / CORPACT_API_CACHE / f"{month:%Y-%m}.json"
        if cache.exists():
            return json.loads(cache.read_text())
        month_end = (month.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        status, content = self.fetch(CORPACT_API.format(start=f"{month:%d-%m-%Y}", end=f"{month_end:%d-%m-%Y}"))
        try:
            records = json.loads(content) if status == 200 else None
        except ValueError:
            records = None
        if isinstance(records, dict):
            records = records.get("data")
        if not isinstance(records, list):
            logger.error("corporate actions %s: status %s, no records", f"{month:%Y-%m}", status)
            return None
        if month_end < date.today() - RECENT_GRACE:          # a month still open is fetched again next time
            atomic_write(cache, json.dumps(records).encode())
        return records

    def backfill_corporate_actions(self, start: DateLike, end: DateLike,
                                   history_end: Optional[DateLike] = None) -> Dict[str, int]:
        """Months ``start``..``end`` of NSE's corporate-actions API as Bc files, ``corpact/<year>/bc<month
        end>.csv`` (rewritten).  The API's months up to ``history_end`` (default today) give the face values;
        ``symbolchange.csv`` the symbols of the time.  Returns records written per month."""
        from nse_engine.data.reference import load_symbol_changes

        def months(a: date, b: date):
            m = a.replace(day=1)
            while m <= b:
                yield m
                m = (m.replace(day=28) + timedelta(days=4)).replace(day=1)

        first, last = to_date(start), to_date(end)
        everything = {m: self._corporate_actions_month(m)
                      for m in months(first, to_date(history_end) if history_end else date.today())}
        missing = [f"{m:%Y-%m}" for m, recs in everything.items() if recs is None]
        if missing:
            raise RuntimeError(f"corporate actions not fetched for {missing}: rerun to resume")
        history = face_value_history(r for recs in everything.values() for r in recs)
        changes = load_symbol_changes(self.root)
        out: Dict[str, int] = {}
        for m in months(first, last):
            month_end = (m.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
            atomic_write(self.root / "corpact" / str(m.year) / f"bc{month_end:%Y%m%d}.csv",
                         corporate_action_bc(everything[m], history, changes))
            out[f"{m:%Y-%m}"] = len(everything[m])
        return out

    def sync_reference(self) -> Dict[str, int]:
        """Download reference lists (always refreshed). Returns bytes per file."""
        out: Dict[str, int] = {}
        for name, path in REFERENCE_FILES.items():
            status, content = self.fetch(ARCHIVE_HOST + path)
            dest = self.root / "reference" / name
            if status == 200 and looks_valid(content, dest):
                atomic_write(dest, content)
                out[name] = len(content)
            else:
                logger.error("reference %s failed (status %s)", name, status)
                out[name] = 0
        return out


def _fmt_counts(counts: Dict[str, Dict[str, int]]) -> str:
    return "; ".join(f"{k}: " + ",".join(f"{r}={n}" for r, n in sorted(v.items())) for k, v in counts.items())


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Mirror NSE bhavcopy / delivery / index archives")
    p.add_argument("--start", required=True)
    p.add_argument("--end", default=date.today().isoformat())
    p.add_argument("--root", default="data/nse_engine/archive")
    p.add_argument("--kinds", default=",".join(KINDS),
                   help=f"comma list; also {FO_KIND!r} (F&O bhavcopy), {PARTICIPANT_KIND!r} (participant OI)")
    p.add_argument("--rps", type=float, default=2.0)
    p.add_argument("--no-reference", action="store_true")
    p.add_argument("--backfill-corpact", action="store_true",
                   help="corporate actions from NSE's API (for 2006-09, before PR zips carry a Bc file)")
    args = p.parse_args(argv)
    # SIGTERM -> SystemExit so the holidays manifest is flushed in ``sync``.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    archive = BhavcopyArchive(args.root, requests_per_second=args.rps)
    if args.backfill_corpact:
        logger.info("corporate actions: %s", json.dumps(archive.backfill_corporate_actions(args.start, args.end)))
        return 0
    if not args.no_reference:
        logger.info("reference: %s", archive.sync_reference())
    counts = archive.sync(args.start, args.end, kinds=[k for k in args.kinds.split(",") if k])
    logger.info("done: %s", json.dumps(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
