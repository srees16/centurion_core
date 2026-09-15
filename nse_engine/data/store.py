"""
Normalised parquet store built from the raw NSE archive.

Layout under ``store_dir``::

    equity/2013.parquet      date, symbol, series, isin, open, high, low, close,
                             prev_close, volume, value_inr, trades, deliv_qty, deliv_pct
    indices/2013.parquet     date, index_name, open, high, low, close
    corpact/2013.parquet     symbol, series, ex_date, purpose, file_date (raw Bc rows)
    corporate_actions.parquet  price-relevant events (whole store, deduplicated)
    dividends.parquet        cash dividends per (symbol, ex_date) (whole store, deduplicated)
    spans.parquet            symbol, isin, first, last, n   (whole store)
    calendar.parquet         date                           (equity sessions)
    reference/*.csv          copies of the archive reference lists
    manifest.json            per-year source fingerprints (idempotent rebuilds)

Only series in ``KEEP_SERIES`` are stored (ETFs trade in EQ).  Delivery comes
from MTO files (all years) or, failing that, ``sec_bhavdata_full``.

CLI::

    python -m nse_engine.data.store --archive data/nse_engine/archive --store data/nse_engine/store
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import re
import shutil
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from nse_engine.data import reference

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

STORE_VERSION = 3  # bump when parsing changes so every year is rebuilt
#: bump when corporate-action / dividend event derivation changes (no year rebuild needed)
EVENTS_VERSION = 3
KEEP_SERIES = ("EQ", "BE")
EQUITY_COLUMNS = ["date", "symbol", "series", "isin", "open", "high", "low", "close",
                  "prev_close", "volume", "value_inr", "trades", "deliv_qty", "deliv_pct"]
INDEX_COLUMNS = ["date", "index_name", "open", "high", "low", "close"]
FLOAT_COLUMNS = ["open", "high", "low", "close", "prev_close", "volume", "value_inr",
                 "trades", "deliv_qty", "deliv_pct"]

_INDEX_ALIASES = {
    "NIFTY": "NIFTY50",
    "NIFTYJUNIOR": "NIFTYNEXT50",
    "NIFTYNIFTYJUNIOR": "NIFTYNEXT50",
    "NIFTYDIVIDEND": "NIFTY50DIVIDENDPOINTS",
    "VIX": "INDIAVIX",
}


# ------------------------------------------------------------------ io
def read_raw_text(path_or_bytes: Union[PathLike, bytes]) -> str:
    """Return the text of a raw archive file, unzipping ``.zip`` content."""
    data = path_or_bytes if isinstance(path_or_bytes, bytes) else Path(path_or_bytes).read_bytes()
    if data[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            name = zf.namelist()[0]
            data = zf.read(name)
    return data.decode("utf-8", errors="replace")


def _num(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "fi":
        return s.astype("float64")
    cleaned = s.astype("string").str.strip().replace({"-": None, "": None})
    return pd.to_numeric(cleaned, errors="coerce").astype("float64")


def _finish_equity(df: pd.DataFrame) -> pd.DataFrame:
    for col in EQUITY_COLUMNS:
        if col not in df:
            df[col] = np.nan
    df = df[EQUITY_COLUMNS].copy()
    for col in ("symbol", "series", "isin"):
        df[col] = df[col].astype("string").str.strip().str.upper()
    for col in FLOAT_COLUMNS:
        df[col] = _num(df[col])
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    return df


# ------------------------------------------------------------- parsers
def _legacy_dates(raw: pd.Series) -> pd.Series:
    """``19-DEC-2019``; a few files use a two-digit year (``13-Jul-20``, 2020-07-13)."""
    raw = raw.str.strip()
    dt = pd.to_datetime(raw, format="%d-%b-%Y", errors="coerce")
    miss = dt.isna()
    if miss.any():
        dt[miss] = pd.to_datetime(raw[miss], format="%d-%b-%y", errors="coerce")
    if dt.isna().any():
        raise ValueError(f"unparseable TIMESTAMP values: {raw[dt.isna()].unique()[:3]}")
    return dt


def parse_legacy_bhavcopy(text: str) -> pd.DataFrame:
    """Legacy ``cmDDMONYYYYbhav.csv`` -> normalised equity rows."""
    df = pd.read_csv(io.StringIO(text), dtype=str, skipinitialspace=True)
    df.columns = [c.strip().upper() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c and not c.startswith("UNNAMED")]]
    out = pd.DataFrame({
        "date": _legacy_dates(df["TIMESTAMP"]),
        "symbol": df["SYMBOL"], "series": df["SERIES"], "isin": df.get("ISIN"),
        "open": df["OPEN"], "high": df["HIGH"], "low": df["LOW"], "close": df["CLOSE"],
        "prev_close": df["PREVCLOSE"], "volume": df["TOTTRDQTY"], "value_inr": df["TOTTRDVAL"],
        "trades": df.get("TOTALTRADES"),
    })
    return _finish_equity(out)


def parse_udiff_bhavcopy(text: str) -> pd.DataFrame:
    """UDiFF ``BhavCopy_NSE_CM_0_0_0_YYYYMMDD_F_0000.csv`` -> normalised equity rows."""
    df = pd.read_csv(io.StringIO(text), dtype=str)
    df.columns = [c.strip() for c in df.columns]
    if "FinInstrmTp" in df:
        df = df[df["FinInstrmTp"].fillna("STK").str.strip().isin(["STK", "ETF"])]
    out = pd.DataFrame({
        "date": pd.to_datetime(df["TradDt"].str.strip(), format="%Y-%m-%d"),
        "symbol": df["TckrSymb"], "series": df["SctySrs"], "isin": df["ISIN"],
        "open": df["OpnPric"], "high": df["HghPric"], "low": df["LwPric"], "close": df["ClsPric"],
        "prev_close": df["PrvsClsgPric"], "volume": df["TtlTradgVol"], "value_inr": df["TtlTrfVal"],
        "trades": df.get("TtlNbOfTxsExctd"),
    })
    return _finish_equity(out)


def parse_full_bhavcopy(text: str) -> pd.DataFrame:
    """``sec_bhavdata_full_DDMMYYYY.csv`` (OHLC + delivery, turnover in lakhs)."""
    df = pd.read_csv(io.StringIO(text), dtype=str, skipinitialspace=True)
    df.columns = [c.strip().upper() for c in df.columns]
    df = df.apply(lambda s: s.str.strip())
    out = pd.DataFrame({
        "date": pd.to_datetime(df["DATE1"], format="%d-%b-%Y"),
        "symbol": df["SYMBOL"], "series": df["SERIES"], "isin": None,
        "open": df["OPEN_PRICE"], "high": df["HIGH_PRICE"], "low": df["LOW_PRICE"],
        "close": df["CLOSE_PRICE"], "prev_close": df["PREV_CLOSE"], "volume": df["TTL_TRD_QNTY"],
        "value_inr": _num(df["TURNOVER_LACS"]) * 1e5, "trades": df["NO_OF_TRADES"],
        "deliv_qty": df["DELIV_QTY"], "deliv_pct": df["DELIV_PER"],
    })
    return _finish_equity(out)


_MTO_DATE = re.compile(r"^10,MTO,(\d{8})", re.MULTILINE)


def parse_mto(text: str) -> pd.DataFrame:
    """``MTO_DDMMYYYY.DAT`` -> DataFrame[date, symbol, series, traded_qty, deliv_qty, deliv_pct].

    Data rows are ``20,<sr>,<symbol>,<series>,<traded>,<deliverable>,<pct>``.
    """
    m = _MTO_DATE.search(text)
    if m is None:
        raise ValueError("MTO file without a '10,MTO,DDMMYYYY' header")
    when = pd.to_datetime(m.group(1), format="%d%m%Y")
    rows = [line.split(",") for line in text.splitlines() if line.startswith("20,")]
    rows = [r[:7] for r in rows if len(r) >= 7]
    df = pd.DataFrame(rows, columns=["rt", "sr", "symbol", "series", "traded_qty", "deliv_qty", "deliv_pct"])
    df["date"] = when
    for col in ("symbol", "series"):
        df[col] = df[col].str.strip().str.upper()
    for col in ("traded_qty", "deliv_qty", "deliv_pct"):
        df[col] = pd.to_numeric(df[col].str.strip(), errors="coerce")
    return df[["date", "symbol", "series", "traded_qty", "deliv_qty", "deliv_pct"]]


def normalise_index_name(name: str) -> str:
    """'CNX Nifty' / 'S&P CNX Nifty' / 'Nifty 50' -> NIFTY50; 'CNX 500' -> NIFTY500; 'India VIX' -> INDIAVIX."""
    n = name.strip().lower()
    n = re.sub(r"^s&p\s+", "", n)
    n = re.sub(r"\bcnx\b", "nifty", n)
    n = re.sub(r"\bnifty\s+nifty\b", "nifty", n)
    key = re.sub(r"[^a-z0-9]", "", n).upper()
    return _INDEX_ALIASES.get(key, key)


_INDEX_DATE_FORMATS = ("%d-%m-%Y", "%d/%m/%Y", "%d-%b-%Y", "%Y-%m-%d")


def _index_dates(raw: pd.Series, file_date: Optional[pd.Timestamp]) -> pd.Series:
    """Parse index dates (dd-mm-yyyy, dd/mm/yyyy in 2014-15 files, ...), checked against the file date.

    Some files (2023-04-06/10/11) write mm-dd-yyyy: when a date disagrees with
    the file's own date but its day/month swap matches, the file date is used;
    rows matching neither are dropped.
    """
    raw = raw.str.strip()
    dt = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    for fmt in _INDEX_DATE_FORMATS:
        miss = dt.isna()
        if not miss.any():
            break
        dt[miss] = pd.to_datetime(raw[miss], format=fmt, errors="coerce")
    if file_date is None or pd.isna(file_date):
        return dt
    fd = pd.Timestamp(file_date).normalize()
    wrong = dt.notna() & (dt != fd)
    if wrong.any():
        swapped = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
        for fmt in ("%m-%d-%Y", "%m/%d/%Y"):
            miss = swapped.isna() & wrong
            swapped[miss] = pd.to_datetime(raw[miss], format=fmt, errors="coerce")
        fixable = wrong & (swapped == fd)
        dt[fixable] = fd
        dropped = wrong & ~fixable
        logger.warning("index file for %s: %d rows with day/month swapped fixed, %d rows with other dates dropped",
                       fd.date(), int(fixable.sum()), int(dropped.sum()))
        dt[dropped] = pd.NaT
    return dt


def parse_index_close(text: str, file_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """``ind_close_all_DDMMYYYY.csv`` -> DataFrame[date, index_name, open, high, low, close]."""
    df = pd.read_csv(io.StringIO(text), dtype=str, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=["Index Name", "Index Date"])
    out = pd.DataFrame({
        "date": _index_dates(df["Index Date"], file_date),
        "index_name": df["Index Name"].map(normalise_index_name),
        "open": _num(df["Open Index Value"]), "high": _num(df["High Index Value"]),
        "low": _num(df["Low Index Value"]), "close": _num(df["Closing Index Value"]),
    })
    out = out.dropna(subset=["date", "close"])
    out["date"] = out["date"].astype("datetime64[ns]")
    out["index_name"] = out["index_name"].astype("string")
    return out.drop_duplicates(["date", "index_name"], keep="first").reset_index(drop=True)


# --------------------------------------------------------- year builder
def _date_of(path: Path) -> str:
    m = re.search(r"(\d{8})", path.name)
    return m.group(1) if m else ""


def year_sources(archive_root: PathLike, year: int) -> Dict[str, List[Path]]:
    root = Path(archive_root)
    out = {}
    for kind in ("equity", "delivery", "indices", "corpact"):
        d = root / kind / str(year)
        out[kind] = sorted(p for p in d.glob("*") if p.is_file() and not p.name.startswith(".")) if d.exists() else []
    return out


def fingerprint(sources: Dict[str, List[Path]]) -> str:
    h = hashlib.sha256(f"v{STORE_VERSION}".encode())
    for kind in sorted(sources):
        for p in sources[kind]:
            st = p.stat()
            h.update(f"{kind}/{p.name}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:20]


def _safe(parser, path: Path) -> Optional[pd.DataFrame]:
    try:
        return parser(read_raw_text(path))
    except Exception as exc:  # corrupt file must not kill the whole build
        logger.error("failed to parse %s: %s", path, exc)
        return None


def build_year_frames(sources: Dict[str, List[Path]], keep_series: Sequence[str] = KEEP_SERIES):
    """Parse one year's raw files -> (equity frame, indices frame, stats)."""
    eq_parts, mto_parts, full_parts, idx_parts = [], [], [], []
    for p in sources.get("equity", []):
        parser = parse_udiff_bhavcopy if p.name.startswith("udiff") else parse_legacy_bhavcopy
        df = _safe(parser, p)
        if df is not None:
            eq_parts.append(df[df["series"].isin(keep_series)])
    for p in sources.get("delivery", []):
        if p.name.startswith("mto"):
            df = _safe(parse_mto, p)
            if df is not None:
                mto_parts.append(df[df["series"].isin(keep_series)])
        elif p.name.startswith("full"):
            df = _safe(parse_full_bhavcopy, p)
            if df is not None:
                full_parts.append(df[df["series"].isin(keep_series)])
    for p in sources.get("indices", []):
        file_date = pd.to_datetime(_date_of(p), format="%Y%m%d", errors="coerce")
        df = _safe(lambda text, fd=file_date: parse_index_close(text, None if pd.isna(fd) else fd), p)
        if df is not None:
            idx_parts.append(df)
    ca_parts = []
    for p in sources.get("corpact", []):
        file_date = pd.to_datetime(_date_of(p), format="%Y%m%d", errors="coerce")
        df = _safe(lambda text, fd=file_date: reference.parse_corporate_actions(text, fd), p)
        if df is not None and len(df):
            ca_parts.append(df[df["series"].isin(keep_series)])

    equity = pd.concat(eq_parts, ignore_index=True) if eq_parts else _finish_equity(pd.DataFrame(columns=EQUITY_COLUMNS))
    full = pd.concat(full_parts, ignore_index=True) if full_parts else None
    stats = {"equity_files": len(sources.get("equity", [])), "delivery_files": len(sources.get("delivery", [])),
             "index_files": len(sources.get("indices", []))}

    if full is not None:  # sessions with no bhavcopy but a full file
        missing = ~full["date"].isin(set(equity["date"].unique()))
        if missing.any():
            stats["equity_from_full_rows"] = int(missing.sum())
            equity = pd.concat([equity, full[missing]], ignore_index=True)

    key = ["date", "symbol", "series"]
    equity = equity.drop_duplicates(key, keep="first")
    deliv = _delivery_table(mto_parts, full)
    if deliv is not None and len(deliv):
        equity = equity.drop(columns=["deliv_qty", "deliv_pct"]).merge(deliv, on=key, how="left")
    equity = equity.sort_values(["date", "symbol", "series"]).reset_index(drop=True)
    indices = (pd.concat(idx_parts, ignore_index=True).drop_duplicates(["date", "index_name"], keep="last")
               .sort_values(["date", "index_name"]).reset_index(drop=True)
               if idx_parts else pd.DataFrame(columns=INDEX_COLUMNS))
    actions = (pd.concat(ca_parts, ignore_index=True).drop_duplicates()
               if ca_parts else pd.DataFrame(columns=reference.CA_COLUMNS))
    for col in ("ex_date", "file_date"):
        actions[col] = pd.to_datetime(actions[col]).astype("datetime64[ns]")
    stats.update(equity_rows=len(equity), index_rows=len(indices), corpact_rows=len(actions),
                 corpact_files=len(sources.get("corpact", [])))
    return equity[EQUITY_COLUMNS], indices, actions, stats


def _delivery_table(mto_parts: List[pd.DataFrame], full: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    key = ["date", "symbol", "series"]
    parts = []
    if mto_parts:
        parts.append(pd.concat(mto_parts, ignore_index=True)[key + ["deliv_qty", "deliv_pct"]])
    if full is not None:
        parts.append(full[key + ["deliv_qty", "deliv_pct"]])
    if not parts:
        return None
    deliv = pd.concat(parts, ignore_index=True)  # MTO first -> preferred
    deliv["symbol"] = deliv["symbol"].astype("string")
    deliv["series"] = deliv["series"].astype("string")
    deliv["date"] = deliv["date"].astype("datetime64[ns]")
    return deliv.dropna(subset=["deliv_qty", "deliv_pct"], how="all").drop_duplicates(key, keep="first")


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    df.to_parquet(tmp, index=False, engine="pyarrow", compression="zstd")
    os.replace(tmp, path)


def _build_year(archive_root: str, store_dir: str, year: int) -> Dict:
    logging.basicConfig(level=logging.INFO)
    sources = year_sources(archive_root, year)
    equity, indices, actions, stats = build_year_frames(sources)
    _write_parquet(equity, Path(store_dir) / "equity" / f"{year}.parquet")
    _write_parquet(indices, Path(store_dir) / "indices" / f"{year}.parquet")
    _write_parquet(actions, Path(store_dir) / "corpact" / f"{year}.parquet")
    return stats


def archive_years(archive_root: PathLike) -> List[int]:
    years = set()
    for kind in ("equity", "delivery", "indices", "corpact"):
        d = Path(archive_root) / kind
        if d.exists():
            years |= {int(p.name) for p in d.iterdir() if p.is_dir() and p.name.isdigit()}
    return sorted(years)


def _copy_reference(archive_root: Path, store: Path) -> None:
    src = archive_root / "reference"
    if not src.exists():
        logger.warning("no reference files under %s (run sync_reference)", src)
        return
    dst = store / "reference"
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.glob("*.csv"):
        shutil.copy2(p, dst / p.name)


def _write_spans_and_calendar(store: Path) -> Dict[str, int]:
    parts = [pd.read_parquet(p, columns=["date", "symbol", "isin"])
             for p in sorted((store / "equity").glob("*.parquet"))]
    rows = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["date", "symbol", "isin"])
    spans = reference.compute_spans(rows)
    calendar = pd.DataFrame({"date": np.sort(rows["date"].unique())})
    _write_parquet(spans, store / "spans.parquet")
    _write_parquet(calendar, store / "calendar.parquet")
    ca = [pd.read_parquet(p) for p in sorted((store / "corpact").glob("*.parquet"))]
    ca = [c for c in ca if len(c)]
    actions = pd.concat(ca, ignore_index=True) if ca else pd.DataFrame(columns=reference.CA_COLUMNS)
    summary = {"spans": len(spans), "sessions": len(calendar)}
    summary.update(_write_events(store, actions))
    return summary


def _write_events(store: Path, actions: Optional[pd.DataFrame] = None) -> Dict[str, int]:
    """Derive corporate_actions.parquet and dividends.parquet from the raw corpact tables."""
    if actions is None:
        ca = [pd.read_parquet(p) for p in sorted((store / "corpact").glob("*.parquet"))]
        ca = [c for c in ca if len(c)]
        actions = pd.concat(ca, ignore_index=True) if ca else pd.DataFrame(columns=reference.CA_COLUMNS)
    events = reference.corporate_action_events(actions)
    events["ex_date"] = pd.to_datetime(events["ex_date"]).astype("datetime64[ns]")
    for col in reference.EVENT_COLUMNS[3:]:
        events[col] = pd.to_numeric(events[col], errors="coerce").astype("float64")
    _write_parquet(events, store / "corporate_actions.parquet")
    dividends = reference.dividend_events(actions)
    _write_parquet(dividends, store / "dividends.parquet")
    return {"corporate_action_events": len(events), "dividend_events": len(dividends)}


def build_store(archive_root: PathLike, store_dir: PathLike, workers: Optional[int] = None,
                years: Optional[Sequence[int]] = None) -> Dict:
    """Parse the raw archive into the parquet store. Only changed years are rebuilt."""
    archive_root, store = Path(archive_root), Path(store_dir)
    store.mkdir(parents=True, exist_ok=True)
    manifest_path = store / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (FileNotFoundError, ValueError):
        manifest = {}
    manifest.setdefault("years", {})

    todo = {}
    for year in (years or archive_years(archive_root)):
        fp = fingerprint(year_sources(archive_root, year))
        have = manifest["years"].get(str(year), {})
        out_ok = (store / "equity" / f"{year}.parquet").exists()
        if have.get("fingerprint") != fp or not out_ok:
            todo[year] = fp
    logger.info("build_store: %d year(s) to rebuild: %s", len(todo), sorted(todo))

    results: Dict[int, Dict] = {}
    workers = workers or max(1, min(4, (os.cpu_count() or 2) // 2))
    if len(todo) > 1 and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {y: ex.submit(_build_year, str(archive_root), str(store), y) for y in sorted(todo)}
            for y, f in futs.items():
                results[y] = f.result()
                logger.info("built %d: %s", y, results[y])
    else:
        for y in sorted(todo):
            results[y] = _build_year(str(archive_root), str(store), y)
            logger.info("built %d: %s", y, results[y])

    for y, stats in results.items():
        manifest["years"][str(y)] = {"fingerprint": todo[y], **stats}
    _copy_reference(archive_root, store)
    summary = {"rebuilt_years": sorted(results), "skipped_years": sorted(
        int(y) for y in manifest["years"] if int(y) not in results)}
    if results or not (store / "spans.parquet").exists():
        summary.update(_write_spans_and_calendar(store))
    elif manifest.get("events_version") != EVENTS_VERSION or not (store / "dividends.parquet").exists():
        summary.update(_write_events(store))
    manifest["store_version"] = STORE_VERSION
    manifest["events_version"] = EVENTS_VERSION
    tmp = manifest_path.with_name(f".manifest.json.tmp{os.getpid()}")
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    os.replace(tmp, manifest_path)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Build the NSE parquet store from the raw archive")
    p.add_argument("--archive", default="data/nse_engine/archive")
    p.add_argument("--store", default="data/nse_engine/store")
    p.add_argument("--workers", type=int, default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(json.dumps(build_store(args.archive, args.store, workers=args.workers), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
