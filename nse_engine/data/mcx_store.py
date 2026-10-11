"""
MCX commodity futures: daily bhavcopy mirror, parquet store and continuous series (tracker GT2).

MCX serves one bhavcopy per session from its website
(``market-data/bhavcopy/GetDateWiseBhavCopy``), from 2003.  A bot filter
rejects plain HTTP clients, so requests go through ``curl_cffi``
impersonating Chrome (already a core dependency).  Commodity futures
(``FUTCOM``) only; options and index futures are not mirrored.

Raw files mirror under ``archive/mcx`` (resumable; written atomically)::

    mcx/2013/mcx20130115.json.gz     the session's FUTCOM rows, as served
    mcx/manifest/missing.json        dates with no session (holidays, most weekends)

The store is kept apart from the equity and F&O stores, so neither
fingerprint moves::

    mcx_store/futures/2013.parquet   date, symbol, expiry, open, high, low, close, prev_close,
                                     contracts, quantity, unit, value_inr, oi
    mcx_store/continuous.parquet     one row per (symbol, session): the contract held
                                     (see below), its OHLC, ret, factor, adj_close, rolled,
                                     oi, contracts, multiplier
    mcx_store/manifest.json          version, per-year source fingerprints, data hash

**Units.**  Prices are as quoted (Rs per barrel, per mmBtu, per kg, per 10 g
...).  ``contracts`` is lots traded, ``quantity`` the traded quantity in
``unit`` (MCX's "volume in thousands" x 1000), ``value_inr`` the traded
value in rupees (MCX quotes it in lakhs), ``oi`` open interest in lots.

**Continuous series.**  Per symbol, one contract is held at a time and
chosen at each close: among contracts more than ``ROLL_GUARD_DAYS``
calendar days from expiry, the one with the most open interest (the
nearer on a tie).  The holding only moves to a later expiry.  The guard
keeps a position out of the tender and delivery period that physically
settled contracts enter about 5 sessions before expiry, when brokers
square off.  A session's ``ret`` is the held contract's own close-to-close
return, so rolls add no artificial gap.  ``rolled`` marks the closes at
which the holding moved: the roll trades both contracts at that close.
``adj_close`` is the back-adjusted level (the compounded returns, scaled to
the last close); ``factor`` = adj_close / close turns the row's raw OHLC
into adjusted OHLC.  ``multiplier`` is rupees per point per lot, implied
as the median over the contract's traded sessions of value_inr /
(contracts x close).  A lot's value is multiplier x close.  Every row
uses only data up to its close.

CLI::

    python -m nse_engine.data.mcx_store sync --start 2005-01-01
    python -m nse_engine.data.mcx_store build
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests

from nse_engine.data import archive

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

MCX_STORE_VERSION = 1  # bump when parsing or the roll rule changes so everything is rebuilt
MCX_HOME = "https://www.mcxindia.com"
BHAVCOPY_PAGE = MCX_HOME + "/market-data/bhavcopy"
BHAVCOPY_API = BHAVCOPY_PAGE + "/GetDateWiseBhavCopy"
KIND = "mcx"
SYNC_START = date(2005, 1, 1)
ROLL_GUARD_DAYS = 10  # calendar days, about 7 sessions
FUTURE_COLUMNS = ["date", "symbol", "expiry", "open", "high", "low", "close", "prev_close",
                  "contracts", "quantity", "unit", "value_inr", "oi"]
CONTINUOUS_COLUMNS = ["date", "symbol", "expiry", "open", "high", "low", "close", "ret", "factor",
                      "adj_close", "rolled", "oi", "contracts", "multiplier"]


# ------------------------------------------------------------------ fetch
class _ChromeSession:
    """A curl_cffi session that MCX's bot filter takes for Chrome; failures raise ``requests`` errors."""

    def __init__(self) -> None:
        from curl_cffi import requests as curl_requests  # lazy: only the sync needs it

        self._errors = curl_requests.exceptions.RequestException
        self._session = curl_requests.Session(impersonate="chrome")
        self._session.headers.update({"Referer": BHAVCOPY_PAGE, "X-Requested-With": "XMLHttpRequest",
                                      "Accept": "application/json, text/javascript, */*; q=0.01"})

    def get(self, url: str, timeout: float = 30.0):
        try:
            return self._session.get(url, timeout=timeout)
        except self._errors as exc:
            raise requests.ConnectionError(str(exc)) from exc

    def close(self) -> None:
        self._session.close()


def day_path(root: PathLike, d: date) -> Path:
    return Path(root) / str(d.year) / f"mcx{d:%Y%m%d}.json.gz"


def sync(root: PathLike, start: date, end: date, requests_per_second: float = 0.67,
         today: Optional[date] = None, log_every: int = 250) -> Dict[str, int]:
    """Mirror every candidate session in [start, end] not yet saved or known missing."""
    mirror = archive.BhavcopyArchive(root, requests_per_second, session_factory=_ChromeSession)
    today = today or date.today()
    counts: Dict[str, int] = {}
    # MCX trades Monday to Saturday (Saturdays until about 2009) and on Diwali Sundays
    dates = [d for d in archive.session_dates(start, end)
             if d.weekday() < 6 or d in archive.SPECIAL_SUNDAY_SESSIONS]
    try:
        for i, d in enumerate(dates, 1):
            path = day_path(root, d)
            if path.exists() or mirror.is_missing(KIND, d):
                result = "skipped"
            else:
                status, content = mirror.fetch(BHAVCOPY_API + "?" + urlencode(
                    {"InstrumentName": "FUTCOM", "fromDate": d.strftime("%d/%m/%Y")}))
                try:
                    payload = json.loads(content) if status == 200 else None
                except ValueError:
                    payload = None
                rows = payload.get("Data") if isinstance(payload, dict) else None
                # a day without a session answers IsSuccess false, "No data found.", Data null
                no_session = isinstance(payload, dict) and not rows and (
                    payload.get("IsSuccess") or "no data" in str(payload.get("Message", "")).lower())
                if rows:
                    archive.atomic_write(path, gzip.compress(json.dumps(rows).encode()))
                    result = "downloaded"
                elif no_session and d < today - archive.RECENT_GRACE:
                    mirror.mark_missing(KIND, d)
                    result = "holiday"
                elif no_session:
                    result = "not_published"
                else:
                    logger.warning("mcx %s: unusable answer (status %s, %d bytes)", d, status, len(content))
                    result = "failed"
            counts[result] = counts.get(result, 0) + 1
            if i % log_every == 0 or i == len(dates):
                mirror.save_missing()
                logger.info("mcx %d/%d (%s): %s", i, len(dates), d, counts)
    finally:
        mirror.save_missing()
    return counts


# ------------------------------------------------------------------ parse
def parse_day(rows: List[dict], d: date) -> pd.DataFrame:
    """One session's FUTCOM rows as FUTURE_COLUMNS (the session date is the file's)."""
    df = pd.DataFrame([r for r in rows if str(r.get("InstrumentName", "")).strip() == "FUTCOM"])
    if not len(df):
        return pd.DataFrame(columns=FUTURE_COLUMNS)
    volume = df["VolumeInThousands"].astype(str).str.strip().str.extract(r"^([-\d.]+)\s*(.*)$")
    out = pd.DataFrame({
        "date": pd.Timestamp(d),
        "symbol": df["Symbol"].astype(str).str.strip(),
        "expiry": pd.to_datetime(df["ExpiryDate"].astype(str).str.strip(), format="%d%b%Y", errors="coerce"),
        "open": pd.to_numeric(df["Open"], errors="coerce"),
        "high": pd.to_numeric(df["High"], errors="coerce"),
        "low": pd.to_numeric(df["Low"], errors="coerce"),
        "close": pd.to_numeric(df["Close"], errors="coerce"),
        "prev_close": pd.to_numeric(df["PreviousClose"], errors="coerce"),
        "contracts": pd.to_numeric(df["Volume"], errors="coerce"),
        "quantity": pd.to_numeric(volume[0], errors="coerce") * 1000.0,
        "unit": volume[1].str.strip(),
        "value_inr": pd.to_numeric(df["Value"], errors="coerce") * 1e5,
        "oi": pd.to_numeric(df["OpenInterest"], errors="coerce"),
    })
    for col in ("open", "high", "low"):  # untraded contracts print 0 with the previous close
        out[col] = out[col].where(out[col] > 0)
    out = out.dropna(subset=["expiry", "close"])
    out = out[out["close"] > 0]
    return out.drop_duplicates(subset=["symbol", "expiry"]).reset_index(drop=True)[FUTURE_COLUMNS]


def _year_files(root: Path, year: int) -> List[Path]:
    d = root / str(year)
    return sorted(p for p in d.glob("mcx*.json.gz") if not p.name.startswith(".")) if d.exists() else []


def fingerprint(files: Sequence[Path]) -> str:
    h = hashlib.sha256(f"mcx-v{MCX_STORE_VERSION}".encode())
    for p in files:
        h.update(f"{p.name}:{p.stat().st_size}".encode())
    return h.hexdigest()[:20]


def _build_year(root: Path, store: Path, year: int) -> Dict[str, object]:
    frames = [parse_day(json.loads(gzip.decompress(p.read_bytes())), pd.Timestamp(p.name[3:11]).date())
              for p in _year_files(root, year)]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=FUTURE_COLUMNS)
    target = store / "futures" / f"{year}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target, index=False)
    return {"year": year, "rows": len(df), "sessions": int(df["date"].nunique()),
            "fingerprint": fingerprint(_year_files(root, year))}


# ------------------------------------------------------------------ continuous
def implied_multipliers(futures: pd.DataFrame) -> pd.Series:
    """Rupees per point per lot for each (symbol, expiry): median of value_inr / (contracts x close)."""
    traded = futures[(futures["contracts"] > 0) & (futures["value_inr"] > 0)]
    ratio = traded["value_inr"] / (traded["contracts"] * traded["close"])
    return ratio.groupby([traded["symbol"], traded["expiry"]]).median().rename("multiplier")


def continuous_series(futures: pd.DataFrame, guard_days: int = ROLL_GUARD_DAYS) -> pd.DataFrame:
    """One held contract per (symbol, session) by the roll rule in the module docstring."""
    multipliers = implied_multipliers(futures)
    guard = pd.Timedelta(days=guard_days)
    out = []
    for symbol, g in futures.groupby("symbol", sort=True):
        by_date = {d: day.set_index("expiry") for d, day in g.groupby("date", sort=True)}
        held: Optional[pd.Timestamp] = None
        prev: Optional[pd.DataFrame] = None
        rows = []
        for d, day in by_date.items():
            before = held  # the contract held through this session
            eligible = day[day.index - d > guard]
            if len(eligible):
                best = eligible.sort_index().sort_values("oi", ascending=False, kind="stable").index[0]
                if held is None or held - d <= guard or held not in day.index or best > held:
                    held = best
            elif held is None or held not in day.index:
                held = day.index.min()
            if before is not None and before in day.index:
                bar = day.loc[before]
                last = prev.loc[before, "close"] if prev is not None and before in prev.index else np.nan
                rows.append({"date": d, "expiry": before, "open": bar["open"], "high": bar["high"],
                             "low": bar["low"], "close": bar["close"], "ret": bar["close"] / last - 1.0,
                             "rolled": held != before, "oi": bar["oi"], "contracts": bar["contracts"]})
            prev = day
        if not rows:
            continue
        s = pd.DataFrame(rows)
        s["symbol"] = symbol
        level = (1.0 + s["ret"].fillna(0.0)).cumprod()
        s["adj_close"] = level * (s["close"].iloc[-1] / level.iloc[-1])
        s["factor"] = s["adj_close"] / s["close"]
        s["multiplier"] = [multipliers.get((symbol, e), np.nan) for e in s["expiry"]]
        out.append(s)
    if not out:
        return pd.DataFrame(columns=CONTINUOUS_COLUMNS)
    return pd.concat(out, ignore_index=True)[CONTINUOUS_COLUMNS]


# ------------------------------------------------------------------ build
def build_mcx_store(archive_root: PathLike, store_dir: PathLike) -> Dict[str, object]:
    """Build (or update) the store; a year is re-parsed only when its source files changed."""
    root, store = Path(archive_root), Path(store_dir)
    store.mkdir(parents=True, exist_ok=True)
    manifest_path = store / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest.get("version") != MCX_STORE_VERSION:
        manifest = {"version": MCX_STORE_VERSION, "years": {}}
    years = sorted(int(p.name) for p in root.glob("[0-9]" * 4))
    todo = [y for y in years if manifest["years"].get(str(y), {}).get("fingerprint")
            != fingerprint(_year_files(root, y))]
    logger.info("mcx store: %d years, %d to build", len(years), len(todo))
    for year in todo:
        manifest["years"][str(year)] = info = _build_year(root, store, year)
        logger.info("year %s: %s", year, info)
    futures = load_futures(store)
    cont = continuous_series(futures)
    cont.to_parquet(store / "continuous.parquet", index=False)
    manifest["continuous_rows"] = len(cont)
    manifest["symbols"] = {s: {"first": str(g["date"].min().date()), "last": str(g["date"].max().date()),
                               "sessions": len(g), "rolls": int(g["rolled"].sum())}
                           for s, g in cont.groupby("symbol")}
    manifest["data_hash"] = hashlib.sha256(json.dumps(
        {y: v["fingerprint"] for y, v in sorted(manifest["years"].items())}, sort_keys=True).encode()).hexdigest()[:16]
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest


# ------------------------------------------------------------------ read
def load_futures(store_dir: PathLike, symbols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    paths = sorted((Path(store_dir) / "futures").glob("*.parquet"))
    filters = [("symbol", "in", list(symbols))] if symbols else None
    frames = [pd.read_parquet(p, filters=filters) for p in paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=FUTURE_COLUMNS)


def load_continuous(store_dir: PathLike, symbol: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_parquet(Path(store_dir) / "continuous.parquet",
                         filters=[("symbol", "==", symbol)] if symbol else None)
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def data_hash(store_dir: PathLike) -> str:
    return json.loads((Path(store_dir) / "manifest.json").read_text()).get("data_hash", "")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="MCX commodity futures: bhavcopy mirror and store")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sync", help="mirror the daily FUTCOM bhavcopies")
    s.add_argument("--start", default=SYNC_START.isoformat())
    s.add_argument("--end", default=date.today().isoformat())
    b = sub.add_parser("build", help="write the parquet store and continuous series")
    b.add_argument("--store", default="data/nse_engine/mcx_store")
    for q in (s, b):
        q.add_argument("--archive", default="data/nse_engine/archive/mcx")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cmd == "build":
        manifest = build_mcx_store(args.archive, args.store)
        logger.info("done: data hash %s, %d continuous rows, %d symbols", manifest["data_hash"],
                    manifest["continuous_rows"], len(manifest["symbols"]))
        return 0
    counts = sync(args.archive, archive.to_date(args.start), archive.to_date(args.end))
    logger.info("sync done: %s", counts)
    return 1 if counts.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
