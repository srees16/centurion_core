"""
Parquet store of derivatives from the NSE F&O bhavcopy (trackers O2, OD2, OD3).

Built from ``archive/fo`` (``python -m nse_engine.data.archive --kinds fo``)
and kept apart from the equity store, so the engine's data fingerprint never
moves::

    fo_store/options/2013.parquet   date, symbol, expiry, strike, option_type, open, high, low,
                                    close, settle, contracts, oi, lot_size
    fo_store/futures/2013.parquet   date, symbol, expiry, close, settle, contracts, value_inr,
                                    oi, lot_size
    fo_store/underlying.parquet     date, index_name, close, source, exact  (from the equity
                                    stores, so a Kaggle kernel needs this store alone)
    fo_store/stock_options/2013.parquet, stock_futures/2013.parquet
                                    the same columns for stock derivatives (OD3)
    fo_store/participant_oi.parquet date, client_type, open interest by instrument and side (OD2)
    fo_store/manifest.json          version, per-year source fingerprints, data hash

Index derivatives on ``SYMBOLS`` are the O2 tables, and only they enter the
data hash.  Stock derivatives (OD3) cover every symbol from ``STOCK_FROM``,
options only where the contract traded or held open interest that day (most
listed strikes do neither); their legacy lots are implied as below and
snapped to the stock lot grid (``_to_lot``).  Participant-wise
open interest (OD2) is NSE's daily file (``archive --kinds participant``,
from January 2012).  UDiFF files (from 2024-07-08)
carry the lot size; for legacy files each (symbol, expiry) takes the lot
implied by its future, turnover / (contracts x close), as the median over
the contract's traded days in the year, then rounded to a multiple of 5
(every index lot so far is one).  A contract's lot never changes during its
life, and the median of the unrounded values removes the days where the
close sat far from the day's average traded price (crashes: 24 Oct 2008;
most of March 2020).  A contract whose future never traded takes that day's
median across the symbol's futures.  An option expiry with no future of
its own (weeklies, long-dated) takes the nearest futures contract's lot.
Expiries are the actual ones (UDiFF ``FininstrmActlXpryDt``).

CLI::

    python -m nse_engine.data.fo_store --archive data/nse_engine/archive \\
        --store data/nse_engine/fo_store   # NIFTY closes from store, then store_ext2006
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from nse_engine.data.external import EXTERNAL_INDEX_FILES, PRIMARY_SOURCE
from nse_engine.data.store import _legacy_dates, _num, read_raw_text

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

FO_STORE_VERSION = 4  # bump when parsing changes so every year is rebuilt
STOCK_STORE_VERSION = 1  # the same for the stock tables (OD3), which never touch the data hash
SYMBOLS = ("NIFTY", "BANKNIFTY")
STOCK_FROM = 2013
INDEX_NAMES = {"NIFTY": "NIFTY50", "BANKNIFTY": "NIFTYBANK"}
OPTION_COLUMNS = ["date", "symbol", "expiry", "strike", "option_type", "open", "high", "low", "close",
                  "settle", "contracts", "oi", "lot_size"]
FUTURE_COLUMNS = ["date", "symbol", "expiry", "close", "settle", "contracts", "value_inr", "oi", "lot_size"]


# ------------------------------------------------------------- parsers

def _split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Typed (options, futures) frames from a normalised frame with an ``is_option`` flag."""
    for col in ("strike", "open", "high", "low", "close", "settle", "contracts", "value_inr", "oi", "lot_size"):
        df[col] = _num(df[col]) if col in df else np.nan
    df["symbol"] = df["symbol"].astype("string").str.strip().str.upper()
    df["option_type"] = df["option_type"].astype("string").str.strip().str.upper()
    opt = df[df["is_option"]].copy()
    fut = df[~df["is_option"]].copy()
    return opt[OPTION_COLUMNS].reset_index(drop=True), fut[FUTURE_COLUMNS].reset_index(drop=True)


def _traded_or_open(options: pd.DataFrame) -> pd.DataFrame:
    """Stock options that traded or held open interest (the rest are untouched listings)."""
    return options[(options["contracts"] > 0) | (options["oi"] > 0)].reset_index(drop=True)


def parse_legacy_fo(text: str, symbols: Sequence[str] = SYMBOLS,
                    stocks: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Legacy ``foDDMONYYYYbhav.csv``: index options and futures on ``symbols``, or with
    ``stocks`` every stock's (options only where traded or open)."""
    raw = pd.read_csv(io.StringIO(text), dtype=str)
    raw.columns = [c.strip() for c in raw.columns]
    raw = raw.rename(columns={"OPTIONTYPE": "OPTION_TYP"})      # the 2003-2006 files' header
    inst = raw["INSTRUMENT"].str.strip()
    option_kind, future_kind = ("OPTSTK", "FUTSTK") if stocks else ("OPTIDX", "FUTIDX")
    raw = raw[inst.isin([option_kind, future_kind]) & (stocks or raw["SYMBOL"].str.strip().isin(symbols))]
    df = pd.DataFrame({
        "date": _legacy_dates(raw["TIMESTAMP"]), "symbol": raw["SYMBOL"],
        "expiry": _legacy_dates(raw["EXPIRY_DT"]), "strike": raw["STRIKE_PR"],
        "option_type": raw["OPTION_TYP"], "open": raw["OPEN"], "high": raw["HIGH"], "low": raw["LOW"],
        "close": raw["CLOSE"], "settle": raw["SETTLE_PR"], "contracts": raw["CONTRACTS"],
        "oi": raw["OPEN_INT"], "is_option": raw["INSTRUMENT"].str.strip() == option_kind,
    })
    df["value_inr"] = _num(raw["VAL_INLAKH"]) * 1e5
    df["lot_size"] = np.nan
    opt, fut = _split(df)
    return (_traded_or_open(opt) if stocks else opt), fut


def parse_udiff_fo(text: str, symbols: Sequence[str] = SYMBOLS,
                   stocks: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """UDiFF ``BhavCopy_NSE_FO_..._F_0000.csv``: IDO / IDF (index options / futures) on
    ``symbols``, or with ``stocks`` STO / STF for every stock (options only where traded or open)."""
    raw = pd.read_csv(io.StringIO(text), dtype=str)
    raw.columns = [c.strip() for c in raw.columns]
    kind = raw["FinInstrmTp"].str.strip()
    option_kind, future_kind = ("STO", "STF") if stocks else ("IDO", "IDF")
    raw = raw[kind.isin([option_kind, future_kind]) & (stocks or raw["TckrSymb"].str.strip().isin(symbols))]
    expiry_col = "FininstrmActlXpryDt" if "FininstrmActlXpryDt" in raw else "XpryDt"
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["TradDt"].str.strip()), "symbol": raw["TckrSymb"],
        "expiry": pd.to_datetime(raw[expiry_col].str.strip()), "strike": raw["StrkPric"],
        "option_type": raw["OptnTp"].fillna("FUT"), "open": raw["OpnPric"], "high": raw["HghPric"],
        "low": raw["LwPric"], "close": raw["ClsPric"], "settle": raw["SttlmPric"],
        "contracts": raw["TtlTradgVol"], "value_inr": raw["TtlTrfVal"], "oi": raw["OpnIntrst"],
        "lot_size": raw["NewBrdLotQty"], "is_option": raw["FinInstrmTp"].str.strip() == option_kind,
    })
    opt, fut = _split(df)
    return (_traded_or_open(opt) if stocks else opt), fut


def _to_lot(values: pd.Series, stocks: bool = False) -> pd.Series:
    """Index lots: the nearest multiple of 5 (every index lot so far is one).  Stock lots: 95% of
    those >= 100 are multiples of 25, the smaller ones multiples of 5 (UDiFF 2024-26), so snap
    there when within 2%; else the nearest share (lots adjusted for corporate actions)."""
    if not stocks:
        return (values / 5).round() * 5
    grid = np.where(values >= 100, 25, 5)
    snapped = (values / grid).round() * grid
    return snapped.where((snapped - values).abs() <= 0.02 * values, values.round())


def fill_implied_lots(options: pd.DataFrame, futures: pd.DataFrame,
                      stocks: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Give legacy rows a lot size from futures turnover (rows that already have one keep it)."""
    fut = futures.copy()
    traded = (fut["contracts"] > 0) & (fut["close"] > 0)
    implied = fut["value_inr"] / (fut["contracts"] * fut["close"])
    known = fut["lot_size"].fillna(implied.where(traded))
    per_expiry = _to_lot(known.groupby([fut["symbol"], fut["expiry"]]).median(), stocks)  # over its traded days
    per_day = _to_lot(known.groupby([fut["date"], fut["symbol"]]).median(), stocks)
    fut_lots = per_expiry.dropna().rename("lot").reset_index().sort_values("expiry", kind="stable")

    def lookup(df: pd.DataFrame) -> pd.Series:
        """The contract's own future, else the nearest futures expiry (weeklies, long-dated), else the day's median."""
        keys = df[["symbol", "expiry"]].reset_index().sort_values("expiry", kind="stable")
        if len(fut_lots):
            near = pd.merge_asof(keys, fut_lots, on="expiry", by="symbol", direction="nearest")
            lot = pd.Series(near["lot"].to_numpy(), index=keys["index"].to_numpy()).reindex(df.index)
        else:
            lot = pd.Series(np.nan, index=df.index)
        by_day = per_day.reindex(pd.MultiIndex.from_frame(df[["date", "symbol"]])).to_numpy()
        return lot.fillna(pd.Series(by_day, index=df.index))

    fut["lot_size"] = fut["lot_size"].fillna(lookup(fut))
    opt = options.copy()
    opt["lot_size"] = opt["lot_size"].fillna(lookup(opt))
    return opt, fut


# ------------------------------------------------------------- build

def year_files(archive_root: PathLike, year: int) -> List[Path]:
    d = Path(archive_root) / "fo" / str(year)
    return sorted(p for p in d.glob("*.zip") if not p.name.startswith(".")) if d.exists() else []


def fingerprint(files: Sequence[Path], stocks: bool = False) -> str:
    h = hashlib.sha256((f"fo-stock-v{STOCK_STORE_VERSION}" if stocks else f"fo-v{FO_STORE_VERSION}").encode())
    for p in files:
        h.update(f"{p.name}:{p.stat().st_size}".encode())
    return h.hexdigest()[:20]


def _parse_file(path: Path, stocks: bool = False) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    try:
        text = read_raw_text(path)
        parse = parse_udiff_fo if path.name.startswith("udiff_fo") else parse_legacy_fo
        return parse(text, stocks=stocks)
    except Exception as exc:  # a corrupt file must not kill the build
        logger.error("failed to parse %s: %s", path, exc)
        return None


def _build_year(archive_root: str, store_dir: str, year: int, stocks: bool = False) -> Dict[str, object]:
    files = year_files(archive_root, year)
    parts = [r for r in (_parse_file(f, stocks) for f in files) if r is not None]
    opts = pd.concat([p[0] for p in parts], ignore_index=True) if parts else pd.DataFrame(columns=OPTION_COLUMNS)
    futs = pd.concat([p[1] for p in parts], ignore_index=True) if parts else pd.DataFrame(columns=FUTURE_COLUMNS)
    opts, futs = fill_implied_lots(opts, futs, stocks)
    store = Path(store_dir)
    prefix = "stock_" if stocks else ""
    for name, df, keys in ((prefix + "options", opts, ["date", "symbol", "expiry", "option_type", "strike"]),
                           (prefix + "futures", futs, ["date", "symbol", "expiry"])):
        out = store / name / f"{year}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.sort_values(keys).to_parquet(out, index=False)
    return {"year": year, "files": len(files), "parsed": len(parts), "options": len(opts), "futures": len(futs),
            "fingerprint": fingerprint(files, stocks)}


PARTICIPANTS = ("Client", "DII", "FII", "Pro", "TOTAL")
#: NSE's column order, used when a file has no header row (a few do; others lack the title line).
PARTICIPANT_COLUMNS = ["client_type", "future_index_long", "future_index_short", "future_stock_long",
                       "future_stock_short", "option_index_call_long", "option_index_put_long",
                       "option_index_call_short", "option_index_put_short", "option_stock_call_long",
                       "option_stock_put_long", "option_stock_call_short", "option_stock_put_short",
                       "total_long_contracts", "total_short_contracts"]


def parse_participant_oi(text: str, session: pd.Timestamp) -> pd.DataFrame:
    """NSE's ``fao_participant_oi_DDMMYYYY.csv``: usually a title line, a header, then one row
    per client type with open contracts by instrument and side (columns snake_cased)."""
    lines = text.splitlines()
    header = next((i for i, line in enumerate(lines) if line.strip().startswith("Client Type")), None)
    if header is not None:
        raw = pd.read_csv(io.StringIO("\n".join(lines[header:])), dtype=str)
        raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]
    else:
        first = next(i for i, line in enumerate(lines) if line.split(",")[0].strip() in PARTICIPANTS)
        raw = pd.read_csv(io.StringIO("\n".join(lines[first:])), header=None, dtype=str,
                          names=PARTICIPANT_COLUMNS)
    raw["client_type"] = raw["client_type"].str.strip()
    raw = raw[raw["client_type"].isin(PARTICIPANTS)].copy()
    for col in raw.columns.drop("client_type"):
        raw[col] = _num(raw[col])
    raw.insert(0, "date", pd.Timestamp(session))
    return raw.reset_index(drop=True)


def _write_participant(archive_root: PathLike, store: Path) -> int:
    """All archived participant files as one table (a few thousand small CSVs; rebuilt whole)."""
    frames = []
    for path in sorted((Path(archive_root) / "participant").glob("*/poi*.csv")):
        try:
            frames.append(parse_participant_oi(read_raw_text(path), pd.Timestamp(path.stem[3:])))
        except Exception as exc:  # a corrupt file must not kill the build
            logger.error("failed to parse %s: %s", path, exc)
    if not frames:
        return 0
    out = pd.concat(frames, ignore_index=True).sort_values(["date", "client_type"])
    out.to_parquet(store / "participant_oi.parquet", index=False)
    return len(out)


def _write_underlying(equity_stores: Sequence[Path], store: Path) -> int:
    """NSE's index closes, then each store's external history for sessions NSE lacks.

    NSE's daily index files start in February 2012; before that NIFTY 50 comes
    from ``nse_engine.data.external`` (Yahoo ^NSEI from Sep 2007, a Sensex
    proxy before).  ``exact`` marks closes equal to NSE's (NSE or Yahoo ^NSEI),
    the only ones an expiry may settle against.
    """
    nse, ext = [], []
    for eq in equity_stores:
        for path in sorted((eq / "indices").glob("*.parquet")):
            df = pd.read_parquet(path, columns=["date", "index_name", "close"])
            nse.append(df[df["index_name"].isin(INDEX_NAMES.values())].assign(source="nse"))
        for name in INDEX_NAMES.values():
            path = eq / "external" / EXTERNAL_INDEX_FILES[name] if name in EXTERNAL_INDEX_FILES else None
            if path is not None and path.exists():
                e = pd.read_parquet(path)
                ext.append(pd.DataFrame({"date": pd.DatetimeIndex(e.index).normalize(), "index_name": name,
                                         "close": e["close"].to_numpy(), "source": e["source"].to_numpy()}))
    out = pd.concat(nse + ext, ignore_index=True).dropna(subset=["close"])
    out = out.drop_duplicates(["date", "index_name"], keep="first")      # NSE rows come first and win
    out["exact"] = out["source"].isin(["nse", PRIMARY_SOURCE])
    out.sort_values(["index_name", "date"]).to_parquet(store / "underlying.parquet", index=False)
    return len(out)


def build_fo_store(archive_root: PathLike, store_dir: PathLike, equity_stores: Sequence[PathLike],
                   workers: int = 4, stocks: bool = True, stock_from: int = STOCK_FROM) -> Dict[str, object]:
    """Build (or update) the store; a year is rebuilt only when its sources changed."""
    store = Path(store_dir)
    store.mkdir(parents=True, exist_ok=True)
    manifest_path = store / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest.get("version") != FO_STORE_VERSION:
        manifest = {"version": FO_STORE_VERSION, "years": {}}
    years = sorted(int(p.name) for p in (Path(archive_root) / "fo").glob("[0-9]" * 4))
    todo = [y for y in years if manifest["years"].get(str(y), {}).get("fingerprint")
            != fingerprint(year_files(archive_root, y))]
    logger.info("fo store: %d years, %d to build", len(years), len(todo))
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for info in pool.map(_build_year, [str(archive_root)] * len(todo), [str(store)] * len(todo), todo):
            manifest["years"][str(info["year"])] = info
            logger.info("year %s: %s", info["year"], info)
    if stocks:
        done = manifest.setdefault("stock_years", {})
        stock_todo = [y for y in years if y >= stock_from and done.get(str(y), {}).get("fingerprint")
                      != fingerprint(year_files(archive_root, y), stocks=True)]
        logger.info("fo store: %d stock years to build", len(stock_todo))
        with ProcessPoolExecutor(max_workers=max(1, min(workers, 2))) as pool:   # stock years are large
            n = len(stock_todo)
            for info in pool.map(_build_year, [str(archive_root)] * n, [str(store)] * n, stock_todo, [True] * n):
                done[str(info["year"])] = info
                logger.info("stock year %s: %s", info["year"], info)
    manifest["participant_rows"] = _write_participant(archive_root, store)
    manifest["underlying_rows"] = _write_underlying([Path(p) for p in equity_stores], store)
    manifest["data_hash"] = hashlib.sha256(json.dumps(
        {y: v["fingerprint"] for y, v in sorted(manifest["years"].items())}, sort_keys=True).encode()).hexdigest()[:16]
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest


# ------------------------------------------------------------- read

def _read_years(store_dir: PathLike, table: str, start: str, end: str, symbol: str) -> pd.DataFrame:
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    frames = []
    for year in range(lo.year, hi.year + 1):
        path = Path(store_dir) / table / f"{year}.parquet"
        if path.exists():
            df = pd.read_parquet(path, filters=[("symbol", "==", symbol)])
            frames.append(df[(df["date"] >= lo) & (df["date"] <= hi)])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_options(store_dir: PathLike, start: str, end: str, symbol: str = "NIFTY") -> pd.DataFrame:
    return _read_years(store_dir, "options", start, end, symbol)


def load_futures(store_dir: PathLike, start: str, end: str, symbol: str = "NIFTY") -> pd.DataFrame:
    return _read_years(store_dir, "futures", start, end, symbol)


def load_stock_options(store_dir: PathLike, start: str, end: str, symbol: str) -> pd.DataFrame:
    return _read_years(store_dir, "stock_options", start, end, symbol)


def load_stock_futures(store_dir: PathLike, start: str, end: str, symbol: str) -> pd.DataFrame:
    return _read_years(store_dir, "stock_futures", start, end, symbol)


def load_participant_oi(store_dir: PathLike) -> pd.DataFrame:
    path = Path(store_dir) / "participant_oi.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def load_underlying(store_dir: PathLike, symbol: str = "NIFTY", exact_only: bool = False) -> pd.Series:
    df = pd.read_parquet(Path(store_dir) / "underlying.parquet")
    df = df[df["index_name"] == INDEX_NAMES[symbol]]
    if exact_only and "exact" in df:
        df = df[df["exact"]]
    return pd.Series(df["close"].to_numpy(dtype=float), index=pd.DatetimeIndex(df["date"]), name=symbol).sort_index()


def data_hash(store_dir: PathLike) -> str:
    return json.loads((Path(store_dir) / "manifest.json").read_text()).get("data_hash", "")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Build the F&O derivatives parquet store")
    p.add_argument("--archive", default="data/nse_engine/archive")
    p.add_argument("--store", default="data/nse_engine/fo_store")
    p.add_argument("--equity-store", action="append",
                   help="repeatable; default: data/nse_engine/store, then store_ext2006 for pre-2012 NIFTY")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-stocks", action="store_true", help="skip the stock derivatives tables (OD3)")
    p.add_argument("--stock-from", type=int, default=STOCK_FROM)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    stores = args.equity_store or [p for p in ("data/nse_engine/store", "data/nse_engine/store_ext2006")
                                   if Path(p).is_dir()]
    manifest = build_fo_store(args.archive, args.store, stores, args.workers, not args.no_stocks, args.stock_from)
    logger.info("done: data hash %s, %d years", manifest["data_hash"], len(manifest["years"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
