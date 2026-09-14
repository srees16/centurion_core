"""
Load the parquet store into a date-aligned, adjusted ``MarketData`` panel.

Pipeline (``load_market_data``):

1. read equity rows in [start, end] for the requested series;
2. map every (symbol, date) to its canonical name (``reference.resolve_symbols``
   over symbolchange.csv + ISIN continuity from the whole store);
3. keep one row per (date, canonical symbol), preferring EQ over BE;
4. drop symbols whose 126-session rolling median traded value never reaches
   ``min_median_value_inr`` (``include_symbols`` always kept);
5. pivot onto the NSE trading calendar and back-adjust prices (volumes
   inversely) - see ``adjustment_multipliers``.  NSE bhavcopy PREVCLOSE is
   *not* adjusted for splits/bonuses (checked 2012, 2019, 2025), so the
   primary source is the corporate-action (Bc) file of the daily PR zip;
6. attach delivery %, ETF set, sectors and index closes (India VIX gaps
   back-filled from yfinance ``^INDIAVIX``).
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from nse_engine.data import reference
from nse_engine.data.validation import factors_from_prev_close, snap_factor
from nse_engine.types import MarketData

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
DateLike = Union[str, date, datetime, pd.Timestamp]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SECTOR_MAP = REPO_ROOT / "data" / "nse_sector_map.json"
INDEX_NAMES = ("NIFTY50", "NIFTY500", "INDIAVIX")
LIQUIDITY_WINDOW = 126
#: Factors this far from 1 must shrink the observed gap, else they are data errors.
LARGE_FACTOR_LOG = 0.10
FACTOR_BOUNDS = (1.0 / 250.0, 50.0)
_READ_COLUMNS = ["date", "symbol", "series", "isin", "open", "high", "low", "close",
                 "prev_close", "volume", "value_inr", "deliv_pct"]


# ------------------------------------------------------------------ reads
def _years(start: pd.Timestamp, end: pd.Timestamp) -> range:
    return range(start.year, end.year + 1)


def read_equity_rows(store_dir: PathLike, start: pd.Timestamp, end: pd.Timestamp,
                     series: Sequence[str]) -> pd.DataFrame:
    """Equity rows in [start, end] for ``series`` (numerics as float32)."""
    parts = []
    for y in _years(start, end):
        path = Path(store_dir) / "equity" / f"{y}.parquet"
        if not path.exists():
            continue
        table = pq.read_table(path, columns=_READ_COLUMNS, read_dictionary=["symbol", "series", "isin"],
                              filters=[("date", ">=", start), ("date", "<=", end), ("series", "in", list(series))])
        df = table.to_pandas(self_destruct=True, split_blocks=True)
        del table
        pa.default_memory_pool().release_unused()
        num = ["open", "high", "low", "close", "prev_close", "volume", "value_inr", "deliv_pct"]
        df[num] = df[num].astype("float32")
        parts.append(df)
    if not parts:
        raise FileNotFoundError(f"no equity parquet for {start.date()}..{end.date()} under {store_dir}")
    rows = pd.concat(parts, ignore_index=True)
    for col in ("symbol", "series", "isin"):  # categories differ per year: unify
        rows[col] = rows[col].astype("string").astype("category") if rows[col].dtype != "category" else rows[col]
    rows["date"] = rows["date"].astype("datetime64[ns]")
    return rows


def read_indices(store_dir: PathLike, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    parts = []
    for y in _years(start, end):
        path = Path(store_dir) / "indices" / f"{y}.parquet"
        if path.exists():
            parts.append(pd.read_parquet(path, filters=[("date", ">=", start), ("date", "<=", end)]))
    if not parts:
        return pd.DataFrame(columns=["date", "index_name", "open", "high", "low", "close"])
    out = pd.concat(parts, ignore_index=True)
    out["date"] = out["date"].astype("datetime64[ns]")
    return out


def _read_optional(path: Path) -> Optional[pd.DataFrame]:
    return pd.read_parquet(path) if path.exists() else None


def _reference_roots(store_dir: PathLike) -> List[Path]:
    store = Path(store_dir)
    return [store, store.parent / "archive"]


def load_change_table(store_dir: PathLike) -> pd.DataFrame:
    """Rename table from the store's reference copy + ISIN continuity over the whole store."""
    root = next((r for r in _reference_roots(store_dir) if reference.reference_path(r, reference.SYMBOL_CHANGES)),
                None)
    changes = (reference.load_symbol_changes(root) if root is not None
               else pd.DataFrame(columns=["old", "new", "date"]))
    spans = _read_optional(Path(store_dir) / "spans.parquet")
    cal = _read_optional(Path(store_dir) / "calendar.parquet")
    calendar = pd.DatetimeIndex(cal["date"]) if cal is not None else None
    return reference.build_change_table(changes, spans, calendar)


def load_corporate_actions(store_dir: PathLike, dates: pd.DatetimeIndex, changes: pd.DataFrame) -> pd.DataFrame:
    """Store corporate-action events inside the calendar, with canonical symbols."""
    path = Path(store_dir) / "corporate_actions.parquet"
    if not path.exists() or len(dates) == 0:
        logger.warning("no corporate_actions.parquet in %s: splits/bonuses rely on prev_close/inference", store_dir)
        return pd.DataFrame()
    ev = pd.read_parquet(path)
    ev = ev[(ev["ex_date"] > dates[0]) & (ev["ex_date"] <= dates[-1])].reset_index(drop=True)
    if ev.empty:
        return ev
    ev["canonical"] = reference.resolve_symbols(ev["symbol"], ev["ex_date"], changes).to_numpy()
    return ev


def _face_values(store_dir: PathLike) -> Dict[str, float]:
    root = next((r for r in _reference_roots(store_dir) if reference.reference_path(r, reference.EQUITY_LIST)), None)
    return reference.load_face_values(root) if root is not None else {}


# -------------------------------------------------------------- transforms
def _as_category(values: pd.Series) -> pd.Categorical:
    return values.array if isinstance(values.dtype, pd.CategoricalDtype) else pd.Categorical(values.astype(str))


def canonicalise(rows: pd.DataFrame, changes: pd.DataFrame, series: Sequence[str]) -> pd.DataFrame:
    """Add categorical ``canonical``; keep one row per (date, canonical), preferring earlier ``series``.

    Only rows whose symbol appears in the rename table are resolved
    row-by-row; everything else is handled on category codes.
    """
    sym = _as_category(rows["symbol"])
    olds = set(changes["old"]) if changes is not None and len(changes) else set()
    needs = np.asarray(pd.Index(sym.categories).isin(list(olds)))[sym.codes]
    canon = np.asarray(sym.categories.astype(str), dtype=object)[sym.codes]
    if needs.any():
        sub = rows.loc[needs]
        canon[needs] = reference.resolve_symbols(sub["symbol"].astype(str), sub["date"], changes).to_numpy()
    canonical = pd.Categorical(canon)
    del canon

    ser = _as_category(rows["series"])
    rank_of = np.array([list(series).index(c) if c in series else len(series) for c in ser.categories.astype(str)],
                       dtype="int8")
    rank = rank_of[ser.codes] if len(rank_of) else np.zeros(len(rows), dtype="int8")
    date_i8 = rows["date"].to_numpy().astype("int64")
    ccode = canonical.codes.astype("int32")
    order = np.lexsort((-rows["value_inr"].to_numpy(dtype="float64"), rank, ccode, date_i8))
    d_sorted, c_sorted = date_i8[order], ccode[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = (d_sorted[1:] != d_sorted[:-1]) | (c_sorted[1:] != c_sorted[:-1])
    dup_pos, kept_pos = order[~first], order[np.maximum.accumulate(np.where(first, np.arange(len(order)), 0))][~first]
    n_clash = int((sym.codes[dup_pos] != sym.codes[kept_pos]).sum())
    if n_clash:
        logger.warning("%d rows collided after symbol linking (kept higher-priority series / value)", n_clash)
    if first.all():  # no duplicates: avoid copying the frame
        rows["canonical"] = canonical
        return rows
    keep = np.sort(order[first])
    out = rows.iloc[keep].reset_index(drop=True)
    out["canonical"] = canonical[keep]
    return out


def pivot(rows: pd.DataFrame, column: str, dates: pd.DatetimeIndex, symbols: Sequence[str],
          dtype: str = "float64") -> pd.DataFrame:
    """Dense date x symbol frame of ``column`` (NaN where not traded)."""
    arr = np.full((len(dates), len(symbols)), np.nan, dtype=dtype)
    di = rows["_di"].to_numpy() if "_di" in rows else dates.get_indexer(rows["date"])
    canon = rows["canonical"]
    if isinstance(canon.dtype, pd.CategoricalDtype):
        lookup = pd.Index(symbols).get_indexer(canon.cat.categories.astype(str))
        si = np.where(canon.cat.codes.to_numpy() >= 0, lookup[canon.cat.codes.to_numpy()], -1)
    else:
        si = pd.Index(symbols).get_indexer(canon)
    ok = (di >= 0) & (si >= 0)
    arr[di[ok], si[ok]] = rows[column].to_numpy(dtype=dtype)[ok]
    return pd.DataFrame(arr, index=dates, columns=list(symbols))


def liquid_symbols(value: pd.DataFrame, min_median_value_inr: float,
                   include: Iterable[str] = (), window: int = LIQUIDITY_WINDOW,
                   min_sessions: int = 21) -> List[str]:
    """Symbols whose rolling median traded value ever reaches the threshold.

    Sessions after a symbol's first trade with no trade count as zero value;
    sessions before it are ignored, so a new listing qualifies once it has
    ``min_sessions`` sessions of history (within ``window``).
    """
    listed = value.notna().cumsum() > 0
    filled = value.where(~listed, value.fillna(0.0))
    min_periods = max(1, min(window, min_sessions, len(value)))
    med = filled.rolling(window, min_periods=min_periods).median()
    peak = med.max(axis=0)
    keep = set(peak.index[peak >= min_median_value_inr]) | (set(include) & set(value.columns))
    return [c for c in value.columns if c in keep]


def _plausible(factor: np.ndarray, close_next: np.ndarray, prior: np.ndarray, slack: float,
               min_log: float, improve: float = 0.0) -> np.ndarray:
    """True where applying ``factor`` does not widen the observed gap.

    Requires ``adj_gap <= raw_gap + slack - improve * |log f|`` (``improve`` > 0
    demands that the factor explains part of the gap).  Factors with
    |log f| <= ``min_log`` pass unchecked; cells without prices pass.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.abs(np.log(close_next / prior))
        adj = np.abs(np.log(close_next / (prior * factor)))
        small = np.abs(np.log(factor)) <= min_log
    unknown = ~np.isfinite(raw)
    in_bounds = (factor >= FACTOR_BOUNDS[0]) & (factor <= FACTOR_BOUNDS[1])
    with np.errstate(invalid="ignore"):
        explains = adj <= raw + slack - improve * np.abs(np.log(factor))
    return in_bounds & (small | unknown | explains)


def event_factor_frames(events: pd.DataFrame, dates: pd.DatetimeIndex, symbols: Sequence[str],
                        prior: np.ndarray, open_next: np.ndarray,
                        face_values: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
    """Candidate factors from corporate-action events, placed on the first session >= ex-date.

    Returns arrays (dates x symbols, NaN = none) for ``ratio`` (bonus / split /
    consolidation), ``rights`` (theoretical ex-rights price) and ``price``
    (demerger / capital reduction: ex-date open over prior close).
    """
    shape = (len(dates), len(symbols))
    out = {k: np.full(shape, np.nan) for k in ("ratio", "rights", "price")}
    if events is None or events.empty:
        return out
    col = pd.Index(symbols)
    ev = events.copy()
    ev["di"] = dates.searchsorted(pd.DatetimeIndex(ev["ex_date"]))
    ev["si"] = col.get_indexer(ev["canonical"])
    ev = ev[(ev["si"] >= 0) & (ev["di"] > 0) & (ev["di"] < len(dates))]
    fv = face_values or {}
    for r in ev.itertuples(index=False):
        di, si = int(r.di), int(r.si)
        if pd.notna(r.factor):
            cur = out["ratio"][di, si]
            out["ratio"][di, si] = r.factor if np.isnan(cur) else cur * r.factor
        elif pd.notna(r.rights_new) and pd.notna(r.rights_premium):
            p = prior[di, si]
            issue = r.rights_premium + fv.get(r.canonical, fv.get(r.symbol, 0.0))
            if np.isfinite(p) and p > issue > 0:
                a, b = r.rights_new, r.rights_held
                out["rights"][di, si] = (b * p + a * issue) / ((a + b) * p)
        elif pd.notna(r.price_based):
            p, o = prior[di, si], open_next[di, si]
            if np.isfinite(p) and np.isfinite(o) and p > 0 and abs(np.log(o / p)) > 0.03:
                out["price"][di, si] = o / p
    return out


def inferred_factors(close: np.ndarray, open_: np.ndarray, prior: np.ndarray, taken: np.ndarray,
                     threshold: float = 0.35, persist_days: int = 5, persist_tolerance: float = 0.20) -> np.ndarray:
    """Price-inferred factors for large unexplained gaps (mirrors ``validation.clean_ohlcv``).

    A session that opened >= ``threshold`` away from the prior close, closed
    near its open, and whose level persists is treated as an ex-date; the
    ratio is snapped to a common split/bonus factor when close.  Genuine
    crashes (e.g. intraday recoveries) are left alone.
    """
    out = np.full(close.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        obs = close / prior
    lo, hi = 1.0 - threshold, 1.0 / (1.0 - threshold)
    cand = np.isfinite(obs) & ((obs <= lo) | (obs >= hi)) & ~taken
    for t, s in zip(*np.nonzero(cand)):
        c, o = close[t, s], open_[t, s]
        window = close[t: t + persist_days, s]
        window = window[np.isfinite(window)]
        if abs(float(np.median(window)) / c - 1.0) > persist_tolerance:
            continue
        gap = o / prior[t, s] if np.isfinite(o) and o > 0 else obs[t, s]
        if lo < gap < hi or abs(c / o - 1.0) > 0.25:
            continue
        out[t, s] = snap_factor(float(obs[t, s]))[0]
    return out


def adjustment_multipliers(close: pd.DataFrame, prev_close: pd.DataFrame, open_: Optional[pd.DataFrame] = None,
                           events: Optional[pd.DataFrame] = None, face_values: Optional[Dict[str, float]] = None,
                           tolerance: float = 0.002, infer_gaps: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cumulative back-adjustment multipliers and the per-date factor frame used.

    Factor sources, first match wins per (date, symbol):

    1. corporate-action ratios (bonus / split / consolidation) from NSE ``Bc`` files;
    2. ``factors_from_prev_close`` - bhavcopy PREVCLOSE differing from the
       last close (NSE adjusts it for some actions, e.g. older demergers);
    3. rights (theoretical ex-rights price) and demerger / capital-reduction
       gaps (ex-date open / prior close) from ``Bc`` files;
    4. price-inferred factors for large, persistent, unexplained gaps.

    Every factor must not widen the observed close-to-close gap (guards wrong
    ex-dates and bad prev_close values).  As in ``adjust_for_factors``, rows
    strictly before an ex-date are multiplied by its factor.
    """
    close64 = close.to_numpy(dtype="float64")
    idx, cols = close.index, list(close.columns)
    prior = close.astype("float64").ffill().shift(1).to_numpy()
    close_next = close.astype("float64").bfill().to_numpy()
    open64 = (open_ if open_ is not None else close).to_numpy(dtype="float64")
    open_next = pd.DataFrame(open64, index=idx).bfill().to_numpy()

    factors = np.full(close64.shape, np.nan)
    sources = np.zeros(close64.shape, dtype="int8")

    def take(candidate: np.ndarray, code: int, slack: float, min_log: float, improve: float = 0.0) -> None:
        ok = np.isfinite(candidate) & np.isnan(factors) & _plausible(candidate, close_next, prior, slack, min_log,
                                                                     improve)
        rejected = np.isfinite(candidate) & np.isnan(factors) & ~ok
        for t, s in list(zip(*np.nonzero(rejected)))[:10]:
            logger.info("rejected factor source=%d %s %s f=%.4g (inconsistent with prices)",
                        code, cols[s], idx[t].date(), candidate[t, s])
        factors[ok] = candidate[ok]
        sources[ok] = code

    ev = event_factor_frames(events, idx, cols, prior, open_next, face_values)
    take(ev["ratio"], 1, slack=0.02, min_log=LARGE_FACTOR_LOG)

    pc = factors_from_prev_close(close.astype("float64").ffill(),
                                 prev_close.astype("float64").where(close.notna()), tolerance=tolerance)
    pc_arr = np.array(pc.to_numpy(dtype="float64"), copy=True)
    # Many symbols disagreeing on one date means a missing session in the data
    # (e.g. an un-archived Saturday session), not corporate actions.
    per_date = np.isfinite(pc_arr).sum(axis=1)
    active = np.isfinite(close64).sum(axis=1)
    mass = per_date > np.maximum(25, 0.2 * active)
    for t in np.flatnonzero(mass):
        logger.warning("%s: %d symbols have prev_close != last close; missing session before this date? "
                       "ignoring prev_close factors for it", idx[t].date(), per_date[t])
    pc_arr[mass] = np.nan
    take(pc_arr, 3, slack=0.0, min_log=0.0, improve=0.5)

    take(ev["rights"], 2, slack=0.02, min_log=LARGE_FACTOR_LOG)
    take(ev["price"], 2, slack=0.02, min_log=0.0)

    if infer_gaps:
        inf = inferred_factors(close64, open64, prior, np.isfinite(factors))
        for t, s in zip(*np.nonzero(np.isfinite(inf))):
            logger.warning("inferred corporate action %s %s factor=%.4g (no NSE record)", cols[s], idx[t].date(), inf[t, s])
        take(inf, 4, slack=0.02, min_log=0.0)

    counts = {name: int((sources == code).sum()) for code, name in
              ((1, "ca_ratio"), (2, "ca_rights_or_price"), (3, "prev_close"), (4, "inferred"))}
    logger.info("adjustment factors by source: %s", counts)
    f = np.where(np.isfinite(factors), factors, 1.0)
    mult = np.cumprod(f[::-1], axis=0)[::-1] / f  # product of factors with ex-date > t
    factor_frame = pd.DataFrame(factors, index=idx, columns=cols).dropna(how="all", axis=1)
    return pd.DataFrame(mult, index=idx, columns=cols), factor_frame


def build_index_close(indices: pd.DataFrame, dates: pd.DatetimeIndex,
                      names: Sequence[str] = INDEX_NAMES) -> pd.DataFrame:
    sub = indices[indices["index_name"].astype(str).isin(names)]
    wide = sub.pivot_table(index="date", columns="index_name", values="close", aggfunc="last")
    wide.columns = [str(c) for c in wide.columns]
    return wide.reindex(index=dates, columns=list(names)).astype("float64")


def fetch_yf_vix(start: pd.Timestamp, end: pd.Timestamp, cache_path: Optional[Path] = None) -> pd.Series:
    """India VIX closes from yfinance (cached to parquet). Empty on failure."""
    if cache_path is not None and cache_path.exists():
        cached = pd.read_parquet(cache_path)["close"]
        cached.index = pd.DatetimeIndex(cached.index)
        if len(cached) and cached.index.min() <= start + pd.Timedelta(days=7) and cached.index.max() >= min(
                end, pd.Timestamp.today().normalize() - pd.Timedelta(days=5)):
            return cached
    try:
        import yfinance as yf

        raw = yf.download("^INDIAVIX", start="2008-01-01", progress=False, auto_adjust=False)
    except Exception as exc:  # network / API failure must not break loading
        logger.warning("yfinance ^INDIAVIX download failed: %s", exc)
        return pd.Series(dtype="float64")
    if raw is None or raw.empty:
        return pd.Series(dtype="float64")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    close = close.dropna().astype("float64").rename("close")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        close.to_frame().to_parquet(cache_path)
    return close


def backfill_vix(index_close: pd.DataFrame, vix: pd.Series) -> Tuple[pd.DataFrame, int]:
    """Fill missing INDIAVIX from ``vix`` aligned by date (no forward-filling)."""
    if "INDIAVIX" not in index_close or vix is None or vix.empty:
        return index_close, 0
    out = index_close.copy()
    gap = out["INDIAVIX"].isna()
    fill = vix.reindex(out.index)
    n = int((gap & fill.notna()).sum())
    out.loc[gap, "INDIAVIX"] = fill[gap]
    return out, n


def etf_symbols(store_dir: PathLike, rows: pd.DataFrame, symbols: Sequence[str]) -> frozenset:
    """Current NSE ETF list plus securities with mutual-fund (INF) ISINs, within ``symbols``.

    The ISIN rule catches ETFs that were delisted and so are absent from
    today's ``eq_etfseclist.csv``.
    """
    root = next((r for r in _reference_roots(store_dir) if reference.reference_path(r, reference.ETF_LIST)), None)
    listed = set(reference.load_etf_symbols(root)) if root is not None else set()
    isin = _as_category(rows["isin"])
    inf_codes = np.flatnonzero(pd.Index(isin.categories.astype(str)).str.startswith("INF"))
    mask = np.isin(isin.codes, inf_codes)
    inf = set(pd.unique(rows.loc[mask, "canonical"].astype(str)))
    return frozenset((listed | inf) & set(symbols))


# ------------------------------------------------------------------- main
def load_market_data(
    store_dir: PathLike,
    start: DateLike,
    end: DateLike,
    *,
    symbols: Optional[Sequence[str]] = None,
    series: Sequence[str] = ("EQ", "BE"),
    min_median_value_inr: float = 2.5e6,
    float_dtype: str = "float32",
    include_symbols: Sequence[str] = ("GOLDBEES", "SILVERBEES"),
    vix_fallback: bool = True,
    sector_map_path: Optional[PathLike] = None,
    index_names: Sequence[str] = INDEX_NAMES,
) -> MarketData:
    """Build an adjusted, calendar-aligned ``MarketData`` from the parquet store."""
    start_ts, end_ts = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    store = Path(store_dir)
    rows = read_equity_rows(store, start_ts, end_ts, series)
    dates = pd.DatetimeIndex(np.sort(rows["date"].unique()), name="date")
    changes = load_change_table(store)
    rows = canonicalise(rows, changes, series)
    rows["_di"] = dates.get_indexer(rows["date"]).astype("int32")
    if symbols is not None:
        wanted = set(symbols) | set(include_symbols)
        rows = rows[rows["canonical"].isin(wanted)]
    rows["canonical"] = rows["canonical"].cat.remove_unused_categories()
    all_syms = sorted(map(str, rows["canonical"].cat.categories))
    value = pivot(rows, "value_inr", dates, all_syms, dtype="float32")
    keep = liquid_symbols(value, min_median_value_inr, include_symbols)
    logger.info("load_market_data %s..%s: %d sessions, %d symbols, %d after liquidity pre-filter (>= %.3g INR)",
                start_ts.date(), end_ts.date(), len(dates), len(all_syms), len(keep), min_median_value_inr)
    rows = rows[rows["canonical"].isin(set(keep))].drop(columns=["series", "date"])
    value = value[keep]

    close = pivot(rows, "close", dates, keep)
    open_ = pivot(rows, "open", dates, keep)
    events = load_corporate_actions(store, dates, changes)
    mult, factors = adjustment_multipliers(close, pivot(rows, "prev_close", dates, keep), open_, events,
                                           _face_values(store))
    n_adj = int(factors.notna().to_numpy().sum())
    logger.info("applied %d corporate-action adjustments across %d symbols", n_adj, factors.shape[1])
    frames = {"close": close * mult, "open": open_ * mult}
    for col in ("high", "low"):
        frames[col] = pivot(rows, col, dates, keep) * mult
    frames["volume"] = pivot(rows, "volume", dates, keep) / mult
    frames = {k: v.astype(float_dtype) for k, v in frames.items()}

    deliv = pivot(rows, "deliv_pct", dates, keep, dtype=float_dtype)
    delivery_pct = deliv if deliv.notna().to_numpy().any() else None

    index_close = build_index_close(read_indices(store, start_ts, end_ts), dates, index_names)
    if vix_fallback and "INDIAVIX" in index_close and index_close["INDIAVIX"].isna().any():
        vix = fetch_yf_vix(start_ts, end_ts, store / "external" / "yf_indiavix.parquet")
        index_close, n_filled = backfill_vix(index_close, vix)
        logger.info("INDIAVIX: %d/%d sessions back-filled from yfinance ^INDIAVIX; %d still missing",
                    n_filled, len(dates), int(index_close["INDIAVIX"].isna().sum()))

    sector_paths = [sector_map_path] if sector_map_path else [DEFAULT_SECTOR_MAP, store / "reference" / "nse_sector_map.json"]
    sector_map = reference.load_sector_map(sector_paths)
    sectors: Dict[str, str] = {s: sector_map[s] for s in keep if s in sector_map}

    data = MarketData(
        dates=dates,
        open=frames["open"], high=frames["high"], low=frames["low"], close=frames["close"],
        volume=frames["volume"], value=value.astype(float_dtype),
        index_close=index_close.astype(float_dtype),
        delivery_pct=delivery_pct,
        etfs=etf_symbols(store, rows, keep),
        sectors=sectors,
        source=f"nse_bhavcopy:{store}",
    )
    data.data_hash = data.compute_hash()
    data.validate()
    return data
