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
6. with ``adjust_dividends`` (default), back-adjust prices for cash dividends
   as well (total-return prices, volumes untouched) - see
   ``dividend_multipliers``;
7. attach delivery %, ETF set, sectors and index closes (India VIX gaps
   back-filled from yfinance ``^INDIAVIX``; ``NIFTY50_TRI`` derived from NSE's
   Nifty 50 dividend-points index - NSE's daily files carry no TRI series).
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
from nse_engine.data.validation import CORPORATE_RATIOS, factors_from_prev_close, snap_factor
from nse_engine.types import MarketData

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
DateLike = Union[str, date, datetime, pd.Timestamp]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SECTOR_MAP = REPO_ROOT / "data" / "nse_sector_map.json"
INDEX_NAMES = ("NIFTY50", "NIFTY500", "INDIAVIX", "NIFTY50_TRI")
#: Derived total-return indices: name -> (price index, dividend-points index).
TRI_SOURCES = {"NIFTY50_TRI": ("NIFTY50", "NIFTY50DIVIDENDPOINTS")}
LIQUIDITY_WINDOW = 126
#: Factors this far from 1 must shrink the observed gap, else they are data errors.
LARGE_FACTOR_LOG = 0.10
FACTOR_BOUNDS = (1.0 / 250.0, 50.0)
#: Snap tolerances for price-measured ratios (relative), and on ISIN-change dates
#: (face-value splits change the ISIN, so a large gap there is almost surely a split).
SNAP_TOLERANCE = 0.06
SNAP_TOLERANCE_ISIN = 0.12
#: No price inference below this prior close: Rs 0.05 ticks move such prices 2x.
MIN_INFER_PRICE = 1.0
#: Dividends at or above this share of the prior close are treated as parse errors.
MAX_DIVIDEND_YIELD = 0.5
#: Factor source codes (``adjustment_multipliers``).
SOURCE_NAMES = {1: "ca_ratio", 2: "ca_rights_or_price", 3: "prev_close", 4: "inferred_snapped",
                5: "inferred_unexplained", 6: "ca_ratio_from_prices"}
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
    for col in reference.EVENT_COLUMNS:
        if col not in ev:  # stores built before a column existed
            ev[col] = np.nan
    ev = ev[(ev["ex_date"] > dates[0]) & (ev["ex_date"] <= dates[-1])].reset_index(drop=True)
    if ev.empty:
        return ev
    ev["canonical"] = reference.resolve_symbols(ev["symbol"], ev["ex_date"], changes).to_numpy()
    return ev


def load_dividends(store_dir: PathLike, dates: pd.DatetimeIndex, changes: pd.DataFrame) -> pd.DataFrame:
    """Store cash-dividend events inside the calendar, with canonical symbols."""
    path = Path(store_dir) / "dividends.parquet"
    if not path.exists() or len(dates) == 0:
        logger.warning("no dividends.parquet in %s (rebuild the store): prices are not dividend-adjusted", store_dir)
        return pd.DataFrame(columns=reference.DIVIDEND_COLUMNS + ["canonical"])
    dv = pd.read_parquet(path)
    dv = dv[(dv["ex_date"] > dates[0]) & (dv["ex_date"] <= dates[-1])].reset_index(drop=True)
    if len(dv):
        dv["canonical"] = reference.resolve_symbols(dv["symbol"], dv["ex_date"], changes).to_numpy()
    else:
        dv["canonical"] = pd.Series(dtype=object)
    return dv


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


def isin_change_frame(rows: pd.DataFrame, dates: pd.DatetimeIndex, symbols: Sequence[str]) -> np.ndarray:
    """Bool (dates x symbols): the row's ISIN differs from the symbol's previous traded row.

    Face-value splits and consolidations change the ISIN, bonuses do not.
    """
    out = np.zeros((len(dates), len(symbols)), dtype=bool)
    if rows.empty or "isin" not in rows:
        return out
    isin = _as_category(rows["isin"]).codes.astype("int64")
    canon = rows["canonical"]
    if isinstance(canon.dtype, pd.CategoricalDtype):
        lookup = pd.Index(symbols).get_indexer(canon.cat.categories.astype(str))
        codes = canon.cat.codes.to_numpy()
        si = np.where(codes >= 0, lookup[codes], -1)
    else:
        si = pd.Index(symbols).get_indexer(canon)
    di = rows["_di"].to_numpy() if "_di" in rows else dates.get_indexer(rows["date"])
    order = np.lexsort((di, si))
    s_o, d_o, i_o = si[order], di[order], isin[order]
    change = np.zeros(len(order), dtype=bool)
    change[1:] = (s_o[1:] == s_o[:-1]) & (i_o[1:] != i_o[:-1]) & (i_o[1:] >= 0) & (i_o[:-1] >= 0)
    change &= (s_o >= 0) & (d_o >= 0)
    out[d_o[change], s_o[change]] = True
    return out


def event_factor_frames(events: pd.DataFrame, dates: pd.DatetimeIndex, symbols: Sequence[str],
                        prior: np.ndarray, open_next: np.ndarray,
                        face_values: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
    """Candidate factors from corporate-action events, placed on the first session >= ex-date.

    Returns arrays (dates x symbols, NaN = none) for ``ratio`` (bonus / split /
    consolidation), ``hint`` (-1 split/bonus, +1 consolidation with an unstated
    ratio), ``rights`` (theoretical ex-rights price) and ``price`` (demerger /
    capital reduction: ex-date open over prior close).
    """
    shape = (len(dates), len(symbols))
    out = {k: np.full(shape, np.nan) for k in ("ratio", "hint", "rights", "price")}
    if events is None or events.empty:
        return out
    col = pd.Index(symbols)
    ev = events.copy()
    if "hint" not in ev:
        ev["hint"] = np.nan
    ev["di"] = dates.searchsorted(pd.DatetimeIndex(ev["ex_date"]))
    ev["si"] = col.get_indexer(ev["canonical"])
    ev = ev[(ev["si"] >= 0) & (ev["di"] > 0) & (ev["di"] < len(dates))]
    fv = face_values or {}
    for r in ev.itertuples(index=False):
        di, si = int(r.di), int(r.si)
        if pd.notna(r.factor):
            cur = out["ratio"][di, si]
            out["ratio"][di, si] = r.factor if np.isnan(cur) else cur * r.factor
        elif pd.notna(r.hint):
            out["hint"][di, si] = r.hint
            if pd.notna(r.price_based):
                p, o = prior[di, si], open_next[di, si]
                if np.isfinite(p) and np.isfinite(o) and p > 0 and abs(np.log(o / p)) > 0.03:
                    out["price"][di, si] = o / p
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


def snap_gap(open_gap: float, close_gap: float, tolerance: float) -> Tuple[float, bool]:
    """Snap a measured price-scale change to ``CORPORATE_RATIOS``.

    Both the ex-date open / prior close and close / prior close are snapped;
    the estimate closer to its snapped ratio wins (the open is NSE's price
    discovery around the adjusted base price but can be a stale auction print
    for ETFs; the close carries the day's market move).  Returns
    ``(close_gap, False)`` when neither is within ``tolerance``.
    """
    best: Optional[Tuple[float, float]] = None
    for gap in (open_gap, close_gap):
        if np.isfinite(gap) and gap > 0:
            f, ok = snap_factor(float(gap), tolerance, CORPORATE_RATIOS)
            err = abs(np.log(gap / f))
            if ok and (best is None or err < best[1]):
                best = (f, err)
    return (best[0], True) if best is not None else (close_gap, False)


def dividend_yields(amounts: np.ndarray, prior: np.ndarray, ex_price: np.ndarray,
                    max_yield: float = MAX_DIVIDEND_YIELD) -> Tuple[np.ndarray, np.ndarray]:
    """Dividend yields ``D / prior`` (NaN where none) and the mask of implausible ones.

    A yield >= ``max_yield`` is kept only when the ex-date price confirms it
    (MAJESCO's Rs 974 on ~Rs 986 in 2020, STAR's Rs 500 on Rs 882 in 2013):
    ``prior - D`` must explain at least half of the observed gap in log terms.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        y = amounts / prior
        big = np.isfinite(y) & (y >= max_yield)
        raw_gap = np.abs(np.log(ex_price / prior))
        div_gap = np.abs(np.log(ex_price / (prior * (1.0 - y))))
        confirmed = big & (y < 1.0) & np.isfinite(div_gap) & (div_gap < 0.5 * raw_gap)
    bad = big & ~confirmed
    y = np.where(np.isfinite(y) & (y > 0) & ~bad, y, np.nan)
    return y, bad


def inferred_factors(close: np.ndarray, open_: np.ndarray, prior: np.ndarray, taken: np.ndarray,
                     isin_change: Optional[np.ndarray] = None, threshold: float = 0.35,
                     isin_threshold: float = 0.25, persist_days: int = 5, persist_tolerance: float = 0.20,
                     min_price: float = MIN_INFER_PRICE) -> Tuple[np.ndarray, np.ndarray]:
    """Price-inferred factors for large unexplained gaps (mirrors ``validation.clean_ohlcv``).

    A session that opened >= ``threshold`` away from the prior close, closed
    near its open, and whose level persists is treated as an ex-date.  On an
    ISIN-change date the gap only needs to reach ``isin_threshold`` and the
    ratio snaps with the wider ``SNAP_TOLERANCE_ISIN``.  The ratio is snapped
    to ``CORPORATE_RATIOS`` (``snap_gap``); when nothing matches the raw
    close ratio is kept and flagged unexplained.  Genuine crashes (intraday
    recoveries) and sub-rupee tick noise are left alone.

    Returns ``(factors, snapped)`` arrays.
    """
    out = np.full(close.shape, np.nan)
    snapped = np.zeros(close.shape, dtype=bool)
    isin = np.zeros(close.shape, dtype=bool) if isin_change is None else isin_change
    with np.errstate(divide="ignore", invalid="ignore"):
        obs = close / prior
    thr = np.where(isin, isin_threshold, threshold)
    lo, hi = 1.0 - thr, 1.0 / (1.0 - thr)
    with np.errstate(invalid="ignore"):
        cand = np.isfinite(obs) & ((obs <= lo) | (obs >= hi)) & ~taken & (prior >= min_price)
    for t, s in zip(*np.nonzero(cand)):
        c, o, p = close[t, s], open_[t, s], prior[t, s]
        window = close[t: t + persist_days, s]
        window = window[np.isfinite(window)]
        if abs(float(np.median(window)) / c - 1.0) > persist_tolerance:
            continue
        gap = o / p if np.isfinite(o) and o > 0 else obs[t, s]
        if lo[t, s] < gap < hi[t, s]:
            continue
        if not isin[t, s] and abs(c / o - 1.0) > 0.25:
            continue
        tol = SNAP_TOLERANCE_ISIN if isin[t, s] else SNAP_TOLERANCE
        out[t, s], snapped[t, s] = snap_gap(gap, float(obs[t, s]), tol)
    return out, snapped


def _hint_factors(hint: np.ndarray, taken: np.ndarray, prior: np.ndarray, open_next: np.ndarray,
                  close_next: np.ndarray, cols: Sequence[str], idx: pd.DatetimeIndex,
                  max_shift: int = 2) -> np.ndarray:
    """Ratios for NSE-recorded splits/bonuses/consolidations whose purpose omits the ratio.

    Measured from the first session within ``max_shift`` of the ex-date whose
    gap points the right way, and snapped to ``CORPORATE_RATIOS``.
    """
    out = np.full(hint.shape, np.nan)
    for t, s in zip(*np.nonzero(np.isfinite(hint) & ~taken)):
        for u in range(t, min(t + max_shift + 1, hint.shape[0])):
            p = prior[u, s]
            if not (np.isfinite(p) and p > 0):
                continue
            o, c = open_next[u, s] / p, close_next[u, s] / p
            ref = o if np.isfinite(o) else c
            if not np.isfinite(ref) or (hint[t, s] < 0 and ref > 0.8) or (hint[t, s] > 0 and ref < 1.25):
                continue
            f, ok = snap_gap(o, c, SNAP_TOLERANCE_ISIN)
            logger.info("corporate action without ratio %s %s: measured %.4g -> %s", cols[s], idx[u].date(),
                        c, f"x{f:.4g}" if ok else "unexplained, using raw ratio")
            out[u, s] = f
            break
    return out


def _slip_rejected(candidate: np.ndarray, rejected: np.ndarray, factors: np.ndarray, prior: np.ndarray,
                   close_next: np.ndarray, offsets: Sequence[int] = (1, -1, 2, -2, 3)) -> np.ndarray:
    """Move rejected large ratio events to a nearby session whose gap they explain.

    NSE occasionally lists an ex-date a session away from the price change
    (e.g. an ETF split that trades at the old scale on the listed date).
    """
    out = np.full(candidate.shape, np.nan)
    n = candidate.shape[0]
    for t, s in zip(*np.nonzero(rejected)):
        f = candidate[t, s]
        lf = abs(np.log(f))
        if lf <= LARGE_FACTOR_LOG:
            continue
        for off in offsets:
            u = t + off
            if not (0 < u < n) or np.isfinite(factors[u, s]) or np.isfinite(out[u, s]):
                continue
            p, c = prior[u, s], close_next[u, s]
            if not (np.isfinite(p) and np.isfinite(c) and p > 0):
                continue
            if abs(np.log(c / p)) > 0.5 * lf and abs(np.log(c / (p * f))) < 0.35 * lf:
                out[u, s] = f
                break
    return out


def adjustment_multipliers(close: pd.DataFrame, prev_close: pd.DataFrame, open_: Optional[pd.DataFrame] = None,
                           events: Optional[pd.DataFrame] = None, face_values: Optional[Dict[str, float]] = None,
                           tolerance: float = 0.002, infer_gaps: bool = True,
                           isin_change: Optional[np.ndarray] = None,
                           dividend_amounts: Optional[np.ndarray] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cumulative split/bonus back-adjustment multipliers and the per-date factor frame used.

    Factor sources, first match wins per (date, symbol):

    1. corporate-action ratios (bonus / split / consolidation) from NSE ``Bc``
       files; an event whose listed ex-date contradicts prices is moved to an
       adjacent session whose gap it explains;
    2. NSE-recorded splits/bonuses whose purpose omits the ratio: measured from
       prices and snapped to ``CORPORATE_RATIOS`` (source 6);
    3. ``factors_from_prev_close`` - bhavcopy PREVCLOSE differing from the
       last close (NSE adjusts it for some actions, e.g. older demergers);
    4. rights (theoretical ex-rights price) and demerger / capital-reduction
       gaps (ex-date open / prior close) from ``Bc`` files;
    5. price-inferred factors for large, persistent, unexplained gaps, snapped
       to common split/bonus ratios (source 4) or kept raw when nothing
       matches (source 5, logged as unexplained).  ISIN changes
       (``isin_change``) lower the gap threshold and widen the snap tolerance.

    Every factor must not widen the observed close-to-close gap (guards wrong
    ex-dates and bad prev_close values).  ``dividend_amounts`` (cash per share
    placed on ex-dates) only removes the dividend drop before inferring gaps.
    As in ``adjust_for_factors``, rows strictly before an ex-date are
    multiplied by its factor.
    """
    close64 = close.to_numpy(dtype="float64")
    idx, cols = close.index, list(close.columns)
    prior = close.astype("float64").ffill().shift(1).to_numpy()
    close_next = close.astype("float64").bfill().to_numpy()
    open64 = (open_ if open_ is not None else close).to_numpy(dtype="float64")
    open_next = pd.DataFrame(open64, index=idx).bfill().to_numpy()

    factors = np.full(close64.shape, np.nan)
    sources = np.zeros(close64.shape, dtype="int8")
    rejected_counts: Dict[int, int] = {}

    def take(candidate: np.ndarray, code: int, slack: float, min_log: float, improve: float = 0.0) -> np.ndarray:
        ok = np.isfinite(candidate) & np.isnan(factors) & _plausible(candidate, close_next, prior, slack, min_log,
                                                                     improve)
        rejected = np.isfinite(candidate) & np.isnan(factors) & ~ok
        rejected_counts[code] = rejected_counts.get(code, 0) + int(rejected.sum())
        for t, s in list(zip(*np.nonzero(rejected)))[:10]:
            logger.info("rejected factor source=%d %s %s f=%.4g (inconsistent with prices)",
                        code, cols[s], idx[t].date(), candidate[t, s])
        factors[ok] = candidate[ok]
        sources[ok] = code
        return rejected

    ev = event_factor_frames(events, idx, cols, prior, open_next, face_values)
    rejected = take(ev["ratio"], 1, slack=0.02, min_log=LARGE_FACTOR_LOG)
    if rejected.any():
        slipped = _slip_rejected(ev["ratio"], rejected, factors, prior, close_next)
        for t, s in zip(*np.nonzero(np.isfinite(slipped))):
            logger.info("corporate action %s f=%.4g moved to %s (listed ex-date inconsistent with prices)",
                        cols[s], slipped[t, s], idx[t].date())
        rejected_counts[1] -= int(np.isfinite(slipped).sum())
        take(slipped, 1, slack=0.02, min_log=LARGE_FACTOR_LOG)
    take(_hint_factors(ev["hint"], np.isfinite(factors), prior, open_next, close_next, cols, idx),
         6, slack=0.02, min_log=0.0)

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
        # prior close in the session's price scale: factors already placed since the
        # symbol's last trade (an ex-date on a non-trading session) are applied
        with np.errstate(divide="ignore"):
            logf = np.cumsum(np.log(np.where(np.isfinite(factors), factors, 1.0)), axis=0)
        rows_i = np.arange(len(idx), dtype="float64")[:, None]
        last = pd.DataFrame(np.where(np.isfinite(close64), rows_i, np.nan)).ffill().shift(1).to_numpy()
        last_i = np.where(np.isfinite(last), last, 0).astype(int)
        since = logf - np.take_along_axis(logf, last_i, axis=0)
        infer_prior = prior * np.exp(np.where(np.isfinite(last), since, 0.0))
        del logf, rows_i, last, last_i, since
        if dividend_amounts is not None:
            y, _ = dividend_yields(dividend_amounts, infer_prior, np.where(np.isfinite(open64), open64, close_next))
            infer_prior = infer_prior * (1.0 - np.nan_to_num(y))
        inf, snapped = inferred_factors(close64, open64, infer_prior, np.isfinite(factors), isin_change)
        isin = np.zeros(close64.shape, dtype=bool) if isin_change is None else isin_change
        for t, s in zip(*np.nonzero(np.isfinite(inf))):
            obs = close64[t, s] / prior[t, s]
            if snapped[t, s]:
                logger.info("inferred corporate action %s %s factor=%.4g (observed %.4g%s, snapped; no NSE record)",
                            cols[s], idx[t].date(), inf[t, s], obs, ", ISIN change" if isin[t, s] else "")
            else:
                logger.warning("unexplained price gap %s %s factor=%.4g (no NSE record, no common ratio%s)",
                               cols[s], idx[t].date(), inf[t, s], ", ISIN change" if isin[t, s] else "")
        take(np.where(snapped, inf, np.nan), 4, slack=0.02, min_log=0.0)
        take(np.where(~snapped, inf, np.nan), 5, slack=0.02, min_log=0.0)

    counts = {name: int((sources == code).sum()) for code, name in SOURCE_NAMES.items()}
    logger.info("adjustment factors by source: %s; rejected by source: %s", counts,
                {SOURCE_NAMES[k]: v for k, v in sorted(rejected_counts.items())})
    f = np.where(np.isfinite(factors), factors, 1.0)
    mult = np.cumprod(f[::-1], axis=0)[::-1] / f  # product of factors with ex-date > t
    factor_frame = pd.DataFrame(factors, index=idx, columns=cols).dropna(how="all", axis=1)
    factor_frame.attrs["source_counts"] = counts
    factor_frame.attrs["rejected_counts"] = {SOURCE_NAMES[k]: v for k, v in rejected_counts.items()}
    return pd.DataFrame(mult, index=idx, columns=cols), factor_frame


def dividend_amount_frame(dividends: Optional[pd.DataFrame], dates: pd.DatetimeIndex,
                          symbols: Sequence[str]) -> np.ndarray:
    """Cash dividend per share (dates x symbols), placed on the first session >= ex-date and summed."""
    out = np.zeros((len(dates), len(symbols)))
    if dividends is not None and len(dividends):
        di = dates.searchsorted(pd.DatetimeIndex(dividends["ex_date"]))
        si = pd.Index(symbols).get_indexer(dividends["canonical"])
        amt = dividends["dividend"].to_numpy(dtype="float64")
        ok = (si >= 0) & (di > 0) & (di < len(dates)) & np.isfinite(amt) & (amt > 0)
        np.add.at(out, (di[ok], si[ok]), amt[ok])
    out[out == 0] = np.nan
    return out


def dividend_multipliers(close: pd.DataFrame, amounts: np.ndarray, split_factors: Optional[np.ndarray] = None,
                         max_yield: float = MAX_DIVIDEND_YIELD, open_: Optional[pd.DataFrame] = None,
                         same_day_units: str = "pre") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Total-return back-adjustment for cash dividends.

    On an ex-date t every earlier price is multiplied by ``1 - D / P`` where D
    is the dividend per share and P the last close before t, both in the
    share units of the dividend.  A dividend paid in a session without a
    split is in that session's units, so the ratio is scale-free and applying
    these multipliers on top of split-adjusted prices is consistent.

    A dividend sharing its ex-date with a split/bonus (one record date, e.g.
    "BONUS 1:1/DIV-RS 30", INFY 2014-12-02) is per *pre*-action share: shares
    allotted after the record date do not receive it.  The ex-date opens agree
    (26 of 34 such NSE events, all large caps: INFY, TCS, GICRE, WELSPUNIND).
    ``same_day_units="post"`` instead scales P by ``split_factors``.  ``D / P >=
    max_yield`` is ignored with a warning (parse error or stale price) unless
    the ex-date open (or close) confirms it (``dividend_yields``).

    Returns ``(multipliers, yields)``; volumes must not be divided by these.
    """
    idx, cols = close.index, list(close.columns)
    prior = close.astype("float64").ffill().shift(1).to_numpy()
    if split_factors is not None:
        scale = np.where(np.isfinite(split_factors), split_factors, 1.0)
        px_scale = 1.0 / scale  # ex-date prices -> pre-action units for the confirmation check
        if same_day_units == "post":
            prior, px_scale = prior * scale, np.ones_like(scale)
    else:
        px_scale = 1.0
    px = close.to_numpy(dtype="float64")
    if open_ is not None:
        o = open_.to_numpy(dtype="float64")
        px = np.where(np.isfinite(o) & (o > 0), o, px)
    px = pd.DataFrame(px).bfill().to_numpy() * px_scale
    y, bad = dividend_yields(amounts, prior, px, max_yield)
    for t, s in list(zip(*np.nonzero(bad)))[:20]:
        logger.warning("ignoring dividend %s %s D=%.4g (%.0f%% of prior close %.4g)",
                       cols[s], idx[t].date(), amounts[t, s], 100 * amounts[t, s] / prior[t, s], prior[t, s])
    if bad.sum() > 20:
        logger.warning("... %d implausible dividends ignored in total", int(bad.sum()))
    g = np.where(np.isfinite(y), 1.0 - y, 1.0)
    mult = np.cumprod(g[::-1], axis=0)[::-1] / g
    n_missing = int((np.isfinite(amounts) & ~np.isfinite(y) & ~bad).sum())
    logger.info("dividend adjustments: %d applied across %d symbols, %d implausible ignored, %d without a prior close",
                int(np.isfinite(y).sum()), int(np.isfinite(y).any(axis=0).sum()), int(bad.sum()), n_missing)
    return pd.DataFrame(mult, index=idx, columns=cols), pd.DataFrame(y, index=idx, columns=cols)


def total_return_index(price: pd.Series, dividend_points: pd.Series, reset_drop: float = 0.5) -> pd.Series:
    """Total-return index from a price index and NSE's cumulative dividend-points index.

    NSE's ``Nifty50 Dividend Points`` accumulates index-point dividends over
    the financial year and resets to ~0 at its end.  Daily dividend points are
    the increase of its running maximum (a drop below ``reset_drop`` of the
    running maximum starts a new year; smaller dips are data noise), and
    ``TRI_t = TRI_{t-1} * (P_t + d_t) / P_{t-1}`` with ``TRI_0 = P_0``.
    """
    df = pd.concat({"p": price, "dp": dividend_points}, axis=1).sort_index()
    df = df[df["p"].notna()]
    if df.empty:
        return pd.Series(dtype="float64", index=price.index)
    inc = np.zeros(len(df))
    runmax = np.nan
    for i, v in enumerate(df["dp"].to_numpy(dtype="float64")):
        if not np.isfinite(v):
            continue
        if not np.isfinite(runmax):
            runmax = v  # history before the window is already in the level
        elif runmax > 0 and v < reset_drop * runmax:
            inc[i], runmax = v, v
        elif v > runmax:
            inc[i], runmax = v - runmax, v
    p = df["p"].to_numpy(dtype="float64")
    growth = np.ones(len(df))
    growth[1:] = (p[1:] + inc[1:]) / p[:-1]
    return pd.Series(p[0] * np.cumprod(growth), index=df.index)


def build_index_close(indices: pd.DataFrame, dates: pd.DatetimeIndex,
                      names: Sequence[str] = INDEX_NAMES) -> pd.DataFrame:
    """Index closes on ``dates``; ``*_TRI`` names in ``TRI_SOURCES`` are derived."""
    raw_names = set(names) | {src for n in names if n in TRI_SOURCES for src in TRI_SOURCES[n]}
    sub = indices[indices["index_name"].astype(str).isin(raw_names)]
    wide = sub.pivot_table(index="date", columns="index_name", values="close", aggfunc="last")
    wide.columns = [str(c) for c in wide.columns]
    for name in names:
        if name in TRI_SOURCES and name not in wide:
            price_name, dp_name = TRI_SOURCES[name]
            if price_name in wide and dp_name in wide:
                on_cal = wide.reindex(dates)
                wide[name] = total_return_index(on_cal[price_name], on_cal[dp_name]).reindex(wide.index)
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
    adjust_dividends: bool = True,
) -> MarketData:
    """Build an adjusted, calendar-aligned ``MarketData`` from the parquet store.

    Prices are back-adjusted for splits, bonuses and other corporate actions;
    with ``adjust_dividends`` (default) also for cash dividends, so price
    returns are total returns (use ``adjust_dividends=False`` for price-only
    series, e.g. to compare with the NIFTY50 price index; ``NIFTY50_TRI`` is
    the total-return counterpart).  Volumes are adjusted for share-count
    changes only.
    """
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
    # Dividends are always read: removing the ex-dividend drop before gap
    # inference keeps split factors identical with and without adjust_dividends.
    div_amounts = dividend_amount_frame(load_dividends(store, dates, changes), dates, keep)
    isin_change = isin_change_frame(rows, dates, keep)
    mult, factors = adjustment_multipliers(close, pivot(rows, "prev_close", dates, keep), open_, events,
                                           _face_values(store), isin_change=isin_change, dividend_amounts=div_amounts)
    del isin_change
    n_adj = int(factors.notna().to_numpy().sum())
    logger.info("applied %d corporate-action adjustments across %d symbols", n_adj, factors.shape[1])
    price_mult = mult
    if adjust_dividends:
        split_f = factors.reindex(columns=keep).to_numpy(dtype="float64")
        div_mult, _ = dividend_multipliers(close, div_amounts, split_f, open_=open_)
        price_mult = mult * div_mult
        del div_mult, split_f
    del div_amounts
    close_unadj = close.astype(float_dtype)          # as printed, for point-in-time price rules
    frames = {"close": close * price_mult, "open": open_ * price_mult}
    for col in ("high", "low"):
        frames[col] = pivot(rows, col, dates, keep) * price_mult
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
        close_unadj=close_unadj,
        etfs=etf_symbols(store, rows, keep),
        sectors=sectors,
        source=f"nse_bhavcopy:{store}" + ("" if adjust_dividends else ":price_only"),
    )
    data.data_hash = data.compute_hash()
    data.validate()
    return data
