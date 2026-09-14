"""
OHLCV validation and corporate-action adjustment for NSE daily data.

Two entry points:

* ``clean_ohlcv`` — for yfinance-style frames (Open/High/Low/Close/Volume)
  that may contain duplicate rows, zero-volume holiday placeholders, bad
  ticks and unadjusted splits/bonuses/demergers (e.g. GOLDBEES 1:100 on
  2019-12-19, TMPV demerger 2025-10-14).
* ``factors_from_prev_close`` — the ratio of bhavcopy's previous close to the
  prior session's close.  NSE does not reliably adjust PREVCLOSE for splits
  or bonuses (HCLTECH 2019-12-05, GOLDBEES 2019-12-19 are unadjusted), so the
  data layer uses it only as a secondary source after NSE's corporate-actions
  file (see ``nse_engine.data.panel.adjustment_multipliers``).

NSE applies daily price bands (circuits), so a >= 35% gap between one
session's close and the next session's open that persists is a corporate
action, not a market move.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_COMMON_FACTORS = sorted(
    {1.0 / k for k in (2, 3, 4, 5, 8, 10, 20, 25, 50, 100)}
    | {b / (a + b) for a in range(1, 11) for b in range(1, 11) if b / (a + b) <= 0.8}
    | {float(k) for k in (2, 3, 4, 5, 10)}  # consolidations
)


@dataclass
class ValidationReport:
    symbol: str = ""
    rows_in: int = 0
    rows_out: int = 0
    duplicates: int = 0
    phantom_rows: int = 0
    invalid_rows: int = 0
    bad_ticks: List[pd.Timestamp] = field(default_factory=list)
    adjustments: List[Tuple[pd.Timestamp, float, bool]] = field(default_factory=list)  # (ex_date, factor, snapped)
    large_moves: List[Tuple[pd.Timestamp, float]] = field(default_factory=list)  # unadjusted >= threshold

    @property
    def unexplained_adjustments(self) -> List[Tuple[pd.Timestamp, float]]:
        return [(d, f) for d, f, snapped in self.adjustments if not snapped]

    def summary(self) -> str:
        parts = [f"{self.symbol}: {self.rows_in}->{self.rows_out} rows"]
        if self.duplicates:
            parts.append(f"{self.duplicates} dup")
        if self.phantom_rows:
            parts.append(f"{self.phantom_rows} phantom")
        if self.invalid_rows:
            parts.append(f"{self.invalid_rows} invalid")
        if self.bad_ticks:
            parts.append(f"{len(self.bad_ticks)} bad ticks")
        for d, f, snapped in self.adjustments:
            parts.append(f"adj {d.date()} x{f:.4g}{'' if snapped else ' (unexplained)'}")
        for d, r in self.large_moves:
            parts.append(f"large move {d.date()} {r - 1:+.0%} (not adjusted)")
        return ", ".join(parts)


def snap_factor(ratio: float, tolerance: float = 0.04) -> Tuple[float, bool]:
    """Snap an observed price ratio to the nearest common split/bonus factor."""
    best = min(_COMMON_FACTORS, key=lambda f: abs(ratio / f - 1.0))
    if abs(ratio / best - 1.0) <= tolerance:
        return best, True
    return ratio, False


def adjust_for_factors(df: pd.DataFrame, factors: pd.Series) -> pd.DataFrame:
    """Back-adjust prices (and inversely volumes) for corporate actions.

    ``factors`` is indexed by ex-date; a factor f means the new price scale is
    f times the old one (1:100 split -> 0.01).  Rows strictly before the
    ex-date are multiplied by f.
    """
    if factors is None or len(factors) == 0:
        return df
    out = df.copy()
    cum = pd.Series(1.0, index=out.index)
    for ex_date, f in factors.sort_index().items():
        cum[cum.index < ex_date] *= float(f)
    for col in ("Open", "High", "Low", "Close", "Adj Close"):
        if col in out.columns:
            out[col] = out[col] * cum
    if "Volume" in out.columns:
        out["Volume"] = out["Volume"] / cum
    return out


def factors_from_prev_close(
    close: pd.Series, prev_close: pd.Series, tolerance: float = 0.002
) -> pd.Series:
    """Adjustment factors from bhavcopy's (ex-date adjusted) previous close.

    factor_t = prev_close_t / close_{t-1}.  Only dates where the factor
    differs from 1 by more than ``tolerance`` are returned.
    """
    prior = close.shift(1)
    ratio = (prev_close / prior).replace([np.inf, -np.inf], np.nan)
    mask = ratio.notna() & ((ratio - 1.0).abs() > tolerance)
    return ratio[mask].astype(float)


def clean_ohlcv(
    df: pd.DataFrame,
    symbol: str = "",
    jump_threshold: float = 0.35,
    persist_days: int = 5,
    persist_tolerance: float = 0.20,
) -> Tuple[pd.DataFrame, ValidationReport]:
    """Validate and repair a single-symbol daily OHLCV frame.

    Returns the cleaned frame and a report.  The input is not modified.
    """
    report = ValidationReport(symbol=symbol, rows_in=len(df))
    if df is None or len(df) == 0:
        return df, report

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out.index = pd.DatetimeIndex(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out = out.sort_index()

    dup = out.index.duplicated(keep="last")
    report.duplicates = int(dup.sum())
    out = out[~dup]

    price_cols = [c for c in ("Open", "High", "Low", "Close") if c in out.columns]
    invalid = out["Close"].isna() | (out[price_cols] <= 0).any(axis=1)
    report.invalid_rows = int(invalid.sum())
    out = out[~invalid]

    if "Volume" in out.columns and {"Open", "High", "Low"}.issubset(out.columns):
        flat = (out["Open"] == out["High"]) & (out["High"] == out["Low"]) & (out["Low"] == out["Close"])
        phantom = (out["Volume"].fillna(0) == 0) & flat & (out["Close"] == out["Close"].shift(1))
        report.phantom_rows = int(phantom.sum())
        out = out[~phantom]

    if {"Open", "High", "Low"}.issubset(out.columns):
        out["Open"] = out["Open"].fillna(out["Close"])
        out["High"] = out[["Open", "High", "Close"]].max(axis=1)
        out["Low"] = out[["Open", "Low", "Close"]].min(axis=1)

    out, bad_ticks = _drop_bad_ticks(out, jump_threshold)
    report.bad_ticks = bad_ticks

    factors = _detect_discontinuities(out, jump_threshold, persist_days, persist_tolerance, report)
    if len(factors):
        out = adjust_for_factors(out, factors)

    report.rows_out = len(out)
    if report.adjustments or report.bad_ticks or report.large_moves or report.duplicates or report.invalid_rows:
        logger.info("OHLCV validation %s", report.summary())
    return out, report


_STRICT_SPLIT_FACTORS = [1.0 / k for k in (2, 3, 4, 5, 10, 20, 50, 100)]


def _drop_bad_ticks(
    df: pd.DataFrame, threshold: float, max_segment: int = 5
) -> Tuple[pd.DataFrame, List[pd.Timestamp]]:
    """Drop short wrong-scale segments that jump away and fully come back.

    Catches one-day spikes and multi-day scale errors such as GOLDBEES trading
    at 1/100 of its price on 2019-12-19..20 in yfinance.
    """
    close = df["Close"].to_numpy(dtype=float)
    bad_pos: List[int] = []
    t = 1
    n = len(close)
    while t < n:
        r = close[t] / close[t - 1] if close[t - 1] > 0 else 1.0
        if (1.0 - threshold) < r < 1.0 / (1.0 - threshold):
            t += 1
            continue
        for u in range(t + 1, min(t + max_segment + 1, n)):
            back = close[u] / close[u - 1] if close[u - 1] > 0 else 1.0
            if abs(back * r - 1.0) < 0.15 and abs(close[u] / close[t - 1] - 1.0) < threshold / 2:
                bad_pos.extend(range(t, u))
                t = u
                break
        else:
            t += 1
            continue
        t += 1
    bad = list(df.index[bad_pos])
    return df.drop(index=bad), bad


def _detect_discontinuities(
    df: pd.DataFrame,
    threshold: float,
    persist_days: int,
    persist_tolerance: float,
    report: ValidationReport,
) -> pd.Series:
    """Find persistent corporate-action gaps.

    Adjusts only when (a) the session opened >= threshold away from the prior
    close and closed near that open (NSE's ex-date price discovery), or
    (b) the close-to-close ratio is within 2% of an exact split ratio.
    Other large moves are real (F&O stocks have no circuits: CANBK +44% on
    2017-10-25, YESBANK -56% on 2020-03-06) and are only reported.
    """
    close = df["Close"].to_numpy(dtype=float)
    opens = df["Open"].to_numpy(dtype=float) if "Open" in df.columns else close
    idx = df.index
    found = {}
    for t in range(1, len(df)):
        ref = close[t - 1]
        if not np.isfinite(ref) or ref <= 0 or not np.isfinite(close[t]) or close[t] <= 0:
            continue
        observed = close[t] / ref
        if (1.0 - threshold) < observed < 1.0 / (1.0 - threshold):
            continue
        window = close[t: t + persist_days]
        window = window[np.isfinite(window)]
        if abs(float(np.median(window)) / close[t] - 1.0) > persist_tolerance:
            continue
        gap = opens[t] / ref if np.isfinite(opens[t]) and opens[t] > 0 else observed
        opened_at_new_level = not ((1.0 - threshold) < gap < 1.0 / (1.0 - threshold)) and abs(close[t] / opens[t] - 1.0) <= 0.25
        strict = min(_STRICT_SPLIT_FACTORS, key=lambda f: abs(observed / f - 1.0))
        if abs(observed / strict - 1.0) <= 0.02:
            factor, snapped = strict, True
        elif opened_at_new_level:
            factor, snapped = snap_factor(observed)
        else:
            report.large_moves.append((idx[t], float(observed)))
            continue
        found[idx[t]] = factor
        report.adjustments.append((idx[t], float(factor), snapped))
    return pd.Series(found, dtype=float)


def assert_no_unadjusted_gaps(df: pd.DataFrame, threshold: float = 0.35) -> Optional[pd.Timestamp]:
    """Return the first date with a >= threshold close-to-close move, else None."""
    r = df["Close"].pct_change().abs()
    hit = r[r >= threshold]
    return None if hit.empty else hit.index[0]
