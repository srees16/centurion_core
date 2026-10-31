"""
Implied-volatility history (tracker OD1): a 30-day at-the-money implied
volatility per session from the NSE F&O store, its rank and percentile over
the past year, the IV level the strategy selector reads, and how realised
volatility then compared with it.

The daily IV: each session, the two expiries around 30 calendar days (at
least ``iv_min_dte`` away, so expiry-week noise stays out) each give the mean
of their at-the-money call and put IVs, solved from closing prices with the
toolkit's Black-Scholes on spot and the risk-free rate (as
:mod:`live_chain` does); the two are interpolated in total variance to
exactly 30 days.  Averaging the call and the put at one strike cancels most
of the error in the rate and dividend assumed, so one rate serves all of
history.  Only contracts that traded that day give an IV.  ``live_iv30``
builds the same number from live quotes, so a live value compares with its
own history.

The IV level (``low`` / ``normal`` / ``high`` / ``very_high``) is the rule
the selector documents (M5 ch. 20, M6 ch. 4): today's IV against the cone of
21-session realised volatility over two years - below the mean - 1 SD is
low, above the mean + 1 SD high, above ``high_iv_multiple`` x the mean very
high.  Realised volatility is annualised over trading days (sqrt 252):
implied volatility accrues only while the market trades, so sqrt 365 (the
PDF's convention in its worked examples) would overstate realised volatility
by about 20% and make every IV look cheap.

The premium record asks history whether option premium is rich at today's
IV percentile: on past sessions in the same percentile bucket, how often the
next 30 days' realised volatility came in below the IV, and by how much.
The windows overlap, so it describes the past; it is not a test.

Histories are cached per symbol in ``<fo_store>/iv/``; each call recomputes
the cache's last year (late files, corrections) and adds anything newer.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from kite_connect.options.options_config import OptionsConfig
from kite_connect.options.theory import CALL, PUT, implied_volatility, volatility_cone
from nse_engine.data import fo_store

logger = logging.getLogger(__name__)

IV_VERSION = 1                       # bump when the construction changes: the caches are rebuilt
TRADING_DAYS = 252
LEVELS = ("low", "normal", "high", "very_high")
HISTORY_COLUMNS = ["date", "iv30", "near_iv", "near_dte", "far_iv", "far_dte", "spot"]


# ── one session ──────────────────────────────────────────────────────────

def expiry_atm_iv(rows: pd.DataFrame, spot: float, dte: float, rate: float, tries: int = 3) -> Optional[float]:
    """Mean IV of the call and put at the strike nearest ``spot`` with a usable traded price.

    ``rows``: one expiry's traded contracts (strike, option_type, close).
    Up to ``tries`` strikes outward when the nearest gives no IV.
    """
    strikes = sorted(rows["strike"].unique(), key=lambda k: abs(k - spot))[:tries]
    for k in strikes:
        at = rows[rows["strike"] == k]
        ivs = []
        for option_type in (CALL, PUT):
            px = at.loc[at["option_type"] == option_type, "close"]
            if len(px):
                iv = implied_volatility(option_type, float(px.iloc[0]), spot, float(k), dte, rate)
                if iv:
                    ivs.append(iv)
        if ivs:
            return float(np.mean(ivs))
    return None


def constant_maturity(points: Sequence[Tuple[float, Optional[float]]], tenor: float) -> Optional[float]:
    """(days, IV) of the expiries around ``tenor`` -> the IV at ``tenor`` days.

    Interpolated in total variance (IV^2 x days); with points on one side only,
    the nearest point's IV.
    """
    pts = sorted((t, v) for t, v in points if v)
    below = [p for p in pts if p[0] <= tenor]
    above = [p for p in pts if p[0] > tenor]
    if below and above:
        (t1, v1), (t2, v2) = below[-1], above[0]
        w = v1 * v1 * t1 + (v2 * v2 * t2 - v1 * v1 * t1) * (tenor - t1) / (t2 - t1)
        return math.sqrt(w / tenor) if w > 0 else None
    if below or above:
        return (below[-1] if below else above[0])[1]
    return None


def session_iv(day: pd.DataFrame, session: pd.Timestamp, spot: float, rate: float,
               tenor: int, min_dte: int) -> Optional[dict]:
    """The 30-day IV of one session from its traded contracts (all expiries)."""
    dtes = {e: (pd.Timestamp(e) - session).days for e in day["expiry"].unique()}
    near = sorted((e for e, d in dtes.items() if min_dte <= d <= tenor), key=lambda e: -dtes[e])
    far = sorted((e for e, d in dtes.items() if d > tenor), key=lambda e: dtes[e])

    def first_iv(expiries) -> Tuple[Optional[float], Optional[int]]:
        for e in expiries:
            iv = expiry_atm_iv(day[day["expiry"] == e], spot, dtes[e], rate)
            if iv:
                return iv, dtes[e]
        return None, None

    near_iv, near_dte = first_iv(near)
    far_iv, far_dte = first_iv(far)
    iv30 = constant_maturity([(near_dte or 0, near_iv), (far_dte or 0, far_iv)], tenor)
    if iv30 is None:
        return None
    return {"date": session, "iv30": iv30, "near_iv": near_iv, "near_dte": near_dte, "far_iv": far_iv,
            "far_dte": far_dte, "spot": spot}


# ── history ──────────────────────────────────────────────────────────────

def _is_index(symbol: str) -> bool:
    return symbol.upper() in fo_store.INDEX_NAMES


def underlying_prices(symbol: str, cfg: OptionsConfig, start: str = "2000-01-01") -> Tuple[pd.Series, pd.Series]:
    """(printed closes, the spot an option was priced on; closes back-adjusted for splits and
    bonuses, for realised volatility).  An index's level serves as both."""
    symbol = symbol.upper()
    if _is_index(symbol):
        level = fo_store.load_underlying(cfg.data.fo_store, symbol)
        return level, level
    from nse_engine.data.panel import load_market_data

    data = load_market_data(cfg.data.equity_store, start, date.today(), symbols=[symbol],
                            min_median_value_inr=0.0, adjust_dividends=False)
    if symbol not in data.close:
        raise ValueError(f"no closes for {symbol} in {cfg.data.equity_store}")
    printed = data.close_unadj if data.close_unadj is not None else data.close
    return (printed[symbol].dropna().astype(float).rename(symbol),
            data.close[symbol].dropna().astype(float).rename(symbol))


def _option_years(symbol: str, cfg: OptionsConfig) -> List[int]:
    table = "options" if _is_index(symbol) else "stock_options"
    return sorted(int(p.stem) for p in (Path(cfg.data.fo_store) / table).glob("[0-9]" * 4 + ".parquet"))


def compute_history(symbol: str, cfg: OptionsConfig, from_year: int, spot: pd.Series) -> pd.DataFrame:
    """The 30-day IV of every session from ``from_year`` (no cache)."""
    symbol = symbol.upper()
    load = fo_store.load_options if _is_index(symbol) else fo_store.load_stock_options
    d, rate = cfg.data, cfg.market.risk_free_rate
    rows: List[dict] = []
    for year in (y for y in _option_years(symbol, cfg) if y >= from_year):
        opts = load(d.fo_store, f"{year}-01-01", f"{year}-12-31", symbol)
        if opts.empty:
            continue
        opts = opts[(opts["contracts"] > 0) & (opts["close"] > 0)]
        opts = opts.assign(spot=spot.reindex(pd.DatetimeIndex(opts["date"])).to_numpy()).dropna(subset=["spot"])
        opts = opts[(opts["strike"] / opts["spot"] - 1).abs() <= d.iv_strike_band]
        for session, day in opts.groupby("date", sort=True):
            row = session_iv(day, pd.Timestamp(session), float(day["spot"].iloc[0]), rate, d.iv_tenor_days,
                             d.iv_min_dte)
            if row:
                rows.append(row)
        logger.info("IV history %s %d: %d sessions", symbol, year, sum(r["date"].year == year for r in rows))
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


def iv_history(symbol: str, cfg: OptionsConfig, spot: Optional[pd.Series] = None) -> pd.DataFrame:
    """The cached 30-day IV history of ``symbol``, brought up to the store's last session."""
    symbol = symbol.upper()
    path = Path(cfg.data.fo_store) / "iv" / f"{symbol}_v{IV_VERSION}.parquet"
    cached = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=HISTORY_COLUMNS)
    years = _option_years(symbol, cfg)
    if not years:
        raise FileNotFoundError(f"no option history for {symbol} in {cfg.data.fo_store} (build the store, OD3)")
    from_year = int(pd.Timestamp(cached["date"].max()).year) if len(cached) else years[0]
    spot = spot if spot is not None else underlying_prices(symbol, cfg, start=f"{years[0]}-01-01")[0]
    fresh = compute_history(symbol, cfg, from_year, spot)
    keep = cached[pd.to_datetime(cached["date"]).dt.year < from_year] if len(cached) else cached
    out = pd.concat([keep, fresh], ignore_index=True).sort_values("date").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


# ── reading it ───────────────────────────────────────────────────────────

def rank_and_percentile(history: pd.Series, value: float, lookback: int) -> Tuple[float, float]:
    """IV rank ((value - min) / (max - min)) and percentile (share of sessions below, ties
    counting half) over the last ``lookback`` values of ``history``, both 0-100."""
    h = history.dropna().to_numpy(dtype=float)[-lookback:]
    if not len(h):
        return math.nan, math.nan
    lo, hi = h.min(), h.max()
    rank = 50.0 if hi == lo else min(max((value - lo) / (hi - lo), 0.0), 1.0) * 100
    return float(rank), mid_rank(h, value)


def mid_rank(past: np.ndarray, value: float) -> float:
    """Percentile of ``value`` among ``past``, 0-100, ties counting half."""
    return float(((past < value).mean() + 0.5 * (past == value).mean()) * 100)


def iv_level(iv: float, prices: pd.Series, cfg: OptionsConfig) -> Tuple[str, Optional[Dict[str, float]]]:
    """The selector's IV level (M5 ch. 20, M6 ch. 4) and the realised-volatility cone row it read."""
    d = cfg.data
    cone = volatility_cone(prices.dropna().to_numpy(dtype=float)[-(d.cone_lookback_sessions + 1):],
                           windows=(d.cone_window,), days_per_year=TRADING_DAYS)
    row = cone.get(d.cone_window)
    if row is None:
        return "normal", None
    if iv > cfg.selector.high_iv_multiple * row["mean"]:
        return "very_high", row
    if iv > row["plus1"]:
        return "high", row
    if iv < row["minus1"]:
        return "low", row
    return "normal", row


def forward_realised(prices: pd.Series, tenor_days: int) -> pd.Series:
    """Per session, the realised volatility of the next ``tenor_days`` calendar days
    (sqrt of the mean squared daily log return x 252); NaN where the window is incomplete."""
    p = prices.dropna().astype(float)
    r = np.log(p / p.shift(1)).iloc[1:]
    dates = r.index.to_numpy()
    sq = np.concatenate([[0.0], np.cumsum(r.to_numpy() ** 2)])
    starts = p.index.to_numpy()
    lo = np.searchsorted(dates, starts, side="right")
    hi = np.searchsorted(dates, starts + np.timedelta64(tenor_days, "D"), side="right")
    n = hi - lo
    complete = starts + np.timedelta64(tenor_days, "D") <= dates[-1] if len(dates) else np.zeros(len(starts), bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        rv = np.sqrt((sq[hi] - sq[lo]) / n * TRADING_DAYS)
    return pd.Series(np.where(complete & (n > 0), rv, np.nan), index=p.index)


def trailing_percentile(values: pd.Series, lookback: int) -> pd.Series:
    """Each session's value as a percentile of the ``lookback`` sessions before it (no look-ahead)."""
    v = values.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    for i in range(lookback, len(v)):
        out[i] = mid_rank(v[i - lookback:i], v[i])
    return pd.Series(out, index=values.index)


def premium_record(history: pd.DataFrame, prices: pd.Series, percentile: float,
                   cfg: OptionsConfig) -> Optional[Dict[str, object]]:
    """On past sessions whose IV percentile fell in today's bucket: how often the next
    ``iv_tenor_days`` of realised volatility came in below the IV, and by how much."""
    d = cfg.data
    if not np.isfinite(percentile) or history.empty:
        return None
    iv = history.set_index(pd.DatetimeIndex(history["date"]))["iv30"].astype(float)
    pct = trailing_percentile(iv, d.lookback_sessions)
    rv = forward_realised(prices, d.iv_tenor_days).reindex(iv.index)
    width = d.premium_bucket_width
    lo = min(math.floor(percentile / width) * width, 100 - width)
    sel = (pct >= lo) & (pct < lo + width if lo + width < 100 else pct <= 100) & rv.notna()
    if not sel.any():
        return None
    gap = (iv[sel] - rv[sel]) * 100
    return {"bucket": (lo, lo + width), "sessions": int(sel.sum()), "since": iv.index[sel][0].date(),
            "below_share": float((rv[sel] < iv[sel]).mean() * 100), "median_gap": float(gap.median())}


@dataclass
class IVContext:
    """What history says about today's implied volatility, for the trader and the selector."""

    symbol: str
    iv30: float
    source: str                      # "live" (Kite quotes now) or "eod <date>" (the store's last session)
    rank: float
    percentile: float
    level: str
    cone: Optional[Dict[str, float]]
    premium: Optional[Dict[str, object]]
    history_from: date

    def lines(self) -> List[str]:
        out = [f"IV {self.symbol} 30-day {self.iv30:.1%} ({self.source}): 1-year rank {self.rank:.0f}, "
               f"percentile {self.percentile:.0f}; level {self.level.replace('_', ' ')}"
               + (f" (21-session realised cone: mean {self.cone['mean']:.1%}, "
                  f"+/-1 SD {self.cone['minus1']:.1%} to {self.cone['plus1']:.1%})" if self.cone else "")]
        p = self.premium
        if p:
            lo, hi = p["bucket"]
            out.append(f"At IV percentile {lo:.0f}-{hi:.0f} ({p['sessions']:,} sessions since {p['since']}): the next "
                       f"30 days' realised volatility came in below IV {p['below_share']:.0f}% of the time, by a "
                       f"median {p['median_gap']:+.1f} vol points (overlapping windows: a description, not a test)")
        return out


def iv_context(symbol: str, cfg: OptionsConfig, live_iv: Optional[float] = None) -> IVContext:
    """Today's IV (``live_iv``, else the store's last session) read against its history."""
    symbol = symbol.upper()
    first = _option_years(symbol, cfg)
    spot, prices = underlying_prices(symbol, cfg, start=f"{first[0] if first else 2000}-01-01")
    history = iv_history(symbol, cfg, spot=spot)
    if history.empty:
        raise ValueError(f"no IV history for {symbol}")
    last = history.iloc[-1]
    value, source = ((live_iv, "live") if live_iv else (float(last["iv30"]), f"eod {pd.Timestamp(last['date']).date()}"))
    past = history["iv30"] if live_iv else history["iv30"].iloc[:-1]
    rank, percentile = rank_and_percentile(past, value, cfg.data.lookback_sessions)
    level, cone = iv_level(value, prices, cfg)
    return IVContext(symbol, value, source, rank, percentile, level, cone,
                     premium_record(history, prices, percentile, cfg), pd.Timestamp(history["date"].iloc[0]).date())


def live_iv30(broker, resolver, underlying: str, now: datetime, cfg: OptionsConfig) -> Optional[float]:
    """The 30-day ATM IV from live quotes, built as the history is (two expiries, total variance)."""
    from kite_connect.options.live_chain import chain_summary, days_to_expiry, fetch_chain

    d = cfg.data
    dtes = {e: days_to_expiry(e, now) for e in resolver.expiries(underlying)}
    near = [e for e in sorted(dtes, key=lambda e: -dtes[e]) if d.iv_min_dte <= dtes[e] <= d.iv_tenor_days][:1]
    far = [e for e in sorted(dtes, key=lambda e: dtes[e]) if dtes[e] > d.iv_tenor_days][:1]
    points = []
    for e in near + far:
        chain, spot = fetch_chain(broker, resolver, underlying, e, now, cfg.market.risk_free_rate, 1)
        atm_iv = chain_summary(chain, spot)["atm_iv"]
        points.append((dtes[e], atm_iv if np.isfinite(atm_iv) else None))
    return constant_maturity(points, d.iv_tenor_days)
