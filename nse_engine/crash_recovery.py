"""
Crash-recovery capital advisory (tracker CR1 / CR1b, plan sections 5ab / 5ac).

When a market index falls deep below its peak, fresh money from a reserve
kept outside the book may be added to an index ETF in stages.  The trigger
is the index's fall, never Centurion's own drawdown.  This module tests the
pre-registered configurations on long index histories and reports where
NIFTY stands today.  It is advisory only: the owner places any trade.

* **Episode:** the index closes at least ``START_DD`` (20%) below its
  running peak.  It ends when the index regains that peak.
* **Tranches** fill at the next session's close (the owner acts on an
  email), each at most once per episode:
  - ``depth``: the close is ``level`` or more below the peak;
  - ``confirm``: the first close above the ``CONFIRM_MA``-session average
    after the index closed below it within the episode;
  - ``bounce``: a close ``level`` above the lowest close since the episode
    started;
  - ``high``: the close is the highest of the last ``HIGH_WINDOW`` sessions;
  - ``cross``: the ``CROSS_FAST``-session average closes above the
    ``CONFIRM_MA``-session average after being at or below it within the
    episode.
  ``confirm`` and ``cross`` invest everything still in cash.
* **Fallback:** anything still in cash goes in when the index regains its
  prior peak.
* **Outcome:** the reserve's value ``HORIZON_YEARS`` after the episode
  starts; cash earns the cash rate.  CR1 values the index part on the price
  index; CR1b adds an assumed dividend yield (``DIVIDEND_YIELD``).

Tested 11 Oct 2026 (plan 5ab): FAIL, so no advisory is built.  Over 89 episodes in 13 indices, A, B and C
beat 4% cash in 66%, 66% and 65% of them (the rule is 70%).  Each median trails a lump sum at -20%.  The
worst episode is the S&P 500 in 1929 for every arm.

Tested 11 Oct 2026 (plan 5ac, CR1b, with dividends): FAIL, and CR1 closes for good.  D and E beat cash in
72% and 74% of episodes, and their worst episode (0.54 / 0.56) beat a lump sum at -20% (0.37).  But only 61%
of the episodes before mid-2000 beat cash (rule 3).  F failed every rule.

CLI::

    python -m nse_engine.crash_recovery report     # CR1, the pre-registered test (2026 sealed)
    python -m nse_engine.crash_recovery report-b   # CR1b (plan 5ac)
    python -m nse_engine.crash_recovery status     # NIFTY today: the live fall and the levels it reached
    python -m nse_engine.crash_recovery alert      # email a level first reached lately (daily job; --dry-run prints)

No buy rule passed its test, but the owner's own plan (13 Oct 2026) stages fresh 5-year capital in thirds: when
NIFTY first closes 20%, 30% or 40% below its peak in a fall, the email asks for one third of the crash reserve
in a NIFTY 50 ETF.  Its 50-day average closing back above the 200-day is a facts-only note.  Each level is
emailed once per fall.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

START_DD = 0.20
CONFIRM_MA = 200
HORIZON_YEARS = 3
CASH_RATE = 0.04                 # (A) a round global rate, used for every market in the test
INDIA_CASH_RATE = 0.065          # India is also reported at this rate (never gating)
INDIA_MARKETS = ("NIFTY50", "^BSESN")
TEST_END = "2025-12-31"          # 2026 is the sealed holdout
GLOBAL_TICKERS = ("^GSPC", "^N225", "^FTSE", "^GDAXI", "^HSI", "^GSPTSE", "^AXJO", "^KS11", "^TWII",
                  "^STI", "^BSESN", "^MXX")
CACHE_DIR = Path("data/nse_engine/crash_recovery")
# CR1b (plan 5ac)
BOUNCE = 0.10                    # a close this far above the episode's lowest close
HIGH_WINDOW = 60                 # the close is the highest of the last HIGH_WINDOW sessions
CROSS_FAST = 50                  # the CROSS_FAST-session average crosses above the CONFIRM_MA one
DIVIDEND_YIELD = {"NIFTY50": 0.013, "^BSESN": 0.013, "^N225": 0.010, "^GDAXI": 0.0}   # DAX already has dividends
DEFAULT_YIELD = 0.02             # every other market
CLUSTERS = {"2000-03": ("2000-01-01", "2003-03-31"), "2008": ("2007-07-01", "2009-03-31")}   # reported without
TAKES_REST = ("confirm", "cross")


@dataclass(frozen=True)
class Tranche:
    kind: str                    # "depth", "confirm", "bounce", "high" or "cross"
    level: float = 0.0           # depth below the peak (depth) or rise above the episode low (bounce)
    share: float = 0.0           # share of the original reserve; TAKES_REST kinds take whatever is left


@dataclass(frozen=True)
class Config:
    name: str
    tranches: Tuple[Tranche, ...]


CONFIGS: Dict[str, Config] = {
    "CR1-A": Config("CR1-A", (Tranche("depth", 0.30, 1 / 3), Tranche("depth", 0.40, 1 / 3), Tranche("confirm"))),
    "CR1-B": Config("CR1-B", (Tranche("confirm"),)),
    "CR1-C": Config("CR1-C", (Tranche("depth", 0.30, 0.5), Tranche("depth", 0.40, 0.5))),
}
CONFIGS_B: Dict[str, Config] = {
    "CR1b-D": Config("CR1b-D", (Tranche("bounce", BOUNCE, 0.3), Tranche("high", share=0.4), Tranche("cross"))),
    "CR1b-E": Config("CR1b-E", (Tranche("depth", 0.30, 0.3), Tranche("depth", 0.40, 0.4), Tranche("cross"))),
    "CR1b-F": Config("CR1b-F", (Tranche("high", share=1.0),)),
}
LUMP20 = Config("L20", (Tranche("depth", START_DD, 1.0),))


@dataclass(frozen=True)
class Episode:
    peak_date: pd.Timestamp
    peak: float
    start: pd.Timestamp                    # first close START_DD below the peak
    bottom: pd.Timestamp
    depth: float
    end: Optional[pd.Timestamp]            # first close back at the peak, None if not yet


# ------------------------------------------------------------------ episodes and signals
def episodes(close: pd.Series, start_dd: float = START_DD) -> List[Episode]:
    """Every fall of at least ``start_dd`` from a running peak, up to the close that regains that peak."""
    c = close.dropna()
    out: List[Episode] = []
    peak_date, peak, start = c.index[0], float(c.iloc[0]), None
    for d, v in c.items():
        if start is None:
            if v >= peak:
                peak_date, peak = d, float(v)
            elif v / peak - 1 <= -start_dd:
                start = d
        elif v >= peak:
            seg = c[start:d]
            out.append(Episode(peak_date, peak, start, seg.idxmin(), float(seg.min() / peak - 1), d))
            peak_date, peak, start = d, float(v), None
    if start is not None:
        seg = c[start:]
        out.append(Episode(peak_date, peak, start, seg.idxmin(), float(seg.min() / peak - 1), None))
    return out


def signal_dates(close: pd.Series, ep: Episode, tranches: Sequence[Tranche],
                 ma: Optional[pd.Series] = None) -> List[Optional[pd.Timestamp]]:
    """The first close in the episode at which each tranche's signal fires; None where it has not."""
    c = close.dropna()
    ma = c.rolling(CONFIRM_MA, min_periods=CONFIRM_MA).mean() if ma is None else ma
    kinds = {t.kind for t in tranches}
    fast = c.rolling(CROSS_FAST, min_periods=CROSS_FAST).mean() if "cross" in kinds else None
    high = c.rolling(HIGH_WINDOW, min_periods=HIGH_WINDOW).max() if "high" in kinds else None
    window = c[ep.peak_date:ep.end] if ep.end is not None else c[ep.peak_date:]
    dates: List[Optional[pd.Timestamp]] = [None] * len(tranches)
    below, under, low = False, False, np.inf

    def fires(t: Tranche, d: pd.Timestamp, v: float, m: float) -> bool:
        # NaN averages (too little history) compare False, so they never fire.
        if t.kind == "depth":
            return v / ep.peak - 1 <= -t.level
        if t.kind == "confirm":
            return below and v > m
        if t.kind == "bounce":
            return v >= low * (1 + t.level)
        if t.kind == "high":
            return bool(v >= high[d])
        if t.kind == "cross":
            return under and bool(fast[d] > m)
        raise ValueError(f"unknown tranche kind {t.kind!r}")

    for d, v in window[window.index >= ep.start].items():
        if all(x is not None for x in dates):
            break
        m = ma.get(d, np.nan)
        below = below or bool(v < m)
        under = under or (fast is not None and bool(fast[d] <= m))
        low = min(low, v)
        for i, t in enumerate(tranches):
            if dates[i] is None and fires(t, d, v, m):
                dates[i] = d
    return dates


def fills(close: pd.Series, ep: Episode, cfg: Config, ma: Optional[pd.Series] = None) -> List[Tuple[pd.Timestamp, float]]:
    """(fill date, share of the original reserve) for each tranche, using only closes up to each signal."""
    c = close.dropna()
    sessions = c.index
    dates = signal_dates(c, ep, cfg.tranches, ma)
    invested, out = 0.0, []

    def fill_after(d: pd.Timestamp) -> Optional[pd.Timestamp]:
        i = sessions.searchsorted(d, side="right")
        return sessions[i] if i < len(sessions) else None

    # In signal order; tranches signalled on the same close keep the configuration's order.
    for i in sorted((i for i, d in enumerate(dates) if d is not None), key=lambda i: dates[i]):
        t = cfg.tranches[i]
        share = (1.0 - invested) if t.kind in TAKES_REST else min(t.share, 1.0 - invested)
        f = fill_after(dates[i])
        if f is not None and share > 1e-12:
            out.append((f, share))
            invested += share
    if ep.end is not None and invested < 1.0 - 1e-12:
        f = fill_after(ep.end)
        if f is not None:
            out.append((f, 1.0 - invested))
    return out


def reserve_value(close: pd.Series, start: pd.Timestamp, fill_list: Sequence[Tuple[pd.Timestamp, float]],
                  cash_rate: float, horizon_years: int = HORIZON_YEARS) -> Optional[float]:
    """Value at the horizon of a reserve of 1.0 in cash at ``start``; None without data to the horizon."""
    c = close.dropna()
    after = c.index[c.index >= start + pd.DateOffset(years=horizon_years)]
    if not len(after):
        return None
    h = after[0]

    def grow(a: pd.Timestamp, b: pd.Timestamp) -> float:
        return (1.0 + cash_rate) ** ((b - a).days / 365.25)

    value, cash_share = 0.0, 1.0
    for f, share in fill_list:
        if f > h:
            continue
        value += share * grow(start, f) / c[f] * c[h]
        cash_share -= share
    return value + cash_share * grow(start, h)


# ------------------------------------------------------------------ evaluation
def with_dividends(close: pd.Series, dividend_yield: float) -> pd.Series:
    """The price index grown by an assumed dividend yield accrued daily: a total-return proxy."""
    if not dividend_yield:
        return close
    years = np.asarray((close.index - close.index[0]).days) / 365.25
    return close * (1.0 + dividend_yield) ** years


def evaluate(markets: Dict[str, Tuple[pd.Series, float]], configs: Dict[str, Config] = CONFIGS,
             yields: Optional[Dict[str, float]] = None, horizon_years: int = HORIZON_YEARS) -> pd.DataFrame:
    """One row per episode with a full horizon: each configuration's, L20's and cash's reserve value.

    Signals always read the price index.  ``yields`` (CR1b) values the invested part with an assumed
    dividend yield per market, ``DEFAULT_YIELD`` for a market not listed; None keeps the price index (CR1).
    """
    rows = []
    for market, (close, rate) in markets.items():
        c = close[close.index <= pd.Timestamp(TEST_END)].dropna()
        value = c if yields is None else with_dividends(c, yields.get(market, DEFAULT_YIELD))
        ma = c.rolling(CONFIRM_MA, min_periods=CONFIRM_MA).mean()
        for ep in episodes(c):
            row = {"market": market, "peak": ep.peak_date.date(), "start": ep.start.date(),
                   "bottom": ep.bottom.date(), "depth": round(ep.depth, 3)}
            values = {}
            for cfg in (*configs.values(), LUMP20):
                fl = fills(c, ep, cfg, ma)
                values[cfg.name] = reserve_value(value, ep.start, fl, rate, horizon_years)
                if cfg.name in configs:
                    row[f"{cfg.name}_tranches"] = len(fl)
            values["cash"] = reserve_value(value, ep.start, [], rate, horizon_years)
            if values["cash"] is None:
                continue
            rows.append({**row, **{k: round(v, 4) for k, v in values.items()}})
    return pd.DataFrame(rows)


def verdict(table: pd.DataFrame, configs: Dict[str, Config] = CONFIGS, halves: bool = False) -> Dict[str, Dict[str, object]]:
    """The pre-registered pass rules, per configuration.

    ``halves`` (CR1b rule 3) also requires the beats-cash rule in the episodes starting before the median
    start date and in those starting on or after it.
    """
    def beats_cash(t: pd.DataFrame, name: str) -> Tuple[float, float]:
        excess = t[name] - t["cash"]
        return float((excess > 0).mean()), float(excess.median())

    out = {}
    worst_l20 = float(table["L20"].min())
    start = pd.to_datetime(table["start"])
    parts = {"first_half": table[start < start.median()], "second_half": table[start >= start.median()]}
    for name in configs:
        beats, median = beats_cash(table, name)
        rec = {
            "episodes": int(len(table)), "beats_cash": round(beats, 3), "median_excess_vs_cash": round(median, 4),
            "worst": round(float(table[name].min()), 4), "worst_L20": round(worst_l20, 4),
            "median_vs_L20": round(float((table[name] - table["L20"]).median()), 4),
        }
        ok = beats >= 0.70 and median > 0 and table[name].min() > worst_l20
        if halves:
            for key, part in parts.items():
                b, m = beats_cash(part, name)
                rec[key] = {"episodes": int(len(part)), "beats_cash": round(b, 3), "median_excess_vs_cash": round(m, 4)}
                ok = ok and b >= 0.70 and m > 0
        rec["pass"] = bool(ok)
        out[name] = rec
    return out


def report_b(markets: Dict[str, pd.Series]) -> Dict[str, object]:
    """CR1b (plan 5ac): the gating verdict, then the reported-only views (5 years, clusters out, India, CR1)."""
    gated = {m: (c, CASH_RATE) for m, c in markets.items()}
    table = evaluate(gated, CONFIGS_B, DIVIDEND_YIELD)
    start = pd.to_datetime(table["start"])
    india = evaluate({m: (markets[m], INDIA_CASH_RATE) for m in INDIA_MARKETS}, CONFIGS_B, DIVIDEND_YIELD)
    return {
        "verdict": verdict(table, CONFIGS_B, halves=True),
        "horizon_5y": verdict(evaluate(gated, CONFIGS_B, DIVIDEND_YIELD, horizon_years=5), CONFIGS_B, halves=True),
        **{f"without_{k}": verdict(table[~start.between(a, b)], CONFIGS_B) for k, (a, b) in CLUSTERS.items()},
        "india": verdict(india, CONFIGS_B),
        "cr1_with_dividends": verdict(evaluate(gated, CONFIGS, DIVIDEND_YIELD)),
        "episodes": table.to_dict(orient="records"),
    }


# ------------------------------------------------------------------ data
def fetch_index(ticker: str, cache_dir: Path = CACHE_DIR) -> pd.Series:
    """Daily closes from Yahoo Finance, cached as parquet (the cache is refreshed when older than a day)."""
    cache = cache_dir / f"{ticker.strip('^').lower()}.parquet"
    if cache.exists() and pd.Timestamp.now() - pd.Timestamp(cache.stat().st_mtime, unit="s") < pd.Timedelta(days=1):
        s = pd.read_parquet(cache)["close"]
    else:
        import yfinance as yf

        raw = yf.download(ticker, period="max", interval="1d", auto_adjust=False, progress=False)
        s = (raw["Close"].iloc[:, 0] if isinstance(raw["Close"], pd.DataFrame) else raw["Close"]).dropna()
        s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
        cache.parent.mkdir(parents=True, exist_ok=True)
        s.rename("close").to_frame().to_parquet(cache)
    s.index = pd.DatetimeIndex(s.index)
    return s.sort_index()


def store_index(name: str, store_dir: str = "data/nse_engine/store") -> pd.Series:
    """One index's closes from the store's index files (e.g. NIFTY50, INDIAVIX)."""
    from nse_engine.data.panel import build_index_close

    idx = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{store_dir}/indices/*.parquet"))])
    idx["date"] = pd.to_datetime(idx["date"])
    return build_index_close(idx, pd.DatetimeIndex(sorted(idx["date"].unique())), (name,))[name].dropna()


def nifty_close(store_dir: str = "data/nse_engine/store",
                ext_history: str = "data/nse_engine/store_ext2006/external/nifty50_history.parquet") -> pd.Series:
    """NIFTY 50 closes: the store's index files, with the 2006-12 history before them."""
    main = store_index("NIFTY50", store_dir)
    parts = [main]
    if Path(ext_history).exists():
        ext = pd.read_parquet(ext_history)["close"]
        ext.index = pd.to_datetime(ext.index)
        parts.insert(0, ext[ext.index < main.index[0]])
    out = pd.concat(parts).sort_index()
    return out[~out.index.duplicated()].ffill()


# ------------------------------------------------------------------ fresh-capital requests
# No buy rule passed its test (plan 5ab / 5ac).  The owner's own plan (13 Oct 2026): fresh capital that can stay
# 5 years, staged in thirds at -20%, -30% and -40% from the peak, so each of those levels requests a third.  The
# 50/200 cross stays a facts-only note.
ALERT_LEVELS: Dict[str, Tranche] = {"dd20": Tranche("depth", 0.20), "dd30": Tranche("depth", 0.30),
                                    "dd40": Tranche("depth", 0.40), "cross": Tranche("cross")}
CAPITAL_THIRDS = {"dd20": 1, "dd30": 2, "dd40": 3}
#: Where each third waits (owner's choice, 13 Oct 2026) and the tax arithmetic, at the rates of Oct 2026.
RESERVE_SOURCE = {
    1: ["From: sell LIQUIDCASE (Zerodha Nifty 1D Rate Liquid ETF, overnight-rate, in your Zerodha demat) and buy "
        "NIFTYBEES in the same session.",
        "Tax: LIQUIDCASE gains are taxed at your slab rate however long held (about 4.1% a year kept from ~6% at "
        "the 30% slab)."],
    2: ["From: redeem your equity arbitrage fund (direct, growth; e.g. Kotak, Tata or Invesco India Arbitrage) on "
        "Coin; the money arrives in 1-2 days, then buy NIFTYBEES.",
        "Tax: arbitrage gains are taxed as equity, 12.5% after 12 months (Rs 1.25 lakh a year exempt) or 20% "
        "within (about 5.2% a year kept from ~6%, against 4.1% in a liquid fund at the 30% slab)."],
}
RESERVE_SOURCE[3] = RESERVE_SOURCE[2]
NIFTYBEES_TAX = "NIFTYBEES itself: held 12+ months, 12.5% on gains above Rs 1.25 lakh a year; sold sooner, 20%."
ALERT_LOOKBACK = 3               # sessions: a level first reached this recently still alerts, so a missed run loses nothing


def _live_episode(c: pd.Series) -> Optional[Episode]:
    eps = episodes(c)
    return eps[-1] if eps and eps[-1].end is None else None


def _level_text(t: Tranche, peak: float) -> str:
    return f"{t.level:.0%} below the peak ({peak * (1 - t.level):,.0f})" if t.kind == "depth" \
        else f"{CROSS_FAST}-day average above the {CONFIRM_MA}-day"


def status(close: pd.Series) -> Dict[str, object]:
    """Where the index stands: no episode, or the live episode with each alert level and when it was first reached."""
    c = close.dropna()
    peak = float(c.cummax().iloc[-1])
    out = {"date": str(c.index[-1].date()), "close": round(float(c.iloc[-1]), 2), "drawdown": round(float(c.iloc[-1] / peak - 1), 4)}
    live = _live_episode(c)
    if live is None:
        return {**out, "episode": None,
                "next": f"an episode starts at a close {START_DD:.0%} below the peak ({peak * (1 - START_DD):,.0f})"}
    dates = signal_dates(c, live, tuple(ALERT_LEVELS.values()))
    return {**out, "episode": {"peak": str(live.peak_date.date()), "peak_level": round(live.peak, 2), "start": str(live.start.date())},
            "levels": {_level_text(t, live.peak): str(d.date()) if d is not None else None
                       for t, d in zip(ALERT_LEVELS.values(), dates)}}


def market_conditions(close: pd.Series, vix: Optional[pd.Series] = None) -> List[str]:
    """Where the market stands at its last close: the fall, the trend and India VIX (for a capital request)."""
    c = close.dropna()
    px, peak = float(c.iloc[-1]), float(c.cummax().iloc[-1])
    since = int(len(c) - 1 - c.index.get_indexer([c[c == peak].index[-1]])[0])
    ma50, ma200 = (float(c.rolling(n).mean().iloc[-1]) for n in (CROSS_FAST, CONFIRM_MA))
    lines = [f"Market: NIFTY {px:,.0f} on {c.index[-1].date()}, {px / peak - 1:+.1%} from its {peak:,.0f} peak "
             f"({since} sessions ago); {px / float(c.iloc[-22]) - 1:+.1%} over the last month.",
             f"Trend: {px / ma200 - 1:+.1%} against its 200-day average; the 50-day average is "
             f"{'below' if ma50 < ma200 else 'above'} the 200-day."]
    v = vix.dropna() if vix is not None else pd.Series(dtype=float)
    v = v[v.index <= c.index[-1]]                         # as of the same close
    if len(v) >= 2:
        year, now = v.iloc[-252:], float(v.iloc[-1])
        mood = "panic" if now >= 30 else "elevated" if now >= 20 else "normal"
        lines.append(f"India VIX {now:.1f} ({mood}; 1-year range {year.min():.0f}-{year.max():.0f}, higher than "
                     f"{float((year < now).mean()):.0%} of the past year).")
    return lines


def level_alerts(close: pd.Series, lookback: int = ALERT_LOOKBACK,
                 vix: Optional[pd.Series] = None) -> List[Tuple[str, str, List[str]]]:
    """(key, title, lines) for each alert level the live episode first reached in the last ``lookback`` sessions.

    The key names the episode by its peak date, so each level alerts once per episode.  ``vix`` (India VIX
    closes) adds the volatility line to the market conditions.
    """
    c = close.dropna()
    live = _live_episode(c)
    if live is None:
        return []
    dates = signal_dates(c, live, tuple(ALERT_LEVELS.values()))
    reached = "; ".join(f"{_level_text(t, live.peak)}: {d.date() if d is not None else 'not yet'}"
                        for t, d in zip(ALERT_LEVELS.values(), dates))
    out = []
    for (key, t), d in zip(ALERT_LEVELS.items(), dates):
        if d is None or d < c.index[-min(lookback, len(c))]:
            continue
        where = (f"{d.date()}: NIFTY closed at {c[d]:,.0f}, {c[d] / live.peak - 1:+.1%} from its peak of "
                 f"{live.peak:,.0f} on {live.peak_date.date()}.")
        conditions = market_conditions(c, vix)
        if key in CAPITAL_THIRDS:
            n = CAPITAL_THIRDS[key]
            title = f"NIFTY {t.level:.0%} below its peak: add fresh capital, third {n} of 3"
            lines = [where,
                     f"Request (your plan of 13 Oct 2026): move third {n} of your crash reserve, money that can stay "
                     f"5 years, into NIFTYBEES (Nippon India NIFTY 50 ETF, CNC) at the next session.  Never into "
                     f"the momentum book.",
                     *RESERVE_SOURCE[n], NIFTYBEES_TAX,
                     *conditions,
                     f"This fall so far: {reached}.",
                     "Expect it to fall further first: after a first close at -20% to -40%, NIFTY-like indices "
                     "fell a further 17-21% (median) before recovering.  The case is the 5-year rebound: 13 indices "
                     "since 1929 gained a median +34% / +43% / +69% in 5 years after -20% / -30% / -40%.",
                     "Thirds not triggered in this fall stay in the reserve for the next one.  Facts, not advice "
                     "from a SEBI-registered adviser: check fund factsheets and confirm tax with your CA."]
        else:
            title = f"NIFTY {CROSS_FAST}-day average closed above its {CONFIRM_MA}-day average"
            lines = [where, *conditions, f"This fall so far: {reached}.",
                     "Facts only: the fall may be ending.  No capital request goes with this note."]
        out.append((f"crash_recovery:{live.peak_date.date()}:{key}", title, lines))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Crash-recovery capital advisory (CR1 / CR1b)")
    p.add_argument("command", choices=["report", "report-b", "status", "alert"])
    p.add_argument("--dry-run", action="store_true", help="alert: print the alerts, send nothing")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    nifty = nifty_close()
    if args.command == "status":
        print(json.dumps(status(nifty), indent=1))
        return 0
    if args.command == "alert":
        try:
            vix = store_index("INDIAVIX")
        except (KeyError, ValueError, OSError) as exc:          # the email goes without the VIX line
            logger.warning("India VIX unavailable: %s", exc)
            vix = None
        pending = level_alerts(nifty, vix=vix)
        if pending and not args.dry_run:
            from services.notifications.alerts import WARNING, alert
        for key, title, lines in pending:
            print(title, *lines, sep="\n  ")
            if not args.dry_run:
                alert(WARNING, key, title, lines, once=True)
        if not pending:
            print(f"no new level: {json.dumps(status(nifty))}")
        return 0
    markets = {"NIFTY50": nifty, **{t: fetch_index(t) for t in GLOBAL_TICKERS}}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if args.command == "report-b":
        result = report_b(markets)
        (CACHE_DIR / "report_b.json").write_text(json.dumps(result, indent=1, default=str))
        print(json.dumps({k: v for k, v in result.items() if k != "episodes"}, indent=1))
        return 0
    table = evaluate({m: (c, CASH_RATE) for m, c in markets.items()})
    india = evaluate({m: (markets[m], INDIA_CASH_RATE) for m in INDIA_MARKETS})
    result = {"verdict": verdict(table), "india": verdict(india),
              "episodes": table.to_dict(orient="records"), "episodes_india": india.to_dict(orient="records")}
    (CACHE_DIR / "report.json").write_text(json.dumps(result, indent=1, default=str))
    print(table.to_string(index=False))
    print(json.dumps({"all": result["verdict"], "india": result["india"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
