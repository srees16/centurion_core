"""US price store (tracker US2 step 4): every S&P 500 member of the research window, priced each day.

Free of survivorship bias by construction: the instruments are the members on
each date of the S&P 500 or the Nasdaq-100 (``data/sp500_membership.json``,
``data/ndx_membership.json``), not today's lists, and companies that no longer
trade are priced too.

Sources, per membership spell:

- Yahoo, under the spell's ticker or its curated alias (``data/sp500_symbol_map.json``);
- Tiingo, for companies Yahoo no longer prices, by the identifier US2 step 3
  resolved (``us_store/tiingo_resolution.csv``; files in ``raw/tiingo/prices``).

An instrument is a price series, not a ticker: renames that share one series
merge (CBS / VIAC / PARA / PSKY all price as SKYD; KORS and CPRI, which the lists
show as members together in 2016-18, count once), and where one ticker covered
two companies the series stay apart (Ingersoll-Rand until 2 Mar 2020 is TT's
series, the new IR another), so no return crosses a change of company.

A priced series is not proof of identity (tickers are reused): a Yahoo series
that starts more than ``LATE_START`` sessions into a member's spell, or any
series that moves more than ``MAX_DAY_MOVE`` in a day on member days without a
reviewed reason (``YAHOO_REVIEWED`` by symbol, ``tiingo.REAL_MOVES`` by
ticker and day), is dropped as another security's and listed in the manifest
(``dropped``).

Prices are split-adjusted, not dividend-adjusted (Yahoo's convention; Tiingo's
raw prices are adjusted here); cash dividends are kept per day so the loader can
build total returns, as the India store does.

Delisting rule (pre-registered in US2): acquired -> the final price; delisted
for cause -> the last real close, then ``DELIST_RECOVERY`` of it on the last
member session (the -30% CRSP-style assumption).  "For cause" is the curated
``FOR_CAUSE`` list (companies that failed while in the index); any other member
whose prices stop before its membership ends keeps its final price and is listed
in the manifest for review.  A halt is not a price: a day whose open, high, low
and close all equal the previous close is dropped (Tiingo repeats SVB's last
close through its March 2023 halt).

Layout under ``data/nse_engine/us_store/panel/``::

    open|high|low|close|volume|dividends.parquet   sessions x instruments
    member.parquet                                   sessions x instruments, True while in the index
    member_sp500.parquet, member_ndx.parquet         the same, per index (the universe choice, US-Y9)
    instruments.csv                                  instrument, tickers, source, symbol, first/last member day
    manifest.json                                    sources, coverage, delisting cases, data hash
    check.json                                       the survivorship check (``check``)

    python -m us_engine.store build
    python -m us_engine.store check      # step 5: equal-weight members vs RSP / QQQE, and the bias
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from us_engine.data import INDEX_TICKERS, _download
from us_engine.membership import (MAX_DAY_MOVE, MEMBERSHIP_PATH, NDX_MEMBERSHIP_PATH, RESEARCH_ANCHOR,
                                  RESEARCH_START, SYMBOL_MAP_PATH, US_STORE, WINDOW, in_window, load_membership,
                                  load_symbol_map, segments, universe_spells)
from us_engine.tiingo import PASS2, PRICES_DIR, REAL_MOVES, RESOLUTION_PATH, split_adjusted

logger = logging.getLogger(__name__)

PANEL_DIR = US_STORE / "panel"
STORE_VERSION = 1
RESEARCH_WINDOW = (RESEARCH_START, WINDOW[1])
FIELDS = ("open", "high", "low", "close", "volume", "dividends")
#: Market series the US engine reads (``market.parquet``, Yahoo): the indices the regime gates on and their
#: volatility indices, the rupee for INR results (US-Y2), and the 3-month T-bill yield (%) for excess returns.
MARKET_SERIES = {"SPX": "^GSPC", "NDX": "^NDX", "VIX": "^VIX", "VXN": "^VXN", "USDINR": "INR=X", "IRX": "^IRX"}
#: Funds the US book is judged against, as total-return levels (``benchmarks.parquet``): SPY and RSP for the
#: S&P 500 (cap and equal weight), QQQ and QQQE for the Nasdaq-100 (QQQE from March 2012).
BENCHMARK_FUNDS = ("SPY", "RSP", "QQQ", "QQQE")
DELIST_GAP = 3
DELIST_RECOVERY = 0.7
#: Members that failed while in the index (2013-25), by membership ticker, with the evidence.
FOR_CAUSE = {
    "SIVB": "Silicon Valley Bank closed by regulators 10 Mar 2023 (FDIC receivership); removed 15 Mar",
    "SBNY": "Signature Bank closed by regulators 12 Mar 2023; removed 15 Mar",
    "FRC": "First Republic Bank seized 1 May 2023 (sold to JPMorgan); removed 4 May",
}
#: Sessions a member's Yahoo series may start after its spell does; later, it is another security that
#: reused the ticker (16 Oct: Yahoo's KG starts in May 2008 near $120, King Pharmaceuticals traded near $10).
LATE_START = 20
#: Yahoo series whose member-day moves beyond MAX_DAY_MOVE were checked by hand and are the member's own
#: (US2, 16 Oct 2026).  An unadjusted spin-off is the member's own price too; the loader (US4) must treat it
#: as a distribution, not a loss.
YAHOO_REVIEWED = {
    "AIG": "2008-09-15 -61% (the rescue weekend); 2009 rebounds +66%, +63%",
    "FITB": "2009-02-06 +60%, bank rally",
    "FMCC": "2008-09-08 -83%, conservatorship",
    "FNMA": "2008-09-08 -90%, conservatorship",
    "GNW": "Sep-Nov 2008 swings of +67% to +89%",
    "HIG": "2008-12-05 +102%",
    "MS": "2008-10-13 +87% (the Mitsubishi UFJ stake)",
    "MTG": "2008 swings of +75% and +61%; 2012-08-02 -64%",
    "VRTX": "2013-04-19 +62% (cystic fibrosis trial results)",
    "CBRE": "2009-03-25 +64%, rebounding from $3",
    "MO": "2008-03-31 -70%: the Philip Morris International spin-off, unadjusted",
    "JCI": "2007-07-02 -61%: Tyco's split into three, unadjusted (before 2016 Yahoo's JCI is Tyco's history; "
           "the membership lists Tyco and the old Johnson Controls both as JCI)",
    "WY": "2010-07-20 -62%: the REIT conversion's special distribution",
}

Key = Tuple[str, str]                      # (source, symbol): one price series


def series_suspect(close: pd.Series, spans: Sequence[Tuple[pd.Timestamp, pd.Timestamp, str]],
                   sessions: pd.DatetimeIndex, key: Key) -> str:
    """Why a series is not the members' own over ``spans`` (lo, hi, ticker), or '' when nothing says so: a
    Yahoo series that starts ``LATE_START`` sessions into a spell (Tiingo's often start late: its history
    of delisted names begins in late 2006), or a move beyond ``MAX_DAY_MOVE`` within a spell without a
    reviewed reason."""
    source, symbol = key
    for lo, hi, ticker in spans:
        days = sessions[(sessions >= lo) & (sessions <= hi)]
        got = close.reindex(days).dropna()
        late = int((days < got.index[0]).sum()) if len(got) else 0
        if source == "yahoo" and late > LATE_START:
            return f"starts {got.index[0].date()}, {late} sessions into a spell from {days[0].date()}"
        moves = got.pct_change().abs()                    # within the spell: a gap between spells is no move
        if source == "yahoo" and symbol in YAHOO_REVIEWED:
            continue
        if source == "tiingo":
            moves = moves[[(ticker, d.date().isoformat()) not in REAL_MOVES for d in moves.index]]
        if len(moves.dropna()) and moves.max() > MAX_DAY_MOVE:
            return f"one-day move {moves.max():.0%} on {moves.idxmax().date()}"
    return ""


def pieces(spells: pd.DataFrame, aliases: pd.DataFrame, resolution: pd.DataFrame,
           window: Sequence[str] = RESEARCH_WINDOW) -> List[Dict]:
    """Each spell cut into (lo, hi, source, symbol) pieces: Tiingo where step 3 resolved it, Yahoo elsewhere."""
    w0, w1 = (pd.Timestamp(x) for x in window)
    res = {(r.ticker, pd.Timestamp(r.start)): r for r in resolution.itertuples(index=False)} if len(resolution) else {}
    out = []
    for s in in_window(spells, window).itertuples(index=False):
        lo = max(s.start, w0)
        hi = min(s.until - pd.Timedelta(days=1), w1) if pd.notna(s.until) else w1
        r = res.get((s.ticker, pd.Timestamp(s.start)))
        tiingo_to = None
        if r is not None and r.status != "absent":
            part = getattr(r, "until_part", "")
            if not (isinstance(part, str) and part):         # the rule's cut-over, also when pass 2 found nothing
                part = PASS2.get(s.ticker, {}).get("until", "")
            tiingo_to = min(hi, pd.Timestamp(part) - pd.Timedelta(days=1)) if part else hi
            out.append({"ticker": s.ticker, "lo": lo, "hi": tiingo_to, "source": "tiingo", "symbol": r.ident})
        if tiingo_to is None or tiingo_to < hi:
            start = lo if tiingo_to is None else tiingo_to + pd.Timedelta(days=1)
            for a, b, sym in segments(s.ticker, start, hi, aliases):
                out.append({"ticker": s.ticker, "lo": a, "hi": b, "source": "yahoo", "symbol": sym})
    return out


def drop_halts(df: pd.DataFrame) -> pd.DataFrame:
    """Blank days with no trading range at the previous close (a halt carried forward, not a price)."""
    prev = df["close"].shift(1)
    flat = (df["open"] == df["close"]) & (df["high"] == df["close"]) & (df["low"] == df["close"]) & (df["close"] == prev)
    out = df.copy()
    out.loc[flat, ["open", "high", "low", "close"]] = np.nan
    out.loc[flat, "volume"] = 0.0
    return out


def _tiingo_frame(symbol: str) -> Optional[pd.DataFrame]:
    """One Tiingo series from all its files: they are saved per membership spell, and a company with two
    spells (COG then CTRA, MXIM's four) is one instrument (16 Oct: only the first spell's file was read)."""
    pattern = re.compile(rf"^{re.escape(symbol)}_\d{{4}}-\d{{2}}-\d{{2}}\.parquet$")
    files = sorted(p for p in PRICES_DIR.glob(f"{symbol}_*.parquet") if pattern.match(p.name))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(p) for p in files])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    factor = df["close"] / split_adjusted(df)               # 1 where no later split
    return pd.DataFrame({"open": df["open"] / factor, "high": df["high"] / factor, "low": df["low"] / factor,
                         "close": df["close"] / factor, "volume": df["volume"] * factor,
                         "dividends": df.get("divCash", 0.0) / factor})


def build(window: Sequence[str] = RESEARCH_WINDOW, out_dir: Path = PANEL_DIR) -> Dict:
    spells, aliases = universe_spells(), load_symbol_map()                 # S&P 500 + Nasdaq-100
    resolution = pd.read_csv(RESOLUTION_PATH, parse_dates=["start", "until"], keep_default_na=False,
                             na_values=[""]) if RESOLUTION_PATH.exists() else pd.DataFrame()
    parts = pieces(spells, aliases, resolution, window)

    yahoo = sorted({p["symbol"] for p in parts if p["source"] == "yahoo"} | {INDEX_TICKERS["SPX"]}
                   | set(MARKET_SERIES.values()) | set(BENCHMARK_FUNDS))
    raw = _download(yahoo, RESEARCH_ANCHOR)
    spx = raw["Close"][INDEX_TICKERS["SPX"]].dropna()
    sessions = spx[(spx.index >= pd.Timestamp(RESEARCH_ANCHOR)) & (spx.index <= pd.Timestamp(window[1]))].index
    market = pd.DataFrame({name: raw["Close"][sym].reindex(sessions).astype("float64") if sym in raw["Close"].columns
                           else np.nan for name, sym in MARKET_SERIES.items()}, index=sessions)
    benchmarks = pd.DataFrame({f: _total_return_level(raw["Close"][f], raw["Dividends"][f]).reindex(sessions)
                               for f in BENCHMARK_FUNDS if f in raw["Close"].columns}, index=sessions)

    series: Dict[Key, pd.DataFrame] = {}
    for p in parts:
        key = (p["source"], p["symbol"])
        if key in series:
            continue
        if p["source"] == "yahoo":
            cols = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume",
                    "dividends": "Dividends"}
            series[key] = pd.DataFrame({f: raw[c][p["symbol"]] if p["symbol"] in raw[c].columns else np.nan
                                        for f, c in cols.items()}, index=raw["Close"].index)
        else:
            df = _tiingo_frame(p["symbol"])
            series[key] = df if df is not None else pd.DataFrame(columns=list(FIELDS), dtype="float64")

    # one instrument per series; its membership is the union of the pieces priced by it
    member_days: Dict[Key, List[Tuple[pd.Timestamp, pd.Timestamp, str]]] = defaultdict(list)
    for p in parts:
        member_days[(p["source"], p["symbol"])].append((p["lo"], p["hi"], p["ticker"]))
    dropped = []                                          # another security's series: left unpriced
    for key, spans in member_days.items():
        why = series_suspect(series[key]["close"], spans, sessions, key)
        if why:
            series[key] = series[key].iloc[0:0]
            dropped.append({"source": key[0], "symbol": key[1], "tickers": " ".join(sorted({t for _, _, t in spans})),
                            "reason": why})
    names: Dict[Key, str] = {}
    for key in sorted(member_days, key=lambda k: -max(hi for _, hi, _ in member_days[k]).value):
        ticker = max(member_days[key], key=lambda x: x[1])[2]          # the ticker of its latest membership
        first = min(lo for lo, _, _ in member_days[key])
        names[key] = ticker if ticker not in names.values() else f"{ticker}~{first.date()}"
    tickers = {k: list(dict.fromkeys(t for _, _, t in v)) for k, v in member_days.items()}

    cols: Dict[str, Dict[str, pd.Series]] = {f: {} for f in FIELDS}
    masks: Dict[str, np.ndarray] = {}
    delisted, rows = [], []
    gaps = []
    for key, spans in member_days.items():
        name, df = names[key], drop_halts(series[key].reindex(sessions))
        mask = np.zeros(len(sessions), dtype=bool)
        for lo, hi, _ in spans:
            mask |= (sessions >= lo) & (sessions <= hi)
        masks[name] = mask
        last_member = sessions[mask][-1] if mask.any() else None
        priced = df["close"].dropna().index
        if last_member is not None and last_member < pd.Timestamp(window[1]) and len(priced):
            last_price = priced[priced <= last_member][-1] if (priced <= last_member).any() else None
            gap = int(((sessions > last_price) & (sessions <= last_member)).sum()) if last_price is not None else 0
            cause = next((FOR_CAUSE[t] for t in tickers[key] if t in FOR_CAUSE), None)
            if last_price is not None and cause and gap > 0:
                px = float(df.at[last_price, "close"]) * DELIST_RECOVERY
                df.loc[last_member, ["open", "high", "low", "close", "volume"]] = [px, px, px, px, 0.0]
                delisted.append({"instrument": name, "last_price": str(last_price.date()),
                                 "last_member_day": str(last_member.date()), "sessions_unpriced": gap,
                                 "evidence": cause})
            elif last_price is not None and gap > DELIST_GAP:
                gaps.append({"instrument": name, "last_price": str(last_price.date()),
                             "last_member_day": str(last_member.date()), "sessions_unpriced": gap})
        for f in FIELDS:
            cols[f][name] = df[f].astype("float64")
        rows.append({"instrument": name, "tickers": " ".join(tickers[key]), "source": key[0], "symbol": key[1],
                     "first_member_day": str(sessions[mask][0].date()) if mask.any() else "",
                     "last_member_day": str(last_member.date()) if last_member is not None else ""})

    frames = {f: pd.DataFrame(cols[f], index=sessions) for f in FIELDS}          # built at once, not column by column
    member = pd.DataFrame(masks, index=sessions)
    index_member = {ix: pd.DataFrame(m, index=sessions)[member.columns] for ix, m in index_masks(member_days, names,
                                                                                                sessions).items()}
    research = (sessions >= pd.Timestamp(window[0]))
    m = member.loc[research]
    priced = frames["close"].loc[research].notna() & m
    coverage = float(priced.to_numpy().sum() / max(m.to_numpy().sum(), 1))
    unpriced = (m & ~priced).sum()
    data_hash = _hash(frames, member, index_member, market)
    manifest = {
        "store_version": STORE_VERSION, "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window": list(window), "data_anchor": RESEARCH_ANCHOR, "sessions": [str(sessions[0].date()), str(sessions[-1].date())],
        "instruments": len(rows), "by_source": pd.Series([r["source"] for r in rows]).value_counts().to_dict(),
        "members_per_day": [int(m.sum(axis=1).min()), int(m.sum(axis=1).max())],
        "member_days_priced": round(coverage, 4),
        "unpriced_member_days": {k: int(v) for k, v in unpriced[unpriced > 0].sort_values(ascending=False).items()},
        "delisting_rule": {"recovery": DELIST_RECOVERY, "cases": delisted,
                           "final_price_kept_for_review": gaps},
        "dropped": dropped,
        "sources": {"membership": [str(MEMBERSHIP_PATH.name), str(NDX_MEMBERSHIP_PATH.name)],
                    "symbol_map": str(SYMBOL_MAP_PATH.name),
                    "tiingo_resolution": str(RESOLUTION_PATH.name)},
        "data_hash": data_hash,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in FIELDS:
        frames[f].to_parquet(out_dir / f"{f}.parquet")
    member.to_parquet(out_dir / "member.parquet")
    market.to_parquet(out_dir / "market.parquet")
    benchmarks.to_parquet(out_dir / "benchmarks.parquet")
    for ix, m in index_member.items():
        m.to_parquet(out_dir / f"member_{ix}.parquet")
    pd.DataFrame(rows).to_csv(out_dir / "instruments.csv", index=False)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def index_masks(member_days: Dict[Key, List[Tuple[pd.Timestamp, pd.Timestamp, str]]], names: Dict[Key, str],
                sessions: pd.DatetimeIndex) -> Dict[str, Dict[str, np.ndarray]]:
    """Per index ("sp500", "ndx"): each instrument's member days, from its pieces' tickers and that index's own
    spells (the union spells merge the two)."""
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for ix, path in (("sp500", MEMBERSHIP_PATH), ("ndx", NDX_MEMBERSHIP_PATH)):
        if not Path(path).exists():
            continue
        spells = {t: g for t, g in load_membership(path).groupby("ticker")}
        out[ix] = {}
        for key, spans in member_days.items():
            m = np.zeros(len(sessions), dtype=bool)
            for lo, hi, ticker in spans:
                for s in (spells[ticker].itertuples(index=False) if ticker in spells else ()):
                    a = max(lo, s.start)
                    b = min(hi, s.until - pd.Timedelta(days=1)) if pd.notna(s.until) else hi
                    m |= (sessions >= a) & (sessions <= b)
            out[ix][names[key]] = m
    return out


def _total_return_level(close: pd.Series, dividends: pd.Series) -> pd.Series:
    """A fund's total-return level (1.0 on its first close), dividends reinvested on their ex-dates.  Over the
    fund's own trading days: the download also holds USD/INR, which trades on US holidays (16 Oct: the day
    after each holiday had lost its return)."""
    c = close.astype("float64").dropna()
    r = (c + dividends.reindex(c.index).fillna(0.0)) / c.shift(1) - 1.0
    return (1.0 + r.fillna(0.0)).cumprod()


def _hash(frames: Dict[str, pd.DataFrame], member: pd.DataFrame,
          index_member: Optional[Dict[str, pd.DataFrame]] = None, market: Optional[pd.DataFrame] = None) -> str:
    """Content hash of the panel (sorted instruments, rounded values), as the NSE store fingerprints its data."""
    cols = sorted(member.columns)
    h = hashlib.sha256(f"us-store-v{STORE_VERSION}".encode())
    h.update(np.asarray(member.index.asi8).tobytes())
    h.update("|".join(cols).encode())
    for f in FIELDS:
        h.update(np.nan_to_num(frames[f][cols].to_numpy(dtype="float64")).round(0 if f == "volume" else 4).tobytes())
    h.update(member[cols].to_numpy(dtype=bool).tobytes())
    for ix in sorted(index_member or {}):
        h.update(ix.encode() + index_member[ix][cols].to_numpy(dtype=bool).tobytes())
    if market is not None:                                # the regime reads these (US4)
        h.update("|".join(market.columns).encode() + np.nan_to_num(market.to_numpy(dtype="float64"), nan=-1.0).round(4).tobytes())
    return h.hexdigest()[:16]


#: The real equal-weight funds the members' basket is held against: index -> Yahoo symbol.
BENCHMARKS = {"sp500": "RSP", "ndx": "QQQE"}
#: The pass window (pre-registered in US2: within 1%/yr of RSP, >= 99% of member-days priced) and the stress
#: window, reported only.  QQQE starts in March 2012, so the Nasdaq-100 is compared over 2013-25.
CHECK_WINDOWS = {"2013-25": ("2013-01-01", "2025-12-31"), "2007-12": ("2007-01-01", "2012-12-31")}
TRACKING_LIMIT = 0.01
COVERAGE_TARGET = 0.99


def _total_returns(close: pd.DataFrame, dividends: pd.DataFrame) -> pd.DataFrame:
    """Daily total returns; a day after a missing close is measured from the last close."""
    return (close + dividends.fillna(0.0)) / close.ffill().shift(1) - 1.0


def basket(r: pd.DataFrame, held: pd.DataFrame, quarterly: bool) -> pd.Series:
    """Daily return of an equal-weight basket of ``held`` (each day's members): reset to equal every day, or
    at each quarter's first session with the weights drifting in between, as RSP; a leaver's weight goes to
    that day's entrants (else to the rest, pro rata).  A member without a price that day holds its value."""
    rv = np.where(held.to_numpy(), np.nan_to_num(r.to_numpy(dtype="float64")), 0.0)
    h = held.to_numpy(dtype=bool)
    if not quarterly:
        n = h.sum(axis=1)
        return pd.Series(np.where(n > 0, rv.sum(axis=1) / np.maximum(n, 1), np.nan), index=r.index)
    quarter = (r.index.year * 4 + (r.index.month - 1) // 3).to_numpy()
    w, out = np.zeros(h.shape[1]), np.full(len(r), np.nan)
    for t in range(len(r)):
        if t == 0 or quarter[t] != quarter[t - 1] or w.sum() <= 0:
            w = h[t] / max(h[t].sum(), 1)
        else:
            gone, new = (w > 0) & ~h[t], h[t] & (w <= 0)
            freed = w[gone].sum()
            w[gone] = 0.0
            if new.any():
                w[new] = freed / new.sum()
            elif w.sum() > 0:
                w *= 1.0 + freed / w.sum()
        if w.sum() > 0:
            out[t] = float((w * rv[t]).sum() / w.sum())
            w = w * (1.0 + rv[t])
    return pd.Series(out, index=r.index)


def _cagr(r: pd.Series) -> float:
    r = r.dropna()
    years = (r.index[-1] - r.index[0]).days / 365.25 if len(r) > 1 else 0.0
    return float(np.prod(1.0 + r.to_numpy()) ** (1.0 / years) - 1.0) if years > 0 else float("nan")


def check(panel_dir: Path = PANEL_DIR) -> Dict:
    """US2 step 5, the survivorship check: per index, an equal-weight basket of each day's members (total
    return, daily and quarterly rebalanced) against the real equal-weight fund, and the bias itself: today's
    members backfilled over the same days, same rules."""
    close, dividends = (pd.read_parquet(panel_dir / f"{f}.parquet") for f in ("close", "dividends"))
    r = _total_returns(close, dividends)
    funds = pd.read_parquet(panel_dir / "benchmarks.parquet")        # total-return levels (build)
    out: Dict = {"limits": {"tracking_per_year": TRACKING_LIMIT, "coverage": COVERAGE_TARGET,
                            "pass_basis": "quarterly rebalanced, as RSP", "pass_window": "2013-25"}}
    for ix, fund in BENCHMARKS.items():
        path = panel_dir / f"member_{ix}.parquet"
        if not path.exists():
            continue
        member = pd.read_parquet(path)[close.columns]
        bench = funds[fund] / funds[fund].shift(1) - 1.0
        today = member.iloc[-1]                              # the last session's members, backfilled
        naive = pd.DataFrame(np.broadcast_to(today.to_numpy(), member.shape), index=member.index,
                             columns=member.columns) & close.notna()
        rows = {}
        for name, (a, b) in CHECK_WINDOWS.items():
            days = (member.index >= pd.Timestamp(a)) & (member.index <= pd.Timestamp(b))
            m, rw = member.loc[days], r.loc[days]
            fund_r = bench.reindex(member.index[days])
            if fund_r.notna().mean() < 0.95:                 # QQQE before March 2012
                continue
            pit_d, pit_q = _cagr(basket(rw, m, False)), _cagr(basket(rw, m, True))
            ref, naive_d = _cagr(fund_r), _cagr(basket(rw, naive.loc[days], False))
            rows[name] = {"coverage": round(float((m & close.loc[days].notna()).to_numpy().sum()
                                                  / max(m.to_numpy().sum(), 1)), 4),
                          "fund": fund, "fund_cagr": round(ref, 4), "members_daily": round(pit_d, 4),
                          "members_quarterly": round(pit_q, 4), "tracking_quarterly": round(pit_q - ref, 4),
                          "tracking_daily": round(pit_d - ref, 4), "todays_members_backfilled": round(naive_d, 4),
                          "survivorship_bias": round(naive_d - pit_d, 4)}
        out[ix] = rows
    sp = out.get("sp500", {}).get("2013-25", {})
    out["pass"] = {"coverage": bool(sp.get("coverage", 0) >= COVERAGE_TARGET),
                   "tracking": bool(abs(sp.get("tracking_quarterly", 1.0)) <= TRACKING_LIMIT)}
    (panel_dir / "check.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m us_engine.store", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="build the panel from the membership, Yahoo and the Tiingo downloads")
    sub.add_parser("check", help="step 5: equal-weight members vs RSP / QQQE, and the survivorship bias")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.cmd == "check":
        print(json.dumps(check(), indent=1))
        return 0
    m = build()
    print(json.dumps({k: m[k] for k in ("instruments", "by_source", "members_per_day", "member_days_priced",
                                        "data_hash")}, indent=1))
    print(f"delisting rule applied to {len(m['delisting_rule']['cases'])}; unpriced member-days in "
          f"{len(m['unpriced_member_days'])} instruments; {len(m['dropped'])} series dropped as another "
          "security's (manifest dropped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
