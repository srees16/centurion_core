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
    instruments.csv                                  instrument, tickers, source, symbol, first/last member day
    manifest.json                                    sources, coverage, delisting cases, data hash

    python -m us_engine.store build
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from us_engine.data import INDEX_TICKERS, _download
from us_engine.membership import (MEMBERSHIP_PATH, NDX_MEMBERSHIP_PATH, RESEARCH_ANCHOR, RESEARCH_START,
                                  SYMBOL_MAP_PATH, US_STORE, WINDOW, in_window, load_symbol_map, segments,
                                  universe_spells)
from us_engine.tiingo import RESOLUTION_PATH, price_file, split_adjusted

logger = logging.getLogger(__name__)

PANEL_DIR = US_STORE / "panel"
STORE_VERSION = 1
RESEARCH_WINDOW = (RESEARCH_START, WINDOW[1])
FIELDS = ("open", "high", "low", "close", "volume", "dividends")
DELIST_GAP = 3
DELIST_RECOVERY = 0.7
#: Members that failed while in the index (2013-25), by membership ticker, with the evidence.
FOR_CAUSE = {
    "SIVB": "Silicon Valley Bank closed by regulators 10 Mar 2023 (FDIC receivership); removed 15 Mar",
    "SBNY": "Signature Bank closed by regulators 12 Mar 2023; removed 15 Mar",
    "FRC": "First Republic Bank seized 1 May 2023 (sold to JPMorgan); removed 4 May",
}

Key = Tuple[str, str]                      # (source, symbol): one price series


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
            tiingo_to = min(hi, pd.Timestamp(part) - pd.Timedelta(days=1)) if isinstance(part, str) and part else hi
            out.append({"ticker": s.ticker, "lo": lo, "hi": tiingo_to, "source": "tiingo", "symbol": r.ident,
                        "spell_start": s.start})
        if tiingo_to is None or tiingo_to < hi:
            start = lo if tiingo_to is None else tiingo_to + pd.Timedelta(days=1)
            for a, b, sym in segments(s.ticker, start, hi, aliases):
                out.append({"ticker": s.ticker, "lo": a, "hi": b, "source": "yahoo", "symbol": sym, "spell_start": None})
    return out


def drop_halts(df: pd.DataFrame) -> pd.DataFrame:
    """Blank days with no trading range at the previous close (a halt carried forward, not a price)."""
    prev = df["close"].shift(1)
    flat = (df["open"] == df["close"]) & (df["high"] == df["close"]) & (df["low"] == df["close"]) & (df["close"] == prev)
    out = df.copy()
    out.loc[flat, ["open", "high", "low", "close"]] = np.nan
    out.loc[flat, "volume"] = 0.0
    return out


def _tiingo_frame(symbol: str, spell_start: pd.Timestamp) -> Optional[pd.DataFrame]:
    path = price_file(symbol, spell_start)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    factor = df["close"] / split_adjusted(df)               # 1 where no later split
    return pd.DataFrame({"open": df["open"] / factor, "high": df["high"] / factor, "low": df["low"] / factor,
                         "close": df["close"] / factor, "volume": df["volume"] * factor,
                         "dividends": df.get("divCash", 0.0) / factor})


def build(window: Sequence[str] = RESEARCH_WINDOW, out_dir: Path = PANEL_DIR) -> Dict:
    spells, aliases = universe_spells(), load_symbol_map()                 # S&P 500 + Nasdaq-100
    resolution = pd.read_csv(RESOLUTION_PATH, parse_dates=["start", "until"], keep_default_na=False,
                             na_values=[""]) if RESOLUTION_PATH.exists() else pd.DataFrame()
    parts = pieces(spells, aliases, resolution, window)

    yahoo = sorted({p["symbol"] for p in parts if p["source"] == "yahoo"} | {INDEX_TICKERS["SPX"]})
    raw = _download(yahoo, RESEARCH_ANCHOR)
    spx = raw["Close"][INDEX_TICKERS["SPX"]].dropna()
    sessions = spx[(spx.index >= pd.Timestamp(RESEARCH_ANCHOR)) & (spx.index <= pd.Timestamp(window[1]))].index

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
            df = _tiingo_frame(p["symbol"], p["spell_start"])
            series[key] = df if df is not None else pd.DataFrame(columns=list(FIELDS), dtype="float64")

    # one instrument per series; its membership is the union of the pieces priced by it
    member_days: Dict[Key, List[Tuple[pd.Timestamp, pd.Timestamp, str]]] = defaultdict(list)
    for p in parts:
        member_days[(p["source"], p["symbol"])].append((p["lo"], p["hi"], p["ticker"]))
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
    research = (sessions >= pd.Timestamp(window[0]))
    m = member.loc[research]
    priced = frames["close"].loc[research].notna() & m
    coverage = float(priced.to_numpy().sum() / max(m.to_numpy().sum(), 1))
    unpriced = (m & ~priced).sum()
    data_hash = _hash(frames, member)
    manifest = {
        "store_version": STORE_VERSION, "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window": list(window), "data_anchor": RESEARCH_ANCHOR, "sessions": [str(sessions[0].date()), str(sessions[-1].date())],
        "instruments": len(rows), "by_source": pd.Series([r["source"] for r in rows]).value_counts().to_dict(),
        "members_per_day": [int(m.sum(axis=1).min()), int(m.sum(axis=1).max())],
        "member_days_priced": round(coverage, 4),
        "unpriced_member_days": {k: int(v) for k, v in unpriced[unpriced > 0].sort_values(ascending=False).items()},
        "delisting_rule": {"recovery": DELIST_RECOVERY, "cases": delisted,
                           "final_price_kept_for_review": gaps},
        "sources": {"membership": [str(MEMBERSHIP_PATH.name), str(NDX_MEMBERSHIP_PATH.name)],
                    "symbol_map": str(SYMBOL_MAP_PATH.name),
                    "tiingo_resolution": str(RESOLUTION_PATH.name)},
        "data_hash": data_hash,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in FIELDS:
        frames[f].to_parquet(out_dir / f"{f}.parquet")
    member.to_parquet(out_dir / "member.parquet")
    pd.DataFrame(rows).to_csv(out_dir / "instruments.csv", index=False)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def _hash(frames: Dict[str, pd.DataFrame], member: pd.DataFrame) -> str:
    """Content hash of the panel (sorted instruments, rounded values), as the NSE store fingerprints its data."""
    cols = sorted(member.columns)
    h = hashlib.sha256(f"us-store-v{STORE_VERSION}".encode())
    h.update(np.asarray(member.index.asi8).tobytes())
    h.update("|".join(cols).encode())
    for f in FIELDS:
        h.update(np.nan_to_num(frames[f][cols].to_numpy(dtype="float64")).round(0 if f == "volume" else 4).tobytes())
    h.update(member[cols].to_numpy(dtype=bool).tobytes())
    return h.hexdigest()[:16]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m us_engine.store", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="build the panel from the membership, Yahoo and the Tiingo downloads")
    ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    m = build()
    print(json.dumps({k: m[k] for k in ("instruments", "by_source", "members_per_day", "member_days_priced",
                                        "data_hash")}, indent=1))
    print(f"delisting rule applied to {len(m['delisting_rule']['cases'])}; unpriced member-days in "
          f"{len(m['unpriced_member_days'])} instruments")
    return 0


if __name__ == "__main__":
    sys.exit(main())
