"""
External index history for sessions the NSE archive does not cover (tracker K4).

NSE's daily index files (``ind_close_all``) start in February 2012, so a store
built from the archive has no NIFTY 50 before then and the regime gate cannot
see 2008.  This module builds a NIFTY 50 close series for those sessions and
caches it in ``<store>/external/``.  ``load_market_data`` uses the cache to
fill gaps only: NSE's own values always win.

Sources, checked 27 Sep 2026:

* from 2007-09-17, Yahoo ``^NSEI``.  It equals NSE's closes to 0.01 on all
  455 sessions compared (2012-02-21 to 2013-12-31);
* before 2007-09-17, and on the ~1% of later sessions Yahoo lacks, the BSE
  Sensex (Yahoo ``^BSESN``) scaled by the NIFTY/Sensex ratio of the nearest
  earlier common session (the mean of the first 20 common sessions before
  any overlap).  The trend state the regime reads, close above its 200-day
  mean, agrees with NIFTY's on 98.4% of 1,339 overlapping sessions.  Rows
  from the proxy are labelled in the ``source`` column;
* sessions neither covers (Diwali Muhurat evenings, Saturday special sessions,
  two Yahoo gaps) carry the previous close forward, labelled as such.

NSE's own daily files also lack NIFTY 50 on a dozen sessions in 2013-2016; the
cache fills those too.

    python -m nse_engine.data.external nifty50 --store data/nse_engine/store_ext2006
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import pandas as pd

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

#: index name -> cache file in ``<store>/external/`` used to fill that index's gaps
EXTERNAL_INDEX_FILES: Dict[str, str] = {"NIFTY50": "nifty50_history.parquet"}
PRIMARY_SOURCE = "yahoo:^NSEI"
PROXY_SOURCE = "proxy:yahoo:^BSESN x NIFTY/Sensex ratio"
CARRIED_SOURCE = "carried forward: no source for this session"
RATIO_SEED_SESSIONS = 20


def read_index_history(store_dir: PathLike, name: str) -> pd.Series:
    """Cached external closes for ``name`` (empty when there is no cache)."""
    fname = EXTERNAL_INDEX_FILES.get(name)
    path = Path(store_dir) / "external" / fname if fname else None
    if path is None or not path.exists():
        return pd.Series(dtype="float64")
    df = pd.read_parquet(path)
    s = df["close"].astype("float64")
    s.index = pd.DatetimeIndex(df.index).normalize()
    return s[~s.index.duplicated(keep="last")].sort_index().dropna()


def build_index_history(primary: pd.Series, proxy: pd.Series, sessions: Sequence,
                        primary_label: str = PRIMARY_SOURCE, proxy_label: str = PROXY_SOURCE) -> pd.DataFrame:
    """Closes on ``sessions``: ``primary`` where it has a value, else ``proxy``
    scaled by the primary/proxy ratio of the nearest earlier common session.

    Returns a frame indexed by date with ``close`` and ``source``; sessions
    neither series covers are left out.
    """
    sessions = pd.DatetimeIndex(pd.to_datetime(list(sessions))).normalize()
    p = pd.Series(primary, dtype="float64").dropna()
    q = pd.Series(proxy, dtype="float64").dropna()
    p.index = pd.DatetimeIndex(p.index).normalize()
    q.index = pd.DatetimeIndex(q.index).normalize()
    common = p.index.intersection(q.index)
    if len(common) == 0:
        raise ValueError("primary and proxy never overlap: no ratio to scale the proxy")
    ratio = (p.loc[common] / q.loc[common]).sort_index()
    seed = float(ratio.iloc[:RATIO_SEED_SESSIONS].mean())
    ratio_on = ratio.reindex(ratio.index.union(sessions)).ffill().reindex(sessions).fillna(seed)
    close = p.reindex(sessions)
    source = pd.Series(primary_label, index=sessions, dtype=object).where(close.notna(), None)
    use_proxy = close.isna() & q.reindex(sessions).notna()
    close[use_proxy] = q.reindex(sessions)[use_proxy] * ratio_on[use_proxy]
    source[use_proxy] = proxy_label
    out = pd.DataFrame({"close": close, "source": source}, index=sessions).dropna(subset=["close"])
    out.index.name = "date"
    return out


def _yahoo_close(ticker: str, start: str, end: str) -> pd.Series:
    import yfinance as yf

    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"Yahoo returned no data for {ticker}")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    return close.dropna().astype("float64")


def write_nifty50_history(store_dir: PathLike) -> Dict[str, object]:
    """Fetch and cache NIFTY 50 closes for the store's sessions that lack them."""
    from nse_engine.data.panel import read_indices

    store = Path(store_dir)
    cal = pd.read_parquet(store / "calendar.parquet")
    sessions = pd.DatetimeIndex(pd.to_datetime(cal.iloc[:, 0])).normalize().sort_values()
    idx = read_indices(store, sessions[0], sessions[-1])
    have = pd.DatetimeIndex(idx.loc[idx["index_name"].astype(str) == "NIFTY50", "date"]).normalize()
    missing = sessions[~sessions.isin(have)]
    if missing.empty:
        return {"store": str(store), "missing_sessions": 0, "written": 0}
    start = (missing[0] - pd.Timedelta(days=400)).date().isoformat()
    end = (missing[-1] + pd.Timedelta(days=400)).date().isoformat()
    hist = build_index_history(_yahoo_close("^NSEI", start, end), _yahoo_close("^BSESN", start, end), missing)
    # Sessions neither source has (Diwali Muhurat evenings, Saturday special
    # sessions, a few Yahoo gaps) carry the previous close forward: causal, and a
    # single missing value would otherwise blank the 200-day mean for 200 sessions.
    nse_own = idx.loc[idx["index_name"].astype(str) == "NIFTY50"].set_index("date")["close"].astype("float64")
    nse_own.index = pd.DatetimeIndex(nse_own.index).normalize()
    known = pd.concat([nse_own, hist["close"]]).sort_index()
    known = known[~known.index.duplicated(keep="first")]
    uncovered = missing[~missing.isin(hist.index)]
    carried = known.reindex(known.index.union(uncovered)).ffill().reindex(uncovered).dropna()
    if len(carried):
        hist = pd.concat([hist, pd.DataFrame({"close": carried, "source": CARRIED_SOURCE})]).sort_index()
        hist.index.name = "date"
    out = store / "external" / EXTERNAL_INDEX_FILES["NIFTY50"]
    out.parent.mkdir(parents=True, exist_ok=True)
    hist.to_parquet(out)
    counts = hist["source"].value_counts().to_dict()
    report = {"store": str(store), "missing_sessions": int(len(missing)), "written": int(len(hist)),
              "uncovered": int(len(missing) - len(hist)), "first": str(hist.index[0].date()),
              "last": str(hist.index[-1].date()), "by_source": counts, "file": str(out)}
    logger.info("NIFTY50 history cached: %s", report)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Cache external index history for sessions the NSE archive lacks")
    ap.add_argument("index", choices=["nifty50"])
    ap.add_argument("--store", required=True, help="store directory, e.g. data/nse_engine/store_ext2006")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(write_nifty50_history(args.store))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
