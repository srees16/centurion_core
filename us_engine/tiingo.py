"""Tiingo prices for S&P 500 members Yahoo no longer prices (tracker US2 step 3).

Tiingo keeps companies that stopped trading, but a reused ticker returns its
latest owner (STI is now Solidion; SunTrust is permaTicker US000000000420).  So
each membership spell is resolved: first its own ticker; if that does not cover
the spell, Tiingo's search (delisted included) and the candidates whose ticker is
a variant of it (STI, HOT-W, DTV1, SIVBQ ...), keeping the one that covers the
most sessions with plausible prices.  A spell nothing covers is left for review.

The token comes from ``TIINGO_API_KEY`` (``.env``) and is sent as a header, never
in a URL or a log line.  The free plan allows about 50 requests an hour, so
requests are paced (``PACE_SECONDS``) and the run resumes from what is on disk:

    nohup python -m us_engine.tiingo fetch --from 2013-01-01 > data/nse_engine/us_store/tiingo_fetch.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from us_engine.membership import (COVERED_SHARE, MAX_DAY_MOVE, PRICE_RANGE, REPO_ROOT, US_STORE, WINDOW,
                                  yahoo_symbol)

logger = logging.getLogger(__name__)

API = "https://api.tiingo.com"
ENV_KEY = "TIINGO_API_KEY"
PACE_SECONDS = 75.0                       # ~48 requests an hour, under the free plan's 50
RATE_LIMIT_WAIT = 900                     # on 429: the cap is a rolling hour
RATE_LIMIT_RETRIES = 12                   # up to 3 hours of waiting before giving up
PRICES_DIR = US_STORE / "raw" / "tiingo" / "prices"
RESOLUTION_PATH = US_STORE / "tiingo_resolution.csv"
MAX_CANDIDATES = 3
#: Search hits that are not the common stock.
NOT_COMMON = ("-P-", "-WS", "-U", "PREFERRED", "DEPOSITARY", "WARRANT", "UNITS", "NOTES", "TRUST PREFERRED")


class TiingoClient:
    """Paced, retrying Tiingo REST client; the token travels in a header only."""

    def __init__(self, token: Optional[str] = None, pace: float = PACE_SECONDS):
        import requests

        if token is None:
            from dotenv import dotenv_values

            token = os.environ.get(ENV_KEY) or dotenv_values(REPO_ROOT / ".env").get(ENV_KEY)
        if not (token or "").strip():
            raise RuntimeError(f"{ENV_KEY} is not set (.env)")
        self._s = requests.Session()
        self._s.headers.update({"Authorization": f"Token {token.strip()}", "Content-Type": "application/json"})
        self.pace = pace
        self._last = 0.0
        self.requests = 0

    def _get(self, path: str, params: Dict) -> Optional[object]:
        import requests

        limited = 0
        for attempt in range(5 + RATE_LIMIT_RETRIES):
            wait = self.pace - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.requests += 1
            try:
                r = self._s.get(f"{API}{path}", params=params, timeout=60)
            except requests.RequestException as exc:          # a dropped connection must not end a long run
                logger.warning("Tiingo request failed (%s); retrying in %ds", type(exc).__name__, 30 * (attempt + 1))
                time.sleep(30 * (attempt + 1))
                continue
            if r.status_code == 404:
                return None
            if r.status_code == 429:                       # the hourly cap: wait it out, many times if need be
                limited += 1
                if limited > RATE_LIMIT_RETRIES:
                    break
                logger.warning("Tiingo 429 on %s; waiting %ds (%d/%d)", path.split("/")[3], RATE_LIMIT_WAIT,
                               limited, RATE_LIMIT_RETRIES)
                time.sleep(RATE_LIMIT_WAIT)
                continue
            if r.status_code >= 500:
                logger.warning("Tiingo %s on %s; retrying in %ds", r.status_code, path.split("/")[3], 30 * (attempt + 1))
                time.sleep(30 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"Tiingo kept failing on {path}")

    def search(self, query: str) -> List[Dict]:
        return self._get("/tiingo/utilities/search",
                         {"query": query, "includeDelisted": "true", "exactTickerMatch": "false", "limit": 25}) or []

    def prices(self, ident: str, start: str, end: str) -> pd.DataFrame:
        rows = self._get(f"/tiingo/daily/{ident}/prices", {"startDate": start, "endDate": end}) or []
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        return df.set_index("date").sort_index()


def _variants(ticker: str) -> set:
    t = yahoo_symbol(ticker).upper()
    base = {t, t.replace("-", "."), t.replace("-", "")}
    return base | {f"{b}{s}" for b in base for s in ("-W", "1", "2", "_1", "Q", "-OLD")}


def candidates(hits: List[Dict], ticker: str) -> List[Dict]:
    """US common-stock hits whose ticker is ``ticker`` or a variant of it (exact first)."""
    want = _variants(ticker)
    out = [h for h in hits if (h.get("countryCode") == "US" and h.get("assetType") == "Stock"
                               and str(h.get("ticker", "")).upper() in want
                               and not any(x in f"{h.get('ticker', '')} {h.get('name', '')}".upper() for x in NOT_COMMON))]
    exact = yahoo_symbol(ticker).upper()
    return sorted(out, key=lambda h: (str(h["ticker"]).upper() != exact, str(h["ticker"])))


def judge(df: pd.DataFrame, sessions: pd.DatetimeIndex) -> Dict:
    """Share of ``sessions`` with a close, and the plausibility flags (as membership.coverage)."""
    if df.empty or not len(sessions):
        return {"share": 0.0, "flags": ""}
    close = df["close"].reindex(sessions)
    share = float(close.notna().mean())
    flags = []
    med = float(close.median()) if close.notna().any() else float("nan")
    if close.notna().any() and not PRICE_RANGE[0] <= med <= PRICE_RANGE[1]:
        flags.append(f"median close {med:,.2f}")
    move = close.dropna().pct_change().abs().max()
    if pd.notna(move) and move > MAX_DAY_MOVE:
        flags.append(f"one-day move {move:.0%}")
    return {"share": round(share, 4), "flags": "; ".join(flags)}


def _spx_sessions(start: str, end: str) -> pd.DatetimeIndex:
    from us_engine.data import INDEX_TICKERS, _fetch

    close = _fetch([INDEX_TICKERS["SPX"]], start, threads=False)["Close"]
    s = close[INDEX_TICKERS["SPX"]].dropna().index
    return s[(s >= pd.Timestamp(start)) & (s <= pd.Timestamp(end))]


def fetch(window_start: str = "2013-01-01", needed: Optional[pd.DataFrame] = None,
          client: Optional[TiingoClient] = None) -> pd.DataFrame:
    """Resolve and download every spell the Yahoo coverage left uncovered; resumable."""
    if needed is None:
        cov = pd.read_csv(US_STORE / f"coverage_yfinance_{window_start[:4]}.csv", parse_dates=["start", "until"])
        needed = cov[cov["status"] != "covered"]
    client = client or TiingoClient()
    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(RESOLUTION_PATH, parse_dates=["start", "until"]) if RESOLUTION_PATH.exists() else pd.DataFrame()
    w0, w1 = pd.Timestamp(window_start), pd.Timestamp(WINDOW[1])
    sessions = _spx_sessions(window_start, WINDOW[1])
    rows = done.to_dict("records")
    seen = {(r["ticker"], pd.Timestamp(r["start"])) for r in rows}
    for s in needed.itertuples(index=False):
        if (s.ticker, pd.Timestamp(s.start)) in seen:
            continue
        lo = max(pd.Timestamp(s.start), w0)
        hi = min(pd.Timestamp(s.until) - pd.Timedelta(days=1), w1) if pd.notna(s.until) else w1
        days = sessions[(sessions >= lo) & (sessions <= hi)]
        a, b = (lo - pd.Timedelta(days=10)).date().isoformat(), hi.date().isoformat()
        tried, best = [], None
        direct = client.prices(yahoo_symbol(s.ticker), a, b)
        verdict = judge(direct, days)
        tried.append(f"{yahoo_symbol(s.ticker)}:{verdict['share']}")
        if verdict["share"] >= COVERED_SHARE and not verdict["flags"]:
            best = {"ident": yahoo_symbol(s.ticker), "name": "", "df": direct, **verdict}
        else:
            for h in candidates(client.search(s.ticker), s.ticker)[:MAX_CANDIDATES]:
                df = client.prices(h["permaTicker"], a, b)
                v = judge(df, days)
                tried.append(f"{h['ticker']}/{h['permaTicker']}:{v['share']}")
                if best is None or (v["share"], not v["flags"]) > (best["share"], not best["flags"]):
                    best = {"ident": h["permaTicker"], "name": h.get("name", ""), "df": df, **v}
                if v["share"] >= COVERED_SHARE and not v["flags"]:
                    break
            if best is None or best["share"] < verdict["share"]:
                best = {"ident": yahoo_symbol(s.ticker), "name": "", "df": direct, **verdict}
        if not best["df"].empty:
            best["df"].to_parquet(PRICES_DIR / f"{best['ident']}_{lo.date()}.parquet")
        status = ("resolved" if best["share"] >= COVERED_SHARE and not best["flags"] else
                  "review" if best["share"] > 0 else "absent")
        rows.append({"ticker": s.ticker, "start": pd.Timestamp(s.start), "until": s.until, "ident": best["ident"],
                     "name": best["name"], "sessions": len(days), "share": best["share"], "flags": best["flags"],
                     "status": status, "tried": " ".join(tried)})
        pd.DataFrame(rows).to_csv(RESOLUTION_PATH, index=False)        # resumable after every spell
        logger.info("%s %s: %s (%.0f%%), %d requests so far", s.ticker, lo.date(), status,
                    100 * best["share"], client.requests)
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m us_engine.tiingo", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="resolve and download the spells Yahoo does not cover (resumable)")
    f.add_argument("--from", dest="start", default="2013-01-01", help="window start (a coverage file must exist)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    res = fetch(args.start)
    print(json.dumps(res["status"].value_counts().to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
