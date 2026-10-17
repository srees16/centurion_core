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
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from us_engine.membership import (COVERED_SHARE, MAX_DAY_MOVE, PRICE_RANGE, REPO_ROOT, RESEARCH_ANCHOR, US_STORE,
                                  WINDOW, yahoo_symbol)

logger = logging.getLogger(__name__)

API = "https://api.tiingo.com"
ENV_KEY = "TIINGO_API_KEY"
PACE_SECONDS = 75.0                       # ~48 requests an hour, under the free plan's 50
RATE_LIMIT_WAIT = 900                     # on 429: the cap is a rolling hour
RATE_LIMIT_RETRIES = 12                   # up to 3 hours of waiting before giving up
PRICES_DIR = US_STORE / "raw" / "tiingo" / "prices"
#: Tiingo's public list of every ticker it prices, with first and last dates: free, outside the allowance.
LISTING_URL = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"
LISTING_PATH = US_STORE / "raw" / "tiingo" / "supported_tickers.csv"
LISTING_MAX_AGE_DAYS = 7
RESOLUTION_PATH = US_STORE / "tiingo_resolution.csv"
MAX_CANDIDATES = 3
#: History kept before a spell's first window day: the engine's slowest signal needs ~256 sessions.
HISTORY_DAYS = 400
#: One-day moves beyond MAX_DAY_MOVE checked against the news and kept: (ticker, date) -> what happened.
REAL_MOVES = {
    ("GMCR", "2015-12-07"): "Keurig Green Mountain agreed to be bought by JAB at a 78% premium",
    ("SIVB", "2023-03-09"): "SVB Financial fell 60% the day before its bank failed",
    # checked 16 Oct (US4): the store drops a Tiingo series with any other move beyond MAX_DAY_MOVE
    ("GGP", "2008-10-28"): "General Growth Properties +72% in the October 2008 rebound, near bankruptcy",
    ("GGP", "2008-11-11"): "General Growth Properties -64% on its bankruptcy warning",
    ("ROH", "2008-07-10"): "Rohm and Haas +64%: Dow agreed to buy it at a 74% premium",
    ("XL", "2009-02-11"): "XL Capital +61% in the February 2009 insurer rally",
    ("LMCA", "2014-07-24"): "Liberty Media's Series C share dividend (two C shares per share): a distribution, "
                            "added back by us_engine.backtest",
    ("LMCK", "2014-07-24"): "the same day on the Series C spell, which Tiingo prices with the same series",
}
#: Pass 2 (US2 step 3): spells pass 1 could not resolve by ticker.  ``direct``: the Tiingo ticker that now
#: carries the company's history; ``names``: company names to search (a hit's name must contain one);
#: ``until``: the part of the spell that is missing (the rest is priced on Yahoo); ``prefix``: share class.
PASS2 = {
    "DISCA": {"direct": "WBD"}, "COG": {"direct": "CTRA"}, "CDAY": {"direct": "DAY"},
    "FRC": {"names": ["First Republic Bank"]}, "INFO": {"names": ["IHS Markit"]},
    "FOXA": {"names": ["Twenty-First Century Fox", "21st Century Fox"], "until": "2019-03-19", "prefix": "FOXA"},
    "FOX": {"names": ["Twenty-First Century Fox", "21st Century Fox"], "until": "2019-03-19", "prefix": "FOX"},
    "ARNC": {"names": ["Alcoa Inc"], "until": "2016-11-01"},
    "CCE": {"names": ["Coca-Cola Enterprises"]}, "ESV": {"names": ["Ensco"]}, "CVC": {"names": ["Cablevision"]},
    "LLTC": {"names": ["Linear Technology"]}, "BRCM": {"names": ["Broadcom Corp"]},
    "PCP": {"names": ["Precision Castparts"]}, "HAR": {"names": ["Harman International"]},
    "HOT": {"names": ["Starwood Hotels"]}, "MNK": {"names": ["Mallinckrodt"]}, "NE": {"names": ["Noble Corp"]},
    "WIN": {"names": ["Windstream"]}, "BTUUQ": {"names": ["Peabody Energy"]},
    "IGT": {"names": ["International Game Technology"]}, "LIFE": {"names": ["Life Technologies"]},
    "MOLX": {"names": ["Molex"]}, "JCP": {"names": ["J C Penney", "J.C. Penney", "JC Penney"]},
    "BMC": {"names": ["BMC Software"]}, "APOL": {"names": ["Apollo Education", "Apollo Group"]},
    "DF": {"names": ["Dean Foods"]}, "BIG": {"names": ["Big Lots"]},
    # Nasdaq-100 (16 Oct): Yahoo became Altaba (AABA, priced on Tiingo); Randgold's ticker GOLD is Barrick's now
    "YHOO": {"direct": "AABA"}, "GOLD": {"names": ["Randgold Resources"]}, "SHPG": {"names": ["Shire"]},
    "LMCA": {"names": ["Liberty Media Corp"]}, "LMCK": {"names": ["Liberty Media Corp"]},
    # 2006-12 (16 Oct): listed by the ticker they traded as, not the later OTC one (RSH 1982-2015; MEMC's
    # history under SUNE, its 2013 name SunEdison)
    "RSHCQ": {"direct": "RSH"}, "SUNEQ": {"names": ["SunEdison", "MEMC Electronic"], "prefix": "SUNE"},
}
#: Spells whose Tiingo series is another company's, checked by hand: left unpriced rather than mispriced.
NOT_THE_COMPANY = {
    ("GAS", "1996-01-02"): "Tiingo's GAS is AGL Resources' history (formerly ATG) throughout: 35.82 on "
                           "2010-06-01, when Nicor traded near 42; Nicor is not carried",
}
#: Search hits that are not the common stock.
NOT_COMMON = ("-P-", "-WS", "-U", "PREFERRED", "DEPOSITARY", "WARRANT", "UNITS", "NOTES", "TRUST PREFERRED")


class TiingoMonthlyLimit(RuntimeError):
    """The free plan's monthly allowance of distinct symbols is used up; resume next month (your call, 16 Oct)."""


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
            if r.status_code == 429 and _monthly(r.text):  # no point waiting: the allowance renews on the 1st
                raise TiingoMonthlyLimit(r.text[:200])
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


class Listing:
    """Tiingo's public ticker list.  Checking it before a price request costs nothing, while a request for
    a company Tiingo does not carry still spends a symbol of the monthly allowance (16 Oct: 37 of the
    2006-12 pass's first 48 requests found nothing, and the list showed none of those companies)."""

    def __init__(self, path: Path = LISTING_PATH):
        df = pd.read_csv(path, keep_default_na=False, na_values=[""], parse_dates=["startDate", "endDate"])
        df = df[df["priceCurrency"].eq("USD") & df["startDate"].notna()]
        self.spans: Dict[str, List[tuple]] = {}
        for t, a, e in zip(df["ticker"].str.upper(), df["startDate"], df["endDate"]):
            self.spans.setdefault(t, []).append((a, e))

    def covers(self, ticker: str, lo: pd.Timestamp, hi: pd.Timestamp, variants: bool = True) -> bool:
        """Whether Tiingo lists ``ticker`` (or, with ``variants``, a variant such as a Q suffix) for any
        day of [lo, hi]."""
        names = _variants(ticker) if variants else {str(ticker).upper()}
        return any(a <= hi and (pd.isna(e) or e >= lo) for n in names for a, e in self.spans.get(n, ()))


def load_listing(max_age_days: int = LISTING_MAX_AGE_DAYS) -> Optional[Listing]:
    """The list, downloaded again when older than ``max_age_days``; None (no filtering) if unavailable."""
    import io
    import zipfile

    import requests

    fresh = LISTING_PATH.exists() and (time.time() - LISTING_PATH.stat().st_mtime) < max_age_days * 86400
    if not fresh:
        try:
            resp = requests.get(LISTING_URL, timeout=120)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                LISTING_PATH.parent.mkdir(parents=True, exist_ok=True)
                LISTING_PATH.write_bytes(z.read("supported_tickers.csv"))
        except Exception as exc:                          # noqa: BLE001 - an old list still filters
            logger.warning("Tiingo's ticker list not refreshed (%s)", exc)
    return Listing() if LISTING_PATH.exists() else None


def candidates(hits: List[Dict], ticker: str) -> List[Dict]:
    """US common-stock hits whose ticker is ``ticker`` or a variant of it (exact first)."""
    want = _variants(ticker)
    out = [h for h in hits if (h.get("countryCode") == "US" and h.get("assetType") == "Stock"
                               and str(h.get("ticker", "")).upper() in want
                               and not any(x in f"{h.get('ticker', '')} {h.get('name', '')}".upper() for x in NOT_COMMON))]
    exact = yahoo_symbol(ticker).upper()
    return sorted(out, key=lambda h: (str(h["ticker"]).upper() != exact, str(h["ticker"])))


def _monthly(text: str) -> bool:
    """Whether a 429 is the monthly symbol allowance rather than the hourly request cap."""
    t = str(text).lower()
    return "month" in t or "unique symbol" in t


def price_file(ident: str, spell_start) -> Path:
    """Where a spell's Tiingo prices live: keyed by the spell's own start, not the window's."""
    return PRICES_DIR / f"{ident}_{pd.Timestamp(spell_start).date()}.parquet"


def _from(spell_start, anchor: str) -> pd.Timestamp:
    """First day to download for a spell: a year of history before it, from the anchor at the earliest."""
    return max(pd.Timestamp(anchor), pd.Timestamp(spell_start) - pd.Timedelta(days=HISTORY_DAYS))


def split_adjusted(df: pd.DataFrame) -> pd.Series:
    """Tiingo's raw close, adjusted for splits only (Yahoo's convention): earlier prices are divided by
    every later ``splitFactor``, so a split is not read as a price move."""
    factor = df["splitFactor"].fillna(1.0) if "splitFactor" in df.columns else pd.Series(1.0, index=df.index)
    later = factor[::-1].cumprod()[::-1].shift(-1, fill_value=1.0)
    return df["close"] / later


def judge(df: pd.DataFrame, sessions: pd.DatetimeIndex, ticker: str = "") -> Dict:
    """Share of ``sessions`` with a close, and the plausibility flags (as membership.coverage); a move in
    ``REAL_MOVES`` is not flagged."""
    if df.empty or not len(sessions):
        return {"share": 0.0, "flags": ""}
    close = split_adjusted(df).reindex(sessions)
    share = float(close.notna().mean())
    flags = []
    med = float(close.median()) if close.notna().any() else float("nan")
    if close.notna().any() and not PRICE_RANGE[0] <= med <= PRICE_RANGE[1]:
        flags.append(f"median close {med:,.2f}")
    moves = close.dropna().pct_change().abs()
    moves = moves[[(ticker, d.date().isoformat()) not in REAL_MOVES for d in moves.index]]
    if len(moves) and moves.max() > MAX_DAY_MOVE:
        flags.append(f"one-day move {moves.max():.0%} on {moves.idxmax().date()}")
    return {"share": round(share, 4), "flags": "; ".join(flags)}


def _spx_sessions(start: str, end: str) -> pd.DatetimeIndex:
    from us_engine.data import INDEX_TICKERS, _fetch

    close = _fetch([INDEX_TICKERS["SPX"]], start, threads=False)["Close"]
    s = close[INDEX_TICKERS["SPX"]].dropna().index
    return s[(s >= pd.Timestamp(start)) & (s <= pd.Timestamp(end))]


def fetch(window_start: str = "2013-01-01", needed: Optional[pd.DataFrame] = None,
          client: Optional[TiingoClient] = None, anchor: str = RESEARCH_ANCHOR, union: bool = False) -> pd.DataFrame:
    """Resolve and download every spell the Yahoo coverage left uncovered; resumable.  ``union``: the
    S&P 500 + Nasdaq-100 coverage file (``membership coverage --universe union``)."""
    if needed is None:
        name = f"coverage_yfinance_{window_start[:4]}{'_union' if union else ''}.csv"
        cov = pd.read_csv(US_STORE / name, parse_dates=["start", "until"])
        needed = cov[cov["status"] != "covered"]
    client = client or TiingoClient()
    listing = load_listing()
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
        a, b = _from(s.start, anchor).date().isoformat(), hi.date().isoformat()
        if listing is not None and not listing.covers(s.ticker, lo, hi):
            rows.append({"ticker": s.ticker, "start": pd.Timestamp(s.start), "until": s.until, "ident": "",
                         "name": "", "sessions": len(days), "share": 0.0, "flags": "", "status": "absent",
                         "tried": "not on Tiingo's list (no request)"})
            pd.DataFrame(rows).to_csv(RESOLUTION_PATH, index=False)
            logger.info("%s %s: absent, not on Tiingo's list (no request)", s.ticker, lo.date())
            continue
        tried, best = [], None
        direct = client.prices(yahoo_symbol(s.ticker), a, b)
        verdict = judge(direct, days, s.ticker)
        tried.append(f"{yahoo_symbol(s.ticker)}:{verdict['share']}")
        if verdict["share"] >= COVERED_SHARE and not verdict["flags"]:
            best = {"ident": yahoo_symbol(s.ticker), "name": "", "df": direct, **verdict}
        else:
            for h in candidates(client.search(s.ticker), s.ticker)[:MAX_CANDIDATES]:
                if listing is not None and not listing.covers(str(h["ticker"]), lo, hi, variants=False):
                    tried.append(f"{h['ticker']}/{h['permaTicker']}:not-listed")
                    continue                                     # no symbol spent on a company not carried then
                df = client.prices(h["permaTicker"], a, b)
                v = judge(df, days, s.ticker)
                tried.append(f"{h['ticker']}/{h['permaTicker']}:{v['share']}")
                if best is None or (v["share"], not v["flags"]) > (best["share"], not best["flags"]):
                    best = {"ident": h["permaTicker"], "name": h.get("name", ""), "df": df, **v}
                if v["share"] >= COVERED_SHARE and not v["flags"]:
                    break
            if best is None or best["share"] < verdict["share"]:
                best = {"ident": yahoo_symbol(s.ticker), "name": "", "df": direct, **verdict}
        if not best["df"].empty:
            best["df"].to_parquet(price_file(best["ident"], s.start))
        status = ("resolved" if best["share"] >= COVERED_SHARE and not best["flags"] else
                  "review" if best["share"] > 0 else "absent")
        rows.append({"ticker": s.ticker, "start": pd.Timestamp(s.start), "until": s.until, "ident": best["ident"],
                     "name": best["name"], "sessions": len(days), "share": best["share"], "flags": best["flags"],
                     "status": status, "tried": " ".join(tried)})
        pd.DataFrame(rows).to_csv(RESOLUTION_PATH, index=False)        # resumable after every spell
        logger.info("%s %s: %s (%.0f%%), %d requests so far", s.ticker, lo.date(), status,
                    100 * best["share"], client.requests)
    return pd.DataFrame(rows)


def _contains(name: str, query: str) -> bool:
    """Whether every word of ``query`` appears, in order, in ``name`` (punctuation ignored)."""
    words = re.sub(r"[^a-z0-9 ]+", " ", str(name).lower()).split()
    want = re.sub(r"[^a-z0-9 ]+", " ", str(query).lower()).split()
    return any(words[i:i + len(want)] == want for i in range(len(words) - len(want) + 1))


def _class_ok(ticker: str, prefix: Optional[str]) -> bool:
    return prefix is None or re.sub(r"(-W|_\d|\d|Q)$", "", ticker.upper()) == prefix


def _status(v: Dict) -> str:
    return "resolved" if v["share"] >= COVERED_SHARE and not v["flags"] else ("review" if v["share"] > 0 else "absent")


def rejudge() -> pd.DataFrame:
    """Judge every downloaded spell again (split-adjusted, REAL_MOVES applied); no API call.

    Always over the spell's member days in the store's window (from RESEARCH_ANCHOR), so a verdict does
    not depend on which pass ran last: judged over 2013-25 only, the spells that ended before 2013 read
    as absent and the store dropped their prices (16 Oct: AKS, CVG, BRL, CEG ...).
    """
    res = pd.read_csv(RESOLUTION_PATH, parse_dates=["start", "until"])
    sessions = _spx_sessions(RESEARCH_ANCHOR, WINDOW[1])
    w0, w1 = pd.Timestamp(RESEARCH_ANCHOR), pd.Timestamp(WINDOW[1])
    for i, r in res.iterrows():
        lo = max(r["start"], w0)
        hi = min(r["until"] - pd.Timedelta(days=1), w1) if pd.notna(r["until"]) else w1
        path = price_file(r["ident"], r["start"])
        if not path.exists():
            continue
        v = judge(pd.read_parquet(path), sessions[(sessions >= lo) & (sessions <= hi)], r["ticker"])
        wrong = NOT_THE_COMPANY.get((r["ticker"], r["start"].date().isoformat()))
        res.loc[i, ["share", "flags", "status"]] = ([0.0, wrong, "absent"] if wrong
                                                    else [v["share"], v["flags"], _status(v)])
    res.to_csv(RESOLUTION_PATH, index=False)
    return res


def fetch_pass2(window_start: str = "2013-01-01", client: Optional[TiingoClient] = None,
                anchor: str = RESEARCH_ANCHOR) -> pd.DataFrame:
    """Resolve the PASS2 spells (those still members on or after ``window_start``) by the Tiingo ticker
    that carries them now, or by company name; candidates are judged as ``rejudge`` judges."""
    res = rejudge()
    client = client or TiingoClient()
    sessions = _spx_sessions(RESEARCH_ANCHOR, WINDOW[1])
    w0, w1 = pd.Timestamp(RESEARCH_ANCHOR), pd.Timestamp(WINDOW[1])
    # text; read back all empty it is a float column, which refuses "" (16 Oct: pass 2 stopped on it)
    res["until_part"] = res["until_part"].astype(object) if "until_part" in res.columns else ""
    for i, r in res.iterrows():
        rule = PASS2.get(r["ticker"])
        if rule is None or r["status"] == "resolved" or "pass 2:" in str(r["tried"]):
            continue                                       # resolved, or already searched by name
        if pd.notna(r["until"]) and r["until"] <= pd.Timestamp(window_start):
            continue                                       # left before the window asked for
        lo = max(r["start"], w0)
        hi = min(r["until"] - pd.Timedelta(days=1), w1) if pd.notna(r["until"]) else w1
        if rule.get("until"):
            hi = min(hi, pd.Timestamp(rule["until"]) - pd.Timedelta(days=1))
        days = sessions[(sessions >= lo) & (sessions <= hi)]
        a, b = _from(r["start"], anchor).date().isoformat(), hi.date().isoformat()
        cands: List[Dict] = [{"permaTicker": rule["direct"], "ticker": rule["direct"], "name": ""}] if rule.get("direct") else []
        for q in rule.get("names", []):
            for h in client.search(q):
                if (h.get("countryCode") == "US" and h.get("assetType") == "Stock" and _contains(h.get("name", ""), q)
                        and _class_ok(str(h.get("ticker", "")), rule.get("prefix"))
                        and not any(x in f"{h.get('ticker', '')} {h.get('name', '')}".upper() for x in NOT_COMMON)
                        and h["permaTicker"] not in {c["permaTicker"] for c in cands}):
                    cands.append(h)
        best, tried = None, []
        listing = load_listing()
        for h in cands[:MAX_CANDIDATES + 1]:
            if (listing is not None and not rule.get("direct")
                    and not listing.covers(str(h["ticker"]), lo, hi, variants=False)):
                tried.append(f"{h['ticker']}/{h['permaTicker']}:not-listed")
                continue
            df = client.prices(h["permaTicker"], a, b)
            v = judge(df, days, r["ticker"])
            tried.append(f"{h['ticker']}/{h['permaTicker']}:{v['share']}")
            if best is None or (v["share"], not v["flags"]) > (best["share"], not best["flags"]):
                best = {"ident": h["permaTicker"], "name": h.get("name", ""), "df": df, **v}
            if v["share"] >= COVERED_SHARE and not v["flags"]:
                break
        if best is None or best["share"] <= float(r["share"]):
            res.loc[i, "tried"] = f"{r['tried']} | pass 2: " + (" ".join(tried) or "no candidates")
            logger.info("%s: pass 2 found nothing better (%s)", r["ticker"], " ".join(tried) or "no candidates")
        else:
            best["df"].to_parquet(price_file(best["ident"], r["start"]))
            res.loc[i, ["ident", "name", "share", "flags", "status", "until_part", "tried"]] = [
                best["ident"], best["name"], best["share"], best["flags"], _status(best), rule.get("until", ""),
                f"{r['tried']} | pass 2: " + " ".join(tried)]
            logger.info("%s: pass 2 %s %s (%.0f%%)", r["ticker"], _status(best), best["ident"], 100 * best["share"])
        res.to_csv(RESOLUTION_PATH, index=False)
    return res


def migrate(old_window_start: str = "2013-01-01") -> int:
    """Rename files saved before 16 Oct (keyed by the window's first day) to ``price_file``'s spell-start key."""
    res = pd.read_csv(RESOLUTION_PATH, parse_dates=["start", "until"])
    moved = 0
    for r in res.itertuples(index=False):
        old = PRICES_DIR / f"{r.ident}_{max(r.start, pd.Timestamp(old_window_start)).date()}.parquet"
        new = price_file(r.ident, r.start)
        if old != new and old.exists() and not new.exists():
            old.rename(new)
            moved += 1
    return moved


def fetch_history(anchor: str = RESEARCH_ANCHOR, client: Optional[TiingoClient] = None) -> int:
    """Top up each Tiingo-priced spell with the history before it (``HISTORY_DAYS``, from ``anchor`` at the
    earliest), so departed companies can be ranked from the first research day as Yahoo-priced ones are;
    files already holding that history are skipped (resumable).  Returns the files re-downloaded."""
    res = pd.read_csv(RESOLUTION_PATH, parse_dates=["start", "until"])
    client = client or TiingoClient()
    w1 = pd.Timestamp(WINDOW[1])
    done = 0
    for r in res[res["status"] != "absent"].itertuples(index=False):
        path = price_file(r.ident, r.start)
        if not path.exists():
            continue
        want = _from(r.start, anchor)
        have = pd.read_parquet(path)
        if len(have) and have.index.min() <= want + pd.Timedelta(days=10):
            continue
        hi = min(r.until - pd.Timedelta(days=1), w1) if pd.notna(r.until) else w1
        df = client.prices(r.ident, want.date().isoformat(), hi.date().isoformat())
        if len(df) > len(have):
            df.to_parquet(path)
            done += 1
            logger.info("%s: history from %s (%d rows)", r.ticker, df.index.min().date(), len(df))
    return done


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m us_engine.tiingo", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="resolve and download the spells Yahoo does not cover (resumable)")
    f.add_argument("--from", dest="start", default="2013-01-01", help="window start (a coverage file must exist)")
    f.add_argument("--universe", choices=("sp500", "union"), default="sp500",
                   help="which coverage file: the S&P 500's, or the S&P 500 + Nasdaq-100 one")
    f2 = sub.add_parser("pass2", help="resolve the PASS2 spells by today's ticker or company name (resumable)")
    f2.add_argument("--from", dest="start", default="2013-01-01")
    f3 = sub.add_parser("history", help="top up the downloaded spells with the history before them (resumable)")
    f3.add_argument("--anchor", default=RESEARCH_ANCHOR)
    sub.add_parser("migrate", help="rename files saved before 16 Oct to the spell-start key")
    sub.add_parser("rejudge", help="judge every download again over the store's window (no API call)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        return _run(args)
    except TiingoMonthlyLimit as exc:
        logger.warning("Tiingo's monthly symbol allowance is used up (%s). Progress is saved; resume on the 1st: "
                       "python -m us_engine.tiingo %s", exc, " ".join(sys.argv[1:]) or args.cmd)
        return 3                                   # distinct exit code: the chain stops, nothing is lost


def _run(args) -> int:
    if args.cmd == "history":
        print(json.dumps({"files_topped_up": fetch_history(args.anchor)}))
        return 0
    if args.cmd == "migrate":
        print(json.dumps({"files_renamed": migrate()}))
        return 0
    res = (fetch(args.start, union=args.universe == "union") if args.cmd == "fetch"
           else rejudge() if args.cmd == "rejudge" else fetch_pass2(args.start))
    print(json.dumps(res["status"].value_counts().to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
