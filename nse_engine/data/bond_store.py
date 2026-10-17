"""
Government-bond and overnight-rate total-return series for the bond stream (tracker BD1, plan 5ag), 2006 on.

- NSE's G-sec indices from niftyindices.com, total return (NSE: every fixed-income index except the 10-year
  clean-price one): the 10-year benchmark G-sec and the 1D rate (overnight) index.  Both start in 2011-12.
- Before that, AMFI's daily NAVs (``api.mfapi.in`` mirrors AMFI) of a 10-year constant-maturity gilt fund and a
  liquid fund, regular growth plans, so net of their own expenses.  Each proxy's returns run until its index's
  first date; the index's after.
- The ETFs the stream trades are in the equity store (SETF10GILT from 2016, LTGILTBEES from 2019, the liquid
  ETFs); their costs are in ``nse_engine.costs`` (no STT on debt ETFs).

    python -m nse_engine.data.bond_store build          # fetch, splice, write data/nse_engine/bond_store
    python -m nse_engine.data.bond_store proxies        # how each candidate proxy tracks its index (overlap)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path("data/nse_engine/bond_store")
NIFTY_HOME = "https://www.niftyindices.com/reports/historical-data"
NIFTY_HISTORY = "https://www.niftyindices.com/BackPage/getHistoricaldatatabletoString"
MFAPI = "https://api.mfapi.in/mf/{code}"
START = date(2006, 1, 1)

#: name -> (niftyindices trading name, its long name)
INDICES = {"gsec10": ("NIFTY GS 10YR", "Nifty 10 yr Benchmark G-Sec"),
           "rate1d": ("NIFTY 1D RATE INDEX", "Nifty 1D Rate Index")}
#: Candidate proxies per series (AMFI scheme codes, regular growth plans that ran through 2008).
CANDIDATES = {"gsec10": {108753: "Bandhan (IDFC) 10 year Constant Maturity Gilt",
                         101002: "SBI 10 year Constant Maturity Gilt",
                         100061: "Aditya Birla Sun Life Constant Maturity 10 Year Gilt"},
              "rate1d": {100851: "Nippon India Liquid", 100047: "Aditya Birla Sun Life Liquid",
                         105274: "SBI Liquid", 100835: "Kotak Liquid"}}
#: The proxy each series uses before its index: the candidate that tracked the index best over the first three
#: years of overlap, 2011-13 (``proxies``; fixed 17 Oct 2026 before any stream result, plan 5ag):
#: - gsec10: ABSL's fund, correlation 0.95, tracking error 2.1%, CAGR 4.5% against the index's 4.3%;
#: - rate1d: SBI Liquid, correlation 0.92, tracking error 0.15%, CAGR 9.2% against 8.0% (a liquid fund holds
#:   paper out to 91 days, so cash before 2011 reads about a point generous).
PROXY: Dict[str, Optional[int]] = {"gsec10": 100061, "rate1d": 105274}


def _session():
    from curl_cffi import requests

    s = requests.Session(impersonate="chrome")
    s.get(NIFTY_HOME, timeout=30)
    return s


def fetch_index(session, key: str, start: date = START, end: Optional[date] = None) -> pd.Series:
    """Daily closes of one NSE index, fetched a year at a time (the site's limit)."""
    trading, long = INDICES[key]
    end = end or date.today()
    out: Dict[pd.Timestamp, float] = {}
    a = start
    while a <= end:
        b = min(a + timedelta(days=360), end)
        body = {"cinfo": "{'name':'%s','startDate':'%s','endDate':'%s','indexName':'%s'}"
                % (trading, a.strftime("%d-%b-%Y"), b.strftime("%d-%b-%Y"), long)}
        r = session.post(NIFTY_HISTORY, json=body, timeout=60,
                         headers={"Referer": NIFTY_HOME, "X-Requested-With": "XMLHttpRequest"})
        r.raise_for_status()
        rows = json.loads(r.text)                     # a JSON list (older pages wrapped it in {"d": "..."})
        if isinstance(rows, dict):
            rows = rows.get("d") or []
        if isinstance(rows, str):
            rows = json.loads(rows)
        for row in rows:
            out[pd.Timestamp(row["HistoricalDate"])] = float(str(row["CLOSE"]).replace(",", ""))
        a = b + timedelta(days=1)
        time.sleep(0.5)
    return pd.Series(out, name=key, dtype="float64").sort_index()


def fetch_nav(session, code: int) -> pd.Series:
    """A fund's daily NAV history from AMFI (via api.mfapi.in)."""
    r = session.get(MFAPI.format(code=code), timeout=60)
    r.raise_for_status()
    rows = r.json()["data"]
    s = pd.Series({pd.Timestamp(pd.to_datetime(x["date"], format="%d-%m-%Y")): float(x["nav"]) for x in rows},
                  dtype="float64").sort_index()
    s = s[s > 0]
    # A one-day move past 50% is a face-value change (liquid funds went from Rs 10 or 100 to Rs 1000 units in
    # 2011-13), not a return: the earlier NAVs are rescaled onto the new unit.
    r = s.pct_change()
    for day in r.index[r.abs() > 0.5]:
        prev = s.index[s.index.get_loc(day) - 1]
        s.loc[:prev] *= s.loc[day] / s.loc[prev]
    return s


def splice(index: pd.Series, proxy: pd.Series) -> pd.Series:
    """One level series: the proxy's returns before the index's first date, the index's from it."""
    first = index.index[0]
    before = proxy[proxy.index <= first]
    if before.empty:
        return index
    level = before / before.iloc[-1] * index.iloc[0]
    return pd.concat([level.iloc[:-1], index]).sort_index()


def overlap_report(index: pd.Series, proxy: pd.Series, years: int = 3) -> Dict[str, float]:
    """How a proxy tracks its index over the first ``years`` of overlap, on returns between common dates."""
    lo = index.index[0]
    hi = lo + pd.DateOffset(years=years)
    common = index.index.intersection(proxy.index)
    common = common[(common >= lo) & (common <= hi)]
    a, b = index.reindex(common).pct_change().dropna(), proxy.reindex(common).pct_change().dropna()
    span = (common[-1] - common[0]).days / 365.25
    return {"days": int(len(a)), "corr": float(a.corr(b)),
            "tracking_error": float((a - b).std() * (252 ** 0.5)),
            "index_cagr": float((index[common[-1]] / index[common[0]]) ** (1 / span) - 1),
            "proxy_cagr": float((proxy[common[-1]] / proxy[common[0]]) ** (1 / span) - 1)}


def proxies(root: Path = ROOT) -> Dict:
    s = _session()
    out = {}
    for key, cands in CANDIDATES.items():
        idx = fetch_index(s, key)
        out[key] = {"index_first": str(idx.index[0].date())}
        for code, name in cands.items():
            nav = fetch_nav(s, code)
            out[key][f"{code} {name}"] = {"first_nav": str(nav.index[0].date()), **overlap_report(idx, nav)}
    return out


def build(root: Path = ROOT) -> Dict:
    """Fetch every series, splice each index onto its proxy, write ``levels.parquet`` and ``manifest.json``."""
    if any(v is None for v in PROXY.values()):
        raise RuntimeError("fix PROXY from `proxies` first (plan 5ag)")
    s = _session()
    root.mkdir(parents=True, exist_ok=True)
    cols, meta = {}, {}
    for key in INDICES:
        idx = fetch_index(s, key)
        nav = fetch_nav(s, PROXY[key])
        cols[f"{key}_index"], cols[f"{key}_proxy"] = idx, nav
        cols[key] = splice(idx, nav)
        meta[key] = {"index": INDICES[key][1], "index_first": str(idx.index[0].date()),
                     "proxy": f"AMFI {PROXY[key]} {CANDIDATES[key][PROXY[key]]}", "proxy_first": str(nav.index[0].date())}
    frame = pd.DataFrame(cols).sort_index()
    frame = frame[frame.index >= pd.Timestamp(START)]
    frame.index.name = "date"
    frame.to_parquet(root / "levels.parquet")
    digest = hashlib.sha256(pd.util.hash_pandas_object(frame.fillna(-1.0)).values.tobytes()).hexdigest()[:16]
    manifest = {"built": date.today().isoformat(), "data_hash": digest, "series": meta,
                "first": str(frame.index[0].date()), "last": str(frame.index[-1].date()),
                "sources": {"indices": NIFTY_HISTORY, "navs": MFAPI}}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def load(root: Path = ROOT) -> pd.DataFrame:
    """The spliced daily levels (``gsec10``, ``rate1d``) and their parts."""
    frame = pd.read_parquet(Path(root) / "levels.parquet")
    frame.index = pd.DatetimeIndex(frame.index)
    return frame


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nse_engine.data.bond_store", description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["build", "proxies"])
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(json.dumps(build() if args.cmd == "build" else proxies(), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
