"""
Measured trading costs of debt and liquid ETFs (India tracker BD2; your call 17 Oct 2026: measure before deciding
LN-T29's liquid sweep and any second bond round).

BD1 priced these ETFs with the equity impact model, and the verdict turned on it.  Two measurements replace it:
- ``history``: each ETF's effective spread from daily high, low and close (Abdi & Ranaldo 2017), by year, from the
  equity store; NIFTYBEES, GOLDBEES and three large caps alongside check the estimator.  The check fails for
  anything volatile (17 Oct 2026: 55-127 bp on RELIANCE, HDFCBANK and INFY, whose spreads are 1-3 bp), so it only
  bounds the near-flat liquid ETFs;
- ``snapshot`` / ``report``: Kite's five-level order book, read in market hours by a GitHub job with the day's
  stored login (a read-only quote call, no orders), kept in Neon (schema ``etf_depth``); the report walks each
  book for the cost of buying and selling Rs 1 lakh to Rs 1 crore.

    python -m nse_engine.etf_costs history
    python -m nse_engine.etf_costs snapshot      # market hours; needs today's Kite login (daily_login)
    python -m nse_engine.etf_costs report
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GILT = ("LTGILTBEES", "SETF10GILT", "GILT5YBEES", "GSEC10IETF", "GSEC10YEAR", "GSEC5IETF", "MOGSEC", "GSEC10ABSL",
        "GILT10BETA", "LTGILTCASE")
BHARAT_BOND = ("EBBETF0430", "EBBETF0431", "EBBETF0433", "BBETF0432")
LIQUID = ("LIQUIDCASE", "LIQUIDBEES", "LIQUIDETF", "LIQUIDIETF", "LIQUIDSBI", "HDFCLIQUID", "LIQUID1", "LIQUIDADD",
          "LIQUIDBETF", "ABSLLIQUID")
#: The estimator's check: instruments whose spreads are known to be a few basis points.
CHECKS = ("NIFTYBEES", "GOLDBEES", "RELIANCE", "HDFCBANK", "INFY")
SIZES_INR = (1e5, 5e5, 25e5, 1e7)
SCHEMA = "etf_depth"
EQUITY_STORE = "data/nse_engine/store"
MARKET = (time(9, 20), time(15, 25))


# ------------------------------------------------------------------ history: spreads from daily bars

def abdi_ranaldo(frame: pd.DataFrame) -> pd.Series:
    """Monthly effective spread (fraction of price) from one symbol's daily high, low and close (Abdi & Ranaldo
    2017, the two-day corrected estimator averaged within each month).  Days without trades are dropped."""
    f = frame[(frame["volume"] > 0) & (frame["low"] > 0) & (frame["high"] >= frame["low"])].sort_values("date")
    c = np.log(f["close"].to_numpy())
    eta = (np.log(f["high"].to_numpy()) + np.log(f["low"].to_numpy())) / 2
    s2 = np.maximum(4 * (c[:-1] - eta[:-1]) * (c[:-1] - eta[1:]), 0.0)
    month = pd.DatetimeIndex(f["date"].to_numpy()[:-1]).to_period("M")
    return np.sqrt(pd.Series(s2, index=month).groupby(level=0).mean())


def history(store: str = EQUITY_STORE, first_year: int = 2016) -> Dict[str, Dict]:
    """Per symbol and year: the median monthly effective spread (bp), its median traded value, the share of days
    it traded."""
    symbols = set(GILT + BHARAT_BOND + LIQUID + CHECKS) | {"NETFLTGILT", "NETFGILT5Y"}
    years = range(first_year, pd.Timestamp.today().year + 1)
    rows = pd.concat([pd.read_parquet(Path(store) / "equity" / f"{y}.parquet",
                                      columns=["date", "symbol", "high", "low", "close", "volume", "value_inr"])
                      for y in years if (Path(store) / "equity" / f"{y}.parquet").exists()])
    rows = rows[rows["symbol"].isin(symbols)]
    out: Dict[str, Dict] = {}
    for sym, f in rows.groupby("symbol"):
        spread = abdi_ranaldo(f)
        f = f.assign(year=pd.DatetimeIndex(f["date"]).year)
        by = {}
        for y, g in f.groupby("year"):
            s = spread[spread.index.year == y]
            by[int(y)] = {"spread_bp": round(float(s.median() * 1e4), 1) if len(s) else None,
                          "value_lakh_median": round(float(g["value_inr"].median() / 1e5), 1),
                          "days_traded": round(float((g["volume"] > 0).mean()), 2)}
        out[str(sym)] = by
    return out


# ------------------------------------------------------------------ depth: Kite's order book

def _book():
    from database.connection import get_db_manager
    from database.paper_cloud import PaperCloudSync

    book = PaperCloudSync(get_db_manager(), schema=SCHEMA)
    book.ensure_tables()
    return book


def snapshot(now: Optional[datetime] = None, kite=None, book=None) -> Dict:
    """One read of every ETF's five-level book, appended to the day's key in Neon.  Skips outside market hours
    (GitHub delivers crons late) and when nobody has logged in to Kite today."""
    from kite_connect.auth.daily_login import IST, kite_from_stored_token
    from nse_engine.nse_calendar import HOLIDAYS

    now = (now or datetime.now(IST)).astimezone(IST)
    if now.weekday() >= 5 or now.date().isoformat() in HOLIDAYS or not (MARKET[0] <= now.time() <= MARKET[1]):
        return {"skipped": f"outside market hours ({now:%Y-%m-%d %H:%M} IST)"}
    book = book or _book()
    kite = kite or kite_from_stored_token(book=None)
    if kite is None:
        return {"skipped": "no Kite login today"}
    names = GILT + BHARAT_BOND + LIQUID + CHECKS
    quotes = kite.quote([f"NSE:{s}" for s in names])
    snap = {"at": now.isoformat(timespec="seconds"), "q": {}}
    for s in names:
        q = quotes.get(f"NSE:{s}")
        if not q:
            continue
        d = q.get("depth") or {}
        snap["q"][s] = {"ltp": q.get("last_price"), "vol": q.get("volume"),
                        "bid": [[x["price"], x["quantity"]] for x in d.get("buy", []) if x.get("quantity")],
                        "ask": [[x["price"], x["quantity"]] for x in d.get("sell", []) if x.get("quantity")]}
    key = f"etf_depth_{now:%Y-%m-%d}"
    day = json.loads(book.read_state().get(key) or "[]")
    day.append(snap)
    if not book.sync_state({key: json.dumps(day, separators=(",", ":"))}):
        raise RuntimeError("could not store the snapshot in Neon")
    return {"stored": key, "snapshots_today": len(day), "symbols": len(snap["q"])}


def walk(levels: List[List[float]], value: float) -> Optional[float]:
    """Average price paid (or received) for ``value`` rupees across the book's levels; None past its depth."""
    left, cost, qty = value, 0.0, 0.0
    for price, quantity in levels:
        take = min(left, price * quantity)
        cost += take
        qty += take / price
        left -= take
        if left <= 1e-6:
            return cost / qty
    return None


def report(book=None) -> Dict[str, Dict]:
    """Per symbol over every snapshot: median half-spread, the median cost (bp from the mid, buys and sells
    averaged) of each size, how often the five levels held it, and the median rupees shown each side."""
    state = (book or _book()).read_state()
    snaps = [s for k in sorted(state) if k.startswith("etf_depth_") for s in json.loads(state[k])]
    rows: Dict[str, Dict[str, list]] = {}
    for snap in snaps:
        for sym, q in snap["q"].items():
            if not q["bid"] or not q["ask"]:
                continue
            bid, ask = q["bid"][0][0], q["ask"][0][0]
            if bid <= 0 or ask < bid:
                continue
            mid = (bid + ask) / 2
            r = rows.setdefault(sym, {"half_spread": [], "depth_bid": [], "depth_ask": [],
                                      **{f"cost_{int(v)}": [] for v in SIZES_INR}})
            r["half_spread"].append((ask - bid) / 2 / mid * 1e4)
            r["depth_bid"].append(sum(p * n for p, n in q["bid"]))
            r["depth_ask"].append(sum(p * n for p, n in q["ask"]))
            for v in SIZES_INR:
                b, s = walk(q["ask"], v), walk(q["bid"][::1], v)
                r[f"cost_{int(v)}"].append(None if b is None or s is None
                                           else ((b / mid - 1) + (1 - s / mid)) / 2 * 1e4)
    out = {}
    for sym, r in sorted(rows.items()):
        o = {"snapshots": len(r["half_spread"]), "half_spread_bp": round(float(np.median(r["half_spread"])), 1),
             "depth_lakh_each_side": round(float(np.median(r["depth_bid"] + r["depth_ask"]) / 1e5), 1)}
        for v in SIZES_INR:
            c = [x for x in r[f"cost_{int(v)}"] if x is not None]
            o[f"Rs {v / 1e5:g} lakh"] = {"cost_bp": round(float(np.median(c)), 1) if c else None,
                                       "filled_in_5_levels": round(len(c) / len(r[f"cost_{int(v)}"]), 2)}
        out[sym] = o
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nse_engine.etf_costs", description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["history", "snapshot", "report"])
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out = {"history": history, "snapshot": snapshot, "report": report}[args.cmd]()
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
