"""
Pre-announcement drift on NSE (tracker GT13, plan section 5w).

Lucca & Moench, "The Pre-FOMC Announcement Drift" (JF 2015): from 1994 to
2011 the S&P 500 earned about 80% of its excess return in the 24 hours before
the eight scheduled FOMC statements a year, and nothing on the news itself.
India has two clocks to wait on:

* RBI policy announcements land during the session (10:00-12:00 IST).  The
  waiting stretch ends at the announcement day's open, so a buy AMO filled at
  the open of the session before and a sell AMO filled at the announcement
  day's open hold exactly that stretch, with no intraday order.
* FOMC statements land after the Indian close (23:30 or 00:30 IST), so the
  Indian session on or before the US decision date is all waiting.  Holding
  it from its open to the next open also holds the reaction, which the paper
  finds flat on average.

Tested 10 Oct 2026 (plan 5w): FAIL.  Over 2013-25 the RBI window earned +10.8 bp (t 1.16) and the
FOMC window +8.4 bp (t 0.91), both negative after a NIFTYBEES round trip.  The windows hold about 13% of
NIFTY's excess return in 2.4-3.2% of sessions: the paper's direction, not its size.

A window is a pair of fills relative to the event's *anchor session*:
``(session offset, "open" | "close")``.  ``drift_schedule`` turns events into
dated orders, ``window_returns`` prices them, and ``drift_report`` sets them
against the rest of the market.  Scheduled announcements only: an
unscheduled one cannot be waited for.  CLI::

    python -m nse_engine.event_drift report
    python -m nse_engine.event_drift schedule --from 2026-10-12
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import date, timedelta
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

Fill = Tuple[int, str]                       # (session offset from the anchor, "open" | "close")
PRICE_POINTS = ("open", "close")              # in time order within a session

#: Scheduled RBI monetary policy announcements, 2012 to the published 2026 calendar (rbi.org.in: the
#: Monetary Policy archive, each press release, the MPC schedules; checked 10 Oct 2026).  Governor's
#: reviews to Aug 2016 (11:00 IST), MPC resolutions from Oct 2016 (14:30, then 11:45 from Feb 2019,
#: 10:00 to 12:00 from Aug 2020): always after the 09:15 open.  Re-dated meetings are kept (they were
#: published in advance); unscheduled actions are left out (Mar 2012, Jul and Oct 2013, Jan and
#: Mar 2015, Mar to May 2020, May 2021, May 2022).
RBI_POLICY_DATES: Tuple[str, ...] = (
    "2012-01-24", "2012-03-15", "2012-04-17", "2012-06-18", "2012-07-31", "2012-09-17", "2012-10-30", "2012-12-18",
    "2013-01-29", "2013-03-19", "2013-05-03", "2013-06-17", "2013-07-30", "2013-09-20", "2013-10-29", "2013-12-18",
    "2014-01-28", "2014-04-01", "2014-06-03", "2014-08-05", "2014-09-30", "2014-12-02", "2015-02-03", "2015-04-07",
    "2015-06-02", "2015-08-04", "2015-09-29", "2015-12-01", "2016-02-02", "2016-04-05", "2016-06-07", "2016-08-09",
    "2016-10-04", "2016-12-07", "2017-02-08", "2017-04-06", "2017-06-07", "2017-08-02", "2017-10-04", "2017-12-06",
    "2018-02-07", "2018-04-05", "2018-06-06", "2018-08-01", "2018-10-05", "2018-12-05", "2019-02-07", "2019-04-04",
    "2019-06-06", "2019-08-07", "2019-10-04", "2019-12-05", "2020-02-06", "2020-08-06", "2020-10-09", "2020-12-04",
    "2021-02-05", "2021-04-07", "2021-06-04", "2021-08-06", "2021-10-08", "2021-12-08", "2022-02-10", "2022-04-08",
    "2022-06-08", "2022-08-05", "2022-09-30", "2022-12-07", "2023-02-08", "2023-04-06", "2023-06-08", "2023-08-10",
    "2023-10-06", "2023-12-08", "2024-02-08", "2024-04-05", "2024-06-07", "2024-08-08", "2024-10-09", "2024-12-06",
    "2025-02-07", "2025-04-09", "2025-06-06", "2025-08-06", "2025-10-01", "2025-12-05", "2026-02-06", "2026-04-08",
    "2026-06-05", "2026-08-05", "2026-10-07", "2026-12-04",
)
#: Scheduled FOMC statements, 2012 to the published 2026 calendar (federalreserve.gov meeting calendars
#: and historical pages; checked 10 Oct 2026): 2:00 pm ET from Mar 2013 (12:20 to 2:15 pm before),
#: after the Indian close.  The unscheduled statements of 11 Oct 2019 and 3, 15 and 23 Mar 2020 are
#: left out.
FOMC_DATES: Tuple[str, ...] = (
    "2012-01-25", "2012-03-13", "2012-04-25", "2012-06-20", "2012-08-01", "2012-09-13", "2012-10-24", "2012-12-12",
    "2013-01-30", "2013-03-20", "2013-05-01", "2013-06-19", "2013-07-31", "2013-09-18", "2013-10-30", "2013-12-18",
    "2014-01-29", "2014-03-19", "2014-04-30", "2014-06-18", "2014-07-30", "2014-09-17", "2014-10-29", "2014-12-17",
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17", "2015-07-29", "2015-09-17", "2015-10-28", "2015-12-16",
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15", "2016-07-27", "2016-09-21", "2016-11-02", "2016-12-14",
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14", "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13",
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13", "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19", "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    "2020-01-29", "2020-04-29", "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16", "2021-01-27",
    "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15", "2022-01-26",
    "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14", "2023-02-01",
    "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13", "2024-01-31",
    "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18", "2025-01-29",
    "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10", "2026-01-28",
    "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
)

#: The pre-registered windows (plan 5w): H1 RBI and H2 FOMC, the tradable and the reported ones.
WINDOWS: Dict[str, Dict[str, object]] = {
    "rbi": {"dates": "RBI_POLICY_DATES", "after_close": False, "entry": (-1, "open"), "exit": (0, "open"),
            "post": ((0, "open"), (0, "close"))},
    "fomc": {"dates": "FOMC_DATES", "after_close": True, "entry": (0, "open"), "exit": (1, "open"),
             "pure": ((-1, "close"), (0, "close")), "post": ((1, "open"), (1, "close"))},
}


def anchor_sessions(sessions: pd.DatetimeIndex, dates: Iterable, after_close: bool) -> pd.DatetimeIndex:
    """The session each announcement belongs to: the one it lands in (``after_close`` False: on its date,
    or the next session when that was a holiday) or the last one before it (True: on or before its date)."""
    d = pd.DatetimeIndex(pd.to_datetime(list(dates))).sort_values()
    pos = sessions.searchsorted(d, side="right") - 1 if after_close else sessions.searchsorted(d, side="left")
    pos = pos[(pos >= 0) & (pos < len(sessions))]
    return pd.DatetimeIndex(sessions[np.unique(pos)])


def drift_schedule(sessions: pd.DatetimeIndex, anchors: Iterable, entry: Fill, exit: Fill) -> pd.DataFrame:
    """One row per event: the buy and sell sessions and price points.  An order filled at the open is
    placed as an AMO the evening before; a window that runs off ``sessions`` is left out."""
    for _, point in (entry, exit):
        if point not in PRICE_POINTS:
            raise ValueError(f"price point must be one of {PRICE_POINTS}, got {point!r}")
    rows = []
    for a in pd.DatetimeIndex(anchors):
        i = sessions.get_loc(a)
        j, k = i + entry[0], i + exit[0]
        if 0 <= j and k < len(sessions) and (j, PRICE_POINTS.index(entry[1])) < (k, PRICE_POINTS.index(exit[1])):
            rows.append({"anchor": a, "buy_session": sessions[j], "buy_at": entry[1],
                         "sell_session": sessions[k], "sell_at": exit[1]})
    return pd.DataFrame(rows, columns=["anchor", "buy_session", "buy_at", "sell_session", "sell_at"])


def window_returns(open_: pd.Series, close: pd.Series, anchors: Iterable, entry: Fill, exit: Fill) -> pd.Series:
    """Each event window's simple return, indexed by its anchor session (``close``'s index is the session
    calendar); a window with a missing price is left out."""
    sched = drift_schedule(pd.DatetimeIndex(close.index), anchors, entry, exit)
    prices = {"open": open_.reindex(close.index), "close": close}
    out = {}
    for r in sched.itertuples(index=False):
        buy, sell = float(prices[r.buy_at][r.buy_session]), float(prices[r.sell_at][r.sell_session])
        if np.isfinite(buy) and np.isfinite(sell) and buy > 0:
            out[r.anchor] = sell / buy - 1.0
    return pd.Series(out, dtype="float64", name="window_return")


def _stats(x: pd.Series) -> Dict[str, float]:
    n = int(len(x))
    sd = float(x.std(ddof=1)) if n > 1 else math.nan
    return {"n": n, "mean": float(x.mean()) if n else math.nan,
            "t": float(x.mean() / sd * math.sqrt(n)) if n > 1 and sd > 0 else math.nan,
            "hit_rate": float((x > 0).mean()) if n else math.nan}


def drift_report(open_: pd.Series, close: pd.Series, anchors: Iterable, entry: Fill, exit: Fill,
                 rf_annual: float = 0.065, cost: float = 0.0, split: str = "2020-01-01",
                 post: Optional[Tuple[Fill, Fill]] = None, pure: Optional[Tuple[Fill, Fill]] = None) -> Dict:
    """The window's returns against cash and against the rest of the market.

    ``cost`` is the round trip as a fraction of value; excess is over ``rf_annual`` accrued for the
    calendar days held.  ``share_of_excess`` is the windows' summed log excess return over that of the
    whole of ``close`` (first to last close): the "80%".
    """
    gross = window_returns(open_, close, anchors, entry, exit)
    sched = drift_schedule(pd.DatetimeIndex(close.index), gross.index, entry, exit).set_index("anchor")
    days = (sched["sell_session"] - sched["buy_session"]).dt.days.reindex(gross.index)
    excess = gross - rf_annual * days / 365.0
    net = excess - cost
    span = close.dropna()
    span_days = (span.index[-1] - span.index[0]).days
    total_log_excess = math.log(span.iloc[-1] / span.iloc[0]) - rf_annual * span_days / 365.0
    held_sessions = int(sum(close.index.get_loc(r.sell_session) - close.index.get_loc(r.buy_session)
                            for r in sched.itertuples()))
    cut = pd.Timestamp(split)
    report = {
        "window": {"entry": list(entry), "exit": list(exit)}, "cost_round_trip": cost,
        "excess": _stats(excess), "net": _stats(net),
        "net_before_split": _stats(net[net.index < cut]), "net_from_split": _stats(net[net.index >= cut]),
        "share_of_excess": float(np.log1p(excess).sum() / total_log_excess) if total_log_excess else math.nan,
        "share_of_sessions": held_sessions / max(len(span) - 1, 1),
        "annual_net_contribution": float(net.sum() / (span_days / 365.25)) if span_days else math.nan,
    }
    for name, pair in (("post", post), ("pure", pure)):
        if pair is not None:
            report[name] = _stats(window_returns(open_, close, gross.index, *pair))
    return report


def passes(report: Dict, min_t: float = 2.0) -> bool:
    """Plan 5w's pass rule: net mean > 0 with t >= ``min_t``, and positive in both halves."""
    n, a, b = report["net"], report["net_before_split"], report["net_from_split"]
    return bool(n["mean"] > 0 and n["t"] >= min_t and a["mean"] > 0 and b["mean"] > 0)


def round_trip_cost(value_inr: float, adv_inr: float, when: date, symbol: str = "NIFTYBEES") -> float:
    """A buy and a sell of ``value_inr`` of ``symbol``: statutory charges (ETF STT) plus the impact model,
    as a fraction of value."""
    from nse_engine.config import CostConfig
    from nse_engine.costs import impact_bps, statutory_cost

    cfg = CostConfig()
    stat = sum(statutory_cost(value_inr, side, when, cfg.dp_charge_inr, symbol=symbol) for side in ("BUY", "SELL"))
    return stat / value_inr + 2 * impact_bps(value_inr, adv_inr, cfg) / 1e4


def future_sessions(start: date, days: int = 120) -> pd.DatetimeIndex:
    """NSE trading sessions from ``start`` on, from the shared calendar (holidays and special sessions)."""
    from nse_engine.nse_calendar import is_trading_day

    return pd.DatetimeIndex([start + timedelta(days=i) for i in range(days) if is_trading_day(start + timedelta(days=i))])


def _event_dates(name: str) -> Sequence[str]:
    return globals()[str(WINDOWS[name]["dates"])]


def _load_prices(store_dir: str, start: str, end: str) -> Dict[str, pd.DataFrame]:
    """NIFTY 50's index opens and closes, and NIFTYBEES's (its traded value prices the round trip), from the
    parquet store."""
    import glob
    from pathlib import Path

    def years(kind: str) -> pd.DataFrame:
        files = [f for f in sorted(glob.glob(str(Path(store_dir) / kind / "*.parquet")))
                 if int(start[:4]) - 1 <= int(Path(f).stem) <= int(end[:4]) + 1]
        return pd.concat([pd.read_parquet(f) for f in files])

    idx = years("indices")
    idx = idx[idx["index_name"] == "NIFTY50"]
    eq = years("equity")
    eq = eq[(eq["symbol"] == "NIFTYBEES") & (eq["series"] == "EQ")]
    out = {}
    for name, df in (("NIFTY50", idx), ("NIFTYBEES", eq)):
        df = df.assign(date=pd.to_datetime(df["date"])).set_index("date").sort_index()
        df = df.loc[start:end]
        out[name] = df[["open", "close"] + (["value_inr"] if "value_inr" in df else [])].apply(pd.to_numeric, errors="coerce")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Pre-announcement drift on NSE (plan 5w)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("report", help="the pre-registered test: H1 RBI, H2 FOMC, on NIFTY 50 and NIFTYBEES")
    p.add_argument("--store", default="data/nse_engine/store")
    p.add_argument("--start", default="2013-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--capital", type=float, default=600_000.0, help="round-trip size for the cost model")
    p = sub.add_parser("schedule", help="the orders for upcoming events: AMO evenings, buy and sell sessions")
    p.add_argument("--from", dest="start", default=date.today().isoformat())
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.cmd == "schedule":
        sessions = future_sessions(date.fromisoformat(args.start))
        out = {}
        for name, w in WINDOWS.items():
            sched = drift_schedule(sessions, anchor_sessions(sessions, _event_dates(name), bool(w["after_close"])),
                                   w["entry"], w["exit"])
            out[name] = [{k: (str(v.date()) if isinstance(v, pd.Timestamp) else v) for k, v in r.items()}
                         for r in sched.to_dict(orient="records")]
        print(json.dumps(out, indent=2))
        return 0

    # NIFTYBEES's bhavcopy opens are stray first prints (tracker D4), so the tradable estimate is the index
    # window less a NIFTYBEES round trip at its latest year's liquidity (plan 5w amendment, as O4's Y3).
    px = _load_prices(args.store, args.start, args.end)
    value = px["NIFTYBEES"]["value_inr"]
    adv = float(value[value.index >= value.index[-1] - pd.Timedelta(days=365)].rolling(20).median().median())
    cost = round_trip_cost(args.capital, adv, date.fromisoformat(args.end))
    nifty = px["NIFTY50"]
    out = {"window": [args.start, args.end], "round_trip_cost": cost, "hypotheses": {}}
    for name, w in WINDOWS.items():
        anchors = anchor_sessions(pd.DatetimeIndex(nifty.index), _event_dates(name), bool(w["after_close"]))
        res = {label: drift_report(nifty["open"], nifty["close"], anchors, w["entry"], w["exit"], cost=c,
                                   post=w.get("post"), pure=w.get("pure"))
               for label, c in (("NIFTY50_gross", 0.0), ("NIFTYBEES_cost_net", cost))}
        res["passes"] = passes(res["NIFTYBEES_cost_net"])
        out["hypotheses"][name] = res
    print(json.dumps(out, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
