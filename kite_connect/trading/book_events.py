"""Corporate actions and dividends in the paper and live books (tracker LN-T4).

The backtest trades on back-adjusted prices, so a split, bonus or dividend
is invisible to it.  The paper and live books hold real quantities, stops
and cash: on an ex-date their positions must move into the new units or a
1:2 split reads as a 50% gap stop and a fake loss (GOLDBEES's 1:100 split
in Dec 2019 as a -34.7% day).  The store's own adjustments decide, as
``MarketData.corporate_events`` and ``dividend_events``:

* share counts change only for NSE's split, bonus and consolidation ratios
  (panel sources 1 and 6) and for inferred, snapped gaps (source 4, with an
  alert); rights, demergers, PREVCLOSE and unexplained factors change no
  quantity and raise an alert;
* prices and stops move by every adjustment since the last processed
  session: ``close / close_unadj`` there, in tonight's data;
* a dividend pays quantity x the cash per share on its ex-date.

The arithmetic follows Lean's ``SecurityPortfolioManager.ApplySplit``:
quantity / factor rounded down, the fraction paid in cash at the close.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

logger = logging.getLogger(__name__)

SHARE_SOURCES = (1, 4, 6)       # panel.SOURCE_NAMES: ca_ratio, inferred_snapped, ca_ratio_from_prices
INFERRED_SOURCE = 4
#: An as-printed close this far below the last one, on a held name with no store event, may be a
#: split the store does not know yet: buys of it are held back and an alert asks for a check.
UNRECORDED_GAP = 0.9


def event_key(symbol: str, day, value: float) -> str:
    return f"{symbol}:{pd.Timestamp(day).date().isoformat()}:{value:.6g}"


def events_between(data, symbols: Iterable[str], after, upto) -> Dict[str, dict]:
    """Per symbol, the adjustments dated after ``after`` up to ``upto`` (both sessions).

    ``{symbol: {"share_factor", "share_keys", "inferred", "other", "dividends": [(date, per_share, key)]}}``
    for symbols with any; ``share_factor`` is the product of the share-count factors.
    """
    syms = set(symbols)
    lo, hi = pd.Timestamp(after), pd.Timestamp(upto)
    out: Dict[str, dict] = {}

    def entry(sym):
        return out.setdefault(sym, {"share_factor": 1.0, "share_keys": [], "inferred": [], "other": [],
                                    "dividends": []})

    ce = getattr(data, "corporate_events", None)
    if ce is not None and len(ce):
        rows = ce[(ce["date"] > lo) & (ce["date"] <= hi) & ce["symbol"].isin(syms)]
        for r in rows.itertuples(index=False):
            e = entry(r.symbol)
            note = f"{pd.Timestamp(r.date).date()} factor {r.factor:.4g}"
            if int(r.source) in SHARE_SOURCES:
                e["share_factor"] *= float(r.factor)
                e["share_keys"].append(event_key(r.symbol, r.date, r.factor))
                if int(r.source) == INFERRED_SOURCE:
                    e["inferred"].append(note)
            else:
                e["other"].append(note)
    de = getattr(data, "dividend_events", None)
    if de is not None and len(de):
        rows = de[(de["date"] > lo) & (de["date"] <= hi) & de["symbol"].isin(syms)]
        for r in rows.itertuples(index=False):
            entry(r.symbol)["dividends"].append((pd.Timestamp(r.date), float(r.dividend),
                                                 "div:" + event_key(r.symbol, r.date, r.dividend)))
    return out


def price_factor(data, symbol: str, after, upto=None) -> float:
    """Every adjustment after session ``after`` up to ``upto`` (default: the data's end), 1.0 when unknown.

    ``close / close_unadj`` at a date is the product of the adjustments after
    it, so the factor for (after, upto] is that ratio at ``after`` over the
    ratio at ``upto``.
    """
    close, raw = getattr(data, "close", None), getattr(data, "close_unadj", None)
    if close is None or raw is None or symbol not in close.columns or symbol not in raw.columns:
        return 1.0

    def ratio(day):
        c, u = close[symbol].loc[:pd.Timestamp(day)].dropna(), raw[symbol].loc[:pd.Timestamp(day)].dropna()
        if c.empty or u.empty or c.index[-1] != u.index[-1] or float(u.iloc[-1]) <= 0:
            return None
        return float(c.iloc[-1]) / float(u.iloc[-1])

    lo = ratio(after)
    hi = ratio(upto) if upto is not None else 1.0
    if lo is None or not hi:
        return 1.0
    f = lo / hi
    return f if math.isfinite(f) and f > 0 else 1.0


def unrecorded_gaps(data, symbols: Iterable[str], after, upto, events: Dict[str, dict]) -> List[str]:
    """Held names whose as-printed close fell below ``UNRECORDED_GAP`` x the last one with no store event."""
    raw = getattr(data, "close_unadj", None)
    if raw is None:
        return []
    out = []
    for sym in symbols:
        if sym in events or sym not in raw.columns:
            continue
        s = raw[sym].loc[:pd.Timestamp(upto)].dropna()
        before, now = s.loc[:pd.Timestamp(after)], s.loc[pd.Timestamp(after):]
        if before.empty or len(now) < 2 or float(before.iloc[-1]) <= 0:
            continue
        r = float(now.iloc[-1]) / float(before.iloc[-1])
        if r < UNRECORDED_GAP:
            out.append(f"{sym} {float(before.iloc[-1]):,.2f} -> {float(now.iloc[-1]):,.2f}")
    return out


def upcoming_share_actions(store_dir, symbols: Iterable[str], session) -> Dict[str, str]:
    """Symbols whose split, bonus or other NSE action goes ex on the next weekday session (by the store)."""
    path = Path(store_dir) / "corporate_actions.parquet"
    if not path.exists():
        return {}
    try:
        ev = pd.read_parquet(path, columns=["symbol", "ex_date", "purpose"])
    except Exception as exc:                              # noqa: BLE001 - an unreadable file defers nothing
        logger.warning("corporate_actions.parquet unreadable (%s): no ex-date deferral tonight", exc)
        return {}
    s = pd.Timestamp(session).normalize()
    nxt = s + pd.offsets.BDay(1)
    ev = ev[(pd.to_datetime(ev["ex_date"]) > s) & (pd.to_datetime(ev["ex_date"]) <= nxt) & ev["symbol"].isin(set(symbols))]
    return {r.symbol: f"{r.purpose} ex {pd.Timestamp(r.ex_date).date()}" for r in ev.itertuples(index=False)}


def rebase_quantity(quantity: int, share_factor: float) -> tuple:
    """(new quantity, fraction of a share paid in cash): quantity / factor, rounded down."""
    exact = int(quantity) / float(share_factor)
    q = int(math.floor(exact + 1e-9))
    return q, max(exact - q, 0.0)


# ── renames and mergers (tracker LN-T5) ─────────────────────────

def renamed(store_dir, symbols: Iterable[str], since, until) -> Dict[str, str]:
    """{old: new} for held ``symbols`` NSE renamed after session ``since`` and in force by ``until``
    (the store's change table; NSE lists some renames before they take effect)."""
    syms = sorted(set(symbols))
    if not syms or since is None:
        return {}
    try:
        from nse_engine.data import reference
        from nse_engine.data.panel import load_change_table

        changes = load_change_table(store_dir)
        changes = changes[pd.to_datetime(changes["date"]) <= pd.Timestamp(until)]
        when = pd.Series(pd.DatetimeIndex([pd.Timestamp(since)] * len(syms)).astype("datetime64[ns]"))
        now = reference.resolve_symbols(pd.Series(syms), when, changes)
    except Exception as exc:                              # noqa: BLE001 - no table, no rename tonight
        logger.warning("symbol change table unavailable (%s): renames not checked", exc)
        return {}
    return {old: str(new) for old, new in zip(syms, now) if str(new) != old}


def upcoming_mergers(store_dir, symbols: Iterable[str], session, days: int = 5) -> Dict[str, str]:
    """Held symbols with an announced merger/amalgamation, delisting or exit offer going ex within ``days``
    weekday sessions (or already ex), from the store's raw NSE corporate-action files: {symbol: what}."""
    syms = set(symbols)
    files = sorted(Path(store_dir).glob("corpact/*.parquet"))[-2:]
    if not syms or not files:
        return {}
    try:
        ev = pd.concat(pd.read_parquet(f, columns=["symbol", "ex_date", "purpose", "file_date"]) for f in files)
    except Exception as exc:                              # noqa: BLE001
        logger.warning("corporate-action archive unreadable (%s): mergers not checked", exc)
        return {}
    s = pd.Timestamp(session).normalize()
    purpose = ev["purpose"].astype(str).str.upper()
    merger = (purpose.str.contains("AMALG") | (purpose.str.contains("MERGER") & ~purpose.str.contains("DEMERGER"))
              | purpose.str.contains("DELIST") | purpose.str.contains("EXIT OFFER"))
    ev = ev[merger & ev["symbol"].isin(syms) & (pd.to_datetime(ev["file_date"]) <= s)
            & (pd.to_datetime(ev["ex_date"]) <= s + pd.offsets.BDay(days))
            & (pd.to_datetime(ev["ex_date"]) > s - pd.offsets.BDay(30))]
    return {r.symbol: f"{r.purpose} ex {pd.Timestamp(r.ex_date).date()}"
            for r in ev.drop_duplicates(["symbol", "ex_date"]).itertuples(index=False)}
