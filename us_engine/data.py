"""US market data: today's S&P 500 members on yfinance, as a ``MarketData`` panel.

For the advisory book (tracker US-A1) only.  Today's members carry
survivorship bias into any backtest, so never record a US trial on this
loader; US2 builds the clean store.

Prices are split-adjusted but not dividend-adjusted (yfinance's own OHLC), so
the latest close is the price a broker quotes, and a stored stop needs only a
split ratio to stay valid.  ``value`` is traded value in USD.  The sessions
are the dates on which the S&P 500 index printed a close.
"""

from __future__ import annotations

import io
import logging
import time as time_module
from dataclasses import dataclass
from datetime import datetime, time, timezone
from typing import Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from nse_engine.types import MarketData

logger = logging.getLogger(__name__)

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
USER_AGENT = "centurion-core/1.0 (research; https://github.com/srees16)"
#: Panel column -> yfinance ticker of the regime inputs (``regime.index_symbol`` / ``vix_symbol``).
INDEX_TICKERS = {"SPX": "^GSPC", "VIX": "^VIX"}
NEW_YORK = ZoneInfo("America/New_York")
#: yfinance's daily bar is final a little after the 16:00 close.
BAR_FINAL = time(16, 30)
#: First session loaded, as the India deployment's data anchor (2 Jan 2012): the
#: forecast normalisers are expanding, so the anchor must not move between runs.
DATA_ANCHOR = "2012-01-03"
#: Share of members allowed without a close on the last session before the load is refused.
MAX_MISSING_SHARE = 0.05
DOWNLOAD_RETRIES = 2
RETRY_PAUSE_SECONDS = 30
FIELDS = ("Open", "High", "Low", "Close", "Volume", "Stock Splits")


class UsDataError(RuntimeError):
    """The download is too incomplete to plan on."""


@dataclass(frozen=True)
class UsPanel:
    data: MarketData
    #: Split ratio per (session, symbol): 2.0 for a 2-for-1 effective that session, 0 for none.
    splits: pd.DataFrame
    members: int
    #: Members without a close on the last session (left out of today's ranking).
    missing: Tuple[str, ...] = ()


def sp500_members() -> pd.DataFrame:
    """Today's S&P 500 members (``symbol`` in yfinance spelling, GICS ``sector``) from Wikipedia."""
    import requests

    resp = requests.get(SP500_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    table = pd.read_html(io.StringIO(resp.text), match="Symbol")[0]
    out = pd.DataFrame({
        "symbol": table["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False),
        "sector": table["GICS Sector"].astype(str).str.strip(),
    }).drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)
    if len(out) < 450:
        raise UsDataError(f"S&P 500 list has {len(out)} rows: the Wikipedia table changed")
    return out


def last_final_session(now: Optional[datetime] = None) -> pd.Timestamp:
    """Latest calendar date whose daily bar is final (New York time); holidays drop out with the data."""
    ny = (now or datetime.now(timezone.utc)).astimezone(NEW_YORK)
    day = pd.Timestamp(ny.date())
    return day if ny.time() >= BAR_FINAL else day - pd.Timedelta(days=1)


def _fetch(tickers: List[str], start: str, threads: bool) -> Dict[str, pd.DataFrame]:
    import yfinance as yf

    raw = yf.download(tickers, start=start, auto_adjust=False, actions=True, group_by="column",
                      progress=False, threads=threads, multi_level_index=True)
    if raw is None or raw.empty:
        return {}
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None).normalize()
    return {f: raw[f] for f in FIELDS if f in raw.columns.get_level_values(0)}


def _incomplete(close: pd.DataFrame, tickers: List[str]) -> List[str]:
    """Tickers with no close at all, or none on the index's latest session (Yahoo serves some late)."""
    spx = close[INDEX_TICKERS["SPX"]].dropna() if INDEX_TICKERS["SPX"] in close.columns else pd.Series(dtype="float64")
    last = spx.index[-1] if len(spx) else close.index[-1]
    return [t for t in tickers if t not in close.columns or pd.isna(close[t].get(last))]


def _download(tickers: Iterable[str], start: str) -> Dict[str, pd.DataFrame]:
    """{field: sessions x tickers}; tickers that failed or lack the latest session are fetched again."""
    tickers = sorted(set(tickers))
    fields = _fetch(tickers, start, threads=True)
    if "Close" not in fields:
        raise UsDataError("yfinance returned no data")
    for _ in range(DOWNLOAD_RETRIES):
        failed = _incomplete(fields["Close"], tickers)
        if not failed:
            break
        logger.info("fetching %d incomplete tickers again in %ds", len(failed), RETRY_PAUSE_SECONDS)
        time_module.sleep(RETRY_PAUSE_SECONDS)
        again = _fetch(failed, start, threads=False)
        if "Close" not in again:
            continue
        for f in FIELDS:
            base = fields[f].drop(columns=[t for t in failed if t in fields[f].columns])
            fields[f] = pd.concat([base, again[f]], axis=1)
    return fields


def load_us_data(start: str = DATA_ANCHOR, members: Optional[pd.DataFrame] = None,
                 extra: Iterable[str] = (), now: Optional[datetime] = None) -> UsPanel:
    """S&P 500 members plus ``extra`` symbols (held names that left the index), through the last final session."""
    members = sp500_members() if members is None else members
    sectors = dict(zip(members["symbol"], members["sector"]))
    symbols = sorted(set(members["symbol"]) | set(extra))
    raw = _download(symbols + list(INDEX_TICKERS.values()), start)
    end = last_final_session(now)
    spx = raw["Close"].get(INDEX_TICKERS["SPX"], pd.Series(dtype="float64"))
    sessions = spx[spx.index <= end].dropna().index
    if len(sessions) == 0:
        raise UsDataError("no S&P 500 index closes in the download")

    def frame(field: str) -> pd.DataFrame:
        return raw[field].reindex(index=sessions, columns=symbols).astype("float64")

    close, volume = frame("Close"), frame("Volume")
    last = sessions[-1]
    missing = tuple(s for s in members["symbol"] if not pd.notna(close.at[last, s]))
    if len(missing) > MAX_MISSING_SHARE * len(members):
        raise UsDataError(f"{len(missing)} of {len(members)} members have no close on {last.date()}")
    if missing:
        logger.warning("%d members without a close on %s: %s", len(missing), last.date(), ", ".join(missing))
    data = MarketData(
        dates=pd.DatetimeIndex(sessions),
        open=frame("Open"), high=frame("High"), low=frame("Low"), close=close,
        volume=volume, value=close * volume,
        index_close=pd.DataFrame({name: raw["Close"][t].reindex(sessions).astype("float64")
                                  for name, t in INDEX_TICKERS.items()}),
        sectors={s: sectors[s] for s in symbols if s in sectors},
        source="yfinance:sp500-today",
    )
    data.validate()
    data.data_hash = data.compute_hash()
    splits = frame("Stock Splits").fillna(0.0)
    return UsPanel(data=data, splits=splits, members=len(members), missing=missing)
