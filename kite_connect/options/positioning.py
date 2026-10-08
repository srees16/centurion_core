"""
FII / DII positioning (tracker OD2) from NSE's daily participant-wise open
interest in equity derivatives (``fo_store/participant_oi.parquet``, from
January 2012): who holds the market's open futures and options, as long
shares and net contracts, each with its percentile over the past year.

It says who is positioned how, not where the market goes next: it is context
for the trader's own view, and the selector never scores it.  Participants:
FII (foreign portfolio investors), DII (domestic institutions), Pro
(brokers' own books) and Client (everyone else, mostly retail).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Tuple

import pandas as pd

from kite_connect.options.iv_history import mid_rank

#: (label, participant, long column, short column, "share" = long % of the two, "net" = long - short)
MEASURES: Tuple[Tuple[str, str, str, str, str], ...] = (
    ("FII index futures", "FII", "future_index_long", "future_index_short", "share"),
    ("FII stock futures", "FII", "future_stock_long", "future_stock_short", "share"),
    ("FII index calls", "FII", "option_index_call_long", "option_index_call_short", "net"),
    ("FII index puts", "FII", "option_index_put_long", "option_index_put_short", "net"),
    ("Client index futures", "Client", "future_index_long", "future_index_short", "share"),
)


@dataclass
class Reading:
    label: str
    kind: str                # "share" (% long) or "net" (contracts long - short)
    value: float
    change: float            # since the previous session
    percentile: float        # among the previous ``lookback`` sessions, 0-100 (ties count half)

    def text(self) -> str:
        if self.kind == "share":
            body = f"{self.value:.0f}% long ({self.change:+.1f} pts on the day)"
        else:
            body = f"net {self.value / 1e3:+,.0f}k contracts ({self.change / 1e3:+,.0f}k on the day)"
        flag = "; near the year's extreme" if self.percentile <= 10 or self.percentile >= 90 else ""
        return f"{self.label}: {body}, 1-year percentile {self.percentile:.0f}{flag}"


@dataclass
class Positioning:
    as_of: date
    readings: List[Reading]

    def lines(self) -> List[str]:
        return [f"Positioning {self.as_of} (NSE participant-wise open interest; context, not a signal):"] + \
               [f"  {r.text()}" for r in self.readings]


def measure_series(df: pd.DataFrame, participant: str, long_col: str, short_col: str, kind: str) -> pd.Series:
    """One measure per session for one participant: % long, or long - short contracts."""
    rows = df[df["client_type"] == participant].set_index("date").sort_index()
    long_, short = rows[long_col].astype(float), rows[short_col].astype(float)
    if kind == "share":
        return (long_ / (long_ + short) * 100).where(long_ + short > 0)
    return long_ - short


def positioning(df: pd.DataFrame, lookback: int = 252) -> Optional[Positioning]:
    """The latest session's readings with their change and 1-year percentile; None without data."""
    if df is None or df.empty:
        return None
    readings = []
    for label, who, long_col, short_col, kind in MEASURES:
        s = measure_series(df, who, long_col, short_col, kind).dropna()
        if len(s) < 2:
            continue
        past = s.iloc[:-1].to_numpy()[-lookback:]
        readings.append(Reading(label, kind, float(s.iloc[-1]), float(s.iloc[-1] - s.iloc[-2]),
                                mid_rank(past, float(s.iloc[-1]))))
    as_of = pd.Timestamp(df["date"].max()).date()
    return Positioning(as_of, readings) if readings else None
