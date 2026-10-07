"""
Instrument resolver: underlying + expiry + strike + CE/PE -> the NFO contract.

Built on Kite's instruments dump (``Broker.instruments``), so lot sizes,
expiries, strike steps and tick sizes are always today's (CONCEPTS.md § Out
of date: none of them is hard-coded).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Optional

import pandas as pd

#: Kite quote keys of the index underlyings; stock options use NSE:<symbol>.
INDEX_SPOT_KEYS = {"NIFTY": "NSE:NIFTY 50", "BANKNIFTY": "NSE:NIFTY BANK",
                   "FINNIFTY": "NSE:NIFTY FIN SERVICE", "MIDCPNIFTY": "NSE:NIFTY MID SELECT"}


def spot_key(underlying: str) -> str:
    """The quote key of the underlying's spot price."""
    return INDEX_SPOT_KEYS.get(underlying.upper(), f"NSE:{underlying.upper()}")


@dataclass(frozen=True)
class Contract:
    tradingsymbol: str
    underlying: str
    expiry: date
    strike: float
    option_type: str
    lot_size: int
    tick_size: float
    instrument_token: int
    exchange: str = "NFO"

    @property
    def quote_key(self) -> str:
        return f"{self.exchange}:{self.tradingsymbol}"

    @property
    def is_index(self) -> bool:
        return self.underlying in INDEX_SPOT_KEYS


class InstrumentResolver:
    """Options of one exchange's instruments dump, indexed by underlying."""

    def __init__(self, instruments: pd.DataFrame):
        df = instruments[instruments["instrument_type"].isin(["CE", "PE"])].copy()
        df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
        df["strike"] = df["strike"].astype(float)
        self._df = df

    def _of(self, underlying: str) -> pd.DataFrame:
        sub = self._df[self._df["name"] == underlying.upper()]
        if sub.empty:
            raise KeyError(f"no options for {underlying!r} in the instruments dump")
        return sub

    def expiries(self, underlying: str) -> List[date]:
        return sorted(self._of(underlying)["expiry"].unique())

    def nearest_expiry(self, underlying: str, today: date, min_days: int = 0) -> date:
        """The first expiry at least ``min_days`` calendar days after ``today``."""
        for e in self.expiries(underlying):
            if (e - today).days >= min_days:
                return e
        raise KeyError(f"no {underlying} expiry {min_days}+ days after {today}")

    def strikes(self, underlying: str, expiry: date) -> List[float]:
        sub = self._of(underlying)
        return sorted(float(k) for k in sub.loc[sub["expiry"] == expiry, "strike"].unique())

    def resolve(self, underlying: str, expiry: date, strike: float, option_type: str) -> Contract:
        sub = self._of(underlying)
        row = sub[(sub["expiry"] == expiry) & (sub["strike"] == float(strike))
                  & (sub["instrument_type"] == option_type.upper())]
        if row.empty:
            raise KeyError(f"{underlying} {expiry} {strike:g} {option_type} is not listed")
        return _contract(row.iloc[0], underlying)

    def chain(self, underlying: str, expiry: date, strikes: Optional[List[float]] = None) -> List[Contract]:
        """Every listed CE and PE of the expiry (or only ``strikes``)."""
        sub = self._of(underlying)
        sub = sub[sub["expiry"] == expiry]
        if strikes is not None:
            sub = sub[sub["strike"].isin([float(k) for k in strikes])]
        return [_contract(r, underlying) for _, r in sub.sort_values(["strike", "instrument_type"]).iterrows()]


def _contract(row: pd.Series, underlying: str) -> Contract:
    return Contract(tradingsymbol=str(row["tradingsymbol"]), underlying=underlying.upper(), expiry=row["expiry"],
                    strike=float(row["strike"]), option_type=str(row["instrument_type"]),
                    lot_size=int(row["lot_size"]), tick_size=float(row["tick_size"]),
                    instrument_token=int(row["instrument_token"]), exchange=str(row.get("exchange", "NFO")))
