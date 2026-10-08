"""
Configuration for the Varsity options toolkit (docs/options/CONCEPTS.md, STRATEGIES.md).

Every rate, threshold and limit the toolkit uses lives here as a frozen
dataclass, as in ``nse_engine.config``.  Market facts that change (lot sizes,
expiries, margins, freeze limits) are never configured: they come from the
Kite instruments dump and margins APIs at runtime.

``load_config(path)`` overrides the defaults from a TOML file whose tables
are named after the fields of :class:`OptionsConfig` (``[market]``,
``[charges]``, ``[slippage]``, ``[limits]``, ``[selector]``, ``[data]``).  Without a path
it reads ``CENTURION_OPTIONS_CONFIG`` when set.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Dict, Optional, Tuple

#: A date-aware rate: ((effective_from ISO date, rate), ...) in ascending order.
#: The first entry's date is the earliest the schedule claims to cover.
Schedule = Tuple[Tuple[str, float], ...]


@dataclass(frozen=True)
class MarketConfig:
    # 91-day T-bill cut-off yield, continuously compounded input to Black-Scholes
    # (M5 ch. 21).  RBI auction of 30 Sep 2026: 5.52%.  Update with its date.
    risk_free_rate: float = 0.0552
    risk_free_as_of: str = "2026-09-30"
    # sigma_annual = sigma_daily * sqrt(N); M5 ch. 16/18 use 365, ch. 17 uses 252.
    vol_days_per_year: int = 365
    # Black-Scholes time to expiry = calendar days / this (reproduces M5 ch. 21, M6 ch. 7).
    bs_days_per_year: int = 365
    # Strikes from ATM at which ITM/OTM becomes "deep" (the PDF gives no number).
    deep_strikes: int = 3


@dataclass(frozen=True)
class ChargesConfig:
    """F&O statutory charges by date (Zerodha's schedule for brokerage).

    STT, exchange and stamp rates are on premium for options and on notional
    for futures.  The SEBI fee and GST/service tax are shared with the equity
    model (``nse_engine.costs``).  The schedules start with STT on derivatives
    (1 Oct 2004: 0.01%, then 0.0133% from Jun 2005 and 0.017% from Jun 2006,
    Finance Acts 2004-06); the exercise levy exists only from 1 Jun 2008.
    Exchange charges before 1 Oct 2024 moved between 0.0495% and 0.053% of
    premium; 0.05% is used throughout.  Stamp duty before 1 Jul 2020 varied
    by state; the uniform rates are used.
    """

    brokerage_per_order_inr: float = 20.0
    # Futures: the lower of the flat fee and this share of the order's notional.
    brokerage_futures_pct: float = 0.0003
    stt_option_sell: Schedule = (
        ("2004-10-01", 0.0001), ("2005-06-01", 0.000133), ("2006-06-01", 0.00017), ("2016-06-01", 0.0005),
        ("2023-04-01", 0.000625), ("2024-10-01", 0.001), ("2026-04-01", 0.0015),
    )
    # Exercised (or ITM-at-expiry) long options, paid by the buyer.  Before
    # 1 Sep 2019 the base was the full settlement value ("the STT trap", M5);
    # from then on it is the intrinsic value.
    stt_option_exercise: Schedule = (("2004-10-01", 0.0), ("2008-06-01", 0.00125), ("2026-04-01", 0.0015))
    stt_exercise_intrinsic_from: str = "2019-09-01"
    stt_futures_sell: Schedule = (
        ("2004-10-01", 0.0001), ("2005-06-01", 0.000133), ("2006-06-01", 0.00017), ("2013-06-01", 0.0001),
        ("2023-04-01", 0.000125), ("2024-10-01", 0.0002), ("2026-04-01", 0.0005),
    )
    exchange_option: Schedule = (("2004-10-01", 0.0005), ("2024-10-01", 0.0003503))
    exchange_futures: Schedule = (("2004-10-01", 0.000019), ("2024-10-01", 0.0000173))
    stamp_option_buy: Schedule = (("2004-10-01", 0.00003),)
    stamp_futures_buy: Schedule = (("2004-10-01", 0.00002),)


@dataclass(frozen=True)
class SlippageConfig:
    """Adverse fill per unit, per side: max(min_ticks * tick, pct * premium).

    Defaults are placeholders until paper fills measure the real spread
    (tracker X1 method); index options are far tighter than stock options.
    """

    tick_size: float = 0.05
    min_ticks: int = 1
    index_pct: float = 0.005
    stock_pct: float = 0.02
    index_underlyings: Tuple[str, ...] = ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX", "BANKEX")


@dataclass(frozen=True)
class LimitsConfig:
    """Hard limits; a basket that breaches any of them is rejected."""

    max_loss_per_trade_inr: float = 25_000.0
    max_lots_per_leg: int = 10
    allowed_underlyings: Tuple[str, ...] = ("NIFTY", "BANKNIFTY")
    # LIMIT price = reference price +/- this fraction (buy above, sell below), at least one tick.
    limit_slippage_cap: float = 0.02
    # Kite allows 10 orders/s; stay below it.
    max_orders_per_second: int = 8
    fill_timeout_seconds: float = 30.0


@dataclass(frozen=True)
class SelectorConfig:
    # "1st half" of a series = more than this many days to expiry (CONCEPTS open decision 5).
    first_half_min_dte: int = 16
    # M5 ch. 18 writing rules: never write beyond 15 DTE; 1 SD at <= 4 DTE, else 2 SD.
    writing_max_dte: int = 15
    writing_one_sd_max_dte: int = 4
    # M6 ch. 13: PCR > high -> expect a rise; < low -> expect a fall.
    pcr_high: float = 1.3
    pcr_low: float = 0.5
    # M6 ch. 13 modified max pain: computed at this DTE, with this upward buffer.
    max_pain_dte: int = 15
    max_pain_buffer: float = 0.05
    # M6 ch. 2 "moderate" move ceilings.
    moderate_move_index: float = 0.05
    moderate_move_stock: float = 0.08
    # M6 ch. 4: avoid ratio back spreads early in a series when IV > this x normal.
    high_iv_multiple: float = 2.0
    # M5 ch. 13 (gamma): no short ATM options at or below this DTE.
    short_atm_min_dte: int = 3
    # The PDF's moneyness labels in listed strikes from ATM (+ = OTM).  It never
    # fixes them: M5 ch. 22 calls 2 strikes "far OTM"; M6 ch. 2 uses 300-point
    # steps at Nifty 8000.
    strike_offsets: Tuple[Tuple[str, int], ...] = (
        ("deep ITM", -3), ("ITM", -2), ("slightly ITM", -1), ("ATM", 0),
        ("slightly OTM", 1), ("OTM", 2), ("far OTM", 3),
    )
    # Width of two-strike spreads in listed strikes (M6 ch. 2: 300 points at 100-point strikes).
    spread_strikes: int = 3


@dataclass(frozen=True)
class DataConfig:
    """The market context from history (trackers OD1, OD2): IV history and positioning."""

    fo_store: str = "data/nse_engine/fo_store"
    equity_store: str = "data/nse_engine/store"
    # IV is quoted at a constant 30 calendar days, interpolated between the two expiries
    # around it; an expiry closer than min_dte is skipped (expiry-week noise).
    iv_tenor_days: int = 30
    iv_min_dte: int = 7
    # ATM strikes searched within this fraction of spot; the IV needs a traded call or put there.
    iv_strike_band: float = 0.10
    # IV rank / percentile and positioning percentiles look back this many sessions (a year).
    lookback_sessions: int = 252
    # The realised-volatility cone the IV level reads (M5 ch. 20): a 21-session window
    # (about 30 calendar days, like the IV) over the last two years.
    cone_window: int = 21
    cone_lookback_sessions: int = 504
    # Realised vs implied is reported for the IV-percentile bucket of this width.
    premium_bucket_width: float = 20.0


@dataclass(frozen=True)
class OptionsConfig:
    market: MarketConfig = field(default_factory=MarketConfig)
    charges: ChargesConfig = field(default_factory=ChargesConfig)
    slippage: SlippageConfig = field(default_factory=SlippageConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    selector: SelectorConfig = field(default_factory=SelectorConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _coerce(value: Any) -> Any:
    """TOML arrays -> tuples, recursively (schedules and name lists are tuples)."""
    if isinstance(value, list):
        return tuple(_coerce(v) for v in value)
    return value


def load_config(path: Optional[str] = None) -> OptionsConfig:
    """Defaults, overridden by the TOML file at ``path`` or ``$CENTURION_OPTIONS_CONFIG``."""
    path = path or os.environ.get("CENTURION_OPTIONS_CONFIG")
    cfg = OptionsConfig()
    if not path:
        return cfg
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    groups = {f.name for f in fields(OptionsConfig)}
    unknown = set(data) - groups
    if unknown:
        raise ValueError(f"unknown config tables: {sorted(unknown)}")
    updates = {}
    for name, values in data.items():
        group = getattr(cfg, name)
        known = {f.name for f in fields(group)}
        bad = set(values) - known
        if bad:
            raise ValueError(f"unknown keys in [{name}]: {sorted(bad)}")
        updates[name] = replace(group, **{k: _coerce(v) for k, v in values.items()})
    return replace(cfg, **updates)
