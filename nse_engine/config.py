"""
Configuration for the NSE long-only engine.

Every tunable lives here as a frozen dataclass so that a run is fully
described by one JSON-serialisable object.  ``EngineConfig.config_hash()``
identifies a configuration in the trial registry (PBO / DSR need every
configuration that was ever evaluated).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, Tuple

#: (group, field) -> value that reproduces the behaviour from before the field
#: existed. Left out of ``config_hash`` at that value; see ``EngineConfig.config_hash``.
HASH_NEUTRAL_DEFAULTS: Dict[Tuple[str, str], Any] = {
    ("signals", "normalizer_window_days"): 0,
    ("signals", "ewm_memory_spans"): 0,
    ("signals", "calendar_schedule"): False,
    ("signals", "delivery_lookback"): 63,
    ("signals", "residual_momentum_months"): 36,
    ("signals", "residual_momentum_window"): 11,
    ("signals", "high_lookback"): 252,
    ("signals", "beta_lookback"): 252,
    ("portfolio", "calendar_schedule"): False,
    ("universe", "calendar_schedule"): False,
    ("universe", "history_window_days"): 0,
    ("allocator", "target_vol_annual"): 0.0,
    ("allocator", "vol_target_min_scale"): 0.3,
}


@dataclass(frozen=True)
class DataConfig:
    """Where market data comes from and how it is filtered at load time."""

    store_dir: str = "data/nse_engine/store"
    series: Tuple[str, ...] = ("EQ", "BE")
    # Pre-filter: drop symbols that never reached this 126-day median traded
    # value.  Uses the whole sample, so it only trims memory; the point-in-time
    # universe filter (UniverseConfig) is what decides tradability.
    load_min_median_value_inr: float = 2.5e6
    float_dtype: str = "float32"
    # Back-adjust prices for cash dividends (total-return prices).  False gives
    # price-only series, e.g. for comparisons with price indices.
    adjust_dividends: bool = True


@dataclass(frozen=True)
class UniverseConfig:
    """Point-in-time liquidity universe (survivorship-free by construction)."""

    top_n_liquid: int = 300
    min_history_days: int = 252
    min_price_inr: float = 20.0
    min_median_value_inr: float = 1.0e7  # Rs 1 crore/day median traded value
    liquidity_lookback_days: int = 126
    refresh_every_n_days: int = 21
    exclude_etfs: bool = True
    # Anchor independence (legacy values keep the legacy behaviour and hash):
    calendar_schedule: bool = False   # refresh on calendar period starts, not row counts
    history_window_days: int = 0      # 0: count history since the first loaded row; >0: within this trailing window


@dataclass(frozen=True)
class SignalConfig:
    """Three forecast groups with fixed, hand-set weights (no optimisation)."""

    group_weights: Tuple[Tuple[str, float], ...] = (
        ("fast_trend", 1.0 / 3.0),
        ("slow_trend", 1.0 / 3.0),
        ("low_vol", 1.0 / 3.0),
    )
    fast_ewmac: Tuple[Tuple[int, int], ...] = ((8, 32), (16, 64))
    slow_ewmac: Tuple[Tuple[int, int], ...] = ((64, 256),)
    momentum_lookback: int = 252
    momentum_skip: int = 21
    low_vol_lookback: int = 252
    vol_span: int = 35
    forecast_cap: float = 20.0
    target_abs_forecast: float = 10.0
    normalizer_min_obs: int = 60
    fdm_cap: float = 2.0
    fdm_lookback_days: int = 504
    fdm_refresh_every_n_days: int = 21
    # Anchor independence (legacy values keep the legacy behaviour and hash):
    normalizer_window_days: int = 0   # 0: expanding since the first loaded row; >0: rolling window of dates
    ewm_memory_spans: int = 0         # 0: pandas EWM (infinite memory); k: kernel truncated at k x span rows
    calendar_schedule: bool = False   # FDM refresh on calendar period starts, not row counts
    # "delivery" group: cross-sectional rank of trailing mean delivery % (share of traded
    # quantity taken to demat, from NSE's MTO files). Only used when group_weights names it.
    delivery_lookback: int = 63
    # Price-based groups screened from awesome-systematic-trading (17 Sep 2026); each is
    # used only when group_weights names it.
    # "residual_momentum": market-model residuals over the last `window` months before the
    # latest one, scaled by their std; the model is fitted on `months` monthly returns.
    residual_momentum_months: int = 36
    residual_momentum_window: int = 11
    # "near_high": close / highest close over `high_lookback` sessions (52-week-high effect).
    high_lookback: int = 252
    # "low_beta": minus the beta of daily returns to the equal-weight universe.
    beta_lookback: int = 252

    def weights(self) -> Dict[str, float]:
        return dict(self.group_weights)

    def max_span(self) -> int:
        return max([s for _, s in self.fast_ewmac] + [s for _, s in self.slow_ewmac] + [self.vol_span])


@dataclass(frozen=True)
class PortfolioConfig:
    """Core long-only stock book (CNC, gross <= 1)."""

    target_positions: int = 20
    max_positions: int = 30
    exit_rank: int = 40
    max_weight: float = 0.08
    rebalance_every_n_days: int = 5
    no_trade_buffer: float = 0.25
    min_trade_value_inr: float = 5_000.0
    sector_cap: float = 0.25
    stop_atr_multiple: float = 3.0
    atr_lookback: int = 20
    stop_cooldown_days: int = 5
    vol_lookback_days: int = 60
    # Anchor independence (legacy value keeps the legacy behaviour and hash):
    calendar_schedule: bool = False   # rebalance on calendar period starts, not row counts


@dataclass(frozen=True)
class RegimeConfig:
    """Market regime from NIFTY trend, breadth and India VIX (not own equity)."""

    index_symbol: str = "NIFTY50"
    trend_ma_days: int = 200
    breadth_ma_days: int = 200
    breadth_risk_on: float = 0.50
    breadth_risk_off: float = 0.35
    vix_symbol: str = "INDIAVIX"
    vix_elevated: float = 25.0
    vix_extreme: float = 35.0
    confirm_days: int = 3
    scale_risk_on: float = 1.0
    scale_neutral: float = 0.6
    scale_risk_off: float = 0.0


@dataclass(frozen=True)
class SleeveConfig:
    """Uncorrelated long-only sleeves: gold and silver ETF trend."""

    gold_symbol: str = "GOLDBEES"
    silver_symbol: str = "SILVERBEES"
    gold_enabled: bool = True
    silver_enabled: bool = True
    trend_ma_days: int = 200
    min_history_days: int = 252


@dataclass(frozen=True)
class AllocatorConfig:
    """Risk budget between the core book and the metal sleeves."""

    core_risk_share: float = 0.65
    core_risk_share_min: float = 0.60
    core_risk_share_max: float = 0.70
    max_gross: float = 1.0
    vol_lookback_days: int = 60
    risk_off_to_metals: bool = True
    # Portfolio volatility target (Harvey et al. 2018; Moreira & Muir 2017): scale the
    # whole book DOWN when its estimated vol exceeds the target. Never up — a CNC
    # book cannot exceed max_gross. 0 = off (legacy behaviour and hash).
    target_vol_annual: float = 0.0
    vol_target_min_scale: float = 0.3   # floor on the scale-down, so a vol spike never empties the book


@dataclass(frozen=True)
class CostConfig:
    """Execution costs: statutory schedule lives in nse_engine.costs."""

    max_participation: float = 0.05
    spread_floor_bps: float = 3.0
    impact_coefficient_bps: float = 25.0
    adv_lookback_days: int = 20
    dp_charge_inr: float = 15.93


@dataclass(frozen=True)
class EngineConfig:
    start: str = "2013-01-01"
    end: str = "2025-12-31"
    initial_capital: float = 500_000.0
    cash_yield_annual: float = 0.06
    risk_free_annual: float = 0.065
    runs_dir: str = "data/nse_engine/runs"
    data: DataConfig = field(default_factory=DataConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    sleeves: SleeveConfig = field(default_factory=SleeveConfig)
    allocator: AllocatorConfig = field(default_factory=AllocatorConfig)
    costs: CostConfig = field(default_factory=CostConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    def config_hash(self) -> str:
        """Hash of everything that affects results (dates and paths excluded).

        Fields added after the registry existed are left out of the hash while
        they hold their legacy value (``HASH_NEUTRAL_DEFAULTS``), so a
        configuration evaluated before the field existed keeps its hash: the
        deployed ``679cbd0c`` stays ``679cbd0c`` and PBO/DSR keep counting it as
        one configuration. A non-legacy value changes results and the hash.
        """
        d = self.to_dict()
        for k in ("start", "end", "runs_dir"):
            d.pop(k, None)
        d.get("data", {}).pop("store_dir", None)
        for (group, key), legacy in HASH_NEUTRAL_DEFAULTS.items():
            if d.get(group, {}).get(key, legacy) == legacy:
                d[group].pop(key, None)
        blob = json.dumps(d, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def anchor_independent(self) -> bool:
        """True when every schedule and filter is calendar-anchored / finite-memory."""
        return bool(
            self.signals.normalizer_window_days > 0
            and self.signals.ewm_memory_spans > 0
            and self.signals.calendar_schedule
            and self.portfolio.calendar_schedule
            and self.universe.calendar_schedule
            and self.universe.history_window_days > 0
        )

    def required_warmup_days(self) -> int:
        """Rows of data needed before ``start`` for results not to depend on the load start.

        Only meaningful for an anchor-independent configuration; with legacy
        expanding normalisers no warm-up length is enough.
        """
        sig = self.signals
        # The chain behind one combined forecast: the raw rule needs its EWM
        # memory (or the momentum lookback); the rule normaliser pools raw values
        # over the previous window, each needing that memory; the group-level
        # normaliser pools again; the FDM pools normalised group forecasts over
        # its lookback. Stages add, they do not overlap.
        active = {g for g, w in sig.group_weights if w}
        # New groups count only when used, so existing configurations keep their warm-up.
        extra = [lookback for group, lookback in (
            ("residual_momentum", (sig.residual_momentum_months + 1) * 23),
            ("near_high", sig.high_lookback),
            ("low_beta", sig.beta_lookback)) if group in active]
        rule_memory = max(sig.ewm_memory_spans * sig.max_span(),
                          sig.momentum_lookback + sig.momentum_skip, sig.low_vol_lookback,
                          sig.delivery_lookback, *extra)
        forecast_chain = (rule_memory + 2 * sig.normalizer_window_days + sig.fdm_lookback_days
                          + sig.normalizer_min_obs)
        return max(
            forecast_chain,
            self.universe.history_window_days or self.universe.min_history_days,
            self.universe.liquidity_lookback_days,
            self.regime.trend_ma_days, self.regime.breadth_ma_days,
            self.sleeves.trend_ma_days, self.sleeves.min_history_days,
            self.costs.adv_lookback_days, self.portfolio.vol_lookback_days,
        )

    def replace(self, **changes: Any) -> "EngineConfig":
        """Return a copy with top-level or dotted-path overrides.

        >>> EngineConfig().replace(**{"portfolio.target_positions": 25})
        """
        return _replace_path(self, changes)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EngineConfig":
        return _from_dict(cls, d)


def _from_dict(cls, d: Dict[str, Any]):
    kwargs = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        value = d[f.name]
        default = f.default if f.default_factory is MISSING else f.default_factory()  # type: ignore[misc]
        if is_dataclass(default) and isinstance(value, dict):
            value = _from_dict(type(default), value)
        elif isinstance(value, list):
            value = _to_tuple(value)
        kwargs[f.name] = value
    return cls(**kwargs)


def _to_tuple(value):
    if isinstance(value, list):
        return tuple(_to_tuple(v) for v in value)
    return value


def _replace_path(obj, changes: Dict[str, Any]):
    from dataclasses import replace

    nested: Dict[str, Dict[str, Any]] = {}
    top: Dict[str, Any] = {}
    for key, value in changes.items():
        head, _, rest = key.partition(".")
        if rest:
            nested.setdefault(head, {})[rest] = value
        else:
            top[head] = value
    for head, sub in nested.items():
        top[head] = _replace_path(getattr(obj, head), sub)
    return replace(obj, **top)
