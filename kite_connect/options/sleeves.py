"""
Options sleeves for the book: the configurations of tracker O2, exactly as
pre-registered in docs/nse_engine_validation_plan.md § 5q (3 Oct 2026, 20:55
IST).  Changing a field changes the configuration hash, which makes it a new
trial in the options family (U32), recorded and counted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

from kite_connect.options.fno_costs import FNO_COST_MODEL_VERSION

#: Union Budget and general-election result days (M5 ch. 18: no writing across events).
EVENT_DATES: Tuple[str, ...] = (
    "2013-02-28", "2014-02-17", "2014-05-16", "2014-07-10", "2015-02-28", "2016-02-29",
    "2017-02-01", "2018-02-01", "2019-02-01", "2019-05-23", "2019-07-05", "2020-02-01",
    "2021-02-01", "2022-02-01", "2023-02-01", "2024-02-01", "2024-06-04", "2024-07-23",
    "2025-02-01", "2026-02-01",
)


@dataclass(frozen=True)
class SleeveConfig:
    """A bear call spread written each month on an index's monthly options.

    ``rule`` "sd": short call above the ``short_sd`` upper bound, wing above
    ``wing_sd`` (M5 ch. 18).  "max_pain": short call at or above max pain x
    (1 + ``max_pain_buffer``), wing ``wing_distance_sd`` period SDs beyond it
    (M6 ch. 13).
    """

    name: str
    rule: str
    entry_max_dte: int
    short_sd: float = 2.0
    wing_sd: float = 3.0
    max_pain_buffer: float = 0.05
    wing_distance_sd: float = 1.0
    early_exit_atm: bool = True
    symbol: str = "NIFTY"
    vol_lookback: int = 252
    deploy_fraction: float = 0.35
    initial_capital: float = 750_000.0
    skip_events: bool = True
    fill_lag: int = 1
    start: str = "2013-01-01"
    end: str = "2025-12-31"

    def __post_init__(self):
        if self.rule not in ("sd", "max_pain"):
            raise ValueError(f"rule must be 'sd' or 'max_pain', got {self.rule!r}")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def config_hash(self) -> str:
        """Behaviour only: name and dates are left out (a window is part of a run), the cost model is in."""
        payload = {k: v for k, v in self.to_dict().items() if k not in ("name", "start", "end")}
        payload["fno_cost_model"] = FNO_COST_MODEL_VERSION
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


CANDIDATES: Dict[str, SleeveConfig] = {
    "A1": SleeveConfig("O2-A1 SD call spread, 15 days", "sd", 15, short_sd=2.0, wing_sd=3.0),
    "A2": SleeveConfig("O2-A2 SD call spread, 4 days", "sd", 4, short_sd=1.0, wing_sd=2.0),
    "B": SleeveConfig("O2-B max-pain call spread", "max_pain", 15, early_exit_atm=False),
}

#: Overlay weight on the book: 25% of capital to option writing (M5 ch. 18).
SLEEVE_WEIGHT = 0.25
#: The reported 2007-25 window (plan 5q addendum), E4's extended run's.
EXTENDED_WINDOW = ("2007-01-02", "2025-12-31")
