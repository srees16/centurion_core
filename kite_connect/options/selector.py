"""
Strategy selector: market view + volatility view + days to expiry -> ranked
Module 6 strategies, using only the PDF's rules (docs/options/STRATEGIES.md § Selector).

Every score change carries the rule it came from, so a ranking can be read
and challenged.  Strike guidance is in the PDF's moneyness labels; turn a
label into a listed strike with :func:`strike_for`.  The selector never
trades.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from kite_connect.options.options_config import SelectorConfig
from kite_connect.options.theory import CALL, atm_strike

VIEWS = ("moderate_bull", "strong_bull", "moderate_bear", "strong_bear",
         "neutral_range", "neutral_big_move", "futures_like")
VOL_VIEWS = ("rising", "falling", "flat")
IV_LEVELS = ("low", "normal", "high", "very_high")


@dataclass(frozen=True)
class MarketContext:
    """What the trader believes, plus what the chain shows.

    ``iv_level`` compares today's IV with the realised-volatility cone (M5
    ch. 20): low below -1 SD, high above +1 SD, very_high above twice normal
    (M6 ch. 4).  ``rich_side`` is "puts" after a fall or "calls" after a rally
    (M6 ch. 3, 8).  ``event_vs_consensus`` is "differs" or "matches" (M6 ch. 10).
    """

    view: str
    days_to_expiry: int
    vol_view: str = "flat"
    days_to_target: Optional[int] = None
    iv_level: str = "normal"
    rich_side: Optional[str] = None
    event: bool = False
    event_vs_consensus: Optional[str] = None
    range_bound: bool = False
    cost_sensitive: bool = False

    def __post_init__(self):
        if self.view not in VIEWS:
            raise ValueError(f"view must be one of {VIEWS}")
        if self.vol_view not in VOL_VIEWS:
            raise ValueError(f"vol_view must be one of {VOL_VIEWS}")
        if self.iv_level not in IV_LEVELS:
            raise ValueError(f"iv_level must be one of {IV_LEVELS}")


@dataclass
class Candidate:
    strategy: str
    chapter: int
    score: float
    strikes: Dict[str, str]
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def adjust(self, delta: float, rule: str) -> None:
        self.score += delta
        self.reasons.append(f"{delta:+.2f} {rule}")


def series_half(days_to_expiry: int, cfg: SelectorConfig = SelectorConfig()) -> str:
    """"1st" when more than 15 days are left, else "2nd" (CONCEPTS.md open decision 5)."""
    return "1st" if days_to_expiry >= cfg.first_half_min_dte else "2nd"


def target_bucket(days_to_expiry: int, days_to_target: Optional[int],
                  cfg: SelectorConfig = SelectorConfig()) -> Tuple[str, str]:
    """(series half, bucket) in the PDF's tables: 5/15/25 days or 'same day'/5/10 days, or 'expiry'."""
    half = series_half(days_to_expiry, cfg)
    if days_to_target is None or days_to_target >= days_to_expiry:
        return half, "expiry"
    limits = ((5, "5d"), (15, "15d"), (25, "25d")) if half == "1st" else ((1, "same_day"), (5, "5d"), (10, "10d"))
    for limit, name in limits:
        if days_to_target <= limit:
            return half, name
    return half, "expiry"


# M5 ch. 22: naked option buying (calls and puts alike).
NAKED_BUY = {
    ("1st", "5d"): "far OTM", ("1st", "15d"): "ATM or slightly OTM", ("1st", "25d"): "slightly ITM",
    ("1st", "expiry"): "ITM",
    ("2nd", "same_day"): "far OTM", ("2nd", "5d"): "slightly OTM", ("2nd", "10d"): "slightly ITM or ATM",
    ("2nd", "expiry"): "ITM",
}
# M6 ch. 2: the bought (lower) strike of a bull call spread.
BULL_CALL_LOWER = {
    ("1st", "5d"): "far OTM", ("1st", "15d"): "slightly OTM", ("1st", "25d"): "ATM", ("1st", "expiry"): "ATM",
    ("2nd", "same_day"): "far OTM", ("2nd", "5d"): "far OTM", ("2nd", "10d"): "slightly OTM",
    ("2nd", "expiry"): "ATM",
}
# M6 ch. 7: (bought higher put, sold lower put).
BEAR_PUT = {
    ("1st", "5d"): ("far OTM", "far OTM"), ("1st", "15d"): ("ATM", "slightly OTM"), ("1st", "25d"): ("ATM", "OTM"),
    ("1st", "expiry"): ("ATM", "OTM"),
    ("2nd", "same_day"): ("OTM", "OTM"), ("2nd", "5d"): ("ITM or OTM", "OTM"), ("2nd", "10d"): ("ITM or OTM", "OTM"),
    ("2nd", "expiry"): ("ITM or OTM", "OTM"),
}
# M6 ch. 8: (bought higher call, sold lower call); 2nd-half rows follow the graphs.
BEAR_CALL = {
    ("1st", "5d"): ("far OTM", "OTM"), ("1st", "15d"): ("far OTM", "OTM"), ("1st", "25d"): ("OTM", "slightly OTM"),
    ("1st", "expiry"): ("OTM", "ATM"),
    ("2nd", "same_day"): ("far OTM", "far OTM"), ("2nd", "5d"): ("far OTM", "slightly OTM"),
    ("2nd", "10d"): ("slightly OTM", "ATM"), ("2nd", "expiry"): ("OTM", "ATM or ITM"),
}


def naked_buy_strike(days_to_expiry: int, days_to_target: Optional[int],
                     cfg: SelectorConfig = SelectorConfig()) -> str:
    """M5 ch. 22's best strike to buy for a target ``days_to_target`` away."""
    return NAKED_BUY[target_bucket(days_to_expiry, days_to_target, cfg)]


def strike_for(label: str, option_type: str, spot: float, strikes: Sequence[float],
               cfg: SelectorConfig = SelectorConfig()) -> float:
    """The listed strike for a moneyness label ("X or Y" takes X); OTM is above spot for calls, below for puts."""
    offsets = dict(cfg.strike_offsets)
    name = label.split(" or ")[0].strip()
    if name not in offsets:
        raise ValueError(f"unknown moneyness label {label!r}")
    ks = sorted(float(k) for k in strikes)
    i = ks.index(atm_strike(spot, ks))
    step = offsets[name] if option_type.upper() == CALL else -offsets[name]
    return ks[min(max(i + step, 0), len(ks) - 1)]


def _candidates(ctx: MarketContext, cfg: SelectorConfig) -> List[Candidate]:
    half, bucket = target_bucket(ctx.days_to_expiry, ctx.days_to_target, cfg)
    n = cfg.spread_strikes
    near_expiry = ctx.days_to_expiry <= 5
    out: List[Candidate] = []

    if ctx.view == "moderate_bull":
        bc = Candidate("Bull Call Spread", 2, 1.0,
                       {"buy CE": BULL_CALL_LOWER[(half, bucket)], "sell CE": f"{n} strikes above the bought one"})
        bp = Candidate("Bull Put Spread", 3, 1.0, {"buy PE": "OTM", "sell PE": "ITM"})
        if ctx.rich_side == "puts" or ctx.iv_level in ("high", "very_high"):
            bp.adjust(+1.0, "M6 ch. 3: after a fall puts are rich and IV high, so take the credit spread")
            bc.adjust(-0.5, "M5 ch. 20 / M6 ch. 2: a debit spread costs more at high IV")
        if ctx.iv_level == "low":
            bc.adjust(+0.5, "M6 ch. 2: the debit spread is cheaper at low IV")
        if half == "1st":
            bp.adjust(+0.25, "M6 ch. 3: needs ample time to expiry")
        out += [bc, bp]

    elif ctx.view == "strong_bull":
        legs = ({"sell 1 CE": "slightly ITM", "buy 2 CE": "slightly OTM"} if half == "1st"
                else {"sell 1 CE": "deep ITM", "buy 2 CE": "slightly ITM"})
        crbs = Candidate("Call Ratio Back Spread", 4, 1.0, legs, warnings=["execute only for a net credit (M6 ch. 4, 9)"])
        ladder = Candidate("Bear Call Ladder", 5, 0.75, {"sell CE (K1)": "ITM", "buy CE (K2)": "ATM", "buy CE (K3)": "OTM"},
                           warnings=["execute only for a net credit; needs a large move up (M6 ch. 5)"])
        for c in (crbs, ladder):
            if ctx.vol_view == "rising" and not near_expiry:
                c.adjust(+0.5 if half == "1st" else +0.25, "M6 ch. 4: rising IV helps with time left (30 d most, 15 d less)")
            if ctx.vol_view == "rising" and near_expiry:
                c.adjust(-0.75, "M6 ch. 4: rising IV near expiry hurts")
            if ctx.iv_level == "very_high" and half == "1st":
                c.adjust(-1.0, "M6 ch. 4: avoid at the start of a series when IV is already high")
        if ctx.event:
            ladder.adjust(+0.5, "M6 ch. 5: the PDF uses the ladder around results")
        out += [crbs, ladder]

    elif ctx.view == "moderate_bear":
        hi_p, lo_p = BEAR_PUT[(half, bucket)]
        hi_c, lo_c = BEAR_CALL[(half, bucket)]
        bpu = Candidate("Bear Put Spread", 7, 1.0, {"buy PE": hi_p, "sell PE": lo_p})
        bca = Candidate("Bear Call Spread", 8, 1.0, {"buy CE": hi_c, "sell CE": lo_c})
        if ctx.rich_side == "calls" or ctx.iv_level in ("high", "very_high"):
            bca.adjust(+1.0, "M6 ch. 8: after a rally calls are rich, so take the credit spread")
        if half == "2nd" and ctx.vol_view == "rising":
            bpu.adjust(+0.5, "M6 ch. 7: in the 2nd half take it when IV is expected to rise")
            bca.adjust(+0.5, "M6 ch. 8: take it when IV is expected to rise")
        if half == "2nd" and ctx.vol_view == "falling":
            bpu.adjust(-0.5, "M6 ch. 7: falling IV cheapens the spread near expiry")
        out += [bpu, bca]

    elif ctx.view == "strong_bear":
        prbs = Candidate("Put Ratio Back Spread", 9, 1.0, {"sell 1 PE": "ITM", "buy 2 PE": "OTM"},
                         warnings=["execute only for a net credit (M6 ch. 9)"])
        if ctx.vol_view == "rising" and half == "1st":
            prbs.adjust(+0.5, "M6 ch. 9: rising IV helps with 30 days left")
        if ctx.vol_view == "rising" and near_expiry:
            prbs.adjust(0.0, "M6 ch. 9: IV has little effect near expiry")
        out.append(prbs)

    elif ctx.view == "neutral_big_move":
        ls = Candidate("Long Straddle", 10, 1.0, {"buy CE": "ATM", "buy PE": "ATM"})
        lg = Candidate("Long Strangle", 12, 0.75, {"buy PE": "OTM", "buy CE": "OTM (equidistant)"})
        for c in (ls, lg):
            c.warnings.append("M6 ch. 10 needs all five: low IV at entry, IV rising, a large move, "
                              "early in the series, an event outcome unlike the consensus")
            if ctx.iv_level == "low":
                c.adjust(+0.5, "M6 ch. 10: IV is low at entry")
            if ctx.iv_level in ("high", "very_high"):
                c.adjust(-1.0, "M6 ch. 10: IV already high at entry")
            if ctx.vol_view == "rising":
                c.adjust(+0.5, "M6 ch. 10: IV expected to rise while held")
            if ctx.vol_view == "falling":
                c.adjust(-1.0, "M6 ch. 10: falling IV (IV crush) breaks the position")
            if ctx.event_vs_consensus == "differs":
                c.adjust(+0.5, "M6 ch. 10: event outcome expected to differ from the consensus")
            if ctx.event_vs_consensus == "matches":
                c.adjust(-1.0, "M6 ch. 10: an outcome that matches expectations crushes IV")
            if near_expiry:
                c.adjust(-0.5, "M6 ch. 10: the move must come well before expiry (theta)")
        if ctx.cost_sensitive:
            lg.adjust(+0.5, "M6 ch. 12: the strangle when cost matters")
        out += [ls, lg]

    elif ctx.view == "neutral_range":
        ss = Candidate("Short Straddle", 11, 1.0, {"sell CE": "ATM", "sell PE": "ATM"},
                       warnings=["unlimited loss; delta drifts with gamma (M6 ch. 11)"])
        sg = Candidate("Short Strangle", 12, 1.0, {"sell PE": "OTM (below the range)", "sell CE": "OTM (above the range)"},
                       warnings=["unlimited loss; watch for a breakout (M6 ch. 12)"])
        for c in (ss, sg):
            if ctx.iv_level in ("high", "very_high"):
                c.adjust(+0.5, "M6 ch. 11: IV high at entry")
            if ctx.vol_view == "falling":
                c.adjust(+0.5, "M6 ch. 11: IV expected to fall")
            if ctx.vol_view == "rising":
                c.adjust(-1.0, "M6 ch. 11: rising IV hurts short options")
        if ctx.range_bound:
            sg.adjust(+0.5, "M6 ch. 12: range-bound underlying, write outside the range")
        if ctx.event:
            ss.adjust(+0.25, "M6 ch. 11: before an event IV inflates premiums; exit after the announcement")
        if ctx.days_to_expiry <= cfg.short_atm_min_dte:
            ss.adjust(-1.5, "M5 ch. 13: never short ATM options near expiry (gamma)")
        out += [ss, sg]

    else:  # futures_like
        out.append(Candidate("Synthetic Long", 6, 1.0, {"buy CE": "ATM", "sell PE": "ATM (same strike)"},
                             warnings=["run the arbitrage check against futures (M6 ch. 6)"]))
    return out


def select(ctx: MarketContext, cfg: SelectorConfig = SelectorConfig()) -> List[Candidate]:
    """Candidates for the view, best first (ties keep the PDF's chapter order)."""
    return sorted(_candidates(ctx, cfg), key=lambda c: -c.score)
