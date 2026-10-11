"""
Options rounds O5-O8 (plan 5x): the owner's books as pre-registered candidates on the structures engine
(``kite_connect.options.structures``).  Every parameter is the book's or a translation fixed in plan 5x
before any run; none is fitted.  CLI::

    python -m kite_connect.options.book_rounds run --candidates O5-A,O5-B \\
        --fo-store data/nse_engine/fo_store --runs-dir data/nse_engine/runs_options
    python -m kite_connect.options.book_rounds evaluate --round O5
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from kite_connect.options.sleeves import EVENT_DATES
from kite_connect.options.structures import (Context, Decisions, Leg, Position, Strategy, evaluate_round, record,
                                             run_structure)
from kite_connect.options.theory import CALL, PUT

logger = logging.getLogger(__name__)

#: Margin assumed for a naked short index option, as a share of its notional (no SPAN history in the store).
NAKED_MARGIN = 0.10
#: Futures re-hedge band for the delta-hedged structures (Natenberg: at least half a lot at this size).
HEDGE_BAND_LOTS = 0.5


# ---------------------------------------------------------------- shared rules

def _wing(ctx: Context, expiry: pd.Timestamp, option_type: str, short: float, short_px: float,
          ratio: float) -> Optional[Tuple[float, float]]:
    """The first further-OTM traded strike whose price is at most ``ratio`` x the short leg's."""
    outward = sorted((k for k in ctx.traded_strikes(expiry, option_type)
                      if (k < short if option_type == PUT else k > short)), reverse=option_type == PUT)
    for k in outward:
        px, _ = ctx.price(expiry, option_type, k)
        if px <= ratio * short_px:
            return k, px
    return None


def _per_unit(p: Position, amount: float) -> float:
    return amount / (p.info["lots"] * p.info["lot_size"])


def _hedge(ctx: Context, p: Position, out: Decisions) -> None:
    """Futures to the nearest whole lot against the legs' delta, when the open delta exceeds the band."""
    lot = p.info["lot_size"]
    deltas = [ctx.delta(l.expiry, l.option_type, l.strike) for l in p.legs]
    if any(d is None for d in deltas):
        return
    net = sum(l.units * d for l, d in zip(p.legs, deltas))
    if abs(net + p.hedge_units) >= HEDGE_BAND_LOTS * lot:
        target = -round(net / lot) * lot
        if target != p.hedge_units:
            out.hedge.append((p.id, float(target)))


def _event_before(ctx: Context, expiry: pd.Timestamp) -> bool:
    return any(ctx.d < pd.Timestamp(e) <= expiry for e in EVENT_DATES)


# ---------------------------------------------------------------- O5: Cordier & Gross

@dataclass
class TrendCredit(Strategy):
    """Sell on the trend side, second monthly, |delta| <= 0.20; the 200% and 10% rules (plan 5x O5)."""

    name: str = "O5-A trend-side credit spread (Cordier & Gross)"
    naked: bool = False
    max_delta: float = 0.20
    wing_premium_ratio: float = 1 / 3
    min_credit_to_risk: float = 0.10
    stop_multiple: float = 2.0
    take_fraction: float = 0.10
    expiry_rank: int = 2
    symbol: str = "NIFTY"
    capital: float = 750_000.0
    deploy_fraction: float = 0.5

    def decide(self, ctx: Context, positions: List[Position]) -> Decisions:
        out = Decisions()
        for p in positions:
            credit, to_close = -p.entry_value, -p.value()
            if credit > 0 and to_close >= self.stop_multiple * credit:
                out.close.append((p.id, "200% rule"))
            elif credit > 0 and to_close <= self.take_fraction * credit:
                out.close.append((p.id, "10% rule"))
        if not ctx.first_after_expiry():
            return out
        regime = ctx.regime()
        side = {"bull": PUT, "bear": CALL}.get(regime)
        expiry = ctx.monthly(self.expiry_rank)
        if side is None or expiry is None:
            return out
        short = ctx.strike_by_delta(expiry, side, self.max_delta)
        if short is None:
            return out
        short_px, _ = ctx.price(expiry, side, short)
        lot = ctx.q.lot_size(ctx.d, expiry)
        info = {"regime": regime, "expiry": str(expiry.date()), "short": short}
        if self.naked:
            out.open.append({"legs": [Leg(expiry, side, short, -1)], "max_loss_per_lot": NAKED_MARGIN * short * lot,
                             "info": {**info, "credit": short_px}})
            return out
        wing = _wing(ctx, expiry, side, short, short_px, self.wing_premium_ratio)
        if wing is None:
            return out
        k, wing_px = wing
        credit, width = short_px - wing_px, abs(short - k)
        if credit <= 0 or credit < self.min_credit_to_risk * (width - credit):
            return out
        out.open.append({"legs": [Leg(expiry, side, short, -1), Leg(expiry, side, k, 1)],
                         "max_loss_per_lot": (width - credit) * lot, "info": {**info, "wing": k, "credit": credit}})
        return out


# ---------------------------------------------------------------- O6: Natenberg

def _forward_vol(iv_front: float, t_front: float, iv_back: float, t_back: float) -> Optional[float]:
    var = (iv_back ** 2 * t_back - iv_front ** 2 * t_front) / (t_back - t_front)
    return math.sqrt(var) if var > 0 and t_back > t_front else None


@dataclass
class NatenbergCalendar(Strategy):
    """Long at-the-forward call calendar when forward volatility is cheap against the long-run mean and the
    front month is not cheap against the forecast; delta-hedged with futures (plan 5x O6-A)."""

    name: str = "O6-A long at-the-forward calendar (Natenberg)"
    margin: float = 0.02
    symbol: str = "NIFTY"
    capital: float = 750_000.0
    deploy_fraction: float = 0.5

    def _fv(self, ctx: Context, front: pd.Timestamp, back: pd.Timestamp, k: float) -> Optional[float]:
        a, b = ctx.iv(front, CALL, k), ctx.iv(back, CALL, k)
        if a is None or b is None:
            return None
        return _forward_vol(a, ctx.dte(front) / 365.0, b, ctx.dte(back) / 365.0)

    def decide(self, ctx: Context, positions: List[Position]) -> Decisions:
        out = Decisions()
        for p in positions:
            expiries = sorted({l.expiry for l in p.legs})
            if len(expiries) < 2 or ctx.sessions_to(expiries[0]) <= 2:
                out.close.append((p.id, "session before front expiry"))
                continue
            fv = self._fv(ctx, expiries[0], expiries[1], p.legs[0].strike)
            if fv is not None and fv >= ctx.long_run_vol():
                out.close.append((p.id, "forward vol at the mean"))
                continue
            _hedge(ctx, p, out)
        if not ctx.first_after_expiry():
            return out
        front, back = ctx.monthly(1), ctx.monthly(2)
        k = ctx.atm(front) if front is not None else None
        if back is None or k is None or k not in ctx.traded_strikes(back, CALL):
            return out
        fv, f_iv = self._fv(ctx, front, back, k), ctx.iv(front, CALL, k)
        if fv is None or f_iv is None:
            return out
        if not (fv <= ctx.long_run_vol() - self.margin and f_iv >= ctx.vol_forecast()):
            return out
        debit = ctx.price(back, CALL, k)[0] - ctx.price(front, CALL, k)[0]
        if debit <= 0:
            return out
        out.open.append({"legs": [Leg(front, CALL, k, -1), Leg(back, CALL, k, 1)],
                         "max_loss_per_lot": debit * ctx.q.lot_size(ctx.d, front),
                         "info": {"strike": k, "forward_vol": fv, "front_iv": f_iv}})
        return out


@dataclass
class ConditionalIronFly(Strategy):
    """Short at-the-forward straddle with +/-0.6 SD wings, only when ATM IV clears the forecast by 2 points and
    no Budget or election date falls before expiry; sized by the margin; delta-hedged (plan 5x O6-B)."""

    name: str = "O6-B conditional short iron butterfly (Natenberg)"
    margin: float = 0.02
    full_size_margin: float = 0.07
    wing_sd: float = 0.6
    exit_days: int = 7
    symbol: str = "NIFTY"
    capital: float = 750_000.0
    deploy_fraction: float = 0.5

    def decide(self, ctx: Context, positions: List[Position]) -> Decisions:
        out = Decisions()
        forecast = ctx.vol_forecast()
        for p in positions:
            expiry, k = p.legs[0].expiry, p.info["strike"]
            iv = ctx.iv(expiry, CALL, k)
            if ctx.dte(expiry) <= self.exit_days:
                out.close.append((p.id, "expiry week"))
            elif iv is not None and iv <= forecast:
                out.close.append((p.id, "IV back at the forecast"))
            else:
                _hedge(ctx, p, out)
        if not ctx.first_after_expiry():
            return out
        expiry = ctx.monthly(2)
        k = ctx.atm(expiry) if expiry is not None else None
        if k is None or _event_before(ctx, expiry):
            return out
        iv = ctx.iv(expiry, CALL, k)
        if iv is None or iv - forecast < self.margin:
            return out
        sd = ctx.q.forward(ctx.d, expiry) * iv * math.sqrt(ctx.dte(expiry) / 365.0)
        up = min((x for x in ctx.traded_strikes(expiry, CALL) if x >= k + self.wing_sd * sd), default=None)
        dn = max((x for x in ctx.traded_strikes(expiry, PUT) if x <= k - self.wing_sd * sd), default=None)
        if up is None or dn is None:
            return out
        px = {key: ctx.price(expiry, t, x)[0] for key, t, x in (("c", CALL, k), ("p", PUT, k), ("cu", CALL, up), ("pd", PUT, dn))}
        credit = px["c"] + px["p"] - px["cu"] - px["pd"]
        width = max(up - k, k - dn)
        if credit <= 0 or width <= credit:
            return out
        out.open.append({"legs": [Leg(expiry, CALL, k, -1), Leg(expiry, PUT, k, -1), Leg(expiry, CALL, up, 1),
                                  Leg(expiry, PUT, dn, 1)],
                         "max_loss_per_lot": (width - credit) * ctx.q.lot_size(ctx.d, expiry),
                         "scale": min(1.0, (iv - forecast) / self.full_size_margin),
                         "info": {"strike": k, "iv": iv, "forecast": forecast, "credit": credit}})
        return out


# ---------------------------------------------------------------- O7: Jabbour & Budwick

@dataclass
class RegimeDebit(Strategy):
    """The outlook-and-IV table of plan 5x O7, debit structures only; ``adjust`` adds the book's lock-in and
    butterfly-to-condor rules (O7-B)."""

    name: str = "O7-A regime debit sleeve, no adjustments (Jabbour & Budwick)"
    adjust: bool = False
    step: float = 0.05
    low_rank: float = 33.0
    high_rank: float = 67.0
    single_exit_dte: int = 30
    fly_min_dte: int = 7
    symbol: str = "NIFTY"
    capital: float = 750_000.0
    deploy_fraction: float = 0.5

    def _w(self, ctx: Context) -> float:
        return max(100.0, round(self.step * ctx.spot / 100.0) * 100.0)

    def decide(self, ctx: Context, positions: List[Position]) -> Decisions:
        out = Decisions()
        decision = ctx.first_after_expiry()
        regime = ctx.regime()
        for p in positions:
            kind, expiry = p.info["kind"], p.legs[0].expiry
            if decision and p.info["regime"] != regime:
                out.close.append((p.id, "regime changed"))
            elif kind == "single" and ctx.dte(expiry) <= self.single_exit_dte:
                out.close.append((p.id, "30 days to expiry"))
            elif kind != "single" and ctx.sessions_to(expiry) <= 2:
                out.close.append((p.id, "session before expiry"))
            elif self.adjust and not p.info.get("adjustments"):          # at most one adjustment
                self._adjust(ctx, p, out)
        if not decision:
            return out
        rank = ctx.iv_rank()
        if not math.isfinite(rank):
            return out
        w = self._w(ctx)
        order = None
        if regime in ("bull", "bear"):
            t, sign = (CALL, 1) if regime == "bull" else (PUT, -1)
            expiry = ctx.monthly(2)
            k = ctx.atm(expiry) if expiry is not None else None
            if k is None:
                return out
            if rank < self.low_rank:
                order = ("single", [Leg(expiry, t, k, 1)])
            elif k + sign * w in ctx.traded_strikes(expiry, t):
                order = ("vertical", [Leg(expiry, t, k, 1), Leg(expiry, t, k + sign * w, -1)])
        elif rank > self.high_rank:
            expiry = ctx.monthly(1)
            k = ctx.atm(expiry) if expiry is not None else None
            if k is not None and {k - w, k + w} <= set(ctx.traded_strikes(expiry, CALL)):
                order = ("fly", [Leg(expiry, CALL, k - w, 1), Leg(expiry, CALL, k, -2), Leg(expiry, CALL, k + w, 1)])
        if order is None:
            return out
        kind, legs = order
        debit = sum(l.units * ctx.price(l.expiry, l.option_type, l.strike)[0] for l in legs)
        if not debit > 0:
            return out
        out.open.append({"legs": legs, "max_loss_per_lot": debit * ctx.q.lot_size(ctx.d, legs[0].expiry),
                         "info": {"kind": kind, "regime": regime, "iv_rank": rank, "w": w, "strike": legs[0].strike}})
        return out

    def _adjust(self, ctx: Context, p: Position, out: Decisions) -> None:
        kind, w, expiry = p.info["kind"], p.info["w"], p.legs[0].expiry
        if kind == "single":                                   # lock in: sell the next strike once it pays the debit
            leg = p.legs[0]
            k2 = leg.strike + w if leg.option_type == CALL else leg.strike - w
            px, traded = ctx.price(expiry, leg.option_type, k2)
            if traded and px >= _per_unit(p, p.entry_value):
                out.adjust.append((p.id, [Leg(expiry, leg.option_type, k2, -1)], "lock-in"))
        elif kind == "fly" and ctx.dte(expiry) >= self.fly_min_dte:
            k = p.info["strike"] + w                          # the body
            debit = _per_unit(p, p.entry_value)
            if ctx.spot > k + w - debit:
                new = [Leg(expiry, CALL, k, 1), Leg(expiry, CALL, k + w, -2), Leg(expiry, CALL, k + 2 * w, 1)]
            elif ctx.spot < k - w + debit:
                new = [Leg(expiry, CALL, k - 2 * w, 1), Leg(expiry, CALL, k - w, -2), Leg(expiry, CALL, k, 1)]
            else:
                return
            if all(l.strike in ctx.traded_strikes(expiry, CALL) for l in new):
                out.adjust.append((p.id, new, "butterfly to condor"))


# ---------------------------------------------------------------- O8: Ursone

@dataclass
class TermCalendarStraddle(Strategy):
    """After a spike that inverts the term structure: short the front at-the-forward straddle, long the
    second-month straddle at the same strike, equal lots, for a debit (plan 5x O8-A).  One at a time."""

    name: str = "O8-A post-spike term-structure calendar straddle (Ursone)"
    min_term_spread: float = 0.03
    min_rank: float = 70.0
    min_front_dte: int = 14
    max_front_dte: int = 30
    symbol: str = "NIFTY"
    capital: float = 750_000.0
    deploy_fraction: float = 0.5

    def decide(self, ctx: Context, positions: List[Position]) -> Decisions:
        out = Decisions()
        spread = ctx.term_spread()
        for p in positions:
            front, k = min(l.expiry for l in p.legs), p.info["strike"]
            if ctx.sessions_to(front) <= 2:
                out.close.append((p.id, "session before front expiry"))
            elif math.isfinite(spread) and spread <= 0:
                out.close.append((p.id, "term structure normal"))
            elif abs(ctx.spot - k) > p.info["front_straddle"]:
                out.close.append((p.id, "spot left the strike"))
        if positions:
            return out
        front, back = ctx.monthly(1), ctx.monthly(2)
        if front is None or back is None or not (self.min_front_dte <= ctx.dte(front) <= self.max_front_dte):
            return out
        if not (spread >= self.min_term_spread and ctx.iv_rank() >= self.min_rank):
            return out
        k = ctx.atm(front)
        if k is None or not {k} <= set(ctx.traded_strikes(back, CALL)) & set(ctx.traded_strikes(back, PUT)):
            return out
        f = ctx.price(front, CALL, k)[0] + ctx.price(front, PUT, k)[0]
        b = ctx.price(back, CALL, k)[0] + ctx.price(back, PUT, k)[0]
        if not b - f > 0:
            return out
        out.open.append({"legs": [Leg(front, CALL, k, -1), Leg(front, PUT, k, -1), Leg(back, CALL, k, 1),
                                  Leg(back, PUT, k, 1)],
                         "max_loss_per_lot": (b - f) * ctx.q.lot_size(ctx.d, front),
                         "info": {"strike": k, "front_straddle": f, "term_spread": spread}})
        return out


CANDIDATES: Dict[str, Strategy] = {
    "O5-A": TrendCredit(),
    "O5-B": TrendCredit(name="O5-B trend-side naked short (Cordier & Gross)", naked=True),
    "O6-A": NatenbergCalendar(),
    "O6-B": ConditionalIronFly(),
    "O7-A": RegimeDebit(),
    "O7-B": RegimeDebit(name="O7-B regime debit sleeve with the book's adjustments (Jabbour & Budwick)", adjust=True),
    "O8-A": TermCalendarStraddle(),
}
EXTENDED_START = "2007-01-02"


def round_of(key: str) -> str:
    return key.split("-")[0]


# ---------------------------------------------------------------- CLI

def main(argv: Optional[Sequence[str]] = None) -> int:
    from kite_connect.options.backtest import BOOK_RUNS_DIR, OPTIONS_RUNS_DIR

    ap = argparse.ArgumentParser(description="Options rounds O5-O8: the owner's books (plan 5x)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="backtest and record candidates in the options registry")
    p.add_argument("--candidates", default=",".join(CANDIDATES))
    p.add_argument("--start", default="2013-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--fo-store", default="data/nse_engine/fo_store")
    p.add_argument("--runs-dir", default=OPTIONS_RUNS_DIR)
    p.add_argument("--no-record", action="store_true", help="print metrics only (debugging)")
    p = sub.add_parser("evaluate", help="plan 5v's gates for one round")
    p.add_argument("--round", required=True, choices=sorted({round_of(k) for k in CANDIDATES}))
    p.add_argument("--out")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cmd == "run":
        for key in args.candidates.split(","):
            cfg = CANDIDATES[key]
            res = run_structure(cfg, args.fo_store, args.start, args.end)
            where = "not recorded" if args.no_record else record(res, args.runs_dir, round_of(key))
            print(key, where, {k: res.metrics[k] for k in ("sharpe", "cagr", "max_drawdown", "positions", "exit_reasons")})
        return 0
    cands = {k: v for k, v in CANDIDATES.items() if round_of(k) == args.round}
    report = evaluate_round(args.round, cands, OPTIONS_RUNS_DIR, BOOK_RUNS_DIR)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
