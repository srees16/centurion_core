"""
Basket executor: place a pre-trade report's legs safely, in one of three modes.

* ``dry_run``: log every order it would send; send nothing.
* ``paper`` (the default everywhere): fill a leg at the touch price (ask for a
  buy, bid for a sell) when its LIMIT is marketable against a fresh quote;
  send nothing.
* ``live``: real DAY LIMIT orders with ``autoslice`` (Kite splits a quantity
  above the freeze limit into up to ten orders).  The caller must already
  have the typed confirmation and the static-IP check (``cli.py``).

Rules in every mode:

1. BUY legs go first: on entry they are the hedges, on exit they close the
   shorts, so the book is never short an option it has not covered.
2. A leg must fill in full, every slice of it, before the next leg starts.
   Live orders are polled until each slice is COMPLETE, REJECTED or
   CANCELLED, or until ``fill_timeout_seconds``; then open slices are
   cancelled.
3. On the first leg that does not fill in full, the basket stops and the
   report states the partial position, and calls out any short not covered
   by a long of the same type (a naked short), so it is never left
   unreported.
4. Every order request and response, simulated or real, is logged
   (``Broker.log``).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from kite_connect.options.broker import FINAL_STATUSES, Broker, OrderError
from kite_connect.options.live_chain import quote_prices
from kite_connect.options.options_config import LimitsConfig
from kite_connect.options.pretrade import LegOrder
from kite_connect.options.theory import BUY, SELL

logger = logging.getLogger(__name__)

MODES = ("dry_run", "paper", "live")
FILLED, PARTIAL, NOT_FILLED, NOT_SENT = "FILLED", "PARTIAL", "NOT FILLED", "NOT SENT (dry run)"
POLL_SECONDS = 1.0


@dataclass
class LegResult:
    leg: LegOrder
    status: str
    filled: int = 0
    average_price: float = 0.0
    order_ids: List[str] = field(default_factory=list)
    message: str = ""

    @property
    def signed_filled(self) -> int:
        return self.filled if self.leg.side == BUY else -self.filled


@dataclass
class ExecutionReport:
    mode: str
    results: List[LegResult]
    stopped: str = ""

    @property
    def completed(self) -> bool:
        return not self.stopped and all(r.status == FILLED for r in self.results)

    def naked_shorts(self) -> List[str]:
        """Filled shorts not covered by filled longs of the same option type."""
        out = []
        for option_type in ("CE", "PE"):
            rs = [r for r in self.results if r.leg.contract.option_type == option_type]
            short = sum(r.filled for r in rs if r.leg.side == SELL)
            long = sum(r.filled for r in rs if r.leg.side == BUY)
            if short > long:
                out.append(f"{short - long} {option_type} units short without a long {option_type} to cover them")
        return out

    def text(self) -> str:
        lines = [f"EXECUTION ({self.mode}): " + ("completed" if self.completed else
                                                 "not sent" if self.mode == "dry_run" else f"STOPPED - {self.stopped}")]
        for r in self.results:
            price = f" avg {r.average_price:.2f}" if r.filled else ""
            ids = f" orders {', '.join(r.order_ids)}" if r.order_ids else ""
            lines.append(f"  {r.leg.side:4s} {r.leg.contract.tradingsymbol:24s} {r.status:18s} "
                         f"{r.filled}/{r.leg.quantity}{price}{ids}{(' - ' + r.message) if r.message else ''}")
        for warning in self.naked_shorts():
            lines.append(f"  WARNING NAKED SHORT: {warning}; close it or add the hedge now")
        return "\n".join(lines)


def execution_order(legs: Sequence[LegOrder]) -> List[LegOrder]:
    """BUY legs first (hedges, or closes of shorts), then SELL legs; spec order within each."""
    return [l for l in legs if l.side == BUY] + [l for l in legs if l.side == SELL]


class BasketExecutor:
    def __init__(self, broker: Broker, mode: str = "paper", limits: LimitsConfig = LimitsConfig(),
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.broker, self.mode, self.limits = broker, mode, limits
        self.clock, self.sleep = clock, sleep

    def execute(self, legs: Sequence[LegOrder], tag: str = "") -> ExecutionReport:
        report = ExecutionReport(self.mode, [])
        for leg in execution_order(legs):
            result = self._one(leg, tag)
            report.results.append(result)
            if self.mode != "dry_run" and result.status != FILLED:
                report.stopped = f"{leg.side} {leg.contract.tradingsymbol} {result.status}: {result.message}"
                break
        for warning in report.naked_shorts():
            logger.error("NAKED SHORT after basket %s: %s", tag, warning)
        return report

    # ── one leg ─────────────────────────────────────────────────
    def _one(self, leg: LegOrder, tag: str) -> LegResult:
        request = {"tradingsymbol": leg.contract.tradingsymbol, "side": leg.side, "quantity": leg.quantity,
                   "price": leg.limit, "tag": tag}
        if self.mode == "dry_run":
            self.broker.log("place", request, mode="dry_run")
            return LegResult(leg, NOT_SENT)
        if self.mode == "paper":
            return self._paper(leg, request)
        return self._live(leg, tag)

    def _paper(self, leg: LegOrder, request: dict) -> LegResult:
        p = quote_prices(self.broker.quotes([leg.contract.quote_key]).get(leg.contract.quote_key))
        touch = p["ask"] if leg.side == BUY else p["bid"]
        touch = touch if touch and touch > 0 else p["ltp"]
        marketable = touch and touch > 0 and (touch <= leg.limit if leg.side == BUY else touch >= leg.limit)
        if not marketable:
            result = LegResult(leg, NOT_FILLED, message=f"limit {leg.limit:.2f} not marketable (touch {touch})")
        else:
            result = LegResult(leg, FILLED, leg.quantity, float(touch))
        self.broker.log("place", request, response={"status": result.status, "price": result.average_price},
                        mode="paper")
        return result

    def _live(self, leg: LegOrder, tag: str) -> LegResult:
        try:
            placed = self.broker.place_limit(leg.contract.tradingsymbol, leg.side, leg.quantity, leg.limit, tag)
        except OrderError as exc:
            return LegResult(leg, NOT_FILLED, message=str(exc))
        orders = self._wait(placed.order_ids)
        for oid, o in orders.items():
            if o.get("status") not in FINAL_STATUSES:
                self.broker.cancel(oid)
                orders[oid] = self._last(oid) or o
        filled = sum(int(o.get("filled_quantity") or 0) for o in orders.values())
        value = sum(int(o.get("filled_quantity") or 0) * float(o.get("average_price") or 0) for o in orders.values())
        notes = placed.errors + [str(o.get("status_message")) for o in orders.values()
                                 if o.get("status") in ("REJECTED", "CANCELLED") and o.get("status_message")]
        status = FILLED if filled >= leg.quantity else PARTIAL if filled else NOT_FILLED
        return LegResult(leg, status, filled, value / filled if filled else 0.0, placed.order_ids, "; ".join(notes))

    def _last(self, order_id: str) -> Optional[dict]:
        history = self.broker.order_history(order_id)
        return history[-1] if history else None

    def _wait(self, order_ids: List[str]) -> Dict[str, dict]:
        """Latest state of every slice, once all are final or the fill timeout passes."""
        deadline = self.clock() + self.limits.fill_timeout_seconds
        latest: Dict[str, dict] = {}
        while True:
            latest = {oid: (self._last(oid) or {}) for oid in order_ids}
            if all(o.get("status") in FINAL_STATUSES for o in latest.values()) or self.clock() >= deadline:
                return latest
            self.sleep(POLL_SECONDS)
