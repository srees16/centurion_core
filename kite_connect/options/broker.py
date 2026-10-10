"""
Kite Connect access for the options toolkit: session, instruments, quotes,
margins and orders, behind one small class.

The rest of the toolkit talks only to :class:`Broker`, so tests pass a fake
client with the same methods and no network.

* **Session.** :func:`connect` uses today's token from the daily email-link
  login (tracker U23, ``daily_login.kite_from_stored_token``); the toolkit
  never logs in by itself.  Calls go through ``CENTURION_KITE_PROXY`` when it
  is set.  Orders also need the egress IP to be the one registered with
  Zerodha (``CENTURION_KITE_STATIC_IP``, mandatory for API orders since April
  2026); market data works from any IP.
* **Rate limits.** Quotes are throttled to Kite's one request a second
  (500 instruments each); orders to ``LimitsConfig.max_orders_per_second``,
  below Kite's ten.
* **Freeze quantity.** Orders go out with ``autoslice=true``: Kite splits a
  quantity above the exchange freeze limit into up to ten orders and returns
  one entry per slice.  ``kiteconnect`` 5.1's ``place_order`` neither takes
  the flag nor parses a list, so orders use the client's request layer.
* **Audit.** Every order request and response, and every simulated one in
  paper or dry-run mode, is appended to a JSON-lines log.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd

from kite_connect.options.options_config import LimitsConfig

logger = logging.getLogger(__name__)

EXCHANGE = "NFO"
PRODUCT = "NRML"
QUOTE_BATCH = 500
QUOTES_PER_SECOND = 1.0
DEFAULT_ORDER_LOG = Path("data/options/orders.jsonl")
#: Order statuses after which an order no longer changes.
FINAL_STATUSES = frozenset({"COMPLETE", "CANCELLED", "REJECTED"})


class OrderError(RuntimeError):
    """An order request Kite refused, or a slice it could not place."""


@dataclass
class SliceResult:
    """What one autoslice order request produced: placed slice ids and the slices that failed."""

    order_ids: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class _Throttle:
    """At most ``per_second`` calls a second (blocking)."""

    def __init__(self, per_second: float, clock: Callable[[], float], sleep: Callable[[float], None]):
        self.interval = 1.0 / per_second if per_second > 0 else 0.0
        self._clock, self._sleep, self._last = clock, sleep, float("-inf")

    def wait(self) -> None:
        delay = self._last + self.interval - self._clock()
        if delay > 0:
            self._sleep(delay)
        self._last = self._clock()


class Broker:
    """A Kite session for NFO options, throttled and audited."""

    def __init__(self, kite: Any, order_log: Path = DEFAULT_ORDER_LOG, limits: LimitsConfig = LimitsConfig(),
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self.kite = kite
        self.order_log = Path(order_log)
        self.sleep = sleep
        self._quotes = _Throttle(QUOTES_PER_SECOND, clock, sleep)
        self._orders = _Throttle(limits.max_orders_per_second, clock, sleep)
        self._instruments: Dict[str, pd.DataFrame] = {}

    # ── market data ─────────────────────────────────────────────
    def instruments(self, exchange: str = EXCHANGE) -> pd.DataFrame:
        """Kite's instruments dump for ``exchange`` (once per Broker): lot sizes, expiries, strikes, ticks."""
        if exchange not in self._instruments:
            self._instruments[exchange] = pd.DataFrame(self.kite.instruments(exchange))
        return self._instruments[exchange]

    def quotes(self, keys: Sequence[str]) -> Dict[str, dict]:
        """Full quotes for ``EXCHANGE:SYMBOL`` keys, in batches of 500 at one request a second."""
        out: Dict[str, dict] = {}
        keys = list(dict.fromkeys(keys))
        for i in range(0, len(keys), QUOTE_BATCH):
            self._quotes.wait()
            out.update(self.kite.quote(keys[i:i + QUOTE_BATCH]) or {})
        return out

    def basket_margins(self, orders: List[dict]) -> dict:
        """Margin and charges of a basket with hedge benefit, counting open positions (``/margins/basket``)."""
        return self.kite.basket_order_margins(orders, consider_positions=True)

    def positions(self) -> List[dict]:
        """The account's net positions."""
        return (self.kite.positions() or {}).get("net", [])

    # ── orders ──────────────────────────────────────────────────
    def place_limit(self, tradingsymbol: str, side: str, quantity: int, price: float,
                    tag: Optional[str] = None) -> SliceResult:
        """A DAY LIMIT order with autoslice; returns the slice order ids and any slice errors."""
        params = {"exchange": EXCHANGE, "tradingsymbol": tradingsymbol, "transaction_type": side,
                  "quantity": int(quantity), "product": PRODUCT, "order_type": "LIMIT", "price": float(price),
                  "validity": "DAY", "autoslice": "true"}
        if tag:
            from kite_connect.trading.order_status import kite_tag

            params["tag"] = kite_tag(tag)
        self._orders.wait()
        try:
            data = self.kite._post("order.place", url_args={"variety": "regular"}, params=params)
        except Exception as exc:                          # noqa: BLE001 - logged, then raised as OrderError
            self.log("place", params, error=str(exc))
            raise OrderError(f"{side} {quantity} {tradingsymbol} @ {price}: {exc}") from exc
        result = _slices(data)
        self.log("place", params, response=data)
        return result

    def orders(self) -> List[dict]:
        """Today's order book."""
        return self.kite.orders() or []

    def order_history(self, order_id: str) -> List[dict]:
        return self.kite.order_history(order_id)

    def cancel(self, order_id: str) -> None:
        self._orders.wait()
        try:
            response = self.kite.cancel_order(variety="regular", order_id=order_id)
            self.log("cancel", {"order_id": order_id}, response=response)
        except Exception as exc:                          # noqa: BLE001 - already filled or cancelled
            self.log("cancel", {"order_id": order_id}, error=str(exc))

    def log(self, action: str, request: dict, response: Any = None, error: Optional[str] = None,
            mode: str = "live") -> None:
        """Append one record to the JSON-lines order log (every mode, every request)."""
        record = {"at": datetime.now(timezone.utc).isoformat(), "mode": mode, "action": action,
                  "request": request, "response": response, "error": error}
        self.order_log.parent.mkdir(parents=True, exist_ok=True)
        with self.order_log.open("a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        logger.info("order %s [%s] %s -> %s", action, mode, request, error or response)


def _slices(data: Any) -> SliceResult:
    """Kite's place-order data: one object (no slicing) or a list with one entry per slice."""
    entries = data if isinstance(data, list) else [data]
    result = SliceResult()
    for entry in entries:
        if isinstance(entry, dict) and entry.get("order_id"):
            result.order_ids.append(str(entry["order_id"]))
        else:
            err = (entry or {}).get("error", entry) if isinstance(entry, dict) else entry
            result.errors.append(str(err.get("message", err) if isinstance(err, dict) else err))
    return result


def connect(order_log: Path = DEFAULT_ORDER_LOG, limits: LimitsConfig = LimitsConfig(),
            for_orders: bool = False) -> Broker:
    """A Broker on today's stored token.  ``for_orders`` also requires the registered static IP.

    Raises when nobody has logged in since 06:00 IST: tap the link in the
    09:00 / 17:30 IST email, then run again.
    """
    from kite_connect.auth import daily_login

    kite = daily_login.kite_from_stored_token()
    if kite is None:
        raise RuntimeError("no Kite token for today: log in with the link in the daily email (U23), then retry")
    if for_orders:
        ip, ok = daily_login.check_egress()
        if not ok:
            raise RuntimeError(f"orders refused: egress IP {ip} is not the registered static IP "
                               f"({daily_login.ENV_STATIC_IP}); set CENTURION_KITE_PROXY")
    return Broker(kite, order_log=order_log, limits=limits)
