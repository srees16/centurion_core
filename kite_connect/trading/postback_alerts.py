"""Alerts from Kite's order postbacks, within a minute of the event (tracker AL5).

Zerodha POSTs every status change of an order placed through the app to ``/stream/postback``.  Two of them
need the owner before the evening run would report them:

* an order the engine sent (tag ``NE<yymmdd><B|S><symbol>``, ``EngineExecutor.order_tag``) that ends
  REJECTED or CANCELLED: ``CRITICAL``, with Kite's reason and what to do about it (``rejection_hint``);
* a stop that filled: a COMPLETE CNC sell that is the order of a triggered stop GTT: ``WARNING``.  The GTTs
  are read with the day's stored Kite login; when they cannot be read (no login today, Kite down) an untagged
  CNC sell still warns, worded as unconfirmed, because a silent stop is the worse mistake.

Everything else (fills of the engine's own orders, OPEN, manual orders that are not stops) raises nothing:
the nightly email carries it.  Each order alerts once (``alert(once=True)``).  ``handle`` never raises: a
postback is answered whatever happens here.  Whether Kite posts back GTT-triggered and hand-placed orders at
all is settled by the first supervised sell (ST2); if it does not, a filled stop is still found by the
evening session's order book, as before.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Dict, Mapping, Optional

from kite_connect.trading.order_status import order_failed, rejection_hint

logger = logging.getLogger(__name__)

#: The engine's order tag, as ``EngineExecutor.order_tag`` writes it.
ENGINE_TAG = re.compile(r"^NE\d{6}[BS]")
_TRIGGERED = "triggered"

StopLookup = Callable[[], Optional[Dict[str, dict]]]


def is_engine_order(tag) -> bool:
    return bool(ENGINE_TAG.match(str(tag or "")))


def triggered_stop_orders(kite) -> Dict[str, dict]:
    """{order id: stop GTT} for every stop GTT that has triggered and names the order it placed."""
    from kite_connect.trading.gtt_stops import list_stop_gtts

    out: Dict[str, dict] = {}
    for g in list_stop_gtts(kite, active_only=False):
        order_id = str((g.get("order_result") or {}).get("order_id") or "")
        if g.get("status") == _TRIGGERED and order_id:
            out[order_id] = g
    return out


def _stored_login_stops() -> Optional[Dict[str, dict]]:
    """The triggered stops read with today's stored Kite login; None when they cannot be read."""
    try:
        from kite_connect.auth import daily_login as dl
        from kite_connect.trading.live_session import live_book

        kite = dl.kite_from_stored_token(live_book())
        return triggered_stop_orders(kite) if kite is not None else None
    except Exception as exc:                              # noqa: BLE001 - unreadable is "unknown", not "no stop"
        logger.warning("postback: stop GTTs unreadable (%s)", exc)
        return None


def _what(p: Mapping) -> str:
    qty = p.get("filled_quantity") or p.get("quantity") or 0
    return f"{str(p.get('transaction_type') or '').upper()} {qty} {p.get('tradingsymbol') or '?'}"


def classify(p: Mapping, stop_lookup: Optional[StopLookup] = None) -> Optional[dict]:
    """The alert a postback ``p`` deserves ({severity, key, title, lines}) or None.  ``stop_lookup`` returns the
    triggered stops by order id, or None when they cannot be read; it is called only for an untagged CNC sell
    that completed."""
    from services.notifications.alerts import CRITICAL, WARNING

    status = str(p.get("status") or "").strip().upper()
    order_id = str(p.get("order_id") or "")
    side = str(p.get("transaction_type") or "").upper()
    if not order_id or not status:
        return None
    if is_engine_order(p.get("tag")):
        if not order_failed(status):
            return None
        reason = str(p.get("status_message") or "").strip()
        hint = rejection_hint(reason, side)
        lines = [f"Kite status: {status}" + (f" - {reason}" if reason else " (no reason given)"),
                 f"Order {order_id}, tag {p.get('tag')}, {p.get('order_type') or '?'} "
                 f"{p.get('product') or '?'} at {p.get('price') or 0}."]
        if hint:
            lines.append(f"What to do: {hint}.")
        lines.append("The book now differs from tonight's plan by this order; the next session re-plans from "
                     "the broker's holdings.")
        return {"severity": CRITICAL, "key": f"postback_failed:{order_id}",
                "title": f"Centurion order {status.split()[0].lower()}: {_what(p)}", "lines": lines}
    if status != "COMPLETE" or side != "SELL" or str(p.get("product") or "").upper() != "CNC":
        return None
    stops = stop_lookup() if stop_lookup is not None else None
    price = p.get("average_price") or p.get("price") or 0
    if stops is None:
        return {"severity": WARNING, "key": f"postback_stop:{order_id}",
                "title": f"Centurion: a sell filled outside the engine's orders: {_what(p)}",
                "lines": [f"Filled at {price} (order {order_id}).",
                          "Probably a stop: the stop GTTs could not be read to confirm (no Kite login today, or "
                          "Kite unreachable).",
                          "The next session books the exit from the broker's holdings."]}
    gtt = stops.get(order_id)
    if gtt is None:
        return None                                       # a sale by hand: not the engine's, not a stop
    return {"severity": WARNING, "key": f"postback_stop:{order_id}",
            "title": f"Centurion stop filled: {_what(p)}",
            "lines": [f"Filled at {price}; the stop's trigger was {gtt.get('trigger')} (GTT {gtt.get('id')}, "
                      f"order {order_id}).",
                      "The next session books the exit from the broker's holdings; no new stop is needed."]}


def handle(p: Mapping, stop_lookup: Optional[StopLookup] = _stored_login_stops) -> Optional[str]:
    """Raise the alert a postback deserves; the alert's key when one was sent.  Never raises."""
    try:
        found = classify(p, stop_lookup)
        if found is None:
            return None
        from services.notifications.alerts import alert

        sent = alert(found["severity"], found["key"], found["title"], found["lines"], book="live", once=True)
        return found["key"] if sent else None
    except Exception as exc:                              # noqa: BLE001 - the postback is answered regardless
        logger.warning("postback alert failed for order %s: %s", p.get("order_id"), exc)
        return None
