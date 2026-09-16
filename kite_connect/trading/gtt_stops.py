"""
GTT stop-losses for CNC (delivery) holdings on Zerodha Kite.

Plain ``SL`` orders are DAY orders: they expire at the close, so a swing
position is only protected on the day it was opened.  A single-leg SELL GTT
(Good Till Triggered) persists for up to a year and is the only broker-side
stop that survives overnight for CNC equity.

Rules implemented here:

* one active stop GTT per (exchange, symbol) - repeated calls modify the
  existing trigger instead of creating a duplicate;
* trigger rounded to the 0.05 tick; the limit price sits
  ``limit_buffer_pct`` below the trigger (rounded down to a tick);
* a stop above the last traded price cannot be placed (Kite rejects it);
  callers get ``error="stop_breached"`` and should exit instead;
* every placement / modification / deletion is persisted and e-mailed with
  the same hooks as regular orders (``order_service``);
* GTT stops protect existing holdings, so they are allowed while the kill
  switch is active (quantity is checked against holdings in that case);
* :func:`reconcile_stop_gtts` makes sure every CNC holding carries exactly
  one stop at the held quantity and deletes orphan stop GTTs.

Only stop-style GTTs (single-leg, SELL, CNC) are managed; OCO or BUY GTTs a
user creates manually on Kite are never modified or deleted.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)

TICK_SIZE = 0.05
DEFAULT_LIMIT_BUFFER_PCT = float(os.environ.get("CENTURION_GTT_LIMIT_BUFFER_PCT", "1.0"))
GTT_ORDER_TYPE_LABEL = "GTT-SL"

_ACTIVE = "active"
_TERMINAL = ("triggered", "cancelled", "deleted", "expired", "rejected", "disabled")


# ── Price helpers ──────────────────────────────────────────────

def round_to_tick(price: float, tick: float = TICK_SIZE, mode: str = "nearest") -> float:
    """Round ``price`` to the exchange tick (``mode``: nearest | down | up)."""
    if price is None:
        return 0.0
    ticks = float(price) / tick
    if mode == "down":
        n = math.floor(ticks + 1e-9)
    elif mode == "up":
        n = math.ceil(ticks - 1e-9)
    else:
        n = math.floor(ticks + 0.5)
    return round(n * tick, 2)


def stop_limit_price(trigger: float, limit_buffer_pct: Optional[float] = None) -> float:
    """Limit price for a SELL stop: ``limit_buffer_pct`` % below the trigger."""
    pct = DEFAULT_LIMIT_BUFFER_PCT if limit_buffer_pct is None else float(limit_buffer_pct)
    limit = round_to_tick(float(trigger) * (1.0 - pct / 100.0), mode="down")
    return max(limit, TICK_SIZE)


def _kill_switch_active() -> bool:
    active = os.environ.get("CENTURION_KILL_SWITCH", "").lower() in ("true", "1", "yes")
    if not active:
        try:
            from config import Config
            active = bool(getattr(Config, "KILL_SWITCH", False))
        except Exception:
            pass
    return active


def _order_hooks():
    """Return (persist, email) hooks from order_service (import-path agnostic)."""
    try:
        from . import order_service as _os_mod
    except Exception:  # pragma: no cover - loaded outside the package
        try:
            from kite_connect.trading import order_service as _os_mod
        except Exception:
            return None, None
    return _os_mod._persist_to_db, _os_mod._send_order_email


def _notify(symbol, exchange, quantity, trigger, gtt_id, status, error=None):
    persist, email = _order_hooks()
    if persist:
        try:
            persist(symbol, exchange, "SELL", int(quantity), GTT_ORDER_TYPE_LABEL, "CNC",
                    trigger, order_id=gtt_id, success=error is None, error_msg=error,
                    status_text=status)
        except Exception as exc:
            logger.debug("GTT persist failed (non-fatal): %s", exc)
    if email:
        try:
            email(symbol, exchange, "SELL", int(quantity), trigger or 0,
                  str(gtt_id or "-"), status, error=error)
        except Exception as exc:
            logger.debug("GTT email failed (non-fatal): %s", exc)


# ── Broker data helpers ────────────────────────────────────────

def _normalise_gtt(g: Mapping) -> Optional[dict]:
    """Flatten a Kite GTT payload; return None for non-stop GTTs."""
    try:
        cond = g.get("condition") or {}
        orders = g.get("orders") or []
        if str(g.get("type", "")).lower() != "single" or len(orders) != 1:
            return None
        o = orders[0]
        if str(o.get("transaction_type", "")).upper() != "SELL":
            return None
        if str(o.get("product", "")).upper() != "CNC":
            return None
        triggers = cond.get("trigger_values") or []
        return {
            "id": g.get("id"),
            "status": str(g.get("status", "")).lower(),
            "symbol": cond.get("tradingsymbol") or o.get("tradingsymbol"),
            "exchange": cond.get("exchange") or o.get("exchange") or "NSE",
            "trigger": float(triggers[0]) if triggers else 0.0,
            "limit": float(o.get("price") or 0.0),
            "quantity": int(o.get("quantity") or 0),
            "order_result": (o.get("result") or {}).get("order_result") if isinstance(o.get("result"), dict) else None,
            "raw": g,
        }
    except Exception:
        return None


def list_stop_gtts(kite, symbol: Optional[str] = None, exchange: str = "NSE",
                   active_only: bool = True) -> List[dict]:
    """Return stop-style GTTs (single-leg SELL CNC), optionally for one symbol."""
    try:
        raw = kite.get_gtts() or []
    except Exception as exc:
        logger.warning("get_gtts failed: %s", exc)
        raise
    out = []
    for g in raw:
        n = _normalise_gtt(g)
        if n is None:
            continue
        if active_only and n["status"] != _ACTIVE:
            continue
        if symbol and (n["symbol"] != symbol or n["exchange"] != exchange):
            continue
        out.append(n)
    return out


def get_held_quantities(kite, exchange: str = "NSE") -> Dict[str, int]:
    """Sellable CNC quantity per symbol: holdings (incl. T1) + today's net CNC trades."""
    held: Dict[str, int] = {}
    for h in (kite.holdings() or []):
        if (h.get("exchange") or exchange) != exchange:
            continue
        sym = h.get("tradingsymbol")
        qty = int(h.get("quantity", 0) or 0) + int(h.get("t1_quantity", 0) or 0)
        if sym:
            held[sym] = held.get(sym, 0) + qty
    try:
        positions = kite.positions() or {}
        for p in positions.get("net", []) or []:
            if str(p.get("product", "")).upper() != "CNC" or p.get("exchange", exchange) != exchange:
                continue
            sym = p.get("tradingsymbol")
            if sym:
                held[sym] = held.get(sym, 0) + int(p.get("quantity", 0) or 0)
    except Exception as exc:
        logger.debug("positions() failed while computing held quantities: %s", exc)
    return {s: q for s, q in held.items() if q > 0}


def _last_price(kite, symbol: str, exchange: str) -> Optional[float]:
    key = f"{exchange}:{symbol}"
    try:
        data = kite.ltp([key]) or {}
        lp = (data.get(key) or {}).get("last_price")
        return float(lp) if lp else None
    except Exception as exc:
        logger.debug("LTP fetch failed for %s: %s", key, exc)
        return None


# ── Place / modify / delete ────────────────────────────────────

def _payload(symbol, exchange, quantity, trigger, limit_buffer_pct):
    limit = stop_limit_price(trigger, limit_buffer_pct)
    orders = [{
        "exchange": exchange,
        "tradingsymbol": symbol,
        "transaction_type": "SELL",
        "quantity": int(quantity),
        "order_type": "LIMIT",
        "product": "CNC",
        "price": limit,
    }]
    return limit, orders


def place_or_update_stop_gtt(
    kite,
    symbol: str,
    quantity: int,
    trigger_price: float,
    last_price: Optional[float] = None,
    exchange: str = "NSE",
    limit_buffer_pct: Optional[float] = None,
    existing: Optional[List[dict]] = None,
) -> dict:
    """Ensure exactly one active stop GTT for ``symbol`` at ``trigger_price``.

    Returns ``{"success", "action", "trigger_id", "trigger", "limit",
    "quantity", "error"}`` where action is placed | modified | unchanged |
    rejected.  ``existing`` lets a caller pass a pre-fetched GTT list.
    """
    quantity = int(quantity or 0)
    trigger = round_to_tick(float(trigger_price or 0.0))
    base = {"symbol": symbol, "exchange": exchange, "quantity": quantity,
            "trigger": trigger, "trigger_id": None, "limit": None}
    if quantity <= 0 or trigger <= 0:
        return {**base, "success": False, "action": "rejected",
                "error": "quantity and trigger must be positive"}

    if _kill_switch_active():
        # Reduce-only protection is allowed, but never for more than is held.
        try:
            held = get_held_quantities(kite, exchange).get(symbol, 0)
        except Exception as exc:
            return {**base, "success": False, "action": "rejected",
                    "error": f"KILL SWITCH active and holdings check failed: {exc}"}
        if held <= 0:
            return {**base, "success": False, "action": "rejected",
                    "error": "KILL SWITCH active: no holding to protect"}
        if quantity > held:
            logger.warning("Kill switch: capping GTT qty for %s from %d to held %d",
                           symbol, quantity, held)
            quantity = held
            base["quantity"] = quantity

    if last_price is None:
        last_price = _last_price(kite, symbol, exchange)
    if last_price is not None and trigger >= float(last_price):
        msg = f"stop_breached: trigger {trigger:.2f} >= LTP {float(last_price):.2f}"
        logger.warning("GTT stop for %s not placed — %s", symbol, msg)
        return {**base, "success": False, "action": "rejected", "error": "stop_breached",
                "detail": msg, "last_price": last_price}
    lp_for_api = float(last_price) if last_price else trigger

    limit, orders = _payload(symbol, exchange, quantity, trigger, limit_buffer_pct)
    base["limit"] = limit

    try:
        if existing is None:
            current = list_stop_gtts(kite, symbol=symbol, exchange=exchange)
        else:
            current = [g for g in existing
                       if g["symbol"] == symbol and g["exchange"] == exchange and g["status"] == _ACTIVE]
    except Exception as exc:
        return {**base, "success": False, "action": "rejected", "error": f"get_gtts failed: {exc}"}

    # Dedupe: keep the first, delete the rest.
    for extra in current[1:]:
        delete_stop_gtt(kite, extra["id"], symbol=symbol, exchange=exchange,
                        quantity=extra["quantity"], reason="duplicate")
    keep = current[0] if current else None

    try:
        if keep is not None:
            if (keep["quantity"] == quantity and abs(keep["trigger"] - trigger) < TICK_SIZE / 2
                    and abs(keep["limit"] - limit) < TICK_SIZE / 2):
                return {**base, "success": True, "action": "unchanged", "trigger_id": keep["id"],
                        "error": None}
            kite.modify_gtt(
                trigger_id=keep["id"], trigger_type="single", tradingsymbol=symbol,
                exchange=exchange, trigger_values=[trigger], last_price=lp_for_api,
                orders=orders,
            )
            logger.info("GTT stop modified: %s qty=%d trigger %.2f -> %.2f (id=%s)",
                        symbol, quantity, keep["trigger"], trigger, keep["id"])
            _notify(symbol, exchange, quantity, trigger, keep["id"], "GTT MODIFIED")
            return {**base, "success": True, "action": "modified", "trigger_id": keep["id"],
                    "previous_trigger": keep["trigger"], "error": None}

        resp = kite.place_gtt(
            trigger_type="single", tradingsymbol=symbol, exchange=exchange,
            trigger_values=[trigger], last_price=lp_for_api, orders=orders,
        )
        trigger_id = (resp or {}).get("trigger_id") if isinstance(resp, dict) else resp
        logger.info("GTT stop placed: %s qty=%d trigger=%.2f limit=%.2f (id=%s)",
                    symbol, quantity, trigger, limit, trigger_id)
        _notify(symbol, exchange, quantity, trigger, trigger_id, "GTT PLACED")
        return {**base, "success": True, "action": "placed", "trigger_id": trigger_id, "error": None}
    except Exception as exc:
        logger.error("GTT stop placement failed for %s: %s", symbol, exc)
        _notify(symbol, exchange, quantity, trigger, None, "GTT FAILED", error=str(exc))
        return {**base, "success": False, "action": "rejected", "error": str(exc)}


def delete_stop_gtt(kite, trigger_id, symbol: str = "", exchange: str = "NSE",
                    quantity: int = 0, reason: str = "") -> dict:
    """Delete one GTT by id (best-effort persist/email)."""
    try:
        kite.delete_gtt(trigger_id)
        logger.info("GTT stop deleted: %s id=%s (%s)", symbol, trigger_id, reason or "requested")
        _notify(symbol, exchange, quantity, 0.0, trigger_id, f"GTT DELETED {reason}".strip())
        return {"success": True, "trigger_id": trigger_id}
    except Exception as exc:
        logger.warning("GTT delete failed for %s id=%s: %s", symbol, trigger_id, exc)
        return {"success": False, "trigger_id": trigger_id, "error": str(exc)}


def delete_stop_gtts_for_symbol(kite, symbol: str, exchange: str = "NSE", reason: str = "") -> int:
    """Delete every active stop GTT for ``symbol``. Returns the count deleted."""
    try:
        gtts = list_stop_gtts(kite, symbol=symbol, exchange=exchange)
    except Exception:
        return 0
    return sum(1 for g in gtts
               if delete_stop_gtt(kite, g["id"], symbol, exchange, g["quantity"], reason)["success"])


def get_stop_gtt_status(kite, trigger_id) -> Optional[dict]:
    """Return the normalised GTT (any status) for ``trigger_id`` or None."""
    try:
        g = kite.get_gtt(trigger_id)
    except Exception as exc:
        logger.debug("get_gtt(%s) failed: %s", trigger_id, exc)
        return None
    return _normalise_gtt(g or {})


# ── Reconciliation ─────────────────────────────────────────────

def reconcile_stop_gtts(
    kite,
    stops: Optional[Mapping[str, float]] = None,
    exchange: str = "NSE",
    default_stop_pct: Optional[float] = None,
    limit_buffer_pct: Optional[float] = None,
    delete_orphans: bool = True,
) -> dict:
    """Make every CNC holding carry exactly one stop GTT at the held quantity.

    Parameters
    ----------
    stops : {symbol: trigger}
        Desired triggers.  Holdings without an entry keep the trigger of
        their existing GTT; if they have none, ``default_stop_pct`` (percent
        below LTP) is used, otherwise they are reported in ``missing_stop``.
    delete_orphans : bool
        Delete active stop GTTs for symbols that are no longer held.

    Returns a report dict with lists: placed, modified, unchanged, deleted,
    missing_stop, breached, errors.
    """
    stops = dict(stops or {})
    report = {k: [] for k in ("placed", "modified", "unchanged", "deleted",
                              "missing_stop", "breached", "errors")}
    try:
        held = get_held_quantities(kite, exchange)
        gtts = list_stop_gtts(kite, exchange=exchange)
    except Exception as exc:
        report["errors"].append({"symbol": "*", "error": f"broker fetch failed: {exc}"})
        return report

    by_symbol: Dict[str, List[dict]] = {}
    for g in gtts:
        if g["exchange"] == exchange:
            by_symbol.setdefault(g["symbol"], []).append(g)

    ltps: Dict[str, float] = {}
    if held:
        keys = [f"{exchange}:{s}" for s in held]
        try:
            data = kite.ltp(keys) or {}
            ltps = {k.split(":", 1)[1]: float(v.get("last_price") or 0) for k, v in data.items()}
        except Exception as exc:
            logger.debug("Bulk LTP failed during reconciliation: %s", exc)

    for sym, qty in sorted(held.items()):
        existing = by_symbol.get(sym, [])
        trigger = stops.get(sym)
        if trigger is None and existing:
            trigger = existing[0]["trigger"]
        if trigger is None and default_stop_pct and ltps.get(sym):
            trigger = ltps[sym] * (1.0 - float(default_stop_pct) / 100.0)
        if trigger is None or trigger <= 0:
            report["missing_stop"].append(sym)
            logger.warning("GTT reconcile: %s (qty=%d) has no stop and no stop level known", sym, qty)
            continue
        res = place_or_update_stop_gtt(
            kite, sym, qty, trigger, last_price=ltps.get(sym) or None, exchange=exchange,
            limit_buffer_pct=limit_buffer_pct, existing=existing,
        )
        if res["success"]:
            report[res["action"]].append({"symbol": sym, "quantity": qty,
                                          "trigger": res["trigger"], "trigger_id": res["trigger_id"]})
        elif res.get("error") == "stop_breached":
            report["breached"].append({"symbol": sym, "quantity": qty, "trigger": res["trigger"],
                                       "last_price": res.get("last_price")})
        else:
            report["errors"].append({"symbol": sym, "error": res.get("error")})

    if delete_orphans:
        for sym, glist in by_symbol.items():
            if sym in held:
                continue
            for g in glist:
                if delete_stop_gtt(kite, g["id"], sym, exchange, g["quantity"], "orphan")["success"]:
                    report["deleted"].append({"symbol": sym, "trigger_id": g["id"]})

    logger.info(
        "GTT reconcile: placed=%d modified=%d unchanged=%d deleted=%d missing=%d breached=%d errors=%d",
        *(len(report[k]) for k in ("placed", "modified", "unchanged", "deleted",
                                   "missing_stop", "breached", "errors")),
    )
    return report
