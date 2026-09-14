"""
RSI-based Auto-Order Strategy Service for Zerodha Kite Connect.

Scans a watchlist of stocks and calculates 14-period RSI (any candle
interval for analysis).  BUY signals fire when RSI < 30 (oversold) with a
bullish close reversal; SELL signals when RSI > 70 with a bearish reversal.

Auto-placement is long-only CNC swing: daily candles only, BUY via
``order_service`` plus a GTT stop, SELL only exits an existing holding.

Designed to be called from the Streamlit UI or run standalone via CLI.
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# Append (not insert-at-0) so the project-root 'auth' package is not shadowed
_kite_dir = os.path.dirname(os.path.dirname(__file__))
if _kite_dir not in sys.path:
    sys.path.append(_kite_dir)

from kiteconnect import KiteConnect, exceptions as kite_exceptions

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# RSI Calculation
# ═══════════════════════════════════════════════════════════════

def calculate_rsi(candles: list[dict], period: int = 14) -> float:
    """
    Compute RSI from a list of OHLC candle dicts (must contain ``"close"``).

    Uses the smoothed (Wilder) method: first *period* bars are simple average,
    subsequent bars use exponential smoothing  ``avg = (prev * (period-1) + current) / period``.

    Returns
    -------
    float
        RSI value rounded to 2 decimal places, or ``50.0`` if data is insufficient.
    """
    if len(candles) < period + 1:
        return 50.0  # neutral fallback

    gains = 0.0
    losses = 0.0

    # --- seed with first `period` changes ---
    for i in range(period):
        change = candles[i + 1]["close"] - candles[i]["close"]
        if change > 0:
            gains += (change / candles[i]["close"]) * 100
        else:
            losses += (abs(change) / candles[i + 1]["close"]) * 100

    avg_gain = gains / period
    avg_loss = losses / period

    # --- smooth through remaining bars ---
    rsi = _rsi_from_avgs(avg_gain, avg_loss)
    for i in range(period, len(candles) - 1):
        change = candles[i + 1]["close"] - candles[i]["close"]
        if change > 0:
            cur_gain = (change / candles[i]["close"]) * 100
            cur_loss = 0.0
        else:
            cur_gain = 0.0
            cur_loss = (abs(change) / candles[i + 1]["close"]) * 100

        avg_gain = (avg_gain * (period - 1) + cur_gain) / period
        avg_loss = (avg_loss * (period - 1) + cur_loss) / period
        rsi = _rsi_from_avgs(avg_gain, avg_loss)

    return round(rsi, 2)


def _rsi_from_avgs(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ═══════════════════════════════════════════════════════════════
# Signal Detection
# ═══════════════════════════════════════════════════════════════

def detect_signal(candles: list[dict], rsi_low: float = 30, rsi_high: float = 70) -> dict:
    """
    Evaluate RSI + close-reversal conditions on the latest candles.

    Returns
    -------
    dict
        ``{"rsi": float, "signal": "BUY" | "SELL" | None, "close": float, "prev_close": float}``
    """
    rsi = calculate_rsi(candles)
    close_now = candles[-1]["close"]
    close_prev = candles[-2]["close"]

    signal = None
    if rsi < rsi_low and close_now > close_prev:
        signal = "BUY"
    elif rsi > rsi_high and close_now < close_prev:
        signal = "SELL"

    return {
        "rsi": rsi,
        "signal": signal,
        "close": close_now,
        "prev_close": close_prev,
    }


# ═══════════════════════════════════════════════════════════════
# Auto-Order Placement (long-only CNC swing, via order_service)
# ═══════════════════════════════════════════════════════════════
# The mandate is long-only CNC swing/positional.  Auto-placement therefore:
#   * uses DAILY candles only (intraday RSI is analysis-only);
#   * BUY  -> CNC BUY through order_service (kill switch, market hours,
#             idempotency, DB/email hooks) + a GTT stop for the new holding;
#   * SELL -> only exits an EXISTING CNC holding (is_exit=True); never shorts.

AUTO_PLACE_INTERVAL = "day"


def _order_service():
    try:
        from . import order_service as _svc
    except Exception:  # loaded as top-level "trading.rsi_strategy"
        from kite_connect.trading import order_service as _svc
    return _svc


def _gtt_stops():
    try:
        from . import gtt_stops as _g
    except Exception:
        from kite_connect.trading import gtt_stops as _g
    return _g


def compute_sl_and_qty(kite: KiteConnect, symbol: str, side: str,
                       capital: float, max_loss: float) -> dict:
    """
    Derive CNC quantity and stop trigger from capital / max-loss constraints.

    ``qty = floor(capital / LTP)`` (delivery needs full cash) and the stop is
    ``max_loss / qty`` below LTP, rounded down to the 0.05 tick.

    Returns
    -------
    dict
        ``{"qty": int, "trigger_price": float, "last_price": float}``
    """
    quote = kite.quote([f"NSE:{symbol}"])
    ltp = float(quote[f"NSE:{symbol}"]["last_price"])
    qty = int(capital // ltp) if ltp > 0 else 0
    if qty <= 0:
        return {"qty": 0, "trigger_price": 0.0, "last_price": ltp}
    sl_offset = max_loss / qty
    raw = ltp - sl_offset if side == "BUY" else ltp + sl_offset
    trigger = _gtt_stops().round_to_tick(max(raw, 0.05), mode="down" if side == "BUY" else "up")
    return {"qty": qty, "trigger_price": trigger, "last_price": ltp}


def _held_cnc_quantity(kite: KiteConnect, symbol: str) -> int:
    try:
        return int(_gtt_stops().get_held_quantities(kite).get(symbol, 0))
    except Exception as e:
        log.warning("Holdings lookup failed for %s: %s", symbol, e)
        return 0


def place_strategy_order(kite: KiteConnect, symbol: str, side: str,
                         capital: float, max_loss: float,
                         order_type: str = "MARKET") -> dict:
    """
    Long-only CNC order for an RSI signal (routed through ``order_service``).

    * ``BUY``: CNC BUY of ``floor(capital / LTP)`` shares, then a GTT stop at
      ``LTP - max_loss / qty`` for the new holding (best-effort).
    * ``SELL``: exits the existing CNC holding only (``is_exit=True``); with no
      holding nothing is placed (no short selling).

    Returns
    -------
    dict
        ``{"success": bool, "order_id": str | None, "error": str | None,
           "qty": int, "trigger_price": float, "last_price": float, "gtt": dict | None}``
    """
    svc = _order_service()
    empty = {"success": False, "order_id": None, "qty": 0, "trigger_price": 0,
             "last_price": 0, "gtt": None}
    try:
        if side == "SELL":
            held = _held_cnc_quantity(kite, symbol)
            if held <= 0:
                return {**empty, "error": "SELL signal ignored: no CNC holding (long-only)"}
            res = svc.place_order(kite, symbol, "NSE", "SELL", held, order_type="MARKET",
                                  product="CNC", is_exit=True)
            if res.get("success"):
                try:
                    _gtt_stops().delete_stop_gtts_for_symbol(kite, symbol, reason="rsi_exit")
                except Exception:
                    pass
            return {**empty, "success": bool(res.get("success")), "order_id": res.get("order_id"),
                    "error": res.get("error"), "qty": held}

        calc = compute_sl_and_qty(kite, symbol, "BUY", capital, max_loss)
        qty, trigger, ltp = calc["qty"], calc["trigger_price"], calc["last_price"]
        if qty <= 0:
            return {**empty, "error": "Capital below one share", "last_price": ltp}
        price = None
        if order_type == "LIMIT":
            price = ltp
        res = svc.place_order(kite, symbol, "NSE", "BUY", qty, order_type=order_type,
                              product="CNC", price=price)
        out = {**empty, "success": bool(res.get("success")), "order_id": res.get("order_id"),
               "error": res.get("error"), "qty": qty, "trigger_price": trigger, "last_price": ltp}
        if res.get("success") and trigger > 0:
            try:
                out["gtt"] = _gtt_stops().place_or_update_stop_gtt(kite, symbol, qty, trigger,
                                                                   last_price=ltp)
            except Exception as e:
                out["gtt"] = {"success": False, "error": str(e)}
        return out
    except kite_exceptions.InputException as e:
        return {**empty, "error": f"Invalid input: {e}"}
    except Exception as e:
        return {**empty, "error": str(e)}


# ═══════════════════════════════════════════════════════════════
# Scan Watchlist (single pass — call repeatedly from UI / loop)
# ═══════════════════════════════════════════════════════════════

def scan_watchlist(kite: KiteConnect, symbols: list[str],
                   capital: float = 50000, max_loss: float = 500,
                   order_limit: int = 5, order_type: str = "MARKET",
                   rsi_low: float = 30, rsi_high: float = 70,
                   interval: str = "5minute", lookback_days: int = 30,
                   auto_place: bool = False) -> list[dict]:
    """
    Scan a list of symbols for RSI signals and optionally auto-place orders.

    Parameters
    ----------
    kite : KiteConnect
    symbols : list[str]          e.g. ["RELIANCE", "HDFCBANK", ...]
    capital : float              capital per trade (₹)
    max_loss : float             max loss per trade (₹)
    order_limit : int            max number of orders to place in this scan
    order_type : str             "MARKET" or "LIMIT"
    rsi_low : float              oversold threshold (default 30)
    rsi_high : float             overbought threshold (default 70)
    interval : str               candle interval (default "5minute")
    lookback_days : int          days of historical data to fetch
    auto_place : bool            if True, places CNC orders (daily interval only); if False, only scans

    Returns
    -------
    list[dict]
        One entry per symbol with RSI, signal, and order result (if placed).
    """
    results = []
    orders_placed = 0
    auto_allowed = interval == AUTO_PLACE_INTERVAL
    if auto_place and not auto_allowed:
        log.warning("RSI auto-placement disabled for interval=%s (daily candles only)", interval)
    if interval == "day":
        lookback_days = max(lookback_days, 90)  # >= 16 daily candles for RSI(14)
    to_date = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    from_date = (datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d") + " 09:15:00"

    # Pre-fetch instrument tokens
    quotes = {}
    try:
        instruments = [f"NSE:{s}" for s in symbols]
        for i in range(0, len(instruments), 200):
            batch = instruments[i:i + 200]
            quotes.update(kite.quote(batch))
    except Exception as e:
        log.warning("Quote pre-fetch failed: %s", e)

    for symbol in symbols:
        if orders_placed >= order_limit:
            break

        entry = {"symbol": symbol, "rsi": None, "signal": None, "order": None}

        # Get instrument token for historical data
        key = f"NSE:{symbol}"
        if key not in quotes:
            entry["error"] = "Quote not found"
            results.append(entry)
            continue

        token = quotes[key].get("instrument_token")
        if not token:
            entry["error"] = "No instrument token"
            results.append(entry)
            continue

        # Fetch historical candles
        try:
            candles = kite.historical_data(token, from_date, to_date, interval)
        except Exception as e:
            entry["error"] = f"Historical data error: {e}"
            results.append(entry)
            continue

        if len(candles) < 16:
            entry["error"] = "Insufficient candle data"
            results.append(entry)
            continue

        # Compute RSI & detect signal
        sig = detect_signal(candles, rsi_low=rsi_low, rsi_high=rsi_high)
        entry["rsi"] = sig["rsi"]
        entry["signal"] = sig["signal"]
        entry["close"] = sig["close"]
        entry["prev_close"] = sig["prev_close"]
        entry["ltp"] = quotes[key].get("last_price", 0)

        # Place order if signal is active (daily candles only — swing mandate)
        if sig["signal"] and auto_place and not auto_allowed:
            entry["order"] = {"success": False, "order_id": None, "qty": 0, "trigger_price": 0,
                              "error": f"Auto-placement requires daily candles (interval='{AUTO_PLACE_INTERVAL}')"}
        elif sig["signal"] and auto_place:
            result = place_strategy_order(
                kite, symbol, sig["signal"],
                capital=capital, max_loss=max_loss,
                order_type=order_type,
            )
            entry["order"] = result
            if result["success"]:
                orders_placed += 1

        results.append(entry)

    return results


# ═══════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json
    try:
        from kite_connect.auth.kite_session import create_kite_session
    except ImportError:
        from auth.kite_session import create_kite_session
    from core.db_service import get_connection

    logging.basicConfig(level=logging.INFO)

    kite = create_kite_session()
    conn = get_connection()

    # Fetch all stock names from DB
    cur = conn.cursor()
    cur.execute("SELECT name FROM stocks ORDER BY name;")
    symbols = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()

    print(f"Scanning {len(symbols)} stocks for RSI signals...")
    results = scan_watchlist(
        kite, symbols,
        capital=50000, max_loss=500, order_limit=3,
        auto_place=False,  # dry-run by default
    )

    for r in results:
        if r.get("rsi") is not None:
            flag = f"  *** {r['signal']} ***" if r["signal"] else ""
            print(f"  {r['symbol']:>15}  RSI: {r['rsi']:6.2f}  LTP: {r.get('ltp', 0):>10.2f}{flag}")
        elif r.get("error"):
            print(f"  {r['symbol']:>15}  ERROR: {r['error']}")

    print(f"\nDone — {sum(1 for r in results if r.get('signal'))} signals found.")
