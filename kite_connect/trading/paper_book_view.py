"""Read-only views of the cloud paper book, for the API and the Trade Center page.

The GitHub Actions engine job writes the paper book to Neon (positions, daily
snapshots, pending orders, signals). The API process on Hugging Face must not
build these views through ``PaperTrader``: its local SQLite copy is filled once
from the cloud and then read in preference to it, so the page would freeze on
the first day's numbers. Everything here reads Neon on every call and writes
nothing.

Shapes match what ``app/(dashboard)/ind-stocks/trade-center`` renders:
``MonitoredTradeDetail`` rows for the trades tab and ``PaperDashboard`` fields
for the metrics grid.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

STOP_REASONS = ("SL", "STOP", "GAP")


def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if out == out else default            # NaN guard
    except (TypeError, ValueError):
        return default


def _is_open(value: Any) -> bool:
    return value in (True, 1, "1", "true", "True", "t")


def _records(df) -> List[dict]:
    if df is None or getattr(df, "empty", True):
        return []
    return [{k: (None if (isinstance(v, float) and v != v) else v) for k, v in r.items()}
            for r in df.to_dict("records")]


def risk_free_annual() -> float:
    """The engine's risk-free rate, so paper and backtest Sharpes are comparable."""
    try:
        from nse_engine.deployment import load_deployment
        return float(load_deployment().engine.risk_free_annual)
    except Exception:                                     # noqa: BLE001 - fallback chain
        try:
            from config import Config
            return float(getattr(Config, "RISK_FREE_RATE_IND", 0.065))
        except Exception:                                 # noqa: BLE001
            return 0.065


# ── trades tab ───────────────────────────────────────────────────

def _position_row(p: dict, is_active: bool) -> dict:
    reason = str(p.get("exit_reason") or "").upper()
    side = str(p.get("side") or "BUY").upper()
    return {
        "symbol": p.get("symbol"),
        "side": side,
        "quantity": int(_f(p.get("quantity"))),
        "entry_price": _f(p.get("entry_price")),
        "stop_loss": _f(p.get("stop_loss")),
        "target_price": _f(p.get("target_price")),
        "entry_order_id": f"PAPER-{p.get('id', '')}-{p.get('opened_at', '')}",
        "sl_order_id": None,
        "tp_order_id": None,
        "entry_filled": True,
        "sl_triggered": (not is_active) and any(w in reason for w in STOP_REASONS),
        "tp_triggered": (not is_active) and reason == "TP",
        "closed": not is_active,
        "scaled_2r": False,
        "scaled_3r": False,
        "sl_failed": False,
        "opened_at": p.get("opened_at"),
        "closed_at": p.get("closed_at") or None,
        "exit_price": _f(p.get("exit_price")),
        "exit_reason": p.get("exit_reason") or "",
        "pnl": _f(p.get("pnl")),
        "direction": "LONG" if side == "BUY" else "SHORT",
        "product": "PAPER",
        "is_active": is_active,
        # numeric for the frontend already deployed; unknown = current_price None (G10)
        "unrealised_pnl_pct": 0.0 if is_active else _f(p.get("pnl_pct")),
        "pnl_pct": None if is_active else _f(p.get("pnl_pct")),
    }


def latest_marks(cloud) -> Tuple[Dict[str, float], Optional[str]]:
    """({symbol: price}, date) from the latest snapshot: the prices the book was marked at."""
    try:
        snaps = _records(cloud.read_snapshots())
    except Exception as exc:                              # noqa: BLE001 - marks are optional
        logger.warning("snapshots unreadable: %s", exc)
        return {}, None
    for snap in sorted(snaps, key=lambda s: str(s.get("date") or ""), reverse=True):
        try:
            positions = (json.loads(snap.get("snapshot_json") or "{}") or {}).get("positions") or []
        except ValueError:
            continue
        marks = {str(q.get("symbol")): _f(q.get("last_price")) for q in positions if _f(q.get("last_price")) > 0}
        if marks or not positions:
            return marks, str(snap.get("date") or "")[:10] or None
    return {}, None


def _mark_rows(rows: List[dict], marks: Dict[str, float], source: str, as_of: Optional[str]) -> Dict[str, Any]:
    """Per-position price, value and unrealised P&L (before exit costs), and the totals (G10)."""
    cost = value = pnl = 0.0
    marked = 0
    for r in rows:
        px = marks.get(str(r["symbol"]))
        qty, entry = r["quantity"], r["entry_price"]
        sign = 1.0 if r["direction"] == "LONG" else -1.0
        if px is None or px <= 0 or entry <= 0:
            r.update(current_price=None, market_value=None, unrealised_pnl=None, unrealised_pnl_pct=0.0,
                     mark_source=None, mark_date=None)
            continue
        upnl = sign * (px - entry) * qty
        r.update(current_price=round(px, 2), market_value=round(px * qty, 2), unrealised_pnl=round(upnl, 2),
                 unrealised_pnl_pct=round(sign * (px / entry - 1.0) * 100.0, 2), pnl=round(upnl, 2),
                 mark_source=source, mark_date=as_of)
        cost += entry * qty; value += px * qty; pnl += upnl; marked += 1
    return {"unrealised_pnl": round(pnl, 2), "unrealised_pnl_pct": round(pnl / cost * 100.0, 2) if cost else None,
            "invested_value": round(cost, 2), "market_value": round(value, 2), "marked_positions": marked,
            "marks_source": source if marked else None, "marks_as_of": as_of if marked else None}


def _pending_row(o: dict) -> dict:
    """An order decided at the close, filling at the next open: shown as Pending."""
    side = str(o.get("side") or "BUY").upper()
    return {
        "symbol": o.get("symbol"),
        "side": side,
        "quantity": int(_f(o.get("quantity"))),
        "entry_price": _f(o.get("ref_price")),
        "stop_loss": _f(o.get("stop_price")),
        "target_price": 0.0,
        "entry_order_id": f"PENDING-{o.get('id', '')}",
        "sl_order_id": None,
        "tp_order_id": None,
        "entry_filled": False,
        "sl_triggered": False,
        "tp_triggered": False,
        "closed": False,
        "scaled_2r": False,
        "scaled_3r": False,
        "sl_failed": False,
        "opened_at": o.get("created_at") or o.get("decision_date"),
        "decision_date": o.get("decision_date"),
        "reason": o.get("reason") or "",
        "direction": "LONG" if side == "BUY" else "SHORT",
        "product": "PAPER",
        "is_active": True,
        "unrealised_pnl_pct": 0.0,
        "current_price": None,
    }


def trades_view(cloud, live_prices: Optional[Callable[[List[str]], Dict[str, float]]] = None) -> Dict[str, Any]:
    """Active positions (plus orders pending for the next open) and closed trades.

    Open positions carry their price, value and unrealised P&L (G10): live
    prices when ``live_prices`` (the API's Kite session) returns them, else
    the close the book was last marked at (the latest snapshot).
    """
    positions = _records(cloud.read_positions())
    try:
        state = cloud.read_state() or {}
    except Exception as exc:                              # noqa: BLE001 - state is optional here
        logger.warning("paper state unreadable: %s", exc)
        state = {}
    try:
        pending = json.loads(state.get("engine_pending_orders") or "[]")
        if not isinstance(pending, list):
            pending = []
    except ValueError:
        pending = []

    active = [_position_row(p, True) for p in positions if _is_open(p.get("is_open"))]
    active.sort(key=lambda r: str(r["opened_at"]), reverse=True)
    pending_rows = [_pending_row(o) for o in pending]
    closed = [_position_row(p, False) for p in positions if not _is_open(p.get("is_open"))]
    closed.sort(key=lambda r: str(r.get("closed_at") or ""), reverse=True)

    marks, as_of = latest_marks(cloud)
    source = "close"
    if live_prices is not None and active:
        try:
            live = {k: _f(v) for k, v in (live_prices(sorted({r["symbol"] for r in active})) or {}).items() if _f(v) > 0}
        except Exception as exc:                          # noqa: BLE001 - fall back to the close
            logger.info("live prices unavailable, using the last close: %s", exc)
            live = {}
        if live:
            marks, source, as_of = {**marks, **live}, "live", None
    totals = _mark_rows(active, marks, source, as_of)
    realised = [r["pnl"] for r in closed]
    return {
        "active_trades": pending_rows + active,
        "closed_trades": closed[:200],
        "total_active": len(active),
        "total_pending": len(pending_rows),
        "total_closed": len(closed),
        **totals,
        "realised_pnl": round(sum(realised), 2),
        "realised_wins": sum(1 for x in realised if x > 0),
        "book_owner": state.get("book_owner", ""),
        "epoch": state.get("epoch", ""),
        "source": "cloud",
    }


# ── daily snapshots ──────────────────────────────────────────────

def snapshots_view(cloud) -> List[dict]:
    """Daily snapshots, oldest first, with ``day_pnl`` / ``cumulative_pnl`` taken from equity.

    Paper rows written before 1 Oct 2026 hold the P&L of the trades closed that
    day and in total, which reads as a loss while the book is up (29 Sep:
    -49,839 against equity +0.59%).  Deriving both from the stored equity
    makes those rows agree with the new ones and with the live book.
    """
    snapshots = sorted(_records(cloud.read_snapshots()), key=lambda s: str(s.get("date")))
    if not snapshots:
        return []
    state = cloud.read_state() or {}
    initial = _f(state.get("initial_capital")) or _f(snapshots[0].get("equity"))
    prev = initial
    for s in snapshots:
        equity = _f(s.get("equity"))
        s["day_pnl"] = round(equity - prev, 2)
        s["cumulative_pnl"] = round(equity - initial, 2)
        prev = equity
    return snapshots


# ── metrics grid ─────────────────────────────────────────────────

def _equity_metrics(snapshots: List[dict], initial: float, rf: float) -> Dict[str, float]:
    """Sharpe / Sortino / Calmar / MaxDD from the daily equity curve (same maths as the backtest)."""
    import pandas as pd

    if len(snapshots) < 2:
        return {}
    equity = pd.Series([_f(s.get("equity")) for s in snapshots],
                       index=pd.DatetimeIndex(pd.to_datetime([str(s.get("date")) for s in snapshots])))
    equity = equity[~equity.index.duplicated(keep="last")].sort_index()
    returns = equity.pct_change().dropna()
    if len(returns) < 2:
        return {}
    from nse_engine.metrics import compute_metrics
    return compute_metrics(returns, equity, rf_annual=rf, initial_capital=float(initial) or None)


def journal_points(cloud, rf_annual: Optional[float] = None, min_sessions: int = 20) -> List[dict]:
    """The book's paper record at each week's last session, for the metrics journal (tracker JR1).

    Return and drawdown from the first session; Sharpe only from
    ``min_sessions`` on, since a few weeks' Sharpe has a standard error near 2.
    No Calmar: an annualised return over weeks overstates it many times.
    """
    import pandas as pd

    snapshots = snapshots_view(cloud)
    if len(snapshots) < 2:
        return []
    state = cloud.read_state() or {}
    initial = _f(state.get("initial_capital")) or _f(snapshots[0].get("equity"))
    rf = risk_free_annual() if rf_annual is None else rf_annual
    weeks = [pd.Timestamp(str(s.get("date"))[:10]).isocalendar()[:2] for s in snapshots]
    points = []
    for i, s in enumerate(snapshots):
        if i + 1 < len(snapshots) and weeks[i + 1] == weeks[i]:
            continue
        em = _equity_metrics(snapshots[: i + 1], initial, rf)
        points.append({"date": str(s.get("date"))[:10], "sessions": i + 1,
                       "total_return": _f(s.get("equity")) / initial - 1.0 if initial else None,
                       "max_dd": em.get("max_drawdown"),
                       "sharpe": em.get("sharpe") if i + 1 >= min_sessions else None})
    return points


def _trade_metrics(closed: List[dict]) -> Dict[str, float]:
    import numpy as np
    import pandas as pd

    from services.risk.risk_metrics import RiskMetrics

    out = {"omega_ratio": 0.0, "cvar_95": 0.0, "profit_factor": 0.0}
    rets = pd.Series([_f(t.get("pnl_pct")) / 100.0 for t in closed], dtype="float64")
    if len(rets) >= 2:
        try:
            out["omega_ratio"] = float(RiskMetrics.omega_ratio(rets))
            out["cvar_95"] = float(RiskMetrics.cvar(rets, alpha=0.05))
            out["profit_factor"] = float(RiskMetrics.profit_factor(rets))
        except Exception as exc:                          # noqa: BLE001 - metrics are decorative here
            logger.debug("trade metrics failed: %s", exc)
    return {k: (v if np.isfinite(v) else 0.0) for k, v in out.items()}


def dashboard_view(cloud, rf_annual: Optional[float] = None) -> Dict[str, Any]:
    """``PaperDashboard``-shaped dict built from Neon: equity from the latest snapshot."""
    import numpy as np

    state = cloud.read_state() or {}
    snapshots = sorted(_records(cloud.read_snapshots()), key=lambda s: str(s.get("date")))
    positions = _records(cloud.read_positions())
    open_pos = [p for p in positions if _is_open(p.get("is_open"))]
    closed = [p for p in positions if not _is_open(p.get("is_open"))]

    cash = _f(state.get("cash"))
    initial = _f(state.get("initial_capital")) or (_f(snapshots[0]["equity"]) if snapshots else cash)
    latest = snapshots[-1] if snapshots else None
    if latest is not None:
        current_capital = _f(latest.get("equity"))
    else:
        current_capital = cash + sum(_f(p.get("entry_price")) * _f(p.get("quantity")) for p in open_pos)

    total_pnl = current_capital - initial
    total_pnl_pct = (current_capital / initial - 1.0) * 100.0 if initial > 0 else 0.0
    wins = [t for t in closed if _f(t.get("pnl")) > 0]
    losses = [t for t in closed if _f(t.get("pnl")) <= 0]

    rf = risk_free_annual() if rf_annual is None else rf_annual
    em = _equity_metrics(snapshots, initial, rf)
    tm = _trade_metrics(closed)

    def clean(v: Any) -> float:
        v = _f(v)
        return v if np.isfinite(v) else 0.0

    max_dd = abs(clean(em.get("max_drawdown"))) * 100.0 if em else 0.0
    return {
        "initial_capital": round(initial, 2),
        "current_capital": round(current_capital, 2),
        "cash": round(cash, 2),
        "open_positions": len(open_pos),
        "closed_trades": len(closed),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 3),
        "win_rate": round(len(wins) / len(closed), 4) if closed else 0.0,
        "avg_win_pct": round(sum(_f(t.get("pnl_pct")) for t in wins) / len(wins), 3) if wins else 0.0,
        "avg_loss_pct": round(sum(_f(t.get("pnl_pct")) for t in losses) / len(losses), 3) if losses else 0.0,
        "max_drawdown_pct": round(max_dd, 3),
        "sharpe_ratio": round(clean(em.get("sharpe")), 4),
        "sortino_ratio": round(clean(em.get("sortino")), 4),
        "calmar_ratio": round(clean(em.get("calmar")), 4),
        "cagr_pct": round(clean(em.get("cagr")) * 100.0, 3),
        "omega_ratio": round(tm["omega_ratio"], 4),
        "cvar_95": round(tm["cvar_95"], 5),
        "profit_factor": round(tm["profit_factor"], 4),
        "positions": [{"symbol": p.get("symbol"), "side": p.get("side"),
                       "quantity": int(_f(p.get("quantity"))), "entry_price": _f(p.get("entry_price")),
                       "stop_loss": _f(p.get("stop_loss")), "opened_at": p.get("opened_at")}
                      for p in open_pos],
        "trading_days": len(snapshots),
        "last_snapshot": str(latest.get("date")) if latest else None,
        "book_owner": state.get("book_owner", ""),
        "epoch": state.get("epoch", ""),
        "source": "cloud",
    }
