"""Read-only views of the cloud paper book, for the API and the trade-monitor page.

The GitHub Actions engine job writes the paper book to Neon (positions, daily
snapshots, pending orders, signals). The API process on Hugging Face must not
build these views through ``PaperTrader``: its local SQLite copy is filled once
from the cloud and then read in preference to it, so the page would freeze on
the first day's numbers. Everything here reads Neon on every call and writes
nothing.

Shapes match what ``app/(dashboard)/ind-stocks/trade-monitor`` renders:
``MonitoredTradeDetail`` rows for the trades tab and ``PaperDashboard`` fields
for the metrics grid.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

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
        "unrealised_pnl_pct": 0.0 if is_active else _f(p.get("pnl_pct")),
    }


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
    }


def trades_view(cloud) -> Dict[str, Any]:
    """Active positions (plus orders pending for the next open) and closed trades."""
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
    return {
        "active_trades": pending_rows + active,
        "closed_trades": closed[:200],
        "total_active": len(active),
        "total_pending": len(pending_rows),
        "total_closed": len(closed),
        "book_owner": state.get("book_owner", ""),
        "epoch": state.get("epoch", ""),
        "source": "cloud",
    }


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
