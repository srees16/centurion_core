"""
Paper Trading Engine — Virtual Order Simulation for Zerodha Kite.

Zerodha does not provide a native paper-trading API.  This module
implements a local virtual broker that:

1. Accepts trade plans from the AutoExecutor (same interface)
2. Simulates fills using live Kite LTP (or last yfinance close)
3. Applies realistic slippage (Config.SLIPPAGE_MODEL_IND_BPS)
4. Manages a virtual portfolio with SL/TP handling
5. Persists all trades + P&L to a SQLite journal
6. Produces a performance dashboard (daily P&L, drawdown, win rate)

Usage::

    from kite_connect.trading.paper_trader import PaperTrader

    pt = PaperTrader(kite=kite, initial_capital=100_000)
    pt.execute_plans(trade_plans)       # simulate order fills
    pt.poll()                           # check SL/TP (call periodically)
    print(pt.dashboard())               # P&L summary

The scheduler can invoke paper-trading instead of live orders by
setting ``PAPER_TRADE_MODE=true`` in the environment or config.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))
_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "paper_trades.sqlite3"

# Fallback statutory cost model (used only if nse_engine.costs is unavailable):
# per-side STT/exchange/SEBI/stamp/GST ≈ 11 bp, plus a DP charge per sell.
_FALLBACK_BUY_COST_BPS = 11.0
_FALLBACK_SELL_COST_BPS = 11.0
_FALLBACK_DP_CHARGE_INR = 15.93


def statutory_cost_inr(side: str, value_inr: float, as_of=None) -> float:
    """Per-side statutory cost for a CNC equity trade of ``value_inr``.

    Uses ``nse_engine.costs.statutory_cost`` (historical schedule, DP charge on
    sells) when importable, else 11 bp per side + DP charge on sells.
    """
    value_inr = abs(float(value_inr or 0.0))
    if value_inr <= 0:
        return 0.0
    side = str(side).upper()
    try:
        from nse_engine.costs import statutory_cost  # (value_inr, side, date, dp_charge_inr)
        when = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(datetime.now(_IST).date())
        return round(float(statutory_cost(value_inr, side, when)), 2)
    except Exception:
        pass
    bps = _FALLBACK_BUY_COST_BPS if side == "BUY" else _FALLBACK_SELL_COST_BPS
    cost = value_inr * bps / 10_000.0
    if side == "SELL":
        cost += _FALLBACK_DP_CHARGE_INR
    return round(cost, 2)


#: Time stamp of engine fills at the session open (IST).
SESSION_OPEN_TIME = "09:15:00"
PENDING, FILLED, CANCELLED = "PENDING", "FILLED", "CANCELLED"


def session_open_timestamp(session_date) -> str:
    return f"{pd.Timestamp(session_date).date().isoformat()}T{SESSION_OPEN_TIME}+05:30"


def _opened_at_session_open(opened_at: str) -> bool:
    return str(opened_at)[10:19] == "T" + SESSION_OPEN_TIME


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        logger.warning("Invalid %s=%r — using %s", name, os.environ.get(name), default)
        return default


def reality_gap_alerts(result: Optional[dict], min_days: int = 30) -> List[str]:
    """Steady live-vs-backtest cost gaps that distribution tests miss.

    Only for a ``same_period`` report with at least ``min_days`` aligned days:
    annualised tracking error above ``CENTURION_SHIFT_MAX_TRACKING_ERROR``
    (default 0.08) or a mean daily gap below ``-CENTURION_SHIFT_MAX_DAILY_GAP_BPS``
    bp (default 3).
    """
    import math

    if not result or result.get("reference_mode") != "same_period":
        return []
    if int(result.get("n_live") or 0) < min_days:
        return []
    max_te = _env_float("CENTURION_SHIFT_MAX_TRACKING_ERROR", 0.08)
    max_gap_bps = _env_float("CENTURION_SHIFT_MAX_DAILY_GAP_BPS", 3.0)
    alerts: List[str] = []
    te = result.get("tracking_error_annual")
    if te is not None and math.isfinite(float(te)) and float(te) > max_te:
        alerts.append(f"tracking error {float(te):.2%}/yr > {max_te:.2%}")
    gap = result.get("mean_daily_gap")
    if gap is not None and math.isfinite(float(gap)) and float(gap) * 1e4 < -max_gap_bps:
        alerts.append(f"mean daily gap {float(gap) * 1e4:+.2f} bp < -{max_gap_bps:g} bp")
    return alerts


# ═══════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════

@dataclass
class PaperPosition:
    """A single virtual position."""
    symbol: str
    side: str               # BUY
    quantity: int
    entry_price: float      # fill price after slippage
    stop_loss: float
    target_price: float
    opened_at: str = ""
    closed_at: str = ""
    exit_price: float = 0.0
    exit_reason: str = ""   # SL / TP / MANUAL / TRAILING_SL
    pnl: float = 0.0
    pnl_pct: float = 0.0
    is_open: bool = True
    peak_price: float = 0.0  # G5: highest price since entry (for trailing SL)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PaperDashboard:
    """Paper-trading performance summary."""
    initial_capital: float
    current_capital: float
    open_positions: int
    closed_trades: int
    total_pnl: float
    total_pnl_pct: float
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    # Advanced risk metrics (Phase 0)
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0
    cvar_95: float = 0.0
    profit_factor: float = 0.0
    positions: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════
# Paper Trader
# ═══════════════════════════════════════════════════════════════

class PaperTrader:
    """Virtual broker for simulated order execution.

    Parameters
    ----------
    kite : KiteConnect | None
        Authenticated Kite session for live LTP.  If ``None``,
        falls back to yfinance last close.
    initial_capital : float
        Starting virtual capital (default: ₹1,00,000).
    slippage_bps : float | None
        Override slippage in basis points.  Defaults to
        ``Config.SLIPPAGE_MODEL_IND_BPS``.
    """

    def __init__(
        self,
        kite=None,
        initial_capital: float = 100_000.0,
        slippage_bps: Optional[float] = None,
        cloud=None,
    ):
        self.kite = kite
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self._positions: List[PaperPosition] = []
        self._price_overrides: Dict[str, float] = {}  # e.g. latest daily close (mark-to-market)
        self.restored_from: str = "new"               # new | local | cloud

        if slippage_bps is not None:
            self._slippage_bps = slippage_bps
            self._tiered_slippage = False
        else:
            try:
                from config import Config
                self._slippage_bps = getattr(Config, "SLIPPAGE_MODEL_IND_BPS", 20.0)
                self._slip_large = getattr(Config, "SLIPPAGE_IND_LARGECAP_BPS", 5.0)
                self._slip_mid = getattr(Config, "SLIPPAGE_IND_MIDCAP_BPS", 20.0)
                self._slip_small = getattr(Config, "SLIPPAGE_IND_SMALLCAP_BPS", 50.0)
                self._tiered_slippage = True
            except Exception:
                self._slippage_bps = 20.0
                self._tiered_slippage = False

        # Build large-cap / mid-cap symbol sets for tiered slippage
        self._largecap_set: set = set()
        self._midcap_set: set = set()
        if getattr(self, "_tiered_slippage", False):
            try:
                from kite_connect.core.config import INDEX_CONSTITUENTS
                self._largecap_set = set(INDEX_CONSTITUENTS.get("NIFTY50", []))
                self._midcap_set = set(INDEX_CONSTITUENTS.get("NIFTY_NEXT50", []))
            except Exception:
                pass

        self._cloud = cloud  # injected store, else lazy-init cloud sync
        self._init_db()
        self._load_state()

    # ── DB schema ──────────────────────────────────────────────

    def _init_db(self):
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(_DB_PATH))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_positions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol      TEXT NOT NULL,
                side        TEXT NOT NULL,
                quantity    INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss   REAL NOT NULL,
                target_price REAL NOT NULL,
                opened_at   TEXT NOT NULL,
                closed_at   TEXT DEFAULT '',
                exit_price  REAL DEFAULT 0,
                exit_reason TEXT DEFAULT '',
                pnl         REAL DEFAULT 0,
                pnl_pct     REAL DEFAULT 0,
                is_open     INTEGER DEFAULT 1
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_state (
                key   TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        # ── Paper validation checkpoint tables ─────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_snapshots (
                date        TEXT PRIMARY KEY,
                equity      REAL NOT NULL,
                cash        REAL NOT NULL,
                open_positions INTEGER DEFAULT 0,
                closed_today   INTEGER DEFAULT 0,
                day_pnl     REAL DEFAULT 0,
                cumulative_pnl REAL DEFAULT 0,
                cumulative_pnl_pct REAL DEFAULT 0,
                max_drawdown_pct   REAL DEFAULT 0,
                signals_generated  INTEGER DEFAULT 0,
                signals_traded     INTEGER DEFAULT 0,
                snapshot_json TEXT DEFAULT '{}'
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                date        TEXT NOT NULL,
                symbol      TEXT NOT NULL,
                forecast    REAL DEFAULT 0,
                combined_forecast REAL DEFAULT 0,
                action      TEXT DEFAULT '',
                entry_price REAL DEFAULT 0,
                stop_loss   REAL DEFAULT 0,
                target_price REAL DEFAULT 0,
                quantity    INTEGER DEFAULT 0,
                pipeline_sources TEXT DEFAULT '',
                was_traded  INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS weekly_checkpoints (
                week_number INTEGER PRIMARY KEY,
                week_start  TEXT NOT NULL,
                week_end    TEXT NOT NULL,
                start_equity REAL DEFAULT 0,
                end_equity   REAL DEFAULT 0,
                week_return_pct REAL DEFAULT 0,
                trades_opened  INTEGER DEFAULT 0,
                trades_closed  INTEGER DEFAULT 0,
                win_rate    REAL DEFAULT 0,
                sharpe_ratio REAL DEFAULT 0,
                max_dd_pct  REAL DEFAULT 0,
                avg_holding_days REAL DEFAULT 0,
                summary_json TEXT DEFAULT '{}'
            )
        """)
        # NSE engine path: orders decided after a close, filled at the next open
        conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_pending_orders (
                id            TEXT PRIMARY KEY,
                decision_date TEXT NOT NULL,
                symbol        TEXT NOT NULL,
                side          TEXT NOT NULL,
                quantity      INTEGER NOT NULL,
                target_qty    INTEGER NOT NULL,
                ref_price     REAL DEFAULT 0,
                stop_price    REAL DEFAULT 0,
                reason        TEXT DEFAULT '',
                status        TEXT NOT NULL DEFAULT 'PENDING',
                created_at    TEXT NOT NULL,
                resolved_at   TEXT DEFAULT '',
                fill_qty      INTEGER DEFAULT 0,
                fill_price    REAL DEFAULT 0,
                costs_inr     REAL DEFAULT 0,
                note          TEXT DEFAULT ''
            )
        """)
        conn.commit()
        conn.close()

    def _load_state(self):
        """Restore positions and cash from local SQLite, else from the cloud.

        GitHub Actions runs start on a fresh disk, so when local SQLite has
        no state and ``CENTURION_DATABASE_URL`` is set (or a cloud store was
        injected) the book is restored from Neon: cash, initial capital,
        open positions (with stops and entry dates) and equity snapshots.
        """
        conn = sqlite3.connect(str(_DB_PATH))
        conn.row_factory = sqlite3.Row

        # Cash
        row = conn.execute(
            "SELECT value FROM paper_state WHERE key='cash'"
        ).fetchone()
        has_local = row is not None
        if row:
            self.cash = float(row["value"])
        cap_row = conn.execute(
            "SELECT value FROM paper_state WHERE key='initial_capital'"
        ).fetchone()
        if cap_row:
            self.initial_capital = float(cap_row["value"])

        # Open positions
        rows = conn.execute(
            "SELECT * FROM paper_positions WHERE is_open=1"
        ).fetchall()
        for r in rows:
            self._positions.append(PaperPosition(
                symbol=r["symbol"], side=r["side"],
                quantity=r["quantity"], entry_price=r["entry_price"],
                stop_loss=r["stop_loss"], target_price=r["target_price"],
                opened_at=r["opened_at"], is_open=True,
                peak_price=r["entry_price"],
            ))
        has_local = has_local or bool(rows)
        conn.close()

        if has_local:
            self.restored_from = "local"
        elif self._cloud is not None or os.environ.get("CENTURION_DATABASE_URL"):
            try:
                if self._restore_from_cloud():
                    self.restored_from = "cloud"
                else:
                    # Very first run: persist the starting book (sets the cloud epoch)
                    self._save_cash()
            except Exception as exc:
                # Never let a fresh local book overwrite an unreadable cloud book
                logger.error("Paper cloud restore FAILED — cloud sync disabled for this run: %s", exc)
                self._cloud = False
                self.restored_from = "cloud_restore_failed"

        logger.info(
            "Paper trader loaded (%s): cash=%.2f, initial=%.0f, %d open positions",
            self.restored_from, self.cash, self.initial_capital, len(self._positions),
        )

    def _restore_from_cloud(self) -> bool:
        """Rebuild local SQLite state from the cloud store. Returns True if restored."""
        cloud = self._get_cloud()
        if not cloud:
            raise RuntimeError("cloud store configured but unavailable")
        from database.paper_cloud import restore_paper_state
        state = restore_paper_state(cloud)
        if not state:
            return False
        if state.get("initial_capital"):
            self.initial_capital = float(state["initial_capital"])
        if state.get("cash") is not None:
            self.cash = float(state["cash"])
        conn = sqlite3.connect(str(_DB_PATH))
        try:
            for p in state.get("positions", []):
                pos = PaperPosition(
                    symbol=p["symbol"], side=p.get("side", "BUY"),
                    quantity=int(p["quantity"]), entry_price=float(p["entry_price"]),
                    stop_loss=float(p.get("stop_loss") or 0.0),
                    target_price=float(p.get("target_price") or 0.0),
                    opened_at=str(p["opened_at"]), is_open=True,
                    peak_price=float(p.get("peak_price") or p["entry_price"]),
                )
                self._positions.append(pos)
                conn.execute("""
                    INSERT INTO paper_positions
                    (symbol, side, quantity, entry_price, stop_loss, target_price,
                     opened_at, closed_at, exit_price, exit_reason, pnl, pnl_pct, is_open)
                    VALUES (?, ?, ?, ?, ?, ?, ?, '', 0, '', 0, 0, 1)
                """, (pos.symbol, pos.side, pos.quantity, pos.entry_price,
                      pos.stop_loss, pos.target_price, pos.opened_at))
            for s in state.get("closed_positions", []):
                conn.execute("""
                    INSERT INTO paper_positions
                    (symbol, side, quantity, entry_price, stop_loss, target_price,
                     opened_at, closed_at, exit_price, exit_reason, pnl, pnl_pct, is_open)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """, (s["symbol"], s.get("side", "BUY"), int(s["quantity"]), float(s["entry_price"]),
                      float(s.get("stop_loss") or 0), float(s.get("target_price") or 0),
                      str(s["opened_at"]), str(s.get("closed_at") or ""),
                      float(s.get("exit_price") or 0), str(s.get("exit_reason") or ""),
                      float(s.get("pnl") or 0), float(s.get("pnl_pct") or 0)))
            for snap in state.get("snapshots", []):
                conn.execute("""
                    INSERT OR REPLACE INTO daily_snapshots
                    (date, equity, cash, open_positions, closed_today, day_pnl,
                     cumulative_pnl, cumulative_pnl_pct, max_drawdown_pct,
                     signals_generated, signals_traded, snapshot_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (snap["date"], float(snap["equity"]), float(snap.get("cash") or 0),
                      int(snap.get("open_positions") or 0), int(snap.get("closed_today") or 0),
                      float(snap.get("day_pnl") or 0), float(snap.get("cumulative_pnl") or 0),
                      float(snap.get("cumulative_pnl_pct") or 0), float(snap.get("max_drawdown_pct") or 0),
                      int(snap.get("signals_generated") or 0), int(snap.get("signals_traded") or 0),
                      snap.get("snapshot_json") or "{}"))
            conn.execute("INSERT OR REPLACE INTO paper_state (key, value) VALUES ('cash', ?)",
                         (str(self.cash),))
            conn.execute("INSERT OR REPLACE INTO paper_state (key, value) VALUES ('initial_capital', ?)",
                         (str(self.initial_capital),))
            conn.commit()
        finally:
            conn.close()
        self._restore_engine_state_from_cloud(cloud)
        logger.info("Paper state restored from cloud: cash=%.2f, %d open positions, %d snapshots",
                    self.cash, len(self._positions), len(state.get("snapshots", [])))
        return True

    def _save_cash(self):
        conn = sqlite3.connect(str(_DB_PATH))
        conn.execute(
            "INSERT OR REPLACE INTO paper_state (key, value) VALUES ('cash', ?)",
            (str(self.cash),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO paper_state (key, value) VALUES ('initial_capital', ?)",
            (str(self.initial_capital),),
        )
        conn.commit()
        conn.close()
        cloud = self._get_cloud()
        if cloud and hasattr(cloud, "sync_state"):
            cloud.sync_state({"cash": self.cash, "initial_capital": self.initial_capital})

    def _get_cloud(self):
        """Lazy-init cloud sync (best-effort, never blocks)."""
        if self._cloud is None:
            try:
                from database.paper_cloud import get_paper_cloud
                self._cloud = get_paper_cloud()
            except Exception:
                self._cloud = False  # sentinel: don't retry
        return self._cloud if self._cloud else None

    def _save_position(self, pos: PaperPosition):
        conn = sqlite3.connect(str(_DB_PATH))
        conn.execute("""
            INSERT INTO paper_positions
            (symbol, side, quantity, entry_price, stop_loss, target_price,
             opened_at, closed_at, exit_price, exit_reason, pnl, pnl_pct, is_open)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pos.symbol, pos.side, pos.quantity, pos.entry_price,
            pos.stop_loss, pos.target_price, pos.opened_at,
            pos.closed_at, pos.exit_price, pos.exit_reason,
            pos.pnl, pos.pnl_pct, 1 if pos.is_open else 0,
        ))
        conn.commit()
        conn.close()
        # Cloud sync (best-effort)
        cloud = self._get_cloud()
        if cloud:
            cloud.sync_position(pos.to_dict())

    def _close_position_db(self, pos: PaperPosition):
        conn = sqlite3.connect(str(_DB_PATH))
        conn.execute("""
            UPDATE paper_positions SET
                closed_at=?, exit_price=?, exit_reason=?,
                pnl=?, pnl_pct=?, is_open=0
            WHERE symbol=? AND is_open=1 AND opened_at=?
        """, (
            pos.closed_at, pos.exit_price, pos.exit_reason,
            pos.pnl, pos.pnl_pct, pos.symbol, pos.opened_at,
        ))
        conn.commit()
        conn.close()
        # Cloud sync (best-effort)
        cloud = self._get_cloud()
        if cloud:
            cloud.sync_position(pos.to_dict())

    # ── Price helpers ──────────────────────────────────────────

    def _get_ltp(self, symbol: str) -> Optional[float]:
        """Get last traded price from overrides, Kite or yfinance."""
        if symbol in self._price_overrides:
            return float(self._price_overrides[symbol])
        if self.kite:
            try:
                key = f"NSE:{symbol}"
                data = self.kite.ltp([key])
                ltp = data.get(key, {}).get("last_price")
                if ltp and ltp > 0:
                    return float(ltp)
            except Exception:
                pass

        # Fallback: Bhavcopy → yfinance
        try:
            from utils import download_ind_ohlcv
            df = download_ind_ohlcv(symbol, period="5d")
            if not df.empty:
                close = df["Close"].iloc[-1]
                if hasattr(close, "item"):
                    return float(close.item())
                return float(close)
        except Exception:
            pass

        return None

    def _get_base_slippage_bps(self, symbol: str = "") -> float:
        """Return market-cap tiered slippage for *symbol*.

        Large-cap (NIFTY50):   ~5 bps
        Mid-cap (NIFTY_NEXT50): ~20 bps
        Small-cap (others):     ~50 bps
        """
        if not self._tiered_slippage or not symbol:
            return self._slippage_bps
        clean = symbol.replace(".NS", "").upper()
        if clean in self._largecap_set:
            return self._slip_large
        if clean in self._midcap_set:
            return self._slip_mid
        return self._slip_small

    def _apply_slippage(self, price: float, side: str, order_qty: int = 0, adv: float = 0.0, symbol: str = "") -> float:
        """Apply volume-aware, market-cap-tiered slippage.

        base_bps is determined by symbol market-cap tier, then
        impact_bps = order_pct_of_volume × 300  is added.
        Falls back to flat slippage if ADV unknown.
        """
        base_bps = self._get_base_slippage_bps(symbol)
        if adv > 0 and order_qty > 0:
            order_pct = abs(order_qty) / adv
            impact_bps = order_pct * 300.0  # 300 bps impact per 100% of ADV
            total_bps = base_bps + impact_bps
        else:
            total_bps = base_bps
        slip = price * (total_bps / 10_000.0)
        if side == "BUY":
            return round(price + slip, 2)
        return round(price - slip, 2)

    # ── Execution ──────────────────────────────────────────────

    def execute_plans(self, plans: list, skip_held: bool = False) -> List[dict]:
        """Simulate order fills for a list of TradePlan objects.

        Long-only CNC: only BUY plans open positions (SELL plans are
        rejected — exits go through :meth:`close_position`).  ``skip_held``
        ignores plans for symbols already held (repeated intraday runs).
        Statutory buy costs are charged to cash.

        Returns a list of result dicts compatible with OrderResult.
        """
        results = []
        held = {p.symbol for p in self._positions if p.is_open}
        for plan in plans:
            symbol = plan.symbol
            if plan.side != "BUY":
                results.append({"symbol": symbol, "success": False,
                                "error": "Paper CNC is long-only: SELL plans are not opened"})
                continue
            if skip_held and symbol in held:
                results.append({"symbol": symbol, "success": False,
                                "error": "Already held (skip_held)"})
                continue
            ltp = self._get_ltp(symbol)
            if ltp is None:
                results.append({
                    "symbol": symbol, "success": False,
                    "error": "No price available",
                })
                continue

            fill_price = self._apply_slippage(ltp, plan.side, order_qty=plan.quantity, symbol=symbol)
            charges = statutory_cost_inr("BUY", fill_price * plan.quantity)
            cost = fill_price * plan.quantity + charges

            if plan.side == "BUY":
                if cost > self.cash:
                    results.append({
                        "symbol": symbol, "success": False,
                        "error": f"Insufficient capital: need {cost:.0f}, have {self.cash:.0f}",
                    })
                    continue
                self.cash -= cost

            pos = PaperPosition(
                symbol=symbol,
                side=plan.side,
                quantity=plan.quantity,
                entry_price=fill_price,
                stop_loss=plan.stop_loss,
                target_price=plan.target_price,
                opened_at=datetime.now(_IST).isoformat(),
                peak_price=fill_price,  # G5: initialize peak at entry
            )
            self._positions.append(pos)
            held.add(symbol)
            self._save_position(pos)
            self._save_cash()

            logger.info(
                "PAPER %s: %s × %d @ %.2f (slip=%.1fbps, cost=%.0f)",
                plan.side, symbol, plan.quantity, fill_price,
                self._slippage_bps, cost,
            )
            results.append({
                "symbol": symbol,
                "success": True,
                "fill_price": fill_price,
                "quantity": plan.quantity,
                "side": plan.side,
            })

        return results

    # ── SL/TP check (call periodically) ────────────────────────

    def _trail_stop(self, pos: PaperPosition, ltp: float) -> None:
        """G5: Ratchet stop-loss using vol-based trailing stop.

        Uses services.vol_trailing_stop.compute_trailing_stop() which:
          - Scales stop distance by daily volatility (2.5σ swing, 3.5σ positional)
          - Activates profit-lock after 4σ gain (tightens to 1.5σ)
          - Guarantees break-even once profit-lock activates
          - Clamps stop between 2% (min) and 12% (max) of peak

        Falls back to simple 3% percentage trail if vol module unavailable.
        """
        # Update peak price
        if pos.peak_price <= 0:
            pos.peak_price = pos.entry_price
        pos.peak_price = max(pos.peak_price, ltp)

        try:
            from services.vol_trailing_stop import compute_trailing_stop
            from services.instrument_volatility import daily_price_volatility
            from utils import download_ind_ohlcv

            df = download_ind_ohlcv(pos.symbol, period="3mo")
            if df is not None and len(df) >= 20:
                close_series = df["Close"] if "Close" in df.columns else df["close"]
                daily_vol = daily_price_volatility(close_series)
            else:
                daily_vol = 0.02

            try:
                from config import Config
                trade_horizon = getattr(Config, "CARVER_TRADE_HORIZON", "swing")
            except Exception:
                trade_horizon = "swing"

            # Fetch current regime for contra-regime trailing stop
            _paper_regime = ""
            try:
                from services.regime_detector import detect_regime
                _snap = detect_regime()
                if _snap and hasattr(_snap, 'regime'):
                    _paper_regime = str(_snap.regime).lower()
            except Exception:
                pass

            state = compute_trailing_stop(
                entry_price=pos.entry_price,
                current_price=ltp,
                peak_price=pos.peak_price,
                daily_price_vol=daily_vol,
                previous_stop=pos.stop_loss,
                trade_horizon=trade_horizon,
                regime=_paper_regime,
            )
            new_sl = state.current_stop
        except Exception:
            # Fallback: simple 3% trail from peak (only activate after 5% profit)
            profit_pct = (ltp - pos.entry_price) / pos.entry_price
            if profit_pct < 0.05:
                return
            new_sl = round(pos.peak_price * 0.97, 2)

        # Only ratchet UP for LONG positions
        if new_sl > pos.stop_loss:
            old_sl = pos.stop_loss
            pos.stop_loss = round(new_sl, 2)
            # Persist updated SL to DB
            self._update_stop_db(pos)
            logger.debug(
                "PAPER TRAIL SL: %s %.2f → %.2f (peak=%.2f, ltp=%.2f)",
                pos.symbol, old_sl, pos.stop_loss, pos.peak_price, ltp,
            )

    def _update_stop_db(self, pos: PaperPosition) -> None:
        """Persist updated stop_loss to the DB for crash recovery."""
        try:
            conn = sqlite3.connect(str(_DB_PATH))
            conn.execute(
                "UPDATE paper_positions SET stop_loss=? WHERE symbol=? AND is_open=1 AND opened_at=?",
                (pos.stop_loss, pos.symbol, pos.opened_at),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass
        # Cloud sync (best-effort)
        cloud = self._get_cloud()
        if cloud:
            cloud.sync_stop_loss(pos.symbol, pos.opened_at, pos.stop_loss)

    def poll(self) -> List[dict]:
        """Check open positions against SL/TP using live prices.

        G5: Also applies vol-based trailing stop ratcheting before
        checking SL/TP triggers. Returns list of close events.
        """
        events = []
        for pos in self._positions:
            if not pos.is_open:
                continue

            ltp = self._get_ltp(pos.symbol)
            if ltp is None:
                continue

            # G5: Trail stop before checking triggers
            self._trail_stop(pos, ltp)

            closed = False
            reason = ""

            if pos.stop_loss > 0 and ltp <= pos.stop_loss:
                closed = True
                reason = "TRAILING_SL" if pos.stop_loss > pos.entry_price * 0.97 else "SL"
                # GTT semantics: fill at the first observed price if it gapped through
                exit_price = self._apply_slippage(min(ltp, pos.stop_loss), "SELL", symbol=pos.symbol)
            elif pos.target_price > 0 and ltp >= pos.target_price:
                closed = True
                reason = "TP"
                exit_price = self._apply_slippage(pos.target_price, "SELL", symbol=pos.symbol)

            if closed:
                events.append(self._book_close(pos, exit_price, reason))

        return events

    # ── Exits, GTT simulation and engine helpers ───────────────

    def _book_close(self, pos: PaperPosition, exit_price: float, reason: str,
                    when: Optional[str] = None, sell_cost: Optional[float] = None) -> dict:
        """Close ``pos`` at ``exit_price`` net of statutory costs (both sides).

        ``sell_cost`` overrides the per-lot statutory sell cost (engine fills
        charge one order-level cost, pro-rated over the lots sold).
        """
        sell_value = exit_price * pos.quantity
        if sell_cost is None:
            sell_cost = statutory_cost_inr("SELL", sell_value)
        buy_cost = statutory_cost_inr("BUY", pos.entry_price * pos.quantity)
        pos.is_open = False
        pos.exit_price = exit_price
        pos.exit_reason = reason
        pos.closed_at = when or datetime.now(_IST).isoformat()
        pos.pnl = (exit_price - pos.entry_price) * pos.quantity - sell_cost - buy_cost
        basis = pos.entry_price * pos.quantity
        pos.pnl_pct = (pos.pnl / basis * 100) if basis > 0 else 0.0
        self.cash += sell_value - sell_cost
        self._close_position_db(pos)
        self._save_cash()
        logger.info(
            "PAPER CLOSE [%s]: %s @ %.2f → %.2f | P&L=%.2f (%.1f%%) costs=%.2f",
            reason, pos.symbol, pos.entry_price, exit_price, pos.pnl, pos.pnl_pct,
            sell_cost + buy_cost,
        )
        return {
            "type": f"PAPER_{reason}",
            "symbol": pos.symbol,
            "entry": pos.entry_price,
            "exit": exit_price,
            "quantity": pos.quantity,
            "pnl": round(pos.pnl, 2),
            "pnl_pct": round(pos.pnl_pct, 2),
        }

    def simulate_gtt_stops(self, bars: Dict[str, dict], cost_config=None) -> List[dict]:
        """Simulate GTT stop fills from daily bars.

        ``bars``: {symbol: {"date", "open", "low", "close"[, "adv"]}}.  A stop
        triggers when ``low <= stop``; the fill is ``min(open, stop)`` (a gap
        through the stop fills at the open).  Bars dated on or before the
        position's entry date are ignored (the entry happened intraday),
        except lots filled at that session's open (engine path).
        The close is recorded as a price override for mark-to-market.

        With ``cost_config`` (engine path) the exit uses the backtest cost
        model instead of the paper slippage tiers: square-root impact on the
        bar's ``adv`` (uncapped, as backtest stops) and one statutory charge
        (one DP charge) per symbol.
        """
        events = []
        dp_charged: set = set()
        for pos in self._positions:
            if not pos.is_open:
                continue
            bar = bars.get(pos.symbol)
            if not bar:
                continue
            if bar.get("close"):
                self._price_overrides[pos.symbol] = float(bar["close"])
            try:
                bar_date = pd.Timestamp(bar.get("date")).date() if bar.get("date") is not None else None
                opened = datetime.fromisoformat(pos.opened_at.replace("Z", "+00:00")).date()
            except Exception:
                bar_date, opened = None, None
            if bar_date is not None and opened is not None and bar_date <= opened:
                # Intraday entries skip their entry bar; engine lots filled AT
                # the session open are exposed to that session's low (backtest).
                if not (bar_date == opened and _opened_at_session_open(pos.opened_at)):
                    continue
            low = float(bar.get("low") or bar.get("close") or 0.0)
            if pos.stop_loss <= 0 or low <= 0 or low > pos.stop_loss:
                continue
            open_px = float(bar.get("open") or pos.stop_loss)
            raw_fill = min(open_px, pos.stop_loss)
            sell_cost = None
            if cost_config is not None:
                from nse_engine.costs import impact_bps
                bps = impact_bps(raw_fill * pos.quantity, float(bar.get("adv") or float("nan")), cost_config)
                exit_price = round(raw_fill * (1 - bps / 1e4), 4)
                sell_cost = statutory_cost_inr("SELL", exit_price * pos.quantity, as_of=bar_date)
                if pos.symbol in dp_charged:
                    sell_cost = max(sell_cost - float(cost_config.dp_charge_inr), 0.0)
                dp_charged.add(pos.symbol)
            else:
                exit_price = self._apply_slippage(raw_fill, "SELL", symbol=pos.symbol)
            when = f"{bar_date.isoformat()}T09:15:00+05:30" if bar_date else None
            reason = "GTT_SL_GAP" if open_px <= pos.stop_loss else "GTT_SL"
            events.append(self._book_close(pos, exit_price, reason, when=when, sell_cost=sell_cost))
        return events

    def holdings(self) -> Dict[str, dict]:
        """Aggregate open positions per symbol (quantity, avg price, stop, entry date)."""
        out: Dict[str, dict] = {}
        for p in self._positions:
            if not p.is_open:
                continue
            h = out.setdefault(p.symbol, {"quantity": 0, "cost": 0.0, "stop_price": None,
                                          "entry_date": p.opened_at})
            h["quantity"] += p.quantity
            h["cost"] += p.entry_price * p.quantity
            if p.stop_loss:
                h["stop_price"] = max(h["stop_price"] or 0.0, p.stop_loss)
            h["entry_date"] = min(h["entry_date"], p.opened_at)
        for h in out.values():
            h["avg_price"] = h["cost"] / h["quantity"] if h["quantity"] else 0.0
        return out

    def buy(self, symbol: str, quantity: int, price: Optional[float] = None,
            stop_loss: float = 0.0, target_price: float = 0.0, reason: str = "") -> dict:
        """Open a CNC long of ``quantity`` shares (cash + statutory costs checked)."""
        quantity = int(quantity)
        if quantity <= 0:
            return {"symbol": symbol, "success": False, "error": "quantity must be positive"}
        ref = price if price is not None else self._get_ltp(symbol)
        if not ref:
            return {"symbol": symbol, "success": False, "error": "No price available"}
        fill = self._apply_slippage(float(ref), "BUY", order_qty=quantity, symbol=symbol)
        charges = statutory_cost_inr("BUY", fill * quantity)
        total = fill * quantity + charges
        if total > self.cash + 1e-6:
            return {"symbol": symbol, "success": False,
                    "error": f"Insufficient capital: need {total:.0f}, have {self.cash:.0f}"}
        self.cash -= total
        pos = PaperPosition(symbol=symbol, side="BUY", quantity=quantity, entry_price=fill,
                            stop_loss=float(stop_loss or 0.0), target_price=float(target_price or 0.0),
                            opened_at=datetime.now(_IST).isoformat(), peak_price=fill)
        self._positions.append(pos)
        self._save_position(pos)
        self._save_cash()
        logger.info("PAPER BUY %s x %d @ %.2f (costs %.2f) %s", symbol, quantity, fill, charges, reason)
        return {"symbol": symbol, "side": "BUY", "success": True, "fill_price": fill,
                "quantity": quantity, "costs": charges, "reason": reason}

    def close_position(self, symbol: str, quantity: Optional[int] = None,
                       price: Optional[float] = None, reason: str = "EXIT",
                       apply_slippage: bool = True, when: Optional[str] = None,
                       total_sell_cost: Optional[float] = None) -> dict:
        """Sell ``quantity`` (default: all) of ``symbol``, oldest lots first.

        A partial lot sale shrinks the open lot in place and books the sold
        shares as a separate closed row (opened_at = time of the trim).
        Engine fills pass ``apply_slippage=False`` (impact already in
        ``price``), the fill time ``when`` and the order-level statutory cost
        ``total_sell_cost`` (pro-rated over lots, so the DP charge is paid once).
        """
        lots = sorted((p for p in self._positions if p.is_open and p.symbol == symbol),
                      key=lambda p: p.opened_at)
        held = sum(p.quantity for p in lots)
        if held <= 0:
            return {"symbol": symbol, "success": False, "error": "no open position"}
        remaining = held if quantity is None else min(int(quantity), held)
        ref = price if price is not None else self._get_ltp(symbol)
        if not ref:
            return {"symbol": symbol, "success": False, "error": "No price available"}
        exit_price = (self._apply_slippage(float(ref), "SELL", order_qty=remaining, symbol=symbol)
                      if apply_slippage else round(float(ref), 4))
        total_qty = remaining

        def lot_cost(q):
            return None if total_sell_cost is None else float(total_sell_cost) * q / total_qty

        events = []
        sold = 0
        for lot in lots:
            if remaining <= 0:
                break
            if lot.quantity <= remaining:
                remaining -= lot.quantity
                sold += lot.quantity
                events.append(self._book_close(lot, exit_price, reason, when=when,
                                               sell_cost=lot_cost(lot.quantity)))
                continue
            # Partial: shrink the open lot, book the trimmed shares separately
            trim = PaperPosition(symbol=symbol, side=lot.side, quantity=remaining,
                                 entry_price=lot.entry_price, stop_loss=lot.stop_loss,
                                 target_price=lot.target_price,
                                 opened_at=datetime.now(_IST).isoformat(), peak_price=lot.peak_price)
            lot.quantity -= remaining
            self._update_quantity_db(lot)
            self._save_position(trim)
            self._positions.append(trim)
            sold += remaining
            events.append(self._book_close(trim, exit_price, reason, when=when,
                                           sell_cost=lot_cost(remaining)))
            remaining = 0
        return {"symbol": symbol, "side": "SELL", "success": True, "quantity": sold,
                "fill_price": exit_price, "events": events, "reason": reason}

    def recent_stop_exits(self, days: int = 30, as_of=None) -> Dict[str, pd.Timestamp]:
        """{symbol: date of its latest stop exit} within ``days`` (engine cooldown)."""
        end = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(datetime.now(_IST).date())
        cutoff = (end - pd.Timedelta(days=days)).date().isoformat()
        out: Dict[str, pd.Timestamp] = {}
        conn = sqlite3.connect(str(_DB_PATH))
        try:
            rows = conn.execute(
                "SELECT symbol, closed_at FROM paper_positions WHERE is_open=0 AND closed_at >= ? "
                "AND (exit_reason LIKE '%SL%' OR exit_reason LIKE 'EXIT:STOP%')",
                (cutoff,),
            ).fetchall()
        finally:
            conn.close()
        for sym, closed in rows:
            try:
                ts = pd.Timestamp(str(closed)[:10])
            except Exception:
                continue
            if sym not in out or ts > out[sym]:
                out[sym] = ts
        return out

    def set_stop(self, symbol: str, trigger: float) -> int:
        """Set the (GTT-like) stop for every open lot of ``symbol``. Returns lots updated."""
        n = 0
        for p in self._positions:
            if p.is_open and p.symbol == symbol and trigger and trigger > 0:
                p.stop_loss = round(float(trigger), 2)
                self._update_stop_db(p)
                n += 1
        return n

    def _update_quantity_db(self, pos: PaperPosition) -> None:
        try:
            conn = sqlite3.connect(str(_DB_PATH))
            conn.execute(
                "UPDATE paper_positions SET quantity=? WHERE symbol=? AND is_open=1 AND opened_at=?",
                (pos.quantity, pos.symbol, pos.opened_at),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass
        cloud = self._get_cloud()
        if cloud:
            cloud.sync_position(pos.to_dict())

    # ── NSE engine: pending next-open orders ───────────────────
    #
    # The engine decides after the close of session S; its paper orders are
    # stored PENDING and fill at the open of the next session (S+1) with the
    # backtest's cost model (participation cap, square-root impact, statutory
    # charges), sells before buys, buys scaled to the available cash.  An
    # order that has not filled by then is cancelled as stale.  PENDING orders
    # and the last processed session are mirrored to the cloud key/value
    # state (``engine_pending_orders``, ``engine_last_session``) so fresh
    # GitHub Actions runners restore them.

    def pending_orders(self, status: Optional[str] = PENDING) -> List[dict]:
        conn = sqlite3.connect(str(_DB_PATH))
        conn.row_factory = sqlite3.Row
        try:
            if status is None:
                rows = conn.execute("SELECT * FROM paper_pending_orders ORDER BY created_at, id").fetchall()
            else:
                rows = conn.execute("SELECT * FROM paper_pending_orders WHERE status=? ORDER BY created_at, id",
                                    (status,)).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]

    def queue_pending_orders(self, decision_date, orders: List[dict]) -> List[dict]:
        """Store orders decided after the close of ``decision_date`` as PENDING.

        ``orders``: dicts with symbol, side, quantity, target_qty (absolute
        post-trade quantity), ref_price, stop_price, reason.  PENDING orders
        from the same decision date are cancelled first (a re-run supersedes).
        """
        import uuid

        d = pd.Timestamp(decision_date).date().isoformat()
        now = datetime.now(_IST).isoformat()
        conn = sqlite3.connect(str(_DB_PATH))
        rows = []
        try:
            n_old = conn.execute(
                "UPDATE paper_pending_orders SET status=?, resolved_at=?, note=? "
                "WHERE status=? AND decision_date=?",
                (CANCELLED, now, "superseded by a later plan for the same session", PENDING, d),
            ).rowcount
            if n_old:
                logger.info("Pending orders: %d superseded for decision date %s", n_old, d)
            for o in orders:
                row = {
                    "id": f"{d}-{o['side'][0]}-{o['symbol']}-{uuid.uuid4().hex[:8]}",
                    "decision_date": d, "symbol": str(o["symbol"]), "side": str(o["side"]).upper(),
                    "quantity": int(o["quantity"]), "target_qty": int(o["target_qty"]),
                    "ref_price": float(o.get("ref_price") or 0.0),
                    "stop_price": float(o.get("stop_price") or 0.0),
                    "reason": str(o.get("reason") or ""), "status": PENDING, "created_at": now,
                }
                conn.execute(
                    "INSERT INTO paper_pending_orders (id, decision_date, symbol, side, quantity, target_qty, "
                    "ref_price, stop_price, reason, status, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    tuple(row.values()))
                rows.append(row)
            conn.commit()
        finally:
            conn.close()
        for r in rows:
            logger.info("PAPER PENDING %s %s x %d (target %d) decided %s ref=%.2f %s",
                        r["side"], r["symbol"], r["quantity"], r["target_qty"], d, r["ref_price"], r["reason"])
        self._sync_engine_state()
        return rows

    def _resolve_pending(self, order_id: str, status: str, note: str = "", fill_qty: int = 0,
                         fill_price: float = 0.0, costs_inr: float = 0.0, when: Optional[str] = None) -> None:
        conn = sqlite3.connect(str(_DB_PATH))
        try:
            conn.execute(
                "UPDATE paper_pending_orders SET status=?, resolved_at=?, note=?, fill_qty=?, fill_price=?, "
                "costs_inr=? WHERE id=?",
                (status, when or datetime.now(_IST).isoformat(), note, int(fill_qty), float(fill_price),
                 float(costs_inr), order_id))
            conn.commit()
        finally:
            conn.close()

    def engine_last_session(self):
        """Last session whose open/stops were processed by the engine path (date | None)."""
        conn = sqlite3.connect(str(_DB_PATH))
        try:
            row = conn.execute("SELECT value FROM paper_state WHERE key='engine_last_session'").fetchone()
        finally:
            conn.close()
        try:
            return pd.Timestamp(row[0]).date() if row and row[0] else None
        except Exception:
            return None

    def set_engine_last_session(self, session_date) -> None:
        conn = sqlite3.connect(str(_DB_PATH))
        try:
            conn.execute("INSERT OR REPLACE INTO paper_state (key, value) VALUES ('engine_last_session', ?)",
                         (pd.Timestamp(session_date).date().isoformat(),))
            conn.commit()
        finally:
            conn.close()
        self._sync_engine_state()

    def _sync_engine_state(self) -> None:
        cloud = self._get_cloud()
        if not cloud or not hasattr(cloud, "sync_state"):
            return
        pending = [{k: o[k] for k in ("id", "decision_date", "symbol", "side", "quantity", "target_qty",
                                      "ref_price", "stop_price", "reason", "created_at")}
                   for o in self.pending_orders()]
        last = self.engine_last_session()
        cloud.sync_state({"engine_pending_orders": json.dumps(pending),
                          "engine_last_session": last.isoformat() if last else ""})

    def _restore_engine_state_from_cloud(self, cloud) -> None:
        """Restore PENDING engine orders and the last processed session."""
        if not cloud or not hasattr(cloud, "read_state"):
            return
        state = cloud.read_state() or {}
        raw = state.get("engine_pending_orders") or "[]"
        try:
            pending = json.loads(raw)
            if not isinstance(pending, list):
                raise ValueError("not a list")
        except ValueError as exc:
            logger.error("Cloud engine_pending_orders unreadable (%s) — no pending orders restored", exc)
            pending = []
        conn = sqlite3.connect(str(_DB_PATH))
        try:
            for o in pending:
                conn.execute(
                    "INSERT OR REPLACE INTO paper_pending_orders (id, decision_date, symbol, side, quantity, "
                    "target_qty, ref_price, stop_price, reason, status, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (o["id"], o["decision_date"], o["symbol"], o["side"], int(o["quantity"]),
                     int(o["target_qty"]), float(o.get("ref_price") or 0), float(o.get("stop_price") or 0),
                     o.get("reason", ""), PENDING, o.get("created_at") or ""))
            if state.get("engine_last_session"):
                conn.execute("INSERT OR REPLACE INTO paper_state (key, value) VALUES ('engine_last_session', ?)",
                             (state["engine_last_session"],))
            conn.commit()
        finally:
            conn.close()
        if pending:
            logger.info("Restored %d pending engine orders from cloud", len(pending))

    def _open_lot(self, symbol: str, quantity: int, entry_price: float, stop_loss: float,
                  charges: float, when: str, reason: str = "") -> dict:
        total = entry_price * quantity + charges
        self.cash -= total
        pos = PaperPosition(symbol=symbol, side="BUY", quantity=int(quantity), entry_price=round(entry_price, 4),
                            stop_loss=round(float(stop_loss or 0.0), 2), target_price=0.0,
                            opened_at=when, peak_price=round(entry_price, 4))
        self._positions.append(pos)
        self._save_position(pos)
        self._save_cash()
        logger.info("PAPER BUY (open fill) %s x %d @ %.2f (charges %.2f) %s", symbol, quantity, entry_price,
                    charges, reason)
        return {"symbol": symbol, "side": "BUY", "success": True, "fill_price": pos.entry_price,
                "quantity": int(quantity), "costs": charges, "reason": reason}

    def fill_pending_orders(self, session_date, sessions, quotes: Dict[str, dict], cost_config=None, *,
                            min_trade_value_inr: float = 0.0, skip_buys=(), max_age_sessions: int = 1) -> dict:
        """Fill PENDING orders at the open of ``session_date``.

        ``sessions``: trading calendar containing the decision dates and
        ``session_date``.  ``quotes``: {symbol: {"open", "adv"}} where ``adv``
        is the median traded value known at the decision (as the backtest).
        Orders decided at the session before ``session_date`` fill; orders
        decided at ``session_date`` or later stay PENDING; older orders are
        cancelled as stale.  Sells first (participation-capped), then buys
        (capped, skipped when the open is at/below the stop or the symbol is
        in ``skip_buys``), scaled down to the cash available.
        """
        import dataclasses
        import math

        from nse_engine.config import CostConfig
        from nse_engine.costs import simulate_fill

        cfg = cost_config or CostConfig()
        session = pd.Timestamp(session_date).normalize()
        cal = pd.DatetimeIndex(pd.to_datetime(list(sessions))).normalize()
        if session not in cal:
            cal = cal.append(pd.DatetimeIndex([session]))
        cal = cal.unique().sort_values()
        s_pos = int(cal.searchsorted(session, side="left"))
        when = session_open_timestamp(session)
        report = {"session": session.date().isoformat(), "filled": [], "cancelled": [], "kept": []}

        def cancel(o, note, level=logging.INFO):
            self._resolve_pending(o["id"], CANCELLED, note)
            report["cancelled"].append({"id": o["id"], "symbol": o["symbol"], "side": o["side"], "note": note})
            logger.log(level, "PAPER PENDING CANCELLED %s %s x %d (decided %s): %s",
                       o["side"], o["symbol"], o["quantity"], o["decision_date"], note)

        def num(x):
            try:
                v = float(x)
            except (TypeError, ValueError):
                return float("nan")
            return v

        def held(sym):
            return sum(p.quantity for p in self._positions if p.is_open and p.symbol == sym)

        fillable = []
        for o in self.pending_orders():
            d = pd.Timestamp(o["decision_date"]).normalize()
            age = s_pos - (int(cal.searchsorted(d, side="right")) - 1)
            if d >= session or age <= 0:
                report["kept"].append(o["id"])
            elif age <= max_age_sessions:
                fillable.append(o)
            else:
                cancel(o, f"stale: decided {d.date()}, {age} sessions before {session.date()}", logging.WARNING)

        for o in [x for x in fillable if x["side"] == "SELL"]:
            sym = o["symbol"]
            q = quotes.get(sym) or {}
            px, adv = num(q.get("open")), num(q.get("adv"))
            have, tgt = held(sym), max(int(o["target_qty"]), 0)
            diff = have - tgt
            if diff <= 0:
                cancel(o, f"nothing to sell (held {have}, target {tgt})")
                continue
            if not (math.isfinite(px) and px > 0):
                cancel(o, f"no open price on {session.date()}")
                continue
            if tgt > 0 and diff * px < min_trade_value_inr:
                cancel(o, "below min trade value at the open")
                continue
            fill = simulate_fill("SELL", diff, px, adv, session, cfg, apply_cap=True)
            if fill.quantity <= 0:
                cancel(o, "participation cap: no liquidity")
                continue
            exit_px = px * (1 - fill.impact_bps / 1e4)
            res = self.close_position(sym, quantity=fill.quantity, price=exit_px, reason=o["reason"].upper()[:30],
                                      apply_slippage=False, when=when, total_sell_cost=fill.statutory_inr)
            note = f"impact {fill.impact_bps:.1f} bp" + (f"; capped {fill.quantity}/{diff}" if fill.capped else "")
            self._resolve_pending(o["id"], FILLED, note, fill.quantity, exit_px, fill.cost_inr, when)
            report["filled"].append({**res, "id": o["id"], "open": px, "impact_bps": fill.impact_bps,
                                     "costs": fill.cost_inr})

        wanted = []
        for o in [x for x in fillable if x["side"] == "BUY"]:
            sym = o["symbol"]
            if sym in set(skip_buys):
                cancel(o, f"stopped out on {session.date()}")
                continue
            q = quotes.get(sym) or {}
            px, adv = num(q.get("open")), num(q.get("adv"))
            have, tgt = held(sym), int(o["target_qty"])
            diff = tgt - have
            if diff <= 0:
                cancel(o, f"nothing to buy (held {have}, target {tgt})")
                continue
            if not (math.isfinite(px) and px > 0):
                cancel(o, f"no open price on {session.date()}")
                continue
            if diff * px < min_trade_value_inr:
                cancel(o, "below min trade value at the open")
                continue
            lot_stops = [p.stop_loss for p in self._positions if p.is_open and p.symbol == sym and p.stop_loss]
            stop = max(lot_stops) if lot_stops else float(o.get("stop_price") or 0.0)
            if stop and px <= stop:
                cancel(o, f"open {px:.2f} at/below stop {stop:.2f}")
                continue
            fill = simulate_fill("BUY", diff, px, adv, session, cfg)
            if fill.quantity <= 0:
                cancel(o, "participation cap: no liquidity")
                continue
            wanted.append([o, fill, adv, stop, diff])

        need = sum(f.value_inr + f.cost_inr for _, f, *_ in wanted)
        avail = max(float(self.cash), 0.0)
        if need > avail and need > 0:
            factor = avail / need
            for item in wanted:
                o, f, adv = item[0], item[1], item[2]
                item[1] = dataclasses.replace(
                    simulate_fill("BUY", int(math.floor(f.quantity * factor)), f.price, adv, session, cfg,
                                  apply_cap=False), requested_quantity=f.requested_quantity)
            wanted.sort(key=lambda it: -it[1].value_inr)
            while wanted and sum(f.value_inr + f.cost_inr for _, f, *_ in wanted) > avail:
                o, f, adv = wanted[0][0], wanted[0][1], wanted[0][2]
                wanted[0][1] = dataclasses.replace(
                    simulate_fill("BUY", max(f.quantity - 1, 0), f.price, adv, session, cfg, apply_cap=False),
                    requested_quantity=f.requested_quantity)
                if wanted[0][1].quantity <= 0:
                    cancel(o, "insufficient cash at the open")
                    wanted.pop(0)
                wanted.sort(key=lambda it: -it[1].value_inr)
        for o, f, adv, stop, diff in wanted:
            if f.quantity <= 0:
                cancel(o, "insufficient cash at the open")
                continue
            entry_px = f.price * (1 + f.impact_bps / 1e4)
            res = self._open_lot(o["symbol"], f.quantity, entry_px, stop, f.statutory_inr, when, o["reason"])
            note = f"impact {f.impact_bps:.1f} bp" + (f"; filled {f.quantity}/{diff}" if f.quantity < diff else "")
            self._resolve_pending(o["id"], FILLED, note, f.quantity, entry_px, f.cost_inr, when)
            report["filled"].append({**res, "id": o["id"], "open": f.price, "impact_bps": f.impact_bps,
                                     "costs": f.cost_inr})

        if fillable or report["cancelled"]:
            self._sync_engine_state()
        logger.info("Pending orders at open %s: filled=%d cancelled=%d kept=%d", session.date(),
                    len(report["filled"]), len(report["cancelled"]), len(report["kept"]))
        return report

    # ── Dashboard ──────────────────────────────────────────────

    def dashboard(self) -> PaperDashboard:
        """Compute performance dashboard from all paper trades."""
        conn = sqlite3.connect(str(_DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM paper_positions").fetchall()
        conn.close()

        open_positions = []
        closed_trades = []

        for r in rows:
            if r["is_open"]:
                open_positions.append(dict(r))
            else:
                closed_trades.append(dict(r))

        total_pnl = sum(t["pnl"] for t in closed_trades)
        wins = [t for t in closed_trades if t["pnl"] > 0]
        losses = [t for t in closed_trades if t["pnl"] <= 0]

        win_rate = len(wins) / len(closed_trades) if closed_trades else 0.0
        avg_win = (
            sum(t["pnl_pct"] for t in wins) / len(wins)
            if wins else 0.0
        )
        avg_loss = (
            sum(t["pnl_pct"] for t in losses) / len(losses)
            if losses else 0.0
        )

        current_capital = self.cash
        # Mark-to-market open positions
        for pos_dict in open_positions:
            ltp = self._get_ltp(pos_dict["symbol"])
            if ltp:
                current_capital += ltp * pos_dict["quantity"]
            else:
                current_capital += pos_dict["entry_price"] * pos_dict["quantity"]

        total_pnl_pct = (
            (current_capital / self.initial_capital - 1) * 100
            if self.initial_capital > 0 else 0.0
        )

        # Approximate max drawdown from closed trade sequence
        equity_curve = [self.initial_capital]
        for t in sorted(closed_trades, key=lambda x: x.get("closed_at", "")):
            equity_curve.append(equity_curve[-1] + t["pnl"])
        peak = equity_curve[0]
        max_dd = 0.0
        for val in equity_curve:
            if val > peak:
                peak = val
            dd = (peak - val) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        # Risk metrics from the DAILY equity curve (never from per-trade returns:
        # annualising those by sqrt(n_trades) makes "Sharpe" scale with activity).
        trade_returns = [t["pnl_pct"] / 100 for t in closed_trades]
        sr = sortino = calmar = omega = cvar95 = pf = 0.0
        daily = {}
        try:
            daily = self.daily_metrics(conn)
        except Exception as exc:
            logger.debug("Daily risk metrics unavailable: %s", exc)
        if daily:
            sr = float(daily.get("sharpe") or 0.0)
            sortino = float(daily.get("sortino") or 0.0)
            calmar = float(daily.get("calmar") or 0.0)
            if daily.get("max_drawdown") is not None and np.isfinite(daily["max_drawdown"]):
                max_dd = abs(float(daily["max_drawdown"]))  # daily curve beats the trade-sequence estimate
        if len(trade_returns) >= 5:
            try:
                from services.risk_metrics import RiskMetrics
                returns_series = pd.Series(trade_returns)
                # Distribution shape of round trips (no annualisation involved).
                omega = RiskMetrics.omega_ratio(returns_series)
                cvar95 = RiskMetrics.cvar(returns_series, alpha=0.05)
                pf = RiskMetrics.profit_factor(returns_series)
            except Exception as exc:
                logger.debug("Trade distribution metrics unavailable: %s", exc)

        return PaperDashboard(
            initial_capital=self.initial_capital,
            current_capital=round(current_capital, 2),
            open_positions=len(open_positions),
            closed_trades=len(closed_trades),
            total_pnl=round(total_pnl, 2),
            total_pnl_pct=round(total_pnl_pct, 2),
            win_rate=round(win_rate, 4),
            avg_win_pct=round(avg_win, 2),
            avg_loss_pct=round(avg_loss, 2),
            max_drawdown_pct=round(max_dd * 100, 2),
            sharpe_ratio=round(sr, 3),
            sortino_ratio=round(sortino, 3),
            calmar_ratio=round(calmar, 3),
            omega_ratio=round(omega, 3),
            cvar_95=round(cvar95, 4),
            profit_factor=round(pf, 3),
            positions=[p.to_dict() for p in self._positions if p.is_open],
        )

    def reset(self):
        """Reset the paper trading journal (start fresh)."""
        conn = sqlite3.connect(str(_DB_PATH))
        conn.execute("DELETE FROM paper_positions")
        conn.execute("DELETE FROM paper_state")
        conn.execute("DELETE FROM paper_pending_orders")
        conn.commit()
        conn.close()
        self.cash = self.initial_capital
        self._positions.clear()
        logger.info("Paper trader reset: capital=%.2f", self.initial_capital)

    # ── Checkpoint methods for 4-week paper validation ─────────

    def snapshot_daily(self, signals_generated: int = 0, signals_traded: int = 0) -> dict:
        """Save end-of-day equity snapshot for equity curve reconstruction.

        Call this once daily (EOD scheduler job). Even if something crashes
        mid-week, we'll have daily granularity up to the crash point.
        """
        today = datetime.now(_IST).strftime("%Y-%m-%d")
        dashboard = self.dashboard()

        # Count trades closed today
        conn = sqlite3.connect(str(_DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM paper_positions WHERE is_open=0 AND closed_at LIKE ?",
            (today + "%",),
        ).fetchall()
        closed_today = len(rows)
        day_pnl = sum(r["pnl"] for r in rows)

        # Compute running max drawdown from daily_snapshots history
        prev_snapshots = conn.execute(
            "SELECT equity FROM daily_snapshots ORDER BY date"
        ).fetchall()
        equities = [r["equity"] for r in prev_snapshots] + [dashboard.current_capital]
        peak = equities[0] if equities else self.initial_capital
        max_dd = 0.0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        snapshot = {
            "date": today,
            "equity": dashboard.current_capital,
            "cash": self.cash,
            "open_positions": dashboard.open_positions,
            "closed_today": closed_today,
            "day_pnl": round(day_pnl, 2),
            "cumulative_pnl": dashboard.total_pnl,
            "cumulative_pnl_pct": dashboard.total_pnl_pct,
            "max_drawdown_pct": round(max_dd * 100, 2),
            "signals_generated": signals_generated,
            "signals_traded": signals_traded,
            "snapshot_json": json.dumps(dashboard.to_dict()),
        }

        conn.execute("""
            INSERT OR REPLACE INTO daily_snapshots
            (date, equity, cash, open_positions, closed_today, day_pnl,
             cumulative_pnl, cumulative_pnl_pct, max_drawdown_pct,
             signals_generated, signals_traded, snapshot_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            snapshot["date"], snapshot["equity"], snapshot["cash"],
            snapshot["open_positions"], snapshot["closed_today"],
            snapshot["day_pnl"], snapshot["cumulative_pnl"],
            snapshot["cumulative_pnl_pct"], snapshot["max_drawdown_pct"],
            snapshot["signals_generated"], snapshot["signals_traded"],
            snapshot["snapshot_json"],
        ))
        conn.commit()
        conn.close()

        logger.info(
            "Daily snapshot: %s equity=%.0f pnl=%.0f (%.1f%%) dd=%.1f%% open=%d closed=%d",
            today, snapshot["equity"], snapshot["cumulative_pnl"],
            snapshot["cumulative_pnl_pct"], snapshot["max_drawdown_pct"],
            snapshot["open_positions"], closed_today,
        )
        # Cloud sync (best-effort)
        cloud = self._get_cloud()
        if cloud:
            cloud.sync_snapshot(snapshot)

        # ── Distribution shift: live vs backtest (runs once ≥30 live returns) ──
        try:
            shift = self._run_distribution_shift(conn=None)
            if shift:
                snapshot["distribution_shift"] = {
                    k: shift.get(k) for k in (
                        "verdict", "calibrated_verdict", "effective_verdict", "wasserstein",
                        "kl_divergence", "sinkhorn", "p_value_wasserstein", "p_value_kl",
                        "n_live", "reference_mode", "drift_onset",
                        "tracking_error_annual", "mean_daily_gap", "reality_gap_alerts",
                        "position_verdict",
                    )
                }
        except Exception as exc:
            logger.warning("Distribution shift check failed (non-fatal): %s", exc)

        return snapshot

    def _risk_free_annual(self) -> float:
        """Risk-free rate used by the engine, so paper and backtest Sharpes match."""
        try:
            from nse_engine.deployment import load_deployment
            return float(load_deployment().engine.risk_free_annual)
        except Exception:
            try:
                from config import Config
                return float(getattr(Config, "RISK_FREE_RATE_IND", 0.065))
            except Exception:
                return 0.065

    def daily_metrics(self, conn=None) -> Dict[str, float]:
        """Sharpe/Sortino/Calmar/MaxDD from the DAILY equity curve.

        Trade-level returns are not a time series: annualising them by
        sqrt(n_trades) makes the "Sharpe" grow with trade count.  These use
        nse_engine.metrics, the same maths (excess over the risk-free rate,
        ddof=1, calendar-year CAGR) the backtest and the validation gates use.
        """
        _close = False
        if conn is None:
            conn = sqlite3.connect(str(_DB_PATH))
            conn.row_factory = sqlite3.Row
            _close = True
        try:
            returns = self._live_daily_returns(conn)
        finally:
            if _close:
                conn.close()
        if len(returns) < 2:
            return {}
        from nse_engine.metrics import compute_metrics
        equity = (1.0 + returns).cumprod() * float(self.initial_capital)
        return compute_metrics(returns, equity, rf_annual=self._risk_free_annual(),
                               initial_capital=float(self.initial_capital))

    def _live_daily_returns(self, conn) -> pd.Series:
        """Dated daily returns of the paper book from ``daily_snapshots``."""
        rows = conn.execute(
            "SELECT date, equity FROM daily_snapshots ORDER BY date"
        ).fetchall()
        if len(rows) < 2:
            return pd.Series(dtype="float64")
        equity = pd.Series(
            [float(r["equity"]) for r in rows],
            index=pd.DatetimeIndex(pd.to_datetime([r["date"] for r in rows])),
        )
        equity = equity[~equity.index.duplicated(keep="last")]
        return equity.pct_change().dropna()

    def _run_distribution_shift(self, conn=None, min_live_days: int = 30) -> Optional[dict]:
        """Compare live daily returns with the backtest once ``min_live_days`` exist.

        The backtest reference is chosen by
        ``services.distribution_shift.load_backtest_reference``: backtest returns
        for the same dates as the live record when available (e.g.
        data/shift_reference_returns.csv from ``run_nse_engine shift-reference``
        or ``CENTURION_SHIFT_REFERENCE_RUN``), otherwise recent backtest history.

        Writes the full report to ``distribution_shift_latest.json`` and a
        state file with a suggested position-size multiplier (1.0 stable,
        0.75 drifting, 0.5 regime_break) based on the more severe of the
        fixed-threshold and calibrated verdicts; sends an alert on regime_break.

        Reality gap: distribution tests miss a steady cost gap, so a
        ``same_period`` report with >= 30 aligned days also checks the
        annualised tracking error and the mean daily gap
        (:func:`reality_gap_alerts`); a breach alerts and counts as
        "drifting" for the multiplier (``position_verdict``).
        """
        from services.distribution_shift import compare_live_to_backtest, more_severe

        _close_conn = False
        if conn is None:
            conn = sqlite3.connect(str(_DB_PATH))
            conn.row_factory = sqlite3.Row
            _close_conn = True
        try:
            live = self._live_daily_returns(conn)
        finally:
            if _close_conn:
                conn.close()

        if len(live) < min_live_days:
            logger.info("Distribution shift: %d live daily returns (< %d) — not run yet",
                        len(live), min_live_days)
            return None

        result = compare_live_to_backtest(live)
        verdict = result.get("effective_verdict") or result.get("verdict")
        if result.get("reference_mode") == "unavailable":
            logger.info("Distribution shift skipped: no backtest reference returns "
                        "(create data/shift_reference_returns.csv or set CENTURION_SHIFT_REFERENCE_RUN)")
            return result

        onset = result.get("drift_onset") or {}
        logger.info(
            "Distribution shift: %s (thresholds=%s, calibrated=%s) Wasserstein=%.5f KL=%.3f "
            "p=(%s, %s) live=%d days reference=%s%s",
            verdict, result.get("verdict"), result.get("calibrated_verdict"),
            result.get("wasserstein") or 0.0, result.get("kl_divergence") or 0.0,
            result.get("p_value_wasserstein"), result.get("p_value_kl"), result.get("n_live", 0),
            result.get("reference_mode"),
            f" drift since {onset.get('start_date')}" if onset else "",
        )

        gap_alerts = reality_gap_alerts(result, min_days=min_live_days)
        result["reality_gap_alerts"] = gap_alerts
        if result.get("mean_daily_gap") is not None or result.get("tracking_error_annual") is not None:
            logger.info("Reality gap (same period): tracking error=%s/yr mean daily gap=%s bp%s",
                        result.get("tracking_error_annual"),
                        None if result.get("mean_daily_gap") is None else round(result["mean_daily_gap"] * 1e4, 2),
                        f" — ALERT: {'; '.join(gap_alerts)}" if gap_alerts else "")
        if gap_alerts:
            logger.warning("REALITY GAP — %s", "; ".join(gap_alerts))
            verdict = more_severe(verdict, "drifting")
        result["position_verdict"] = verdict

        multiplier = {"stable": 1.0, "drifting": 0.75, "regime_break": 0.5}.get(verdict)
        now = datetime.now(_IST).isoformat()
        try:
            (_DB_PATH.parent / "distribution_shift_latest.json").write_text(
                json.dumps({**result, "updated_at": now}, default=str, indent=2))
            if multiplier is not None:
                (_DB_PATH.parent / "distribution_shift_state.json").write_text(json.dumps({
                    "verdict": verdict,
                    "threshold_verdict": result.get("verdict"),
                    "calibrated_verdict": result.get("calibrated_verdict"),
                    "wasserstein": result.get("wasserstein"),
                    "kl_divergence": result.get("kl_divergence"),
                    "position_size_multiplier": multiplier,
                    "reference_mode": result.get("reference_mode"),
                    "drift_onset": onset or None,
                    "n_live_days": result.get("n_live"),
                    "tracking_error_annual": result.get("tracking_error_annual"),
                    "mean_daily_gap": result.get("mean_daily_gap"),
                    "reality_gap_alerts": gap_alerts,
                    "updated_at": now,
                }, default=str))
        except OSError as exc:
            logger.warning("Could not persist distribution shift state: %s", exc)

        if gap_alerts and verdict != "regime_break":
            try:
                from services.notifications.manager import NotificationManager
                NotificationManager().send_alert(
                    subject="REALITY GAP — live paper returns trail the same-period backtest",
                    body=(
                        f"{'; '.join(gap_alerts)}\n"
                        f"Aligned days: {result.get('n_live')}  reference: {result.get('reference_source')}\n"
                        f"Position-size multiplier: {multiplier} (treated as drifting)."
                    ),
                )
            except Exception as exc:
                logger.debug("Reality-gap alert failed (non-fatal): %s", exc)

        if verdict == "regime_break":
            logger.warning("REGIME BREAK — live returns diverge from the backtest distribution")
            try:
                from services.notifications.manager import NotificationManager
                NotificationManager().send_alert(
                    subject="REGIME BREAK — Distribution Shift Detected",
                    body=(
                        f"Live paper returns diverge from the backtest ({result.get('reference_mode')}).\n"
                        f"Wasserstein={result.get('wasserstein')}  KL={result.get('kl_divergence')}  "
                        f"p=({result.get('p_value_wasserstein')}, {result.get('p_value_kl')})\n"
                        f"Live days: {result.get('n_live')}"
                        + (f"  Drift since: {onset.get('start_date')}" if onset else "")
                        + (f"\nReality gap: {'; '.join(gap_alerts)}" if gap_alerts else "")
                        + "\nReview the strategy before sizing up."
                    ),
                )
            except Exception as exc:
                logger.debug("Regime-break alert failed (non-fatal): %s", exc)
        return result

    def log_signals(self, date_str: str, signals: list) -> None:
        """Persist forecast signals for backtest-vs-live comparison.

        Parameters
        ----------
        date_str : str
            Date string (YYYY-MM-DD).
        signals : list of dict
            Each dict: {symbol, forecast, combined_forecast, action,
                       entry_price, stop_loss, target_price, quantity,
                       pipeline_sources, was_traded}
        """
        if not signals:
            return
        conn = sqlite3.connect(str(_DB_PATH))
        for sig in signals:
            conn.execute("""
                INSERT INTO signal_log
                (date, symbol, forecast, combined_forecast, action,
                 entry_price, stop_loss, target_price, quantity,
                 pipeline_sources, was_traded)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                date_str,
                sig.get("symbol", ""),
                sig.get("forecast", 0),
                sig.get("combined_forecast", 0),
                sig.get("action", ""),
                sig.get("entry_price", 0),
                sig.get("stop_loss", 0),
                sig.get("target_price", 0),
                sig.get("quantity", 0),
                sig.get("pipeline_sources", ""),
                1 if sig.get("was_traded") else 0,
            ))
        conn.commit()
        conn.close()
        logger.debug("Signal log: %d signals for %s", len(signals), date_str)
        # Cloud sync (best-effort)
        cloud = self._get_cloud()
        if cloud:
            cloud.sync_signals(date_str, signals)

    def checkpoint_weekly(self) -> Optional[dict]:
        """Save weekly aggregated checkpoint for crash-resilient analysis.

        Call this once per week (e.g., Friday EOD or Saturday).
        Returns the checkpoint dict, or None if no data.
        """
        conn = sqlite3.connect(str(_DB_PATH))
        conn.row_factory = sqlite3.Row

        # Determine week number from daily snapshots
        snapshots = conn.execute(
            "SELECT * FROM daily_snapshots ORDER BY date"
        ).fetchall()
        if not snapshots:
            conn.close()
            return None

        # Get existing weekly checkpoints to determine current week
        existing_weeks = conn.execute(
            "SELECT week_number FROM weekly_checkpoints ORDER BY week_number DESC LIMIT 1"
        ).fetchone()
        current_week = (existing_weeks["week_number"] + 1) if existing_weeks else 1

        # Get last checkpoint date to determine this week's range
        last_ckpt = conn.execute(
            "SELECT week_end FROM weekly_checkpoints ORDER BY week_number DESC LIMIT 1"
        ).fetchone()
        if last_ckpt:
            week_snapshots = [s for s in snapshots if s["date"] > last_ckpt["week_end"]]
        else:
            week_snapshots = list(snapshots)

        if not week_snapshots:
            conn.close()
            return None

        week_start = week_snapshots[0]["date"]
        week_end = week_snapshots[-1]["date"]
        start_equity = week_snapshots[0]["equity"]
        end_equity = week_snapshots[-1]["equity"]
        week_return = ((end_equity / start_equity) - 1) * 100 if start_equity > 0 else 0

        # Count trades in this week
        trades_opened = conn.execute(
            "SELECT COUNT(*) FROM paper_positions WHERE opened_at >= ? AND opened_at <= ?",
            (week_start, week_end + "T23:59:59"),
        ).fetchone()[0]
        trades_closed = conn.execute(
            "SELECT COUNT(*) FROM paper_positions WHERE closed_at >= ? AND closed_at <= ? AND is_open=0",
            (week_start, week_end + "T23:59:59"),
        ).fetchone()[0]

        # Win rate for this week's closed trades
        week_closed = conn.execute(
            "SELECT pnl FROM paper_positions WHERE closed_at >= ? AND closed_at <= ? AND is_open=0",
            (week_start, week_end + "T23:59:59"),
        ).fetchall()
        wins = sum(1 for r in week_closed if r["pnl"] > 0)
        win_rate = wins / len(week_closed) if week_closed else 0

        # Sharpe from daily returns this week
        import numpy as np
        daily_returns = []
        for i in range(1, len(week_snapshots)):
            prev_eq = week_snapshots[i - 1]["equity"]
            curr_eq = week_snapshots[i]["equity"]
            if prev_eq > 0:
                daily_returns.append(curr_eq / prev_eq - 1)
        if len(daily_returns) >= 2:
            # Excess over the risk-free rate, ddof=1 — same convention as the backtest
            rf_daily = self._risk_free_annual() / 252.0
            excess = np.asarray(daily_returns, dtype=float) - rf_daily
            sd = float(np.std(excess, ddof=1))
            sharpe = float(np.mean(excess) / sd * np.sqrt(252)) if sd > 0 else 0.0
        else:
            sharpe = 0.0

        max_dd = max((s["max_drawdown_pct"] for s in week_snapshots), default=0)

        # Average holding days for closed trades this week
        avg_hold = 0.0
        if week_closed:
            hold_days = []
            for r in conn.execute(
                "SELECT opened_at, closed_at FROM paper_positions WHERE closed_at >= ? AND closed_at <= ? AND is_open=0",
                (week_start, week_end + "T23:59:59"),
            ).fetchall():
                try:
                    opened = datetime.fromisoformat(r["opened_at"].replace("Z", "+00:00"))
                    closed = datetime.fromisoformat(r["closed_at"].replace("Z", "+00:00"))
                    hold_days.append((closed - opened).total_seconds() / 86400)
                except Exception:
                    pass
            if hold_days:
                avg_hold = sum(hold_days) / len(hold_days)

        checkpoint = {
            "week_number": current_week,
            "week_start": week_start,
            "week_end": week_end,
            "start_equity": round(start_equity, 2),
            "end_equity": round(end_equity, 2),
            "week_return_pct": round(week_return, 2),
            "trades_opened": trades_opened,
            "trades_closed": trades_closed,
            "win_rate": round(win_rate, 4),
            "sharpe_ratio": round(sharpe, 3),
            "max_dd_pct": round(max_dd, 2),
            "avg_holding_days": round(avg_hold, 1),
        }

        summary = self.dashboard().to_dict()
        conn.execute("""
            INSERT OR REPLACE INTO weekly_checkpoints
            (week_number, week_start, week_end, start_equity, end_equity,
             week_return_pct, trades_opened, trades_closed, win_rate,
             sharpe_ratio, max_dd_pct, avg_holding_days, summary_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            checkpoint["week_number"], checkpoint["week_start"],
            checkpoint["week_end"], checkpoint["start_equity"],
            checkpoint["end_equity"], checkpoint["week_return_pct"],
            checkpoint["trades_opened"], checkpoint["trades_closed"],
            checkpoint["win_rate"], checkpoint["sharpe_ratio"],
            checkpoint["max_dd_pct"], checkpoint["avg_holding_days"],
            json.dumps(summary),
        ))
        conn.commit()
        conn.close()

        logger.info(
            "Weekly checkpoint W%d: %s→%s equity=%.0f→%.0f ret=%.1f%% "
            "trades=%d/%d wr=%.0f%% sharpe=%.2f dd=%.1f%%",
            current_week, week_start, week_end,
            start_equity, end_equity, week_return,
            trades_opened, trades_closed, win_rate * 100,
            sharpe, max_dd,
        )
        # Cloud sync (best-effort)
        cloud = self._get_cloud()
        if cloud:
            cloud.sync_weekly(checkpoint)
        return checkpoint
