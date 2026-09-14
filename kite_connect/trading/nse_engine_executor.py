"""
NSE engine executor: TargetPortfolio -> CNC orders + GTT stops.

``EngineExecutor.plan()`` has no side effects: it loads market data, asks the
engine for target weights (``nse_engine.engine.generate_targets``) and diffs
them against current holdings x equity.  ``execute(plan)`` sends the orders
through ``order_service`` (live) or ``PaperTrader`` (paper) and arms GTT
stops through ``gtt_stops``.

Real orders are sent only when BOTH ``CENTURION_PAPER_TRADE=false`` and
``CENTURION_NSE_ENGINE_LIVE=true``; otherwise execution is forced to paper
and the reason is logged.

Planning rules (see docs/nse_engine.md):
  * market data is always loaded from the SAME anchor (``config.start``
    minus ``WARMUP_YEARS``, as ``runners/run_nse_engine.py``) so rebalance-day
    counting and expanding normalisers match the backtest;
  * ``generate_targets`` receives holdings (with stop prices), ``equity`` and
    ``stopped_out`` — the engine then applies its own no-trade buffer / min
    trade value and keeps drifted weights on non-rebalance days, so only
    exits, stops and regime changes trade between rebalance days;
  * sells first, then buys; ``TargetPortfolio.exits`` ("stop",
    "forecast_exit", "rank_exit", "sleeve_trend_exit") are sold in full as
    reduce-only SELLs (``is_exit=True``);
  * executor-side no-trade buffer (``apply_buffer``) is OFF by default because
    the engine already buffered with ``equity`` — buffering twice would skip
    the engine's trades to the buffer edge; it can be enabled for target
    functions that do not buffer;
  * trades below ``portfolio.min_trade_value_inr`` are skipped (full exits are
    never skipped);
  * integer shares; buys never exceed available cash (cash + expected sell
    proceeds x ``SELL_PROCEEDS_CREDIT``, net of estimated costs).
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))

# Limit orders are placed this far through the reference (last close) price.
ORDER_LIMIT_BAND_BPS = float(os.environ.get("CENTURION_ENGINE_LIMIT_BAND_BPS", "100"))
# Share of same-day sell proceeds treated as available for buys.
SELL_PROCEEDS_CREDIT = float(os.environ.get("CENTURION_ENGINE_SELL_CREDIT", "1.0"))
# Cost allowance when checking cash for buys (statutory + slippage).
BUY_COST_ALLOWANCE_BPS = 25.0
# Warm-up years loaded before config.start (same anchor as runners/run_nse_engine.py)
WARMUP_YEARS = 2
# Stop-outs within this many calendar days are passed to the engine (cooldown)
STOPPED_OUT_LOOKBACK_DAYS = 30
EXIT_REASONS = ("stop", "forecast_exit", "rank_exit", "sleeve_trend_exit")
TICK = 0.05


def _truthy(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("true", "1", "yes")


def live_orders_allowed() -> Tuple[bool, str]:
    """(allowed, reason). Real orders need CENTURION_PAPER_TRADE=false AND
    CENTURION_NSE_ENGINE_LIVE=true."""
    if _truthy("CENTURION_PAPER_TRADE", "true"):
        return False, "CENTURION_PAPER_TRADE is not 'false'"
    if not _truthy("CENTURION_NSE_ENGINE_LIVE", "false"):
        return False, "CENTURION_NSE_ENGINE_LIVE is not 'true'"
    return True, "live trading enabled by CENTURION_PAPER_TRADE=false and CENTURION_NSE_ENGINE_LIVE=true"


def _tick(price: float, mode: str) -> float:
    n = price / TICK
    n = math.floor(n + 1e-9) if mode == "down" else math.ceil(n - 1e-9)
    return round(max(n, 1) * TICK, 2)


@dataclass
class PlannedOrder:
    symbol: str
    side: str            # BUY | SELL
    quantity: int
    ref_price: float     # last close used for sizing
    limit_price: float   # LIMIT price (band through ref)
    reason: str          # rebalance | entry | exit:<why>
    current_qty: int = 0
    target_qty: int = 0
    target_weight: float = 0.0

    @property
    def value(self) -> float:
        return self.quantity * self.ref_price


@dataclass
class StopInstruction:
    symbol: str
    quantity: int        # expected post-trade quantity
    trigger: float


@dataclass
class ExecutionPlan:
    as_of: pd.Timestamp
    equity: float
    cash: float
    orders: List[PlannedOrder] = field(default_factory=list)
    stop_instructions: List[StopInstruction] = field(default_factory=list)
    skipped: List[dict] = field(default_factory=list)   # {symbol, reason, ...}
    target: Optional[object] = None                     # TargetPortfolio snapshot
    notes: List[str] = field(default_factory=list)

    @property
    def sells(self) -> List[PlannedOrder]:
        return [o for o in self.orders if o.side == "SELL"]

    @property
    def buys(self) -> List[PlannedOrder]:
        return [o for o in self.orders if o.side == "BUY"]

    def to_dict(self) -> dict:
        return {
            "as_of": str(self.as_of), "equity": self.equity, "cash": self.cash,
            "orders": [asdict(o) for o in self.orders],
            "stop_instructions": [asdict(s) for s in self.stop_instructions],
            "skipped": list(self.skipped), "notes": list(self.notes),
        }


class EngineExecutor:
    """Turn engine targets into CNC orders and GTT stops.

    Parameters
    ----------
    kite : KiteConnect | None
    paper : bool
        Paper execution (default).  ``paper=False`` still falls back to paper
        unless :func:`live_orders_allowed` is true.
    config : EngineConfig | None
    target_fn : callable
        ``(data, config, as_of, holdings=...) -> TargetPortfolio``
        (default ``nse_engine.engine.generate_targets``).
    data_loader : callable
        ``(store_dir, start, end, **kw) -> MarketData``
        (default ``nse_engine.data.panel.load_market_data``).
    holdings_fn : callable
        ``() -> (holdings: {symbol: Holding | dict}, cash: float)``.
        Default: PaperTrader book when paper, Kite holdings + margins when live.
    paper_trader : PaperTrader | None
        Book used for paper planning/execution (created lazily).
    stopped_out_fn : callable
        ``() -> {symbol: pd.Timestamp}`` of recent stop exits (cooldown).
        Default: the PaperTrader journal when paper, empty when live.
    apply_buffer : bool
        Apply the executor-side no-trade buffer (default False: the engine
        buffers when given ``equity``).
    """

    def __init__(self, kite=None, paper: bool = True, config=None,
                 target_fn: Optional[Callable] = None, data_loader: Optional[Callable] = None,
                 holdings_fn: Optional[Callable] = None, paper_trader=None,
                 stopped_out_fn: Optional[Callable] = None, apply_buffer: bool = False):
        self.kite = kite
        self.config = config
        self._stopped_out_fn = stopped_out_fn
        self.apply_buffer = bool(apply_buffer)
        self._target_fn = target_fn
        self._data_loader = data_loader
        self._holdings_fn = holdings_fn
        self._paper_trader = paper_trader
        allowed, reason = live_orders_allowed()
        if not paper and not allowed:
            logger.warning("EngineExecutor: live requested but forcing PAPER — %s", reason)
        if not paper and allowed and kite is None:
            logger.warning("EngineExecutor: live allowed but no Kite session — forcing PAPER")
            allowed = False
        self.paper = bool(paper or not allowed)
        self.mode_reason = reason if not self.paper else (
            "paper requested" if paper else f"forced paper: {reason}")

    # ── dependencies ───────────────────────────────────────────

    def _cfg(self):
        if self.config is None:
            from nse_engine.config import EngineConfig
            self.config = EngineConfig()
        return self.config

    def _targets(self):
        if self._target_fn is None:
            from nse_engine.engine import generate_targets
            self._target_fn = generate_targets
        return self._target_fn

    def _loader(self):
        if self._data_loader is None:
            from nse_engine.data.panel import load_market_data
            self._data_loader = load_market_data
        return self._data_loader

    def _pt(self):
        if self._paper_trader is None:
            from kite_connect.trading.paper_trader import PaperTrader
            self._paper_trader = PaperTrader(kite=self.kite)
        return self._paper_trader

    def _current_book(self) -> Tuple[Dict[str, object], float]:
        if self._holdings_fn is not None:
            return self._holdings_fn()
        if self.paper:
            pt = self._pt()
            return pt.holdings(), float(pt.cash)
        from kite_connect.trading.gtt_stops import get_held_quantities, list_stop_gtts
        qty = get_held_quantities(self.kite)
        avg = {h.get("tradingsymbol"): float(h.get("average_price") or 0.0)
               for h in (self.kite.holdings() or [])}
        try:
            stops = {g["symbol"]: g["trigger"] for g in list_stop_gtts(self.kite)}
        except Exception as exc:
            logger.warning("EngineExecutor: GTT stop lookup failed (%s) — holdings without stop prices", exc)
            stops = {}
        margins = self.kite.margins("equity") or {}
        cash = float((margins.get("available") or {}).get("live_balance",
                     (margins.get("available") or {}).get("cash", 0.0)) or 0.0)
        return {s: {"quantity": q, "avg_price": avg.get(s, 0.0), "stop_price": stops.get(s)}
                for s, q in qty.items()}, cash

    def _stopped_out(self, as_of: pd.Timestamp) -> Dict[str, pd.Timestamp]:
        if self._stopped_out_fn is not None:
            return dict(self._stopped_out_fn() or {})
        if not self.paper:
            return {}
        try:
            return self._pt().recent_stop_exits(days=STOPPED_OUT_LOOKBACK_DAYS, as_of=as_of)
        except Exception as exc:
            logger.debug("stopped_out lookup failed: %s", exc)
            return {}

    def _load(self, cfg, end) -> object:
        from datetime import date as _date
        start = _date.fromisoformat(cfg.start)
        anchor = _date(start.year - WARMUP_YEARS, 1, 1)
        return self._loader()(
            cfg.data.store_dir, anchor.isoformat(), pd.Timestamp(end).date().isoformat(),
            series=cfg.data.series, min_median_value_inr=cfg.data.load_min_median_value_inr,
            float_dtype=cfg.data.float_dtype,
            include_symbols=(cfg.sleeves.gold_symbol, cfg.sleeves.silver_symbol),
        )

    # ── planning ───────────────────────────────────────────────

    def plan(self, as_of=None) -> ExecutionPlan:
        cfg = self._cfg()
        pcfg = cfg.portfolio
        raw_holdings, cash = self._current_book()
        from nse_engine.types import Holding, holdings_from_mapping
        holdings: Dict[str, Holding] = {}
        plain = {}
        for sym, h in (raw_holdings or {}).items():
            if isinstance(h, Holding):
                holdings[sym] = h
            else:
                plain[sym] = h
        holdings.update(holdings_from_mapping(plain))

        end = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(datetime.now(_IST).date())
        data = self._load(cfg, end)
        if as_of is None:
            as_of = data.dates[-1]
        as_of = pd.Timestamp(as_of)
        plan = ExecutionPlan(as_of=as_of, equity=0.0, cash=float(cash))

        # Session freshness: never trade on stale data
        try:
            from services.carver_pipeline import last_completed_nse_session
            expected = last_completed_nse_session()
            if as_of is not None and pd.Timestamp(data.dates[-1]).date() < expected and self._as_of_is_live(as_of):
                plan.notes.append(f"stale data: last bar {data.dates[-1].date()} < session {expected}")
                logger.warning("EngineExecutor: %s — no orders planned", plan.notes[-1])
                plan.skipped.append({"symbol": "*", "reason": "stale_data"})
                return plan
        except Exception as exc:
            logger.debug("Freshness check unavailable: %s", exc)

        view = data.until(as_of) if hasattr(data, "until") else data
        # Last available close per symbol (forward-filled, as the engine marks equity)
        closes = view.close.ffill().iloc[-1] if len(view.close) else pd.Series(dtype=float)
        prices: Dict[str, float] = {}
        for sym in closes.index:
            px = closes.get(sym)
            if px is not None and not pd.isna(px) and float(px) > 0:
                prices[sym] = float(px)
        equity = float(cash) + sum(h.quantity * prices.get(s, h.avg_price) for s, h in holdings.items())
        plan.equity = equity

        target = self._targets()(view, cfg, as_of, holdings=holdings,
                                 equity=equity, stopped_out=self._stopped_out(as_of))
        plan.target = target
        plan.notes.extend(str(n) for n in (target.notes or []))
        symbols = list(dict.fromkeys(list(holdings) + list(target.weights) + list(target.exits)))

        sells: List[PlannedOrder] = []
        buys: List[PlannedOrder] = []
        post_qty: Dict[str, int] = {s: h.quantity for s, h in holdings.items()}

        for sym in symbols:
            cur = holdings[sym].quantity if sym in holdings else 0
            px = prices.get(sym)
            if px is None:
                plan.skipped.append({"symbol": sym, "reason": "no_price"})
                continue
            exit_reason = target.exits.get(sym)
            w = 0.0 if exit_reason else float(target.weights.get(sym, 0.0))
            target_value = w * equity
            # small epsilon: drifted weights (w = qty*px/equity) must map back to qty exactly
            tgt_qty = int(math.floor(target_value / px + 1e-6)) if w > 0 else 0
            delta = tgt_qty - cur
            if delta == 0:
                continue
            full_exit = tgt_qty == 0 and cur > 0
            if not full_exit:
                if (self.apply_buffer and cur > 0 and w > 0
                        and abs(tgt_qty * px - cur * px) <= pcfg.no_trade_buffer * target_value):
                    plan.skipped.append({"symbol": sym, "reason": "within_buffer",
                                         "current_qty": cur, "target_qty": tgt_qty})
                    continue
                if abs(delta) * px < pcfg.min_trade_value_inr:
                    plan.skipped.append({"symbol": sym, "reason": "below_min_trade_value",
                                         "value": round(abs(delta) * px, 2)})
                    continue
            if delta < 0:
                reason = f"exit:{exit_reason}" if exit_reason else ("exit:weight_zero" if full_exit else "rebalance")
                sells.append(PlannedOrder(sym, "SELL", -delta, px, _tick(px * (1 - ORDER_LIMIT_BAND_BPS / 1e4), "down"),
                                          reason, cur, tgt_qty, w))
                post_qty[sym] = tgt_qty
            else:
                buys.append(PlannedOrder(sym, "BUY", delta, px, _tick(px * (1 + ORDER_LIMIT_BAND_BPS / 1e4), "up"),
                                         "entry" if cur == 0 else "rebalance", cur, tgt_qty, w))

        # Cash budget: never exceed available cash
        est_sell_costs = sum(o.value * 0.0011 + 16.0 for o in sells)
        budget = float(cash) + SELL_PROCEEDS_CREDIT * (sum(o.value for o in sells) - est_sell_costs)
        buys.sort(key=lambda o: (-o.target_weight, o.symbol))
        kept: List[PlannedOrder] = []
        for o in buys:
            unit = o.limit_price * (1 + BUY_COST_ALLOWANCE_BPS / 1e4)
            affordable = int(math.floor(max(budget, 0.0) / unit)) if unit > 0 else 0
            if affordable < o.quantity:
                if affordable <= 0 or affordable * o.ref_price < pcfg.min_trade_value_inr:
                    plan.skipped.append({"symbol": o.symbol, "reason": "insufficient_cash",
                                         "wanted_qty": o.quantity, "budget": round(budget, 2)})
                    continue
                plan.notes.append(f"{o.symbol}: qty cut {o.quantity}->{affordable} (cash)")
                o.quantity = affordable
            budget -= o.quantity * unit
            post_qty[o.symbol] = o.current_qty + o.quantity
            kept.append(o)

        plan.orders = sells + kept   # sells first
        for sym, trig in (target.stops or {}).items():
            q = int(post_qty.get(sym, 0))
            if q > 0 and trig and trig > 0:
                plan.stop_instructions.append(StopInstruction(sym, q, round(float(trig), 2)))
        logger.info("EngineExecutor plan %s: equity=%.0f cash=%.0f sells=%d buys=%d stops=%d skipped=%d",
                    as_of.date(), equity, cash, len(sells), len(kept),
                    len(plan.stop_instructions), len(plan.skipped))
        return plan

    @staticmethod
    def _as_of_is_live(as_of) -> bool:
        """Freshness applies to live-date planning, not historical replays."""
        return pd.Timestamp(as_of).date() >= (datetime.now(_IST).date() - timedelta(days=7))

    # ── execution ──────────────────────────────────────────────

    def execute(self, plan: ExecutionPlan) -> List[dict]:
        allowed, reason = live_orders_allowed()
        paper = self.paper or not allowed or self.kite is None
        if not self.paper and paper:
            logger.warning("EngineExecutor.execute: forcing PAPER — %s", reason if not allowed else "no Kite session")
        return self._execute_paper(plan) if paper else self._execute_live(plan)

    def _execute_paper(self, plan: ExecutionPlan) -> List[dict]:
        pt = self._pt()
        results: List[dict] = []
        overrides = getattr(pt, "_price_overrides", {}) or {}
        for o in plan.orders:
            # Live LTP when a Kite session exists, else the latest known close
            px = None if pt.kite else float(overrides.get(o.symbol, o.ref_price))
            if o.side == "SELL":
                res = pt.close_position(o.symbol, quantity=o.quantity, price=px,
                                        reason=o.reason.upper()[:30])
            else:
                res = pt.buy(o.symbol, o.quantity, price=px, reason=o.reason)
            if not res.get("success") and "No price" in str(res.get("error")):
                if o.side == "SELL":
                    res = pt.close_position(o.symbol, quantity=o.quantity, price=o.ref_price,
                                            reason=o.reason.upper()[:30])
                else:
                    res = pt.buy(o.symbol, o.quantity, price=o.ref_price, reason=o.reason)
            results.append({"mode": "paper", "symbol": o.symbol, "side": o.side,
                            "quantity": o.quantity, "reason": o.reason, **res})
        for s in plan.stop_instructions:
            n = pt.set_stop(s.symbol, s.trigger)
            results.append({"mode": "paper", "type": "stop", "symbol": s.symbol,
                            "trigger": s.trigger, "success": n > 0})
        return results

    def _execute_live(self, plan: ExecutionPlan) -> List[dict]:
        from kite_connect.trading.order_service import place_order, get_order_book
        from kite_connect.trading import gtt_stops

        results: List[dict] = []
        existing_tags = {o.get("tag") for o in (get_order_book(self.kite) or [])
                         if o.get("status") not in ("REJECTED", "CANCELLED")}
        for o in plan.orders:
            tag = f"NE{plan.as_of:%y%m%d}{o.side[0]}{o.symbol}"[:20]
            if tag in existing_tags:
                results.append({"mode": "live", "symbol": o.symbol, "side": o.side, "success": False,
                                "error": "duplicate (already placed today)", "tag": tag})
                continue
            res = place_order(self.kite, symbol=o.symbol, exchange="NSE", transaction_type=o.side,
                              quantity=o.quantity, order_type="LIMIT", product="CNC",
                              price=o.limit_price, tag=tag, is_exit=(o.side == "SELL"))
            results.append({"mode": "live", "symbol": o.symbol, "side": o.side, "quantity": o.quantity,
                            "limit_price": o.limit_price, "reason": o.reason, "tag": tag, **res})
        # GTT stops at the quantity actually held now; the reconciliation job
        # re-syncs quantities after pending orders fill.
        stops = {s.symbol: s.trigger for s in plan.stop_instructions}
        report = gtt_stops.reconcile_stop_gtts(self.kite, stops=stops)
        results.append({"mode": "live", "type": "gtt_reconcile", "success": not report.get("errors"),
                        "report": report})
        return results
