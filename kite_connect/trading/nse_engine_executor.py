"""
NSE engine executor: TargetPortfolio -> CNC orders + GTT stops.

``EngineExecutor.plan()`` has no side effects: it loads market data, asks the
engine for target weights (``nse_engine.engine.generate_targets``) and diffs
them against current holdings x equity.  ``execute(plan)`` sends the orders
through ``order_service`` (live) or ``PaperTrader`` (paper) and arms GTT
stops through ``gtt_stops``.

Real orders are sent only when BOTH ``CENTURION_PAPER_TRADE=false`` and
``CENTURION_NSE_ENGINE_LIVE=true`` AND the deployment file
(``config/nse_engine_deployed.json``, see ``nse_engine.deployment``) is not a
placeholder; otherwise execution is forced to paper and the reason is logged.

The engine config comes from the deployment file (``Deployment.live_config``:
the approved EngineConfig with ``start = paper_start_date``) unless a config
is injected.

Paper execution matches the backtest timeline (decide after close ``S``, fill
at the open of ``S+1``): ``execute(plan)`` stores the orders as PENDING in the
PaperTrader book and arms stops on current holdings; ``run_paper_session()``
at the next EOD run processes that session from the NSE store — gap stops at
the open, pending fills at the open with ``nse_engine.costs`` (participation
cap, impact, statutory charges; sells first, cash-limited buys), intraday
stops at ``min(open, stop)`` — then marks to the close and plans again.

Distribution shift: NEW-risk target weights (entries / increases, never exits
or stops) are scaled by ``position_size_multiplier`` from
``data/distribution_shift_state.json`` when that file was updated within the
last ``SHIFT_STATE_MAX_AGE_SESSIONS`` trading sessions.

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

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
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
# Shift-state file older than this many trading sessions is ignored
SHIFT_STATE_MAX_AGE_SESSIONS = 5
SHIFT_STATE_FILE = "distribution_shift_state.json"
_REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_FILL_NEXT_OPEN = "next_open"
PAPER_FILL_IMMEDIATE = "immediate"


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


def _default_shift_state_path() -> Path:
    env = os.environ.get("CENTURION_SHIFT_STATE_PATH")
    if env:
        return Path(env)
    try:  # the PaperTrader writes the state next to its SQLite book
        from kite_connect.trading import paper_trader
        return Path(paper_trader._DB_PATH).parent / SHIFT_STATE_FILE
    except Exception:
        return _REPO_ROOT / "data" / SHIFT_STATE_FILE


def load_shift_multiplier(as_of, sessions=None, path=None,
                          max_age_sessions: int = SHIFT_STATE_MAX_AGE_SESSIONS) -> Tuple[float, str]:
    """(multiplier, reason) from the distribution-shift state file.

    Returns 1.0 when the file is absent, malformed (warning), stale — updated
    more than ``max_age_sessions`` trading sessions before ``as_of`` (warning)
    — or dated after ``as_of`` (historical replay).  Sessions are counted on
    ``sessions`` (the market-data calendar), extended with weekdays past its
    end.
    """
    p = Path(path) if path is not None else _default_shift_state_path()
    if not p.exists():
        return 1.0, f"no distribution shift state at {p}"
    try:
        state = json.loads(p.read_text())
        mult = float(state["position_size_multiplier"])
        updated = pd.Timestamp(state["updated_at"])
    except Exception as exc:
        logger.warning("Ignoring malformed distribution shift state %s: %s", p, exc)
        return 1.0, f"malformed shift state ({type(exc).__name__})"
    if not math.isfinite(mult) or not 0.0 < mult <= 1.0:
        logger.warning("Ignoring distribution shift state %s: multiplier %r outside (0, 1]", p, mult)
        return 1.0, f"malformed shift state (multiplier {mult!r})"
    if updated.tzinfo is not None:
        updated = updated.tz_convert("Asia/Kolkata").tz_localize(None)
    upd = updated.normalize()
    as_of_d = pd.Timestamp(as_of).normalize()
    if upd > as_of_d:
        return 1.0, f"shift state updated {upd.date()} after as_of {as_of_d.date()} (ignored)"
    if sessions is not None and len(sessions):
        cal = pd.DatetimeIndex(pd.to_datetime(list(sessions))).normalize()
        age = int(((cal > upd) & (cal <= as_of_d)).sum())
        last = cal.max()
        if as_of_d > last:
            age += int(np.busday_count(max(last, upd).date() + timedelta(days=1),
                                       as_of_d.date() + timedelta(days=1)))
    else:
        age = int(np.busday_count(upd.date() + timedelta(days=1), as_of_d.date() + timedelta(days=1)))
    verdict = state.get("verdict", "?")
    if age > max_age_sessions:
        logger.warning("Ignoring stale distribution shift state %s: updated %s, %d sessions before %s (> %d)",
                       p, upd.date(), age, as_of_d.date(), max_age_sessions)
        return 1.0, f"stale shift state (updated {upd.date()}, {age} sessions old)"
    alerts = state.get("reality_gap_alerts") or []
    extra = f"; reality gap: {'; '.join(map(str, alerts))}" if alerts else ""
    return mult, f"verdict={verdict} updated {upd.date()} ({age} sessions ago){extra}"


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
    shift_multiplier: float = 1.0
    shift_reason: str = ""

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
            "shift_multiplier": self.shift_multiplier, "shift_reason": self.shift_reason,
        }


class EngineExecutor:
    """Turn engine targets into CNC orders and GTT stops.

    Parameters
    ----------
    kite : KiteConnect | None
    paper : bool
        Paper execution (default).  ``paper=False`` still falls back to paper
        unless :func:`live_orders_allowed` is true AND the deployment is not a
        placeholder.
    config : EngineConfig | None
        Injected config (tests / replays).  Default: the deployment file's
        :meth:`~nse_engine.deployment.Deployment.live_config`.
    deployment : Deployment | None
        Pre-loaded deployment (default: loaded from ``deployment_path``).
    deployment_path : str | None
        Default ``config/nse_engine_deployed.json`` (or env CENTURION_NSE_DEPLOYMENT).
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
    shift_state_path : str | None
        Distribution-shift state file (default ``data/distribution_shift_state.json``
        next to the paper book, or env CENTURION_SHIFT_STATE_PATH).
    paper_fill : str
        ``"next_open"`` (default): paper orders are queued PENDING and filled
        by :meth:`run_paper_session` at the next open.  ``"immediate"``: fill
        at the last close at once (legacy behaviour, not backtest-consistent).
    """

    def __init__(self, kite=None, paper: bool = True, config=None,
                 target_fn: Optional[Callable] = None, data_loader: Optional[Callable] = None,
                 holdings_fn: Optional[Callable] = None, paper_trader=None,
                 stopped_out_fn: Optional[Callable] = None, apply_buffer: bool = False,
                 deployment=None, deployment_path: Optional[str] = None,
                 shift_state_path: Optional[str] = None, paper_fill: str = PAPER_FILL_NEXT_OPEN):
        self.kite = kite
        self.config = config
        self._deployment = deployment
        self._deployment_path = deployment_path
        self._shift_state_path = shift_state_path
        if paper_fill not in (PAPER_FILL_NEXT_OPEN, PAPER_FILL_IMMEDIATE):
            raise ValueError(f"paper_fill must be {PAPER_FILL_NEXT_OPEN!r} or {PAPER_FILL_IMMEDIATE!r}")
        self.paper_fill = paper_fill
        self._stopped_out_fn = stopped_out_fn
        self.apply_buffer = bool(apply_buffer)
        self._target_fn = target_fn
        self._data_loader = data_loader
        self._holdings_fn = holdings_fn
        self._paper_trader = paper_trader
        allowed, reason = live_orders_allowed()
        if not paper and not allowed:
            logger.warning("EngineExecutor: live requested but forcing PAPER — %s", reason)
        if not paper and allowed:
            dep_ok, dep_reason = self._deployment_live_check()
            if not dep_ok:
                logger.warning("EngineExecutor: live requested but forcing PAPER — %s", dep_reason)
                allowed, reason = False, dep_reason
        if not paper and allowed and kite is None:
            logger.warning("EngineExecutor: live allowed but no Kite session — forcing PAPER")
            allowed = False
        self.paper = bool(paper or not allowed)
        self.mode_reason = reason if not self.paper else (
            "paper requested" if paper else f"forced paper: {reason}")

    # ── dependencies ───────────────────────────────────────────

    def deployment(self, required: bool = True):
        """The deployment (loaded lazily). Raises DeploymentError when required."""
        if self._deployment is None:
            from nse_engine.deployment import DeploymentError, load_deployment
            try:
                self._deployment = load_deployment(self._deployment_path)
            except DeploymentError:
                if required:
                    raise
                return None
            logger.info("EngineExecutor: deployment %s", self._deployment.summary())
        return self._deployment

    def _deployment_live_check(self) -> Tuple[bool, str]:
        try:
            dep = self.deployment(required=True)
        except Exception as exc:
            return False, f"deployment unavailable: {exc}"
        return dep.live_allowed()

    def _cfg(self):
        if self.config is None:
            dep = self.deployment(required=True)
            if dep.is_placeholder:
                logger.warning("EngineExecutor: deployment %s is a PLACEHOLDER (default EngineConfig) — "
                               "paper trading only, live orders are refused", dep.path)
            self.config = dep.live_config()
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
        # Only a deployment already in use (injected, or the source of the config)
        # pins the anchor; an injected config without one keeps the warm-up rule.
        dep = self._deployment
        if dep is not None and getattr(dep, "data_anchor_date", None) is not None:
            anchor = dep.data_anchor_date  # same first row as the validated runs
        return self._loader()(
            cfg.data.store_dir, anchor.isoformat(), pd.Timestamp(end).date().isoformat(),
            series=cfg.data.series, min_median_value_inr=cfg.data.load_min_median_value_inr,
            float_dtype=cfg.data.float_dtype,
            include_symbols=(cfg.sleeves.gold_symbol, cfg.sleeves.silver_symbol),
            adjust_dividends=cfg.data.adjust_dividends,
        )

    # ── planning ───────────────────────────────────────────────

    def plan(self, as_of=None, data=None) -> ExecutionPlan:
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
                h = dict(h)
                if h.get("entry_date"):
                    # session date: an entry timestamp (e.g. 09:15 IST) must map to its own session
                    h["entry_date"] = str(h["entry_date"])[:10]
                plain[sym] = h
        holdings.update(holdings_from_mapping(plain))

        end = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(datetime.now(_IST).date())
        if data is None:
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
        weights = self._apply_shift_multiplier(plan, target, holdings, prices, equity, view)
        symbols = list(dict.fromkeys(list(holdings) + list(weights) + list(target.exits)))

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
            w = 0.0 if exit_reason else float(weights.get(sym, 0.0))
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
        logger.info("EngineExecutor plan %s: equity=%.0f cash=%.0f sells=%d buys=%d stops=%d skipped=%d "
                    "shift_multiplier=%.2f",
                    as_of.date(), equity, cash, len(sells), len(kept),
                    len(plan.stop_instructions), len(plan.skipped), plan.shift_multiplier)
        return plan

    def _apply_shift_multiplier(self, plan: ExecutionPlan, target, holdings, prices, equity, view) -> Dict[str, float]:
        """Scale NEW-risk weights (entries / increases) by the shift multiplier.

        ``w' = max(current_weight, w * multiplier)`` for targets above the
        current weight; exits, reductions, unchanged (drifted) weights and
        stops are untouched.
        """
        weights = dict(target.weights or {})
        mult, reason = load_shift_multiplier(plan.as_of, sessions=getattr(view, "dates", None),
                                             path=self._shift_state_path)
        plan.shift_multiplier, plan.shift_reason = mult, reason
        if mult >= 1.0:
            logger.info("EngineExecutor: position size multiplier 1.00 (%s)", reason)
            return weights
        scaled = {}
        for sym, w in weights.items():
            if sym in (target.exits or {}) or w <= 0:
                continue
            h = holdings.get(sym)
            cur_w = (h.quantity * prices.get(sym, h.avg_price) / equity) if (h is not None and equity > 0) else 0.0
            if w > cur_w + 1e-9:
                new_w = max(cur_w, w * mult)
                scaled[sym] = (w, new_w)
                weights[sym] = new_w
        msg = (f"distribution shift: position size multiplier {mult:.2f} ({reason}) — "
               f"{len(scaled)} new-risk weight(s) scaled")
        plan.notes.append(msg)
        logger.warning("EngineExecutor: %s", msg)
        for sym, (w, nw) in sorted(scaled.items()):
            logger.info("  %s target weight %.4f -> %.4f", sym, w, nw)
        return weights

    @staticmethod
    def _as_of_is_live(as_of) -> bool:
        """Freshness applies to live-date planning, not historical replays."""
        return pd.Timestamp(as_of).date() >= (datetime.now(_IST).date() - timedelta(days=7))

    # ── execution ──────────────────────────────────────────────

    def execute(self, plan: ExecutionPlan) -> List[dict]:
        allowed, reason = live_orders_allowed()
        if not self.paper and allowed:
            allowed, dep_reason = self._deployment_live_check()
            reason = reason if allowed else dep_reason
        paper = self.paper or not allowed or self.kite is None
        if not self.paper and paper:
            logger.warning("EngineExecutor.execute: forcing PAPER — %s", reason if not allowed else "no Kite session")
        if not paper:
            return self._execute_live(plan)
        if self.paper_fill == PAPER_FILL_IMMEDIATE:
            return self._execute_paper_immediate(plan)
        return self._execute_paper_pending(plan)

    def _execute_paper_pending(self, plan: ExecutionPlan) -> List[dict]:
        """Queue the plan as PENDING next-open orders; arm stops on current holdings."""
        pt = self._pt()
        if any(s.get("reason") == "stale_data" for s in plan.skipped):
            # Never supersede orders queued from fresh data with a stale-data (empty) plan
            logger.warning("EngineExecutor: stale-data plan — pending orders left unchanged")
            return []
        stops = {s.symbol: s.trigger for s in plan.stop_instructions}
        rows = pt.queue_pending_orders(plan.as_of, [
            {"symbol": o.symbol, "side": o.side, "quantity": o.quantity,
             "target_qty": (o.current_qty - o.quantity) if o.side == "SELL" else (o.current_qty + o.quantity),
             "ref_price": o.ref_price, "stop_price": stops.get(o.symbol, 0.0), "reason": o.reason}
            for o in plan.orders
        ])
        results: List[dict] = [{"mode": "paper", "status": "PENDING", "success": True, "id": r["id"],
                                "symbol": r["symbol"], "side": r["side"], "quantity": r["quantity"],
                                "reason": r["reason"]} for r in rows]
        held = pt.holdings()
        for s in plan.stop_instructions:
            if s.symbol not in held:
                continue  # new entries carry their stop on the pending order
            n = pt.set_stop(s.symbol, s.trigger)
            results.append({"mode": "paper", "type": "stop", "symbol": s.symbol,
                            "trigger": s.trigger, "success": n > 0})
        return results

    def _execute_paper_immediate(self, plan: ExecutionPlan) -> List[dict]:
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

    # ── paper session (EOD) ────────────────────────────────────

    @classmethod
    def _session_bars(cls, view, symbols, adv_lookback: int = 20) -> Dict[str, dict]:
        """{symbol: {date, open, low, close, adv}} for the last session of ``view``
        (``adv``: median traded value at the previous session, as backtest stops)."""
        bars: Dict[str, dict] = {}
        if not len(view.dates):
            return bars
        session = pd.Timestamp(view.dates[-1])
        advs = cls._open_quotes(view, symbols, adv_lookback)
        for sym in symbols:
            if sym not in view.close.columns:
                continue
            o, lo, c = (float(getattr(view, f)[sym].iloc[-1]) for f in ("open", "low", "close"))
            if all(math.isfinite(x) and x > 0 for x in (o, lo, c)):
                bars[sym] = {"date": session, "open": o, "low": lo, "close": c,
                             "adv": (advs.get(sym) or {}).get("adv", float("nan"))}
        return bars

    @staticmethod
    def _open_quotes(view, symbols, adv_lookback: int) -> Dict[str, dict]:
        """{symbol: {open (last session), adv (median traded value at the previous session)}}."""
        from nse_engine.costs import median_traded_value

        cols = [s for s in dict.fromkeys(symbols) if s in view.close.columns]
        if not cols or not len(view.dates):
            return {}
        adv = median_traded_value(view.value[cols], adv_lookback)
        adv_row = adv.iloc[-2] if len(adv) >= 2 else adv.iloc[-1] * np.nan
        opens = view.open[cols].iloc[-1]
        return {s: {"open": float(opens[s]), "adv": float(adv_row[s])} for s in cols}

    def run_paper_session(self, as_of=None) -> dict:
        """EOD paper run for the latest store session: stops, next-open fills, plan, queue.

        1. gap stops at the session open (lots from earlier sessions);
        2. PENDING orders decided at the previous session fill at this open
           (``PaperTrader.fill_pending_orders``); older ones are cancelled;
        3. intraday stops at ``min(open, stop)`` (including lots just filled);
        4. mark to the session close;
        5. plan from the close (shift multiplier applied) and queue the orders.

        Steps 1-3 run once per session (``engine_last_session``), so a re-run
        on the same day only re-plans (superseding that day's pending orders).
        """
        if not self.paper:
            raise RuntimeError("run_paper_session is paper-only")
        cfg = self._cfg()
        pt = self._pt()
        end = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(datetime.now(_IST).date())
        data = self._load(cfg, end)
        view = data.until(end) if hasattr(data, "until") else data
        if not len(view.dates):
            raise RuntimeError(f"no market data up to {end.date()} in {cfg.data.store_dir}")
        session = pd.Timestamp(view.dates[-1])
        report: dict = {"session": session.date().isoformat(), "stops": [], "fills": None,
                        "plan": None, "results": [], "notes": []}

        last = pt.engine_last_session()
        if last is not None and session.date() <= last:
            report["notes"].append(f"session {session.date()} already processed (last {last}); re-planning only")
            logger.info("EngineExecutor: %s", report["notes"][-1])
        else:
            pending_syms = [o["symbol"] for o in pt.pending_orders()]
            held_syms = sorted({p.symbol for p in pt._positions if p.is_open})
            bars = self._session_bars(view, list(dict.fromkeys(held_syms + pending_syms)),
                                      cfg.costs.adv_lookback_days)
            stops_by_sym: Dict[str, float] = {}
            for p in pt._positions:
                if p.is_open and p.stop_loss:
                    stops_by_sym[p.symbol] = max(stops_by_sym.get(p.symbol, 0.0), p.stop_loss)
            gap_bars = {s: b for s, b in bars.items() if s in stops_by_sym and b["open"] <= stops_by_sym[s]}
            report["stops"].extend(pt.simulate_gtt_stops(gap_bars, cost_config=cfg.costs))
            stopped_today = {e["symbol"] for e in report["stops"]}
            quotes = self._open_quotes(view, pending_syms, cfg.costs.adv_lookback_days)
            report["fills"] = pt.fill_pending_orders(
                session, view.dates, quotes, cfg.costs,
                min_trade_value_inr=cfg.portfolio.min_trade_value_inr, skip_buys=stopped_today)
            held_after = sorted({p.symbol for p in pt._positions if p.is_open})
            bars.update(self._session_bars(view, [s for s in held_after if s not in bars],
                                           cfg.costs.adv_lookback_days))
            report["stops"].extend(pt.simulate_gtt_stops(bars, cost_config=cfg.costs))
            pt.set_engine_last_session(session)

        # Mark to market at the session close (forward-filled for suspended names)
        closes = view.close.ffill().iloc[-1]
        for sym in {p.symbol for p in pt._positions if p.is_open}:
            px = closes.get(sym) if sym in closes.index else None
            if px is not None and math.isfinite(float(px)) and float(px) > 0:
                pt._price_overrides[sym] = float(px)

        dep = self._deployment
        if dep is not None and session.date() < dep.paper_start_date:
            report["notes"].append(f"session {session.date()} is before paper_start_date "
                                   f"{dep.paper_start_date}: no plan")
            logger.info("EngineExecutor: %s", report["notes"][-1])
            return report
        plan = self.plan(as_of=session, data=data)
        report["plan"] = plan
        report["results"] = self.execute(plan)
        return report

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
