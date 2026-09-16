"""
NSE engine: target generation (shared by backtest and live) and the
event-driven backtest simulator with run recording.

Timeline of one backtest day ``u``
----------------------------------
1. Idle cash accrues ``cash_yield_annual / 252``.
2. Held symbols with no data after an earlier date (delisted) are liquidated
   at their last close.
3. Gap stops: a stop set at an earlier close triggers at the open when
   ``open_u <= stop`` and fills at the open.
4. Orders decided at close ``u - 1 - lag_days`` fill at ``open_u``: sells
   first, then buys with the available cash (scaled down if insufficient),
   each capped at ``max_participation`` of the median traded value known at
   the decision, with impact and statutory costs.
5. Intraday stops: ``low_u <= stop`` fills at ``min(open_u, stop)``.
6. Mark to market at the close (last close for suspended symbols).
7. After the close, :func:`generate_targets` produces the next targets; stops
   of current holdings are updated immediately (never lowered).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from nse_engine.allocator import allocate, basket_vol
from nse_engine.config import EngineConfig
from nse_engine.costs import median_traded_value, simulate_fill
from nse_engine.metrics import compute_metrics
from nse_engine.portfolio import (
    EXIT_RANK,
    apply_no_trade_buffer,
    atr_panel,
    core_weights,
    exit_reason,
    select_names,
    stop_fill_price,
    trailing_stop,
)
from nse_engine.regime import compute_regime
from nse_engine.signals import compute_signal_panels, daily_returns, realised_vol
from nse_engine.sleeves import compute_sleeve_panels, sleeve_weights
from nse_engine.types import BacktestResult, Holding, MarketData, TargetPortfolio, Trade
from nse_engine.universe import compute_universe_panel

logger = logging.getLogger(__name__)

EXIT_SLEEVE_TREND = "sleeve_trend_exit"
NOTE_REBALANCE = "rebalance_day"
REPO_ROOT = Path(__file__).resolve().parents[1]


# ============================================================================
# cache
# ============================================================================


class EngineCache:
    """Causal indicator panels for one ``MarketData`` + ``EngineConfig``.

    Every panel's row ``t`` depends only on rows ``<= t``, so a cache built on
    the full history gives the same decision at ``t`` as one built on
    ``data.until(t)``.
    """

    def __init__(self, data: MarketData, config: EngineConfig):
        t0 = time.perf_counter()
        self.config = config
        self.dates: pd.DatetimeIndex = pd.DatetimeIndex(data.dates)
        self._dates_i8 = self.dates.as_unit("ns").asi8
        self.columns: pd.Index = data.close.columns
        self.symbols: List[str] = [str(c) for c in self.columns]
        self.sym_index: Dict[str, int] = {s: i for i, s in enumerate(self.symbols)}
        self.name_rank = np.argsort(np.argsort(np.array(self.symbols, dtype=object)))
        self._data_id = id(data.close)
        self.sectors: Dict[str, str] = dict(data.sectors or {})
        self.data_hash = data.data_hash

        close_df = data.close.astype("float64")
        self.close = close_df.to_numpy()
        self.open = data.open.to_numpy(dtype="float64")
        self.high = data.high.to_numpy(dtype="float64")
        self.low = data.low.to_numpy(dtype="float64")
        self.close_ffill = close_df.ffill().to_numpy()
        finite = np.isfinite(self.close)
        n = len(self.dates)
        has = finite.any(axis=0)
        self.last_valid_pos = np.where(has, n - 1 - np.argmax(finite[::-1], axis=0), -1)

        returns_df = daily_returns(close_df)
        self.returns = returns_df.to_numpy()
        self.vol_ann = realised_vol(returns_df, config.portfolio.vol_lookback_days).to_numpy()
        self.atr = atr_panel(data.high, data.low, data.close, config.portfolio.atr_lookback).to_numpy()
        self.adv = median_traded_value(data.value, config.costs.adv_lookback_days).to_numpy()

        self.sleeves = compute_sleeve_panels(close_df, config.sleeves, config.allocator.vol_lookback_days)
        self.sleeve_syms = list(self.sleeves.symbols)
        self.sleeve_idx = np.array([self.sym_index[s] for s in self.sleeve_syms], dtype=int)
        self.sleeve_in_trend = self.sleeves.in_trend.to_numpy(dtype=bool)
        self.sleeve_vol = self.sleeves.vol.to_numpy()

        self.universe = compute_universe_panel(data, config.universe, exclude=self.sleeve_syms)
        self.universe_mask = self.universe.mask.to_numpy()

        self.signals = compute_signal_panels(close_df, self.universe.mask, config.signals, returns_df,
                                             delivery_pct=data.delivery_pct)
        self.combined = self.signals.combined.to_numpy()
        self.warmup = self.signals.warmup.to_numpy()

        self.regime = compute_regime(data.index_close, close_df, self.universe.mask, config.regime)
        self.regime_state = self.regime.state.to_numpy()
        self.regime_scale = self.regime.scale.to_numpy()
        self.regime_switched = self.regime.switched().to_numpy()
        # Rebalance days: calendar period starts (anchor-independent) or every n rows from row 0 (legacy)
        every = max(int(config.portfolio.rebalance_every_n_days), 1)
        if config.portfolio.calendar_schedule:
            from nse_engine.calendar import period_start_mask
            self.rebalance_day = period_start_mask(data.dates, every)
        else:
            self.rebalance_day = (np.arange(n) % every) == 0
        self.build_seconds = time.perf_counter() - t0
        logger.info("EngineCache built in %.1fs (%d dates x %d symbols)", self.build_seconds, n, len(self.symbols))

    def position(self, as_of: pd.Timestamp) -> int:
        """Row of the last trading date <= ``as_of``."""
        return int(np.searchsorted(self._dates_i8, _ts_i8(as_of), side="right")) - 1

    def first_position_on_or_after(self, date: pd.Timestamp) -> int:
        return int(np.searchsorted(self._dates_i8, _ts_i8(date), side="left"))

    def matches(self, data: MarketData, config: EngineConfig) -> bool:
        if not (config is self.config or config == self.config):
            return False
        if id(data.close) == self._data_id and len(data.dates) == len(self.dates):
            return True
        return len(data.dates) == len(self.dates) and data.dates.equals(self.dates) and data.close.columns.equals(self.columns)


def _ts_i8(date) -> int:
    ts = pd.Timestamp(date)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return int(ts.as_unit("ns").value)


def _ensure_cache(data: MarketData, config: EngineConfig, cache: Optional[EngineCache]) -> EngineCache:
    if cache is not None and cache.matches(data, config):
        return cache
    if cache is not None:
        logger.warning("EngineCache does not match data/config; rebuilding")
    return EngineCache(data, config)


# ============================================================================
# targets
# ============================================================================


def _stop_for(cache: EngineCache, j: int, pos: int, h: Optional[Holding]) -> Optional[float]:
    k = cache.config.portfolio.stop_atr_multiple
    atr = cache.atr[pos, j]
    if h is None:
        return trailing_stop(cache.close[pos, j], atr, k, None)
    ep = cache.first_position_on_or_after(h.entry_date)
    ep = min(max(ep, 0), pos)
    seg = cache.close[ep : pos + 1, j]
    highest = float(np.nanmax(seg)) if np.isfinite(seg).any() else float("nan")
    return trailing_stop(highest, atr, k, h.stop_price)


def generate_targets(
    data: MarketData,
    config: EngineConfig,
    as_of: pd.Timestamp,
    holdings: Optional[Mapping[str, Holding]] = None,
    cache: Optional[EngineCache] = None,
    *,
    equity: Optional[float] = None,
    stopped_out: Optional[Mapping[str, pd.Timestamp]] = None,
) -> TargetPortfolio:
    """Target portfolio after the close of ``as_of`` (uses rows <= as_of only).

    Parameters
    ----------
    holdings : current positions (core stocks and sleeve ETFs).
    equity : total account equity in INR.  Needed for the no-trade buffer,
        the minimum trade value and for keeping drifted weights on
        non-rebalance days; without it targets are pure model weights.
    stopped_out : symbol -> date of its last stop-out, for the stop cooldown.
    """
    cache = _ensure_cache(data, config, cache)
    pos = cache.position(as_of)
    if pos < 0:
        raise ValueError(f"as_of {as_of} precedes the data")
    pcfg = config.portfolio
    date = cache.dates[pos]
    holdings = {s: h for s, h in (holdings or {}).items() if h is not None and int(h.quantity) > 0}
    notes: List[str] = []
    if cache.warmup[pos]:
        notes.append("signal_warmup: forecast normalisers have fewer than normalizer_min_obs dates")

    sleeve_set = set(cache.sleeve_syms)
    uni = np.flatnonzero(cache.universe_mask[pos])
    fc = cache.combined[pos]
    cand = uni[np.isfinite(fc[uni]) & (fc[uni] > 0)]
    order = cand[np.lexsort((cache.name_rank[cand], -fc[cand]))]
    rank_syms = [cache.symbols[j] for j in order]
    ranks = pd.Series(np.arange(1, len(order) + 1, dtype="int64"), index=rank_syms)
    rank_map = dict(zip(rank_syms, range(1, len(order) + 1)))

    close_row = cache.close_ffill[pos]
    current_w: Dict[str, float] = {}
    if equity is not None and equity > 0:
        for s, h in holdings.items():
            j = cache.sym_index.get(s)
            px = close_row[j] if j is not None else np.nan
            if np.isfinite(px):
                current_w[s] = float(h.quantity) * float(px) / float(equity)

    exits: Dict[str, str] = {}
    stops: Dict[str, float] = {}
    kept: List[str] = []
    frozen: List[str] = []  # held but not trading today
    for s, h in holdings.items():
        if s in sleeve_set:
            continue
        j = cache.sym_index.get(s)
        if j is None:
            exits[s] = EXIT_RANK
            notes.append(f"{s}: not in market data; exit")
            continue
        if not np.isfinite(cache.close[pos, j]):
            frozen.append(s)
            if h.stop_price is not None:
                stops[s] = float(h.stop_price)
            notes.append(f"{s}: no price on {date.date()}; held without evaluation")
            continue
        reason = exit_reason(rank_map.get(s), float(fc[j]), float(cache.low[pos, j]), h.stop_price, pcfg.exit_rank)
        if reason:
            exits[s] = reason
            continue
        kept.append(s)
        st = _stop_for(cache, j, pos, h)
        if st is not None:
            stops[s] = st

    blocked = set(exits)  # nothing that exits today is re-bought today
    for s, d in (stopped_out or {}).items():
        dp = cache.first_position_on_or_after(d)
        if 0 <= pos - dp < pcfg.stop_cooldown_days:
            blocked.add(s)

    in_trend = {cache.sleeve_syms[k]: bool(cache.sleeve_in_trend[pos, k]) for k in range(len(cache.sleeve_syms))}
    for s in holdings:
        if s in sleeve_set and not in_trend.get(s, False):
            exits[s] = EXIT_SLEEVE_TREND

    core_holdings = [s for s in holdings if s not in sleeve_set]
    rebalance = bool(cache.rebalance_day[pos]) or not core_holdings or bool(cache.regime_switched[pos])
    scale = float(cache.regime_scale[pos])
    state = str(cache.regime_state[pos])
    if cache.regime_switched[pos]:
        notes.append(f"regime_switch:{state}")

    def allocate_book(core_names: List[str], sleeve_names: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
        idx = [cache.sym_index[s] for s in core_names]
        rel = core_weights(
            pd.Series(fc[idx], index=core_names), pd.Series(cache.vol_ann[pos, idx], index=core_names), pcfg, cache.sectors
        ) if core_names else pd.Series(dtype="float64")
        s_k = [cache.sleeve_syms.index(s) for s in sleeve_names]
        rel_s = sleeve_weights(
            pd.Series(True, index=sleeve_names), pd.Series(cache.sleeve_vol[pos, s_k], index=sleeve_names)
        ) if sleeve_names else pd.Series(dtype="float64")
        lb = config.allocator.vol_lookback_days
        lo = max(0, pos - lb + 1)
        c_vol = basket_vol(rel.to_numpy(), cache.returns[lo : pos + 1, [cache.sym_index[s] for s in rel.index]]) if len(rel) else float("nan")
        s_vol = basket_vol(rel_s.to_numpy(), cache.returns[lo : pos + 1, [cache.sym_index[s] for s in rel_s.index]]) if len(rel_s) else float("nan")
        alloc = allocate(c_vol, s_vol, bool(len(rel) and rel.sum() > 0), bool(len(rel_s)), scale, config.allocator)
        notes.extend(alloc.notes)
        cw = {s: float(w * alloc.core_capital) for s, w in rel.items() if w * alloc.core_capital > 0}
        sw = {s: float(w * alloc.sleeve_capital) for s, w in rel_s.items() if w * alloc.sleeve_capital > 0}
        return cw, sw

    if rebalance:
        notes.append(NOTE_REBALANCE)
        selected, dropped = select_names(kept, ranks, blocked | set(frozen), pcfg)
        for s in dropped:
            exits[s] = EXIT_RANK
            stops.pop(s, None)
        active_sleeves = [s for s in cache.sleeve_syms if in_trend[s]]
        core_t, sleeve_t = allocate_book(selected, active_sleeves)
        target = {**core_t, **sleeve_t}
        for s in frozen:
            if s in current_w:
                target[s] = current_w[s]
        if current_w:
            buffered = apply_no_trade_buffer(target, current_w, float(equity), pcfg.no_trade_buffer, pcfg.min_trade_value_inr)
            if sum(buffered.values()) > config.allocator.max_gross + 1e-12:
                notes.append("buffer disabled: buffered weights exceeded max_gross")
                buffered = {s: w for s, w in target.items() if w > 0}
            weights = buffered
        else:
            weights = {s: w for s, w in target.items() if w > 0}
        for s in weights:
            if s not in stops and s not in sleeve_set:
                j = cache.sym_index[s]
                st = _stop_for(cache, j, pos, holdings.get(s))
                if st is not None:
                    stops[s] = st
    else:
        if current_w:
            weights = {s: current_w[s] for s in kept + frozen if s in current_w}
            for s in cache.sleeve_syms:
                if s in holdings and s not in exits and s in current_w:
                    weights[s] = current_w[s]
        else:
            notes.append("no equity given on a non-rebalance day: model weights over current holdings")
            held_sleeves = [s for s in cache.sleeve_syms if s in holdings and s not in exits]
            core_t, sleeve_t = allocate_book(kept, held_sleeves)
            weights = {**core_t, **sleeve_t}

    gross = sum(weights.values())
    if gross > config.allocator.max_gross + 1e-12:
        k = config.allocator.max_gross / gross
        weights = {s: w * k for s, w in weights.items()}
        notes.append("weights scaled to max_gross")
    stops = {s: v for s, v in stops.items() if s in weights and s not in sleeve_set}

    forecasts = {s: float(fc[cache.sym_index[s]]) for s in rank_syms}
    for s in holdings:
        j = cache.sym_index.get(s)
        if j is not None and np.isfinite(fc[j]):
            forecasts[s] = float(fc[j])
    return TargetPortfolio(
        as_of=date,
        weights=weights,
        stops=stops,
        forecasts=forecasts,
        ranks=rank_map,
        core_weights={s: w for s, w in weights.items() if s not in sleeve_set},
        sleeve_weights={s: w for s, w in weights.items() if s in sleeve_set},
        exits=exits,
        regime=state,
        regime_scale=scale,
        universe_size=int(len(uni)),
        notes=notes,
    )


# ============================================================================
# backtest
# ============================================================================


@dataclass
class _Order:
    decision_pos: int
    target_qty: Dict[str, int]
    stops: Dict[str, float]
    reasons: Dict[str, str]


@dataclass
class _Book:
    cash: float
    positions: Dict[str, Holding] = field(default_factory=dict)
    trades: List[Trade] = field(default_factory=list)
    stopped_out: Dict[str, pd.Timestamp] = field(default_factory=dict)
    stopped_pos: Dict[str, int] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def _record_trade(book: _Book, date, sym, side, fill, reason) -> None:
    book.trades.append(
        Trade(
            date=date, symbol=sym, side=side, quantity=int(fill.quantity), price=float(fill.price),
            value_inr=float(fill.value_inr), cost_inr=float(fill.cost_inr), reason=reason,
            requested_quantity=int(fill.requested_quantity),
        )
    )


def _sell(book: _Book, cache: EngineCache, sym: str, qty: int, price: float, adv: float, date, reason: str, cap: bool) -> int:
    fill = simulate_fill("SELL", qty, price, adv, date, cache.config.costs, apply_cap=cap)
    if fill.quantity <= 0:
        return 0
    book.cash += fill.value_inr - fill.cost_inr
    h = book.positions[sym]
    h.quantity -= fill.quantity
    if h.quantity <= 0:
        del book.positions[sym]
    _record_trade(book, date, sym, "SELL", fill, reason)
    return fill.quantity


def _execute_order(book: _Book, cache: EngineCache, order: _Order, u: int, skip: set) -> None:
    cfg = cache.config
    date = cache.dates[u]
    min_val = cfg.portfolio.min_trade_value_inr
    dpos = order.decision_pos
    # sells first
    for sym in list(book.positions):
        h = book.positions[sym]
        tgt = int(order.target_qty.get(sym, 0))
        if tgt >= h.quantity:
            continue
        j = cache.sym_index[sym]
        o = cache.open[u, j]
        if not np.isfinite(o) or o <= 0:
            book.notes.append(f"{date.date()} {sym}: no open, sell skipped")
            continue
        diff = h.quantity - tgt
        if tgt > 0 and diff * o < min_val:
            continue
        got = _sell(book, cache, sym, diff, o, cache.adv[dpos, j], date, order.reasons.get(sym, "rebalance"), True)
        if got < diff:
            book.notes.append(f"{date.date()} {sym}: sell capped {got}/{diff}")
    # buys with available cash
    wanted = []
    for sym, tgt in order.target_qty.items():
        if sym in skip:
            continue
        cur = book.positions[sym].quantity if sym in book.positions else 0
        if tgt <= cur:
            continue
        j = cache.sym_index[sym]
        o = cache.open[u, j]
        if not np.isfinite(o) or o <= 0:
            continue
        diff = int(tgt - cur)
        if diff * o < min_val:
            continue
        st = order.stops.get(sym) if sym not in book.positions else book.positions[sym].stop_price
        if st is not None and o <= st:
            book.notes.append(f"{date.date()} {sym}: open at/below stop, buy skipped")
            continue
        fill = simulate_fill("BUY", diff, o, cache.adv[dpos, j], date, cfg.costs)
        if fill.quantity > 0:
            wanted.append([sym, j, fill])
    need = sum(f.value_inr + f.cost_inr for _, _, f in wanted)
    avail = max(book.cash, 0.0)
    if need > avail and need > 0:
        factor = avail / need
        for item in wanted:
            sym, j, f = item
            q = int(math.floor(f.quantity * factor))
            item[2] = simulate_fill("BUY", q, f.price, cache.adv[dpos, j], date, cfg.costs, apply_cap=False)
            item[2] = dataclasses.replace(item[2], requested_quantity=f.requested_quantity)
        # integer rounding plus concave impact can still overshoot by pennies
        wanted.sort(key=lambda it: -it[2].value_inr)
        while sum(f.value_inr + f.cost_inr for _, _, f in wanted) > avail and wanted:
            sym, j, f = wanted[0]
            q = f.quantity - 1
            wanted[0][2] = dataclasses.replace(
                simulate_fill("BUY", q, f.price, cache.adv[dpos, j], date, cfg.costs, apply_cap=False),
                requested_quantity=f.requested_quantity,
            )
            if q <= 0:
                wanted.pop(0)
            wanted.sort(key=lambda it: -it[2].value_inr)
    for sym, j, f in wanted:
        if f.quantity <= 0:
            continue
        book.cash -= f.value_inr + f.cost_inr
        if sym in book.positions:
            h = book.positions[sym]
            tot = h.quantity + f.quantity
            h.avg_price = (h.avg_price * h.quantity + f.value_inr) / tot
            h.quantity = tot
        else:
            book.positions[sym] = Holding(
                symbol=sym, quantity=int(f.quantity), avg_price=float(f.price), entry_date=date,
                stop_price=order.stops.get(sym),
            )
        _record_trade(book, date, sym, "BUY", f, order.reasons.get(sym, "rebalance"))


def _projected_holdings(
    book: _Book, pending: Dict[int, _Order], cache: EngineCache, close_row: np.ndarray
) -> Dict[str, Holding]:
    """Holdings as they will be once in-flight orders fill (``lag_days > 0``).

    Each order carries absolute target quantities, so the latest pending order
    defines the projected book.  Deciding against it stops a delayed
    execution from re-sending, or undoing, orders that are already in flight.
    """
    latest_pos = max(pending)
    order = pending[latest_pos]
    out: Dict[str, Holding] = {}
    for sym in set(book.positions) | set(order.target_qty):
        qty = int(order.target_qty.get(sym, 0))
        h = book.positions.get(sym)
        if qty <= 0 or (h is None and book.stopped_pos.get(sym, -1) > order.decision_pos):
            continue
        if h is not None:
            out[sym] = dataclasses.replace(h, quantity=qty)
        else:
            px = float(close_row[cache.sym_index[sym]])
            out[sym] = Holding(sym, qty, px, cache.dates[min(latest_pos, len(cache.dates) - 1)], order.stops.get(sym))
    return out


def run_backtest(
    data: MarketData,
    config: EngineConfig,
    *,
    record: bool = True,
    tag: str = "",
    lag_days: int = 0,
    cache: Optional[EngineCache] = None,
) -> BacktestResult:
    """Simulate the engine between ``config.start`` and ``config.end``.

    Integer shares, INR accounting, fills at the open of ``t + 1 + lag_days``.
    Pass ``cache`` to reuse precomputed panels across runs with the same
    data and config (e.g. lag sensitivity).
    """
    t0 = time.perf_counter()
    cache = _ensure_cache(data, config, cache)
    dates = cache.dates
    s0 = int(dates.searchsorted(pd.Timestamp(config.start), side="left"))
    s1 = int(dates.searchsorted(pd.Timestamp(config.end), side="right")) - 1
    if s1 < s0:
        raise ValueError("no trading dates between config.start and config.end")
    lag = max(int(lag_days), 0)
    book = _Book(cash=float(config.initial_capital))
    sleeve_set = set(cache.sleeve_syms)
    pending: Dict[int, _Order] = {}
    daily_yield = config.cash_yield_annual / 252.0
    eq_vals: List[float] = []
    w_rows: List[Dict[str, float]] = []
    cfg_costs = config.costs

    for u in range(s0, s1 + 1):
        date = dates[u]
        if u > s0 and book.cash > 0:
            book.cash += book.cash * daily_yield
        # delisted symbols: liquidate at last close
        for sym in list(book.positions):
            j = cache.sym_index[sym]
            lv = cache.last_valid_pos[j]
            if lv < u:
                h = book.positions[sym]
                px = cache.close[lv, j] if lv >= 0 else np.nan
                fill = simulate_fill("SELL", h.quantity, px, cache.adv[max(lv, 0), j], date, cfg_costs, apply_cap=False)
                book.cash += fill.value_inr - fill.cost_inr
                del book.positions[sym]
                _record_trade(book, date, sym, "SELL", fill, "delisted")
                book.notes.append(f"{date.date()} {sym}: stopped trading after {dates[lv].date()}; liquidated at last close {px:.2f}")
        stopped_today: set = set()
        # gap stops at the open
        for sym in list(book.positions):
            h = book.positions[sym]
            if h.stop_price is None or sym in sleeve_set:
                continue
            j = cache.sym_index[sym]
            o = cache.open[u, j]
            if np.isfinite(o) and o <= h.stop_price:
                _sell(book, cache, sym, h.quantity, o, cache.adv[u - 1, j] if u > 0 else np.nan, date, "stop", False)
                book.stopped_out[sym] = date
                book.stopped_pos[sym] = u
                stopped_today.add(sym)
        order = pending.pop(u, None)
        if order is not None:
            # never re-buy a name stopped out after the order was decided
            skip = stopped_today | {s for s, p in book.stopped_pos.items() if p > order.decision_pos}
            _execute_order(book, cache, order, u, skip)
        # intraday stops
        for sym in list(book.positions):
            h = book.positions[sym]
            if h.stop_price is None or sym in sleeve_set:
                continue
            j = cache.sym_index[sym]
            px = stop_fill_price(cache.open[u, j], cache.low[u, j], h.stop_price)
            if px is not None:
                _sell(book, cache, sym, h.quantity, px, cache.adv[u - 1, j] if u > 0 else np.nan, date, "stop", False)
                book.stopped_out[sym] = date
                book.stopped_pos[sym] = u
        # mark to market
        close_row = cache.close_ffill[u]
        pos_val = {s: h.quantity * close_row[cache.sym_index[s]] for s, h in book.positions.items()}
        equity = book.cash + float(sum(pos_val.values()))
        eq_vals.append(equity)
        w_rows.append({s: v / equity for s, v in pos_val.items()} if equity > 0 else {})
        # decide after the close
        exec_pos = u + 1 + lag
        if exec_pos > s1:
            continue
        cooldown = config.portfolio.stop_cooldown_days
        recent_stops = {k: book.stopped_out[k] for k, p in book.stopped_pos.items() if u - p < cooldown}
        decision_holdings = _projected_holdings(book, pending, cache, close_row) if pending else book.positions
        tp = generate_targets(
            data, config, date, decision_holdings, cache, equity=equity, stopped_out=recent_stops
        )
        for s, h in book.positions.items():
            if s in tp.stops:
                h.stop_price = tp.stops[s]
        for o in pending.values():  # in-flight entries carry the latest stop too
            for s in list(o.stops):
                if s in tp.stops:
                    o.stops[s] = max(o.stops[s], tp.stops[s])
        tq: Dict[str, int] = {}
        for s, w in tp.weights.items():
            j = cache.sym_index[s]
            px = close_row[j]
            if np.isfinite(px) and px > 0 and w > 0:
                tq[s] = int(math.floor(w * equity / px + 1e-6))
        reasons: Dict[str, str] = {}
        for s in set(tq) | set(book.positions):
            if s in tp.exits:
                reasons[s] = tp.exits[s]
            elif s in sleeve_set:
                reasons[s] = "sleeve"
            elif s not in tq and tp.regime_scale <= 0:
                reasons[s] = "regime"
            else:
                reasons[s] = "rebalance"
        pending[exec_pos] = _Order(decision_pos=u, target_qty=tq, stops=dict(tp.stops), reasons=reasons)

    idx = dates[s0 : s1 + 1]
    equity_s = pd.Series(eq_vals, index=idx, name="equity")
    prev = equity_s.shift(1)
    prev.iloc[0] = float(config.initial_capital)
    returns_s = (equity_s / prev - 1.0).rename("return")
    weights_df = pd.DataFrame(w_rows, index=idx, dtype="float64").fillna(0.0)
    weights_df.columns = [str(c) for c in weights_df.columns]
    trade_cols = [f.name for f in dataclasses.fields(Trade)]
    trades_df = pd.DataFrame([dataclasses.asdict(t) for t in book.trades], columns=trade_cols)
    metrics = compute_metrics(
        returns_s, equity_s, trades_df, weights_df, config.risk_free_annual, config.initial_capital
    )
    metrics["lag_days"] = float(lag)
    data_hash = data.data_hash or data.compute_hash()
    result = BacktestResult(
        equity=equity_s, returns=returns_s, weights=weights_df, trades=trades_df, metrics=metrics,
        config=config, data_hash=data_hash, notes=book.notes,
    )
    logger.info(
        "backtest %s..%s done in %.1fs (cache %.1fs): sharpe=%.2f cagr=%.3f",
        idx[0].date(), idx[-1].date(), time.perf_counter() - t0, cache.build_seconds,
        metrics.get("sharpe", float("nan")), metrics.get("cagr", float("nan")),
    )
    if record:
        record_run(result, config, tag=tag, lag_days=lag)
    return result


# ============================================================================
# run recording
# ============================================================================


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=15)
        return out.stdout if out.returncode == 0 else None
    except Exception:  # pragma: no cover - git missing
        return None


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def record_run(result: BacktestResult, config: EngineConfig, *, tag: str = "", lag_days: int = 0) -> Path:
    """Write the run directory and set ``result.run_id`` / ``result.run_dir``."""
    chash = config.config_hash()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{stamp}_{chash[:8]}"
    run_dir = Path(config.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    manifest = {
        "run_id": run_id,
        "tag": tag,
        "config_hash": chash,
        "git_commit": commit.strip() if commit else None,
        "git_dirty": bool(status.strip()) if status is not None else None,
        "data_hash": result.data_hash,
        "start": str(result.equity.index[0].date()) if len(result.equity) else None,
        "end": str(result.equity.index[-1].date()) if len(result.equity) else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics": _json_safe(result.metrics),
        "lag_days": int(lag_days),
    }
    (run_dir / "config.json").write_text(config.to_json())
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    result.returns.rename("return").rename_axis("date").to_frame().to_csv(run_dir / "returns.csv")
    result.equity.rename("equity").rename_axis("date").to_frame().to_csv(run_dir / "equity.csv")
    result.trades.to_csv(run_dir / "trades.csv", index=False)
    result.weights.rename_axis("date").to_parquet(run_dir / "weights.parquet")
    result.run_id = run_id
    result.run_dir = str(run_dir)
    logger.info("run recorded at %s", run_dir)
    return run_dir
