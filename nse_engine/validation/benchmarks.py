"""
Naive benchmarks under the engine's execution assumptions, and the gate the
strategy must pass against them.

Benchmarks (daily net simple returns on the trading calendar within
``[config.start, config.end]``):

* ``ew_hold_universe`` -- equal weight across the point-in-time liquidity
  universe, rebalanced monthly.
* ``momentum_12_1_top15`` -- top 15 by 12-1 momentum
  (``close[t - skip] / close[t - lookback] - 1`` with
  ``config.signals.momentum_lookback/skip``) within the same universe, equal
  weight, monthly.
* ``nifty50_price_index`` -- NIFTY50 PRICE index (no dividends, no costs);
* ``nifty50_tri`` -- NIFTY50 total return index when ``index_close`` has it;
  informational only.
* ``cash`` -- ``config.cash_yield_annual`` accrued daily.

Execution (same as the engine): decide after the close of the first session
``>= start`` and of the last session of each month; fill at the next open;
trade value per symbol capped at ``costs.max_participation`` x the 20-day
median traded value read at the decision date; per-side costs =
``nse_engine.costs`` statutory charges + square-root impact (if that module
cannot be imported, a flat 15 bp per side is used and the series is flagged
in ``Series.attrs["cost_model"]``); gross <= 1 (target values are set on equity
net of estimated costs); idle cash earns the cash yield.  Positions in a
symbol that stops trading are marked at its last close and written out at
that price (with sell costs) after ``stale_days`` sessions without a close.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .dsr import excess_sharpe

logger = logging.getLogger(__name__)

FALLBACK_COST_BPS = 15.0
CostFn = Callable[[np.ndarray, np.ndarray, pd.Timestamp], np.ndarray]


def _default_rf() -> float:
    try:
        from nse_engine.config import EngineConfig

        return float(EngineConfig().risk_free_annual)
    except Exception:  # pragma: no cover
        return 0.065


def make_cost_fn(config: Any) -> tuple:
    """Return ``(cost_fn, cost_model_label)``.

    ``cost_fn(signed_trade_value_inr, adv_inr, date) -> cost_inr`` per symbol.
    """
    try:
        from nse_engine import costs as ncosts  # lazy: built in parallel

        ccfg = config.costs

        def cost_fn(trade: np.ndarray, adv: np.ndarray, date: pd.Timestamp) -> np.ndarray:
            out = np.zeros(trade.shape[0])
            for i in np.flatnonzero(np.abs(trade) > 1e-9):
                v = float(abs(trade[i]))
                side = "BUY" if trade[i] > 0 else "SELL"
                imp = ncosts.impact_bps(v, float(adv[i]), ccfg) * v / 1e4
                out[i] = imp + ncosts.statutory_cost(v, side, date, ccfg.dp_charge_inr)
            return out

        return cost_fn, "nse_engine.costs"
    except Exception as exc:  # ImportError or missing attributes
        logger.warning("nse_engine.costs unavailable (%s); benchmarks use a FLAT %.0f bp "
                       "per side cost", exc, FALLBACK_COST_BPS)

        def flat(trade: np.ndarray, adv: np.ndarray, date: pd.Timestamp) -> np.ndarray:
            return np.abs(trade) * FALLBACK_COST_BPS / 1e4

        return flat, f"flat_{FALLBACK_COST_BPS:.0f}bp_fallback"


def _default_universe_mask(data: Any, config: Any) -> pd.DataFrame:
    """Point-in-time universe mask via ``nse_engine.universe`` (sleeve ETFs excluded)."""
    from nse_engine.universe import compute_universe_panel  # lazy

    exclude = (config.sleeves.gold_symbol, config.sleeves.silver_symbol)
    return compute_universe_panel(data, config.universe, exclude=exclude).mask


def decision_positions(dates: pd.DatetimeIndex, s0: int, s1: int) -> np.ndarray:
    """Row positions of decisions: ``s0`` and each month's last session in [s0, s1)."""
    d = pd.DatetimeIndex(dates)
    period = d.to_period("M")
    last_of_month = np.r_[period[1:] != period[:-1], True]
    pos = [p for p in np.flatnonzero(last_of_month) if s0 <= p < s1]
    return np.array(sorted(set([s0] + pos)), dtype=int)


def simulate_weights(data: Any, targets: Dict[int, Dict[int, float]], s0: int, s1: int,
                     config: Any, cost_fn: CostFn, stale_days: int = 21) -> pd.Series:
    """Daily net returns of a target-weight schedule.

    ``targets`` maps decision row position -> {column index: weight}; each is
    filled at the open of the next row.  Returns cover rows ``s0..s1``.
    """
    dates = data.dates
    opn = data.open.to_numpy(dtype="float64")
    cls = data.close.to_numpy(dtype="float64")
    try:
        from nse_engine.costs import median_traded_value

        adv_df = median_traded_value(data.value, config.costs.adv_lookback_days)
    except Exception:
        adv_df = data.value.astype("float64").fillna(0.0).rolling(
            config.costs.adv_lookback_days, min_periods=1).median()
    adv = adv_df.to_numpy(dtype="float64")
    m = cls.shape[1]
    y_daily = float(config.cash_yield_annual) / 252.0  # same accrual as the engine
    max_part = float(config.costs.max_participation)

    q = np.zeros(m)
    px = np.full(m, np.nan)
    stale = np.zeros(m, dtype=int)
    cash = float(config.initial_capital)
    prev_value = cash
    out = np.empty(s1 - s0 + 1)
    pending: Optional[Dict[int, float]] = None
    pending_pos = -1
    for k, t in enumerate(range(s0, s1 + 1)):
        cash *= 1.0 + y_daily
        if pending is not None:
            o = opn[t]
            mark = np.where(np.isfinite(o), o, px)
            held = q != 0
            equity = cash + float(np.nansum(q[held] * mark[held]))
            w = np.zeros(m)
            for j, wt in pending.items():
                w[j] = wt
            tradable = np.isfinite(o) & (o > 0)
            w[~tradable] = 0.0
            if w.sum() > 1.0:
                w /= w.sum()
            cur_val = np.where(held, q * np.nan_to_num(mark), 0.0)
            cap = max_part * np.nan_to_num(adv[pending_pos], nan=0.0)
            est_cost = 0.0
            for _ in range(2):  # set targets on equity net of estimated costs
                tgt = w * max(equity - est_cost, 0.0)
                trade = np.where(tradable, tgt - cur_val, 0.0)
                trade = np.clip(trade, -cap, cap)
                cost = cost_fn(trade, adv[pending_pos], dates[t])
                est_cost = float(cost.sum())
            dq = np.where(tradable, trade / np.where(tradable, o, 1.0), 0.0)
            q = q + dq
            q[np.abs(q) < 1e-12] = 0.0
            cash -= float(trade.sum()) + est_cost
            pending = None
        c = cls[t]
        has = np.isfinite(c)
        px = np.where(has, c, px)
        stale = np.where(has, 0, stale + 1)
        dead = (q != 0) & (stale >= stale_days) & np.isfinite(px)
        if dead.any():
            val = q[dead] * px[dead]
            trade = np.zeros(m)
            trade[dead] = -val
            cash += float(val.sum()) - float(cost_fn(trade, np.zeros(m), dates[t]).sum())
            q[dead] = 0.0
        value = cash + float(np.nansum(q[q != 0] * px[q != 0]))
        out[k] = value / prev_value - 1.0 if prev_value > 0 else 0.0
        prev_value = value
        if t in targets and t < s1:
            pending = targets[t]
            pending_pos = t
    return pd.Series(out, index=dates[s0:s1 + 1], name="return")


def _window_positions(dates: pd.DatetimeIndex, start: Any, end: Any) -> tuple:
    s0 = int(dates.searchsorted(pd.Timestamp(start), side="left"))
    s1 = int(dates.searchsorted(pd.Timestamp(end), side="right")) - 1
    if s0 >= len(dates) or s1 <= s0:
        raise ValueError(f"benchmark window {start}..{end} has fewer than 2 sessions in data")
    return s0, s1


def run_benchmarks(data: Any, config: Any,
                   universe_fn: Optional[Callable[[Any, Any], pd.DataFrame]] = None,
                   cost_fn: Optional[CostFn] = None, momentum_top_n: int = 15,
                   stale_days: int = 21) -> Dict[str, pd.Series]:
    """Daily net returns of the naive benchmarks (see module docstring).

    ``universe_fn(data, config) -> date x symbol bool DataFrame`` of
    point-in-time universe membership (default: ``nse_engine.universe``).
    ``cost_fn`` overrides the cost model (tests).
    """
    dates = pd.DatetimeIndex(data.dates)
    s0, s1 = _window_positions(dates, config.start, config.end)
    if cost_fn is None:
        cost_fn, cost_label = make_cost_fn(config)
    else:
        cost_label = "custom"
    mask_df = (universe_fn or _default_universe_mask)(data, config)
    mask_df = mask_df.reindex(index=dates, columns=data.close.columns, fill_value=False)
    mask = mask_df.to_numpy(dtype=bool)
    close_ff = data.close.astype("float64").ffill().to_numpy()
    look = int(config.signals.momentum_lookback)
    skip = int(config.signals.momentum_skip)

    ew_t: Dict[int, Dict[int, float]] = {}
    mom_t: Dict[int, Dict[int, float]] = {}
    for p in decision_positions(dates, s0, s1):
        members = np.flatnonzero(mask[p])
        ew_t[p] = {int(j): 1.0 / members.size for j in members} if members.size else {}
        if p - look >= 0 and members.size:
            with np.errstate(divide="ignore", invalid="ignore"):
                score = close_ff[p - skip, members] / close_ff[p - look, members] - 1.0
            ok = np.isfinite(score)
            ranked = members[ok][np.argsort(-score[ok], kind="stable")][:momentum_top_n]
            mom_t[p] = {int(j): 1.0 / ranked.size for j in ranked} if ranked.size else {}
        else:
            mom_t[p] = {}

    out: Dict[str, pd.Series] = {}
    for name, tg in (("ew_hold_universe", ew_t), ("momentum_12_1_top15", mom_t)):
        s = simulate_weights(data, tg, s0, s1, config, cost_fn, stale_days=stale_days)
        s.name = name
        s.attrs["cost_model"] = cost_label
        s.attrs["execution"] = "decide at close, fill next open, monthly rebalance"
        out[name] = s

    idx_col = getattr(getattr(config, "regime", None), "index_symbol", "NIFTY50")
    if idx_col in getattr(data, "index_close", pd.DataFrame()).columns:
        lvl = data.index_close[idx_col].astype("float64")
        nifty = lvl.pct_change(fill_method=None).iloc[s0:s1 + 1]
        nifty.iloc[0] = 0.0 if not np.isfinite(nifty.iloc[0]) else nifty.iloc[0]
        nifty = nifty.fillna(0.0).rename("nifty50_price_index")
        nifty.attrs["note"] = f"{idx_col} PRICE index (no dividends, no costs)"
        out["nifty50_price_index"] = nifty
    else:
        logger.warning("index_close has no %s column; NIFTY benchmark skipped", idx_col)

    tri_col = f"{idx_col}_TRI"
    if tri_col in getattr(data, "index_close", pd.DataFrame()).columns:
        tri = data.index_close[tri_col].astype("float64").pct_change(fill_method=None).iloc[s0:s1 + 1]
        tri.iloc[0] = 0.0 if not np.isfinite(tri.iloc[0]) else tri.iloc[0]
        tri = tri.fillna(0.0).rename("nifty50_tri")
        tri.attrs["note"] = f"{tri_col} total return index (dividends reinvested, no costs)"
        out["nifty50_tri"] = tri

    y = float(config.cash_yield_annual) / 252.0
    out["cash"] = pd.Series(y, index=dates[s0:s1 + 1], name="cash")
    return out


def benchmark_gate(returns: pd.Series, benchmarks: Dict[str, pd.Series], margin: float = 0.3,
                   rf_annual: Optional[float] = None,
                   required: Iterable[str] = ("ew_hold_universe", "momentum_12_1_top15"),
                   ) -> Dict[str, Any]:
    """Compare annualised excess Sharpe of the strategy with each benchmark.

    Each comparison uses the dates common to the strategy and that benchmark.
    The gate passes only if the strategy beats every ``required`` benchmark
    (present in ``benchmarks``) by at least ``margin`` Sharpe; a missing
    required benchmark fails the gate.  Others are reported with
    ``beats`` (strategy > benchmark) and do not affect ``passed``.
    """
    rf = _default_rf() if rf_annual is None else float(rf_annual)
    strat = pd.Series(returns, dtype="float64").dropna()
    strat.index = pd.DatetimeIndex(strat.index)
    required = list(required)
    rows: Dict[str, Any] = {}
    for name, bench in benchmarks.items():
        b = pd.Series(bench, dtype="float64").dropna()
        b.index = pd.DatetimeIndex(b.index)
        joined = pd.concat([strat.rename("s"), b.rename("b")], axis=1, join="inner").dropna()
        s_sr = excess_sharpe(joined["s"], rf) if len(joined) > 1 else float("nan")
        b_sr = excess_sharpe(joined["b"], rf) if len(joined) > 1 else float("nan")
        diff = s_sr - b_sr
        is_req = name in required
        rows[name] = {
            "strategy_excess_sharpe": s_sr,
            "benchmark_excess_sharpe": b_sr,
            "difference": diff,
            "n_obs": int(len(joined)),
            "required": is_req,
            "margin": margin if is_req else 0.0,
            "beats": bool(np.isfinite(diff) and diff > 0),
            "passed": bool(np.isfinite(diff) and diff >= (margin if is_req else 0.0)),
        }
    missing = [r for r in required if r not in rows]
    passed = not missing and all(rows[r]["passed"] for r in required)
    return {
        "passed": bool(passed),
        "margin": margin,
        "rf_annual": rf,
        "strategy_excess_sharpe": excess_sharpe(strat, rf),
        "benchmarks": rows,
        "missing_required": missing,
    }
