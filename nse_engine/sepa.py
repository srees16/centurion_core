"""
SEPA-lite breakout sleeve (tracker O9-A, plan 5x): Mark Minervini's Trend Template, volatility-contraction
(VCP) bases and pivot breakouts with tight stops, as a separate equity family.

End-of-day and event-driven, on the adjusted store and the core's universe:

* each close, a stock in the universe that passes all eight Trend Template criteria and closes above its
  base's pivot for the first time, on volume above its 50-day average, while the market gate is on, is a
  candidate; candidates are taken by 12-month relative strength while slots are free;
* it is bought at the next open if that open is no more than 3% above the pivot (else the trade is
  missed), at 10% of equity;
* from the next session a 7.5% stop (moved to entry once a close is 22.5% up) fills at the stop or a lower
  open; a close below the 20-day mean within 10 sessions of the breakout, the largest one-day fall since
  the base began on above-average volume, or a close below the 50-day mean sells at the next open.

Swing points are confirmed ``swing`` sessions after they print, so no base is drawn with hindsight.
Results go to their own registry (``data/nse_engine/runs_sepa``) and ``evaluate`` applies plan 5x's
gates.  CLI::

    python -m nse_engine.sepa run [--no-record]
    python -m nse_engine.sepa evaluate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SEPA_RUNS_DIR = "data/nse_engine/runs_sepa"
COMBINED_WEIGHT = 0.25
BOOK_RF = 0.065


@dataclass(frozen=True)
class SepaConfig:
    """Plan 5x O9-A: the book's numbers, with the translations marked there."""

    name: str = "O9-A SEPA-lite breakout sleeve (Minervini)"
    swing: int = 5
    base_min: int = 15
    base_max: int = 325
    min_contractions: int = 2
    max_contractions: int = 6
    first_max_depth: float = 0.35
    first_max_depth_bear: float = 0.50
    bear_index_drawdown: float = 0.20
    last_max_depth: float = 0.10
    index_fall_check: float = 0.08
    depth_vs_index: float = 2.5
    dryup_near_min: float = 1.10
    rs_min: float = 70.0
    chase_limit: float = 0.03
    slots: int = 10
    initial_slots: int = 5
    slot_weight: float = 0.10
    stop: float = 0.075
    breakeven_gain: float = 0.225
    sma20_window: int = 10
    initial_capital: float = 750_000.0
    start: str = "2013-01-01"
    end: str = "2025-12-31"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def config_hash(self) -> str:
        from nse_engine.costs import COST_MODEL_VERSION

        payload = {k: v for k, v in asdict(self).items() if k not in ("name", "start", "end")}
        payload.update(kind="sepa", cost_model=COST_MODEL_VERSION)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class SepaResult:
    config: SepaConfig
    returns: pd.Series
    equity: pd.Series
    trades: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    data_hash: str
    run_id: str = ""
    run_dir: Optional[str] = None


# ---------------------------------------------------------------- the base

def confirmed_swings(high: np.ndarray, low: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Boolean arrays: swing high / low at i (the extreme of i-k..i+k); usable from session i+k on."""
    n = len(high)
    sh, sl = np.zeros(n, bool), np.zeros(n, bool)
    for i in range(k, n - k):
        window_h, window_l = high[i - k: i + k + 1], low[i - k: i + k + 1]
        if np.isfinite(high[i]) and high[i] >= np.nanmax(window_h):
            sh[i] = True
        if np.isfinite(low[i]) and low[i] <= np.nanmin(window_l):
            sl[i] = True
    return sh, sl


def vcp_pivot(i: int, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray,
              sh: np.ndarray, sl: np.ndarray, index: np.ndarray, cfg: SepaConfig) -> Optional[Dict[str, Any]]:
    """The base ending at session ``i`` (swings confirmed by i), or None when it is not a VCP."""
    last = i - cfg.swing                                   # latest session whose swing status is known
    lo = max(0, i - cfg.base_max)
    highs = [j for j in range(lo, last + 1) if sh[j]]
    # base start: the latest swing high no later close (before today) exceeded, at least base_min old
    start = None
    for j in reversed(highs):
        if i - j < cfg.base_min:
            continue
        if np.nanmax(close[j + 1: i]) < high[j]:
            start = j
            break
    if start is None:
        return None
    # alternating swing highs and lows after the start: (high, low) pairs are the contractions
    events = sorted([(j, "H") for j in range(start, last + 1) if sh[j]] + [(j, "L") for j in range(start + 1, last + 1) if sl[j]])
    seq: List[Tuple[int, str]] = []
    for j, kind in events:
        if seq and seq[-1][1] == kind:                    # keep the more extreme of two in a row
            prev = seq[-1][0]
            better = high[j] > high[prev] if kind == "H" else low[j] < low[prev]
            if better and not (kind == "H" and prev == start):
                seq[-1] = (j, kind)
            continue
        seq.append((j, kind))
    if not seq or seq[0][0] != start:
        return None
    pairs = [(seq[t][0], seq[t + 1][0]) for t in range(0, len(seq) - 1, 2) if seq[t][1] == "H" and seq[t + 1][1] == "L"]
    if not (cfg.min_contractions <= len(pairs) <= cfg.max_contractions):
        return None
    depths = [(high[h] - low[l]) / high[h] for h, l in pairs]
    if any(b > a for a, b in zip(depths, depths[1:])):
        return None
    idx_window = index[start: i + 1]
    idx_dd = 1.0 - index[i] / np.nanmax(index[max(0, i - 252): i + 1])
    first_cap = cfg.first_max_depth_bear if idx_dd > cfg.bear_index_drawdown else cfg.first_max_depth
    if depths[0] > first_cap or depths[-1] > cfg.last_max_depth:
        return None
    peak = np.nanmax(idx_window)
    idx_fall = 1.0 - np.nanmin(idx_window[int(np.nanargmax(idx_window)):]) / peak
    if idx_fall >= cfg.index_fall_check and depths[0] > cfg.depth_vs_index * idx_fall:
        return None
    h_last, l_last = pairs[-1]
    pivot_at = next(j for j, kind in reversed(seq) if kind == "H")     # the latest swing high
    contraction_vol = volume[h_last: l_last + 1]
    avg50 = np.nanmean(volume[max(0, l_last - 49): l_last + 1])
    if not (np.nanmean(contraction_vol) < avg50 and np.nanmin(contraction_vol) <= cfg.dryup_near_min * np.nanmin(volume[start: i])):
        return None
    return {"pivot": float(high[pivot_at]), "base_start": start, "contractions": len(pairs), "depths": depths}


# ---------------------------------------------------------------- the simulation

def run_sepa(cfg: SepaConfig = SepaConfig(), data=None, store_dir: Optional[str] = None) -> SepaResult:
    from nse_engine.config import CostConfig, EngineConfig
    from nse_engine.costs import impact_bps, median_traded_value, statutory_cost
    from nse_engine.metrics import compute_metrics
    from nse_engine.universe import compute_universe_panel
    from runners.run_nse_engine import _load_data

    engine_cfg = EngineConfig().replace(start=cfg.start, end=cfg.end,
                                        **({"data.store_dir": store_dir} if store_dir else {}))
    data = data if data is not None else _load_data(engine_cfg, data_start="2011-01-01")
    costs = CostConfig()
    close, open_, high, low, vol = (data.close.astype("float64"), data.open.astype("float64"),
                                    data.high.astype("float64"), data.low.astype("float64"),
                                    data.volume.astype("float64"))
    members = compute_universe_panel(data, engine_cfg.universe, exclude=engine_cfg.sleeves.symbols).mask
    members = members.reindex(index=close.index, columns=close.columns).fillna(False)
    sma = {n: close.rolling(n, min_periods=n).mean() for n in (20, 50, 150, 200)}
    sma200_past = sma[200].shift(21)
    hi52, lo52 = high.rolling(252, min_periods=200).max(), low.rolling(252, min_periods=200).min()
    vol50 = vol.rolling(50, min_periods=40).mean()
    adv = median_traded_value(data.value.astype("float64"), costs.adv_lookback_days)
    rs = (close / close.shift(252) - 1).where(members).rank(axis=1, pct=True) * 100
    template = ((close > sma[150]) & (close > sma[200]) & (sma[150] > sma[200]) & (sma[200] > sma200_past)
                & (sma[50] > sma[150]) & (sma[50] > sma[200]) & (close > sma[50])
                & (close >= 1.30 * lo52) & (close >= 0.75 * hi52) & (rs >= cfg.rs_min) & members)
    nifty = data.index_close["NIFTY50"].astype("float64").ffill()
    gate = (nifty > nifty.rolling(200).mean()) & (
        ((high >= hi52) & members).sum(axis=1) > ((low <= lo52) & members).sum(axis=1))
    dates = close.index
    s0, s1 = int(dates.searchsorted(pd.Timestamp(cfg.start))), int(dates.searchsorted(pd.Timestamp(cfg.end), side="right"))
    swings: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    arrays = {s: (high[s].to_numpy(), low[s].to_numpy(), close[s].to_numpy(), vol[s].to_numpy()) for s in close.columns}
    nifty_np = nifty.to_numpy()

    cash, positions, trades, pending_buys, pending_sells = float(cfg.initial_capital), {}, [], [], []
    curve: Dict[pd.Timestamp, float] = {}
    invested: Dict[pd.Timestamp, float] = {}

    def charge(value: float, side: str, d, sym: str) -> float:
        a = float(adv[sym].iloc[dates.get_loc(d) - 1]) if dates.get_loc(d) > 0 else math.nan
        return statutory_cost(value, side, d, costs.dp_charge_inr) + value * impact_bps(value, a, costs) / 1e4

    def close_trade(sym: str, d, px: float, reason: str) -> None:
        nonlocal cash
        p = positions.pop(sym)
        value = p["shares"] * px
        cash += value - charge(value, "SELL", d, sym)
        trades.append({**{k: v for k, v in p.items() if k not in ("stop",)}, "exit_date": str(d.date()),
                       "exit_price": px, "exit_reason": reason,
                       "pnl_inr": value - p["shares"] * p["entry_price"] - p["entry_cost"]})

    for t in range(s0, s1):
        d = dates[t]
        # 1. orders decided at the last close fill at this open
        for sym, reason in pending_sells:
            if sym in positions and np.isfinite(open_[sym].iloc[t]):
                close_trade(sym, d, float(open_[sym].iloc[t]), reason)
        for cand in pending_buys:
            sym, pivot = cand["symbol"], cand["pivot"]
            o = float(open_[sym].iloc[t])
            if sym in positions or not np.isfinite(o):
                continue
            if o > pivot * (1 + cfg.chase_limit):
                trades.append({"symbol": sym, "entry_date": str(d.date()), "exit_reason": "missed: open above the limit",
                               "pivot": pivot, "open": o})
                continue
            equity = cash + sum(p["shares"] * float(close[s].iloc[t - 1]) for s, p in positions.items())
            shares = int(cfg.slot_weight * equity // o)
            value = shares * o
            cost = charge(value, "BUY", d, sym) if shares > 0 else 0.0
            if shares <= 0 or value + cost > cash:
                continue
            cash -= value + cost
            c_np = arrays[sym][2]
            base_falls = c_np[cand["base_start"] + 1: t] / c_np[cand["base_start"]: t - 1] - 1
            positions[sym] = {"symbol": sym, "entry_date": str(d.date()), "entry_pos": t, "entry_price": o,
                              "shares": shares, "entry_cost": cost, "pivot": pivot, "stop": o * (1 - cfg.stop),
                              "contractions": cand["contractions"],
                              "max_drop": float(np.nanmin(base_falls)) if len(base_falls) else 0.0}
        pending_buys, pending_sells = [], []
        # 2. stops, live from the session after entry: fill at the stop or a lower open
        for sym in list(positions):
            p = positions[sym]
            if t <= p["entry_pos"]:
                continue
            lo_t, o_t = float(low[sym].iloc[t]), float(open_[sym].iloc[t])
            if np.isfinite(lo_t) and lo_t <= p["stop"]:
                close_trade(sym, d, min(o_t, p["stop"]) if np.isfinite(o_t) else p["stop"], "stop")
        # 3. mark and decide exits at the close
        for sym, p in positions.items():
            c, c_prev = float(close[sym].iloc[t]), float(close[sym].iloc[t - 1])
            if not np.isfinite(c):
                continue
            if c >= p["entry_price"] * (1 + cfg.breakeven_gain):
                p["stop"] = max(p["stop"], p["entry_price"])
            drop = c / c_prev - 1 if np.isfinite(c_prev) and c_prev > 0 else 0.0
            heavy = float(vol[sym].iloc[t]) > float(vol50[sym].iloc[t])
            if t - p["entry_pos"] <= cfg.sma20_window and c < float(sma[20][sym].iloc[t]):
                pending_sells.append((sym, "below the 20-day mean after the breakout"))
            elif drop < 0 and drop < p["max_drop"] and heavy and t > p["entry_pos"]:
                pending_sells.append((sym, "largest fall since the base began, on volume"))
            elif c < float(sma[50][sym].iloc[t]):
                pending_sells.append((sym, "below the 50-day mean"))
            p["max_drop"] = min(p["max_drop"], drop)
        held = sum(p["shares"] * float(close[s].iloc[t]) for s, p in positions.items() if np.isfinite(close[s].iloc[t]))
        curve[d] = cash + held
        invested[d] = held / curve[d] if curve[d] > 0 else 0.0
        # 4. entries: first close above a VCP pivot on volume, gate on, slots free
        if not bool(gate.iloc[t]):
            continue
        closed = [x for x in trades if "pnl_inr" in x][-5:]
        allowed = cfg.slots if len(closed) == 5 and sum(x["pnl_inr"] for x in closed) > 0 else cfg.initial_slots
        free = allowed - len(positions)
        if free <= 0:
            continue
        cands = []
        row = template.iloc[t]
        for sym in row.index[row.to_numpy()]:
            if sym in positions:
                continue
            h, l, c, v = arrays[sym]
            if not (np.isfinite(c[t]) and v[t] > float(vol50[sym].iloc[t])):
                continue
            if sym not in swings:
                swings[sym] = confirmed_swings(h, l, cfg.swing)
            sh, sl = swings[sym]
            base = vcp_pivot(t, h, l, c, v, sh, sl, nifty_np, cfg)
            if base and c[t] > base["pivot"] >= c[t - 1]:
                cands.append({"symbol": sym, "rs": float(rs[sym].iloc[t]), **base})
        pending_buys = sorted(cands, key=lambda x: -x["rs"])[:free]

    eq = pd.Series(curve, name="equity")
    rets = eq.pct_change().fillna(eq.iloc[0] / cfg.initial_capital - 1.0)
    done = [x for x in trades if "pnl_inr" in x]
    pnl = np.array([x["pnl_inr"] for x in done])
    m = compute_metrics(rets, eq, rf_annual=BOOK_RF, initial_capital=cfg.initial_capital)
    m.update(trades=len(done), missed=sum(1 for x in trades if str(x.get("exit_reason", "")).startswith("missed")),
             hit_rate=float((pnl > 0).mean()) if len(pnl) else None,
             win_loss=float(pnl[pnl > 0].mean() / -pnl[pnl <= 0].mean()) if (pnl > 0).any() and (pnl <= 0).any() else None,
             exit_reasons={r: sum(x["exit_reason"] == r for x in done) for r in sorted({x["exit_reason"] for x in done})},
             average_invested=float(np.mean(list(invested.values()))) if invested else 0.0)
    return SepaResult(cfg, rets, eq, trades, m, getattr(data, "data_hash", "") or data.compute_hash())


def record(result: SepaResult, runs_dir: str = SEPA_RUNS_DIR) -> str:
    from nse_engine.validation.trials import record_result

    cfg = result.config
    from kite_connect.options.backtest import _run_id

    result.run_id = result.run_id or _run_id(cfg.config_hash())
    run_dir = record_result(result, tag=cfg.name, runs_dir=runs_dir, window=(cfg.start, cfg.end),
                            extra={"family": "equity_sleeve", "round": "O9"})
    pd.DataFrame(result.trades).to_csv(Path(run_dir) / "trades.csv", index=False)
    return run_dir


def evaluate(runs_dir: str = SEPA_RUNS_DIR, book_runs: str = "data/nse_engine/runs",
             equity_store: str = "data/nse_engine/store") -> Dict[str, Any]:
    """Plan 5x O9: gate 1, the sleeve's deflated Sharpe at its family's N; gate 2, 0.75 x the live book +
    0.25 x the sleeve must raise CAGR, excess Sharpe and alpha with MaxDD at most 2 points worse."""
    from kite_connect.options.structures import _live_book_returns, _nifty_tri_returns
    from nse_engine.validation.diagnostics import alpha_beta
    from nse_engine.validation.dsr import deflated_sharpe, performance_summary
    from nse_engine.validation.trials import TrialRegistry

    reg = TrialRegistry(runs_dir)
    trials = reg.list_trials()
    run_id = str(trials["run_id"].iloc[-1])
    sleeve = reg.load_returns(run_id)
    book, book_run = _live_book_returns(book_runs)
    book = book.reindex(sleeve.index).dropna()
    sleeve = sleeve.reindex(book.index).fillna(0.0)
    tri = _nifty_tri_returns(equity_store, book.index)
    mat = reg.returns_matrix(window=(SepaConfig().start, SepaConfig().end))
    dsr = deflated_sharpe(sleeve, trials_matrix=mat if mat.shape[1] > 1 else None,
                          n_trials=None if mat.shape[1] > 1 else 1, rf_annual=BOOK_RF)["dsr"]
    comb = (1 - COMBINED_WEIGHT) * book + COMBINED_WEIGHT * sleeve
    base, cs, ss = performance_summary(book, BOOK_RF), performance_summary(comb, BOOK_RF), performance_summary(sleeve, BOOK_RF)
    ba, ca = alpha_beta(book, tri, rf_annual=BOOK_RF)["alpha_annual"], alpha_beta(comb, tri, rf_annual=BOOK_RF)["alpha_annual"]
    checks = {"cagr": cs["cagr"] > base["cagr"], "excess_sharpe": cs["excess_sharpe"] > base["excess_sharpe"],
              "alpha": ca > ba, "max_drawdown": cs["max_drawdown"] >= base["max_drawdown"] - 0.02}
    gate1 = bool(ss["excess_sharpe"] > 0 and dsr >= 0.95)
    return {"round": "O9", "run_id": run_id, "live_book_run": book_run, "family_n": int(mat.shape[1]),
            "sleeve": {k: ss[k] for k in ("cagr", "excess_sharpe", "max_drawdown")}, "dsr": dsr, "gate1": gate1,
            "correlation_with_book": float(np.corrcoef(book, sleeve)[0, 1]),
            "base": {**{k: base[k] for k in ("cagr", "excess_sharpe", "max_drawdown")}, "alpha_annual": ba},
            "combined": {**{k: cs[k] for k in ("cagr", "excess_sharpe", "max_drawdown")}, "alpha_annual": ca,
                         "checks": checks},
            "gate2": all(checks.values()),
            "verdict": "PASS: paper-trade O9-A" if gate1 and all(checks.values()) else "O9 closed: O9-A failed its gates"}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="SEPA-lite breakout sleeve (O9-A, plan 5x)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--no-record", action="store_true")
    p.add_argument("--start")
    p.add_argument("--end")
    sub.add_parser("evaluate")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cmd == "run":
        from dataclasses import replace

        cfg = SepaConfig()
        cfg = replace(cfg, start=args.start or cfg.start, end=args.end or cfg.end)
        res = run_sepa(cfg)
        where = "not recorded" if args.no_record else record(res)
        print(where, json.dumps({k: res.metrics.get(k) for k in ("cagr", "sharpe", "max_drawdown", "trades", "missed",
                                                                  "hit_rate", "win_loss", "exit_reasons")}, default=str))
        return 0
    print(json.dumps(evaluate(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
