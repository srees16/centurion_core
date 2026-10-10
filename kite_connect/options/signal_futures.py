"""
Options round 2 (tracker O3, plan § 5r): the four strategies of
github.com/PyPatel/Options-Trading-Strategies-in-Python (commit c7b8a0d) on
NIFTY futures, as pre-registered on 9 Oct 2026, 19:41 IST.

The repository times the S&P 500 future with a put-call ratio or TRIN band
crossing or a VIX level, and trades three stocks on a 55-day breakout.  Here
each rule reads NSE data (the F&O store: options for the put-call ratio,
``market.parquet`` for India VIX and breadth, ``underlying.parquet`` for
NIFTY 50) and trades the near-month NIFTY future: decided on a session's
close, filled at the next session's close, unlevered, rolled at expiry,
charged with ``fno_costs``.  ``evaluate`` applies § 5r's gates, reusing
§ 5q's overlay on E4.  CLI::

    python -m kite_connect.options.signal_futures run --candidates X1,X2,X3,X4 \\
        --fo-store data/nse_engine/fo_store --runs-dir data/nse_engine/runs_options
    python -m kite_connect.options.signal_futures evaluate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from kite_connect.options.fno_costs import FNO_COST_MODEL_VERSION, futures_charges
from kite_connect.options.options_config import OptionsConfig
from kite_connect.options.theory import BUY, CALL, PUT, SELL
from nse_engine.data import fo_store

logger = logging.getLogger(__name__)

RULES = ("bands", "vix", "breakout", "expiry_week")
INDICATORS = ("pcr", "trin")
#: Sessions of history loaded before the window, for the 55-day channel and the bands.
WARMUP_DAYS = 200
EXTENDED_WINDOW = ("2007-01-02", "2025-12-31")
#: Gate 2's base: E4's run on the current equity data, and its 2007-25 run for the reported view,
#: both cost model 4 (tracker IC1; §§ 5r-5s used the model-3 runs 20261008T080306839468Z_93cf6c4d
#: and 20261001T070903385179Z_93cf6c4d).
E4_RUN_ID = "20261009T185901820673Z_93cf6c4d"
E4_EXT_RUN_ID = "20261009T191853176431Z_93cf6c4d"
OVERLAY_WEIGHT = 0.25
#: Reported beside gate 2: one NIFTY lot (~Rs 16 lakh of notional) on the Rs 30 lakh book.
ONE_LOT_WEIGHT = 0.5


@dataclass(frozen=True)
class SignalConfig:
    """One of § 5r's strategies.

    ``bands``: the ``indicator`` (put-call ratio by volume, or log TRIN) against
    its ``window``-session mean +/- ``band_k`` sigma, stop bands ``stop_k``
    sigma beyond, and an ``abs_stop`` on NIFTY from the entry.  ``vix``: long
    while India VIX >= ``vix_level``, out at +``take_profit`` / -``stop_loss``.
    ``breakout``: long above the prior ``breakout_days`` high, short below the
    low, out across their mean.  ``expiry_week`` (round 3, § 5s): long through
    the week of each monthly expiry.
    """

    name: str
    rule: str
    indicator: str = ""
    window: int = 20
    band_k: float = 1.5
    stop_k: float = 2.0
    abs_stop: float = 0.01
    vix_level: float = 22.0
    take_profit: float = 0.05
    stop_loss: float = 0.05
    breakout_days: int = 55
    symbol: str = "NIFTY"
    slippage_bp: float = 1.0
    initial_capital: float = 2_000_000.0
    start: str = "2013-01-01"
    end: str = "2025-12-31"

    def __post_init__(self):
        if self.rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}, got {self.rule!r}")
        if self.rule == "bands" and self.indicator not in INDICATORS:
            raise ValueError(f"a bands rule needs an indicator in {INDICATORS}, got {self.indicator!r}")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def config_hash(self) -> str:
        """Behaviour only (name and dates left out), with the cost model and the instrument."""
        payload = {k: v for k, v in self.to_dict().items() if k not in ("name", "start", "end")}
        payload.update(fno_cost_model=FNO_COST_MODEL_VERSION, instrument="near-month future")
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


CANDIDATES: Dict[str, SignalConfig] = {
    "X1": SignalConfig("O3-X1 PCR bands (PyPatel)", "bands", "pcr", window=20),
    "X2": SignalConfig("O3-X2 TRIN bands (PyPatel)", "bands", "trin", window=22),
    "X3": SignalConfig("O3-X3 India VIX >= 22 (PyPatel)", "vix"),
    "X4": SignalConfig("O3-X4 55-day breakout (PyPatel)", "breakout"),
}


@dataclass
class SignalResult:
    config: SignalConfig
    returns: pd.Series
    equity: pd.Series
    trades: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    data_hash: str
    store_dir: str = ""
    run_id: str = ""
    run_dir: Optional[str] = None


# ---------------------------------------------------------------- inputs

def put_call_ratio(options: pd.DataFrame) -> pd.Series:
    """Put contracts traded over call contracts traded, all expiries, per session."""
    by = options.groupby(["date", "option_type"])["contracts"].sum().unstack("option_type")
    return (by[PUT] / by[CALL].where(by[CALL] > 0)).rename("pcr")


def log_trin(market: pd.DataFrame) -> pd.Series:
    """log[(advancers / decliners) / (advancing volume / declining volume)]."""
    ad = market["advancers"] / market["decliners"].where(market["decliners"] > 0)
    vol = market["adv_volume"] / market["dec_volume"].where(market["dec_volume"] > 0)
    trin = ad / vol.where(vol > 0)
    return np.log(trin.where(trin > 0)).rename("trin")


def bands(x: pd.Series, window: int, band_k: float, stop_k: float) -> pd.DataFrame:
    """The repository's bands: sigma is the root mean, over window - 1 sessions, of (x - that day's mean)^2."""
    mean = x.rolling(window).mean()
    sigma = np.sqrt(((x - mean) ** 2).rolling(window - 1).mean())
    upper, lower = mean + band_k * sigma, mean - band_k * sigma
    return pd.DataFrame({"x": x, "prev": x.shift(1), "mean": mean, "upper": upper, "lower": lower,
                         "upper_stop": upper + stop_k * sigma, "lower_stop": lower - stop_k * sigma})


def breakout_target(close: pd.Series, days: int) -> pd.Series:
    """Long above the prior ``days`` high until a close below their mean; short below the low until above it.

    Each side is carried forward on its own before they are added (the
    repository adds first, so its mean exit never applied).
    """
    prior = close.shift(1).rolling(days)
    high, low, mean = prior.max(), prior.min(), prior.mean()
    long_leg = pd.Series(np.where(close > high, 1.0, np.where(close < mean, 0.0, np.nan)), index=close.index)
    short_leg = pd.Series(np.where(close < low, -1.0, np.where(close > mean, 0.0, np.nan)), index=close.index)
    return (long_leg.ffill().fillna(0.0) + short_leg.ffill().fillna(0.0)).rename("target")


def settlement_sessions(rows: pd.DataFrame, max_gap_days: int = 7) -> Dict[pd.Timestamp, pd.Timestamp]:
    """Each expiry label of ``rows`` (futures or options) mapped to the session its contracts settled on:
    one settlement source per contract (Lean's pattern, tracker LN-T20).

    A label settles on its last row, which NSE moves off a holiday (24 Apr 2014 to 23 Apr; 27 Nov 2008 to
    28 Nov).  NSE also relabels contracts part-way through their life (27 Feb 2014 to 26 Feb in December
    2013; 29 Jun 2023 to 28 Jun; every Thursday expiry to a Tuesday on 1 Aug 2025): when a label's rows
    stop before its date and a label of the same month starts the very next session, the contracts carry
    on under that one, so the label settles when it does (the latest-settling such label, followed
    recursively: 26 Mar 2026 to 31 Mar to 30 Mar).  Left out: a label still trading at the end of
    ``rows``, and one whose rows stop more than ``max_gap_days`` before its date with no such successor.
    """
    span = rows.groupby("expiry")["date"].agg(["min", "max"])
    sessions = pd.DatetimeIndex(sorted(rows["date"].unique()))
    following = dict(zip(sessions[:-1], sessions[1:]))
    starting = span.reset_index().groupby("min")["expiry"].apply(list).to_dict()
    settled: Dict[pd.Timestamp, Optional[pd.Timestamp]] = {}

    def resolve(label: pd.Timestamp) -> Optional[pd.Timestamp]:
        if label not in settled:
            last = span.at[label, "max"]
            if label > sessions[-1] and last == sessions[-1]:
                settled[label] = None                                  # still trading
            elif last < label:
                successors = [e for e in starting.get(following.get(last), [])
                              if e != label and (e.year, e.month) == (label.year, label.month)]
                if successors:
                    ends = [t for t in map(resolve, successors) if t is not None]
                    settled[label] = max(ends) if ends else None
                else:
                    settled[label] = last if last >= label - pd.Timedelta(days=max_gap_days) else None
            else:
                settled[label] = last
        return settled[label]

    resolved = {e: resolve(e) for e in span.index}
    return {pd.Timestamp(e): pd.Timestamp(t) for e, t in resolved.items() if t is not None}


def rekey_to_settlement(rows: pd.DataFrame) -> pd.DataFrame:
    """``rows`` with each contract's expiry replaced by the session it settles on (``settlement_sessions``),
    so a relabelled contract runs on under one key and settles on the right day.  A label the map leaves
    out (still trading at the end of ``rows``, or unexplained) keeps its own date."""
    expiry = rows["expiry"].map(settlement_sessions(rows))
    return rows.assign(expiry=pd.to_datetime(expiry.fillna(rows["expiry"])))


def expiry_week_target(dates: pd.DatetimeIndex, expiries) -> pd.Series:
    """The position to take at the next session's close so it is held through each expiry's week (§ 5s Y1).

    Held from the close of the last session before the expiry's ISO week to the
    close of the session before expiry day; nothing when the expiry is the
    week's first session.  ``expiries`` are the sessions the contracts settled
    on (``settlement_sessions``).  A calendar, so it is decided one session ahead.
    """
    held = pd.Series(0.0, index=dates)
    pos = {d: i for i, d in enumerate(dates)}
    iso = dates.isocalendar()
    week_start = pd.Series(np.arange(len(dates)), index=dates).groupby([iso["year"].to_numpy(),
                                                                         iso["week"].to_numpy()]).min()
    for expiry in sorted(set(expiries)):
        if expiry not in pos:
            continue
        y, w = expiry.isocalendar()[:2]
        first, last = int(week_start[(y, w)]) - 1, pos[expiry] - 1   # enter on `first`'s close, exit on `last`'s
        if first >= 0 and last > first:
            held.iloc[first:last] = 1.0
    return held.shift(-1).fillna(0.0).rename("target")


def _crossed_up(row, level: str) -> bool:
    return bool(row["x"] > row[level] and row["prev"] < row[level])


def _crossed_down(row, level: str) -> bool:
    return bool(row["x"] < row[level] and row["prev"] > row[level])


class _Futures:
    """Near-month NIFTY futures prices: close if traded, else settlement."""

    def __init__(self, futures: pd.DataFrame):
        f = futures[(futures["close"] > 0) | (futures["settle"] > 0)].copy()
        f["price"] = np.where((f["contracts"] > 0) & (f["close"] > 0), f["close"], f["settle"])
        f = f[f["price"] > 0]
        self.price = f.set_index(["date", "expiry"])["price"].to_dict()
        self.expiries = f.groupby("date")["expiry"].apply(lambda s: sorted(set(s))).to_dict()

    def front(self, date: pd.Timestamp) -> pd.Timestamp:
        """The nearest expiry after ``date``: held overnight, rolled on its expiry day's close."""
        return next(e for e in self.expiries[date] if e > date)

    def at(self, date: pd.Timestamp, expiry: pd.Timestamp, default: float) -> float:
        """The contract's price that session, or ``default`` (its last mark) when it has no row."""
        return float(self.price.get((date, expiry), default))


# ---------------------------------------------------------------- simulation

def _trade_cost(units: float, price: float, side: str, date: pd.Timestamp, cfg: SignalConfig,
                opt_cfg: OptionsConfig) -> float:
    notional = abs(units) * price
    slip = max(opt_cfg.slippage.min_ticks * opt_cfg.slippage.tick_size, cfg.slippage_bp / 1e4 * price)
    return futures_charges(notional, side, date, cfg=opt_cfg.charges).total + abs(units) * slip


def run_strategy(cfg: SignalConfig, store_dir: str, opt_cfg: OptionsConfig = OptionsConfig()) -> SignalResult:
    """Replay ``cfg`` session by session over its window (§ 5r)."""
    start, end = pd.Timestamp(cfg.start), pd.Timestamp(cfg.end)
    load_from = (start - pd.Timedelta(days=int(WARMUP_DAYS * 1.6))).strftime("%Y-%m-%d")
    futures = fo_store.load_futures(store_dir, load_from, cfg.end, cfg.symbol)
    fut = _Futures(rekey_to_settlement(futures))                  # rolls on the settlement session
    dates = pd.DatetimeIndex(sorted(fut.expiries))
    index = fo_store.load_underlying(store_dir, cfg.symbol).reindex(dates).ffill()

    signal = pd.DataFrame(index=dates)
    if cfg.rule == "bands":
        if cfg.indicator == "pcr":
            x = put_call_ratio(fo_store.load_options(store_dir, load_from, cfg.end, cfg.symbol))
        else:
            x = log_trin(fo_store.load_market(store_dir))
        signal = bands(x.reindex(dates), cfg.window, cfg.band_k, cfg.stop_k)
    elif cfg.rule == "vix":
        signal["vix"] = fo_store.load_market(store_dir)["india_vix"].reindex(dates)
    elif cfg.rule == "expiry_week":
        signal["target"] = expiry_week_target(dates, settlement_sessions(futures).values())
    else:
        signal["target"] = breakout_target(index, cfg.breakout_days)

    window = dates[(dates >= start) & (dates <= end)]
    equity, units, contract, mark, entry_level = cfg.initial_capital, 0.0, None, 0.0, 0.0
    pending: Optional[tuple] = None                    # (target sign, reason) decided on the last close
    trades: List[Dict[str, Any]] = []
    curve, costs_total, rolls, in_market = {}, 0.0, 0, 0
    for d in window:
        # 1. the overnight position moves with its contract
        if units:
            px = fut.at(d, contract, mark)
            equity += units * (px - mark)
            mark = px
            in_market += 1
        # 2. yesterday's decision fills at today's close
        if pending is not None:
            target, reason = pending
            pending = None
            if units:
                px = fut.at(d, contract, mark)
                cost = _trade_cost(units, px, SELL if units > 0 else BUY, d, cfg, opt_cfg)
                equity -= cost
                costs_total += cost
                t = trades[-1]
                t.update(exit_date=d, exit_level=float(index[d]), exit_price=px, exit_reason=reason)
                t["pnl_inr"] = equity - t["equity_before"]
                units = 0.0
            if target:
                contract = fut.front(d)
                px = fut.at(d, contract, np.nan)
                units = target * equity / px
                cost = _trade_cost(units, px, BUY if target > 0 else SELL, d, cfg, opt_cfg)
                trades.append({"side": "long" if target > 0 else "short", "entry_date": d, "entry_reason": reason,
                               "entry_level": float(index[d]), "entry_price": px, "units": units,
                               "equity_before": equity})
                equity -= cost
                costs_total += cost
                mark, entry_level = px, float(index[d])
        # 3. a contract on its expiry day rolls at the close
        if units and contract <= d:
            old_px = fut.at(d, contract, mark)
            contract = fut.front(d)
            mark = fut.at(d, contract, np.nan)
            cost = (_trade_cost(units, old_px, SELL if units > 0 else BUY, d, cfg, opt_cfg)
                    + _trade_cost(units, mark, BUY if units > 0 else SELL, d, cfg, opt_cfg))
            equity -= cost
            costs_total += cost
            rolls += 1
        curve[d] = equity
        # 4. decide on today's close
        held = int(np.sign(units))
        pending = _decide(cfg, signal.loc[d], held, float(index[d]), entry_level)

    eq = pd.Series(curve, name="equity")
    rets = eq.pct_change().fillna(eq.iloc[0] / cfg.initial_capital - 1.0)
    for t in trades:
        t.pop("equity_before")
    metrics = _metrics(rets, trades, costs_total, rolls, in_market / max(len(window), 1))
    return SignalResult(cfg, rets, eq, trades, metrics, fo_store.data_hash(store_dir), store_dir=store_dir)


def _decide(cfg: SignalConfig, row: pd.Series, held: int, level: float, entry: float) -> Optional[tuple]:
    """(target, reason) when the rule acts on this close, else None."""
    if cfg.rule in ("breakout", "expiry_week"):
        target = int(row["target"])
        return (target, "channel" if cfg.rule == "breakout" else "calendar") if target != held else None
    if cfg.rule == "vix":
        if held == 0:
            return (1, "vix") if row["vix"] >= cfg.vix_level else None
        if level >= entry * (1 + cfg.take_profit):
            return 0, "target"
        if level <= entry * (1 - cfg.stop_loss):
            return 0, "stop"
        return None
    if held == 0:                                      # bands: the repository's order
        if _crossed_up(row, "upper"):
            return 1, "upper band"
        if _crossed_down(row, "lower"):
            return -1, "lower band"
        return None
    if held > 0:
        if _crossed_down(row, "mean"):
            return 0, "mean"
        if _crossed_up(row, "upper_stop"):
            return 0, "stop band"
        if level <= entry * (1 - cfg.abs_stop):
            return 0, "abs stop"
        return None
    if _crossed_up(row, "mean"):
        return 0, "mean"
    if _crossed_down(row, "lower_stop"):
        return 0, "stop band"
    if level >= entry * (1 + cfg.abs_stop):
        return 0, "abs stop"
    return None


def _metrics(rets: pd.Series, trades: List[Dict[str, Any]], costs: float, rolls: int,
             time_in_market: float) -> Dict[str, Any]:
    from nse_engine.validation.dsr import performance_summary

    m = performance_summary(rets, rf_annual=0.0)
    closed = [t for t in trades if "pnl_inr" in t]
    pnl = np.array([t["pnl_inr"] for t in closed])
    m.update(sharpe=m["excess_sharpe"], calmar=(m["cagr"] / abs(m["max_drawdown"]) if m["max_drawdown"] else None),
             trades=len(trades), closed_trades=len(closed),
             long_trades=sum(t["side"] == "long" for t in trades),
             win_rate=float((pnl > 0).mean()) if len(pnl) else None,
             avg_trade_inr=float(pnl.mean()) if len(pnl) else None,
             worst_trade_inr=float(pnl.min()) if len(pnl) else None,
             best_trade_inr=float(pnl.max()) if len(pnl) else None,
             avg_days_held=float(np.mean([(t["exit_date"] - t["entry_date"]).days for t in closed])) if closed else None,
             time_in_market=time_in_market, rolls=rolls, charges_inr=costs,
             exit_reasons={r: sum(t.get("exit_reason") == r for t in closed)
                           for r in sorted({t.get("exit_reason") for t in closed})})
    return m


def _market_hash(store_dir: str) -> str:
    path = Path(store_dir) / "market.parquet"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else ""


def record(result: SignalResult, runs_dir: str, round_label: str = "O3") -> str:
    """Write the run to the options registry (U32), with its trades."""
    from kite_connect.options.backtest import _run_id
    from nse_engine.validation.trials import record_result

    cfg = result.config
    result.run_id = result.run_id or _run_id(cfg.config_hash())
    run_dir = record_result(result, tag=cfg.name, runs_dir=runs_dir, window=(cfg.start, cfg.end),
                            extra={"family": "options", "round": round_label, "fno_cost_model": FNO_COST_MODEL_VERSION,
                                   "market_hash": _market_hash(result.store_dir)})
    pd.DataFrame(result.trades).to_csv(Path(run_dir) / "trades.csv", index=False)
    return run_dir


# ---------------------------------------------------------------- evaluation (§ 5r)

def evaluate(options_runs: Optional[str] = None, book_runs: Optional[str] = None, start: str = "2013-01-01",
             end: str = "2025-12-31", record_combined: bool = True, candidates: Optional[Dict[str, Any]] = None,
             label: str = "O3") -> Dict[str, Any]:
    """Gate 1 on each of a round's ``candidates`` (default round 2's) at N = the whole options family,
    gate 2 on E4 for the best passer; ``label`` tags the combined run."""
    from kite_connect.options.backtest import (
        BOOK_RF,
        BOOK_RUNS_DIR,
        OPTIONS_RUNS_DIR,
        _record_combined,
        overlay,
    )
    from nse_engine.validation.dsr import deflated_sharpe, performance_summary
    from nse_engine.validation.pbo import cscv_pbo
    from nse_engine.validation.trials import TrialRegistry

    options_runs, book_runs = options_runs or OPTIONS_RUNS_DIR, book_runs or BOOK_RUNS_DIR
    reg = TrialRegistry(options_runs)
    by_tag = reg.list_trials().set_index("run_id")["tag"]
    mat = reg.returns_matrix(window=(start, end))
    candidates = candidates or CANDIDATES
    keys = {cfg.name: key for key, cfg in candidates.items()}
    report: Dict[str, Any] = {"options_configurations": int(mat.shape[1]), "gate1": {}}
    if mat.shape[1] >= 2:
        report["options_pbo"] = {k: v for k, v in cscv_pbo(mat, n_splits=16, rf_annual=0.0).items()   # excess of cash
                                 if k != "logits"}
    passing = []
    for run_id in mat.columns:
        key = keys.get(by_tag[run_id])
        if key is None:                                # round 1's sleeves count in N only
            continue
        dsr = deflated_sharpe(mat[run_id], trials_matrix=mat, rf_annual=0.0)
        summ = performance_summary(mat[run_id], rf_annual=0.0)
        metrics = json.loads((Path(options_runs) / run_id / "manifest.json").read_text())["metrics"]
        ok = bool(summ["excess_sharpe"] > 0 and dsr["dsr"] >= 0.95)
        report["gate1"][key] = {"run_id": run_id, "sharpe": summ["excess_sharpe"], "cagr": summ["cagr"],
                                "max_drawdown": summ["max_drawdown"], "dsr": dsr["dsr"], "pass": ok,
                                **{k: metrics.get(k) for k in ("trades", "win_rate", "avg_trade_inr",
                                                               "time_in_market", "charges_inr", "calmar")}}
        if ok:
            passing.append((-summ["excess_sharpe"], list(candidates).index(key), key, run_id))
    report["extended"] = _extended(reg, by_tag, keys)
    if not passing:
        report["selected"] = None
        report["verdict"] = f"{label} closed: no strategy passed gate 1"
        return report

    _, _, key, run_id = min(passing)
    report["selected"] = key
    book_reg = TrialRegistry(book_runs)
    book = book_reg.load_returns(E4_RUN_ID)
    book = book[(book.index >= pd.Timestamp(start)) & (book.index <= pd.Timestamp(end))]
    strategy = reg.load_returns(run_id)
    combined = overlay(book, strategy, OVERLAY_WEIGHT)
    base, comb = performance_summary(book, BOOK_RF), performance_summary(combined, BOOK_RF)
    one_lot = performance_summary(overlay(book, strategy, ONE_LOT_WEIGHT), BOOK_RF)
    checks = {"cagr": (comb["cagr"], comb["cagr"] >= 0.25),
              "excess_sharpe": (comb["excess_sharpe"], comb["excess_sharpe"] >= base["excess_sharpe"]),
              "max_drawdown": (comb["max_drawdown"], comb["max_drawdown"] >= base["max_drawdown"] - 0.02)}
    since_2017 = lambda r: performance_summary(r[r.index >= pd.Timestamp("2017-01-01")], BOOK_RF)["excess_sharpe"]
    report["gate2"] = {"e4": base, "combined": comb, "checks": {k: {"value": v, "pass": p} for k, (v, p) in checks.items()},
                       "pass": all(p for _, p in checks.values()),
                       "reported": {"one_lot_overlay": one_lot,
                                    "calmar_e4": base["cagr"] / abs(base["max_drawdown"]),
                                    "calmar_combined": comb["cagr"] / abs(comb["max_drawdown"]),
                                    "excess_sharpe_2017_25_e4": since_2017(book),
                                    "excess_sharpe_2017_25_combined": since_2017(combined)}}
    if record_combined:
        e4_manifest = json.loads((Path(book_runs) / E4_RUN_ID / "manifest.json").read_text())
        manifest = json.loads((Path(options_runs) / run_id / "manifest.json").read_text())
        run_dir = _record_combined(combined, comb, e4_manifest, manifest, run_id, key, book_runs, (start, end),
                                   label=label, weight=OVERLAY_WEIGHT)
        book_mat = book_reg.returns_matrix(data_hash=e4_manifest["data_hash"], window=(start, end),
                                           cost_model=int(e4_manifest.get("cost_model", 3)))
        n_total = int(book_mat.shape[1]) + int(mat.shape[1])
        report["gate2"]["run_dir"] = run_dir
        report["gate2"]["reported"].update(
            book_configurations=int(book_mat.shape[1]), total_trials=n_total,
            dsr_total=deflated_sharpe(book_mat[Path(run_dir).name], trials_matrix=book_mat, n_trials=n_total,
                                      rf_annual=BOOK_RF)["dsr"])
    report["verdict"] = ("PASS: paper-trade the strategy beside E4" if report["gate2"]["pass"]
                         else f"FAIL: {label} closed without re-tuning")
    return report


def _extended(reg, by_tag: pd.Series, keys: Dict[str, str]) -> Dict[str, Any]:
    """§ 5r: the four over 2007-25, reported, never gating."""
    from nse_engine.validation.dsr import performance_summary

    mat = reg.returns_matrix(window=EXTENDED_WINDOW)
    return {"window": EXTENDED_WINDOW,
            "strategies": {keys[by_tag[r]]: {"run_id": r, **performance_summary(mat[r], rf_annual=0.0)}
                           for r in mat.columns if by_tag[r] in keys}}


# ---------------------------------------------------------------- CLI

def main(argv: Optional[List[str]] = None) -> int:
    from kite_connect.options.backtest import OPTIONS_RUNS_DIR

    ap = argparse.ArgumentParser(description="Options round 2 (O3): PyPatel's signal strategies on NIFTY futures")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="backtest and record candidates in the options registry")
    p.add_argument("--candidates", default=",".join(CANDIDATES))
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--fo-store", default="data/nse_engine/fo_store")
    p.add_argument("--runs-dir", default=OPTIONS_RUNS_DIR)
    p = sub.add_parser("evaluate", help="§ 5r's gates over the recorded runs")
    p.add_argument("--no-record", action="store_true", help="do not record the combined run in the book registry")
    p.add_argument("--out", default="data/nse_engine/o3_results.json")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cmd == "run":
        for key in args.candidates.split(","):
            cfg = CANDIDATES[key]
            res = run_strategy(replace(cfg, start=args.start or cfg.start, end=args.end or cfg.end), args.fo_store)
            print(key, record(res, args.runs_dir), {k: res.metrics[k] for k in ("sharpe", "cagr", "max_drawdown")})
        return 0
    report = evaluate(record_combined=not args.no_record)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in ("gate1", "selected", "verdict")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
