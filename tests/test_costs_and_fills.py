"""Costs and fills: orders fill at prices the decision could not see, on the tick grid, and cost money at the
dated statutory rates (PL-15); an ETF pays its own STT and its stray open is held near the close; a stock frozen at
its circuit cannot trade on the locked side (RB8)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from kite_connect.trading.gtt_stops import round_to_tick
from kite_connect.trading.nse_engine_executor import TICK, _tick
from kite_connect.trading.nse_instruments import SLABS, on_tick, round_price
from nse_engine.config import CostConfig
from nse_engine.costs import impact_bps, statutory_cost


DAY = "2025-06-02"


TICKS = tuple(t for _, t in SLABS)


#: NSE's slab edges (circular 33/2025) with a paisa and a tick either side, and the smallest prices, where float
#: division is least forgiving.
EDGES = sorted({round(b + d, 2) for b, _ in SLABS[:-1] for d in (-0.05, -0.01, 0.0, 0.01, 0.05)}
               | {0.01, 0.05, 0.10, 0.15, 0.30, 0.35})


def _prices(n: int = 4000, seed: int = 5) -> list:
    """Seeded prices from Rs 0.05 to 1.5 lakh at 0-3 decimals (on and off every grid), plus ``EDGES``."""
    rng = np.random.default_rng(seed)
    raw = np.exp(rng.uniform(np.log(0.05), np.log(150_000), n))
    return [round(float(p), int(d)) for p, d in zip(raw, rng.integers(0, 4, n))] + EDGES


@pytest.mark.parametrize("rounder", [round_price, round_to_tick], ids=["round_price", "round_to_tick"])
def test_tick_rounding_lands_on_the_grid_in_the_asked_direction(rounder):
    """Kite rejects a price off the tick (LN-T1): every order and GTT price lands on it, ``down`` never rises,
    ``up`` never falls, and nothing moves a tick or more (half a tick for ``nearest``)."""
    for p in _prices():
        for tick in TICKS:
            if p < tick:
                continue
            for mode, most in (("nearest", tick / 2), ("down", tick), ("up", tick)):
                r = rounder(p, tick, mode)
                assert on_tick(r, tick), (p, tick, mode, r)
                assert abs(r - p) <= most + 1e-9, (p, tick, mode, r)
                assert (mode != "down" or r <= p + 1e-9) and (mode != "up" or r >= p - 1e-9), (p, tick, mode, r)


def test_the_planned_limit_rounding_keeps_the_band_side():
    """``_tick`` (the plan's 0.05 grid): a sell limit rounds down, a buy limit up, by less than a tick."""
    for p in _prices():
        if p < TICK:
            continue
        down, up = _tick(p, "down"), _tick(p, "up")
        assert on_tick(down, TICK) and on_tick(up, TICK), p
        assert p - TICK - 1e-9 < down <= p + 1e-9 and p - 1e-9 <= up < p + TICK + 1e-9, (p, down, up)


def test_each_slab_tick_divides_the_next():
    """The coarser-slab fallback (``safe_tick``) is valid whichever neighbouring slab NSE assigned only because
    each tick is a whole multiple of the one below it."""
    assert all(on_tick(coarse, fine) for fine, coarse in zip(TICKS, TICKS[1:]))


def test_statutory_costs_are_charged_on_both_sides():
    buy = statutory_cost(100_000.0, "BUY", DAY)
    sell = statutory_cost(100_000.0, "SELL", DAY)
    assert buy > 0 and sell > 0
    assert sell > buy, "the sell side carries DP charges on top"


def test_costs_scale_with_traded_value():
    small = statutory_cost(10_000.0, "BUY", DAY)
    large = statutory_cost(1_000_000.0, "BUY", DAY)
    assert large > small * 50, "statutory charges are mostly proportional"


def test_statutory_charges_known_answer_at_todays_rates():
    """Rs 1 lakh of shares on 13 Nov 2026: STT 0.1% a side, stamp 0.015% on the buy, exchange 0.00297%, SEBI
    Rs 10 a crore, GST 18% on the last two, Rs 15.93 to the depository on the sell."""
    from nse_engine.costs import statutory_charges

    buy, sell = (statutory_charges(100_000.0, side, "2026-11-13") for side in ("BUY", "SELL"))
    assert (buy.stt, buy.stamp_duty, buy.dp) == (100.0, pytest.approx(15.0), 0.0)
    assert (buy.exchange, buy.sebi, buy.gst) == (pytest.approx(2.97), pytest.approx(0.1), pytest.approx(0.5526))
    assert buy.total == pytest.approx(118.6226) and sell.total == pytest.approx(119.5526)
    assert (sell.stamp_duty, sell.dp) == (0.0, 15.93)


@pytest.mark.lesson("L-RATES", pattern="F8", date="2026-11-14", source="PL-15",
                    summary="A rate could change in a cost model, or stay stale, with no dated record of its source")
def test_the_share_models_rates_are_the_dated_table_on_each_change_date_and_today():
    from datetime import date

    from nse_engine.validation import rates

    table = {}
    for charge, since, rate, source in rates.EQUITY:
        assert source.strip(), f"{charge} from {since} names no source"
        table.setdefault(charge, []).append((since, rate))
    assert set(table) == {"stt_delivery", "stamp_buy", "exchange", "sebi", "indirect_tax", "dp_sell"}
    for charge, rows in table.items():
        assert rows[0][0] == rates.START and [s for s, _ in rows] == sorted(s for s, _ in rows), charge
        for i, (since, rate) in enumerate(rows):
            assert rates.model_rate("equity", charge, since) == rate, f"{charge} from {since}"
            if i:
                eve = pd.Timestamp(since) - pd.Timedelta(days=1)
                assert rates.model_rate("equity", charge, eve) == rows[i - 1][1], f"{charge} on the eve of {since}"
        assert rates.model_rate("equity", charge, date.today()) == rows[-1][1], f"{charge} today"


@pytest.mark.lesson("L-RATES")
@pytest.mark.parametrize("model", ["fno", "mcx"])
def test_every_dated_rate_of_the_derivative_models_names_its_source(model):
    from nse_engine.validation import rates

    sources = {"fno": rates.FNO_SOURCES, "mcx": rates.MCX_SOURCES}[model]
    schedules = rates.schedules(model)
    assert schedules and {since for rows in schedules.values() for since, _ in rows} == set(sources)
    for name, rows in schedules.items():
        assert [s for s, _ in rows] == sorted(s for s, _ in rows), name
        assert rates.model_rate(model, name, rows[-1][0]) == rows[-1][1]


@pytest.mark.lesson("L-RATES")
def test_a_published_rate_not_applied_yet_is_an_open_lesson_until_it_is():
    from datetime import date, timedelta

    from nse_engine.validation import rates
    from tools import lessons

    ledger = {r["id"]: r for r in lessons.build()}
    for p in rates.PENDING:
        applied = rates.model_rate(p["model"], p["charge"], date.today())
        assert applied != p["published"], f"{p['model']}.{p['charge']} is applied now: take it out of PENDING"
        assert ledger[p["lesson"]]["status"] in ("open", "waived") and p["source"].strip()
    assert [rates.check_due(d) for d in ("2026-11-14", "2027-02-15", "2027-04-20", "2027-10-15")] == [
        date(2027, 2, 15), date(2027, 4, 15), date(2027, 10, 15), date(2028, 2, 15)]
    due = rates.check_due()
    assert not rates.status(due).startswith("!") and rates.status(due + timedelta(days=1)).startswith("!")


#: Zerodha's contract notes for the INFY sell test (PL-14; the PDFs stay in the gitignored research_inbox), each
#: note's charges in rupees as printed: (day, side, each fill's value, STT, stamp duty, exchange, SEBI, CGST + SGST,
#: the note's total charges).  The note rounds STT and stamp duty to the rupee and adds Rs 0.01 of taxable brokerage.
CONTRACT_NOTES = (
    ("2026-11-11", "BUY", (4611.20,), 5.00, 1.00, 0.14, 0.00, 0.02, 6.18),
    ("2026-11-16", "SELL", (1150.00, 1148.90), 2.00, 0.00, 0.07, 0.00, 0.02, 2.10),
)
#: The ledger's DP charge for both 16 Nov sales: one debit per stock and day (Rs 13 + 18% GST), not on the note.
DP_LEDGER_2026_11_16 = 15.34


@pytest.mark.lesson("L-COST-NOTES", pattern="F8", date="2026-11-11", source="A1M",
                    summary="No cost or margin figure has been compared with a broker contract note (PL-14, from the "
                            "INFY sell test's fills)")
@pytest.mark.parametrize("day, side, fills, stt, stamp, exchange, sebi, gst, total", CONTRACT_NOTES)
def test_the_share_cost_model_reproduces_zerodhas_contract_notes(day, side, fills, stt, stamp, exchange, sebi, gst,
                                                                 total):
    from nse_engine.costs import statutory_charges

    parts = [statutory_charges(v, side, day, dp_charge_inr=0.0) for v in fills]
    model = {k: sum(getattr(p, k) for p in parts) for k in ("stt", "stamp_duty", "exchange", "sebi", "gst")}
    assert (round(model["stt"]), round(model["stamp_duty"])) == (stt, stamp)
    assert model["exchange"] == pytest.approx(exchange, abs=0.005) and model["sebi"] == pytest.approx(sebi, abs=0.005)
    assert model["gst"] == pytest.approx(gst, abs=0.01)       # the note rounds CGST and SGST apart
    rounded = round(model["stt"]) + round(model["stamp_duty"]) + model["exchange"] + model["sebi"] + model["gst"]
    assert rounded == pytest.approx(total, abs=0.02)


@pytest.mark.lesson("L-COST-NOTES")
def test_the_model_never_charges_less_dp_than_the_ledger():
    from nse_engine.costs import statutory_charges

    assert statutory_charges(1150.00, "SELL", "2026-11-16").dp >= DP_LEDGER_2026_11_16


@pytest.mark.lesson("L-U25", pattern="F8", date="2026-09-30", source="U25",
                    summary="The metal ETFs were charged the delivery STT of shares, 0.1% a side")
def test_etfs_pay_their_own_stt_not_the_share_rate():
    from nse_engine.costs import statutory_charges, stt_for

    assert stt_for("GOLDBEES", "BUY", DAY) == stt_for("GOLDBEES", "SELL", DAY) == stt_for("LIQUIDBEES", "SELL", DAY) == 0.0
    assert (stt_for("SILVERBEES", "BUY", DAY), stt_for("SILVERBEES", "SELL", DAY)) == (0.0, 0.00001)
    assert stt_for("INFY", "SELL", DAY) == stt_for(None, "BUY", DAY) == 0.001
    assert statutory_charges(100_000.0, "SELL", DAY, symbol="GOLDBEES").stt == 0.0
    assert statutory_charges(100_000.0, "SELL", DAY, symbol="SILVERBEES").stt == pytest.approx(1.0)


@pytest.mark.lesson("L-D4", pattern="F6", date="2026-10-01", source="D4",
                    summary="A metal ETF's stray first print was taken as the open its orders fill at")
def test_an_etfs_stray_open_is_held_near_the_close_and_a_shares_open_is_kept():
    from nse_engine.costs import ETF_OPEN_BAND, fillable_open

    opens = [48.98, 43.70, 40.0]                          # GOLDBEES 24 Dec 2020 opened at 48.98 and closed at 43.73
    open_ = pd.DataFrame({"GOLDBEES": opens, "INFY": opens})
    close = pd.DataFrame({"GOLDBEES": [43.73] * 3, "INFY": [43.73] * 3})
    out = fillable_open(open_, close, {"GOLDBEES"})
    assert out["GOLDBEES"].tolist() == pytest.approx([43.73 * (1 + ETF_OPEN_BAND), 43.70, 43.73 * (1 - ETF_OPEN_BAND)])
    assert out["INFY"].tolist() == opens and open_["GOLDBEES"].tolist() == opens
    assert fillable_open(open_, close, set()) is open_


def test_impact_grows_with_participation():
    cfg = CostConfig()
    adv = 1e8
    light = impact_bps(1e5, adv, cfg)       # 0.1% of ADV
    heavy = impact_bps(5e6, adv, cfg)       # 5% of ADV
    assert cfg.spread_floor_bps <= light < heavy
    assert heavy > 2 * light, "square-root impact should grow with size"


def test_impact_never_below_the_spread_floor():
    cfg = CostConfig()
    assert impact_bps(1.0, 1e12, cfg) >= cfg.spread_floor_bps


@pytest.mark.slow
def test_backtest_fills_use_the_next_open_not_todays_close(store_path):
    """A one-day delay must change results; a same-day fill would make lag meaningless."""
    from nse_engine.deployment import load_deployment
    from nse_engine.engine import run_backtest
    from runners.run_nse_engine import _load_data

    cfg = load_deployment("config/nse_engine_deployed.json").engine.replace(
        start="2024-01-01", end="2024-12-31")
    data = _load_data(cfg, data_start="2022-01-03")
    base = run_backtest(data, cfg, record=False)
    lagged = run_backtest(data, cfg, record=False, lag_days=1)
    assert not np.allclose(base.returns.to_numpy(), lagged.returns.to_numpy(), atol=1e-12)


@pytest.mark.slow
def test_live_funding_never_spends_a_sale_at_its_own_open(store_path):
    """FND1 (LN-T11): with funding="live", no session's buys exceed the cash free before that open, and the run
    differs from the model's; it is a reference run only."""
    from nse_engine.deployment import load_deployment
    from nse_engine.engine import run_backtest
    from runners.run_nse_engine import _load_data

    cfg = load_deployment("config/nse_engine_deployed.json").engine.replace(start="2024-01-01", end="2024-12-31")
    data = _load_data(cfg, data_start="2022-01-03")
    model = run_backtest(data, cfg, record=False)
    live = run_backtest(data, cfg, record=False, funding="live")
    assert not np.allclose(model.returns.to_numpy(), live.returns.to_numpy(), atol=1e-12)
    t = live.trades.assign(flow=lambda d: np.where(d["side"] == "BUY", -1, 1) * d["value_inr"] - d["cost_inr"])
    cash = cfg.initial_capital
    for day, g in t.groupby("date", sort=True):
        assert -g.loc[g["side"] == "BUY", "flow"].sum() <= cash + 1e-6, day
        cash += g["flow"].sum()
    for bad in (dict(funding="sweep"), dict(funding="live", record=True), dict(funding="live", lag_days=1)):
        with pytest.raises(ValueError):
            run_backtest(data, cfg, **{"record": False, **bad})


@pytest.mark.slow
def test_trading_costs_reduce_returns(store_path):
    """Doubling modelled impact must lower CAGR and raise the cost drag."""
    from nse_engine.deployment import load_deployment
    from nse_engine.engine import run_backtest
    from runners.run_nse_engine import _load_data

    cfg = load_deployment("config/nse_engine_deployed.json").engine.replace(
        start="2024-01-01", end="2024-12-31")
    data = _load_data(cfg, data_start="2022-01-03")
    cheap = run_backtest(data, cfg, record=False)
    dear = run_backtest(data, cfg.replace(**{
        "costs.impact_coefficient_bps": cfg.costs.impact_coefficient_bps * 3,
        "costs.spread_floor_bps": cfg.costs.spread_floor_bps * 3}), record=False)
    assert dear.metrics["cagr"] < cheap.metrics["cagr"]
    assert dear.metrics["cost_drag"] > cheap.metrics["cost_drag"]


@pytest.mark.slow
def test_every_book_reproduces_its_recorded_backtest(store_path):
    """The nightly canary (LN-T13) on this machine, before a merge: every paper book's 2013-25 backtest against
    config/canary_expected.json, to the last trade.  A refactor step must leave it identical (owner, 26 Nov).  A store
    whose 2013-25 data changed, or a book configured since the record, is re-recorded with ``run_nse_engine canary
    --record-registry`` (part of every re-baseline), never waved through here."""
    from pathlib import Path

    from nse_engine.books import discover_books
    from runners.run_nse_engine import CANARY_EXPECTED, CANARY_WINDOW, _load_data, canary_compare, canary_stats

    books = {b.name: b.deployment.engine.replace(start=CANARY_WINDOW[0], end=CANARY_WINDOW[1])
             for b in discover_books()}
    data = _load_data(next(iter(books.values())), data_start="2012-01-02")   # as the nightly canary loads it
    now = {name: canary_stats(data, cfg) for name, cfg in books.items()}
    status, moved, fresh = canary_compare(now, json.loads(Path(CANARY_EXPECTED).read_text())["registry"])
    assert "deployed" in now and (status, moved, fresh) == ("ok", [], []), (status, moved, fresh)


# ---- the book's weights under name and sector caps (portfolio.capped_weights; tracker CVX1)
def _fill(r: np.ndarray, cap: float, total: float) -> np.ndarray:
    """min(cap, t r_i) summing to ``total`` (at most len(r) x cap): the k largest at the cap, the rest t r_i."""
    order = np.argsort(-r)
    rest = np.cumsum(r[order][::-1])[::-1]
    for k in range(len(r)):
        t = (total - k * cap) / rest[k]
        if t * r[order][k] <= cap + 1e-15:
            out = np.where(np.arange(len(r)) < k, cap, t * r[order])
            return out[np.argsort(order)]
    return np.full(len(r), cap)


def _water_filled(raw: pd.Series, cap: float, sector_cap: float, sectors) -> pd.Series:
    """The KKT solution (Boyd & Vandenberghe 2004, examples 5.2 and 5.4) of: weights nearest raw / sum(raw) in
    relative entropy, each at most ``cap``, each known sector at most ``sector_cap``, summing to 1 where the caps
    allow.  It is w_i = min(cap, c theta_s raw_i): one c for the book, theta_s < 1 only in a sector at its cap."""
    groups = {}
    for name in raw.index:
        s = sectors.get(name)
        groups.setdefault(("solo", name) if s is None else s, []).append(name)

    def at(c):
        w = {}
        for g, names in groups.items():
            r = raw[names].to_numpy(dtype=float)
            base = np.minimum(cap, c * r)
            if not isinstance(g, tuple) and base.sum() > sector_cap:
                base = _fill(r, cap, sector_cap)
            w.update(zip(names, base, strict=True))
        return pd.Series(w)[raw.index]

    lo, hi = 0.0, 1.0
    while at(hi).sum() < 1.0:
        hi *= 2.0
    for _ in range(200):
        lo, hi = ((lo + hi) / 2, hi) if at((lo + hi) / 2).sum() < 1.0 else (lo, (lo + hi) / 2)
    return at(hi)


def test_with_name_caps_alone_the_weights_are_water_filling():
    """The KKT conditions of the capped proportional book: uncapped names share one weight-to-score ratio, a
    capped name's score would have carried it past the cap, and the weights sum to 1 (or to every cap, when 1 is
    out of reach).  400 seeded books of 5 to 30 names."""
    from nse_engine.portfolio import capped_weights

    rng = np.random.default_rng(7)
    for _ in range(400):
        n = int(rng.integers(5, 31))
        raw = pd.Series(rng.lognormal(0.0, 1.0, n), index=[f"S{i}" for i in range(n)])
        cap = float(rng.uniform(0.6 / n, 0.5))
        w = capped_weights(raw, cap)
        assert np.allclose(w.to_numpy(), _fill(raw.to_numpy(), cap, min(1.0, n * cap)), atol=1e-12)
        free = w < cap - 1e-12
        ratio = (w[free] / raw[free]).to_numpy()
        assert np.ptp(ratio) <= 1e-9 * ratio.max() if free.any() else True
        assert (raw[~free] * (ratio[0] if free.any() else np.inf) >= cap - 1e-12).all()
        assert abs(w.sum() - min(1.0, n * cap)) <= 1e-12


def test_a_sector_over_its_cap_is_scaled_whole_not_water_filled():
    """One worked book: A1 (score 4) and A2-A4 (score 1) in one sector capped at 25%, twelve names of score 1
    with no sector, every name capped at 10%.  Today: A1 is capped at 10%, the sector then holds 10% + 3/19 and is
    scaled to 25% whole, so A1 = 25% x 1.9/4.9 and A2-A4 = 25%/4.9 each.  Water-filling keeps A1 at 10% and gives
    A2-A4 5% each.  The twelve others hold 6.25% either way.  Switching method moves real weights (tracker CVX1)."""
    from nse_engine.portfolio import capped_weights

    raw = pd.Series({"A1": 4.0, "A2": 1.0, "A3": 1.0, "A4": 1.0, **{f"B{i}": 1.0 for i in range(12)}})
    sectors = {"A1": "A", "A2": "A", "A3": "A", "A4": "A"}
    today = capped_weights(raw, 0.10, 0.25, sectors)
    assert today["A1"] == pytest.approx(0.25 * 1.9 / 4.9, abs=1e-12)
    assert today[["A2", "A3", "A4"]].to_numpy() == pytest.approx([0.25 / 4.9] * 3, abs=1e-12)
    filled = _water_filled(raw, 0.10, 0.25, sectors)
    assert filled["A1"] == pytest.approx(0.10, abs=1e-12) and filled[["A2", "A3", "A4"]].to_numpy() == \
        pytest.approx([0.05] * 3, abs=1e-12)
    for w in (today, filled):
        assert w.filter(like="B").to_numpy() == pytest.approx([0.0625] * 12, abs=1e-12)
        assert w.sum() == pytest.approx(1.0, abs=1e-12) and w[list(sectors)].sum() == pytest.approx(0.25, abs=1e-12)


def test_caps_that_do_not_settle_are_refused_not_clipped(monkeypatch):
    """A book that needs two rounds, given one, stops with an error rather than returning a half-capped book."""
    from nse_engine import portfolio as pf

    raw = pd.Series({"A": 10.0, "B": 3.0, "C": 1.0, "D": 1.0})
    assert pf.capped_weights(raw, 0.40).to_numpy() == pytest.approx([0.40, 0.36, 0.12, 0.12])
    monkeypatch.setattr(pf, "WEIGHT_CAP_MAX_ITER", 1)
    with pytest.raises(RuntimeError, match="did not settle in 1 rounds"):
        pf.capped_weights(raw, 0.40)


def test_circuit_locks_known_answers():
    """No range and a moved close is a lock; a flat day, a ranging day or a missing bar is not (RB8)."""
    from nse_engine.costs import circuit_locks

    nan = float("nan")
    close = np.array([[100.0], [95.0], [99.75], [99.75], [nan], [94.76], [90.0], [90.0]])
    high = np.array([[101.0], [95.0], [99.75], [99.75], [nan], [94.76], [91.0], [90.0]])
    low = np.array([[99.0], [95.0], [99.75], [99.75], [nan], [94.76], [89.0], [nan]])
    down, up = circuit_locks(high, low, close)
    assert down[:, 0].tolist() == [False, True, False, False, False, True, False, False]
    assert up[:, 0].tolist() == [False, False, True, False, False, False, False, False]


def test_circuit_locks_properties():
    """Over random panels: a bar is never both, a ranging bar is never a lock, and scaling prices changes nothing."""
    from nse_engine.costs import circuit_locks

    rng = np.random.default_rng(8)
    for _ in range(50):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, (300, 6)), axis=0))
        spread = rng.uniform(0, 1, close.shape) * (rng.uniform(0, 1, close.shape) > 0.1) * close * 0.02
        high, low = close + spread, close - spread
        close[rng.uniform(0, 1, close.shape) < 0.02] = np.nan
        down, up = circuit_locks(high, low, close)
        assert not (down & up).any()
        assert not ((down | up) & (spread > 1e-3 * np.nan_to_num(close, nan=1e9))).any()
        d2, u2 = circuit_locks(high * 37, low * 37, close * 37)
        assert (d2 == down).all() and (u2 == up).all()


def _locked_bar(panel, symbol, row, move):
    """The panel with ``symbol`` frozen on ``row`` at ``move`` from the close before it."""
    import dataclasses

    frames = {n: getattr(panel, n).copy() for n in ("open", "high", "low", "close")}
    col = frames["close"].columns.get_loc(symbol)
    px = float(frames["close"].iloc[row - 1, col]) * (1 + move)
    for f in frames.values():
        f.iloc[row, col] = px
    return dataclasses.replace(panel, **frames)


def _lock_config(panel, on):
    from nse_engine.config import EngineConfig

    dates = panel.dates
    return EngineConfig().replace(start=str(dates[300].date()), end=str(dates[-1].date()),
                                  **{"portfolio.respect_circuit_locks": on, "universe.top_n_liquid": 40,
                                     "universe.min_history_days": 60, "universe.liquidity_lookback_days": 40,
                                     "universe.min_median_value_inr": 0.0, "universe.min_price_inr": 0.0})


@pytest.mark.lesson("L-RB8", pattern="F5", date="2026-10-31", source="RB8",
                    summary="The backtest filled at the open of a stock frozen at its circuit")
@pytest.mark.parametrize("side, move, opposite", [("SELL", -0.02, 0.02), ("BUY", 0.02, -0.02)])
def test_a_locked_bar_stops_only_the_side_that_cannot_trade(panel, side, move, opposite):
    """Off, the order fills on the locked day as before.  On, it does not; the free side still fills (RB8)."""
    from nse_engine.engine import run_backtest

    trades = run_backtest(panel, _lock_config(panel, False), record=False).trades
    picked = trades[(trades["side"] == side) & (trades["reason"] == "rebalance") & (trades["date"] > panel.dates[320])]
    trade = picked.sort_values("value_inr", ascending=False).iloc[0]
    row, sym, day = panel.dates.get_loc(trade["date"]), trade["symbol"], trade["date"]

    def fills(data, on):
        out = run_backtest(data, _lock_config(data, on), record=False).trades
        return out[(out["symbol"] == sym) & (out["side"] == side) & (out["date"] == day)]

    locked = _locked_bar(panel, sym, row, move)
    assert len(fills(locked, False)) == 1, "legacy behaviour is unchanged"
    assert fills(locked, True).empty
    assert len(fills(_locked_bar(panel, sym, row, opposite), True)) == 1, "the free side still trades"


def test_a_stop_cannot_fill_on_a_locked_bar_and_goes_out_next_session(panel):
    """A stop under a lower lock waits, as a real GTT would, then sells at the next tradable open (RB8)."""
    from nse_engine.engine import run_backtest

    trades = run_backtest(panel, _lock_config(panel, False), record=False).trades
    stop = trades[(trades["side"] == "SELL") & (trades["reason"] == "stop") & (trades["date"] > panel.dates[320])].iloc[0]
    row, sym, day = panel.dates.get_loc(stop["date"]), stop["symbol"], stop["date"]
    data = _locked_bar(panel, sym, row, float(stop["price"]) / float(panel.close[sym].iloc[row - 1]) - 1.03)

    def sells(on):
        out = run_backtest(data, _lock_config(data, on), record=False).trades
        return out[(out["symbol"] == sym) & (out["side"] == "SELL") & (out["date"] >= day)]

    assert sells(False)["date"].iloc[0] == day, "legacy fills on the locked day"
    held = sells(True)
    assert held["date"].iloc[0] > day and held["reason"].iloc[0] == "stop"
