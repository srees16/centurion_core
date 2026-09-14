"""Tests for the NSE engine core (costs .. engine) on synthetic data.

Run: python -m unittest tests.test_nse_engine_core -v
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from nse_engine.allocator import allocate
from nse_engine.config import AllocatorConfig, CostConfig, UniverseConfig
from nse_engine.costs import (
    cap_quantity,
    impact_bps,
    simulate_fill,
    statutory_charges,
    statutory_cost,
)
from nse_engine.engine import EngineCache, generate_targets, run_backtest
from nse_engine.metrics import compute_metrics
from nse_engine.portfolio import (
    apply_no_trade_buffer,
    capped_weights,
    stop_fill_price,
    trailing_stop,
)
from nse_engine.regime import RISK_OFF, RISK_ON, apply_hysteresis, compute_regime
from nse_engine.signals import ewmac_raw, pit_forecast_scalar
from nse_engine.types import Holding, MarketData
from nse_engine.universe import compute_universe_panel, select_universe

from tests.nse_engine_fixtures import make_market_data, small_config


def _replace_frames(data: MarketData, **frames) -> MarketData:
    kw = dict(
        dates=data.dates, open=data.open, high=data.high, low=data.low, close=data.close,
        volume=data.volume, value=data.value, index_close=data.index_close, etfs=data.etfs,
        sectors=data.sectors, source=data.source,
    )
    kw.update(frames)
    out = MarketData(**kw)
    out.data_hash = out.compute_hash()
    return out


class TestCosts(unittest.TestCase):
    def test_schedule_dates(self):
        v = 1_000_000.0
        old = statutory_charges(v, "BUY", "2017-06-30", dp_charge_inr=0)
        gst = statutory_charges(v, "BUY", "2017-07-01", dp_charge_inr=0)
        self.assertAlmostEqual(old.stt, 1000.0)
        self.assertAlmostEqual(old.stamp_duty, 100.0)  # 0.01% before 2020-07-01
        self.assertAlmostEqual(old.exchange, 32.5)
        self.assertAlmostEqual(old.sebi, 1.0)
        self.assertAlmostEqual(old.gst, 0.15 * 33.5)
        self.assertAlmostEqual(gst.gst, 0.18 * 33.5)
        self.assertAlmostEqual(statutory_charges(v, "BUY", "2020-06-30").stamp_duty, 100.0)
        self.assertAlmostEqual(statutory_charges(v, "BUY", "2020-07-01").stamp_duty, 150.0)
        self.assertAlmostEqual(statutory_charges(v, "BUY", "2024-09-30").exchange, 32.5)
        self.assertAlmostEqual(statutory_charges(v, "BUY", "2024-10-01").exchange, 29.7)
        sell = statutory_charges(v, "SELL", "2024-10-01", dp_charge_inr=15.93)
        self.assertEqual(sell.stamp_duty, 0.0)
        self.assertAlmostEqual(sell.dp, 15.93)
        self.assertEqual(statutory_charges(v, "BUY", "2024-10-01", dp_charge_inr=15.93).dp, 0.0)
        self.assertAlmostEqual(sell.total, 1000 + 29.7 + 1.0 + 0.18 * 30.7 + 15.93)
        self.assertEqual(statutory_cost(0.0, "SELL", "2024-10-01"), 0.0)

    def test_impact_formula(self):
        cfg = CostConfig()
        self.assertAlmostEqual(impact_bps(1e5, 1e7, cfg), cfg.spread_floor_bps + cfg.impact_coefficient_bps)
        self.assertAlmostEqual(impact_bps(4e5, 1e7, cfg), cfg.spread_floor_bps + 2 * cfg.impact_coefficient_bps)

    def test_participation_cap(self):
        cfg = CostConfig(max_participation=0.05)
        # ADV 1e6 at price 100 -> max 500 shares
        self.assertEqual(cap_quantity(2000, 100.0, 1e6, 0.05), 500)
        self.assertEqual(cap_quantity(300, 100.0, 1e6, 0.05), 300)
        self.assertEqual(cap_quantity(300, 100.0, float("nan"), 0.05), 0)
        f = simulate_fill("BUY", 2000, 100.0, 1e6, "2024-01-02", cfg)
        self.assertEqual(f.requested_quantity, 2000)
        self.assertEqual(f.quantity, 500)
        self.assertTrue(f.capped)
        self.assertLessEqual(f.participation, 0.05 + 1e-12)
        self.assertAlmostEqual(f.value_inr, 50_000.0)
        self.assertGreater(f.cost_inr, 0)


class TestUniverse(unittest.TestCase):
    def setUp(self):
        self.data = make_market_data(40, 400, seed=3, listing={"S005": 250}, delist={"S007": 300})
        self.cfg = small_config().universe

    def test_point_in_time_and_excludes_etfs(self):
        panel = compute_universe_panel(self.data, self.cfg, exclude=("GOLDBEES", "SILVERBEES"))
        for as_of in self.data.dates[[130, 200, 262, 301, 350]]:
            members = set(panel.members(as_of))
            single = set(select_universe(self.data.until(as_of), self.cfg, as_of, exclude=("GOLDBEES", "SILVERBEES")))
            self.assertEqual(members, single)
            self.assertNotIn("NIFTYBEES", members)
            self.assertNotIn("GOLDBEES", members)
            self.assertLessEqual(len(members), self.cfg.top_n_liquid)
        # future data does not change past universes
        v = self.data.value.copy()
        v.iloc[260:] *= 0.0
        changed = compute_universe_panel(_replace_frames(self.data, value=v), self.cfg)
        pd.testing.assert_frame_equal(changed.mask.iloc[:260], compute_universe_panel(self.data, self.cfg).mask.iloc[:260])
        # late listing needs min_history; delisted symbol leaves
        self.assertNotIn("S005", panel.members(self.data.dates[300]))
        self.assertFalse(panel.mask["S007"].iloc[301:].any())

    def test_etfs_included_when_not_excluded(self):
        cfg = UniverseConfig(**{**self.cfg.__dict__, "exclude_etfs": False, "top_n_liquid": 100})
        panel = compute_universe_panel(self.data, cfg)
        self.assertIn("NIFTYBEES", panel.members(self.data.dates[200]))


class TestSignals(unittest.TestCase):
    def test_ewmac_scale_invariance(self):
        d = make_market_data(10, 300, seed=5)
        a = ewmac_raw(d.close, 16, 64, 35)
        b = ewmac_raw(d.close * 37.0, 16, 64, 35)
        np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=1e-9, equal_nan=True)
        self.assertTrue(np.isfinite(a.iloc[-1]).all())

    def test_normaliser_uses_past_only(self):
        rng = np.random.default_rng(0)
        raw = pd.DataFrame(rng.standard_normal((100, 8)), index=pd.bdate_range("2020-01-01", periods=100))
        mask = pd.DataFrame(True, index=raw.index, columns=raw.columns)
        s1, warm = pit_forecast_scalar(raw, mask, 10.0, 20)
        bumped = raw.copy()
        bumped.iloc[50:] *= 100.0  # row 50 and later
        s2, _ = pit_forecast_scalar(bumped, mask, 10.0, 20)
        np.testing.assert_allclose(s1.iloc[:51].to_numpy(), s2.iloc[:51].to_numpy(), equal_nan=True)
        self.assertFalse(np.isclose(s1.iloc[52], s2.iloc[52]))
        self.assertTrue(np.isnan(s1.iloc[0]))
        self.assertTrue(warm.iloc[:20].all() and not warm.iloc[21:].any())
        self.assertAlmostEqual(s1.iloc[1], 10.0 / raw.iloc[0].abs().mean())


class TestRegime(unittest.TestCase):
    def test_hysteresis(self):
        raw = ["risk_on", "risk_on", "neutral", "risk_on", "risk_on", "risk_on", "risk_off", "risk_off", "risk_on", "risk_off", "risk_off", "risk_off"]
        out = apply_hysteresis(raw, 3)
        self.assertEqual(out[:5], ["neutral"] * 5)
        self.assertEqual(out[5], "risk_on")
        self.assertEqual(out[6:11], ["risk_on"] * 5)
        self.assertEqual(out[11], "risk_off")
        self.assertEqual(apply_hysteresis(["risk_off"], 1), ["risk_off"])

    def test_states_and_vix_fallback(self):
        d = make_market_data(30, 400, seed=2, drift=0.001)
        cfg = small_config().regime
        mask = pd.DataFrame(True, index=d.dates, columns=d.close.columns)
        ic = d.index_close.copy()
        ic.iloc[300:310, ic.columns.get_loc("INDIAVIX")] = np.nan
        panel = compute_regime(ic, d.close, mask, cfg)
        self.assertTrue(panel.vix_is_fallback.iloc[300:310].all())
        self.assertTrue(np.isfinite(panel.vix.iloc[300:310]).all())
        self.assertIn(RISK_ON, set(panel.state))  # strongly up market
        ic2 = ic.copy()
        ic2.iloc[200:, ic2.columns.get_loc("INDIAVIX")] = 50.0
        p2 = compute_regime(ic2, d.close, mask, cfg)
        self.assertEqual(p2.raw_state.iloc[200], RISK_OFF)
        self.assertNotEqual(p2.state.iloc[200], RISK_OFF)  # needs confirmation
        self.assertEqual(p2.state.iloc[200 + cfg.confirm_days - 1], RISK_OFF)
        # causal
        p3 = compute_regime(ic.iloc[:250], d.close.iloc[:250], mask.iloc[:250], cfg)
        self.assertTrue((p3.state == panel.state.iloc[:250]).all())


class TestPortfolioPieces(unittest.TestCase):
    def test_trailing_stop_never_lowered(self):
        self.assertEqual(trailing_stop(110.0, 5.0, 3.0, None), 95.0)
        self.assertEqual(trailing_stop(100.0, 5.0, 3.0, 95.0), 95.0)
        self.assertEqual(trailing_stop(120.0, 5.0, 3.0, 95.0), 105.0)
        self.assertEqual(trailing_stop(float("nan"), 5.0, 3.0, 95.0), 95.0)

    def test_stop_fill_price(self):
        self.assertEqual(stop_fill_price(90.0, 85.0, 95.0), 90.0)  # gap below stop -> open
        self.assertEqual(stop_fill_price(100.0, 94.0, 95.0), 95.0)  # intraday -> stop
        self.assertIsNone(stop_fill_price(100.0, 96.0, 95.0))

    def test_caps_and_buffer(self):
        raw = pd.Series({"A": 10.0, "B": 1.0, "C": 1.0, "D": 1.0, "E": 1.0, "F": 1.0, "G": 1.0, "H": 1.0})
        w = capped_weights(raw, 0.2)
        self.assertAlmostEqual(w.sum(), 1.0)
        self.assertLessEqual(w.max(), 0.2 + 1e-12)
        sectors = {"A": "X", "B": "X", "C": "X"}
        w2 = capped_weights(pd.Series(1.0, index=list("ABCDEFGHIJ")), 0.5, 0.2, sectors)
        self.assertLessEqual(w2[["A", "B", "C"]].sum(), 0.2 + 1e-9)
        self.assertAlmostEqual(w2.sum(), 1.0)
        out = apply_no_trade_buffer({"A": 0.10, "B": 0.10, "C": 0.001}, {"A": 0.09, "B": 0.05, "D": 0.1}, 1e6, 0.25, 5000)
        self.assertEqual(out["A"], 0.09)
        self.assertEqual(out["B"], 0.10)
        self.assertNotIn("C", out)  # new entry below min trade value
        self.assertNotIn("D", out)  # exit always trades

    def test_allocator(self):
        cfg = AllocatorConfig()
        a = allocate(0.20, 0.15, True, True, 1.0, cfg)
        self.assertAlmostEqual(a.gross, 1.0)
        share = a.core_capital * 0.20 / (a.core_capital * 0.20 + a.sleeve_capital * 0.15)
        self.assertAlmostEqual(share, cfg.core_risk_share)
        off = allocate(0.20, 0.15, True, True, 0.0, cfg)
        self.assertEqual(off.core_capital, 0.0)
        self.assertGreaterEqual(off.sleeve_capital, a.sleeve_capital)
        self.assertLessEqual(off.gross, 1.0)
        no_metals = allocate(0.20, float("nan"), True, False, 0.6, cfg)
        self.assertAlmostEqual(no_metals.core_capital, 0.6)


class TestEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="nse_engine_test_")
        cls.data = make_market_data(60, 700, seed=1, median_value_inr=5e8, drift=0.0003, drift_dispersion=0.0012)
        cls.cfg = small_config(runs_dir=cls.tmp)
        cls.cache = EngineCache(cls.data, cls.cfg)
        cls.result = run_backtest(cls.data, cls.cfg, record=True, tag="unittest", cache=cls.cache)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _held_on(self, pos):
        w = self.result.weights.iloc[pos]
        return [s for s in w.index[w > 0] if s.startswith("S")]

    def test_until_equivalence(self):
        d, cfg = self.data, self.cfg
        for pos in (250, 333, 420, 505, 640):
            as_of = d.dates[pos]
            base = generate_targets(d, cfg, as_of, cache=self.cache)
            held = list(base.core_weights)[:4] + ["S001", "S002"]
            holdings = {
                s: Holding(s, 100, float(d.close[s].iloc[pos - 30]), d.dates[pos - 30],
                           stop_price=float(d.close[s].iloc[pos - 5]) * 0.8)
                for s in dict.fromkeys(held)
            }
            holdings["GOLDBEES"] = Holding("GOLDBEES", 50, 40.0, d.dates[pos - 60])
            stopped = {"S003": d.dates[pos - 2]}
            for eq in (None, 5e6):
                full = generate_targets(d, cfg, as_of, holdings, self.cache, equity=eq, stopped_out=stopped)
                cut = generate_targets(d.until(as_of), cfg, as_of, holdings, equity=eq, stopped_out=stopped)
                self.assertEqual(set(full.weights), set(cut.weights))
                for s in full.weights:
                    self.assertAlmostEqual(full.weights[s], cut.weights[s], delta=1e-9)
                self.assertEqual(set(full.stops), set(cut.stops))
                for s in full.stops:
                    self.assertAlmostEqual(full.stops[s], cut.stops[s], delta=1e-9)
                self.assertEqual(full.exits, cut.exits)
                self.assertEqual(full.ranks, cut.ranks)
                self.assertEqual((full.regime, full.universe_size), (cut.regime, cut.universe_size))
                self.assertLessEqual(full.gross, cfg.allocator.max_gross + 1e-9)
                self.assertNotIn("S003", {s for s in full.weights if s not in holdings})

    def test_rank_and_forecast_exits(self):
        d = self.data
        pos = 500
        cfg = self.cfg.replace(**{"portfolio.exit_rank": 3})
        cache = EngineCache(d, cfg)
        base = generate_targets(d, cfg, d.dates[pos], cache=cache)
        ranked = sorted(base.ranks, key=base.ranks.get)
        low_rank = ranked[5]
        uni_mask = cache.universe_mask[pos]
        neg_uni = [s for s in cache.symbols if s.startswith("S") and uni_mask[cache.sym_index[s]]
                   and np.isfinite(cache.combined[pos, cache.sym_index[s]]) and cache.combined[pos, cache.sym_index[s]] <= 0]
        top = ranked[0]
        holdings = {s: Holding(s, 10, 1.0, d.dates[pos - 10]) for s in [low_rank, top] + neg_uni[:1]}
        tp = generate_targets(d, cfg, d.dates[pos], holdings, cache)
        self.assertEqual(tp.exits.get(low_rank), "rank_exit")
        self.assertNotIn(top, tp.exits)
        self.assertNotIn(low_rank, tp.weights)
        if neg_uni:
            self.assertEqual(tp.exits.get(neg_uni[0]), "forecast_exit")
        # stop hit today -> stop exit
        low_today = float(d.low[top].iloc[pos])
        tp2 = generate_targets(d, cfg, d.dates[pos], {top: Holding(top, 10, 1.0, d.dates[pos - 10], stop_price=low_today + 0.01)}, cache)
        self.assertEqual(tp2.exits.get(top), "stop")

    def test_stops_never_lowered_through_time(self):
        d, cfg = self.data, self.cfg
        pos0 = 400
        tp = generate_targets(d, cfg, d.dates[pos0], cache=self.cache)
        sym = next(iter(tp.core_weights))
        h = Holding(sym, 10, float(d.close[sym].iloc[pos0]), d.dates[pos0 + 1], stop_price=tp.stops[sym])
        prev = h.stop_price
        for pos in range(pos0 + 1, pos0 + 60):
            t = generate_targets(d, cfg, d.dates[pos], {sym: h}, self.cache)
            if sym in t.exits:
                break
            self.assertGreaterEqual(t.stops[sym], prev - 1e-12)
            prev = h.stop_price = t.stops[sym]

    def test_gross_and_accounting(self):
        r = self.result
        self.assertTrue((r.weights.sum(axis=1) <= 1.0 + 1e-9).all())
        self.assertTrue((r.weights >= 0).all().all())
        self.assertGreater(len(r.trades), 0)
        t = r.trades
        self.assertTrue((t.quantity > 0).all())
        self.assertTrue((t.requested_quantity >= t.quantity).all())
        self.assertTrue(np.isfinite(r.equity).all())
        # fills are at the open of a day after the decision (never same close)
        buys = t[(t.side == "BUY")]
        opens = [d_open for d_open in (self.data.open.at[row.date, row.symbol] for row in buys.itertuples())]
        np.testing.assert_allclose(buys.price.to_numpy(), np.asarray(opens))

    def test_gap_down_stop_fills_at_open(self):
        d, cfg, r = self.data, self.cfg, self.result
        w = r.weights
        pos = next(p for p in range(450, 650) if len(self._held_on(p)) and len(self._held_on(p - 1)))
        sym = sorted(set(self._held_on(pos)) & set(self._held_on(pos - 1)))[0]
        u = pos + 1
        frames = {k: getattr(d, k).copy() for k in ("open", "high", "low", "close")}
        col = d.close.columns.get_loc(sym)
        for k in frames:
            frames[k].iloc[u:, col] = frames[k].iloc[u:, col] * 0.5
        crashed = _replace_frames(d, **frames)
        res = run_backtest(crashed, cfg, record=False)
        tr = res.trades[(res.trades.symbol == sym) & (res.trades.date == d.dates[u])]
        self.assertEqual(list(tr.reason), ["stop"])
        self.assertAlmostEqual(float(tr.price.iloc[0]), float(frames["open"].iloc[u, col]))
        # intraday: open above stop, low far below -> fill at the stop level (< open, > low)
        frames2 = {k: getattr(d, k).copy() for k in ("open", "high", "low", "close")}
        frames2["low"].iloc[u, col] = frames2["open"].iloc[u, col] * 0.5
        res2 = run_backtest(_replace_frames(d, **frames2), cfg, record=False)
        tr2 = res2.trades[(res2.trades.symbol == sym) & (res2.trades.date == d.dates[u]) & (res2.trades.reason == "stop")]
        self.assertEqual(len(tr2), 1)
        px = float(tr2.price.iloc[0])
        self.assertLess(px, float(frames2["open"].iloc[u, col]))
        self.assertGreater(px, float(frames2["low"].iloc[u, col]))
        del w

    def test_delisted_liquidated_at_last_close(self):
        d, cfg = self.data, self.cfg
        pos = next(p for p in range(500, 650) if len(self._held_on(p)))
        sym = self._held_on(pos)[0]
        frames = {k: getattr(d, k).copy() for k in ("open", "high", "low", "close", "volume", "value")}
        for f in frames.values():
            f.iloc[pos + 1 :, f.columns.get_loc(sym)] = np.nan
        res = run_backtest(_replace_frames(d, **frames), cfg, record=False)
        tr = res.trades[(res.trades.symbol == sym) & (res.trades.reason == "delisted")]
        self.assertEqual(len(tr), 1)
        self.assertAlmostEqual(float(tr.price.iloc[0]), float(d.close[sym].iloc[pos]))
        self.assertEqual(tr.date.iloc[0], d.dates[pos + 1])
        self.assertTrue((res.weights[sym].iloc[pos + 1 :] == 0).all())

    def test_cash_accrues_yield(self):
        d = make_market_data(20, 150, seed=4, metals=False)
        cfg = small_config(runs_dir=self.tmp, **{"universe.min_median_value_inr": 1e15})
        res = run_backtest(d, cfg, record=False)
        self.assertEqual(len(res.trades), 0)
        expected = cfg.initial_capital * (1 + cfg.cash_yield_annual / 252) ** (len(d.dates) - 1)
        self.assertAlmostEqual(float(res.equity.iloc[-1]), expected, places=4)

    def test_run_directory(self):
        r = self.result
        run_dir = Path(r.run_dir)
        for name in ("config.json", "manifest.json", "returns.csv", "equity.csv", "trades.csv", "weights.parquet"):
            self.assertTrue((run_dir / name).exists(), name)
        man = json.loads((run_dir / "manifest.json").read_text())
        for key in ("run_id", "tag", "config_hash", "git_commit", "git_dirty", "data_hash", "start", "end", "created_at", "metrics", "lag_days"):
            self.assertIn(key, man)
        self.assertEqual(man["run_id"], r.run_id)
        self.assertTrue(r.run_id.endswith(self.cfg.config_hash()[:8]))
        rets = pd.read_csv(run_dir / "returns.csv", parse_dates=["date"])
        self.assertEqual(list(rets.columns), ["date", "return"])
        self.assertEqual(len(rets), len(r.returns))

    def test_lag_does_not_inflate_turnover(self):
        base = self.result.metrics["annual_turnover"]
        for lag in (1, 2):
            res = run_backtest(self.data, self.cfg, record=False, lag_days=lag, cache=self.cache)
            self.assertEqual(res.metrics["lag_days"], float(lag))
            self.assertTrue((res.weights.sum(axis=1) <= 1.0 + 1e-9).all())
            self.assertLess(res.metrics["annual_turnover"], 1.5 * base)
            self.assertGreater(res.metrics["annual_turnover"], base / 1.5)
            # delayed fills happen at the open lag+1 sessions after a decision
            self.assertTrue((res.trades.date >= self.data.dates[lag + 1]).all())

    def test_trending_positive_flat_not(self):
        self.assertGreater(self.result.metrics["total_return"], 0.10)
        self.assertGreater(self.result.metrics["cagr"], 0.0)
        flat = make_market_data(60, 700, seed=1, median_value_inr=5e8)
        fres = run_backtest(flat, self.cfg, record=False)
        self.assertLess(fres.metrics["sharpe"], 0.3)
        self.assertLess(fres.metrics["sharpe"], self.result.metrics["sharpe"])


class TestMetrics(unittest.TestCase):
    def test_basic(self):
        idx = pd.bdate_range("2020-01-01", periods=504)
        rng = np.random.default_rng(0)
        r = pd.Series(0.0004 + 0.01 * rng.standard_normal(504), index=idx)
        eq = 1e6 * (1 + r).cumprod()
        m = compute_metrics(r, eq, None, None, rf_annual=0.065, initial_capital=1e6)
        ex = r - 0.065 / 252
        self.assertAlmostEqual(m["sharpe"], ex.mean() / ex.std() * np.sqrt(252))
        years = (idx[-1] - idx[0]).days / 365.25
        self.assertAlmostEqual(m["cagr"], (eq.iloc[-1] / 1e6) ** (1 / years) - 1)
        self.assertLessEqual(m["max_drawdown"], 0)
        trades = pd.DataFrame({
            "date": [idx[0], idx[5], idx[6], idx[9]], "symbol": ["A", "A", "B", "B"],
            "side": ["BUY", "SELL", "BUY", "SELL"], "quantity": [10, 10, 5, 5],
            "price": [100, 110, 100, 90], "value_inr": [1000.0, 1100.0, 500.0, 450.0], "cost_inr": [1.0, 1.0, 1.0, 1.0],
        })
        m2 = compute_metrics(r, eq, trades, None, 0.0, 1e6)
        self.assertEqual(m2["n_round_trips"], 2)
        self.assertAlmostEqual(m2["hit_rate"], 0.5)
        self.assertAlmostEqual(m2["cost_drag"], 4.0 / eq.mean() / years)


if __name__ == "__main__":
    unittest.main()
