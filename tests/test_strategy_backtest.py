"""The legacy single-stock strategy backtest (strategies/, trading_strategies/, tools/strategy_benchmark.py): it
fills as a broker would, on prices that mean something, and its strategies sell (BT1)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

# ── BT1: the single-stock backtest (strategies/, trading_strategies/, tools/strategy_benchmark.py) ──


def _bars(closes, opens=None, start="2024-01-01"):
    closes = np.asarray(closes, dtype="float64")
    opens = closes if opens is None else np.asarray(opens, dtype="float64")
    df = pd.DataFrame({"Open": opens, "High": np.maximum(opens, closes) * 1.01, "Low": np.minimum(opens, closes) * 0.99,
                       "Close": closes, "Volume": 1e6}, index=pd.bdate_range(start, periods=len(closes)))
    df.attrs["market"] = "IND"
    return df


def _plain():
    from strategies.base_strategy import BaseStrategy

    class Plain(BaseStrategy):
        def run(self, *a, **k):
            raise NotImplementedError

    return Plain()


def _risk(share=1.0):
    from strategies.base_strategy import RiskParams

    return RiskParams(max_position_size=share)


def _fees(strategy, value, side, day):
    return strategy._trade_cost(value, side, pd.Timestamp(day), "IND")


def test_a_decision_fills_at_the_next_open_never_its_own_close():
    """Decided on the 2nd close (100), bought at the 3rd open (110), though the close that decided was cheaper."""
    s = _plain()
    df = _bars([100, 100, 120, 120], opens=[100, 100, 110, 120])
    df["positions"] = [0, 1, 1, 1]
    p = s._fill_positions(df, df["positions"], 100_000, _risk())
    assert p["shares"].tolist() == [0, 0, 906, 906]               # 909 at 110 would leave Rs 10 for about Rs 340 of costs
    day = df.index[2]
    assert p["cash"].iloc[2] == pytest.approx(100_000 - 906 * 110 - _fees(s, 906 * 110, "BUY", day))
    assert p["cash"].iloc[2] >= 0
    assert p["total_value"].iloc[-1] == pytest.approx(p["cash"].iloc[-1] + 906 * 120)


def test_a_sale_fills_at_the_next_open_too_and_pays_both_ways():
    s = _plain()
    df = _bars([100, 100, 100, 130, 90], opens=[100, 100, 100, 100, 125])
    df["positions"] = [0, 1, 1, 0, 0]
    p = s._fill_positions(df, df["positions"], 100_000, _risk())
    buy, sell = df.index[2], df.index[4]
    qty = int(p["shares"].max())
    want = 100_000 - qty * 100 - _fees(s, qty * 100, "BUY", buy) + qty * 125 - _fees(s, qty * 125, "SELL", sell)
    assert p["total_value"].iloc[-1] == pytest.approx(want)       # sold at 125, not at the 130 close or the 90 after it
    assert p["cumulative_cost"].iloc[-1] > 0.002 * qty * 100


def test_a_flat_strategy_is_worth_its_capital_and_pays_nothing():
    df = _bars([100, 50, 200, 150])
    df["positions"] = 0
    p = _plain()._fill_positions(df, df["positions"], 100_000, _risk())
    assert p["total_value"].tolist() == [100_000.0] * 4 and p["cumulative_cost"].iloc[-1] == 0


def test_the_share_count_does_not_know_the_future():
    """The old sizing divided by the highest close of the whole sample."""
    calm = _bars([100.0] * 6)
    soar = _bars([100.0, 100, 100, 100, 100, 10_000])
    for df in (calm, soar):
        df["positions"] = [0, 1, 1, 1, 1, 1]
    s = _plain()
    a = s._fill_positions(calm, calm["positions"], 100_000, _risk(0.25))
    b = s._fill_positions(soar, soar["positions"], 100_000, _risk(0.25))
    assert a["shares"].iloc[-1] == b["shares"].iloc[-1] == 250


def test_position_size_is_a_share_of_the_equity_so_gains_compound():
    s = _plain()
    df = _bars([100, 100, 100, 200, 200, 200, 200], opens=[100, 100, 100, 200, 200, 200, 200])
    df["positions"] = [0, 1, 1, 0, 1, 1, 1]
    p = s._fill_positions(df, df["positions"], 100_000, _risk(0.5))
    first, second = int(p["shares"].iloc[2]), int(p["shares"].iloc[-1])
    assert first == 500 and second > first / 2                    # half the bigger equity, not of the first capital
    assert second == int(p["cash"].iloc[4] * 0.5 // 200)


def test_a_missing_open_holds_the_fill_to_the_next_session():
    df = _bars([100, 100, 100, 100, 100], opens=[100, 100, np.nan, np.nan, 104])
    df["positions"] = [0, 1, 1, 1, 1]
    p = _plain()._fill_positions(df, df["positions"], 100_000, _risk())
    assert p["shares"].tolist()[:4] == [0, 0, 0, 0] and p["shares"].iloc[4] == 958     # 104 and costs, in cash


def test_the_last_close_decision_has_nothing_to_fill_at():
    df = _bars([100, 100, 100])
    df["positions"] = [0, 0, 1]
    assert _plain()._fill_positions(df, df["positions"], 100_000, _risk())["shares"].iloc[-1] == 0


def test_a_share_above_the_cap_still_trades_and_never_overdraws():
    df = _bars([900.0, 900, 900])
    df["positions"] = [1, 1, 1]
    p = _plain()._fill_positions(df, df["positions"], 1_000, _risk(0.25))
    assert p["shares"].iloc[-1] == 1 and p["cash"].min() >= 0     # the old rule: one share while the cash covers it
    broke = _plain()._fill_positions(df, df["positions"], 500, _risk(0.25))
    assert broke["shares"].iloc[-1] == 0 and broke["cash"].iloc[-1] == 500


def test_a_long_only_book_never_goes_short():
    """A sell-side decision closes a long; delivery cannot open a short."""
    df = _bars([100.0, 100, 100, 100, 100])
    df["positions"] = [0, 1, -1, -1, -1]
    p = _plain()._calculate_portfolio_long_only(df, 100_000, _risk())
    assert p["shares"].min() == 0 and p["shares"].iloc[2] > 0 and p["shares"].iloc[-1] == 0
    assert (_plain()._calculate_portfolio(df, 100_000, _risk())["shares"].iloc[-1] < 0)     # the default still may


def test_a_portfolio_without_open_prices_is_refused():
    df = _bars([100, 100]).drop(columns=["Open"])
    with pytest.raises(ValueError, match="Open"):
        _plain()._fill_positions(df, pd.Series([0, 1], index=df.index), 100_000, _risk())


def test_us_bars_pay_us_costs():
    s = _plain()
    day = pd.Timestamp("2024-01-02")
    assert s._trade_cost(100_000, "BUY", day, "US") == pytest.approx(100_000 * 5 / 1e4)    # no STT, 5 bp of slippage
    assert s._trade_cost(100_000, "BUY", day, "IND") > 3 * s._trade_cost(100_000, "BUY", day, "US")


@pytest.mark.parametrize("seed", range(4))
def test_the_portfolio_never_reads_a_later_bar_and_keeps_its_books(seed):
    """Seeded random paths and decisions: values to a cut survive any change after it, shares stay whole, cash stays
    positive, and value always equals cash plus holdings."""
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300)))
    df = _bars(c, opens=c * (1 + rng.normal(0, 0.005, 300)))
    df["positions"] = (rng.random(300) > 0.7).astype(int)
    s = _plain()
    a = s._fill_positions(df, df["positions"], 100_000, _risk())
    cut = 200
    later = df.copy()
    k = rng.uniform(0.2, 5.0, 300 - cut - 1)
    for col in ("Open", "High", "Low", "Close"):
        later.iloc[cut + 1:, later.columns.get_loc(col)] = later[col].iloc[cut + 1:].to_numpy() * k
    later.iloc[cut + 1:, later.columns.get_loc("positions")] = rng.integers(0, 2, 300 - cut - 1)
    b = s._fill_positions(later, later["positions"], 100_000, _risk())
    pd.testing.assert_frame_equal(a.iloc[:cut + 1], b.iloc[:cut + 1])
    assert (a["shares"] == a["shares"].round()).all() and (a["cash"] >= 0).all()
    assert (a["total_value"] - a["holdings"] - a["cash"]).abs().max() < 1e-6


def test_metrics_measure_the_strategy_never_the_stock():
    """Swing Combo's equity column was missed, so its metrics were the stock's: a -15.8% run reported -99%."""
    s = _plain()
    df = _bars([1_000.0, 1_000, 1_000])
    book = pd.DataFrame({"close": [1000.0, 1000, 1000], "equity": [100_000.0, 101_000, 102_000]}, index=df.index)
    m = s.calculate_metrics(book, df, 100_000)
    assert m["total_return"] == pytest.approx(2.0) and m["final_value"] == 102_000
    assert "calculation_error" in s.calculate_metrics(book[["close"]], df, 100_000)      # no value column: no number


def test_sharpe_uses_the_markets_risk_free_rate():
    from config import Config

    s = _plain()
    df = _bars([100.0, 100, 100])
    assert s._risk_free_rate(df) == Config.RISK_FREE_RATE_IND
    df.attrs["market"] = "US"
    assert s._risk_free_rate(df) == Config.RISK_FREE_RATE_US
    df.attrs.pop("market")
    assert s._risk_free_rate(df) == 0.02


def test_a_raw_price_jump_is_a_corporate_action():
    from strategies.data_service import raw_price_jump

    assert raw_price_jump(_bars([100.0, 102, 104])) is None
    assert raw_price_jump(_bars([100.0, 100, 100], opens=[100, 120, 100])) is None      # the top of the widest band
    df = _bars([100.0, 100, 50, 51], opens=[100, 100, 50, 51])
    assert raw_price_jump(df) == df.index[2]


@pytest.fixture
def service(monkeypatch):
    from strategies.data_service import DataService

    d = DataService()
    d.clear_cache()
    monkeypatch.setattr("utils.download_ind_ohlcv", lambda t, start, end: _raw[t].copy())
    return d


_raw = {"BONUS.NS": _bars([100.0, 100, 50, 51], opens=[100, 100, 50, 51]), "CALM.NS": _bars([100.0, 101, 102, 103])}


def test_raw_prices_through_a_bonus_give_way_to_adjusted_ones(service, monkeypatch):
    adjusted = _bars([50.0, 50, 50, 51])
    monkeypatch.setattr(service, "_fetch_from_yfinance", lambda *a: adjusted.copy())
    df = service.get_ohlcv("BONUS.NS", "2024-01-01", "2024-01-05")
    assert df["Close"].tolist() == [50.0, 50, 50, 51] and df.attrs["market"] == "IND"


@pytest.mark.parametrize("yahoo", [lambda *a: pd.DataFrame(), lambda *a: _raw["BONUS.NS"].copy()], ids=["empty", "still raw"])
def test_a_bonus_with_no_adjusted_prices_is_refused_never_run(service, monkeypatch, yahoo):
    monkeypatch.setattr(service, "_fetch_from_yfinance", yahoo)
    with pytest.raises(ValueError, match="refused"):
        service.get_ohlcv("BONUS.NS", "2024-01-01", "2024-01-05")


def test_a_yahoo_failure_is_still_a_refusal(service, monkeypatch):
    def down(*a):
        raise ConnectionError("blocked")

    monkeypatch.setattr(service, "_fetch_from_yfinance", down)
    with pytest.raises(ValueError, match="refused"):
        service.get_ohlcv("BONUS.NS", "2024-01-01", "2024-01-05")


def test_bars_are_tagged_with_their_market(service, monkeypatch):
    """The tag picks the costs and the risk-free rate.  Neither frame here carries one of its own."""
    def bare(*a, **k):
        df = _bars([100.0, 101, 102])
        df.attrs.clear()
        return df

    monkeypatch.setattr("utils.download_ind_ohlcv", bare)
    monkeypatch.setattr(service, "_fetch_from_yfinance", bare)
    assert service.get_ohlcv("CALM.NS", "2024-01-01", "2024-01-03").attrs["market"] == "IND"
    assert service.get_ohlcv("AAPL", "2024-01-01", "2024-01-03").attrs["market"] == "US"
    assert service.get_ohlcv("CALM.NS", "2024-01-01", "2024-01-03").copy().attrs["market"] == "IND"    # the cached copy too


def test_quiet_raw_prices_are_used_as_they_are(service, monkeypatch):
    monkeypatch.setattr(service, "_fetch_from_yfinance", lambda *a: pytest.fail("no need for Yahoo"))
    assert service.get_ohlcv("CALM.NS", "2024-01-01", "2024-01-05")["Close"].iloc[-1] == 103


def test_the_avwap_restarts_at_each_anchor_and_starts_at_the_first():
    from strategies.utils import running_avwap

    df = pd.DataFrame({"High": [10.0, 20, 30, 40], "Low": [10.0, 20, 30, 40], "Close": [10.0, 20, 30, 40], "Volume": [1.0, 1, 3, 1]})
    got = running_avwap(df, pd.Series([False, True, False, True]))
    assert np.isnan(got[0]) and got[1:].tolist() == [20.0, (20 + 90) / 4, 40.0]


def test_a_volume_anchor_is_judged_against_the_past_only():
    from strategies.utils import volume_anchors

    v = pd.Series(np.arange(1.0, 301.0))
    a = volume_anchors(v, 95)
    v2 = v.copy()
    v2.iloc[250:] = 1e9                                           # a huge future volume must not lower a past anchor
    assert a.iloc[:250].equals(volume_anchors(v2, 95).iloc[:250]) and not a.iloc[:60].any() and a.iloc[100]


def _strategies():
    pytest.importorskip("matplotlib")
    from tools.strategy_benchmark import STRATEGIES, strategy_class

    return STRATEGIES, strategy_class


def _market(seed, n=600):
    rng = np.random.default_rng(seed)
    c = 500 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, n)))
    o = c * (1 + rng.normal(0, 0.005, n))
    df = pd.DataFrame({"Open": o, "High": np.maximum(o, c) * (1 + rng.uniform(0.001, 0.02, n)),
                       "Low": np.minimum(o, c) * (1 - rng.uniform(0.001, 0.02, n)), "Close": c,
                       "Volume": rng.uniform(1e5, 1e6, n)}, index=pd.bdate_range("2019-01-01", periods=n))
    df.attrs["market"] = "IND"
    return df


def _positions(key, df, tag):
    """A strategy's decisions and portfolio through its own ``run()``."""
    from strategies.data_service import DataService

    from tools.strategy_benchmark import _prime, strategy_class

    logging.disable(logging.CRITICAL)
    try:
        start, end = str(df.index[0].date()), str(df.index[-1].date())
        _prime(f"{tag}_{start}_{end}_1d", df)
        st = strategy_class(key)()
        st._create_charts = lambda *a, **k: []
        res = st.run([tag], start, end, 100_000.0, risk_params={"max_position_size": 1.0})
        assert res.success, res.error_message
        return res.signals["positions"].astype(float), res.portfolio, res.metrics[tag]
    finally:
        logging.disable(logging.NOTSET)
        DataService._cache.pop(f"{tag}_{df.index[0].date()}_{df.index[-1].date()}_1d", None)


SELLERS = ["volume_profile", "anchored_vwap", "order_flow", "liquidity_sweep", "bollinger_pattern"]


ALL_KEYS = ("macd", "awesome_oscillator", "heikin_ashi", "parabolic_sar", "rsi_pattern", "support_resistance",
            "bollinger_pattern", "volume_profile", "anchored_vwap", "order_flow", "liquidity_sweep", "swing_combo",
            "positional_combo")


@pytest.mark.parametrize("key", SELLERS)
def test_the_strategies_that_never_sold_now_sell(key):
    """Each wrote a sell as 0 and then erased it with replace(0, nan).ffill(), so it bought once and held for ever."""
    _strategies()
    exits = sum(int((_positions(key, _market(seed), f"S{seed}")[0].diff() < 0).sum()) for seed in range(6))
    assert exits > 0


@pytest.mark.parametrize("key", ALL_KEYS)
def test_no_strategy_reads_the_future(key):
    """Scramble every bar after a cut: no decision, value or cost up to it may move.  Found look-ahead in Support
    Resistance and Positional Combo; Anchored VWAP and Bollinger's were hidden behind their sell bugs."""
    _strategies()
    for seed in (1, 2):
        base = _market(seed)
        p0, b0, _ = _positions(key, base, f"A{seed}")
        for cut in (350, 500):
            rng = np.random.default_rng(seed * 1000 + cut)
            x = base.copy()
            k = rng.uniform(0.3, 2.5, len(x) - cut - 1)
            for col in ("Open", "High", "Low", "Close"):
                x.iloc[cut + 1:, x.columns.get_loc(col)] = x[col].iloc[cut + 1:].to_numpy() * k
            x.iloc[cut + 1:, x.columns.get_loc("Volume")] = x["Volume"].iloc[cut + 1:].to_numpy() * rng.uniform(0.2, 5, len(x) - cut - 1)
            p1, b1, _ = _positions(key, x, f"B{seed}{cut}")
            assert p0.iloc[:cut + 1].equals(p1.iloc[:cut + 1]), f"{key}: decisions moved with the future"
            assert (b0["total_value"].iloc[:cut + 1] - b1["total_value"].iloc[:cut + 1]).abs().max() < 1e-6


def test_the_combos_report_their_own_return():
    """Their metrics were the stock's: a -15.8% run reported -99%."""
    _strategies()
    for key in ("swing_combo", "positional_combo"):
        pos, book, m = _positions(key, _market(3, 900), "C1")
        assert pos.isin([0, 1]).all()
        assert m["final_value"] == pytest.approx(book["total_value"].iloc[-1])
        assert book["equity"].iloc[-1] == book["total_value"].iloc[-1] and m["costs_included"]


@pytest.mark.parametrize("key, feeds, args", [
    ("swing_combo", ("_sweep_signals", "_order_flow_signals", "_rsi_signals"), (10, 1.5, 20, 14, 14, 30, 70, 2)),
    ("positional_combo", ("_volume_profile_signals", "_anchored_vwap_signals", "_sweep_signals"), (60, 50, 0.7, 95, 0.005, 10, 1.5, 2)),
], ids=["swing", "positional"])
def test_a_combo_holds_from_its_buy_to_its_sell(key, feeds, args):
    """Their position was the day's signal, so a swing trade lasted one day and a sell shorted the next."""
    _strategies()
    from tools.strategy_benchmark import strategy_class

    st = strategy_class(key)()
    df = _market(1, 40)
    vote = pd.Series(0, index=df.index)
    vote.iloc[10], vote.iloc[20] = 1, -1
    for feed in feeds:
        setattr(st, feed, lambda *a, **k: vote)
    s = st._generate_signals(df, *args)
    assert s["positions"].tolist() == [0] * 10 + [1] * 10 + [0] * 20
    assert s["signal"].iloc[20] == -1                              # the charts still see the sell


@pytest.mark.parametrize("key", ALL_KEYS)
def test_a_strategy_keeps_its_markets_costs(key):
    """The market travels on the bars: a strategy that dropped it would pay India's costs on a US stock."""
    _strategies()
    df = _market(4, 500)
    us = df.copy()
    us.attrs["market"] = "US"
    _, india, _ = _positions(key, df, "M1")
    _, america, _ = _positions(key, us, "M2")
    if india["cumulative_cost"].iloc[-1] > 0:
        assert america["cumulative_cost"].iloc[-1] < india["cumulative_cost"].iloc[-1]


@pytest.mark.parametrize("method", ["_is_engulfing", "_is_star"])
def test_a_candle_is_judged_by_its_own_price_not_the_sample(method):
    """The minimum body came from the mean close of the whole history, later prices included."""
    _strategies()
    from tools.strategy_benchmark import strategy_class

    st = strategy_class("support_resistance")()
    df = pd.DataFrame({"Open": [100.0, 99.0, 100.0], "High": [100.5, 100.8, 100.2], "Low": [98.5, 98.9, 98.0], "Close": [99.0, 100.5, 100.1]})
    if method == "_is_star":
        df.loc[1, ["Open", "High", "Low", "Close"]] = [99.0, 104.0, 99.0, 100.0]      # a long upper wick
    now = getattr(st, method)(df, 1)
    soared = pd.concat([df, pd.DataFrame({c: [1e6] * 400 for c in df.columns})], ignore_index=True)
    assert now != 0 and getattr(st, method)(soared, 1) == now      # later prices change nothing


def test_support_resistance_opens_on_a_buy_after_repeated_sells():
    """cumsum().clip(0, 1) let two sells cancel the next buy."""
    _strategies()
    from tools.strategy_benchmark import strategy_class

    st = strategy_class("support_resistance")()
    df = _market(1, 80)
    picks = {"a": 60, "b": 62, "c": 65}
    st._is_engulfing = lambda d, i: {60: 1, 62: 1, 65: 2}.get(i, 0)
    st._is_star = lambda d, i: 0
    st._close_to_resistance = lambda d, i, levels, prox: i in (60, 62)
    st._close_to_support = lambda d, i, levels, prox: i == 65
    pos = st._generate_signals(df, 2, 2, 30, 0.01)["positions"]
    assert pos.iloc[64] == 0 and pos.iloc[65] == 1 and pos.iloc[-1] == 1 and picks


def test_a_pivot_is_used_only_once_its_later_bars_exist():
    _strategies()
    from tools.strategy_benchmark import strategy_class

    st = strategy_class("support_resistance")()
    seen = {}
    df = _market(2, 80)
    st._is_engulfing = lambda d, i: 2
    st._is_star = lambda d, i: 0
    st._close_to_resistance = lambda *a: False
    st._is_support = lambda d, i, n1, n2: i == 50
    st._is_resistance = lambda d, i, n1, n2: False

    def near(d, i, levels, prox):
        seen[i] = list(levels)
        return False

    st._close_to_support = near
    st._generate_signals(df, 2, 3, 30, 0.01)
    assert seen[52] == [] and seen[53] == [df["Low"].iloc[50]]     # known three bars on, not before


def test_the_benchmark_picks_its_names_without_hindsight():
    pytest.importorskip("matplotlib")
    from tools.strategy_benchmark import pick_names

    idx = pd.bdate_range("2012-01-02", periods=200)
    mask = pd.DataFrame(True, index=idx, columns=list("ABC"))
    value = pd.DataFrame({"A": 300.0, "B": 200.0, "C": 100.0}, index=idx)
    start = idx[50]
    assert pick_names(mask, value, start, 2) == ["A", "B"]
    later = value.copy()
    later.loc[start:, "C"] = 1e12                                  # a later winner fills the window, and is not picked
    assert pick_names(mask, later, start, 2) == ["A", "B"]
    with pytest.raises(ValueError):
        pick_names(mask, value, start, 4)


def test_the_benchmark_book_is_equal_sleeves_that_keep_their_own_money():
    pytest.importorskip("matplotlib")
    from tools.strategy_benchmark import book_of

    idx = pd.bdate_range("2013-01-01", periods=3)
    up = pd.DataFrame({"value": [100.0, 150.0, 150.0], "invested": 1.0}, index=idx)
    gone = pd.DataFrame({"value": [100.0, 50.0], "invested": 1.0}, index=idx[:2])    # delisted: its value stays
    ret, invested = book_of({"a": up, "b": gone}, idx[0], idx)
    assert ret.tolist() == pytest.approx([0.0, 0.0, 0.0])          # +50% and -50% on equal money, then nothing
    assert invested.iloc[0] == 1.0 and invested.iloc[-1] == pytest.approx(0.75)


def test_the_benchmark_tags_its_bars():
    pytest.importorskip("matplotlib")
    from types import SimpleNamespace

    from tools.strategy_benchmark import bars_for

    df = _bars([100.0, 101, 102])
    data = SimpleNamespace(**{k.lower(): pd.DataFrame({"X": df[k]}) for k in ("Open", "High", "Low", "Close", "Volume")})
    assert bars_for(data, "X", df.index[0], df.index[-1]).attrs["market"] == "IND"


def test_the_benchmark_will_not_open_the_holdout():
    pytest.importorskip("matplotlib")
    from tools.strategy_benchmark import main

    with pytest.raises(SystemExit, match="holdout"):
        main(["--end", "2026-01-31"])


@pytest.mark.parametrize("included, oos", [(True, 0.10), (False, 0.10 - 10 * 0.004)], ids=["simulated", "not simulated"])
def test_walk_forward_charges_costs_once(included, oos):
    """A portfolio that paid real costs is not charged the flat estimate again; one that did not still is."""
    from types import SimpleNamespace

    from services.research.walk_forward import walk_forward_validate

    class Fake:
        name = "fake"

        def run(self, **k):
            return SimpleNamespace(success=True, metrics={"T": {"sharpe_ratio": 1.0, "total_return": 10.0, "total_trades": 10,
                                                                "costs_included": included}})

    got = walk_forward_validate(Fake, "T", train_days=100, test_days=50, total_days=200, param_grid=[{}])
    assert got.folds and all(f.out_of_sample_return == pytest.approx(oos) for f in got.folds)


def test_a_run_cannot_start_before_its_regime_index(panel):
    """No index, no trend vote: the book would run another rule without a word.  The default store's NIFTY 50
    starts in 2012, so a 2008 run on it lost 9 points more that year than on store_ext2006."""
    from nse_engine.config import EngineConfig
    from nse_engine.engine import EngineCache
    from nse_engine.regime import require_index_from

    cfg = EngineConfig()
    dates = panel.index_close.index
    late = panel.index_close.copy()
    late.iloc[:100, late.columns.get_loc("NIFTY50")] = np.nan
    require_index_from(late, str(dates[100].date()), cfg.regime)        # from its first print: fine
    with pytest.raises(ValueError, match="regime index NIFTY50 starts"):
        require_index_from(late, str(dates[99].date()), cfg.regime)
    with pytest.raises(ValueError, match="regime index NIFTY50 starts"):
        require_index_from(late.assign(NIFTY50=np.nan), str(dates[0].date()), cfg.regime)
    require_index_from(late.drop(columns="NIFTY50"), str(dates[0].date()), cfg.regime)    # the fallback's case

    panel.index_close = late
    with pytest.raises(ValueError, match="regime index NIFTY50 starts"):
        EngineCache(panel, cfg.replace(start=str(dates[50].date()), end=str(dates[-1].date())))


def test_a_stock_moved_to_series_bz_can_be_sold_but_never_bought(monkeypatch):
    """A failing stock leaves EQ and BE: a holding must stay sellable (SB3), and nothing new goes there."""
    from types import SimpleNamespace

    from kite_connect.trading.nse_engine_executor import EngineExecutor
    from kite_connect.trading.nse_instruments import to_broker, to_engine

    ticks = {"DEAD-BZ": 0.01, "HFCL-BE": 0.01, "INFY": 0.05}
    assert to_broker("DEAD", ticks, exit_order=True)[0] == "DEAD-BZ"
    assert to_broker("DEAD", ticks) == (None, "DEAD: not in Kite's NSE equity list (no EQ or BE instrument)")
    assert to_broker("HFCL", ticks)[0] == "HFCL-BE" and to_broker("INFY", ticks, exit_order=True) == ("INFY", "")
    assert to_engine("DEAD-BZ") == "DEAD"

    order = lambda side: SimpleNamespace(symbol="DEAD", side=side, limit_price=4.0, ref_price=4.0, quantity=10,  # noqa: E731
                                         reason="delisted", current_qty=10, target_qty=0)
    plan = SimpleNamespace(sells=[order("SELL")], buys=[order("BUY")], as_of="2026-10-30")
    executor = EngineExecutor.__new__(EngineExecutor)
    executor.kite, executor.next_lower_circuits = object(), lambda symbols: {}
    monkeypatch.setattr("kite_connect.trading.nse_instruments.ticks_for", lambda kite: ticks)
    sell, buy = executor.live_orders(plan, variety="amo")
    assert sell["symbol"] == "DEAD-BZ" and sell["unresolved"] is None and "exit only" in sell["series_note"]
    assert buy["symbol"] == "DEAD" and buy["unresolved"], "a buy is not resolved to a restricted series"
