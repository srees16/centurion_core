"""Survivorship guards (tracker SB2): a backtest never sees history through today's names.

The universe's own look-ahead guard is test_universe_lookahead.py; these cover
the rest: stocks that stop trading stay in the data and can be picked while
they trade, and the sector map (today's NIFTY 500) only acts from the day its
dated snapshot was built.  The store build also refuses bars no traded share
can print (tracker RB1), so a glitch cannot read as a stop hit.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from nse_engine.config import EngineConfig, UniverseConfig
from nse_engine.data.panel import liquid_symbols
from nse_engine.data.reference import build_sector_map, load_sector_history
from nse_engine.data.store import build_year_frames, invalid_bars
from nse_engine.engine import EngineCache
from nse_engine.universe import compute_universe_panel

UNIVERSE = UniverseConfig(top_n_liquid=40, min_history_days=60, liquidity_lookback_days=40,
                          refresh_every_n_days=21, min_median_value_inr=0.0, min_price_inr=0.0)


def _delist(data, symbol: str, after: int):
    """The same panel with ``symbol`` trading for the last time on row ``after``."""
    frames = {name: getattr(data, name).copy() for name in ("open", "high", "low", "close", "volume", "value")}
    for f in frames.values():
        f.iloc[after + 1:, f.columns.get_loc(symbol)] = np.nan
    return dataclasses.replace(data, **frames)


def test_a_stock_that_stops_trading_is_picked_while_it_trades(panel):
    """Delisted names are not dropped from history: the universe holds them until their last day."""
    data = _delist(panel, "SYM05", after=600)
    member = compute_universe_panel(data, UNIVERSE).mask["SYM05"]
    assert member.iloc[:601].any(), "a stock that later stops trading must be selectable before it does"
    assert not member.iloc[601:].any(), "and never after its last trade"


def test_the_load_filter_keeps_stocks_that_stopped_trading():
    """The load-time filter judges a stock by its liquid past, so delisting cannot remove it."""
    dates = pd.bdate_range("2020-01-01", periods=300)
    value = pd.DataFrame({"LIQUID_THEN_DELISTED": 5e7, "NEVER_LIQUID": 1e5}, index=dates)
    value.iloc[150:, 0] = np.nan                     # stops trading half-way
    kept = liquid_symbols(value, min_median_value_inr=2.5e6)
    assert kept == ["LIQUID_THEN_DELISTED"]


def test_the_sector_cap_uses_only_snapshots_dated_on_or_before_the_decision(panel):
    first, second = panel.dates[300], panel.dates[600]
    history = [(first, {"SYM01": "Banks"}), (second, {"SYM01": "Banks", "SYM02": "IT"})]
    cache = EngineCache(dataclasses.replace(panel, sector_history=history), EngineConfig())
    assert cache.sectors_on(299) == {}, "before the first snapshot nothing is capped by sector"
    assert cache.sectors_on(300) == {"SYM01": "Banks"}
    assert cache.sectors_on(599) == {"SYM01": "Banks"}
    assert cache.sectors_on(len(panel.dates) - 1) == {"SYM01": "Banks", "SYM02": "IT"}


def test_an_in_memory_map_without_snapshots_holds_for_every_day(panel):
    cache = EngineCache(panel, EngineConfig())
    assert cache.sectors_on(0) == panel.sectors


def test_building_the_sector_map_keeps_one_dated_snapshot_per_change(tmp_path):
    archive = tmp_path / "archive"
    (archive / "reference").mkdir(parents=True)
    nifty500 = archive / "reference" / "ind_nifty500list.csv"
    nifty500.write_text("Company Name,Industry,Symbol,Series,ISIN Code\nA Ltd,Banks,AAA,EQ,INE000000001\n")
    out = tmp_path / "data" / "nse_sector_map.json"
    build_sector_map(archive, out)
    build_sector_map(archive, out)                    # unchanged list: no second snapshot
    history = load_sector_history(out.parent / "nse_sector_maps")
    assert len(history) == 1 and history[0][1] == {"AAA": "Banks"}
    assert history[0][0] == pd.Timestamp.today().normalize()


_BHAV = ("SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,TOTALTRADES,ISIN\n"
         "{rows}")


def _bhav_row(symbol, o, h, lo, c, volume=1000, series="EQ"):
    return f"{symbol},{series},{o},{h},{lo},{c},{c},{c},{volume},{volume * 100},02-JAN-2024,10,INE{symbol[:3]}01011\n"


def test_the_store_refuses_bars_no_traded_share_can_print(tmp_path):
    """RB1: a zero low would read as a stop hit and a zero close as a -100% day, so the build drops such rows,
    keeps a valid duplicate of the same (date, symbol, series) and counts what it refused."""
    rows = (_bhav_row("GOOD", 100, 105, 99, 102)
            + _bhav_row("ZEROLOW", 100, 105, 0, 102)
            + _bhav_row("ZEROCLOSE", 100, 105, 99, 0)
            + _bhav_row("HILO", 100, 99, 105, 102)            # high below the low
            + _bhav_row("NEGVOL", 100, 105, 99, 102, volume=-5)
            + _bhav_row("TWIN", 100, 105, -1, 102)            # refused first, so its valid twin below is kept
            + _bhav_row("TWIN", 100, 105, 99, 103)
            + _bhav_row("NOOHL", "-", "-", "-", 50))           # a missing open/high/low is allowed
    path = tmp_path / "cm02JAN2024bhav.csv"
    path.write_text(_BHAV.format(rows=rows))
    equity, _, _, stats = build_year_frames({"equity": [path]})
    assert sorted(equity["symbol"]) == ["GOOD", "NOOHL", "TWIN"]
    assert float(equity.loc[equity["symbol"] == "TWIN", "close"].iloc[0]) == 103.0
    assert stats["invalid_equity_rows"] == 5


def test_a_clean_build_reports_nothing_refused(tmp_path):
    path = tmp_path / "cm02JAN2024bhav.csv"
    path.write_text(_BHAV.format(rows=_bhav_row("GOOD", 100, 105, 99, 102)))
    _, _, _, stats = build_year_frames({"equity": [path]})
    assert "invalid_equity_rows" not in stats, "a clean year's manifest entry must not change"


@pytest.mark.parametrize("bar, refused", [
    (dict(open=100.0, high=105.0, low=99.0, close=102.0, volume=10.0), False),
    (dict(open=0.05, high=0.05, low=0.05, close=0.05, volume=0.0), False),      # one tick, no trades: valid
    (dict(open=np.nan, high=np.nan, low=np.nan, close=50.0, volume=np.nan), False),
    (dict(open=100.0, high=105.0, low=99.0, close=106.0, volume=10.0), False),  # closing auction may print outside
    (dict(open=100.0, high=105.0, low=99.0, close=np.nan, volume=10.0), True),
    (dict(open=100.0, high=105.0, low=99.0, close=np.inf, volume=10.0), True),
    (dict(open=100.0, high=np.inf, low=99.0, close=102.0, volume=10.0), True),
    (dict(open=-1.0, high=105.0, low=99.0, close=102.0, volume=10.0), True),
    (dict(open=100.0, high=105.0, low=99.0, close=-102.0, volume=10.0), True),
])
def test_invalid_bars_known_answers(bar, refused):
    assert bool(invalid_bars(pd.DataFrame([bar])).iloc[0]) is refused


def test_invalid_bars_passes_every_consistent_bar_and_refuses_any_one_broken_field():
    """Seeded sweep at the store's float32: Rs 0.01 to 1.5 lakh, ranges up to 40%, opens and closes anywhere in
    them.  Each price set to 0, negated or infinite, a NaN close, the high and low swapped or a negative volume
    is refused every time; an empty frame refuses nothing."""
    rng = np.random.default_rng(3)
    n = 20_000
    low = np.exp(rng.uniform(np.log(0.01), np.log(150_000), n))
    high = low * (1 + rng.uniform(0.001, 0.4, n))
    bars = pd.DataFrame({"open": low + (high - low) * rng.uniform(0, 1, n), "high": high, "low": low,
                         "close": low + (high - low) * rng.uniform(0, 1, n),
                         "volume": np.floor(rng.uniform(0, 1e7, n))}).astype("float32")
    assert not invalid_bars(bars).any()
    assert invalid_bars(bars.iloc[:0]).empty
    for col in ("open", "high", "low", "close"):
        for broken in (0.0, -bars[col], np.inf):
            assert invalid_bars(bars.assign(**{col: broken})).all(), (col, broken if np.isscalar(broken) else "negated")
    assert invalid_bars(bars.assign(close=np.nan)).all()
    assert invalid_bars(bars.assign(high=bars["low"], low=bars["high"])).all()
    assert invalid_bars(bars.assign(volume=-1.0)).all()


@pytest.mark.slow
def test_the_store_has_no_bar_the_build_would_refuse(store_path):
    """RB1 changed no stored number: 8.36M EQ/BE rows (2006-2026) pass, so no year needs a rebuild."""
    import glob

    for p in sorted(glob.glob(str(store_path / "equity" / "*.parquet"))):
        df = pd.read_parquet(p, columns=["date", "symbol", "open", "high", "low", "close", "volume"])
        assert not invalid_bars(df).any(), f"{p}: {df[invalid_bars(df)].head(3).to_dict('records')}"


@pytest.mark.slow
def test_the_store_holds_stocks_that_stopped_trading(store_path):
    """If delisted names ever vanish from the store, every backtest becomes survivor-only."""
    spans = pd.read_parquet(store_path / "spans.parquet")
    last = spans.groupby("symbol")["last"].max()
    stopped = (pd.to_datetime(last) < pd.to_datetime(last).max() - pd.Timedelta(days=60)).sum()
    assert stopped >= 100, f"only {stopped} symbols stop trading before the store's end"


@pytest.mark.slow
def test_a_backtest_of_the_past_runs_without_todays_sector_map(store_path):
    """The deployed config's 2013-2025 window predates every snapshot: no sector cap, as on Kaggle."""
    from nse_engine.deployment import load_deployment
    from runners.run_nse_engine import _load_data

    cfg = load_deployment("config/nse_engine_deployed.json").engine.replace(start="2025-01-01", end="2025-12-31")
    data = _load_data(cfg, data_start="2024-01-01")
    assert data.sector_history, "the repo's dated snapshots must load"
    cache = EngineCache(data, cfg)
    assert cache.sectors_on(len(data.dates) - 1) == {}
