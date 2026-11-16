"""Survivorship guards (tracker SB2): a backtest never sees history through today's names.

The universe's own look-ahead guard is test_universe_lookahead.py; these cover
the rest: stocks that stop trading stay in the data and can be picked while
they trade, and the sector map (today's NIFTY 500) only acts from the day its
dated snapshot was built.  The store build also refuses bars no traded share
can print (tracker RB1), so a glitch cannot read as a stop hit.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from nse_engine.config import EngineConfig, UniverseConfig
from nse_engine.data.panel import liquid_symbols
from nse_engine.data.reference import build_sector_map, load_sector_history
from nse_engine.data.reference import FOR_CAUSE_EXITS, load_for_cause_exits
from nse_engine.data.store import build_year_frames, for_cause_exits, invalid_bars
from nse_engine.engine import EngineCache, run_backtest
from nse_engine.universe import compute_universe_panel

pytestmark = pytest.mark.lesson(
    "L-SB2", pattern="F5", date="2026-10-08", source="SB2",
    summary="A backtest could see history through today's names: dead stocks, today's sector map")

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


def _prints(closes, start="2020-02-03", symbol="X"):
    days = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"date": days, "symbol": symbol, "close": closes})


def _left(symbol="X", close=100.0, day="2020-01-31"):
    return pd.DataFrame({"date": [pd.Timestamp(day)], "close": [close]}, index=[symbol])


def test_a_for_cause_exit_takes_the_price_a_holder_could_get():
    """Twenty prints later, or the last one; a later gain is not credited; no later print is a normal exit."""
    assert for_cause_exits(_prints([50.0 - i for i in range(30)]), _left()) == {"X": 0.31}
    assert for_cause_exits(_prints([60.0, 40.0]), _left()) == {"X": 0.4}
    assert for_cause_exits(_prints([150.0] * 25), _left()) == {"X": 1.0}
    assert for_cause_exits(_prints([50.0], start="2019-01-01"), _left()) == {}, "a print before it left is no exit"
    assert for_cause_exits(_prints([50.0], symbol="Y"), _left()) == {}


def test_the_recorded_exits_are_sane_and_keep_buyouts_out():
    ratios = load_for_cause_exits()
    assert len(ratios) >= 200 and all(0 < r <= 1 for r in ratios.values())
    assert ratios["BINANIIND"] < 0.2, "a known collapse"
    assert not {"RANBAXY", "HEXAWARE", "PATNI", "BANKRAJAS"} & set(ratios), "a buyout's last close is a fair exit"
    assert '"built_to": "2025-12-31"' in FOR_CAUSE_EXITS.read_text(), "2026 stays sealed"


def test_a_bad_exit_ratio_is_refused(tmp_path):
    path = tmp_path / "exits.json"
    path.write_text('{"ratios": {"X": 1.5}}')
    with pytest.raises(ValueError, match="must lie in"):
        load_for_cause_exits(path)


def test_a_held_stock_that_left_for_cause_is_sold_at_its_exit_ratio(panel, monkeypatch):
    """The backtest fill falls by the ratio, and the US store's own rule is left alone."""
    dates = panel.dates
    cfg = EngineConfig().replace(start=str(dates[300].date()), end=str(dates[-1].date()),
                                 **{"universe.top_n_liquid": 40, "universe.min_history_days": 60,
                                    "universe.liquidity_lookback_days": 40, "universe.min_median_value_inr": 0.0,
                                    "universe.min_price_inr": 0.0})
    held = run_backtest(panel, cfg, record=False).weights.iloc[400]
    name = held[held > 0].index[0]
    gone = _delist(panel, name, after=dates.get_loc(held.name))

    def fills(ratio):
        monkeypatch.setattr("nse_engine.engine.load_for_cause_exits", lambda: {name: ratio})
        trades = run_backtest(gone, cfg, record=False).trades
        return trades[(trades["symbol"] == name) & (trades["reason"] == "delisted")]

    whole, half = fills(1.0), fills(0.5)
    assert len(whole) == len(half) == 1
    assert float(half["price"].iloc[0]) == pytest.approx(float(whole["price"].iloc[0]) * 0.5, rel=1e-3)
    assert EngineCache(gone, cfg).exit_ratio.min() == 0.5
    assert EngineCache(gone, cfg.replace(**{"costs.market": "US"})).exit_ratio.min() == 1.0


def test_a_factor_that_covers_only_part_of_the_move_is_reported():
    import numpy as np
    import pandas as pd

    from nse_engine.data.panel import partial_adjustments

    days = pd.bdate_range("2011-03-01", periods=5)
    close = pd.DataFrame({"HIT": [230.0, 234.0, 46.0, 47.0, 48.0], "CLEAN": [100.0, 101.0, 100.0, 99.0, np.nan],
                          "UP": [50.0, 50.0, 80.0, 81.0, 82.0], "QUIET": [10.0, 10.0, 7.5, 7.0, 7.0]}, index=days)
    events = pd.DataFrame({"date": [days[2]] * 4 + [days[4]] + [pd.Timestamp("2030-01-01")],
                           "symbol": ["HIT", "CLEAN", "UP", "QUIET", "CLEAN", "HIT"],
                           "factor": [0.5, 0.5, 2.0, 0.9, 0.5, 0.5], "source": 1})
    got = partial_adjustments(close, events)
    assert sorted(got["symbol"]) == ["HIT", "UP"]                    # QUIET fell 25%; CLEAN has no price that day
    assert got.set_index("symbol").at["HIT", "move"] == pytest.approx(46.0 / 234.0 - 1.0)
    assert partial_adjustments(close, None).empty



# ── DQ1: every NSE file format, parsed against values read by hand from the real file (6 Nov 2026) ──────────

LEGACY_CM = """SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,TOTALTRADES,ISIN,
RELIANCE,EQ,888.1,896.05,884.3,885.55,884.95,887.9,1675827,1490461120.25,02-JAN-2015,31269,INE002A01018,
SBIN,EQ,314.35,318.3,314.35,315.25,315.6,314,9935094,3147388629.9,02-JAN-2015,79553,INE062A01020,
"""
UDIFF_CM = """TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,FininstrmActlXpryDt,StrkPric,OptnTp,FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,LastPric,PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,ChngInOpnIntrst,TtlTradgVol,TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,Rsvd1,Rsvd2,Rsvd3,Rsvd4
2025-01-02,2025-01-02,CM,NSE,STK,3045,INE062A01020,SBIN,EQ,,,,,STATE BANK OF INDIA,795.00,802.40,787.00,801.20,802.25,793.20,,801.20,,,11398319,9071709128.05,206144,F1,1,,,,,
2025-01-02,2025-01-02,CM,NSE,STK,2885,INE002A01018,RELIANCE,EQ,,,,,RELIANCE INDUSTRIES LTD,1221.25,1244.45,1220.00,1241.80,1240.55,1221.25,,1241.80,,,15486276,19115027208.35,271108,F1,1,,,,,
"""
MTO = """Security Wise Delivery Position - Compulsory Rolling Settlement
10,MTO,02012025,1270292470,0002626
20,1944,RELIANCE,EQ,15486276,9771015,63.09
20,2063,SBIN,EQ,11398319,5725585,50.23
"""
INDEX_2013 = """Index Name,Index Date,Open Index Value,High Index Value,Low Index Value,Closing Index Value,Points Change,Change(%),Volume,Turnover (Rs. Cr.),P/E,P/B,Div Yield
S&P CNX Nifty,01-01-2013,5937.65,5963.9,5935.2,5950.85,45.75,.77,77902745,3298.74,18.82,3.16,1.39
"""
INDEX_2025 = """Index Name,Index Date,Open Index Value,High Index Value,Low Index Value,Closing Index Value,Points Change,Change(%),Volume,Turnover (Rs. Cr.),P/E,P/B,Div Yield
Nifty 50,01-01-2025,23637.65,23822.8,23562.8,23742.9,98.1,.41,154921938,14266.26,21.88,3.54,1.27
"""
CORPACT = """SERIES,SYMBOL,SECURITY,RECORD_DT,BC_STRT_DT,BC_END_DT,EX_DT,ND_STRT_DT,ND_END_DT,PURPOSE
BE,GARFIBRES,Garware TFL,03/01/2025, , ,03/01/2025, , ,BONUS 4:1                
"""
LEGACY_FO = """INSTRUMENT,SYMBOL,EXPIRY_DT,STRIKE_PR,OPTION_TYP,OPEN,HIGH,LOW,CLOSE,SETTLE_PR,CONTRACTS,VAL_INLAKH,OPEN_INT,CHG_IN_OI,TIMESTAMP,
FUTIDX,NIFTY,29-Jan-2015,0,XX,8352.25,8470.9,8352.25,8458.4,8458.4,384440,810571.13,21427925,287375,02-JAN-2015,
OPTIDX,NIFTY,29-Jan-2015,8300,CE,150.3,220.5,150.3,209.2,209.2,168391,357419.22,2362700,-615600,02-JAN-2015,
"""
UDIFF_FO = """TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,FininstrmActlXpryDt,StrkPric,OptnTp,FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,LastPric,PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,ChngInOpnIntrst,TtlTradgVol,TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,Rsvd1,Rsvd2,Rsvd3,Rsvd4
2025-01-02,2025-01-02,FO,NSE,IDF,35006,,NIFTY,,2025-01-30,2025-01-30,,,NIFTY25JANFUT,23911.00,24314.00,23863.10,24282.65,24270.10,23897.85,24188.65,24282.65,12424650,-845800,339021,204499154340.00,145919,F1,25,,,,,
2025-01-02,2025-01-02,FO,NSE,IDO,46343,,NIFTY,,2025-01-30,2025-01-30,24000.00,CE,NIFTY25JAN24000CE,335.05,547.60,304.95,523.00,515.50,327.45,24188.65,523.00,3699375,-462850,231589,141456425822.50,88641,F1,25,,,,,
"""


def _row(df, **match):
    m = df
    for k, v in match.items():
        m = m[m[k] == v]
    assert len(m) == 1, f"expected one row for {match}, got {len(m)}"
    return m.iloc[0]


@pytest.mark.parametrize("parser, text", [("parse_legacy_bhavcopy", LEGACY_CM), ("parse_udiff_bhavcopy", UDIFF_CM)],
                         ids=["legacy", "udiff"])
def test_both_bhavcopy_formats_parse_to_the_printed_numbers(parser, text):
    from nse_engine.data import store

    df = getattr(store, parser)(text)
    r = _row(df, symbol="RELIANCE")
    expect = ({"date": "2015-01-02", "open": 888.1, "high": 896.05, "low": 884.3, "close": 885.55, "prev_close": 887.9,
               "volume": 1675827, "value_inr": 1490461120.25, "trades": 31269}
              if parser == "parse_legacy_bhavcopy" else
              {"date": "2025-01-02", "open": 1221.25, "high": 1244.45, "low": 1220.0, "close": 1241.8,
               "prev_close": 1221.25, "volume": 15486276, "value_inr": 19115027208.35, "trades": 271108})
    assert str(r["date"].date()) == expect.pop("date")
    assert r["series"] == "EQ" and r["isin"] == "INE002A01018"
    for k, v in expect.items():
        assert r[k] == pytest.approx(v), k
    assert not store.invalid_bars(df).any()


def test_delivery_parses_and_agrees_with_the_same_days_bhavcopy():
    from nse_engine.data.store import parse_mto, parse_udiff_bhavcopy

    d = _row(parse_mto(MTO), symbol="RELIANCE")
    assert str(d["date"].date()) == "2025-01-02"                          # 02012025 is day first
    assert (d["traded_qty"], d["deliv_qty"], d["deliv_pct"]) == (15486276, 9771015, 63.09)
    assert d["traded_qty"] == _row(parse_udiff_bhavcopy(UDIFF_CM), symbol="RELIANCE")["volume"]


@pytest.mark.parametrize("text, date, close", [(INDEX_2013, "2013-01-01", 5950.85), (INDEX_2025, "2025-01-01", 23742.9)],
                         ids=["2013-names", "2025-names"])
def test_index_files_map_both_eras_names_to_one_index(text, date, close):
    from nse_engine.data.store import parse_index_close

    r = _row(parse_index_close(text, pd.Timestamp(date)), index_name="NIFTY50")   # 'S&P CNX Nifty' = 'Nifty 50'
    assert r["close"] == pytest.approx(close) and str(r["date"].date()) == date


def test_corporate_action_dates_are_day_first():
    from nse_engine.data.reference import classify_purpose, parse_corporate_actions

    r = _row(parse_corporate_actions(CORPACT, pd.Timestamp("2025-01-02")), symbol="GARFIBRES")
    assert str(pd.Timestamp(r["ex_date"]).date()) == "2025-01-03", "03/01/2025 is 3 January, not 1 March"
    assert classify_purpose(str(r["purpose"]))["factor"] == pytest.approx(0.2), "a 4:1 bonus: 5 shares for 1"


@pytest.mark.parametrize("parser, text, expiry, lot", [("parse_legacy_fo", LEGACY_FO, "2015-01-29", None),
                                                       ("parse_udiff_fo", UDIFF_FO, "2025-01-30", 25)],
                         ids=["legacy", "udiff"])
def test_both_fo_formats_parse_futures_and_options(parser, text, expiry, lot):
    from nse_engine.data import fo_store

    opt, fut = getattr(fo_store, parser)(text)
    f, o = fut.iloc[0], opt.iloc[0]
    assert len(fut) == 1 and len(opt) == 1
    assert str(pd.Timestamp(f["expiry"]).date()) == expiry and str(pd.Timestamp(o["expiry"]).date()) == expiry
    if parser == "parse_legacy_fo":
        assert (f["close"], f["settle"], f["oi"]) == (8458.4, 8458.4, 21427925)
        assert f["value_inr"] == pytest.approx(810571.13e5), "VAL_INLAKH is in lakh"
        assert (o["strike"], o["option_type"], o["close"], o["oi"]) == (8300, "CE", 209.2, 2362700)
    else:
        assert (f["close"], f["settle"], f["oi"]) == (24282.65, 24282.65, 12424650)
        assert (o["strike"], o["option_type"], o["close"], o["oi"]) == (24000, "CE", 523.0, 3699375)
        assert f["lot_size"] == lot


@pytest.mark.lesson("L-DQ1", pattern="F1", date="2026-11-06", source="DQ1",
                    summary="The store build skipped a file it could not parse and lost that session's bars")
def test_a_file_that_will_not_parse_stops_the_build_and_names_itself(tmp_path):
    """DQ1 regression: the build used to log the failure and skip the file, so that session's bars vanished."""
    from nse_engine.data.store import build_year_frames

    good, bad = tmp_path / "cm20150102.csv", tmp_path / "cm20150105.csv"
    good.write_text(LEGACY_CM)
    bad.write_text(LEGACY_CM.replace("TIMESTAMP", "TIME_STAMP"))           # a renamed column: NSE changed the file
    with pytest.raises(ValueError, match="cm20150105"):
        build_year_frames({"equity": [good, bad], "delivery": [], "indices": [], "corpact": []})
    empty = tmp_path / "cm20150106.csv"
    empty.write_text(LEGACY_CM.splitlines()[0] + "\n")                        # a header and no bars
    with pytest.raises(ValueError, match="no equity bars"):
        build_year_frames({"equity": [good, empty], "delivery": [], "indices": [], "corpact": []})
    build_year_frames({"equity": [good], "delivery": [], "indices": [], "corpact": []})   # a clean file still builds


@pytest.mark.lesson("L-CAN1", pattern="F6", date="2026-10-20", source="CAN1",
                    summary="A special session for 14 gold ETFs entered the store and shifted every later rebalance day")
def test_a_session_for_a_few_instruments_stays_out_of_the_store(tmp_path):
    from nse_engine.data import store

    assert "20121111" in store.PARTIAL_SESSIONS                           # Dhanteras 2012, a Sunday
    day = tmp_path / "equity" / "2012"
    day.mkdir(parents=True)
    for name in ("cm20121109.csv.zip", "cm20121111.csv.zip", "cm20121112.csv.zip"):   # the archive keeps all three
        (day / name).write_text("x")
    assert [p.name for p in store.year_sources(tmp_path, 2012)["equity"]] == ["cm20121109.csv.zip", "cm20121112.csv.zip"]


@pytest.mark.lesson("L-LNT32", pattern="F6", date="2026-10-13", source="LN-T32",
                    summary="A rights issue was priced with today's face value, so a later consolidation moved old prices")
def test_a_rights_issue_reads_the_face_value_of_its_own_date():
    from nse_engine.data.reference import FaceValues

    fv = FaceValues(history={"CCAVENUE": [("2025-01-01", 1.0), ("2026-10-13", 10.0)]}, current={"NEW": 2.0})
    assert fv.on("CCAVENUE", "2025-06-26") == 1.0                         # its rights issue: not today's Rs 10
    assert fv.on("CCAVENUE", "2026-10-13") == 10.0 and fv.on("CCAVENUE", "2020-01-01") == 1.0
    assert fv.on("NEW", "2025-06-26") == 2.0 and fv.on("GONE", "2025-06-26") is None


# ── the nightly download (nse_engine/data/archive.py), against NSE's site in memory ──

class _Site:
    """NSE's archive host: ``pages`` {url: (status, bytes) or a list of them, served in turn}; the rest is a 404."""

    def __init__(self, pages=None):
        self.pages, self.asked, self.closed = dict(pages or {}), [], 0

    def get(self, url, timeout=None):
        from types import SimpleNamespace

        self.asked.append(url)
        page = self.pages.get(url, (404, b""))
        if isinstance(page, list):
            page = page.pop(0) if len(page) > 1 else page[0]
        if isinstance(page, Exception):
            raise page
        return SimpleNamespace(status_code=page[0], content=page[1])

    def close(self):
        self.closed += 1


def _zipped(name="data.csv", body=b"a,b\n1,2\n") -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, body)
    return buf.getvalue()


@pytest.fixture
def mirror(tmp_path, monkeypatch):
    """(the archive module, a mirror under tmp_path with no pacing or back-off, its site)."""
    from nse_engine.data import archive

    monkeypatch.setattr(archive.time, "sleep", lambda seconds: None)
    site = _Site()
    made = []
    arch = archive.BhavcopyArchive(tmp_path, requests_per_second=0, max_retries=3,
                                   session_factory=lambda: made.append(1) or site)
    arch.sessions_made = made
    return archive, arch, site


def test_each_days_files_have_one_name_and_the_formats_overlap_only_at_their_boundary(tmp_path):
    from datetime import date

    from nse_engine.data.archive import ARCHIVE_HOST as H, candidate_files

    (legacy,) = candidate_files(tmp_path, "equity", date(2013, 1, 15))
    assert legacy.url == f"{H}/content/historical/EQUITIES/2013/JAN/cm15JAN2013bhav.csv.zip"
    assert legacy.path == tmp_path / "equity" / "2013" / "cm20130115.csv.zip"
    assert [c.fmt for c in candidate_files(tmp_path, "equity", date(2024, 7, 8))] == ["udiff", "legacy"]
    assert [c.fmt for c in candidate_files(tmp_path, "equity", date(2024, 7, 5))] == ["legacy", "udiff"]
    (udiff,) = candidate_files(tmp_path, "equity", date(2026, 11, 13))
    assert udiff.url == f"{H}/content/cm/BhavCopy_NSE_CM_0_0_0_20261113_F_0000.csv.zip" and udiff.path.name == "udiff20261113.csv.zip"
    mto, full = candidate_files(tmp_path, "delivery", date(2026, 11, 13))
    assert (mto.url, full.url) == (f"{H}/archives/equities/mto/MTO_13112026.DAT",
                                   f"{H}/products/content/sec_bhavdata_full_13112026.csv")
    assert candidate_files(tmp_path, "indices", date(2026, 11, 13))[0].path.name == "ind20261113.csv"
    pr = candidate_files(tmp_path, "corpact", date(2026, 11, 13))[0]
    assert pr.url.endswith("/pr/PR131126.zip") and pr.path.name == "bc20261113.csv"
    assert candidate_files(tmp_path, "fo", date(2026, 11, 13))[0].path.name == "udiff_fo20261113.csv.zip"
    assert candidate_files(tmp_path, "participant", date(2026, 11, 13))[0].path.name == "poi20261113.csv"
    with pytest.raises(ValueError, match="unknown kind"):
        candidate_files(tmp_path, "bonds", date(2026, 11, 13))


def test_every_day_is_a_candidate_session_weekends_included():
    from datetime import date

    from nse_engine.data.archive import session_dates

    week = session_dates(date(2026, 1, 30), date(2026, 2, 2))                 # Fri, Sat, Sun (the Budget), Mon
    assert week == [date(2026, 1, 30), date(2026, 1, 31), date(2026, 2, 1), date(2026, 2, 2)]
    weekdays = session_dates(date(2013, 11, 1), date(2013, 11, 4), include_saturdays=False)
    assert weekdays == [date(2013, 11, 1), date(2013, 11, 3), date(2013, 11, 4)]     # the Muhurat Sunday is listed


def test_an_error_page_or_a_broken_zip_is_never_stored_as_data(tmp_path):
    from nse_engine.data.archive import RemoteFile, looks_valid, transform_content

    zip_path, csv_path = tmp_path / "cm.csv.zip", tmp_path / "ind.csv"
    assert looks_valid(_zipped(), zip_path) and looks_valid(b"Index Name,Close\nNifty 50,24000\n", csv_path)
    for bad in (b"", b"<html>Access Denied</html>", b"PK-not-really-a-zip"):
        assert not looks_valid(bad, zip_path), bad
    assert not looks_valid(b"  <!DOCTYPE html><html>resource not found</html>", csv_path)
    pr = RemoteFile("corpact", "pr", "u", tmp_path / "bc.csv")
    assert transform_content(pr, _zipped("Bc131126.csv", b"SERIES,SYMBOL\nEQ,INFY\n")) == b"SERIES,SYMBOL\nEQ,INFY\n"
    assert transform_content(pr, _zipped("Pd131126.csv")).startswith(b"SERIES,SYMBOL,SECURITY,RECORD_DT")   # no actions that day
    assert transform_content(pr, b"<html>denied</html>") is None and transform_content(pr, b"PK-broken") is None
    eq = RemoteFile("equity", "udiff", "u", zip_path)
    assert transform_content(eq, b"<html>denied</html>") is None and transform_content(eq, _zipped()) == _zipped()


def test_a_file_is_fetched_once_and_a_holiday_is_remembered(mirror):
    from datetime import date

    archive, arch, site = mirror
    day, today = date(2026, 11, 12), date(2026, 11, 20)
    (cand,) = archive.candidate_files(arch.root, "equity", day)
    site.pages[cand.url] = (200, _zipped())
    assert arch.sync_one("equity", day, today) == "downloaded" and cand.path.read_bytes() == _zipped()
    assert arch.sync_one("equity", day, today) == "skipped" and len(site.asked) == 1          # already here: no request
    assert not list(cand.path.parent.glob(".*tmp*"))                                           # written whole
    holiday = date(2026, 11, 10)
    assert arch.sync_one("equity", holiday, today) == "holiday" and arch.is_missing("equity", holiday)
    asked = len(site.asked)
    assert arch.sync_one("equity", holiday, today) == "skipped" and len(site.asked) == asked    # one request, once
    arch.save_missing()
    assert archive.BhavcopyArchive(arch.root, session_factory=lambda: site).is_missing("equity", holiday)


def test_a_file_not_out_yet_or_unusable_is_neither_stored_nor_called_a_holiday(mirror):
    from datetime import date

    archive, arch, site = mirror
    today = date(2026, 11, 20)
    assert arch.sync_one("equity", date(2026, 11, 19), today) == "not_published"                # inside the grace days
    assert not arch.is_missing("equity", date(2026, 11, 19))
    day = date(2026, 11, 12)
    (cand,) = archive.candidate_files(arch.root, "equity", day)
    site.pages[cand.url] = (200, b"<html>Access Denied</html>")
    assert arch.sync_one("equity", day, today) == "failed" and not cand.path.exists() and not arch.is_missing("equity", day)
    site.pages[cand.url] = (200, _zipped())                                                     # the equity session exists,
    assert arch.sync_one("equity", day, today) == "downloaded"
    assert arch.sync_one("indices", day, today) == "failed"                                     # so a missing index file is
    assert not arch.is_missing("indices", day)                                                  # late, not a holiday (LN-T19)
    assert arch.sync_one("indices", date(2026, 11, 10), today) == "holiday"


def test_a_refused_request_is_retried_with_a_fresh_session_and_a_dead_site_gives_up(mirror):
    from datetime import date

    import requests

    archive, arch, site = mirror
    day = date(2026, 11, 12)
    (cand,) = archive.candidate_files(arch.root, "equity", day)
    site.pages[cand.url] = [(403, b"denied"), requests.ConnectionError("reset"), (200, _zipped())]
    assert arch.sync_one("equity", day, date(2026, 11, 20)) == "downloaded"
    assert len(site.asked) == 3 and len(arch.sessions_made) == 3 and site.closed == 2
    site.pages["https://x/dead"] = [(503, b"")]
    assert arch.fetch("https://x/dead") == (None, b"") and site.asked.count("https://x/dead") == 3     # max_retries
    site.pages["https://x/odd"] = (400, b"bad request")
    assert arch.fetch("https://x/odd") == (400, b"bad request")


def test_near_the_format_change_the_other_format_is_tried(mirror):
    from datetime import date

    archive, arch, site = mirror
    day = date(2024, 7, 10)
    udiff, legacy = archive.candidate_files(arch.root, "equity", day)
    site.pages[legacy.url] = (200, _zipped())
    assert arch.sync_one("equity", day, date(2026, 11, 20)) == "downloaded"
    assert legacy.path.exists() and not udiff.path.exists() and site.asked == [udiff.url, legacy.url]
    assert arch.sync_one("equity", day, date(2026, 11, 20)) == "skipped"


def test_a_day_with_no_equity_session_asks_for_nothing_else(mirror, monkeypatch):
    from datetime import date

    from nse_engine import nse_calendar

    archive, arch, site = mirror
    monkeypatch.setattr(nse_calendar, "special_session_dates", lambda: [])
    monkeypatch.setattr(archive, "RECENT_GRACE", archive.timedelta(days=-10_000))        # every 404 is old enough
    traded, holiday = date(2026, 11, 9), date(2026, 11, 10)
    for kind in archive.KINDS:
        cand = archive.candidate_files(arch.root, kind, traded)[0]
        site.pages[cand.url] = (200, _zipped("Bc091126.csv") if kind in ("equity", "corpact") else b"a,b\n1,2\n")
    counts = arch.sync(traded, holiday)
    assert counts == {kind: {"downloaded": 1, "holiday": 1} for kind in archive.KINDS}
    assert all("10112026" not in url and "101126" not in url for url in site.asked if "BhavCopy_NSE_CM" not in url)
    assert (arch.root / "manifest" / "missing.json").exists() and arch.is_missing("delivery", holiday)
    assert arch.sync(traded, holiday) == {kind: {"skipped": 2} for kind in archive.KINDS}


def test_requests_are_spaced_by_the_rate_limit():
    from nse_engine.data.archive import RateLimiter

    now, slept = [100.0], []

    def sleep(seconds):
        slept.append(round(seconds, 3))
        now[0] += seconds

    limiter = RateLimiter(2.0, clock=lambda: now[0], sleep=sleep)                # at most two a second
    limiter.wait()
    limiter.wait()
    now[0] += 0.2
    limiter.wait()
    now[0] += 5.0
    limiter.wait()
    assert slept == [0.5, 0.3]
    free = RateLimiter(0, clock=lambda: now[0], sleep=sleep)
    free.wait(), free.wait()
    assert slept == [0.5, 0.3]


# ── the data catalogue and the store doctor (tools/data_catalogue.py, PL-10, PL-11) ──

_NOV = ["2026-10-26", "2026-10-27", "2026-10-28", "2026-10-29", "2026-10-30", "2026-11-02", "2026-11-03",
        "2026-11-04", "2026-11-05", "2026-11-06", "2026-11-08", "2026-11-09", "2026-11-11", "2026-11-12",
        "2026-11-13"]                                     # NSE's real sessions: 8 Nov Muhurat (Sun), 10 Nov a holiday


def _catalogue_store(root, sessions, rows=None, suspects=(), holidays=None, refreshed=("equity_store",)):
    """A live equity store under ``root``: its calendar, one row per symbol and session, a manifest; the
    download's holiday record; the refresh status."""
    import json

    data = root / "data" / "nse_engine"
    store = data / "store"
    (store / "equity").mkdir(parents=True, exist_ok=True)
    days = pd.to_datetime(sorted(sessions))
    pd.DataFrame({"date": days}).to_parquet(store / "calendar.parquet")
    for year in sorted({d.year for d in days}):
        frames = [pd.DataFrame({"date": d, "symbol": [f"S{i}" for i in range((rows or {}).get(d.date().isoformat(), 50))]})
                  for d in days if d.year == year]
        pd.concat(frames).to_parquet(store / "equity" / f"{year}.parquet")
    years = {str(y): {"equity_rows": 50, "fingerprint": f"fp{y}"} for y in sorted({d.year for d in days})}
    (store / "manifest.json").write_text(json.dumps({"years": years, "suspect_missing_sessions": list(suspects)}))
    if holidays is not None:
        (data / "archive" / "manifest").mkdir(parents=True, exist_ok=True)
        (data / "archive" / "manifest" / "missing.json").write_text(json.dumps({"equity": holidays}))
    (data / "refresh_status.json").write_text(json.dumps({"finished_at": "2026-11-13T19:36:35+0530", "ok": True,
                                                         "stores": list(refreshed)}))
    return root


def _monday_evening():
    from tools.data_catalogue import IST

    return datetime(2026, 11, 16, 7, 0, tzinfo=IST)     # before Monday's close: the last completed session is the 13th


def test_a_clean_store_is_catalogued_and_its_first_baseline_is_todays_state(tmp_path):
    from tools import data_catalogue as dc

    root = _catalogue_store(tmp_path, _NOV, holidays=[])
    cat = dc.build(root, _monday_evening(), previous={})
    (equity,) = cat["stores"]
    assert (equity["name"], equity["first"], equity["last"], equity["dates"]) == ("equity", "2026-10-26", "2026-11-13", 15)
    assert equity["refreshed"] and equity["behind"] == 0 and not equity["stale"] and len(equity["build_fingerprint"]) == 16
    doc = cat["doctor"]
    assert (doc["calendar_years"], doc["extra"], doc["missing"], doc["unexplained_gaps"]) == ([2026], [], [], [])
    assert doc["weekend_unlisted"] == [] and cat["findings"] == []               # 8 Nov is a listed special session
    assert cat["last_completed_session"] == "2026-11-13"


@pytest.mark.lesson("L-CAN1")
def test_the_doctor_names_a_session_nse_did_not_hold_and_a_trading_day_the_store_lacks(tmp_path):
    from tools import data_catalogue as dc

    sessions = [d for d in _NOV if d != "2026-11-12"] + ["2026-11-10", "2026-11-07"]    # a holiday and a Saturday in
    cat = dc.build(_catalogue_store(tmp_path, sessions, holidays=[]), _monday_evening(),
                   previous={"baseline": {"weekend_sessions": {}, "short_sessions": {}}})
    assert cat["findings"] == ["2026-11-07: a session on a day NSE did not trade",
                               "2026-11-10: a session on a day NSE did not trade",
                               "2026-11-12: an NSE trading day missing from the store",
                               "2026-11-07: a weekend session not listed in nse_calendar.SPECIAL_SESSIONS (CAN1)"]


def test_a_new_short_or_weekend_session_is_reported_until_acknowledged_with_a_reason(tmp_path):
    from tools import data_catalogue as dc

    root = _catalogue_store(tmp_path, _NOV, rows={"2026-11-12": 20}, holidays=[])
    cat = dc.build(root, _monday_evening(), previous={"baseline": {"weekend_sessions": {}, "short_sessions": {}}})
    assert cat["findings"] == ["2026-11-12: a short session, under 70% of the rows of the 20 before"]
    dc.write(cat, root)
    with pytest.raises(ValueError, match="give the reason"):
        dc.acknowledge("2026-11-12", "  ", root)
    with pytest.raises(ValueError, match="not a weekend or short session"):
        dc.acknowledge("2026-11-13", "a normal day", root)
    dc.acknowledge("2026-11-12", "a half-day the exchange announced", root)
    again = dc.build(root, _monday_evening())                                    # the saved baseline carries over
    assert again["findings"] == [] and again["baseline"]["short_sessions"] == {
        "2026-11-12": "a half-day the exchange announced"}
    first = dc.build(_catalogue_store(tmp_path / "fresh", _NOV, rows={"2026-11-12": 20}, holidays=[]),
                     _monday_evening(), previous={})
    assert first["findings"] == [] and "2026-11-12" in first["baseline"]["short_sessions"]   # a first catalogue


def test_a_weekday_gap_without_a_recorded_holiday_is_named_and_needs_the_download_record(tmp_path):
    from tools import data_catalogue as dc

    sessions = ["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-11",
                "2024-01-12"]
    cat = dc.build(_catalogue_store(tmp_path, sessions, holidays=["2024-01-10"]), _monday_evening(), previous={})
    assert cat["doctor"]["calendar_years"] == [] and cat["doctor"]["unexplained_gaps"] == ["2024-01-03"]
    assert "2024-01-03: a weekday missing that the download did not record as a holiday" in cat["findings"]
    blind = dc.build(_catalogue_store(tmp_path / "no_record", sessions), _monday_evening(), previous={})
    assert blind["doctor"]["unexplained_gaps"] is None                           # not checked, never "none"


def test_a_refreshed_store_that_falls_behind_is_a_finding_and_a_snapshot_is_not(tmp_path):
    from tools import data_catalogue as dc
    from tools.data_catalogue import IST

    root = _catalogue_store(tmp_path, _NOV, holidays=[], refreshed=("equity_store",))
    tuesday = datetime(2026, 11, 18, 7, 0, tzinfo=IST)                            # the 16th and 17th have closed
    cat = dc.build(root, tuesday, previous={})
    assert cat["stores"][0]["behind"] == 2 and "equity store: 2 session(s) behind 17 Nov" in cat["findings"]
    snapshot = dc.build(_catalogue_store(tmp_path / "snap", _NOV, holidays=[], refreshed=()), tuesday, previous={})
    assert snapshot["stores"][0]["behind"] == 2 and snapshot["findings"] == []
    gone = dc.build(tmp_path / "empty", tuesday, previous={})
    assert gone["findings"] == ["the live equity store is not on this machine"] and gone["stores"] == []


def test_the_catalogue_status_line_says_when_it_is_missing_old_or_has_findings(tmp_path):
    import json
    from datetime import date

    from tools import data_catalogue as dc
    from tools import guards

    assert dc.status(tmp_path).startswith("! no catalogue on this machine")
    root = _catalogue_store(tmp_path, _NOV, holidays=[])
    dc.write(dc.build(root, _monday_evening(), previous={}), root)
    assert dc.status(root, date(2026, 11, 16)) == "catalogue of 16 Nov: 0 finding(s); equity to 13 Nov"
    assert "over 8 days old" in dc.status(root, date(2026, 11, 25))
    cat = json.loads((root / dc.CATALOGUE).read_text())
    cat["findings"] = ["2026-11-12: an NSE trading day missing from the store"]
    dc.write(cat, root)
    assert dc.status(root, date(2026, 11, 16)).startswith("! catalogue of 16 Nov: 1 finding(s)")
    record = root / guards.RECORD
    record.write_text(json.dumps({"when": "2026-11-16T09:00+05:30", "passed": 9, "failed": 0, "skipped": 0,
                                  "data_findings": 1}))
    line = guards.status(root, date(2026, 11, 16))
    assert line.startswith("! last run Mon 16 Nov 2026: 9 passed, 0 failed, 0 skipped; store doctor: 1 finding(s)")
