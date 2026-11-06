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
from nse_engine.data.reference import FOR_CAUSE_EXITS, load_for_cause_exits
from nse_engine.data.store import build_year_frames, for_cause_exits, invalid_bars
from nse_engine.engine import EngineCache, run_backtest
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
