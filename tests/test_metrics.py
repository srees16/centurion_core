"""Metric formulas: annualisation, drawdown, the deflated Sharpe's trial penalty, and how often the daily
email's expected move holds (RB3), no gate statistic from an infinite return (RB4), and the scorecard's crash
table and metric conventions against Sun's thesis (TH1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nse_engine.books import MOVE_DAYS, MOVE_FELL, MOVE_INSIDE, expected_move_line, move_coverage
from nse_engine.metrics import compute_metrics, max_drawdown
from nse_engine.validation.dsr import deflated_sharpe, excess_sharpe, performance_summary


def _series(daily: float, n: int = 1260) -> pd.Series:
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(daily, index=idx)


def test_sharpe_of_a_constant_series_is_not_finite_nonsense():
    r = _series(0.0004)
    assert not np.isfinite(excess_sharpe(r, 0.0)) or excess_sharpe(r, 0.0) == 0.0


def test_excess_sharpe_matches_the_definition():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0005, 0.01, 2520), index=pd.bdate_range("2015-01-01", periods=2520))
    rf = 0.065
    rf_daily = rf / 252            # arithmetic, matching the engine's cash accrual
    expected = (r - rf_daily).mean() / (r - rf_daily).std(ddof=1) * np.sqrt(252)
    assert abs(excess_sharpe(r, rf) - expected) < 1e-9


def test_cagr_compounds_over_calendar_years_everywhere():
    """One convention (B2): validation's summary and the engine's metrics agree,
    and both use elapsed calendar time, not sessions / 252."""
    rng = np.random.default_rng(3)
    # 248 sessions a year, as on the NSE: three calendar years of business days minus holidays
    idx = pd.bdate_range("2021-01-01", "2023-12-31")
    idx = idx[rng.random(len(idx)) > 0.05]
    r = pd.Series(rng.normal(0.0006, 0.011, len(idx)), index=idx)
    equity = 500_000 * (1 + r).cumprod()
    p = performance_summary(r, 0.065)
    m = compute_metrics(r, equity, rf_annual=0.065, initial_capital=500_000.0)
    assert p["cagr_convention"] == "calendar"
    assert abs(p["cagr"] - m["cagr"]) < 1e-12
    years = (idx[-1] - idx[0]).days / 365.25
    assert abs(p["cagr"] - (float(equity.iloc[-1]) / 500_000) ** (1 / years) + 1) < 1e-12
    sessions_cagr = (float(equity.iloc[-1]) / 500_000) ** (252 / len(r)) - 1
    assert p["cagr"] != sessions_cagr, "fewer than 252 sessions a year must give a different (lower) figure"


def test_cagr_without_dates_falls_back_to_sessions_and_says_so():
    r = pd.Series([0.001] * 252)          # a RangeIndex: no calendar to measure
    p = performance_summary(r, 0.0)
    assert p["cagr_convention"] == "sessions"
    assert abs(p["cagr"] - ((1.001 ** 252) - 1)) < 1e-9


def test_max_drawdown_is_measured_from_the_peak():
    equity = pd.Series([100, 120, 90, 150], index=pd.bdate_range("2024-01-01", periods=4))
    assert abs(max_drawdown(equity) - (-0.25)) < 1e-12


def test_metrics_are_self_consistent():
    rng = np.random.default_rng(5)
    r = pd.Series(rng.normal(0.0006, 0.012, 756), index=pd.bdate_range("2021-01-01", periods=756))
    equity = 500_000 * (1 + r).cumprod()
    m = compute_metrics(r, equity, rf_annual=0.065, initial_capital=500_000.0)
    assert abs(m["total_return"] - (float(equity.iloc[-1]) / 500_000 - 1)) < 1e-9
    assert m["ann_vol"] > 0 and np.isfinite(m["sharpe"])
    assert -1 < m["max_drawdown"] <= 0
    assert abs(m["calmar"] - m["cagr"] / abs(m["max_drawdown"])) < 1e-9


def test_deflated_sharpe_penalises_more_trials():
    rng = np.random.default_rng(11)
    r = pd.Series(rng.normal(0.0007, 0.011, 2520), index=pd.bdate_range("2015-01-01", periods=2520))
    few = deflated_sharpe(r, n_trials=5, rf_annual=0.0, sr_variance=1e-4)
    many = deflated_sharpe(r, n_trials=5000, rf_annual=0.0, sr_variance=1e-4)
    assert many["sr0_annual"] > few["sr0_annual"], "the hurdle must rise with the number of trials"
    assert many["dsr"] <= few["dsr"]


# ── RB3: the expected move's stated coverage ─────────────────────────────

_DAYS = pd.date_range("2024-01-01", periods=200, freq="D")          # calendar days: the close 30 days on is exact


def _nifty(month_return: float) -> pd.Series:
    """A NIFTY whose every 30-day move is ``month_return``."""
    return pd.Series(20000.0 * (1 + month_return) ** (np.arange(len(_DAYS)) / MOVE_DAYS), index=_DAYS)


def test_move_coverage_known_answers():
    vix = pd.Series(20.0, index=_DAYS)                                 # one SD = 0.20 x sqrt(30/365) = 5.73%
    assert move_coverage(_nifty(0.03), vix) == {"inside": 1.0, "fell": 0.0, "rose": 0.0, "sessions": 170}
    assert move_coverage(_nifty(0.10), vix)["rose"] == 1.0
    assert move_coverage(_nifty(-0.10), vix)["fell"] == 1.0
    edge = 0.20 * np.sqrt(30 / 365)
    assert move_coverage(_nifty(edge * 0.999), vix)["inside"] == 1.0  # on the line counts as inside
    assert move_coverage(_nifty(edge * 1.001), vix)["rose"] == 1.0


def test_move_coverage_refuses_bad_data_and_never_fills():
    vix = pd.Series(20.0, index=_DAYS)
    with pytest.raises(ValueError):
        move_coverage(_nifty(0.03).iloc[:20], vix)                      # no close 30 days later
    with pytest.raises(ValueError):
        move_coverage(_nifty(0.03), vix.where(vix.index != _DAYS[5], 0.0))
    with pytest.raises(ValueError):
        move_coverage(_nifty(0.03).iloc[:0], vix)
    gappy = vix.copy()
    gappy.iloc[10:40] = np.nan                                          # missing VIX sessions are left out
    assert move_coverage(_nifty(0.03), gappy)["sessions"] == 170 - 30


def test_move_coverage_properties():
    """Seeded: the shares add to one, NIFTY's level does not matter, and a wider move never covers less."""
    rng = np.random.default_rng(13)
    for _ in range(50):
        nifty = pd.Series(np.exp(np.cumsum(rng.normal(0.0004, 0.012, len(_DAYS)))) * 18000, index=_DAYS)
        vix = pd.Series(rng.uniform(9, 45, len(_DAYS)), index=_DAYS)
        c = move_coverage(nifty, vix)
        assert abs(c["inside"] + c["fell"] + c["rose"] - 1.0) < 1e-12
        assert move_coverage(nifty * 3.7, vix) == c
        assert move_coverage(nifty, vix * 1.5)["inside"] >= c["inside"]


def test_the_email_states_the_measured_coverage_not_the_normal_curve():
    """RB3: the line said 'about 2 months in 3 stay inside'; since 2014, 74% did and about 1 in 10 fell further."""
    line = expected_move_line([], _nifty(0.0), pd.Series(15.0, index=_DAYS), _DAYS[-1].date())
    assert "2 months in 3" not in line
    assert f"{MOVE_INSIDE:.0%} of such moves stayed inside" in line and f"{MOVE_FELL:.0%} fell further" in line


@pytest.mark.slow
def test_the_stated_coverage_matches_the_store(store_path):
    """If the store gains years and the shares move by 3 points or more, MOVE_INSIDE / MOVE_FELL need re-measuring."""
    import glob

    idx = pd.concat([pd.read_parquet(p) for p in glob.glob(str(store_path / "indices" / "*.parquet"))])
    px = idx.pivot_table(index="date", columns="index_name", values="close").sort_index()
    got = move_coverage(px["NIFTY50"], px["INDIAVIX"])
    assert abs(got["inside"] - MOVE_INSIDE) < 0.03 and abs(got["fell"] - MOVE_FELL) < 0.03, got


# ── RB4: no gate statistic from an infinite return ────────────────────────

def _gate_series(seed: int = 21, n: int = 1500) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0006, 0.01, n), index=pd.bdate_range("2018-01-01", periods=n))


def test_an_infinite_return_is_refused_not_turned_into_a_dsr():
    """Before RB4 one +inf gave DSR 0.24 (the Sharpe dropped it, the skew and kurtosis did not) and t = nan."""
    from nse_engine.validation.significance import mean_test

    rng = np.random.default_rng(5)
    for _ in range(25):
        r = _gate_series(int(rng.integers(1_000_000)))
        r.iloc[int(rng.integers(len(r)))] = float(rng.choice([np.inf, -np.inf]))
        with pytest.raises(ValueError, match="infinite"):
            deflated_sharpe(r, n_trials=10, rf_annual=0.0, sr_variance=1e-4)
        with pytest.raises(ValueError, match="infinite"):
            mean_test(r)


def test_missing_returns_are_still_left_out_as_before():
    """RB4 changes nothing for finite data: a NaN day is dropped exactly as before."""
    from nse_engine.validation.significance import mean_test

    r = _gate_series()
    gappy = r.copy()
    gappy.iloc[[3, 700, 1499]] = np.nan
    kept = r.drop(r.index[[3, 700, 1499]])
    assert deflated_sharpe(gappy, n_trials=10, rf_annual=0.0, sr_variance=1e-4)["dsr"] == \
        deflated_sharpe(kept, n_trials=10, rf_annual=0.0, sr_variance=1e-4)["dsr"]
    assert mean_test(gappy)["t"] == mean_test(kept)["t"]
    assert mean_test(r.iloc[:2])["error"] == "fewer than 3 returns"


# ── TH1: Sun (2024) thesis - crash episodes (sec. 7.3.7) and metric conventions (sec. 2.1.8) ───────

def _index(levels) -> pd.Series:
    return pd.Series(levels, index=pd.bdate_range("2020-01-01", periods=len(levels)), dtype="float64")


_PATH = [100, 110, 120, 108, 96, 110, 125, 120, 112.5, 126, 130, 117, 104]   # -20%, -10%, then -20% still open


def test_crash_episodes_known_answers():
    from nse_engine.scorecard import crash_episodes

    got = crash_episodes(_index(_PATH), 0.15)
    assert [(e["peak"], e["trough"], round(e["fall"], 6), e["open"]) for e in got] == [
        ("2020-01-03", "2020-01-07", -0.2, False), ("2020-01-15", "2020-01-17", -0.2, True)]
    assert len(crash_episodes(_index(_PATH), 0.10)) == 3                      # exactly -10% counts at 10% (float tolerance)
    assert crash_episodes(_index([100.0] * 30), 0.15) == []


def test_crash_episodes_refuse_bad_data_and_never_fill():
    from nse_engine.scorecard import crash_episodes

    for bad in (0.0, -1.0, np.inf):
        with pytest.raises(ValueError):
            crash_episodes(_index(_PATH[:5] + [bad] + _PATH[6:]))
    for share in (0.0, 1.0, -0.2):
        with pytest.raises(ValueError):
            crash_episodes(_index(_PATH), share)
    gappy = _index(_PATH)
    gappy.iloc[[1, 5]] = np.nan                                               # missing closes are left out
    assert crash_episodes(gappy, 0.15) == crash_episodes(_index(_PATH).drop(gappy.index[[1, 5]]), 0.15)
    assert crash_episodes(pd.Series(dtype="float64")) == []


def test_crash_episodes_properties():
    """Seeded: episodes are ordered and disjoint, each falls at least the threshold from peak to trough, the
    index's level does not matter, and a higher threshold never finds an episode a lower one missed."""
    from nse_engine.scorecard import crash_episodes

    rng = np.random.default_rng(17)
    for _ in range(40):
        idx = _index(np.exp(np.cumsum(rng.normal(0.0003, 0.015, 1500))) * 10000)
        lo, hi = crash_episodes(idx, 0.10), crash_episodes(idx, 0.20)
        scaled = crash_episodes(idx * 7.3, 0.10)
        assert [(e["peak"], e["trough"], e["open"]) for e in scaled] == [(e["peak"], e["trough"], e["open"]) for e in lo]
        assert np.allclose([e["fall"] for e in scaled], [e["fall"] for e in lo], rtol=0, atol=1e-12)
        assert all(e["peak"] < e["trough"] and e["fall"] <= -0.10 for e in lo)
        assert all(a["trough"] < b["peak"] for a, b in zip(lo, lo[1:]))
        assert {(e["peak"], e["trough"]) for e in hi} <= {(e["peak"], e["trough"]) for e in lo}


def test_crash_windows_judge_only_what_the_record_spans():
    from nse_engine.scorecard import crash_windows

    idx = _index(_PATH)
    book = pd.Series(-0.01, index=idx.index)                                  # -1% a day throughout
    cw = crash_windows(book, idx, 0.15)
    first = cw["episodes"][0]                                                  # peak 3 Jan, trough 7 Jan: 2 sessions
    assert first["sessions"] == 2 and abs(first["book_return"] - (0.99 ** 2 - 1)) < 1e-12
    assert abs(first["book_max_drawdown"] - (0.99 ** 2 - 1)) < 1e-12 and cw["outside_record"] == 0
    late = crash_windows(book[book.index >= idx.index[6]], idx, 0.15)         # the record starts after the first peak
    assert late["outside_record"] == 1 and [e["peak"] for e in late["episodes"]] == ["2020-01-15"]
    assert crash_windows(pd.Series(dtype="float64"), idx, 0.15)["outside_record"] == 2
    with pytest.raises(ValueError, match="infinite"):
        crash_windows(book.mask(book.index == idx.index[3], np.inf), idx, 0.15)


def test_metric_conventions_against_the_thesis():
    """Sec. 2.1.8 defines SR = E[ret] / sigma[ret] on daily returns with no risk-free rate or annualising, Calmar as
    E[ret] / MDD and downside deviation over the negative returns only.  Ours: Sharpe is the thesis SR x sqrt(252)
    (to 1e-12, rf 0); Sortino's downside is the root mean square of min(r, 0) over every day (Sortino & Price) and
    Calmar is CAGR / |MDD|, so neither equals the thesis's on purpose."""
    r = pd.Series([0.02, -0.01, 0.01, -0.03, 0.015, 0.005], index=pd.bdate_range("2024-01-01", periods=6))
    m = compute_metrics(r, (1 + r).cumprod(), rf_annual=0.0, initial_capital=1.0)
    thesis_sr = r.mean() / r.std(ddof=1)
    assert abs(m["sharpe"] - thesis_sr * np.sqrt(252)) < 1e-12
    ours_down = np.sqrt(np.mean(np.minimum(r.to_numpy(), 0.0) ** 2))         # 0.0129099...
    thesis_down = r[r < 0].std(ddof=1)                                        # 0.0141421...
    assert abs(m["sortino"] - r.mean() / ours_down * np.sqrt(252)) < 1e-12
    assert abs(ours_down - 0.0129099444873581) < 1e-15 and abs(thesis_down - 0.0141421356237310) < 1e-15
    assert abs(m["calmar"] - m["cagr"] / abs(m["max_drawdown"])) < 1e-12


def _panel(close: dict, members: dict):
    from types import SimpleNamespace

    idx = pd.bdate_range("2024-01-01", periods=len(next(iter(close.values()))))
    return SimpleNamespace(close=pd.DataFrame(close, index=idx, dtype="float64")), pd.DataFrame(members, index=idx)


def test_universe_average_known_answers_and_no_look_ahead():
    """Equal weight over yesterday's members; a member that did not trade is marked flat at its last close."""
    from nse_engine.scorecard import universe_average

    data, mask = _panel({"A": [100, 110, 121, 121], "B": [50, 50, np.nan, 45]},
                        {"A": [True, True, True, True], "B": [True, True, True, False]})
    avg = universe_average(data, mask)
    assert np.allclose(avg.to_numpy(), [0.05, 0.05, (0.0 - 0.1) / 2])    # day 3: B flat (no trade); day 4: B's -10%
    later = mask.copy()
    later.iloc[3] = [False, True]                                         # changing today's membership...
    assert np.allclose(universe_average(data, later).iloc[:3], avg.iloc[:3])   # ...cannot move earlier days


def test_universe_average_refuses_an_infinite_day_and_skips_empty_ones():
    from nse_engine.scorecard import universe_average

    data, mask = _panel({"A": [100, 0, 50]}, {"A": [True, True, True]})
    with pytest.raises(ValueError, match="infinite"):
        universe_average(data, mask)
    data, mask = _panel({"A": [100, 101, 102]}, {"A": [False, False, True]})
    assert universe_average(data, mask).empty                             # nobody was a member the day before


def test_market_average_compares_over_common_sessions():
    from nse_engine.scorecard import market_average

    idx = pd.bdate_range("2023-01-02", periods=300)
    wave = np.where(np.arange(len(idx)) % 2 == 0, 2.0, 0.0)              # alternate days, so no moment is degenerate
    book, avg, nifty = (pd.Series(x * wave, index=idx) for x in (0.001, 0.0005, 0.0004))
    out = market_average(book, avg, nifty.iloc[10:], rf=0.0)
    assert out["sessions"] == 290 and out["window"][0] == str(idx[10].date())
    assert out["book"]["cagr"] > out["market_average"]["cagr"] > out["nifty"]["cagr"]
    in_2023 = book.iloc[10:][book.index[10:].year == 2023]
    assert abs(out["by_year"]["2023"]["book"] - (float(np.prod(1 + in_2023.to_numpy())) - 1)) < 1e-12
    assert "error" in market_average(book.iloc[:30], avg, nifty, rf=0.0)
