"""Varsity Module 5 worked examples (kite_connect/options/CONCEPTS.md, IDs M5.x.y)."""

import numpy as np
import pytest

from kite_connect.options import theory as t
from kite_connect.options.theory import BUY, CALL, PUT, SELL, Greeks

approx = pytest.approx


# ---- ch. 1 to 2: call basics
def test_m5_1_land_deal_and_stock_call():
    assert t.expiry_pnl(CALL, BUY, 500_000, 100_000, 1_000_000) == 400_000           # M5.1.1
    assert t.expiry_pnl(CALL, BUY, 500_000, 100_000, 300_000) == -100_000            # M5.1.2
    assert t.expiry_pnl(CALL, BUY, 500_000, 100_000, 500_000) == -100_000            # M5.1.3
    assert [t.expiry_pnl(CALL, BUY, 75, 5, s) for s in (85, 65, 75)] == [5, -5, -5]  # M5.1.4


def test_m5_2_jp_associates():
    lot = 8000
    assert 1.35 * lot == approx(10_800)                                               # M5.2.1
    cash = t.intrinsic_value(CALL, 32, 25) * lot
    profit = t.expiry_pnl(CALL, BUY, 25, 1.35, 32) * lot
    assert (cash, profit) == (approx(56_000), approx(45_200))                         # M5.2.2
    assert profit / (1.35 * lot) * 100 == approx(418.5, abs=0.05)


# ---- ch. 3 to 6: the four positions
def test_m5_3_long_call_bajaj():
    pnl = lambda s: t.expiry_pnl(CALL, BUY, 2050, 6.35, s)
    assert all(pnl(s) == approx(-6.35) for s in range(1990, 2051, 10))                 # M5.3.1
    assert [pnl(s) for s in (2060, 2070, 2080, 2090, 2100)] == approx([3.65, 13.65, 23.65, 33.65, 43.65])
    assert [pnl(s) for s in range(2051, 2060)] == approx(
        [-5.35, -4.35, -3.35, -2.35, -1.35, -0.35, 0.65, 1.65, 2.65])                   # M5.3.3
    assert [pnl(s) for s in (2023, 2072, 2055)] == approx([-6.35, 15.65, -1.35])       # M5.3.4
    be = t.breakeven(CALL, 2050, 6.35)
    assert be == approx(2056.35) and pnl(be) == approx(0)                               # M5.3.5


def test_m5_4_short_call():
    pnl = lambda s: t.expiry_pnl(CALL, SELL, 2050, 6.35, s)
    assert all(pnl(s) == approx(6.35) for s in range(1990, 2051, 10))                   # M5.4.1
    assert [pnl(s) for s in (2060, 2070, 2080, 2090, 2100)] == approx([-3.65, -13.65, -23.65, -33.65, -43.65])
    assert [pnl(s) for s in (2023, 2072, 2055)] == approx([6.35, -15.65, 1.35])        # M5.4.3 (PDF -15.56)
    assert [pnl(s) for s in range(2050, 2060)] == approx(
        [6.35, 5.35, 4.35, 3.35, 2.35, 1.35, 0.35, -0.65, -1.65, -2.65])                # M5.4.4
    assert t.breakeven(CALL, 2050, 6.35) == approx(2056.35)                             # M5.4.5
    spots = np.arange(1900, 2200, 7.0)                                                  # M5.4.6
    assert t.expiry_pnl(CALL, BUY, 2050, 6.35, spots) + pnl(spots) == approx(np.zeros_like(spots))


BN_SPOTS = (16195, 16510, 16825, 17140, 17455, 17770, 18085)


def test_m5_5_long_put_banknifty():
    assert [t.intrinsic_value(PUT, s, 18400) for s in BN_SPOTS] == [2205, 1890, 1575, 1260, 945, 630, 315]
    assert [t.expiry_pnl(PUT, BUY, 18400, 315, s) for s in BN_SPOTS] == [1890, 1575, 1260, 945, 630, 315, 0]
    assert all(t.expiry_pnl(PUT, BUY, 18400, 315, s) == -315 for s in (18400, 18715, 19030, 19345, 19660))
    assert [t.expiry_pnl(PUT, BUY, 18400, 315, s) for s in (16510, 19660)] == [1575, -315]   # M5.5.3
    assert t.breakeven(PUT, 18400, 315) == 18085                                             # M5.5.4
    assert t.expiry_pnl(PUT, BUY, 18400, 315, 17000) == 1085                                 # M5.5.5


def test_m5_6_short_put_mirrors():
    grid = BN_SPOTS + (18400, 18715, 19030, 19345, 19660)
    for s in grid:                                                                          # M5.6.1
        assert t.expiry_pnl(PUT, SELL, 18400, 315, s) == -t.expiry_pnl(PUT, BUY, 18400, 315, s)
    assert [t.expiry_pnl(PUT, SELL, 18400, 315, s) for s in (16510, 19660)] == [-1575, 315]
    assert t.breakeven(PUT, 18400, 315) == 18085                                            # M5.6.3


def test_m5_7_point_capture():
    assert 2 * 1000 == 2000 and 2 * 2000 == 4000                                            # M5.7.1, M5.7.2


# ---- ch. 8: moneyness
def test_m5_8_moneyness():
    assert t.intrinsic_value(CALL, 8070, 8050) == 20                                        # M5.8.1
    assert [t.intrinsic_value(CALL, 310, 280), t.intrinsic_value(PUT, 980, 1040),
            t.intrinsic_value(CALL, 918, 920), t.intrinsic_value(PUT, 88, 80)] == [30, 60, 0, 0]
    nifty = list(range(7100, 8701, 50))
    assert t.atm_strike(8060, nifty) == 8050                                                # M5.8.3
    assert [t.moneyness(CALL, 8060, k, nifty) for k in (7100, 7500, 8100, 8300)] == \
        ["DEEP ITM", "DEEP ITM", "OTM", "DEEP OTM"]                                         # M5.8.4
    assert t.intrinsic_value(CALL, 8060, 7100) == 960 and t.intrinsic_value(CALL, 8060, 7500) == 560
    assert [t.moneyness(CALL, 8060, k) for k in (7100, 7500, 8100, 8300)] == ["ITM", "ITM", "OTM", "OTM"]
    assert t.atm_strike(8202, nifty) == 8200                                                # M5.8.5
    assert [t.moneyness(PUT, 8200, k) for k in (7500, 8000, 8300, 8500)] == ["OTM", "OTM", "ITM", "ITM"]
    assert [t.intrinsic_value(PUT, 8200, k) for k in (8300, 8500)] == [100, 300]
    assert [t.intrinsic_value(PUT, 8202, k) for k in (8300, 8500)] == [98, 298]
    assert t.atm_strike(68.7, np.arange(60, 77.5, 2.5)) == 67.5                             # M5.8.6
    assert -t.expiry_pnl(CALL, BUY, 920, 15, 918) == 15                                     # M5.8.7


# ---- ch. 9 to 11: delta
@pytest.mark.parametrize("premium,delta,ds,expected", [
    (133, 0.55, 22, 145.1), (133, 0.55, -88, 84.6),                                          # M5.9.1, M5.9.2
    (128, -0.55, 42, 104.9), (128, -0.55, -38, 148.9),                                       # M5.9.4, M5.9.5
    (12, 0.05, 100, 17), (20, 0.25, 100, 45), (60, 0.5, 100, 110),                           # M5.10.1 to 3
    (105, 0.8, 100, 185), (210, 1.0, 100, 310), (450, 1.0, 30, 480),                         # M5.10.4, M5.10.5
])
def test_m5_9_10_delta_first_order(premium, delta, ds, expected):
    assert t.delta_gamma_step(premium, delta, 0.0, ds)[0] == approx(expected)


def test_m5_9_3_and_10_6_delta_effects():
    assert (0.05 * 100, 0.2 * 100) == (5, 20)                                                # M5.9.3
    bajaj = [(0.05, 3), (0.3, 7), (0.5, 12), (0.7, 22), (1.0, 75)]                            # M5.10.6
    new = [t.delta_gamma_step(p, d, 0, 30)[0] for d, p in bajaj]
    assert new == approx([4.5, 16, 27, 43, 105])
    assert [(n - p) / p * 100 for n, (_, p) in zip(new, bajaj)] == approx([50, 128.57, 125, 95.45, 40], abs=0.01)
    assert (17 - 12) / 12 * 100 == approx(41.67, abs=0.01)


def _pos_delta(items):
    return t.position_greeks((q, Greeks(delta=d)) for q, d in items).delta


def test_m5_11_position_delta():
    base = [(1, 0.7), (1, 0.5), (1, 0.05)]
    assert _pos_delta(base) == approx(1.25) and _pos_delta(base) * 50 == approx(62.5)        # M5.11.1
    assert _pos_delta(base + [(1, -1.0)]) == approx(0.25)                                    # M5.11.2
    assert _pos_delta(base + [(1, -1.0)]) * 50 == approx(12.5)
    assert _pos_delta(base + [(2, -1.0)]) == approx(-0.75)                                   # M5.11.3
    assert _pos_delta(base + [(2, -1.0)]) * 50 == approx(-37.5)
    assert _pos_delta([(1, 0.5), (1, -0.5)]) == approx(0)                                    # M5.11.4
    assert _pos_delta([(-1, 0.5), (1, -0.5)]) == approx(-1.0)                                # M5.11.5
    assert _pos_delta([(5, 1.0)]) == 5 and _pos_delta([(-5, -1.0)]) == 5                     # M5.11.6
    assert t.position_greeks([(1, Greeks(price=8000, delta=1.0))]).gamma == 0                # futures: delta 1, gamma 0


def test_m5_11_7_delta_as_probability_is_labelled():
    p = t.probability_itm(CALL, 8000, 8300, 20, 0.0552, 0.15)
    assert set(p) == {"n_d2", "delta_approx"}
    assert 0 < p["n_d2"] < p["delta_approx"] < 1         # N(d2) < N(d1) for a call
    assert t.black_scholes(CALL, 8000, 8300, 20, 0.0552, 0.15).delta == approx(p["delta_approx"])


# ---- ch. 12 to 13: gamma
def test_m5_13_delta_gamma_steps():
    p, d = t.delta_gamma_step(26, 0.3, 0.0025, 70)
    assert (p, d) == (approx(47), approx(0.475))                                             # M5.13.1
    p, d = t.delta_gamma_step(p, d, 0.0025, 70)
    assert (p, d) == (approx(80.25), approx(0.65))                                           # M5.13.2
    p, d = t.delta_gamma_step(p, d, 0.0025, -50)
    assert (p, d) == (approx(47.75), approx(0.525))                                          # M5.13.3
    assert t.delta_gamma_step(0, -0.5, 0.004, 10)[1] == approx(-0.46)                        # M5.13.4
    assert t.delta_gamma_step(0, -0.5, 0.004, -10)[1] == approx(-0.54)
    new_delta = t.delta_gamma_step(0, 0.5, 0.005, 70)[1]                                     # M5.13.5
    assert 10 * 0.5 == 5 and 10 * new_delta == approx(8.5)
    assert t.black_scholes(CALL, 8000, 8000, 10, 0.06, 0.15).gamma == \
        approx(t.black_scholes(PUT, 8000, 8000, 10, 0.06, 0.15).gamma)                       # same for calls and puts


# ---- ch. 14: theta
def test_m5_14_time_value_and_decay():
    assert [t.intrinsic_value(CALL, 8423, 8350), t.intrinsic_value(CALL, 8423, 8450),
            t.intrinsic_value(PUT, 8423, 8400), t.intrinsic_value(PUT, 8423, 8450)] == [73, 0, 0, 27]
    assert t.time_value(99.4, CALL, 8531, 8600) == approx(99.4)                              # M5.14.2
    assert t.time_value(87.9, CALL, 8537.9, 8600) == approx(87.9) and 99.4 - 87.9 == approx(11.5)
    assert (t.intrinsic_value(CALL, 8514.5, 8450), t.time_value(160, CALL, 8514.5, 8450)) == (approx(64.5), approx(95.5))
    assert t.time_value(0.30, CALL, 179.6, 190) == approx(0.30)                              # M5.14.5
    assert 2.75 - 0.05 == approx(2.70)                                                       # M5.14.6
    assert 54 - 3 * 0.75 == approx(51.75) and 54 - 51.75 == approx(2.25)                     # M5.14.7


# ---- ch. 15 to 17: volatility and ranges
def test_m5_15_mean_sd_and_simple_range():
    billy, mike = [20, 23, 21, 24, 19, 23], [45, 13, 18, 12, 26, 19]
    m, sd = t.mean_sd(billy)
    assert (sum(billy), m, sd ** 2, sd) == (130, approx(21.67, abs=0.005), approx(3.22, abs=0.005), approx(1.79, abs=0.01))
    m2, sd2 = t.mean_sd(mike)
    assert (sum(mike), m2, sd2) == (133, approx(22.17, abs=0.005), approx(11.19, abs=0.005))
    assert (21.6 - sd, 21.6 + sd) == (approx(19.81, abs=0.01), approx(23.39, abs=0.01))     # M5.15.3, PDF mean 21.6
    assert (m2 - sd2, m2 + sd2) == (approx(10.98, abs=0.01), approx(33.36, abs=0.01))         # PDF 33.34 from SD 11.18
    assert t.price_range(8547, 0.165, method="simple") == (approx(7136.7, abs=0.05), approx(9957.3, abs=0.05))
    assert t.price_range(2585, 0.27, method="simple") == (approx(1887.05), approx(3282.95))  # M5.15.5


WIPRO = [558.75, 570.9, 576.85, 551.05, 557.05, 550.75, 544.4, 536, 548.65, 549.55, 551.4, 552.65, 548.05, 542.95]


def test_m5_16_log_returns_and_annualisation():
    r = t.log_returns(WIPRO) * 100                                                            # M5.16.1
    assert list(np.round(r, 2)) == approx([2.15, 1.04, -4.58, 1.08, -1.14, -1.16, -1.56, 2.33, 0.16, 0.34, 0.23, -0.84, -0.93])
    assert t.annualize_vol(0.0147) == approx(0.2808, abs=5e-5)                                # M5.16.2
    assert t.deannualize_vol(0.255) == approx(0.0133, abs=5e-5)                              # M5.16.3


def test_m5_17_lognormal_ranges():
    assert t.price_range(8337, 0.1661, 0.0966, 1, "lognormal") == (approx(7777, abs=1), approx(10841, abs=1))
    assert t.price_range(8337, 0.1661, 0.0966, 2, "lognormal") == (approx(6587, abs=1), approx(12800, abs=1))
    assert t.price_range(8337, 0.0573, 0.0115, 1, "lognormal") == (approx(7963, abs=1), approx(8930, abs=1))
    assert t.price_range(8337, 0.0573, 0.0115, 2, "lognormal") == (approx(7520, abs=1), approx(9457, abs=1))
    assert t.annualize_vol(0.01046, 252) == approx(0.1660, abs=1e-4)                         # M5.17.4
    assert t.period_vol(0.01046, 30) == approx(0.0573, abs=1e-4)
    assert [t.sd_coverage(k) for k in (1, 2, 3)] == approx([0.6827, 0.9545, 0.9973], abs=1e-4)


# ---- ch. 18: applications
def test_m5_18_writing_range_and_strikes():
    lo, hi = t.expected_range(8462, 0.0089, 16, 0.0004)[1]
    assert (lo, hi) == (approx(8214, abs=1), approx(8818, abs=1))                             # M5.18.1
    chain = [8600, 8650, 8700, 8750, 8800, 8850, 8900, 8950]
    picks = t.sd_writing_strikes(8462, 0.0089, 16, chain, CALL, k=1, mean_daily=0.0004)
    assert picks[:2] == [8850, 8900] and min(picks) >= 8850                                   # M5.18.2
    assert 7.45 * 25 == approx(186.25) and 186.25 / 12_000 * 100 == approx(1.55, abs=0.01)    # M5.18.3
    cap = 500_000
    assert (0.35 * cap, 0.40 * cap, 0.25 * cap, 0.35 * 0.25 * cap) == (175_000, 200_000, 125_000, 43_750)
    assert t.writing_sd_multiple(16) is None and t.writing_sd_multiple(10) == 2 and t.writing_sd_multiple(4) == 1


def test_m5_18_volatility_stop_and_rrr():
    assert t.period_vol(0.018, 5) == approx(0.0402, abs=5e-5)                                 # M5.18.5
    assert t.volatility_stop_loss(395, 0.018, 5) == approx(379.1, abs=0.05)
    assert t.volatility_stop_loss(395, 0.018, 5, SELL) == approx(410.9, abs=0.05)
    assert 417 - 395 == 22                                                                    # M5.18.6 (PDF 32)
    assert t.risk_reward(395, 417, 385) == approx(2.2) and t.risk_reward(395, 417, 375) == approx(1.1)


# ---- ch. 19 to 20: vega, cone
def test_m5_19_20_vega_and_cone():
    assert 0.15 * 1 == 0.15                                                                   # M5.19.1
    cone = t.cone_stats([41, 38, 33, 28, 28, 41, 26, 22, 56, 19, 13, 34, 17, 41, 21])         # M5.20.1
    assert cone["max"] == 56 and cone["min"] == 13 and cone["mean"] == approx(30.5, abs=0.05)
    assert (cone["plus1"], cone["plus2"], cone["minus1"], cone["minus2"]) == \
        (approx(42.1, abs=0.05), approx(53.7, abs=0.05), approx(19.0, abs=0.05), approx(7.4, abs=0.05))
    assert t.black_scholes(PUT, 7794.05, 6800, 13, 0.0725, 0.4145).price == approx(8.6, abs=0.5)   # M5.20.2


def test_m5_20_3_directional_properties():
    vegas = [t.black_scholes(CALL, 8000, 8000, d, 0.06, 0.15).vega for d in (30, 15, 5)]
    assert vegas[0] > vegas[1] > vegas[2]
    gammas = [t.black_scholes(CALL, 8000, 8000, d, 0.06, 0.15).gamma for d in (30, 15, 5, 1)]
    assert gammas == sorted(gammas)
    assert t.black_scholes(CALL, 8000, 8800, 20, 0.06, 0.40).delta > t.black_scholes(CALL, 8000, 8800, 20, 0.06, 0.10).delta


def test_m5_20_volatility_cone_from_prices():
    rng = np.random.default_rng(7)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
    cone = t.volatility_cone(prices, windows=(10, 30, 90))
    assert set(cone) == {10, 30, 90}
    for row in cone.values():
        assert row["min"] <= row["mean"] <= row["max"] and row["minus1"] < row["mean"] < row["plus1"]
        assert 0.1 < row["mean"] < 0.3          # 1% daily x sqrt(365) ~ 19%


# ---- ch. 21: Black-Scholes, IV, parity
def test_m5_21_1_icici_calculator():
    c = t.black_scholes(CALL, 272.7, 280, 1, 0.074769, 0.4355)
    p = t.black_scholes(PUT, 272.7, 280, 1, 0.074769, 0.4355)
    assert (c.price, p.price) == (approx(0.39, abs=0.005), approx(7.63, abs=0.005))
    assert (c.delta, p.delta) == (approx(0.127, abs=0.001), approx(-0.873, abs=0.001))
    assert c.gamma == approx(0.0336, abs=1e-4) and c.vega == approx(0.030, abs=0.001)
    assert (c.theta, p.theta) == (approx(-0.656, abs=0.001), approx(-0.598, abs=0.001))
    assert (c.rho, p.rho) == (approx(0.001, abs=0.0005), approx(-0.007, abs=0.0005))


def test_m5_21_2_iv_round_trip_and_bounds():
    price = t.black_scholes(CALL, 272.7, 280, 1, 0.074769, 0.4355).price
    assert t.implied_volatility(CALL, price, 272.7, 280, 1, 0.074769) == approx(0.4355, abs=1e-4)
    for ty in (CALL, PUT):
        px = t.black_scholes(ty, 7485, 7600, 16, 0.0725, 0.18).price
        assert t.implied_volatility(ty, px, 7485, 7600, 16, 0.0725) == approx(0.18, abs=1e-6)
    assert t.implied_volatility(PUT, 80, 7485, 7600, 16, 0.0725) is None      # below K e^-rT - S = 90.9
    assert t.implied_volatility(CALL, 5, 7485, 7600, 0, 0.0725) is None       # no time left


def test_m5_21_3_4_put_call_parity():
    for s in (1100, 1350):                                                   # M5.21.3: put + share = call + cash
        a = t.intrinsic_value(PUT, s, 1200) + s
        b = t.intrinsic_value(CALL, s, 1200) + 1200
        assert a == b == max(s, 1200)
    c = t.black_scholes(CALL, 272.7, 280, 1, 0.074769, 0.4355).price         # M5.21.4
    p = t.black_scholes(PUT, 272.7, 280, 1, 0.074769, 0.4355).price
    assert t.parity_residual(c, p, 272.7, 280, 1, 0.074769) == approx(0, abs=1e-9)
    c2 = t.black_scholes(CALL, 7485, 7400, 16, 0.0725, 0.18, dividend=0.012).price   # any strike, with q
    p2 = t.black_scholes(PUT, 7485, 7400, 16, 0.0725, 0.18, dividend=0.012).price
    assert t.parity_residual(c2, p2, 7485, 7400, 16, 0.0725, 0.012) == approx(0, abs=1e-9)


def test_black_scholes_at_expiry_is_intrinsic():
    g = t.black_scholes(CALL, 8100, 8000, 0, 0.06, 0.2)
    assert (g.price, g.delta, g.gamma) == (100, 1.0, 0.0)
    assert t.black_scholes(PUT, 8100, 8000, 0, 0.06, 0.2).price == 0


# ---- ch. 22 to 23: buying strikes and case studies
def test_m5_23_case_studies():
    assert (52 - 45.75) == approx(6.25) and (52 - 45.75) * 500 == approx(3125)                # M5.23.1 (PDF 7)
    assert (203 + 176) - (191 + 178) == 10                                                     # M5.23.2
    assert (48 + 47) - (55 + 20) == 20                                                         # M5.23.3
    assert 41.5 - 18.9 == approx(22.6) and (41.5 - 18.9) / 18.9 * 100 == approx(119.6, abs=0.05)
