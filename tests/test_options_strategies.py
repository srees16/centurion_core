"""Varsity Module 6 worked examples and the selector (kite_connect/options/STRATEGIES.md, IDs M6.x.y)."""

import math

import pytest

from kite_connect.options import selector as sel
from kite_connect.options import strategies as st
from kite_connect.options.theory import BUY, CALL, PUT, SELL, Greeks

approx = pytest.approx
INF = math.inf


def payoffs(strategy, spots):
    return [strategy.payoff(s) for s in spots]


def assert_formulas_match_legs(strategy):
    assert strategy.check_generalization() == {}


# ---- ch. 2: bull call spread
def test_m6_2_1_2_bull_call_spread():
    s = st.BullCallSpread(7800, 79, 7900, 25)
    assert (s.net_debit, s.max_loss, s.max_profit, s.breakevens()) == (54, 54, 46, [7854])
    assert payoffs(s, range(7000, 7801, 100)) == [-54] * 9
    assert payoffs(s, range(7900, 8501, 100)) == [46] * 7
    assert_formulas_match_legs(s)


@pytest.mark.parametrize("legs,debit,max_profit,be", [
    ((7700, 296, 7800, 227), 69, 31, 7769),        # M6.2.3, the spec's test
    ((7800, 227, 7900, 167), 60, 40, 7860),        # M6.2.4
    ((7900, 167, 8000, 116), 51, 49, 7951),        # M6.2.5
])
def test_m6_2_3_to_6_bull_call_sets(legs, debit, max_profit, be):
    s = st.BullCallSpread(*legs)
    assert (s.net_debit, s.max_profit, s.max_loss, s.breakevens()) == (debit, max_profit, debit, [be])
    assert_formulas_match_legs(s)                  # M6.2.6


# ---- ch. 3: bull put spread
def test_m6_3_1_2_bull_put_spread():
    s = st.BullPutSpread(7700, 72, 7900, 163)
    assert (s.net_credit, s.max_profit, s.max_loss, s.breakevens()) == (91, 91, 109, [7809])
    assert payoffs(s, range(7000, 7701, 100)) == [-109] * 8
    assert s.payoff(7800) == -9 and payoffs(s, range(7900, 8501, 100)) == [91] * 7
    assert_formulas_match_legs(s)


@pytest.mark.parametrize("legs,credit,max_loss,be", [
    ((7500, 62, 7700, 137), 75, 125, 7625), ((7400, 40, 7800, 198), 158, 242, 7642), ((7500, 62, 7800, 198), 136, 164, 7664),
])
def test_m6_3_3_to_5_bull_put_variants(legs, credit, max_loss, be):
    s = st.BullPutSpread(*legs)
    assert (s.net_credit, s.max_loss, s.breakevens()) == (credit, max_loss, [be])
    assert_formulas_match_legs(s)


# ---- ch. 4: call ratio back spread
def test_m6_4_call_ratio_back_spread():
    s = st.CallRatioBackSpread(7600, 201, 7800, 78)
    assert (s.net_credit, s.max_loss, s.max_loss_region(), s.breakevens()) == (45, 155, (7800, 7800), [7645, 7955])
    assert s.max_profit == INF
    spots = [7000, 7300, 7600, 7645, 7700, 7800, 7900, 7955, 8000, 8100, 8500]
    assert payoffs(s, spots) == [45, 45, 45, 0, -55, -155, -55, 0, 45, 145, 545]
    assert_formulas_match_legs(s)
    debit = st.CallRatioBackSpread(7600, 150, 7800, 78)
    assert debit.conditions() and pytest.raises(ValueError, debit.check_generalization)


# ---- ch. 5: bear call ladder
def test_m6_5_bear_call_ladder():
    s = st.BearCallLadder(7600, 247, 7800, 117, 7900, 70)
    assert (s.net_credit, s.max_loss, s.max_loss_region(), s.breakevens()) == (60, 140, (7800, 7900), [7660, 8040])
    spots = [7000, 7600, 7660, 7700, 7800, 7900, 8000, 8040, 8100, 8300, 8700]
    assert payoffs(s, spots) == [60, 60, 0, -40, -140, -140, -40, 0, 60, 260, 660]
    assert_formulas_match_legs(s)


# ---- ch. 6: synthetic long and its arbitrage
def test_m6_6_synthetic_long():
    s = st.SyntheticLong(7400, 107, 80)
    assert (s.net_debit, s.breakevens()) == (27, [7427])
    assert payoffs(s, [6700, 7200, 7400, 7427, 7600, 8400]) == [-727, -227, -27, 0, 173, 973]
    assert (s.payoff(7627), s.payoff(7227)) == (200, -200)                     # M6.6.3
    assert_formulas_match_legs(s)
    with_iv = s.with_ivs([0.15, 0.15])
    assert with_iv.net_greeks(7389, 10, 0.06).delta == approx(1.0, abs=0.01)   # mimics futures


def test_m6_6_4_arbitrage_constant_and_costs():
    arb = st.synthetic_long_arbitrage(7316, 7300, 79.5, 73.85)
    assert arb["gross"] == approx(10.35)
    assert payoffs(arb["position"], range(6700, 8401, 100)) == approx([10.35] * 18)
    costed = st.synthetic_long_arbitrage(7316, 7300, 79.5, 73.85, days_to_expiry=7, rate=0.0725, charges=3.0, buffer=2.0)
    assert costed["after_carry"] < costed["gross"] and costed["net"] == approx(costed["after_carry"] - 3.0)
    assert costed["opportunity"] is True
    assert st.synthetic_long_arbitrage(7316, 7300, 79.5, 73.85, charges=9.0, buffer=2.0)["opportunity"] is False


# ---- ch. 7: bear put spread
def test_m6_7_1_2_bear_put_spread():
    s = st.BearPutSpread(7400, 73, 7600, 165)
    assert (s.net_debit, s.breakevens(), s.max_profit, s.max_loss) == (92, [7508], 108, 92)
    assert payoffs(s, range(6600, 7401, 100)) == [108] * 9
    assert (s.payoff(7500), s.payoff(7508)) == (8, 0) and payoffs(s, range(7600, 8101, 100)) == [-92] * 6
    assert_formulas_match_legs(s)


@pytest.mark.parametrize("strike,call,put,dc,dp,tc,tp,rc,rp,gamma,vega", [
    (7600, 73.52, 164.41, 0.382, -0.618, -3.913, -2.408, 1.220, -2.101, 0.0014, 5.974),   # M6.7.3
    (7400, 174.23, 65.75, 0.658, -0.342, -4.181, -2.716, 2.082, -1.152, 0.0013, 5.757),   # M6.7.4
])
def test_m6_7_3_4_black_scholes_screens(strike, call, put, dc, dp, tc, tp, rc, rp, gamma, vega):
    from kite_connect.options.theory import black_scholes
    c = black_scholes(CALL, 7485, strike, 16, 0.0725, 0.18)
    p = black_scholes(PUT, 7485, strike, 16, 0.0725, 0.18)
    assert (c.price, p.price) == (approx(call, abs=0.01), approx(put, abs=0.01))
    assert (c.delta, p.delta) == (approx(dc, abs=0.001), approx(dp, abs=0.001))
    assert (c.theta, p.theta) == (approx(tc, abs=0.001), approx(tp, abs=0.001))
    assert (c.rho, p.rho) == (approx(rc, abs=0.001), approx(rp, abs=0.001))
    assert c.gamma == approx(gamma, abs=1e-4) and c.vega == approx(vega, abs=0.001)


def test_m6_7_5_bear_put_net_delta():
    s = st.BearPutSpread(7400, 73, 7600, 165, iv=0.18)
    assert s.net_greeks(7485, 16, 0.0725).delta == approx(-0.276, abs=0.001)


# ---- ch. 8: bear call spread
def test_m6_8_bear_call_spread():
    s = st.BearCallSpread(7100, 136, 7400, 38)
    assert (s.net_credit, s.breakevens(), s.max_profit, s.max_loss) == (98, [7198], 98, 202)
    assert payoffs(s, range(6600, 7101, 100)) == [98] * 6
    assert payoffs(s, [7198, 7202, 7302]) == [0, -4, -104]
    assert payoffs(s, range(7402, 8103, 100)) == [-202] * 8
    assert_formulas_match_legs(s)
    given = st.Strategy([st.Leg(CALL, BUY, 7400, 38, greeks=Greeks(delta=0.32)),
                         st.Leg(CALL, SELL, 7100, 136, greeks=Greeks(delta=0.89))])
    assert given.net_greeks(7222, 10, 0.07).delta == approx(-0.57)            # M6.8.3


# ---- ch. 9: put ratio back spread
def test_m6_9_put_ratio_back_spread():
    s = st.PutRatioBackSpread(7200, 46, 7500, 134)
    assert (s.net_credit, s.max_loss, s.max_loss_region(), s.breakevens()) == (42, 258, (7200, 7200), [6942, 7458])
    spots = [6500, 6800, 6900, 6942, 7000, 7100, 7200, 7300, 7400, 7458, 7500, 7700, 8000]
    assert payoffs(s, spots) == [442, 142, 42, 0, -58, -158, -258, -158, -58, 0, 42, 42, 42]
    assert_formulas_match_legs(s)
    given = st.Strategy([st.Leg(PUT, SELL, 7500, 134, greeks=Greeks(delta=-0.55)),
                         st.Leg(PUT, BUY, 7200, 46, ratio=2, greeks=Greeks(delta=-0.29))])
    assert given.net_greeks(7506, 10, 0.07).delta == approx(-0.03)            # M6.9.3


# ---- ch. 10 to 12: straddles and strangles
STRADDLE_GRID = [6500, 7200, 7435, 7500, 7600, 7700, 7765, 7800, 8000, 8700]


def test_m6_10_long_straddle():
    s = st.LongStraddle(7600, 77, 88)
    assert (s.net_debit, s.breakevens(), s.max_loss, s.max_loss_region()) == (165, [7435, 7765], 165, (7600, 7600))
    assert s.max_profit == INF
    assert payoffs(s, STRADDLE_GRID) == [935, 235, 0, -65, -165, -65, 0, 35, 235, 935]
    assert 165 / 7600 * 100 == approx(2.17, abs=0.005)                          # M6.10.3
    assert_formulas_match_legs(s)


def test_m6_11_short_straddle():
    s = st.ShortStraddle(7600, 77, 88)
    assert (s.net_credit, s.breakevens(), s.max_profit, s.max_loss) == (165, [7435, 7765], 165, INF)
    assert payoffs(s, STRADDLE_GRID) == [-x for x in payoffs(st.LongStraddle(7600, 77, 88), STRADDLE_GRID)]
    assert_formulas_match_legs(s)
    assert (48 + 47) - (55 + 20) == 20                                          # M6.11.3


def test_m6_12_strangles():
    s = st.LongStrangle(7700, 28, 8100, 32)
    assert (s.net_debit, s.breakevens(), s.max_loss, s.max_loss_region()) == (60, [7640, 8160], 60, (7700, 8100))
    grid = [7000, 7500, 7600, 7640, 7700, 7900, 8100, 8160, 8200, 8300, 8800]
    assert payoffs(s, grid) == [640, 140, 40, 0, -60, -60, -60, 0, 40, 140, 640]
    assert_formulas_match_legs(s)
    short = st.ShortStrangle(7700, 28, 8100, 32)                              # M6.12.3
    assert (short.net_credit, short.breakevens(), short.max_profit_region()) == (60, [7640, 8160], (7700, 8100))
    assert payoffs(short, grid) == [-x for x in payoffs(s, grid)]
    assert_formulas_match_legs(short)
    assert st.LongStraddle(5900, 66, 57).breakevens() == [5777, 6023]         # M6.12.4 (PDF 5798 / 6044)
    given = st.Strategy([st.Leg(PUT, BUY, 7700, 28, greeks=Greeks(delta=-0.3)),
                         st.Leg(CALL, BUY, 8100, 32, greeks=Greeks(delta=0.3))])
    assert given.net_greeks(7921, 10, 0.07).delta == approx(0)                # M6.12.5


# ---- ch. 13: max pain and PCR
def test_m6_13_1_three_strike_max_pain():
    mp, table = st.max_pain([7700, 7800, 7900], [1_823_400, 3_448_575, 5_367_450], [5_783_025, 4_864_125, 2_559_375])
    assert list(table.total_loss) == [998_287_500, 438_277_500, 709_537_500] and mp == 7800


STRIKES = list(range(7000, 8601, 100))
CALL_OI = [1404300, 335700, 482100, 422475, 963900, 999975, 785550, 1823400, 3448575, 5367450, 6510975,
           5900325, 5113350, 3844500, 2135625, 2252250, 1083750]
PUT_OI = [4087050, 1029150, 2977875, 1975650, 2336700, 4548450, 3690900, 5783025, 4864125, 2559375, 1447125,
          310500, 248775, 355725, 255525, 488475, 58500]
PDF_TOTALS = [20691180000, 17538622500, 14522550000, 11852475000, 9422212500, 7322010000, 5776650000, 4678935000,
              4341862500, 4836060000, 6122940000, 8205630000, 10909402500, 14149387500, 17809395000, 21708517500,
              25881712500]


def test_m6_13_2_3_full_table_and_pcr():
    mp, table = st.max_pain(STRIKES, CALL_OI, PUT_OI)
    assert list(table.total_loss) == PDF_TOTALS
    assert mp == 7800 and table.total_loss.min() == 4_341_862_500
    assert (sum(PUT_OI), sum(CALL_OI)) == (37_016_925, 42_874_200)
    pcr = st.put_call_ratio(PUT_OI, CALL_OI)
    assert pcr == approx(0.8634, abs=5e-5) and st.pcr_signal(pcr) == "normal"
    assert st.pcr_signal(1.4).startswith("bullish") and st.pcr_signal(0.4).startswith("bearish")
    assert st.pcr_signal(1.2) == "normal"                                      # 1 to 1.3: undefined, treated as normal


def test_m6_13_4_modified_max_pain():
    band = st.modified_max_pain_band(7800, 0.05, strike_step=100)
    assert (band["low"], band["high"], band["write_calls_from"]) == (7800, approx(8190), 8200)
    assert st.modified_max_pain_band(7800, 0.05, strikes=STRIKES)["write_calls_from"] == 8200


# ---- engine behaviour
def test_payoff_table_and_chart(tmp_path):
    s = st.BullCallSpread(7700, 296, 7800, 227)
    df = s.payoff_table(7600, 7900, 100)
    assert list(df.spot) == [7600, 7700, 7800, 7900] and list(df.net) == [-69, -69, 31, 31]
    assert len(df.columns) == 4
    pytest.importorskip("matplotlib")
    path = tmp_path / "bcs.png"
    s.payoff_chart(7500, 8000, 5, path=str(path))
    assert path.stat().st_size > 0


def test_net_greeks_signed_sum():
    s = st.LongStraddle(7600, 77, 88, iv=0.16)
    g = s.net_greeks(7600, 10, 0.06)
    assert abs(g.delta) < 0.1 and g.gamma > 0 and g.theta < 0 and g.vega > 0
    short = st.ShortStraddle(7600, 77, 88, iv=0.16).net_greeks(7600, 10, 0.06)
    assert (short.gamma, short.vega) == (approx(-g.gamma), approx(-g.vega))
    with pytest.raises(ValueError):
        st.LongStraddle(7600, 77, 88).net_greeks(7600, 10, 0.06)               # no iv, no given greeks


def test_check_generalization_reports_a_wrong_formula():
    class Broken(st.BullCallSpread):
        def generalization(self):
            g = super().generalization()
            g["breakevens"] = (self.k_high + 1,)
            return g
    assert set(Broken(7700, 296, 7800, 227).check_generalization()) == {"breakevens"}


def test_all_strategies_registered():
    assert len(st.STRATEGIES) == 12 and {c.chapter for c in st.STRATEGIES.values()} == set(range(2, 13))


# ---- selector
@pytest.mark.parametrize("dte,target,bucket", [
    (25, 3, ("1st", "5d")), (25, 12, ("1st", "15d")), (30, 20, ("1st", "25d")), (25, None, ("1st", "expiry")),
    (10, 0, ("2nd", "same_day")), (10, 4, ("2nd", "5d")), (14, 9, ("2nd", "10d")), (6, 6, ("2nd", "expiry")),
])
def test_target_bucket(dte, target, bucket):
    assert sel.target_bucket(dte, target) == bucket


def test_m5_22_1_naked_buy_table():
    rows = {(25, 4): "far OTM", (25, 14): "ATM or slightly OTM", (30, 24): "slightly ITM", (25, None): "ITM",
            (10, 0): "far OTM", (10, 5): "slightly OTM", (12, 10): "slightly ITM or ATM", (7, None): "ITM"}
    for (dte, target), label in rows.items():
        assert sel.naked_buy_strike(dte, target) == label


def test_strike_for_labels():
    ks = list(range(7500, 8501, 100))
    assert sel.strike_for("ATM", CALL, 8020, ks) == 8000
    assert sel.strike_for("far OTM", CALL, 8020, ks) == 8300 and sel.strike_for("far OTM", PUT, 8020, ks) == 7700
    assert sel.strike_for("slightly ITM or ATM", PUT, 8020, ks) == 8100
    assert sel.strike_for("deep ITM", CALL, 7520, ks) == 7500                  # clipped to the chain


def _top(**kw):
    return sel.select(sel.MarketContext(**kw))[0]


def test_selector_rankings_follow_the_pdf():
    assert _top(view="moderate_bull", days_to_expiry=25, rich_side="puts").strategy == "Bull Put Spread"
    assert _top(view="moderate_bull", days_to_expiry=10, iv_level="low").strategy == "Bull Call Spread"
    assert _top(view="moderate_bear", days_to_expiry=20, rich_side="calls").strategy == "Bear Call Spread"
    assert _top(view="strong_bear", days_to_expiry=20).strategy == "Put Ratio Back Spread"
    assert _top(view="neutral_range", days_to_expiry=2).strategy == "Short Strangle"
    assert _top(view="neutral_big_move", days_to_expiry=20, cost_sensitive=True).strategy == "Long Strangle"
    assert _top(view="futures_like", days_to_expiry=20).strategy == "Synthetic Long"
    crush = _top(view="neutral_big_move", days_to_expiry=20, event_vs_consensus="matches", vol_view="falling")
    assert crush.score < 0 and any("IV crush" in r for r in crush.reasons)


def test_selector_strike_guidance():
    early = {c.strategy: c for c in sel.select(sel.MarketContext(view="strong_bull", days_to_expiry=25))}
    late = {c.strategy: c for c in sel.select(sel.MarketContext(view="strong_bull", days_to_expiry=8))}
    assert early["Call Ratio Back Spread"].strikes == {"sell 1 CE": "slightly ITM", "buy 2 CE": "slightly OTM"}
    assert late["Call Ratio Back Spread"].strikes == {"sell 1 CE": "deep ITM", "buy 2 CE": "slightly ITM"}
    bear = {c.strategy: c for c in sel.select(sel.MarketContext(view="moderate_bear", days_to_expiry=25, days_to_target=12))}
    assert bear["Bear Put Spread"].strikes == {"buy PE": "ATM", "sell PE": "slightly OTM"}
    bull = sel.select(sel.MarketContext(view="moderate_bull", days_to_expiry=8, days_to_target=1))
    assert {c.strategy: c for c in bull}["Bull Call Spread"].strikes["buy CE"] == "far OTM"


def test_selector_rejects_unknown_views():
    with pytest.raises(ValueError):
        sel.MarketContext(view="sideways", days_to_expiry=5)
