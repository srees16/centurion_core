"""F&O charges, slippage and the toolkit config (kite_connect/options/fno_costs.py, options_config.py)."""

import pytest

from kite_connect.options import fno_costs as fc
from kite_connect.options.options_config import ChargesConfig, OptionsConfig, load_config
from kite_connect.options.strategies import BullCallSpread
from kite_connect.options.theory import BUY, CALL, PUT, SELL

approx = pytest.approx
CH = ChargesConfig()


@pytest.mark.parametrize("date,rate", [
    ("2015-03-02", 0.00017), ("2016-05-31", 0.00017), ("2016-06-01", 0.0005), ("2023-04-01", 0.000625),
    ("2024-09-30", 0.000625), ("2024-10-01", 0.001), ("2026-03-31", 0.001), ("2026-04-01", 0.0015),
])
def test_option_stt_schedule(date, rate):
    assert fc.rate_on(CH.stt_option_sell, date) == rate


def test_futures_stt_schedule_and_bounds():
    assert [fc.rate_on(CH.stt_futures_sell, d) for d in ("2013-05-31", "2013-06-01", "2024-10-01", "2026-04-01")] == \
        [0.00017, 0.0001, 0.0002, 0.0005]
    with pytest.raises(ValueError):
        fc.rate_on(CH.stt_option_sell, "2001-01-01")


def test_option_sell_charges_2026():
    c = fc.option_charges(100 * 65, SELL, "2026-10-05")
    assert c.brokerage == 20 and c.stt == approx(9.75) and c.stamp_duty == 0
    assert c.exchange == approx(6500 * 0.0003503) and c.sebi == approx(6500 * 1e-6)
    assert c.gst == approx(0.18 * (20 + c.exchange + c.sebi))
    assert c.total == approx(20 + 9.75 + 2.27695 + 0.0065 + 0.18 * (20 + 2.27695 + 0.0065))


def test_option_buy_charges_and_slices():
    c = fc.option_charges(6500, BUY, "2026-10-05", orders=3)
    assert c.stt == 0 and c.stamp_duty == approx(6500 * 0.00003) and c.brokerage == 60
    old = fc.option_charges(6500, BUY, "2016-01-04")
    assert old.exchange == approx(6500 * 0.0005) and old.gst == approx(0.15 * (20 + old.exchange + old.sebi))
    assert fc.option_charges(0, BUY, "2026-10-05").total == 0


def test_futures_charges_brokerage_cap():
    small = fc.futures_charges(50_000, BUY, "2026-10-05")
    assert small.brokerage == approx(15.0)                       # 0.03% of 50,000 < Rs 20
    big = fc.futures_charges(1_600_000, SELL, "2026-10-05")
    assert big.brokerage == 20 and big.stt == approx(800)


def test_exercise_stt_trap_and_intrinsic():
    assert fc.exercise_stt(CALL, 8000, 8100, 75, "2018-06-28") == approx(0.00125 * 8100 * 75)   # full value
    assert fc.exercise_stt(CALL, 8000, 8100, 75, "2020-06-25") == approx(0.00125 * 100 * 75)    # intrinsic
    assert fc.exercise_stt(PUT, 8000, 7900, 65, "2026-10-06") == approx(0.0015 * 100 * 65)
    assert fc.exercise_stt(CALL, 8000, 7900, 65, "2026-10-06") == 0                           # OTM


def test_slippage_and_fill_price():
    assert fc.slippage_per_unit(100, "NIFTY") == approx(0.5)
    assert fc.slippage_per_unit(2, "NIFTY") == approx(0.05)                                  # one tick floor
    assert fc.slippage_per_unit(20, "INFY") == approx(0.4)
    assert fc.fill_price(100, BUY, "NIFTY") == approx(100.5) and fc.fill_price(100, SELL, "NIFTY") == approx(99.5)
    assert fc.fill_price(0.05, SELL, "NIFTY") == approx(0.05)


def test_legs_costs_open_and_close():
    s = BullCallSpread(7700, 296, 7800, 227)
    opened = fc.legs_costs(s.legs, 65, "2026-10-05", "NIFTY")
    assert [r["side"] for r in opened["rows"]] == [BUY, SELL]
    expected = fc.option_charges(296 * 65, BUY, "2026-10-05") + fc.option_charges(227 * 65, SELL, "2026-10-05")
    assert opened["charges"].total == approx(expected.total)
    assert opened["slippage_inr"] == approx((296 + 227) * 0.005 * 65)
    closed = fc.legs_costs(s.legs, 65, "2026-10-05", "NIFTY", closing=True, orders_per_leg={0: 2})
    assert [r["side"] for r in closed["rows"]] == [SELL, BUY] and closed["rows"][0]["orders"] == 2
    assert fc.expiry_costs(s.legs, 65, 7850, "2026-10-06") == approx(0.0015 * 150 * 65)       # long 7700 CE only


def test_charges_from_kite_response():
    orders = [{"charges": {"transaction_tax": 9.75, "exchange_turnover_charge": 2.28, "sebi_turnover_charge": 0.01,
                           "brokerage": 20, "stamp_duty": 0, "gst": {"igst": 4.01, "total": 4.01}, "total": 36.05}},
              {"charges": {"transaction_tax": 0, "exchange_turnover_charge": 1.0, "sebi_turnover_charge": 0.0,
                           "brokerage": 20, "stamp_duty": 0.2, "gst": {"total": 3.78}}}]
    c = fc.charges_from_kite(orders)
    assert (c.stt, c.brokerage, c.stamp_duty) == (9.75, 40, 0.2) and c.total == approx(36.05 + 24.98)


def test_load_config_overrides(tmp_path):
    path = tmp_path / "options.toml"
    path.write_text('[limits]\nmax_lots_per_leg = 4\nallowed_underlyings = ["NIFTY"]\n'
                    '[charges]\nstt_option_sell = [["2008-06-01", 0.001]]\n')
    cfg = load_config(str(path))
    assert cfg.limits.max_lots_per_leg == 4 and cfg.limits.allowed_underlyings == ("NIFTY",)
    assert fc.rate_on(cfg.charges.stt_option_sell, "2026-10-05") == 0.001
    assert cfg.market == OptionsConfig().market
    path.write_text("[limits]\nmax_lots = 4\n")
    with pytest.raises(ValueError):
        load_config(str(path))


def test_default_config_without_env(monkeypatch):
    monkeypatch.delenv("CENTURION_OPTIONS_CONFIG", raising=False)
    assert load_config() == OptionsConfig()
