"""Options toolkit on Kite (O1 step 3): every Kite call mocked, no network."""

import io
import json
from datetime import date, datetime

import pytest

from kite_connect.options import cli
from kite_connect.options.basket_executor import FILLED, NOT_FILLED, NOT_SENT, BasketExecutor, ExecutionReport, LegResult
from kite_connect.options.broker import Broker
from kite_connect.options.instruments import InstrumentResolver, spot_key
from kite_connect.options.live_chain import IST, build_chain, chain_summary, days_to_expiry
from kite_connect.options.options_config import LimitsConfig, OptionsConfig
from kite_connect.options.position_monitor import PositionLedger, evaluate
from kite_connect.options.pretrade import check_limits, limit_price, parse_legs
from kite_connect.options.selector import MarketContext, select
from kite_connect.options.theory import black_scholes

approx = pytest.approx
EXPIRY, NEXT = date(2026, 10, 13), date(2026, 10, 20)
NOW = datetime(2026, 10, 7, 14, 0, tzinfo=IST)
SPOT, IV, RATE, LOT, FREEZE = 25010.0, 0.12, 0.0552, 65, 1800


class FakeKite:
    """The KiteConnect methods the toolkit uses, with orders recorded and fills scripted."""

    def __init__(self, reject=(), stuck=()):
        self.reject, self.stuck = set(reject), set(stuck)
        self.placed, self.cancelled, self.quote_calls = [], [], 0
        self.instruments_dump = []
        for e in (EXPIRY, NEXT):
            for k in range(24700, 25351, 50):
                for t in ("CE", "PE"):
                    self.instruments_dump.append({
                        "instrument_token": len(self.instruments_dump) + 1, "tradingsymbol": f"NIFTY{e:%y%m%d}{k}{t}",
                        "name": "NIFTY", "expiry": e, "strike": float(k), "tick_size": 0.05, "lot_size": LOT,
                        "instrument_type": t, "segment": "NFO-OPT", "exchange": "NFO"})
        self.by_symbol = {i["tradingsymbol"]: i for i in self.instruments_dump}

    def instruments(self, exchange):
        return self.instruments_dump

    def quote(self, keys):
        self.quote_calls += 1
        out = {}
        for key in keys:
            if key == "NSE:NIFTY 50":
                out[key] = {"last_price": SPOT}
                continue
            i = self.by_symbol[key.split(":", 1)[1]]
            px = black_scholes(i["instrument_type"], SPOT, i["strike"], days_to_expiry(i["expiry"], NOW), RATE, IV).price
            px = round(max(px, 0.1) / 0.05) * 0.05
            out[key] = {"last_price": px, "oi": 100000 + i["strike"], "volume": 5000,
                        "depth": {"buy": [{"price": round(px - 0.5, 2)}], "sell": [{"price": round(px + 0.5, 2)}]}}
        return out

    def basket_order_margins(self, orders, consider_positions=True):
        charge = {"transaction_tax": 1.0, "exchange_turnover_charge": 2.0, "sebi_turnover_charge": 0.1,
                  "brokerage": 20.0, "stamp_duty": 0.5, "gst": {"total": 4.0}, "total": 27.6}
        return {"initial": {"total": 90000.0}, "final": {"total": 12000.0}, "orders": [{"charges": charge}] * len(orders)}

    def _post(self, route, url_args=None, params=None):
        assert route == "order.place" and url_args == {"variety": "regular"} and params["autoslice"] == "true"
        if params["tradingsymbol"] in self.reject:
            raise RuntimeError("InputException: rejected by the fake")
        slices, qty = [], params["quantity"]
        while qty > 0:
            q = min(qty, FREEZE)
            oid = f"o{len(self.placed) + 1}"
            self.placed.append({**params, "quantity": q, "order_id": oid})
            slices.append({"order_id": oid})
            qty -= q
        return slices if len(slices) > 1 else slices[0]

    def order_history(self, order_id):
        o = next(p for p in self.placed if p["order_id"] == order_id)
        if order_id in self.cancelled:
            return [{"status": "CANCELLED", "filled_quantity": 0, "average_price": 0, "status_message": "cancelled"}]
        if o["tradingsymbol"] in self.stuck:
            return [{"status": "OPEN", "filled_quantity": 0, "average_price": 0}]
        return [{"status": "COMPLETE", "filled_quantity": o["quantity"], "average_price": o["price"]}]

    def cancel_order(self, variety, order_id):
        self.cancelled.append(order_id)
        return {"order_id": order_id}

    def positions(self):
        return {"net": []}


class Clock:
    def __init__(self):
        self.t, self.slept = 0.0, []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def _broker(tmp_path, kite=None, clock=None):
    clock = clock or Clock()
    return Broker(kite or FakeKite(), order_log=tmp_path / "orders.jsonl", clock=clock, sleep=clock.sleep)


def _report(broker, legs="BUY CE 25000, SELL CE 25150", lots=1, cfg=OptionsConfig()):
    resolver = InstrumentResolver(broker.instruments())
    return cli.prepare(broker, resolver, "NIFTY", EXPIRY, parse_legs(legs), lots, cfg, NOW)


# ---- instruments, chain
def test_resolver_reads_runtime_facts():
    r = InstrumentResolver(__import__("pandas").DataFrame(FakeKite().instruments("NFO")))
    c = r.resolve("NIFTY", EXPIRY, 25000, "CE")
    assert (c.tradingsymbol, c.lot_size, c.tick_size, c.quote_key) == ("NIFTY26101325000CE", LOT, 0.05, "NFO:NIFTY26101325000CE")
    assert r.expiries("NIFTY") == [EXPIRY, NEXT] and r.nearest_expiry("NIFTY", date(2026, 10, 13), min_days=1) == NEXT
    assert r.strikes("NIFTY", EXPIRY)[:2] == [24700.0, 24750.0] and len(r.chain("NIFTY", EXPIRY)) == 28
    with pytest.raises(KeyError):
        r.resolve("NIFTY", EXPIRY, 25025, "CE")
    assert spot_key("NIFTY") == "NSE:NIFTY 50" and spot_key("INFY") == "NSE:INFY"


def test_chain_solves_iv_from_quotes(tmp_path):
    broker = _broker(tmp_path)
    resolver = InstrumentResolver(broker.instruments())
    contracts = resolver.chain("NIFTY", EXPIRY, [24950, 25000, 25050])
    chain = build_chain(contracts, broker.quotes([c.quote_key for c in contracts]), SPOT, NOW, RATE)
    assert chain["iv"].tolist() == approx([IV] * 6, abs=0.01)        # mid = the fake's price
    calls = chain[chain["option_type"] == "CE"].set_index("strike")
    assert 0 < calls.loc[25000, "delta"] < 1 and calls.loc[25000, "theta"] < 0
    s = chain_summary(chain, SPOT)
    assert s["atm_strike"] == 25000 and s["atm_iv"] == approx(IV, abs=0.01) and s["pcr"] > 0
    assert days_to_expiry(EXPIRY, NOW) == approx(6 + 1.5 / 24)


def test_quotes_are_batched_and_throttled(tmp_path):
    clock = Clock()
    broker = _broker(tmp_path, clock=clock)
    keys = [f"NFO:{i['tradingsymbol']}" for i in broker.kite.instruments_dump][:52] * 1
    broker.quotes(keys)
    broker.quotes(keys[:1])
    assert broker.kite.quote_calls == 2 and clock.slept == approx([1.0])  # one request a second


# ---- pre-trade report
def test_legs_and_limit_prices():
    assert parse_legs("BUY CE 25000, SELL 2 CE 25150") == [("BUY", "CE", 25000.0, 1), ("SELL", "CE", 25150.0, 2)]
    with pytest.raises(ValueError):
        parse_legs("HOLD CE 25000")
    assert limit_price(100.0, "BUY", 0.02, 0.05) == 102.0 and limit_price(100.03, "BUY", 0.0, 0.05) == 100.05
    assert limit_price(100.0, "SELL", 0.02, 0.05) == 98.0 and limit_price(0.04, "SELL", 0.02, 0.05) == 0.05


def test_pre_trade_report(tmp_path):
    rep = _report(_broker(tmp_path))
    buy, sell = rep.legs
    assert (buy.side, buy.quantity, sell.side, sell.quantity) == ("BUY", LOT, "SELL", LOT)
    q = FakeKite().quote([buy.contract.quote_key, sell.contract.quote_key])
    assert buy.reference == q[buy.contract.quote_key]["depth"]["sell"][0]["price"]       # ask for a buy
    assert sell.reference == q[sell.contract.quote_key]["depth"]["buy"][0]["price"]      # bid for a sell
    debit = buy.reference - sell.reference
    assert rep.net_premium_inr == approx(debit * LOT)
    assert rep.max_loss_inr == approx(debit * LOT) and rep.max_profit_inr == approx((150 - debit) * LOT)
    assert rep.strategy.breakevens() == approx([25000 + debit])
    g = rep.greeks()
    assert g and 0 < g["delta"] < LOT
    assert set(rep.ranges()) == {1, 2, 3} and rep.ranges()[1][0] < SPOT < rep.ranges()[1][1]
    assert rep.kite_charges().total == approx(2 * 27.6) and rep.cost_model["charges"].total > 0
    assert rep.ok
    text = rep.text()
    for needle in ("PRE-TRADE REPORT", "max loss", "Breakevens", "Net Greeks", "Expected range", "Kite basket margin",
                   "Entry costs (model)", "Entry charges (Kite)", "Limits: all within"):
        assert needle in text, needle


def test_hard_limits(tmp_path):
    broker = _broker(tmp_path)
    naked = _report(broker, "SELL CE 25000")
    assert any("unlimited" in v for v in naked.violations)
    big = _report(broker, lots=11)
    assert any("11 lots" in v for v in big.violations)
    tight = OptionsConfig(limits=LimitsConfig(max_loss_per_trade_inr=1000.0, allowed_underlyings=("BANKNIFTY",)))
    rep = _report(broker, cfg=tight)
    assert len(check_limits(rep, tight.limits)) == 2


# ---- execution
def test_live_buys_first_waits_for_every_slice(tmp_path):
    clock = Clock()
    broker = _broker(tmp_path, clock=clock)
    rep = _report(broker, "SELL CE 25150, BUY CE 25000", lots=30)              # 1,950 units: 2 slices a leg
    out = BasketExecutor(broker, "live", clock=clock, sleep=clock.sleep).execute(rep.legs, tag="t")
    sides = [p["transaction_type"] for p in broker.kite.placed]
    assert sides == ["BUY", "BUY", "SELL", "SELL"]                             # the hedge before the short
    assert [p["quantity"] for p in broker.kite.placed] == [1800, 150, 1800, 150]
    assert out.completed and all(r.status == FILLED and r.filled == 1950 for r in out.results)
    assert len(out.results[0].order_ids) == 2 and not out.naked_shorts()
    log = [json.loads(l) for l in (tmp_path / "orders.jsonl").read_text().splitlines()]
    assert [r["action"] for r in log] == ["place", "place"] and log[0]["response"] == [{"order_id": "o1"}, {"order_id": "o2"}]


def test_live_stops_when_the_hedge_fails(tmp_path):
    kite = FakeKite(reject={"NIFTY26101325000CE"})
    broker = _broker(tmp_path, kite)
    rep = _report(broker)
    out = BasketExecutor(broker, "live").execute(rep.legs)
    assert not out.completed and "BUY NIFTY26101325000CE" in out.stopped
    assert kite.placed == [] and len(out.results) == 1                          # the short was never sent
    assert not out.naked_shorts()


def test_live_cancels_an_unfilled_slice_and_stops(tmp_path):
    clock = Clock()
    kite = FakeKite(stuck={"NIFTY26101325000CE"})
    broker = _broker(tmp_path, kite, clock)
    rep = _report(broker)
    out = BasketExecutor(broker, "live", LimitsConfig(fill_timeout_seconds=5), clock=clock, sleep=clock.sleep).execute(rep.legs)
    assert kite.cancelled == ["o1"] and out.results[0].status == NOT_FILLED and len(out.results) == 1
    assert "cancelled" in out.results[0].message and "STOPPED" in out.text()


def test_naked_short_is_reported():
    class L:
        def __init__(self, side, t):
            self.side, self.quantity = side, 65
            self.contract = type("C", (), {"option_type": t, "tradingsymbol": f"X{t}"})()
    rep = ExecutionReport("live", [LegResult(L("BUY", "CE"), FILLED, 65), LegResult(L("SELL", "PE"), FILLED, 65)])
    assert rep.naked_shorts() == ["65 PE units short without a long PE to cover them"]
    assert "WARNING NAKED SHORT" in rep.text()


def test_paper_and_dry_run_send_nothing(tmp_path):
    broker = _broker(tmp_path)
    rep = _report(broker)
    paper = BasketExecutor(broker, "paper").execute(rep.legs)
    assert paper.completed and paper.results[0].average_price == rep.legs[0].reference   # the touch, inside the limit
    dry = BasketExecutor(broker, "dry_run").execute(rep.legs)
    assert [r.status for r in dry.results] == [NOT_SENT, NOT_SENT] and broker.kite.placed == []
    modes = [json.loads(l)["mode"] for l in (tmp_path / "orders.jsonl").read_text().splitlines()]
    assert modes == ["paper", "paper", "dry_run", "dry_run"]


def test_live_needs_a_terminal_and_the_exact_phrase(tmp_path):
    rep = _report(_broker(tmp_path))

    class TTY(io.StringIO):
        def isatty(self):
            return True
    assert cli.confirm_live(rep, stdin=io.StringIO(), ask=lambda _: "PLACE 2 ORDERS") is False   # CI: no terminal
    assert cli.confirm_live(rep, stdin=TTY(), ask=lambda _: "yes") is False
    assert cli.confirm_live(rep, stdin=TTY(), ask=lambda _: "PLACE 2 ORDERS") is True


def test_a_breach_is_refused_in_every_mode(tmp_path):
    broker = _broker(tmp_path)
    rep = _report(broker, "SELL CE 25000")
    ledger = PositionLedger(tmp_path / "positions.json")
    for mode in ("paper", "live"):
        assert cli.run_trade(broker, rep, mode, OptionsConfig(), NOW, ledger) is None
    assert broker.kite.placed == [] and ledger.positions == []


# ---- ledger and monitor
def test_monitor_alerts(tmp_path):
    broker = _broker(tmp_path)
    rep = _report(broker)
    ledger = PositionLedger(tmp_path / "positions.json")
    out = cli.run_trade(broker, rep, "paper", OptionsConfig(), NOW, ledger)
    pos = PositionLedger(tmp_path / "positions.json").open_positions()[0]           # survives a reload
    assert out.completed and pos.mode == "paper" and [l.quantity for l in pos.legs] == [LOT, LOT]
    entry = {l.tradingsymbol: l.price for l in pos.legs}
    calm = evaluate(pos, entry, SPOT, NOW, RATE)
    assert calm.pnl_inr == 0 and calm.greeks and not any("stop-loss" in a for a in calm.alerts)
    assert not any("breakeven" in a for a in calm.alerts)       # a debit spread starts below its breakeven: no alarm
    later = datetime(2026, 10, 9, 14, 0, tzinfo=IST)
    crash = evaluate(pos, {k: 0.5 for k in entry}, 24600.0, later, RATE)
    assert any("volatility stop-loss" in a for a in crash.alerts)
    assert any("of the maximum" in a for a in crash.alerts) and crash.pnl_inr < 0
    credit = cli.run_trade(broker, _report(broker, "SELL CE 25100, BUY CE 25250"), "paper", OptionsConfig(), NOW, ledger)
    bear = PositionLedger(tmp_path / "positions.json").open_positions()[1]
    assert credit.completed and bear.strategy().payoff(SPOT) > 0                # entered on the profit side
    prices = {l.tradingsymbol: l.price for l in bear.legs}
    assert any("near" not in a and "crossed a breakeven" in a for a in evaluate(bear, prices, 25300.0, NOW, RATE).alerts)
    near = bear.strategy().breakevens()[0] - 50
    assert any("within" in a for a in evaluate(bear, prices, near, NOW, RATE).alerts)


def test_demo_spread_from_the_selector():
    bull = select(MarketContext(view="moderate_bull", days_to_expiry=6))[0]
    strikes = [float(k) for k in range(24700, 25351, 50)]
    assert bull.strategy == "Bull Call Spread"
    assert cli.vertical_spec(bull, SPOT, strikes, OptionsConfig().selector) == [("BUY", "CE", 25000.0, 1), ("SELL", "CE", 25150.0, 1)]
    bear = next(c for c in select(MarketContext(view="moderate_bear", days_to_expiry=6)) if c.strategy == "Bear Put Spread")
    (b_side, _, hi, _), (s_side, _, lo, _) = cli.vertical_spec(bear, SPOT, strikes, OptionsConfig().selector)
    assert (b_side, s_side) == ("BUY", "SELL") and hi > lo
