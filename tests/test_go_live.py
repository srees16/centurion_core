"""Going live and staying live: the capital ladder and its gap band, the go-live checks and what the web page
shows before them, the paper record a live start needs (V5), and the fills and coverage the live ledger records."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from nse_engine.costs import impact_bps, statutory_cost


# ---- the go-live ramp (GT2I)
def _ladder(rung=0, sessions=100):
    from nse_engine.capital_ladder import LadderState

    return LadderState(rung=rung, since="2027-01-01", sessions=sessions, last_session="2027-06-01")


def _ladder_gate(costs="PASS"):
    names = ("tracking error", "daily gap", "drawdown", "regime break")
    return {"checks": [{"name": n, "status": "PASS"} for n in names] + [{"name": "costs", "status": costs}]}


def _ladder_eval(state, gate=None, ramp=None, requested=None):
    from nse_engine.capital_ladder import evaluate

    return evaluate(state, "2027-06-02", gate=gate or _ladder_gate(), drawdown_state="normal", book_dd=0.02,
                    nifty_dd=0.02, shift_multipliers=[], requested_capital=requested, ramp=ramp)


def _ramp(alarm=False, low=-1.0, volatility=True, turnover=True):
    return {"cusum": {"low": low, "alarm": alarm, "sessions": 40},
            "volatility": {"value": 1.0 if volatility else 1.6, "ok": volatility, "display": "1.00x"},
            "turnover": {"value": 1.0 if turnover else 0.4, "ok": turnover, "display": "1.00x"}}


def test_the_gap_band_reads_at_its_checkpoints_and_holds_only_from_120():
    """MON1 (plan 5bu): G(n) < -2 sigma sqrt(n) at 60 / 120 / 250 / every 250; WATCH at 60, a ladder hold from 120
    until the next checkpoint or the owner's acknowledgement; nothing before sigma is fixed or before 60."""
    from nse_engine import live_monitor as lm

    assert [lm.checkpoint_for(n) for n in (59, 60, 119, 120, 249, 250, 499, 500, 760)] == \
        [None, 60, 60, 120, 120, 250, 250, 500, 750]
    s = 0.003
    leak = lambda n, per: pd.Series([per] * n)            # noqa: E731 - a steady daily gap
    assert lm.gap_band(leak(80, -0.01), None)["status"] == "not armed"
    assert lm.gap_band(leak(59, -0.01), s)["status"] == "waiting"
    band60 = -2 * s * 60 ** 0.5                            # -4.6%: a -0.1%/day leak is -6% at 60
    ok = lm.gap_band(leak(70, band60 / 60 * 0.9), s)
    assert ok["status"] == "OK" and not ok["hold"] and ok["checkpoint"] == 60
    watch = lm.gap_band(leak(70, -0.001), s)
    assert watch["status"] == "WATCH" and not watch["hold"]
    breach = lm.gap_band(leak(130, -0.001), s)
    assert breach["status"] == "BREACH" and breach["hold"] and "ladder holds" in breach["line"]
    assert breach["cum_gap"] == pytest.approx(-0.12)       # read at 120, not at 130
    assert not lm.gap_band(leak(130, -0.001), s, ack="120")["hold"]
    assert lm.gap_band(leak(130, -0.001), s, ack="60")["hold"], "an acknowledgement covers its own checkpoint"

    d = _ladder_eval(_ladder(sessions=59), ramp={**_ramp(), "gap_band": breach}, requested=600_000)
    assert d.action != "GO" and any("Gap band" in r for r in d.reasons) and any("Gap band" in a for a in d.alerts)
    d = _ladder_eval(_ladder(sessions=59), ramp={**_ramp(), "gap_band": watch}, requested=600_000)
    assert d.action == "GO" and any("WATCH" in a for a in d.alerts), "a WATCH at 60 informs, never holds"


def test_the_gap_sigma_is_fixed_once_at_ra1(tmp_path):
    from nse_engine import live_monitor as lm

    assert lm.sigma_from_te(0.08) == pytest.approx(0.08 / 252 ** 0.5)
    assert lm.sigma_from_te(0.001) == lm.SIGMA_FLOOR
    assert lm.te_from_gate_summary({"checks": [{"name": "tracking error", "display": "6.2%/yr"}]}) == pytest.approx(0.062)
    for bad in ({"checks": [{"name": "tracking error", "display": "n/a"}]}, {"checks": []}):
        with pytest.raises(ValueError):
            lm.te_from_gate_summary(bad)
    path = tmp_path / "sigma.json"
    with pytest.raises(RuntimeError, match="fixed at 60"):
        lm.fix_sigma(0.05, "test", 59, path=path)
    doc = lm.fix_sigma(0.05, "test", 60, path=path)
    assert lm.load_sigma(path) == pytest.approx(doc["sigma_daily"])
    with pytest.raises(RuntimeError, match="fixed once"):
        lm.fix_sigma(0.06, "test", 61, path=path)
    assert lm.load_sigma(tmp_path / "none.json") is None


def test_the_monthly_rank_ic_scores_complete_months_only():
    """MON1: Spearman of the forecast against the next 21 sessions, per finished month, read at 12 months."""
    from nse_engine import live_monitor as lm

    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2027-01-01", periods=120)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (120, 60)), axis=0))
    universe = np.ones_like(close, dtype=bool)
    p = int(dates.get_loc(lm.month_ends(dates, dates[0])[0]))
    perfect = np.zeros_like(close)
    perfect[p] = close[p + 21] / close[p] - 1.0             # a forecast that knew the next 21 sessions
    assert lm.rank_ic(perfect, universe, close, p) == pytest.approx(1.0)
    assert lm.rank_ic(perfect, universe, close, 110) is None, "horizon not complete"
    ends = lm.month_ends(dates, dates[0])
    assert all(e.month != dates[-1].month for e in ends), "a running month has no month-end yet"
    done = {ends[0].date().isoformat(): 0.1}
    due = lm.due_ics(dates, dates[0], done)
    assert ends[0] not in due and all(int(dates.get_loc(m)) + 21 < len(dates) for m in due)
    s = lm.ic_summary({f"2027-{m:02d}-28": 0.05 + 0.01 * (m % 3) for m in range(1, 13)})
    assert s["months"] == 12 and s["t"] > 2 and s["verdict"].startswith("supports")
    assert lm.ic_summary({"2027-01-29": 0.05})["verdict"] == "read at 12 months (1 so far)"
    assert lm.ic_summary({f"2027-{m:02d}-28": -0.02 + 0.001 * m for m in range(1, 13)})["verdict"].startswith("REVIEW")


def test_the_canary_needs_three_months_before_go():
    held = _ladder(sessions=58)                    # evaluate counts tonight's session
    assert "59 of 60 sessions" in _ladder_eval(held, requested=600_000).reasons[0]
    assert held.rung == 0
    moved = _ladder(sessions=59)
    d = _ladder_eval(moved, ramp=_ramp(), requested=600_000)
    assert d.action == "GO" and d.capital == 600_000 and moved.rung == 1


@pytest.mark.parametrize("gate,ramp,why", [
    (_ladder_gate("n/a"), _ramp(), "costs"),
    (_ladder_gate("WATCH"), _ramp(), "costs"),
    (_ladder_gate(), _ramp(volatility=False), "volatility"),
    (_ladder_gate(), _ramp(turnover=False), "turnover"),
])
def test_go_needs_measured_fills_and_behaviour(gate, ramp, why):
    state = _ladder(rung=1)
    d = _ladder_eval(state, gate=gate, ramp=ramp, requested=1_200_000)
    assert d.action == "HOLD" and state.rung == 1
    assert why in d.alerts[0]


def test_the_cusum_alarm_steps_down_one_rung_and_alerts():
    state = _ladder(rung=2)
    d = _ladder_eval(state, ramp=_ramp(alarm=True, low=-12.5))
    assert d.action == "STEP DOWN" and d.capital == 600_000 and "CUSUM" in d.reasons[0]
    assert d.critical


def test_the_cusum_alarm_at_the_canary_alerts_and_keeps_the_rung():
    state = _ladder(rung=0)
    d = _ladder_eval(state, ramp=_ramp(alarm=True, low=-12.5))
    assert state.rung == 0 and d.flow == 0.0 and "kill switch" in d.critical[0]


def test_gap_cusum_catches_a_break_and_ignores_noise():
    from nse_engine.capital_ladder import gap_cusum

    rng = np.random.default_rng(3)
    noise = pd.Series(rng.normal(0.0, 0.0025, 120))
    assert not gap_cusum(noise)["alarm"]
    assert gap_cusum(noise - 0.0025)["alarm"]
    assert gap_cusum(pd.Series(-0.01, index=range(19)))["sessions"] == 19      # too few to read
    assert not gap_cusum(noise.head(19))["alarm"]
    assert not gap_cusum(pd.Series(0.0, index=range(40)))["alarm"]             # identical to the backtest


def test_ramp_checks_measure_fills_against_the_backtest():
    from nse_engine.capital_ladder import ramp_checks

    days = pd.bdate_range("2027-01-04", periods=30)
    ref = pd.Series(np.random.default_rng(1).normal(0.0005, 0.01, 30), index=days)
    equity = 300_000 * (1 + ref).cumprod()
    fills = pd.DataFrame({"session_date": [days[5].date().isoformat()] * 2, "status": ["FILLED", "CANCELLED"],
                          "quantity": [100, 100], "fill_price": [100.0, 100.0], "costs_inr": [5.0, 0.0],
                          "impact_bps": [2.0, 0.0]})
    trades = pd.DataFrame({"date": [days[5], days[6]], "value_inr": [15_000.0, 5_000.0]})
    out = ramp_checks(equity, ref, fills, trades)
    assert out["volatility"]["ok"] and not out["cusum"]["alarm"]
    assert out["turnover"]["value"] == pytest.approx(0.5) and not out["turnover"]["ok"]      # half the trading
    assert ramp_checks(equity, ref, None, trades)["turnover"]["display"] == "not measurable"


def test_go_live_refuses_any_start_but_the_canary():
    from nse_engine.capital_ladder import readiness

    assert dict((n, ok) for n, ok, _ in readiness({}, [], capital=300_000))["canary"]
    assert not dict((n, ok) for n, ok, _ in readiness({}, [], capital=600_000))["canary"]
    assert "canary" not in dict((n, ok) for n, ok, _ in readiness({}, []))


# ---- a book no long-only account can hold
@pytest.mark.parametrize("weights,why", [({"A": float("nan")}, "not finite"), ({"A": float("inf")}, "not finite"),
                                         ({"A": -0.01, "B": 0.5}, "negative"), ({"A": 0.7, "B": 0.4}, "above max_gross")])
def test_a_book_no_account_can_hold_is_refused(weights, why):
    from nse_engine.portfolio import check_book_weights

    with pytest.raises(ValueError, match=why):
        check_book_weights(weights, 1.0)


def test_a_full_or_empty_book_passes():
    from nse_engine.portfolio import check_book_weights

    check_book_weights({}, 1.0)
    check_book_weights({"A": 0.6, "B": 0.4 + 1e-12}, 1.0)


# ---- what the web page shows before go-live (Trade Center > Paper Validation)
class _StatusBook:
    def __init__(self, state):
        self.state = state

    def read_state(self):
        return self.state


def _go_live_state(**extra):
    from nse_engine.capital_ladder import DRY_RUNS_KEY, SELL_PATH_KEY

    sell = {"user_id": "AB1234", "ddpi_confirmed_on": "2026-11-02", "amo_status": "COMPLETE", "amo_side": "SELL",
            "gtt_status": "triggered", "gtt_order_status": "COMPLETE", "verified_at": "2026-11-03"}
    return {DRY_RUNS_KEY: json.dumps([{"session": str(i), "clean": True} for i in range(5)]),
            SELL_PATH_KEY: json.dumps(sell), **extra}


def _go_live(monkeypatch, state, gate):
    from kite_connect.trading import live_session as ls

    monkeypatch.delenv("CENTURION_LIVE_CAPITAL", raising=False)
    return ls.go_live_status(_StatusBook(state), _StatusBook({"paper_gate": json.dumps(gate)}), trial_gates={})


def test_the_live_book_ensures_its_tables_once_per_process(monkeypatch):
    """KL1: ensure_tables (about 15 statements) ran on every live_book() call, so the Kite login and the Kite page
    waited seconds per request.  Now once per schema and process, and only once a read confirms the store answered;
    a first call during an outage is retried on the next."""
    from kite_connect.trading import live_session as ls

    made = []

    class Book:
        def __init__(self, mgr, schema=None):
            self.schema, self.ensured = schema, 0
            made.append(self)

        def ensure_tables(self):
            self.ensured += 1

        def read_state(self):
            if down["now"]:
                raise ConnectionError("store down")
            return {}

    down = {"now": True}
    monkeypatch.setenv("CENTURION_DATABASE_URL", "postgresql://test")
    monkeypatch.setattr("database.connection.get_db_manager", lambda: "mgr")
    monkeypatch.setattr("database.paper_cloud.PaperCloudSync", Book)
    monkeypatch.setattr(ls, "_LIVE_BOOKS", {})
    ls.live_book()
    down["now"] = False
    first, again = ls.live_book(), ls.live_book()
    assert len(made) == 2 and first is again, "the outage call was not kept; the next was, and is reused"
    assert ls.live_book("live_family") is not first and len(made) == 3
    assert [b.ensured for b in made] == [1, 1, 1]


def test_erasing_an_account_forgets_its_cached_live_book(monkeypatch):
    """erase drops the account's schema, but KL1's cache still said its tables were ensured: a holder re-registering
    under the same name (same id, same schema) would write to tables that no longer exist until a restart."""
    from dataclasses import asdict
    from kite_connect.auth import accounts
    from kite_connect.trading import live_session as ls

    class Book:
        def __init__(self, mgr, schema=None):
            self.schema = schema

        def ensure_tables(self):
            pass

        def read_state(self):
            return {}

    class Registry:
        def __init__(self):
            self.state = {accounts.REGISTRY_KEY: json.dumps({"ravi": asdict(accounts.Account("ravi", "Ravi", "AB1234", "k"))})}

        def read_state(self):
            return self.state

        def sync_state(self, kv):
            self.state.update(kv)
            return True

    dropped = []

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, stmt):
            dropped.append(str(stmt))

    class Manager:
        def get_session(self):
            return Session()

    monkeypatch.setenv("CENTURION_DATABASE_URL", "postgresql://test")
    monkeypatch.setattr("database.connection.get_db_manager", lambda: Manager())
    monkeypatch.setattr("database.paper_cloud.PaperCloudSync", Book)
    monkeypatch.setattr(ls, "_LIVE_BOOKS", {})
    ls._LIVE_BOOKS.update({("m", "live_ravi"): Book(None, "live_ravi"), ("m", "live_other"): Book(None, "live_other")})
    accounts.erase("ravi", book=Registry())
    assert dropped == ['DROP SCHEMA IF EXISTS "live_ravi" CASCADE']
    assert [k[1] for k in ls._LIVE_BOOKS] == ["live_other"], "only the dropped schema is forgotten"


@pytest.mark.parametrize("capital", [None, 0, -300000.0, float("nan"), float("inf"), float("-inf")])
def test_a_first_ledger_needs_a_positive_finite_capital(capital):
    """--capital is parsed with float(), which takes "nan" and "inf"; NaN passed `not capital or capital <= 0`, so the
    first real session stored a ledger with NaN capital and cash in Neon, read back by every later session."""
    from kite_connect.trading.live_session import load_ledger

    with pytest.raises(RuntimeError, match="give the book's capital"):
        load_ledger({}, capital)


def _fill(sym, side, qty, px, status="FILLED"):
    return {"status": status, "symbol": sym, "side": side, "quantity": qty, "fill_price": px, "order_id": "o1"}


@pytest.mark.parametrize("px", [0.0, -1512.5, float("nan"), float("inf"), None])
def test_a_fill_without_a_real_price_is_refused_and_nothing_is_booked(px):
    """A FILLED row at price 0 added the shares at no cost (cash unchanged); at NaN the ledger's cash became NaN,
    stored in Neon for every later session.  Both now stop before the ledger is touched (vectorbt's pre-checks)."""
    from kite_connect.trading.live_session import apply_fills

    led = {"capital": 3e5, "cash": 3e5, "positions": {"TCS": 5}, "symbols": ["TCS"]}
    with pytest.raises(ValueError, match="INFY BUY x10 .* nothing is recorded"):
        apply_fills(led, [_fill("TCS", "SELL", 5, 4000.0), _fill("INFY", "BUY", 10, px)], "2026-11-06", 15.93)
    assert led == {"capital": 3e5, "cash": 3e5, "positions": {"TCS": 5}, "symbols": ["TCS"]}


def test_unfilled_rows_need_no_price():
    from kite_connect.trading.live_session import apply_fills

    led = {"capital": 3e5, "cash": 3e5, "positions": {}, "symbols": []}
    out = apply_fills(led, [_fill("INFY", "BUY", 10, 0.0, status="REJECTED"), _fill("TCS", "BUY", 0, None)],
                      "2026-11-06", 15.93)
    assert (out["cash"], out["positions"], out["session"]) == (3e5, {}, "2026-11-06")


def test_fills_book_the_known_cash_and_cap_a_sale_at_the_holding():
    from kite_connect.trading.live_session import apply_fills
    from nse_engine.costs import statutory_cost

    day = pd.Timestamp("2026-11-06")
    led = {"capital": 3e5, "cash": 3e5, "positions": {"TCS": 5}, "symbols": ["TCS"]}
    out = apply_fills(led, [_fill("INFY", "BUY", 10, 1512.5), _fill("TCS", "SELL", 8, 4000.0)], day, 15.93)
    want = (3e5 - 15125.0 - statutory_cost(15125.0, "BUY", day, 15.93, symbol="INFY")
            + 20000.0 - statutory_cost(20000.0, "SELL", day, 15.93, symbol="TCS"))  # 5 held: 5 sold, not 8
    assert out["cash"] == pytest.approx(want, abs=1e-6) and out["positions"] == {"INFY": 10}


def test_random_valid_fills_keep_the_ledger_finite_and_the_cash_identity():
    """Property check (seeded, 300 sessions of random valid fills): cash stays finite, positions whole and non-negative,
    and cash = start - what the buys cost + what the sells raised, sells capped at the holding."""
    from kite_connect.trading.live_session import apply_fills
    from nse_engine.costs import statutory_cost

    rng = np.random.default_rng(20261106)
    led = {"capital": 3e5, "cash": 3e5, "positions": {}, "symbols": []}
    cash = 3e5
    days = pd.bdate_range("2025-01-01", periods=300)
    for day in days:
        fills, held = [], dict(led["positions"])
        for _ in range(int(rng.integers(0, 4))):
            sym, side = str(rng.choice(["INFY", "TCS", "SBIN"])), str(rng.choice(["BUY", "SELL"]))
            qty, px = int(rng.integers(1, 50)), float(np.round(rng.uniform(1.0, 20000.0), 2))
            fills.append(_fill(sym, side, qty, px))
            q = qty if side == "BUY" else min(qty, held.get(sym, 0))
            if q > 0:
                held[sym] = held.get(sym, 0) + (q if side == "BUY" else -q)
                value = q * px
                cash += (-value - statutory_cost(value, "BUY", day, 15.93, symbol=sym) if side == "BUY"
                         else value - statutory_cost(value, "SELL", day, 15.93, symbol=sym))
        led = apply_fills(led, fills, day, 15.93)
        assert math.isfinite(led["cash"]) and all(isinstance(q, int) and q > 0 for q in led["positions"].values())
    assert led["cash"] == pytest.approx(cash, abs=1e-4)


@pytest.mark.parametrize("item", ["INFY=40@nan", "INFY=40@0", "INFY=40@-5", "INFY=-1@1512.5", "INFY=40@inf"])
def test_reconcile_refuses_a_price_or_quantity_that_cannot_be_booked(item, capsys):
    """--reconcile INFY=40@nan wrote a NaN cash straight to Neon; refused now, before any book is read."""
    from kite_connect.trading import live_session as ls

    with pytest.raises(SystemExit) as exit_:
        ls.main(["--reconcile", item])
    assert exit_.value.code == 2 and "PX a positive price" in capsys.readouterr().err


def test_a_first_ledger_starts_with_its_capital_in_cash():
    from kite_connect.trading.live_session import load_ledger

    led = load_ledger({}, 300000.0)
    assert (led["capital"], led["cash"], led["positions"]) == (300000.0, 300000.0, {})


def test_go_live_status_says_ready_only_when_every_real_check_passes(monkeypatch):
    done = _go_live(monkeypatch, _go_live_state(), {"verdict": "PASS", "sessions": 60})
    assert done["ready"] and not done["live"] and done["ladder"] is None
    assert [c["name"] for c in done["checks"]] == ["paper book G4", "live dry runs", "sell path"]
    early = _go_live(monkeypatch, _go_live_state(), {"verdict": "PASS", "sessions": 59})
    assert not early["ready"] and not early["checks"][0]["ok"]              # a fine Sharpe is not a check


def test_go_live_status_holds_a_start_above_the_canary(monkeypatch):
    from kite_connect.trading import live_session as ls

    monkeypatch.setenv("CENTURION_LIVE_CAPITAL", "600000")
    got = ls.go_live_status(_StatusBook(_go_live_state()),
                            _StatusBook({"paper_gate": json.dumps({"verdict": "PASS", "sessions": 60})}), trial_gates={})
    assert not got["ready"] and got["checks"][-1]["name"] == "canary"


def test_go_live_status_shows_the_rung_once_live(monkeypatch):
    from kite_connect.trading.live_session import LIVE_LEDGER_KEY
    from nse_engine.capital_ladder import STATE_KEY, LadderState

    state = _go_live_state(**{STATE_KEY: LadderState(rung=0, since="2026-12-15", sessions=12).dump(), LIVE_LEDGER_KEY: "{}"})
    got = _go_live(monkeypatch, state, {"verdict": "PASS", "sessions": 60})
    assert got["live"]
    assert got["ladder"] == {"rung": 1, "rungs": 5, "capital": 300_000.0, "sessions": 12, "sessions_needed": 60,
                             "since": "2026-12-15"}
    assert got["monitors"] == {"gap_band": None, "rank_ic": None}, "live, no reading yet"


def test_go_live_status_shows_the_live_monitors(monkeypatch):
    """MON1 on the web page: the gap band the last live session stored with its G4 summary, and the rank IC."""
    from kite_connect.trading.live_session import LIVE_LEDGER_KEY
    from nse_engine import live_monitor as lm
    from nse_engine.paper_gate import STATE_KEY as G4

    assert _go_live(monkeypatch, _go_live_state(), {"verdict": "PASS", "sessions": 60})["monitors"] is None
    band = lm.gap_band(pd.Series([-0.001] * 130), 0.003)
    state = _go_live_state(**{LIVE_LEDGER_KEY: "{}", G4: json.dumps({"verdict": "PASS", "gap_band": band}),
                              lm.IC_STATE_KEY: json.dumps({"2027-01-29": 0.06})})
    got = _go_live(monkeypatch, state, {"verdict": "PASS", "sessions": 60})["monitors"]
    assert got["gap_band"]["status"] == "BREACH" and got["gap_band"]["hold"] and got["gap_band"]["checkpoint"] == 120
    assert got["rank_ic"]["months"] == 1 and got["rank_ic"]["mean"] == pytest.approx(0.06)
    assert got["rank_ic"]["t"] is None, "a NaN t is sent as null, never as NaN JSON"
    json.dumps(got, allow_nan=False)


def test_control_coverage_counts_only_what_the_paper_record_ran():
    from nse_engine.capital_ladder import control_coverage

    got = control_coverage(["REBALANCE", "EXIT:RANK_EXIT", "GTT_SL", "GTT_SL_GAP", "EXIT:STOP", "EXIT:SLEEVE_TREND_EXIT",
                            None, float("nan")], ["normal", "half", "risk_off", None])
    assert {r["path"]: r["count"] for r in got} == {"rebalance": 1, "rank exit": 1, "stop": 3, "sleeve exit": 1,
                                                    "regime exit": 0, "drawdown rule": 2}
    assert [r["count"] for r in control_coverage(["exit:regime"], [])][4] == 1


class _RecordBook(_StatusBook):
    def __init__(self, state, reasons, drawdown):
        super().__init__(state)
        self.positions, self.sessions = pd.DataFrame(reasons), pd.DataFrame(drawdown)

    def read_positions(self):
        return self.positions

    def read_sessions(self):
        return self.sessions


def test_coverage_informs_the_go_live_page_and_never_moves_ready(monkeypatch):
    from kite_connect.trading import live_session as ls

    monkeypatch.delenv("CENTURION_LIVE_CAPITAL", raising=False)
    gate = {"paper_gate": json.dumps({"verdict": "PASS", "sessions": 60})}
    quiet = _RecordBook(gate, {"exit_reason": ["REBALANCE"]}, {"drawdown_state": ["normal"]})
    got = ls.go_live_status(_StatusBook(_go_live_state()), quiet, trial_gates={})
    assert got["ready"]                                                     # a control at 0 holds nothing back ...
    assert {r["path"]: r["count"] for r in got["coverage"]}["regime exit"] == 0    # ... and the page says so


@pytest.mark.parametrize("book", [_StatusBook({}), _RecordBook({}, {}, {}), _RecordBook({}, {"exit_reason": []}, {})])
def test_coverage_stays_silent_when_no_record_can_be_read(book):
    from kite_connect.trading import live_session as ls

    assert ls._control_coverage(book) is None              # an unread record is not every control at 0


@pytest.mark.lesson("L-V5", pattern="F1", date="2026-10-04", source="V5",
                    summary="A promotion restarted the go-live clock: go-live read only the deployed book's own paper count")
def test_go_live_reads_the_paper_record_of_the_configuration_that_will_trade():
    from nse_engine.capital_ladder import GO_LIVE_MIN_PAPER_SESSIONS as N, go_live_evidence

    own = {"config_hash": "new", "sessions": 3, "book_last": "2026-12-24"}       # just promoted: its count restarted
    trial = {"config_hash": "new", "sessions": N, "book_last": "2026-12-24", "verdict": "PASS"}
    gate, source = go_live_evidence(own, "new", {"e4": trial})
    assert gate is trial and "e4" in source
    for spoiled in ({"verdict": "FAIL"}, {"sessions": N - 1}, {"book_last": "2026-12-01"}, {"config_hash": "other"}):
        assert go_live_evidence(own, "new", {"e4": {**trial, **spoiled}}) == (own, "deployed paper book"), spoiled
    old = {"config_hash": "old", "sessions": N + 30, "verdict": "PASS"}
    assert go_live_evidence(old, "new", {})[0] is None                    # an old configuration's record never counts
    full = {"config_hash": "new", "sessions": N}
    assert go_live_evidence(full, "new", {"e4": trial}) == (full, "deployed paper book")
