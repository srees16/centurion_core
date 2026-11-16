"""The paper book: which rows belong to it, what a session records, who may write it.

Regression tests for three fixes that were each found in production:
  U10  fills stamped at the session open predate a book started later that day
  G5   execution costs must outlive the runner's local SQLite
  G7   only one runner may write the book
and one found by audit (RB2): a book write that fails must fail the run, not leave it half-written in silence.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

import database.paper_cloud as pc


def _row(symbol, opened, is_open=True, closed=""):
    return {"symbol": symbol, "side": "BUY", "quantity": 1, "entry_price": 100.0, "stop_loss": 90.0,
            "target_price": 0.0, "opened_at": opened, "closed_at": closed, "exit_price": 0.0,
            "exit_reason": "", "pnl": 0.0, "pnl_pct": 0.0, "is_open": is_open}


def _cloud(state, rows):
    obj = pc.PaperCloudSync.__new__(pc.PaperCloudSync)
    obj.read_state = lambda: dict(state)
    df = pd.DataFrame(rows)
    obj._read = lambda sql: df.copy()
    return obj


class TestBookMembership:
    """U10: a book started mid-day still owns that morning's fills."""

    STATE = {"epoch": "2026-09-17T05:16:25+00:00", "cash": "50000", "initial_capital": "3500000"}
    OLD = [_row("OLDNAME", "2026-09-15T09:15:00+05:30")]
    NEW = [_row(f"NEW{i}", "2026-09-17T09:15:00+05:30") for i in range(3)]

    def test_fills_at_the_session_open_belong_to_a_book_started_later_that_day(self):
        cloud = _cloud(self.STATE, self.OLD + self.NEW)
        assert len(cloud.read_positions()) == 3

    def test_rows_from_earlier_books_stay_out(self):
        got = _cloud(self.STATE, self.OLD + self.NEW).read_positions()
        assert "OLDNAME" not in set(got.symbol)

    def test_explicit_book_start_wins_over_the_fallback(self):
        state = {**self.STATE, "book_start": "2026-09-17T10:00:00+00:00"}
        assert len(_cloud(state, self.OLD + self.NEW).read_positions()) == 0

    def test_no_epoch_means_no_filtering(self):
        assert len(_cloud({"cash": "1"}, self.OLD + self.NEW).read_positions()) == 4


class TestFillRecords:
    """G5: every execution event reaches the cloud with its costs."""

    def test_fills_cancels_and_stops_are_all_recorded(self, tmp_path, monkeypatch):
        from nse_engine.config import CostConfig
        import kite_connect.trading.paper_trader as ptmod
        monkeypatch.setattr(ptmod, "_DB_PATH", tmp_path / "paper.sqlite3")

        class FakeCloud:
            def __init__(self): self.fills = []
            def sync_fills(self, rows): self.fills.extend(rows); return True
            def sync_position(self, pos): return True
            def sync_state(self, values): return True
            def read_state(self): return {}

        cloud = FakeCloud()
        pt = ptmod.PaperTrader(kite=None, initial_capital=1_000_000)
        pt._get_cloud = lambda: cloud
        decided, session = pd.Timestamp("2026-09-17"), pd.Timestamp("2026-09-18")
        pt.queue_pending_orders(decided, [
            {"symbol": "AAA", "side": "BUY", "quantity": 100, "target_qty": 100, "ref_price": 1000.0,
             "stop_price": 900.0, "reason": "entry"},
            {"symbol": "BBB", "side": "BUY", "quantity": 10, "target_qty": 10, "ref_price": 500.0,
             "stop_price": 450.0, "reason": "entry"}])          # BBB gets no quote -> cancelled
        pt.fill_pending_orders(session, pd.DatetimeIndex([decided, session]),
                               {"AAA": {"open": 1010.0, "adv": 5e8}}, CostConfig(),
                               min_trade_value_inr=5000)
        pt.simulate_gtt_stops({"AAA": {"date": pd.Timestamp("2026-09-21"), "open": 880.0, "low": 870.0,
                                       "close": 875.0, "adv": 5e8}}, cost_config=CostConfig())
        kinds = {f["source"] for f in cloud.fills}
        assert kinds == {"pending_open", "cancel", "stop"}
        fill = next(f for f in cloud.fills if f["source"] == "pending_open")
        assert fill["fill_price"] > fill["ref_price"], "a buy fills above the decision price after impact"
        assert fill["impact_bps"] > 0 and fill["costs_inr"] > 0
        stop = next(f for f in cloud.fills if f["source"] == "stop")
        assert stop["pnl"] < 0 and stop["quantity"] == fill["quantity"]

    def test_a_failing_cloud_never_breaks_a_session(self, tmp_path, monkeypatch):
        import kite_connect.trading.paper_trader as ptmod
        monkeypatch.setattr(ptmod, "_DB_PATH", tmp_path / "paper.sqlite3")

        class Broken:
            def sync_fills(self, rows): raise RuntimeError("neon down")
            def sync_position(self, pos): return True
            def sync_state(self, values): return True
            def read_state(self): return {}

        pt = ptmod.PaperTrader(kite=None, initial_capital=100_000)
        pt._get_cloud = lambda: Broken()
        pt._record_fill(order_id="x", session_date="2026-09-18", source="pending_open", symbol="AAA")
        assert pt._flush_fills() == 0          # reported as not written, but no exception


class TestWeeklyReport:
    """G11: week numbering survives a fresh runner, and a 2-day-old book is not judged."""

    def _trader(self, tmp_path, monkeypatch, weekly_rows, snapshots):
        import kite_connect.trading.paper_trader as ptmod
        monkeypatch.setattr(ptmod, "_DB_PATH", tmp_path / "paper.sqlite3")
        pt = ptmod.PaperTrader(kite=None, initial_capital=3_500_000)
        import sqlite3
        conn = sqlite3.connect(str(ptmod._DB_PATH))
        for d, eq in snapshots:
            conn.execute("INSERT OR REPLACE INTO daily_snapshots (date, equity, cash, open_positions,"
                         " closed_today, day_pnl, cumulative_pnl, cumulative_pnl_pct, max_drawdown_pct,"
                         " signals_generated, signals_traded, snapshot_json) "
                         "VALUES (?, ?, 0, 0, 0, 0, 0, 0, 0, 0, 0, '{}')", (d, eq))
        for w in weekly_rows:
            conn.execute("INSERT OR REPLACE INTO weekly_checkpoints (week_number, week_start, week_end,"
                         " start_equity, end_equity, week_return_pct, trades_opened, trades_closed,"
                         " win_rate, sharpe_ratio, max_dd_pct, avg_holding_days, summary_json) "
                         "VALUES (?, ?, ?, 0, 0, 0, 0, 0, 0, 0, 0, 0, '{}')", w)
        conn.commit(); conn.close()
        return pt

    def test_week_number_continues_after_a_restore(self, tmp_path, monkeypatch):
        """With last week's checkpoint restored, this Saturday is Week 2, not Week 1 again."""
        pt = self._trader(tmp_path, monkeypatch,
                          weekly_rows=[(1, "2026-09-17", "2026-09-18")],
                          snapshots=[("2026-09-17", 3_523_526), ("2026-09-18", 3_577_634),
                                     ("2026-09-21", 3_600_000), ("2026-09-25", 3_650_000)])
        ckpt = pt.checkpoint_weekly()
        assert ckpt["week_number"] == 2
        assert ckpt["week_start"] == "2026-09-21", "the week must start after the last checkpoint"

    def test_without_the_restore_it_would_repeat_week_one(self, tmp_path, monkeypatch):
        """The bug being fixed: an empty local table restarts the numbering."""
        pt = self._trader(tmp_path, monkeypatch, weekly_rows=[],
                          snapshots=[("2026-09-21", 3_600_000), ("2026-09-25", 3_650_000)])
        assert pt.checkpoint_weekly()["week_number"] == 1

    def test_session_count_drives_the_verdict(self, tmp_path, monkeypatch):
        pt = self._trader(tmp_path, monkeypatch, weekly_rows=[],
                          snapshots=[("2026-09-17", 3_523_526), ("2026-09-18", 3_577_634)])
        assert pt.session_count() == 2, "two sessions is not a track record"


class TestDispatchBreadcrumb:
    """The Space's logs are unreachable from outside, so a dispatch attempt
    must record what happened where the book can be read (21 Sep 2026: nothing
    ran and there was no way to tell whether the Space had even tried)."""

    class _Cloud:
        def __init__(self, last_session=""):
            self.last, self.written = last_session, {}
        def read_state(self): return {"engine_last_session": self.last}
        def sync_state(self, values): self.written.update(values); return True

    def _run(self, monkeypatch, cloud, urlopen):
        import urllib.request
        scheduler = pytest.importorskip("scheduler")
        monkeypatch.setattr(scheduler, "_save_run", lambda *a, **k: None)
        monkeypatch.setattr(pc, "get_paper_cloud", lambda: cloud)
        monkeypatch.setattr(urllib.request, "urlopen", urlopen)
        scheduler._dispatch_nse_paper_workflow()

    def test_a_successful_dispatch_is_recorded(self, monkeypatch):
        monkeypatch.setenv("CENTURION_GH_DISPATCH_TOKEN", "tok")

        class Resp:
            status = 204
            def __enter__(self): return self
            def __exit__(self, *a): return False

        cloud = self._Cloud(last_session="2026-09-18")
        self._run(monkeypatch, cloud, lambda req, timeout=None: Resp())
        assert cloud.written["nse_dispatch_status"] == "dispatched"

    def test_a_rejected_token_is_recorded_with_its_code(self, monkeypatch):
        import io, urllib.error
        monkeypatch.setenv("CENTURION_GH_DISPATCH_TOKEN", "tok")

        def boom(req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, 401, "err", {}, io.BytesIO(b'{"message":"bad"}'))

        cloud = self._Cloud(last_session="2026-09-18")
        self._run(monkeypatch, cloud, boom)
        assert cloud.written["nse_dispatch_status"] == "http_401"

    def test_the_retry_skips_a_session_already_processed(self, monkeypatch):
        from datetime import datetime, timedelta, timezone
        monkeypatch.setenv("CENTURION_GH_DISPATCH_TOKEN", "tok")
        today = datetime.now(timezone(timedelta(hours=5, minutes=30))).date().isoformat()

        def must_not_post(req, timeout=None):
            raise AssertionError("the retry posted although the session was done")

        cloud = self._Cloud(last_session=today)
        self._run(monkeypatch, cloud, must_not_post)
        assert cloud.written == {}


class TestSessionActivity:
    """A day with no trades must be recorded as a decision, not left blank.

    18-23 Sep 2026: four sessions ran, the book held inside its no-trade buffer,
    and the Daily Detail tab showed nothing at all for any of them.
    """

    class _Plan:
        def __init__(self, notes, buys=(), sells=(), stops=(), skipped=()):
            self.notes, self.skipped, self.stop_instructions = notes, list(skipped), list(stops)
            self.buys, self.sells, self.shift_multiplier = list(buys), list(sells), 1.0

    def _record(self, monkeypatch, plan, queued=0, fills=None, stops=()):
        import cloud_paper_runner as runner

        captured = {}

        class Cloud:
            def sync_session(self, row): captured.update(row); return True

        pt = type("PT", (), {"_get_cloud": lambda self: Cloud(), "cash": 50_000.0})()
        # fills present = a freshly processed session (a re-plan has fills=None)
        session = {"session": "2026-09-23", "notes": [], "stops": list(stops),
                   "fills": fills or {}}
        snapshot = {"equity": 3_621_800.0, "open_positions": 21}
        runner._record_session_activity(pt, session, snapshot, plan, queued, fills or {})
        return captured

    def test_a_rebalance_day_that_needed_no_trades_says_so(self, monkeypatch):
        row = self._record(monkeypatch, self._Plan(notes=["rebalance_day"], stops=[1] * 20))
        assert row["rebalance_day"] is True
        assert row["planned_buys"] == 0 and row["planned_sells"] == 0
        assert "no-trade buffer" in row["outcome"]
        assert row["stops_armed"] == 20 and row["open_positions"] == 21

    def test_a_hold_day_is_distinguished_from_a_rebalance_day(self, monkeypatch):
        row = self._record(monkeypatch, self._Plan(notes=[]))
        assert row["rebalance_day"] is False
        assert "not a rebalance day" in row["outcome"]

    def test_an_active_day_lists_what_happened(self, monkeypatch):
        row = self._record(monkeypatch, self._Plan(notes=["rebalance_day"], buys=[1, 2], sells=[3]),
                           queued=3, fills={"filled": [1, 2], "cancelled": [3]}, stops=[9])
        assert row["filled"] == 2 and row["cancelled"] == 1 and row["queued"] == 3
        assert row["stops_triggered"] == 1
        assert "filled at the open" in row["outcome"]

    def test_a_replan_does_not_overwrite_what_the_session_did(self, monkeypatch):
        """25 Sep 2026: four late backups re-planned and zeroed the day's 2 fills."""
        import cloud_paper_runner as runner

        written = []

        class Cloud:
            def sync_session(self, row): written.append(row); return True

        pt = type("PT", (), {"_get_cloud": lambda self: Cloud(), "cash": 0.0})()
        replan = {"session": "2026-09-25", "notes": ["already processed"], "stops": [],
                  "fills": None}
        runner._record_session_activity(pt, replan, {"equity": 1.0, "open_positions": 22},
                                        self._Plan(notes=[]), 0, {})
        assert written == [], "a re-plan must leave the recorded session alone"

    def test_recording_never_breaks_a_session(self, monkeypatch):
        import cloud_paper_runner as runner

        class Broken:
            def sync_session(self, row): raise RuntimeError("neon down")

        pt = type("PT", (), {"_get_cloud": lambda self: Broken(), "cash": 0.0})()
        runner._record_session_activity(pt, {"session": "2026-09-23", "notes": [], "stops": [],
                                             "fills": {}},
                                        {"equity": 1.0, "open_positions": 0},
                                        self._Plan(notes=[]), 0, {})      # must not raise


class TestNeonUrl:
    """25 Sep 2026: every paper run died with "No module named 'psycopg'".

    SQLAlchemy changed which driver a bare postgresql:// URL resolves to, and
    this project installs psycopg2. The driver must be named in the URL.
    """

    @pytest.mark.parametrize("raw", [
        "postgresql://u:p@host/db",
        "postgres://u:p@host/db",
        "postgresql://u:p@host/db?sslmode=require&channel_binding=require",
    ])
    def test_the_driver_is_always_named(self, raw):
        from cloud_paper_runner import _neon_url
        url = _neon_url(raw)
        assert url.startswith("postgresql+psycopg2://"), url

    def test_channel_binding_is_dropped_and_ssl_required(self):
        from cloud_paper_runner import _neon_url
        url = _neon_url("postgresql://u:p@host/db?channel_binding=require")
        assert "channel_binding" not in url and "sslmode=require" in url

    def test_an_explicit_driver_is_left_alone(self):
        from cloud_paper_runner import _neon_url
        assert _neon_url("postgresql+psycopg://u:p@h/d").startswith("postgresql+psycopg://")


class TestDeploymentContext:
    """The container build must be able to open every requirements file.

    25 Sep 2026: requirements.txt began with `-r requirements-core.txt`, the
    Dockerfiles copied only requirements.txt, and the Space died with
    "Could not open requirements file" - the site returned a plain-text 503
    that the login page could not parse as JSON.
    """

    def test_every_dockerfile_copies_the_included_file(self):
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        included = [l.split("-r ", 1)[1].strip()
                    for l in (root / "requirements.txt").read_text().splitlines()
                    if l.strip().startswith("-r ")]
        assert included, "requirements.txt no longer includes another file; update this test"
        for docker in root.glob("deployment/Dockerfile*"):
            text = docker.read_text()
            if "requirements.txt" not in text:
                continue
            for name in included:
                assert name in text, f"{docker.name} installs requirements.txt but never copies {name}"

    @staticmethod
    def _pins(path):
        """{normalised name: version} of the ``name==version`` lines of a requirements file."""
        import re
        out = {}
        for line in path.read_text().splitlines():
            m = re.match(r"^([A-Za-z0-9._-]+)(?:\[[^\]]*\])?==([^\s#;]+)", line.split("#")[0].strip())
            if m:
                out[re.sub(r"[-_.]+", "-", m.group(1)).lower()] = m.group(2)
        return out

    @pytest.mark.lesson('L-LOCK',
                        pattern='F10',
                        date='2026-11-14',
                        source='PG1',
                        summary='The app installed whatever was newest at each rebuild; kiteconnect drifted to 5.2.2 unnoticed (CK1-P8)')
    def test_the_space_installs_against_its_lock(self):
        """PG1 (14 Nov 2026): the Space's web stack floated to the newest release at every package rebuild."""
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        docker = (root / "deployment" / "Dockerfile.hf").read_text()
        assert "-r requirements.txt -c constraints-hf.txt" in docker
        assert "constraints-hf.txt ./" in docker, "the build context must hold the lock, or pip cannot open it"
        lines = [l.strip() for l in (root / "constraints-hf.txt").read_text().splitlines()
                 if l.strip() and not l.startswith("#")]
        lock = self._pins(root / "constraints-hf.txt")
        assert len(lock) == len(lines) > 150, "every line of the lock is one exact pin"

    @pytest.mark.lesson('L-LOCK')
    def test_a_pin_and_the_lock_never_disagree(self):
        """A pin moved in a requirements file but not in the lock fails the Space's build on the conflict."""
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        lock = self._pins(root / "constraints-hf.txt")
        for name in ("requirements-core.txt", "requirements.txt"):
            for pkg, version in self._pins(root / name).items():
                assert lock.get(pkg) == version, (f"{name} pins {pkg}=={version} but constraints-hf.txt has "
                                                  f"{lock.get(pkg)}: regenerate the lock (see its header)")


class TestPartialClose:
    """G16: selling part of a lot must keep the entry time of the shares sold.

    The first partial sale in the paper book (RBLBANK, 25 Sep 2026) was stamped
    with the time of the trim, so the closed trade read "opened 21:50, closed
    09:15" and the weekly report showed an average holding period of -0.5 days.
    """

    def test_a_trimmed_lot_keeps_its_entry_time(self, tmp_path, monkeypatch):
        import kite_connect.trading.paper_trader as ptmod
        monkeypatch.setattr(ptmod, "_DB_PATH", tmp_path / "paper.sqlite3")

        class Cloud:
            def sync_position(self, pos): return True
            def sync_state(self, values): return True
            def read_state(self): return {}

        pt = ptmod.PaperTrader(kite=None, initial_capital=1_000_000)
        pt._get_cloud = lambda: Cloud()
        entry = "2026-09-17T09:15:00+05:30"
        pt._open_lot("AAA", 100, 1000.0, 900.0, 50.0, entry, "entry")
        events = pt.close_position("AAA", quantity=40, price=1100.0,
                                   when="2026-09-25T09:15:00+05:30", reason="REBALANCE")
        assert events, "the partial sale booked nothing"
        sold = next(p for p in pt._positions if not p.is_open)
        assert sold.quantity == 40
        assert sold.opened_at.startswith("2026-09-17T09:15:00"), sold.opened_at
        assert sold.opened_at != entry, "the row must stay distinguishable from the parent lot"
        # the microsecond offset makes this 7d 23:59:59.999999, so round it
        held = round((pd.Timestamp(sold.closed_at) - pd.Timestamp(sold.opened_at)).total_seconds() / 86400)
        assert held == 8, f"holding period should be 8 days, got {held}"
        parent = next(p for p in pt._positions if p.is_open)
        assert parent.quantity == 60 and parent.opened_at == entry


def _neon_down(*_args, **_kwargs):
    from sqlalchemy.exc import OperationalError

    raise OperationalError("INSERT INTO paper_positions ...", {"symbol": "RELIANCE", "quantity": 7},
                           Exception("server closed the connection unexpectedly"))


def _unreachable_book():
    obj = pc.PaperCloudSync.__new__(pc.PaperCloudSync)
    obj.schema, obj._factory, obj._session = None, None, _neon_down
    return obj


class TestFailedWrites:
    """RB2: each write to the book commits on its own, so a failed one leaves a position without its cash or
    fills without their snapshot.  The run must end as failed (the session-failed email), as a live session does
    (LN-T10), instead of reporting success and letting the next session build on a half-written book."""

    @pytest.fixture(autouse=True)
    def _fresh(self, monkeypatch):
        monkeypatch.setattr(pc, "_failed_writes", [])

    def test_every_write_method_records_its_failure(self):
        book = _unreachable_book()
        calls = {"position": lambda: book.sync_position(_row("AAA", "2026-10-28T09:15:00+05:30")),
                 "snapshot": lambda: book.sync_snapshot({"date": "2026-10-28", "equity": 1.0}),
                 "signals": lambda: book.sync_signals("2026-10-28", [{"symbol": "AAA"}]),
                 "session": lambda: book.sync_session({"session": "2026-10-28"}),
                 "fills": lambda: book.sync_fills([{"symbol": "AAA"}]),
                 "weekly": lambda: book.sync_weekly({"week_number": 1}),
                 "SL update": lambda: book.sync_stop_loss("AAA", "2026-10-28T09:15:00+05:30", 90.0),
                 "state": lambda: book.sync_state({"cash": 1.0})}
        for what, call in calls.items():
            assert call() is False, what
        assert pc.failed_writes() == [f"{w} (OperationalError)" for w in calls]

    def test_a_failed_write_fails_the_run(self, monkeypatch):
        import cloud_paper_runner as runner

        statuses = []
        def session():
            _unreachable_book().sync_fills([{"symbol": "AAA"}])   # the rest of the session went through
            return "ok", "1 fill"
        monkeypatch.setattr(runner, "install_public_log_filter", lambda: False)
        monkeypatch.setattr(runner, "_check_active", lambda: True)
        monkeypatch.setattr(runner, "_engine_enabled", lambda argv=None: True)
        monkeypatch.setattr(runner, "_parse_args", lambda argv=None: type("A", (), {"new_book": False})())
        monkeypatch.setattr(runner, "_run_engine_paper", lambda new_book=False: session())
        monkeypatch.setattr(runner, "_run_weekly_checkpoint", lambda use_engine=False: session())   # on a Saturday
        monkeypatch.setattr(runner, "_update_run_status", lambda status, message: statuses.append((status, message)))
        with pytest.raises(SystemExit) as exit_:
            runner.main([])
        assert exit_.value.code == 1
        assert statuses[-1][0] == "error" and "1 write(s) to the book in Neon failed: fills (OperationalError)" \
            in statuses[-1][1]

    def test_a_clean_run_still_reports_its_own_status(self, monkeypatch):
        import cloud_paper_runner as runner

        statuses = []
        for name, value in {"install_public_log_filter": lambda: False, "_check_active": lambda: True,
                            "_engine_enabled": lambda argv=None: True,
                            "_parse_args": lambda argv=None: type("A", (), {"new_book": False})(),
                            "_run_engine_paper": lambda new_book=False: ("ok", "done"),
                            "_run_weekly_checkpoint": lambda use_engine=False: ("ok", "done"),
                            "_update_run_status": lambda status, message: statuses.append((status, message))}.items():
            monkeypatch.setattr(runner, name, value)
        runner.main([])
        assert statuses == [("ok", "done")]

    def test_on_actions_the_log_names_only_the_error_class(self):
        """PL1: the driver's error text quotes the row (symbol, quantity); the public log must not."""
        import logging

        from kite_connect.trading.public_log import PublicLogFilter

        class Keep(logging.Handler):
            def __init__(self):
                super().__init__()
                self.lines = []
            def emit(self, record):
                self.lines.append(record.getMessage())

        def logged(on_actions: bool) -> list:
            handler = Keep()
            if on_actions:
                handler.addFilter(PublicLogFilter())
            pc.logger.addHandler(handler)
            try:
                _unreachable_book().sync_position(_row("RELIANCE", "2026-10-28T09:15:00+05:30"))
            finally:
                pc.logger.removeHandler(handler)
            return handler.lines

        assert logged(on_actions=True) == ["Cloud sync position failed (OperationalError)"]
        assert "RELIANCE" in logged(on_actions=False)[0], "run locally, the full error still prints"


class TestUnusableNumbers:
    """RB4: G4 (RA1's input) never judges a book on equity that is missing, infinite or not positive, or on an
    infinite reference return; before, one infinite day still produced a WATCH.  The refusal fails the run."""

    DAYS = pd.bdate_range("2026-08-03", periods=60)

    def _equity(self, bad=None):
        import numpy as np

        eq = pd.Series(1_000_000.0 * np.cumprod(1 + np.random.default_rng(4).normal(0.0005, 0.008, len(self.DAYS))),
                       index=self.DAYS)
        if bad is not None:
            eq.iloc[30] = bad
        return eq

    def _ref(self):
        return self._equity().pct_change().fillna(0.0)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -5.0])
    def test_unusable_equity_is_refused_by_date_not_amount(self, bad):
        from nse_engine import paper_gate

        with pytest.raises(paper_gate.UnusableInput) as err:
            paper_gate.evaluate(self._equity(bad), self._ref())
        assert str(err.value) == "paper equity missing, infinite or not positive on 1 date(s), first 2026-09-14"

    @pytest.mark.parametrize("bad", [float("inf"), -1.0, -1.5])
    def test_an_impossible_reference_return_is_refused(self, bad):
        from nse_engine import paper_gate

        ref = self._ref()
        ref.iloc[10] = bad
        with pytest.raises(paper_gate.UnusableInput):
            paper_gate.evaluate(self._equity(), ref)

    def test_clean_numbers_are_judged_as_before(self):
        from nse_engine import paper_gate

        ref = self._ref()
        ref.iloc[5] = float("nan")                        # a missing reference day is still just left out
        rep = paper_gate.evaluate(self._equity(), ref)
        assert rep["verdict"] in (paper_gate.PASS, paper_gate.WATCH, paper_gate.FAIL) and rep["sessions"] == 58
        assert paper_gate.evaluate(pd.Series(dtype="float64"), ref)["verdict"] == paper_gate.NOT_ENOUGH

    def test_a_refusal_fails_the_run(self, tmp_path, monkeypatch):
        import cloud_paper_runner as runner

        csv = tmp_path / "ref.csv"
        self._ref().rename("return").rename_axis("date").to_csv(csv)
        monkeypatch.setenv("CENTURION_SHIFT_REFERENCE_CSV", str(csv))
        monkeypatch.setattr(runner, "_refusals", [])
        monkeypatch.setattr(pc, "_failed_writes", [])
        broken = self._equity(0.0)
        pt = type("PT", (), {"_get_cloud": lambda self: None, "equity_history": lambda self: broken})()
        assert runner._paper_gate(pt) is None
        with pytest.raises(RuntimeError, match="paper gate: paper equity missing, infinite or not positive"):
            runner._raise_on_failed_writes()

    def test_the_nightly_statistics_skip_a_book_they_cannot_test(self):
        from types import SimpleNamespace

        from nse_engine import books as bk

        day = lambda name, eq: SimpleNamespace(book=SimpleNamespace(name=name), paper=SimpleNamespace(equity=eq))
        sig = bk.paper_significance([day("good", self._equity()), day("bad", self._equity(0.0))])
        assert "t" in sig["good"]["test"] and "error" in sig["bad"]["test"]
        assert "bad" not in (bk._significance_line(sig) or "")


class TestReadableEmail:
    """EM2: the daily and Saturday emails go out in Calibri at larger sizes; the words never change."""

    HTML = ("<html><body style='font-family:Segoe UI,Arial,sans-serif;font-size:13px;'><h2 style='font-size:18px;'>"
            "Paper books</h2><p style=\"font-size:11px;color:#666\">The book's note</p>"
            "<td style='font-family:monospace;font-size:12px;'>RELIANCE x 7</td></body></html>")

    @staticmethod
    def _text(html):
        import re
        from html.parser import HTMLParser

        class T(HTMLParser):
            def __init__(self):
                super().__init__()
                self.out, self.skip = [], 0

            def handle_starttag(self, tag, attrs):
                self.skip += tag in ("style", "head")

            def handle_endtag(self, tag):
                self.skip -= tag in ("style", "head")

            def handle_data(self, data):
                if not self.skip:
                    self.out.append(data)
        t = T()
        t.feed(html)
        return re.sub(r"\s+", " ", "".join(t.out)).strip()

    def test_calibri_larger_sizes_and_the_same_words(self):
        from nse_engine.books import EMAIL_FONT, readable

        out = readable(self.HTML)
        assert self._text(out) == self._text(self.HTML)
        assert f"font-family:{EMAIL_FONT}" in out and "Segoe UI,Arial,sans-serif;" not in out.split("</head>")[1]
        assert "font-family:monospace" in out                                    # codes keep their face
        assert "font-size:15px" in out and "font-size:20px" in out and "font-size:12px" in out   # 13, 18, 11 -> floor 12
        assert "font-size:13px" in out                                            # monospace 12 -> 13

    def test_a_second_pass_changes_nothing(self):
        from nse_engine.books import readable

        once = readable(self.HTML)
        assert readable(once) == once and once.count("<head>") == 1


class TestBookPnlInTheDailyReport:
    """A book with a sleeve or overlay leads with its whole value's P&L and labels the shares as shares; a plain
    book keeps "Total P&L" (6 Nov 2026: the futures book read "Total P&L -₹43,638" while it was up ₹20.8 lakh)."""

    BASE = {"session": "2026-11-04", "deployment": "x", "equity": 15_956_362, "initial_capital": 16_000_000,
            "cash": 1, "pnl": -43_638, "pnl_pct": -0.27, "max_drawdown_pct": 3.0, "open_positions": 20,
            "filled": [], "cancelled": [], "stops": [], "queued": [], "notes": [], "alerts": []}
    FUT = {"nav": 22_079_502, "initial": 20_000_000, "day_pnl": 445_963, "exposure": 1.73, "notional": 23_192_137,
           "legs": 4, "biggest_pct": 0.37, "futures_cash": 6_356_653, "margin": 4_638_427, "margin_pct": 0.21,
           "futures_pnl": 582_210, "costs": 0, "roll_costs": 0, "metrics": {}, "holdings": [], "fills": [], "queued": []}

    def _rows(self, report):
        import re
        from services.notifications.manager import NotificationManager

        _, html = NotificationManager.__new__(NotificationManager).render_engine_daily_report(report)
        return [(k, re.sub("<[^>]+>", "", v)) for k, v in
                re.findall(r"<td[^>]*>([^<]*)</td><td[^>]*>(.*?)</td>", html)]

    def test_a_futures_book_leads_with_its_whole_value(self):
        rows = self._rows({**self.BASE, "futures": self.FUT})
        assert rows[0] == ("Book P&amp;L (shares + futures)", "₹2,079,502.00 (+10.40%)")
        labels = [k for k, _ in rows]
        assert "Shares P&amp;L" in labels and "Total P&amp;L" not in labels

    def test_a_plain_book_keeps_total_pnl(self):
        labels = [k for k, _ in self._rows(self.BASE)]
        assert "Total P&amp;L" in labels and not any(k.startswith("Book P&amp;L") for k in labels)

    def _day(self, name, status, objective):
        """A book's kept report (the real template) as the daily summary reads it."""
        from types import SimpleNamespace

        from services.notifications.manager import NotificationManager

        subject, html = NotificationManager.__new__(NotificationManager).render_engine_daily_report(
            {**self.BASE, "objective": objective})
        return SimpleNamespace(book=SimpleNamespace(name=name, deployment=SimpleNamespace(status=status)),
                               email={"session": "2026-11-04", "subject": subject, "html": html})

    def test_the_trials_objective_is_said_once_under_the_findings(self):
        """13 Nov 2026 (the owner's ask): the same paragraph closed each of the six trial books' reports."""
        from datetime import date

        from nse_engine import books as bk
        from nse_engine.deployment import STATUS_CANDIDATE

        trial = "Objective: trial configuration 2d64ba4c beside the deployed book."
        own = "Objective: rehearse the deployed configuration 679cbd0c."
        days = [self._day("deployed", "deployed", own), self._day("candidate", STATUS_CANDIDATE, trial),
                self._day("e4", STATUS_CANDIDATE, trial.replace("2d64ba4c", "93cf6c4d"))]
        assert trial in days[1].email["html"]                    # a report sent on its own still carries it
        sections = bk._book_sections(days, date(2026, 11, 4))
        assert "Objective: trial configuration" not in sections  # gone from every trial book's section
        assert sections.count(own) == 1                          # the deployed book's own line stays
        assert "Session 2026-11-04" in sections and sections.count("Open positions") == 3   # nothing else removed
        once = bk.trials_objective(days)
        assert once.startswith("Objective of the trial books (candidate, e4): ")
        assert f"After {bk.fg.FORWARD_MIN_SESSIONS} sessions the forward gate decides" in once
        assert f"within {bk.fg.WF_SHARPE_TOLERANCE} of the deployed one's." in once
        body = bk._summary_body(["Book"], [["x"]], ["a finding"], "the footer", once)
        assert body.count(once) == 1 and body.index("a finding") < body.index(once) < body.index("the footer")
        assert "Objective" not in bk._summary_body(["Book"], [["x"]], ["a finding"], "the footer")

    def test_no_trial_books_no_objective_line(self):
        from nse_engine import books as bk

        assert bk.trials_objective([self._day("deployed", "deployed", "Objective: rehearse.")]) == ""


@pytest.mark.lesson('L-FL2',
                    pattern='F1',
                    date='2026-11-14',
                    source='FL2',
                    summary='A missed futures session could skip a bonus adjustment, strand an expired contract and fill orders a day late')
class TestFuturesLegCatchUp:
    """FL1 (14 Nov 2026): on a night NSE's F&O file came after the last run the futures leg lost the session
    (2-4 and 12-13 Nov).  It now runs the sessions it missed before tonight's, and a gap it cannot close is
    marked and cannot corrupt the contracts."""

    import pandas as _pd

    SESS = _pd.bdate_range("2026-12-07", periods=8)          # Mon 7 Dec .. Wed 16 Dec, no NSE holiday
    E1, E2 = SESS[5], _pd.Timestamp("2027-01-26")            # the near contract expires on the sixth session
    WEIGHTS = {"AAA": 0.5}

    @classmethod
    def _contracts(cls, i, lot=100.0, scale=1.0):
        px = (100.0 + i) * scale
        rows = [{"symbol": "AAA", "expiry": cls.E2, "open": px + 0.5 * scale, "settle": px + 1.0 * scale,
                 "lot_size": lot, "value_inr": 4e8}]
        if i <= 5:
            rows.insert(0, {"symbol": "AAA", "expiry": cls.E1, "open": px - 0.5 * scale, "settle": px,
                            "lot_size": lot, "value_inr": 5e8})
        return pd.DataFrame(rows)

    @classmethod
    def _spot(cls, i, j=None, raw=None):
        """The store's view on session i against session j (the one before unless given); ``raw`` overrides
        the unadjusted closes {session: close} (a bonus)."""
        j = i - 1 if j is None else j
        adj = lambda k: 100.0 + k - 0.2
        r = lambda k: (raw or {}).get(k, adj(k))
        return {"AAA": {"open": adj(i) - 0.4, "raw_close": r(i), "raw_prev": r(j), "adj_close": adj(i),
                        "adj_prev": adj(j), "vol": 0.02}}

    @staticmethod
    def _equity(i):
        return 1_000_000.0 + 5_000.0 * i

    def _night(self, fo, ledger, i, traded=(), contracts=None, spot=None):
        eq = self._equity(i)
        return fo.run_session(ledger, fo.OverlayConfig(), self.SESS[i], self._contracts(i) if contracts is None else contracts,
                              share_equity=eq, share_invested=0.9 * eq, target_weights=self.WEIGHTS, traded=set(traded),
                              spot=self._spot(i) if spot is None else spot, sessions=self.SESS)

    def _book_rows(self, days):
        return {self.SESS[i].date().isoformat(): {"equity": self._equity(i), "cash": 0.1 * self._equity(i)} for i in days}

    def test_no_gap_nothing_is_missed_and_the_adjustment_reads_the_session_before(self):
        from types import SimpleNamespace

        from kite_connect.trading import futures_overlay as fo

        ledger = fo.Ledger(cash=250_000.0)
        assert fo.missed_sessions(ledger, self.SESS[3], self.SESS) == []               # no session yet
        self._night(fo, ledger, 0, traded={"AAA"})
        assert fo.missed_sessions(ledger, self.SESS[1], self.SESS) == []               # an ordinary next night
        assert fo.missed_sessions(ledger, self.SESS[3], self.SESS) == list(self.SESS[1:3])
        close = pd.DataFrame({"AAA": [10.0 + k for k in range(8)]}, index=self.SESS)
        data = SimpleNamespace(close=close, close_unadj=close * 2, open=close - 0.5)
        plain = fo.spot_inputs(data, self.SESS[4], ["AAA"])
        assert plain == fo.spot_inputs(data, self.SESS[4], ["AAA"], since=self.SESS[3].date().isoformat())
        assert plain == fo.spot_inputs(data, self.SESS[4], ["AAA"], since="") == fo.spot_inputs(
            data, self.SESS[4], ["AAA"], since="2020-01-01")                           # not in the store: as before
        gap = fo.spot_inputs(data, self.SESS[4], ["AAA"], since=self.SESS[1].date().isoformat())["AAA"]
        assert (gap["adj_prev"], gap["raw_prev"]) == (11.0, 22.0) and gap["adj_close"] == plain["AAA"]["adj_close"]

    def test_catching_up_gives_the_ledger_an_unbroken_run_would_have(self):
        """Sessions 1 and 2 are missed (an order due at session 1's open, a roll due at session 2) and caught
        up before session 3: every number matches the run that never missed them."""
        from kite_connect.trading import futures_overlay as fo

        straight = fo.Ledger(cash=250_000.0)
        self._night(fo, straight, 0, traded={"AAA"})
        for i in (1, 2, 3):
            self._night(fo, straight, i)
        assert straight.history[1]["futures_pnl"] != 0 and straight.legs["AAA"].expiry == str(self.E2.date())   # filled, rolled
        assert straight.history[2]["roll_costs"] > 0

        late = fo.Ledger(cash=250_000.0)
        self._night(fo, late, 0, traded={"AAA"})
        seen = []
        res = fo.catch_up(late, fo.OverlayConfig(), self.SESS[3], self.SESS, lambda m: self._contracts(self.SESS.get_loc(m)),
                          self._book_rows((1, 2)),
                          lambda m, since: seen.append((m, since)) or self._spot(self.SESS.get_loc(m)))
        assert res == {"caught_up": [self.SESS[1].date().isoformat(), self.SESS[2].date().isoformat()], "skipped": {}}
        assert seen == [(self.SESS[1], self.SESS[0].date().isoformat()), (self.SESS[2], self.SESS[1].date().isoformat())]
        self._night(fo, late, 3)
        assert [r.pop("caught_up", False) for r in late.history] == [False, True, True, False]
        assert late.history == straight.history and late.cash == straight.cash
        assert late.legs == straight.legs and late.pending == straight.pending and late.fills == straight.fills
        assert not any("gap_sessions" in r for r in late.history)

    def test_a_session_that_cannot_be_caught_up_stays_a_marked_gap_and_a_bonus_in_it_still_adjusts(self):
        from kite_connect.trading import futures_overlay as fo

        ledger = fo.Ledger(cash=250_000.0)
        self._night(fo, ledger, 0, traded={"AAA"})
        self._night(fo, ledger, 1)
        qty, lot, ref = ledger.legs["AAA"].qty, ledger.legs["AAA"].lot, ledger.legs["AAA"].ref

        def contracts_for(m):                                # session 2's file never came; a 1:1 bonus fell on it
            i = self.SESS.get_loc(m)
            if i == 2:
                raise FileNotFoundError("no F&O bhavcopy")
            return self._contracts(i, lot=200.0, scale=0.5)

        raw = {0: 99.8, 1: 100.8, 2: 50.9, 3: 51.4, 4: 51.9}   # the unadjusted close halves on session 2
        res = fo.catch_up(ledger, fo.OverlayConfig(), self.SESS[4], self.SESS, contracts_for, self._book_rows((2, 3)),
                          lambda m, since: self._spot(self.SESS.get_loc(m), self.SESS.get_loc(pd.Timestamp(since)), raw))
        assert res == {"caught_up": [self.SESS[3].date().isoformat()],
                       "skipped": {self.SESS[2].date().isoformat(): "no F&O bhavcopy for it"}}
        row = ledger.history[-1]
        assert row["gap_sessions"] == 1 and row["caught_up"] is True
        leg = ledger.legs["AAA"]
        assert leg.lot == pytest.approx(2 * lot, rel=0.02) and leg.qty == pytest.approx(2 * qty, rel=0.05)   # NSE's adjustment
        assert abs(row["futures_pnl"]) < 0.05 * qty * ref     # two days' move, not the bonus read as a 50% loss
        no_row = fo.catch_up(ledger, fo.OverlayConfig(), self.SESS[6], self.SESS, contracts_for, {},
                             lambda m, since: self._spot(self.SESS.get_loc(m)))
        assert set(no_row["skipped"].values()) == {"the share book has no row for it"} and not no_row["caught_up"]

    def test_a_leg_past_its_expiry_stops_the_leg_loudly(self):
        """Never rolled because the sessions before its expiry were missed: an error, not a contract that sits
        in the book unmarked for good."""
        from kite_connect.trading import futures_overlay as fo

        ledger = fo.Ledger(cash=250_000.0)
        self._night(fo, ledger, 0, traded={"AAA"})
        self._night(fo, ledger, 1)
        assert ledger.legs["AAA"].expiry == str(self.E1.date())
        saved = ledger.to_json()
        with pytest.raises(RuntimeError, match="past its expiry and was never rolled"):
            self._night(fo, ledger, 6)                       # sessions 2-5, the roll window and the expiry, were lost
        assert fo.Ledger.from_json(saved).last_session == self.SESS[1].date().isoformat()


@pytest.mark.lesson("L-NEO6", pattern="F1", date="2026-10-25", source="NEO6",
                    summary="A new book replayed from its start stamped tonight as its epoch, hiding every replayed row")
def test_a_new_book_replayed_from_its_start_is_dated_from_that_start():
    from datetime import date
    from types import SimpleNamespace

    import cloud_paper_runner as runner

    written = []
    runner._date_book_start(SimpleNamespace(_get_cloud=lambda: SimpleNamespace(sync_state=written.append)), date(2026, 10, 21))
    assert written == [{"epoch": "2026-10-21T00:00:00+05:30", "book_start": "2026-10-21T00:00:00+05:30"}]
    runner._date_book_start(SimpleNamespace(_get_cloud=lambda: None), date(2026, 10, 21))      # no database: nothing to date


# ── a paper book end to end: the executor's nightly session on the synthetic panel ──

def _paper_book(tmp_path, monkeypatch, panel):
    """An engine paper book over ``panel``: its own local state, no database, the default rules on a small universe."""
    import kite_connect.trading.paper_trader as ptmod
    from kite_connect.trading.nse_engine_executor import EngineExecutor
    from nse_engine.config import EngineConfig

    monkeypatch.setattr(ptmod, "_DB_PATH", tmp_path / "paper.sqlite3")
    pt = ptmod.PaperTrader(kite=None, initial_capital=3_000_000)
    pt._get_cloud = lambda: None
    dates = panel.dates
    cfg = EngineConfig().replace(start=str(dates[300].date()), end=str(dates[-1].date()),
                                 **{"universe.top_n_liquid": 40, "universe.min_history_days": 60,
                                    "universe.liquidity_lookback_days": 40, "universe.min_median_value_inr": 0.0,
                                    "universe.min_price_inr": 0.0})
    ex = EngineExecutor(paper=True, config=cfg, data_loader=lambda *a, **k: panel, paper_trader=pt,
                        shift_state_path=str(tmp_path / "shift.json"))
    return ex, pt, cfg


def _queued(report) -> dict:
    return {r["id"]: (r["symbol"], r["side"], r["quantity"]) for r in report["results"] if r.get("status") == "PENDING"}


def test_a_paper_book_fills_tonights_orders_at_the_next_open_and_never_at_the_close(tmp_path, monkeypatch, panel):
    ex, pt, cfg = _paper_book(tmp_path, monkeypatch, panel)
    days = list(panel.dates[-12:])
    queued, fills, stops = {}, 0, 0
    for prev, day in zip([None] + days, days):
        report = ex.run_paper_session(as_of=day, replay=True)
        assert report["session"] == day.date().isoformat() and report["caught_up"] == [report["session"]]
        filled = report["fills"]["filled"]
        assert {f["id"] for f in filled} <= set(queued.get(prev, {}))       # only orders decided the session before
        for f in filled:
            session_open = float(panel.open.loc[day, f["symbol"]])
            assert f["open"] == pytest.approx(session_open) and f["open"] != pytest.approx(
                float(panel.close.loc[prev, f["symbol"]]))                   # never the close that made the decision
            assert f["fill_price"] >= f["open"] if f["side"] == "BUY" else f["fill_price"] <= f["open"]
            assert f["costs"] > 0 and (f["symbol"], f["side"]) == queued[prev][f["id"]][:2]
        queued[day] = _queued(report)
        assert all(o["decision_date"] == report["session"] for o in pt.pending_orders())
        held = pt.holdings()
        assert pt.cash >= 0 and len(held) <= cfg.portfolio.max_positions
        for sym, h in held.items():                                          # nothing ends a session through its stop
            assert h["quantity"] > 0 and 0 < h["stop_price"] < float(panel.close.loc[day, sym]), sym
        assert all(e["symbol"] not in held for e in report["stops"])
        fills, stops = fills + len(filled), stops + len(report["stops"])
    assert fills >= 20 and stops >= 3, "the panel no longer exercises fills and stops: the checks above ran on nothing"
    assert pt.engine_last_session() == days[-1].date()


def test_a_second_run_of_the_same_session_only_replans(tmp_path, monkeypatch, panel):
    ex, pt, _ = _paper_book(tmp_path, monkeypatch, panel)
    for day in panel.dates[-12:-10]:
        ex.run_paper_session(as_of=day, replay=True)
    first = ex.run_paper_session(as_of=panel.dates[-10], replay=True)
    cash, held, orders = pt.cash, pt.holdings(), sorted(_queued(first).values())
    assert orders, "no order was queued: nothing to supersede"
    again = ex.run_paper_session(as_of=panel.dates[-10], replay=True)
    assert again["fills"] is None and again["stops"] == [] and "re-planning only" in again["notes"][0]
    assert pt.cash == cash and pt.holdings() == held                        # no fill, no stop, twice
    assert sorted(_queued(again).values()) == orders
    assert sorted((o["symbol"], o["side"], o["quantity"]) for o in pt.pending_orders()) == orders   # superseded, not doubled


def test_missed_sessions_are_caught_up_in_order_and_pending_orders_fill_at_their_own_open(tmp_path, monkeypatch, panel):
    """LN-T7: after sessions nobody ran, each one's stops and fills are applied in order.  An order decided before
    the gap fills at the first missed session's open: it is neither cancelled as stale nor filled at a later price."""
    ex, pt, _ = _paper_book(tmp_path, monkeypatch, panel)
    days = list(panel.dates[-12:])
    for day in days[:3]:
        before = ex.run_paper_session(as_of=day, replay=True)
    waiting = _queued(before)
    assert waiting, "no order was waiting at the gap"
    report = ex.run_paper_session(as_of=days[6], replay=True)                # days 3, 4 and 5 never ran
    assert report["caught_up"] == [d.date().isoformat() for d in days[3:7]]
    filled = {f["id"]: f for f in report["fills"]["filled"]}
    assert set(filled) == set(waiting) and report["fills"]["cancelled"] == []
    for f in filled.values():
        assert f["open"] == pytest.approx(float(panel.open.loc[days[3], f["symbol"]]))
    held = pt.holdings()
    assert report["stops"] and all(e["symbol"] not in held for e in report["stops"])
    assert pt.engine_last_session() == days[6].date()
    assert all(o["decision_date"] == days[6].date().isoformat() for o in pt.pending_orders())


def test_a_live_only_call_is_refused_for_a_paper_session_and_bad_settings_are_named(tmp_path, monkeypatch, panel):
    from kite_connect.trading.nse_engine_executor import EngineExecutor

    ex, _, _ = _paper_book(tmp_path, monkeypatch, panel)
    ex.paper = False
    with pytest.raises(RuntimeError, match="paper-only"):
        ex.run_paper_session(as_of=panel.dates[-1])
    with pytest.raises(ValueError, match="sell_credit must be in"):
        EngineExecutor(paper=True, sell_credit=1.5)
    with pytest.raises(ValueError, match="paper_fill must be"):
        EngineExecutor(paper=True, paper_fill="yesterday")


# ── the nightly runner around that session (cloud_paper_runner) ──

def _runner_deployment(cfg, **sleeves):
    from types import SimpleNamespace

    return SimpleNamespace(**{"status": "candidate", "engine": cfg, "futures_overlay": None, "bond_sleeve": None,
                              "mcx_sleeve": None, "options_overlay": None, **sleeves})


def test_the_runner_records_each_session_and_a_sleeve_that_fails_never_fails_the_share_book(tmp_path, monkeypatch,
                                                                                            panel):
    import re

    import cloud_paper_runner as runner

    ex, pt, cfg = _paper_book(tmp_path, monkeypatch, panel)
    monkeypatch.setenv("CENTURION_SHIFT_REFERENCE_CSV", str(tmp_path / "no_reference.csv"))
    days = list(panel.dates[-12:])
    status, first = runner._engine_session(pt, _runner_deployment(cfg), ex, as_of=days[0])
    queued = int(re.search(r"queued=(\d+)", first).group(1))
    assert status == "success" and queued > 0 and f"engine session={days[0].date()} deployment=candidate filled=0" in first
    status, second = runner._engine_session(pt, _runner_deployment(cfg), ex, as_of=days[1])
    assert status == "success" and f"filled={queued} cancelled=0" in second          # last night's orders, at this open
    assert [pd.Timestamp(d).date() for d in pt.equity_history().index] == [days[0].date(), days[1].date()]

    def broken(*args, **kwargs):
        raise RuntimeError("no bars for the session")

    def not_out_yet(*args, **kwargs):
        raise FileNotFoundError("the F&O bhavcopy is not published yet")

    monkeypatch.setattr(runner, "_futures_overlay_session", not_out_yet)
    for name in ("_bond_sleeve_session", "_mcx_sleeve_session", "_options_overlay_session"):
        monkeypatch.setattr(runner, name, broken)
    every_sleeve = _runner_deployment(cfg, futures_overlay={"on": 1}, bond_sleeve={"on": 1}, mcx_sleeve={"on": 1},
                                      options_overlay={"on": 1})
    status, third = runner._engine_session(pt, every_sleeve, ex, as_of=days[2])
    assert status == "success" and len(pt.equity_history()) == 3                      # the share book's session stands
    for said in ("futures deferred (F&O bhavcopy not published yet)", "GILT SLEEVE FAILED: no bars for the session",
                 "MCX SLEEVE FAILED: no bars for the session", "OPTIONS OVERLAY FAILED: no bars for the session"):
        assert said in third
    public = runner._counts_only(third)                                               # the Actions log is public
    assert "cash=" not in public and "no bars for the session" not in public and "GILT SLEEVE FAILED" in public


def test_the_daily_email_goes_once_for_a_new_session_and_a_dead_mail_server_fails_nothing(tmp_path, monkeypatch, panel):
    from pathlib import Path

    import cloud_paper_runner as runner
    from nse_engine.deployment import load_deployment
    from services.notifications.manager import NotificationManager

    ex, pt, _ = _paper_book(tmp_path, monkeypatch, panel)
    for name in ("CENTURION_PAPER_BOOK_LABEL", "CENTURION_BOOK_EMAILS"):
        monkeypatch.delenv(name, raising=False)
    sent = []
    monkeypatch.setattr(NotificationManager, "_send_html_email", lambda self, s, h: sent.append((s, h)) or True)
    dep = load_deployment(Path(__file__).resolve().parent.parent / "config" / "nse_engine_deployed.json")
    for day in panel.dates[-12:-10]:
        session = ex.run_paper_session(as_of=day, replay=True)
    snapshot = pt.snapshot_daily(signals_generated=0, signals_traded=0, session_date=session["session"])
    runner._email_engine_session(pt, dep, session, snapshot, {})
    ((subject, html),) = sent
    filled = len(session["fills"]["filled"])
    assert subject.startswith(f"Centurion paper [deployed {dep.engine.config_hash()[:8]}] {session['session']}")
    assert f"{filled} filled, 0 stops, 0 queued" in subject and filled > 0
    assert f"Filled at the open ({filled})" in html and "Drawdown rule" in html and "Objective: rehearse" in html
    again = ex.run_paper_session(as_of=panel.dates[-11], replay=True)                  # a backup run: re-plans only
    runner._email_engine_session(pt, dep, again, snapshot, {})
    assert len(sent) == 1

    def dead(self, subject, html):
        raise ConnectionError("smtp down")

    monkeypatch.setattr(NotificationManager, "_send_html_email", dead)
    runner._email_engine_session(pt, dep, session, snapshot, {})                       # logged, never raised


def test_the_runner_lists_the_nights_no_run_processed(monkeypatch):
    from datetime import date
    from types import SimpleNamespace

    import cloud_paper_runner as runner
    from nse_engine.data import store

    asked = []

    def sessions_after(store_dir, after, on_or_before=None):
        asked.append((store_dir, after))
        return ["2026-11-11", "2026-11-12", "2026-11-13"]

    monkeypatch.setattr(store, "sessions_after", sessions_after)
    dep = SimpleNamespace(engine=SimpleNamespace(data=SimpleNamespace(store_dir="the-store")), paper_start_date=date(2026, 11, 11))
    assert runner._missed_sessions(dep, date(2026, 11, 9)) == ["2026-11-11", "2026-11-12"]   # the latest is tonight's
    assert runner._missed_sessions(dep, None, brand_new=True) == ["2026-11-11", "2026-11-12"]
    assert asked == [("the-store", date(2026, 11, 9)), ("the-store", date(2026, 11, 10))]    # from the day before its start
    assert runner._missed_sessions(dep, None) == []                                         # --new-book: starts tonight
    monkeypatch.setattr(store, "sessions_after", lambda *a, **k: [])
    assert runner._missed_sessions(dep, date(2026, 11, 13)) == []


# ── the gilt and MCX paper sleeves beside the share book ────────

def _gilt_bars():
    return pd.DataFrame({"open": [30.0, 30.1, 33.0, 30.4], "close": [30.0, 30.2, 30.3, 30.5],
                         "value_inr": [5e7, 6e7, 5e7, 7e7]},
                        index=pd.to_datetime(["2026-11-11", "2026-11-12", "2026-11-13", "2026-11-17"]))


def test_the_gilt_sleeve_queues_its_cash_buys_at_the_next_open_and_marks_at_the_close():
    from kite_connect.trading import bond_sleeve as bs

    cfg, bars, initial = bs.SleeveConfig(), _gilt_bars(), 4_000_000.0
    led = bs.Ledger(cash=1_000_000.0)
    first = bs.run_session(led, cfg, pd.Timestamp("2026-11-11"), bars, 3_000_000.0, initial)
    assert (first["nav"], first["units"], first["day_pnl"], first["fills"]) == (4_000_000.0, 0, 0.0, [])
    assert first["queued"] == [{"side": "BUY", "value": 1_000_000.0}]
    assert bs.run_session(led, cfg, pd.Timestamp("2026-11-11"), bars, 3_000_000.0, initial) == first   # a second run
    assert len(led.history) == 1
    day = pd.Timestamp("2026-11-12")
    second = bs.run_session(led, cfg, day, bars, 3_010_000.0, initial)
    units = 33_056                                                       # floor(995,000 / 30.10): 0.5% kept for costs
    value = units * 30.1
    costs = bs._cost(value, "BUY", day, 5e7)                              # at the ETF's own median traded value
    (fill,) = second["fills"]
    assert (fill["units"], fill["price"], fill["open_used"]) == (units, 30.1, True) and fill["costs"] == pytest.approx(costs)
    assert costs > 0 and led.cash == pytest.approx(1_000_000 - value - costs) and second["queued"] == []
    assert second["nav"] == pytest.approx(3_010_000 + led.cash + units * 30.2)
    assert second["weight"] == pytest.approx(units * 30.2 / second["nav"])
    absent = bs.run_session(led, cfg, pd.Timestamp("2026-11-16"), bars, 3_010_000.0, initial)   # no bar that day
    assert absent["price"] == 30.2                                        # the last close it has
    assert led.history[-1]["traded"] is False and absent["units"] == units and absent["fills"] == []


def test_the_gilt_sleeve_fills_at_the_close_when_the_open_is_a_stray_print():
    from kite_connect.trading import bond_sleeve as bs

    led = bs.Ledger(cash=1_000_000.0, pending={"side": "BUY", "value": 1_000_000.0})
    out = bs.run_session(led, bs.SleeveConfig(), pd.Timestamp("2026-11-13"), _gilt_bars(), 3_000_000.0, 4_000_000.0)
    (fill,) = out["fills"]
    assert (fill["price"], fill["open_used"]) == (30.3, False)            # the open was 33.0, 9% from the close
    tiny = bs.Ledger(cash=20.0, pending={"side": "BUY", "value": 20.0})
    assert bs.run_session(tiny, bs.SleeveConfig(), pd.Timestamp("2026-11-13"), _gilt_bars(), 0.0, 20.0)["fills"] == []
    assert tiny.cash == 20.0 and tiny.pending == {"side": "BUY", "value": 20.0}       # nothing bought: queued again


@pytest.mark.parametrize("module,raw,why", [
    ("bond_sleeve", {"weight": 0.25, "ticker": "X"}, "unknown bond_sleeve keys"),
    ("bond_sleeve", {"weight": 1.0}, "weight must be between 0 and 1"),
    ("bond_sleeve", {"weight": 0.25, "symbol": ""}, "symbol must be an NSE symbol"),
    ("mcx_sleeve", {"weight": 0.25, "lots": 2}, "unknown mcx_sleeve keys"),
    ("mcx_sleeve", {"weight": 0}, "weight must be between 0 and 1"),
    ("mcx_sleeve", {"weight": 0.25, "config": "GT5-Z"}, "config must be one of"),
])
def test_a_sleeves_settings_are_refused_when_they_cannot_be_meant(module, raw, why):
    import importlib

    sleeve = importlib.import_module(f"kite_connect.trading.{module}")
    with pytest.raises(ValueError, match=why):
        sleeve.SleeveConfig.from_dict(raw)
    assert sleeve.SleeveConfig.from_dict({"weight": 0.2}).weight == 0.2


def test_a_sleeves_ledger_is_kept_in_the_books_state_and_never_lost_silently():
    from types import SimpleNamespace

    from kite_connect.trading import bond_sleeve as bs, mcx_sleeve as ms

    state = {}
    cloud = SimpleNamespace(read_state=lambda: dict(state), sync_state=lambda values: state.update(values) or True)
    gilt = bs.load_ledger(cloud, 1_000_000.0)
    gilt.units = 7
    bs.save_ledger(cloud, gilt)
    assert bs.load_ledger(cloud, 5.0) == gilt and bs.load_ledger(None, 5.0) == bs.Ledger(cash=5.0)
    mcx = ms.load_ledger(cloud, "2026-10-21", 1_000_000.0)
    ms.save_ledger(cloud, mcx)
    assert ms.load_ledger(cloud, "2020-01-01", 1.0) == mcx
    refuses = SimpleNamespace(read_state=lambda: {}, sync_state=lambda values: False)
    for sleeve, ledger, name in ((bs, gilt, "gilt"), (ms, mcx, "MCX")):
        with pytest.raises(RuntimeError, match=f"could not save the {name} sleeve's ledger"):
            sleeve.save_ledger(refuses, ledger)
        with pytest.raises(RuntimeError, match=f"the {name} sleeve needs the book's cloud state"):
            sleeve.save_ledger(None, ledger)


def test_the_combined_book_of_shares_and_gilts_is_measured_from_its_own_history():
    from kite_connect.trading import bond_sleeve as bs

    assert bs.metrics(bs.Ledger(cash=1.0), 4_000_000.0) == {}
    led = bs.Ledger(cash=0.0, history=[
        {"nav": 4_040_000.0, "weight": 0.24, "costs": 300.0, "traded": True},
        {"nav": 3_999_600.0, "weight": 0.26, "costs": 0.0, "traded": False}])
    m = bs.metrics(led, 4_000_000.0)
    assert (m["sessions"], m["untraded_days"], m["costs_total"]) == (2, 1, 300.0) and "sharpe" not in m
    assert m["return"] == pytest.approx(-0.0001) and m["max_drawdown"] == pytest.approx(-0.01) and m["weight_mean"] == 0.25
    led.history = [{"nav": 4_000_000.0 * 1.001 ** i * (1.002 if i % 2 else 1.0), "weight": 0.25, "costs": 0.0,
                    "traded": True} for i in range(1, 25)]
    assert bs.metrics(led, 4_000_000.0)["sharpe"] > 0


def test_the_mcx_sleeve_replays_its_trend_book_and_says_when_the_data_is_stale_or_revised(monkeypatch):
    from types import SimpleNamespace

    from kite_connect.trading import mcx_sleeve as ms
    from nse_engine import mcx_trend

    calendar = pd.to_datetime(["2026-10-21", "2026-10-22", "2026-10-23"])
    equity = pd.Series([1_000_000.0, 1_010_000.0, 1_020_000.0], index=calendar)
    trades = [{"date": "2026-10-22", "instrument": "GOLD", "contract": "GOLDM 2026-12-04", "side": "BUY", "lots": 2,
               "fee": 40.0, "slippage": 60.0}]

    def run_trend(config, prepared):
        end = pd.Timestamp(config.end)
        return SimpleNamespace(equity=equity[equity.index <= end], trades=[t for t in trades if t["date"] <= config.end],
                               decisions=pd.DataFrame([{"date": "2026-10-21", "GOLD_lots": 2, "CRUDE_lots": 0}]))

    gold = SimpleNamespace(name="GOLD", close=lambda contract, day: 80_000.0, multiplier=lambda contract: 1.0)
    prepared = {"calendar": calendar, "instruments": [gold]}
    monkeypatch.setattr(mcx_trend, "run_trend", run_trend)
    liquid = pd.Series([100.0, 100.02, 100.04], index=calendar)
    cfg, led = ms.SleeveConfig(), ms.Ledger(start="2026-10-21", capital=1_000_000.0)
    first = ms.run_session(led, cfg, calendar[0], 3_000_000.0, 4_000_000.0, prepared, liquid)
    assert first["nav"] == pytest.approx(4_000_000.0) and led.history[-1]["gross"] == 0.0
    second = ms.run_session(led, cfg, calendar[1], 3_000_000.0, 4_000_000.0, prepared, liquid)
    sleeve = 1_010_000.0 * 100.02 / 100.0                                  # the trend book, grown at the liquid rate
    assert second["nav"] == pytest.approx(3_000_000 + sleeve) and led.history[-1]["costs"] == 100.0
    assert led.history[-1]["gross"] == pytest.approx(160_000 / sleeve) and len(led.trades) == 1 and len(led.decisions) == 1
    assert ms.run_session(led, cfg, calendar[1], 3_000_000.0, 4_000_000.0, prepared, liquid)["nav"] == second["nav"]
    assert len(led.history) == 2                                           # a second run of the night adds nothing
    equity.iloc[1] = 1_020_000.0                                           # MCX revises a day already recorded
    late = ms.run_session(led, cfg, pd.Timestamp("2026-10-30"), 3_000_000.0, 4_000_000.0, prepared, liquid)
    assert led.history[-1]["mcx_asof"] == "2026-10-23" and led.history[-1]["alerts"] == 2 and len(led.trades) == 1
    alerts = " | ".join(late.get("alerts") or [])
    assert "MCX data is stale: latest session 2026-10-23" in alerts and "MCX replay moved 1.00% of the sleeve" in alerts
    m = ms.metrics(led, 4_000_000.0)
    assert (m["sessions"], m["decisions"], m["zero_lot_decisions"]) == (3, 1, {"GOLD": 0, "CRUDE": 1})
    held = ms.positions(trades + [{**trades[0], "side": "SELL"}], prepared, calendar[2])
    assert held == []                                                      # bought and rolled out: nothing held


# ── the options overlay beside the share book (A1, paper only) ──

def _a1_night(monkeypatch, equity, cycles=(), events=True):
    """A1's own simulator replaced by its result: the overlay's arithmetic on top is what is under test."""
    from types import SimpleNamespace

    from kite_connect.options import backtest, sleeves

    monkeypatch.setattr(backtest, "run_sleeve", lambda scfg, store, skip_entry_dates=None: SimpleNamespace(
        equity=equity[equity.index <= pd.Timestamp(scfg.end)], cycles=list(cycles)))
    monkeypatch.setattr(sleeves, "events_ahead", lambda day: events)


def test_the_options_overlay_adds_its_weight_times_a1s_return_to_the_book(monkeypatch):
    from kite_connect.trading import options_overlay as ov

    cfg = ov.OverlayConfig()                                                  # A1 at 3x, no cap
    capital = float(ov.sleeve_config(cfg, "2026-10-26", "2026-10-27").initial_capital)
    days = pd.to_datetime(["2026-10-26", "2026-10-27", "2026-10-28"])
    equity = pd.Series([capital, capital * 1.01, capital * 1.01 * 0.98], index=days)
    _a1_night(monkeypatch, equity)
    led = ov.Ledger(start="2026-10-26")
    first = ov.run_session(led, cfg, days[0], 3_000_000.0, 3_000_000.0, "fo-store")
    assert (first["nav"], first["overlay"], first["alerts"], first["position"]) == (3_000_000.0, 0.0, [], None)
    second = ov.run_session(led, cfg, days[1], 3_020_000.0, 3_000_000.0, "fo-store")
    assert second["overlay_pnl"] == pytest.approx(3 * 0.01 * 3_000_000)       # on the book's previous value
    assert second["nav"] == pytest.approx(3_020_000 + 90_000) and second["sizing"] == "3x"
    assert ov.run_session(led, cfg, days[1], 9.0, 3_000_000.0, "fo-store")["nav"] == second["nav"]   # a second run
    assert len(led.history) == 2
    third = ov.run_session(led, cfg, days[2], 3_020_000.0, 3_000_000.0, "fo-store")
    assert third["overlay_pnl"] == pytest.approx(3 * -0.02 * second["nav"])
    assert third["overlay"] == pytest.approx(90_000 + third["overlay_pnl"]) and third["metrics"]["sessions"] == 3


def test_the_overlay_says_when_its_data_is_stale_revised_or_its_event_list_has_run_out(monkeypatch):
    from kite_connect.trading import options_overlay as ov

    cfg = ov.OverlayConfig()
    capital = float(ov.sleeve_config(cfg, "2026-10-26", "2026-10-27").initial_capital)
    days = pd.to_datetime(["2026-10-26", "2026-10-27"])
    equity = pd.Series([capital, capital * 1.01], index=days)
    _a1_night(monkeypatch, equity)
    led = ov.Ledger(start="2026-10-26")
    ov.run_session(led, cfg, days[0], 3_000_000.0, 3_000_000.0, "fo-store")
    ov.run_session(led, cfg, days[1], 3_000_000.0, 3_000_000.0, "fo-store")
    equity.iloc[0] = capital * 1.001                                          # NSE revises a recorded day
    _a1_night(monkeypatch, equity, events=False)
    late = ov.run_session(led, cfg, pd.Timestamp("2026-11-05"), 3_000_000.0, 3_000_000.0, "fo-store")
    said = " | ".join(late["alerts"])
    assert "the event list ends" in said and "F&O data is stale: latest session 2026-10-27" in said
    assert "the A1 replay moved 0.30% of the book on a recorded date" in said and led.history[-1]["alerts"] == 3
    guarded = ov.OverlayConfig(candidate="A1G")
    with pytest.raises(ValueError, match="A1G has a crash guard: the guarded sessions are needed"):
        ov.run_session(ov.Ledger(start="2026-10-26"), guarded, days[0], 3_000_000.0, 3_000_000.0, "fo-store")


def test_a_capped_overlay_sizes_each_spread_so_its_maximum_loss_is_a_share_of_the_book(monkeypatch):
    from kite_connect.trading import options_overlay as ov

    days = pd.to_datetime(["2026-10-26", "2026-10-27", "2026-10-28", "2026-10-29", "2026-10-30"])
    equity = pd.Series([1_000_000.0, 1_000_000.0, 1_004_000.0, 1_006_000.0, 1_006_000.0], index=days)
    cycle = {"action": "traded", "fill_date": "2026-10-27", "exit_date": "2026-10-29", "max_loss_inr": 150_000.0,
             "expiry": "2026-11-24", "decision_date": "2026-10-26", "short": 26_500, "wing": 27_000, "lots": 2,
             "credit": 40.0, "exit": "expiry", "pnl_inr": 6_000.0}
    skipped = {"action": "skipped: event ahead", "fill_date": None, "exit_date": None}
    capped = ov.OverlayConfig(max_loss_cap=0.10)
    weights = ov.cycle_weights(capped, [cycle, skipped], equity)
    assert weights.tolist() == pytest.approx([0.0, 2 / 3, 2 / 3, 2 / 3, 0.0])  # 10% of 10 lakh / a 1.5 lakh maximum loss
    assert ov.cycle_weights(ov.OverlayConfig(max_loss_cap=0.9), [cycle], equity).max() == 3.0   # never above the weight
    assert ov.cycle_weights(ov.OverlayConfig(), [cycle], equity).tolist() == [3.0] * 5
    still_open = {**cycle, "exit_date": None, "exit": None, "pnl_inr": None}
    assert ov.cycle_weights(capped, [still_open], equity).tolist() == pytest.approx([0.0] + [2 / 3] * 4)
    _a1_night(monkeypatch, equity, cycles=[still_open])
    monkeypatch.setattr(ov, "sleeve_config", lambda cfg, start, end: __import__("types").SimpleNamespace(
        initial_capital=1_000_000.0, entry_guard=False, start=start, end=end))
    led = ov.Ledger(start="2026-10-26")
    out = ov.run_session(led, capped, days[4], 3_000_000.0, 3_000_000.0, "fo-store")
    assert out["sizing"] == "max loss ≤ 10% of the book" and out["position"]["weight"] == pytest.approx(2 / 3)
    assert out["position"]["max_loss_share"] == pytest.approx(150_000 / 1_006_000 * 2 / 3)      # about 10% of the book
    assert out["overlay_pnl"] == pytest.approx(2 / 3 * (0.004 + 2_000 / 1_004_000) * 3_000_000)   # each day's return, weighted
    assert out["position"]["book_lots"] == round(2 * (2 / 3) * out["nav"] / 1_006_000, 1)


@pytest.mark.parametrize("raw,why", [
    ({"candidate": "A9"}, "candidate must be one of"), ({"weight": 3.5}, "weight must be above 0 and at most 3"),
    ({"weight": 0}, "weight must be above 0"), ({"max_loss_cap": 1.0}, "max_loss_cap must be 0"),
    ({"leverage": 2}, "unknown options_overlay keys"),
])
def test_the_overlays_settings_stay_inside_what_was_tested(raw, why):
    from kite_connect.trading import options_overlay as ov

    with pytest.raises(ValueError, match=why):
        ov.OverlayConfig.from_dict(raw)
    assert ov.OverlayConfig.from_dict({"candidate": "A1G", "weight": 3.0, "max_loss_cap": 0.1}).max_loss_cap == 0.1


# ── the paper book through a split, a consolidation and a rename (LN-T4, LN-T5) ──

def _holding_aaa(tmp_path, monkeypatch):
    """A paper book holding 105 AAA bought at an open of 1,010 with a stop at 900, and 10 more queued."""
    import kite_connect.trading.paper_trader as ptmod
    from nse_engine.config import CostConfig

    monkeypatch.setattr(ptmod, "_DB_PATH", tmp_path / "paper.sqlite3")
    pt = ptmod.PaperTrader(kite=None, initial_capital=1_000_000)
    pt._get_cloud = lambda: None
    decided, session = pd.Timestamp("2026-09-17"), pd.Timestamp("2026-09-18")
    pt.queue_pending_orders(decided, [{"symbol": "AAA", "side": "BUY", "quantity": 105, "target_qty": 105,
                                       "ref_price": 1000.0, "stop_price": 900.0, "reason": "entry"}])
    pt.fill_pending_orders(session, pd.DatetimeIndex([decided, session]), {"AAA": {"open": 1010.0, "adv": 5e8}},
                           CostConfig(), min_trade_value_inr=5000)
    pt.queue_pending_orders(session, [{"symbol": "AAA", "side": "BUY", "quantity": 10, "target_qty": 115,
                                       "ref_price": 1020.0, "stop_price": 910.0, "reason": "rebalance"}])
    return pt


def test_a_bonus_moves_the_paper_lots_stops_and_queued_orders_into_the_new_units_once(tmp_path, monkeypatch):
    pt = _holding_aaa(tmp_path, monkeypatch)
    before, cash = pt.holdings()["AAA"], pt.cash
    out = pt.apply_share_change("AAA", 0.5, 0.5, 505.0, "AAA|bonus|2026-09-21", "2026-09-21")     # a 1:1 bonus
    assert out == {"symbol": "AAA", "factor": 0.5, "before": 105, "after": 210, "cash_in_lieu": 0.0}
    held = pt.holdings()["AAA"]
    assert (held["quantity"], held["stop_price"]) == (210, 450.0) and pt.cash == cash
    assert held["avg_price"] == pytest.approx(before["avg_price"] / 2, abs=1e-3)       # the cost of the lot is kept
    assert held["cost"] == pytest.approx(before["cost"], rel=1e-5)
    (queued,) = pt.pending_orders()
    assert (queued["quantity"], queued["target_qty"], queued["ref_price"], queued["stop_price"]) == (20, 230, 510.0, 455.0)
    assert pt.apply_share_change("AAA", 0.5, 0.5, 505.0, "AAA|bonus|2026-09-21", "2026-09-21") is None   # once per event
    assert pt.holdings()["AAA"]["quantity"] == 210
    import kite_connect.trading.paper_trader as ptmod

    restarted = ptmod.PaperTrader(kite=None, initial_capital=1_000_000)                # the same file, read afresh
    assert restarted.holdings()["AAA"]["quantity"] == 210 and restarted.holdings()["AAA"]["stop_price"] == 450.0


def test_a_consolidation_pays_the_fraction_in_cash_and_a_lot_left_with_nothing_is_closed(tmp_path, monkeypatch):
    pt = _holding_aaa(tmp_path, monkeypatch)
    cash = pt.cash
    out = pt.apply_share_change("AAA", 10.0, 10.0, 10_200.0, "AAA|consolidation|2026-09-21", "2026-09-21")
    assert (out["before"], out["after"]) == (105, 10) and out["cash_in_lieu"] == pytest.approx(0.5 * 10_200.0)
    assert pt.cash == pytest.approx(cash + 5_100.0) and pt.holdings()["AAA"]["quantity"] == 10
    assert pt.holdings()["AAA"]["stop_price"] == 9_000.0
    gone = pt.apply_share_change("AAA", 100.0, 100.0, 1_020_000.0, "AAA|consolidation|2026-10-21", "2026-10-21")
    assert gone["after"] == 0 and "AAA" not in pt.holdings()                           # ten shares are a tenth of one


def test_a_renamed_stock_carries_its_lots_and_queued_orders_under_the_new_name_once(tmp_path, monkeypatch):
    pt = _holding_aaa(tmp_path, monkeypatch)
    pt._price_overrides["AAA"] = 1015.0
    held = pt.holdings()["AAA"]
    assert pt.rename_symbol("AAA", "AAANEW", "AAA|rename|2026-09-21", "2026-09-21") == 1
    assert "AAA" not in pt.holdings() and pt.holdings()["AAANEW"] == held              # the same lot, from the same entry
    assert [o["symbol"] for o in pt.pending_orders()] == ["AAANEW"] and pt._price_overrides == {"AAANEW": 1015.0}
    assert pt.rename_symbol("AAA", "AAANEW", "AAA|rename|2026-09-21", "2026-09-21") == 0
    assert pt.rename_symbol("ZZZ", "ZZZNEW", "ZZZ|rename|2026-09-21", "2026-09-21") == 0   # nothing held: nothing moved


# ── the runner's words for a session, and its Saturday checkpoint ──

@pytest.mark.parametrize("plan,queued,fills,stops,rebalance,said", [
    ("plan", 3, {"filled": [1, 2], "cancelled": [1]}, 1, True,
     "2 order(s) filled at the open; 1 cancelled; 1 stop exit(s); 3 order(s) queued for the next open"),
    (None, 0, {}, 0, False, "no plan: the session was already processed, so only stops and marks were refreshed"),
    ("plan", 0, {}, 0, False, "held: not a rebalance day, so no orders were planned"),
    ("plan", 0, {}, 0, True, "held: rebalance day, but every position was already within the no-trade buffer, "
                             "so nothing needed trading"),
    ("halt", 0, {}, 0, False, "DRAWDOWN RULE halt (16.2% below peak): no new entries or adds; held: not a rebalance "
                              "day, so no orders were planned"),
    ("risk_off", 2, {}, 0, True, "DRAWDOWN RULE risk_off (31.0% below peak): core book to cash / metals; 2 order(s) "
                                 "queued for the next open"),
])
def test_the_runner_says_in_one_sentence_what_a_session_did_or_why_it_did_nothing(plan, queued, fills, stops,
                                                                                    rebalance, said):
    from types import SimpleNamespace

    import cloud_paper_runner as runner

    plans = {"plan": SimpleNamespace(drawdown_state="normal", drawdown_pct=0.0),
             "halt": SimpleNamespace(drawdown_state="halt", drawdown_pct=16.2),
             "risk_off": SimpleNamespace(drawdown_state="risk_off", drawdown_pct=31.0), None: None}
    assert runner._session_outcome(plans[plan], queued, fills, stops, rebalance) == said


def test_the_weekly_email_reads_the_stored_gate_and_never_calls_a_young_book_a_pass():
    import json
    from types import SimpleNamespace

    import cloud_paper_runner as runner
    from nse_engine import paper_gate as pg

    def trader(gate):
        state = {pg.STATE_KEY: json.dumps(gate)} if gate is not None else {}
        return SimpleNamespace(_get_cloud=lambda: SimpleNamespace(read_state=lambda: state))

    checks = [{"name": "tracking error", "display": "4.1%", "status": pg.PASS}]
    assert runner._weekly_gate_verdict(trader(None))[0].startswith("NOT ENOUGH DATA — the paper gate (G4) has not run")
    young = runner._weekly_gate_verdict(trader({"verdict": pg.NOT_ENOUGH, "sessions": 12, "min_sessions": 30,
                                                "checks": checks, "book_last": "2026-11-13"}))
    assert young[0] == "NOT ENOUGH DATA — 12 of 30 sessions (G4)" and "(PASS)" not in young[2]
    assert young[2] == "For information only, as of 2026-11-13: tracking error 4.1%"
    passed = runner._weekly_gate_verdict(trader({"verdict": pg.PASS, "sessions": 41, "checks": checks,
                                                 "book_last": "2026-11-13"}))
    assert passed[0].startswith("PASS") and passed[1] == "#15803d"
    assert passed[2] == ("41 sessions as of 2026-11-13: tracking error 4.1% (PASS). Promotion still needs 60 sessions "
                         "and the forward gate (V3).")
    failed = runner._weekly_gate_verdict(trader({"verdict": pg.FAIL, "sessions": 41, "checks": checks}))
    assert failed[0].startswith("FAIL — does not behave like its backtest") and failed[1] == "#dc2626"
    assert runner._weekly_gate_verdict(trader({"verdict": pg.WATCH, "sessions": 41, "checks": []}))[1] == "#d97706"
    down = SimpleNamespace(_get_cloud=lambda: SimpleNamespace(read_state=lambda: 1 / 0))
    assert runner._weekly_gate_verdict(down)[0].startswith("NOT ENOUGH DATA")          # unreadable: never a verdict


def test_the_saturday_checkpoint_emails_the_week_once_and_nothing_for_a_week_already_done(tmp_path, monkeypatch, panel):
    import re

    import cloud_paper_runner as runner
    from services.notifications.manager import NotificationManager

    ex, pt, cfg = _paper_book(tmp_path, monkeypatch, panel)
    monkeypatch.setenv("CENTURION_SHIFT_REFERENCE_CSV", str(tmp_path / "no_reference.csv"))
    for name in ("CENTURION_PAPER_BOOK_LABEL", "CENTURION_BOOK_EMAILS"):
        monkeypatch.delenv(name, raising=False)
    for day in panel.dates[-12:-4]:
        runner._engine_session(pt, _runner_deployment(cfg), ex, as_of=day)
    sent = []
    monkeypatch.setattr(NotificationManager, "_send_html_email",
                        staticmethod(lambda *args, **kw: sent.append(kw or args) or True))
    monkeypatch.setattr(runner, "_open_paper_trader", lambda capital=None: pt)
    status, message = runner._run_weekly_checkpoint(use_engine=True)
    assert status == "success" and len(sent) == 1
    mail = sent[0]
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", mail["html_body"]))
    assert "W1" in mail["subject"] or "Week" in mail["subject"]
    assert "NOT ENOUGH DATA" in text                                                   # a week-old book is not judged
    assert runner._week_session_count(pt, {"week_start": "2022-05-30", "week_end": "2022-06-03"}) == 5
    assert runner._run_weekly_checkpoint(use_engine=True) == ("success", "No weekly data") and len(sent) == 1


# ── a book restored from Neon is the current book only (database/paper_cloud.py) ──

class _StoredBook:
    """A book's Neon tables, in memory."""

    def __init__(self, state=None, positions=(), snapshots=(), weekly=()):
        self.state = state if isinstance(state, Exception) else dict(state or {})
        self.rows = {"positions": list(positions), "snapshots": list(snapshots), "weekly": list(weekly)}

    def read_state(self):
        if isinstance(self.state, Exception):
            raise self.state
        return dict(self.state)

    def read_positions(self):
        return pd.DataFrame(self.rows["positions"])

    def read_snapshots(self):
        return pd.DataFrame(self.rows["snapshots"])

    def read_weekly(self):
        return pd.DataFrame(self.rows["weekly"])


def test_a_restore_keeps_only_the_current_books_rows_and_its_first_fills_at_the_open():
    """17 Sep 2026: a book started at 10:46 IST filled its first orders at that day's 09:15 open, before its epoch."""
    from database.paper_cloud import restore_paper_state

    state = {"cash": "2500000", "initial_capital": "3500000", "epoch": "2026-09-17T05:16:25+00:00"}
    positions = [
        {"symbol": "OLD", "is_open": True, "opened_at": "2026-09-10T09:15:00+05:30"},          # an older book's
        {"symbol": "AAA", "is_open": 1, "opened_at": "2026-09-17T09:15:00+05:30"},             # this book's first fill
        {"symbol": "BBB", "is_open": "f", "opened_at": "2026-09-18T09:15:00+05:30", "pnl": float("nan")},
    ]
    snapshots = [{"date": "2026-09-18", "equity": 1.0}, {"date": "2026-09-16", "equity": 2.0},
                 {"date": "2026-09-17", "equity": 3.0}]
    weekly = [{"week_number": 2}, {"week_number": 1}]
    out = restore_paper_state(_StoredBook(state, positions, snapshots, weekly))
    assert (out["cash"], out["initial_capital"]) == (2_500_000.0, 3_500_000.0)
    assert [p["symbol"] for p in out["positions"]] == ["AAA"] and [p["symbol"] for p in out["closed_positions"]] == ["BBB"]
    assert out["closed_positions"][0]["pnl"] is None                             # NaN read back as nothing
    assert [s["date"] for s in out["snapshots"]] == ["2026-09-17", "2026-09-18"]   # in order, from the epoch's day
    assert [w["week_number"] for w in out["weekly"]] == [1, 2] and out["shift_state"] is None
    late = restore_paper_state(_StoredBook({**state, "book_start": "2026-09-17T05:16:25+00:00"}, positions))
    assert [p["symbol"] for p in late["positions"]] == []                         # an explicit start is honoured


def test_a_restore_never_starts_a_fresh_book_over_one_it_cannot_read():
    from database.paper_cloud import restore_paper_state

    assert restore_paper_state(None) is None
    legacy = _StoredBook({}, [{"symbol": "X", "is_open": True}], [{"date": "2026-09-01"}])
    assert restore_paper_state(legacy) is None                                    # no saved cash: not a book
    with pytest.raises(ConnectionError):
        restore_paper_state(_StoredBook(ConnectionError("neon suspended")))       # never read as "no book"
    no_epoch = restore_paper_state(_StoredBook({"cash": "100", "distribution_shift_state": "{}"},
                                               [{"symbol": "X", "is_open": True, "opened_at": "2020-01-01"}]))
    assert [p["symbol"] for p in no_epoch["positions"]] == ["X"] and no_epoch["shift_state"] == "{}"


def test_reads_are_cut_at_the_books_start_by_date_or_by_instant(monkeypatch):
    from database.paper_cloud import PaperCloudSync

    sync = PaperCloudSync.__new__(PaperCloudSync)
    state = {"epoch": "2026-09-17T05:16:25+00:00"}
    sync.read_state = lambda: state
    dated = pd.DataFrame({"date": ["2026-09-16", "2026-09-17", "2026-09-18"]})
    assert sync._since_epoch(dated, "date", True)["date"].tolist() == ["2026-09-17", "2026-09-18"]
    assert len(sync._since_epoch(dated, "date", False)) == 3
    stamped = pd.DataFrame({"opened_at": ["2026-09-17T09:14:00+05:30", "2026-09-17T09:15:00+05:30",
                                          "2026-09-18T09:15:00+05:30"]})
    assert sync._since_epoch(stamped, "opened_at", True)["opened_at"].tolist() == [
        "2026-09-17T09:15:00+05:30", "2026-09-18T09:15:00+05:30"]               # from the epoch day's open
    state.clear()
    assert len(sync._since_epoch(dated, "date", True)) == 3                     # no book started: nothing cut

    def down():
        raise ConnectionError("neon down")

    sync.read_state = down
    assert len(sync._since_epoch(dated, "date", True)) == 3                     # a dashboard read shows everything
    with pytest.raises(ConnectionError):
        sync._since_epoch(dated, "date", True, strict=True)                     # the live book never guesses


@pytest.mark.parametrize("value,schema", [("", None), ("candidate", "candidate"), ("e4_trial", "e4_trial")])
def test_a_second_books_schema_is_named_by_the_environment(monkeypatch, value, schema):
    from database import paper_cloud

    monkeypatch.setenv(paper_cloud.ENV_SCHEMA, value)
    assert paper_cloud.paper_schema_from_env() == schema


# ── the Saturday email's block for each sleeve (cloud_paper_runner._weekly_*_block) ──

_WEEK = [("2026-11-06", 1010.0, 10.0), ("2026-11-09", 1030.0, 20.0), ("2026-11-11", 1040.4, 10.4)]   # Fri, Mon, Wed


def _sleeve_rows(kind):
    rows = []
    for day, nav, pnl in _WEEK:
        row = {"date": day, "nav": nav, "day_pnl": pnl, "costs": 2.0}
        row.update({"gilt": {"units": 10, "price": 25.0, "value": 250.0, "weight": 0.24, "cash": 5.0, "traded": True},
                    "mcx": {"sleeve": 260.0, "weight": 0.25, "gross": 1.4, "mcx_asof": day},
                    "options": {"overlay": nav - 1000.0, "overlay_pnl": pnl, "a1_asof": day, "a1_equity": 1_000_000.0,
                                "open_weight": 0.0, "max_loss_share": 0.0, "a1_return": 0.0, "share_equity": 1000.0}}[kind])
        rows.append(row)
    return rows


def _weekly_block(monkeypatch, kind, state, sleeve):
    from types import SimpleNamespace

    import cloud_paper_runner as runner

    dep = SimpleNamespace(**{"paper_capital_inr": 1000.0, "futures_overlay": None, "bond_sleeve": None,
                             "mcx_sleeve": None, "options_overlay": None, kind: sleeve})
    monkeypatch.setattr(runner, "_load_engine_deployment", lambda: dep)
    trader = SimpleNamespace(_get_cloud=lambda: SimpleNamespace(read_state=lambda: dict(state)))
    return {"futures_overlay": runner._weekly_futures_block, "bond_sleeve": runner._weekly_bond_block,
            "mcx_sleeve": runner._weekly_mcx_block, "options_overlay": runner._weekly_options_block}[kind](trader)


@pytest.mark.parametrize("kind,key,ledger,title", [
    ("bond_sleeve", "bond_ledger", lambda: {"cash": 5.0, "units": 10, "history": _sleeve_rows("gilt")},
     "Gilt sleeve (paper, BD3)"),
    ("mcx_sleeve", "mcx_ledger", lambda: {"start": "2026-11-06", "capital": 250.0, "history": _sleeve_rows("mcx"),
                                          "decisions": [{"date": "2026-11-02", "GOLD_lots": 2, "CRUDE_lots": -1}]},
     "MCX trend sleeve (paper, DL1)"),
    ("options_overlay", "options_overlay_ledger", lambda: {"start": "2026-11-06", "history": _sleeve_rows("options")},
     "A1 options overlay (paper, A1P)"),
])
def test_each_sleeve_reports_its_week_from_the_nav_before_the_weeks_first_session(monkeypatch, kind, key, ledger, title):
    import json

    sleeve = {"weight": 3.0} if kind == "options_overlay" else {"weight": 0.25}
    html, tag = _weekly_block(monkeypatch, kind, {key: json.dumps(ledger())}, sleeve)
    assert title in html and tag == " | book +3.0%"                               # 1,040.40 / 1,010 - 1
    assert "+3.01% over 2 session(s)" in html and "₹1,000 → ₹1,040 (+4.04%)" in html
    if kind == "mcx_sleeve":
        assert "lots GOLD +2, CRUDE -1" in html and "1 (the test wants at least 6)" in html
    assert _weekly_block(monkeypatch, kind, {}, sleeve)[0].endswith("no session yet.</p>")
    broken = _weekly_block(monkeypatch, kind, {key: "{not json"}, sleeve)
    assert "report failed" in broken[0] and broken[1] == ""                     # the weekly email still goes
    assert _weekly_block(monkeypatch, kind, {key: json.dumps(ledger())}, None) == ("", "")    # a book without it


def test_the_futures_block_says_when_there_is_no_session_and_never_stops_the_weekly_email(monkeypatch):
    html, tag = _weekly_block(monkeypatch, "futures_overlay", {}, {"leverage": 1.73})
    assert html == "<p style='color:#6b7280;'>Futures leg: no session yet.</p>" and tag == ""
    broken = _weekly_block(monkeypatch, "futures_overlay", {"futures_ledger": "{not json"}, {"leverage": 1.73})
    assert "Futures leg: report failed" in broken[0]
    assert _weekly_block(monkeypatch, "futures_overlay", {}, None) == ("", "")


# ── the drift check: live returns against the backtest's, and the size it suggests ──

@pytest.mark.parametrize("result,alerts", [
    ({"reference_mode": "same_period", "n_live": 40, "tracking_error_annual": 0.05, "mean_daily_gap": -0.0001}, []),
    ({"reference_mode": "same_period", "n_live": 40, "tracking_error_annual": 0.12, "mean_daily_gap": 0.0},
     ["tracking error 12.00%/yr > 8.00%"]),
    ({"reference_mode": "same_period", "n_live": 40, "tracking_error_annual": None, "mean_daily_gap": -0.0005},
     ["mean daily gap -5.00 bp < -3 bp"]),
    ({"reference_mode": "same_period", "n_live": 29, "tracking_error_annual": 0.5, "mean_daily_gap": -0.01}, []),
    ({"reference_mode": "recent_history", "n_live": 40, "tracking_error_annual": 0.5, "mean_daily_gap": -0.01}, []),
    (None, []),
])
def test_a_steady_gap_to_the_backtest_is_named_only_on_the_same_period_and_enough_days(monkeypatch, result, alerts):
    from kite_connect.trading.paper_trader import reality_gap_alerts

    for name in ("CENTURION_SHIFT_MAX_TRACKING_ERROR", "CENTURION_SHIFT_MAX_DAILY_GAP_BPS"):
        monkeypatch.delenv(name, raising=False)
    assert reality_gap_alerts(result) == alerts


@pytest.fixture
def drift(tmp_path, monkeypatch):
    """A paper book whose live returns and drift report are given; its alerts and cloud writes recorded."""
    import json
    from types import SimpleNamespace

    import kite_connect.trading.paper_trader as ptmod
    from services.notifications.manager import NotificationManager
    from services.research import distribution_shift as ds

    monkeypatch.setattr(ptmod, "_DB_PATH", tmp_path / "paper.sqlite3")
    pt = ptmod.PaperTrader(kite=None, initial_capital=1_000_000)
    n = SimpleNamespace(pt=pt, report={}, alerts=[], synced={}, days=40, path=tmp_path, json=json)
    pt._get_cloud = lambda: SimpleNamespace(sync_state=lambda values: n.synced.update(values) or True)
    monkeypatch.setattr(pt, "_live_daily_returns", lambda conn: pd.Series([0.001] * n.days))
    monkeypatch.setattr(ds, "compare_live_to_backtest", lambda live: dict(n.report))
    monkeypatch.setattr(NotificationManager, "__init__", lambda self: None)
    monkeypatch.setattr(NotificationManager, "send_alert", lambda self, subject, body: n.alerts.append(subject) or True)
    for name in ("CENTURION_SHIFT_MAX_TRACKING_ERROR", "CENTURION_SHIFT_MAX_DAILY_GAP_BPS"):
        monkeypatch.delenv(name, raising=False)
    return n


def _stored_state(drift):
    from database.paper_cloud import SHIFT_STATE_KEY
    from kite_connect.trading.paper_trader import SHIFT_STATE_FILENAME

    local = drift.json.loads((drift.path / SHIFT_STATE_FILENAME).read_text())
    assert drift.json.loads(drift.synced[SHIFT_STATE_KEY]) == local                # the next runner reads Neon's copy
    return local


def test_the_drift_check_waits_for_thirty_days_and_for_a_reference(drift):
    from kite_connect.trading.paper_trader import SHIFT_STATE_FILENAME

    drift.days = 29
    assert drift.pt._run_distribution_shift() is None
    drift.days, drift.report = 40, {"reference_mode": "unavailable", "verdict": "stable"}
    assert drift.pt._run_distribution_shift()["reference_mode"] == "unavailable"
    assert not (drift.path / SHIFT_STATE_FILENAME).exists() and drift.synced == {} and drift.alerts == []


@pytest.mark.parametrize("report,verdict,multiplier,alerted", [
    ({"verdict": "stable", "reference_mode": "same_period", "n_live": 40, "tracking_error_annual": 0.04,
      "mean_daily_gap": 0.0}, "stable", 1.0, []),
    ({"verdict": "stable", "calibrated_verdict": "drifting", "effective_verdict": "drifting",
      "reference_mode": "recent_history", "n_live": 40}, "drifting", 0.75, []),
    ({"verdict": "stable", "reference_mode": "same_period", "n_live": 40, "tracking_error_annual": 0.12,
      "mean_daily_gap": 0.0}, "drifting", 0.75, ["REALITY GAP — live paper returns trail the same-period backtest"]),
    ({"verdict": "regime_break", "reference_mode": "same_period", "n_live": 40, "tracking_error_annual": 0.12,
      "mean_daily_gap": 0.0}, "regime_break", 0.5, ["REGIME BREAK — Distribution Shift Detected"]),
])
def test_the_drift_check_sizes_the_book_by_its_most_severe_reading_and_alerts_once(drift, report, verdict, multiplier,
                                                                                   alerted):
    drift.report = report
    out = drift.pt._run_distribution_shift()
    state = _stored_state(drift)
    assert (out["position_verdict"], state["verdict"], state["position_size_multiplier"]) == (verdict, verdict, multiplier)
    assert drift.alerts == alerted


def test_a_replayed_nights_verdict_is_dated_that_night(drift):
    drift.report = {"verdict": "stable", "reference_mode": "recent_history", "n_live": 40}
    drift.pt._run_distribution_shift(session_date="2026-10-22")
    assert _stored_state(drift)["updated_at"] == "2026-10-22T23:00:00+05:30"


def test_the_saturday_email_carries_the_operations_block_after_the_findings():
    from datetime import date
    from pathlib import Path
    from types import SimpleNamespace

    from nse_engine.books import Book, DayBook, PaperBook, build_weekly

    days = pd.bdate_range(end="2026-11-13", periods=4)
    engine = SimpleNamespace(config_hash=lambda: "deployed00000000")
    book = Book("deployed", Path("config/nse_engine_deployed.json"),
                SimpleNamespace(paper_start_date=date(2026, 11, 10), status="approved", engine=engine))
    paper = PaperBook("deployed", pd.Series([3.5e6, 3.51e6, 3.52e6, 3.53e6], index=days), pd.DatetimeIndex(days), None)
    day = DayBook(book, paper, 3.5e6)
    report = {"comparisons": {}, "gates": {}, "cleared": [], "body": "<p>COMPARISON</p>"}
    ops = {"head": "Operations, week from 2026-11-07: 1 item(s) need action; 1 incident(s)",
           "lines": ["a nightly session failed: 1x ..."], "html": "<h3>Operations</h3><p>OPS BLOCK</p>"}
    plain = build_weekly([day], None, date(2026, 11, 13), report, kept_on=date(2026, 11, 14))
    with_ops = build_weekly([day], None, date(2026, 11, 13), report, kept_on=date(2026, 11, 14), operations=ops)
    assert "OPS BLOCK" not in plain["html"] and plain["html"] == with_ops["html"].replace(ops["html"], "")
    html = with_ops["html"]
    assert html.index("Findings") < html.index("OPS BLOCK") < html.index("COMPARISON")
    assert with_ops["text"].endswith("Operations, week from 2026-11-07: 1 item(s) need action; 1 incident(s)\n"
                                     "- a nightly session failed: 1x ...")


def _alarm_day(path, name="deployed"):
    from datetime import date
    from pathlib import Path
    from types import SimpleNamespace

    from nse_engine.books import Book, DayBook, PaperBook

    days = pd.bdate_range(end="2026-11-13", periods=len(path))
    engine = SimpleNamespace(config_hash=lambda: "deployed00000000")
    book = Book(name, Path(f"config/nse_engine_{name}.json"),
                SimpleNamespace(paper_start_date=days[0].date(), status="approved", engine=engine))
    return DayBook(book, PaperBook(name, pd.Series(path, index=days, dtype="float64"), pd.DatetimeIndex(days), None),
                   float(path[0]))


def _alarm_bands(name="deployed"):
    """Flat bands: the 5th percentile return is -5% at every horizon from 5 to 756 sessions."""
    import numpy as np

    pcts = [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 99]
    ret = sorted([-0.05 if p == 5 else v for p, v in zip(pcts, np.linspace(-0.10, 0.30, len(pcts)))])
    dd = sorted(np.linspace(-0.20, 0.0, len(pcts)).tolist())
    return {"percentiles": pcts, "books": {name: {"return": {"5": ret, "756": ret}, "drawdown": {"5": dd, "756": dd}}}}


def test_the_edge_alarm_trips_at_its_length_and_never_stops_the_nightly_email(monkeypatch):
    """MCA1 (owner's decision 16 Nov 2026): 65 sessions in a row with the return since the start below the 5th
    percentile; a line first in the nightly email and its subject; a broken bands file drops the line, not the email."""
    from datetime import date

    from nse_engine import books as bk
    from nse_engine import mc_bands

    n = bk.PAPER_ALARM_SESSIONS
    below = lambda k: [100.0] * 5 + [90.0] * k                       # noqa: E731 - -10% from session 5 on
    assert not bk.paper_alarms([_alarm_day(below(n - 1))], _alarm_bands())["deployed"]["tripped"]
    alarms = bk.paper_alarms([_alarm_day(below(n))], _alarm_bands())
    assert alarms["deployed"]["tripped"] and alarms["deployed"]["streak"] == n
    assert bk.paper_alarms([_alarm_day(below(n))], None) == {}                     # no bands: nothing claimed
    monkeypatch.setattr(mc_bands, "load", lambda path=None: _alarm_bands())
    report = bk.build_daily([_alarm_day(below(n))], None, date(2026, 11, 13))
    assert report["findings"][0].startswith("EDGE ALARM (MCA1): deployed's return since its start (-10.0%")
    assert f"for {n} sessions in a row" in report["findings"][0] and "EDGE ALARM in deployed" in report["subject"]
    calm = bk.build_daily([_alarm_day([100.0 + i for i in range(80)])], None, date(2026, 11, 13))
    assert "EDGE ALARM" not in calm["subject"] and not any("EDGE ALARM" in f for f in calm["findings"])

    def broken(path=None):
        raise ValueError("bands file unreadable")

    monkeypatch.setattr(mc_bands, "load", broken)
    still = bk.build_daily([_alarm_day(below(n))], None, date(2026, 11, 13))
    assert still["alarms"] == {} and "EDGE ALARM" not in still["subject"]


def test_the_edge_alarm_emails_once_a_run_on_the_night_it_trips():
    from datetime import date

    from nse_engine import books as bk
    from tools import books_report as br

    n, calls = bk.PAPER_ALARM_SESSIONS, []
    alarm = lambda streak: {"streak": streak, "since": date(2026, 8, 3), "return": -0.10, "return_pct": 2.0,  # noqa: E731
                            "tripped": streak >= n}

    def fake_alert(severity, key, title, lines, book=None, once=False):
        calls.append((severity, key, book, once))
        return True

    sent = br.send_alarms({"e4": alarm(n), "bonds": alarm(n + br.ALARM_GRACE_SESSIONS), "mcx": alarm(n - 1)},
                          fake_alert)
    assert sent == ["e4"]                                  # bonds tripped long ago: emailed then, not every night
    assert calls == [("WARNING", "mca1:e4:2026-08-03", "e4", True)]

    def down(*a, **k):
        raise RuntimeError("smtp down")

    assert br.send_alarms({"e4": alarm(n)}, down) == []    # never raises: the summary carries the line
