"""The alarm for a day that silently did not trade (18 Sep 2026)."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from tools.paper_heartbeat import check, sessions_behind

IST = timezone(timedelta(hours=5, minutes=30))
FRI, SAT, SUN, MON, TUE = (date(2026, 9, 18), date(2026, 9, 19), date(2026, 9, 20),
                           date(2026, 9, 21), date(2026, 9, 22))


def _ist(d: date, hh: int = 0, mm: int = 0) -> datetime:
    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=IST)


@pytest.mark.parametrize("latest,now,expected,why", [
    (FRI, _ist(FRI, 20), 0, "traded today, after the close"),
    (date(2026, 9, 17), _ist(FRI, 12), 0, "Friday's session is not over at noon"),
    (FRI, _ist(SAT, 0, 15), 0, "Saturday: nothing was due"),
    (FRI, _ist(MON, 0, 15), 0, "Monday after midnight: Monday's session is not due yet"),
    (FRI, _ist(TUE, 0, 15), 1, "Monday's night run is still due (late or retried)"),
    (FRI, _ist(date(2026, 9, 23), 0, 15), 2, "Monday and Tuesday were lost"),
    (date(2026, 9, 11), _ist(FRI, 22), 4, "a whole week missed; Mon 14 Sep was a holiday, not a session"),
    (date(2026, 11, 9), _ist(date(2026, 11, 11), 5, 13), 0,
     "11 Nov 2026's false alarm: Tue 10 Nov an NSE holiday, Wednesday's session not yet due"),
    (date(2026, 11, 9), _ist(date(2026, 11, 12), 5, 13), 1, "Wednesday 11 Nov's run is due by Thursday morning"),
    (None, _ist(FRI, 20), 0, "no book yet"),
])
def test_sessions_behind(latest, now, expected, why):
    assert sessions_behind(latest, now) == expected, why


class _Cloud:
    def __init__(self, latest): self._latest = latest
    def read_state(self): return {"epoch": "2026-09-17T05:16:25+00:00", "last_run_at": "x",
                                  "book_writer": "github_actions"}
    def read_snapshots(self, since_epoch=True):
        return pd.DataFrame({"date": [self._latest], "equity": [3_577_634.0]})


def _at(monkeypatch, latest, today):
    import tools.paper_heartbeat as hb
    monkeypatch.setattr("database.paper_cloud.get_paper_cloud", lambda: _Cloud(latest))

    class FrozenDatetime(hb.datetime):
        @classmethod
        def now(cls, tz=None): return hb.datetime.combine(today, hb.datetime.min.time(), tz)

    monkeypatch.setattr(hb, "datetime", FrozenDatetime)
    return hb.check()


def test_a_book_that_traded_today_is_healthy(monkeypatch):
    result = _at(monkeypatch, "2026-09-18", FRI)
    assert result["ok"] and not result["stale"] and result["sessions_behind"] == 0


def test_one_session_still_due_is_tolerated(monkeypatch):
    result = _at(monkeypatch, "2026-09-18", TUE)        # Tuesday 00:00: Monday's night run still due
    assert result["ok"] and not result["stale"]
    assert "one NSE session behind" in result["reason"]


def test_a_holiday_before_today_is_not_a_missed_session(monkeypatch):
    result = _at(monkeypatch, "2026-11-09", date(2026, 11, 11))   # the 11 Nov 2026 false alarm
    assert result["ok"] and not result["stale"] and result["sessions_behind"] == 0


def test_two_missed_sessions_raise_the_alarm(monkeypatch):
    result = _at(monkeypatch, "2026-09-18", date(2026, 9, 23))
    assert result["stale"] and not result["ok"]
    assert "missed" in result["reason"]


def test_no_database_is_treated_as_stale(monkeypatch):
    monkeypatch.setattr("database.paper_cloud.get_paper_cloud", lambda: None)
    assert check()["stale"] is True


def test_a_database_error_is_reported_not_raised(monkeypatch):
    """A blip must fail the run with a readable reason, not a traceback."""
    def boom():
        raise RuntimeError("could not connect to server")

    monkeypatch.setattr("database.paper_cloud.get_paper_cloud", boom)
    result = check()
    assert result["stale"] is True
    assert "database unreachable" in result["reason"]


FRI_NIGHT = datetime(2026, 10, 30, 20, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))
FRI_SESSION = "2026-10-30"


def _finished(monkeypatch, *, deployed=None, other=None, live=None, mode="dry_run", started_later=False):
    """The gate on a Friday night, with Neon holding the given state per schema."""
    from types import SimpleNamespace as Book

    from kite_connect.trading.live_session import LIVE_DRY_LAST_SESSION_KEY, live_schema
    from nse_engine.books import DAILY_STATE_KEY
    from tools.paper_heartbeat import session_finished

    done = {"engine_last_session": FRI_SESSION}
    states = {None: {**done, DAILY_STATE_KEY: FRI_SESSION, **(deployed or {})},
              "e4": {**done, **(other or {})},
              live_schema(): {LIVE_DRY_LAST_SESSION_KEY: FRI_SESSION, **(live or {})}}

    class Cloud:
        def __init__(self, mgr, schema=None): self.schema = schema
        def read_state(self): return states[self.schema]

    def book(name, schema, start="2026-10-01"):
        return Book(name=name, schema=schema, deployment=Book(paper_start_date=date.fromisoformat(start)))

    books = [book("deployed", None), book("e4", "e4", "2027-01-01" if started_later else "2026-10-01")]
    monkeypatch.setenv("CENTURION_LIVE_MODE", mode)
    monkeypatch.setattr("database.connection.get_db_manager", lambda: object())
    monkeypatch.setattr("database.paper_cloud.PaperCloudSync", Cloud)
    monkeypatch.setattr("nse_engine.books.discover_books", lambda: books)
    return session_finished(FRI_NIGHT)


def test_a_finished_night_stops_the_backup_runs(monkeypatch):
    assert _finished(monkeypatch) == (True, f"session {FRI_SESSION} finished")


@pytest.mark.parametrize("state,why", [
    ({"deployed": {"engine_last_session": "2026-10-29"}}, "1 of 2 books behind"),
    ({"other": {"engine_last_session": "2026-10-29"}}, "1 of 2 books behind"),
    ({"other": {"engine_last_session": None}}, "1 of 2 books behind"),
    ({"live": {"live_dry_run_last_session": "2026-10-29"}}, "live session has not run"),
    ({"deployed": {"daily_summary_sent": "2026-10-29"}}, "nightly email"),
])
def test_anything_left_undone_lets_the_backup_run_go(monkeypatch, state, why):
    finished, reason = _finished(monkeypatch, **state)
    assert not finished and why in reason


def _finished_with_a_futures_book(monkeypatch, ledger):
    """Friday night with a levered book whose share book is done and whose futures ledger is ``ledger``."""
    from types import SimpleNamespace as Book

    from kite_connect.trading.futures_overlay import STATE_KEY
    from kite_connect.trading.live_session import LIVE_DRY_LAST_SESSION_KEY, live_schema
    from nse_engine.books import DAILY_STATE_KEY
    from tools.paper_heartbeat import session_finished

    done = {"engine_last_session": FRI_SESSION}
    states = {None: {**done, DAILY_STATE_KEY: FRI_SESSION},
              "futures": {**done, **({} if ledger is None else {STATE_KEY: ledger})},
              live_schema(): {LIVE_DRY_LAST_SESSION_KEY: FRI_SESSION}}

    class Cloud:
        def __init__(self, mgr, schema=None): self.schema = schema
        def read_state(self): return states[self.schema]

    start = date(2026, 10, 1)
    books = [Book(name="deployed", schema=None, deployment=Book(paper_start_date=start)),
             Book(name="futures", schema="futures",
                  deployment=Book(paper_start_date=start, futures_overlay={"leverage": 1.75}))]
    monkeypatch.setenv("CENTURION_LIVE_MODE", "dry_run")
    monkeypatch.setattr("database.connection.get_db_manager", lambda: object())
    monkeypatch.setattr("database.paper_cloud.PaperCloudSync", Cloud)
    monkeypatch.setattr("nse_engine.books.discover_books", lambda: books)
    return session_finished(FRI_NIGHT)


@pytest.mark.parametrize("ledger", [
    '{"last_session": "2026-10-29", "history": []}',      # the F&O file came late: the leg was deferred
    None,                                                 # its first night, deferred: no ledger yet
    '{"history": []}',                                    # a ledger that never marked a session
    "not json",                                           # unreadable is not finished
])
def test_a_futures_leg_not_marked_tonight_keeps_the_backup_runs_on(monkeypatch, ledger):
    """FL1 (14 Nov 2026): the backups were stopped with the futures leg deferred, and the futures book lost
    five sessions (2-4 and 12-13 Nov)."""
    finished, why = _finished_with_a_futures_book(monkeypatch, ledger)
    assert not finished and why == f"1 futures leg(s) not marked at {FRI_SESSION} (NSE's F&O file was late)"


def test_a_futures_leg_marked_tonight_lets_the_night_finish(monkeypatch):
    ledger = '{"last_session": "%s", "history": [{"date": "%s"}]}' % (FRI_SESSION, FRI_SESSION)
    assert _finished_with_a_futures_book(monkeypatch, ledger) == (True, f"session {FRI_SESSION} finished")


def test_a_no_login_skip_is_not_finished(monkeypatch):
    """A later Kite login can still run the session, so the backups stay on (unlike ``nightly_health``)."""
    from kite_connect.trading.live_session import LIVE_DRY_LAST_SESSION_KEY, LIVE_SKIP_NOTIFIED_KEY

    finished, _ = _finished(monkeypatch, live={LIVE_DRY_LAST_SESSION_KEY: "2026-10-29",
                                               LIVE_SKIP_NOTIFIED_KEY: FRI_SESSION})
    assert not finished


def test_the_live_key_is_ignored_with_the_live_book_off(monkeypatch):
    assert _finished(monkeypatch, mode="", live={"live_dry_run_last_session": "2026-10-29"})[0]


def test_a_book_not_started_is_not_waited_for(monkeypatch):
    assert _finished(monkeypatch, other={"engine_last_session": None}, started_later=True)[0]


def test_unreadable_neon_lets_the_backup_run_go_and_names_no_host(monkeypatch):
    from tools.paper_heartbeat import session_finished

    def boom():
        raise RuntimeError("could not connect to secret-host.neon.tech")

    monkeypatch.setattr("database.connection.get_db_manager", boom)
    finished, why = session_finished(FRI_NIGHT)
    assert not finished and why == "Neon unreadable: RuntimeError"


def test_the_gate_writes_its_output_only_when_finished(monkeypatch, tmp_path):
    import tools.paper_heartbeat as hb

    out = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setattr("sys.argv", ["paper_heartbeat", "--session-finished"])
    monkeypatch.setattr(hb, "session_finished", lambda: (False, "behind"))
    assert hb.main() == 0 and not out.exists()
    monkeypatch.setattr(hb, "session_finished", lambda: (True, "done"))
    assert hb.main() == 0 and out.read_text() == "session_finished=true\n"


class TestSmtpCredentials:
    """Gmail App Passwords are shown in groups of four; copying brings spaces."""

    @pytest.mark.parametrize("raw,why", [
        ("abcd efgh ijkl mnop", "as Google displays it"),
        ("abcd efgh ijkl mnop", "copied with non-breaking spaces"),
        ("abcdefghijklmnop\n", "trailing newline from a paste"),
        ("  abcdefghijklmnop  ", "padded"),
        ("abcdefghijklmnop", "already clean"),
    ])
    def test_credentials_are_stripped_to_something_smtplib_can_send(self, raw, why, monkeypatch):
        from services.notifications.manager import _smtp_settings

        monkeypatch.setenv("CENTURION_EMAIL_USER", " bot@example.com ")
        monkeypatch.setenv("CENTURION_EMAIL_PASS", raw)
        host, port, user, password = _smtp_settings()
        assert password == "abcdefghijklmnop", why
        assert user == "bot@example.com"
        password.encode("ascii")        # smtplib does this; a stray NBSP used to raise here
        assert port == 587 and host == "smtp.gmail.com"


@pytest.mark.parametrize("late,expected", [
    (1, "hf scheduler retry: 1 futures leg(s) not marked"),      # FL1: the leg is waiting for NSE's late file
    (0, None),                                                   # all done: the retry stays quiet, as before
    (RuntimeError("neon down"), None),                           # unknown: left to the backup crons
])
def test_the_evening_retry_starts_a_run_only_for_a_futures_leg_still_waiting(monkeypatch, late, expected):
    """The share books are done at 19:00; the 20:30 retry used to do nothing whatever the futures leg's state."""
    pytest.importorskip("apscheduler")                           # the scheduler module is the app's, not the core's
    import scheduler
    import tools.paper_heartbeat as hb

    class Cloud:
        def read_state(self): return {"engine_last_session": "9999-12-31"}
        def sync_state(self, state): return True

    def behind():
        if isinstance(late, Exception):
            raise late
        return late

    sent = []
    monkeypatch.setenv("CENTURION_GH_DISPATCH_TOKEN", "t")
    monkeypatch.setattr("database.paper_cloud.get_paper_cloud", lambda: Cloud())
    monkeypatch.setattr(hb, "futures_behind_tonight", behind)
    monkeypatch.setattr(scheduler, "_github_dispatch", lambda wf, inputs, token: sent.append(inputs["reason"]) or 204)
    monkeypatch.setattr(scheduler, "_save_run", lambda *a, **k: None)
    scheduler._dispatch_nse_paper_workflow()
    assert sent == ([expected] if expected else [])


def test_futures_behind_tonight_counts_only_started_levered_books(monkeypatch):
    from types import SimpleNamespace as Book

    from kite_connect.trading.futures_overlay import STATE_KEY
    import tools.paper_heartbeat as hb

    states = {None: {}, "futures": {STATE_KEY: '{"last_session": "2026-10-29"}'}, "later": {}}

    class Cloud:
        def __init__(self, mgr, schema=None): self.schema = schema
        def read_state(self): return states[self.schema]

    dep = lambda start, fut: Book(paper_start_date=date.fromisoformat(start), futures_overlay=fut)
    books = [Book(name="deployed", schema=None, deployment=dep("2026-10-01", None)),
             Book(name="futures", schema="futures", deployment=dep("2026-10-01", {"leverage": 1.75})),
             Book(name="later", schema="later", deployment=dep("2027-01-01", {"leverage": 1.75}))]
    monkeypatch.setattr("database.connection.get_db_manager", lambda: object())
    monkeypatch.setattr("database.paper_cloud.PaperCloudSync", Cloud)
    monkeypatch.setattr("nse_engine.books.discover_books", lambda: books)
    assert hb.futures_behind_tonight(FRI_NIGHT) == 1
    states["futures"] = {STATE_KEY: '{"last_session": "%s"}' % FRI_SESSION}
    assert hb.futures_behind_tonight(FRI_NIGHT) == 0
