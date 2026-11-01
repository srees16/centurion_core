"""The alarm for a day that silently did not trade (18 Sep 2026)."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from tools.paper_heartbeat import check, weekdays_behind

FRI, SAT, SUN, MON, TUE = (date(2026, 9, 18), date(2026, 9, 19), date(2026, 9, 20),
                           date(2026, 9, 21), date(2026, 9, 22))


@pytest.mark.parametrize("latest,today,expected,why", [
    (FRI, FRI, 0, "traded today"),
    (FRI, SAT, 0, "Saturday: nothing was due"),
    (FRI, SUN, 0, "Sunday"),
    (FRI, MON, 1, "Monday evening before the run, or a Monday holiday"),
    (FRI, TUE, 2, "Monday was lost"),
    (date(2026, 9, 14), FRI, 4, "a whole week missed"),
    (None, FRI, 0, "no book yet"),
])
def test_weekdays_behind(latest, today, expected, why):
    assert weekdays_behind(latest, today) == expected, why


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
    assert result["ok"] and not result["stale"] and result["weekdays_behind"] == 0


def test_one_missed_weekday_is_tolerated_as_a_holiday(monkeypatch):
    result = _at(monkeypatch, "2026-09-18", MON)
    assert result["ok"] and not result["stale"]
    assert "holiday" in result["reason"]


def test_two_missed_weekdays_raise_the_alarm(monkeypatch):
    result = _at(monkeypatch, "2026-09-18", TUE)
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
