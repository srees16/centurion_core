"""The alarm for a day that silently did not trade (18 Sep 2026)."""
from __future__ import annotations

from datetime import date

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
