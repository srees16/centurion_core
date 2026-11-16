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
@pytest.mark.lesson('L-HB2',
                    pattern='F2',
                    date='2026-11-11',
                    source='HB2',
                    summary="The heartbeat counted weekdays, so a holiday raised a false 'has not traded' alarm")
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
@pytest.mark.lesson('L-FL1',
                    pattern='F1',
                    date='2026-11-14',
                    source='FL1',
                    summary='The backup-run gate called a night finished with the futures leg still waiting; five sessions were lost')
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


def test_the_space_sends_the_newsletter_on_saturdays_only_with_the_users_key(monkeypatch):
    """NL1: the newsletter's jobs run where the users' data key is, and a failure never stops the scheduler."""
    pytest.importorskip("apscheduler")
    import scheduler
    from api import newsletter

    jobs = []

    class Scheduler:
        def __init__(self, **_): pass
        def add_job(self, fn, trigger, id, **_): jobs.append((id, str(trigger)))
        def start(self): pass

    monkeypatch.setattr("apscheduler.schedulers.blocking.BlockingScheduler", Scheduler)
    monkeypatch.setattr(scheduler, "_init_cache_db", lambda: None)
    monkeypatch.delenv("CENTURION_GH_DISPATCH_TOKEN", raising=False)
    monkeypatch.setenv("CENTURION_USER_DATA_KEY", "key")
    scheduler.start_scheduler()
    assert [j for j, _ in jobs] == ["newsletter", "newsletter_retry"]
    assert all("day_of_week='sat'" in t for _, t in jobs) and ["10", "13"] == [t.split("hour='")[1][:2] for _, t in jobs]
    jobs.clear()
    monkeypatch.delenv("CENTURION_USER_DATA_KEY")
    scheduler.start_scheduler()
    assert jobs == []

    def down(*a, **k):
        raise RuntimeError("neon down")

    monkeypatch.setattr(newsletter, "send_week", down)
    scheduler._send_newsletter()                          # logged; the 13:00 retry tries again


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


# ---------------------------------------------------------------- RS1: the outside trigger's evening rescue

THU_1945 = datetime(2026, 10, 29, 19, 45, tzinfo=timezone(timedelta(hours=5, minutes=30)))


def _decide(monkeypatch, status, now=THU_1945, kite_login=None):
    import tools.paper_heartbeat as hb

    monkeypatch.setattr(hb, "night_status", lambda n=None: status)
    return hb.rescue_decision(now, kite_login=kite_login)


@pytest.mark.parametrize("status,login,start,needle", [
    (("books", "7 of 7 books behind 2026-10-29"), None, True, "books behind"),          # the 19:00 start never ran
    (("email", "the nightly email for 2026-10-29 has not gone"), None, True, "email"),
    (("unreadable", "Neon unreadable: OperationalError"), None, True, "Neon unreadable"),  # let a run find out
    (("live", "the live session has not run for 2026-10-29"), True, True, "live session"),  # a late Kite login
    (("live", "the live session has not run for 2026-10-29"), False, False, "no Kite login today"),
    (("futures", "1 futures leg(s) not marked at 2026-10-29"), None, False, "20:30 retry"),
    (("finished", "session 2026-10-29 finished"), None, False, "finished"),
])
@pytest.mark.lesson('L-RS1',
                    pattern='F2',
                    date='2026-11-14',
                    source='RS1',
                    summary='The nightly run had one on-time trigger; the documented second one was not firing')
def test_the_rescue_starts_a_run_only_for_what_a_run_can_still_do(monkeypatch, status, login, start, needle):
    decided, why = _decide(monkeypatch, status, kite_login=login)
    assert decided is start and needle in why


@pytest.mark.parametrize("now,why", [
    (datetime(2026, 10, 31, 19, 45, tzinfo=timezone(timedelta(hours=5, minutes=30))), "a Saturday"),
    (datetime(2026, 11, 10, 19, 45, tzinfo=timezone(timedelta(hours=5, minutes=30))), "an NSE holiday"),
    (datetime(2026, 10, 29, 14, 0, tzinfo=timezone(timedelta(hours=5, minutes=30))), "the market is still open"),
])
def test_the_rescue_does_nothing_when_no_session_closed_today(monkeypatch, now, why):
    decided, reason = _decide(monkeypatch, ("books", "7 of 7 books behind"), now=now)
    assert decided is False and "nothing is due" in reason, why


def test_night_status_and_session_finished_agree(monkeypatch):
    """The gate's answer is unchanged by the split: finished only when nothing is left."""
    import tools.paper_heartbeat as hb

    assert _finished(monkeypatch) == (True, f"session {FRI_SESSION} finished")
    assert hb.night_status(FRI_NIGHT) == (hb.FINISHED, f"session {FRI_SESSION} finished")


class _Api:
    """GitHub's Actions API as the rescue calls it: every call recorded."""

    def __init__(self, running=0, queued=0, dispatch=204, fail=()):
        self.counts, self.dispatch, self.fail, self.calls = {"in_progress": running, "queued": queued}, dispatch, set(fail), []

    def __call__(self, method, path, token, body=None):
        self.calls.append((method, path, body))
        if method in self.fail:
            raise OSError("api down")
        if method == "GET":
            return 200, {"total_count": self.counts[path.split("status=")[1].split("&")[0]]}
        return self.dispatch, None


def test_the_rescue_starts_the_nightly_workflow_with_its_reason(monkeypatch):
    import tools.paper_heartbeat as hb

    monkeypatch.setattr(hb, "night_status", lambda n=None: ("books", "7 of 7 books behind 2026-10-29"))
    api = _Api()
    assert hb.rescue(THU_1945, token="t", api=api) == ("started", "7 of 7 books behind 2026-10-29")
    method, path, body = api.calls[-1]
    assert (method, path) == ("POST", "workflows/nse-paper-trading.yml/dispatches")
    assert body == {"ref": "main", "inputs": {"reason": "rescue: 7 of 7 books behind 2026-10-29"}}


@pytest.mark.parametrize("api,token,outcome,needle", [
    (_Api(running=1), "t", "skipped", "a nightly run is in progress"),          # the 19:00 run is still going
    (_Api(queued=1), "t", "skipped", "a nightly run is queued"),
    (_Api(fail={"GET"}), "t", "started", "books behind"),                      # the run list unreadable: start it
    (_Api(dispatch=422), "t", "failed", "HTTP 422"),
    (_Api(fail={"POST"}), "t", "failed", "the dispatch failed: OSError"),
    (_Api(), "", "failed", "no GITHUB_TOKEN"),
])
def test_the_rescue_leaves_a_running_night_alone_and_reports_a_failed_start(monkeypatch, api, token, outcome, needle):
    import tools.paper_heartbeat as hb

    monkeypatch.setattr(hb, "night_status", lambda n=None: ("books", "7 of 7 books behind 2026-10-29"))
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    got, why = hb.rescue(THU_1945, token=token, api=api)
    assert got == outcome and needle in why
    if outcome == "skipped":
        assert not any(c[0] == "POST" for c in api.calls)


@pytest.mark.parametrize("outcome,exit_code,alert_key", [
    (("skipped", "session finished"), 0, None),
    (("started", "7 of 7 books behind"), 0, "nightly_rescued"),
    (("failed", "the dispatch answered HTTP 422"), 1, "nightly_rescue_failed"),
])
def test_the_rescue_command_alerts_when_it_acts_and_fails_the_job_only_when_it_could_not(monkeypatch, outcome,
                                                                                         exit_code, alert_key):
    import tools.paper_heartbeat as hb
    from services.notifications import alerts

    sent = []
    monkeypatch.setattr(hb, "rescue", lambda: outcome)
    monkeypatch.setattr(alerts, "alert", lambda sev, key, title, lines=(), **k: sent.append(key) or True)
    monkeypatch.setattr("sys.argv", ["paper_heartbeat", "--rescue"])
    assert hb.main() == exit_code and sent == ([alert_key] if alert_key else [])


def test_the_heartbeat_workflow_runs_the_rescue_only_when_asked():
    """Read as text: PyYAML is not in the core install the tests run on in CI."""
    import re
    from pathlib import Path

    text = (Path(__file__).resolve().parent.parent / ".github/workflows/paper-heartbeat.yml").read_text()
    assert re.search(r"\n      rescue:\n(?:\s+#.*\n)*\s+description: .*\n\s+required: false\n\s+type: boolean\n\s+default: false", text)
    assert re.search(r"\npermissions:\n  contents: read\n  actions: write", text)   # to read the nightly's runs, start one
    assert ("      - name: Check when the book last traded\n        if: ${{ !inputs.rescue }}\n"
            "        run: python -m tools.paper_heartbeat\n") in text
    assert ("      - name: Rescue - start tonight's run if it is needed\n        if: ${{ inputs.rescue }}\n"
            "        env:\n          GITHUB_TOKEN: ${{ github.token }}\n"
            "        run: python -m tools.paper_heartbeat --rescue") in text


@pytest.mark.parametrize("argv,env,severity,key,title,line", [
    (["123 (live book)"], {}, "CRITICAL", "session_failed:live book", "Centurion live book FAILED",
     "Open the failed run to see why: https://github.com/o/r/actions/runs/123"),
    (["123 (live book fam1)"], {}, "CRITICAL", "session_failed:live book fam1", "Centurion live book fam1 FAILED",
     "The paper books run in their own steps and email their own results."),
    (["123 (e4 book)"], {}, "WARNING", "session_failed:e4 book", "Centurion paper session FAILED (e4 book)",
     "Open the failed run to see why: https://github.com/o/r/actions/runs/123"),
    (["123"], {"CENTURION_LIVE_SKIPPED": "true"}, "CRITICAL", "session_failed:paper",
     "Centurion paper session FAILED, live book not run",
     "The live book did not run either: its step comes after the one that failed."),
    ([], {"GITHUB_RUN_ID": "77"}, "WARNING", "session_failed:paper", "Centurion paper session FAILED",
     "Open the failed run to see why: https://github.com/o/r/actions/runs/77"),
    (["https://example.org/run/9"], {}, "WARNING", "session_failed:paper", "Centurion paper session FAILED",
     "Open the failed run to see why: https://example.org/run/9"),
])
def test_a_failed_session_alerts_by_what_failed_and_never_masks_the_failure(monkeypatch, argv, env, severity, key,
                                                                              title, line):
    """25 Sep 2026: every run died before reaching the book and no mail went out.  A live failure is CRITICAL, a
    paper one a warning unless it also kept the live book from running; the tool itself always exits 0."""
    from services.notifications import alerts
    from tools import session_failed

    sent = []
    monkeypatch.setattr(alerts, "alert", lambda sev, k, t, lines, **kw: sent.append((sev, k, t, lines)) or False)
    monkeypatch.setattr("sys.argv", ["session_failed", *argv])
    for name in ("CENTURION_LIVE_SKIPPED", "GITHUB_RUN_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert session_failed.main() == 0                                     # even when the alert was not sent
    ((sev, k, t, lines),) = sent
    assert (sev, k, t) == (getattr(alerts, severity), key, title) and line in lines


# ── the week's operations (tools/ops_report.py, PL-16): incidents against the lessons that guard them ──

def _run(workflow, conclusion, created="2026-11-11T18:30:00Z", branch="main"):
    return {"workflow": workflow, "branch": branch, "event": "schedule", "conclusion": conclusion, "created": created,
            "url": f"https://github.com/o/r/actions/runs/{abs(hash(workflow + created)) % 1000}"}


def test_the_weeks_incidents_name_the_lesson_that_guards_each_and_flag_the_rest():
    from datetime import date

    from tools import ops_report as o

    runs = [_run("nse-paper-trading.yml", "failure"), _run("paper-heartbeat.yml", "failure", "2026-11-10T23:43:00Z"),
            _run("tests.yml", "failure"), _run("nse-paper-trading.yml", "failure", branch="feature/x"),
            _run("deploy.yml", "success"), _run("nse-paper-trading.yml", "cancelled")]
    books = {"deployed": {"alerts": {"heartbeat_stale": "2026-11-11", "session_failed:paper": "2026-11-12",
                                     "paper_g4_fail:deployed": "2026-11-12", "live_no_login:own": "2026-11-01"},
                          "sessions": [{"session_date": "2026-11-09", "notes": "REPLAYED 1 missed session(s), ..."},
                                       {"session_date": "2026-11-12", "notes": "CAUGHT UP 2 session(s) since ..."},
                                       {"session_date": "2026-11-02", "notes": "CAUGHT UP 1 session(s)"}]},
             "futures": {"alerts": {}, "sessions": [{"session_date": "2026-11-13",
                                                     "notes": "futures deferred (F&O bhavcopy not published yet)"}]}}
    found = o.incidents(runs, books, date(2026, 11, 9))
    kinds = sorted((i["type"], i["day"]) for i in found["incidents"])
    assert kinds == [("caught_up", "2026-11-12"), ("futures_deferred", "2026-11-13"), ("heartbeat_stale", "2026-11-11"),
                     ("replayed", "2026-11-09"), ("run_failed:nse-paper-trading.yml", "2026-11-12"),   # 18:30 UTC: IST next day
                     ("run_failed:paper-heartbeat.yml", "2026-11-11"), ("session_failed", "2026-11-12")]
    assert [r["type"] for r in found["risk_events"]] == ["paper_g4_fail"]        # a signal, not a fault
    out = o.summarise(found, [], {})
    assert out["head"] == "Operations, week from 2026-11-09: 2 item(s) need action; 7 incident(s); 1 risk event(s), in the books' reports"
    assert out["lines"] == [
        "a nse-paper-trading.yml run failed: 1x (2026-11-12; nse-paper-trading.yml): NO LESSON: decide whether it needs a guard",
        "a nightly session failed: 1x (2026-11-12; deployed): NO LESSON: decide whether it needs a guard"]
    assert out["info"] == ["guarded, no action: sessions caught up after a missed night 1x (L-NEO6); the futures leg was "
                           "deferred 1x (L-FL1); heartbeat: a paper book did not trade or could not be read 1x (L-HB2); "
                           "missed sessions replayed 1x (L-NEO6); a paper-heartbeat.yml run failed 1x (L-HB2)"]
    assert "<li style='margin-bottom:4px;'>a nse-paper-trading.yml run failed" in out["html"]


def test_a_quiet_week_is_one_line_and_an_unreadable_source_is_named_never_raised(monkeypatch, tmp_path):
    from datetime import date, datetime

    from tools import ops_report as o

    quiet = o.summarise(o.incidents([], {}, date(2026, 11, 9)), [{"text": "lessons: 42 of 50", "action": False}], {})
    assert quiet["head"] == "Operations, week from 2026-11-09: no incidents; nothing in the loop needs action"
    assert quiet["lines"] == [] and quiet["info"] == [] and "<ul" not in quiet["html"]
    guarded = o.summarise(o.incidents([], {"b": {"alerts": {"heartbeat_stale": "2026-11-11"}}}, date(2026, 11, 9)), [], {})
    assert guarded["head"] == "Operations, week from 2026-11-09: 1 incident(s), all guarded; nothing needs action"

    def down(*args, **kwargs):
        raise ConnectionError("no route")

    monkeypatch.setattr(o, "github_runs", down)
    monkeypatch.setattr(o, "neon_books", down)
    monkeypatch.setattr(o, "loop_health", lambda root, today: [{"text": "x", "action": False}])
    out = o.block(now=datetime(2026, 11, 16, 9, 0, tzinfo=o.IST))
    assert out["lines"] == ["GitHub's run history could not be read (ConnectionError): its incidents are not in this list",
                            "Neon could not be read (ConnectionError): its incidents are not in this list"]
    offline = o.block(now=datetime(2026, 11, 16, 9, 0, tzinfo=o.IST), offline=True)
    assert offline["head"].startswith("Operations (records only; GitHub and Neon not read), week from 2026-11-09")
    monkeypatch.setattr(o, "loop_health", lambda root, today: 1 / 0)
    assert "loop health could not be computed (ZeroDivisionError" in o.block(offline=True)["lines"][0]


def test_the_run_history_is_read_page_by_page_with_the_token_when_there_is_one():
    import json
    from datetime import date

    from tools import ops_report as o

    asked = []

    class Reply:
        def __init__(self, runs):
            self.body = json.dumps({"workflow_runs": runs}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return self.body

    run = {"path": ".github/workflows/nse-paper-trading.yml", "head_branch": "main", "event": "schedule",
           "conclusion": "failure", "created_at": "2026-11-12T14:00:00Z", "html_url": "u"}

    def opener(req, timeout):
        asked.append((req.full_url, req.headers.get("Authorization")))
        return Reply([run] * (100 if len(asked) == 1 else 3))

    runs = o.github_runs(date(2026, 11, 9), token="t0k", repo="o/r", opener=opener)
    assert len(runs) == 103 and runs[0] == {"workflow": "nse-paper-trading.yml", "branch": "main", "event": "schedule",
                                            "conclusion": "failure", "created": "2026-11-12T14:00:00Z", "url": "u"}
    import re

    assert [re.search(r"[?&]page=(\d+)", u).group(1) for u, _ in asked] == ["1", "2"] and asked[0][1] == "Bearer t0k"
    assert "created=%3E%3D2026-11-09" in asked[0][0]
    asked.clear()
    o.github_runs(date(2026, 11, 9), repo="o/r", opener=opener)
    assert asked[0][1] is None                                                   # a public repository: no token needed


def test_the_saturday_job_may_read_its_run_history_and_hands_the_report_a_token():
    from pathlib import Path

    text = (Path(__file__).resolve().parent.parent / ".github/workflows/nse-paper-trading.yml").read_text()
    job = text.split("\n  paper-trade:\n", 1)[1].split("\n  engine-window:\n", 1)[0]
    assert "permissions:" in job and "actions: read" in job and "contents: read" in job
    step = job.split("- name: Paper books - one Saturday email", 1)[1]
    assert "GITHUB_TOKEN: ${{ github.token }}" in step and "continue-on-error: true" in step
