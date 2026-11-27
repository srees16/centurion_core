"""The process loop's own checks (docs/process_loop_plan.md): the briefing both assistants read, the lessons ledger
and its guards, the session preflight, the store-guards record, prior art and Phase 0, the stamped pre-registration,
leftover files, the journal against the tracker, and the trial counts against the registries."""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.lesson('L-RULES-DRIFT',
                    pattern='F3',
                    date='2026-11-14',
                    source='U41',
                    summary="Copies of the owner's working rules drifted apart")
def test_copilot_reads_the_same_briefing_as_claude():
    """The owner's working rules reach every coding assistant (14 Nov 2026): Copilot's file had drifted (it
    still said "never intraday" two weeks after the amendment, and had none of the standing rules).  CLAUDE.md
    is the source; the Copilot file is the same text under its own title."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    claude = (root / "CLAUDE.md").read_text().split("\n", 1)
    copilot = (root / ".github" / "copilot-instructions.md").read_text().split("\n", 1)
    assert claude[0].startswith("# Claude Code briefing") and copilot[0].startswith("# Copilot instructions")
    assert copilot[1] == claude[1], "CLAUDE.md changed: copy its text below the title into .github/copilot-instructions.md"


# ---------------------------------------------------------------- PL-02: the lessons ledger

def test_the_lessons_ledger_is_current_and_every_guard_exists():
    """docs/lessons.jsonl is generated from the guards (tools/lessons.py): it is out of date, a guard has gone
    or an open lesson is past its review date -> this fails."""
    from tools import lessons

    assert lessons.check() == []
    rows = lessons.build()
    assert len({r["id"] for r in rows}) == len(rows) >= 15
    assert all(r["guard_location"] for r in rows if r["status"] == "guarded")
    assert all(r["status"] != "guarded" for r in rows if not r["guard_location"])


def _lessons_repo(tmp_path, test_source):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text(test_source)
    return tmp_path


def test_a_lesson_comes_from_its_marker_and_a_second_test_adds_its_place(tmp_path, monkeypatch):
    from datetime import date

    from tools import lessons

    root = _lessons_repo(tmp_path, '''import pytest
pytestmark = pytest.mark.lesson("L-M", pattern="F5", date="2026-01-02", source="M1", summary="module")

@pytest.mark.lesson("L-A", pattern="F1", date="2026-01-03", source="A1", summary="a bug")
def test_one(): pass

class TestB:
    @pytest.mark.lesson("L-A")
    def test_two(self): pass
''')
    monkeypatch.setattr(lessons, "OTHER", [])
    monkeypatch.setattr(lessons, "OPEN", [{"id": "L-O", "date": "2026-01-04", "source": "O1", "pattern": "F6",
                                           "status": "open", "summary": "not yet", "review_by": "2026-02-01"}])
    rows = {r["id"]: r for r in lessons.build(root)}
    assert rows["L-A"]["guard_location"] == ["tests/test_x.py::test_one", "tests/test_x.py::TestB::test_two"]
    assert rows["L-M"]["guard_location"] == ["tests/test_x.py"] and rows["L-M"]["status"] == "guarded"
    assert rows["L-O"]["status"] == "open" and rows["L-O"]["guard_type"] == "none"
    (root / "docs").mkdir()
    assert lessons.check(root, today=date(2026, 1, 10)) == ["docs/lessons.jsonl is out of date: run `python -m tools.lessons build`"]
    (root / lessons.LEDGER_PATH).write_text(lessons.render(lessons.build(root)))
    assert lessons.check(root, today=date(2026, 1, 10)) == []
    overdue = lessons.check(root, today=date(2026, 2, 2))
    assert len(overdue) == 1 and "L-O is open past its review date" in overdue[0]


@pytest.mark.parametrize("source,other,open_,needle", [
    ('import pytest\n@pytest.mark.lesson("L-A", pattern="F1", date="2026-01-03", source="A1")\ndef test_one(): pass\n',
     [], [], "L-A: no summary"),
    ('import pytest\n@pytest.mark.lesson("L-A", pattern="F99", date="2026-01-03", source="A1", summary="s")\ndef test_one(): pass\n',
     [], [], "unknown pattern"),
    ("", [{"id": "L-G", "date": "2026-01-01", "source": "G1", "pattern": "F1", "guard_type": "preflight", "summary": "s",
           "guard_location": ["tools/gone.py::def nothing"]}], [], "its guard is gone"),
    ("", [], [{"id": "L-W", "date": "2026-01-01", "source": "W1", "pattern": "F1", "status": "waived", "summary": "s"}],
     "owner's decision id"),
])
def test_a_malformed_lesson_or_a_vanished_guard_stops_the_build(tmp_path, monkeypatch, source, other, open_, needle):
    from tools import lessons

    root = _lessons_repo(tmp_path, source or "def test_nothing(): pass\n")
    monkeypatch.setattr(lessons, "OTHER", other)
    monkeypatch.setattr(lessons, "OPEN", open_)
    with pytest.raises(ValueError, match=needle):
        lessons.build(root)
    assert needle in lessons.check(root)[0]


# ---------------------------------------------------------------- PL-04: the session preflight

def test_the_preflight_knows_what_a_run_started_now_would_do():
    """FL3 (14 Nov 2026): a manual nightly run on a Saturday ran only the weekly checkpoint.  The runner reads
    the weekday from its own clock, UTC on GitHub: Saturday 02:00 IST is still its Friday."""
    from datetime import datetime, timedelta, timezone

    from tools import preflight

    ist = timezone(timedelta(hours=5, minutes=30))
    assert "weekly checkpoint only" in preflight.runner_mode(datetime(2026, 11, 14, 20, 44, tzinfo=ist))   # the FL3 run
    assert "nightly session" in preflight.runner_mode(datetime(2026, 11, 14, 2, 30, tzinfo=ist))           # Friday's late cron
    assert "nightly session" in preflight.runner_mode(datetime(2026, 11, 16, 19, 0, tzinfo=ist))
    assert "weekly checkpoint only" in preflight.runner_mode(datetime(2026, 11, 15, 4, 0, tzinfo=ist))     # Sunday early IST


def test_the_preflight_dates_the_sixtieth_session_from_the_calendar():
    """'14 Dec' was written everywhere for RA1; 39 sessions on 13 Nov and one holiday (24 Nov) make it 15 Dec."""
    from datetime import date, datetime, timedelta, timezone

    from tools import preflight

    assert preflight.session_date(39, date(2026, 11, 13)) == date(2026, 12, 15)
    assert preflight.session_date(59, date(2026, 12, 14)) == date(2026, 12, 15)
    assert preflight.session_date(60, date(2026, 12, 15)) is None
    ist = timezone(timedelta(hours=5, minutes=30))
    sat = preflight.clock(datetime(2026, 11, 14, 20, 0, tzinfo=ist))
    assert (sat["trading_day"], sat["last_completed_session"], sat["next_session"]) == (False, "2026-11-13", "2026-11-16")
    noon = preflight.clock(datetime(2026, 11, 16, 12, 0, tzinfo=ist))
    assert (noon["last_completed_session"], noon["next_session"]) == ("2026-11-13", "2026-11-16")
    eve = preflight.clock(datetime(2026, 11, 23, 19, 0, tzinfo=ist))                # the day before a holiday
    assert (eve["last_completed_session"], eve["next_session"]) == ("2026-11-23", "2026-11-25")


def test_the_preflight_runs_offline_and_lists_the_owners_open_items():
    from tools import preflight

    lines = preflight.facts(offline=True)
    text = "\n".join(lines)
    assert lines[0].startswith("Now: ") and "A nightly run started now:" in text and "Trials: core " in text
    assert "not read (offline)" in text and "Journal: 0 verdict(s) not journaled" in text and "Lessons: " in text
    tracker = (" 4. NEEDS YOU [!]\n >>> SELL TEST, MON (ST2)\n     step\n >>> (was) OLD ITEM\n >>> SAT 31 OCT - YOUR ACTIONS\n"
               " >>> ZERODHA: QUESTIONS\n 5. IN PROGRESS\n >>> NOT MINE\n")
    assert preflight.owner_items(tracker) == ["SELL TEST, MON (ST2)", "ZERODHA: QUESTIONS"]
    assert "Store guards: " in text and "Cost rates: read against the published schedule on " in text


# ---------------------------------------------------------------- PL-13: the guards CI cannot run

def test_the_store_guards_last_run_is_on_record_and_a_stale_failed_or_skipped_one_is_flagged(tmp_path):
    from datetime import date

    from tools import guards

    assert guards.parse_summary("....\n9 passed, 742 deselected in 400.12s (0:06:40)\n") == {
        "passed": 9, "failed": 0, "skipped": 0}
    assert guards.parse_summary("1 failed, 7 passed, 1 skipped, 2 errors in 3s") == {"passed": 7, "failed": 3, "skipped": 1}
    assert guards.parse_summary("no tests ran") == {"passed": 0, "failed": 0, "skipped": 0}
    assert guards.status(tmp_path) == "! never run on this machine: python -m tools.guards"
    record = tmp_path / guards.RECORD
    record.parent.mkdir(parents=True)

    def ran(**counts):
        record.write_text(json.dumps({"when": "2026-11-14T22:30+05:30", "passed": 9, "failed": 0, "skipped": 0, **counts}))

    ran()
    assert guards.status(tmp_path, date(2026, 11, 21)) == (
        "last run Sat 14 Nov 2026: 9 passed, 0 failed, 0 skipped; due again by 21 Nov")
    late = guards.status(tmp_path, date(2026, 11, 22))
    assert late.startswith("! last run Sat 14 Nov 2026") and "over 7 days ago" in late
    ran(failed=1)
    assert guards.status(tmp_path, date(2026, 11, 15)).endswith("a guard failed")
    ran(skipped=2)
    assert "a store is missing" in guards.status(tmp_path, date(2026, 11, 15))
    record.write_text("not json")
    assert guards.status(tmp_path).startswith("! never run")


# ---------------------------------------------------------------- PL-05: prior art and Phase 0

def test_prior_art_finds_the_earlier_verdicts_on_an_idea():
    from tools import prereg

    hits = {h["id"]: h for h in prereg.prior_art(["open", "offer"])}
    assert hits["NS1-8"]["verdict"] == "FAIL"                                    # tested 9 Nov 2026
    assert "GF1" in {h["id"] for h in prereg.prior_art(["gap", "FADE"])}        # any case
    assert prereg.prior_art(["zzzz-no-such-idea"]) == []
    with pytest.raises(ValueError):
        prereg.prior_art([" "])


_PHASE0 = """Phase 0 (before anything else is written or downloaded):
  Prior art: NS1-8 (FAIL), GF1 (FAIL)
  Data: NSE filings; reachable by script and allowed by its terms: yes
  coverage 2008-25: 2011-25, 2008-10 missing; point-in-time: yes; delisted names: yes
  Cost model for this instrument: exists, model 4
  Phase 0 verdict: GO
"""


@pytest.mark.parametrize("header,body,problems", [
    ("## 5zz. NEW1: a study (NEW1; pre-registered 20 Nov 2026, 10:00 IST)", _PHASE0, []),
    ("## 5zz. NEW1: a study (NEW1; pre-registered 20 Nov 2026, 10:00 IST)", "Source / hypothesis: x\n",
     ["section 5zz: no Phase 0 block with a GO or STOP verdict"]),
    ("## 5zz. NEW1: a study (NEW1; pre-registered 20 Nov 2026, 10:00 IST)",
     _PHASE0.replace("Prior art: NS1-8 (FAIL), GF1 (FAIL)", "Prior art: <ids and verdicts>"),
     ["section 5zz: Phase 0 leaves 'Prior art:' unfilled"]),
    ("## 5zz. NEW1: a study (NEW1; pre-registered 20 Nov 2026, 10:00 IST)",
     _PHASE0.replace("Phase 0 verdict: GO", "Phase 0 verdict: <GO / STOP (reason)>"),
     ["section 5zz: no Phase 0 block with a GO or STOP verdict"]),
    ("## 5cb. GF1: the opening gap (GF1; pre-registered 07 Nov 2026, 00:56 IST)", "Source / hypothesis: x\n", []),   # before the rule
    ("## 6. Stage D - paper trading", "no pre-registration here\n", []),
])
def test_a_new_pre_registration_needs_its_phase_0_filled(header, body, problems):
    from tools import prereg

    assert prereg.check_phase0(f"# Plan\n\n{header}\n\n{body}\n## 9. Commands\n") == problems


def test_the_real_plan_and_the_protocols_template_carry_phase_0():
    from tools import prereg

    root = Path(__file__).resolve().parent.parent
    assert prereg.check_phase0((root / prereg.PLAN_PATH).read_text()) == []
    assert prereg.stamp_problems((root / prereg.PLAN_PATH).read_text()) == []
    template = (root / "docs" / "research_protocol.md").read_text().split("## Pre-registration template", 1)[1]
    assert all(field in template for field in prereg.PHASE0_FIELDS + ("Phase 0 verdict:",))


# ---------------------------------------------------------------- PL-08: the pre-registration is stamped, not typed

@pytest.mark.lesson("L-PREREG-TIME", pattern="F4", date="2026-11-06", source="BT7",
                    summary="Pre-registration times were typed by hand and three were in the future (BT7, BT7-S2, GF1)")
def test_a_pre_registration_is_stamped_from_the_clock_and_a_future_stamp_is_named():
    from datetime import datetime

    from tools import prereg

    now = datetime(2026, 11, 20, 10, 0, tzinfo=prereg.IST)
    plan = "# Plan\n\n## 5cb. GF1: gap (GF1; pre-registered 07 Nov 2026, 00:56 IST)\n\nold\n\n## 6. Stage D\n\nlater\n"
    new_plan, heading = prereg.add_section(plan, "NEW1", "a study", _PHASE0 + "Rules: x\n", now=now)
    assert heading == "## 5cc. NEW1: a study (India tracker NEW1; pre-registered 20 Nov 2026, 10:00 IST)"
    assert new_plan.index("## 5cb.") < new_plan.index(heading) < new_plan.index("## 6. Stage D")
    assert new_plan.replace(f"{heading}\n\n{_PHASE0}Rules: x\n\n", "") == plan       # added whole, nothing else moved
    assert prereg.check_phase0(new_plan) == [] and prereg.stamp_problems(new_plan, now) == []
    with pytest.raises(ValueError, match="no Phase 0 block"):
        prereg.add_section(plan, "NEW1", "a study", "Rules: x\n", now=now)
    with pytest.raises(ValueError, match="'Data:' unfilled"):
        prereg.add_section(plan, "NEW1", "a study", _PHASE0.replace("Data: NSE filings;", "Data: <source>;"), now=now)
    a_minute_before = datetime(2026, 11, 20, 9, 59, tzinfo=prereg.IST)
    assert prereg.stamp_problems(new_plan, a_minute_before) == [
        "section 5cc: pre-registered 20 Nov 2026, 10:00 IST, which is after now (20 Nov 2026, 09:59 IST)"]
    dated = "## 5cd. X: y (X; pre-registered 21 Nov 2026)\n"
    assert len(prereg.stamp_problems(dated, now)) == 1 and prereg.stamp_problems(dated, datetime(2026, 11, 21, 0, 1, tzinfo=prereg.IST)) == []


@pytest.mark.parametrize("ids,following", [([], "5a"), (["5a"], "5b"), (["5z"], "5aa"), (["5az", "5b"], "5ba"),
                                           (["5cb", "5by"], "5cc"), (["5zz"], "5aaa")])
def test_the_next_plan_section_follows_the_last(ids, following):
    from tools import prereg

    assert prereg.next_section("".join(f"## {i}. t\n\n" for i in ids) + "## 6. Stage D\n") == following


def test_the_template_comes_with_the_familys_count_and_the_prior_art():
    from nse_engine.validation import families
    from tools import prereg

    text = prereg.skeleton("intraday", ["gap", "fade"])
    n = families.load()["intraday"]["n"]
    assert f"Family and budget: intraday, <k> of no fixed cap configurations; N {n} -> <{n} + k>" in text
    assert "Prior art: " in text and "GF1 (FAIL)" in text
    assert "none found for: zzzz-no-such-idea" in prereg.skeleton("core", ["zzzz-no-such-idea"])
    assert prereg.check_phase0("## 5zz. X: y (X; pre-registered 20 Nov 2026)\n\n" + text)    # a template is not a section
    with pytest.raises(ValueError, match="unknown family"):
        prereg.family_line("no-such-family")
    protocol = (ROOT / "docs" / "research_protocol.md").read_text().split("## Pre-registration template", 1)[1]
    for line in prereg.TEMPLATE.splitlines():                                 # one template, in two places
        assert line.split(":")[0].strip() in protocol, line


# ---------------------------------------------------------------- PL-07: leftovers of a failed study

@pytest.mark.lesson("L-LEFTOVER", pattern="F13", date="2026-11-09", source="UI5",
                    summary="Empty modules of failed studies came back twice (index_events, promoter_buys, tail_hedge)")
def test_no_empty_source_file_is_left_in_the_code():
    """A failed idea leaves no code; an empty module is what a half-undone deletion leaves behind."""
    root = Path(__file__).resolve().parent.parent
    skip = {"myenv", "data", "research_inbox", "node_modules", ".git", "__pycache__", "frontend"}
    empty = [p.relative_to(root).as_posix() for p in root.rglob("*.py")
             if not skip & set(p.relative_to(root).parts) and p.name != "__init__.py" and p.stat().st_size == 0]
    assert empty == [], "delete these (after the owner's OK) or give them content"


@pytest.mark.lesson("L-C5", pattern="F10", date="2026-10-01", source="C5",
                    summary="The API could not start without Streamlit: a package's __init__ imported the removed UI")
def test_nothing_imports_the_removed_streamlit_ui():
    skip = {"myenv", "data", "research_inbox", "node_modules", ".git", "__pycache__", "frontend", "tests"}
    found = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*.py") if not skip & set(p.relative_to(ROOT).parts)
             and any(l.startswith(("import streamlit", "from streamlit")) for l in p.read_text(errors="ignore").splitlines())]
    assert found == []


def test_a_studys_files_are_listed_for_approval_and_never_deleted(tmp_path):
    from tools import study

    (tmp_path / "data" / "x").mkdir(parents=True)
    (tmp_path / "data" / "x" / "a.parquet").write_bytes(b"12345")
    (tmp_path / "script.py").write_text("print(1)\n")
    study.start("X1", ["data/x"], root=tmp_path)
    study.start("X1", ["script.py"], root=tmp_path)                             # a later path is added, not replaced
    assert study.files("X1", root=tmp_path) == [{"path": "data/x/a.parquet", "bytes": 5}, {"path": "script.py", "bytes": 9}]
    assert (tmp_path / "data" / "x" / "a.parquet").exists() and (tmp_path / "script.py").exists()
    with pytest.raises(ValueError, match="outside the repository"):
        study.start("X1", ["../elsewhere"], root=tmp_path)


# ---------------------------------------------------------------- PL-01, PL-06: the tracker against the journal

_TRACKER = """\
 7. BACKLOG
[ ] P2  X9   something queued: FAIL would not count here
--------------------------------------------------------------------------------
 8. DONE - most recent first
--------------------------------------------------------------------------------
14 Nov  FL9  A fix (your ask; uncommitted).  The futures leg FAILED once in the log; not a verdict:
             this entry states none in its opening.  Later text may say CLOSED or FAIL, in the open, without
             being read: the check reads the opening only, and this line is far past it, far past it, far past it,
             far past it, FAIL.
9 Nov   GF9  Opening-gap fade (plan 5cb; intraday N = 2): FAIL on its pass route.
             Second line of the entry.
9 Nov   AX9  A Medium post (pasted by you): not adopted, no code.
3 Nov   NS9  Round summary (plan 5bp): FAIL across the round.
1 Nov   OLD9 An old study (plan 5zz): FAIL, closed.
"""


@pytest.fixture
def journal_file(tmp_path):
    from nse_engine import journal as j

    path = tmp_path / "journal.csv"
    j.add_experiment("AX9", "2026-11-09", "AX9: a Medium post", "NOT ADOPTED", "nothing new", path=path)
    return j, path


def test_the_tracker_is_read_entry_by_entry_with_its_year(journal_file):
    from datetime import date

    j, _ = journal_file
    ents = j.tracker_entries(_TRACKER, today=date(2026, 11, 14))
    assert [(e["id"], e["date"]) for e in ents] == [("FL9", "2026-11-14"), ("GF9", "2026-11-09"), ("AX9", "2026-11-09"),
                                                   ("NS9", "2026-11-03"), ("OLD9", "2026-11-01")]
    assert ents[1]["text"].endswith("Second line of the entry.")
    assert j.tracker_entries("\n 8. DONE\n20 Dec  A1   x: FAIL.\n", today=date(2027, 1, 5))[0]["date"] == "2026-12-20"
    with pytest.raises(ValueError):
        j.tracker_entries("no such section")


@pytest.mark.parametrize("opening,verdict", [
    ("Study (plan 5x): FAIL, closed, code removed.", "FAIL"), ("Check (desk): fails, closed.", "CLOSED"),
    ("Brief (plan 5z): CLOSED at Phase 0.", "CLOSED"), ("A video (your ask): not adopted, no code.", "NOT ADOPTED"),
    ("A paper: does not fit, no code.", "NOT ADOPTED"), ("A lever (plan 5aj): PASS, in sample.", "PASS"),
    ("A fix (your ask).  Tests pass; the run failed once.", None), ("Decision: NO-GO on leverage.", "NO-GO"),
])
def test_a_verdict_is_read_from_an_entrys_opening(opening, verdict):
    from nse_engine import journal as j

    assert j.stated_verdict(opening) == verdict


@pytest.mark.lesson("L-JOURNAL", pattern="F3", date="2026-11-14", source="PL0",
                    summary="22 verdicts of 2-13 Nov never reached the metrics journal: the step was done by hand")
def test_a_verdict_in_the_tracker_without_a_journal_row_is_named(journal_file, monkeypatch):
    from datetime import date

    j, path = journal_file
    monkeypatch.setattr(j, "JOURNAL_EXEMPT", {"NS9": "a round summary"})
    found = j.check(_TRACKER, path=path, today=date(2026, 11, 14))
    assert found["unjournaled"] == [{"id": "GF9", "date": "2026-11-09", "verdict": "FAIL"}]   # not FL9, AX9, NS9, OLD9
    j.add_experiment("GF9", "2026-11-09", "GF9: opening-gap fade", "FAIL", path=path)
    assert j.check(_TRACKER, path=path, today=date(2026, 11, 14))["unjournaled"] == []


def test_a_row_from_the_cutoff_needs_its_post_mortem(journal_file):
    from datetime import date

    j, path = journal_file
    j.add_experiment("NEW1", "2026-11-20", "NEW1: a new study", "FAIL", path=path, phase_stopped="2 run", hours=3.5)
    found = j.check(_TRACKER, path=path, today=date(2026, 11, 21))["no_postmortem"]
    assert found == [{"ref": "NEW1", "date": "2026-11-20", "missing": ["trials_used", "earliest_stop", "lesson_ids"]}]
    j.add_experiment("NEW2", "2026-11-21", "NEW2: stopped early", "CLOSED", path=path, phase_stopped="0 feasibility",
                     hours=0.5, trials_used=0, earliest_stop="0 feasibility: the source refuses scripts",
                     lesson_ids="none: Phase 0 worked", idea_type="event study", market="India")
    assert [r["ref"] for r in j.check(_TRACKER, path=path, today=date(2026, 11, 21))["no_postmortem"]] == ["NEW1"]
    row = j.read_journal(path).set_index("ref").loc["NEW2"]
    assert row["trials_used"] == 0 and row["hours"] == 0.5 and row["idea_type"] == "event study"   # a zero is not blank
    with pytest.raises(ValueError, match="phase_stopped"):
        j.add_experiment("NEW3", "2026-11-22", "NEW3", "FAIL", path=path, phase_stopped="halfway")


def test_the_real_journal_holds_every_verdict_the_tracker_states():
    """The guard itself, on the repository's own records: a verdict written to the tracker and not journaled,
    or a new row without its post-mortem, fails here."""
    from nse_engine import journal as j

    found = j.check()
    assert found["unjournaled"] == [], "run `python -m nse_engine.journal experiment ...` for each (see `journal check`)"
    assert found["no_postmortem"] == [], "give each row its post-mortem (docs/research_protocol.md, section 8)"


# ---------------------------------------------------------------- PL-03: family trial counts

@pytest.mark.lesson("L-FAMILY-N", pattern="F3", date="2026-11-14", source="PL0",
                    summary="The protocol's trial counts were typed by hand and stale (options 17 for 20, intraday 0 for 2)")
def test_the_protocols_trial_counts_match_the_families_file():
    from nse_engine.validation import families

    assert families.problems(registries=False) == []
    stated = families.protocol_counts()
    assert stated["core"] == 61 or stated["core"] == families.load()["core"]["n"]
    assert all(n is not None for n in stated.values()), f"a protocol row states no 'N = <n>': {stated}"


def test_a_stale_protocol_row_or_a_blown_budget_is_named(tmp_path):
    from nse_engine.validation import families

    (tmp_path / "config").mkdir(); (tmp_path / "docs").mkdir()
    doc = {"families": {"options": {"registry": None, "protocol_row": "Options", "n": 20, "max_configurations": None},
                        "value": {"registry": "data/runs_value", "count": "distinct_configs", "protocol_row": "Value",
                                  "n": 4, "max_configurations": 3}}}
    (tmp_path / families.FAMILIES_PATH).write_text(__import__("json").dumps(doc))
    (tmp_path / families.PROTOCOL_PATH).write_text("| Family | Registry | Budget |\n| Options (U32) | x | N = 17 after O8 |\n"
                                                   "| Value (GT4) | y | N = 4 after GT4 |\n")
    found = families.problems(tmp_path)
    assert found == ["options: the protocol says N = 17, config/research_families.json says 20",
                     "value: N = 4 is past its budget of 3"]
    assert families.registry_count(doc["families"]["value"], tmp_path) is None      # no registry on this machine


@pytest.mark.slow
def test_the_families_file_matches_the_registries_on_the_research_machine():
    from nse_engine.deployment import REPO_ROOT
    from nse_engine.validation import families

    if not (REPO_ROOT / "data/nse_engine/runs").is_dir():
        pytest.skip("the run registries live only on the research machine")
    assert families.problems(registries=True) == []


# ---------------------------------------------------------------- the tests, by the functionality they guard

def test_every_test_file_says_which_functionality_it_guards(tmp_path):
    from tools import test_map

    assert test_map.problems() == []
    table = {r["area"]: r for r in test_map.rows()}
    assert all(r["tests"] > 0 for r in table.values()), "an area with no test"
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_new_thing.py").write_text("def test_x(): pass\n")
    found = test_map.problems(tmp_path)
    assert "tests/test_new_thing.py belongs to no area: add it to tools/test_map.py AREAS" in found
    report = {"files": {"api/main.py": {"summary": {"covered_lines": 50, "num_statements": 200}},
                        "auth/shared_session.py": {"summary": {"covered_lines": 50, "num_statements": 50}},
                        "nse_engine/costs.py": {"summary": {"covered_lines": 9, "num_statements": 10}}}}
    covered = {r["area"]: r["coverage"] for r in test_map.rows(coverage=report)}
    assert covered["Web app and API"] == 40.0 and covered["Engine, costs and data"] == 90.0 and covered["Options"] is None


def test_the_deploy_waits_for_the_tests_and_the_nightly_run_never_does():
    """Owner's call, 15 Nov 2026: nothing reaches the Space on a red suite; a broken test never stops exits."""
    flows = ROOT / ".github" / "workflows"
    deploy, tests = (flows / "deploy.yml").read_text(), (flows / "tests.yml").read_text()
    assert "uses: ./.github/workflows/tests.yml" in deploy and "needs: tests" in deploy
    assert "needs.tests.result == 'success' || needs.tests.result == 'skipped'" in deploy
    assert "workflow_call:" in tests and "tests-${{ github.workflow }}-${{ github.ref }}" in tests
    triggers = tests.split("\njobs:")[0]                 # owner's call, 16 Nov 2026: one run per change
    assert "\n  pull_request:" in triggers and "\n  push:" not in triggers
    assert 'CENTURION_NO_SKIPS: "1"' in tests
    assert "tests.yml" not in (flows / "nse-paper-trading.yml").read_text()


def test_a_run_that_must_not_skip_fails_when_a_test_skips(tmp_path):
    import subprocess
    import sys

    (tmp_path / "conftest.py").write_text((ROOT / "tests" / "conftest.py").read_text().split("ROOT = ")[0]
                                          + "import os\n" + "".join(
        "\n\n" + block for block in (ROOT / "tests" / "conftest.py").read_text().split("\n\n\n")
        if block.startswith(("# CENTURION_NO_SKIPS", "def pytest_runtest_logreport", "def pytest_sessionfinish"))))
    (tmp_path / "test_x.py").write_text("import pytest\ndef test_a(): pass\ndef test_b(): pytest.skip('no store')\n")

    def run(flag):
        return subprocess.run([sys.executable, "-m", "pytest", str(tmp_path), "-q", "-p", "no:cacheprovider",
                               "-c", "/dev/null"], capture_output=True, text=True,
                              env={**__import__("os").environ, "CENTURION_NO_SKIPS": flag})

    assert run("0").returncode == 0
    strict = run("1")
    assert strict.returncode == 1 and "test_x.py::test_b" in strict.stdout


# ── the loop's health (tools/ops_report.py, PL-17), on the repository's own records ──

def test_the_loops_health_warns_before_an_open_lesson_fails_the_suite():
    from datetime import date

    from tools import lessons, ops_report as o

    open_due = sorted((r["review_by"], r["id"]) for r in lessons.build() if r["status"] == "open")
    first_due = date.fromisoformat(open_due[0][0])
    early = o.loop_health(today=first_due - timedelta(days=o.DUE_SOON_DAYS + 1))
    assert not any("reach their review" in h["text"] for h in early)
    near = o.loop_health(today=first_due - timedelta(days=5))
    due = [h for h in near if "reach their review" in h["text"]]
    assert len(due) == 1 and due[0]["action"] and open_due[0][1] in due[0]["text"]
    assert near[0]["text"].startswith("lessons: ") and near[0]["action"] is False
    assert any(h["text"].startswith("post-mortems: ") for h in near)
    assert near[-1]["text"].startswith("cost-model error against contract notes: waits for PL-14")


def test_the_shared_settings_read_the_database_url_as_the_callers_did(monkeypatch):
    """RF0 phase 1: ``shared.settings.database_url`` replaces ``CENTURION_DATABASE_URL or DATABASE_URL`` in the API
    and the book export; an empty first variable falls through to the second, as before."""
    from datetime import timedelta

    from shared.clock import IST
    from shared.settings import database_url

    for name in ("CENTURION_DATABASE_URL", "DATABASE_URL"):
        monkeypatch.delenv(name, raising=False)
    assert database_url() == "" and IST.utcoffset(None) == timedelta(hours=5, minutes=30)
    monkeypatch.setenv("DATABASE_URL", "postgresql://second")
    assert database_url() == "postgresql://second"
    monkeypatch.setenv("CENTURION_DATABASE_URL", "")
    assert database_url() == "postgresql://second"
    monkeypatch.setenv("CENTURION_DATABASE_URL", "postgresql://first")
    assert database_url() == "postgresql://first"


def _launcher_tree(root):
    """A repository in miniature: one package, a script, a workflow that starts them, and code that dispatches."""
    (root / "pkg").mkdir()
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "mod.py").write_text("X = 1\ntry:\n    from os import sep as SEP\nexcept ImportError:\n    pass\n\n"
                                         "def f():\n    inner = 2\n")
    (root / "pkg" / "lazy.py").write_text("def __getattr__(name):\n    return name\n")
    (root / "runner.py").write_text('import subprocess\nsubprocess.run(["python", "-m", "pkg.mod"])\n'
                                    'WORKFLOW = "nightly.yml"  # dispatched by name\n'
                                    'dispatch("retired.yml")\n')
    wf = root / ".github" / "workflows"
    wf.mkdir(parents=True)
    (wf / "nightly.yml").write_text(
        "jobs:\n  a:\n    steps:\n"
        "      - run: python -m pkg.mod --flag\n"
        "      - run: python runner.py && python missing.py\n"
        "      - run: python -m pip install x && python -m pkg.gone\n"
        "      - run: X=$(python -c \"from pkg.mod import X, SEP; print(X)\")\n"
        "      - run: |\n"
        "          python - \"$A\" <<'EOF'\n"
        "          import os\n"
        "          from pkg.mod import (f,\n              inner)\n"
        "          from pkg.lazy import anything\n"
        "          EOF\n")


def test_the_refactor_check_finds_what_the_launchers_start(tmp_path):
    """RF0h: every repository module, script and inline import a workflow starts, and every workflow the code
    dispatches by name, resolves or is named; names bound inside try blocks count, a function's locals do not, a
    lazy module accepts any name, and outside modules (pip, os) are not the repository's to check."""
    from tools.refactor_check import entry_points

    _launcher_tree(tmp_path)
    assert entry_points(tmp_path) == {
        ".github/workflows/nightly.yml: -m pkg.gone": "module missing",
        ".github/workflows/nightly.yml: -m pkg.mod": "ok",
        ".github/workflows/nightly.yml: from pkg.lazy import anything": "ok",
        ".github/workflows/nightly.yml: from pkg.mod import X, SEP": "ok",
        ".github/workflows/nightly.yml: from pkg.mod import f, inner": "missing: inner",
        ".github/workflows/nightly.yml: missing.py": "file missing",
        ".github/workflows/nightly.yml: runner.py": "ok",
        "runner.py: -m pkg.mod": "ok",
        "runner.py: workflow nightly.yml": "ok",
        "runner.py: workflow retired.yml": "workflow missing"}


def test_the_refactor_check_reads_the_frontends_calls_and_matches_them_to_routes(tmp_path):
    """The client's calls and the literal backend URLs, with interpolated segments as values and query builders
    dropped; a relative fetch of the frontend's own /api/<name> route and a page link are not the backend's."""
    from tools.refactor_check import frontend_calls, match

    (tmp_path / "hooks").mkdir()
    (tmp_path / "hooks" / "use-x.ts").write_text(
        'api.get<Array<Foo>>("/api/v1/a/b");\n'
        "api.post(`/api/v1/items/${id}/run`, body);\n"
        "fetch(`${BACKEND_URL}/api/v1/rag/upload`, init);\n"
        "fetch(`${BACKEND_URL}/api/v1/rag/docs/${name}/chunks`);\n"
        "api.get(`/api/v1/list${qs}`);\n"
        'api.get("/api/v1/old?x=1");\n'
        'fetch("/api/paper-trading");\n'
        '<Link href="/ind-stocks/options" />\n')
    calls = frontend_calls(tmp_path)
    assert calls == ["/api/v1/a/b", "/api/v1/items/{}/run", "/api/v1/list", "/api/v1/old",
                     "/api/v1/rag/docs/{}/chunks", "/api/v1/rag/upload"]
    got = match(calls, ["GET /api/v1/a/b", "POST /api/v1/items/{item_id}/run", "GET /api/v1/list",
                        "POST /api/v1/rag/{rest:path}"])
    assert got == {"/api/v1/a/b": "/api/v1/a/b", "/api/v1/items/{}/run": "/api/v1/items/{item_id}/run",
                   "/api/v1/list": "/api/v1/list", "/api/v1/old": "", "/api/v1/rag/upload": "/api/v1/rag/{rest:path}",
                   "/api/v1/rag/docs/{}/chunks": "/api/v1/rag/{rest:path}"}


def test_the_refactor_check_names_every_kind_of_difference():
    """Problems: an entry point that stopped resolving or went, a new broken one, a route gone, a frontend call cut
    off, a replayed session or the replay's summary that moved.  Notes only: routes added, calls that never
    reached a route.  The same state against itself has no problem."""
    from copy import deepcopy

    from tools.refactor_check import differences

    base = {"surface": {"entry_points": {"wf: -m a": "ok", "wf: -m b": "ok", "wf: -m c": "module missing"},
                        "routes": ["GET /api/v1/a", "GET /api/v1/b/{x}"]},
            "replay": {"sessions": [{"day": "2024-05-02", "sent": 20, "equity": 300000.0},
                                    {"day": "2024-05-03", "sent": 0, "equity": 297675.11}],
                       "summary": {"orders_sent": 20, "verdict": "PASS"}}}
    calls = ["/api/v1/a", "/api/v1/b/{}", "/api/v1/carver"]
    known = "frontend calls that reach no route, before or now: /api/v1/carver"
    assert differences(base, deepcopy(base), calls) == ([], [known])
    now = deepcopy(base)
    now["surface"]["entry_points"] = {"wf: -m a": "missing: f", "wf: -m c": "module missing",
                                      "wf: -m d": "file missing"}
    now["surface"]["routes"] = ["GET /api/v1/a", "POST /api/v1/new"]
    now["replay"]["sessions"][1]["equity"] = 297675.12
    now["replay"]["summary"]["verdict"] = "FAIL"
    problems, notes = differences(base, now, calls)
    assert problems == ["entry point wf: -m a: missing: f", "entry point wf: -m b: gone",
                        "entry point wf: -m d: file missing (new)", "route gone: GET /api/v1/b/{x}",
                        "frontend call /api/v1/b/{} reaches nothing now (it reached /api/v1/b/{x})",
                        "replay 2024-05-03 equity: 297675.11 -> 297675.12", "replay summary verdict: PASS -> FAIL"]
    assert notes[0] == "routes added: POST /api/v1/new"
    del now["replay"]["sessions"][1]
    assert "replay: 1 sessions, the baseline had 2" in differences(base, now, calls)[0]
