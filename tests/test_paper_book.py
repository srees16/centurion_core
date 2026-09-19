"""The paper book: which rows belong to it, what a session records, who may write it.

Regression tests for three fixes that were each found in production:
  U10  fills stamped at the session open predate a book started later that day
  G5   execution costs must outlive the runner's local SQLite
  G7   only one runner may write the book
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


class TestWriterGuard:
    """G7: a second runner must not write the book."""

    @pytest.mark.parametrize("writer,allowed", [
        ("github_actions", False), ("hf_scheduler", True), ("", False), ("other", False)])
    def test_only_the_recorded_writer_may_run_a_paper_session(self, writer, allowed, monkeypatch):
        scheduler = pytest.importorskip("scheduler")   # needs the full app dependencies
        monkeypatch.setenv("CENTURION_NSE_ENGINE", "true")
        monkeypatch.delenv("CENTURION_NSE_ENGINE_HF_PAPER", raising=False)
        monkeypatch.setattr(pc, "get_paper_cloud", lambda: _cloud({"book_writer": writer}, []))
        assert scheduler._hf_paper_session_allowed()[0] is allowed

    def test_it_fails_closed_when_the_database_is_unreachable(self, monkeypatch):
        scheduler = pytest.importorskip("scheduler")
        monkeypatch.delenv("CENTURION_NSE_ENGINE_HF_PAPER", raising=False)

        def boom():
            raise RuntimeError("no database")

        monkeypatch.setattr(pc, "get_paper_cloud", boom)
        assert scheduler._hf_paper_session_allowed()[0] is False

    def test_explicit_opt_in_overrides(self, monkeypatch):
        scheduler = pytest.importorskip("scheduler")
        monkeypatch.setenv("CENTURION_NSE_ENGINE_HF_PAPER", "true")
        assert scheduler._hf_paper_session_allowed()[0] is True


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
