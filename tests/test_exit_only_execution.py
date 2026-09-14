"""Exits for existing holdings run even when no new entries are allowed."""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from kite_connect.trading.auto_executor import AutoExecutor


def _executor():
    ex = AutoExecutor.__new__(AutoExecutor)  # skip Kite/config-heavy __init__
    ex.kite = MagicMock()
    ex.auto_place = True
    ex._carver_enabled = True
    ex._vol_target = object()
    ex._trade_monitor = None
    return ex


def _screened():
    return pd.DataFrame({"symbol": ["TCS", "INFY"], "close": [4000.0, 1500.0], "score": [80, 70]})


class ExitOnlyExecutionTests(unittest.TestCase):
    def _run(self, ex, **kwargs):
        def fake_plans(df, _cb):
            ex._pending_exits = {"HDFCBANK": "rank_exit"}
            ex.seen_symbols = list(df["symbol"])
            return []

        with patch.object(ex, "_current_cnc_holdings", return_value={"HDFCBANK": 10}), \
             patch.object(ex, "_generate_trade_plans", side_effect=fake_plans), \
             patch.object(ex, "_execute_rank_exits", return_value=[{"symbol": "HDFCBANK", "success": True}]) as exits, \
             patch("kite_connect.trading.order_service.get_holdings", return_value=[]):
            report = ex.run(pre_screened_df=_screened(), **kwargs)
        return report, exits

    def test_exits_run_when_entries_not_allowed(self):
        ex = _executor()
        report, exits = self._run(ex, signal_verdicts={"TCS": "STRONG_BUY"}, entries_allowed=False)
        exits.assert_called_once()
        self.assertEqual(report.exit_signals, {"HDFCBANK": "rank_exit"})
        self.assertEqual(report.plans_count, 0)
        self.assertEqual(ex.seen_symbols, [])  # no entry candidates were planned

    def test_exits_run_when_signal_filter_leaves_nothing(self):
        ex = _executor()
        report, exits = self._run(ex, signal_verdicts={"TCS": "SELL", "INFY": "SELL"})
        exits.assert_called_once()
        self.assertEqual(report.plans_count, 0)

    def test_no_holdings_returns_before_planning(self):
        ex = _executor()
        with patch.object(ex, "_current_cnc_holdings", return_value={}), \
             patch.object(ex, "_generate_trade_plans") as plans:
            ex.run(pre_screened_df=_screened(), signal_verdicts={"TCS": "SELL", "INFY": "SELL"})
        plans.assert_not_called()


if __name__ == "__main__":
    unittest.main()
