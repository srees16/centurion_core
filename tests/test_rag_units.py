"""The RAG engine's helpers that every question passes through (refactor R3, 25 Nov 2026).

These cases were the self-tests inside the production modules (``python -m rag_pipeline...``), moved here
unchanged in substance:
- ``utils.time_budget``: the per-question time budget, its thresholds, cut-off messages and environment factory;
- ``core.query_classifier``: intent, pipeline stages, the code hint and the strict-code override that the fast
  path and the retriever read;
- ``core.retriever``'s two switches the query engine reads on every question: fast mode and simple-query mode.

And one guard from the retirement of unused RAG modules (R4, 25 Nov 2026): every public name of the package
still points at code that exists.
"""

from __future__ import annotations

import ast
import time
from pathlib import Path

import pytest

from rag_pipeline.core.query_classifier import (
    classify_intent,
    classify_pipeline_stages,
    classify_query,
    needs_code,
)
from rag_pipeline.core.retriever import is_fast_mode, is_simple_query
from rag_pipeline.utils.time_budget import (
    DEFAULT_LLM_BUDGET_S,
    DEFAULT_QUERY_BUDGET_S,
    DEFAULT_RETRIEVAL_SLOW_S,
    TimeBudget,
    create_time_budget,
)

BUDGET_ENV = ("CENTURION_RAG_QUERY_BUDGET", "CENTURION_RAG_RETRIEVAL_SLOW", "CENTURION_RAG_LLM_BUDGET")


# ---------------------------------------------------------------- time budget
def test_the_budget_defaults_and_its_state_before_start():
    assert (DEFAULT_QUERY_BUDGET_S, DEFAULT_RETRIEVAL_SLOW_S, DEFAULT_LLM_BUDGET_S) == (300, 30, 300)
    tb = TimeBudget()
    assert (tb.total_budget_s, tb.retrieval_slow_s, tb.llm_budget_s) == (300, 30, 300)
    assert tb.elapsed == 0.0 and tb.remaining == 300.0 and not tb.is_expired()
    custom = TimeBudget(total_budget_s=60, retrieval_slow_s=5, llm_budget_s=30)
    assert (custom.total_budget_s, custom.retrieval_slow_s, custom.llm_budget_s) == (60, 5, 30)


def test_start_runs_the_clock_and_returns_the_budget():
    tb = TimeBudget(total_budget_s=5.0).start()
    assert tb.elapsed > 0 and tb.remaining < 5.0 and not tb.is_expired()
    assert tb.start() is tb


def test_a_spent_budget_is_expired_with_nothing_remaining():
    zero = TimeBudget(total_budget_s=0.0).start()
    assert zero.is_expired() and zero.remaining == 0.0
    tiny = TimeBudget(total_budget_s=0.001).start()
    time.sleep(0.005)
    assert tiny.is_expired()


def test_a_slow_retrieval_is_flagged_and_stays_flagged():
    tb = TimeBudget(retrieval_slow_s=10.0).start()
    assert not tb.retrieval_exceeded(5.0) and not tb.was_retrieval_slow
    assert tb.retrieval_exceeded(25.0) and tb.was_retrieval_slow
    assert tb.retrieval_exceeded(25.0)


def test_the_llm_is_cut_off_by_its_own_budget_or_the_question_s():
    tb = TimeBudget(total_budget_s=120.0, llm_budget_s=90.0).start()
    assert not tb.llm_exceeded(10.0) and not tb.was_llm_aborted
    assert tb.llm_exceeded(91.0) and tb.was_llm_aborted
    spent = TimeBudget(total_budget_s=0.001, llm_budget_s=90.0).start()
    time.sleep(0.005)
    assert spent.llm_exceeded(0.0)


def test_the_llm_s_remaining_time_is_the_smaller_of_the_two_budgets():
    by_question = TimeBudget(total_budget_s=50.0, llm_budget_s=90.0).start().llm_remaining()
    assert 0 < by_question <= 50.0
    assert TimeBudget(total_budget_s=200.0, llm_budget_s=30.0).start().llm_remaining() <= 30.0


def test_the_cut_off_message_names_what_ran_out():
    tb = TimeBudget(total_budget_s=120.0, llm_budget_s=90.0).start()
    assert tb.cutoff_message() == ""
    tb.llm_exceeded(100.0)
    llm = tb.cutoff_message()
    assert "90s" in llm and "truncated" in llm.lower()
    spent = TimeBudget(total_budget_s=0.001).start()
    time.sleep(0.005)
    assert "budget" in spent.cutoff_message().lower()


def test_the_budget_reports_itself():
    d = TimeBudget(total_budget_s=60.0, retrieval_slow_s=15.0, llm_budget_s=45.0).start().as_dict()
    assert (d["total_budget_s"], d["retrieval_slow_s"], d["llm_budget_s"]) == (60.0, 15.0, 45.0)
    assert "elapsed_s" in d and "remaining_s" in d
    assert all(isinstance(d[k], bool) for k in ("expired", "retrieval_switched", "llm_aborted"))
    r = repr(TimeBudget(total_budget_s=120.0).start())
    assert "TimeBudget" in r and "elapsed" in r and "remaining" in r


def test_the_factory_reads_the_environment_and_an_argument_wins(monkeypatch):
    for k, v in zip(BUDGET_ENV, ("60", "10", "45"), strict=True):
        monkeypatch.setenv(k, v)
    tb = create_time_budget()
    assert (tb.total_budget_s, tb.retrieval_slow_s, tb.llm_budget_s) == (60.0, 10.0, 45.0)
    assert create_time_budget(total_budget_s=99).total_budget_s == 99.0


# ---------------------------------------------------------------- query classifier
@pytest.mark.parametrize("query, intent", [
    ("Write a function to compute returns", "code_generation"),
    ("Implement a class for portfolio rebalancing", "code_generation"),
    ("Give me code for z-score signal", "code_generation"),
    ("Refactor the risk module", "code_generation"),
    ("Debug the stop-loss error", "code_generation"),
    ("Build a data pipeline for OHLCV", "code_generation"),
    ("Optimize the momentum strategy", "analysis"),
    ("Reduce the drawdown on this portfolio", "analysis"),
    ("How can I improve the Sharpe?", "analysis"),
    ("Compare mean reversion vs momentum", "analysis"),
    ("Evaluate this strategy performance", "analysis"),
    ("Backtest this on 2020 data", "analysis"),
    ("How much drawdown does this strategy have?", "analysis"),
    ("Why does the momentum factor work?", "conceptual"),
    ("Explain the Kelly criterion", "conceptual"),
    ("What is the Sharpe ratio?", "conceptual"),
    ("Define alpha in the context of returns", "conceptual"),
    ("Describe mean reversion strategy", "conceptual"),
    ("Difference between Sharpe and Sortino", "conceptual"),
    ("Hello", "conceptual"),                                   # no rule matches: the default
    ("", "conceptual"),
    ("Write code to optimize the portfolio", "code_generation"),        # code generation outranks analysis
    ("Implement an improvement to reduce drawdown", "code_generation"),
])
def test_intent(query, intent):
    assert classify_intent(query) == intent


@pytest.mark.parametrize("query, stage", [
    ("Calculate the Sharpe ratio for momentum returns", "evaluation"),
    ("Reduce the max drawdown risk on the portfolio", "risk_management"),
    ("How does the z-score signal generate entries?", "signal_generation"),
    ("Backtest the walk-forward strategy with historical simulation", "backtesting"),
    ("Optimize portfolio allocation using mean-variance", "portfolio_construction"),
    ("Clean missing data and handle outliers", "data_processing"),
    ("Reduce slippage on market orders for TWAP", "execution"),
    ("Train an LSTM with cross-validation for feature importance", "machine_learning"),
    ("Improve liquidity in the order book spread", "market_microstructure"),
])
def test_pipeline_stage(query, stage):
    assert stage in classify_pipeline_stages(query)


def test_pipeline_stages_several_none_and_sorted():
    both = classify_pipeline_stages("Optimize the Sharpe ratio while reducing drawdown risk")
    assert "evaluation" in both and "risk_management" in both
    assert classify_pipeline_stages("Hello world") == []
    mixed = classify_pipeline_stages("The z-score signal has too much drawdown risk and poor Sharpe")
    assert mixed == sorted(mixed)


@pytest.mark.parametrize("query, intent, wanted", [
    ("Write a function", "code_generation", True),
    ("Optimize returns", "analysis", False),
    ("Explain the `calculate_sharpe` function", "conceptual", True),      # backticks
    ("What does strategy.py do?", "conceptual", True),                    # a .py file
    ("Optimize the function for speed", "analysis", True),               # the word "function"
])
def test_needs_code(query, intent, wanted):
    assert needs_code(query, intent) is wanted


@pytest.mark.parametrize("query, intent, stage, code, strict", [
    ("Write a function to calculate Sharpe ratio", "code_generation", "evaluation", True, True),
    ("Why does the z-score signal lag on volatile days?", "conceptual", "signal_generation", False, False),
    ("Optimize the drawdown limit for momentum strategy", "analysis", "risk_management", False, False),
])
def test_classify_query(query, intent, stage, code, strict):
    r = classify_query(query)
    assert (r["intent"], r["needs_code"], r["strict_code_mode"]) == (intent, code, strict)
    assert stage in r["pipeline_stage"]
    assert r["priority"] == ("code_strict" if strict else "normal")


def test_a_conceptual_question_about_a_file_still_needs_code():
    r = classify_query("Explain the `risk_manager.py` module")
    assert r["intent"] == "conceptual" and r["needs_code"] is True


@pytest.mark.parametrize("query", ["", "   "])
def test_an_empty_question_gets_the_defaults(query):
    assert classify_query(query) == {"intent": "conceptual", "pipeline_stage": [], "needs_code": False,
                                     "strict_code_mode": False, "priority": "normal"}


@pytest.mark.parametrize("query", [
    "what is triple barrier labeling? how can i implement using python?",
    "show me the code for CUSUM filter",
    "what function calculates daily volatility?",
    "how to apply meta-labelling in python?",
    "calculate gaps series using python",
])
def test_strict_code_words_force_code_generation(query):
    r = classify_query(query)
    assert r["strict_code_mode"] is True and r["priority"] == "code_strict"
    assert r["intent"] == "code_generation" and r["needs_code"] is True


def test_no_strict_code_word_leaves_the_question_normal():
    r = classify_query("explain the concept of mean reversion")
    assert r["strict_code_mode"] is False and r["priority"] == "normal"


# ---------------------------------------------------------------- the engine's two switches
@pytest.mark.parametrize("value, on", [("true", True), ("TRUE", True), ("false", False), (None, False)])
def test_fast_mode_is_read_from_the_environment_on_every_call(monkeypatch, value, on):
    if value is None:
        monkeypatch.delenv("RAG_FAST_MODE", raising=False)
    else:
        monkeypatch.setenv("RAG_FAST_MODE", value)
    assert is_fast_mode() is on


def test_a_simple_query_is_one_under_the_word_threshold(monkeypatch):
    monkeypatch.delenv("RAG_SIMPLE_QUERY_THRESHOLD", raising=False)            # default: 30 words
    assert is_simple_query("What is Sharpe ratio") is True
    assert is_simple_query(" ".join(f"word{i}" for i in range(29))) is True
    assert is_simple_query(" ".join(f"word{i}" for i in range(30))) is False
    monkeypatch.setenv("RAG_SIMPLE_QUERY_THRESHOLD", "30")
    assert is_simple_query(" ".join(f"word{i}" for i in range(35))) is False
    monkeypatch.setenv("RAG_SIMPLE_QUERY_THRESHOLD", "0")                      # 0 turns the mode off
    assert is_simple_query("short query") is False


# ---------------------------------------------------------------- the package's public names
def _top_level_names(body) -> set:
    names: set = set()
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, ast.If):
            names |= _top_level_names(node.body) | _top_level_names(node.orelse)
        elif isinstance(node, ast.Try):
            names |= _top_level_names(node.body) | _top_level_names(node.orelse) | _top_level_names(node.finalbody)
            for h in node.handlers:
                names |= _top_level_names(h.body)
    return names


def test_every_public_name_points_at_code_that_exists():
    """Each lazy name of ``rag_pipeline`` names a module file that defines it at top level, so a retired module
    leaves no name behind.  Read from the source: nothing heavy is imported."""
    import rag_pipeline

    assert set(rag_pipeline.__all__) == set(rag_pipeline._LAZY_IMPORTS)
    root = Path(rag_pipeline.__file__).resolve().parent.parent
    for name, module in rag_pipeline._LAZY_IMPORTS.items():
        path = root.joinpath(*module.split(".")).with_suffix(".py")
        assert path.is_file(), f"{name}: {module} does not exist"
        assert name in _top_level_names(ast.parse(path.read_text(encoding="utf-8")).body), \
            f"{name} is not defined in {module}"
