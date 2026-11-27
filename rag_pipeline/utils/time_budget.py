"""
Global Time Budget Controller for Centurion Capital LLC RAG Pipeline.

Enforces a hard wall-clock budget per query so no single request can
hang indefinitely.  Thresholds:

    ==================  =============  ==============================
    Budget              Default (s)    Action
    ==================  =============  ==============================
    Total query         300            Abort & return partial answer
    Retrieval slow      30             Switch to FAST_MODE
    LLM generation      300            Abort & return collected tokens
    ==================  =============  ==============================

Usage::

    from rag_pipeline.utils.time_budget import TimeBudget

    budget = TimeBudget()        # uses defaults (300 / 30 / 300)
    budget.start()

    # ... after retrieval finishes ...
    if budget.retrieval_exceeded(retrieval_seconds):
        # switch to fast mode for remaining stages
        ...

    # inside LLM streaming loop:
    for token in llm.generate_stream(q, ctx):
        if budget.llm_exceeded(llm_elapsed):
            break  # abort LLM, return partial

    if budget.is_expired():
        answer += budget.cutoff_message()

Tech stack: Python 3.11 · stdlib only · No external deps.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Defaults (overridable via env vars or constructor)
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_QUERY_BUDGET_S = 300        # total seconds per query
DEFAULT_RETRIEVAL_SLOW_S = 30 # retrieval threshold FAST_MODE
DEFAULT_LLM_BUDGET_S = 300          # max seconds for LLM generation

# Graceful cutoff messages
_CUTOFF_TOTAL = (
    "\n\n---\n"
    " **Response truncated** — the total processing time exceeded "
    "the allowed budget ({budget}s).  The answer above may be "
    "incomplete.  Please try a more specific query or enable "
    "FAST_MODE for quicker results."
)

_CUTOFF_LLM = (
    "\n\n---\n"
    " **Response truncated** — the language model took longer than "
    "{budget}s.  The partial answer above contains the tokens "
    "generated so far.  Try shortening your query or reducing the "
    "context window."
)


# ═══════════════════════════════════════════════════════════════════════════
# TimeBudget controller
# ═══════════════════════════════════════════════════════════════════════════

class TimeBudget:
    """Track and enforce time limits across the RAG query pipeline.

    Parameters
    ----------
    total_budget_s : float
        Hard wall-clock cap per query (default 300 s).
    retrieval_slow_s : float
        If a retrieval stage takes longer than this, the caller
        should switch to FAST_MODE (default 30 s).
    llm_budget_s : float
        Max wall-clock time for LLM generation.  After this the
        caller should abort and return partial output (default 300 s).
    """

    def __init__(
        self,
        total_budget_s: float = DEFAULT_QUERY_BUDGET_S,
        retrieval_slow_s: float = DEFAULT_RETRIEVAL_SLOW_S,
        llm_budget_s: float = DEFAULT_LLM_BUDGET_S,
    ) -> None:
        self.total_budget_s = total_budget_s
        self.retrieval_slow_s = retrieval_slow_s
        self.llm_budget_s = llm_budget_s

        self._start: float = 0.0
        self._retrieval_switched = False
        self._llm_aborted = False

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> "TimeBudget":
        """Mark query start and return *self* for chaining."""
        self._start = time.perf_counter()
        return self

    # ── Elapsed / remaining ───────────────────────────────────────────

    @property
    def elapsed(self) -> float:
        """Seconds elapsed since ``start()``."""
        if self._start <= 0:
            return 0.0
        return time.perf_counter() - self._start

    @property
    def remaining(self) -> float:
        """Seconds remaining in the total budget (≥ 0)."""
        return max(0.0, self.total_budget_s - self.elapsed)

    def is_expired(self) -> bool:
        """Return *True* when the total query budget has been exceeded."""
        return self.elapsed >= self.total_budget_s

    # ── Retrieval threshold ───────────────────────────────────────────

    def retrieval_exceeded(self, retrieval_elapsed_s: float) -> bool:
        """Return *True* if retrieval took longer than the slow threshold.

        Also sets an internal flag so that ``was_retrieval_slow`` can
        be queried later.
        """
        exceeded = retrieval_elapsed_s > self.retrieval_slow_s
        if exceeded and not self._retrieval_switched:
            self._retrieval_switched = True
            logger.warning(
                "TimeBudget: retrieval took %.1fs (> %.1fs) — "
                "switching to FAST_MODE for remaining stages.",
                retrieval_elapsed_s, self.retrieval_slow_s,
            )
        return exceeded

    @property
    def was_retrieval_slow(self) -> bool:
        """Whether retrieval exceeded the slow threshold at any point."""
        return self._retrieval_switched

    # ── LLM threshold ─────────────────────────────────────────────────

    def llm_remaining(self) -> float:
        """Max seconds the LLM stage may still run.

        Returns the *lesser* of the LLM budget and the total remaining
        time, so that the LLM cannot exceed the global cap.
        """
        return max(0.0, min(self.llm_budget_s, self.remaining))

    def llm_exceeded(self, llm_elapsed_s: float) -> bool:
        """Return *True* if the LLM has run past its budget.

        Also checks the global budget — if the total query time is
        already exceeded, this returns *True* immediately.
        """
        if self.is_expired():
            self._llm_aborted = True
            return True
        if llm_elapsed_s >= self.llm_budget_s:
            self._llm_aborted = True
            logger.warning(
                "TimeBudget: LLM generation %.1fs exceeded %.1fs budget "
                "— aborting with partial answer.",
                llm_elapsed_s, self.llm_budget_s,
            )
            return True
        return False

    @property
    def was_llm_aborted(self) -> bool:
        """Whether the LLM stage was aborted due to a timeout."""
        return self._llm_aborted

    # ── Cutoff messages ───────────────────────────────────────────────

    def cutoff_message(self) -> str:
        """Return a user-facing message explaining why the response
        was truncated.  Returns an empty string if no cutoff occurred.
        """
        if self._llm_aborted:
            return _CUTOFF_LLM.format(budget=int(self.llm_budget_s))
        if self.is_expired():
            return _CUTOFF_TOTAL.format(budget=int(self.total_budget_s))
        return ""

    # ── Summary dict (for PipelineTrace / logging) ────────────────────

    def as_dict(self) -> dict:
        """Return a JSON-friendly summary of the budget state."""
        return {
            "total_budget_s": self.total_budget_s,
            "retrieval_slow_s": self.retrieval_slow_s,
            "llm_budget_s": self.llm_budget_s,
            "elapsed_s": round(self.elapsed, 3),
            "remaining_s": round(self.remaining, 3),
            "expired": self.is_expired(),
            "retrieval_switched": self._retrieval_switched,
            "llm_aborted": self._llm_aborted,
        }

    def __repr__(self) -> str:
        return (
            f"TimeBudget(elapsed={self.elapsed:.1f}s / "
            f"{self.total_budget_s}s, "
            f"remaining={self.remaining:.1f}s, "
            f"expired={self.is_expired()})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Factory helper
# ═══════════════════════════════════════════════════════════════════════════

def create_time_budget(
    total_budget_s: Optional[float] = None,
    retrieval_slow_s: Optional[float] = None,
    llm_budget_s: Optional[float] = None,
) -> TimeBudget:
    """Create a ``TimeBudget`` from env vars / explicit overrides.

    Environment variables (override defaults):
        ``CENTURION_RAG_QUERY_BUDGET``   — total seconds (default 300)
        ``CENTURION_RAG_RETRIEVAL_SLOW`` — retrieval threshold (default 30)
        ``CENTURION_RAG_LLM_BUDGET``     — LLM budget (default 300)
    """
    t = total_budget_s or float(
        os.getenv("CENTURION_RAG_QUERY_BUDGET", str(DEFAULT_QUERY_BUDGET_S))
    )
    r = retrieval_slow_s or float(
        os.getenv("CENTURION_RAG_RETRIEVAL_SLOW", str(DEFAULT_RETRIEVAL_SLOW_S))
    )
    l_ = llm_budget_s or float(
        os.getenv("CENTURION_RAG_LLM_BUDGET", str(DEFAULT_LLM_BUDGET_S))
    )
    return TimeBudget(total_budget_s=t, retrieval_slow_s=r, llm_budget_s=l_)
