"""
The query engine's two per-question switches (``core/query_engine.py``).

- ``is_fast_mode``: ``RAG_FAST_MODE=true`` makes the engine skip the cross-encoder re-rank.
- ``is_simple_query``: a question under ``RAG_SIMPLE_QUERY_THRESHOLD`` words (default 30) skips the
  metadata boost and the re-rank.

Both read the environment on every call, so they change without a restart.  This module's older
retrieval orchestrator (the ``Retriever`` class, ``retrieve`` and ``retrieve_context``), which the
engine had stopped using, was retired on 25 Nov 2026 (docs/AUDIT.md, R4).
"""

from __future__ import annotations

import os

# Questions shorter than this many words skip the engine's heavy stages.
SIMPLE_QUERY_THRESHOLD = 30         # default; override via RAG_SIMPLE_QUERY_THRESHOLD


def is_fast_mode() -> bool:
    """Return *True* when low-latency fast-mode is active.

    Checked **dynamically** on every call so that the env-var
    ``RAG_FAST_MODE=true`` can be toggled without restarting.
    """
    return os.getenv("RAG_FAST_MODE", "false").lower() == "true"


def is_simple_query(query: str) -> bool:
    """Return *True* when *query* is shorter than the simple-query threshold.

    For a simple query the query engine skips the metadata boost and the
    cross-encoder re-rank.

    The threshold is read **dynamically** from the env-var
    ``RAG_SIMPLE_QUERY_THRESHOLD`` (default 30) so it can be
    toggled at runtime — set to ``0`` to disable simple-query mode.
    """
    threshold = int(os.getenv("RAG_SIMPLE_QUERY_THRESHOLD",
                              str(SIMPLE_QUERY_THRESHOLD)))
    if threshold <= 0:
        return False
    return len(query.split()) < threshold
