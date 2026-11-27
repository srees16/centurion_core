"""
RAG Query Engine for Centurion Capital LLC.

Orchestrates the full latency-optimised RAG pipeline:

    1. **Semantic cache** — skip pipeline if a near-duplicate response
       is already cached (< 5 ms).
    2. **FAQ fast-path** — check a small, high-confidence FAQ collection
       before running the full pipeline (< 50 ms).
    3. **Query rewriting** — expand the user query into multiple
       paraphrases for broader recall.
    4. **Concurrent embedding** — embed rewritten queries in parallel
       with ``concurrent.futures``.
    5. **Hybrid search** — BM25 + vector retrieval with RRF merging.
    6. **Threshold filter & dedup** — remove low-score / duplicate chunks.
    7. **Cross-encoder re-rank** — score the top candidates with a
       lightweight cross-encoder model.
    8. **Token-budget context builder** — dynamically trim the context
       to fit within a token budget before sending to the LLM.
    9. **LLM generation** — blocking or streaming answer generation.
   10. **Performance tracing** — per-stage latency logged and surfaced
       in the ``RAGResponse``.

The engine is LLM-agnostic: plug in Ollama, Claude, OpenAI, or any
object that satisfies ``generate(query, context) -> str``.
"""

import logging
import os
import re as _re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Protocol, Sequence, Tuple, Union

from rag_pipeline.config import RAGConfig
from rag_pipeline.storage.embeddings import EmbeddingService
from rag_pipeline.llm.evaluation import RetrievalLogger
from rag_pipeline.core.hybrid_search import HybridSearcher
from rag_pipeline.llm.llm_service import create_llm_backend
from rag_pipeline.utils.perf_trace import PipelineTrace
from rag_pipeline.core.query_rewriter import QueryRewriter, expand_query
from rag_pipeline.core.reranker import CrossEncoderReranker
from rag_pipeline.core.retriever import is_fast_mode, is_simple_query
from rag_pipeline.core.semantic_cache import SemanticCache
from rag_pipeline.ingestion.tiered_retrieval import TieredRetriever
from rag_pipeline.core.fastpath import try_fastpath
from rag_pipeline.utils.time_budget import create_time_budget
from rag_pipeline.utils.token_counter import budget_chunks, count_tokens
from rag_pipeline.storage.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retrieval hard-timeout fallback
# ---------------------------------------------------------------------------
# If the full retrieval stage (embed + search across all query variants)
# exceeds this wall-clock limit, cancel the in-flight futures and fall
# back to a minimal top-k vector-only search.
_RETRIEVAL_HARD_TIMEOUT: int = int(
    os.getenv("CENTURION_RAG_RETRIEVAL_HARD_TIMEOUT", "30")
)
_FALLBACK_TOP_K: int = 3   # vector-only results when timeout fires

# ---------------------------------------------------------------------------
# SLA thresholds (seconds) — used for health-check warnings
# ---------------------------------------------------------------------------
_SLA_RETRIEVAL_S: float = float(os.getenv("CENTURION_RAG_SLA_RETRIEVAL", "5"))
_SLA_LLM_START_S: float = float(os.getenv("CENTURION_RAG_SLA_LLM_START", "10"))
_SLA_TOTAL_S:     float = float(os.getenv("CENTURION_RAG_SLA_TOTAL", "60"))

# ---------------------------------------------------------------------------
# Metadata boost — exact match for snippet / section references in query
# ---------------------------------------------------------------------------

_QUERY_SNIPPET_RE = _re.compile(
    r"(?:snippet|code\s*(?:listing|example)?|listing|algorithm|example)"
    r"\s*([\d.]+)",
    _re.IGNORECASE,
)
_QUERY_SECTION_RE = _re.compile(
    r"(?:section|chapter)\s*([\d.]+)", _re.IGNORECASE,
)
_QUERY_NUMBERED_REF_RE = _re.compile(
    r"\b(\d+\.\d+)\b",  # any N.N that might be a section/snippet ref
)

_BOOST_STOPWORDS = frozenset(
    "a an and are as at be but by for from give gives has have how i if in "
    "into is it me my no not of on or our so such that the their them then "
    "there these they this to us using was we what when which who will with".split()
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

def _source_list(source_filter: Optional[Union[str, Sequence[str]]]) -> List[str]:
    """One source or several as a sorted list without blanks (a cache key and a filter alike)."""
    if not source_filter:
        return []
    items = [source_filter] if isinstance(source_filter, str) else list(source_filter)
    return sorted({str(s) for s in items if s})


def context_sources(chunks: Sequence["RetrievedChunk"], excerpt_chars: int = 400) -> List[Dict[str, Any]]:
    """The passages an answer is built from, for the page's Sources list (tracker RAG3): the document, its page,
    a relevance (one less the vector distance, clamped to 0..1) and an excerpt."""
    out = []
    for c in chunks:
        page = (c.metadata or {}).get("page_number")
        out.append({"source": c.source, "page": page if page not in (None, "", "?") else None,
                    "chunk_index": c.chunk_index, "score": round(max(0.0, min(1.0, 1.0 - float(c.distance))), 3),
                    "content": (c.text or "")[:excerpt_chars]})
    return out


@dataclass
class RetrievedChunk:
    """A single retrieved document chunk with metadata."""
    text: str
    source: str
    chunk_index: int
    distance: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResponse:
    """Complete response from the RAG query engine."""
    query: str
    answer: str
    chunks: List[RetrievedChunk]
    rag_enabled: bool = True
    cached: bool = False
    faq_hit: bool = False
    trace: Optional[PipelineTrace] = None


@dataclass
class _Prepared:
    """What the stages before the LLM produced, for ``query`` and ``query_stream`` alike (docs/AUDIT.md R2).

    ``answered_by`` is "cache", "faq" or "fastpath" when the answer came before any search; otherwise ``chunks``
    holds the passages and ``context`` the text the LLM gets, and no passage means the relevance gate.
    """
    t0: float
    trace: PipelineTrace
    budget: Any = None
    timings: Dict[str, float] = field(default_factory=dict)
    cache_space: str = ""
    answered_by: Optional[str] = None
    answer: str = ""
    chunks: List["RetrievedChunk"] = field(default_factory=list)
    context: str = ""


#: The relevance gate's answers: no passage survived, so the LLM is not asked (it would invent one).
_NO_PASSAGE_ANSWER = (
    "I could not find relevant information in the uploaded "
    "documents to answer this question.\n\n"
    "**Possible reasons:**\n"
    "- The relevant document may not have been uploaded yet\n"
    "- The query may need to be rephrased to match the "
    "document content\n"
    "- The uploaded documents may not cover this topic\n\n"
    "Please upload the relevant PDF or try rephrasing your query."
)
_NO_PASSAGE_ANSWER_STREAM = (
    "I could not find relevant information in the uploaded "
    "documents to answer this question.\n\n"
    "Please upload the relevant PDF or try rephrasing your query."
)


# ---------------------------------------------------------------------------
# LLM callback protocol (plug-in any model)
# ---------------------------------------------------------------------------

class LLMBackend(Protocol):
    """
    Any callable that takes (query, context_str) answer_str.

    Implement this to plug in OpenAI, Anthropic, local LLM, etc.
    """

    def generate(self, query: str, context: str) -> str:
        ...


class DefaultLLMBackend:
    """
    Fallback – returns the retrieved context verbatim.

    Used only when Ollama is not configured / available.
    """

    def generate(self, query: str, context: str) -> str:
        if not context.strip():
            return (
                " No relevant documents found in the knowledge base. "
                "Please upload strategy PDFs first."
            )
        return (
            f"**Retrieved context for:** *{query}*\n\n"
            "---\n\n"
            f"{context}"
        )


# ---------------------------------------------------------------------------
# Query engine
# ---------------------------------------------------------------------------

class RAGQueryEngine:
    """
    High-level RAG query orchestrator with latency optimisations.

    Pipeline flow:
        cache check FAQ fast-path query rewrite concurrent embed
         hybrid search threshold + dedup re-rank token-budget
        context build LLM generate cache store

    Usage:
        engine = RAGQueryEngine(vector_store)
        response = engine.query("What momentum indicators are used?")
        print(response.answer)
    """

    def __init__(
        self,
        vector_store: VectorStoreManager,
        config: Optional[RAGConfig] = None,
        embedding_service: Optional[EmbeddingService] = None,
        llm_backend: Optional[LLMBackend] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        query_rewriter: Optional[QueryRewriter] = None,
    ) -> None:
        self._vs = vector_store
        self._config = config or RAGConfig()
        self._embedder = embedding_service or EmbeddingService(self._config)
        self._reranker = reranker
        if self._reranker is None and self._config.reranker_enabled:
            self._reranker = CrossEncoderReranker(self._config)

        # Query rewriter for multi-query expansion
        self._rewriter = query_rewriter
        if self._rewriter is None and self._config.query_rewrite_enabled:
            self._rewriter = QueryRewriter(self._config)

        # Hybrid BM25+vector search
        self._hybrid = None
        if self._config.hybrid_search_enabled:
            self._hybrid = HybridSearcher(
                self._vs, self._embedder, self._config
            )

        # LLM backend: use provided or auto-create from config
        if llm_backend is not None:
            self._llm = llm_backend
        else:
            self._llm = create_llm_backend(self._config)

        # Semantic cache
        self._cache: Optional[SemanticCache] = None
        cache_enabled = getattr(self._config, "cache_enabled", False)
        if cache_enabled:
            self._cache = SemanticCache(
                embedding_fn=self._embedder.embed_query,
                similarity_threshold=getattr(
                    self._config, "cache_similarity_threshold", 0.95
                ),
                ttl_seconds=getattr(self._config, "cache_ttl_seconds", 3600),
                max_entries=getattr(self._config, "cache_max_entries", 256),
            )

        # Tiered retrieval (FAQ fast-path)
        self._tiered: Optional[TieredRetriever] = None
        faq_enabled = getattr(self._config, "faq_enabled", False)
        if faq_enabled:
            self._tiered = TieredRetriever(
                vector_store=self._vs,
                embedding_service=self._embedder,
                config=self._config,
                faq_similarity_threshold=getattr(
                    self._config, "faq_similarity_threshold", 0.90
                ),
            )

        # Token budget for context
        self._context_token_budget: int = getattr(
            self._config, "context_token_budget", 2000
        )

        # Performance tracing toggle
        self._perf_logging: bool = getattr(
            self._config, "perf_logging_enabled", True
        )

        # Streaming toggle
        self._streaming_enabled: bool = getattr(
            self._config, "streaming_enabled", False
        )

        # Retrieval logger for offline analysis
        self._retrieval_logger = RetrievalLogger()

        # Global time-budget controller (per query)
        self._time_budget_factory = create_time_budget

    # ------------------------------------------------------------------ #
    # Cache invalidation callback (wire into ingestion service)
    # ------------------------------------------------------------------ #

    def invalidate_cache(self, source_name: str, action: str = "changed") -> None:
        """
        Invalidate semantic cache entries related to a source document.

        This should be called whenever documents are ingested or deleted
        so that stale cached answers are not served.

        Args:
            source_name: The PDF filename whose cache entries to invalidate.
            action: Descriptive action string (\"ingested\", \"deleted\").
        """
        if self._cache is not None:
            removed = self._cache.invalidate(source=source_name)
            logger.info(
                "Cache invalidated for '%s' (action=%s): %d entries removed",
                source_name, action, removed,
            )
        # Also force BM25 index rebuild on next search
        if self._hybrid is not None:
            self._hybrid._bm25_count = -1
            logger.debug("Forced BM25 index rebuild after %s of '%s'", action, source_name)

    # ------------------------------------------------------------------ #
    # Main query entry point
    # ------------------------------------------------------------------ #

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
        source_filter: Optional[Union[str, Sequence[str]]] = None,
        skip_cache: bool = False,
    ) -> RAGResponse:
        """
        Run the full latency-optimised RAG pipeline and answer in one piece.

        Steps:
            1-7. ``_prepare``, shared with ``query_stream``: semantic cache, FAQ and fast path; query
                 expansion and rewrite; concurrent embed and search; threshold filter and dedup; metadata
                 boost and re-rank; broad fallback; token-budget context
            8. LLM generate
            9. Cache store, retrieval log and trace summary
        """
        run = self._prepare(query_text, top_k, where, source_filter, skip_cache)
        trace, budget = run.trace, run.budget
        if run.answered_by:
            return RAGResponse(
                query=query_text,
                answer=run.answer,
                chunks=[],
                rag_enabled=True,
                cached=run.answered_by == "cache",
                faq_hit=run.answered_by == "faq",
                trace=trace,
            )
        if not run.chunks:                                # the relevance gate: no passage, no model
            return RAGResponse(
                query=query_text,
                answer=_NO_PASSAGE_ANSWER,
                chunks=[],
                rag_enabled=True,
                trace=trace,
            )
        chunks, context = run.chunks, run.context

        # --- Time-budget check before LLM ---
        if budget.is_expired():
            logger.warning(
                "TimeBudget: expired (%.1fs) before LLM — returning "
                "context-only answer.", budget.elapsed,
            )
            trace.stop()
            if self._perf_logging:
                trace.summary()
            return RAGResponse(
                query=query_text,
                answer=(
                    "_The system could not complete LLM generation "
                    "within the time budget._\n\n"
                    "**Retrieved context (unprocessed):**\n\n"
                    + context[:2000]
                    + budget.cutoff_message()
                ),
                chunks=chunks,
                rag_enabled=True,
                trace=trace,
            )

        # --- Step 8: LLM generate (with time-budget guard) ---
        _t_llm = time.time()
        run.timings["llm_start_offset"] = _t_llm - run.t0  # wall-clock to LLM start
        with trace.span("llm_generate"):
            answer = self._llm.generate(query_text, context)
        run.timings["llm_call"] = time.time() - _t_llm

        # Check if total budget was exceeded during LLM
        if budget.is_expired():
            answer += budget.cutoff_message()

        # --- Step 9: Cache store + trace ---
        self._store_answer(query_text, answer, chunks, run.cache_space)

        # --- Step 10: Log retrieval for offline analysis ---
        try:
            self._retrieval_logger.log(
                query=query_text,
                retrieved_ids=[
                    c.metadata.get("chunk_hash", f"{c.source}_{c.chunk_index}")
                    for c in chunks
                ],
                retrieved_sources=[c.source for c in chunks],
                distances=[c.distance for c in chunks],
                rerank_scores=[
                    c.metadata.get("rerank_score", 0.0) for c in chunks
                ],
                extra={
                    "space_id": run.cache_space,
                    "latency_ms": round(trace.total_ms, 1),
                },
            )
        except Exception as exc:
            logger.debug("Retrieval logging failed: %s", exc)

        run.timings["total"] = time.time() - run.t0

        trace.stop()
        if self._perf_logging:
            trace.summary()
        self._log_stage_breakdown(run.timings)

        return RAGResponse(
            query=query_text,
            answer=answer,
            chunks=chunks,
            rag_enabled=True,
            trace=trace,
        )

    # ------------------------------------------------------------------ #
    # Streaming query
    # ------------------------------------------------------------------ #

    def query_stream(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
        source_filter: Optional[Union[str, Sequence[str]]] = None,
        skip_cache: bool = False,
        llm_model: Optional[str] = None,
        llm_meta: Optional[Dict[str, Any]] = None,
        on_context: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
    ) -> Generator[str, None, None]:
        """
        Run the retrieval pipeline, then stream the LLM answer
        token-by-token.

        ``llm_model``: the Claude model the user picked (tracker RAG2), passed to a backend that takes one; the
        answer cache is kept per model.  ``llm_meta`` (a dict) receives what answered: the LLM and model, or the
        cache, FAQ or fast path when no LLM ran.  ``on_context`` receives the passages the answer is built from
        (``context_sources``), once, before the LLM starts (tracker RAG3).

        Everything up to the LLM call is ``_prepare``, the same stages as ``query()``.
        The LLM step uses ``generate_stream()`` to yield tokens.
        Includes TTFT (time-to-first-token) tracing.
        """
        run = self._prepare(query_text, top_k, where, source_filter, skip_cache, llm_model=llm_model, stream=True)
        trace, budget = run.trace, run.budget
        if run.answered_by:                               # the cache, the FAQ or the fast path
            if llm_meta is not None:
                llm_meta.clear()
                llm_meta["source"] = run.answered_by
            yield run.answer
            return
        if not run.chunks:                                # the relevance gate: no passage, no model
            yield _NO_PASSAGE_ANSWER_STREAM
            return
        chunks, context = run.chunks, run.context
        if on_context is not None:                        # the page's Sources list (RAG3); never stops the answer
            try:
                on_context(context_sources(chunks))
            except Exception as exc:                      # noqa: BLE001
                logger.warning("on_context failed: %s", exc)

        # --- Time-budget check before LLM streaming ---
        if budget.is_expired():
            logger.warning(
                "TimeBudget (stream): expired (%.1fs) before LLM.",
                budget.elapsed,
            )
            yield (
                "_The system could not start LLM generation within "
                "the time budget._\n\n"
                + budget.cutoff_message()
            )
            trace.stop()
            if self._perf_logging:
                trace.summary()
            return

        # Stream LLM tokens with TTFT tracking + time-budget abort
        _t_llm_s = time.time()
        run.timings["llm_start_offset"] = _t_llm_s - run.t0
        if hasattr(self._llm, "generate_stream"):
            collected: List[str] = []
            llm_start = time.perf_counter()
            ttft_recorded = False
            llm_aborted = False
            llm_kwargs = ({"model": llm_model, "meta": llm_meta}
                          if getattr(self._llm, "accepts_model", False) else {})
            if not llm_kwargs and llm_meta is not None:
                llm_meta.update(source="llm", provider=type(self._llm).__name__)
            for token in self._llm.generate_stream(query_text, context, **llm_kwargs):
                if not ttft_recorded:
                    ttft_ms = (time.perf_counter() - llm_start) * 1000.0
                    # Record TTFT as a span with metadata
                    with trace.span("llm_ttft", ttft_ms=round(ttft_ms, 1)):
                        pass
                    ttft_recorded = True
                collected.append(token)
                yield token

                # --- LLM time-budget guard ---
                llm_elapsed = time.perf_counter() - llm_start
                if budget.llm_exceeded(llm_elapsed):
                    llm_aborted = True
                    logger.warning(
                        "TimeBudget (stream): LLM aborted after %.1fs.",
                        llm_elapsed,
                    )
                    break

            # Record total LLM streaming time
            llm_total_ms = (time.perf_counter() - llm_start) * 1000.0
            with trace.span("llm_stream_total", total_ms=round(llm_total_ms, 1)):
                pass

            # Yield cutoff message if LLM was aborted
            if llm_aborted or budget.is_expired():
                cutoff = budget.cutoff_message()
                if cutoff:
                    yield cutoff
            # Cache the full answer
            self._store_answer(query_text, "".join(collected), chunks, run.cache_space)
        else:
            with trace.span("llm_generate"):
                answer = self._llm.generate(query_text, context)
            yield answer

        run.timings["llm_call"] = time.time() - _t_llm_s
        run.timings["total"] = time.time() - run.t0

        trace.stop()
        if self._perf_logging:
            trace.summary()
        self._log_stage_breakdown(run.timings)

    # ------------------------------------------------------------------ #
    # The stages before the LLM, shared by query and query_stream
    # ------------------------------------------------------------------ #

    def _prepare(
        self,
        query_text: str,
        top_k: Optional[int],
        where: Optional[Dict[str, Any]],
        source_filter: Optional[Union[str, Sequence[str]]],
        skip_cache: bool,
        llm_model: Optional[str] = None,
        stream: bool = False,
    ) -> "_Prepared":
        """Every stage before the LLM (docs/AUDIT.md R2): the cache, FAQ and fast-path answers; query expansion and
        rewrite; the concurrent search; threshold, dedup, boost and re-rank; the broad fallback; the context.

        The trace is stopped when it answers early (``answered_by``) or finds no passage (``chunks`` empty).
        ``llm_model`` keys the stream's cache per model (RAG2); ``stream`` only marks the log lines.
        """
        tag = " (stream)" if stream else ""
        run = _Prepared(t0=time.time(), trace=PipelineTrace())
        trace = run.trace
        trace.start()
        run.budget = budget = self._time_budget_factory()
        budget.start()
        top_k = top_k or self._config.top_k
        # Step 2: ensure we always retrieve at least minimum_return_top_k
        effective_top_k = max(top_k, self._config.minimum_return_top_k)
        space_id = self._config.default_space_id

        # Include source_filter in cache key so switching sources
        # does not return stale cached answers from a different PDF.
        _sources = _source_list(source_filter)
        run.cache_space = f"{space_id}::src={','.join(_sources)}" if _sources else space_id
        if llm_model:                                     # one model's answer is not another's (RAG2)
            run.cache_space = f"{run.cache_space}::llm={llm_model}"

        def _answered(kind: str, answer: str) -> "_Prepared":
            trace.stop()
            if self._perf_logging:
                trace.summary()
            run.answered_by, run.answer = kind, answer
            return run

        # --- Step 1: Semantic cache check ---
        if self._cache is not None and not skip_cache:
            with trace.span("cache_lookup"):
                hit = self._cache.lookup(query_text, space_id=run.cache_space)
            if hit is not None:
                return _answered("cache", hit.answer)

        # --- Step 2: FAQ fast-path ---
        if self._tiered is not None:
            with trace.span("faq_lookup"):
                faq_hit = self._tiered.check_faq(query_text)
            if faq_hit is not None:
                return _answered("faq", faq_hit.answer)

        # --- Step 2b: Fastpath bypass (short quant questions) ---
        _t_cls = time.time()
        with trace.span("fastpath_check"):
            fastpath_answer = try_fastpath(query_text)
        run.timings["classification"] = time.time() - _t_cls
        if fastpath_answer is not None:
            logger.info("Fastpath HIT%s — bypassing full RAG pipeline.", tag)
            self._store_answer(query_text, fastpath_answer, [], run.cache_space)
            return _answered("fastpath", fastpath_answer)

        # Build metadata filter
        effective_where = self._build_where_filter(where, source_filter)

        # --- Step 3: Query rewriting ---
        queries = [query_text]

        # Step 4: Structural query expansion (deterministic, no LLM)
        expanded_text = expand_query(query_text)
        if expanded_text != query_text:
            queries[0] = expanded_text
            logger.info(
                "Query expanded%s: '%s' → '%s'",
                tag, query_text, expanded_text,
            )

        # Step 7: DEBUG_RETRIEVAL logging
        if self._config.debug_retrieval:
            logger.info(
                "DEBUG_RETRIEVAL%s: original_query='%s', expanded_query='%s', "
                "effective_top_k=%d",
                tag, query_text, queries[0], effective_top_k,
            )

        if self._rewriter:
            with trace.span("query_rewrite"):
                queries = self._rewriter.rewrite(expanded_text)
            logger.info("Multi-query expansion%s: %d variants", tag, len(queries))

        # --- Step 4: Concurrent embed + retrieve (hard-timeout guarded) ---
        _t_vec = time.time()
        chunks, timed_out = self._retrieve(queries, query_text, effective_top_k, effective_where, trace, tag)
        run.timings["vector_search"] = time.time() - _t_vec
        # Sort by distance (lower = better)
        chunks.sort(key=lambda c: c.distance)

        # --- Time-budget check: retrieval slow force FAST_MODE ---
        retrieval_span = next(
            (s for s in trace.spans if s.name == "embed_and_retrieve"), None
        )
        _retrieval_elapsed = retrieval_span.elapsed_s if retrieval_span else 0.0
        force_fast = timed_out or budget.retrieval_exceeded(_retrieval_elapsed)
        if force_fast:
            logger.warning(
                "TimeBudget%s: retrieval took %.1fs — forcing FAST_MODE "
                "for reranking/remaining stages.%s",
                tag, _retrieval_elapsed,
                " (hard-timeout fallback)" if timed_out else "",
            )

        logger.info(
            "Retrieved %d chunks%s (distances: %s)",
            len(chunks), tag,
            [f"{c.distance:.4f}" for c in chunks[:5]],
        )

        # Step 7: DEBUG_RETRIEVAL — log per-chunk details before filtering
        if self._config.debug_retrieval:
            for ci, c in enumerate(chunks[:15]):
                logger.info(
                    "DEBUG_RETRIEVAL chunk[%d]: dist=%.4f source=%s "
                    "page=%s section=%s chapter=%s text=%.120s",
                    ci, c.distance,
                    c.metadata.get("source", "?"),
                    c.metadata.get("page_number", "?"),
                    c.metadata.get("section", ""),
                    c.metadata.get("chapter", ""),
                    c.text.replace("\n", " "),
                )

        chunks = self._select(chunks, query_text, expanded_text, where, source_filter, force_fast,
                              trace, run.timings, tag)

        # --- Relevance gate: refuse to hallucinate if no relevant chunks ---
        if not chunks:
            logger.warning(
                "RELEVANCE GATE%s: No relevant chunks survived for query: %.80s",
                tag, query_text,
            )
            trace.stop()
            if self._perf_logging:
                trace.summary()
            return run

        # --- Step 7: Token-budget context build ---
        max_ctx = self._config.max_context_chunks
        if len(chunks) > max_ctx:
            logger.info(
                "Trimming %d chunks to max_context_chunks=%d%s",
                len(chunks), max_ctx, tag,
            )
            chunks = chunks[:max_ctx]

        _t_ctx = time.time()
        with trace.span("build_context", n_chunks=len(chunks)):
            run.context = self._build_context(chunks, self._context_token_budget)
        run.timings["context_build"] = time.time() - _t_ctx

        # Step 7: DEBUG_RETRIEVAL — final context token count
        if self._config.debug_retrieval:
            logger.info(
                "DEBUG_RETRIEVAL%s: final context: %d chunks, %d tokens",
                tag, len(chunks), count_tokens(run.context),
            )
        run.chunks = chunks
        return run

    def _retrieve(
        self,
        queries: List[str],
        query_text: str,
        top_k: int,
        where: Optional[Dict[str, Any]],
        trace: PipelineTrace,
        tag: str,
    ) -> Tuple[List[RetrievedChunk], bool]:
        """Embed and search every query variant at once under the hard timeout; a variant that fails is skipped.
        When the timeout fires with fewer than ``_FALLBACK_TOP_K`` passages, one small vector-only search of the
        question fills in.  Returns the passages (the first of each text prefix) and whether it timed out."""
        all_chunks: List[RetrievedChunk] = []
        seen_texts: set = set()
        timed_out = False

        def _keep(found: List[RetrievedChunk]) -> None:
            for chunk in found:
                chunk_key = chunk.text[:200]
                if chunk_key not in seen_texts:
                    seen_texts.add(chunk_key)
                    all_chunks.append(chunk)

        with trace.span("embed_and_retrieve", n_queries=len(queries)):
            with ThreadPoolExecutor(max_workers=min(len(queries), 4)) as pool:
                futures = {
                    pool.submit(self._embed_and_search, q, top_k, where): q
                    for q in queries
                }
                try:
                    for future in as_completed(
                        futures, timeout=_RETRIEVAL_HARD_TIMEOUT,
                    ):
                        try:
                            _keep(future.result())
                        except Exception as exc:
                            logger.warning(
                                "Query variant failed%s: %s", tag, exc,
                                exc_info=True,
                            )
                except TimeoutError:
                    timed_out = True
                    # Cancel any still-running futures
                    for f in futures:
                        f.cancel()
                    logger.warning(
                        "RETRIEVAL TIMEOUT%s: embed+retrieve exceeded %ds "
                        "— activating fallback (top-%d vector-only, "
                        "skip reranker + parent expansion).",
                        tag, _RETRIEVAL_HARD_TIMEOUT, _FALLBACK_TOP_K,
                    )

        # If the main retrieval timed out and yielded no (or few)
        # usable chunks, run a quick single-query vector-only search.
        if timed_out and len(all_chunks) < _FALLBACK_TOP_K:
            with trace.span("retrieval_fallback", top_k=_FALLBACK_TOP_K):
                try:
                    fallback_chunks = self._embed_and_search(
                        query_text, _FALLBACK_TOP_K, where,
                    )
                    _keep(fallback_chunks)
                    logger.info(
                        "Retrieval fallback%s returned %d chunks.",
                        tag, len(fallback_chunks),
                    )
                except Exception as exc:
                    logger.error(
                        "Retrieval fallback%s also failed: %s", tag, exc,
                    )
        return all_chunks, timed_out

    def _select(
        self,
        chunks: List[RetrievedChunk],
        query_text: str,
        expanded_text: str,
        where: Optional[Dict[str, Any]],
        source_filter: Optional[Union[str, Sequence[str]]],
        force_fast: bool,
        trace: PipelineTrace,
        timings: Dict[str, float],
        tag: str,
    ) -> List[RetrievedChunk]:
        """Threshold and dedup; the metadata boost and the cross-encoder re-rank, both skipped for a simple
        question (and the re-rank in fast mode); then the broad fallback when too few or only weak passages
        are left, inside the same scope (RAG3)."""
        shown = "stream, " if tag else ""

        # --- Step 5: Threshold filter + dedup ---
        # Step 2: Log similarity scores but do NOT hard-filter by threshold
        #         when we have fewer than minimum_return_top_k results.
        with trace.span("filter_and_dedup", before=len(chunks)):
            threshold = self._config.similarity_threshold
            filtered = [c for c in chunks if c.distance <= threshold]
            min_k = self._config.minimum_return_top_k
            if len(filtered) < min_k:
                # Relax: keep the top min_k results regardless of threshold
                logger.info(
                    "Threshold filter too aggressive%s: %d/%d survived "
                    "(threshold=%.2f). Keeping top %d instead.",
                    tag, len(filtered), len(chunks), threshold, min_k,
                )
                chunks = chunks[:min_k]
            else:
                chunks = filtered
            chunks = self._deduplicate_chunks(chunks)
        logger.info(
            "%d chunks after threshold+dedup%s (threshold=%.2f)",
            len(chunks), tag, threshold,
        )

        # --- Simple-query detection (used to skip heavy stages) ---
        _simple_q = is_simple_query(query_text)

        # --- Step 5b: Metadata boost (exact match on snippet/section) ---
        _t_expand = time.time()
        if chunks and not _simple_q:
            with trace.span("metadata_boost", n_candidates=len(chunks)):
                chunks = self._apply_metadata_boost(query_text, chunks)
        elif _simple_q and chunks:
            logger.info(
                "[SIMPLE_QUERY] Skipping metadata boost (%s%d chunks).",
                shown, len(chunks),
            )

        timings["parent_expansion"] = time.time() - _t_expand

        # --- Step 6: Re-rank ---
        # Step 3: DISABLE_RERANKING flag check
        # In FAST_MODE, forced-fast by time budget, or simple queries:
        # skip cross-encoder.
        _t_rerank = time.time()
        _skip_rerank = (
            is_fast_mode()
            or force_fast
            or _simple_q
            or self._config.disable_reranking
        )
        if _skip_rerank:
            logger.info(
                "[%s] Skipping cross-encoder re-rank (%s%d chunks).%s%s",
                "DISABLE_RERANKING" if self._config.disable_reranking else
                ("SIMPLE_QUERY" if _simple_q else "FAST_MODE"),
                shown, len(chunks),
                " (time-budget forced)" if force_fast else "",
                " (config override)" if self._config.disable_reranking else "",
            )
        elif self._reranker and chunks:
            with trace.span("rerank", n_candidates=len(chunks)):
                chunks = self._reranker.rerank(
                    query_text, chunks, top_n=self._config.rerank_top_n
                )
            if not chunks:
                logger.warning(
                    "All chunks filtered out by reranker score threshold%s "
                    "(threshold=%.2f). No relevant context for LLM.",
                    tag, getattr(self._config, "rerank_score_threshold", 0.25),
                )

        timings["reranker"] = time.time() - _t_rerank

        # --- Step 6: Retrieval fallback strategy ---
        if self._should_fallback(chunks):
            logger.info(
                "FALLBACK%s: chunks=%d, triggering broad retrieval "
                "(top_k=%d, scope kept, no noise filter, no reranker).",
                tag, len(chunks), self._config.fallback_top_k,
            )
            with trace.span("retrieval_fallback_broad",
                            top_k=self._config.fallback_top_k):
                fallback_chunks = self._fallback_retrieve(
                    query_text, expanded_text,
                    where=self._build_where_filter(where, source_filter, noise_filter=False),
                )
            if fallback_chunks:
                # Merge: deduplicate against existing
                existing_keys = {c.text[:200] for c in chunks}
                for fc in fallback_chunks:
                    if fc.text[:200] not in existing_keys:
                        chunks.append(fc)
                        existing_keys.add(fc.text[:200])
                chunks.sort(key=lambda c: c.distance)
                logger.info(
                    "FALLBACK%s: merged to %d total chunks.", tag, len(chunks),
                )
        return chunks

    def _store_answer(
        self, query_text: str, answer: str, chunks: List[RetrievedChunk], cache_space: str,
    ) -> None:
        """Keep the answer in the semantic cache, with the passages it came from, when the cache is on."""
        if self._cache is not None:
            chunks_summary = [
                {"source": c.source, "chunk_index": c.chunk_index}
                for c in chunks
            ]
            self._cache.store(
                query_text, answer, chunks_summary, space_id=cache_space
            )

    # ------------------------------------------------------------------ #
    # Stage breakdown logger with SLA health checks
    # ------------------------------------------------------------------ #

    @staticmethod
    def _log_stage_breakdown(timings: Dict[str, float]) -> None:
        """Print a structured wall-clock breakdown with SLA checks.

        Thresholds (configurable via env vars):
            retrieval (vector_search)  < ``_SLA_RETRIEVAL_S``  (default 5 s)
            LLM start offset           < ``_SLA_LLM_START_S``  (default 10 s)
            total end-to-end           < ``_SLA_TOTAL_S``      (default 60 s)
        """
        def _f(key: str) -> str:
            """Format seconds with 2-decimal precision, or 'skip'."""
            v = timings.get(key)
            if v is None:
                return "  (skip)"
            return f"{v:>7.2f}s"

        retrieval_s = timings.get("vector_search", 0.0)
        llm_start   = timings.get("llm_start_offset", 0.0)
        total_s     = timings.get("total", 0.0)

        # Build health-check labels
        def _check(value: float, limit: float) -> str:
            return "OK" if value <= limit else "SLOW"

        lines = [
            "",
            "=" * 60,
            "  STAGE BREAKDOWN (wall-clock seconds)",
            "=" * 60,
            f"  classification ........... {_f('classification')}",
            f"  vector_search ............ {_f('vector_search')}",
            f"  parent_expansion ......... {_f('parent_expansion')}",
            f"  reranker ................. {_f('reranker')}",
            f"  context_build ............ {_f('context_build')}",
            f"  llm_call ................. {_f('llm_call')}",
            "-" * 60,
            f"  TOTAL .................... {_f('total')}",
            f"  LLM start offset ......... {_f('llm_start_offset')}",
            "-" * 60,
            "  SLA HEALTH CHECKS:",
            f"    retrieval  < {_SLA_RETRIEVAL_S:.0f}s ........ "
            f"{retrieval_s:.2f}s  [{_check(retrieval_s, _SLA_RETRIEVAL_S)}]",
            f"    LLM start  < {_SLA_LLM_START_S:.0f}s ....... "
            f"{llm_start:.2f}s  [{_check(llm_start, _SLA_LLM_START_S)}]",
            f"    total      < {_SLA_TOTAL_S:.0f}s ....... "
            f"{total_s:.2f}s  [{_check(total_s, _SLA_TOTAL_S)}]",
            "=" * 60,
        ]
        text = "\n".join(lines)
        logger.info(text)

        # Emit warnings for SLA breaches
        if retrieval_s > _SLA_RETRIEVAL_S:
            logger.warning(
                "SLA BREACH: retrieval took %.2fs (limit %.0fs)",
                retrieval_s, _SLA_RETRIEVAL_S,
            )
        if llm_start > _SLA_LLM_START_S:
            logger.warning(
                "SLA BREACH: LLM start offset %.2fs (limit %.0fs)",
                llm_start, _SLA_LLM_START_S,
            )
        if total_s > _SLA_TOTAL_S:
            logger.warning(
                "SLA BREACH: total %.2fs (limit %.0fs)",
                total_s, _SLA_TOTAL_S,
            )

    # ------------------------------------------------------------------ #
    # Embed + search helper (used by ThreadPoolExecutor)
    # ------------------------------------------------------------------ #

    def _embed_and_search(
        self,
        query_text: str,
        top_k: int,
        where: Optional[Dict[str, Any]],
    ) -> List[RetrievedChunk]:
        """Embed a single query variant and retrieve chunks.

        If the filtered query fails (e.g. ChromaDB index corruption),
        automatically retries without metadata filters so retrieval
        degrades gracefully instead of returning zero results.
        """
        query_embedding = self._embedder.embed_query(query_text)

        try:
            results = self._run_search(
                query_text, query_embedding, top_k, where,
            )
        except Exception as exc:
            if where is not None:
                logger.warning(
                    "Filtered retrieval failed (%s) — retrying without "
                    "metadata filter.", exc,
                )
                results = self._run_search(
                    query_text, query_embedding, top_k, None,
                )
            else:
                raise
        return self._parse_results(results)

    def _run_search(
        self,
        query_text: str,
        query_embedding: List[float],
        top_k: int,
        where: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Execute a single vector (or hybrid) search."""
        if self._hybrid:
            return self._hybrid.search(
                query_text=query_text,
                query_embedding=query_embedding,
                top_k=top_k,
                where=where,
                bm25_weight=self._config.hybrid_bm25_weight,
                vector_weight=self._config.hybrid_vector_weight,
            )
        return self._vs.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    def _build_where_filter(
        self,
        where: Optional[Dict[str, Any]],
        source_filter: Optional[Union[str, Sequence[str]]],
        noise_filter: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Merge caller-supplied filter with convenience parameters.

        ``noise_filter=False`` keeps the scope (the caller's filter, the sources, the space) and drops only the
        short-chunk filter: the broad fallback's search, which must never leave the ticked sources (RAG3).

        ``source_filter``: one source, or several (the page's ticked sources: all of them are searched, and only
        they; tracker RAG3).  Enforces multi-tenant space isolation when enabled in config.
        Returns a ChromaDB ``where`` dict or None.
        """
        filters: List[Dict[str, Any]] = []

        if where:
            filters.append(where)

        sources = _source_list(source_filter)
        if len(sources) == 1:
            filters.append({"source": sources[0]})
        elif sources:
            filters.append({"source": {"$in": sources}})

        # Multi-tenant space isolation
        if self._config.enforce_space_isolation:
            filters.append({"space_id": self._config.default_space_id})

        # Filter out very short chunks (noise) at query time
        if noise_filter and self._config.enable_metadata_filtering:
            filters.append({
                "word_count": {"$gte": self._config.min_chunk_word_count}
            })

        if not filters:
            return None
        if len(filters) == 1:
            return filters[0]
        return {"$and": filters}

    # ------------------------------------------------------------------
    # Step 1: Ingestion verification diagnostic
    # ------------------------------------------------------------------

    def verify_chapter_indexing(self, chapter_number: int) -> None:
        """Diagnostic: verify that a chapter was indexed properly.

        Queries ChromaDB with ``query_text="Chapter {n}"`` and prints
        the top-10 results with their first 300 chars, metadata, and
        similarity score.  Call this after ingestion to audit coverage.
        """
        query_text = f"Chapter {chapter_number}"
        query_embedding = self._embedder.embed_query(query_text)
        results = self._vs.query(
            query_embeddings=[query_embedding],
            n_results=10,
            include=["documents", "metadatas", "distances"],
        )

        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        if not ids:
            print(f"Chapter {chapter_number} not indexed properly")
            logger.warning(
                "verify_chapter_indexing: Chapter %d — no results returned.",
                chapter_number,
            )
            return

        print(f"\n{'=' * 60}")
        print(f"  CHAPTER {chapter_number} INDEXING VERIFICATION")
        print(f"{'=' * 60}")
        for i, doc_id in enumerate(ids):
            similarity = 1.0 - dists[i]
            meta = metas[i] if i < len(metas) else {}
            text_preview = (docs[i][:300] if i < len(docs) else "").replace("\n", " ")
            print(f"\n--- Result {i+1} (ID: {doc_id}) ---")
            print(f"  Similarity: {similarity:.4f}  (distance: {dists[i]:.4f})")
            print(f"  Metadata:   {meta}")
            print(f"  Text:       {text_preview}")
        print(f"\n{'=' * 60}\n")

    # ------------------------------------------------------------------
    # Step 6: Fallback helpers
    # ------------------------------------------------------------------

    def _should_fallback(self, chunks: List[RetrievedChunk]) -> bool:
        """Decide whether to trigger the broad fallback retrieval.

        Returns True when:
        - Fewer than ``fallback_min_chunks`` survived, OR
        - ALL chunks have distance >= ``low_score_distance`` (very low
          confidence).
        """
        if len(chunks) < self._config.fallback_min_chunks:
            return True
        if chunks and all(
            c.distance >= self._config.low_score_distance for c in chunks
        ):
            logger.info(
                "All %d chunks have distance >= %.2f — triggering fallback.",
                len(chunks), self._config.low_score_distance,
            )
            return True
        return False

    def _fallback_retrieve(
        self,
        original_query: str,
        expanded_query: str,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """Broad fallback: re-retrieve with high top_k, no noise filter,
        no reranker.

        Uses the expanded query (or original if identical) for a
        single vector-only search.  ``where`` is the scope only
        (``_build_where_filter(..., noise_filter=False)``): it used to
        search the full collection, so an answer limited to the ticked
        sources, or to one space, quietly drew on the others (RAG3).
        """
        search_text = expanded_query or original_query
        try:
            query_embedding = self._embedder.embed_query(search_text)
            results = self._vs.query(
                query_embeddings=[query_embedding],
                n_results=self._config.fallback_top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            return self._parse_results(results)
        except Exception as exc:
            logger.error("Fallback retrieval failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _deduplicate_chunks(
        self, chunks: List[RetrievedChunk]
    ) -> List[RetrievedChunk]:
        """Remove near-duplicate chunks based on chunk_hash or text overlap.

        Uses ``chunk_hash`` metadata when available (fast, O(n)).
        Falls back to simple text-prefix dedup otherwise.
        """
        if not chunks:
            return chunks

        seen_hashes: set = set()
        deduped: List[RetrievedChunk] = []

        for chunk in chunks:
            h = chunk.metadata.get("chunk_hash", "")
            if h and h in seen_hashes:
                continue
            if h:
                seen_hashes.add(h)
            deduped.append(chunk)

        removed = len(chunks) - len(deduped)
        if removed:
            logger.info("Dedup removed %d near-duplicate chunks", removed)
        return deduped

    @staticmethod
    def _parse_results(results: Dict[str, Any]) -> List[RetrievedChunk]:
        """Convert raw ChromaDB results into ``RetrievedChunk`` objects."""
        chunks: List[RetrievedChunk] = []
        if not results.get("ids") or not results["ids"][0]:
            return chunks

        ids = results["ids"][0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            chunks.append(
                RetrievedChunk(
                    text=docs[i] if i < len(docs) else "",
                    source=meta.get("source", "unknown"),
                    chunk_index=meta.get("chunk_index", 0),
                    distance=dists[i] if i < len(dists) else 1.0,
                    metadata=meta,
                )
            )

        # Sort by similarity (lower distance = more similar for cosine)
        chunks.sort(key=lambda c: c.distance)
        return chunks

    # ------------------------------------------------------------------ #
    # Metadata boost — re-score chunks matching snippet / section refs
    # ------------------------------------------------------------------ #

    @staticmethod
    def _apply_metadata_boost(
        query: str,
        chunks: List[RetrievedChunk],
        snippet_boost: float = 0.15,
        section_boost: float = 0.10,
        keyword_boost: float = 0.08,
    ) -> List[RetrievedChunk]:
        """Reduce distance (= boost) for chunks whose metadata matches
        explicit snippet / section references found in the query.

        Boost magnitudes (applied as distance reduction):
            - Snippet ID match:  ``snippet_boost``  (strongest)
            - Section ID match:  ``section_boost``
            - Title keyword overlap:  ``keyword_boost`` × overlap count
            - Code chunk when query asks for code:  0.05

        The resulting chunk list is re-sorted by boosted distance.
        """
        if not chunks:
            return chunks

        query_lower = query.lower()

        # Extract explicit references from the query
        snippet_refs = set(
            m.group(1) for m in _QUERY_SNIPPET_RE.finditer(query)
        )
        section_refs = set(
            m.group(1) for m in _QUERY_SECTION_RE.finditer(query)
        )
        numbered_refs = set(
            m.group(1) for m in _QUERY_NUMBERED_REF_RE.finditer(query)
        )

        # Content keywords (minus stopwords)
        query_words = set(query_lower.split()) - _BOOST_STOPWORDS

        # Does the query ask for code?
        wants_code = any(
            kw in query_lower
            for kw in (
                "code", "snippet", "function", "method",
                "implementation", "algorithm", "listing",
            )
        )

        boosted = False
        for chunk in chunks:
            meta = chunk.metadata
            boost = 0.0

            # --- Snippet ID match (strongest) ---
            chunk_snippet = str(meta.get("snippet_id", ""))
            if chunk_snippet and (
                chunk_snippet in snippet_refs
                or chunk_snippet in numbered_refs
            ):
                boost = max(boost, snippet_boost)

            # --- Section ID match ---
            chunk_section = str(meta.get("section_id", ""))
            if chunk_section and (
                chunk_section in section_refs
                or chunk_section in numbered_refs
            ):
                boost = max(boost, section_boost)

            # --- Title keyword overlap ---
            titles = " ".join([
                str(meta.get("snippet_title", "")),
                str(meta.get("section_title", "")),
                str(meta.get("section", "")),
            ]).lower()
            if titles.strip() and query_words:
                title_words = set(titles.split()) - _BOOST_STOPWORDS
                overlap = len(query_words & title_words)
                if overlap >= 2:
                    boost = max(boost, keyword_boost * min(overlap, 4))

            # --- Code chunk boost when query asks for code ---
            if wants_code and meta.get("contains_code"):
                boost = max(boost, 0.05)

            if boost > 0:
                chunk.distance = max(0.0, chunk.distance - boost)
                boosted = True

        if boosted:
            chunks.sort(key=lambda c: c.distance)
            logger.info(
                "Metadata boost applied: snippet_refs=%s, section_refs=%s, "
                "numbered_refs=%s",
                snippet_refs, section_refs, numbered_refs,
            )

        return chunks

    @staticmethod
    def _build_context(
        chunks: List[RetrievedChunk],
        token_budget: int = 4000,
    ) -> str:
        """Format retrieved chunks into a structured context string for the LLM.

        Each chunk is wrapped in clear delimiters with metadata headers so
        the LLM can:
          - Distinguish chunk boundaries unambiguously
          - Cite sources accurately using the provided metadata
          - Prioritise high-relevance chunks via the relevance/rerank scores

        Enforces a **token budget** — chunks are added in order of
        relevance until the budget is exhausted, ensuring the LLM
        prompt stays within a predictable size.
        """
        if not chunks:
            return ""

        # Build each chunk's formatted text, then apply budget
        formatted: List[str] = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.metadata
            page = meta.get("page_number", "?")
            section = meta.get("section", "")
            similarity = 1.0 - chunk.distance
            rerank_score = meta.get("rerank_score")
            title = meta.get("title", chunk.source)
            word_count = meta.get("word_count", len(chunk.text.split()))
            contains_code = meta.get("contains_code", False)

            # Structured header with all available metadata
            header_parts = [
                f"=== CONTEXT CHUNK {i}/{len(chunks)} ===",
                f"Source: {chunk.source}",
                f"Title: {title}",
                f"Page: {page}",
            ]
            if section:
                header_parts.append(f"Section: {section}")

            # Structural metadata (chapter / section / snippet)
            chapter = meta.get("chapter", "")
            chapter_title = meta.get("chapter_title", "")
            if chapter:
                ch_label = f"Chapter: {chapter}"
                if chapter_title:
                    ch_label += f" — {chapter_title}"
                header_parts.append(ch_label)
            section_id = meta.get("section_id", "")
            section_title = meta.get("section_title", "")
            if section_id:
                sec_label = f"Section ID: {section_id}"
                if section_title:
                    sec_label += f" — {section_title}"
                header_parts.append(sec_label)
            snippet_id = meta.get("snippet_id", "")
            snippet_title = meta.get("snippet_title", "")
            if snippet_id:
                snip_label = f"Snippet: {snippet_id}"
                if snippet_title:
                    snip_label += f" — {snippet_title}"
                header_parts.append(snip_label)

            header_parts.append(f"Vector Similarity: {similarity:.1%}")
            if rerank_score is not None:
                header_parts.append(f"Reranker Relevance: {rerank_score:.4f}")
            header_parts.append(f"Word Count: {word_count}")
            if contains_code:
                header_parts.append(
                    "THIS CHUNK CONTAINS SOURCE CODE — reproduce it "
                    "EXACTLY as written. Do NOT rewrite, paraphrase, or "
                    "generate alternative code."
                )
            header_parts.append(f"--- Content Start ---")
            header_parts.append(chunk.text)
            header_parts.append(f"--- Content End ---")

            formatted.append("\n".join(header_parts))

        # Apply token budget — include as many chunks as fit
        kept_indices = budget_chunks(
            formatted, max_total_tokens=token_budget, separator_tokens=15
        )
        if len(kept_indices) < len(formatted):
            logger.info(
                "Token budget trimmed context from %d to %d chunks "
                "(budget=%d tokens)",
                len(formatted), len(kept_indices), token_budget,
            )

        parts = [formatted[i] for i in kept_indices]
        return "\n\n".join(parts)
