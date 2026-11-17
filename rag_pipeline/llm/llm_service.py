"""
LLM Service for Centurion Capital LLC RAG Pipeline.

Provides LLM-powered answer generation from RAG-retrieved context.
Supports multiple backends:
    - **Ollama**   — Local inference (Llama 3, Mistral, etc.)
    - **Claude**   — Anthropic Messages API
    - **OpenAI**   — OpenAI Chat Completions API (GPT-4o, GPT-4, etc.)

Each backend supports:
    - ``generate(query, context) -> str``   — blocking, full response
    - ``generate_stream(query, context) -> Generator[str, None, None]``
      — yields tokens as they arrive (reduces perceived latency)

Switching providers:
    Set ``CENTURION_RAG_LLM_PROVIDER`` in ``.env`` to one of:
        ``ollama`` | ``claude`` | ``openai``

Usage:
    from rag_pipeline.llm.llm_service import create_llm_backend
    from rag_pipeline.config import RAGConfig

    llm = create_llm_backend(RAGConfig())
    answer = llm.generate(query="What is RSI?", context="RSI is ...")

    # Streaming:
    for token in llm.generate_stream(query="What is RSI?", context="RSI is ..."):
        print(token, end="", flush=True)
"""

import json
import logging
import time
from typing import Generator, Optional

import requests

from rag_pipeline.config import RAGConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt for RAG grounding
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT_FULL = """\
You are a precise document question-answering assistant.

Your task is to answer the user's question using ONLY the context chunks provided below. \
Follow these rules STRICTLY — violations are unacceptable:

GROUNDING RULES (most important):
1. Base your answer EXCLUSIVELY on the provided context chunks. Do NOT use outside knowledge, \
prior training data, or assumptions.
2. NEVER fabricate, infer, or hallucinate ANY information not explicitly stated in the context — \
this includes names, numbers, dates, statistics, company descriptions, product features, or any \
other facts.
3. If the context does not contain enough information to answer the question, you MUST respond \
with EXACTLY: "I could not find sufficient information in the uploaded documents to answer \
this question." — then list what specific information is missing.
4. If you can only partially answer, clearly state which parts are supported by the context \
and which parts you cannot answer. NEVER fill gaps with invented information.
5. Do NOT describe or characterize the document, its authors, or its publisher beyond what \
is explicitly written in the context chunks.

CITATION RULES:
6. ALWAYS cite the source document and page number for EVERY claim, using the format: \
(Source: filename, Page N). Pull these from the chunk headers in the context.
7. When synthesising across multiple chunks, cite each source inline next to the relevant fact.

RESPONSE QUALITY:
8. Synthesise information from multiple context chunks into a single coherent answer. \
Do NOT simply repeat or list chunks verbatim.
9. Use clear, professional language. Use markdown formatting (headers, bullets, bold) \
for readability.
10. If the context contains contradictory information, explicitly note the discrepancy \
and cite both sources so the user can judge.
11. Be concise but thorough. Prefer structured answers (bullet points, numbered lists) \
over long prose.

CODE / FORMULA REPRODUCTION RULES:
12. When the context contains code snippets, source code, formulas, or pseudocode, you MUST \
reproduce them EXACTLY and VERBATIM as they appear in the context. Do NOT rewrite, refactor, \
simplify, translate to another library, or "improve" the code in any way.
13. NEVER generate new code from your training data. The ONLY code you may include in your \
answer is code that appears word-for-word in the provided context chunks. If the user asks \
for code that is not present in the context, respond with: "The requested code was not found \
in the uploaded documents."
14. Preserve the exact indentation, variable names, function names, class names, and comments \
from the source code in the context. Do NOT rename variables or restructure the code.
15. When presenting code from the context, wrap it in a fenced code block with the appropriate \
language tag (e.g., ```python) and indicate its source: (Source: filename, Page N).
16. You MUST reuse the exact function implementation from the provided context if available. \
Do NOT rewrite it unless explicitly asked. If a full implementation exists in context, return \
it verbatim.

CONFIDENCE SIGNAL:
17. End every answer with a confidence indicator on a new line:
    - **High confidence** — answer is fully supported by multiple context chunks
    - **Medium confidence** — answer is supported but from limited context
    - **Low confidence** — answer is partially supported; some gaps exist

OUTPUT LENGTH (mandatory):
18. Provide concise, implementation-focused answers. \
Limit your response to 600 tokens maximum. \
Avoid unnecessary explanations, preambles, or filler text. \
Get straight to the point.\
"""

# ---------------------------------------------------------------------------
# Compact system prompt for local / CPU-bound models (Ollama).
#
# Reduces the system prompt from ~850 tokens to ~150 tokens.  This cuts
# prompt-evaluation time by 5–10× on CPU — the difference between a
# 30 s response and a 200+ s timeout.  All core grounding rules are
# preserved in condensed form.
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT_COMPACT = """\
You are a document QA assistant. Answer ONLY from the provided context chunks.

Rules:
- NEVER use outside knowledge or hallucinate facts.
- If context is insufficient, say: "I could not find sufficient information in the uploaded documents."
- Cite every claim: (Source: filename, Page N).
- Reproduce code/formulas EXACTLY as they appear in the context — never rewrite or invent code.
- If a full function implementation exists in context, return it VERBATIM. Do NOT rewrite.
- Use markdown formatting (bullets, headers, bold). Be concise.
- End with confidence: **High** / **Medium** / **Low**.
- Limit response to 600 tokens.\
"""

# Default: use compact prompt for speed.  Cloud backends (Claude, OpenAI)
# can use the full prompt via CENTURION_RAG_COMPACT_PROMPT=false.
RAG_SYSTEM_PROMPT = RAG_SYSTEM_PROMPT_COMPACT

NO_CONTEXT_SYSTEM_PROMPT = """\
IMPORTANT: No relevant documents were found in the knowledge base for this query.

You MUST respond with EXACTLY this message and nothing else:

"I could not find relevant documents in the knowledge base to answer this question. \
Please upload the relevant PDFs first, or rephrase your question."

Do NOT answer from general knowledge. Do NOT guess. Do NOT add any other text. \
Do NOT attempt to be helpful by providing information from your training data.\
"""


# ---------------------------------------------------------------------------
# Helper: build user message from query + context
# ---------------------------------------------------------------------------

def _build_user_message(query: str, context: str) -> str:
    """Return the user-facing message combining context and query."""
    if context.strip():
        return (
            f"## Context\n\n{context}\n\n"
            f"---\n\n"
            f"## Question\n\n{query}"
        )
    return query


def _pick_system_prompt(context: str, *, compact: bool = True) -> str:
    """Return the appropriate system prompt.

    Parameters
    ----------
    context : str
        The retrieved context text.  If empty, the no-context prompt
        is returned regardless of *compact*.
    compact : bool, default True
        When *True* (the default for Ollama / local inference), use
        the 150-token compact prompt.  Set *False* to use the full
        850-token prompt (recommended for cloud LLMs like Claude or
        OpenAI where prompt-eval is near-instant).
    """
    if not context.strip():
        return NO_CONTEXT_SYSTEM_PROMPT
    return RAG_SYSTEM_PROMPT_COMPACT if compact else RAG_SYSTEM_PROMPT_FULL


# ---------------------------------------------------------------------------
# Ollama backend (local, free, runs Llama 3 / Mistral / etc.)
# ---------------------------------------------------------------------------

# Default flush interval for buffered streaming (seconds).
_STREAM_FLUSH_INTERVAL = 1.5


class OllamaLLMBackend:
    """
    LLM backend using Ollama's local REST API.

    Ollama serves models at http://localhost:11434 by default.
    Uses a persistent ``requests.Session`` to avoid TCP/TLS handshake
    overhead on every call.

    Streaming behaviour
    ~~~~~~~~~~~~~~~~~~~
    Both ``generate()`` and ``generate_stream()`` use Ollama's
    streaming endpoint (``"stream": True``).  Instead of a single
    global *timeout*, each **chunk** (token batch) has its own
    read-timeout (*chunk_timeout*, default 30 s).  If no data
    arrives within that window the request is cancelled — this
    prevents the old 600 s hard-hang.

    ``generate_stream()`` additionally **buffers** tokens and yields
    them every ~1–2 s so the client refreshes smoothly without
    per-token overhead.
    """

    # Hard ceiling — num_predict can never exceed this regardless of
    # what the config or env var says.  Prevents runaway generation.
    _MAX_NUM_PREDICT_CEILING = 800

    # Simple-query threshold: queries shorter than this use a reduced
    # num_predict to save latency.
    _SIMPLE_QUERY_WORD_LIMIT = 30
    _SIMPLE_QUERY_NUM_PREDICT = 300

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.2,
        top_p: float = 0.9,
        num_ctx: int = 4096,
        num_predict: int = 400,
        repeat_penalty: float = 1.1,
        max_tokens: int = 400,
        timeout: int = 600,
        chunk_timeout: int = 30,
        first_token_timeout: int = 180,
        warmup: bool = True,
        compact_prompt: bool = True,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.num_ctx = num_ctx
        # Enforce ceiling on num_predict to prevent unbounded generation
        self.num_predict = min(num_predict, self._MAX_NUM_PREDICT_CEILING)
        self.repeat_penalty = repeat_penalty
        self.max_tokens = min(max_tokens, self._MAX_NUM_PREDICT_CEILING)
        self.timeout = timeout
        self.chunk_timeout = chunk_timeout
        self.first_token_timeout = first_token_timeout
        self.compact_prompt = compact_prompt
        # Persistent HTTP session — keeps TCP connection alive
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        logger.info(
            "OllamaLLMBackend initialised: model=%s, temp=%.2f, "
            "top_p=%.2f, num_ctx=%d, num_predict=%d, repeat_penalty=%.2f, "
            "first_token_timeout=%ds, chunk_timeout=%ds",
            self.model, self.temperature, self.top_p,
            self.num_ctx, self.num_predict, self.repeat_penalty,
            self.first_token_timeout, self.chunk_timeout,
        )
        # Optionally warm up the model so subsequent queries skip the
        # multi-second model-load phase.
        if warmup:
            self._warmup()

    # ------------------------------------------------------------------ #
    # Model warm-up (pre-load into memory)
    # ------------------------------------------------------------------ #

    def _warmup(self) -> None:
        """Send a minimal request to pre-load the model into memory.

        Ollama keeps models in RAM/VRAM after the first request, so
        a cheap warm-up eliminates the 5–10 s load delay on the first
        real query.  Uses ``num_predict=1`` and a tiny context window
        to minimise overhead.
        """
        try:
            t0 = time.monotonic()
            resp = self._session.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "options": {"num_predict": 1, "num_ctx": 128},
                },
                timeout=(10, self.first_token_timeout),
            )
            resp.raise_for_status()
            elapsed = time.monotonic() - t0
            logger.info(
                " Ollama warm-up complete: model=%s loaded in %.1fs",
                self.model, elapsed,
            )
        except requests.ConnectionError:
            logger.warning(
                " Ollama warm-up skipped: cannot connect to %s",
                self.base_url,
            )
        except Exception as e:
            logger.warning(" Ollama warm-up failed: %s", e)

    def _build_prompt(self, query: str, context: str, system_prompt: Optional[str] = None) -> str:
        """Build a single prompt string for the Ollama generate API."""
        system_msg = system_prompt if system_prompt is not None else _pick_system_prompt(
            context, compact=self.compact_prompt
        )
        user_msg = _build_user_message(query, context)
        return f"{system_msg}\n\n{user_msg}"

    def _build_messages(self, query: str, context: str, system_prompt: Optional[str] = None) -> list:
        """Build structured messages for the Ollama chat API.

        When *system_prompt* is given it replaces the default RAG/no-context
        system prompt.  This is important for internal pipeline steps
        (query rewriting, HyDE, classification) that need their own
        task-specific instructions.
        """
        sys_content = system_prompt if system_prompt is not None else _pick_system_prompt(
            context, compact=self.compact_prompt
        )
        user_content = query if system_prompt is not None else _build_user_message(query, context)
        return [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": user_content},
        ]

    def generate(self, query: str, context: str, *, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response using Ollama's chat API.

        Internally uses **streaming** with a per-chunk read timeout
        (``chunk_timeout``) to avoid the old global hard-timeout.
        Tokens are collected and returned as a single string.

        Falls back to a helpful error message if Ollama is unreachable.
        """
        try:
            tokens = list(
                self.generate_stream(query, context, system_prompt=system_prompt)
            )
            answer = "".join(tokens)
            if not answer:
                return "The LLM returned an empty response. Please try again."
            return answer
        except Exception as e:
            logger.error("Unexpected LLM error: %s", e, exc_info=True)
            return f"Unexpected error: {e}"

    def generate_stream(
        self, query: str, context: str, *, system_prompt: Optional[str] = None
    ) -> Generator[str, None, None]:
        """Stream tokens from Ollama's chat API with **two-tier timeout**.

        Tokens are buffered internally and flushed to the caller every
        ~1–2 seconds (``_STREAM_FLUSH_INTERVAL``) so the client
        refreshes smoothly without per-token rerender overhead.

        Two-tier timeout behaviour
        ~~~~~~~~~~~~~~~~~~~~~~~~~~
        - **Connection timeout**: 10 s (fast fail if Ollama is down).
        - **Socket read timeout**: ``self.first_token_timeout`` (default
          120 s).  This is the socket-level deadline applied by
          ``requests`` to each ``recv()`` call.  It must be large enough
          to cover model loading + prompt evaluation — Ollama sends
          **zero bytes** during those phases.
        - **Inter-chunk gap timeout**: ``self.chunk_timeout`` (default
          30 s).  After the first token arrives, if no new token
          appears within this window the stream is aborted.  This
          detects mid-generation hangs without penalising slow
          initial start-up.
        """
        messages = self._build_messages(query, context, system_prompt=system_prompt)

        # ── Query-aware num_predict ──────────────────────────────────
        # Short queries (< 30 words) get a smaller generation budget
        # to shave latency.  The hard ceiling is always enforced.
        effective_num_predict = self.num_predict
        query_words = len(query.split())
        if query_words < self._SIMPLE_QUERY_WORD_LIMIT:
            effective_num_predict = min(
                effective_num_predict, self._SIMPLE_QUERY_NUM_PREDICT,
            )
            logger.info(
                "Ollama: simple query (%d words < %d) "
                "num_predict reduced to %d",
                query_words, self._SIMPLE_QUERY_WORD_LIMIT,
                effective_num_predict,
            )
        effective_num_predict = min(
            effective_num_predict, self._MAX_NUM_PREDICT_CEILING,
        )

        # ── Prompt-size safety guard ────────────────────────────────
        # Estimate total input tokens.  If the prompt is too large for
        # the configured num_ctx, truncate the user message so that
        # prompt_eval stays within the first_token_timeout.
        # Rule of thumb: keep input < num_ctx - num_predict - 64 (headroom)
        _max_input = max(self.num_ctx - effective_num_predict - 64, 256)
        total_chars = sum(len(m.get("content", "")) for m in messages)
        _approx_tokens = total_chars // 4   # ~4 chars per token
        if _approx_tokens > _max_input:
            # Trim the longest message (usually the user message)
            trim_target = int(_max_input * 4)  # back to chars
            for m in reversed(messages):
                if len(m.get("content", "")) > 200:
                    m["content"] = m["content"][:trim_target] + (
                        "\n\n[Context truncated to fit model window]"
                    )
                    break
            logger.warning(
                "Prompt truncated: ~%d tokens exceeded num_ctx=%d "
                "(max_input=%d). Trimmed to ~%d chars.",
                _approx_tokens, self.num_ctx, _max_input, trim_target,
            )

        try:
            # Socket timeout = (connect, read).
            # read = first_token_timeout so the socket survives the
            # model-load + prompt-eval phase (can be 30–90 s on CPU).
            response = self._session.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "num_predict": effective_num_predict,
                        "num_ctx": self.num_ctx,
                        "repeat_penalty": self.repeat_penalty,
                    },
                },
                timeout=(10, self.first_token_timeout),
                stream=True,
            )
            response.raise_for_status()

            buffer: list[str] = []
            last_flush = time.monotonic()
            stream_start = time.monotonic()
            last_token_time = time.monotonic()
            token_count = 0
            first_token_received = False

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                token = data.get("message", {}).get("content", "")
                if token:
                    now = time.monotonic()

                    # ── TTFT (time to first token) logging ───────────
                    if not first_token_received:
                        first_token_received = True
                        ttft = now - stream_start
                        logger.info(
                            "Ollama TTFT: %.2fs (model=%s)",
                            ttft, self.model,
                        )

                    # ── Inter-chunk gap check ────────────────────────
                    # After the first token, abort if the gap between
                    # consecutive tokens exceeds chunk_timeout.
                    if first_token_received and token_count > 0:
                        gap = now - last_token_time
                        if gap > self.chunk_timeout:
                            logger.warning(
                                "Inter-chunk gap %.1fs > %ds — aborting "
                                "stream (model=%s, tokens so far=%d).",
                                gap, self.chunk_timeout,
                                self.model, token_count,
                            )
                            if buffer:
                                yield "".join(buffer)
                                buffer.clear()
                            yield (
                                f"\n\nModel stopped responding for "
                                f"{int(gap)}s mid-generation — stream "
                                f"aborted after {token_count} tokens."
                            )
                            response.close()
                            return

                    last_token_time = now
                    buffer.append(token)
                    token_count += 1

                # Time-based flush: yield accumulated tokens every
                # ~_STREAM_FLUSH_INTERVAL seconds (1–2 s) or when the
                # model signals completion.
                now = time.monotonic()
                is_done = data.get("done", False)
                if buffer and (is_done or now - last_flush >= _STREAM_FLUSH_INTERVAL):
                    yield "".join(buffer)
                    buffer.clear()
                    last_flush = now

                if is_done:
                    # ── Performance logging ──────────────────────────
                    total_s = now - stream_start
                    # Ollama may report eval_count in the final message
                    eval_count = data.get("eval_count", token_count)
                    tps = eval_count / total_s if total_s > 0 else 0.0
                    load_s = data.get("load_duration", 0) / 1e9
                    prompt_eval_s = data.get("prompt_eval_duration", 0) / 1e9
                    # eval_duration is generation time only (nanoseconds)
                    eval_dur_s = data.get("eval_duration", 0) / 1e9
                    gen_tps = (
                        eval_count / eval_dur_s
                        if eval_dur_s > 0 else tps
                    )
                    logger.info(
                        "Ollama perf: model=%s, tokens=%d, "
                        "time=%.2fs, tokens/sec=%.1f (gen=%.1f), "
                        "num_predict=%d, load=%.1fs, prompt_eval=%.1fs",
                        self.model, eval_count, total_s, tps, gen_tps,
                        effective_num_predict, load_s, prompt_eval_s,
                    )
                    break

            # Flush any remaining tokens left in the buffer
            if buffer:
                yield "".join(buffer)

            # Performance log for streams that ended without done=True
            if not token_count:
                logger.warning(
                    "Ollama stream ended with 0 tokens (model=%s).",
                    self.model,
                )

        except requests.ConnectionError:
            logger.error("Cannot connect to Ollama at %s", self.base_url)
            yield (
                "\u26a0\ufe0f **Cannot connect to Ollama.**\n\n"
                "Please ensure Ollama is running:\n"
                "1. Install from https://ollama.com/download\n"
                "2. Run: `ollama pull llama3` (or `mistral`)\n"
                "3. Ollama starts automatically, or run: `ollama serve`"
            )
        except requests.ReadTimeout:
            logger.error(
                "Ollama: no data received in %ds (first_token_timeout) "
                "— cancelling request (model=%s).",
                self.first_token_timeout, self.model,
            )
            yield (
                f"\u26a0\ufe0f **No response from Ollama for "
                f"{self.first_token_timeout}s** "
                "— request cancelled.\n\n"
                "The model may be loading or the context is too large.\n"
                "Try:\n"
                "- Increase `CENTURION_RAG_LLM_FIRST_TOKEN_TIMEOUT` "
                "in `.env`\n"
                f"- Use a smaller model (current: `{self.model}`)\n"
                "- Reduce `CENTURION_RAG_LLM_NUM_CTX`"
            )
        except requests.Timeout:
            logger.error("Ollama connection timed out.")
            yield (
                "\u26a0\ufe0f **Ollama connection timed out.**\n\n"
                "Ensure Ollama is running and reachable."
            )
        except requests.HTTPError as e:
            logger.error("Ollama HTTP error: %s", e)
            if "model" in str(e).lower() or (
                hasattr(response, "status_code") and response.status_code == 404
            ):
                yield (
                    f"\u26a0\ufe0f **Model '{self.model}' not found.**\n\n"
                    f"Pull it first: `ollama pull {self.model}`"
                )
            else:
                yield f"\u26a0\ufe0f LLM error: {e}"
        except Exception as e:
            logger.error("Ollama streaming error: %s", e, exc_info=True)
            yield f"\u26a0\ufe0f Streaming error: {e}"

    def is_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/tags", timeout=5
            )
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            # Ollama model names may include :latest suffix
            return any(
                self.model in m or m.startswith(self.model)
                for m in models
            )
        except Exception:
            return False

    def list_models(self) -> list:
        """List all models available in the local Ollama instance."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/tags", timeout=5
            )
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Anthropic Claude backend (cloud API)
# ---------------------------------------------------------------------------

#: The Claude models a user may pick on the RAG page (owner's ask, 17 Nov 2026; tracker RAG2).  Prices are US dollars
#: per million input / output tokens on Anthropic's API; ``effort`` is each model's setting for a cited answer from
#: retrieved context; ``fallbacks``: the API re-runs a declined request on another model by itself (not on Haiku).
CLAUDE_MODELS = (
    {"id": "claude-opus-5-5", "label": "Claude Opus 5.5", "input_usd": 4.0, "output_usd": 20.0,
     "effort": "medium", "fallbacks": True},
    {"id": "claude-sonnet-5-5", "label": "Claude Sonnet 5.5", "input_usd": 2.0, "output_usd": 10.0,
     "effort": "medium", "fallbacks": True},
    {"id": "claude-haiku-5-5", "label": "Claude Haiku 5.5", "input_usd": 0.10, "output_usd": 0.50,
     "effort": "medium", "fallbacks": False},
    {"id": "claude-fable-5-2", "label": "Claude Fable 5.2", "input_usd": 8.0, "output_usd": 40.0,
     "effort": "high", "fallbacks": True},
)
DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
#: A non-streaming answer's output cap (thinking included); streaming uses the config's larger cap.
NON_STREAM_MAX_TOKENS = 16000
#: What answered, for the page's "Answered by" line, by backend class.
BACKEND_LABELS = {"OllamaLLMBackend": "Ollama (local)", "HFInferenceLLMBackend": "Hugging Face Inference",
                  "OpenAILLMBackend": "OpenAI", "ClaudeLLMBackend": "Claude"}


def claude_spec(model: Optional[str]) -> dict:
    """The catalogue entry for ``model`` (the default when None); raises ValueError for one not offered."""
    wanted = model or DEFAULT_CLAUDE_MODEL
    for spec in CLAUDE_MODELS:
        if spec["id"] == wanted:
            return spec
    raise ValueError(f"model {wanted!r} is not offered (one of: {', '.join(s['id'] for s in CLAUDE_MODELS)})")


def claude_error_message(exc: Exception, model: str) -> str:
    """A user-facing line for an API failure, by the SDK's typed errors.  It starts with the warning sign, so the
    fallback chain answers with its free model instead."""
    import anthropic

    if isinstance(exc, anthropic.AuthenticationError):
        return ("⚠️ **Claude API key rejected.** The server needs an API key from the Anthropic Console "
                "(sk-ant-api...); a Claude Pro subscription does not include API access.")
    if isinstance(exc, anthropic.PermissionDeniedError):
        return "⚠️ **Claude API key lacks permission** for this request."
    if isinstance(exc, anthropic.NotFoundError):
        return f"⚠️ **Model not available to this API key:** {model}"
    if isinstance(exc, anthropic.RateLimitError):
        return "⚠️ **Claude rate limit reached.** Try again in a minute."
    if isinstance(exc, anthropic.BadRequestError):
        text = str(getattr(exc, "message", exc))
        if "credit balance" in text.lower():
            return "⚠️ **Claude credit balance too low.** Add credit in the Anthropic Console."
        return f"⚠️ **Claude rejected the request:** {text[:200]}"
    if isinstance(exc, anthropic.APIStatusError):
        return f"⚠️ **Claude API error {exc.status_code}** (retried by the SDK first)."
    if isinstance(exc, anthropic.APIConnectionError):
        return "⚠️ **Claude unreachable** (network)."
    return f"⚠️ **Claude error:** {type(exc).__name__}"


class ClaudeLLMBackend:
    """
    LLM backend using the Anthropic Messages API (``anthropic`` 2.x, the version the Space pins).

    The retrieved context and the question go to the chosen Claude model (``CLAUDE_MODELS``; per request through
    ``model``).  Current models always think before answering, so there is no thinking setting and no temperature,
    only each model's ``effort``; ``max_tokens`` leaves room for the thinking.  Opus, Sonnet and Fable requests carry
    the API's own refusal fallback.  The SDK retries rate limits, server errors and dropped connections itself.
    ``meta`` (a dict) receives which model answered.

    Needs ``ANTHROPIC_API_KEY``: an Anthropic Console API key, not a Claude Pro subscription.
    """

    accepts_model = True

    def __init__(self, api_key: str, model: str = DEFAULT_CLAUDE_MODEL, max_tokens: int = 64000) -> None:
        self.api_key = api_key
        self.model = claude_spec(model)["id"]
        self.max_tokens = max_tokens
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError("The 'anthropic' package is required for Claude: pip install anthropic")
            self._client = anthropic.Anthropic(api_key=self.api_key, max_retries=3)
            logger.info("Anthropic client initialised (default model=%s)", self.model)
        return self._client

    def _request(self, model: Optional[str], query: str, context: str, max_tokens: int) -> dict:
        spec = claude_spec(model or self.model)
        request = {"model": spec["id"], "max_tokens": max_tokens,
                   "system": _pick_system_prompt(context, compact=False),
                   "messages": [{"role": "user", "content": _build_user_message(query, context)}],
                   "output_config": {"effort": spec["effort"]}}
        if spec["fallbacks"]:
            request.update(betas=[FALLBACK_BETA], fallbacks="default")
        return request

    @staticmethod
    def _record(meta: Optional[dict], requested: str, message) -> None:
        if meta is not None:
            meta.update(source="llm", provider="Claude", model=str(getattr(message, "model", requested)),
                        requested=requested, stop_reason=str(getattr(message, "stop_reason", "")))

    @staticmethod
    def _refusal_text(message) -> str:
        details = getattr(message, "stop_details", None)
        category = getattr(details, "category", None) if details else None
        return ("Claude declined to answer this question"
                + (f" ({category})" if category else "") + ". Try rephrasing it.")

    def generate(self, query: str, context: str, model: Optional[str] = None, meta: Optional[dict] = None) -> str:
        """One answer, not streamed."""
        request = self._request(model, query, context, min(self.max_tokens, NON_STREAM_MAX_TOKENS))
        try:
            message = self.client.beta.messages.create(**request)
        except Exception as exc:                          # noqa: BLE001 - mapped to a user-facing line
            logger.warning("Claude request failed (%s)", type(exc).__name__)
            return claude_error_message(exc, request["model"])
        self._record(meta, request["model"], message)
        if message.stop_reason == "refusal":
            return self._refusal_text(message)
        answer = "".join(b.text for b in message.content if getattr(b, "type", "") == "text")
        logger.info("Claude answer (model=%s, input_tokens=%s, output_tokens=%s)", message.model,
                    getattr(message.usage, "input_tokens", "?"), getattr(message.usage, "output_tokens", "?"))
        return answer or "⚠️ **Claude returned an empty answer.**"

    def generate_stream(self, query: str, context: str, model: Optional[str] = None,
                        meta: Optional[dict] = None) -> Generator[str, None, None]:
        """The answer token by token; a failure yields one user-facing line instead."""
        request = self._request(model, query, context, self.max_tokens)
        try:
            with self.client.beta.messages.stream(**request) as stream:
                for text in stream.text_stream:
                    yield text
                message = stream.get_final_message()
        except Exception as exc:                          # noqa: BLE001 - mapped to a user-facing line
            logger.warning("Claude stream failed (%s)", type(exc).__name__)
            yield claude_error_message(exc, request["model"])
            return
        self._record(meta, request["model"], message)
        if message.stop_reason == "refusal":
            yield "\n\n" + self._refusal_text(message)

    def is_available(self) -> bool:
        """True when a key is configured (``claude_status`` checks that the API accepts it)."""
        return bool(self.api_key)


_claude_status: dict = {"at": None, "key": None, "value": None}


def claude_status(api_key: str, ttl_s: float = 600.0, clock=time.monotonic) -> dict:
    """Whether the API accepts the server's key: one free models call, cached ``ttl_s`` seconds.  ``note`` explains
    an absent or rejected key in plain words (the RAG page shows it beside the model picker)."""
    if not api_key:
        return {"available": False, "note": "No Claude API key on the server: answers come from the free fallback "
                                            "model. Add an Anthropic Console API key (sk-ant-api...)."}
    now = clock()
    cached = _claude_status
    if cached["value"] is not None and cached["key"] == api_key and now - cached["at"] < ttl_s:
        return cached["value"]
    import anthropic

    try:
        anthropic.Anthropic(api_key=api_key, max_retries=1).models.list(limit=1)
        value = {"available": True, "note": None}
    except anthropic.AuthenticationError:
        value = {"available": False, "note": "The server's Claude API key was rejected. It must be an API key from "
                                             "the Anthropic Console (sk-ant-api...): a Claude Pro subscription does "
                                             "not include API access. Until then answers come from the free "
                                             "fallback model."}
    except Exception as exc:                              # noqa: BLE001 - reported, checked again next time
        return {"available": False, "note": f"Claude could not be reached ({type(exc).__name__})."}
    cached.update(at=now, key=api_key, value=value)
    return value


# ---------------------------------------------------------------------------
# OpenAI backend (cloud API — GPT-4o, GPT-4, GPT-3.5-turbo, etc.)
# ---------------------------------------------------------------------------

class OpenAILLMBackend:
    """
    LLM backend using the OpenAI Chat Completions API.

    Requires the ``openai`` Python package::

        pip install openai

    Set ``OPENAI_API_KEY`` in your ``.env`` file.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        timeout: int = 600,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "The 'openai' package is required for OpenAI. "
                    "Install it with: pip install openai"
                )
            self._client = openai.OpenAI(
                api_key=self.api_key,
                timeout=self.timeout,
            )
            logger.info("OpenAI client initialised (model=%s, timeout=%ds)", self.model, self.timeout)
        return self._client

    def generate(self, query: str, context: str) -> str:
        """
        Generate a response using the OpenAI Chat Completions API.

        Retries up to 3 times on transient connection errors with
        exponential backoff (mirrors Claude retry logic).
        """
        system_prompt = _pick_system_prompt(context, compact=False)
        user_message = _build_user_message(query, context)

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                )

                answer = response.choices[0].message.content or ""

                if not answer:
                    logger.warning("OpenAI returned empty response")
                    return "The LLM returned an empty response. Please try again."

                usage = response.usage
                logger.info(
                    "OpenAI response generated (model=%s, prompt_tokens=%s, "
                    "completion_tokens=%s)",
                    self.model,
                    getattr(usage, "prompt_tokens", "?"),
                    getattr(usage, "completion_tokens", "?"),
                )
                return answer

            except Exception as e:
                error_type = type(e).__name__
                err_str = str(e).lower()

                # Non-retryable errors — fail immediately
                if "authentication" in err_str or "api_key" in err_str or "invalid" in err_str:
                    return (
                        "\u26a0\ufe0f **OpenAI authentication failed.**\n\n"
                        "Please check your `OPENAI_API_KEY` in the `.env` file."
                    )
                if "rate" in err_str:
                    return "\u26a0\ufe0f **Rate limited.** Please wait a moment and try again."
                if "model" in err_str and "not found" in err_str:
                    return (
                        f"\u26a0\ufe0f **Model '{self.model}' not available.**\n\n"
                        "Check your OpenAI plan or try a different model."
                    )
                if "billing" in err_str or "quota" in err_str or "insufficient" in err_str:
                    return (
                        "\u26a0\ufe0f **OpenAI quota/billing error.**\n\n"
                        "Please check your OpenAI billing dashboard."
                    )

                # Retryable (connection errors, timeouts, 5xx)
                is_connection = "connection" in error_type.lower() or "connect" in err_str
                is_timeout = "timeout" in err_str
                is_server = "500" in err_str or "502" in err_str or "503" in err_str
                retryable = is_connection or is_timeout or is_server

                if retryable and attempt < max_retries:
                    wait = 2 ** attempt  # 2s, 4s
                    logger.warning(
                        "OpenAI transient error (attempt %d/%d, retrying in %ds): %s",
                        attempt, max_retries, wait, e,
                    )
                    time.sleep(wait)
                    self._client = None  # reset to force fresh connection
                    continue

                logger.error("OpenAI API error (%s): %s", error_type, e, exc_info=True)
                return f"\u26a0\ufe0f OpenAI API error: {e}"

        return "\u26a0\ufe0f OpenAI API error after retries."

    def generate_stream(
        self, query: str, context: str
    ) -> Generator[str, None, None]:
        """
        Stream tokens from the OpenAI Chat Completions API.

        Uses ``stream=True`` on ``chat.completions.create()``.
        """
        system_prompt = _pick_system_prompt(context, compact=False)
        user_message = _build_user_message(query, context)

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )

            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield delta.content

        except Exception as e:
            logger.error("OpenAI streaming error: %s", e, exc_info=True)
            yield f"\u26a0\ufe0f OpenAI streaming error: {e}"

    def is_available(self) -> bool:
        """Check if the OpenAI API key is set and client can be created."""
        if not self.api_key:
            return False
        try:
            _ = self.client
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# HuggingFace Inference API backend (OpenAI-compatible, free with HF_TOKEN)
# ---------------------------------------------------------------------------

class HFInferenceLLMBackend:
    """
    LLM backend using HuggingFace's Inference API via the OpenAI SDK.

    HF Inference exposes an OpenAI-compatible ``/v1/chat/completions``
    endpoint, so we reuse the ``openai`` Python package with a custom
    ``base_url``.  This is the **cloud fallback** when Ollama is not
    available (e.g. on HuggingFace Spaces, Vercel, etc.).

    Set ``HF_TOKEN`` in environment (automatically available on HF Spaces).
    """

    _HF_BASE_URL = "https://router.huggingface.co/hf-inference/v1"

    def __init__(
        self,
        api_key: str,
        model: str = "mistralai/Mistral-7B-Instruct-v0.3",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        timeout: int = 120,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "The 'openai' package is required for HF Inference. "
                    "Install it with: pip install openai"
                )
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self._HF_BASE_URL,
                timeout=self.timeout,
            )
            logger.info(
                "HF Inference client initialised (model=%s)", self.model,
            )
        return self._client

    def generate(self, query: str, context: str) -> str:
        system_prompt = _pick_system_prompt(context, compact=False)
        user_message = _build_user_message(query, context)

        max_retries = 2
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                )
                answer = response.choices[0].message.content or ""
                if not answer:
                    return "The LLM returned an empty response. Please try again."
                logger.info("HF Inference response (model=%s)", self.model)
                return answer

            except Exception as e:
                err_str = str(e).lower()
                if "rate" in err_str or "429" in err_str:
                    return "\u26a0\ufe0f **HF Inference rate limited.** Please wait and retry."
                if attempt < max_retries:
                    time.sleep(2)
                    self._client = None
                    continue
                logger.error("HF Inference error: %s", e, exc_info=True)
                return f"\u26a0\ufe0f HF Inference error: {e}"

        return "\u26a0\ufe0f HF Inference error after retries."

    def generate_stream(
        self, query: str, context: str
    ) -> Generator[str, None, None]:
        system_prompt = _pick_system_prompt(context, compact=False)
        user_message = _build_user_message(query, context)

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield delta.content
        except Exception as e:
            logger.error("HF Inference streaming error: %s", e, exc_info=True)
            yield f"\u26a0\ufe0f HF Inference streaming error: {e}"

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            _ = self.client
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Factory: create the right backend from RAGConfig
# ---------------------------------------------------------------------------

_PROVIDERS = {
    "ollama": "OllamaLLMBackend",
    "claude": "ClaudeLLMBackend",
    "openai": "OpenAILLMBackend",
    "hf": "HFInferenceLLMBackend",
}


def _create_ollama_backend(config: RAGConfig) -> OllamaLLMBackend:
    """Helper to build an Ollama backend from config."""
    return OllamaLLMBackend(
        model=config.llm_model,
        base_url=config.llm_base_url,
        temperature=config.llm_temperature,
        top_p=config.llm_top_p,
        num_ctx=config.llm_num_ctx,
        num_predict=config.llm_num_predict,
        repeat_penalty=config.llm_repeat_penalty,
        max_tokens=config.llm_max_tokens,
        timeout=config.llm_timeout,
        chunk_timeout=config.llm_chunk_timeout,
        first_token_timeout=config.llm_first_token_timeout,
        compact_prompt=config.llm_compact_prompt,
    )


def _create_fallback_backend(config: RAGConfig):
    """Build the best available fallback backend.

    Priority:
    1. Ollama (if reachable at llm_base_url)
    2. HuggingFace Inference API (if HF_TOKEN is set)
    3. Raw Ollama backend (will produce a helpful error if unreachable)
    """
    ollama = _create_ollama_backend(config)
    if ollama.is_available():
        logger.info("Fallback: Ollama is available at %s", config.llm_base_url)
        return ollama

    if config.hf_api_key:
        logger.info(
            "Fallback: Ollama unreachable — using HF Inference API (model=%s)",
            config.hf_model,
        )
        return HFInferenceLLMBackend(
            api_key=config.hf_api_key,
            model=config.hf_model,
            temperature=config.hf_temperature,
            max_tokens=config.hf_max_tokens,
        )

    logger.warning("Fallback: Neither Ollama nor HF_TOKEN available")
    return ollama


def create_llm_backend(config: Optional[RAGConfig] = None):
    """
    Instantiate the appropriate LLM backend based on ``config.llm_provider``.

    Supported providers:
      - ``ollama``  — local Ollama inference (default)
      - ``claude``  — Anthropic Claude API (with smart fallback)
      - ``openai``  — OpenAI API (with smart fallback)
      - ``hf``      — HuggingFace Inference API

    Fallback order: Ollama → HF Inference API → error message.

    Returns an object with ``.generate(query, context) -> str`` and
    ``.generate_stream(query, context)`` methods.
    """
    config = config or RAGConfig()
    provider = config.llm_provider  # already lowered in config

    # Auto-detect: if provider is default "ollama" but an API key for a
    # cloud provider is available, promote it to primary automatically.
    # This lets containerised deployments (HF Spaces) work without Ollama.
    if provider == "ollama":
        if config.claude_api_key:
            provider = "claude"
            logger.info("Auto-detected ANTHROPIC_API_KEY — using Claude as primary LLM")
        elif config.openai_api_key:
            provider = "openai"
            logger.info("Auto-detected OPENAI_API_KEY — using OpenAI as primary LLM")
        elif config.hf_api_key:
            provider = "hf"
            logger.info("Auto-detected HF_TOKEN — using HF Inference as primary LLM")

    # --- Claude ---
    if provider == "claude":
        if not config.claude_api_key:
            logger.warning(
                "Claude selected but ANTHROPIC_API_KEY is not set — "
                "falling back to default backend."
            )
            return _FallbackLLMBackend("claude", "ANTHROPIC_API_KEY")
        primary = ClaudeLLMBackend(
            api_key=config.claude_api_key,
            model=config.claude_model,
            max_tokens=config.claude_max_tokens,
        )
        fallback = _create_fallback_backend(config)
        return _FallbackChainBackend(primary, fallback)

    # --- OpenAI ---
    if provider == "openai":
        if not config.openai_api_key:
            logger.warning(
                "OpenAI selected but OPENAI_API_KEY is not set — "
                "falling back to default backend."
            )
            return _FallbackLLMBackend("openai", "OPENAI_API_KEY")
        primary = OpenAILLMBackend(
            api_key=config.openai_api_key,
            model=config.openai_model,
            temperature=config.openai_temperature,
            max_tokens=config.openai_max_tokens,
            timeout=config.llm_timeout,
        )
        fallback = _create_fallback_backend(config)
        return _FallbackChainBackend(primary, fallback)

    # --- HuggingFace Inference ---
    if provider == "hf":
        if not config.hf_api_key:
            logger.warning("HF selected but HF_TOKEN not set")
            return _FallbackLLMBackend("hf", "HF_TOKEN")
        return HFInferenceLLMBackend(
            api_key=config.hf_api_key,
            model=config.hf_model,
            temperature=config.hf_temperature,
            max_tokens=config.hf_max_tokens,
        )

    # --- Ollama (default) ---
    if provider != "ollama":
        logger.warning(
            "Unknown LLM provider '%s'. Supported: %s. Falling back to Ollama.",
            provider, ", ".join(_PROVIDERS),
        )
    return _create_ollama_backend(config)


# ---------------------------------------------------------------------------
# Fallback chain: primary Ollama
# ---------------------------------------------------------------------------

class _FallbackChainBackend:
    """
    Wraps a primary LLM backend with an Ollama fallback.

    If ``primary.generate()`` returns an error message (starts with )
    or raises an exception, the request is transparently forwarded to
    the fallback (Ollama) backend.
    """

    def __init__(self, primary, fallback) -> None:
        self._primary = primary
        self._fallback = fallback
        self._primary_name = type(primary).__name__
        self._fallback_name = type(fallback).__name__
        logger.info(
            "LLM fallback chain: %s %s",
            self._primary_name, self._fallback_name,
        )

    accepts_model = True

    def _primary_kwargs(self, model: Optional[str], meta: Optional[dict]) -> dict:
        """The chosen model and the metadata dict, for a primary that takes them (Claude)."""
        return {"model": model, "meta": meta} if getattr(self._primary, "accepts_model", False) else {}

    def _fell_back(self, meta: Optional[dict], why: str) -> None:
        if meta is not None:
            meta.clear()
            meta.update(source="llm", provider=BACKEND_LABELS.get(self._fallback_name, self._fallback_name),
                        fallback=True, reason=why.replace("\u26a0\ufe0f", "").replace("**", "").strip()[:240])

    def generate(self, query: str, context: str, model: Optional[str] = None, meta: Optional[dict] = None) -> str:
        try:
            answer = self._primary.generate(query, context, **self._primary_kwargs(model, meta))
            # Detect error responses from the primary backend
            if self._is_error_token(answer):
                logger.warning(
                    "%s returned error — falling back to %s",
                    self._primary_name, self._fallback_name,
                )
                fallback_answer = self._fallback.generate(query, context)
                self._fell_back(meta, answer)
                # If fallback also errored, return a single clean message
                if self._is_error_token(fallback_answer):
                    return (
                        f"\u26a0\ufe0f **{self._primary_name} unavailable** "
                        f"(credit/billing issue) and **{self._fallback_name}** "
                        f"is not reachable.\n\n"
                        f"Please either top-up your Claude/OpenAI credits or "
                        f"ensure Ollama is running locally."
                    )
                # Fallback succeeded — return only the fallback answer
                return fallback_answer
            return answer
        except Exception as e:
            logger.error(
                "%s failed (%s) — falling back to %s",
                self._primary_name, e, self._fallback_name,
            )
            return self._fallback.generate(query, context)

    @staticmethod
    def _is_error_token(text: str) -> bool:
        """Return True if *text* looks like an LLM-backend error string."""
        return text.lstrip().startswith("\u26a0\ufe0f")

    @staticmethod
    def _is_user_actionable_error(text: str) -> bool:
        """Errors the user must fix (billing, auth) — don't fall back."""
        lower = text.lower()
        return any(kw in lower for kw in (
            "credit balance", "authentication failed",
            "billing", "api_key", "api key",
        ))

    def generate_stream(
        self, query: str, context: str, model: Optional[str] = None, meta: Optional[dict] = None
    ) -> Generator[str, None, None]:
        """Stream with immediate fallback on first-token error detection.

        Instead of buffering *all* primary tokens (which defeats
        streaming), we peek at the first token.  If it is an error
        marker we switch to the fallback immediately so the user sees
        output without delay.
        """
        try:
            primary_gen = self._primary.generate_stream(query, context, **self._primary_kwargs(model, meta))
            first_token = next(primary_gen, None)

            # Primary yielded nothing — fall back
            if first_token is None:
                logger.warning(
                    "%s stream yielded nothing — falling back to %s",
                    self._primary_name, self._fallback_name,
                )
                self._fell_back(meta, f"{self._primary_name} gave no answer")
                yield from self._fallback.generate_stream(query, context)
                return

            # First token is an error — try fallback silently
            if self._is_error_token(first_token):
                logger.warning(
                    "%s stream returned error — falling back to %s: %s",
                    self._primary_name, self._fallback_name,
                    first_token[:120],
                )
                # Buffer the first fallback token to check if IT also errors
                fallback_gen = self._fallback.generate_stream(query, context)
                fb_first = next(fallback_gen, None)
                if fb_first is None or self._is_error_token(fb_first):
                    # Both failed — yield a single clean error
                    yield (
                        f"\u26a0\ufe0f **{self._primary_name} unavailable** "
                        f"and **{self._fallback_name}** is not reachable.\n\n"
                        f"Please either top-up your Claude/OpenAI credits or "
                        f"ensure Ollama is running locally."
                    )
                    return
                # Fallback is healthy — stream it (no primary error shown; the reason goes in meta)
                self._fell_back(meta, first_token)
                yield fb_first
                yield from fallback_gen
                return

            # Primary is healthy — stream normally
            yield first_token
            for token in primary_gen:
                # Mid-stream error (rare but possible)
                if self._is_error_token(token):
                    logger.warning(
                        "%s mid-stream error — falling back to %s",
                        self._primary_name, self._fallback_name,
                    )
                    yield (
                        f"\n\n\u26a0\ufe0f *{self._primary_name} failed mid-stream — "
                        f"re-generating via Ollama…*\n\n"
                    )
                    self._fell_back(meta, token)
                    yield from self._fallback.generate_stream(query, context)
                    return
                yield token

        except Exception as e:
            logger.error(
                "%s stream failed (%s) — falling back to %s",
                self._primary_name, e, self._fallback_name,
            )
            yield (
                f"\u26a0\ufe0f *{self._primary_name} unavailable — "
                f"falling back to {self._fallback_name}*\n\n"
            )
            self._fell_back(meta, f"{self._primary_name} unavailable ({type(e).__name__})")
            yield from self._fallback.generate_stream(query, context)

    def is_available(self) -> bool:
        return self._primary.is_available() or self._fallback.is_available()


class _FallbackLLMBackend:
    """Helpful fallback that tells the user what to configure."""

    def __init__(self, provider: str, missing_key: str = "") -> None:
        self._provider = provider
        self._missing_key = missing_key

    def generate(self, query: str, context: str) -> str:
        if self._missing_key:
            return (
                f"\u26a0\ufe0f **{self._provider.title()} is selected but "
                f"`{self._missing_key}` is not set.**\n\n"
                f"Add it to the project root `.env` and restart the app."
            )
        supported = ", ".join(f"`{p}`" for p in _PROVIDERS)
        return (
            f"\u26a0\ufe0f **Unknown LLM provider `{self._provider}`.**\n\n"
            f"Set `CENTURION_RAG_LLM_PROVIDER` to one of: {supported}"
        )

    def generate_stream(
        self, query: str, context: str
    ) -> Generator[str, None, None]:
        """Fallback streaming — yields the full error message at once."""
        yield self.generate(query, context)

    def is_available(self) -> bool:
        return False
