"""
Deterministic Query Classifier for Centurion Capital LLC RAG Pipeline.

Classifies a raw user query string into a structured dict containing:
    - **intent** – ``"code_generation"`` | ``"analysis"`` | ``"conceptual"``
    - **pipeline_stage** – ``list[str]`` of matched pipeline stages
    - **needs_code** – ``bool`` (``True`` when intent is code_generation
      *or* the query explicitly references code artefacts)

All classification is **rule-based**: deterministic keyword matching,
no ML models, no LLM calls.

Pipeline-stage keywords once mirrored the chunk tags of ``chunking.py``
(retired 25 Nov 2026); today they drive query routing only.

Usage::

    from rag_pipeline.core.query_classifier import classify_query

    result = classify_query("Write a function to calculate Sharpe ratio")
    # {
    #     "intent": "code_generation",
    #     "pipeline_stage": ["evaluation"],
    #     "needs_code": True,
    # }
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# 1. Intent keywords
# ═══════════════════════════════════════════════════════════════════════════
#
# Order matters: rules are evaluated top-to-bottom and the **first**
# matching intent wins.  This gives ``code_generation`` priority over
# ``analysis`` when both could match (e.g. "implement an optimization").
# ═══════════════════════════════════════════════════════════════════════════

_INTENT_RULES: List[Tuple[str, re.Pattern]] = [
    # ── code_generation ──────────────────────────────────────────────
    (
        "code_generation",
        re.compile(
            r"\b(?:"
            r"write|implement|create\s+(?:a\s+)?(?:function|class|module|script)"
            r"|code(?:\s+(?:for|to|that|a|an|the))?"
            r"|function\b"
            r"|def\s+\w+"
            r"|snippet"
            r"|generate\s+(?:a\s+)?(?:function|class|code|script)"
            r"|refactor"
            r"|debug"
            r"|fix\s+(?:the\s+)?(?:bug|error|issue|code)"
            r"|add\s+(?:a\s+)?(?:method|parameter|class|test)"
            r"|build\s+(?:a\s+)?(?:\w+\s+)?(?:function|module|pipeline|class)"
            r"|program"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    # ── analysis ─────────────────────────────────────────────────────
    (
        "analysis",
        re.compile(
            r"\b(?:"
            r"optimize|optimise"
            r"|improve"
            r"|reduce\s+(?:the\s+)?(?:drawdown|risk|volatility|loss|exposure)"
            r"|increase\s+(?:the\s+)?(?:sharpe|returns|alpha|hit\s+rate)"
            r"|compare"
            r"|evaluate"
            r"|benchmark"
            r"|backtest"
            r"|analyze|analyse"
            r"|diagnose"
            r"|profile"
            r"|measure"
            r"|performance"
            r"|trade-off|tradeoff"
            r"|sensitivity"
            r"|what\s+(?:is|are)\s+the\s+(?:best|optimal|top)"
            r"|how\s+(?:much|many|well|does|did|do)"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    # ── conceptual (catch-all explanation / theory queries) ──────────
    (
        "conceptual",
        re.compile(
            r"\b(?:"
            r"why"
            r"|explain"
            r"|what\s+(?:is|are|does|do)\b"
            r"|how\s+(?:does|do|is|are)\b"
            r"|define"
            r"|definition"
            r"|describe"
            r"|overview"
            r"|summary"
            r"|summarize|summarise"
            r"|concept"
            r"|theory"
            r"|meaning"
            r"|difference\s+between"
            r"|when\s+(?:should|to|do)"
            r"|tell\s+me"
            r")\b",
            re.IGNORECASE,
        ),
    ),
]

_DEFAULT_INTENT = "conceptual"

# ═══════════════════════════════════════════════════════════════════════════
# 2. Pipeline-stage keywords
# ═══════════════════════════════════════════════════════════════════════════
#
# Intentionally kept in sync with ``chunking._PIPELINE_STAGE_RULES``.
# Any stage that scores ≥ 1 keyword hit is included in the output list.
# ═══════════════════════════════════════════════════════════════════════════

_PIPELINE_STAGE_RULES: Dict[str, List[str]] = {
    "risk_management": [
        "drawdown", "max drawdown", "risk", "stop-loss", "stop loss",
        "stoploss", "var ", "value at risk", "position sizing",
        "risk parity", "tail risk", "volatility target",
        "kelly criterion", "risk budget", "exposure limit",
        "margin call", "liquidation", "hedging",
    ],
    "evaluation": [
        "sharpe", "sharpe ratio", "sortino", "calmar", "returns",
        "performance", "benchmark", "alpha", "beta ", "information ratio",
        "treynor", "omega ratio", "annualized", "cumulative return",
        "hit rate", "profit factor", "payoff ratio", "win rate",
        "loss rate", "expectancy",
    ],
    "signal_generation": [
        "signal", "z-score", "zscore", "z score", "indicator",
        "oscillator", "macd", "rsi", "bollinger", "moving average",
        "crossover", "divergence", "stochastic", "atr ",
        "adx", "obv", "volume signal", "buy signal", "sell signal",
        "entry signal", "exit signal", "trigger",
    ],
    "data_processing": [
        "data cleaning", "missing data", "imputation", "outlier",
        "feature engineering", "normalization", "standardization",
        "scaling", "resampling", "ohlcv", "tick data", "bar data",
        "data pipeline", "etl", "data quality", "data schema",
        "time series", "frequency",
    ],
    "backtesting": [
        "backtest", "backtesting", "historical simulation",
        "walk-forward", "walk forward", "out-of-sample",
        "in-sample", "paper trading", "simulation",
        "event-driven", "vectorized backtest", "lookback",
    ],
    "portfolio_construction": [
        "portfolio", "allocation", "rebalancing", "diversification",
        "mean-variance", "mean variance", "efficient frontier",
        "black-litterman", "minimum variance", "equal weight",
        "risk parity portfolio", "correlation matrix", "covariance",
        "asset allocation",
    ],
    "execution": [
        "execution", "order", "slippage", "market impact",
        "transaction cost", "latency", "fill rate",
        "limit order", "market order", "twap", "vwap",
        "smart order", "broker", "api call",
    ],
    "machine_learning": [
        "machine learning", "deep learning", "neural network",
        "random forest", "gradient boosting", "xgboost",
        "lstm", "transformer", "training", "validation",
        "cross-validation", "hyperparameter", "overfitting",
        "regularization", "feature importance", "model selection",
        "ensemble", "classification", "regression",
    ],
    "market_microstructure": [
        "bid-ask", "spread", "order book", "market maker",
        "liquidity", "price impact", "tick size",
        "high-frequency", "hft", "latency arbitrage",
    ],
}

# Pre-compile for fast matching
_STAGE_PATTERNS: Dict[str, re.Pattern] = {}
for _stage, _keywords in _PIPELINE_STAGE_RULES.items():
    _parts = []
    for kw in _keywords:
        escaped = re.escape(kw)
        if len(kw) <= 5:
            _parts.append(rf"\b{escaped}\b")
        else:
            _parts.append(escaped)
    _STAGE_PATTERNS[_stage] = re.compile("|".join(_parts), re.IGNORECASE)

# ═══════════════════════════════════════════════════════════════════════════
# 3. "needs_code" secondary keywords
# ═══════════════════════════════════════════════════════════════════════════
#
# Even if the intent isn't ``code_generation``, some queries implicitly
# need code context (e.g. "explain the `calculate_sharpe` function").
# ═══════════════════════════════════════════════════════════════════════════

_CODE_HINT_RE = re.compile(
    r"\b(?:"
    r"function|class|method|variable|module|import|def |return |lambda"
    r"|script|snippet|source code|codebase|repository|repo\b"
    r"|`.+`"                       # inline code in backticks
    r"|\.py\b"                     # python file reference
    r")\b",
    re.IGNORECASE,
)

# ═══════════════════════════════════════════════════════════════════════════
# 3b. Strict code-mode keywords
# ═══════════════════════════════════════════════════════════════════════════
#
# When a query contains any of these strong code-intent signals the
# classifier forces ``intent = "code_generation"``, ``needs_code = True``,
# ``priority = "code_strict"``, and ``strict_code_mode = True``.
# This overrides the normal first-match intent logic so that mixed
# queries like "what is X? how can I implement using python?" are
# correctly routed to code retrieval.
# ═══════════════════════════════════════════════════════════════════════════

_STRICT_CODE_RE = re.compile(
    r"\b(?:"
    r"implement"
    r"|python"
    r"|code"
    r"|function"
    r"|using\s+python"
    r")\b",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Public API
# ═══════════════════════════════════════════════════════════════════════════

def classify_intent(query: str) -> str:
    """Classify the user's intent from the query string.

    Evaluates intent rules in priority order and returns the first
    match.  Falls back to ``"conceptual"`` if no rule fires.

    Args:
        query: Raw user query string.

    Returns:
        One of ``"code_generation"``, ``"analysis"``, ``"conceptual"``.
    """
    for intent, pattern in _INTENT_RULES:
        if pattern.search(query):
            return intent
    return _DEFAULT_INTENT


def classify_pipeline_stages(query: str) -> List[str]:
    """Detect all matching pipeline stages in the query.

    Every stage whose keyword pattern matches at least once is
    included.  Stages are returned in a deterministic order
    (alphabetically sorted).

    Args:
        query: Raw user query string.

    Returns:
        Sorted list of stage names (may be empty).
    """
    stages: List[str] = []
    for stage, pattern in _STAGE_PATTERNS.items():
        if pattern.search(query):
            stages.append(stage)
    stages.sort()
    return stages


def needs_code(query: str, intent: str) -> bool:
    """Determine whether the query needs code context.

    Returns ``True`` when:
    - intent is ``"code_generation"``, **or**
    - the query contains explicit code-artefact references
      (function names, ``.py`` files, backtick code snippets, etc.)

    Args:
        query: Raw user query string.
        intent: Already-classified intent string.

    Returns:
        Boolean.
    """
    if intent == "code_generation":
        return True
    return bool(_CODE_HINT_RE.search(query))


def classify_query(query: str) -> Dict[str, Any]:
    """Classify a user query into intent, pipeline stages, and code need.

    This is the primary entry point for the module.

    Args:
        query: Raw user query string.

    Returns:
        Dict with keys::

            {
                "intent":           str,   # "code_generation" | "analysis" | "conceptual"
                "pipeline_stage":   list[str],
                "needs_code":       bool,
                "strict_code_mode": bool,   # True when strong code keywords detected
                "priority":         str,    # "code_strict" | "normal"
            }

        When the query contains strong code-intent keywords ("implement",
        "python", "code", "function", "using python") the classifier
        forces ``intent = "code_generation"``, ``needs_code = True``,
        ``priority = "code_strict"``, and ``strict_code_mode = True``
        regardless of which intent pattern matched first.

    Examples::

        >>> classify_query("Write a function to calculate Sharpe ratio")
        {'intent': 'code_generation', 'pipeline_stage': ['evaluation'], 'needs_code': True, 'strict_code_mode': True, 'priority': 'code_strict'}

        >>> classify_query("Why does the z-score signal lag on volatile days?")
        {'intent': 'conceptual', 'pipeline_stage': ['signal_generation'], 'needs_code': False, 'strict_code_mode': False, 'priority': 'normal'}

        >>> classify_query("Optimize the drawdown limit for momentum strategy")
        {'intent': 'analysis', 'pipeline_stage': ['risk_management'], 'needs_code': False, 'strict_code_mode': False, 'priority': 'normal'}
    """
    if not query or not query.strip():
        logger.warning("classify_query called with empty query.")
        return {
            "intent": _DEFAULT_INTENT,
            "pipeline_stage": [],
            "needs_code": False,
            "strict_code_mode": False,
            "priority": "normal",
        }

    query_clean = query.strip()
    intent = classify_intent(query_clean)
    stages = classify_pipeline_stages(query_clean)
    code_needed = needs_code(query_clean, intent)

    # ── Strict code-mode override ────────────────────────────────
    strict = bool(_STRICT_CODE_RE.search(query_clean))
    if strict:
        intent = "code_generation"
        code_needed = True
        priority = "code_strict"
    else:
        priority = "normal"

    logger.debug(
        "classify_query(%r) intent=%s, stages=%s, needs_code=%s, "
        "strict_code_mode=%s, priority=%s",
        query_clean[:80], intent, stages, code_needed, strict, priority,
    )
    return {
        "intent": intent,
        "pipeline_stage": stages,
        "needs_code": code_needed,
        "strict_code_mode": strict,
        "priority": priority,
    }
