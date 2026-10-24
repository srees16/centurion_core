"""
Health check API endpoints.
"""

import asyncio
import logging
import time
from datetime import datetime

from fastapi import APIRouter

from api.schemas.common import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])

#: How long a database check stands.  The Space's Docker HEALTHCHECK calls /health every 30 s and the frontend's
#: keep-alive cron pings it too; a SELECT 1 each time kept Neon awake around the clock (it scales to zero only
#: after 5 idle minutes) and used up the free plan's 100 CU-hours by 20 Oct 2026 (Sentry CENTURION-CORE-4B).
DB_CHECK_INTERVAL_S = 6 * 3600
_db_status = {"checked_at": 0.0, "ok": False}


def _check_database() -> bool:
    from api.dependencies import get_db_service
    return get_db_service() is not None


async def _database_ok(force: bool = False) -> bool:
    """The database's status, re-checked at most every ``DB_CHECK_INTERVAL_S`` (or when ``force``)."""
    now = time.monotonic()
    if force or not _db_status["checked_at"] or now - _db_status["checked_at"] > DB_CHECK_INTERVAL_S:
        try:
            _db_status["ok"] = await asyncio.to_thread(_check_database)
        except Exception:                                   # noqa: BLE001 - a health probe never raises
            _db_status["ok"] = False
        _db_status["checked_at"] = now
    return _db_status["ok"]


@router.api_route(
    "/health",
    methods=["GET", "HEAD"],
    response_model=HealthResponse,
    summary="System health check",
)
async def health_check(deep: bool = False):
    """
    Check the health of all system components.  The database is re-checked at most every 6 hours, so the
    30-second probes let Neon scale to zero; ``?deep=1`` checks it now.
    """
    components: dict[str, bool] = {}

    # Database
    db_ok = await _database_ok(force=deep)
    components["database"] = db_ok

    # RAG / ChromaDB
    rag_ok = False
    try:
        from rag_pipeline.storage.vector_store import VectorStoreManager
        vs = VectorStoreManager()
        stats = vs.get_stats()
        rag_ok = stats is not None
    except Exception:
        pass
    components["rag_vector_store"] = rag_ok

    # Cache backend
    cache_info = {}
    try:
        from infrastructure.cache import cache
        cache_info = cache.health_check()
    except Exception:
        cache_info = {"backend": "unknown", "healthy": False}
    components["cache"] = cache_info.get("healthy", False)

    overall = "healthy" if db_ok else "degraded"

    return HealthResponse(
        status=overall,
        database=db_ok,
        version="1.0.0",
        timestamp=datetime.utcnow(),
        components={**components, "cache_detail": cache_info},
    )


@router.get(
    "/health/infra",
    summary="Infrastructure monitoring dashboard",
)
async def infra_health():
    """
    Detailed infrastructure health including event bus stats,
    latency tracking, execution mode, and event throughput.
    """
    from infrastructure.latency_tracker import latency_tracker
    from infrastructure.execution_context import execution_ctx
    from infrastructure.event_bus import event_bus

    latency_data = {}
    for label, stats in latency_tracker.get_all_stats().items():
        latency_data[label] = {
            "count": stats.count,
            "p50_ms": round(stats.p50_ms, 2),
            "p95_ms": round(stats.p95_ms, 2),
            "p99_ms": round(stats.p99_ms, 2),
            "max_ms": round(stats.max_ms, 2),
            "sla_breaches": stats.sla_breaches,
        }

    try:
        from infrastructure.monitoring import monitoring_service
        monitoring_data = monitoring_service.get_health()
    except Exception:
        monitoring_data = {}

    return {
        "execution_mode": execution_ctx.mode,
        "event_bus_topics": event_bus.topics,
        "event_log_size": len(event_bus.get_log()),
        "latency": latency_data,
        "monitoring": monitoring_data,
    }


@router.get(
    "/health/scheduler",
    summary="Scheduler job execution log",
)
async def scheduler_job_log(limit: int = 50, job_id: str | None = None):
    """
    Return recent scheduler job executions with status, timing, and error details.

    - **limit**: max rows to return (default 50)
    - **job_id**: optional filter by job id (e.g. ``pre_market_pipeline``)
    """
    from scheduler import get_job_log

    rows = get_job_log(limit=limit, job_id=job_id)
    return {"jobs": rows, "count": len(rows)}
