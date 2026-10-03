"""/api/v1/rag/* routes: sources, uploads, ingest status, queries.

Moved from v1_gateway.py (tracker H4), which includes this router under /api/v1.
"""

import asyncio
from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse

from api.dependencies import get_rag_engine
from api.routers.v1.common import logger

router = APIRouter()


# ─── RAG Pipeline ────────────────────────────────────────────────────────

@router.get("/rag/sources")
async def rag_sources():
    """List knowledge base sources."""
    engine = get_rag_engine()
    if not engine:
        return []
    try:
        names = await asyncio.to_thread(engine._vs.list_sources)
        results = []
        for name in names:
            details = await asyncio.to_thread(engine._vs.get_source_details, name)
            results.append({
                "id": name,
                "name": name,
                "type": name.rsplit(".", 1)[-1] if "." in name else "pdf",
                "doc_count": 1,
                "chunk_count": details.get("chunks", 0),
                "page_count": details.get("page_count"),
                "ingested_at": details.get("ingested_at"),
                "file_size_bytes": details.get("file_size_bytes"),
            })
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/upload")
async def rag_upload(files: List[UploadFile] = File(...)):
    """Upload and ingest documents asynchronously in background threads."""
    from rag_pipeline.ingestion.background_ingest import get_ingestion_manager

    mgr = get_ingestion_manager()
    tasks = []
    for file in files:
        content = await file.read()
        task = mgr.submit(file.filename or "unknown", content)
        tasks.append({"task_id": task.task_id, "file_name": task.file_name, "status": task.status.value})
    return {"submitted": len(tasks), "tasks": tasks}


@router.get("/rag/ingest-status")
async def rag_ingest_status():
    """Poll ingestion task status for all active and recently completed tasks."""
    from rag_pipeline.ingestion.background_ingest import get_ingestion_manager

    mgr = get_ingestion_manager()
    active = mgr.get_active_tasks()
    recent = mgr.get_recently_completed(max_age_s=300)
    all_tasks = active + recent
    return [
        {
            "task_id": t.task_id,
            "file_name": t.file_name,
            "status": t.status.value,
            "stage": t.stage,
            "stage_pct": t.stage_pct,
            "error": t.error,
        }
        for t in all_tasks
    ]


@router.delete("/rag/sources/{source_id}")
async def rag_delete_source(source_id: str):
    """Delete a document source from the knowledge base."""
    engine = get_rag_engine()
    if not engine:
        raise HTTPException(status_code=503, detail="RAG engine unavailable")
    try:
        from rag_pipeline.ingestion.pdf_ingestion import PDFIngestionService

        svc = PDFIngestionService(
            vector_store=engine._vs,
            config=engine._config,
            embedding_service=engine._embedder,
            on_change_callback=engine.invalidate_cache,
        )
        deleted = await asyncio.to_thread(svc.delete_source, source_id)
        return {"deleted": True, "chunks_removed": deleted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag/query")
async def rag_query(
    q: str,
    rag: str = "true",
    sources: str = "",
    token: str = "",
):
    """SSE streaming RAG query with real token-by-token LLM output.

    Uses ``query_stream()``, the engine's streaming pipeline, so retrieval,
    context-building and LLM generation match the non-streaming query.
    """
    engine = get_rag_engine()
    rag_enabled = rag.lower() == "true"
    source_ids = [s for s in sources.split(",") if s] if sources else None

    async def event_stream():
        import json

        if not engine or not rag_enabled:
            try:
                from rag_pipeline.llm.llm_service import create_llm_backend
                llm = create_llm_backend()
                tokens = llm.generate_stream(q, "")
                for tok in tokens:
                    yield f"event: token\ndata: {json.dumps(tok)}\n\n"
                yield f"event: done\ndata: done\n\n"
            except Exception as e:
                yield f"event: token\ndata: {json.dumps(f'Error: {e}')}\n\n"
                yield f"event: done\ndata: done\n\n"
            return

        try:
            source_filter = source_ids[0] if source_ids and len(source_ids) == 1 else None

            # Use query_stream() — real token-by-token streaming.
            stream_gen = engine.query_stream(q, source_filter=source_filter)

            # query_stream() is a blocking generator; iterate in a
            # thread so we don't block the asyncio event loop.
            import queue, threading

            token_queue: queue.Queue = queue.Queue()
            _SENTINEL = object()

            def _run_stream():
                try:
                    for tok in stream_gen:
                        token_queue.put(tok)
                except Exception as exc:
                    token_queue.put(exc)
                finally:
                    token_queue.put(_SENTINEL)

            thread = threading.Thread(target=_run_stream, daemon=True)
            thread.start()

            while True:
                # Wait for the next token (with a generous timeout
                # to cover model-loading / prompt-eval pauses).
                try:
                    item = await asyncio.to_thread(token_queue.get, True, 300)
                except Exception:
                    break

                if item is _SENTINEL:
                    break
                if isinstance(item, Exception):
                    yield f"event: token\ndata: {json.dumps(f'Error: {item}')}\n\n"
                    break

                yield f"event: token\ndata: {json.dumps(item)}\n\n"

            yield f"event: done\ndata: done\n\n"
        except Exception as e:
            logger.exception("RAG query SSE error")
            yield f"event: token\ndata: {json.dumps(f'Error: {e}')}\n\n"
            yield f"event: done\ndata: done\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
