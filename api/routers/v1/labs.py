"""/api/v1/{fml,tts,aronson,ehlers,vince}/* routes: the book labs' chapter runs.

Moved from v1_gateway.py (tracker H4), which includes this router under /api/v1.
"""

import asyncio
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.dependencies import get_db_service
from api.routers.v1.common import logger

router = APIRouter()


class ChapterRunRequest(BaseModel):
    chapters: List[str]
    tickers: Optional[List[str]] = None
    date_start: Optional[str] = None
    date_end: Optional[str] = None


# ─── Financial ML ────────────────────────────────────────────────────────

@router.get("/fml/chapters")
async def fml_chapters():
    """List available Financial ML chapters."""
    try:
        from references.financial_ml.applied import get_chapters
        return get_chapters()
    except ImportError:
        # Fallback: scan the readings directory for chapter files
        try:
            from pathlib import Path
            fml_dir = Path(__file__).resolve().parent.parent.parent / "financial_ML" / "readings"
            chapters = []
            if fml_dir.exists():
                for f in sorted(fml_dir.iterdir()):
                    if f.suffix == ".py" and not f.name.startswith("_"):
                        chapters.append({
                            "key": f.stem,
                            "title": f.stem.replace("_", " ").title(),
                            "category": "Readings",
                        })
            return chapters
        except Exception:
            return []


@router.post("/fml/run")
async def fml_run(req: ChapterRunRequest):
    """Run selected Financial ML chapters. Returns a batch_id for progress tracking."""
    import uuid
    batch_id = str(uuid.uuid4())

    # Start async execution
    try:
        from references.financial_ml.applied import run_chapters_async
        asyncio.create_task(run_chapters_async(
            batch_id, req.chapters,
            tickers=req.tickers,
            date_start=req.date_start,
            date_end=req.date_end,
        ))
    except ImportError:
        logger.warning("FML module not found — run will be a no-op")

    return {"batch_id": batch_id}


@router.post("/fml/abort/{batch_id}")
async def fml_abort(batch_id: str):
    """Abort a running FML batch."""
    try:
        from references.financial_ml.applied import abort_batch
        ok = abort_batch(batch_id)
        return {"aborted": ok}
    except ImportError:
        raise HTTPException(404, "FML module not available")


@router.get("/fml/progress/{batch_id}")
async def fml_progress(batch_id: str):
    """SSE stream for FML batch progress."""
    async def event_stream():
        try:
            from references.financial_ml.applied import get_batch_progress
            import json
            while True:
                progress = get_batch_progress(batch_id)
                if progress:
                    yield f"data: {json.dumps(progress)}\n\n"
                    if progress.get("completed", 0) >= progress.get("total", 1):
                        break
                    if progress.get("status") == "aborted":
                        break
                await asyncio.sleep(1)
        except ImportError:
            import json
            yield f"data: {json.dumps({'batch_id': batch_id, 'total': 0, 'completed': 0, 'chapters': {}})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/fml/history")
async def fml_history(page: int = 1, limit: int = 50):
    """Get Financial ML batch run history."""
    db = get_db_service()
    if not db:
        return {"data": [], "total": 0}
    try:
        data = db.get_fml_history(page=page, limit=limit)
        total = db.count_fml_runs()
        return {"data": data, "total": total}
    except Exception:
        return {"data": [], "total": 0}


# ─── Test & Tune Trading Systems ─────────────────────────────────────────

@router.get("/tts/chapters")
async def tts_chapters():
    """List available Test & Tune chapters."""
    try:
        from references.testune.applied import get_chapters
        return get_chapters()
    except ImportError:
        try:
            from pathlib import Path
            tts_dir = Path(__file__).resolve().parent.parent.parent / "testune_trade_sys"
            chapters = []
            if tts_dir.exists():
                for f in sorted(tts_dir.iterdir()):
                    if f.suffix == ".py" and not f.name.startswith("_"):
                        chapters.append({
                            "key": f.stem,
                            "title": f.stem.replace("_", " ").title(),
                            "category": "Trading",
                        })
            return chapters
        except Exception:
            return []


@router.post("/tts/run")
async def tts_run(req: ChapterRunRequest):
    """Run selected Test & Tune chapters."""
    import uuid
    batch_id = str(uuid.uuid4())

    try:
        from references.testune.applied import run_chapters_async
        asyncio.create_task(run_chapters_async(
            batch_id, req.chapters,
            tickers=req.tickers,
            date_start=req.date_start,
            date_end=req.date_end,
        ))
    except ImportError:
        logger.warning("TTS module not found — run will be a no-op")

    return {"batch_id": batch_id}


@router.post("/tts/abort/{batch_id}")
async def tts_abort(batch_id: str):
    """Abort a running TTS batch."""
    try:
        from references.testune.applied import abort_batch
        ok = abort_batch(batch_id)
        return {"aborted": ok}
    except ImportError:
        raise HTTPException(404, "TTS module not available")


@router.get("/tts/progress/{batch_id}")
async def tts_progress(batch_id: str):
    """SSE stream for TTS batch progress."""
    async def event_stream():
        try:
            from references.testune.applied import get_batch_progress
            import json
            while True:
                progress = get_batch_progress(batch_id)
                if progress:
                    yield f"data: {json.dumps(progress)}\n\n"
                    if progress.get("completed", 0) >= progress.get("total", 1):
                        break
                    if progress.get("status") == "aborted":
                        break
                await asyncio.sleep(1)
        except ImportError:
            import json
            yield f"data: {json.dumps({'batch_id': batch_id, 'total': 0, 'completed': 0, 'chapters': {}})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/tts/history")
async def tts_history(page: int = 1, limit: int = 50):
    """Get Test & Tune batch run history."""
    db = get_db_service()
    if not db:
        return {"data": [], "total": 0}
    try:
        data = db.get_tts_history(page=page, limit=limit)
        total = db.count_tts_runs()
        return {"data": data, "total": total}
    except Exception:
        return {"data": [], "total": 0}


# ─── Aronson Validator Lab ───────────────────────────────────────────────

@router.get("/aronson/chapters")
async def aronson_chapters():
    """List available Aronson EBTA chapters."""
    try:
        from references.aronson.applied import get_chapters
        return get_chapters()
    except ImportError:
        return []


@router.post("/aronson/run")
async def aronson_run(req: ChapterRunRequest):
    """Run selected Aronson chapters."""
    import uuid
    batch_id = str(uuid.uuid4())
    try:
        from references.aronson.applied import run_chapters_async
        asyncio.create_task(run_chapters_async(
            batch_id, req.chapters,
            tickers=req.tickers, date_start=req.date_start, date_end=req.date_end,
        ))
    except ImportError:
        logger.warning("Aronson module not found — run will be a no-op")
    return {"batch_id": batch_id}


@router.post("/aronson/abort/{batch_id}")
async def aronson_abort(batch_id: str):
    """Abort a running Aronson batch."""
    try:
        from references.aronson.applied import abort_batch
        return {"aborted": abort_batch(batch_id)}
    except ImportError:
        raise HTTPException(404, "Aronson module not available")


@router.get("/aronson/progress/{batch_id}")
async def aronson_progress(batch_id: str):
    """SSE stream for Aronson batch progress."""
    async def event_stream():
        try:
            from references.aronson.applied import get_batch_progress
            import json
            while True:
                progress = get_batch_progress(batch_id)
                if progress:
                    yield f"data: {json.dumps(progress)}\n\n"
                    if progress.get("completed", 0) >= progress.get("total", 1):
                        break
                    if progress.get("status") == "aborted":
                        break
                await asyncio.sleep(1)
        except ImportError:
            import json
            yield f"data: {json.dumps({'batch_id': batch_id, 'total': 0, 'completed': 0, 'chapters': {}})}\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/aronson/history")
async def aronson_history(page: int = 1, limit: int = 50):
    """Get Aronson Lab batch run history."""
    db = get_db_service()
    if not db:
        return {"data": [], "total": 0}
    try:
        data = db.get_lab_history("aronson", page=page, limit=limit)
        total = db.count_lab_runs("aronson")
        return {"data": data, "total": total}
    except Exception:
        return {"data": [], "total": 0}


# ─── Ehlers DSP Lab ─────────────────────────────────────────────────────

@router.get("/ehlers/chapters")
async def ehlers_chapters():
    """List available Ehlers DSP chapters."""
    try:
        from references.ehlers.applied import get_chapters
        return get_chapters()
    except ImportError:
        return []


@router.post("/ehlers/run")
async def ehlers_run(req: ChapterRunRequest):
    """Run selected Ehlers chapters."""
    import uuid
    batch_id = str(uuid.uuid4())
    try:
        from references.ehlers.applied import run_chapters_async
        asyncio.create_task(run_chapters_async(
            batch_id, req.chapters,
            tickers=req.tickers, date_start=req.date_start, date_end=req.date_end,
        ))
    except ImportError:
        logger.warning("Ehlers module not found — run will be a no-op")
    return {"batch_id": batch_id}


@router.post("/ehlers/abort/{batch_id}")
async def ehlers_abort(batch_id: str):
    """Abort a running Ehlers batch."""
    try:
        from references.ehlers.applied import abort_batch
        return {"aborted": abort_batch(batch_id)}
    except ImportError:
        raise HTTPException(404, "Ehlers module not available")


@router.get("/ehlers/progress/{batch_id}")
async def ehlers_progress(batch_id: str):
    """SSE stream for Ehlers batch progress."""
    async def event_stream():
        try:
            from references.ehlers.applied import get_batch_progress
            import json
            while True:
                progress = get_batch_progress(batch_id)
                if progress:
                    yield f"data: {json.dumps(progress)}\n\n"
                    if progress.get("completed", 0) >= progress.get("total", 1):
                        break
                    if progress.get("status") == "aborted":
                        break
                await asyncio.sleep(1)
        except ImportError:
            import json
            yield f"data: {json.dumps({'batch_id': batch_id, 'total': 0, 'completed': 0, 'chapters': {}})}\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/ehlers/history")
async def ehlers_history(page: int = 1, limit: int = 50):
    """Get Ehlers DSP Lab batch run history."""
    db = get_db_service()
    if not db:
        return {"data": [], "total": 0}
    try:
        data = db.get_lab_history("ehlers", page=page, limit=limit)
        total = db.count_lab_runs("ehlers")
        return {"data": data, "total": total}
    except Exception:
        return {"data": [], "total": 0}


# ─── Vince Risk Lab ─────────────────────────────────────────────────────

@router.get("/vince/chapters")
async def vince_chapters():
    """List available Vince Risk Lab chapters."""
    try:
        from references.vince.applied import get_chapters
        return get_chapters()
    except ImportError:
        return []


@router.post("/vince/run")
async def vince_run(req: ChapterRunRequest):
    """Run selected Vince chapters."""
    import uuid
    batch_id = str(uuid.uuid4())
    try:
        from references.vince.applied import run_chapters_async
        asyncio.create_task(run_chapters_async(
            batch_id, req.chapters,
            tickers=req.tickers, date_start=req.date_start, date_end=req.date_end,
        ))
    except ImportError:
        logger.warning("Vince module not found — run will be a no-op")
    return {"batch_id": batch_id}


@router.post("/vince/abort/{batch_id}")
async def vince_abort(batch_id: str):
    """Abort a running Vince batch."""
    try:
        from references.vince.applied import abort_batch
        return {"aborted": abort_batch(batch_id)}
    except ImportError:
        raise HTTPException(404, "Vince module not available")


@router.get("/vince/progress/{batch_id}")
async def vince_progress(batch_id: str):
    """SSE stream for Vince batch progress."""
    async def event_stream():
        try:
            from references.vince.applied import get_batch_progress
            import json
            while True:
                progress = get_batch_progress(batch_id)
                if progress:
                    yield f"data: {json.dumps(progress)}\n\n"
                    if progress.get("completed", 0) >= progress.get("total", 1):
                        break
                    if progress.get("status") == "aborted":
                        break
                await asyncio.sleep(1)
        except ImportError:
            import json
            yield f"data: {json.dumps({'batch_id': batch_id, 'total': 0, 'completed': 0, 'chapters': {}})}\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/vince/history")
async def vince_history(page: int = 1, limit: int = 50):
    """Get Vince Risk Lab batch run history."""
    db = get_db_service()
    if not db:
        return {"data": [], "total": 0}
    try:
        data = db.get_lab_history("vince", page=page, limit=limit)
        total = db.count_lab_runs("vince")
        return {"data": data, "total": total}
    except Exception:
        return {"data": [], "total": 0}
