"""
Explanatory Video Generation API
─────────────────────────────────
The connection point with the video-generation partner service. This app
owns the request (POST /generate) and the poll (GET /jobs/{id}); the video
worker -- in-process, a separate service, a script, whatever it ends up
being -- owns claiming pending work (GET /jobs?status=pending) and
reporting the result back (PATCH /jobs/{id}). Neither side needs to share
Python code: the video_jobs table + these four endpoints are the whole
contract.
"""
import uuid

from fastapi import APIRouter, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from app.config import get_settings, get_tenant_id
from app.models.db import get_engine
from app.models.schemas import (
    VideoGenerateRequest,
    VideoJobOut,
    VideoJobStatus,
    VideoJobUpdateRequest,
)

router = APIRouter(prefix="/api/v1/video", tags=["video"])

_engine = None
_SessionLocal = None

_COLUMNS = (
    "id, tenant_id, session_id, input_text, title, language, status, "
    "video_url, error_message, created_at"
)


def _get_session():
    global _engine, _SessionLocal
    if _engine is None:
        # The process-wide engine + pool (app.models.db), not a private
        # one -- see that module for why four independent pools for the
        # same database URL was a real resource problem. The globals stay
        # so tests can monkeypatch an in-memory SQLite engine in.
        _engine = get_engine()
        _SessionLocal = sessionmaker(bind=_engine)
    return _SessionLocal()


def _row_to_out(row) -> VideoJobOut:
    return VideoJobOut(
        id=str(row.id),
        tenant_id=row.tenant_id,
        session_id=row.session_id,
        input_text=row.input_text,
        title=row.title,
        language=row.language,
        status=VideoJobStatus(row.status),
        video_url=row.video_url,
        error_message=row.error_message,
        created_at=row.created_at,
    )


@router.post("/generate", response_model=VideoJobOut)
def generate_video(request: VideoGenerateRequest):
    """Our side: create a pending job and return immediately -- the caller
    polls GET /jobs/{id} for the result, same pattern as file upload."""
    tenant_id = request.tenant_id or get_tenant_id()
    job_id = uuid.uuid4()
    session = _get_session()
    try:
        session.execute(
            text(
                "INSERT INTO video_jobs (id, tenant_id, session_id, input_text, title, language, status) "
                "VALUES (:id, :tenant_id, :session_id, :input_text, :title, :language, 'pending')"
            ),
            {
                "id": str(job_id),
                "tenant_id": tenant_id,
                "session_id": request.session_id,
                "input_text": request.text,
                "title": request.title,
                "language": request.language.value,
            },
        )
        session.commit()
        row = session.execute(
            text(f"SELECT {_COLUMNS} FROM video_jobs WHERE id = :id"), {"id": str(job_id)}
        ).fetchone()
    finally:
        session.close()
    return _row_to_out(row)


@router.get("/jobs", response_model=list[VideoJobOut])
def list_jobs(status: VideoJobStatus = None):
    """Video worker's side: poll with ?status=pending to claim work."""
    session = _get_session()
    try:
        if status is not None:
            rows = session.execute(
                text(f"SELECT {_COLUMNS} FROM video_jobs WHERE status = :s ORDER BY created_at"),
                {"s": status.value},
            ).fetchall()
        else:
            rows = session.execute(
                text(f"SELECT {_COLUMNS} FROM video_jobs ORDER BY created_at DESC")
            ).fetchall()
    finally:
        session.close()
    return [_row_to_out(r) for r in rows]


@router.get("/jobs/{job_id}", response_model=VideoJobOut)
def get_job(job_id: str):
    """Our side: poll for the result."""
    session = _get_session()
    try:
        row = session.execute(
            text(f"SELECT {_COLUMNS} FROM video_jobs WHERE id = :id"), {"id": job_id}
        ).fetchone()
    finally:
        session.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Video job not found.")
    return _row_to_out(row)


@router.patch("/jobs/{job_id}", response_model=VideoJobOut)
def update_job(job_id: str, request: VideoJobUpdateRequest):
    """Video worker's side: report progress/result. Required fields per
    status: 'ready' -> video_url, 'error' -> error_message."""
    if request.status == VideoJobStatus.READY and not request.video_url:
        raise HTTPException(status_code=400, detail="video_url is required when status='ready'.")
    if request.status == VideoJobStatus.ERROR and not request.error_message:
        raise HTTPException(status_code=400, detail="error_message is required when status='error'.")

    session = _get_session()
    try:
        result = session.execute(
            text(
                "UPDATE video_jobs SET status = :status, video_url = :video_url, "
                "error_message = :error_message, updated_at = now() WHERE id = :id"
            ),
            {
                "status": request.status.value,
                "video_url": request.video_url,
                "error_message": request.error_message,
                "id": job_id,
            },
        )
        if result.rowcount == 0:
            session.rollback()
            raise HTTPException(status_code=404, detail="Video job not found.")
        session.commit()
        row = session.execute(
            text(f"SELECT {_COLUMNS} FROM video_jobs WHERE id = :id"), {"id": job_id}
        ).fetchone()
    finally:
        session.close()
    return _row_to_out(row)
