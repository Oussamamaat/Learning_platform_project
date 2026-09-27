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

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from app.config import get_settings, get_tenant_id
from app.models.db import get_engine
from app.models.schemas import (
    VideoGenerateRequest,
    VideoJobOut,
    VideoJobStatus,
    VideoMode,
    VideoJobUpdateRequest,
)
from app.services.roles import COURSE_AUTHOR_ROLES, Role, get_role, require_role

router = APIRouter(prefix="/api/v1/video", tags=["video"])

_engine = None
_SessionLocal = None

_COLUMNS = (
    "id, tenant_id, session_id, input_text, title, language, mode, status, "
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
        mode=VideoMode(row.mode),
        status=VideoJobStatus(row.status),
        video_url=row.video_url,
        error_message=row.error_message,
        created_at=row.created_at,
    )


@router.post("/generate", response_model=VideoJobOut)
def generate_video(request: VideoGenerateRequest, role: Role = Depends(get_role)):
    """Our side: create a pending job and return immediately -- the caller
    polls GET /jobs/{id} for the result, same pattern as file upload.

    Admin/Tenant only (2026-09-21). Generating a video is part of AUTHORING
    a course, not taking one, so a tenant's employees are refused here even
    though they use every other endpoint on this service. `role` is a
    parameter with a dependency default rather than a route-level
    `dependencies=[...]` so this suite's direct-call tests (no TestClient
    anywhere -- see tests/test_ingest_router.py) can exercise the gate by
    passing role= explicitly.

    The gate is deliberately on THIS endpoint only. The other three are the
    partner-facing contract frozen 2026-08-18
    (docs/PARTNER_VIDEO_ONBOARDING.md); gating their worker's poll and
    report calls needs a shared secret between the two services, which is
    authentication and a larger change than this one. Recorded as an open
    gap in docs/architecture/video-generation-interface.md rather than
    half-done here.
    """
    require_role(role, COURSE_AUTHOR_ROLES, "generate videos")

    tenant_id = request.tenant_id or get_tenant_id()
    job_id = uuid.uuid4()
    session = _get_session()
    try:
        session.execute(
            text(
                # created_at/updated_at are written EXPLICITLY. VideoJob
                # declares them with a Python-side `default=` (app/models/
                # database.py), which SQLAlchemy applies only on an ORM
                # insert -- this is raw SQL, so the column took no default
                # at all and landed NULL, and _row_to_out then failed
                # VideoJobOut validation ("created_at: Input should be a
                # valid datetime"). POST /generate therefore 500'd on any
                # database, not just in tests; it went unnoticed because
                # nothing called this endpoint until the Studio UI and the
                # worker landed. Setting it here rather than adding a
                # server_default keeps it correct on already-created
                # tables, with no migration to run.
                "INSERT INTO video_jobs "
                "(id, tenant_id, session_id, input_text, title, language, mode, status, created_at, updated_at) "
                "VALUES (:id, :tenant_id, :session_id, :input_text, :title, :language, :mode, 'pending', "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            ),
            {
                "id": str(job_id),
                "tenant_id": tenant_id,
                "session_id": request.session_id,
                "input_text": request.text,
                "title": request.title,
                "language": request.language.value,
                "mode": request.mode.value,
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
    """Video worker's side: poll with ?status=pending to claim work. Also
    the Studio UI's side: called without a filter to list this tenant's
    jobs newest-first.

    Scoped to get_tenant_id() as of 2026-09-21. It previously returned
    EVERY tenant's rows -- including their input_text, which is the
    tutor's generated course content. That was invisible while nothing
    called it; the Studio UI is the first caller that would have rendered
    another tenant's jobs on screen. Single-tenant MVP means this is
    currently a no-op in practice (ADR 0001: get_tenant_id is a
    process-lifetime constant), which is exactly why it had to be fixed
    before a second tenant makes it a real leak rather than after.
    """
    tenant_id = get_tenant_id()
    session = _get_session()
    try:
        if status is not None:
            rows = session.execute(
                text(
                    f"SELECT {_COLUMNS} FROM video_jobs "
                    "WHERE tenant_id = :t AND status = :s ORDER BY created_at"
                ),
                {"t": tenant_id, "s": status.value},
            ).fetchall()
        else:
            rows = session.execute(
                text(
                    f"SELECT {_COLUMNS} FROM video_jobs "
                    "WHERE tenant_id = :t ORDER BY created_at DESC"
                ),
                {"t": tenant_id},
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
                # CURRENT_TIMESTAMP, not now(): identical in Postgres
                # (both are transaction-start timestamptz) but now() is a
                # Postgres extension, so this statement could not run
                # against the in-memory SQLite the router tests use --
                # which left the whole PATCH path, the one the video
                # worker reports every result through, untestable.
                "UPDATE video_jobs SET status = :status, video_url = :video_url, "
                "error_message = :error_message, updated_at = CURRENT_TIMESTAMP WHERE id = :id"
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
