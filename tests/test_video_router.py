"""
Tests for app/routers/video.py -- the explanatory-video API, and the
Admin/Tenant-only rule on generating one.

Same pattern as tests/test_ingest_router.py: router functions called
directly (no TestClient/HTTP layer anywhere in this suite) against an
in-memory SQLite engine monkeypatched into app.routers.video._engine /
_SessionLocal.

The role gate is testable this way precisely because generate_video takes
`role` as a parameter with a dependency default rather than declaring
`dependencies=[Depends(...)]` on the route -- a route-level dependency
runs only inside FastAPI's request cycle and would be invisible to a
direct call, i.e. the gate would look tested while never actually running.
"""
import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.errors import ForbiddenError
from app.models.database import Base, VideoJob
from app.models.schemas import (
    Language,
    VideoGenerateRequest,
    VideoJobStatus,
    VideoJobUpdateRequest,
    VideoMode,
)
from app.routers import video as video_router
from app.services.roles import Role


@pytest.fixture
def sqlite_session(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine, tables=[VideoJob.__table__])
    SessionLocal = sessionmaker(bind=engine)
    monkeypatch.setattr(video_router, "_engine", engine)
    monkeypatch.setattr(video_router, "_SessionLocal", SessionLocal)
    return SessionLocal


def _request(**overrides) -> VideoGenerateRequest:
    payload = {
        "text": "Pendant une ronde de nuit, le port du gilet réfléchissant est obligatoire.",
        "title": "Ronde de nuit",
        "language": Language.FRENCH,
    }
    payload.update(overrides)
    return VideoGenerateRequest(**payload)


# ---------------------------------------------------------------------------
# The rule: only Admins and Tenants generate videos
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("role", [Role.ADMIN, Role.TENANT])
def test_course_authors_can_generate(sqlite_session, role):
    job = video_router.generate_video(_request(), role=role)
    assert job.status == VideoJobStatus.PENDING
    assert job.input_text.startswith("Pendant une ronde")


def test_employee_cannot_generate(sqlite_session):
    """The 2026-09-21 requirement correction: a tenant's employees consume
    courses, they do not author them."""
    with pytest.raises(ForbiddenError) as excinfo:
        video_router.generate_video(_request(), role=Role.EMPLOYEE)

    assert excinfo.value.status_code == 403
    assert excinfo.value.code == "ROLE_FORBIDDEN"
    # The message must name both the refused role and what would be
    # allowed -- the frontend shows it verbatim in a toast.
    assert "employee" in excinfo.value.message
    assert "admin" in excinfo.value.message and "tenant" in excinfo.value.message


def test_refused_generate_writes_no_row(sqlite_session):
    """A refusal must not leave a job behind for the worker to pick up."""
    with pytest.raises(ForbiddenError):
        video_router.generate_video(_request(), role=Role.EMPLOYEE)

    session = sqlite_session()
    try:
        assert session.execute(text("SELECT COUNT(*) FROM video_jobs")).scalar() == 0
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Role resolution from the header / configured default
# ---------------------------------------------------------------------------

def test_header_absent_falls_back_to_configured_default(monkeypatch):
    from app.config import get_settings
    from app.services import roles

    settings = get_settings().model_copy()
    monkeypatch.setattr(roles, "get_settings", lambda: settings)

    settings.default_user_role = "tenant"
    assert roles.get_role(None) is Role.TENANT

    # Flipping the default is how a deployment becomes deny-by-default.
    settings.default_user_role = "employee"
    assert roles.get_role(None) is Role.EMPLOYEE


def test_unknown_role_falls_back_rather_than_erroring(monkeypatch):
    """A stale or typo'd header degrades to the default (and is then
    refused if that default cannot author), instead of returning a shape
    error that enumerates valid role names to a probing caller."""
    from app.config import get_settings
    from app.services import roles

    settings = get_settings().model_copy()
    settings.default_user_role = "employee"
    monkeypatch.setattr(roles, "get_settings", lambda: settings)

    assert roles.get_role("superuser") is Role.EMPLOYEE


def test_role_header_is_case_insensitive():
    from app.services.roles import resolve_role

    assert resolve_role("ADMIN") is Role.ADMIN
    assert resolve_role(" Tenant ") is Role.TENANT


# ---------------------------------------------------------------------------
# mode (scene | avatar)
# ---------------------------------------------------------------------------

def test_mode_defaults_to_scene(sqlite_session):
    assert video_router.generate_video(_request(), role=Role.TENANT).mode == VideoMode.SCENE


def test_avatar_mode_round_trips(sqlite_session):
    """Avatar is stored as asked, not silently rewritten to scene -- the
    worker refuses it with a truthful reason instead (that gap is upstream,
    and hiding it here would make the failure look like ours)."""
    job = video_router.generate_video(_request(mode=VideoMode.AVATAR), role=Role.ADMIN)
    assert job.mode == VideoMode.AVATAR
    assert video_router.get_job(job.id).mode == VideoMode.AVATAR


# ---------------------------------------------------------------------------
# Tenant scoping
# ---------------------------------------------------------------------------

def test_list_jobs_only_returns_the_active_tenant(sqlite_session, monkeypatch):
    """Regression: list_jobs used to return every tenant's rows, including
    input_text -- the tutor's generated course content."""
    mine = video_router.generate_video(_request(), role=Role.TENANT)

    session = sqlite_session()
    try:
        session.execute(
            text(
                "INSERT INTO video_jobs (id, tenant_id, input_text, language, mode, status) "
                "VALUES (:id, 'someone_else', 'their private content', 'fr', 'scene', 'pending')"
            ),
            {"id": str(uuid.uuid4())},
        )
        session.commit()
    finally:
        session.close()

    listed = video_router.list_jobs()
    assert [j.id for j in listed] == [mine.id]

    pending = video_router.list_jobs(status=VideoJobStatus.PENDING)
    assert [j.id for j in pending] == [mine.id]
    assert all("private" not in j.input_text for j in pending)


# ---------------------------------------------------------------------------
# The worker's report path
# ---------------------------------------------------------------------------

def test_worker_reports_ready(sqlite_session):
    job = video_router.generate_video(_request(), role=Role.TENANT)
    updated = video_router.update_job(
        job.id,
        VideoJobUpdateRequest(
            status=VideoJobStatus.READY, video_url=f"/media/{job.id}/video_finale.mp4"
        ),
    )
    assert updated.status == VideoJobStatus.READY
    assert updated.video_url.endswith("video_finale.mp4")


def test_worker_reports_error(sqlite_session):
    job = video_router.generate_video(_request(), role=Role.TENANT)
    updated = video_router.update_job(
        job.id,
        VideoJobUpdateRequest(status=VideoJobStatus.ERROR, error_message="ffmpeg exploded"),
    )
    assert updated.status == VideoJobStatus.ERROR
    assert updated.error_message == "ffmpeg exploded"


@pytest.mark.parametrize(
    "update",
    [
        VideoJobUpdateRequest(status=VideoJobStatus.READY),
        VideoJobUpdateRequest(status=VideoJobStatus.ERROR),
    ],
)
def test_terminal_status_requires_its_payload(sqlite_session, update):
    job = video_router.generate_video(_request(), role=Role.TENANT)
    with pytest.raises(HTTPException) as excinfo:
        video_router.update_job(job.id, update)
    assert excinfo.value.status_code == 400


def test_update_unknown_job_is_404(sqlite_session):
    with pytest.raises(HTTPException) as excinfo:
        video_router.update_job(
            str(uuid.uuid4()),
            VideoJobUpdateRequest(status=VideoJobStatus.PROCESSING),
        )
    assert excinfo.value.status_code == 404
