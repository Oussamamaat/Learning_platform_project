"""
Tests for app/routers/ingest.py -- the tenant document upload API.

Same pattern as tests/test_chat.py and tests/test_sources.py: router
functions called directly (no TestClient/HTTP layer anywhere in this
suite), against an in-memory SQLite engine (monkeypatching
app.routers.ingest._engine/_SessionLocal) for the DB-touching endpoints.
app.services.ingest_queue.submit is monkeypatched to a no-op so these
stay unit tests of the router's own validate/store/respond logic, not the
async ingestion pipeline (that's app/services/ingest_jobs.py's own concern,
plus the real end-to-end coverage in probe_upload_e2e.py).

upload_sources is the one async endpoint here -- called via asyncio.run(),
no pytest-asyncio needed for a single top-level await per test.
"""
import asyncio
import io
import uuid
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi import HTTPException, UploadFile
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.models.database import Base, Document, SourceFile
from app.models.schemas import SourceStatus, SourceToggleRequest
from app.routers import ingest as ingest_router


@pytest.fixture
def sqlite_session(monkeypatch, tmp_path):
    """Also patches app.routers.ingest.get_settings to a real Settings
    instance with upload_dir pointed at tmp_path -- mutating that SAME
    instance's attributes (see set_setting below) is far more reliable
    than monkeypatching Settings the pydantic CLASS, whose fields are
    resolved through pydantic-settings' own validation/env machinery
    rather than plain class attributes (a class-level monkeypatch was
    silently ignored in practice).
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine, tables=[SourceFile.__table__, Document.__table__])
    SessionLocal = sessionmaker(bind=engine)
    monkeypatch.setattr(ingest_router, "_engine", engine)
    monkeypatch.setattr(ingest_router, "_SessionLocal", SessionLocal)

    from app.config import get_settings
    test_settings = get_settings().model_copy()
    test_settings.upload_dir = str(tmp_path / "uploads")
    monkeypatch.setattr(ingest_router, "get_settings", lambda: test_settings)
    return SessionLocal


def set_setting(monkeypatch, name: str, value) -> None:
    """Mutate the SAME Settings instance app.routers.ingest.get_settings
    returns (installed by the sqlite_session fixture above) for one test's
    scenario -- e.g. set_setting(monkeypatch, "uploads_read_only", True)."""
    setattr(ingest_router.get_settings(), name, value)


def _upload_file(filename: str, content: bytes = b"# Title\n\nBody text.") -> UploadFile:
    return UploadFile(filename=filename, file=io.BytesIO(content))


def _make_source(session_factory, **overrides):
    new_id = uuid.uuid4()
    defaults = dict(
        id=new_id,
        tenant_id="company_abc",
        filename="doc.pdf",
        stored_path="/tmp/doc.pdf",
        content_type="application/pdf",
        size_bytes=1234,
        sha256="abc123",
        status="ready",
        enabled=True,
    )
    defaults.update(overrides)
    session = session_factory()
    try:
        session.add(SourceFile(**defaults))
        session.commit()
    finally:
        session.close()
    return new_id.hex


# --- upload_sources: validation -------------------------------------------

def test_legacy_binary_upload_rejected_with_actionable_message(sqlite_session):
    with patch.object(ingest_router, "ingest_queue"):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(ingest_router.upload_sources(files=[_upload_file("old.doc")], domain=None))
    assert exc_info.value.status_code == 400
    assert ".docx" in exc_info.value.detail


def test_unsupported_extension_rejected(sqlite_session):
    with patch.object(ingest_router, "ingest_queue"):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(ingest_router.upload_sources(files=[_upload_file("doc.rtf")], domain=None))
    assert exc_info.value.status_code == 400


def test_oversize_upload_rejected(sqlite_session, monkeypatch):
    set_setting(monkeypatch, "max_upload_bytes", 10)
    with patch.object(ingest_router, "ingest_queue"):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(ingest_router.upload_sources(
                files=[_upload_file("doc.txt", b"way more than ten bytes of content")], domain=None,
            ))
    assert exc_info.value.status_code == 413


def test_too_many_files_in_one_request_rejected(sqlite_session, monkeypatch):
    set_setting(monkeypatch, "max_upload_files_per_request", 2)
    files = [_upload_file(f"doc{i}.txt") for i in range(3)]
    with patch.object(ingest_router, "ingest_queue"):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(ingest_router.upload_sources(files=files, domain=None))
    assert exc_info.value.status_code == 413


def test_uploads_read_only_blocks_upload(sqlite_session, monkeypatch):
    set_setting(monkeypatch, "uploads_read_only", True)
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(ingest_router.upload_sources(files=[_upload_file("doc.txt")], domain=None))
    assert exc_info.value.status_code == 403


# --- upload_sources: happy path + idempotency -----------------------------

def test_successful_upload_creates_pending_row_and_submits_job(sqlite_session):
    with patch.object(ingest_router, "ingest_queue") as mock_queue:
        results = asyncio.run(ingest_router.upload_sources(
            files=[_upload_file("guide.md", b"# Guide\n\nContent here.")], domain=None,
        ))
    assert len(results) == 1
    assert results[0].status == "pending"
    assert results[0].filename == "guide.md"
    mock_queue.submit.assert_called_once()
    # process_source_file, source_id positional, stored_path/tenant_id kwargs
    call = mock_queue.submit.call_args
    assert call.kwargs["tenant_id"] == "company_abc"


def test_duplicate_sha256_upload_short_circuits_without_resubmitting(sqlite_session):
    content = b"# Same content\n\nEvery time."
    with patch.object(ingest_router, "ingest_queue") as mock_queue:
        first = asyncio.run(ingest_router.upload_sources(files=[_upload_file("a.md", content)], domain=None))
        # Mark that row 'ready' so the second upload's sha256 lookup matches
        # it. Raw SQL, not session.get(SourceFile, ...) -- this row was
        # inserted by the router's own raw text() INSERT, which stores the
        # id in whatever string form it was bound with (dashed, since the
        # router passes str(uuid_obj)); the ORM's UUID column type stores
        # SQLite values in dashless hex on its own insert path, so an
        # ORM-level lookup for a raw-SQL-inserted row silently misses under
        # SQLite specifically (real Postgres has no such divergence -- see
        # tests/test_sources.py's _make_source docstring for the same class
        # of quirk). Matching the router's own raw-SQL style sidesteps it.
        session = sqlite_session()
        try:
            session.execute(
                text("UPDATE source_files SET status = 'ready' WHERE id = :id"),
                {"id": first[0].id},
            )
            session.commit()
        finally:
            session.close()

        second = asyncio.run(ingest_router.upload_sources(files=[_upload_file("b.md", content)], domain=None))

    assert second[0].duplicate_of == first[0].id
    mock_queue.submit.assert_called_once()  # NOT called again for the duplicate


# --- list_sources / get_source ---------------------------------------------

def test_list_sources_reports_ready_count_and_total_chunks(sqlite_session):
    _make_source(sqlite_session, status="ready", chunk_count=3)
    _make_source(sqlite_session, status="ready", chunk_count=5)
    _make_source(sqlite_session, status="processing", chunk_count=0)

    result = ingest_router.list_sources()
    assert len(result.sources) == 3
    assert result.ready_count == 2
    assert result.total_chunks == 8


def test_list_sources_scoped_to_tenant(sqlite_session):
    _make_source(sqlite_session, tenant_id="other-tenant", status="ready")
    result = ingest_router.list_sources()
    assert result.sources == []


def test_get_source_404_for_unknown_id(sqlite_session):
    with pytest.raises(HTTPException) as exc_info:
        ingest_router.get_source("nonexistent-id")
    assert exc_info.value.status_code == 404


def test_get_source_returns_the_row(sqlite_session):
    source_id = _make_source(sqlite_session, filename="report.pdf")
    result = ingest_router.get_source(source_id)
    assert result.filename == "report.pdf"


def test_get_source_decodes_unprocessed_pages_on_partial(sqlite_session):
    """SQLite has no native JSON cast on a raw text() SELECT (unlike
    psycopg2 against a real Postgres json/jsonb column, which
    auto-deserializes) -- _row_to_out's _decode_unprocessed_pages must
    handle the stored string explicitly rather than relying on that."""
    source_id = _make_source(
        sqlite_session, status="partial", chunk_count=12,
        unprocessed_pages=[{"page": 4, "reason": "ocr_required"}],
    )
    result = ingest_router.get_source(source_id)
    assert result.status == SourceStatus.PARTIAL
    assert result.unprocessed_pages == [{"page": 4, "reason": "ocr_required"}]


# --- update_source (PATCH) --------------------------------------------------

def test_update_source_toggles_enabled(sqlite_session):
    source_id = _make_source(sqlite_session, enabled=True)
    result = ingest_router.update_source(source_id, SourceToggleRequest(enabled=False))
    assert result.enabled is False


def test_update_source_404_for_unknown_id(sqlite_session):
    with pytest.raises(HTTPException) as exc_info:
        ingest_router.update_source("nonexistent-id", SourceToggleRequest(enabled=False))
    assert exc_info.value.status_code == 404


def test_update_source_blocked_when_read_only(sqlite_session, monkeypatch):
    set_setting(monkeypatch, "uploads_read_only", True)
    source_id = _make_source(sqlite_session)
    with pytest.raises(HTTPException) as exc_info:
        ingest_router.update_source(source_id, SourceToggleRequest(enabled=False))
    assert exc_info.value.status_code == 403


# --- delete_source -----------------------------------------------------------

def test_delete_source_removes_row_and_its_document_chunks(sqlite_session, tmp_path):
    stored_file = tmp_path / "doc.pdf"
    stored_file.write_bytes(b"fake pdf bytes")
    source_id = _make_source(sqlite_session, stored_path=str(stored_file))

    session = sqlite_session()
    try:
        session.add(Document(
            id=uuid.uuid4(), tenant_id="company_abc", content="chunk 1",
            source_name="doc.pdf", source_type="upload", language="fr",
            source_file_id=uuid.UUID(source_id),
        ))
        session.add(Document(
            id=uuid.uuid4(), tenant_id="company_abc", content="chunk 2",
            source_name="doc.pdf", source_type="upload", language="fr",
            source_file_id=uuid.UUID(source_id),
        ))
        session.commit()
    finally:
        session.close()

    result = ingest_router.delete_source(source_id)
    assert result["deleted_chunks"] == 2

    session = sqlite_session()
    try:
        assert session.get(SourceFile, uuid.UUID(source_id)) is None
        remaining = session.query(Document).filter_by(source_file_id=uuid.UUID(source_id)).count()
        assert remaining == 0
    finally:
        session.close()

    assert not stored_file.exists()  # file on disk actually unlinked


def test_delete_source_404_for_unknown_id(sqlite_session):
    with pytest.raises(HTTPException) as exc_info:
        ingest_router.delete_source("nonexistent-id")
    assert exc_info.value.status_code == 404


def test_delete_source_blocked_when_read_only(sqlite_session, monkeypatch):
    set_setting(monkeypatch, "uploads_read_only", True)
    source_id = _make_source(sqlite_session)
    with pytest.raises(HTTPException) as exc_info:
        ingest_router.delete_source(source_id)
    assert exc_info.value.status_code == 403


def test_delete_never_touches_another_tenants_source(sqlite_session):
    other_id = _make_source(sqlite_session, tenant_id="other-tenant")
    with pytest.raises(HTTPException) as exc_info:
        ingest_router.delete_source(other_id)  # get_tenant_id() always returns company_abc
    assert exc_info.value.status_code == 404
