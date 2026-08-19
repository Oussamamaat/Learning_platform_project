"""
Tenant Document Upload API
────────────────────────────
Upload / list / toggle / delete for tenant-wide, permanent documents that
join a tenant's own retrievable corpus (app.services.search's
source_ids filter, app.services.sources.active_source_ids). Not
session-scoped -- see the "Multi-format ingestion + Sources panel" plan's
Context section for why (single tenant, no auth, uploads are tenant-wide).

Processing is async (app.services.ingest_queue's single-worker executor):
this router's job is validate -> store bytes -> insert a source_files row
-> submit the job -> return immediately. The actual parse/chunk/embed work
(app.services.ingest_jobs.process_source_file) runs off-thread; the client
polls GET /sources for status.
"""
import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.config import get_settings, get_tenant_id
from app.models.schemas import SourceFileOut, SourceListResponse, SourceStatus, SourceToggleRequest
from app.services import ingest_queue
from app.services.ingest_jobs import process_source_file
from app.services.ingestion import SUPPORTED_EXTENSIONS, LEGACY_BINARY_FORMATS

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/ingest", tags=["ingest"])

_engine = None
_SessionLocal = None

_COLUMNS = (
    "id, filename, status, error_message, enabled, domain, language, "
    "chunk_count, page_count, parser, ocr_engine, unprocessed_pages, size_bytes, created_at"
)


def _get_session():
    global _engine, _SessionLocal
    if _engine is None:
        _engine = create_engine(
            get_settings().database_url,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 2},
        )
        _SessionLocal = sessionmaker(bind=_engine)
    return _SessionLocal()


def _decode_unprocessed_pages(value) -> Optional[list]:
    """psycopg2 auto-deserializes a json/jsonb column read through a raw
    text() SELECT into a Python list; SQLite (tests/test_ingest_router.py's
    sqlite_session fixture) has no such driver-level cast and returns the
    stored string as-is -- the exact class of Postgres/SQLite divergence
    ingest_jobs.py's module docstring warns about. Handle both explicitly
    rather than relying on either driver's behavior implicitly."""
    if value is None or isinstance(value, list):
        return value
    return json.loads(value)


def _row_to_out(row) -> SourceFileOut:
    return SourceFileOut(
        id=str(row.id),
        filename=row.filename,
        status=SourceStatus(row.status),
        error_message=row.error_message,
        enabled=row.enabled,
        domain=row.domain,
        language=row.language,
        chunk_count=row.chunk_count,
        page_count=row.page_count,
        parser=row.parser,
        ocr_engine=row.ocr_engine,
        unprocessed_pages=_decode_unprocessed_pages(row.unprocessed_pages),
        size_bytes=row.size_bytes,
        created_at=row.created_at,
    )


def _require_writable() -> None:
    # No-auth kill-switch (app/config.py's uploads_read_only docstring): an
    # upload+delete API reachable by anyone who can reach the port can
    # otherwise wipe the tenant's corpus. GET is unaffected.
    if get_settings().uploads_read_only:
        raise HTTPException(status_code=403, detail="Uploads are read-only in this deployment.")


@router.post("/upload", response_model=list[SourceFileOut])
async def upload_sources(
    files: list[UploadFile] = File(...),
    domain: Optional[str] = Form(None),
):
    _require_writable()
    settings = get_settings()
    tenant_id = get_tenant_id()

    if len(files) > settings.max_upload_files_per_request:
        raise HTTPException(
            status_code=413,
            detail=f"Too many files in one request (max {settings.max_upload_files_per_request}).",
        )

    upload_root = Path(settings.upload_dir) / tenant_id
    upload_root.mkdir(parents=True, exist_ok=True)

    session = _get_session()
    results: list[SourceFileOut] = []
    try:
        for upload in files:
            suffix = Path(upload.filename or "").suffix.lower()

            if suffix in LEGACY_BINARY_FORMATS:
                raise HTTPException(
                    status_code=400,
                    detail=f"{upload.filename}: legacy binary {suffix} is not supported. "
                    f"Re-save it as {LEGACY_BINARY_FORMATS[suffix]} and upload again.",
                )
            if suffix not in SUPPORTED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"{upload.filename}: unsupported file type {suffix!r}.",
                )

            body = await upload.read()
            if len(body) > settings.max_upload_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"{upload.filename}: exceeds max upload size "
                    f"({settings.max_upload_bytes} bytes).",
                )

            sha256 = hashlib.sha256(body).hexdigest()

            # Idempotency: a byte-identical file already ready for this
            # tenant short-circuits re-ingestion entirely.
            existing = session.execute(
                text(
                    f"SELECT {_COLUMNS} "
                    "FROM source_files WHERE tenant_id = :t AND sha256 = :h "
                    "AND status IN ('ready', 'partial') LIMIT 1"
                ),
                {"t": tenant_id, "h": sha256},
            ).fetchone()
            if existing is not None:
                out = _row_to_out(existing)
                out.duplicate_of = out.id
                results.append(out)
                continue

            source_id = uuid.uuid4()
            stored_path = upload_root / f"{source_id.hex}{suffix}"
            stored_path.write_bytes(body)

            now = datetime.now(timezone.utc)
            session.execute(
                text(
                    # chunk_count and created_at are explicit even though
                    # the ORM model declares defaults for both -- those
                    # defaults are applied by SQLAlchemy when IT
                    # constructs an INSERT (session.add()/flush), not by
                    # the database, so a raw text() INSERT that omits them
                    # either violates chunk_count's NOT NULL constraint
                    # outright, or (created_at, nullable but required by
                    # SourceFileOut's response schema) silently stores
                    # NULL and only crashes later, serializing the
                    # response (caught by tests/test_ingest_router.py
                    # against a real schema).
                    "INSERT INTO source_files "
                    "(id, tenant_id, filename, stored_path, content_type, size_bytes, sha256, "
                    " status, enabled, domain, chunk_count, created_at, updated_at) "
                    "VALUES (:id, :tenant_id, :filename, :stored_path, :content_type, :size_bytes, "
                    " :sha256, 'pending', true, :domain, 0, :now, :now)"
                ),
                {
                    "id": str(source_id),
                    "tenant_id": tenant_id,
                    "filename": upload.filename or stored_path.name,
                    "stored_path": str(stored_path),
                    "content_type": upload.content_type,
                    "size_bytes": len(body),
                    "sha256": sha256,
                    "domain": domain,
                    "now": now,
                },
            )
            session.commit()

            row = session.execute(
                text(f"SELECT {_COLUMNS} FROM source_files WHERE id = :id"),
                {"id": str(source_id)},
            ).fetchone()
            results.append(_row_to_out(row))

            ingest_queue.submit(
                process_source_file,
                str(source_id),
                stored_path=str(stored_path),
                tenant_id=tenant_id,
                filename=upload.filename or stored_path.name,
            )
    finally:
        session.close()

    return results


@router.get("/sources", response_model=SourceListResponse)
def list_sources():
    tenant_id = get_tenant_id()
    session = _get_session()
    try:
        rows = session.execute(
            text(
                f"SELECT {_COLUMNS} "
                "FROM source_files WHERE tenant_id = :t ORDER BY created_at DESC"
            ),
            {"t": tenant_id},
        ).fetchall()
    finally:
        session.close()

    sources = [_row_to_out(r) for r in rows]
    return SourceListResponse(
        sources=sources,
        # PARTIAL counts as ready: its chunks are already stored and
        # retrievable (app.services.sources.active_source_ids treats
        # ready/partial identically) -- only the pages that needed OCR were
        # skipped, not the whole document.
        ready_count=sum(1 for s in sources if s.status in (SourceStatus.READY, SourceStatus.PARTIAL)),
        total_chunks=sum(s.chunk_count for s in sources),
    )


@router.get("/sources/{source_id}", response_model=SourceFileOut)
def get_source(source_id: str):
    tenant_id = get_tenant_id()
    session = _get_session()
    try:
        row = session.execute(
            text(f"SELECT {_COLUMNS} FROM source_files WHERE tenant_id = :t AND id = :id"),
            {"t": tenant_id, "id": source_id},
        ).fetchone()
    finally:
        session.close()

    if row is None:
        raise HTTPException(status_code=404, detail="Source not found.")
    return _row_to_out(row)


@router.patch("/sources/{source_id}", response_model=SourceFileOut)
def update_source(source_id: str, request: SourceToggleRequest):
    _require_writable()
    tenant_id = get_tenant_id()
    session = _get_session()
    try:
        result = session.execute(
            text(
                "UPDATE source_files SET enabled = :enabled, updated_at = :now "
                "WHERE tenant_id = :t AND id = :id"
            ),
            {"enabled": request.enabled, "t": tenant_id, "id": source_id, "now": datetime.now(timezone.utc)},
        )
        if result.rowcount == 0:
            session.rollback()
            raise HTTPException(status_code=404, detail="Source not found.")
        session.commit()

        row = session.execute(
            text(f"SELECT {_COLUMNS} FROM source_files WHERE id = :id"),
            {"id": source_id},
        ).fetchone()
    finally:
        session.close()

    return _row_to_out(row)


@router.delete("/sources/{source_id}")
def delete_source(source_id: str):
    _require_writable()
    tenant_id = get_tenant_id()
    session = _get_session()
    try:
        row = session.execute(
            text("SELECT stored_path FROM source_files WHERE tenant_id = :t AND id = :id"),
            {"t": tenant_id, "id": source_id},
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Source not found.")

        # Vector delete FIRST: a partial failure after this point must
        # never leave orphan chunks retrievable under a source_files row
        # that looks deleted.
        deleted = session.execute(
            text("DELETE FROM documents WHERE tenant_id = :t AND source_file_id = :id"),
            {"t": tenant_id, "id": source_id},
        )
        deleted_chunks = deleted.rowcount
        session.execute(
            text("DELETE FROM source_files WHERE tenant_id = :t AND id = :id"),
            {"t": tenant_id, "id": source_id},
        )
        session.commit()
    finally:
        session.close()

    if row.stored_path:
        Path(row.stored_path).unlink(missing_ok=True)

    return {"deleted_chunks": deleted_chunks}
