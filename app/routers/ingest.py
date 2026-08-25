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
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from app.config import get_settings, get_tenant_id
from app.models.db import get_engine
from app.models.schemas import SourceFileOut, SourceListResponse, SourceStatus, SourceToggleRequest
from app.services import ingest_queue
from app.services.ingest_jobs import process_source_file
from app.services.ingestion import SUPPORTED_EXTENSIONS, LEGACY_BINARY_FORMATS

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/ingest", tags=["ingest"])

_engine = None
_SessionLocal = None

# 1MB read granularity: large enough that a 25MB upload is ~25 awaits
# rather than thousands, small enough that no single chunk is a
# meaningful allocation.
_UPLOAD_CHUNK_BYTES = 1024 * 1024
# Uploads under this stay entirely in memory; larger ones spill to a temp
# file instead of being held as one contiguous bytes object.
_SPOOL_MAX_MEMORY_BYTES = 2 * 1024 * 1024

_COLUMNS = (
    "id, filename, status, error_message, enabled, domain, language, "
    "chunk_count, page_count, pages_done, parser, ocr_engine, unprocessed_pages, "
    "size_bytes, created_at"
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
        pages_done=row.pages_done,
        parser=row.parser,
        ocr_engine=row.ocr_engine,
        unprocessed_pages=_decode_unprocessed_pages(row.unprocessed_pages),
        size_bytes=row.size_bytes,
        created_at=row.created_at,
    )


async def _spool_upload(upload: UploadFile, max_bytes: int):
    """Read one upload in bounded chunks, hashing as we go.

    Returns (sha256_hex, size_bytes, spooled_file). `spooled_file` is None
    -- and the partial data discarded -- when the upload exceeds
    max_bytes, which is detected DURING the read rather than after the
    whole thing is in memory.

    The bytes land in a SpooledTemporaryFile, which keeps small uploads
    entirely in memory and only spills the large ones to disk. That is
    what bounds this endpoint's memory: the previous `await upload.read()`
    held the full file as a single `bytes` object, and a 20-file
    drag-and-drop (settings.max_upload_files_per_request) walked that
    allocation 20 times in one request.
    """
    hasher = hashlib.sha256()
    size = 0
    spooled = tempfile.SpooledTemporaryFile(max_size=_SPOOL_MAX_MEMORY_BYTES)
    while True:
        chunk = await upload.read(_UPLOAD_CHUNK_BYTES)
        if not chunk:
            break
        size += len(chunk)
        if size > max_bytes:
            spooled.close()
            return hasher.hexdigest(), size, None
        hasher.update(chunk)
        spooled.write(chunk)
    spooled.seek(0)
    return hasher.hexdigest(), size, spooled


def _finalize_upload(spooled, stored_path: Path) -> None:
    """Copy the spooled upload to its permanent path and release the
    spool. Runs in a worker thread (see the call site) because both halves
    block."""
    try:
        with open(stored_path, "wb") as out:
            shutil.copyfileobj(spooled, out, length=_UPLOAD_CHUNK_BYTES)
    finally:
        spooled.close()


def _insert_and_read_back(session, params: dict) -> None:
    session.execute(
        text(
            # chunk_count and created_at are explicit even though the ORM
            # model declares defaults for both -- those defaults are
            # applied by SQLAlchemy when IT constructs an INSERT
            # (session.add()/flush), not by the database, so a raw text()
            # INSERT that omits them either violates chunk_count's NOT
            # NULL constraint outright, or (created_at, nullable but
            # required by SourceFileOut's response schema) silently stores
            # NULL and only crashes later, serializing the response
            # (caught by tests/test_ingest_router.py against a real
            # schema).
            "INSERT INTO source_files "
            "(id, tenant_id, filename, stored_path, content_type, size_bytes, sha256, "
            " status, enabled, domain, chunk_count, created_at, updated_at) "
            "VALUES (:id, :tenant_id, :filename, :stored_path, :content_type, :size_bytes, "
            " :sha256, 'pending', true, :domain, 0, :now, :now)"
        ),
        params,
    )
    session.commit()


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

            # Streamed in chunks with the hash computed as we go, instead
            # of `body = await upload.read()`. Three reasons that mattered:
            #   1. read() materializes the WHOLE file in memory (up to
            #      settings.max_upload_bytes, 25MB) and then hashlib walked
            #      it a second time -- two full passes plus a large
            #      short-lived allocation per file, x20 files per request.
            #   2. An oversize file was only rejected AFTER being fully
            #      read into memory; now it is rejected the moment the
            #      running total crosses the limit, so an over-limit upload
            #      costs bounded memory instead of its full size.
            #   3. The bytes never need to be held at all -- the very next
            #      thing that happens to them is a write to disk.
            sha256, size_bytes, spooled = await _spool_upload(
                upload, settings.max_upload_bytes
            )
            if spooled is None:
                raise HTTPException(
                    status_code=413,
                    detail=f"{upload.filename}: exceeds max upload size "
                    f"({settings.max_upload_bytes} bytes).",
                )

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
                # Release the spool: this file is a known duplicate, so
                # nothing further will read it. Without this, every
                # duplicate in a re-uploaded batch leaks a
                # SpooledTemporaryFile (and, above the spill threshold, a
                # real temp file) for the life of the process.
                spooled.close()
                out = _row_to_out(existing)
                out.duplicate_of = out.id
                results.append(out)
                continue

            source_id = uuid.uuid4()
            stored_path = upload_root / f"{source_id.hex}{suffix}"
            # run_in_threadpool, not a bare rename/write: this is an
            # `async def` endpoint, so ANY blocking call in it runs on the
            # event loop and stalls every other in-flight request --
            # including the status polls the Sources panel fires while
            # this same upload is processing. A 25MB write is milliseconds
            # on a good day and much worse on a busy disk; either way it
            # does not belong on the loop.
            await run_in_threadpool(_finalize_upload, spooled, stored_path)

            now = datetime.now(timezone.utc)
            # Deliberately NOT run_in_threadpool, unlike the file write
            # above: a SQLAlchemy Session has thread affinity (SQLite
            # enforces it outright, which is what tests/test_ingest_router
            # .py's fixture uses), and these are single-row OLTP
            # statements measured in a millisecond or two -- nothing like
            # the up-to-25MB copy the write is. Offloading bulk I/O is
            # worth it; offloading a sub-millisecond INSERT to dodge a
            # session's threading contract is not.
            _insert_and_read_back(session, {
                "id": str(source_id),
                "tenant_id": tenant_id,
                "filename": upload.filename or stored_path.name,
                "stored_path": str(stored_path),
                "content_type": upload.content_type,
                "size_bytes": size_bytes,
                "sha256": sha256,
                "domain": domain,
                "now": now,
            })
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
