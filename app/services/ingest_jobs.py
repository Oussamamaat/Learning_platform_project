"""
Ingest Job: one uploaded file's Pending -> Processing -> Ready/Error path.

The single function app.services.ingest_queue.submit runs off the main
thread. Every exception is caught and turned into source_files.status=
'error' with error_message -- an ingest job that raises past this function
would otherwise leave a row stuck at 'processing' forever (until
app.services.sources.reap_orphaned_processing catches it at the NEXT
server restart), so this is the primary error boundary, not a backstop.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.config import get_settings
from app.services.ingestion import (
    chunk_document,
    detect_document_language,
    ingest_file,
    parse_document_to_markdown,
)
from app.services.ocr import OcrUnavailableError
from app.services.routing import vote_domain
from app.services.search import search_similar_chunks

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal = None


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


def _set_status(source_file_id: str, **fields) -> None:
    """Small helper: UPDATE source_files SET <fields> WHERE id = :id, with
    updated_at bumped automatically. Used at every stage transition so a
    poll (GET /api/v1/ingest/sources) always reflects the latest known
    state even if a later stage fails.
    """
    if not fields:
        return
    set_clause = ", ".join(f"{k} = :{k}" for k in fields)
    session = _get_session()
    try:
        session.execute(
            # Bound Python-side timestamp, not SQL now() -- now() is
            # Postgres-specific and would silently break against the
            # SQLite engine tests/test_sources.py uses for portable
            # coverage (the exact class of SQLite/Postgres divergence
            # that let the pinned_fingerprint VARCHAR(64) overflow go
            # undetected in this codebase before -- see
            # docs/architecture/data-and-retrieval.md's pgvector section).
            text(f"UPDATE source_files SET {set_clause}, updated_at = :now WHERE id = :id"),
            {**fields, "id": source_file_id, "now": datetime.now(timezone.utc)},
        )
        session.commit()
    finally:
        session.close()


def _resolve_upload_domain(text_content: str, tenant_id: str) -> Optional[str]:
    """Display/badge-only domain for an uploaded file -- retrieval never
    filters an uploaded source by domain (app.services.search's source_ids
    clause bypasses the domain filter whenever source_file_id IS NOT NULL,
    see that function's docstring), so a wrong guess here costs nothing
    functionally. Best-effort vote over the document's own text; falls
    back to settings.default_domain on any failure or a clearing vote.
    """
    try:
        chunks = chunk_document(text_content)
        if not chunks:
            return get_settings().default_domain
        # Reuse chunk 0 (or a short prefix of the whole doc if that's more
        # representative) as a stand-in query for the vote -- cheap, no new
        # infra, "close enough for a badge" is the bar here, not retrieval
        # accuracy.
        probe = chunks[0]["content"][:500]
        candidates = search_similar_chunks(query=probe, tenant_id=tenant_id, top_k=10, domain=None)
        voted = vote_domain(candidates)
        return voted or get_settings().default_domain
    except Exception:
        logger.exception("domain vote failed for an uploaded file; falling back to tenant default")
        return get_settings().default_domain


def process_source_file(
    source_file_id: str, *, stored_path: str, tenant_id: str, filename: str,
) -> None:
    """The whole Pending -> Processing -> Ready/Error pipeline for one
    upload. Runs off-thread via app.services.ingest_queue.submit.

    `filename` is the tenant's ORIGINAL upload name (source_files.filename),
    distinct from stored_path's server-generated <uuid>.<ext> -- passed
    through to ingest_file's source_name so citations show something a
    human can recognize, not a UUID (see ingest_file's docstring; caught
    live by probe_upload_e2e.py).
    """
    try:
        _set_status(source_file_id, status="processing")

        unprocessed_pages: list = []
        text_content = parse_document_to_markdown(Path(stored_path), unprocessed_pages=unprocessed_pages)
        language = detect_document_language(text_content)
        domain = _resolve_upload_domain(text_content, tenant_id)

        result = ingest_file(
            file_path=stored_path,
            tenant_id=tenant_id,
            source_type="upload",
            language=language,
            domain=domain,
            source_file_id=source_file_id,
            source_name=filename,
        )

        # 'partial' when some pages were skipped rather than failing the
        # whole document (app.services.ingestion._parse_pdf) -- the chunks
        # from every page that DID parse are already stored by ingest_file
        # above, and app.services.sources.active_source_ids treats
        # 'partial' as retrievable exactly like 'ready'.
        final_status = "partial" if unprocessed_pages else "ready"
        _set_status(
            source_file_id,
            status=final_status,
            chunk_count=result["chunks_created"],
            language=language,
            domain=domain,
            unprocessed_pages=json.dumps(unprocessed_pages) if unprocessed_pages else None,
        )
        logger.info(
            "source_file=%s ingested: %d chunks, domain=%s, language=%s, status=%s%s",
            source_file_id, result["chunks_created"], domain, language, final_status,
            f", {len(unprocessed_pages)} page(s) skipped" if unprocessed_pages else "",
        )

    except OcrUnavailableError as e:
        # Specific, actionable message -- the exact failure decision 2
        # ("degrade gracefully when unavailable locally") calls for: tell
        # the user what to do (convert the file, or enable OCR), don't
        # just log a stack trace and leave the row silently stuck.
        logger.warning("source_file=%s needs OCR, unavailable: %s", source_file_id, e)
        _set_status(source_file_id, status="error", error_message=str(e))
    except Exception as e:
        logger.exception("source_file=%s ingestion failed", source_file_id)
        _set_status(source_file_id, status="error", error_message=str(e)[:500])
