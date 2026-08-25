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

from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from app.config import get_settings
from app.models.db import get_engine
from app.services.ingestion import (
    _IMAGE_EXTENSIONS,
    chunk_document,
    detect_document_language,
    ingest_file,
    parse_document_to_markdown,
)
from app.services.ocr import OcrUnavailableError
from app.services.routing import vote_domain
from app.services.search import search_similar_chunks

logger = logging.getLogger(__name__)

# Vocabulary SourceFileOut.parser documents (app/models/schemas.py) but
# that, before this fix, nothing ever wrote -- source_files.parser and
# .ocr_engine stayed NULL for every upload despite existing specifically
# as the "was this text OCR'd, by which engine" audit trail a citation-
# grounded system needs to answer without re-parsing the file.
_NON_PDF_PARSER_NAMES = {
    ".md": "text", ".txt": "text",
    ".docx": "docx", ".pptx": "pptx", ".xlsx": "xlsx", ".csv": "csv",
}


def _infer_parser_metadata(stored_path: str, page_stats: Optional[dict] = None) -> dict:
    """Best-effort {"parser", "page_count", "ocr_engine"} for the audit
    trail. Non-fatal by design (returns {} on any failure) -- this is
    metadata about a file that ALREADY ingested successfully by the time
    it's called, so a failure here must never turn a real success into an
    error.

    For PDFs, `page_stats` is the {"page_count", "ocr_pages"} dict
    app.services.ingestion._parse_pdf filled in DURING the real parse.
    This function used to compute those numbers itself by re-opening the
    file and running extract_text() + classify_page over every page again,
    describing that as "effectively zero cost" -- true of classify_page,
    but it fed on extract_text(), which is the dominant per-page cost on a
    native PDF. An 80-page document therefore paid for two full text
    extractions per upload, the second one purely to fill in three
    columns. Two passes could also disagree, reporting an audit trail that
    didn't describe the parse that actually stored the chunks.

    The re-parse survives only as a fallback for a caller that has no
    stats to hand (and it is skipped entirely if reading the file fails),
    so the audit columns still get filled rather than silently going NULL.
    """
    try:
        path = Path(stored_path)
        suffix = path.suffix.lower()

        if suffix in _NON_PDF_PARSER_NAMES:
            return {"parser": _NON_PDF_PARSER_NAMES[suffix]}

        if suffix in _IMAGE_EXTENSIONS:
            return {"parser": "image_ocr", "page_count": 1, "ocr_engine": get_settings().ocr_engine}

        if suffix != ".pdf":
            return {}

        if page_stats and "page_count" in page_stats:
            page_count = page_stats["page_count"]
            ocr_pages = page_stats.get("ocr_pages", 0)
        else:
            page_count, ocr_pages = _classify_pdf_pages(path)

        if ocr_pages == 0:
            parser = "pdf_text"
        elif ocr_pages >= page_count:
            parser = "pdf_ocr"
        else:
            parser = "pdf_mixed"

        result = {"parser": parser, "page_count": page_count}
        if ocr_pages > 0:
            result["ocr_engine"] = get_settings().ocr_engine
        return result
    except Exception:
        logger.exception("could not infer parser metadata for %s (non-fatal)", stored_path)
        return {}


def _classify_pdf_pages(path: Path) -> tuple[int, int]:
    """(page_count, ocr_pages) by re-reading the PDF. Only reached when no
    page_stats were handed forward from the real parse -- see
    _infer_parser_metadata."""
    from pypdf import PdfReader

    from app.services.pdf_classify import PageStrategy, classify_page

    reader = PdfReader(str(path))
    page_count = len(reader.pages)
    ocr_pages = 0
    for page in reader.pages:
        text_ = (page.extract_text() or "").strip()
        if classify_page(page, text=text_).strategy != PageStrategy.NATIVE:
            ocr_pages += 1
    return page_count, ocr_pages

_engine = None
_SessionLocal = None


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
    def _on_page_processed(page_num: int, total_pages: int) -> None:
        # Live progress while status='processing' (see SourceFile.
        # pages_done's docstring for why this exists). Writes page_count
        # on every call too, not just the first -- simpler than a
        # write-once branch, and harmless: it is the same value every
        # time (parse_document_to_markdown's total_pages is fixed for the
        # whole run). No throttling: a local Postgres UPDATE costs a few
        # ms, negligible next to even a fast native page's ~200ms render
        # cost, let alone an OCR page's tens of seconds -- revisit only if
        # this ever points at a remote/networked Postgres.
        _set_status(source_file_id, page_count=total_pages, pages_done=page_num)

    try:
        _set_status(source_file_id, status="processing")

        unprocessed_pages: list = []
        # page_stats is filled in by the parse itself, so the audit-trail
        # columns below (parser/page_count/ocr_engine) no longer need a
        # second full pass over the PDF -- see _infer_parser_metadata.
        page_stats: dict = {}
        text_content = parse_document_to_markdown(
            Path(stored_path), unprocessed_pages=unprocessed_pages,
            on_page_processed=_on_page_processed, page_stats=page_stats,
        )
        language = detect_document_language(text_content)
        domain = _resolve_upload_domain(text_content, tenant_id)

        # text=text_content: this file was ALREADY parsed above to get
        # language/domain -- pass that result through instead of letting
        # ingest_file parse it again from scratch. For a PDF needing OCR,
        # skipping the second parse is what keeps every OCR'd page from
        # running through the resident worker (app.services.ocr) twice
        # per upload; see ingest_file's `text` parameter docstring.
        result = ingest_file(
            file_path=stored_path,
            tenant_id=tenant_id,
            source_type="upload",
            language=language,
            domain=domain,
            source_file_id=source_file_id,
            source_name=filename,
            text=text_content,
        )

        if result["chunks_created"] == 0:
            # ingest_text already knows this -- it returns
            # {"chunks_created": 0, "status": "empty"} -- but that result
            # was previously read only for chunks_created and this branch
            # never existed, so EVERY non-PDF format (and an all-blank PDF,
            # whose pages ARE individually recorded but sum to nothing
            # retrievable) landed here as status='ready'/'partial' with
            # chunk_count=0 and error_message=NULL. Two concrete
            # consequences that made this worse than a cosmetic gap:
            #   1. app/routers/ingest.py's sha256 dedupe short-circuit
            #      matches on status IN ('ready','partial'), so a 0-chunk
            #      row was a valid match -- re-uploading the identical file
            #      returned duplicate_of and never re-ingested. 'error' is
            #      not in that IN-list, so this trap closes here.
            #   2. The frontend renders 'ready' as an unqualified green
            #      success pill (SourceItem.tsx) with no indication the
            #      document is empty.
            # A heading-only pptx/csv ("## Slide 1", "# name") is
            # non-empty as a raw STRING, which is why this checks
            # chunks_created rather than `not text_content.strip()` --
            # that weaker check would miss exactly those two formats.
            logger.warning(
                "source_file=%s produced 0 chunks (nothing retrievable); "
                "marking error instead of %s",
                source_file_id, "partial" if unprocessed_pages else "ready",
            )
            _set_status(
                source_file_id,
                status="error",
                error_message=(
                    "No retrievable content was found in this document. It may be "
                    "empty, contain only images text extraction/OCR could not read, "
                    "or use a layout this parser does not support. Delete this "
                    "upload and try again with a re-exported or re-scanned version."
                ),
                chunk_count=0,
                language=language,
                domain=domain,
                unprocessed_pages=json.dumps(unprocessed_pages) if unprocessed_pages else None,
                pages_done=None,  # terminal state -- see SourceFile.pages_done's docstring
            )
            return

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
            # parser/page_count/ocr_engine: the audit trail SourceFileOut
            # (app/models/schemas.py) documents -- "was this text OCR'd,
            # by which engine" -- but that these columns went unwritten by
            # every upload until now, always NULL despite existing
            # specifically to answer that question without re-parsing.
            **_infer_parser_metadata(stored_path, page_stats),
            pages_done=None,  # terminal state -- see SourceFile.pages_done's docstring
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
        _set_status(source_file_id, status="error", error_message=str(e), pages_done=None)
    except Exception as e:
        logger.exception("source_file=%s ingestion failed", source_file_id)
        _set_status(source_file_id, status="error", error_message=str(e)[:500], pages_done=None)
