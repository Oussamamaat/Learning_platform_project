"""
Tests for app/services/ingest_jobs.py::process_source_file -- the whole
Pending -> Processing -> Ready/Error/Partial pipeline for one upload.

Previously ZERO direct test coverage: the closest existing tests (test_
ingest_router.py) construct source_files rows by hand and never call this
function, and test_ingestion.py mocks ingest_file only to test unrelated
domain-resolution helpers. That gap is exactly why the "0 chunks -> ready"
defect this suite pins went unnoticed.

Same in-memory SQLite pattern as tests/test_sources.py (monkeypatching
ingest_jobs._get_session by way of its module-level _engine/_SessionLocal)
-- portable coverage of real SQL behaviour, not a re-mock of the query
builder, and it exercises the same now()-vs-bound-timestamp portability
_set_status depends on.

parse_document_to_markdown, ingest_file, and _resolve_upload_domain are
monkeypatched: process_source_file's own job is orchestrating THEIR return
values into the right source_files.status, not re-parsing a real document
or re-embedding real text -- those are ingestion.py's own tests' job.
"""
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.database import Base, SourceFile
from app.services import ingest_jobs
from app.services.ocr import OcrUnavailableError

REAL_PDF_DIR = Path(__file__).parent / "data" / "real_pdfs"
pytestmark_real = pytest.mark.skipif(
    not REAL_PDF_DIR.exists(), reason="tests/data/real_pdfs/ fixtures not present"
)


@pytest.fixture
def sqlite_session(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine, tables=[SourceFile.__table__])
    SessionLocal = sessionmaker(bind=engine)
    monkeypatch.setattr(ingest_jobs, "_engine", engine)
    monkeypatch.setattr(ingest_jobs, "_SessionLocal", SessionLocal)
    return SessionLocal


def _make_source(session_factory, **overrides):
    """Returns the id in dashless hex form, matching what raw text() SQL
    against SQLite's UUID-as-string column round-trips as -- same
    reasoning as tests/test_sources.py's identical helper."""
    new_id = uuid.uuid4()
    defaults = dict(
        id=new_id,
        tenant_id="t1",
        filename="doc.pdf",
        stored_path="/tmp/doc.pdf",
        content_type="application/pdf",
        size_bytes=1234,
        sha256="abc123",
        status="pending",
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


def _row(session_factory, source_id: str):
    session = session_factory()
    try:
        return session.get(SourceFile, uuid.UUID(source_id))
    finally:
        session.close()


def _patch_pipeline(monkeypatch, *, text="some parsed content", extra_pages=None,
                     chunks_created=3, domain="industrial", parse_raises=None,
                     ingest_raises=None, simulated_page_count=None):
    """Stands in for the three heavy dependencies process_source_file
    orchestrates: parse_document_to_markdown (real OCR/parsing),
    ingest_file (real embedding + Postgres insert), and
    _resolve_upload_domain (a real similarity search). Each is patched at
    its ingest_jobs binding, matching how `from ... import X` resolves
    names -- patching the origin module would not affect ingest_jobs's
    already-bound reference.

    simulated_page_count: when given, fake_parse calls the
    on_page_processed callback process_source_file passes in, once per
    page from 1..N -- simulating what a real multi-page PDF does (see
    app.services.ingestion._parse_pdf), so tests can verify
    process_source_file builds and wires that callback correctly without
    a real PDF or OCR run."""

    def fake_parse(path, unprocessed_pages=None, on_page_processed=None, page_stats=None):
        if parse_raises is not None:
            raise parse_raises
        if unprocessed_pages is not None and extra_pages:
            unprocessed_pages.extend(extra_pages)
        if simulated_page_count and on_page_processed is not None:
            for page_num in range(1, simulated_page_count + 1):
                on_page_processed(page_num, simulated_page_count)
        # The real parser fills page_stats in as it classifies pages, so
        # _infer_parser_metadata no longer has to re-read the whole PDF to
        # learn them (app.services.ingestion._parse_pdf).
        if page_stats is not None and simulated_page_count:
            page_stats["page_count"] = simulated_page_count
            page_stats["ocr_pages"] = len(extra_pages or ())
        return text

    def fake_ingest_file(**kwargs):
        if ingest_raises is not None:
            raise ingest_raises
        return {"chunks_created": chunks_created, "source_name": kwargs.get("source_name"), "status": "success"}

    monkeypatch.setattr(ingest_jobs, "parse_document_to_markdown", fake_parse)
    monkeypatch.setattr(ingest_jobs, "ingest_file", fake_ingest_file)
    monkeypatch.setattr(ingest_jobs, "_resolve_upload_domain", lambda text_content, tenant_id: domain)


# --- the fix this file exists to pin ---------------------------------------

def test_zero_chunks_marks_error_not_ready(sqlite_session, monkeypatch):
    """The core defect: ingest_text already knows a document produced
    nothing retrievable (chunks_created=0), but that knowledge used to be
    discarded -- this asserts it is now surfaced as status='error' with an
    actionable message, not a silent 'ready' with chunk_count=0."""
    source_id = _make_source(sqlite_session)
    _patch_pipeline(monkeypatch, chunks_created=0)

    ingest_jobs.process_source_file(
        source_id, stored_path="/tmp/empty.docx", tenant_id="t1", filename="empty.docx",
    )

    row = _row(sqlite_session, source_id)
    assert row.status == "error"
    assert row.chunk_count == 0
    assert row.error_message  # non-empty, actionable
    assert "content" in row.error_message.lower() or "empty" in row.error_message.lower()


def test_zero_chunks_with_unprocessed_pages_still_marks_error(sqlite_session, monkeypatch):
    """The trickier case: an all-blank PDF has EVERY page individually
    recorded in unprocessed_pages (so the old `partial if unprocessed_pages
    else ready` logic would land on 'partial'), but if that sums to zero
    chunks there is nothing to retrieve either way -- this must still be
    'error', not a 'partial' that looks like a partial success."""
    source_id = _make_source(sqlite_session)
    _patch_pipeline(
        monkeypatch, chunks_created=0,
        extra_pages=[{"page": 1, "reason": "empty_page"}, {"page": 2, "reason": "empty_page"}],
    )

    ingest_jobs.process_source_file(
        source_id, stored_path="/tmp/blank.pdf", tenant_id="t1", filename="blank.pdf",
    )

    row = _row(sqlite_session, source_id)
    assert row.status == "error"
    assert row.chunk_count == 0


def test_zero_chunk_error_row_is_excluded_from_dedupe_match(sqlite_session, monkeypatch):
    """Closes the re-upload trap directly: app/routers/ingest.py's dedupe
    short-circuit matches `status IN ('ready', 'partial')`. Asserting the
    literal status value here is what guarantees a re-upload of the same
    file will actually retry instead of returning this dead row as
    duplicate_of forever."""
    source_id = _make_source(sqlite_session)
    _patch_pipeline(monkeypatch, chunks_created=0)

    ingest_jobs.process_source_file(
        source_id, stored_path="/tmp/empty.csv", tenant_id="t1", filename="empty.csv",
    )

    row = _row(sqlite_session, source_id)
    assert row.status not in ("ready", "partial")


# --- unchanged behaviour: normal success and partial-success paths --------

def test_nonzero_chunks_no_unprocessed_pages_marks_ready(sqlite_session, monkeypatch):
    source_id = _make_source(sqlite_session)
    _patch_pipeline(monkeypatch, chunks_created=5)

    ingest_jobs.process_source_file(
        source_id, stored_path="/tmp/doc.pdf", tenant_id="t1", filename="doc.pdf",
    )

    row = _row(sqlite_session, source_id)
    assert row.status == "ready"
    assert row.chunk_count == 5
    assert row.error_message is None


def test_nonzero_chunks_with_unprocessed_pages_marks_partial(sqlite_session, monkeypatch):
    """A document that mostly succeeded (has real chunks) but skipped some
    pages must still report 'partial', not 'error' -- the zero-chunks fix
    must not regress this, already-correct, case."""
    source_id = _make_source(sqlite_session)
    _patch_pipeline(
        monkeypatch, chunks_created=40,
        extra_pages=[{"page": 3, "reason": "ocr_required", "detail": "timeout"}],
    )

    ingest_jobs.process_source_file(
        source_id, stored_path="/tmp/doc.pdf", tenant_id="t1", filename="doc.pdf",
    )

    row = _row(sqlite_session, source_id)
    assert row.status == "partial"
    assert row.chunk_count == 40
    assert row.unprocessed_pages is not None


# --- exception boundaries ---------------------------------------------------

def test_ocr_unavailable_marks_error_with_its_own_message(sqlite_session, monkeypatch):
    source_id = _make_source(sqlite_session)
    _patch_pipeline(monkeypatch, parse_raises=OcrUnavailableError("OCR is not enabled"))

    ingest_jobs.process_source_file(
        source_id, stored_path="/tmp/scan.pdf", tenant_id="t1", filename="scan.pdf",
    )

    row = _row(sqlite_session, source_id)
    assert row.status == "error"
    assert "OCR is not enabled" in row.error_message


def test_unexpected_exception_marks_error_not_stuck_processing(sqlite_session, monkeypatch):
    """process_source_file is documented as the primary error boundary --
    an uncaught exception here must not leave a row stuck at 'processing'
    (that fallback is app.services.sources.reap_orphaned_processing at the
    NEXT server restart, a much worse user experience)."""
    source_id = _make_source(sqlite_session)
    _patch_pipeline(monkeypatch, ingest_raises=RuntimeError("Postgres connection refused"))

    ingest_jobs.process_source_file(
        source_id, stored_path="/tmp/doc.pdf", tenant_id="t1", filename="doc.pdf",
    )

    row = _row(sqlite_session, source_id)
    assert row.status == "error"
    assert "Postgres connection refused" in row.error_message


# --- _infer_parser_metadata: the audit trail SourceFileOut documents but
# that, before this fix, no upload ever populated ---------------------------

def test_infer_parser_metadata_non_pdf_formats_need_no_file_access():
    """docx/pptx/xlsx/csv/md/txt dispatch on suffix alone -- no filesystem
    access, so a stored_path that doesn't exist (as in every mocked test
    above, which never write real files) must not raise."""
    assert ingest_jobs._infer_parser_metadata("/nonexistent/doc.docx") == {"parser": "docx"}
    assert ingest_jobs._infer_parser_metadata("/nonexistent/deck.pptx") == {"parser": "pptx"}
    assert ingest_jobs._infer_parser_metadata("/nonexistent/sheet.xlsx") == {"parser": "xlsx"}
    assert ingest_jobs._infer_parser_metadata("/nonexistent/table.csv") == {"parser": "csv"}
    assert ingest_jobs._infer_parser_metadata("/nonexistent/note.md") == {"parser": "text"}


def test_infer_parser_metadata_image_is_always_ocr():
    result = ingest_jobs._infer_parser_metadata("/nonexistent/scan.png")
    assert result["parser"] == "image_ocr"
    assert result["page_count"] == 1
    assert result["ocr_engine"]


def test_infer_parser_metadata_missing_file_is_non_fatal():
    """The bar for this helper (called AFTER a document already ingested
    successfully) is 'never turn a real success into an error' -- a
    missing/unreadable .pdf must degrade to {} silently, not raise."""
    assert ingest_jobs._infer_parser_metadata("/nonexistent/scan.pdf") == {}


@pytestmark_real
def test_infer_parser_metadata_all_native_pdf_is_pdf_text():
    """avis_cese_sst_fr.pdf is pinned elsewhere (tests/test_pdf_classify.py)
    as exactly 40 pages, every one NATIVE -- the real-world case this
    helper must report as 'pdf_text' with no ocr_engine, not 'pdf_mixed'
    or 'pdf_ocr'."""
    result = ingest_jobs._infer_parser_metadata(str(REAL_PDF_DIR / "avis_cese_sst_fr.pdf"))
    assert result["parser"] == "pdf_text"
    assert result["page_count"] == 40
    assert "ocr_engine" not in result


@pytestmark_real
def test_infer_parser_metadata_mixed_pdf_reports_pdf_mixed():
    """guide_rh_sante_ar.pdf is pinned elsewhere as 27 pages, 25 NATIVE and
    2 needing OCR -- neither all-text nor all-scanned, so this must report
    'pdf_mixed' with an ocr_engine set (some pages WERE routed to OCR,
    even though this helper itself never calls it)."""
    result = ingest_jobs._infer_parser_metadata(str(REAL_PDF_DIR / "guide_rh_sante_ar.pdf"))
    assert result["parser"] == "pdf_mixed"
    assert result["page_count"] == 27
    assert result["ocr_engine"]


# --- pages_done / page_count: 2026-08-23 live-progress signal --------------
#
# Before this, chunk_count stayed 0 and status stayed 'processing' for the
# whole run (tens of minutes for an OCR-heavy PDF), with no other signal a
# poll could show -- indistinguishable from a hang. process_source_file
# builds an on_page_processed callback and passes it into
# parse_document_to_markdown; _patch_pipeline's fake_parse (simulated_page_
# count) invokes it exactly the way a real multi-page PDF would.

def test_pages_done_and_page_count_update_incrementally(sqlite_session, monkeypatch):
    """The whole point: a poll DURING processing must see real, growing
    progress, not just the final result. Reads the row from inside the
    fake parse callback itself, mid-run, rather than only after
    process_source_file returns."""
    source_id = _make_source(sqlite_session)
    seen = []

    def fake_parse(path, unprocessed_pages=None, on_page_processed=None, page_stats=None):
        for page_num in (1, 2, 3):
            on_page_processed(page_num, 3)
            row = _row(sqlite_session, source_id)
            seen.append((row.pages_done, row.page_count, row.status))
        if page_stats is not None:
            page_stats["page_count"] = 3
            page_stats["ocr_pages"] = 0
        return "content"

    monkeypatch.setattr(ingest_jobs, "parse_document_to_markdown", fake_parse)
    monkeypatch.setattr(ingest_jobs, "ingest_file", lambda **kw: {"chunks_created": 5, "status": "success"})
    monkeypatch.setattr(ingest_jobs, "_resolve_upload_domain", lambda text_content, tenant_id: "industrial")

    ingest_jobs.process_source_file(
        source_id, stored_path="/tmp/doc.pdf", tenant_id="t1", filename="doc.pdf",
    )

    assert seen == [(1, 3, "processing"), (2, 3, "processing"), (3, 3, "processing")]


@pytest.mark.parametrize(
    "kwargs,expect_status",
    [
        ({"chunks_created": 5, "simulated_page_count": 4}, "ready"),
        ({"chunks_created": 0, "simulated_page_count": 4}, "error"),
        ({"parse_raises": OcrUnavailableError("nope"), "simulated_page_count": None}, "error"),
        ({"ingest_raises": RuntimeError("boom"), "simulated_page_count": 4}, "error"),
    ],
)
def test_pages_done_is_cleared_on_every_terminal_state(sqlite_session, monkeypatch, kwargs, expect_status):
    """pages_done describes an IN-FLIGHT run (SourceFile.pages_done's
    docstring) -- it must not linger on a finished row, whatever the
    outcome, or a client reading it after completion would misread a
    finished document as still processing."""
    source_id = _make_source(sqlite_session)
    _patch_pipeline(monkeypatch, **kwargs)

    ingest_jobs.process_source_file(
        source_id, stored_path="/tmp/doc.pdf", tenant_id="t1", filename="doc.pdf",
    )

    row = _row(sqlite_session, source_id)
    assert row.status == expect_status
    assert row.pages_done is None
