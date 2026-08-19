"""
Tests for app/services/sources.py -- the tenant-upload logic layer
app/routers/chat.py and app/routers/ingest.py both call into.

Same pattern as tests/test_history.py: an in-memory SQLite engine
(monkeypatching sources._get_session), not live Postgres -- portable
coverage of real SQL behaviour, not a re-mock of the query builder.
Deliberately exercises the exact `now()`-vs-bound-timestamp portability
fix this module needed (see sources.py/ingest_jobs.py comments): a raw
Postgres `now()` in SQL text would silently break here.
"""
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.database import Base, SourceFile
from app.services import sources


@pytest.fixture
def sqlite_session(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine, tables=[SourceFile.__table__])
    SessionLocal = sessionmaker(bind=engine)
    monkeypatch.setattr(sources, "_engine", engine)
    monkeypatch.setattr(sources, "_SessionLocal", SessionLocal)
    return SessionLocal


def _make_source(session_factory, **overrides):
    """Returns the id in dashless hex form (uuid.UUID.hex), not str(uuid) --
    SQLite has no native UUID type, so the Postgres-specific UUID column
    (app.models.database.SourceFile.id) round-trips through raw text() SQL
    (what active_source_ids/corpus_version actually use) as a dashless hex
    string here, unlike real Postgres's canonical dashed format. Returning
    the same format the module's own queries will produce keeps every
    assertion and every `requested=[...]` list in these tests internally
    consistent, without needing per-call normalization. Production code
    never sees this divergence (it only ever talks to real Postgres) --
    the same class of SQLite/Postgres quirk already documented for
    pinned_fingerprint elsewhere in this codebase.
    """
    new_id = uuid.uuid4()
    defaults = dict(
        id=new_id,
        tenant_id="t1",
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


# --- active_source_ids -------------------------------------------------

def test_active_source_ids_returns_ready_and_enabled(sqlite_session):
    ready_id = _make_source(sqlite_session, status="ready", enabled=True)
    _make_source(sqlite_session, status="processing", enabled=True)  # not ready -- excluded
    _make_source(sqlite_session, status="ready", enabled=False)  # disabled -- excluded

    assert sources.active_source_ids("t1") == [ready_id]


def test_active_source_ids_treats_partial_as_active(sqlite_session):
    """A 'partial' source (some pages skipped for OCR, the rest already
    chunked -- app.services.ingestion._parse_pdf) must be just as
    retrievable as 'ready'. Excluding it would make a document's
    successfully-ingested pages permanently unreachable over a handful of
    OCR-requiring ones -- the exact failure the 'partial' status exists to
    avoid."""
    partial_id = _make_source(sqlite_session, status="partial", enabled=True)
    assert sources.active_source_ids("t1") == [partial_id]


def test_active_source_ids_scoped_to_tenant(sqlite_session):
    _make_source(sqlite_session, tenant_id="other-tenant", status="ready", enabled=True)
    assert sources.active_source_ids("t1") == []


def test_active_source_ids_client_hint_only_narrows(sqlite_session):
    ready_a = _make_source(sqlite_session, status="ready", enabled=True)
    ready_b = _make_source(sqlite_session, status="ready", enabled=True)
    disabled = _make_source(sqlite_session, status="ready", enabled=False)

    # A client list containing a disabled id and a foreign/nonexistent one
    # must never resurrect or invent access -- only intersect with the
    # server-side ready+enabled set.
    result = sources.active_source_ids(
        "t1", requested=[ready_a, disabled, "nonexistent-id"]
    )
    assert result == [ready_a]
    assert ready_b not in result  # narrowed away, not in the client's list


def test_active_source_ids_requested_none_returns_full_server_set(sqlite_session):
    ready_a = _make_source(sqlite_session, status="ready", enabled=True)
    ready_b = _make_source(sqlite_session, status="ready", enabled=True)
    assert sorted(sources.active_source_ids("t1")) == sorted([ready_a, ready_b])


def test_active_source_ids_returns_empty_list_on_db_failure(monkeypatch):
    def _raise():
        raise RuntimeError("connection refused")
    monkeypatch.setattr(sources, "_get_session", _raise)
    assert sources.active_source_ids("t1") == []


# --- corpus_version ------------------------------------------------------

def test_corpus_version_changes_when_a_source_becomes_ready(sqlite_session):
    v_before = sources.corpus_version("t1")
    _make_source(sqlite_session, status="ready", enabled=True)
    v_after = sources.corpus_version("t1")
    assert v_before != v_after


def test_corpus_version_stable_across_unrelated_write(sqlite_session):
    _make_source(sqlite_session, status="ready", enabled=True)
    v1 = sources.corpus_version("t1")
    # An unrelated tenant's write must not perturb this tenant's version.
    _make_source(sqlite_session, tenant_id="other-tenant", status="ready", enabled=True)
    v2 = sources.corpus_version("t1")
    assert v1 == v2


def test_corpus_version_changes_on_toggle(sqlite_session):
    source_id = _make_source(sqlite_session, status="ready", enabled=True)
    v_enabled = sources.corpus_version("t1")

    session = sqlite_session()
    try:
        row = session.get(SourceFile, uuid.UUID(source_id))
        row.enabled = False
        session.commit()
    finally:
        session.close()

    v_disabled = sources.corpus_version("t1")
    assert v_enabled != v_disabled


def test_corpus_version_changes_on_delete(sqlite_session):
    source_id = _make_source(sqlite_session, status="ready", enabled=True)
    v_present = sources.corpus_version("t1")

    session = sqlite_session()
    try:
        row = session.get(SourceFile, uuid.UUID(source_id))
        session.delete(row)
        session.commit()
    finally:
        session.close()

    v_deleted = sources.corpus_version("t1")
    assert v_present != v_deleted


def test_corpus_version_unaffected_by_non_ready_status(sqlite_session):
    """A file still uploading/processing must not perturb the version --
    only rows retrieval could actually return ('ready' or 'partial') are
    part of it."""
    v_before = sources.corpus_version("t1")
    _make_source(sqlite_session, status="pending", enabled=True)
    _make_source(sqlite_session, status="processing", enabled=True)
    v_after = sources.corpus_version("t1")
    assert v_before == v_after


def test_corpus_version_changes_when_a_source_becomes_partial(sqlite_session):
    """A 'partial' source is retrievable, so its arrival must invalidate a
    pin exactly like 'ready' does -- otherwise a chat session pinned before
    the upload finished would never see its (already-chunked) content."""
    v_before = sources.corpus_version("t1")
    _make_source(sqlite_session, status="partial", enabled=True)
    v_after = sources.corpus_version("t1")
    assert v_before != v_after


def test_corpus_version_returns_none_on_db_failure_the_anti_false_reset_guarantee(monkeypatch):
    """The single most important contract: a DB failure must return None
    (unknown), never a computed-but-wrong value or an exception -- callers
    (app.routers.chat._resolve_turn_context) treat None as 'never
    invalidate', which is what prevents a Postgres hiccup from firing a
    false segment reset on every turn."""
    def _raise():
        raise RuntimeError("connection refused")
    monkeypatch.setattr(sources, "_get_session", _raise)
    assert sources.corpus_version("t1") is None


# --- reap_orphaned_processing ---------------------------------------------

def test_reap_orphaned_processing_marks_pending_and_processing_as_error(sqlite_session):
    pending_id = _make_source(sqlite_session, status="pending")
    processing_id = _make_source(sqlite_session, status="processing")
    ready_id = _make_source(sqlite_session, status="ready")
    error_id = _make_source(sqlite_session, status="error")

    count = sources.reap_orphaned_processing("t1")
    assert count == 2

    session = sqlite_session()
    try:
        assert session.get(SourceFile, uuid.UUID(pending_id)).status == "error"
        assert session.get(SourceFile, uuid.UUID(processing_id)).status == "error"
        assert session.get(SourceFile, uuid.UUID(ready_id)).status == "ready"  # untouched
        assert session.get(SourceFile, uuid.UUID(error_id)).status == "error"  # already error
    finally:
        session.close()


def test_reap_orphaned_processing_sets_actionable_error_message(sqlite_session):
    pending_id = _make_source(sqlite_session, status="pending")
    sources.reap_orphaned_processing("t1")

    session = sqlite_session()
    try:
        row = session.get(SourceFile, uuid.UUID(pending_id))
        assert "restarted" in row.error_message.lower()
    finally:
        session.close()


def test_reap_orphaned_processing_returns_zero_on_clean_start(sqlite_session):
    _make_source(sqlite_session, status="ready")
    assert sources.reap_orphaned_processing("t1") == 0


def test_reap_orphaned_processing_fails_open_on_db_error(monkeypatch):
    def _raise():
        raise RuntimeError("connection refused")
    monkeypatch.setattr(sources, "_get_session", _raise)
    assert sources.reap_orphaned_processing("t1") == 0


def test_reap_orphaned_processing_does_not_touch_other_tenants(sqlite_session):
    """The bug this scoping fixes: restarting the backend to serve one
    tenant (e.g. switching DEFAULT_TENANT_ID from company_abc to
    company_efg) must not error out an unrelated tenant's in-flight
    uploads -- this process cannot see or fix them for the rest of its
    lifetime (app.config.get_tenant_id is a process-lifetime constant)."""
    own_pending_id = _make_source(sqlite_session, tenant_id="t1", status="pending")
    other_pending_id = _make_source(sqlite_session, tenant_id="t2", status="pending")
    other_processing_id = _make_source(sqlite_session, tenant_id="t2", status="processing")

    count = sources.reap_orphaned_processing("t1")
    assert count == 1

    session = sqlite_session()
    try:
        assert session.get(SourceFile, uuid.UUID(own_pending_id)).status == "error"
        assert session.get(SourceFile, uuid.UUID(other_pending_id)).status == "pending"
        assert session.get(SourceFile, uuid.UUID(other_processing_id)).status == "processing"
    finally:
        session.close()
