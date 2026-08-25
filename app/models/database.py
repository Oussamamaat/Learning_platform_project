import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector

from app.config import get_settings

Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(String(255), nullable=False, index=True)
    content = Column(Text, nullable=False)
    source_name = Column(String(500), nullable=False, index=True)
    source_type = Column(String(50), nullable=False, default="text")
    # First-class so retrieval can filter `WHERE domain = ...` directly --
    # previously only reachable via metadata.relative_path, which meant
    # app/services/search.py's build_rag_context had no domain filter at
    # all (a quiz labelled "industrial" could ground in blockchain chunks).
    domain = Column(String(50), nullable=True, index=True)
    # Groups the documents written by one ingestion run so a re-ingest's
    # corpus_version stamp (Stage 3 pinned-context invalidation) can tell
    # "the corpus changed" from "this row is old".
    ingest_batch_id = Column(String(64), nullable=True, index=True)
    language = Column(String(10), nullable=False, default="fr")
    # Dimension follows settings.embedding_dim (1024, bge-m3 as of
    # 2026-08-13 -- was 384/MiniLM). NULL means "belongs to the tenant's
    # always-on global corpus" (raw/shared, ingested via the CLI); a UUID
    # means "belongs to this uploaded source_files row" and is only
    # retrievable when that row is enabled (app.services.search's
    # source_ids filter). Deliberately no ForeignKey: the pre-existing
    # domain=NULL legacy rows and every raw/shared row are NULL here too,
    # and a hard FK would force a delete ordering for zero benefit -- the
    # ingest router already deletes documents before source_files.
    embedding = Column(Vector(get_settings().embedding_dim))
    source_file_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        # Every retrieval query filters `WHERE tenant_id = ... AND (domain
        # = ... OR source_file_id IS NOT NULL)` before ordering by vector
        # distance (app.services.search.search_similar_chunks). The
        # separate single-column indexes on tenant_id and domain make
        # Postgres either pick one and re-check the other per row, or
        # BitmapAnd two scans; a composite covers the actual predicate in
        # one. It matters more, not less, as the corpus grows -- with no
        # ANN index (see app/models/db_init.py for why that is deliberate
        # at this size) the vector ordering is a sequential scan over
        # whatever this predicate leaves, so shrinking that set is the
        # whole optimization.
        Index("ix_documents_tenant_domain", "tenant_id", "domain"),
        # The delete path (app/routers/ingest.py's DELETE /sources/{id})
        # and the source_ids retrieval filter both scope by tenant AND
        # source file.
        Index("ix_documents_tenant_source_file", "tenant_id", "source_file_id"),
    )

    def __repr__(self):
        return f"<Document(id={self.id}, source={self.source_name}, tenant={self.tenant_id})>"


class ChatSession(Base):
    """One conversation. `pinned_*` hold the context retrieved for the
    current segment (app/services/history.py) so a follow-up turn re-sends
    a byte-identical system block instead of re-retrieving -- see
    docs/architecture/ plan: this is the KV-prefix-reuse mechanism, not a
    response cache.
    """

    __tablename__ = "chat_sessions"

    session_id = Column(String(64), primary_key=True)
    tenant_id = Column(String(255), nullable=False, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    domain = Column(String(50), nullable=False)
    language = Column(String(10), nullable=False)
    segment_id = Column(Integer, nullable=False, default=1)
    pinned_context = Column(Text, nullable=True)
    pinned_sources = Column(JSON, nullable=True)
    pinned_fingerprint = Column(String(64), nullable=True)
    # Hash of the tenant's retrievable source set (app.services.sources.
    # corpus_version) at the moment this pin was written. Compared against
    # the CURRENT corpus_version in app.routers.chat._resolve_turn_context
    # so a newly-uploaded/toggled/deleted source invalidates a stale pin
    # instead of being silently invisible until an unrelated topic shift.
    # NOT the same job as pinned_fingerprint above -- that key includes the
    # query text and changes every turn, so it can never serve as a cache
    # key; this column is the dedicated one.
    pinned_corpus_version = Column(String(64), nullable=True)
    # Sticky in-message language instruction ("réponds en darija") --
    # app.services.routing.resolve_language's stored_override /
    # stored_override_query_lang. NULL means "no active override" (the
    # common case: response_lang just follows the message's own script).
    # See the "Automatic Domain Routing + Language/Script Detection" plan,
    # section 1d, for the stickiness rules that read/write these.
    response_lang_override = Column(String(10), nullable=True)
    override_query_lang = Column(String(10), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_active_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self):
        return f"<ChatSession(session_id={self.session_id}, tenant={self.tenant_id})>"


class ChatMessage(Base):
    """One turn of one session. `domain`/`language` are stamped per-message
    (not just read off the parent session) so history replay can filter
    strictly -- a domain switch or a language switch mid-session must not
    silently replay a turn from the wrong context into a new one (ADR 0002
    decision 5's contamination concern, applied to history instead of
    retrieval).
    """

    __tablename__ = "chat_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(64), ForeignKey("chat_sessions.session_id"), nullable=False)
    turn_index = Column(Integer, nullable=False)
    segment_id = Column(Integer, nullable=False)
    role = Column(String(10), nullable=False)
    content = Column(Text, nullable=False)
    domain = Column(String(50), nullable=False)
    language = Column(String(10), nullable=False)
    sources = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant')", name="ck_chat_messages_role"),
        UniqueConstraint("session_id", "turn_index", name="uq_chat_messages_session_turn"),
        Index("ix_chat_messages_session_turn_desc", "session_id", turn_index.desc()),
    )

    def __repr__(self):
        return f"<ChatMessage(session_id={self.session_id}, turn_index={self.turn_index}, role={self.role})>"


class SourceFile(Base):
    """One tenant-uploaded document (app/routers/ingest.py). Tenant-scoped
    and permanent -- unlike a session-scoped scratch attachment, this joins
    the tenant's own retrievable corpus once ingestion finishes, which is
    what lets Document.source_file_id point back at it.

    `status` is the single source of truth for the Uploading -> Processing
    -> Ready/Error lifecycle the UI polls (app.services.ingest_queue is a
    single in-process worker with no external state store, so a poll must
    read Postgres, never worker memory -- see app.services.sources.
    reap_orphaned_processing for what happens to a row stuck mid-flight
    across a server restart).

    `parser`/`ocr_engine` are an audit trail: a citation-grounded system
    must be able to answer "was this text OCR'd, by which engine" without
    re-parsing the file.
    """

    __tablename__ = "source_files"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(String(255), nullable=False, index=True)
    filename = Column(String(500), nullable=False)
    # Server-generated path (data/uploads/<tenant_id>/<uuid><ext>) -- never
    # the client-supplied filename, to avoid path traversal/collisions.
    # NULL once the file's bytes are purged (kept for status='error' rows
    # if useful for diagnosis; cleared on delete).
    stored_path = Column(String(1000), nullable=True)
    content_type = Column(String(100), nullable=True)
    size_bytes = Column(Integer, nullable=False, default=0)
    # sha256 of the uploaded bytes -- cheap re-upload idempotency (a
    # matching, status='ready' row short-circuits a duplicate upload
    # instead of re-ingesting it).
    sha256 = Column(String(64), nullable=False, index=True)
    status = Column(String(20), nullable=False, default="pending")
    error_message = Column(Text, nullable=True)
    # Toggled from the Sources panel without deleting -- disabled sources
    # are excluded from app.services.sources.active_source_ids, so
    # retrieval simply never sees them, without touching their chunks.
    enabled = Column(Boolean, nullable=False, default=True)
    # Display/badge only. Retrieval never filters an uploaded source by
    # domain (app.services.search's source_ids clause bypasses the domain
    # filter for any row with a non-NULL source_file_id) -- an explicit
    # upload-and-enable action is a stronger signal than the ~0.78-accuracy
    # automatic domain vote, so gating a hand-picked source on it would be
    # the wrong tradeoff.
    domain = Column(String(50), nullable=True)
    language = Column(String(10), nullable=True)
    chunk_count = Column(Integer, nullable=False, default=0)
    page_count = Column(Integer, nullable=True)
    # LIVE progress counter, updated per-page WHILE status='processing' --
    # distinct from page_count (only known/written once parsing finishes,
    # since computing it means reading the whole file). Added 2026-08-23:
    # before this, chunk_count stayed 0 and status stayed 'processing' for
    # a whole OCR-heavy run (tens of minutes on a laptop GPU) with no
    # other signal a poll could show, indistinguishable from a hang. NULL
    # once a document is ready/partial/error (see app.services.ingest_jobs
    # .process_source_file) -- it describes an in-flight run, not a
    # finished one; page_count is the number to read after completion.
    pages_done = Column(Integer, nullable=True)
    parser = Column(String(30), nullable=True)
    ocr_engine = Column(String(30), nullable=True)
    # Populated only for status='partial': pages a parser skipped rather
    # than failing the whole document over (app.services.ingestion._parse_pdf's
    # per-page OCR_REQUIRED handling). Plain JSON, not the Postgres-only
    # JSONB, for the same reason metadata_/pinned_sources/sources above are
    # JSON -- tests/test_sources.py's sqlite_session fixture needs the same
    # column type to work on both engines.
    # [{"page": 4, "reason": "ocr_required"}, ...]
    unprocessed_pages = Column(JSON, nullable=True)
    ingest_batch_id = Column(String(64), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending','processing','ready','partial','error')",
            name="ck_source_files_status",
        ),
        Index("ix_source_files_tenant_status", "tenant_id", "status"),
    )

    def __repr__(self):
        return f"<SourceFile(id={self.id}, filename={self.filename}, status={self.status})>"


class VideoJob(Base):
    """One explanatory-video request. This table (not shared Python code)
    is the connection point with the video-generation partner service: we
    write 'pending' rows and read the result; their worker (in-process,
    separate service, whatever) claims pending rows and writes the result
    back -- see app/routers/video.py.
    """

    __tablename__ = "video_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(String(255), nullable=False, index=True)
    session_id = Column(String(64), nullable=True)
    input_text = Column(Text, nullable=False)
    title = Column(String(300), nullable=True)
    language = Column(String(10), nullable=False, default="fr")
    status = Column(String(20), nullable=False, default="pending")
    video_url = Column(String(1000), nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending','processing','ready','error')",
            name="ck_video_jobs_status",
        ),
        Index("ix_video_jobs_tenant_status", "tenant_id", "status"),
    )

    def __repr__(self):
        return f"<VideoJob(id={self.id}, status={self.status})>"
