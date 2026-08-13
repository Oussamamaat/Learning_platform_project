import uuid
from datetime import datetime, timezone
from sqlalchemy import (
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
    embedding = Column(Vector(384))
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

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
