"""
Tenant Source Management
─────────────────────────
The logic layer app/routers/ingest.py and app/routers/chat.py both call
into for tenant-uploaded documents (source_files). Kept separate from the
router so retrieval code (app.services.search, app.routers.chat) doesn't
need to import fastapi to reach active_source_ids/corpus_version.

Three responsibilities:

1. active_source_ids: which uploaded sources retrieval may use this turn.
   Server state (status IN ('ready','partial') AND enabled) is
   authoritative; a client-supplied list can only NARROW it, never widen
   it -- same posture as app.config.get_tenant_id: a client-supplied
   identifier is a hint, never an authorization. 'partial' counts as active
   because its chunks are real (app.services.ingest_jobs.process_source_file
   only reaches 'partial' after ingest_file has already stored them for
   every page that DID parse) -- excluding it would make a document's
   successfully-ingested pages permanently unretrievable over a handful of
   OCR-requiring pages, the exact failure this status exists to avoid.
2. corpus_version: the pin-invalidation signal app.routers.chat's
   _resolve_turn_context compares against ChatSession.pinned_corpus_version
   so a newly-uploaded/toggled/deleted source invalidates a stale pin
   instead of silently staying invisible until an unrelated topic shift.
3. reap_orphaned_processing: startup-time honesty for the single-worker
   in-process ingest queue (app.services.ingest_queue) -- a server restart
   mid-job must not leave a source_files row spinning "Processing..."
   forever. Scoped to the booting process's own tenant_id -- a restart
   must never touch another tenant's in-flight rows.
"""
import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.config import get_settings

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal = None


def _get_session():
    """Lazy engine/session creation, mirroring app.services.history's
    _get_engine exactly (including the connect_timeout) -- a module-level
    SQLAlchemy engine built from settings.database_url on first use, not
    at import time, so importing this module never requires a reachable
    Postgres.

    connect_timeout=2 matters for the fail-open contract itself, not just
    speed: without it, an unreachable Postgres can hang on the platform's
    default TCP timeout (multiple seconds observed on this machine's IPv6
    attempt before falling back to IPv4) -- every caller here
    (active_source_ids, corpus_version) is on chat's request path and must
    fail fast, not fail slow.
    """
    global _engine, _SessionLocal
    if _engine is None:
        _engine = create_engine(
            get_settings().database_url,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 2},
        )
        _SessionLocal = sessionmaker(bind=_engine)
    return _SessionLocal()


def active_source_ids(tenant_id: str, *, requested: Optional[list[str]] = None) -> list[str]:
    """Which of this tenant's uploaded sources retrieval may use this turn.

    Server-side state (status IN ('ready','partial') AND enabled) is the
    only source of truth. `requested` (an optional client-supplied list, e.g.
    ChatRequest.active_source_ids) can only INTERSECT with that set, never
    add to it -- a stale client can't resurrect a source the tenant just
    disabled, and a request naming a foreign/nonexistent id is silently
    dropped rather than erroring.

    Returns [] on any DB failure. This is correct BY CONSTRUCTION, not just
    a safe fallback: [] means "no tenant-upload filter" to
    search_similar_chunks (app/services/search.py), reproducing exactly
    the pre-upload-feature global-only retrieval behaviour.
    """
    try:
        session = _get_session()
        try:
            rows = session.execute(
                text(
                    "SELECT id FROM source_files "
                    "WHERE tenant_id = :tenant_id AND status IN ('ready', 'partial') "
                    "AND enabled = true"
                ),
                {"tenant_id": tenant_id},
            ).fetchall()
        finally:
            session.close()
    except Exception:
        logger.exception("active_source_ids failed for tenant=%s; treating as no uploads", tenant_id)
        return []

    ready_ids = {str(r[0]) for r in rows}
    if requested is None:
        return sorted(ready_ids)
    return sorted(ready_ids & {str(r) for r in requested})


def corpus_version(tenant_id: str) -> Optional[str]:
    """sha256 of this tenant's retrievable-upload set: the sorted
    (id, enabled) pairs of every status IN ('ready','partial') source_files
    row, plus the count. Changes exactly when what retrieval COULD return
    changes -- a file finishing ingestion, a toggle, a delete -- and is
    unaffected by a file still uploading/processing (only rows retrieval can
    actually see count).

    Returns None on ANY failure. None means "unknown", and callers
    (app.routers.chat._resolve_turn_context) MUST treat unknown as "do not
    invalidate" -- a sentinel string here would make every turn's version
    mismatch the stored one during a Postgres hiccup, firing a false
    segment reset (and possibly deterministic_refusal) on every single
    turn, exactly the failure mode _resolve_turn_context's existing guards
    are already biased against.

    Deliberately NOT cached (no lru_cache) -- it IS the invalidation
    signal; caching it would defeat its own purpose.
    """
    try:
        session = _get_session()
        try:
            rows = session.execute(
                text(
                    "SELECT id, enabled FROM source_files "
                    "WHERE tenant_id = :tenant_id AND status IN ('ready', 'partial') "
                    "ORDER BY id"
                ),
                {"tenant_id": tenant_id},
            ).fetchall()
        finally:
            session.close()
    except Exception:
        logger.exception("corpus_version failed for tenant=%s; returning None (no invalidation)", tenant_id)
        return None

    key = "|".join(f"{r[0]}:{r[1]}" for r in rows) + f"|n={len(rows)}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def reap_orphaned_processing(tenant_id: str) -> int:
    """Called once at app startup (app/main.py). The single-worker
    in-process ingest queue (app.services.ingest_queue) keeps NO state
    outside Postgres, so a server restart mid-job leaves a source_files
    row stuck in 'pending'/'processing' forever unless something explicitly
    reaps it -- this is that explicit reap, run before any request can poll
    a stale-but-eternally-"Processing" row.

    Scoped to `tenant_id` -- the only tenant this process will ever serve
    (app.config.get_tenant_id is a process-lifetime constant, ADR 0001).
    Before this scoping existed, restarting the backend under a DIFFERENT
    tenant_id (e.g. to switch from company_abc to company_efg) would mark
    every OTHER tenant's in-flight pending/processing rows as error too --
    a single-worker restart for one tenant should never touch another
    tenant's uploads, especially since nothing on this server can even see
    or fix them for that tenant in this process's lifetime.

    Returns the number of rows reaped (0 on a clean start or a DB error --
    fail-open, since a boot-time DB hiccup here must not crash the app).
    """
    try:
        session = _get_session()
        try:
            result = session.execute(
                text(
                    "UPDATE source_files SET status = 'error', "
                    "error_message = 'Server restarted while this file was processing. "
                    "Delete and re-upload.', updated_at = :now "
                    "WHERE status IN ('pending', 'processing') AND tenant_id = :tenant_id"
                ),
                # Bound Python-side timestamp, not SQL now() -- now() is
                # Postgres-specific and would silently no-op-fail (unknown
                # function) against the SQLite engine tests/test_sources.py
                # uses for portable coverage.
                {"now": datetime.now(timezone.utc), "tenant_id": tenant_id},
            )
            session.commit()
            count = result.rowcount
        finally:
            session.close()
    except Exception:
        logger.exception("reap_orphaned_processing failed for tenant=%s; leaving any stuck rows as-is", tenant_id)
        return 0

    if count:
        logger.warning(
            "reap_orphaned_processing: marked %d orphaned source_files row(s) as error for tenant=%s",
            count, tenant_id,
        )
    return count
