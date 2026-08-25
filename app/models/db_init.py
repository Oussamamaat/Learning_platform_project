"""
Database Schema and Vector Extension Initialization
───────────────────────────────────────────────────
Enables pgvector and initializes tables in PostgreSQL.

Usage:
    python -m app.models.db_init
"""
import sys
from sqlalchemy import create_engine, text
from app.config import get_settings
from app.models.database import Base


def init_db():
    settings = get_settings()
    db_url = settings.database_url
    # Print database host for confirmation, hiding credentials
    safe_db_url = db_url.split("@")[-1] if "@" in db_url else db_url
    print(f"Connecting to database to initialize: {safe_db_url}")

    try:
        # Create engine
        engine = create_engine(db_url)

        # 1. Enable pgvector extension
        with engine.connect() as conn:
            conn = conn.execution_options(isolation_level="AUTOCOMMIT")
            print("Enabling pgvector extension...")
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

        # 2. Create tables
        print("Creating tables...")
        Base.metadata.create_all(bind=engine)
        print("Database tables initialized successfully.")

        # create_all only creates tables that don't exist yet -- it never
        # ALTERs an existing one. Confirmed live 2026-08-10: a pre-existing
        # `documents` table (84 rows, predating the `domain`/
        # `ingest_batch_id` columns added for retrieval.py's domain filter)
        # was silently left without them, and search_similar_chunks's SELECT
        # then failed with UndefinedColumn. There is no Alembic in this
        # project (a deliberate choice, see CLAUDE.md), so on an existing
        # database run this once by hand after adding a column to a model
        # that already has live rows:
        #
        #   ALTER TABLE documents ADD COLUMN IF NOT EXISTS domain VARCHAR(50);
        #   ALTER TABLE documents ADD COLUMN IF NOT EXISTS ingest_batch_id VARCHAR(64);
        #   CREATE INDEX IF NOT EXISTS ix_documents_domain ON documents (domain);
        #   CREATE INDEX IF NOT EXISTS ix_documents_ingest_batch_id ON documents (ingest_batch_id);
        #
        # Same gap, same fix, for chat_sessions.response_lang_override /
        # override_query_lang (added 2026-08-11 for sticky in-message
        # language overrides -- app.services.routing.resolve_language):
        #
        #   ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS response_lang_override VARCHAR(10);
        #   ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS override_query_lang VARCHAR(10);
        #
        # Same again for the 2026-08-13 tenant-upload feature
        # (source_file_id / pinned_corpus_version) and the bge-m3 embedding
        # migration (documents.embedding's width, 384 -> 1024) -- but
        # unlike the additive cases above, a pgvector column's dimension
        # cannot be ALTERed in place with rows present, so that migration
        # is NOT a hand-run ALTER TABLE; see scripts/migrate_to_bge_m3.py
        # (archives the legacy domain=NULL rows, truncates, alters the
        # column, then re-ingests). The additive columns from that same
        # change (documents.source_file_id, chat_sessions.
        # pinned_corpus_version, the new source_files table) ride along in
        # that same script rather than needing their own hand-run ALTERs.
        #
        # Same again for video_jobs.title (added 2026-08-18 when the partner
        # video-generation payload was locked -- see
        # docs/PARTNER_VIDEO_ONBOARDING.md "Locked payload"):
        #
        #   ALTER TABLE video_jobs ADD COLUMN IF NOT EXISTS title VARCHAR(300);
        #
        # Same again for source_files.unprocessed_pages (added 2026-08-18 when
        # a document with a mix of native and OCR_REQUIRED pages stopped
        # failing the whole upload -- app.services.ingestion._parse_pdf now
        # skips just the unreadable pages and app.services.ingest_jobs.
        # process_source_file records which ones). The CHECK constraint also
        # gained the 'partial' status value, which Postgres's ALTER TABLE
        # ... DROP CONSTRAINT / ADD CONSTRAINT does not do implicitly:
        #
        #   ALTER TABLE source_files ADD COLUMN IF NOT EXISTS unprocessed_pages JSON;
        #   ALTER TABLE source_files DROP CONSTRAINT IF EXISTS ck_source_files_status;
        #   ALTER TABLE source_files ADD CONSTRAINT ck_source_files_status
        #       CHECK (status IN ('pending','processing','ready','partial','error'));
        #
        # Same again for source_files.pages_done (added 2026-08-23 so a poll
        # during a still-processing upload has a live progress signal --
        # app.services.ingest_jobs.process_source_file writes page_count as
        # soon as it knows the total, then bumps pages_done after every page
        # app.services.ingestion._parse_pdf's on_page_processed callback
        # fires for; a document could otherwise sit at chunk_count=0,
        # status='processing' for tens of minutes with nothing else to show):
        #
        #   ALTER TABLE source_files ADD COLUMN IF NOT EXISTS pages_done INTEGER;
        #
        # Same again for the two composite indexes added to Document
        # (documents.__table_args__) so the retrieval predicate
        # `tenant_id = ... AND domain = ...` is covered by ONE index
        # instead of a BitmapAnd of two single-column ones. create_all
        # does create indexes for a table it is creating, but it will not
        # add one to a table that already exists:
        #
        #   CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_documents_tenant_domain
        #     ON documents (tenant_id, domain);
        #   CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_documents_tenant_source_file
        #     ON documents (tenant_id, source_file_id);
        #
        # CONCURRENTLY so an existing corpus stays readable while they
        # build; it cannot run inside a transaction block, so run these by
        # hand rather than adding them to this function.
        #
        # A fresh database never hits any of this -- create_all builds the
        # full current model shape correctly the first time.

        # Deliberately NOT creating an ANN index (HNSW/ivfflat) on
        # documents.embedding. At the current corpus size (~200 chunks) a
        # sequential scan is sub-millisecond, and HNSW post-filters under
        # `WHERE tenant_id = ...` -- a small tenant partition can silently
        # return fewer than top_k rows, which in this codebase fires
        # deterministic_refusal on a question the corpus does answer. Add
        # one only once a tenant's chunk count exceeds ~10k, and re-run
        # scripts/eval_retrieval.py after to confirm recall didn't regress:
        #
        #   CREATE INDEX CONCURRENTLY ix_documents_embedding_hnsw
        #     ON documents USING hnsw (embedding vector_cosine_ops);
        #
        # Prefer a per-tenant partial index or pgvector 0.8+ iterative
        # scans over a single global index, so a small tenant isn't
        # starved by a large one's post-filter behavior.

    except Exception as e:
        print(f"Error initializing database: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    init_db()
