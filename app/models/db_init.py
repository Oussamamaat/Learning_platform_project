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
        # A fresh database never hits this -- create_all builds the full
        # current model shape correctly the first time.

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
