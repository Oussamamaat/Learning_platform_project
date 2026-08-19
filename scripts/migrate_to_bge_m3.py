"""
Migrate the pgvector `documents` table from the old embedding model
(paraphrase-multilingual-MiniLM-L12-v2, 384-dim) to the new one
(BAAI/bge-m3, 1024-dim -- see app/config.py's embedding_model/embedding_dim
docstring for why).

This is destructive: pgvector rejects an in-place dimension change on a
column with existing rows, so the only safe sequence is archive -> truncate
-> alter -> re-ingest. Everything in `documents` except the 84 legacy
`domain IS NULL` rows (company_abc/demo_tenant, 42 each -- an unrelated
road-traffic corpus, not reproducible from raw/) is reproducible from
raw/shared via ingest_directory, so only those 84 rows need preserving,
and they're archived rather than destroyed.

Also lands the source_file_id / source_files / pinned_corpus_version
schema pieces needed by the later upload feature (Phase 3/4 of the
"Multi-format ingestion + Sources panel" plan) in this same run, so there
is exactly ONE destructive migration + re-ingest instead of two.

Usage:
    .gguf_venv/Scripts/python.exe scripts/migrate_to_bge_m3.py --yes-destructive
    .gguf_venv/Scripts/python.exe scripts/migrate_to_bge_m3.py --yes-destructive --source-dir raw/shared --tenant-id company_abc

Rollback: set EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2 and
EMBEDDING_DIM=384 in .env (or the environment), then re-run this script.
`documents_legacy_384` is untouched by either direction and must be dropped
by hand if it's no longer wanted.
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import create_engine, text

from app.config import get_settings
from app.models.database import Base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--yes-destructive", action="store_true",
        help="Required acknowledgement -- this truncates the documents table.",
    )
    parser.add_argument("--source-dir", default="./raw/shared")
    parser.add_argument("--tenant-id", default=None, help="Defaults to settings.default_tenant_id.")
    parser.add_argument(
        "--skip-reingest", action="store_true",
        help="Only do the schema migration (archive/truncate/alter/create_all), skip ingest_directory. "
             "Useful for re-running ingest separately or for CI dry-runs against an empty DB.",
    )
    args = parser.parse_args()

    if not args.yes_destructive:
        print(
            "Refusing to run without --yes-destructive. This TRUNCATEs the "
            "documents table (after archiving the 84 legacy domain=NULL rows "
            "to documents_legacy_384) and re-ingests raw/shared from scratch.",
            file=sys.stderr,
        )
        sys.exit(1)

    settings = get_settings()
    tenant_id = args.tenant_id or settings.default_tenant_id
    db_url = settings.database_url
    safe_db_url = db_url.split("@")[-1] if "@" in db_url else db_url

    print(f"Target database: {safe_db_url}")
    print(f"New embedding model: {settings.embedding_model} (dim={settings.embedding_dim})")
    print(f"Tenant to re-ingest: {tenant_id}")
    print(f"Source dir: {args.source_dir}")

    engine = create_engine(db_url)

    with engine.connect() as conn:
        conn = conn.execution_options(isolation_level="AUTOCOMMIT")

        before = conn.execute(text("SELECT count(*) FROM documents")).scalar()
        print(f"\ndocuments row count before migration: {before}")

        print("1/5 Archiving legacy domain=NULL rows to documents_legacy_384 ...")
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS documents_legacy_384 AS "
            "SELECT * FROM documents WHERE domain IS NULL"
        ))
        archived = conn.execute(text("SELECT count(*) FROM documents_legacy_384")).scalar()
        print(f"    archived {archived} rows")

        print("2/5 Truncating documents (everything else is reproducible from raw/shared) ...")
        conn.execute(text("TRUNCATE TABLE documents"))

        print(f"3/5 Changing documents.embedding to vector({settings.embedding_dim}) ...")
        conn.execute(text("ALTER TABLE documents DROP COLUMN embedding"))
        conn.execute(text(f"ALTER TABLE documents ADD COLUMN embedding vector({settings.embedding_dim})"))

        print("4/5 Adding upload-plumbing columns (source_file_id, pinned_corpus_version) ...")
        conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_file_id UUID"))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_documents_source_file_id ON documents (source_file_id)"
        ))
        conn.execute(text(
            "ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS pinned_corpus_version VARCHAR(64)"
        ))

    print("5/5 Running create_all() to build any brand-new tables (source_files) ...")
    Base.metadata.create_all(bind=engine)

    if args.skip_reingest:
        print("\n--skip-reingest set -- schema migration done, documents table left empty.")
        return

    print(f"\nRe-ingesting {args.source_dir} for tenant={tenant_id} with the new embedding model ...")
    print("(this loads bge-m3 and re-embeds the whole corpus -- may take a few minutes)")
    from app.services.ingestion import ingest_directory

    results = ingest_directory(source_dir=args.source_dir, tenant_id=tenant_id, database_url=db_url)
    total_chunks = sum(r["chunks_created"] for r in results)

    with engine.connect() as conn:
        after_total = conn.execute(text("SELECT count(*) FROM documents")).scalar()
        after_with_embedding = conn.execute(
            text("SELECT count(*) FROM documents WHERE embedding IS NOT NULL")
        ).scalar()

    print(f"\nDone. {len(results)} files ingested, {total_chunks} chunks created.")
    print(f"documents row count after migration: {after_total}")
    print(f"rows with a non-NULL embedding:       {after_with_embedding}")
    if after_total != after_with_embedding:
        print(
            "WARNING: some rows have a NULL embedding -- ingestion partially failed. "
            "Do not proceed to re-benchmarking until this is investigated.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        "\nNext step: re-run scripts/eval_retrieval.py (pgvector, disk, --auto-domain) "
        "and record the new numbers in docs/architecture/data-and-retrieval.md before "
        "starting any work downstream of the embedding swap."
    )


if __name__ == "__main__":
    main()
