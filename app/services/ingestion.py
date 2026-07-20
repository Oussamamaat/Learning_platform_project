"""
Document Ingestion Service
─────────────────────────
Reads documents → chunks → embeds → stores in pgvector.

Usage:
    python -m app.services.ingestion --source_dir ./data/raw --tenant_id company_abc
"""

import os
import uuid
from pathlib import Path
from typing import Optional

import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
BATCH_SIZE = 64


def get_db_connection(database_url: str):
    """Connect to PostgreSQL."""
    return psycopg2.connect(database_url)


def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    """Load and cache the embedding model."""
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"Model loaded. Embedding dimension: {model.get_embedding_dimension()}")
    return model


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks using LangChain."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def embed_chunks(
    model: SentenceTransformer,
    chunks: list[str],
) -> list[list[float]]:
    """Generate embeddings for a list of text chunks."""
    embeddings = model.encode(chunks, show_progress_bar=False, batch_size=BATCH_SIZE)
    return embeddings.tolist()


def insert_documents(
    conn,
    tenant_id: str,
    source_name: str,
    source_type: str,
    language: str,
    chunks: list[str],
    embeddings: list[list[float]],
) -> int:
    """Insert chunked documents with embeddings into pgvector."""
    cursor = conn.cursor()

    rows = []
    for chunk, embedding in zip(chunks, embeddings):
        doc_id = str(uuid.uuid4())
        rows.append((
            doc_id,
            tenant_id,
            chunk,
            source_name,
            source_type,
            language,
            str(embedding),
        ))

    insert_query = """
        INSERT INTO documents (id, tenant_id, content, source_name, source_type, language, embedding)
        VALUES %s
    """

    execute_values(
        cursor,
        insert_query,
        rows,
        template="(%s::uuid, %s, %s, %s, %s, %s, %s::vector)",
        page_size=BATCH_SIZE,
    )

    conn.commit()
    return len(rows)


def ingest_text(
    text: str,
    source_name: str,
    tenant_id: str,
    source_type: str = "text",
    language: str = "fr",
    database_url: Optional[str] = None,
    embedding_model: Optional[SentenceTransformer] = None,
) -> dict:
    """Ingest a single text document: chunk → embed → store."""
    db_url = database_url or os.getenv(
        "DATABASE_URL",
        "postgresql://assistant:changeme@localhost:5432/iblog_assistant",
    )

    if embedding_model is None:
        embedding_model = load_embedding_model()

    chunks = chunk_text(text)
    if not chunks:
        return {"chunks_created": 0, "source_name": source_name, "status": "empty"}

    embeddings = embed_chunks(embedding_model, chunks)

    conn = get_db_connection(db_url)
    try:
        count = insert_documents(
            conn, tenant_id, source_name, source_type, language, chunks, embeddings
        )
        return {"chunks_created": count, "source_name": source_name, "status": "success"}
    finally:
        conn.close()


def ingest_file(
    file_path: str,
    tenant_id: str,
    source_type: str = "text",
    language: str = "fr",
    database_url: Optional[str] = None,
    embedding_model: Optional[SentenceTransformer] = None,
) -> dict:
    """Ingest a single file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    text = path.read_text(encoding="utf-8")
    return ingest_text(
        text=text,
        source_name=path.name,
        tenant_id=tenant_id,
        source_type=source_type,
        language=language,
        database_url=database_url,
        embedding_model=embedding_model,
    )


def ingest_directory(
    source_dir: str,
    tenant_id: str,
    language: str = "fr",
    database_url: Optional[str] = None,
) -> list[dict]:
    """Ingest all .txt files from a directory."""
    source_path = Path(source_dir)
    if not source_path.is_dir():
        raise NotADirectoryError(f"Directory not found: {source_dir}")

    embedding_model = load_embedding_model()
    results = []

    for file_path in sorted(source_path.glob("**/*.txt")):
        rel_path = file_path.relative_to(source_path)
        print(f"Ingesting: {rel_path}")

        result = ingest_file(
            file_path=str(file_path),
            tenant_id=tenant_id,
            source_type="text",
            language=language,
            database_url=database_url,
            embedding_model=embedding_model,
        )
        results.append(result)
        print(f"  -> {result['chunks_created']} chunks created")

    total_chunks = sum(r["chunks_created"] for r in results)
    print(f"\nDone. {len(results)} files ingested, {total_chunks} total chunks.")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest documents into pgvector")
    parser.add_argument("--source_dir", required=True, help="Directory with .txt files")
    parser.add_argument("--tenant_id", required=True, help="Tenant identifier")
    parser.add_argument("--language", default="fr", help="Content language (default: fr)")
    parser.add_argument("--database_url", default=None, help="PostgreSQL connection URL")
    args = parser.parse_args()

    results = ingest_directory(
        source_dir=args.source_dir,
        tenant_id=args.tenant_id,
        language=args.language,
        database_url=args.database_url,
    )
