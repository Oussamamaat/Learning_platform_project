"""
Document Ingestion Service
─────────────────────────
Reads documents → strips formatting → chunks → embeds → stores in pgvector.

Supports .txt and .md files. Markdown syntax is stripped before chunking
so the embedding model processes clean content, not formatting noise.

Usage:
    python -m app.services.ingestion --source_dir ./raw/shared --tenant_id company_abc
"""

import os
import re
import uuid
import json
from pathlib import Path
from typing import Optional

import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings


DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
BATCH_SIZE = 64


def strip_markdown(text: str) -> str:
    """Remove markdown formatting artifacts so embedding model processes clean content."""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove markdown headers (##, ###, etc.)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    # Remove inline code backticks
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Remove blockquote markers
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
    # Remove horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Remove markdown table borders and alignment
    text = re.sub(r"^\|[-:| ]+\|\s*$", "", text, flags=re.MULTILINE)
    # Collapse remaining table rows into plain text
    text = re.sub(r"\|\s*", " ", text)
    # Remove link syntax, keep display text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


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
    metadata_list: Optional[list[dict]] = None,
) -> int:
    """Insert chunked documents with embeddings into pgvector."""
    cursor = conn.conn.cursor() if hasattr(conn, "conn") else conn.cursor()

    if not metadata_list:
        metadata_list = [{} for _ in chunks]

    rows = []
    for chunk, embedding, meta in zip(chunks, embeddings, metadata_list):
        doc_id = str(uuid.uuid4())
        rows.append((
            doc_id,
            tenant_id,
            chunk,
            source_name,
            source_type,
            language,
            str(embedding),
            json.dumps(meta),
        ))

    insert_query = """
        INSERT INTO documents (id, tenant_id, content, source_name, source_type, language, embedding, metadata)
        VALUES %s
    """

    execute_values(
        cursor,
        insert_query,
        rows,
        template="(%s::uuid, %s, %s, %s, %s, %s, %s::vector, %s::jsonb)",
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
    metadata: Optional[dict] = None,
    database_url: Optional[str] = None,
    embedding_model: Optional[SentenceTransformer] = None,
) -> dict:
    """Ingest a single text document: strip markdown → chunk → embed → store."""
    db_url = database_url or get_settings().database_url

    if embedding_model is None:
        embedding_model = load_embedding_model()

    clean_text = strip_markdown(text)
    chunks = chunk_text(clean_text)
    if not chunks:
        return {"chunks_created": 0, "source_name": source_name, "status": "empty"}

    embeddings = embed_chunks(embedding_model, chunks)

    # Replicate/extend metadata for all chunks
    base_meta = metadata.copy() if metadata else {}
    base_meta.update({
        "source_name": source_name,
        "source_type": source_type,
        "language": language,
    })
    metadata_list = [base_meta for _ in chunks]

    conn = get_db_connection(db_url)
    try:
        count = insert_documents(
            conn, tenant_id, source_name, source_type, language, chunks, embeddings, metadata_list
        )
        return {"chunks_created": count, "source_name": source_name, "status": "success"}
    finally:
        conn.close()


def ingest_file(
    file_path: str,
    tenant_id: str,
    source_type: str = "text",
    language: str = "fr",
    metadata: Optional[dict] = None,
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
        metadata=metadata,
        database_url=database_url,
        embedding_model=embedding_model,
    )


def ingest_directory(
    source_dir: str,
    tenant_id: str,
    language: str = "fr",
    database_url: Optional[str] = None,
) -> list[dict]:
    """Ingest all .txt and .md files from a directory."""
    source_path = Path(source_dir)
    if not source_path.is_dir():
        raise NotADirectoryError(f"Directory not found: {source_dir}")

    embedding_model = load_embedding_model()
    results = []

    for file_path in sorted(source_path.glob("**/*.md")) + sorted(source_path.glob("**/*.txt")):
        rel_path = file_path.relative_to(source_path)
        file_type = "markdown" if file_path.suffix == ".md" else "text"
        print(f"Ingesting [{file_type}]: {rel_path}")

        result = ingest_file(
            file_path=str(file_path),
            tenant_id=tenant_id,
            source_type=file_type,
            language=language,
            metadata={"source_dir": source_dir, "relative_path": str(rel_path)},
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
    parser.add_argument("--source_dir", required=True, help="Directory with .txt or .md files")
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
