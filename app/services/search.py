"""
Semantic Search Service
──────────────────────
Converts user queries to vectors and retrieves relevant document chunks from pgvector.
"""

import os
from typing import Optional

import psycopg2
from sentence_transformers import SentenceTransformer

from app.services.ingestion import (
    load_embedding_model,
    get_db_connection,
)


def search_similar_chunks(
    query: str,
    tenant_id: str,
    top_k: int = 5,
    database_url: Optional[str] = None,
    embedding_model: Optional[SentenceTransformer] = None,
) -> list[dict]:
    """
    Search for the most relevant document chunks for a given query.

    Args:
        query: User's question text
        tenant_id: Company/tenant identifier (isolation)
        top_k: Number of results to return
        database_url: PostgreSQL connection URL
        embedding_model: Pre-loaded SentenceTransformer model

    Returns:
        List of dicts with keys: content, source_name, similarity, metadata
    """
    db_url = database_url or os.getenv(
        "DATABASE_URL",
        "postgresql://assistant:changeme@localhost:5432/iblog_assistant",
    )

    if embedding_model is None:
        embedding_model = load_embedding_model()

    query_embedding = embedding_model.encode([query])[0].tolist()

    conn = get_db_connection(db_url)
    try:
        cursor = conn.cursor()

        search_query = """
            SELECT
                id,
                content,
                source_name,
                source_type,
                language,
                metadata,
                1 - (embedding <=> %s::vector) AS similarity
            FROM documents
            WHERE tenant_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """

        embedding_str = str(query_embedding)
        cursor.execute(search_query, (embedding_str, tenant_id, embedding_str, top_k))

        results = []
        for row in cursor.fetchall():
            results.append({
                "id": str(row[0]),
                "content": row[1],
                "source_name": row[2],
                "source_type": row[3],
                "language": row[4],
                "metadata": row[5],
                "similarity": float(row[6]),
            })

        return results
    finally:
        conn.close()


def build_rag_context(
    query: str,
    tenant_id: str,
    top_k: int = 5,
    max_context_length: int = 2000,
    database_url: Optional[str] = None,
    embedding_model: Optional[SentenceTransformer] = None,
) -> tuple[str, list[str]]:
    """
    Search for relevant chunks and build a context string for the LLM.

    Returns:
        Tuple of (context_string, list_of_source_names)
    """
    chunks = search_similar_chunks(
        query=query,
        tenant_id=tenant_id,
        top_k=top_k,
        database_url=database_url,
        embedding_model=embedding_model,
    )

    if not chunks:
        return "", []

    context_parts = []
    sources = []
    current_length = 0

    for chunk in chunks:
        content = chunk["content"]
        if current_length + len(content) > max_context_length:
            remaining = max_context_length - current_length
            if remaining > 100:
                content = content[:remaining] + "..."
                context_parts.append(content)
            break

        context_parts.append(content)
        sources.append(chunk["source_name"])
        current_length += len(content)

    context = "\n\n---\n\n".join(context_parts)
    unique_sources = list(dict.fromkeys(sources))

    return context, unique_sources
