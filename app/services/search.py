"""
Semantic Search Service
──────────────────────
Converts user queries to vectors and retrieves relevant document chunks from pgvector.
"""

from typing import Optional

import psycopg2
from sentence_transformers import SentenceTransformer

from app.config import get_settings
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
    domain: Optional[str] = None,
) -> list[dict]:
    """
    Search for the most relevant document chunks for a given query.

    Args:
        query: User's question text
        tenant_id: Company/tenant identifier (isolation)
        top_k: Number of results to return
        database_url: PostgreSQL connection URL
        embedding_model: Pre-loaded SentenceTransformer model
        domain: Optional domain filter, applied IN THE SQL QUERY (not
            after fetching). Filtering in Python after a fixed-size LIMIT
            was measured live to starve a domain's own chunks out of the
            candidate pool entirely on a multi-domain tenant: a query's
            true top-k for one domain can rank below the LIMIT cutoff once
            two other domains' chunks (and any legacy undomained rows)
            are competing for the same slots. Scoping in SQL means the
            LIMIT applies to an already-correct candidate set.

    Returns:
        List of dicts with keys: content, source_name, similarity, metadata
    """
    db_url = database_url or get_settings().database_url

    if embedding_model is None:
        embedding_model = load_embedding_model()

    query_embedding = embedding_model.encode([query])[0].tolist()

    conn = get_db_connection(db_url)
    try:
        cursor = conn.conn.cursor() if hasattr(conn, "conn") else conn.cursor()

        domain_clause = "AND domain = %s" if domain else ""
        search_query = f"""
            SELECT
                id,
                content,
                source_name,
                source_type,
                domain,
                language,
                metadata,
                1 - (embedding <=> %s::vector) AS similarity
            FROM documents
            WHERE tenant_id = %s
            {domain_clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """

        embedding_str = str(query_embedding)
        params = [embedding_str, tenant_id]
        if domain:
            params.append(domain)
        params.extend([embedding_str, top_k])
        cursor.execute(search_query, params)

        results = []
        for row in cursor.fetchall():
            results.append({
                "id": str(row[0]),
                "content": row[1],
                "source_name": row[2],
                "source_type": row[3],
                "domain": row[4],
                "language": row[5],
                "metadata": row[6],
                "similarity": float(row[7]),
            })

        return results
    finally:
        conn.close()


def build_rag_context(
    query: str,
    tenant_id: str,
    top_k: int = 5,
    similarity_threshold: float = 0.15,
    max_context_length: int = 2000,
    database_url: Optional[str] = None,
    embedding_model: Optional[SentenceTransformer] = None,
    domain: Optional[str] = None,
    ui_lang: Optional[str] = None,
) -> tuple[str, list[str]]:
    """
    Search for relevant chunks and build a context string for the LLM.

    Thin (context, sources) wrapper around app.services.retrieval.retrieve()
    -- kept as its own function (not just calling retrieve() at every call
    site) so this module's existing signature and the tests that import it
    directly are unaffected. `database_url`/`embedding_model` are accepted
    for backward compatibility but unused: retrieve()'s pgvector path
    always uses the configured settings connection and the shared cached
    embedding model, same as before.

    `similarity_threshold` default is 0.15, not the original 0.4 -- measured
    live against the real ingested corpus (2026-08-10): correct, gold-source
    matches for cross-lingual (French-embedding-model-scoring-Darija) queries
    scored as low as 0.155-0.177, so 0.4 discarded them outright regardless
    of the domain/language-affinity fixes. A threshold sweep (0.0-0.4) traded
    recall@4 against out-of-domain refusal rate on a straight curve with no
    free lunch; 0.15 was chosen because it holds recall@4 near the ceiling
    (0.956 vs 0.978 at 0.0) while keeping some refusal protection, matching
    this project's established priority that a false refusal mid-conversation
    is worse than an occasional missed refusal on a junk query -- see
    docs/architecture/data-and-retrieval.md for the full sweep table.

    Returns:
        Tuple of (context_string, list_of_source_names)
    """
    from app.services.retrieval import retrieve  # local import: retrieval.py imports this module

    result = retrieve(
        query=query,
        domain=domain or "",
        backend="pgvector",  # this module's identity, independent of settings.retrieval_backend
        tenant_id=tenant_id,
        ui_lang=ui_lang,
        top_k=top_k,
        threshold=similarity_threshold,
        max_context_length=max_context_length,
    )
    return result.context, result.sources
