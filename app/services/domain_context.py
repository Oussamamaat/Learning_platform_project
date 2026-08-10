"""
Demo-mode Domain Context Provider
──────────────────────────────────
`app/routers/chat.py`'s domain selector (industrial / securite / blockchain)
already reaches every request, but `app.services.search.build_rag_context`
answers from whatever a tenant has *ingested into Postgres* -- for this
demo's `company_abc` tenant that's an unrelated road-traffic corpus, not the
domains the selector offers, so real questions in-domain were retrieving
nothing and falling through to the empty-context refusal. Not a gate bug;
a tenant/ingestion mismatch.

This module answers from the domain's own source documents instead
(`raw/shared/<domain>/text/*.md` -- the same regulatory corpus the model's
training context chunks were drawn from), using the same embedding model,
chunking (`app.services.ingestion.strip_markdown`/`chunk_text`), and
`(context, sources)` contract as `build_rag_context`, so
`app/routers/chat.py`'s call site and everything downstream of it
(`_build_system_prompt`, `deterministic_refusal`) are unaffected -- only the
retrieval backend changes, not the shape the model sees. No Postgres
involved; each domain's small, pre-curated corpus is chunked and embedded
once per process and cached.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np

from app.services.ingestion import chunk_text, load_embedding_model, strip_markdown

REPO_ROOT = Path(__file__).resolve().parents[2]

# "securite" (Domain enum / DOMAIN_LABELS key) vs. "securite_physique"
# (the raw/ folder name) is a pre-existing naming mismatch elsewhere in the
# repo too -- not introduced here.
DOMAIN_TEXT_DIRS = {
    "industrial": REPO_ROOT / "raw" / "shared" / "industrial" / "text",
    "securite": REPO_ROOT / "raw" / "shared" / "securite_physique" / "text",
    "blockchain": REPO_ROOT / "raw" / "shared" / "blockchain" / "text",
}


@lru_cache(maxsize=len(DOMAIN_TEXT_DIRS))
def _domain_chunks(domain: str) -> tuple:
    """(chunk, source_filename, embedding) tuples for one domain's corpus,
    computed once per process. Small, pre-curated corpora (~30-50KB each)
    -- safe to hold fully in memory."""
    text_dir = DOMAIN_TEXT_DIRS.get(domain)
    if text_dir is None or not text_dir.is_dir():
        return ()

    chunks: list[str] = []
    sources: list[str] = []
    for path in sorted(text_dir.glob("*.md")):
        cleaned = strip_markdown(path.read_text(encoding="utf-8"))
        for chunk in chunk_text(cleaned):
            chunks.append(chunk)
            sources.append(path.name)

    if not chunks:
        return ()

    embeddings = load_embedding_model().encode(chunks)
    return tuple(zip(chunks, sources, embeddings))


def build_domain_context(
    query: str,
    domain: str,
    top_k: int = 4,
    max_context_length: int = 2000,
) -> tuple[str, list[str]]:
    """Same (context, sources) contract as
    `app.services.search.build_rag_context`, ranking the domain's own
    documents by cosine similarity to `query` instead of a pgvector lookup.

    No similarity threshold: unlike a mixed multi-tenant corpus, every chunk
    here already belongs to the selected domain, so the lowest-ranked chunk
    is still on-topic -- thresholding would only risk reintroducing false
    empty-context refusals, the exact failure mode this module exists to fix.
    """
    entries = _domain_chunks(domain)
    if not entries:
        return "", []

    model = load_embedding_model()
    query_vec = model.encode([query])[0]
    query_norm = np.linalg.norm(query_vec)

    scored = sorted(
        (
            (
                float(np.dot(query_vec, emb) / (query_norm * np.linalg.norm(emb) + 1e-9)),
                chunk,
                source,
            )
            for chunk, source, emb in entries
        ),
        key=lambda t: t[0],
        reverse=True,
    )

    context_parts: list[str] = []
    sources_used: list[str] = []
    current_length = 0
    for _sim, chunk, source in scored:
        if len(context_parts) >= top_k:
            break
        if current_length + len(chunk) > max_context_length:
            continue
        context_parts.append(chunk)
        sources_used.append(source)
        current_length += len(chunk)

    context = "\n\n---\n\n".join(context_parts)
    unique_sources = list(dict.fromkeys(sources_used))
    return context, unique_sources
