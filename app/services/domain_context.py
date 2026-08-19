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
chunking, and `(context, sources)` contract as `build_rag_context`, so
`app/routers/chat.py`'s call site and everything downstream of it
(`_build_system_prompt`, `deterministic_refusal`) are unaffected -- only the
retrieval backend changes, not the shape the model sees. No Postgres
involved; each domain's small, pre-curated corpus is chunked and embedded
once per process and cached.

The actual candidate-gathering and caching now live in
app.services.retrieval (moved there to avoid a circular import once that
module needed to dispatch between this backend and the pgvector one) --
this module is the thin, disk-forced wrapper kept for its existing call
sites and their test monkeypatches.
"""

from pathlib import Path
from typing import Optional

# Re-exported for backward compatibility -- nothing outside this module
# reads them directly today, but they lived here before app.services.
# retrieval.py existed and are cheap to keep stable.
from app.services.retrieval import DOMAIN_TEXT_DIRS, REPO_ROOT  # noqa: F401


def build_domain_context(
    query: str,
    domain: str,
    top_k: int = 4,
    max_context_length: int = 6000,
    ui_lang: Optional[str] = None,
) -> tuple[str, list[str]]:
    """Same (context, sources) contract as
    `app.services.search.build_rag_context`, ranking the domain's own
    documents by cosine similarity to `query` instead of a pgvector lookup.

    Thin wrapper around app.services.retrieval.retrieve(), always forced
    to the "disk" backend regardless of settings.retrieval_backend -- this
    module exists specifically to answer from the domain's own
    pre-curated corpus (see module docstring), and must never silently
    start hitting Postgres.

    No similarity threshold: unlike a mixed multi-tenant corpus, every chunk
    here already belongs to the selected domain, so the lowest-ranked chunk
    is still on-topic -- thresholding would only risk reintroducing false
    empty-context refusals, the exact failure mode this module exists to fix.
    """
    from app.services.retrieval import retrieve  # local import: retrieval.py imports this module

    result = retrieve(
        query=query,
        domain=domain,
        backend="disk",  # this module's identity, independent of settings.retrieval_backend
        ui_lang=ui_lang,
        top_k=top_k,
        max_context_length=max_context_length,
    )
    return result.context, result.sources
