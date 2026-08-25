"""
Unified Retrieval Entry Point
──────────────────────────────
`retrieve()` is the one function chat, quiz, and the eval runner
(scripts/eval_retrieval.py) all call. `app.services.domain_context.
build_domain_context` and `app.services.search.build_rag_context` become
thin (context, sources) wrappers around it, so existing call sites and
their test monkeypatches (tests/test_chat.py patches
app.routers.chat.build_domain_context; tests/test_quiz.py patches
app.services.llm.urllib...) are unaffected -- only what runs underneath
changed.

Owns the disk-backend candidate gathering directly (moved from
domain_context.py) rather than importing it from there, to avoid a
circular import: domain_context.py now imports FROM this module.

Language affinity (ADR 0002 decision 5, committed but previously
unimplemented): prefer chunks whose language matches ui_lang; only fall
back to the full candidate pool -- setting cross_language=True -- when too
few same-language chunks clear the bar. Do NOT implement this as
`ORDER BY (language = %s) DESC, ...` -- that ranks every same-language
chunk above a far better cross-language one; the two-pass shape here
always prefers the highest-similarity same-language set as a whole, and
only crosses the language line when that set is too small.
"""

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from app.config import get_settings
from app.services.ingestion import chunk_document, detect_document_language, load_embedding_model
from app.services.search import search_similar_chunks

REPO_ROOT = Path(__file__).resolve().parents[2]

# "securite" (Domain enum / DOMAIN_LABELS key) vs. "securite_physique" (the
# raw/ folder name) is a pre-existing naming mismatch elsewhere in the repo
# too -- not introduced here. See app.services.ingestion.DOMAIN_DIR_ALIASES,
# the same mapping applied at ingest time for the pgvector backend.
DOMAIN_TEXT_DIRS = {
    "industrial": REPO_ROOT / "raw" / "shared" / "industrial" / "text",
    "securite": REPO_ROOT / "raw" / "shared" / "securite_physique" / "text",
    "blockchain": REPO_ROOT / "raw" / "shared" / "blockchain" / "text",
}

# "darija" (the model-routing vocabulary app.services.llm uses) -> "ar" (the
# document-language vocabulary app.services.ingestion.detect_document_language
# and the documents.language column use). Different vocabularies for
# different things -- one is "which persona/template", the other is "what
# script is this text written in" -- so translate at the boundary rather
# than merging them into one.
UI_LANG_TO_DOC_LANG = {"darija": "ar", "fr": "fr"}


@dataclass
class RetrievalResult:
    context: str
    sources: list[str] = field(default_factory=list)
    cross_language: bool = False
    fingerprint: str = ""


@lru_cache(maxsize=len(DOMAIN_TEXT_DIRS))
def _disk_candidates(domain: str) -> tuple:
    """(chunk, source_filename, language, embedding) tuples for one
    domain's corpus, computed once per process. Small, pre-curated corpora
    (~30-50KB each) -- safe to hold fully in memory.

    Heading-aware chunking (app.services.ingestion.chunk_document): a
    chunk carries its section heading inline even when the body text never
    repeats it, so a reference like "Art. 283" or "المادة 4" can't be
    silently separated from the text it's embedded in.
    """
    text_dir = DOMAIN_TEXT_DIRS.get(domain)
    if text_dir is None or not text_dir.is_dir():
        return ()

    chunks: list[str] = []
    sources: list[str] = []
    languages: list[str] = []
    for path in sorted(text_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        file_language = detect_document_language(text)
        for doc_chunk in chunk_document(text):
            chunks.append(doc_chunk["content"])
            sources.append(path.name)
            languages.append(file_language)

    if not chunks:
        return ()

    embeddings = load_embedding_model().encode(chunks)
    return tuple(zip(chunks, sources, languages, embeddings))


def _disk_search(query: str, domain: str, top_k: int) -> list[dict]:
    entries = _disk_candidates(domain)
    if not entries:
        return []

    model = load_embedding_model()
    query_vec = model.encode([query])[0]
    query_norm = np.linalg.norm(query_vec)

    scored = sorted(
        (
            {
                "content": chunk,
                "source_name": source,
                "language": language,
                "similarity": float(np.dot(query_vec, emb) / (query_norm * np.linalg.norm(emb) + 1e-9)),
            }
            for chunk, source, language, emb in entries
        ),
        key=lambda c: c["similarity"],
        reverse=True,
    )
    # Overfetch: language-affinity selection needs more than top_k
    # candidates to have a same-language pool worth preferring.
    return scored[: max(top_k * 4, 20)]


def _pgvector_search(
    query: str, tenant_id: str, domain: Optional[str], top_k: int,
    source_ids: Optional[list[str]] = None,
) -> list[dict]:
    # domain is passed straight into search_similar_chunks's SQL WHERE
    # clause, not filtered in Python after a fixed-size LIMIT. Measured
    # live against a multi-domain tenant: filtering after the fetch let
    # two other domains' (and undomained legacy rows') chunks crowd a
    # domain's own true top-k out of the fetched pool entirely -- e.g. 6
    # of 7 real industrial+Arabic chunks never made a 20-candidate global
    # pool once industrial/securite/blockchain/legacy rows all competed
    # for the same slots. Scoping in SQL means the LIMIT is applied to an
    # already-correct candidate set. Also fixes the recorded bug where a
    # quiz labelled "industrial" could ground in blockchain chunks (no
    # domain filter existed at all, old quiz.py:28).
    return search_similar_chunks(
        query=query, tenant_id=tenant_id, top_k=max(top_k * 4, 20), domain=domain,
        source_ids=source_ids,
    )


def _select_with_affinity(
    candidates: list[dict], *, target_lang: Optional[str], top_k: int, threshold: Optional[float]
) -> tuple[list[dict], bool]:
    """Two-pass language-affinity selection, shared by both backends.

    threshold=None preserves the disk backend's existing no-cutoff
    philosophy (every candidate already belongs to the selected domain, so
    the lowest-ranked one is still on-topic); a float preserves the
    pgvector backend's existing similarity cutoff.

    target_lang=None means the caller did not opt into language affinity
    (no ui_lang was supplied) -- plain top-k by similarity, cross_language
    always False. This is deliberately opt-in, not a new default: without
    it, every existing caller that doesn't pass ui_lang (quiz.py today,
    any test that calls build_domain_context/build_rag_context directly)
    would otherwise be silently biased toward whatever ui_lang happened to
    default to, which is exactly the kind of unannounced behaviour change
    this codebase's own tests are meant to catch.
    """
    passing = [c for c in candidates if threshold is None or c["similarity"] >= threshold]

    if target_lang is None:
        return passing[:top_k], False

    same_lang = [c for c in passing if c.get("language") == target_lang]
    if len(same_lang) >= top_k:
        return same_lang[:top_k], False

    selected = passing[:top_k]
    cross_language = any(c.get("language") != target_lang for c in selected)
    return selected, cross_language


# Two chunks from the same document are built with CHUNK_OVERLAP=250
# characters of deliberate overlap (app.services.ingestion), so when both
# land in the same top-k the tail of one is literally the head of the
# next. Trimming that repeat is pure signal-to-noise: it costs the model
# nothing to lose a duplicate paragraph, and 250 characters is ~4% of the
# 6000-character context budget per adjacent pair.
_MIN_OVERLAP_TO_TRIM = 60


def _trim_leading_overlap(previous: str, content: str) -> str:
    """Drop the head of `content` that is already the tail of `previous`.

    Only exact overlaps are removed, longest first, and only when the
    repeat is at least _MIN_OVERLAP_TO_TRIM characters -- a short
    coincidental match (a shared heading line, a repeated article number)
    is NOT redundancy worth cutting, and cutting it could remove the one
    copy of a citation the answer needed. Bounded by the configured
    overlap, so this can never eat a chunk that merely resembles its
    predecessor.
    """
    from app.services.ingestion import CHUNK_OVERLAP

    limit = min(len(previous), len(content), CHUNK_OVERLAP * 2)
    for size in range(limit, _MIN_OVERLAP_TO_TRIM - 1, -1):
        if previous.endswith(content[:size]):
            return content[size:].lstrip()
    return content


def _build_context(selected: list[dict], *, max_context_length: int) -> tuple[str, list[str]]:
    context_parts: list[str] = []
    sources: list[str] = []
    current_length = 0
    seen_content: set[str] = set()
    for c in selected:
        content = c["content"]
        # Exact duplicates: the same chunk can legitimately be returned
        # twice once uploaded and global-corpus rows both match (a tenant
        # re-uploading a document that is also in raw/shared), and paying
        # context budget to say the same thing twice makes the answer
        # worse, not just longer.
        if content in seen_content:
            continue
        seen_content.add(content)
        if context_parts:
            content = _trim_leading_overlap(context_parts[-1], content)
            if not content:
                continue
        if current_length + len(content) > max_context_length:
            remaining = max_context_length - current_length
            if remaining > 100:
                context_parts.append(content[:remaining] + "...")
                # Was missing before the chunk-size increase to ~2000
                # chars (2026-08-13) -- at the old 400-char chunk size this
                # branch essentially never fired, so the gap was latent.
                # With larger chunks the FIRST chunk alone can exhaust
                # max_context_length, and without this line the answer
                # comes back with real (truncated) context but sources==[],
                # which reads as "the sidebar is broken" rather than what
                # it actually is: a citation silently dropped on truncation.
                sources.append(c.get("source_name", ""))
            break
        context_parts.append(content)
        sources.append(c.get("source_name", ""))
        current_length += len(content)

    context = "\n\n---\n\n".join(context_parts)
    unique_sources = list(dict.fromkeys(s for s in sources if s))
    return context, unique_sources


def _fingerprint(*, domain: str, ui_lang: Optional[str], query: str) -> str:
    key = f"{domain}|{ui_lang}|{query}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def retrieve(
    query: str,
    *,
    domain: str,
    backend: str,
    tenant_id: Optional[str] = None,
    ui_lang: Optional[str] = None,
    top_k: int = 4,
    threshold: Optional[float] = None,
    max_context_length: int = 6000,
    source_ids: Optional[list[str]] = None,
) -> RetrievalResult:
    """Retrieve context for `query` in `domain`.

    `ui_lang` ("fr" | "darija") is optional and opt-in: omitted, retrieval
    is plain top-k by similarity (today's pre-existing behaviour, exactly).
    Supplied, it applies ADR 0002 decision 5's language-affinity two-pass
    -- prefer chunks whose language matches, only cross the language line
    when too few same-language chunks exist. Callers that don't have a
    resolved ui_lang (quiz.py today) must not have one silently assumed
    for them; see _select_with_affinity's docstring.

    `backend` ("disk" | "pgvector") is REQUIRED, not read from settings
    internally -- app.services.domain_context.build_domain_context and
    app.services.search.build_rag_context each force their own historical
    backend (disk and pgvector respectively) regardless of
    settings.retrieval_backend, so quiz's always-pgvector behaviour can
    never be silently redirected to disk by a flag meant for chat's
    migration. The flag itself is read at the one call site it's meant to
    control -- app/routers/chat.py -- which picks which of those two
    wrappers to call. See docs/architecture/data-and-retrieval.md for why
    the cutover is gated on scripts/eval_retrieval.py, not assumed.

    `source_ids`: forwarded to the pgvector path only (see
    search_similar_chunks's docstring for the SQL). The disk backend
    reads raw/shared/<domain>/text/*.md, which by definition never
    contains a tenant upload, so this parameter has nothing to filter
    there and is silently ignored on that branch rather than erroring --
    a caller building one code path for both backends (e.g. the
    pgvector-with-disk-fallback in app/routers/chat.py) shouldn't have to
    branch just to drop this argument for disk.
    """
    settings = get_settings()
    target_lang = UI_LANG_TO_DOC_LANG.get(ui_lang) if ui_lang else None

    if backend == "pgvector":
        candidates = _pgvector_search(
            query, tenant_id or settings.default_tenant_id, domain, top_k, source_ids=source_ids
        )
        # settings.similarity_threshold -- see docs/architecture/
        # data-and-retrieval.md for the live sweep this default is from.
        # Re-swept for bge-m3 as part of the 2026-08-13 embedding-model
        # migration; was a hardcoded 0.15 benchmarked against MiniLM.
        selected, cross_language = _select_with_affinity(
            candidates, target_lang=target_lang, top_k=top_k,
            threshold=threshold if threshold is not None else settings.similarity_threshold,
        )
    elif backend == "disk":
        candidates = _disk_search(query, domain, top_k)
        selected, cross_language = _select_with_affinity(
            candidates, target_lang=target_lang, top_k=top_k, threshold=threshold
        )
    else:
        raise ValueError(f"unknown retrieval backend {backend!r}")

    context, sources = _build_context(selected, max_context_length=max_context_length)
    return RetrievalResult(
        context=context,
        sources=sources,
        cross_language=cross_language,
        fingerprint=_fingerprint(domain=domain, ui_lang=ui_lang, query=query),
    )
