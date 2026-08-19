"""
Turn Routing — Domain + Language Resolution
─────────────────────────────────────────────
Single seam chat.py and quiz.py call instead of each duplicating the
UI_LANG_TO_MODEL_LANG / detect_query_language block. Two responsibilities:

1. Language: which language RETRIEVAL should search in (`query_lang`) vs.
   which language the MODEL should answer in (`response_lang`). These were
   one variable until an explicit in-message instruction ("comment on fait
   cela ? réponds en darija") could make them diverge -- see
   resolve_language() below.
2. Domain: which tenant domain (industrial/securite/blockchain) this turn
   belongs to, when the caller doesn't already know (page context) -- see
   resolve_domain() below. (Added alongside the language resolver; DB-backed
   voting lands with the retrieval wiring.)

resolve_language() is a pure function over its inputs -- no DB/session
access -- so callers own persistence (chat_sessions.response_lang_override
etc.) and it stays fully unit-testable without Postgres. resolve_domain()
does reach Postgres for its tier-2 vote (vote_domain() itself is pure and
tested directly against fabricated candidates); it fails open to the tier-3
tenant default on any exception, the same fail-open contract
app/routers/chat.py's _retrieve_context already applies to the pgvector
retrieval path -- a Postgres hiccup degrades routing rather than crashing
the request.
"""
import logging
from dataclasses import dataclass
from typing import Optional

from app.config import get_settings
from app.services.llm import detect_language_instruction, detect_query_language
from app.services.search import search_similar_chunks

logger = logging.getLogger(__name__)


@dataclass
class LanguageResolution:
    query_lang: str
    response_lang: str
    # "explicit_field" | "explicit_instruction" | "sticky" | "script"
    response_lang_source: str
    # What the caller should persist to chat_sessions for the NEXT turn's
    # stickiness check. None means "clear the stored override".
    override_to_persist: Optional[str]
    override_query_lang_to_persist: Optional[str]


def resolve_language(
    message: str,
    *,
    explicit_language: Optional[str] = None,
    stored_override: Optional[str] = None,
    stored_override_query_lang: Optional[str] = None,
) -> LanguageResolution:
    """Resolve `query_lang` (what retrieval searches in) and `response_lang`
    (what the model answers in) for one turn.

    Precedence, first hit wins:

    0. `explicit_language` -- an API-supplied language (ChatRequest.language
       / QuizRequest.language), authoritative for both query and response,
       exactly as it was before this module existed. Not "page context" in
       the domain-routing sense, but the same idea: an explicit caller-known
       value always beats detection.
    1. An explicit instruction IN THIS MESSAGE ("réponds en darija") --
       response_lang follows the instruction while query_lang stays the
       message's own script (a French question asking for a Darija answer
       still retrieves best against French chunks). Persisted so the next
       turn's stickiness check (case 2) has a baseline to compare against.
    2. A stored override exists AND this message's query_lang still equals
       the query_lang that was in effect when the override was set -- keep
       the override. This is the actual "stickiness": the user keeps
       writing in the same script and keeps getting the overridden answer
       language without repeating the instruction every turn, at the cost
       of one model swap total rather than one per turn.
    3. Otherwise -- response_lang just follows query_lang (today's
       pre-override behaviour), and any stored override is cleared. This
       guarantees a session can never get stuck answering in a language the
       user's own script no longer implies.
    """
    query_lang = detect_query_language(message)

    if explicit_language is not None:
        return LanguageResolution(
            query_lang=explicit_language,
            response_lang=explicit_language,
            response_lang_source="explicit_field",
            override_to_persist=None,
            override_query_lang_to_persist=None,
        )

    instruction = detect_language_instruction(message)
    if instruction is not None:
        return LanguageResolution(
            query_lang=query_lang,
            response_lang=instruction,
            response_lang_source="explicit_instruction",
            override_to_persist=instruction,
            override_query_lang_to_persist=query_lang,
        )

    if stored_override is not None and stored_override_query_lang == query_lang:
        return LanguageResolution(
            query_lang=query_lang,
            response_lang=stored_override,
            response_lang_source="sticky",
            override_to_persist=stored_override,
            override_query_lang_to_persist=stored_override_query_lang,
        )

    return LanguageResolution(
        query_lang=query_lang,
        response_lang=query_lang,
        response_lang_source="script",
        override_to_persist=None,
        override_query_lang_to_persist=None,
    )


def vote_domain(candidates: list[dict], *, threshold: Optional[float] = None) -> Optional[str]:
    """Similarity-weighted domain vote over UNFILTERED retrieval candidates
    (each a dict with 'domain' and 'similarity', the shape
    search_similar_chunks returns). Returns the domain with the highest
    summed similarity, or None if nothing clears the threshold.

    Weighted by similarity, not a raw per-domain count, so one strong match
    isn't outvoted by several weaker ones from a different domain. A
    candidate with domain=None (a legacy/untagged row) never votes -- an
    undomained chunk carries no information about which domain a query
    belongs to, and letting it vote would be exactly the kind of silent
    cross-domain leak the domain column exists to prevent.
    """
    effective_threshold = threshold if threshold is not None else get_settings().domain_vote_threshold
    scores: dict[str, float] = {}
    for c in candidates:
        domain = c.get("domain")
        similarity = c.get("similarity", 0.0)
        if not domain or similarity < effective_threshold:
            continue
        scores[domain] = scores.get(domain, 0.0) + similarity
    if not scores:
        return None
    return max(scores, key=scores.get)


def resolve_domain(
    query: str, *, tenant_id: str, backend: str, threshold: Optional[float] = None,
    source_ids: Optional[list[str]] = None,
) -> tuple[str, str]:
    """Resolve which domain this turn belongs to, for callers that don't
    already have one from page context (tier 1 -- handled entirely by the
    caller before this is ever invoked; this function only implements
    tiers 2 and 3). Returns (domain, domain_source).

    Tier 2, pgvector backend only: one UNFILTERED search across the tenant's
    whole corpus (search_similar_chunks(domain=None)), then vote_domain()
    picks the winner. The caller re-retrieves filtered to that domain
    afterward -- two searches, deliberately: an unfiltered pool is exactly
    the condition search_similar_chunks's own docstring records as starving
    a domain's true top-k out of a fixed-size candidate pool. Sub-ms at this
    corpus size.

    `source_ids`, forwarded to the tier-2 vote: without it, a DISABLED
    tenant upload could still swing the automatic domain vote even though
    it's excluded from every other part of retrieval. None (default)
    means no filter, same as before uploads existed.

    Tier 3: settings.default_domain, under one of two DISTINCT sources --
    the distinction is the point, so callers can tell "we don't know" from
    "nothing matched":

      "tenant_default" -- routing could not form an opinion at all: backend
          == "disk" (the disk backend has no single cross-domain corpus to
          vote over without embedding all three domains' text at once --
          app.services.retrieval._disk_candidates is keyed per-domain by
          design, so it skips straight to the tenant default), or tier 2's
          search itself raised (fail-open -- a Postgres hiccup degrades
          routing to the tenant default rather than crashing the request,
          the same contract app.routers.chat._retrieve_context already
          applies to pgvector retrieval).

      "no_match" -- tier 2 ran successfully and found NOTHING above
          domain_vote_threshold. That is a real, positive signal that the
          query is out of corpus, and it used to be discarded: every branch
          collapsed into "tenant_default", so an off-topic question ("how do
          I bake bread") was silently answered under the default domain.
          Retrieval then almost always returns SOMETHING for it, so
          app.routers.chat's `if not context.strip()` refusal never fired.
          Returned so that caller can refuse; the domain is still the tenant
          default, since a caller that chooses NOT to refuse needs a usable
          domain to carry on with.
    """
    settings = get_settings()
    if backend == "pgvector":
        try:
            candidates = search_similar_chunks(
                query=query, tenant_id=tenant_id, top_k=20, domain=None, source_ids=source_ids
            )
            voted = vote_domain(candidates, threshold=threshold)
            if voted:
                return voted, "retrieval"
            return settings.default_domain, "no_match"
        except Exception:
            logger.exception(
                "tier-2 domain routing failed for tenant=%s; falling back to tenant default",
                tenant_id,
            )
    return settings.default_domain, "tenant_default"
