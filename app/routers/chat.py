import logging
import uuid
from typing import Optional
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.config import get_settings, get_tenant_id, get_user_id
from app.models.schemas import ChatRequest, ChatResponse
from app.services import history
from app.services.domain_context import build_domain_context
from app.services.retrieval import _fingerprint
from app.services.routing import resolve_domain, resolve_language
from app.services.search import build_rag_context
from app.services.learner_state import get_learner_state
from app.services.llm import (
    generate_llm_response,
    deterministic_refusal,
    condense_retrieval_query,
    is_anaphoric_followup,
    UI_LANG_TO_MODEL_LANG,
)
from app.errors import AppError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

# UI-facing signal only, computed against the FINAL sources chat answers
# with (fresh retrieval or a reused pin, either way) -- not threaded through
# the retrieval/pin machinery itself, which already tracks and reasons
# about cross-language internally (app.services.retrieval's
# _select_with_affinity). Same "_ar_" filename convention search.py's own
# eval tooling (scripts/eval_retrieval.py) already uses. Compared against
# response_lang (not query_lang): the question this answers is "do the
# sources I'm about to phrase an answer FROM match the language I'm
# phrasing it IN" -- always true without an override, and the exact signal
# worth surfacing to the UI when an override makes them diverge.
def _sources_look_cross_language(sources: list[str], response_lang: str) -> bool:
    if not sources:
        return False
    target_is_arabic = response_lang == "darija"
    return any(("_ar_" in s) != target_is_arabic for s in sources)


def _retrieve_context(
    query: str, *, domain: str, top_k: int, ui_lang: str, tenant_id: str
) -> tuple[str, list[str]]:
    """Single retrieval indirection point for chat -- reads
    settings.retrieval_backend (the flag app.services.domain_context.
    build_domain_context and app.services.search.build_rag_context each
    deliberately ignore, so quiz's always-pgvector behaviour can never be
    silently redirected by a flag meant for chat) and calls the matching
    backend.

    Fail-open on the pgvector path specifically: unlike build_domain_context
    (zero external infra beyond the embedding model), build_rag_context
    requires a reachable Postgres. Making pgvector the default retrieval
    backend must not turn a Postgres hiccup mid-demo into a crashed chat
    request -- it should degrade to the disk corpus, the same fail-open
    contract app/services/history.py already keeps for conversation memory.
    A successful pgvector call is returned as-is; only a raised exception
    triggers the fallback, so an empty-but-valid pgvector result (a real
    "no matching content") is never overridden.
    """
    backend = get_settings().retrieval_backend
    if backend == "pgvector":
        try:
            return build_rag_context(
                query=query, tenant_id=tenant_id, top_k=top_k, domain=domain, ui_lang=ui_lang,
            )
        except Exception:
            logger.exception(
                "pgvector retrieval failed for domain=%s; falling back to the disk corpus", domain
            )
    return build_domain_context(query=query, domain=domain, top_k=top_k, ui_lang=ui_lang)


def _resolve_turn_context(
    session_id: str,
    message: str,
    requested_domain: Optional[str],
    query_lang: str,
    tenant_id: str,
):
    """Decide which context this turn answers from, which domain and
    segment it belongs to -- the pinned-context / segment-reset design
    (memory+RAG plan, Stage 3), extended 2026-08-11 to also resolve domain
    automatically (the "Automatic Domain Routing" plan) instead of taking
    it as a given. Retrieve once per segment and reuse the pin verbatim for
    same-topic follow-ups (byte-identical system block -> full KV-prefix
    reuse); only start a new segment on a real topic shift, and only call
    the (DB-touching) domain router when the pin can't just be tried first.

    The two failure directions are asymmetric -- a missed reset is a bad
    but VISIBLE answer (wrong sources shown), a false reset can fire
    `deterministic_refusal` mid-conversation, the single most visible
    possible demo failure. So this is biased against false resets, via
    three guards:

    1. An anaphoric follow-up (is_anaphoric_followup) can NEVER trigger a
       reset or a routing decision -- by definition it carries no
       standalone retrieval signal, so it isn't even evaluated as a reset
       candidate. Reuses the pin outright (domain included), no router
       call, no retrieval call.
    2. Retrieval for a reset-CANDIDATE check always runs on the condensed
       query (prior turn + message), not the bare message -- so a
       vaguely-connected follow-up without an anaphora marker ("Et les
       mains ?") still retrieves against real content instead of a
       fragment before its sources are compared to the pin.
    3. If that reset-candidate retrieval comes back empty, fall back to
       the OLD pinned context rather than refusing -- a stale-but-relevant
       answer beats a false refusal, and the returned sources still expose
       it as the previous topic.

    Domain routing (app.services.routing.resolve_domain) runs once per
    non-anaphoric turn whenever the caller didn't supply an explicit
    domain -- including same-topic continuations, not just new segments.
    That's a deliberate simplification over trying to guess "is this still
    the pinned domain" before ever calling the router: a speculative guess
    risks silently grounding a genuine cross-domain topic shift in the
    WRONG (stale) domain, which is exactly the class of bug the domain
    column exists to prevent. The vote is an unfiltered search, "sub-ms at
    this corpus size" per the routing plan -- paying it every non-anaphoric
    turn is cheap; guessing wrong is not.

    A query_lang switch always starts a new segment outright (retrieval
    affinity depends on it, so there is no sensible "same topic" check
    across that boundary) -- unlike response_lang, which can flip via a
    sticky override without touching the pin at all (see
    app.services.routing.resolve_language).

    Returns (context, sources, segment_id, is_new_pin, domain, domain_source).
    domain_source is "page_context" whenever `requested_domain` was given
    (regardless of whether the pin was reused), or "retrieval"/
    "tenant_default" (a fresh tier 2/3 decision) / "pinned" (the router's
    decision was abandoned in favour of stale-but-relevant pinned content,
    guard 3) when it wasn't.
    """
    pinned = history.get_pinned(session_id)

    # Guard 1: an anaphoric follow-up never triggers a routing decision or
    # a reset -- reuse the pin outright. Requires the pin's language to
    # still match this turn's query_lang (a language switch is never "just
    # a continuation") and, when the caller supplied an explicit domain,
    # that it still matches the pin's -- an explicit page-context switch is
    # a real, deliberate signal that must not be silently overridden by
    # "the message looked vague". When domain ISN'T explicit, no such check
    # is possible without calling the router itself, which this guard
    # exists specifically to avoid -- so an anaphoric message is trusted to
    # stay in the pin's domain unconditionally in that case.
    pin_still_plausible = (
        pinned is not None
        and pinned["language"] == query_lang
        and (requested_domain is None or requested_domain == pinned["domain"])
    )
    if pin_still_plausible and is_anaphoric_followup(message):
        domain_source = "page_context" if requested_domain is not None else "pinned"
        return (
            pinned["context"], pinned["sources"], pinned["segment_id"], False,
            pinned["domain"], domain_source,
        )

    if requested_domain is not None:
        domain, domain_source = requested_domain, "page_context"
    else:
        domain, domain_source = resolve_domain(
            message, tenant_id=tenant_id, backend=get_settings().retrieval_backend
        )

    same_pin_scope = bool(pinned) and pinned["domain"] == domain and pinned["language"] == query_lang

    prior_turn = None
    if same_pin_scope:
        window = history.load_window(
            session_id, domain=domain, language=query_lang, segment_id=pinned["segment_id"], max_messages=2
        )
        if window:
            prior_turn = window[0]["content"]
    retrieval_query = condense_retrieval_query(message, prior_turn)
    probe_context, probe_sources = _retrieve_context(
        retrieval_query, domain=domain, top_k=4, ui_lang=query_lang, tenant_id=tenant_id
    )

    if not same_pin_scope:
        new_segment_id = (pinned["segment_id"] + 1) if pinned else 1
        if not probe_context.strip() and pinned and pinned["context"].strip():
            return (
                pinned["context"], pinned["sources"], pinned["segment_id"], False,
                pinned["domain"], "pinned",
            )
        return probe_context, probe_sources, new_segment_id, True, domain, domain_source

    if set(probe_sources) & set(pinned["sources"]):
        # Same topic: reuse the pin verbatim. The probe retrieval above was
        # only a topic-shift check -- using its result here instead would
        # re-render the system block every turn and defeat the pin.
        return pinned["context"], pinned["sources"], pinned["segment_id"], False, domain, domain_source

    if not probe_context.strip():
        logger.info(
            "session=%s: reset-candidate retrieval empty, falling back to pinned context",
            session_id,
        )
        return pinned["context"], pinned["sources"], pinned["segment_id"], False, domain, domain_source

    return probe_context, probe_sources, pinned["segment_id"] + 1, True, domain, domain_source


@router.post("/", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Text-based conversation endpoint.
    Receives user message, performs RAG retrieval, and queries the local LLM.
    """
    tenant_id = get_tenant_id(request.tenant_id)
    user_id = get_user_id()
    # A session_id omitted by the client must never collide with another
    # anonymous session -- "new-session" as a literal shared ID was fine
    # when nothing persisted, but chat_messages now keys history on it.
    session_id = request.session_id or uuid.uuid4().hex

    # Language: explicit field > in-message instruction ("réponds en
    # darija") > a sticky prior override > script detection. query_lang
    # drives retrieval affinity; response_lang drives the model/system
    # prompt/refusal wording -- see app.services.routing.resolve_language
    # for why these can diverge.
    explicit_lang = (
        UI_LANG_TO_MODEL_LANG.get(request.language.value) if request.language else None
    )
    stored_override, stored_override_query_lang = history.get_language_state(session_id)
    lang = resolve_language(
        request.message,
        explicit_language=explicit_lang,
        stored_override=stored_override,
        stored_override_query_lang=stored_override_query_lang,
    )
    query_lang = lang.query_lang
    response_lang = lang.response_lang

    # 1. Resolve domain + context + segment via the pinned-context/
    # segment-reset/domain-routing design above. Fail-open underneath
    # (history.get_pinned/load_window never raise, resolve_domain falls
    # back to the tenant default on a DB error), so a Postgres hiccup here
    # degrades to segment_id=1, always-fresh-retrieval, tenant-default
    # domain -- never a crash.
    requested_domain = request.domain.value if request.domain else None
    context, sources, segment_id, is_new_pin, domain, domain_source = _resolve_turn_context(
        session_id, request.message, requested_domain, query_lang, tenant_id
    )

    # 2. Empty context: refuse deterministically rather than let the model
    # compose its own refusal. The fine-tuned model's refusal register is
    # welded to tenant #1's safety domain regardless of the actual tenant
    # domain -- this bypasses that bias entirely instead of prompting
    # around it. See app/services/llm.py:deterministic_refusal. History is
    # not loaded or extended on a refusal -- there is nothing to reply to.
    if not context.strip():
        return ChatResponse(
            response=deterministic_refusal(domain, response_lang),
            session_id=session_id,
            sources=[],
            tokens_used=0,
            domain=domain,
            domain_source=domain_source,
            language=response_lang,
        )

    # 3. Load prior turns for this exact (session, domain, query_lang,
    # segment) -- fail-open: an empty list here means "no history", never
    # an error, whether that's a new session, a fresh segment, or a
    # Postgres hiccup.
    prior_turns = history.load_window(
        session_id, domain=domain, language=query_lang, segment_id=segment_id
    )

    # Learner-state seam (ADR 0003 owns the answer via quiz_attempts + an
    # IRT ability estimate; chat only reads, never writes). Returns {}
    # until that table exists, so nothing is injected into the prompt yet
    # -- the injection point exists so wiring it up later is a one-line
    # change here, not a new call site.
    _learner_state = get_learner_state(user_id, tenant_id, domain)

    # 4. Call local Ollama model to generate answer based on context + history
    try:
        ai_reply = generate_llm_response(
            query=request.message,
            context=context,
            domain=domain,
            language=response_lang,
            history=prior_turns,
        )
    except AppError as e:
        logger.error("LLM error in chat: %s", e.code)
        return JSONResponse(
            status_code=e.status_code,
            content={"error": {"code": e.code, "message": e.message}},
        )

    # 5. Persist the exchange for the next turn, including the sticky
    # language-override state (None/None clears a previously active one --
    # see app.services.routing.resolve_language's case 3). Fail-open
    # (history.py logs and swallows) -- a storage failure must not turn a
    # good answer into a 500.
    history.append_exchange(
        session_id,
        tenant_id=tenant_id,
        user_id=user_id,
        domain=domain,
        language=query_lang,
        segment_id=segment_id,
        user_content=request.message,
        assistant_content=ai_reply,
        sources=sources,
        response_lang_override=lang.override_to_persist,
        override_query_lang=lang.override_query_lang_to_persist,
    )

    # Pin this segment's context only when it's genuinely new (a fresh
    # segment or the session's first turn) -- never on a reused pin or a
    # same-topic continuation, so the pin (and therefore the system block
    # every subsequent same-topic turn sends) stays byte-identical.
    if is_new_pin:
        history.pin_context(
            session_id,
            tenant_id=tenant_id,
            user_id=user_id,
            domain=domain,
            language=query_lang,
            segment_id=segment_id,
            context=context,
            sources=sources,
            # Hashed, not the raw "domain|ui_lang|message" string -- that
            # overflowed pinned_fingerprint's VARCHAR(64) on any real
            # message and silently failed every pin write (caught live,
            # fail-open masked it as a logged exception, not a visible
            # chat failure -- but the pin never persisted, defeating the
            # KV-prefix-reuse it exists for).
            fingerprint=_fingerprint(domain=domain, ui_lang=query_lang, query=request.message),
        )

    # 6. Socratic questions already asked in turns that fell out of the
    # replay window -- deterministic extraction, UI-only, never re-injected
    # into a future prompt. See history.extract_dropped_questions.
    prior_questions = history.extract_dropped_questions(
        session_id, domain=domain, language=query_lang, segment_id=segment_id
    )
    if prior_questions:
        logger.info(
            "session=%s carries %d previously-asked question(s) outside the replay window",
            session_id, len(prior_questions),
        )

    return ChatResponse(
        response=ai_reply,
        session_id=session_id,
        sources=sources,
        tokens_used=0,
        domain=domain,
        domain_source=domain_source,
        language=response_lang,
        prior_questions=prior_questions,
        cross_language=_sources_look_cross_language(sources, response_lang),
    )
