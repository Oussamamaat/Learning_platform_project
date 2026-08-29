"""
Turn Pipeline (shared by voice, and available to any future non-HTTP
caller)
─────────────────────────────────────────────────────────────────────────
Wraps app.routers.chat's turn-resolution machinery
(_resolve_turn_context: pinned-context reuse, segment-reset, automatic
domain routing -- see that function's own docstring for the full design)
so app/routers/voice.py can answer a turn with the exact same grounding,
refusal, and memory semantics as text chat, without re-deriving any of
that logic here.

Deliberately does NOT extract or modify chat.py. An earlier version of
this plan called for turning chat()'s body into a shared `run_turn()` that
chat.py itself would call -- the safer choice turned out to be reuse BY
IMPORT instead: chat.py's turn logic is covered by tests/test_chat.py
(419-test suite, python's own convention: 30 tests in that file alone),
and this machine currently has no reachable Postgres to run that suite
against (see docs/architecture/cloud-scaling-plan.md -- laptop MVP has no
local Postgres; the test suite genuinely needs one to pass, several of its
paths hit a real connection attempt rather than mocking it). Editing
chat.py's body without being able to verify the result against its own
regression suite was judged not worth the risk. This module instead
imports `_resolve_turn_context` directly from app.routers.chat -- Python
does not enforce module-privacy on an underscore-prefixed name, and this
is the one place outside chat.py itself that calls it, so chat.py's
tested behavior is reused verbatim, not duplicated. If a future change
needs to touch that function, do it with a reachable Postgres and a green
tests/test_chat.py, then this module needs no changes at all.

Diagram intent (chat.py step 2b) is deliberately NOT reproduced here --
diagrams have no audio rendering, and detecting diagram intent in speech
is out of scope for the voice MVP (docs/architecture/voice-assistant.md).
A voice turn that would have triggered a diagram in text chat is answered
as an ordinary grounded turn instead.
"""
import logging
import uuid
from dataclasses import dataclass
from typing import Optional

from app.services import history
from app.services import sources as source_service
from app.services.retrieval import _fingerprint
from app.services.routing import resolve_language
from app.services.llm import deterministic_refusal

logger = logging.getLogger(__name__)


@dataclass
class TurnContext:
    """Everything resolved about one turn before the model is called --
    the voice-pipeline analogue of chat.py's local variables between its
    steps 1-2 and step 5. Mutating this after resolve_turn() returns it is
    not supported; persist_turn() reads it as a snapshot."""

    session_id: str
    tenant_id: str
    user_id: str
    message: str
    domain: str
    domain_source: str
    query_lang: str
    response_lang: str
    context: str
    sources: list[str]
    segment_id: int
    is_new_pin: bool
    degraded: bool
    corpus_version: Optional[str]
    override_to_persist: Optional[str]
    override_query_lang_to_persist: Optional[str]

    @property
    def is_refusal(self) -> bool:
        """Same two-trigger rule as chat.py step 3 -- see that comment for
        why domain_source == 'no_match' is checked independently of
        `context` being empty."""
        return self.domain_source == "no_match" or not self.context.strip()


def resolve_turn(
    message: str,
    *,
    tenant_id: str,
    user_id: str,
    session_id: Optional[str] = None,
    requested_domain: Optional[str] = None,
    explicit_language: Optional[str] = None,
    active_source_ids_hint: Optional[list[str]] = None,
) -> TurnContext:
    """Resolve language, domain, retrieval context, segment, and pin state
    for one turn -- mirrors app.routers.chat.chat()'s steps 1-2 exactly,
    reusing that module's own _resolve_turn_context (see this module's
    docstring). Does not call the model and does not persist anything;
    call persist_turn() after the caller has produced (or refused) an
    answer.
    """
    from app.routers.chat import _resolve_turn_context

    session_id = session_id or uuid.uuid4().hex

    stored_override, stored_override_query_lang = history.get_language_state(session_id)
    lang = resolve_language(
        message,
        explicit_language=explicit_language,
        stored_override=stored_override,
        stored_override_query_lang=stored_override_query_lang,
    )
    query_lang = lang.query_lang
    response_lang = lang.response_lang

    active_source_ids, corpus_version_now = source_service.active_sources_and_version(
        tenant_id, requested=active_source_ids_hint
    )

    (
        context, sources, segment_id, is_new_pin, domain, domain_source,
        degraded, corpus_version,
    ) = _resolve_turn_context(
        session_id, message, requested_domain, query_lang, tenant_id,
        source_ids=active_source_ids, corpus_version=corpus_version_now,
    )

    return TurnContext(
        session_id=session_id, tenant_id=tenant_id, user_id=user_id, message=message,
        domain=domain, domain_source=domain_source, query_lang=query_lang,
        response_lang=response_lang, context=context, sources=sources,
        segment_id=segment_id, is_new_pin=is_new_pin, degraded=degraded,
        corpus_version=corpus_version,
        override_to_persist=lang.override_to_persist,
        override_query_lang_to_persist=lang.override_query_lang_to_persist,
    )


def refusal_text(turn: TurnContext) -> str:
    """Same deterministic refusal chat.py step 3 returns -- see
    app.services.llm.deterministic_refusal. Never call the model for this;
    that is the entire point of the refusal gate (chat.py's comment on why:
    the fine-tune's own refusal register is welded to tenant #1's domain
    regardless of the actual tenant)."""
    return deterministic_refusal(turn.domain, turn.response_lang)


def load_prior_turns(turn: TurnContext) -> list[dict]:
    """chat.py step 4 -- prior alternating (user, assistant) turns for this
    exact (session, domain, query_lang, segment). Fail-open: [] on any
    history failure, identical contract to history.load_window itself."""
    return history.load_window(
        turn.session_id, domain=turn.domain, language=turn.query_lang, segment_id=turn.segment_id,
    )


def persist_turn(turn: TurnContext, *, assistant_content: str) -> None:
    """chat.py steps 6-7 (minus prior_questions, which chat.py's response
    schema surfaces for the UI and voice has no equivalent use for yet).
    Fail-open, same contract as the history functions it calls -- a
    storage failure here must not turn a good spoken answer into an error
    event.

    `assistant_content` is the FULL text actually delivered to the user --
    for a voice turn interrupted by barge-in, this must be only the prefix
    that was actually spoken/heard before the interrupt, not the full
    generated answer (app/routers/voice.py's responsibility to track and
    pass correctly; see that module for why: the model must never be told
    it said something the user never heard).
    """
    history.append_exchange(
        turn.session_id,
        tenant_id=turn.tenant_id,
        user_id=turn.user_id,
        domain=turn.domain,
        language=turn.query_lang,
        segment_id=turn.segment_id,
        user_content=turn.message,
        assistant_content=assistant_content,
        sources=turn.sources,
        response_lang_override=turn.override_to_persist,
        override_query_lang=turn.override_query_lang_to_persist,
    )
    if turn.is_new_pin:
        history.pin_context(
            turn.session_id,
            tenant_id=turn.tenant_id,
            user_id=turn.user_id,
            domain=turn.domain,
            language=turn.query_lang,
            segment_id=turn.segment_id,
            context=turn.context,
            sources=turn.sources,
            fingerprint=_fingerprint(domain=turn.domain, ui_lang=turn.query_lang, query=turn.message),
            corpus_version=turn.corpus_version,
        )
