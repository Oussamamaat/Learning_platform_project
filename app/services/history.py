"""
Conversation History
─────────────────────
Server-side (Postgres) storage for multi-turn chat, and the window builder
that turns stored rows into the `messages` list app.services.llm sends.

Fail-open by design: chat has zero Postgres dependency today
(app.services.domain_context reads from disk), so making history storage a
hard dependency of chat would be a demo-reliability regression. Every
public function here swallows and logs its own exceptions rather than
raising -- a Postgres hiccup degrades chat to exactly today's stateless
behaviour, it never breaks the response.

Filtering is strict on (domain, language, segment_id): a turn from a
different domain, a different ui_lang, or a closed-out topic segment is
never replayed into a new prompt. Replaying a Darija transcript into a
French request would reintroduce the exact context-script contamination
ADR 0002 decision 5 exists to prevent at the retrieval layer; the same
rule applies to history.
"""

import logging
import re
import uuid
from typing import Optional

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models.database import ChatMessage, ChatSession
from app.services.generate_training_data import _QUESTION_MARK

logger = logging.getLogger(__name__)

# 2 prior exchanges (4 messages: user, assistant, user, assistant) -- the
# byte-exact trained shape is 1 exchange; the token budget has room for far
# more (~11-14 exchanges), but the model has never seen more than one prior
# exchange in training (generate_training_data.py's multi-turn prompts ask
# for "exactly 4 messages"). Validate at the live dry-run that a 3rd
# assistant turn still stops cleanly and stays in-language; drop to 1
# exchange (MAX_WINDOW_MESSAGES = 2) if not.
MAX_WINDOW_MESSAGES = 4
MAX_WINDOW_CHARS = 2400  # ~800 tokens, per the token budget in the memory/RAG plan

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        # connect_timeout matters for the fail-open contract itself, not
        # just test speed: without it, an unreachable Postgres (a dead
        # container mid-demo, not just "not started") can hang on the
        # platform's default TCP timeout -- observed multiple seconds on
        # this machine's IPv6 (::1) attempt before falling back to IPv4 --
        # turning every chat request slow instead of gracefully stateless.
        _engine = create_engine(
            get_settings().database_url,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 2},
        )
    return _engine


def _build_alternating_window(rows: list[ChatMessage], *, max_messages: int, max_chars: int) -> list[dict]:
    """Slice stored rows (already in chronological order) into a window
    the shipped chat template will accept: strictly alternating roles,
    starting on 'user', ending on 'assistant'.

    Defensive rather than load-bearing -- app.services.history.append_exchange
    writes each (user, assistant) pair in one transaction, so a partial
    pair should not normally reach storage. Kept because the shipped Jinja
    template raises `'Conversation roles must alternate ...'` on the first
    violation, and a crash here is worse than a slightly short window.
    """
    rows = list(rows)

    # Drop a dangling trailing 'user' with no reply, and a dangling leading
    # 'assistant' with no preceding user turn.
    while rows and rows[-1].role != "assistant":
        rows.pop()
    while rows and rows[0].role != "user":
        rows.pop(0)

    if len(rows) > max_messages:
        rows = rows[-max_messages:]
        if rows and rows[0].role != "user":
            rows = rows[1:]

    total = sum(len(r.content) for r in rows)
    while total > max_chars and len(rows) >= 2:
        removed = rows[:2]
        rows = rows[2:]
        total -= sum(len(m.content) for m in removed)

    return [{"role": r.role, "content": r.content} for r in rows]


def load_window(
    session_id: str,
    *,
    domain: str,
    language: str,
    segment_id: int,
    max_messages: int = MAX_WINDOW_MESSAGES,
    max_chars: int = MAX_WINDOW_CHARS,
) -> list[dict]:
    """Load the most recent alternating (user, assistant) window for this
    session, filtered to (domain, language, segment_id).

    Returns [] on any failure (no session, no Postgres, a query error) --
    callers must treat an empty window as "no history", identical to a
    brand new session, never as an error to surface.
    """
    try:
        with Session(_get_engine()) as db:
            rows = (
                db.execute(
                    select(ChatMessage)
                    .where(
                        ChatMessage.session_id == session_id,
                        ChatMessage.domain == domain,
                        ChatMessage.language == language,
                        ChatMessage.segment_id == segment_id,
                    )
                    .order_by(ChatMessage.turn_index.desc())
                    .limit(max_messages * 3)
                )
                .scalars()
                .all()
            )
    except Exception:
        logger.exception(
            "history.load_window failed for session_id=%s; falling back to stateless", session_id
        )
        return []

    rows = list(reversed(rows))
    return _build_alternating_window(rows, max_messages=max_messages, max_chars=max_chars)


def get_language_state(session_id: str) -> tuple[Optional[str], Optional[str]]:
    """Return (response_lang_override, override_query_lang) for the sticky
    in-message language override (app.services.routing.resolve_language).

    Deliberately independent of get_pinned()/the segment mechanism: a
    language override persists ACROSS a domain/topic segment reset (a
    "réponds en darija" instruction is about the answer's language, not
    which documents ground it), whereas pinned_context is scoped to one
    topic segment and clears on a reset. Coupling the two would mean a
    plain topic shift silently drops an active override the user never
    asked to clear.

    (None, None) on a brand-new session or any failure -- fail-open, same
    contract as every other function here.
    """
    try:
        with Session(_get_engine()) as db:
            session = db.get(ChatSession, session_id)
            if session is None:
                return None, None
            return session.response_lang_override, session.override_query_lang
    except Exception:
        logger.exception("history.get_language_state failed for session_id=%s", session_id)
        return None, None


def append_exchange(
    session_id: str,
    *,
    tenant_id: str,
    user_id: str,
    domain: str,
    language: str,
    segment_id: int,
    user_content: str,
    assistant_content: str,
    sources: Optional[list] = None,
    response_lang_override: Optional[str] = None,
    override_query_lang: Optional[str] = None,
) -> None:
    """Persist one (user, assistant) exchange, creating the session row if
    this is its first turn. Fail-open: logs and swallows any exception.

    `response_lang_override`/`override_query_lang` are written here (not a
    separate call) because this runs unconditionally on every successful
    turn, unlike pin_context() which only runs on a new segment -- the
    natural "every turn" persistence point for resolve_language()'s
    stickiness state. Passing None for both clears a previously stored
    override, matching resolve_language()'s own case-3 clearing behaviour.
    """
    try:
        with Session(_get_engine()) as db:
            session = db.get(ChatSession, session_id)
            if session is None:
                session = ChatSession(
                    session_id=session_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    domain=domain,
                    language=language,
                    segment_id=segment_id,
                    response_lang_override=response_lang_override,
                    override_query_lang=override_query_lang,
                )
                db.add(session)
                db.flush()
            else:
                session.domain = domain
                session.language = language
                session.segment_id = segment_id
                session.response_lang_override = response_lang_override
                session.override_query_lang = override_query_lang

            next_turn = db.execute(
                select(func.coalesce(func.max(ChatMessage.turn_index), 0)).where(
                    ChatMessage.session_id == session_id
                )
            ).scalar_one()

            db.add(
                ChatMessage(
                    id=uuid.uuid4(),
                    session_id=session_id,
                    turn_index=next_turn + 1,
                    segment_id=segment_id,
                    role="user",
                    content=user_content,
                    domain=domain,
                    language=language,
                )
            )
            db.add(
                ChatMessage(
                    id=uuid.uuid4(),
                    session_id=session_id,
                    turn_index=next_turn + 2,
                    segment_id=segment_id,
                    role="assistant",
                    content=assistant_content,
                    domain=domain,
                    language=language,
                    sources=sources or [],
                )
            )
            db.commit()
    except Exception:
        logger.exception("history.append_exchange failed for session_id=%s", session_id)


def pin_context(
    session_id: str,
    *,
    tenant_id: str,
    user_id: str,
    domain: str,
    language: str,
    segment_id: int,
    context: str,
    sources: list,
    fingerprint: str,
    corpus_version: Optional[str] = None,
) -> None:
    """Store the retrieved context for the current segment so a follow-up
    turn can reuse it without re-retrieving (app/services/retrieval.py,
    Stage 3). Fail-open, same contract as append_exchange.

    `corpus_version` (app.services.sources.corpus_version, computed at the
    moment this pin is written) is the pin-invalidation signal
    app.routers.chat._resolve_turn_context compares against on later
    turns -- a mismatch means a tenant source was uploaded/toggled/deleted
    since. Default None for any caller that hasn't threaded it through
    (there are none left, but keeps this function callable the same way
    tests may have exercised it before uploads existed).
    """
    try:
        with Session(_get_engine()) as db:
            session = db.get(ChatSession, session_id)
            if session is None:
                session = ChatSession(
                    session_id=session_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    domain=domain,
                    language=language,
                    segment_id=segment_id,
                )
                db.add(session)
            session.domain = domain
            session.language = language
            session.segment_id = segment_id
            session.pinned_context = context
            session.pinned_sources = sources
            session.pinned_fingerprint = fingerprint
            session.pinned_corpus_version = corpus_version
            db.commit()
    except Exception:
        logger.exception("history.pin_context failed for session_id=%s", session_id)


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?؟])\s+")


def extract_dropped_questions(
    session_id: str,
    *,
    domain: str,
    language: str,
    segment_id: int,
    kept_messages: int = MAX_WINDOW_MESSAGES,
) -> list[str]:
    """Socratic questions the tutor already asked, from assistant turns
    that fell out of the replay window.

    Deterministic, not a summary: hints already given and the tutor's own
    prior questions ARE the sentences in those turns, so this reuses
    generate_training_data's own question-boundary regex rather than
    asking a model to re-describe them. Never fed back into the prompt --
    that would either exceed the trained turn-count shape or, if placed in
    the retrieval context, dress model-generated text up as source
    material. Callers surface this on the API response for the UI only.

    Fail-open like every other function here: returns [] on any error.
    """
    try:
        with Session(_get_engine()) as db:
            rows = (
                db.execute(
                    select(ChatMessage)
                    .where(
                        ChatMessage.session_id == session_id,
                        ChatMessage.domain == domain,
                        ChatMessage.language == language,
                        ChatMessage.segment_id == segment_id,
                        ChatMessage.role == "assistant",
                    )
                    .order_by(ChatMessage.turn_index.asc())
                )
                .scalars()
                .all()
            )
    except Exception:
        logger.exception(
            "history.extract_dropped_questions failed for session_id=%s", session_id
        )
        return []

    # The window replays the most recent `kept_messages // 2` assistant
    # turns verbatim; only earlier ones are "dropped" and need their
    # questions surfaced separately.
    kept_assistant_count = kept_messages // 2
    dropped = rows[:-kept_assistant_count] if kept_assistant_count else rows

    questions = []
    for row in dropped:
        for sentence in _SENTENCE_SPLIT.split(row.content):
            sentence = sentence.strip()
            if sentence and _QUESTION_MARK.search(sentence):
                questions.append(sentence)
    return questions


def get_pinned(session_id: str) -> Optional[dict]:
    """Return the current segment's pinned context, or None on any failure
    or if the session has never pinned anything (fail-open -- caller falls
    back to a fresh retrieval).
    """
    try:
        with Session(_get_engine()) as db:
            session = db.get(ChatSession, session_id)
            if session is None or session.pinned_context is None:
                return None
            return {
                "context": session.pinned_context,
                "sources": session.pinned_sources or [],
                "fingerprint": session.pinned_fingerprint,
                "segment_id": session.segment_id,
                "domain": session.domain,
                "language": session.language,
                "corpus_version": session.pinned_corpus_version,
            }
    except Exception:
        logger.exception("history.get_pinned failed for session_id=%s", session_id)
        return None
