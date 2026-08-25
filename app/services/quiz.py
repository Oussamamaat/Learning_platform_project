"""
Quiz Generation Service
────────────────────────
Generates quiz questions from retrieved RAG context via the same fine-tuned
model and system prompt chat serving uses (`_build_system_prompt` in
`app/services/llm.py`), and the same user-turn phrasing quiz training rows
were built with (`QUIZ_USER_FALLBACKS(_FR)` in `generate_training_data.py`).
Reusing that exact shape keeps the request in-distribution for the
fine-tuned model, rather than inventing a new prompt format it was never
trained on.

Fabrication is filtered downstream by `app/services/grounding.py`, not
prevented here — the JSON `format` schema below only constrains structure
(4 options, an answer index, etc.), not truthfulness.
"""

import json
import logging
import random

from app.config import get_settings
from app.errors import OllamaConnectionError, GenerationError
from app.services.llm import _build_system_prompt, _call_ollama_generate
from app.services.generate_training_data import (
    QUIZ_USER_FALLBACKS,
    QUIZ_USER_FALLBACKS_FR,
)

logger = logging.getLogger(__name__)

# Mirrors the question-object shape in generate_training_data.QUIZ_CONTENT_SCHEMA
# (kept as a separate copy, not imported, because serving needs a different
# top-level "required" -- just "questions", never "request", since the user
# turn is fixed here rather than generated). Keep the per-question shape in
# sync with QUIZ_CONTENT_SCHEMA if that one changes.
_QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string", "minLength": 10},
        "options": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {"type": "string", "minLength": 1},
        },
        "answer": {"type": "integer", "minimum": 0, "maximum": 3},
        "explanation": {"type": "string", "minLength": 10},
    },
    "required": ["question", "options", "answer", "explanation"],
}


def _quiz_format_schema(max_questions: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "minItems": 1,
                "maxItems": max(1, max_questions),
                "items": _QUESTION_SCHEMA,
            }
        },
        "required": ["questions"],
    }


def generate_quiz_questions(
    topic: str,
    context: str,
    domain: str = "industrial",
    language: str = "darija",
    n: int = 5,
    avoid_questions: list | None = None,
) -> list:
    """Query the fine-tuned model for quiz questions grounded in `context`.

    Returns raw (unvalidated, unfiltered) question dicts -- callers must run
    them through app.services.grounding before returning to an API client.

    `avoid_questions`, when given, is a list of question strings already
    obtained (e.g. from an earlier call whose output the grounding filter
    partly rejected) -- passed back to the model so a top-up call fills the
    gap with new material instead of paraphrasing what was already kept.
    """
    settings = get_settings()

    system_prompt = _build_system_prompt(domain, context, language)
    user_turn = random.choice(
        QUIZ_USER_FALLBACKS_FR if language == "fr" else QUIZ_USER_FALLBACKS
    )
    # The fallback turns above never state a count, and the adapter was
    # fine-tuned exclusively on 3-question quiz exemplars (see
    # build_quiz_prompt in generate_training_data.py) -- without this, the
    # model just emits ~3 questions regardless of `n`, no matter how high
    # `format`'s maxItems is set. This is a mitigation, not a fix: the count
    # is out-of-distribution for the adapter, so reliability past ~4-5 still
    # needs a retrain on variable-count quiz rows.
    count_hint = (
        f" Fais-moi exactement {n} questions."
        if language == "fr"
        else f" عطيني {n} ديال الأسئلة، ماشي غير شي وحدة أو جوج."
    )
    user_turn = user_turn + count_hint

    if avoid_questions:
        avoid_list = "\n".join(f"- {q}" for q in avoid_questions)
        avoid_hint = (
            f"\nNe repete pas ces questions deja posees:\n{avoid_list}"
            if language == "fr"
            else f"\nما تكررش هاد الأسئلة اللي سبق طرحهم:\n{avoid_list}"
        )
        user_turn = user_turn + avoid_hint

    # Same French/Darija model split llm.py:534 and demo.py:70 use -- this
    # branch was previously missing here, so every French-language quiz was
    # silently served by the Darija-tuned model instead of iblog-tutor-fr.
    model = settings.ollama_model_fr if language == "fr" else settings.ollama_model

    logger.info(
        "Calling Ollama for quiz model=%s domain=%s topic=%s",
        model, domain, topic,
    )
    # Through app.services.llm._call_ollama_generate rather than this
    # module's own urllib block, which was a near-copy of it. The copy had
    # drifted in a way that mattered: it sent `options={"temperature": 0.2}`
    # with NO num_ctx, so every quiz ran at each Modelfile's 4096 default
    # while chat ran at 8192. Ollama truncates from the FRONT when the
    # window is exceeded -- so a quiz built on a full 6000-character
    # retrieved context (app/services/retrieval.py's max_context_length)
    # was having that context silently eaten before generation, which is
    # precisely the failure llm.py's num_ctx comment exists to prevent.
    # Sharing the transport also gives quiz the keep_alive, the bounded
    # retry on transient failures, and the HTTPError-vs-URLError split
    # (a 404 "no such model" no longer reports as a connection failure).
    raw = _call_ollama_generate(
        model, user_turn, system_prompt, format_schema=_quiz_format_schema(n)
    )

    try:
        payload_obj = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON from Ollama (quiz): %s", e)
        raise GenerationError(f"Invalid JSON response: {e}") from e

    questions = payload_obj.get("questions")
    if not isinstance(questions, list):
        raise GenerationError("Quiz response missing a questions array")
    return questions[:n]
