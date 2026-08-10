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
import urllib.request
import urllib.error

from app.config import get_settings
from app.errors import OllamaConnectionError, GenerationError
from app.services.llm import _build_system_prompt
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
) -> list:
    """Query the fine-tuned model for quiz questions grounded in `context`.

    Returns raw (unvalidated, unfiltered) question dicts -- callers must run
    them through app.services.grounding before returning to an API client.
    """
    settings = get_settings()

    system_prompt = _build_system_prompt(domain, context, language)
    user_turn = random.choice(
        QUIZ_USER_FALLBACKS_FR if language == "fr" else QUIZ_USER_FALLBACKS
    )

    url = f"{settings.ollama_base_url.rstrip('/')}/api/generate"
    payload = {
        "model": settings.ollama_model,
        "prompt": user_turn,
        "system": system_prompt,
        "stream": False,
        "format": _quiz_format_schema(n),
        "options": {"temperature": 0.2},
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        logger.info(
            "Calling Ollama for quiz model=%s domain=%s topic=%s",
            settings.ollama_model, domain, topic,
        )
        with urllib.request.urlopen(req, timeout=180) as response:
            res_body = response.read().decode("utf-8")
            res_json = json.loads(res_body)
            raw = res_json.get("response", "").strip()
            if not raw:
                raise GenerationError("Ollama returned empty response")
            payload_obj = json.loads(raw)
            questions = payload_obj.get("questions")
            if not isinstance(questions, list):
                raise GenerationError("Quiz response missing a questions array")
            return questions[:n]
    except urllib.error.URLError as e:
        logger.error("Ollama connection failed (quiz): %s", e)
        raise OllamaConnectionError(settings.ollama_model, settings.ollama_base_url) from e
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON from Ollama (quiz): %s", e)
        raise GenerationError(f"Invalid JSON response: {e}") from e
    except (OllamaConnectionError, GenerationError):
        raise
    except Exception as e:
        logger.error("Unexpected LLM error (quiz): %s", e)
        raise GenerationError(str(e)) from e
