"""
Quiz Grounding Verifier
───────────────────────
Deterministic, zero-retrain backstop against fabricated quiz questions.

Thin serving-facing wrapper: imports the fabrication detectors the
generation pipeline already relies on (`generate_training_data.py`) rather
than restating them, so serving and training data are held to the identical
standard instead of a second, independently-drifting set of checks. See
`_explanation_supports_answer` and `row_has_ungrounded_reference` there for
why these specific checks and not a bag-of-words overlap score: legal
references are the observed fabrication shape (e.g. المادة 15/16 guard
limits invented outside the retrieved chunk), and a generic keyword-overlap
gate would risk rejecting a correct question over Arabic morphology alone.

Import direction is one-way (this module depends on generate_training_data,
never the reverse) to avoid a circular import: `validate_quiz_question` is
defined in generate_training_data.py itself, next to `build_quiz_row`, which
calls it directly.
"""

import json

from app.services.generate_training_data import (
    row_has_ungrounded_reference,
    validate_quiz_question,
)

__all__ = [
    "validate_quiz_question",
    "question_is_grounded",
    "filter_grounded_questions",
]


def question_is_grounded(q: dict, context: str) -> tuple[bool, list[str]]:
    """True (with no offenders) when every reference in `q` appears in
    `context`. Wraps `row_has_ungrounded_reference`, which already has a
    quiz-aware branch: it scans the question, the explanation, and only the
    CORRECT option -- a distractor is supposed to be a plausible wrong
    answer, not a claim about the source, so it is excluded deliberately
    (same reasoning `NUMERIC_GROUNDED_COMPONENTS` excludes quiz_generation
    from the numeric-fabrication gate entirely: distractors legitimately
    carry plausible-but-wrong numbers, so that gate does not belong here,
    and this wrapper deliberately does not call it).
    """
    row = {
        "component": "quiz_generation",
        "messages": [
            {
                "role": "assistant",
                "content": json.dumps({"questions": [q]}, ensure_ascii=False),
            }
        ],
    }
    offenders = row_has_ungrounded_reference(row, context)
    return (not offenders, offenders)


def filter_grounded_questions(
    questions: list, context: str
) -> tuple[list[dict], list[dict]]:
    """Split `questions` into (kept, dropped). `dropped` entries carry their
    rejection reason under "_reject_reason" for logging -- never returned to
    the API caller.
    """
    kept: list[dict] = []
    dropped: list[dict] = []

    for q in questions:
        if not validate_quiz_question(q):
            dropped.append({"question": q, "_reject_reason": "invalid_structure"})
            continue
        grounded, offenders = question_is_grounded(q, context)
        if not grounded:
            dropped.append({
                "question": q,
                "_reject_reason": "ungrounded_reference",
                "_offenders": offenders,
            })
            continue
        kept.append(q)

    return kept, dropped
