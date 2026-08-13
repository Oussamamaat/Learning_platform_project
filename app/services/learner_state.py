"""
Learner State — Read-Only Seam
────────────────────────────────
Wires a prompt-assembly injection point for "what this learner knows"
without building the thing that answers it.

ADR 0003 already owns that answer: a `quiz_attempts` table (user_id,
question_id, tenant_id, answered_at, correct, ability_after) plus an
IRT/Rasch ability estimate, used to pick the next quiz item and exclude
recently-seen questions (spaced repetition). Chat has no business
duplicating that store or computing a second, divergent notion of mastery
from conversation turns alone.

So: one function, returns {} today, chat calls it and injects nothing.
When ADR 0003's tables land, this function starts reading them and the
call site in app/routers/chat.py does not change shape. Chat never writes
learner state -- one owner, no duplication.
"""


def get_learner_state(user_id: str, tenant_id: str, domain: str) -> dict:
    """What this learner is known to have mastered or struggled with in
    `domain`, sourced from ADR 0003's quiz_attempts + ability estimate once
    that exists. Returns {} until then -- an empty dict must be treated by
    every caller as "nothing known yet", not as an error.
    """
    return {}
