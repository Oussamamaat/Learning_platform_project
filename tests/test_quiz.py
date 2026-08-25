"""
Regression tests for the quiz generation route and its grounding backstop.

First tests for app/routers/quiz.py (previously a pure placeholder stub)
and app/services/quiz.py. Written alongside the fix for a live-reproduced
defect: 2-3 of 7 generated quiz questions invented facts (e.g. an article
number) absent from the retrieved context. No Ollama or Postgres required --
build_rag_context and urllib.request.urlopen are both monkeypatched.
The urlopen seam is app.services.llm's, not app.services.quiz's: quiz
generation now goes through the shared llm._call_ollama_generate
transport instead of its own near-duplicate urllib block (which had
drifted -- it omitted num_ctx, so quizzes silently ran at 4096).
"""

import json
from unittest.mock import patch

from app.models.schemas import QuizRequest
from app.routers.quiz import generate_quiz


CONTEXT = "Selon Article 15, le port du casque est obligatoire sur le chantier."


def _context(*args, **kwargs):
    return CONTEXT, ["doc1.pdf"]


def _empty_context(*args, **kwargs):
    return "", []


def _fails_if_called(*args, **kwargs):
    raise AssertionError("Ollama must not be called when context is empty")


class _FakeResponse:
    def __init__(self, body):
        self._body = body

    def read(self):
        return json.dumps(self._body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_ollama(questions):
    return _FakeResponse({"response": json.dumps({"questions": questions})})


GROUNDED_Q = {
    "question": "Que dit Article 15 ?",
    "options": ["casque", "gants", "bottes", "masque"],
    "answer": 0,
    "explanation": "Article 15 mentionne le casque obligatoire",
}

FABRICATED_Q = {
    "question": "Que dit Article 99 ?",
    "options": ["casque", "gants", "bottes", "masque"],
    "answer": 0,
    "explanation": "Article 99 mentionne le casque obligatoire",
}


def test_empty_context_returns_refusal_without_calling_ollama():
    with patch("app.routers.quiz.build_rag_context", side_effect=_empty_context), \
         patch("app.services.llm.urllib.request.urlopen", side_effect=_fails_if_called):
        response = generate_quiz(QuizRequest(topic="hors sujet", num_questions=7))
        assert response.total_questions == 0
        assert response.questions == []
        assert response.message
        assert response.requested_questions == 7


def test_fabricated_question_filtered_grounded_question_kept():
    with patch("app.routers.quiz.build_rag_context", side_effect=_context), \
         patch("app.services.llm.urllib.request.urlopen",
               return_value=_fake_ollama([GROUNDED_Q, FABRICATED_Q])):
        response = generate_quiz(QuizRequest(topic="securite chantier"))
        assert response.total_questions == 1
        assert "Article 15" in response.questions[0].question
        assert response.message is None


def test_all_fabricated_returns_zero_questions_and_message():
    with patch("app.routers.quiz.build_rag_context", side_effect=_context), \
         patch("app.services.llm.urllib.request.urlopen",
               return_value=_fake_ollama([FABRICATED_Q])):
        response = generate_quiz(QuizRequest(topic="securite chantier"))
        assert response.total_questions == 0
        assert response.questions == []
        assert response.message
        assert response.requested_questions == 5


def test_malformed_payload_dropped_duplicate_options():
    malformed = {
        "question": "Que dit Article 15 ?",
        "options": ["casque", "casque", "bottes", "masque"],
        "answer": 0,
        "explanation": "Article 15 mentionne le casque",
    }
    with patch("app.routers.quiz.build_rag_context", side_effect=_context), \
         patch("app.services.llm.urllib.request.urlopen",
               return_value=_fake_ollama([malformed, GROUNDED_Q])):
        response = generate_quiz(QuizRequest(topic="securite chantier"))
        assert response.total_questions == 1
        assert "Article 15" in response.questions[0].question


def test_malformed_payload_dropped_bad_answer_index():
    malformed = {
        "question": "Que dit Article 15 ?",
        "options": ["casque", "gants", "bottes", "masque"],
        "answer": 7,
        "explanation": "Article 15 mentionne le casque",
    }
    with patch("app.routers.quiz.build_rag_context", side_effect=_context), \
         patch("app.services.llm.urllib.request.urlopen",
               return_value=_fake_ollama([malformed, GROUNDED_Q])):
        response = generate_quiz(QuizRequest(topic="securite chantier"))
        assert response.total_questions == 1


def test_grounded_response_carries_sources():
    with patch("app.routers.quiz.build_rag_context", side_effect=_context), \
         patch("app.services.llm.urllib.request.urlopen",
               return_value=_fake_ollama([GROUNDED_Q])):
        response = generate_quiz(QuizRequest(topic="securite chantier"))
        assert response.sources == ["doc1.pdf"]


# --- Automatic Domain Routing / language resolution (2026-08-11) ----------

def test_explicit_domain_is_page_context_and_skips_router():
    from app.models.schemas import Domain
    with patch("app.routers.quiz.build_rag_context", side_effect=_context), \
         patch("app.routers.quiz.resolve_domain") as rd, \
         patch("app.services.llm.urllib.request.urlopen",
               return_value=_fake_ollama([GROUNDED_Q])):
        response = generate_quiz(QuizRequest(topic="securite chantier", domain=Domain.SECURITE))
    rd.assert_not_called()
    assert response.domain == "securite"
    assert response.domain_source == "page_context"


def test_omitted_domain_autoroutes_and_reports_it():
    with patch("app.routers.quiz.resolve_domain", return_value=("blockchain", "retrieval")) as rd, \
         patch("app.routers.quiz.build_rag_context", side_effect=_context), \
         patch("app.services.llm.urllib.request.urlopen",
               return_value=_fake_ollama([GROUNDED_Q])):
        response = generate_quiz(QuizRequest(topic="smart contracts"))
    rd.assert_called_once()
    assert response.domain == "blockchain"
    assert response.domain_source == "retrieval"


def test_out_of_corpus_topic_refuses_even_when_context_is_nonempty():
    """Quiz's half of the out-of-domain refusal fix. Context is deliberately
    NON-empty, so the only possible trigger is domain_source == "no_match"
    -- without it, an off-topic quiz topic routes to the tenant default
    domain, pulls its nearest-but-irrelevant chunks, and generates questions
    "grounded" in unrelated material."""
    with patch("app.routers.quiz.resolve_domain", return_value=("industrial", "no_match")), \
         patch("app.routers.quiz.build_rag_context", side_effect=_context), \
         patch("app.services.llm.urllib.request.urlopen", side_effect=_fails_if_called):
        response = generate_quiz(QuizRequest(topic="sourdough bread baking"))
    assert response.domain_source == "no_match"
    assert response.questions == []
    assert response.total_questions == 0
    assert response.message  # a real refusal, not an empty string


def test_response_language_field_reflects_resolution():
    with patch("app.routers.quiz.build_rag_context", side_effect=_context), \
         patch("app.services.llm.urllib.request.urlopen",
               return_value=_fake_ollama([GROUNDED_Q])):
        response = generate_quiz(QuizRequest(topic="securite chantier"))
        assert response.language in ("fr", "darija")


# --- Top-up retry on a grounding-filter shortfall (2026-08-13) ------------

GROUNDED_Q2 = {
    "question": "Ou le port du casque est-il obligatoire selon Article 15 ?",
    "options": ["sur le chantier", "au bureau", "a la maison", "dans le parking"],
    "answer": 0,
    "explanation": "Article 15 precise que le casque est obligatoire sur le chantier",
}


def test_shortfall_triggers_topup_call_to_reach_requested_count():
    # First call: 1 grounded + 1 fabricated (kept=1, short of num_questions=2).
    # Second call (the top-up): a second, distinct grounded question, which
    # should be merged in to reach the full requested count.
    with patch("app.routers.quiz.build_rag_context", side_effect=_context), \
         patch("app.services.llm.urllib.request.urlopen",
               side_effect=[
                   _fake_ollama([GROUNDED_Q, FABRICATED_Q]),
                   _fake_ollama([GROUNDED_Q2]),
               ]):
        response = generate_quiz(
            QuizRequest(topic="securite chantier", num_questions=2)
        )
        assert response.total_questions == 2
        questions = {q.question for q in response.questions}
        assert GROUNDED_Q["question"] in questions
        assert GROUNDED_Q2["question"] in questions


def test_topup_gives_up_after_max_rounds_without_erroring():
    # Model repeats the exact same fabricated question every round -- the
    # loop must not hang or error, just return what it managed to keep (0).
    with patch("app.routers.quiz.build_rag_context", side_effect=_context), \
         patch("app.services.llm.urllib.request.urlopen",
               return_value=_fake_ollama([FABRICATED_Q])):
        response = generate_quiz(
            QuizRequest(topic="securite chantier", num_questions=5)
        )
        assert response.total_questions == 0
        assert response.message


def test_topup_deduplicates_repeated_question_across_rounds():
    # Every round returns the SAME grounded question -- the dedup-by-text
    # merge must not count it twice, so with num_questions=3 the loop should
    # exhaust its retry budget rather than silently double/triple-counting.
    with patch("app.routers.quiz.build_rag_context", side_effect=_context), \
         patch("app.services.llm.urllib.request.urlopen",
               return_value=_fake_ollama([GROUNDED_Q])):
        response = generate_quiz(
            QuizRequest(topic="securite chantier", num_questions=3)
        )
        assert response.total_questions == 1


def test_response_reports_requested_vs_delivered_shortfall():
    # Source material only supports 1 distinct grounded question -- the
    # response must say so explicitly via requested_questions rather than
    # silently returning fewer than asked.
    with patch("app.routers.quiz.build_rag_context", side_effect=_context), \
         patch("app.services.llm.urllib.request.urlopen",
               return_value=_fake_ollama([GROUNDED_Q])):
        response = generate_quiz(
            QuizRequest(topic="securite chantier", num_questions=6)
        )
        assert response.requested_questions == 6
        assert response.total_questions == 1
        assert response.total_questions < response.requested_questions
