"""
Regression tests for the quiz generation route and its grounding backstop.

First tests for app/routers/quiz.py (previously a pure placeholder stub)
and app/services/quiz.py. Written alongside the fix for a live-reproduced
defect: 2-3 of 7 generated quiz questions invented facts (e.g. an article
number) absent from the retrieved context. No Ollama or Postgres required --
build_rag_context and urllib.request.urlopen are both monkeypatched.
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
         patch("app.services.quiz.urllib.request.urlopen", side_effect=_fails_if_called):
        response = generate_quiz(QuizRequest(topic="hors sujet"))
        assert response.total_questions == 0
        assert response.questions == []
        assert response.message


def test_fabricated_question_filtered_grounded_question_kept():
    with patch("app.routers.quiz.build_rag_context", side_effect=_context), \
         patch("app.services.quiz.urllib.request.urlopen",
               return_value=_fake_ollama([GROUNDED_Q, FABRICATED_Q])):
        response = generate_quiz(QuizRequest(topic="securite chantier"))
        assert response.total_questions == 1
        assert "Article 15" in response.questions[0].question
        assert response.message is None


def test_all_fabricated_returns_zero_questions_and_message():
    with patch("app.routers.quiz.build_rag_context", side_effect=_context), \
         patch("app.services.quiz.urllib.request.urlopen",
               return_value=_fake_ollama([FABRICATED_Q])):
        response = generate_quiz(QuizRequest(topic="securite chantier"))
        assert response.total_questions == 0
        assert response.questions == []
        assert response.message


def test_malformed_payload_dropped_duplicate_options():
    malformed = {
        "question": "Que dit Article 15 ?",
        "options": ["casque", "casque", "bottes", "masque"],
        "answer": 0,
        "explanation": "Article 15 mentionne le casque",
    }
    with patch("app.routers.quiz.build_rag_context", side_effect=_context), \
         patch("app.services.quiz.urllib.request.urlopen",
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
         patch("app.services.quiz.urllib.request.urlopen",
               return_value=_fake_ollama([malformed, GROUNDED_Q])):
        response = generate_quiz(QuizRequest(topic="securite chantier"))
        assert response.total_questions == 1


def test_grounded_response_carries_sources():
    with patch("app.routers.quiz.build_rag_context", side_effect=_context), \
         patch("app.services.quiz.urllib.request.urlopen",
               return_value=_fake_ollama([GROUNDED_Q])):
        response = generate_quiz(QuizRequest(topic="securite chantier"))
        assert response.sources == ["doc1.pdf"]
