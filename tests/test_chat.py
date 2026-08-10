"""
Regression tests for the deterministic refusal backstop in the chat route.

First tests for app/routers/chat.py. Written alongside the fix for a live-
reproduced defect: off-topic refusals self-identified as a "safety"
assistant regardless of the tenant's actual domain (3/3 on securite/
blockchain questions). No Ollama or Postgres required -- both
build_domain_context and urllib.request.urlopen are monkeypatched.
"""

from unittest.mock import patch

import pytest

from app.models.schemas import ChatRequest, Domain, Language
from app.routers.chat import chat
from app.services.llm import DOMAIN_LABELS_AR, DOMAIN_LABELS_FR


def _empty_context(*args, **kwargs):
    return "", []


def _fails_if_called(*args, **kwargs):
    raise AssertionError("Ollama must not be called when context is empty")


@pytest.mark.parametrize("domain", [Domain.INDUSTRIAL, Domain.SECURITE, Domain.BLOCKCHAIN])
def test_empty_context_never_calls_ollama(domain):
    with patch("app.routers.chat.build_domain_context", side_effect=_empty_context), \
         patch("app.services.llm.urllib.request.urlopen", side_effect=_fails_if_called):
        response = chat(ChatRequest(message="test", domain=domain))
        assert response.response
        assert response.sources == []
        assert response.tokens_used == 0


@pytest.mark.parametrize(
    "domain,expected,forbidden",
    [
        (Domain.INDUSTRIAL, DOMAIN_LABELS_AR["industrial"], DOMAIN_LABELS_AR["securite"]),
        (Domain.SECURITE, DOMAIN_LABELS_AR["securite"], DOMAIN_LABELS_AR["industrial"]),
        (Domain.BLOCKCHAIN, DOMAIN_LABELS_AR["blockchain"], DOMAIN_LABELS_AR["industrial"]),
    ],
)
def test_darija_refusal_names_correct_domain_no_leakage(domain, expected, forbidden):
    """Direct regression for the reproduced defect: refusals must name the
    tenant's actual domain, and never another domain's label."""
    with patch("app.routers.chat.build_domain_context", side_effect=_empty_context), \
         patch("app.services.llm.urllib.request.urlopen", side_effect=_fails_if_called):
        response = chat(ChatRequest(
            message="شنو هوما القواعد؟", domain=domain, language=Language.DARIJA,
        ))
        assert expected in response.response
        assert forbidden not in response.response


def test_french_ui_lang_gives_french_refusal():
    with patch("app.routers.chat.build_domain_context", side_effect=_empty_context), \
         patch("app.services.llm.urllib.request.urlopen", side_effect=_fails_if_called):
        response = chat(ChatRequest(
            message="Quelle est la politique de securite ?",
            domain=Domain.SECURITE,
            language=Language.FRENCH,
        ))
        assert DOMAIN_LABELS_FR["securite"] in response.response
        assert DOMAIN_LABELS_AR["securite"] not in response.response


def test_omitted_language_falls_back_to_heuristic_darija_default():
    """Guards the Darija demo default: a client that omits `language`
    entirely (as every pre-existing caller does) must still get the
    heuristic's answer, not silently flip to French."""
    with patch("app.routers.chat.build_domain_context", side_effect=_empty_context), \
         patch("app.services.llm.urllib.request.urlopen", side_effect=_fails_if_called):
        response = chat(ChatRequest(message="شنو هوما القواعد؟", domain=Domain.INDUSTRIAL))
        assert DOMAIN_LABELS_AR["industrial"] in response.response


def test_omitted_language_french_message_routes_to_french_refusal():
    with patch("app.routers.chat.build_domain_context", side_effect=_empty_context), \
         patch("app.services.llm.urllib.request.urlopen", side_effect=_fails_if_called):
        response = chat(ChatRequest(
            message="Quelle est la politique de securite ?", domain=Domain.SECURITE,
        ))
        assert DOMAIN_LABELS_FR["securite"] in response.response


def test_nonempty_context_still_calls_the_model():
    """Confirms the interception is scoped to empty context only -- the
    happy path must be unaffected."""
    import json

    class FakeResponse:
        def __init__(self, body):
            self._body = body

        def read(self):
            return json.dumps(self._body).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_context(*args, **kwargs):
        return "Selon Article 8, le port du casque est obligatoire.", ["doc1.pdf"]

    fake_ollama = FakeResponse({"response": "Le port du casque est obligatoire (Article 8)."})

    with patch("app.routers.chat.build_domain_context", side_effect=fake_context), \
         patch("app.services.llm.urllib.request.urlopen", return_value=fake_ollama):
        response = chat(ChatRequest(message="Que dit le document ?", domain=Domain.INDUSTRIAL))
        assert "Article 8" in response.response
        assert response.sources == ["doc1.pdf"]
