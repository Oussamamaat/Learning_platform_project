"""
Regression tests for the deterministic refusal backstop in the chat route.

First tests for app/routers/chat.py. Written alongside the fix for a live-
reproduced defect: off-topic refusals self-identified as a "safety"
assistant regardless of the tenant's actual domain (3/3 on securite/
blockchain questions). No Ollama or Postgres required -- both
_retrieve_context (the backend-selection indirection point in front of
build_domain_context/build_rag_context -- see settings.retrieval_backend,
"pgvector" by default since 2026-08-10) and urllib.request.urlopen are
monkeypatched. Patching at _retrieve_context means these tests exercise
the SAME refusal/payload-shape behaviour regardless of which backend is
configured; _retrieve_context's own backend-selection and fail-open logic
has its own dedicated tests below.
"""

from unittest.mock import patch

import pytest

from app.models.schemas import ChatRequest, Domain, Language
from app.routers.chat import _sources_look_cross_language, chat
from app.services.llm import DOMAIN_LABELS_AR, DOMAIN_LABELS_FR


def test_cross_language_detection():
    assert _sources_look_cross_language(["1.1_code_du_travail.md"], "darija") is True
    assert _sources_look_cross_language(["1.11_ar_code_travail.md"], "darija") is False
    assert _sources_look_cross_language(["1.11_ar_code_travail.md"], "fr") is True
    assert _sources_look_cross_language(["1.1_code_du_travail.md"], "fr") is False
    assert _sources_look_cross_language([], "fr") is False


def _empty_context(*args, **kwargs):
    return "", []


def _fails_if_called(*args, **kwargs):
    raise AssertionError("Ollama must not be called when context is empty")


@pytest.mark.parametrize("domain", [Domain.INDUSTRIAL, Domain.SECURITE, Domain.BLOCKCHAIN])
def test_empty_context_never_calls_ollama(domain):
    with patch("app.routers.chat._retrieve_context", side_effect=_empty_context), \
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
    with patch("app.routers.chat._retrieve_context", side_effect=_empty_context), \
         patch("app.services.llm.urllib.request.urlopen", side_effect=_fails_if_called):
        response = chat(ChatRequest(
            message="شنو هوما القواعد؟", domain=domain, language=Language.DARIJA,
        ))
        assert expected in response.response
        assert forbidden not in response.response


def test_french_ui_lang_gives_french_refusal():
    with patch("app.routers.chat._retrieve_context", side_effect=_empty_context), \
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
    with patch("app.routers.chat._retrieve_context", side_effect=_empty_context), \
         patch("app.services.llm.urllib.request.urlopen", side_effect=_fails_if_called):
        response = chat(ChatRequest(message="شنو هوما القواعد؟", domain=Domain.INDUSTRIAL))
        assert DOMAIN_LABELS_AR["industrial"] in response.response


def test_omitted_language_french_message_routes_to_french_refusal():
    with patch("app.routers.chat._retrieve_context", side_effect=_empty_context), \
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

    # chat.py now calls generate_llm_response -> _call_ollama_chat, which
    # POSTs /api/chat and reads message.content, not the flat `response`
    # key /api/generate used -- see probe_history_parity.py for why /api/chat
    # is the parity-preserving transport for a history-carrying request.
    fake_ollama = FakeResponse(
        {"message": {"role": "assistant", "content": "Le port du casque est obligatoire (Article 8)."}}
    )

    with patch("app.routers.chat._retrieve_context", side_effect=fake_context), \
         patch("app.services.llm.urllib.request.urlopen", return_value=fake_ollama):
        response = chat(ChatRequest(message="Que dit le document ?", domain=Domain.INDUSTRIAL))
        assert "Article 8" in response.response
        assert response.sources == ["doc1.pdf"]


def test_empty_context_never_calls_ollama_even_with_history_present():
    """The refusal-on-empty-context backstop must not be bypassable by a
    session that already has history -- history is loaded AFTER the empty
    context check in chat.py, but this guards the ordering explicitly so a
    future refactor can't silently move it earlier."""
    with patch("app.routers.chat._retrieve_context", side_effect=_empty_context), \
         patch("app.routers.chat.history.load_window",
               return_value=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]), \
         patch("app.services.llm.urllib.request.urlopen", side_effect=_fails_if_called):
        response = chat(ChatRequest(message="test", domain=Domain.INDUSTRIAL))
        assert response.response
        assert response.sources == []


def test_request_payload_is_well_formed_chat_messages():
    """Capture the actual payload sent to Ollama's /api/chat and assert its
    shape: system first, strictly alternating roles after it, ending on
    user, stream False, temperature 0.2 -- the contract render_conversation
    /probe_history_parity.py verified /api/chat honours."""
    import json

    captured = {}

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

    def fake_urlopen(req, timeout=None):
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse({"message": {"role": "assistant", "content": "Reponse."}})

    with patch("app.routers.chat._retrieve_context", side_effect=fake_context), \
         patch("app.routers.chat.history.load_window",
               return_value=[{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}]), \
         patch("app.routers.chat.history.append_exchange"), \
         patch("app.services.llm.urllib.request.urlopen", side_effect=fake_urlopen):
        chat(ChatRequest(message="Que dit le document ?", domain=Domain.INDUSTRIAL))

    payload = captured["payload"]
    assert payload["stream"] is False
    assert payload["options"]["temperature"] == 0.2
    messages = payload["messages"]
    assert messages[0]["role"] == "system"
    roles = [m["role"] for m in messages[1:]]
    assert roles == ["user", "assistant", "user"], roles
    assert messages[-1]["content"] == "Que dit le document ?"


# --- _retrieve_context: backend selection + fail-open --------------------
# settings.retrieval_backend defaults to "pgvector" (2026-08-10, live-
# verified against real Postgres). These are the only tests that reach
# past _retrieve_context into build_domain_context/build_rag_context
# themselves -- everything above patches _retrieve_context and is backend-
# agnostic by design.

from app.routers.chat import _retrieve_context  # noqa: E402


def test_pgvector_default_calls_build_rag_context_not_disk():
    with patch("app.routers.chat.get_settings") as mock_settings, \
         patch("app.routers.chat.build_rag_context",
               return_value=("pgvector context", ["doc.md"])) as brc, \
         patch("app.routers.chat.build_domain_context") as bdc:
        mock_settings.return_value.retrieval_backend = "pgvector"
        context, sources = _retrieve_context(
            "query", domain="industrial", top_k=4, ui_lang="fr", tenant_id="company_abc"
        )
    assert context == "pgvector context"
    assert sources == ["doc.md"]
    brc.assert_called_once()
    bdc.assert_not_called()


def test_disk_backend_calls_build_domain_context_not_pgvector():
    with patch("app.routers.chat.get_settings") as mock_settings, \
         patch("app.routers.chat.build_domain_context",
               return_value=("disk context", ["doc.md"])) as bdc, \
         patch("app.routers.chat.build_rag_context") as brc:
        mock_settings.return_value.retrieval_backend = "disk"
        context, sources = _retrieve_context(
            "query", domain="industrial", top_k=4, ui_lang="fr", tenant_id="company_abc"
        )
    assert context == "disk context"
    bdc.assert_called_once()
    brc.assert_not_called()


def test_pgvector_failure_falls_back_to_disk_corpus():
    """A Postgres hiccup must degrade chat to the disk corpus, not crash
    the request -- the same fail-open contract app/services/history.py
    already keeps for conversation memory, now extended to retrieval since
    pgvector is the default and needs Postgres reachable to work at all."""
    with patch("app.routers.chat.get_settings") as mock_settings, \
         patch("app.routers.chat.build_rag_context", side_effect=RuntimeError("connection refused")), \
         patch("app.routers.chat.build_domain_context",
               return_value=("fallback context", ["doc.md"])) as bdc:
        mock_settings.return_value.retrieval_backend = "pgvector"
        context, sources = _retrieve_context(
            "query", domain="industrial", top_k=4, ui_lang="fr", tenant_id="company_abc"
        )
    assert context == "fallback context"
    assert sources == ["doc.md"]
    bdc.assert_called_once()


# --- Automatic Domain Routing / language resolution (2026-08-11) ----------
# response.domain/domain_source/language report what app.services.routing
# actually decided for the turn -- exercised through the real chat()
# endpoint, not _resolve_turn_context directly (that's
# tests/test_segment_reset.py's job).

def test_response_carries_explicit_domain_as_page_context():
    with patch("app.routers.chat._retrieve_context", side_effect=_empty_context):
        response = chat(ChatRequest(message="test", domain=Domain.SECURITE))
    assert response.domain == "securite"
    assert response.domain_source == "page_context"
    assert response.language == "fr"  # Latin script, no instruction -> script default


def test_domain_omitted_triggers_autorouting_and_reports_it():
    with patch("app.routers.chat.resolve_domain", return_value=("blockchain", "retrieval")) as rd, \
         patch("app.routers.chat._retrieve_context", side_effect=_empty_context):
        response = chat(ChatRequest(message="Qu'est-ce qu'un smart contract ?"))
    rd.assert_called_once()
    assert response.domain == "blockchain"
    assert response.domain_source == "retrieval"


def test_in_message_instruction_diverges_query_lang_from_response_lang():
    """A French question asking for a Darija answer must retrieve in French
    (query_lang) while answering in Darija (response_lang) -- the split
    app.services.routing.resolve_language exists for."""
    with patch("app.routers.chat._retrieve_context",
               return_value=("Selon Article 8, le casque est obligatoire.", ["doc1.pdf"])) as rc, \
         patch("app.services.llm.urllib.request.urlopen") as mock_urlopen:
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

        mock_urlopen.return_value = FakeResponse(
            {"message": {"role": "assistant", "content": "جواب بالدارجة."}}
        )
        response = chat(ChatRequest(
            message="Comment on fait cela ? Reponds en darija.", domain=Domain.INDUSTRIAL,
        ))
    assert response.language == "darija"
    # Retrieval must have run with the message's own script (French), not
    # the overridden response language.
    assert rc.call_args.kwargs["ui_lang"] == "fr"


def test_pgvector_empty_but_valid_result_is_not_overridden_by_fallback():
    """A real 'nothing matched' result from pgvector (empty context, no
    exception) must be returned as-is -- only a raised exception triggers
    the disk fallback, so an empty result is never silently replaced."""
    with patch("app.routers.chat.get_settings") as mock_settings, \
         patch("app.routers.chat.build_rag_context", return_value=("", [])), \
         patch("app.routers.chat.build_domain_context") as bdc:
        mock_settings.return_value.retrieval_backend = "pgvector"
        context, sources = _retrieve_context(
            "query", domain="industrial", top_k=4, ui_lang="fr", tenant_id="company_abc"
        )
    assert context == ""
    assert sources == []
    bdc.assert_not_called()
