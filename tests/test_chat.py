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


@pytest.fixture(autouse=True)
def _no_live_corpus_version_lookup():
    """chat()'s corpus_version pin-invalidation check (app/routers/chat.py:182,
    added alongside the tenant-upload API on 2026-08-13) hits a real Postgres
    connection through app.services.sources.corpus_version -- a dependency this
    file's tests predate and were never updated for. Without this, the first
    such test in file order hangs forever on psycopg2.connect() with no
    Postgres reachable (no timeout on that call), rather than failing fast;
    confirmed live via `pytest --timeout=15`, which is what caught this.

    Returning None matches corpus_version's own documented contract for a
    genuinely unreachable Postgres (see its docstring: None means "unknown,
    do not invalidate"), so this is the faithful mock for "no live DB" here,
    not an arbitrary stub. Tests that specifically exercise corpus_version's
    own value (e.g. test_corpus_version_is_not_re_queried_after_the_merged_
    lookup) already patch it explicitly within their own `with` block, which
    safely overrides this fixture for their duration."""
    with patch("app.routers.chat.source_service.corpus_version", return_value=None):
        yield


def test_cross_language_detection():
    assert _sources_look_cross_language(["1.1_code_du_travail.md"], "darija") is True
    assert _sources_look_cross_language(["1.11_ar_code_travail.md"], "darija") is False
    assert _sources_look_cross_language(["1.11_ar_code_travail.md"], "fr") is True
    assert _sources_look_cross_language(["1.1_code_du_travail.md"], "fr") is False
    assert _sources_look_cross_language([], "fr") is False


def _empty_context(*args, **kwargs):
    return "", [], False


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
        return "Selon Article 8, le port du casque est obligatoire.", ["doc1.pdf"], False

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


def test_out_of_corpus_query_refuses_even_when_context_is_nonempty():
    """The out-of-domain refusal fix, isolated from the pre-existing
    empty-context path: context here is deliberately NON-empty, so the only
    thing that can trigger a refusal is domain_source == "no_match".

    This is the exact hole the fix closes. Domain-scoped retrieval almost
    always returns SOMETHING once a domain is chosen, so an off-topic
    question ("how do I bake bread") used to route to the tenant default
    domain, pull its nearest-but-irrelevant chunks, and get answered as if
    grounded -- `if not context.strip()` never fired. The tier-2 vote is the
    only signal that sees the whole corpus at once and can say "none of this
    is about that"."""
    def nonempty_context(*args, **kwargs):
        return "Selon Article 8, le port du casque est obligatoire.", ["doc1.pdf"], False

    with patch("app.routers.chat._retrieve_context", side_effect=nonempty_context), \
         patch("app.routers.chat.resolve_domain", return_value=("industrial", "no_match")), \
         patch("app.services.llm.urllib.request.urlopen", side_effect=_fails_if_called):
        # No explicit domain -> resolve_domain runs (an explicit domain would
        # short-circuit to "page_context" and never consult the vote).
        response = chat(ChatRequest(message="How do I bake sourdough bread?"))
        assert response.domain_source == "no_match"
        assert response.sources == []
        assert response.tokens_used == 0
        assert response.response  # a real refusal, not an empty string


def test_in_corpus_query_with_nonempty_context_is_unaffected_by_the_ood_gate():
    """Guards the other direction: a normally-routed query ("retrieval")
    must still reach the model. The OOD gate must not become a blanket
    refusal for every auto-routed turn."""
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

    def nonempty_context(*args, **kwargs):
        return "Selon Article 8, le port du casque est obligatoire.", ["doc1.pdf"], False

    fake_ollama = FakeResponse(
        {"message": {"role": "assistant", "content": "Le port du casque est obligatoire (Article 8)."}}
    )

    with patch("app.routers.chat._retrieve_context", side_effect=nonempty_context), \
         patch("app.routers.chat.resolve_domain", return_value=("industrial", "retrieval")), \
         patch("app.services.llm.urllib.request.urlopen", return_value=fake_ollama):
        response = chat(ChatRequest(message="Que dit le document sur le casque ?"))
        assert response.domain_source == "retrieval"
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
        return "Selon Article 8, le port du casque est obligatoire.", ["doc1.pdf"], False

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
        context, sources, degraded = _retrieve_context(
            "query", domain="industrial", top_k=4, ui_lang="fr", tenant_id="company_abc"
        )
    assert context == "pgvector context"
    assert sources == ["doc.md"]
    assert degraded is False
    brc.assert_called_once()
    bdc.assert_not_called()


def test_disk_backend_calls_build_domain_context_not_pgvector():
    with patch("app.routers.chat.get_settings") as mock_settings, \
         patch("app.routers.chat.build_domain_context",
               return_value=("disk context", ["doc.md"])) as bdc, \
         patch("app.routers.chat.build_rag_context") as brc:
        mock_settings.return_value.retrieval_backend = "disk"
        context, sources, degraded = _retrieve_context(
            "query", domain="industrial", top_k=4, ui_lang="fr", tenant_id="company_abc"
        )
    assert context == "disk context"
    assert degraded is False
    bdc.assert_called_once()
    brc.assert_not_called()


def test_pgvector_failure_falls_back_to_disk_corpus():
    """A Postgres hiccup must degrade chat to the disk corpus, not crash
    the request -- the same fail-open contract app/services/history.py
    already keeps for conversation memory, now extended to retrieval since
    pgvector is the default and needs Postgres reachable to work at all.
    degraded must be True here specifically -- this is the one case
    ChatResponse.degraded exists to surface, since the disk backend it
    falls back to knows nothing about tenant uploads."""
    with patch("app.routers.chat.get_settings") as mock_settings, \
         patch("app.routers.chat.build_rag_context", side_effect=RuntimeError("connection refused")), \
         patch("app.routers.chat.build_domain_context",
               return_value=("fallback context", ["doc.md"])) as bdc:
        mock_settings.return_value.retrieval_backend = "pgvector"
        context, sources, degraded = _retrieve_context(
            "query", domain="industrial", top_k=4, ui_lang="fr", tenant_id="company_abc"
        )
    assert context == "fallback context"
    assert sources == ["doc.md"]
    assert degraded is True
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
               return_value=("Selon Article 8, le casque est obligatoire.", ["doc1.pdf"], False)) as rc, \
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
        context, sources, degraded = _retrieve_context(
            "query", domain="industrial", top_k=4, ui_lang="fr", tenant_id="company_abc"
        )
    assert context == ""
    assert sources == []
    assert degraded is False


# --- Tenant uploads: active_source_ids + degraded (2026-08-13) ------------

def test_chat_response_degraded_flag_reaches_the_client():
    """End-to-end through the real chat() endpoint: a pgvector exception
    (falling back to disk) must surface as ChatResponse.degraded=True, not
    just internally to _retrieve_context -- the UI needs this to warn that
    uploaded sources are temporarily unreachable."""
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

    with patch("app.routers.chat.get_settings") as mock_settings, \
         patch("app.routers.chat.build_rag_context", side_effect=RuntimeError("connection refused")), \
         patch("app.routers.chat.build_domain_context", return_value=("fallback context", ["doc.md"])), \
         patch("app.services.llm.urllib.request.urlopen",
               return_value=FakeResponse({"message": {"role": "assistant", "content": "Reponse."}})):
        mock_settings.return_value.retrieval_backend = "pgvector"
        response = chat(ChatRequest(message="test", domain=Domain.INDUSTRIAL))
    assert response.degraded is True


def test_chat_response_not_degraded_on_happy_path():
    with patch("app.routers.chat._retrieve_context",
               return_value=("Selon Article 8, le casque est obligatoire.", ["doc1.pdf"], False)), \
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
            {"message": {"role": "assistant", "content": "Le casque est obligatoire (Article 8)."}}
        )
        response = chat(ChatRequest(message="Que dit le document ?", domain=Domain.INDUSTRIAL))
    assert response.degraded is False


def test_active_source_ids_from_request_reaches_retrieve_context():
    """ChatRequest.active_source_ids must reach _retrieve_context's
    source_ids kwarg -- threaded through source_service.
    active_sources_and_version (mocked here to isolate this test from real
    Postgres state) and _resolve_turn_context.

    Patches active_sources_and_version, not active_source_ids: chat() now
    gets the active-id list and the pin-invalidation corpus_version from
    ONE query instead of two (both derive from the same source_files rows
    -- see app.services.sources). The contract under test is unchanged;
    only which function chat calls to satisfy it moved.
    """
    with patch("app.routers.chat.source_service.active_sources_and_version",
               return_value=(["src-1", "src-2"], "corpus-v1")) as asv,          patch("app.routers.chat._retrieve_context",
               return_value=("ctx", ["doc.md"], False)) as rc:
        chat(ChatRequest(
            message="Que dit le document uploade ?", domain=Domain.INDUSTRIAL,
            active_source_ids=["src-1", "src-2", "src-unrelated"],
        ))
    # The client-supplied list is passed through as a NARROWING hint --
    # active_sources_and_version itself (mocked here) owns the actual
    # intersection logic, tested directly in tests/test_sources.py.
    asv.assert_called_once_with("company_abc", requested=["src-1", "src-2", "src-unrelated"])
    assert rc.call_args.kwargs["source_ids"] == ["src-1", "src-2"]


def test_corpus_version_is_not_re_queried_after_the_merged_lookup():
    """The corpus_version chat() already holds from active_sources_and_version
    must be threaded into _resolve_turn_context, not fetched a second time --
    that duplicate SELECT over identical rows is what merging the two
    lookups removed."""
    with patch("app.routers.chat.source_service.active_sources_and_version",
               return_value=([], "corpus-v1")),          patch("app.routers.chat.source_service.corpus_version") as cv,          patch("app.routers.chat._retrieve_context",
               return_value=("ctx", ["doc.md"], False)),          patch("app.routers.chat.generate_llm_response", return_value="Reponse."):
        chat(ChatRequest(message="Que dit le document ?", domain=Domain.INDUSTRIAL))
    cv.assert_not_called()


# --- Diagram generation (app.services.diagrams) ---------------------------
#
# The trigger is the message's own text (app.services.diagrams.
# detect_diagram_intent), checked BEFORE the empty-context refusal gate --
# see chat.py's step 2b. No separate diagram endpoint exists.


def test_diagram_request_returns_populated_diagram():
    def fake_context(*args, **kwargs):
        return "Article 12 : couper l'alimentation puis verrouiller la machine.", ["doc1.pdf"], False

    diagram_spec = {
        "title": "Consignation",
        "caption": "Voici les etapes de consignation avant intervention.",
        "direction": "TD",
        "nodes": [{"id": "a", "label": "Couper l'alimentation"}, {"id": "b", "label": "Verrouiller"}],
        "edges": [{"source": "a", "target": "b"}],
    }

    class FakeResponse:
        def __init__(self, body):
            self._body = body
        def read(self):
            import json
            return json.dumps(self._body).encode("utf-8")
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False

    with patch("app.routers.chat._retrieve_context", side_effect=fake_context), \
         patch("app.services.llm.urllib.request.urlopen",
               return_value=FakeResponse({"response": __import__("json").dumps(diagram_spec)})):
        response = chat(ChatRequest(
            message="Dessine-moi un schema des etapes de consignation",
            domain=Domain.INDUSTRIAL,
        ))
    assert response.diagram is not None
    assert response.diagram.kind == "flowchart"
    assert response.diagram.grounded is True
    assert "flowchart TD" in response.diagram.mermaid
    assert response.response == response.diagram.caption


def test_ordinary_question_never_populates_diagram():
    """A normal question containing no diagram keyword must return
    diagram=None and never even attempt diagram generation."""
    def fake_context(*args, **kwargs):
        return "Selon Article 8, le port du casque est obligatoire.", ["doc1.pdf"], False

    class FakeResponse:
        def __init__(self, body):
            self._body = body
        def read(self):
            import json
            return json.dumps(self._body).encode("utf-8")
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False

    fake_ollama = FakeResponse(
        {"message": {"role": "assistant", "content": "Le port du casque est obligatoire (Article 8)."}}
    )
    with patch("app.routers.chat._retrieve_context", side_effect=fake_context), \
         patch("app.services.llm.urllib.request.urlopen", return_value=fake_ollama):
        response = chat(ChatRequest(message="Que dit le document sur le port du casque ?", domain=Domain.INDUSTRIAL))
    assert response.diagram is None


def test_diagram_request_with_empty_context_still_produces_ungrounded_diagram():
    """The empty-context refusal gate must not fire for a diagram request --
    an ungrounded diagram (grounded=False) is produced instead, never a
    refusal. This is the ordering invariant chat.py's step 2b docstring
    calls out explicitly."""
    diagram_spec = {
        "title": "Marteau haussier",
        "caption": "Illustration d'un marteau haussier.",
        "candles": [
            {"label": "J1", "open": 10, "high": 12, "low": 8, "close": 11},
            {"label": "J2", "open": 11, "high": 13, "low": 9, "close": 12},
        ],
    }

    class FakeResponse:
        def __init__(self, body):
            self._body = body
        def read(self):
            import json
            return json.dumps(self._body).encode("utf-8")
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False

    with patch("app.routers.chat._retrieve_context", side_effect=_empty_context), \
         patch("app.services.llm.urllib.request.urlopen",
               return_value=FakeResponse({"response": __import__("json").dumps(diagram_spec)})):
        response = chat(ChatRequest(message="Montre-moi une bougie en marteau haussier", domain=Domain.INDUSTRIAL))
    assert response.diagram is not None
    assert response.diagram.kind == "candlestick"
    assert response.diagram.grounded is False


def test_diagram_request_ignores_stale_pinned_context_on_no_match():
    """Regression test for a defect caught live via a real same-session,
    two-turn browser conversation (Playwright): a grounded flowchart turn
    followed by an off-topic candlestick follow-up. _resolve_turn_context's
    own documented fallback ("a stale-but-relevant answer beats a false
    refusal") reused turn 1's pinned lockout/tagout context and sources for
    turn 2, even though domain_source == "no_match" (this turn's own vote
    found nothing candlestick-relevant). Before the fix, generate_diagram
    saw that non-empty stale context, set grounded=True, and returned an
    unrelated safety document as the candlestick's "source" -- exactly the
    confident-but-wrong signal ChatResponse.degraded's docstring warns
    against for the pgvector-fallback case, here caused by pin reuse
    instead. chat.py's diagram branch now treats domain_source == "no_match"
    as empty context/sources for generate_diagram's purposes, same as the
    refusal gate below it already does."""
    def stale_pin_context(*args, **kwargs):
        # Simulates _resolve_turn_context falling back to a same-session
        # pin from an earlier, unrelated (but genuinely grounded) turn.
        return (
            "Article 12 : couper l'alimentation puis verrouiller la machine.",  # context
            ["1.5_lockout_tagout_procedures.md"],  # sources
            1,      # segment_id
            False,  # is_new_pin
            "industrial",  # domain
            "no_match",    # domain_source -- THIS turn's own vote found nothing
            False,  # degraded
            None,   # corpus_version
        )

    diagram_spec = {
        "title": "Marteau haussier",
        "caption": "Illustration d'un marteau haussier.",
        "candles": [
            {"label": "J1", "open": 10, "high": 12, "low": 8, "close": 11},
            {"label": "J2", "open": 11, "high": 13, "low": 9, "close": 12},
        ],
    }

    class FakeResponse:
        def __init__(self, body):
            self._body = body
        def read(self):
            import json
            return json.dumps(self._body).encode("utf-8")
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False

    with patch("app.routers.chat._resolve_turn_context", side_effect=stale_pin_context), \
         patch("app.services.llm.urllib.request.urlopen",
               return_value=FakeResponse({"response": __import__("json").dumps(diagram_spec)})):
        response = chat(ChatRequest(message="Montre-moi une bougie en marteau haussier", domain=Domain.INDUSTRIAL))
    assert response.diagram is not None
    assert response.diagram.grounded is False
    assert response.diagram.sources == []
    assert response.sources == []
