"""
Tests for app.services.turn -- the voice pipeline's reuse of
app.routers.chat's turn-resolution machinery (see turn.py's module
docstring for why this reuses _resolve_turn_context BY IMPORT rather than
extracting/duplicating it).

No Postgres required: history.*, source_service.active_sources_and_version,
and app.routers.chat._resolve_turn_context are all monkeypatched, same
convention as tests/test_chat.py.
"""
from unittest.mock import patch

import app.services.turn as turn_mod
from app.services.turn import resolve_turn, persist_turn, load_prior_turns, refusal_text, TurnContext


def _fake_resolve_turn_context(*args, **kwargs):
    return (
        "Selon Article 8, le port du casque est obligatoire.",  # context
        ["doc1.pdf"],  # sources
        3,  # segment_id
        True,  # is_new_pin
        "industrial",  # domain
        "retrieval",  # domain_source
        False,  # degraded
        "corpus-hash-abc",  # corpus_version
    )


def _patched():
    return (
        patch.object(turn_mod.history, "get_language_state", return_value=(None, None)),
        patch.object(
            turn_mod.source_service, "active_sources_and_version",
            return_value=(["src1"], "corpus-hash-abc"),
        ),
        patch("app.routers.chat._resolve_turn_context", side_effect=_fake_resolve_turn_context),
    )


def test_resolve_turn_populates_turn_context_from_chat_machinery():
    p1, p2, p3 = _patched()
    with p1, p2, p3:
        turn = resolve_turn(
            "Que dit le texte sur le casque ?", tenant_id="company_abc", user_id="u1",
        )
    assert isinstance(turn, TurnContext)
    assert turn.domain == "industrial"
    assert turn.domain_source == "retrieval"
    assert turn.context == "Selon Article 8, le port du casque est obligatoire."
    assert turn.sources == ["doc1.pdf"]
    assert turn.segment_id == 3
    assert turn.is_new_pin is True
    assert turn.degraded is False
    assert turn.corpus_version == "corpus-hash-abc"
    assert turn.query_lang == "fr"
    assert turn.response_lang == "fr"
    assert turn.session_id  # auto-generated when omitted


def test_resolve_turn_reuses_supplied_session_id():
    p1, p2, p3 = _patched()
    with p1, p2, p3:
        turn = resolve_turn(
            "hello", tenant_id="t", user_id="u", session_id="fixed-session-id",
        )
    assert turn.session_id == "fixed-session-id"


def test_is_refusal_true_on_no_match():
    def no_match(*a, **k):
        return ("Selon Article 8...", ["doc1.pdf"], 1, True, "industrial", "no_match", False, None)

    with patch.object(turn_mod.history, "get_language_state", return_value=(None, None)), \
         patch.object(turn_mod.source_service, "active_sources_and_version", return_value=([], None)), \
         patch("app.routers.chat._resolve_turn_context", side_effect=no_match):
        turn = resolve_turn("random unrelated question", tenant_id="t", user_id="u")
    assert turn.is_refusal is True


def test_is_refusal_true_on_empty_context():
    def empty_context(*a, **k):
        return ("", [], 1, True, "industrial", "retrieval", False, None)

    with patch.object(turn_mod.history, "get_language_state", return_value=(None, None)), \
         patch.object(turn_mod.source_service, "active_sources_and_version", return_value=([], None)), \
         patch("app.routers.chat._resolve_turn_context", side_effect=empty_context):
        turn = resolve_turn("q", tenant_id="t", user_id="u")
    assert turn.is_refusal is True


def test_is_refusal_false_when_grounded():
    p1, p2, p3 = _patched()
    with p1, p2, p3:
        turn = resolve_turn("q", tenant_id="t", user_id="u")
    assert turn.is_refusal is False


def _grounded_turn(**overrides) -> TurnContext:
    defaults = dict(
        session_id="sess1", tenant_id="t1", user_id="u1", message="q",
        domain="industrial", domain_source="retrieval", query_lang="fr",
        response_lang="fr", context="ctx", sources=["doc1.pdf"], segment_id=1,
        is_new_pin=False, degraded=False, corpus_version="v1",
        override_to_persist=None, override_query_lang_to_persist=None,
    )
    defaults.update(overrides)
    return TurnContext(**defaults)


def test_persist_turn_calls_append_exchange_with_turn_fields():
    turn = _grounded_turn()
    with patch.object(turn_mod.history, "append_exchange") as mock_append, \
         patch.object(turn_mod.history, "pin_context") as mock_pin:
        persist_turn(turn, assistant_content="The answer.")
    mock_append.assert_called_once()
    _, kwargs = mock_append.call_args
    assert kwargs["tenant_id"] == "t1"
    assert kwargs["domain"] == "industrial"
    assert kwargs["language"] == "fr"
    assert kwargs["segment_id"] == 1
    assert kwargs["user_content"] == "q"
    assert kwargs["assistant_content"] == "The answer."
    assert kwargs["sources"] == ["doc1.pdf"]
    mock_pin.assert_not_called()  # is_new_pin=False


def test_persist_turn_pins_context_only_when_is_new_pin():
    turn = _grounded_turn(is_new_pin=True)
    with patch.object(turn_mod.history, "append_exchange"), \
         patch.object(turn_mod.history, "pin_context") as mock_pin:
        persist_turn(turn, assistant_content="The answer.")
    mock_pin.assert_called_once()
    _, kwargs = mock_pin.call_args
    assert kwargs["context"] == "ctx"
    assert kwargs["sources"] == ["doc1.pdf"]
    assert kwargs["corpus_version"] == "v1"


def test_load_prior_turns_delegates_to_history_load_window():
    turn = _grounded_turn(session_id="sess-xyz", domain="securite", query_lang="darija", segment_id=7)
    with patch.object(turn_mod.history, "load_window", return_value=[{"role": "user", "content": "hi"}]) as mock_load:
        result = load_prior_turns(turn)
    mock_load.assert_called_once_with(
        "sess-xyz", domain="securite", language="darija", segment_id=7,
    )
    assert result == [{"role": "user", "content": "hi"}]


def test_refusal_text_delegates_to_deterministic_refusal():
    turn = _grounded_turn(domain="blockchain", response_lang="darija")
    with patch.object(turn_mod, "deterministic_refusal", return_value="REFUSAL TEXT") as mock_refusal:
        text = refusal_text(turn)
    mock_refusal.assert_called_once_with("blockchain", "darija")
    assert text == "REFUSAL TEXT"
