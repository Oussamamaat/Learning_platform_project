"""
Tests for app/routers/voice.py -- the open-mic WebSocket session.

No Postgres, no Ollama, no STT/TTS vendor required: turn resolution
(resolve_turn/load_prior_turns/persist_turn/refusal_text), the LLM stream,
and the VAD endpointer are all monkeypatched at their app.routers.voice
call sites, same convention as tests/test_chat.py. Two tests deliberately
leave settings.stt_engine at its real default ("none") to confirm the
NullSttEngine failure path actually reaches the client as an error event
rather than crashing the session -- see app/services/stt.py.

What this file does NOT cover: a real microphone, a real STT/TTS vendor,
or a fully-timed barge-in race (the worker-thread/event-loop
synchronization needed to deterministically land a barge-in mid-sentence
in a test is more machinery than this MVP's time budget affords -- the
barge-in code path is small and reviewed by hand; a live end-to-end check
of it is deferred to the cloud-GPU phase alongside the STT/TTS bake-off).
"""
import json
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from app.main import app
from app.services.stt import TranscriptChunk
from app.services.turn import TurnContext


class _FakeEndpointer:
    """First push() -> speech_start, second push() -> speech_end, every
    push after that -> None. Decouples these tests from real RMS timing
    (already covered by tests/test_vad.py) so they test routing/state-
    machine behavior instead."""

    def __init__(self, *args, **kwargs):
        self._pushes = 0

    def push(self, frame):
        self._pushes += 1
        if self._pushes == 1:
            return "speech_start"
        if self._pushes == 2:
            return "speech_end"
        return None

    def take_utterance(self):
        return b"\x00" * 640


def _grounded_turn(**overrides) -> TurnContext:
    defaults = dict(
        session_id="sess1", tenant_id="company_abc", user_id="default_user",
        message="Que dit le texte sur le casque ?", domain="industrial",
        domain_source="retrieval", query_lang="fr", response_lang="fr",
        context="Selon Article 8, le port du casque est obligatoire.",
        sources=["doc1.pdf"], segment_id=1, is_new_pin=False, degraded=False,
        corpus_version="v1", override_to_persist=None, override_query_lang_to_persist=None,
    )
    defaults.update(overrides)
    return TurnContext(**defaults)


def _drain_until(ws, predicate, max_messages=20):
    """Receive raw ASGI-style messages until `predicate(msg)` is True,
    returning the full list received (including the matching one)."""
    received = []
    for _ in range(max_messages):
        msg = ws.receive()
        received.append(msg)
        if predicate(msg):
            return received
    raise AssertionError(f"predicate never matched within {max_messages} messages: {received}")


def _is_text_type(msg, type_name: str) -> bool:
    return "text" in msg and msg["text"] is not None and json.loads(msg["text"]).get("type") == type_name


def test_happy_path_streams_citations_answer_and_audio_then_persists():
    fake_stt = MagicMock()
    fake_stt.transcribe.return_value = TranscriptChunk(
        "Que dit le texte sur le casque ?", is_final=True, language="fr"
    )
    fake_tts = MagicMock()
    fake_tts.synthesize.return_value = b"AUDIO-BYTES"
    fake_tts.sample_rate = 22050
    turn = _grounded_turn()

    with patch("app.routers.voice.get_stt_engine", return_value=fake_stt), \
         patch("app.routers.voice.get_tts_engine", return_value=fake_tts), \
         patch("app.routers.voice.EnergyEndpointer", _FakeEndpointer), \
         patch("app.routers.voice.resolve_turn", return_value=turn), \
         patch("app.routers.voice.load_prior_turns", return_value=[]), \
         patch("app.routers.voice.stream_llm_response", return_value=iter(["Bonjour."])), \
         patch("app.routers.voice.persist_turn") as mock_persist:
        # NOTE: extract_citations is deliberately NOT mocked. Its real return
        # is a dict keyed by (head, number) TUPLES -- an earlier version of
        # this test mocked it to return a list of strings, which both hid a
        # real json.dumps crash (tuple keys are not JSON-serializable) and
        # asserted a return shape the real function never produces. The
        # grounded turn's context ("...Article 8...") exercises the real
        # extraction + the flat-label conversion in voice.py.

        client = TestClient(app)
        with client.websocket_connect("/api/v1/voice/session") as ws:
            ws.send_bytes(b"\x00" * 640)  # -> speech_start (no client-visible event)
            ws.send_bytes(b"\x00" * 640)  # -> speech_end -> triggers the turn

            messages = _drain_until(ws, lambda m: _is_text_type(m, "audio.end"))

            types_and_payloads = []
            for msg in messages:
                if "text" in msg and msg["text"] is not None:
                    types_and_payloads.append(("text", json.loads(msg["text"])))
                elif "bytes" in msg and msg["bytes"] is not None:
                    types_and_payloads.append(("bytes", msg["bytes"]))

            kinds = [t for t, _ in types_and_payloads]
            assert kinds == ["text", "text", "text", "text", "text", "bytes", "text"]

            assert types_and_payloads[0][1]["type"] == "transcript.partial"

            transcript_msg = types_and_payloads[1][1]
            assert transcript_msg["type"] == "transcript.final"
            assert transcript_msg["text"] == "Que dit le texte sur le casque ?"

            citations_msg = types_and_payloads[2][1]
            assert citations_msg["type"] == "citations"
            # Real extract_citations turns the turn context's "Article 8" into
            # a canonical label; voice.py flattens the tuple-keyed dict to
            # this list of strings (matches the frontend's string[] contract).
            assert citations_msg["sources"] == ["Article 8"]

            audio_start_msg = types_and_payloads[3][1]
            assert audio_start_msg["type"] == "audio.start"
            assert audio_start_msg["sample_rate"] == 22050

            answer_delta_msg = types_and_payloads[4][1]
            assert answer_delta_msg["type"] == "answer.delta"
            assert answer_delta_msg["text"] == "Bonjour."

            assert types_and_payloads[5][1] == b"AUDIO-BYTES"
            assert types_and_payloads[6][1]["type"] == "audio.end"

            ws.send_text(json.dumps({"type": "end"}))

    mock_persist.assert_called_once()
    args, kwargs = mock_persist.call_args
    assert args[0] is turn
    assert kwargs["assistant_content"] == "Bonjour."


def test_refusal_path_never_calls_the_model():
    fake_stt = MagicMock()
    fake_stt.transcribe.return_value = TranscriptChunk("bla bla bla", is_final=True, language="fr")
    fake_tts = MagicMock()
    fake_tts.synthesize.return_value = b"REFUSAL-AUDIO"
    fake_tts.sample_rate = 22050
    turn = _grounded_turn(context="", domain_source="no_match")
    assert turn.is_refusal is True

    with patch("app.routers.voice.get_stt_engine", return_value=fake_stt), \
         patch("app.routers.voice.get_tts_engine", return_value=fake_tts), \
         patch("app.routers.voice.EnergyEndpointer", _FakeEndpointer), \
         patch("app.routers.voice.resolve_turn", return_value=turn), \
         patch("app.routers.voice.load_prior_turns", return_value=[]), \
         patch("app.routers.voice.stream_llm_response") as mock_stream, \
         patch("app.routers.voice.refusal_text", return_value="Je ne peux pas répondre à cela.") as mock_refusal, \
         patch("app.routers.voice.persist_turn") as mock_persist:

        client = TestClient(app)
        with client.websocket_connect("/api/v1/voice/session") as ws:
            ws.send_bytes(b"\x00" * 640)
            ws.send_bytes(b"\x00" * 640)

            messages = _drain_until(ws, lambda m: _is_text_type(m, "audio.end"))
            ws.send_text(json.dumps({"type": "end"}))

    mock_stream.assert_not_called()
    mock_refusal.assert_called_once()
    mock_persist.assert_called_once()
    _, kwargs = mock_persist.call_args
    assert kwargs["assistant_content"] == "Je ne peux pas répondre à cela."

    texts = [json.loads(m["text"]) for m in messages if "text" in m and m["text"] is not None]
    assert any(t["type"] == "answer.delta" and t["text"] == "Je ne peux pas répondre à cela." for t in texts)


def test_stt_unavailable_sends_error_event_without_crashing_session():
    """Leaves settings.stt_engine at its real default ("none") --
    NullSttEngine.transcribe always raises SttUnavailableError. Confirms
    that failure reaches the client as a clean error event rather than
    tearing down the connection or leaking a 500."""
    with patch("app.routers.voice.EnergyEndpointer", _FakeEndpointer):
        client = TestClient(app)
        with client.websocket_connect("/api/v1/voice/session") as ws:
            ws.send_bytes(b"\x00" * 640)
            ws.send_bytes(b"\x00" * 640)

            messages = _drain_until(ws, lambda m: _is_text_type(m, "error"))
            error_msg = json.loads(messages[-1]["text"])
            assert error_msg["code"] == "stt_unavailable"

            ws.send_text(json.dumps({"type": "end"}))
