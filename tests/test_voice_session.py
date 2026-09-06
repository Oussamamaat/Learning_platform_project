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
    """Every odd push() -> speech_start, every even push() -> speech_end --
    cycles indefinitely so a single test can drive MULTIPLE utterances
    through one session (send 2 frames per utterance), not just one.
    Decouples these tests from real RMS timing (already covered by
    tests/test_vad.py) so they test routing/state-machine behavior
    instead."""

    def __init__(self, *args, **kwargs):
        self._pushes = 0

    def push(self, frame):
        self._pushes += 1
        return "speech_start" if self._pushes % 2 == 1 else "speech_end"

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


def test_tts_failure_other_than_unavailable_surfaces_error_and_recovers():
    """A TTS engine raising anything OTHER than TtsUnavailableError (a
    piper-tts API mismatch, a phonemizer failure, a bad ONNX load) used to
    propagate uncaught out of _answer_worker's speak() and out of
    run_and_persist -- the client still got audio.end and looked healthy
    with no audio ever played, AND `state` never reset to LISTENING,
    silently routing every later utterance to the barge-in branch instead
    of transcription. This is the exact live-reported symptom ("no voice
    is heard... doesn't detect when I talk"), confirmed live 2026-09-05/06
    by reading app/routers/voice.py, not a hypothesis. Locks both halves
    of the fix: an error event reaches the client, AND a second utterance
    in the same session is still transcribed afterward."""
    fake_stt = MagicMock()
    fake_stt.transcribe.return_value = TranscriptChunk(
        "Que dit le texte sur le casque ?", is_final=True, language="fr"
    )
    fake_tts = MagicMock()
    fake_tts.synthesize.side_effect = RuntimeError("piper-tts API mismatch")
    fake_tts.sample_rate = 22050
    turn = _grounded_turn()

    with patch("app.routers.voice.get_stt_engine", return_value=fake_stt), \
         patch("app.routers.voice.get_tts_engine", return_value=fake_tts), \
         patch("app.routers.voice.EnergyEndpointer", _FakeEndpointer), \
         patch("app.routers.voice.resolve_turn", return_value=turn), \
         patch("app.routers.voice.load_prior_turns", return_value=[]), \
         patch("app.routers.voice.stream_llm_response",
               side_effect=lambda **kwargs: iter(["Bonjour."])), \
         patch("app.routers.voice.persist_turn"):
        # side_effect (not return_value): return_value=iter([...]) hands
        # EVERY call the same iterator, so the second turn would silently
        # receive an already-exhausted one and generate nothing.

        client = TestClient(app)
        with client.websocket_connect("/api/v1/voice/session") as ws:
            ws.send_bytes(b"\x00" * 640)
            ws.send_bytes(b"\x00" * 640)

            messages = _drain_until(ws, lambda m: _is_text_type(m, "audio.end"))
            texts = [json.loads(m["text"]) for m in messages if "text" in m and m["text"] is not None]
            error_events = [t for t in texts if t["type"] == "error"]
            assert error_events, f"expected a tts_failed error event, got: {texts}"
            assert error_events[0]["code"] == "tts_failed"
            # No audio bytes made it through -- synthesize() raised every time.
            assert not any("bytes" in m and m["bytes"] is not None for m in messages)

            # The session must have recovered to LISTENING, not stuck in
            # SPEAKING routing every frame to barge-in -- send a second
            # utterance and confirm it is transcribed AND answered through
            # to audio.end. Draining to audio.end (rather than stopping at
            # transcript.final) is what makes this deterministic: it proves
            # the second turn's worker actually ran, instead of racing the
            # server's teardown.
            ws.send_bytes(b"\x00" * 640)
            ws.send_bytes(b"\x00" * 640)
            messages2 = _drain_until(ws, lambda m: _is_text_type(m, "audio.end"))
            assert any(_is_text_type(m, "transcript.final") for m in messages2)

            ws.send_text(json.dumps({"type": "end"}))


def test_language_pinning_off_by_default_stt_always_autodetects():
    """Default settings.voice_language_pinning=False: STT's language_hint
    must be None on EVERY utterance (never forced to a prior turn's
    language), and resolve_turn's explicit_language must be None too --
    both used to be unconditionally set to `pinned_language` after turn 1,
    which force-decoded a later Darija utterance as French (whisper's
    `language=` argument disables auto-detection, it does not just bias
    it) and made resolve_language's precedence-0 explicit_language slot
    short-circuit script/instruction detection. Confirmed live
    2026-09-05/06, not a hypothesis."""
    fake_stt = MagicMock()
    fake_stt.transcribe.side_effect = [
        TranscriptChunk("Bonjour.", is_final=True, language="fr"),
        TranscriptChunk("خصك تلبس الكاسك.", is_final=True, language="ar"),
    ]
    fake_tts = MagicMock()
    fake_tts.synthesize.return_value = b"AUDIO"
    fake_tts.sample_rate = 22050
    turn_fr = _grounded_turn(response_lang="fr")
    turn_darija = _grounded_turn(response_lang="darija", query_lang="darija")

    with patch("app.routers.voice.get_stt_engine", return_value=fake_stt), \
         patch("app.routers.voice.get_tts_engine", return_value=fake_tts), \
         patch("app.routers.voice.EnergyEndpointer", _FakeEndpointer), \
         patch("app.routers.voice.resolve_turn", side_effect=[turn_fr, turn_darija]) as mock_resolve, \
         patch("app.routers.voice.load_prior_turns", return_value=[]), \
         patch("app.routers.voice.stream_llm_response", return_value=iter(["Ok."])), \
         patch("app.routers.voice.persist_turn"):

        client = TestClient(app)
        with client.websocket_connect("/api/v1/voice/session") as ws:
            for _ in range(2):
                ws.send_bytes(b"\x00" * 640)
                ws.send_bytes(b"\x00" * 640)
                _drain_until(ws, lambda m: _is_text_type(m, "audio.end"))
            ws.send_text(json.dumps({"type": "end"}))

    assert fake_stt.transcribe.call_count == 2
    for call in fake_stt.transcribe.call_args_list:
        assert call.kwargs["language_hint"] is None
    assert mock_resolve.call_count == 2
    for call in mock_resolve.call_args_list:
        assert call.kwargs["explicit_language"] is None


def test_language_pinning_on_forces_response_language_but_stt_still_autodetects():
    """settings.voice_language_pinning=True: after turn 1 resolves to
    "fr", turn 2's resolve_turn call must receive explicit_language="fr"
    (the deliberate VRAM-avoidance override this setting exists for) --
    but the STT call must STILL pass language_hint=None on every turn,
    since forcing the ANSWER language is a separate decision from
    correctly transcribing what the user actually said."""
    fake_stt = MagicMock()
    fake_stt.transcribe.side_effect = [
        TranscriptChunk("Bonjour.", is_final=True, language="fr"),
        TranscriptChunk("خصك تلبس الكاسك.", is_final=True, language="ar"),
    ]
    fake_tts = MagicMock()
    fake_tts.synthesize.return_value = b"AUDIO"
    fake_tts.sample_rate = 22050
    turn_fr = _grounded_turn(response_lang="fr")
    turn_2 = _grounded_turn(response_lang="fr", query_lang="darija")

    fake_settings = MagicMock()
    fake_settings.voice_language_pinning = True
    fake_settings.voice_echo_mode = False
    fake_settings.vad_threshold = 500.0
    fake_settings.vad_hangover_ms = 400
    fake_settings.vad_min_speech_ms = 200
    fake_settings.vad_debug_log = False

    with patch("app.routers.voice.get_settings", return_value=fake_settings), \
         patch("app.routers.voice.get_stt_engine", return_value=fake_stt), \
         patch("app.routers.voice.get_tts_engine", return_value=fake_tts), \
         patch("app.routers.voice.EnergyEndpointer", _FakeEndpointer), \
         patch("app.routers.voice.resolve_turn", side_effect=[turn_fr, turn_2]) as mock_resolve, \
         patch("app.routers.voice.load_prior_turns", return_value=[]), \
         patch("app.routers.voice.stream_llm_response", return_value=iter(["Ok."])), \
         patch("app.routers.voice.persist_turn"):

        client = TestClient(app)
        with client.websocket_connect("/api/v1/voice/session") as ws:
            for _ in range(2):
                ws.send_bytes(b"\x00" * 640)
                ws.send_bytes(b"\x00" * 640)
                _drain_until(ws, lambda m: _is_text_type(m, "audio.end"))
            ws.send_text(json.dumps({"type": "end"}))

    for call in fake_stt.transcribe.call_args_list:
        assert call.kwargs["language_hint"] is None
    assert mock_resolve.call_args_list[0].kwargs["explicit_language"] is None
    assert mock_resolve.call_args_list[1].kwargs["explicit_language"] == "fr"


def test_echo_mode_never_calls_resolve_turn_or_the_llm():
    """settings.voice_echo_mode=True bypasses resolve_turn/RAG/the LLM
    entirely -- this is what lets the STT/VAD/TTS/WebSocket pipeline be
    exercised against a live browser mic with ZERO LLM VRAM loaded.
    Confirms the bypass is real (resolve_turn/stream_llm_response never
    called) and the reply comes back in the detected language."""
    fake_stt = MagicMock()
    fake_stt.transcribe.return_value = TranscriptChunk(
        "خصك تلبس الكاسك.", is_final=True, language="ar"
    )
    fake_tts = MagicMock()
    fake_tts.synthesize.return_value = b"ECHO-AUDIO"
    fake_tts.sample_rate = 22050

    fake_settings = MagicMock()
    fake_settings.voice_language_pinning = False
    fake_settings.voice_echo_mode = True
    fake_settings.vad_threshold = 500.0
    fake_settings.vad_hangover_ms = 400
    fake_settings.vad_min_speech_ms = 200
    fake_settings.vad_debug_log = False

    with patch("app.routers.voice.get_settings", return_value=fake_settings), \
         patch("app.routers.voice.get_stt_engine", return_value=fake_stt), \
         patch("app.routers.voice.get_tts_engine", return_value=fake_tts), \
         patch("app.routers.voice.EnergyEndpointer", _FakeEndpointer), \
         patch("app.routers.voice.resolve_turn") as mock_resolve, \
         patch("app.routers.voice.stream_llm_response") as mock_stream:

        client = TestClient(app)
        with client.websocket_connect("/api/v1/voice/session") as ws:
            ws.send_bytes(b"\x00" * 640)
            ws.send_bytes(b"\x00" * 640)
            messages = _drain_until(ws, lambda m: _is_text_type(m, "audio.end"))
            ws.send_text(json.dumps({"type": "end"}))

    mock_resolve.assert_not_called()
    mock_stream.assert_not_called()
    fake_tts.synthesize.assert_called_once()
    assert fake_tts.synthesize.call_args.kwargs["language"] == "darija"
    texts = [json.loads(m["text"]) for m in messages if "text" in m and m["text"] is not None]
    reply = next(t for t in texts if t["type"] == "answer.delta")
    assert "خصك تلبس الكاسك." in reply["text"]
    assert any("bytes" in m and m["bytes"] == b"ECHO-AUDIO" for m in messages)


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
