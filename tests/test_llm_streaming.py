"""
Tests for app.services.llm's streaming path (_stream_ollama_chat,
stream_llm_response) -- added for the voice pipeline
(app/routers/voice.py). No Ollama or Postgres required: urllib.request.
urlopen is monkeypatched, same convention as tests/test_chat.py.
"""
import io
import json
import urllib.error

import pytest
from unittest.mock import patch

from app.errors import GenerationError, OllamaConnectionError
from app.services.llm import _stream_ollama_chat, stream_llm_response


class _FakeStream:
    """Mimics the object urllib.request.urlopen returns for a streaming
    response: a context manager that iterates raw bytes lines (Ollama's
    NDJSON-over-HTTP shape)."""

    def __init__(self, lines: list[bytes], *, raise_after: Exception = None):
        self._lines = lines
        self._raise_after = raise_after

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        for line in self._lines:
            yield line
        if self._raise_after is not None:
            raise self._raise_after


def _ndjson(*objs) -> list[bytes]:
    return [json.dumps(o).encode("utf-8") + b"\n" for o in objs]


def _http_error(code: int, body: bytes = b"") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url="http://x", code=code, msg="err", hdrs=None, fp=io.BytesIO(body))


def test_stream_yields_deltas_in_order_and_stops_at_done():
    lines = _ndjson(
        {"message": {"content": "Bonjour"}, "done": False},
        {"message": {"content": ", le"}, "done": False},
        {"message": {"content": " casque."}, "done": False},
        {"message": {"content": ""}, "done": True},
    )
    with patch("app.services.llm.urllib.request.urlopen", return_value=_FakeStream(lines)):
        deltas = list(_stream_ollama_chat("some-model", [{"role": "user", "content": "hi"}]))
    assert "".join(deltas) == "Bonjour, le casque."


def test_malformed_ndjson_line_is_skipped_not_fatal():
    lines = [b"not json at all\n"] + _ndjson(
        {"message": {"content": "ok"}, "done": True},
    )
    with patch("app.services.llm.urllib.request.urlopen", return_value=_FakeStream(lines)):
        deltas = list(_stream_ollama_chat("some-model", []))
    assert "".join(deltas) == "ok"


def test_empty_stream_raises_generation_error():
    lines = _ndjson({"message": {"content": ""}, "done": True})
    with patch("app.services.llm.urllib.request.urlopen", return_value=_FakeStream(lines)):
        with pytest.raises(GenerationError):
            list(_stream_ollama_chat("some-model", []))


def test_explicit_error_chunk_raises_generation_error():
    lines = _ndjson({"error": "model requires more system memory"})
    with patch("app.services.llm.urllib.request.urlopen", return_value=_FakeStream(lines)):
        with pytest.raises(GenerationError):
            list(_stream_ollama_chat("some-model", []))


def test_urlerror_before_any_content_raises_connection_error():
    with patch(
        "app.services.llm.urllib.request.urlopen",
        side_effect=urllib.error.URLError("refused"),
    ):
        with pytest.raises(OllamaConnectionError):
            list(_stream_ollama_chat("some-model", []))


def test_mid_stream_drop_after_content_raises_generation_error_not_connection_error():
    """Once real content has already reached the caller, a dropped
    connection must surface as a distinct GenerationError (the caller may
    have already spoken/displayed the partial text) rather than the
    generic OllamaConnectionError a caller might treat as 'nothing
    happened yet'."""
    lines = _ndjson({"message": {"content": "Bonj"}, "done": False})
    stream = _FakeStream(lines, raise_after=urllib.error.URLError("reset"))
    with patch("app.services.llm.urllib.request.urlopen", return_value=stream):
        deltas = []
        with pytest.raises(GenerationError):
            for delta in _stream_ollama_chat("some-model", []):
                deltas.append(delta)
        assert deltas == ["Bonj"]


def test_http_404_names_the_model():
    with patch(
        "app.services.llm.urllib.request.urlopen",
        side_effect=_http_error(404, b"model not found"),
    ):
        with pytest.raises(GenerationError, match="missing-model"):
            list(_stream_ollama_chat("missing-model", []))


def test_stream_llm_response_routes_french_to_french_model():
    lines = _ndjson({"message": {"content": "Bonjour."}, "done": True})
    with patch("app.services.llm.urllib.request.urlopen", return_value=_FakeStream(lines)) as mock_urlopen:
        result = "".join(stream_llm_response(
            query="Bonjour", context="contexte", domain="industrial", language="fr",
        ))
    assert result == "Bonjour."
    sent_request = mock_urlopen.call_args[0][0]
    payload = json.loads(sent_request.data.decode("utf-8"))
    assert payload["stream"] is True
    from app.config import get_settings
    assert payload["model"] == get_settings().ollama_model_fr


def test_stream_llm_response_routes_darija_to_darija_model():
    lines = _ndjson({"message": {"content": "مرحبا."}, "done": True})
    with patch("app.services.llm.urllib.request.urlopen", return_value=_FakeStream(lines)) as mock_urlopen:
        list(stream_llm_response(
            query="السلام", context="سياق", domain="industrial", language="darija",
        ))
    sent_request = mock_urlopen.call_args[0][0]
    payload = json.loads(sent_request.data.decode("utf-8"))
    from app.config import get_settings
    assert payload["model"] == get_settings().ollama_model


def test_stream_llm_response_does_not_inject_citations_into_the_stream():
    """Deliberate divergence from generate_llm_response, documented on
    stream_llm_response itself: citation injection is a post-hoc rewrite
    over the complete answer and has no incremental equivalent, so the
    streamed text must be exactly what the model said, un-rewritten."""
    context = "Selon Article 8, le port du casque est obligatoire."
    lines = _ndjson({"message": {"content": "Le port du casque est obligatoire."}, "done": True})
    with patch("app.services.llm.urllib.request.urlopen", return_value=_FakeStream(lines)):
        result = "".join(stream_llm_response(
            query="Que dit le texte ?", context=context, domain="industrial", language="fr",
        ))
    assert result == "Le port du casque est obligatoire."
