"""
Tests for app.services.llm's vLLM transport (_post_vllm, _call_vllm_generate,
_call_vllm_chat, _stream_vllm_chat, _vllm_base_url) and the backend-neutral
dispatchers (llm_generate, llm_chat, llm_stream_chat, resolve_model_name)
under llm_backend="vllm". No vLLM or Postgres required: urllib.request.urlopen
is monkeypatched, same convention as tests/test_llm_streaming.py and
tests/test_chat.py.

Settings are swapped via monkeypatch.setattr(llm_module, "get_settings", ...)
rather than mutating the real (lru_cache'd) Settings singleton, so nothing
here can leak into other test modules that assume llm_backend="ollama"
(today's default, and every other test file's implicit assumption).
"""
import io
import json
import types
import urllib.error

import pytest
from unittest.mock import patch

import app.services.llm as llm_module
from app.errors import GenerationError, LLMConnectionError
from app.services.llm import (
    _call_vllm_chat,
    _call_vllm_generate,
    _post_vllm,
    _stream_vllm_chat,
    _vllm_base_url,
    llm_chat,
    llm_generate,
    llm_stream_chat,
    render_conversation,
    resolve_model_name,
)


def _vllm_settings(**overrides) -> types.SimpleNamespace:
    """A minimal stand-in for Settings with the fields the vLLM code path
    reads, mirroring app/config.py's real defaults so these tests exercise
    production values rather than arbitrary ones."""
    base = dict(
        llm_backend="vllm",
        llm_base_url="http://vllm-darija:8101",
        llm_base_url_fr="http://vllm-french:8102",
        llm_model_darija="iblog-tutor-darija-awq",
        llm_model_fr="iblog-tutor-fr-awq",
        llm_max_tokens=1024,
        llm_max_concurrent=64,
        ollama_timeout_seconds=180,
        ollama_model="IBLOG_TUTOR:latest",
        ollama_model_fr="iblog-tutor-fr:latest",
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _vllm_backend(monkeypatch):
    """Every test in this module runs with llm_backend="vllm" by default --
    individual tests override via monkeypatch.setattr(llm_module,
    "get_settings", lambda: _vllm_settings(...)) when they need different
    field values."""
    monkeypatch.setattr(llm_module, "get_settings", lambda: _vllm_settings())


class _FakeStream:
    """Mimics urllib.request.urlopen's context-manager return value for a
    streaming response: iterates raw bytes lines. Same shape as
    tests/test_llm_streaming.py's _FakeStream, reused here for vLLM's SSE
    framing (data: {...}\\n\\n, terminated by data: [DONE])."""

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


def _sse(*texts, done=True) -> list[bytes]:
    lines = [
        f"data: {json.dumps({'choices': [{'text': t}]})}\n\n".encode("utf-8")
        for t in texts
    ]
    if done:
        lines.append(b"data: [DONE]\n\n")
    return lines


def _http_error(code: int, body: bytes = b"") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url="http://x", code=code, msg="err", hdrs=None, fp=io.BytesIO(body))


def _fake_completion_response(text: str) -> "_FakeJsonResponse":
    return _FakeJsonResponse({"choices": [{"text": text}], "usage": {"prompt_tokens": 42}})


class _FakeJsonResponse:
    """Mimics urlopen's return value for a non-streaming /v1/completions
    call: a context manager whose .read() gives the JSON body _post_vllm
    decodes."""

    def __init__(self, obj: dict):
        self._body = json.dumps(obj).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


# -- _vllm_base_url: routes by served model name, not a single shared URL --

def test_base_url_routes_french_model_to_french_port():
    assert _vllm_base_url("iblog-tutor-fr-awq") == "http://vllm-french:8102"


def test_base_url_routes_darija_model_to_darija_port():
    assert _vllm_base_url("iblog-tutor-darija-awq") == "http://vllm-darija:8101"


def test_base_url_defaults_to_darija_port_for_an_unrecognized_model():
    """Anything that isn't exactly the French served-model name falls back
    to the primary llm_base_url -- mirrors resolve_model_name's own
    "anything else treated as Darija" convention."""
    assert _vllm_base_url("some-other-model") == "http://vllm-darija:8101"


# -- _post_vllm / _call_vllm_generate / _call_vllm_chat: payload shape ------

def test_call_vllm_generate_sends_rendered_prompt_not_raw_strings():
    with patch("app.services.llm.urllib.request.urlopen",
               return_value=_fake_completion_response("Bonjour.")) as mock_urlopen:
        result = _call_vllm_generate("iblog-tutor-fr-awq", "Quelle heure est-il ?", "Tu es un tuteur.")
    assert result == "Bonjour."
    sent = mock_urlopen.call_args[0][0]
    payload = json.loads(sent.data.decode("utf-8"))
    expected_prompt = render_conversation(
        [{"role": "system", "content": "Tu es un tuteur."},
         {"role": "user", "content": "Quelle heure est-il ?"}]
    )
    assert payload["prompt"] == expected_prompt


def test_call_vllm_generate_sets_max_tokens_explicitly():
    """vLLM's /v1/completions defaults max_tokens to 16 (see app/config.py's
    llm_max_tokens comment) -- omitting this silently truncates every
    answer to a sentence fragment with no error, so it must always be in
    the payload."""
    with patch("app.services.llm.urllib.request.urlopen",
               return_value=_fake_completion_response("ok")) as mock_urlopen:
        _call_vllm_generate("iblog-tutor-darija-awq", "p", "s")
    payload = json.loads(mock_urlopen.call_args[0][0].data.decode("utf-8"))
    assert payload["max_tokens"] == 1024


def test_call_vllm_generate_sends_stop_sequences():
    """Ollama gets these for free from the Modelfile's PARAMETER stop; vLLM
    never sees the Modelfile, so they must be explicit per request."""
    with patch("app.services.llm.urllib.request.urlopen",
               return_value=_fake_completion_response("ok")) as mock_urlopen:
        _call_vllm_generate("iblog-tutor-darija-awq", "p", "s")
    payload = json.loads(mock_urlopen.call_args[0][0].data.decode("utf-8"))
    assert payload["stop"] == ["<end_of_turn>", "<start_of_turn>"]


def test_call_vllm_generate_never_sends_keep_alive():
    """vLLM has no keep_alive analogue -- a served model is always
    resident. Sending Ollama's field would be silently ignored by vLLM but
    signals a copy-paste transport bug if it ever appears."""
    with patch("app.services.llm.urllib.request.urlopen",
               return_value=_fake_completion_response("ok")) as mock_urlopen:
        _call_vllm_generate("iblog-tutor-darija-awq", "p", "s")
    payload = json.loads(mock_urlopen.call_args[0][0].data.decode("utf-8"))
    assert "keep_alive" not in payload


def test_call_vllm_chat_omits_structured_outputs_when_no_schema():
    with patch("app.services.llm.urllib.request.urlopen",
               return_value=_fake_completion_response("ok")) as mock_urlopen:
        _call_vllm_chat("iblog-tutor-darija-awq", [{"role": "user", "content": "hi"}])
    payload = json.loads(mock_urlopen.call_args[0][0].data.decode("utf-8"))
    assert "structured_outputs" not in payload


def test_call_vllm_chat_sends_structured_outputs_not_guided_json():
    """guided_json was removed in vLLM v0.12.0 (docs.vllm.ai); the current
    field is structured_outputs: {"json": <schema>}. rev 1 of this
    migration shipped guided_json, which v0.29.0 (the pinned image) simply
    ignores -- structured output would silently stop being enforced."""
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    with patch("app.services.llm.urllib.request.urlopen",
               return_value=_fake_completion_response("ok")) as mock_urlopen:
        _call_vllm_chat("iblog-tutor-darija-awq", [{"role": "user", "content": "hi"}],
                         json_schema=schema)
    payload = json.loads(mock_urlopen.call_args[0][0].data.decode("utf-8"))
    assert "guided_json" not in payload
    assert payload["structured_outputs"] == {"json": schema}


def test_call_vllm_chat_routes_french_request_to_french_port():
    with patch("app.services.llm.urllib.request.urlopen",
               return_value=_fake_completion_response("ok")) as mock_urlopen:
        _call_vllm_chat("iblog-tutor-fr-awq", [{"role": "user", "content": "bonjour"}])
    sent_url = mock_urlopen.call_args[0][0].full_url
    assert sent_url == "http://vllm-french:8102/v1/completions"


def test_call_vllm_chat_routes_darija_request_to_darija_port():
    with patch("app.services.llm.urllib.request.urlopen",
               return_value=_fake_completion_response("ok")) as mock_urlopen:
        _call_vllm_chat("iblog-tutor-darija-awq", [{"role": "user", "content": "مرحبا"}])
    sent_url = mock_urlopen.call_args[0][0].full_url
    assert sent_url == "http://vllm-darija:8101/v1/completions"


def test_call_vllm_generate_empty_choices_raises_generation_error():
    with patch("app.services.llm.urllib.request.urlopen",
               return_value=_FakeJsonResponse({"choices": []})):
        with pytest.raises(GenerationError):
            _call_vllm_generate("iblog-tutor-darija-awq", "p", "s")


# -- _post_vllm: error mapping (retry, 404, 400 context-length, connection) -

def test_post_vllm_retries_transient_503_then_succeeds():
    responses = [_http_error(503), _fake_completion_response("recovered")]

    def _side_effect(req, timeout=None):
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    with patch("app.services.llm.urllib.request.urlopen", side_effect=_side_effect), \
         patch("app.services.llm.time.sleep"):
        result = _post_vllm({"model": "iblog-tutor-darija-awq", "prompt": "p"})
    assert result["choices"][0]["text"] == "recovered"


def test_post_vllm_404_names_the_missing_model():
    with patch("app.services.llm.urllib.request.urlopen",
               side_effect=_http_error(404, b"model not found")):
        with pytest.raises(GenerationError, match="missing-model"):
            _post_vllm({"model": "missing-model", "prompt": "p"})


def test_post_vllm_400_context_length_is_named_not_silently_truncated():
    """Assumption 5 (plan doc): a vLLM 400 context-length error must become
    an explicit GenerationError, never a silent truncation the caller can't
    detect."""
    body = b'{"error": "This model\'s maximum context length is 8192 tokens"}'
    with patch("app.services.llm.urllib.request.urlopen",
               side_effect=_http_error(400, body)):
        with pytest.raises(GenerationError, match="max_model_len|maximum context length"):
            _post_vllm({"model": "iblog-tutor-darija-awq", "prompt": "p"})


def test_post_vllm_connection_failure_raises_llm_connection_error_with_correct_port():
    with patch("app.services.llm.urllib.request.urlopen",
               side_effect=urllib.error.URLError("refused")), \
         patch("app.services.llm.time.sleep"):
        with pytest.raises(LLMConnectionError) as exc_info:
            _post_vllm({"model": "iblog-tutor-fr-awq", "prompt": "p"})
    assert "8102" in str(exc_info.value) or "vllm-french" in str(exc_info.value)


# -- _stream_vllm_chat: SSE framing, ordering, error semantics -------------

def test_stream_vllm_chat_yields_deltas_in_order_and_stops_at_done():
    lines = _sse("Bon", "jour", ", le casque.")
    with patch("app.services.llm.urllib.request.urlopen", return_value=_FakeStream(lines)):
        deltas = list(_stream_vllm_chat("iblog-tutor-fr-awq", [{"role": "user", "content": "hi"}]))
    assert "".join(deltas) == "Bonjour, le casque."


def test_stream_vllm_chat_skips_malformed_sse_line():
    lines = [b"data: not json at all\n\n"] + _sse("ok")
    with patch("app.services.llm.urllib.request.urlopen", return_value=_FakeStream(lines)):
        deltas = list(_stream_vllm_chat("iblog-tutor-darija-awq", []))
    assert "".join(deltas) == "ok"


def test_stream_vllm_chat_empty_stream_raises_generation_error():
    lines = [b"data: [DONE]\n\n"]
    with patch("app.services.llm.urllib.request.urlopen", return_value=_FakeStream(lines)):
        with pytest.raises(GenerationError):
            list(_stream_vllm_chat("iblog-tutor-darija-awq", []))


def test_stream_vllm_chat_urlerror_before_any_content_raises_connection_error():
    with patch("app.services.llm.urllib.request.urlopen",
               side_effect=urllib.error.URLError("refused")):
        with pytest.raises(LLMConnectionError):
            list(_stream_vllm_chat("iblog-tutor-darija-awq", []))


def test_stream_vllm_chat_mid_stream_drop_after_content_raises_generation_error():
    """Mirrors _stream_ollama_chat's semantics verbatim (per the module
    docstring): once real content has reached the caller, a dropped
    connection is a distinct GenerationError, not a connection error the
    caller might treat as 'nothing happened yet'."""
    lines = _sse("Bonj", done=False)
    stream = _FakeStream(lines, raise_after=urllib.error.URLError("reset"))
    with patch("app.services.llm.urllib.request.urlopen", return_value=stream):
        deltas = []
        with pytest.raises(GenerationError):
            for delta in _stream_vllm_chat("iblog-tutor-darija-awq", []):
                deltas.append(delta)
        assert deltas == ["Bonj"]


def test_stream_vllm_chat_http_404_names_the_model():
    with patch("app.services.llm.urllib.request.urlopen",
               side_effect=_http_error(404, b"model not found")):
        with pytest.raises(GenerationError, match="missing-model"):
            list(_stream_vllm_chat("missing-model", []))


def test_stream_vllm_chat_releases_semaphore_permit_on_early_generator_close():
    """voice.py's cancel_flag abandons the generator mid-stream via
    GeneratorExit -- the semaphore permit acquired at the top of
    _stream_vllm_chat must still be released via the `finally` block, or a
    cancelled voice turn permanently leaks a concurrency slot."""
    semaphore = llm_module._get_vllm_semaphore()
    before = semaphore._value
    lines = _sse("first chunk", "second chunk", done=False)
    with patch("app.services.llm.urllib.request.urlopen", return_value=_FakeStream(lines)):
        gen = _stream_vllm_chat("iblog-tutor-darija-awq", [])
        next(gen)
        assert semaphore._value == before - 1
        gen.close()
    assert semaphore._value == before


# -- resolve_model_name / llm_generate / llm_chat / llm_stream_chat: dispatch -

def test_resolve_model_name_vllm_backend_french():
    assert resolve_model_name("fr") == "iblog-tutor-fr-awq"


def test_resolve_model_name_vllm_backend_darija():
    assert resolve_model_name("darija") == "iblog-tutor-darija-awq"


def test_resolve_model_name_ollama_backend_unaffected(monkeypatch):
    """The vLLM migration is additive -- flipping llm_backend back to
    "ollama" must still resolve the pre-migration ollama_model* names."""
    monkeypatch.setattr(llm_module, "get_settings",
                         lambda: _vllm_settings(llm_backend="ollama"))
    assert resolve_model_name("fr") == "iblog-tutor-fr:latest"
    assert resolve_model_name("darija") == "IBLOG_TUTOR:latest"


def test_llm_generate_dispatches_to_vllm_when_backend_is_vllm():
    with patch("app.services.llm.urllib.request.urlopen",
               return_value=_fake_completion_response("réponse")) as mock_urlopen:
        result = llm_generate("iblog-tutor-fr-awq", "question", "system")
    assert result == "réponse"
    sent_url = mock_urlopen.call_args[0][0].full_url
    assert sent_url == "http://vllm-french:8102/v1/completions"


def test_llm_chat_passes_format_schema_through_as_structured_outputs():
    schema = {"type": "array", "minItems": 4, "maxItems": 4}
    with patch("app.services.llm.urllib.request.urlopen",
               return_value=_fake_completion_response("[]")) as mock_urlopen:
        llm_chat("iblog-tutor-darija-awq", [{"role": "user", "content": "quiz"}],
                 format_schema=schema)
    payload = json.loads(mock_urlopen.call_args[0][0].data.decode("utf-8"))
    assert payload["structured_outputs"] == {"json": schema}


def test_llm_stream_chat_dispatches_to_vllm_sse_framing():
    lines = _sse("Sa", "lam")
    with patch("app.services.llm.urllib.request.urlopen", return_value=_FakeStream(lines)):
        deltas = list(llm_stream_chat("iblog-tutor-darija-awq", [{"role": "user", "content": "hi"}]))
    assert "".join(deltas) == "Salam"
