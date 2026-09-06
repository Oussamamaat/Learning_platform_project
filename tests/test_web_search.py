"""Tests for app/services/web_search.py -- the opt-in web-search fallback
engine used by chat.py's refusal gate. No network calls: urlopen is
monkeypatched, same convention as tests/test_chat.py uses for Ollama.
"""
import json
from unittest.mock import patch

import pytest

from app.services.web_search import (
    NullWebSearchEngine,
    TavilyWebSearchEngine,
    WebResult,
    get_web_search_engine,
)


@pytest.fixture(autouse=True)
def _clear_engine_cache():
    """get_web_search_engine() is @lru_cache'd for the process lifetime
    (consistent with settings, which are also cached) -- tests that patch
    settings.web_search_engine/tavily_api_key need a fresh read each time,
    not the previous test's cached engine instance."""
    get_web_search_engine.cache_clear()
    yield
    get_web_search_engine.cache_clear()


def test_null_engine_returns_nothing():
    assert NullWebSearchEngine().search("anything", max_results=3) == []


def test_get_web_search_engine_defaults_to_null():
    with patch("app.services.web_search.get_settings") as mock_settings:
        mock_settings.return_value.web_search_engine = "none"
        engine = get_web_search_engine()
    assert isinstance(engine, NullWebSearchEngine)


def test_get_web_search_engine_tavily_without_key_falls_back_to_null():
    """A misconfiguration (engine selected, key never set) must degrade to
    'no web fallback', not crash the whole chat turn."""
    with patch("app.services.web_search.get_settings") as mock_settings:
        mock_settings.return_value.web_search_engine = "tavily"
        mock_settings.return_value.tavily_api_key = None
        engine = get_web_search_engine()
    assert isinstance(engine, NullWebSearchEngine)


def test_get_web_search_engine_tavily_with_key():
    with patch("app.services.web_search.get_settings") as mock_settings:
        mock_settings.return_value.web_search_engine = "tavily"
        mock_settings.return_value.tavily_api_key = "test-key-123"
        mock_settings.return_value.web_search_timeout_seconds = 12.0
        engine = get_web_search_engine()
    assert isinstance(engine, TavilyWebSearchEngine)


class _FakeHttpResponse:
    def __init__(self, body: dict):
        self._body = json.dumps(body).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_tavily_engine_parses_results():
    body = {
        "results": [
            {"title": "Result One", "url": "https://example.com/1", "content": "snippet one"},
            {"title": "Result Two", "url": "https://example.com/2", "content": "snippet two"},
        ]
    }
    with patch("app.services.web_search.urllib.request.urlopen", return_value=_FakeHttpResponse(body)):
        results = TavilyWebSearchEngine("fake-key", timeout_seconds=5.0).search("test query", max_results=3)
    assert results == [
        WebResult(title="Result One", url="https://example.com/1", snippet="snippet one"),
        WebResult(title="Result Two", url="https://example.com/2", snippet="snippet two"),
    ]


def test_tavily_engine_respects_max_results():
    body = {"results": [{"title": f"R{i}", "url": f"https://x.com/{i}", "content": ""} for i in range(10)]}
    with patch("app.services.web_search.urllib.request.urlopen", return_value=_FakeHttpResponse(body)):
        results = TavilyWebSearchEngine("fake-key", timeout_seconds=5.0).search("q", max_results=2)
    assert len(results) == 2


def test_tavily_engine_skips_results_with_no_url():
    body = {"results": [{"title": "No URL", "content": "x"}, {"title": "Has URL", "url": "https://x.com", "content": "y"}]}
    with patch("app.services.web_search.urllib.request.urlopen", return_value=_FakeHttpResponse(body)):
        results = TavilyWebSearchEngine("fake-key", timeout_seconds=5.0).search("q", max_results=5)
    assert len(results) == 1
    assert results[0].url == "https://x.com"


def test_tavily_engine_fails_open_on_network_error():
    """A network/timeout failure must return [] (chat.py's caller then
    falls through to the ORIGINAL deterministic refusal), never raise into
    a 500 on top of an already-unanswerable question."""
    import urllib.error

    with patch(
        "app.services.web_search.urllib.request.urlopen",
        side_effect=urllib.error.URLError("connection refused"),
    ):
        results = TavilyWebSearchEngine("fake-key", timeout_seconds=5.0).search("q", max_results=3)
    assert results == []


def test_tavily_engine_fails_open_on_http_error():
    import urllib.error
    import io

    err = urllib.error.HTTPError(
        "https://api.tavily.com/search", 401, "Unauthorized", {}, io.BytesIO(b"bad key")
    )
    with patch("app.services.web_search.urllib.request.urlopen", side_effect=err):
        results = TavilyWebSearchEngine("fake-key", timeout_seconds=5.0).search("q", max_results=3)
    assert results == []


def test_tavily_engine_fails_open_on_malformed_json():
    class BadResponse:
        def read(self):
            return b"not json"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    with patch("app.services.web_search.urllib.request.urlopen", return_value=BadResponse()):
        results = TavilyWebSearchEngine("fake-key", timeout_seconds=5.0).search("q", max_results=3)
    assert results == []
