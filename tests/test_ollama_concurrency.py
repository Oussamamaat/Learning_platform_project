"""
Tests for app.services.llm's Ollama concurrency semaphore
(POST_LEASE_MVP_SPRINT_PLAN.md item 5 / ADR 0008). Before this, there was
NO limit on concurrent in-flight Ollama requests anywhere in the codebase
(docs/architecture/cloud-scaling-plan.md #3) -- N concurrent users meant N
concurrent Ollama requests. These tests prove the semaphore actually bounds
concurrency, not just that it's constructed with the right number.

No real Ollama server needed: urllib.request.urlopen is monkeypatched with
a fake that blocks briefly and records how many calls were in flight at
once, which is what a real Ollama server under load would experience.
"""
import threading
import time
from contextlib import contextmanager
from unittest.mock import patch

import pytest

import app.services.llm as llm_module


@pytest.fixture(autouse=True)
def _reset_semaphore_singleton():
    """The semaphore is a module-level singleton, built lazily from
    get_settings().ollama_max_concurrent on first use (app.services.llm.
    _get_ollama_semaphore) -- reset it before/after each test so a
    monkeypatched setting actually takes effect instead of reusing a
    semaphore sized for a different test."""
    llm_module._ollama_semaphore = None
    yield
    llm_module._ollama_semaphore = None


class _ConcurrencyTracker:
    """Records the peak number of simultaneously in-flight fake requests."""

    def __init__(self, hold_seconds: float = 0.1):
        self._lock = threading.Lock()
        self._in_flight = 0
        self.peak = 0
        self.hold_seconds = hold_seconds

    @contextmanager
    def track(self):
        with self._lock:
            self._in_flight += 1
            self.peak = max(self.peak, self._in_flight)
        try:
            time.sleep(self.hold_seconds)
            yield
        finally:
            with self._lock:
                self._in_flight -= 1


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _make_fake_urlopen(tracker: _ConcurrencyTracker):
    def _fake_urlopen(req, timeout=None):
        with tracker.track():
            pass
        return _FakeResponse(b'{"response": "ok"}')
    return _fake_urlopen


def test_post_ollama_bounds_concurrent_requests_to_configured_limit():
    """5 threads call _post_ollama simultaneously with
    ollama_max_concurrent=2 -- the observed peak concurrency must never
    exceed 2, proving the semaphore actually blocks the 3rd+ caller rather
    than just being constructed and ignored."""
    tracker = _ConcurrencyTracker(hold_seconds=0.15)
    fake_settings = llm_module.get_settings()
    with patch.object(fake_settings, "ollama_max_concurrent", 2), \
         patch("app.services.llm.get_settings", return_value=fake_settings), \
         patch("urllib.request.urlopen", side_effect=_make_fake_urlopen(tracker)):
        threads = [
            threading.Thread(target=llm_module._post_ollama, args=("/api/generate", {"model": "x"}))
            for _ in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

    assert tracker.peak <= 2, f"expected peak concurrency <= 2, observed {tracker.peak}"
    assert tracker.peak >= 2, "expected the semaphore to actually allow 2 concurrent requests, not serialize to 1"


def test_post_ollama_releases_semaphore_after_success_so_later_calls_proceed():
    """A slot freed after one call completes must be usable by the next
    caller -- the semaphore must not leak permits."""
    tracker = _ConcurrencyTracker(hold_seconds=0.05)
    fake_settings = llm_module.get_settings()
    with patch.object(fake_settings, "ollama_max_concurrent", 1), \
         patch("app.services.llm.get_settings", return_value=fake_settings), \
         patch("urllib.request.urlopen", side_effect=_make_fake_urlopen(tracker)):
        for _ in range(4):
            llm_module._post_ollama("/api/generate", {"model": "x"})

    assert tracker.peak == 1


def test_post_ollama_releases_semaphore_on_error():
    """A failed request must still release its slot -- otherwise one
    Ollama error would permanently shrink the concurrency budget."""
    fake_settings = llm_module.get_settings()

    def _raise(req, timeout=None):
        raise ConnectionError("simulated failure")

    with patch.object(fake_settings, "ollama_max_concurrent", 1), \
         patch("app.services.llm.get_settings", return_value=fake_settings), \
         patch("urllib.request.urlopen", side_effect=_raise):
        for _ in range(3):
            with pytest.raises(Exception):
                llm_module._post_ollama("/api/generate", {"model": "x"}, timeout=1)

    # If the semaphore leaked (never released on error), this acquire would
    # be the 4th attempted permit against a budget of 1 that was never
    # freed -- assert it's still acquirable immediately (non-blocking).
    sem = llm_module._get_ollama_semaphore()
    acquired = sem.acquire(blocking=False)
    assert acquired, "semaphore permit was not released after an error"
    sem.release()
