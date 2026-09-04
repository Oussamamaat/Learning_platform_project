"""
Make "no Postgres running" fail FAST instead of slow.

Why this file exists (it is the only conftest.py in the repo -- mocking is
otherwise per-file, deliberately):

Most tests here are written to need no Postgres at all: the fail-open paths in
app.services.history / app.services.sources return their documented sentinels
(None, []) when the DB is unreachable, and the assertions hold either way. That
works, but only if "unreachable" is *quick*.

On this Windows dev box it is not. Nothing listens on 5432 and the SYNs are
silently dropped rather than refused, so libpq burns its full connect_timeout
once per address family -- measured at 4.72s per connection attempt
(2s for ::1 + 2s for 127.0.0.1 + overhead). app/routers/chat.py makes four to
six such calls per turn (history.get_pinned / load_window / get_language_state /
append_exchange / pin_context, plus source_service.active_sources_and_version),
so ONE chat test costs 20-30s and the suite crawls for tens of minutes with
every process sitting at 0% CPU in a socket wait. That is what made `pytest`
look hung rather than slow.

The fix is not to lower app.models.db.CONNECT_TIMEOUT_SECONDS -- 2s is a
deliberate production value (see that module: it is the fail-open contract on
chat's request path, sized for a real network). Instead, when Postgres is
genuinely unreachable, replace psycopg2.connect with one that raises the same
OperationalError immediately. Every fail-open handler already catches exactly
that, so behaviour is identical to today -- just microseconds instead of
seconds.

Deliberately conditional: the probe below runs once per session, and if a real
Postgres IS reachable (Docker up, CI, the ingest/e2e work) nothing is patched
and tests hit the real database exactly as before. Tests that want a working DB
monkeypatch their own in-memory SQLite engine (tests/test_sources.py,
test_ingest_jobs.py, test_ingest_router.py) and are unaffected either way --
different dialect, different connect path.
"""
import socket
from urllib.parse import urlsplit

import pytest


def _tcp_reachable(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except Exception:
        return False


def _postgres_is_reachable() -> bool:
    """One short TCP probe, once per session. Never raises: any failure to
    even parse the URL is treated as 'not reachable', which only ever makes
    the suite faster, never wrong."""
    try:
        from app.config import get_settings

        parts = urlsplit(get_settings().database_url)
        host = parts.hostname or "localhost"
        port = parts.port or 5432
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def postgres_reachable() -> bool:
    return _postgres_is_reachable()


@pytest.fixture(autouse=True)
def _fail_fast_when_no_postgres(postgres_reachable, monkeypatch):
    """Autouse: with no Postgres listening, make psycopg2.connect raise
    instantly instead of waiting out the TCP timeout.

    The raised type is psycopg2.OperationalError -- the same class libpq
    raises on a refused/timed-out connection -- so every `except` that
    already handles an unreachable Postgres keeps handling it unchanged.
    """
    if postgres_reachable:
        return

    import psycopg2

    def _refuse(*args, **kwargs):
        raise psycopg2.OperationalError(
            "connection refused: no Postgres reachable (tests/conftest.py "
            "fail-fast; start the db service to run against a real database)"
        )

    monkeypatch.setattr(psycopg2, "connect", _refuse)


@pytest.fixture(scope="session")
def ollama_target() -> tuple[str, int]:
    from app.config import get_settings

    parts = urlsplit(get_settings().ollama_base_url)
    return (parts.hostname or "localhost", parts.port or 11434)


@pytest.fixture(scope="session")
def ollama_reachable(ollama_target) -> bool:
    return _tcp_reachable(*ollama_target)


@pytest.fixture(autouse=True)
def _fail_fast_when_no_ollama(ollama_reachable, ollama_target, monkeypatch):
    """Same treatment as Postgres above, for the same reason: an unreachable
    Ollama costs 4.11s per call here (measured), and the generation paths are
    reached by far more tests than the ones that bother to stub urlopen.

    Scoped to the Ollama host:port only -- any other URL still goes to the
    real urlopen, so this can never silently swallow a test that genuinely
    means to fetch something else. Tests that stub
    app.services.llm.urllib.request.urlopen themselves patch the same module
    attribute from inside the test body, so their stub still wins for their
    duration and this wrapper is restored afterwards.
    """
    if ollama_reachable:
        return

    import urllib.error
    import urllib.request

    host, port = ollama_target
    real_urlopen = urllib.request.urlopen

    def _maybe_refuse(url, *args, **kwargs):
        target = getattr(url, "full_url", url)
        if isinstance(target, str) and f"{host}:{port}" in target:
            raise urllib.error.URLError(
                "connection refused: no Ollama reachable (tests/conftest.py "
                "fail-fast; start ollama serve to run against a real model)"
            )
        return real_urlopen(url, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", _maybe_refuse)
