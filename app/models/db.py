"""
Shared Database Engine
──────────────────────
One SQLAlchemy Engine (and therefore one connection pool) for the whole
process.

Before this module, five call sites each built their OWN engine from the
same settings.database_url -- app/services/history.py, app/services/
sources.py, app/services/ingest_jobs.py, app/routers/ingest.py and
app/routers/video.py. SQLAlchemy's default QueuePool is pool_size=5 with
max_overflow=10, so a single process could hold up to 5 x 15 = 75
Postgres backends while never reusing a connection across modules: a
chat turn touching history + sources checked out connections from two
unrelated pools, and a Postgres `max_connections` of 100 (the default)
was one uvicorn worker away from being exhausted by a single-tenant demo.

Callers keep their existing module-level `_engine`/`_SessionLocal`
globals -- the test suites monkeypatch exactly those names to swap in an
in-memory SQLite engine (tests/test_sources.py, tests/test_ingest_jobs.py,
tests/test_ingest_router.py) -- they just initialize them FROM here
instead of building their own.

Pool sizing rationale: the serving path is FastAPI's sync-endpoint
threadpool (40 threads by default) plus one ingest worker thread
(app.services.ingest_queue), and every query here is a short OLTP
statement, so pool_size=10/max_overflow=10 covers the realistic
concurrency with a hard ceiling well under Postgres's default limit.
`pool_recycle` is set because a laptop Postgres container being
restarted (or an idle-connection reaper in front of a managed one)
otherwise hands back a dead socket on the next checkout; `pool_pre_ping`
already catches that, and recycling keeps pre-ping from being the only
line of defence.
"""
import logging
import threading

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from app.config import get_settings

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_lock = threading.Lock()

# connect_timeout=2 is the fail-open contract, not a test-speed knob:
# app.services.history / app.services.sources are on chat's request path
# and must fail FAST when Postgres is unreachable (multi-second IPv6-then-
# IPv4 fallbacks were measured on this machine) so the caller can degrade
# to stateless behaviour instead of hanging the request.
CONNECT_TIMEOUT_SECONDS = 2
POOL_SIZE = 10
MAX_OVERFLOW = 10
POOL_RECYCLE_SECONDS = 1800


def get_engine() -> Engine:
    """The process-wide Engine, built on first use (never at import time --
    importing any service must not require a reachable Postgres)."""
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                settings = get_settings()
                _engine = create_engine(
                    settings.database_url,
                    pool_pre_ping=True,
                    pool_size=POOL_SIZE,
                    max_overflow=MAX_OVERFLOW,
                    pool_recycle=POOL_RECYCLE_SECONDS,
                    connect_args={"connect_timeout": CONNECT_TIMEOUT_SECONDS},
                )
                logger.info(
                    "shared SQLAlchemy engine created (pool_size=%d, max_overflow=%d)",
                    POOL_SIZE, MAX_OVERFLOW,
                )
    return _engine


def dispose_engine() -> None:
    """Close every pooled connection. Called from app shutdown so uvicorn's
    reloader (and a test process) doesn't leave Postgres backends open."""
    global _engine
    with _lock:
        if _engine is not None:
            _engine.dispose()
            logger.info("shared SQLAlchemy engine disposed")
        _engine = None
