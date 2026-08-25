"""
Tests for the performance/resource-lifecycle wiring added in the
2026-08-24 RAG + ingestion audit.

Every test here pins a behaviour that is invisible in a functional test --
the pipeline produced correct answers before these changes and produces
correct answers after them. What changed is how much work, memory and how
many OS resources it takes to do that, and those are exactly the
properties that silently regress. So each test asserts the mechanism, not
the timing: "the connection went back to the pool", "the model was loaded
once", "the pipe was closed" -- never "it took under N ms", which would be
flaky on a shared machine and would tell us nothing about why.
"""
import threading

import pytest

from app.services import ingestion


# --- psycopg2 connection pooling (app.services.ingestion) -----------------
#
# The pgvector path used to open and close a fresh connection per search,
# and a chat turn issues two searches (the tier-2 domain vote, then the
# domain-scoped retrieval). Measured against this project's own Postgres:
# ~31ms per connect, i.e. ~62ms of pure connection setup on every message
# before a single vector was compared.


class _FakeRawConnection:
    """Minimal stand-in for a psycopg2 connection: enough to observe what
    _PooledConnection does to it on close()."""

    def __init__(self):
        self.closed = 0
        self.rolled_back = 0
        self.committed = 0
        self.transaction_status = 0  # TRANSACTION_STATUS_IDLE

    def get_transaction_status(self):
        return self.transaction_status

    def rollback(self):
        self.rolled_back += 1
        self.transaction_status = 0

    def commit(self):
        self.committed += 1

    def cursor(self, *a, **kw):
        return object()


class _FakePool:
    def __init__(self):
        self.returned = []

    def putconn(self, conn, close=False):
        self.returned.append((conn, close))


def test_pooled_close_returns_the_connection_instead_of_dropping_it():
    """The whole point of the wrapper: existing call sites are all written
    as `conn = get_db_connection(...)` / `finally: conn.close()`, and that
    close must now mean 'give it back', not 'tear down the socket'."""
    raw, pool = _FakeRawConnection(), _FakePool()
    conn = ingestion._PooledConnection(raw, pool)

    conn.close()

    assert pool.returned == [(raw, False)], "connection was not returned to the pool"
    assert raw.closed == 0, "the underlying socket was torn down anyway"


def test_pooled_close_rolls_back_an_open_transaction_before_reuse():
    """A connection handed back mid-transaction poisons the NEXT borrower:
    every statement it runs fails with InFailedSqlTransaction even though
    nothing is wrong with it. This is the classic way 'we added pooling and
    everything broke' happens, so it is asserted rather than assumed."""
    raw, pool = _FakeRawConnection(), _FakePool()
    raw.transaction_status = 3  # TRANSACTION_STATUS_INERROR
    conn = ingestion._PooledConnection(raw, pool)

    conn.close()

    assert raw.rolled_back == 1
    assert pool.returned == [(raw, False)]


def test_pooled_close_discards_a_dead_connection_rather_than_pooling_it():
    raw, pool = _FakeRawConnection(), _FakePool()
    raw.closed = 1
    conn = ingestion._PooledConnection(raw, pool)

    conn.close()

    assert pool.returned == [(raw, True)], "a dead connection must not re-enter the pool"


def test_pooled_close_is_idempotent():
    """A double close (a `with` block inside a caller's own finally) must
    not return the same connection to the pool twice -- two borrowers would
    then hold the same socket."""
    raw, pool = _FakeRawConnection(), _FakePool()
    conn = ingestion._PooledConnection(raw, pool)

    conn.close()
    conn.close()

    assert len(pool.returned) == 1


# --- query-embedding memoization -----------------------------------------


def test_embed_query_encodes_once_for_a_repeated_query(monkeypatch):
    """A chat turn embeds the SAME text twice: once for the domain vote
    (app.services.routing.resolve_domain), once for the retrieval that
    follows it. bge-m3 is a 2.2GB transformer, so the second pass was pure
    waste -- measured at ~38ms per encode on this machine's GPU."""
    calls = []

    class _FakeModel:
        def encode(self, texts):
            calls.append(tuple(texts))
            return [[0.5, 0.25]]

    monkeypatch.setattr(ingestion, "_load_embedding_model", lambda name: _FakeModel())
    ingestion._load_embedding_model_cached.cache_clear()
    ingestion._embed_query_cached.cache_clear()

    first = ingestion.embed_query("quelles sont les regles ?")
    second = ingestion.embed_query("quelles sont les regles ?")

    assert first == second
    assert len(calls) == 1, f"expected one encode, got {len(calls)}"

    ingestion.embed_query("une question differente")
    assert len(calls) == 2, "a different query must not hit the cache"

    ingestion._embed_query_cached.cache_clear()
    ingestion._load_embedding_model_cached.cache_clear()


def test_load_embedding_model_loads_once_across_both_call_spellings(monkeypatch):
    """Regression test for a real bug introduced (and caught live) while
    adding the embed cache: with @lru_cache applied directly to
    load_embedding_model, `load_embedding_model()` and
    `load_embedding_model(DEFAULT_EMBEDDING_MODEL)` are two different cache
    keys for the same model, because lru_cache keys on the arguments as
    given and knows nothing about defaults. Both spellings exist in this
    codebase, so the process held TWO copies of a 2.2GB model and paid the
    load twice -- it surfaced as an 8-second 'cold' embed on a model the
    app had already preloaded at startup."""
    loads = []

    def _fake_load(name):
        loads.append(name)
        return object()

    monkeypatch.setattr(ingestion, "_load_embedding_model", _fake_load)
    ingestion._load_embedding_model_cached.cache_clear()

    a = ingestion.load_embedding_model()
    b = ingestion.load_embedding_model(ingestion.DEFAULT_EMBEDDING_MODEL)

    assert a is b
    assert loads == [ingestion.DEFAULT_EMBEDDING_MODEL], f"model loaded {len(loads)} times"

    ingestion._load_embedding_model_cached.cache_clear()


# --- pgvector literal formatting ------------------------------------------


def test_vector_literal_is_compact_and_round_trips_through_float32():
    """`str(embedding)` emitted full float64 repr precision -- digits the
    float32 model never produced -- at ~22.7KB per 1024-dim vector. That
    cost is paid once per query on the search path and once per CHUNK on
    the ingest path."""
    import numpy as np

    vec = np.linspace(-1.0, 1.0, 1024, dtype=np.float32)
    literal = ingestion.to_vector_literal(vec)

    assert literal.startswith("[") and literal.endswith("]")
    parsed = [float(x) for x in literal[1:-1].split(",")]
    assert len(parsed) == 1024
    # 7 significant digits is lossless for float32, so nothing the model
    # actually computed is thrown away.
    assert np.allclose(parsed, vec, rtol=1e-6, atol=1e-7)
    assert len(literal) < len(str(vec.tolist())) * 0.7


# --- context assembly signal-to-noise -------------------------------------


def test_build_context_trims_the_overlap_between_adjacent_chunks():
    """Chunks are built with CHUNK_OVERLAP=250 characters of deliberate
    overlap, so two adjacent chunks landing in the same top-k repeat that
    text verbatim inside a 6000-character context budget."""
    from app.services.retrieval import _build_context

    shared = "L'article 12 impose la consignation avant toute intervention. " * 3
    first = "Chapitre 4 - Consignation.\n" + shared
    second = shared + "\nLe non-respect expose a une sanction."

    context, sources = _build_context(
        [
            {"content": first, "source_name": "guide.md"},
            {"content": second, "source_name": "guide.md"},
        ],
        max_context_length=6000,
    )

    assert context.count("Le non-respect expose a une sanction.") == 1
    assert "Chapitre 4 - Consignation." in context
    # The repeated block survives exactly once, not twice.
    assert context.count(shared.strip()) == 1
    assert sources == ["guide.md"]


def test_build_context_drops_an_exactly_duplicated_chunk():
    """The same text can legitimately come back twice -- an uploaded copy
    of a document that is also in the global corpus. Spending context
    budget to say it twice makes the answer worse, not just longer."""
    from app.services.retrieval import _build_context

    chunk = "Le port du casque est obligatoire dans la zone de production."
    context, sources = _build_context(
        [
            {"content": chunk, "source_name": "upload.pdf"},
            {"content": chunk, "source_name": "global.md"},
        ],
        max_context_length=6000,
    )

    assert context.count(chunk) == 1
    assert sources == ["upload.pdf"]


def test_build_context_keeps_a_short_coincidental_repeat():
    """Only real redundancy is trimmed. A short shared prefix (a repeated
    heading, an article number) is NOT overlap, and cutting it could remove
    the only copy of a citation the answer needed."""
    from app.services.retrieval import _build_context

    context, _ = _build_context(
        [
            {"content": "Article 8. Le casque est obligatoire.", "source_name": "a.md"},
            {"content": "Article 8. Les gants sont obligatoires.", "source_name": "b.md"},
        ],
        max_context_length=6000,
    )

    assert context.count("Article 8.") == 2


# --- resident OCR worker lifecycle ---------------------------------------


class _FakeStream:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _FakeProc:
    def __init__(self):
        self.stdin, self.stdout, self.stderr = _FakeStream(), _FakeStream(), _FakeStream()
        self.killed = False
        self.waited = False

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        self.waited = True
        return 0

    def poll(self):
        return 0 if self.killed else None


def test_kill_closes_every_pipe_and_reaps_the_child():
    """_kill() is NOT a once-per-process teardown: the idle-release timer
    kills the worker after every settings.ocr_worker_idle_release_seconds
    of inactivity and _ensure_alive restarts it on the next page, and the
    GPU-memory and timeout paths kill it too. `proc.kill()` alone left
    three PIPE handles open and an unreaped child per cycle, so a server
    ingesting documents through the day leaked a handle set per cycle until
    the next Popen failed for reasons unrelated to OCR."""
    from app.services.ocr import _ResidentOcrWorker

    worker = _ResidentOcrWorker("python", "worker.py", idle_release_seconds=0)
    proc = _FakeProc()
    worker._proc = proc

    worker._kill()

    assert proc.stdin.closed and proc.stdout.closed and proc.stderr.closed
    assert proc.killed, "the subprocess was not killed"
    assert proc.waited, "the subprocess was never reaped"
    assert worker._proc is None


def test_kill_is_safe_when_no_worker_is_running():
    from app.services.ocr import _ResidentOcrWorker

    worker = _ResidentOcrWorker("python", "worker.py", idle_release_seconds=0)
    worker._kill()  # must not raise
    assert worker._proc is None


def test_shutdown_resident_worker_releases_the_singleton(monkeypatch):
    """App shutdown must release the OCR child process: it holds GPU
    memory, and on an 8GB card a restart that leaves the old worker
    resident starves the new process's own model load."""
    from app.services import ocr

    worker = ocr._ResidentOcrWorker("python", "worker.py", idle_release_seconds=0)
    proc = _FakeProc()
    worker._proc = proc
    monkeypatch.setattr(ocr, "_resident_worker", worker)

    ocr.shutdown_resident_worker()

    assert proc.killed and proc.waited
    assert ocr._resident_worker is None
    ocr.shutdown_resident_worker()  # idempotent


# --- shared SQLAlchemy engine --------------------------------------------


def test_one_shared_engine_backs_every_module():
    """Five modules used to each build their own create_engine() from the
    same settings.database_url. SQLAlchemy's default QueuePool is
    pool_size=5 + max_overflow=10, so one process could hold up to 75
    Postgres backends and still never reuse a connection across modules --
    a chat turn touching history and sources checked out of two unrelated
    pools."""
    from app.models import db

    engine = db.get_engine()
    try:
        assert db.get_engine() is engine, "get_engine must be a singleton"
    finally:
        db.dispose_engine()


def test_get_engine_is_thread_safe():
    """First use is lazy (importing a service must not require a reachable
    Postgres), and under uvicorn the first use can easily be two concurrent
    requests -- which must not produce two engines, i.e. two pools."""
    from app.models import db

    db.dispose_engine()
    engines = []
    barrier = threading.Barrier(8)

    def grab():
        barrier.wait()
        engines.append(db.get_engine())

    threads = [threading.Thread(target=grab) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    try:
        assert len(set(map(id, engines))) == 1, "concurrent first use built more than one engine"
    finally:
        db.dispose_engine()
