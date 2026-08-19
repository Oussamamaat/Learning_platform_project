"""
Single-Worker Ingest Queue
───────────────────────────
Async processing for tenant document uploads (app/routers/ingest.py),
deliberately NOT Redis and NOT FastAPI's BackgroundTasks.

Not Redis: `redis` appears in zero Python files in this repo and is not in
config/requirements.txt -- the Redis container running in this project's
docker-compose is infrastructure that exists but that no application code
actually uses yet. Adopting it here means a new dependency, a second
process, and compose/Dockerfile changes -- real operational weight for a
single-tenant demo whose only concurrency requirement is "exactly one
ingest job at a time."

Not BackgroundTasks: for a sync endpoint, FastAPI runs background tasks in
the SAME shared anyio threadpool the rest of the app (including chat's
request handling) uses. A 20-file drag-drop upload would fire 20
concurrent embed/OCR jobs, starve chat requests of that same threadpool,
and guarantee a GPU OOM if two OCR jobs (each loading an ~8GB model) ever
overlapped.

max_workers=1 is the design, not a placeholder or a default left
un-tuned: GPU OCR cannot run concurrently with itself on one card
(app.services.ocr's engines load their model per job), and embedding is
already internally batched (app.services.ingestion.BATCH_SIZE) -- a
second concurrent job would only contend for the same GPU/CPU resource,
never add real throughput.

Status lives in Postgres (source_files.status), never in this module's
memory -- so a status poll (GET /api/v1/ingest/sources) always reads
truth regardless of which process/thread is doing the work, and a server
restart is handled by app.services.sources.reap_orphaned_processing at
startup, not by anything here.

Swapping to a real task queue later (RQ, once Redis is actually adopted)
is a one-function change: only `submit`'s body needs to change; every
caller (app/routers/ingest.py) stays the same.
"""
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

logger = logging.getLogger(__name__)

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")


def submit(fn: Callable, *args, **kwargs) -> None:
    """Fire-and-forget: the caller (app/routers/ingest.py's upload
    endpoint) does not wait on or inspect the returned Future -- job
    outcome is read back from source_files.status via polling, not from
    this call. Exceptions inside `fn` are still logged here (a bare
    ThreadPoolExecutor swallows them into the discarded Future otherwise,
    which would turn a real ingest failure into total silence) even though
    `fn` itself (app.services.ingest_jobs.process_source_file) is expected
    to catch its own exceptions and write status='error' -- this is a
    backstop for a bug in that error handling, not the primary path.
    """
    future = _EXECUTOR.submit(fn, *args, **kwargs)

    def _log_if_failed(f):
        exc = f.exception()
        if exc is not None:
            logger.exception("ingest job %r raised unexpectedly", fn, exc_info=exc)

    future.add_done_callback(_log_if_failed)
