import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.config import get_settings, get_tenant_id
from app.routers import chat, audio, quiz, ingest, video, voice
from app.errors import AppError
from app.models.db import dispose_engine
from app.services.ingestion import close_db_pools, load_embedding_model
from app.services.ocr import shutdown_resident_worker
from app.services.sources import reap_orphaned_processing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title=get_settings().app_name,
    version=get_settings().app_version,
    description="AI Assistant microservice for IBLOG e-learning platform",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logging.exception("Unhandled error")
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "INTERNAL_ERROR", "message": "An unexpected error occurred"}},
    )


app.include_router(chat.router)
app.include_router(audio.router)
app.include_router(quiz.router)
app.include_router(ingest.router)
app.include_router(video.router)
app.include_router(voice.router)


@app.on_event("startup")
async def _preload_embedding_model() -> None:
    """bge-m3 is ~2.2GB (settings.embedding_model, 2026-08-13) -- loading
    it lazily on the first user request means the first chat/quiz call of
    a fresh process eats that load time. Preloading at boot moves the cost
    off the demo path and surfaces a bad model name / dimension mismatch
    (see load_embedding_model's assertion) as a startup failure instead of
    a mid-demo one.
    """
    load_embedding_model()


@app.on_event("startup")
async def _warm_tts_engine() -> None:
    """A GPU-resident TTS engine (settings.tts_engine="xtts_darija") pays a
    ~51s checkpoint load on its first synthesize(). Calling get_tts_engine()
    here (in a thread, off the event loop) moves that load off a live voice
    turn, where it would look like a hang mid-conversation -- same
    reasoning as _preload_embedding_model above.

    get_tts_engine() itself now does the probing and fallback decision
    (app.services.tts docstring) -- this hook just needs to make sure that
    happens at boot, not on the first WS connection, and that a total
    failure (no engine loadable, no fallback configured) does not crash
    server startup. /health reports the outcome via active_tts_engine_status().
    """
    import asyncio

    from app.services.tts import get_tts_engine, TtsUnavailableError

    try:
        await asyncio.to_thread(get_tts_engine)
    except TtsUnavailableError:
        return  # no TTS configured; voice sessions already fail loudly on their own


@app.on_event("startup")
async def _raise_threadpool_for_vllm() -> None:
    """Second concurrency ceiling, easy to miss (plan Step 4): chat() and
    generate_quiz() (app/routers/chat.py, app/services/quiz.py) are plain
    `def`, so FastAPI/Starlette dispatches them to anyio's worker
    threadpool -- default 40 tokens, shared with every other sync endpoint
    including ingestion. Raising app.services.llm's vLLM semaphore
    (settings.llm_max_concurrent) past 40 buys nothing if requests are
    already queueing here first.

    Only touches the limiter when llm_backend="vllm" -- Ollama's own
    threading.Semaphore(ollama_max_concurrent) is well under 40 already, so
    this is a no-op change for the default (Ollama) deployment. anyio's
    limiter is per-running-event-loop, so this must run from inside a
    startup hook, not at import time.

    This is the MINIMUM fix (raise the ceiling), not the full one --
    switching chat()/generate_quiz() to `async def` with an async HTTP
    client is larger scope and belongs in its own change (see the plan's
    Step 4).
    """
    settings = get_settings()
    if settings.llm_backend != "vllm":
        return
    from anyio import to_thread

    limiter = to_thread.current_default_thread_limiter()
    if limiter.total_tokens < settings.llm_max_concurrent:
        limiter.total_tokens = settings.llm_max_concurrent
        logging.info(
            "Raised anyio threadpool tokens to %d for llm_backend=vllm",
            settings.llm_max_concurrent,
        )


@app.on_event("startup")
async def _reap_orphaned_uploads() -> None:
    """The single-worker in-process ingest queue (app.services.ingest_queue)
    keeps no state outside Postgres -- a server restart mid-job would
    otherwise leave a source_files row stuck at 'processing' forever. Must
    run before any request can poll a stale-but-eternally-"Processing" row.

    Scoped to this process's own tenant (get_tenant_id() is a process-
    lifetime constant, ADR 0001) -- restarting to serve one tenant must
    never mark another tenant's in-flight uploads as errored.
    """
    reap_orphaned_processing(get_tenant_id())


@app.on_event("shutdown")
def _release_resources() -> None:
    """Give back everything this process holds outside its own heap.

    Without this, a uvicorn --reload cycle (or any restart) left three
    things behind: pooled Postgres backends from the shared SQLAlchemy
    engine and the psycopg2 pool, and -- the expensive one -- the resident
    OCR worker subprocess, which holds GPU memory. On an 8GB card the new
    process then tries to load its models alongside the old worker's,
    which is exactly the contention app/config.py's
    ocr_worker_idle_release_seconds docstring records as pushing the tutor
    model onto 31% CPU and past a live request's timeout.

    Sync, not async: every call here is blocking, and a shutdown hook is
    the one place where running blocking work on the loop costs nothing.
    """
    for label, release in (
        ("psycopg2 pools", close_db_pools),
        ("SQLAlchemy engine", dispose_engine),
        ("resident OCR worker", shutdown_resident_worker),
    ):
        try:
            release()
        except Exception:
            logging.exception("could not release %s at shutdown", label)


@app.get("/health", tags=["health"])
async def health_check():
    """Includes voice-engine status, not just process liveness -- the
    2026-09-06 lease's TTS failure would have returned {"status":"ok"} from
    this endpoint the whole time, since it said nothing about whether audio
    could actually be produced. `tts` reflects
    app.services.tts.active_tts_engine_status() (empty until
    get_tts_engine() has run at least once, e.g. at startup); `stt` reports
    only the configured engine name since app.services.stt has no
    equivalent probe/fallback today.
    """
    from app.services.tts import active_tts_engine_status

    settings = get_settings()
    return {
        "status": "ok",
        "version": settings.app_version,
        "tts": active_tts_engine_status(),
        "stt": {"configured": settings.stt_engine},
    }
