"""
OCR Engine Seam
───────────────
Swappable backend for turning a scanned/image page into markdown text, used
by app.services.ingestion's PDF/image parsers when a page has no usable
embedded text layer.

Default is "paddleocr" (settings.ocr_engine, app/config.py) as of
2026-08-18, once scripts/verify_ocr_arabic.py started passing all four
gates against a live PaddleOCR-VL run -- see that setting's comment for
what changed. "none" is still what an environment WITHOUT .ocr_venv set up
effectively gets: PaddleOcrEngine raises OcrUnavailableError with an
actionable message rather than either silently ingesting nothing or
attempting an unverified read, so a laptop/testing environment missing the
OCR stack still fails loudly and specifically, not silently.

Which engine, and why not the one originally named in ingestion.py's old
docstring (baidu/Unlimited-OCR):

  - Unlimited-OCR is real (MIT, 3B params, "one-shot long-horizon parsing"
    via R-SWA attention -- genuinely well-suited to long documents), but
    its model card names no explicit language list and reporting describes
    it as optimized heavily for English/Chinese with community-confirmed
    Cyrillic support, not Arabic. That corroborates rather than resolves
    the original docstring's caution about tenant #1's Arabic-script
    Darija corpus. It also pins torch==2.10.0/CUDA 12.9, incompatible with
    .gguf_venv's torch 2.11.0+cu128 -- it needs its own environment
    regardless of which engine is the default.
  - PaddleOCR-VL (also Baidu-lineage) has explicit, dedicated Arabic
    recognition models, 100+ language coverage, and the strongest
    published OmniDocBench v1.6 score in its class. It is the default
    production engine here. Unlimited-OCR is kept wired as a second
    pluggable backend worth benchmarking on real tenant documents later.

Neither engine's Arabic fidelity is assumed correct -- see
scripts/verify_ocr_arabic.py, which must pass against a committed
ground-truth image before any engine is enabled outside a test run.
"""
import json
import logging
import queue
import threading
from functools import lru_cache
from typing import Optional, Protocol

from app.config import get_settings

logger = logging.getLogger(__name__)


class OcrUnavailableError(RuntimeError):
    """No OCR engine is configured or loadable in this environment.

    Raised, never swallowed -- app.services.ingest_jobs.process_source_file
    catches this specifically to mark a source_files row status='error'
    with this message (actionable: tells the user to convert the file or
    enable OCR) instead of silently ingesting an empty document that would
    look like a successful upload with nothing retrievable in it.
    """


class OcrEngine(Protocol):
    name: str

    def image_to_markdown(
        self, image_bytes: bytes, *, lang_hint: str = "ar+fr", tier: Optional[str] = None
    ) -> str:
        """Return markdown text extracted from one page's raster image.

        Callers (app.services.ingestion's PDF/image parsers) are
        responsible for wrapping the result in a `## Page N` heading where
        applicable -- this returns body text only, not document structure.

        `tier` selects between the heavy and light pipelines for engines
        that have both: "heavy" (settings.ocr_paddle_engine) or "light"
        (settings.ocr_light_engine). None means "whatever this engine
        does by default", which for every engine except PaddleOcrEngine is
        its only behaviour. This is a per-CALL choice, not a per-process
        one, because the right engine depends on the individual page --
        see app.services.ingestion._ocr_pdf_page's two-tier routing.
        """
        ...


class NullOcrEngine:
    """settings.ocr_engine == "none" -- the default. Every call raises, so
    a PDF/image actually needing OCR fails loudly and specifically instead
    of silently producing an empty or wrong document.
    """

    name = "none"

    def image_to_markdown(
        self, image_bytes: bytes, *, lang_hint: str = "ar+fr", tier: Optional[str] = None
    ) -> str:
        raise OcrUnavailableError(
            "This page has no usable embedded text layer and OCR is not "
            "enabled in this environment (settings.ocr_engine='none'). "
            "Convert the document to .md/.txt, or enable OCR -- see "
            "config/requirements-ocr.txt and scripts/verify_ocr_arabic.py."
        )


class TesseractEngine:
    """CPU baseline (settings.ocr_engine='tesseract'), pytesseract + a
    system Tesseract binary with 'ara'+'fra' traineddata installed. Not the
    production engine -- it's the honest floor scripts/verify_ocr_arabic.py
    measures the GPU engines against, and a no-GPU smoke-test path.
    """

    name = "tesseract"

    def image_to_markdown(
        self, image_bytes: bytes, *, lang_hint: str = "ar+fr", tier: Optional[str] = None
    ) -> str:
        try:
            import pytesseract
            from PIL import Image
            import io
        except ImportError as e:
            raise OcrUnavailableError(
                "settings.ocr_engine='tesseract' but pytesseract/Pillow are not "
                "installed -- see config/requirements-ocr.txt."
            ) from e

        lang = lang_hint.replace("ar", "ara").replace("fr", "fra").replace("+", "+")
        try:
            image = Image.open(io.BytesIO(image_bytes))
            return pytesseract.image_to_string(image, lang=lang)
        except pytesseract.TesseractNotFoundError as e:
            raise OcrUnavailableError(
                "pytesseract is installed but the system 'tesseract' binary was "
                "not found on PATH -- install Tesseract OCR plus the 'ara' and "
                "'fra' traineddata files."
            ) from e


# Substrings that mark an OCR worker error as a GPU-memory failure rather
# than an ordinary per-page application error. Measured live: one page's
# CUDA OOM (RuntimeError: "CUDA error(2)... cudaErrorMemoryAllocation")
# was immediately followed by 5 more OOM failures on completely
# different, unrelated pages within the SAME still-alive worker process --
# _handle_ocr (scripts/ocr_worker_resident.py) catches the exception and
# returns {"ok": false}, so the process never exits and _ensure_alive
# never has a reason to give it a fresh CUDA context. A GPU allocator that
# has hit OOM once commonly cannot recover cleanly within that process
# even once VRAM pressure elsewhere passes, so _ResidentOcrWorker.ocr
# treats this class of error as fatal to the WORKER (kills it, same as a
# crash) rather than just fatal to the one page.
_GPU_MEMORY_ERROR_MARKERS = (
    "out of memory", "outofmemoryerror", "cuda error", "cudaerrormemoryallocation",
)


def _looks_like_gpu_memory_error(message: str) -> bool:
    lower = (message or "").lower()
    return any(marker in lower for marker in _GPU_MEMORY_ERROR_MARKERS)


class _ResidentOcrWorker:
    """Manages ONE persistent .ocr_venv subprocess running
    scripts/ocr_worker_resident.py, reused for every OCR call in this
    process's lifetime instead of spawning (and cold-loading a 3B model
    into) a fresh subprocess per page.

    This is the fix for the dominant cost measured under the old design
    (scripts/ocr_paddleocr_worker.py, one process per page): 193-200s
    wall-clock per page, almost entirely PaddleOCR-VL's own model-load
    time, repeated on every single page of every document. A 25-page OCR
    workload that used to take ~85 minutes pays that load cost ONCE.

    IPC is JSON-lines over stdin/stdout (see ocr_worker_resident.py's
    docstring for the protocol) rather than pickling or a socket -- both
    processes are local, short-lived-per-run, and the payload (a file path
    in, a markdown string out) is small enough that a subprocess pipe adds
    no measurable overhead next to the seconds-scale OCR call itself.

    A background thread drains stdout into a queue so `_send` can honor a
    per-call timeout via `queue.get(timeout=...)` -- Windows named pipes
    don't support `select()`, so a blocking `readline()` with no thread
    would have no way to time out a hung worker. A second background
    thread drains stderr into the logger for the same reason (an
    un-drained pipe can deadlock the child if it fills the OS pipe buffer)
    and so the worker's own load/diagnostic lines are never lost.

    Self-healing, not fault-tolerant: if the subprocess dies (crash, OOM,
    timeout-triggered kill), `_ensure_alive` restarts it on the NEXT call.
    A single ingestion run's worth of state is a page number and an image
    path, both owned by the caller, so losing an in-flight subprocess
    costs at most the one page that was mid-flight, not the whole run.

    IDLE RELEASE: unlike the model-load cost above, there was previously
    NO mechanism to free this worker's VRAM once an ingestion run finished
    -- `_kill()` existed but was only ever reached on a timeout or crash,
    so the subprocess (and PaddleOCR-VL's ~2GB) stayed resident for this
    app process's entire remaining lifetime. Measured consequence: with
    this worker resident, `IBLOG_TUTOR` (the Darija tutor model, 7.5GB)
    could not fit in the remaining VRAM on an 8GB card and Ollama loaded it
    31% CPU / 69% GPU -- slow enough that a live chat request exceeded its
    180s client timeout. `_arm_idle_timer`/`_release_idle` below fix that:
    after `idle_release_seconds` with no OCR call, the subprocess is killed
    and its VRAM freed; `_ensure_alive` transparently restarts it (paying
    the cold-load cost again) on the next call, which is the correct
    tradeoff for a single-worker ingest queue that processes one document
    at a time and is otherwise idle for long stretches.
    """

    def __init__(self, venv_python: str, worker_script: str, *, idle_release_seconds: float = 120.0):
        self._venv_python = venv_python
        self._worker_script = worker_script
        self._proc = None
        self._lock = threading.Lock()
        self._out_q: "queue.Queue[Optional[str]]" = queue.Queue()
        self._idle_release_seconds = idle_release_seconds
        self._idle_timer: Optional[threading.Timer] = None

    def _drain_stdout(self, proc, out_q: "queue.Queue[Optional[str]]") -> None:
        # ValueError/OSError here means _kill() closed the pipe out from
        # under this thread, which is a normal shutdown, not an error --
        # the finally still posts the EOF sentinel so nothing waits forever.
        # `out_q` is the queue captured at spawn time, NOT `self._out_q` --
        # deliberately, to close a race hit live by the GPU-memory-error
        # kill path: `_kill()` can end a process that is still genuinely
        # alive (mid-request), so THIS thread is still blocked reading its
        # stdout at the moment `_ensure_alive()` spawns a replacement and
        # reassigns `self._out_q` to a fresh queue. If this loop read
        # `self._out_q` instead of a captured reference, this now-dead
        # process's late EOF would arrive as a None on the REPLACEMENT's
        # queue, killing the next page's genuinely healthy response.
        try:
            for line in proc.stdout:
                out_q.put(line)
        except (ValueError, OSError):
            pass
        finally:
            out_q.put(None)  # signals EOF / process exit to any waiting _send

    def _drain_stderr(self, proc) -> None:
        try:
            for line in proc.stderr:
                logger.debug("ocr_worker_resident: %s", line.rstrip())
        except (ValueError, OSError):
            pass

    def _cancel_idle_timer(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _release_idle(self) -> None:
        """Timer callback (runs on its own thread, not the caller's) --
        acquires the same lock `ocr()` does, so it can never kill a
        subprocess an in-flight call is actively using. Narrow benign
        race: if this fires the instant a new `ocr()` call is about to
        start, that call may pay one extra cold reload rather than reusing
        a warm worker -- self-healing (`_ensure_alive`) absorbs it exactly
        like a crash would, so it is not worth a generation counter to
        close completely."""
        with self._lock:
            self._idle_timer = None
            if self._proc is not None:
                self._kill()
                logger.info(
                    "resident OCR worker idle for %.0fs -- released to free VRAM "
                    "(will cold-restart on the next OCR call)",
                    self._idle_release_seconds,
                )

    def _arm_idle_timer(self) -> None:
        """Call with self._lock already held. Cheap no-op when disabled
        (idle_release_seconds <= 0)."""
        self._cancel_idle_timer()
        if self._idle_release_seconds > 0:
            self._idle_timer = threading.Timer(self._idle_release_seconds, self._release_idle)
            self._idle_timer.daemon = True
            self._idle_timer.start()

    def _ensure_alive(self) -> None:
        import subprocess

        # Cancel any pending idle-release the moment this worker is about
        # to be used -- this call is the first thing `ocr()` does inside
        # the lock, so there is no window for the idle timer to fire
        # against a process a caller is actively about to use.
        self._cancel_idle_timer()

        if self._proc is not None and self._proc.poll() is None:
            return
        self._out_q = queue.Queue()
        self._proc = subprocess.Popen(
            [self._venv_python, self._worker_script],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        threading.Thread(target=self._drain_stdout, args=(self._proc, self._out_q), daemon=True).start()
        threading.Thread(target=self._drain_stderr, args=(self._proc,), daemon=True).start()

    def _kill(self) -> None:
        """Kill the worker AND reclaim everything the OS gave us for it.

        `self._proc.kill()` alone -- which is all this used to do -- leaves
        three open pipe handles (stdin/stdout/stderr, all subprocess.PIPE)
        and an unreaped child. That is a real leak here rather than a
        theoretical one, because this is not a once-per-process teardown:
        `_release_idle` kills the worker after every
        settings.ocr_worker_idle_release_seconds of inactivity and
        `_ensure_alive` restarts it on the next page, and the GPU-memory
        and timeout paths kill it too. A server that ingests documents
        through the day therefore cycles the worker many times, leaking a
        handle set per cycle, until the process hits its descriptor/handle
        ceiling -- at which point the NEXT Popen fails and OCR looks
        broken for reasons that have nothing to do with OCR.

        stdin is closed first, on purpose: the worker's read loop sees EOF
        and can exit on its own, which is a clean shutdown rather than a
        kill. The kill still follows, because a worker blocked in a CUDA
        call will not notice EOF promptly and this must not block the
        caller (it runs under self._lock).
        """
        proc, self._proc = self._proc, None
        if proc is None:
            return
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass
        try:
            proc.kill()
        except Exception:
            pass
        try:
            # Reap it. Without a wait() the child stays a zombie on POSIX
            # and its handle stays open on Windows; the drain threads have
            # already been detached from it by the stream closes above, so
            # this returns promptly.
            proc.wait(timeout=5)
        except Exception:
            pass

    def ocr(self, image_path: str, *, engine: str, timeout: float) -> str:
        with self._lock:
            self._ensure_alive()
            req = json.dumps({"cmd": "ocr", "id": "1", "image": image_path, "engine": engine})
            try:
                self._proc.stdin.write(req + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                self._kill()
                raise OcrUnavailableError(f"resident OCR worker's stdin pipe broke: {e}") from e

            try:
                line = self._out_q.get(timeout=timeout)
            except queue.Empty:
                self._kill()
                raise OcrUnavailableError(
                    f"resident OCR worker (engine={engine!r}) did not respond within {timeout}s "
                    f"-- killed and will restart on the next call."
                )
            if line is None:
                self._kill()
                raise OcrUnavailableError(
                    f"resident OCR worker process exited unexpectedly (engine={engine!r}); "
                    f"see logger.debug ocr_worker_resident lines for its stderr."
                )
            resp = json.loads(line)
            if not resp.get("ok"):
                error_message = resp.get("error", "")
                if _looks_like_gpu_memory_error(error_message):
                    # _handle_ocr (ocr_worker_resident.py) catches CUDA OOM
                    # like any other exception and keeps the process alive --
                    # but a GPU allocator that has hit OOM once tends to stay
                    # poisoned for that process's lifetime. Measured live: one
                    # OOM on an unrelated page was immediately followed by 5
                    # more OOM failures on completely different pages, all in
                    # this same still-alive worker. Kill it here so the next
                    # call gets a fresh CUDA context instead of reusing the
                    # poisoned one -- do not arm the idle timer on a process
                    # that no longer exists.
                    self._kill()
                    raise OcrUnavailableError(
                        f"resident OCR worker (engine={engine!r}) hit a GPU-memory error and was "
                        f"killed to avoid poisoning subsequent pages: {error_message}"
                    )
                # The process itself is alive and about to go idle regardless
                # of whether THIS call's result was an application-level
                # error -- arm the release timer either way. Only the
                # exception paths above (broken pipe, timeout, process exit,
                # GPU-memory error) already killed the process and correctly
                # skip this.
                self._arm_idle_timer()
                raise OcrUnavailableError(f"resident OCR worker (engine={engine!r}) error: {error_message}")
            self._arm_idle_timer()
            return resp.get("markdown", "")


_resident_worker: Optional[_ResidentOcrWorker] = None
_resident_worker_lock = threading.Lock()


def shutdown_resident_worker() -> None:
    """Kill the resident OCR worker if one is running. Called from app
    shutdown (app/main.py): the worker is a CHILD process holding GPU
    memory, and without this an app restart could leave the old worker
    holding VRAM while the new one tries to load its own model onto the
    same 8GB card."""
    global _resident_worker
    with _resident_worker_lock:
        worker = _resident_worker
        _resident_worker = None
    if worker is not None:
        with worker._lock:
            worker._cancel_idle_timer()
            worker._kill()
        logger.info("resident OCR worker shut down")


def _get_resident_worker(venv_python: str, worker_script: str) -> _ResidentOcrWorker:
    """Process-lifetime singleton, not @lru_cache -- the subprocess it owns
    must be reused across every OCR call regardless of which OcrEngine
    instance triggers the first one (get_ocr_engine() itself is
    lru_cache'd separately and could in principle be re-resolved)."""
    global _resident_worker
    if _resident_worker is None:
        with _resident_worker_lock:
            if _resident_worker is None:
                _resident_worker = _ResidentOcrWorker(
                    venv_python, worker_script,
                    idle_release_seconds=get_settings().ocr_worker_idle_release_seconds,
                )
    return _resident_worker


class PaddleOcrEngine:
    """settings.ocr_engine='paddleocr' -- the recommended production
    engine (see module docstring), gated by scripts/verify_ocr_arabic.py.

    paddlepaddle-gpu + paddleocr are installed in a SEPARATE environment
    from .gguf_venv (this repo's .ocr_venv) to avoid conflicting CUDA/torch
    pins:

        python -m pip install paddlepaddle-gpu==3.2.1 \\
            -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
        python -m pip install -U "paddleocr[doc-parser]"

    Because of that separation, this class never imports paddleocr itself --
    it talks to scripts/ocr_worker_resident.py, a persistent .ocr_venv
    subprocess (via _ResidentOcrWorker/_get_resident_worker above) started
    once and reused for every page, rather than the fresh
    scripts/ocr_paddleocr_worker.py subprocess this class used to spawn
    per call. settings.ocr_keep_resident is therefore now honored by this
    class too (previously only UnlimitedOcrEngine did): the resident
    worker's model stays loaded in ITS process for as long as this app
    process runs, regardless of that setting, which only controls whether
    UnlimitedOcrEngine additionally keeps ITS in-process model resident.

    settings.ocr_paddle_engine selects which of the worker's three
    pipelines handles each call ("vl" | "structure" | "classic") -- see
    scripts/ocr_worker_resident.py's docstring and scripts/ocr_bakeoff.py
    for what each trades off and how the default was chosen.
    """

    name = "paddleocr"

    _WORKER_SCRIPT = "scripts/ocr_worker_resident.py"
    _TIMEOUT_SECONDS = 300

    def image_to_markdown(
        self, image_bytes: bytes, *, lang_hint: str = "ar+fr", tier: Optional[str] = None
    ) -> str:
        import tempfile
        import os
        from pathlib import Path

        settings = get_settings()
        venv_python = Path(settings.ocr_venv_python)
        if not venv_python.exists():
            raise OcrUnavailableError(
                f"settings.ocr_engine='paddleocr' but the dedicated OCR venv's "
                f"interpreter was not found at {venv_python} (settings."
                f"ocr_venv_python) -- see this class's docstring for how to set "
                f"up .ocr_venv, or set ocr_venv_python to point at an existing one."
            )
        worker_script = Path(__file__).resolve().parents[2] / self._WORKER_SCRIPT
        if not worker_script.exists():
            raise OcrUnavailableError(f"OCR worker script not found: {worker_script}")

        # tier -> which of the worker's pipelines handles THIS page. The
        # resident worker keeps each pipeline it has been asked for loaded
        # for the rest of its life, so mixing tiers within one document
        # costs one extra model load total, not one per page.
        if tier == "light":
            engine = settings.ocr_light_engine
        else:
            engine = settings.ocr_paddle_engine

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(image_bytes)
            image_path = f.name
        try:
            worker = _get_resident_worker(str(venv_python), str(worker_script))
            return worker.ocr(image_path, engine=engine, timeout=self._TIMEOUT_SECONDS)
        finally:
            os.unlink(image_path)


class UnlimitedOcrEngine:
    """settings.ocr_engine='unlimited_ocr' -- baidu/Unlimited-OCR, kept as
    a pluggable alternative worth benchmarking (see module docstring for
    why it isn't the default). Requires transformers + torch==2.10.0/
    CUDA 12.9 in a dedicated environment, distinct from both .gguf_venv
    and the PaddleOCR-VL venv.
    """

    name = "unlimited_ocr"

    def __init__(self) -> None:
        self._model = None
        self._processor = None

    def image_to_markdown(
        self, image_bytes: bytes, *, lang_hint: str = "ar+fr", tier: Optional[str] = None
    ) -> str:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
            from PIL import Image
            import io
        except ImportError as e:
            raise OcrUnavailableError(
                "settings.ocr_engine='unlimited_ocr' but its dependencies "
                "(torch==2.10.0, transformers, Pillow) are not installed in "
                "this environment -- see huggingface.co/baidu/Unlimited-OCR."
            ) from e

        keep_resident = get_settings().ocr_keep_resident

        # settings.ocr_keep_resident used to do NOTHING here beyond
        # skipping empty_cache(): the model was a local, so it was dropped
        # and re-loaded from scratch on the next call whether the flag was
        # set or not. self._model/self._processor existed for exactly this
        # and were never assigned. An ~8GB BF16 load per page is not a
        # subtle regression -- the flag's only reason to exist is to avoid
        # it on a box with the VRAM to spare.
        if keep_resident and self._model is not None:
            model, processor = self._model, self._processor
        else:
            model = AutoModelForCausalLM.from_pretrained(
                "baidu/Unlimited-OCR", trust_remote_code=True, torch_dtype=torch.bfloat16
            ).cuda()
            processor = AutoProcessor.from_pretrained("baidu/Unlimited-OCR", trust_remote_code=True)
            if keep_resident:
                self._model, self._processor = model, processor

        image = None
        try:
            image = Image.open(io.BytesIO(image_bytes))
            inputs = processor(images=image, return_tensors="pt").to(model.device)
            output_ids = model.generate(**inputs, max_new_tokens=4096)
            return processor.decode(output_ids[0], skip_special_tokens=True)
        finally:
            # PIL keeps the decoded buffer alive until the image object is
            # closed; these are full-page renders at OCR_RENDER_DPI, so on
            # a long document the un-closed ones add up.
            if image is not None:
                try:
                    image.close()
                except Exception:
                    pass
            # Load-per-job, free-after (see module docstring): an ~8GB
            # BF16 model cannot co-reside with the resident tutor model on
            # an 8GB card, so a reload per document is the accepted cost
            # rather than risk an OOM mid-demo.
            if not keep_resident:
                del model
                del processor
                torch.cuda.empty_cache()


_ENGINES = {
    "none": NullOcrEngine,
    "tesseract": TesseractEngine,
    "paddleocr": PaddleOcrEngine,
    "unlimited_ocr": UnlimitedOcrEngine,
}


@lru_cache(maxsize=1)
def get_ocr_engine() -> OcrEngine:
    """Cached singleton, keyed off settings.ocr_engine at first call
    (settings.ocr_engine is therefore effectively read-once per process --
    changing it after the first call has no effect until the process
    restarts). UnlimitedOcrEngine frees its in-process model per job by
    default (see its docstring, ocr_keep_resident); PaddleOcrEngine keeps
    NO model in this process at all -- its resident worker subprocess
    (scripts/ocr_worker_resident.py, spawned once via
    _get_resident_worker) is what stays loaded, unconditionally, for the
    life of this app process. Either way, caching this wrapper instance
    itself is cheap.
    """
    engine_name = get_settings().ocr_engine
    engine_cls = _ENGINES.get(engine_name)
    if engine_cls is None:
        raise OcrUnavailableError(
            f"Unknown settings.ocr_engine={engine_name!r}. Valid values: "
            f"{sorted(_ENGINES)}."
        )
    return engine_cls()
