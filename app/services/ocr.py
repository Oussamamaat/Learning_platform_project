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

    def image_to_markdown(self, image_bytes: bytes, *, lang_hint: str = "ar+fr") -> str:
        """Return markdown text extracted from one page's raster image.

        Callers (app.services.ingestion's PDF/image parsers) are
        responsible for wrapping the result in a `## Page N` heading where
        applicable -- this returns body text only, not document structure.
        """
        ...


class NullOcrEngine:
    """settings.ocr_engine == "none" -- the default. Every call raises, so
    a PDF/image actually needing OCR fails loudly and specifically instead
    of silently producing an empty or wrong document.
    """

    name = "none"

    def image_to_markdown(self, image_bytes: bytes, *, lang_hint: str = "ar+fr") -> str:
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

    def image_to_markdown(self, image_bytes: bytes, *, lang_hint: str = "ar+fr") -> str:
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
    """

    def __init__(self, venv_python: str, worker_script: str):
        self._venv_python = venv_python
        self._worker_script = worker_script
        self._proc = None
        self._lock = threading.Lock()
        self._out_q: "queue.Queue[Optional[str]]" = queue.Queue()

    def _drain_stdout(self, proc) -> None:
        for line in proc.stdout:
            self._out_q.put(line)
        self._out_q.put(None)  # signals EOF / process exit to any waiting _send

    def _drain_stderr(self, proc) -> None:
        for line in proc.stderr:
            logger.debug("ocr_worker_resident: %s", line.rstrip())

    def _ensure_alive(self) -> None:
        import subprocess

        if self._proc is not None and self._proc.poll() is None:
            return
        self._out_q = queue.Queue()
        self._proc = subprocess.Popen(
            [self._venv_python, self._worker_script],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        threading.Thread(target=self._drain_stdout, args=(self._proc,), daemon=True).start()
        threading.Thread(target=self._drain_stderr, args=(self._proc,), daemon=True).start()

    def _kill(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass
            self._proc = None

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
                raise OcrUnavailableError(f"resident OCR worker (engine={engine!r}) error: {resp.get('error')}")
            return resp.get("markdown", "")


_resident_worker: Optional[_ResidentOcrWorker] = None
_resident_worker_lock = threading.Lock()


def _get_resident_worker(venv_python: str, worker_script: str) -> _ResidentOcrWorker:
    """Process-lifetime singleton, not @lru_cache -- the subprocess it owns
    must be reused across every OCR call regardless of which OcrEngine
    instance triggers the first one (get_ocr_engine() itself is
    lru_cache'd separately and could in principle be re-resolved)."""
    global _resident_worker
    if _resident_worker is None:
        with _resident_worker_lock:
            if _resident_worker is None:
                _resident_worker = _ResidentOcrWorker(venv_python, worker_script)
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

    def image_to_markdown(self, image_bytes: bytes, *, lang_hint: str = "ar+fr") -> str:
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

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(image_bytes)
            image_path = f.name
        try:
            worker = _get_resident_worker(str(venv_python), str(worker_script))
            return worker.ocr(image_path, engine=settings.ocr_paddle_engine, timeout=self._TIMEOUT_SECONDS)
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

    def image_to_markdown(self, image_bytes: bytes, *, lang_hint: str = "ar+fr") -> str:
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

        model = AutoModelForCausalLM.from_pretrained(
            "baidu/Unlimited-OCR", trust_remote_code=True, torch_dtype=torch.bfloat16
        ).cuda()
        processor = AutoProcessor.from_pretrained("baidu/Unlimited-OCR", trust_remote_code=True)
        try:
            image = Image.open(io.BytesIO(image_bytes))
            inputs = processor(images=image, return_tensors="pt").to(model.device)
            output_ids = model.generate(**inputs, max_new_tokens=4096)
            return processor.decode(output_ids[0], skip_special_tokens=True)
        finally:
            # Load-per-job, free-after (see module docstring): an ~8GB
            # BF16 model cannot co-reside with the resident tutor model on
            # an 8GB card, so a reload per document is the accepted cost
            # rather than risk an OOM mid-demo.
            if not get_settings().ocr_keep_resident:
                del model
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
