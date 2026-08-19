"""
OCR Engine Seam
───────────────
Swappable backend for turning a scanned/image page into markdown text, used
by app.services.ingestion's PDF/image parsers when a page has no usable
embedded text layer.

Default is "none" (settings.ocr_engine): no OCR dependency is installed by
default (see config/requirements-ocr.txt, deliberately excluded from
config/requirements.txt), so a PDF/image needing OCR fails loudly with
OcrUnavailableError instead of either silently ingesting nothing or
attempting an unverified read. This keeps the laptop/testing environment
installable and correct without an OCR stack.

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
ground-truth image before any engine is enabled outside a test run. Until
it does, ocr_engine stays "none" in every real deployment.
"""
import logging
from functools import lru_cache
from typing import Protocol

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
    it shells out to scripts/ocr_paddleocr_worker.py running under
    settings.ocr_venv_python (that .ocr_venv interpreter), passing the page
    image and reading back the markdown that script writes. Each call is a
    fresh subprocess: settings.ocr_keep_resident, which lets the OTHER GPU
    engine (UnlimitedOcrEngine) keep its model loaded between calls, has no
    equivalent here -- there is no long-lived pipeline object in THIS
    process to keep resident, only a worker process that starts and exits
    per page. See the module-level note on GPU co-residency with the tutor
    model in app/services/ingestion.py's PDF docstring history for why that
    per-call reload cost is the accepted tradeoff rather than a resident
    cross-process worker.
    """

    name = "paddleocr"

    _WORKER_SCRIPT = "scripts/ocr_paddleocr_worker.py"
    _TIMEOUT_SECONDS = 180

    def image_to_markdown(self, image_bytes: bytes, *, lang_hint: str = "ar+fr") -> str:
        import subprocess
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
        out_fd, out_path = tempfile.mkstemp(suffix=".md")
        os.close(out_fd)
        try:
            result = subprocess.run(
                [
                    str(venv_python), str(worker_script),
                    "--image", image_path, "--out", out_path, "--lang-hint", lang_hint,
                ],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=self._TIMEOUT_SECONDS,
            )
            if result.returncode != 0:
                raise OcrUnavailableError(
                    f"PaddleOCR-VL subprocess (via {venv_python}) failed with exit "
                    f"code {result.returncode}: {result.stderr.strip()[-2000:]}"
                )
            return Path(out_path).read_text(encoding="utf-8")
        except subprocess.TimeoutExpired as e:
            raise OcrUnavailableError(
                f"PaddleOCR-VL subprocess did not finish within "
                f"{self._TIMEOUT_SECONDS}s (via {venv_python})."
            ) from e
        finally:
            os.unlink(image_path)
            if os.path.exists(out_path):
                os.unlink(out_path)


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
    """Cached singleton, keyed off settings.ocr_engine at first call. GPU
    engines free their model per job internally (see PaddleOcrEngine/
    UnlimitedOcrEngine), so caching the wrapper instance is cheap; it does
    not pin GPU memory between calls unless ocr_keep_resident is set.
    """
    engine_name = get_settings().ocr_engine
    engine_cls = _ENGINES.get(engine_name)
    if engine_cls is None:
        raise OcrUnavailableError(
            f"Unknown settings.ocr_engine={engine_name!r}. Valid values: "
            f"{sorted(_ENGINES)}."
        )
    return engine_cls()
