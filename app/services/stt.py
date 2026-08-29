"""
STT Engine Seam
───────────────
Swappable speech-to-text backend for the voice pipeline
(app/routers/voice.py). Mirrors app.services.ocr's Protocol/registry/
resident-worker pattern deliberately -- same shape of problem (a heavy
GPU model that must not permanently compete with the resident tutor model
for an 8GB card, best kept in a separate venv and talked to as a
persistent subprocess rather than imported into this process).

Default is "none" (settings.stt_engine, app/config.py): no STT vendor has
been selected yet. docs/architecture/voice-assistant.md's Phase 0 bake-off
(scripts/eval_stt.py) decides between faster-whisper (large-v3-turbo,
strong French, weak/MSA-biased Darija out of the box), a community Darija
Whisper fine-tune, and SeamlessM4T-v2 (explicitly lists Moroccan Arabic
"ary"). That bake-off needs real audio and a GPU this laptop's 8GB card
cannot spare alongside the resident tutor model -- see
docs/architecture/cloud-scaling-plan.md -- so it has not been run, and this
module's "whisper"/"seamless" engines below are UNVERIFIED scaffolding: the
class shape and resident-worker wiring are complete and follow
PaddleOcrEngine verbatim, but nothing here has been exercised against a
loaded model on this machine. Do not flip stt_engine away from "none" in
any deployment before running the bake-off and validating
scripts/speech_worker_resident.py end-to-end.
"""
import json
import logging
from functools import lru_cache
from typing import Optional, Protocol

from app.config import get_settings

logger = logging.getLogger(__name__)


class SttUnavailableError(RuntimeError):
    """No STT engine is configured or loadable in this environment.

    Raised, never swallowed -- a voice session that cannot transcribe must
    fail loudly (surfaced to the client as an error event) rather than
    silently produce an empty transcript that would fall through to
    deterministic_refusal and look like an out-of-corpus question instead
    of a broken pipeline.
    """


class TranscriptChunk:
    """One STT result. `is_final=False` chunks are partials -- fast, may
    still change; `is_final=True` is VAD-endpointed and authoritative,
    the only kind app.routers.voice feeds into the RAG/LLM turn."""

    __slots__ = ("text", "is_final", "language")

    def __init__(self, text: str, *, is_final: bool, language: Optional[str] = None):
        self.text = text
        self.is_final = is_final
        self.language = language


class SttEngine(Protocol):
    name: str

    def transcribe(
        self, audio_bytes: bytes, *, sample_rate: int = 16000, language_hint: Optional[str] = None
    ) -> TranscriptChunk:
        """Transcribe one VAD-endpointed utterance (already the FULL
        utterance, not a streaming chunk -- see app.routers.voice's VAD
        integration for why endpointing happens before this call, not
        inside it). `language_hint` narrows the model's language ID when
        the caller already knows it (e.g. a session pinned to French);
        None lets the engine detect it, which is how the FIRST utterance
        in a session picks the response language.

        Returns a single is_final=True TranscriptChunk. Partial/streaming
        transcription (is_final=False, updated as audio arrives) is a
        latency optimization some engines support natively (faster-whisper
        does not; SeamlessM4T's streaming mode might) -- not required by
        this Protocol, and callers must not assume it's available.
        """
        ...


class NullSttEngine:
    """settings.stt_engine == "none" -- the default. Every call raises, so
    a voice session in an environment with no STT vendor selected fails
    loudly and specifically instead of silently transcribing nothing.
    """

    name = "none"

    def transcribe(
        self, audio_bytes: bytes, *, sample_rate: int = 16000, language_hint: Optional[str] = None
    ) -> TranscriptChunk:
        raise SttUnavailableError(
            "Speech-to-text is not enabled in this environment "
            "(settings.stt_engine='none'). Run the Phase 0 bake-off "
            "(scripts/eval_stt.py) and set stt_engine to its winner -- see "
            "docs/architecture/voice-assistant.md."
        )


class _ResidentSttWorker:
    """Manages ONE persistent .speech_venv subprocess running
    scripts/speech_worker_resident.py, reused for every transcription call
    instead of cold-loading the model per utterance.

    Deliberately the SAME shape as app.services.ocr._ResidentOcrWorker
    (JSON-lines IPC over stdin/stdout, a drain thread per stream, idle
    self-release, kill-and-restart on any failure) -- that design already
    solved this exact problem (a heavy model in a separate venv that must
    not permanently hold VRAM the tutor model needs) for OCR, and an open-
    mic voice session has the same bursty-then-idle usage shape ingestion
    does. Kept as a distinct class rather than a shared base: the two
    workers' request/response payloads differ (image path + engine name vs.
    audio path + language hint) and OCR's class is not written as a base
    class today -- see this module's docstring for why this is unverified
    scaffolding rather than a proven-in-production copy.
    """

    def __init__(self, venv_python: str, worker_script: str, *, idle_release_seconds: float = 120.0):
        import queue
        import threading

        self._venv_python = venv_python
        self._worker_script = worker_script
        self._proc = None
        self._lock = threading.Lock()
        self._out_q: "queue.Queue[Optional[str]]" = queue.Queue()
        self._idle_release_seconds = idle_release_seconds
        self._idle_timer = None

    def _drain_stdout(self, proc, out_q) -> None:
        try:
            for line in proc.stdout:
                out_q.put(line)
        except (ValueError, OSError):
            pass
        finally:
            out_q.put(None)

    def _drain_stderr(self, proc) -> None:
        try:
            for line in proc.stderr:
                logger.debug("speech_worker_resident: %s", line.rstrip())
        except (ValueError, OSError):
            pass

    def _cancel_idle_timer(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _release_idle(self) -> None:
        with self._lock:
            self._idle_timer = None
            if self._proc is not None:
                self._kill()
                logger.info(
                    "resident STT worker idle for %.0fs -- released to free VRAM "
                    "(will cold-restart on the next call)",
                    self._idle_release_seconds,
                )

    def _arm_idle_timer(self) -> None:
        import threading

        self._cancel_idle_timer()
        if self._idle_release_seconds > 0:
            self._idle_timer = threading.Timer(self._idle_release_seconds, self._release_idle)
            self._idle_timer.daemon = True
            self._idle_timer.start()

    def _ensure_alive(self) -> None:
        import queue
        import subprocess
        import threading

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
            proc.wait(timeout=5)
        except Exception:
            pass

    def transcribe(self, audio_path: str, *, engine: str, language_hint: Optional[str], timeout: float) -> dict:
        import queue

        with self._lock:
            self._ensure_alive()
            req = json.dumps({
                "cmd": "transcribe", "id": "1", "audio": audio_path,
                "engine": engine, "language_hint": language_hint,
            })
            try:
                self._proc.stdin.write(req + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                self._kill()
                raise SttUnavailableError(f"resident STT worker's stdin pipe broke: {e}") from e

            try:
                line = self._out_q.get(timeout=timeout)
            except queue.Empty:
                self._kill()
                raise SttUnavailableError(
                    f"resident STT worker (engine={engine!r}) did not respond within "
                    f"{timeout}s -- killed and will restart on the next call."
                )
            if line is None:
                self._kill()
                raise SttUnavailableError(
                    f"resident STT worker process exited unexpectedly (engine={engine!r})."
                )
            resp = json.loads(line)
            if not resp.get("ok"):
                self._arm_idle_timer()
                raise SttUnavailableError(
                    f"resident STT worker (engine={engine!r}) error: {resp.get('error', '')}"
                )
            self._arm_idle_timer()
            return resp


_resident_worker: Optional[_ResidentSttWorker] = None


def _get_resident_worker(venv_python: str, worker_script: str) -> _ResidentSttWorker:
    global _resident_worker
    if _resident_worker is None:
        _resident_worker = _ResidentSttWorker(
            venv_python, worker_script,
            idle_release_seconds=get_settings().speech_worker_idle_release_seconds,
        )
    return _resident_worker


class _ResidentSttEngine:
    """Shared base for the "whisper" and "seamless" engines below -- both
    talk to the same resident-worker subprocess shape, differing only in
    which `engine` string they pass through to
    scripts/speech_worker_resident.py. See this module's docstring: neither
    concrete subclass has been run against a real model on this machine.
    """

    _WORKER_SCRIPT = "scripts/speech_worker_resident.py"
    _TIMEOUT_SECONDS = 60

    def transcribe(
        self, audio_bytes: bytes, *, sample_rate: int = 16000, language_hint: Optional[str] = None
    ) -> TranscriptChunk:
        import tempfile
        import os
        import wave
        from pathlib import Path

        settings = get_settings()
        venv_python = Path(settings.stt_venv_python)
        if not venv_python.exists():
            raise SttUnavailableError(
                f"settings.stt_engine={self.name!r} but the dedicated speech venv's "
                f"interpreter was not found at {venv_python} (settings.stt_venv_python)."
            )
        worker_script = Path(__file__).resolve().parents[2] / self._WORKER_SCRIPT
        if not worker_script.exists():
            raise SttUnavailableError(f"STT worker script not found: {worker_script}")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        try:
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_bytes)
            worker = _get_resident_worker(str(venv_python), str(worker_script))
            resp = worker.transcribe(
                wav_path, engine=self.name, language_hint=language_hint,
                timeout=self._TIMEOUT_SECONDS,
            )
        finally:
            os.unlink(wav_path)
        return TranscriptChunk(
            resp.get("text", ""), is_final=True, language=resp.get("language"),
        )


class WhisperEngine(_ResidentSttEngine):
    """settings.stt_engine='whisper' -- faster-whisper (CTranslate2),
    settings.stt_model selects the checkpoint. Best-in-class French;
    Darija quality depends entirely on which checkpoint stt_model names --
    stock Whisper's `ar` is MSA-biased (see this module's docstring)."""

    name = "whisper"


class SeamlessEngine(_ResidentSttEngine):
    """settings.stt_engine='seamless' -- SeamlessM4T-v2-large, the
    candidate that explicitly lists Moroccan Arabic ("ary") as a source
    language rather than generic MSA "ar"."""

    name = "seamless"


_ENGINES = {
    "none": NullSttEngine,
    "whisper": WhisperEngine,
    "seamless": SeamlessEngine,
}


@lru_cache(maxsize=1)
def get_stt_engine() -> SttEngine:
    """Cached singleton, keyed off settings.stt_engine at first call --
    same read-once-per-process contract as app.services.ocr.get_ocr_engine.
    """
    engine_name = get_settings().stt_engine
    engine_cls = _ENGINES.get(engine_name)
    if engine_cls is None:
        raise SttUnavailableError(
            f"Unknown settings.stt_engine={engine_name!r}. Valid values: {sorted(_ENGINES)}."
        )
    return engine_cls()
