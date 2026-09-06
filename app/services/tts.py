"""
TTS Engine Seam
───────────────
Swappable text-to-speech backend for the voice pipeline
(app/routers/voice.py). Sibling to app.services.stt, but CPU-only and
in-process rather than a resident subprocess -- Piper (see PiperEngine
below) is small enough (ONNX, real-time-factor ~0.05) that it does not
need OCR/STT's dedicated-venv-plus-subprocess treatment; it shares
.gguf_venv directly.

Default is "none" (settings.tts_engine, app/config.py): no TTS vendor has
been selected yet -- this has been resurrection.md's single largest open
MVP item since the project began (docs/architecture/rectified/
analyze_01.md: "Darija TTS is the highest-risk unknown in the whole
program"). docs/architecture/voice-assistant.md's Phase 0 bake-off
(scripts/eval_tts.py) has not been run. PiperEngine below is UNVERIFIED --
the class shape is complete but nothing here has been exercised against a
loaded voice model on this machine.

Why Piper over the higher-quality alternatives (recorded here so the
choice isn't silently re-litigated): XTTS-v2 and MMS-TTS both sound
better, but both license under non-commercial terms (Coqui CPML / CC-BY-NC)
-- unusable in a B2B product. Piper is MIT. The accepted quality trade-off,
confirmed with the user: MSA/Jordanian-accented Arabic TTS reading Darija
script is intelligible but not native Darija prosody; a Piper voice
fine-tuned on atlasia/DODa-audio-dataset is the tracked follow-up, not
built.
"""
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional, Protocol

from app.config import get_settings

logger = logging.getLogger(__name__)


class TtsUnavailableError(RuntimeError):
    """No TTS engine is configured or loadable in this environment.

    Raised, never swallowed -- a voice session that cannot synthesize
    audio must fail loudly (surfaced to the client as an error event, with
    the answer text still delivered over the data channel for the UI to
    show as captions) rather than silently produce empty audio.
    """


class TtsEngine(Protocol):
    name: str
    # PCM sample rate of whatever synthesize() returns -- the client needs
    # this to play the audio back at the right pitch/speed (a WAV/PCM
    # buffer carries no rate of its own). Fixed per engine instance, not
    # reported per-call: Piper voices commonly standardize on 22050Hz, and
    # treating both settings.tts_voice_fr/tts_voice_ar as sharing one rate
    # is a documented MVP simplification (see PiperEngine) rather than a
    # per-synthesize-call return value -- revisit if the bake-off picks
    # voices with genuinely different native rates.
    sample_rate: int

    def synthesize(self, text: str, *, language: str) -> bytes:
        """Synthesize one chunk of text (a sentence, per
        app.routers.voice's sentence-level streaming split -- see
        app.services.llm.stream_llm_response's docstring for why citation
        text is excluded from what reaches here) to 16-bit mono PCM audio
        bytes at this engine's native sample rate.

        `language`: "fr" or "darija" -- selects the voice
        (settings.tts_voice_fr / tts_voice_ar), not the text's script;
        callers are responsible for only sending text that is actually in
        that language.
        """
        ...


class NullTtsEngine:
    """settings.tts_engine == "none" -- the default. Every call raises, so
    a voice session in an environment with no TTS vendor selected fails
    loudly and specifically instead of silently producing silence.
    """

    name = "none"
    sample_rate = 22050  # arbitrary -- synthesize() always raises, never actually produces audio

    def synthesize(self, text: str, *, language: str) -> bytes:
        raise TtsUnavailableError(
            "Text-to-speech is not enabled in this environment "
            "(settings.tts_engine='none'). Run the Phase 0 bake-off "
            "(scripts/eval_tts.py) and set tts_engine to its winner -- see "
            "docs/architecture/voice-assistant.md."
        )


class PiperEngine:
    """settings.tts_engine='piper'. CPU-only ONNX synthesis via the
    `piper-tts` pip package (installed and pinned to 1.8.0 by
    config/Dockerfile.gpu; confirmed working end to end 2026-09-06 via
    scripts/voice_selftest.py, French and Arabic-script Darija both).

    Voice models are NOT bundled with the package -- download the
    {voice}.onnx + {voice}.onnx.json pair for settings.tts_voice_fr and
    tts_voice_ar from Piper's voice catalogue
    (https://github.com/rhasspy/piper/blob/master/VOICES.md) into
    settings.tts_voice_dir before enabling this engine.

    Loaded voices are cached per-process (self._voices) -- a ~50-150MB ONNX
    load per sentence would defeat the whole point of a low-latency engine.
    """

    name = "piper"
    # Last-resort fallback, used only when neither configured voice's
    # .onnx.json can be read (missing/corrupt download). Piper's catalogue
    # standardizes almost every voice on 22050Hz, and both currently
    # configured voices were confirmed at 22050 on disk 2026-09-06 -- but
    # that is no longer *assumed*: see the sample_rate property.
    _FALLBACK_SAMPLE_RATE = 22050

    def __init__(self) -> None:
        self._voices: dict[str, object] = {}
        self._declared_sample_rate: Optional[int] = None

    @property
    def sample_rate(self) -> int:
        """Cached read of settings.tts_voice_fr/tts_voice_ar's .onnx.json --
        a few KB of JSON, no ONNX load.

        Read from the CONFIG rather than from an already-loaded voice on
        purpose. app.routers.voice sends this to the client in
        `audio.start`, which happens BEFORE the turn's first synthesize()
        call, so a loaded-voice lookup would report a guess on a process's
        first turn and the truth afterwards -- and would also report
        whichever language spoke LAST, changing mid-session. The config is
        known up front and is stable for the whole session.

        KNOWN LIMITATION, surfaced rather than hidden: TtsEngine.sample_rate
        is one rate per ENGINE, but Piper voices carry a rate each. Today
        both configured voices are 22050 (verified on disk 2026-09-06) so
        the distinction is moot; if a future voice swap (ADR 0006's Darija
        replacement) introduces a genuinely different rate, this logs a
        warning naming both, and the real fix at that point is a
        per-language rate on the Protocol itself -- not a different guess
        here.
        """
        if self._declared_sample_rate is not None:
            return self._declared_sample_rate

        settings = get_settings()
        rates: dict[str, int] = {}
        for voice_name in (settings.tts_voice_fr, settings.tts_voice_ar):
            config_path = Path(settings.tts_voice_dir) / f"{voice_name}.onnx.json"
            try:
                rates[voice_name] = json.loads(config_path.read_text(encoding="utf-8"))["audio"]["sample_rate"]
            except (OSError, ValueError, KeyError, TypeError):
                continue

        if not rates:
            self._declared_sample_rate = self._FALLBACK_SAMPLE_RATE
        else:
            if len(set(rates.values())) > 1:
                logger.warning(
                    "Piper voices declare different sample rates (%s) -- app.routers.voice "
                    "sends ONE rate per session in audio.start, so one language will play at "
                    "the wrong pitch/speed. Fix by giving TtsEngine a per-language rate.",
                    rates,
                )
            self._declared_sample_rate = rates.get(settings.tts_voice_fr, next(iter(rates.values())))
        return self._declared_sample_rate

    def _load_voice(self, voice_name: str):
        if voice_name in self._voices:
            return self._voices[voice_name]
        try:
            from piper import PiperVoice
        except ImportError as e:
            raise TtsUnavailableError(
                "settings.tts_engine='piper' but the `piper-tts` package is "
                "not installed -- see config/requirements.txt."
            ) from e

        settings = get_settings()
        model_path = Path(settings.tts_voice_dir) / f"{voice_name}.onnx"
        config_path = Path(settings.tts_voice_dir) / f"{voice_name}.onnx.json"
        if not model_path.exists() or not config_path.exists():
            raise TtsUnavailableError(
                f"Piper voice {voice_name!r} not found under "
                f"{settings.tts_voice_dir} (settings.tts_voice_dir). Download "
                f"{voice_name}.onnx and {voice_name}.onnx.json from Piper's "
                f"voice catalogue first."
            )
        voice = PiperVoice.load(str(model_path), config_path=str(config_path))
        self._voices[voice_name] = voice
        return voice

    def synthesize(self, text: str, *, language: str) -> bytes:
        settings = get_settings()
        voice_name = settings.tts_voice_fr if language == "fr" else settings.tts_voice_ar
        voice = self._load_voice(voice_name)

        # PiperVoice.synthesize yields AudioChunk objects (piper-tts>=1.3)
        # with a raw int16 PCM `.audio_int16_bytes` payload; concatenated
        # here into one buffer per sentence-chunk, matching this engine's
        # synchronous, one-chunk-in/one-chunk-out contract. A future
        # streaming-within-a-sentence optimization would yield these
        # directly instead -- not needed while whole sentences are already
        # the streaming granularity (see app.routers.voice).
        buffer = bytearray()
        for audio_chunk in voice.synthesize(text):
            buffer.extend(audio_chunk.audio_int16_bytes)
        if not buffer:
            raise TtsUnavailableError(f"Piper produced no audio for voice {voice_name!r}")
        return bytes(buffer)


class _ResidentTtsWorker:
    """Manages ONE persistent subprocess running
    scripts/tts_worker_resident.py in its own venv, reused for every
    sentence instead of cold-loading a 5.6GB checkpoint per call.

    Deliberately the SAME shape as app.services.stt._ResidentSttWorker
    (JSON-lines IPC over stdin/stdout, a drain thread per stream, idle
    self-release, kill-and-restart on any failure) -- same problem, same
    already-proven solution. Kept as a distinct class rather than a shared
    base for the same reason stt.py gives: the two workers' payloads differ
    (audio path + language hint vs. text + language + output path), and
    neither is written as a base class today.
    """

    def __init__(
        self, venv_python: str, worker_script: str, *,
        idle_release_seconds: float = 300.0, env: Optional[dict] = None,
    ):
        import queue
        import threading

        self._venv_python = venv_python
        self._worker_script = worker_script
        self._env = env
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
                logger.debug("tts_worker_resident: %s", line.rstrip())
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
                    "resident TTS worker idle for %.0fs -- released to free VRAM "
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
        # The worker reads its model paths from the environment (it runs in
        # a venv that deliberately has no pydantic-settings / app imports --
        # same contract as scripts/speech_worker_resident.py), so settings
        # are handed over explicitly rather than relying on inheritance:
        # they come from .env, which never reaches os.environ.
        import os as _os

        child_env = None
        if self._env:
            child_env = {**_os.environ, **{k: v for k, v in self._env.items() if v}}
        self._proc = subprocess.Popen(
            [self._venv_python, self._worker_script],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1, env=child_env,
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

    def synthesize(self, text: str, *, language: str, out_path: str, timeout: float) -> dict:
        import queue

        with self._lock:
            self._ensure_alive()
            req = json.dumps({
                "cmd": "synthesize", "id": "1", "text": text,
                "language": language, "out": out_path,
            }, ensure_ascii=False)
            try:
                self._proc.stdin.write(req + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                self._kill()
                raise TtsUnavailableError(f"resident TTS worker's stdin pipe broke: {e}") from e

            try:
                line = self._out_q.get(timeout=timeout)
            except queue.Empty:
                self._kill()
                raise TtsUnavailableError(
                    f"resident TTS worker did not respond within {timeout}s -- killed and "
                    f"will restart on the next call."
                )
            if line is None:
                self._kill()
                raise TtsUnavailableError("resident TTS worker process exited unexpectedly.")
            resp = json.loads(line)
            self._arm_idle_timer()
            if not resp.get("ok"):
                raise TtsUnavailableError(f"resident TTS worker error: {resp.get('error', '')}")
            return resp


_resident_tts_worker: Optional[_ResidentTtsWorker] = None


def _get_resident_tts_worker(venv_python: str, worker_script: str) -> _ResidentTtsWorker:
    global _resident_tts_worker
    if _resident_tts_worker is None:
        settings = get_settings()
        _resident_tts_worker = _ResidentTtsWorker(
            venv_python, worker_script,
            idle_release_seconds=settings.tts_worker_idle_release_seconds,
            env={
                "TTS_XTTS_MODEL_DIR": settings.tts_xtts_model_dir,
                "TTS_XTTS_SPEAKER_REF": settings.tts_xtts_speaker_ref,
                "FFMPEG_SHARED_BIN": settings.tts_xtts_ffmpeg_bin,
            },
        )
    return _resident_tts_worker


class XttsDarijaEngine:
    """settings.tts_engine='xtts_darija' -- `medmac01/darija_xtt_2.0`, an
    XTTS-v2 fine-tune on Moroccan Darija, chosen on live listening as the
    best-sounding Darija voice found (ADR 0006's own acceptance test is
    listening, not a synthesis-speed number).

    NOT the default, and the default must not be changed to this without a
    licensing decision: the checkpoint declares no license and therefore
    inherits XTTS-v2's CPML, which forbids commercial use -- the same wall
    that rejected XTTS-v2 for this project once already (see this module's
    docstring and scripts/tts_worker_resident.py). Selecting it is a
    deliberate, per-deployment act.

    Unlike PiperEngine this is GPU-resident and multilingual: XTTS-v2
    handles French natively too, so this engine serves BOTH languages
    rather than being spliced with Piper. That keeps one sample rate
    (24000) for the whole session, which app/routers/voice.py's `audio.start`
    contract requires -- a Piper/XTTS hybrid would need a per-language rate
    on TtsEngine itself. Cost, measured 2026-09-06: RTF 0.39-0.69 vs
    Piper's ~0.05, and it occupies VRAM Piper does not.
    """

    name = "xtts_darija"
    _WORKER_SCRIPT = "scripts/tts_worker_resident.py"
    # Generous: the FIRST call pays the ~5.6GB checkpoint load plus speaker
    # latent computation. Later calls are seconds.
    _TIMEOUT_SECONDS = 600
    sample_rate = 24000  # matches scripts/tts_worker_resident.py's SAMPLE_RATE

    def warmup(self) -> None:
        """Force the checkpoint load + speaker-latent computation now.

        Measured 2026-09-06: the first synthesize() costs ~51s (5.6GB
        checkpoint, then conditioning latents), every later one ~2s. Without
        this, that ~50s lands on a user's first spoken sentence, inside a
        live voice session, looking exactly like a hang. app/main.py calls
        it at startup for the same reason it preloads bge-m3 there.

        Best-effort: a warmup failure must not stop the server from booting
        -- the real synthesize() call will surface the error properly to the
        client (as a tts_failed event) if the engine is genuinely broken.
        """
        try:
            self.synthesize("مرحبا", language="darija")
        except Exception:
            logger.exception("XTTS warmup failed -- first real synthesis will pay the load cost")

    def synthesize(self, text: str, *, language: str) -> bytes:
        import os
        import tempfile
        from pathlib import Path

        settings = get_settings()
        venv_python = Path(settings.tts_xtts_venv_python)
        if not venv_python.exists():
            raise TtsUnavailableError(
                f"settings.tts_engine='xtts_darija' but the dedicated TTS venv's interpreter "
                f"was not found at {venv_python} (settings.tts_xtts_venv_python)."
            )
        worker_script = Path(__file__).resolve().parents[2] / self._WORKER_SCRIPT
        if not worker_script.exists():
            raise TtsUnavailableError(f"TTS worker script not found: {worker_script}")

        with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as f:
            out_path = f.name
        try:
            worker = _get_resident_tts_worker(str(venv_python), str(worker_script))
            worker.synthesize(
                text, language=language, out_path=out_path, timeout=self._TIMEOUT_SECONDS,
            )
            audio = Path(out_path).read_bytes()
        finally:
            try:
                os.unlink(out_path)
            except OSError:
                pass

        if not audio:
            raise TtsUnavailableError("XTTS produced no audio.")
        return audio


_ENGINES = {
    "none": NullTtsEngine,
    "piper": PiperEngine,
    "xtts_darija": XttsDarijaEngine,
}


@lru_cache(maxsize=1)
def get_tts_engine() -> TtsEngine:
    """Cached singleton, keyed off settings.tts_engine at first call --
    same read-once-per-process contract as app.services.ocr.get_ocr_engine.
    """
    engine_name = get_settings().tts_engine
    engine_cls = _ENGINES.get(engine_name)
    if engine_cls is None:
        raise TtsUnavailableError(
            f"Unknown settings.tts_engine={engine_name!r}. Valid values: {sorted(_ENGINES)}."
        )
    return engine_cls()
