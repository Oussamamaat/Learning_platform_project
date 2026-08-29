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
import logging
from functools import lru_cache
from pathlib import Path
from typing import Protocol

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
    `piper-tts` pip package (not yet in config/requirements.txt -- add it
    alongside the Phase 0 decision).

    Voice models are NOT bundled with the package -- download the
    {voice}.onnx + {voice}.onnx.json pair for settings.tts_voice_fr and
    tts_voice_ar from Piper's voice catalogue
    (https://github.com/rhasspy/piper/blob/master/VOICES.md) into
    settings.tts_voice_dir before enabling this engine.

    Loaded voices are cached per-process (self._voices) -- a ~50-150MB ONNX
    load per sentence would defeat the whole point of a low-latency engine.
    """

    name = "piper"
    # Piper's own voice catalogue standardizes almost every voice
    # (including fr_FR-siwis-medium and ar_JO-kareem-medium) on 22050Hz --
    # see this class's docstring / TtsEngine.sample_rate's comment for why
    # this is a fixed constant rather than read per-voice from each loaded
    # PiperVoice.config.sample_rate. Verify against the actual downloaded
    # voice configs once the Phase 0 bake-off picks final voices.
    sample_rate = 22050

    def __init__(self) -> None:
        self._voices: dict[str, object] = {}

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


_ENGINES = {
    "none": NullTtsEngine,
    "piper": PiperEngine,
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
