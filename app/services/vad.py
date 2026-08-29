"""
Voice Activity Detection (endpointing)
───────────────────────────────────────
EnergyEndpointer: a dependency-free RMS-threshold VAD for
app/routers/voice.py's open-mic session. A REAL, working default -- not a
placeholder -- but coarser than Silero VAD, which
docs/architecture/voice-assistant.md names as the intended upgrade:
Silero needs onnxruntime, not yet a project dependency, and pulling in a
new heavy dependency for VAD specifically was judged not worth it before
the Phase 0 STT/TTS bake-off (which already needs new dependencies) has
even landed. Swap this out once that dependency is already being added.

Deliberately not built on the stdlib `audioop` module (RMS in one call)
despite being simpler: audioop is deprecated and REMOVED in Python 3.13
(PEP 594) -- this project pins Python 3.12 today (CLAUDE.md), but a VAD
this trivial has no reason to take on a dependency with a known removal
date. Pure `array`-module arithmetic instead.
"""
import array
from typing import Optional

SAMPLE_RATE = 16000
FRAME_MS = 20
# 16-bit mono PCM @16kHz, 20ms/frame -- the frame size
# frontend/src/hooks/useVoiceSession.ts's AudioWorklet is expected to send.
FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * 2


class EnergyEndpointer:
    """Feed it FRAME_BYTES-sized 16-bit PCM frames, in order, from one
    audio stream. Not safe to call from multiple threads concurrently --
    app/routers/voice.py owns exactly one instance per session and only
    ever calls it from the WebSocket's own receive loop.

    `threshold` is a fixed RMS floor, not adaptive to ambient noise --
    the honest limitation named in this module's docstring. Tune per
    deployment (a laptop mic in a quiet room vs. an office) until this is
    replaced.
    """

    def __init__(
        self,
        *,
        threshold: float = 500.0,
        hangover_ms: int = 400,
        min_speech_ms: int = 200,
    ):
        self._threshold = threshold
        self._hangover_frames = max(1, hangover_ms // FRAME_MS)
        self._min_speech_frames = max(1, min_speech_ms // FRAME_MS)
        self._speaking = False
        self._silence_run = 0
        self._speech_run = 0
        self._buffer = bytearray()

    @staticmethod
    def _rms(frame: bytes) -> float:
        samples = array.array("h")
        try:
            samples.frombytes(frame)
        except ValueError:
            # An odd byte count (a truncated final frame at stream end) --
            # drop the dangling byte rather than raise; one lost sample is
            # inaudible and this must never crash the receive loop.
            samples.frombytes(frame[: len(frame) - (len(frame) % 2)])
        if not samples:
            return 0.0
        return (sum(s * s for s in samples) / len(samples)) ** 0.5

    def push(self, frame: bytes) -> Optional[str]:
        """Feed one frame. Returns "speech_start" the frame min_speech_ms
        of continuous loud audio is confirmed, "speech_end" once
        hangover_ms of continuous quiet follows confirmed speech, or None
        otherwise. Audio is accumulated into the internal buffer for the
        whole span between speech_start and speech_end (inclusive of the
        confirming frames on both ends) -- retrieve it with
        take_utterance()."""
        level = self._rms(frame)
        is_loud = level >= self._threshold

        if is_loud:
            self._silence_run = 0
            self._speech_run += 1
            if self._speaking:
                self._buffer.extend(frame)
                return None
            self._buffer.extend(frame)
            if self._speech_run >= self._min_speech_frames:
                self._speaking = True
                return "speech_start"
            return None

        self._speech_run = 0
        if not self._speaking:
            return None
        self._buffer.extend(frame)
        self._silence_run += 1
        if self._silence_run >= self._hangover_frames:
            self._speaking = False
            self._silence_run = 0
            return "speech_end"
        return None

    def take_utterance(self) -> bytes:
        """Return and clear the buffered audio for the utterance that just
        ended (call after a "speech_end" event)."""
        audio = bytes(self._buffer)
        self._buffer.clear()
        return audio
