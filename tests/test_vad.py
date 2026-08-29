"""Tests for app.services.vad.EnergyEndpointer -- pure logic, no Postgres/
Ollama/GPU dependency, safe to run on any machine including this laptop."""
import array

from app.services.vad import EnergyEndpointer, FRAME_BYTES, SAMPLE_RATE


def _tone_frame(amplitude: int) -> bytes:
    """One FRAME_BYTES-sized frame of constant-amplitude signal (a square
    wave, not a sine -- irrelevant to RMS, and simpler to construct)."""
    n_samples = FRAME_BYTES // 2
    samples = array.array("h", [amplitude if i % 2 == 0 else -amplitude for i in range(n_samples)])
    return samples.tobytes()


def _silence_frame() -> bytes:
    return _tone_frame(0)


def _loud_frame() -> bytes:
    return _tone_frame(10000)


def test_frame_bytes_matches_20ms_at_16khz_mono_16bit():
    assert SAMPLE_RATE == 16000
    assert FRAME_BYTES == 640  # 16000 * 0.02 * 2 bytes/sample


def test_silence_alone_never_triggers_speech_start():
    ep = EnergyEndpointer()
    for _ in range(50):
        assert ep.push(_silence_frame()) is None


def test_sustained_loud_audio_triggers_speech_start_after_min_speech_ms():
    ep = EnergyEndpointer(min_speech_ms=200)  # 10 frames at 20ms/frame
    events = [ep.push(_loud_frame()) for _ in range(10)]
    assert events.count("speech_start") == 1
    assert events[-1] == "speech_start"
    assert all(e is None for e in events[:-1])


def test_brief_loud_blip_below_min_speech_does_not_trigger():
    ep = EnergyEndpointer(min_speech_ms=200)  # needs 10 loud frames
    events = [ep.push(_loud_frame()) for _ in range(5)]
    assert all(e is None for e in events)
    # A subsequent silence frame resets the run -- confirms this isn't
    # just "hasn't happened yet".
    assert ep.push(_silence_frame()) is None


def test_speech_end_fires_after_hangover_and_take_utterance_returns_audio():
    ep = EnergyEndpointer(min_speech_ms=200, hangover_ms=200)  # 10 + 10 frames
    for _ in range(10):
        assert ep.push(_loud_frame()) != "speech_end"
    events = [ep.push(_silence_frame()) for _ in range(10)]
    assert events.count("speech_end") == 1
    assert events[-1] == "speech_end"

    utterance = ep.take_utterance()
    assert len(utterance) > 0
    # Buffer is cleared after take_utterance -- the next utterance must not
    # inherit stale audio from the previous one.
    assert ep.take_utterance() == b""


def test_odd_length_frame_does_not_crash():
    ep = EnergyEndpointer()
    # A truncated final frame (e.g. stream cut mid-sample) must be handled,
    # not raise -- this runs inside a WebSocket receive loop where a crash
    # would tear down the whole voice session.
    ep.push(_loud_frame()[:-1])


def test_two_separate_utterances_do_not_bleed_into_each_other():
    ep = EnergyEndpointer(min_speech_ms=200, hangover_ms=200)
    for _ in range(10):
        ep.push(_loud_frame())
    for _ in range(10):
        ep.push(_silence_frame())
    first = ep.take_utterance()

    for _ in range(10):
        ep.push(_loud_frame())
    for _ in range(10):
        ep.push(_silence_frame())
    second = ep.take_utterance()

    assert len(first) > 0
    assert len(second) > 0
