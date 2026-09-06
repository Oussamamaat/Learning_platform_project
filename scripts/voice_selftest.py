"""
Offline voice pipeline round-trip -- no FastAPI, no WebSocket, no Ollama,
no Postgres. Settles two live-reported defects
(POST_LEASE_MVP_SPRINT_PLAN.md voice items) on the laptop, for free:

  1. "No voice is heard" -- app/routers/voice.py's speak() used to catch
     only TtsUnavailableError, silently swallowing any other Piper
     exception (piper-tts API mismatch, missing espeak-ng, bad ONNX load)
     while the client still received audio.end and looked healthy. This
     script calls PiperEngine.synthesize() directly and prints the full
     traceback on any failure -- nothing here can hide behind an
     asyncio background task the way the live session did.
  2. "Doesn't detect Arabic after a French turn" -- app/routers/voice.py
     used to pin every session to turn 1's language and feed that pin
     into whisper's language_hint, disabling auto-detection. This script
     feeds each synthesized sentence back through STT with
     language_hint=None and checks the DETECTED language against the
     sentence's actual language.

Reuses scripts/eval_tts.py's TEST_SENTENCES (already covers French and
Arabic-script Darija) and scripts/calibrate_vad.py's frame-feeding
pattern against the real EnergyEndpointer.

Requires (per POST_LEASE_MVP_SPRINT_PLAN.md's local voice setup):
  - .gguf_venv: piper-tts installed, data/tts_voices/{fr_FR-siwis-medium,
    ar_JO-kareem-medium}.{onnx,onnx.json} downloaded.
  - .speech_venv (settings.stt_venv_python): faster-whisper + ctranslate2
    + soundfile (no torch needed -- see config/requirements-speech.txt's
    own comment on why SeamlessM4T alone needs torch).
  - .env: STT_ENGINE=whisper, STT_MODEL=large-v3-turbo.

Run (from repo root):
    .gguf_venv/Scripts/python.exe scripts/voice_selftest.py

Uses ZERO LLM VRAM -- Ollama does not need to be running.
"""
import sys
import time
import traceback
import wave
from pathlib import Path

# Windows console default codepage (cp1252) cannot encode Arabic script --
# this script prints Darija test sentences and their transcripts directly.
# Confirmed crash 2026-09-06 (UnicodeEncodeError mid-run, after the French
# rows had already printed) without this.
sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.services.vad import EnergyEndpointer, FRAME_BYTES, SAMPLE_RATE  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent

TEST_SENTENCES = {
    "fr": [
        "Le port du casque est obligatoire dans cette zone.",
        "Selon l'article 8, la consigne doit être vérifiée avant toute intervention.",
    ],
    "darija": [
        "خصك تلبس الكاسك ديال الحماية.",
        "على حساب المادة 8، خاصك تتأكد من العزلة قبل ما تبدا.",
    ],
}

# speech_worker_resident.py's own decoded-language codes (see its
# _WHISPER_LANG_MAP / faster-whisper's ISO 639-1 detection) -- "ar" covers
# both MSA and Darija since whisper has no separate Darija class.
_EXPECTED_STT_LANG = {"fr": "fr", "darija": "ar"}


def _write_wav(path: Path, pcm_bytes: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def _resample_linear(pcm_bytes: bytes, orig_sr: int, target_sr: int) -> bytes:
    """Quick linear-interpolation resample -- good enough for THIS
    round-trip test (proving the pipeline runs end to end, not judging
    audio quality; eval_tts.py's own .wav files are what you actually
    listen to). Avoids adding scipy/librosa to .gguf_venv for a self-test."""
    import numpy as np

    if orig_sr == target_sr:
        return pcm_bytes
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float64)
    n_out = int(len(samples) * target_sr / orig_sr)
    x_old = np.linspace(0, 1, num=len(samples), endpoint=False)
    x_new = np.linspace(0, 1, num=n_out, endpoint=False)
    resampled = np.interp(x_new, x_old, samples)
    return resampled.astype(np.int16).tobytes()


def _frames(pcm_bytes: bytes) -> list[bytes]:
    return [pcm_bytes[i:i + FRAME_BYTES] for i in range(0, len(pcm_bytes) - FRAME_BYTES + 1, FRAME_BYTES)]


def main() -> None:
    from app.config import get_settings
    from app.services.tts import PiperEngine, TtsUnavailableError
    from app.services.stt import get_stt_engine, SttUnavailableError

    settings = get_settings()
    print(f"tts_engine={settings.tts_engine!r} stt_engine={settings.stt_engine!r} "
          f"stt_model={settings.stt_model!r}\n")

    tts = PiperEngine()
    try:
        stt = get_stt_engine()
    except SttUnavailableError as e:
        print(f"STT unavailable, aborting: {e}")
        sys.exit(1)

    results = []
    for language, sentences in TEST_SENTENCES.items():
        for i, sentence in enumerate(sentences):
            row = {"language": language, "sentence": sentence}
            print(f"--- {language} #{i}: {sentence[:50]}")

            # 1. Synthesis -- the actual swallowed-exception bug lived here.
            t0 = time.time()
            try:
                audio = tts.synthesize(sentence, language=language)
                row["tts_ok"] = True
                row["tts_seconds"] = round(time.time() - t0, 2)
                row["reported_sample_rate"] = tts.sample_rate
            except (TtsUnavailableError, Exception) as e:
                row["tts_ok"] = False
                row["tts_error"] = f"{type(e).__name__}: {e}"
                print(f"  TTS FAILED: {type(e).__name__}: {e}")
                traceback.print_exc()
                results.append(row)
                continue

            out_wav = OUT_DIR / f"voice_selftest_{language}_{i}.wav"
            _write_wav(out_wav, audio, tts.sample_rate)
            print(f"  TTS ok in {row['tts_seconds']}s, sample_rate={tts.sample_rate} -> {out_wav.name}")

            # 2. VAD -- does the endpointer fire on our own synthesized speech?
            pcm_16k = _resample_linear(audio, tts.sample_rate, SAMPLE_RATE)
            ep = EnergyEndpointer(
                threshold=settings.vad_threshold, hangover_ms=settings.vad_hangover_ms,
                min_speech_ms=settings.vad_min_speech_ms,
            )
            fired = False
            for frame in _frames(pcm_16k):
                if ep.push(frame) == "speech_start":
                    fired = True
            row["vad_fired"] = fired
            print(f"  VAD speech_start fired: {fired}")

            # 3. STT -- language_hint=None always (the actual fix for the
            # language-switching bug): does auto-detect land on the right
            # language for what we just synthesized?
            t0 = time.time()
            try:
                transcript = stt.transcribe(pcm_16k, sample_rate=SAMPLE_RATE, language_hint=None)
                row["stt_ok"] = True
                row["stt_seconds"] = round(time.time() - t0, 2)
                row["transcript"] = transcript.text
                row["detected_language"] = transcript.language
                expected = _EXPECTED_STT_LANG[language]
                row["language_match"] = transcript.language == expected
                print(f"  STT ok in {row['stt_seconds']}s: {transcript.text!r} "
                      f"(detected={transcript.language}, expected={expected}, "
                      f"match={row['language_match']})")
            except SttUnavailableError as e:
                row["stt_ok"] = False
                row["stt_error"] = str(e)
                print(f"  STT FAILED: {e}")

            results.append(row)
            print()

    n = len(results)
    tts_ok = sum(1 for r in results if r.get("tts_ok"))
    vad_ok = sum(1 for r in results if r.get("vad_fired"))
    stt_ok = sum(1 for r in results if r.get("stt_ok"))
    lang_ok = sum(1 for r in results if r.get("language_match"))
    print("=" * 60)
    print(f"TTS synthesis:      {tts_ok}/{n}")
    print(f"VAD fired:          {vad_ok}/{n}")
    print(f"STT succeeded:      {stt_ok}/{n}")
    print(f"Language detected correctly: {lang_ok}/{n}")
    if tts_ok < n:
        print("\nTTS failures found -- THIS is what was silently swallowed live. See tracebacks above.")
    if lang_ok < n:
        print("\nLanguage mismatches found -- check .env's STT_MODEL and that language_hint is really None.")

    import json
    (OUT_DIR / "voice_selftest_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote {OUT_DIR / 'voice_selftest_results.json'} and per-sentence .wav files next to this script.")


if __name__ == "__main__":
    main()
