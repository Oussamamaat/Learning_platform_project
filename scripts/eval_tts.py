"""
TTS engine bake-off -- Phase 0 of docs/architecture/voice-assistant.md.
Never run on this laptop: needs `piper-tts` installed plus its downloaded
voice models (neither is present -- see app/services/tts.py's PiperEngine
docstring). Mirrors scripts/ocr_bakeoff.py's shape: a measurement tool,
never asserts / exits nonzero, writes a JSON result file plus one .wav per
(voice, sentence) next to itself FOR MANUAL LISTENING -- "does this sound
acceptably like Darija" is not a number this script can produce; it can
only measure latency and confirm the pipeline runs, then hand you the
audio to judge.

Test sentences deliberately include: a plain sentence, a sentence with a
number/citation-shaped token (Article numbers are exactly what
app.services.llm.inject_citations rewrites in text chat, and voice
excludes them from the SPOKEN stream by design -- see
stream_llm_response's docstring -- but TTS still needs to pronounce
ordinary numbers correctly elsewhere in an answer), and a
longer multi-clause sentence (Piper's prosody tends to degrade on run-ons
more than on short sentences).

Run (once piper-tts + voice models exist; from repo root, .gguf_venv):
    .gguf_venv/Scripts/python.exe scripts/eval_tts.py
"""
import json
import sys
import time
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = Path(__file__).resolve().parent

TEST_SENTENCES = {
    "fr": [
        "Le port du casque est obligatoire dans cette zone.",
        "Selon l'article 8, la consigne doit être vérifiée avant toute intervention.",
        "Avant de commencer, assurez-vous que l'équipement est correctement isolé, "
        "que les vannes sont fermées, et que le permis de travail a été signé par le responsable.",
    ],
    "darija": [
        "خصك تلبس الكاسك ديال الحماية.",
        "على حساب المادة 8، خاصك تتأكد من العزلة قبل ما تبدا.",
        "قبل ما تبدا الخدمة، تأكد أن التجهيزات معزولة مزيان وأن الترخيص ديال الخدمة موقع.",
    ],
}


def _write_wav(path: Path, pcm_bytes: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def main() -> None:
    from app.services.tts import PiperEngine, TtsUnavailableError

    engine = PiperEngine()
    results: dict[str, list[dict]] = {}

    for language, sentences in TEST_SENTENCES.items():
        per_language = []
        for i, sentence in enumerate(sentences):
            t0 = time.time()
            try:
                audio = engine.synthesize(sentence, language=language)
                elapsed = time.time() - t0
                duration = len(audio) / 2 / engine.sample_rate  # 16-bit mono
                out_path = OUT_DIR / f"eval_tts_{language}_{i}.wav"
                _write_wav(out_path, audio, engine.sample_rate)
                per_language.append({
                    "sentence": sentence, "ok": True, "error": None,
                    "seconds": round(elapsed, 2),
                    "rtf": round(elapsed / duration, 3) if duration > 0 else None,
                    "wav": str(out_path.name),
                })
                print(f"{language:<8} #{i} {elapsed:>5.2f}s  rtf={per_language[-1]['rtf']}  -> {out_path.name}")
            except TtsUnavailableError as e:
                elapsed = time.time() - t0
                per_language.append({
                    "sentence": sentence, "ok": False, "error": str(e), "seconds": round(elapsed, 2),
                })
                print(f"{language:<8} #{i} ERROR: {e}")
        results[language] = per_language

    (OUT_DIR / "eval_tts_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote {OUT_DIR / 'eval_tts_results.json'} and per-sentence .wav files next to this script.")
    print("Listen to the .wav files before deciding -- latency numbers alone cannot judge Darija prosody.")


if __name__ == "__main__":
    main()
