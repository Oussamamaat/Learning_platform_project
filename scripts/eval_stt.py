"""
STT engine bake-off -- Phase 0 of docs/architecture/voice-assistant.md.
Never run on this laptop: needs both real labeled Darija/French/code-
switched audio (not committed to this repo -- collect it first, see
below) and a GPU this laptop's 8GB card cannot spare alongside the
resident tutor model. Mirrors scripts/ocr_bakeoff.py's shape: a
measurement tool, never asserts / exits nonzero, writes a JSON result
file next to itself.

Candidates (see app/services/stt.py's module docstring for why these
three and not others):
  whisper   -- faster-whisper, settings.stt_model (e.g. "large-v3-turbo").
               Best-in-class French; Darija via stock Whisper's "ar" is
               expected to be MSA-biased -- this run either confirms or
               refutes that.
  seamless  -- SeamlessM4T-v2-large, source lang "ary" (Moroccan Arabic
               specifically, not generic MSA "arb").

Run (once a GPU + labeled audio exist; from repo root, .gguf_venv):
    .gguf_venv/Scripts/python.exe scripts/eval_stt.py

Expects a directory of (audio, reference transcript) pairs:
    tests/data/voice_eval/<id>.wav          -- 16kHz mono PCM WAV
    tests/data/voice_eval/<id>.txt          -- reference transcript (UTF-8)
    tests/data/voice_eval/<id>.lang         -- "fr" | "ary" (one word)
Naming convention only, not enforced by any other code -- build this set
by recording (or sourcing, e.g. atlasia/DODa-audio-dataset for Darija)
real utterances, INCLUDING code-switched ones: that is the case ad hoc
manual testing is most likely to miss and where a bake-off earns its keep.

WER here is a simple whitespace-tokenized Levenshtein word error rate --
deliberately not the `jiwer` package (not a project dependency) since this
script's job is a relative ranking between candidates, not a
publication-grade WER with punctuation/casing normalization rules.
"""
import json
import sys
import time
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

EVAL_DIR = REPO_ROOT / "tests" / "data" / "voice_eval"
OUT_DIR = Path(__file__).resolve().parent
ENGINES = ("whisper", "seamless")


def _word_error_rate(reference: str, hypothesis: str) -> float:
    ref = reference.split()
    hyp = hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0

    # Standard Levenshtein DP over words (substitutions/insertions/
    # deletions all cost 1) -- O(len(ref) * len(hyp)), fine at utterance
    # length.
    prev = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        curr = [i] + [0] * len(hyp)
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1] / len(ref)


def _wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def _load_examples() -> list[dict]:
    examples = []
    for wav_path in sorted(EVAL_DIR.glob("*.wav")):
        txt_path = wav_path.with_suffix(".txt")
        lang_path = wav_path.with_suffix(".lang")
        if not txt_path.exists():
            print(f"skipping {wav_path.name}: no matching .txt reference")
            continue
        examples.append({
            "id": wav_path.stem,
            "wav": wav_path,
            "reference": txt_path.read_text(encoding="utf-8").strip(),
            "language_hint": lang_path.read_text(encoding="utf-8").strip() if lang_path.exists() else None,
        })
    return examples


def main() -> None:
    if not EVAL_DIR.exists() or not any(EVAL_DIR.glob("*.wav")):
        print(
            f"No labeled audio found under {EVAL_DIR}.\n"
            "This bake-off needs real (audio, reference transcript) pairs before it can run "
            "-- see this file's module docstring for the naming convention and where to "
            "source Darija/French/code-switched samples (e.g. atlasia/DODa-audio-dataset)."
        )
        return

    from app.services.stt import WhisperEngine, SeamlessEngine

    engines = {"whisper": WhisperEngine(), "seamless": SeamlessEngine()}
    examples = _load_examples()
    print(f"Loaded {len(examples)} labeled example(s) from {EVAL_DIR}")

    results: dict[str, dict] = {}
    for engine_name in ENGINES:
        engine = engines[engine_name]
        per_engine = []
        for ex in examples:
            audio_bytes = ex["wav"].read_bytes()
            duration = _wav_duration_seconds(ex["wav"])
            t0 = time.time()
            try:
                chunk = engine.transcribe(audio_bytes, language_hint=ex["language_hint"])
                elapsed = time.time() - t0
                wer = _word_error_rate(ex["reference"], chunk.text)
                ok, error, hypothesis = True, None, chunk.text
            except Exception as e:
                elapsed = time.time() - t0
                ok, error, hypothesis, wer = False, f"{type(e).__name__}: {e}", "", None

            per_engine.append({
                "id": ex["id"], "ok": ok, "error": error, "seconds": round(elapsed, 2),
                "rtf": round(elapsed / duration, 3) if duration > 0 else None,
                "wer": round(wer, 3) if wer is not None else None,
                "reference": ex["reference"], "hypothesis": hypothesis,
            })
            status = f"WER={wer:.2f}" if ok else f"ERROR: {error}"
            print(f"{engine_name:<10} {ex['id']:<20} {elapsed:>6.2f}s  {status}", flush=True)

        valid = [r for r in per_engine if r["wer"] is not None]
        results[engine_name] = {
            "examples": per_engine,
            "mean_wer": round(sum(r["wer"] for r in valid) / len(valid), 3) if valid else None,
            "mean_rtf": round(sum(r["rtf"] for r in valid) / len(valid), 3) if valid else None,
            "failures": len(per_engine) - len(valid),
        }

    (OUT_DIR / "eval_stt_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote {OUT_DIR / 'eval_stt_results.json'}")
    for name, summary in results.items():
        print(f"{name}: mean_wer={summary['mean_wer']} mean_rtf={summary['mean_rtf']} failures={summary['failures']}")


if __name__ == "__main__":
    main()
