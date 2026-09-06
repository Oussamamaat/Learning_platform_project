"""
Tashkeel (Arabic diacritics) A/B eval for the Darija XTTS voice
(medmac01/darija_xtt_2.0, evaluated in scripts/eval_darija_xtts.py).

RESULT (2026-09-07): RUN, LISTENED TO, REJECTED. User's verdict on the paired
bare/diacritized output: "without tashkeel is better, tashkeel is trash." Per
this eval's own decision gate (see bottom of this docstring), tashkeel is NOT
wired into production and this script is kept for reference only, so the idea
isn't re-tried blind later -- see docs/LESSONS_LEARNED.md #12 for the short
version and why (no Darija-specific diacritizer exists, and the fine-tune was
never trained on diacritized text -- both were flagged as real risks BEFORE
this ran, and both are exactly what sank it).

EVAL-ONLY, LISTENING-DECIDES. Not wired into anything -- no settings flag,
no production code path touches this. See docs/architecture/rectified/
(the plan this script came out of) for the full investigation; the short
version:

  - There is no existing text normalization anywhere between the LLM's
    answer and XTTS's model.inference() call (checked: llm.py -> voice.py's
    speak() -> tts.py's XttsDarijaEngine.synthesize() -> this checkpoint's
    tts_worker_resident.py -- the only transform anywhere is `.strip()`).
    A tashkeel step would be entirely new, not replacing an existing one.
  - The fine-tune (IBLOG_TUTOR) was NOT trained on diacritized text: measured
    directly against data/v11_merged/train.jsonl, only ~4.9% of rows contain
    ANY diacritic mark at all, and those are sparse single-character noise
    (a stray shadda), not systematic tashkeel. Its learned pronunciation is
    conditioned overwhelmingly on bare consonant-skeleton input.
  - No Darija-specific diacritizer exists. Every automatic Arabic
    diacritization tool -- including the one this script uses -- targets
    Modern Standard Arabic. Applying MSA vowelization rules to genuine
    Darija vocabulary can produce grammatically-plausible-LOOKING but
    linguistically wrong case endings: a quick manual check while building
    this script showed exactly that -- "خصك" (Darija "you must", no MSA
    case system applies) came back "خَصُّكَ" with an invented gemination/
    case ending, and "مزيان" (Darija "good/well") picked up a genitive
    ending it would never take colloquially. Whether XTTS's LEARNED
    PRONUNCIATION improves or degrades when fed this is an empirical,
    ears-only question -- not something readable off the diacritics
    themselves. Hence this script produces audio to listen to, not a
    text-quality report.
  - Whether medmac01/darija_xtt_2.0's own fine-tune audio corpus was itself
    diacritized is undocumented and unverifiable in-repo (no README/model
    card -- see ADR 0006). This script is the only way to find out
    empirically on THIS checkpoint.

Diacritizer used: `arabic-diacritizer` (PyPI), MIT-licensed (confirmed via
its GitHub repo's LICENSE file 2026-09-06 -- PyPI's own license metadata
field is empty, which is a packaging omission, not a licensing gap), a
BiLSTM+attention model with a bundled pretrained checkpoint (no separate
download). Picked because it installed cleanly with no extra setup into
the existing .tts_eval_venv (torch already present) and exposes a plain
`Diacritizer.from_pretrained().diacritize(text)` call -- no claim here that
it is the best available diacritizer, only that it is a working one to
start the comparison with. Swap it if this eval's outcome makes it worth
trying a second one.

Runs in a short-lived SUBPROCESS of this same venv's interpreter, not
imported directly into this script's own process: confirmed 2026-09-06 that
loading arabic-diacritizer's own torch model in-process alongside coqui-tts's
torch/torchcodec load segfaults reliably, right after XTTS's
model.load_checkpoint(). Same class of native-library conflict this
codebase always isolates into a separate process for elsewhere (see
scripts/{tts,speech,ocr}_worker_resident.py) -- diacritizing all sentences
happens first, entirely before XTTS is ever imported in the main process.

Requires the SAME dedicated venv as eval_darija_xtts.py (isolated from
.gguf_venv/.speech_venv -- see that script's own docstring):
    .tts_eval_venv/Scripts/python.exe -m pip install arabic-diacritizer

Reuses the XTTS checkpoint already cached by eval_darija_xtts.py at
data/tts_eval_cache/darija_xtts/ -- run that script first if this errors
looking for it.

Run (from repo root):
    .tts_eval_venv/Scripts/python.exe scripts/eval_darija_tashkeel.py

Writes paired .wav files next to this script:
    tashkeel_eval_<n>_bare.wav          -- current production behavior
    tashkeel_eval_<n>_diacritized.wav   -- same sentence, tashkeel added
Listen to each pair back-to-back. If diacritized is clearly better across
most sentences, that's the signal to build a real production step (a new
pluggable stage ahead of synthesize(), gated by a settings flag). If it's a
wash or worse, this ends here -- a real negative result, not a shipped
regression.
"""
import json
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# arabic-diacritizer's own torch model load and coqui-tts's torch/torchcodec
# load segfault when both happen in the SAME process (confirmed 2026-09-06:
# reproducible, immediate SIGSEGV right after XTTS's model.load_checkpoint(),
# with the diacritizer loaded first) -- the exact class of native-library
# conflict this codebase always isolates into a separate process for
# (scripts/{tts,speech,ocr}_worker_resident.py all exist for this reason,
# each in its own dedicated venv). Diacritizing is cheap and CPU-fine, so it
# runs in a short-lived subprocess of THIS SAME venv's interpreter, entirely
# before XTTS is ever imported in the main process.
_DIACRITIZE_SUBPROCESS_SCRIPT = """
import json, sys
sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
from diacritize import Diacritizer
sentences = json.load(sys.stdin)
d = Diacritizer.from_pretrained()
json.dump([d.diacritize(s) for s in sentences], sys.stdout, ensure_ascii=False)
"""


def _diacritize_all(sentences: list[str]) -> list[str]:
    proc = subprocess.run(
        [sys.executable, "-c", _DIACRITIZE_SUBPROCESS_SCRIPT],
        input=json.dumps(sentences, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    return json.loads(proc.stdout)


def _register_ffmpeg_dll_dir() -> None:
    """Same Windows/torchcodec DLL-search-path issue as eval_darija_xtts.py
    -- see that script's docstring for the full explanation."""
    ffmpeg_bin = os.environ.get("FFMPEG_SHARED_BIN")
    if ffmpeg_bin and hasattr(os, "add_dll_directory") and os.path.isdir(ffmpeg_bin):
        os.add_dll_directory(ffmpeg_bin)


_register_ffmpeg_dll_dir()

OUT_DIR = Path(__file__).resolve().parent
CACHE_DIR = REPO_ROOT / "data" / "tts_eval_cache" / "darija_xtts"
VOICE_EVAL_DIR = REPO_ROOT / "tests" / "data" / "voice_eval"

# Same three sentences eval_darija_xtts.py used (ADR 0006's own darija
# bucket) plus a few real doda_ary_*.txt fixtures for variety beyond the
# hand-picked set -- both sources already exist in this repo, no new data.
TEST_SENTENCES = [
    "خصك تلبس الكاسك ديال الحماية.",
    "على حساب المادة 8، خاصك تتأكد من العزلة قبل ما تبدا.",
    "قبل ما تبدا الخدمة، تأكد أن التجهيزات معزولة مزيان وأن الترخيص ديال الخدمة موقع.",
]


def _load_doda_sentences(n: int = 3) -> list[str]:
    sentences = []
    for i in range(n):
        path = VOICE_EVAL_DIR / f"doda_ary_{i:02d}.txt"
        if path.exists():
            sentences.append(path.read_text(encoding="utf-8").strip())
    return sentences


def _write_wav_from_float(path: Path, wav_array, sample_rate: int) -> None:
    import numpy as np

    pcm16 = (np.clip(wav_array, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


def main() -> None:
    import torch
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    config_path = CACHE_DIR / "config.json"
    vocab_path = CACHE_DIR / "vocab.json"
    model_path = CACHE_DIR / "model.pth"
    speaker_path = CACHE_DIR / "speaker_ref.wav"
    for p in (config_path, vocab_path, model_path, speaker_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing -- run scripts/eval_darija_xtts.py first to "
                f"download/cache the XTTS checkpoint."
            )

    sentences = TEST_SENTENCES + _load_doda_sentences()

    print("Diacritizing all sentences (subprocess -- see module docstring for why)...")
    diacritized_sentences = _diacritize_all(sentences)

    config = XttsConfig()
    config.load_json(str(config_path))
    print("Loading medmac01/darija_xtt_2.0 (XTTS-v2 fine-tune)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config, checkpoint_path=str(model_path), use_deepspeed=False,
        vocab_path=str(vocab_path), eval=True,
    )
    model.to(device)

    print("Computing speaker latents from the bundled reference clip...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[str(speaker_path)])

    results = []
    for i, (sentence, diacritized) in enumerate(zip(sentences, diacritized_sentences)):
        print(f"\n#{i}")
        print(f"  bare:        {sentence}")
        print(f"  diacritized: {diacritized}")

        row = {"index": i, "bare": sentence, "diacritized": diacritized}
        for variant, text in (("bare", sentence), ("diacritized", diacritized)):
            t0 = time.time()
            # "ar" -- same XTTS-v2 language code eval_darija_xtts.py uses;
            # this checkpoint has no dialect selector of its own.
            out = model.inference(text, "ar", gpt_cond_latent, speaker_embedding, temperature=0.65)
            elapsed = time.time() - t0
            wav = out["wav"]
            duration = len(wav) / 24000
            out_path = OUT_DIR / f"tashkeel_eval_{i}_{variant}.wav"
            _write_wav_from_float(out_path, wav, 24000)
            rtf = round(elapsed / duration, 3) if duration > 0 else None
            row[f"{variant}_seconds"] = round(elapsed, 2)
            row[f"{variant}_rtf"] = rtf
            row[f"{variant}_wav"] = out_path.name
            print(f"  {variant:<11} {elapsed:>5.2f}s  rtf={rtf}  -> {out_path.name}")
        results.append(row)

    import json
    (OUT_DIR / "eval_darija_tashkeel_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote {OUT_DIR / 'eval_darija_tashkeel_results.json'} and paired .wav files next to this script.")
    print("Listen to each tashkeel_eval_<n>_bare.wav / _diacritized.wav pair back-to-back.")
    print("Do NOT wire tashkeel into production based on this script's text output alone --")
    print("the diacritizer is MSA-trained and can add case endings that don't apply to real")
    print("Darija words; only the audio comparison is a valid signal here.")


if __name__ == "__main__":
    main()
