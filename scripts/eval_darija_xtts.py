"""
Darija TTS candidate eval: medmac01/darija_xtt_2.0 (XTTS-v2 fine-tune),
the model behind https://huggingface.co/spaces/medmac01/Darija-Arabic-TTS.

EVAL-ONLY. NOT a TtsEngine, NOT wired into app/services/tts.py's _ENGINES
registry, NOT settable via settings.tts_engine. This is deliberate, not an
oversight:

  medmac01/darija_xtt_2.0 has no license file of its own (confirmed via the
  HF API 2026-09-06: no `license` field, no README). Absent an override, a
  fine-tune inherits its base model's license, and the base here is Coqui
  XTTS-v2 -- CPML, explicitly NON-COMMERCIAL (see app/services/tts.py's own
  module docstring: this is the exact license wall that already ruled out
  XTTS-v2 for this project once). Shipping this as a selectable production
  engine in a B2B tutor risks a tenant deployment flipping it on and
  violating that license. ADR 0006 (docs/architecture/rectified/adr/
  0006-darija-tts-survey.md) lists this candidate for the same reason.

This script exists ONLY so the model's actual output quality can be
compared against Piper's ar_JO-kareem-medium on the SAME sentences this
project already uses for that judgment (scripts/eval_tts.py's
TEST_SENTENCES) -- ADR 0006's own listening-test methodology, not a
synthesis-speed number. Listen to the .wav files; do not ship this engine.

Requires a DEDICATED venv (isolated from .gguf_venv/.speech_venv for the
same reason those exist -- see config/requirements-speech.txt's comment):
    python -m venv .tts_eval_venv
    .tts_eval_venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
    .tts_eval_venv/Scripts/python.exe -m pip install TTS

Downloads ~5.6GB of checkpoint + a bundled 4-5s speaker reference wav (the
Space's own default) into data/tts_eval_cache/darija_xtts/ on first run.

Run (from repo root):
    .tts_eval_venv/Scripts/python.exe scripts/eval_darija_xtts.py
"""
import os
import sys
import time
import wave
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _register_ffmpeg_dll_dir() -> None:
    """Windows: coqui-tts imports torchcodec, which needs FFmpeg's SHARED
    libraries (avcodec-*.dll etc.), not just an ffmpeg.exe on PATH -- the
    usual WinGet/gyan.dev "full_build" is static and ships no DLLs.

    Putting the DLL directory on PATH does NOT work: Python 3.8+ on Windows
    resolves extension-module dependencies with
    LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, which deliberately excludes PATH --
    os.add_dll_directory() is the only thing that registers it (confirmed
    2026-09-06, after a PATH-only attempt failed identically).

    Point FFMPEG_SHARED_BIN at the `bin` directory of a shared build (e.g.
    BtbN/FFmpeg-Builds' ffmpeg-nX-latest-win64-gpl-shared-X.zip). torchcodec
    ships loaders for FFmpeg 4 through 9.
    """
    ffmpeg_bin = os.environ.get("FFMPEG_SHARED_BIN")
    if ffmpeg_bin and hasattr(os, "add_dll_directory") and os.path.isdir(ffmpeg_bin):
        os.add_dll_directory(ffmpeg_bin)


_register_ffmpeg_dll_dir()

OUT_DIR = Path(__file__).resolve().parent
CACHE_DIR = REPO_ROOT / "data" / "tts_eval_cache" / "darija_xtts"
BASE_URL = "https://huggingface.co/medmac01/darija_xtt_2.0/resolve/main"

# Same sentences as scripts/eval_tts.py's "darija" bucket -- the direct,
# apples-to-apples comparison against the currently-wired Piper voice.
TEST_SENTENCES = [
    "خصك تلبس الكاسك ديال الحماية.",
    "على حساب المادة 8، خاصك تتأكد من العزلة قبل ما تبدا.",
    "قبل ما تبدا الخدمة، تأكد أن التجهيزات معزولة مزيان وأن الترخيص ديال الخدمة موقع.",
]


def _ensure_checkpoint() -> tuple[Path, Path, Path, Path]:
    import urllib.request

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    files = {
        "config.json": "config.json",
        "vocab.json": "vocab.json",
        "model.pth": "model_2.1.pth",
        "speaker_ref.wav": "speaker_ref.wav",
    }
    paths = {}
    for local_name, remote_name in files.items():
        path = CACHE_DIR / local_name
        paths[local_name] = path
        if path.exists():
            continue
        print(f"Downloading {remote_name} -> {path} ...")
        urllib.request.urlretrieve(f"{BASE_URL}/{remote_name}", path)
    return paths["config.json"], paths["vocab.json"], paths["model.pth"], paths["speaker_ref.wav"]


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

    config_path, vocab_path, model_path, speaker_path = _ensure_checkpoint()

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
    for i, sentence in enumerate(TEST_SENTENCES):
        t0 = time.time()
        # "ar" is XTTS-v2's own language code -- the Space's own app.py
        # passes this unconditionally; the model's Darija-ness comes
        # entirely from the fine-tune data, not a dialect-specific code
        # (XTTS-v2 has none).
        out = model.inference(sentence, "ar", gpt_cond_latent, speaker_embedding, temperature=0.65)
        elapsed = time.time() - t0
        wav = out["wav"]
        duration = len(wav) / 24000  # this checkpoint's fixed output rate (see app.py)
        out_path = OUT_DIR / f"eval_darija_xtts_{i}.wav"
        _write_wav_from_float(out_path, wav, 24000)
        rtf = round(elapsed / duration, 3) if duration > 0 else None
        results.append({"sentence": sentence, "seconds": round(elapsed, 2), "rtf": rtf, "wav": out_path.name})
        print(f"#{i} {elapsed:>5.2f}s  rtf={rtf}  -> {out_path.name}")

    import json
    (OUT_DIR / "eval_darija_xtts_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote {OUT_DIR / 'eval_darija_xtts_results.json'} and per-sentence .wav files next to this script.")
    print("Listen and compare against eval_tts_darija_*.wav (Piper) before deciding anything.")
    print("REMINDER: this checkpoint is CPML-derived (non-commercial) -- eval only, cannot ship as-is.")


if __name__ == "__main__":
    main()
