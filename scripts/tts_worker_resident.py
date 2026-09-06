"""
Resident XTTS TTS IPC worker.

Runs `medmac01/darija_xtt_2.0` -- an XTTS-v2 fine-tune on Moroccan Darija
(the model behind https://huggingface.co/spaces/medmac01/Darija-Arabic-TTS),
selected on live listening as the best-sounding Darija voice available.

LICENSING -- READ BEFORE SHIPPING. The checkpoint declares no license of
its own (confirmed via the HF API 2026-09-06: no `license` field, no
README), so absent an override it inherits its base's: Coqui XTTS-v2,
CPML, which forbids commercial use. app/services/tts.py's own module
docstring records XTTS-v2 being rejected once already on exactly this
basis. This worker exists because it was explicitly requested for
evaluation on real infrastructure; keeping `settings.tts_engine` on
"piper" is what keeps the deployed product inside its licensing. See
docs/architecture/rectified/adr/0006-darija-tts-survey.md.

Runs inside a DEDICATED venv (settings.tts_xtts_venv_python), never inside
.gguf_venv: coqui-tts pins transformers<5 (it breaks on 5.x with
`ImportError: isin_mps_friendly`) and drags in torchcodec, neither of which
the app's own venv should be forced to match -- the same isolation
reasoning as app/services/ocr.py's PaddleOcrEngine and
scripts/speech_worker_resident.py.

app/services/tts.py's _ResidentTtsWorker spawns this ONCE per app process
and reuses it for every sentence in every voice session, so the ~5.6GB
checkpoint load is paid once, not once per sentence.

Protocol -- one JSON object per line, both directions, flushed immediately:
  in:  {"cmd": "ping"}
  out: {"ok": true, "cmd": "pong"}

  in:  {"cmd": "synthesize", "id": "<echoed back>", "text": "...",
        "language": "fr" | "darija", "out": "<path to write raw PCM to>"}
  out: {"ok": true,  "id": "...", "path": "...", "sample_rate": 24000}
  out: {"ok": false, "id": "...", "error": "..."}

Audio comes back as a FILE of raw 16-bit mono PCM, not inline in the JSON:
one sentence is ~200-400KB, and base64 through a line-oriented pipe would
be both slower and a second place for encoding bugs to hide. Mirrors how
speech_worker_resident.py receives its audio (a temp path), inverted.

stdout carries ONLY JSON response lines -- all diagnostics go to stderr,
same convention as ocr_worker_resident.py / speech_worker_resident.py.
"""
import json
import os
import sys
import traceback


def _register_ffmpeg_dll_dir() -> None:
    """Windows: coqui-tts imports torchcodec, which needs FFmpeg's SHARED
    libraries (avcodec-*.dll etc.). The usual WinGet/gyan.dev "full_build"
    is static and ships none, and putting a shared build on PATH does not
    help -- Python 3.8+ on Windows resolves extension-module DLLs with
    LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, which excludes PATH. Only
    os.add_dll_directory() works (confirmed 2026-09-06).

    No-op on Linux (config/Dockerfile.gpu installs FFmpeg via apt, where
    the normal loader path finds it).
    """
    ffmpeg_bin = os.environ.get("FFMPEG_SHARED_BIN")
    if ffmpeg_bin and hasattr(os, "add_dll_directory") and os.path.isdir(ffmpeg_bin):
        try:
            os.add_dll_directory(ffmpeg_bin)
        except OSError:
            pass


_register_ffmpeg_dll_dir()

_STATE: dict = {}

# XTTS-v2's own language codes. It has no Moroccan-Darija code -- "ar" is
# the code the Space's own app.py passes, and this checkpoint's Darija-ness
# comes from its fine-tune data, not from a dialect selector.
_LANG_MAP = {"darija": "ar", "ary": "ar", "ar": "ar", "fr": "fr", "en": "en"}

# This checkpoint's fixed output rate (the Space hardcodes 24000 too).
SAMPLE_RATE = 24000


def _load():
    if "model" in _STATE:
        return _STATE["model"], _STATE["latents"]

    import torch
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    model_dir = os.environ.get("TTS_XTTS_MODEL_DIR", "./data/tts_eval_cache/darija_xtts")
    speaker_ref = os.environ.get("TTS_XTTS_SPEAKER_REF", os.path.join(model_dir, "speaker_ref.wav"))

    config_path = os.path.join(model_dir, "config.json")
    vocab_path = os.path.join(model_dir, "vocab.json")
    model_path = os.path.join(model_dir, "model.pth")
    for required in (config_path, vocab_path, model_path, speaker_ref):
        if not os.path.exists(required):
            raise FileNotFoundError(
                f"XTTS asset missing: {required}. Download medmac01/darija_xtt_2.0's "
                f"config.json, vocab.json, model_2.1.pth (as model.pth) and speaker_ref.wav "
                f"into {model_dir} (TTS_XTTS_MODEL_DIR)."
            )

    config = XttsConfig()
    config.load_json(config_path)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config, checkpoint_path=model_path, use_deepspeed=False,
        vocab_path=vocab_path, eval=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"[tts_worker_resident] loaded darija_xtt_2.0 on {device}", file=sys.stderr, flush=True)

    # Speaker latents are computed ONCE from the reference clip, not per
    # sentence -- get_conditioning_latents() is the expensive part of an
    # XTTS call and the reference never changes within a deployment.
    latents = model.get_conditioning_latents(audio_path=[speaker_ref])
    print("[tts_worker_resident] speaker latents ready", file=sys.stderr, flush=True)

    _STATE["model"] = model
    _STATE["latents"] = latents
    return model, latents


def _handle_synthesize(req: dict) -> dict:
    rid = req.get("id")
    text = (req.get("text") or "").strip()
    out_path = req.get("out")
    language = _LANG_MAP.get(req.get("language"), "ar")
    if not text:
        return {"ok": False, "id": rid, "error": "empty text"}
    if not out_path:
        return {"ok": False, "id": rid, "error": "missing 'out' path"}

    import numpy as np

    model, (gpt_cond_latent, speaker_embedding) = _load()
    out = model.inference(text, language, gpt_cond_latent, speaker_embedding, temperature=0.65)
    wav = np.asarray(out["wav"], dtype=np.float32)
    pcm16 = (np.clip(wav, -1.0, 1.0) * 32767).astype("<i2")
    with open(out_path, "wb") as f:
        f.write(pcm16.tobytes())
    return {"ok": True, "id": rid, "path": out_path, "sample_rate": SAMPLE_RATE}


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"ok": False, "error": "bad JSON"}), flush=True)
            continue

        cmd = req.get("cmd")
        try:
            if cmd == "ping":
                resp = {"ok": True, "cmd": "pong"}
            elif cmd == "synthesize":
                resp = _handle_synthesize(req)
            else:
                resp = {"ok": False, "id": req.get("id"), "error": f"unknown cmd {cmd!r}"}
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            resp = {"ok": False, "id": req.get("id"), "error": f"{type(e).__name__}: {e}"}
        print(json.dumps(resp, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
