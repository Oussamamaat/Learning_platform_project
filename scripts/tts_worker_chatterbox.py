"""
Resident Chatterbox TTS IPC worker.

Runs the Chatterbox Multilingual v3 base model (ResembleAI, MIT) + the
`cs-run1` LoRA adapter -- this project's own Darija fine-tune, trained to
replace `medmac01/darija_xtt_2.0` (app/services/tts.py's XttsDarijaEngine,
scripts/tts_worker_resident.py), whose checkpoint declares no license and
therefore inherits XTTS-v2's non-commercial CPML. See
tts_test/docs/CURRENT_MODEL.md (a sibling project, not part of this repo)
for the full training/verification history and this project's memory
`darija-tts-current-working-version.md`.

Serves BOTH languages this tenant needs (French and Darija), same as
XttsDarijaEngine, so one worker covers the whole voice session -- no
Piper/Chatterbox hybrid, no per-language sample rate. Unlike XTTS this is
NOT voice cloning: Chatterbox's own built-in conditioning (`conds.pt`) is
used, no speaker_ref.wav.

Runs inside a DEDICATED venv (settings.tts_chatterbox_venv_python), never
inside .gguf_venv: this pins transformers==4.46.3 (older than the app's
own, and the exact version `fast_t3.py`'s CUDA-graph patch was written
against -- it reaches into transformers.generation internals and legacy
KV-cache tuples) -- same isolation reasoning as app/services/ocr.py's
PaddleOcrEngine, scripts/speech_worker_resident.py, and this module's XTTS
sibling.

Protocol -- one JSON object per line, both directions, flushed immediately
(IDENTICAL to scripts/tts_worker_resident.py's, so app.services.tts's
_ResidentTtsWorker is reused unchanged for this engine too):
  in:  {"cmd": "ping"}
  out: {"ok": true, "cmd": "pong"}

  in:  {"cmd": "synthesize", "id": "<echoed back>", "text": "...",
        "language": "fr" | "darija", "out": "<path to write raw PCM to>"}
  out: {"ok": true,  "id": "...", "path": "...", "sample_rate": 24000}
  out: {"ok": false, "id": "...", "error": "..."}

Audio comes back as a FILE of raw 16-bit mono PCM, not inline in the
JSON -- same reasoning as tts_worker_resident.py (one sentence is a few
hundred KB; base64 through a line-oriented pipe is slower and a second
place for encoding bugs to hide).

stdout carries ONLY JSON response lines -- all diagnostics go to stderr,
same convention as every other resident worker in this repo.
"""
import json
import os
import sys
import threading
import traceback
from typing import Optional

# Force UTF-8 on stdin/stdout regardless of the parent's environment or the
# host's console codepage. Confirmed necessary on Windows 2026-09-27: without
# this, sys.stdin's default encoding is the console's OEM/ANSI codepage, not
# UTF-8, and JSON lines carrying Arabic-script text get decoded with lone
# surrogates -- valid Python `str` objects, but ones the Rust `tokenizers`
# binding rejects with the misleading "TextInputSequence must be str"
# (confirmed: identical text, same tokenizer, works standalone as a literal
# but fails once round-tripped through a misdecoded stdin). Mirrors
# scripts/tts_worker_resident.py's stdout/stderr reconfigure -- extended here
# to stdin too, since this worker's OWN input (unlike that one's) routinely
# carries non-ASCII text.
for _s in (sys.stdin, sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Reroute the OS-level stdout file descriptor (fd 1) to stderr (fd 2), and keep
# a private duplicate of the ORIGINAL fd 1 for this worker's own JSON replies
# (_response_out). Confirmed necessary 2026-09-27: perth (the watermarker
# chatterbox_.mtl_tts depends on) does a bare `print("loaded PerthNet
# (Implicit) at step ...")` during model load -- a raw write to fd 1, not
# something a Python-level `sys.stdout = ...` swap or a logging config can
# intercept, since it bypasses neither. That line landed on the SAME pipe
# app.services.tts._ResidentTtsWorker's _drain_stdout reads as this worker's
# JSON responses, so its strict line-by-line json.loads() got that print
# instead of a reply and raised a confusing JSONDecodeError -- reproduced only
# through the real _ResidentTtsWorker, not through manually piping stdin,
# which is why scripts/tts_worker_resident.py's sibling documented "stdout
# carries ONLY JSON" as a convention without needing to enforce it: none of
# the libraries its OWN engine loads happen to violate it.
_response_fd = os.dup(1)
_response_out = os.fdopen(_response_fd, "w", encoding="utf-8", errors="replace", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr

# scripts/tts_chatterbox/ -- the vendored inference-only slice of
# chatterbox-finetuning (see its README.md for what is and is not
# vendored, and why). Inserted first so `import src.chatterbox_...` and
# `import fast_t3` / `import fast_s3gen` / `import text_normalize` resolve
# to the vendored copies, not anything importable from elsewhere on the
# venv's path.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts_chatterbox"))

_STATE: dict = {}
_LOCK = threading.Lock()

# This checkpoint's fixed output rate -- ChatterboxMultilingualTTS.sr, matches
# tts_test/remote/chatterbox-finetuning/tts_server.py's SR and this repo's
# XttsDarijaEngine.sample_rate (both 24000), so app/routers/voice.py's
# one-rate-per-session `audio.start` contract holds across engines too.
SAMPLE_RATE = 24000

# text_normalize.prepare_for_tts's own language_id values ("fr"/"ar") already
# decide the per-sentence language from the text's script -- this map is only
# the fallback for the degenerate case of empty/non-lettered text, where
# prepare_for_tts has nothing to key off and this repo's own "darija"/"fr"
# request vocabulary (app.services.tts.TtsEngine.synthesize's `language`
# param) needs translating to Chatterbox's "ar"/"fr" language_id.
_LANG_MAP = {"darija": "ar", "ary": "ar", "ar": "ar", "fr": "fr", "en": "en"}

# Generation settings every listening round (fr2, dar1, csr1-final, spd1) was
# judged against -- tts_test/docs/CURRENT_MODEL.md "Generate with
# engine.generate()". Changing these re-opens a question the user already
# closed by ear; if a change is needed, validate it in tts_test first.
_GEN_KWARGS = dict(
    exaggeration=0.5, cfg_weight=0.5, temperature=0.8, repetition_penalty=2.0,
)


def _load():
    if "engine" in _STATE:
        return _STATE["engine"]

    import torch
    from peft import PeftModel
    from src.chatterbox_.mtl_tts import ChatterboxMultilingualTTS
    from src.chatterbox_.models.t3.t3 import T3
    from src.chatterbox_.models.t3.modules.t3_config import T3Config

    model_dir = os.environ.get("TTS_CHATTERBOX_MODEL_DIR", "./data/tts_models/chatterbox/base")
    adapter_dir = os.environ.get("TTS_CHATTERBOX_ADAPTER_DIR", "./data/tts_models/chatterbox/cs-run1")
    use_fast = os.environ.get("TTS_CHATTERBOX_FAST", "1").strip().lower() not in ("0", "false", "no")

    if not os.path.isfile(os.path.join(adapter_dir, "adapter_config.json")):
        raise FileNotFoundError(
            f"No PEFT adapter at {adapter_dir} (TTS_CHATTERBOX_ADAPTER_DIR) -- expected "
            f"adapter_config.json + adapter_model.safetensors from HF "
            f"Oussamamaat/darija-chatterbox-checkpoints, cs-run1/final_adapter/."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[tts_worker_chatterbox] loading base model from {model_dir} on {device} ...",
          file=sys.stderr, flush=True)

    # Mirrors tts_test/remote/chatterbox-finetuning/finetune_multilingual.py's
    # build_model() (inference-only: no training-config dataclass, no
    # resize/vocab-transfer step -- this adapter's vocab already matches the
    # base 1:1, so what build_model() calls "resize_and_load_t3_weights" is a
    # plain state_dict load here).
    engine = ChatterboxMultilingualTTS.from_local(model_dir, device="cpu")
    if getattr(engine.t3.hp, "text_tokens_dict_size", None) != 2454:
        print(f"[tts_worker_chatterbox] WARNING: base vocab size "
              f"{getattr(engine.t3.hp, 'text_tokens_dict_size', '?')} != the 2454 cs-run1 was "
              f"trained against -- the adapter may not apply cleanly.", file=sys.stderr, flush=True)
    if hasattr(engine.t3.hp, "use_cache"):
        engine.t3.hp.use_cache = False

    # Freeze BEFORE merging the adapter in -- mirrors finetune_multilingual.py's
    # build_model() (ve/s3gen frozen immediately after from_local(), t3 frozen
    # before LoRA is attached). Not just tidiness: fast_s3gen.py's graph-capture
    # path (_capture(), called from install_fast_s3gen's prewarm) runs s3gen's
    # forward with NO inference_mode/no_grad context (only its replay path,
    # __call__, is decorated). With requires_grad still at its from_local()
    # default of True, that capture builds and retains a full autograd graph
    # per length bucket -- confirmed 2026-09-27 on this repo's own RTX 4060
    # laptop: OOM at ~12GB allocated on an 8GB card, on a model that peaks at
    # ~7.4GB when frozen (tts_test/docs/CURRENT_MODEL.md).
    for m in (engine.ve, engine.s3gen):
        for p in m.parameters():
            p.requires_grad = False
    for p in engine.t3.parameters():
        p.requires_grad = False

    engine.t3 = PeftModel.from_pretrained(engine.t3, adapter_dir)
    nz = sum(1 for n, t in engine.t3.named_parameters()
             if "lora_B" in n and t.detach().abs().sum().item() > 0)
    if nz == 0:
        raise RuntimeError(f"adapter at {adapter_dir} is all-zero: it encodes no training")
    print(f"[tts_worker_chatterbox] adapter check: {nz} non-zero lora_B tensors", file=sys.stderr, flush=True)
    engine.t3 = engine.t3.merge_and_unload()
    for p in engine.t3.parameters():
        p.requires_grad = False

    for m in (engine.t3, engine.ve, engine.s3gen):
        m.to(device).eval()
    if getattr(engine, "conds", None) is not None:
        engine.conds = engine.conds.to(device)
    engine.device = device

    if device == "cuda":
        # The alignment/hallucination guard needs attention maps, which SDPA
        # can't return -- same requirement fast_t3.install_fast_t3 documents.
        engine.t3.tfmr.config._attn_implementation = "eager"

        if use_fast:
            from fast_t3 import install_fast_t3
            install_fast_t3(engine)
            print("[tts_worker_chatterbox] fast_t3 installed", file=sys.stderr, flush=True)
            from fast_s3gen import install_fast_s3gen
            g = install_fast_s3gen(engine)
            print(f"[tts_worker_chatterbox] fast_s3gen installed: {len(g.graphs)} CFM buckets",
                  file=sys.stderr, flush=True)
    elif use_fast:
        print("[tts_worker_chatterbox] no CUDA device -- skipping fast_t3/fast_s3gen "
              "(CPU-only inference will be slow)", file=sys.stderr, flush=True)

    _STATE["engine"] = engine
    print("[tts_worker_chatterbox] engine ready", file=sys.stderr, flush=True)
    return engine


def _synthesize_text(engine, text: str, language: str):
    """Split `text` into sentence/language parts with prepare_for_tts (the
    same normalization every listening round was judged on -- French-run
    tagging, number/unit expansion, the خ fix), generate each part, and
    concatenate. `language` is only the fallback for parts prepare_for_tts
    can't classify from the text itself (see _LANG_MAP's docstring)."""
    import numpy as np
    import torch
    from text_normalize import prepare_for_tts

    fallback_lang = _LANG_MAP.get(language, "ar")
    parts = prepare_for_tts(text) or [(text, fallback_lang)]

    with _LOCK, torch.inference_mode():
        chunks = []
        for part_text, language_id in parts:
            wav = engine.generate(part_text, language_id=language_id, audio_prompt_path=None,
                                  **_GEN_KWARGS)
            chunks.append(wav.squeeze(0).detach().cpu().numpy().astype(np.float32))
        full = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    return full


def _handle_synthesize(req: dict) -> dict:
    rid = req.get("id")
    text = (req.get("text") or "").strip()
    out_path = req.get("out")
    language = req.get("language") or "darija"
    if not text:
        return {"ok": False, "id": rid, "error": "empty text"}
    if not out_path:
        return {"ok": False, "id": rid, "error": "missing 'out' path"}

    import numpy as np

    engine = _load()
    wav = _synthesize_text(engine, text, language)

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
            print(json.dumps({"ok": False, "error": "bad JSON"}), file=_response_out, flush=True)
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
        print(json.dumps(resp, ensure_ascii=False), file=_response_out, flush=True)


if __name__ == "__main__":
    main()
