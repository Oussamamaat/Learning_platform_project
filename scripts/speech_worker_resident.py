"""
Resident STT IPC worker.

UNVERIFIED SCAFFOLDING -- see app/services/stt.py's module docstring. This
mirrors scripts/ocr_worker_resident.py's protocol and process shape
exactly (same problem: a heavy GPU model, best kept in a separate venv and
talked to as a persistent subprocess), but no bake-off has picked a
winning STT engine/checkpoint yet, and nothing in this file has been run
against a loaded model. Both engine branches below are best-effort
sketches of the two Phase 0 candidates named in
docs/architecture/voice-assistant.md, not proven integrations.

Runs inside .speech_venv, never inside .gguf_venv (the app's own venv) --
faster-whisper (ctranslate2) and SeamlessM4T-v2 (transformers/torch) are
each liable to pin CUDA/torch versions that conflict with the app's own
pin, same reasoning as app/services/ocr.py's PaddleOcrEngine docstring.

app/services/stt.py's _ResidentSttWorker spawns this ONCE per app process
and reuses it for every utterance in every voice session, so the model
load cost is paid once, not once per turn.

Protocol -- one JSON object per line, both directions, flushed immediately:
  in:  {"cmd": "ping"}
  out: {"ok": true, "cmd": "pong"}

  in:  {"cmd": "transcribe", "id": "<echoed back>", "audio": "<path to a
        16kHz mono 16-bit PCM WAV>", "engine": "whisper" | "seamless",
        "language_hint": "fr" | "ary" | null}
  out: {"ok": true,  "id": "...", "text": "...", "language": "fr"}
  out: {"ok": false, "id": "...", "error": "..."}

stdout carries ONLY JSON response lines -- all diagnostics go to stderr,
same convention as ocr_worker_resident.py.
"""
import json
import sys
import traceback

_MODELS: dict = {}


def _get_whisper_model():
    if "whisper" in _MODELS:
        return _MODELS["whisper"]
    from faster_whisper import WhisperModel
    from app.config import get_settings

    model = WhisperModel(get_settings().stt_model, device="cuda", compute_type="int8_float16")
    _MODELS["whisper"] = model
    print(f"[speech_worker_resident] loaded faster-whisper model", file=sys.stderr, flush=True)
    return model


def _transcribe_whisper(audio_path: str, language_hint) -> dict:
    model = _get_whisper_model()
    # faster-whisper's own language codes: "fr", "ar" -- language_hint here
    # is expected to already be one of those (app.services.stt maps this
    # module's Protocol's "fr"/"darija" callers to whatever the winning
    # engine's own codes are; that mapping is NOT written yet -- see this
    # file's module docstring).
    segments, info = model.transcribe(audio_path, language=language_hint, beam_size=5)
    text = "".join(seg.text for seg in segments)
    return {"text": text.strip(), "language": info.language}


def _get_seamless_model():
    if "seamless" in _MODELS:
        return _MODELS["seamless"]
    import torch
    from transformers import AutoProcessor, SeamlessM4Tv2Model

    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large").to("cuda")
    _MODELS["seamless"] = (processor, model)
    print(f"[speech_worker_resident] loaded SeamlessM4T-v2-large", file=sys.stderr, flush=True)
    return processor, model


def _transcribe_seamless(audio_path: str, language_hint) -> dict:
    import torchaudio

    processor, model = _get_seamless_model()
    waveform, sample_rate = torchaudio.load(audio_path)
    # SeamlessM4T-v2's source-language code for Moroccan Arabic is "ary"
    # (distinct from generic MSA "arb") -- this is the entire reason it is
    # a bake-off candidate; see app/services/stt.py's module docstring.
    src_lang = language_hint or "ary"
    inputs = processor(audios=waveform, sampling_rate=sample_rate, return_tensors="pt").to("cuda")
    output_tokens = model.generate(**inputs, tgt_lang=src_lang, generate_speech=False)
    text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
    return {"text": text.strip(), "language": src_lang}


def _handle_transcribe(req: dict) -> dict:
    rid = req.get("id")
    engine = req.get("engine", "whisper")
    audio_path = req.get("audio")
    language_hint = req.get("language_hint")
    try:
        if engine == "whisper":
            result = _transcribe_whisper(audio_path, language_hint)
        elif engine == "seamless":
            result = _transcribe_seamless(audio_path, language_hint)
        else:
            raise ValueError(f"unknown engine {engine!r} (expected 'whisper' or 'seamless')")
        return {"ok": True, "id": rid, **result}
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return {"ok": False, "id": rid, "error": f"{type(e).__name__}: {e}"}


def main() -> None:
    print(f"[speech_worker_resident] ready, pid={__import__('os').getpid()}", file=sys.stderr, flush=True)
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            print(json.dumps({"ok": False, "error": f"bad json: {e}"}), flush=True)
            continue

        cmd = req.get("cmd")
        if cmd == "ping":
            print(json.dumps({"ok": True, "cmd": "pong"}), flush=True)
        elif cmd == "transcribe":
            print(json.dumps(_handle_transcribe(req)), flush=True)
        else:
            print(json.dumps({"ok": False, "error": f"unknown cmd {cmd!r}"}), flush=True)


if __name__ == "__main__":
    main()
