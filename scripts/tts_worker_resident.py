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
from typing import Optional


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

# Same Arabic-Unicode-block convention used everywhere else in this repo for
# script detection (e.g. app.services.generate_training_data.has_arabic_script,
# app.services.routing.resolve_domain, app.services.llm.detect_query_language)
# -- U+0621 ('ء') through U+06FF ('ۿ') covers Arabic letters, digits, and
# punctuation (؟ included).
_ARABIC_LO, _ARABIC_HI = "ء", "ۿ"

# Language this tenant's code-switching actually uses for embedded technical
# terms (data/code_switching_rules.md, generate_training_data.py's
# build_code_switching_prompt) -- French, never a generic "foreign" default.
_LATIN_SPAN_LANG = "fr"


def _token_script(token: str) -> str:
    """Classify one whitespace-delimited token as 'arabic', 'latin', or
    'neutral' (digits/punctuation-only, e.g. "27-06", "؟", "?") -- neutral
    tokens have no script of their own and must not force a language switch."""
    if any(_ARABIC_LO <= c <= _ARABIC_HI for c in token):
        return "arabic"
    if any(c.isalpha() for c in token):
        return "latin"
    return "neutral"


def _split_language_spans(text: str, default_lang: str = "ar") -> list[tuple[str, str]]:
    """Split `text` into consecutive same-script spans, each tagged with the
    XTTS language code to synthesize it with.

    Why this exists: XTTS's Xtts.inference() takes exactly ONE language code
    per call -- it prefixes the whole input string with a single [lang] token
    and runs the entire text through that language's cleaners/tokenizer in
    one pass (confirmed by reading TTS/tts/models/xtts.py and
    TTS/tts/layers/xtts/tokenizer.py directly). This tenant's fine-tune is
    DELIBERATELY trained to embed literal Latin-script French/English
    technical terms inside otherwise-Arabic-script Darija sentences
    (generate_training_data.py's build_code_switching_prompt/
    row_is_code_switched -- an enforced convention, not an edge case; see
    real examples in tests/data/voice_eval/codeswitch_*.txt, e.g.
    "واش خاصني نلبس le casque ديال sécurité..."). Synthesizing a whole such
    sentence under one language tag mispronounces whichever script isn't
    the tag's own. Splitting into per-script spans and calling inference()
    once per span (app.services.tts.py's caller concatenates the resulting
    audio) is the only way to get correct pronunciation for both halves.

    Neutral tokens (bare digits/punctuation, e.g. a law reference "27-06" or
    a lone "؟") attach to the CURRENT span rather than starting a new one --
    they carry no script signal of their own, and splitting on them would
    fragment audio for no linguistic reason. `default_lang` (the request's
    own overall language, already `_LANG_MAP`-resolved) is used only for the
    degenerate case where the entire text is neutral.
    """
    tokens = text.split()
    if not tokens:
        return []

    span_tokens: list[list[str]] = []
    span_langs: list[Optional[str]] = []
    for token in tokens:
        script = _token_script(token)
        if script == "neutral":
            if span_tokens:
                span_tokens[-1].append(token)
            else:
                span_tokens.append([token])
                span_langs.append(None)
            continue
        lang = "arabic" == script and "ar" or _LATIN_SPAN_LANG
        if span_langs and span_langs[-1] in (lang, None):
            span_tokens[-1].append(token)
            span_langs[-1] = lang
        else:
            span_tokens.append([token])
            span_langs.append(lang)

    return [(" ".join(toks), lang or default_lang) for toks, lang in zip(span_tokens, span_langs)]


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


# Splicing independently-synthesized per-language spans back together is OFF
# by default -- it was built, listened to on real lease audio, and rejected.
# See _split_language_spans's docstring for why the idea is sound in principle
# (XTTS takes one language tag per inference() call, so a mixed-script sentence
# necessarily mispronounces one of its two scripts) and docs/LESSONS_LEARNED.md
# #13 for why it loses in practice: XTTS is autoregressive and conditioned on
# speaker prosody, so each span is generated as its own COMPLETE utterance,
# with its own utterance-initial and utterance-final prosody. Concatenating
# those never reads as one sentence, and short spans (a lone "ديال" or
# "sécurité") trail off into audible hallucinated filler. Two live listening
# rounds (2026-09-06/07) both landed worse than the plain single-call path --
# the second markedly so, because it tried to fix that filler by appending
# terminal punctuation to each fragment as a stop cue, and sentence-final
# punctuation is exactly what tells the model to apply sentence-final prosody
# to what is only a mid-sentence clause.
#
# Set TTS_SPAN_SPLIT=1 to re-enable for A/B experiments. Mispronunciation of
# embedded French terms under the single-call path is a real, still-open
# limitation -- the promising untried fix is Arabic-script transliteration of
# those terms (keeping ONE inference call, no splicing), not splitting.
_SPAN_SPLIT_ENABLED = os.environ.get("TTS_SPAN_SPLIT", "").strip().lower() in ("1", "true", "yes")

# Silence inserted between spans when span splitting is explicitly enabled --
# long enough to avoid an audible click at the splice, short enough not to
# read as a pause inside what should be one sentence.
_SPAN_GAP_SECONDS = 0.04


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

    # Default path: ONE inference() call for the whole sentence -- the
    # behavior that demoed successfully live. Span splitting is opt-in and
    # off by default; see _SPAN_SPLIT_ENABLED above.
    if _SPAN_SPLIT_ENABLED:
        spans = _split_language_spans(text, default_lang=language) or [(text, language)]
    else:
        spans = [(text, language)]

    if len(spans) == 1:
        span_text, span_lang = spans[0]
        out = model.inference(span_text, span_lang, gpt_cond_latent, speaker_embedding, temperature=0.65)
        wav = np.asarray(out["wav"], dtype=np.float32)
    else:
        gap = np.zeros(int(SAMPLE_RATE * _SPAN_GAP_SECONDS), dtype=np.float32)
        parts: list[np.ndarray] = []
        for i, (span_text, span_lang) in enumerate(spans):
            out = model.inference(span_text, span_lang, gpt_cond_latent, speaker_embedding, temperature=0.65)
            parts.append(np.asarray(out["wav"], dtype=np.float32))
            if i != len(spans) - 1:
                parts.append(gap)
        wav = np.concatenate(parts)

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
