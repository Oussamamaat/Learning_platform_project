#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — the voice bake-off (~1.5 h). Run lease-00-seed.sh first (it fetches
# tests/data/voice_eval/ -- if that step reported no wav files, build the eval
# set locally per plan Part 3.1, commit+push, then re-run lease-00-seed.sh's
# git-clone block before this script; no redeploy needed).
#
# No redeploy needed either way -- STT_ENGINE=whisper / TTS_ENGINE=piper are on
# from the initial deploy (deploy/akash-deploy.yaml).
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
APP_PY=/app/.gguf_venv/bin/python

echo "== triggering one transcription now so the whisper/seamless weight download"
echo "   (3-9 GB, first use only) overlaps with reading this instead of blocking on its own =="
ls /app/tests/data/voice_eval/*.wav >/dev/null 2>&1 && \
  "$APP_PY" -c "
from app.services.stt import get_stt_engine
import glob
w = glob.glob('/app/tests/data/voice_eval/*.wav')[0]
print('warming STT on', w)
audio = open(w, 'rb').read()
print(get_stt_engine().transcribe(audio, sample_rate=16000).text[:80])
" || echo "  (skipped -- no eval wavs yet)"
echo

echo "== STT bake-off (WER + RTF, faster-whisper vs SeamlessM4T-v2) + TTS synthesis check =="
bash scripts/benchmark/run_bakeoffs.sh
echo "  -> scripts/eval_stt_results.json, scripts/eval_tts_results.json + per-sentence .wav files"
echo

echo "== end-of-speech -> first-audio latency (needs a real utterance.wav -- plan Part 3.2) =="
if [ -f /app/tests/data/utterance.wav ]; then
  VOICE_WAV=/app/tests/data/utterance.wav bash scripts/benchmark/bench_all.sh
  echo "  -> benchmark_voice.json; target ~1.2-2.2s (voice-assistant.md sec.4, an estimate, never measured before this)"
else
  echo "  SKIPPED -- put a 16kHz mono utterance.wav at /app/tests/data/utterance.wav first"
  echo "  (git-clone it in same as the other fixtures, or curl/paste it directly -- it's small)"
fi

echo
echo "== copy these out now =="
echo "  scripts/eval_stt_results.json  scripts/eval_tts_results.json  benchmark_voice.json (if it ran)"
echo "  and every synthesized .wav from the TTS check -- listen to them, 'sounds like Darija' isn't"
echo "  a number any script produces."
echo
echo "Reminder: faster-whisper/CTranslate2 expects language codes fr/ar, Seamless expects ary --"
echo "the app passes through fr/darija untranslated (speech_worker_resident.py's own docstring"
echo "flags this as unwritten). If STT quality looks wrong, check this before blaming the model."
echo
echo "If STT_ENGINE=whisper fails outright (unlikely on rtx8000, more likely on rtx5090/pro6000se --"
echo "see plan doc 1.1b), that needs a redeploy to STT_ENGINE=seamless -- the one expected mid-lease"
echo "redeploy in this plan."
