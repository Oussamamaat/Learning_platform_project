#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Phase 0 voice bake-offs — the "which STT model is best?" decision this deploy
# exists to unblock (docs/architecture/voice-assistant.md).
# Run INSIDE the container (the STT engines spawn the .speech_venv worker):
#     bash scripts/benchmark/run_bakeoffs.sh
#
# PREREQUISITE (user-supplied): labeled audio under tests/data/voice_eval/
#     <id>.wav   16 kHz mono PCM
#     <id>.txt   reference transcript (UTF-8)
#     <id>.lang  "fr" | "ary"
# e.g. from atlasia/DODa-audio-dataset. Without it, eval_stt.py has nothing to
# score — this script says so and still runs the TTS synthesis check.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/../.."

APP_PY=${APP_PY:-/app/.gguf_venv/bin/python}
EVAL_DIR=tests/data/voice_eval

echo "== STT bake-off (faster-whisper vs SeamlessM4T-v2) =="
if [ -d "$EVAL_DIR" ] && ls "$EVAL_DIR"/*.wav >/dev/null 2>&1; then
    "$APP_PY" scripts/eval_stt.py && echo "  -> scripts/eval_stt_results.json"
else
    echo "  SKIP: no labeled audio in $EVAL_DIR/*.wav (see this script's header)."
    echo "  Add real Darija/French/code-switched utterances there, then re-run."
fi

echo
echo "== TTS synthesis check (Piper fr + ar voices) =="
"$APP_PY" scripts/eval_tts.py && echo "  -> scripts/eval_tts_results.json + per-sentence .wav to listen to"

echo
echo "Done. Listen to the TTS .wav files and compare eval_stt_results.json to pick winners,"
echo "then set STT_ENGINE / TTS_ENGINE in deploy/akash.env accordingly."
