#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# One-shot benchmark harness. Run INSIDE the container (needs nvidia-smi + the
# venvs); samples VRAM while the LLM/OCR benchmarks run and writes a report.
#     bash scripts/benchmark/bench_all.sh
# Env:
#   BASE_URL    (default http://localhost:8000)   FastAPI base
#   OLLAMA_URL  (default http://localhost:11434)  for raw token-rate
#   OCR_IMAGE   (optional) page image/PDF for the OCR bench
#   VOICE_WAV   (optional) 16 kHz mono WAV to also run the voice bench
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/../.."

APP_PY=${APP_PY:-/app/.gguf_venv/bin/python}
BASE_URL=${BASE_URL:-http://localhost:8000}
OLLAMA_URL=${OLLAMA_URL:-http://localhost:11434}
OUT=${OUT:-benchmark_report.md}
VRAM_LOG=$(mktemp)

echo "IBLOG Tutor — RTX 5090 benchmark  ($(date -u +%FT%TZ))" | tee "$OUT"
echo "base_url=$BASE_URL" | tee -a "$OUT"

# ── VRAM sampler in the background ───────────────────────────────────────────
( while true; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null >> "$VRAM_LOG"
    sleep 1
  done ) &
SAMPLER_PID=$!
trap 'kill $SAMPLER_PID 2>/dev/null' EXIT

section() { echo; echo "## $1" | tee -a "$OUT"; }

# ── 1. GPU ───────────────────────────────────────────────────────────────────
section "GPU / CUDA"
"$APP_PY" scripts/benchmark/bench_gpu.py | tee -a "$OUT"

# ── 2. Health + models ───────────────────────────────────────────────────────
section "Service health"
curl -sf "$BASE_URL/health" | tee -a "$OUT"; echo
echo '```' >> "$OUT"; ollama list >> "$OUT" 2>&1; echo '```' >> "$OUT"

# ── 3. LLM latency + language switch ─────────────────────────────────────────
section "LLM latency / throughput"
"$APP_PY" scripts/benchmark/bench_llm.py --base-url "$BASE_URL" \
    --ollama-url "$OLLAMA_URL" --out benchmark_llm.json | tee -a "$OUT"

# ── 4. Voice (optional) ──────────────────────────────────────────────────────
if [ -n "${VOICE_WAV:-}" ]; then
    section "Voice pipeline"
    "$APP_PY" -m pip show websockets >/dev/null 2>&1 || "$APP_PY" -m pip install -q websockets
    "$APP_PY" scripts/benchmark/bench_voice.py --base-url "$BASE_URL" \
        --wav "$VOICE_WAV" --out benchmark_voice.json | tee -a "$OUT"
else
    section "Voice pipeline"
    echo "SKIPPED (set VOICE_WAV=... and STT_ENGINE/TTS_ENGINE off 'none')" | tee -a "$OUT"
fi

# ── 5. OCR (optional) ────────────────────────────────────────────────────────
if [ -n "${OCR_IMAGE:-}" ]; then
    section "OCR (heavy PaddleOCR-VL)"
    "$APP_PY" scripts/benchmark/bench_ocr.py --image "$OCR_IMAGE" --out benchmark_ocr.json | tee -a "$OUT"
else
    section "OCR"
    echo "SKIPPED (set OCR_IMAGE=/path/to/page.png)" | tee -a "$OUT"
fi

# ── 6. Peak VRAM ─────────────────────────────────────────────────────────────
kill $SAMPLER_PID 2>/dev/null; trap - EXIT
section "Peak VRAM during run"
if [ -s "$VRAM_LOG" ]; then
    PEAK=$(sort -n "$VRAM_LOG" | tail -1)
    echo "peak GPU memory used: ${PEAK} MiB" | tee -a "$OUT"
fi
rm -f "$VRAM_LOG"

echo; echo "Report written to $OUT (+ benchmark_*.json)."
