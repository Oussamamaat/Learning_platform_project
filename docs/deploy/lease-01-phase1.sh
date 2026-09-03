#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — prove the box and get the headline number (~45 min).
# Run lease-00-seed.sh first. Exit criteria (plan doc Part 5, phase 1):
#   Darija/French turns in 2-4s (laptop: 17-24s / 44-144s), alternating-language
#   turns costing ~= same-language turns (proof both tutors are resident), peak
#   VRAM well under the accepted card's total.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

export BASE_URL=http://localhost:8000
export OLLAMA_URL=http://localhost:11434
# VOICE_WAV left unset here on purpose -- phase 2 does the real voice bake-off
# (run_bakeoffs.sh + a dedicated bench_voice.py pass). Setting it here would
# just duplicate that under a less complete report.

bash scripts/benchmark/bench_all.sh

echo
echo "== copy these out now, don't wait until the lease closes =="
echo "  benchmark_report.md"
echo "  benchmark_llm.json"
echo
echo "Read benchmark_report.md's 'LLM latency / throughput' section for the"
echo "Darija/French/alternating numbers, and 'Peak VRAM during run' for headroom."
echo "If both look right, proceed to lease-02-phase2.sh (needs tests/data/voice_eval/"
echo "populated locally first -- plan doc Part 3.1)."
