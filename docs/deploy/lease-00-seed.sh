#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Paste this FIRST into the lease shell, once, right after /health comes up.
# Seeds the corpus (raw/ is baked into the image but never ingested — see the
# plan doc Part 2) and verifies the deploy is actually usable before spending
# any more of the lease on benchmarks. Nothing below needs to run again this
# lease — it survives phases 1-3 since the lease stays open the whole time.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
APP_PY=/app/.gguf_venv/bin/python

echo "== bench_gpu.py — confirm the card, compute capability, live matmul =="
"$APP_PY" scripts/benchmark/bench_gpu.py
echo "  -> if cuda_available is false or you see 'no kernel image', STOP and close the lease now."
echo

echo "== ollama list — expect BOTH tutor models =="
ollama list
echo

echo "== /health =="
curl -sf http://localhost:8000/health && echo
echo

echo "== seeding the corpus from raw/ (baked in, never auto-ingested) =="
"$APP_PY" -c "
from app.services.ingestion import ingest_directory
r = ingest_directory('raw/shared', tenant_id='company_abc')
print(len(r), 'files ingested')
"
echo "  -> expect '25 files ingested'. If it's 0, something about raw/'s layout changed — stop and check before benchmarking."
echo

echo "== verify with a real chat turn (must be grounded, not a refusal) =="
curl -sf -X POST http://localhost:8000/api/v1/chat/ \
  -H 'Content-Type: application/json' \
  -d '{"message": "شنو هي معدات الحماية الشخصية الإجبارية؟", "tenant_id": "company_abc"}' \
  | tee /tmp/seed_check.json
echo
echo "  -> read /tmp/seed_check.json: if it's the deterministic refusal, the DB is still empty and every"
echo "     downstream benchmark number in phases 1-3 would be meaningless. Do not proceed until this is grounded."
echo
echo "Seed + verification done."
echo

echo "== fetching phase 2/3 fixtures (not baked into the image) =="
echo "   arabic_test.pdf (phase 3 OCR/ingest test) + tests/data/voice_eval/ (phase 2"
echo "   STT bake-off) are git-committed but not COPY'd into the Dockerfile -- pull"
echo "   them from the public repo once, here, rather than fighting the upload API."
rm -rf /tmp/repo_fixtures
git clone --depth 1 https://github.com/Oussamamaat/Learning_platform_project /tmp/repo_fixtures
mkdir -p /app/ocr_test /app/tests/data
cp /tmp/repo_fixtures/arabic_test.pdf /app/ocr_test/arabic_test.pdf 2>/dev/null \
  && echo "  -> /app/ocr_test/arabic_test.pdf ready" \
  || echo "  -> arabic_test.pdf not in the clone yet (commit+push it before phase 3)"
if [ -d /tmp/repo_fixtures/tests/data/voice_eval ]; then
  cp -r /tmp/repo_fixtures/tests/data/voice_eval /app/tests/data/voice_eval
  echo "  -> /app/tests/data/voice_eval ready ($(ls /app/tests/data/voice_eval/*.wav 2>/dev/null | wc -l) wav files)"
else
  echo "  -> tests/data/voice_eval/ not in the clone yet -- build it locally (plan Part 3.1),"
  echo "     commit+push, then re-run just this git-clone block (no need to redeploy the lease)."
fi
cp /tmp/repo_fixtures/tests/data/utterance.wav /app/tests/data/utterance.wav 2>/dev/null \
  && echo "  -> /app/tests/data/utterance.wav ready (phase 2 end-to-end latency test)" \
  || echo "  -> tests/data/utterance.wav not in the clone yet (commit+push it before phase 2's latency step)"

echo
echo "Proceed to lease-01-phase1.sh."
