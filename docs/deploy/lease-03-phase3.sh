#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — OCR and ingestion (~2 h, OPTIONAL). Lowest priority; run only if
# phases 1-2 left budget and the questions still matter. Run lease-00-seed.sh
# first (it fetches /app/ocr_test/arabic_test.pdf).
#
# No redeploy, no UPLOADS_READ_ONLY flip -- OCR_ENGINE=paddleocr is on from the
# initial deploy, and this calls the ingestion pipeline directly instead of the
# (still-locked) upload API.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
APP_PY=/app/.gguf_venv/bin/python

if [ ! -f /app/ocr_test/arabic_test.pdf ]; then
  echo "arabic_test.pdf missing -- re-run lease-00-seed.sh's git-clone block first."
  exit 1
fi

echo "== bench_ocr.py — one page, direct kernel-support check =="
echo "   A clean paddle INSTALL does not prove kernel support on whatever card you're on"
echo "   (Dockerfile.gpu installs paddle from the cu126 index). If this fails, that's the"
echo "   accepted fallback -- leave OCR results out and stop here, don't chase it further."
"$APP_PY" scripts/benchmark/bench_ocr.py --pdf /app/ocr_test/arabic_test.pdf --page 0 --out benchmark_ocr.json
echo

echo "== full ingest timing, direct pipeline call (not the HTTP upload endpoint) =="
echo "   NOTE: ingest_directory() (used by lease-00-seed.sh) only globs .md/.txt --"
echo "   it would silently ingest 0 files here. ingest_file() is the one that routes"
echo "   PDFs through parse_document_to_markdown() -> OCR, so that's what this calls."
"$APP_PY" -c "
from app.services.ingestion import ingest_file
import time
t0 = time.time()
r = ingest_file('/app/ocr_test/arabic_test.pdf', tenant_id='company_abc', domain='securite')
print(r, 'in', time.time() - t0, 's')
"
echo "  -> laptop baseline: 30min14s after the two-tier OCR optimization (55min14s before it)"
echo
echo "Copy benchmark_ocr.json out. This is the last phase -- once you've pulled every"
echo "artifact from phases 1-3, close the lease (plan doc Part 4, money rule 2)."
