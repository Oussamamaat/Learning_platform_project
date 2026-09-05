# Phase 3 — OCR & ingestion (2026-09-04)

Ran on the same RTX 8000 lease as phases 1-2, `app-9469d6fb8-bxg4z`, against
`arabic_test.pdf` (76 pages, 13.1 MB).

## OCR (bench_ocr.py, page 0)

`OCR_ENGINE=paddleocr`, `paddle_engine=vl` (PaddleOCR-VL). Failed to initialize:

```
RuntimeError: A dependency error occurred during pipeline creation. Please
refer to the installation documentation to ensure all required dependencies
are installed.
```

This is the accepted fallback risk flagged in `deploy/akash-deploy.yaml`'s own
header notes and `Dockerfile.gpu` (`paddlepaddle-gpu` install wrapped in
`|| true` specifically because kernel support isn't guaranteed). Not
investigated further — OCR was never one of the three core questions this
lease existed to answer.

## Full-document ingest (ingest_file, direct call — not the HTTP upload API)

9 of 76 pages needed OCR (scanned pages / no text layer / image-only content)
and were skipped gracefully rather than crashing the run — each skip logged
with its specific reason (full-page raster, embedded images too small to
count as a raster, no font resources, etc.).

Result:
```
{'chunks_created': 33, 'source_name': 'arabic_test.pdf', 'status': 'success'}
in 76.26 s
```

**76.3 s vs the laptop baseline of 30 min 14 s (1814 s) — ~24x faster.**

The pipeline's graceful-degradation path (OCR unavailable → skip page → keep
going) worked correctly under real conditions, not just in the mocked test
suite — a useful confirmation independent of whether OCR itself succeeded.
