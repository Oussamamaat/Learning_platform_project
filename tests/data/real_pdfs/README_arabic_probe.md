# `arabic_probe.pdf` — 10-page structural subset of `arabic_test.pdf`

Built 2026-08-24 with `pypdf` by copying 10 pages out of the 80-page
`arabic_test.pdf`. Purpose: **iterate on ingestion in ~3 min instead of ~16**,
without weakening what is actually being verified.

## Why a subset is not a weaker test

`arabic_test.pdf`'s 80 pages resolve to only **four** classifier branches, and
50 of the 80 are the same trivial `NATIVE` path. Re-running all 80 re-tests
that one path 50 times. The only thing the extra 70 pages measure is **how long
OCR takes**, which is a throughput question, not a correctness one — so it
belongs in a deliberate timing run, not in every iteration.

Every ground-truth needle asserted by `probe_rag_groundtruth_company_efg.py`
lives on a page kept here, so the STORED/RETRIEVED checks are **identical in
strength** against this file.

## Page map

| fixture | original | strategy | why this page is here |
|--------:|---------:|----------|-----------------------|
|  1 |  3 | `OCR_REQUIRED` | *no font resources at all* — a distinct classifier branch from a raster scan |
|  2 |  7 | `EMPTY`        | no-raster empty page: must be **reported** as unprocessed, not silently dropped |
|  3 | 12 | `NATIVE`       | densest native page (1826 chars) — heading/article-rich, exercises chunking + citation extraction |
|  4 | 15 | `OCR_PREFERRED`| **the single most valuable page.** Carries native-only values (`1970`, `0.999600`) *and* OCR-only ones (the CAD-layer table). The one page that proves native+OCR is a **merge, not a replace** |
|  5 | 18 | `NATIVE`       | TC-05 needle `خط الدفان` |
|  6 | 51 | `OCR_REQUIRED` | Tier-1 hand-verified. `1.5765`, `0.158`, `86400` — **OCR-only**, absent from the text layer. Was dropped entirely by the pre-fix classifier |
|  7 | 52 | `OCR_REQUIRED` | full-page raster scan (157 DPI) — the classic scanned-page branch |
|  8 | 61 | `NATIVE`       | TC-06 needle `12,000` |
|  9 | 64 | `OCR_PREFERRED`| OCR_PREFERRED with *rich* native text (1102 chars) — merge behaviour when native is substantial, vs. p15 where it is sparse |
| 10 | 80 | `NATIVE`       | TC-07 needle `BB1` |

Coverage: all 4 strategies, both `OCR_REQUIRED` sub-branches (raster scan vs.
embedded images vs. no font resources), and both `OCR_PREFERRED` regimes
(sparse vs. rich native text).

Verified after extraction: every page keeps the **same** classification it had
in the parent document (not automatic — classification reads embedded image
resources a naive page copy can drop), all 5 native-layer needles are present,
and all 6 OCR-only needles are correctly still absent pre-OCR.

## Running the probe against it

```
PROBE_ARABIC_DOC=arabic_probe.pdf .gguf_venv/Scripts/python.exe probe_rag_groundtruth_company_efg.py
```

`PROBE_ARABIC_DOC` defaults to `arabic_test.pdf`, so existing full-corpus runs
are unaffected. `PROBE_TENANT` likewise defaults to `company_efg`.

## Deliberately excluded

**Original p2** — `OCR_REQUIRED` with **72** embedded images. It reliably
exhausts the 300 s OCR timeout and yields nothing, costing 5 minutes per run
for a page that always fails. Add it back only when specifically testing
timeout handling / per-page failure isolation; keeping it out is what makes
this fixture fast.

Original pages carrying duplicate needles were also dropped: p35 (duplicate
`خط الدفان`, kept p18) and p26 (duplicate `BB1`, kept p80 — the page the
tenant benchmark actually cites).

## What this fixture does *not* replace

A **timing / throughput** run, and any check of behaviour that only emerges at
length (sustained VRAM pressure, the OOM-cascade class of bug). Those need the
full 80-page `arabic_test.pdf`, which is still in the repo root. Use it
deliberately, not by default.
