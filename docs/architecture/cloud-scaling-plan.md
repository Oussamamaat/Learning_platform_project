# Cloud scaling plan — ingestion and serving

**Status: target architecture, not built.** Everything in `docs/architecture/` except this
file describes what is actually deployed. This file describes what should change when the
platform leaves the single-laptop deployment, and it is written against **measured** local
numbers so the projections can be checked rather than believed.

Every "today" figure below was measured on this laptop (RTX 4060 Laptop, 8 GB VRAM,
`arabic_test.pdf` 80 pages / `french_test.pdf` 39 pages, tenant `company_efg`). Every
"cloud" figure is an estimate and is labelled as one.

---

## 1. The measured starting point

| Stage | Measured today | Notes |
|---|---|---|
| Native text parse | whole document in seconds | CPU only; 50 of 80 pages take this path |
| Page render (open + rasterise + PNG) | **~200 ms/page** | of which opening the PDF is **1 ms** |
| OCR — heavy (PaddleOCR-VL, preprocessing off) | **52.5 s/page** warm | 62.0 s/page with default preprocessing |
| OCR — light (PP-OCRv5 classic) | **5.5 s/page** warm | ~10× faster; 0/4 on formulas, 4/5 on tables |
| Model construct (heavy, preprocessing off) | 9.2 s | 30.8 s with default preprocessing |
| Chat turn — Darija | 17–24 s | model fits in VRAM |
| Chat turn — French | 44–144 s, one 180 s timeout | 7.5 GB model, **31% CPU offload** |

**The shape of the problem: 99.8% of per-page ingestion time is inside the OCR call.**
Rendering, PDF I/O, chunking, and embedding are collectively noise. Any optimisation that
does not reduce the number or cost of OCR calls is not worth doing.

### What the two-tier router does today

`app/services/pdf_classify.py` decides per page, and `_ocr_pdf_page` routes by that decision:

| Page class | Route | `arabic_test.pdf` | `french_test.pdf` |
|---|---|---|---|
| native text usable | **no OCR** | 50 pages | 38 pages |
| `OCR_REQUIRED` (no usable text layer) | heavy | 9 pages | 0 |
| `OCR_PREFERRED` (native text + embedded table/figure) | light, escalating to heavy when the result shows no numeric evidence | 16 pages (9 escalated) | 0 |
| genuinely blank | skipped, recorded | 5 pages | 1 page |

Measured end-to-end, both documents: **55 min → 30 min (−45%)** with ground-truth recovery
unchanged at **13/13 stored, 4/4 retrieved**. `french_test.pdf` needs **zero** OCR and
ingests in seconds — the router is why.

**Two 300 s OCR timeouts inside that 30 min cost ~10 min on their own.** On a laptop that
is the VRAM ceiling showing through; it is one of the first things the cloud move removes.

---

## 2. Worker architecture

### Today

```
FastAPI (1 process)
  └─ ingest_queue: ThreadPoolExecutor(max_workers=1)   ← serial, by design
       └─ _ocr_pdf_page  ──JSON-lines/stdio──►  resident .ocr_venv subprocess (1)
                                                   └─ PaddleOCR-VL + PP-OCRv5
```

`max_workers=1` is deliberate, not a placeholder (see `app/services/ingest_queue.py`): two
concurrent OCR jobs would OOM an 8 GB card. The consequence is that **documents ingest
strictly serially** — 10 documents ≈ 10 × 30 min ≈ **5 hours**. For a B2B platform whose
onboarding step is "upload your corpus", that is the single most important thing to fix,
ahead of any per-page tuning.

### Target

Three tiers, each scaling independently, because each has a different bottleneck
(CPU-bound parse, GPU-bound OCR, GPU-bound generation):

```
API (stateless, N replicas)  ──►  job queue (SQS / Redis Streams / Celery)
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        ▼                            ▼                            ▼
  parse workers (CPU)         OCR workers (GPU)            LLM serving (GPU)
  - native text extraction    - heavy: PaddleOCR-VL        - vLLM / TGI
  - render pages to images    - light: PP-OCRv5            - both tutor models
  - chunk + embed             - autoscale on queue depth     resident
```

Four changes carry the work:

1. **Page-level fan-out, not document-level.** Today a document is one job. It should be
   *one job per page needing OCR*, so an 80-page document with 18 heavy pages occupies 18
   workers for one page's duration rather than one worker for 18 pages' duration. This is
   the difference between linear and near-constant time in document length, and it is the
   single biggest lever — bigger than any per-page speedup.
2. **Raise `ingest_queue`'s `max_workers`** (or replace it with the real queue). It exists
   solely to prevent an 8 GB OOM; that constraint does not survive the move.
3. **OCR as a service, not a subprocess.** `_ResidentOcrWorker`'s stdio contract already
   isolates this — swapping the transport for HTTP against a pool changes one class, not
   the pipeline. PaddleOCR already supports `vl_rec_backend="vllm-server"` /
   `"sglang-server"` with `vl_rec_server_url`, so this is configuration plus a deployment,
   not a rewrite.
4. **LLM serving off Ollama.** Ollama loads one model at a time and evicts on switch —
   visible today as the 5.7 GB unload/reload whenever a conversation changes language.
   vLLM/TGI keeps both tutor models resident with continuous batching.

### Reconciling this with existing invariants

- **Tenant isolation must survive the split.** `get_tenant_id()` (`app/config.py:141`) is
  a process-lifetime constant today, so a tenant switch means a restart. Distributed
  workers make that untenable: tenant must become a **job attribute**, validated from a
  JWT claim at the API edge, and carried through the queue message. The seam is already
  shaped for it (`get_tenant_id(request_tenant_id)` accepts and ignores an argument
  precisely so call sites don't change when the claim replaces it).
- **`reap_orphaned_processing` becomes wrong again.** It marks *this tenant's*
  `pending`/`processing` rows as errored at boot, which is right for one process owning
  one queue. With N stateless workers, a restarting worker must not reap jobs another
  worker is actively running. Replace with a **lease/heartbeat**: a row is reclaimable only
  after its lease expires, not because some process started.
- **Idempotency.** Page-level retries mean a page may be OCR'd twice. Chunk writes must be
  keyed on `(source_file_id, page_number, chunk_index)` so a retry replaces rather than
  duplicates. Today `insert_documents` generates a fresh `uuid4()` per chunk with no
  natural key — that is safe only because nothing retries.

---

## 3. Concurrency and multi-tenancy

**Batch upload (10+ documents).** Fan out at the page level and the aggregate is bounded by
worker-pool size, not document count. 10 copies of `arabic_test.pdf` = 180 heavy pages; at
32 OCR workers that is ~6 waves ≈ a few minutes, against ~5 hours serially today.

**Fairness.** A single tenant uploading 200 documents must not starve everyone else. Use
per-tenant queues with weighted round-robin, or a concurrency cap per tenant. Without this
the first large tenant to onboard becomes an outage for the rest — the classic noisy-
neighbour failure, and cheap to prevent up front.

**Isolation.** Postgres row-level security on `tenant_id`, per-tenant S3 prefixes for
uploads, and tenant on every queue message. The retrieval path is already scoped
(`search.py`'s `WHERE tenant_id = %s`, `sources.active_source_ids`); the *ingestion* path is
what currently leans on one-tenant-per-process.

**Backpressure.** Queue depth is the autoscaling signal. Cap in-flight jobs per tenant, and
surface honest progress — `source_files.status` already models `pending/processing/
partial/ready/error`, so per-page progress is a counter away.

**GPU sizing.** Heavy OCR and LLM serving should not share a card. PaddleOCR-VL is ~2 GB;
the tutor models are 5.7–7.5 GB each. On a 24 GB A10G you can run several OCR workers on
one card, or co-locate both tutors — not comfortably both.

---

## 4. Projected performance

**Estimates.** Basis: per-page OCR scales with GPU throughput (A10G ≈ 2–3× this laptop's
4060 for this workload, and removes the CPU-offload penalty entirely); parallel speedup is
linear in workers until the pool exceeds pages needing OCR.

Per-page, single worker:

| Stage | Today (4060 8 GB) | Est. A10G 24 GB | Est. L40S/A100 |
|---|---|---|---|
| heavy OCR | 52.5 s | ~20 s | ~10 s |
| light OCR | 5.5 s | ~2 s | ~1 s |
| native parse | ~0 | ~0 | ~0 |

End-to-end for `arabic_test.pdf` (80 pages: 50 native, 9 heavy, 16 light of which 9 escalate
→ **18 heavy + 7 light** OCR calls):

| Configuration | Estimated | vs today |
|---|---|---|
| Today, measured | **30 min** | baseline |
| A10G, 1 worker | ~6.5 min | 4.6× |
| A10G, 4 workers | ~2 min | 15× |
| A10G, 8 workers | **~1 min** | ~30× |
| L40S, 8 workers | **~30 s** | ~60× |

By document type:

| Type | Example | OCR pages | Est. (A10G, 8 workers) |
|---|---|---|---|
| Pure digital text | `french_test.pdf` | 0 | **seconds** — no OCR at all |
| Mixed layout | typical regulation w/ some figures | ~20% | ~15–30 s |
| Heavy visual / CAD / formula | `arabic_test.pdf` | ~31% | **~1 min** |

**Serving latency**, once the 8 GB ceiling and CPU offload are gone:

| Metric | Today | Est. cloud |
|---|---|---|
| Darija turn | 17–24 s | **2–4 s** |
| French turn | 44–144 s (31% CPU) | **2–4 s** |
| Language switch | full model unload/reload | **0** — both resident |
| Timeouts | 1 of 12 turns hit 180 s | none expected |

The French-vs-Darija asymmetry is **entirely** a VRAM artefact and should vanish; it is not
a property of the models.

---

## 5. Sequencing

Ordered by value per unit of risk:

1. **Page-level fan-out + real queue.** The dominant win; turns document length from a
   linear cost into a parallel one. Requires the idempotency key above.
2. **Tenant as a job attribute (JWT claim).** Unblocks 1 — distributed workers cannot each
   own a tenant by process identity.
3. **Lease/heartbeat instead of boot-time reaping.** Required before more than one worker
   can safely exist.
4. **LLM serving on vLLM/TGI with both tutors resident.** Fixes the largest user-visible
   latency and the language-switch stall.
5. **OCR behind an HTTP pool** (`vl_rec_backend="vllm-server"`). Mostly configuration.
6. **Re-tune the two-tier router.** With cheap parallel heavy OCR, the light tier's value
   drops and the honest move may be to route more pages to the heavy engine for fidelity.
   The router is a `settings.ocr_two_tier` flag away from being turned off.

## 6. What would invalidate these estimates

Stated plainly so they can be checked rather than trusted:

- **A10G ≈ 2–3× the 4060 for this workload is an inference, not a benchmark.** Measure one
  page on the real instance before committing to a worker count.
- **Linear parallel speedup assumes no shared bottleneck.** If OCR workers contend on the
  same Postgres or S3, the curve flattens earlier.
- **`PPStructureV3` is still unmeasured** — its cold-start download exceeded 10 minutes
  twice locally. `scripts/ocr_bakeoff.py` exists to finish that comparison; it may beat VL
  on tables at a fraction of the cost, which would change the routing table above.
- **Chunking and embedding are assumed negligible.** True at 80 pages; re-measure at
  thousands before assuming it holds.

**Detail & rationale:** measured numbers in
[data-and-retrieval.md](data-and-retrieval.md) ("Ingestion repair" and the two-tier entry);
routing code in `app/services/pdf_classify.py` and `app/services/ingestion.py`.
