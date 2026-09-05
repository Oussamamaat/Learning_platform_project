# ADR 0007: Ingestion Idempotency Key; Fan-out Deferred

**Status:** Idempotency key implemented and tested; fan-out deferred, to be revisited with a
Phase B multi-document ingest measurement
**Date:** 2026-09-04
**Depends on:** `docs/architecture/cloud-scaling-plan.md` §7, `POST_LEASE_MVP_SPRINT_PLAN.md` item 4

## Problem

`docs/architecture/cloud-scaling-plan.md` ranks two related levers: (1) an idempotency key on
`insert_documents`, since it currently mints a fresh `uuid4()` per chunk with no natural key —
"safe only because nothing retries" — and (2) page-level ingest fan-out, calling
`app/services/ingest_queue.py`'s `ThreadPoolExecutor(max_workers=1)` "the '10 documents ≈ 5
hours' problem" and "the single biggest lever." The plan named (1) as the prerequisite for (2).

## Investigation method

Read the actual `documents` schema (`app/models/database.py`) and the chunking pipeline
(`app/services/ingestion.py`'s `_parse_pdf`, `split_by_headings`, `chunk_document`) to check
whether the proposed key — `(source_file_id, page_number, chunk_index)` — is actually
constructible from what exists at insert time. Read `app/services/ingest_queue.py` and
`app/services/ingest_jobs.py`'s `process_source_file` pipeline to find where fan-out would attach
and what state it would need to update concurrently. Checked the connection/session layer
(`app/models/db.py`, `app/services/ingestion.py`'s psycopg2 pool) for thread-safety. Cross-checked
the "10 documents ≈ 5 hours" premise against this sprint's own lease measurement
(`benchmark_results/README.md`: 76.3s for a 76-page PDF).

## Options considered

- **Implement the proposed `(source_file_id, page_number, chunk_index)` key** — rejected: none of
  the three components survive to insert time as a stable identity. `_parse_pdf` emits page
  boundaries as `## Page N` markdown headings (`ingestion.py:408`); `split_by_headings` then
  treats them identically to any other heading (e.g. `### Article 5`); `chunk_document` packs
  adjacent short sections together (`ingestion.py:1058-1065`), so one chunk can span two pages.
  Per-chunk metadata is only `{heading, section_index}` — `section_index` is an index into
  `split_by_headings`'s output, neither stable across re-chunks nor 1:1 with an emitted chunk.
- **A chunk-level content hash, scoped per tenant + source identity** — chosen (see Decision).
- **Implement fan-out alongside the key, as originally ranked** — rejected for this sprint: the
  documented justification for fan-out ("10 documents ≈ 5 hours") is a **laptop** figure, and this
  sprint's own lease measurement (76.3s for a 76-page PDF) implies roughly 13 minutes for 10
  documents on the GPU box, not 5 hours. `ingest_queue.py`'s `max_workers=1` is also not an
  oversight — its own docstring: "GPU OCR cannot run concurrently with itself on one card... a
  second concurrent job would only contend for the same GPU/CPU resource, never add real
  throughput." Building fan-out against a justification that may have evaporated risks real
  engineering effort (page-boundary-preserving re-chunking, concurrent progress-tracking on
  `source_files.pages_done`) for a problem that might not exist at the scale this deployment
  actually runs at. Phase B measures a real multi-document batch to settle this with evidence
  instead of extrapolation.

## Evidence

- `documents` (`app/models/database.py`) has no `page_number`, no `chunk_index` column, and (before
  this change) no unique constraint beyond the `uuid4()` primary key.
- `source_files.sha256` (`app/routers/ingest.py:104-128, 239-246`) already provides file-level
  dedup — a re-uploaded identical file short-circuits before `ingest_queue.submit` is even called.
  Nothing equivalent existed at the chunk level.
- Connection/session safety was already engineered correctly and needed no change: SQLAlchemy uses
  a fresh `Session` per `_get_session()` call over one locked shared engine (`app/models/db.py:58-
  78`); raw psycopg2 uses a `ThreadedConnectionPool` (`app/services/ingestion.py:821-830`).
- **Local test, 3 identical ingests of a real corpus file into a fresh tenant**: chunk count after
  1st ingest = 2, after 2nd = 2 (0 new), after 3rd = 2 (0 new) —
  `tests/test_insert_documents_idempotency.py::test_reingesting_identical_chunks_is_a_noop` and
  four further cases (partial-overlap re-ingest inserts only the changed chunk; identical content
  in a different source document still inserts; identical content for a different tenant still
  inserts; two uploads sharing a filename but different `source_file_id` both insert). All 5 pass
  against real Postgres.
- No pre-existing test called `insert_documents` directly at all — `tests/test_ingestion.py`'s own
  comment confirms it was always mocked around. This was a real coverage gap, now closed.

## Decision

1. **Idempotency key implemented**: `documents.content_hash` (nullable `VARCHAR(64)`,
   `app/models/database.py`), computed as
   `sha256(f"{tenant_id}|{source_file_id or source_name}|{chunk_content}")`, enforced by a
   `UniqueConstraint("tenant_id", "content_hash")`. `insert_documents`
   (`app/services/ingestion.py`) now builds this hash per chunk and inserts with
   `ON CONFLICT (tenant_id, content_hash) DO NOTHING RETURNING id`, returning the true post-dedup
   insert count (not the attempted count) so a caller can tell "nothing changed" from "this
   ingested fresh."
   - Scoped to `source_file_id or source_name`, not content alone: identical boilerplate text
     appearing in two genuinely different source documents must insert twice, not collide as if it
     were a re-ingest of the same document (tested explicitly).
   - Legacy rows keep `content_hash = NULL` — Postgres treats every `NULL` as distinct under a
     unique index, so they are silently exempt from the constraint rather than colliding with each
     other or requiring a backfill that would retroactively claim a guarantee for insert paths
     that never enforced one. Migration for an already-provisioned database (e.g. the Akash lease's
     volume) is documented by hand in `app/models/db_init.py`, following the repo's existing
     precedent for `pages_done`/the composite indexes (no Alembic in this project).
2. **Fan-out is deferred, not built this sprint.** The 76.3s-for-76-pages lease measurement already
   weakens its original justification; Phase B will measure a real multi-document batch ingest on
   the redeployed lease and either confirm the deferral or bring fan-out back with actual evidence.

## Rationale

Implementing a key that cannot actually be constructed from what the pipeline produces would have
been building on the plan doc's premise without checking it — exactly the mistake the standing
rule (confirm before fixing) exists to prevent. A content hash scoped to genuinely-available
identity (tenant, source, chunk text) delivers the real goal (safe retries, safe re-ingestion)
without inventing page/chunk-index fields the chunking pipeline doesn't preserve. Deferring fan-out
on a re-examined justification, rather than building it to satisfy a rank ordering set before this
session's own measurements existed, keeps effort proportional to a real, current bottleneck instead
of a laptop-era one.

## Constraints acknowledged

No GPU needed for any of this — pure database/application-layer work, fully verifiable locally
against real Postgres. What remains genuinely open is whether 10-document batches on the actual
deployment target behave like the single 76-page extrapolation suggests; that requires the
redeployed lease and is scoped to Phase B rather than guessed at here.
