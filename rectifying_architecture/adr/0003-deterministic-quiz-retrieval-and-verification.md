# ADR 0003: Deterministic Quiz Retrieval & Verification

**Status:** Proposed — describes a target design; current state is a stub (see Context).
**Amended 2026-08-03** for the laptop MVP's serial model loading.
**Date:** 2026-08-03
**Depends on:** ADR 0002 (`ui_lang`), ADR 0001 (model topology)

> **Changed 2026-08-03 — a new conflict, created by serial model loading.**
> The write path picks its model by `ui_lang`. On the laptop exactly **one** model is resident, so
> **background French quiz generation needs Gemma loaded while interactive Darija tutoring needs
> Atlas loaded** — a direct contention for the single GPU that did not exist in the cloud design.
> Resolved in the new section "Model residency conflict" below.
>
> The T0→T3 read path is **unaffected and is what makes the demo safe**: it never calls a model, so
> it cannot contend for VRAM regardless of which model is resident.

## Context

This ADR was requested as a "lock-in" of an existing warm/cold-path quiz architecture. Code
inspection found there is no existing architecture to lock in:

- `app/routers/quiz.py` returns hardcoded placeholder questions (`"[Placeholder] Question about
  {topic}?"`) with `TODO: Week 2 - RAG retrieval` and `TODO: Week 4 - LLM quiz generation` — it
  never calls the LLM, the database, or Redis.
- `Redis` is configured (`app/config.py:11`) but imported nowhere in the codebase.
- `QuizQuestion` (`app/models/database.py:29-42`) has no `language`, `difficulty`, `status`, or
  `source_document_id` column, and there is no attempt-history table.
- `_explanation_supports_answer()` (`generate_training_data.py:1745`) exists and is real, but its own
  docstring states it "cannot verify a quiz is factually right — only that it does not contradict
  itself," and it deliberately returns `True` on zero vocabulary overlap. It is one check inside a
  larger validation battery (`build_quiz_row()`, `generate_training_data.py:1766-1819`) that also
  catches malformed schemas, out-of-range answer indices, duplicate options, and CJK contamination.

So this ADR specifies what to build, not what already exists.

## Decision

### Read path — four tiers, deterministic, LLM never in the synchronous request path

| Tier | Source | Latency | Guarantee |
|---|---|---|---|
| T0 | Redis buffer, `LPOP` per `tenant:user:lang` | <50ms | Personalised, pre-validated |
| T1 | Postgres vetted bank, filtered by `tenant_id` + `language` + topic, excluding recently-seen | <50ms | Deterministic, tenant-scoped |
| T2 | Global seed bank (`tenant_id IS NULL`), same `language` | <50ms | Non-empty even for a brand-new tenant |
| T3 | Templated "content being prepared" response | 0ms | Never an error; never blocks on a model call |

Rationale: the pattern in `/theory-guide` (cold path bypasses the LLM on cache miss) is correct but
incomplete — it assumes a populated bank and has no answer for a new tenant or new topic with zero
rows. T2/T3 close that gap and guarantee the read path always terminates without invoking a model.

### Write path — admission-gated, async, never blocking a learner

- Quiz generation happens **off the request path**: triggered on module completion (or scheduled),
  using the query's `ui_lang` (ADR 0002) to select French or Darija generation.
- **Every LLM-generated item passes the full validation battery before admission to Redis or
  Postgres** — not just `_explanation_supports_answer()` in isolation, but the complete
  `build_quiz_row()` battery (schema shape, answer-index bounds, duplicate-option detection, CJK
  check, self-consistency). This is a cache-**admission** gate, applied once at write time, not a
  read-time check.
- **Trust model — two tiers**, matching the honest scope of what each check actually verifies:

  | `status` | Meaning | Serves | Guarantee |
  |---|---|---|---|
  | `auto` | Passed the full admission gate | T1 | Machine-checked self-consistency, **not** correctness |
  | `vetted` | A domain expert approved it | T2 (global seed bank) | Human-approved — this is the tier a new tenant falls back to, so it is the one that must be trustworthy |
  | `retired` | Withdrawn | none | — |

  The word "vetted" is never applied to gate-passed-only output. Review effort is spent once, on the
  seed set, rather than on every generated item.

### Runtime schema enforcement

Use the serving runtime's native structured-output facility — Ollama `format` (JSON schema) today,
vLLM `guided_json` in the cloud deployment (`analyze_03.md`) — not an external constrained-decoding
library (Guidance, LMQL, Instructor). This addresses the measured defect where the tuned model
invented an unrequested `{"questions": […]}` wrapper. No such call exists yet in `llm.py`; this is
new code, not a config change.

### Additional patterns adopted, not in the source theory-guide

- **IRT / Rasch-model item selection** for the cold path: pick the bank item whose difficulty is
  closest to the learner's current ability estimate, rather than an undifferentiated pool. Requires
  only a `difficulty` float per item and a per-learner ability estimate; pure SQL/Python, fully
  deterministic. This makes the "fallback" path genuinely adaptive rather than merely safe — the
  domain-correct choice for a platform whose product claim is adaptive learning.
- **Spaced-repetition exclusion** (Leitner/SM-2-style) for "recently seen" filtering, giving the
  attempt-history table a second purpose beyond dedup.
- **Postgres-backed job queue** (`SELECT … FOR UPDATE SKIP LOCKED`) instead of SQS for the async
  refill trigger — durable, well-proven, and adds no new infrastructure dependency to a stack that
  is already Postgres + Redis.
- **Transactional outbox**: enqueue the refill job in the same transaction that records module
  completion, avoiding a dual-write inconsistency between Postgres and the queue.
- **Bulkhead**: quiz generation runs on a separate worker pool/concurrency budget from interactive
  tutoring, so a quiz-generation backlog cannot starve the chat path of GPU capacity.

## ⚠️ Model residency conflict (added 2026-08-03) — laptop MVP only

**The conflict.** Quiz generation selects its model by `ui_lang` (French → Gemma, Darija → Atlas).
With serial loading, a background French quiz job requires evicting Atlas mid-conversation — and
vice versa. Left unmanaged this thrashes: evict, load (~30 s), evict, load.

**Resolution, by phase:**

### Demo — background generation OFF, bank pre-seeded
Disable the refill worker entirely and **pre-seed the quiz bank before the demo**. T0/T1/T2 all
serve from Redis/Postgres with **zero GPU involvement**, which the T0→T3 design already guarantees
by construction. The demo therefore has *no* quiz-related GPU contention at all, and quiz serving
stays <50 ms while the resident model is whatever the learner's dropdown selected.

### Laptop production — language-batched draining
Add `language` to `quiz_jobs` and make it the **batching key**. The worker drains **all pending jobs
for the currently-resident model**, then swaps **once** and drains the other language.

This amortizes the ~30 s swap across a whole batch instead of paying it per job — the difference
between one swap per drain cycle and one swap per item. Additional rules:
- Interactive sessions take priority; the worker yields and does not swap while a user is active.
- Prefer running drains during idle windows.

### Cloud phase — conflict disappears
Both 9B models co-reside on one 24 GB L4; generation and tutoring proceed concurrently. No batching
needed. This is a further concrete argument for the Phase 1 migration (`analyze_03.md`).

### ❌ Rejected: "generate with whichever model is currently resident"

Superficially the obvious optimization — it eliminates swaps entirely. **Rejected**, because it
would let Atlas write French quizzes (its documented French drift) or Gemma write Darija ones
(measured incoherent — `analyze_04`). That reintroduces precisely the language defect this entire
architecture exists to prevent, in **stored, reusable** content that then persists in the bank and
is served to future learners as `auto`-tier. A latency optimization is never worth corrupting the
durable artifact. **The model is selected by the item's language, always — never by convenience.**

## Schema changes (blocking)

```
quiz_questions:    + language              (existing Language enum, schemas.py:6)
                   + difficulty            (float, for IRT/Rasch selection)
                   + status                (auto | vetted | retired)
                   + source_document_id    (FK -> documents.id)
                   ~ tenant_id RLS predicate: USING (tenant_id = current_setting('app.tenant_id')
                                                      OR tenant_id IS NULL)   -- NULL = global seed
NEW quiz_attempts: user_id, question_id, tenant_id, answered_at, correct, ability_after
NEW quiz_jobs:     durable refill queue (SKIP LOCKED)
                   + language   -- added 2026-08-03: batching key for language-batched
                                --  draining under serial model loading (see above)
```

## Verification

- Feed the 12 historically-known bad rows (16% answer-key defect, `CHANGELOG.md`) through the
  admission gate; assert all 12 are rejected.
- Assert a `language=fr` request can never return an `ary` item, at every tier (T0–T3).
- Brand-new tenant, empty personalised bank: assert the request reaches T2 or T3 and never 500s.
- Assert the quiz read endpoint makes zero LLM calls (mock and assert zero invocations) — the
  read path must be pure.
- RLS: two tenants, `WHERE` clause deliberately removed, assert isolation still holds (fail-closed).

## Rejected alternatives

- **SQS** for the refill queue — adds an AWS dependency to a Postgres+Redis stack for no capability
  the existing infrastructure lacks.
- **Treating `_explanation_supports_answer()` alone as sufficient** — its docstring explicitly
  disclaims correctness verification; the full `build_quiz_row()` battery is the actual gate.
- **A single trust tier** — collapses "passed a heuristic" into "human-approved," which would let
  unreviewed content reach the global seed bank that every new tenant depends on.
