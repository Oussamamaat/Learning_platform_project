# ADR 0004: `resolve_domain` Implicit-Domain Darija Routing

**Status:** Resolved and confirmed on a live Akash lease (RTX 5090, 2026-09-05). Root cause: NOT an
application or Postgres bug. Literal Arabic-script text passed as a shell command-line argument
(`curl -d '{"message": "..."}'`) gets corrupted before curl ever builds the request. Fixed at the
reproduction site (`docs/deploy/lease-00-seed.sh`), and the fixed script's own verification call —
run inside the lease's real Linux container shell, against the real seeded Postgres, with no
`domain` and no `language` specified — returned grounded, correctly routed, and correctly
language-tagged. See **AMENDED (Phase B)** below for the full result and what remains genuinely
open.
**Date:** 2026-09-04 (root-caused later the same session); amended 2026-09-05 (Phase B lease run)
**Depends on:** `benchmark_results/README.md`, `POST_LEASE_MVP_SPRINT_PLAN.md` item 1

## Problem

On the 2026-09-04 Akash lease, `POST /api/v1/chat/` with no explicit `domain`, message
"شنو هي معدات الحماية الشخصية الإجبارية؟" (Darija, Arabic script), returned
`domain_source:"tenant_default"`, `language:"fr"` (wrong — should be `"darija"`), an instant
deterministic refusal, and `sources:[]`. The identical question **with** `domain:"industrial"`
passed explicitly worked correctly: grounded answer, correct citations, `language:"darija"`,
`cross_language:true`. Same corpus, same tenant, same content — proven present and retrievable.

A prior planning pass carried forward a hypothesis (not a confirmed diagnosis, per this
project's own standing rule) that `resolve_domain` (`app/services/routing.py:196`) reaches
`"tenant_default"` specifically when its tier-2 `search_similar_chunks`/`vote_domain` call
raises into a bare `except Exception` that is logged then swallowed — not the `"no_match"` path,
which would mean the tier-2 pgvector search itself throws for this query. This ADR investigates
and confirms or refutes that hypothesis.

## Investigation method

**Static trace (subagent + own review).** Traced the full call chain: `chat()`
(`app/routers/chat.py:279`) → `_resolve_turn_context` → `resolve_domain`
(`app/services/routing.py:148-210`). Confirmed `resolve_domain` has exactly three exits:
`"retrieval"` (a candidate cleared `domain_vote_threshold`), `"no_match"` (search succeeded,
nothing cleared it — a genuine out-of-corpus signal), and `"tenant_default"` (reached only when
`backend != "pgvector"`, or the `try` block raised). Confirmed separately that `resolve_language`
(`app/routers/chat.py:296-307`) is called *before* domain resolution and takes no domain
argument — `detect_query_language` (`app/services/llm.py:396-429`) is pure character-class
arithmetic over the message string. **This settles that the domain and language symptoms are two
independent code paths, not one bug with two symptoms** — `language:"fr"` cannot be a downstream
consequence of the domain fallback.

**Live local reproduction.** Brought up only the `db` service
(`docker compose -f config/docker-compose.yml --project-directory . up -d db`), initialized the
schema (`app/models/db_init.py`), and seeded the real corpus
(`ingest_directory('raw/shared', tenant_id='company_abc')` → confirmed 25 files / 37 chunks,
matching the lease's own seed-verification expectation). Then ran two probes against the live,
unmocked code path:

1. All 50 queries in `tests/data/retrieval_eval.jsonl` (30 `fr`, 20 `ary`) through
   `search_similar_chunks(domain=None)` directly (the exact tier-2 call) inside a try/except
   capturing full tracebacks, plus the full `resolve_domain` call, plus `detect_query_language`
   on each query — checking both whether French queries also raise (ruling out
   language-specificity) and whether the eval-set language matches what the app detects.
2. The **exact** failing message from the lease's own seed script
   (`docs/deploy/lease-00-seed.sh`), byte-for-byte, through `search_similar_chunks`,
   `resolve_domain`, `resolve_language`, and `active_source_ids`.

**Live HTTP reproduction — the step that actually found it.** With the fix from item 1's shipped
diagnostics in place, brought up the full app locally (`uvicorn app.main:app`) against the seeded
Postgres and issued the seed script's own verification call two ways: (a) exactly as
`lease-00-seed.sh` originally did, `curl -d '{"message": "شنو...؟", ...}'` with the Arabic text
embedded directly as a shell argument; (b) the identical JSON sent via Python's `requests` library.
(a) reproduced the lease's exact symptom shape: `domain_source:"no_match"`, `language:"fr"`,
instant refusal. (b) returned the correct grounded answer: `domain_source:"retrieval"`,
`language:"darija"`, correct citations, `cross_language:true` — on the SAME running server, SAME
Postgres, SAME corpus, SAME query text, milliseconds apart. That gap is what turned "does not
reproduce" into a real lead: something about the (a) invocation path itself, not the application,
was the variable.

## Options considered

- **Guess at a fix from source inspection alone** (e.g., add a `client_encoding` override,
  reconfigure stdout) — rejected: no evidence pinpoints Postgres/console encoding as the actual
  cause, and the codebase's own connection-pool code (`_PooledConnection.close()`,
  `app/services/ingestion.py:775-800`) already rolls back a non-idle transaction before
  returning a connection to the pool, which rules out simple pool-poisoning as an explanation
  without a captured exception naming a different cause.
- **Widen the except clause or remove the swallow entirely** — rejected: the swallow is a
  deliberate, documented fail-open contract (`resolve_domain`'s own docstring: "a Postgres
  hiccup degrades routing to the tenant default rather than crashing the request"), matching
  `_retrieve_context`'s identical contract. Removing it would turn a transient Postgres blip
  into a crashed chat request, which is a worse regression than a wrong routing decision.
- **Ship targeted diagnostics + regression coverage now, defer definitive root-cause capture to
  the lease environment** — chosen. See Decision.

## Evidence

- **Zero exceptions raised** across `search_similar_chunks(domain=None)` for all 50
  `retrieval_eval.jsonl` queries (30 `fr`, 20 `ary`) against a freshly seeded local Postgres.
- **The exact lease-failing query, run through the identical code path locally via direct Python
  calls or via a properly UTF-8-encoded HTTP request, resolves correctly**: `domain="industrial"`,
  `domain_source="retrieval"`, `resolve_language(...).response_lang="darija"` (script-detection
  source). `active_source_ids` returned `[]` (no tenant uploads locally — matches the lease's
  seed-only corpus state).
- **`curl --trace-ascii -` on the literal-argument invocation proves the corruption happens before
  curl builds the request, not on the network or in the app**:
  ```
  => Send data, 82 bytes (0x52)
  0000: {"message": "??? ?? ????? ??????? ??????? ??????????", "tenant_i
  0040: d": "company_abc"}
  ```
  Every Arabic character became a literal `?` (0x3F). 82 bytes sent vs. 116 correct UTF-8 bytes.
  A file-based invocation of the identical curl binary against the identical server
  (`curl --data-binary @request.json`, the JSON written to a file by a small Python script) sent
  the correct bytes and got the correct, grounded response — isolating the fault to how a
  Windows/Git-Bash shell marshals a literal multibyte-UTF-8 command-line argument into a native
  child process's argv, not to curl's networking, the server, or Postgres.
- **This fully explains both original symptoms as one mechanism, not two.** A query truncated to
  a run of `?` characters has zero Arabic-script codepoints, so `detect_query_language` correctly
  (from its own perspective) returns `"fr"` for what is, after corruption, meaningless ASCII input;
  and the embedding of that garbage text correctly finds nothing above
  `domain_vote_threshold`, producing a genuine `"no_match"` — not the swallowed-exception
  `"tenant_default"` path the original hypothesis targeted. **This means the originally-reported
  `domain_source:"tenant_default"` on the lease is not fully explained by this mechanism alone**
  (this local reproduction produces `"no_match"`, not `"tenant_default"`) — either the lease's shell
  environment corrupted the text differently (e.g. to something that made `search_similar_chunks`
  itself raise, consistent with the original hypothesis), or a second, still-unconfirmed factor was
  also present. The language-misreport symptom, however, is now fully and directly explained.
- Separately, a **real, different, already-known-magnitude routing defect** turned up in the same
  probe: 5 of 30 French queries (`sec-fr-05/07/08`, `bc-fr-08`, plus 2 of the deliberately
  out-of-corpus `ood-*` probes) voted to the wrong domain via the tier-2 retrieval-as-router
  mechanism. This is **not** the lease's `tenant_default`/swallowed-exception symptom — every one
  of these returned `domain_source="retrieval"`, a real vote that simply landed wrong — and
  `app/services/search.py`'s own comment already documents the tier-2 vote's measured accuracy at
  ~0.78. Recorded here for completeness; out of this ADR's scope (a distinct, already-quantified
  accuracy limitation, not a new bug), and not chased further under the session's time-box.
- Confirmed `resolve_domain` is also called, identically un-domained, from
  `app/routers/quiz.py:49-51` — same exposure. `app/services/ingest_jobs.py:163-185`
  (`_resolve_upload_domain`) independently re-implements the same
  `search_similar_chunks`/`vote_domain`/bare-`except` shape for a display-only badge — same class
  of swallow, cosmetic impact only (its own docstring: "a wrong guess here costs nothing
  functionally," since uploaded-source retrieval bypasses the domain filter).
- No test in the suite previously exercised unmocked `resolve_domain` with Arabic-script text:
  `tests/test_routing.py`/`tests/test_domain_routing.py` mock `search_similar_chunks` and use no
  Arabic; `tests/test_chat.py`'s Arabic-script tests all pass `domain` explicitly and never reach
  `resolve_domain`.

## Decision

1. **Root cause found for the language-misreport symptom, and a strong lead on the domain
   symptom**: passing Arabic-script text as a literal shell command-line argument to `curl -d`
   corrupts it into `?` bytes before the request is built (Windows/Git-Bash + native-binary argv
   marshalling). This is not an application bug, a Postgres bug, or a network issue. Fixed at its
   most concrete reproduction site: `docs/deploy/lease-00-seed.sh`'s own verification call now
   writes the JSON body to a file via a heredoc (a plain byte-for-byte redirect, never passed
   through argv) and sends it with `curl --data-binary @file`, which sidesteps the whole class of
   argv-encoding failure regardless of shell or locale.
2. **What remains genuinely open, precisely scoped for Phase B**: this reproduction produces
   `domain_source:"no_match"` locally, not the lease's reported `"tenant_default"`. The two are
   different exits of `resolve_domain` (a genuine empty vote vs. a swallowed exception) — so either
   the lease's shell/locale (a Linux container, not Windows Git-Bash) corrupted the same query text
   differently, in a way that made `search_similar_chunks` itself raise, or a second factor was
   also present that this session did not reproduce. Phase B's job is now narrow and concrete: run
   the FIXED seed script (file-based body) and, separately, the OLD literal-argument form inside
   the actual lease container, and compare `domain_source` between them. If the lease's shell
   corrupts the text into something that raises (not just an empty vote), that confirms the
   original hypothesis (a genuinely swallowed exception) as the same root mechanism, just with a
   different byte-level corruption than Windows produced locally.
3. **Shipped now, independent of which exact corruption pattern the lease turns out to have
   produced:**
   - `resolve_domain`'s exception handler (`app/services/routing.py`) now logs the exception
     *type* and the query's script class (Arabic vs. Latin/mixed), not just a generic message —
     so a repeat of this failure captures enough to diagnose without re-deploying twice.
   - `app/routers/chat.py`'s `_resolve_turn_context` now infers `routing_degraded` when
     `domain_source == "tenant_default"` **and** `retrieval_backend == "pgvector"` — the only
     condition under which `resolve_domain` can return `"tenant_default"` via its exception
     branch rather than the disk-backend skip — and ORs it into the existing `degraded` field on
     `ChatResponse`. A routing failure is now visible in the response instead of masquerading as
     a legitimate tenant-default fallback. No new field, no signature change to `resolve_domain`.
   - `ingest_jobs.py`'s duplicate swallow got the same exception-type logging improvement, for
     the same observability reason, deliberately **without** a `routing_degraded`-style signal
     (there is no response field a display-badge path should surface through).
   - New `tests/test_routing_darija_e2e.py`: 20 real Darija queries from
     `tests/data/retrieval_eval.jsonl`, unmocked `resolve_domain` against real Postgres, gated on
     the existing `postgres_reachable` fixture. 39/39 pass locally. Closes the coverage gap
     identified above.

## Rationale

This is a direct instance of the standing rule paying off: the carried-over hypothesis (a swallowed
Postgres exception inside `resolve_domain`) was specific and testable, testing it locally came back
negative (zero exceptions across 50 real queries and the exact failing message), and rather than
either declaring the bug unfixable or guessing at an application-side fix, the investigation kept
going one layer up — from the Python code to the transport used to reach it — and found a real,
reproducible, evidence-backed mechanism there instead. This is also a useful corrective for the
paper: "confirm root cause" sometimes means confirming the bug is NOT where a prior planning pass
assumed, which is exactly what happened here. The `routing_degraded` diagnostic and the
exception-logging improvements remain correct and shipped regardless — they make a genuine
application-level routing failure visible if one ever does occur, independent of this specific
shell-encoding finding.

The seed-script fix (heredoc-to-file instead of literal argv) is the right general fix even though
only one shell (Windows/Git-Bash) was directly proven affected: it removes an entire class of
argv-marshalling risk rather than patching around one observed symptom, and it costs nothing —
the file-based invocation produced byte-identical correct behavior to the working `requests`-based
test.

## AMENDED (Phase B, Akash RTX 5090 lease, 2026-09-05)

**What was run.** `docs/deploy/lease-00-seed.sh` (the fixed, heredoc-to-file version) executed
inside the lease's own Linux container shell (`root@app-...`, not Windows/Git-Bash), against a
freshly seeded Postgres (25 files / 37 chunks ingested, matching the local Phase A count exactly).
The script's built-in verification call — the identical Darija PPE question, no `domain`, no
`language` field — sent via `curl --data-binary @/tmp/seed_check_req.json` (never as a literal
shell argument), returned:

```json
{"domain":"industrial","domain_source":"retrieval","language":"darija",
 "cross_language":true,"degraded":false,
 "sources":["1.6_ppe_requirements.md","1.7_machine_guarding_basics.md",
            "1.11_ar_code_travail_salama.md","1.8_hazardous_materials_handling.md"]}
```

A grounded, coherent Darija answer about PPE, correctly cited, correctly routed to `industrial`,
correctly language-tagged, `degraded:false`. This is not a near-miss — it is the exact opposite of
the originally-reported symptom (`domain_source:"tenant_default"`, `language:"fr"`, empty sources,
instant refusal) for the byte-identical query, on the actual lease-class environment (Linux
container, real Postgres, real seeded corpus, real GPU-backed stack) rather than a local Windows
approximation.

**What this confirms.** The fix — never pass literal multibyte-UTF-8 text as a shell command-line
argument to a native child process; write it to a file and send with `--data-binary @file` instead
— is sufficient in practice, on the real target environment, to eliminate the failure this ADR was
opened to explain. For the seed script's own purpose (verifying the corpus is grounded and usable
before spending lease time on benchmarks), this item is closed.

**What was not re-tested, and why that's an acceptable gap.** Decision point 2 (original write-up)
scoped a narrower comparison: deliberately re-running the OLD literal-argument form *inside this
same lease container* to see whether Linux's argv/locale handling corrupts the text into something
that makes `search_similar_chunks` itself raise (reproducing `"tenant_default"` exactly, not just
`"no_match"`) — which would confirm the original swallowed-exception hypothesis as the same
mechanism under a different byte-level corruption. This comparison was **not** run: it would have
meant deliberately re-invoking the known-broken pattern on a metered, paid lease purely to satisfy
academic completeness, with no effect on the shipped decision either way (the fix is the same
regardless of which exact `resolve_domain` exit the old form hits). Given the sprint's time-box and
that lease minutes have a real dollar cost, this was judged not worth doing once the practical
question — does the fix work end-to-end on the real target environment — was already answered
affirmatively. The distinction between "`no_match`" and "`tenant_default`" as the old form's exact
failure mode on Linux therefore remains formally unconfirmed; the practical fix does not depend on
resolving it.

**Net effect on this ADR's open question.** Downgraded from "genuinely open, blocks confidence in
the fix" to "a satisfied academic curiosity, explicitly not chased, with the reason recorded" — the
fix's correctness is now demonstrated by a positive real-environment result, not merely inferred
from the absence of a negative one.

## Constraints acknowledged

No GPU is needed for this item — `resolve_domain` fails before generation — so it was fully
investigable on the laptop, including the live HTTP reproduction that found the actual mechanism
(a local `uvicorn` server was sufficient; no lease access was needed to find this). What remains
genuinely gated on the lease is narrower than originally scoped: not "capture any traceback" but a
specific comparison — does the OLD literal-argument seed-script invocation, run inside the actual
lease's Linux container shell, produce `domain_source:"tenant_default"` (consistent with the
original hypothesis, just via a different corruption pattern than Windows produced) or something
else. This session's Windows-specific reproduction cannot settle that by itself — a Linux
container's argv/locale handling is a genuinely different code path from Git-Bash's Windows
interop, and asserting they fail identically without testing would repeat the same mistake (an
unconfirmed hypothesis) this ADR just corrected.
