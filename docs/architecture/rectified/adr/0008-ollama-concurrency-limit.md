# ADR 0008: Ollama Concurrency Limit

**Status:** Mechanism implemented and tested; the default was measured on an RTX 5090 on
2026-09-11 (see *Amendment*). Production serving moves to vLLM (ADR 0011), so this limit now
governs the Ollama rollback path only.
**Date:** 2026-09-04 (amended 2026-09-11)
**Depends on:** `docs/architecture/cloud-scaling-plan.md` #3, `POST_LEASE_MVP_SPRINT_PLAN.md` item 5

## Problem

`docs/architecture/cloud-scaling-plan.md` names this directly: "There is **no** semaphore, lock,
or queue in front of Ollama anywhere. `chat()` and `generate_quiz()` are sync endpoints, so N
concurrent users = N concurrent Ollama requests, unmodelled." Confirmed by grep across `app/`:
every call site (`_post_ollama`, `_call_ollama_generate`, `_call_ollama_chat`,
`_stream_ollama_chat` in `app/services/llm.py`, reached from chat, quiz, diagrams, and voice) had
no concurrency primitive anywhere in the path.

## Investigation method

Traced every Ollama call site and confirmed the sync/async shape of each caller:
`chat()` (`app/routers/chat.py:279`) and `generate_quiz()` (`app/routers/quiz.py:29`) are plain
`def`, dispatched to FastAPI's worker threadpool; `voice_session` (`app/routers/voice.py:158`) is
`async def`, but its own Ollama call happens inside `_answer_worker`, a plain `def` run via
`asyncio.to_thread`. Confirmed the embedding path (ingestion-time `embed_chunks`, query-time
`search.py`) never touches Ollama at all — both use an in-process `SentenceTransformer` — so a
generation-side limiter cannot starve retrieval.

## Options considered

- **`asyncio.Semaphore`** — rejected: all four call surfaces execute the actual blocking `urllib`
  call on a worker thread (via FastAPI's threadpool or `asyncio.to_thread`), never on the event
  loop itself, so an asyncio primitive would not correctly gate them.
- **`threading.Semaphore`, one shared instance** — chosen: correctly gates chat, quiz, diagrams,
  and voice uniformly regardless of which async/sync wrapper got them onto a worker thread.
- **Pick an arbitrary "reasonable" default and call it done** — rejected. The brief this sprint
  runs under asks for a number derived from measured resource use, and no concurrent-load
  measurement exists yet: phase 1 of the lease ran serially, and the 8GB laptop can't hold both
  tutors resident to even approximate concurrent load. Inventing a number and presenting it as
  measured would misrepresent the evidence in exactly the way this sprint's own standing rule
  warns against.
- **Ship the mechanism now with an explicitly-labeled placeholder, measure and set the real value
  in Phase B** — chosen.

## Evidence

- `tests/test_ollama_concurrency.py` (3 tests, all passing, no real Ollama server needed —
  `urllib.request.urlopen` is monkeypatched with a fake that blocks briefly and records peak
  simultaneous in-flight calls):
  - 5 threads calling `_post_ollama` concurrently with `ollama_max_concurrent=2`: observed peak
    concurrency = exactly 2, never higher, never falling back to serialized (1) — proves the
    semaphore actually blocks the 3rd+ caller rather than merely being constructed and ignored.
  - A slot freed after a successful call is immediately usable by the next caller (no permit
    leak on the happy path).
  - A slot freed after a *failed* call (simulated connection error) is immediately re-acquirable
    (no permit leak on the error path) — verified via a non-blocking `acquire()` after 3
    consecutive failures against a budget of 1.
- Full local test suite: 725 passed (up from the prior 676 across this sprint's added tests),
  including all pre-existing `test_llm.py`/`test_generation_gates.py` coverage — no behavioral
  regression from wrapping `_post_ollama` and `_stream_ollama_chat` in the semaphore.

## Decision

1. **Mechanism**: one shared `threading.Semaphore(settings.ollama_max_concurrent)`
   (`app/services/llm.py`, lazily built the same way `app.services.ingestion._get_pool` builds its
   connection pool), acquired for the full duration of `_post_ollama` (including its retries) and
   for the full lifetime of `_stream_ollama_chat`'s generator (connect through final token,
   released in a `finally` so early client abandonment — e.g. voice.py's `cancel_flag` — still
   frees the slot via `GeneratorExit`).
2. **Config**: `ollama_max_concurrent: int = 4` (`app/config.py`), env-overridable
   (`OLLAMA_MAX_CONCURRENT`) via the existing pydantic-settings mechanism, no new plumbing.
3. **The default (4) is explicitly labeled a conservative placeholder, not a measured value.**
   Phase B runs a real concurrency sweep (N = 1, 2, 4, 8 simultaneous chat requests against the
   redeployed 32GB lease) measuring latency percentiles, error rate, and peak VRAM, and this ADR
   is amended with the chosen limit read off that curve.

## Rationale

Shipping the mechanism now is correct independent of the exact number: today there is zero limit,
so any bound — even an unmeasured placeholder — is strictly safer than the status quo, and the
mechanism (not the number) is what needed engineering care (thread-vs-async correctness, retry and
error-path permit hygiene, generator lifetime). Deferring the number itself to a measured sweep
rather than guessing keeps this sprint's evidentiary standard consistent with items 1, 2, and 6:
name what's unconfirmed, ship what's correct regardless, measure before committing to a specific
value.

## Constraints acknowledged

The mechanism, its config surface, and its test coverage are all local/free work, fully verifiable
without GPU or lease access. The one thing genuinely gated on Phase B is the number itself — no
concurrent-load data exists yet on this deployment's actual hardware, and inventing one would not
meet the sprint's own bar for what counts as a measured decision.

## Amendment (2026-09-11): measured on an RTX 5090

**Method.** Same card and same rendered prompts as ADR 0011: 20 real RAG prompts per language,
`max_tokens` 300, temperature 0, N = 1, 2, 4, 8, 16, 32 concurrent workers, 2 requests per worker.
Two Ollama v0.33.2 servers: port 11434 with `NUM_PARALLEL` unset, port 11435 with
`NUM_PARALLEL=4`. Results: `results/public/bench_ollama_{unset,4}_{darija,fr}.json` in the private
dataset `Oussamamaat/iblog-vllm-lease`.

| Config | Language | p95 N=4 | p95 N=16 | p95 N=32 | Throughput N=16 | Errors N=32 | Peak VRAM |
|---|---|---|---|---|---|---|---|
| `NUM_PARALLEL` unset | Darija | 32.9 s | 113.2 s | 231.4 s | 0.15 req/s | 0/64 | 8,510 MiB |
| `NUM_PARALLEL` unset | French | 31.0 s | 125.8 s | 251.7 s | 0.13 req/s | 0/64 | 17,019 MiB |
| `NUM_PARALLEL=4` | Darija | 173.1 s | 419.9 s | 400.9 s | 0.03 req/s | 44/64 | 31,160 MiB |
| `NUM_PARALLEL=4` | French | 24.2 s | 82.7 s | 170.7 s | 0.23 req/s | 0/64 | 15,064 MiB |

**Findings.**

- With `NUM_PARALLEL` unset, throughput is flat from N=2 to N=32 for both tutors: Ollama is
  strictly serial, and p95 grows with the queue.
- `NUM_PARALLEL=4` raised French throughput about 1.7x and cut its p95 at N=16 from 125.8 s to
  82.7 s.
- The Darija `NUM_PARALLEL=4` row is not a valid measurement. That run started with the other
  server's model still loaded, so the card peaked at 31,160 of 32,607 MiB; its slowdown and its
  timeouts (9/32 at N=16, 44/64 at N=32) come from memory starvation, not from the setting.
- Neither configuration meets the p95 < 10 s target at N=16. That is why production moves to vLLM
  (ADR 0011).

**Decision.** Keep `ollama_max_concurrent = 4`, paired with `OLLAMA_NUM_PARALLEL=4` in the Ollama
rollback SDL. The clean French run shows 4 parallel slots improving throughput without errors, and a
semaphore of 4 matches the server's slot count, so requests wait in the app rather than inside
Ollama.
