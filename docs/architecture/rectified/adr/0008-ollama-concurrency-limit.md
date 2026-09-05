# ADR 0008: Ollama Concurrency Limit

**Status:** Mechanism implemented and tested with a conservative placeholder default; the actual
limit is deferred to a Phase B measured concurrency sweep
**Date:** 2026-09-04
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
