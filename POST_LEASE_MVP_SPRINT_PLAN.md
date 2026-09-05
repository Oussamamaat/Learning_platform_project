# Post-Lease MVP Sprint — local fixes, then one verification lease

## Context

The Akash RTX 8000 lease (2026-09-04) answered the three questions an 8 GB laptop
couldn't, and left a punch list behind (`benchmark_results/README.md`, plan doc
`now-that-i-have-fuzzy-hopper.md` Part 6). Akash quota remains, so GPU is available
again — the constraint is no longer "no GPU" but "lease hours are spendable and should
be batched." That reinstates the repo's own prior discipline (plan doc Part 3, *"Local
prep, before spending a single GPU minute"*), which is why the work below splits into
**Phase A (local, free)** and **Phase B (one lease run)**.

Equal weight to the code: this becomes an academic paper. Every item needs
*problem → method → options → evidence → decision → rationale → constraints*.

**The grounding pass overturned three of the brief's carried-over hypotheses.** They
were planning-pass guesses, and the standing rule (§6) says confirm before fixing — so
the plan below is shaped by measurement, not by the punch list's wording:

1. **VAD is probably not a threshold problem.** Swept against the repo's 30 real
   16 kHz speech wavs, the current `threshold=500.0` fires `speech_start` on **30/30
   files**. Lowering it is unjustified by the only real-speech evidence in the repo.
2. **Item 4's proposed idempotency key is not constructible.** `documents` has no
   `page_number` and no `chunk_index` column, and `chunk_document` flattens `## Page N`
   headings and packs adjacent sections, so page identity is destroyed before insert.
   Fan-out's premise died too — see item 4.
3. **The reported Darija STT WER is largely a scoring artifact.** Added as item 6.

---

## Grounding summary — confirmed working, confirmed broken, untested

**Confirmed working (measured on the lease).** Both 9B tutors resident in 32 GB: chat
latency 2.0–3.4s across French, Darija, and alternating turns, language-switch penalty
gone (laptop paid 44–144s/flip). 76-page PDF ingested in 76.3s vs. the laptop's
30m14s. PaddleOCR-VL available. The **explicit-domain** chat path is correct
end-to-end — the same Darija question with `domain:"industrial"` returns a grounded
answer, correct citations, `language:"darija"`, `cross_language:true`.

**Confirmed broken.** (a) `resolve_domain` mis-routes at least one implicit-domain
Darija query to `tenant_default` + `language:"fr"` + instant refusal. (b) The live
voice interface did not reliably detect speech, and barge-in never fired. (c) Piper's
`ar_JO-kareem-medium` rejected on live listening.

**Untested / newly suspect.** Darija STT accuracy below the headline number. And no
model-output *quality* evidence exists at all — every lease artifact measures latency,
WER, or RTF; none scores answer correctness. So §3's finetune question currently has
no evidence base pointing either way, which is itself the finding.

**Environment.** Docker 29.1.3 up, no containers running; `optimized_elearning_platform_pgdata`
volume exists. Ollama installed, not serving; `IBLOG_TUTOR`, `iblog-tutor-fr`, `gemma2`
present (25 GB blobs). `BAAI/bge-m3` cached (4.3 GB). Full local repro is available —
and `resolve_domain` fails *before* generation, so item 1 needs only Postgres + bge-m3,
not the VRAM-tight LLM.

---

## Existing assets to reuse (do not rebuild)

| Asset | Why it matters |
|---|---|
| `tests/data/retrieval_eval.jsonl` | 50 real labelled queries — 30 `fr`, 20 `ary` — each with `domain` + `gold_sources`. The item-1 regression corpus: real phrasing, not toy strings. |
| `scripts/eval_retrieval.py` | Existing runner, `(context, sources)` contract, `EVAL_LANGUAGE_TO_UI_LANG = {"fr": "fr", "ary": "darija"}`. Extend, don't fork. |
| `tests/data/voice_eval/*.wav` (30) | Real 16 kHz mono 16-bit speech — 16 Darija, 8 French, 6 code-switched. Offline VAD corpus. |
| `benchmark_results/phase2_*_eval_stt_results.json` | Stores every `reference`/`hypothesis` pair → STT re-scores with **no model, no audio, no GPU**. |
| `app/services/citations.py` — `fold_arabic`, `arabic_variant_pattern` | Arabic orthographic normalization, already written and already proven (it unblocked the PaddleOCR gate). Reuse in item 6. |
| `app/routers/ingest.py:104-128, 239-246` | SHA-256 spool-hash + `source_files.sha256` duplicate short-circuit. Item 4's key keys off this, doesn't re-derive file identity. |
| `app/services/tts.py` `TtsEngine` Protocol + `_ENGINES` | A new TTS engine = one class + one dict line. `voice.py` unchanged. |
| `app/config.py` `Settings` (pydantic-settings) | New fields are env-overridable for free. Follow the `retrieval_backend` comment-block style. |
| `tests/conftest.py:67` `postgres_reachable` fixture | Existing convention for DB-dependent tests. |
| `docs/deploy/lease-00-seed.sh`, `scripts/benchmark/`, `deploy/make-local-sdl.sh` | The lease run's existing, already-proven harness. Phase B extends it; it does not get rewritten. |

---

## Decision log

`docs/architecture/rectified/adr/` holds `0001`–`0003`. This sprint continues that
series as **`0004`–`0009`**, one ADR per item, each carrying the brief's seven
headings (Problem / Investigation method / Options considered / Evidence / Decision /
Rationale / Constraints acknowledged). Rationale: it is the repo's only existing
decision-record convention, and `rectified/` is already the register for long-form,
history-preserving, work-in-progress records. A new top-level file would fragment it.

Each ADR is written in Phase A with its local evidence, then **amended in place** with
Phase B's lease evidence — matching how `0001` already records amendments inline
(`Changed`/`AMENDED`). Also update `benchmark_results/README.md` §"Two real bugs this
run found" to reflect resolution, and correct its item-2 STT framing if item 6 changes
the engine verdict.

---

# Phase A — local, free

## Item 1 — `resolve_domain` implicit-domain routing (ADR 0004)

**This is two independent bugs, not one.** Traced: `resolve_language` is called at
`app/routers/chat.py:296-307`, *before* `_resolve_turn_context`, and takes no domain
argument; `detect_query_language` (`app/services/llm.py:396-429`) is pure character-class
arithmetic. So `language:"fr"` **cannot** be a downstream consequence of the domain
fallback. Both symptoms need separate explanations.

`resolve_domain` (`app/services/routing.py:148-210`) has exactly three exits:
`"retrieval"`, `"no_match"` (search succeeded, nothing cleared `domain_vote_threshold`),
and `"tenant_default"` — reached only when `backend != "pgvector"` **or** the `try`
block raised into a bare `except Exception` that is logged then swallowed. The SDL
(`deploy/akash-deploy.yaml`) sets no `RETRIEVAL_BACKEND`, so it defaulted to
`"pgvector"` — **the exception branch is confirmed as the cause of the domain symptom.**
The exception itself is not yet identified; `logger.exception` writes to stderr with no
`FileHandler` anywhere, so the lease traceback is gone.

### Reproduce locally first (fast, free, and may settle it outright)
1. `docker compose -f config/docker-compose.yml --project-directory . up -d db` — only
   the `db` service; run the app from `.gguf_venv`, not the container.
2. Seed: `ingest_directory('raw/shared', tenant_id='company_abc')` → expect 25 files.
3. Probe script (scratchpad, not committed): call `search_similar_chunks(query, tenant_id=...,
   top_k=20, domain=None, source_ids=active_source_ids(...))` — the *exact* failing
   call — inside try/except for **all 50** `retrieval_eval.jsonl` queries, printing the
   full traceback on failure. This settles the plan doc's own fastest-narrowing question
   ("a French query with domain omitted was never tried"): if French raises too, the bug
   is not language-specific.
4. Separately probe the language symptom: `detect_query_language` on the 20 `ary`
   queries, plus a cp1252/utf-8 round-trip of the seed message to test the mojibake
   hypothesis. `docs/deploy/lease-00-seed.sh` issues the identical message via `curl`
   with no `domain` and no `language` — so shell-mangled Arabic text is a live
   candidate for `language:"fr"`, and it is testable locally.

### Fix, by what the repro shows
- **Raises reproducibly** → fix the raiser; narrowest correct change.
- **Does not raise locally** → do *not* guess. The lease is the environment where it
  actually failed (different Postgres image, different locale/encoding, different corpus
  state), so this becomes a **Phase B** item: ship the diagnostics below, capture the
  real traceback there, then fix. This is the single strongest argument for the
  redeploy, and it is why item 1 no longer has to end in a hypothesis.

**Ship regardless of which branch fires** — these are correct on their own merits and are
what make Phase B's capture possible:
- Log the exception *type* and the query's script class at the swallow point, not just a
  generic message.
- Surface the swallowed failure in the response — `degraded` already exists on
  `ChatResponse`; reuse it, so a routing failure is visible instead of masquerading as a
  legitimate tenant-default.
- Persist server logs on the lease (the prior run had stderr only, which is exactly why
  the traceback was lost). Redirect to a file in the Phase B run script.

**Same flaw elsewhere — name it, per the brief.** `app/routers/quiz.py:49-51` calls
`resolve_domain` identically when `domain` is omitted (same exposure, real impact).
`app/services/ingest_jobs.py:163-185` `_resolve_upload_domain` re-implements the same
`search_similar_chunks` + `vote_domain` + bare-`except` pattern independently (cosmetic
impact only — a display badge). Fix the shared function; the quiz path inherits it.
Decide explicitly on the `ingest_jobs.py` duplicate rather than leaving it silent.

### Definition of done
- Root cause identified with a captured traceback — locally if it reproduces, on the
  lease if not.
- New `tests/test_routing_darija_e2e.py`: parametrized over the 20 `ary` rows of
  `retrieval_eval.jsonl`, unmocked `resolve_domain` against real Postgres, gated on the
  existing `postgres_reachable` fixture. **This closes a real coverage gap** — today no
  test anywhere exercises unmocked `resolve_domain` with Arabic-script text
  (`tests/test_routing.py` and `tests/test_domain_routing.py` mock `search_similar_chunks`
  and use no Arabic; `tests/test_chat.py`'s Arabic tests all pass `domain` explicitly and
  never reach `resolve_domain`).
- ADR 0004.

---

## Item 2 — VAD: calibrate and instrument locally, diagnose on the lease (ADR 0005)

**Setup, confirmed.** `EnergyEndpointer` — `threshold=500.0` (RMS of raw int16, not dB),
`hangover_ms=400`, `min_speech_ms=200`, `SAMPLE_RATE=16000`, `FRAME_MS=20`,
`FRAME_BYTES=640`. Instantiated bare at `app/routers/voice.py:178`, no config path.
`push()` is called from `:293` (barge-in, SPEAKING) and `:303` (endpointing, LISTENING)
— one instance, one root cause for both symptoms. `tests/test_vad.py` uses only
synthetic square waves (amplitude 0 and 10000), so nothing has ever tested the default
against real speech.

**Measured this session** (offline sweep, real `tests/data/voice_eval/*.wav`, driving
the real `EnergyEndpointer`):

| threshold | files detected | mean segments | mean start |
|---|---|---|---|
| **500.0 (current)** | **30/30** | 1.20 | 834 ms |
| 300.0 | 30/30 | 1.17 | 785 ms |
| 200.0 | 30/30 | 1.13 | 759 ms |
| 100.0 | 30/30 | 1.10 | 715 ms |
| 60.0 | 30/30 | 1.03 | 675 ms |

Frame-RMS distribution over all 6291 frames: p25 = 145, **p50 = 964**, p90 = 4728;
61.4% of frames exceed 500. **Conclusion: the threshold is not the confirmed cause.**
Lowering it costs false-triggers (p25 of these files is silence at 145) and buys
nothing measurable on real speech.

**Where the real cause more likely lives — the hypotheses Phase B tests:**
- The frontend requests `getUserMedia({echoCancellation, noiseSuppression, autoGainControl})`
  all `true` (`useVoiceSession.ts:141-150`) — browser AGC and noise suppression alter
  the level before the server ever sees a sample, and `noiseSuppression` can gate
  low-energy speech to near-zero.
- `new AudioContext({sampleRate: 16000})` is a *request*. If the browser/OS declines it,
  the worklet's 320-sample frames are no longer 20 ms, so `min_speech_ms`/`hangover_ms`
  silently mean something else, and the audio handed to STT is at the wrong rate.
- Frame-size assumptions: nothing in `voice.py` validates that an inbound binary
  message is exactly `FRAME_BYTES`.
- Barge-in specifically: at `voice.py:293-300`, a `speech_start` while SPEAKING awaits
  `worker_task` inside the receive loop — confirm that isn't stalling intake.

### Phase A deliverables
1. **Instrumentation.** A debug path in `voice.py` logging per inbound frame: byte
   length, computed RMS, current state, endpointer event. Gated behind a setting, off by
   default. This is what makes the Phase B live run diagnostic rather than anecdotal.
2. **Config knob.** `vad_threshold`, `vad_hangover_ms`, `vad_min_speech_ms` in
   `app/config.py` following the `retrieval_backend` comment pattern, read at
   `voice.py:178`. Correct independently of the root cause — the module's own docstring
   says "tune per deployment", and today that requires editing Python. Free env-var
   override; no new plumbing. **This is also what makes a Phase B threshold change a
   restart instead of an image rebuild** — the exact reason the fix was skipped last time.
3. **Commit the offline sweep as a script** so the table above is reproducible for the
   paper, not a one-off console run.
4. Test in `tests/test_vad.py` covering the configurable path.

**Phase B (lease):** live mic run against the real STT/TTS stack — the environment where
it actually failed, and one the 8 GB laptop cannot faithfully reproduce (both engines
resident). Capture the RMS trace, identify the cause, tune via env var, re-test without a
rebuild.

**Record the limitation.** Whatever the live run shows is one mic, one browser, one
room. Say so in the ADR — it is exactly the generalization caveat the paper needs, and
it is the argument for the Silero VAD upgrade the module docstring already flags
(blocked on `onnxruntime`, not yet a dependency).

---

## Item 3 — Darija TTS: survey now, fine-tune only if the survey justifies it (ADR 0006)

**What's already settled and must not be re-litigated** (`app/services/tts.py` docstring):
XTTS-v2 and MMS-TTS were already rejected on licensing — Coqui CPML and CC-BY-NC are
both non-commercial, unusable in a B2B product. Piper was chosen because it is MIT.
Any candidate that fails the commercial-license test is out regardless of quality.

**Survey** (subagent, web research) — for each candidate capture: license (commercial
use permitted?), offline/self-hostable, Darija vs. MSA vs. Maghrebi-adjacent, quality
signal (samples/community feedback), and usable-as-is vs. needs-finetuning. Cover at
minimum: Piper checkpoints fine-tuned on Moroccan Darija; other MIT/Apache engines with
Arabic voices; Maghrebi-adjacent voices as an interim; and the Moroccan-NLP community's
output (`atlasia/*` on HF — `atlasia/DODa-audio-dataset` is already named in
`voice-assistant.md` as the intended fine-tune corpus, and its metadata is already in
the local HF cache).

**Ranked recommendation, two buckets:** (a) usable now, no training; (b) needs
fine-tuning. Integration cost for (a) is genuinely small — a class implementing the
`TtsEngine` Protocol plus one line in `_ENGINES` (`tts.py:166-169`); `voice.py` doesn't
change.

**Fine-tune decision is gated on the survey, not assumed.** If an off-the-shelf
commercially-licensed Moroccan Darija voice exists, training one is wasted lease time.
If nothing qualifies, the survey's deliverable is a ready-to-run Piper/DODa fine-tune
plan **with eval criteria defined before the run** — and Phase B executes it on the same
lease. Defining the criteria first is the point: it is what lets the paper say why the
fine-tune was worth doing.

**Third option that must be on the ballot:** ship Darija **text-only** and French voice
for the MVP. `tts_voice_fr = "fr_FR-siwis-medium"` is a native French voice and was not
rejected. The plan doc already raises this. Given the deadline, do not let it lose by
default.

---

## Item 4 — Idempotency key; fan-out deferred, and now empirically testable (ADR 0007)

**Both of this item's premises failed investigation.**

*The proposed key does not exist.* `documents` (`app/models/database.py:25-78`) has no
`page_number`, no `chunk_index`, and no UNIQUE constraint — only the `uuid4()` PK and
two non-unique composite indexes. `_parse_pdf` emits page boundaries as `## Page N`
markdown headings (`ingestion.py:408`); `split_by_headings` then treats them
indistinguishably from `### Article 5`, and `chunk_document` packs adjacent short
sections together (`ingestion.py:1058-1065`), so one chunk can span two pages. Per-chunk
metadata is only `{heading, section_index}` — `section_index` is neither stable nor
1:1 with a chunk.

*Fan-out's justification evaporated.* "10 documents ≈ 5 hours" is a **laptop** figure.
The lease measured **76.3s for a 76-page PDF** → 10 documents ≈ 13 minutes. And
`ingest_queue.py`'s `max_workers=1` is a documented deliberate choice, not an oversight:
"GPU OCR cannot run concurrently with itself on one card... a second concurrent job
would only contend for the same GPU/CPU resource, never add real throughput."

### Decision: implement the key, defer fan-out, and settle the deferral with a measurement
**Key:** a chunk-level content hash — `sha256(tenant_id, source_file_id-or-source_name,
chunk_content)` — which *is* constructible from what genuinely exists at insert time,
paired with a new nullable column plus a partial UNIQUE index, and `ON CONFLICT DO
NOTHING` in `insert_documents`'s `execute_values`. It keys off `source_files.sha256`'s
existing file-level identity rather than re-deriving it. No Alembic in this repo
(`db_init.py:43`) — follow the existing hand-rolled `CREATE INDEX CONCURRENTLY`
precedent at `db_init.py:100-114`.

**Concurrency is already safe** and should be stated rather than re-solved: SQLAlchemy
uses a fresh `Session` per `_get_session()` call over a locked shared engine
(`app/models/db.py:58-78`), and raw psycopg2 uses a `ThreadedConnectionPool`
(`ingestion.py:821-830`). The barrier to more workers is GPU contention, not DB safety.

**Test coverage gap to close:** there is currently **zero** test of `insert_documents`
itself — every ingestion test mocks around it, so a duplicate-insert regression passes
the whole suite today. Add a real-Postgres test asserting that ingesting the same file
twice yields the same chunk count.

**Phase B upgrade over the previous plan:** the fan-out deferral no longer rests on
extrapolating from one 76-page document. Measure a real **multi-document batch** ingest
on the lease and put that number in the ADR. If it contradicts the extrapolation,
fan-out comes back on the table with evidence instead of assumption.

---

## Item 5 — Ollama concurrency limit, with a *measured* number (ADR 0008)

**Confirmed:** no semaphore, lock, queue, or rate limit anywhere in front of Ollama.
Call sites: `_post_ollama` (`llm.py:615`), `_call_ollama_generate` (`:698`),
`_call_ollama_chat` (`:732`), `_stream_ollama_chat` (`:765`, its own `urllib` call that
bypasses `_post_ollama`), reached from chat (`chat.py:279`), quiz (`quiz.py:29` →
`quiz.py:136`), diagrams (`diagrams.py:748`), and voice (`voice.py:134`).

**`threading.Semaphore`, not `asyncio`** — and the reason is worth recording: `chat()`
and `generate_quiz()` are plain `def`, so FastAPI runs them on its worker threadpool;
voice's handler is `async def` but the Ollama call happens inside `_answer_worker`, a
plain `def` dispatched via `asyncio.to_thread`. All four surfaces end up on worker
threads, so one shared threading primitive gates all of them uniformly.

**No starvation risk:** embeddings never touch Ollama — both ingestion-time
(`embed_chunks`) and query-time (`search.py`) embedding run through an in-process
`SentenceTransformer`. A generation semaphore cannot starve retrieval. (They still
contend for the physical GPU, which no software semaphore controls — say so.)

**The number is now measurable, so measure it.** Previously this would have been an
invented default dressed up as a decision. Phase A lands the mechanism plus an
`ollama_max_concurrent` setting; **Phase B runs a concurrency sweep** on the 32 GB box
(N = 1, 2, 4, 8 concurrent chat requests → latency percentiles, error rate, peak VRAM)
and the ADR records the chosen limit against that curve. This is exactly what the brief
asked for and what the closed lease made impossible.

**Sequencing:** the mechanism lands last in Phase A — a concurrency limit added earlier
could mask or complicate reproduction in items 1–2.

---

## Item 6 — Re-score the STT bake-off (ADR 0009) *(new; approved)*

**Problem.** The headline Darija numbers (`mean_wer` 0.583 seamless / 0.728 whisper) are
inflated by scoring, not transcription. `scripts/eval_stt.py:51-52` scores raw
`reference.split()` vs `hypothesis.split()` — no normalization, no punctuation
stripping — and its own docstring admits it is "deliberately not `jiwer`... not
publication-grade WER with punctuation/casing normalization rules."

**Evidence already in hand** (from the stored pairs, no re-run needed):

| id | reference | hypothesis | WER | actual defect |
|---|---|---|---|---|
| `doda_ary_03` | غالبا غيجريو عليه من الخدمة | غالبا غيجريو عليه من الخدمة**.** | 0.2 | a trailing period |
| `doda_ary_00` | هوما مخبيين… انا | هما مخبين… أنا | 0.5 | orthographic variants, identical meaning |
| `doda_ary_07` | غنمرض | غان مرض. | **2.0** | word-split + punctuation |
| `codeswitch_00` | …le casque ديال sécurité… | …كاسك ديال "لا سيكوريتي"… | 0.45 | correct, transliterated into Arabic script |
| `codeswitch_05` | …la loi 27-06… | …la loi vingt sept zéro six… | 1.0 | correct, digits spelled out |

Darija has no standardized orthography, so raw WER penalizes spelling choices as
transcription errors. This is a measurement-validity problem with direct paper value.

**Method (free, offline, no model/audio/GPU).** Re-score the stored pairs with: Arabic
orthographic folding via the existing `fold_arabic` (`app/services/citations.py`),
punctuation stripping, digit normalization, and **CER alongside WER** — CER is the
appropriate primary metric for a dialect without standard orthography. Report raw and
normalized side by side; never silently replace the published number.

**Why it matters beyond the metric.** (a) It feeds §3 directly — a bad transcript looks
exactly like a bad model, and finetuning against a metric artifact is the failure mode
the repo's own history already records. (b) It may change the engine verdict: seamless
wins Darija by ~15 raw WER points but costs 1.6x the RTF (0.336 vs 0.208). If
normalization closes that gap, whisper becomes the better MVP choice on latency — and
Phase B can then A/B the two engines live on the same box to confirm.

---

# Phase B — one lease run

**Pre-flight (all local, all before spending a minute).**
- Full suite green with Postgres up.
- Fixes merged and the GPU image rebuilt/pushed by the existing CI workflow
  (`ghcr.io/oussamamaat/iblog-tutor:gpu`) — the lease pulls the image, so **code fixes
  must be in the image before deploy**, not applied in-pod.
- `bash deploy/make-local-sdl.sh` exits clean; re-run the plan doc's Verification
  checks (no `persistent`, no `storage:` params, `docker manifest inspect` succeeds).
- Add server-log persistence to the run script — stderr-only is why the item-1 traceback
  was lost last time.
- Have the Phase B agenda scripted before deploy, in the style of
  `docs/deploy/lease-00-seed.sh`. Copy artifacts out as each item finishes, not at the end.

**Agenda, in dependency order:**
1. Seed + the existing health/model gates (`lease-00-seed.sh`), unchanged.
2. **Item 1 verification** — the seed's own no-`domain`/no-`language` Darija call must
   return grounded with citations. If item 1 did not reproduce locally, capture the
   traceback here; that is this lease's highest-value minute.
3. **Item 2 live mic** — real STT/TTS resident, RMS trace captured, cause identified,
   threshold tuned via env var and re-tested without a rebuild.
4. **Item 5 concurrency sweep** — N = 1/2/4/8, latency percentiles + error rate + peak
   VRAM. This produces the number the ADR commits to.
5. **Item 4 multi-document ingest timing** — settles the fan-out deferral with a real
   batch measurement instead of an extrapolation.
6. **§3 evidence gathering** at 2–3s/turn (see below).
7. **Item 3 fine-tune**, only if the survey concluded nothing off-the-shelf qualifies,
   and only with eval criteria already written.
8. Item 6's engine A/B (whisper vs. seamless live), if the re-score changed the verdict.

---

## §3 — Finetune assessment (evidence-only, as briefed)

**Do this after item 1's fix**, per the brief's sequencing point. Two confounds are now
concrete and must be stated up front: a misrouted query refuses without the model ever
seeing the context, and item 6 shows the Darija transcript feeding the voice path is
scored — and possibly produced — worse than it reads.

**The finding that shapes this section:** there is currently **no answer-quality
evidence at all**. Every lease artifact measures latency, WER, or RTF. `mean_wer` is
transcription, not tutoring. So the honest assessment starts by saying the question
cannot be answered from existing artifacts, and then generates the missing evidence:

- Run `scripts/eval_retrieval.py` (50 labelled queries) post-fix for grounded retrieval
  quality, and `scripts/eval_refusal.py` for refusal behaviour.
- Sample real outputs across routing / RAG / voice, and name **specific** failure modes
  — fabrication, Darija fluency, refusal-language mismatch, tone — not a general
  impression. Two known-live rough edges to check first, both already recorded: French
  refusals falling back to Darija (untrained path), and the v11 adapter's refusal
  scope-mismatch wording.
- Mark every observed failure as *confounded by item 1* / *confounded by item 6* /
  *genuine model quality*. That three-way tag is the deliverable's core.

**Output:** a written assessment answering "sufficient for MVP as-is, or not" with named
failures; and if warranted, a ready-to-execute plan — target behaviours, data needed,
**eval criteria defined before the run** — gated on `green_light_model.md` §4.1 per the
existing convention. **No training this round**: `analyze_05` scopes the French LoRA at
~1,800 rows and 3–5 weeks of data work, so the long pole is data, not GPU time, and
launching against an unvalidated metric is the failure mode this project has already
paid for once.

---

## Sequencing

**Phase A (local):** item 1 alone first — fast, foundational, and everything downstream
is measured on a clean signal only after it lands. Once its root cause is identified (not
necessarily fixed), items 2, 3, and 6 run in parallel; items 4 and 5 land after those are
stable. Then pre-flight and image build.

**Phase B (lease):** the agenda above, once, in dependency order.

Deviate if investigation warrants — and record why in the ADR, per the brief.

---

## Verification

- `.gguf_venv/Scripts/python.exe -m pytest` green (676 expected), with Postgres up so
  the DB-gated tests actually run rather than skip.
- Item 1: `tests/test_routing_darija_e2e.py` passes unmocked against real Postgres —
  20/20 `ary` queries resolve to their labelled domain with `domain_source == "retrieval"`.
  On the lease, the seed script's own verification call returns grounded with citations.
- Item 2: the committed offline sweep reproduces the table above; the live mic RMS trace
  is captured and attached to ADR 0005; `VAD_THRESHOLD` demonstrably overrides via env
  without an image rebuild.
- Item 4: ingest the same file twice → chunk count unchanged, no duplicate rows; the
  lease's multi-document batch timing is recorded.
- Item 5: a concurrent-request test confirms the semaphore bounds in-flight calls, and
  the lease sweep produces the curve the chosen limit is read off.
- Item 6: the re-scored table reproduces from the committed JSON with one command.
- Six ADRs (`0004`–`0009`) exist, each with all seven headings, each amended with Phase B
  evidence, each readable standalone by someone who wasn't in this session.

---

# Phase A session log (2026-09-04/05) — what actually happened

Everything below is retrospective: what Phase A produced, in the order it happened, including a
mid-session finding that changed item 1's shape materially. Full detail lives in the six ADRs
(`docs/architecture/rectified/adr/0004`–`0009`); this is the connective narrative between them.

## Item 1 — resolved further than expected

Local reproduction against all 50 real eval queries found **zero exceptions** in the tier-2
pgvector path — the swallowed-exception hypothesis didn't reproduce. Diagnostics and a regression
suite (`tests/test_routing_darija_e2e.py`, 39 tests) shipped regardless (ADR 0004).

Then, testing the exact lease query through a locally running full server (`uvicorn`) two ways —
`curl -d '{"message": "شنو...؟"}'` with the Arabic text as a literal shell argument, vs. the
identical JSON sent via Python's `requests` — reproduced the **exact lease symptom shape**
(`domain_source:"no_match"`, `language:"fr"`, refusal) on the curl path, while the `requests` path
returned the correct grounded Darija answer, on the same running server, same Postgres, same
corpus, milliseconds apart. `curl --trace-ascii -` proved why: the shell/argv marshalling between
Git-Bash and the native `curl.exe` child process silently replaced every Arabic character with a
literal `?` before the request was ever built (82 garbled bytes sent vs. 116 correct UTF-8 bytes).
**This is not an application bug or a Postgres bug** — it fully explains the language-misreport
symptom, and is a strong, still-not-fully-confirmed lead on the domain symptom (this reproduction
lands on `"no_match"`, not the lease's reported `"tenant_default"` — see ADR 0004 for exactly what
Phase B needs to check to close that gap).

`docs/deploy/lease-00-seed.sh`'s own verification call used exactly this vulnerable pattern and has
been fixed (heredoc → file → `curl --data-binary @file`), which sidesteps the whole class of
argv-encoding risk regardless of shell or locale — the general fix, not a patch for one symptom.

## Item 6 — re-scored, verdict sharpened rather than reversed

Normalizing the STT bake-off's WER (Arabic orthographic folding, punctuation stripping) plus adding
CER did not close the whisper/seamless gap as the plan's own working hypothesis guessed — it
**widened** it on pure Darija (seamless normalized WER 0.428 vs. whisper's 0.699), while confirming
both engines are weak on code-switched utterances even after normalization. Full breakdown in
ADR 0009; `scripts/eval_stt_rescore.py` reproduces it from the committed JSON with one command.

## Item 2 — threshold hypothesis refuted by measurement

The offline sweep (`scripts/calibrate_vad.py`, 30 real utterances) found the current
`threshold=500.0` fires correctly on 30/30 files — not the bottleneck the carried-over hypothesis
named. Shipped anyway, because they're correct regardless of root cause: `vad_threshold`/
`vad_hangover_ms`/`vad_min_speech_ms` config knobs (env-overridable, confirmed), and
`vad_debug_log` per-frame RMS instrumentation for the Phase B live-mic run. ADR 0005.

## Items 3, 4, 5 — as scoped

TTS survey (ADR 0006): no candidate clears the immediate-use bar; ranked fine-tune plan (OuteTTS-
1.0 + DarijaTTS-clean/DODa) identified, gated on eval criteria before any GPU spend; text-only
Darija for MVP evaluated as a real option, not a fallback.

Idempotency key (ADR 0007): the plan's proposed `(source_file_id, page_number, chunk_index)` key
isn't constructible (page boundaries don't survive chunking); implemented a chunk-content-hash key
instead, `ON CONFLICT DO NOTHING`, tested 5/5 against real Postgres (identical re-ingest → 0 new
rows; partial edits → only the changed chunk inserts; different tenants/sources/uploads with
identical content don't collide). Fan-out deferred — its "10 docs ≈ 5 hours" justification was a
laptop figure; the lease's own 76.3s/76-page number implies ~13 minutes on the actual deployment
target, and `max_workers=1` was already a deliberate GPU-contention choice, not an oversight.

Ollama concurrency (ADR 0008): a shared `threading.Semaphore` now gates chat/quiz/diagrams/voice
uniformly; `tests/test_ollama_concurrency.py` proves it actually bounds concurrent calls (not just
constructed and ignored) and releases correctly on both success and error paths. The limit itself
(`ollama_max_concurrent=4`) is an explicitly-labeled placeholder — no concurrent-load data exists
yet on this hardware — pending Phase B's measured sweep.

## §3 — Finetune assessment (evidence-only, as briefed)

**Sequenced after item 1's investigation**, per the brief. Two confounds named up front: a
misrouted query never reaches the model, and item 6 shows the Darija transcript feeding voice is
scored (and possibly transcribed) worse than raw WER suggested.

**Retrieval and refusal, measured post-fix** (`scripts/eval_retrieval.py --backend pgvector`,
`scripts/eval_refusal.py`, against the real 25-file/37-chunk corpus):

- `recall@4 = 1.000`, `gold_substring_present = 1.000`, `MRR = 0.978` — retrieval itself is
  excellent for in-corpus content. Not a source of quality problems.
- `ood_refusal_rate = 0.474` (9/19) — well below the `0.8` headline the similarity-threshold sweep
  reported, because `eval_refusal.py`'s expanded set specifically probes harder categories the
  original sweep didn't: by group, `adjacent=0.167`, `everyday=0.600`, `english=0.667`,
  `labeled=0.600`. **This is a genuine, newly-quantified defect** — adjacent-but-outside-domain
  questions (tax rates, visa requirements, medication dosage, tenancy law) mostly retrieve *some*
  context and are NOT refused by the deterministic gate.
- `false_refusal_rate = 0.000` (0/45) — the gate never wrongly refuses in-corpus content. The
  problem is one-directional: too permissive on adjacent OOD, not too strict on real content.

**Real generation samples, production code path, both tutor models loaded locally** (Ollama +
`uvicorn`, RTX 4060 8GB — feasible for one model at a time):

- A confirmed-refused OOD French query ("recette du tajine aux pruneaux") got the correct French
  deterministic-refusal template — that path is a pure template lookup, not a model call, and is
  correct by construction.
- An adjacent-domain French query the gate did NOT refuse ("taux de l'impôt sur les sociétés au
  Maroc", blockchain-domain context retrieved) — the **model itself** correctly declined
  ("Je ne suis pas en mesure de répondre... je vous recommande de consulter... un
  expert-comptable"), despite the gate passing it through. Encouraging: model judgment partially
  compensates for the gate's permissiveness here.
- A second adjacent-domain French query the gate did NOT refuse ("délai légal de notification
  d'une fuite de données personnelles (RGPD)", securite-domain context retrieved) — the model
  answered confidently: *"Selon la Loi N° 09-08, vous devez déclarer une fuite de données
  personnelles à l'autorité compensée dans un délai maximal de 72 heures."* Verified against the
  actual retrieved source (`2.4_incident_reporting_procedures.md`): **"Loi 09-08" is genuinely
  present in that document** (correctly grounded), but **"72 heures" appears nowhere in either
  retrieved source file** — a specific, checkable, invented factual detail stated with full
  confidence, violating the system prompt's explicit "Never invent facts" instruction. This is
  real, reproducible, genuine-model-quality evidence, not confounded by item 1 (routing was
  correct here — `domain_source:"retrieval"`) or item 6 (this is text chat, no STT involved).

**Assessment.** Current behavior is **not uniformly sufficient for MVP as-is** — but the gap is
narrower and more specific than "the model isn't convincing": retrieval is excellent, in-corpus
refusal is reliable, and the model *sometimes* self-corrects a permissive gate. The concrete,
named failure is **partial fabrication on adjacent-domain, gate-missed queries** — a real citation
anchor blended with an unsupported specific claim. This is a refusal-gate/threshold calibration
problem *and* a grounding-discipline problem, not evidence that the underlying fine-tune
(`IBLOG_TUTOR`, `iblog-tutor-fr`) is broken — recall/MRR/false-refusal-rate all say the core
tutoring behavior is sound. **Recommendation: do not launch a new fine-tune from this evidence.**
The cheaper, better-targeted fix is tightening `domain_vote_threshold`/`similarity_threshold` for
the adjacent-OOD case (re-run `scripts/eval_refusal.py` against threshold candidates, same
methodology as the existing 2026-08-13 sweep) and only revisit fine-tuning if that alone can't
close the adjacent-refusal gap — with eval criteria (a fabrication-rate metric over held-out
adjacent-domain queries, not just recall) defined before any such run starts, exactly as ADR 0006
already commits to for TTS.

**What remains untested**: Darija-language generation quality specifically (both real-output
samples above were French — `iblog-tutor-fr`); voice-pipeline output quality end-to-end (STT →
LLM → TTS chained); and whether the same adjacent-domain fabrication pattern holds for Darija
queries. Named as open, not assumed either way — a natural first item for Phase B, where both
tutor models can stay resident together and the voice pipeline is testable end-to-end.

