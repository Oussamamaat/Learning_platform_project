# Lessons learned

An index, not a duplicate — each entry is condensed from a fuller account elsewhere in
this repo. Follow the pointer for the full incident. Use this to avoid regressing on a
fix that already cost real time or GPU hours to find.

### 1. A fine-tune can land below its own base model
**Problem:** the first fine-tuned adapter confidently fabricated a law citation
(«حسب القانون 27-04») on a prompt where base Atlas-Chat correctly declined
(«ماكاينش شي قانون محدد»), 2/2 each way in a controlled comparison.
**Root cause:** distributional, not a few bad rows — legal Q&A training rows almost
always cited *something*, so the model learned "legal question → emit citation" and
filled the slot from pretrained knowledge when the given context had none.
**Proven fix:** every new adapter gets compared against the *base* model, not just the
previous adapter — deleting flagged rows would have left the teaching distribution
fully intact and missed this.
→ `resurrection.md` Q8.1, `LOCKEDIN_PLAN.md` §6.2

### 2. Row count is not an acceptance criterion
**Problem:** two iteration cycles were spent tuning `COMPONENT_CONFIG`'s
`multi_turn_pct` — a generation-time probability — as though it were a gate threshold.
**Root cause:** conflating a knob that shapes *how* data is generated with the
acceptance criteria that decide *whether* a run is good enough.
**Proven fix:** acceptance criteria live only in `green_light_model.md` §4.1, never in
generation config.
→ `resurrection.md` Q4.4

### 3. Population mismatch manufactures false failures
**Problem:** the fine-tuned model's citation recall (61.1%) looked like a regression
against a documented baseline (78.9%).
**Root cause:** the baseline was measured over the entire 3,064-row pool (train+eval);
the model was only evaluated on the 307-row eval split. Re-measuring the same gate on
the eval split's own gold answers gave 65.6% — parity, not a regression.
**Proven fix:** thresholds must name the population they were measured on; compare
model output against gold answers on identical rows, not against a differently-scoped
baseline.
→ `resurrection.md` Q8.2

### 4. Gemma-2 has no system-role chat turn
**Problem:** Atlas's shipped chat template raises `'System role not supported'`, but
every training row starts with a system instruction (persona, retrieved context,
grounding rules).
**Root cause:** Gemma-2 lineage models only support user/model turns.
**Proven fix:** a custom template merges system content into the first user turn,
asserted byte-identical between the training data and the production prompt at build
time — not just documented, checked. A silent mismatch here degrades quality invisibly
with no error.
→ `resurrection.md` Q6.2

### 5. Output language follows the context's language, not the question's
**Problem:** a French question over Arabic-sourced retrieved context came back in
Darija.
**Root cause:** reproduced on base Atlas-Chat with **no adapter at all** — the
document's language was dominating the question's language by default. Never an
adapter defect.
**Proven fix:** a separate French system-prompt template with an explicit "answer in
French even if the context is Arabic" instruction, verified necessary (without it, a
French system prompt over Arabic context still returned Arabic).
→ `app/services/llm.py:53-57`

### 6. Silent Kaggle throughput and dedup failures
**Problem:** a headless `kaggle kernels push` once landed on a single P100 instead of
the requested dual T4 (half the throughput, no error); GPU-accelerated dedup silently
failed once, shipping 32% of a run undeduplicated.
**Root cause:** unpinned accelerator requests and an unverified GPU dedup path, both
failing without raising.
**Proven fix:** explicit `--accelerator NvidiaTeslaT4`, a fail-fast `nvidia-smi -L`
assertion before a run starts, and cross-shard dedup forced onto CPU.
→ `resurrection.md` Q5.2

### 7. Two different-looking local merge failures, two different causes
**Problem:** merging the v11 adapter to a standalone GGUF failed twice, with different
symptoms each time — a SIGSEGV mid-merge, then later a `RuntimeError` claiming the
output directory didn't exist.
**Root cause:** the first was RAM exhaustion on a 16GB-RAM machine (Unsloth's default
`maximum_memory_usage=0.85` too aggressive). The second was a transient failure in a
live Hugging Face Hub reachability check inside the merge step, which silently
no-op'd the merge (only a `warnings.warn`, no raised exception) — the real symptom
only surfaced later, at GGUF conversion.
**Proven fix:** lower `maximum_memory_usage` (a RAM-pacing knob, not a precision
control) for the first; the second was a one-off network blip, confirmed by testing
Hub connectivity directly, and resolved on retry with no code change.
→ this session, 2026-08-03

### 8. Fail-open can silently swallow a real bug, not just a real outage
**Problem:** `chat.py` wrote `pinned_fingerprint` as a raw `f"{domain}|{ui_lang}|
{message}"` string instead of the intended 64-char sha256 hash; it overflowed the
column's `VARCHAR(64)` on any real message. `history.pin_context`'s fail-open design
caught the resulting `StringDataRightTruncation`, logged it, and returned normally —
so chat kept working and the pin silently never persisted, defeating the KV-prefix
reuse it exists for, on every single turn, undetected until a live (non-SQLite)
Postgres run surfaced it.
**Root cause:** fail-open is the correct contract for a genuine external-dependency
outage, but it also hides a code defect that always throws — the two look identical
from inside the `try`. SQLite-backed unit tests couldn't have caught this either;
SQLite doesn't enforce `VARCHAR` length the way Postgres does.
**Proven fix:** call the actual hash helper (`retrieval.py`'s `_fingerprint()`)
instead of hand-building the string; add a Postgres-backed (not SQLite-only)
regression test at the hash-length level.
→ `docs/architecture/data-and-retrieval.md` §pgvector backend, `tests/test_retrieval.py`

### 9. An error handler that itself throws turns a clean failure into a crash
**Problem:** `AppError.__init__` never set `self.message` (Python's base `Exception`
has no such attribute), and all three catch sites (`app/main.py`'s global handler,
`chat.py`, `quiz.py`) read `.message` directly. A transient Ollama disconnect during
live verification became `AttributeError: 'OllamaConnectionError' object has no
attribute 'message'` instead of the intended structured `503` response.
**Root cause:** the error class was exercised in unit tests via mocks that never
triggered the real attribute-access path; only a live, unmocked failure hit it.
**Proven fix:** `AppError.__init__` now sets `self.message`; regression-tested in
`tests/test_errors.py`.
→ `docs/architecture/data-and-retrieval.md` §pgvector backend, `app/errors.py`

### 10. A fix inside a container's writable layer is not a fix
**Problem:** a live 2026-09-06 lease hand-patched a broken XTTS `torchcodec`/CUDA
link (`nvidia-cuda-runtime`/`nvidia-cuda-nvrtc` packages + loader symlinks) directly
inside the running pod. A restart minutes later (a separate, still-unexplained
incident) wiped it, and every voice session went silently unable to speak again for
the rest of the lease.
**Root cause:** the fix lived only in the container's ephemeral writable layer
(`.tts_venv`'s site-packages, `/usr/lib` symlinks) — Akash storage is ephemeral by
the SDL's own design, and container restarts are exactly the case that design does
not survive.
**Proven fix:** bake the shim into `config/Dockerfile.gpu`'s `.tts_venv` build step,
with an import assertion (`torchcodec.decoders.AudioDecoder`) right after it, so a
build that can't produce a working voice engine fails in CI, not at a lease's demo
time.
→ `docs/deploy/lease-2026-09-06-incident-log.md` #3

### 11. A self-test that instantiates a different engine than production tests nothing
**Problem:** `scripts/voice_selftest.py` hardcoded `PiperEngine()`. The engine that
actually failed on a live lease — `XttsDarijaEngine` → `_ResidentTtsWorker` →
`scripts/tts_worker_resident.py` — had never once been exercised outside a paid GPU
lease, despite the checkpoint and a compatible venv (`.tts_eval_venv`) already sitting
on the laptop that wrote the self-test.
**Root cause:** the self-test proved the *shape* of the pipeline (VAD, STT
round-trip) works, and quietly let that stand in for proving the *configured*
engine works — a difference invisible unless someone reads which class the script
constructs.
**Proven fix:** an `--engine {piper,xtts_darija,auto}` flag routes through the real
engine classes (or the real `get_tts_engine()` resolution for `auto`), so running the
self-test against the actually-deployed engine choice is one flag away instead of a
code edit. Also caught, live, on the first run with the new flag: a real (not
simulated) XTTS failure on this machine (missing `FFMPEG_SHARED_BIN` on Windows),
which correctly exercised the new fallback-to-Piper path with real synthesized audio
as output — proof the fallback mechanism (lesson learned alongside this one) actually
works against a genuine failure, not just a mocked one.
→ `docs/deploy/local-preflight.md`, `app/services/tts.py`

### 12. Tashkeel made Darija TTS worse, not better — disproven, not shipped
**Problem:** after a live lease demo, the expectation was that adding tashkeel
(Arabic diacritics) before XTTS synthesis would sharpen Darija pronunciation, the
way it does for some Arabic TTS systems. A/B eval built (bare vs. diacritized audio,
same sentences, same voice), listened to by the user. Verdict: "without tashkeel is
better, tashkeel is trash."
**Root cause:** two compounding facts, either alone would have predicted this. (1)
No Darija-specific diacritizer exists — every available tool targets Modern Standard
Arabic, and applying MSA vowelization/case-ending rules to genuine Darija vocabulary
produces grammatically-plausible-*looking* but linguistically wrong forms (confirmed
by inspection before even listening: Darija "خصك" ("you must", no MSA case system
applies) came back "خَصُّكَ" with an invented case ending). (2) The fine-tune's own
training corpus was never diacritized — measured directly against
`data/v11_merged/train.jsonl`, only ~4.9% of rows contain any diacritic at all, as
sparse incidental noise, not systematic tashkeel — so the model's learned
pronunciation has no reliable mapping for heavily-vowelized input it never trained on.
**Proven fix:** there isn't one — this is a genuine negative result, not a bug.
Confirmed via ADR 0006's own methodology (build the eval, listen, decide) rather than
assuming an NLP technique that works for other Arabic TTS systems would transfer to
this fine-tuned Darija checkpoint. No production code changed as a result; the eval
script and its finding are kept for reference so the idea isn't re-tried blind.
→ `scripts/eval_darija_tashkeel.py`

### 13. Splicing per-language TTS spans loses to mispronouncing them
**Problem:** XTTS mispronounces the Latin-script French technical terms this
tenant's fine-tune is deliberately trained to embed inside Arabic-script Darija
sentences (`build_code_switching_prompt`, enforced by `row_is_code_switched`).
The structurally obvious fix — split each sentence into same-script spans, call
`Xtts.inference()` once per span with that span's own language tag, concatenate
the audio — was built, unit-tested, and listened to. It made the voice worse,
twice, and was reverted.
**Root cause:** the premise is correct but incomplete. `inference()` really does
take exactly one language code and prefix the whole string with a single `[lang]`
token, so one call genuinely cannot pronounce both scripts — that part of the
analysis held up. What it missed is that XTTS is autoregressive and conditioned
on speaker prosody, so each span is generated as its own *complete utterance*,
with utterance-initial and utterance-final prosody of its own. Butting those
together never reads as one sentence, and short spans (a lone «ديال», a bare
«sécurité») have no context to stop cleanly on and trail off into audible
hallucinated filler — heard by the user as "aaaaaah" at every language switch.
The follow-up mitigation made it markedly worse: appending terminal punctuation
to each non-final fragment as an explicit stop cue is precisely what tells the
model to apply *sentence-final* prosody to what is only a mid-sentence clause,
so the Darija itself degraded on top of the filler.
**Proven fix:** none for splitting — it is the wrong shape of fix. Reverted to
the single-call path (kept behind `TTS_SPAN_SPLIT=1`, off by default, so the A/B
can be re-run without re-implementing it). The mispronunciation remains a real,
open limitation. The promising untried direction keeps **one** inference call and
changes the *text* instead: transliterate embedded French terms into Arabic
script («sécurité» → «سيكوريتي»), which is both how Moroccan speakers actually
pronounce these loanwords and how they are commonly written in Arabic script —
no splice, no fragment, no prosody discontinuity. Judge it the same way, by ear.
**Follow-up (2026-09-07): the transliteration route was built and evaluated
too, and not adopted.** `scripts/darija_tts_normalization.py` rewrites embedded
French into Arabic script before a single `inference()` call -- a ~230-entry
lexicon mined from the actual Latin-token frequency of assistant turns in
`data/v11_merged` (75.3% of which are mixed-script), plus a rule-based
French-orthography fallback. 25 A/B pairs were synthesized, 21 of them from real
corpus sentences stratified across all six domains and every mechanism
(acronyms, hard /g/, multi-word phrases, rule fallback, digits). Verdict: keep
the current behavior. Note this is a *product decision, not a disproof* -- the
audio was never judged broken, it simply was not clearly better than leaving the
French in Latin script. The module stays in-tree behind `TTS_TRANSLITERATE_FR=1`
(off, and imported lazily so a disabled experiment can never break worker
startup) so the A/B can be re-run without rebuilding it.

Three findings from that work are worth keeping regardless of the verdict:
1. **The checkpoint's tokenizer has only 59 Arabic characters, and lacks گ
   (U+06AF), پ (U+067E) and ڤ (U+06A4)** -- precisely the Maghrebi letters used
   to write /g/, /p/ and /v/. Spelling "protection" the standard way
   ("پروتيكسيون") pushes پ through as an unknown and silently mangles the word.
   Any future work that puts Arabic text in front of this checkpoint must check
   its characters against the vocab; `out_of_vocab()` and its tests do this
   mechanically.
2. A nasal guard used `break` where it needed `continue`, exiting the rule loop
   without advancing the cursor *and* skipping the for/else fallback -- an
   infinite loop on any word with a nasal digraph before a vowel ("animation").
   In production that hangs the TTS worker on the first such word.
3. `s` was missing from the character table, so every non-intervocalic `s` was
   silently deleted ("poste" -> "بو"). A test now asserts every ASCII letter has
   a mapping.

**Wider lesson:** the same one as #12, from the opposite direction — #12 was an
NLP technique assumed to transfer and disproven by listening; this was a
correct-on-paper architectural fix defeated by a property of the model class
(autoregressive utterance-level prosody) that no amount of reading the *API* would
have surfaced. Both were caught only by ADR 0006's rule: TTS quality is judged by
listening, never assumed.
→ `scripts/tts_worker_resident.py` (`_split_language_spans`, `_SPAN_SPLIT_ENABLED`,
  `_TRANSLITERATE_FR`), `scripts/darija_tts_normalization.py`,
  `tests/test_darija_tts_normalization.py`,
  `tests/test_tts_language_spans.py`
