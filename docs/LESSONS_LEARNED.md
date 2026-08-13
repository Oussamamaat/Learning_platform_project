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
