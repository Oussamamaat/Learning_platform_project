# Locked-In Plan — IBLOG Adaptive Learning Tutor

**Status as of 2026-08-02.** Supersedes the prior revision of this document
in full — that revision was written before generation or fine-tuning had
actually happened; everything in it was a plan. This revision reflects what
was actually built, actually measured, and actually decided since, including
one finding serious enough to change the fine-tune's evidence bar: the first
trained adapter regressed below the untouched base model on citation
grounding (§6.2). Superseded pre-generation planning choices are removed
rather than kept as crossed-out history — that history lives in
`CHANGELOG.md` and `what_next.md` session logs; this document states current
decisions only, with just enough "why" to keep them from being re-litigated.

Detailed rationale for specific areas now lives in dedicated documents rather
than here: `FINETUNING_RATIONALE.md` (every hyperparameter, numbered),
`FINETUNE_AND_DEPLOY.md` (the training runbook and go/no-go gates),
`green_light_model.md` (the dataset acceptance checklist and its measured
status). This document is the one-level-up strategy reference: what was
decided, why, and what's still open — not the full derivation.

## 0. MVP scope (CEO-confirmed, unchanged)

The MVP is scoped to the **AI Assistant module**, three features:

1. Conversational assistant — text and audio, multilingual (French primary,
   Darija secondary but mandatory, English in scope); implemented so far for
   tenant #1's Moroccan dialect + French
2. Personalized explanations (Socratic tutoring)
3. Quiz generation from training content

All three are implemented in the current dataset and fine-tune. Audio (TTS)
is the one MVP feature with no model decision yet — see §7.

---

## 1. The decision, in one paragraph

**Tenant #1 instance: generate with Atlas-Chat-9B, fine-tune Atlas-Chat-9B.**
Not Qwen. Atlas-Chat-9B's Darija fluency comes from LoRA-256 tuning on
~450,000 Darija instructions (its own model card); this project's dataset is
a few thousand rows — under 1% of that. A dataset this size cannot teach a
model Darija from nothing; it can only reshape *how* a model that already
speaks Darija behaves. Atlas-Chat already speaks it: DarijaMMLU 58.23 (vs.
Jais-13B's 45.20), DODa BLEU 28.08 (vs. Llama-3.1-8B's 0.92 — a same-size
general model essentially cannot produce Darija at all). This reframes the
fine-tune as **behavior transfer, not language acquisition** — Socratic
pedagogy, French/Darija code-switching register, grounded citation
discipline, structured quiz output — which is what makes a dataset in the
low thousands of rows a plausible size at all (§3), and 2 epochs at r=16 the
right LoRA scale rather than an under-powered one (`FINETUNING_RATIONALE.md`
§3). This has now been executed twice (§6), not just argued for.

**This is a tenant #1 decision, not the platform base decision.** Tenant #1
mandates Darija, and Atlas-Chat is the strongest open Darija-fluent model,
so tenant #1's instance is built on it. The platform direction is a neutral
multilingual base (French primary, Darija mandatory, English in scope) with
per-tenant LoRA adapters: a new tenant's language/domain becomes an adapter
on a shared base, not a baked-in base choice. Base-model selection for the
platform remains open (§7) and must not be assumed from tenant #1's
instance.

**External validation (GemMaroc, arXiv:2505.17082):** a LoRA-tuned Gemma
3-4B trained on 5K *mixed* instructions (Darija + 20% retained English +
math/coding/science) lifted DarijaMMLU 32.8 → 42.7, reaching 47.5 with more
reasoning-dense data, **with no English regression**. Their 27B variant
matches Atlas-Chat on DarijaMMLU and beats it on Darija HellaSwag (60.5 vs.
48.4) at under 2% of the training energy. Two things carry over directly:
(a) a few thousand well-composed rows on a Darija-primed base is a real
lever, and (b) mixing non-Darija-specific reasoning data into the same
fine-tune prevents the regression a narrow behavioral dataset would
otherwise cause — why `reasoning_preservation` and `darija_preservation`
exist as dedicated components (§3) rather than the dataset being 100%
domain-specific tutoring content.

---

## 2. Model architecture

### 2.1 Base model — decided, executed

| | Decision |
|---|---|
| Data generation model | Atlas-Chat-9B, Q4_K_M GGUF, via Ollama |
| Fine-tune base model | Atlas-Chat-9B (`google/gemma-2-9b-it` lineage) — tenant #1 instance |
| Serving | vLLM multi-LoRA (§2.2) — Ollama for generation/dev only |

### 2.2 Serving architecture — vLLM multi-LoRA, implemented

**Decision:** one frozen Atlas-Chat-9B base, multiple narrow LoRA adapters,
hot-swapped per request via vLLM multi-LoRA — not one adapter trying to do
everything, and not a separate model per capability.

```
Atlas-Chat-9B (frozen base)
├── darija-tutor   ← Socratic + code-switching + quiz + grounding (THIS RUN)
└── iblog-diagram  ← Mermaid + Darija labels                       (future, not started)
```

**This multi-LoRA mechanism is exactly how new tenants are onboarded:** a
tenant's language/domain mix becomes another narrow adapter hot-swapped onto
the same frozen base — no new serving model, no code change.

**Quiz is a single adapter, not a separate service** — schema-constrained
generation gives 0/N JSON parse failures across every dataset revision, so
quiz at inference is the fine-tuned Darija voice plus a JSON schema on the
decode call, the same pattern already proven at generation time. No second
model, no rules-based generator.

**The train/serve parity problem this required solving, and how:**
Atlas-Chat's shipped chat template raises `'System role not supported'` —
Gemma-2 has no system turn, and every dataset row starts with one. Resolved
by construction, not convention: a custom `chat_template` merges `system`
into the first user turn, and the fine-tuning notebook **asserts** this
produces byte-identical output to the hand-written training-label encoder,
checked against every real row (not a sample) before being trusted. The
patched tokenizer ships inside the adapter directory, so any serving stack
loading the tokenizer from there reproduces the training format with no
serving-side code needed to remember the merge rule. Full detail:
`FINETUNING_RATIONALE.md` §1.2 and §5, `FINETUNE_AND_DEPLOY.md` §5.

**Standing invariant, unchanged since the original plan:**
`generate_training_data.PRODUCTION_SYSTEM_PROMPT_TEMPLATE` and
`llm.SYSTEM_PROMPT_TEMPLATE` must stay textually identical. Asserted at
fine-tune build time when `app/` is included in the Kaggle upload.

### 2.3 Licensing

Atlas-Chat is Gemma-2-based and ships under the **Gemma license**, not
Apache 2.0. Gemma's Terms of Use define "Model Derivatives" to explicitly
include synthetic-data-distillation outputs — any model fine-tuned on
Atlas-generated data is a Gemma Model Derivative regardless of what base
it's trained on. Commercial use is permitted; distribution carries
obligations (pass through Gemma terms, mark modifications, honor the
Prohibited Use Policy). Flagged for a real legal read before external
distribution — not legal advice, unchanged since first raised.

### 2.4 Roadmap capabilities that do not constrain the LLM choice

| Capability | Model class | Constrains this decision? |
|---|---|---|
| Darija speech-to-text | Whisper (Darija fine-tune) | No — separate model |
| Darija text-to-speech | TTS, vendor **not yet selected** (§7) | No — separate model, constrains output script only (§2.5, already resolved) |
| Explanatory video (future) | video generation model | No — not active |

`atlasia/DODa-audio-dataset` (12,743 parallel speech+text samples,
transcribed in both Latin and Arabic script plus English) remains the
identified candidate resource for TTS evaluation; no evaluation has been
run against it yet.

### 2.5 Output script — Arabic script, decided and implemented

**Decision: the model answers in Arabic script.** `messages` holds Arabic
script and is what the fine-tune trains on; `messages_arabizi`
(`generate_training_data.py:2381`) keeps a character-mapped Latin rendering
per row so an Arabizi display mode stays possible later, but nothing trains
on it and no product surface currently renders it.

This closed what was originally an open blocker: forcing Arabizi output via
a character-map transliteration pass produced *Arabic-in-Latin-letters*,
not natural Moroccan Arabizi, and could not invent French vocabulary that
wasn't in the Arabic source — measured at 76% unusable-style rows on an
early pilot. Removing the transliteration step (not improving it) fixed
this, and is why the pipeline has no LLM- or DODa-grounded transliteration
stage: it was evaluated and dropped as unnecessary once Arabic-script-only
became the target. Confirmed at 100% Arabic-script output across every
generation run since. `messages_arabizi`'s character map is a lighter-weight
fallback specifically because nothing downstream depends on its quality.

Residual, un-resolved: whether target users prefer to *read* Arabic script
or Arabizi is still an empirical product question, not something the
dataset work can settle. One session with 3-5 target users would settle it;
`messages_arabizi` makes turning on a Latin-script display mode cheap if the
answer is "Arabizi."

---

## 3. Dataset — current specification (`data/v11_merged`, 3,064 rows)

### 3.1 Composition — 11 components, not the original 6

The dataset grew from the original 6-component plan (socratic,
code_switching, grounded_refusal, quiz_generation, darija_preservation,
reasoning_preservation) to 11. The five additions were not scope creep —
each closed a **measured zero-coverage gap** found by behavioral evaluation
of the first fine-tuned adapter (§6.2), not a hypothetical:

| Component | Weight | v11_merged rows (train/eval) | Why it exists |
|---|---:|---:|---|
| socratic | 800 | 341/40 | Explain-then-question pedagogy — MVP feature #2 |
| grounded_refusal | 700 | 417/38 | Answer from context + cite; refuse when context insufficient |
| code_switching | 700 | 314/43 | French/Darija technical vocabulary carried by Darija grammar |
| quiz_generation | 400 | 462/50 | Structured JSON quiz from a content chunk — MVP feature #3 |
| darija_preservation | 200 | 192/22 | Everyday Darija fluency; regularization, not acquisition — prevents catastrophic forgetting of dialect |
| reasoning_preservation | 200 | 273/35 | Math/step-by-step reasoning, mostly non-Darija; same regularization role, for general capability |
| structured_explanation | 200 | 298/27 | Multi-step procedures with enforced Markdown structure — audit found only 0.4% of prose completions had any structure at all |
| learner_adaptation | 150 | 196/20 | Re-explain on "I didn't understand" with a simplified, job-context example — cahier §3.1.7, measured 0.9% coverage before this component existed |
| general_knowledge_disclosed | 150 | 116/19 | Genuine general knowledge, answered with mandatory non-company-sourced disclosure — found 0/4 correct on this distinction pre-fix |
| no_context_refusal | 150 | 102/8 | Empty-context prompts must refuse — found 0/4 refused pre-fix |
| injection_resistance | 100 | 46/5 | Resist a prompt-injection override while staying a helpful, on-topic tutor — found 3/4 succeeded (i.e. failed to resist) pre-fix |
| **Total** | | **3,064** | 2,757 train / 307 eval, ~90/10 |

`darija_preservation`'s prompt (`build_darija_preservation_prompt`) used to
take no arguments and return a static string, capping its achievable yield
at 129/200 (64%) across two generation attempts — a structural ceiling, not
stochastic underperformance. Fixed by giving it a rotating `topic` argument;
this pool has 214 rows there, above target for the first time.

Domain split within grounded components stays 80/20 client/generalization:
3 real client domains (industrial, securite, blockchain) plus 6 synthetic
generalization domains (medical, legal, automotive, RH, logistique,
hôtellerie), teaching the behavior is domain-agnostic rather than bound to
the current 3 verticals. Source routing (`pick_source_doc()`) varies by
component: `grounded_refusal` draws 50/50 Arabic/French-script documents;
`socratic`/`code_switching` draw French-script only (French vocabulary
density is the point); `quiz_generation` draws Arabic-script only.

### 3.2 Why row count stopped being the target metric

The original plan sized the dataset by row count (3,000, derived from
corpus ceiling, LoRA capacity, and review capacity — reasoning that still
holds and is why the total didn't balloon alongside the component count).
That framing turned out to be the wrong lever partway through: two
iteration cycles were spent chasing **generation-time knobs** (`multi_turn_pct`
values in `COMPONENT_CONFIG`, meant only as the probability the generator
*asks* for a multi-turn sample) as if they were **acceptance thresholds**,
when the actual gate (`green_light_model.md` RF8) is ≥40% (design ~50%)
multi-turn share over socratic+code_switching *combined* — a target the
pool already passed before the last two Kaggle runs spent GPU hours on it.
See `verify-the-gate-before-optimizing` in project memory. **The acceptance
criteria live in `green_light_model.md` §4.1, not in `COMPONENT_CONFIG`** —
this is now the standing rule before spending GPU time on any dataset
metric.

### 3.3 Gate architecture — the recurring failure mode this project kept hitting

At least ten separate times across this project's history, a prompt asked
the model for a behavior and nothing verified compliance — component
weights and prompt text encode *intent*, not *enforcement*. Every fix
followed the same shape: measure the actual rate, add a budget-capped
reject-and-retry gate (never unbounded — a gate that never disables itself
can starve a whole component if the model structurally can't comply), and
add a regression test. Current enforced gates, by name:

- **Structural:** `validate_chatml` (script/CJK), `row_lacks_structure`
  (`structured_explanation`), `row_has_repeated_turn`, `turn_count_mismatch`
  (multi-turn shape actually matches what was asked)
- **Grounding:** `row_cites` / `CITATION_ENFORCED_COMPONENTS`,
  `row_has_ungrounded_reference` (fabrication, hard gate, never disabled),
  `row_has_ungrounded_number` / `NUMERIC_GROUNDED_COMPONENTS`,
  `row_refusal_cites_something` (a refusal citing something real is still
  wrong — refusals have nothing to cite by definition)
- **Register:** `row_is_code_switched` (French per row + Darija per turn),
  `row_is_grounded_darija`, `french_term_count` on `grounded_refusal`
  answerable rows
- **Behavioral:** `row_is_socratic` (answer-dump / question-only rejection),
  `row_discloses_general_knowledge`, `row_is_learner_adaptation`,
  `row_has_injection_marker`

Cross-shard deduplication (`merge_shards.py`, forced CPU — GPU dedup
silently failed on Kaggle/CUDA once and shipped 32% of a run undeduped
before that was caught) runs once across all GPU-shard outputs before the
final train/eval split, closing a leakage risk a per-shard dedup can't see.

---

## 4. Generation infrastructure

### 4.1 Platform and the GPU-pinning lesson

Kaggle, dual T4 (two `CUDA_VISIBLE_DEVICES`-pinned Ollama worker processes),
Q4_K_M GGUF. vLLM was evaluated for *generation* and set aside — no AWQ/GPTQ
quant exists for Atlas-Chat, and T4's PCIe-only interconnect makes
tensor-parallel bf16 serving not worth the integration risk at this scale.
This is a separate decision from vLLM as the *fine-tuned model's* serving
target (§2.2), which stands.

**Operational lesson, now standing practice:** a headless
`kaggle kernels push` has no equivalent of an interactive `!nvidia-smi`
check — one run silently landed on a single P100 instead of the requested
T4, and the dual-GPU worker split pointed its second worker at a
`CUDA_VISIBLE_DEVICES` index that didn't exist, so half the intended
throughput silently vanished with no error. Fixed by (a) always pushing with
`--accelerator NvidiaTeslaT4` explicitly rather than leaving it to Kaggle's
account default, and (b) a fail-fast `nvidia-smi -L` GPU-count assertion
early in the driver script that exits loudly instead of running degraded.

### 4.2 Fine-tuning is single-GPU by design — no pinning ambiguity there

Unlike generation, the fine-tuning notebook pins exactly one GPU
(`CUDA_VISIBLE_DEVICES=0`, set before any `torch` import) regardless of
accelerator — Unsloth's OSS path is single-GPU, and an unpinned 2×T4 box
gets silently wrapped in `nn.DataParallel` by HF `Trainer`, which breaks
Unsloth's patched kernels. This makes the fine-tuning step immune to the
§4.1 failure mode by construction, not by discipline.

---

## 5. Fine-tuning configuration — decided and executed

Full derivation lives in `FINETUNING_RATIONALE.md`; this is the summary a
strategy read needs.

| Setting | Value | One-line why |
|---|---|---|
| `r` / `lora_alpha` | 16 / 16 | ~54M trainable params (0.585% of the model) against ~300-450K supervised tokens/epoch — r=32 would double capacity without doubling signal, risking phrasing-level overfit rather than behavior transfer |
| `lora_dropout` | 0 | Required by Unsloth's fused/fast LoRA kernel path |
| target modules | all 7 linear (`q,k,v,o,gate,up,down`) | No `embed_tokens`/`lm_head` — not teaching new tokens |
| `max_seq_length` | 4096 | Clears measured p99 comfortably, inside Gemma-2's native window, no RoPE scaling; overlong rows are dropped, never truncated |
| batch / accum | 1 × 16 | Zero padding waste on this dataset's length variance; T4 is dequant-bound, not batch-throughput-bound |
| epochs | 2 | Checkpointed per epoch specifically so an overfitting epoch 2 doesn't cost the run — epoch 1 stays on disk |
| lr / schedule | 2e-4, cosine, 3% warmup | Unsloth's own validated pairing at this rank |
| precision | fp16 (T4/P100 have no bf16) | Auto-detected; smoke-tests a finite first-batch loss before committing, since fp16 + Gemma-2 softcapping is where silent overflow would appear |
| `optim` | `adamw_8bit` | Halves optimizer-state VRAM, no measured quality cost |

Go/no-go gates asserted by the training run itself (not just this document's
say-so): post-training eval loss must beat the base-model baseline (LoRA
init is B=0, so pre-training eval loss *is* base Atlas-Chat's own loss —
no improvement means the config or masking is broken, not that the fine-
tune "underperformed"); per-component eval loss flags anything past 1.6×
the overall mean; a 5-generation smoke test checks Arabic-script share,
quiz JSON validity, clean stopping (the `<end_of_turn>` stop-token patch —
served unpatched, the model would emit it, nothing would recognize it as a
stop condition, and generation would continue by role-playing the user's
next turn), and CJK contamination. Full checklist: `FINETUNE_AND_DEPLOY.md`
§4.

---

## 6. Fine-tune run history

### 6.1 First run — 2026-07-30, `dataset_export_top_up` (2,655 rows, 6 components)

Executed with the configuration in §5. `eval_loss` 1.1651 → 0.85062 (best
at step 148/298), `train_loss` 0.70677. Report: `TRAINING_REPORT.txt`. Loss
curve looked healthy — this run would have read as shippable on loss alone.

### 6.2 The finding that reopened the dataset — citation fabrication below the base model's own floor

A controlled post-training comparison (same Ollama stack, same system
prompt, a context containing no legal reference) found: **base Atlas-Chat
correctly declines to cite** («ماكاينش شي قانون محدد فالمعلومات اللي
عطاونا» — "there is no specific law in the information we were given"),
while **the fine-tuned adapter confidently fabricates one**
(«حسب القانون 27-04...»), 2/2 each way.

Root cause was distributional, not a handful of bad rows: the training
pool's legal Q&A rows almost always cited *something* (source contexts
usually contained a reference, and the only enforced citation gate at the
time was on `grounded_refusal`), so the model learned "legal question →
emit a citation" and filled the slot from Atlas-Chat's own pretrained
Moroccan-law knowledge whenever the retrieved context had none. Deleting
the handful of flagged rows would not have fixed this — it would have left
the distribution that taught the behavior fully intact.

**Consequence: this is now a standing evaluation requirement, not a one-off
finding.** Any new adapter must be compared against **base Atlas-Chat**, not
only against the previous adapter — the base is the honest floor on
grounding, and this run proved a fine-tune can land below it. See
`finetune-degrades-citation-grounding` in project memory.

### 6.3 What changed in response, and the second run

This finding, combined with a separate behavioral audit that found zero
training coverage for empty-context fabrication, prompt-injection
compliance, and the general-knowledge-vs-ungroundable distinction, drove
the five new components in §3.1 and the gate-enforcement work in §3.3.
Measured directly against the pool the first fine-tune actually trained on
vs. the current one:

| | first fine-tune's pool | `v11_merged` (current) |
|---|---:|---:|
| Rows | 2,655 | 3,064 |
| Fabricated references | **653 (24.6%)** | **0** |
| Citation-free context, answered with no citation | 44.7% of pool | 69.6% of pool |
| Citation recall on citable rows | 60.4% | 78.9% |
| `grounded_refusal` zero-French rate | 77.1% | 0.0% |
| Multi-turn share (socratic+code_switching) | 41.3% | 50.0% |

`data/v11_merged` passed every check in `green_light_model.md` §4.1 as of
2026-08-02 (all eight automated checks, with margin). **A second fine-tune
run against `v11_merged` completed 2026-08-02** (`kaggle_finetune_v11.ipynb`,
kernel `darija-tutor-finetune-v11-run`), same configuration as §5:
`eval_loss` 1.1823 → 0.8407, all 11 components within the 1.6× underfit
threshold. The GGUF export did not make it into the deliverable zip (Unsloth
built it successfully but the notebook's own copy-into-`/kaggle/working`
step found nothing — recoverable with a ~15 min local rebuild); the LoRA
adapter itself is complete and confirmed intact.

**§6.2's question — does it clear the base-model grounding floor — is
answered: yes, and it does better than "not regressed."** A same-process
base-vs-adapter comparison (`darija-tutor-grounding-check-v11` kernel,
2026-08-02) ran two prompts: a fabrication-bait prompt (real context with no
citable law) and a should-cite prompt (real context with a citable article).
Base Atlas-Chat fabricated a citation on the bait prompt (a different
instance of the same failure the first adapter had, confirming base is not
a universally reliable floor either) and, on the should-cite prompt,
paraphrased without naming the article. **The v11 adapter declined instead
of fabricating on the bait prompt, and explicitly named the article on the
should-cite prompt — better than base on both.** Full detail and the one
rough edge found (a refusal that claimed an incorrect topic-scope
restriction — safe, but would look like a bug in a live demo) recorded in
project memory `v11-adapter-fixed-citation-fabrication`.

---

## 7. What's still genuinely open

- **TTS vendor/model selection.** Unchanged since first flagged — still
  MVP-critical per §0, still no evaluation run. `DarijaTTS-v0.1-500M`,
  a SpeechT5-darija HF Space, and `atlasia/DODa-audio-dataset` remain
  candidates, not a decision. The single largest unresolved MVP item at
  this point in the project.
- **Refusal-template scope mismatch.** The v11 adapter's fabrication-bait
  refusal (§6.3) claimed it could only help with "workplace safety and
  health" topics on a question that was actually in-scope (blockchain/crypto
  compliance). Not a grounding defect — it declined rather than fabricating
  — but would read as a bug in a live demo. Worth checking refusal-template
  variety for `general_knowledge_disclosed`/`no_context_refusal` before
  §4.2's live demo script runs.
- **`grounded_refusal` French density — is it even wanted?** The component
  passes its enforced gate, but whether French vocabulary belongs in a
  refusal that's quoting Arabic legal text is an open product question, not
  a data-quality one. Needs a native-speaker read of a handful of rows, not
  more generation.
- **Arabic-vs-Arabizi user preference (§2.5).** Not resolved by any measurement
  this project can produce internally — needs a small user session.
- **Diagram generation adapter.** Roadmapped (§2.2), not started.
  `reasoning_preservation` deliberately keeps this option open without
  committing to it.
- **Video generation.** Explicitly a future hope, no architectural
  commitment.
- **Neutral multilingual base-model selection (platform).** Which base the
  platform's French-primary/multilingual, domain-agnostic direction sits on
  before tenant #2 — not yet evaluated. Tenant #1's Atlas choice (§1) is an
  instance decision, not the platform base decision.

---

## 8. Traceable evidence

- Atlas-Chat-9B model card: DarijaMMLU 58.23 vs. Jais-13B 45.20; DODa BLEU
  28.08 vs. Llama-3.1-8B 0.92; LoRA-256 on ~450K Darija instructions.
- GemMaroc (arXiv:2505.17082): 5K mixed instructions, DarijaMMLU 32.8→47.5,
  no English regression; 27B variant matches/beats Atlas-Chat at <2% energy.
- Gemma Terms of Use: "Model Derivatives" definition includes synthetic-data
  distillation outputs.
- `TRAINING_REPORT.txt` (2026-07-30 run): eval_loss 1.1651→0.85062,
  per-component breakdown, 0 dropped-overlong rows, smoke-test quiz JSON
  parse failure (inference-time, not dataset — 0/N invalid in the data
  itself both then and now).
- Controlled base-vs-adapter grounding comparison (2026-07-31, §6.2): the
  finding that reopened the dataset.
- `green_light_model.md` §4.1d (2026-08-02): `v11_merged` measured results,
  all eight automated checks passing.
- `FINETUNING_RATIONALE.md`, `FINETUNE_AND_DEPLOY.md`: full hyperparameter
  derivation and training runbook, both executed as written.
- CEO-confirmed MVP scope (§0): AI Assistant module — conversational
  assistant (text + audio, dialect), personalized explanations, quiz
  generation.
- `atlasia/DODa-audio-dataset`: 12,743 parallel speech/text samples in both
  scripts plus English — candidate TTS resource, not yet evaluated.
