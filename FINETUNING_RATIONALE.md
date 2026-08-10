# Fine-Tuning Rationale — Tenant #1 Instance (Atlas-Chat-9B LoRA)

Companion to `unsloth_finetune_atlas.ipynb` and `FINETUNE_AND_DEPLOY.md`. That
pair covers *what to run and watch*; this document covers *why every number
in it is that number*. Every figure below is either measured directly against
`dataset_export_top_up/` or `MBZUAI-Paris/Atlas-Chat-9B`'s published config,
not a template default — where a number is estimated rather than measured,
that is stated explicitly.

---

## 1. Executive summary & architecture overview

### 1.1 Why Atlas-Chat-9B (tenant #1's instance base), and why LoRA at all

Atlas-Chat-9B was chosen as the **tenant #1 instance** base: tenant #1
mandates Darija, and Atlas is the strongest open Darija-fluent model.
`LOCKEDIN_PLAN.md` §1 made this decision before any data was generated, and
it constrains everything downstream for that instance: **generate with
Atlas-Chat-9B, fine-tune Atlas-Chat-9B.** The original plan generated data
with Atlas and intended to fine-tune Qwen2.5-7B on the theory that Qwen is
the stronger general model.
That combination does not work, for one measurable reason: Atlas-Chat-9B's
Darija fluency comes from LoRA-256 tuning on **~450,000 Darija instructions**
(its own model card). Our dataset — 2,655 rows before cleanup — is under 1%
of that. A dataset this size cannot teach a model Darija from nothing; it can
only reshape *how* a model that already speaks Darija behaves. Atlas-Chat
already speaks it: DarijaMMLU 58.23 (vs. Jais-13B's 45.20), DODa BLEU 28.08
(vs. Llama-3.1-8B's 0.92 on the same benchmark — a same-size general model
essentially cannot produce Darija at all). Qwen has no Darija pretraining to
transfer style onto.

This reframes the entire fine-tune: **the dataset is not teaching language
acquisition, it is teaching behavior transfer** — Socratic pedagogy,
French/Darija code-switching register, grounded citation discipline,
structured quiz output. That reframing is what makes 2,655 rows a plausible
dataset size at all (§2) and what makes 2 epochs at r=16 the right LoRA scale
rather than an under-powered one (§3).

**External validation for "a few thousand rows genuinely moves Darija
behavior":** GemMaroc (arXiv:2505.17082) LoRA-tuned a *Gemma 3-4B* — a
smaller, less Darija-primed base than ours — on 5K mixed instructions (Darija
+ 20% retained English + math/coding/science) and lifted DarijaMMLU 32.8 →
42.7, reaching 47.5 with more reasoning-dense data, **with no English
regression**. Their 27B variant matches Atlas-Chat on DarijaMMLU and beats it
on Darija HellaSwag (60.5 vs. 48.4) at under 2% of the training energy. Two
things carry over directly to this run: (a) a few thousand well-composed rows
on a Darija-primed base is a real lever, not noise, and (b) **mixing in
non-Darija-specific reasoning data during the same fine-tune prevents the
regression a narrow behavioral dataset would otherwise cause** — which is
exactly why `reasoning_preservation` and `darija_preservation` exist as
dedicated components (§2) rather than the dataset being 100% domain-specific
tutoring content.

**Why LoRA and not full fine-tuning:** not really a choice at this scale.
Full fine-tuning a 9.24B-parameter model needs the optimizer state for every
parameter (Adam: ~2 extra copies per param) — infeasible on a single 16 GB
T4. LoRA freezes the base and trains low-rank adapters on top, which is also
what makes the serving architecture in §1.2 possible in the first place.

**Scope note — this is the tenant #1 instance decision, not the platform base
decision.** Atlas-Chat-9B is tenant #1's instance base because tenant #1
mandates Darija; it is not a neutral platform-wide choice. The platform
direction is a neutral multilingual base model with per-tenant LoRA adapters,
so onboarding tenant #2 means a per-tenant language/domain adapter on that
shared base, not a new base-model decision (§1.2).

### 1.2 Serving architecture — why this shapes the training format

`LOCKEDIN_PLAN.md` §2.3 locks the production serving model as **one frozen
Atlas-Chat-9B base, multiple narrow LoRA adapters, hot-swapped per request via
vLLM multi-LoRA** — not one adapter trying to do everything, and not a
separate model per capability. That mechanism is exactly how a future
tenant's language/domain adapter gets added — a new narrow LoRA beside this
one on the frozen base. This run trains a single adapter carrying all
six dataset components (Socratic tutoring, code-switching, grounded refusal,
quiz generation, and the two preservation components), following §2.3's
explicit resolution: *"Quiz is a single adapter, not a separate service...
Do not build a second model."* Splitting quiz into its own adapter remains an
option later, gated on the per-component eval loss this run produces (§4.2)
— not a default.

This architectural choice has a direct training-format consequence covered in
full in §5: because the adapter must reproduce exactly what production sends
at inference time, and Atlas-Chat's stock chat template cannot even represent
a system message, the training format had to be constructed rather than
assumed. That construction — and proving it matches what gets served — is
the single most safety-critical part of the notebook, and is why it is
verified against all 2,655 real rows before being trusted (§5.1).

### 1.3 Licensing note (carried from `LOCKEDIN_PLAN.md` §2.2)

Atlas-Chat is Gemma-2-based and ships under the **Gemma license**, not Apache
2.0 as an earlier draft of the plan assumed. Gemma's Terms of Use define
"Model Derivatives" to explicitly include methods based on the generation of
synthetic data outputs by Gemma for training that model — meaning any model
fine-tuned on this Atlas-generated dataset is a Gemma Model Derivative
regardless of what base it's trained on. Commercial use is permitted;
distribution carries obligations (pass through the Gemma terms, mark
modifications, honor the Prohibited Use Policy). This is flagged for a real
legal read before external distribution — not legal advice, and unchanged by
anything in this fine-tuning run.

---

## 2. Dataset composition analysis

### 2.1 Why 2,655 rows (and not 3,000, and not 7,500)

The original plan sized the dataset at 7,500 rows under the Qwen-base
assumption, where the dataset had to carry language acquisition.
`LOCKEDIN_PLAN.md` §3.1 re-derives the number under Atlas-as-base from four
independent constraints, all still binding at fine-tune time:

1. **Corpus ceiling.** The real client corpus is 36 documents (~1,625
   chars/doc). Past roughly 15-20 rows generated per document, generation
   stops adding information and starts paraphrasing — a ceiling on
   `grounded_refusal` specifically, and part of why its 700-row target (not
   2,000, the original figure) was already corpus-capacity-matched before
   generation started.
2. **LoRA capacity.** Even at the reduced r=16 used here (§3.1), a
   multi-thousand-row dataset is comfortably past minimum signal for that
   capacity (quantified in §3.1's params/tokens ratio); scaling row count
   further does not scale value proportionally.
3. **Review capacity.** A 15% manual-review plan is realistic at 3,000 rows
   (450 rows) and was not realistic at 7,500 (1,125 rows).
4. **GemMaroc precedent (§1.1).** 5K mixed instructions moved a *smaller*
   base model's Darija proficiency meaningfully. A comparable row count on a
   9B model with stronger Darija priors than Gemma-3-4B is at least as good a
   bet as blindly scaling further.

The **3,000-row design target** (`LOCKEDIN_PLAN.md` §3.1 table) was not fully
met by generation — two Kaggle runs (a full run + one targeted top-up)
yielded **2,655 rows** (89% of target) before this fine-tune's own cleanup
step removed 9 more (§2.4), landing at **2,646** actually trained/evaluated
on. This is analyzed in full in `dataset_evaluation.md`; the summary relevant
here:

| component | rows (top-up export) | design target | % of target |
|---|---:|---:|---:|
| socratic | 711 | 800 | 89% |
| grounded_refusal | 686 | 700 | 98% |
| code_switching | 567 | 700 | 81% |
| quiz_generation | 374 | 400 | 94% |
| reasoning_preservation | 188 | 200 | 94% |
| darija_preservation | 129 | 200 | **64%** |
| **total** | **2,655** | **3,000** | **89%** |

Every component landed within 6-36% of its design target — none catastrophic,
none at zero. `darija_preservation` is the outlier and is addressed directly
in §2.2, because "why is the smallest component small" is a fair question to
ask before trusting an adapter trained on it.

**Why proceed at 89% of target rather than closing the gap first:** the
shortfall was already investigated by attempting exactly that. A second
top-up run specifically targeting the five under-target components (860 rows
computed as needed) delivered only 515 net new rows — most components closed
most of their gap, but `darija_preservation` filled only 37% of its own
deficit on a *second* independent attempt, which is the evidence that its
shortfall is structural (§2.2), not stochastic, and that a third top-up would
plateau the same way. Closing it requires a code change (a varying prompt
seed), which is out of scope for a data-collection pass and explicitly
deferred to a future targeted regeneration if the per-component eval loss in
`FINETUNE_AND_DEPLOY.md` §3 shows it mattered.

### 2.2 Why 116-129 rows is structurally sufficient for `darija_preservation`

This deserves a real argument, not just "it's the smallest by design," because
129 rows failing to reach a 200-row target twice is a legitimate reason to
worry about adapter quality. Two things separate this case from an
under-sampled skill:

**First, the root cause is prompt entropy, not corpus scarcity — and it is
provably not about lack of *available* material.** Every other
document-grounded component (`socratic`, `code_switching`, `grounded_refusal`,
`quiz_generation`) draws a different real corpus chunk on every call, which
gives each generation attempt genuinely different content to work from.
`build_darija_preservation_prompt()` (`generate_training_data.py:808`) takes
**no arguments** and returns the exact same static string on every single
call — no domain, no source document, no varying terms. Sampling repeatedly
from a zero-variation prompt produces a much higher rate of literal and
near-duplicate output, which the pipeline's own in-generation exact-match
dedup (`seen_texts`, `generate_training_data.py:2102`) and the final
cosine>0.95 semantic dedup both correctly strip. The component's ceiling is
therefore an artifact of prompt design burning attempts on duplicates, not a
signal that only 129 rows' worth of "everyday Darija fluency" content exists
to draw on — there is no reason to think the underlying skill space is
actually that narrow.

**Second, and more load-bearing: the component's job is regularization, not
acquisition.** `LOCKEDIN_PLAN.md` §3.1 states its purpose directly —
*"everyday Darija fluency, prevents catastrophic forgetting of dialect"* —
and sized it at 200 rows against `socratic`'s 800 and `grounded_refusal`'s
700 specifically because it is the smallest-weighted component **by design**
(`weight: 200` in `generate_training_data.py:58`, a quarter of `socratic`'s
weight), not an oversight. This is the same principle GemMaroc's mixed-data
recipe demonstrates from the other direction (§1.1): a narrow behavioral
fine-tune risks eroding capabilities the base model already has, and a small
proportion of general-purpose data in the mix is what prevents that erosion —
it does not need to be large to do its job, because its job is not to install
a new capability but to keep gradient pressure toward the existing one
present throughout training. Atlas-Chat's Darija fluency was itself built on
~450,000 rows (§1.1); 118 rows (the post-parrot-filter train count, §2.4) is
not remotely trying to compete with that — it is 118 reminders, distributed
across every training epoch, not to drift away from it.

Contrast this directly with `socratic` (638 train rows post-filter): that
component IS the primary product behavior — MVP feature #2, personalized
explanations — and needs enough coverage to generalize the Socratic-questioning
*pattern* across domains and phrasings, which is a genuine acquisition task.
The same row count that would be thin for `socratic` is proportionate for
`darija_preservation` because the two components are not doing the same kind
of work.

**What this argument does not claim:** that 129 rows is *optimal*, or that
the shortfall should be ignored going forward. It claims the shortfall is
low-risk for *this* run specifically, and gives a concrete, cheap
verification path rather than asking for trust: `FINETUNE_AND_DEPLOY.md` §3's
per-component eval loss table has `darija_preservation` flagged if it exceeds
1.6× the overall mean loss/token. If it is not flagged, the argument above is
confirmed by the training run itself. If it is flagged, the fix is already
scoped and cheap — add a rotating topic seed to
`build_darija_preservation_prompt()` and run a targeted regeneration for that
one component, not a full re-run.

### 2.3 Known dataset flags: deliberate design trade-offs, not training defects

Two numbers in `dataset_evaluation.md` cross the checklist's kill lines. Both
were root-caused by direct measurement against the actual export, and both
resolve to composition properties of the source corpus and generation
behavior — not corruption, mislabeling, or a broken pipeline. Restated here
because a decision-maker evaluating whether to trust this fine-tune needs to
know these are *known and understood*, not *undiscovered*.

#### `grounded_refusal` zero-French rate: 81% (kill line: >80%)

The naive read — "the French/Arabic source split isn't working" — is
**disproven by direct measurement**:

| source document script | rows | rows with French in the answer | rate |
|---|---:|---:|---:|
| Arabic | 277 | 37 | 13% |
| Latin (French) | 334 | 77 | **23%** |
| overall | 611 | 114 | 19% |

`pick_source_doc()`'s 50/50 split **is landing** — 334/611 = 55% of rows
drew a French-script source document, slightly *over* the intended 50%, and
per-domain the three domains with both scripts available split close to
evenly. The actual defect is one step downstream: **even given a French
source document in front of it, the model carries French vocabulary into its
answer only 23% of the time.** `FRENCH_SOURCE_CITATION_RULE` instructs it to,
but that is a soft prompt instruction with no gate behind it —
`row_is_grounded_darija()` passes at 100% and never checks French presence at
all, unlike `row_is_code_switched()`, which does gate `socratic`/
`code_switching` and passes at 100% there. The pipeline's own code comment
(`generate_training_data.py:519-528`) records that instructing the model to
supply French was already tried once before and "barely moved the rate" — so
this is a known, previously-encountered model behavior, not a new anomaly.

**Why this is not being fixed before this fine-tune:** the fix requires a
hard reject-and-retry French gate (reject-budget risk) and regenerating all
~700 `grounded_refusal` rows, not a code tweak. More importantly, a prior
top-up run's own math shows a partial fix cannot move the aggregate number
meaningfully — adding rows to the *existing* 611-row population at even 100%
French success would still leave the bulk of the component in its current
state, so a full regeneration is the only real fix, and that is gated on an
open product question that has nothing to do with data quality: **is French
density even desired in a refusal that is quoting Arabic legal text?** The
`>80%` kill threshold in `QUALITY_FLAGS.md` was written as an expectation
("should be well under 100% — roughly half") when the *only* measured
baseline was 100% zero-French, pre-dating the source split — it was never a
validated product requirement. Settling it needs a native-speaker read of a
handful of the 611 existing rows, not more GPU time.

#### Citation recall: 63% (informational floor: 70%) — reclassified as expected, not regressed

| source document script | citable rows | cited | recall |
|---|---:|---:|---:|
| Arabic | 277 of 277 (**100% citable**) | 179 | 65% |
| Latin (French) | 47 of 334 (**14% citable**) | 25 | 53% |

This is a **composition effect of the 50/50 source split**, not a quality
regression. French-script corpus documents rarely contain extractable
numbered legal references — only 14% of rows drawing a French source even
have a citation available to recall, versus 100% of rows drawing an Arabic
source. The 72% baseline this is compared against in `QUALITY_FLAGS.md` was
measured on 45 rows generated *before* the 50/50 split existed — i.e.,
Arabic-source only. Comparing 63% (mixed-source) against 72%
(Arabic-source-only) compares two different row populations by construction.
Measured *within* each population, recall is healthy — 65% Arabic, 53%
French — and trading some aggregate citation recall for French-register
diversity was the explicit, documented intent of introducing the split in the
first place (`pick_source_doc()`'s own docstring). **No action item follows
from this number.**

### 2.4 Cleanup applied at fine-tune time: 9-row parrot filter

While verifying the training-format masking against the real dataset (needed
regardless, per §5.2), 9 rows (0.34% of 2,655, all in `code_switching`) were
found where the assistant's answer repeats ≥50 characters of the user's own
turn verbatim before responding — `QUALITY_FLAGS.md` §7's "few-shot
parroting" failure mode, not previously caught because dedup operates on
*whole-row* similarity and an echo-then-answer row is not a near-duplicate of
anything else in the dataset. `unsloth_finetune_atlas.ipynb` §2 drops these
before training (asserting the rate stays under 5%, in case a future
generation run regresses this into something systemic rather than 9 isolated
rows). Post-filter: **train 2,389 → 2,380, eval 266 → 266** (all 9 happened to
land in train).

---

## 3. Hyperparameter rationale — number by number

### 3.1 Rank (r) = 16, Alpha = 16

**Why r=16 and not the r=32 the original Qwen-era plan specified:**
`LOCKEDIN_PLAN.md` §6 flags this explicitly as needing re-derivation for
Atlas-Chat at this row count, not carried over by default. The concrete
argument is a capacity-vs-signal ratio, computed exactly against Atlas-Chat's
real architecture (`hidden_size=3584`, `intermediate_size=14336`,
`num_attention_heads=16`, `num_key_value_heads=8`, `head_dim=256`,
`num_hidden_layers=42`):

LoRA adds two low-rank matrices per targeted linear layer; trainable params
per layer = `r × (in_features + out_features)`, summed across all 7 targeted
projections (`q,k,v,o,gate,up,down`) and all 42 layers:

| module | in | out | params/layer at r=16 |
|---|---:|---:|---:|
| q_proj | 3,584 | 4,096 (16×256) | 122,880 |
| k_proj | 3,584 | 2,048 (8×256) | 90,112 |
| v_proj | 3,584 | 2,048 | 90,112 |
| o_proj | 4,096 | 3,584 | 122,880 |
| gate_proj | 3,584 | 14,336 | 286,720 |
| up_proj | 3,584 | 14,336 | 286,720 |
| down_proj | 14,336 | 3,584 | 286,720 |
| **per layer** | | | **1,286,144** |

× 42 layers = **54,018,048 trainable parameters** — **0.585%** of
Atlas-Chat-9B's ~9.24B total. `embed_tokens`/`lm_head` are deliberately not
targeted: this fine-tune is not teaching new tokens, and Gemma-2's 256,000-row
tied embedding matrix (917.5M parameters on its own — 17× the entire LoRA
adapter) would dominate gradient updates and risk exactly the kind of
generic-capability drift the preservation components (§2.2) exist to guard
against.

Against that capacity: the dataset supplies roughly 300,000-450,000
supervised (assistant-only) tokens per epoch — measured precisely: 1,058,256
supervised characters across 2,380 post-filter train rows (18.0% of total row
characters), converted at an estimated 2.2-4.0 characters/token range typical
for a multilingual SentencePiece tokenizer on mixed Arabic-script/French
content (the notebook's §5 prints the exact tokenized count; this range
brackets it before that run). At r=32 (~108M params, the original plan's
figure), the tokens-to-parameters ratio would be roughly halved again from an
already-thin ~0.008 supervised tokens per trainable parameter per epoch at
r=16. Doubling capacity without doubling signal does not make training
better — LoRA at r=32 on this row count risks the adapter over-fitting to
specific phrasings rather than generalizing the *behavior* the dataset is
trying to transfer, which is the entire point of choosing a Darija-primed
base in the first place (§1.1). r=16 keeps meaningful headroom over the
adapter's rank while not chasing capacity the data cannot back up; it was
selected over an even smaller r=8 because six distinct behavioral modes
(Socratic questioning, code-switching register, citation discipline, JSON
quiz structure, and two preservation styles) sharing one adapter benefit from
some spare capacity to represent that many modes without excessive
interference — but not the 2× capacity r=32 would add for signal that isn't
there.

**Alpha=16** sets the LoRA scaling factor (`alpha/r`) to exactly 1.0 — no
amplification, no dampening of the adapter's contribution relative to its
rank. This is Unsloth's own validated default pairing for r=16 with a 2e-4
learning rate (§3.2); scaling above 1.0 effectively raises the learning rate
on the adapter weights specifically, which is a second lever to pull only if
1.0 under-fits — not a starting assumption to second-guess without evidence.

**Dropout = 0** is not a quality trade-off; it is a hard requirement of
Unsloth's fused/fast LoRA kernel path. Any nonzero value falls back to a
slower, unfused implementation.

### 3.2 Learning rate: 2e-4, cosine schedule, 3% warmup

2e-4 is the standard, empirically-validated starting point for LoRA
fine-tuning at this rank across the Unsloth ecosystem (their own maintained
Gemma-2-9B Kaggle reference notebook uses the same value at a comparable
rank), and there is no dataset-specific reason found in this project to
deviate from it — unlike `r`, where `LOCKEDIN_PLAN.md` gave an explicit reason
to move off the inherited Qwen-era default, no such signal exists for the
learning rate.

**Cosine decay** rather than linear or constant: with only 298 total optimizer
steps (§3.3), the run needs to reach a useful learning rate quickly and then
anneal smoothly rather than dropping off a cliff or staying elevated into the
final steps where it would perturb an already-converging adapter. **3% warmup**
(≈9 steps) is enough to avoid a destabilizing first gradient update — Adam's
second-moment estimates are unreliable in the first few steps — without
wasting meaningful budget on ramp-up in a run this short (a 10% warmup on a
298-step run would spend nearly 30 steps just ramping, a much larger fraction
of the total than on a run with thousands of steps).

### 3.3 Batch size 1, gradient accumulation 16 — the Kaggle T4 trade-off

Effective batch size 16 (`1 × 16`), reached through pure gradient
accumulation rather than a larger micro-batch, for three reasons specific to
this hardware and this dataset's shape:

1. **Sequence length variance makes batching expensive, not cheap.** Row
   length in this dataset spans roughly 1,000-10,800 characters
   (`code_switching`'s max) with a right tail — batching rows of very
   different lengths together means padding every row in a micro-batch up to
   the longest one, wasting compute and VRAM on padding tokens. At
   micro-batch=1 there is zero padding waste; every forward pass processes
   exactly the tokens that row has.
2. **VRAM is the binding constraint, not compute.** A T4 has 16 GB. The base
   model is already 4-bit quantized (~5-6 GB resident) via Unsloth
   specifically because Gemma-2's `attn_logit_softcapping=50` /
   `final_logit_softcapping=30` are incompatible with FlashAttention-2 and
   overflow-prone in naive fp16 — Unsloth's patched kernels are what make a
   9B model on a T4 tractable at all. Two long sequences landing in the same
   micro-batch is a realistic OOM risk this dataset's tail (max 10,833 chars)
   makes concrete; at micro-batch=1 that risk is structurally eliminated.
3. **The T4 is 4-bit-dequantization-bound anyway.** On this hardware, the
   per-step cost is dominated by dequantizing NF4 weights back to compute
   precision, not by GPU occupancy from a larger batch — so micro-batch=1
   does not leave meaningful throughput on the table the way it would on a
   compute-bound A100 run.

**Effective batch 16**, not larger: with 2,380 train rows this gives 149
steps/epoch (`ceil(2380/16)`), 298 steps across 2 epochs (§3.4) — enough
optimizer steps to make the cosine schedule meaningful (§3.2) without either
so few steps that the schedule barely moves, or accumulation so deep that
gradient staleness across 32+ micro-batches becomes a concern.

### 3.4 Epoch count: 2 — sized against catastrophic forgetting, not against wall-clock

The relevant number is not "how many passes over the data" in isolation but
the same tokens-to-parameters ratio from §3.1, now including the epoch
multiplier: at an estimated 300,000-450,000 supervised tokens/epoch against
54,018,048 trainable parameters, even 2 epochs yields roughly
0.011-0.017 supervised tokens seen per trainable parameter — the adapter is
still heavily over-parameterized relative to total signal at 2 epochs, which
is the direct, quantified version of `LOCKEDIN_PLAN.md` §3.1's point 2
("3,000 rows × ~500 tokens × 3 epochs is already well past minimum signal for
[r=32's] capacity"). At r=16's roughly half the capacity, this run's 2 epochs
sits in a comparable place on that same curve, not an under-trained one.

A 3rd epoch was considered and rejected specifically **because** of the
catastrophic-forgetting risk this document has flagged twice already (§1.1,
§2.2): more passes over the same 2,380 rows increases the model's exposure to
this dataset's specific phrasings relative to Atlas-Chat's own ~450,000-row
Darija pretraining and general capabilities, without adding new information —
each additional epoch is pure repetition, which is exactly the mechanism by
which a narrow fine-tune erodes a base model's broader competence. `save_strategy="epoch"`
checkpoints after each pass specifically so this is not a one-way bet: if
epoch 2's eval loss (§4) comes back worse than epoch 1's — the classic
overfitting signature on a small dataset — the epoch-1 checkpoint is already
on disk and is the one to ship, with no retraining required.

### 3.5 Precision: fp16, auto-detected

T4 (compute capability SM 7.5) has no native bf16 support; the notebook
detects this at runtime (`torch.cuda.is_bf16_supported()`) and selects fp16
accordingly, with bf16 used automatically if the run happens to land on
Ampere-class hardware instead (P100 also lacks bf16 and falls to fp16 the same
way). This is why the smoke-test forward pass in the notebook's §7 explicitly
asserts a finite loss before committing to training — fp16 combined with
Gemma-2's logit softcapping is exactly the combination where a silent
overflow could otherwise go undetected until hours into a run.

---

## 4. Loss & evaluation strategy

### 4.1 What "improvement" means for this specific setup

LoRA's adapter matrices are initialized with `B=0` (Unsloth's standard init),
which means the model's output is mathematically identical to the frozen
base at step 0. Consequently, **the pre-training `eval_loss` baseline is
base Atlas-Chat's own loss on `eval.jsonl`** — not an arbitrary starting
point, but a direct, causally connected zero point. `unsloth_finetune_atlas.ipynb`
§8 measures this baseline explicitly before calling `trainer.train()`, and
asserts the post-training eval loss is lower than it — if it is not, that is
not "the fine-tune underperformed," it is evidence the LoRA configuration or
the masking implementation (§5.2) is broken, and the run is not shippable
regardless of how it looks otherwise.

### 4.2 Aggregate loss curve, and why it is watched during training, not just at the end

The notebook's `Heartbeat` callback prints train loss, learning rate, elapsed
time, ETA, and peak VRAM every `logging_steps=5`, and `eval_loss` is computed
every `eval_steps` (≈75, half an epoch) via `TrainingArguments`. Two shapes
matter:

- **Train loss falling, eval loss falling with it:** healthy — the adapter is
  generalizing the behavior, not just memorizing training rows.
- **Train loss falling, eval loss flattening or rising:** overfitting. Given
  only 298 total steps, this is checked for explicitly at the end of §8 by
  comparing every recorded `eval_loss` checkpoint and flagging if the best one
  was not the final one — in which case the corresponding per-epoch checkpoint
  (§3.4), not the final adapter, is the one to promote.

### 4.3 Per-component eval loss — the evaluation this dataset actually needs

Aggregate eval loss is a weighted average across six behaviorally distinct
components with unequal support (§2.1's table), and a single well-fit large
component can mask a poorly-fit small one in the aggregate number. The
notebook's §9 computes eval loss **separately per component**, using the
model's own fused cross-entropy (`labels=` on the forward pass) rather than
materializing full logits — at `max_seq_length=4096` and a 256,000-token
vocabulary, an fp32 logits tensor would be roughly 4 GB *per row*, which would
OOM a T4 if computed the naive way.

A component is flagged if its loss/token exceeds **1.6× the overall mean**.
That threshold is deliberately wide rather than tight: the six components are
not equally hard by nature (free-form Darija conversation is genuinely
higher-entropy than schema-constrained quiz JSON), so some spread across
components is expected and not itself a problem. 1.6× is chosen to be loose
enough not to fire on that expected spread while still catching a component
the adapter functionally failed to learn. Given `darija_preservation`'s 11
eval rows (§2.4) — the smallest eval slice by a wide margin — its reading on
this table is the noisiest in the set; a single flagged result there is
grounds to look at its actual generations by hand (§4.4), not to treat the
number alone as conclusive.

### 4.4 Generation smoke test — what loss cannot see

Loss is a token-prediction proxy; it cannot directly verify four things this
project has specifically broken on before (`CHANGELOG.md`), which is why §10
of the notebook generates real completions from real eval prompts and checks
each explicitly rather than trusting the loss curve to imply them:

1. **Arabic-script output share** — `LOCKEDIN_PLAN.md` §2.5 made Arabic
   script the fixed training target (not Arabizi) after measuring a 76%
   unusable-output rate under the old Arabizi-first approach; a fine-tune
   drifting back toward Latin-script output would silently undo that fix.
2. **Quiz JSON validity** — MVP feature #3. The dataset itself has 0/374
   invalid JSON rows (schema-constrained generation), so any parse failure in
   the fine-tuned model's own output is a new regression introduced by
   training, not an inherited one.
3. **Clean stopping** — see §5.3. A model that has learned to emit
   `<end_of_turn>` but is served with the wrong stop-token configuration will
   continue generating and role-play the user's next turn; the smoke test
   catches this by checking for turn-marker leakage in raw output.
4. **CJK contamination** — 0/2,655 in the source dataset; `validate_chatml()`
   rejects it at generation time, so any reappearance in fine-tuned model
   output specifically indicates a decoding-side regression, not a dataset
   issue.

This is explicitly a smoke test (5 generations), not a rate measurement — it
catches catastrophe, not drift. `FINETUNE_AND_DEPLOY.md` §4 defines the
larger go/no-go checklist this feeds into, including the recommended next
step of re-running the project's own quality gates
(`row_is_code_switched`, `row_is_grounded_darija`, `french_term_count`, the
citation extractor) against fine-tuned *model output* rather than dataset
rows, so post-training numbers are directly comparable to the pre-training
figures in §2.3 rather than a new, incomparable metric.

---

## 5. Deployment & inference guidance

### 5.1 The parity problem this section exists to solve

`MBZUAI-Paris/Atlas-Chat-9B`'s shipped `tokenizer_config.json` contains a chat
template that opens with:

```jinja
{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}
```

Gemma-2 has no system turn at all, and every row in this dataset starts with
one. Calling `apply_chat_template()` with the stock template raises on 100%
of the training data — confirmed directly (not assumed) against this exact
dataset before the training notebook was finalized. `LOCKEDIN_PLAN.md` §6
flagged this requirement ("this must be merged into the first user turn, and
the same merge must happen in `llm.py` at serving time to preserve train/serve
parity") and marked it not yet implemented; this run implements it.

**The resolution is parity-by-construction, verified, not assumed:** a
custom `chat_template` is installed on the tokenizer that merges `system`
into the first user turn (separated by a blank line), and the notebook's §4
**asserts** that this template produces byte-identical output to the
hand-written function used to build training labels — checked at build time
against all 2,655 real dataset rows (not a sample), 0 mismatches. The patched
tokenizer is then saved *inside* the adapter directory, so any serving stack
that loads the tokenizer from there reproduces the training format
automatically, with no serving-side code needed to remember the merge rule.

The trained/served format, exactly:

```
<bos><start_of_turn>user
{SYSTEM}

{USER_1}<end_of_turn>
<start_of_turn>model
{ASSISTANT_1}<end_of_turn>
```

### 5.2 Loading the LoRA adapter — vLLM multi-LoRA (primary path)

Per `LOCKEDIN_PLAN.md` §2.3, this is the production serving target. No
serving-code change is required for the chat template specifically, provided
one rule is followed: **load the tokenizer from the adapter's `lora_model/`
directory, never from `MBZUAI-Paris/Atlas-Chat-9B` directly** — the upstream
tokenizer still carries the stock template that rejects system messages.

```python
# vLLM server, sketch — base stays frozen, adapter hot-swapped per request
from vllm import LLM
from vllm.lora.request import LoRARequest

llm = LLM(model="MBZUAI-Paris/Atlas-Chat-9B", enable_lora=True, max_lora_rank=16)
lora_req = LoRARequest("darija-tutor", 1, "/path/to/lora_model")

# tokenizer for prompt construction is the one INSIDE lora_model/, not the base repo's
```

The exported zip's `TRAINING_REPORT.json` records every hyperparameter, both
loss curves, the per-component eval-loss table, and the smoke-test findings —
load-bearing for any handoff to whoever operates the vLLM deployment, since it
is the only artifact that answers "was this adapter actually good" without
re-running the notebook.

### 5.3 Loading the GGUF — Ollama (secondary / local dev path)

`app/services/llm.py` currently posts to Ollama's `/api/generate` with a
separate `system` field; Ollama then renders the model's own template
server-side. The exported `Modelfile`'s `TEMPLATE` merges `.System` into the
first user turn exactly as trained, so **`llm.py`'s request shape needs no
change** — only `settings.ollama_model` needs to point at the new model:

```bash
cd <unzipped finetuned_model_export>
ollama create atlas-darija-tutor -f Modelfile
ollama show --template atlas-darija-tutor      # verify before trusting it
```

The verification step is not optional: if `ollama create` silently fell back
to a built-in gemma2 template instead of the shipped one, the system prompt
would be joined differently than what the model was trained on, and the
served model would be receiving a prompt shape it has never seen. The output
of `ollama show --template` must show `{{ .System }}` and `{{ .Prompt }}`
inside the same `<start_of_turn>user` block, separated by a blank line — not
in separate blocks.

**Stop-token patch — required, not optional.** Atlas-Chat's
`generation_config.json` (the file `.generate()` actually consults, distinct
from `config.json`'s own `eos_token_id`) lists only `eos_token_id: 1`
(`<eos>`). This dataset trains the model to terminate turns with
`<end_of_turn>` (token id 107) — deliberately kept inside the supervised span
during training specifically so the model learns to emit it. Served against
the unpatched generation config, the model emits `<end_of_turn>`, nothing
recognizes it as a stop condition, and generation continues by role-playing
the user's next turn — the most likely way a technically-successful fine-tune
would present as broken in a live demo. The export's `generation_config.json`
is patched to `eos_token_id: [1, 107]`, and the Modelfile carries a matching
`PARAMETER stop "<end_of_turn>"`. It also drops Atlas-Chat's upstream
`repetition_penalty`, which is shipped as the **string** `"1.2"` rather than a
float and causes some loaders to throw.

### 5.4 The standing invariant, restated

`generate_training_data.PRODUCTION_SYSTEM_PROMPT_TEMPLATE` and
`llm.SYSTEM_PROMPT_TEMPLATE` must stay textually identical — the training
notebook asserts this at build time when `app/` is included in the Kaggle
upload, and it is the same invariant `LOCKEDIN_PLAN.md` has held since before
generation started. If either template is ever edited, the other must be
edited to match, or the *next* dataset generation run silently trains against
a system prompt production does not actually send — the exact failure mode
this whole section exists to prevent from happening a second time, in the
other direction (serving against a format training did not send).
