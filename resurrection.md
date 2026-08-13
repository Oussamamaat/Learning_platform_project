# Resurrection — Decision Register

**Purpose.** Every choice that shaped this fine-tune, restated as a question you
can answer from scratch. Not a tutorial and not a history: a list of forks in
the road, what was taken, what it cost, and what taking the other one would
mean now.

**How to use it.** Answer by number (`Q7: B`). Each question carries a blast
radius marker — that is the honest cost of changing your mind, not a
discouragement:

- 🔴 **Load-bearing.** Changing this invalidates the dataset, the adapter, or
  both. Weeks of rework.
- 🟡 **Moderate.** Costs a regeneration run or a retrain. Days.
- 🟢 **Cheap.** Config, prompt, or serving change. Hours, reversible.

**Standing constraint.** `prd_mvp.md` sets delivery at **2026-08-30**. Every
🔴 answered differently from today's value very likely does not fit inside
that. Answer them anyway — but answer the date question (Q0.3) honestly first,
because it governs which of the rest you can afford.

**Evidence rule.** Where a "why" is stated below, it was measured in this
project, not assumed. Where something was never measured, it says so. Do not
grant the current value more authority than its evidence line supports.

---

## 0. Scope and constraints

The frame everything else sits inside. Answer these first — several later
questions dissolve depending on what you say here.

| | Current | Where |
|---|---|---|
| MVP features | 3: conversational assistant (text+audio), personalized explanations, quiz generation | `prd_mvp.md` §2, CEO-confirmed |
| Domain | Moroccan safety/security regulations, multi-tenant | `prd_mvp.md` §1 |
| Delivery | 2026-08-30 | `prd_mvp.md` header |
| Deferred | multilingual, auto-summaries, difficulty analysis | `prd_mvp.md` §2 |

**Q0.1** 🔴 Are all three MVP features still in scope?
- **A.** Yes, unchanged.
- **B.** Drop quiz generation → `quiz_generation` (462 train rows, the single
  largest component) becomes dead weight; also removes the structured-output
  pressure that `LOCKEDIN_PLAN.md` §3.1 credits with stopping the adapter
  drifting into conversation-only behaviour. Removing it is not free.
- **C.** Drop audio → Q0.2 disappears entirely.
- **D.** Add something (say what).

**Q0.2** 🔴 Audio: what actually ships?
STT has a path (community Darija Whisper). **TTS has no vendor and no
evaluation** — flagged as open since the project began and still the single
largest unresolved MVP item (`LOCKEDIN_PLAN.md` §7). Candidates named but never
tested: `DarijaTTS-v0.1-500M`, a SpeechT5-darija Space, `atlasia/DODa-audio-dataset`.
- **A.** Text-only MVP; audio post-MVP. *(Honest, given 4 weeks and zero TTS work done.)*
- **B.** STT only — user speaks, model answers in text.
- **C.** Full duplex; commit to evaluating a TTS vendor this week.

**Q0.3** 🔴 Is 2026-08-30 movable?
- **A.** Hard date. → Restrict yourself to 🟢 and at most one 🟡 below.
- **B.** Soft. → 🟡s are affordable.
- **C.** Already slipped / renegotiated. → State the real date; 🔴s open up.

**Q0.4** 🟡 Who signs off that output quality is acceptable?
No native-speaker review has ever been run on generated or model output. Every
quality claim in this project is machine-measured. `LOCKEDIN_PLAN.md` §7 flags
two open questions that *only* a human read can settle.
- **A.** You alone.
- **B.** CEO demo is the bar (`green_light_model.md` §4.2's 10 questions).
- **C.** Recruit 3–5 target users for one session. *(The only option that
  resolves Q3.1 and Q4.7.)*

---

## 1. Base model and licensing

| | Current | Where |
|---|---|---|
| Fine-tune base | Atlas-Chat-9B (Gemma-2-9b-it lineage) | `LOCKEDIN_PLAN.md` §2.1 |
| Generation model | Atlas-Chat-9B, Q4_K_M GGUF via Ollama | same |
| Method | QLoRA, 4-bit, Unsloth | `prd_mvp.md` §3 |
| License | Gemma ToU (not Apache 2.0) | `LOCKEDIN_PLAN.md` §2.3 |

**Q1.1** 🔴 Keep Atlas-Chat-9B as the fine-tune base?

The argument for it is the strongest evidence in this project and you should
know it before overturning it: Atlas-Chat's Darija comes from LoRA-256 on
~450,000 Darija instructions. Your dataset is ~3,000 rows — **under 1% of
that**. A dataset this size cannot teach Darija; it can only reshape how a
model that already speaks it behaves. Benchmarks: DarijaMMLU 58.23 (Jais-13B:
45.20); DODa BLEU 28.08 (Llama-3.1-8B: **0.92** — a same-size general model
essentially cannot produce Darija at all).

- **A.** Keep. The reframe from "language acquisition" to "behaviour transfer"
  is what makes every downstream size/rank choice coherent.
- **B.** GemMaroc-27B — matches Atlas on DarijaMMLU, beats it on Darija
  HellaSwag (60.5 vs 48.4). Cost: 27B will not fine-tune on your 8GB laptop
  and will not serve on it either. Kaggle-only, different serving story.
- **C.** A newer Atlas / Gemma-3 base. Requires re-running every benchmark
  claim above; none of it transfers automatically.
- **D.** Non-Darija base (Qwen, Llama). **This was the original plan and was
  explicitly reversed** — see the DODa BLEU 0.92 figure. Choosing it means
  your 3,000 rows must now teach a language, which they cannot.

**Q1.2** 🟡 Should the *generator* be the same model as the *base*?

Currently yes. This is self-distillation: Atlas generates the data that trains
Atlas. It cannot teach knowledge Atlas lacks — only reshape behaviour it
already has. This is a real ceiling that was never explicitly chosen, it just
happened.
- **A.** Keep — cheap, offline, no API cost, license-clean.
- **B.** Generate with a stronger teacher (Claude/GPT) for the pedagogy-heavy
  components, keeping Atlas for register/Darija fidelity. Better ceiling; adds
  API cost and a second license to check.
- **C.** Human-authored seed rows for the components that keep failing gates
  (`code_switching` is the current candidate — see Q7.2).

**Q1.3** 🟡 Does the model get distributed outside IBLOG?
Gemma's ToU defines "Model Derivatives" to include synthetic-data-distillation
outputs — so anything trained on Atlas-generated data is a Gemma derivative
*regardless of base*. Commercial use is permitted; **distribution** carries
pass-through obligations. Never given a real legal read.
- **A.** Internal only → obligations largely dormant.
- **B.** Shipped to clients / hosted externally → get the legal read **before**
  the demo, not after.

---

## 2. Corpus and retrieval (the untested half)

| | Current | Where |
|---|---|---|
| Corpus | 37 files, ~1,625 chars/doc | `raw/`, `CORPUS_INDEX.csv` |
| Chunking | `RecursiveCharacterTextSplitter`, size 400, overlap 50 | `ingestion.py:29-30` |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (384-dim) | `ingestion.py:28` |
| Retrieval | pgvector cosine (default backend since 2026-08-10), top-k 4-5, similarity threshold 0.15 (was 0.4, re-benchmarked), language-affinity two-pass selection, heading-aware chunking | `search.py`, `retrieval.py` |
| Domains | industrial, securite, blockchain (+6 synthetic); auto-routed per turn since 2026-08-11 | `generate_training_data.py:150`, `routing.py` |

> **Read this before answering anything in this section.** The gate evaluation
> run on 2026-08-02 measured the *generation* half only. It fed the model gold
> context lifted straight from each eval row's system prompt
> (`context_from_system_prompt`, marker `CONTEXTE :\n`) and **never touched
> pgvector**. That was the right call for isolating the fine-tune — but it
> means **retrieval quality in this system has never been measured at all.**
> If retrieval returns the wrong document, the model will faithfully ground in
> the wrong document and every gate in `green_light_model.md` still passes.

**Q2.1** 🔴 Is the 37-document corpus the real corpus, or a placeholder?
`prd_mvp.md` Task 1.4 describes synthetic data as *"a placeholder while
awaiting official course materials."* Those materials never arrived. The
corpus ceiling (~15–20 rows per document before generation starts paraphrasing
rather than adding information) is a primary reason the dataset stopped at
~3,000 rows.
- **A.** This is the corpus. Accept the ceiling; stop trying to scale rows.
- **B.** Real client materials are still coming → **pause dataset work**, the
  ceiling moves and every row-count argument is rewritten.
- **C.** Expand synthetically (more generalization domains).

**Q2.2** 🟡 Keep `paraphrase-multilingual-MiniLM-L12-v2`?
384-dim, small, fast, multilingual — chosen by default, never benchmarked
against your actual Arabic/French mixed corpus. Retrieval accuracy on this
corpus is an unknown number, not a good one.
- **A.** Keep unmeasured. *(Only defensible if you answer Q2.4 = A.)*
- **B.** BGE-M3 or multilingual-e5 — materially stronger on Arabic, larger and
  slower.
- **C.** Benchmark before choosing. Requires Q2.4.

**Q2.3** 🟢 chunk 400 / overlap 50 / top-k 5 — keep?
400 chars is small for legal prose; an article can split across chunks, which
directly threatens the verbatim-citation behaviour the whole grounding design
depends on. Never tuned.
- **A.** Keep. **B.** Larger chunks (800–1000) to keep articles whole.
- **C.** Structure-aware splitting on article boundaries. **D.** Tune top-k
  against a real retrieval eval.

**Q2.4** 🔴 Build a retrieval evaluation?
Needs: Postgres+pgvector running, the corpus ingested, and ~50 labelled
query→document pairs. Without it you cannot distinguish "the model hallucinated"
from "retrieval handed it the wrong page" — and today you cannot tell those
apart at all.
- **A.** Yes, before the demo. *(Recommended — it is the largest blind spot in
  the system.)*
- **B.** After MVP; accept the blind spot knowingly.
- **C.** No; the gold-context evaluation is enough. *(Only honest if you also
  accept you are shipping an unmeasured RAG pipeline.)*

---

## 3. Output language and script

| | Current | Where |
|---|---|---|
| Answer script | Arabic-script Darija | `LOCKEDIN_PLAN.md` §2.5 |
| Technical vocab | French, Latin letters, never translated | `llm.py:33-35` |
| Arabizi input | Understood, answered in Arabic script | `llm.py:31-32` |
| French input | **Answered in French** (added 2026-08-02) | `llm.py:120-149` |
| French refusals | **Fall back to Darija — known, unfixed** | `llm.py:68-76` |

**Q3.1** 🔴 Arabic script or Arabizi as the primary reading experience?
**Still unresolved, and not resolvable by any measurement this project can
produce internally.** Arabic script was chosen partly because forcing Arabizi
via character-map transliteration produced *Arabic-in-Latin-letters*, not
natural Moroccan Arabizi (76% unusable on an early pilot). `messages_arabizi`
is kept per-row (`generate_training_data.py:2381`), so a Latin display mode is
cheap to switch on — but nothing trains on it.
- **A.** Arabic script. *(Current. Trained. Safe.)*
- **B.** Arabizi display mode using the existing character map — cheap, quality
  unvalidated.
- **C.** Settle it with users first (Q0.4 = C). *(The only answer that produces
  evidence rather than another assumption.)*

**Q3.2** 🟡 Should the model answer French questions in French at all?
Added 2026-08-02 in response to a French question being answered in Darija.
Root cause was measured and is worth knowing: the **document's** language drove
the output language, not the question's — reproduced on base Atlas-Chat with no
adapter, so it was never an adapter defect. The fix uses a *separate* French
template so the trained prompt stays byte-identical (Q6.2).
- **A.** Keep French routing. **B.** Darija-only; drop it (the platform is a
  Darija-only product). **C.** Make it a per-tenant setting.

**Q3.3** 🟡 Fix French refusals?
Answerable French questions return French. **Refusals still come back in
Darija.** Measured against three attempted fixes — a direct instruction, a hard
negative constraint, and a French refusal exemplar (the model copied the
exemplar's content and still answered in Darija). Cause: `grounded_refusal` is
417 training rows, **100% Arabic-script**. That prior beats any prompt.
- **A.** Accept and document. *(Current.)*
- **B.** Generate French refusal rows and retrain. **The only fix that can
  work** — this is a data problem, not a prompt problem.
- **C.** Post-process refusals through a translation pass. Cheap, ugly.

---

## 4. Dataset design

| | Current | Where |
|---|---|---|
| Size | 3,064 rows (2,757 train / 307 eval, ~90/10) | `data/v11_merged` |
| Components | 11 | `COMPONENT_CONFIG`, `generate_training_data.py:50` |
| Domain split | 80/20 client/generalization | `LOCKEDIN_PLAN.md` §3.1 |
| Multi-turn | ~50% of socratic+code_switching | `green_light_model.md` RF8 |

Component weights as they stand: socratic 800, grounded_refusal 700,
code_switching 700, quiz_generation 400, darija_preservation 200,
reasoning_preservation 200, structured_explanation 200, learner_adaptation 150,
general_knowledge_disclosed 150, no_context_refusal 150, injection_resistance 100.

**Q4.1** 🔴 Is 11 components the right taxonomy?
It grew from 6. The five additions were **not** scope creep — each closed a
*measured* zero-coverage failure found by behavioural evaluation of the first
adapter: empty-context fabrication (0/4 refused), prompt-injection compliance
(3/4 succeeded, i.e. failed to resist), and no signal whatsoever for the
"genuine general knowledge vs. ungroundable company question" distinction.
- **A.** Keep 11. **B.** Merge overlapping ones (`no_context_refusal` vs
  `grounded_refusal`; `structured_explanation` vs `socratic`). **C.** Add for a
  gap you have measured — *measured*, not suspected. **D.** Cut to 6; accept
  the failure modes come back.

**Q4.2** 🟡 Are the weights right?
Never independently derived — they encode intent, and intent is not
enforcement. Concretely: `code_switching` at weight 700 produced training data
scoring 93.8% on the code-switch gate, yet the trained model scores **60.8%**
on the same gate. Weight did not buy the behaviour.
- **A.** Keep. **B.** Reweight toward measured weakness (`code_switching`).
- **C.** Derive weights from per-component eval loss instead of by hand.

**Q4.3** 🟡 Keep the preservation components?
`darija_preservation` (200) and `reasoning_preservation` (200) are
regularization, not acquisition — they exist so a narrow behavioural fine-tune
doesn't erode what Atlas already does. Grounded in GemMaroc (arXiv:2505.17082):
mixed data lifted DarijaMMLU 32.8 → 47.5 **with no English regression**. Their
cost shows up in your metrics — `reasoning_preservation` rows return JSON and
non-Darija by design, so they drag the aggregate Arabic-script number down and
the metric is simply wrong for them.
- **A.** Keep both. **B.** Keep, and **exclude them from aggregate metrics** —
  they are currently being scored against a gate they are designed to fail.
- **C.** Drop; accept regression risk.

**Q4.4** 🟡 3,000 rows — still the target?
Derived from four constraints, all still binding: corpus ceiling (Q2.1), LoRA
capacity at r=16, review capacity (15% of 3,000 = 450 rows is realistic; of
7,500 = 1,125 is not), and the GemMaroc precedent at 5K.
**Cautionary note, and it cost GPU hours:** row count stopped being the useful
lever partway through. Two iteration cycles were spent tuning
`COMPONENT_CONFIG`'s `multi_turn_pct` — a *generation-time probability* — as
though it were an *acceptance threshold*, chasing a gate the pool had already
passed. **Acceptance criteria live in `green_light_model.md` §4.1, never in
`COMPONENT_CONFIG`.**
- **A.** 3,000 is enough. **B.** Scale to 5,000 (GemMaroc's number) — needs
  Q2.1 = B/C, or you just generate paraphrases. **C.** Stop counting rows;
  target measured behaviours instead.

**Q4.5** 🟢 90/10 train/eval split?
307 eval rows. Small per-component: `injection_resistance` has **5**, and no
conclusion drawn from 5 rows is trustworthy.
- **A.** Keep. **B.** 85/15 for tighter per-component confidence.
- **C.** Stratified minimum (≥20/component) — the fix aimed directly at the
  actual problem.

**Q4.6** 🟡 Keep the 80/20 client/generalization domain split?
3 real domains + 6 synthetic (medical, legal, automotive, RH, logistique,
hôtellerie), teaching that the behaviour is domain-agnostic rather than bound
to today's three verticals.
- **A.** Keep. **B.** 100% client domains — sharper now, brittle when a
  4th vertical arrives. **C.** More generalization if you expect to expand.

**Q4.7** 🟡 Should refusals contain French at all?
`grounded_refusal` French density passes its gate, but whether French
vocabulary *belongs* in a refusal quoting Arabic legal text is a **product**
question that has never been asked of a human. Needs a native-speaker read of
a handful of rows — not more GPU time.
- **A.** Keep. **B.** Pure Darija refusals. **C.** Ask a native speaker (Q0.4).

**Q4.8** 🟢 Per-component source-document routing?
Currently: `grounded_refusal` 50/50 Arabic/French; `socratic` + `code_switching`
French-only (French density is the point); `quiz_generation` Arabic-only.
This routing has a measured consequence people keep misreading as a bug:
French-script documents rarely carry extractable numbered references, so only
**14%** of French-source rows are even citable versus **100%** of Arabic-source
rows. Aggregate citation recall therefore *looks* low by composition. Within
each population it is healthy.
- **A.** Keep. **B.** Rebalance. **C.** Uniform routing; accept register loss.

---

## 5. Generation infrastructure

| | Current | Where |
|---|---|---|
| Platform | Kaggle, dual T4, `CUDA_VISIBLE_DEVICES`-pinned Ollama workers | `LOCKEDIN_PLAN.md` §4.1 |
| Quant | Q4_K_M GGUF | same |
| Dedup | cross-shard, cosine > 0.95, **forced CPU** | `merge_shards.py` |

**Q5.1** 🟢 Stay on Kaggle for generation?
- **A.** Yes — free, dual T4, proven. **B.** Paid cloud (no 12h session cap;
  the remaining-deficit estimate was ~14 GPU-hours, which *cannot* fit in one
  Kaggle session). **C.** Local — 8GB VRAM makes this painful.

**Q5.2** 🟢 Keep the two hard-won operational guards?
Both exist because of silent failures that cost real runs: a headless
`kaggle kernels push` landed on a **single P100** instead of the requested T4
and the second worker pointed at a nonexistent CUDA index — half the throughput
vanished with no error. And GPU dedup **silently failed** once on Kaggle/CUDA,
shipping 32% of a run undeduplicated.
- **A.** Keep both (explicit `--accelerator NvidiaTeslaT4`, fail-fast
  `nvidia-smi -L` assertion, CPU-forced dedup). *(Strongly recommended — these
  were paid for.)*
- **B.** Relax the CPU dedup for speed. Re-accepts a known silent failure.

---

## 6. Fine-tune configuration

| Setting | Value | Why (from `FINETUNING_RATIONALE.md` §3) |
|---|---|---|
| r / alpha | 16 / 16 | 54,018,048 params = 0.585% of 9.24B, against ~300–450K supervised tokens/epoch |
| dropout | 0 | required by Unsloth's fused kernel path |
| targets | q,k,v,o,gate,up,down | no `embed_tokens`/`lm_head` — not teaching new tokens |
| max_seq | 4096 | clears measured p99; overlong rows dropped, never truncated |
| batch × accum | 1 × 16 | zero padding waste; T4 is dequant-bound |
| epochs | 2 | checkpointed per epoch so a bad epoch 2 doesn't cost the run |
| lr / sched | 2e-4, cosine, 3% warmup | Unsloth's validated pairing at this rank |
| precision | fp16 | T4/P100 have no bf16 |
| optim | `adamw_8bit` | halves optimizer VRAM, no measured quality cost |

**Q6.1** 🟡 r=16 / alpha=16?
r=32 doubles capacity without doubling signal — the risk is phrasing-level
overfit instead of behaviour transfer. Not arbitrary: `embed_tokens`/`lm_head`
are excluded because Gemma-2's tied 256,000-row embedding matrix is 917.5M
parameters — **17× the entire adapter** — and would dominate every gradient
update.
- **A.** Keep. **B.** r=32 — justified *only* if you also grow the dataset.
- **C.** r=8. **D.** Decouple alpha (e.g. r=16/α=32) for a stronger update.

**Q6.2** 🔴 Keep the train/serve parity invariant?
`PRODUCTION_SYSTEM_PROMPT_TEMPLATE` (generation) must stay **byte-identical**
to `SYSTEM_PROMPT_TEMPLATE` (`llm.py`), asserted at build time. Underneath it:
Atlas's shipped chat template raises `'System role not supported'` — Gemma-2
has no system turn and every row starts with one. Resolved by a custom template
merging system into the first user turn, **asserted byte-identical against every
real row**, with the patched tokenizer shipped inside the adapter directory so
any serving stack reproduces it with no serving-side memory.
- **A.** Keep, absolutely. *(This is the most safety-critical invariant in the
  project. The French template was deliberately added as a **separate**
  template precisely to avoid touching it.)*
- **B.** Relax it → you must then re-verify format parity by hand on every
  deploy, forever.

**Q6.3** 🟢 2 epochs?
Run 1: eval_loss 1.1651 → 0.85062. Run 2 (v11): 1.1823 → 0.8407, all 11
components inside the 1.6× underfit threshold.
- **A.** Keep. **B.** 3 epochs (watch overfit; epoch-1 checkpoint is retained).
- **C.** Early-stop on eval loss.

**Q6.4** 🟢 QLoRA 4-bit, or LoRA on 16-bit weights?
4-bit was chosen for the 8GB laptop constraint. Kaggle T4s (16GB) could run
16-bit LoRA — slightly better fidelity, no dequant noise during training.
- **A.** Keep QLoRA. **B.** 16-bit LoRA on Kaggle.

---

## 7. Serving

| | Current | Where |
|---|---|---|
| **Deployed now** | Two **merged** standalone Q4_K_M GGUFs, Ollama: `IBLOG_TUTOR:latest` (Darija) + `iblog-tutor-fr:latest` (French), selected per-turn by resolved response language | `config.py` |
| Planned | vLLM multi-LoRA, frozen base + hot-swapped adapters | `LOCKEDIN_PLAN.md` §2.2 |
| Params | temp 0.2, num_ctx 4096, timeout 180s | Modelfile, `llm.py:210` |

> **Note the drift.** The plan is vLLM multi-LoRA with a frozen base. What is
> actually running is a single fully-merged model in Ollama. Merging is the
> *opposite* direction from multi-LoRA — you cannot hot-swap adapters out of
> merged weights. Both are defensible; running one while documenting the other
> is not.

**Q7.1** 🔴 vLLM multi-LoRA, or merged single model?
- **A.** Merged Ollama (**what you have now**) — simplest, fastest to ship, one
  model per capability, no hot-swap.
- **B.** vLLM multi-LoRA as planned — needed if `iblog-diagram` and other
  adapters are real. Costs an integration you have not started.
- **C.** Merged for MVP, multi-LoRA post-MVP. *(Recommended: matches reality,
  keeps the roadmap alive, and requires no decision reversal now.)*

**Q7.2** 🟡 Ship the current adapter, given the code-switch gap?
The 2026-08-02 gate evaluation, corrected for population (model output vs. the
*same 307 rows'* gold answers — not the whole-pool §4.1d figures):

| Gate | Model | Gold, same rows | Read |
|---|---|---|---|
| Citation recall | 61.1% | 65.6% | parity — the 70% bar does not fit this split |
| Arabic-script | 91.2% | 92.2% | parity |
| **Code-switch** | **60.8%** | **93.8%** | **real 33pp gap, 0.8pp above the floor** |
| CJK | 1 | 0 | real, rare |
| Quiz JSON | 94% (47/50) | 100% | real, rare |

Citation recall is **not** a regression — the training data itself scores 65.6%
on this split, and you cannot exceed your own targets. The threshold was
calibrated on a different population. The code-switch gap is the genuine
finding, and it passes only by 0.8pp — inside run-to-run variance.

- **A.** Ship; log the gap. **B.** Fix code-switch first (Q4.2 = B) and
  retrain. **C.** **Run the missing control first** — the same eval against
  `atlas-darija-tutor-v11` (pre-merge). ~70 min. It is the only thing that
  distinguishes *"the merge cost us French density"* from *"the fine-tune never
  had it"*, and that distinction changes the fix. *(Recommended before A or B.)*

**Q7.3** 🟢 Q4_K_M quantization?
- **A.** Keep (5.8GB, fits 8GB VRAM). **B.** Q5_K_M / Q8 — better fidelity,
  more VRAM. **C.** fp16 — needs a bigger GPU than you have.

**Q7.4** 🟢 temperature 0.2?
Low, chosen for grounding discipline. Never A/B'd against pedagogical quality —
Socratic teaching may want more variety than citation accuracy does.
- **A.** Keep. **B.** Raise for conversational components. **C.** Per-endpoint
  (low for quiz/citation, higher for socratic).

**Q7.5** 🟢 Conversation history — does the product need it?
**Closed, option A.** `app/services/history.py` (added 2026-08-11) gives
`generate_llm_response()` a `history` parameter backed by server-side Postgres
storage, with pinned-context/segment-reset so same-topic follow-ups reuse a
byte-identical system block (KV-prefix reuse) and a real topic shift starts a
fresh segment. Fail-open: a Postgres hiccup degrades a turn to the old
stateless behaviour rather than crashing. Live-verified via
`probe_multiturn_coherence.py` and `probe_language_routing.py` against real
Ollama + Postgres. Detail: `docs/architecture/serving.md` §Conversation
history, `docs/architecture/data-and-retrieval.md`.
- **A.** Add session history. *(Done — this is now what's deployed.)*
- **B.** Stateless MVP; document the limitation loudly.
- **C.** Client-side history replayed into the prompt.

---

## 8. Evaluation

| | Current | Where |
|---|---|---|
| Dataset gates | 8 automated checks | `green_light_model.md` §4.1 |
| Live demo | 10 CEO-facing questions | §4.2 |
| Model-output gates | dataset gates re-pointed at generations | 2026-08-02 run |
| Base-vs-adapter | **standing requirement** | `LOCKEDIN_PLAN.md` §6.2 |

**Q8.1** 🔴 Keep the base-comparison requirement?
This exists because of the most important finding in the project. A controlled
comparison found base Atlas-Chat **correctly declining** to cite («ماكاينش شي
قانون محدد») where the first fine-tuned adapter **confidently fabricated** a law
(«حسب القانون 27-04»), 2/2 each way. Root cause was distributional, not a few
bad rows: legal Q&A rows almost always cited *something*, so the model learned
"legal question → emit citation" and filled the slot from pretrained knowledge
when context had none. Deleting flagged rows would have left the teaching
distribution fully intact. **A fine-tune can land below its own base model.**
- **A.** Keep — every new adapter compared against base, not just the previous
  adapter. *(Strongly recommended. It has already caught one catastrophic
  regression, and v11 passes it: v11 declined on the bait prompt and named the
  article on the should-cite prompt — better than base on both.)*
- **B.** Compare only against the previous adapter. Re-opens the exact hole.

**Q8.2** 🟡 Re-baseline the thresholds?
`green_light_model.md` §4.1d was measured on all 3,064 rows; the model was
scored on the 307-row eval split. Comparing them produced a **false failure**
(citation recall "61.1% vs 78.9%") that dissolved once the gold answers on the
*same rows* were measured at 65.6%. Thresholds must name their population.
- **A.** Re-baseline per split and record both. *(Recommended.)*
- **B.** Keep whole-pool numbers, annotate the mismatch. **C.** Drop fixed
  thresholds; always compare model vs. gold on identical rows.

**Q8.3** 🟡 Add human evaluation?
Every quality claim in this project is machine-measured. Gates cannot see
whether the Darija sounds natural, whether the pedagogy actually teaches, or
whether a refusal reads as helpful or curt.
- **A.** Machine gates only. **B.** Native-speaker review of ~50 outputs.
- **C.** Structured user testing (also settles Q3.1 and Q4.7).

**Q8.4** 🟢 Fix the metric bugs the audit surfaced?
Three, all real, none yet fixed: (1) `reasoning_preservation` rows are scored
against an Arabic-script gate they are *designed* to fail; (2) French-routed
answers count as Arabic-script failures despite being correct behaviour; (3)
refusal rows count against citation recall although a refusal has nothing to
cite by definition.
- **A.** Fix all three; re-score. *(Cheap, and the current numbers are wrong
  until you do.)*
- **B.** Leave; interpret manually each time.

---

## 9. Known-open, carried forward

Not questions — commitments already made that nothing has closed. Listed so
they are not rediscovered later as surprises.

1. **TTS vendor** — MVP-critical, zero evaluation run. (Q0.2)
2. **Retrieval quality** — measured 2026-08-10/11 (baseline, heading-aware
   chunking, language affinity, pgvector cutover, auto-domain routing); see
   `docs/architecture/data-and-retrieval.md`. Domain-routing tier-2 vote
   accuracy (0.778) is the open sub-item now. (Q2.4)
3. **Conversation history** — closed 2026-08-11, now servable. (Q7.5)
4. **Arabic vs Arabizi** — needs users, not measurement. (Q3.1)
5. **French refusals** — needs data, not prompts. (Q3.3)
6. **Refusal scope mismatch** — v11 declined a blockchain question claiming it
   only handles "workplace safety and health." Safe (it declined rather than
   fabricating) but reads as a bug in a live demo.
7. **Code-switch 33pp gap** — merge vs. fine-tune not yet distinguished. (Q7.2 = C)
8. **Context leakage** — one socratic generation echoed the raw source document
   verbatim, HTML `<br>` markup included, instead of answering.
9. **Gemma license read** — never done. (Q1.3)
10. **Diagram adapter** — roadmapped, not started; `reasoning_preservation`
    keeps the option open without committing to it.

---

## 10. Answer sheet

Copy, fill, hand back. Answer what you have a view on; blanks are fine and
mean "keep current."

```
Q0.1 __   Q0.2 __   Q0.3 __   Q0.4 __
Q1.1 __   Q1.2 __   Q1.3 __
Q2.1 __   Q2.2 __   Q2.3 __   Q2.4 __
Q3.1 __   Q3.2 __   Q3.3 __
Q4.1 __   Q4.2 __   Q4.3 __   Q4.4 __   Q4.5 __   Q4.6 __   Q4.7 __   Q4.8 __
Q5.1 __   Q5.2 __
Q6.1 __   Q6.2 __   Q6.3 __   Q6.4 __
Q7.1 __   Q7.2 __   Q7.3 __   Q7.4 __   Q7.5 __
Q8.1 __   Q8.2 __   Q8.3 __   Q8.4 __
```

**If you only answer four**, answer these — they gate the most downstream work:

- **Q0.3** (is the date real) — governs what you can afford to change at all.
- **Q2.1** (is the corpus real or placeholder) — rewrites every dataset sizing
  argument if it is a placeholder.
- **Q7.5** (conversation history) — an MVP feature currently cannot work as
  designed without it.
- **Q7.2** (ship, fix, or run the control) — the immediate next action either
  way.
