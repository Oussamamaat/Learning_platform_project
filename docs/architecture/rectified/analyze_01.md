# Strategic Architecture Challenge Report — IBLOG_SERVICE

> **Changed 2026-08-03:** one amendment only — §5 🗑️ #1 (the base-model swap) is now **split by
> path**: adopted for French, still rejected for Darija. See the inline note there.
>
> **Everything else in this report stands**, and two of its calls were later vindicated against my
> own mid-session retraction: the residual French drift really is **refusal-scoped** (`analyze_02`
> wrongly "corrected" this; `analyze_04` restored it), and the deterministic-guardrail design is
> unchanged. Still current: the data-vs-code principle (§0), the probabilistic/deterministic
> boundary audit (§2), the quiz misdiagnosis (§2), retrieval language-affinity (§3), the system
> topology (§4), and the `tenant_id`-from-request-header **security bug** (§5 honourable mentions).

**Type:** Zero-based architectural challenge. Analysis only, no ADRs.
**Inputs:** `rectified-architecture-blueprint.md`, `architecture-failure-history.md`, `theory-guide` skill.
**Date:** 2026-08-03

---

## 0. Headline verdict

Your hypothesis is half right, and the wrong half is the expensive one.

**Right:** deterministic work was offloaded to probabilistic weights. That is real, it is the
root of most of your defects, and the rectified blueprint only half-fixes it — in places it
makes it worse.

**Wrong:** the base model. Atlas-Chat-9B is a **Gemma-2-9B derivative** (MBZUAI-Paris). Gemma-2-9B
is a strong French model. Your own failure history disproves the "we deprioritized French in the
base" theory twice over:

- The French-question-answered-in-Darija bug **reproduced on base Atlas-Chat** — your doc says so
  explicitly. It was caused by the *document's* language driving output, and was fixed with a
  router. Not a base-model capability gap.
- The residual French-refusal-in-Darija is caused by `grounded_refusal` being **417 rows, 100%
  Arabic-script**. A different base with the same dataset produces the same bug.

Swapping the base fixes neither and costs you Darija (DODa BLEU 28.08 → 0.92), the GREEN §4.1 gate,
and a finished second fine-tune (eval_loss 0.8407, above the base grounding floor).

**The actual root cause is deeper than either.** Your blueprint and your data pipeline both treat
"French output" as *a distribution to shift* rather than *an invariant to enforce*. You sold
corporate clients "100% French precision." No SFT mixture yields 100%. Only code does.

> **Data moves a distribution. Code enforces an invariant.**
> Anything you sold as a guarantee must live in code, not weights.

---

## 1. Verdict on base model & serving stack

### 1.1 Base model — KEEP Atlas-Chat-9B for tenant #1

| Axis | Keep Atlas-Chat-9B | Switch to Qwen2.5 / Mistral / Llama-3.1 |
|---|---|---|
| French | Inherits Gemma-2-9B French — strong | Qwen2.5 French good, Llama-3.1 good, Mistral good — **marginal gain at best** |
| Darija | 28.08 DODa BLEU | ~0.92 — effectively none |
| Your dataset | 3,064 rows of **behavior transfer** — works | Would need **language acquisition** — 3k rows cannot do this |
| Schedule | GREEN gate, 2nd fine-tune done | Full reset: regenerate, retrain, re-gate |
| Fixes the French bug? | — | **No.** Bug is data prior + retrieval contamination |

**Trade-off:** keeping Atlas costs you a slightly weaker French ceiling than a French-first base and
keeps you on Gemma-2's awkward serving profile (softcapping, sliding-window attention). It buys you
the only Darija competence available at 9B and preserves a working, gated pipeline. Rejected
alternative — Qwen2.5-14B + narrow Darija adapter — fails because Darija is not adapter-learnable
from a 3k behavior dataset; it is a pretraining-scale property.

**Condition on this verdict — the experiment that would prove me wrong.** You have a quantitative
Darija number (DODa BLEU) and **zero quantitative French number**. Your French position is a hunch
on both sides. Cheap disproof, ~half a day:

> Run base Gemma-2-9B, base Atlas-Chat-9B, and your adapter over 50 French regulatory Q&A.
> Score deterministically: French-purity (script + lexicon), citation grounding, refusal correctness.
> If Atlas-Chat's French is materially below stock Gemma-2-9B's, its Darija tuning taxed French and
> the swap debate reopens **with evidence**. Until then it is unfounded.

**Do not conflate this with the neutral-base question.** "Neutral multilingual base + per-tenant
LoRA" is a **tenant #2** decision, already recorded in your decisions table. Dragging it into MVP is
what makes this feel like a base-model crisis. It isn't one.

### 1.2 Serving stack — the blueprint is internally contradictory

The blueprint prescribes, in the same document:

- §2 diagram + Pattern 2: **vLLM stack with LoRA hot-swapping**
- §3B + §5 table: **pin Q4_K_M GGUF**

**These are mutually exclusive.** GGUF is a llama.cpp/Ollama format; vLLM's GGUF support is
experimental and does **not** support multi-LoRA hot-swap over a GGUF base. vLLM multi-LoRA needs an
unquantized (or specifically-supported-quant) base — for Gemma-2-9B that's ~18 GB fp16, which
matches your own record of an 18 GB merged export. Your dev box has **8 GB VRAM**. The blueprint's
serving tier does not run on your hardware and does not run on its own quantization advice.

Three further gotchas the blueprint waves past:
- **Gemma-2 attention.** Logit soft-capping + sliding-window attention have historically needed a
  specific vLLM attention backend (FlashInfer) to be numerically correct. Verify against your vLLM
  version before assuming parity with Ollama output.
- **"gRPC SSE Stream (Sub-50ms TTFT)"** is not a thing. gRPC and SSE are different transports, and
  50 ms is the *Redis LPOP* figure from the source pattern, not a 9B model's TTFT. Realistic TTFT on
  your hardware is 200–800 ms. The diagram is copying a pattern without checking the number.
- You already rejected vLLM once (no AWQ/GPTQ for Atlas, T4 PCIe). That rejection was for
  *generation*, but the quantization half of it applies to serving too.

**Recommendation — stop pretending it's one stack.**

| Phase | Runtime | Rationale |
|---|---|---|
| MVP / demo (now) | **Ollama, merged adapter, pinned `Q4_K_M`** | One tenant, one adapter. Multi-LoRA solves an N-adapter problem you do not have. Native structured outputs + GBNF grammars are already there. |
| Tenant #2+ | **vLLM `--enable-lora`, fp16 base, ≥24 GB (L4/A10G/A100)** | Only then does hot-swap earn its cost. |

**Trade-off:** Ollama-now costs you the throughput and multi-LoRA ceiling of vLLM and means a
migration later. It buys you a stack that actually runs on the hardware you own, today, with the
quantization you already validated. Rejected alternative — go straight to vLLM — requires buying/renting
a ≥24 GB GPU before you have a single paying tenant.

---

## 2. Probabilistic / deterministic boundary audit

| Job | Where it lives now | Where it belongs | Note |
|---|---|---|---|
| Output language | Weights (SFT prior) + prompt | **Code** — route in, filter retrieval, validate out | You proved prompts can't fix it (3 failed attempts) |
| Refusal decision | Weights (`grounded_refusal`, 417 rows) | **Mostly code** — it's a retrieval-confidence state | See §3 |
| Refusal language | Weights | **Code** — template in query language | 100% guaranteed, zero rows needed |
| Citation formatting | Already deterministic post-processing ✅ | Keep | Your best existing decision |
| Quiz JSON syntax | Weights + prompt | **Runtime grammar** (Ollama `format` / vLLM `guided_json`) | Native — no new dependency |
| Quiz answer-key correctness | Weights | **Code** — `_explanation_supports_answer()` | ⚠️ see below |
| Answer grading | Weights | **Code** — tool call to a real comparator | Blueprint gets this right |
| Tenant isolation | SQL filter | **Postgres RLS** + validated JWT claim | Blueprint gets this dangerously wrong — see §5 |

### The quiz misdiagnosis — the blueprint's most important error

Blueprint Symptom B calls the quiz failure "nondeterministic schema / schema corruption" and
prescribes Guidance/LMQL constrained decoding.

**Your failure history records no schema-corruption defect.** The actual, measured quiz defect was
**answer keys contradicting their own explanations, 16% (12/76)** — a learner told "correct" who was
in fact wrong. That is a **semantic** failure. Grammar-constrained decoding guarantees well-formed
JSON; it cannot make the answer key *true*. You would buy a framework, ship it, and the 16% defect
would survive untouched.

You already built the correct fix — `_explanation_supports_answer()`. It runs in the *generation*
pipeline. **Promote it to a serve-time and cache-admission gate.** No quiz enters the Redis buffer
or reaches a learner without passing it.

**On constrained decoding generally:** worth doing, but as a *runtime flag*, not a framework
purchase. Ollama takes a JSON schema in `format`; llama.cpp takes GBNF; vLLM takes `guided_json`.
Adding Guidance or LMQL as a dependency to reach a feature your runtime already exposes is exactly
the over-engineering the rest of this report is arguing against.

### And the irony worth naming

Blueprint §4.2 makes **"Language Match" a 1–5 LLM-as-a-judge score at 30% weight.** Language match is
a regex. You already have script detection at `app/services/citations.py:262`. The document that
correctly diagnoses "you offloaded deterministic work to the model" then proposes to offload a
*regex* to GPT-4o. Same for schema validity — that's a parser, scored 0 or 1, not a judge.

---

## 3. Data pipeline root cause & the rebalance

**Why French refusals revert to Darija.** `grounded_refusal` is 417 rows, 100% Arabic-script. The
conditional distribution P(script | intent=refuse) has **zero mass on French**. In a 9B model, a
prompt cannot summon a behavior with zero training support — which is precisely what your three
failed attempts (instruction, hard constraint, exemplar) demonstrate. The exemplar attempt is the
tell: the model *copied the exemplar* and still answered in Darija. That is a prior overwhelming
in-context evidence.

**Fix in this order. The order is the whole point.**

1. **Code (the invariant).** A refusal is largely a *system state*, not a model judgment: retrieval
   returned nothing above threshold. Detect it in the orchestrator and emit a **templated refusal in
   the query's language**. Guaranteed 100%, costs zero training rows, ships this week.
2. **Retrieval (the contamination).** A French query must not be handed Arabic-only context — the
   blueprint's own Symptom A secondary cause. Add a language-affinity term to retrieval: prefer
   same-language chunks, and when only cross-language chunks pass the threshold, mark the context
   and force the French system template. Skipping this re-contaminates everything fixed in step 1.
3. **Data (the naturalness).** *Then* add French refusal rows — for the residual judgment-call
   refusals ("documents retrieved but they don't answer this"), which templates can't cover.

**Budget correction.** Blueprint says *add* 200 FR + 100 EN rows on top of 3,064. Your own record
says the corpus ceiling, LoRA capacity, and review capacity all cap at ~3,000, and that
`grounded_refusal` **over-generated** and drove 20% dedup loss. So: **convert within the component,
don't append.** 417 → roughly 250 Darija / 165 French. Total row count unchanged, ceiling respected.

**Cut English from the MVP.** Your MVP ground truth is French primary, Darija secondary. English is
"in scope" but not an MVP feature. 100 English SFT rows + 20 English golden rows spend a capped
budget on a language no MVP feature requires.

**⚠️ The FLAN anchor is backwards.** Blueprint §3A prescribes "10% FLAN or OpenAssistant" as a
catastrophic-forgetting anchor. **FLAN and Alpaca are English-dominant.** Injecting 10% English
tokens into a French-primary model that is *actively losing a language-drift fight* is the wrong
regularizer — you would be adding a third attractor to a system that can't hold two. If you want an
anchor, use a French/multilingual instruction set. Better for MVP: skip the anchor, and detect
forgetting empirically via the golden set instead of pre-medicating for it.

**⚠️ Unaddressed and higher-priority than anything in the blueprint: train/serve skew on Socratic.**
Your history logs a **73% train/serve RAG context mismatch — socratic and code_switching were trained
with no real document**, but at serve time there is always a document. That is **MVP feature #2
trained off-distribution.** The blueprint never mentions it. This outranks the entire warm-path
caching section.

---

## 4. Proposed system topology

Language appears **three times** — route, constrain, verify. That is what makes it an invariant
instead of a hope. Constrained decoding is *inside* the runtime, not a service.

```
 CLIENT ── query + JWT
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│ GATEWAY                                                          │
│  1. AuthN/Z  → tenant_id from VALIDATED JWT CLAIM (never header)  │
│  2. set_config('app.tenant_id') → Postgres RLS session            │
│  3. LANG DETECT ①  fasttext + script regex → {fr, ary, en}        │
│  4. PII scrub, token cap, rate limit                              │
└─────────────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│ RETRIEVAL  (pgvector, RLS-enforced)                              │
│  • tenant partition / per-tenant partial HNSW index               │
│  • LANG AFFINITY ②  prefer same-language chunks; flag if none     │
│  • confidence threshold ──► BELOW? ──┐                            │
└──────────────────────────────────────┼──────────────────────────┘
   │ above threshold                    │
   │                                    ▼
   │                          ╔══════════════════════════╗
   │                          ║ DETERMINISTIC REFUSAL     ║
   │                          ║ template in query lang    ║──► CLIENT
   │                          ║ (no LLM call, 100% lang)  ║
   │                          ╚══════════════════════════╝
   ▼
┌─────────────────────────────────────────────────────────────────┐
│ ORCHESTRATOR — task router                                       │
│   chat/socratic → stream   │  quiz → LPOP warm cache (later)      │
│   grading → TOOL CALL to real comparator, never the model         │
│   system template selected by LANG ① (FR / AR)                    │
└─────────────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│ MODEL RUNTIME   Ollama (MVP) → vLLM+LoRA (tenant #2+)            │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │ CONSTRAINED DECODE — native (`format` / GBNF / guided_json)│   │
│  │ grammar active for quiz & any structured task              │   │
│  └───────────────────────────────────────────────────────────┘   │
│  Atlas-Chat-9B + merged LoRA, pinned Q4_K_M                       │
└─────────────────────────────────────────────────────────────────┘
   │  token stream
   ▼
┌─────────────────────────────────────────────────────────────────┐
│ POST-GENERATION VALIDATOR   ← the layer the blueprint omits       │
│  • LANG VERIFY ③  script + lexicon vs LANG ① → mismatch = repair  │
│  • citation post-processing (CiteFix)                             │
│  • citation ⊆ retrieved context, else strip + flag                │
│  • quiz: schema parse + _explanation_supports_answer()            │
│  • fail ⇒ ONE bounded repair retry ⇒ else deterministic fallback  │
└─────────────────────────────────────────────────────────────────┘
   │
   ▼  SSE (plain HTTP/2, not "gRPC SSE")
 CLIENT
```

Streaming note: post-generation validation and token streaming are in tension. Practical
resolution — stream freely for chat/socratic but run the language check on the **first ~20 tokens**
and abort-and-restart on mismatch (cheap, catches drift at the point it happens); buffer fully for
quiz and any structured output, which are precomputed anyway and not latency-sensitive.

---

## 5. Top 3 things to throw in the trash immediately

### 🗑️ 1. The base-model swap — including the premise behind this whole review

> ⚠️ **AMENDED 2026-08-03 — this verdict was right to demand evidence, wrong to treat "the base
> swap" as one indivisible decision.** The benchmark demanded below was run
> (`analyze_04_corrected_drift.md`), and it **splits the question by path**:
> - **French → swap ADOPTED.** Stock `gemma2:9b` scored **12/12 `PASS_FR`** zero-shot, avoiding both
>   of Atlas's documented failure modes. Combined with the owner's UI-dropdown decision, French now
>   routes to Gemma. This part of the verdict is overturned **by the measurement it asked for.**
> - **Darija → swap STILL FIRMLY REJECTED**, and now on direct evidence rather than BLEU inference.
>   Gemma's Darija response *scored* `PASS_ARY` but was **incoherent on inspection** — it rephrased
>   the question twice without answering. A script-ratio metric cannot tell "Darija tutoring" from
>   "Arabic-script non-answer." **Darija → `IBLOG_TUTOR`, always.**
>
> **The cost argument below survives intact and has simply changed job:** a second base *does*
> double the data pipeline — which is exactly why the French **LoRA** stays deferred and gated
> (ADR 0001 Stage 2), even though the French **base** is adopted. Adopting a base ≠ fine-tuning it.

*Original argument, retained for history:*

This is the controversial one because you asked me to endorse it. The evidence in your own
`architecture-failure-history.md` contradicts it: the French bug **reproduced on base Atlas**, and
the refusal bug is a 417/417 data prior. Both survive a base swap intact. You would burn the GREEN
gate, the finished second fine-tune, and your only Darija competence to fix nothing. Kill it — and
replace it with the 50-row French benchmark in §1.1, which is the only thing that could legitimately
revive it.

### 🗑️ 2. `grounded_refusal` as a *learned behavior* — and the +300-row patch as the primary fix

Three failed prompt attempts and a proposed 300-row data patch, all to make a model reliably choose
a language for a message that is **mostly a templated response to a retrieval-confidence threshold**.
You have been training a 9B model to do an `if` statement. Refusal *language* becomes a code
invariant (100%, this week); refusal *rows* get rebalanced within the existing 417 for the residual
judgment cases, not appended on top of a capped budget.

### 🗑️ 3. The SQS/Redis warm-path quiz cache — and the Guidance/LMQL purchase

Both are answers to problems you do not have. The warm path optimizes p99 latency for **zero
users**, and imports SQS into a stack that is Postgres + Redis with no AWS dependency (use Redis
Streams or a Postgres queue if you ever need it). Constrained decoding targets *schema corruption*
you never measured, while your real, measured 16% answer-key defect is semantic and untouched by
grammars. Ship the semantic gate at serve time; take constrained decoding as a one-line runtime
flag; defer the cache until a real tenant has real latency complaints.

### Honourable mentions

- **🔴 Security bug in the blueprint.** §2 Pattern 1.3 derives `tenant_id` **from request headers**.
  If that header is client-controllable, that is cross-tenant data access — the exact catastrophe the
  section claims to prevent. It must come from a **validated JWT/session claim**, and the filter
  should be backed by **Postgres Row-Level Security** so a forgotten `WHERE` clause fails *closed*
  rather than leaking. Defense in depth, not a single hand-written predicate.
- **pgvector + HNSW + `WHERE tenant_id`** as written under-returns: HNSW post-filters, so a tenant
  with few chunks can silently get <5 results. Use per-tenant partial indexes, table partitioning,
  or pgvector 0.8+ iterative scans.
- **Judge can't read Darija.** GPT-4o/Claude scoring Darija groundedness is a weak signal, and
  shipping tenant regulatory documents to a third-party judge contradicts the isolation posture the
  same document argues for. Use deterministic gates for groundedness/language/schema; reserve the
  LLM judge for Socratic *tone* only — the one genuinely subjective axis.
- **Reuse, don't rebuild.** The blueprint proposes a CI eval harness from scratch. You already own
  ~10 enforced gates in `generate_training_data.py`, script detection in `citations.py`, and
  `_explanation_supports_answer()`. The CI gate is those, pointed at a golden set.
- **The blueprint omits audio entirely.** MVP feature #4 — Darija STT + TTS — appears **zero times**
  in a document titled "Rectified Architecture Blueprint," while your own notes call TTS selection
  the **single largest open MVP item**. Darija TTS is the highest-risk unknown in the whole program
  and it is unarchitected. That gap outranks every optimization in the blueprint.
- **`theory-guide` skill carries its sources' assumptions** — "L1 cache in JVM local memory," SQS,
  Java-centric serving. You are Python/FastAPI/Postgres/Redis. Read it as patterns, translate the
  substrate.

---

## 6. Suggested sequencing (not ADRs — a proposed order to argue about)

1. Language invariant in code: router ① + retrieval affinity ② + post-gen verify ③ + templated refusals.
2. Serve-time gates: `_explanation_supports_answer()` + schema parse + citation ⊆ context.
3. Tenant isolation hardening: JWT-claim tenant_id + Postgres RLS + partial HNSW indexes.
4. The 50-row French benchmark (§1.1) — settles the base-model question with data.
5. Fix socratic train/serve skew (regenerate socratic/code_switching **with** real RAG context).
6. Rebalance `grounded_refusal` within budget (417 → ~250 ary / ~165 fr). Drop English.
7. Golden set + deterministic CI gate, reusing existing gate code. LLM judge for tone only.
8. **Audio spike** — Darija TTS vendor/model selection. Start now; it has the longest lead time.
9. Defer: warm-path cache, vLLM multi-LoRA, neutral base — all tenant #2 concerns.
