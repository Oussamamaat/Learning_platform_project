# Empirical French-Drift Test — Results

**Date:** 2026-08-03 · **Hardware:** RTX 4060 Laptop, 8188 MiB · **Ollama** 0.32.4
**Harness:** 5 prompts × 2 models × 3 runs = 30 generations, `temperature 0.2`, `num_predict 350`
**Scoring:** deterministic only (Unicode-block ratios + citation-span whitelisting). No LLM judge.

> 🔴 **Changed 2026-08-03 — READ `analyze_04_corrected_drift.md` FIRST.**
> **The 0/12 headline in §1 is retracted.** This harness posted directly to `/api/generate` with a
> naive system prompt, bypassing `generate_llm_response()` — the real serving path, which selects
> `SYSTEM_PROMPT_TEMPLATE_FR` and calls `detect_query_language()`. Re-run through production code,
> `IBLOG_TUTOR` produces **clean French 9/9** when context actually answers the question.
> **Superseded below:** Correction 1 (drift is *not* total — it is confined to refusals and
> Arabic-context contamination), Correction 3 (premise was the retracted 0/12), Correction 4
> (guardrails have a working generator to fall through to).
> **Still valid:** §1/§2 raw data as a record of *that* harness; Correction 2; the schema-enforcement
> finding; the TTFT finding. §3's dual-model verdict is amended in place — see below.
>
> ⚠️ *Original note, retained for history:* "This result contradicts `analyze_01.md`. That report
> stated the French routing bug 'was fixed with a router' and only *refusals* still drifted. That is
> wrong at serve time." — **This retraction was itself wrong.** `analyze_01.md` was right: the
> residual really is refusal-scoped. Corrections are marked ❗ below.

---

## 1. Headline numbers

| Model | French pass | Darija pass | Median latency |
|---|---|---|---|
| `IBLOG_TUTOR:latest` (tuned) | **0 / 12 (0%)** | 3 / 3 (100%) | 8.8 s |
| `Atlas-Chat-9B-GGUF:Q4_K_M` (base) | **2 / 12 (17%)** | 3 / 3 (100%) | 7.7 s |

**Every French prompt was sent with an explicit French system prompt containing:**
> *"Réponds UNIQUEMENT en français. N'utilise jamais l'arabe ou le darija."*

It was ignored 12/12 times by the tuned model, at **temperature 0.2** — i.e. near-greedy decoding.

## 2. Per-prompt breakdown

| Prompt | Tuned | Base | Reading |
|---|---|---|---|
| **P1** simple FR instruction | 3/3 DRIFT | 3/3 DRIFT | Not a refusal edge case — *basic French instruction-following is gone* |
| **P2** FR refusal, no context | 3/3 DRIFT | 3/3 DRIFT | ❗ Base drifts **identically** — see §3 |
| **P3** FR + JSON schema | 3/3 DRIFT | **2/3 PASS_FR**, 1 MIXED | ❗ The adapter **destroyed** French the base still had |
| **P4** FR + Arabic context | 3/3 DRIFT | 3/3 DRIFT | Contamination confirmed — but additive, not causal |
| **P5** Darija tutoring | 3/3 PASS | 3/3 PASS | ✅ Non-negotiable #2 is **satisfied today** |

### Evidence — P2 refusal, tuned vs base

```
TUNED:  سمح ليا، ولكن هاد المعلومة ماكايناش فالمعلومات لي عطيتيني...
BASE:   سمح ليا، ولكن ما نقدرش نعطيك هاد المعلومة حيت ماشي جزء من السياق...
```
Same language, same register, same refusal. The behaviour is **inherited from Atlas-Chat**.

### Evidence — P3, the adapter regression

```
BASE (PASS_FR):  {"question": "Quel équipement de protection individuelle est utilisé
                  pour protéger les yeux et le visage ?", "options": ["Masque", ...]}

TUNED (DRIFT):   {"questions": [{"question": "شنو هي الوظيفة الرئيسية ديال معدات
                  الحماية الشخصية (EPI)؟", ...
```
The base produced clean French JSON. The fine-tune produced Darija JSON **and** invented a
`{"questions": [...]}` wrapper the schema never asked for.

---

## 3. What this changes

### ❗ Correction 1 — French drift is total, not residual
`analyze_01.md` scoped this to `grounded_refusal`. **P1 shows a plain French instruction fails too.**
The served model has no working French path at all. The `detect_query_language()` +
`SYSTEM_PROMPT_TEMPLATE_FR` fix exists in the generation pipeline and was never wired into serving.

### ❗ Correction 2 — the 417 Darija refusal rows are not the root cause
The blueprint blamed the 417 rows exclusively. **Base Atlas-Chat drifts identically on P2 with zero
exposure to your dataset.** Your 417 rows *reinforced* a pre-existing Atlas-Chat prior; they did not
create it. Rebalancing them therefore cannot fully fix this — it removes your contribution to a
defect you inherited.

### ❗ Correction 3 — my "keep Atlas because it's Gemma-2-based" argument is now in doubt
I argued Atlas-Chat inherits Gemma-2-9B's strong French. **The base scores 17% French.** Atlas-Chat's
Darija instruction-tuning appears to have installed a near-unconditional Darija output prior that
overrides input language *and* explicit system instructions. This is the experiment I said would
prove me wrong, and it partially did.

**The decisive missing measurement is stock `gemma2:9b` on the identical prompt set.**
- If Gemma-2 passes French at high rate → the culprit is precisely Atlas-Chat's Darija tuning, and
  the base-model question genuinely reopens.
- If Gemma-2 also drifts → the cause is the Gemma-2 chat template / prompt plumbing, and Atlas is
  exonerated.

Until that runs, **do not** treat "keep Atlas" as settled. Cost: one ~5.4 GB pull + ~6 min.

### ❗ Correction 4 — guardrails make the failure *safe*, not *fixed*
I proposed: post-generation validator → one repair retry → deterministic French fallback. At a **0%**
French generation rate, that pipeline hits the templated fallback essentially every time. That is a
canned-response bot, not a tutor. **The guardrail is still necessary — it is what converts a silent
wrong-language answer into a controlled one — but it is not sufficient.** French *generation
capability* has to be restored underneath it.

### ✅ Confirmed — schema enforcement does earn its place
`analyze_01.md` dismissed "schema corruption" as unmeasured. It is now measured: 2/3 tuned and 1/3
base runs failed to parse, and the tuned model invented a wrapper key. Ollama's native `format`
parameter fixes the structural half. Still a runtime flag, **not** a Guidance/LMQL dependency.

### ⚠️ AMENDED 2026-08-03 — "dual-model routing is dead on this hardware"

**The arithmetic stands; the verdict was too broad.** 8188 MiB total. Atlas-9B Q4_K_M ≈ 5.8 GB +
~1 GB KV ≈ 7 GB. A second 9B model **cannot co-reside** — that part is correct and unchanged.

But co-residency is not the only dual-model topology. **Serial loading — exactly one model resident
at a time, selected by the UI language dropdown — fits comfortably** (~7 GB of 8.19 GB, ~1 GB
headroom) and is now the **adopted MVP topology** (owner decision, 2026-08-03): French →
`gemma2:9b`, Darija → `IBLOG_TUTOR`, never simultaneously.

So the cost is not "impossible," it is **a ~30 s model swap on language switch** — the cold-load
figure this very table measured (34–38 s first call vs 5–9 s warm) is precisely that swap cost.
See `analyze_04_corrected_drift.md` §"Serial-load feasibility" for VRAM math, Ollama configuration,
and demo UX handling.

### ✅ Confirmed — "sub-50 ms TTFT" is fiction
Warm generations ran 5–21 s for 350 tokens (~17–60 tok/s).

---

## 4. Revised recommendation

**Ordering changes.** Restoring French generation now outranks building the guardrail, because the
guardrail has nothing to fall through to.

1. **Run stock `gemma2:9b` on this same harness.** ~6 min. It is the single highest-information
   measurement available and it gates the base-model decision. Nothing else should be decided first.
2. **Re-test the tuned model with a French *few-shot prefix*** (2–3 French Q→French A exemplars in
   the prompt), not just an instruction. Your history says exemplars failed for refusals; this
   checks whether that holds for ordinary answers. Cheap, and it bounds what prompting can achieve.
3. **Guardrail layer** — still build it (asymmetric FR/Darija validators, citation-span whitelisting,
   deterministic terminal fallback). It converts a silent defect into a visible, controlled one.
4. **Deterministic refusal path** — retrieval-miss → templated refusal in query language, no LLM call.
   Unaffected by all of the above; ships independently.
5. **Then** decide the base question with data from step 1, and rebalance SFT accordingly.

**Unchanged:** Qwen2.5 base-swap stays rejected — it violates non-negotiable #2 (0.92 vs 28.08 DODa
BLEU), and P5 confirms Darija currently works and is the thing you would be trading away.

---

## 5. Scope and caveats

- n=3 per cell, 5 prompts, one tenant domain. Directional, not a benchmark. The 0/12 vs 2/12 gap is
  large enough to act on; the P3-only base advantage is **not** — it rests on 2 samples.
- System prompt was delivered via the Ollama `system` field. Gemma-2 has **no system role**, so the
  template folds it into the user turn — the instruction competes with user text rather than
  outranking it. This is a property of the model family, not a harness bug, and it is part of why
  prompt-based language control is structurally weak here.
- Harness + raw outputs: `scratchpad/fr_drift_test.py`, `scratchpad/results.json`.
