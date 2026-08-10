# Corrected French-Drift Measurement (via production code path)

> **Changed 2026-08-03:** appended the **Laptop MVP (serial dual-path)** amendment at the end of
> this file — serial-load VRAM math and swap latency (§A), the Ollama configuration and demo UX for
> the language dropdown, and the fine-tuning answer (§B: stock Gemma for the demo, LoRA later for
> *behavior* not language). The measurements in the body are unchanged; only the decisions layered
> on top of them have moved. Also records one **open gap**: Darija-query-against-French-context
> contamination is untested.

**Supersedes the severity claim in `analyze_02_test_results.md`.** That test called raw
`/api/generate` directly with a naive system prompt, bypassing `app/services/llm.py` entirely —
missing `SYSTEM_PROMPT_TEMPLATE_FR`'s load-bearing anti-Darija-refusal line (`llm.py:77-79`) and
`detect_query_language()`'s routing. This re-runs the same intent through the **actual production
function**, `generate_llm_response()`.

## Corrected results (`IBLOG_TUTOR`, production path, n=3/cell)

| Prompt | Verdict | What it actually tests |
|---|---|---|
| P1 (context did not answer the question) | 3/3 `DRIFT_DARIJA` | **Refusal**, not plain instruction-following — see below |
| P2 (unanswerable, salary question) | 3/3 `DRIFT_DARIJA` | Refusal — matches documented residual bug exactly |
| P3 (JSON quiz, matching FR context) | **3/3 `PASS_FR`** | Structured French generation — works |
| P4 (FR query, **Arabic**-script context) | 3/3 `DRIFT_DARIJA` | Context-contamination path (blueprint Symptom A secondary cause) |
| P5 (Darija) | 3/3 `PASS_ARY` | Non-negotiable #2 — unaffected |
| **Supplementary: P1 re-run with context that actually answers it** | **6/6 `PASS_FR`** | Isolates the confound |

## What changed

**My original P1 was confounded.** The context I supplied (general PPE/employer-obligation text)
did not actually answer "explain electrical safety in a factory" — the model correctly judged it
ungrounded and refused. The refusal came out in Darija, which is real and matches the documented
bug, but I had mis-labeled it a "plain instruction-following failure." It is a refusal-language
failure. When I re-ran the identical instruction with context that genuinely answers it: **6/6
French, clean.**

**The 0/12 in `analyze_02.md` was an artifact of the harness, not the model.** Through the real
production path, with matching context, French generation is not broken. That claim, and my
"Correction 1 — French drift is total" note in `analyze_02.md`, is **retracted**.

## What is confirmed real (unchanged from `analyze_01.md`'s original diagnosis)

Two, and only two, mechanisms drift to Darija — both already documented in
`architecture-failure-history.md`, both reproduced here through the real serving code:

1. **Refusal language.** Any refusal — whether context is totally absent (P2) or present but
   non-responsive (P1's original form) — reverts to Darija regardless of query language, despite
   the explicit anti-Darija instruction in `SYSTEM_PROMPT_TEMPLATE_FR`. Root cause unchanged:
   `grounded_refusal` is 417 rows, 100% Arabic-script; the prior beats the prompt.
2. **Context-script contamination.** A French query against Arabic-script retrieved context (P4)
   pulls the answer into Darija even though the question is answerable and the instruction is
   explicit. This is real and is not a refusal — the model engaged with the content, in Darija.

**Everything else works.** Plain French instructions with matching French context: clean. French
JSON quiz generation with matching context: clean. Darija: unaffected.

## Consequences for the architecture decisions already made

- **The dual-path (Gemma-2-9B French base) decision in `analyze_03` was made on the wrong severity
  premise** (0/12, "total collapse"). The corrected picture — a working generator with two narrow,
  well-understood failure modes — is arguably fixable **without a second base model or a second
  fine-tune**, via the deterministic guardrail + retrieval-affinity + refusal-template fixes already
  designed in `analyze_01.md` §3–4. **The Gemma-2-9B test is still worth running** (below) because it
  answers a different, still-open question — whether Gemma-2 is a *stronger* French ceiling — but
  it is no longer rescuing a broken generator. It's an upgrade decision, not a repair decision.
- **The guardrail design does not change.** Deterministic refusal-in-query-language (kills failure
  mode 1 outright, zero training rows) and retrieval language-affinity (kills failure mode 2) remain
  exactly as designed in `analyze_01.md` §3–4 and `analyze_03.md` §6 step 2. If anything, this
  measurement **strengthens** confidence in that fix, because the failure surface is now known to be
  narrow and structural rather than a diffuse capability gap.
- **Recommendation: build the guardrail + refusal-template + retrieval-affinity fix first, on the
  existing `IBLOG_TUTOR`, and re-measure before committing to a second French fine-tune.** The
  French-fine-tune workstream (`analyze_03` §5 step 3, the multi-week item) may turn out to be
  unnecessary or reducible to a much smaller "refusal + contamination" data patch rather than a
  full second SFT set — pending the Gemma comparison and pending a post-guardrail re-measurement.

## Gemma-2-9B comparison (complete)

Stock `gemma2:9b`, zero fine-tuning, same prompt-building logic (`_build_system_prompt` +
`detect_query_language`), n=3/cell:

| Prompt | Verdict |
|---|---|
| P1 (refusal, mismatched context) | **3/3 `PASS_FR`** |
| P2 (refusal, unanswerable) | **3/3 `PASS_FR`** |
| P3 (JSON quiz) | 3/3 `PASS_FR` |
| P4 (Arabic-context contamination) | **3/3 `PASS_FR`** |
| P5 (Darija) | 3/3 `PASS_ARY` — **see caveat below, this is not a clean pass** |

### Reading — French: this is real signal

Stock Gemma-2-9B avoids **both** documented failure modes zero-shot. P4 is the clearest case: given
Arabic-script context, it answers in French anyway — including translating the Arabic legal
citation into French prose ("l'article 12... la loi numéro 27.06"). This locates the cause of both
failure modes specifically in **Atlas-Chat's Darija SFT overriding the instruction**, not in the
Gemma-2 architecture or chat template. That's a materially different, better-supported conclusion
than "the model can't do French" — it's "a differently-tuned model doesn't fight the instruction."

⚠️ One side effect worth flagging, not treating as blocking: Gemma paraphrased the legal citation
rather than reproducing it verbatim, which conflicts with the "cite exactly as written" requirement.
Production `generate_llm_response()` does not trust model-generated citations — `extract_citations()`
+ `inject_citations()` (`llm.py:216-228`) derive citations deterministically from retrieved context
and overwrite whatever the model wrote. This likely neutralizes the issue, but it should be verified
against Gemma's actual output shape before relying on it, not assumed.

### Reading — Darija: the "pass" is a false positive, and it matters

Raw P5 output: *"واش تقول لي، ماشي كاين شي مخاطر ديال الكهربا فالمعمل؟ تقول لي، كيفاش يمكن نتعرف على
المخاطر ديال الكهرباء فالمعمل؟"* — two rephrasings of the question back at the user, no answer, no
refusal, no Socratic engagement. It scores `PASS_ARY` because it uses Darija-flavored function words
(`واش`, `ديال`, `كاين` — likely present in Gemma-2's pretraining web corpus) and Arabic-script
dominance clears the 0.6 threshold. **The deterministic script-ratio metric cannot distinguish
"genuine Darija tutoring" from "incoherent non-answer that happens to use Arabic script and a few
Darija particles."** This is the same category of measurement gap the project already hit once
(`architecture-failure-history.md`: "Darija detector blind to ك-prefix... a real sentence scored
`dar=0` and was misread") — here it cuts the other way, a broken response scoring as a pass.

`IBLOG_TUTOR`'s Darija responses across every test in this session, by contrast, are coherent,
on-topic, and correctly follow the trained refusal/Socratic patterns. **This is not a contest Gemma
wins by matching Atlas's DODa BLEU (28.08) — it reconfirms Atlas's Darija SFT is doing real,
necessary work that a script-ratio check cannot see, let alone replace.**

### Conclusion this measurement supports (original — see laptop-MVP amendment below)

Not "swap to Gemma." Rather: **the French failure modes are narrow, instruction-following artifacts
of Atlas's Darija tuning — likely fixable without a second base model at all** via the guardrail +
deterministic-refusal + retrieval-affinity design already specified in `analyze_01.md` §3–4. If a
post-guardrail re-measurement still shows a gap, a Gemma-2-9B + French LoRA dual-path
(`analyze_03.md`) remains a sound *fallback*, and this test is encouraging about how little
correction that base would need — while Atlas-Chat/`IBLOG_TUTOR` continues to own Darija exclusively,
now on direct evidence rather than inference from BLEU alone.

---

# Amendment 2026-08-03 — Laptop MVP (serial dual-path)

**Changed:** the owner has set new ground truth — the MVP and CEO demo run on the laptop (RTX 4060,
8 GB, Ollama) with **serial** one-model-at-a-time routing driven by a `fr`/`ary` dropdown. The
conclusion above ("not swap to Gemma") is **superseded as a topology decision**: French now routes
to `gemma2:9b`, Darija to `IBLOG_TUTOR`. What the measurement *showed* is unchanged; what is being
decided on top of it has changed.

## Why this measurement supports the laptop MVP

- **French → stock Gemma is viable for the demo without any fine-tuning.** 12/12 `PASS_FR` zero-shot,
  clearing both of Atlas's documented failure modes. No French LoRA is needed to demo.
- **Darija → Gemma is permanently barred.** The `PASS_ARY` is a measured false positive (incoherent
  content clearing a script-ratio check). This is not a close call and should not be revisited
  without a coherence-aware metric.
- Routing French away from Atlas **architecturally eliminates** Atlas's two French failure modes —
  they are routed around, not fixed. The guardrails survive for different reasons (below).

## A. Serial-load feasibility on the laptop

### VRAM
One 9B Q4_K_M ≈ **5.8 GB weights + ~1 GB KV @ `num_ctx 4096` ≈ 7 GB** of 8188 MiB — fits with
roughly 1 GB headroom. Two co-resident would need ~14 GB and does not fit (this is what
`analyze_02` §3 correctly measured; that verdict is amended, not reversed — see that file).

⚠️ **Do not raise `num_ctx` to 8192.** KV roughly doubles to ~2 GB, putting total at ~7.8 GB — at
the edge of an 8.19 GB card that also serves the Windows desktop. The existing Modelfiles already
pin `num_ctx 4096`; leave them.

### Swap latency
Measured in this session: **34–38 s cold first call vs 5–9 s warm**. Model load is the dominant term
(~25–30 s). That is the price of every language switch.

### Ollama configuration — emergent vs contractual
Ollama evicts under VRAM pressure, so with two 9B models on 8 GB you *get* serial behavior by
default — but as a **side effect of memory pressure, not a guarantee**, and eviction timing is not
deterministic. Make it explicit:

| Setting | Value | Why |
|---|---|---|
| `OLLAMA_MAX_LOADED_MODELS` | `1` | Makes serial loading contractual rather than emergent |
| `OLLAMA_KEEP_ALIVE` | long (e.g. `60m` or `-1`) | Prevents the resident model unloading during demo Q&A pauses — a silent 30 s stall mid-demo is the worst failure mode |

### Demo UX for the dropdown
1. **Preload at app start**: fire a warm-up call (`num_predict: 1`) for the dropdown's default
   language so the first real question is warm.
2. **On switch**: immediately show an explicit loading state ("Chargement du modèle… ~30 s"). Never
   leave a live-looking input box in front of a model that is not resident — a user typing into a
   dead box for 30 s reads as a crash.
3. **Disable send** while swapping; re-enable on warm-up completion.
4. **Demo tactic:** script **at most one** language switch, at a planned narrative beat, and talk
   over the load. The stall is unavoidable on this hardware; it is manageable if it is expected.

⚠️ **Concurrency hazard:** any background job (e.g. quiz generation) requesting the *other* model
while a user is chatting will thrash — evict, load, evict. On the laptop, background generation must
be disabled or strictly serialized. See ADR 0003.

## B. Does the French path need a LoRA?

> 🔴 **REVERSED 2026-08-03 — "Demo: no" is wrong. See `analyze_05_french_finetune_plan.md`.**
> This section optimized for *language compliance* (12/12 `PASS_FR`) when the MVP bar is *adaptive
> learning tutoring in both languages*. Stock Gemma answers questions; it does not tutor. Shown next
> to a fine-tuned `IBLOG_TUTOR`, the asymmetry would undercut the product claim in front of the CEO.
> **The French LoRA is in scope for the demo** (owner decision), scoped to ~1,800 rows over a 3–5
> week window. The capability table below remains accurate and is in fact the *justification* for
> fine-tuning — read the "Needs LoRA?" column as the work list, not as deferrable.

**Original recommendation (superseded): Demo: no. Production: yes — for behavior, not language.**

| Capability | Stock Gemma-2-9B | Needs LoRA? |
|---|---|---|
| French language compliance | ✅ measured 12/12 | No |
| Correct French refusal | ✅ measured (P1/P2) | No |
| Arabic-context handling | ✅ measured (P4) | No |
| **Verbatim legal citations** | ❌ **paraphrased** («la loi numéro 27.06») | Mitigated by `inject_citations()` — **verify**, don't assume |
| **Socratic register** | ⚠️ incidental, untrained | Yes, for production |
| **Refusal conventions** | ⚠️ correct language, untrained register | Yes, for production |
| **Quiz conventions** | ❌ structure via `format`, pedagogy untrained | Yes, for production |

**SFT scope — now committed, not hypothetical:** neither the full ~3 k 11-component set nor the
refusal+contamination patch. That patch was designed for *Atlas's language drift*, which Gemma does
not have. The correct scope is a **behavior-focused French set** reusing the existing component
architecture and dropping Darija-specific components (`code_switching`, `darija_preservation`) —
**~1,800 rows across 8 components.** Full config, pipeline gate changes, timeline, and acceptance
gate: **`analyze_05_french_finetune_plan.md`**.

**Export path — changed from `analyze_03`.** On the laptop the French model follows the **identical
Atlas path**, not vLLM: LoRA (Kaggle T4, r=16/α=16, 7 linear targets, fp16) → merge fp16 → GGUF →
**Q4_K_M** → `ollama create`. Same architecture ⇒ the entire existing convert/quantize toolchain is
reused unchanged. Two gotchas: **chat-template parity** (reuse the same `.System`-folded-into-user-turn
convention — this project has already been bitten by template mismatch) and **~18–25 GB transient
disk** for the merged fp16 export.

## Open gap this amendment creates

⚠️ **Darija-query-against-French-context contamination is untested** — the mirror of P4, never run.
Atlas still serves Darija, and French-language source documents exist in the corpus. Worth one
harness run before the demo.
