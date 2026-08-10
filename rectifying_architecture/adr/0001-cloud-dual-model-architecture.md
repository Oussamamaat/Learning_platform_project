# ADR 0001: Model Architecture (French + Darija) — Laptop MVP → Cloud

**Status:** Amended 2026-08-03 — staged decision; topology now fixed, fine-tune still gated
**Date:** 2026-08-03
**Depends on:** `analyze_01.md`, `analyze_04_corrected_drift.md` (both empirical tests), ADR 0002

> **Changed 2026-08-03.** Scope widened from cloud-only to **laptop MVP → cloud**. The owner has
> decided the MVP and CEO demo run on the laptop (RTX 4060, 8 GB, Ollama) with **serial**
> one-model-at-a-time routing: French → stock `gemma2:9b`, Darija → `IBLOG_TUTOR`, never both loaded.
>
> **The key distinction this amendment introduces:**
> the laptop dual-path adopts **Stage 3's topology early, but not for Stage 3's trigger reason.**
> Stage 3 below was written as *remediation* for Atlas's French failure. The laptop dual-path is a
> **product decision** (explicit dropdown, best-French-model-for-French).
>
> 🔴 **Second amendment, same day — the French LoRA is now COMMITTED, not gated.**
> An earlier version of this note said "fine-tuning Gemma: still gated." **That was wrong**, and the
> owner corrected it: the MVP promises *adaptive learning tutoring in both languages*, so a stock
> French model that answers but does not tutor fails the MVP scope regardless of its language score.
> Both the topology **and** the French fine-tune are in scope for the demo.
> - **Adopting Gemma as the French base:** decided.
> - **Fine-tuning Gemma with a French LoRA:** **committed** — ~1,800 rows, 3–5 weeks. See
>   `analyze_05_french_finetune_plan.md`.
> - **Darija stays `IBLOG_TUTOR`, untouched, in every scenario.**
>
> Stages 1–3 are retained as written, with Stage 2's gate redefined and Stage 3 recommitted.

## Context

This ADR was explicitly held pending two measurements, because the first attempt at it
(`analyze_03.md`) was built on a since-retracted premise: a naive test harness reported `IBLOG_TUTOR`
generating French 0/12 times, which read as total generation failure and argued for an immediate
second fine-tuned base model.

Re-testing through the actual production code path (`generate_llm_response()`, not raw Ollama calls)
overturned that: `IBLOG_TUTOR` generates clean French reliably — 9/9 across plain instructions and
JSON quiz generation with matching context. The real defect is exactly two mechanisms, both already
named in `architecture-failure-history.md`: **refusal language** (reverts to Darija regardless of
query language) and **context-script contamination** (French query + Arabic-script retrieved context
→ Darija answer). See `analyze_04_corrected_drift.md` for the full measurement.

A follow-up test ran the identical prompts through stock `gemma2:9b` (zero fine-tuning). It avoided
both failure modes on French (12/12 `PASS_FR`), which locates their cause in Atlas-Chat's Darija SFT
overriding instructions, not in Gemma-2 generally. But the same test's Darija result is a **false
positive**: Gemma's response scored as passing on script-ratio alone while being incoherent content —
two rephrasings of the question, no answer. This reconfirms Atlas-Chat's Darija SFT is doing real,
necessary work a script check cannot see.

## Decision

**Do not commit to a second fine-tuned base model yet.** Ship the cheaper fix first, on the existing
`IBLOG_TUTOR`, and let a re-measurement — not a hypothesis — decide whether a second model is needed.

### Stage 1 (this cycle): close both failure modes deterministically, zero new training

> ⚠️ **Amended — Stage 1 survives, with a different justification.** Routing French to Gemma
> *architecturally eliminates* Atlas's two French failure modes — Atlas never serves French again,
> so they are routed around rather than fixed. Stage 1 is nonetheless **still required**, because:
> 1. `ui_lang` is an invariant to enforce on **both** paths — the dropdown fixes *intent*, never
>    *compliance* (ADR 0002);
> 2. the **Darija path still runs Atlas** and needs its own protection;
> 3. **Gemma paraphrases legal citations** (measured) — and verbatim citation is a product claim to
>    a regulatory client, so `inject_citations()` must be **verified against Gemma's output shape**,
>    not assumed to cover it.

1. **Deterministic refusal-in-`ui_lang`** (ADR 0002, `analyze_01.md` §3 step 1). When retrieval
   confidence is below threshold, respond with a templated refusal in the query's declared language —
   no model call for that path at all. This removes failure mode 1 by construction; it cannot drift
   because the model never generates it.
2. **Retrieval language-affinity** (`analyze_01.md` §3 step 2). Prefer same-language chunks; when only
   cross-language chunks clear the retrieval threshold, flag the context and force the matching system
   template. This directly targets failure mode 2 — P4's exact scenario.
3. **Output validator as a backstop, not the primary fix** (`analyze_01.md` §4, `analyze_03.md` §6).
   Asymmetric FR/Darija rules (citation-span-aware for French, script-dominance for Darija), streaming
   prefix check, one bounded repair retry, deterministic terminal fallback.

### Stage 2 — ⚠️ GATE REDEFINED 2026-08-03: behavioral parity, not language parity

**Original gate (now moot for French):** re-run the harness against `IBLOG_TUTOR` behind the Stage 1
guardrails, requiring `arabic_outside_citations` = 0 on 100% of French turns. This is moot because
**Atlas no longer serves French** — the question it asked cannot arise.

**Redefined gate:** does **stock Gemma + Stage 1 guardrails** meet the *tutoring-behavior* bar?

| Dimension | Pass condition |
|---|---|
| Language compliance | `arabic_outside_citations` = 0 on 100% of French turns (unchanged, now measured on Gemma) |
| **Verbatim citations** | `inject_citations()` demonstrably overwrites Gemma's paraphrase («la loi numéro 27.06» → source form) |
| **Socratic register** | Guides rather than answers outright, at an acceptable rate on the golden set |
| **Refusal register** | Refuses in French *and* in the project's trained style |
| **Quiz conventions** | Schema valid via `format` **and** passes `_explanation_supports_answer()` |

**If this gate passes, no French LoRA is needed** — for the demo *or* for early production. If it
fails on the behavioral rows (the likely outcome, since stock Gemma has no exposure to these
conventions), only then does Stage 3 trigger — and it triggers scoped to *behavior*, not language.

### Stage 3 — ⚠️ COMMITTED 2026-08-03: topology **and** the French LoRA are both in scope

**Topology: adopted** (owner decision). French → Gemma-2-9B, Darija → `IBLOG_TUTOR`, serial on the
laptop, co-resident in the cloud phase.

**The French LoRA: no longer gated — it is committed for the CEO demo.** 🔴 This reverses the
"still gated / demo does not trigger it" position recorded here earlier, and reverses my own
recommendation in `analyze_04` §B.

**Why the reversal.** That position rested on stock Gemma's 12/12 `PASS_FR` — a *language
compliance* measurement. The MVP's claim is **adaptive learning tutoring in both languages**. Stock
Gemma answers questions; it does not tutor Socratically, cite verbatim, refuse in the trained
register, or follow quiz conventions. Demonstrated beside a fine-tuned `IBLOG_TUTOR`, that asymmetry
is immediately visible and undercuts the product claim. **Language compliance was the wrong
acceptance bar for an MVP whose product *is* the tutoring behavior.**

`analyze_01.md` §5 #1's cost objection (a second base doubles the data pipeline) is **accepted and
paid**, not dodged — the mitigation is scope, not deferral: **~1,800 rows over 8 components**, not
the full 3 k 11-component build, exploiting the fact that 30 of 36 corpus documents are already
French and `citations.py` already parses French citation forms.

**Full specification — component weights, the three Darija-shaped pipeline gates that must be
language-parameterized, 3–5 week timeline, acceptance gate, and risks:
`analyze_05_french_finetune_plan.md`.**

**Stage 2's gate is not bypassed — it becomes the acceptance gate for the French adapter**
(behavioral parity: Socratic register, verbatim citation, refusal register, quiz conventions), now
applied to `Gemma + LoRA` rather than to `Gemma stock`.

**Darija is not part of this contingency in any scenario.** Every measurement in this session
reconfirms `IBLOG_TUTOR`/Atlas-Chat-9B as the sole Darija path: Gemma's Darija output is incoherent
despite passing a naive script check, and no instruction-tuned Darija alternative to Atlas-Chat
exists in the current model landscape (`analyze_03.md` §2 — AtlasIA's 2025–26 Terjman v2 releases
are translation-only, not chat/instruction models).

## Consequences

- Cheapest path is tried first and is independently valuable regardless of Stage 3's outcome — the
  guardrail work is required either way (a second model does not remove the need for deterministic
  refusal/validation; `analyze_01.md`, `analyze_03.md` both establish that a declared or trained
  language intent is never a compliance guarantee on its own).
- Schedule risk is bounded: Stage 1 is on the order of days, not the multi-week French-SFT estimate
  in `analyze_03.md`. A second base model is deferred, not abandoned — Stage 3 stays fully specified
  and ready to execute if Stage 2's gate fails.
- Cloud cost estimates in `analyze_03.md` (single-GPU dual-path, ~$150–265/mo) remain valid as the
  Stage 3 contingency figure; Stage 1/2 add no infrastructure, only application code.

## Rejected alternatives

- **Committing to the dual-model swap now, on this session's data.** Rejected — the evidence supports
  "Atlas's Darija SFT causes narrow, routable failure modes," not "Atlas cannot serve French." Swapping
  now would spend weeks solving a problem the guardrail may close in days, and would do so on a
  12-sample directional signal rather than a production-gated measurement.
- **Treating Gemma's Darija `PASS_ARY` as evidence it could also serve Darija.** Rejected on direct
  inspection of the output — the metric passed, the content did not. Recorded here so this false
  positive is not rediscovered and re-argued later without the caveat attached.
- **Skipping Stage 1 and going straight to guardrails-plus-fine-tune in parallel.** Rejected for now
  as premature parallel spend; Stage 2's gate is the correct decision point, and running both at once
  forecloses the (well-supported) possibility that the fine-tune turns out to be unnecessary.

## Demo-deferred vs still-required guardrails (added 2026-08-03 — canonical list)

The laptop frontend is deliberately demo-grade (chat box, send button, language dropdown). That
justifies deferring infrastructure, **not** correctness. The split:

### Required even in the demo

| Guardrail | Why it cannot be deferred |
|---|---|
| **Post-generation language validator** | A Darija answer to a French question is the single most embarrassing possible demo failure. It is a regex — hours, not days. |
| **Deterministic refusal in `ui_lang`** | Removes the highest-probability drift path by construction; no model call, so it cannot drift. |
| **Verify `inject_citations()` against Gemma's output** | Gemma **paraphrases** citations (measured). Verbatim legal citation is a *product claim* to a regulatory client — the one thing this audience will check. |
| **`_explanation_supports_answer()` + schema validation** | Only if a quiz is shown. A wrong answer key in front of the CEO is fatal and unrecoverable in the room. |
| **Model preload + switch UX** | Without it the demo has an unexplained ~30 s dead input box that reads as a crash. |

### Deferred to production

Postgres RLS / `FORCE ROW LEVEL SECURITY` / non-owner role · PgBouncer pooling mode · durable
`SKIP LOCKED` job queue + transactional outbox · observability dashboards (P99, cache-hit, TTFT,
cost ceilings) · circuit breakers and bulkheads · semantic + L1/L2 caching · scale-to-zero and
warm scheduling.

Justification: single tenant, single user, no concurrency, no adversary in the room, no background
generation (see ADR 0003).

> ⚠️ **Deferral condition — non-negotiable.** Keep `tenant_id` in **every** query even while RLS is
> off, so RLS later lands as defense-in-depth rather than a rewrite — and so the code never *looks*
> like it is isolating tenants when it is not. Deferring enforcement is acceptable; writing code
> that implies an isolation guarantee it does not have is not.

## Verification

Identical to `analyze_04_corrected_drift.md`'s recommendation and ADR 0002/0003's verification
sections: re-run the 5-prompt harness through `generate_llm_response()` post-Stage-1, gate on 100%
`arabic_outside_citations = 0` for French and no Darija regression on P5-equivalent prompts.

**Added 2026-08-03:** run the same harness against **stock `gemma2:9b` behind the Stage 1
guardrails** for the redefined Stage 2 gate, including the behavioral rows (citation verbatimness,
Socratic register, refusal register, quiz conventions). Also close the open gap noted in
`analyze_04`: **Darija-query-against-French-context contamination is untested** — the mirror of P4.
