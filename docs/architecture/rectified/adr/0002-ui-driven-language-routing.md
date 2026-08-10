# ADR 0002: UI-Driven Language Routing

**Status:** Proposed — **confirmed 2026-08-03** under the laptop MVP, with one consequence added
**Date:** 2026-08-03
**Depends on:** `analyze_01.md`, `analyze_04_corrected_drift.md`

> **Changed 2026-08-03 — confirmed, but not quite unchanged.**
> Every decision below stands as written. One consequence is **new and material**: under the laptop
> MVP's **serial** model loading, `ui_lang` no longer selects only a *prompt template* — it selects
> **which model is resident in VRAM**. A dropdown change is now an **infrastructure** event
> (a ~30 s model swap), not just a string substitution. See the added Consequences entry.
>
> The central principle is *strengthened*, not weakened, by the laptop topology: `ui_lang` now
> determines routing at two levels (model + template) while still supplying **zero** evidence of
> compliance. **The dropdown fixes intent, not compliance** — the validator remains mandatory.

## Context

The frontend will add an explicit language dropdown (French / Darija). This changes how language
*intent* enters the system, but it does not change how language *compliance* is enforced — those
are separate problems and this ADR is careful not to conflate them.

`analyze_04_corrected_drift.md` establishes, via the real production code path
(`generate_llm_response()`), that `IBLOG_TUTOR` generates correct French reliably when a query is
answerable from matching context (9/9 clean across P1-supplementary and P3), and Darija reliably
(3/3). The model drifts to Darija in exactly two situations: refusals, and French queries answered
against Arabic-script retrieved context. Both are already explained in
`architecture-failure-history.md` and neither is fixed by knowing the user's declared language —
a declared-language field tells the system what to *aim for*, not that the model will *hit* it.

## Decision

1. **`ui_lang` (values: `fr`, `ary`) becomes the authoritative routing input** for system-prompt
   selection, replacing implicit reliance on `detect_query_language()` for that purpose.
2. **`detect_query_language()` (`app/services/llm.py:120-149`) is retained**, not removed — repurposed
   as a **cross-check / telemetry signal**. A mismatch between `ui_lang` and the detected query
   language (e.g., dropdown set to French, user typed in Darija) is logged and surfaced as a metric;
   it is not silently overridden in either direction, since either could be a legitimate user choice
   (a Darija-fluent user asking a French-labeled question, or vice versa).
3. **The post-generation output validator (see `analyze_01.md` §4, `analyze_03.md` §6) remains
   mandatory and unaffected by this ADR.** The dropdown fixes intent; it supplies zero evidence that
   the model complied. Do not treat "user selected French" as a substitute for verifying the output
   is French.
4. **Deterministic refusal-in-`ui_lang`** is adopted as the primary fix for failure mode 1 (refusal
   language): when retrieval confidence is below threshold, or when a candidate answer is generated
   but validated as a refusal, respond with a templated refusal in `ui_lang` — zero model risk, zero
   new training rows.
5. **Retrieval language-affinity** is adopted as the primary fix for failure mode 2 (context
   contamination): prefer same-language chunks; when only cross-language chunks clear the retrieval
   threshold, flag the context and force the query's declared-language system template.

## Consequences

- Simpler validator: expected language is known exactly from `ui_lang`, so the check is an
  **assertion**, not a detection problem. No fastText or other language-ID model needed in the
  request path.
- `ui_lang` must be threaded through the gateway → orchestrator → prompt-builder → validator as a
  single explicit parameter, replacing the current implicit inference at generation time for prompt
  selection (detection is kept, but demoted to telemetry).
- This ADR does **not** decide the base-model question (single French/Darija model on `IBLOG_TUTOR`
  vs. a second fine-tuned model). That is ADR 0001, gated on the Gemma-2-9B comparison in
  `analyze_04_corrected_drift.md`.

### ⚠️ Added 2026-08-03 — `ui_lang` has an infrastructure consequence under serial loading

On the laptop MVP exactly one model is resident (8 GB VRAM; two 9B models cannot co-reside). So
`ui_lang` selects **the model**, not merely the system prompt:

- `fr` → `gemma2:9b` · `ary` → `IBLOG_TUTOR` — **never both loaded.**
- **Changing the dropdown costs a ~30 s model swap** (measured: 34–38 s cold vs 5–9 s warm).
- Required UX, not optional polish: preload the default language at startup (`num_predict: 1`
  warm-up call); on switch show an explicit loading state and **disable send** until warm. A
  live-looking input box in front of a non-resident model reads as a crash.
- Configuration: set `OLLAMA_MAX_LOADED_MODELS=1` to make serial behavior **contractual** rather
  than an emergent side effect of VRAM pressure, plus a long `OLLAMA_KEEP_ALIVE` so the resident
  model does not silently unload during a pause.
- ⚠️ **Any background job requesting the other model will thrash the GPU.** This is the conflict
  resolved in ADR 0003.
- **This consequence disappears in the cloud phase** — both models co-reside on one L4 and switching
  is free. `ui_lang`'s *routing* semantics are identical in both phases; only the cost differs.

Full VRAM math, swap latency, and demo UX: `analyze_04_corrected_drift.md` §A.

## Rejected alternatives

- **Using `ui_lang` to bypass the output validator entirely** ("the user told us the language, so
  trust the model"). Rejected — this is precisely the assumption `analyze_04` disproves for the two
  known failure modes; a declared intent is not a compliance guarantee.
- **Removing `detect_query_language()`.** Rejected — it costs nothing to keep and its mismatch signal
  is the cheapest way to notice a labeling problem (e.g., a mislabeled UI default, or a bilingual
  user overriding their own dropdown mid-conversation) before it becomes a support ticket.
