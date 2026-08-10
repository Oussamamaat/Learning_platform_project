# French LoRA Fine-Tune Plan — Gemma-2-9B

**Date:** 2026-08-03
**Status:** Committed scope for the CEO demo (3–5 week window)
**Supersedes:** `analyze_04_corrected_drift.md` §B's "demo: stock Gemma, no LoRA" recommendation,
and ADR 0001's classification of the French LoRA as *gated contingency*.

---

## Why this reverses my previous recommendation

I recommended stock Gemma for the demo on the strength of a **12/12 `PASS_FR`** language-compliance
measurement. **That optimized the wrong variable.** The MVP's claim is *adaptive learning tutoring in
both languages*, not *speaks French*. Stock Gemma answers questions; it does not tutor Socratically,
does not cite verbatim, does not refuse in the trained register, and has no quiz conventions.

Demonstrated side by side with a properly fine-tuned `IBLOG_TUTOR`, the asymmetry would be immediate
and would undercut the product claim in front of the audience it most needs to convince. **A French
LoRA is in scope for the demo.** Owner decision, accepted.

## What makes this tractable in 3–5 weeks

Verified by inspection this session — the French path inherits far more than expected:

| Asset | Status |
|---|---|
| **French source corpus** | ✅ **30 of 36 documents are French** (only 6 are `ar_`-prefixed). French is the *majority* of the corpus. |
| **French citation extraction** | ✅ Already implemented — `citations.py` handles `Loi n° X`, `Article X`, `Décret n° X`, `ISO X`, `Chapitre`, `Section`, `Paragraphe`, `Annexe`, `EN X`, `P. X` |
| **LoRA recipe** | ✅ Identical — Gemma-2-9B is Atlas-Chat's base. r=16/α=16, 7 linear targets, fp16, Kaggle T4 |
| **Export toolchain** | ✅ Identical — merge fp16 → GGUF → Q4_K_M → `ollama create` |
| **Gate architecture** | ✅ Reject-and-retry structure reusable; the *gates themselves* need language-parameterizing (below) |
| **Generation pipeline** | ⚠️ Darija-shaped in three specific places — the real work |

---

## 1. French component config

Dropped: `code_switching` (Darija↔French, no French analogue), `darija_preservation`,
`reasoning_preservation` (see Risk 1). Freed weight redistributed into core tutoring.

```python
FRENCH_COMPONENT_CONFIG = {
    "socratic":                    {"weight": 550, "multi_turn_pct": 0.75},
    "structured_explanation":      {"weight": 300, "multi_turn_pct": 0.0},
    "quiz_generation":             {"weight": 300, "multi_turn_pct": 0.0},
    "learner_adaptation":          {"weight": 250, "multi_turn_pct": 1.0},
    "grounded_refusal":            {"weight": 200, "multi_turn_pct": 0.3},
    "injection_resistance":        {"weight":  80, "multi_turn_pct": 0.0},
    "no_context_refusal":          {"weight":  70, "multi_turn_pct": 0.0},
    "general_knowledge_disclosed": {"weight":  50, "multi_turn_pct": 0.0},
}   # total 1800
```

**Core tutoring (socratic + structured_explanation + learner_adaptation) = 61% of the set.** That
proportion is the point: this model has to *read* as a tutor, which is the exact gap that motivated
the reversal above.

### Why `grounded_refusal` drops 700 → 200

Atlas needed 700 because it had a refusal **capability** defect — it refused in Darija regardless of
query language, across three failed prompt-fix attempts. Gemma refuses **correctly in French
zero-shot** (measured 3/3 on P2, `analyze_04`). So the French set needs refusal **register**
(project tone, "indique ce que l'utilisateur devrait étudier"), not refusal capability. Sizing it
like Atlas's would spend ~500 rows re-teaching a behavior the base already has.

### Keep French-output-from-Arabic-source rows

Even with `code_switching` dropped, allocate a slice of `socratic` and `structured_explanation` to
rows whose **source document is Arabic while the target output is French**. This is not
code-switching — it is **cross-lingual grounding**, and it is a real serving condition (P4: a French
learner's query retrieves Arabic regulatory text). It also directly trains the translate-and-explain
behavior, and is the single best place to teach **verbatim Arabic citation inside French prose**.

⚠️ This partially offsets the citation-paraphrase weakness left exposed by dropping
`code_switching` — see Risk 2.

## 2. Pipeline changes required (the real engineering)

The pipeline generates **Darija output containing French technical terms**. French mode needs
**French output**. Three gates are Darija-shaped and must be parameterized by target language:

| Location | Current behavior | French mode |
|---|---|---|
| `generate_training_data.py:1677` | Enforces **Arabic script**, rejects Arabizi/CJK | Enforce **Latin script**; reject Arabic **outside citation spans**; keep CJK rejection |
| `generate_training_data.py:709` | Darija marker scoring (ك-prefix, etc.) | Replace with **French-quality scoring** — reuse `_FRENCH_MARKERS` (`llm.py:102-113`), a curated French-not-English function-word list that already exists |
| `FRENCH_GATED_COMPONENTS` (`:132`) | Rejects rows *lacking* French vocabulary | Inverted/moot — French rows are French by construction. Replace with the **Arabic-intrusion** gate above |
| `ARABIC_SOURCE_COMPONENTS` (`:128`) | grounded_refusal, quiz prefer Arabic source | Prefer French source, **but deliberately retain a minority Arabic-source slice** (cross-lingual grounding, above) |

**The asymmetric-validator design from `analyze_01.md` §2.3 is exactly what the new script gate
needs** — Arabic permitted only inside citation spans, zero outside. Same rule, two consumers
(pipeline gate + serving validator). Build it once in the shared module ADR 0003 already calls for.

Everything else — reject-and-retry loop, dedup (forced CPU, cross-shard), citation extraction and
injection, `build_quiz_row()` battery, `_explanation_supports_answer()` — carries over untouched.

## 3. Timeline (3–5 weeks)

| Phase | Duration | Output |
|---|---|---|
| 1. Pipeline language-parameterization | ~4–5 days | Gates above + French system-prompt templates for each component |
| 2. Generation run (Kaggle) | ~3–4 days | ~1,800 gated rows; reuse the proven kernel notebook pattern |
| 3. Gate tuning + manual review | ~5–6 days | 15% review capacity, per existing discipline |
| 4. LoRA training (Kaggle T4) | ~1–2 days | Adapter; same recipe as Atlas |
| 5. Merge → GGUF Q4_K_M → Ollama | ~1 day | `IBLOG_TUTOR_FR` (or similar) registered locally |
| 6. Eval + base-vs-adapter | ~2–3 days | Drift harness + behavioral gate (below) |
| **Buffer** | ~3–5 days | Demo prep, dry runs |

⚠️ **Disk:** the merge stage needs **~18–25 GB transient** for fp16. The Atlas fp16 export was
previously deleted to reclaim space — make that room again before phase 5.
⚠️ **Template parity:** reuse the same `.System`-folded-into-user-turn Modelfile convention
(`FINETUNE_AND_DEPLOY.md §5.2`). Gemma-2 has no native system role and this project has already lost
time to a template mismatch.

## 4. Acceptance gate (before demo)

Both must pass; neither alone is sufficient.

**Language (regression check — Gemma already passes stock, so this only catches damage):**
- `arabic_outside_citations` = 0 on 100% of French turns
- P5-equivalent Darija unaffected (`IBLOG_TUTOR` untouched, but re-run to be certain)

**Behavior (the reason this fine-tune exists):**
- Socratic: guides rather than answers outright, at an agreed rate on the golden set
- **Verbatim citation**: Arabic legal references reproduced character-for-character in French prose
- Refusal: in French *and* in project register
- Quiz: schema-valid via `format` **and** passes `_explanation_supports_answer()`
- **Base-vs-adapter comparison** — standing project requirement; the adapter must not fall below
  stock Gemma's grounding floor (this is exactly the check that caught the original citation-
  fabrication defect on the Darija side)

## 5. Risks

**Risk 1 — dropping `reasoning_preservation` may cost quiz quality.** ⚠️ Flagging precisely because
the in-code rationale (`generate_training_data.py:65-69`) ties it specifically to quizzes: *"Without
it a narrow tutor adapter loses the ability quizzes and (later) diagrams depend on."* Quiz is 300
weight (17%) of this set and a CEO-confirmed MVP feature.
→ **Trip-wire:** if quiz eval regresses against stock Gemma in phase 6, add ~100 rows of
`reasoning_preservation` and retrain. Budgeted within the buffer. Not pre-emptively included, per
the scope decision.

**Risk 2 — citation paraphrase.** Dropping `code_switching` leaves verbatim citation dependent on
(a) the cross-lingual-grounding slice above and (b) `inject_citations()` post-processing, which
already runs in `generate_llm_response()` (`llm.py:220-228`) and overwrites model citations from
retrieved context.
→ **Required verification, not an assumption:** confirm `inject_citations()` behaves correctly on
Gemma's output shape, where the model wrote *French prose paraphrasing an Arabic citation*. This is
the one measured Gemma weakness and verbatim citation is a product claim to a regulatory client.

**Risk 3 — schedule.** Phases 1–3 are the uncertain ones (new gate code + generation quality). The
Darija set needed multiple regeneration cycles. Mitigation: the pipeline, corpus, and gate
architecture all exist; this is parameterization, not a new build.

**Risk 4 — two models, one 8 GB laptop.** Unchanged from `analyze_04` §A: serial loading, ~30 s
swap, `OLLAMA_MAX_LOADED_MODELS=1`. The French model is a second ~5.4 GB GGUF — confirm disk
headroom alongside the existing Atlas GGUFs.
