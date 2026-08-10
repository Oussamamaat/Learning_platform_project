# EXECUTION COMPLETE - READY FOR REVIEW

Autonomous overnight pipeline run, 2026-08-04. Stages 1-2 (smoke test evaluation,
full dataset generation) completed and validated. **Stopped before Stage 3
(fine-tuning) for explicit human review**, per the standing instruction — a bad
unattended training run is expensive and hard to cheaply undo; a bad generation
run is not, and this session already found four real bugs that only surfaced
under close inspection.

## TL;DR

- **French dataset: 1256 rows** (1130 train / 126 eval) across all 8
  `FRENCH_COMPONENT_CONFIG` components, saved to
  `data/fr_resume_output/final_export_fr/` (also zipped in the Kaggle kernel
  output as `fr_dataset_export.zip` — not yet copied into the repo's `data/`
  tree, see "Next steps" below).
- **Zero disallowed Arabic** (script/register gate) across all 1256 rows,
  verified directly, not just trusted from gate logs.
- **4 real Darija-only-regex bugs found and fixed** this session, each
  confirmed against live generation before being trusted:
  1. `row_is_french_clean` had no English-prose detection
  2. `_REFUSAL_MARKERS` was Darija-only
  3. `_CONFUSION_MARKERS` was Darija-only (caused a ~3h stopped Kaggle run —
     misread as a hang, actually a 100%-rejection gate bug)
  4. `_DISCLOSURE_MARKERS` was Darija-only (found by systematic audit, never
     live-tested before due to an unrelated bug masking it)
- **1 rounding bug fixed**: `scale_component_targets` could send a component's
  target negative at small row counts.
- **185→190 tests added/passing** across the session; full suite green.

---

## Timeline

| Time (local) | Event |
|---|---|
| Earlier session | Phase 1 (dataset parameterization) + Phase 2 code work (French prompt builders, gate fixes #1-#2), first Kaggle dual-GPU run stopped at 1150 raw rows (learner_adaptation appeared deadlocked) |
| ~03:00 | User handed off, autonomous mode begins |
| 03:11-03:31 | Stage 1: single-T4 smoke test (100 rows) — **PASSED**, all 8 components, all fixes confirmed live |
| 04:20-04:41 | Result evaluated, sampled, logged |
| ~04:50 | Stage 2 launched: dual-T4 resume+top-up run (`darija-tutor-fr-generation-resume`) |
| 04:50-05:40 | Monitored every 15-20 min, healthy throughout, no stalls |
| ~05:40 | Stage 2 **COMPLETE** — ~50 minutes total runtime, well under the 5h internal cap and 2.5h external checkpoint |
| 05:57 | Output fetched, evaluated, this summary written |

Total autonomous runtime: ~3 hours. Total Kaggle GPU time consumed this
session (including the earlier stopped run and both smoke tests): roughly
4-5 GPU-hours across single- and dual-T4 kernels.

---

## Bugs found and fixed (with file:line references)

All in `app/services/generate_training_data.py` unless noted.

### 1. English prose slipping past the French-quality gate
`row_is_french_clean` (line 1099) only checked for the *absence* of stray
Arabic and the *presence* of a couple of French function words — it never
checked for English. A `structured_explanation` smoke-test row came back as
English prose under French-language section headers and passed anyway.
**Fix:** added `_ENGLISH_INTRUSION_MARKERS` (line 1040) /
`english_marker_count()` (line 1049), rejecting rows where English markers
exceed a small tolerance.

### 2. `_REFUSAL_MARKERS` was Darija-only
(line 1127) Used by `row_is_refusal()`, gating `no_context_refusal` and
`grounded_refusal`'s refusal samples. Almost entirely Arabic-script phrasing
plus two French tokens — a real Gemma French refusal like "Je n'ai pas cette
information..." never matched it, so `no_context_refusal` scored 0/2 on the
first real Kaggle run. **Fix:** extended the same regex with ~10 French
refusal phrasings + `re.IGNORECASE` (the case-sensitivity gap was found
*while* fixing this — "Je" vs "je" — and fixed at the same time, benefiting
every existing pattern too).

### 3. `_CONFUSION_MARKERS` was Darija-only — the incident that cost the most time
Used by `row_is_learner_adaptation()` for `learner_adaptation`'s "learner
signals confusion" check. 100% Arabic-script. The French prompt asks for a
French confusion turn ("Je ne comprends toujours pas") that could never
match it. On the first real dual-GPU Kaggle run this produced a **100%
rejection rate for ~65-90 minutes**, which read as a hung/frozen process in
the live monitor (the `STALL` watchdog's wording said "Process is alive but
not producing output... check whether Ollama is still responding" — actively
misleading). The raw generator log showed the truth: waves completing every
~30s the whole time, attempts climbing steadily to 516/750, Ollama never
stopped responding. The run was stopped (recovering 1150 rows as raw
per-component files, since the merge step never got to run) before this was
diagnosed. **Fix:** added `_CONFUSION_MARKERS_FR` (line 1328), branched
`row_is_learner_adaptation(row, language)` by language, and — separately —
**fixed the watchdog's wording** to distinguish "still working, every attempt
just failing a gate" from an actual hang (tracks whether `attempts` is still
climbing). Also extended `_ADAPTATION_STOPWORDS` with the full
`_FRENCH_QUALITY_MARKERS` list, since a sparse 13-word French stopword set was
inflating apparent overlap between two genuinely different explanations —
same component, a secondary quality issue caught in the same audit pass.

### 4. `_DISCLOSURE_MARKERS` was Darija-only — found by audit, not by failure
(line 1182) Used by `row_discloses_general_knowledge()` for
`general_knowledge_disclosed`. Also 100% Arabic-script. This one was **never
live-tested before being found** — it kept getting silently skipped by bug #5
below in every earlier check, so the bug had zero chance to surface on its
own. Found by systematically grepping every Arabic-script regex in the file
and checking each call site for language-branching, after the user
specifically asked "did you make sure there are no more Darija imposters?"
**Fix:** added `_DISCLOSURE_MARKERS_FR`, branched
`row_discloses_general_knowledge(row, language)`.

### 5. `scale_component_targets` rounding bug (not language-specific)
`scale_component_targets()` (line 396) used "round every component, dump the
remainder on whichever is listed last" — at small `--target-rows` values the
accumulated rounding error could exceed the last component's own fair share,
sending it *negative*. Found running a 16-row local sanity check
(`general_knowledge_disclosed: -1`). **Fix:** rewrote using largest-remainder
apportionment (floor every share, hand the leftover rows one each to the
biggest fractional remainders) — guaranteed non-negative, guaranteed exact
sum, verified for every `target_rows` 1-59 in both languages.

---

## Final dataset composition

| Component | Rows | Original target | % of target |
|---|---|---|---|
| socratic | 465 | 550 | 85% |
| quiz_generation | 183 | 300 | 61% |
| learner_adaptation | 181 | 250 | 72% |
| structured_explanation | 156 | 300 | 52% |
| grounded_refusal | 118 | 200 | 59% |
| injection_resistance | 57 | 80 | 71% |
| no_context_refusal | 54 | 70 | 77% |
| general_knowledge_disclosed | 42 | 50 | 84% |
| **Total** | **1256** | **1800** | **70%** |

`socratic` didn't grow (465 vs. the pre-Stage-2 466) — expected and by
design: the resume run was deliberately sized so its already-banked rows
already met the per-GPU target, so it generated ~0 new rows rather than
wasting attempts re-generating a component that didn't need it.

**Quality checks performed, not just trusted from gate logs:**
- Ran `has_arabic_outside_citations()` directly against all 1256 rows'
  assistant text: **0 rows have disallowed Arabic**. The only Arabic present
  anywhere is in `socratic` (55 rows, 11.8%) and `structured_explanation` (26
  rows, 16.7%) — matching the deliberate ~20%
  `FRENCH_CROSS_LINGUAL_ARABIC_SOURCE_RATE`, always confined to verbatim
  legal citations (by design — cross-lingual grounding, not leakage).
- Manually sampled 2 rows each from `learner_adaptation`,
  `no_context_refusal`, `general_knowledge_disclosed`, `grounded_refusal` (the
  components that were broken or untested before this session) at the final
  1256-row scale: all clean, natural, correctly-registered French, citations
  preserved verbatim (EN 166, EN 397, NM EN 166), genuine reformulation with
  concrete examples in `learner_adaptation`, correct disclosure phrasing, no
  fabricated facts in refusals.

**Known limitation, not fixed this session:** ~35-46% cross-shard dedup loss
was observed on the first run for `structured_explanation`/`quiz_generation`
specifically (small 36-document corpus + templated output shape = high
near-duplicate rate across two independent generators). The resume run's
top-up narrowed the gap but didn't close it — both remain below their
original target share. Likely near the corpus's genuine content ceiling for
these two components; more generation may hit diminishing returns rather
than closing the gap further.

---

## What's ready, what's not

**Ready:**
- Language-parameterized generation pipeline (`--language fr`), fully tested,
  fully verified against live generation twice (smoke test + full run).
- 1256-row French dataset, zero script-gate violations, content-sampled clean.
- All code changes covered by regression tests (190 passing).

**Not done, needs your decision:**
- **Stage 3 (fine-tuning) was not started.** No Kaggle fine-tune kernel has
  been written or adapted for the French/Gemma path yet — `kaggle_finetune_v11.ipynb`
  exists for the Darija/Atlas-Chat path but the French base model
  (`gemma2:9b`, per `analyze_05_french_finetune_plan.md`) needs its own
  fine-tune notebook, not yet built.
  - This project's own `docs/LESSONS_LEARNED.md` #1 documents a real prior
    incident where a fine-tune landed *below* its own base model, and
    mandates a base-vs-adapter comparison before trusting any new adapter.
    This would be the first-ever fine-tune of this exact pipeline's output.
  - **My recommendation:** review the dataset (a `data/fr_resume_output/`
    sample, or the full `fr_dataset_export.zip`) yourself before greenlighting
    a fine-tune — the checks above are real but I'm one reviewer, and this is
    the highest-stakes, least-reversible step in the roadmap so far.
- ~~The 1256-row dataset currently lives only in the Kaggle kernel's output~~
  **Done during summary-writing:** copied to `data/fr_v1_merged/train.jsonl`
  (1130 rows) + `eval.jsonl` (126 rows), matching this project's `*_merged`
  naming convention. `data/fr_partial_v1/` is now superseded by this (it was
  the pre-Stage-2 partial set the resume run was seeded from).
- No decision made on whether 1256/1800 (70%) is "enough" to fine-tune on, or
  whether to run one more top-up pass first (structured_explanation/
  quiz_generation are the furthest below target).

## Kaggle resources created this session
- Dataset: `maataouioussama/darija-tutor-pipeline-v2` (multiple versions pushed)
- Kernels: `darija-tutor-fr-smoke-test` (v1, v2), `darija-tutor-fr-generation-full`
  (stopped, recovered), `darija-tutor-fr-generation-resume` (completed)
- All private, all still exist on Kaggle if you want to inspect them directly.
