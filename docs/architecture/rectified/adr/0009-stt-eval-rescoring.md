# ADR 0009: Re-scoring the STT Bake-off with Normalized WER + CER

**Status:** Resolved — normalization applied, engine verdict re-examined and confirmed unchanged
in direction (seamless still wins Darija), materially more precise in magnitude
**Date:** 2026-09-04
**Depends on:** `benchmark_results/phase2_2026-09-04_eval_stt_results.json`, `scripts/eval_stt.py`

## Problem

`scripts/eval_stt.py`'s own docstring admits its WER is "deliberately not ... publication-grade
... with punctuation/casing normalization rules" — a simple whitespace-tokenized Levenshtein WER,
built for a relative ranking, not an absolute quality signal. The headline Darija numbers from the
2026-09-04 lease (`mean_wer` 0.583 seamless / 0.728 whisper) were nonetheless being read as
absolute evidence for the §3 finetune question. Inspecting the stored `reference`/`hypothesis`
pairs directly showed this concern was concrete, not hypothetical: `doda_ary_03`'s WER of 0.2 is a
single trailing period; `doda_ary_00`'s WER of 0.5 is `هوما/هما` + `انا/أنا`, spelling variants with
identical meaning; `doda_ary_07`'s WER of 2.0 is a word-split-plus-punctuation artifact. Darija has
no standardized orthography, so raw word-level WER penalizes legitimate spelling choice as if it
were a transcription failure.

## Investigation method

Re-scored every stored `(reference, hypothesis)` pair from
`benchmark_results/phase2_2026-09-04_eval_stt_results.json` — no model re-run, no audio, no GPU —
with: NFKC normalization, `app.services.citations.fold_arabic` (already proven correct; it is what
unblocked the PaddleOCR Arabic gate), Arabic- and Latin-script punctuation stripping, and
lowercasing. Reported both word-level WER and character-level CER (CER is the more defensible
primary metric for a dialect without standard word-boundary conventions in casual transcription).
Built as `scripts/eval_stt_rescore.py`, a standalone script over the committed JSON, so this is
reproducible with one command and does not modify `eval_stt.py`'s own raw scoring (both are kept:
raw for continuity with the original bake-off record, normalized as the corrected reading).

## Options considered

- **Adopt the `jiwer` package** — rejected, same reasoning `eval_stt.py`'s own docstring already
  gives: not a project dependency, and the existing `fold_arabic` already solves the
  Darija-specific half of the problem (orthographic variants) that a generic normalizer wouldn't
  know to fold.
- **Silently replace the published raw numbers** — rejected: both raw and normalized are reported
  side by side, so neither the original bake-off record nor the corrected reading is lost.
- **Re-score only in aggregate** — rejected once the per-category breakdown (below) showed the
  aggregate number obscures a real, category-specific split; both are reported.

## Evidence

Aggregate (all 30 utterances):

| engine | raw mean WER | normalized mean WER | normalized mean CER | RTF |
|---|---|---|---|---|
| whisper | 0.555 | 0.514 | 0.269 | 0.208 |
| seamless | 0.473 | 0.362 | 0.206 | 0.336 |

By category (n=6 codeswitch, n=16 doda_ary, n=8 french):

| engine | category | raw WER | normalized WER | normalized CER |
|---|---|---|---|---|
| whisper | codeswitch | 0.623 | 0.589 | 0.604 |
| whisper | **doda_ary (pure Darija)** | 0.728 | **0.699** | **0.266** |
| whisper | french | 0.159 | 0.087 | 0.023 |
| seamless | codeswitch | 0.588 | 0.540 | 0.635 |
| seamless | **doda_ary (pure Darija)** | 0.583 | **0.428** | **0.124** |
| seamless | french | 0.166 | 0.098 | 0.048 |

**The aggregate verdict and the category verdict point the same direction but for different
reasons — worth stating precisely rather than only reporting the aggregate.** Normalization
*widens* the whisper/seamless gap overall (raw +0.082 → normalized +0.152 favoring seamless), but
this is driven almost entirely by pure Darija: normalization barely moves whisper's Darija score
(0.728 → 0.699 — most of its errors are genuine transcription failures, not orthography) while it
sharply improves seamless's (0.583 → 0.428 WER, and CER of 0.124 is roughly half whisper's 0.266).
**Seamless's win on Darija specifically is not a scoring artifact — it is a real, and larger than
first reported, accuracy advantage.** Whisper remains narrowly better on French (0.087 vs. 0.098
normalized WER), matching its documented French strength.

**A separate, genuine finding, not previously visible in the aggregate:** code-switched utterances
are poorly transcribed by *both* engines even after normalization (WER ~0.54–0.59, and CER
*exceeds* WER for both engines on this category specifically — 0.604–0.635 vs. 0.54–0.59 WER,
the opposite of the pattern on pure Darija). This is flagged, not root-caused, under the session's
time-box; the CER > WER inversion is unusual enough to warrant a specific look at whether
punctuation-stripping is interacting badly with French contractions (e.g. `l'incident` → `l
incident`) before trusting the code-switch CER number specifically.

## Decision

1. Adopt normalized WER + CER as the reported metric for any future Darija/French STT comparison;
   keep raw WER alongside it as the original-record baseline, never silently replaced.
2. **Engine choice does not change**: seamless remains the pick for Darija-heavy traffic, and the
   normalized numbers make that case more strongly, not less, than the original report. Whisper
   remains preferable for French-only traffic. A per-utterance or per-session engine selection
   (already named as a candidate in `voice-assistant.md`) is worth revisiting given how much the
   category split matters, but is out of this ADR's scope.
3. `benchmark_results/README.md`'s STT-comparison framing is corrected to cite the normalized,
   category-broken-out numbers rather than only the aggregate raw WER.
4. Code-switched-utterance quality is carried forward as a named open item for the §3 finetune
   evidence-gathering, not resolved here.

## Rationale

A metric-validity problem masquerading as a model-quality problem is exactly the failure mode this
project's own history already records (see `finetune-degrades-citation-grounding` and
`v11-adapter-fixed-citation-fabrication` in the project's memory of past incidents) — training or
selecting against an unvalidated number risks optimizing for the wrong target. Re-scoring first,
free and offline, is strictly cheaper than discovering the same problem after committing lease time
or GPU time to a decision the raw number would have mis-informed.

## Constraints acknowledged

No GPU, no audio re-recording, no model re-run — entirely a re-analysis of already-collected
artifacts, which is why this was tractable inside the session's local-only phase. The code-switch
CER anomaly is explicitly left open rather than chased further, per the session's 30–45 minute
per-item time-box; it is named as a specific next check (verify punctuation-stripping's interaction
with French contractions) rather than left as a vague "needs more work."
