# Autonomous Pipeline — Task Log

Started: 2026-08-04 (overnight autonomous run, user asleep)
Mandate: monitor/evaluate/fix/execute Stages 1-2 (smoke test → full French dataset
generation) autonomously; pause before Stage 3 (fine-tuning) for explicit review.

## Known tooling constraint
No Kaggle CLI subcommand cancels a running kernel (`kaggle kernels --help` has no
stop/cancel). Timeout detection = stop treating the run as viable + log it; actual
termination requires the web UI or the kernel's own internal MAX_HOURS cutoff.

## Stage 1 — Smoke test (single T4, --target-rows 100)

- Kernel: `maataouioussama/darija-tutor-fr-smoke-test` (v2, pushed with 4 gate fixes)
- Purpose: verify 4 Darija-only-regex bugs found and fixed this session:
  1. English-prose detection missing from `row_is_french_clean`
  2. `_REFUSAL_MARKERS` was Darija-only (no_context_refusal 0/2 on real Kaggle run)
  3. `_CONFUSION_MARKERS` was Darija-only (learner_adaptation 100% reject, read as a
     hang — real cause: gate rejection, not a stall. STALL watchdog wording also fixed.)
  4. `_DISCLOSURE_MARKERS` was Darija-only (found by audit, general_knowledge_disclosed
     never live-tested before due to an earlier unrelated bug)
- Estimated runtime: ~30-40 min (single GPU, 100 rows, all 8 components)
- Timeout cap: 1h40m from push (est + 1h buffer)
- Status: launched, monitoring

## Stage 2 — Full generation resume/top-up

- Kernel: `maataouioussama/darija-tutor-fr-generation-resume` (v1), dual-GPU T4
- Mechanism: seeded out_fr_gpu{0,1}/{component}_raw.jsonl with half the banked
  rows each (socratic 233/233, structured_explanation 61/62, quiz_generation
  81/81 — split via `fr_resume_seed/` in the dataset), launched
  `--resume --target-rows 750 --concurrency 4` per GPU. At 750/GPU,
  scale_component_targets gives socratic 229/GPU (already met by the 233 seed,
  generates ~0 new), structured_explanation 125/GPU (tops up from 61-62),
  quiz_generation 125/GPU (tops up from 81), and the 5 never-run components
  generate fully fresh (learner_adaptation 104, grounded_refusal 83,
  injection_resistance 34, no_context_refusal 29, general_knowledge_disclosed 21,
  all per GPU).
- Estimated runtime: ~1-1.5h generation (extrapolated from smoke test throughput)
  + ~10min setup. Script's own internal MAX_HOURS=5.0 self-terminates and still
  merges/packages whatever's on disk if it runs long — the safety net since
  there's no CLI kernel-cancel mechanism.
- My external timeout-guard checkpoint: ~2.5h from push (est 1.5h + 1h buffer) —
  past that, flag as needing attention in tasks.md, but the run's own MAX_HOURS
  is the actual hard stop.
- Status: launched, monitoring
- 05:04 — RUNNING, ~14min elapsed, well within 2.5h checkpoint. No action needed.
- 05:20 — RUNNING, ~30min elapsed, well within 2.5h checkpoint. No action needed.
- 05:36 — RUNNING, ~46min elapsed, well within 2.5h checkpoint. Approaching estimated
  completion window (~1-1.5h generation + setup). No action needed.

## Stage 2 result: COMPLETE, PASSED

1256 rows (1130 train / 126 eval) across all 8 components, 0 disallowed Arabic,
content-sampled clean. Copied to data/fr_v1_merged/. Full detail in pipeline_summary.md.

## Stage 3 — Fine-tuning (LoRA + Unsloth) — NOT started, stopped for human review

Reason: see pipeline_summary.md "What's ready, what's not". Autonomous loop
stopped here per standing instruction — dataset is ready pending your review,
no fine-tune kernel has been built yet for the French/Gemma path.

## Log

- 04:20 — Stage 1 smoke test: RUNNING, ~10min elapsed, well within 1h40m cap. No action needed.
- 04:41 — Stage 1 COMPLETE, PASSED CLEANLY. All 8/8 components hit or exceeded target
  (socratic 30/30, structured_explanation 17/17, quiz_generation 17/17,
  learner_adaptation 14/14, grounded_refusal 12/11, injection_resistance 4/4,
  no_context_refusal 4/4, general_knowledge_disclosed 3/3). 96 rows total
  (82 train + 14 eval), minimal dedup loss. arabic_intrusion near-zero everywhere
  (0-1 per component). Content-sampled learner_adaptation, general_knowledge_disclosed,
  no_context_refusal — all clean, correctly registered French, no fabrication. All 4
  gate fixes confirmed working against live generation. Proceeding to Stage 2.
