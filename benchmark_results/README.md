# Akash GPU lease — validation results (2026-09-04)

Overview of the rented-GPU validation run (RTX 8000, Akash/ZenCloud) that answered
the three questions an 8 GB laptop couldn't: multi-model VRAM residency,
STT engine choice, and Darija TTS acceptability. Full narrative, rationale, and
the complete follow-up punch list live in the planning doc:
`C:\Users\oussa\.claude\plans\now-that-i-have-fuzzy-hopper.md` (Part 6 = findings
and follow-ups, Verification = the checklist this run satisfied). This file is
just the map to the raw artifacts.

## The three questions, answered

1. **Does the French↔Darija switch cost drop to zero with both 9B tutors resident?**
   Yes. See [`phase1_2026-09-04_benchmark_report.md`](phase1_2026-09-04_benchmark_report.md)
   — alternating-language turns (2.1–3.4s) cost about the same as same-language
   turns (2.5–3.1s), versus the laptop's 5.7 GB evict+reload penalty on every flip.
2. **Which STT engine wins for Darija — faster-whisper or SeamlessM4T-v2?**
   Seamless, and by more than first reported: the original raw-WER numbers
   (0.555 whisper / 0.473 seamless, whitespace-tokenized, no orthographic
   normalization) undersold the gap. Re-scored with Arabic-orthography
   folding + punctuation stripping + CER
   (`scripts/eval_stt_rescore.py`, see
   [ADR 0009](../docs/architecture/rectified/adr/0009-stt-eval-rescoring.md)):
   on **pure Darija specifically**, seamless's normalized WER is 0.428 vs.
   whisper's 0.699 (CER 0.124 vs. 0.266) — whisper's Darija errors are
   mostly genuine, seamless's were mostly orthographic noise. Whisper still
   wins French narrowly (normalized WER 0.087 vs. 0.098). Both engines
   struggle about equally on code-switched utterances even after
   normalization (WER ~0.54–0.59) — a real, separate finding, not a scoring
   artifact. Both comfortably real-time (RTF 0.208 whisper / 0.336
   seamless). Full per-utterance numbers in
   [`phase2_2026-09-04_eval_stt_results.json`](phase2_2026-09-04_eval_stt_results.json)
   (raw) and
   [`phase2_2026-09-04_eval_stt_results_normalized.json`](phase2_2026-09-04_eval_stt_results_normalized.json)
   (normalized).
3. **Does Piper's Darija-approximation voice (`ar_JO-kareem-medium`) sound
   acceptable?** No — rejected on live listening, both via the isolated bake-off
   sentences and through the real voice interface. Jordanian accent doesn't fit
   Moroccan Darija's prosody. Synthesis speed data (not the deciding factor) is in
   [`phase2_2026-09-04_eval_tts_results.json`](phase2_2026-09-04_eval_tts_results.json).
   Full verdict and what to do about it is in the plan doc, Part 6 item 9.

## Artifacts by phase

| Phase | What it measured | Files |
|---|---|---|
| Seed | Corpus ingestion (25 files), grounded-chat sanity check | — (not a benchmark output; see `docs/deploy/lease-00-seed.sh`) |
| 1 — LLM latency | Per-language chat latency, language-switch cost, raw Ollama TTFT, peak VRAM | [`phase1_2026-09-04_benchmark_report.md`](phase1_2026-09-04_benchmark_report.md), [`phase1_2026-09-04_benchmark_llm.json`](phase1_2026-09-04_benchmark_llm.json) |
| 2 — Voice bake-off | STT WER/RTF (whisper vs. seamless, 30 utterances), TTS synthesis RTF, one end-to-end voice-pipeline sanity check | [`phase2_2026-09-04_benchmark_report.md`](phase2_2026-09-04_benchmark_report.md), [`phase2_2026-09-04_eval_stt_results.json`](phase2_2026-09-04_eval_stt_results.json), [`phase2_2026-09-04_eval_tts_results.json`](phase2_2026-09-04_eval_tts_results.json) |
| 3 — OCR & ingest | PaddleOCR-VL availability, full-document ingest timing (76-page PDF) | [`phase3_2026-09-04_report.md`](phase3_2026-09-04_report.md), [`phase3_2026-09-04_benchmark_ocr.json`](phase3_2026-09-04_benchmark_ocr.json) |

`tts_samples/` is an empty placeholder — the raw `.wav` files from the TTS
bake-off were never pulled off the lease (the live-interface listening test made
that redundant; the verdict above is already decisive either way). They no
longer exist since the lease's ephemeral filesystem is gone once closed.

## Headline numbers worth remembering

- **Phase 3 ingest: 76.3s for a 76-page PDF**, vs. the laptop's 30m14s baseline —
  roughly 24x faster.
- **Phase 1 chat latency: 2.0–3.4s** across French, Darija, and alternating turns,
  vs. the laptop's 17–24s (French) / 44–144s (language-switch penalty).
- Total lease spend stayed well inside the $20 budget on the RTX 8000 tier
  (~$0.21/hr starting bid).

## Two real bugs this run found (fixed or filed)

- **`scripts/speech_worker_resident.py`** — three real, previously-untested bugs
  fixed live during Phase 2 (missing `sys.path` for the `app` import, wrong
  language codes for both whisper and Seamless, a renamed `transformers` kwarg).
  All fixed and pushed; see commits `f1be500`, `6423b2f`, `c0b2875`, `1ad8376`.
- **`app/services/vad.py`'s fixed RMS threshold** and **`app/services/routing.py`'s
  `resolve_domain`** Darija mis-routing — both found live, not fixed on the paid
  lease (neither blocks the core three questions). Full detail and fix plan in
  the plan doc, Part 6 items 7 and 10.
  **Update, post-lease (`POST_LEASE_MVP_SPRINT_PLAN.md` item 1, `ADR 0004`):** the
  VAD threshold hypothesis did not hold up against an offline sweep on real
  speech (fires correctly at 30/30 files; see `ADR 0005`) — root cause deferred
  to a live-mic Phase B run instead. The `resolve_domain` finding root-caused
  differently than originally hypothesized: it is **not** an application or
  Postgres bug. Reproduced locally that passing Arabic-script text as a literal
  shell command-line argument to `curl -d '...'` corrupts it into `?` bytes
  before the request is even built (confirmed via `curl --trace-ascii -`) —
  this fully explains the `language:"fr"` misreport, and plausibly (not yet
  100% confirmed on the lease's own Linux shell) the `domain_source` symptom
  too. `docs/deploy/lease-00-seed.sh`'s own verification call used exactly this
  vulnerable pattern and has been fixed (heredoc-to-file + `--data-binary @file`
  instead of a literal argv). See `ADR 0004` for the full evidence chain and
  what Phase B still needs to confirm on the lease's own shell.
