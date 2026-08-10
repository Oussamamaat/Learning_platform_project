# Dataset Generation — Problems Faced and Fixes Applied (Tenant #1)

Concerns the tenant #1 (Moroccan safety/security, Darija + French) dataset
generation pipeline. All entries below are preserved verbatim.

**Scope:** Everything encountered while building the training-data generation pipeline, from the first working version through the Kaggle validation run.
**Companion docs:** `DATASET_WALKTHROUGH.md` (decisions and reasoning, narrative form) · `KAGGLE_GENERATION_GUIDE.md` (how to run the full generation) · `DATA_PREP_PLAN.md` (original spec)
**Status key:** ✅ Fixed and verified · 🔶 Identified, fix not yet applied · ➖ Investigated, determined not a problem

---

## 1. Original pipeline — zero rows produced

The first implementation had ten structural bugs (train/serve prompt mismatch, dropped guide files, an in-memory buffer that risked a crash on long runs, `max_tokens` too low causing JSON truncation, among others). Net result: 0 rows generated. Fixed prior to the work in this document — full detail in `DATASET_WALKTHROUGH.md` §5.

## 2. Five correctness defects, found by measuring output, not reading code

| Defect | Root cause | Fix | Status |
|---|---|---|---|
| JSON repair too brittle | One stray character after a closing quote discarded an otherwise-valid response | Progressive repair + regex salvage fallback | ✅ +18% valid rows/call |
| **Dedup false-collapse** | Dedup embedding included the full system prompt, near-identical across a domain — swamped the real conversation signal | Embed conversation turns only | ✅ Live: 24→4 (broken) → 24→23 (fixed) |
| Malformed turn order | Rows could open on an assistant turn with no preceding user turn | Trim to first-user → last-assistant window | ✅ 0 defects in validated batches |
| Silent row loss at export | `component` field trusted from the model's own JSON; a mismatch dropped the row invisibly | Stamp component/domain from generation context | ✅ Eliminated this loss class |
| No run visibility | `logging.basicConfig` never called — a multi-hour run produced zero console output | Configured logging + `--log-level` | ✅ Restored |

Full detail: `DATASET_WALKTHROUGH.md` §6.

## 3. Q2_K quantization — the model was running on the worst available quant

`ollama show` revealed the model in use was **Q2_K** (3.8 GB, ~3.3 bits/weight) — the worst of 14 published quantizations. Root cause: an untagged `ollama pull` resolves to whichever filename sorts first alphabetically, and `Q2_K` does. Plausibly explains prior "inherent model behavior" symptoms (malformed JSON, empty content, echoed placeholders). **Fix:** pulled `Q4_K_M` (5.8 GB, ~4.8 bits/weight) explicitly. ✅ Confirmed via `ollama show`; quality improvement verified in §5 below.

## 4. Socratic component — explains, but often doesn't ask

An external review raised: does the socratic component actually teach, or just interrogate? Checked against the pipeline: the five reference examples in `few_shot_examples.md` were already correctly scaffolded (explain, then check); the prompt itself never said so explicitly, and a real sample showed pure question-only turns with no explanation.

**Fix applied:** added an instruction requiring every assistant turn to state the relevant fact before asking, and marking a question-only turn invalid. ✅

**Revised after review:** on inspecting the Kaggle sample, 35% of socratic turns ended with no question at all. Raised as a possible defect — **on review, determined not to be one.** ➖ The actual requirement is good pedagogical explanation; a follow-up question is welcome when it fits but not mandatory. The "explain before questioning" instruction stays; no "must end in `?`" requirement was added.

## 5. Speed and reliability optimization (Q4_K_M)

Once Q4_K_M was confirmed correct but slow (~38 s/row, no batching headroom on 8 GB), five changes were applied to `generate_training_data.py`:

| Change | Reasoning |
|---|---|
| Schema-constrained output (`format` + JSON schema on the Ollama call) | Ollama enforces row shape during sampling — malformed JSON becomes structurally impossible instead of something to repair after the fact. `minLength: 15` on content also kills the `"..."` placeholder echo at the source. |
| `num_ctx` 8192 → 4096 | Prompts are ≤1,500 chars; 8192 wasted KV cache and prompt-eval time. |
| `num_predict` 2048 → 1024 | Measured outputs are 250–400 tokens; 2048 was never reached. |
| `code_switching` prompt: 7,355 → 1,501 chars | Was injecting the full rules file and all five few-shot examples every call. Now samples one example, like the other builders. |
| `keep_alive: 30m` | Prevents Ollama evicting/reloading the 5.8 GB weights mid-run. |

**Measured result** (fixed 12-row validation run, all four components):

| | Q2_K baseline | Q4_K_M, no fixes | Q4_K_M + fixes |
|---|---:|---:|---:|
| Parse failures | 31% | 14% | **0%** |
| Salvage fallbacks | 4 | 0 | **0** |
| Seconds/row | ~8.8 | ~38 | **~18** |

✅ Roughly halved Q4_K_M generation time and eliminated parse failures entirely. Output coherence also improved sharply — Q2_K samples contained hallucinated, off-topic content; Q4_K_M output was on-topic and grammatical.

**Local full-run estimate at this point: ~40–42 hours** (7,500 rows, single 8 GB GPU, no batching).

## 6. Why local generation can't be sped up further without more VRAM

Decode is memory-bandwidth-bound: every token requires reading the full model weights from VRAM, regardless of how many sequences are in flight. Single-stream generation on one GPU therefore can't benefit from software optimization alone. Confirmed empirically earlier: forcing concurrent requests on the 8 GB card pushed part of the model off-GPU and throughput *dropped* (~45 → ~13 tok/s) rather than improving. The fix is VRAM headroom to batch, not a faster chip — this is what motivated moving to Kaggle.

## 7. Kaggle hardware constraints (gemma2 on T4)

Before running on Kaggle, several model-specific constraints were identified:

- T4 has no native bf16 — gemma2 was trained in bf16; forcing fp16 risks numerical overflow.
- Gemma2 needs attention logit soft-capping, which requires the FlashInfer backend in vLLM.
- bf16 weights (~18.5 GB) don't fit one 16 GB T4 — needs `TP=2` across both GPUs.
- No AWQ/GPTQ quantization of Atlas-Chat exists — only bf16 and GGUF.

**Decision:** attempt vLLM time-boxed (2 h) with explicit abort criteria (NaN/empty output, OOM, FlashInfer install failure); fall back to Ollama + parallelism, which reuses the already-validated GGUF. Full detail: `KAGGLE_GENERATION_GUIDE.md`.

## 8. Added `--concurrency` and `--resume` to the pipeline

**`--concurrency N`** — restructured the generation loop to submit a wave of requests through a `ThreadPoolExecutor` instead of one at a time, sized to the remaining target so the final wave doesn't overshoot. ✅ Tested at `--concurrency 2`: 0 parse failures.

**`--resume`** — counts existing `*_raw.jsonl` rows toward each component's target and seeds the dedup set from them, so a session that dies mid-generation (Kaggle's ~12 h cap is the real-world trigger) doesn't lose progress or regenerate duplicates. Malformed trailing lines from a mid-write kill are skipped rather than aborting the resume. ✅ Verified: seeded 2 existing rows, correctly generated only 1 more to reach a target of 3.

**Known limitation:** raw files are consumed and deleted at the dedup/split stage, so `--resume` only helps if the run dies *during* generation — the realistic failure mode, but worth knowing the boundary.

## 9. Kaggle 200-row validation run — results

Ran on 2× T4, Ollama + `OLLAMA_NUM_PARALLEL`, as two independent processes (one per GPU). 204 rows in ~15 minutes.

**Speed: confirmed the batching hypothesis.** ~4.5 s/row vs ~18 s/row local single-GPU — a 4× speedup. Extrapolated full-run estimate: **~9.5 hours**, fitting inside one Kaggle session.

Four things surfaced by inspecting the output, not assumed:

### 9a. French code-switching has collapsed 🔶
Only 8% of assistant turns (27/334) contain a real French technical term; **95% of `code_switching`-component rows (63/66) contain zero French at all.** The spec requires 30% French + 30% mixed. Root cause: the model generates Arabic-script Darija, and in Arabic script it reaches for Arabic vocabulary, not French loanwords — a content-generation issue, not a transliteration one. **Fix identified, not yet applied:** structurally require French technical terms per turn in the `code_switching` and `socratic` prompts, rather than mentioning it as a style note.

### 9b. Socratic question-ending ➖
See §4 — investigated, determined not a defect per actual requirements.

### 9c. Cross-GPU deduplication gap 🔶
Each GPU ran as a separate process with its own output directory, so each deduplicated only against itself. Running the pipeline's own `deduplicate()` across both combined: **204 → 193 rows, 11 near-duplicates (5.4%) that neither process could see.** At 7,500 rows this projects to roughly 400 duplicate-equivalent rows. **Fix identified, not yet applied:** a merge step — combine both GPUs' raw output, dedup once, re-split — rather than trusting each process's independent pass.

### 9d. The Arabizi ceiling, confirmed at scale
100% of the 204 rows (203/204) came back `transliterated: true` — Atlas-Chat emits Arabic script regardless of quantization or prompting, and the pipeline's character-mapper converts it. Structural, not fixable via prompting or quant choice.

## 10. Transliteration quality — testing a free alternative to the paid API pass

Before committing to a paid transliteration pass, tested whether Atlas-Chat could transliterate its own raw output better than the pipeline's character-mapper, by asking it directly. Three real samples, same raw Arabic-script input compared against both:

**Atlas self-transliteration wins clearly on loanwords** — `blockchain`, `atelier`, `consignation`, `LOTO` all came back correctly spelled, versus the mapper's phonetic mangling (`alblwkchyn`, `atlyr`, `kwnsygnatywn`, `lwtw`). This directly addresses part of §9a.

**But it makes real errors the mapper structurally can't** — one sample changed `م3a` ("with") into `mn` ("from"), a meaning error, not a style difference; another used the wrong numeral (`mtwaf3` instead of `mtwaf9`, confusing ع with ق); a third had a stray capitalization artifact. Two clear errors across three short sentences — a rate that matters at 7,500 rows if the bar is "top notch."

**Decision:** stack both rather than choosing one. 🔶 Not yet implemented:
1. Replace the character-mapper with Atlas-Chat self-transliteration — free, and a genuine upgrade on the loanword problem.
2. Still run a frontier-model (Claude Haiku, batch) pass on top, now catching residual errors in already-decent text rather than doing the full Arabic→Arabizi conversion from scratch — likely cheaper than the original ~$8 estimate, not additive to it.

---

## Open items (not yet implemented, as of this document)

1. French-term enforcement in `code_switching` / `socratic` prompts (§9a)
2. Merge-dedup step across parallel GPU output directories (§9c)
3. Atlas-Chat self-transliteration wired into the pipeline, replacing the character-mapper (§10)
4. Frontier-model QA/correction pass on the transliterated output (§10)
5. A validation batch confirming 1–3 actually work, before committing the full 7,500-row / ~9.5h Kaggle run
