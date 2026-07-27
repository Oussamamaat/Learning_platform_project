# Training Dataset — Walkthrough, Decisions, and Problems Encountered

**Date:** 2026-07-27
**Scope:** Synthetic training data generation for the LoRA fine-tune (persona, tone, Darija/French code-switching)
**Status:** Pipeline validated end-to-end at small scale; full 7,500-row run pending two open decisions (below)

---

## 1. Original plan

Written 2026-07-26 (`DATA_PREP_PLAN.md`). The core split: RAG handles tenant-specific facts, a LoRA fine-tune handles *how the assistant talks* — persona, tone, code-switching between French and Darija — across three domains: industrial safety, physical security, blockchain compliance.

Hardware was fixed from the start: RTX 4060 (8 GB VRAM) locally for data generation, Kaggle's free T4 GPU for the actual LoRA training run.

**Dataset specification (unchanged since the original plan):**

| Component | Rows | Purpose |
|---|---:|---|
| Socratic pedagogy | 2,500 | Tutor explains a concept, then checks understanding |
| Code-switching fluency | 2,500 | Natural French/Darija mixing |
| Grounded context + refusal | 2,000 | Answer from context; refuse when context is insufficient |
| General capability preservation | 500 | Prevent catastrophic forgetting of general Darija fluency |
| **Total** | **7,500** | |

Language split target: 40% Darija Arabizi, 30% French, 30% mixed. Turn format: 60% multi-turn, 40% single-turn.

### Decision: generation model

The plan's own text initially named `qwen2.5-coder` as the data-generation model (already installed, reliable structured JSON output, fast). Its licensing table, in the same document, instead locks in **Atlas-Chat-9B** as the generator — a self-correction made before generation started.

**Reasoning:** `qwen2.5-coder` is a strong instruction-following generalist but has no particular Darija competency. The product requirement is "strict competency in the Moroccan dialect (Darija)" — that needs a generator that is native in the target dialect, not merely good at following instructions written in it. Atlas-Chat is explicitly a Darija-tuned base model built for this.

---

## 2. The discarded premade dataset

An earlier generator, `app/services/generate_data.py`, is still present in the repo history — the plan's own file inventory labels it "old road safety generator (deprecated)." The corpus it produced (`accidents_de_la_route.txt`, `code_de_la_route.txt`, `signalisation_routière.txt`, `sécurité_des_piétons.txt`, and others) covers traffic and road safety — a domain unrelated to what's live today.

**This was the mock data used for last week's RAG demo**, built before the CEO specified the real topics. Once the real domains (industrial, physical security, blockchain) were confirmed, this placeholder corpus was deleted rather than kept "just in case."

**Reasoning:** demoing or training against content nobody asked for wastes review time and risks anchoring the model on the wrong domain vocabulary.

---

## 3. Building the real grounding corpus (`raw/`)

`app/services/generate_corpus.py` produces 21 documents across the three real domains — Code du Travail, ISO 45001, LOTO, PPE for industrial; Loi 27-06 and 032-26 for physical security; Bill 42-25, FATF AML/CFT guidance, BAM/AMMC statements for blockchain.

**Decision:** the corpus is synthetic-but-realistic — written to match real Moroccan regulatory structure and terminology, not scraped from actual statute text.

**Reasoning:** scraping real legal text creates licensing exposure and a real risk of misquoting law the company could be held liable for. A generated corpus gives full control over content shape and — critically — guarantees no overlap between what the model was grounded on during training and what a real tenant's actual documents look like at inference. That non-overlap is an explicit requirement (contamination prevention), not incidental.

---

## 4. The generation pipeline (`app/services/generate_training_data.py`)

Four weighted prompt builders (matching the table in §1), each producing ChatML rows: `{"messages": [...], "component": ..., "domain": ...}`.

**The single most important design decision:** the system prompt used to generate every training row is byte-identical to `SYSTEM_PROMPT_TEMPLATE` in `app/services/llm.py`, the one used at serving time.

**Reasoning:** a model fine-tuned against one prompt shape and served a different one at inference degrades in ways that look like a model failure but are actually a data-generation bug — and are much harder to diagnose after the fact than to prevent up front.

---

## 5. First implementation — zero usable rows

The pipeline as originally written had ten structural bugs, caught in an earlier review: train/serve prompt mismatch, dropped guide files (orthography rules, code-switching rules loaded but never injected into prompts), an in-memory row buffer that risked a crash on long runs, `max_tokens` too low causing JSON truncation, and others. Net result: **0 rows produced.** That round of fixes was completed before this write-up's scope.

---

## 6. Correctness defects found this session

Each of these was caught by measuring pipeline output against known-good invariants, not by reading the code — none were visible by inspection alone.

| Defect | Root cause | Fix | Measured effect |
|---|---|---|---|
| JSON repair too brittle | A single stray character after a closing quote discarded an otherwise-valid response | Progressive repair + regex salvage as fallback | 1.06 → 1.25 valid rows/call (+18%), fixed 32-response test corpus |
| **Dedup false-collapse** | Deduplication embedding included the full system prompt, near-identical across a domain — swamped the actual conversation signal | Embed conversation turns only, exclude system prompt | Live: 24 rows → 4 survivors (broken) → 24 → 23 (fixed) |
| Malformed turn order | Rows could open on an assistant turn with no preceding user turn | Trim to first-user → last-assistant window | 0 defects, final validated batch |
| Silent row loss at export | `component` field trusted from the model's own JSON output; a mismatch dropped the row invisibly | Stamp component/domain from generation context, never from model output | Eliminated this loss class |
| No run visibility | `logging.basicConfig` was never called — a multi-hour run produced zero console output | Configured logging + `--log-level` flag | Restored |

**Why the dedup bug matters most:** it produced no error and no warning — just a smaller, plausible-looking output file. At the full 7,500-row target, this exact bug would have silently discarded most of the dataset with no signal anything was wrong until someone counted rows.

---

## 7. The quantization discovery

Running `ollama show` against the model in active use revealed it was running **Q2_K** — the worst of 14 quantizations published for Atlas-Chat-9B (3.8 GB, ~3.3 bits/weight).

**Root cause:** the HuggingFace repo publishes 14 quant files. An untagged `ollama pull` (no explicit `:Q_X` tag) resolves to whichever filename sorts first alphabetically — `Q2_K` does. Nobody selected this quantization deliberately.

**Why it matters:** Q2_K is documented to specifically degrade instruction-following and structured output — plausibly explaining symptoms previously attributed to "inherent model behavior" (malformed JSON, empty content fields, the model echoing the literal `"..."` placeholder from the prompt instead of a real answer).

**Status:** `Q4_K_M` (5.8 GB, ~4.8 bits/weight) has been pulled and confirmed via `ollama show`. **Not yet A/B-validated** against the Q2_K baseline — this is the first item under Next Steps.

---

## 8. The socratic-component teaching gap

An external review (independent LLM critique) raised a valid concern: does the socratic component actually *teach*, or does it just interrogate — asking a question without ever explaining anything first?

**Checked against the actual pipeline:**

- The five reference examples in `data/few_shot_examples.md` are correctly scaffolded — every assistant turn explains the relevant rule or principle, *then* asks one checking question. The reference material was never the problem.
- The prompt builder, `build_socratic_prompt()`, never explicitly instructed this pattern — it showed one randomly-sampled example and relied on the model to infer the structure on its own.
- A real generated sample examined this session showed exactly the failure mode described: both assistant turns were pure questions, with no explanation at all.

**Fix applied:** added an explicit instruction to `build_socratic_prompt()` requiring every assistant turn to state the relevant fact or principle in 1–2 sentences before ending with one checking question, and explicitly marking a question-only turn as invalid.

**Status:** untested — low risk, since it reinforces the pattern the reference examples already demonstrate rather than introducing a new one. Worth a quick validation pass on a handful of rows before the full run.

---

## 9. Open decisions

### 9.1 Arabizi vs. Arabic script — a real cost, not an open requirement

`DATA_PREP_PLAN.md` locks in Arabizi (Latin script + numerals) as a requirement, with a full numeral-mapping table. That part is settled — it is not up for debate.

What's actually unresolved: **Atlas-Chat's native output is Arabic script, not Arabizi** — a fixed property of the base model, not a prompting failure. Two ways to close that gap, each with a real cost:

1. Force compliance harder at generation time. Measured so far: only 20–40% of calls produce Arabizi without any post-processing.
2. Accept post-hoc transliteration as a permanent step. This is lossy — Arabic script omits the short vowels Arabizi needs, so transliterated text can never fully recover native-quality Arabizi.

**The decision needed:** which of these two costs to accept as the standard path for the remaining ~70% of generations that don't produce Arabizi natively.

### 9.2 Native-speaker validation

Before committing to the full 7,500-row run, a small native-speaker review (~50 rows) is the cheapest way to catch a systemic quality issue that can't be judged from the generation side alone — particularly relevant given the two items above are language-quality questions.

---

## 10. Next steps

1. Validate `Q4_K_M` against the `Q2_K` baseline — same fixed prompt set used for the §6 measurements, compared head-to-head on parse-failure rate, salvage rate, and Arabizi-compliance rate.
2. Resolve the Arabizi/transliteration decision (§9.1) before committing the full run to one path.
3. Native-speaker review of a small sample (§9.2).
4. Move the full 7,500-row generation run to Kaggle. Local generation was tracking toward 12+ hours serially, and concurrent requests on the 8 GB card made throughput *worse* (~45 tok/s → ~13 tok/s) rather than better, since concurrency forces part of the model off-VRAM. Kaggle's free tier (T4/P100, ~30 GPU-hrs/week) removes both constraints at no cost.
5. Run the full generation, then proceed to the QLoRA fine-tune on Kaggle per the original plan.
