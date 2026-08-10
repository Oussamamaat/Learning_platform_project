# IBLOG Adaptive Learning Tutor — Architecture, Decisions, and Failure History

Status report generated from the project documentation (2026-08-03).

---

## 1. Architecture

**Platform model.** B2B multi-tenant, domain-agnostic adaptive learning platform. Each tenant uploads its own course materials and brings its own language mix; the AI tutor (RAG + fine-tuned LLM) tutors that tenant's users from their own documents. Onboarding = documents + per-tenant LoRA adapter, no code changes. Multilingual by design — French primary, Darija secondary but mandatory, English in scope.

**Tenant #1 (current live instance).** Moroccan safety/security regulations ("sécurité et sûreté"), language mix Darija (Arabic script) + French. Served by `atlas-darija-tutor` (Atlas-Chat-9B LoRA, GGUF via Ollama).

**Serving architecture** (`LOCKEDIN_PLAN.md` §2):
- One **frozen base** + multiple narrow **per-task LoRA adapters**, hot-swapped per request via **vLLM multi-LoRA** (production target). Quiz is an adapter + JSON schema on decode, not a second model.
- **Ollama** used for generation/dev only. **vLLM** was evaluated for generation and rejected (no AWQ/GPTQ quant for Atlas; T4 PCIe interconnect).
- **Audio**: separate models — Darija Whisper STT (tenant #1); TTS vendor still **not selected** (single largest open MVP item).
- **RAG**: PostgreSQL + pgvector, chunked (~400 chars / 50 overlap), top-5 retrieval.
- **Fine-tune hardware**: ASUS ROG Strix G16, 8GB VRAM; Kaggle dual-T4 for generation.

**Data pipeline** (`LOCKEDIN_PLAN.md` §3): 11 components, 3,064 rows (`v11_merged`), ~90/10 train/eval. Gate architecture rejects-and-retries on ~10 enforced gates (structural, grounding, register, behavioral) — every fix is a *gate*, not stronger prompt wording.

**Fine-tune config** (`FINETUNING_RATIONALE.md`): r=16/alpha=16, dropout=0, fp16 (T4 has no bf16), batch 1×16 accum, 2 epochs, lr 2e-4 cosine 3% warmup, adamw_8bit, all 7 linear targets (no embeddings), max_seq 4096, custom chat-template merge for train/serve byte-parity.

## 2. Decisions made (and the ones now locked)

| Decision | Outcome |
|---|---|
| Generate **and** fine-tune Atlas-Chat-9B, not Qwen2.5-7B | Qwen has ~0 Darija (DODa BLEU 0.92 vs Atlas 28.08); dataset is behavior transfer, not language acquisition. **Now scoped as tenant #1's instance decision.** |
| **Platform base direction** | Neutral multilingual base + per-tenant LoRA **before tenant #2** — a tenant's language/domain becomes an adapter, not a baked-in base. |
| Output script = **Arabic script**, not Arabizi | Forced Arabizi caused the 76%-unusable defect; Arabic script gives natural French code-switching. |
| Dataset **3,000 rows**, not 7,500 | Corpus ceiling, LoRA capacity, and 15% review capacity all cap at ~3,000. |
| **6 → 11 components** | 5 added to close measured zero-coverage gaps (structured_explanation, learner_adaptation, general_knowledge_disclosed, no_context_refusal, injection_resistance). |
| Citation handling = **deterministic post-processing** (CiteFix-style), not model-memory | Correct + train/serve identical. |
| Cross-shard dedup, forced CPU | GPU dedup silently failed once, shipped 32% undeduped. |
| 80/20 domain split (client/generalization) | Domain-agnostic behavior without diluting the current tenant's data. |

## 3. Problematic outputs (chronological, worst first)

**The one that reopened the dataset — citation fabrication below the base's floor** (`LOCKEDIN_PLAN.md` §6.2). First fine-tuned adapter: on a context with no citable law, it **confidently fabricated** «حسب القانون 27-04...» 2/2 times, while **base Atlas-Chat correctly refused** 2/2. Root cause was distributional (legal Q&A always cited something), not a few bad rows. The first training pool had **653 fabricated references (24.6%)**; `v11_merged` has 0. This made base-vs-adapter comparison a *standing* requirement.

**French answers / routing** (`green_light_model.md` §4.2, §4.2a):
- French question ("Explique-moi la procédure LOTO.") answered in Darija. Root-caused: the **document's** language, not the question's, drove output language — reproduced on base Atlas, so not an adapter defect. Fixed with separate `SYSTEM_PROMPT_TEMPLATE_FR` + `detect_query_language()` router. Bonus routing bug: `"explique-moi"` didn't split under whitespace-only tokenization; fixed by splitting hyphens + pronoun markers.
- **Residual**: French **refusals still come back in Darija**. Three fix attempts failed (instruction, hard constraint, exemplar — model copied the exemplar and still answered Darija). `grounded_refusal` is 417 rows, 100% Arabic-script; that prior beats any prompt. Needs French refusal rows in the data, not prompt work.

**Zero-French rate 91.7%** (`green_light_model.md` §4.1b). "MANDATORY: use ≥1 French term" was pure instruction with no gate; even French-sourced documents only produced French 23% of the time. Fix: answerable-row French gate → 0.0% in `v11_merged`. Also a metric-scoping error (refusal rows counted, floor of ~45% regardless of quality).

**Quiz answer keys contradicting explanations — 16%** (12/76) (`CHANGELOG.md`). A learner told "correct" would actually be wrong. Fixed with `_explanation_supports_answer()` self-consistency gate.

**Arabizi transliteration → 76% unusable** (`LOCKEDIN_PLAN.md` §2.5). Letter-mapped Latin output like `"Bach nbdaw had al3mlia, wach mmkn t9wl lyna smyt alchrka"` — no French at all. Fixed by deleting the transliteration stage entirely, not improving it.

**Darija detector blind to ك-prefix** (`CHANGELOG.md`). Missed Darija's most distinctive feature (كيطلب، كتقول); a real sentence scored `dar=0` and was misread as MSA drift — invalidated a "drifted to MSA" conclusion that was wrong.

**Multi-turn collapse 37.9% vs ~50%** (`CHANGELOG.md`, `green_light_model.md` §4.1). 56/76 rows single-turn; the gate required every turn to pass French thresholds, so multi-turn rows had 2–3 chances to fail. Also `multi_turn_pct` in config was **dead config** — never read anywhere. Fixed by per-attempt roll + `turn_count_mismatch` gate.

**Code-switch gate at the median** — gate threshold 2, median French-per-turn 3, 55% of turns sat at the boundary → ~33% pass. A gate at the median bisects the distribution instead of cutting the bad tail.

**Dedup failures** (`CHANGELOG.md`, `what_next.md`): (a) dedup embedding included the near-identical system prompt, swamping the conversation signal (24→4 instead of 24→23); (b) `deduplicate()` **silently swallowed CUDA failures** via broad `except Exception` and shipped **969 undeduped rows (32.3%)** as if successful — forced CPU + loud failure; (c) per-shard dedup missed cross-GPU duplicates → train/test leakage, fixed by `merge_shards.py`.

**Citation extractor — 4 bugs** (`CHANGELOG.md`): prefix collision (`"المادة 1"` corrupting `"المادة 12"`), double-wrap gloss, missing law-number extraction (`القانون رقم 27.06` invisible), and instruction-example leakage (example citations in the prompt counted as real context). Net: apparent 40% recall was really 61% once the tooling could detect the model's actual citations.

**Citation recall 53.2% → reclassified**: part metric scoping (refusal rows in the "must cite" denominator; real answerable recall was 72.6%), part genuinely missing gates on socratic/code_switching/quiz (real gap 57–63%), and part composition (French-source docs only 14% citable). Final `v11_merged`: 78.9%.

**CJK contamination** — 2% of rows had stray CJK mid-Arabic (`"شنو هي義務 المشغل"`). Fixed with `_CJK` regex.

**First pipeline produced 0 rows** — ten structural bugs (prompt mismatch, dropped files, buffer crash risk, JSON truncation, silent logging). Fixed before the work documented.

**Quantization trap** — the model was running on **Q2_K** (worst of 14 quants) because untagged `ollama pull` resolves alphabetically; plausibly explained earlier "inherent model behavior" symptoms. Fixed by pinning Q4_K_M.

**Other notable incidents**: 73% train/serve RAG context mismatch (socratic/code_switching trained on no real document); `grounded_refusal` over-generation feeding 20% dedup loss (`maxItems: 2` fix); French-source citation low on refusal; the v11 refusal that wrongly claimed topic-scope restriction (safe but demo-ugly); GGUF missing from the deliverable zip (adapter intact); the stall/died 3,000-row Kaggle run (silent restart, single-GPU usage, P100-vs-T4 accelerator mismatch — now a fail-fast `nvidia-smi -L` assert).

## 4. Current status

§4.1 dataset gate is **GREEN** (all 8 checks pass with margin). Second fine-tune completed (`eval_loss` 1.1823 → 0.8407), **above the base-model grounding floor**. Remaining: fine-tune → live demo (≥8/10) → manual read → CEO acceptance; TTS selection and neutral base-model selection open.
