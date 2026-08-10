# Fine-tune pipeline

## Base model

Atlas-Chat-9B (Gemma-2-9b-it lineage), pretrained on ~450,000 Darija instructions.
Chosen because the project's own training set (~3,000 rows) is under 1% of that —
too small to teach a language, sized only to reshape behavior in a model that already
has one. Same model used for synthetic data generation (self-distillation): it can
reshape behavior, not teach knowledge Atlas doesn't already have.

## Generation

Kaggle, dual T4, `CUDA_VISIBLE_DEVICES`-pinned Ollama worker processes, Q4_K_M GGUF.
Two operational guards exist because of past silent failures: an explicit
`--accelerator NvidiaTeslaT4` flag (an unpinned request once landed on a single P100),
and a fail-fast `nvidia-smi -L` assertion plus forced-CPU cross-shard dedup (GPU dedup
silently failed once on Kaggle/CUDA, shipping 32% of a run undeduplicated).

Current dataset: 3,064 rows (2,757 train / 307 eval), 11 components, weighted by
intent (socratic 800, grounded_refusal 700, code_switching 700, quiz_generation 400,
and 7 smaller components — see `FINETUNING_RATIONALE.md` for the full table).

**Proven-working notebook for this task:** [`kaggle_finetune_v11.ipynb`](../../kaggle_finetune_v11.ipynb)
— the notebook that produced the adapter currently in production. Copy this one when
rebuilding a fine-tune kernel, not an older version.

## Fine-tune configuration

LoRA via Unsloth, QLoRA 4-bit on the T4s.

| Setting | Value |
|---|---|
| r / alpha | 16 / 16 (54,018,048 trainable params, 0.585% of 9.24B) |
| Targets | q, k, v, o, gate, up, down (not `embed_tokens`/`lm_head` — not teaching new tokens) |
| Dropout | 0 (required by Unsloth's fused kernel path) |
| Epochs | 2, checkpointed per epoch |
| LR / schedule | 2e-4, cosine, 3% warmup |
| Precision | fp16 (T4/P100 have no bf16) |
| Optimizer | `adamw_8bit` |

## Merge to standalone

The adapter is merged into standalone fp16 weights, then quantized to GGUF (Q4_K_M),
via Unsloth locally (`model.save_pretrained_gguf(..., quantization_method="q4_k_m")`).
`maximum_memory_usage` is a RAM-pacing knob only — it does not affect output precision,
which is governed solely by `quantization_method`. On an 8GB-VRAM / 16GB-RAM laptop,
this step takes roughly 40 minutes and is RAM-bound, not GPU-bound; lowering
`maximum_memory_usage` trades speed for headroom if the default value causes a RAM
crash mid-merge.

The merged GGUF is registered in Ollama via a `Modelfile` (`FROM <gguf path>`, no
`ADAPTER` line — the adapter is already baked in) and pointed to by
`app/config.py`'s `ollama_model` setting.

**Detail & rationale:** `../../FINETUNING_RATIONALE.md`, `../../LOCKEDIN_PLAN.md`,
`../../resurrection.md` §1, §5, §6.
