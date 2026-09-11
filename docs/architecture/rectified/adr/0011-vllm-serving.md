# ADR 0011: vLLM Serving for Both Tutors

**Status:** Accepted on Phase B evidence. Production flags are provisional until the D5 staging run.
**Date:** 2026-09-11
**Depends on:** ADR 0008 (amended 2026-09-11), ADR 0003 (structured quiz output), `docs/architecture/cloud-scaling-plan.md` §3 and §5 item 4

## Problem

Today's deploy runs Ollama with `OLLAMA_NUM_PARALLEL` unset, so each tutor model answers one
request at a time. Measured on an RTX 5090 (below), throughput stays flat at about 0.14 requests
per second no matter how many learners are waiting, and p95 latency with 16 simultaneous requests
is 113 s (Darija) and 126 s (French). Every extra concurrent learner adds a full answer's worth of
waiting.

## Investigation method

- **Hardware and runs.** One RTX 5090 (32,607 MiB) on Akash, 2026-09-11. The unattended job
  `scripts/lease/job.sh` ran twice: kit `c9480aa`, then a rerun with kit `2ff045d` after the fixes
  listed under *Defects found*. All results are in the private dataset
  `Oussamamaat/iblog-vllm-lease` under `results/public/` and `results/logs/`; the rerun overwrote
  run 1's invalid French vLLM and quality files, which remain in that repo's history.
- **Identical input.** Both backends received the same rendered prompts: 20 real RAG prompts per
  language from `scripts/vllm/fixtures/bench_prompts.json` (longest 2,015 tokens),
  `max_tokens` 300, temperature 0, N = 1, 2, 4, 8, 16, 32 concurrent workers, 2 requests per worker.
- **Parity.** `scripts/vllm/parity_probe.py` compares Ollama's `prompt_eval_count` with vLLM's
  `usage.prompt_tokens` for the same rendered conversation.
- **Quality.** `scripts/vllm/quality_sample.py` asks both backends the 7 `green_light_model.md`
  cases with the same retrieved context. Ollama's answers are collected while it has the GPU to
  itself, vLLM's afterwards. Graded by reading, not by script.
- **AWQ builds.** `scripts/vllm/build_merged_awq.py`: LoRA merge, then llm-compressor AWQ
  (`W4A16_ASYM`) with 256 calibration rows per language, published to `Oussamamaat/iblog-tutor-awq`
  (`darija/`, `french/`; `model.safetensors` 6.16 GB each).

## Options considered

- **Stay on Ollama and raise `NUM_PARALLEL`.** Measured; fails the gate. French with
  `NUM_PARALLEL=4` still has p95 82.7 s at N=16 (ADR 0008 amendment).
- **vLLM, two processes on one GPU, AWQ INT4 weights.** Chosen.
- **vLLM with FP8-W8A8 weights.** Rejected: about 9.5 GB per model leaves little KV cache for two
  instances on one card, and it is not like-for-like with Ollama's Q4_K_M.
- **vLLM `/v1/chat/completions`.** Rejected: the adapter's chat template emits `{{ bos_token }}`,
  which would double the BOS token. The app renders the prompt itself with `render_conversation()`
  and calls `/v1/completions`.
- **One base model with multiple LoRAs.** Not possible: the tutors use different base models
  (Atlas-Chat-9B for Darija, gemma-2-9b for French).

## Evidence

### Parity (plan criterion 3): pass

| Conversation | Ollama `prompt_eval_count` | vLLM `prompt_tokens` |
|---|---|---|
| trained-shape 4-message (French) | 152 | 152 |
| single-turn (Darija) | 109 | 109 |
| trained-shape 4-message (Darija) | 151 | 151 |
| one-past-trained 6-message (Darija) | 195 | 195 |

### Performance (plan criterion 5: vLLM p95 under 10 s at N=16 per language): pass

p95 latency in seconds, throughput in requests per second. Full per-N tables, including p50, peak
VRAM and per-request rows, are in the `bench_*.json` files.

**Darija**

| Config | p95 N=1 | p95 N=8 | p95 N=16 | p95 N=32 | Throughput N=16 | Throughput N=32 | Errors N=32 |
|---|---|---|---|---|---|---|---|
| Ollama, `NUM_PARALLEL` unset | 79.2 ¹ | 59.5 | 113.2 | 231.4 | 0.15 | 0.14 | 0/64 |
| Ollama, `NUM_PARALLEL=4` ² | 88.0 ¹ | 335.3 | 419.9 | 400.9 | 0.03 | 0.03 | 44/64 |
| vLLM AWQ, bf16 KV cache | 0.8 | 1.1 | **1.7** | 3.0 | 10.72 | 10.93 | 0/64 |
| vLLM AWQ, fp8 KV cache | — | — | 3.2 | 2.4 | 7.35 | 13.51 | 0/64 |

**French**

| Config | p95 N=1 | p95 N=8 | p95 N=16 | p95 N=32 | Throughput N=16 | Throughput N=32 | Errors N=32 |
|---|---|---|---|---|---|---|---|
| Ollama, `NUM_PARALLEL` unset | 13.7 | 60.8 | 125.8 | 251.7 | 0.13 | 0.13 | 0/64 |
| Ollama, `NUM_PARALLEL=4` | 16.2 | 45.7 | 82.7 | 170.7 | 0.23 | 0.20 | 0/64 |
| vLLM AWQ, bf16 KV cache | 0.7 | 1.3 | **2.0** | 3.5 | 8.74 | 9.63 | 0/64 |

¹ Includes loading the model on the first request.
² Not a fair measurement of `NUM_PARALLEL=4`: the run started with the other Ollama server's model
still resident (peak 31,160 of 32,607 MiB), so it was starved of GPU memory. See ADR 0008's
amendment.

Median output length was comparable across backends (Darija: 55 tokens vLLM, 66–76 Ollama;
French: 70 vLLM, 82 Ollama), so vLLM's latency advantage is not an artefact of shorter answers.

Measurement conditions that make these vLLM numbers conservative or partial:

- Both vLLM instances ran with `--gpu-memory-utilization 0.45 --max-num-seqs 8`
  (`serve_pair.sh`'s auto-KV path), so each processed at most 8 requests together. N=16 and N=32
  were queueing inside vLLM.
- `max_tokens` was 300; the app sends 1,024.
- Each language was benchmarked alone; mixed Darija/French load was not measured.

### Quality (plan criterion 4): pass on the sample

| Case (`green_light_model.md`) | Ollama Q4_K_M | vLLM AWQ |
|---|---|---|
| A5/D3 grounded with citation, French | Article 283 and Loi N° 65-99 (both in context) | Article 283 |
| A5/D3 grounded with citation, Darija | Article 283, plus training duty (in context) | Article 283, plus Article 284's equipment list without naming 284 |
| D2/RF9 insufficient-context refusal, French | Refuses, points to the Code du Travail | Refuses, names the document's scope |
| D2/RF9 insufficient-context refusal, Darija | Refuses | Refuses (near-identical wording) |
| D1/RF11 off-topic refusal, French | Refuses | Refuses |
| B1 explain-then-question, Darija | Explains, asks no question | Explains, asks no question |
| E2 domain isolation, French | Says the document doesn't cover crypto-assets | Same |

- No fabricated citations (RF10) on either side: every article and law number appears in the
  retrieved context.
- No wrong-language answers.
- Failures shared by both backends, so not introduced by AWQ or vLLM: no follow-up question in
  B1 (RF6), and "(EPI)" bracketed after the Arabic term (RF4).
- fp8 KV cache answers match the bf16 run almost word for word (`quality_transcripts_fp8.md`).

### Defects found on the lease and fixed

1. **Modelfiles missing from `Oussamamaat/iblog-tutor-gguf`.** Only the two `.gguf` files had been
   uploaded; `job.sh` also needs each tutor's Modelfile to register it. Uploaded from the local
   Ollama registration (`ollama show --modelfile`), with `FROM` rewritten to the relative GGUF
   filename, the same convention as `scripts/docker/prepare_models.py`.
2. **French answers ran to `max_tokens`.** vLLM matches stop strings after special tokens are
   stripped from the decoded text, so `<end_of_turn>` never stopped generation. Darija stopped only
   because Atlas-Chat-9B's config lists token 107 as EOS; the French build from
   `unsloth/gemma-2-9b` lists only `<eos>`. Run 1's French answers had a median of 287 tokens and
   invented `user`/`model` turns. Every vLLM request now also sends `stop_token_ids: [106, 107]`
   (commit `3d99a51`, lease kit `2ff045d`); the rerun's French median is 73 tokens.
3. **Ollama and vLLM shared the GPU.** In run 1, B4 queried Ollama while vLLM held 90% of the card:
   Ollama got 1.5 GiB, ran mostly on CPU, and all 7 quality requests timed out at 3 minutes. Ollama's
   runner then stayed loaded long enough to leave B6's French fp8 instance 0.68 GiB short at
   startup. `job.sh` now collects Ollama's quality answers before vLLM starts, unloads Ollama and
   waits for an idle GPU before every vLLM start, and skips benchmarks and AWQ builds already
   published (commit `3d99a51`, lease kit `2ff045d`).

## Decision

1. Production LLM serving moves to vLLM `v0.29.0` (`vllm/vllm-openai:v0.29.0`), selected with
   `llm_backend="vllm"`. Ollama stays as a one-redeploy rollback (`llm_backend="ollama"`, still the
   default).
2. Two vLLM processes on one RTX 5090, launched by `scripts/vllm/serve_pair.sh`: Darija
   (`iblog-tutor-darija-awq`, port 8101) and French (`iblog-tutor-fr-awq`, port 8102), serving the
   AWQ W4A16 builds from `Oussamamaat/iblog-tutor-awq`.
3. Requests use raw `/v1/completions` with prompts from `render_conversation()`, the `stop` strings
   plus `stop_token_ids: [106, 107]`, and `structured_outputs` for quiz and diagram JSON.
4. Provisional production flags, to confirm in D5 from vLLM's own startup log:

   | Flag | Starting value | Basis |
   |---|---|---|
   | KV cache dtype | `auto` (bf16) | fp8 answers matched, but fp8 was only measured under an 8-request cap, which hides the extra capacity it exists to provide; revisit in D5 |
   | `--kv-cache-memory-bytes` | `6G` (6 GiB) per instance | Computed, not measured (see below). After both models' weights (5.7 GiB each) this leaves about 8 GiB of the card for CUDA contexts, graphs and activations. The uppercase `G` matters: vLLM reads `6g` as 10^9 bytes |
   | `--max-num-seqs` | `16` | Phase B's cap of 8 was already saturated at N=16 |
   | `--max-model-len` | `8192` | Prompt budget in `app/config.py` |
   | `--enable-prefix-caching` | on | Shared system prompt across requests |
   | `llm_max_concurrent` (app) | `64` | 32 per language, the highest load measured with zero errors |

   KV cache cost per token, from the models' `config.json` (both: 42 layers, 8 KV heads, head
   dimension 256): 2 × 42 × 8 × 256 × 2 bytes ≈ **0.33 MiB in bf16**, 0.16 MiB in fp8. Gemma-2's
   sliding-window layers (4,096 tokens) save nothing at these prompt lengths. A 6 GiB cache is
   about 18,700 tokens, roughly 6–9 simultaneous requests of 2,000–3,000 tokens per language
   before prefix-cache sharing.

## Rationale

- With 16 simultaneous requests, vLLM's p95 is about 65 times lower than today's Ollama
  (Darija 1.7 s vs 113.2 s, French 2.0 s vs 125.8 s), throughput is 65–75 times higher, and there
  were no errors up to 32 simultaneous requests, all while capped at 8 requests processed together.
- The same prompt becomes the same tokens on both servers, and the answers are behaviourally
  equivalent on the sample.
- Rollback is one setting and a redeploy, not a code change.

## Constraints acknowledged

- **KV cache sizes were not captured.** vLLM logs `GPU KV cache size` and `Maximum concurrency` at
  startup, but `job.sh` did not publish those lines, and they were lost when the lease closed. The
  capacity figures above are computed. D5 must record the real lines.
- **`serve_pair.sh`'s explicit `--kv-cache-memory-bytes` path has not run on real hardware.** Phase B
  used the auto path; D5 is its first real run.
- **Benchmark scope.** `max_tokens` 300 rather than 1,024, one language at a time, 2 requests per
  worker.
- **fp8 KV cache is inconclusive.** At N=16 it was slower than bf16 (p95 3.2 s vs 1.7 s); at N=32
  faster (2.4 s vs 3.0 s, 13.5 vs 10.9 requests per second).
- **Quality sample is 7 cases.** D5 exercises the full verification list.
- **The French AWQ artifact still lists only `<eos>` as EOS in its config.** The app is unaffected
  because it sends `stop_token_ids`; any other client of that artifact must do the same, or the
  build must write token 107 into `generation_config.json`.
