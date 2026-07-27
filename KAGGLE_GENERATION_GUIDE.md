# Generating the Full Dataset on Kaggle

**Goal:** 7,500 ChatML rows, optimized for quality first and speed second.
**Strategy:** attempt vLLM (fast), fall back to Ollama + parallelism (safe). Validate on a small batch before committing to the full run.

---

## 0. Why this plan looks the way it does

Three measured constraints drive every choice below.

**Decode is memory-bandwidth-bound, not compute-bound.** For every token, the GPU reads all 5.8 GB of model weights. That read costs the same whether you generate one sequence or thirty-two. Single-stream generation therefore wastes most of the card. **Batching is the entire speedup** — not a faster chip.

**The local 8 GB card cannot batch.** Measured: concurrent requests forced part of the model off-GPU and throughput *dropped* from ~45 to ~13 tok/s. Model (5.8 GB) + multiple KV slots does not fit in 8 GB. A 16 GB T4 does have that headroom — this is the actual reason to move to Kaggle.

**Kaggle sessions cap at ~12 hours.** The local estimate is ~41 h. Anything short of a ~3.5× speedup needs checkpointing, which is why `--resume` exists (§5).

### Hardware gotchas specific to this model

Atlas-Chat-9B is **gemma2**-based. On Kaggle's T4 (compute capability 7.5):

| Issue | Consequence |
|---|---|
| T4 has **no native bf16** | Gemma2 was trained in bf16. Forcing fp16 risks overflow/NaN — fp16 has a much smaller dynamic range. |
| Gemma2 needs attention logit **soft-capping** | vLLM requires the FlashInfer backend for this. |
| bf16 weights ≈ **18.5 GB** | Does not fit one 16 GB T4. Needs `TP=2` across both T4s. |
| **No AWQ/GPTQ quant exists** | Only `MBZUAI-Paris/Atlas-Chat-9B` (bf16) and GGUF variants. The easy vLLM path is unavailable. |

This is why vLLM is time-boxed rather than assumed.

---

## 1. Pre-flight (do this before opening Kaggle)

The repo already contains everything needed. Confirm these three changes are present in `app/services/generate_training_data.py`:

- `--concurrency N` — parallel in-flight requests (thread pool around the generation loop)
- `--resume` — counts existing `*_raw.jsonl` rows toward each component target instead of deleting them
- Schema-constrained output (`ROW_SCHEMA` / `ROW_LIST_SCHEMA` passed as Ollama's `format`)

Verify locally:

```bash
python -m py_compile app/services/generate_training_data.py
python -m app.services.generate_training_data --help | grep -E "concurrency|resume"
```

**Upload as a Kaggle Dataset** (not a notebook file — it persists across sessions):

- `app/` (the whole package)
- `data/` — the four foundation files: `ortho_guide.md`, `code_switching_rules.md`, `refusal_templates.md`, `few_shot_examples.md`
- `raw/` — all 21 corpus documents

Kaggle mounts this at `/kaggle/input/<dataset-name>/`.

---

## 2. Notebook setup

Settings → **Accelerator: GPU T4 ×2** · **Internet: On** · **Persistence: Variables and Files**

```python
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

Confirm two T4s with ~15,360 MiB each before proceeding.

```python
import shutil, os
SRC = "/kaggle/input/<your-dataset-name>"
for d in ["app", "data", "raw"]:
    shutil.copytree(f"{SRC}/{d}", f"/kaggle/working/{d}", dirs_exist_ok=True)
os.chdir("/kaggle/working")
!ls raw/shared/*/text | head
```

---

## 3. Path A — vLLM attempt (time-box: 2 hours)

Set a timer. If it isn't producing coherent Darija within two hours, stop and go to §4. The failure modes here are real, not hypothetical.

```python
!pip install -q vllm
```

```python
import os
# Gemma2 soft-capping requires FlashInfer; without it vLLM either errors
# or silently produces degraded output.
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
```

Start the OpenAI-compatible server (background), so the existing HTTP client shape still applies:

```python
!nohup python -m vllm.entrypoints.openai.api_server \
  --model MBZUAI-Paris/Atlas-Chat-9B \
  --dtype float16 \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --port 8000 > /kaggle/working/vllm.log 2>&1 &
```

Model download is ~18.5 GB — expect 5–15 minutes before the server is ready.

```python
import time, urllib.request, json
for i in range(60):
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5)
        print("server up"); break
    except Exception:
        time.sleep(20)
else:
    print("did not come up — check vllm.log")
!tail -30 /kaggle/working/vllm.log
```

### Go / no-go criteria

Run one completion and read the output yourself:

```python
req = urllib.request.Request(
    "http://127.0.0.1:8000/v1/completions",
    data=json.dumps({
        "model": "MBZUAI-Paris/Atlas-Chat-9B",
        "prompt": "Chno khassni ndir bach nst3ml l-casque de securite f l-atelier?",
        "max_tokens": 200, "temperature": 0.7,
    }).encode(),
    headers={"Content-Type": "application/json"}, method="POST",
)
print(json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]["text"])
```

**Abort to §4 if any of these:**

- Output is `nan`, empty, or repeated single tokens → fp16 overflow, the documented gemma2-on-T4 failure
- Server OOMs during load → lower `--gpu-memory-utilization` to `0.85`, retry once, then abort
- FlashInfer install fails → abort (soft-capping is not optional for gemma2)
- Output is fluent but not Darija → wrong model resolved; check the log

**If it works:** you need an adapter, because the pipeline speaks Ollama's `/api/generate`, not OpenAI's `/v1/completions`. The schema-constrained `format` field is also Ollama-specific — vLLM uses `guided_json` instead. Budget ~30 minutes to write a `call_vllm()` alongside `call_ollama()` and switch on a flag. With `TP=2` and `--concurrency 16`, expect **~2–4 h** for the full run.

---

## 4. Path B — Ollama + parallelism (the safe path)

Reuses the exact GGUF and pipeline already validated locally. No model conversion, no dtype risk.

```python
!curl -fsSL https://ollama.com/install.sh | sh
```

**The critical setting** — `OLLAMA_NUM_PARALLEL` must be set on the *server process*, not the client. This is what your 8 GB card couldn't afford:

```python
import subprocess, os, time
env = dict(os.environ,
           OLLAMA_NUM_PARALLEL="4",
           OLLAMA_MAX_LOADED_MODELS="1",
           OLLAMA_KEEP_ALIVE="60m")
subprocess.Popen(["ollama", "serve"], env=env,
                 stdout=open("/kaggle/working/ollama.log","w"),
                 stderr=subprocess.STDOUT)
time.sleep(10)
!ollama pull hf.co/QuantFactory/Atlas-Chat-9B-GGUF:Q4_K_M
```

**VRAM budget on a 16 GB T4:** model 5.8 GB + 4 KV slots × ~0.8 GB (at `num_ctx=4096`) ≈ **9 GB**. Comfortable. You can try `OLLAMA_NUM_PARALLEL=8` (≈12.2 GB) for more throughput — but verify with `nvidia-smi` that nothing spilled to CPU, since spilling makes it *slower*, not faster.

**`--concurrency` must match `OLLAMA_NUM_PARALLEL`.** Exceeding it just queues requests server-side and gains nothing.

### Using both T4s

Ollama does not tensor-parallelize across GPUs. To use the second card, run a second server bound to it and split the work:

```python
env2 = dict(env, CUDA_VISIBLE_DEVICES="1", OLLAMA_HOST="127.0.0.1:11435")
subprocess.Popen(["ollama", "serve"], env=env2, ...)
```

Then run two generation processes with different `--ollama-url` and `--output-dir`, and merge the outputs. Roughly doubles throughput. Skip this on the first pass — get one GPU working first.

---

## 5. Staged execution

### Stage 1 — small batch (~200 rows)

```python
!python -u -m app.services.generate_training_data \
  --target-rows 200 \
  --concurrency 4 \
  --model hf.co/QuantFactory/Atlas-Chat-9B-GGUF:Q4_K_M \
  --output-dir /kaggle/working/out_sample
```

**Review gate — do not skip.** Check:

- `parse_failures` should be ~0 (schema-constrained). Anything above ~5% means the schema isn't being applied.
- `transliterated` percentage — expect ~100%; this is the known Arabizi ceiling, not a bug.
- Dedup survival: `Total rows before dedup` vs after. A large drop means prompt diversity is too low at scale — the first real signal worth acting on.
- Read ~10 rows yourself. Socratic rows must **explain, then ask** — never question-only.

Download `out_sample/train.jsonl` for the native-speaker review (§7).

### Stage 2 — full run

Time each component from Stage 1 and extrapolate before launching. If the projection exceeds ~10 h, split by component across sessions.

```python
!python -u -m app.services.generate_training_data \
  --target-rows 7500 \
  --concurrency 4 \
  --model hf.co/QuantFactory/Atlas-Chat-9B-GGUF:Q4_K_M \
  --output-dir /kaggle/working/out_full 2>&1 | tee /kaggle/working/run.log
```

**Copy `*_raw.jsonl` to `/kaggle/working/` frequently** — that is what survives a session kill and what `--resume` reads.

### If the session dies

```python
!python -u -m app.services.generate_training_data \
  --target-rows 7500 --resume \
  --concurrency 4 \
  --model hf.co/QuantFactory/Atlas-Chat-9B-GGUF:Q4_K_M \
  --output-dir /kaggle/working/out_full
```

Existing rows count toward each target and seed the dedup set, so nothing is duplicated or regenerated.

> **Important:** `--resume` only helps if the run died **during generation**. Raw files are consumed and deleted at the dedup stage, so a crash after that point means re-running. That matches the realistic failure (session timeout mid-generation).

---

## 6. Known scaling risk

`deduplicate()` rebuilds a growing NumPy array on every iteration — O(n²) with reallocation. Invisible at 21 rows; at 7,500 it may take considerably longer than the ~20–30 min estimated. If it stalls, the fix is to pre-allocate the embedding matrix once and vectorize the comparison. Worth watching, not worth pre-optimizing.

---

## 7. Native-speaker review → the Arabizi decision

This is the gate that determines whether the dataset is actually usable, and it cannot be resolved by any GPU or quantization choice.

Every row carries a `transliterated` flag. Current measurement: **100% of rows are transliterated** — Atlas-Chat emits Arabic script, and the character mapper converts it. Arabic script omits short vowels, so `l-masna3` becomes `almsn3`. That loss is structural.

Have a Darija speaker read ~50 rows from the Stage 1 sample and answer one question: **is the Arabizi natural enough to train on?**

- **Yes** → proceed with the full run as-is.
- **No** → add a transliteration pass (Claude Haiku + Batch API, ~$8 for the full dataset) that converts Arabic script to idiomatic Arabizi. This is re-scripting text you already have, not generating new Darija — the task where a frontier model is strongest and Atlas-Chat is weakest. Filter on the `transliterated` flag to route only affected rows.

---

## 8. Quick reference

| Setting | Value | Why |
|---|---|---|
| `OLLAMA_NUM_PARALLEL` | 4 (try 8) | Server-side; must exceed 1 or nothing batches |
| `--concurrency` | match the above | Client-side in-flight requests |
| `num_ctx` | 4096 (in code) | Prompts are ≤1,500 chars; 8192 wasted KV cache |
| `num_predict` | 1024 (in code) | Measured outputs are 250–400 tokens |
| `keep_alive` | 30m (in code) | Stops Ollama evicting 5.8 GB mid-run |
| vLLM `--dtype` | `float16` | T4 has no bf16 — this is mandatory, not a preference |
| vLLM `--tensor-parallel-size` | 2 | 18.5 GB bf16 weights don't fit one 16 GB T4 |
