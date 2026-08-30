# Runbook — deploy IBLOG Tutor to a rented RTX 5090 (Akash)

Full-throughput / VRAM-heavy testing on a 32 GB Blackwell card: both tutor models
resident, the voice (STT/TTS) pipeline, PaddleOCR-VL, and high-batch embeddings —
all the things the 8 GB laptop can't run at once.

**What you build once, reuse every session:** a GPU image with the three venvs,
embeddings (bge-m3), TTS voices, and the STT engines baked in.

**The two tutor GGUFs are NOT in the image** — at ~5.5 GB each they blew out the
build host's disk, so they live in a private HuggingFace repo and the entrypoint
downloads them on first boot onto the persistent `/models` volume. That download
happens **once per fresh volume**: keep the volume and restarts are instant;
destroy it and the next boot re-fetches ~11 GB.

**What this runbook does *not* do for you:** sign or pay Akash transactions (needs
your Keplr/Leap wallet + AKT). Every signing step is called out.

---

## 0. Prerequisites (one-time)

- **Docker** on the dev laptop (the machine that already has the models in Ollama).
- **Ollama models present locally** — verify:
  ```
  ollama list        # must show IBLOG_TUTOR:latest and iblog-tutor-fr:latest
  ```
- **A container registry.** This repo is wired for **GHCR**:
  `ghcr.io/oussamamaat/iblog-tutor:gpu` (lowercase is a registry requirement).
  You need a **GitHub PAT (classic) with `write:packages`** to push
  (<https://github.com/settings/tokens>).
  The package **must end up Public** — an Akash SDL has no `imagePullSecrets`, so
  providers pull anonymously and a private package simply fails to deploy. The
  image holds no secrets and no tenant uploads (`raw/` is public regulatory
  explainer text), so publishing it is safe.
- **A HuggingFace READ token** for the boot-time GGUF download
  (<https://huggingface.co/settings/tokens>). Make it **fine-grained, read-only,
  scoped to `Oussamamaat/iblog-tutor-gguf`** — Akash providers can read SDL env
  values, so never put the write token used for the upload in there.
- **An Akash wallet** with a few AKT + a 0.5 AKT deposit per deployment
  (Keplr/Leap), and either the **Akash Console** (console.akash.network) or the
  `provider-services` CLI.
- Disk: ~15 GB free locally for the extracted GGUFs + image build.

---

## 1. Extract the tutor GGUFs (local, one-time per model change)

Pulls each model's GGUF out of Ollama's blob store and writes a Modelfile next to
it into `config/models/` (git-ignored):

```
python scripts/docker/prepare_models.py
```

Expect `config/models/IBLOG_TUTOR.gguf`, `iblog-tutor-fr.gguf`, their `.Modelfile`s,
and `manifest.json` (~11.6 GB total).

> These `.gguf` files are the **only copies** of the fine-tunes outside Ollama's
> blob store. Don't delete them until step 1b has finished uploading.

---

## 1b. Host the GGUFs on HuggingFace (one-time per model change)

The image ships only the tiny `.Modelfile`s; the weights are fetched at boot.

```
setx HF_TOKEN hf_xxx          # a WRITE token, for the upload only
.gguf_venv/Scripts/python.exe scripts/docker/upload_models_hf.py \
    --repo Oussamamaat/iblog-tutor-gguf
```

Uploads ~11 GB and prints the exact env lines. They are already filled into
`deploy/akash-deploy.yaml`, because the filenames are pinned by the `.Modelfile`
stems that ship in the image:

| Env var | Value |
|---|---|
| `IBLOG_TUTOR_GGUF_URL` | `https://huggingface.co/Oussamamaat/iblog-tutor-gguf/resolve/main/IBLOG_TUTOR.gguf` |
| `IBLOG_TUTOR_FR_GGUF_URL` | `https://huggingface.co/Oussamamaat/iblog-tutor-gguf/resolve/main/iblog-tutor-fr.gguf` |

Verify both files are actually present before deploying — a missing file shows up
much later as a confusing "no model named…" at chat time:

```
.gguf_venv/Scripts/python.exe -c "from huggingface_hub import HfApi,get_token;\
print([s.rfilename for s in HfApi(token=get_token()).repo_info('Oussamamaat/iblog-tutor-gguf').siblings])"
```

---

## 2. Build the GPU image

From the **repo root** (build context = `.`):

```
docker build -f config/Dockerfile.gpu -t ghcr.io/oussamamaat/iblog-tutor:gpu .
```

Notes:
- First build is long (three venvs incl. CUDA torch, + a ~2.2 GB embedding
  pre-download). Later builds after only code changes reuse the cached model/dep
  layers and are fast.
- If `paddlepaddle-gpu` has no Blackwell wheel, the build **prints a warning and
  continues** — you'll run with `OCR_ENGINE=none` (chat/voice/embeddings
  unaffected). This is expected "OCR best-effort".

---

## 3. Push to GHCR

Log in with the **PAT** (`write:packages`) — not your GitHub password:

```
echo <YOUR_PAT> | docker login ghcr.io -u Oussamamaat --password-stdin
docker push ghcr.io/oussamamaat/iblog-tutor:gpu
```

The push is large the first time (~20–25 GB). Subsequent code-only pushes send
just the thin top layer.

**Then make the package Public** (required — Akash pulls anonymously):
github.com → your profile → **Packages** → `iblog-tutor` → **Package settings** →
*Danger Zone* → **Change visibility → Public**.

Confirm an anonymous pull works, which is exactly what the provider will do:

```
docker logout ghcr.io
docker manifest inspect ghcr.io/oussamamaat/iblog-tutor:gpu > /dev/null && echo "publicly pullable"
```

If that errors, the provider will fail the same way and the lease will sit in
`pending` — fix visibility before deploying.

---

## 4. Fill in the SDL

`deploy/akash-deploy.yaml` is the **committed template** and must stay
secret-free. Render the deployable copy (gitignored) instead:

```
HF_READ_TOKEN=hf_xxx bash deploy/make-local-sdl.sh
```

That writes `deploy/akash-deploy.local.yaml` with a freshly generated 32-char
Postgres password (substituted in `POSTGRES_PASSWORD` *and* `DATABASE_URL`, which
must always match) and your HF read token, then fails loudly if any placeholder
survived. **Deploy the `.local.yaml`.**

Leave `STT_ENGINE=none`/`TTS_ENGINE=none`/`OCR_ENGINE=paddleocr` for the first
deploy (bring voice up in step 7). Set `OCR_ENGINE=none` if paddle didn't build.

---

## 5. Deploy

### Path A — Akash Console (matches the web UI; easiest)

1. Go to **console.akash.network → Deployments → Build Your Own → Upload SDL**.
2. Upload `deploy/akash-deploy.local.yaml` (the rendered one from step 4 — the
   template still has placeholders and will not boot).
3. **Create Deployment** → approve the deposit in your wallet **(signing step)**.
4. Wait for bids. **If none appear**, no provider currently has a free RTX 5090 —
   raise `placement.dcloud.pricing.app.amount`, or uncomment a fallback GPU
   (`rtx4090` / `a6000` / `l40s` / `h100`) under `app.resources.gpu.attributes`
   and re-submit.
5. Pick a bid → **Accept** **(signing step)** → the lease is created.
6. Open the lease → **Leases/URI** tab: note the public URI mapped to port 80
   (this is your `BASE_URL`).

### Path B — CLI (`provider-services`)

```
provider-services tx deployment create deploy/akash-deploy.local.yaml --from <key> --gas auto -y
provider-services query market bid list --owner <addr> --dseq <DSEQ>
provider-services tx market lease create --dseq <DSEQ> --provider <PROVIDER> --from <key> -y
provider-services lease-status --dseq <DSEQ> --provider <PROVIDER> --from <key>   # shows the URI
```

---

## 6. Verify the box (first thing, every deploy)

> **First boot is not instant.** The entrypoint downloads ~11 GB of GGUFs onto the
> `/models` volume before uvicorn starts, so `/health` will refuse connections for
> several minutes. Watch the lease logs for the `[entrypoint]` lines — 
> `downloading 'IBLOG_TUTOR' GGUF …` then `registering model …`. Only treat it as
> broken once those lines stop advancing. Later restarts skip this entirely.

Open a shell into the app container (Console → lease → **Shell**, or
`provider-services lease-shell ... app`), then:

```
/app/.gguf_venv/bin/python scripts/benchmark/bench_gpu.py
```

Expect: **RTX 5090, compute capability 12.0, CUDA ≥ 12.8, ~32 GB VRAM, matmul OK**.
If `cuda_available: false` or the live matmul errors with "no kernel image", the
base image's CUDA is too old for Blackwell — rebuild from a CUDA ≥ 12.8 base.

Then, from anywhere that can reach the URI:

```
curl https://<lease-uri>/health          # {"status":"ok",...}
```

and in the container shell: `ollama list` shows both tutor models.

---

## 7. Run the benchmarks

### LLM latency + language-switch (the headline)

```
BASE_URL=http://localhost:8000 OLLAMA_URL=http://localhost:11434 \
  bash scripts/benchmark/bench_all.sh
```

Success signals (vs the 4060 baselines in
`docs/architecture/cloud-scaling-plan.md §4`): Darija + French turns in **2–4 s**
(laptop: 17–144 s), and **alternating-language turns ≈ same cost as same-language**
ones — proof both models are resident (no eviction/reload).

### Voice pipeline (Siri-like chat)

1. Set `STT_ENGINE=whisper` (or `seamless`) and `TTS_ENGINE=piper` in the SDL env
   and redeploy — the Piper voices are already baked into `data/tts_voices`.
2. With a **16 kHz mono WAV of a real utterance** (`utterance.wav`):
   ```
   VOICE_WAV=utterance.wav bash scripts/benchmark/bench_all.sh
   ```
   Reports **end-of-speech → first audio** (target ≈1.2–2.2 s, voice doc §4).

### STT bake-off (pick the best voice model)

1. Put labeled audio in `tests/data/voice_eval/<id>.{wav,txt,lang}` (e.g. from
   `atlasia/DODa-audio-dataset`; include code-switched utterances).
2. ```
   bash scripts/benchmark/run_bakeoffs.sh
   ```
   → `scripts/eval_stt_results.json` (WER + RTF per engine) and TTS `.wav` files to
   listen to. Set `STT_ENGINE`/`TTS_ENGINE` to the winners.

### OCR (best-effort)

```
OCR_IMAGE=/path/to/page.png bash scripts/benchmark/bench_all.sh
```
Times a heavy PaddleOCR-VL page (laptop baseline 52.5 s). A clean "OCR unavailable"
here just means paddle didn't build on Blackwell — the accepted fallback.

Every run appends to `benchmark_report.md` and writes `benchmark_*.json`, including
**peak VRAM** (proof all modules co-resided on the 32 GB card).

---

## 8. Stop / start to save money

- **Pause paying:** close the lease (Console → **Close**, or
  `provider-services tx deployment close --dseq <DSEQ> ...`). This stops billing.
  Closing releases **both** persistent volumes, so a re-deploy starts with a fresh
  DB (re-run ingestion) **and** re-downloads the ~11 GB of GGUFs on first boot.
  Budget a few extra minutes on every re-create — this is the cost of not baking
  the weights into the image.
- **Cheaper still:** keep the lease but scale the app to 0? Akash has no "stop" —
  billing is per active lease, so *close* to stop paying and *re-create* to resume.
  Because the image is self-contained, re-create → ready in the time it takes the
  provider to pull the image.

---

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `bench_gpu` says CUDA unavailable / "no kernel image" | Base image CUDA < 12.8; rebuild from `nvidia/cuda:12.8+`. |
| Build warns paddle install failed | No sm_120 paddle wheel yet → set `OCR_ENGINE=none`. Retry later with paddle's develop channel. |
| STT worker errors on load | CTranslate2 has no Blackwell build → use `STT_ENGINE=seamless` (pure torch cu128) instead of `whisper`. |
| Chat returns HTTP 404 "no model named…" | Models didn't register; check container logs for the entrypoint's `ollama create`, and that `OLLAMA_MODEL`/`_FR` match the baked stems. |
| Entrypoint logs `download of 'IBLOG_TUTOR' failed` / HTTP 401 | `HF_TOKEN` missing, expired, or not scoped to `Oussamamaat/iblog-tutor-gguf` (the repo is private). Re-render the SDL via `make-local-sdl.sh`. |
| Entrypoint logs `is missing and $..._GGUF_URL is not set` | You deployed the **template** instead of `akash-deploy.local.yaml`. |
| Lease stays `pending`, provider never pulls | The GHCR package is still Private. Akash has no `imagePullSecrets` — make it Public (step 3). |
| Local build hangs mid-download at a fixed % | Seen repeatedly on the ollama tarball: the connection dies but `curl` waits forever (0 B/s at the NIC while the process looks alive). The Dockerfile now uses `--speed-limit/--speed-time` to turn that hang into a retryable error, plus `-C -` to resume. If another step hangs the same way, add the same flags. |
| No bids for the deployment | No free RTX 5090; raise the price ceiling or uncomment a fallback GPU model in the SDL. |
| Image push painfully slow | Build on a cloud VM near your registry; or split rarely-changing base layers into a separate pushed base image. |
| App reachable but you're nervous about exposure | It has **no auth** and CORS `*`. `UPLOADS_READ_ONLY=true` is already set; don't share the URI, and close the lease when done. |

---

## 10. File map

| File | Purpose |
|---|---|
| `scripts/docker/prepare_models.py` | Extract GGUFs+Modelfiles from local Ollama (run first, local) |
| `scripts/docker/upload_models_hf.py` | Upload those GGUFs to the private HF repo (step 1b) |
| `deploy/make-local-sdl.sh` | Render the secret-bearing `akash-deploy.local.yaml` from the template |
| `config/Dockerfile.gpu` | Blackwell CUDA image: 3 venvs + embeddings + voices (GGUFs fetched at boot) |
| `config/requirements-speech.txt` | `.speech_venv`: faster-whisper + SeamlessM4T |
| `config/requirements-ocr-paddle.txt` | `.ocr_venv`: PaddleOCR (best-effort) |
| `scripts/docker/entrypoint.sh` | Boots Ollama, registers models, inits DB, launches uvicorn |
| `deploy/akash-deploy.yaml` | Akash SDL: db (CPU) + app (1× rtx5090) |
| `deploy/akash.env` | Every env override, documented |
| `scripts/benchmark/bench_gpu.py` | GPU/CUDA/VRAM sanity |
| `scripts/benchmark/bench_llm.py` | Latency + language-switch + raw tok/s |
| `scripts/benchmark/bench_voice.py` | WS voice end-to-end latency |
| `scripts/benchmark/bench_ocr.py` | Heavy OCR page timing |
| `scripts/benchmark/run_bakeoffs.sh` | STT/TTS Phase 0 bake-off wrapper |
| `scripts/benchmark/bench_all.sh` | Runs everything + samples peak VRAM → report |
