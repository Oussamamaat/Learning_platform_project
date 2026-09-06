# Incident Log — 2026-09-06 Akash Lease: XTTS Darija Full-Pipeline Test

Concerns the `overclock` (na-us-southeast, rtx5090, $0.77/hr) lease used to test the
full voice pipeline with `medmac01/darija_xtt_2.0` per
[`lease-xtts-darija-checklist.md`](lease-xtts-darija-checklist.md). Lease URL:
`http://3ubs9r6usp9fdeqdii2rb3ao7s.ingress.5090.mel.val.akash.pub`. Preserved so the
next lease doesn't re-diagnose the same failures from scratch, and so the permanent
fixes it points to (still open at time of writing) have a paper trail back to their
live evidence.

---

## 1. Update Deployment appeared to do nothing — code from this session was missing

**Symptom:** After deploying, `grep -c "darija_xtts\|XTTS_BASE" /app/scripts/docker/entrypoint.sh`
and `grep -c "XttsDarijaEngine" /app/app/services/tts.py` both returned `0` in the running
pod, despite the image having been rebuilt and pushed after all of this session's commits.

**Root cause:** Akash renders the SDL as a Kubernetes pod. k8s defaults
`imagePullPolicy` to `IfNotPresent` for any tag that isn't `:latest`. The provider
node had already pulled `ghcr.io/oussamamaat/iblog-tutor:gpu` for the prior
(2026-09-05) lease and reused that cached layer for the new lease under the same
tag name. **"Update Deployment" does not fix this** — it resolves the same tag and
hits the same cache.

**False leads eliminated before landing on the real cause:**
- First suspected the registry image itself was stale — disproven by pulling the
  small `COPY` layers directly from GHCR and `md5sum`-comparing them against the
  local working tree; they matched exactly (`tts.py` → `3cd24dac…`,
  `entrypoint.sh` → `fb006660…`), and the image config's `created` timestamp
  (`2026-09-06T15:54:57Z`) was after all relevant commits. An earlier `grep -c`
  against the downloaded layer had itself returned `0` and briefly reinforced the
  wrong theory — the extraction was silently checking the wrong path
  (`app/services/tts.py` instead of the tar's real `app/app/services/tts.py`
  prefix), not evidence of anything.
- Suspected CI hadn't actually rebuilt — disproven via the Actions API: build for
  `9d314ee` completed 2026-09-06T16:02:29Z, 14 minutes before the lease booted.

**Fix:** No rebuild needed — the correct image was already published. Added
`.github/workflows/retag-gpu-image.yml` (manual `workflow_dispatch`), which runs
`docker buildx imagetools create` to copy the existing `:gpu` manifest to a new,
immutable tag (`:gpu-9d314ee`) the node has never seen, forcing a real pull. This
workflow's filename is deliberately absent from `build-gpu-image.yml`'s `paths:`
filter so adding it doesn't trigger a ~22 min rebuild. Took ~10 seconds to run.
Updated `deploy/akash-deploy.local.yaml`'s `image:` to the new tag and hit
**Update Deployment** in Console — same lease, same escrow, no new deployment cost.
`/models` survived the restart, so both GGUFs were re-registered from cache in
~3 minutes instead of re-downloading 11.6GB.

**Verified:** post-update, both `grep -c` checks returned the expected non-zero
counts (`3` and `2`), matching the registry image exactly.

**Carried forward, not yet fixed:** the SDL should move to an immutable
(git-sha-derived) tag as a matter of course, not just as an incident response —
see "Open items" below.

---

## 2. SDL paste had a duplicated `image:` key

**Symptom:** Akash Console's SDL editor flagged line 74 with "Nested mappings are
not allowed in compact mappings" / "Incorrect type. Expected 'string'".

**Cause:** `image: image: ghcr.io/oussamamaat/iblog-tutor:gpu-9d314ee` — the key
got typed twice while editing the Console's text area directly (this is the
Console's own SDL editor, not a local file — a point of confusion mid-session,
since the fix had already been applied correctly to the local
`deploy/akash-deploy.local.yaml` and needed to be transferred over, not re-typed).

**Fix:** Validated the local file parsed clean with no unfilled placeholders
(`yaml.safe_load` + placeholder grep), then had the user copy its full contents to
the clipboard directly (`Get-Content -Raw | Set-Clipboard`) and paste-replace the
entire Console editor contents, rather than hand-editing one line in two places.

---

## 3. XTTS worker: `OSError: Could not load this library: libtorchcodec_image.so`

**Symptom:** `[tts_worker_resident] warmup` traceback in the boot logs;
`TtsUnavailableError: resident TTS worker error: OSError: ...`. Voice sessions
connected successfully (ingress handshake fine, `voice.py` accepted the socket) but
closed immediately after — `get_tts_engine()` raising inside the WebSocket handler
sends `{"type":"error","code":"voice_unavailable"}` and closes, which is
indistinguishable in the browser from "Connecting… then quits."

**False lead eliminated:** the `config/requirements-tts.txt` comment blames FFmpeg
("Needs FFmpeg's SHARED libraries present"). `ffmpeg -version` in the pod showed
6.1.1 fully installed with all `libav*`/`libsw*` present — not the actual cause.

**Root cause, found via `ldd` on the actual `.so`:**
```
libcudart.so.13 => not found
libnvrtc.so.13  => not found
```
`torch==2.11.0+cu128` in `.tts_venv` is a **CUDA 12.8** build. `config/requirements-tts.txt`
pins `torchcodec>=0.16.0`, unversioned by CUDA — PyPI resolved this to a wheel built
against **CUDA 13**, but `download.pytorch.org/whl/cu128`'s torchcodec index only
goes up to `0.9.1`, so pip never had a matching cu128 build available and silently
fell back to the mismatched PyPI one.

**Fix applied live, in-pod (ephemeral — see risk below):**
```bash
pkill -f tts_worker_resident.py   # the ORIGINAL worker had cached the load failure
                                   # in a still-alive process; synthesize() only
                                   # respawns a worker that's actually dead
/app/.tts_venv/bin/pip install "nvidia-cuda-runtime==13.3.29" "nvidia-cuda-nvrtc==13.3.33"
ln -sf <found libcudart.so.13 path> /usr/lib/x86_64-linux-gnu/
ln -sf <found libnvrtc.so.13 path>  /usr/lib/x86_64-linux-gnu/
ldconfig
```
Verified with `import torchcodec; from torchcodec.decoders import AudioDecoder` →
`torchcodec OK 0.16.0+cu130` (loads fine despite the version-string mismatch — CUDA's
runtime API is forward-compatible here).

**False lead in package naming:** `nvidia-cuda-runtime-cu13` / `nvidia-cuda-nvrtc-cu13`
are deprecated stub packages that fail to build with a redirect notice; the real
packages are `nvidia-cuda-runtime` / `nvidia-cuda-nvrtc`, versioned `13.x.y` directly.

**Risk flagged and then realized:** this fix lives entirely in the container's
writable layer (`.tts_venv`'s site-packages, `/usr/lib/x86_64-linux-gnu` symlinks),
not on the persistent `/models` volume. **Any container restart wipes it.** This is
exactly what happened in incident #5 below.

**Not yet done:** pin `torchcodec` from the `cu128` PyTorch index (once/if it
publishes a matching build) or add the two `nvidia-cuda-*` packages plus the symlink
step to `config/Dockerfile.gpu` / `config/requirements-tts.txt` permanently. See
"Open items."

---

## 4. Chat/voice answers refusing everything — corpus was never seeded

**Symptom:** every chat turn came back `domain_source: "no_match"`, `sources: []`,
a deterministic refusal — including on-topic industrial-safety questions that
should have matched the tenant corpus easily.

**Root cause:** not a bug — Akash storage is ephemeral per the SDL's own header
comment, and `raw/` ships baked into the image but is never auto-ingested.
`docs/deploy/lease-00-seed.sh` (Phase 3, step 1 of the checklist) hadn't been run
yet on this lease.

**Fix:** ran the seed script's ingestion step directly in the pod (the full script
also clones the repo for fixtures not baked into the image, not needed yet at this
point):
```python
from app.services.ingestion import ingest_directory
ingest_directory('raw/shared', tenant_id='company_abc')
```
→ `25 files ingested, 37 total chunks`. Re-tested chat turn immediately after:
`domain_source: "retrieval"`, real cited source (`2.6_ar_loi_27_06.md`), coherent
grounded Darija answer. Confirmed the RAG pipeline itself was never actually
broken — this was 100% a "forgot to seed" gap, not a defect.

**Lesson for next lease:** run `lease-00-seed.sh` (or at minimum its ingest step)
**immediately** after `/health` goes green, before any other testing — exactly
what the checklist already says, skipped here under time pressure.

---

## 5. Container restarted — the incident #3 fix was wiped, cause under investigation

**Symptom:** mid-voice-testing, `ollama list` in the pod suddenly showed only one
of the two tutor models instead of both. This is the signature of a fresh
`entrypoint.sh` run (the GGUF registration loop hadn't finished re-running yet),
meaning the container had restarted.

**Immediate consequence:** the incident #3 XTTS fix (symlinks + `.tts_venv`
packages, all in the writable layer) is gone. Any voice turn will hit the original
`libcudart.so.13`/`libnvrtc.so.13` "not found" error again until it's reapplied.

**Status at time of writing: OPEN, not yet root-caused.** Two live hypotheses,
neither confirmed:
1. **OOM kill.** `OLLAMA_KEEP_ALIVE=24h` keeps both 9B tutors resident permanently;
   layered with bge-m3, whisper `large-v3`, and the 5.6GB XTTS checkpoint all
   resident at once on a 32Gi host-memory limit, total footprint may be tight
   enough to trip a Kubernetes OOM kill under the added load of a live voice
   session (STT + generation + TTS all active simultaneously).
2. Some other provider-side event (eviction, node issue) unrelated to memory.

**Diagnostic commands issued, not yet returned as of this entry:**
```bash
free -h
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
ollama list
curl -s http://localhost:8000/health
```
Plus a request to check the Console **Events** tab (not Logs) for the literal
restart reason — `OOMKilled` vs `Evicted` vs something else — since that
distinction determines whether this is a recurring resource-limit problem (real
risk for a live demo) or a one-off.

**To be filled in once resolved:** actual cause, whether it recurred, and whether
`profiles.compute.app.resources.memory.size` in `deploy/akash-deploy.yaml` (currently
`32Gi`) needs raising, or whether something should be made non-resident
(`OLLAMA_KEEP_ALIVE` lowered, XTTS idle-released) to fit the current limit safely.

---

## Open items carried out of this lease

- [x] **Immutable image tags as standard practice**, not just incident response —
  `build-gpu-image.yml` now pushes `ghcr.io/oussamamaat/iblog-tutor:gpu-${{ github.sha }}`
  alongside `:gpu` on every build. `deploy/akash-deploy.yaml`'s `image:` line carries a
  comment naming this; point it at the sha tag before every lease going forward.
- [x] **Pin `torchcodec` correctly** — `config/Dockerfile.gpu`'s `.tts_venv` step now
  installs `nvidia-cuda-runtime==13.3.29`/`nvidia-cuda-nvrtc==13.3.33`, points the
  loader at them via `/etc/ld.so.conf.d/tts-cuda13.conf` + `ldconfig`, and asserts
  `from torchcodec.decoders import AudioDecoder` right after — a build that can't load
  it now fails in CI, not at a lease's demo time. `config/requirements-tts.txt` was
  also added to `build-gpu-image.yml`'s `paths:` filter (confirmed missing, as
  suspected below).
- [ ] **Resolve incident #5** — still open. The memory profile was raised
  `32Gi` → `48Gi` as a precaution (both 9B tutors pinned at
  `OLLAMA_KEEP_ALIVE=24h` + whisper `large-v3` + resident PaddleOCR + bge-m3 +
  a `torch.load()` of the 5.6GB XTTS checkpoint, which materializes in HOST ram
  before `.to(cuda)`, is tight at 32Gi), but the actual `free -h`/`nvidia-smi`/Events
  output was never captured, so OOM is still unconfirmed. Check the Events tab again
  this lease regardless of whether a restart happens.
- [x] Updated `lease-xtts-darija-checklist.md` Phase 2 with an explicit "seed
  immediately, before anything else" callout right after the new `/health`
  `tts.active`/`tts.fallback_used` assertion.
- [x] **A second, previously-undocumented cause of "doesn't speak" found during this
  review**: `TTS_WORKER_IDLE_RELEASE_SECONDS` was never set in `deploy/akash-deploy.yaml`
  or `scripts/docker/entrypoint.sh`, unlike its OCR/speech siblings (both `=0` there) —
  so the app's own 300s default silently released the XTTS worker after 5 idle
  minutes (easy during a demo/meeting), and the next spoken sentence paid the full
  ~51s reload. Both files now set it to `0`, matching the other resident workers.
- [x] **TTS now degrades instead of going silent.** `app/services/tts.py`'s
  `get_tts_engine()` probes the configured engine once at startup and falls back to
  `settings.tts_fallback_engine` ("piper" by default) if it fails to load — this is
  new this session, not something the 2026-09-06 lease had. `/health` now reports
  `tts.active`/`tts.fallback_used`/`tts.error` instead of only `{"status":"ok"}`.
  Verified live on this laptop against a *real* XTTS failure (missing
  `FFMPEG_SHARED_BIN` on Windows — a different root cause than the Linux
  `libcudart.so.13` one above, same symptom class): resolution fell back to
  `PiperEngine` and produced real audio, not a mocked result.
- [ ] **ADR 0010's web-search fallback provisioning gap** (not one of this incident
  log's original items, found while addressing the others): `gemma2:9b` — the model
  `generate_web_fallback_response` calls — was never added to `entrypoint.sh`'s
  provisioning, so enabling `WEB_SEARCH_ENGINE=tavily` would have called a model
  that was never pulled. `entrypoint.sh` now runs `ollama pull gemma2:9b` when that
  env var is `tavily`, gated so a lease with the fallback off pays nothing extra.
  `scripts/web_search_selftest.py` (new) is meant to close ADR 0010's other open
  item — no live Tavily key ever exercised — locally before this lease; whether
  that actually ran is on whoever runs `docs/deploy/local-preflight.md` before
  deploying.
