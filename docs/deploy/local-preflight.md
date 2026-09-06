# Local preflight — RTX 4060 8GB, before leasing

Purpose: catch what the 2026-09-06 lease found out live (voice silent, latency
worse, web fallback never run) on the laptop instead, for free, before spending
lease money. Not a substitute for the lease's own checklist
(`lease-xtts-darija-checklist.md`) — this is what to run *first*.

**Discipline: never run two GPU stages at once.** Nothing on this 8GB card fits
alongside anything else — XTTS ~5.7GB, bge-m3 ~2.2GB, whisper `large-v3-turbo`
~1.6GB, a 9B tutor ~5.8GB. Run `nvidia-smi --query-gpu=memory.used,memory.total
--format=csv` between every stage and confirm it's back near zero before starting
the next one. Running the pytest suite alongside a resident XTTS server has
already produced a `torch...to()` timeout once (`lease-xtts-darija-checklist.md`).

## Stage 0 — full suite, no GPU

```
.gguf_venv/Scripts/python.exe -m pytest
```
**Confirmed this session: 750 passed** (742 + 8 new — `tests/test_tts_engine_selection.py`).
GPU idle throughout — everything here is mocked or pure logic.

## Stage 1 — XTTS resident-worker path, direct

This is the path that actually failed on the lease
(`XttsDarijaEngine` → `_ResidentTtsWorker` → `scripts/tts_worker_resident.py`),
and — before `scripts/voice_selftest.py --engine` existed — had **never run
outside a paid lease**. `.tts_eval_venv` and the 5.3GB checkpoint under
`data/tts_eval_cache/darija_xtts/` are already on this machine and are what
`app/config.py`'s defaults (`tts_xtts_venv_python`, `tts_xtts_model_dir`) point at.

```
.gguf_venv/Scripts/python.exe scripts/voice_selftest.py --engine xtts_darija
```

**Known Windows-only friction, already documented in ADR 0006 and
`scripts/eval_darija_xtts.py`'s own docstring:** `coqui-tts` needs torchcodec,
which needs FFmpeg's *shared* libraries — a plain `ffmpeg.exe` on PATH is not
enough, and Windows resolves DLL search paths in a way that excludes PATH
entirely. Without `FFMPEG_SHARED_BIN` set to a shared FFmpeg build's `bin/`
directory (e.g. a BtbN/FFmpeg-Builds `-shared` zip), this stage's `synthesize()`
call raises `TtsUnavailableError` citing `libtorchcodec` — **confirmed live during
this session**, real failure, not simulated. This is a Windows-local-only
problem: the Linux image installs FFmpeg via `apt`, where the normal loader path
finds it, and the CUDA-13 shim added to `config/Dockerfile.gpu` this session
handles the separate (Linux-only) `libcudart.so.13` failure mode. Two different
causes, same symptom class ("torchcodec can't load"), each fixed on its own
platform.

- **If you set `FFMPEG_SHARED_BIN`:** passes when the worker logs
  `loaded darija_xtt_2.0 on cuda`/`cpu`, `speaker latents ready`, and a `.wav`
  file is written that plays back real Darija speech.
- **If you don't:** this stage will raise `TtsUnavailableError` — expected on
  this OS without that env var. Proceed to stage 1b, which is the one that
  actually matters for the demo.

## Stage 1b — fallback-to-Piper on a real failure

Confirmed working live this session (no FFmpeg shim needed to see this):

```
.gguf_venv/Scripts/python.exe scripts/voice_selftest.py --engine auto
```
With `TTS_ENGINE=xtts_darija` and `TTS_FALLBACK_ENGINE=piper` (the default) set,
`get_tts_engine()` probes XTTS, the probe fails (real `libtorchcodec` error on
this machine), and it falls back to `PiperEngine` — passes when the script still
produces audio and prints `resolved TTS engine: PiperEngine`. This is the
mechanism that keeps a lease's voice session alive if XTTS breaks again instead
of the whole session going silent for the rest of the process's life.

## Stage 2 — STT

```
.gguf_venv/Scripts/python.exe scripts/eval_stt.py
```
Over `tests/data/voice_eval/`. Needs `.speech_venv` + `STT_ENGINE=whisper`;
~1.6GB VRAM for `large-v3-turbo` (already cached on this laptop).

**Confirmed this session (whisper): mechanically fine (0 failures / 30
examples, RTF ~0.3), but read the numbers through ADR 0009, not raw.** Raw
`mean_wer` here was 0.598 aggregate — that overstates it; ADR 0009's own
normalized re-scoring (already done, 2026-09-04 lease, real GPU) found
whisper's **pure-Darija** WER is 0.699 normalized vs. **seamless's 0.428** —
a real, confirmed, not-a-scoring-artifact gap. ADR 0009's Decision section
says *"seamless remains the pick for Darija-heavy traffic"*.

**Decision (2026-09-06): switched to `seamless`.** `deploy/akash-deploy.yaml`
now sets `STT_ENGINE=seamless`, matching ADR 0009's already-made call
instead of the drifted `whisper` default it shipped with. Important
distinction: `seamless` is proven on a **real GPU lease** already — ADR
0009's WER table came from an actual successful 2026-09-04 run, fixed up
across three commits (`6423b2f`/`c0b2875`/`1ad8376`) — but it was **not**
re-verified in this session's local stages 2/3 above (those ran against
`whisper`; this laptop's `.speech_venv` has no `torch`/`transformers`
installed, and pulling `facebook/seamless-m4t-v2-large` locally just to
re-confirm wasn't worth the time under this timeline). If you want it
locally verified too before leasing: `pip install torch --index-url
.../cu128 transformers sentencepiece soundfile` into `.speech_venv`, then
re-run stages 2/3 with `STT_ENGINE=seamless`.

## Stage 3 — voice pipeline, no LLM

```
$env:VOICE_ECHO_MODE = "true"
.gguf_venv/Scripts/python.exe -m uvicorn app.main:app --port 8123
```
then, in another shell:
```
.gguf_venv/Scripts/python.exe scripts/benchmark/bench_voice.py --base-url http://127.0.0.1:8123 --wav tests/data/utterance.wav
```
(note the path: `tests/data/utterance.wav`, not `tests/data/voice_eval/` —
that subdirectory holds the STT eval set for stage 2, not this file.)

Whisper + TTS resident (~2.9GB: piper is ~0 VRAM, whisper ~1.6GB, bge-m3
~2.2GB all preload at startup regardless of echo mode), no Ollama needed.
Passes when audio bytes come back and no `tts_*`/`stt_*` error event fires.
**This is the stage that would have caught "doesn't speak" before the
lease** — nothing before this session ever drove `voice.py`'s real
WebSocket path against a non-mocked TTS engine.

**Confirmed this session — real STT, real TTS, not mocked.** Backgrounding
a live `uvicorn` process in this particular shell environment didn't work
cleanly (output never flushed, port never bound, across three different
launch methods — a sandbox/tooling quirk, not a code issue); verified
instead via Starlette's in-process `TestClient` driving the exact same ASGI
app and WebSocket path with no OS socket involved. Result: real whisper
correctly transcribed a real Darija utterance
(`شنوية معيدة الحماية الشخصية اللي خاصني اللبسها`), echo mode formatted the
reply, real Piper produced 323KB of actual audio, full event sequence
(`transcript.partial` → `transcript.final` → `answer.delta` → `audio.start`
→ audio bytes → `audio.end`) matched what a production client expects. One
run hit a transient Piper `numpy` allocation failure (this laptop started
at ~4GB free RAM with bge-m3's CUDA context also loaded in the same
process) — it degraded correctly: a proper `tts_failed` error event fired,
`audio.end` still closed the turn, nothing hung or went silent. A retry
passed clean. If you hit this on the actual `uvicorn` command above, it's
this laptop's RAM pressure, not the lease (48Gi there).

## Stage 4 — web-search fallback, live

```
$env:WEB_SEARCH_ENGINE = "tavily"
$env:TAVILY_API_KEY = "<your key>"
.gguf_venv/Scripts/python.exe scripts/web_search_selftest.py
```
XTTS/whisper off for this stage (frees VRAM for `gemma2:9b`, ~5.8GB). Needs
`ollama pull gemma2:9b` locally first (already pulled on this laptop per ADR
0010's live test) and a real Tavily key. Passes when both languages return a
real answer with `external_sources`, and neither shows the `{title}`/`{titre}`
placeholder-echo defect ADR 0010 caught by hand.

Separately, confirm a grounded (in-corpus) question does **not** go to the web:
seed `raw/shared` into the running `iblog-pgvector` container
(`from app.services.ingestion import ingest_directory; ingest_directory('raw/shared', tenant_id='company_abc')`)
and check a real grounded chat turn still returns `answered_from_web=False`
with real tenant `sources`.

## Greenlight

Lease when: stage 0 is green, stage 1b is green (fallback proven — stage 1
itself is nice-to-have, gated on Windows FFmpeg setup that doesn't affect the
Linux lease), stage 3 is green, stage 4 is green, and CI has published a
`gpu-<sha>` image (`docker manifest inspect ghcr.io/oussamamaat/iblog-tutor:gpu-<sha>`
succeeds). Point the SDL's `image:` at that sha tag before deploying.
