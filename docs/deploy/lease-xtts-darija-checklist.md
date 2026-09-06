# Lease checklist — full voice pipeline with the medmac01 Darija voice

Everything needed to lease and test the **complete** pipeline (mic → VAD → whisper STT →
routing/RAG → tutor LLM → XTTS Darija TTS → playback) on Akash, with
`medmac01/darija_xtt_2.0` as the voice.

**Updated 2026-09-06, same day, second pass:** two more things shipped since this checklist was
first written and are folded into Phase 1/3 below —
1. A real live-browser defect in the sample-rate guard this session's earlier voice-pipeline fix
   added: it *rejected* the session outright when the browser declined the requested 16kHz
   `AudioContext`, instead of converting. Reproduced within the hour on a real browser (zero
   WebSocket connections ever reached uvicorn — "Connecting…" then silently gone). Fixed by
   resampling in `frontend/src/audio/pcm-worklet.js` instead of refusing; see ADR 0005's amendment
   for the full account, including the second bug the same incident exposed (`stop()` was
   overwriting error states back to "idle", hiding every startup failure).
2. `docs/architecture/rectified/adr/0010-web-search-fallback.md` — an opt-in web-search fallback
   for chat.py's refusal gate (`web_search_engine=none` by default, **not enabled for this
   lease** — no live Tavily key exercised yet). Nothing to test for it on this lease unless you
   specifically want to; it's mentioned here only because it changed `app/config.py`,
   `app/routers/chat.py`, and the pytest count below.

**Licensing, stated once and not buried:** `medmac01/darija_xtt_2.0` declares no license of its
own and therefore inherits Coqui XTTS-v2's **CPML — non-commercial**. This is an *evaluation*
deployment. Anything customer-facing must switch `TTS_ENGINE` back to `piper`. See
`docs/architecture/rectified/adr/0006-darija-tts-survey.md`.

---

## Why the image must be rebuilt (answering "can we push code to a running lease?")

The lease pulls `ghcr.io/oussamamaat/iblog-tutor:gpu` at container start. Code changes reach it
three ways, in descending order of trustworthiness:

| How | What happens | Use it for |
|---|---|---|
| **Rebuild + push the image, then lease** | Container starts on the new code | The real path. Do this. |
| **Update a running deployment** (Console → Update Deployment) | Restarts containers, re-pulls the image; `/models` volume state is *not* guaranteed to survive | Changing env vars / SDL only |
| **`git clone` inside the pod** | Patch lives until the container restarts | A quick one-off, like last session's seed-script fix |

This change touches `config/Dockerfile.gpu` (a whole new `.tts_venv`) and
`scripts/docker/entrypoint.sh`, so **option 1 is required** — neither can be applied to a running
container.

---

## Phase 1 — before spending anything (local, free)

- [ ] `git status` clean-ish; commit the voice + web-search-fallback work.
- [ ] `.gguf_venv/Scripts/python.exe -m pytest` → **750 passed** (742 + 8 new for the TTS
      fallback-engine resolution added this session — `tests/test_tts_engine_selection.py`).
      Run it with the GPU idle:
      the suite loads bge-m3, and on the 8GB laptop it will stall or OOM if a local XTTS server
      is holding 5.7GB at the same time (observed 2026-09-06 — a `torch...to()` timeout, not a
      code failure).
- [ ] `bash deploy/make-local-sdl.sh` exits clean, and `deploy/akash-deploy.local.yaml` has a real
      `HF_TOKEN`. **Rotate the old token first** — it was pasted in plaintext in a previous
      session's `ps aux` output.
- [ ] Push to `main` so CI builds and pushes `ghcr.io/oussamamaat/iblog-tutor:gpu`.
- [ ] `docker manifest inspect ghcr.io/oussamamaat/iblog-tutor:gpu` succeeds — the image is
      actually there. **Do not lease before this passes**; a lease that pulls a missing image
      burns escrow doing nothing.
- [ ] Confirm the SDL says `TTS_ENGINE=xtts_darija` (already set in `deploy/akash-deploy.yaml`).

## Phase 2 — deploy

- [ ] Create the deployment. Expect a longer first boot than last time: the image gained a
      `.tts_venv` (~3-4GB) and the entrypoint now downloads a **5.6GB XTTS checkpoint** onto
      `/models` before uvicorn starts.
- [ ] Watch the logs for, in order:
      - `downloading XTTS asset 'model_2.1.pth' ...`
      - `[tts_worker_resident] loaded darija_xtt_2.0 on cuda`
      - `[tts_worker_resident] speaker latents ready`
      - `starting uvicorn on 0.0.0.0:8000`
- [ ] `curl -sf http://<ingress>/health` → check `tts.active == "xtts_darija"` and
      `tts.fallback_used == false`. **This is the assertion the 2026-09-06 lease didn't have** --
      `/health` used to say only `{"status":"ok"}` regardless of whether the voice engine actually
      loaded, so a broken XTTS looked identical to a healthy one from this one curl. If
      `tts.fallback_used == true`, the session is speaking Piper, not the Darija voice this lease
      exists to evaluate -- check `tts.error` for why before going further, don't assume it's fine
      because the session isn't silent.
- [ ] Check the Console **Events** tab (not Logs) once, now, before any load — a baseline read for
      comparing against later if a restart happens (2026-09-06 incident #5, never root-caused:
      `OOMKilled` vs `Evicted` vs something else). The memory profile was raised to 48Gi this
      session for exactly this risk; confirm the Events tab agrees nothing already happened.
- [ ] **Seed immediately, before anything else** — `bash docs/deploy/lease-00-seed.sh`. Skipping
      this under time pressure is exactly what produced incident #4 (every chat/voice turn refusing
      with `domain_source: "no_match"` because the corpus was never ingested) on the last lease.
      Do this before opening a browser, before the voice-only test below, before anything.
- [ ] **Repoint the frontend at the new lease.** `frontend/.env` currently points at
      `http://127.0.0.1:8123` (this session's local voice testing) — `frontend/.env.akash-lease.bak`
      holds the *previous* (2026-09-05) lease's ingress URL, which is now dead; a new lease gets a
      new hostname every time. Set `VITE_API_BASE=http://<this lease's ingress>` in `frontend/.env`
      (or a new `.env.akash-lease.bak`) before testing through the browser UI, not just via curl.

If the checkpoint download stalls (it did on a laptop connection), `aria2c -x16 -s16 -k1M` inside
the pod is the known-good workaround, writing to `/models/darija_xtts/model.pth`.

**Rollback, no image rebuild needed:** if XTTS misbehaves live (won't load, sounds worse than
expected, runs out of time to debug), set `TTS_ENGINE=piper` in the Console SDL editor and hit
Update Deployment. Piper and its voices already ship in the image; nothing else in this change is
XTTS-specific. As of this session, an XTTS load failure now does this automatically at boot anyway
(`tts_fallback_engine=piper`) — this manual override is for "XTTS loads fine but you want Piper for
other reasons," not for a load failure, which no longer needs manual intervention.

## Phase 3 — verify the pipeline

- [ ] Corpus + gates already confirmed above (seed ran right after `/health`, per Phase 2).
- [ ] **Voice-only, no LLM** — the fastest way to prove mic→speaker works before involving the
      tutor. Set `VOICE_ECHO_MODE=true` and open the frontend:
      it transcribes you and speaks back *"سمعتك كتقول: …"* / *"Je vous ai entendu dire : …"*
      in whichever language you spoke.
- [ ] **Language switching** — the regression that motivated this work. In ONE session: speak
      French, then speak Darija. Turn 2 must come back in Darija. Before the fix this was
      structurally impossible (whisper was pinned to turn 1's language).
- [ ] **Full pipeline** — turn echo mode off, ask a real grounded question by voice, confirm the
      answer is grounded, cited, and spoken.
- [ ] Listen: is `medmac01/darija_xtt_2.0` acceptable where `ar_JO-kareem-medium` was not?
      That judgement is the whole point of this deployment.

## Phase 4 — the rest of the Phase B agenda (unchanged)

Item 5 Ollama concurrency sweep (N=1/2/4/8), item 4 multi-document ingest timing, §3 output-quality
sampling. See `POST_LEASE_MVP_SPRINT_PLAN.md`.

---

## What to expect, measured locally 2026-09-06

| | Piper (`piper`) | XTTS (`xtts_darija`) |
|---|---|---|
| First call | <1s | **~51s** (5.6GB load + speaker latents) |
| Later calls | 0.16–0.38s | ~2s, RTF 0.39–0.69 |
| VRAM | 0 (CPU/ONNX) | ~5.7GB |
| Licence | MIT — shippable | CPML — **evaluation only** |

The ~51s cold start is why `app/main.py` warms the engine at startup: without it that delay lands
on the user's first spoken sentence and looks like a hang. Boot is correspondingly slower — that
is intentional, not a stall.

**VRAM note:** XTTS (5.7GB) + bge-m3 (2.2GB) ≈ 8GB, which is the whole laptop card — locally you
must not run the suite and an XTTS server at once. On the lease's 32GB card, XTTS sits alongside
both 9B tutors comfortably.

## Rollback

`TTS_ENGINE=piper` in the SDL and update the deployment. No image rebuild needed — Piper and its
voices are still in the image, and nothing else in this change is XTTS-specific.
