# Voice Assistant — Real-Time Open-Mic Pipeline

**Status as of 2026-09-07: live, vendor-validated, deployed, and verified against a real
microphone on a rented GPU.** Every item this document's original "Next steps" list named
(below §Next steps) is done. The sections below the amendment are kept close to their
2026-08-25 original wording as a design-rationale record — the cascaded-pipeline argument,
the latency-budget structure, and the "Known MVP simplifications" still describe the real
system — but treat any sentence still claiming something is "unverified," "an estimate," or
"not deployed" as superseded by this block, not as current truth.

**What actually shipped, and where the record of it lives:**
- **STT vendor bake-off run and decided**: SeamlessM4T-v2 for Darija-heavy traffic,
  faster-whisper competitive for French-only — real labeled audio, re-scored for Darija's
  lack of standard orthography. See `docs/architecture/rectified/adr/0009-stt-eval-rescoring.md`
  and `benchmark_results/README.md`.
- **TTS vendor survey run and decided**: Piper's closest Darija-ish voice rejected on live
  listening (wrong accent); an XTTS-v2 fine-tune sounds right but is licensed non-commercial
  and is demo/evaluation-only, not a shipping default; a clean-licensed production fine-tune
  is scoped, not yet executed. See `docs/architecture/rectified/adr/0006-darija-tts-survey.md`.
- **`.speech_venv` stood up and validated end-to-end** against `scripts/speech_worker_resident.py`
  on the leased GPU — this is no longer scaffolding.
- **Live microphone + live barge-in exercised** against a real deployed session, real speaker
  audio, both languages.
- **Real, measured latency** replaces every estimate in the table below: cloud chat turns
  measured at 2.0–3.4s regardless of language (see `cloud-scaling-plan.md`'s confirmation
  note); voice adds STT/TTS on top of that, not a separate re-measurement documented here yet.
- **A real production defect was found and fixed post-deployment**: SeamlessM4T was silently
  translating French speech into Darija rather than transcribing it (a `tgt_lang`/no-hint
  interaction, not a routing bug) — see `docs/LESSONS_LEARNED.md` #14.
- **Language pinning was tried, found to cause a worse bug, and removed** — see the
  "Known MVP simplifications" section below, which still describes it as shipped; it is not.
  The live pipeline re-detects language on every utterance instead (fixed alongside the
  seamless defect above, same lesson entry).

Closes `resurrection.md` Q0.2 — "Audio: what actually ships?" — flagged
open since the project began, "the single largest unresolved MVP item." Now closed.

## Why cascaded, not end-to-end

Cascaded: VAD → STT → text turn (RAG + LLM) → TTS. Rejected end-to-end
multimodal (Qwen-Omni, Moshi, GLM-4-Voice) because none support Moroccan
Darija, and because it would discard every grounding guarantee this
platform exists to provide — citation injection (`extract_citations`/
`inject_citations`, `app/services/citations.py`), the deterministic
refusal gate (`deterministic_refusal`, `app/services/llm.py:374`), and the
whole pinned-context/segment-reset design (`_resolve_turn_context`,
`app/routers/chat.py:95`) all operate on text. For a tool answering from
safety regulations, an ungrounded fluent voice answer is the worst
possible failure mode.

## What's built

| Piece | File | Status |
|---|---|---|
| Streaming LLM generation | `app/services/llm.py`: `stream_llm_response`, `_stream_ollama_chat` | Built, unit-tested (`tests/test_llm_streaming.py`, mocked `urlopen`) |
| Turn resolution (shared with text chat) | `app/services/turn.py` | Built, unit-tested (`tests/test_turn.py`). Reuses `chat.py`'s `_resolve_turn_context` **by import**, not by extraction — see that module's docstring for why |
| VAD (endpointing) | `app/services/vad.py`: `EnergyEndpointer` | Built and tested (`tests/test_vad.py`, 7/7 pass). Dependency-free RMS-threshold — a real working default, coarser than Silero (tracked upgrade, needs `onnxruntime`) |
| STT engine seam | `app/services/stt.py` | **Live**: `seamless` is the deployed engine (ADR 0009); `whisper` bench-tested and kept as the better French-only option |
| TTS engine seam | `app/services/tts.py` | **Live**: `xtts_darija` deployed for demo/evaluation (ADR 0006, non-commercial license — not a shipping default); `PiperEngine` bench-tested and rejected on live listening |
| Resident STT worker | `scripts/speech_worker_resident.py` | **Live**, validated end-to-end on the leased GPU; one production defect found and fixed post-deployment (`docs/LESSONS_LEARNED.md` #14) |
| WebSocket voice session | `app/routers/voice.py` (`WS /api/v1/voice/session`) | Built, integration-tested against mocked STT/TTS/turn dependencies (`tests/test_voice_session.py`, 3/3 pass — happy path, refusal path, STT-unavailable error path) |
| Frontend mic capture + playback | `frontend/src/hooks/useVoiceSession.ts`, `frontend/src/audio/pcm-worklet.js`, mic button in `InputArea.tsx` | **Live**, exercised against a real microphone and real server audio on the leased GPU |
| Config settings | `app/config.py`: `stt_engine`, `tts_engine`, `stt_model`, `tts_voice_fr`, `tts_voice_ar`, `stt_venv_python`, `tts_voice_dir`, `speech_worker_idle_release_seconds` | Deployed with `stt_engine=seamless`, `tts_engine=xtts_darija` — no longer `"none"` |

The `"none"`/`NullSttEngine`/`NullTtsEngine` fail-loudly path described above still exists and
still matters for any environment that hasn't set real engines (mirrors `app/services/ocr.py`'s
`NullOcrEngine` contract) — it's just not what the deployed lease runs.

## What was deferred to a rented GPU — now done

Every item below, from the original 2026-08-25 version of this section, is complete:

- **Phase 0 vendor bake-off** — run on real labeled Darija/French/code-switched audio;
  see ADR 0009 (STT) and ADR 0006 (TTS).
- **`faster-whisper` / `transformers`+`torchaudio` (SeamlessM4T) installed**, model weights
  downloaded onto the lease's persistent volume.
- **The live-stack regression pass** — real Postgres, real Ollama, real STT/TTS — replaced the
  mocked version this section originally deferred.
- **Live end-to-end barge-in** exercised against real audio hardware on a real session.

## Latency budget — superseded by real measurement

The table below is kept as the *original estimate*, for comparison. It is no longer the current
number: see `cloud-scaling-plan.md`'s confirmation note for what was actually measured (chat
turns 2.0–3.4s regardless of language, both tutors resident, switch cost genuinely ~0 — matching
this table's prediction closely). No separate STT+TTS-inclusive voice latency number has been
measured and written up yet as of this date.

Cloud L4 24GB, warm, both tutors resident — see
`docs/architecture/cloud-scaling-plan.md` for the *measured text-chat*
numbers this extrapolates from (Darija turn 17–24s / French 44–144s on
this laptop, estimated 2–4s on a rented GPU):

| Stage | ms (p50, estimated) |
|---|---|
| VAD endpoint hangover | 300–500 |
| STT (5s utterance, warm) | 150–300 |
| Retrieval (bge-m3 + pgvector) | 80–150 |
| LLM TTFT (~2k prefill, 9B, warm) | 400–900 |
| First sentence + Piper synthesis | 250–400 |
| **End-of-speech → first audio** | **≈1.2–2.2s** |

The single biggest lever: sentence-chunked TTS streaming
(`app/routers/voice.py`'s `_answer_worker` — splits `stream_llm_response`'s
output on sentence boundaries and synthesizes+sends each one as soon as
it completes, so sentence 2 generates while sentence 1 is already
playing). This is implemented, not just planned.

## Deployment: laptop vs. cloud

Text chat tolerates the laptop (a 20s answer is annoying but usable). Voice never could —
two independent blockers: the 8GB card cannot hold an STT model alongside the resident tutor,
and the tutor alone is ~15x over a conversational latency budget. **This is why a rented GPU
was required before voice could be demoed with real audio — done**: voice is deployed and
verified on Akash-leased hardware (rtx8000/rtx5090 class, not L4/A10G as originally scoped,
picked on real Akash bid pricing — see `deploy/akash-deploy.local.yaml`'s header comment).

## Known MVP simplifications (deliberate, documented at the call site)

- **Transport is raw PCM16 over WebSocket, not Opus.** No codec
  dependency; revisit before a real network deployment (bandwidth, not
  correctness).
- **TTS sample rate is a fixed per-engine constant** (`TtsEngine.
  sample_rate`, assumed 22050Hz for both Piper voices), not read
  per-voice. Verify against actual downloaded voice configs once Phase 0
  picks final voices.
- **Barge-in cancellation is cooperative, not a hard abort.** Setting the
  worker thread's `cancel_flag` stops further output reaching the client
  but does not abort the in-flight Ollama HTTP request — it finishes
  generating server-side regardless. Correct user-visible effect (instant
  silence), not a correct server-side cancel.
- **Citations are never spoken.** `stream_llm_response` deliberately skips
  `inject_citations` (a post-hoc full-text rewrite with no incremental
  form) — the client gets `extract_citations(context)` as a separate
  `citations` message for the UI instead.
- **Diagram intent is not handled in voice.** `app/services/turn.py`
  skips `chat.py`'s diagram branch entirely — no audio rendering exists
  for a Mermaid/candlestick diagram, so a voice turn that would have
  triggered one in text chat is answered as ordinary grounded prose.
- **Language pinning was tried and removed, not shipped.** The original plan was to pin
  language per session after the first turn to avoid a mid-session VRAM swap — since both
  tutors are resident simultaneously on the leased GPU (see `cloud-scaling-plan.md`), that
  swap no longer happens, and pinning caused a real bug instead: it forced STT to decode in
  the first-heard language for the rest of the session, so a later utterance in the other
  language was mis-transcribed. Removed; the live pipeline re-detects language fresh on
  every utterance (`app/routers/voice.py`, see `docs/LESSONS_LEARNED.md` #14).

## Next steps — original list, now complete

1. ~~Rent a GPU (L4/A10G class).~~ Done — Akash rtx8000/rtx5090, not L4/A10G (real bid
   pricing made the substitution; see `deploy/akash-deploy.local.yaml`).
2. ~~Collect labeled Darija/French/code-switched audio.~~ Done — `tests/data/voice_eval/`.
3. ~~Install bake-off dependencies, run the bake-off, pick winners.~~ Done — ADR 0009 (STT),
   ADR 0006 (TTS).
4. ~~Set `stt_engine`/`tts_engine` away from `"none"`.~~ Done — `seamless` / `xtts_darija`.
5. ~~Set up `.speech_venv`, validate the resident worker end-to-end.~~ Done.
6. ~~First live microphone test against a real deployed session.~~ Done.
7. ~~Live barge-in timing check.~~ Done.

**Open next, not on the original list:** a production (commercially-licensed) Darija TTS
fine-tune (scoped in ADR 0006, not started), sharpening exactly when the web-search fallback
should trigger versus stay in-domain, and migrating LLM serving to vLLM for concurrent
production traffic (see `serving.md`).
