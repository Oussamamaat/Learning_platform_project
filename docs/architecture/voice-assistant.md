# Voice Assistant — Real-Time Open-Mic Pipeline

**Status: code scaffolding built (2026-08-25), NOT vendor-validated, NOT
deployed.** This is a `rectified/`-register document: long-form, describes
a target with real code behind it, but that code has not been exercised
against a live STT/TTS vendor, a live microphone, or a rented GPU. Treat
every latency/quality number below as either a measurement from the
*text* pipeline (labeled as such) or an estimate, never as a proven voice
number.

Closes `resurrection.md` Q0.2 — "Audio: what actually ships?" — flagged
open since the project began, "the single largest unresolved MVP item."

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
| STT engine seam | `app/services/stt.py` | Interface + registry built (mirrors `app/services/ocr.py`'s pattern exactly). `whisper`/`seamless` engine bodies are **unverified scaffolding** — never run against a loaded model |
| TTS engine seam | `app/services/tts.py` | Interface + registry built. `PiperEngine` is **unverified scaffolding** |
| Resident STT worker | `scripts/speech_worker_resident.py` | Unverified scaffolding, mirrors `scripts/ocr_worker_resident.py`'s IPC protocol |
| WebSocket voice session | `app/routers/voice.py` (`WS /api/v1/voice/session`) | Built, integration-tested against mocked STT/TTS/turn dependencies (`tests/test_voice_session.py`, 3/3 pass — happy path, refusal path, STT-unavailable error path) |
| Frontend mic capture + playback | `frontend/src/hooks/useVoiceSession.ts`, `frontend/src/audio/pcm-worklet.js`, mic button in `InputArea.tsx` | Built, type-checks (`tsc -b`) and lints clean. **Never exercised against a real microphone or real server audio** |
| Config settings | `app/config.py`: `stt_engine`, `tts_engine`, `stt_model`, `tts_voice_fr`, `tts_voice_ar`, `stt_venv_python`, `tts_voice_dir`, `speech_worker_idle_release_seconds` | Built, all default to `"none"` |

Both engine defaults are `"none"` — `NullSttEngine`/`NullTtsEngine` raise
`SttUnavailableError`/`TtsUnavailableError` on any call, same fail-loudly
contract as `app/services/ocr.py`'s `NullOcrEngine`. A voice session
connects successfully today; sending audio produces a clean `error` event
(`code: "stt_unavailable"`), not a crash — this is intentional and tested.

## What's explicitly deferred to a rented GPU

- **Phase 0 vendor bake-off** (`scripts/eval_stt.py`, `scripts/eval_tts.py`
  — written, never run). Needs real labeled Darija/French/code-switched
  audio (none committed to this repo yet) and a GPU this laptop's 8GB card
  cannot spare alongside the resident tutor model.
- Installing `faster-whisper` / `transformers`+`torchaudio` (SeamlessM4T)
  / `piper-tts` and downloading any model weights.
- The full `tests/test_chat.py` / `tests/test_ingestion.py`-style
  regression suite for voice against a *live* stack (Postgres, Ollama,
  real STT/TTS) — everything above was verified with mocks instead
  because this laptop's Postgres is not reachable (`docs/architecture/
  cloud-scaling-plan.md`) and no GPU is available for the vendor calls.
- A live end-to-end barge-in check (mid-sentence interrupt → playback
  flush → correct history truncation). The code path is small and
  hand-reviewed; a real-timing test needs real audio hardware.

## Latency budget (estimate, not measured)

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

Text chat tolerates the laptop (a 20s answer is annoying but usable).
Voice does not — two independent blockers: the 8GB card cannot hold an
STT model alongside the resident tutor, and the tutor alone is ~15x over
a conversational latency budget. **A rented GPU (L4/A10G 24GB class) is
required before this can be demoed with real audio**, not just recommended.

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
- **Language is pinned per session** after the first turn
  (`app/routers/voice.py`'s `pinned_language`) to avoid a mid-session
  French↔Darija VRAM swap. An explicit in-message override still works
  (`resolve_language`'s precedence is unchanged).

## Next steps, in order

1. Rent a GPU (L4/A10G class).
2. Collect labeled Darija/French/code-switched audio under
   `tests/data/voice_eval/` (see `scripts/eval_stt.py`'s docstring for the
   naming convention) — consider `atlasia/DODa-audio-dataset`.
3. Install bake-off dependencies (commented in `config/requirements.txt`),
   run `scripts/eval_stt.py` and `scripts/eval_tts.py`, listen to the TTS
   output, pick winners.
4. Set `stt_engine`/`tts_engine` away from `"none"`, download Piper voice
   models into `settings.tts_voice_dir`.
5. Set up `.speech_venv` (mirrors `.ocr_venv`'s reasoning — conflicting
   torch/CUDA pins), validate `scripts/speech_worker_resident.py`
   end-to-end.
6. First live microphone test against a real deployed session.
7. Live barge-in timing check.
