# Architecture — current state

This folder is the fast path for "what's actually true right now." For *why* each
decision was made, follow the "detail & rationale" pointer at the bottom of each file
into the root-level docs — those remain the record; nothing here duplicates them, it
distills them.

## How the pieces fit

A tenant's question arrives at `POST /api/v1/chat` ([chat.py](../../app/routers/chat.py)), or
by voice over `WS /api/v1/voice/session` ([voice-assistant.md](voice-assistant.md)). The backend
auto-resolves domain and language per turn (no UI selector for either), reuses a pinned
retrieval context across same-topic follow-ups via server-side conversation history, embeds
the query, retrieves the top-k matching chunks from that tenant's documents in pgvector, builds
a system prompt in the resolved language (Darija or French, each served by its own fine-tuned
model) around that context and any prior turns, and sends it to Ollama. The model is instructed
to ground its answer strictly in the retrieved context and to refuse rather than fabricate when
the context doesn't cover the question — and since 2026-09-06, a refusal can instead fall back
to a clearly-labeled live web search (opt-in, `ADR 0010`) rather than dead-ending. A tenant
grows its own corpus through `POST /api/v1/ingest/upload` (`app/routers/ingest.py`), parsed,
chunked, and embedded automatically.

- [data-and-retrieval.md](data-and-retrieval.md) — corpus, chunking, embeddings, pgvector
- [finetune-pipeline.md](finetune-pipeline.md) — base model, LoRA config, generation + training pipeline
- [serving.md](serving.md) — what's deployed today vs. the documented target architecture
- [video-generation-interface.md](video-generation-interface.md) — contract with the
  explanatory-video feature (separate contributor, separate model)
- [diagram-generation.md](diagram-generation.md) — Mermaid + candlestick diagrams
  generated from a chat message: JSON spec from the model, deterministic rendering,
  the heal/gate/retry pipeline, and the real-parser CI gate
- [cloud-scaling-plan.md](cloud-scaling-plan.md) — **the serving migration it proposed has
  happened** (Akash-leased GPUs, both tutors resident); ingestion fan-out has not. Its
  laptop-vs-cloud numbers are measured on both sides now, not estimated — read the status note
  at the top before trusting any older sentence in the body
- [voice-assistant.md](voice-assistant.md) — **live and vendor-validated**: the open-mic voice
  pipeline (VAD → STT → RAG/LLM → TTS over a WebSocket), deployed on a rented GPU with a
  real STT (SeamlessM4T-v2) and TTS (XTTS-v2, demo/evaluation license only) vendor chosen by
  bake-off. Read this before touching `app/routers/voice.py`,
  `app/services/stt.py`/`tts.py`/`vad.py`/`turn.py`, or `frontend/src/hooks/useVoiceSession.ts`
  — its status amendment at the top says exactly what shipped and where the evidence lives

## Not architecture

`_archive/`, any notebook version superseded by a newer one of the same task, and
dataset snapshots under `data/` other than the current `*_merged` folder are project
history, not a reference for how the system works now. `blueprint.md` at the project
root is external reference material (condensed from a published systems-design book,
not this project's own history) — useful for pattern ideas at much larger scale, not a
description of what's built here.

**Detail & rationale:** `../../CLAUDE.md`, `../../resurrection.md` (current status/open
decisions — `PROJECT_STATE.md` was superseded and archived on 2026-08-03).
