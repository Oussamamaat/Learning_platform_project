# Architecture — current state

This folder is the fast path for "what's actually true right now." For *why* each
decision was made, follow the "detail & rationale" pointer at the bottom of each file
into the root-level docs — those remain the record; nothing here duplicates them, it
distills them.

## How the pieces fit

A tenant's question arrives at `POST /api/v1/chat` ([chat.py](../../app/routers/chat.py)).
The backend auto-resolves domain and language per turn (no UI selector for either),
reuses a pinned retrieval context across same-topic follow-ups via server-side
conversation history, embeds the query, retrieves the top-k matching chunks from that
tenant's documents in pgvector, builds a system prompt in the resolved language (Darija
or French, each served by its own fine-tuned model) around that context and any prior
turns, and sends it to Ollama. The model is instructed to ground its answer strictly in
the retrieved context and to refuse rather than fabricate when the context doesn't
cover the question.

- [data-and-retrieval.md](data-and-retrieval.md) — corpus, chunking, embeddings, pgvector
- [finetune-pipeline.md](finetune-pipeline.md) — base model, LoRA config, generation + training pipeline
- [serving.md](serving.md) — what's deployed today vs. the documented target architecture
- [video-generation-interface.md](video-generation-interface.md) — contract with the
  explanatory-video feature (separate contributor, separate model)
- [diagram-generation.md](diagram-generation.md) — Mermaid + candlestick diagrams
  generated from a chat message: JSON spec from the model, deterministic rendering,
  the heal/gate/retry pipeline, and the real-parser CI gate
- [cloud-scaling-plan.md](cloud-scaling-plan.md) — **target, not built**: how ingestion and
  serving change off the single-laptop deployment. Unlike every other file here it
  describes what does *not* exist yet, so read it as a migration brief; its "today" figures
  are measured and its cloud figures are labelled estimates
- [voice-assistant.md](voice-assistant.md) — **code scaffolding built, not vendor-validated**:
  the open-mic voice pipeline (VAD → STT → RAG/LLM → TTS over a WebSocket). Read this
  before touching `app/routers/voice.py`, `app/services/stt.py`/`tts.py`/`vad.py`/`turn.py`,
  or `frontend/src/hooks/useVoiceSession.ts` — it records exactly what's tested vs. still
  unverified (no STT/TTS vendor selected yet) and why

## Not architecture

`_archive/`, any notebook version superseded by a newer one of the same task, and
dataset snapshots under `data/` other than the current `*_merged` folder are project
history, not a reference for how the system works now. `blueprint.md` at the project
root is external reference material (condensed from a published systems-design book,
not this project's own history) — useful for pattern ideas at much larger scale, not a
description of what's built here.

**Detail & rationale:** `../../CLAUDE.md`, `../../resurrection.md` (current status/open
decisions — `PROJECT_STATE.md` was superseded and archived on 2026-08-03).
