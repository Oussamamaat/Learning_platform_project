# Architecture — current state

This folder is the fast path for "what's actually true right now." For *why* each
decision was made, follow the "detail & rationale" pointer at the bottom of each file
into the root-level docs — those remain the record; nothing here duplicates them, it
distills them.

## How the pieces fit

A tenant's question arrives at `POST /api/v1/chat` ([chat.py](../../app/routers/chat.py)).
The backend embeds the query, retrieves the top-k matching chunks from that tenant's
documents in pgvector, builds a system prompt in the detected language (Darija or
French) around that retrieved context, and sends it to the tenant's model in Ollama.
The model is instructed to ground its answer strictly in the retrieved context and to
refuse rather than fabricate when the context doesn't cover the question.

- [data-and-retrieval.md](data-and-retrieval.md) — corpus, chunking, embeddings, pgvector
- [finetune-pipeline.md](finetune-pipeline.md) — base model, LoRA config, generation + training pipeline
- [serving.md](serving.md) — what's deployed today vs. the documented target architecture

## Not architecture

`_archive/`, any notebook version superseded by a newer one of the same task, and
dataset snapshots under `data/` other than the current `*_merged` folder are project
history, not a reference for how the system works now. `blueprint.md` at the project
root is external reference material (condensed from a published systems-design book,
not this project's own history) — useful for pattern ideas at much larger scale, not a
description of what's built here.

**Detail & rationale:** `../../CLAUDE.md`, `../../PROJECT_STATE.md`.
