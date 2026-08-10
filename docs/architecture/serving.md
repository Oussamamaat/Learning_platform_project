# Serving

## What's actually deployed

A single, fully-merged model in Ollama (`IBLOG_TUTOR`, `app/config.py`) — the adapter
baked permanently into the weights. `app/routers/chat.py` does pgvector retrieval, then
calls `generate_llm_response` ([llm.py](../../app/services/llm.py)), which:

1. Detects query language (`detect_query_language`) — Darija by default, French if
   French markers/accents are found and Arabizi markers aren't (Arabizi always routes
   to the Darija template, since the user wants an Arabic-script answer back).
2. Builds a system prompt from one of two templates in that language, with the
   retrieved context embedded after a `CONTEXTE :` marker.
3. Calls Ollama's `/api/generate` with `temperature 0.2`, `num_ctx 4096`.

**Known gap:** French *refusals* (context insufficient) still render in Darija
regardless of question language — `grounded_refusal` training data is 417 rows, 100%
Arabic-script, and that prior outweighs prompt instructions. Answerable French
questions work correctly. Fix requires new French refusal training data, not a prompt
change.

**Known gap:** no conversation-history mechanism — every request is independent. Roughly
half of the Socratic/code-switching training data is multi-turn dialogue, so this is a
trained-but-unservable behavior, not a model defect.

## Documented target vs. reality

The architecture plan on file is **vLLM with multi-LoRA**: one frozen base model,
multiple adapters hot-swapped per capability. What's running is the **opposite
direction** — a merged model can't be un-merged back into a hot-swappable adapter.
This is a deliberate MVP-speed trade, not an oversight: ship merged now, treat
multi-LoRA as a post-MVP migration once more than one capability needs to share a base
model.

## Train/serve parity — safety-critical

Gemma-2 has no native system-role chat turn. Training data assumes one exists, so a
custom template merges system content into the first user turn. **This template is
asserted byte-identical between the training data and the production prompt, checked
at build time** — a silent mismatch here degrades output quality invisibly, with no
error and no obvious signal pointing back to the cause. The French system prompt
template was added as a *separate* template specifically to avoid touching this
invariant.

**Detail & rationale:** `../../resurrection.md` §7, §Q6.2, `../../app/services/llm.py`.
