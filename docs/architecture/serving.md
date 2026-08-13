# Serving

## What's actually deployed

Two fully-merged models in Ollama, `app/config.py`: `IBLOG_TUTOR:latest` (Darija,
`ollama_model`) and `iblog-tutor-fr:latest` (French, `ollama_model_fr`) — each an
adapter baked permanently into its own weights. `app/routers/chat.py` resolves domain,
language, and turn context (see below), then calls `generate_llm_response`
([llm.py](../../app/services/llm.py)), which:

1. Takes `query_lang` (retrieval affinity) and `response_lang` (which model/template
   answers) as already-resolved inputs from `app.services.routing.resolve_language` —
   see [data-and-retrieval.md](data-and-retrieval.md#automatic-domain-routing--language-detection-2026-08-11).
   Script detection itself (`detect_query_language`) is a two-branch Arabic-script-count
   check; Arabizi is out of scope, so the Latin default is `fr`, not `darija`.
2. Picks the model by `response_lang` (`ollama_model_fr` for `"fr"`, `ollama_model`
   otherwise) and builds a system prompt from the matching template, with the retrieved
   context embedded after a `CONTEXTE :` marker plus any prior turns (see Conversation
   history below).
3. Calls Ollama's `/api/generate` with `temperature 0.2`, `num_ctx 4096`.

**Known gap:** French *refusals* (context insufficient) still render in Darija
regardless of question language — `grounded_refusal` training data is 417 rows, 100%
Arabic-script, and that prior outweighs prompt instructions. Answerable French
questions work correctly. Fix requires new French refusal training data, not a prompt
change.

## Conversation history

`app/services/history.py` gives chat server-side (Postgres) multi-turn memory —
[data-and-retrieval.md](data-and-retrieval.md) covers the pinned-context/segment-reset
design and automatic domain/language routing that sit around it in `chat.py`. Fail-open
throughout: a Postgres hiccup degrades a turn to stateless behavior, never a crash.
Replay is strict on `(domain, language, segment_id)` — a different domain, a different
`query_lang`, or a closed-out topic segment is never replayed into a new prompt, since
doing so would reintroduce cross-language/cross-domain context contamination at the
history layer. The trained shape is one prior exchange; the window builder currently
loads up to 2.

## Quiz generation

`app/routers/quiz.py` / `app/services/quiz.py`: retrieve context, call the
language-appropriate Ollama model with the requested count stated explicitly in the
prompt (`"Fais-moi exactement N questions."` / the Darija equivalent — the adapter was
fine-tuned exclusively on 3-question exemplars, so an unstated count regresses toward
3 regardless of the JSON schema's `maxItems`), then run `filter_grounded_questions`
to drop fabricated/malformed questions. A bounded top-up loop (`QUIZ_MAX_EXTRA_ROUNDS =
1`, `QUIZ_TOPUP_BUDGET_S = 180`) re-calls with the shortfall count and an
`avoid_questions` list (cross-round dedup) when grounding attrition leaves the result
short. `QuizResponse.requested_questions` always reports what was asked for so a
genuine shortfall (thin source material, not a bug) is visible in the API and surfaced
in the UI (`QuizCard.tsx`, `AppContext.tsx` toast) rather than silently returned as a
smaller quiz. Reliably honoring counts well above ~5 without top-up is ultimately a
training fix (variable-count quiz exemplars); the prompt hint + top-up loop are
serving-side mitigations.

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
