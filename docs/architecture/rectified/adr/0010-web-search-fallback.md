# ADR 0010: Web-Search Fallback for Out-of-Corpus Questions

**Status:** Implemented, opt-in (`web_search_engine=none` default), verified locally with a live
Ollama call; no live Tavily key exercised yet (network calls are mocked in tests)
**Date:** 2026-09-06
**Depends on:** `app/routers/chat.py` step 3's refusal gate, `app/services/turn.py`'s
`TurnContext.is_refusal` (the same condition, used by voice.py — see Scope below)

## Problem

Tenant RAG only ever answers from the tenant's own uploaded/curated corpus. When retrieval finds
nothing — `domain_source == "no_match"` (the tier-2 domain vote found nothing anywhere in the
corpus) or `not context.strip()` (nothing cleared `similarity_threshold`) — `chat.py` refuses
deterministically via `deterministic_refusal()`. Correct default: answering from nowhere would
either hallucinate or silently borrow the base model's untrusted world knowledge, which is exactly
the citation-fabrication failure mode this platform's fine-tune already has a documented history
of (the untouched base model correctly says "not in the text" where the adapter fabricates a law
number). But it means every out-of-corpus question — including ordinary ones with no relation to
the tenant's domain at all — gets the same refusal, with no path to a real answer.

This ADR adds an **opt-in** fallback: when the refusal gate would fire, try a live web search
first; if it returns something, answer from that instead of refusing — clearly marked, end to end,
as NOT from the tenant's own documents.

## Design

**Engine pattern**, mirroring `app/services/{ocr,stt,tts}.py`: an ABC (`WebSearchEngine`), a
`NullWebSearchEngine` default (`search()` always returns `[]`, so `web_search_engine=none` makes
the new code path in `chat.py` a true no-op — behaviour is byte-identical to before this existed),
and one real engine, `TavilyWebSearchEngine`, behind `tavily_api_key`.

**Provider: Tavily**, not Google/Bing Custom Search or scraping DuckDuckGo/Google result pages.
Tavily is built for feeding LLM/RAG pipelines specifically — results come back as short
pre-extracted snippets, not raw HTML to parse, so there is no scraper to maintain (this platform
already treats fail-open degradation as a hard requirement elsewhere; a scraper that silently
breaks on a markup change is the opposite of that). A free tier exists. Google/Bing Custom Search
need a separate search-engine-ID resource provisioned outside just an API key.

**Transport: stdlib `urllib.request`**, not a new `httpx`/`requests` dependency. The rest of this
codebase's network calls (`app/services/llm.py`'s Ollama calls) already use `urllib.request`
deliberately (see `_post_ollama`'s docstring); one JSON POST doesn't justify a second HTTP client
in the dependency tree.

**Kept structurally separate from tenant-grounded answers, not layered on top of them**, in three
places:

1. **`ChatResponse.sources` is never touched.** A new `answered_from_web: bool` and
   `external_sources: list[ExternalSource]` (title + url) carry the web-fallback answer instead.
   `sources` means "from the tenant's own documents" everywhere else this platform reads it
   (history, pins, the frontend's `SourcesRow`) — conflating the two would silently misrepresent a
   web-fallback answer as tenant-grounded, which is worse than the refusal it replaces.
2. **No history pin.** `is_new_pin`/`history.pin_context` are skipped for this path — there is no
   tenant context here to pin. `history.append_exchange` still runs (so follow-up turns keep
   conversational memory), with `sources=[]` for the same reason as above; a follow-up in the same
   segment simply re-triggers the web-fallback path if the topic is still out of corpus. Persisting
   the web URLs themselves into history is left for a future pass if this proves useful in
   practice — out of scope here to avoid a DB-schema question this ADR doesn't need to answer yet.
3. **A dedicated prompt and generation function**, `app.services.llm.generate_web_fallback_response`
   — NOT a codepath through `generate_llm_response`. Two reasons:
   - `extract_citations`/`inject_citations` pattern-match tenant legal-citation shapes
     (`app/services/citations.py`) that cannot appear in web content; running them here would do
     nothing useful at best, or "inject" a phantom citation into prose that never claimed one at
     worst.
   - The system prompt (`WEB_FALLBACK_PROMPT_TEMPLATE_FR`/`_DARIJA`) explicitly forbids
     article/law-style citation and asks for plain "according to `<real source name>`" attribution
     instead — instructing the model to invent a legal reference for content that has none would
     recreate the exact citation-fabrication failure mode named above.
   - The "this is a web answer, not your documents" disclaimer is **prepended in code**, not left
     to the model to remember — the same "don't trust the model for a hard invariant" reasoning
     `deterministic_refusal()` already applies to refusals.

## Finding: the fine-tuned tutor models are the wrong model for this path

First implementation routed the web-fallback generation through the same `ollama_model_fr`/
`ollama_model` (`iblog-tutor-fr` / `IBLOG_TUTOR`) the rest of the app uses. Live-tested locally
(2026-09-06, scratchpad script, not committed) with a French out-of-domain question
("Comment faire du pain au levain ?") and a web snippet that stated the answer verbatim:

| Model | Runs | Correct | Failure mode |
|---|---|---|---|
| `iblog-tutor-fr` (fine-tuned) | 5 | 1/5 | **False refusal** — "Je n'ai pas cette information dans les extraits fournis," despite the answer being in the extract and an explicit system-prompt instruction to use it |
| `gemma2:9b` (base, no fine-tune, already pulled locally) | 3 | 3/3 | Correct, attributed to the real source name, no fabrication |

This is the same defect `llm.py`'s `SYSTEM_PROMPT_TEMPLATE` docstring already documents for the
ordinary refusal path — "the fine-tuned model's own refusals are welded to tenant #1's safety
domain" — surfacing here as a **false refusal on content it was explicitly given**, rather than a
misidentified refusal persona. The fine-tune's training data is dominated by refusal examples for
one narrow domain; it pattern-matches "this doesn't look like an HSE question" and refuses,
independent of what the prompt asks for. This is a weight-level bias, not a prompt-wording gap —
tightening the instruction wording did not fix it (see below), so no further prompt-only fix was
attempted.

A live Darija check (`gemma2:9b`, "شحال الجو اليوم فالرباط؟" against a weather snippet) also came
back correct, in Arabic script, with correct real-source attribution ("حسب Meteo Rabat") — so
`gemma2:9b` was adopted for **both** languages rather than adding a third model to route between.
This is a deliberately narrow, empirically-verified choice for this one fallback path; it is not a
recommendation to route ordinary tenant-grounded answers through `gemma2:9b` — the fine-tune is
still strictly better there, this defect only shows up on genuinely out-of-corpus content.

`settings.web_search_fallback_model = "gemma2:9b"` is a new, separate setting rather than reusing
`ollama_model`/`ollama_model_fr` — deliberately: an operator changing the tutor model for the main
chat path must not silently also change what answers web-fallback questions.

### A second bug the same live test caught

The first prompt draft used a literal `{{title}}`/`{{titre}}` Python-format placeholder inside an
*example* sentence ("e.g. \"according to {title}\""), intending `.format()` to leave it as literal
text the model would generalize from. Instead both models echoed the placeholder text verbatim —
"selon {titre}" — into the answer. Fixed by naming the real attribution target explicitly ("the
site or document name shown in brackets in the snippets") and telling the model never to write the
literal word "title"/"titre". This is exactly the kind of defect that only running the real prompt
against a real model catches — a prompt that reads correctly to a human can still teach the model
the wrong lesson from an ambiguous example.

## Scope: chat.py only, not voice.py

`app/services/turn.py`'s `TurnContext.is_refusal` encodes the identical `no_match`-or-empty-context
condition and is what `voice.py`'s real (non-echo) pipeline uses via `refusal_text()`. This ADR
does **not** wire the web-fallback into the voice path. Voice answers stream sentence-by-sentence
through `stream_llm_response` for immediate TTS chunking; plugging a non-streaming web-search round
trip into that adds real latency-shape complexity (the mic session would sit in `SPEAKING`-adjacent
silence during the search+generate call, with no partial output to speak) that deserves its own
design pass rather than being folded into this one. Follow-up work, not a defect in this ADR's
scope.

## Verification

- `tests/test_web_search.py` (10 tests): engine parsing, `max_results` truncation, missing-URL
  results skipped, fail-open on `URLError`/`HTTPError`/malformed JSON, `tavily` selected without a
  key falls back to `NullWebSearchEngine` (never crashes the chat turn).
- `tests/test_chat.py` (+3 tests): default (`web_search_engine=none`) is byte-identical to the
  pre-existing refusal, a configured engine with results answers with `answered_from_web=True` and
  `sources=[]`, a configured engine that legitimately finds nothing still falls through to the
  original refusal.
- Full suite: 729 → 742 passed (13 new), GPU free during the run.
- Live Ollama smoke test (both languages, both models, see table above) — not part of the
  committed test suite (network/model-dependent), but is what caught both the model-choice defect
  and the placeholder-echo defect above; neither would have been caught by mocked tests alone.

## Open, for later

- No live Tavily key has been exercised — `TavilyWebSearchEngine`'s HTTP shape is verified against
  Tavily's documented response format and unit-tested with mocked `urlopen`, not against the real
  API. Get a key and run one real query before enabling this for anyone.
- **`gemma2:9b` is not provisioned by `scripts/docker/entrypoint.sh` or baked into
  `config/Dockerfile.gpu`** — it was pulled ad hoc on the dev laptop for this ADR's live test and
  is not part of any lease's image or startup provisioning. `deploy/akash-deploy.yaml` ships with
  `WEB_SEARCH_ENGINE=none` for exactly this reason: enabling `tavily` on a lease today would call
  `generate_web_fallback_response`, which would call Ollama for a model that was never pulled, and
  fail. Before flipping `WEB_SEARCH_ENGINE=tavily` on any lease, add `ollama pull gemma2:9b`
  (~5.4GB) to `entrypoint.sh`'s provisioning step, gated on the same env var so a lease running
  with the fallback off doesn't pay that download for nothing.
- Voice-path integration (see Scope above).
- Whether to persist `external_sources`/web URLs into conversation history for later reference.
