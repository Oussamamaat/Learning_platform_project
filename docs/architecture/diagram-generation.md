# Diagram generation

Triggered by the chat message itself — no separate endpoint, no button. `POST
/api/v1/chat` ([chat.py](../../app/routers/chat.py)) runs a deterministic keyword
router ([`detect_diagram_intent`](../../app/services/diagrams.py)) over the message
before the usual empty-context refusal gate; a match routes to
[`generate_diagram`](../../app/services/diagrams.py) instead of the normal
`generate_llm_response` call, and `ChatResponse.diagram` carries the result. Six
kinds: `flowchart`, `sequence`, `mindmap`, `pie`, `xy`, `candlestick`.

## Why the model never writes Mermaid directly

`_archive/docs_superseded/STRATEGY_V2.md:124-142` measured the fine-tune writing
Mermaid syntax directly: semantically reliable (6/6) but syntactically broken in two
recurring ways — unquoted labels and malformed edge-label syntax. Both are retired
structurally, not by teaching the model better syntax: the model only ever emits a
JSON object, constrained by an Ollama `format` schema per kind
([`diagrams.py`](../../app/services/diagrams.py)'s `_SCHEMAS`); Python renders the
validated spec into Mermaid ([`diagram_render.py`](../../app/services/diagram_render.py)).
This also solves candlesticks, which Mermaid has no chart type for at all — those
render client-side from raw JSON instead of Mermaid.

## Three tiers between the model and the response

1. **Heal** (`diagrams.py`'s `_heal_*` functions) — salvages rather than discards. A
   dangling edge is dropped, not repaired by inventing a phantom node; every repair is
   recorded in `DiagramPayload.repairs`, never applied silently. Below a per-kind
   salvage floor (e.g. a flowchart needs ≥2 nodes and ≥1 edge), the generation attempt
   is rejected instead of rendering a husk.
2. **Gate** (language + grounding) — every structural label (not the caption) must be
   Latin-script French; when retrieval found context, every label is run through the
   same `row_has_ungrounded_reference` detector `grounding.py` wraps for quiz
   questions, so a diagram can't fabricate a legal reference any more than a quiz can.
3. **Retry once**, then give up — the chat turn falls back to an ordinary prose answer
   with no diagram, the same fallback the archived design specified, reached
   deterministically instead of by a second model guess.

## Grounding is hybrid, not strict

Unlike quiz generation, an empty retrieved context does **not** refuse a diagram
request — `DiagramPayload.grounded` is set to `false` instead, and the diagram is
still produced and shown. A candlestick pattern has no corpus under the current
three-domain enum (`industrial` / `securite` / `blockchain`); refusing every
candlestick request outright would make the feature useless for its own stated use
case. `ChatRequest.candles` lets a caller supply real OHLC data, rendered verbatim —
the model only invents illustrative values when none is supplied.

## Real-parser CI gate

Every rendered fixture — including healed output — is piped through the actual
`mermaid.parse()` (Node, via jsdom) in
[`frontend/scripts/verify_mermaid.mjs`](../../frontend/scripts/verify_mermaid.mjs),
invoked by [`tests/test_diagram_mermaid_parses.py`](../../tests/test_diagram_mermaid_parses.py).
This caught a real defect during development: `sequenceDiagram`'s grammar is not the
HTML-entity-sanitized text flowchart/pie/xy/mindmap labels go through — `&amp;`/`&quot;`
are parse errors there, where they are required everywhere else — which is why
`diagram_render.py` has a second, sequence-specific escaping path
(`_sequence_label`/`_sequence_message_text`). The archived plan put this check at
serve time; it lives in CI here instead, because the model no longer writes Mermaid
directly — the shape is already fixed by the JSON schema, so a serve-time re-parse
would add Node and latency to every request without catching anything new.

## Frontend

`frontend/src/components/workspace/DiagramCard.tsx` lazy-loads `mermaid` and
`dompurify` together (both stay out of the main bundle — confirmed via `npm run
build`'s chunk output — since most sessions never trigger a diagram) and sanitizes
mermaid's rendered SVG string before `dangerouslySetInnerHTML`, belt-and-braces over
mermaid's own `securityLevel: "strict"`: diagram labels ultimately come from a model
this platform's own archived data-gen work measured as prompt-injectable
(`injection_resistance` exists in `generate_training_data.py` because of it), and
tenant-uploaded documents feed that model. Candlesticks render through
`CandlestickChart.tsx`, a hand-written React SVG component reading the JSON spec
directly — no dependency, and no `dangerouslySetInnerHTML` on that path at all, since
React escapes text children by construction.

## Explicitly out of scope

- **No `iblog-diagram` LoRA adapter, no `diagram_generation` training rows.** The
  archived plan needed both because it had the model writing Mermaid directly;
  deterministic rendering removes the reason (see above).
- **No server-side Mermaid-to-SVG/PNG rendering** — would require Node in the serving
  path. Only candlesticks render server-side-adjacent (as JSON, not SVG); every
  Mermaid kind renders in the browser.
- **No streaming** — nothing in this codebase streams (every Ollama call is `stream:
  false`); a diagram is a single artifact per turn regardless.
- **Tier 2 (semantic) intent detection is a stub.** `detect_diagram_intent` only
  implements Tier 1 (deterministic keywords) today; `_semantic_fallback` in
  `diagrams.py` documents the intended cheap implementation (cosine similarity
  against the already-resident embedding model) for implicit requests that carry no
  keyword.

**Detail & rationale:** the plan this feature was built from; `app/services/diagrams.py`
and `app/services/diagram_render.py`'s module docstrings; `_archive/docs_superseded/STRATEGY_V2.md:124-142`.
