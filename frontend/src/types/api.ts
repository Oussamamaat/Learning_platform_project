// Mirrors app/models/schemas.py exactly — keep in sync with the backend.
export type Language = "fr" | "en" | "ar-MA";
export type Domain = "industrial" | "securite" | "blockchain";
// Backend's internal response-language vocabulary (app.services.llm /
// app.services.routing) -- distinct from Language above (the request-side
// enum, which also carries "en" and spells Darija "ar-MA"). Mirrors
// ChatResponse.language / QuizResponse.language exactly.
export type ResponseLang = "fr" | "darija";
// How ChatResponse.domain / QuizResponse.domain was resolved --
// app.services.routing.resolve_domain's three tiers, plus "pinned" when a
// chat turn reused session continuity instead of a fresh tier 1/2/3
// decision (chat only; quiz has no session so never returns "pinned").
//
// "no_match" means the tier-2 vote ran and nothing cleared the routing
// threshold: the query is out of corpus, and the accompanying response is a
// deterministic refusal. Distinct from "tenant_default", which means routing
// could form no opinion at all (disk backend, or the vote's search failed).
export type DomainSource =
  | "page_context"
  | "retrieval"
  | "no_match"
  | "tenant_default"
  | "pinned";

// app.models.schemas.DiagramKind
export type DiagramKind = "flowchart" | "sequence" | "mindmap" | "pie" | "xy" | "candlestick";

// app.models.schemas.Candle -- one OHLC bar. Used both as ChatRequest.candles
// (caller-supplied real data, rendered verbatim) and inside
// DiagramPayload.spec.candles (whichever generate_diagram actually used).
export interface Candle {
  label: string;
  open: number;
  high: number;
  low: number;
  close: number;
}

// app.models.schemas.DiagramPayload
export interface DiagramPayload {
  kind: DiagramKind;
  // How `kind` was chosen: "keyword" (Tier 1, deterministic) or "semantic"
  // (Tier 2 fallback, not yet implemented -- see app.services.diagrams).
  kind_source: string;
  title: string;
  // Also used as ChatResponse.response for this turn -- follows the turn's
  // response language, unlike the diagram's own structural labels, which
  // always follow the backend's diagram_label_language setting (French).
  caption: string;
  // Mermaid source, populated for every kind except "candlestick" (Mermaid
  // has no OHLC diagram type -- see `spec` instead).
  mermaid?: string;
  // Raw render data for kinds Mermaid can't express. Only populated for
  // "candlestick": { candles: Candle[] }.
  spec: { candles?: Candle[] };
  // True when retrieval found tenant context and every structural label
  // passed the ungrounded-reference check. False means the diagram is
  // illustrative (e.g. a candlestick pattern with no corpus) -- still
  // shown, just flagged as not sourced from the tenant's own documents.
  grounded: boolean;
  // Deterministic salvage actions taken before rendering (e.g. a dangling
  // edge dropped) -- never a fabrication, only a removal.
  repairs: string[];
  sources: string[];
}

export interface ChatRequest {
  message: string;
  session_id?: string;
  tenant_id?: string;
  // Omitted since 2026-08-11 (Automatic Domain Routing) -- the server
  // resolves both via app.services.routing when these are absent. Still
  // accepted for a future caller that already knows its own domain (e.g.
  // a course-module page) or wants to force a specific response language.
  domain?: Domain;
  language?: Language;
  // Which of the tenant's uploaded sources may ground this turn -- a
  // NARROWING hint only (app.services.sources.active_source_ids
  // intersects this against server-side ready/partial+enabled state, never
  // widens it). Omitted means every currently-enabled uploaded source is
  // eligible; the global corpus is never affected by this field.
  active_source_ids?: string[];
  // Real OHLC data for a candlestick diagram, when the caller already has
  // it. Rendered verbatim -- overrides whatever the model would have
  // invented. Ignored unless the message's detected diagram intent is
  // "candlestick"; omitted means the model invents illustrative values for
  // the named pattern.
  candles?: Candle[];
}

export interface ChatResponse {
  response: string;
  session_id: string;
  sources: string[];
  tokens_used: number;
  domain: Domain; // the domain this turn was actually answered in
  domain_source: DomainSource;
  language: ResponseLang; // the response language actually used
  prior_questions: string[]; // Socratic questions already asked outside the replay window
  cross_language: boolean; // true when grounded sources are in a different script than the reply
  // True when pgvector retrieval failed and this answer fell back to the
  // disk backend, which knows nothing about tenant uploads -- the answer
  // is still grounded in the built-in corpus, but any uploaded sources are
  // silently absent from it until Postgres recovers.
  degraded: boolean;
  // Set when the message's own text triggered diagram intent detection --
  // there is no separate diagram endpoint. `response` is the diagram's own
  // caption in that case. Undefined for every ordinary chat turn.
  diagram?: DiagramPayload;
}

export interface QuizRequest {
  topic: string;
  num_questions?: number; // 1-20, default 5
  tenant_id?: string;
  domain?: Domain; // omit -> auto-routed, same as chat
  language?: Language;
}

export interface QuizQuestion {
  question: string;
  options: string[];
  correct_index: number;
  explanation: string;
}

export interface QuizResponse {
  questions: QuizQuestion[];
  topic: string;
  total_questions: number;
  // What was actually asked for (QuizRequest.num_questions) -- compare
  // against total_questions to detect a shortfall, e.g. thin source material.
  requested_questions: number;
  message?: string; // set when 0 grounded questions survived
  sources: string[];
  domain: Domain;
  domain_source: DomainSource;
  language: ResponseLang;
}

export interface ApiErrorPayload {
  error?: { code?: string; message?: string };
}

export type SourceStatus = "pending" | "processing" | "ready" | "partial" | "error";

export interface UnprocessedPage {
  page: number;
  reason: string;
  // Why the OCR attempt itself failed (subprocess error, timeout, missing
  // engine) -- distinct from `reason`, which is why the page was classified
  // as needing OCR in the first place. Optional: only OCR_REQUIRED skips
  // populate it.
  detail?: string;
}

export interface SourceFile {
  id: string;
  filename: string;
  status: SourceStatus;
  error_message?: string;
  enabled: boolean;
  domain?: Domain | string;
  language?: string;
  chunk_count: number;
  page_count?: number;
  // Live progress while status="processing": pages attempted so far, out
  // of page_count. Undefined/null once ready/partial/error -- read
  // chunk_count/page_count for the finished result instead.
  pages_done?: number;
  // How the file was parsed -- pdf_text | pdf_ocr | pdf_mixed | docx |
  // pptx | xlsx | csv | text | image_ocr -- an audit trail for "was this
  // text OCR'd".
  parser?: string;
  ocr_engine?: string;
  // Set when status="partial": pages skipped rather than failing the whole
  // document over (e.g. a scanned page with no OCR engine configured) --
  // the other pages' chunks are already stored and retrievable.
  unprocessed_pages?: UnprocessedPage[];
  size_bytes: number;
  created_at: string; // ISO datetime
  // Set when this upload's sha256 matched an existing ready source -- the
  // returned row IS that existing source, not a new one.
  duplicate_of?: string;
}

export interface SourceListResponse {
  sources: SourceFile[];
  ready_count: number;
  total_chunks: number;
}
