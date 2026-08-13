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
export type DomainSource = "page_context" | "retrieval" | "tenant_default" | "pinned";

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
