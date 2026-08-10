// Mirrors app/models/schemas.py exactly — keep in sync with the backend.
export type Language = "fr" | "en" | "ar-MA";
export type Domain = "industrial" | "securite" | "blockchain";

export interface ChatRequest {
  message: string;
  session_id?: string;
  tenant_id?: string;
  domain?: Domain; // defaults server-side to "industrial"
  language?: Language; // omit -> server detects from message text
}

export interface ChatResponse {
  response: string;
  session_id: string;
  sources: string[];
  tokens_used: number;
}

export interface QuizRequest {
  topic: string;
  num_questions?: number; // 1-20, default 5
  tenant_id?: string;
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
  message?: string; // set when 0 grounded questions survived
  sources: string[];
}

export interface ApiErrorPayload {
  error?: { code?: string; message?: string };
}
