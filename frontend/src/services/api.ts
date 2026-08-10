import type {
  ApiErrorPayload,
  ChatRequest,
  ChatResponse,
  QuizRequest,
  QuizResponse,
} from "../types/api";

const env = (import.meta as { env?: { VITE_API_BASE?: string } }).env;
export const API_BASE = env?.VITE_API_BASE ?? "http://127.0.0.1:8000";

export class ApiError extends Error {
  code: string;

  constructor(code: string, message: string) {
    super(message);
    this.name = "ApiError";
    this.code = code;
  }
}

async function request<T>(path: string, body: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new ApiError(
      "NETWORK_ERROR",
      `Cannot reach backend at ${API_BASE} — is uvicorn running?`,
    );
  }

  if (!res.ok) {
    let payload: ApiErrorPayload | undefined;
    try {
      payload = (await res.json()) as ApiErrorPayload;
    } catch {
      payload = undefined;
    }
    throw new ApiError(
      payload?.error?.code ?? `HTTP_${res.status}`,
      payload?.error?.message ?? `Request failed with status ${res.status}`,
    );
  }

  return (await res.json()) as T;
}

export async function sendChatMessage(req: ChatRequest): Promise<ChatResponse> {
  return request<ChatResponse>("/api/v1/chat/", req);
}

export async function generateQuiz(req: QuizRequest): Promise<QuizResponse> {
  return request<QuizResponse>("/api/v1/quiz/", req);
}

export async function pingHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`, { method: "GET" });
    if (!res.ok) return false;
    const body = (await res.json()) as { status?: string };
    return body.status === "ok";
  } catch {
    return false;
  }
}
