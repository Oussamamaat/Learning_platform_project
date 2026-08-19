import type {
  ApiErrorPayload,
  ChatRequest,
  ChatResponse,
  QuizRequest,
  QuizResponse,
  SourceFile,
  SourceListResponse,
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

async function handleResponse<T>(res: Response): Promise<T> {
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

async function request<T>(
  path: string,
  body?: unknown,
  method: "GET" | "POST" | "PATCH" | "DELETE" = "POST",
): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      method,
      headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
  } catch {
    throw new ApiError(
      "NETWORK_ERROR",
      `Cannot reach backend at ${API_BASE} — is uvicorn running?`,
    );
  }
  return handleResponse<T>(res);
}

// Multipart upload -- deliberately NOT reusing request<T>, which hardcodes
// a JSON Content-Type header. A FormData body needs NO Content-Type set
// explicitly: the browser computes the multipart boundary itself and sets
// the header accordingly, so setting it manually here would break the
// upload by omitting that boundary.
async function requestForm<T>(path: string, form: FormData): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, { method: "POST", body: form });
  } catch {
    throw new ApiError(
      "NETWORK_ERROR",
      `Cannot reach backend at ${API_BASE} — is uvicorn running?`,
    );
  }
  return handleResponse<T>(res);
}

export async function sendChatMessage(req: ChatRequest): Promise<ChatResponse> {
  return request<ChatResponse>("/api/v1/chat/", req);
}

export async function generateQuiz(req: QuizRequest): Promise<QuizResponse> {
  return request<QuizResponse>("/api/v1/quiz/", req);
}

export async function uploadSources(files: File[], domain?: string): Promise<SourceFile[]> {
  const form = new FormData();
  for (const file of files) form.append("files", file);
  if (domain) form.append("domain", domain);
  return requestForm<SourceFile[]>("/api/v1/ingest/upload", form);
}

export async function listSources(): Promise<SourceListResponse> {
  return request<SourceListResponse>("/api/v1/ingest/sources", undefined, "GET");
}

export async function setSourceEnabled(id: string, enabled: boolean): Promise<SourceFile> {
  return request<SourceFile>(`/api/v1/ingest/sources/${id}`, { enabled }, "PATCH");
}

export async function deleteSource(id: string): Promise<{ deleted_chunks: number }> {
  return request<{ deleted_chunks: number }>(`/api/v1/ingest/sources/${id}`, undefined, "DELETE");
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
