# Atlas Tutor — Pipeline Testing Dashboard

A real UI to drive `/api/v1/chat` and `/api/v1/quiz` live against the Atlas Tutor
backend (RAG retrieval → LLM → citation injection / quiz grounding), so the
pipeline can be exercised click by click instead of read from logs.

## Run it

```bash
cd frontend
npm install      # first time only
npm run dev      # serves on http://localhost:5173
```

The Vite dev server talks directly to the backend at `http://localhost:8000`
(no proxy — the FastAPI app already allows all CORS origins). Override the base
URL if needed: `VITE_API_BASE=https://… npm run dev`.

## Backend prerequisites

- The FastAPI app: `../.gguf_venv/Scripts/python.exe -m uvicorn app.main:app --reload`
  (from the repo root).
- Ollama serving `IBLOG_TUTOR:latest`.
- **Postgres/pgvector and Redis** for real RAG retrieval. `build_rag_context`
  needs a live DB connection even to return an empty result — if Postgres is
  down, chat and quiz return a 500 connection error, which the dashboard
  surfaces as an error toast. With the DB up but no tenant documents ingested,
  you get the deterministic refusal path instead (a clean, domain-correct
  message — exactly what the dashboard exists to exercise).

## What you can test

- **Domain-correct refusals** — switch the domain selector to `Sécurité` or
  `Blockchain` and ask an off-topic question with no matching context: the
  refusal must name that domain, never "safety" (السلامة) wording.
- **Quiz grounding** — "Generate Quiz" opens a modal (topic, question count,
  language); the response is injected into the chat stream as an interactive
  quiz card. Click an option to reveal correct/incorrect + explanation. If no
  questions survive the grounding filter, the API's `message` is rendered as a
  plain assistant bubble instead.
- **Chat history** — sessions persist to `localStorage`; reload the page and
  the conversation survives. A fresh browser profile starts empty.
- **Pipeline status** — the sidebar indicator pings `/health` every 20 s;
  green = backend reachable, red = offline.

## Notes

- File attachment is a UI-only mock (wiring the ingestion/embedding pipeline
  into a router is out of scope).
- "Generate Video" is a placeholder that shows a "coming soon" toast.
- Chat history is client-side only: `session_id` is sent to the backend and
  echoed back, but nothing is persisted server-side.

## Stack

Vite + React + TypeScript, Tailwind CSS v4 (CSS-first config in `src/index.css`),
lucide-react icons, react-markdown + remark-gfm for assistant replies, Google
Fonts (Plus Jakarta Sans / Outfit / JetBrains Mono). No router, no state library
— React Context + hooks, backed by `localStorage`.
