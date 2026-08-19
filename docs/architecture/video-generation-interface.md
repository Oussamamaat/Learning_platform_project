# Video generation — interface contract

Explanatory video generation is a separate feature owned by a second contributor,
using his own model/pipeline. This doc is the whole connection point between his work
and the rest of the platform: a shared `video_jobs` table plus four endpoints in
[`app/routers/video.py`](../../app/routers/video.py). Neither side needs to read the
other's code — only this contract.

## Flow

1. Our app calls `POST /api/v1/video/generate` with the text to turn into a video.
   A `video_jobs` row is created with `status='pending'`; the request returns
   immediately with a job id.
2. The video worker (his side — in-process, a separate service, a script, doesn't
   matter which) polls `GET /api/v1/video/jobs?status=pending` to find work.
3. It does its own thing to generate the video, then reports the result with
   `PATCH /api/v1/video/jobs/{id}`: `status='ready'` + `video_url`, or
   `status='error'` + `error_message`.
4. Our app polls `GET /api/v1/video/jobs/{id}` until `status` is `ready`/`error`, same
   pattern as document upload (`app/routers/ingest.py`).

## Endpoints

| Method | Path | Called by | Purpose |
|---|---|---|---|
| POST | `/api/v1/video/generate` | us | create a job |
| GET | `/api/v1/video/jobs?status=pending` | his worker | claim work |
| PATCH | `/api/v1/video/jobs/{id}` | his worker | report result |
| GET | `/api/v1/video/jobs/{id}` | us | poll result |

## Job fields (`video_jobs` table / `VideoJobOut`)

- `input_text` — the content to turn into a video (required in)
- `language` — `fr` \| `en` \| `ar-MA` (required in)
- `status` — `pending` → `processing` → `ready` \| `error`
- `video_url` — set by his worker when `status='ready'`
- `error_message` — set by his worker when `status='error'`
- `session_id` / `tenant_id` — optional linkage back to the chat/quiz turn that
  triggered it

## Explicitly out of scope here

- **Where video generation gets triggered from** (a chat turn, a quiz explanation, a
  manual button) — a product decision, not part of this contract. Whatever calls
  `POST /generate` just needs `input_text` + `language`.
- **How/where the video worker runs, what model it uses, storage for the video
  file itself** — his side entirely. `video_url` can point anywhere reachable by the
  frontend (S3, local static path, CDN).
- Auth between the two sides — this repo currently has no auth on any endpoint
  (see `app/config.py`'s `uploads_read_only` for the closest analog); revisit before
  this is multi-tenant-exposed.
