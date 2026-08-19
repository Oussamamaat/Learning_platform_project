# Video generation — what you need from me

## Overview

You're building the explanatory-video model as its own feature. My side (this repo) is
an AI tutor: it retrieves the right course content and generates a grounded text
explanation. **You don't need my retrieval/context pipeline, and you don't need
embeddings** — by the time text reaches you, it's already a finished, plain-language
explanation string. You're a clean text → video black box.

The whole connection between us is one database table (`video_jobs`) and 4 HTTP
endpoints. No shared code.

**Important:** `input_text` is always MY tutor's own finished answer — never the raw
question the user typed. E.g. a user asking "explique-moi les bonnes pratiques de
sécurité" never gets forwarded to you as-is; my side retrieves the relevant document
content and generates the actual grounded explanation first, and *that* generated text
is what you receive. You never need document access.

Note this is **not** the same text shape as a chat answer. Normal chat/quiz answers
use a Socratic method (a question back to the learner before the direct answer) — that
doesn't work for a one-shot video with no way for the viewer to respond. For video
mode, my side generates a *purely explanatory* variant instead: a direct, adaptive
walkthrough of the topic, no questions posed to the viewer. Grounding (citations,
legal references, French-term handling) stays the same as chat; only the framing
changes. You're not deciding *how* to explain something, only turning an
already-written explanation into video.

## How it works

1. My backend creates a job: `POST /generate` with the explanation text + language.
   Status starts `pending`.
2. Your worker polls `GET /jobs?status=pending` to find work waiting.
3. You generate the video however you want — your model, your pipeline.
4. You report back: `PATCH /jobs/{id}` with `status=ready` + a `video_url`
   (or `status=error` + a message if it failed).
5. My frontend polls `GET /jobs/{id}` until it sees the result.

## Endpoints (base URL: `TBD — I'll send you this once the server's reachable`)

| Method | Path | Who calls it | Body / notes |
|---|---|---|---|
| POST | `/api/v1/video/generate` | me | `{ text, title?, language, session_id?, tenant_id? }` |
| GET | `/api/v1/video/jobs?status=pending` | you | claim pending work |
| PATCH | `/api/v1/video/jobs/{id}` | you | `{ status: "ready", video_url }` or `{ status: "error", error_message }` |
| GET | `/api/v1/video/jobs/{id}` | me | poll result |

`language` is one of `fr` / `en` / `ar-MA`.

## Locked payload — build against this

**This shape is frozen as of 2026-08-18.** If it has to change, you get told
before it does; it won't move under you.

You never call `POST /generate` in production — I do. What *you* consume is the
job object returned by `GET /jobs?status=pending`. Note the field is
**`input_text`**, not `text` (`text` is only the name on my request side):

```json
{
  "id": "3f2b1c8e-5a91-4d77-9c04-2e6b8a1f0d33",
  "tenant_id": "iblog",
  "session_id": null,
  "input_text": "Pendant une ronde de nuit sur un site industriel, ...",
  "title": "Ronde de nuit : équipement obligatoire",
  "language": "fr",
  "status": "pending",
  "video_url": null,
  "error_message": null,
  "created_at": "2026-08-18T09:14:22.118Z"
}
```

| Field | Type | Notes |
|---|---|---|
| `id` | string (UUID) | Echo this back in your `PATCH /jobs/{id}` |
| `tenant_id` | string | Which company the content belongs to |
| `session_id` | string or `null` | Chat session it came from; often `null`, ignore it |
| `input_text` | string, 1–8000 chars | The finished explanation. This is your model's input |
| `title` | string ≤300 chars, or `null` | Short topic label for an opening frame. Same language as `input_text`. Treat as optional — handle `null` |
| `language` | `"fr"` or `"ar-MA"` | Only these two for now. `"en"` exists in the enum but I'm not sending it; don't build for it yet |
| `status` | `"pending"` \| `"processing"` \| `"ready"` \| `"error"` | Always `"pending"` on the rows you claim |
| `video_url` | string or `null` | Always `null` inbound — you fill this |
| `error_message` | string or `null` | Always `null` inbound — you fill this on failure |
| `created_at` | ISO-8601 timestamp (UTC) | |

Two things to plan for in `ar-MA`:

1. **Bidirectional text.** Darija is Arabic script, but French technical terms
   stay in Latin letters inside the same string (`la ronde`, `gilet
   réfléchissant`). One string, mixed RTL and LTR. This is the shape most
   likely to break subtitle layout and TTS alignment, so test it early.
2. **UTF-8, unescaped.** Sample files ship as UTF-8 with real Arabic
   characters, not `\uXXXX` escapes.

### Sample batch files

Ahead of live wiring I'll send you `.jsonl` files: one job object per line, in
exactly the shape above, so each line is what a single `GET /jobs` element
looks like. Read them line by line — no outer array, no wrapper object.

## Testing before my context pipeline is fully wired

You don't have to wait on me — since you only ever receive plain text, you can create
your own test jobs with sample sentences (any length/language you expect) directly via
`POST /generate`, then run your worker against them exactly like real jobs. My side of
the pipeline (turning a real tutoring answer into that text) can keep changing
independently without affecting your testing at all.

## Sample input text (for adapting your model today, no live wiring needed)

These are illustrative — written to match how the tutor explains a topic directly
(grounded in a legal/procedural reference, French technical terms kept in Latin
letters even in the Darija version) — not pulled from the live corpus. Use them to
calibrate tone, structure, and length; don't treat the specific facts as verified
content.

**French** (`language: "fr"`):

> Pendant une ronde de nuit sur un site industriel, le principal risque pour un agent
> de sécurité est l'exposition à des zones mal éclairées ou à un équipement en
> mouvement sans signalisation. L'article 12 du règlement intérieur de sécurité impose
> donc le port d'un gilet réfléchissant et d'une lampe frontale homologuée pendant
> toute la durée de la ronde, en plus du talkie-walkie de liaison avec le poste
> central.

> Pour le contrôle d'accès, la procédure de verrouillage repose sur deux éléments
> complémentaires : le badge nominatif, qui trace chaque passage horodaté, et le
> registre de passage papier, tenu en doublon en cas de panne du système
> électronique. Une zone à accès restreint doit être reverrouillée immédiatement après
> le passage d'un technicien — c'est la responsabilité de la personne qui l'a ouverte,
> pas du poste de sécurité.

**Darija, Arabic script** (`language: "ar-MA"` — note French terms stay in Latin
letters, exactly as the tutor is required to write them, never transliterated into
Arabic):

> منين كيدور حارس الأمن la ronde ديال الليل فموقع صناعي، الخطر الأساسي هو التعرض
> للمناطق اللي ماشي مضويين مزيان أولا لمعدات كتتحرك بلا إشارة. حيت هادشي، la
> procedure ديال المادة 12 كتلزم لبس gilet réfléchissant ولامب فراسو معتمد طيلة
> la ronde، زيادة على talkie-walkie ديال الاتصال مع البوسط المركزي.

> بالنسبة لمراقبة الدخول، la procedure ديال الكادناص كتقوم على جوج حوايج: البادج
> الشخصي اللي كيسجل كل مرور بالوقت، و le registre ديال المرور بالورقة، محفوظ زيادة
> إلا وقع عطل فالنظام الإلكتروني. المنطقة محدودة الدخول خاصها تترد تسد مباشرة منين
> يخرج منها التقني — هادي مسؤولية الشخص اللي حلها، ماشي ديال poste de sécurité.

Note the shape: direct, structured explanation grounded in a concrete procedure or
article, no questions posed to the viewer, legal references kept verbatim, French
technical vocabulary preserved untranslated in both languages. Whatever your model
receives will look like this.

## Decided (you don't need to answer these)

- **Title/topic label:** included, as the optional `title` field above. Handle
  `null` and you're covered either way.
- **Languages:** `fr` and `ar-MA` only. No English samples are coming, so don't
  build for `en` yet.
