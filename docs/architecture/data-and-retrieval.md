# Data and retrieval

## Corpus

37 documents, ~1,625 chars/doc average, covering three domains for tenant #1
(industrial safety, "sécurité," blockchain-adjacent compliance) — see
[raw/CORPUS_INDEX.csv](../../raw/CORPUS_INDEX.csv). **This is a placeholder corpus**,
not the client's real regulatory documents (`prd_mvp.md` Task 1.4 describes it as such
while awaiting official materials). Every downstream sizing decision — chunk count,
row-generation ceiling, when dataset generation stopped — is scoped to this corpus and
will need revisiting once real documents replace it.

## Ingestion → chunking → embedding

[ingestion.py](../../app/services/ingestion.py): `.txt`/`.md`/`.pdf`/`.docx`/`.pptx`/
`.xlsx`/`.csv`/images, all normalized to markdown before chunking so the embedding model
sees clean content and a single heading-aware chunker covers every format. See the
2026-08-13 section below for the multi-format parsing and embedding-model migration.

| Setting | Value | Since |
|---|---|---|
| Chunker | `RecursiveCharacterTextSplitter` | — |
| Chunk size | 2000 chars | 2026-08-13 (was 400) |
| Chunk overlap | 250 chars | 2026-08-13 (was 50) |
| Embedding model | `BAAI/bge-m3` (1024-dim, 8192-token window) | 2026-08-13 (was `paraphrase-multilingual-MiniLM-L12-v2`, 384-dim) |
| Batch size | 64 | — |

The old 400-char chunk size was not an arbitrary framework default, as first thought —
it was implicitly matched to the old embedding model's actual `max_seq_length: 128`
(confirmed from its cached `sentence_bert_config.json`), which silently truncated
anything longer. bge-m3's 8192-token window removes that ceiling; chunk size raised
accordingly.

## Retrieval

[search.py](../../app/services/search.py): query → embedding → pgvector cosine
similarity → top-k, tenant-isolated by `tenant_id`. `chat.py` calls `build_rag_context`
with `top_k=4`, `similarity_threshold=0.35`.

## The untested half

**Retrieval quality has never been measured in this project.** The one evaluation run
performed to date fed the model gold context lifted directly from each eval row's
training data — it never queried pgvector. That was the right call for isolating
fine-tune quality, but it means a wrong-document retrieval and a model hallucination
currently look identical in every metric this project has. Building a ~50-pair labeled
retrieval eval is the largest measurement gap in the system.

### Baseline measurement (2026-08-10)

That eval now exists: [`tests/data/retrieval_eval.jsonl`](../../tests/data/retrieval_eval.jsonl)
(50 pairs — 30 French / 12 Arabic-script / 8 Arabizi, 15 with a `prior_turn`, 5
deliberately out-of-domain), run via
[`scripts/eval_retrieval.py`](../../scripts/eval_retrieval.py). First run, against
today's actual chat backend (`build_domain_context`, `top_k=4`):

| Metric | Value |
|---|---|
| `recall@4` | 0.911 |
| `MRR` | 0.839 |
| `gold_substring_present_in_context` | 0.778 |
| `cross_language_rate` | 0.222 |
| `empty_context_rate` on the 5 out-of-domain queries | **0.000** |

**The last row is the important one.** `build_domain_context` deliberately has no
similarity threshold (every chunk already belongs to the selected domain, so the
docstring's assumption is "the lowest-ranked chunk is still on-topic"). Measured
result: that assumption doesn't hold against a genuinely off-topic question. All 5
out-of-domain probes — a tajine recipe, a GDPR question, a football score, a weight-loss
question, a misrouted Arabizi blockchain query — retrieved a top-4 context and would
**never reach `deterministic_refusal`**, which only fires on `not context.strip()`. This
is the concrete version of "a wrong retrieval and a hallucination look identical" cited
above: today, an off-topic question doesn't even get a chance to be caught before
generation.

Four recall misses at this baseline (`sec-ar-02`, `sec-arz-02`, `bc-ar-01`,
`bc-arz-01` — all Arabic-script or Arabizi queries against Arabic-script sources)
and the 0.222 cross-language rate are consistent with there being no language-affinity
filtering yet (ADR 0002 decision 5, unimplemented) and no per-file language correctness
in ingestion (Arabic files are stored with a tree-wide `language="fr"` default — see
`app/services/ingestion.py:222`).

**This baseline is what every later retrieval change (heading-aware chunking, the
pgvector cutover, language affinity, query condensing) is measured against — re-run
`scripts/eval_retrieval.py` after each and compare, not just eyeball it.**

### Heading-aware chunking (2026-08-10) — mixed result, kept anyway

`app/services/ingestion.py`'s `strip_markdown` deleted `^#{1,6}\s+` **before**
chunking, then `RecursiveCharacterTextSplitter` cut every 400 chars across section
boundaries. Verified concretely: `1.1_code_du_travail_health_safety.md`'s article
number lives *only* in its heading (`## Obligations du travailleur (Art. 283)`) — the
body paragraph below it never repeats "Art. 283" — so a chunk boundary landing between
heading and body silently drops the only citable reference for that section.
`chunk_document()` fixes this structurally: every chunk now carries its heading inline,
so a reference can never be separated from the text it's embedded in.

First attempt (one chunk per heading, regardless of section length) measured **worse**
across the board — this corpus's `المادة 1..7`-style articles are often one or two
sentences each, and isolating every tiny section lost the cross-boundary context flat
chunking used to blend between short adjacent sections. Fixed by packing short sections
together up to `chunk_size` (each piece still renders `heading\n\nbody` before packing,
so every packed section's reference stays inline regardless of grouping).

Full-eval comparison, flat vs. packed heading-aware, disk backend:

| Metric | Flat (baseline) | Packed heading-aware |
|---|---|---|
| `recall@4` | 0.911 | **0.933** |
| `MRR` | 0.839 | 0.813 |
| `gold_substring_present_in_context` (all 45 in-domain rows) | 0.778 | 0.667 |
| `gold_substring_present_in_context` (19 rows whose substring is an actual citation pattern — `المادة N`, `Art. N`, `الباب N` — the specific thing this change targets) | 0.579 | 0.526 |

**Honest read:** `recall@4` improved (finds the right *document* more often — the
primary retrieval-quality measure) and misses dropped from 4 to 3, but the specific
citation-substring metric this change was meant to move is a wash within noise (one
query's difference on 19 rows), and the broader substring metric (which mixes in
generic topical keywords like "EPI" or "consignation", not just citation patterns)
declined. One case regressed outright (`ind-ar-04`, a Darija paraphrase against a
2-sentence source) — plausibly an embedding-model/corpus-scale limitation (this
project's own multilingual embedding choice was never benchmarked against this corpus,
noted above) rather than a chunking defect.

**Decision: keep it.** The structural guarantee (a reference can no longer be silently
separated from its section) is real and worth having regardless of this measurement's
noise, aggregate recall improved, and the corpus is an explicit 37-doc *placeholder* —
tuning chunking heuristics further against its specific idiosyncrasies (many
single-sentence articles) has diminishing return before the client's real documents
arrive. Re-run this comparison once they do.

### Language-affinity retrieval (2026-08-10) — ADR 0002 decision 5, unambiguous win

Implemented in `app/services/retrieval.py` (`retrieve()`, `_select_with_affinity`): a
two-pass selection that prefers chunks whose language matches `ui_lang`, only crossing
the language line when too few same-language chunks exist. Deliberately **opt-in** —
`ui_lang=None` (the default) is plain top-k-by-similarity, identical to pre-existing
behaviour, so a caller that doesn't resolve a `ui_lang` (quiz.py, as of this writing)
is never silently biased. `app/routers/chat.py` passes its resolved `ui_lang` explicitly.

`scripts/eval_retrieval.py --language-affinity` measures the actual effect, on top of
the packed heading-aware chunking above:

| Metric | Off (packed chunking alone) | **On (+ language affinity)** |
|---|---|---|
| `recall@4` | 0.933 | **0.978** |
| `MRR` | 0.813 | **0.887** |
| `gold_substring_present_in_context` | 0.667 | **0.756** |
| `cross_language_rate` | 0.267 | **0.156** |
| misses | 3 (`ind-ar-04`, `sec-ar-02`, `sec-arz-02`) | 1 (`sec-arz-02`) |

Unlike heading-aware chunking's mixed result, this is a clean win on every metric,
including `MRR` (0.887) now exceeding the original flat-chunking baseline (0.839). The
one remaining miss (`sec-arz-02`, an Arabizi query) is plausibly the embedding model's
own Darija-via-Latin-script limitation, not a retrieval-selection defect.

### pgvector backend: two live bugs found and fixed against real Postgres (2026-08-10)

The disk-backend measurements above were re-run against the pgvector backend once
Docker Desktop was available, ingesting the real `raw/shared` corpus under tenant
`company_abc` (which already held 84 unrelated legacy rows — an old road-traffic
corpus from an earlier ingest, `demo_tenant`/`company_abc`, exactly the tenant/corpus
mismatch `domain_context.py`'s own docstring describes). Two real defects surfaced
that the disk backend, having no SQL layer, could never have exposed:

**1. Domain filtering after the fetch, not in the query, starved a domain's own
results.** `search_similar_chunks` originally filtered by `tenant_id` only in SQL,
fetched `top_k*4` candidates by pure similarity, then filtered by `domain` in
Python. Measured directly: for one query, only **1 of 7** real
`industrial`+Arabic-script chunks survived into a 20-candidate pool once
`industrial`/`securite`/`blockchain`/undomained-legacy rows all competed for the
same slots — the other 6 were correctly-matching chunks that never got the chance to
be domain-filtered because they'd already been cut by the `LIMIT`. Fixed by pushing
`domain` into the SQL `WHERE` clause (`search_similar_chunks(..., domain=...)`), so
the `LIMIT` applies to an already-scoped candidate set — the same fix also
transparently excludes the legacy undomained rows, since `domain = 'industrial'`
never matches `NULL`.

**2. The pre-existing `similarity_threshold` (0.4, `search.py`; 0.35, the old
`quiz.py`) was never benchmarked and discarded correct answers outright.** With the
domain-filter fix alone, `recall@4` was still only 0.778 — barely moved. Root cause,
measured directly: correct, gold-source matches for cross-lingual (this
French-centric embedding model scoring Darija) queries scored as low as
**0.155–0.177** — comfortably below the 0.4 cutoff regardless of how correct the
match was. A threshold sweep on the real ingested corpus, `language-affinity` on:

| `similarity_threshold` | `recall@4` | `MRR` | out-of-domain refusal rate |
|---|---|---|---|
| 0.0 – 0.1 | 0.978 | 0.887 | 0.000 |
| **0.15 (new default)** | **0.956** | **0.835** | **0.200** |
| 0.2 | 0.911 | 0.817 | 0.200 |
| 0.25 | 0.867 | 0.802 | 0.400 |
| 0.3 | 0.822 | 0.774 | 0.600 |
| 0.4 (old default) | 0.778 | 0.756 | 0.800 |

A straight tradeoff curve, no free lunch: a higher threshold catches more
out-of-domain junk but throws away more legitimate cross-lingual matches. Set to
**0.15** (`build_rag_context`'s new default, `app/services/search.py`;
`retrieve()`'s pgvector default, `app/services/retrieval.py`) — holds recall near
the ceiling (0.956 vs 0.978 at zero cutoff) while keeping some refusal protection,
consistent with this project's established priority throughout Stage 3 that a false
refusal mid-conversation is the worse failure mode. `quiz.py` no longer passes its
own stale `0.35` override; it inherits this tuned default.

With both fixes, the pgvector backend reaches parity with the disk backend at the
`0.0` threshold (0.978/0.887/0.156, one miss) and lands at 0.956/0.835/0.200 at the
shipped 0.15 default — confirming the domain-filter and language-affinity logic
themselves are correct; the gap that remained before this fix was entirely the
unbenchmarked threshold constant, not the retrieval logic built in Stage 3.

### `pgvector` locked in as chat's default retrieval backend (2026-08-10)

`settings.retrieval_backend` default flipped `"disk"` → `"pgvector"` (`app/config.py`).
Final locked-in benchmark, `scripts/eval_retrieval.py --backend pgvector --language-affinity`
(50-pair set, real ingested corpus, threshold 0.15):

| Metric | Value |
|---|---|
| `recall@4` | **0.956** |
| `MRR` | **0.835** |
| `gold_substring_present_in_context` | 0.733 |
| `cross_language_rate` | 0.222 |
| `empty_context_rate` (out-of-domain, want 1.0) | **0.200** |
| misses | 2 of 50 (`ind-ar-04`, `sec-arz-02`) |

Flipping the default alone would have been a no-op: `app/routers/chat.py` called
`build_domain_context` directly and unconditionally — that function forces the
`"disk"` backend internally regardless of `settings.retrieval_backend`, deliberately,
so quiz's always-pgvector behaviour could never be silently redirected by a flag
meant for chat. Wiring the lock-in required a real code change: a new
`_retrieve_context()` indirection point in `chat.py` that reads the setting and
dispatches to `build_rag_context` (pgvector) or `build_domain_context` (disk).

**Fail-open, extending the same contract `history.py` already keeps.** Disk needs
zero external infra beyond the embedding model; pgvector needs a reachable Postgres.
Locking chat onto pgvector without a safety net would mean a Postgres hiccup mid-demo
crashes the chat request instead of degrading gracefully. `_retrieve_context` catches
any exception from `build_rag_context` and falls back to `build_domain_context` —
only a raised exception triggers the fallback, so a real "nothing matched" empty
result from pgvector is never overridden. Override via the `RETRIEVAL_BACKEND` env
var (standard `pydantic-settings` behavior) to force `"disk"` if needed.

**A second real bug surfaced live during this change, unrelated to retrieval:**
`AppError` never set `self.message` (Python 3's `Exception` has no such attribute by
default), and all three of its catch sites (`app/main.py`'s global handler,
`chat.py`, `quiz.py`) read `.message` directly. A transient Ollama disconnect during
verification turned into `AttributeError: 'OllamaConnectionError' object has no
attribute 'message'` instead of the intended structured `503` response — the error
handler crashed while handling an error. Fixed in `app/errors.py`
(`AppError.__init__` now sets `self.message`); regression-tested in
`tests/test_errors.py`.

**A third bug, also live-only, surfaced from the same unmocked `chat()` call:**
`chat.py` wrote `pinned_fingerprint` as the raw `f"{domain}|{ui_lang}|{message}"`
string rather than calling `retrieval.py`'s `_fingerprint()` hash helper --
`chat_sessions.pinned_fingerprint` is `VARCHAR(64)`, so any real (non-trivial)
message overflowed it. `history.pin_context`'s own fail-open swallowed the
resulting `StringDataRightTruncation` and logged it, so the chat response itself
was unaffected -- but the pin silently never persisted, defeating the KV-prefix-
reuse pinning exists for, on every single turn, undetected until this point.
**`tests/test_history.py`'s SQLite-backed suite could never have caught this** --
SQLite does not enforce `VARCHAR` length the way Postgres does. Fixed by calling
`_fingerprint()` (always a 64-char sha256 hex digest) instead; regression-tested at
the hash-length level in `tests/test_retrieval.py`, which is portable across both
databases.

Full suite (**331 tests**, includes the new backend-selection/fail-open/error-
message/fingerprint tests) green under the pgvector default; `probe_serving_
backstops.py` and a fully live, unmocked `chat()` call -- including a real
Postgres write-and-read-back of the pin row -- both re-verified after all three
fixes.

**Detail & rationale:** `../../resurrection.md` §2, `../../LOCKEDIN_PLAN.md`.

### Automatic domain routing + language detection (2026-08-11)

Both `ChatRequest.domain` and `ChatRequest.language` became optional — no UI selector
picks either anymore. `app/services/routing.py` resolves both server-side per turn:

- **Domain** — three tiers, first hit wins: `page_context` (caller supplied it) →
  `retrieval` (a similarity-weighted vote over one *unfiltered* pgvector search,
  `resolve_domain`'s tier 2) → `tenant_default` (`settings.default_domain`, tier 3).
  The disk backend skips tier 2 (no single cross-domain corpus to vote over) and goes
  straight to tier 3.
- **Language** — `detect_query_language` simplified from a five-branch French-vs-
  Arabizi word-marker heuristic to a two-branch Arabic-script-count check (Arabizi is
  out of scope; the Latin default flipped from `darija` → `fr` as the direct
  consequence). Retrieval affinity (`query_lang`) and the model/system-prompt choice
  (`response_lang`) are now two separate values, resolved together by
  `resolve_language`: explicit field → an in-message instruction ("réponds en darija")
  → a sticky prior override → script default. The split matters because an override
  only changes *how the answer is phrased*, not *what content grounds it*.

**Domain-routing tier 2, measured, not assumed:** `scripts/eval_retrieval.py
--auto-domain --tenant-id company_abc` drops each of the 50 eval pairs' labelled
domain and checks whether the vote picks it back from the query alone.

| Metric | Value |
|---|---|
| accuracy (45 in-domain rows; the 5 out-of-domain rows are excluded — no correct domain exists to check them against) | **0.778** (35/45) |

Misses cluster almost entirely on `securite`, confused with `industrial` (4) and
`blockchain` (4), plus 2 `blockchain` rows routed to `industrial`. Recorded honestly
as a first measurement, not tuned against — the mechanism is unit-tested (weighted
vote, NULL-domain non-voting, fail-open on a DB error — `tests/test_domain_routing.py`)
and live-verified as correctly wired into `chat()`/`generate_quiz()`, but the vote's
*accuracy* at the current corpus size and chunking is a real, open number, not a
solved one — a future pass at tuning the vote (weighting scheme, chunk overfetch,
`securite`'s apparent overlap with the other two domains' vocabulary) is scoped
separately from the wiring itself.

**Language stickiness, live-verified:** `probe_language_routing.py` runs 5 real turns
through the actual `chat()` router with real Postgres persistence
(`chat_sessions.response_lang_override` / `override_query_lang`) and real Ollama
generation — French question (→ `fr`, script default) → French question + "réponds en
darija" (→ `darija`, explicit instruction) → a plain French follow-up with *no*
instruction (→ `darija`, the actual stickiness proof: script-default alone would give
`fr`) → an Arabic-script question (→ `darija`, script default *and* clears the
override since `query_lang` changed) → a plain French follow-up again (→ `fr`, proves
turn 4 truly cleared the override rather than coincidentally matching). All 5 turns
passed. `probe_serving_backstops.py` re-run afterward: refusal interception still 3/3
domains, Ollama still correctly bypassed, grounded happy path and quiz grounding
filter both unaffected.

**Ingestion never writes `domain=NULL` going forward** — `ingest_directory` gained a
`--domain` CLI flag and now resolves per file: path convention → `--domain` flag →
`settings.default_domain`, logging `[INGEST WARNING]` on that last fallback. The
existing 84 `domain IS NULL` rows (`company_abc`/`demo_tenant`, 42 each) turned out to
be a genuinely unrelated pre-existing road-traffic corpus (`code_de_la_route.txt`,
`accidents_de_la_route.txt`, etc.), not stray industrial/securite/blockchain docs —
backfilling them to a real domain would have reintroduced the exact cross-domain
contamination the `domain` column exists to prevent. **Decision: left as `NULL`**,
permanently unreachable by any domain-filtered query, data not destroyed.

Full suite (**364 tests**) green. Frontend `tsc`/`oxlint`/`vite build` all clean;
`DomainSelector.tsx`/`LanguageSelector.tsx` deleted, `TopBar`/`ChatStream` now show
read-only resolved domain (badged `auto` when `domain_source === "retrieval"`) and
language, fed from each response.

**Detail & rationale:** plan file `now-i-wanna-tackle-sleepy-valley.md`
("Automatic Domain Routing + Language/Script Detection").

### Embedding model migration: MiniLM → bge-m3 (2026-08-13)

Driven by the tenant document-upload feature ("Multi-format ingestion + Sources panel"
plan): real uploaded PDFs/slides run far longer than this corpus's short regulatory
articles, and the old embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) silently
truncates at 128 tokens (`max_seq_length` in its own `sentence_bert_config.json`) — the
400-char chunk size it shipped with wasn't an arbitrary default, it was implicitly sized
to that ceiling. Swapped to `BAAI/bge-m3` (1024-dim, 8192-token window, no
instruction-prefix requirement, stronger Arabic). This is a **breaking change** to
`documents.embedding`'s vector width (384 → 1024) — `scripts/migrate_to_bge_m3.py`
archives the 84 legacy `domain IS NULL` road-traffic rows to `documents_legacy_384`
(not reproducible from `raw/`, unlike every domained row), truncates `documents`,
alters the column, and fully re-ingests `raw/shared` with the new model.

**Every number below is fresh — the old MiniLM-era tables above this section remain as
history, not as a comparison baseline; the two models' cosine distributions aren't on
the same scale.** All measured via `scripts/eval_retrieval.py` (pgvector + disk
backends) and a single-process sweep script (avoids a Windows-specific issue where
`sentence-transformers`/CUDA repeatedly spawned from a shell loop would intermittently
stall on process startup — loading the model once and reusing it across thresholds
sidesteps it entirely) against the same 50-pair `tests/data/retrieval_eval.jsonl`.

**Ceiling (threshold 0.0), pgvector, language-affinity on:**

| Metric | bge-m3 | Old MiniLM ceiling |
|---|---|---|
| `recall@4` | **1.000** | 0.978 |
| `MRR` | **0.989** | 0.887 |
| `gold_substring_present_in_context` | **0.978** | 0.756 |
| `cross_language_rate` | **0.133** | 0.156 |

A clean win on every axis before any threshold is even applied.

**Threshold sweep, pgvector, language-affinity on** (`empty_context_rate_ood` = out-of-
domain refusal rate, want 1.0):

| `similarity_threshold` | `recall@4` | `MRR` | `empty_context_rate_ood` |
|---|---|---|---|
| 0.0 – 0.2 | 1.000 | 0.989 | 0.000 |
| 0.3 | 1.000 | 0.989 | 0.200 |
| **0.4 (new default)** | **1.000** | **0.989** | **0.800** |
| 0.5 | 0.800 | 0.778 | 0.800 |
| 0.6 | 0.511 | 0.500 | 1.000 |
| 0.7 | 0.178 | 0.178 | 1.000 |

**0.4 is an unusually clean optimum, not just a compromise point**: it's the *last*
threshold with zero recall cost (still a perfect 1.000) while already capturing most of
the achievable refusal protection (0.8 of the 1.0 ceiling) — above it, recall craters
faster than refusal improves. This is a much sharper separation between relevant and
irrelevant matches than the old model had at its own tuned value (0.15) — a direct
relevant-vs-irrelevant spot check (a French and an Arabic-script query each against a
real corpus doc and an off-topic "tajine recipe" control) scored relevant pairs at
0.58–0.64 against irrelevant pairs at 0.23–0.37, a much wider margin than the old
model showed at this corpus's scale.

`similarity_threshold` (`app/config.py`) set to **0.4** (was 0.15).

**Domain-vote threshold, `--auto-domain`** (separate question from the above — "does
this chunk get a vote," not "is this chunk usable"), swept independently:

| `domain_vote_threshold` | accuracy (45 in-domain rows) |
|---|---|
| 0.15 | 0.667 |
| 0.3 | 0.689 |
| **0.4 (new default)** | **0.889** |
| 0.5 | 0.867 |

0.4 also wins here, and by a wide margin over the low end — **an improvement over the
old model's own measured baseline (0.778)**, not just a like-for-like swap. Misses at
0.4 remain concentrated on `securite`, confused mostly with `industrial`/`blockchain`,
consistent with the pre-existing (unrelated to this migration) vocabulary-overlap note
recorded above. `domain_vote_threshold` (`app/config.py`) set to **0.4** (was 0.15).

**Disk backend**, language-affinity on: `recall@4=1.000`, `MRR=0.989` — matches the
pgvector ceiling exactly, as expected (no threshold on this backend).

**Corpus re-ingest**: 25 files, 37 chunks (down from the old model's larger chunk count,
expected given `CHUNK_SIZE` 400→2000 — most of this placeholder corpus's short articles
now fit in a single chunk). All 37 rows confirmed with a non-NULL `embedding`.

**Two context-budget fixes landed in the same change**, both promoted from theoretical
to certain by the larger chunk size: `retrieval._build_context`'s truncation branch was
dropping a chunk's `source_name` before appending its (truncated) text — at 400-char
chunks this essentially never fired, but at ~2000-char chunks the *first* chunk alone
can exhaust the context budget, which would have produced answers with real grounded
text but an empty `sources` list. Fixed to append the source before truncating.
Separately, `app/services/llm.py`'s two Ollama calls now send an explicit
`"num_ctx": 8192` (verified via `ollama show --modelfile`: both `IBLOG_TUTOR` and
`iblog-tutor-fr` ship `num_ctx 4096`, too little headroom for a 6000-char/~1500-token
context budget plus conversation history plus the response itself). `max_context_length`
raised 2000 → 6000 chars to match.

**PaddleOCR-VL vs. `baidu/Unlimited-OCR`** (the OCR engine named in a prior stalled
spike, referenced in `app/services/ingestion.py`'s original docstring): checked, not
assumed. Unlimited-OCR is real (MIT, 3B params, genuinely well-suited to long documents)
but its model card names no explicit language list and it's reported as optimized
heavily for English/Chinese; it also pins `torch==2.10.0`/CUDA 12.9, incompatible with
`.gguf_venv`'s torch 2.11.0+cu128. PaddleOCR-VL has explicit, dedicated Arabic
recognition and the strongest published OmniDocBench v1.6 score in its class — set as
the default engine (`app/services/ocr.py`), with Unlimited-OCR kept as a second
pluggable backend. Neither is trusted by default: `settings.ocr_engine` stays `"none"`
until `scripts/verify_ocr_arabic.py` passes against the committed ground-truth image
(`tests/data/ocr/1.12_ar_procedure_consignation.png`) — see that script for the four
gates (fidelity, logical order, not-reversed, digit integrity).

**Detail & rationale:** plan file `you-are-an-expert-shiny-breeze.md` ("Multi-format
ingestion + Sources panel"), Phase 1.

### OCR gate run: PaddleOCR-VL fails on one character (2026-08-14)

First real run of `scripts/verify_ocr_arabic.py --engine paddleocr` against the committed
ground-truth image. **Result: FAILED — `settings.ocr_engine` stays `"none"`.**

| Gate | Measured | Verdict |
|---|---|---|
| 1. Fidelity | 0.946 (≥ 0.90 required) | PASS |
| 2. Logical order | gold `[1,2,3,4,5]` vs ocr `[1,4]` | **FAIL** |
| 3. Not reversed | forward 0.946 vs reversed 0.009 | PASS |
| 4. Digit integrity | no Arabic-Indic ↔ ASCII substitution | PASS |

Gate 3 passing this decisively is the useful half: it answers the question the original
stalled OCR spike never reached. PaddleOCR-VL emits **logical-order** RTL, not visual
order, so the engine is not disqualified in principle.

Gate 2's failure has a single, precise cause — worth recording because the aggregate
fidelity score actively hides it. The engine did **not** drop the three headings and did
not scramble their order or digits. It read **teh marbuta `ة` (U+0629) as heh `ه`
(U+0647)** in three of five headings:

```
gold:  المادة 2: مجال التطبيق
ocr:   الماده 2: مجال التطبيق      <- one character
```

The arithmetic closes exactly: 2 correct markers + 3 heh-variants = the 5 gold headings.
A one-character-in-six substitution costs almost nothing on `SequenceMatcher` (hence
0.946) while breaking **exact-match citation extraction completely**. This is precisely
the class of defect gates 2–4 exist to catch and a similarity score alone cannot.

**Carried forward — this is not only an OCR problem.** `app/services/citations.py:46`
matches `re.compile(r'المادة\s+(\d+)')`, an exact literal, and there is no Arabic
orthographic normalization anywhere in the extraction path. Moroccan writers routinely
type `ه` for final `ة` by hand, so human-authored corpus text hits the identical miss —
independent of OCR. Normalizing orthographic variants (`ة`/`ه`, `ا`/`أ`/`إ`/`آ`,
`ى`/`ي`) before matching would fix both at once, but it changes a core, citation-sensitive
regex covered by `tests/test_citations.py` and was outside this plan's scope; it is left
as an open decision rather than a silent edit. **The gate was deliberately not loosened to
make the engine pass** — under today's exact-match extraction the failure is real.

Two harness defects fixed while getting this number, both of which had been burning a
full multi-minute model load per diagnosis: the script now writes raw OCR output to
`tests/data/ocr/<name>.<engine>.out.txt` **before** gating (so a failure can be diagnosed
without re-running the model), and reconfigures stdout/stderr to UTF-8 (a Windows console
is cp1252 and raised `UnicodeEncodeError` on any Arabic diagnostic, killing the run
*after* the model load).

### Ingestion repair: the 12 silently-dropped pages (2026-08-19)

An 80-page Arabic regulatory PDF (`arabic_test.pdf`, Bahraini urban-planning manual)
ingested as **28 chunks / 45,549 chars** and reported `status="ready"`. A tenant benchmark
then scored the tutor as hallucinating and non-Socratic. Both conclusions were wrong about
the cause: **12 of 80 pages had never entered the corpus at all**, and the upload reported
complete success.

**Root cause — inset raster images, not vector art.** Measured, after an initial
"vector-drawn content" hypothesis was checked and disproved (page 15 has only 4 path
operators; the genuinely-blank `profil_sst_maroc_fr.pdf` p138 has 36). The lost content is
embedded *images*: p15's CAD-layer table is a 2884×850 (2.45 MP) strip, p51's formula table
is 1120×1007 (1.13 MP). `_full_page_raster` rejects them because their aspect ratio doesn't
match the page's within `RASTER_ASPECT_TOLERANCE` — and since **every** OCR branch in
`classify_page` was gated on `has_full_page_raster`, such a page could only ever become
`NATIVE` (sparse caption text kept, table lost) or `EMPTY` (dropped entirely, and *not*
recorded in `unprocessed_pages`, hence the silent `ready`).

**Two signals added, both verified against all three pinned fixtures before adoption:**

| Rule | Condition | Fixture impact |
|---|---|---|
| A | `char_density < DENSITY_EMPTY_MAX` **and** `image_count >= 1` → `OCR_REQUIRED` | fires only on `profil` p136 and `guide_rh` p27 — both *already* `OCR_REQUIRED`. Zero change. `profil` p138 (0 images) correctly stays `EMPTY`. |
| B | `max_image_megapixels >= 1.0` → `OCR_PREFERRED` (native ∪ OCR **merge**) | flips **zero** `NATIVE` pages, even though `guide_rh`/`profil` contain 2.18/2.64 MP images (on already-OCR pages). `avis_cese` maxes at 0.15 MP. |

Merge, not replace, is load-bearing and was verified: `0.999600` exists **only** in p15's
native text layer; the CAD table exists **only** in its OCR output. Trusting either alone
loses real content.

**A second, independent bug found during the fix.** `ingest_jobs.process_source_file`
parsed the document once (for language/domain), then `ingest_file` parsed it **again from
scratch** — so every OCR page ran twice per upload. Worse than the wasted time: the two
parses hit *different* transient OCR failures, so `unprocessed_pages` (from parse #1)
reported pages 2,3 failing while the chunks actually stored (parse #2) were missing 1,
51–54, 62. Reported state and real state disagreed. `ingest_file` now accepts pre-parsed
`text`.

**Measured, before → after** (`company_efg`, RTX 4060 8GB, `ocr_paddle_engine="vl"`):

| | before | after |
|---|---|---|
| chunks / chars | 28 / 45,549 | **41 / 63,855** |
| pages present | 68 of 80 (51–54, 62 absent) | **74 of 80** |
| `unprocessed_pages` vs. reality | disagreed (7 reported, 12 actually missing) | **agree** (6 reported, 6 missing) |
| ground truth stored | p51 formulas absent, p15 table absent | **13/13** (Tier-1 hand-verified 10/10) |
| ground truth retrieved | TC-04 unanswerable | **4/4** |

**Triage is the efficiency win, not engine speed.** The classifier routes only **25 of 80**
pages to OCR (69% skip it), and `french_test.pdf` needs **zero** — it ingests in seconds.
Warm PaddleOCR-VL measures **~2 min/page** on this laptop, so `arabic_test.pdf` takes ~55
min; that is a one-time background cost (uploads are async, `Pending→Ready`) and the reason
a cheaper engine was not adopted is fidelity, not preference: in a three-engine bake-off,
classic PP-OCRv5 scored **0/5** on p51's formula constants (4.6 s/page) against VL's full
recovery. `PPStructureV3` remains unmeasured — its cold-start model download exceeded 10
minutes and crashed twice; `scripts/ocr_bakeoff.py` is committed to finish that comparison
later.

**Still open.** One page (p2) times out at the 300 s per-page ceiling under sustained GPU
load; it is now correctly *reported* rather than silently dropped. And
`reap_orphaned_processing` had no `tenant_id` predicate — every backend restart errored out
**every other tenant's** in-flight uploads; now scoped, with a regression test.
