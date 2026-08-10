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

[ingestion.py](../../app/services/ingestion.py): `.txt`/`.md` files, markdown stripped
before chunking so the embedding model sees clean content.

| Setting | Value |
|---|---|
| Chunker | `RecursiveCharacterTextSplitter` |
| Chunk size | 400 chars |
| Chunk overlap | 50 chars |
| Embedding model | `paraphrase-multilingual-MiniLM-L12-v2` (384-dim) |
| Batch size | 64 |

None of these were benchmarked against this corpus — they're framework defaults. 400
chars is small for legal prose; an article can split across chunks, which is a risk to
the verbatim-citation behavior the whole grounding design depends on.

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

**Detail & rationale:** `../../resurrection.md` §2, `../../LOCKEDIN_PLAN.md`.
