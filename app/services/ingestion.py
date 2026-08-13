"""
Document Ingestion Service
─────────────────────────
Reads documents → strips formatting → chunks → embeds → stores in pgvector.

Supports .txt and .md files. Markdown syntax is stripped before chunking
so the embedding model processes clean content, not formatting noise.

Usage:
    python -m app.services.ingestion --source_dir ./raw/shared --tenant_id company_abc
"""

import re
import uuid
import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings


DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
BATCH_SIZE = 64

# raw/shared/<domain>/text/*.md -> "securite_physique" is the folder name,
# but "securite" is the Domain enum value used everywhere else in the app
# (app/models/schemas.py). Pre-existing mismatch, noted where it first bit
# (app/services/domain_context.py) -- normalized once, here, at ingest time.
DOMAIN_DIR_ALIASES = {"securite_physique": "securite"}

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

# Extensions that would need an OCR stage to reach markdown at all. Kept as
# a named constant, not a silent per-call guess, so there is one place to
# update once that stage is actually verified and enabled (see
# parse_document_to_markdown).
_OCR_REQUIRED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff"}


def parse_document_to_markdown(file_path: Path) -> str:
    """Document -> markdown: the ingestion pipeline's first stage.

    .md/.txt pass through unchanged -- this is the entire current corpus,
    so the demo path carries zero risk here.

    .pdf/image inputs need an OCR stage (baidu/Unlimited-OCR, MIT license,
    ~8GB VRAM in BF16 -- offline batch only, never in the request path,
    since it cannot co-reside with a resident tutor model on an 8GB card)
    to reach markdown at all. That model's Arabic-script fidelity (RTL
    ordering, ligature rendering -- tenant #1's corpus is Arabic-script
    Darija + French) has NOT been verified: the verification spike was
    infrastructure-blocked (a stalled multi-GB download), not run to a
    pass/fail result. Raising clearly here beats two worse alternatives:
    silently attempting `.read_text()` on binary PDF bytes (an opaque
    UnicodeDecodeError deep in the pipeline), or silently shipping an
    unverified OCR path into a citation-grounded system. Re-run the spike
    (harness: raw/shared/industrial/text/1.12_ar_procedure_consignation.md
    rendered to a PNG via headless Edge is a ready-made ground-truth
    image) and wire the verified call in here once it passes.
    """
    suffix = file_path.suffix.lower()
    if suffix in (".md", ".txt"):
        return file_path.read_text(encoding="utf-8")
    if suffix in _OCR_REQUIRED_EXTENSIONS:
        raise NotImplementedError(
            f"{file_path}: OCR document parsing is not enabled -- its "
            "Arabic-script fidelity has not been verified in this "
            "environment. Convert to .md/.txt first, or re-run the OCR "
            "spike and wire the verified path into parse_document_to_markdown."
        )
    raise ValueError(f"{file_path}: unsupported file type {suffix!r}")


def strip_markdown(text: str) -> str:
    """Remove markdown formatting artifacts so embedding model processes clean content."""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove markdown headers (##, ###, etc.)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    # Remove inline code backticks
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Remove blockquote markers
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
    # Remove horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Remove markdown table borders and alignment
    text = re.sub(r"^\|[-:| ]+\|\s*$", "", text, flags=re.MULTILINE)
    # Collapse remaining table rows into plain text
    text = re.sub(r"\|\s*", " ", text)
    # Remove link syntax, keep display text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def get_db_connection(database_url: str):
    """Connect to PostgreSQL."""
    return psycopg2.connect(database_url)


@lru_cache(maxsize=4)
def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    """Load and cache the embedding model.

    Was previously re-loading from disk on every call (docstring claimed
    caching that never existed) -- harmless for one-off ingestion scripts,
    but every chat/quiz request calls this via build_rag_context, so serving
    was paying a multi-second reload per message.
    """
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks using LangChain."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def split_by_headings(text: str) -> list[tuple[str, str]]:
    """Split raw (not yet markdown-stripped) text into (heading, body)
    pairs, one per `#`-`######` section. Any text before the first heading
    (title line, metadata block) is kept as its own section with an empty
    heading, so nothing is silently dropped.
    """
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("", text)]

    sections: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("", preamble))

    for i, m in enumerate(matches):
        heading = m.group(2).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        sections.append((heading, body))

    return sections


def chunk_document(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Heading-aware chunking: every chunk carries its own section
    heading(s) inline, so a chunk boundary can never separate a heading
    (where an article/section reference like "Art. 283" or "المادة 4"
    often lives, never repeated in the body prose) from its body. Without
    this, retrieval can return just the body half, and extract_citations
    (app/services/citations.py) never sees the reference -- citation
    injection is silently disabled for that answer.

    Short adjacent sections (this corpus's "المادة 1" .. "المادة 7"-style
    articles are frequently one or two sentences each) are PACKED together
    up to chunk_size rather than isolated one-per-chunk: measured against
    tests/data/retrieval_eval.jsonl, isolating every section regardless of
    size cost recall (many small, heading-heavy chunks embed less
    precisely than a few chunks with more surrounding context) -- see
    docs/architecture/data-and-retrieval.md's baseline. Packing keeps that
    cross-boundary context while still guaranteeing every packed section's
    own heading text sits next to its own body in the packed chunk's
    content (not just the first one's), because each piece is rendered
    "heading\n\nbody" *before* packing.

    Only a section long enough to exceed chunk_size on its own is split
    with chunk_text, each split piece re-prefixed with the heading so a
    long section's later pieces don't lose their reference either.

    Returns a list of {"content", "heading", "section_index"} dicts.
    `heading` on a packed chunk is its first section's heading (bookkeeping
    only -- every packed section's heading is already inline in `content`,
    which is what extract_citations and embedding actually see).
    """
    sections = split_by_headings(text)
    chunks: list[dict] = []

    buffer_parts: list[str] = []
    buffer_len = 0
    buffer_first_heading = ""
    buffer_first_index = 0

    def flush():
        nonlocal buffer_parts, buffer_len, buffer_first_heading
        if buffer_parts:
            chunks.append({
                "content": "\n\n".join(buffer_parts),
                "heading": buffer_first_heading,
                "section_index": buffer_first_index,
            })
        buffer_parts, buffer_len, buffer_first_heading = [], 0, ""

    for section_index, (heading, body) in enumerate(sections):
        clean_body = strip_markdown(body)
        if not clean_body:
            continue
        piece = f"{heading}\n\n{clean_body}" if heading else clean_body

        if len(piece) > chunk_size:
            # Long enough to need its own split -- flush whatever is
            # buffered first so it isn't glued onto this section, then
            # chunk this section alone, re-prefixing every piece.
            flush()
            for body_chunk in chunk_text(clean_body, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
                content = f"{heading}\n\n{body_chunk}" if heading else body_chunk
                chunks.append({"content": content, "heading": heading, "section_index": section_index})
            continue

        if buffer_parts and buffer_len + len(piece) + 2 > chunk_size:
            flush()

        if not buffer_parts:
            buffer_first_heading = heading
            buffer_first_index = section_index
        buffer_parts.append(piece)
        buffer_len += len(piece) + 2

    flush()
    return chunks


def detect_document_language(text: str) -> str:
    """Per-file language from Arabic-script character ratio -- same
    arithmetic app.services.llm.detect_query_language uses for a query,
    applied here to a whole document.

    Fixes ingest_directory previously stamping every file in a tree with
    one language default ("fr"), which silently stored Arabic-script files
    (e.g. 1.11_ar_code_travail_salama.md) as French. Committed
    language-affinity retrieval (ADR 0002 decision 5) is meaningless until
    each document's stored language is actually correct.
    """
    arabic = sum(1 for c in text if "؀" <= c <= "ۿ")
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    return "ar" if arabic > latin else "fr"


def detect_document_domain(file_path: Path, source_dir: Path) -> Optional[str]:
    """Best-effort domain from a file's path relative to source_dir,
    assuming the raw/shared/<domain>/text/*.md convention. Returns None if
    the path doesn't fit that shape -- callers should pass domain
    explicitly in that case rather than guess.
    """
    try:
        rel_parts = file_path.resolve().relative_to(source_dir.resolve()).parts
    except ValueError:
        return None
    if len(rel_parts) >= 3 and rel_parts[-2] == "text":
        domain = rel_parts[-3]
        return DOMAIN_DIR_ALIASES.get(domain, domain)
    return None


def embed_chunks(
    model: SentenceTransformer,
    chunks: list[str],
) -> list[list[float]]:
    """Generate embeddings for a list of text chunks."""
    embeddings = model.encode(chunks, show_progress_bar=False, batch_size=BATCH_SIZE)
    return embeddings.tolist()


def insert_documents(
    conn,
    tenant_id: str,
    source_name: str,
    source_type: str,
    language: str,
    chunks: list[str],
    embeddings: list[list[float]],
    metadata_list: Optional[list[dict]] = None,
    domain: Optional[str] = None,
    ingest_batch_id: Optional[str] = None,
) -> int:
    """Insert chunked documents with embeddings into pgvector."""
    cursor = conn.conn.cursor() if hasattr(conn, "conn") else conn.cursor()

    if not metadata_list:
        metadata_list = [{} for _ in chunks]

    rows = []
    for chunk, embedding, meta in zip(chunks, embeddings, metadata_list):
        doc_id = str(uuid.uuid4())
        rows.append((
            doc_id,
            tenant_id,
            chunk,
            source_name,
            source_type,
            domain,
            ingest_batch_id,
            language,
            str(embedding),
            json.dumps(meta),
        ))

    insert_query = """
        INSERT INTO documents
            (id, tenant_id, content, source_name, source_type, domain, ingest_batch_id, language, embedding, metadata)
        VALUES %s
    """

    execute_values(
        cursor,
        insert_query,
        rows,
        template="(%s::uuid, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s::jsonb)",
        page_size=BATCH_SIZE,
    )

    conn.commit()
    return len(rows)


def ingest_text(
    text: str,
    source_name: str,
    tenant_id: str,
    source_type: str = "text",
    language: str = "fr",
    domain: Optional[str] = None,
    ingest_batch_id: Optional[str] = None,
    metadata: Optional[dict] = None,
    database_url: Optional[str] = None,
    embedding_model: Optional[SentenceTransformer] = None,
) -> dict:
    """Ingest a single text document: split into heading-aware sections →
    chunk each → embed → store. See chunk_document() for why chunking is
    heading-aware rather than a flat strip-then-split pass."""
    db_url = database_url or get_settings().database_url

    if embedding_model is None:
        embedding_model = load_embedding_model()

    doc_chunks = chunk_document(text)
    if not doc_chunks:
        return {"chunks_created": 0, "source_name": source_name, "status": "empty"}

    chunk_texts = [c["content"] for c in doc_chunks]
    embeddings = embed_chunks(embedding_model, chunk_texts)

    base_meta = metadata.copy() if metadata else {}
    base_meta.update({
        "source_name": source_name,
        "source_type": source_type,
        "language": language,
    })
    metadata_list = [
        {**base_meta, "heading": c["heading"], "section_index": c["section_index"]}
        for c in doc_chunks
    ]

    conn = get_db_connection(db_url)
    try:
        count = insert_documents(
            conn, tenant_id, source_name, source_type, language, chunk_texts, embeddings,
            metadata_list, domain=domain, ingest_batch_id=ingest_batch_id,
        )
        return {"chunks_created": count, "source_name": source_name, "status": "success"}
    finally:
        conn.close()


def ingest_file(
    file_path: str,
    tenant_id: str,
    source_type: str = "text",
    language: Optional[str] = None,
    domain: Optional[str] = None,
    ingest_batch_id: Optional[str] = None,
    metadata: Optional[dict] = None,
    database_url: Optional[str] = None,
    embedding_model: Optional[SentenceTransformer] = None,
) -> dict:
    """Ingest a single file. `language` omitted -> auto-detected per file
    from its own content (detect_document_language), not a caller-wide
    default."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    text = parse_document_to_markdown(path)
    resolved_language = language or detect_document_language(text)
    return ingest_text(
        text=text,
        source_name=path.name,
        tenant_id=tenant_id,
        source_type=source_type,
        language=resolved_language,
        domain=domain,
        ingest_batch_id=ingest_batch_id,
        metadata=metadata,
        database_url=database_url,
        embedding_model=embedding_model,
    )


def ingest_directory(
    source_dir: str,
    tenant_id: str,
    language: Optional[str] = None,
    domain: Optional[str] = None,
    database_url: Optional[str] = None,
) -> list[dict]:
    """Ingest all .txt and .md files from a directory.

    `language` is optional and now per-file by default (Arabic-script
    files are no longer silently stored as French because a caller passed
    one tree-wide default) -- pass it explicitly only to force every file
    in the tree to one language, overriding detection.

    `domain` resolves per file, in order, and is NEVER left as None:
    1. The raw/shared/<domain>/text/ path convention (detect_document_domain).
    2. This explicit `domain` argument (the --domain CLI flag), when the
       file doesn't fit that path shape.
    3. settings.default_domain, logged as an [INGEST WARNING] so a
       mis-filed document is a loud, audited fallback rather than a silent
       guess.
    A domain=NULL row is invisible to every domain-filtered query
    (`WHERE domain = %s` never matches NULL) -- confirmed live during the
    2026-08-11 E2E audit, where two untagged files had to be backfilled by
    hand after the fact. Ingestion must never reproduce that.

    One ingest_batch_id is stamped across the whole run so a later
    corpus_version check (Stage 3) can tell "the corpus changed" from
    "this row predates domain/language tracking".
    """
    source_path = Path(source_dir)
    if not source_path.is_dir():
        raise NotADirectoryError(f"Directory not found: {source_dir}")

    embedding_model = load_embedding_model()
    batch_id = uuid.uuid4().hex
    results = []

    for file_path in sorted(source_path.glob("**/*.md")) + sorted(source_path.glob("**/*.txt")):
        rel_path = file_path.relative_to(source_path)
        file_type = "markdown" if file_path.suffix == ".md" else "text"

        file_domain = detect_document_domain(file_path, source_path)
        if file_domain is None and domain is not None:
            file_domain = domain
        if file_domain is None:
            file_domain = get_settings().default_domain
            print(
                f"[INGEST WARNING] No domain provided for {rel_path}. "
                f"Falling back to default domain: {file_domain!r}"
            )
        print(f"Ingesting [{file_type}] domain={file_domain}: {rel_path}")

        result = ingest_file(
            file_path=str(file_path),
            tenant_id=tenant_id,
            source_type=file_type,
            language=language,
            domain=file_domain,
            ingest_batch_id=batch_id,
            metadata={"source_dir": source_dir, "relative_path": str(rel_path)},
            database_url=database_url,
            embedding_model=embedding_model,
        )
        results.append(result)
        print(f"  -> {result['chunks_created']} chunks created")

    total_chunks = sum(r["chunks_created"] for r in results)
    print(f"\nDone. {len(results)} files ingested, {total_chunks} total chunks. batch={batch_id}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest documents into pgvector")
    parser.add_argument("--source_dir", required=True, help="Directory with .txt or .md files")
    parser.add_argument("--tenant_id", required=True, help="Tenant identifier")
    parser.add_argument(
        "--language", default=None,
        help="Force this language for every file. Omit to auto-detect per file (default).",
    )
    parser.add_argument(
        "--domain", default=None,
        help=(
            "Fallback domain for files that don't fit the "
            "raw/shared/<domain>/text/ path convention. Omit to fall back "
            "further to settings.default_domain (logged as an "
            "[INGEST WARNING] when that happens)."
        ),
    )
    parser.add_argument("--database_url", default=None, help="PostgreSQL connection URL")
    args = parser.parse_args()

    results = ingest_directory(
        source_dir=args.source_dir,
        tenant_id=args.tenant_id,
        language=args.language,
        domain=args.domain,
        database_url=args.database_url,
    )
