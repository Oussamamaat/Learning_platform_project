"""
Document Ingestion Service
─────────────────────────
Reads documents → strips formatting → chunks → embeds → stores in pgvector.

Supports .txt, .md, .pdf, .docx, .pptx, .xlsx, .csv, and images (via OCR --
see app.services.ocr). Every format is normalized to markdown BEFORE
chunking -- see parse_document_to_markdown -- so a single heading-aware
chunker (chunk_document) and a single citation-preservation guarantee
cover all of them, rather than one parser per format needing its own
chunking logic.

Usage:
    python -m app.services.ingestion --source_dir ./raw/shared --tenant_id company_abc
"""

import csv
import io
import logging
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

logger = logging.getLogger(__name__)


# Reads settings.embedding_model at import time (not hardcoded) so
# scripts/migrate_to_bge_m3.py's rollback path (set EMBEDDING_MODEL back to
# the MiniLM name via env/.env) works without a code change. Does NOT
# affect app.services.generate_training_data.DEDUP_MODEL, which is a
# separate, deliberately still-MiniLM dataset-dedup threshold -- see
# app/config.py's embedding_model docstring.
DEFAULT_EMBEDDING_MODEL = get_settings().embedding_model
# 2000/250 (bge-m3, 8192-token window) as of 2026-08-13 -- was 400/50,
# sized against the OLD embedding model's actual 128-token max_seq_length
# (paraphrase-multilingual-MiniLM-L12-v2's own sentence_bert_config.json),
# not an arbitrary default. Raising this without also raising
# max_context_length (app/services/retrieval.py) and setting Ollama's
# num_ctx (app/services/llm.py) silently truncates citations -- see
# retrieval._build_context's docstring-adjacent comment for the concrete
# failure mode this caused.
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 250
BATCH_SIZE = 64

# raw/shared/<domain>/text/*.md -> "securite_physique" is the folder name,
# but "securite" is the Domain enum value used everywhere else in the app
# (app/models/schemas.py). Pre-existing mismatch, noted where it first bit
# (app/services/domain_context.py) -- normalized once, here, at ingest time.
DOMAIN_DIR_ALIASES = {"securite_physique": "securite"}

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

# Legacy binary Office formats: rejected outright rather than attempted --
# no parser here reads the old OLE-based .doc/.ppt/.xls container format,
# and silently misreading one as UTF-8 text would be worse than a clear
# "convert first" error.
LEGACY_BINARY_FORMATS = {".doc": ".docx", ".ppt": ".pptx", ".xls": ".xlsx"}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}
SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".docx", ".pptx", ".xlsx", ".csv"} | _IMAGE_EXTENSIONS

# Rasterization DPI for pages that app.services.pdf_classify.classify_page
# sends to OCR (app.services.ocr). ~200 DPI balances legibility for small
# print (this corpus's article numbers/footnotes) against OCR call latency.
OCR_RENDER_DPI = 200
# Hard cap on rows rendered from one spreadsheet sheet -- silent truncation
# is unacceptable in a citation system, so a sheet over this size gets an
# explicit "[... N more rows omitted ...]" marker rather than either
# hanging on a huge sheet or dropping rows with no trace.
XLSX_MAX_ROWS = 5000


def _parse_text(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8")


def _parse_docx(file_path: Path) -> str:
    """`.docx` -> markdown. Word's own heading styles ("Heading 1".."Heading
    6", "Title") map directly to `#`.."######" so split_by_headings sees
    real document structure instead of a flat, unheaded text dump -- this
    is what lets the citation-preservation guarantee (chunk_document's
    docstring) extend to uploaded Word documents for free.
    """
    import docx

    document = docx.Document(str(file_path))
    lines: list[str] = []
    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style_name = (para.style.name or "").strip() if para.style else ""
        heading_match = re.match(r"^Heading (\d)$", style_name)
        if style_name == "Title":
            lines.append(f"# {text}")
        elif heading_match:
            level = min(int(heading_match.group(1)), 6)
            lines.append(f"{'#' * level} {text}")
        else:
            lines.append(text)

    for table in document.tables:
        lines.append("")
        for row in table.rows:
            cells = [c.text.strip().replace("\n", " ") for c in row.cells]
            lines.append("| " + " | ".join(cells) + " |")

    return "\n\n".join(lines)


def _parse_pptx(file_path: Path) -> str:
    """`.pptx` -> markdown. Slide title -> `## `, other text-frame shapes
    as body paragraphs, speaker notes appended as a blockquote (kept for
    embedding -- notes often carry the actual explanatory content a slide's
    bullet points only gesture at).
    """
    import pptx

    presentation = pptx.Presentation(str(file_path))
    sections: list[str] = []
    for i, slide in enumerate(presentation.slides, start=1):
        title = None
        body_lines: list[str] = []
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            text = shape.text_frame.text.strip()
            if not text:
                continue
            if shape == slide.shapes.title:
                title = text
            else:
                body_lines.append(text)

        heading = f"## {title}" if title else f"## Slide {i}"
        section = [heading]
        section.extend(body_lines)

        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                quoted = "\n".join(f"> {line}" for line in notes.splitlines())
                section.append(quoted)

        sections.append("\n\n".join(section))

    return "\n\n".join(sections)


def _render_row(cells: list[str]) -> str:
    return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"


def _parse_xlsx(file_path: Path) -> str:
    """`.xlsx` -> markdown. `read_only=True, data_only=True` streams the
    file rather than loading the whole workbook, and reads each formula
    cell's last-cached value instead of the formula text. One `## <sheet
    name>` section per sheet, rows rendered as a markdown table, capped at
    XLSX_MAX_ROWS with an explicit truncation note.
    """
    import openpyxl

    workbook = openpyxl.load_workbook(str(file_path), read_only=True, data_only=True)
    sections: list[str] = []
    try:
        for sheet in workbook.worksheets:
            rows_rendered = 0
            lines = [f"## {sheet.title}"]
            for row in sheet.iter_rows(values_only=True):
                if all(cell is None for cell in row):
                    continue
                if rows_rendered >= XLSX_MAX_ROWS:
                    lines.append(f"[... remaining rows omitted after {XLSX_MAX_ROWS} ...]")
                    break
                cells = ["" if c is None else str(c) for c in row]
                lines.append(_render_row(cells))
                rows_rendered += 1
            if rows_rendered:
                sections.append("\n".join(lines))
    finally:
        workbook.close()

    return "\n\n".join(sections)


def _parse_csv(file_path: Path) -> str:
    """`.csv` -> markdown. `utf-8-sig` strips a BOM if present; delimiter
    sniffed rather than assumed comma, since this corpus's genre (exported
    regulatory/compliance spreadsheets) commonly uses `;` in French locale
    exports.
    """
    raw = file_path.read_bytes().decode("utf-8-sig")
    try:
        dialect = csv.Sniffer().sniff(raw[:2048])
    except csv.Error:
        dialect = csv.excel

    reader = csv.reader(io.StringIO(raw), dialect)
    lines = [f"# {file_path.stem}"]
    for i, row in enumerate(reader):
        if not any(cell.strip() for cell in row):
            continue
        if i >= XLSX_MAX_ROWS:
            lines.append(f"[... remaining rows omitted after {XLSX_MAX_ROWS} ...]")
            break
        lines.append(_render_row(row))

    return "\n".join(lines)


def _parse_image(file_path: Path) -> str:
    from app.services.ocr import get_ocr_engine

    engine = get_ocr_engine()
    return engine.image_to_markdown(file_path.read_bytes())


def _parse_pdf(file_path: Path, unprocessed_pages: Optional[list] = None) -> str:
    """`.pdf` -> markdown. The text-vs-OCR decision is made PER PAGE, not
    per document: a digitally authored regulation with a scanned annex
    appended is the norm in this corpus's genre, and a document-level "does
    it have a text layer?" check would silently drop every scanned page of
    an otherwise-digital file. Each page is classified by
    app.services.pdf_classify.classify_page (font/image/density signals
    measured against real Moroccan PDFs, not a raw character-count floor --
    see that module's docstring for why) into one of four strategies:

      NATIVE        -- use the embedded text as-is.
      EMPTY         -- nothing readable; emit an empty section, never call
                       OCR (a blank page must not fail an otherwise fully
                       readable document).
      OCR_PREFERRED -- embedded text exists but is degraded (e.g. a diagram
                       whose layout the raw text order loses); OCR improves
                       it, but OcrUnavailableError falls back to the native
                       text here rather than failing the document.
      OCR_REQUIRED  -- embedded text is unusable; if OCR is unavailable,
                       this ONE page is skipped (recorded into
                       `unprocessed_pages`, if given, as {"page", "reason"})
                       and parsing continues -- a real 80-page administrative
                       guide measured 4 OCR_REQUIRED pages against 66 NATIVE
                       ones, and failing the whole document over 5% of its
                       pages would discard the other 95% for no reason. The
                       one exception: if EVERY page that could contribute
                       text failed this way and nothing else in the document
                       produced any text either, that is not a partial
                       document, it is an unreadable one -- the original
                       OcrUnavailableError is re-raised so
                       app.services.ingest_jobs.process_source_file can
                       still fail loudly and actionably in that case.

    Emits `## Page N` headings so page numbers survive into chunks as a
    citation affordance.
    """
    from pypdf import PdfReader

    from app.services.ocr import OcrUnavailableError
    from app.services.pdf_classify import PageStrategy, classify_page

    reader = PdfReader(str(file_path))
    sections: list[str] = []
    local_unprocessed: list[dict] = []
    any_text = False
    last_error: Optional[OcrUnavailableError] = None

    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        decision = classify_page(page, text=text)

        if decision.strategy == PageStrategy.EMPTY:
            text = ""
        elif decision.strategy == PageStrategy.OCR_REQUIRED:
            try:
                text = _ocr_pdf_page(file_path, page_num - 1).strip()
            except OcrUnavailableError as e:
                # `e` -- the OCR call's OWN failure reason (subprocess error,
                # timeout, missing engine) -- not decision.reason (why this
                # page was classified OCR_REQUIRED in the first place). Those
                # are two different questions; logging decision.reason here
                # made every OCR failure look identical to a classification
                # note ("no font resources"), with no way to tell "OCR is
                # off" from "OCR ran and crashed" from a log line alone.
                logger.warning(
                    "page=%d of %s: OCR required (%s), but the OCR attempt "
                    "itself failed: %s -- page skipped, rest of document "
                    "continues",
                    page_num, file_path.name, decision.reason, e,
                )
                local_unprocessed.append({
                    "page": page_num, "reason": "ocr_required", "detail": str(e),
                })
                last_error = e
                text = ""
        elif decision.strategy == PageStrategy.OCR_PREFERRED:
            try:
                text = _ocr_pdf_page(file_path, page_num - 1).strip()
            except OcrUnavailableError:
                logger.info(
                    "page=%d of %s: OCR preferred but unavailable (%s) -- "
                    "falling back to embedded text",
                    page_num, file_path.name, decision.reason,
                )
        # else NATIVE: text is already the embedded extraction.

        if text:
            any_text = True

        if decision.strategy != PageStrategy.NATIVE:
            logger.info(
                "page=%d of %s: %s -> %s",
                page_num, file_path.name, decision.strategy.value, decision.reason,
            )

        sections.append(f"## Page {page_num}\n\n{text}")

    if not any_text and last_error is not None:
        # Every page that could have produced text failed identically --
        # this isn't a partial document, it's an entirely unreadable one.
        # Fail loudly rather than silently ingest something empty that
        # would look like a successful upload with nothing retrievable.
        raise last_error

    if unprocessed_pages is not None:
        unprocessed_pages.extend(local_unprocessed)

    return "\n\n".join(sections)


def _ocr_pdf_page(file_path: Path, page_index: int) -> str:
    import pypdfium2 as pdfium

    from app.services.ocr import get_ocr_engine

    pdf = pdfium.PdfDocument(str(file_path))
    try:
        page = pdf[page_index]
        scale = OCR_RENDER_DPI / 72
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        return get_ocr_engine().image_to_markdown(buf.getvalue())
    finally:
        pdf.close()


_PARSERS = {
    ".md": _parse_text,
    ".txt": _parse_text,
    ".docx": _parse_docx,
    ".pptx": _parse_pptx,
    ".xlsx": _parse_xlsx,
    ".csv": _parse_csv,
}


def parse_document_to_markdown(file_path: Path, unprocessed_pages: Optional[list] = None) -> str:
    """Document -> markdown: the ingestion pipeline's first stage. Every
    format is normalized to markdown WITH real `#`/`##` headings where the
    source format has structure (Word/PowerPoint styles, PDF page breaks) --
    split_by_headings only recognizes `^#{1,6}\\s+`, so a parser returning
    flat prose would silently drop the citation-preservation guarantee
    (chunk_document's docstring) for that format.

    .md/.txt pass through unchanged. .pdf/images may invoke OCR
    (app.services.ocr) per page/whole-image when no usable embedded text
    layer exists. For .pdf, a page that needs OCR and can't get it is
    skipped rather than failing the whole document -- pass a list via
    `unprocessed_pages` to receive {"page", "reason"} entries for each one
    (see _parse_pdf's docstring for when this instead raises
    OcrUnavailableError: only when the WHOLE document produced no text).
    A whole-image upload (.png/.jpg/...) has no such partial case -- OCR
    failing there fails the only "page" there is, so OcrUnavailableError
    propagates uncaught, exactly as before.

    .doc/.ppt/.xls (legacy binary Office formats) are rejected outright --
    no parser here reads the old OLE container format, and misreading one
    as text would be worse than a clear conversion instruction.

    Every format's output passes through
    app.services.citations.normalize_arabic_text (NFKC + strip tatweel/
    harakat/bidi marks) before returning -- a PDF's table text can extract
    as Unicode Presentation Forms B glyphs instead of standard Arabic
    letters (measured on a real 80-page administrative guide: a
    presentation-form spelling of a teh-marbuta-ending word contained no
    teh marbuta codepoint at all, so citation regexes failed outright on
    it), and this is the single choke point every parser's output already
    passes through. Lossless and a no-op on non-Arabic/already-clean text,
    so applying it unconditionally is never harmful -- see that function's
    docstring for why this is safe to store/embed, unlike the lossy,
    comparison-only fold_arabic in the same module.
    """
    suffix = file_path.suffix.lower()
    if suffix in LEGACY_BINARY_FORMATS:
        raise ValueError(
            f"{file_path.name}: legacy binary {suffix} is not supported. "
            f"Re-save it as {LEGACY_BINARY_FORMATS[suffix]} and upload again."
        )
    if suffix == ".pdf":
        raw = _parse_pdf(file_path, unprocessed_pages=unprocessed_pages)
    elif suffix in _IMAGE_EXTENSIONS:
        raw = _parse_image(file_path)
    else:
        parser = _PARSERS.get(suffix)
        if parser is None:
            raise ValueError(f"{file_path}: unsupported file type {suffix!r}")
        raw = parser(file_path)

    from app.services.citations import normalize_arabic_text
    return normalize_arabic_text(raw)


def strip_markdown(text: str) -> str:
    """Remove markdown formatting artifacts so embedding model processes clean content.

    NOTE: kept byte-for-byte in sync with
    scripts/verify_ocr_arabic.py::_strip_markdown_standalone (that copy
    exists for OCR-only venvs that can't import this module's heavy
    dependencies) -- change both together.
    """
    # Turn HTML structural boundaries into whitespace BEFORE stripping tags.
    # The generic `<[^>]+>` strip below deletes tags with no replacement,
    # which FUSES the text on either side: measured on a real scanned page,
    # PaddleOCR-VL returns tables as HTML, and
    # "<td>الترخيص</td><td>450 متر</td>" collapsed to "الترخيص450 متر" --
    # two unrelated cells welded into one nonsense token, embedded and
    # retrieved that way. Markdown tables never had this problem (the
    # `\|\s*` rule further down already separates their cells); only the
    # HTML path did, which is exactly the path OCR'd tables arrive on.
    text = re.sub(r"</\s*(?:td|th)\s*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"</\s*(?:tr|p|div|li|h[1-6])\s*>|<\s*br\s*/?\s*>", "\n", text, flags=re.IGNORECASE
    )
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

    Asserts the loaded model's actual dimension matches
    settings.embedding_dim -- documents.embedding is a fixed-width
    pgvector column (Vector(settings.embedding_dim)), so a mismatch here
    would otherwise surface as a cryptic pgvector INSERT failure deep in a
    batch instead of a clear error at the first load. Loud at boot beats
    silent corruption mid-ingest.
    """
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    actual_dim = model.get_sentence_embedding_dimension()
    print(f"Model loaded. Embedding dimension: {actual_dim}")
    expected_dim = get_settings().embedding_dim
    if actual_dim != expected_dim:
        raise RuntimeError(
            f"Embedding model {model_name!r} produces {actual_dim}-dim vectors, "
            f"but settings.embedding_dim is {expected_dim}. The documents.embedding "
            f"column is a fixed-width vector({expected_dim}) -- update settings."
            f"embedding_dim (and re-run scripts/migrate_to_bge_m3.py-style migration) "
            f"or load a matching model."
        )
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
    source_file_id: Optional[str] = None,
) -> int:
    """Insert chunked documents with embeddings into pgvector.

    source_file_id: NULL (default) means "belongs to the tenant's always-on
    global corpus" -- every raw/shared CLI ingest leaves this unset, exactly
    today's behaviour. A uuid string ties every chunk back to a
    source_files row (app/routers/ingest.py's upload pipeline), which is
    what lets app.services.search's source_ids filter and the
    enable/disable toggle work per-upload.
    """
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
            source_file_id,
        ))

    insert_query = """
        INSERT INTO documents
            (id, tenant_id, content, source_name, source_type, domain, ingest_batch_id, language, embedding, metadata, source_file_id)
        VALUES %s
    """

    execute_values(
        cursor,
        insert_query,
        rows,
        template="(%s::uuid, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s::jsonb, %s::uuid)",
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
    source_file_id: Optional[str] = None,
) -> dict:
    """Ingest a single text document: split into heading-aware sections →
    chunk each → embed → store. See chunk_document() for why chunking is
    heading-aware rather than a flat strip-then-split pass.

    source_file_id: forwarded to insert_documents -- None (default) for
    every raw/shared CLI ingest; a uuid string for an app/routers/ingest.py
    upload."""
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
            source_file_id=source_file_id,
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
    source_file_id: Optional[str] = None,
    source_name: Optional[str] = None,
) -> dict:
    """Ingest a single file. `language` omitted -> auto-detected per file
    from its own content (detect_document_language), not a caller-wide
    default.

    `source_name` omitted -> path.name (the CLI's raw/shared/<domain>/
    text/*.md convention, where the on-disk filename already IS the
    citation-worthy name). Explicit for uploads (app/routers/ingest.py):
    an uploaded file's stored_path is a server-generated
    <uuid>.<ext> (deliberately, to avoid path traversal/collisions -- see
    that router), which would otherwise become the citation shown to
    users (e.g. "3393e3ef014f4e1faf7f014aac6705ab.md" instead of
    "politique_consignation_nord.md") -- caught live by probe_upload_e2e.py.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    text = parse_document_to_markdown(path)
    resolved_language = language or detect_document_language(text)
    return ingest_text(
        text=text,
        source_name=source_name or path.name,
        tenant_id=tenant_id,
        source_type=source_type,
        language=resolved_language,
        domain=domain,
        ingest_batch_id=ingest_batch_id,
        metadata=metadata,
        database_url=database_url,
        embedding_model=embedding_model,
        source_file_id=source_file_id,
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
