from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    app_name: str = "IBLOG AI Assistant"
    app_version: str = "0.1.0"
    database_url: str = "postgresql://assistant:changeme@localhost:5432/iblog_assistant"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "IBLOG_TUTOR:latest"
    ollama_model_fr: str = "iblog-tutor-fr:latest"
    default_tenant_id: str = "company_abc"
    default_user_id: str = "default_user"
    # Tier-3 fallback for app.services.routing's domain router when tier 1
    # (page context) is absent and tier 2 (retrieval-as-router) finds no
    # candidate clearing the similarity threshold. Also ingestion's
    # last-resort domain when neither the raw/shared/<domain>/text/ path
    # convention nor an explicit --domain flag applies (app/services/
    # ingestion.py) -- ingestion never writes domain=NULL (rows with a NULL
    # domain are invisible to every domain-filtered query; see the 2026-08-11
    # E2E audit finding this caused for the untagged e2e_intake files).
    default_domain: str = "industrial"
    # "disk" | "pgvector". Locked to "pgvector" 2026-08-10 after live
    # verification against real Postgres + the real ingested corpus (see
    # docs/architecture/data-and-retrieval.md's pgvector section): domain
    # isolation and language affinity confirmed correct, similarity
    # threshold re-tuned from an unbenchmarked 0.4 to a measured 0.15.
    # app/routers/chat.py's _retrieve_context is the only call site that
    # reads this -- it falls back to "disk" on any pgvector exception, so a
    # Postgres hiccup degrades chat rather than crashing it. Override via
    # the RETRIEVAL_BACKEND env var (standard pydantic-settings behavior,
    # no extra code needed) to force "disk" if Postgres is unavailable.
    # Delete this flag after the demo rather than keeping it around.
    retrieval_backend: str = "pgvector"

    # Embedding model. Swapped 2026-08-13 from paraphrase-multilingual-
    # MiniLM-L12-v2 (384-dim) to bge-m3 (1024-dim) -- the MiniLM model's
    # own sentence_bert_config.json caps it at max_seq_length=128 tokens,
    # which is what the old 400-char CHUNK_SIZE was actually sized against,
    # not an arbitrary default. Real uploaded PDFs/slides need more room
    # than that. embedding_dim MUST match documents.embedding's declared
    # vector() width -- app.services.ingestion.load_embedding_model asserts
    # this at load time so a mismatch fails loudly at boot, not silently
    # mid-ingest. Changing either value requires running
    # scripts/migrate_to_bge_m3.py (or its equivalent for a future swap),
    # never just editing this file. app.services.generate_training_data's
    # DEDUP_MODEL is a SEPARATE, deliberately-unrelated MiniLM usage
    # (dataset dedup, tuned to that model's score distribution) -- do not
    # point it at this setting.
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024

    # Cosine-similarity floor for a retrieved chunk to be usable
    # (app.services.search / app.services.retrieval) and for a chunk to
    # count toward app.services.routing's domain vote, respectively.
    # Re-swept 2026-08-13 for bge-m3 (single-process sweep over
    # tests/data/retrieval_eval.jsonl, pgvector backend, language-affinity
    # on): recall@4 stays a PERFECT 1.000 all the way through threshold
    # 0.4, where empty_context_rate_ood (out-of-domain refusal) jumps from
    # 0.2 to 0.8 -- the best point on the curve, since it's the last one
    # with zero recall cost. Above 0.4, recall starts falling (0.8 at 0.5,
    # 0.51 at 0.6) faster than refusal improves (0.8 -> 1.0). This is a
    # much cleaner separation than the old MiniLM value it replaces (0.15,
    # where recall was already trading off against refusal from a much
    # lower starting threshold -- see docs/architecture/data-and-
    # retrieval.md's historical table). Kept as two separate settings
    # because they answer two different questions ("is this chunk usable"
    # vs. "does this chunk get a vote in domain routing"), even though the
    # sweep landed them at the same value again.
    similarity_threshold: float = 0.4
    domain_vote_threshold: float = 0.4

    # OCR: "none" | "tesseract" | "paddleocr" | "unlimited_ocr".
    #
    # "paddleocr" since 2026-08-18: scripts/verify_ocr_arabic.py now passes
    # all four gates against a LIVE PaddleOCR-VL run (fidelity 0.945, all 5
    # المادة markers in ascending order, not reversed, digit integrity OK).
    # The gate previously failed only because PaddleOCR reads teh marbuta
    # (ة) as heh (ه) -- visible in the live output as "المسطره" for
    # "المسطرة" -- which the Arabic orthographic normalization added to
    # app/services/citations.py (fold_arabic / arabic_variant_pattern) now
    # absorbs. No engine change was needed; the defect was never the
    # blocker, the missing normalization was.
    #
    # Requires .ocr_venv (see app.services.ocr.PaddleOcrEngine, which talks
    # to scripts/ocr_worker_resident.py -- a persistent subprocess under
    # settings.ocr_venv_python, started once and reused for every page --
    # rather than importing paddleocr into this process). If that venv is
    # absent, OCR raises OcrUnavailableError with an actionable message and
    # -- since 2026-08-18 -- a PDF's other pages still ingest, with the
    # unreadable ones recorded in source_files.unprocessed_pages and the row
    # marked status='partial' rather than failing the whole document.
    #
    # Measured on this laptop (RTX 4060 8GB) under the OLD per-page-
    # subprocess design (scripts/ocr_paddleocr_worker.py, retired once the
    # resident worker replaced it): ~80-200s for one page cold, including
    # model load EVERY call. The same page took ~17 MINUTES before
    # paddlepaddle-gpu replaced a mistakenly-installed CPU-only
    # paddlepaddle wheel -- if OCR is ever inexplicably slow again, check
    # `paddle.device.is_compiled_with_cuda()` in .ocr_venv first.
    ocr_engine: str = "paddleocr"
    # Only consulted by UnlimitedOcrEngine's in-process model (load-per-job,
    # free-after by default: del model; torch.cuda.empty_cache() -- so it
    # never has to co-reside with the resident tutor model on an 8GB card).
    # Set True only on a deploy box with enough VRAM to keep it resident too.
    # PaddleOcrEngine does NOT read this: its worker subprocess
    # (scripts/ocr_worker_resident.py) is unconditionally kept resident for
    # this app process's whole lifetime once started -- that subprocess's
    # own VRAM use, not this app process's, is what must fit alongside the
    # tutor model; see PaddleOcrEngine's docstring.
    ocr_keep_resident: bool = False
    # Path to the DEDICATED venv's interpreter (paddlepaddle/paddleocr pin
    # CUDA/torch versions that conflict with this app's own torch pin -- see
    # app/services/ocr.py's PaddleOcrEngine docstring) that
    # scripts/ocr_worker_resident.py actually runs under, as a persistent
    # subprocess started once and reused for every page. PaddleOcrEngine
    # launches that script via this interpreter rather than importing
    # paddleocr into this process. Only read when ocr_engine="paddleocr".
    ocr_venv_python: str = "./.ocr_venv/Scripts/python.exe"
    # Which pipeline app.services.ocr.PaddleOcrEngine's resident worker
    # (scripts/ocr_worker_resident.py) runs: "vl" (PaddleOCR-VL, the 3B
    # vision-language pipeline -- highest fidelity, slowest, the only one
    # with a proven ground-truth recovery on this corpus as of the
    # scripts/ocr_bakeoff.py run this default was set from), "structure"
    # (PPStructureV3, layout+table recognition -- lighter), or "classic"
    # (PaddleOCR/PP-OCRv5 -- lightest, no table/layout structure at all).
    # See scripts/ocr_bakeoff.py's docstring for the measured wall-clock
    # and ground-truth-token-survival numbers each was chosen from.
    ocr_paddle_engine: str = "vl"

    # Tenant document uploads (app/routers/ingest.py).
    upload_dir: str = "./data/uploads"
    max_upload_bytes: int = 25 * 1024 * 1024
    max_upload_files_per_request: int = 20
    # Kill-switch for any public/unattended deployment: there is no auth in
    # this codebase (get_tenant_id/get_user_id both ignore client input by
    # design -- ADR 0001), so an upload+delete API reachable by anyone who
    # can reach the port can otherwise wipe the tenant's corpus. Gates
    # POST/PATCH/DELETE on the ingest router with a 403; GET is unaffected.
    uploads_read_only: bool = False

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_tenant_id(request_tenant_id: Optional[str] = None) -> str:
    """Single source of truth for the active tenant.

    Single-tenant MVP (ADR 0001): a client-supplied tenant_id must never be
    trusted directly -- a request could claim any tenant's data by simply
    naming it (the recorded security bug at quiz.py's original
    `request.tenant_id or "company_abc"`). This ignores `request_tenant_id`
    entirely for now; the parameter exists so call sites don't change shape
    once a validated JWT claim replaces this body.
    """
    return get_settings().default_tenant_id


def get_user_id(request_user_id: Optional[str] = None) -> str:
    """Single source of truth for the active user. Same seam and same
    reasoning as get_tenant_id -- there is no auth yet (single-user MVP),
    so this is a fixed default until one exists.
    """
    return get_settings().default_user_id
