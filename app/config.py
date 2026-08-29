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
    # How long Ollama keeps a model resident after a request. Ollama's own
    # default is 5 minutes, and this deployment's tutor model is ~7.5GB:
    # a demo with a pause longer than that in it paid a FULL cold model
    # load on the next question -- minutes, indistinguishable from a hang
    # from the user's side. "30m" covers a realistic session; set "-1" to
    # pin the model for the process's whole lifetime, or "0" to unload
    # immediately (useful on a card that must free VRAM for an OCR run
    # between questions). Sent per-request by app.services.llm._post_ollama.
    ollama_keep_alive: str = "30m"
    # Per-request generation timeout. Was a hardcoded 180 in two places in
    # app/services/llm.py; a cold model load on this laptop has been
    # measured at 4-6 minutes, which is exactly the case a timeout should
    # survive rather than turn into a failed request, so it needs to be
    # tunable per deployment instead of a literal.
    ollama_timeout_seconds: int = 300
    # Context window requested per call, overriding each Modelfile's 4096
    # default. Ollama truncates from the FRONT when the window is exceeded,
    # i.e. it silently eats the system block holding the retrieved RAG
    # context first -- so this must stay comfortably above
    # max_context_length (app/services/retrieval.py) plus the history
    # window plus the response. Was duplicated as a literal 8192 in both
    # of llm.py's request builders, which could drift apart.
    ollama_num_ctx: int = 8192
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
    # Chunks per forward pass in app.services.ingestion.embed_chunks. 32,
    # not the 64 that was hardcoded there before: that number predates the
    # 2026-08-13 bge-m3 migration, which raised CHUNK_SIZE from 400 to
    # 2000 characters without revisiting it -- a 64-chunk batch is now a
    # 5x larger activation spike than when 64 was chosen, on a card that
    # is simultaneously holding the Ollama tutor model and (mid-upload)
    # the resident OCR worker. embed_chunks halves this and retries on a
    # CUDA OOM rather than failing the document, so this is a throughput
    # knob, not a correctness one -- raise it on a box with spare VRAM.
    embedding_batch_size: int = 32

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
    # The HEAVY tier: which pipeline app.services.ocr.PaddleOcrEngine's
    # resident worker (scripts/ocr_worker_resident.py) runs for a page with
    # no usable text layer at all. "vl" (PaddleOCR-VL, the 3B
    # vision-language pipeline -- highest fidelity, slowest, and the only
    # engine measured to recover this corpus's formula constants),
    # "structure" (PPStructureV3, layout+table recognition), or "classic"
    # (PaddleOCR/PP-OCRv5 -- no layout/table structure at all).
    # See scripts/ocr_bakeoff.py for the measured numbers behind this.
    ocr_paddle_engine: str = "vl"
    # The LIGHT tier, used for OCR_PREFERRED pages when ocr_two_tier is on.
    # Measured warm on arabic_test.pdf: 5.5s/page vs "vl"'s 52.5s (~10x),
    # scoring 4/5 on p15's CAD-layer table -- its only miss there
    # (0.999600) comes from the page's native text layer anyway, which
    # _parse_pdf merges in. It scores 0/4 on p51's formulas, which is
    # exactly why formula-bearing pages must not silently land here; see
    # ocr_two_tier below.
    ocr_light_engine: str = "classic"
    # Two-tier page routing (app.services.ingestion._ocr_pdf_page):
    #   OCR_REQUIRED  -> ocr_paddle_engine (heavy). No native text exists
    #                    to fall back on, so fidelity outranks speed.
    #   OCR_PREFERRED -> ocr_light_engine (light), because the page's
    #                    native text is already being merged in and OCR is
    #                    only being asked for the embedded table/figure.
    #                    Escalates to the heavy engine when the light
    #                    engine's output looks numerically empty (see
    #                    ingestion._classic_ocr_looks_incomplete).
    # Set False to send every OCR page to ocr_paddle_engine, the pre-
    # 2026-08-19 behaviour -- slower, and the fallback if the light tier
    # is ever found to drop content on a real tenant document.
    ocr_two_tier: bool = True
    # Seconds of no OCR call before app.services.ocr._ResidentOcrWorker
    # kills its subprocess to free VRAM, restarting (cold-loading again)
    # on the next call. 0 disables release -- the pre-2026-08-23 behaviour,
    # where the worker held VRAM for this app process's whole remaining
    # lifetime once started. Measured consequence of that: with the worker
    # resident, the 7.5GB Darija tutor model could not fit alongside it on
    # an 8GB card and Ollama loaded it 31% CPU / 69% GPU, slow enough that
    # a live chat request exceeded its 180s client timeout. 120s matches
    # ingest_queue's single-worker design (one document at a time, likely
    # idle between uploads) without releasing so eagerly that back-to-back
    # pages within one document's OCR pass keep re-paying the cold-load.
    ocr_worker_idle_release_seconds: float = 120.0

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

    # Speech (voice pipeline: app/routers/voice.py, app/services/stt.py,
    # app/services/tts.py). Both default to "none" -- 2026-08-25's bake-off
    # (scripts/eval_stt.py / eval_tts.py, docs/architecture/voice-assistant.md)
    # has not been RUN yet (needs a rented GPU; this laptop's 8GB card
    # cannot hold an STT model alongside the tutor -- see
    # docs/architecture/cloud-scaling-plan.md), so no vendor is selected.
    # "none" makes that an honest, loud failure (SttUnavailableError /
    # TtsUnavailableError) instead of a mounted endpoint that silently does
    # nothing, matching app.services.ocr's NullOcrEngine precedent.
    stt_engine: str = "none"  # "none" | "whisper" | "seamless"
    # Which STT model the "whisper" engine's resident worker loads --
    # candidate names only until the bake-off picks one. faster-whisper
    # naming convention (e.g. "large-v3-turbo"); a Darija-specific
    # community checkpoint's HF repo id is expected to win for Arabic per
    # the blueprint's bake-off plan, at which point this becomes two
    # settings (per-language) rather than one -- not built yet because
    # nothing depends on it until Phase 0 reports a result.
    stt_model: str = "large-v3-turbo"
    # Path to the DEDICATED venv's interpreter for the STT resident worker
    # (scripts/speech_worker_resident.py), same reasoning as
    # settings.ocr_venv_python: STT deps (ctranslate2, onnxruntime, or
    # SeamlessM4T's torch pin) are expected to conflict with .gguf_venv's
    # own torch pin, so they get their own venv rather than sharing it.
    # Only read when stt_engine="whisper" or "seamless".
    stt_venv_python: str = "./.speech_venv/Scripts/python.exe"
    # Seconds of no STT call before the resident worker releases its VRAM,
    # same idle-release contract as settings.ocr_worker_idle_release_seconds
    # (see that setting's comment) -- an open-mic voice session is bursty,
    # not continuous, and STT must not permanently steal VRAM the tutor
    # model needs to stay resident.
    speech_worker_idle_release_seconds: float = 120.0
    tts_engine: str = "none"  # "none" | "piper"
    # Piper voice models (ONNX, downloaded separately -- see
    # app.services.tts.PiperEngine's docstring for why Piper was chosen
    # over XTTS-v2/MMS-TTS: CPU-only, ~zero VRAM contention with the
    # resident tutor model, and MIT-licensed where the higher-quality
    # alternatives are non-commercial). ar_JO is Jordanian Arabic, the
    # closest off-the-shelf Piper voice to Moroccan Darija at MVP time --
    # intelligible but noticeably not native Darija prosody; a Piper voice
    # fine-tuned on atlasia/DODa-audio-dataset is the tracked follow-up
    # (docs/architecture/voice-assistant.md), not built yet.
    tts_voice_fr: str = "fr_FR-siwis-medium"
    tts_voice_ar: str = "ar_JO-kareem-medium"
    # Directory holding each voice's downloaded {name}.onnx + {name}.onnx.json
    # pair (Piper's model format) -- Piper voices are not bundled with the
    # pip package and must be fetched separately per
    # app.services.tts.PiperEngine's docstring.
    tts_voice_dir: str = "./data/tts_voices"

    # Diagram generation (app/services/diagrams.py). Kill-switch first: a
    # chat turn falling back to prose on a stuck Ollama/GPU is much less
    # visible than every diagram request failing loudly, so this can be
    # flipped off without touching code if diagram generation ever needs to
    # be pulled from a live deployment quickly.
    diagrams_enabled: bool = True
    # Language every STRUCTURAL diagram label (node/edge/participant/slice/
    # axis text) must be written in, regardless of the turn's own response
    # language -- the caption alone follows response_lang. "fr" is the only
    # value app.services.diagrams's language gate currently implements
    # (Latin-script-only enforcement); this is a knob, not a hardcode,
    # because the platform is multilingual and a future tenant may want a
    # different structural-label language without a code change.
    diagram_label_language: str = "fr"
    # Node-count ceiling per diagram (flowchart/sequence/mindmap nodes, pie
    # slices, xy points, candlesticks) enforced by the heal tier
    # (app.services.diagrams's per-kind heal functions) -- keeps a diagram
    # legible in the chat panel and keeps a single Ollama call's JSON output
    # bounded. Excess items are truncated, not rejected outright.
    diagram_max_nodes: int = 14

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
