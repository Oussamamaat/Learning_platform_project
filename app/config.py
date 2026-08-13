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
