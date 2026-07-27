"""
Typed Error Hierarchy for IBLOG AI Assistant
─────────────────────────────────────────────
Provides structured errors with codes and HTTP status mapping.
"""

import logging

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base application error."""

    def __init__(
        self,
        message: str,
        code: str,
        status_code: int = 500,
        details: dict | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.details = details or {}


class OllamaConnectionError(AppError):
    """Failed to connect to Ollama."""

    def __init__(self, model: str, url: str):
        super().__init__(
            message=f"Cannot reach Ollama at {url}. Is it running with model '{model}'?",
            code="OLLAMA_CONNECTION_ERROR",
            status_code=503,
            details={"model": model, "url": url},
        )


class OllamaTimeoutError(AppError):
    """Ollama request timed out."""

    def __init__(self, model: str, timeout: int):
        super().__init__(
            message=f"Ollama request timed out after {timeout}s for model '{model}'.",
            code="OLLAMA_TIMEOUT",
            status_code=504,
            details={"model": model, "timeout": timeout},
        )


class GenerationError(AppError):
    """LLM generation failed or returned empty."""

    def __init__(self, reason: str = "Empty or invalid response from LLM"):
        super().__init__(
            message=reason,
            code="GENERATION_FAILED",
            status_code=502,
        )


class IngestionError(AppError):
    """Document ingestion failed."""

    def __init__(self, file_path: str, reason: str):
        super().__init__(
            message=f"Ingestion failed for {file_path}: {reason}",
            code="INGESTION_FAILED",
            status_code=422,
            details={"file_path": file_path},
        )


class DatasetError(AppError):
    """Training dataset generation error."""

    def __init__(self, component: str, reason: str):
        super().__init__(
            message=f"Dataset generation failed for {component}: {reason}",
            code="DATASET_ERROR",
            status_code=500,
            details={"component": component},
        )


def log_error(error: AppError, context: str = "") -> None:
    """Log an AppError with context."""
    prefix = f"[{context}] " if context else ""
    logger.error(
        "%s%s (code=%s, status=%d)",
        prefix,
        error.message,
        error.code,
        error.status_code,
        exc_info=True,
    )
