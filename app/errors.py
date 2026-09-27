"""
Typed Error Hierarchy for IBLOG AI Assistant
─────────────────────────────────────────────
Provides structured errors with codes and HTTP status mapping.
"""


class AppError(Exception):
    """Base application error."""

    def __init__(
        self,
        message: str,
        code: str,
        status_code: int = 500,
    ):
        super().__init__(message)
        # Exception.args[0] holds this too, but every catch site
        # (app/main.py's global handler, chat.py, quiz.py) reads
        # `.message` directly -- base Exception has no such attribute in
        # Python 3, so every one of those sites raised AttributeError
        # instead of returning the intended structured error response.
        # Caught live: a transient Ollama disconnect turned into
        # "'OllamaConnectionError' object has no attribute 'message'"
        # instead of a clean 503 JSON body.
        self.message = message
        self.code = code
        self.status_code = status_code


class LLMConnectionError(AppError):
    """Failed to connect to the configured LLM backend (Ollama or vLLM).

    Backend-neutral supertype added for the vLLM migration
    (app.services.llm's llm_generate/llm_chat/llm_stream_chat dispatchers) so
    a caller that just wants "the inference server is unreachable" can catch
    one type regardless of settings.llm_backend. OllamaConnectionError below
    is kept as a subclass with its original message/code unchanged, since
    tests/test_errors.py and every existing catch site key off that exact
    name and wording -- this class is additive, not a replacement.
    """

    def __init__(self, backend: str, model: str, url: str):
        super().__init__(
            message=f"Cannot reach {backend} at {url}. Is it running with model '{model}'?",
            code="LLM_CONNECTION_ERROR",
            status_code=503,
        )


class OllamaConnectionError(LLMConnectionError):
    """Failed to connect to Ollama."""

    def __init__(self, model: str, url: str):
        # Bypasses LLMConnectionError.__init__ deliberately: this preserves
        # the exact message text ("Cannot reach Ollama at...", no backend
        # name repeated) and code ("OLLAMA_CONNECTION_ERROR") every existing
        # caller and test already depends on, while still being an
        # LLMConnectionError for anything that wants the neutral catch.
        AppError.__init__(
            self,
            message=f"Cannot reach Ollama at {url}. Is it running with model '{model}'?",
            code="OLLAMA_CONNECTION_ERROR",
            status_code=503,
        )


class GenerationError(AppError):
    """LLM generation failed or returned empty."""

    def __init__(self, reason: str = "Empty or invalid response from LLM"):
        super().__init__(
            message=reason,
            code="GENERATION_FAILED",
            status_code=502,
        )


class ForbiddenError(AppError):
    """The caller's role is not permitted to perform this action.

    Distinct from the 403 app/routers/ingest.py raises for
    settings.uploads_read_only: that one is a deployment-wide kill switch
    ("nobody may write"), this one is per-caller ("you specifically may
    not"). Both surface through app/main.py's AppError handler as the same
    structured {"error": {"code", "message"}} body, so the frontend can
    tell them apart by code without parsing prose.
    """

    def __init__(self, action: str, role: str, allowed: str):
        super().__init__(
            message=f"Role '{role}' cannot {action}. Allowed roles: {allowed}.",
            code="ROLE_FORBIDDEN",
            status_code=403,
        )
