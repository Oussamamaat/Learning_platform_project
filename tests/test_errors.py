"""
Regression test for a bug caught live during the pgvector lock-in (2026-08-10):
a transient Ollama disconnect surfaced as
"'OllamaConnectionError' object has no attribute 'message'" instead of the
intended structured error response, because AppError never set `.message`
-- every catch site (app/main.py, app/routers/chat.py, app/routers/quiz.py)
reads it directly, and base Exception has no such attribute in Python 3.
"""
from app.errors import AppError, GenerationError, OllamaConnectionError


def test_app_error_has_message_attribute():
    e = AppError("something broke", "TEST_CODE", status_code=418)
    assert e.message == "something broke"
    assert e.code == "TEST_CODE"
    assert e.status_code == 418


def test_every_subclass_sets_message():
    """Each subclass composes its own message string internally -- confirm
    it actually reaches `.message`, not just `args[0]`."""
    assert "model1" in OllamaConnectionError("model1", "http://x").message
    assert GenerationError("empty reply").message == "empty reply"


def test_error_handler_shape_does_not_crash():
    """The exact pattern every catch site uses -- must not raise."""
    e = OllamaConnectionError("IBLOG_TUTOR:latest", "http://localhost:11434")
    payload = {"error": {"code": e.code, "message": e.message}}
    assert payload["error"]["message"]
