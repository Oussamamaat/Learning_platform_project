"""Tests for app.services.tts.get_tts_engine()'s probe-and-fallback
resolution (added 2026-09-06, after a lease where a torchcodec/CUDA
mismatch inside the XTTS worker left every voice session on that lease
silently unable to speak -- no health signal, no fallback, and the failure
only surfaced once a human opened a browser).

No real XTTS/Piper/subprocess involved: the primary engine's warmup() is
monkeypatched to succeed or raise, same "fake the boundary, test the
decision" convention as tests/test_web_search.py uses for its engine
registry.
"""
from unittest.mock import MagicMock, patch

import pytest

import app.services.tts as tts_module
from app.services.tts import TtsUnavailableError, active_tts_engine_status, get_tts_engine


@pytest.fixture(autouse=True)
def _reset_engine_singleton():
    """get_tts_engine() resolves once per process and caches the result --
    tests that patch settings.tts_engine/tts_fallback_engine need a fresh
    resolution each time, not the previous test's cached instance."""
    tts_module._active_engine = None
    tts_module._active_status.clear()
    yield
    tts_module._active_engine = None
    tts_module._active_status.clear()


def _patched_settings(**overrides):
    settings = MagicMock()
    settings.tts_engine = "none"
    settings.tts_fallback_engine = "piper"
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


def test_engine_with_no_warmup_is_returned_directly():
    """"none" and "piper" define no warmup() -- resolution must not require
    one, and must report ok=True with no fallback."""
    with patch("app.services.tts.get_settings", return_value=_patched_settings(tts_engine="none")):
        engine = get_tts_engine()
    assert isinstance(engine, tts_module.NullTtsEngine)
    status = active_tts_engine_status()
    assert status == {"configured": "none", "active": "none", "fallback_used": False, "ok": True}


def test_successful_primary_warmup_stays_on_primary():
    with patch("app.services.tts.get_settings",
               return_value=_patched_settings(tts_engine="xtts_darija", tts_fallback_engine="piper")), \
         patch.object(tts_module.XttsDarijaEngine, "warmup", return_value=None):
        engine = get_tts_engine()
    assert isinstance(engine, tts_module.XttsDarijaEngine)
    status = active_tts_engine_status()
    assert status["active"] == "xtts_darija"
    assert status["fallback_used"] is False
    assert status["ok"] is True


def test_failed_primary_warmup_falls_back_to_configured_engine():
    with patch("app.services.tts.get_settings",
               return_value=_patched_settings(tts_engine="xtts_darija", tts_fallback_engine="piper")), \
         patch.object(tts_module.XttsDarijaEngine, "warmup",
                      side_effect=OSError("libcudart.so.13: cannot open shared object file")):
        engine = get_tts_engine()
    assert isinstance(engine, tts_module.PiperEngine)
    status = active_tts_engine_status()
    assert status["configured"] == "xtts_darija"
    assert status["active"] == "piper"
    assert status["fallback_used"] is True
    assert status["ok"] is False
    assert "libcudart" in status["error"]


def test_failed_primary_warmup_with_fallback_disabled_stays_on_primary_and_fails_loudly():
    """settings.tts_fallback_engine="none" must preserve the pre-existing
    contract: the broken engine is returned unchanged, and callers find out
    at synthesize() time, not silently via an unannounced substitution."""
    with patch("app.services.tts.get_settings",
               return_value=_patched_settings(tts_engine="xtts_darija", tts_fallback_engine="none")), \
         patch.object(tts_module.XttsDarijaEngine, "warmup", side_effect=RuntimeError("boom")):
        engine = get_tts_engine()
    assert isinstance(engine, tts_module.XttsDarijaEngine)
    status = active_tts_engine_status()
    assert status["active"] == "xtts_darija"
    assert status["fallback_used"] is False
    assert status["ok"] is False


def test_fallback_equal_to_primary_does_not_double_instantiate():
    """A misconfiguration (fallback == primary) must not be treated as a
    real fallback -- there is nothing else to fall back to."""
    with patch("app.services.tts.get_settings",
               return_value=_patched_settings(tts_engine="xtts_darija", tts_fallback_engine="xtts_darija")), \
         patch.object(tts_module.XttsDarijaEngine, "warmup", side_effect=RuntimeError("boom")):
        engine = get_tts_engine()
    assert isinstance(engine, tts_module.XttsDarijaEngine)
    assert active_tts_engine_status()["fallback_used"] is False


def test_unknown_engine_name_raises():
    with patch("app.services.tts.get_settings", return_value=_patched_settings(tts_engine="bogus")):
        with pytest.raises(TtsUnavailableError):
            get_tts_engine()


def test_resolution_happens_once_and_is_cached():
    settings = _patched_settings(tts_engine="piper")
    with patch("app.services.tts.get_settings", return_value=settings) as mock_get_settings:
        first = get_tts_engine()
        second = get_tts_engine()
    assert first is second
    assert mock_get_settings.call_count == 1


def test_health_status_empty_before_any_resolution():
    assert active_tts_engine_status() == {}
