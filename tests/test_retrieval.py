"""
Tests for app/services/retrieval.py: the language-affinity two-pass
selection (ADR 0002 decision 5) and the backend-forcing contract that
keeps quiz's pgvector identity and chat's disk identity independent of
settings.retrieval_backend. No embeddings, no DB -- _select_with_affinity
is tested directly against fabricated candidate dicts.
"""
from unittest.mock import patch

from app.services.retrieval import _fingerprint, _select_with_affinity, retrieve


def _candidate(content, source, language, similarity):
    return {"content": content, "source_name": source, "language": language, "similarity": similarity}


# --- _select_with_affinity ------------------------------------------------

def test_no_ui_lang_is_plain_top_k_by_similarity():
    """target_lang=None (no ui_lang supplied) must not apply any language
    preference -- this is the opt-in contract every pre-existing caller
    that doesn't resolve a ui_lang depends on."""
    candidates = [
        _candidate("a", "s1.md", "ar", 0.9),
        _candidate("b", "s2.md", "fr", 0.8),
        _candidate("c", "s3.md", "ar", 0.7),
    ]
    selected, cross_language = _select_with_affinity(
        candidates, target_lang=None, top_k=2, threshold=None
    )
    assert [c["content"] for c in selected] == ["a", "b"]
    assert cross_language is False


def test_prefers_same_language_when_enough_available():
    candidates = [
        _candidate("fr-best", "s1.md", "fr", 0.95),
        _candidate("ar-best", "s2.md", "ar", 0.90),
        _candidate("fr-second", "s3.md", "fr", 0.85),
        _candidate("ar-second", "s4.md", "ar", 0.80),
    ]
    selected, cross_language = _select_with_affinity(
        candidates, target_lang="fr", top_k=2, threshold=None
    )
    assert [c["content"] for c in selected] == ["fr-best", "fr-second"]
    assert cross_language is False


def test_crosses_language_line_when_too_few_same_language():
    candidates = [
        _candidate("fr-only", "s1.md", "fr", 0.95),
        _candidate("ar-best", "s2.md", "ar", 0.90),
        _candidate("ar-second", "s3.md", "ar", 0.85),
    ]
    selected, cross_language = _select_with_affinity(
        candidates, target_lang="fr", top_k=2, threshold=None
    )
    # Only 1 french candidate exists (< top_k=2), so falls back to the
    # full pool ranked by similarity, not "french first no matter what".
    assert [c["content"] for c in selected] == ["fr-only", "ar-best"]
    assert cross_language is True


def test_threshold_none_keeps_disk_backend_no_cutoff_philosophy():
    candidates = [_candidate("low-sim", "s1.md", "fr", 0.05)]
    selected, _ = _select_with_affinity(candidates, target_lang=None, top_k=4, threshold=None)
    assert len(selected) == 1  # not dropped despite low similarity


def test_threshold_float_filters_low_similarity_candidates():
    candidates = [
        _candidate("passes", "s1.md", "fr", 0.5),
        _candidate("fails", "s2.md", "fr", 0.1),
    ]
    selected, _ = _select_with_affinity(candidates, target_lang=None, top_k=4, threshold=0.4)
    assert [c["content"] for c in selected] == ["passes"]


def test_exact_top_k_same_language_does_not_cross():
    candidates = [
        _candidate("fr1", "s1.md", "fr", 0.9),
        _candidate("fr2", "s2.md", "fr", 0.8),
        _candidate("ar1", "s3.md", "ar", 0.95),
    ]
    selected, cross_language = _select_with_affinity(
        candidates, target_lang="fr", top_k=2, threshold=None
    )
    assert [c["content"] for c in selected] == ["fr1", "fr2"]
    assert cross_language is False


# --- retrieve(): backend forcing -------------------------------------------

def test_unknown_backend_raises():
    import pytest
    with pytest.raises(ValueError, match="unknown retrieval backend"):
        retrieve("q", domain="industrial", backend="something-else")


def test_pgvector_backend_never_reads_disk_corpus():
    """Regression guard for the bug caught during implementation: routing
    every call through settings.retrieval_backend would have silently
    redirected quiz's pgvector calls to disk whenever the global default
    ('disk') was active. backend is a required, explicit argument instead."""
    with patch("app.services.retrieval._disk_search") as disk_search, \
         patch("app.services.retrieval._pgvector_search", return_value=[]) as pg_search:
        retrieve("q", domain="industrial", backend="pgvector", tenant_id="t1")
        pg_search.assert_called_once()
        disk_search.assert_not_called()


def test_disk_backend_never_touches_pgvector():
    with patch("app.services.retrieval._pgvector_search") as pg_search, \
         patch("app.services.retrieval._disk_search", return_value=[]) as disk_search:
        retrieve("q", domain="industrial", backend="disk")
        disk_search.assert_called_once()
        pg_search.assert_not_called()


# --- _fingerprint: must always fit chat_sessions.pinned_fingerprint VARCHAR(64) --
# Regression for a bug caught live (2026-08-10): app/routers/chat.py originally
# wrote the RAW "domain|ui_lang|message" string as the fingerprint instead of
# calling this hash function, overflowing VARCHAR(64) on any real message and
# silently failing every pin write (fail-open masked it as a logged exception,
# not a visible chat failure -- but pinning never persisted). SQLite (used by
# tests/test_history.py) doesn't enforce column length the way Postgres does,
# so that suite could never have caught this; a length assertion on the hash
# itself is the portable test.

def test_fingerprint_is_always_64_chars_regardless_of_input_length():
    short = _fingerprint(domain="industrial", ui_lang="fr", query="q")
    long_query = "Quels equipements de protection individuelle sont obligatoires " * 5
    long = _fingerprint(domain="industrial", ui_lang="fr", query=long_query)
    assert len(short) == 64
    assert len(long) == 64


def test_fingerprint_is_valid_hex():
    fp = _fingerprint(domain="industrial", ui_lang="fr", query="test")
    int(fp, 16)  # raises ValueError if not valid hex


def test_fingerprint_differs_by_query_domain_or_language():
    base = _fingerprint(domain="industrial", ui_lang="fr", query="q")
    assert _fingerprint(domain="securite", ui_lang="fr", query="q") != base
    assert _fingerprint(domain="industrial", ui_lang="darija", query="q") != base
    assert _fingerprint(domain="industrial", ui_lang="fr", query="q2") != base
