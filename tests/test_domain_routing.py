"""
Tests for app/services/routing.py's domain router (tier 2/3 of the
"Automatic Domain Routing" plan, 2026-08-11): vote_domain() is pure and
tested directly against fabricated candidates (no DB); resolve_domain()'s
DB-touching search is monkeypatched at its import site in routing.py, the
same pattern tests/test_retrieval.py already uses for search_similar_chunks.
"""
from unittest.mock import patch

from app.services.routing import resolve_domain, vote_domain


def _candidate(domain, similarity):
    return {"domain": domain, "similarity": similarity}


# --- vote_domain -------------------------------------------------------

def test_vote_picks_highest_summed_similarity_not_raw_count():
    candidates = [
        _candidate("industrial", 0.9),          # one strong industrial hit
        _candidate("securite", 0.2),             # three weak securite hits
        _candidate("securite", 0.18),
        _candidate("securite", 0.16),
    ]
    # securite: 0.2+0.18+0.16 = 0.54 < industrial: 0.9 -- industrial wins
    # despite fewer candidates, because the vote is similarity-weighted.
    assert vote_domain(candidates, threshold=0.15) == "industrial"


def test_vote_ignores_candidates_below_threshold():
    candidates = [_candidate("blockchain", 0.05)]
    assert vote_domain(candidates, threshold=0.15) is None


def test_vote_null_domain_never_votes():
    candidates = [
        _candidate(None, 0.99),   # legacy/untagged row -- must not vote
        _candidate("securite", 0.16),
    ]
    assert vote_domain(candidates, threshold=0.15) == "securite"


def test_vote_empty_pool_returns_none():
    assert vote_domain([], threshold=0.15) is None


def test_vote_all_candidates_null_domain_returns_none():
    candidates = [_candidate(None, 0.9), _candidate(None, 0.8)]
    assert vote_domain(candidates, threshold=0.15) is None


# --- resolve_domain ------------------------------------------------------

def test_resolve_domain_pgvector_tier2_wins_when_vote_clears_threshold():
    with patch(
        "app.services.routing.search_similar_chunks",
        return_value=[_candidate("blockchain", 0.5)],
    ) as mock_search:
        domain, source = resolve_domain("q", tenant_id="t1", backend="pgvector")
        assert (domain, source) == ("blockchain", "retrieval")
        # Unfiltered -- domain=None is the whole point of tier 2's search.
        mock_search.assert_called_once()
        assert mock_search.call_args.kwargs["domain"] is None


def test_resolve_domain_pgvector_falls_to_tenant_default_when_vote_empty():
    with patch("app.services.routing.search_similar_chunks", return_value=[]):
        domain, source = resolve_domain("q", tenant_id="t1", backend="pgvector")
        assert source == "tenant_default"
        assert domain  # settings.default_domain, non-empty


def test_resolve_domain_fails_open_to_tenant_default_on_db_error():
    """A Postgres hiccup during tier 2's search must degrade routing to the
    tenant default, not crash the request -- same fail-open contract as
    app.routers.chat._retrieve_context's pgvector fallback."""
    with patch(
        "app.services.routing.search_similar_chunks",
        side_effect=ConnectionError("db unreachable"),
    ):
        domain, source = resolve_domain("q", tenant_id="t1", backend="pgvector")
        assert source == "tenant_default"
        assert domain


def test_resolve_domain_disk_backend_skips_tier2_entirely():
    """Disk backend has no single cross-domain corpus to vote over -- must
    go straight to the tenant default without calling search_similar_chunks
    (which is pgvector-only) at all."""
    with patch("app.services.routing.search_similar_chunks") as mock_search:
        domain, source = resolve_domain("q", tenant_id="t1", backend="disk")
        assert source == "tenant_default"
        mock_search.assert_not_called()
