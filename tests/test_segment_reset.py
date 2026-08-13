"""
Tests for app.routers.chat._resolve_turn_context -- the pinned-context /
segment-reset design (memory+RAG plan, Stage 3) and its three guards
against false resets, extended 2026-08-11 (Automatic Domain Routing plan)
to also resolve domain automatically instead of taking it as a given:

1. An anaphoric follow-up can never trigger a reset or a routing decision.
2. A reset-candidate's retrieval always runs on the condensed query.
3. An empty reset-candidate retrieval falls back to the old pin instead
   of refusing.
4. (new) The domain router (resolve_domain) only runs when there is no
   usable pin to try first -- a same-topic continuation never calls it.

No live Postgres or Ollama -- app.routers.chat.history.*,
app.routers.chat.resolve_domain, and app.routers.chat._retrieve_context
(the backend-selection indirection point, not the backend-specific
build_domain_context/build_rag_context functions themselves) are
monkeypatched directly, same pattern as tests/test_chat.py. Patching at
_retrieve_context means these tests cover the guard logic identically
regardless of settings.retrieval_backend -- which backend gets called is
_retrieve_context's own concern, covered separately in test_chat.py.

`_resolve_turn_context` now returns a 6-tuple:
(context, sources, segment_id, is_new_pin, domain, domain_source).
Every test here passes an explicit `requested_domain` (tier 1 / page
context) unless it's specifically testing auto-routing (tier 2/3) --
that keeps these tests focused on the pin/reset guards, with
tests/test_domain_routing.py owning resolve_domain's own vote logic.
"""
from unittest.mock import patch

from app.routers.chat import _resolve_turn_context


PINNED_INDUSTRIAL_FR = {
    "context": "Article 283: obligations du travailleur...",
    "sources": ["1.1_code_du_travail_health_safety.md"],
    "fingerprint": "abc",
    "segment_id": 3,
    "domain": "industrial",
    "language": "fr",
}


def test_no_pin_first_turn_always_retrieves_fresh_and_starts_segment_1():
    with patch("app.routers.chat.history.get_pinned", return_value=None), \
         patch("app.routers.chat._retrieve_context",
               return_value=("fresh context", ["doc.md"])) as rc:
        context, sources, segment_id, is_new_pin, domain, domain_source = _resolve_turn_context(
            "sess-1", "Quelle est l'obligation du travailleur ?", "industrial", "fr", "t1"
        )
    assert context == "fresh context"
    assert sources == ["doc.md"]
    assert segment_id == 1
    assert is_new_pin is True
    assert domain == "industrial"
    assert domain_source == "page_context"
    # First turn: query passed through unchanged (no prior_turn to condense with).
    assert rc.call_args.args[0] == "Quelle est l'obligation du travailleur ?"


def test_guard1_anaphoric_followup_never_resets_reuses_pin_without_retrieving():
    with patch("app.routers.chat.history.get_pinned", return_value=PINNED_INDUSTRIAL_FR), \
         patch("app.routers.chat._retrieve_context") as rc, \
         patch("app.routers.chat.resolve_domain") as rd:
        context, sources, segment_id, is_new_pin, domain, domain_source = _resolve_turn_context(
            "sess-1", "Pourquoi ?", "industrial", "fr", "t1"
        )
    rc.assert_not_called()
    rd.assert_not_called()
    assert context == PINNED_INDUSTRIAL_FR["context"]
    assert sources == PINNED_INDUSTRIAL_FR["sources"]
    assert segment_id == PINNED_INDUSTRIAL_FR["segment_id"]
    assert is_new_pin is False
    assert domain == "industrial"
    # requested_domain was explicit -- "page_context", even though guard 1
    # short-circuited to the pin's stored content.
    assert domain_source == "page_context"


def test_same_topic_reuses_pin_verbatim_not_the_fresh_probe_result():
    """Sources overlap with the pin -- must return the PIN's context (byte
    identical, for KV-prefix reuse), not the freshly-probed one, even
    though the probe retrieval did run."""
    with patch("app.routers.chat.history.get_pinned", return_value=PINNED_INDUSTRIAL_FR), \
         patch("app.routers.chat.history.load_window", return_value=[{"role": "user", "content": "prior"}]), \
         patch("app.routers.chat._retrieve_context",
               return_value=("different-text-same-source", ["1.1_code_du_travail_health_safety.md"])):
        context, sources, segment_id, is_new_pin, domain, domain_source = _resolve_turn_context(
            "sess-1", "Et pourquoi cette obligation existe-t-elle vraiment ?", "industrial", "fr", "t1"
        )
    assert context == PINNED_INDUSTRIAL_FR["context"]  # NOT "different-text-same-source"
    assert segment_id == PINNED_INDUSTRIAL_FR["segment_id"]
    assert is_new_pin is False
    assert domain == "industrial"
    assert domain_source == "page_context"


def test_genuine_topic_shift_starts_new_segment_with_fresh_context():
    with patch("app.routers.chat.history.get_pinned", return_value=PINNED_INDUSTRIAL_FR), \
         patch("app.routers.chat.history.load_window", return_value=[{"role": "user", "content": "prior"}]), \
         patch("app.routers.chat._retrieve_context",
               return_value=("EPI content", ["1.6_ppe_requirements.md"])):
        context, sources, segment_id, is_new_pin, domain, domain_source = _resolve_turn_context(
            "sess-1", "Quels equipements de protection individuelle sont obligatoires ?", "industrial", "fr", "t1"
        )
    assert context == "EPI content"
    assert sources == ["1.6_ppe_requirements.md"]
    assert segment_id == PINNED_INDUSTRIAL_FR["segment_id"] + 1
    assert is_new_pin is True
    # requested_domain was explicit -- always "page_context", regardless
    # of same-topic vs. topic-shift.
    assert domain == "industrial"
    assert domain_source == "page_context"


def test_guard3_empty_reset_candidate_retrieval_falls_back_to_pin_not_refusal():
    """A reset-candidate whose own retrieval comes back empty must reuse
    the old pin rather than surface an empty context (which would fire
    deterministic_refusal in the caller) -- the false-refusal guard."""
    with patch("app.routers.chat.history.get_pinned", return_value=PINNED_INDUSTRIAL_FR), \
         patch("app.routers.chat.history.load_window", return_value=[{"role": "user", "content": "prior"}]), \
         patch("app.routers.chat._retrieve_context", return_value=("", [])):
        context, sources, segment_id, is_new_pin, domain, domain_source = _resolve_turn_context(
            "sess-1", "Quelque chose de totalement hors sujet ici", "industrial", "fr", "t1"
        )
    assert context == PINNED_INDUSTRIAL_FR["context"]
    assert sources == PINNED_INDUSTRIAL_FR["sources"]
    assert is_new_pin is False
    assert domain_source == "page_context"


def test_genuinely_off_topic_opening_question_still_empty_no_pin_to_fall_back_to():
    """Guard 3 must not disable grounded refusal outright -- with NO prior
    pin to fall back to (e.g. session's first turn), an empty retrieval
    stays empty and the caller's deterministic_refusal still fires."""
    with patch("app.routers.chat.history.get_pinned", return_value=None), \
         patch("app.routers.chat._retrieve_context", return_value=("", [])):
        context, sources, segment_id, is_new_pin, domain, domain_source = _resolve_turn_context(
            "sess-1", "Quelle est la recette du tajine ?", "industrial", "fr", "t1"
        )
    assert context == ""
    assert sources == []
    assert segment_id == 1
    assert is_new_pin is True
    assert domain == "industrial"
    assert domain_source == "page_context"


def test_domain_switch_always_starts_new_segment_even_with_overlapping_sources():
    """A domain switch is a hard boundary -- the system prompt itself
    names a different domain, so there is no meaningful 'same topic'
    check to run, regardless of what the fresh retrieval returns."""
    with patch("app.routers.chat.history.get_pinned", return_value=PINNED_INDUSTRIAL_FR), \
         patch("app.routers.chat._retrieve_context",
               return_value=("blockchain content", ["3.1_bill_42_25_draft.md"])) as rc:
        context, sources, segment_id, is_new_pin, domain, domain_source = _resolve_turn_context(
            "sess-1", "Qu'est-ce que la loi 42.25 ?", "blockchain", "fr", "t1"
        )
    assert segment_id == PINNED_INDUSTRIAL_FR["segment_id"] + 1
    assert is_new_pin is True
    assert domain == "blockchain"
    assert domain_source == "page_context"
    # No prior_turn window lookup for a scope switch -- nothing to condense with.
    assert rc.call_args.args[0] == "Qu'est-ce que la loi 42.25 ?"


def test_language_switch_always_starts_new_segment():
    darija_message = "شنو هي القواعد؟"
    with patch("app.routers.chat.history.get_pinned", return_value=PINNED_INDUSTRIAL_FR), \
         patch("app.routers.chat._retrieve_context",
               return_value=("darija content", ["1.11_ar_code_travail_salama.md"])):
        context, sources, segment_id, is_new_pin, domain, domain_source = _resolve_turn_context(
            "sess-1", darija_message, "industrial", "darija", "t1"
        )
    assert segment_id == PINNED_INDUSTRIAL_FR["segment_id"] + 1
    assert is_new_pin is True


# --- domain auto-routing (requested_domain=None) ----------------------------
# resolve_domain's own vote logic is tested directly in
# tests/test_domain_routing.py; these tests only cover WHEN chat calls it.

def test_autorouting_first_turn_calls_resolve_domain():
    with patch("app.routers.chat.history.get_pinned", return_value=None), \
         patch("app.routers.chat.resolve_domain", return_value=("securite", "retrieval")) as rd, \
         patch("app.routers.chat._retrieve_context", return_value=("ctx", ["s.md"])):
        context, sources, segment_id, is_new_pin, domain, domain_source = _resolve_turn_context(
            "sess-1", "Quelles cameras sont obligatoires ?", None, "fr", "t1"
        )
    rd.assert_called_once()
    assert domain == "securite"
    assert domain_source == "retrieval"


def test_autorouting_anaphoric_followup_never_calls_resolve_domain():
    """Guard 1: an anaphoric follow-up (this message matches the "pourquoi"
    marker) never reaches the domain router at all when the pin's language
    still matches -- it's trusted to stay in the pin's domain, no vote
    needed. A non-anaphoric same-topic continuation DOES still call
    resolve_domain (see test_autorouting_first_turn_calls_resolve_domain
    for that shape) -- routing isn't skipped for topic continuity, only for
    messages that carry no standalone signal to route by."""
    with patch("app.routers.chat.history.get_pinned", return_value=PINNED_INDUSTRIAL_FR), \
         patch("app.routers.chat.history.load_window", return_value=[{"role": "user", "content": "prior"}]), \
         patch("app.routers.chat._retrieve_context",
               return_value=("different-text-same-source", ["1.1_code_du_travail_health_safety.md"])), \
         patch("app.routers.chat.resolve_domain") as rd:
        context, sources, segment_id, is_new_pin, domain, domain_source = _resolve_turn_context(
            "sess-1", "Et pourquoi cette obligation existe-t-elle vraiment ?", None, "fr", "t1"
        )
    rd.assert_not_called()
    assert domain == "industrial"
    assert domain_source == "pinned"


def test_autorouting_language_switch_breaks_scope_and_calls_resolve_domain():
    """A pinned session whose language no longer matches this turn's
    query_lang can't be tried speculatively -- must re-route."""
    darija_message = "شنو هي القواعد؟"
    with patch("app.routers.chat.history.get_pinned", return_value=PINNED_INDUSTRIAL_FR), \
         patch("app.routers.chat.resolve_domain", return_value=("industrial", "retrieval")) as rd, \
         patch("app.routers.chat._retrieve_context",
               return_value=("darija content", ["1.11_ar_code_travail_salama.md"])):
        context, sources, segment_id, is_new_pin, domain, domain_source = _resolve_turn_context(
            "sess-1", darija_message, None, "darija", "t1"
        )
    rd.assert_called_once()
    assert domain_source == "retrieval"


def test_autorouting_non_anaphoric_same_topic_still_calls_resolve_domain():
    """A non-anaphoric same-topic continuation IS routed every turn -- this
    codebase deliberately does not try to guess "still the pinned domain"
    before calling the router, because a wrong guess would silently ground
    a genuine cross-domain shift in a stale domain. The router's own vote
    (mocked here) still returns the same domain the pin already had, so
    the overlap check below reuses the pin's content regardless."""
    with patch("app.routers.chat.history.get_pinned", return_value=PINNED_INDUSTRIAL_FR), \
         patch("app.routers.chat.history.load_window", return_value=[{"role": "user", "content": "prior"}]), \
         patch("app.routers.chat.resolve_domain", return_value=("industrial", "retrieval")) as rd, \
         patch("app.routers.chat._retrieve_context",
               return_value=("different-text-same-source", ["1.1_code_du_travail_health_safety.md"])):
        context, sources, segment_id, is_new_pin, domain, domain_source = _resolve_turn_context(
            "sess-1", "Quels equipements de protection individuelle sont obligatoires ?", None, "fr", "t1"
        )
    rd.assert_called_once()
    assert context == PINNED_INDUSTRIAL_FR["context"]  # sources overlapped -- pin reused
    assert domain_source == "retrieval"


def test_guard1_does_not_fire_when_explicit_domain_disagrees_with_pin():
    """An anaphoric message with a genuinely different EXPLICIT domain must
    not blindly reuse the old pin -- a page-context switch is a real,
    deliberate signal (the user navigated to a different course module)
    that overrides "the message looked vague"."""
    with patch("app.routers.chat.history.get_pinned", return_value=PINNED_INDUSTRIAL_FR), \
         patch("app.routers.chat._retrieve_context",
               return_value=("blockchain content", ["3.1_bill_42_25_draft.md"])):
        context, sources, segment_id, is_new_pin, domain, domain_source = _resolve_turn_context(
            "sess-1", "Pourquoi ?", "blockchain", "fr", "t1"
        )
    assert domain == "blockchain"
    assert domain_source == "page_context"
    assert context == "blockchain content"
    assert segment_id == PINNED_INDUSTRIAL_FR["segment_id"] + 1
    assert is_new_pin is True
