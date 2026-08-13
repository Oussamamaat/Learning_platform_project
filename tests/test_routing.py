"""
Tests for app/services/routing.py's resolve_language(): the query_lang vs.
response_lang split and the stickiness rules that let "réponds en darija"
persist across follow-up turns without repeating the instruction (2026-08-11
plan, "Automatic Domain Routing + Language/Script Detection", section 1).
"""
from app.services.routing import resolve_language


# --- precedence: explicit_language wins outright ----------------------------

def test_explicit_language_forces_both_query_and_response_lang():
    r = resolve_language("N'importe quoi ici", explicit_language="darija")
    assert r.query_lang == "darija"
    assert r.response_lang == "darija"
    assert r.response_lang_source == "explicit_field"
    assert r.override_to_persist is None


def test_explicit_language_beats_in_message_instruction():
    r = resolve_language("Reponds en francais", explicit_language="darija")
    assert r.response_lang == "darija"
    assert r.response_lang_source == "explicit_field"


# --- in-message instruction: sets response_lang, leaves query_lang alone ---

def test_instruction_diverges_query_and_response_lang():
    r = resolve_language("Comment on fait cela ? Reponds en darija.")
    assert r.query_lang == "fr"          # the message itself is French
    assert r.response_lang == "darija"   # but the answer was asked for in Darija
    assert r.response_lang_source == "explicit_instruction"


def test_instruction_persists_override_and_its_query_lang_baseline():
    r = resolve_language("Comment on fait cela ? Reponds en darija.")
    assert r.override_to_persist == "darija"
    assert r.override_query_lang_to_persist == "fr"


# --- stickiness: case 2, same script as when the override was set ----------

def test_sticky_override_kept_when_script_unchanged():
    r = resolve_language(
        "Et pourquoi cette regle existe-t-elle ?",   # still French
        stored_override="darija",
        stored_override_query_lang="fr",
    )
    assert r.query_lang == "fr"
    assert r.response_lang == "darija"
    assert r.response_lang_source == "sticky"
    # Re-persists the same pair so the session doesn't need special-casing.
    assert r.override_to_persist == "darija"
    assert r.override_query_lang_to_persist == "fr"


# --- stickiness: case 3, script changed -> override clears -----------------

def test_override_clears_when_script_changes():
    r = resolve_language(
        "شنو كاين فهاد الموضوع؟",   # now Arabic script
        stored_override="darija",
        stored_override_query_lang="fr",
    )
    assert r.query_lang == "darija"
    assert r.response_lang == "darija"       # follows query_lang directly now
    assert r.response_lang_source == "script"
    assert r.override_to_persist is None
    assert r.override_query_lang_to_persist is None


def test_no_stored_override_and_no_instruction_follows_script():
    r = resolve_language("Quelles sont les sanctions prevues par la loi 27.06 ?")
    assert r.query_lang == "fr"
    assert r.response_lang == "fr"
    assert r.response_lang_source == "script"
    assert r.override_to_persist is None


def test_a_new_instruction_overrides_an_active_sticky_session():
    r = resolve_language(
        "En fait reponds en francais maintenant",
        stored_override="darija",
        stored_override_query_lang="fr",
    )
    assert r.response_lang == "fr"
    assert r.response_lang_source == "explicit_instruction"
    assert r.override_to_persist == "fr"
