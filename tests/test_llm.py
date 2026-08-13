"""
Regression tests for the serving-layer language router.

First tests for app/services/llm.py. Nothing here existed before this run —
the 2026-08-02 hyphenated-imperative bug ("Explique-moi la procedure LOTO."
routing to Darija instead of French) was caught by a live CEO-demo dry run,
not by a test, and nothing protected against it regressing.
"""

from app.services.llm import (
    detect_query_language,
    detect_language_instruction,
    _build_system_prompt,
    condense_retrieval_query,
    deterministic_refusal,
    is_anaphoric_followup,
    SYSTEM_PROMPT_TEMPLATE,
    SYSTEM_PROMPT_TEMPLATE_FR,
    DOMAIN_LABELS,
    DOMAIN_LABELS_FR,
    DOMAIN_LABELS_AR,
    UI_LANG_TO_MODEL_LANG,
)


# --- detect_query_language -------------------------------------------------
#
# Simplified 2026-08-11: Arabizi is out of scope, so with it gone Latin
# script is unambiguously French and this collapsed from a five-branch
# French-vs-Arabizi heuristic to a two-branch script check. The direct
# consequence: undecidable/empty input now defaults to 'fr' (Latin default),
# not 'darija' -- the old tests asserting the opposite default are updated
# below, not preserved, per that decision.

def test_arabic_script_routes_to_darija():
    assert detect_query_language("شنو هوما les EPI لي خاصين لـ le soudeur؟") == "darija"


def test_empty_query_defaults_to_fr():
    assert detect_query_language("") == "fr"


def test_arabizi_markers_route_to_darija():
    assert detect_query_language("chno howa dyal securite") == "darija"


def test_plain_latin_query_routes_to_fr():
    assert detect_query_language("Ou se trouve la procedure a suivre?") == "fr"


def test_single_word_latin_query_still_routes_to_fr():
    # No Arabizi marker, no Arabic script -- script check alone is enough,
    # no minimum French-marker count needed anymore.
    assert detect_query_language("Le chat noir dort") == "fr"


def test_hyphenated_arabizi_form_still_tokenizes_for_marker_check():
    """Hyphen-splitting still matters for an Arabizi marker fused into a
    single whitespace-delimited token via a hyphen -- the same tokenization
    bug class that caused the 2026-08-02 live-demo misroute, now on the
    Arabizi side since that's the only branch left that can override the
    Latin default."""
    query = "3andi-chi mochkil f had ta9rir"
    assert detect_query_language(query) == "darija"

    # Prove *why*: without splitting on the hyphen, "3andi-chi" as a whole
    # token doesn't match any _ARABIZI_MARKERS entry ("3andi" does, alone).
    from app.services.llm import _ARABIZI_MARKERS
    whole_tokens = {w.strip(".,!?;:()\"'").lower() for w in query.split()}
    assert not (whole_tokens & set(_ARABIZI_MARKERS)), (
        "test no longer isolates the hyphen-split fix -- a marker matches "
        "without splitting on hyphens"
    )


def test_arabizi_wins_over_plain_latin_text():
    # An Arabizi marker in an otherwise plain-looking Latin sentence still
    # routes to darija -- it's the only thing that can override the Latin
    # default now that French markers no longer participate in the decision.
    assert detect_query_language("chno est le probleme") == "darija"


def test_arabic_character_majority_wins_even_with_latin_words():
    # Enough Arabic-script characters to outnumber the Latin ones despite a
    # couple of Latin words/acronym mixed in -- the arabic>latin count check
    # must win before any word-level check runs.
    query = "شنو كاين فهاد الموضوع بالضبط hia ISO؟"
    assert detect_query_language(query) == "darija"


# --- detect_language_instruction --------------------------------------------

def test_instruction_darija_after_response_verb():
    assert detect_language_instruction("Comment on fait cela ? Reponds en darija.") == "darija"


def test_instruction_fr_after_response_verb():
    assert detect_language_instruction("chno hiya l'procedure? repond en francais") == "fr"


def test_instruction_as_trailing_clause_after_comma():
    assert detect_language_instruction("Comment on fait cela, en darija ?") == "darija"


def test_content_question_about_arabic_is_not_an_instruction():
    """The precision guard: a language phrase mid-sentence, with no response
    verb before it and not trailing after a clause boundary, is content --
    NOT an instruction. This is the exact case the guard exists for."""
    assert detect_language_instruction("Quels documents sont disponibles en arabe ?") is None


def test_no_instruction_returns_none():
    assert detect_language_instruction("Quelles sont les sanctions prevues par la loi 27.06 ?") is None


def test_empty_message_has_no_instruction():
    assert detect_language_instruction("") is None


def test_arabic_script_instruction_with_arabic_verb():
    assert detect_language_instruction("شنو كاين؟ جاوبني بالفرنسية") == "fr"


# --- _build_system_prompt ---------------------------------------------------

def test_build_system_prompt_darija_default_uses_darija_template():
    prompt = _build_system_prompt("industrial", "CONTEXT TEXT", language="darija")
    assert "Moroccan Darija" in prompt
    assert DOMAIN_LABELS["industrial"] in prompt
    assert "CONTEXT TEXT" in prompt
    assert "Tu es un tuteur" not in prompt  # not the French template


def test_build_system_prompt_fr_uses_french_template():
    prompt = _build_system_prompt("securite", "TEXTE DE CONTEXTE", language="fr")
    assert "Reponds en francais" in prompt
    assert DOMAIN_LABELS_FR["securite"] in prompt
    assert "TEXTE DE CONTEXTE" in prompt
    assert "Moroccan Darija" not in prompt  # not the Darija template


def test_build_system_prompt_unknown_domain_falls_back_to_raw_label():
    prompt = _build_system_prompt("medical", "ctx", language="darija")
    assert "medical" in prompt


def test_build_system_prompt_fr_unknown_domain_falls_back_to_raw_label():
    prompt = _build_system_prompt("medical", "ctx", language="fr")
    assert "medical" in prompt


def test_build_system_prompt_matches_template_format():
    assert _build_system_prompt("industrial", "X", "darija") == \
        SYSTEM_PROMPT_TEMPLATE.format(domain=DOMAIN_LABELS["industrial"], context="X")
    assert _build_system_prompt("industrial", "X", "fr") == \
        SYSTEM_PROMPT_TEMPLATE_FR.format(domain=DOMAIN_LABELS_FR["industrial"], context="X")


# --- deterministic_refusal / UI_LANG_TO_MODEL_LANG --------------------------
#
# Regression coverage for the live-reproduced defect: off-topic refusals
# self-identified as a "safety" assistant regardless of the tenant's actual
# domain (3/3 on securite/blockchain questions). deterministic_refusal
# bypasses the model entirely for this case -- see app/routers/chat.py.

def test_ui_lang_map_covers_french_and_darija_only():
    assert UI_LANG_TO_MODEL_LANG == {"fr": "fr", "ar-MA": "darija"}
    # "en" (Language.ENGLISH) is deliberately unmapped -- the model was never
    # trained on English, so an unmapped value must fall through to the
    # heuristic default rather than silently serving English.
    assert "en" not in UI_LANG_TO_MODEL_LANG


def test_deterministic_refusal_darija_names_correct_domain():
    for domain in ("industrial", "securite", "blockchain"):
        text = deterministic_refusal(domain, "darija")
        assert DOMAIN_LABELS_AR[domain] in text
        for other in ("industrial", "securite", "blockchain"):
            if other != domain:
                assert DOMAIN_LABELS_AR[other] not in text


def test_deterministic_refusal_french_names_correct_domain():
    for domain in ("industrial", "securite", "blockchain"):
        text = deterministic_refusal(domain, "fr")
        assert DOMAIN_LABELS_FR[domain] in text
        for other in ("industrial", "securite", "blockchain"):
            if other != domain:
                assert DOMAIN_LABELS_FR[other] not in text


def test_deterministic_refusal_defaults_to_darija():
    assert deterministic_refusal("industrial") == deterministic_refusal("industrial", "darija")


def test_deterministic_refusal_unknown_domain_falls_back_to_raw_label():
    assert "medical" in deterministic_refusal("medical", "fr")
    assert "medical" in deterministic_refusal("medical", "darija")


def test_deterministic_refusal_not_derived_from_prompt_templates():
    """Refusal strings must stay outside SYSTEM_PROMPT_TEMPLATE(_FR) -- those
    are under the byte-identical train/serve parity invariant
    (test_generation_gates.py) and must never be extended with serving-only
    text."""
    refusal = deterministic_refusal("industrial", "fr")
    assert refusal not in SYSTEM_PROMPT_TEMPLATE_FR
    assert refusal not in SYSTEM_PROMPT_TEMPLATE


# --- is_anaphoric_followup / condense_retrieval_query -----------------------
# Stage 3: the guard against a segment-reset firing on a bare fragment
# (which would surface as a false deterministic_refusal mid-conversation).

def test_short_message_is_anaphoric_regardless_of_content():
    assert is_anaphoric_followup("d'accord") is True
    assert is_anaphoric_followup("Et donc ?") is True


def test_french_anaphora_marker_detected():
    assert is_anaphoric_followup("Et pourquoi cette regle existe-t-elle vraiment ?") is True


def test_darija_anaphora_marker_detected():
    assert is_anaphoric_followup("علاش خاصها تكون هكداك بالضبط فهاد الحالة؟") is True


def test_self_contained_question_is_not_anaphoric():
    assert is_anaphoric_followup("Quelles sont les sanctions prevues par la loi 27.06 ?") is False


def test_empty_message_is_not_anaphoric():
    assert is_anaphoric_followup("") is False
    assert is_anaphoric_followup("   ") is False


def test_hyphenated_form_still_tokenizes_for_anaphora_check():
    # Same hyphen-split fix as detect_query_language's "explique-moi" case --
    # a short hyphenated message must still fall under the token-count check.
    assert is_anaphoric_followup("dis-moi") is True


def test_condense_appends_prior_turn_when_anaphoric():
    result = condense_retrieval_query("Pourquoi ?", "Quels EPI sont obligatoires pour la tete ?")
    assert result == "Quels EPI sont obligatoires pour la tete ? Pourquoi ?"


def test_condense_leaves_self_contained_query_untouched():
    query = "Quelles sont les sanctions prevues par la loi 27.06 ?"
    assert condense_retrieval_query(query, "Une question precedente sans rapport ici") == query


def test_condense_with_no_prior_turn_returns_message_unchanged():
    assert condense_retrieval_query("Pourquoi ?", None) == "Pourquoi ?"
    assert condense_retrieval_query("Pourquoi ?", "") == "Pourquoi ?"
