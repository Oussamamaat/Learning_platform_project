"""
Regression tests for the serving-layer language router.

First tests for app/services/llm.py. Nothing here existed before this run —
the 2026-08-02 hyphenated-imperative bug ("Explique-moi la procedure LOTO."
routing to Darija instead of French) was caught by a live CEO-demo dry run,
not by a test, and nothing protected against it regressing.
"""

from app.services.llm import (
    detect_query_language,
    _build_system_prompt,
    deterministic_refusal,
    SYSTEM_PROMPT_TEMPLATE,
    SYSTEM_PROMPT_TEMPLATE_FR,
    DOMAIN_LABELS,
    DOMAIN_LABELS_FR,
    DOMAIN_LABELS_AR,
    UI_LANG_TO_MODEL_LANG,
)


# --- detect_query_language -------------------------------------------------

def test_arabic_script_routes_to_darija():
    assert detect_query_language("شنو هوما les EPI لي خاصين لـ le soudeur؟") == "darija"


def test_empty_query_defaults_to_darija():
    assert detect_query_language("") == "darija"


def test_arabizi_markers_route_to_darija():
    assert detect_query_language("chno howa dyal securite") == "darija"


def test_french_accent_routes_to_fr():
    assert detect_query_language("Ou se trouve la procedure a suivre?") == "fr"


def test_two_french_markers_without_accent_routes_to_fr():
    # No accented characters here -- must be decided by the >=2 marker-word
    # count, not the accent shortcut.
    assert detect_query_language("Est-ce que vous pouvez m'aider avec les EPI ?") == "fr"


def test_single_french_marker_is_not_enough():
    # Only "le" matches _FRENCH_MARKERS -- below the >=2 threshold, so this
    # falls through to the safe default rather than misrouting on one word.
    assert detect_query_language("Le chat noir dort") == "darija"


def test_hyphenated_imperative_pronoun_routes_to_fr():
    """Regression for the 2026-08-02 live-demo bug.

    "explique-moi" is one whitespace-delimited token, so "moi" (a listed
    French marker) was invisible to the word-set check until the router
    started splitting on hyphens too. Without that split this exact sentence
    only matches "la" (1 marker) and falls through to Darija -- reproduced
    below by checking the pre-fix tokenization would have failed.
    """
    query = "Explique-moi la procedure LOTO."
    assert detect_query_language(query) == "fr"

    # Prove *why* it passes: hyphen-splitting is what pushes it over the
    # >=2 threshold, not some other marker already in the sentence.
    words_without_hyphen_split = {
        w.strip(".,!?;:()\"'").lower() for w in query.split()
    }
    from app.services.llm import _FRENCH_MARKERS
    assert len(words_without_hyphen_split & set(_FRENCH_MARKERS)) < 2, (
        "test no longer isolates the hyphen-split fix -- sentence now has "
        "a second marker outside the hyphenated token"
    )


def test_dis_moi_needs_hyphen_split_to_reach_threshold():
    """Minimal case: exactly one marker ("pourquoi") outside the hyphenated
    token, and "moi" only clears the >=2 bar once split out of "dis-moi"."""
    assert detect_query_language("Dis-moi pourquoi.") == "fr"


def test_arabizi_wins_over_french_markers_when_both_present():
    # detect_query_language's own contract: Arabizi and French want opposite
    # answers, so when a query could plausibly be read either way, Arabizi
    # markers must win over French ones (checked first in the function).
    assert detect_query_language("chno est le probleme") == "darija"


def test_arabic_character_majority_wins_even_with_latin_words():
    query = "شنو hia la ISO 45001؟"
    assert detect_query_language(query) == "darija"


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
